/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "ctbsv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

void ctbsv_kernel_do(const CtbsvTilingData &tiling, void *stream);

static constexpr uint32_t CTBSV_SIMT_THRESHOLD = SIMT_MIN_THREAD_NUM;
static constexpr uint32_t CTBSV_SIMT_MIN_BAND_TRANS = 16;
static constexpr uint32_t CTBSV_SIMT_MIN_BAND_NOTRANS = 24;
// Trans/C is sequential across columns. Narrow bands use a single warp
// (32 threads) so the per-column reduce skips inter-warp syncthreads.
// Wider bands go to the AIV vector gather (numThreads=0).
static constexpr uint32_t CTBSV_SIMT_TRANS_THREADS = 32;

static aclblasStatus_t ValidateCtbsvParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, int lda, int incx, const aclblasComplex* A, const aclblasComplex* x)
{
    CHECK_RET(
        handle != nullptr, OP_LOGE("aclblasCtbsv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasCtbsv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasCtbsv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
        OP_LOGE("aclblasCtbsv", "invalid diag=%d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_ENUM);

    CHECK_RET(n >= 0, OP_LOGE("aclblasCtbsv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasCtbsv", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    // Banded A is (k+1) by n: lda >= k+1. Do not compare lda with n.
    CHECK_RET(
        lda > k, OP_LOGE("aclblasCtbsv", "invalid lda=%d, k=%d", lda, k);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasCtbsv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        incx != INT_MIN, OP_LOGE("aclblasCtbsv", "incx must not be INT_MIN");
        return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(A != nullptr, OP_LOGE("aclblasCtbsv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasCtbsv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static uint32_t ChooseCtbsvNumThreads(aclblasOperation_t trans, uint32_t nU32, uint32_t kU32)
{
    const uint32_t maxBand = std::min(kU32, nU32 > 0 ? nU32 - 1 : 0U) + 1U;
    const bool isTrans = (trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C);
    const uint32_t minBand = isTrans ? CTBSV_SIMT_MIN_BAND_TRANS : CTBSV_SIMT_MIN_BAND_NOTRANS;
    if (nU32 < CTBSV_SIMT_THRESHOLD || maxBand < minBand) {
        return 0;
    }
    if (isTrans) {
        // Wide T/C is sequential across n columns; extra warps pay two
        // syncthreads per column and lose the official 0.4 bar. Keep a
        // single warp for narrow bands, and send wide bands to the AIV
        // vector gather (numThreads=0).
        if (maxBand <= 48U) {
            return CTBSV_SIMT_TRANS_THREADS;
        }
        return 0;
    }
    const uint32_t threadsForBand =
        CeilAlign<uint32_t>(std::max(maxBand, SIMT_MIN_THREAD_NUM), SIMT_MIN_THREAD_NUM);
    return std::min({nU32, SIMT_MAX_THREAD_NUM, threadsForBand});
}

static void FillCtbsvTiling(
    const aclblasComplex* A, aclblasComplex* x, int n, int k, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int lda, int incx, uint32_t numThreads,
    CtbsvTilingData& tiling)
{
    tiling.a = reinterpret_cast<uint64_t>(A);
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = incx;
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.numThreads = numThreads;
}

aclblasStatus_t aclblasCtbsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
{
    const aclblasStatus_t st = ValidateCtbsvParams(handle, uplo, trans, diag, n, k, lda, incx, A, x);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (n == 0 || (k == 0 && diag == ACLBLAS_UNIT)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    CtbsvTilingData tiling{};
    FillCtbsvTiling(
        A, x, n, k, uplo, trans, diag, lda, incx,
        ChooseCtbsvNumThreads(trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k)), tiling);
    OP_LOGD(
        "aclblasCtbsv", "tiling: n=%u k=%u uplo=%u trans=%u diag=%u incx=%d lda=%u numThreads=%u",
        tiling.n, tiling.k, tiling.uplo, tiling.trans, tiling.diag, tiling.incx, tiling.lda, tiling.numThreads);
    OP_LOGI("aclblasCtbsv", "launching kernel");
    ctbsv_kernel_do(tiling, handle->stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE("aclblasCtbsv", "kernel launch failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
