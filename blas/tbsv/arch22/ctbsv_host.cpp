/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "ctbsv_kernel.h"
#include "ctbsv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static constexpr uint32_t CTBSV_UB_MAX_N = 4096U;
static constexpr uint32_t CTBSV_UB_MAX_K = 2048U;

static aclblasStatus_t ValidateCtbsvParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    int lda, int incx, const aclblasComplex* A, const aclblasComplex* x)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCtbsv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasCtbsv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasCtbsv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
              OP_LOGE("aclblasCtbsv", "invalid diag=%d", static_cast<int>(diag));
              return ACLBLAS_STATUS_INVALID_VALUE);

    CHECK_RET(n >= 0, OP_LOGE("aclblasCtbsv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasCtbsv", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    // Banded A is (k+1) by n: lda >= k+1. Do not compare lda with n.
    CHECK_RET(lda > k, OP_LOGE("aclblasCtbsv", "invalid lda=%d, k=%d", lda, k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasCtbsv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE("aclblasCtbsv", "incx must not be INT_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(A != nullptr, OP_LOGE("aclblasCtbsv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasCtbsv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static void FillCtbsvTiling(
    const aclblasComplex* A, aclblasComplex* x, int n, int k, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int lda, int incx, CtbsvTilingData& tiling)
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
    tiling.useGmFallback =
        (static_cast<uint32_t>(n) > CTBSV_UB_MAX_N || static_cast<uint32_t>(k) > CTBSV_UB_MAX_K) ? 1U : 0U;
}

aclblasStatus_t aclblasCtbsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
{
    const aclblasStatus_t st = ValidateCtbsvParams(handle, uplo, trans, diag, n, k, lda, incx, A, x);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (n == 0 || (k == 0 && diag == ACLBLAS_UNIT)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    CtbsvTilingData tiling;
    FillCtbsvTiling(A, x, n, k, uplo, trans, diag, lda, incx, tiling);
    OP_LOGD(
        "aclblasCtbsv", "tiling: n=%u k=%u uplo=%u trans=%u diag=%u incx=%d lda=%u gmFallback=%u", tiling.n, tiling.k,
        tiling.uplo, tiling.trans, tiling.diag, tiling.incx, tiling.lda, tiling.useGmFallback);
    OP_LOGI("aclblasCtbsv", "launching kernel");
    ctbsv_kernel_do(tiling, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
