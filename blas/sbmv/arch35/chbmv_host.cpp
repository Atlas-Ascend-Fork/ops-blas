/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chbmv_host.cpp
 * \brief Host implementation of aclblasChbmv for Ascend 950 (arch35).
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "chbmv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

namespace {

static constexpr uint32_t UB_X_COMPLEX_LIMIT = 8192U;

aclblasStatus_t ValidateChbmvParams(
    aclblasFillMode_t uplo, int n, int k, int lda, int incx, int incy, const aclblasComplex* alpha,
    const aclblasComplex* beta, const aclblasComplex* A, const aclblasComplex* x, const aclblasComplex* y)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasChbmv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasChbmv", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        static_cast<int64_t>(lda) >= static_cast<int64_t>(k) + 1,
        OP_LOGE("aclblasChbmv", "invalid lda=%d, k=%d", lda, k);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasChbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasChbmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr && beta != nullptr,
        OP_LOGE("aclblasChbmv", "alpha or beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasChbmv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasChbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasChbmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    const uint64_t maxElements = std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
    CHECK_RET(
        static_cast<uint64_t>(lda) * static_cast<uint64_t>(n) <= maxElements,
        OP_LOGE("aclblasChbmv", "lda * n overflows");
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

ChbmvTilingData CalChbmvTilingData(
    uint32_t useNumBlocks, int n, int k, int lda, aclblasFillMode_t uplo, const aclblasComplex& alpha,
    const aclblasComplex& beta, int incx, int incy)
{
    ChbmvTilingData tilingData{};
    // A zero block count would divide by zero below; the caller always passes at least one block,
    // and the helper keeps returning an empty tiling instead of crashing when it does not.
    if (useNumBlocks == 0U) {
        return tilingData;
    }
    uint32_t rowsPerBlock =
        static_cast<uint32_t>((static_cast<uint64_t>(n) + useNumBlocks - 1ULL) / useNumBlocks);
    uint64_t band = static_cast<uint64_t>(std::min(k, n - 1));
    uint64_t maxBlockXLen = std::min<uint64_t>(
        static_cast<uint64_t>(n), static_cast<uint64_t>(rowsPerBlock) + 2ULL * band);
    bool alphaZero = (alpha.real == 0.0f && alpha.imag == 0.0f);
    bool canUseUb = (incx == 1 && n >= 32 && !alphaZero && maxBlockXLen <= UB_X_COMPLEX_LIMIT);

    uint32_t targetThreads = rowsPerBlock;
    if (canUseUb && k >= 8) {
        uint32_t tpr = (k >= 256) ? 16U : (k >= 64 ? 8U : 4U);
        targetThreads = rowsPerBlock * tpr;
    }
    tilingData.numThreads =
        std::min(CeilAlign<uint32_t>(targetThreads, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.numBlocks = useNumBlocks;
    tilingData.rowsPerBlock = rowsPerBlock;
    tilingData.useUb = canUseUb ? 1U : 0U;
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.k = static_cast<uint32_t>(k);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alphaReal = alpha.real;
    tilingData.alphaImag = alpha.imag;
    tilingData.betaReal = beta.real;
    tilingData.betaImag = beta.imag;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);
    return tilingData;
}

} // namespace

aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    // The handle is checked before n so a null handle is always reported as
    // ACLBLAS_STATUS_HANDLE_IS_NULLPTR, matching the complex siblings cher2 / chpr2 and letting
    // callers use one error-code contract across the family.
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasChbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasChbmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateChbmvParams(uplo, n, k, lda, incx, incy, alpha, beta, A, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (alpha->real == 0.0f && alpha->imag == 0.0f && beta->real == 1.0f && beta->imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclrtStream useStream = handle->stream;
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasChbmv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t minRowsPerBlock = (incx == 1 && k >= 8) ? 8U : 32U;
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(n), minRowsPerBlock), aivCoreNum);

    ChbmvTilingData tiling = CalChbmvTilingData(useNumBlocks, n, k, lda, uplo, *alpha, *beta, incx, incy);

    chbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(y), nullptr, useNumBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
