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
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "chbmv_kernel.h"

namespace {
bool ValidStride(int n, int inc)
{
    if (inc == 0 || inc == INT_MIN) {
        return false;
    }
    const int64_t magnitude = inc > 0 ? static_cast<int64_t>(inc) : -static_cast<int64_t>(inc);
    return static_cast<int64_t>(n - 1) * magnitude <= INT_MAX;
}

aclblasStatus_t ValidateChbmv(
    aclblasFillMode_t uplo, int n, int k, const aclblasComplex* alpha, const aclblasComplex* a, int lda,
    const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    // The task's INVALID_VALUE contract also applies to invalid uplo (not INVALID_ENUM).
    if ((uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) || k < 0 ||
        static_cast<int64_t>(lda) < static_cast<int64_t>(k) + 1 || !ValidStride(n, incx) || !ValidStride(n, incy) ||
        alpha == nullptr || beta == nullptr || a == nullptr || x == nullptr || y == nullptr) {
        OP_LOGE("aclblasChbmv", "invalid dimensions, stride, uplo or pointer");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool UsesBalancedWideOwnership(const ChbmvTilingData& t)
{
    return t.path != CHBMV_PATH_DIAGONALS && t.incx == 1 && t.incy == 1 && t.effectiveK >= 64U && t.lda > 64U &&
           static_cast<uint64_t>(t.lda - 1U) * 8U <= UINT32_MAX && std::isfinite(t.alphaR) && std::isfinite(t.alphaI) &&
           std::isfinite(t.betaR) && std::isfinite(t.betaI);
}

ChbmvTilingData MakeChbmvTiling(
    aclblasFillMode_t uplo, int n, int k, int lda, int incx, int incy, aclblasComplex alpha, aclblasComplex beta,
    uint32_t cores, const aclblasComplex* y)
{
    ChbmvTilingData t{};
    t.n = static_cast<uint32_t>(n);
    t.k = static_cast<uint32_t>(k);
    t.effectiveK = std::min(t.k, t.n - 1U);
    t.lda = static_cast<uint32_t>(lda);
    t.upper = uplo == ACLBLAS_UPPER ? 1U : 0U;
    t.incx = incx;
    t.incy = incy;
    t.alphaR = alpha.real;
    t.alphaI = alpha.imag;
    t.betaR = beta.real;
    t.betaI = beta.imag;
    const uint64_t spanBytes =
        (static_cast<uint64_t>(CHBMV_DIAGONAL_ROWS) + 2ULL * t.effectiveK) * t.lda * sizeof(aclblasComplex);
    if (incx == 1 && incy == 1 && t.effectiveK <= CHBMV_SMALL_BAND && spanBytes <= CHBMV_DIAGONAL_SPAN_BYTES) {
        t.path = CHBMV_PATH_DIAGONALS;
    }
    const uint32_t diagonalBlockRows =
        t.n <= 64U && 2U * t.effectiveK >= t.n ?
            8U :
            (t.n <= CHBMV_DIAGONAL_SMALL_N ?
                 CHBMV_DIAGONAL_SMALL_ROWS :
                 (t.n <= CHBMV_DIAGONAL_MEDIUM_N ? CHBMV_DIAGONAL_MEDIUM_ROWS : CHBMV_DIAGONAL_ROWS));
    const uint32_t rowsPerBlock = t.path == CHBMV_PATH_DIAGONALS ? diagonalBlockRows : CHBMV_ROWS;
    t.blockCount = std::min(cores, (t.n - 1U) / rowsPerBlock + 1U);
    if (UsesBalancedWideOwnership(t)) {
        // Assign whole 32-byte output groups; only the final group may be partial.
        t.blockCount = std::min(cores, (t.n - 1U) / 4U + 1U);
    }
    // Strided or unaligned stores may share a 32-byte GM transaction across row blocks.
    if (incy != 1 || reinterpret_cast<uintptr_t>(y) % 32U != 0U) {
        t.blockCount = 1;
    }
    return t;
}
} // namespace

aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const aclblasComplex* alpha, const aclblasComplex* a,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t status = ValidateChbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    // Host scalars are read only after all required non-null checks, including A/x/y.
    if (alpha->real == 0.0f && alpha->imag == 0.0f && beta->real == 1.0f && beta->imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0U) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const ChbmvTilingData t = MakeChbmvTiling(uplo, n, k, lda, incx, incy, *alpha, *beta, cores, y);
    chbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), t.blockCount, t,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
