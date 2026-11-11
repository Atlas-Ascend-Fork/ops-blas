/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t ChbmvCpuCheckParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, int lda, int incx, int incy,
    const aclblasComplex* alpha, const aclblasComplex* a, const aclblasComplex* x,
    const aclblasComplex* beta, aclblasComplex* y)
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
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k < 0 || lda < k + 1 || incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr || a == nullptr || x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// When beta == (0,0), y old values must not be read (e.g. TC_ED_243 beta0_y_nan)
inline void ChbmvCpuZeroY(int n, int incy, aclblasComplex* y)
{
    const int64_t absIncy = std::abs(incy);
    for (int i = 0; i < n; ++i) {
        int64_t idx = (incy > 0) ? (static_cast<int64_t>(i) * incy) : (static_cast<int64_t>(n - 1 - i) * absIncy);
        y[idx] = {0.0f, 0.0f};
    }
}

// Enforce Hermitian diagonal constraint (imaginary part of diagonal is 0) before calling cblas_chbmv
inline std::vector<aclblasComplex> ChbmvCpuHermitianCopy(
    aclblasFillMode_t uplo, int n, int k, int lda, const aclblasComplex* a)
{
    std::vector<aclblasComplex> aHerm(a, a + static_cast<size_t>(lda) * static_cast<size_t>(n));
    const int64_t diagRow = (uplo == ACLBLAS_UPPER) ? static_cast<int64_t>(k) : 0;
    for (int j = 0; j < n; ++j) {
        aHerm[diagRow + static_cast<int64_t>(j) * lda].imag = 0.0f;
    }
    return aHerm;
}

inline aclblasStatus_t aclblasChbmv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy)
{
    const aclblasStatus_t status =
        ChbmvCpuCheckParams(handle, uplo, n, k, lda, incx, incy, alpha, a, x, beta, y);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (beta->real == 0.0f && beta->imag == 0.0f) {
        ChbmvCpuZeroY(n, incy, y);
    }
    const std::vector<aclblasComplex> aHerm = ChbmvCpuHermitianCopy(uplo, n, k, lda, a);
    cblas_chbmv(CblasColMajor, ToCblasUplo(uplo), n, k, alpha, aHerm.data(), lda, x, incx, beta, y, incy);
    return ACLBLAS_STATUS_SUCCESS;
}
