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

#include <cstdint>

#include "cann_ops_blas.h"

inline aclblasComplex CsyrMul(const aclblasComplex& lhs, const aclblasComplex& rhs)
{
    return {lhs.real * rhs.real - lhs.imag * rhs.imag, lhs.real * rhs.imag + lhs.imag * rhs.real};
}

inline void CsyrUpdateGolden(
    aclblasFillMode_t uplo, int n, const aclblasComplex& alpha, const aclblasComplex* x, int incx, aclblasComplex* A,
    int lda)
{
    const int64_t xStart = (incx >= 0) ? 0 : static_cast<int64_t>(1 - n) * incx;
    for (int row = 0; row < n; ++row) {
        const aclblasComplex axRow = CsyrMul(alpha, x[xStart + static_cast<int64_t>(row) * incx]);
        const int colStart = (uplo == ACLBLAS_UPPER) ? row : 0;
        const int colEnd = (uplo == ACLBLAS_UPPER) ? n : row + 1;
        for (int col = colStart; col < colEnd; ++col) {
            const aclblasComplex update = CsyrMul(axRow, x[xStart + static_cast<int64_t>(col) * incx]);
            aclblasComplex& out = A[static_cast<int64_t>(col) * lda + row];
            out.real += update.real;
            out.imag += update.imag;
        }
    }
}

inline aclblasStatus_t aclblasCsyr_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, aclblasComplex* A, int lda)
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
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < n || incx == 0 || alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha->real == 0.0f && alpha->imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || A == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CsyrUpdateGolden(uplo, n, *alpha, x, incx, A, lda);
    return ACLBLAS_STATUS_SUCCESS;
}
