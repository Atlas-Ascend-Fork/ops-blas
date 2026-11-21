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
#include <cstdint>

#include "cann_ops_blas.h"

static inline aclblasStatus_t CdgmmValidateParams(
    aclblasHandle_t handle, aclblasSideMode_t mode,
    int m, int n, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (mode != ACLBLAS_SIDE_LEFT && mode != ACLBLAS_SIDE_RIGHT) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (mode == ACLBLAS_SIDE_RIGHT) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (m < 0 || n < 0 || incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n) || ldc < std::max(1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && (x == nullptr || A == nullptr || C == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == C && lda != ldc) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Row-major complex multiply
inline aclblasComplex ComplexMul(aclblasComplex a, aclblasComplex b)
{
    return {a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real};
}

// Row-major LEFT golden: C[i,j] = x[i] * A[i,j]
// A and C are row-major with row strides lda and ldc (in complex elements).
// x has logical length m with stride incx.
// Data is stored as aclblasComplex (interleaved float pairs).
inline aclblasStatus_t aclblasCdgmm_cpu(
    aclblasHandle_t handle,
    aclblasSideMode_t mode,
    int m, int n,
    const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx,
    aclblasComplex* C, int ldc)
{
    aclblasStatus_t st = CdgmmValidateParams(handle, mode, m, n, A, lda, x, incx, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const int xLen = m;
    const int64_t absIncx = (incx >= 0) ? static_cast<int64_t>(incx)
                                        : -static_cast<int64_t>(incx);

    for (int i = 0; i < m; i++) {
        int64_t xIdx = (incx >= 0)
            ? static_cast<int64_t>(i) * incx
            : static_cast<int64_t>(xLen - 1 - i) * absIncx;
        aclblasComplex xVal = x[xIdx];

        for (int j = 0; j < n; j++) {
            C[static_cast<int64_t>(i) * ldc + j] =
                ComplexMul(A[static_cast<int64_t>(i) * lda + j], xVal);
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
