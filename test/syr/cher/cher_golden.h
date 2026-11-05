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
#include <climits>
#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t ValidateCherParamsCpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* A, int lda)
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
    if (lda < std::max(1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incx == INT_MIN) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (*alpha == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || A == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Explicit Hermitian rank-1 golden with Netlib-style negative-incx physical indexing.
// Avoids depending on external cblas_cher quick-returning on incx < 0.
inline void ComputeGoldenCher(
    aclblasFillMode_t uplo, int n, float alpha, const aclblasComplex* x, int incx, aclblasComplex* A, int lda)
{
    const int absIncx = std::abs(incx);
    for (int j = 0; j < n; ++j) {
        const int physJ = (incx < 0) ? (n - 1 - j) * absIncx : j * absIncx;
        const aclblasComplex xj = x[physJ];
        const int iBegin = (uplo == ACLBLAS_UPPER) ? 0 : j;
        const int iEnd = (uplo == ACLBLAS_UPPER) ? j : (n - 1);
        for (int i = iBegin; i <= iEnd; ++i) {
            const int physI = (incx < 0) ? (n - 1 - i) * absIncx : i * absIncx;
            const aclblasComplex xi = x[physI];
            aclblasComplex& aij = A[static_cast<size_t>(j) * static_cast<size_t>(lda) + static_cast<size_t>(i)];
            if (i == j) {
                aij.real += alpha * (xi.real * xi.real + xi.imag * xi.imag);
                aij.imag = 0.0f;
            } else {
                // A_ij += alpha * x_i * conj(x_j)
                aij.real += alpha * (xi.real * xj.real + xi.imag * xj.imag);
                aij.imag += alpha * (xi.imag * xj.real - xi.real * xj.imag);
            }
        }
    }
}

inline aclblasStatus_t aclblasCher_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* A, int lda)
{
    aclblasStatus_t st = ValidateCherParamsCpu(handle, uplo, n, alpha, x, incx, A, lda);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (n == 0 || *alpha == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    ComputeGoldenCher(uplo, n, *alpha, x, incx, A, lda);
    return ACLBLAS_STATUS_SUCCESS;
}
