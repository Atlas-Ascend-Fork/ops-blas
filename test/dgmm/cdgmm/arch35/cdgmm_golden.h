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
 * \file cdgmm_golden.h
 * \brief aclblasCdgmm CPU golden — arch35 (Ascend 950PR / 950DT) contract.
 *
 *        Column-major (BLAS) semantics, aligned with cuBLAS cublasCdgmm:
 *            LEFT (ACLBLAS_SIDE_LEFT) : C(i,j) = x(idx(i)) * A(i,j)   (C = diag(x) * A)
 *            RIGHT(ACLBLAS_SIDE_RIGHT): C(i,j) = A(i,j) * x(idx(j))   (C = A * diag(x))
 *            idx(k) = k * incx              (incx > 0)
 *                   = (L - 1 - k) * |incx|   (incx < 0), L = m (LEFT) / n (RIGHT)
 *
 *        A(i,j) = A[i + j*lda], C(i,j) = C[i + j*ldc]; C padding rows
 *        (i >= m) are left untouched. Complex products are evaluated in
 *        double precision from float32 inputs and narrowed back to float32,
 *        as required by the task's host float64 golden.
 *
 *        Validation order and status codes follow task book §2.5:
 *            handle -> mode -> m<0 -> n<0 -> incx==0 -> lda -> ldc
 *            -> early return (m==0 || n==0) -> null A/x/C -> C==A alias check.
 *        Invalid mode returns ACLBLAS_STATUS_INVALID_VALUE (cuBLAS aligned),
 *        not ACLBLAS_STATUS_INVALID_ENUM.
 *
 *        NOTE: this is NOT the legacy arch22 row-major LEFT-only golden
 *        (test/dgmm/cdgmm/cdgmm_golden.h); the two contracts must not be
 *        mixed in one translation unit.
 */

#pragma once

#include <algorithm>
#include <cstdint>

#include "cann_ops_blas.h"

static inline aclblasStatus_t CdgmmValidateParams(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (mode != ACLBLAS_SIDE_LEFT && mode != ACLBLAS_SIDE_RIGHT) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Leading-dimension checks precede the m==0 / n==0 early return (§2.5).
    if (lda < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || A == nullptr || C == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == C && lda != ldc) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Complex multiply evaluated in float64, narrowed to float32:
//   (ar + i*ai) * (br + i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
static inline aclblasComplex CdgmmComplexMul(const aclblasComplex& a, const aclblasComplex& b)
{
    const double ar = static_cast<double>(a.real);
    const double ai = static_cast<double>(a.imag);
    const double br = static_cast<double>(b.real);
    const double bi = static_cast<double>(b.imag);
    aclblasComplex out;
    out.real = static_cast<float>(ar * br - ai * bi);
    out.imag = static_cast<float>(ar * bi + ai * br);
    return out;
}

// Column-major complex DGMM golden for arch35.
//   A, C: column-major with leading dimensions lda, ldc (complex elements).
//   x   : logical length m (LEFT) or n (RIGHT), signed stride incx.
inline aclblasStatus_t aclblasCdgmm_cpu(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, aclblasComplex* C, int ldc)
{
    aclblasStatus_t st = CdgmmValidateParams(handle, mode, m, n, A, lda, x, incx, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const int xLen = (mode == ACLBLAS_SIDE_LEFT) ? m : n;
    const int64_t absIncx = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const int k = (mode == ACLBLAS_SIDE_LEFT) ? i : j;
            const int64_t xIdx =
                (incx < 0) ? static_cast<int64_t>(xLen - 1 - k) * absIncx : static_cast<int64_t>(k) * incx;
            const int64_t aIdx = static_cast<int64_t>(i) + static_cast<int64_t>(j) * lda;
            const int64_t cIdx = static_cast<int64_t>(i) + static_cast<int64_t>(j) * ldc;
            C[cIdx] = CdgmmComplexMul(A[aIdx], x[xIdx]);
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
