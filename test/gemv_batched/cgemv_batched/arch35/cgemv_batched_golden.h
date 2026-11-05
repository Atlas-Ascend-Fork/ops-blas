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
 * \file cgemv_batched_golden.h
 * \brief CPU golden reference for complex batched GEMV.
 *        Implements y[b] = alpha * op(A[b]) * x[b] + beta * y[b] with
 *        column-major storage and negative-stride support, matching Netlib
 *        cgemv semantics. Complex values are interleaved float32 pairs.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t validateCgemvBatchedCpuParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha, int lda, int incx,
    const aclblasComplex* beta, int incy, int batchCount)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0 || lda < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (batchCount < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

namespace cgemv_batched_golden {

// Map an element index to a buffer offset, honouring the sign of the stride.
inline int StridedIndex(int idx, int count, int inc) { return (inc > 0) ? (idx * inc) : ((count - 1 - idx) * (-inc)); }

// Complex multiply helper: returns c = a * b.
inline aclblasComplex ComplexMul(const aclblasComplex& a, const aclblasComplex& b)
{
    return aclblasComplex{a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

// Accumulate helper: c += a * b.
inline void ComplexMulAdd(aclblasComplex& c, const aclblasComplex& a, const aclblasComplex& b)
{
    c.real += a.real * b.real - a.imag * b.imag;
    c.imag += a.real * b.imag + a.imag * b.real;
}

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0f && v.imag == 0.0f; }

// Fetch A(outIdx, k) for the requested transpose mode, applying the
// conjugation that trans=C requires.
inline aclblasComplex LoadAElement(
    const aclblasComplex* aB, int outIdx, int k, int lda, aclblasOperation_t trans, bool isTransN)
{
    aclblasComplex av;
    if (isTransN) {
        av = aB[outIdx + k * lda];
    } else {
        av = aB[k + outIdx * lda];
        if (trans == ACLBLAS_OP_C) {
            av.imag = -av.imag;
        }
    }
    return av;
}

// y = beta * y for the whole batch output (alpha == 0 fast path).
inline void ScaleYByBeta(aclblasComplex* yB, int yCount, int incy, const aclblasComplex& beta)
{
    for (int i = 0; i < yCount; ++i) {
        aclblasComplex& yv = yB[StridedIndex(i, yCount, incy)];
        yv = ComplexMul(beta, yv);
    }
}

// Full GEMV for one batch: y = alpha * op(A) * x + beta * y.
inline void ComputeBatchGmv(
    const aclblasComplex* aB, const aclblasComplex* xB, aclblasComplex* yB, int yCount, int xCount, int lda, int incx,
    int incy, const aclblasComplex& alpha, const aclblasComplex& beta, aclblasOperation_t trans, bool isTransN)
{
    for (int outIdx = 0; outIdx < yCount; ++outIdx) {
        aclblasComplex sum{0.0f, 0.0f};
        for (int k = 0; k < xCount; ++k) {
            aclblasComplex av = LoadAElement(aB, outIdx, k, lda, trans, isTransN);
            ComplexMulAdd(sum, av, xB[StridedIndex(k, xCount, incx)]);
        }
        aclblasComplex out = ComplexMul(alpha, sum);
        int yIdx = StridedIndex(outIdx, yCount, incy);
        if (!IsComplexZero(beta)) {
            ComplexMulAdd(out, beta, yB[yIdx]);
        }
        yB[yIdx] = out;
    }
}

} // namespace cgemv_batched_golden

// CPU golden: A/x/y are contiguous host buffers with per-batch strides.
// a[b] starts at a + b*lda*n, x[b] at x + b*xStride, y[b] at y + b*yStride.
inline aclblasStatus_t aclblasCgemvBatched_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy, int batchCount)
{
    using namespace cgemv_batched_golden;

    aclblasStatus_t validRet =
        validateCgemvBatchedCpuParams(handle, trans, m, n, alpha, lda, incx, beta, incy, batchCount);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return validRet;
    if (m == 0 || n == 0 || batchCount == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const bool isTransN = (trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? n : m;
    const int yCount = isTransN ? m : n;
    const size_t xStride = static_cast<size_t>((xCount - 1) * std::abs(incx) + 1);
    const size_t yStride = static_cast<size_t>((yCount - 1) * std::abs(incy) + 1);
    const size_t aStride = static_cast<size_t>(lda) * n;

    const bool alphaIsZero = IsComplexZero(*alpha);
    const bool betaIsOne = (beta->real == 1.0f && beta->imag == 0.0f);

    for (int b = 0; b < batchCount; ++b) {
        const aclblasComplex* aB = a + b * aStride;
        const aclblasComplex* xB = x + b * xStride;
        aclblasComplex* yB = y + b * yStride;

        if (alphaIsZero && betaIsOne)
            continue; // quick return
        if (alphaIsZero) {
            ScaleYByBeta(yB, yCount, incy, *beta);
            continue;
        }
        ComputeBatchGmv(aB, xB, yB, yCount, xCount, lda, incx, incy, *alpha, *beta, trans, isTransN);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
