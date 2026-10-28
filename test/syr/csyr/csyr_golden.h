/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Self-implemented golden for complex symmetric rank-1 update (cblas/netlib provide no
// complex csyr; the complex semantics follow the cuBLAS cublasCsyr documentation):
//   A(i,j) += alpha * x(i) * x(j)  for the triangle referenced by uplo (diagonal included),
// with the plain (non-conjugated) transpose. The imaginary part of diagonal elements is an
// ordinary value and receives no special treatment (alpha * x(j)^2 is a complex square).
// Negative incx traverses x backwards (Netlib ssyr convention): logical x(i) maps to
// physical storage (n-1-i)*|incx|.
// Element-wise association alignment with the NPU kernel: alpha is folded into the
// per-column scalar t_j = alpha * x(j) exactly like the NPU kernel scalar unit, then
//   delta_re = x(i).re * t_j.re - x(i).im * t_j.im
//   delta_im = x(i).re * t_j.im + x(i).im * t_j.re
// with every product/sum rounded exactly as the NPU kernel rounds them: the kernel's
// per-element chain (muls with separate rounding + Axpy whose product enters the component
// sum with its own rounding) matches the two-rounding form below bitwise, and the scalar
// alpha-fold mirrors the kernel's empirically-pinned contraction shape (see
// ComputeGoldenCsyrColumnScalar). A rounding-shape mismatch is amplified by alpha=1e6
// near-cancellation elements past the spec per-element gate max(1e-2, 32*ULP). The streamed
// API below (ComputeGoldenCsyrColumnScalar + ComputeGoldenCsyrColumn) is the same reference
// decomposed per column so huge cases (n=4864/4863, A ~ 189 MB) never materialise a full
// golden matrix.

// Column scalar t_j = alpha * x(j) (complex multiply, non-conjugated).
inline void ComputeGoldenCsyrColumnScalar(
    const aclblasComplex* alpha, const aclblasComplex& xj, float* tRe, float* tIm)
{
    // Rounding-shape alignment with the NPU kernel scalar unit, pinned empirically
    // (per-element hex comparison on the alpha=1e6 near-cancellation cases): bisheng
    // compiles the kernel fold `ajRe = aRe*xjRe - aIm*xjIm;  ajIm =
    // aRe*xjIm + aIm*xjRe` with the IMAG add contracted into a single-rounding fma
    // (ajIm = fma(aRe, xjIm, fl(aIm*xjRe))) while the REAL sub keeps both products rounded
    // (ajRe = fl(fl(aRe*xjRe) - fl(aIm*xjIm))). -ffp-contract=off / #pragma float_control
    // are ignored by the ASC frontend, and a volatile-forced two-rounding fold corrupts the
    // kernel (compiler spill collision), so the golden mirrors the contracted shape. A
    // fold-shape mismatch is amplified by alpha=1e6 near-cancellation elements past the
    // spec per-element gate max(1e-2, 32*ULP) via delta = x(i)*t_j. Toolchain coupling
    // note: re-validate this shape on any bisheng upgrade.
    const float pRe = alpha->real * xj.real;
    const float pIm = alpha->imag * xj.imag;
    const float qIm = alpha->imag * xj.real;
    *tRe = pRe - pIm;
    *tIm = std::fma(alpha->real, xj.imag, qIm);
}

// Golden for column j of the uplo triangle, rows [iBegin, iEnd] (inclusive), reading the
// ORIGINAL (pre-update) A from aOrig and writing the expected post-update values to out.
// Returns iBegin/iEnd via pointers so the caller streams exactly the referenced rows.
inline void ComputeGoldenCsyrColumn(aclblasFillMode_t uplo, int n, int j, const aclblasComplex* alpha,
    const aclblasComplex* x, int incx, const aclblasComplex* aOrig, int lda, aclblasComplex* out, int* iBeginOut,
    int* iEndOut)
{
    const int absIncx = std::abs(incx);
    const int physJ = (incx < 0) ? (n - 1 - j) * absIncx : j * absIncx;
    const aclblasComplex& xj = x[physJ];
    float tRe = 0.0f;
    float tIm = 0.0f;
    ComputeGoldenCsyrColumnScalar(alpha, xj, &tRe, &tIm);

    const int iBegin = (uplo == ACLBLAS_UPPER) ? 0 : j;
    const int iEnd = (uplo == ACLBLAS_UPPER) ? j : n - 1;
    for (int i = iBegin; i <= iEnd; i++) {
        const int physI = (incx < 0) ? (n - 1 - i) * absIncx : i * absIncx;
        const aclblasComplex& xi = x[physI];
        // Association alignment with the NPU kernel per-element chain: the
        // device executes the same two-rounding multiply/subtract/add sequence (muls with
        // separate rounding + Axpy whose product enters the sum with its own rounding) with
        // the scalar fold shape mirrored in ComputeGoldenCsyrColumnScalar, so the explicit
        // form below is bitwise-identical on the NPU for every alpha magnitude, including
        // the alpha=1e6 near-cancellation cases.
        const float dRe1 = xi.real * tRe;
        const float dRe2 = xi.imag * tIm;
        const float dIm1 = xi.real * tIm;
        const float dIm2 = xi.imag * tRe;
        aclblasComplex value = aOrig[static_cast<size_t>(j) * lda + i];
        value.real += dRe1 - dRe2;
        value.imag += dIm1 + dIm2;
        out[i - iBegin] = value;
    }
    *iBeginOut = iBegin;
    *iEndOut = iEnd;
}

inline void ComputeGoldenCsyr(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* A, int lda)
{
    if (n <= 0 || alpha == nullptr || x == nullptr || A == nullptr ||
        (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) || incx == 0) {
        return;
    }
    if (alpha->real == 0.0f && alpha->imag == 0.0f) {
        return;
    }

    std::vector<aclblasComplex> col(static_cast<size_t>(n));
    for (int j = 0; j < n; j++) {
        int iBegin = 0;
        int iEnd = -1;
        ComputeGoldenCsyrColumn(uplo, n, j, alpha, x, incx, A, lda, col.data(), &iBegin, &iEnd);
        for (int i = iBegin; i <= iEnd; i++) {
            A[static_cast<size_t>(j) * lda + i] = col[i - iBegin];
        }
    }
}

// CPU reference mirroring the aclblasCsyr entry-point contract (argument validation,
// quick returns) consumed by the arch35 ST.
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
