/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <vector>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

// Sentinel: all validations passed, proceed to compute.
static const int CHER2K_CPU_VALID_OK = 0x7FFFFFFF;

static bool Cher2kIsValidUplo(aclblasFillMode_t uplo) { return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER; }

static bool Cher2kIsValidTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

// Parameter-mirror validation, replicating the operator validation-order table §4.1 #1..#11
// one-for-one (test plan §6.4). Runs before the cblas call because cblas would
// crash / xerbla-abort on null pointers. Returns aclblasStatus_t so the NPU and
// CPU paths can be compared in the mirror-consistency host tests (§6.6).
static int ValidateCher2kCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc,
    const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* B, const float* beta, aclblasComplex* C,
    float betaVal)
{
    // #1 handle == nullptr
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    // #2 uplo ∈ {UPPER(121), LOWER(122)} -> INVALID_ENUM (cher2k-specific, differs
    // from cherk which returns INVALID_VALUE)
    if (!Cher2kIsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    // #3 trans ∈ {OP_N, OP_T, OP_C} -> INVALID_ENUM for out-of-range
    if (!Cher2kIsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    // #4 trans == OP_T: legal enum but unsupported -> INVALID_VALUE
    if (trans == ACLBLAS_OP_T) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // #5 n >= 0, k >= 0
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // #6/#7 lda/ldb >= (N ? max(1,n) : max(1,k))
    int minLd = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < minLd || ldb < minLd) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // #8 ldc >= max(1,n)
    if (ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // #9 alpha != null, beta != null
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // #10 A/B != null when n > 0 and k > 0
    if ((A == nullptr || B == nullptr) && n > 0 && k > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // #11 C == null && n > 0: beta != 0 -> INVALID_VALUE; beta == 0 -> SUCCESS
    // without any write (C-pointer semantics, design §1.1/§4.1 #11)
    if (C == nullptr && n > 0) {
        if (betaVal != 0.0f) {
            return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
        }
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
    }
    return CHER2K_CPU_VALID_OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// [A' golden dual chain] Precision reference chain upgraded to double: the main
// criterion chain feeds the float inputs up to double complex, calls cblas_zher2k
// and keeps the output in double (no round-back to float, which would reintroduce a
// ~0.5ULP_f32 rounding step in the reference chain). The original float
// cblas_cher2k chain is kept verbatim as a CSV cross-check column and does not
// participate in the pass/fail decision.
// All thresholds are unchanged (atol 2^-16 / rtol 2^-10 / max(1e-2,32ULP) / ratio>=0.99);
// complex64 real/imag are still judged as FLOAT32 (only the golden generation precision
// changes, not the comparison precision).
// ═══════════════════════════════════════════════════════════════════════════════

// Host buffer for the double main-criterion chain: ldc*n column-major, each element
// {double real; double imag}. The non-uplo triangle and ldc padding keep the
// up-converted double values of the old input C -- the float->double conversion is
// exact (float is a subset of double), so on compare-back the float old value equals
// the double old value and the non-uplo EXACT / canary criteria are not relaxed.
struct Cher2kGoldenDbl {
    std::vector<aclblasDoubleComplex> c; // ldc*n, column-major (main-criterion golden)
};

// Manual beta scaling of the golden when k == 0 and A/B are null (shared by the float
// and double numeric chains): only the uplo triangle is written; the diagonal imaginary
// part is forced to 0 (BLAS CHER2K contract); when beta == 0 the whole triangle is
// zeroed without reading the old value (avoiding NaN propagation), matching the
// operator's isBetaZero combine branch.
template <typename ComplexT, typename RealT>
static void Cher2kScaleUploTriangle(ComplexT* c, int n, int ldc, bool upper, RealT beta)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            const bool isUplo = upper ? (i <= j) : (i >= j);
            if (!isUplo) {
                continue;
            }
            const size_t idx = i + static_cast<size_t>(j) * ldc;
            if (beta == RealT(0)) {
                c[idx] = ComplexT{0, 0};
            } else {
                const ComplexT old = c[idx];
                c[idx] = (i == j) ? ComplexT{beta * old.real, RealT(0)} : ComplexT{beta * old.real, beta * old.imag};
            }
        }
    }
}

// CPU golden -- double main-criterion chain (A'). The numeric output is written to d.c
// (double, no round-back to float); the return-code semantics mirror operator §4.1/§4.2
// one-for-one (validation happens before the numeric chain and is orthogonal to precision).
// trans mapping (design §6.4): CHER2K accepts N and C only --
//   trans == OP_N -> CblasNoTrans
//   trans == OP_C -> CblasConjTrans
// OP_T is rejected during validation above (unlike cherk, which merges T/C).
// Diagonal imaginary part zeroed / uplo triangle only / k=0 and alpha=0 manual branches /
// C-null semantics are all kept equivalent on the double chain.
static aclblasStatus_t Cher2kRunGoldenDbl(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const float* beta, const aclblasComplex* C,
    int ldc, Cher2kGoldenDbl& d)
{
    const aclblasComplex alphaZero{0.0f, 0.0f};
    const aclblasComplex alphaVal = (alpha != nullptr) ? *alpha : alphaZero;
    const float betaVal = (beta != nullptr) ? *beta : 0.0f;

    int st = ValidateCher2kCpuParams(
        handle, uplo, trans, n, k, lda, ldb, ldc, alpha, A, B, beta, const_cast<aclblasComplex*>(C), betaVal);
    if (st != CHER2K_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    // Up-convert the old input C to double (exact); the non-uplo triangle and padding keep the old values.
    // Note: this runs before the quick-return check -- the quick-return branch's "does not touch C"
    // semantics are expressed by "d.c == old input C" (equivalent to the float chain's cGolden not
    // being written), so on compare the NPU output (also unwritten) matches the old C exactly. For
    // n==0 / C==nullptr the caller does not enter the comparison, and we return early here to avoid
    // dereferencing.
    if (n == 0 || C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const size_t cElems = static_cast<size_t>(ldc) * static_cast<size_t>(n);
    d.c.assign(cElems, aclblasDoubleComplex{0.0, 0.0});
    for (size_t i = 0; i < cElems; i++) {
        d.c[i] = aclblasDoubleComplex{static_cast<double>(C[i].real), static_cast<double>(C[i].imag)};
    }

    // quick returns mirroring operator §4.2 (validation has already passed):
    // A/B/C not accessed -> d.c keeps the old input C (filled above), matching the
    // operator's "no write" behavior.
    if ((alphaVal.real == 0.0f && alphaVal.imag == 0.0f || k == 0) && betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // alpha == 0 or k == 0 reduce to beta scaling of the uplo triangle; cblas
    // handles that naturally (A/B may be null here only when k == 0), but A/B
    // must still be non-null pointers for the cblas call, so guard k == 0.
    // (double chain manual branch: branch-for-branch equivalent to the float chain, with
    //  the arithmetic carried out in double)
    if (k == 0 && (A == nullptr || B == nullptr)) {
        Cher2kScaleUploTriangle(d.c.data(), n, ldc, uplo == ACLBLAS_UPPER, static_cast<double>(betaVal));
        return ACLBLAS_STATUS_SUCCESS;
    }

    CBLAS_TRANSPOSE cblasTrans = (trans == ACLBLAS_OP_N) ? CblasNoTrans : CblasConjTrans;

    // Up-convert the float inputs to double complex (exact, no rounding); layout/leading
    // dimension/column-major order unchanged.
    const int abCols = (trans == ACLBLAS_OP_N) ? k : n;
    const size_t aElems = static_cast<size_t>(lda) * static_cast<size_t>(abCols);
    const size_t bElems = static_cast<size_t>(ldb) * static_cast<size_t>(abCols);
    std::vector<aclblasDoubleComplex> aD(aElems);
    std::vector<aclblasDoubleComplex> bD(bElems);
    for (size_t i = 0; i < aElems; i++) {
        aD[i] = aclblasDoubleComplex{static_cast<double>(A[i].real), static_cast<double>(A[i].imag)};
    }
    for (size_t i = 0; i < bElems; i++) {
        bD[i] = aclblasDoubleComplex{static_cast<double>(B[i].real), static_cast<double>(B[i].imag)};
    }
    const aclblasDoubleComplex alphaD{static_cast<double>(alphaVal.real), static_cast<double>(alphaVal.imag)};

    // Main-criterion call: the standard C interface cblas_zher2k (provided by the local
    // netlib 3.10.0-2ubuntu1; nm -D libblas.so.3 shows cblas_zher2k T). beta is double.
    cblas_zher2k(
        CblasColMajor, ToCblasUplo(uplo), cblasTrans, n, k, static_cast<const void*>(&alphaD),
        static_cast<const void*>(aD.data()), lda, static_cast<const void*>(bD.data()), ldb,
        static_cast<double>(betaVal), static_cast<void*>(d.c.data()), ldc);

    return ACLBLAS_STATUS_SUCCESS;
}

// Cross-check chain: the original float cblas golden kept verbatim (the sole criterion chain
// before A'), its output used only as CSV cross-check columns (C_uplo_*_float_xcheck) and never
// in the pass/fail decision. Signature and semantics are identical to the pre-upgrade
// aclblasCher2k_cpu -- the host tests' return-code mirror consistency (16 cases) still goes through
// this chain, and the return-code semantics are orthogonal to numeric precision, so behavior is
// unchanged.
inline aclblasStatus_t aclblasCher2k_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const float* beta, aclblasComplex* C, int ldc)
{
    const aclblasComplex alphaZero{0.0f, 0.0f};
    const aclblasComplex alphaVal = (alpha != nullptr) ? *alpha : alphaZero;
    const float betaVal = (beta != nullptr) ? *beta : 0.0f;

    int st = ValidateCher2kCpuParams(handle, uplo, trans, n, k, lda, ldb, ldc, alpha, A, B, beta, C, betaVal);
    if (st != CHER2K_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    // quick returns mirroring operator §4.2 (validation has already passed)
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (C == nullptr) { // n > 0 with beta == 0: success without write
        return ACLBLAS_STATUS_SUCCESS;
    }
    if ((alphaVal.real == 0.0f && alphaVal.imag == 0.0f || k == 0) && betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS; // A/B/C not accessed
    }

    // alpha == 0 or k == 0 reduce to beta scaling of the uplo triangle; cblas
    // handles that naturally (A/B may be null here only when k == 0), but A/B
    // must still be non-null pointers for the cblas call, so guard k == 0.
    if (k == 0 && (A == nullptr || B == nullptr)) {
        Cher2kScaleUploTriangle(C, n, ldc, uplo == ACLBLAS_UPPER, betaVal);
        return ACLBLAS_STATUS_SUCCESS;
    }

    CBLAS_TRANSPOSE cblasTrans = (trans == ACLBLAS_OP_N) ? CblasNoTrans : CblasConjTrans;

    // aclblasComplex = {float real; float imag} is binary-compatible with the
    // cblas complex layout, so the void* cast is a pure reinterpret.
    cblas_cher2k(
        CblasColMajor, ToCblasUplo(uplo), cblasTrans, n, k, static_cast<const void*>(&alphaVal),
        static_cast<const void*>(A), lda, static_cast<const void*>(B), ldb, betaVal, static_cast<void*>(C), ldc);

    return ACLBLAS_STATUS_SUCCESS;
}
