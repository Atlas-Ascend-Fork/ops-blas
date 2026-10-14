/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSPR2_GOLDEN_H
#define SSPR2_GOLDEN_H

#include <climits>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t ValidateSspr2ParamsCpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const float* alpha, const float* x, int incx, const float* y, int incy, float* ap)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;
    if (alpha == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == INT_MIN) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incy == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incy == INT_MIN) return ACLBLAS_STATUS_INVALID_VALUE;
    if (x == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (y == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (ap == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSspr2_cpu(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const float* alpha,
    const float* x,
    int incx,
    const float* y,
    int incy,
    float* ap)
{
    aclblasStatus_t st = ValidateSspr2ParamsCpu(handle, uplo, n, alpha, x, incx, y, incy, ap);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    float alphaVal = *alpha;
    if (alphaVal == 0.0f) return ACLBLAS_STATUS_SUCCESS;

    cblas_sspr2(CblasColMajor, ToCblasUplo(uplo), n, alphaVal, x, incx, y, incy, ap);

    return ACLBLAS_STATUS_SUCCESS;
}

// ── sspr2 arch22 golden: built-in cblas_sspr2 reference + tolerance checks ──
// Reference semantics per Netlib BLAS sspr2.f:
//   1) alpha == 0 quick return: x/y/AP are not referenced;
//   2) n <= 0 quick return (n == 0 is a legal no-op);
//   3) negative stride reverse traversal: logical element xv[i] lives at
//      x[(n-1-i)*|incx|] (0-based equivalent of the Netlib backward start);
//   4) column-major packed indexing (diagonal included):
//        UPPER: k = i + j*(j+1)/2            (j >= i)
//        LOWER: k = j*(2n-j-1)/2 + i         (i >= j)
//      consistent with cblas (Accelerate/OpenBLAS) and the arch35 kernel
//      (Sspr2ColumnBase col*(2n-col+1)/2);
//   5) accumulation order aligned with the Netlib F77 inner loop:
//      AP[k] += X(I)*TEMP1 + Y(I)*TEMP2 with TEMP1 = ALPHA*Y(JY),
//      TEMP2 = ALPHA*X(JX). The inner products are kept contract-free
//      (volatile blocks the default -ffp-contract=fast FMA contraction),
//      making the rounding sequence of this reference bitwise deterministic
//      under any toolchain and identical to the arch22 kernel update chain
//      (Muls/Muls/Add/Add, same rounding order).
// The Netlib zero-column skip (X(JX)==0 && Y(JY)==0) is a value-only
// optimization and is not replicated here (unconditional accumulation); on
// the finite-valued case domain the two forms are element-wise identical.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sspr2_st {

enum class RefUplo : int { REF_UPPER = 121, REF_LOWER = 122 };

inline std::size_t PackedIndexUpper(int i, int j) { return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * (static_cast<std::size_t>(j) + 1) / 2; }
// Column-major lower-triangular packing (column j contiguous, i=j..n-1);
// n is passed because the layout depends on the order.
inline std::size_t PackedIndexLower(int i, int j, int n)
{
    return static_cast<std::size_t>(j) * (2 * static_cast<std::size_t>(n) - static_cast<std::size_t>(j) - 1) / 2 +
           static_cast<std::size_t>(i);
}

// Logical element access: xv[i] (i in 0..n-1), negative-stride backward start.
template <typename VecT>
inline float RefStridedAt(const VecT& v, int n, int i, int inc)
{
    if (inc > 0) {
        return v[static_cast<std::size_t>(i) * static_cast<std::size_t>(inc)];
    }
    const std::int64_t absInc = -static_cast<std::int64_t>(inc);
    return v[static_cast<std::size_t>(static_cast<std::int64_t>(n - 1 - i) * absInc)];
}

// C++ reference implementation of Netlib sspr2.f semantics (AP accumulated
// in place). uplo: RefUplo; n >= 0; alpha/x/y/ap are host-side float
// pointers (no handle / status-code semantics at the golden layer).
inline void cblas_sspr2_ref(RefUplo uplo, int n, float alpha,
                            const float* x, int incx, const float* y, int incy, float* ap)
{
    if (n <= 0 || alpha == 0.0f) {
        return; // Netlib quick return (before any x/y/AP reference)
    }
    for (int j = 0; j < n; j++) {
        const float xj = RefStridedAt(x, n, j, incx);
        const float yj = RefStridedAt(y, n, j, incy);
        const float temp1 = alpha * yj;
        const float temp2 = alpha * xj;
        if (uplo == RefUplo::REF_UPPER) {
            const std::size_t colBase = PackedIndexUpper(0, j);
            for (int i = 0; i <= j; i++) {
                // Contract-free sequential rounding: products rounded
                // separately, then RN(h1+h2), then RN(ap+h). volatile blocks
                // the mul+add FMA contraction applied by default at -O3
                // (-ffp-contract=fast); the arch22 kernel update chain
                // (Muls/Muls/Add/Add) follows exactly this rounding order,
                // giving a bitwise-equal golden.
                const float xi = RefStridedAt(x, n, i, incx);
                const float yi = RefStridedAt(y, n, i, incy);
                volatile float h1 = xi * temp1;
                volatile float h2 = yi * temp2;
                ap[colBase + static_cast<std::size_t>(i)] += h1 + h2;
            }
        } else {
            for (int i = j; i < n; i++) {
                // Same contract-free sequence as the UPPER branch (above).
                const float xi = RefStridedAt(x, n, i, incx);
                const float yi = RefStridedAt(y, n, i, incy);
                volatile float h1 = xi * temp1;
                volatile float h2 = yi * temp2;
                ap[PackedIndexLower(i, j, n)] += h1 + h2;
            }
        }
    }
}

// Tolerance criteria: per-element |actual - golden| <= atol + rtol * |golden| (see Pass()
// for the full dual-criterion policy and the NaN/Inf / EXACT handling).

struct ToleranceReport {
    std::size_t count = 0;
    std::size_t mismatchCount = 0;   // elements violating the atol+rtol bound
    std::size_t limitViolations = 0; // elements violating max(1e-2, 32*ULP)
    std::size_t nanMismatch = 0;     // NaN/Inf special-value mismatches
    double maxAbsErr = 0.0;
    double matchedRatio = 1.0;

    bool Pass(double requiredRatio) const
    {
        // Dual-criterion policy:
        // (1) matched_ratio >= required (allows <= 1% of the elements to
        //     exceed the per-element atol+rtol bound);
        // (2) every element |diff| <= max(1e-2, 32*ULP@|golden|)
        //     (limitViolations == 0); NaN/Inf positions must match the
        //     golden (nanMismatch == 0).
        // mismatchCount == 0 is NOT additionally required (that would force
        // matched_ratio == 1.0): the fixed NPU accumulation order can differ
        // from the cblas reference by a few elements that exceed the
        // per-element bound while staying far below 32*ULP at large
        // magnitudes.
        return nanMismatch == 0 && matchedRatio >= requiredRatio && limitViolations == 0;
    }
};

inline double GetUlpsAt(double magnitude)
{
    constexpr int kMantissaBits = 23; // float32
    constexpr int kEmin = -126;
    double absMag = std::abs(magnitude);
    if (absMag == 0.0) {
        return std::pow(2.0, static_cast<double>(kEmin - kMantissaBits));
    }
    int exp = 0;
    std::frexp(absMag, &exp);
    return std::ldexp(1.0, std::max(exp - 1, kEmin) - kMantissaBits);
}

// Per-element comparison (equal_nan / same-source Inf).
inline ToleranceReport VerifyGoldenTolerance(const float* actual, const float* golden, std::size_t count,
                                             double rtol, double atol, double fixedMaxAbsLimit)
{
    ToleranceReport r;
    r.count = count;
    for (std::size_t i = 0; i < count; i++) {
        const double a = static_cast<double>(actual[i]);
        const double g = static_cast<double>(golden[i]);
        const bool aNan = std::isnan(a);
        const bool gNan = std::isnan(g);
        const bool aInf = std::isinf(a);
        const bool gInf = std::isinf(g);
        bool matched;
        if (aNan && gNan) {
            matched = true;
        } else if (aInf && gInf) {
            matched = (a > 0) == (g > 0);
        } else if (aNan || gNan || aInf || gInf) {
            matched = false;
            r.nanMismatch++;
        } else {
            const double diff = std::abs(a - g);
            if (diff > r.maxAbsErr) {
                r.maxAbsErr = diff;
            }
            const double tolLimit = atol + rtol * std::abs(g);
            const double ulpLimit = std::max(fixedMaxAbsLimit, 32.0 * GetUlpsAt(std::abs(g)));
            matched = diff <= tolLimit;
            if (!matched) {
                r.mismatchCount++;
            }
            if (diff > ulpLimit) {
                r.limitViolations++;
            }
        }
    }
    const std::size_t totalMismatch = r.mismatchCount + r.nanMismatch;
    const std::size_t matchedCount = (count >= totalMismatch) ? count - totalMismatch : 0;
    r.matchedRatio = (count > 0) ? static_cast<double>(matchedCount) / static_cast<double>(count) : 1.0;
    return r;
}

// EXACT bitwise comparison (AP_UNCHANGED invariant; NaN bit patterns must
// match as well).
inline bool VerifyExactBits(const float* actual, const float* golden, std::size_t count)
{
    for (std::size_t i = 0; i < count; i++) {
        const unsigned char* pa = reinterpret_cast<const unsigned char*>(&actual[i]);
        const unsigned char* pg = reinterpret_cast<const unsigned char*>(&golden[i]);
        for (std::size_t b = 0; b < sizeof(float); b++) {
            if (pa[b] != pg[b]) {
                return false;
            }
        }
    }
    return true;
}

inline std::size_t CountNan(const float* v, std::size_t count)
{
    std::size_t c = 0;
    for (std::size_t i = 0; i < count; i++) {
        if (std::isnan(v[i])) {
            c++;
        }
    }
    return c;
}

} // namespace sspr2_st

// Golden computation (host side, accumulated in place; the alpha=0 / n=0
// quick returns are inherent to the semantics).
inline void Sspr2GoldenCompute(int uplo, int n, float alpha,
                               const float* x, int incx, const float* y, int incy, float* ap)
{
    const sspr2_st::RefUplo u =
        (uplo == 121) ? sspr2_st::RefUplo::REF_UPPER : sspr2_st::RefUplo::REF_LOWER;
    sspr2_st::cblas_sspr2_ref(u, n, alpha, x, incx, y, incy, ap);
}

#endif // SSPR2_GOLDEN_H
