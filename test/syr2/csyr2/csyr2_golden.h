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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "cann_ops_blas_common.h"

namespace csyr2_test {

// Deliberately independent from the Kernel implementation.  Keep this routine
// scalar and explicit so a shared indexing/multiply bug cannot make Golden and
// NPU results agree for the wrong reason.
inline aclblasComplex Mul(const aclblasComplex& a, const aclblasComplex& b)
{
    aclblasComplex r;
    r.real = a.real * b.real - a.imag * b.imag;
    r.imag = a.real * b.imag + a.imag * b.real;
    return r;
}

inline int64_t LogicalOffset(int logicalIndex, int n, int increment)
{
    const int64_t i = static_cast<int64_t>(logicalIndex);
    const int64_t inc = static_cast<int64_t>(increment);
    if (inc >= 0) {
        return i * inc;
    }
    // BLAS negative-increment convention: logical element 0 starts at
    // physical offset (n-1)*abs(inc), then walks backwards.
    return static_cast<int64_t>(n - 1 - logicalIndex) * (-inc);
}

inline bool IsUpper(aclblasFillMode_t uplo)
{
    return uplo == ACLBLAS_UPPER;
}

inline bool IsZero(const aclblasComplex& value)
{
    return value.real == 0.0f && value.imag == 0.0f;
}

inline void Csyr2GoldenColumn(
    aclblasFillMode_t uplo, int n, const aclblasComplex& alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, int col, aclblasComplex* column)
{
    if (n <= 0 || IsZero(alpha)) {
        return;
    }

    // The caller owns argument validation.  The reference intentionally does
    // not reinterpret invalid pointers or dimensions as a successful call.
    const int64_t xCol = LogicalOffset(col, n, incx);
    const int64_t yCol = LogicalOffset(col, n, incy);

    const int rowBegin = IsUpper(uplo) ? 0 : col;
    const int rowEnd = IsUpper(uplo) ? col : n - 1;
    for (int row = rowBegin; row <= rowEnd; ++row) {
        const aclblasComplex xRow = x[LogicalOffset(row, n, incx)];
        const aclblasComplex yRow = y[LogicalOffset(row, n, incy)];
        // Scale the row operands. This is the same symmetric rank-2 formula,
        // and also matches cuBLAS's observed Inf/NaN propagation. Scaling the
        // column operands instead can exchange Inf and NaN across triangles.
        const aclblasComplex xy = Mul(x[xCol], Mul(alpha, yRow));
        const aclblasComplex yx = Mul(y[yCol], Mul(alpha, xRow));

        aclblasComplex& value = column[row];
        // Preserve the reference evaluation order used by the design:
        // (A + term0) + term1.  Do not algebraically reassociate this.
        value.real = (value.real + xy.real) + yx.real;
        value.imag = (value.imag + xy.imag) + yx.imag;
    }
}

inline void Csyr2Golden(
    aclblasFillMode_t uplo, int n, const aclblasComplex& alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (n <= 0 || IsZero(alpha)) {
        return;
    }
    for (int col = 0; col < n; ++col) {
        Csyr2GoldenColumn(uplo, n, alpha, x, incx, y, incy, col,
            A + static_cast<int64_t>(col) * static_cast<int64_t>(lda));
    }
}

inline std::vector<aclblasComplex> MakeLogicalVector(const std::vector<aclblasComplex>& physical,
                                                     int n, int increment)
{
    std::vector<aclblasComplex> logical(static_cast<size_t>(std::max(n, 0)));
    for (int i = 0; i < n; ++i) {
        logical[static_cast<size_t>(i)] = physical[static_cast<size_t>(LogicalOffset(i, n, increment))];
    }
    return logical;
}

inline bool BitwiseEqual(const aclblasComplex& a, const aclblasComplex& b)
{
    return std::memcmp(&a, &b, sizeof(aclblasComplex)) == 0;
}

inline double UlpForFloat(double value)
{
    const double magnitude = std::abs(value);
    if (magnitude == 0.0) {
        return std::ldexp(1.0, -149);
    }
    if (!std::isfinite(magnitude)) {
        return std::numeric_limits<double>::infinity();
    }

    int exponent = 0;
    std::frexp(magnitude, &exponent);
    // float mantissa has 23 explicit bits.  For subnormals the ULP is 2^-149.
    return std::ldexp(1.0, std::max(exponent - 1, -126) - 23);
}

struct AccuracyConfig {
    double atol = std::ldexp(1.0, -16);
    double rtol = std::ldexp(1.0, -10);
    double capFixed = 1e-2;
    double mereThreshold = std::ldexp(1.0, -13);
    double mareMultiplier = 10.0;
    double mereEpsilon = std::ldexp(1.0, -14);
    double minMatchedRatio = 0.99;
};

struct ComponentAccuracy {
    size_t total = 0;
    size_t matched = 0;
    size_t capFailCount = 0;
    size_t specialMismatch = 0;
    size_t mereEligible = 0;
    size_t nearZeroSkipped = 0;
    size_t outlierCount = 0;
    double mereSum = 0.0;
    double mare = 0.0;
    double maxAbsError = 0.0;
    size_t worstIndex = 0;
    double worstActual = 0.0;
    double worstGolden = 0.0;

    double matchedRatio() const
    {
        return total == 0 ? 1.0 : static_cast<double>(matched) / static_cast<double>(total);
    }

    double mere() const
    {
        return mereEligible == 0 ? 0.0 : mereSum / static_cast<double>(mereEligible);
    }
};

struct ComplexAccuracy {
    ComponentAccuracy real;
    ComponentAccuracy imag;

    bool pass() const
    {
        return real.total == imag.total;
    }
};

inline bool BothNaN(double actual, double golden)
{
    return std::isnan(actual) && std::isnan(golden);
}

inline bool SameInf(double actual, double golden)
{
    return std::isinf(actual) && std::isinf(golden) && std::signbit(actual) == std::signbit(golden);
}

inline void AccumulateComponent(ComponentAccuracy& stats, double actual, double golden,
                                size_t index, const AccuracyConfig& cfg)
{
    ++stats.total;

    if (BothNaN(actual, golden) || SameInf(actual, golden)) {
        ++stats.matched;
        return;
    }

    const bool actualFinite = std::isfinite(actual);
    const bool goldenFinite = std::isfinite(golden);
    if (!actualFinite || !goldenFinite) {
        ++stats.specialMismatch;
        return;
    }

    const double diff = std::abs(actual - golden);
    const double tolerance = cfg.atol + cfg.rtol * std::abs(golden);
    if (diff <= tolerance) {
        ++stats.matched;
    }

    const double cap = std::max(cfg.capFixed, 32.0 * UlpForFloat(golden));
    if (diff > cap) {
        ++stats.capFailCount;
    }

    if (diff > stats.maxAbsError) {
        stats.maxAbsError = diff;
        stats.worstIndex = index;
        stats.worstActual = actual;
        stats.worstGolden = golden;
    }

    // MERE/MARE follow the repository's observed verification contract:
    // exact matches are ignored and |golden| below the threshold is excluded.
    if (actual == golden || std::abs(golden) < cfg.mereThreshold) {
        if (actual != golden) {
            ++stats.nearZeroSkipped;
        }
        return;
    }

    const double relativeError = diff / (std::abs(golden) + cfg.mereEpsilon);
    stats.mereSum += relativeError;
    ++stats.mereEligible;
    stats.mare = std::max(stats.mare, relativeError);
    if (relativeError > cfg.mareMultiplier * cfg.mereThreshold) {
        ++stats.outlierCount;
    }
}

inline void Accumulate(ComplexAccuracy& stats, const aclblasComplex& actual,
                       const aclblasComplex& golden, size_t index, const AccuracyConfig& cfg = {})
{
    AccumulateComponent(stats.real, actual.real, golden.real, index, cfg);
    AccumulateComponent(stats.imag, actual.imag, golden.imag, index, cfg);
}

inline void MergeComponent(ComponentAccuracy& dst, const ComponentAccuracy& src)
{
    dst.total += src.total;
    dst.matched += src.matched;
    dst.capFailCount += src.capFailCount;
    dst.specialMismatch += src.specialMismatch;
    dst.mereEligible += src.mereEligible;
    dst.nearZeroSkipped += src.nearZeroSkipped;
    dst.outlierCount += src.outlierCount;
    dst.mereSum += src.mereSum;

    if (src.maxAbsError > dst.maxAbsError) {
        dst.maxAbsError = src.maxAbsError;
        dst.worstIndex = src.worstIndex;
        dst.worstActual = src.worstActual;
        dst.worstGolden = src.worstGolden;
    }
    dst.mare = std::max(dst.mare, src.mare);
}

inline void Merge(ComplexAccuracy& dst, const ComplexAccuracy& src)
{
    MergeComponent(dst.real, src.real);
    MergeComponent(dst.imag, src.imag);
}

inline bool ComponentPass(const ComponentAccuracy& stats, const AccuracyConfig& cfg)
{
    return stats.matchedRatio() >= cfg.minMatchedRatio && stats.capFailCount == 0 && stats.specialMismatch == 0;
}

inline bool MereMarePass(const ComponentAccuracy& stats, const AccuracyConfig& cfg)
{
    if (stats.specialMismatch != 0) {
        return false;
    }
    if (stats.mereEligible == 0) {
        return true;
    }
    return stats.mere() < cfg.mereThreshold && stats.mare < cfg.mareMultiplier * cfg.mereThreshold;
}

inline bool CasePass(const ComplexAccuracy& stats, const AccuracyConfig& cfg = {})
{
    return ComponentPass(stats.real, cfg) && ComponentPass(stats.imag, cfg) &&
           MereMarePass(stats.real, cfg) && MereMarePass(stats.imag, cfg);
}

} // namespace csyr2_test

namespace csyr2_arch22_test {

void Csyr2Golden(
    aclblasFillMode_t uplo,
    int n,
    aclblasComplex alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);

}  // namespace csyr2_arch22_test
