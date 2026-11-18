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
#include <limits>
#include <iostream>
#include <string>
#include "cann_ops_blas_common.h"

namespace chemv_test {
inline int64_t VectorIndex(int i, int n, int inc)
{
    return (inc < 0 ? (1 - static_cast<int64_t>(n)) * inc : 0) + static_cast<int64_t>(i) * inc;
}
struct ComparisonStats {
    size_t finite = 0, matched = 0, nonfinite = 0, specialMatched = 0;
    size_t relativeCount = 0, exactFinite = 0, nearZero = 0, nearZeroViolations = 0;
    double maxError = 0;
    double sumRelativeError = 0, maxRelativeError = 0;
    bool maxPassed = true;
};

inline void AccumulateFinite(ComparisonStats& stats, float a, float g, double mereThreshold)
{
    constexpr double tolerance = 1.0 / 8192.0;
    constexpr double maxAbsLimit = 0.01;
    constexpr double relativeEpsilon = 1.0 / 16384.0;
    ++stats.finite;
    const double error = std::abs(static_cast<double>(a) - g);
    stats.maxError = std::max(stats.maxError, error);
    stats.matched += error <= tolerance * (1.0 + std::abs(static_cast<double>(g)));
    // Conservative hard cap satisfies either interpretation of "0.01 or
    // 32 ULP". The fixed-version specification's ULP definition is pending
    // confirmation; do not loosen the primary gate by output magnitude.
    stats.maxPassed &= error <= maxAbsLimit;
    // Supplementary finite-value MERE/MARE follows test/frame/verify.h.
    // Special values were handled above using this task's explicit rules.
    if (a == g) {
        ++stats.exactFinite;
    } else if (std::abs(static_cast<double>(g)) < mereThreshold) {
        ++stats.nearZero;
        stats.nearZeroViolations += std::abs(static_cast<double>(a)) > maxAbsLimit;
    } else {
        const double relative = error / (std::abs(static_cast<double>(g)) + relativeEpsilon);
        stats.sumRelativeError += relative;
        stats.maxRelativeError = std::max(stats.maxRelativeError, relative);
        ++stats.relativeCount;
    }
}

inline void LogComparison(
    const ComparisonStats& stats, const std::string& name, bool imaginary, double ratio, double mere,
    double mereThreshold, double mareMultiplier, bool mixedPassed, bool relativePassed, bool pass)
{
    constexpr double maxAbsLimit = 0.01;
    std::cout << "[" << name << (imaginary ? "_imag" : "_real") << "] " << (pass ? "PASSED" : "FAILED")
              << " matched_ratio=" << ratio << " max_abs_error=" << stats.maxError << " nonfinite=" << stats.nonfinite
              << " nonfinite_matched=" << stats.specialMatched << " max_abs_limit=" << maxAbsLimit
              << " mixed=" << (mixedPassed ? "PASS" : "FAIL") << " MERE=" << mere << " MARE=" << stats.maxRelativeError
              << " mere_threshold=" << mereThreshold << " mare_multiplier=" << mareMultiplier
              << " relative_count=" << stats.relativeCount << " exact_finite=" << stats.exactFinite
              << " near_zero=" << stats.nearZero << " near_zero_cap_violations=" << stats.nearZeroViolations
              << " supplementary=" << (relativePassed ? "PASS" : "FAIL") << std::endl;
}

inline bool Compare(
    const aclblasComplex* actual, const aclblasComplex* golden, int n, int inc, bool imaginary, const std::string& name,
    double mereThreshold = 1.0 / 8192.0, double mareMultiplier = 10.0)
{
    ComparisonStats stats;
    for (int i = 0; i < n; ++i) {
        const auto idx = VectorIndex(i, n, inc);
        const float a = imaginary ? actual[idx].imag : actual[idx].real;
        const float g = imaginary ? golden[idx].imag : golden[idx].real;
        if (!std::isfinite(a) || !std::isfinite(g)) {
            ++stats.nonfinite;
            if (!std::isfinite(a) && !std::isfinite(g) && (std::isnan(a) || std::isnan(g) || a == g)) {
                ++stats.specialMatched;
            }
            continue;
        }
        AccumulateFinite(stats, a, g, mereThreshold);
    }
    const double ratio = stats.finite == 0 ? 1.0 : static_cast<double>(stats.matched) / stats.finite;
    const double mere = stats.relativeCount == 0 ? 0.0 : stats.sumRelativeError / stats.relativeCount;
    const bool mixedPassed = ratio >= 0.99 && stats.maxPassed && stats.nonfinite == stats.specialMatched;
    const bool relativePassed = std::isfinite(mereThreshold) && mereThreshold > 0 && std::isfinite(mareMultiplier) &&
                                mareMultiplier > 0 && stats.nearZeroViolations == 0 && mere < mereThreshold &&
                                stats.maxRelativeError < mareMultiplier * mereThreshold;
    const bool pass = mixedPassed && relativePassed;
    LogComparison(
        stats, name, imaginary, ratio, mere, mereThreshold, mareMultiplier, mixedPassed, relativePassed, pass);
    return pass;
}
} // namespace chemv_test
