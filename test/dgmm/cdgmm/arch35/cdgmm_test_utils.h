/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include "cann_ops_blas_common.h"

namespace cdgmm_arch35_test {

constexpr aclblasComplex SENTINEL = {-999.0F, 997.0F};
constexpr double TOLERANCE = 0.0001220703125; // Task-specific 2^-13, both atol and rtol.

inline uint64_t AbsIncrement(int incx)
{
    return incx < 0 ? static_cast<uint64_t>(-static_cast<int64_t>(incx)) : static_cast<uint64_t>(incx);
}

inline bool SameBits(const aclblasComplex& a, const aclblasComplex& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

struct ComponentStats {
    uint64_t finite = 0;
    uint64_t matched = 0;
    uint64_t nonfinite = 0;
    uint64_t nonfiniteMatched = 0;
    uint64_t hardFailures = 0;
    double maxAbsError = 0.0;

    void Observe(float actual, float golden)
    {
        if (!std::isfinite(actual) || !std::isfinite(golden)) {
            ++nonfinite;
            const bool bothNonfinite = !std::isfinite(actual) && !std::isfinite(golden);
            const bool equivalent = bothNonfinite && (std::isnan(actual) || std::isnan(golden) ||
                                                      std::signbit(actual) == std::signbit(golden));
            nonfiniteMatched += equivalent ? 1U : 0U;
            return;
        }
        ++finite;
        const double error = std::abs(static_cast<double>(actual) - golden);
        maxAbsError = std::max(maxAbsError, error);
        matched += error <= TOLERANCE + TOLERANCE * std::abs(static_cast<double>(golden)) ? 1U : 0U;
        int exponent = 0;
        std::frexp(std::abs(static_cast<double>(golden)), &exponent);
        const double ulp = std::ldexp(1.0, std::max(exponent - 1, -126) - 23);
        hardFailures += error > std::max(0.01, 32.0 * ulp) ? 1U : 0U;
    }

    bool Passed() const
    {
        return (finite == 0U || static_cast<double>(matched) / finite >= 0.99) && hardFailures == 0U &&
               nonfinite == nonfiniteMatched;
    }

    void Print(const std::string& caseName, const char* component) const
    {
        std::cout << "CDGMM_PRECISION case=" << caseName << " component=" << component << " finite=" << finite
                  << " matched=" << matched << " max_abs_error=" << maxAbsError << " nonfinite=" << nonfinite
                  << " nonfinite_matched=" << nonfiniteMatched << " hard_failures=" << hardFailures << '\n';
    }
};

struct MatrixStats {
    ComponentStats real;
    ComponentStats imag;
    uint64_t paddingFailures = 0;
    uint64_t unwritten = 0;

    void Observe(const aclblasComplex& actual, const aclblasComplex& golden, bool padding)
    {
        if (padding) {
            paddingFailures += SameBits(actual, SENTINEL) ? 0U : 1U;
            return;
        }
        unwritten += SameBits(actual, SENTINEL) && !SameBits(golden, SENTINEL) ? 1U : 0U;
        real.Observe(actual.real, golden.real);
        imag.Observe(actual.imag, golden.imag);
    }

    bool Passed() const { return real.Passed() && imag.Passed() && paddingFailures == 0U && unwritten == 0U; }
};

} // namespace cdgmm_arch35_test
