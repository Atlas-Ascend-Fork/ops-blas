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
#include <limits>

struct ChemvComponentStats {
    size_t finite = 0;
    size_t matched = 0;
    size_t nonfinite = 0;
    size_t nonfiniteMatched = 0;
    double maxAbs = 0.0;
    double relativeSum = 0.0;
    double maxRelative = 0.0;
    bool withinMaxAbsLimit = true;

    void Observe(float actual, float golden)
    {
        if (!std::isfinite(actual) || !std::isfinite(golden)) {
            ++nonfinite;
            const bool bothNonfinite = !std::isfinite(actual) && !std::isfinite(golden);
            const bool equivalent = bothNonfinite && (std::isnan(actual) || std::isnan(golden) || actual == golden);
            if (equivalent) {
                ++nonfiniteMatched;
            }
            return;
        }
        ++finite;
        const double error = std::abs(static_cast<double>(actual) - static_cast<double>(golden));
        constexpr double kTol = 1.0 / 8192.0; // 2^-13
        if (error <= kTol + kTol * std::abs(static_cast<double>(golden))) {
            ++matched;
        }
        maxAbs = std::max(maxAbs, error);
        const float mag = std::abs(golden);
        const float nextMag = std::nextafter(mag, std::numeric_limits<float>::infinity());
        const double ulpLimit = 32.0 * static_cast<double>(nextMag - mag);
        if (error > std::max(1.0e-2, ulpLimit)) {
            withinMaxAbsLimit = false;
        }
        const double relative = error / std::max(std::abs(static_cast<double>(golden)), 1.0e-30);
        relativeSum += relative;
        maxRelative = std::max(maxRelative, relative);
    }

    double MatchedRatio() const { return finite == 0 ? 1.0 : static_cast<double>(matched) / finite; }

    bool Passed() const
    {
        return MatchedRatio() >= 0.99 && withinMaxAbsLimit && (nonfinite == nonfiniteMatched);
    }
};
