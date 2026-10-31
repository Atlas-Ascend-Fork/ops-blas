/*
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
#include <random>
#include <limits>
#include "fill.h"
#include "verify.h"
#include "cann_ops_blas.h"

inline float CtrsvSeedParameter(uint32_t seed, uint32_t salt, float lower, float upper)
{
    constexpr uint32_t buckets = 10001U;
    const uint32_t mixed = seed * 1664525U + 1013904223U + salt;
    const float fraction = static_cast<float>(mixed % buckets) / static_cast<float>(buckets - 1U);
    return lower + (upper - lower) * fraction;
}

class ComplexGenerator {
public:
    ComplexGenerator(const BlasFillMode& fill, uint32_t seed, bool varyNormalParameters)
        : fill_(fill),
          realRng_(seed == 0 ? 42U : seed),
          imagRng_((seed == 0 ? 42U : seed) + 7919U),
          uniformReal_(-std::abs(fill.val1), std::abs(fill.val2)),
          uniformImag_(-std::abs(fill.val1), std::abs(fill.val2)),
          normalReal_(
              varyNormalParameters ? CtrsvSeedParameter(seed, 17U, -5.0f, 5.0f) : 0.0f,
              varyNormalParameters ? CtrsvSeedParameter(seed, 31U, 0.1f, 2.0f) : 1.0f),
          normalImag_(
              varyNormalParameters ? CtrsvSeedParameter(seed, 47U, -5.0f, 5.0f) : 0.0f,
              varyNormalParameters ? CtrsvSeedParameter(seed, 61U, 0.1f, 2.0f) : 1.0f),
          useNormal_(varyNormalParameters)
    {}

    aclblasComplex Next(size_t index)
    {
        if (fill_.pattern == BlasFillMode::P_ALTER) {
            const float real = (index & 1U) == 0U ? static_cast<float>(index + 1) : -static_cast<float>(index + 1);
            const float imag = (index & 1U) == 0U ? -real : real;
            return {real, imag};
        }
        if (fill_.pattern == BlasFillMode::P_EXTREME) {
            constexpr float values[] = {
                1.0f,
                0.0f,
                -1.0f,
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::min(),
                -std::numeric_limits<float>::max(),
                std::numeric_limits<float>::denorm_min()};
            constexpr size_t count = sizeof(values) / sizeof(values[0]);
            return {values[index % count], values[(index + 3U) % count]};
        }
        if (fill_.method == BlasFillMode::M_VALUE) {
            return {fill_.val1, fill_.val1};
        }
        if (fill_.method == BlasFillMode::M_INDEX) {
            const float value = static_cast<float>(index + 1);
            return {value, -value};
        }
        if (useNormal_) {
            return {normalReal_(realRng_), normalImag_(imagRng_)};
        }
        return {uniformReal_(realRng_), uniformImag_(imagRng_)};
    }

private:
    BlasFillMode fill_;
    std::mt19937 realRng_;
    std::mt19937 imagRng_;
    std::uniform_real_distribution<float> uniformReal_;
    std::uniform_real_distribution<float> uniformImag_;
    std::normal_distribution<float> normalReal_;
    std::normal_distribution<float> normalImag_;
    bool useNormal_;
};

struct PrecisionStats {
    size_t count = 0;
    size_t matched = 0;
    double maxAbsError = 0.0;
    double maxLimitRatio = 0.0;
    float worstOutput = 0.0f;
    float worstGolden = 0.0f;
    bool absoluteLimitPassed = true;
};

inline void AccumulatePrecision(float output, float golden, PrecisionStats& stats)
{
    ++stats.count;
    if (output == golden || (std::isnan(output) && std::isnan(golden))) {
        ++stats.matched;
        return;
    }
    if (!std::isfinite(output) || !std::isfinite(golden)) {
        stats.absoluteLimitPassed = false;
        return;
    }

    constexpr double rtol = 0.0009765625;       // 2^-10
    constexpr double atol = 0.0000152587890625; // 2^-16
    const double difference = std::abs(static_cast<double>(output) - golden);
    stats.maxAbsError = std::max(stats.maxAbsError, difference);
    if (difference <= atol + rtol * std::abs(static_cast<double>(golden))) {
        ++stats.matched;
    }
    const double elementLimit = std::max(1e-2, 32.0 * getUlpsAt(std::abs(static_cast<double>(golden)), 23, -126));
    if (!(elementLimit > 0.0)) {
        stats.absoluteLimitPassed = false;
        return;
    }
    const double limitRatio = difference / elementLimit;
    if (limitRatio > stats.maxLimitRatio) {
        stats.maxLimitRatio = limitRatio;
        stats.worstOutput = output;
        stats.worstGolden = golden;
    }
    if (difference > elementLimit) {
        stats.absoluteLimitPassed = false;
    }
}
