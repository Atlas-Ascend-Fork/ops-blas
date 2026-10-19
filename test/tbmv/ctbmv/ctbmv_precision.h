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

// FLOAT32 component criteria from the Ctbmv task book. Real and imaginary
// components are evaluated independently; matching nonfinite values have
// zero error, and a nonfinite mismatch never disappears from the error max.
struct CtbmvComponentMetrics {
    static constexpr double ATOL = 1.0 / 65536.0;
    static constexpr double RTOL = 1.0 / 1024.0;
    static constexpr double REQUIRED_RATIO = 0.99;
    static constexpr double MAX_ABS = 1.0e-2;
    static constexpr uint64_t MAX_ULP = 32U;
    size_t matched = 0U;
    size_t total = 0U;
    double maxAbs = 0.0;
    uint64_t maxUlp = 0U;

    static uint64_t OrderedBits(float value)
    {
        uint32_t bits = 0U;
        static_assert(sizeof(bits) == sizeof(value), "FLOAT32 storage must have 32 bits");
        const auto* source = reinterpret_cast<const unsigned char*>(&value);
        auto* destination = reinterpret_cast<unsigned char*>(&bits);
        std::copy_n(source, sizeof(bits), destination);
        return (bits & 0x80000000U) ? 0x80000000ULL - (bits & 0x7FFFFFFFU) : 0x80000000ULL + bits;
    }

    bool Observe(float actual, float expected)
    {
        ++total;
        bool match;
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            match = (std::isnan(actual) && std::isnan(expected)) || actual == expected;
            if (!match) {
                maxAbs = std::numeric_limits<double>::infinity();
                maxUlp = std::numeric_limits<uint64_t>::max();
            }
        } else {
            const double error = std::abs(static_cast<double>(actual) - expected);
            const uint64_t a = OrderedBits(actual);
            const uint64_t e = OrderedBits(expected);
            maxAbs = std::max(maxAbs, error);
            maxUlp = std::max(maxUlp, a > e ? a - e : e - a);
            match = error <= ATOL + RTOL * std::abs(static_cast<double>(expected));
        }
        matched += static_cast<size_t>(match);
        return match;
    }

    double Ratio() const { return total == 0U ? 1.0 : static_cast<double>(matched) / total; }

    bool Pass() const { return Ratio() >= REQUIRED_RATIO && (maxAbs <= MAX_ABS || maxUlp <= MAX_ULP); }
};
