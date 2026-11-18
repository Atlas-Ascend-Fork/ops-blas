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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "chbmv_reference.h"

namespace chbmv_test_data {
inline int64_t Index(int i, int n, int inc)
{
    const int64_t stride = inc;
    return inc > 0 ? static_cast<int64_t>(i) * stride : (static_cast<int64_t>(i) + 1 - n) * stride;
}

inline size_t Length(int n, int inc)
{
    const int64_t stride = inc;
    return n > 0 ? 1 + static_cast<size_t>(n - 1) * static_cast<size_t>(stride < 0 ? -stride : stride) : 0;
}

inline std::vector<aclblasComplex> Fill(size_t count, const std::string& pattern, uint32_t seed, bool normal)
{
    if (pattern == "NULLPTR") {
        return {};
    }
    std::vector<aclblasComplex> result(count);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
    std::uniform_real_distribution<float> sigma(0.1f, 2.0f);
    const float mean = uniform(rng);
    std::normal_distribution<float> gaussian(mean, sigma(rng));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float extreme[] = {
        1,
        0,
        -1,
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::min(),
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::denorm_min()};
    auto component = [&](size_t i) {
        if (pattern == "VALUE_NORM_0") {
            return 0.0f;
        }
        if (pattern == "VALUE_NORM_INF") {
            return inf;
        }
        if (pattern == "VALUE_NORM_NAN") {
            return nan;
        }
        if (pattern == "RANDOM_EXTREME") {
            return extreme[i % 7];
        }
        if (pattern == "RANDOM_ALTER") {
            return (i % 2 == 0 ? 1.0f : -1.0f) * static_cast<float>(i + 1);
        }
        if (pattern != "RANDOM_NORM_5_5" && pattern != "GAUSS_NORM_5_2") {
            throw std::runtime_error("Unknown fill: " + pattern);
        }
        return normal || pattern == "GAUSS_NORM_5_2" ? gaussian(rng) : uniform(rng);
    };
    for (size_t i = 0; i < count; ++i) {
        result[i] = {component(i), component(i + count)};
    }
    return result;
}

struct Data {
    std::vector<aclblasComplex> a;
    std::vector<aclblasComplex> x;
    std::vector<aclblasComplex> y;
    explicit Data(const ChbmvParam& p, bool poisonUnused = false)
    {
        const bool normal = (p.randomSeed & 1U) == 0;
        // Invalid spans must reach API validation without allocating gigabytes.
        // Preserve explicitly null arguments; all other buffers are one element.
        if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
            a = Fill(1, p.aFill, p.randomSeed, normal);
            x = Fill(1, p.xFill, p.randomSeed + 1, normal);
            y = Fill(1, p.yFill, p.randomSeed + 2, normal);
            return;
        }
        a = Fill(p.n > 0 && p.lda > 0 ? static_cast<size_t>(p.n) * p.lda : 0, p.aFill, p.randomSeed, normal);
        x = Fill(Length(p.n, p.incx), p.xFill, p.randomSeed + 1, normal);
        y = Fill(Length(p.n, p.incy), p.yFill, p.randomSeed + 2, normal);
        if (p.n <= 0 || p.k < 0 || static_cast<int64_t>(p.lda) < static_cast<int64_t>(p.k) + 1 || a.empty()) {
            return;
        }
        const float unused = poisonUnused ? std::numeric_limits<float>::quiet_NaN() : 0.0f;
        for (int c = 0; c < p.n; ++c) {
            for (int b = 0; b < p.lda; ++b) {
                const int64_t row =
                    p.uplo == ACLBLAS_UPPER ? static_cast<int64_t>(c) + b - p.k : static_cast<int64_t>(c) + b;
                auto& v = a[static_cast<size_t>(c) * p.lda + b];
                if (row < 0 || row >= p.n || b > p.k) {
                    v = {unused, unused};
                } else if (row == c) {
                    v.imag = unused;
                }
            }
        }
    }
};

inline bool ComponentMatches(
    const ChbmvParam& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden,
    bool imag)
{
    ChbmvReference::Stats stats;
    for (int i = 0; i < p.n; ++i) {
        const size_t idx = static_cast<size_t>(Index(i, p.n, p.incy));
        stats.Add(imag ? actual[idx].imag : actual[idx].real, imag ? golden[idx].imag : golden[idx].real);
    }
    std::cout << p.caseName << (imag ? " imag" : " real") << " max_abs=" << stats.maxError
              << " special=" << stats.specialMatched << "/" << stats.special << '\n';
    return stats.Pass();
}

inline bool GapsUnchanged(
    const ChbmvParam& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& before)
{
    const int64_t stride = p.incy < 0 ? -static_cast<int64_t>(p.incy) : p.incy;
    if (stride == 0) {
        return true;
    }
    for (size_t i = 0; i < before.size(); ++i) {
        if (i % stride != 0 && std::memcmp(&actual[i], &before[i], sizeof(aclblasComplex)) != 0) {
            return false;
        }
    }
    return true;
}
} // namespace chbmv_test_data
