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
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "cann_ops_blas.h"
#include "fill.h"

inline size_t CdotcStorageCount(int n, int inc)
{
    if (n <= 0) {
        return 1;
    }
    const int64_t absInc = inc >= 0 ? static_cast<int64_t>(inc) : -static_cast<int64_t>(inc);
    return static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1;
}

inline std::vector<float> MakeCdotcNormalStrided(int n, int inc, uint32_t seed)
{
    std::vector<float> values(CdotcStorageCount(n, inc), 0.0f);
    if (n <= 0) {
        return values;
    }

    std::mt19937 rng(seed ? seed : 42U);
    std::uniform_real_distribution<float> meanDistribution(-5.0f, 5.0f);
    std::uniform_real_distribution<float> sigmaDistribution(0.1f, 2.0f);
    std::normal_distribution<float> valueDistribution(meanDistribution(rng), sigmaDistribution(rng));
    const int64_t absInc = inc >= 0 ? static_cast<int64_t>(inc) : -static_cast<int64_t>(inc);
    for (int i = 0; i < n; ++i) {
        const int64_t index = inc > 0 ? static_cast<int64_t>(i) * inc : static_cast<int64_t>(n - 1 - i) * absInc;
        values[static_cast<size_t>(index)] = valueDistribution(rng);
    }
    return values;
}

inline std::vector<float> MakeCdotcUniformStrided(int n, int inc, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<float> values(CdotcStorageCount(n, inc), 0.0f);
    if (n <= 0) {
        return values;
    }

    std::mt19937 rng(seed ? seed : 42U);
    const float lower = std::min(fill.val1, fill.val2);
    const float upper = std::max(fill.val1, fill.val2);
    std::uniform_real_distribution<float> valueDistribution(lower, upper);
    const int64_t absInc = inc >= 0 ? static_cast<int64_t>(inc) : -static_cast<int64_t>(inc);
    for (int i = 0; i < n; ++i) {
        const int64_t index = inc > 0 ? static_cast<int64_t>(i) * inc : static_cast<int64_t>(n - 1 - i) * absInc;
        values[static_cast<size_t>(index)] = valueDistribution(rng);
    }
    return values;
}

inline std::vector<aclblasComplex> MakeCdotcComplexStrided(
    int n, int inc, const BlasFillMode& fill, const std::string& distribution, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }

    const bool useNormal = fill.method == BlasFillMode::M_RANDOM && distribution == "normal";
    const bool useUniform = fill.method == BlasFillMode::M_RANDOM && distribution == "uniform";
    std::vector<float> realPart = useNormal  ? MakeCdotcNormalStrided(n, inc, seed) :
                                  useUniform ? MakeCdotcUniformStrided(n, inc, fill, seed) :
                                               makeBlasStrided(n, inc, fill, seed);
    std::vector<float> imagPart = useNormal  ? MakeCdotcNormalStrided(n, inc, seed + 1000U) :
                                  useUniform ? MakeCdotcUniformStrided(n, inc, fill, seed + 1000U) :
                                               makeBlasStrided(n, inc, fill, seed + 1000U);
    const size_t size = std::min(realPart.size(), imagPart.size());
    std::vector<aclblasComplex> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return result;
}
