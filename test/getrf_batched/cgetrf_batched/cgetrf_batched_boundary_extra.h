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
#include <random>
#include <vector>

template <class Complex, class Param>
std::vector<Complex> CgetrfMakeBoundaryInput(const Param& p, int batch)
{
    std::vector<Complex> result(static_cast<size_t>(p.lda) * p.n, {-1234.5f, 987.25f});
    std::mt19937 rng(p.randomSeed + batch);
    std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
    for (int j = 0; j < p.n; ++j)
        for (int i = 0; i < p.n; ++i)
            result[i + static_cast<size_t>(j) * p.lda] = {dist(rng) + (i == j ? 4.0f : 0.0f), dist(rng)};
    if (p.pivot && p.n > 1)
        for (int j = 0; j < p.n; ++j)
            std::swap(result[j * p.lda], result[p.n - 1 + j * p.lda]);
    return result;
}
