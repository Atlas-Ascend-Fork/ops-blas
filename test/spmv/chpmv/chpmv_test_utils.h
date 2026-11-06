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
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "chpmv_param.h"
#include "verify.h"

namespace chpmv_test {
inline std::vector<Complex> Generate(size_t count, const std::string& fill, uint32_t seed)
{
    if (fill == "NULLPTR" || count == 0)
        return {};
    std::vector<Complex> data(count);
    std::mt19937 rng(seed);
    if (fill == "GAUSS_NORM_5_2") {
        std::uniform_real_distribution<float> mu(-5, 5), sd(0.1f, 2);
        const float mr = mu(rng), sr = sd(rng), mi = mu(rng), si = sd(rng);
        std::normal_distribution<float> real(mr, sr), imag(mi, si);
        std::cout << "DISTRIBUTION seed=" << seed << " mu_re=" << mr << " sigma_re=" << sr << " mu_im=" << mi
                  << " sigma_im=" << si << '\n';
        for (auto& z : data)
            z = {real(rng), imag(rng)};
    } else {
        const std::set<std::string> supported{"RANDOM_NORM_5_5", "VALUE_NORM_0",   "RANDOM_ALTER",
                                              "RANDOM_EXTREME",  "VALUE_NORM_INF", "VALUE_NORM_NAN"};
        if (!supported.count(fill))
            throw std::runtime_error("unsupported fill: " + fill);
        auto generator = createGenerator(BlasFillMode(fill), rng);
        for (size_t i = 0; i < count; ++i)
            data[i] = {generator->at(2 * i), generator->at(2 * i + 1)};
    }
    return data;
}

struct ComponentResult {
    size_t finite = 0, special = 0, specialMatched = 0;
    bool passed = false;
};

inline bool VerifyFiniteComponent(
    const std::vector<float>& actual, const std::vector<float>& golden, const std::string& name)
{
    if (actual.empty()) {
        return true;
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, golden.data(), golden.size());
    const bool frameworkPassed = Verifier::verifyVector(actual.data(), golden.data(), actual.size(), 1, config, name);
    // 任务书另有明确容差，同时检查，不能以接入框架为由放宽原有精度要求。
    config.mixedAtol = std::ldexp(1.0, -13);
    config.mixedRtol = std::ldexp(1.0, -13);
    const bool taskPassed =
        Verifier::verifyVector(actual.data(), golden.data(), actual.size(), 1, config, name + "_task");
    return frameworkPassed && taskPassed;
}

inline ComponentResult VerifyComponent(
    const std::vector<Complex>& actual, const std::vector<Complex>& golden, int n, int inc, bool imaginary,
    const std::string& name = "ChpmvComparator")
{
    if (n <= 0 || inc == 0 || actual.size() < Count(n, inc) || golden.size() < Count(n, inc)) {
        throw std::runtime_error("invalid component verification range");
    }
    ComponentResult result;
    std::vector<float> finiteActual, finiteGolden;
    for (int i = 0; i < n; ++i) {
        const size_t k = Offset(i, n, inc);
        const float a = imaginary ? actual[k].imag : actual[k].real;
        const float g = imaginary ? golden[k].imag : golden[k].real;
        if (std::isfinite(a) && std::isfinite(g)) {
            finiteActual.push_back(a);
            finiteGolden.push_back(g);
        } else {
            // 任务书规定特殊值单独判定，不计入有限值匹配率。
            ++result.special;
            if (!std::isfinite(a) && !std::isfinite(g) &&
                (std::isnan(a) || std::isnan(g) || std::signbit(a) == std::signbit(g))) {
                ++result.specialMatched;
            }
        }
    }
    result.finite = finiteActual.size();
    const bool finitePassed = VerifyFiniteComponent(finiteActual, finiteGolden, name);
    result.passed = finitePassed && result.special == result.specialMatched;
    return result;
}
} // namespace chpmv_test
