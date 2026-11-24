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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cdgmm_test_utils.h"

namespace cdgmm_arch35_test {

// Same fill vocabulary and ranges as test/frame/fill.h, generated directly
// into complex buffers to avoid two full-size temporary float matrices.
class CdgmmData {
public:
    CdgmmData(const std::string& method, uint32_t seed) : method_(method), engine_(seed), normal_(0.0F, 1.0F)
    {
        if (method_ == "GAUSS_NORM_5_2") {
            std::uniform_real_distribution<float> mean(-5.0F, 5.0F);
            std::uniform_real_distribution<float> deviation(0.1F, 2.0F);
            const float mu = mean(engine_);
            const float sigma = deviation(engine_);
            normal_.param(std::normal_distribution<float>::param_type(mu, sigma));
        } else if (
            method_ != "RANDOM_NORM_5_5" && method_ != "VALUE_NORM_0" && method_ != "RANDOM_ALTER" &&
            method_ != "RANDOM_EXTREME" && method_ != "VALUE_NORM_INF" && method_ != "VALUE_NORM_NAN" &&
            method_ != "INDEX") {
            throw std::runtime_error("unsupported fill: " + method_);
        }
    }

    float At(size_t index)
    {
        if (method_ == "RANDOM_NORM_5_5") {
            return uniform_(engine_);
        }
        if (method_ == "GAUSS_NORM_5_2") {
            return std::min(5.0F, std::max(-5.0F, normal_(engine_)));
        }
        if (method_ == "VALUE_NORM_0") {
            return 0.0F;
        }
        if (method_ == "VALUE_NORM_INF") {
            return std::numeric_limits<float>::infinity();
        }
        if (method_ == "VALUE_NORM_NAN") {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (method_ == "RANDOM_ALTER") {
            return index % 2U == 0U ? static_cast<float>(index + 1U) : -static_cast<float>(index + 1U);
        }
        if (method_ == "RANDOM_EXTREME") {
            static constexpr float values[] = {
                1.0F,
                0.0F,
                -1.0F,
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::min(),
                -std::numeric_limits<float>::max(),
                std::numeric_limits<float>::denorm_min()};
            return values[index % 7U];
        }
        return static_cast<float>(index + 1U);
    }

private:
    std::string method_;
    std::mt19937 engine_;
    std::uniform_real_distribution<float> uniform_{-5.0F, 5.0F};
    std::normal_distribution<float> normal_;
};

inline std::vector<aclblasComplex> MakeMatrix(int m, int n, int ld, const std::string& fill, uint32_t seed)
{
    if (m < 0 || n < 0 || ld < m) {
        throw std::invalid_argument("Invalid matrix dimensions or leading dimension");
    }
    std::vector<aclblasComplex> values(static_cast<size_t>(ld) * n, SENTINEL);
    CdgmmData real(fill, seed);
    CdgmmData imag(fill, seed + 1000U);
    size_t sequence = 0;
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < m; ++row) {
            values[static_cast<size_t>(column) * ld + row] = {real.At(sequence), imag.At(sequence)};
            ++sequence;
        }
    }
    return values;
}

inline std::vector<aclblasComplex> MakeVector(int count, int incx, const std::string& fill, uint32_t seed)
{
    const uint64_t step = AbsIncrement(incx);
    std::vector<aclblasComplex> values((static_cast<uint64_t>(count) - 1U) * step + 1U, SENTINEL);
    CdgmmData real(fill, seed);
    CdgmmData imag(fill, seed + 1000U);
    for (int index = 0; index < count; ++index) {
        const uint64_t offset =
            incx < 0 ? (static_cast<uint64_t>(count) - 1U - index) * step : static_cast<uint64_t>(index) * step;
        values[offset] = {real.At(static_cast<size_t>(index)), imag.At(static_cast<size_t>(index))};
    }
    return values;
}

} // namespace cdgmm_arch35_test
