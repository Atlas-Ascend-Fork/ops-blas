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
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace complex_triangular_test {

class ScalarFill {
public:
    ScalarFill(const std::string& name, uint32_t seed) : engine_(seed)
    {
        if (name == "RANDOM_NORM_5_5") {
            kind_ = UNIFORM;
        } else if (name == "GAUSS_NORM_5_2") {
            kind_ = GAUSS;
            std::uniform_real_distribution<float> mean(-5.0F, 5.0F);
            std::uniform_real_distribution<float> deviation(0.1F, 2.0F);
            const float mu = mean(engine_);
            const float sigma = deviation(engine_);
            normal_.param(std::normal_distribution<float>::param_type(mu, sigma));
        } else if (name == "VALUE_NORM_0") {
            kind_ = ZERO;
        } else if (name == "VALUE_NORM_N999") {
            kind_ = SENTINEL;
        } else if (name == "INDEX") {
            kind_ = INDEX;
        } else if (name == "RANDOM_ALTER") {
            kind_ = ALTER;
        } else if (name == "RANDOM_EXTREME") {
            kind_ = EXTREME;
        } else if (name == "VALUE_NORM_INF") {
            kind_ = POSITIVE_INFINITY;
        } else if (name == "VALUE_NORM_NAN") {
            kind_ = QUIET_NAN;
        } else {
            throw std::runtime_error("unsupported fill: " + name);
        }
    }

    float At(uint64_t index)
    {
        switch (kind_) {
            case UNIFORM:
                return uniform_(engine_);
            case GAUSS:
                return std::min(5.0F, std::max(-5.0F, normal_(engine_)));
            case ZERO:
                return 0.0F;
            case SENTINEL:
                return -999.0F;
            case INDEX:
                return static_cast<float>(index + 1U);
            case ALTER:
                return index % 2U == 0U ? static_cast<float>(index + 1U) : -static_cast<float>(index + 1U);
            case POSITIVE_INFINITY:
                return std::numeric_limits<float>::infinity();
            case QUIET_NAN:
                return std::numeric_limits<float>::quiet_NaN();
            case EXTREME: {
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
            default:
                throw std::runtime_error("invalid scalar fill kind");
        }
    }

private:
    enum Kind { UNIFORM, GAUSS, ZERO, SENTINEL, INDEX, ALTER, EXTREME, POSITIVE_INFINITY, QUIET_NAN };
    Kind kind_ = ZERO;
    std::mt19937 engine_;
    std::uniform_real_distribution<float> uniform_{-5.0F, 5.0F};
    std::normal_distribution<float> normal_{0.0F, 1.0F};
};

} // namespace complex_triangular_test
