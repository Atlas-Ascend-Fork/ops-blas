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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cann_ops_blas.h"
#include "fill.h"

// Identical input construction for official GTest, accuracy and profiling tools.
inline std::vector<aclblasComplex> MakeCtbmvComplexArray(size_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count == 0) {
        return {};
    }
    std::vector<float> real = makeBlasArray(static_cast<int64_t>(count), fill, seed);
    std::vector<float> imag = makeBlasArray(static_cast<int64_t>(count), fill, seed + 1000U);
    std::vector<aclblasComplex> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = {real[i], imag[i]};
    }
    return values;
}
