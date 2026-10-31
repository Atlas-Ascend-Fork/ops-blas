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

#include <cstdint>

struct SswapTilingData {
    uint32_t totalN;    // total element count
    uint32_t perCoreN;  // elements per core, aligned down to ELEMENTS_PER_BLOCK
    uint32_t remainder; // tail elements assigned to the last core
    uint32_t tileSize;  // UB tile size, aligned to ELEMENTS_PER_BLOCK
    int32_t incx;
    int32_t incy;
    int64_t kx;
    int64_t ky;
    uint64_t xLength;
    uint64_t yLength;
};

// elements per 32-byte block for FP32 (= 8)
constexpr uint32_t ELEMENTS_PER_BLOCK = 32 / sizeof(float);
