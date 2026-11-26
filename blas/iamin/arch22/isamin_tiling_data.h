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

// A2/A3 ReduceMin processes at most 255 repeats of 64 FP32 lanes.
constexpr uint32_t ISAMIN_ARCH22_TILE_SIZE = 16320;
constexpr uint32_t ISAMIN_ARCH22_ALIGN_ELEMENTS = 8U;

struct IsaminTilingData {
    uint32_t totalN;
    int32_t incx;
    uint32_t useCoreNum;
    uint32_t perCoreN;
    uint32_t tileSize;
};
