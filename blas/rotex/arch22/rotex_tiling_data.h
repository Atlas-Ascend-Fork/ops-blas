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

inline constexpr uint32_t ROTEX_ARCH22_TILE = 4096U;
inline constexpr uint32_t ROTEX_ARCH22_WIN_MAX = 4096U;

struct RotExTilingData {
    uint32_t totalN;
    uint32_t perCoreN;
    uint32_t remainder;
    uint32_t numBlocks;
    int64_t incx;
    int64_t incy;
    int64_t startOffsetX;
    int64_t startOffsetY;
    float cReal;
    float sReal;
    uint32_t xType;
};
