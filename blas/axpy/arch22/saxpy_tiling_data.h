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

constexpr uint32_t SAXPY_UB_BYTES = 192U * 1024U;
constexpr uint32_t SAXPY_QUEUE_NUM = 2U;
constexpr uint32_t SAXPY_BUFFER_NUM = 2U;
constexpr uint32_t SAXPY_ELEM_BYTES = 4U;
constexpr uint32_t SAXPY_ELEMS_PER_BLOCK = 32U / SAXPY_ELEM_BYTES;
constexpr uint32_t SAXPY_TILE_BYTES = SAXPY_UB_BYTES / (SAXPY_QUEUE_NUM * SAXPY_BUFFER_NUM);
constexpr uint32_t SAXPY_TILE_ELEMS = SAXPY_TILE_BYTES / SAXPY_ELEM_BYTES;

static_assert(
    SAXPY_TILE_ELEMS * SAXPY_ELEM_BYTES * SAXPY_QUEUE_NUM * SAXPY_BUFFER_NUM == SAXPY_UB_BYTES,
    "saxpy UB budget must cover all queue buffers");

struct SaxpyTilingData {
    uint32_t totalN;
    uint32_t activeBlocks;
    uint32_t perBlockN;
    uint32_t remainder;
    uint32_t tileSize;
    uint32_t mode;
    float alpha;
    int64_t incx;
    int64_t incy;
};
