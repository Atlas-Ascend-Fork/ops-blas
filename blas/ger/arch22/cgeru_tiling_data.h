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

#ifndef CGERU_TILING_H
#define CGERU_TILING_H

#include <cstdint>

// 行方向分块上限（单位：复数）。列主序矩阵的一列在 UB 内是 2*rowTile 个 float，
// 需要保持 32B 对齐（rowTile 取 32 的整数倍），且总 UB 占用不超过单核预算。
constexpr uint32_t CGERU_ROW_TILE_MAX = 2048U;
// 复数按 32 个一组对齐：批处理指令以 32 个复数（64 个 float）为处理粒度。
constexpr uint32_t CGERU_ALIGN_COMPLEX = 32U;
// UB 内单块容量（单位：复数），比 rowTile 多留一组对齐余量。
constexpr uint32_t CGERU_TILE_COMPLEX = CGERU_ROW_TILE_MAX + CGERU_ALIGN_COMPLEX;
constexpr uint32_t CGERU_TILE_FLOATS = 2U * CGERU_TILE_COMPLEX;
// UB 内两张模式表的定容（按 float 个数计，各一份）：表 0 是成对交换的**字节**偏移
// 4*(k^1)（按 uint32 存放），表 1 是交错正负号模式 (-1, +1, -1, +1, ...)（float）。
// 两张表都由 kernel 按算术级数在 UB 内现算，不占 handle 常驻缓冲，也不读 GM。
constexpr uint32_t CGERU_PATTERN_FLOATS = CGERU_TILE_FLOATS;

struct CgeruTilingData {
    uint64_t A;
    uint64_t x;
    uint64_t y;
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    uint32_t colsPerCore;
    uint32_t rowTile;
    int32_t incx;
    int32_t incy;
    float alphaReal;
    float alphaImag;
};

#endif // CGERU_TILING_H
