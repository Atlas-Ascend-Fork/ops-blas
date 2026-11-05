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

// No pointers or library-owned device allocations. All pitches are FP32 elements.
struct LtMatmul22Tiling {
    uint32_t m, n, k;
    uint32_t lda, ldb, ldc, ldd;
    uint32_t nCube;
    uint32_t transA, transB;
    uint32_t productLd;
    float alpha, beta;
    uint32_t cubeTile = 64;
    uint32_t cubeTileN = 64;
    uint32_t shortK = 0;
};

constexpr uint32_t LT_MATMUL22_TILE_M = 64;
constexpr uint32_t LT_MATMUL22_TILE_N = 64;
constexpr uint32_t LT_MATMUL22_H1_TILE_M = 128;
constexpr uint32_t LT_MATMUL22_H1_TILE_N = 128;
