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

constexpr uint32_t SGEMMV_ALIGN = 8;
constexpr uint32_t SGEMMV_PARTITION_ALIGN = 16;
constexpr uint32_t SGEMMV_T_REDUCE_TILE = 1024;
constexpr uint32_t SGEMMV_T_COL_TILE = 64;
constexpr uint32_t SGEMMV_SMALL_M_LIMIT = 16;
constexpr uint32_t SGEMMV_T_SMALL_M_LIMIT = 32;
constexpr uint32_t SGEMMV_SMALL_N_COL_TILE = 4096;
constexpr uint32_t SGEMMV_FAST_T_MAX_ROWS = 4224;
constexpr uint32_t SGEMMV_CUBE_MAX_BLOCKS = 20;
constexpr uint32_t SGEMMV_CUBE_MAX_COLS = 1024;
constexpr uint32_t SGEMMV_CUBE_BASE_COLS = 512;
constexpr uint32_t SGEMMV_CUBE_ODD_COLS = 1032;
constexpr uint32_t SGEMMV_CUBE_TARGET_COLS = 256;
constexpr uint32_t SGEMMV_CUBE_SPLIT_MAX = 4;
constexpr uint32_t SGEMMV_CUBE_SPLIT_HARD_MAX = 8;
constexpr uint32_t SGEMMV_CUBE_SPLIT_REDUCE_TILE = 512;

enum class SgemvPath : uint32_t {
    NORMAL_N = 0,
    SMALL_M_N = 1,
    NORMAL_T = 2,
    SMALL_M_T = 3,
};

struct SgemvTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t outPerBlock;
    uint32_t numBlocks;
    uint32_t path;
    float alpha;
    float beta;
    int64_t incx;
    int64_t incy;
    uint32_t cubeSplitK;
    uint32_t cubeKPerSlice;
};
