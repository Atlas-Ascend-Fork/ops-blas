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

constexpr uint32_t SSYRK_UPLO_UPPER = 121;
constexpr uint32_t SSYRK_UPLO_LOWER = 122;

constexpr uint32_t SSYRK_TILE_M = 128;
constexpr uint32_t SSYRK_TILE_N = 128;
constexpr uint32_t SSYRK_BASE_K = 64;
constexpr uint32_t SSYRK_SINGLE_K = 8192;
constexpr uint32_t SSYRK_TEMP_ALIGN = 128;

struct SsyrkGemmTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t tempLd;
    uint32_t uploMode;
    uint32_t kBase;
    uint32_t kCount;
    uint32_t enAtomic;
    uint32_t mBlocks;
    uint32_t nBlocks;
};

struct SsyrkCombineTilingData {
    uint32_t n;
    uint32_t ldc;
    uint32_t tempLd;
    uint32_t uploMode;
    uint32_t hasTemp;
    float alpha;
    float beta;
};
