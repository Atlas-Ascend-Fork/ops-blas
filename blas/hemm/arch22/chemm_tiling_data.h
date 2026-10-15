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

constexpr uint32_t CHEMM_MICRO_TILE = 16U;
constexpr uint32_t CHEMM_AIV_CACHE_FEEDS = 32U;
constexpr uint32_t CHEMM_AIV_LEFT_MAX_FEEDS = 8U;
constexpr uint32_t CHEMM_FLAG_RIGHT = 1U;
constexpr uint32_t CHEMM_FLAG_LOWER = 2U;
constexpr uint32_t CHEMM_FLAG_FULL = 4U;
constexpr uint32_t CHEMM_K_FEED = 32U;
constexpr uint32_t CHEMM_INPUT_RING = 3U;
constexpr uint32_t CHEMM_MAX_PANEL = 128U;
constexpr uint32_t CHEMM_MAX_ROWS = 256U;
constexpr uint32_t CHEMM_MAX_P_ELEMENTS = CHEMM_MAX_PANEL * 2U * CHEMM_K_FEED;
constexpr uint32_t CHEMM_L0_ELEMENTS = 65536U / sizeof(float);
constexpr uint32_t CHEMM_L1_ELEMENTS = (524288U - 256U) / sizeof(float);

enum class ChemmVectorMode : uint32_t { LOCAL, CACHED, ORDERED };

struct ChemmAivTilingData {
    ChemmVectorMode mode;
    uint32_t kernelFlags;
    int32_t m;
    int32_t n;
    int32_t dimA;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t blockDim;
    uint32_t rowTile;
    uint32_t reductionBanks;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
};

// P=[output columns,Re(K)|Im(K)], Q=[Re/Im output rows,expanded K].
// Workspace offsets and local spans are FP32 elements, not bytes.
struct ChemmMixedTilingData {
    uint32_t kernelFlags;
    int32_t m;
    int32_t n;
    int32_t dimA;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    uint32_t panelM;
    uint32_t rowTile;
    uint32_t expandedK;
    uint32_t workTiles;
    uint32_t rawK;
    uint32_t rowOwned;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t blockDim;
    uint32_t kFeeds;
    uint32_t tilesPerCore;
    uint32_t extraTiles;
    uint32_t pElements;
    uint32_t qElements;
    uint32_t inputSlotStride;
    uint32_t pCacheOffset;
    uint32_t inputCoreStride;
    uint32_t residentP;
    uint32_t l0BankMask;
};

void chemm_mixed_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
void chemm_aiv_kernel_do(uint8_t* a, uint8_t* b, uint8_t* c, const ChemmAivTilingData& tiling, void* stream);
