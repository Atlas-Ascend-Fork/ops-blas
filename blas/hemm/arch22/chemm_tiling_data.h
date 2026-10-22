/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chemm (arch22) tiling parameter structures shared by host and device.

#pragma once

#include <cstdint>

// Launch record passed by value through the AIV-pack -> AIC-Cube -> AIV-combine
// pipeline. Workspace offsets are expressed in FP32 elements.
struct ChemmMixedTilingData {
    int32_t m;
    int32_t n;
    int32_t dimA;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    uint32_t tileRowBase;
    uint32_t tileColBase;
    uint32_t workspaceCores;
    uint32_t blockDim;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t kFeeds;
    uint32_t cachePanels;
    uint32_t cacheFeeds;
    uint32_t cacheSlots;
    uint64_t inputSlotStride;
    uint64_t inputCoreStride;
    uint64_t productOffset;
    uint64_t productSlotStride;
    uint64_t productCoreStride;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    // Device MIX geometry. Zero selects the regular 64-row pipeline.
    uint32_t rowTile;
    // RIGHT narrow algorithm selected from work shape and local-memory fit.
    uint32_t narrowMode;
    // Uniform launch controls. These select one algorithm at kernel entry;
    // they are not consulted inside the element-wise vector loops.
    uint32_t kernelFlags;
    uint32_t panelM;
    uint32_t tileM;
};

constexpr uint32_t CHEMM_FLAG_RIGHT = 1U << 0U;
constexpr uint32_t CHEMM_FLAG_LOWER = 1U << 1U;
constexpr uint32_t CHEMM_FLAG_FULL = 1U << 2U;
constexpr uint32_t CHEMM_FLAG_SIMPLE = 1U << 3U;
constexpr uint32_t CHEMM_FLAG_DIRECT = 1U << 5U;
constexpr uint32_t CHEMM_FLAG_ONE_PANEL = 1U << 6U;
constexpr uint32_t CHEMM_FLAG_KEEP_DUMMY = 1U << 7U;
constexpr uint32_t CHEMM_FLAG_RESIDENT = 1U << 8U;

constexpr uint32_t CHEMM_NARROW_STREAM = 0U;
constexpr uint32_t CHEMM_NARROW_RESIDENT = 1U;
constexpr uint32_t CHEMM_NARROW_STAGED = 2U;

struct ChemmAivTilingData {
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
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    // Appended to preserve the offsets of the existing launch record fields.
    uint32_t tileBase;
    uint32_t kernelFlags;
};

// One expanded real MMAD computes a 64x128 complex output tile:
// P[128,64] * Q[64,128] = [Re(C)^T | Im(C)^T].
constexpr uint32_t CHEMM_MICRO_TILE = 16;
// RunAivGrid keeps one 16x16 stationary operand per K feed in each yCache
// plane.  Keep the host eligibility check and device allocation in lockstep.
constexpr uint32_t CHEMM_AIV_CACHE_FEEDS = 32;
// LEFT Grid stages validCols rows with a 32-float stride in its 4096-float
// broadcast buffer before converting them into the larger stationary cache.
constexpr uint32_t CHEMM_AIV_LEFT_MAX_FEEDS = 8;
constexpr uint32_t CHEMM_TILE_M = 64;
constexpr uint32_t CHEMM_MAX_MIX_TILE_M = 128;
constexpr uint32_t CHEMM_TILE_N = 128;
// One MIX work tile has two 64x128 panels. Q is shared by both panels while
// L0A/L0B/L0C use two equal banks to overlap MTE1, Cube and FixPipe.
constexpr uint32_t CHEMM_WORK_N = 2U * CHEMM_TILE_N;
constexpr uint32_t CHEMM_K_FEED = 32;
constexpr uint32_t CHEMM_EXP_M = 128;
constexpr uint32_t CHEMM_EXP_K = 64;
constexpr uint32_t CHEMM_EXP_N = 128;
// RIGHT panels with 17..24 complex K values use a direct-B narrow pipeline.
// P is split into two 12-column panels.  Each MMAD produces one 64 KiB
// [X;Y] product in an L0C ping-pong bank for the paired AIV consumers.
constexpr uint32_t CHEMM_NARROW_K = 24;
constexpr uint32_t CHEMM_NARROW_EXP_K = 2U * CHEMM_NARROW_K;
constexpr uint32_t CHEMM_NARROW_TILE_M = 128;
constexpr uint32_t CHEMM_NARROW_PANEL_N = 12;
constexpr uint32_t CHEMM_NARROW_PANEL_M = 32;
constexpr uint32_t CHEMM_NARROW_PRODUCT_N = 2U * CHEMM_NARROW_TILE_M;
constexpr uint32_t CHEMM_NARROW_PRODUCT_ELEMENTS = CHEMM_NARROW_PANEL_M * CHEMM_NARROW_PRODUCT_N;
// One product slot contains both 12-column panels of one row tile. Keeping
// the two NZ2ND images adjacent lets each AIV fetch X/Y for both panels with
// one four-burst GM-to-UB transfer.
constexpr uint32_t CHEMM_NARROW_PRODUCT_TILE_ELEMENTS = 2U * CHEMM_NARROW_PRODUCT_ELEMENTS;
constexpr uint32_t CHEMM_NARROW_PRODUCT_RING = 3;
constexpr uint32_t CHEMM_NARROW_L1_K = 32;
constexpr uint32_t CHEMM_NARROW_L1_BANK_ELEMENTS = CHEMM_NARROW_L1_K * CHEMM_NARROW_PRODUCT_N;
constexpr uint32_t CHEMM_NARROW_L0B_ELEMENTS = CHEMM_NARROW_L1_K * CHEMM_NARROW_PRODUCT_N;
constexpr uint32_t CHEMM_AIC_L1_BYTES = 512U * 1024U - 256U;
constexpr uint32_t CHEMM_NARROW_P_L1_ELEMENTS = 2U * CHEMM_NARROW_PANEL_M * CHEMM_NARROW_K;
constexpr uint32_t CHEMM_NARROW_RESIDENT_Q_MAX_ELEMENTS =
    CHEMM_AIC_L1_BYTES / sizeof(float) - CHEMM_NARROW_P_L1_ELEMENTS;

// The staged narrow path handles 32 rows first, then 89..104 rows. The
// rotated tail keeps four physical K8 groups, including the dummy group that
// MMAD does not consume. These bounds follow the local-memory layout; the host
// derives eligibility from each core's row range.
constexpr uint32_t CHEMM_NARROW_STAGE0_N = 64U;
constexpr uint32_t CHEMM_NARROW_STAGE1_MIN_N = 192U;
constexpr uint32_t CHEMM_NARROW_STAGE1_MAX_N = 208U;
constexpr uint32_t CHEMM_NARROW_STAGE_ROW_ALIGN = 8U;
constexpr uint32_t CHEMM_NARROW_STAGE0_ROWS = CHEMM_NARROW_STAGE0_N / 2U;
constexpr uint32_t CHEMM_NARROW_STAGE_MIN_ROWS =
    CHEMM_NARROW_STAGE0_ROWS + (CHEMM_NARROW_STAGE1_MIN_N - CHEMM_MICRO_TILE + 2U) / 2U;
constexpr uint32_t CHEMM_NARROW_STAGE_MAX_ROWS = CHEMM_NARROW_STAGE0_ROWS + CHEMM_NARROW_STAGE1_MAX_N / 2U;
constexpr uint32_t CHEMM_NARROW_STAGE_N = CHEMM_NARROW_STAGE0_N + CHEMM_NARROW_STAGE1_MAX_N;
constexpr uint32_t CHEMM_NARROW_STAGE_P_ELEMENTS = CHEMM_NARROW_K * CHEMM_NARROW_EXP_K;
constexpr uint32_t CHEMM_NARROW_STAGE_TAIL_ELEMENTS = 32U * CHEMM_NARROW_STAGE1_MAX_N;
constexpr uint32_t CHEMM_NARROW_STAGE_Q_ELEMENTS = CHEMM_NARROW_EXP_K * CHEMM_NARROW_STAGE_N;
constexpr uint32_t CHEMM_NARROW_STAGE_C_ELEMENTS = CHEMM_NARROW_PANEL_M * CHEMM_NARROW_STAGE_N;
constexpr uint32_t CHEMM_INPUT_MATRIX_ELEMENTS = CHEMM_EXP_M * CHEMM_EXP_K;
constexpr uint32_t CHEMM_PANEL_L1_ELEMENTS = 12U * CHEMM_INPUT_MATRIX_ELEMENTS;
constexpr uint32_t CHEMM_PRODUCT_ELEMENTS = CHEMM_EXP_M * CHEMM_EXP_N;
// The general-coefficient path keeps the four real products separate.  Its
// P operand is [128, 32] real followed by [128, 32] imag; Q is [64, 32]
// real followed by [64, 32] imag.  One product slot stores RR/II/RI/IR.
constexpr uint32_t CHEMM_PLANE_N = CHEMM_TILE_M;
constexpr uint32_t CHEMM_PLANE_P_ELEMENTS = CHEMM_EXP_M * CHEMM_K_FEED;
constexpr uint32_t CHEMM_PLANE_Q_ELEMENTS = CHEMM_PLANE_N * CHEMM_K_FEED;
constexpr uint32_t CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS = CHEMM_EXP_M * CHEMM_PLANE_N;
constexpr uint32_t CHEMM_PLANAR_PRODUCT_ELEMENTS = 4U * CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS;
constexpr uint32_t CHEMM_INPUT_RING = 3;
constexpr uint32_t CHEMM_PRODUCTS_PER_GROUP = 2;
constexpr uint32_t CHEMM_PRODUCT_GROUPS = 2;
constexpr uint32_t CHEMM_PRODUCT_RING = CHEMM_PRODUCTS_PER_GROUP * CHEMM_PRODUCT_GROUPS;
