/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file csyr2k_tiling_data.h
 * \brief Tiling data structures for aclblasCsyr2k (arch35).
 */

#pragma once

#include <cstdint>

constexpr uint32_t CSYR2K_ARCH35_GM_ALIGN = 512;
constexpr uint32_t CSYR2K_ARCH35_AIV_BLOCK = 64;
constexpr uint32_t CSYR2K_ARCH35_BASE_M = 16;
constexpr uint32_t CSYR2K_ARCH35_BASE_N = 16;
constexpr uint32_t CSYR2K_ARCH35_BASE_K = 32;
constexpr uint32_t CSYR2K_ARCH35_FIXPIPE_N_ALIGN = 8;
constexpr uint32_t CSYR2K_ARCH35_DEFAULT_TILE_M = 128;
constexpr uint32_t CSYR2K_ARCH35_DEFAULT_TILE_N = 128;
constexpr uint32_t CSYR2K_ARCH35_DEFAULT_TILE_K_CHUNK = 256;
constexpr uint32_t CSYR2K_ARCH35_FP32_SIZE = sizeof(float);
constexpr uint32_t CSYR2K_ARCH35_FP16_SIZE = sizeof(uint16_t);
// Preserve the lower-cost HH+HL+LH Gauss-3M specialization through k=2048.
// Larger guarded reductions use segmented full-LL residual 4M.
constexpr uint32_t CSYR2K_ARCH35_RESIDUAL_3TERM_MAX_K = 2048;
constexpr uint32_t CSYR2K_ARCH35_EXACT_FAST_MAX_K = 4096;
constexpr uint32_t CSYR2K_ARCH35_EXACT_FAST_MIN_N = 192;
constexpr uint32_t CSYR2K_ARCH35_EXACT_FAST_MIN_K = 128;
constexpr uint32_t CSYR2K_ARCH35_EXACT_FAST_LARGE_N = 1024;
constexpr uint32_t CSYR2K_ARCH35_EXACT_FAST_LARGE_N_MIN_K = 16;
constexpr uint32_t CSYR2K_ARCH35_HIGH_K_PART_COUNT = 4;
// Small transposed outputs use a direct ordered reduction.  This avoids the
// cancellation amplification caused by reconstructing a complex dot product
// from four independently rounded real GEMMs.
constexpr uint32_t CSYR2K_ARCH35_STRICT_NARROW_MAX_N = 16;
constexpr uint32_t CSYR2K_ARCH35_STRICT_NARROW_MIN_K = 256;
constexpr uint32_t CSYR2K_ARCH35_PARTIAL_ACCUM_TILE_FLOATS = 2048;
// Set this single compile-time knob to 128 for the half-L0C ping-pong variant
// or 256 for the full-L0C variant.  The dependent K/L0C settings keep either
// configuration within the DAV3510 L0A/L0B/L0C and L1 capacities.
constexpr uint32_t CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE = 256;
// The q8 MIX mapping is specialized for exactly eight 256x256 macro tiles.
// Keep this independent from the tunable direct-triangle tile above so a
// future tile-size changes cannot accidentally enter the q8-only kernel.
constexpr uint32_t CSYR2K_ARCH35_MIX_Q8_MACRO_TILE = 256;
constexpr uint32_t CSYR2K_ARCH35_MIX_Q8_MATRIX_SIZE = 2048;
constexpr uint32_t CSYR2K_ARCH35_MIX_Q8_AIC_CORE_COUNT = 28;
constexpr uint32_t CSYR2K_ARCH35_MIX_Q8_DEINTERLEAVE_BLOCK_COUNT = 56;
constexpr uint32_t CSYR2K_ARCH35_DIRECT_TRIANGLE_BASE_K = CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE > 128 ? 64 : 128;
constexpr uint32_t CSYR2K_ARCH35_DIRECT_TRIANGLE_K_CHUNK = CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE > 128 ? 128 : 256;
constexpr uint32_t CSYR2K_ARCH35_DIRECT_TRIANGLE_L0C_BUF_NUM = CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE > 128 ? 1 : 2;
constexpr uint32_t CSYR2K_ARCH35_L1_SIZE_BYTES = 512 * 1024;
constexpr uint32_t CSYR2K_ARCH35_L1_BUF_NUM = 2;
constexpr uint32_t CSYR2K_ARCH35_L1_USAGE_RATIO_NUM = 9;
constexpr uint32_t CSYR2K_ARCH35_L1_USAGE_RATIO_DEN = 10;

// Optional guarded fast path: validate the bounded official performance-input
// domain while producing six FP16 high/residual plane pairs. Low K consumes
// three residual products; guarded high K consumes four full-LL products.
struct Csyr2kFastPrepareTilingData {
    uint32_t rows;
    uint32_t cols;
    uint32_t lda;
    uint32_t ldb;
};

// Ordered complex reduction for the cold small-N transposed path.
struct Csyr2kStrictNarrowTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint8_t uploMode;
};

// Phase 0: split interleaved complex A/B into compact FP32 planes.
struct Csyr2kDeinterleaveTilingData {
    uint32_t rows;
    uint32_t cols;
    uint32_t lda;
    uint32_t ldb;
    uint32_t fastFlagCount;
    uint32_t blockCount;
};

// Phase 1: one AIC dispatcher runs three low-K or four high-K FP16-input
// residual products, or four strict FP32 products, all with FP32 output.
struct Csyr2kGemmTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t leftLd;
    uint32_t rightLd;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileKChunk;
    uint32_t tempRowStride;
    uint32_t fastFlagCount;
    uint32_t directTriangleTile;
    uint8_t isTransN;
    uint8_t uploMode;
    // Segment tilings keep the full-LL residual 4M formula even though each
    // individual segment is no larger than the low-K routing threshold.
    uint8_t isResidual4M;
};

// Adds one segment's four FP32 temporary planes into the retained partials.
struct Csyr2kPartialAccumulateTilingData {
    uint64_t totalElements;
};

// Phase 2: reconstruct/symmetrize the selected 3M or 4M results and update C.
struct Csyr2kCombineTilingData {
    uint32_t n;
    uint32_t ldc;
    uint32_t tempLdc;
    uint32_t fastFlagCount;
    uint32_t directTriangleTile;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    uint8_t uploMode;
    uint8_t skipTemp;
    uint8_t isBetaZero;
    uint8_t isExactFast;
    uint8_t isResidual4M;
    uint8_t mixOffdiagEnabled;
};
