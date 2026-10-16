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
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Vector and DMA units. An FP32 register holds 64 lanes; interleaved complex
// loads contain one real and one imaginary FP32 value for each lane.
constexpr uint32_t CGEMM_SB_FP32_LANES = 64;
constexpr uint32_t CGEMM_SB_COMPLEX_REGISTER_FLOATS = 128;
constexpr uint32_t CGEMM_SB_DMA_BLOCK_BYTES = 32;
constexpr uint32_t CGEMM_SB_FP32_PER_BLOCK = 8;
constexpr uint32_t CGEMM_SB_FP16_PER_BLOCK = 16;
constexpr uint32_t CGEMM_SB_FP32_BYTES = 4;
constexpr uint32_t CGEMM_SB_FP16_BYTES = 2;
constexpr uint32_t CGEMM_SB_COMPLEX_BYTES = 8;
constexpr uint32_t CGEMM_SB_FRACTAL_EDGE = 16;
constexpr uint32_t CGEMM_SB_FP32_SIGN_SHIFT = 31;

// Magnitudes are reduced as sign-cleared IEEE-754 bits, so NaN and infinity
// also exceed these existing fallback thresholds. Keep bitwise comparisons.
constexpr int32_t CGEMM_SB_HALF_FALLBACK_BITS = 0x461C4000;  // 10000.0f
constexpr int32_t CGEMM_SB_FLOAT_FALLBACK_BITS = 0x501502F9; // 1.0e10f
constexpr uint32_t CGEMM_SB_FP32_ABS_MASK = 0x7FFFFFFFU;
constexpr uint32_t CGEMM_SB_FP32_INFINITY_BITS = 0x7F800000U;
constexpr uint32_t CGEMM_SB_FP32_QUIET_NAN_BIT = 0x00400000U;
constexpr uint32_t CGEMM_SB_FP32_INVALID_NAN_BITS = 0xFFC00000U;
constexpr uint32_t CGEMM_SB_INT32_INDEX_MAX = 0x7FFFFFFFU;

// Reduction storage: 64 running maxima, a result at 64, and temporary
// reduction storage at 128. The local single-slice path uses 192 int32 slots.
constexpr uint32_t CGEMM_SB_MAX_RESULT_OFFSET = 64;
constexpr uint32_t CGEMM_SB_MAX_TEMP_OFFSET = 128;
constexpr uint32_t CGEMM_SB_MAX_FLAG_OFFSET = 96;
constexpr uint32_t CGEMM_SB_MAX_STORAGE_ELEMENTS = 192;
constexpr uint32_t CGEMM_SB_MAX_SCRATCH_BYTES = 1024;

// Existing memory partitions and launch sizes for the arch35 implementation.
constexpr uint32_t CGEMM_SB_LOCAL_SMALL_SLOT_BYTES = 64 * 1024;
constexpr uint32_t CGEMM_SB_LOCAL_LARGE_SLOT_BYTES = 128 * 1024;
constexpr uint32_t CGEMM_SB_LOCAL_L1_REGION_BYTES = 256 * 1024;
constexpr uint32_t CGEMM_SB_L0_HALF_BANK_BYTES = 32768;
constexpr uint32_t CGEMM_SB_L0_HALF_BANK_ELEMENTS = 16384;
constexpr uint32_t CGEMM_SB_L0C_SLOT_BYTES = 128 * 1024;
constexpr uint32_t CGEMM_SB_UB_BUDGET_BYTES = 224 * 1024;
constexpr uint32_t CGEMM_SB_SIMT_THREADS = 256;
constexpr uint32_t CGEMM_SB_SIMT_LAUNCH_BOUND = 512;
// Host routing and device input-moment/reference decisions share this cutoff.
constexpr uint32_t CGEMM_SB_ORDERED_MIN_K = 512;

// Vector dispatch modes; values are shared by Host launch and device dispatch.
constexpr uint32_t CGEMM_SB_MODE_SCALE = 0;
constexpr uint32_t CGEMM_SB_MODE_DIRECT = 1;
constexpr uint32_t CGEMM_SB_MODE_PACK = 2;
constexpr uint32_t CGEMM_SB_MODE_COMBINE = 3;
constexpr uint32_t CGEMM_SB_MODE_UNIT = 4;
constexpr uint32_t CGEMM_SB_MODE_TINY = 6;

// Reference-selection bits and cross-core event IDs retain the existing
// protocol. Local pipeline events occupy consecutive per-slot ranges.
constexpr uint32_t CGEMM_SB_REFERENCE_REAL = 1;
constexpr uint32_t CGEMM_SB_REFERENCE_NONE = 0;
constexpr uint32_t CGEMM_SB_REFERENCE_IMAG = 2;
constexpr uint32_t CGEMM_SB_REFERENCE_BOTH = 3;
constexpr uint32_t CGEMM_SB_REFERENCE_EXCEPTIONAL = 4;
constexpr uint16_t CGEMM_SB_FUSED_VECTOR_READY = 8;
constexpr uint16_t CGEMM_SB_FUSED_PACK_READY = 9;
constexpr uint16_t CGEMM_SB_CUBE_DONE = 10;
constexpr uint16_t CGEMM_SB_CUBE_BARRIER = 11;
constexpr uint16_t CGEMM_SB_PREFIX_READY = 13;
constexpr uint16_t CGEMM_SB_PREFIX_BARRIER = 14;
constexpr uint16_t CGEMM_SB_LOCAL_PACK_EVENT_BASE = 8;
constexpr uint16_t CGEMM_SB_LOCAL_RELEASE_EVENT_BASE = 12;

// Packed operands contain real/imaginary planes, or real/imaginary/sum for 3M.
// Compensated FP16 packing adds the corresponding three residual planes.
constexpr uint32_t CGEMM_SB_COMPONENT_PLANES = 2;
constexpr uint32_t CGEMM_SB_THREE_M_PLANES = 3;
constexpr uint16_t CGEMM_SB_HALF_PLANES = 6;
constexpr uint32_t CGEMM_SB_GENERAL_PRODUCTS = 4;
constexpr uint32_t CGEMM_SB_PARTIAL_PRODUCTS = 2;
// Plane lengths in tiling are FP32 units: three FP16 high planes occupy
// six half-element strides before the residual planes in Cube addressing.
constexpr uint32_t CGEMM_SB_HALF_RESIDUAL_OFFSET = 6;
constexpr uint32_t CGEMM_SB_PRODUCT_RR = 0;
constexpr uint32_t CGEMM_SB_PRODUCT_II = 1;
constexpr uint32_t CGEMM_SB_PRODUCT_IR = 2;
constexpr uint32_t CGEMM_SB_PRODUCT_RI = 3;
constexpr uint32_t CGEMM_SB_PRODUCT_SUM = 2;
// Operand-plane indices are distinct from the four product-plane indices.
constexpr uint32_t CGEMM_SB_PLANE_SUM = 2;

// Each complex element occupies eight bytes, so one 32-byte DMA block
// contains four complex elements. Gather byte indices use these shifts.
constexpr uint32_t CGEMM_SB_COMPLEX_PER_BLOCK = 4;
constexpr uint32_t CGEMM_SB_COMPLEX_BYTE_SHIFT = 3;
constexpr uint32_t CGEMM_SB_FP32_BYTE_SHIFT = 2;
// The local 2x2 real representation, including FP16 high/residual storage,
// occupies four FP32 storage units per logical A element.
constexpr uint32_t CGEMM_SB_LOCAL_A_STORAGE_FLOATS = 4;

// Producer and consumer share an aligned flag block: exception indicator,
// sums of A/B real and imaginary components, then their element counts.
constexpr uint32_t CGEMM_SB_META_EXCEPTION = 0;
constexpr uint32_t CGEMM_SB_META_A_REAL = 1;
constexpr uint32_t CGEMM_SB_META_A_IMAG = 2;
constexpr uint32_t CGEMM_SB_META_B_REAL = 3;
constexpr uint32_t CGEMM_SB_META_B_IMAG = 4;
constexpr uint32_t CGEMM_SB_META_A_COUNT = 5;
constexpr uint32_t CGEMM_SB_META_B_COUNT = 6;
constexpr uint32_t CGEMM_SB_MOMENT_PLANES = 4;

// Each core writes a complete aligned 32-byte block through MTE3, avoiding scalar cache-line writeback races.
constexpr uint32_t CGEMM_SB_FLAG_STRIDE = 8;

constexpr uint32_t CGEMM_SB_LOCAL_SLICE_ROWS = 32;

struct CgemmSbTiling {
    uint32_t m, n, k, batch;
    uint32_t lda, ldb, ldc, transA, transB;
    int64_t strideA, strideB, strideC;
    float ar, ai, br, bi;
    uint32_t mp, np, kp, batchesA, batchesB;
    uint32_t tileM, tileN, baseK;
    uint32_t threeM, flagCores;
    uint32_t densePack, denseCombine, compensatedHalf, batchGroup, localBatchMerge, localInputPad, localSlices;
    uint32_t localSliceRows;
    // Offsets and plane lengths in FP32 elements, all allocation boundaries 512B aligned.
    uint64_t aSize, bSize, cSize, bOffset, cOffset;
    uint64_t flagOffset;
    uint32_t orderedM, orderedN, orderedK, normalizeOrdered;
    uint64_t orderedAOffset, orderedBOffset;
};

void cgemm_sb_vector_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling& t,
    uint32_t mode);
void cgemm_sb_cube_do(uint32_t blocks, void* stream, GM_ADDR workspace, const CgemmSbTiling& t);

void cgemm_sb_fused_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t);
void cgemm_sb_dense_vector_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling& t,
    bool pack);

void cgemm_sb_local_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t);

void cgemm_sb_hybrid_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t);

void cgemm_sb_normalize_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR ws, const CgemmSbTiling& t);
