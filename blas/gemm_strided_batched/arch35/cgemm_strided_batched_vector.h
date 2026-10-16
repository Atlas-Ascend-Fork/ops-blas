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
#include "cgemm_strided_batched_reference.h"

// Vector packing primitives and compensated-input preparation.
namespace {

namespace {
constexpr uint32_t TILE = 2048;
// Real, imaginary and sum planes each have FP16 high/residual parts.
constexpr uint16_t HALF_PACK_PLANES = CGEMM_SB_HALF_PLANES;
constexpr uint16_t PADDING_READY_EVENT = 3;
// Partial combination reads two product planes and the previous complex output.
constexpr uint32_t PARTIAL_INPUT_PLANES = 4;

template <class Register, class Mask>
__simd_callee__ inline void PackComponent(
    Register& value, __ubuf__ half* dst, uint32_t offset, uint32_t lowOffset, Mask mask);
__simd_vf__ inline void MicroPackHalf(
    __ubuf__ float* src, __ubuf__ half* dst, __ubuf__ int32_t* maximum, uint32_t count, bool conjugate,
    uint32_t stride);
template <uint32_t STRIDE>
__simd_vf__ inline void MicroCombineUnit(__ubuf__ float* src, __ubuf__ float* dst, uint32_t count);
__aicore__ inline uint32_t ScalarMin(uint32_t a, uint32_t b);

// Tile in physical column order. Several short columns share one DMA.
struct ColumnTile {
    uint32_t row;
    uint32_t col;
    uint32_t rows;
    uint32_t cols;
    uint32_t validRows;
    uint32_t validCols;
};

__aicore__ inline uint64_t TileCount(uint32_t rows, uint32_t cols, uint32_t capacity = TILE);

__aicore__ inline ColumnTile DecodeTile(
    uint64_t index, uint32_t paddedRows, uint32_t paddedCols, uint32_t logicalRows, uint32_t logicalCols,
    uint32_t capacity = TILE);

__aicore__ inline void ReadColumns(
    AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, const ColumnTile& v, uint32_t ld);

struct PackGeometry {
    uint32_t rows;
    uint32_t cols;
    uint32_t paddedRows;
    uint32_t paddedCols;
    uint32_t ld;
    uint32_t capacity;
    bool dense;
    bool flat;
    int64_t stride;
    uint64_t matrix;
    uint64_t elements;
    uint64_t tiles;
    uint64_t total;
};

__aicore__ inline PackGeometry MakePackGeometry(const CgemmSbTiling& t, bool isA, uint32_t capacity);

__aicore__ inline void ReadPackTile(
    AscendC::LocalTensor<float> x, AscendC::GlobalTensor<float> source, const PackGeometry& g, uint64_t index,
    uint64_t& batch, uint64_t& offset, uint32_t& count);

template <uint32_t SIZE>
class FastHalfPacker {
public:
    __aicore__ inline void Run(
        GM_ADDR a, GM_ADDR b, GM_ADDR workspace, CgemmSbTiling t, uint32_t core, uint32_t cores, AscendC::TPipe& pipe);

private:
    __aicore__ inline void PublishFlag(AscendC::GlobalTensor<float> wg, const CgemmSbTiling& t, uint32_t core);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> input;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> output, flagq;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch;
};
template <uint32_t SIZE>
__simd_vf__ inline void MicroMergeComponent(
    __ubuf__ float* src, __ubuf__ float* dst, uint32_t count, uint32_t component);

template <uint32_t SIZE, bool Partial = false, uint32_t OUTPUT_SLOTS = 2>
class FastUnitCombiner {
public:
    __aicore__ inline void Run(
        GM_ADDR c, GM_ADDR workspace, CgemmSbTiling t, uint32_t core, uint32_t cores, AscendC::TPipe& pipe,
        uint32_t component = 0);

private:
    __aicore__ inline void ComputeTile(
        AscendC::LocalTensor<float> x, AscendC::LocalTensor<float> y, uint32_t count, uint32_t component);

    __aicore__ inline void ReadCurrentColumns(
        AscendC::LocalTensor<float> x, AscendC::GlobalTensor<float> c, const CgemmSbTiling& t, uint32_t cols);
    __aicore__ inline void StoreTile(
        AscendC::GlobalTensor<float> cg, AscendC::LocalTensor<float> y, const CgemmSbTiling& t, bool flat, int64_t dest,
        uint32_t count, uint32_t cols);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> input;
    AscendC::TQue<AscendC::TPosition::VECOUT, OUTPUT_SLOTS> output;
};

} // namespace
} // namespace

// Vector tile capacities and scratch-slot layout.
namespace {
// Existing UB-sized fast-path tiles; keep capacities and routing unchanged.
constexpr uint32_t FAST_TILE_ELEMENTS = 8192;
constexpr uint32_t LARGE_FAST_TILE_ELEMENTS = 12544;
constexpr uint32_t PARTIAL_TILE_ELEMENTS = 4096;
constexpr uint32_t SHORT_HALF_K_LIMIT = 512;
constexpr uint32_t FAST_CUBE_DIM = 128;
// Eight TILE-sized scratch slots are reused by packing, batch
// indexing and combination with the phase-specific roles below.
constexpr uint32_t VECTOR_CALC_TILE_SLOTS = 8;
constexpr uint32_t MERGE_FOUR_BATCH_GROUP = 4;

// Phase-specific scratch slots share the same eight-slot allocation above.
constexpr uint32_t VECTOR_PACK_ABS_SLOT = 2;
constexpr uint32_t VECTOR_PACK_MASK_SLOT = 4;
constexpr uint32_t VECTOR_MERGE_POSITION_SLOT = 2;
constexpr uint32_t VECTOR_MERGE_COLUMN_SLOT = 3;
constexpr uint32_t VECTOR_MERGE_BATCH_SLOT = 6;
constexpr uint32_t VECTOR_MERGE_VALUES_SLOT = 6;
constexpr uint32_t VECTOR_REDUCTION_SCRATCH_SLOT = 6;
constexpr uint32_t VECTOR_REDUCTION_RESULT_SLOT = 7;
constexpr uint32_t VECTOR_COMBINE_REAL_SLOT = 2;
constexpr uint32_t VECTOR_COMBINE_IMAG_SLOT = 3;
constexpr uint32_t VECTOR_COMBINE_TEMP_SLOT = 4;
constexpr uint32_t VECTOR_COMBINE_BETA_REAL_SLOT = 5;
constexpr uint32_t VECTOR_COMBINE_BETA_IMAG_SLOT = 6;
constexpr uint32_t VECTOR_COMBINE_AUX_SLOT = 7;
// Power-of-two shifts used by the stacked batch-index layout.
constexpr uint32_t MERGE_SMALL_DIM_SHIFT = 4;
constexpr uint32_t MERGE_LARGE_DIM_SHIFT = 5;
constexpr uint32_t MERGE_FOUR_BATCH_SHIFT = 30;
constexpr uint32_t MERGE_TWO_BATCH_SHIFT = 31;
} // namespace

// Vector packing, product combination and output writeback.
namespace {

namespace {
class CgemmStridedBatchedVectorImpl {
public:
    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgemmSbTiling& t, bool pack, bool mixed,
        AscendC::TPipe& pipe);

    __aicore__ inline void Pack();

    __aicore__ inline void Combine();

private:
    __aicore__ inline void StoreCombinedTile(
        AscendC::LocalTensor<float> out, const ColumnTile& view, int64_t dest, uint32_t count);

    __aicore__ inline void PackShortHalf(
        AscendC::LocalTensor<float> data, bool isA, uint64_t batch, uint64_t matrix, uint64_t offset, uint32_t count);

    __aicore__ inline void MakePlanes(
        AscendC::LocalTensor<float> data, AscendC::LocalTensor<float> out, bool isA, uint32_t count);

    __aicore__ inline void MergeBPlanes(AscendC::LocalTensor<float> out, uint64_t batch, uint32_t count);

    __aicore__ inline void StoreHalfPlanes(
        AscendC::LocalTensor<float> out, bool isA, uint64_t batch, uint64_t matrix, uint64_t offset, uint32_t count);

    __aicore__ inline void PublishMetadata();

    __aicore__ inline void CombineGeneral(
        AscendC::LocalTensor<float> data, AscendC::LocalTensor<float> out, const ColumnTile& view, int64_t dest,
        uint32_t count);

    AscendC::TQue<AscendC::TPosition::VECIN, 2> input_, old_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> output_, halfOutput_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_, indices_, moments_;
    AscendC::LocalTensor<uint32_t> even_;
    AscendC::LocalTensor<uint32_t> odd_;
    AscendC::GlobalTensor<float> a_;
    AscendC::GlobalTensor<float> b_;
    AscendC::GlobalTensor<float> c_;
    AscendC::GlobalTensor<float> ws_;
    CgemmSbTiling t_{};
    uint64_t countA_ = 0;
    uint64_t countB_ = 0;
    uint32_t core_ = 0;
    uint32_t cores_ = 0;
};
} // namespace

template <bool Mixed>
__aicore__ inline void CgemmSbRunExceptional(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t, uint32_t core, uint32_t cores);

__aicore__ inline bool UseFastHalfPacker(const CgemmSbTiling& t, bool pack);

__aicore__ inline bool UseFastUnitCombiner(const CgemmSbTiling& t, bool pack);

template <bool Mixed>
__aicore__ inline void CgemmSbRunVector(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, CgemmSbTiling t, bool pack);

} // namespace
