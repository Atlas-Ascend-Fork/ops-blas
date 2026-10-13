/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file cgeam_kernel.cpp
 * @brief Cgeam kernel implementation for arch35
 *        Computes: C = alpha * op(A) + beta * op(B) for complex matrices
 *
 *        Three general paths are selected by data layout and scalar semantics:
 *          - Packed SIMD path for compact C = A or C = A + B.
 *          - Hardware-transpose tiled path for compact unit-scalar transpose cases.
 *          - SIMT path for arbitrary scalars, leading dimensions, and operations.
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cgeam_kernel.h"

namespace {

using namespace AscendC;
using CgeamConstants::BLOCK_BYTES;
using CgeamConstants::INPUT_MATRIX_COUNT;
using CgeamConstants::L2_CAPACITY_BYTES;
using CgeamConstants::WORKSET_MATRIX_COUNT;

constexpr uint32_t COMPLEX_FLOAT_COUNT = 2;
constexpr uint32_t FLOATS_PER_BLOCK = BLOCK_BYTES / sizeof(float);
constexpr uint32_t COMPLEX_PER_BLOCK = BLOCK_BYTES / (COMPLEX_FLOAT_COUNT * sizeof(float));
constexpr uint32_t SIMT_LAUNCH_BOUND_THREADS = 2048;
// A float transpose repeat has 16 row addresses; each output column spans two blocks.
constexpr uint32_t TRANSPOSE_REPEAT_ROWS = 16U;
constexpr uint32_t TRANSPOSE_OUTPUT_BLOCKS_PER_COLUMN = TRANSPOSE_REPEAT_ROWS / FLOATS_PER_BLOCK;
// Each repeat advances four complex columns, with strides expressed in 32-byte blocks.
constexpr uint32_t TRANSPOSE_REPEAT_STRIDE_DIVISOR = FLOATS_PER_BLOCK / COMPLEX_PER_BLOCK;
constexpr uint32_t VECTOR_MASK_WORD_COUNT = 2U;
// Interleaved complex floats place imaginary components in the odd-numbered lanes.
constexpr uint64_t IMAGINARY_COMPONENT_MASK = 0xAAAAAAAAAAAAAAAAULL;
constexpr uint32_t NN_INPUT_A_QUEUE_DEPTH = 2U;
constexpr uint32_t NN_INPUT_B_QUEUE_DEPTH = 1U;
// Tag 4 reuses each consumed B slot as transpose output; its queue has depth two.
constexpr uint32_t MERGED_REUSE_PIPELINE_TAG = 4U;
constexpr uint32_t SHALLOW_TRANSPOSE_PIPELINE_DEPTH = 2U;
constexpr uint32_t DEEP_TRANSPOSE_PIPELINE_DEPTH = 3U;

union PackedComplex {
    uint64_t packed;
    float components[COMPLEX_FLOAT_COUNT];
};

__aicore__ inline uint64_t LogicalMatrixBytes(const CgeamTilingData& tiling)
{
    return static_cast<uint64_t>(tiling.m) * tiling.n * sizeof(PackedComplex);
}

__aicore__ inline bool OneInputBenefitsFromL2Retention(const CgeamTilingData& tiling)
{
    const uint64_t matrixBytes = LogicalMatrixBytes(tiling);
    // Capacity tier 3: one input fits, but two inputs do not fit together.
    return INPUT_MATRIX_COUNT * matrixBytes >= L2_CAPACITY_BYTES && matrixBytes <= L2_CAPACITY_BYTES;
}

__aicore__ inline bool InputsFitButFullWorksetDoesNot(const CgeamTilingData& tiling)
{
    const uint64_t matrixBytes = LogicalMatrixBytes(tiling);
    // Capacity tier 2: A and B fit together, but adding C exceeds L2.
    return INPUT_MATRIX_COUNT * matrixBytes < L2_CAPACITY_BYTES &&
        WORKSET_MATRIX_COUNT * matrixBytes >= L2_CAPACITY_BYTES;
}

__aicore__ inline bool UseDeepTransposePipeline(const CgeamTilingData& tiling)
{
    return WORKSET_MATRIX_COUNT * LogicalMatrixBytes(tiling) >= L2_CAPACITY_BYTES;
}

__aicore__ inline void ConfigureL2Policy(
    GlobalTensor<float>& aGm, GlobalTensor<float>& bGm, GlobalTensor<float>& cGm,
    const CgeamTilingData& tiling)
{
    if (InputsFitButFullWorksetDoesNot(tiling)) {
        cGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    } else if (OneInputBenefitsFromL2Retention(tiling)) {
        const uint64_t matrixBytes = LogicalMatrixBytes(tiling);
        aGm.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
        // Once one logical input reaches L2 capacity, retaining A leaves no
        // useful room for B, so B bypasses L2 instead of competing with A.
        if (matrixBytes >= L2_CAPACITY_BYTES) {
            bGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        // C is write-only; bypass it to protect the retained input.
        cGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
class CgeamTCBlockAIV {
public:
    static_assert(
        PIPELINE_DEPTH == MERGED_REUSE_PIPELINE_TAG ||
        PIPELINE_DEPTH == SHALLOW_TRANSPOSE_PIPELINE_DEPTH ||
        PIPELINE_DEPTH == DEEP_TRANSPOSE_PIPELINE_DEPTH,
        "unsupported transpose pipeline configuration");
    static_assert(PIPELINE_DEPTH != MERGED_REUSE_PIPELINE_TAG || (!SINGLE_INPUT && !MIXED_TRANSPOSE),
        "buffer reuse requires matching dual-input transpose operations");
    static constexpr uint32_t QUEUE_DEPTH = PIPELINE_DEPTH == MERGED_REUSE_PIPELINE_TAG ?
        SHALLOW_TRANSPOSE_PIPELINE_DEPTH : PIPELINE_DEPTH;
    __aicore__ inline CgeamTCBlockAIV() = default;
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessShallowPipeline(uint32_t tileStart, uint32_t tileCount);
    __aicore__ inline void ProcessDeepPipeline(uint32_t tileStart, uint32_t tileCount);
    __aicore__ inline void TransposeOne(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowOffset);
    __aicore__ inline void CombineMixedTransposedInputs(
        const LocalTensor<float>& inputA, const LocalTensor<float>& inputB, uint32_t floatCount);
    __aicore__ inline void CopyTransposedAsync(
        GlobalTensor<float>& gm, const LocalTensor<float>& input, uint32_t ld,
        uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols);
    __aicore__ inline void CopyNormalAsync(
        GlobalTensor<float>& gm, const LocalTensor<float>& input, uint32_t ld,
        uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols);
    __aicore__ inline void DecodeTile(uint32_t tileIdx, uint32_t& rowTile, uint32_t& colTile);
    __aicore__ inline void IssueQueuedPrefetch(
        uint32_t tileIdx, uint32_t slot, LocalTensor<float>& inputA);
    __aicore__ inline void TransformQueuedTile(
        uint32_t tileIdx, uint32_t slot, LocalTensor<float>& inputA);
    __aicore__ inline void ComputeQueuedTile(LocalTensor<float>& output, uint32_t slot);
    __aicore__ inline void StoreQueuedTile(uint32_t tileIdx, LocalTensor<float>& output);
    __aicore__ inline LocalTensor<float> GetInputB(uint32_t slot);
    __aicore__ inline LocalTensor<float> GetTransposeBuffer(uint32_t slot);
    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> cGm_;
    TBuf<TPosition::VECCALC> inputBBuf_;
    TBuf<TPosition::VECCALC> inputBAltBuf_;
    TBuf<TPosition::VECCALC> inputBThirdBuf_;
    TBuf<TPosition::VECCALC> transBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, QUEUE_DEPTH> inputOutputQueue_;

    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t lda_ = 0;
    uint32_t ldb_ = 0;
    uint32_t ldc_ = 0;
    uint32_t opA_ = ACLBLAS_OP_N;
    uint32_t opB_ = ACLBLAS_OP_N;
    uint32_t tileRows_ = 0;
    uint32_t tileCols_ = 0;
    uint32_t rowTiles_ = 0;
    uint32_t totalTiles_ = 0;
};

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::Init(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, TPipe* pipe)
{
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B));
    cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C));
    if constexpr (PIPELINE_DEPTH == MERGED_REUSE_PIPELINE_TAG) {
        // For the unaligned matching-transpose path, merged input bursts bypass
        // L2 when both complete inputs cannot fit. Ordinary C stores can then
        // coalesce without competing with retained A.
        if (OneInputBenefitsFromL2Retention(tiling)) {
            aGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            bGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        } else {
            ConfigureL2Policy(aGm_, bGm_, cGm_, tiling);
        }
    } else if constexpr (!SINGLE_INPUT) {
        ConfigureL2Policy(aGm_, bGm_, cGm_, tiling);
    }
    m_ = tiling.m;
    n_ = tiling.n;
    lda_ = tiling.lda;
    ldb_ = tiling.ldb;
    ldc_ = tiling.ldc;
    opA_ = tiling.opA;
    opB_ = tiling.opB;
    tileRows_ = tiling.transposeTileRows;
    tileCols_ = tiling.transposeTileCols;
    rowTiles_ = tiling.transposeRowTiles;
    totalTiles_ = tiling.transposeTotalTiles;

    const uint32_t tileBytes = tileRows_ * tileCols_ * COMPLEX_FLOAT_COUNT * sizeof(float);
    if constexpr (!SINGLE_INPUT) {
        pipe->InitBuffer(inputBBuf_, tileBytes);
        pipe->InitBuffer(inputBAltBuf_, tileBytes);
        if constexpr (PIPELINE_DEPTH == DEEP_TRANSPOSE_PIPELINE_DEPTH) {
            pipe->InitBuffer(inputBThirdBuf_, tileBytes);
        }
    }
    if constexpr (PIPELINE_DEPTH != MERGED_REUSE_PIPELINE_TAG) {
        pipe->InitBuffer(transBuf_, tileBytes);
    }
    pipe->InitBuffer(inputOutputQueue_, QUEUE_DEPTH, tileBytes);
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline LocalTensor<float> CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::GetInputB(
    uint32_t slot)
{
    if (slot == 0U) {
        return inputBBuf_.Get<float>();
    }
    if constexpr (PIPELINE_DEPTH == DEEP_TRANSPOSE_PIPELINE_DEPTH) {
        if (slot == DEEP_TRANSPOSE_PIPELINE_DEPTH - 1U) {
            return inputBThirdBuf_.Get<float>();
        }
    }
    return inputBAltBuf_.Get<float>();
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline LocalTensor<float>
CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::GetTransposeBuffer(uint32_t slot)
{
    if constexpr (PIPELINE_DEPTH == MERGED_REUSE_PIPELINE_TAG) {
        return GetInputB(slot);
    } else {
        return transBuf_.Get<float>();
    }
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::TransposeOne(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowOffset)
{
    LocalTensor<float> srcList[TRANSPOSE_REPEAT_ROWS];
    LocalTensor<float> dstList[TRANSPOSE_REPEAT_ROWS];
    const uint32_t inputRowFloats = tileCols_ * COMPLEX_FLOAT_COUNT;
    for (uint32_t i = 0; i < TRANSPOSE_REPEAT_ROWS; ++i) {
        srcList[i] = src[(rowOffset + i) * inputRowFloats];
    }
    const uint32_t planeElements = tileRows_ * tileCols_;
    for (uint32_t i = 0; i < FLOATS_PER_BLOCK; ++i) {
        const uint32_t component = i & 1U;
        const uint32_t complexCol = i >> 1U;
        const uint32_t dstOffset = component * planeElements + complexCol * tileRows_ + rowOffset;
        dstList[i * TRANSPOSE_OUTPUT_BLOCKS_PER_COLUMN] = dst[dstOffset];
        dstList[i * TRANSPOSE_OUTPUT_BLOCKS_PER_COLUMN + 1U] = dst[dstOffset + FLOATS_PER_BLOCK];
    }
    const uint8_t repeats = static_cast<uint8_t>(inputRowFloats / FLOATS_PER_BLOCK);
    // Every repeat consumes four complex columns. Keep all real columns in the first
    // plane and all imaginary columns in the second so one Interleave can form AoS output.
    const uint16_t dstRepStride = static_cast<uint16_t>(tileRows_ / TRANSPOSE_REPEAT_STRIDE_DIVISOR);
    TransDataTo5HDParams params(false, false, repeats, dstRepStride, 1U);
    TransDataTo5HD<float>(dstList, srcList, params);
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::CombineMixedTransposedInputs(
    const LocalTensor<float>& inputA, const LocalTensor<float>& inputB, uint32_t floatCount)
{
    LocalTensor<float> conjugatedInput = opA_ == ACLBLAS_OP_C ? inputA : inputB;
    uint64_t imaginaryMask[VECTOR_MASK_WORD_COUNT] = {IMAGINARY_COMPONENT_MASK, 0U};
    constexpr uint32_t FLOATS_PER_REPEAT = 64U;
    constexpr uint32_t MAX_REPEATS = 255U;
    constexpr uint8_t REPEAT_STRIDE_BLOCKS = FLOATS_PER_REPEAT / FLOATS_PER_BLOCK;
    uint32_t processedFloats = 0U;
    while (processedFloats < floatCount) {
        const uint32_t remainingRepeats = (floatCount - processedFloats) / FLOATS_PER_REPEAT;
        const uint8_t repeats = static_cast<uint8_t>(
            remainingRepeats > MAX_REPEATS ? MAX_REPEATS : remainingRepeats);
        Muls<float, true>(conjugatedInput[processedFloats], conjugatedInput[processedFloats],
            -1.0f, imaginaryMask, repeats, UnaryRepeatParams{1U, 1U, REPEAT_STRIDE_BLOCKS, REPEAT_STRIDE_BLOCKS});
        processedFloats += static_cast<uint32_t>(repeats) * FLOATS_PER_REPEAT;
    }
    PipeBarrier<PIPE_V>();
    Add(inputA, inputA, inputB, floatCount);
    PipeBarrier<PIPE_V>();
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::CopyTransposedAsync(
    GlobalTensor<float>& gm, const LocalTensor<float>& input, uint32_t ld,
    uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols)
{
    const uint32_t rowBytes = validCols * COMPLEX_FLOAT_COUNT * sizeof(float);
    const uint64_t offset =
        COMPLEX_FLOAT_COUNT * (static_cast<uint64_t>(rowBase) * ld + colBase);
    const int64_t srcStride = static_cast<int64_t>(ld - validCols) * sizeof(PackedComplex);
    const uint32_t fullRowBlocks =
        tileCols_ * COMPLEX_FLOAT_COUNT * sizeof(float) / BLOCK_BYTES;
    const uint32_t usedRowBlocks = (rowBytes + BLOCK_BYTES - 1U) / BLOCK_BYTES;
    const int64_t dstStride = static_cast<int64_t>(fullRowBlocks - usedRowBlocks);
    DataCopyExtParams params{
        static_cast<uint16_t>(validRows), rowBytes, srcStride, dstStride, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    DataCopyPad(input, gm[offset], params, noPad);
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::CopyNormalAsync(
    GlobalTensor<float>& gm, const LocalTensor<float>& input, uint32_t ld,
    uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols)
{
    const uint32_t rowBytes = validRows * COMPLEX_FLOAT_COUNT * sizeof(float);
    const uint64_t offset =
        COMPLEX_FLOAT_COUNT * (static_cast<uint64_t>(colBase) * ld + rowBase);
    const int64_t srcStride = static_cast<int64_t>(ld - validRows) * sizeof(PackedComplex);
    const uint32_t fullRowBlocks =
        tileRows_ * COMPLEX_FLOAT_COUNT * sizeof(float) / BLOCK_BYTES;
    const uint32_t usedRowBlocks = (rowBytes + BLOCK_BYTES - 1U) / BLOCK_BYTES;
    const int64_t dstStride = static_cast<int64_t>(fullRowBlocks - usedRowBlocks);
    DataCopyExtParams params{
        static_cast<uint16_t>(validCols), rowBytes, srcStride, dstStride, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    DataCopyPad(input, gm[offset], params, noPad);
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::DecodeTile(
    uint32_t tileIdx, uint32_t& rowTile, uint32_t& colTile)
{
    const uint64_t matrixBytes = static_cast<uint64_t>(m_) * n_ * sizeof(PackedComplex);
    const bool favorTransposedReadOrder =
        WORKSET_MATRIX_COUNT * matrixBytes >= L2_CAPACITY_BYTES && matrixBytes <= L2_CAPACITY_BYTES;
    if (favorTransposedReadOrder) {
        const uint32_t colTiles = (n_ + tileCols_ - 1U) / tileCols_;
        rowTile = tileIdx / colTiles;
        colTile = tileIdx % colTiles;
    } else {
        rowTile = tileIdx % rowTiles_;
        colTile = tileIdx / rowTiles_;
    }
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::IssueQueuedPrefetch(
    uint32_t tileIdx, uint32_t slot, LocalTensor<float>& inputA)
{
    uint32_t rowTile = 0U;
    uint32_t colTile = 0U;
    DecodeTile(tileIdx, rowTile, colTile);
    const uint32_t rowBase = rowTile * tileRows_;
    const uint32_t colBase = colTile * tileCols_;
    const uint32_t validRows = (m_ - rowBase < tileRows_) ? (m_ - rowBase) : tileRows_;
    const uint32_t validCols = (n_ - colBase < tileCols_) ? (n_ - colBase) : tileCols_;
    inputA = inputOutputQueue_.template AllocTensor<float>();
    if constexpr (SINGLE_INPUT) {
        CopyTransposedAsync(aGm_, inputA, lda_, rowBase, colBase, validRows, validCols);
        return;
    }
    LocalTensor<float> inputB = GetInputB(slot);
    // Edge tiles do not need a separate padding pass. Transpose is an elementwise
    // permutation and StoreQueuedTile writes only the valid output rectangle, so
    // values outside validRows/validCols cannot affect a stored element.
    if (opA_ == ACLBLAS_OP_N) {
        CopyNormalAsync(aGm_, inputA, lda_, rowBase, colBase, validRows, validCols);
        CopyTransposedAsync(bGm_, inputB, ldb_, rowBase, colBase, validRows, validCols);
    } else if (opB_ == ACLBLAS_OP_N) {
        CopyTransposedAsync(aGm_, inputA, lda_, rowBase, colBase, validRows, validCols);
        CopyNormalAsync(bGm_, inputB, ldb_, rowBase, colBase, validRows, validCols);
    } else {
        CopyTransposedAsync(aGm_, inputA, lda_, rowBase, colBase, validRows, validCols);
        CopyTransposedAsync(bGm_, inputB, ldb_, rowBase, colBase, validRows, validCols);
    }
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::TransformQueuedTile(
    uint32_t tileIdx, uint32_t slot, LocalTensor<float>& inputA)
{
    uint32_t rowTile = 0U;
    uint32_t colTile = 0U;
    DecodeTile(tileIdx, rowTile, colTile);
    const uint32_t rowBase = rowTile * tileRows_;
    const uint32_t validRows = (m_ - rowBase < tileRows_) ? (m_ - rowBase) : tileRows_;
    LocalTensor<float> trans = GetTransposeBuffer(slot);
    LocalTensor<float> source = inputA;
    if constexpr (!SINGLE_INPUT) {
        LocalTensor<float> inputB = GetInputB(slot);
        if (opA_ == ACLBLAS_OP_N) {
            source = inputB;
        } else if (opB_ != ACLBLAS_OP_N) {
            const uint32_t floatCount = validRows * tileCols_ * COMPLEX_FLOAT_COUNT;
            if constexpr (MIXED_TRANSPOSE) {
                CombineMixedTransposedInputs(inputA, inputB, floatCount);
            } else {
                Add(inputA, inputA, inputB, floatCount);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    for (uint32_t rowOffset = 0; rowOffset < validRows; rowOffset += TRANSPOSE_REPEAT_ROWS) {
        TransposeOne(trans, source, rowOffset);
    }
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::ComputeQueuedTile(
    LocalTensor<float>& output, uint32_t slot)
{
    if constexpr (SINGLE_INPUT) {
        LocalTensor<float> trans = GetTransposeBuffer(slot);
        const uint32_t planeElements = tileRows_ * tileCols_;
        if (opA_ == ACLBLAS_OP_C) {
            Muls(trans[planeElements], trans[planeElements], -1.0f, planeElements);
            PipeBarrier<PIPE_V>();
        }
        Interleave(output, output[planeElements], trans, trans[planeElements],
            static_cast<int32_t>(planeElements));
        inputOutputQueue_.template EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(output);
        output = inputOutputQueue_.template DeQue<QuePosition::VECCALC, QuePosition::VECOUT, float>();
        return;
    }
    LocalTensor<float> inputB = GetInputB(slot);
    LocalTensor<float> transA = GetTransposeBuffer(slot);
    const uint32_t planeElements = tileRows_ * tileCols_;
    if (opA_ == ACLBLAS_OP_N || opB_ == ACLBLAS_OP_N) {
        if (opA_ == ACLBLAS_OP_N) {
            if (opB_ == ACLBLAS_OP_C) {
                Muls(transA[planeElements], transA[planeElements], -1.0f, planeElements);
                PipeBarrier<PIPE_V>();
            }
            Interleave(inputB, inputB[planeElements], transA, transA[planeElements],
                static_cast<int32_t>(planeElements));
            PipeBarrier<PIPE_V>();
            Add(output, output, inputB, planeElements * COMPLEX_FLOAT_COUNT);
        } else {
            if (opA_ == ACLBLAS_OP_C) {
                Muls(transA[planeElements], transA[planeElements], -1.0f, planeElements);
                PipeBarrier<PIPE_V>();
            }
            Interleave(output, output[planeElements], transA, transA[planeElements],
                static_cast<int32_t>(planeElements));
            PipeBarrier<PIPE_V>();
            Add(output, output, inputB, planeElements * COMPLEX_FLOAT_COUNT);
        }
    } else if constexpr (MIXED_TRANSPOSE) {
        Interleave(output, output[planeElements], transA, transA[planeElements],
            static_cast<int32_t>(planeElements));
    } else {
        // The launcher routes mixed T/C pairs to MIXED_TRANSPOSE. With no N
        // operand here, the remaining dual-input operations are matching TT/CC.
        if (opA_ == ACLBLAS_OP_C) {
            Muls(transA[planeElements], transA[planeElements], -1.0f, planeElements);
            PipeBarrier<PIPE_V>();
        }
        Interleave(output, output[planeElements], transA, transA[planeElements],
            static_cast<int32_t>(planeElements));
    }
    inputOutputQueue_.template EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(output);
    output = inputOutputQueue_.template DeQue<QuePosition::VECCALC, QuePosition::VECOUT, float>();
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::StoreQueuedTile(
    uint32_t tileIdx, LocalTensor<float>& output)
{
    uint32_t rowTile = 0U;
    uint32_t colTile = 0U;
    DecodeTile(tileIdx, rowTile, colTile);
    const uint32_t rowBase = rowTile * tileRows_;
    const uint32_t colBase = colTile * tileCols_;
    const uint32_t validRows = (m_ - rowBase < tileRows_) ? (m_ - rowBase) : tileRows_;
    const uint32_t validCols = (n_ - colBase < tileCols_) ? (n_ - colBase) : tileCols_;
    if (validRows == tileRows_ && validCols == tileCols_) {
        const uint32_t fullRowBytes = tileRows_ * COMPLEX_FLOAT_COUNT * sizeof(float);
        const int64_t fullDstStride =
            static_cast<int64_t>(ldc_ - tileRows_) * sizeof(PackedComplex);
        const uint64_t fullOutputOffset =
            COMPLEX_FLOAT_COUNT * (static_cast<uint64_t>(colBase) * ldc_ + rowBase);
        DataCopyExtParams fullParams{
            static_cast<uint16_t>(tileCols_), fullRowBytes, 0, fullDstStride, 0};
        DataCopyPad(cGm_[fullOutputOffset], output, fullParams);
    } else {
        const uint32_t rowBytes = validRows * COMPLEX_FLOAT_COUNT * sizeof(float);
        const uint32_t fullUbRowBlocks =
            tileRows_ * COMPLEX_FLOAT_COUNT * sizeof(float) / BLOCK_BYTES;
        const uint32_t usedUbRowBlocks = (rowBytes + BLOCK_BYTES - 1U) / BLOCK_BYTES;
        const int64_t srcStride = static_cast<int64_t>(fullUbRowBlocks - usedUbRowBlocks);
        const int64_t dstStride = static_cast<int64_t>(ldc_ - validRows) * sizeof(PackedComplex);
        const uint64_t outputOffset =
            COMPLEX_FLOAT_COUNT * (static_cast<uint64_t>(colBase) * ldc_ + rowBase);
        DataCopyExtParams params{
            static_cast<uint16_t>(validCols), rowBytes, srcStride, dstStride, 0};
        DataCopyPad(cGm_[outputOffset], output, params);
    }
    inputOutputQueue_.FreeTensor(output);
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::ProcessShallowPipeline(
    uint32_t tileStart, uint32_t tileCount)
{
    uint32_t slot = 0U;
    LocalTensor<float> currentInput;
    const uint32_t firstTileIdx = tileStart;
    IssueQueuedPrefetch(firstTileIdx, slot, currentInput);
    inputOutputQueue_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(currentInput);
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t tileIdx = tileStart + tile;
        currentInput = inputOutputQueue_.template DeQue<QuePosition::VECIN, QuePosition::VECCALC, float>();
        TransformQueuedTile(tileIdx, slot, currentInput);
        const bool hasNext = tile + 1U < tileCount;
        LocalTensor<float> nextInput;
        if (hasNext) {
            const uint32_t nextTileIdx = tileStart + tile + 1U;
            IssueQueuedPrefetch(nextTileIdx, slot ^ 1U, nextInput);
        }
        ComputeQueuedTile(currentInput, slot);
        StoreQueuedTile(tileIdx, currentInput);
        if (hasNext) {
            inputOutputQueue_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(nextInput);
        }
        slot ^= 1U;
    }
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::ProcessDeepPipeline(
    uint32_t tileStart, uint32_t tileCount)
{
    constexpr uint32_t prefetchDistance = PIPELINE_DEPTH - 1U;
    const uint32_t firstTileIdx = tileStart;
    LocalTensor<float> currentInput;
    IssueQueuedPrefetch(firstTileIdx, 0U, currentInput);
    inputOutputQueue_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(currentInput);
    currentInput = inputOutputQueue_.template DeQue<QuePosition::VECIN, QuePosition::VECCALC, float>();

    LocalTensor<float> nextInput;
    LocalTensor<float> followingInput;
    if (tileCount > 1U) {
        const uint32_t nextTileIdx = tileStart + 1U;
        IssueQueuedPrefetch(nextTileIdx, 1U, nextInput);
    }
    if (tileCount > prefetchDistance) {
        const uint32_t followingTileIdx = tileStart + prefetchDistance;
        IssueQueuedPrefetch(followingTileIdx, prefetchDistance, followingInput);
    }
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t slot = tile % PIPELINE_DEPTH;
        const uint32_t tileIdx = tileStart + tile;
        TransformQueuedTile(tileIdx, slot, currentInput);
        ComputeQueuedTile(currentInput, slot);
        StoreQueuedTile(tileIdx, currentInput);
        if (tile + 1U >= tileCount) {
            continue;
        }

        inputOutputQueue_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(nextInput);
        currentInput = inputOutputQueue_.template DeQue<QuePosition::VECIN, QuePosition::VECCALC, float>();
        if (tile + prefetchDistance < tileCount) {
            nextInput = followingInput;
        }
        const uint32_t prefetch = tile + PIPELINE_DEPTH;
        if (prefetch < tileCount) {
            const uint32_t prefetchTileIdx = tileStart + prefetch;
            const uint32_t prefetchSlot = prefetch % PIPELINE_DEPTH;
            IssueQueuedPrefetch(prefetchTileIdx, prefetchSlot, followingInput);
        }
    }
}

template <bool MIXED_TRANSPOSE, uint32_t PIPELINE_DEPTH, bool SINGLE_INPUT>
__aicore__ inline void CgeamTCBlockAIV<MIXED_TRANSPOSE, PIPELINE_DEPTH, SINGLE_INPUT>::Process()
{
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t tilesPerBlock = totalTiles_ / blockNum;
    const uint32_t extraBlocks = totalTiles_ % blockNum;
    const uint32_t tileStart =
        blockIdx * tilesPerBlock + (blockIdx < extraBlocks ? blockIdx : extraBlocks);
    const uint32_t tileCount = tilesPerBlock + (blockIdx < extraBlocks ? 1U : 0U);
    if (tileCount == 0U) {
        return;
    }
    if constexpr (QUEUE_DEPTH == SHALLOW_TRANSPOSE_PIPELINE_DEPTH) {
        ProcessShallowPipeline(tileStart, tileCount);
    } else {
        ProcessDeepPipeline(tileStart, tileCount);
    }
}

__simt_callee__ __aicore__ inline void LoadCgeamElement(
    __gm__ const uint64_t* input, uint32_t op, uint32_t row, uint32_t col, uint32_t ld,
    float& real, float& imag)
{
    const uint64_t index = op == ACLBLAS_OP_N ?
        static_cast<uint64_t>(col) * ld + row : static_cast<uint64_t>(row) * ld + col;
    PackedComplex value{};
    value.packed = input[index];
    real = value.components[0];
    imag = op == ACLBLAS_OP_C ? -value.components[1] : value.components[1];
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_LAUNCH_BOUND_THREADS) inline void CgeamElementwiseSimt(
    uint64_t totalComplex, uint32_t m, uint32_t lda, uint32_t ldb, uint32_t ldc,
    uint32_t opA, uint32_t opB, uint32_t alphaIsZero, uint32_t betaIsZero,
    float alphaR, float alphaI, float betaR, float betaI,
    __gm__ const uint64_t* a, __gm__ const uint64_t* b, __gm__ uint64_t* c)
{
    const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t threadCount = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint32_t row = static_cast<uint32_t>(thread % m);
    uint32_t col = static_cast<uint32_t>(thread / m);
    const uint32_t rowStep = static_cast<uint32_t>(threadCount % m);
    const uint32_t colStep = static_cast<uint32_t>(threadCount / m);
    for (uint64_t logical = thread; logical < totalComplex; logical += threadCount) {
        float resultR = 0.0f;
        float resultI = 0.0f;
        if (alphaIsZero == 0U) {
            float aR = 0.0f;
            float aI = 0.0f;
            LoadCgeamElement(a, opA, row, col, lda, aR, aI);
            resultR = alphaR * aR - alphaI * aI;
            resultI = alphaR * aI + alphaI * aR;
        }
        if (betaIsZero == 0U) {
            float bR = 0.0f;
            float bI = 0.0f;
            LoadCgeamElement(b, opB, row, col, ldb, bR, bI);
            resultR += betaR * bR - betaI * bI;
            resultI += betaR * bI + betaI * bR;
        }
        PackedComplex result{};
        result.components[0] = resultR;
        result.components[1] = resultI;
        c[static_cast<uint64_t>(col) * ldc + row] = result.packed;
        uint32_t nextRow = row + rowStep;
        col += colStep;
        if (nextRow >= m) {
            nextRow -= m;
            ++col;
        }
        row = nextRow;
    }
}

// Packed-complex contiguous path for C = A or C = A + B.
template <bool SINGLE_INPUT>
class CgeamNNFastAIV {
public:
    __aicore__ inline CgeamNNFastAIV() = default;
    __aicore__ inline void Init(
        GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void EnqueueInputs(LocalTensor<float>& aLocal, LocalTensor<float>& bLocal);
    __aicore__ inline void CopyAIn(
        uint64_t complexOffset, uint32_t complexCount, LocalTensor<float>& aLocal);
    __aicore__ inline void CopyBIn(
        uint64_t complexOffset, uint32_t complexCount, LocalTensor<float>& bLocal);
    __aicore__ inline void ComputeInPlace(
        LocalTensor<float>& aLocal, LocalTensor<float>& bLocal, uint32_t complexCount);
    __aicore__ inline void CopyOut(
        uint64_t complexOffset, uint32_t complexCount, LocalTensor<float>& output);

    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> cGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, NN_INPUT_A_QUEUE_DEPTH> inOutQueueA_;
    TQue<TPosition::VECIN, NN_INPUT_B_QUEUE_DEPTH> inQueueB_;

    uint64_t startComplex_ = 0;
    uint64_t complexCount_ = 0;
    uint32_t tileComplex_ = 0;
};

template <bool SINGLE_INPUT>
__aicore__ inline void CgeamNNFastAIV<SINGLE_INPUT>::Init(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, TPipe* pipe)
{
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B));
    cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C));
    if constexpr (!SINGLE_INPUT) {
        ConfigureL2Policy(aGm_, bGm_, cGm_, tiling);
    }
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    complexCount_ = tiling.fastPerCoreComplex;
    if (blockIdx < tiling.fastExtraBlockCores) {
        complexCount_ += COMPLEX_PER_BLOCK;
    }
    if (blockIdx == blockNum - 1U) {
        complexCount_ += tiling.fastTailComplex;
    }
    const uint32_t priorExtra = blockIdx < tiling.fastExtraBlockCores ?
        blockIdx : tiling.fastExtraBlockCores;
    startComplex_ = static_cast<uint64_t>(blockIdx) * tiling.fastPerCoreComplex +
        static_cast<uint64_t>(priorExtra) * COMPLEX_PER_BLOCK;
    tileComplex_ = tiling.fastTileComplex;

    const uint32_t tileBytes = tileComplex_ * COMPLEX_FLOAT_COUNT * sizeof(float);
    pipe->InitBuffer(inOutQueueA_, NN_INPUT_A_QUEUE_DEPTH, tileBytes);
    if constexpr (!SINGLE_INPUT) {
        pipe->InitBuffer(inQueueB_, NN_INPUT_B_QUEUE_DEPTH, tileBytes);
    }
}

__aicore__ inline void CopyContiguousInput(
    GlobalTensor<float>& input, const LocalTensor<float>& local,
    uint64_t complexOffset, uint32_t complexCount)
{
    const uint64_t floatOffset = complexOffset * COMPLEX_FLOAT_COUNT;
    const uint32_t floatCount = complexCount * COMPLEX_FLOAT_COUNT;
    const uint32_t remainder = floatCount % FLOATS_PER_BLOCK;
    if (remainder == 0U) {
        DataCopy(local, input[floatOffset], floatCount);
    } else {
        const uint32_t byteCount = floatCount * sizeof(float);
        const uint8_t rightPadding = static_cast<uint8_t>(FLOATS_PER_BLOCK - remainder);
        DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, rightPadding, 0.0f};
        DataCopyPad(local, input[floatOffset], copyParams, padParams);
    }
}

template <bool SINGLE_INPUT>
__aicore__ inline void CgeamNNFastAIV<SINGLE_INPUT>::CopyAIn(
    uint64_t complexOffset, uint32_t complexCount, LocalTensor<float>& aLocal)
{
    aLocal = inOutQueueA_.AllocTensor<float>();
    CopyContiguousInput(aGm_, aLocal, complexOffset, complexCount);
}

template <bool SINGLE_INPUT>
__aicore__ inline void CgeamNNFastAIV<SINGLE_INPUT>::CopyBIn(
    uint64_t complexOffset, uint32_t complexCount, LocalTensor<float>& bLocal)
{
    bLocal = inQueueB_.AllocTensor<float>();
    CopyContiguousInput(bGm_, bLocal, complexOffset, complexCount);
}

template <bool SINGLE_INPUT>
__aicore__ inline void CgeamNNFastAIV<SINGLE_INPUT>::ComputeInPlace(
    LocalTensor<float>& aLocal, LocalTensor<float>& bLocal, uint32_t complexCount)
{
    const uint32_t floatCount = complexCount * COMPLEX_FLOAT_COUNT;
    if constexpr (!SINGLE_INPUT) {
        Add(aLocal, aLocal, bLocal, floatCount);
    }
    inOutQueueA_.EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(aLocal);
    aLocal = inOutQueueA_.DeQue<QuePosition::VECCALC, QuePosition::VECOUT, float>();
    if constexpr (!SINGLE_INPUT) {
        inQueueB_.FreeTensor(bLocal);
    }
}

template <bool SINGLE_INPUT>
__aicore__ inline void CgeamNNFastAIV<SINGLE_INPUT>::CopyOut(
    uint64_t complexOffset, uint32_t complexCount, LocalTensor<float>& output)
{
    const uint64_t floatOffset = complexOffset * COMPLEX_FLOAT_COUNT;
    const uint32_t floatCount = complexCount * COMPLEX_FLOAT_COUNT;
    if (floatCount % FLOATS_PER_BLOCK == 0) {
        DataCopy(cGm_[floatOffset], output, floatCount);
    } else {
        DataCopyExtParams copyParams{1, floatCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(cGm_[floatOffset], output, copyParams);
    }
    inOutQueueA_.FreeTensor(output);
}

template <bool SINGLE_INPUT>
__aicore__ inline void CgeamNNFastAIV<SINGLE_INPUT>::EnqueueInputs(
    LocalTensor<float>& aLocal, LocalTensor<float>& bLocal)
{
    inOutQueueA_.EnQue<QuePosition::VECIN, QuePosition::VECCALC>(aLocal);
    if constexpr (!SINGLE_INPUT) {
        inQueueB_.EnQue<float>(bLocal);
    }
}

template <bool SINGLE_INPUT>
__aicore__ inline void CgeamNNFastAIV<SINGLE_INPUT>::Process()
{
    if (complexCount_ == 0 || tileComplex_ == 0) {
        return;
    }

    uint64_t remaining = complexCount_;
    uint32_t currentCount = remaining > tileComplex_ ? tileComplex_ : static_cast<uint32_t>(remaining);
    uint64_t currentOffset = startComplex_;
    LocalTensor<float> currentA;
    LocalTensor<float> currentB;
    CopyAIn(currentOffset, currentCount, currentA);
    if constexpr (!SINGLE_INPUT) {
        CopyBIn(currentOffset, currentCount, currentB);
    }
    EnqueueInputs(currentA, currentB);
    remaining -= currentCount;

    while (true) {
        currentA = inOutQueueA_.DeQue<QuePosition::VECIN, QuePosition::VECCALC, float>();
        if constexpr (!SINGLE_INPUT) {
            currentB = inQueueB_.DeQue<float>();
        }
        const uint32_t nextCount = remaining > tileComplex_ ?
            tileComplex_ : static_cast<uint32_t>(remaining);
        const bool hasNext = remaining > 0;
        const uint64_t nextOffset = currentOffset + currentCount;
        LocalTensor<float> nextA;
        LocalTensor<float> nextB;
        if (hasNext) {
            CopyAIn(nextOffset, nextCount, nextA);
        }

        ComputeInPlace(currentA, currentB, currentCount);
        if constexpr (!SINGLE_INPUT) {
            if (hasNext) {
                CopyBIn(nextOffset, nextCount, nextB);
            }
        }
        CopyOut(currentOffset, currentCount, currentA);

        if (!hasNext) {
            break;
        }
        EnqueueInputs(nextA, nextB);
        currentOffset = nextOffset;
        currentCount = nextCount;
        remaining -= nextCount;
    }
}

} // namespace

// Kernel entry points.
extern "C" __global__ __aicore__ void cgeam_kernel(GM_ADDR A, GM_ADDR B, GM_ADDR C, CgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.m == 0 || tiling.n == 0) {
        return;
    }

    if (tiling.transposeTilePath != CgeamTransposePath::NONE) {
        if (UseDeepTransposePipeline(tiling)) {
            TPipe pipe;
            CgeamTCBlockAIV<false, DEEP_TRANSPOSE_PIPELINE_DEPTH, false> op;
            op.Init(A, B, C, tiling, &pipe);
            op.Process();
        } else {
            TPipe pipe;
            CgeamTCBlockAIV<false, SHALLOW_TRANSPOSE_PIPELINE_DEPTH, false> op;
            op.Init(A, B, C, tiling, &pipe);
            op.Process();
        }
        return;
    }

    const uint64_t totalComplex = static_cast<uint64_t>(tiling.m) * tiling.n;
    asc_vf_call<CgeamElementwiseSimt>(
        dim3{tiling.simtThreads, 1, 1}, totalComplex, tiling.m, tiling.lda, tiling.ldb, tiling.ldc,
        static_cast<uint32_t>(tiling.opA), static_cast<uint32_t>(tiling.opB), tiling.alphaIsZero,
        tiling.betaIsZero, tiling.alphaR, tiling.alphaI, tiling.betaR, tiling.betaI,
        reinterpret_cast<__gm__ const uint64_t*>(A), reinterpret_cast<__gm__ const uint64_t*>(B),
        reinterpret_cast<__gm__ uint64_t*>(C));
}

extern "C" __global__ __aicore__ void cgeam_mixed_transpose_kernel(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, CgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (UseDeepTransposePipeline(tiling)) {
        TPipe pipe;
        CgeamTCBlockAIV<true, DEEP_TRANSPOSE_PIPELINE_DEPTH, false> op;
        op.Init(A, B, C, tiling, &pipe);
        op.Process();
    } else {
        TPipe pipe;
        CgeamTCBlockAIV<true, SHALLOW_TRANSPOSE_PIPELINE_DEPTH, false> op;
        op.Init(A, B, C, tiling, &pipe);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void cgeam_nn_fast_kernel(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, CgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgeamNNFastAIV<false> op;
    op.Init(A, B, C, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void cgeam_copy_nn_kernel(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, CgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgeamNNFastAIV<true> op;
    op.Init(A, B, C, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void cgeam_copy_transpose_kernel(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, CgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgeamTCBlockAIV<false, DEEP_TRANSPOSE_PIPELINE_DEPTH, true> op;
    op.Init(A, B, C, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void cgeam_merged_transpose_kernel(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, CgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgeamTCBlockAIV<false, MERGED_REUSE_PIPELINE_TAG, false> op;
    op.Init(A, B, C, tiling, &pipe);
    op.Process();
}

void cgeam_kernel_do(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, uint32_t numBlocks, aclrtStream stream)
{
    if (tiling.betaIsZero != 0U && tiling.fastPath != 0U) {
        cgeam_copy_nn_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tiling);
    } else if (tiling.betaIsZero != 0U && tiling.transposeTilePath != CgeamTransposePath::NONE) {
        cgeam_copy_transpose_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tiling);
    } else if (tiling.transposeTilePath == CgeamTransposePath::MERGED) {
        cgeam_merged_transpose_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tiling);
    } else if (tiling.fastPath != 0U) {
        cgeam_nn_fast_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tiling);
    } else if (tiling.transposeTilePath != CgeamTransposePath::NONE && tiling.opA != tiling.opB &&
        tiling.opA != ACLBLAS_OP_N && tiling.opB != ACLBLAS_OP_N) {
        cgeam_mixed_transpose_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tiling);
    } else {
        cgeam_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tiling);
    }
}
