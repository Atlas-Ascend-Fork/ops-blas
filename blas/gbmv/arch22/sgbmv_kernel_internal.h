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
 * \file sgbmv_kernel_internal.h
 * \brief Shared constants, inline helpers and cross-TU entry declarations for the
 * arch22 sgbmv kernel sources (sgbmv_kernel.cpp / sgbmv_kernel_vec.cpp / sgbmv_kernel_l1.cpp).
 */

#pragma once

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "common/compute/simd.h"
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/memory/mem.h"
#include "kernel_operator.h"
#include "sgbmv_kernel.h"
#include "sgbmv_tiling_data.h"

using AscendC::CMPMODE;
using AscendC::GatherRepeatParams;
using AscendC::ReduceType;
using AscendC::RoundMode;
using AscendC::SELMODE;
using AscendC::Abs;
using AscendC::Add;
using AscendC::Adds;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::Duplicate;
using AscendC::Gather;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Max;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::ReduceDataBlock;
using AscendC::SyncAll;
using AscendC::WholeReduceSum;

constexpr uint32_t BYTES_PER_FLOAT = sizeof(float);
constexpr uint32_t CORE_ALIGN_SIZE = 32U;
constexpr uint32_t VECTOR_ALIGN_MASK = 31U;
constexpr uint32_t SIMD_WIDTH = 8U;
constexpr uint32_t UNROLL_STEP = 8U;

constexpr uint32_t MAX_FAST_LDA = 256U;
constexpr uint32_t MAX_UB_A_BYTES = 160000U;
constexpr uint32_t FAST_TILE_M = 128U;
constexpr uint32_t FAST_COL_BATCH = 64U;
constexpr uint32_t MIN_GATHER_BANDWIDTH = 33U;
constexpr uint32_t MAX_GATHER_BANDWIDTH = 129U;
// Small-band fast paths (band <= 32) run the scalar row/column loops, which are
// scalar-pipe bound (utilization 0.88-0.97). Below this threshold they switch to
// a "diagonal" vector formulation: y[lane] += AB(j, laneIndex(j)) * x[laneIndex(j)] with one
// strided Gather per band row j, moving the work to the vector pipe.
constexpr uint32_t MAX_SMALLBAND_VEC = 32U;
// Zero-margin columns staged below colStartAligned on the N side: spurious
// diagonal reads reach at most kl-1 <= 31 columns below colStart (only the
// first tile of the matrix), so 32 zero columns cover every lane.
constexpr uint32_t SMALLBAND_A_MARGIN_COLS = 32U;

// Byte offsets in the 192 KiB UB. Gather rows need 32-byte alignment;
// eight rows of at most 136 floats occupy 4352 bytes per operand/table.
constexpr uint32_t GATHER_PADDED_MAX = (MAX_GATHER_BANDWIDTH + SIMD_WIDTH - 1U) / SIMD_WIDTH * SIMD_WIDTH;
constexpr uint32_t GATHER_WORK_BYTES = SIMD_WIDTH * GATHER_PADDED_MAX * BYTES_PER_FLOAT;
constexpr uint32_t UB_OFFSET_Y = 0U;
constexpr uint32_t UB_OFFSET_Y_ORIG = 1024U;
constexpr uint32_t UB_OFFSET_EDGE_MASK = 1792U;
constexpr uint32_t UB_OFFSET_X = 2048U;
constexpr uint32_t UB_OFFSET_SUMS = 4096U;
constexpr uint32_t UB_OFFSET_A_INDICES = 8192U;
constexpr uint32_t UB_OFFSET_X_INDICES = UB_OFFSET_A_INDICES + GATHER_WORK_BYTES;
constexpr uint32_t UB_OFFSET_A_VALUES = UB_OFFSET_X_INDICES + GATHER_WORK_BYTES;
constexpr uint32_t UB_OFFSET_X_VALUES = UB_OFFSET_A_VALUES + GATHER_WORK_BYTES;
constexpr uint32_t UB_OFFSET_EDGE_INDICES = UB_OFFSET_X_VALUES + GATHER_WORK_BYTES;
// Linear step-4 gather table (64-128 entries) and float scratch for the
// small-band paths; both live in the tail of the edge-indices region.
constexpr uint32_t UB_OFFSET_TBL4 = UB_OFFSET_EDGE_INDICES + GATHER_WORK_BYTES;
constexpr uint32_t UB_OFFSET_SB_SCRATCH = UB_OFFSET_EDGE_INDICES;
constexpr uint32_t UB_OFFSET_A = 32768U;
static_assert(UB_OFFSET_EDGE_INDICES + GATHER_WORK_BYTES <= UB_OFFSET_A, "Gather tables must not overlap A");
static_assert(
    UB_OFFSET_Y_ORIG + GATHER_PADDED_MAX * BYTES_PER_FLOAT <= UB_OFFSET_EDGE_MASK,
    "Gatherb offsets must not overlap the edge mask");
static_assert(UB_OFFSET_EDGE_MASK + SIMD_WIDTH * GATHER_PADDED_MAX / 8U <= UB_OFFSET_X, "Edge mask must not overlap x");
static_assert(UB_OFFSET_A + MAX_UB_A_BYTES <= 192U * 1024U, "SGBMV buffers must fit in UB");
static_assert(UB_OFFSET_TBL4 + 128U * BYTES_PER_FLOAT <= UB_OFFSET_A, "tbl4 must fit before the A region");

template <typename T>
__aicore__ inline T MinVal(T a, T b)
{
    return (a < b) ? a : b;
}

template <typename T>
__aicore__ inline T MaxVal(T a, T b)
{
    return (a > b) ? a : b;
}

__aicore__ inline void WriteY(
    uint32_t outIdx, uint32_t dim, int64_t incy, float alpha, float beta, float acc, __gm__ float* yGm)
{
    int64_t yIdx =
        (incy >= 0) ? (static_cast<int64_t>(outIdx) * incy) : (static_cast<int64_t>(dim - 1U - outIdx) * (-incy));
    float alphaTerm = (alpha == 0.0f) ? 0.0f : (alpha * acc);
    yGm[yIdx] = (beta == 0.0f) ? alphaTerm : (alphaTerm + beta * yGm[yIdx]);
}

__aicore__ inline void CopyGmToUb(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, uint32_t len)
{
    uint32_t lenBurst = len * BYTES_PER_FLOAT;
    uint8_t rightPaddingNum = static_cast<uint8_t>((SIMD_WIDTH - (len % SIMD_WIDTH)) % SIMD_WIDTH);
    AscendC::DataCopyPad(
        dst, src, AscendC::DataCopyExtParams(1, lenBurst, 0, 0, 0),
        AscendC::DataCopyPadExtParams<float>(false, 0, rightPaddingNum, 0.0f));
}

__aicore__ inline void CopyUbToGm(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src, uint32_t len)
{
    uint32_t lenBurst = len * BYTES_PER_FLOAT;
    AscendC::DataCopyPad(dst, src, AscendC::DataCopyExtParams(1, lenBurst, 0, 0, 0));
}

__aicore__ inline void CalcCoreRowRange(
    uint32_t outDim, uint32_t numBlocks, uint32_t blkIdx, uint32_t& startRow, uint32_t& endRow,
    uint32_t rowAlignment = CORE_ALIGN_SIZE)
{
    startRow = 0U;
    endRow = outDim;
    if (rowAlignment == 0U || numBlocks == 0U) {
        return;
    }
    if (numBlocks <= 1U) {
        return;
    }
    // A contiguous FP32 DMA write needs eight elements (32 bytes) of alignment.
    // Keep scalar GM writers on their original, wider core boundaries.
    if (rowAlignment == 0U) {
        return;
    }
    uint32_t numChunks = outDim / rowAlignment;
    if (numBlocks == 0U) {
        return;
    }
    uint32_t chunksPerCore = numChunks / numBlocks;
    uint32_t remChunks = numChunks % numBlocks;
    uint32_t startChunk = 0U;
    uint32_t myChunks = 0U;
    if (blkIdx < remChunks) {
        startChunk = blkIdx * (chunksPerCore + 1U);
        myChunks = chunksPerCore + 1U;
    } else {
        startChunk = remChunks * (chunksPerCore + 1U) + (blkIdx - remChunks) * chunksPerCore;
        myChunks = chunksPerCore;
    }
    startRow = startChunk * rowAlignment;
    endRow = (blkIdx == numBlocks - 1U) ? outDim : (startRow + myChunks * rowAlignment);
}

__aicore__ inline void ProcessAlphaZero(
    uint32_t startRow, uint32_t endRow, uint32_t outDim, int64_t incy, float beta, __gm__ float* yGm)
{
    for (uint32_t outIdx = startRow; outIdx < endRow; ++outIdx) {
        int64_t yIdx = (incy >= 0) ? (static_cast<int64_t>(outIdx) * incy) :
                                     (static_cast<int64_t>(outDim - 1U - outIdx) * (-incy));
        yGm[yIdx] = (beta == 0.0f) ? 0.0f : (beta * yGm[yIdx]);
    }
}

// Vector writeback whose vector-op counts are the caller's 8-aligned lane count
// n8 (>= numRows): lanes in [numRows, n8) are scratch and never leave UB. The
// legacy WriteBackContiguousVector passes raw numRows into vector API counts,
// which is only exercised with 8-multiple row counts on the gather path; the
// small-band paths produce arbitrary tail counts, so they use this variant.
__aicore__ inline void WriteBackContiguousVec8(
    uint32_t tileStart, uint32_t numRows, uint32_t n8, float alpha, float beta, GlobalTensor<float>& yGlobal,
    LocalTensor<float>& ubY, AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    PIPE_BARRIER(V);
    if (alpha != 1.0f) {
        Muls(ubY, ubY, alpha, n8);
        PIPE_BARRIER(V);
    }
    if (beta != 0.0f) {
        auto original = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_Y_ORIG);
        uint32_t lenBurst = numRows * BYTES_PER_FLOAT;
        uint8_t rightPad = static_cast<uint8_t>((SIMD_WIDTH - (numRows % SIMD_WIDTH)) % SIMD_WIDTH);
        AscendC::DataCopyPad(
            original, yGlobal[tileStart], AscendC::DataCopyExtParams(1, lenBurst, 0, 0, 0),
            AscendC::DataCopyPadExtParams<float>(true, 0, rightPad, 0.0f));
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        Muls(original, original, beta, n8);
        PIPE_BARRIER(V);
        Add(ubY, ubY, original, n8);
        PIPE_BARRIER(V);
    }
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
    CopyUbToGm(yGlobal[tileStart], ubY, numRows);
    SET_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    PIPE_BARRIER(ALL);
}

// Seed one aligned block on the scalar pipeline, then extend the arithmetic
// progression by doubling vector copies. No implicit TPipe event allocation.
__aicore__ inline void MakeOffsetSequence(LocalTensor<int32_t> values, uint32_t count, int32_t step)
{
    auto ptr = reinterpret_cast<__ubuf__ int32_t*>(values.GetPhyAddr());
    for (uint32_t k = 0; k < SIMD_WIDTH; ++k) {
        ptr[k] = static_cast<int32_t>(k) * step;
    }
    SET_FLAG(S, V, EVENT_ID0);
    WAIT_FLAG(S, V, EVENT_ID0);
    for (uint32_t filled = SIMD_WIDTH; filled < count; filled *= 2U) {
        Adds(values[filled], values, static_cast<int32_t>(filled) * step, MinVal(filled, count - filled));
        PIPE_BARRIER(V);
    }
}

constexpr uint32_t VEC_L_MAX = 512U;      // N row subtile cap (floats)
constexpr uint32_t VEC_T_L_MAX = 256U;    // T column subtile cap
constexpr uint32_t VEC_ROW_MIN = 32U;     // min subtile height for vector N
constexpr uint32_t VEC_XWIN_ELEMS = 8512U; // x window region (floats, 8-mult)
constexpr uint32_t VEC_ABUF_ELEMS = 6144U; // per ping-pong A half (floats)
constexpr uint32_t VEC_ROWP_MAX = 4096U;  // T pow2 row cap (band <= 4096)
constexpr uint32_t VEC_ZERO_SLOT = VEC_ROWP_MAX - 8U; // zero slot index inside ubT region

// DataCopyPad GM->UB that guarantees zero fill up to 8-element alignment.
__aicore__ inline void VecCopyIn(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t len)
{
    uint32_t lenBurst = len * BYTES_PER_FLOAT;
    uint8_t rightPad = static_cast<uint8_t>((SIMD_WIDTH - (len % SIMD_WIDTH)) % SIMD_WIDTH);
    AscendC::DataCopyPad(
        dst, src, AscendC::DataCopyExtParams(1, lenBurst, 0, 0, 0),
        AscendC::DataCopyPadExtParams<float>(true, 0, rightPad, 0.0f));
}

// Smallest 8 * 2^k >= v (v >= 1). Branch form: the while-loop version costs
// ~10 scalar iterations and is called per column, which measurably regressed
// the wide-band TVec cases.
__aicore__ inline uint32_t VecPow2RoundUp(uint32_t v)
{
    if (v > 512U) {
        return (v > 2048U) ? ((v > 4096U) ? 8192U : 4096U) : ((v > 1024U) ? 2048U : 1024U);
    }
    if (v > 64U) {
        return (v > 256U) ? 512U : ((v > 128U) ? 256U : 128U);
    }
    return (v > 32U) ? 64U : ((v > 16U) ? 32U : ((v > 8U) ? 16U : SIMD_WIDTH));
}

// Ping-pong MTE2->V event signalling for the two-slot staging rings: the slot
// of parity 0 signals/waits EVENT_ID0, the slot of parity 1 EVENT_ID1. The
// issue direction signals the NEXT slot's id before waiting the CURRENT
// slot's id (batch rings in the vec/l1 tile loops).
__aicore__ inline void SetMte2SlotReady(uint32_t slotParity)
{
    if (slotParity == 0U) {
        SET_FLAG(MTE2, V, EVENT_ID0);
    } else {
        SET_FLAG(MTE2, V, EVENT_ID1);
    }
}

__aicore__ inline void WaitMte2SlotReady(uint32_t slotParity)
{
    if (slotParity == 0U) {
        WAIT_FLAG(MTE2, V, EVENT_ID0);
    } else {
        WAIT_FLAG(MTE2, V, EVENT_ID1);
    }
}

__aicore__ inline bool UseVecFallbackN(const SgbmvTilingData& tiling)
{
    if (tiling.trans != 0U || tiling.incx != 1 || tiling.incy != 1) {
        return false;
    }
    if (tiling.lda < VEC_ROW_MIN + 1U) {
        return false;
    }
    uint32_t band = tiling.kl + tiling.ku + 1U;
    return band + VEC_L_MAX + 8U <= VEC_XWIN_ELEMS;
}

__aicore__ inline bool UseVecFallbackT(const SgbmvTilingData& tiling)
{
    if (tiling.trans == 0U || tiling.incx != 1 || tiling.incy != 1) {
        return false;
    }
    uint32_t band = tiling.kl + tiling.ku + 1U;
    uint32_t band8 = (band + SIMD_WIDTH - 1U) / SIMD_WIDTH * SIMD_WIDTH;
    uint32_t rowP = VecPow2RoundUp(band8);
    if (rowP > VEC_ROWP_MAX) {
        return false;
    }
    return band + VEC_T_L_MAX + rowP + 8U <= VEC_XWIN_ELEMS;
}

__aicore__ inline bool IsFastpathN(const SgbmvTilingData& tiling)
{
    if (tiling.trans != 0U || tiling.incx != 1 || tiling.incy != 1) {
        return false;
    }
    if (tiling.lda < (tiling.kl + tiling.ku + 1U) || tiling.lda > MAX_FAST_LDA) {
        return false;
    }
    uint32_t aBytes = tiling.lda * ((FAST_TILE_M + tiling.kl + tiling.ku + 63U) / CORE_ALIGN_SIZE * CORE_ALIGN_SIZE) *
                      BYTES_PER_FLOAT;
    return aBytes <= MAX_UB_A_BYTES;
}

__aicore__ inline bool IsFastpathT(const SgbmvTilingData& tiling)
{
    return (
        tiling.trans != 0U && tiling.incx == 1 && tiling.incy == 1 && tiling.lda >= (tiling.kl + tiling.ku + 1U) &&
        tiling.lda <= MAX_FAST_LDA);
}

// ============================================================================
// Cross-translation-unit entries: each is defined exactly once across the
// sgbmv kernel sources and called from a sibling translation unit, so they
// carry external linkage (definitions live in the file noted per declaration).
// ============================================================================
// sgbmv_kernel.cpp: scalar N fallback (also reached from the vector N path).
__aicore__ void ProcessSgbmvNFallback(
    uint32_t startRow, uint32_t endRow, const SgbmvTilingData& tiling, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm);
// sgbmv_kernel.cpp: top-level compute dispatch (also used by the L1 entry).
__aicore__ void DispatchSgbmvCompute(
    uint32_t startRow, uint32_t endRow, uint32_t outDim, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm, bool fastN);
// sgbmv_kernel_vec.cpp: small-band staged-diagonal fast tile (N fast path).
__aicore__ bool ProcessSgbmvNFastTileSmallBand(
    uint32_t tileStart, uint32_t tileEnd, uint32_t colStartAligned, uint32_t colEndAligned,
    const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal, GlobalTensor<float>& xGlobal,
    GlobalTensor<float>& yGlobal, uint64_t aGmOff, LocalTensor<float>& ubA, LocalTensor<float>& ubY,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf);
// sgbmv_kernel_vec.cpp: vector N fallback driver (wide-band row-split path).
__aicore__ void ProcessSgbmvNVec(
    uint32_t startRow, uint32_t endRow, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm);
// sgbmv_kernel_vec.cpp: vector T fallback driver (wide-band column-dot path).
__aicore__ void ProcessSgbmvTVec(
    uint32_t startCol, uint32_t endCol, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y);
// sgbmv_kernel_vec.cpp: scalar T fallback (strided/contiguous column dots).
__aicore__ void ProcessSgbmvTFallback(
    uint32_t startRow, uint32_t endRow, const SgbmvTilingData& tiling, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm);
