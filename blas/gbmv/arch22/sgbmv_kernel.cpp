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
 * \file sgbmv_kernel.cpp
 * \brief Single-precision general banded matrix-vector multiply for Atlas A2/A3 (arch22).
 *
 * Fast staged paths, scalar fallbacks and the kernel entry points. The vector
 * fallback family lives in sgbmv_kernel_vec.cpp and the L1 column-split path in
 * sgbmv_kernel_l1.cpp; shared helpers are in sgbmv_kernel_internal.h.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "common/compute/simd.h"
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/memory/mem.h"
#include "kernel_operator.h"
#include "sgbmv_kernel.h"
#include "sgbmv_tiling_data.h"
#include "sgbmv_kernel_internal.h"

#include "sgbmv_kernel_vec.cpp"
#include "sgbmv_kernel_l1.cpp"

// Signed floor to a multiple of 8 (negative-safe).
__aicore__ inline int32_t Floor8Signed(int32_t v)
{
    return (v >= 0) ? (v & ~static_cast<int32_t>(SIMD_WIDTH - 1U))
                    : -((-v + static_cast<int32_t>(SIMD_WIDTH - 1U)) & ~static_cast<int32_t>(SIMD_WIDTH - 1U));
}

__aicore__ inline void WriteBackContiguousVector(
    uint32_t tileStart, uint32_t numRows, float alpha, float beta, GlobalTensor<float>& yGlobal,
    LocalTensor<float>& ubY, AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    PIPE_BARRIER(V);
    if (alpha != 1.0f) {
        Muls(ubY, ubY, alpha, numRows);
    }
    if (beta != 0.0f) {
        auto original = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_Y_ORIG);
        // This region previously held Gatherb offsets; finish their vector
        // reads before the next DMA overwrites it with the original y.
        SET_FLAG(V, MTE2, EVENT_ID1);
        WAIT_FLAG(V, MTE2, EVENT_ID1);
        CopyGmToUb(original, yGlobal[tileStart], numRows);
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        Muls(original, original, beta, numRows);
        PIPE_BARRIER(V);
        Add(ubY, ubY, original, numRows);
    }
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
    CopyUbToGm(yGlobal[tileStart], ubY, numRows);
    SET_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    PIPE_BARRIER(ALL);
}

__aicore__ inline void WriteBackContiguousScalar(
    uint32_t tileStart, uint32_t numRows, float alpha, float beta, GlobalTensor<float>& yGlobal,
    LocalTensor<float>& ubY, AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    auto yPtr = reinterpret_cast<__ubuf__ float*>(ubY.GetPhyAddr());
    if (beta == 0.0f) {
        if (alpha != 1.0f) {
            for (uint32_t i = 0U; i < numRows; ++i) {
                yPtr[i] = alpha * yPtr[i];
            }
        }
    } else {
        LocalTensor<float> ubYOrig = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_Y_ORIG);
        auto yOrigPtr = reinterpret_cast<__ubuf__ float*>(ubYOrig.GetPhyAddr());
        CopyGmToUb(ubYOrig, yGlobal[tileStart], numRows);
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        SET_FLAG(MTE2, S, EVENT_ID1);
        WAIT_FLAG(MTE2, S, EVENT_ID1);
        PIPE_BARRIER(ALL);
        for (uint32_t i = 0U; i < numRows; ++i) {
            yPtr[i] = alpha * yPtr[i] + beta * yOrigPtr[i];
        }
    }

    PIPE_BARRIER(ALL);
    SET_FLAG(S, MTE3, EVENT_ID0);
    WAIT_FLAG(S, MTE3, EVENT_ID0);
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);

    // DataCopyPad also handles the final partial block. Avoid scalar GM stores
    // on that tail, whose cache line may be shared with another core's DMA.
    CopyUbToGm(yGlobal[tileStart], ubY, numRows);
    SET_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    PIPE_BARRIER(ALL);
}

__aicore__ inline void WriteBackContiguous(
    uint32_t tileStart, uint32_t numRows, float alpha, float beta, GlobalTensor<float>& yGlobal,
    LocalTensor<float>& ubY, AsdopsBuffer<ArchType::ASCEND_V220>& buf, bool vectorOutput = false)
{
    if (vectorOutput) {
        WriteBackContiguousVector(tileStart, numRows, alpha, beta, yGlobal, ubY, buf);
    } else {
        WriteBackContiguousScalar(tileStart, numRows, alpha, beta, yGlobal, ubY, buf);
    }
}

__aicore__ inline float FastNComputeRow(
    uint32_t row, uint32_t colStartAligned, uint32_t n, uint32_t ku, uint32_t kl, uint32_t lda,
    __ubuf__ const float* __restrict__ aPtr, __ubuf__ const float* __restrict__ xPtr)
{
    uint32_t cStart = (row >= kl) ? (row - kl) : 0U;
    uint32_t cEnd = (row + ku < n) ? (row + ku + 1U) : n;
    if (cStart >= cEnd) {
        return 0.0f;
    }
    uint32_t len = cEnd - cStart;
    int32_t aRowOff = static_cast<int32_t>(ku + row) - static_cast<int32_t>(cStart);
    auto aCol = aPtr + static_cast<int64_t>(cStart - colStartAligned) * lda + aRowOff;
    auto xCol = xPtr + (cStart - colStartAligned);
    uint32_t stride = lda - 1U;
    uint32_t step = stride * 8U;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    uint32_t k = 0U;
    for (; k + 7U < len; k += UNROLL_STEP) {
        acc0 += aCol[0] * xCol[0];
        acc1 += aCol[stride] * xCol[1];
        acc2 += aCol[stride * 2U] * xCol[2];
        acc3 += aCol[stride * 3U] * xCol[3];
        acc4 += aCol[stride * 4U] * xCol[4];
        acc5 += aCol[stride * 5U] * xCol[5];
        acc6 += aCol[stride * 6U] * xCol[6];
        acc7 += aCol[stride * 7U] * xCol[7];
        aCol += step;
        xCol += UNROLL_STEP;
    }
    float acc = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
    for (; k < len; ++k) {
        acc += (*aCol) * (*xCol++);
        aCol += stride;
    }
    return acc;
}

__aicore__ inline void FastNCompute8RowsPrefix(
    uint32_t rBase, uint32_t colStartAligned, uint32_t kl, uint32_t ku, uint32_t lda, uint32_t cMin, uint32_t cMidStart,
    __ubuf__ const float* __restrict__ aPtr, __ubuf__ const float* __restrict__ xPtr, float* __restrict__ y)
{
    for (uint32_t c = cMin; c < cMidStart; ++c) {
        float xc = xPtr[c - colStartAligned];
        if (xc == 0.0f) {
            continue;
        }
        uint32_t kEnd = MinVal(c + kl - rBase + 1U, 8U);
        int32_t aRowOff = static_cast<int32_t>(ku + rBase) - static_cast<int32_t>(c);
        auto aCol = aPtr + static_cast<int64_t>(c - colStartAligned) * lda + aRowOff;
        for (uint32_t k = 0U; k < kEnd; ++k) {
            y[k] += aCol[k] * xc;
        }
    }
}

__aicore__ inline void FastNCompute8RowsMid(
    uint32_t rBase, uint32_t colStartAligned, uint32_t ku, uint32_t lda, uint32_t cMidStart, uint32_t cMidEnd,
    __ubuf__ const float* __restrict__ aPtr, __ubuf__ const float* __restrict__ xPtr, float* __restrict__ y)
{
    int32_t aStartRowOff = static_cast<int32_t>(ku + rBase) - static_cast<int32_t>(cMidStart);
    auto aCol = aPtr + static_cast<int64_t>(cMidStart - colStartAligned) * lda + aStartRowOff;
    auto xCol = xPtr + (cMidStart - colStartAligned);
    uint32_t stride = lda - 1U;
    uint32_t stride2 = stride * 2U;

    uint32_t c = cMidStart;
    for (; c + 1U < cMidEnd; c += 2U) {
        float xc0 = xCol[0];
        float xc1 = xCol[1];
        auto aCol0 = aCol;
        auto aCol1 = aCol + stride;
        for (uint32_t k = 0U; k < 8U; ++k) {
            y[k] += aCol0[k] * xc0 + aCol1[k] * xc1;
        }
        aCol += stride2;
        xCol += 2U;
    }
    if (c < cMidEnd) {
        float xc = *xCol++;
        for (uint32_t k = 0U; k < 8U; ++k) {
            y[k] += aCol[k] * xc;
        }
    }
}

__aicore__ inline void FastNCompute8RowsSuffix(
    uint32_t rBase, uint32_t colStartAligned, uint32_t ku, uint32_t lda, uint32_t cMidEnd, uint32_t cMax,
    __ubuf__ const float* __restrict__ aPtr, __ubuf__ const float* __restrict__ xPtr, float* __restrict__ y)
{
    for (uint32_t cRem = cMidEnd; cRem < cMax; ++cRem) {
        float xc = xPtr[cRem - colStartAligned];
        if (xc == 0.0f) {
            continue;
        }
        uint32_t kStart = (cRem > ku + rBase) ? (cRem - ku - rBase) : 0U;
        int32_t aRowOff = static_cast<int32_t>(ku + rBase) - static_cast<int32_t>(cRem);
        auto aColRem = aPtr + static_cast<int64_t>(cRem - colStartAligned) * lda + aRowOff;
        for (uint32_t k = kStart; k < 8U; ++k) {
            y[k] += aColRem[k] * xc;
        }
    }
}

__aicore__ inline void FastNCompute8RowsBoundary(
    uint32_t rBase, uint32_t colStartAligned, uint32_t kl, uint32_t ku, uint32_t lda, uint32_t cMin, uint32_t cMax,
    __ubuf__ const float* __restrict__ aPtr, __ubuf__ const float* __restrict__ xPtr, float* __restrict__ y)
{
    for (uint32_t c = cMin; c < cMax; ++c) {
        float xc = xPtr[c - colStartAligned];
        if (xc == 0.0f) {
            continue;
        }
        int32_t kStartSigned = static_cast<int32_t>(c) - static_cast<int32_t>(ku + rBase);
        uint32_t kStart = (kStartSigned > 0) ? static_cast<uint32_t>(kStartSigned) : 0U;
        int32_t kEndSigned = static_cast<int32_t>(c + kl + 1U) - static_cast<int32_t>(rBase);
        uint32_t kEnd = (kEndSigned > 0) ? MinVal(static_cast<uint32_t>(kEndSigned), 8U) : 0U;
        int32_t aRowOff = static_cast<int32_t>(ku + rBase) - static_cast<int32_t>(c);
        auto aCol = aPtr + static_cast<int64_t>(c - colStartAligned) * lda + aRowOff;
        for (uint32_t k = kStart; k < kEnd; ++k) {
            y[k] += aCol[k] * xc;
        }
    }
}

__aicore__ inline void FastNCompute8Rows(
    uint32_t rBase, uint32_t colStartAligned, uint32_t n, uint32_t ku, uint32_t kl, uint32_t lda,
    __ubuf__ const float* __restrict__ aPtr, __ubuf__ const float* __restrict__ xPtr, __ubuf__ float* __restrict__ yDst)
{
    constexpr uint32_t B = 8U;
    uint32_t cMin = (rBase >= kl) ? (rBase - kl) : 0U;
    uint32_t cMax = MinVal(rBase + 7U + ku + 1U, n);
    if (cMin >= cMax) {
        for (uint32_t i = 0U; i < B; ++i) {
            yDst[i] = 0.0f;
        }
        return;
    }

    float y[8] = {0.0f};
    uint32_t cMidStart = (rBase + 7U >= kl) ? (rBase + 7U - kl) : 0U;
    uint32_t cMidEnd = (rBase + ku < n) ? (rBase + ku + 1U) : n;

    if (cMidStart < cMidEnd) {
        FastNCompute8RowsPrefix(rBase, colStartAligned, kl, ku, lda, cMin, cMidStart, aPtr, xPtr, y);
        FastNCompute8RowsMid(rBase, colStartAligned, ku, lda, cMidStart, cMidEnd, aPtr, xPtr, y);
        FastNCompute8RowsSuffix(rBase, colStartAligned, ku, lda, cMidEnd, cMax, aPtr, xPtr, y);
    } else {
        FastNCompute8RowsBoundary(rBase, colStartAligned, kl, ku, lda, cMin, cMax, aPtr, xPtr, y);
    }

    for (uint32_t i = 0U; i < B; ++i) {
        yDst[i] = y[i];
    }
}

__aicore__ inline void FastNComputeTileRows(
    uint32_t tileStart, uint32_t tileEnd, uint32_t colStartAligned, const SgbmvTilingData& tiling,
    __ubuf__ const float* __restrict__ aPtr, __ubuf__ const float* __restrict__ xPtr, __ubuf__ float* __restrict__ yPtr)
{
    uint32_t r = tileStart;
    for (; r + 8U <= tileEnd; r += 8U) {
        FastNCompute8Rows(
            r, colStartAligned, tiling.n, tiling.ku, tiling.kl, tiling.lda, aPtr, xPtr, yPtr + (r - tileStart));
    }
    for (; r < tileEnd; ++r) {
        yPtr[r - tileStart] =
            FastNComputeRow(r, colStartAligned, tiling.n, tiling.ku, tiling.kl, tiling.lda, aPtr, xPtr);
    }
}

template <bool TRANSPOSE>
__aicore__ inline void InitGatherOffsets(
    LocalTensor<int32_t>& aIndices, LocalTensor<int32_t>& xIndices, LocalTensor<float>& edgeIndices,
    LocalTensor<uint32_t>& blockOffsets, uint32_t bandwidth, uint32_t padded, uint32_t blockColumns,
    const SgbmvTilingData& tiling, bool useBlocks, bool maskEdges)
{
    MakeOffsetSequence(aIndices, padded, 1);
    Mins(aIndices, aIndices, static_cast<int32_t>(bandwidth - 1U), padded);
    PIPE_BARRIER(V);
    if (!useBlocks) {
        Muls(xIndices, aIndices, static_cast<int32_t>(BYTES_PER_FLOAT), padded);
        PIPE_BARRIER(V);
    }
    int32_t aStep =
        static_cast<int32_t>((TRANSPOSE ? 1U : (useBlocks ? SIMD_WIDTH : tiling.lda - 1U)) * BYTES_PER_FLOAT);
    Muls(aIndices, aIndices, aStep, padded);
    if (useBlocks) {
        MakeOffsetSequence(
            blockOffsets.ReinterpretCast<int32_t>(), blockColumns,
            static_cast<int32_t>((tiling.lda - 1U) * BYTES_PER_FLOAT));
        MakeOffsetSequence(xIndices, 64U, 1);
        ShiftRight(xIndices, xIndices, static_cast<int32_t>(3), 64U);
        PIPE_BARRIER(V);
        Muls(xIndices, xIndices, static_cast<int32_t>(BYTES_PER_FLOAT), 64U);
        PIPE_BARRIER(V);
        for (uint32_t k = 1; k < blockColumns / SIMD_WIDTH; ++k) {
            Adds(xIndices[k * 64U], xIndices, static_cast<int32_t>(k * SIMD_WIDTH * BYTES_PER_FLOAT), 64U);
        }
    }
    PIPE_BARRIER(V);
    if (maskEdges) {
        auto integerIndices = edgeIndices.ReinterpretCast<int32_t>();
        MakeOffsetSequence(integerIndices, padded, 1);
        Cast(edgeIndices, integerIndices, RoundMode::CAST_NONE, padded);
        PIPE_BARRIER(V);
    }
}

template <bool TRANSPOSE>
__aicore__ inline void UpdateMultiRowOffsets(
    LocalTensor<int32_t>& aIndices, LocalTensor<int32_t>& xIndices, LocalTensor<float>& edgeIndices, uint32_t rows,
    uint32_t padded, bool left, bool useBlocks, bool maskEdges, const SgbmvTilingData& tiling)
{
    for (uint32_t r = 1; r < rows; ++r) {
        int32_t aRowStep = static_cast<int32_t>(
            r * (left ? (TRANSPOSE ? tiling.lda - 1U : 1U) : (useBlocks ? SIMD_WIDTH + 1U : tiling.lda)) *
            BYTES_PER_FLOAT);
        Adds(aIndices[r * padded], aIndices, aRowStep, padded);
        if (!useBlocks) {
            Adds(xIndices[r * padded], xIndices, static_cast<int32_t>(left ? 0U : r * BYTES_PER_FLOAT), padded);
        }
        if (maskEdges) {
            float edgeStep = static_cast<float>(left ? -static_cast<int32_t>(r) : static_cast<int32_t>(r));
            Adds(edgeIndices[r * padded], edgeIndices, edgeStep, padded);
        }
    }
    PIPE_BARRIER(V);
}

__aicore__ inline void GatherRowBlockMultiply(
    LocalTensor<float>& aValues, LocalTensor<float>& xValues, LocalTensor<float>& ubA, LocalTensor<float>& ubX,
    LocalTensor<uint32_t>& aOffsets, LocalTensor<uint32_t>& xOffsets, LocalTensor<uint32_t>& blockOffsets,
    uint32_t aBase, uint32_t first, uint32_t xStartAligned, uint32_t rows, uint32_t padded, uint32_t blockColumns,
    bool useBlocks)
{
    if (useBlocks) {
        Gatherb(
            aValues.ReinterpretCast<uint32_t>(), ubA[aBase].ReinterpretCast<uint32_t>(), blockOffsets,
            blockColumns / SIMD_WIDTH, GatherRepeatParams(1, 8));
        Gather(xValues, ubX, xOffsets, (first - xStartAligned) * BYTES_PER_FLOAT, blockColumns * SIMD_WIDTH);
        PIPE_BARRIER(V);
        Mul(aValues, aValues, xValues, blockColumns * SIMD_WIDTH);
        PIPE_BARRIER(V);
        Gather(xValues, aValues, aOffsets, 0U, rows * padded);
    } else {
        Gather(aValues, ubA, aOffsets, aBase * BYTES_PER_FLOAT, rows * padded);
        Gather(xValues, ubX, xOffsets, (first - xStartAligned) * BYTES_PER_FLOAT, rows * padded);
        PIPE_BARRIER(V);
        Mul(aValues, aValues, xValues, rows * padded);
    }
    PIPE_BARRIER(V);
}

__aicore__ inline void ReduceGatherRowBatch(
    LocalTensor<float>& sums, LocalTensor<float>& products, LocalTensor<float>& edgeIndices,
    LocalTensor<uint8_t>& edgeMask, uint32_t row, uint32_t rows, uint32_t tileStart, uint32_t bandwidth,
    uint32_t padded, bool batchedEdge, uint32_t limit)
{
    if (batchedEdge) {
        Compares(edgeMask, edgeIndices, static_cast<float>(static_cast<int32_t>(limit)), CMPMODE::LT, rows * padded);
        PIPE_BARRIER(V);
        Select(products, edgeMask, products, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, rows * padded);
        PIPE_BARRIER(V);
    }
    for (uint32_t segment = 0; segment * 64U < bandwidth; ++segment) {
        WholeReduceSum(
            sums[(row - tileStart) * SIMD_WIDTH + segment], products[segment * 64U],
            MinVal(64U, bandwidth - segment * 64U), rows, SIMD_WIDTH, 1, padded / SIMD_WIDTH);
    }
}

__aicore__ inline void ReduceGatherRowIrregular(
    LocalTensor<float>& sums, LocalTensor<float>& products, uint32_t row, uint32_t rows, uint32_t tileStart,
    uint32_t lower, uint32_t upper, uint32_t inputDim, uint32_t padded)
{
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t begin = row + r >= lower ? row + r - lower : 0U;
        uint32_t end = MinVal(row + r + upper + 1U, inputDim);
        uint32_t count = end > begin ? end - begin : 0U;
        for (uint32_t segment = 0; segment * 64U < count; ++segment) {
            WholeReduceSum(
                sums[(row + r - tileStart) * SIMD_WIDTH + segment], products[r * padded + segment * 64U],
                MinVal(64U, count - segment * 64U), 1, 1, 1, 8);
        }
    }
}

__aicore__ inline void ReduceGatherRowSegment(
    LocalTensor<float>& sums, LocalTensor<float>& products, LocalTensor<float>& edgeIndices,
    LocalTensor<uint8_t>& edgeMask, uint32_t row, uint32_t rows, uint32_t tileStart, uint32_t first, uint32_t lower,
    uint32_t upper, uint32_t inputDim, uint32_t bandwidth, uint32_t padded, bool left, bool uniform, bool maskEdges)
{
    const bool interior = row >= lower && row + rows - 1U + upper < inputDim;
    const bool batchedEdge = !interior && maskEdges && rows == SIMD_WIDTH && uniform;
    if (interior || batchedEdge) {
        const uint32_t limit = left ? row + upper + 1U : (first < inputDim ? inputDim - first : 0U);
        ReduceGatherRowBatch(
            sums, products, edgeIndices, edgeMask, row, rows, tileStart, bandwidth, padded, batchedEdge, limit);
    } else {
        ReduceGatherRowIrregular(sums, products, row, rows, tileStart, lower, upper, inputDim, padded);
    }
}

template <bool TRANSPOSE>
__aicore__ inline void ProcessGatherRowChunk(
    uint32_t row, uint32_t rows, uint32_t tileStart, uint32_t aColStart, uint32_t xStartAligned, uint32_t lower,
    uint32_t upper, uint32_t inputDim, uint32_t bandwidth, uint32_t padded, uint32_t blockColumns, bool useBlocks,
    bool maskEdges, bool left, bool uniform, const SgbmvTilingData& tiling, LocalTensor<float>& ubA,
    LocalTensor<float>& ubX, LocalTensor<uint32_t>& aOffsets, LocalTensor<uint32_t>& xOffsets,
    LocalTensor<uint32_t>& blockOffsets, LocalTensor<float>& aValues, LocalTensor<float>& xValues,
    LocalTensor<float>& edgeIndices, LocalTensor<uint8_t>& edgeMask, LocalTensor<float>& sums)
{
    uint32_t first = row >= lower ? row - lower : 0U;
    uint32_t aBase = TRANSPOSE ? (row - aColStart) * tiling.lda + tiling.ku + first - row :
                                 (first - aColStart) * tiling.lda + tiling.ku + row - first;
    GatherRowBlockMultiply(
        aValues, xValues, ubA, ubX, aOffsets, xOffsets, blockOffsets, aBase, first, xStartAligned, rows, padded,
        blockColumns, useBlocks);
    auto products = useBlocks ? xValues : aValues;
    ReduceGatherRowSegment(
        sums, products, edgeIndices, edgeMask, row, rows, tileStart, first, lower, upper, inputDim, bandwidth, padded,
        left, uniform, maskEdges);
    PIPE_BARRIER(V);
}

__aicore__ inline void FinalizeGatherTileReduction(LocalTensor<float>& ubY, LocalTensor<float>& sums, uint32_t tileRows)
{
    uint32_t fullRows = tileRows / SIMD_WIDTH * SIMD_WIDTH;
    if (fullRows != 0U) {
        ReduceDataBlock<ReduceType::SUM>(ubY, sums, 64U, fullRows / SIMD_WIDTH, 1, 1, 8);
    }
    if (tileRows > fullRows) {
        ReduceDataBlock<ReduceType::SUM>(
            ubY[fullRows], sums[fullRows * SIMD_WIDTH], (tileRows - fullRows) * SIMD_WIDTH, 1, 1, 1, 8);
    }
}

template <bool TRANSPOSE>
__aicore__ inline void ComputeTileGatherRows(
    uint32_t tileStart, uint32_t tileEnd, uint32_t aColStart, uint32_t xStartAligned, uint32_t lower, uint32_t upper,
    uint32_t inputDim, uint32_t bandwidth, uint32_t padded, uint32_t blockColumns, bool useBlocks, bool maskEdges,
    const SgbmvTilingData& tiling, LocalTensor<float>& ubA, LocalTensor<float>& ubX, LocalTensor<uint32_t>& aOffsets,
    LocalTensor<uint32_t>& xOffsets, LocalTensor<uint32_t>& blockOffsets, LocalTensor<float>& aValues,
    LocalTensor<float>& xValues, LocalTensor<float>& edgeIndices, LocalTensor<uint8_t>& edgeMask,
    LocalTensor<float>& sums)
{
    auto aIndices = aOffsets.ReinterpretCast<int32_t>();
    auto xIndices = xOffsets.ReinterpretCast<int32_t>();
    int32_t previousMode = -1;
    for (uint32_t row = tileStart; row < tileEnd;) {
        uint32_t rows = MinVal(8U, tileEnd - row);
        bool left = row + rows <= lower;
        bool uniform = left || row >= lower;
        if (!uniform) {
            rows = 1U;
        }
        int32_t mode = left ? 0 : 1;
        if (rows > 1U && mode != previousMode) {
            UpdateMultiRowOffsets<TRANSPOSE>(
                aIndices, xIndices, edgeIndices, rows, padded, left, useBlocks, maskEdges, tiling);
            previousMode = mode;
        }
        ProcessGatherRowChunk<TRANSPOSE>(
            row, rows, tileStart, aColStart, xStartAligned, lower, upper, inputDim, bandwidth, padded, blockColumns,
            useBlocks, maskEdges, left, uniform, tiling, ubA, ubX, aOffsets, xOffsets, blockOffsets, aValues, xValues,
            edgeIndices, edgeMask, sums);
        row += rows;
    }
}

template <bool TRANSPOSE>
__aicore__ inline void ComputeTileGather(
    uint32_t tileStart, uint32_t tileEnd, uint32_t aColStart, uint32_t xStartAligned, const SgbmvTilingData& tiling,
    LocalTensor<float>& ubA, LocalTensor<float>& ubX, LocalTensor<float>& ubY, AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    auto sums = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_SUMS);
    auto aOffsets = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_OFFSET_A_INDICES);
    auto xOffsets = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_OFFSET_X_INDICES);
    auto aValues = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_A_VALUES);
    auto xValues = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_X_VALUES);
    auto edgeIndices = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_EDGE_INDICES);
    auto edgeMask = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(UB_OFFSET_EDGE_MASK);
    auto blockOffsets = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_OFFSET_Y_ORIG);
    uint32_t bandwidth = tiling.kl + tiling.ku + 1U;
    uint32_t padded = (bandwidth + SIMD_WIDTH - 1U) / SIMD_WIDTH * SIMD_WIDTH;
    const uint32_t lower = TRANSPOSE ? tiling.ku : tiling.kl;
    const uint32_t upper = TRANSPOSE ? tiling.kl : tiling.ku;
    const uint32_t inputDim = TRANSPOSE ? tiling.m : tiling.n;
    const bool maskEdges = inputDim >= bandwidth && (tileStart < lower || tileEnd - 1U + upper >= inputDim);
    bool useBlocks = !TRANSPOSE && tiling.kl % SIMD_WIDTH == 0U && tiling.ku % SIMD_WIDTH == 0U &&
                     (tiling.lda - 1U) % SIMD_WIDTH == 0U;
    uint32_t blockColumns = (bandwidth + 14U) / SIMD_WIDTH * SIMD_WIDTH;
    auto aIndices = aOffsets.ReinterpretCast<int32_t>();
    auto xIndices = xOffsets.ReinterpretCast<int32_t>();
    InitGatherOffsets<TRANSPOSE>(
        aIndices, xIndices, edgeIndices, blockOffsets, bandwidth, padded, blockColumns, tiling, useBlocks, maskEdges);
    Duplicate(sums, 0.0f, FAST_TILE_M * SIMD_WIDTH);
    PIPE_BARRIER(V);
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID1);
    ComputeTileGatherRows<TRANSPOSE>(
        tileStart, tileEnd, aColStart, xStartAligned, lower, upper, inputDim, bandwidth, padded, blockColumns,
        useBlocks, maskEdges, tiling, ubA, ubX, aOffsets, xOffsets, blockOffsets, aValues, xValues, edgeIndices,
        edgeMask, sums);
    FinalizeGatherTileReduction(ubY, sums, tileEnd - tileStart);
}

__aicore__ inline void ProcessSgbmvNFastTile(
    uint32_t tileStart, uint32_t tileEnd, const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal,
    GlobalTensor<float>& xGlobal, GlobalTensor<float>& yGlobal, LocalTensor<float>& ubA, LocalTensor<float>& ubX,
    LocalTensor<float>& ubY, AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    uint32_t numRows = tileEnd - tileStart;
    uint32_t colStart = (tileStart >= tiling.kl) ? (tileStart - tiling.kl) : 0U;
    uint32_t colEnd = (tileEnd + tiling.ku < tiling.n) ? (tileEnd + tiling.ku) : tiling.n;
    colStart = MinVal(colStart, tiling.n);
    colEnd = MinVal(colEnd, tiling.n);

    if (colStart >= colEnd) {
        auto yPtr = reinterpret_cast<__ubuf__ float*>(ubY.GetPhyAddr());
        for (uint32_t i = 0U; i < numRows; ++i) {
            yPtr[i] = 0.0f;
        }
        WriteBackContiguous(tileStart, numRows, 1.0f, tiling.beta, yGlobal, ubY, buf);
        return;
    }

    uint32_t colStartAligned = (colStart / CORE_ALIGN_SIZE) * CORE_ALIGN_SIZE;
    uint32_t colEndAligned = MinVal(((colEnd + VECTOR_ALIGN_MASK) / CORE_ALIGN_SIZE) * CORE_ALIGN_SIZE, tiling.n);
    uint32_t totalCols = colEndAligned - colStartAligned;
    uint32_t aCount = totalCols * tiling.lda;
    uint64_t aGmOff = static_cast<uint64_t>(tiling.lda) * colStartAligned;

    if (tiling.kl + tiling.ku + 1U <= MAX_SMALLBAND_VEC && tiling.kl + tiling.ku + 1U >= 3U &&
        ProcessSgbmvNFastTileSmallBand(tileStart, tileEnd, colStartAligned, colEndAligned, tiling, aGlobal, xGlobal,
            yGlobal, aGmOff, ubA, ubY, buf)) {
        return;
    }

    CopyGmToUb(ubX, xGlobal[colStartAligned], totalCols);
    SET_FLAG(MTE2, V, EVENT_ID0);
    CopyGmToUb(ubA, aGlobal[aGmOff], aCount);
    // Make x visible to the vector pipe while the larger matrix transfer is in flight.
    WAIT_FLAG(MTE2, V, EVENT_ID0);

    auto yPtr = reinterpret_cast<__ubuf__ float*>(ubY.GetPhyAddr());
    auto xPtr = reinterpret_cast<__ubuf__ const float*>(ubX.GetPhyAddr());
    auto aPtr = reinterpret_cast<__ubuf__ const float*>(ubA.GetPhyAddr());
    uint32_t bandwidth = tiling.kl + tiling.ku + 1U;
    bool useGather = bandwidth >= MIN_GATHER_BANDWIDTH && bandwidth <= MAX_GATHER_BANDWIDTH;
    if (useGather) {
        ComputeTileGather<false>(tileStart, tileEnd, colStartAligned, colStartAligned, tiling, ubA, ubX, ubY, buf);
    } else {
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        PIPE_BARRIER(ALL);
        FastNComputeTileRows(tileStart, tileEnd, colStartAligned, tiling, aPtr, xPtr, yPtr);
    }

    WriteBackContiguous(tileStart, numRows, tiling.alpha, tiling.beta, yGlobal, ubY, buf, useGather);
}

__aicore__ inline void ProcessSgbmvNFast(
    uint32_t startRow, uint32_t endRow, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y)
{
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> xGlobal;
    GlobalTensor<float> yGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * tiling.n);
    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling.n);
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), tiling.m);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    LocalTensor<float> ubY = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_Y);
    LocalTensor<float> ubX = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_X);
    LocalTensor<float> ubA = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_A);
    for (uint32_t tileStart = startRow; tileStart < endRow; tileStart += FAST_TILE_M) {
        uint32_t tileEnd = MinVal(tileStart + FAST_TILE_M, endRow);
        ProcessSgbmvNFastTile(tileStart, tileEnd, tiling, aGlobal, xGlobal, yGlobal, ubA, ubX, ubY, buf);
    }
}

__aicore__ inline float FastTComputeCol(
    uint32_t col, uint32_t c, uint32_t rStartAligned, uint32_t m, uint32_t ku, uint32_t kl, uint32_t lda,
    __ubuf__ const float* __restrict__ aPtr, __ubuf__ const float* __restrict__ xPtr)
{
    uint32_t rowStart = (col >= ku) ? (col - ku) : 0U;
    uint32_t rowEnd = MinVal(col + kl + 1U, m);
    if (rowStart >= rowEnd) {
        return 0.0f;
    }
    uint32_t len = rowEnd - rowStart;
    uint32_t aLocalOff = c * lda + (ku + rowStart - col);
    uint32_t xLocalOff = rowStart - rStartAligned;
    auto aCol = aPtr + aLocalOff;
    auto xCol = xPtr + xLocalOff;
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float s5 = 0.0f;
    float s6 = 0.0f;
    float s7 = 0.0f;
    uint32_t k = 0U;
    for (; k + 7U < len; k += UNROLL_STEP) {
        s0 += aCol[k + 0U] * xCol[k + 0U];
        s1 += aCol[k + 1U] * xCol[k + 1U];
        s2 += aCol[k + 2U] * xCol[k + 2U];
        s3 += aCol[k + 3U] * xCol[k + 3U];
        s4 += aCol[k + 4U] * xCol[k + 4U];
        s5 += aCol[k + 5U] * xCol[k + 5U];
        s6 += aCol[k + 6U] * xCol[k + 6U];
        s7 += aCol[k + 7U] * xCol[k + 7U];
    }
    float acc = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (; k < len; ++k) {
        acc += aCol[k] * xCol[k];
    }
    return acc;
}

// Small-band (band <= 32) transposed batch: one strided Gather per band row j
// produces AB(j, colBase..colBase+curCols8) as a vector, multiplied by the
// diagonally-shifted x window (zero-padded at both matrix edges) and
// accumulated into y. Replaces the per-column scalar dot loop, which is
// scalar-pipe bound.
__aicore__ inline void TFastColBatchSmallBand(
    uint32_t colBase, uint32_t curCols, const SgbmvTilingData& tiling, GlobalTensor<float>& xGlobal,
    GlobalTensor<float>& yGlobal, LocalTensor<float>& ubA, LocalTensor<float>& ubY,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    const uint32_t band = tiling.kl + tiling.ku + 1U;
    const uint32_t curCols8 = (curCols + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    // A is staged by the caller (full columns): ubA[i*lda + j] = AB(j, colBase+i).
    // Zero the tail columns' storage so lanes in [curCols, curCols8) contribute
    // exact zeros regardless of the x values they gather. The pad starts at the
    // 8-float-aligned end of the staged run (the DMA's own right pad covers the
    // sub-alignment slack with zeros); vector-op tensor offsets must stay
    // 8-float aligned.
    if (curCols8 * tiling.lda > curCols * tiling.lda) {
        uint32_t padFrom = ((curCols * tiling.lda) + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
        uint32_t padEnd = (((curCols8 * tiling.lda) + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U)) + SIMD_WIDTH;
        Duplicate(ubA[padFrom], 0.0f, padEnd - padFrom);
    }
    auto ubX = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_X);
    auto ubB = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_A_VALUES);
    auto ubS = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_X_VALUES);
    auto ubT = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_SB_SCRATCH);
    auto tbl = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_OFFSET_A_INDICES);
    auto tbl4 = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_OFFSET_TBL4);

    // x window: ubX[0] <-> row xr0 (8-aligned, may be negative). Rows outside
    // [0, m) stay zero, which zeroes every out-of-range (column, j) pair.
    const int32_t xr0 = Floor8Signed(static_cast<int32_t>(colBase) - static_cast<int32_t>(tiling.ku));
    const int32_t xHi = static_cast<int32_t>(colBase + curCols8 + tiling.kl + 8U);
    const uint32_t xWin8 = static_cast<uint32_t>((xHi - xr0 + SIMD_WIDTH - 1U) & ~static_cast<int32_t>(SIMD_WIDTH - 1U));
    const int32_t cpyLo = (xr0 >= 0) ? xr0 : 0;
    const int32_t cpyHi = MinVal(static_cast<int32_t>(tiling.m), xHi);
    if (cpyHi > cpyLo) {
        CopyGmToUb(ubX[cpyLo - xr0], xGlobal[cpyLo], cpyHi - cpyLo);
        if (cpyLo > xr0) {
            Duplicate(ubX, 0.0f, static_cast<uint32_t>(cpyLo - xr0));
        }
        uint32_t xTailFrom = static_cast<uint32_t>(cpyLo - xr0) +
            ((static_cast<uint32_t>(cpyHi - cpyLo) + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U));
        if (xWin8 > xTailFrom) {
            Duplicate(ubX[xTailFrom], 0.0f, xWin8 - xTailFrom);
        }
    } else {
        Duplicate(ubX, 0.0f, xWin8);
    }
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);

    MakeOffsetSequence(tbl.ReinterpretCast<int32_t>(), curCols8, static_cast<int32_t>(tiling.lda * BYTES_PER_FLOAT));
    MakeOffsetSequence(tbl4.ReinterpretCast<int32_t>(), curCols8, static_cast<int32_t>(BYTES_PER_FLOAT));
    Duplicate(ubY, 0.0f, curCols8);
    PIPE_BARRIER(V);
    for (uint32_t j = 0U; j < band; ++j) {
        Gather(ubB, ubA, tbl, j * BYTES_PER_FLOAT, curCols8);
        uint32_t xOff = static_cast<uint32_t>(static_cast<int32_t>(colBase + j) - static_cast<int32_t>(tiling.ku) - xr0);
        Gather(ubS, ubX, tbl4, xOff * BYTES_PER_FLOAT, curCols8);
        PIPE_BARRIER(V);
        Mul(ubT, ubB, ubS, curCols8);
        PIPE_BARRIER(V);
        Add(ubY, ubY, ubT, curCols8);
        PIPE_BARRIER(V);
    }
    WriteBackContiguousVec8(colBase, curCols, curCols8, tiling.alpha, tiling.beta, yGlobal, ubY, buf);
}

__aicore__ inline void ProcessSgbmvTFastColBatch(
    uint32_t colBase, uint32_t curCols, const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal,
    GlobalTensor<float>& xGlobal, GlobalTensor<float>& yGlobal, LocalTensor<float>& ubA, LocalTensor<float>& ubX,
    LocalTensor<float>& ubY, AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    uint32_t colLast = colBase + curCols - 1U;
    uint32_t rStart = (colBase >= tiling.ku) ? (colBase - tiling.ku) : 0U;
    uint32_t rEnd = MinVal(colLast + tiling.kl + 1U, tiling.m);

    uint32_t rStartAligned = (rStart / CORE_ALIGN_SIZE) * CORE_ALIGN_SIZE;
    uint32_t rEndAligned = (rEnd < tiling.m) ?
                               MinVal(((rEnd + VECTOR_ALIGN_MASK) / CORE_ALIGN_SIZE) * CORE_ALIGN_SIZE, tiling.m) :
                               tiling.m;
    uint32_t xCountAligned = (rEndAligned > rStartAligned) ? (rEndAligned - rStartAligned) : 0U;

    CopyGmToUb(ubA, aGlobal[static_cast<uint64_t>(tiling.lda) * colBase], curCols * tiling.lda);
    if (tiling.kl + tiling.ku + 1U <= MAX_SMALLBAND_VEC && tiling.kl + tiling.ku + 1U > SIMD_WIDTH) {
        TFastColBatchSmallBand(colBase, curCols, tiling, xGlobal, yGlobal, ubA, ubY, buf);
        return;
    }
    if (xCountAligned > 0U) {
        CopyGmToUb(ubX, xGlobal[rStartAligned], xCountAligned);
    }

    auto yPtr = reinterpret_cast<__ubuf__ float*>(ubY.GetPhyAddr());
    auto xPtr = reinterpret_cast<__ubuf__ const float*>(ubX.GetPhyAddr());
    auto aPtr = reinterpret_cast<__ubuf__ const float*>(ubA.GetPhyAddr());

    uint32_t bandwidth = tiling.kl + tiling.ku + 1U;
    const bool useGather = bandwidth >= MIN_GATHER_BANDWIDTH && bandwidth <= MAX_GATHER_BANDWIDTH;
    if (useGather) {
        ComputeTileGather<true>(colBase, colBase + curCols, colBase, rStartAligned, tiling, ubA, ubX, ubY, buf);
    } else {
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        PIPE_BARRIER(ALL);
        for (uint32_t c = 0U; c < curCols; ++c) {
            yPtr[c] =
                FastTComputeCol(colBase + c, c, rStartAligned, tiling.m, tiling.ku, tiling.kl, tiling.lda, aPtr, xPtr);
        }
    }

    WriteBackContiguous(colBase, curCols, tiling.alpha, tiling.beta, yGlobal, ubY, buf, useGather);
}

__aicore__ inline void ProcessSgbmvTFast(
    uint32_t startRow, uint32_t endRow, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y)
{
    // Wide bands (>= 130) have no gather table; use the vector column-dot path.
    if (tiling.kl + tiling.ku + 1U > MAX_GATHER_BANDWIDTH) {
        ProcessSgbmvTVec(startRow, endRow, tiling, a, x, y);
        return;
    }
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> xGlobal;
    GlobalTensor<float> yGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * tiling.n);
    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling.m);
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), tiling.n);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    LocalTensor<float> ubY = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_Y);
    LocalTensor<float> ubX = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_X);
    LocalTensor<float> ubA = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_A);
    for (uint32_t colBase = startRow; colBase < endRow; colBase += FAST_COL_BATCH) {
        uint32_t curCols = MinVal(FAST_COL_BATCH, endRow - colBase);
        ProcessSgbmvTFastColBatch(colBase, curCols, tiling, aGlobal, xGlobal, yGlobal, ubA, ubX, ubY, buf);
    }
}

__aicore__ void ProcessSgbmvNFallback(
    uint32_t startRow, uint32_t endRow, const SgbmvTilingData& tiling, __gm__ const float* aGm, __gm__ const float* xGm,
    __gm__ float* yGm)
{
    constexpr uint32_t kTileSize = 128U;
    for (uint32_t tileStart = startRow; tileStart < endRow; tileStart += kTileSize) {
        uint32_t tileEnd = MinVal(tileStart + kTileSize, endRow);
        uint32_t numRows = tileEnd - tileStart;
        float acc[kTileSize]{0.0f};

        uint32_t colStart = (tileStart >= tiling.kl) ? (tileStart - tiling.kl) : 0U;
        uint32_t colEnd = MinVal(tileEnd + tiling.ku, tiling.n);

        for (uint32_t col = colStart; col < colEnd; ++col) {
            uint32_t r0 = (col >= tiling.ku) ? MaxVal(col - tiling.ku, tileStart) : tileStart;
            uint32_t r1 = MinVal(col + tiling.kl + 1U, tileEnd);
            if (r0 < r1) {
                int64_t xIdx = (tiling.incx >= 0) ? (static_cast<int64_t>(col) * tiling.incx) :
                                                    (static_cast<int64_t>(tiling.n - 1U - col) * (-tiling.incx));
                float xVal = xGm[xIdx];
                if (xVal == 0.0f) {
                    continue;
                }
                uint64_t aOff = static_cast<uint64_t>(tiling.lda) * col + (tiling.ku + r0 - col);
                uint32_t accOff = r0 - tileStart;
                uint32_t len = r1 - r0;
                for (uint32_t k = 0U; k < len; ++k) {
                    acc[accOff + k] += aGm[aOff + k] * xVal;
                }
            }
        }

        for (uint32_t i = 0U; i < numRows; ++i) {
            WriteY(tileStart + i, tiling.m, tiling.incy, tiling.alpha, tiling.beta, acc[i], yGm);
        }
    }
}

// Row-split granularity for the vector N fallback: wider chunks give larger
// DMA blocks (Lt*4 bytes per column) while keeping >= 8 cores active.
__aicore__ inline uint32_t VecNRowAlign(uint32_t outDim)
{
    if (outDim >= 2048U) {
        return 32U; // max stream count
    }
    if (outDim >= 1024U) {
        return 64U;
    }
    if (outDim >= 512U) {
        return 32U;
    }
    return CORE_ALIGN_SIZE;
}

__aicore__ void DispatchSgbmvCompute(
    uint32_t startRow, uint32_t endRow, uint32_t outDim, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm, bool fastN)
{
    if (tiling.alpha == 0.0f) {
        ProcessAlphaZero(startRow, endRow, outDim, tiling.incy, tiling.beta, yGm);
        return;
    }
    if (fastN) {
        ProcessSgbmvNFast(startRow, endRow, tiling, a, x, y);
        return;
    }
    if (IsFastpathT(tiling)) {
        ProcessSgbmvTFast(startRow, endRow, tiling, a, x, y);
        return;
    }
    if (tiling.trans == 0U) {
        if (UseVecFallbackN(tiling)) {
            ProcessSgbmvNVec(startRow, endRow, tiling, a, x, y, aGm, xGm, yGm);
        } else {
            ProcessSgbmvNFallback(startRow, endRow, tiling, aGm, xGm, yGm);
        }
    } else {
        if (UseVecFallbackT(tiling)) {
            ProcessSgbmvTVec(startRow, endRow, tiling, a, x, y);
        } else {
            ProcessSgbmvTFallback(startRow, endRow, tiling, aGm, xGm, yGm);
        }
    }
}

// Large split tail: every block writes one 32-float-aligned slice of the
// [tailBase, outDim) zero-fill directly (see the caller for the alignment
// constraint rationale). Full slices are 32B aligned at both ends; the final
// partial remainder goes to the last full-slice block, which then ends at
// outDim as a single writer (same shape as the legacy last-block range).
// Scratch: the legacy Y region [0, 1024) floats, re-initialized by every
// domain path on first use.
__aicore__ inline void SgbmvWriteSplitTailLarge(
    uint32_t blkIdx, uint32_t numBlocks, uint32_t outDim, uint32_t tailBase, GM_ADDR y, const SgbmvTilingData& tiling)
{
    uint32_t tailLen = outDim - tailBase;
    const uint32_t safeBlocks = (numBlocks == 0U) ? 1U : numBlocks;
    uint32_t tailPer = ((tailLen + safeBlocks - 1U) / safeBlocks + CORE_ALIGN_SIZE - 1U) & ~(CORE_ALIGN_SIZE - 1U);
    uint32_t nFull = tailLen / tailPer;
    // full slices are 32B aligned at both ends; the final partial
    // remainder goes to the last full-slice block, which then ends at
    // outDim as a single writer (same shape as the legacy last-block
    // range).
    uint32_t sliceEnd = tailBase + (blkIdx + 1U) * tailPer;
    if (blkIdx + 1U == nFull) {
        sliceEnd = outDim;
    }
    uint32_t t0 = tailBase + blkIdx * tailPer;
    if (blkIdx < nFull && t0 < sliceEnd) {
        AsdopsBuffer<ArchType::ASCEND_V220> tailBuf;
        GlobalTensor<float> yTail;
        yTail.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), outDim);
        auto tailY = tailBuf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_Y);
        while (t0 < sliceEnd) {
            uint32_t len = MinVal(1024U, sliceEnd - t0);
            Duplicate(tailY, 0.0f, (len + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U));
            WriteBackContiguous(t0, len, 0.0f, tiling.beta, yTail, tailY, tailBuf);
            t0 += len;
        }
    }
}

__global__ __aicore__ void sgbmv_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SgbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;

    uint32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t numBlocks = tiling.useCoreNum;
    if (numBlocks == 0U || blkIdx >= numBlocks) {
        return;
    }

    uint32_t outDim = (tiling.trans == 0U) ? tiling.m : tiling.n;
    uint32_t startRow = 0U;
    uint32_t endRow = outDim;
    const bool fastN = tiling.alpha != 0.0f && IsFastpathN(tiling);
    const bool vecN = tiling.alpha != 0.0f && !fastN && UseVecFallbackN(tiling);
    // Work rebalance: only outputs [0, min(outDim, crossDim+k))
    // have any band overlap (N row r spans columns [r-kl, r+ku], empty iff
    // r >= n+kl; T column j spans rows [j-ku, j+kl], empty iff j >= m+ku).
    // Splitting the full outDim leaves (outDim - valid) outputs' worth of cores
    // idle on wide-output/short-input shapes. Split the
    // valid range instead and cover the all-zero tail (a pure y = beta*y0
    // fill) separately.
    uint32_t crossDim = (tiling.trans == 0U) ? tiling.n : tiling.m;
    // incx/incy != 1: the domain paths address x/y through the strides; the
    // tail writer below only knows contiguous y.
    uint32_t splitDim = (tiling.incx == 1 && tiling.incy == 1)
                            ? MinVal(outDim, crossDim + (tiling.trans == 0U ? tiling.kl : tiling.ku))
                            : outDim;
    // TVec split granularity: the wide-band T vector
    // column-dot path covers whole output columns and writes y through one
    // contiguous vector DMA per core, whose only hard requirement is the
    // 8-float (32B) sector boundary. Splitting at CORE_ALIGN_SIZE (32)
    // quantizes medium column counts badly. Split the TVec path at 8.
    // Gate mirrors the host-side core-count chunk gate (CalSgbmvTilingData).
    const bool vecT = tiling.alpha != 0.0f && tiling.trans != 0U && tiling.incx == 1 && tiling.incy == 1 &&
                      (tiling.kl + tiling.ku + 1U) > MAX_GATHER_BANDWIDTH &&
                      (IsFastpathT(tiling) || UseVecFallbackT(tiling));
    CalcCoreRowRange(
        splitDim, numBlocks, blkIdx, startRow, endRow,
        fastN ? SIMD_WIDTH : (vecN ? VecNRowAlign(splitDim) : (vecT ? SIMD_WIDTH : CORE_ALIGN_SIZE)));
    uint32_t tailBase = (splitDim + CORE_ALIGN_SIZE - 1U) & ~(CORE_ALIGN_SIZE - 1U);
    if (splitDim < outDim) {
        if (outDim - splitDim > 512U) {
            // Large tail: every block writes one 32-float-aligned slice of it
            // directly (handing the whole tail to one block makes that block
            // the critical path). Slice boundaries
            // MUST stay 32B aligned: y cache sectors at a boundary are shared
            // between neighboring blocks' DMA writes, and a misaligned
            // partial write races with the neighbor. The alignment
            // gap [splitDim, tailBase) goes to the last block's normal path.
            // Scratch: the legacy Y region [0, 1024) floats, re-initialized
            // by every domain path on first use.
            SgbmvWriteSplitTailLarge(blkIdx, numBlocks, outDim, tailBase, y, tiling);
            if (blkIdx + 1U == numBlocks && endRow < tailBase) {
                endRow = MinVal(tailBase, outDim);
            }
        } else if (blkIdx + 1U == numBlocks) {
            // Small tail: the last block absorbs it through its normal path
            // (a distributed fill costs every block ~0.5-1us of fixed
            // writeback overhead, which measurably regresses short-tail
            // shapes whose tails are only ~200 outputs).
            endRow = outDim;
        }
    }
    if (startRow >= endRow) {
        return;
    }

    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);

    DispatchSgbmvCompute(startRow, endRow, outDim, tiling, a, x, y, aGm, xGm, yGm, fastN);
}

void sgbmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgbmvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    sgbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
}

