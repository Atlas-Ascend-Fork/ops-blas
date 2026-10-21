/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Reuse private device templates while keeping kernel UB allocation isolated.
#define CTRSM_BATCHED_SHARED_ONLY
#include "ctrsmbatched_kernel.cpp"

namespace CtrsmBatchedBlockedVector {
using namespace CtrsmBatchedSimd;
using namespace AscendC;
using namespace AscendC::Reg;

// SPLIT: 0 = interleaved complex FMA, 1 = four independent sums,
// 2 = select the accumulation layout for the physical RHS tail.
struct PanelAccumulator {
    RegTensor<float> rr, ri, ir, ii, cr, ci;
};

__simd_callee__ inline void ClearPanel(PanelAccumulator& sum, MaskReg& mask)
{
    Duplicate(sum.rr, 0.0f, mask);
    Duplicate(sum.ri, 0.0f, mask);
    Duplicate(sum.ir, 0.0f, mask);
    Duplicate(sum.ii, 0.0f, mask);
    Duplicate(sum.cr, 0.0f, mask);
    Duplicate(sum.ci, 0.0f, mask);
}

template <bool LEFT, uint32_t SPLIT>
__simd_callee__ inline void AccumulatePanel(
    PanelAccumulator& sum, RegTensor<float>& ar, RegTensor<float>& ai, RegTensor<float>& xr, RegTensor<float>& xi,
    MaskReg& mask)
{
    if constexpr (SPLIT != 0) {
        MulAddDst(sum.rr, ar, xr, mask);
        MulAddDst(sum.ri, ar, xi, mask);
        MulAddDst(sum.ir, ai, xr, mask);
        MulAddDst(sum.ii, ai, xi, mask);
    }
    if constexpr (SPLIT != 1) {
        Muls(sum.cr, sum.cr, -1.0f, mask);
        MulAddDst(sum.cr, ai, xi, mask);
        Muls(sum.cr, sum.cr, -1.0f, mask);
        MulAddDst(sum.cr, ar, xr, mask);
        if constexpr (LEFT) {
            MulAddDst(sum.ci, ar, xi, mask);
            MulAddDst(sum.ci, ai, xr, mask);
        } else {
            MulAddDst(sum.ci, ai, xr, mask);
            MulAddDst(sum.ci, ar, xi, mask);
        }
    }
}

template <uint32_t SPLIT>
__simd_callee__ inline void FinishPanel(
    RegTensor<float>& real, RegTensor<float>& imag, PanelAccumulator& sum, MaskReg& mask, MaskReg& tail)
{
    RegTensor<float> sr, si, temp, cross;
    if constexpr (SPLIT != 0) {
        Sub(sr, sum.rr, sum.ii, mask);
        Add(si, sum.ri, sum.ir, mask);
        if constexpr (SPLIT == 2) {
            Select(sr, sum.cr, sr, tail);
            Select(si, sum.ci, si, tail);
        }
    } else {
        sr = sum.cr;
        si = sum.ci;
    }
    // Preserve zero cross terms for exceptional GEMM values.
    Muls(cross, si, 0.0f, mask);
    Sub(temp, real, cross, mask);
    Sub(real, temp, sr, mask);
    Muls(cross, sr, 0.0f, mask);
    Add(temp, imag, cross, mask);
    Sub(imag, temp, si, mask);
}

template <uint32_t WIDTH, bool LEFT, bool FORWARD, uint32_t SPLIT, uint32_t CHUNKS>
__simd_callee__ inline void AccumulatePanelColumns(
    PanelAccumulator& sum0, PanelAccumulator& sum1, PanelAccumulator& sum2, PanelAccumulator& sum3, __ubuf__ float* aa,
    __ubuf__ float* bb, uint16_t row, uint16_t begin, uint16_t end, MaskReg& mask)
{
    constexpr uint32_t STRIDE = 2 * WIDTH + 8;
    RegTensor<float> ar, ai, xr, xi;
    for (uint16_t step = begin; step < end; ++step) {
        const uint16_t col = FORWARD ? step : end - 1 - (step - begin);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + row * (row + 1) + 2 * col);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + row * (row + 1) + 2 * col + 1);
        if constexpr (CHUNKS > 0) {
            DataCopy(xr, bb + col * STRIDE + 0);
            DataCopy(xi, bb + col * STRIDE + WIDTH + 0);
            AccumulatePanel<LEFT, SPLIT>(sum0, ar, ai, xr, xi, mask);
        }
        if constexpr (CHUNKS > 1) {
            DataCopy(xr, bb + col * STRIDE + 64);
            DataCopy(xi, bb + col * STRIDE + WIDTH + 64);
            AccumulatePanel<LEFT, SPLIT>(sum1, ar, ai, xr, xi, mask);
        }
        if constexpr (CHUNKS > 2) {
            DataCopy(xr, bb + col * STRIDE + 128);
            DataCopy(xi, bb + col * STRIDE + WIDTH + 128);
            AccumulatePanel<LEFT, SPLIT>(sum2, ar, ai, xr, xi, mask);
        }
        if constexpr (CHUNKS > 3) {
            DataCopy(xr, bb + col * STRIDE + 192);
            DataCopy(xi, bb + col * STRIDE + WIDTH + 192);
            AccumulatePanel<LEFT, SPLIT>(sum3, ar, ai, xr, xi, mask);
        }
    }
}

template <uint32_t WIDTH, bool LEFT, bool FORWARD, uint32_t SPLIT, uint32_t CHUNKS>
__simd_callee__ inline void SubtractPanel(
    RegTensor<float>& v0r, RegTensor<float>& v0i, RegTensor<float>& v1r, RegTensor<float>& v1i, RegTensor<float>& v2r,
    RegTensor<float>& v2i, RegTensor<float>& v3r, RegTensor<float>& v3i, __ubuf__ float* aa, __ubuf__ float* bb,
    uint16_t row, uint16_t begin, uint16_t end, MaskReg& mask, MaskReg& tail0, MaskReg& tail1, MaskReg& tail2,
    MaskReg& tail3)
{
    constexpr uint32_t STRIDE = 2 * WIDTH + 8;
    PanelAccumulator sum0, sum1, sum2, sum3;
    if constexpr (CHUNKS > 0)
        ClearPanel(sum0, mask);
    if constexpr (CHUNKS > 1)
        ClearPanel(sum1, mask);
    if constexpr (CHUNKS > 2)
        ClearPanel(sum2, mask);
    if constexpr (CHUNKS > 3)
        ClearPanel(sum3, mask);
    AccumulatePanelColumns<WIDTH, LEFT, FORWARD, SPLIT, CHUNKS>(sum0, sum1, sum2, sum3, aa, bb, row, begin, end, mask);
    if constexpr (CHUNKS > 0)
        FinishPanel<SPLIT>(v0r, v0i, sum0, mask, tail0);
    if constexpr (CHUNKS > 1)
        FinishPanel<SPLIT>(v1r, v1i, sum1, mask, tail1);
    if constexpr (CHUNKS > 2)
        FinishPanel<SPLIT>(v2r, v2i, sum2, mask, tail2);
    if constexpr (CHUNKS > 3)
        FinishPanel<SPLIT>(v3r, v3i, sum3, mask, tail3);
}

template <uint32_t WIDTH, uint32_t CHUNKS>
__simd_callee__ inline void SubtractChunks(
    ChunkValues& v, __ubuf__ float* aa, __ubuf__ float* bb, uint16_t row, uint16_t col, MaskReg& mask)
{
    constexpr uint32_t STRIDE = 2 * WIDTH + 8;
    RegTensor<float> ar, ai, xr, xi;
    DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + row * (row + 1) + 2 * col);
    DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + row * (row + 1) + 2 * col + 1);
    if constexpr (CHUNKS > 0) {
        DataCopy(xr, bb + col * STRIDE + 0);
        DataCopy(xi, bb + col * STRIDE + WIDTH + 0);
        SubtractProduct(v.r0, v.i0, ar, ai, xr, xi, mask);
    }
    if constexpr (CHUNKS > 1) {
        DataCopy(xr, bb + col * STRIDE + 64);
        DataCopy(xi, bb + col * STRIDE + WIDTH + 64);
        SubtractProduct(v.r1, v.i1, ar, ai, xr, xi, mask);
    }
    if constexpr (CHUNKS > 2) {
        DataCopy(xr, bb + col * STRIDE + 128);
        DataCopy(xi, bb + col * STRIDE + WIDTH + 128);
        SubtractProduct(v.r2, v.i2, ar, ai, xr, xi, mask);
    }
    if constexpr (CHUNKS > 3) {
        DataCopy(xr, bb + col * STRIDE + 192);
        DataCopy(xi, bb + col * STRIDE + WIDTH + 192);
        SubtractProduct(v.r3, v.i3, ar, ai, xr, xi, mask);
    }
}

template <bool LEFT, bool FORWARD>
__aicore__ inline uint16_t MicroblockSize(uint16_t remain)
{
    constexpr uint16_t width = LEFT ? 8 : 2;
    uint16_t size = width;
    if constexpr (FORWARD) {
        if (remain < width) {
            size = 1;
            while (size * 2 <= remain)
                size *= 2;
        }
    } else {
        const uint16_t residual = remain % width;
        if (residual)
            size = residual & (0U - residual);
    }
    return size;
}

template <uint32_t WIDTH, bool LEFT, bool FORWARD, uint32_t SPLIT, uint32_t CHUNKS>
__simd_callee__ inline void SubtractChunkPanel(
    ChunkValues& v, ChunkTails& tail, __ubuf__ float* aa, __ubuf__ float* bb, uint16_t row, uint16_t begin,
    uint16_t end, MaskReg& mask)
{
    SubtractPanel<WIDTH, LEFT, FORWARD, SPLIT, CHUNKS>(
        v.r0, v.i0, v.r1, v.i1, v.r2, v.i2, v.r3, v.i3, aa, bb, row, begin, end, mask, tail.t0, tail.t1, tail.t2,
        tail.t3);
}

template <uint32_t WIDTH, bool LEFT, bool FORWARD, bool PARTIAL, uint32_t CHUNKS>
__simd_callee__ inline void ComputeBlockRow(
    ChunkValues& v, ChunkTails& tail, __ubuf__ float* aa, __ubuf__ float* bb, uint16_t row, uint16_t panelBegin,
    uint16_t blockBegin, uint16_t size, MaskReg& mask)
{
    LoadChunks<WIDTH, CHUNKS>(v, bb, row);
    if (panelBegin != 0)
        SubtractChunkPanel<WIDTH, LEFT, FORWARD, PARTIAL ? 2 : 1, CHUNKS>(v, tail, aa, bb, row, 0, panelBegin, mask);
    if (blockBegin != panelBegin) {
        if constexpr (LEFT) {
            if (size == 8)
                SubtractChunkPanel<WIDTH, LEFT, FORWARD, 1, CHUNKS>(v, tail, aa, bb, row, panelBegin, blockBegin, mask);
            else
                SubtractChunkPanel<WIDTH, LEFT, FORWARD, 0, CHUNKS>(v, tail, aa, bb, row, panelBegin, blockBegin, mask);
        } else {
            SubtractChunkPanel<WIDTH, LEFT, FORWARD, PARTIAL ? 2 : 1, CHUNKS>(
                v, tail, aa, bb, row, panelBegin, blockBegin, mask);
        }
    }
    for (uint16_t col = blockBegin; col < row; ++col)
        SubtractChunks<WIDTH, CHUNKS>(v, aa, bb, row, col, mask);
    UnitChunks<CHUNKS>(v, mask);
    StoreChunks<WIDTH, CHUNKS>(v, bb, row, mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

constexpr uint32_t SHORT_WIDTH = 256;
constexpr uint32_t SHORT_STRIDE = 2 * SHORT_WIDTH + 8;

// Adjacent LEFT rows share solved-B loads while retaining separate FMA sums.
template <bool FORWARD, uint32_t CHUNKS>
__simd_callee__ inline void AccumulateLeftPair(
    PanelAccumulator& sum00, PanelAccumulator& sum01, PanelAccumulator& sum10, PanelAccumulator& sum11,
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t row, uint16_t end, MaskReg& mask)
{
    RegTensor<float> ar0, ai0, ar1, ai1, xr, xi;
    for (uint16_t step = 0; step < end; ++step) {
        const uint16_t col = FORWARD ? step : end - 1 - step;
        DataCopy<float, LoadDist::DIST_BRC_B32>(ar0, aa + row * (row + 1) + 2 * col);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ai0, aa + row * (row + 1) + 2 * col + 1);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ar1, aa + (row + 1) * (row + 2) + 2 * col);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ai1, aa + (row + 1) * (row + 2) + 2 * col + 1);
        DataCopy(xr, bb + col * SHORT_STRIDE);
        DataCopy(xi, bb + col * SHORT_STRIDE + SHORT_WIDTH);
        AccumulatePanel<true, 1>(sum00, ar0, ai0, xr, xi, mask);
        AccumulatePanel<true, 1>(sum10, ar1, ai1, xr, xi, mask);
        if constexpr (CHUNKS > 1) {
            DataCopy(xr, bb + col * SHORT_STRIDE + 64);
            DataCopy(xi, bb + col * SHORT_STRIDE + SHORT_WIDTH + 64);
            AccumulatePanel<true, 1>(sum01, ar0, ai0, xr, xi, mask);
            AccumulatePanel<true, 1>(sum11, ar1, ai1, xr, xi, mask);
        }
    }
}

template <bool FORWARD, uint32_t CHUNKS>
__simd_callee__ inline void SubtractLeftPairPanel(
    ChunkValues& v, ChunkValues& w, __ubuf__ float* aa, __ubuf__ float* bb, uint16_t row, uint16_t end, MaskReg& mask)
{
    PanelAccumulator sum00, sum01, sum10, sum11;
    ClearPanel(sum00, mask);
    ClearPanel(sum10, mask);
    if constexpr (CHUNKS > 1) {
        ClearPanel(sum01, mask);
        ClearPanel(sum11, mask);
    }
    AccumulateLeftPair<FORWARD, CHUNKS>(sum00, sum01, sum10, sum11, aa, bb, row, end, mask);
    FinishPanel<1>(v.r0, v.i0, sum00, mask, mask);
    FinishPanel<1>(w.r0, w.i0, sum10, mask, mask);
    if constexpr (CHUNKS > 1) {
        FinishPanel<1>(v.r1, v.i1, sum01, mask, mask);
        FinishPanel<1>(w.r1, w.i1, sum11, mask, mask);
    }
}

template <uint32_t CHUNKS>
__simd_callee__ inline void UpdateSecondLeftRow(
    ChunkValues& v, ChunkValues& w, __ubuf__ float* aa, uint16_t row, MaskReg& mask)
{
    RegTensor<float> ar, ai;
    DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + (row + 1) * (row + 2) + 2 * row);
    DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + (row + 1) * (row + 2) + 2 * row + 1);
    SubtractProduct(w.r0, w.i0, ar, ai, v.r0, v.i0, mask);
    if constexpr (CHUNKS > 1)
        SubtractProduct(w.r1, w.i1, ar, ai, v.r1, v.i1, mask);
}

template <bool FORWARD, uint32_t CHUNKS>
__simd_callee__ inline void ComputeLeftPair(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t row, uint16_t blockBegin, MaskReg& mask)
{
    ChunkValues v, w;
    LoadChunks<SHORT_WIDTH, CHUNKS>(v, bb, row);
    LoadChunks<SHORT_WIDTH, CHUNKS>(w, bb, row + 1);
    if (blockBegin != 0)
        SubtractLeftPairPanel<FORWARD, CHUNKS>(v, w, aa, bb, row, blockBegin, mask);
    for (uint16_t col = blockBegin; col < row; ++col) {
        SubtractChunks<SHORT_WIDTH, CHUNKS>(v, aa, bb, row, col, mask);
        SubtractChunks<SHORT_WIDTH, CHUNKS>(w, aa, bb, row + 1, col, mask);
    }
    UnitChunks<CHUNKS>(v, mask);
    UpdateSecondLeftRow<CHUNKS>(v, w, aa, row, mask);
    UnitChunks<CHUNKS>(w, mask);
    StoreChunks<SHORT_WIDTH, CHUNKS>(v, bb, row, mask);
    StoreChunks<SHORT_WIDTH, CHUNKS>(w, bb, row + 1, mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <uint32_t WIDTH, bool LEFT, bool FORWARD, bool PARTIAL, uint32_t CHUNKS>
__aicore__ inline void ComputeChunkRow(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t firstRhs, uint32_t rhsCount, uint16_t row, uint16_t size,
    uint16_t panelBegin, uint16_t blockBegin)
{
    __VEC_SCOPE__
    {
        auto mask = CreateMask<float>();
        ChunkTails tail;
        ChunkValues values;
        RegTensor<int32_t> lanes;
        Arange(lanes, int32_t(firstRhs));
        Compares<int32_t, CMPMODE::GE>(tail.t0, lanes, int32_t(rhsCount / 8 * 8), mask);
        Arange(lanes, int32_t(firstRhs + 64));
        Compares<int32_t, CMPMODE::GE>(tail.t1, lanes, int32_t(rhsCount / 8 * 8), mask);
        Arange(lanes, int32_t(firstRhs + 128));
        Compares<int32_t, CMPMODE::GE>(tail.t2, lanes, int32_t(rhsCount / 8 * 8), mask);
        Arange(lanes, int32_t(firstRhs + 192));
        Compares<int32_t, CMPMODE::GE>(tail.t3, lanes, int32_t(rhsCount / 8 * 8), mask);
        ComputeBlockRow<WIDTH, LEFT, FORWARD, PARTIAL, CHUNKS>(
            values, tail, aa, bb, row, panelBegin, blockBegin, size, mask);
    }
}

template <bool FORWARD, uint32_t CHUNKS>
__aicore__ inline void ComputeChunkPair(__ubuf__ float* aa, __ubuf__ float* bb, uint16_t row, uint16_t blockBegin)
{
    __VEC_SCOPE__
    {
        auto mask = CreateMask<float>();
        ComputeLeftPair<FORWARD, CHUNKS>(aa, bb, row, blockBegin, mask);
    }
}

template <uint32_t WIDTH, bool LEFT, bool FORWARD, bool PARTIAL, uint32_t CHUNKS>
__aicore__ inline void ComputeChunkBlock(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, uint16_t originalK, uint32_t firstRhs, uint32_t rhsCount,
    uint16_t blockBegin, uint16_t blockEnd, uint16_t size, uint16_t panelBegin)
{
    if constexpr (WIDTH == SHORT_WIDTH && LEFT && CHUNKS <= 2) {
        if (size == 8 && k == originalK) {
            for (uint16_t row = blockBegin; row < blockEnd; row += 2)
                ComputeChunkPair<FORWARD, CHUNKS>(aa, bb, row, blockBegin);
            return;
        }
    }
    for (uint16_t row = blockBegin; row < blockEnd; ++row)
        ComputeChunkRow<WIDTH, LEFT, FORWARD, PARTIAL, CHUNKS>(
            aa, bb, firstRhs, rhsCount, row, size, panelBegin, blockBegin);
}

template <uint32_t WIDTH, bool LEFT, bool FORWARD, bool PARTIAL, uint32_t CHUNKS>
__aicore__ inline void ComputeChunks(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, uint16_t originalK, uint32_t firstRhs, uint32_t rhsCount,
    uint16_t count)
{
    const uint16_t panelLength = !LEFT && !FORWARD ? (originalK - 1) % 192 + 1 : (originalK < 192 ? originalK : 192);
    uint16_t panelBegin = 0, panelSize = panelLength;
    while (panelBegin < k) {
        const uint16_t panelEnd = panelBegin + panelSize;
        uint16_t blockBegin = panelBegin;
        while (blockBegin < panelEnd && blockBegin < k) {
            const uint16_t size = MicroblockSize<LEFT, FORWARD>(panelEnd - blockBegin);
            const uint16_t blockEnd = blockBegin + size < k ? blockBegin + size : k;
            ComputeChunkBlock<WIDTH, LEFT, FORWARD, PARTIAL, CHUNKS>(
                aa, bb, k, originalK, firstRhs, rhsCount, blockBegin, blockEnd, size, panelBegin);
            blockBegin += size;
        }
        panelBegin = panelEnd;
        panelSize = 192;
    }
}

// Share each coefficient load across independent RHS vectors. The chunk
// count is selected outside the dependency loops so inactive chunks compile
// out.
template <uint32_t WIDTH, bool LEFT, bool FORWARD, bool PARTIAL>
__aicore__ inline void Compute(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, uint16_t originalK, uint32_t firstRhs, uint32_t rhsCount,
    uint16_t count)
{
    if constexpr (WIDTH == 256 && LEFT) {
        // Bound the number of simultaneous accumulators without changing RHS order.
        if (count > 128) {
            ComputeChunks<WIDTH, LEFT, FORWARD, PARTIAL, 2>(aa, bb, k, originalK, firstRhs, rhsCount, 128);
            if (count > 192)
                ComputeChunks<WIDTH, LEFT, FORWARD, PARTIAL, 2>(
                    aa, bb + 128, k, originalK, firstRhs + 128, rhsCount, count - 128);
            else
                ComputeChunks<WIDTH, LEFT, FORWARD, PARTIAL, 1>(
                    aa, bb + 128, k, originalK, firstRhs + 128, rhsCount, count - 128);
            return;
        }
    } else if constexpr (WIDTH == 256) {
        if (count > 192) {
            ComputeChunks<WIDTH, LEFT, FORWARD, PARTIAL, 4>(aa, bb, k, originalK, firstRhs, rhsCount, count);
            return;
        }
        if (count > 128) {
            ComputeChunks<WIDTH, LEFT, FORWARD, PARTIAL, 3>(aa, bb, k, originalK, firstRhs, rhsCount, count);
            return;
        }
    }
    if (count > 64)
        ComputeChunks<WIDTH, LEFT, FORWARD, PARTIAL, 2>(aa, bb, k, originalK, firstRhs, rhsCount, count);
    else
        ComputeChunks<WIDTH, LEFT, FORWARD, PARTIAL, 1>(aa, bb, k, originalK, firstRhs, rhsCount, count);
}

// Two columns of a RIGHT microblock share every solved-B load. Each
// row and RHS still has its own four accumulators and the original FMA order.
template <bool FORWARD>
__simd_callee__ inline void AccumulateRightPair(
    PanelAccumulator& sum00, PanelAccumulator& sum01, PanelAccumulator& sum10, PanelAccumulator& sum11,
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t row, MaskReg& mask)
{
    RegTensor<float> ar0, ai0, ar1, ai1, xr, xi;
    for (uint16_t step = 0; step < row; ++step) {
        const uint16_t col = FORWARD ? step : row - 1 - step;
        DataCopy<float, LoadDist::DIST_BRC_B32>(ar0, aa + row * (row + 1) + 2 * col);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ai0, aa + row * (row + 1) + 2 * col + 1);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ar1, aa + (row + 1) * (row + 2) + 2 * col);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ai1, aa + (row + 1) * (row + 2) + 2 * col + 1);
        DataCopy(xr, bb + col * 264 + 0);
        DataCopy(xi, bb + col * 264 + 128);
        MulAddDst(sum00.rr, ar0, xr, mask);
        MulAddDst(sum00.ri, ar0, xi, mask);
        MulAddDst(sum00.ir, ai0, xr, mask);
        MulAddDst(sum00.ii, ai0, xi, mask);
        MulAddDst(sum10.rr, ar1, xr, mask);
        MulAddDst(sum10.ri, ar1, xi, mask);
        MulAddDst(sum10.ir, ai1, xr, mask);
        MulAddDst(sum10.ii, ai1, xi, mask);
        DataCopy(xr, bb + col * 264 + 64);
        DataCopy(xi, bb + col * 264 + 192);
        MulAddDst(sum01.rr, ar0, xr, mask);
        MulAddDst(sum01.ri, ar0, xi, mask);
        MulAddDst(sum01.ir, ai0, xr, mask);
        MulAddDst(sum01.ii, ai0, xi, mask);
        MulAddDst(sum11.rr, ar1, xr, mask);
        MulAddDst(sum11.ri, ar1, xi, mask);
        MulAddDst(sum11.ir, ai1, xr, mask);
        MulAddDst(sum11.ii, ai1, xi, mask);
    }
}

template <bool FORWARD>
__simd_callee__ inline void SubtractRightPair(
    RegTensor<float>& v0r, RegTensor<float>& v0i, RegTensor<float>& v1r, RegTensor<float>& v1i, RegTensor<float>& w0r,
    RegTensor<float>& w0i, RegTensor<float>& w1r, RegTensor<float>& w1i, __ubuf__ float* aa, __ubuf__ float* bb,
    uint16_t row, MaskReg& mask)
{
    RegTensor<float> ar0, ai0, ar1, ai1, xr, xi;
    PanelAccumulator sum00;
    ClearPanel(sum00, mask);
    PanelAccumulator sum01;
    ClearPanel(sum01, mask);
    PanelAccumulator sum10;
    ClearPanel(sum10, mask);
    PanelAccumulator sum11;
    ClearPanel(sum11, mask);
    AccumulateRightPair<FORWARD>(sum00, sum01, sum10, sum11, aa, bb, row, mask);
    FinishPanel<1>(v0r, v0i, sum00, mask, mask);
    FinishPanel<1>(v1r, v1i, sum01, mask, mask);
    FinishPanel<1>(w0r, w0i, sum10, mask, mask);
    FinishPanel<1>(w1r, w1i, sum11, mask, mask);
}

template <bool FORWARD>
__simd_callee__ inline void SolveRightPairs(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, uint16_t pairStart, MaskReg& mask)
{
    RegTensor<float> v0r, v0i, v1r, v1i, w0r, w0i, w1r, w1i, ar, ai, pr, pi;
    const uint16_t pairStop = k - 1;
    for (uint16_t row = pairStart; row < pairStop; row += 2) {
        DataCopy(v0r, bb + (row + 0) * 264 + 0);
        DataCopy(v0i, bb + (row + 0) * 264 + 128);
        DataCopy(v1r, bb + (row + 0) * 264 + 64);
        DataCopy(v1i, bb + (row + 0) * 264 + 192);
        DataCopy(w0r, bb + (row + 1) * 264 + 0);
        DataCopy(w0i, bb + (row + 1) * 264 + 128);
        DataCopy(w1r, bb + (row + 1) * 264 + 64);
        DataCopy(w1i, bb + (row + 1) * 264 + 192);
        if (row != 0)
            SubtractRightPair<FORWARD>(v0r, v0i, v1r, v1i, w0r, w0i, w1r, w1i, aa, bb, row, mask);
        UnitIdentity(v0r, v0i, mask);
        UnitIdentity(v1r, v1i, mask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + (row + 1) * (row + 2) + 2 * row);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + (row + 1) * (row + 2) + 2 * row + 1);
        SubtractProduct(w0r, w0i, ar, ai, v0r, v0i, mask);
        UnitIdentity(w0r, w0i, mask);
        SubtractProduct(w1r, w1i, ar, ai, v1r, v1i, mask);
        UnitIdentity(w1r, w1i, mask);
        DataCopy(bb + (row + 0) * 264 + 0, v0r, mask);
        DataCopy(bb + (row + 0) * 264 + 128, v0i, mask);
        DataCopy(bb + (row + 0) * 264 + 64, v1r, mask);
        DataCopy(bb + (row + 0) * 264 + 192, v1i, mask);
        DataCopy(bb + (row + 1) * 264 + 0, w0r, mask);
        DataCopy(bb + (row + 1) * 264 + 128, w0i, mask);
        DataCopy(bb + (row + 1) * 264 + 64, w1r, mask);
        DataCopy(bb + (row + 1) * 264 + 192, w1i, mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <bool FORWARD>
__aicore__ inline void ComputeRightPairs(__ubuf__ float* aa, __ubuf__ float* bb, uint16_t k)
{
    __VEC_SCOPE__
    {
        auto mask = CreateMask<float>();
        RegTensor<float> v0r, v0i, v1r, v1i, w0r, w0i, w1r, w1i, ar, ai, pr, pi;
        uint16_t row = 0;
        if constexpr (!FORWARD) {
            if (k % 2) {
                DataCopy(v0r, bb);
                DataCopy(v0i, bb + 128);
                DataCopy(v1r, bb + 64);
                DataCopy(v1i, bb + 192);
                UnitIdentity(v0r, v0i, mask);
                UnitIdentity(v1r, v1i, mask);
                DataCopy(bb, v0r, mask);
                DataCopy(bb + 128, v0i, mask);
                DataCopy(bb + 64, v1r, mask);
                DataCopy(bb + 192, v1i, mask);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                row = 1;
            }
        }
        const uint16_t pairStart = row;
        SolveRightPairs<FORWARD>(aa, bb, k, pairStart, mask);
        row = pairStart + ((k - pairStart) / 2) * 2;
        if (row < k) {
            DataCopy(v0r, bb + row * 264);
            DataCopy(v0i, bb + row * 264 + 128);
            DataCopy(v1r, bb + row * 264 + 64);
            DataCopy(v1i, bb + row * 264 + 192);
            if (row != 0)
                SubtractPanel<128, false, FORWARD, 1, 2>(
                    v0r, v0i, v1r, v1i, w0r, w0i, w1r, w1i, aa, bb, row, 0, row, mask, mask, mask, mask, mask);
            UnitIdentity(v0r, v0i, mask);
            UnitIdentity(v1r, v1i, mask);
            DataCopy(bb + row * 264, v0r, mask);
            DataCopy(bb + row * 264 + 128, v0i, mask);
            DataCopy(bb + row * 264 + 64, v1r, mask);
            DataCopy(bb + row * 264 + 192, v1i, mask);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }
}

template <uint32_t WIDTH, bool LEFT, bool PARTIAL>
__aicore__ inline void ComputeDirection(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, uint16_t originalK, uint32_t firstRhs, uint32_t rhs,
    uint16_t count, bool forward)
{
    if (forward)
        Compute<WIDTH, LEFT, true, PARTIAL>(aa, bb, k, originalK, firstRhs, rhs, count);
    else
        Compute<WIDTH, LEFT, false, PARTIAL>(aa, bb, k, originalK, firstRhs, rhs, count);
}

template <uint32_t WIDTH>
__aicore__ inline void Run(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, CtrsmBatchedTilingData t, uint32_t firstRhs, uint16_t count)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper;
    const uint16_t originalK = left ? t.m : t.n;
    const uint32_t rhs = left ? t.n : t.m;
    if constexpr (WIDTH == 128) {
        if (!left && k == originalK && k <= 96 && count > 64 && firstRhs + count <= rhs / 8 * 8) {
            if (forward)
                ComputeRightPairs<true>(aa, bb, k);
            else
                ComputeRightPairs<false>(aa, bb, k);
            return;
        }
    }
    if (left)
        ComputeDirection<WIDTH, true, false>(aa, bb, k, originalK, firstRhs, rhs, count, forward);
    else if (firstRhs + count > rhs / 8 * 8)
        ComputeDirection<WIDTH, false, true>(aa, bb, k, originalK, firstRhs, rhs, count, forward);
    else
        ComputeDirection<WIDTH, false, false>(aa, bb, k, originalK, firstRhs, rhs, count, forward);
}
} // namespace CtrsmBatchedBlockedVector

// Keep this translation unit separate: UNIT kernels need long SIMT branches;
// applying that compiler option to NON_UNIT runtime rows changes their results.
namespace CtrsmBatchedUnit {
using namespace AscendC;
using namespace AscendC::Reg;
constexpr uint32_t K = 96, WIDTH = 128, STRIDE = 264, THREADS = 256;
static_assert(K * (K + 1) + K * STRIDE + WIDTH + 264 + 4 * STRIDE + 32 <= 53248);

// Once both components of every active RHS are NaN, all later block
// products propagate that pair, including zero coefficients.
__simt_vf__ __aicore__
LAUNCH_BOUND(THREADS) inline void ScanLast(__ubuf__ float* bb, __ubuf__ uint32_t* flags, uint32_t k, uint32_t count)
{
    uint32_t value = 1;
    if (threadIdx.x < count)
        value = __builtin_isnan(bb[(k - 1) * STRIDE + threadIdx.x]) &&
                __builtin_isnan(bb[(k - 1) * STRIDE + WIDTH + threadIdx.x]);
    const auto reduced = asc_reduce_min(value);
    if (threadIdx.x % warpSize == 0)
        flags[threadIdx.x / warpSize] = reduced;
    asc_syncthreads();
    if (threadIdx.x == 0) {
        uint32_t all = 1;
        for (uint32_t i = 0; i < THREADS / warpSize; ++i)
            all &= flags[i];
        flags[18] = all;
    }
}

struct PrefixState {
    CtrsmBatchedTilingData t;
    const __gm__ aclblasComplex* ap;
    const __gm__ aclblasComplex* bp;
    uint32_t originalK, firstRhs, count;
    bool left, forward;
};

__simt_callee__ CTRSM_INLINE inline PrefixState MakePrefixState(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t originalK = left ? t.m : t.n, rhsCount = left ? t.n : t.m;
    const uint32_t tiles = (rhsCount + WIDTH - 1) / WIDTH;
    const uint64_t batch = job <= 0xffffffffULL ? static_cast<uint32_t>(job) / tiles : job / tiles;
    const uint32_t tile = job <= 0xffffffffULL ? static_cast<uint32_t>(job) % tiles : job % tiles;
    const uint32_t firstRhs = tile * WIDTH, count = rhsCount - firstRhs < WIDTH ? rhsCount - firstRhs : WIDTH;
    const auto ap = reinterpret_cast<const __gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
    const auto bp = reinterpret_cast<const __gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper;
    return {t, ap, bp, originalK, firstRhs, count, left, forward};
}

template <uint32_t PART>
__simt_callee__ CTRSM_INLINE inline aclblasComplex ReadPrefixA(
    const PrefixState& ctx, __ubuf__ aclblasComplex* raw, uint32_t r, uint32_t c, uint32_t end)
{
    aclblasComplex v;
    const uint32_t row = ctx.forward ? r : ctx.originalK - 1 - r;
    const uint32_t col = ctx.forward ? c : ctx.originalK - 1 - c;
    if constexpr (PART == 1) {
        uint32_t rr = ctx.forward ? r : end - 1 - r, cc = ctx.forward ? c : end - 1 - c;
        if (!ctx.left) {
            const auto tmp = rr;
            rr = cc;
            cc = tmp;
        }
        const uint32_t pitch = (end + 3) / 4 * 4;
        const uint32_t off = ctx.t.trans == ACLBLAS_OP_N ? rr + cc * pitch : cc + rr * pitch;
        v = {raw[off].real, raw[off].imag};
        if (ctx.t.trans == ACLBLAS_OP_C)
            v.imag = -v.imag;
    } else
        v = ctx.left ? CtrsmBatched::LoadA(ctx.ap, row, col, ctx.t) : CtrsmBatched::LoadA(ctx.ap, col, row, ctx.t);
    return v;
}

template <uint32_t PART>
__simt_callee__ CTRSM_INLINE inline bool TransformPrefixA(
    const PrefixState& ctx, __ubuf__ float* aa, uint32_t first, uint32_t end, __ubuf__ aclblasComplex* raw)
{
    bool unsafe = false;
    for (uint32_t q = threadIdx.x; q < end * end; q += THREADS) {
        const bool columnMajor = ctx.left == (ctx.t.trans == ACLBLAS_OP_N);
        const uint32_t r = columnMajor ? q % end : q / end, c = columnMajor ? q / end : q % end;
        if (r < first || r < c)
            continue;
        aclblasComplex v{1, 0};
        if (r != c) {
            v = ReadPrefixA<PART>(ctx, raw, r, c, end);
        }
        unsafe |= CtrsmBatchedSmallVector::Unsafe(v.real) || CtrsmBatchedSmallVector::Unsafe(v.imag);
        aa[r * (r + 1) + 2 * c] = v.real;
        aa[r * (r + 1) + 2 * c + 1] = v.imag;
    }
    return unsafe;
}

template <uint32_t PART>
__simt_callee__ CTRSM_INLINE inline aclblasComplex ReadPrefixB(
    const PrefixState& ctx, __ubuf__ aclblasComplex* raw, uint32_t step, uint32_t rhs, uint32_t first, uint32_t end)
{
    aclblasComplex v;
    const uint32_t row = ctx.forward ? step : ctx.originalK - 1 - step;
    const uint64_t off = ctx.left ? row + static_cast<uint64_t>(ctx.firstRhs + rhs) * ctx.t.ldb :
                                    ctx.firstRhs + rhs + static_cast<uint64_t>(row) * ctx.t.ldb;
    if constexpr (PART == 2) {
        const uint32_t rr = ctx.forward ? step - first : end - 1 - step;
        const uint32_t pitch = (end - first + 3) / 4 * 4;
        const uint32_t src = ctx.left ? rhs * pitch + rr : rr * WIDTH + rhs;
        v = {raw[src].real, raw[src].imag};
    } else
        v = {ctx.bp[off].real, ctx.bp[off].imag};
    return v;
}

template <uint32_t PART>
__simt_callee__ CTRSM_INLINE inline bool TransformPrefixB(
    const PrefixState& ctx, __ubuf__ float* bb, uint32_t first, uint32_t end, __ubuf__ aclblasComplex* raw)
{
    bool unsafe = false;
    for (uint32_t q = threadIdx.x; q < (end - first) * WIDTH; q += THREADS) {
        const uint32_t step = first + (ctx.left ? q % (end - first) : q / WIDTH);
        const uint32_t rhs = ctx.left ? q / (end - first) : q % WIDTH;
        aclblasComplex v{0, 0};
        if (rhs < ctx.count) {
            v = ReadPrefixB<PART>(ctx, raw, step, rhs, first, end);
            unsafe |= CtrsmBatchedSmallVector::Unsafe(v.real) || CtrsmBatchedSmallVector::Unsafe(v.imag);
            if (ctx.t.alpha.real != 1 || ctx.t.alpha.imag != 0)
                v = CtrsmBatched::ScaleInput(ctx.t.alpha, v);
        }
        bb[step * STRIDE + rhs] = v.real;
        bb[step * STRIDE + WIDTH + rhs] = v.imag;
    }
    return unsafe;
}

template <uint32_t PART = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void TransformPrefix(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, __ubuf__ float* aa, __ubuf__ float* bb,
    __ubuf__ uint32_t* indices, __ubuf__ uint32_t* bad, uint32_t first, uint32_t end, __ubuf__ aclblasComplex* raw)
{
    const auto ctx = MakePrefixState(a, b, t, job);
    bool unsafe = (PART == 2 || first != 0) && bad[threadIdx.x];
    unsafe |= CtrsmBatchedSmallVector::Unsafe(t.alpha.real) || CtrsmBatchedSmallVector::Unsafe(t.alpha.imag);
    unsafe |= !ctx.left && t.trans != ACLBLAS_OP_N && (t.alpha.real != 1 || t.alpha.imag != 0);
    if (first == 0 && threadIdx.x < WIDTH)
        indices[threadIdx.x] = 0;
    if constexpr (PART != 2)
        unsafe |= TransformPrefixA<PART>(ctx, aa, first, end, raw);
    if constexpr (PART != 1)
        unsafe |= TransformPrefixB<PART>(ctx, bb, first, end, raw);
    bad[threadIdx.x] = unsafe ? 1 : 0;
}

__aicore__ inline void LoadPrefix(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, __ubuf__ float* aa, __ubuf__ float* bb,
    __ubuf__ uint32_t* indices, __ubuf__ uint32_t* bad, uint32_t first, uint32_t end)
{
    constexpr uint32_t CAPACITY = 53248 - (K * (K + 1) + K * STRIDE + WIDTH + 264 + 4 * STRIDE + 32);
    auto raw = reinterpret_cast<__ubuf__ aclblasComplex*>(bad + 264 + 4 * STRIDE + 32);
    const uint32_t aPitch = (end + 3) / 4 * 4;
    if (2 * end * aPitch > CAPACITY) {
        asc_vf_call<TransformPrefix<0>>(dim3{THREADS, 1, 1}, a, b, t, job, aa, bb, indices, bad, first, end, raw);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t originalK = left ? t.m : t.n, rhs = left ? t.n : t.m, tiles = (rhs + WIDTH - 1) / WIDTH;
    const uint32_t firstRhs = job % tiles * WIDTH, count = rhs - firstRhs < WIDTH ? rhs - firstRhs : WIDTH;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N), forward = left ? !upper : upper;
    auto ap = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(a)[job / tiles]);
    auto bp = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[job / tiles]);
    const uint32_t corner = forward ? 0 : originalK - end;
    asc_copy_gm2ub_align(
        reinterpret_cast<__ubuf__ float*>(raw), ap + 2 * (corner + static_cast<uint64_t>(corner) * t.lda), end, end * 8,
        0, 0, false, 0, static_cast<uint64_t>(t.lda) * 8, aPitch * 8);
    PipeBarrier<PIPE_ALL>();
    asc_vf_call<TransformPrefix<1>>(dim3{THREADS, 1, 1}, a, b, t, job, aa, bb, indices, bad, first, end, raw);
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t CHUNK = CAPACITY / (2 * WIDTH) < 64 ? CAPACITY / (2 * WIDTH) : 64;
    for (uint32_t begin = first; begin < end; begin += CHUNK) {
        const uint32_t finish = begin + CHUNK < end ? begin + CHUNK : end, rows = finish - begin;
        const uint32_t row = forward ? begin : originalK - finish;
        const uint64_t off =
            left ? row + static_cast<uint64_t>(firstRhs) * t.ldb : firstRhs + static_cast<uint64_t>(row) * t.ldb;
        asc_copy_gm2ub_align(
            reinterpret_cast<__ubuf__ float*>(raw), bp + off * 2, left ? count : rows, (left ? rows : count) * 8, 0, 0,
            false, 0, static_cast<uint64_t>(t.ldb) * 8, (left ? (rows + 3) / 4 * 4 : WIDTH) * 8);
        PipeBarrier<PIPE_ALL>();
        asc_vf_call<TransformPrefix<2>>(dim3{THREADS, 1, 1}, a, b, t, job, aa, bb, indices, bad, begin, finish, raw);
        PipeBarrier<PIPE_ALL>();
    }
}

template <bool LEFT, bool FORWARD>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void PackOutput(
    __ubuf__ float* bb, __ubuf__ aclblasComplex* raw, uint32_t first, uint32_t rows, uint32_t count)
{
    for (uint32_t q = threadIdx.x; q < rows * WIDTH; q += THREADS) {
        const uint32_t step = q / WIDTH, rhs = q % WIDTH;
        if (rhs >= count)
            continue;
        const uint32_t row = FORWARD ? step : rows - 1 - step;
        const uint32_t source = (first + step) * STRIDE + rhs;
        const uint32_t dest = LEFT ? rhs * 64 + row : row * WIDTH + rhs;
        raw[dest].real = bb[source];
        raw[dest].imag = bb[source + WIDTH];
    }
}

// Completed short systems can pack into the now-unused input staging area
// and use DMA for the strided GM write instead of individual SIMT stores.
__aicore__ inline void StoreCompleted(
    GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    constexpr uint32_t CAPACITY = 53248 - (K * (K + 1) + K * STRIDE + WIDTH + 264 + 4 * STRIDE + 32);
    static_assert(CAPACITY >= 2 * 64 * WIDTH);
    auto raw = reinterpret_cast<__ubuf__ aclblasComplex*>(bad + 264 + 4 * STRIDE + 32);
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper;
    const uint32_t k = left ? t.m : t.n, rhs = left ? t.n : t.m, tiles = (rhs + WIDTH - 1) / WIDTH;
    const uint32_t firstRhs = job % tiles * WIDTH, count = rhs - firstRhs < WIDTH ? rhs - firstRhs : WIDTH;
    auto bp = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[job / tiles]);
    for (uint32_t begin = 0; begin < k; begin += 64) {
        const uint32_t end = begin + 64 < k ? begin + 64 : k, rows = end - begin;
        if (left) {
            if (forward)
                asc_vf_call<PackOutput<true, true>>(dim3{THREADS, 1, 1}, bb, raw, begin, rows, count);
            else
                asc_vf_call<PackOutput<true, false>>(dim3{THREADS, 1, 1}, bb, raw, begin, rows, count);
        } else {
            if (forward)
                asc_vf_call<PackOutput<false, true>>(dim3{THREADS, 1, 1}, bb, raw, begin, rows, count);
            else
                asc_vf_call<PackOutput<false, false>>(dim3{THREADS, 1, 1}, bb, raw, begin, rows, count);
        }
        PipeBarrier<PIPE_ALL>();
        const uint32_t row = forward ? begin : k - end;
        const uint64_t offset =
            left ? row + static_cast<uint64_t>(firstRhs) * t.ldb : firstRhs + static_cast<uint64_t>(row) * t.ldb;
        asc_copy_ub2gm_align(
            bp + 2 * offset, reinterpret_cast<__ubuf__ float*>(raw), left ? count : rows, (left ? rows : count) * 8, 0,
            static_cast<uint64_t>(t.ldb) * 8, (left ? 64 : WIDTH) * 8);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void RunBlocked(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, __ubuf__ float* storage)
{
    auto aa = storage;
    auto bb = aa + K * (K + 1);
    auto indices = reinterpret_cast<__ubuf__ uint32_t*>(bb + K * STRIDE);
    auto bad = indices + WIDTH;
    auto flags = bad + 264 + 4 * STRIDE;
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t originalK = left ? t.m : t.n, rhs = left ? t.n : t.m;
    const uint16_t firstK = originalK < 80 ? originalK : 80;
    const uint32_t tiles = (rhs + WIDTH - 1) / WIDTH;
    const uint64_t jobs = static_cast<uint64_t>(tiles) * t.batchCount;
    const uint64_t quotient = jobs / GetBlockNum(), remainder = jobs % GetBlockNum();
    const uint64_t firstJob = GetBlockIdx() * quotient + (GetBlockIdx() < remainder ? GetBlockIdx() : remainder);
    const uint64_t endJob = firstJob + quotient + (GetBlockIdx() < remainder ? 1 : 0);
    for (uint64_t job = firstJob; job < endJob; ++job) {
        const uint32_t firstRhs = job % tiles * WIDTH;
        const uint32_t count = rhs - firstRhs < WIDTH ? rhs - firstRhs : WIDTH;
        uint16_t k = firstK;
        LoadPrefix(a, b, t, job, aa, bb, indices, bad, 0, k);
        PipeBarrier<PIPE_ALL>();
        CtrsmBatchedBlockedVector::Run<WIDTH>(aa, bb, k, t, firstRhs, count);
        PipeBarrier<PIPE_ALL>();
        if (originalK > k) {
            asc_vf_call<ScanLast>(dim3{THREADS, 1, 1}, bb, flags, k, count);
            PipeBarrier<PIPE_ALL>();
            if (!reinterpret_cast<volatile __ubuf__ uint32_t*>(flags)[18]) {
                k = originalK < K ? originalK : K;
                LoadPrefix(a, b, t, job, aa, bb, indices, bad, 0, k);
                PipeBarrier<PIPE_ALL>();
                CtrsmBatchedBlockedVector::Run<WIDTH>(aa, bb, k, t, firstRhs, count);
                PipeBarrier<PIPE_ALL>();
            }
        }
        if (k == originalK)
            StoreCompleted(b, t, job, bb, bad);
        else
            asc_vf_call<CtrsmBatchedSmallVector::Transfer<K, 1, WIDTH, true, true>>(
                dim3{THREADS, 1, 1}, a, b, t, job, aa, bb, indices, bad, k);
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace CtrsmBatchedUnit

extern "C" __global__ __aicore__ void ctrsm_batched_unit_prefix_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float storage[CtrsmBatchedSmallVector::STORAGE_FLOATS];
    CtrsmBatchedUnit::RunBlocked(a, b, t, storage);
}

// Keep UNIT wide-tile instantiations and their UB allocation in this translation unit.
template <uint32_t K, bool FIXED>
__global__ __aicore__ void ctrsm_batched_unit_short_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float storage[CtrsmBatchedSmallVector::STORAGE_FLOATS];
    t.diag = ACLBLAS_UNIT;
    CtrsmBatchedSmallVector::Run<K, 1, 256, FIXED ? K : 0, 0, 512>(a, b, t, storage);
}

void ctrsm_batched_unit_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream)
{
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const uint32_t rhs = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    if (k <= 40 && rhs > 128) {
        if (k == 16)
            ctrsm_batched_unit_short_kernel<16, true><<<blocks, nullptr, stream>>>(a, b, t);
        else if (k == 24)
            ctrsm_batched_unit_short_kernel<24, true><<<blocks, nullptr, stream>>>(a, b, t);
        else if (k == 32)
            ctrsm_batched_unit_short_kernel<32, true><<<blocks, nullptr, stream>>>(a, b, t);
        else
            ctrsm_batched_unit_short_kernel<40, false><<<blocks, nullptr, stream>>>(a, b, t);
    } else {
        ctrsm_batched_unit_prefix_kernel<<<blocks, nullptr, stream>>>(a, b, t);
    }
}
