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
 * \file cgemv_batched_kernel.cpp
 * \brief Complex batched GEMV kernel for arch35 (Ascend 950PR).
 *        y[b] = alpha * op(A[b]) * x[b] + beta * y[b],  op = N / T / C
 *
 * Two vectorized paths plus a SIMT fallback:
 *   - trans=T/C (AIV "dot" path): each AIV handles a row-range of one batch and
 *     reduces over the dot dimension. A columns are contiguous, so a row block
 *     is copied with DataCopyPad (blockLen = dotCnt, srcStride = lda-dotCnt).
 *   - trans=N  (AIV "column" path): each AIV handles a column-range of one
 *     batch. A column is contiguous, so a column block is copied with
 *     DataCopyPad (blockLen = outCnt, srcStride = 0) and x is broadcast per
 *     column with DIST_BRC_B32. Mirrors the arch22 layout: vectorize over rows.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "cgemv_batched_tiling_data.h"

using namespace AscendC;

// ---------------------------------------------------------------------------
// GM pointer-array helpers
// ---------------------------------------------------------------------------

template <typename T>
__simt_callee__ __aicore__ inline __gm__ T* CgemvReadPtrFromArray(GM_ADDR arrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(arrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ T*>(rawAddr);
}

template <typename T>
__aicore__ inline __gm__ T* CgemvReadPtrFromAivArray(GM_ADDR arrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(arrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ T*>(rawAddr);
}

// ---------------------------------------------------------------------------
// AIV vectorized path for trans=T/C (reduce over rows, vectorize over dots)
// ---------------------------------------------------------------------------

// One complex multiply-accumulate step of the dot reduction:
//   acc += a * x, with the imaginary sign following the transpose mode
// (trans=T: a_re*x_re + a_im*x_im, trans=C: a_re*x_re - a_im*x_im).
// mask is the element mask, outMask the accumulate mask.
template <bool IS_TRANS_T>
__simd_callee__ inline void CgemvDotAccStep(
    Reg::RegTensor<float>& realAcc, Reg::RegTensor<float>& imagAcc, Reg::RegTensor<float>& aReal,
    Reg::RegTensor<float>& aImag, Reg::RegTensor<float>& xReal, Reg::RegTensor<float>& xImag, Reg::MaskReg mask,
    Reg::MaskReg outMask)
{
    Reg::RegTensor<float> term;
    Reg::RegTensor<float> scratch;
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(term, aReal, xReal, mask);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(scratch, aImag, xImag, mask);
    if constexpr (IS_TRANS_T) {
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(term, term, scratch, mask);
    } else {
        Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(term, term, scratch, mask);
    }
    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(realAcc, realAcc, term, outMask);

    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(term, aReal, xImag, mask);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(scratch, aImag, xReal, mask);
    if constexpr (IS_TRANS_T) {
        Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(term, term, scratch, mask);
    } else {
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(term, term, scratch, mask);
    }
    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(imagAcc, imagAcc, term, outMask);
}

// Reduce one full-width (VL-lane) chunk of the dot dimension.
template <bool IS_TRANS_T>
__simd_callee__ inline void CgemvAivDotFullChunk(
    Reg::RegTensor<float>& realAcc, Reg::RegTensor<float>& imagAcc, __ubuf__ float* aAddr, __ubuf__ float* xAddr,
    uint32_t offset, Reg::MaskReg maskAll)
{
    Reg::RegTensor<float> aReal;
    Reg::RegTensor<float> aImag;
    Reg::RegTensor<float> xReal;
    Reg::RegTensor<float> xImag;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(aReal, aImag, aAddr + 2u * offset);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(xReal, xImag, xAddr + 2u * offset);
    CgemvDotAccStep<IS_TRANS_T>(realAcc, imagAcc, aReal, aImag, xReal, xImag, maskAll, maskAll);
}

// Reduce the trailing partial chunk (fewer than VL lanes) of the dot dimension.
template <bool IS_TRANS_T>
__simd_callee__ inline void CgemvAivDotTailChunk(
    Reg::RegTensor<float>& realAcc, Reg::RegTensor<float>& imagAcc, __ubuf__ float* aAddr, __ubuf__ float* xAddr,
    uint32_t offset, uint32_t tail, Reg::MaskReg maskAll)
{
    Reg::RegTensor<float> aReal;
    Reg::RegTensor<float> aImag;
    Reg::RegTensor<float> xReal;
    Reg::RegTensor<float> xImag;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(aReal, aImag, aAddr + 2u * offset);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(xReal, xImag, xAddr + 2u * offset);
    uint32_t tailMaskCount = tail;
    auto tailMask = Reg::UpdateMask<float>(tailMaskCount);
    CgemvDotAccStep<IS_TRANS_T>(realAcc, imagAcc, aReal, aImag, xReal, xImag, tailMask, maskAll);
}

template <bool IS_TRANS_T>
__simd_vf__ inline void CgemvAivDotAllVf(
    __ubuf__ float* aBase, __ubuf__ float* xAddr, __ubuf__ float* resultAddr, uint32_t dotCount, uint32_t outStart,
    uint32_t outEnd, uint32_t dotTile)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    auto maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    for (uint32_t outIdx = outStart; outIdx < outEnd; ++outIdx) {
        __ubuf__ float* aAddr = aBase + outIdx * dotTile;

        Reg::RegTensor<float> realAcc;
        Reg::RegTensor<float> imagAcc;
        Reg::Duplicate<float>(realAcc, 0.0f, maskAll);
        Reg::Duplicate<float>(imagAcc, 0.0f, maskAll);

        const uint16_t fullLoopNum = static_cast<uint16_t>(dotCount / VL);
        for (uint16_t i = 0; i < fullLoopNum; ++i) {
            CgemvAivDotFullChunk<IS_TRANS_T>(realAcc, imagAcc, aAddr, xAddr, static_cast<uint32_t>(i) * VL, maskAll);
        }

        const uint32_t tail = dotCount - static_cast<uint32_t>(fullLoopNum) * VL;
        if (tail > 0u) {
            CgemvAivDotTailChunk<IS_TRANS_T>(
                realAcc, imagAcc, aAddr, xAddr, static_cast<uint32_t>(fullLoopNum) * VL, tail, maskAll);
        }

        Reg::ReduceSum<float>(realAcc, realAcc, maskAll);
        Reg::ReduceSum<float>(imagAcc, imagAcc, maskAll);

        __ubuf__ float* outAddr = resultAddr + outIdx * 2u;
        Reg::Store<float>(outAddr, realAcc, 1u);
        Reg::Store<float>(outAddr + 1u, imagAcc, 1u);
    }
}

// ---------------------------------------------------------------------------
// AIV vectorized path for trans=N (reduce over columns, vectorize over rows).
// A column j is contiguous: A[i + j*lda]. x[j] is broadcast with DIST_BRC_B32.
// acc[i] += A[i,j] * x[j];  real += ar*xr - ai*xi, imag += ar*xi + ai*xr
// ---------------------------------------------------------------------------

__simd_vf__ inline void CgemvAivNormalVf(
    __ubuf__ float* aBase, __ubuf__ float* xAddr, __ubuf__ float* resultAddr, uint32_t outCount, uint32_t colStart,
    uint32_t colEnd, uint32_t outTile)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    auto maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    uint32_t outVecCount = (outCount + VL - 1u) / VL;

    for (uint32_t ov = 0u; ov < outVecCount; ++ov) {
        uint32_t outOff = ov * VL;
        uint32_t curOut = (outOff + VL > outCount) ? (outCount - outOff) : VL;
        uint32_t curMaskCount = curOut;
        auto curMask = Reg::UpdateMask<float>(curMaskCount);

        Reg::RegTensor<float> realAcc;
        Reg::RegTensor<float> imagAcc;
        Reg::Duplicate<float>(realAcc, 0.0f, maskAll);
        Reg::Duplicate<float>(imagAcc, 0.0f, maskAll);

        for (uint32_t j = colStart; j < colEnd; ++j) {
            __ubuf__ float* colAddr = aBase + (j - colStart) * outTile * 2u + outOff * 2u;

            Reg::RegTensor<float> aReal;
            Reg::RegTensor<float> aImag;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(aReal, aImag, colAddr);

            Reg::RegTensor<float> xRealBc;
            Reg::RegTensor<float> xImagBc;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(xRealBc, xAddr + (j - colStart) * 2u);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(xImagBc, xAddr + (j - colStart) * 2u + 1u);

            Reg::RegTensor<float> term;
            Reg::RegTensor<float> scratch;
            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(term, aReal, xRealBc, curMask);
            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(scratch, aImag, xImagBc, curMask);
            Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(term, term, scratch, maskAll);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(realAcc, realAcc, term, maskAll);

            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(term, aReal, xImagBc, curMask);
            Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(scratch, aImag, xRealBc, curMask);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(term, term, scratch, maskAll);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(imagAcc, imagAcc, term, maskAll);
        }

        Reg::Store<float>(resultAddr + outOff, realAcc, curOut);
        Reg::Store<float>(resultAddr + outVecCount * VL + outOff, imagAcc, curOut);
    }
}

// ---------------------------------------------------------------------------
// Vectorized UB→UB copy: replaces scalar GetValue/SetValue loop.
// ---------------------------------------------------------------------------
__simd_vf__ inline void CgemvUbCopy(__ubuf__ float* src, __ubuf__ float* dst, uint32_t floatCount)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    auto maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    uint32_t fullVecs = floatCount / VL;
    for (uint32_t v = 0u; v < fullVecs; ++v) {
        Reg::RegTensor<float> reg;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(reg, src + v * VL);
        Reg::Store<float>(dst + v * VL, reg, VL);
    }
    uint32_t tail = floatCount - fullVecs * VL;
    if (tail > 0u) {
        uint32_t tailOff = fullVecs * VL;
        Reg::RegTensor<float> reg;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(reg, src + tailOff);
        Reg::Store<float>(dst + tailOff, reg, tail);
    }
}

// ---------------------------------------------------------------------------
// AIV engine: trans=T/C
// ---------------------------------------------------------------------------

// Rebind GM pointers for the next batch without re-allocating UB buffers.
__aicore__ inline void CgemvSetBatchPointers(
    GlobalTensor<float>& aGm, GlobalTensor<float>& xGm, GlobalTensor<float>& yGm, __gm__ float* aPtr,
    __gm__ float* xPtr, __gm__ float* yPtr)
{
    aGm.SetGlobalBuffer(aPtr);
    xGm.SetGlobalBuffer(xPtr);
    yGm.SetGlobalBuffer(yPtr);
}

// Fold beta*y[b] into the scaled complex product. Shared by both engines so the
// read-modify-write ordering against GM stays identical.
__aicore__ inline void CgemvAccumulateBeta(
    const CgemvBatchedTilingData& tiling, GlobalTensor<float>& yGm, uint32_t outIdx, float& outR, float& outI)
{
    if (tiling.betaIsZero == 0U) {
        uint64_t yOff = static_cast<uint64_t>(outIdx) * 2U;
        float yOldR = yGm.GetValue(yOff);
        float yOldI = yGm.GetValue(yOff + 1U);
        outR += tiling.betaR * yOldR - tiling.betaI * yOldI;
        outI += tiling.betaI * yOldR + tiling.betaR * yOldI;
    }
}

class CgemvBatchedAIVDot {
public:
    static constexpr uint32_t VL = 256u / sizeof(float);
    // Config A: small dotSize (<=512) → larger out tile to reduce writeback overhead
    static constexpr uint32_t AIV_OUT_TILE_A = 56u;
    static constexpr uint32_t AIV_DOT_TILE_A = 512u;
    // Config B: large dotSize (>512) → larger dot tile to reduce dot-loop overhead
    static constexpr uint32_t AIV_OUT_TILE_B = 28u;
    static constexpr uint32_t AIV_DOT_TILE_B = 1024u;
    // Config C: dotSize fits one narrow dot pitch. Keeping the row tile small
    // keeps the DMA block count low while the tighter pitch removes most of the
    // UB write gap.
    static constexpr uint32_t AIV_OUT_TILE_C = 56u;
    static constexpr uint32_t AIV_DOT_TILE_C = 128u;
    // Both configs use the same aBuf size (229376 bytes); allocate for the max
    static constexpr uint32_t AIV_OUT_TILE_MAX = AIV_OUT_TILE_A; // 56 > 28
    static constexpr uint32_t AIV_DOT_TILE_MAX = AIV_DOT_TILE_B; // 1024 > 512
    // Upper bound on the dot dimension routed to the vector path; beyond this
    // the dot loop no longer amortises the per-tile setup cost.
    static constexpr uint32_t AIV_DOT_DIM_LIMIT = 8192u;

    __aicore__ inline CgemvBatchedAIVDot() {}

    __aicore__ inline void Init(
        __gm__ float* aGm, __gm__ float* xGm, __gm__ float* yGm, const CgemvBatchedTilingData& tdata)
    {
        tiling_ = tdata;
        aGm_.SetGlobalBuffer(aGm);
        xGm_.SetGlobalBuffer(xGm);
        yGm_.SetGlobalBuffer(yGm);

        // Allocate for the worst-case config (both configs have equal aBuf size)
        uint32_t numChunksB = (AIV_DOT_TILE_MAX + VL - 1u) / VL;
        uint32_t dotTilePaddedB = numChunksB * 2u * VL;
        uint32_t dotTileBytesB = dotTilePaddedB * sizeof(float);

        pipe_.InitBuffer(aBuf_, AIV_OUT_TILE_B * dotTileBytesB);
        pipe_.InitBuffer(xBuf_, dotTileBytesB);
        pipe_.InitBuffer(partialBuf_, (AIV_OUT_TILE_MAX * 2u + VL) * sizeof(float));
        pipe_.InitBuffer(resultBuf_, (AIV_OUT_TILE_MAX * 2u + VL) * sizeof(float));
        pipe_.InitBuffer(yBuf_, (AIV_OUT_TILE_MAX * 2u + VL) * sizeof(float));
    }

    // Rebind GM pointers for the next batch without re-allocating UB buffers.
    __aicore__ inline void SetBatchPointers(__gm__ float* aGm, __gm__ float* xGm, __gm__ float* yGm)
    {
        CgemvSetBatchPointers(aGm_, xGm_, yGm_, aGm, xGm, yGm);
    }

    __aicore__ inline void ProcessBatch(uint32_t outStart, uint32_t outEnd)
    {
        if (tiling_.alphaIsZero != 0u) {
            ProcessBatchImpl<false>(outStart, outEnd);
        } else if (tiling_.trans == 2U) {
            ProcessBatchImpl<true>(outStart, outEnd);
        } else {
            ProcessBatchImpl<false>(outStart, outEnd);
        }
    }

private:
    // Pick the (outTile, dotTile) pair for this batch. The choice is driven by
    // dotSize and lda: a full A column (lda == dotSz) lets the DMA move whole
    // columns in one burst, so x is cached once per batch instead of per tile.
    __aicore__ inline void SelectTileConfig(
        uint32_t dotSz, uint32_t outCount, bool& cacheX, uint32_t& outTile, uint32_t& dotTile) const
    {
        cacheX = (tiling_.lda == dotSz) && (dotSz > AIV_DOT_TILE_B) && (dotSz <= AIV_DOT_DIM_LIMIT) &&
                 (dotSz % 1024u == 0u) && (outCount >= 2u);
        if (cacheX) {
            outTile = 2u;
            dotTile = dotSz;
        } else if (dotSz <= AIV_DOT_TILE_C) {
            outTile = AIV_OUT_TILE_C;
            dotTile = AIV_DOT_TILE_C;
        } else if (dotSz <= AIV_DOT_TILE_A) {
            outTile = AIV_OUT_TILE_A;
            dotTile = AIV_DOT_TILE_A;
        } else {
            outTile = AIV_OUT_TILE_B;
            dotTile = AIV_DOT_TILE_B;
        }
    }

    // Stage one dot slice of x into dst, splitting the burst every kXChunk
    // elements so the block length stays on the HW fast path. The slice always
    // lands at the start of dst; dotOff only selects the GM source offset.
    __aicore__ inline void CopyXChunk(LocalTensor<float>& dst, uint32_t dstOff, uint32_t dotOff, uint32_t dotCnt) const
    {
        constexpr uint32_t kXChunk = 1024u;
        uint32_t incx = static_cast<uint32_t>(tiling_.incx);
        if (incx == 1u) {
            for (uint32_t xo = 0u; xo < dotCnt; xo += kXChunk) {
                uint32_t xc = (xo + kXChunk > dotCnt) ? (dotCnt - xo) : kXChunk;
                DataCopyPad(
                    dst[dstOff + xo * 2u], xGm_[(dotOff + xo) * 2u],
                    DataCopyExtParams{1, static_cast<uint32_t>(xc * 2u * sizeof(float)), 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        } else {
            uint32_t elemBytes = 2u * sizeof(float);
            int64_t srcStrideX = static_cast<int64_t>(incx - 1u) * static_cast<int64_t>(elemBytes);
            for (uint32_t xo = 0u; xo < dotCnt; xo += kXChunk) {
                uint32_t xc = (xo + kXChunk > dotCnt) ? (dotCnt - xo) : kXChunk;
                DataCopyPad(
                    dst[dstOff + xo * 2u], xGm_[(dotOff + xo) * incx * 2u],
                    DataCopyExtParams{static_cast<uint16_t>(xc), elemBytes, srcStrideX, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
        }
    }

    // Cache the whole x vector in UB once per batch (contiguous-column path).
    // It is shared by every output of the batch, so staging it once amortises
    // the DMA over the whole dot reduction.
    __aicore__ inline void CacheXVector(LocalTensor<float>& aLocal, uint32_t xCacheOff, uint32_t dotSz) const
    {
        if (tiling_.alphaIsZero == 0u) {
            CopyXChunk(aLocal, xCacheOff, 0u, dotSz);
        }
    }

    // Double-buffered A prefetch for the compact single-dot config: the DMA
    // for tile i+1 is issued before tile i is reduced, so MTE2 overlaps the
    // vector pass. Only this config has two A tiles that fit in the UB budget
    // reserved for aBuf_.
    __aicore__ inline void PrefetchDualA(
        LocalTensor<float>& aLocal, LocalTensor<float>& xLocal, uint32_t outStart, uint32_t outEnd, uint32_t outTile,
        uint32_t dotSz, uint32_t dotTileBytes)
    {
        uint32_t incx = static_cast<uint32_t>(tiling_.incx);
        if (incx == 1u) {
            DataCopyPad(
                xLocal, xGm_, DataCopyExtParams{1, static_cast<uint32_t>(dotSz * 2u * sizeof(float)), 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            uint32_t elemBytes = 2u * sizeof(float);
            int64_t srcStrideX = static_cast<int64_t>(incx - 1u) * static_cast<int64_t>(elemBytes);
            DataCopyPad(
                xLocal, xGm_, DataCopyExtParams{static_cast<uint16_t>(dotSz), elemBytes, srcStrideX, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        uint32_t firstCnt = (outStart + outTile > outEnd) ? (outEnd - outStart) : outTile;
        CopyATile(aLocal, 0u, outStart, firstCnt, dotSz, dotTileBytes);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    // Copy rows [outOff, outOff+outCnt) of one dot slice into the A tile buffer.
    // A row is dotCnt complex values with lda stride; the destination is padded
    // to the tile pitch so the vector loop can always run full-width.
    __aicore__ inline void CopyATile(
        LocalTensor<float>& aLocal, uint32_t dstOff, uint32_t outOff, uint32_t outCnt, uint32_t dotCnt,
        uint32_t dotTileBytes)
    {
        uint32_t blk = dotCnt * 2u * sizeof(float);
        uint32_t aligned = (blk + 31u) & ~31u;
        uint32_t dstStride = (dotTileBytes - aligned) / 32u;
        int64_t sStride = static_cast<int64_t>(tiling_.lda - dotCnt) * 2u * sizeof(float);
        DataCopyPad(
            aLocal[dstOff], aGm_[static_cast<uint64_t>(outOff) * tiling_.lda * 2u],
            DataCopyExtParams{static_cast<uint16_t>(outCnt), blk, sStride, dstStride, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
    }

    // One buffered A tile: prefetch the next tile, wait for the current one and
    // hand it to the dot kernel. chunkIdx tracks which buffer is in flight.
    template <bool IS_TRANS_T>
    __aicore__ inline void RunDualADot(
        LocalTensor<float>& aLocal, LocalTensor<float>& xLocal, LocalTensor<float>& resultLocal, uint32_t outOff,
        uint32_t outEnd, uint32_t outCnt, uint32_t outTile, uint32_t aBufStride, uint32_t dotSz, uint32_t dotTilePadded,
        uint32_t dotTileBytes, uint32_t& chunkIdx)
    {
        uint32_t bufSel = chunkIdx & 1u;
        uint32_t nextOff = outOff + outTile;
        if (nextOff < outEnd) {
            uint32_t nextCnt = (nextOff + outTile > outEnd) ? (outEnd - nextOff) : outTile;
            CopyATile(aLocal, bufSel == 0u ? aBufStride : 0u, nextOff, nextCnt, dotSz, dotTileBytes);
            SetFlag<HardEvent::MTE2_V>(bufSel == 0u ? EVENT_ID1 : EVENT_ID0);
        }
        if (bufSel == 0u) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
        if (chunkIdx == 0u) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        }

        __ubuf__ float* aAddr = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
        __ubuf__ float* xAddr = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
        __ubuf__ float* accAddr = reinterpret_cast<__ubuf__ float*>(resultLocal.GetPhyAddr());
        asc_vf_call<CgemvAivDotAllVf<IS_TRANS_T>>(
            aAddr + bufSel * aBufStride, xAddr, accAddr, dotSz, 0u, outCnt, dotTilePadded);
        PipeBarrier<PIPE_V>();
        chunkIdx++;
    }

    // Non-buffered dot path: copy one reduction slice at a time and accumulate
    // it into the result tile.
    template <bool IS_TRANS_T>
    __aicore__ inline void RunLoopDot(
        LocalTensor<float>& aLocal, LocalTensor<float>& xLocal, LocalTensor<float>& partialLocal,
        LocalTensor<float>& resultLocal, uint32_t outOff, uint32_t outCnt, uint32_t outCntAlign, uint32_t dotTile,
        uint32_t dotSz, uint32_t dotTilePadded, uint32_t dotTileBytes, bool cacheX, uint32_t xCacheOff,
        bool singleDotCfg, bool singleDot)
    {
        __ubuf__ float* aAddr = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
        __ubuf__ float* xAddr = cacheX ? (reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr()) + xCacheOff) :
                                         reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
        __ubuf__ float* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());
        __ubuf__ float* accAddr = singleDot ? reinterpret_cast<__ubuf__ float*>(resultLocal.GetPhyAddr()) : partialAddr;

        bool xCached = false;
        for (uint32_t dotOff = 0; dotOff < dotSz; dotOff += dotTile) {
            uint32_t dotCnt = (dotOff + dotTile > dotSz) ? (dotSz - dotOff) : dotTile;

            if (dotOff > 0u) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            CopyDotTile(
                aLocal, xLocal, aAddr, xAddr, outOff, outCnt, dotOff, dotCnt, dotTileBytes, cacheX, singleDotCfg,
                xCached);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            asc_vf_call<CgemvAivDotAllVf<IS_TRANS_T>>(aAddr, xAddr, accAddr, dotCnt, 0u, outCnt, dotTilePadded);
            PipeBarrier<PIPE_V>();

            if (!singleDot) {
                Add(resultLocal, resultLocal, partialLocal, outCntAlign);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    template <bool IS_TRANS_T>
    __aicore__ inline void ProcessBatchImpl(uint32_t outStart, uint32_t outEnd)
    {
        uint32_t dotSz = tiling_.dotSize;

        // Select tiling config based on dotSize: small dot → larger out tile
        // When a full A column is contiguous (lda == dotSz) the DMA can move
        // whole columns in one strided-free burst, so switch to a
        // contiguous-column config and cache x once per batch (x is identical
        // for every output of a batch).
        bool cacheX = false;
        uint32_t outTile = 0u;
        uint32_t dotTile = 0u;
        SelectTileConfig(dotSz, outEnd - outStart, cacheX, outTile, dotTile);

        LocalTensor<float> aLocal = aBuf_.Get<float>();
        LocalTensor<float> xLocal = xBuf_.Get<float>();
        LocalTensor<float> partialLocal = partialBuf_.Get<float>();
        LocalTensor<float> resultLocal = resultBuf_.Get<float>();
        LocalTensor<float> yLocal = yBuf_.Get<float>();

        uint32_t numChunks = (dotTile + VL - 1u) / VL;
        uint32_t dotTilePadded = numChunks * 2u * VL;
        uint32_t dotTileBytes = dotTilePadded * sizeof(float);

        // Contiguous-column path: cache the whole x vector in UB once per
        // batch (it is shared by every output). It is placed right after the
        // A columns inside aBuf_, so no extra UB allocation is required.
        uint32_t xCacheOff = 0u;
        if (cacheX) {
            xCacheOff = outTile * dotTilePadded;
            CacheXVector(aLocal, xCacheOff, dotSz);
        }

        bool singleDotCfg = (dotSz <= dotTile);

        uint32_t aBufStride = outTile * dotTilePadded;
        bool dualA = (!cacheX) && singleDotCfg && (dotSz <= AIV_DOT_TILE_C) && (tiling_.alphaIsZero == 0u);
        uint32_t chunkIdx = 0u;
        if (dualA) {
            PrefetchDualA(aLocal, xLocal, outStart, outEnd, outTile, dotSz, dotTileBytes);
        }

        for (uint32_t outOff = outStart; outOff < outEnd; outOff += outTile) {
            uint32_t outCnt = (outOff + outTile > outEnd) ? (outEnd - outOff) : outTile;
            uint32_t outCntAlign = (outCnt * 2u + VL - 1u) & ~(VL - 1u);

            // A single dot slice needs no cross-slice accumulation: let the
            // vector kernel write straight into the result tile and skip the
            // zero-init plus the element-wise Add afterwards.
            bool singleDot = (dotSz <= dotTile);
            if (!singleDot) {
                Duplicate(resultLocal, 0.0f, outCntAlign);
                PipeBarrier<PIPE_V>();
            }

            if (dualA) {
                RunDualADot<IS_TRANS_T>(
                    aLocal, xLocal, resultLocal, outOff, outEnd, outCnt, outTile, aBufStride, dotSz, dotTilePadded,
                    dotTileBytes, chunkIdx);
            } else if (tiling_.alphaIsZero == 0u) {
                RunLoopDot<IS_TRANS_T>(
                    aLocal, xLocal, partialLocal, resultLocal, outOff, outCnt, outCntAlign, dotTile, dotSz,
                    dotTilePadded, dotTileBytes, cacheX, xCacheOff, singleDotCfg, singleDot);
            }

            WriteDotResult(resultLocal, yLocal, outOff, outCnt, outCntAlign);
        }
    }

    // Copy one dot slice of A (and the matching x slice unless it is cached)
    // into the working tiles, then hand them to the vector kernel.
    __aicore__ inline void CopyDotTile(
        LocalTensor<float>& aLocal, LocalTensor<float>& xLocal, __ubuf__ float* aAddr, __ubuf__ float* xAddr,
        uint32_t outOff, uint32_t outCnt, uint32_t dotOff, uint32_t dotCnt, uint32_t dotTileBytes, bool cacheX,
        bool singleDotCfg, bool& xCached)
    {
        uint32_t aBlkLenBytes = dotCnt * 2u * sizeof(float);
        uint32_t aBlkLenAligned = (aBlkLenBytes + 31u) & ~31u;
        uint32_t matDstStride = (dotTileBytes - aBlkLenAligned) / 32u;
        uint64_t aSrcOffset = static_cast<uint64_t>(outOff) * tiling_.lda * 2u + dotOff * 2u;
        int64_t srcStride = static_cast<int64_t>(tiling_.lda - dotCnt) * 2u * sizeof(float);

        if (cacheX) {
            // Contiguous columns: splice the per-column payload into 8 KiB
            // sub-blocks (blockLen > 8192 triggers a HW slow path) and merge
            // them into a single bulk descriptor.
            constexpr uint32_t kMaxBlkBytes = 8192u;
            uint32_t nSub = aBlkLenBytes / kMaxBlkBytes;
            DataCopyPad(
                aLocal, aGm_[aSrcOffset],
                DataCopyExtParams{static_cast<uint16_t>(outCnt * nSub), kMaxBlkBytes, static_cast<int64_t>(0), 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            DataCopyPad(
                aLocal, aGm_[aSrcOffset],
                DataCopyExtParams{static_cast<uint16_t>(outCnt), aBlkLenBytes, srcStride, matDstStride, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        if (!cacheX) {
            // x does not depend on the output tile, so with a single dot slice
            // it only has to be staged once per batch.
            if (!(singleDotCfg && xCached)) {
                CopyXChunk(xLocal, 0u, dotOff, dotCnt);
                xCached = true;
            }
        }
    }

    // Apply alpha/beta and write the reduced tile back to GM. With
    // alpha=1+0i, beta=0 the interleaved result is handed to the copy engine
    // directly, skipping the scalar scaling pass.
    __aicore__ inline void WriteDotResult(
        LocalTensor<float>& resultLocal, LocalTensor<float>& yLocal, uint32_t outOff, uint32_t outCnt,
        uint32_t outCntAlign)
    {
        bool directWrite = (tiling_.betaIsZero == 1u) && (tiling_.alphaR == 1.0f) && (tiling_.alphaI == 0.0f);
        if (directWrite) {
            if (tiling_.alphaIsZero != 0u) {
                Duplicate(resultLocal, 0.0f, outCntAlign);
            }

            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            DataCopyPad(
                yGm_[static_cast<uint64_t>(outOff) * 2U], resultLocal,
                DataCopyExtParams{1, outCnt * 2u * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        } else {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);

            for (uint32_t i = 0; i < outCnt; ++i) {
                float accR = 0.0f;
                float accI = 0.0f;
                if (tiling_.alphaIsZero == 0u) {
                    accR = resultLocal.GetValue(i * 2u);
                    accI = resultLocal.GetValue(i * 2u + 1u);
                }
                float outR = tiling_.alphaR * accR - tiling_.alphaI * accI;
                float outI = tiling_.alphaR * accI + tiling_.alphaI * accR;

                uint32_t outIdx = outOff + i;
                CgemvAccumulateBeta(tiling_, yGm_, outIdx, outR, outI);

                yLocal.SetValue(i * 2u, outR);
                yLocal.SetValue(i * 2u + 1u, outI);
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(
                yGm_[static_cast<uint64_t>(outOff) * 2U], yLocal,
                DataCopyExtParams{1, outCnt * 2u * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    CgemvBatchedTilingData tiling_;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> aBuf_;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> partialBuf_;
    TBuf<QuePosition::VECCALC> resultBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;
};

// ---------------------------------------------------------------------------
// Vectorized split→interleaved layout conversion for directWrite path.
// Reads real[0..outCnt-1] and imag[imagBase..imagBase+outCnt-1] from srcSplit
// and writes interleaved (re,im,re,im,...) to dstIntlv for full VL vectors.
// ---------------------------------------------------------------------------
__simd_vf__ inline void CgemvAivNormalInterleave(
    __ubuf__ float* srcSplit, __ubuf__ float* dstIntlv, uint32_t outCnt, uint32_t imagBase)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    auto maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    uint32_t fullVecs = outCnt / VL;
    for (uint32_t v = 0u; v < fullVecs; ++v) {
        Reg::RegTensor<float> realReg;
        Reg::RegTensor<float> imagReg;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(realReg, srcSplit + v * VL);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(imagReg, srcSplit + imagBase + v * VL);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(dstIntlv + v * 2u * VL, realReg, imagReg, maskAll);
    }
}

// Interleave two separate UB sources (realSrc, imagSrc) into interleaved dst.
__simd_vf__ inline void CgemvInterleaveTwoSrc(
    __ubuf__ float* realSrc, __ubuf__ float* imagSrc, __ubuf__ float* dstIntlv, uint32_t outCnt)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    auto maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    uint32_t fullVecs = outCnt / VL;
    for (uint32_t v = 0u; v < fullVecs; ++v) {
        Reg::RegTensor<float> realReg;
        Reg::RegTensor<float> imagReg;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(realReg, realSrc + v * VL);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(imagReg, imagSrc + v * VL);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(dstIntlv + v * 2u * VL, realReg, imagReg, maskAll);
    }
}
// A column j is contiguous: copy with DataCopyPad blockLen=outCnt, dstStride=0.
// ---------------------------------------------------------------------------

class CgemvBatchedAIVNormal {
public:
    static constexpr uint32_t VL = 256u / sizeof(float);
    // Config A: many rows per core -> wide output tile, 1 writeback, short columns
    static constexpr uint32_t AIV_OUT_TILE_A = 128u;
    static constexpr uint32_t AIV_COL_TILE_A = 248u;
    // Config B: few rows per core (<= 64) -> narrow output tile so the column
    // tile can be doubled; keeps DMA descriptors large and cuts loop count.
    static constexpr uint32_t AIV_OUT_TILE_B = 64u;
    static constexpr uint32_t AIV_COL_TILE_B = 496u;
    static constexpr uint32_t AIV_OUT_TILE_MAX = (AIV_OUT_TILE_A > AIV_OUT_TILE_B) ? AIV_OUT_TILE_A : AIV_OUT_TILE_B;
    static constexpr uint32_t AIV_COL_TILE_MAX = (AIV_COL_TILE_A > AIV_COL_TILE_B) ? AIV_COL_TILE_A : AIV_COL_TILE_B;
    static constexpr uint32_t AIV_OUT_TILE_PAD_A = (AIV_OUT_TILE_A + VL - 1u) & ~(VL - 1u);
    static constexpr uint32_t AIV_OUT_TILE_PAD_B = (AIV_OUT_TILE_B + VL - 1u) & ~(VL - 1u);
    static constexpr uint32_t AIV_A_TILE_A = AIV_COL_TILE_A * AIV_OUT_TILE_PAD_A * 2u;
    static constexpr uint32_t AIV_A_TILE_B = AIV_COL_TILE_B * AIV_OUT_TILE_PAD_B * 2u;
    static constexpr uint32_t AIV_A_TILE_MAX = (AIV_A_TILE_A > AIV_A_TILE_B) ? AIV_A_TILE_A : AIV_A_TILE_B;
    static constexpr uint32_t AIV_X_TILE_MAX =
        (AIV_COL_TILE_A > AIV_COL_TILE_B) ? (AIV_COL_TILE_A * 2u) : (AIV_COL_TILE_B * 2u);
    static constexpr uint32_t AIV_OUT_TILE_PAD_MAX =
        (AIV_OUT_TILE_PAD_A > AIV_OUT_TILE_PAD_B) ? AIV_OUT_TILE_PAD_A : AIV_OUT_TILE_PAD_B;
    // Config C (2D column split): one core owns ALL output rows of a batch and
    // a slice of the reduction dimension. Rows are walked in chunks of
    // AIV2D_OUT_PAD so outSize is unbounded while a single DMA burst still
    // spans whole padded columns.
    static constexpr uint32_t AIV2D_OUT_PAD = 256u;
    static constexpr uint32_t AIV2D_COL_TILE = 124u;

    __aicore__ inline CgemvBatchedAIVNormal() {}

    __aicore__ inline void Init(
        __gm__ float* aGm, __gm__ float* xGm, __gm__ float* yGm, const CgemvBatchedTilingData& tdata)
    {
        tiling_ = tdata;
        aGm_.SetGlobalBuffer(aGm);
        xGm_.SetGlobalBuffer(xGm);
        yGm_.SetGlobalBuffer(yGm);

        // Rows are padded to a multiple of VL so the vector scope can always
        // run with a full-width mask; the short tail row tile is the only one
        // that uses a partial mask.
        // Pick the tiling config once per core: with at most 64 output rows a
        // narrower tile frees UB for a ~2x wider column tile, which keeps each
        // DMA descriptor large when the per-core row count is small.
        smallRows_ = (tdata.batchPerCore <= 64u);

        // Both configs use the same aBuf footprint (248*128 == 496*64), so the
        // allocation stays a compile-time constant and keeps UB layout stable.
        pipe_.InitBuffer(aBuf_, AIV_A_TILE_MAX * sizeof(float));
        pipe_.InitBuffer(xBuf_, AIV_X_TILE_MAX * sizeof(float));
        pipe_.InitBuffer(partialBuf_, AIV2D_OUT_PAD * 2u * sizeof(float));
        pipe_.InitBuffer(yBuf_, AIV2D_OUT_PAD * 2u * sizeof(float));
    }

    // Same batch rebinding as the dot engine, shared via the file-level helper.
    __aicore__ inline void SetBatchPointers(__gm__ float* aGm, __gm__ float* xGm, __gm__ float* yGm)
    {
        CgemvSetBatchPointers(aGm_, xGm_, yGm_, aGm, xGm, yGm);
    }

    __aicore__ inline void ProcessBatch(uint32_t outStart, uint32_t outEnd)
    {
        if (smallRows_) {
            ProcessBatchTiled<AIV_OUT_TILE_B, AIV_COL_TILE_B, AIV_OUT_TILE_PAD_B>(outStart, outEnd);
        } else {
            ProcessBatchTiled<AIV_OUT_TILE_A, AIV_COL_TILE_A, AIV_OUT_TILE_PAD_A>(outStart, outEnd);
        }
    }

    // Zero the whole batch output so the per-slice atomic sums start from 0.
    // Done in row chunks because the scratch buffer only holds AIV2D_OUT_PAD
    // complex rows, while outSize may be larger.
    __aicore__ inline void ZeroOutput2D(uint32_t outSize)
    {
        LocalTensor<float> t = yBuf_.Get<float>();
        for (uint32_t r0 = 0; r0 < outSize; r0 += AIV2D_OUT_PAD) {
            uint32_t cnt = (r0 + AIV2D_OUT_PAD > outSize) ? (outSize - r0) : AIV2D_OUT_PAD;
            uint32_t cntPadded = (cnt * 2u + VL - 1u) & ~(VL - 1u);
            Duplicate(t, 0.0f, cntPadded);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(yGm_[r0], t, DataCopyExtParams{1, cnt * 2u * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    // Reduce one column tile into accLocal: the first tile seeds accLocal, later
    // tiles accumulate through partialLocal. OUT_TILE_PAD stays a template
    // parameter so it remains a compile-time constant in the vector loop.
    template <uint32_t OUT_TILE_PAD>
    __aicore__ inline void ReduceColumnTile(
        LocalTensor<float>& aLocal, LocalTensor<float>& xLocal, LocalTensor<float>& accLocal,
        LocalTensor<float>& partialLocal, __ubuf__ float* aAddr, __ubuf__ float* xAddr, __ubuf__ float* accAddr,
        __ubuf__ float* partialAddr, uint64_t aSrcOffset, uint32_t colOff, uint32_t colCnt, uint32_t outCnt,
        uint32_t outCntPadded, uint32_t aBlkLenBytes, int64_t srcStride, uint32_t aDstStride, bool& firstCol)
    {
        if (!firstCol) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        DataCopyPad(
            aLocal, aGm_[aSrcOffset],
            DataCopyExtParams{static_cast<uint16_t>(colCnt), aBlkLenBytes, srcStride, aDstStride, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(
            xLocal, xGm_[colOff * 2u],
            DataCopyExtParams{1, static_cast<uint32_t>(colCnt * 2u * sizeof(float)), 0, 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        if (firstCol) {
            asc_vf_call<CgemvAivNormalVf>(aAddr, xAddr, accAddr, outCnt, 0u, colCnt, OUT_TILE_PAD);
        } else {
            asc_vf_call<CgemvAivNormalVf>(aAddr, xAddr, partialAddr, outCnt, 0u, colCnt, OUT_TILE_PAD);
            PipeBarrier<PIPE_V>();
            Add(accLocal, accLocal, partialLocal, outCntPadded * 2u);
        }
        PipeBarrier<PIPE_V>();
        firstCol = false;
    }

    // Scale the split (real, imag) accumulator in yLocal by alpha and interleave
    // it back into aLocal. The scalar tail rows are handled here too so the
    // caller can copy aLocal straight to GM.
    __aicore__ inline void ScaleAndInterleave(
        LocalTensor<float>& aLocal, LocalTensor<float>& partialLocal, LocalTensor<float>& yLocal, __ubuf__ float* aAddr,
        __ubuf__ float* partialAddr, __ubuf__ float* yAddr, uint32_t outCnt, uint32_t imagBase)
    {
        uint32_t outCntPadded = (outCnt + VL - 1u) & ~(VL - 1u);
        Muls(partialLocal, yLocal, tiling_.alphaR, outCntPadded);
        Muls(aLocal, yLocal[imagBase], -tiling_.alphaI, outCntPadded);
        Add(partialLocal, partialLocal, aLocal, outCntPadded);
        Muls(aLocal, yLocal, tiling_.alphaI, outCntPadded);
        Muls(yLocal[imagBase], yLocal[imagBase], tiling_.alphaR, outCntPadded);
        Add(yLocal[imagBase], yLocal[imagBase], aLocal, outCntPadded);

        PipeBarrier<PIPE_V>();
        asc_vf_call<CgemvInterleaveTwoSrc>(partialAddr, yAddr + imagBase, aAddr, outCnt);

        uint32_t fullVecs = outCnt / VL;
        uint32_t tailStart = fullVecs * VL;
        if (tailStart < outCnt) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (uint32_t i = tailStart; i < outCnt; ++i) {
                float outR = partialLocal.GetValue(i);
                float outI = yLocal.GetValue(imagBase + i);
                aLocal.SetValue(i * 2u, outR);
                aLocal.SetValue(i * 2u + 1u, outI);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    // 2D path: reduce columns [col0, col1) for ALL outSize rows, scale by alpha
    // and atomically accumulate into y. Each core therefore reads long
    // contiguous column runs instead of short strided row fragments.
    __aicore__ inline void ProcessColSlice2D(uint32_t col0, uint32_t col1, uint32_t outSize)
    {
        constexpr uint32_t colTile = AIV2D_COL_TILE;
        constexpr uint32_t outTile = AIV2D_OUT_PAD;
        uint32_t lda = tiling_.lda;

        LocalTensor<float> aLocal = aBuf_.Get<float>();
        LocalTensor<float> xLocal = xBuf_.Get<float>();
        LocalTensor<float> partialLocal = partialBuf_.Get<float>();
        LocalTensor<float> yLocal = yBuf_.Get<float>();

        __ubuf__ float* aAddr = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
        __ubuf__ float* xAddr = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
        __ubuf__ float* yAddr = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
        __ubuf__ float* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());

        // Rows are processed in chunks of the scratch tile so outSize may exceed
        // the UB row capacity; each chunk still reads long contiguous runs of
        // every column before the slice is atomically merged into y.
        for (uint32_t r0 = 0; r0 < outSize; r0 += outTile) {
            uint32_t outCnt = (r0 + outTile > outSize) ? (outSize - r0) : outTile;
            uint32_t outCntPadded = (outCnt + VL - 1u) & ~(VL - 1u);
            uint32_t outVecCount = (outCnt + VL - 1u) / VL;
            uint32_t imagBase = outVecCount * VL;

            Duplicate(yLocal, 0.0f, outCntPadded * 2u);
            PipeBarrier<PIPE_V>();

            uint32_t aBlkLenBytes = outCnt * 2u * sizeof(float);
            uint32_t aBlkLenAligned = (aBlkLenBytes + 31u) & ~31u;
            uint32_t aDstStride = (outTile * 2u * sizeof(float) - aBlkLenAligned) / 32u;
            int64_t srcStride = static_cast<int64_t>(lda - outCnt) * 2u * sizeof(float);

            // alpha == 0: the contract is y = beta*y = 0 here (this path is only
            // enabled for beta == 0), so A and x must not be read. Skipping the
            // reduction also keeps NaN/Inf in the operands out of the atomic
            // merge, matching the dot / SIMT paths.
            if (tiling_.alphaIsZero == 0u) {
                bool firstCol = true;
                for (uint32_t colOff = col0; colOff < col1; colOff += colTile) {
                    uint32_t colCnt = (colOff + colTile > col1) ? (col1 - colOff) : colTile;
                    uint64_t aSrcOffset = static_cast<uint64_t>(colOff) * lda * 2u + static_cast<uint64_t>(r0) * 2u;
                    ReduceColumnTile<outTile>(
                        aLocal, xLocal, yLocal, partialLocal, aAddr, xAddr, yAddr, partialAddr, aSrcOffset, colOff,
                        colCnt, outCnt, outCntPadded, aBlkLenBytes, srcStride, aDstStride, firstCol);
                }

                ScaleAndInterleave(aLocal, partialLocal, yLocal, aAddr, partialAddr, yAddr, outCnt, imagBase);

                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                SetAtomicAdd<float>();
                DataCopyPad(
                    yGm_[r0], aLocal,
                    DataCopyExtParams{1, outCnt * 2u * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
                SetAtomicNone();
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

private:
    // Write one reduced output tile back to GM. With beta == 0 the scaled,
    // interleaved tile is handed straight to the copy engine; otherwise the
    // scalar path folds beta*y in before the copy.
    __aicore__ inline void WriteTiledResult(
        LocalTensor<float>& aLocal, LocalTensor<float>& partialLocal, LocalTensor<float>& yLocal, __ubuf__ float* aAddr,
        __ubuf__ float* partialAddr, __ubuf__ float* yAddr, uint32_t outOff, uint32_t outCnt, uint32_t imagBase)
    {
        if (tiling_.betaIsZero != 0U) {
            ScaleAndInterleave(aLocal, partialLocal, yLocal, aAddr, partialAddr, yAddr, outCnt, imagBase);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(
                yGm_[static_cast<uint64_t>(outOff) * 2U], aLocal,
                DataCopyExtParams{1, outCnt * 2u * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            return;
        }

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        float accRArr[128], accIArr[128];
        for (uint32_t i = 0; i < outCnt; ++i) {
            accRArr[i] = yLocal.GetValue(i);
            accIArr[i] = yLocal.GetValue(imagBase + i);
        }

        for (uint32_t i = 0; i < outCnt; ++i) {
            float accR = accRArr[i];
            float accI = accIArr[i];
            float outR = tiling_.alphaR * accR - tiling_.alphaI * accI;
            float outI = tiling_.alphaR * accI + tiling_.alphaI * accR;

            CgemvAccumulateBeta(tiling_, yGm_, outOff + i, outR, outI);
            yLocal.SetValue(i * 2u, outR);
            yLocal.SetValue(i * 2u + 1u, outI);
        }

        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(
            yGm_[static_cast<uint64_t>(outOff) * 2U], yLocal,
            DataCopyExtParams{1, outCnt * 2u * static_cast<uint32_t>(sizeof(float)), 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    // Both tiling configs are instantiated separately so the tile sizes stay
    // compile-time constants; passing them through members instead blocks
    // constant folding inside the copy/vector loops.
    template <uint32_t OUT_TILE, uint32_t COL_TILE, uint32_t OUT_TILE_PAD>
    __aicore__ inline void ProcessBatchTiled(uint32_t outStart, uint32_t outEnd)
    {
        constexpr uint32_t outTile = OUT_TILE;
        constexpr uint32_t colTile = COL_TILE;
        constexpr uint32_t outTilePadded = OUT_TILE_PAD;
        uint32_t lda = tiling_.lda;
        uint32_t dotSz = tiling_.dotSize; // n

        LocalTensor<float> aLocal = aBuf_.Get<float>();
        LocalTensor<float> xLocal = xBuf_.Get<float>();
        LocalTensor<float> partialLocal = partialBuf_.Get<float>();
        LocalTensor<float> yLocal = yBuf_.Get<float>();

        for (uint32_t outOff = outStart; outOff < outEnd; outOff += outTile) {
            uint32_t outCnt = (outOff + outTile > outEnd) ? (outEnd - outOff) : outTile;
            uint32_t outCntPadded = (outCnt + VL - 1u) & ~(VL - 1u);
            uint64_t aSrcBase = static_cast<uint64_t>(outOff) * 2u;

            __ubuf__ float* aAddr = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
            __ubuf__ float* xAddr = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
            __ubuf__ float* yAddr = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
            __ubuf__ float* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());

            Duplicate(yLocal, 0.0f, outCntPadded * 2u);
            PipeBarrier<PIPE_V>();

            uint32_t outVecCount = (outCnt + VL - 1u) / VL;
            uint32_t imagBase = outVecCount * VL;

            uint32_t aBlkLenBytes = outCnt * 2u * sizeof(float);
            uint32_t aBlkLenAligned = (aBlkLenBytes + 31u) & ~31u;
            uint32_t aDstStride = (outTilePadded * 2u * sizeof(float) - aBlkLenAligned) / 32u;
            int64_t srcStride = static_cast<int64_t>(lda - outCnt) * 2u * sizeof(float);

            // alpha == 0 means A and x must not be read at all: the contract is
            // y = beta*y. Skipping the reduction keeps the accumulator at zero,
            // so a later Inf/NaN in A/x cannot turn 0*Inf into NaN. The dot and
            // SIMT paths gate the same way.
            if (tiling_.alphaIsZero == 0u) {
                bool firstCol = true;
                for (uint32_t colOff = 0; colOff < dotSz; colOff += colTile) {
                    uint32_t colCnt = (colOff + colTile > dotSz) ? (dotSz - colOff) : colTile;
                    uint64_t aSrcOffset = aSrcBase + static_cast<uint64_t>(colOff) * lda * 2u;
                    ReduceColumnTile<outTilePadded>(
                        aLocal, xLocal, yLocal, partialLocal, aAddr, xAddr, yAddr, partialAddr, aSrcOffset, colOff,
                        colCnt, outCnt, outCntPadded, aBlkLenBytes, srcStride, aDstStride, firstCol);
                }
            }

            WriteTiledResult(aLocal, partialLocal, yLocal, aAddr, partialAddr, yAddr, outOff, outCnt, imagBase);
        }
    }

    CgemvBatchedTilingData tiling_;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    bool smallRows_ = false;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> aBuf_;
    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> partialBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;
};

// ---------------------------------------------------------------------------
// SIMT path
// ---------------------------------------------------------------------------

constexpr uint32_t CGEMV_SIMT_MAX_THREADS = 2048;

// Walk the batch range [startBatch, endBatch) on one core, rebinding the GM
// pointers per batch. The first batch allocates the UB buffers via Init, the
// rest only rebind them, so the per-core UB layout is set up exactly once.
template <typename OpT>
__aicore__ inline void CgemvRunBatchRange(
    OpT& op, GM_ADDR Aarray, GM_ADDR xarray, GM_ADDR yarray, const CgemvBatchedTilingData& td, uint32_t startBatch,
    uint32_t endBatch, uint32_t startOut, uint32_t endOut, uint32_t outSz)
{
    bool inited = false;
    for (uint32_t b = startBatch; b < endBatch; ++b) {
        __gm__ float* aGm = CgemvReadPtrFromAivArray<float>(Aarray, b);
        __gm__ float* xGm = CgemvReadPtrFromAivArray<float>(xarray, b);
        __gm__ float* yGm = CgemvReadPtrFromAivArray<float>(yarray, b);

        uint32_t outStart = (b == startBatch) ? (startOut % outSz) : 0u;
        uint32_t outEnd = (b == endBatch - 1u) ? ((endOut - 1u) % outSz + 1u) : outSz;

        if (!inited) {
            op.Init(aGm, xGm, yGm, td);
            inited = true;
        } else {
            op.SetBatchPointers(aGm, xGm, yGm);
        }
        op.ProcessBatch(outStart, outEnd);
    }
}

// Linear A index of dot element (dotIdx + lane) for output outIdx.
// trans==0 walks columns (A[i + j*lda]), the transpose modes walk rows.
// issue #848: int64 域计算——大形状下 dotIdx*lda / outIdx*lda 可超 INT32_MAX，
// host 已按跨度上限显式拒绝超限形状，此处 64 位化作双保险。
__simt_callee__ __aicore__ inline int64_t CgemvSimtAIndex(
    uint32_t dotIdx, uint32_t lane, uint32_t outIdx, int32_t lda, uint32_t trans)
{
    if (trans == 0u) {
        return static_cast<int64_t>(dotIdx + lane) * lda + static_cast<int64_t>(outIdx);
    }
    return static_cast<int64_t>(outIdx) * lda + static_cast<int64_t>(dotIdx + lane);
}

// Dot reduction over the vectorised (4-way unrolled) part of the dot dimension.
// trans selects the A index mapping, trans==2 additionally conjugates A.
// issue #848: xOff 与 A/x 下标按 int64 域计算（host 跨度校验的双保险）。
__simt_callee__ __aicore__ inline void CgemvSimtDot4(
    const __gm__ float* batchA, const __gm__ float* batchX, uint32_t outIdx, uint32_t dotDim, int32_t lda, int32_t incx,
    int64_t xOff, uint32_t trans, float& accR, float& accI)
{
    uint32_t dot4 = dotDim & ~3u;
    for (uint32_t dotIdx = 0; dotIdx < dot4; dotIdx += 4u) {
        int64_t aIdx0 = CgemvSimtAIndex(dotIdx, 0u, outIdx, lda, trans);
        int64_t aIdx1 = CgemvSimtAIndex(dotIdx, 1u, outIdx, lda, trans);
        int64_t aIdx2 = CgemvSimtAIndex(dotIdx, 2u, outIdx, lda, trans);
        int64_t aIdx3 = CgemvSimtAIndex(dotIdx, 3u, outIdx, lda, trans);
        int64_t xIdx0 = static_cast<int64_t>(dotIdx) * incx;
        int64_t xIdx1 = static_cast<int64_t>(dotIdx + 1u) * incx;
        int64_t xIdx2 = static_cast<int64_t>(dotIdx + 2u) * incx;
        int64_t xIdx3 = static_cast<int64_t>(dotIdx + 3u) * incx;

        float ar0 = batchA[aIdx0 * 2];
        float ai0 = batchA[aIdx0 * 2 + 1];
        float ar1 = batchA[aIdx1 * 2];
        float ai1 = batchA[aIdx1 * 2 + 1];
        float ar2 = batchA[aIdx2 * 2];
        float ai2 = batchA[aIdx2 * 2 + 1];
        float ar3 = batchA[aIdx3 * 2];
        float ai3 = batchA[aIdx3 * 2 + 1];
        if (trans == 2u) {
            ai0 = -ai0;
            ai1 = -ai1;
            ai2 = -ai2;
            ai3 = -ai3;
        }
        float xr0 = batchX[(xOff + xIdx0) * 2];
        float xi0 = batchX[(xOff + xIdx0) * 2 + 1];
        float xr1 = batchX[(xOff + xIdx1) * 2];
        float xi1 = batchX[(xOff + xIdx1) * 2 + 1];
        float xr2 = batchX[(xOff + xIdx2) * 2];
        float xi2 = batchX[(xOff + xIdx2) * 2 + 1];
        float xr3 = batchX[(xOff + xIdx3) * 2];
        float xi3 = batchX[(xOff + xIdx3) * 2 + 1];

        accR += ar0 * xr0 - ai0 * xi0;
        accI += ar0 * xi0 + ai0 * xr0;
        accR += ar1 * xr1 - ai1 * xi1;
        accI += ar1 * xi1 + ai1 * xr1;
        accR += ar2 * xr2 - ai2 * xi2;
        accI += ar2 * xi2 + ai2 * xr2;
        accR += ar3 * xr3 - ai3 * xi3;
        accI += ar3 * xi3 + ai3 * xr3;
    }
}

// Dot reduction over the trailing (dotDim % 4) elements.
__simt_callee__ __aicore__ inline void CgemvSimtDotTail(
    const __gm__ float* batchA, const __gm__ float* batchX, uint32_t outIdx, uint32_t dotDim, int32_t lda, int32_t incx,
    int64_t xOff, uint32_t trans, float& accR, float& accI)
{
    for (uint32_t dotIdx = dotDim & ~3u; dotIdx < dotDim; dotIdx++) {
        int64_t aIdx = CgemvSimtAIndex(dotIdx, 0u, outIdx, lda, trans);
        int64_t xIdx = static_cast<int64_t>(dotIdx) * incx;
        float ar = batchA[aIdx * 2];
        float ai = batchA[aIdx * 2 + 1];
        if (trans == 2u) {
            ai = -ai;
        }
        float xr = batchX[(xOff + xIdx) * 2];
        float xi = batchX[(xOff + xIdx) * 2 + 1];
        accR += ar * xr - ai * xi;
        accI += ar * xi + ai * xr;
    }
}

// Reduce one output of one batch through the dot dimension.
__simt_callee__ __aicore__ inline void CgemvSimtDotAll(
    const __gm__ float* batchA, const __gm__ float* batchX, uint32_t outIdx, uint32_t dotDim, int32_t lda, int32_t incx,
    int64_t xOff, uint32_t trans, uint32_t alphaIsZero, float& accR, float& accI)
{
    accR = 0.0f;
    accI = 0.0f;
    if (alphaIsZero == 0u) {
        CgemvSimtDot4(batchA, batchX, outIdx, dotDim, lda, incx, xOff, trans, accR, accI);
        CgemvSimtDotTail(batchA, batchX, outIdx, dotDim, lda, incx, xOff, trans, accR, accI);
    }
}

// Scale the reduced accumulator by alpha, fold in beta*y and store back.
// issue #848: yOff/yIdx 按 int64 域计算（host 跨度校验的双保险）。
__simt_callee__ __aicore__ inline void CgemvSimtWriteOut(
    __gm__ float* batchY, uint32_t outIdx, float accR, float accI, float alphaR, float alphaI, float betaR, float betaI,
    uint32_t betaIsZero, int32_t incy, int64_t yOff)
{
    float outR = alphaR * accR - alphaI * accI;
    float outI = alphaR * accI + alphaI * accR;
    int64_t yIdx = static_cast<int64_t>(outIdx) * incy;
    if (betaIsZero == 0u) {
        float yOldR = batchY[(yOff + yIdx) * 2];
        float yOldI = batchY[(yOff + yIdx) * 2 + 1];
        outR += betaR * yOldR - betaI * yOldI;
        outI += betaR * yOldI + betaI * yOldR;
    }
    batchY[(yOff + yIdx) * 2] = outR;
    batchY[(yOff + yIdx) * 2 + 1] = outI;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_SIMT_MAX_THREADS) inline void CgemvSimt(
    uint32_t m, uint32_t n, uint64_t startOut, uint64_t numOuts, float alphaR, float alphaI, float betaR, float betaI,
    uint32_t alphaIsZero, uint32_t betaIsZero, int32_t lda, int32_t incx, int32_t incy, uint32_t trans,
    GM_ADDR AarrayBase, GM_ADDR xarrayBase, GM_ADDR yarrayBase)
{
    uint32_t outDim = (trans == 0u) ? m : n;
    uint32_t dotDim = (trans == 0u) ? n : m;

    for (uint64_t tid = threadIdx.x; tid < numOuts; tid += blockDim.x) {
        uint64_t globalOut = startOut + tid;
        uint32_t b = static_cast<uint32_t>(globalOut / outDim);
        uint32_t outIdx = static_cast<uint32_t>(globalOut % outDim);

        __gm__ float* batchA = CgemvReadPtrFromArray<float>(AarrayBase, b);
        __gm__ float* batchX = CgemvReadPtrFromArray<float>(xarrayBase, b);
        __gm__ float* batchY = CgemvReadPtrFromArray<float>(yarrayBase, b);

        // issue #848: 负步长基址偏移按 int64 域计算，避免 uint32 乘法截断
        int64_t xOff = 0;
        int64_t yOff = 0;
        if (incx < 0) {
            xOff += static_cast<int64_t>(dotDim - 1u) * static_cast<int64_t>(-incx);
        }
        if (incy < 0) {
            yOff += static_cast<int64_t>(outDim - 1u) * static_cast<int64_t>(-incy);
        }

        float accR = 0.0f;
        float accI = 0.0f;
        CgemvSimtDotAll(batchA, batchX, outIdx, dotDim, lda, incx, xOff, trans, alphaIsZero, accR, accI);
        CgemvSimtWriteOut(batchY, outIdx, accR, accI, alphaR, alphaI, betaR, betaI, betaIsZero, incy, yOff);
    }
}

// ---------------------------------------------------------------------------
// Kernel entry
// ---------------------------------------------------------------------------

// 2D column-split path (trans=N, beta==0): every core owns all output rows
// of one batch and a slice of the reduction dimension; the slices are
// merged with GM atomics after a device-wide barrier.
__aicore__ inline void CgemvRunColSplit2D(
    GM_ADDR Aarray, GM_ADDR xarray, GM_ADDR yarray, const CgemvBatchedTilingData& td)
{
    uint32_t idx = GetBlockIdx();
    uint32_t bIdx = idx / td.colSplits;
    uint32_t split = idx % td.colSplits;

    __gm__ float* aGm = CgemvReadPtrFromAivArray<float>(Aarray, bIdx);
    __gm__ float* xGm = CgemvReadPtrFromAivArray<float>(xarray, bIdx);
    __gm__ float* yGm = CgemvReadPtrFromAivArray<float>(yarray, bIdx);
    CgemvBatchedAIVNormal op;
    op.Init(aGm, xGm, yGm, td);
    if (split == 0u) {
        op.ZeroOutput2D(td.outSize);
    }
    SyncAll();
    if (bIdx < td.batchCount) {
        uint32_t col0 = split * td.colPerSplit;
        uint32_t col1 = col0 + td.colPerSplit;
        if (col1 > td.dotSize)
            col1 = td.dotSize;
        if (col0 < col1) {
            op.ProcessColSlice2D(col0, col1, td.outSize);
        }
    }
}

extern "C" __global__ __aicore__ void cgemv_batched(
    GM_ADDR Aarray, GM_ADDR xarray, GM_ADDR yarray, CgemvBatchedTilingData td)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (td.colSplits > 1u && td.trans == 0u) {
        CgemvRunColSplit2D(Aarray, xarray, yarray, td);
        return;
    }

    uint32_t startOut = GetBlockIdx() * td.batchPerCore;
    // issue #848: totalOutputs 乘法在 uint64 域计算，消除 uint32 回绕；
    // host 已限 batchCount*outSize <= INT32_MAX，uint32 切片算术（含
    // startOut+batchPerCore 的非末核上界推导）在该域内安全。
    const uint64_t totalOutputs = static_cast<uint64_t>(td.batchCount) * td.outSize;
    const uint32_t endOut =
        (GetBlockIdx() == td.usedCoreNum - 1u) ? static_cast<uint32_t>(totalOutputs) : (startOut + td.batchPerCore);
    if (startOut >= totalOutputs)
        return;

    // AIV paths require unit strides. trans=T/C uses the dot engine; trans=N
    // uses the column engine (column-major A columns are contiguous).
    // Fallback to SIMT for precision-sensitive dot-heavy / output-light cases
    // and for very large dot dimension where tiling overhead dominates.
    bool incxOk = (td.incx == 1) || (td.incx == 2 && td.trans == 2u);
    bool useAIV = (td.trans == 1u || td.trans == 2u || td.trans == 0u) && incxOk && td.incy == 1 &&
                  !(td.dotSize >= 1024u && td.outSize <= 16u) && td.dotSize <= CgemvBatchedAIVDot::AIV_DOT_DIM_LIMIT;

    if (useAIV) {
        uint32_t outSz = td.outSize;
        uint32_t startBatch = startOut / outSz;
        uint32_t endBatch = (endOut - 1u) / outSz + 1u;

        if (td.trans == 0u) {
            CgemvBatchedAIVNormal op;
            CgemvRunBatchRange(op, Aarray, xarray, yarray, td, startBatch, endBatch, startOut, endOut, outSz);
        } else {
            CgemvBatchedAIVDot op;
            CgemvRunBatchRange(op, Aarray, xarray, yarray, td, startBatch, endBatch, startOut, endOut, outSz);
        }
        return;
    }

    uint32_t numOuts = endOut - startOut;
    asc_vf_call<CgemvSimt>(
        dim3{CGEMV_SIMT_MAX_THREADS, 1, 1}, td.m, td.n, startOut, numOuts, td.alphaR, td.alphaI, td.betaR, td.betaI,
        td.alphaIsZero, td.betaIsZero, td.lda, td.incx, td.incy, td.trans, Aarray, xarray, yarray);
}

void cgemv_batched_kernel_do(
    uint8_t* Aarray, uint8_t* xarray, uint8_t* yarray, const CgemvBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cgemv_batched<<<numBlocks, nullptr, stream>>>(Aarray, xarray, yarray, tiling);
}
