/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "chemv_kernel.h"
#include "chemv_tiling_data.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t MAX_N = 4096U;
constexpr uint32_t VEC_FLOATS = 64U;
constexpr uint32_t BLK_FLOATS = 8U;
constexpr uint32_t REDUCE_CHUNK = 64U;  // 64 complex rows per cross-core reduction chunk
constexpr uint32_t GEN_CHUNK = 256U;
constexpr uint32_t PLANE_FLOATS = MAX_N;                  // 4096 floats (16.00 KiB)
constexpr uint32_t AOS_FLOATS = PLANE_FLOATS * 2U;        // 8192 floats (32.00 KiB)
constexpr uint32_t CHUNK_COMPLEX = 2176U;                 // 2176 complex per DMA chunk
constexpr uint32_t CHUNK_AOS_FLOATS = CHUNK_COMPLEX * 2U; // 4352 floats (17.00 KiB)
constexpr uint32_t MAX_COLS_PER_CORE = 128U;
constexpr uint32_t MAX_CHUNKS_PER_CORE = 160U;

__aicore__ inline uint32_t AlignUp(uint32_t val, uint32_t align)
{
    if (align == 0U) {
        return val;
    }
    return ((val + align - 1U) / align) * align;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline bool IsFiniteFloat(float v)
{
    union {
        float f;
        uint32_t u;
    } bits{v};
    return (bits.u & 0x7F800000U) != 0x7F800000U;
}

__aicore__ inline uint64_t PhysicalIndex(uint32_t logical, uint32_t n, int64_t inc, uint32_t absInc)
{
    return (inc >= 0) ? (static_cast<uint64_t>(logical) * absInc) :
                        (static_cast<uint64_t>(n - 1U - logical) * absInc);
}

__aicore__ inline void CopyBytesGmToUb(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t bytes)
{
    const uint32_t elems = bytes / sizeof(float);
    const uint8_t rightPad = static_cast<uint8_t>((BLK_FLOATS - (elems % BLK_FLOATS)) % BLK_FLOATS);
    DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{rightPad != 0U, 0, rightPad, 0.0f};
    DataCopyPad(dst, src, copyParams, padParams);
}

__aicore__ inline void CopyBytesUbToGm(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t bytes)
{
    DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
    DataCopyPad(dst, src, copyParams);
}

__aicore__ inline void SplitInterleavedToPlanes(
    LocalTensor<float> dstRe, LocalTensor<float> dstIm, LocalTensor<float> srcInterleaved, uint32_t complexCountAligned32)
{
    const uint16_t repeats = static_cast<uint16_t>((complexCountAligned32 * 2U + VEC_FLOATS - 1U) / VEC_FLOATS);
    uint64_t rsvdCnt = 0;
    GatherMask(dstRe, srcInterleaved, 1, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    GatherMask(dstIm, srcInterleaved, 2, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    ResetMask();
}

// ============================================================================
// Path 0: Scale-only kernel (alpha == (0, 0), beta != (1, 0))
// ============================================================================
__aicore__ inline void ScaleStridedRange(
    GlobalTensor<float>& yGM, LocalTensor<float>& ioBuf, LocalTensor<float>& outBuf, uint32_t rowLo, uint32_t rowHi,
    uint32_t n, int64_t incy, uint32_t absIncy, float br, float bi, bool betaZero)
{
    for (uint32_t i = rowLo; i < rowHi; ++i) {
        const uint64_t pos = PhysicalIndex(i, n, incy, absIncy) * 2ULL;
        if (betaZero) {
            outBuf.SetValue(0, 0.0f);
            outBuf.SetValue(1, 0.0f);
        } else {
            CopyBytesGmToUb(ioBuf, yGM[pos], 2U * sizeof(float));
            PipeBarrier<PIPE_ALL>();
            const float yr = ioBuf.GetValue(0);
            const float yi = ioBuf.GetValue(1);
            outBuf.SetValue(0, br * yr - bi * yi);
            outBuf.SetValue(1, br * yi + bi * yr);
        }
        PipeBarrier<PIPE_ALL>();
        CopyBytesUbToGm(yGM[pos], outBuf, 2U * sizeof(float));
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void ProcessScaleOnly(GM_ADDR y, const ChemvTilingData& tiling, TPipe& pipe)
{
    const uint32_t blkIdx = GetBlockIdx();
    const uint32_t numBlocks = tiling.useCoreNum;
    if (numBlocks == 0U || blkIdx >= numBlocks) {
        return;
    }
    const uint32_t n = tiling.n;
    const uint32_t rowLo = static_cast<uint32_t>((static_cast<uint64_t>(blkIdx) * n) / numBlocks);
    const uint32_t rowHi = static_cast<uint32_t>((static_cast<uint64_t>(blkIdx + 1U) * n) / numBlocks);
    if (rowLo >= rowHi) {
        return;
    }

    const int64_t incy = tiling.incy;
    const uint32_t absIncy = (incy >= 0) ? static_cast<uint32_t>(incy) : static_cast<uint32_t>(-incy);
    const float br = tiling.betaReal;
    const float bi = tiling.betaImag;
    const bool betaZero = (br == 0.0f && bi == 0.0f);

    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, GEN_CHUNK * 4U * sizeof(float));
    LocalTensor<float> ioBuf = buf.Get<float>();
    LocalTensor<float> outBuf = ioBuf[GEN_CHUNK * 2U];

    GlobalTensor<float> yGM;
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), (static_cast<uint64_t>(n - 1U) * absIncy + 1U) * 2ULL);

    if (incy == 1 && betaZero) {
        Duplicate(outBuf, 0.0f, GEN_CHUNK * 2U);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t r = rowLo; r < rowHi; r += GEN_CHUNK) {
            const uint32_t len = MinU32(GEN_CHUNK, rowHi - r);
            CopyBytesUbToGm(yGM[static_cast<uint64_t>(r) * 2ULL], outBuf, len * 2U * sizeof(float));
            PipeBarrier<PIPE_ALL>();
        }
        return;
    }
    ScaleStridedRange(yGM, ioBuf, outBuf, rowLo, rowHi, n, incy, absIncy, br, bi, betaZero);
}

struct ChemvColSchedule {
    uint32_t myCols[MAX_COLS_PER_CORE];
    float myT1r[MAX_COLS_PER_CORE];
    float myT1i[MAX_COLS_PER_CORE];
    uint16_t chColIdx[MAX_CHUNKS_PER_CORE];
    uint16_t chCOff[MAX_CHUNKS_PER_CORE];
    uint16_t chLen[MAX_CHUNKS_PER_CORE];
    uint16_t chCopyComplex[MAX_CHUNKS_PER_CORE];
    uint16_t chRBase4[MAX_CHUNKS_PER_CORE];
    uint32_t numCols = 0U;
    uint32_t numChunks = 0U;
};

// ============================================================================
// Path 1: Direct-AoS Ping-Pong Vector Kernel (16 < n <= 4096)
//   - Keeps matrix A in native interleaved (AoS) layout — ZERO per-column GatherMask!
//   - Pre-builds xAos = [xr, xi, ...] and xSwap = [xi, -xr, ...] once per core
//   - Accumulates into w0 (sum t1r * A) and w1 (sum t1i * A) via 2 Axpy calls per chunk
//   - Computes conjugated dot product via 2 Mul calls (slot * xAos, slot * xSwap) + ReduceSum
// ============================================================================
class ChemvColSplitOp {
public:
    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const ChemvTilingData& tiling, TPipe& pipe)
    {
        blkIdx_ = GetBlockIdx();
        n_ = tiling.n;
        lda_ = tiling.lda;
        numBlocks_ = tiling.useCoreNum;
        uplo_ = tiling.uplo;
        strideFloats_ = tiling.strideFloats;
        ar_ = tiling.alphaReal;
        ai_ = tiling.alphaImag;
        br_ = tiling.betaReal;
        bi_ = tiling.betaImag;

        aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2ULL);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(n_) * 2ULL);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint64_t>(n_) * 2ULL);
        wsGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workSpace),
            static_cast<uint64_t>(CHEMV_MAX_CORES) * 2ULL * CHEMV_COLSPLIT_MAX_N);

        // Total UB = 4*32.00 + 2*16.00 + 2*8.00 + 11.00 = 187.0 KiB <= 188.0 KiB (zero compiler UB spill!)
        pipe.InitBuffer(xAosBuf_, AOS_FLOATS * sizeof(float));
        pipe.InitBuffer(xSwapBuf_, AOS_FLOATS * sizeof(float));
        pipe.InitBuffer(w0Buf_, AOS_FLOATS * sizeof(float));
        pipe.InitBuffer(w1Buf_, AOS_FLOATS * sizeof(float));
        pipe.InitBuffer(aPingBuf_, CHUNK_AOS_FLOATS * sizeof(float));
        pipe.InitBuffer(aPongBuf_, CHUNK_AOS_FLOATS * sizeof(float));
        pipe.InitBuffer(t0Buf_, 2048U * sizeof(float));
        pipe.InitBuffer(t1Buf_, 2048U * sizeof(float));
        pipe.InitBuffer(miscBuf_, 2816U * sizeof(float));

        xAos_ = xAosBuf_.Get<float>();
        xSwap_ = xSwapBuf_.Get<float>();
        w0_ = w0Buf_.Get<float>();
        w1_ = w1Buf_.Get<float>();
        aPing_ = aPingBuf_.Get<float>();
        aPong_ = aPongBuf_.Get<float>();
        t0_ = t0Buf_.Get<float>();
        t1_ = t1Buf_.Get<float>();

        LocalTensor<float> misc = miscBuf_.Get<float>();
        partRe_ = misc[0];        // 4 * 256 = 1024 floats (batched first-stage vcgadd)
        partIm_ = misc[1024];     // 4 * 256 = 1024 floats (batched first-stage vcgadd)
        interleaveTbl_ = misc[2048].ReinterpretCast<uint32_t>(); // 128 uint32_t (64 complex)
        swapTbl_ = misc[2176].ReinterpretCast<uint32_t>();       // 128 uint32_t (64 complex pair-swap)
        dotReArr_ = misc[2304];   // 192 floats (contiguous 1-float-per-col second-stage WholeReduceSum)
        dotImArr_ = misc[2496];   // 192 floats
        workLocal_ = misc[2688];  // 128 floats
    }

    __aicore__ inline void Process()
    {
        if (numBlocks_ == 0U || blkIdx_ >= numBlocks_) {
            return;
        }
        RunPipeline();
        if (numBlocks_ == 1U) {
            StoreSingleCoreDirectly();
        } else {
            WriteWorkspaceAndSync();
            ReduceAndStoreY();
        }
    }

private:
    __aicore__ inline void StageChunkDma(uint32_t col, uint32_t rStart, uint32_t chunkComplex, LocalTensor<float> slot)
    {
        const uint64_t gmOff = (static_cast<uint64_t>(col) * lda_ + rStart) * 2ULL;
        const uint32_t floats = chunkComplex * 2U;
        if (((gmOff & (BLK_FLOATS - 1ULL)) == 0ULL) && ((floats & (BLK_FLOATS - 1U)) == 0U)) {
            DataCopy(slot, aGM_[gmOff], floats);
        } else {
            CopyBytesGmToUb(slot, aGM_[gmOff], floats * sizeof(float));
        }
    }

    __aicore__ inline void MaskChunkDiagonalAndTail(
        uint32_t col, uint32_t rBase32, uint32_t cOff, uint32_t chunkComplex, uint32_t alignedFloats, bool isFirstChunk,
        bool isLastChunk, LocalTensor<float> slot)
    {
        if (uplo_ == ACLBLAS_UPPER) {
            if (isLastChunk) {
                const uint32_t diagRel = col - cOff;
                const uint32_t blk64 = alignedFloats - VEC_FLOATS;
                const uint32_t laneRe = 2U * diagRel - blk64;
                const uint32_t laneIm = laneRe + 1U;
                uint64_t maskFromDiagIm[2] = {(~0ULL) << laneIm, 0ULL};
                uint64_t maskDiagRe[2] = {1ULL << laneRe, 0ULL};
                Duplicate<float>(slot[blk64], 0.0f, maskFromDiagIm, 1, 1, 8);
                Muls<float>(slot[blk64], slot[blk64], 0.5f, maskDiagRe, 1, {1, 1, 8, 8});
                ResetMask();
                PipeBarrier<PIPE_V>();
            }
            return;
        }
        if (isFirstChunk) {
            const uint32_t prefix = col - rBase32;
            uint64_t maskHead = ((1ULL << (2U * prefix)) - 1ULL) | (1ULL << (2U * prefix + 1U));
            const uint32_t tailBlk64 = alignedFloats - VEC_FLOATS;
            const uint32_t tailValidFloats = chunkComplex * 2U - tailBlk64;
            if (isLastChunk && tailValidFloats < VEC_FLOATS) {
                const uint64_t maskTail = (~0ULL) << tailValidFloats;
                if (tailBlk64 == 0U) {
                    maskHead |= maskTail;
                } else {
                    uint64_t maskTailArr[2] = {maskTail, 0ULL};
                    Duplicate<float>(slot[tailBlk64], 0.0f, maskTailArr, 1, 1, 8);
                }
            }
            uint64_t maskHeadArr[2] = {maskHead, 0ULL};
            uint64_t maskDiagRe[2] = {1ULL << (2U * prefix), 0ULL};
            Duplicate<float>(slot, 0.0f, maskHeadArr, 1, 1, 8);
            Muls<float>(slot, slot, 0.5f, maskDiagRe, 1, {1, 1, 8, 8});
            ResetMask();
            PipeBarrier<PIPE_V>();
        } else if (isLastChunk) {
            const uint32_t tailBlk64 = alignedFloats - VEC_FLOATS;
            const uint32_t tailValidFloats = chunkComplex * 2U - tailBlk64;
            if (tailValidFloats < VEC_FLOATS) {
                uint64_t maskTailArr[2] = {(~0ULL) << tailValidFloats, 0ULL};
                Duplicate<float>(slot[tailBlk64], 0.0f, maskTailArr, 1, 1, 8);
                ResetMask();
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void AccumulateDotChunkVmla(
        uint32_t rStart, uint32_t alignedFloats, uint32_t firstFloats, bool isFirstChunk, LocalTensor<float> slot)
    {
        if (isFirstChunk) {
            Mul(t0_, slot, xAos_[2U * rStart], firstFloats);
            Mul(t1_, slot, xSwap_[2U * rStart], firstFloats);
        } else {
#if defined(__DAV_C220_VEC__)
            const uint8_t firstReps = static_cast<uint8_t>(firstFloats / VEC_FLOATS);
            vmla(
                reinterpret_cast<__ubuf__ float*>(t0_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(slot.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(xAos_[2U * rStart].GetPhyAddr()), firstReps, 1, 1, 1, 8, 8, 8);
            vmla(
                reinterpret_cast<__ubuf__ float*>(t1_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(slot.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(xSwap_[2U * rStart].GetPhyAddr()), firstReps, 1, 1, 1, 8, 8, 8);
#endif
        }
        if (alignedFloats > 2048U) {
#if defined(__DAV_C220_VEC__)
            const uint32_t rem2048 = MinU32(2048U, alignedFloats - 2048U);
            const uint8_t remReps = static_cast<uint8_t>(rem2048 / VEC_FLOATS);
            vmla(
                reinterpret_cast<__ubuf__ float*>(t0_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(slot[2048].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(xAos_[2U * rStart + 2048U].GetPhyAddr()), remReps, 1, 1, 1, 8, 8, 8);
            vmla(
                reinterpret_cast<__ubuf__ float*>(t1_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(slot[2048].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(xSwap_[2U * rStart + 2048U].GetPhyAddr()), remReps, 1, 1, 1, 8, 8, 8);
            if (alignedFloats > 4096U) {
                const uint8_t remReps2 = static_cast<uint8_t>((alignedFloats - 4096U) / VEC_FLOATS);
                vmla(
                    reinterpret_cast<__ubuf__ float*>(t0_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(slot[4096].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(xAos_[2U * rStart + 4096U].GetPhyAddr()), remReps2, 1, 1, 1, 8,
                    8, 8);
                vmla(
                    reinterpret_cast<__ubuf__ float*>(t1_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(slot[4096].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(xSwap_[2U * rStart + 4096U].GetPhyAddr()), remReps2, 1, 1, 1, 8,
                    8, 8);
            }
#endif
        }
    }

    __aicore__ inline int32_t ConsumeChunkAoS(
        uint32_t col, uint32_t rBase32, uint32_t cOff, uint32_t chunkComplex, uint32_t copyComplex, uint32_t batchSlot,
        float t1r, float t1i, LocalTensor<float> slot)
    {
        const uint32_t rStart = rBase32 + cOff;
        const uint32_t alignedFloats = AlignUp(chunkComplex * 2U, VEC_FLOATS);
        const bool isFirstChunk = (cOff == 0U);
        const bool isLastChunk = (cOff + chunkComplex == copyComplex);

        MaskChunkDiagonalAndTail(col, rBase32, cOff, chunkComplex, alignedFloats, isFirstChunk, isLastChunk, slot);

        Axpy(w0_[2U * rStart], slot, t1r, alignedFloats);
        Axpy(w1_[2U * rStart], slot, t1i, alignedFloats);

        const uint32_t firstFloats = MinU32(2048U, alignedFloats);
        AccumulateDotChunkVmla(rStart, alignedFloats, firstFloats, isFirstChunk, slot);

        int32_t evFree = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(evFree);

        if (!isLastChunk) {
            return evFree;
        }
        PipeBarrier<PIPE_V>();

        uint32_t curFloats = isFirstChunk ? firstFloats : 2048U;
#if defined(__DAV_C220_VEC__)
        const int32_t reps1 = static_cast<int32_t>(curFloats / VEC_FLOATS);
        vcgadd(
            reinterpret_cast<__ubuf__ float*>(partRe_[batchSlot * 256U].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(t0_.GetPhyAddr()), reps1, 1, 1, 8);
        vcgadd(
            reinterpret_cast<__ubuf__ float*>(partIm_[batchSlot * 256U].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(t1_.GetPhyAddr()), reps1, 1, 1, 8);
#endif
        return evFree;
    }

    __aicore__ inline void BuildColumnAndChunkSchedule(ChemvColSchedule& sched) const
    {
        sched.numCols = 0U;
        sched.numChunks = 0U;
        const uint32_t halfCols = (n_ + 1U) / 2U;
        for (uint32_t pairIdx = blkIdx_; pairIdx < halfCols; pairIdx += numBlocks_) {
            sched.myCols[sched.numCols++] = pairIdx;
            const uint32_t mirrorCol = n_ - 1U - pairIdx;
            if (mirrorCol != pairIdx) {
                sched.myCols[sched.numCols++] = mirrorCol;
            }
        }
        for (uint32_t idx = 0U; idx < sched.numCols; ++idx) {
            const uint32_t col = sched.myCols[idx];
            const uint32_t rBase32 = (uplo_ == ACLBLAS_UPPER) ? 0U : ((col / 32U) * 32U);
            const uint32_t copyComplex = (uplo_ == ACLBLAS_UPPER) ? (col + 1U) : (n_ - rBase32);
            for (uint32_t cOff = 0U; cOff < copyComplex; cOff += CHUNK_COMPLEX) {
                sched.chColIdx[sched.numChunks] = static_cast<uint16_t>(idx);
                sched.chCOff[sched.numChunks] = static_cast<uint16_t>(cOff);
                sched.chLen[sched.numChunks] = static_cast<uint16_t>(MinU32(CHUNK_COMPLEX, copyComplex - cOff));
                sched.chCopyComplex[sched.numChunks] = static_cast<uint16_t>(copyComplex);
                sched.chRBase4[sched.numChunks] = static_cast<uint16_t>(rBase32);
                ++sched.numChunks;
            }
        }
    }

    __aicore__ inline void PrepareXSwapAndScalars(uint32_t activeAosFloats, ChemvColSchedule& sched)
    {
        const uint32_t tailStart = AlignUp(n_ * 2U, BLK_FLOATS);
        for (uint32_t i = n_ * 2U; i < tailStart; ++i) {
            xAos_.SetValue(i, 0.0f);
        }
        int32_t evS2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evS2V);
        WaitFlag<HardEvent::S_V>(evS2V);
        if (tailStart < activeAosFloats) {
            Duplicate(xAos_[tailStart], 0.0f, activeAosFloats - tailStart);
            Duplicate(xSwap_[tailStart], 0.0f, activeAosFloats - tailStart);
            PipeBarrier<PIPE_V>();
        }

        for (uint32_t r0 = 0U; r0 < strideFloats_; r0 += REDUCE_CHUNK) {
            Gather(xSwap_[r0 * 2U], xAos_[r0 * 2U], swapTbl_, 0U, REDUCE_CHUNK * 2U);
        }
        const uint32_t checkLen0 = MinU32(2048U, activeAosFloats);
        Muls(t0_, xAos_, 0.0f, checkLen0);
        PipeBarrier<PIPE_V>();

        uint64_t oddLaneMask[2] = {0xAAAAAAAAAAAAAAAAULL, 0ULL};
        const uint32_t totalSwapRepeats = activeAosFloats / VEC_FLOATS;
        Muls<float>(xSwap_, xSwap_, -1.0f, oddLaneMask, static_cast<uint8_t>(totalSwapRepeats), {1, 1, 8, 8});
        ResetMask();

        for (uint32_t off = 2048U; off < activeAosFloats; off += 2048U) {
            const uint32_t rem = MinU32(2048U, activeAosFloats - off);
            Axpy(t0_, xAos_[off], 0.0f, rem);
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float>(t1_, t0_, VEC_FLOATS, static_cast<uint8_t>(checkLen0 / VEC_FLOATS), 1, 1, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float>(dotReArr_, t1_, checkLen0 / VEC_FLOATS, 1, 1, 1, 8);
        ResetMask();
        int32_t evFinV2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evFinV2S);

        const __ubuf__ float* xAosPtr = reinterpret_cast<const __ubuf__ float*>(xAos_.GetPhyAddr());
        for (uint32_t idx = 0U; idx < sched.numCols; ++idx) {
            const uint32_t col = sched.myCols[idx];
            const float xjr = xAosPtr[2U * col];
            const float xji = xAosPtr[2U * col + 1U];
            sched.myT1r[idx] = ar_ * xjr - ai_ * xji;
            sched.myT1i[idx] = ar_ * xji + ai_ * xjr;
        }
        WaitFlag<HardEvent::V_S>(evFinV2S);
        xAllFinite_ = IsFiniteFloat(dotReArr_.GetValue(0));
    }

    __aicore__ inline void FlushDotBatch(
        uint32_t cIdx, uint32_t numCols, uint32_t& batchBase, uint32_t& batchSlot)
    {
        constexpr uint32_t BATCH_COLS = 4U;
        if (batchSlot + 1U != BATCH_COLS && cIdx + 1U != numCols) {
            ++batchSlot;
            return;
        }
        PipeBarrier<PIPE_V>();
        const uint8_t batchCount = static_cast<uint8_t>(batchSlot + 1U);
#if defined(__DAV_C220_VEC__)
        vcgadd(
            reinterpret_cast<__ubuf__ float*>(t0_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(partRe_.GetPhyAddr()), static_cast<int32_t>(batchCount * 4U), 1, 1, 8);
        vcgadd(
            reinterpret_cast<__ubuf__ float*>(t1_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(partIm_.GetPhyAddr()), static_cast<int32_t>(batchCount * 4U), 1, 1, 8);
        PipeBarrier<PIPE_V>();
#endif
        WholeReduceSum<float>(dotReArr_[batchBase], t0_, 32, batchCount, 1, 1, 4);
        WholeReduceSum<float>(dotImArr_[batchBase], t1_, 32, batchCount, 1, 1, 4);
        ResetMask();
        if (cIdx + 1U < numCols) {
            Duplicate(partRe_, 0.0f, BATCH_COLS * 256U);
            Duplicate(partIm_, 0.0f, BATCH_COLS * 256U);
        }
        PipeBarrier<PIPE_V>();
        batchBase = cIdx + 1U;
        batchSlot = 0U;
    }

    __aicore__ inline void StreamScheduledChunks(const ChemvColSchedule& sched, int32_t evCurM2V)
    {
        int32_t evSlotFree[2] = {-1, -1};
        uint32_t batchSlot = 0U;
        uint32_t batchBase = 0U;
        for (uint32_t ch = 0U; ch < sched.numChunks; ++ch) {
            const uint32_t curBufIdx = ch & 1U;
            LocalTensor<float> curSlot = (curBufIdx == 0U) ? aPing_ : aPong_;
            int32_t evNextM2V = -1;

            if (ch + 1U < sched.numChunks) {
                const uint32_t nextBufIdx = curBufIdx ^ 1U;
                if (evSlotFree[nextBufIdx] != -1) {
                    WaitFlag<HardEvent::V_MTE2>(evSlotFree[nextBufIdx]);
                    evSlotFree[nextBufIdx] = -1;
                }
                LocalTensor<float> nextSlot = (nextBufIdx == 0U) ? aPing_ : aPong_;
                const uint32_t nextCol = sched.myCols[sched.chColIdx[ch + 1U]];
                StageChunkDma(nextCol, sched.chRBase4[ch + 1U] + sched.chCOff[ch + 1U], sched.chLen[ch + 1U], nextSlot);
                evNextM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(evNextM2V);
            }

            WaitFlag<HardEvent::MTE2_V>(evCurM2V);
            const uint32_t cIdx = sched.chColIdx[ch];
            const bool isLastChunk = (sched.chCOff[ch] + sched.chLen[ch] == sched.chCopyComplex[ch]);
            evSlotFree[curBufIdx] = ConsumeChunkAoS(
                sched.myCols[cIdx], sched.chRBase4[ch], sched.chCOff[ch], sched.chLen[ch], sched.chCopyComplex[ch],
                batchSlot, sched.myT1r[cIdx], sched.myT1i[cIdx], curSlot);

            if (isLastChunk) {
                FlushDotBatch(cIdx, sched.numCols, batchBase, batchSlot);
            }
            if (ch + 1U < sched.numChunks) {
                evCurM2V = evNextM2V;
            }
        }
        for (uint32_t b = 0U; b < 2U; ++b) {
            if (evSlotFree[b] != -1) {
                WaitFlag<HardEvent::V_MTE2>(evSlotFree[b]);
            }
        }
    }

    __aicore__ inline void FinalizeLocalAccumulators(const ChemvColSchedule& sched)
    {
        for (uint32_t pos = 0U; pos < strideFloats_; pos += CHUNK_COMPLEX) {
            const uint32_t planeLen = MinU32(CHUNK_COMPLEX, strideFloats_ - pos);
            SplitInterleavedToPlanes(aPing_[0], aPing_[CHUNK_COMPLEX], w0_[pos * 2U], planeLen);
            SplitInterleavedToPlanes(aPong_[0], aPong_[CHUNK_COMPLEX], w1_[pos * 2U], planeLen);
            PipeBarrier<PIPE_V>();
            Sub(w0_[pos], aPing_[0], aPong_[CHUNK_COMPLEX], planeLen);
            Add(w1_[pos], aPing_[CHUNK_COMPLEX], aPong_[0], planeLen);
            PipeBarrier<PIPE_V>();
        }

        const uint32_t dotSpan = AlignUp(sched.numCols, VEC_FLOATS);
        Muls(t0_[0], dotReArr_, ar_, dotSpan);
        Muls(t1_[0], dotImArr_, ai_, dotSpan);
        Muls(t0_[VEC_FLOATS * 2U], dotReArr_, ai_, dotSpan);
        Muls(t1_[VEC_FLOATS * 2U], dotImArr_, ar_, dotSpan);
        PipeBarrier<PIPE_V>();
        Sub(t0_[0], t0_[0], t1_[0], dotSpan);
        Add(t1_[0], t0_[VEC_FLOATS * 2U], t1_[VEC_FLOATS * 2U], dotSpan);

        int32_t evV2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evV2S);
        WaitFlag<HardEvent::V_S>(evV2S);

        const __ubuf__ float* dRePtr = reinterpret_cast<const __ubuf__ float*>(t0_.GetPhyAddr());
        const __ubuf__ float* dImPtr = reinterpret_cast<const __ubuf__ float*>(t1_.GetPhyAddr());
        __ubuf__ float* w0Ptr = reinterpret_cast<__ubuf__ float*>(w0_.GetPhyAddr());
        __ubuf__ float* w1Ptr = reinterpret_cast<__ubuf__ float*>(w1_.GetPhyAddr());
        for (uint32_t idx = 0U; idx < sched.numCols; ++idx) {
            const uint32_t col = sched.myCols[idx];
            w0Ptr[col] += dRePtr[idx];
            w1Ptr[col] += dImPtr[idx];
        }

        int32_t evDotS2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evDotS2V);
        WaitFlag<HardEvent::S_V>(evDotS2V);
    }

    __aicore__ inline void RunPipeline()
    {
        const uint32_t activeAosFloats = strideFloats_ * 2U;
        CopyBytesGmToUb(xAos_, xGM_, n_ * 2U * sizeof(float));
        int32_t evXM2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(evXM2S);
        int32_t evXM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evXM2V);

        constexpr uint32_t BATCH_COLS = 4U;
        Duplicate(w0_, 0.0f, activeAosFloats);
        Duplicate(w1_, 0.0f, activeAosFloats);
        Duplicate(partRe_, 0.0f, BATCH_COLS * 256U);
        Duplicate(partIm_, 0.0f, BATCH_COLS * 256U);

        for (uint32_t k = 0U; k < REDUCE_CHUNK; ++k) {
            interleaveTbl_.SetValue(2U * k, k * sizeof(float));
            interleaveTbl_.SetValue(2U * k + 1U, (REDUCE_CHUNK + k) * sizeof(float));
            swapTbl_.SetValue(2U * k, (2U * k + 1U) * sizeof(float));
            swapTbl_.SetValue(2U * k + 1U, (2U * k) * sizeof(float));
        }

        ChemvColSchedule sched;
        BuildColumnAndChunkSchedule(sched);

        int32_t evCurM2V = -1;
        if (sched.numChunks > 0U) {
            StageChunkDma(sched.myCols[sched.chColIdx[0]], sched.chRBase4[0] + sched.chCOff[0], sched.chLen[0], aPing_);
            evCurM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evCurM2V);
        }

        WaitFlag<HardEvent::MTE2_S>(evXM2S);
        WaitFlag<HardEvent::MTE2_V>(evXM2V);
        PrepareXSwapAndScalars(activeAosFloats, sched);

        if (!xAllFinite_) {
            if (evCurM2V != -1) {
                WaitFlag<HardEvent::MTE2_V>(evCurM2V);
            }
            Duplicate(w0_, 0.0f, activeAosFloats);
            Duplicate(w1_, 0.0f, activeAosFloats);
            PipeBarrier<PIPE_ALL>();
            RunNonFiniteXFallback();
            return;
        }
        if (sched.numCols == 0U) {
            return;
        }
        StreamScheduledChunks(sched, evCurM2V);
        FinalizeLocalAccumulators(sched);
    }

    __aicore__ inline void RunNonFiniteColumn(uint32_t col)
    {
        const uint64_t colBase = static_cast<uint64_t>(col) * lda_ * 2ULL;
        CopyBytesGmToUb(dotReArr_, aGM_[colBase + static_cast<uint64_t>(col) * 2ULL], sizeof(float));
        const uint32_t rowStart = (uplo_ == ACLBLAS_UPPER) ? 0U : (col + 1U);
        const uint32_t fullLen = (uplo_ == ACLBLAS_UPPER) ? col : (n_ - 1U - col);
        PipeBarrier<PIPE_ALL>();

        const float xjr = xAos_.GetValue(2U * col);
        const float xji = xAos_.GetValue(2U * col + 1U);
        const float t1r = ar_ * xjr - ai_ * xji;
        const float t1i = ar_ * xji + ai_ * xjr;
        const float dj = dotReArr_.GetValue(0);
        float t2r = 0.0f;
        float t2i = 0.0f;

        for (uint32_t off = 0U; off < fullLen; off += CHUNK_COMPLEX) {
            const uint32_t segLen = MinU32(CHUNK_COMPLEX, fullLen - off);
            CopyBytesGmToUb(
                aPing_, aGM_[colBase + static_cast<uint64_t>(rowStart + off) * 2ULL], segLen * 2U * sizeof(float));
            PipeBarrier<PIPE_ALL>();
            for (uint32_t k = 0U; k < segLen; ++k) {
                const uint32_t r = rowStart + off + k;
                const float aijR = aPing_.GetValue(2U * k);
                const float aijI = aPing_.GetValue(2U * k + 1U);
                const float xir = xAos_.GetValue(2U * r);
                const float xii = xAos_.GetValue(2U * r + 1U);
                w0_.SetValue(r, w0_.GetValue(r) + (t1r * aijR - t1i * aijI));
                w1_.SetValue(r, w1_.GetValue(r) + (t1r * aijI + t1i * aijR));
                t2r += aijR * xir + aijI * xii;
                t2i += aijR * xii - aijI * xir;
            }
            PipeBarrier<PIPE_ALL>();
        }
        w0_.SetValue(col, w0_.GetValue(col) + t1r * dj + (ar_ * t2r - ai_ * t2i));
        w1_.SetValue(col, w1_.GetValue(col) + t1i * dj + (ar_ * t2i + ai_ * t2r));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RunNonFiniteXFallback()
    {
        const uint32_t halfCols = (n_ + 1U) / 2U;
        for (uint32_t pairIdx = blkIdx_; pairIdx < halfCols; pairIdx += numBlocks_) {
            RunNonFiniteColumn(pairIdx);
            const uint32_t mirrorCol = n_ - 1U - pairIdx;
            if (mirrorCol != pairIdx) {
                RunNonFiniteColumn(mirrorCol);
            }
        }
    }

    __aicore__ inline void ApplyBetaSingleCore()
    {
        const bool betaOne = (br_ == 1.0f && bi_ == 0.0f);
        Duplicate(aPing_, 0.0f, strideFloats_ * 2U);
        int32_t evV2M = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(evV2M);
        WaitFlag<HardEvent::V_MTE2>(evV2M);
        CopyBytesGmToUb(aPing_, yGM_, n_ * 2U * sizeof(float));
        int32_t evM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evM2V);
        WaitFlag<HardEvent::MTE2_V>(evM2V);

        SplitInterleavedToPlanes(aPong_[0], aPong_[CHUNK_COMPLEX], aPing_, strideFloats_);
        PipeBarrier<PIPE_V>();
        if (betaOne) {
            Add(w0_, w0_, aPong_[0], strideFloats_);
            Add(w1_, w1_, aPong_[CHUNK_COMPLEX], strideFloats_);
        } else {
            LocalTensor<float> s0 = t0_[0];
            LocalTensor<float> s1 = t0_[CHUNK_COMPLEX];
            Muls(s0, aPong_[0], br_, strideFloats_);
            Muls(s1, aPong_[CHUNK_COMPLEX], bi_, strideFloats_);
            PipeBarrier<PIPE_V>();
            Sub(s0, s0, s1, strideFloats_);
            Add(w0_, w0_, s0, strideFloats_);

            Muls(s0, aPong_[0], bi_, strideFloats_);
            Muls(s1, aPong_[CHUNK_COMPLEX], br_, strideFloats_);
            PipeBarrier<PIPE_V>();
            Add(s0, s0, s1, strideFloats_);
            Add(w1_, w1_, s0, strideFloats_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreSingleCoreDirectly()
    {
        if (br_ != 0.0f || bi_ != 0.0f) {
            ApplyBetaSingleCore();
        }
        for (uint32_t r0 = 0U; r0 < n_; r0 += REDUCE_CHUNK) {
            const uint32_t validLen = MinU32(REDUCE_CHUNK, n_ - r0);
            Adds(t0_[0], w0_[r0], 0.0f, REDUCE_CHUNK);
            Adds(t0_[REDUCE_CHUNK], w1_[r0], 0.0f, REDUCE_CHUNK);
            PipeBarrier<PIPE_V>();
            Gather(aPong_, t0_, interleaveTbl_, 0U, REDUCE_CHUNK * 2U);
            int32_t evV2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(evV2M3);
            WaitFlag<HardEvent::V_MTE3>(evV2M3);
            CopyBytesUbToGm(yGM_[static_cast<uint64_t>(r0) * 2ULL], aPong_, validLen * 2U * sizeof(float));
            int32_t evM3V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(evM3V);
            WaitFlag<HardEvent::MTE3_V>(evM3V);
        }
    }

    __aicore__ inline void WriteWorkspaceAndSync()
    {
        int32_t evV2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evV2M3);
        WaitFlag<HardEvent::V_MTE3>(evV2M3);

        const uint64_t wsReOff = static_cast<uint64_t>(2U * blkIdx_) * strideFloats_;
        const uint64_t wsImOff = static_cast<uint64_t>(2U * blkIdx_ + 1U) * strideFloats_;
        DataCopy(wsGM_[wsReOff], w0_, strideFloats_);
        DataCopy(wsGM_[wsImOff], w1_, strideFloats_);

        int32_t evM3S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(evM3S);
        WaitFlag<HardEvent::MTE3_S>(evM3S);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
    }

    __aicore__ inline void ApplyBetaToReducedChunk(uint32_t r0, uint32_t validLen, bool betaOne)
    {
        Duplicate(t0_, 0.0f, REDUCE_CHUNK * 2U);
        int32_t evV2M = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(evV2M);
        WaitFlag<HardEvent::V_MTE2>(evV2M);
        CopyBytesGmToUb(t0_, yGM_[static_cast<uint64_t>(r0) * 2ULL], validLen * 2U * sizeof(float));
        int32_t evYM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evYM2V);
        WaitFlag<HardEvent::MTE2_V>(evYM2V);

        LocalTensor<float> yOldRe = w0_[0];
        LocalTensor<float> yOldIm = w0_[REDUCE_CHUNK];
        SplitInterleavedToPlanes(yOldRe, yOldIm, t0_, REDUCE_CHUNK);
        PipeBarrier<PIPE_V>();
        if (betaOne) {
            Add(aPing_, aPing_, yOldRe, REDUCE_CHUNK);
            Add(aPong_, aPong_, yOldIm, REDUCE_CHUNK);
        } else {
            LocalTensor<float> scRe = w0_[REDUCE_CHUNK * 2U];
            LocalTensor<float> scIm = w0_[REDUCE_CHUNK * 3U];
            LocalTensor<float> scTmp = w0_[REDUCE_CHUNK * 4U];
            Muls(scRe, yOldRe, br_, REDUCE_CHUNK);
            Muls(scTmp, yOldIm, bi_, REDUCE_CHUNK);
            PipeBarrier<PIPE_V>();
            Sub(scRe, scRe, scTmp, REDUCE_CHUNK);

            Muls(scIm, yOldRe, bi_, REDUCE_CHUNK);
            Muls(scTmp, yOldIm, br_, REDUCE_CHUNK);
            PipeBarrier<PIPE_V>();
            Add(scIm, scIm, scTmp, REDUCE_CHUNK);
            PipeBarrier<PIPE_V>();

            Add(aPing_, aPing_, scRe, REDUCE_CHUNK);
            Add(aPong_, aPong_, scIm, REDUCE_CHUNK);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreReducedChunkToGm(uint32_t r0, uint32_t validLen)
    {
        LocalTensor<float> packSrc = w0_[0];
        Adds(packSrc[0], aPing_, 0.0f, REDUCE_CHUNK);
        Adds(packSrc[REDUCE_CHUNK], aPong_, 0.0f, REDUCE_CHUNK);
        PipeBarrier<PIPE_V>();
        Gather(t0_, packSrc, interleaveTbl_, 0U, REDUCE_CHUNK * 2U);

        int32_t evV2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evV2M3);
        WaitFlag<HardEvent::V_MTE3>(evV2M3);
        CopyBytesUbToGm(yGM_[static_cast<uint64_t>(r0) * 2ULL], t0_, validLen * 2U * sizeof(float));
        int32_t evM3V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(evM3V);
        WaitFlag<HardEvent::MTE3_V>(evM3V);
        int32_t evV2M2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(evV2M2);
        WaitFlag<HardEvent::V_MTE2>(evV2M2);
    }

    __aicore__ inline void ReduceAndStoreY()
    {
        if (numBlocks_ == 0U) {
            return;
        }
        const uint32_t nChunks = (n_ + REDUCE_CHUNK - 1U) / REDUCE_CHUNK;
        const uint32_t chLo = static_cast<uint32_t>((static_cast<uint64_t>(blkIdx_) * nChunks) / numBlocks_);
        const uint32_t chHi = static_cast<uint32_t>((static_cast<uint64_t>(blkIdx_ + 1U) * nChunks) / numBlocks_);
        if (chLo >= chHi) {
            return;
        }
        uint32_t foldW = 1U;
        while (foldW < numBlocks_) {
            foldW <<= 1U;
        }
        const uint16_t blockLenBlocks = static_cast<uint16_t>(REDUCE_CHUNK / BLK_FLOATS);
        const uint16_t srcStrideBlocks = static_cast<uint16_t>((2U * strideFloats_ - REDUCE_CHUNK) / BLK_FLOATS);
        const DataCopyParams wsCopy{static_cast<uint16_t>(numBlocks_), blockLenBlocks, srcStrideBlocks, 0};
        const bool betaZero = (br_ == 0.0f && bi_ == 0.0f);
        const bool betaOne = (br_ == 1.0f && bi_ == 0.0f);

        for (uint32_t ch = chLo; ch < chHi; ++ch) {
            const uint32_t r0 = ch * REDUCE_CHUNK;
            const uint32_t validLen = MinU32(REDUCE_CHUNK, n_ - r0);
            DataCopy(aPing_, wsGM_[r0], wsCopy);
            DataCopy(aPong_, wsGM_[strideFloats_ + r0], wsCopy);
            int32_t evM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);

            if (foldW > numBlocks_) {
                Duplicate(aPing_[numBlocks_ * REDUCE_CHUNK], 0.0f, (foldW - numBlocks_) * REDUCE_CHUNK);
                Duplicate(aPong_[numBlocks_ * REDUCE_CHUNK], 0.0f, (foldW - numBlocks_) * REDUCE_CHUNK);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t half = foldW >> 1U; half >= 1U; half >>= 1U) {
                Add(aPing_, aPing_, aPing_[half * REDUCE_CHUNK], half * REDUCE_CHUNK);
                Add(aPong_, aPong_, aPong_[half * REDUCE_CHUNK], half * REDUCE_CHUNK);
                PipeBarrier<PIPE_V>();
            }
            if (!betaZero) {
                ApplyBetaToReducedChunk(r0, validLen, betaOne);
            }
            StoreReducedChunkToGm(r0, validLen);
        }
    }

    GlobalTensor<float> aGM_;
    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    GlobalTensor<float> wsGM_;

    TBuf<TPosition::VECCALC> xAosBuf_;
    TBuf<TPosition::VECCALC> xSwapBuf_;
    TBuf<TPosition::VECCALC> w0Buf_;
    TBuf<TPosition::VECCALC> w1Buf_;
    TBuf<TPosition::VECCALC> aPingBuf_;
    TBuf<TPosition::VECCALC> aPongBuf_;
    TBuf<TPosition::VECCALC> t0Buf_;
    TBuf<TPosition::VECCALC> t1Buf_;
    TBuf<TPosition::VECCALC> miscBuf_;

    LocalTensor<float> xAos_;
    LocalTensor<float> xSwap_;
    LocalTensor<float> w0_;
    LocalTensor<float> w1_;
    LocalTensor<float> aPing_;
    LocalTensor<float> aPong_;
    LocalTensor<float> t0_;
    LocalTensor<float> t1_;
    LocalTensor<float> partRe_;
    LocalTensor<float> partIm_;
    LocalTensor<float> workLocal_;
    LocalTensor<float> dotReArr_;
    LocalTensor<float> dotImArr_;
    LocalTensor<uint32_t> interleaveTbl_;
    LocalTensor<uint32_t> swapTbl_;

    uint32_t blkIdx_ = 0U;
    uint32_t n_ = 0U;
    uint32_t lda_ = 0U;
    uint32_t numBlocks_ = 1U;
    uint32_t uplo_ = ACLBLAS_UPPER;
    uint32_t strideFloats_ = 64U;
    float ar_ = 1.0f;
    float ai_ = 0.0f;
    float br_ = 0.0f;
    float bi_ = 0.0f;
    bool xAllFinite_ = true;
};

// ============================================================================
// Path 2: Lightweight general kernel (no SyncAll, zero workspace)
//   - Single-DMA fast path for n <= 16 (incx=incy=1)
//   - Exact Netlib row-owner path for strided, Inf/NaN scalars, and n > 4096
// ============================================================================
class ChemvGeneralOp {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const ChemvTilingData& tiling, TPipe& pipe)
    {
        blkIdx_ = GetBlockIdx();
        n_ = tiling.n;
        lda_ = tiling.lda;
        numBlocks_ = tiling.useCoreNum;
        uplo_ = tiling.uplo;
        ar_ = tiling.alphaReal;
        ai_ = tiling.alphaImag;
        br_ = tiling.betaReal;
        bi_ = tiling.betaImag;
        incx_ = tiling.incx;
        incy_ = tiling.incy;
        absIncx_ = (incx_ >= 0) ? static_cast<uint32_t>(incx_) : static_cast<uint32_t>(-incx_);
        absIncy_ = (incy_ >= 0) ? static_cast<uint32_t>(incy_) : static_cast<uint32_t>(-incy_);

        aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2ULL);
        xGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(x), (static_cast<uint64_t>(n_ - 1U) * absIncx_ + 1U) * 2ULL);
        yGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(y), (static_cast<uint64_t>(n_ - 1U) * absIncy_ + 1U) * 2ULL);

        pipe.InitBuffer(xChunkBuf_, GEN_CHUNK * BLK_FLOATS * sizeof(float));
        pipe.InitBuffer(aChunkBuf_, GEN_CHUNK * BLK_FLOATS * sizeof(float));
        pipe.InitBuffer(ioBuf_, 64U * sizeof(float));

        xChunk_ = xChunkBuf_.Get<float>();
        aChunk_ = aChunkBuf_.Get<float>();
        io_ = ioBuf_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if (numBlocks_ == 0U || blkIdx_ >= numBlocks_) {
            return;
        }
        if (numBlocks_ == 1U && incx_ == 1 && incy_ == 1 && n_ <= 16U) {
            ProcessTinySingleDma();
            return;
        }
        for (uint32_t row = blkIdx_; row < n_; row += numBlocks_) {
            ProcessSingleRow(row);
        }
    }

private:
    __aicore__ inline void LoadTinyInputs(uint32_t colStride, float (&xr)[16], float (&xi)[16], float (&yr)[16], float (&yi)[16])
    {
        if (lda_ <= 16U) {
            CopyBytesGmToUb(aChunk_, aGM_[0], lda_ * n_ * 2U * sizeof(float));
        } else {
            const uint32_t blockBytes = n_ * 2U * sizeof(float);
            const uint32_t alignedBytes = AlignUp(blockBytes, 32U);
            const uint8_t rightPad = static_cast<uint8_t>((alignedBytes - blockBytes) / sizeof(float));
            const uint32_t srcGapBytes = (lda_ - n_) * 2U * sizeof(float);
            const uint32_t dstGapBlocks = (colStride * sizeof(float) - alignedBytes) / 32U;

            DataCopyExtParams cp{static_cast<uint16_t>(n_), blockBytes, srcGapBytes, dstGapBlocks, 0};
            DataCopyPadExtParams<float> pp{rightPad != 0U, 0, rightPad, 0.0f};
            DataCopyPad(aChunk_, aGM_[0], cp, pp);
        }

        LocalTensor<float> xLocal = xChunk_[0];
        LocalTensor<float> yLocal = xChunk_[64];
        CopyBytesGmToUb(xLocal, xGM_, n_ * 2U * sizeof(float));
        const bool betaZero = (br_ == 0.0f && bi_ == 0.0f);
        const bool betaOne = (br_ == 1.0f && bi_ == 0.0f);
        if (!betaZero) {
            CopyBytesGmToUb(yLocal, yGM_, n_ * 2U * sizeof(float));
        }
        int32_t evM2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(evM2S);
        WaitFlag<HardEvent::MTE2_S>(evM2S);

        for (uint32_t i = 0U; i < n_; ++i) {
            xr[i] = xLocal.GetValue(2U * i);
            xi[i] = xLocal.GetValue(2U * i + 1U);
            if (betaZero) {
                yr[i] = 0.0f;
                yi[i] = 0.0f;
            } else {
                const float orVal = yLocal.GetValue(2U * i);
                const float oiVal = yLocal.GetValue(2U * i + 1U);
                yr[i] = betaOne ? orVal : (br_ * orVal - bi_ * oiVal);
                yi[i] = betaOne ? oiVal : (br_ * oiVal + bi_ * orVal);
            }
        }
    }

    __aicore__ inline void ProcessTinySingleDma()
    {
        const uint32_t colStride = (lda_ <= 16U) ? (lda_ * 2U) : 32U;
        float xr[16];
        float xi[16];
        float yr[16];
        float yi[16];
        LoadTinyInputs(colStride, xr, xi, yr, yi);

        for (uint32_t j = 0U; j < n_; ++j) {
            const uint32_t base = j * colStride;
            const float t1r = ar_ * xr[j] - ai_ * xi[j];
            const float t1i = ar_ * xi[j] + ai_ * xr[j];
            const float dj = aChunk_.GetValue(base + 2U * j);
            float t2r = 0.0f;
            float t2i = 0.0f;
            const uint32_t rLo = (uplo_ == ACLBLAS_UPPER) ? 0U : (j + 1U);
            const uint32_t rHi = (uplo_ == ACLBLAS_UPPER) ? j : n_;
            const __ubuf__ float* colPtr = reinterpret_cast<const __ubuf__ float*>(aChunk_.GetPhyAddr()) + base;
            for (uint32_t r = rLo; r < rHi; ++r) {
                const float aijR = colPtr[2U * r];
                const float aijI = colPtr[2U * r + 1U];
                yr[r] += t1r * aijR - t1i * aijI;
                yi[r] += t1r * aijI + t1i * aijR;
                t2r += aijR * xr[r] + aijI * xi[r];
                t2i += aijR * xi[r] - aijI * xr[r];
            }
            yr[j] += t1r * dj + (ar_ * t2r - ai_ * t2i);
            yi[j] += t1i * dj + (ar_ * t2i + ai_ * t2r);
        }

        for (uint32_t i = 0U; i < n_; ++i) {
            io_.SetValue(2U * i, yr[i]);
            io_.SetValue(2U * i + 1U, yi[i]);
        }
        int32_t evS2M3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evS2M3);
        WaitFlag<HardEvent::S_MTE3>(evS2M3);
        CopyBytesUbToGm(yGM_, io_, n_ * 2U * sizeof(float));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StageXSubrange(uint32_t colLo, uint32_t count)
    {
        if (incx_ == 1) {
            CopyBytesGmToUb(xChunk_, xGM_[static_cast<uint64_t>(colLo) * 2ULL], count * 2U * sizeof(float));
        } else if (incx_ > 0) {
            const uint32_t srcGapBytes = (absIncx_ - 1U) * 2U * sizeof(float);
            DataCopyExtParams copyParams{static_cast<uint16_t>(count), 2U * sizeof(float), srcGapBytes, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 6, 0.0f};
            DataCopyPad(xChunk_, xGM_[static_cast<uint64_t>(colLo) * absIncx_ * 2ULL], copyParams, padParams);
        } else {
            const uint64_t firstLog = colLo + count - 1U;
            const uint64_t physStart = static_cast<uint64_t>(n_ - 1U - firstLog) * absIncx_ * 2ULL;
            const uint32_t srcGapBytes = (absIncx_ - 1U) * 2U * sizeof(float);
            DataCopyExtParams copyParams{static_cast<uint16_t>(count), 2U * sizeof(float), srcGapBytes, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 6, 0.0f};
            DataCopyPad(xChunk_, xGM_[physStart], copyParams, padParams);
        }
    }

    __aicore__ inline void ReadStagedX(uint32_t idxInChunk, uint32_t count, float& xr, float& xi) const
    {
        if (incx_ == 1) {
            xr = xChunk_.GetValue(2U * idxInChunk);
            xi = xChunk_.GetValue(2U * idxInChunk + 1U);
        } else if (incx_ > 0) {
            xr = xChunk_.GetValue(idxInChunk * BLK_FLOATS);
            xi = xChunk_.GetValue(idxInChunk * BLK_FLOATS + 1U);
        } else {
            const uint32_t rev = count - 1U - idxInChunk;
            xr = xChunk_.GetValue(rev * BLK_FLOATS);
            xi = xChunk_.GetValue(rev * BLK_FLOATS + 1U);
        }
    }

    __aicore__ inline void AccumulateMirroredSegment(uint32_t row, uint32_t segLo, uint32_t segHi, float& sumR, float& sumI)
    {
        const uint64_t colBase = static_cast<uint64_t>(row) * lda_ * 2ULL;
        for (uint32_t c = segLo; c < segHi; c += GEN_CHUNK) {
            const uint32_t cnt = MinU32(GEN_CHUNK, segHi - c);
            StageXSubrange(c, cnt);
            CopyBytesGmToUb(aChunk_, aGM_[colBase + static_cast<uint64_t>(c) * 2ULL], cnt * 2U * sizeof(float));
            PipeBarrier<PIPE_ALL>();
            for (uint32_t k = 0U; k < cnt; ++k) {
                float xr = 0.0f;
                float xi = 0.0f;
                ReadStagedX(k, cnt, xr, xi);
                const float arVal = aChunk_.GetValue(2U * k);
                const float aiVal = -aChunk_.GetValue(2U * k + 1U);
                sumR += arVal * xr - aiVal * xi;
                sumI += arVal * xi + aiVal * xr;
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void AccumulateDirectRowSegment(
        uint32_t row, uint32_t segLo, uint32_t segHi, float& sumR, float& sumI)
    {
        const uint32_t srcGapBytes = (lda_ - 1U) * 2U * sizeof(float);
        for (uint32_t c = segLo; c < segHi; c += GEN_CHUNK) {
            const uint32_t cnt = MinU32(GEN_CHUNK, segHi - c);
            StageXSubrange(c, cnt);
            DataCopyExtParams copyParams{static_cast<uint16_t>(cnt), 2U * sizeof(float), srcGapBytes, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 6, 0.0f};
            DataCopyPad(
                aChunk_, aGM_[(static_cast<uint64_t>(c) * lda_ + row) * 2ULL], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t k = 0U; k < cnt; ++k) {
                float xr = 0.0f;
                float xi = 0.0f;
                ReadStagedX(k, cnt, xr, xi);
                const float arVal = aChunk_.GetValue(k * BLK_FLOATS);
                const float aiVal = aChunk_.GetValue(k * BLK_FLOATS + 1U);
                sumR += arVal * xr - aiVal * xi;
                sumI += arVal * xi + aiVal * xr;
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void AccumulateRowOffDiag(
        uint32_t row, float dRow, float t1RowR, float t1RowI, float& outR, float& outI)
    {
        if (uplo_ == ACLBLAS_UPPER) {
            float mirR = 0.0f;
            float mirI = 0.0f;
            if (row > 0U) {
                AccumulateMirroredSegment(row, 0U, row, mirR, mirI);
            }
            outR += t1RowR * dRow + (ar_ * mirR - ai_ * mirI);
            outI += t1RowI * dRow + (ar_ * mirI + ai_ * mirR);
            if (row + 1U < n_) {
                float dirR = 0.0f;
                float dirI = 0.0f;
                AccumulateDirectRowSegment(row, row + 1U, n_, dirR, dirI);
                outR += ar_ * dirR - ai_ * dirI;
                outI += ar_ * dirI + ai_ * dirR;
            }
            return;
        }
        if (row > 0U) {
            float dirR = 0.0f;
            float dirI = 0.0f;
            AccumulateDirectRowSegment(row, 0U, row, dirR, dirI);
            outR += ar_ * dirR - ai_ * dirI;
            outI += ar_ * dirI + ai_ * dirR;
        }
        outR += t1RowR * dRow;
        outI += t1RowI * dRow;
        if (row + 1U < n_) {
            float mirR = 0.0f;
            float mirI = 0.0f;
            AccumulateMirroredSegment(row, row + 1U, n_, mirR, mirI);
            outR += ar_ * mirR - ai_ * mirI;
            outI += ar_ * mirI + ai_ * mirR;
        }
    }

    __aicore__ inline void ProcessSingleRow(uint32_t row)
    {
        const uint64_t yPos = PhysicalIndex(row, n_, incy_, absIncy_) * 2ULL;
        const bool betaZero = (br_ == 0.0f && bi_ == 0.0f);
        const bool betaOne = (br_ == 1.0f && bi_ == 0.0f);
        float outR = 0.0f;
        float outI = 0.0f;
        if (!betaZero) {
            CopyBytesGmToUb(io_, yGM_[yPos], 2U * sizeof(float));
            PipeBarrier<PIPE_ALL>();
            const float yr = io_.GetValue(0);
            const float yi = io_.GetValue(1);
            outR = betaOne ? yr : (br_ * yr - bi_ * yi);
            outI = betaOne ? yi : (br_ * yi + bi_ * yr);
        }

        const uint64_t xRowPos = PhysicalIndex(row, n_, incx_, absIncx_) * 2ULL;
        CopyBytesGmToUb(io_[8], aGM_[(static_cast<uint64_t>(row) * lda_ + row) * 2ULL], sizeof(float));
        CopyBytesGmToUb(io_[16], xGM_[xRowPos], 2U * sizeof(float));
        PipeBarrier<PIPE_ALL>();
        const float dRow = io_.GetValue(8);
        const float xRowR = io_.GetValue(16);
        const float xRowI = io_.GetValue(17);
        const float t1RowR = ar_ * xRowR - ai_ * xRowI;
        const float t1RowI = ar_ * xRowI + ai_ * xRowR;

        AccumulateRowOffDiag(row, dRow, t1RowR, t1RowI, outR, outI);

        io_.SetValue(24, outR);
        io_.SetValue(25, outI);
        PipeBarrier<PIPE_ALL>();
        CopyBytesUbToGm(yGM_[yPos], io_[24], 2U * sizeof(float));
        PipeBarrier<PIPE_ALL>();
    }

    GlobalTensor<float> aGM_;
    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;

    TBuf<TPosition::VECCALC> xChunkBuf_;
    TBuf<TPosition::VECCALC> aChunkBuf_;
    TBuf<TPosition::VECCALC> ioBuf_;

    LocalTensor<float> xChunk_;
    LocalTensor<float> aChunk_;
    LocalTensor<float> io_;

    uint32_t blkIdx_ = 0U;
    uint32_t n_ = 0U;
    uint32_t lda_ = 0U;
    uint32_t numBlocks_ = 1U;
    uint32_t uplo_ = ACLBLAS_UPPER;
    float ar_ = 1.0f;
    float ai_ = 0.0f;
    float br_ = 0.0f;
    float bi_ = 0.0f;
    int64_t incx_ = 1;
    int64_t incy_ = 1;
    uint32_t absIncx_ = 1U;
    uint32_t absIncy_ = 1U;
};

} // namespace

__global__ __aicore__ void chemv_scale_kernel(GM_ADDR y, ChemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ProcessScaleOnly(y, tiling, pipe);
}

__global__ __aicore__ void chemv_colsplit_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, ChemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ChemvColSplitOp op;
    op.Init(a, x, y, workSpace, tiling, pipe);
    op.Process();
}

__global__ __aicore__ void chemv_general_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, ChemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ChemvGeneralOp op;
    op.Init(a, x, y, tiling, pipe);
    op.Process();
}

void chemv_scale_kernel_do(uint8_t* y, const ChemvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    chemv_scale_kernel<<<numBlocks, nullptr, stream>>>(y, tiling);
}

void chemv_colsplit_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const ChemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    chemv_colsplit_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
}

void chemv_general_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, const ChemvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    chemv_general_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, tiling);
}
