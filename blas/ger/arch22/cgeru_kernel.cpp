// AI生成
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgeru_kernel.cpp
 * \brief Device side implementation for cgeru (arch22): complex64 non-conjugate
 *        rank-1 update A = alpha * x * y^T + A, column-major, y NOT conjugated.
 *
 * Two paths share one entry point:
 *
 *  - Fast path (incx == 1 and lda == m, i.e. both x and every column of A are
 *    contiguous). The whole column vector x is staged *once* as two UB vectors:
 *    xKeep holds the interleaved x (xr, xi, xr, xi, ...) and xSwap holds the
 *    adjacent-pair swap of xKeep with the even lane negated (-xi, xr, -xi, xr,
 *    ...). With those two vectors a whole column of A updates with two fused Axpy
 *    ops,
 *        out = A + cr * xKeep + ci * xSwap
 *    whose even lane is cr*xr - ci*xi and odd lane cr*xi + ci*xr, i.e. exactly the
 *    non-conjugate complex product interleaved in place. No per-column Gather and
 *    no per-column interleave table are needed. A whole burst of columns is copied
 *    in and out contiguously so the MTE latency is amortised over many columns.
 *
 *  - General path (incx != 1 or lda > m): per-column TPipe/TQue pipeline that
 *    gathers x apart with GatherMask and re-interleaves the product with Gather.
 *
 * On this SoC the DataCopyPad-based MTE copies are only usable inside a TPipe
 * managed pipeline, so every buffer below is created through the TPipe.
 */

#include <cstdint>

#include "kernel_operator.h"

#include "cgeru_tiling_data.h"

using namespace AscendC;

namespace {

constexpr uint32_t CGERU_COMPONENTS = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t CHUNK_ROWS = CGERU_MAX_CHUNK_ROWS;
constexpr uint32_t PLANE_F = CHUNK_ROWS;
constexpr uint32_t IDX_ENTRIES = CGERU_INTERLEAVE_ENTRIES;

// Fast path constants. A vector register covers 256 B = 64 fp32 lanes, and the
// smallest addressable UB block is 32 B = 8 fp32 lanes.
constexpr uint32_t VREG_FLOATS = 64U;
constexpr uint32_t ALIGN_FLOATS_32B = 8U;
// Total UB is 192 KiB (blas/common/arch/hardware.h); leave headroom below it so
// the four per-core scratch vectors plus at least one A column burst fit.
constexpr uint32_t UB_BUDGET_FAST = 176U * 1024U;

__aicore__ __inline__ uint32_t CgeruCeilDiv(uint32_t a, uint32_t b) { return (b == 0U) ? 0U : ((a + b - 1U) / b); }
__aicore__ __inline__ uint32_t CgeruAlignUp(uint32_t a, uint32_t b) { return CgeruCeilDiv(a, b) * b; }

// coeff = alpha * y. When one component of alpha is zero, the reference
// cblas_cgeru keeps the other component's product instead of forming 0 * Inf
// (TC_FL_103's golden mixes +/-Inf and NaN), so mirror that branch here to
// match the non-finite patterns. For finite inputs both forms are identical.
__aicore__ __inline__ void CgeruCoeff(
    float alphaReal, float alphaImag, float yReal, float yImag, float& coeffReal, float& coeffImag)
{
    if (alphaImag == 0.0f) {
        coeffReal = alphaReal * yReal;
        coeffImag = alphaReal * yImag;
        return;
    }
    if (alphaReal == 0.0f) {
        coeffReal = -alphaImag * yImag;
        coeffImag = alphaImag * yReal;
        return;
    }
    coeffReal = alphaReal * yReal - alphaImag * yImag;
    coeffImag = alphaReal * yImag + alphaImag * yReal;
}

// Set and immediately wait on one hardware event, the idiom the arch22 complex
// helpers use to sequence a TBuf producer/consumer pair.
template <HardEvent EVENT>
__aicore__ __inline__ void CgeruSync()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    SetFlag<EVENT>(ev);
    WaitFlag<EVENT>(ev);
}

class CgeruKernel {
public:
    __aicore__ inline CgeruKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR A, const CgeruTilingData& tiling, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->m = tiling.m;
        this->n = tiling.n;
        this->lda = tiling.lda;
        this->alphaReal = tiling.alphaReal;
        this->alphaImag = tiling.alphaImag;
        this->incx = tiling.incx;
        this->incy = tiling.incy;
        this->xStart = tiling.xStart;
        this->yStart = tiling.yStart;

        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        this->colStart = coreIdx * tiling.colsPerCore;
        if (tiling.colsPerCore == 0U || this->colStart >= this->n) {
            this->colCount = 0U;
        } else {
            const uint32_t remaining = this->n - this->colStart;
            this->colCount = (remaining < tiling.colsPerCore) ? remaining : tiling.colsPerCore;
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));

        // The fast path needs x contiguous and every A column contiguous so both
        // sides of a burst copy are plain memory runs. A whole interleaved column
        // must land on a 32 B boundary, which holds iff m is a multiple of 4.
        this->useFast = (this->incx == 1) && (this->lda == this->m) && (this->m > 0U) && (this->colCount > 0U) &&
                        ((this->m * CGERU_COMPONENTS) % ALIGN_FLOATS_32B == 0U);
        if (this->useFast && InitFast()) {
            return;
        }
        this->useFast = false;
        InitGeneral();
    }

    __aicore__ inline void Process()
    {
        if (colCount == 0U || m == 0U) {
            return;
        }
        if (useFast) {
            FastProcess();
            return;
        }
        BuildIndex();
        for (uint32_t rowOff = 0U; rowOff < m; rowOff += CHUNK_ROWS) {
            const uint32_t rows = ((m - rowOff) < CHUNK_ROWS) ? (m - rowOff) : CHUNK_ROWS;
            LoadXChunk(rowOff, rows);
            for (uint32_t j = 0U; j < colCount; ++j) {
                ColumnChunk(colStart + j, rowOff, rows);
            }
        }
    }

private:
    // ---------------------------------------------------------------------
    //  Fast path
    // ---------------------------------------------------------------------

    // Allocates the fast-path scratch and stages the per-core x vectors. Returns
    // false when the shape cannot afford even one column burst, in which case the
    // caller falls back to the general path.
    __aicore__ inline bool InitFast()
    {
        const uint32_t nFloat = m * CGERU_COMPONENTS;              // floats per interleaved column of x / A
        const uint32_t xTotal = CgeruAlignUp(nFloat, VREG_FLOATS); // whole 256 B repeats for the Gather span
        const uint32_t vecBytes = xTotal * BYTENUM_PER_FLOAT32;
        const uint32_t colBytes = nFloat * BYTENUM_PER_FLOAT32;

        // xKeep + xSwap + swapIdx, then the A column burst(s) in what is left.
        // y is NOT staged: its scalars are read straight from GM, which needs no
        // extra pipeline event and keeps the fixed per-kernel cost down.
        const uint32_t reserved = 3U * vecBytes;
        if (reserved + colBytes > UB_BUDGET_FAST) {
            return false;
        }
        uint32_t batch = (UB_BUDGET_FAST - reserved) / colBytes;
        if (batch > colCount) {
            batch = colCount;
        }
        if (batch == 0U) {
            return false;
        }

        pipe->InitBuffer(xKeepBuf, vecBytes);
        pipe->InitBuffer(xSwapBuf, vecBytes);
        pipe->InitBuffer(swapIdxBuf, vecBytes);
        pipe->InitBuffer(aBlkBuf, batch * colBytes);

        xKeep = xKeepBuf.Get<float>();
        xSwap = xSwapBuf.Get<float>();
        swapIdxU32 = swapIdxBuf.Get<uint32_t>();
        aBlk = aBlkBuf.Get<float>();

        fastN = nFloat;
        fastBatch = batch;

        // Overlap: the x copy is issued first, then the idx table is built while
        // the MTE2 transfer is in flight; only afterwards does the kernel wait.
        LoadFastX(nFloat);
        BuildFastSwapIndex(xTotal);
        CgeruSync<HardEvent::MTE2_V>();
        BuildFastSwap(xTotal);
        return true;
    }

    // idx[i] = 4 * (i ^ 1): the byte offset that turns xKeep into its adjacent-pair
    // swap. Built from an 8-entry seed and doubled with Adds so the table costs a
    // handful of vector ops and no host-side auxiliary buffer.
    __aicore__ inline void BuildFastSwapIndex(uint32_t total)
    {
        LocalTensor<int32_t> idx = swapIdxBuf.Get<int32_t>();
        idx.SetValue(0, 4);
        idx.SetValue(1, 0);
        idx.SetValue(2, 12);
        idx.SetValue(3, 8);
        idx.SetValue(4, 20);
        idx.SetValue(5, 16);
        idx.SetValue(6, 28);
        idx.SetValue(7, 24);
        CgeruSync<HardEvent::S_V>();

        uint32_t filled = 8U;
        while (filled < total) {
            const uint32_t addN = (filled < (total - filled)) ? filled : (total - filled);
            Adds(idx[filled], idx, static_cast<int32_t>(filled) * static_cast<int32_t>(BYTENUM_PER_FLOAT32), addN);
            PipeBarrier<PIPE_V>();
            filled += addN;
        }
    }

    // Stage x once (issue only — the caller decides when to wait, so the idx
    // table can be built while this MTE2 transfer is still in flight).
    __aicore__ inline void LoadFastX(uint32_t nFloat)
    {
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyExtParams xParams{1, nFloat * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPad(xKeep, xGm[xStart * CGERU_COMPONENTS], xParams, noPad);
    }

    // xSwap[i] = (-1)^i * xKeep[i ^ 1]: the adjacent-pair swap of xKeep with the
    // even lane negated. The negation uses a per-lane vector mask instead of a
    // staged sign vector, which costs one masked Muls and no UB buffer.
    __aicore__ inline void BuildFastSwap(uint32_t total)
    {
        Gather(xSwap, xKeep, swapIdxU32, 0U, total);
        PipeBarrier<PIPE_V>();

        constexpr uint64_t kEvenMask = 0x5555555555555555ULL;
        const uint16_t repeats = static_cast<uint16_t>(CgeruCeilDiv(total, VREG_FLOATS));
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(kEvenMask, kEvenMask);
        AscendC::Muls<float, false>(
            xSwap, xSwap, -1.0F, static_cast<uint64_t>(0), repeats, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }

    __aicore__ inline void FastProcess()
    {
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        for (uint32_t base = 0U; base < colCount; base += fastBatch) {
            const uint32_t cols = ((colCount - base) < fastBatch) ? (colCount - base) : fastBatch;
            const uint64_t aOff = static_cast<uint64_t>(colStart + base) * lda * CGERU_COMPONENTS;
            const uint32_t burstBytes = cols * fastN * BYTENUM_PER_FLOAT32;

            DataCopyExtParams inParams{1, burstBytes, 0, 0, 0};
            DataCopyPad(aBlk, aGm[aOff], inParams, noPad);
            CgeruSync<HardEvent::MTE2_V>();

            for (uint32_t f = 0U; f < cols; ++f) {
                // y scalars are read straight from GM: no UB staging, so no extra
                // MTE2->S event is needed and the fixed per-kernel cost stays down.
                const int64_t yElem = static_cast<int64_t>(yStart) + static_cast<int64_t>(colStart + base + f) * incy;
                const float yReal = yGm.GetValue(yElem * CGERU_COMPONENTS);
                const float yImag = yGm.GetValue(yElem * CGERU_COMPONENTS + 1U);
                // Netlib cgeru leaves A untouched for a zero y column, so a 0 * Inf
                // lane in the matching column of A must not fabricate NaN (see the
                // arch22 test ZeroYColumnSkipsInfAndNanX).
                if (yReal == 0.0f && yImag == 0.0f) {
                    continue;
                }
                float coeffReal = 0.0f;
                float coeffImag = 0.0f;
                CgeruCoeff(alphaReal, alphaImag, yReal, yImag, coeffReal, coeffImag);

                LocalTensor<float> dst = aBlk[f * fastN];
                Axpy(dst, xKeep, coeffReal, fastN);
                PipeBarrier<PIPE_V>();
                Axpy(dst, xSwap, coeffImag, fastN);
            }
            PipeBarrier<PIPE_V>();

            CgeruSync<HardEvent::V_MTE3>();
            DataCopyExtParams outParams{1, burstBytes, 0, 0, 0};
            DataCopyPad(aGm[aOff], aBlk, outParams);
            // Only the next batch reuses aBlk, so the trailing MTE3->MTE2 wait is
            // needed between batches but not after the last one: kernel exit
            // already drains every pipe.
            if (base + fastBatch < colCount) {
                CgeruSync<HardEvent::MTE3_MTE2>();
            }
        }
    }

    // ---------------------------------------------------------------------
    //  General path
    // ---------------------------------------------------------------------

    __aicore__ inline void InitGeneral()
    {
        this->pipe->InitBuffer(xQueue, BUFFER_NUM, 2U * CHUNK_ROWS * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(aInQueue, BUFFER_NUM, 2U * CHUNK_ROWS * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(aOutQueue, BUFFER_NUM, 2U * CHUNK_ROWS * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(planeBuf, 6U * PLANE_F * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(idxBuf, IDX_ENTRIES * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(buildBuf, 2U * IDX_ENTRIES * BYTENUM_PER_FLOAT32);

        LocalTensor<float> planes = planeBuf.Get<float>();
        xR = planes;
        xI = planes[PLANE_F];
        t0 = planes[2U * PLANE_F];
        t1 = planes[3U * PLANE_F];
        calcR = planes[4U * PLANE_F];
        calcI = planes[5U * PLANE_F];
        idxU32 = idxBuf.Get<uint32_t>();
    }

    // idx[2t] = 4*t ; idx[2t+1] = 4*(CHUNK_ROWS + t), built once per core.
    __aicore__ inline void BuildIndex()
    {
        LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
        LocalTensor<int32_t> bA = buildBuf.Get<int32_t>();
        LocalTensor<int32_t> bB = buildBuf.Get<int32_t>()[IDX_ENTRIES];

        CreateVecIndex(idx, 0, IDX_ENTRIES);
        PipeBarrier<PIPE_V>();
        ShiftRight(bA, idx, 1, IDX_ENTRIES);
        PipeBarrier<PIPE_V>();
        Muls(bB, bA, 2, IDX_ENTRIES);
        PipeBarrier<PIPE_V>();
        Sub(bB, idx, bB, IDX_ENTRIES);
        PipeBarrier<PIPE_V>();
        Muls(bA, bA, static_cast<int32_t>(BYTENUM_PER_FLOAT32), IDX_ENTRIES);
        Muls(bB, bB, static_cast<int32_t>(CHUNK_ROWS * BYTENUM_PER_FLOAT32), IDX_ENTRIES);
        PipeBarrier<PIPE_V>();
        Add(idx, bA, bB, IDX_ENTRIES);
        PipeBarrier<PIPE_V>();
    }

    // Stage the x chunk as deinterleaved planes (xR / xI).
    __aicore__ inline void LoadXChunk(uint32_t rowOff, uint32_t rows)
    {
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        LocalTensor<float> xIn = xQueue.AllocTensor<float>();
        if (incx == 1) {
            DataCopyExtParams xParams{1, static_cast<uint32_t>(rows * CGERU_COMPONENTS * BYTENUM_PER_FLOAT32), 0, 0, 0};
            DataCopyPad(xIn, xGm[(xStart + rowOff) * CGERU_COMPONENTS], xParams, noPad);
        } else {
            for (uint32_t k = 0U; k < rows; ++k) {
                const int64_t elem = static_cast<int64_t>(xStart) + static_cast<int64_t>(rowOff + k) * incx;
                xIn.SetValue(k * CGERU_COMPONENTS, xGm.GetValue(elem * CGERU_COMPONENTS));
                xIn.SetValue(k * CGERU_COMPONENTS + 1U, xGm.GetValue(elem * CGERU_COMPONENTS + 1U));
            }
        }
        xQueue.EnQue<float>(xIn);
        event_t evIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evIn);
        WaitFlag<HardEvent::MTE2_V>(evIn);

        LocalTensor<float> inLocal = xQueue.DeQue<float>();
        const uint32_t repeatTime = (rows * CGERU_COMPONENTS + 63U) / 64U;
        uint64_t rsvdCnt = 0;
        GatherMask(xR, inLocal, 1, false, 0, {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);
        GatherMask(xI, inLocal, 2, false, 0, {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        xQueue.FreeTensor(inLocal);
    }

    // out(rows*2 lanes) = interleave(calcR, calcI) = coeff * x ; then A += out.
    __aicore__ inline void ColumnChunk(uint32_t col, uint32_t rowOff, uint32_t rows)
    {
        const int64_t yElem = static_cast<int64_t>(yStart) + static_cast<int64_t>(col) * incy;
        const float yReal = yGm.GetValue(yElem * CGERU_COMPONENTS);
        const float yImag = yGm.GetValue(yElem * CGERU_COMPONENTS + 1U);
        // Netlib cgeru leaves A untouched for a zero y column, so a 0 * Inf lane in
        // the matching column of A must not fabricate NaN (see the arch22 test
        // ZeroYColumnSkipsInfAndNanX).
        if (yReal == 0.0f && yImag == 0.0f) {
            return;
        }
        float coeffReal = 0.0f;
        float coeffImag = 0.0f;
        CgeruCoeff(alphaReal, alphaImag, yReal, yImag, coeffReal, coeffImag);

        // outR = cr * xR - ci * xI ; outI = cr * xI + ci * xR
        Muls(t0, xR, coeffReal, rows);
        Muls(t1, xI, coeffImag, rows);
        PipeBarrier<PIPE_V>();
        Sub(calcR, t0, t1, rows);
        Muls(t0, xI, coeffReal, rows);
        Muls(t1, xR, coeffImag, rows);
        PipeBarrier<PIPE_V>();
        Add(calcI, t0, t1, rows);
        PipeBarrier<PIPE_V>();

        const uint32_t interCount = (rows * CGERU_COMPONENTS + 63U) / 64U * 64U;
        LocalTensor<float> outLocal = aOutQueue.AllocTensor<float>();
        Gather(outLocal, calcR, idxU32, 0U, interCount);
        PipeBarrier<PIPE_V>();

        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        const uint64_t aOff = (static_cast<uint64_t>(col) * lda + rowOff) * CGERU_COMPONENTS;
        LocalTensor<float> aIn = aInQueue.AllocTensor<float>();
        DataCopyExtParams aParams{1, static_cast<uint32_t>(rows * CGERU_COMPONENTS * BYTENUM_PER_FLOAT32), 0, 0, 0};
        DataCopyPad(aIn, aGm[aOff], aParams, noPad);
        aInQueue.EnQue<float>(aIn);
        event_t evIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evIn);
        WaitFlag<HardEvent::MTE2_V>(evIn);

        LocalTensor<float> aInLocal = aInQueue.DeQue<float>();
        Add(outLocal, outLocal, aInLocal, rows * CGERU_COMPONENTS);
        aInQueue.FreeTensor(aInLocal);
        PipeBarrier<PIPE_V>();

        aOutQueue.EnQue<float>(outLocal);
        event_t evOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evOut);
        WaitFlag<HardEvent::V_MTE3>(evOut);

        LocalTensor<float> outStored = aOutQueue.DeQue<float>();
        DataCopyExtParams oParams{1, static_cast<uint32_t>(rows * CGERU_COMPONENTS * BYTENUM_PER_FLOAT32), 0, 0, 0};
        DataCopyPad(aGm[aOff], outStored, oParams);
        aOutQueue.FreeTensor(outStored);

        event_t evBack = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(evBack);
        WaitFlag<HardEvent::MTE3_MTE2>(evBack);
    }

    TPipe* pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> xQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> aInQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> aOutQueue;
    TBuf<TPosition::VECCALC> planeBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    TBuf<TPosition::VECCALC> buildBuf;

    TBuf<TPosition::VECCALC> xKeepBuf;
    TBuf<TPosition::VECCALC> xSwapBuf;
    TBuf<TPosition::VECCALC> swapIdxBuf;
    TBuf<TPosition::VECCALC> aBlkBuf;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> aGm;
    LocalTensor<float> xR;
    LocalTensor<float> xI;
    LocalTensor<float> t0;
    LocalTensor<float> t1;
    LocalTensor<float> calcR;
    LocalTensor<float> calcI;
    LocalTensor<uint32_t> idxU32;

    LocalTensor<float> xKeep;
    LocalTensor<float> xSwap;
    LocalTensor<uint32_t> swapIdxU32;
    LocalTensor<float> aBlk;

    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t colStart;
    uint32_t colCount;
    float alphaReal;
    float alphaImag;
    int64_t incx;
    int64_t incy;
    uint64_t xStart;
    uint64_t yStart;
    uint32_t fastN;
    uint32_t fastBatch;
    bool useFast;
};

} // namespace

__global__ __aicore__ __vector__ void cgeru_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR A, const CgeruTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgeruKernel op;
    op.Init(x, y, A, tiling, &pipe);
    op.Process();
}

void cgeru_kernel_do(
    const uint8_t* x, const uint8_t* y, uint8_t* A, const CgeruTilingData& tiling, uint32_t numBlocks, void* stream)
{
    GM_ADDR kernelX = const_cast<uint8_t*>(x);
    GM_ADDR kernelY = const_cast<uint8_t*>(y);
    cgeru_kernel<<<numBlocks, nullptr, stream>>>(kernelX, kernelY, A, tiling);
}