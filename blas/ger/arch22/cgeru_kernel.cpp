// AI生成
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgeru_kernel.cpp
 * \brief Device side of arch22 cgeru: complex64 non-conjugate rank-1 update
 *        A = alpha * x * y^T + A (column-major, y NOT conjugated).
 *
 * Elementwise the update per column j is A[:,j] += x * t_j with t_j = alpha*y_j,
 * i.e. re: A_re += xr*t_jr - xi*t_ji, im: A_im += xr*t_ji + xi*t_jr, evaluated
 * with plain mul/sub/add so the rounding order matches the reference CBLAS
 * driver (no fused multiply-adds). Columns whose y_j is (0,0) are skipped
 * entirely (reference semantics: A keeps its old values there, which matters
 * for Inf/NaN fills). The gate tests y itself, never the computed coefficient:
 * with a non-finite alpha component the coefficient of a zero-y column is NaN,
 * which must not turn the skipped column active.
 *
 * Two paths share the column-split across AIV cores
 * (colStart = n*blockIdx/useCoreNum, no core carries more than one extra
 * column):
 *
 *  - Fast path (incx == 1, lda == m, m % 4 == 0, m <= 4096, <=256 cols/core):
 *    x is staged once as the interleaved vector X plus its pair-swapped,
 *    even-negated copy S = (-xi, xr, ...). A column run then accumulates with
 *    three vector ops on the interleaved layout, out = A + cr*X - ci*S. The
 *    core slab is one contiguous GM range and is streamed in 16 KiB chunks
 *    through double-buffered VECIN/VECOUT queues, so MTE2 of chunk k+1, the
 *    vector work on chunk k and MTE3 of chunk k-1 overlap.
 *  - General path (strided x, padded lda, or oversized shapes): row bands of
 *    up to 2048 rows; per band X/S are rebuilt once and each column piece is
 *    pipelined through the same queues. Strided x is gathered by scalar loads.
 */

#include <cstdint>

#include "kernel_operator.h"

#include "cgeru_tiling_data.h"

using namespace AscendC;

namespace {

constexpr uint32_t CGERU_COMPONENTS = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t VREG_FLOATS = 64U; // one vector register covers 256 B = 64 fp32 lanes

// Fast path shape envelope (mirrors the host side checks).
constexpr uint32_t CGERU_FAST_CHUNK_ELEMS = 2048; // complex elements per streaming chunk (16 KiB)
constexpr uint32_t CGERU_FAST_MAX_COLS_PER_CORE = 256;
constexpr uint32_t CGERU_BUF_NUM = 2;
constexpr uint32_t CGERU_COEFF_COLS_ALIGN = 64U; // padding granularity of the coefficient planes

// General path row chunk (plane based, per column pipeline).
constexpr uint32_t CHUNK_ROWS = 1024;
constexpr uint32_t PLANE_F = CHUNK_ROWS;
constexpr uint32_t IDX_ENTRIES = 2 * CHUNK_ROWS;
constexpr uint32_t BUFFER_NUM = 2;
// Row slices bound one vector stage at a fixed size; larger runs are sliced.
constexpr uint32_t CGERU_SLICE_ROWS = 1024;
// Guard padding for the fast-path staged vectors: the block-granular even
// lane negation stores up to one whole 64 lane block past the valid tail.
constexpr uint32_t CGERU_GUARD_FLOATS = 64;

__aicore__ __inline__ uint32_t CgeruCeilDiv(uint32_t a, uint32_t b) { return (b == 0U) ? 0U : ((a + b - 1U) / b); }

__aicore__ __inline__ uint64_t CgeruMinU64(uint64_t a, uint64_t b) { return (a < b) ? a : b; }

// |v| without library calls; NaN passes through so activity flags stay non-zero.
__aicore__ __inline__ float CgeruFAbs(float v) { return (v < 0.0f) ? -v : v; }

// Set and immediately wait on one hardware event; the idiom used to sequence
// producer/consumer pairs on TBuf planes.
template <HardEvent EVENT>
__aicore__ __inline__ void CgeruSync()
{
    const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    SetFlag<EVENT>(ev);
    WaitFlag<EVENT>(ev);
}

// Coefficient pair of one column, plain mul/sub/add rounding:
//   cr = aR*yr - aI*yi ; ci = aI*yr + aR*yi
__aicore__ __inline__ void CgeruCoeffScalar(
    float alphaReal, float alphaImag, float yReal, float yImag, float& coeffReal, float& coeffImag)
{
    coeffReal = alphaReal * yReal - alphaImag * yImag;
    coeffImag = alphaImag * yReal + alphaReal * yImag;
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
        this->incx = tiling.incx;
        this->incy = tiling.incy;
        this->xStart = tiling.xStart;
        this->yStart = tiling.yStart;
        this->alphaReal = tiling.alphaReal;
        this->alphaImag = tiling.alphaImag;
        this->useFast = (tiling.useFast != 0U);

        const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint64_t colLo = static_cast<uint64_t>(n) * blockIdx;
        const uint64_t colHi = static_cast<uint64_t>(n) * (blockIdx + 1U);
        this->colStart = static_cast<uint32_t>(colLo / tiling.useCoreNum);
        const uint32_t colEnd = static_cast<uint32_t>(colHi / tiling.useCoreNum);
        this->colCount = colEnd - this->colStart;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    }

    __aicore__ inline void Process()
    {
        if (colCount == 0U || m == 0U || n == 0U) {
            return;
        }
        if (useFast) {
            InitFast();
            FastProcess();
        } else {
            InitGeneral();
            GeneralProcess();
        }
    }

private:
    // ------------------------------------------------------------------
    // Shared helpers
    // ------------------------------------------------------------------

    // idx[i] = 4 * (i ^ 1): the byte offset table that walks xKeep into its
    // adjacent-pair swap. Built from an 8 entry seed doubled with Adds, so no
    // extra scratch beyond the table itself is needed.
    __aicore__ inline void BuildSwapTable(uint32_t entries)
    {
        LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
        const uint32_t seedN = (entries < 8U) ? entries : 8U;
        for (uint32_t i = 0U; i < seedN; ++i) {
            idx.SetValue(i, static_cast<int32_t>(4U * (i ^ 1U)));
        }
        CgeruSync<HardEvent::S_V>();

        uint32_t filled = seedN;
        while (filled < entries) {
            const uint32_t remain = entries - filled;
            const uint32_t addN = (filled < remain) ? filled : remain;
            Adds(idx[filled], idx, static_cast<int32_t>(filled) * static_cast<int32_t>(BYTENUM_PER_FLOAT32), addN);
            PipeBarrier<PIPE_V>();
            filled += addN;
        }
    }

    // swap[i] = (-1)^i * keep[i ^ 1]: the pair swap of keep with every even
    // lane negated. The negation uses a per-lane vector mask and costs one
    // masked Muls without any sign buffer.
    __aicore__ inline void BuildSwapPattern(LocalTensor<float>& swap, LocalTensor<float>& keep, uint32_t totalFloats)
    {
        // Gather walks whole 64 lane blocks; the table covers the rounded up
        // count (BuildSwapTable was called with the same rounding) and the
        // tail lanes land in the swap guard.
        const uint32_t total = CgeruCeilDiv(totalFloats, VREG_FLOATS) * VREG_FLOATS;
        Gather(swap, keep, idxBuf.Get<uint32_t>(), 0U, total);
        PipeBarrier<PIPE_V>();

        constexpr uint64_t kEvenMask = 0x5555555555555555ULL;
        const uint16_t repeats = static_cast<uint16_t>(CgeruCeilDiv(total, VREG_FLOATS));
        SetMaskNorm();
        SetVectorMask<float>(kEvenMask, kEvenMask);
        Muls<float, false>(swap, swap, -1.0F, static_cast<uint64_t>(0), repeats, UnaryRepeatParams(1, 1, 8, 8));
        PipeBarrier<PIPE_V>();
        SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }

    // ------------------------------------------------------------------
    // Fast path
    // ------------------------------------------------------------------

    __aicore__ inline void InitFast()
    {
        const uint32_t floats = m * CGERU_COMPONENTS;
        // xSwap is written by the block-granular even-lane negation: round its
        // extent up to whole vector blocks so the masked tail stays in bounds.
        const uint32_t swapFloats = CgeruCeilDiv(floats, VREG_FLOATS) * VREG_FLOATS;
        pipe->InitBuffer(xKeepBuf, floats * BYTENUM_PER_FLOAT32);
        pipe->InitBuffer(xSwapBuf, swapFloats * BYTENUM_PER_FLOAT32);
        pipe->InitBuffer(idxBuf, swapFloats * BYTENUM_PER_FLOAT32);

        const uint32_t coeffCols = CgeruCeilDiv(colCount, CGERU_COEFF_COLS_ALIGN) * CGERU_COEFF_COLS_ALIGN;
        pipe->InitBuffer(
            yInBuf, CgeruCeilDiv(coeffCols * CGERU_COMPONENTS, VREG_FLOATS) * VREG_FLOATS * BYTENUM_PER_FLOAT32);
        pipe->InitBuffer(coeffBuf, 6U * coeffCols * BYTENUM_PER_FLOAT32);

        const uint32_t queueSlotFloats = CGERU_FAST_CHUNK_ELEMS * CGERU_COMPONENTS + CGERU_GUARD_FLOATS;
        pipe->InitBuffer(scratchBuf, CGERU_SLICE_ROWS * CGERU_COMPONENTS * BYTENUM_PER_FLOAT32);
        pipe->InitBuffer(aInQueue, CGERU_BUF_NUM, queueSlotFloats * BYTENUM_PER_FLOAT32);
        pipe->InitBuffer(aOutQueue, CGERU_BUF_NUM, queueSlotFloats * BYTENUM_PER_FLOAT32);
        scratch = scratchBuf.Get<float>();

        yearPlane = coeffBuf.Get<float>()[0];
        yiimPlane = coeffBuf.Get<float>()[coeffCols];
        crPlane = coeffBuf.Get<float>()[2U * coeffCols];
        ciPlane = coeffBuf.Get<float>()[3U * coeffCols];
        yflagPlane = coeffBuf.Get<float>()[4U * coeffCols];

        // The swap table (V pipe) is built while the x/y loads (MTE2) are in
        // flight; the loads are waited on just before the Gather that needs
        // them. The table covers whole vector blocks for the block-granular
        // Gather.
        BuildSwapTable(swapFloats);
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyExtParams xParams{1, floats * BYTENUM_PER_FLOAT32, 0, 0, 0};
        LocalTensor<float> keep = xKeepBuf.Get<float>();
        DataCopyPad(keep, xGm[xStart * CGERU_COMPONENTS], xParams, noPad);
        if (incy == 1) {
            // This core updates columns [colStart, colStart+colCount): stage
            // that slab of y, not y[0..colCount).
            DataCopyExtParams yParams{1, colCount * CGERU_COMPONENTS * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPad(yInBuf.Get<float>(), yGm[(yStart + colStart) * CGERU_COMPONENTS], yParams, noPad);
        }
        primeOff = static_cast<uint64_t>(colStart) * m;
        // Length of THIS core's slab from primeOff (colCount*m would underflow
        // for colStart > colCount and overrun the neighbouring slab).
        const uint64_t slabEnd = static_cast<uint64_t>(colStart + colCount) * m;
        primeElems = static_cast<uint32_t>(CgeruMinU64(slabEnd - primeOff, CGERU_FAST_CHUNK_ELEMS));
        CgeruSync<HardEvent::MTE2_V>();

        xKeep = keep;
        xSwap = xSwapBuf.Get<float>();
        BuildSwapPattern(xSwap, xKeep, floats);
        if (incy == 1) {
            FastCoeffsVector(coeffCols);
        } else {
            FastCoeffsScalar();
        }
    }

    // Deinterleave the staged y slab into real/imag planes and form the
    // per-column coefficient planes (vector form, incy == 1).
    __aicore__ inline void FastCoeffsVector(uint32_t coeffCols)
    {
        LocalTensor<float> yIn = yInBuf.Get<float>();
        const uint32_t repeats = CgeruCeilDiv(colCount * CGERU_COMPONENTS, VREG_FLOATS);
        uint64_t rsvdCnt = 0;
        GatherMask(yearPlane, yIn, 1, false, 0, {1, static_cast<uint16_t>(repeats), 8, 8}, rsvdCnt);
        GatherMask(yiimPlane, yIn, 2, false, 0, {1, static_cast<uint16_t>(repeats), 8, 8}, rsvdCnt);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> tmp = coeffBuf.Get<float>()[5U * coeffCols];
        // Activity flag per the reference gate IF (Y(JY).NE.ZERO): |yr| + |yi|
        // is exactly zero only when y_j == (0,0); Inf/NaN components stay
        // non-zero. The coefficients themselves drive no skip decision: with a
        // non-finite alpha a zero-y column would carry NaN coefficients.
        Abs(yflagPlane, yearPlane, colCount);
        Abs(tmp, yiimPlane, colCount);
        Add(yflagPlane, yflagPlane, tmp, colCount);
        PipeBarrier<PIPE_V>();

        Muls(crPlane, yearPlane, alphaReal, colCount);
        Muls(tmp, yiimPlane, alphaImag, colCount);
        Sub(crPlane, crPlane, tmp, colCount);
        Muls(ciPlane, yearPlane, alphaImag, colCount);
        Muls(tmp, yiimPlane, alphaReal, colCount);
        Add(ciPlane, ciPlane, tmp, colCount);
        PipeBarrier<PIPE_V>();
        // The column walk below reads the planes with scalar GetValue (S pipe).
        CgeruSync<HardEvent::V_S>();
    }

    // Strided y: scalar gather of the per-core columns and scalar coefficients.
    __aicore__ inline void FastCoeffsScalar()
    {
        for (uint32_t c = 0U; c < colCount; ++c) {
            const int64_t yElem = static_cast<int64_t>(yStart) + static_cast<int64_t>(colStart + c) * incy;
            const float yReal = yGm.GetValue(static_cast<uint64_t>(yElem) * CGERU_COMPONENTS);
            const float yImag = yGm.GetValue(static_cast<uint64_t>(yElem) * CGERU_COMPONENTS + 1U);
            float coeffReal = 0.0f;
            float coeffImag = 0.0f;
            CgeruCoeffScalar(alphaReal, alphaImag, yReal, yImag, coeffReal, coeffImag);
            crPlane.SetValue(c, coeffReal);
            ciPlane.SetValue(c, coeffImag);
            // Activity flag from y itself (reference gate), never from the
            // coefficient: non-finite alpha makes a zero-y column's NaN.
            yflagPlane.SetValue(c, CgeruFAbs(yReal) + CgeruFAbs(yImag));
        }
        CgeruSync<HardEvent::S_V>();
    }

    // True when at least one column overlapping the chunk [off, off+elems) has
    // y_j != (0,0) (the reference gate; computed from y itself so a non-finite
    // alpha cannot make a skipped column look active). Chunks with nothing to
    // update are skipped whole: their VECOUT staging would otherwise be stored
    // back to GM uninitialized and clobber A (y == 0 columns leave A untouched).
    __aicore__ inline bool ChunkHasWork(uint64_t off, uint32_t elems)
    {
        const uint64_t end = off + elems;
        uint64_t j = off / m;
        const uint64_t jEnd = static_cast<uint64_t>(colStart) + colCount;
        while (j < jEnd && j * m < end) {
            const uint32_t c = static_cast<uint32_t>(j - colStart);
            if (yflagPlane.GetValue(c) != 0.0f) {
                return true;
            }
            ++j;
        }
        return false;
    }

    __aicore__ inline void FastProcess()
    {
        const uint64_t slabHi = static_cast<uint64_t>(colStart + colCount) * m;
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};

        // Prime the first chunk here rather than in InitFast: issuing it after
        // the swap/coeff build keeps that 16 KiB transfer from sitting in
        // front of the MTE2->V barrier the Gather waits on.
        inTile = aInQueue.AllocTensor<float>();
        DataCopyExtParams primeParams{1, primeElems * CGERU_COMPONENTS * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPad(inTile, aGm[primeOff * CGERU_COMPONENTS], primeParams, noPad);
        aInQueue.EnQue<float>(inTile);

        // Serial chunk walk. The first chunk was already issued inside
        // InitFast, so its copy overlapped the swap/coeff build; every later
        // chunk is loaded after the previous store was issued, letting the
        // MTE3 transfer run alongside the next MTE2 load.
        uint64_t off = primeOff;
        uint32_t elems = primeElems;
        bool primed = true;
        while (off < slabHi) {
            inTile = aInQueue.DeQue<float>();
            const bool hasWork = ChunkHasWork(off, elems);
            if (hasWork) {
                const uint32_t bytes = elems * CGERU_COMPONENTS * BYTENUM_PER_FLOAT32;
                LocalTensor<float> out = aOutQueue.AllocTensor<float>();
                FastWalkColumns(off, elems, out);
                aOutQueue.EnQue<float>(out);
                out = aOutQueue.DeQue<float>();
                DataCopyExtParams outParams{1, bytes, 0, 0, 0};
                DataCopyPad(aGm[off * CGERU_COMPONENTS], out, outParams);
                aOutQueue.FreeTensor(out);
            }
            aInQueue.FreeTensor(inTile);

            const uint64_t nextOff = off + CGERU_FAST_CHUNK_ELEMS;
            if (nextOff >= slabHi) {
                break;
            }
            elems = static_cast<uint32_t>(CgeruMinU64(slabHi - nextOff, CGERU_FAST_CHUNK_ELEMS));
            inTile = aInQueue.AllocTensor<float>();
            DataCopyExtParams inParams{1, elems * CGERU_COMPONENTS * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPad(inTile, aGm[nextOff * CGERU_COMPONENTS], inParams, noPad);
            aInQueue.EnQue<float>(inTile);
            off = nextOff;
            primed = false;
        }
        (void)primed;
    }

    // Accumulate every column overlapping the chunk [off, off+elems) into out.
    __aicore__ inline void FastWalkColumns(uint64_t off, uint32_t elems, LocalTensor<float>& out)
    {
        const uint64_t end = off + elems;
        uint64_t j = off / m;
        while (j < (static_cast<uint64_t>(colStart) + colCount) && j * m < end) {
            const uint64_t colLo = j * m;
            const uint64_t runLo = (colLo > off) ? colLo : off;
            const uint64_t runHi = CgeruMinU64(colLo + m, end);
            if (runLo < runHi) {
                const uint32_t c = static_cast<uint32_t>(j - colStart);
                const float coeffReal = crPlane.GetValue(c);
                const float coeffImag = ciPlane.GetValue(c);
                const bool columnActive = yflagPlane.GetValue(c) != 0.0f;
                const uint32_t rows = static_cast<uint32_t>(runHi - runLo);
                const uint32_t dstFloats = static_cast<uint32_t>(runLo - off) * CGERU_COMPONENTS;
                if (columnActive) {
                    const uint32_t rowRel = static_cast<uint32_t>(runLo - colLo);
                    AccumulateRows(out, dstFloats, xKeep, xSwap, rowRel, rows, coeffReal, coeffImag);
                } else {
                    // Zero-y column: the reference gate leaves A untouched (even
                    // under a non-finite alpha whose coefficient is NaN), and
                    // this chunk is stored back as a whole, so its original
                    // bytes must be carried over to out.
                    Adds(out[dstFloats], inTile[dstFloats], 0.0f, rows * CGERU_COMPONENTS);
                }
            }
            ++j;
        }
    }

    // out = A + cr*keep + ci*swap over `rows` rows starting at row `rowRel`
    // inside the staged x band, mirrored at float offset `dstFloats` of out.
    // swap carries the even-negated pair swap, so its lanes are (-xi*ci, xr*ci)
    // and the lane values come out as (A + cr*xr) - ci*xi / (A + cr*xi) + ci*xr
    // with plain mul/sub/add rounding, matching the reference expression
    // order. Sliced at CGERU_SLICE_ROWS to bound the scratch plane.
    __aicore__ inline void AccumulateRows(
        LocalTensor<float>& out, uint32_t dstFloats, LocalTensor<float>& keep, LocalTensor<float>& swap,
        uint32_t rowRel, uint32_t rows, float coeffReal, float coeffImag)
    {
        uint32_t done = 0U;
        while (done < rows) {
            const uint32_t rowsNow = ((rows - done) < CGERU_SLICE_ROWS) ? (rows - done) : CGERU_SLICE_ROWS;
            const uint32_t floats = rowsNow * CGERU_COMPONENTS;
            const uint32_t srcOff = (rowRel + done) * CGERU_COMPONENTS;
            const uint32_t dstOff = dstFloats + done * CGERU_COMPONENTS;
            Muls(out[dstOff], keep[srcOff], coeffReal, floats);
            Muls(scratch, swap[srcOff], coeffImag, floats);
            Add(out[dstOff], out[dstOff], scratch, floats);
            Add(out[dstOff], inTile[dstOff], out[dstOff], floats);
            done += rowsNow;
        }
    }

    // ------------------------------------------------------------------
    //  General path (plane based, per column pipeline)
    // ---------------------------------------------------------------------

    __aicore__ inline void InitGeneral()
    {
        this->pipe->InitBuffer(xQueue, BUFFER_NUM, 2U * CHUNK_ROWS * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(aInQueue, BUFFER_NUM, 2U * CHUNK_ROWS * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(aOutQueue, BUFFER_NUM, 2U * CHUNK_ROWS * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(planeBuf, 6U * PLANE_F * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(genIdxBuf, IDX_ENTRIES * BYTENUM_PER_FLOAT32);
        this->pipe->InitBuffer(buildBuf, 2U * IDX_ENTRIES * BYTENUM_PER_FLOAT32);

        LocalTensor<float> planes = planeBuf.Get<float>();
        xR = planes;
        xI = planes[PLANE_F];
        t0 = planes[2U * PLANE_F];
        t1 = planes[3U * PLANE_F];
        calcR = planes[4U * PLANE_F];
        calcI = planes[5U * PLANE_F];
        idxU32 = genIdxBuf.Get<uint32_t>();
    }

    // idx[2t] = 4*t ; idx[2t+1] = 4*(CHUNK_ROWS + t), built once per core.
    __aicore__ inline void BuildIndex()
    {
        LocalTensor<int32_t> idx = genIdxBuf.Get<int32_t>();
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
                xIn.SetValue(k * CGERU_COMPONENTS, xGm.GetValue(static_cast<uint64_t>(elem) * CGERU_COMPONENTS));
                xIn.SetValue(
                    k * CGERU_COMPONENTS + 1U, xGm.GetValue(static_cast<uint64_t>(elem) * CGERU_COMPONENTS + 1U));
            }
        }
        xQueue.EnQue<float>(xIn);
        if (incx == 1) {
            // Staged chunk came from MTE2 (GM -> UB).
            event_t evIn = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evIn);
            WaitFlag<HardEvent::MTE2_V>(evIn);
        } else {
            // The strided fallback fills UB with scalar stores; GatherMask
            // below needs an S->V barrier (MTE2->V would not order them).
            event_t evScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(evScalar);
            WaitFlag<HardEvent::S_V>(evScalar);
        }

        LocalTensor<float> inLocal = xQueue.DeQue<float>();
        const uint32_t repeatTime = (rows * CGERU_COMPONENTS + 63U) / 64U;
        uint64_t rsvdCnt = 0;
        GatherMask(xR, inLocal, 1, false, 0, {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);
        GatherMask(xI, inLocal, 2, false, 0, {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        xQueue.FreeTensor(inLocal);
    }

    // out(rows*2 lanes) = interleave(calcR, calcI) = x * t ; then A += out.
    __aicore__ inline void ColumnChunk(uint32_t col, uint32_t rowOff, uint32_t rows)
    {
        const int64_t yElem = static_cast<int64_t>(yStart) + static_cast<int64_t>(col) * incy;
        const float yReal = yGm.GetValue(static_cast<uint64_t>(yElem) * CGERU_COMPONENTS);
        const float yImag = yGm.GetValue(static_cast<uint64_t>(yElem) * CGERU_COMPONENTS + 1U);
        // Reference gate: a column whose y is (0,0) is left untouched
        // (NaN/Inf count as non-zero and are not skipped).
        if (yReal == 0.0f && yImag == 0.0f) {
            return;
        }
        float coeffReal = 0.0f;
        float coeffImag = 0.0f;
        CgeruCoeffScalar(alphaReal, alphaImag, yReal, yImag, coeffReal, coeffImag);

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

    __aicore__ inline void GeneralProcess()
    {
        BuildIndex();
        for (uint32_t rowOff = 0U; rowOff < m; rowOff += CHUNK_ROWS) {
            const uint32_t rows = ((m - rowOff) < CHUNK_ROWS) ? (m - rowOff) : CHUNK_ROWS;
            LoadXChunk(rowOff, rows);
            for (uint32_t j = colStart; j < colStart + colCount; ++j) {
                ColumnChunk(j, rowOff, rows);
            }
        }
    }

    TPipe* pipe;
    TBuf<TPosition::VECCALC> xKeepBuf;
    TBuf<TPosition::VECCALC> xSwapBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    TBuf<TPosition::VECCALC> yInBuf;
    TBuf<TPosition::VECCALC> coeffBuf;
    TBuf<TPosition::VECCALC> scratchBuf;
    TQue<TPosition::VECIN, BUFFER_NUM> xQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> aInQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> aOutQueue;
    LocalTensor<float> scratch;
    TBuf<TPosition::VECCALC> planeBuf;
    TBuf<TPosition::VECCALC> genIdxBuf;
    TBuf<TPosition::VECCALC> buildBuf;
    LocalTensor<float> xR;
    LocalTensor<float> xI;
    LocalTensor<float> t0;
    LocalTensor<float> t1;
    LocalTensor<float> calcR;
    LocalTensor<float> calcI;
    LocalTensor<uint32_t> idxU32;

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> aGm;

    LocalTensor<float> xKeep;
    LocalTensor<float> xSwap;
    LocalTensor<float> inTile;
    LocalTensor<float> yearPlane;
    LocalTensor<float> yiimPlane;
    LocalTensor<float> crPlane;
    LocalTensor<float> ciPlane;
    LocalTensor<float> yflagPlane;

    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t colStart;
    uint32_t colCount;
    int64_t incx;
    int64_t incy;
    uint64_t xStart;
    uint64_t yStart;
    float alphaReal;
    float alphaImag;
    bool useFast;
    uint64_t primeOff;   // offset of the chunk primed during startup
    uint32_t primeElems; // elements of the chunk primed during startup
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
