/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cgemm_strided_batched_common.h"

// ---- pack implementation ----

namespace cgemm_sb {
// The input tile is [physical column][physical row][real/imag]. One indexed
// vector read both transposes and deinterleaves it into contiguous output rows.
class TransposePack {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(inputQ_, 1, kElements * 2U * sizeof(float));
        pipe.InitBuffer(realQ_, 1, kElements * sizeof(float));
        pipe.InitBuffer(imagQ_, 1, kElements * sizeof(float));
        pipe.InitBuffer(offsetBuf_, 2U * kElements * sizeof(uint32_t));
        auto real = offsetBuf_.GetWithOffset<int32_t>(kElements, 0U);
        auto imag = offsetBuf_.GetWithOffset<int32_t>(kElements, kElements * sizeof(uint32_t));
        AscendC::CreateVecIndex(real, 0, kTile);
        AscendC::Muls(real, real, static_cast<int32_t>(kTile * 2U * sizeof(float)), kTile);
        for (uint32_t row = 1; row < kTile; ++row) {
            AscendC::Adds(real[row * kTile], real, static_cast<int32_t>(row * 2U * sizeof(float)), kTile);
        }
        AscendC::Adds(imag, real, static_cast<int32_t>(sizeof(float)), kElements);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(
        AscendC::GlobalTensor<float>& src, AscendC::GlobalTensor<float>& dst, uint64_t realOffset, uint64_t imagOffset,
        uint32_t rows, uint32_t cols, uint32_t ld, bool conjugate)
    {
        auto even = offsetBuf_.GetWithOffset<uint32_t>(kElements, 0U);
        auto odd = offsetBuf_.GetWithOffset<uint32_t>(kElements, kElements * sizeof(uint32_t));
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyExtParams inCopy{kTile, kTile * 2U * sizeof(float), (ld - kTile) * 8U, 0U, 0U};
        DataCopyExtParams outCopy{kTile, kTile * sizeof(float), 0U, (cols - kTile) * 4U, 0U};
        const uint32_t colTiles = cols / kTile;
        const uint64_t tileCount = static_cast<uint64_t>(rows / kTile) * colTiles;
        for (uint64_t tile = GetBlockIdx(); tile < tileCount; tile += GetBlockNum()) {
            const uint32_t rowBase = static_cast<uint32_t>(tile / colTiles) * kTile;
            const uint32_t colBase = static_cast<uint32_t>(tile % colTiles) * kTile;
            auto input = inputQ_.AllocTensor<float>();
            DataCopyPad(input, src[(static_cast<uint64_t>(colBase) * ld + rowBase) * 2U], inCopy, pad);
            inputQ_.EnQue(input);
            input = inputQ_.DeQue<float>();
            auto real = realQ_.AllocTensor<float>();
            auto imag = imagQ_.AllocTensor<float>();
            Gather(real, input, even, 0U, kElements);
            Gather(imag, input, odd, 0U, kElements);
            if (conjugate) {
                Muls(imag, imag, -1.0f, kElements);
            }
            PipeBarrier<PIPE_ALL>();
            inputQ_.FreeTensor(input);
            realQ_.EnQue(real);
            imagQ_.EnQue(imag);
            real = realQ_.DeQue<float>();
            imag = imagQ_.DeQue<float>();
            const uint64_t out = static_cast<uint64_t>(rowBase) * cols + colBase;
            DataCopyPad(dst[realOffset + out], real, outCopy);
            DataCopyPad(dst[imagOffset + out], imag, outCopy);
            realQ_.FreeTensor(real);
            imagQ_.FreeTensor(imag);
        }
    }

private:
    static constexpr uint16_t kTile = 64U;
    static constexpr uint32_t kElements = kTile * kTile;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQ_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> realQ_, imagQ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
};
} // namespace cgemm_sb

// ---- general pack implementation ----

namespace cgemm_sb {
template <uint32_t tileSize = 64U>
class GeneralPack {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        pipe.InitBuffer(inputQ_, 1, (tileSize * (tileSize * 2U + 8U)) * sizeof(float));
        pipe.InitBuffer(realQ_, 1, (tileSize * tileSize) * sizeof(float));
        pipe.InitBuffer(imagQ_, 1, (tileSize * tileSize) * sizeof(float));
        pipe.InitBuffer(offsets_, 4U * (tileSize * tileSize) * sizeof(uint32_t));
        auto tr = offsets_.Get<int32_t>();
        auto direct =
            offsets_.GetWithOffset<int32_t>((tileSize * tileSize), (2U * tileSize * tileSize) * sizeof(uint32_t));
        AscendC::CreateVecIndex(tr, 0, tileSize);
        AscendC::CreateVecIndex(direct, 0, tileSize);
        AscendC::Muls(tr, tr, static_cast<int32_t>(tileSize * 8U + 32U), tileSize);
        AscendC::Muls(direct, direct, 8, tileSize);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t row = 1; row < tileSize; row *= 2U) {
            const uint32_t part = tileSize - row < row ? tileSize - row : row;
            AscendC::Adds(tr[row * tileSize], tr, static_cast<int32_t>(row * 8U), part * tileSize);
            AscendC::Adds(
                direct[row * tileSize], direct, static_cast<int32_t>(row * (tileSize * 8U + 32U)), part * tileSize);
        }
        AscendC::Adds(tr[(tileSize * tileSize)], tr, 4, (tileSize * tileSize));
        AscendC::Adds(direct[(tileSize * tileSize)], direct, 4, (tileSize * tileSize));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12, typename Arg13>
    __aicore__ inline void SbCopyGeneralTile(
        AscendC::GlobalTensor<float>& src, AscendC::GlobalTensor<float>& dst, Arg0& realOffset, Arg1& imagOffset,
        Arg2& paddedRows, Arg3& paddedCols, Arg4& ld, Arg5& trans, Arg6& conjugate, Arg7& even, Arg8& odd, Arg9& rb,
        Arg10& cb, Arg11& nr, Arg12& input, Arg13& inCopy)
    {
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>((8U - (nr * 2U) % 8U) % 8U), 0};
        DataCopyPad(input, src[(static_cast<uint64_t>(cb) * ld + rb) * 2U], inCopy, pad);
        inputQ_.EnQue(input);
        input = inputQ_.template DeQue<float>();
        auto real = realQ_.template AllocTensor<float>();
        auto imag = imagQ_.template AllocTensor<float>();
        Gather(real, input, even, 0U, (tileSize * tileSize));
        Gather(imag, input, odd, 0U, (tileSize * tileSize));
        if (conjugate) {
            Muls(imag, imag, -1.0f, (tileSize * tileSize));
        }
        PipeBarrier<PIPE_ALL>();
        inputQ_.FreeTensor(input);
        realQ_.EnQue(real);
        imagQ_.EnQue(imag);
        real = realQ_.template DeQue<float>();
        imag = imagQ_.template DeQue<float>();
        const uint32_t outRow = trans ? cb : rb;
        const uint32_t outCol = trans ? rb : cb;
        const uint32_t outRows = paddedRows - outRow < tileSize ? paddedRows - outRow : tileSize;
        const uint32_t outCols = paddedCols - outCol < tileSize ? paddedCols - outCol : tileSize;
        DataCopyExtParams outCopy{
            static_cast<uint16_t>(outRows), outCols * 4U, (tileSize - outCols) / 8U, (paddedCols - outCols) * 4U, 0U};
        const uint64_t out = static_cast<uint64_t>(outRow) * paddedCols + outCol;
        DataCopyPad(dst[realOffset + out], real, outCopy);
        DataCopyPad(dst[imagOffset + out], imag, outCopy);
        realQ_.FreeTensor(real);
        imagQ_.FreeTensor(imag);
    }

    __aicore__ inline void Process(
        AscendC::GlobalTensor<float>& src, AscendC::GlobalTensor<float>& dst, uint64_t realOffset, uint64_t imagOffset,
        uint32_t rows, uint32_t cols, uint32_t paddedRows, uint32_t paddedCols, uint32_t ld, bool trans, bool conjugate,
        uint32_t batch)
    {
        const uint32_t physicalRows = trans ? cols : rows;
        const uint32_t physicalCols = trans ? rows : cols;
        const uint32_t rowTiles = (physicalRows + (tileSize - 1U)) / tileSize;
        const uint32_t colTiles = (physicalCols + (tileSize - 1U)) / tileSize;
        const uint64_t tiles = static_cast<uint64_t>(rowTiles) * colTiles;
        const uint64_t first = (GetBlockIdx() + GetBlockNum() - (batch * tiles) % GetBlockNum()) % GetBlockNum();
        const uint32_t indexOffset = trans ? (2U * tileSize * tileSize) * sizeof(uint32_t) : 0U;
        auto even = offsets_.GetWithOffset<uint32_t>((tileSize * tileSize), indexOffset);
        auto odd = offsets_.GetWithOffset<uint32_t>(
            (tileSize * tileSize), indexOffset + (tileSize * tileSize) * sizeof(uint32_t));
        for (uint64_t tile = first; tile < tiles; tile += GetBlockNum()) {
            const uint32_t rb = static_cast<uint32_t>(tile / colTiles) * tileSize;
            const uint32_t cb = static_cast<uint32_t>(tile % colTiles) * tileSize;
            const uint32_t nr = physicalRows - rb < tileSize ? physicalRows - rb : tileSize;
            const uint32_t nc = physicalCols - cb < tileSize ? physicalCols - cb : tileSize;
            auto input = inputQ_.AllocTensor<float>();
            Duplicate(input, 0.0f, (tileSize * (tileSize * 2U + 8U)));
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams inCopy{
                static_cast<uint16_t>(nc), nr * 8U, (ld - nr) * 8U,
                ((tileSize * 8U + 32U) - ((nr * 8U + 31U) / 32U * 32U)) / 32U, 0U};
            SbCopyGeneralTile(
                src, dst, realOffset, imagOffset, paddedRows, paddedCols, ld, trans, conjugate, even, odd, rb, cb, nr,
                input, inCopy);
        }
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQ_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> realQ_, imagQ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsets_;
};
} // namespace cgemm_sb

// ---- small implementation ----

// Vector outer products over several compact square matrices per core.
template <
    uint32_t dimension, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12,
    typename Arg13, typename Arg14>
__aicore__ inline void SbAccumulateSmallProduct(
    Arg0& t, Arg1& tmp, Arg2& ar, Arg3& ai, Arg4& br, Arg5& bi, Arg6& product, Arg7& joined, Arg8& ao, Arg9& bo,
    Arg10& nb, Arg11& active, Arg12& av, Arg13& bv, Arg14& kk)
{
    const uint32_t ak = kk * (t.transA == ACLBLAS_OP_N ? t.m : 1U) * 8U;
    const uint32_t bk = kk * (t.transB == ACLBLAS_OP_N ? 1U : t.n) * 8U;
    AscendC::Gather(ar, av, ao, ak, active);
    AscendC::Gather(ai, av, ao, ak + 4U, active);
    if constexpr (dimension == 8U || dimension == 16U) {
        AscendC::Gather(tmp, bv, bo, bk, nb * dimension);
        AscendC::Gather(product, bv, bo, bk + 4U, nb * dimension);
        AscendC::Brcb(br, tmp, static_cast<uint8_t>(nb * dimension / 8U), {dimension / 8U, dimension});
        AscendC::Brcb(bi, product, static_cast<uint8_t>(nb * dimension / 8U), {dimension / 8U, dimension});
        if constexpr (dimension == 16U) {
            AscendC::Brcb(br[8U], tmp, static_cast<uint8_t>(nb * 2U), {2U, 16U});
            AscendC::Brcb(bi[8U], product, static_cast<uint8_t>(nb * 2U), {2U, 16U});
        }
    } else {
        AscendC::Gather(br, bv, bo, bk, active);
        AscendC::Gather(bi, bv, bo, bk + 4U, active);
    }
    if (t.transA == ACLBLAS_OP_C) {
        AscendC::Muls(ai, ai, -1.0f, active);
    }
    if (t.transB == ACLBLAS_OP_C) {
        AscendC::Muls(bi, bi, -1.0f, active);
    }
    AscendC::Mul(tmp, ar, br, active);
    AscendC::Mul(product, ai, bi, active);
    AscendC::Sub(tmp, tmp, product, active);
    AscendC::Add(joined, joined, tmp, active);
    AscendC::Mul(tmp, ar, bi, active);
    AscendC::Mul(product, ai, br, active);
    AscendC::Add(tmp, tmp, product, active);
}

template <
    uint32_t dimension, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11>
__aicore__ inline void SbBuildSmallFloatIndices(
    Arg0& t, Arg1& tmp, Arg2& ar, Arg3& ai, Arg4& br, Arg5& bi, Arg6& ia, Arg7& ib, Arg8& io, Arg9& length, Arg10& mf,
    Arg11& cf)
{
    constexpr uint32_t cap = 2048U;
    AscendC::Muls(ar, ar, cf, length); // Batch offset.
    AscendC::Sub(ai, tmp, ar, length); // Logical output index within matrix.
    AscendC::Muls(br, ai, 1.0f / mf, length);
    AscendC::Cast(ia, br, AscendC::RoundMode::CAST_FLOOR, length);
    AscendC::Cast(br, ia, AscendC::RoundMode::CAST_NONE, length); // Column.
    AscendC::Muls(bi, br, -mf, length);
    AscendC::Add(bi, bi, ai, length); // Row.
    if (t.transA != ACLBLAS_OP_N) {
        AscendC::Muls(bi, bi, mf, length);
    }
    if (t.transB == ACLBLAS_OP_N) {
        AscendC::Muls(br, br, mf, length);
    }
    AscendC::Add(bi, bi, ar, length);
    AscendC::Add(br, br, ar, length);
    AscendC::Muls(bi, bi, 8.0f, length);
    AscendC::Muls(br, br, 8.0f, length);
    AscendC::Cast(ia, bi, AscendC::RoundMode::CAST_RINT, length);
    AscendC::Cast(ib, br, AscendC::RoundMode::CAST_RINT, length);

    // Reuse work storage for interleaving offsets, in two halves.
    for (uint32_t start = 0; start < 2U * length; start += cap) {
        const uint32_t size = 2U * length - start < cap ? 2U * length - start : cap;
        AscendC::CreateVecIndex(ar, static_cast<float>(static_cast<int32_t>(start)), size);
        AscendC::Muls(ai, ar, 0.5f, size);
        AscendC::Cast(io[start], ai, AscendC::RoundMode::CAST_FLOOR, size);
        AscendC::Cast(ai, io[start], AscendC::RoundMode::CAST_NONE, size);
        AscendC::Muls(ar, ar, 8192.0f, size);
        AscendC::Muls(ai, ai, -16380.0f, size);
        AscendC::Add(ai, ai, ar, size);
        AscendC::Cast(io[start], ai, AscendC::RoundMode::CAST_RINT, size);
    }
}

template <
    uint32_t dimension, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12,
    typename Arg13, typename Arg14, typename Arg15>
__aicore__ inline void SbComputeSmallBatch(
    Arg0& t, Arg1& aq, Arg2& bq, Arg3& tmp, Arg4& ar, Arg5& ai, Arg6& br, Arg7& bi, Arg8& product, Arg9& joined,
    Arg10& ao, Arg11& bo, Arg12& nb, Arg13& active, Arg14& av, Arg15& bv)
{
    constexpr uint32_t cap = 2048U;
    aq.EnQue(av);
    bq.EnQue(bv);
    av = aq.template DeQue<float>();
    bv = bq.template DeQue<float>();
    AscendC::Duplicate(joined, 0.0f, 2U * cap);
    for (uint32_t kk = 0; kk < dimension; ++kk) {
        SbAccumulateSmallProduct<dimension>(t, tmp, ar, ai, br, bi, product, joined, ao, bo, nb, active, av, bv, kk);
        AscendC::Add(joined[cap], joined[cap], tmp, active);
    }
    // The transposed reference applies complex alpha after the dot product.
    // This path has alpha=(1,0). Preserve the complex multiply's signed
    // infinity recovery when its ordinary cross terms produce NaNs.
    if (t.transA != ACLBLAS_OP_N) {
        const uint32_t aligned = (active + 63U) / 64U * 64U;
        AscendC::Muls(ar, joined, t.alphaReal, aligned);
        AscendC::Muls(ai, joined[cap], t.alphaImag, aligned);
        AscendC::Muls(br, joined, t.alphaImag, aligned);
        AscendC::Muls(bi, joined[cap], t.alphaReal, aligned);
        AscendC::Sub(ar, ar, ai, aligned);
        AscendC::Add(bi, br, bi, aligned);
        auto infinite = product.template ReinterpretCast<uint8_t>();
        AscendC::Abs(tmp, joined, aligned);
        AscendC::Compares(infinite, tmp, 3.402823466e38f, AscendC::CMPMODE::GT, aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(joined, infinite, joined, ar, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Abs(tmp, joined[cap], aligned);
        AscendC::Compares(infinite, tmp, 3.402823466e38f, AscendC::CMPMODE::GT, aligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(joined[cap], infinite, joined[cap], bi, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, aligned);
    }
}

template <
    uint32_t dimension, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12,
    typename Arg13, typename Arg14, typename Arg15, typename Arg16>
__aicore__ inline void SbRunSmallBatches(
    Arg0& a, Arg1& b, Arg2& c, Arg3& t, Arg4& aq, Arg5& bq, Arg6& cq, Arg7& indexBuf, Arg8& tmp, Arg9& ar, Arg10& ai,
    Arg11& br, Arg12& bi, Arg13& product, Arg14& joined, Arg15& count, Arg16& batches)
{
    constexpr uint32_t cap = 2048U;
    AscendC::PipeBarrier<PIPE_ALL>();
    auto ao = indexBuf.template Get<uint32_t>();
    auto bo = indexBuf.template GetWithOffset<uint32_t>(cap, cap * 4U);
    auto co = indexBuf.template GetWithOffset<uint32_t>(2U * cap, 2U * cap * 4U);
    AscendC::GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    for (uint32_t first = AscendC::GetBlockIdx() * batches; first < t.batchCount;
         first += AscendC::GetBlockNum() * batches) {
        const uint32_t nb = t.batchCount - first < batches ? t.batchCount - first : batches;
        const uint32_t active = nb * count;
        auto av = aq.template AllocTensor<float>();
        auto bv = bq.template AllocTensor<float>();
        AscendC::DataCopy(av, ag[static_cast<uint64_t>(first) * count * 2U], active * 2U);
        AscendC::DataCopy(bv, bg[static_cast<uint64_t>(first) * count * 2U], active * 2U);
        SbComputeSmallBatch<dimension>(t, aq, bq, tmp, ar, ai, br, bi, product, joined, ao, bo, nb, active, av, bv);
        AscendC::PipeBarrier<PIPE_ALL>();
        aq.FreeTensor(av);
        bq.FreeTensor(bv);
        auto out = cq.template AllocTensor<float>();
        AscendC::Gather(out, joined, co, 0U, active * 2U);
        cq.EnQue(out);
        out = cq.template DeQue<float>();
        AscendC::DataCopy(cg[static_cast<uint64_t>(first) * count * 2U], out, active * 2U);
        cq.FreeTensor(out);
    }
}

template <
    uint32_t dimension, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12,
    typename Arg13, typename Arg14, typename Arg15, typename Arg16, typename Arg17>
__aicore__ inline void SbPrepareSmallRightIndices(
    Arg0& a, Arg1& b, Arg2& c, Arg3& t, Arg4& aq, Arg5& bq, Arg6& cq, Arg7& indexBuf, Arg8& tmp, Arg9& ar, Arg10& ai,
    Arg11& br, Arg12& bi, Arg13& product, Arg14& joined, Arg15& ib, Arg16& count, Arg17& batches)
{
    if constexpr (dimension == 8U || dimension == 16U) {
        AscendC::CreateVecIndex(ib, 0, dimension);
        AscendC::Muls(ib, ib, static_cast<int32_t>(t.transB == ACLBLAS_OP_N ? dimension * 8U : 8U), dimension);
        for (uint32_t batch = 1U; batch < batches; batch *= 2U) {
            const uint32_t part = batches - batch < batch ? batches - batch : batch;
            AscendC::Adds(ib[batch * dimension], ib, static_cast<int32_t>(batch * count * 8U), part * dimension);
        }
    }
    SbRunSmallBatches<dimension>(
        a, b, c, t, aq, bq, cq, indexBuf, tmp, ar, ai, br, bi, product, joined, count, batches);
}

template <
    uint32_t dimension, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6>
__aicore__ inline void SbBuildSmallIntegerIndices(
    Arg0& t, Arg1& ia, Arg2& ib, Arg3& io, Arg4& count, Arg5& batches, Arg6& length)
{
    constexpr uint32_t cap = 2048U;
    AscendC::CreateVecIndex(ia, 0, dimension);
    AscendC::Muls(ia, ia, static_cast<int32_t>(t.transA == ACLBLAS_OP_N ? 8U : dimension * 8U), dimension);
    AscendC::Duplicate(ib, 0, dimension);
    for (uint32_t columns = 1U; columns < dimension; columns *= 2U) {
        const uint32_t part = dimension - columns < columns ? dimension - columns : columns;
        AscendC::Adds(ia[columns * dimension], ia, 0, part * dimension);
        AscendC::Adds(
            ib[columns * dimension], ib,
            static_cast<int32_t>(columns * (t.transB == ACLBLAS_OP_N ? dimension : 1U) * 8U), part * dimension);
    }
    for (uint32_t batch = 1U; batch < batches; batch *= 2U) {
        const uint32_t part = batches - batch < batch ? batches - batch : batch;
        AscendC::Adds(ia[batch * count], ia, static_cast<int32_t>(batch * count * 8U), part * count);
        AscendC::Adds(ib[batch * count], ib, static_cast<int32_t>(batch * count * 8U), part * count);
    }
    AscendC::CreateVecIndex(io, 0, 16U);
    AscendC::ShiftRight(io, io, 1, 16U);
    AscendC::Muls(io, io, 4, 16U);
    uint64_t oddSeed[2] = {0xaaaaULL, 0U};
    AscendC::Adds(io, io, static_cast<int32_t>(cap * 4U), oddSeed, 1U, {1, 1, 8, 8});
    for (uint32_t step = 16U; step < 2U * length; step *= 2U) {
        const uint32_t part = 2U * length - step < step ? 2U * length - step : step;
        AscendC::Adds(io[step], io, static_cast<int32_t>(step * 2U), part);
    }
}

template <
    uint32_t dimension, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__aicore__ inline void SbRunSmallVector(
    Arg0& a, Arg1& b, Arg2& c, Arg3& t, Arg4& pipe, Arg5& aq, Arg6& bq, Arg7& cq, Arg8& workBuf, Arg9& indexBuf,
    Arg10& resultBuf)
{
    constexpr uint32_t cap = 2048U;
    pipe.InitBuffer(indexBuf, 4U * cap * sizeof(uint32_t));
    pipe.InitBuffer(resultBuf, 2U * cap * sizeof(float));
    auto tmp = workBuf.template Get<float>();
    auto ar = workBuf.template GetWithOffset<float>(cap, cap * 4U);
    auto ai = workBuf.template GetWithOffset<float>(cap, cap * 8U);
    auto br = workBuf.template GetWithOffset<float>(cap, cap * 12U);
    auto bi = workBuf.template GetWithOffset<float>(cap, cap * 16U);
    auto product = workBuf.template GetWithOffset<float>(cap, cap * 20U);
    auto joined = resultBuf.template Get<float>();
    auto ia = indexBuf.template Get<int32_t>();
    auto ib = indexBuf.template GetWithOffset<int32_t>(cap, cap * 4U);
    auto io = indexBuf.template GetWithOffset<int32_t>(2U * cap, 2U * cap * 4U);
    const uint32_t count = t.m * t.n;
    const uint32_t balanced = (t.batchCount + AscendC::GetBlockNum() - 1U) / AscendC::GetBlockNum();
    const uint32_t batches = balanced < cap / count ? balanced : cap / count;
    const uint32_t length = batches * count;
    if constexpr (dimension == 8U || dimension == 16U) {
        // Integer periodic indices avoid division, flooring and float conversions.
        SbBuildSmallIntegerIndices<dimension>(t, ia, ib, io, count, batches, length);
    } else {
        const float mf = static_cast<float>(static_cast<int32_t>(t.m));
        const float cf = static_cast<float>(static_cast<int32_t>(count));
        AscendC::CreateVecIndex(tmp, 0.0f, length);
        AscendC::Muls(ar, tmp, 1.0f / cf, length);
        AscendC::Cast(ia, ar, AscendC::RoundMode::CAST_FLOOR, length);
        AscendC::Cast(ar, ia, AscendC::RoundMode::CAST_NONE, length);
        SbBuildSmallFloatIndices<dimension>(t, tmp, ar, ai, br, bi, ia, ib, io, length, mf, cf);
    }
    SbPrepareSmallRightIndices<dimension>(
        a, b, c, t, aq, bq, cq, indexBuf, tmp, ar, ai, br, bi, product, joined, ib, count, batches);
}

template <uint32_t dimension>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_small_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t cap = 2048U;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> aq, bq;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> cq;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf, indexBuf, resultBuf;
    pipe.InitBuffer(aq, 1, 2U * cap * sizeof(float));
    pipe.InitBuffer(bq, 1, 2U * cap * sizeof(float));
    pipe.InitBuffer(cq, 1, 2U * cap * sizeof(float));
    pipe.InitBuffer(workBuf, 6U * cap * sizeof(float));
    SbRunSmallVector<dimension>(a, b, c, t, pipe, aq, bq, cq, workBuf, indexBuf, resultBuf);
}

using namespace AscendC;

__aicore__ inline uint64_t SbAr(const CgemmSbTiling& t) { return 0; }
__aicore__ inline uint64_t SbAi(const CgemmSbTiling& t) { return static_cast<uint64_t>(t.paddedM) * t.paddedK; }
__aicore__ inline uint64_t SbBr(const CgemmSbTiling& t) { return 2ULL * t.paddedM * t.paddedK; }
__aicore__ inline uint64_t SbBi(const CgemmSbTiling& t)
{
    return SbBr(t) + static_cast<uint64_t>(t.paddedK) * t.paddedN;
}
__aicore__ inline uint64_t SbRr(const CgemmSbTiling& t)
{
    return SbBi(t) + static_cast<uint64_t>(t.paddedK) * t.paddedN;
}
__aicore__ inline uint64_t SbIi(const CgemmSbTiling& t)
{
    return SbRr(t) + static_cast<uint64_t>(t.paddedM) * t.paddedN;
}
__aicore__ inline uint64_t SbRi(const CgemmSbTiling& t)
{
    return SbIi(t) + static_cast<uint64_t>(t.paddedM) * t.paddedN;
}
__aicore__ inline uint64_t SbIr(const CgemmSbTiling& t)
{
    return SbRi(t) + static_cast<uint64_t>(t.paddedM) * t.paddedN;
}

// For aligned shapes, N operands can be deinterleaved directly; T/C operands
// use tiled vector transposition. Workspace stores row-major op(A)^T/op(B)^T.
// Compute C^T = B^T A^T so both input planes and final C are contiguous.
constexpr uint32_t kSbVectorTile = 256U;

__aicore__ inline void SbBuildVectorOffsets(LocalTensor<int32_t>& even, LocalTensor<int32_t>& odd)
{
    CreateVecIndex(even, 0, 8U);
    Muls(even, even, 8, 8U);
    for (uint32_t group = 1U; group < kSbVectorTile / 8U; ++group) {
        Adds(even[group * 8U], even, static_cast<int32_t>(group * 64U), 8U);
    }
    Adds(odd, even, 4, kSbVectorTile);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void SbPackContiguousColumns(
    GlobalTensor<float>& src, GlobalTensor<float>& ws, uint64_t realOffset, uint64_t imagOffset, uint32_t rows,
    uint32_t cols, uint32_t ld, TQue<TPosition::VECIN, 2>& inputQ, TQue<TPosition::VECOUT, 2>& realQ,
    TQue<TPosition::VECOUT, 2>& imagQ, LocalTensor<uint32_t>& even, LocalTensor<uint32_t>& odd)
{
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyExtParams inCopy{1, kSbVectorTile * 2U * sizeof(float), 0, 0, 0};
    DataCopyExtParams outCopy{1, kSbVectorTile * sizeof(float), 0, 0, 0};
    const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blocks = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t col = block; col < cols; col += blocks) {
        for (uint32_t row = 0; row < rows; row += kSbVectorTile) {
            auto input = inputQ.AllocTensor<float>();
            DataCopyPad(input, src[(static_cast<uint64_t>(col) * ld + row) * 2U], inCopy, pad);
            inputQ.EnQue(input);
            input = inputQ.DeQue<float>();
            auto real = realQ.AllocTensor<float>();
            auto imag = imagQ.AllocTensor<float>();
            Gather(real, input, even, 0U, kSbVectorTile);
            Gather(imag, input, odd, 0U, kSbVectorTile);
            PipeBarrier<PIPE_ALL>();
            inputQ.FreeTensor(input);
            realQ.EnQue(real);
            imagQ.EnQue(imag);
            real = realQ.DeQue<float>();
            imag = imagQ.DeQue<float>();
            const uint64_t out = static_cast<uint64_t>(col) * rows + row;
            DataCopyPad(ws[realOffset + out], real, outCopy);
            DataCopyPad(ws[imagOffset + out], imag, outCopy);
            realQ.FreeTensor(real);
            imagQ.FreeTensor(imag);
        }
    }
}

extern "C" __attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_pack_fast_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const CgemmSbTiling t, const CgemmSbTiling s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TQue<TPosition::VECIN, 2> inputQ;
    TQue<TPosition::VECOUT, 2> realQ, imagQ;
    TBuf<TPosition::VECCALC> offsetBuf;
    pipe.InitBuffer(inputQ, 2, kSbVectorTile * 2U * sizeof(float));
    pipe.InitBuffer(realQ, 2, kSbVectorTile * sizeof(float));
    pipe.InitBuffer(imagQ, 2, kSbVectorTile * sizeof(float));
    pipe.InitBuffer(offsetBuf, 2U * kSbVectorTile * sizeof(uint32_t));
    auto evenI32 = offsetBuf.GetWithOffset<int32_t>(kSbVectorTile, 0U);
    auto oddI32 = offsetBuf.GetWithOffset<int32_t>(kSbVectorTile, kSbVectorTile * sizeof(uint32_t));
    SbBuildVectorOffsets(evenI32, oddI32);
    auto even = offsetBuf.GetWithOffset<uint32_t>(kSbVectorTile, 0U);
    auto odd = offsetBuf.GetWithOffset<uint32_t>(kSbVectorTile, kSbVectorTile * sizeof(uint32_t));
    GlobalTensor<float> ag, bg, ws;
    cgemm_sb::TransposePack transpose;
    if (t.transA != ACLBLAS_OP_N || t.transB != ACLBLAS_OP_N) {
        transpose.Init(pipe);
    }
    for (uint32_t batch = 0; batch < t.batchCount; ++batch) {
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a) + static_cast<int64_t>(batch) * t.strideA * 2);
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b) + static_cast<int64_t>(batch) * t.strideB * 2);
        ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + batch * t.workspaceStride);
        // op(B)^T is the left operand; op(A)^T is the right operand.
        if (t.transB == ACLBLAS_OP_N) {
            SbPackContiguousColumns(bg, ws, SbAr(s), SbAi(s), t.k, t.n, t.ldb, inputQ, realQ, imagQ, even, odd);
        } else {
            transpose.Process(bg, ws, SbAr(s), SbAi(s), t.n, t.k, t.ldb, t.transB == ACLBLAS_OP_C);
        }
        if (t.transA == ACLBLAS_OP_N) {
            SbPackContiguousColumns(ag, ws, SbBr(s), SbBi(s), t.m, t.k, t.lda, inputQ, realQ, imagQ, even, odd);
        } else {
            transpose.Process(ag, ws, SbBr(s), SbBi(s), t.k, t.m, t.lda, t.transA == ACLBLAS_OP_C);
        }
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
    PipeBarrier<PIPE_ALL>();
}

constexpr uint32_t kCgemmSbCubeBaseM = 64U;
constexpr uint32_t kCgemmSbCubeBaseN = 64U;

// The Cube path consumes the row-major SoA matrices produced by preprocess.
// Four CO1 tiles retain the real products in one K loop. Each input plane
// is loaded once per K tile and reused by two MMADs. FIX_M protects reuse
// of all four accumulators after their M -> Fixpipe lifecycle.
template <uint32_t kCgemmSbCubeBaseK, uint32_t baseMN = 64U>
class CgemmSbBasicMmad {
public:
    __aicore__ inline void Init(
        __gm__ float* ar, __gm__ float* ai, __gm__ float* br, __gm__ float* bi, __gm__ float* rr, __gm__ float* ii,
        __gm__ float* ri, __gm__ float* ir, uint32_t m, uint32_t n, uint32_t k)
    {
        ar_.SetGlobalBuffer(ar, static_cast<uint64_t>(m) * k);
        ai_.SetGlobalBuffer(ai, static_cast<uint64_t>(m) * k);
        br_.SetGlobalBuffer(br, static_cast<uint64_t>(k) * n);
        bi_.SetGlobalBuffer(bi, static_cast<uint64_t>(k) * n);
        rr_.SetGlobalBuffer(rr, static_cast<uint64_t>(m) * n);
        ii_.SetGlobalBuffer(ii, static_cast<uint64_t>(m) * n);
        ri_.SetGlobalBuffer(ri, static_cast<uint64_t>(m) * n);
        ir_.SetGlobalBuffer(ir, static_cast<uint64_t>(m) * n);
        m_ = m;
        n_ = n;
        k_ = k;
    }

    __aicore__ inline void PrepareCo1() { SetFlag<HardEvent::FIX_M>(EVENT_ID0); }

    template <typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
    __aicore__ inline void SbCopyCubeOperands(
        Arg0& ar1, Arg1& ai1, Arg2& br1, Arg3& bi1, Arg4& mBase, Arg5& nBase, Arg6& kBase)
    {
        using namespace AscendC;

        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        Nd2NzParams aParams;
        aParams.ndNum = 1;
        aParams.nValue = baseMN;
        aParams.dValue = kCgemmSbCubeBaseK;
        aParams.srcNdMatrixStride = 0;
        aParams.srcDValue = k_;
        aParams.dstNzC0Stride = baseMN;
        aParams.dstNzNStride = 1;
        aParams.dstNzMatrixStride = 0;
        const uint64_t aOffset = static_cast<uint64_t>(mBase) * k_ + kBase;
        DataCopy(ar1, ar_[aOffset], aParams);
        DataCopy(ai1, ai_[aOffset], aParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        Nd2NzParams bParams;
        bParams.ndNum = 1;
        bParams.nValue = kCgemmSbCubeBaseK;
        bParams.dValue = baseMN;
        bParams.srcNdMatrixStride = 0;
        bParams.srcDValue = n_;
        bParams.dstNzC0Stride = kCgemmSbCubeBaseK;
        bParams.dstNzNStride = 1;
        bParams.dstNzMatrixStride = 0;
        const uint64_t bOffset = static_cast<uint64_t>(kBase) * n_ + nBase;
        DataCopy(br1, br_[bOffset], bParams);
        DataCopy(bi1, bi_[bOffset], bParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8>
    __aicore__ inline void SbLoadCubeOperands(
        Arg0& ar1, Arg1& ai1, Arg2& br1, Arg3& bi1, Arg4& ar2, Arg5& ai2, Arg6& br2, Arg7& bi2, Arg8& loadA)
    {
        using namespace AscendC;
        LoadData(ar2, ar1, loadA);
        LoadData(ai2, ai1, loadA);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        LoadData3DParamsV2<float> loadB;
        cgemm_sb::SbInitLoadParams(loadB, kCgemmSbCubeBaseK, baseMN, baseMN, kCgemmSbCubeBaseK, true);
        LoadData(br2, br1, loadB);
        LoadData(bi2, bi1, loadB);
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12, typename Arg13,
        typename Arg14>
    __aicore__ inline void SbAccumulateCubeTile(
        Arg0& ar1, Arg1& ai1, Arg2& br1, Arg3& bi1, Arg4& ar2, Arg5& ai2, Arg6& br2, Arg7& bi2, Arg8& rr1, Arg9& ii1,
        Arg10& ri1, Arg11& ir1, Arg12& mBase, Arg13& nBase, Arg14& kBase)
    {
        using namespace AscendC;

        SbCopyCubeOperands(ar1, ai1, br1, bi1, mBase, nBase, kBase);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        LoadData3DParamsV2<float> loadA;
        cgemm_sb::SbInitLoadParams(loadA, baseMN, kCgemmSbCubeBaseK, kCgemmSbCubeBaseK, baseMN, false);
        SbLoadCubeOperands(ar1, ai1, br1, bi1, ar2, ai2, br2, bi2, loadA);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        MmadParams params = {};
        params.m = baseMN;
        params.n = baseMN;
        params.k = kCgemmSbCubeBaseK;
        params.cmatrixInitVal = kBase == 0U;
        params.cmatrixSource = false;
        params.kDirectionAlign = false;
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(rr1, ar2, br2, params);
        Mmad(ii1, ai2, bi2, params);
        Mmad(ri1, ar2, bi2, params);
        Mmad(ir1, ai2, br2, params);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12, typename Arg13>
    __aicore__ inline void SbComputeCubeTile(
        Arg0& ar1, Arg1& ai1, Arg2& br1, Arg3& bi1, Arg4& ar2, Arg5& ai2, Arg6& br2, Arg7& bi2, Arg8& rr1, Arg9& ii1,
        Arg10& ri1, Arg11& ir1, Arg12& nTiles, Arg13& tile)
    {
        using namespace AscendC;

        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        const uint32_t mBase = static_cast<uint32_t>((tile / nTiles) * baseMN);
        const uint32_t nBase = static_cast<uint32_t>((tile % nTiles) * baseMN);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        for (uint32_t kBase = 0U; kBase < k_; kBase += kCgemmSbCubeBaseK) {
            SbAccumulateCubeTile(ar1, ai1, br1, bi1, ar2, ai2, br2, bi2, rr1, ii1, ri1, ir1, mBase, nBase, kBase);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 outParams = {};
        outParams.mSize = baseMN;
        outParams.nSize = baseMN;
        outParams.srcStride = baseMN;
        outParams.dstStride = n_;
        outParams.ndNum = 1;
        outParams.srcNdStride = 0;
        outParams.dstNdStride = 0;
        const uint64_t outOffset = static_cast<uint64_t>(mBase) * n_ + nBase;
        Fixpipe(rr_[outOffset], rr1, outParams);
        Fixpipe(ii_[outOffset], ii1, outParams);
        Fixpipe(ri_[outOffset], ri1, outParams);
        Fixpipe(ir_[outOffset], ir1, outParams);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void Process(uint32_t tileIndex, uint32_t tileStride)
    {
        constexpr uint32_t aSize = baseMN * kCgemmSbCubeBaseK;
        constexpr uint32_t bSize = kCgemmSbCubeBaseK * baseMN;
        constexpr uint32_t cSize = baseMN * baseMN;
        LocalTensor<float> ar1(TPosition::A1, 0U, aSize);
        LocalTensor<float> ai1(TPosition::A1, aSize * sizeof(float), aSize);
        LocalTensor<float> br1(TPosition::B1, 2U * aSize * sizeof(float), bSize);
        LocalTensor<float> bi1(TPosition::B1, (2U * aSize + bSize) * sizeof(float), bSize);
        LocalTensor<float> ar2(TPosition::A2, 0U, aSize);
        LocalTensor<float> ai2(TPosition::A2, aSize * sizeof(float), aSize);
        LocalTensor<float> br2(TPosition::B2, 0U, bSize);
        LocalTensor<float> bi2(TPosition::B2, bSize * sizeof(float), bSize);
        LocalTensor<float> rr1(TPosition::CO1, 0U, cSize);
        LocalTensor<float> ii1(TPosition::CO1, cSize * sizeof(float), cSize);
        LocalTensor<float> ri1(TPosition::CO1, 2U * cSize * sizeof(float), cSize);
        LocalTensor<float> ir1(TPosition::CO1, 3U * cSize * sizeof(float), cSize);
        const uint32_t nTiles = n_ / baseMN;
        const uint64_t tileCount = static_cast<uint64_t>(m_ / baseMN) * nTiles;
        for (uint64_t tile = tileIndex; tile < tileCount; tile += tileStride) {
            SbComputeCubeTile(ar1, ai1, br1, bi1, ar2, ai2, br2, bi2, rr1, ii1, ri1, ir1, nTiles, tile);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

private:
    GlobalTensor<float> ar_;
    GlobalTensor<float> ai_;
    GlobalTensor<float> br_;
    GlobalTensor<float> bi_;
    GlobalTensor<float> rr_;
    GlobalTensor<float> ii_;
    GlobalTensor<float> ri_;
    GlobalTensor<float> ir_;
    uint32_t m_;
    uint32_t n_;
    uint32_t k_;
};

template <uint32_t tileK, uint32_t baseMN = 64U>
__aicore__ inline void SbComputeTiles(__gm__ float* base, const CgemmSbTiling& t, uint32_t batch)
{
    const uint64_t tiles = static_cast<uint64_t>(t.paddedM / baseMN) * (t.paddedN / baseMN);
    const uint32_t first =
        static_cast<uint32_t>((GetBlockIdx() + GetBlockNum() - (batch * tiles) % GetBlockNum()) % GetBlockNum());
    if (first >= tiles) {
        return;
    }
    CgemmSbBasicMmad<tileK, baseMN> mm;
    mm.PrepareCo1();
    mm.Init(
        base + SbAr(t), base + SbAi(t), base + SbBr(t), base + SbBi(t), base + SbRr(t), base + SbIi(t), base + SbRi(t),
        base + SbIr(t), t.paddedM, t.paddedN, t.paddedK);
    mm.Process(first, static_cast<uint32_t>(GetBlockNum()));
}

extern "C" __attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_cube_kernel(
    GM_ADDR workspace, const CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    __gm__ float* base = reinterpret_cast<__gm__ float*>(workspace);
    // K=128 fills the 64 KiB L0A/L0B banks with two complex planes and
    // amortizes transfer/barrier setup over four times more arithmetic.
    const uint32_t mn = t.paddedM == 32U && t.paddedN == 32U ? 32U : 64U;
    const uint64_t tiles = static_cast<uint64_t>(t.paddedM / mn) * (t.paddedN / mn);
    for (uint64_t global = GetBlockIdx(); global < tiles * t.batchCount;) {
        const uint32_t batch = static_cast<uint32_t>(global / tiles);
        auto* batchBase = base + batch * t.workspaceStride;
        if (mn == 32U) {
            SbComputeTiles<32U, 32U>(batchBase, t, batch);
        } else if (t.paddedK % 128U == 0U) {
            SbComputeTiles<128U>(batchBase, t, batch);
        } else {
            SbComputeTiles<64U>(batchBase, t, batch);
        }
        global = (batch + 1U) * tiles;
        global += (GetBlockIdx() + GetBlockNum() - global % GetBlockNum()) % GetBlockNum();
    }
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(base);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
    PipeBarrier<PIPE_ALL>();
}

template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12, typename Arg13,
    typename Arg14, typename Arg15, typename Arg16, typename Arg17, typename Arg18, typename Arg19, typename Arg20>
__aicore__ inline void SbWriteFastTile(
    Arg0& t, Arg1& s, Arg2& rrQ, Arg3& iiQ, Arg4& riQ, Arg5& irQ, Arg6& outQ, Arg7& offsets, Arg8& concat, Arg9& cg,
    Arg10& ws, Arg11& pad, Arg12& inCopy, Arg13& outCopy, Arg14& col, Arg15& row, Arg16& work, Arg17& rr, Arg18& ii,
    Arg19& ri, Arg20& ir)
{
    using namespace AscendC;

    DataCopyPad(ii, ws[SbIi(s) + work], inCopy, pad);
    DataCopyPad(ri, ws[SbRi(s) + work], inCopy, pad);
    DataCopyPad(ir, ws[SbIr(s) + work], inCopy, pad);
    rrQ.EnQue(rr);
    iiQ.EnQue(ii);
    riQ.EnQue(ri);
    irQ.EnQue(ir);
    rr = rrQ.template DeQue<float>();
    ii = iiQ.template DeQue<float>();
    ri = riQ.template DeQue<float>();
    ir = irQ.template DeQue<float>();
    Sub(rr, rr, ii, kSbVectorTile);
    Add(ri, ri, ir, kSbVectorTile);
    PipeBarrier<PIPE_ALL>();
    Adds(concat, rr, 0.0f, kSbVectorTile);
    Adds(concat[kSbVectorTile], ri, 0.0f, kSbVectorTile);
    PipeBarrier<PIPE_ALL>();
    auto out = outQ.template AllocTensor<float>();
    Gather(out, concat, offsets, 0U, 2U * kSbVectorTile);
    PipeBarrier<PIPE_ALL>();
    rrQ.FreeTensor(rr);
    iiQ.FreeTensor(ii);
    riQ.FreeTensor(ri);
    irQ.FreeTensor(ir);
    outQ.EnQue(out);
    out = outQ.template DeQue<float>();
    DataCopyPad(cg[(static_cast<uint64_t>(col) * t.ldc + row) * 2U], out, outCopy);
    outQ.FreeTensor(out);
}

template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11>
__aicore__ inline void SbRunFastEpilogue(
    Arg0& c, Arg1& workspace, Arg2& t, Arg3& s, Arg4& rrQ, Arg5& iiQ, Arg6& riQ, Arg7& irQ, Arg8& outQ, Arg9& offsetBuf,
    Arg10& concatBuf, Arg11& offsetsI32)
{
    using namespace AscendC;

    PipeBarrier<PIPE_ALL>();
    for (uint32_t group = 1U; group < kSbVectorTile / 8U; ++group) {
        Adds(offsetsI32[group * 16U], offsetsI32, static_cast<int32_t>(group * 8U * sizeof(float)), 16U);
    }
    PipeBarrier<PIPE_ALL>();
    auto offsets = offsetBuf.template Get<uint32_t>();
    auto concat = concatBuf.template Get<float>();
    GlobalTensor<float> cg, ws;
    for (uint32_t batch = 0; batch < t.batchCount; ++batch) {
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c) + static_cast<int64_t>(batch) * t.strideC * 2);
        ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + batch * t.workspaceStride);
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyExtParams inCopy{1, kSbVectorTile * sizeof(float), 0, 0, 0};
        DataCopyExtParams outCopy{1, kSbVectorTile * 2U * sizeof(float), 0, 0, 0};
        const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t blocks = static_cast<uint32_t>(GetBlockNum());
        for (uint32_t col = block; col < t.n; col += blocks) {
            for (uint32_t row = 0; row < t.m; row += kSbVectorTile) {
                const uint64_t work = static_cast<uint64_t>(col) * t.m + row;
                auto rr = rrQ.template AllocTensor<float>();
                auto ii = iiQ.template AllocTensor<float>();
                auto ri = riQ.template AllocTensor<float>();
                auto ir = irQ.template AllocTensor<float>();
                DataCopyPad(rr, ws[SbRr(s) + work], inCopy, pad);
                SbWriteFastTile(
                    t, s, rrQ, iiQ, riQ, irQ, outQ, offsets, concat, cg, ws, pad, inCopy, outCopy, col, row, work, rr,
                    ii, ri, ir);
            }
        }
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(cg);
}

extern "C" __attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_epilogue_fast_kernel(
    GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling t, const CgemmSbTiling s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TQue<TPosition::VECIN, 2> rrQ, iiQ, riQ, irQ;
    TQue<TPosition::VECOUT, 2> outQ;
    TBuf<TPosition::VECCALC> offsetBuf, concatBuf;
    pipe.InitBuffer(rrQ, 2, kSbVectorTile * sizeof(float));
    pipe.InitBuffer(iiQ, 2, kSbVectorTile * sizeof(float));
    pipe.InitBuffer(riQ, 2, kSbVectorTile * sizeof(float));
    pipe.InitBuffer(irQ, 2, kSbVectorTile * sizeof(float));
    pipe.InitBuffer(outQ, 2, kSbVectorTile * 2U * sizeof(float));
    pipe.InitBuffer(offsetBuf, 2U * kSbVectorTile * sizeof(uint32_t));
    pipe.InitBuffer(concatBuf, 2U * kSbVectorTile * sizeof(float));
    // Seed one vector block; generate the remaining byte offsets with
    // vector Adds instead of 512 scalar UB stores on every launch.
    auto offsetsI32 = offsetBuf.Get<int32_t>();
    cgemm_sb::SbInterleavedSeed(offsetsI32, kSbVectorTile);
    SbRunFastEpilogue(c, workspace, t, s, rrQ, iiQ, riQ, irQ, outQ, offsetBuf, concatBuf, offsetsI32);
    PipeBarrier<PIPE_ALL>();
}

// Compact planes are partitioned globally across batches and copied in large runs.
template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12>
__aicore__ inline void SbPackCompactOperand(
    Arg0& a, Arg1& b, Arg2& workspace, Arg3& t, Arg4& s, Arg5& inputQ, Arg6& realQ, Arg7& imagQ, Arg8& offsets,
    Arg9& source, Arg10& ws, Arg11& operand, Arg12& count)
{
    using namespace AscendC;
    constexpr uint32_t cap = 8192U;
    const int64_t stride = operand == 0U ? t.strideB : t.strideA;
    const uint64_t realOffset = operand == 0U ? SbAr(s) : SbBr(s);
    const uint64_t imagOffset = operand == 0U ? SbAi(s) : SbBi(s);
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(operand == 0U ? b : a));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    const uint64_t tiles = (count + cap - 1U) / cap;
    for (uint64_t global = GetBlockIdx(); global < tiles * t.batchCount; global += GetBlockNum()) {
        const uint32_t batch = static_cast<uint32_t>(global / tiles);
        const uint64_t offset = (global % tiles) * cap;
        const uint32_t size = count - offset < cap ? static_cast<uint32_t>(count - offset) : cap;
        auto input = inputQ.template AllocTensor<float>();
        DataCopy(input, source[(static_cast<int64_t>(batch) * stride + offset) * 2U], size * 2U);
        inputQ.EnQue(input);
        input = inputQ.template DeQue<float>();
        auto real = realQ.template AllocTensor<float>();
        auto imag = imagQ.template AllocTensor<float>();
        Gather(real, input, offsets, 0U, size);
        Gather(imag, input, offsets, 4U, size);
        inputQ.FreeTensor(input);
        realQ.EnQue(real);
        imagQ.EnQue(imag);
        real = realQ.template DeQue<float>();
        imag = imagQ.template DeQue<float>();
        DataCopy(ws[batch * t.workspaceStride + realOffset + offset], real, size);
        DataCopy(ws[batch * t.workspaceStride + imagOffset + offset], imag, size);
        realQ.FreeTensor(real);
        imagQ.FreeTensor(imag);
    }
}

extern "C" __attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_pack_nn_compact_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const CgemmSbTiling t, const CgemmSbTiling s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t cap = 8192U;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inputQ;
    TQue<TPosition::VECOUT, 1> realQ, imagQ;
    TBuf<TPosition::VECCALC> indexBuf;
    pipe.InitBuffer(inputQ, 1, cap * 2U * sizeof(float));
    pipe.InitBuffer(realQ, 1, cap * sizeof(float));
    pipe.InitBuffer(imagQ, 1, cap * sizeof(float));
    pipe.InitBuffer(indexBuf, cap * sizeof(uint32_t));
    auto ix = indexBuf.Get<int32_t>();
    CreateVecIndex(ix, 0, cap);
    Muls(ix, ix, 8, cap);
    PipeBarrier<PIPE_ALL>();
    auto offsets = indexBuf.Get<uint32_t>();
    GlobalTensor<float> source, ws;
    for (uint32_t operand = 0; operand < 2U; ++operand) {
        const uint64_t count = operand == 0U ? static_cast<uint64_t>(t.k) * t.n : static_cast<uint64_t>(t.m) * t.k;
        SbPackCompactOperand(a, b, workspace, t, s, inputQ, realQ, imagQ, offsets, source, ws, operand, count);
    }
}

template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__aicore__ inline void SbWriteCompactTile(
    Arg0& t, Arg1& s, Arg2& sumQ, Arg3& otherQ, Arg4& outQ, Arg5& offsets, Arg6& cg, Arg7& ws, Arg8& count, Arg9& tiles,
    Arg10& global)
{
    using namespace AscendC;
    constexpr uint32_t cap = 4096U;
    const uint32_t batch = static_cast<uint32_t>(global / tiles);
    const uint64_t offset = (global % tiles) * cap;
    const uint32_t size = count - offset < cap ? static_cast<uint32_t>(count - offset) : cap;
    const uint64_t base = batch * t.workspaceStride + offset;
    auto sum = sumQ.template AllocTensor<float>();
    auto other = otherQ.template AllocTensor<float>();
    DataCopy(sum, ws[base + SbRr(s)], size);
    DataCopy(sum[cap], ws[base + SbRi(s)], size);
    DataCopy(other, ws[base + SbIi(s)], size);
    DataCopy(other[cap], ws[base + SbIr(s)], size);
    sumQ.EnQue(sum);
    otherQ.EnQue(other);
    sum = sumQ.template DeQue<float>();
    other = otherQ.template DeQue<float>();
    Sub(sum, sum, other, size);
    Add(sum[cap], sum[cap], other[cap], size);
    auto out = outQ.template AllocTensor<float>();
    Gather(out, sum, offsets, 0U, size * 2U);
    sumQ.FreeTensor(sum);
    otherQ.FreeTensor(other);
    outQ.EnQue(out);
    out = outQ.template DeQue<float>();
    DataCopy(cg[(static_cast<int64_t>(batch) * t.strideC + offset) * 2U], out, size * 2U);
    outQ.FreeTensor(out);
}

extern "C" __attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void
cgemm_sb_epilogue_compact_kernel(GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling t, const CgemmSbTiling s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t cap = 4096U;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> sumQ, otherQ;
    TQue<TPosition::VECOUT, 1> outQ;
    TBuf<TPosition::VECCALC> indexBuf;
    pipe.InitBuffer(sumQ, 1, cap * 2U * sizeof(float));
    pipe.InitBuffer(otherQ, 1, cap * 2U * sizeof(float));
    pipe.InitBuffer(outQ, 1, cap * 2U * sizeof(float));
    pipe.InitBuffer(indexBuf, cap * 2U * sizeof(uint32_t));
    auto ix = indexBuf.Get<int32_t>();
    cgemm_sb::SbInterleavedSeed(ix, cap);
    for (uint32_t group = 1U; group < cap / 8U; group *= 2U) {
        Adds(ix[group * 16U], ix, static_cast<int32_t>(group * 32U), group * 16U);
    }
    PipeBarrier<PIPE_ALL>();
    auto offsets = indexBuf.Get<uint32_t>();
    GlobalTensor<float> cg, ws;
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    const uint64_t count = static_cast<uint64_t>(t.m) * t.n;
    const uint64_t tiles = (count + cap - 1U) / cap;
    for (uint64_t global = GetBlockIdx(); global < tiles * t.batchCount; global += GetBlockNum()) {
        SbWriteCompactTile(t, s, sumQ, otherQ, outQ, offsets, cg, ws, count, tiles, global);
    }
}

// Keep transposed-pack entry placement stable when other kernel families move.
constexpr uint32_t kCgemmSbPackKernelAlignment = 4096U;

template <uint32_t tileSize>
__attribute__((aligned(kCgemmSbPackKernelAlignment))) __global__ __aicore__ void cgemm_sb_pack_transposed_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const CgemmSbTiling t, const CgemmSbTiling s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    cgemm_sb::GeneralPack<tileSize> pack;
    pack.Init(pipe);
    GlobalTensor<float> source, ws;
    for (uint32_t operand = 0; operand < 2U; ++operand) {
        const uint32_t rows = operand == 0U ? t.n : t.k;
        const uint32_t cols = operand == 0U ? t.k : t.m;
        const uint32_t paddedRows = operand == 0U ? s.paddedM : s.paddedK;
        const uint32_t paddedCols = operand == 0U ? s.paddedK : s.paddedN;
        const uint32_t ld = operand == 0U ? t.ldb : t.lda;
        const uint32_t trans = operand == 0U ? t.transB : t.transA;
        const int64_t stride = operand == 0U ? t.strideB : t.strideA;
        const uint64_t realOffset = operand == 0U ? SbAr(s) : SbBr(s);
        const uint64_t imagOffset = operand == 0U ? SbAi(s) : SbBi(s);
        const uint64_t tiles =
            static_cast<uint64_t>((rows + tileSize - 1U) / tileSize) * ((cols + tileSize - 1U) / tileSize);
        for (uint64_t global = GetBlockIdx(); global < tiles * t.batchCount;) {
            const uint32_t batch = static_cast<uint32_t>(global / tiles);
            source.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(operand == 0U ? b : a) + static_cast<int64_t>(batch) * stride * 2);
            ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + batch * t.workspaceStride);
            pack.Process(
                source, ws, realOffset, imagOffset, rows, cols, paddedRows, paddedCols, ld, trans == ACLBLAS_OP_N,
                trans == ACLBLAS_OP_C, batch);
            global = (batch + 1U) * tiles;
            global += (GetBlockIdx() + GetBlockNum() - global % GetBlockNum()) % GetBlockNum();
        }
    }
}

template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__aicore__ inline void SbApplyColumnBeta(
    Arg0& t, Arg1& sumQ, Arg2& swapBuf, Arg3& offsets, Arg4& cg, Arg5& sum, Arg6& other, Arg7& pad, Arg8& dst,
    Arg9& outputCount, Arg10& out)
{
    using namespace AscendC;

    Gather(out, sum, offsets, 0U, outputCount);
    if (t.betaReal != 0.0f || t.betaImag != 0.0f) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams cp{1U, outputCount * 4U, 0U, 0U, 0U};
        DataCopyPad(other, cg[dst * 2U], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (t.betaImag == 0.0f) {
            if (t.betaReal != 1.0f) {
                Muls(other, other, t.betaReal, outputCount);
            }
        } else {
            auto swapIndex = swapBuf.template Get<uint32_t>();
            Gather(sum, other, swapIndex, 0U, outputCount);
            Muls(sum, sum, t.betaImag, outputCount);
            uint64_t evenMask[2] = {0x5555555555555555ULL, 0U};
            Muls(sum, sum, -1.0f, evenMask, static_cast<uint8_t>((outputCount + 63U) / 64U), {1, 1, 8, 8});
            if (t.betaReal == 0.0f) {
                Adds(other, sum, 0.0f, outputCount);
            } else {
                if (t.betaReal != 1.0f) {
                    Muls(other, other, t.betaReal, outputCount);
                }
                Add(other, other, sum, outputCount);
            }
        }
        Add(out, out, other, outputCount);
    }
    sumQ.FreeTensor(sum);
}

template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11>
__aicore__ inline void SbApplyColumnAlpha(
    Arg0& t, Arg1& s, Arg2& sumQ, Arg3& otherQ, Arg4& ws, Arg5& cols, Arg6& alignedRows, Arg7& base, Arg8& sum,
    Arg9& other, Arg10& ic, Arg11& pad)
{
    using namespace AscendC;
    constexpr uint32_t cap = 4096U;
    DataCopyPad(sum[cap], ws[base + SbRi(s)], ic, pad);
    DataCopyPad(other, ws[base + SbIi(s)], ic, pad);
    DataCopyPad(other[cap], ws[base + SbIr(s)], ic, pad);
    sumQ.EnQue(sum);
    otherQ.EnQue(other);
    sum = sumQ.template DeQue<float>();
    other = otherQ.template DeQue<float>();
    Sub(sum, sum, other, cols * alignedRows);
    Add(sum[cap], sum[cap], other[cap], cols * alignedRows);
    const uint32_t count = cols * alignedRows;
    if (t.alphaImag == 0.0f) {
        if (t.alphaReal != 1.0f) {
            Muls(sum, sum, t.alphaReal, count);
            Muls(sum[cap], sum[cap], t.alphaReal, count);
        }
    } else if (t.alphaReal == 0.0f) {
        Muls(other, sum, t.alphaImag, count);
        Muls(sum, sum[cap], -t.alphaImag, count);
        Adds(sum[cap], other, 0.0f, count);
    } else {
        Muls(other, sum, t.alphaReal, count);
        Muls(other[cap], sum[cap], -t.alphaImag, count);
        Add(other, other, other[cap], count);
        Muls(other[cap], sum[cap], t.alphaReal, count);
        Muls(sum, sum, t.alphaImag, count);
        Add(sum[cap], sum, other[cap], count);
        Adds(sum, other, 0.0f, count);
    }
}

template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12, typename Arg13,
    typename Arg14, typename Arg15>
__aicore__ inline void SbWriteColumnSpan(
    Arg0& t, Arg1& s, Arg2& sumQ, Arg3& otherQ, Arg4& outQ, Arg5& swapBuf, Arg6& localRows, Arg7& offsets, Arg8& cg,
    Arg9& ws, Arg10& position, Arg11& end, Arg12& batch, Arg13& col, Arg14& row, Arg15& rows)
{
    using namespace AscendC;
    constexpr uint32_t cap = 4096U;
    uint32_t cols = 1U;
    if (row == 0U && rows == t.m && t.ldc == t.m) {
        cols = cap / localRows;
        if (cols > t.n - col) {
            cols = t.n - col;
        }
        if (cols > (end - position) / t.m) {
            cols = static_cast<uint32_t>((end - position) / t.m);
        }
    }
    const uint32_t alignedRows = (rows + 7U) / 8U * 8U;
    const uint64_t base = batch * t.workspaceStride + static_cast<uint64_t>(col) * s.paddedN + row;
    auto sum = sumQ.template AllocTensor<float>();
    auto other = otherQ.template AllocTensor<float>();
    DataCopyExtParams ic{static_cast<uint16_t>(cols), rows * 4U, (s.paddedN - rows) * 4U, 0U, 0U};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(sum, ws[base + SbRr(s)], ic, pad);
    SbApplyColumnAlpha(t, s, sumQ, otherQ, ws, cols, alignedRows, base, sum, other, ic, pad);
    const uint64_t dst = static_cast<int64_t>(batch) * t.strideC + static_cast<uint64_t>(col) * t.ldc + row;
    const uint32_t outputCount = cols * rows * 2U;
    auto out = outQ.template AllocTensor<float>();
    SbApplyColumnBeta(t, sumQ, swapBuf, offsets, cg, sum, other, pad, dst, outputCount, out);
    otherQ.FreeTensor(other);
    outQ.EnQue(out);
    out = outQ.template DeQue<float>();
    DataCopyExtParams oc{1U, cols * rows * 8U, 0U, 0U, 0U};
    DataCopyPad(cg[dst * 2U], out, oc);
    outQ.FreeTensor(out);
    position += static_cast<uint64_t>(cols) * rows;
}

template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__aicore__ inline void SbRunColumnTiles(
    Arg0& c, Arg1& workspace, Arg2& t, Arg3& s, Arg4& alignmentBias, Arg5& sumQ, Arg6& otherQ, Arg7& outQ,
    Arg8& swapBuf, Arg9& localRows, Arg10& offsets)
{
    using namespace AscendC;
    constexpr uint32_t cap = 4096U;
    GlobalTensor<float> cg, ws;
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    const bool compact = t.strideC == static_cast<int64_t>(t.ldc) * t.n;
    const uint32_t span = compact ? t.n : (t.n + 7U) / 8U * 8U;
    const uint64_t batchElements = static_cast<uint64_t>(span) * t.m;
    const uint64_t totalElements = batchElements * t.batchCount;
    const uint64_t tileElements = static_cast<uint64_t>(8U) * t.m;
    for (uint64_t tile = GetBlockIdx(); tile * tileElements < totalElements + alignmentBias; tile += GetBlockNum()) {
        const uint64_t rawBegin = tile * tileElements;
        const uint64_t rawEnd = (tile + 1U) * tileElements;
        uint64_t position = rawBegin < alignmentBias ? 0U : rawBegin - alignmentBias;
        const uint64_t end = rawEnd - alignmentBias < totalElements ? rawEnd - alignmentBias : totalElements;
        while (position < end) {
            const uint32_t batch = static_cast<uint32_t>(position / batchElements);
            const uint64_t inBatch = position % batchElements;
            const uint32_t col = static_cast<uint32_t>(inBatch / t.m);
            const uint32_t row = static_cast<uint32_t>(inBatch % t.m);
            if (col >= t.n) {
                position += batchElements - inBatch;
                continue;
            }
            uint32_t rows = t.m - row < cap ? t.m - row : cap;
            if (rows > end - position) {
                rows = static_cast<uint32_t>(end - position);
            }
            SbWriteColumnSpan(
                t, s, sumQ, otherQ, outQ, swapBuf, localRows, offsets, cg, ws, position, end, batch, col, row, rows);
        }
    }
}

template <typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
__aicore__ inline void SbBuildColumnIndices(
    Arg0& sumQ, Arg1& otherQ, Arg2& outQ, Arg3& ix, Arg4& logicalRows, Arg5& localRows)
{
    using namespace AscendC;
    constexpr uint32_t cap = 4096U;
    auto u = sumQ.template AllocTensor<float>();
    auto v = otherQ.template AllocTensor<float>();
    auto w = outQ.template AllocTensor<float>();
    CreateVecIndex(w, 0.0f, 2U * cap);
    Muls(u, w, 0.5f, 2U * cap);
    Cast(ix, u, RoundMode::CAST_FLOOR, 2U * cap);
    Cast(u, ix, RoundMode::CAST_NONE, 2U * cap);
    Muls(v, u, -2.0f, 2U * cap);
    Add(v, v, w, 2U * cap);
    Muls(v, v, 16384.0f, 2U * cap);
    Muls(w, u, 1.0f / static_cast<float>(static_cast<int32_t>(logicalRows)), 2U * cap);
    Cast(ix, w, RoundMode::CAST_FLOOR, 2U * cap);
    Cast(w, ix, RoundMode::CAST_NONE, 2U * cap);
    Muls(w, w, static_cast<float>(static_cast<int32_t>((localRows - logicalRows) * 4U)), 2U * cap);
    Muls(u, u, 4.0f, 2U * cap);
    Add(u, u, w, 2U * cap);
    Add(u, u, v, 2U * cap);
    Cast(ix, u, RoundMode::CAST_RINT, 2U * cap);
    PipeBarrier<PIPE_ALL>();
    sumQ.FreeTensor(u);
    otherQ.FreeTensor(v);
    outQ.FreeTensor(w);
}

extern "C" __attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void
cgemm_sb_epilogue_columns_kernel(
    GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling t, const CgemmSbTiling s, uint32_t alignmentBias)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t cap = 4096U;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> sumQ, otherQ;
    TQue<TPosition::VECOUT, 1> outQ;
    TBuf<TPosition::VECCALC> indexBuf, swapBuf;
    pipe.InitBuffer(sumQ, 1, cap * 2U * sizeof(float));
    pipe.InitBuffer(otherQ, 1, cap * 2U * sizeof(float));
    pipe.InitBuffer(outQ, 1, cap * 2U * sizeof(float));
    pipe.InitBuffer(indexBuf, cap * 2U * sizeof(uint32_t));
    auto ix = indexBuf.Get<int32_t>();
    const uint32_t logicalRows = t.m < cap ? t.m : cap;
    const uint32_t localRows = (logicalRows + 7U) / 8U * 8U;
    {
        SbBuildColumnIndices(sumQ, otherQ, outQ, ix, logicalRows, localRows);
    }
    if (t.betaImag != 0.0f) {
        pipe.InitBuffer(swapBuf, cap * 2U * sizeof(uint32_t));
        auto swapIndex = swapBuf.Get<int32_t>();
        CreateVecIndex(swapIndex, 0, 16U);
        Muls(swapIndex, swapIndex, static_cast<int32_t>(sizeof(float)), 16U);
        Adds(swapIndex, swapIndex, static_cast<int32_t>(sizeof(float)), 16U);
        // Odd lanes subtract two floats to exchange each adjacent real/imag pair.
        uint64_t oddLanes[2] = {0xaaaaULL, 0U};
        Adds(swapIndex, swapIndex, -static_cast<int32_t>(2U * sizeof(float)), oddLanes, 1U, {1, 1, 8, 8});
        for (uint32_t length = 16U; length < 2U * cap; length *= 2U) {
            Adds(swapIndex[length], swapIndex, static_cast<int32_t>(length * 4U), length);
        }
        PipeBarrier<PIPE_ALL>();
    }
    auto offsets = indexBuf.Get<uint32_t>();
    SbRunColumnTiles(c, workspace, t, s, alignmentBias, sumQ, otherQ, outQ, swapBuf, localRows, offsets);
}

bool cgemm_sb::CgemmSbDispatcher::LaunchSmall()
{
    if (!(!(tiling.m == 16U && tiling.batchCount >= 16U) && CgemmSbSmallShape(tiling) &&
          reinterpret_cast<uintptr_t>(c) % 64U == 0U)) {
        return false;
    }

    if (tiling.m == 8U) {
        cgemm_sb_small_kernel<8U><<<aivCores, nullptr, stream>>>(a, b, c, tiling);
    } else if (tiling.m == 16U) {
        cgemm_sb_small_kernel<16U><<<aivCores, nullptr, stream>>>(a, b, c, tiling);
    } else if (tiling.m == 4U) {
        cgemm_sb_small_kernel<4U><<<aivCores, nullptr, stream>>>(a, b, c, tiling);
    } else {
        cgemm_sb_small_kernel<12U><<<aivCores, nullptr, stream>>>(a, b, c, tiling);
    }
    return true;
}

void cgemm_sb::CgemmSbDispatcher::LaunchFast()
{
    CgemmSbTiling s = tiling;
    s.m = tiling.n;
    s.n = tiling.m;
    s.paddedM = tiling.paddedN;
    s.paddedN = tiling.paddedM;
    if (tiling.transA == ACLBLAS_OP_N && tiling.transB == ACLBLAS_OP_N && tiling.lda == tiling.m &&
        tiling.ldb == tiling.k) {
        cgemm_sb_pack_nn_compact_kernel<<<aivCores, nullptr, stream>>>(a, b, workspace, tiling, s);
    } else {
        cgemm_sb_pack_fast_kernel<<<aivCores, nullptr, stream>>>(a, b, workspace, tiling, s);
    }
    const uint64_t tiles = static_cast<uint64_t>(s.paddedM / 64U) * (s.paddedN / 64U);
    const uint64_t totalTiles = tiles * tiling.batchCount;
    const uint32_t cubeBlocks = totalTiles < aicCores ? static_cast<uint32_t>(totalTiles) : aicCores;
    cgemm_sb_cube_kernel<<<cubeBlocks, nullptr, stream>>>(workspace, s);
    if (tiling.ldc == tiling.m) {
        cgemm_sb_epilogue_compact_kernel<<<aivCores, nullptr, stream>>>(c, workspace, tiling, s);
    } else {
        cgemm_sb_epilogue_fast_kernel<<<aivCores, nullptr, stream>>>(c, workspace, tiling, s);
    }
}

void cgemm_sb::CgemmSbDispatcher::LaunchGeneral()
{
    CgemmSbTiling s = tiling;
    s.m = tiling.n;
    s.n = tiling.m;
    s.paddedM = tiling.paddedN;
    s.paddedN = tiling.paddedM;
    const uint32_t mn = s.paddedM == 32U && s.paddedN == 32U ? 32U : 64U;
    if (mn == 32U) {
        cgemm_sb_pack_transposed_kernel<32U><<<aivCores, nullptr, stream>>>(a, b, workspace, tiling, s);
    } else {
        cgemm_sb_pack_transposed_kernel<64U><<<aivCores, nullptr, stream>>>(a, b, workspace, tiling, s);
    }
    const uint64_t tiles = static_cast<uint64_t>(s.paddedM / mn) * (s.paddedN / mn) * tiling.batchCount;
    const uint32_t blocks = tiles < aicCores ? static_cast<uint32_t>(tiles) : aicCores;
    cgemm_sb_cube_kernel<<<blocks, nullptr, stream>>>(workspace, s);
    const bool dense = tiling.ldc == tiling.m && tiling.strideC == static_cast<int64_t>(tiling.m) * tiling.n &&
                       reinterpret_cast<uintptr_t>(c) % 8U == 0U;
    const uint32_t outputBlocks =
        dense || (reinterpret_cast<uintptr_t>(c) % 64U == 0U && tiling.strideC % 8 == 0) ? aivCores : 1U;
    const uint32_t bias = dense ? static_cast<uint32_t>((reinterpret_cast<uintptr_t>(c) % 64U) / 8U) : 0U;
    cgemm_sb_epilogue_columns_kernel<<<outputBlocks, nullptr, stream>>>(c, workspace, tiling, s, bias);
}

void cgemm_sb_launch(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling& tiling, uint32_t aivCores,
    uint32_t aicCores, bool useCube, bool parallelC, void* stream)
{
    cgemm_sb::CgemmSbDispatcher dispatch{
        a,
        b,
        c,
        workspace,
        tiling,
        aivCores,
        aicCores,
        stream,
        {tiling.batchCount, static_cast<uint16_t>(tiling.m), static_cast<uint8_t>(tiling.transA),
         static_cast<uint8_t>(tiling.transB)}};
    if (dispatch.LaunchSmall()) {
        return;
    }
    if (!useCube) {
        const bool independent = tiling.strideC >= static_cast<int64_t>(tiling.ldc) * tiling.n;
        const uint32_t blocks = parallelC && independent ? aivCores : 1U;
        cgemm_sb_direct_batched_launch(a, b, c, tiling, blocks, stream);
        return;
    }
    if (dispatch.LaunchDiagonal() || dispatch.LaunchSquare() || dispatch.LaunchAugmented()) {
        return;
    }
    if (parallelC && CgemmSbFastShape(tiling)) {
        dispatch.LaunchFast();
    } else {
        dispatch.LaunchGeneral();
    }
}
