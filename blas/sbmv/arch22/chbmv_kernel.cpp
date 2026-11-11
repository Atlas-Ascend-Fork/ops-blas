/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Band addressing/row ownership follows arch22 Ssbmv; complex planes follow arch22 Cgbmv.
#include <cstdint>
#include "kernel_operator.h"
#include "chbmv_kernel.h"
#define CHBMV_SOFT_INLINE __aicore__ inline
#include "finite_netlib_row.hpp"
#undef CHBMV_SOFT_INLINE

using namespace AscendC;

namespace {
template <HardEvent EVENT>
__aicore__ inline void Sync()
{
    SetFlag<EVENT>(EVENT_ID0);
    WaitFlag<EVENT>(EVENT_ID0);
}

__aicore__ inline int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline int64_t Max(int64_t a, int64_t b) { return a > b ? a : b; }

__aicore__ inline int64_t VectorPosition(int64_t i, uint32_t n, int64_t inc)
{
    return inc > 0 ? i * inc : (static_cast<int64_t>(n) - 1 - i) * (-inc);
}

// Numeric exceptional path uses actual values, never dimensions or test identities.
__aicore__ inline bool ChbmvScalarRisk(float value)
{
    const uint32_t bits = chbmv_numeric_range::Bits(value) & 0x7fffffffU;
    return bits != 0U && bits < 0x7f800000U && (bits > 0x47800000U || bits < 0x37800000U);
}

__aicore__ inline bool ChbmvScalarRisk(const ChbmvTilingData& t)
{
    return ChbmvScalarRisk(t.alphaR) || ChbmvScalarRisk(t.alphaI) || ChbmvScalarRisk(t.betaR) ||
           ChbmvScalarRisk(t.betaI);
}

struct ChbmvFiniteAccess {
    const GlobalTensor<float>& a;
    const GlobalTensor<float>& x;
    const GlobalTensor<float>& y;
    const ChbmvTilingData& t;

    __aicore__ inline chbmv_numeric_range::FloatComplex LoadA(int64_t index) const
    {
        return {a.GetValue(2 * index), a.GetValue(2 * index + 1)};
    }
    __aicore__ inline float LoadDiagonal(int64_t index) const { return a.GetValue(2 * index); }
    __aicore__ inline chbmv_numeric_range::FloatComplex LoadX(uint32_t row) const
    {
        const int64_t index = 2 * VectorPosition(row, t.n, t.incx);
        return {x.GetValue(index), x.GetValue(index + 1)};
    }
    __aicore__ inline chbmv_numeric_range::FloatComplex LoadY(uint32_t row) const
    {
        const int64_t index = 2 * VectorPosition(row, t.n, t.incy);
        return {y.GetValue(index), y.GetValue(index + 1)};
    }
};

__aicore__ inline bool ChbmvTryFiniteRow(
    const GlobalTensor<float>& a, const GlobalTensor<float>& x, const GlobalTensor<float>& y, const ChbmvTilingData& t,
    int64_t row, float& real, float& imag)
{
    const ChbmvFiniteAccess access{a, x, y, t};
    return chbmv_numeric_range::RecomputeFiniteNetlibRow(
        access, t.n, t.k, t.lda, t.upper != 0U, static_cast<uint32_t>(row), {t.alphaR, t.alphaI}, {t.betaR, t.betaI},
        real, imag);
}

__aicore__ inline void ChbmvCommitExact(
    const GlobalTensor<float>& y, int64_t firstPhysicalElement, const LocalTensor<float>& staging,
    uint32_t complexCount)
{
    Sync<HardEvent::S_MTE3>();
    DataCopyPad(y[2 * firstPhysicalElement], staging, DataCopyExtParams{1U, complexCount * 8U, 0U, 0U, 0U});
    Sync<HardEvent::MTE3_S>();
    Sync<HardEvent::MTE3_V>();
    Sync<HardEvent::MTE3_MTE2>();
}

__aicore__ inline void Load(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t offset, uint32_t count)
{
    DataCopyPad(
        dst, src[offset], DataCopyExtParams{1U, count * 8U, 0U, 0U, 0U},
        DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
}

__aicore__ inline void ComplexProducts(
    const LocalTensor<float>& ar, const LocalTensor<float>& ai, const LocalTensor<float>& xr,
    const LocalTensor<float>& xi, const LocalTensor<float>& rr, const LocalTensor<float>& ii,
    const LocalTensor<float>& ri, const LocalTensor<float>& ir, uint32_t count)
{
    Mul(rr, ar, xr, count);
    Mul(ii, ai, xi, count);
    Mul(ri, ar, xi, count);
    Mul(ir, ai, xr, count);
    PipeBarrier<PIPE_V>();
    Sub(rr, rr, ii, count);
    Add(ri, ri, ir, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Scale(
    const LocalTensor<float>& r, const LocalTensor<float>& i, float sr, float si, const LocalTensor<float>& p,
    const LocalTensor<float>& q, const LocalTensor<float>& u, const LocalTensor<float>& v, uint32_t count)
{
    Muls(p, r, sr, count);
    Muls(q, i, si, count);
    Muls(u, r, si, count);
    Muls(v, i, sr, count);
    PipeBarrier<PIPE_V>();
    Sub(r, p, q, count);
    Add(i, u, v, count);
    PipeBarrier<PIPE_V>();
}

class ChbmvDiagonals {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const ChbmvTilingData& t, TPipe& pipe)
    {
        t_ = t;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        pipe.InitBuffer(spanBuf_, CHBMV_DIAGONAL_SPAN_BYTES);
        pipe.InitBuffer(xBuf_, (CHBMV_DIAGONAL_ROWS + 2U * CHBMV_SMALL_BAND) * 8U);
        pipe.InitBuffer(planeBuf_, 10U * CHBMV_DIAGONAL_ROWS * 4U);
        pipe.InitBuffer(ioBuf_, 2U * CHBMV_DIAGONAL_ROWS * 4U);
        pipe.InitBuffer(indexBuf_, 6U * CHBMV_DIAGONAL_ROWS * 4U);
        span_ = spanBuf_.Get<float>();
        xWin_ = xBuf_.Get<float>();
        auto p = planeBuf_.Get<float>();
        accR_ = p;
        accI_ = p[CHBMV_DIAGONAL_ROWS];
        ar_ = p[2U * CHBMV_DIAGONAL_ROWS];
        ai_ = p[3U * CHBMV_DIAGONAL_ROWS];
        xr_ = p[4U * CHBMV_DIAGONAL_ROWS];
        xi_ = p[5U * CHBMV_DIAGONAL_ROWS];
        rr_ = p[6U * CHBMV_DIAGONAL_ROWS];
        ii_ = p[7U * CHBMV_DIAGONAL_ROWS];
        ri_ = p[8U * CHBMV_DIAGONAL_ROWS];
        ir_ = p[9U * CHBMV_DIAGONAL_ROWS];
        io_ = ioBuf_.Get<float>();
        auto idx = indexBuf_.Get<int32_t>();
        idxA_ = idx.ReinterpretCast<uint32_t>();
        idxX_ = idx[CHBMV_DIAGONAL_ROWS].ReinterpretCast<uint32_t>();
        idxPack_ = idx[2U * CHBMV_DIAGONAL_ROWS].ReinterpretCast<uint32_t>();
        idxFlat_ = idx[4U * CHBMV_DIAGONAL_ROWS];
        idxShift_ = idx[5U * CHBMV_DIAGONAL_ROWS];
        BuildIndices();
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockRows =
            t_.n <= 64U && 2U * t_.effectiveK >= t_.n ?
                8U :
                (t_.n <= CHBMV_DIAGONAL_SMALL_N ?
                     CHBMV_DIAGONAL_SMALL_ROWS :
                     (t_.n <= CHBMV_DIAGONAL_MEDIUM_N ? CHBMV_DIAGONAL_MEDIUM_ROWS : CHBMV_DIAGONAL_ROWS));
        const int64_t stride = static_cast<int64_t>(t_.blockCount) * blockRows;
        for (int64_t row = static_cast<int64_t>(GetBlockIdx()) * blockRows; row < t_.n; row += stride) {
            const uint32_t count = static_cast<uint32_t>(Min(blockRows, static_cast<int64_t>(t_.n) - row));
            Duplicate(accR_, 0.0f, count);
            Duplicate(accI_, 0.0f, count);
            PipeBarrier<PIPE_V>();
            if (t_.alphaR != 0.0f || t_.alphaI != 0.0f) {
                Compute(row, count);
                Scale(accR_, accI_, t_.alphaR, t_.alphaI, rr_, ii_, ri_, ir_, count);
            }
            FoldBeta(row, count);
            // Inspect before any GM y store; the helper still sees original old y.
            Sync<HardEvent::V_S>();
            const bool scalarRisk = ChbmvScalarRisk(t_);
            for (uint32_t i = 0; i < count; ++i) {
                float real = accR_.GetValue(i);
                float imag = accI_.GetValue(i);
                if (scalarRisk || !chbmv_numeric_range::Finite(real) || !chbmv_numeric_range::Finite(imag)) {
                    if (ChbmvTryFiniteRow(a_, x_, y_, t_, row + i, real, imag)) {
                        accR_.SetValue(i, real);
                        accI_.SetValue(i, imag);
                    }
                }
            }
            Sync<HardEvent::S_V>();
            Gather(io_, accR_, idxPack_, 0U, 2U * count);
            PipeBarrier<PIPE_V>();
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(y_[2 * row], io_, DataCopyExtParams{1U, count * 8U, 0U, 0U, 0U});
            Sync<HardEvent::MTE3_MTE2>();
            Sync<HardEvent::MTE3_V>();
        }
    }

private:
    __aicore__ inline void BuildIndices()
    {
        // Reuse adjacent scratch planes before Compute populates them.
        auto sequence = ar_.ReinterpretCast<int32_t>();
        auto half = xr_.ReinterpretCast<int32_t>();
        auto packed = idxPack_.ReinterpretCast<int32_t>();
        CreateVecIndex(sequence, static_cast<int32_t>(0), 2U * CHBMV_DIAGONAL_ROWS);
        PipeBarrier<PIPE_V>();
        ShiftRight(half, sequence, static_cast<int32_t>(1), 2U * CHBMV_DIAGONAL_ROWS);
        Muls(idxFlat_, sequence, static_cast<int32_t>(4), CHBMV_DIAGONAL_ROWS);
        Muls(idxX_.ReinterpretCast<int32_t>(), sequence, static_cast<int32_t>(8), CHBMV_DIAGONAL_ROWS);
        Muls(idxA_.ReinterpretCast<int32_t>(), sequence, static_cast<int32_t>(t_.lda * 8U), CHBMV_DIAGONAL_ROWS);
        PipeBarrier<PIPE_V>();
        Muls(packed, half, static_cast<int32_t>(2), 2U * CHBMV_DIAGONAL_ROWS);
        PipeBarrier<PIPE_V>();
        Sub(packed, sequence, packed, 2U * CHBMV_DIAGONAL_ROWS);
        PipeBarrier<PIPE_V>();
        Muls(packed, packed, static_cast<int32_t>(CHBMV_DIAGONAL_ROWS), 2U * CHBMV_DIAGONAL_ROWS);
        PipeBarrier<PIPE_V>();
        Add(packed, packed, half, 2U * CHBMV_DIAGONAL_ROWS);
        PipeBarrier<PIPE_V>();
        Muls(packed, packed, static_cast<int32_t>(4), 2U * CHBMV_DIAGONAL_ROWS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(int64_t row, uint32_t count)
    {
        const int64_t lo = Max(0, row - t_.effectiveK);
        const int64_t hi = Min(t_.n, row + count + t_.effectiveK);
        Sync<HardEvent::V_MTE2>();
        // Every DMA byte is inside the physical lda*n array; Gather below selects only valid band entries.
        Load(span_, a_, 2 * lo * t_.lda, static_cast<uint32_t>((hi - lo) * t_.lda));
        Load(xWin_, x_, 2 * lo, static_cast<uint32_t>(hi - lo));
        Sync<HardEvent::MTE2_V>();
        for (int64_t d = -static_cast<int64_t>(t_.effectiveK); d <= t_.effectiveK; ++d) {
            const int64_t first = Max(row, -d);
            const int64_t end = Min(row + count, static_cast<int64_t>(t_.n) - d);
            if (first < end) {
                AddDiagonal(row, lo, d, first, static_cast<uint32_t>(end - first));
            }
        }
    }

    __aicore__ inline void AddAlignedDiagonal(int64_t d, uint32_t aBase, uint32_t count, bool mirror)
    {
        // Keep the common aligned rows in their accumulator planes.
        if (d == 0) {
            PipeBarrier<PIPE_V>();
            MulAddDst(accR_, ar_, xr_, count);
            MulAddDst(accI_, ar_, xi_, count);
        } else {
            Gather(ai_, span_, idxA_, aBase + 4U, count);
            PipeBarrier<PIPE_V>();
            Mul(ii_, ai_, xi_, count);
            Mul(ir_, ai_, xr_, count);
            MulAddDst(accR_, ar_, xr_, count);
            MulAddDst(accI_, ar_, xi_, count);
            PipeBarrier<PIPE_V>();
            if (mirror) {
                Add(accR_, accR_, ii_, count);
                Sub(accI_, accI_, ir_, count);
            } else {
                Sub(accR_, accR_, ii_, count);
                Add(accI_, accI_, ir_, count);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddDiagonal(int64_t row, int64_t lo, int64_t d, int64_t first, uint32_t count)
    {
        const bool mirror = t_.upper ? d < 0 : d > 0;
        const int64_t column = mirror ? first : first + d;
        const int64_t bandRow = t_.upper ? static_cast<int64_t>(t_.k) - (d < 0 ? -d : d) : (d < 0 ? -d : d);
        const uint32_t aBase = static_cast<uint32_t>(((column - lo) * t_.lda + bandRow) * 8);
        const uint32_t xBase = static_cast<uint32_t>((first + d - lo) * 8);
        Gather(ar_, span_, idxA_, aBase, count);
        Gather(xr_, xWin_, idxX_, xBase, count);
        Gather(xi_, xWin_, idxX_, xBase + 4U, count);
        const uint32_t offset = static_cast<uint32_t>(first - row);
        if (offset == 0U) {
            AddAlignedDiagonal(d, aBase, count, mirror);
            return;
        }
        if (d == 0) {
            PipeBarrier<PIPE_V>();
            // A diagonal is real: do not manufacture 0*Inf by treating its ignored imaginary part as an operand.
            Mul(rr_, ar_, xr_, count);
            Mul(ri_, ar_, xi_, count);
        } else {
            Gather(ai_, span_, idxA_, aBase + 4U, count);
            PipeBarrier<PIPE_V>();
            if (mirror) {
                Muls(ai_, ai_, -1.0f, count);
                PipeBarrier<PIPE_V>();
            }
            ComplexProducts(ar_, ai_, xr_, xi_, rr_, ii_, ri_, ir_, count);
        }
        PipeBarrier<PIPE_V>();
        {
            // Vector operands need 32-byte alignment. Shift the result with Gather, rather than using acc[offset].
            Adds(idxShift_, idxFlat_, -static_cast<int32_t>(offset * 4U), CHBMV_DIAGONAL_ROWS);
            PipeBarrier<PIPE_V>();
            Maxs(idxShift_, idxShift_, 0, CHBMV_DIAGONAL_ROWS);
            PipeBarrier<PIPE_V>();
            Gather(ar_, rr_, idxShift_.ReinterpretCast<uint32_t>(), 0U, offset + count);
            Gather(ai_, ri_, idxShift_.ReinterpretCast<uint32_t>(), 0U, offset + count);
            PipeBarrier<PIPE_V>();
            Duplicate(ar_, 0.0f, offset);
            Duplicate(ai_, 0.0f, offset);
            PipeBarrier<PIPE_V>();
            Add(accR_, accR_, ar_, offset + count);
            Add(accI_, accI_, ai_, offset + count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FoldBeta(int64_t row, uint32_t count)
    {
        if (t_.betaR == 0.0f && t_.betaI == 0.0f) {
            return;
        }
        Sync<HardEvent::V_MTE2>();
        Load(io_, y_, 2 * row, count);
        Sync<HardEvent::MTE2_V>();
        Gather(ar_, io_, idxX_, 0U, count);
        Gather(ai_, io_, idxX_, 4U, count);
        PipeBarrier<PIPE_V>();
        // Netlib leaves y untouched for beta==1. In particular, do not form 0*Inf in the other component.
        if (t_.betaR != 1.0f || t_.betaI != 0.0f) {
            Scale(ar_, ai_, t_.betaR, t_.betaI, rr_, ii_, ri_, ir_, count);
        }
        Add(accR_, accR_, ar_, count);
        Add(accI_, accI_, ai_, count);
        PipeBarrier<PIPE_V>();
    }

    ChbmvTilingData t_{};
    GlobalTensor<float> a_, x_, y_;
    TBuf<TPosition::VECCALC> spanBuf_, xBuf_, planeBuf_, ioBuf_, indexBuf_;
    LocalTensor<float> span_, xWin_, accR_, accI_, ar_, ai_, xr_, xi_, rr_, ii_, ri_, ir_, io_;
    LocalTensor<uint32_t> idxA_, idxX_, idxPack_;
    LocalTensor<int32_t> idxFlat_, idxShift_;
};

class ChbmvRowDot {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const ChbmvTilingData& t, TPipe& pipe)
    {
        t_ = t;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        pipe.InitBuffer(ioBuf_, 16U * CHBMV_REDUCE * 4U);
        pipe.InitBuffer(planeBuf_, 12U * CHBMV_REDUCE * 4U);
        pipe.InitBuffer(idxBuf_, 2U * CHBMV_REDUCE * 4U);
        pipe.InitBuffer(outputBuf_, 2U * CHBMV_ROWS * 4U);
        output_ = outputBuf_.Get<float>();
        aIo_ = ioBuf_.Get<float>();
        xIo_ = aIo_[8U * CHBMV_REDUCE];
        auto p = planeBuf_.Get<float>();
        ar_ = p;
        ai_ = p[CHBMV_REDUCE];
        xr_ = p[2U * CHBMV_REDUCE];
        xi_ = p[3U * CHBMV_REDUCE];
        rr_ = p[4U * CHBMV_REDUCE];
        ii_ = p[5U * CHBMV_REDUCE];
        ri_ = p[6U * CHBMV_REDUCE];
        ir_ = p[7U * CHBMV_REDUCE];
        tmp_ = p[8U * CHBMV_REDUCE];
        red_ = p[9U * CHBMV_REDUCE];
        idxA_ = idxBuf_.Get<uint32_t>();
        idxX_ = idxA_[CHBMV_REDUCE];
    }

    __aicore__ inline void Process()
    {
        const int64_t rowStride = static_cast<int64_t>(t_.blockCount) * CHBMV_ROWS;
        for (int64_t base = static_cast<int64_t>(GetBlockIdx()) * CHBMV_ROWS; base < t_.n; base += rowStride) {
            ProcessRange(base, Min(t_.n, base + CHBMV_ROWS));
        }
    }

    // Compute and commit an uncommitted logical row range.
    __aicore__ inline void ProcessRange(int64_t first, int64_t end)
    {
        for (int64_t base = first; base < end; base += CHBMV_ROWS) {
            const uint32_t count = static_cast<uint32_t>(Min(CHBMV_ROWS, end - base));
            for (uint32_t i = 0; i < count; ++i) {
                ComputeRow(base + i, i);
            }
            if (t_.incy == 1) {
                ChbmvCommitExact(y_, base, output_, count);
            } else {
                for (uint32_t i = 0; i < count; ++i) {
                    red_.SetValue(0, output_.GetValue(2U * i));
                    red_.SetValue(1, output_.GetValue(2U * i + 1U));
                    ChbmvCommitExact(y_, VectorPosition(base + i, t_.n, t_.incy), red_, 1U);
                }
            }
        }
    }

private:
    __aicore__ inline void LoadStepped(
        const LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t first, int64_t step, uint32_t count,
        const LocalTensor<uint32_t>& indices, bool reverse)
    {
        const uint32_t pitch = step == 1 ? 2U : 8U;
        for (uint32_t i = 0; i < count; ++i) {
            indices.SetValue(i, (reverse ? count - 1U - i : i) * pitch * 4U);
        }
        Sync<HardEvent::S_V>();
        if (step == 1) {
            Load(dst, src, 2 * first, count);
        } else if (static_cast<uint64_t>(step - 1) * 8U > UINT32_MAX) {
            // The address is 64-bit, but the DMA stride field is 32-bit. Keep valid large strides exact.
            for (uint32_t i = 0; i < count; ++i) {
                dst.SetValue(i * 8U, src.GetValue(2 * (first + i * step)));
                dst.SetValue(i * 8U + 1U, src.GetValue(2 * (first + i * step) + 1));
            }
            Sync<HardEvent::S_V>();
        } else {
            DataCopyPad(
                dst, src[2 * first],
                DataCopyExtParams{static_cast<uint16_t>(count), 8U, static_cast<uint32_t>((step - 1) * 8), 0U, 0U},
                DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
        }
    }

    __aicore__ inline void SumSegment(int64_t row, int64_t first, int64_t end, bool mirror, float& sr, float& si)
    {
        for (int64_t j = first; j < end; j += CHBMV_REDUCE) {
            const uint32_t count = static_cast<uint32_t>(Min(CHBMV_REDUCE, end - j));
            const int64_t aPos = t_.upper ? (mirror ? row * t_.lda + t_.k + j - row : j * t_.lda + t_.k + row - j) :
                                            (mirror ? row * t_.lda + j - row : j * t_.lda + row - j);
            const int64_t aStep = mirror ? 1 : static_cast<int64_t>(t_.lda) - 1;
            const int64_t xFirst = VectorPosition(t_.incx > 0 ? j : j + count - 1U, t_.n, t_.incx);
            Sync<HardEvent::V_MTE2>();
            LoadStepped(aIo_, a_, aPos, aStep, count, idxA_, false);
            LoadStepped(xIo_, x_, xFirst, t_.incx > 0 ? t_.incx : -t_.incx, count, idxX_, t_.incx < 0);
            Sync<HardEvent::MTE2_V>();
            Gather(ar_, aIo_, idxA_, 0U, count);
            Gather(ai_, aIo_, idxA_, 4U, count);
            Gather(xr_, xIo_, idxX_, 0U, count);
            Gather(xi_, xIo_, idxX_, 4U, count);
            PipeBarrier<PIPE_V>();
            if (mirror) {
                Muls(ai_, ai_, -1.0f, count);
                PipeBarrier<PIPE_V>();
            }
            ComplexProducts(ar_, ai_, xr_, xi_, rr_, ii_, ri_, ir_, count);
            ReduceSum(red_, rr_, tmp_, count);
            PipeBarrier<PIPE_V>();
            ReduceSum(red_[8], ri_, tmp_, count);
            Sync<HardEvent::V_S>();
            sr += red_.GetValue(0);
            si += red_.GetValue(8);
        }
    }

    __aicore__ inline void ComputeRow(int64_t row, uint32_t lane)
    {
        float sr = 0.0f;
        float si = 0.0f;
        if (t_.alphaR != 0.0f || t_.alphaI != 0.0f) {
            const int64_t diag = row * t_.lda + (t_.upper ? t_.k : 0U);
            const float diagonal = a_.GetValue(2 * diag);
            const int64_t xp = 2 * VectorPosition(row, t_.n, t_.incx);
            sr = diagonal * x_.GetValue(xp);
            si = diagonal * x_.GetValue(xp + 1);
            SumSegment(row, Max(0, row - t_.effectiveK), row, t_.upper != 0U, sr, si);
            SumSegment(row, row + 1, Min(t_.n, row + t_.effectiveK + 1), t_.upper == 0U, sr, si);
            const float r = t_.alphaR * sr - t_.alphaI * si;
            si = t_.alphaR * si + t_.alphaI * sr;
            sr = r;
        }
        const int64_t yp = 2 * VectorPosition(row, t_.n, t_.incy);
        if (t_.betaR != 0.0f || t_.betaI != 0.0f) {
            const float yr = y_.GetValue(yp);
            const float yi = y_.GetValue(yp + 1);
            if (t_.betaR == 1.0f && t_.betaI == 0.0f) {
                sr += yr;
                si += yi;
            } else {
                sr += t_.betaR * yr - t_.betaI * yi;
                si += t_.betaR * yi + t_.betaI * yr;
            }
        }
        if (ChbmvScalarRisk(t_) || !chbmv_numeric_range::Finite(sr) || !chbmv_numeric_range::Finite(si)) {
            ChbmvTryFiniteRow(a_, x_, y_, t_, row, sr, si);
        }
        output_.SetValue(2U * lane, sr);
        output_.SetValue(2U * lane + 1U, si);
    }

    ChbmvTilingData t_{};
    GlobalTensor<float> a_, x_, y_;
    TBuf<TPosition::VECCALC> ioBuf_, planeBuf_, idxBuf_, outputBuf_;
    LocalTensor<float> aIo_, xIo_, ar_, ai_, xr_, xi_, rr_, ii_, ri_, ir_, tmp_, red_, output_;
    LocalTensor<uint32_t> idxA_, idxX_;
};
/*
 * Continuous wide-band panels share input windows across owned rows.
 * Irregular and non-finite rows retain the general row path.
 */

constexpr uint32_t CHBMV_WIDE_ROWS = 32U;
constexpr uint32_t CHBMV_WIDE_COLS = 64U;
constexpr uint32_t CHBMV_WIDE_LANES = CHBMV_WIDE_ROWS * CHBMV_WIDE_COLS;

__aicore__ inline bool ChbmvWideFinite(float value)
{
    constexpr float LIMIT = 3.40282346638528859812e38F;
    return value == value && value <= LIMIT && value >= -LIMIT;
}

__aicore__ inline bool ChbmvUseWidePanel(const ChbmvTilingData& t)
{
    // These are geometric/API safety guards, not claims about performance.
    // The host balances whole 32-byte output groups across the available cores.
    return t.incx == 1 && t.incy == 1 && t.effectiveK >= CHBMV_WIDE_COLS && t.lda > CHBMV_WIDE_COLS &&
           static_cast<uint64_t>(t.lda - 1U) * 8U <= UINT32_MAX && ChbmvWideFinite(t.alphaR) &&
           ChbmvWideFinite(t.alphaI) && ChbmvWideFinite(t.betaR) && ChbmvWideFinite(t.betaI);
}

class ChbmvWidePanel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const ChbmvTilingData& t, TPipe& pipe)
    {
        t_ = t;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        pipe.InitBuffer(aIoBuf_, CHBMV_WIDE_LANES * 8U);
        pipe.InitBuffer(xIoBuf_, CHBMV_WIDE_COLS * 8U);
        pipe.InitBuffer(planeBuf_, 6U * CHBMV_WIDE_LANES * 4U);
        pipe.InitBuffer(indexBuf_, 3U * CHBMV_WIDE_LANES * 4U);
        pipe.InitBuffer(accBuf_, 4U * CHBMV_WIDE_ROWS * 4U);
        pipe.InitBuffer(outputBuf_, 2U * CHBMV_WIDE_ROWS * 4U);
        output_ = outputBuf_.Get<float>();
        aIo_ = aIoBuf_.Get<float>();
        xIo_ = xIoBuf_.Get<float>();
        auto planes = planeBuf_.Get<float>();
        ar_ = planes;
        ai_ = planes[CHBMV_WIDE_LANES];
        xr_ = planes[2U * CHBMV_WIDE_LANES];
        xi_ = planes[3U * CHBMV_WIDE_LANES];
        pr_ = planes[4U * CHBMV_WIDE_LANES];
        pi_ = planes[5U * CHBMV_WIDE_LANES];
        auto index = indexBuf_.Get<int32_t>();
        directIndex_ = index;
        mirrorIndex_ = index[CHBMV_WIDE_LANES];
        xIndex_ = index[2U * CHBMV_WIDE_LANES];
        auto acc = accBuf_.Get<float>();
        accR_ = acc;
        accI_ = acc[CHBMV_WIDE_ROWS];
        redR_ = acc[2U * CHBMV_WIDE_ROWS];
        redI_ = acc[3U * CHBMV_WIDE_ROWS];
        BuildIndices();
        InitWp2(pipe);
        // Allocate the original fallback once per core, never inside the tile loop.
        // Combined explicit UB: panel 108,672 B + row-dot 31,232 B = 139,904 B.
        fallback_.Init(a, x, y, t, pipe);
    }

    __aicore__ inline void Process()
    {
        const uint32_t groups = (t_.n - 1U) / 4U + 1U;
        const uint32_t equalGroups = groups / t_.blockCount;
        const uint32_t extraGroups = groups % t_.blockCount;
        const uint32_t core = GetBlockIdx();
        const uint32_t firstGroup = core * equalGroups + (core < extraGroups ? core : extraGroups);
        const uint32_t ownedGroups = equalGroups + (core < extraGroups ? 1U : 0U);
        const int64_t first = static_cast<int64_t>(firstGroup) * 4;
        const int64_t end = Min(t_.n, static_cast<int64_t>(firstGroup + ownedGroups) * 4);
        // Every inter-core boundary is a complete 32-byte transaction. Keep each
        // core's rows contiguous instead of assigning it a second distant wave.
        for (int64_t row = first; row < end; row += CHBMV_WIDE_ROWS) {
            const uint32_t rows = static_cast<uint32_t>(Min(CHBMV_WIDE_ROWS, end - row));
            ProcessPanel(row, rows);
        }
    }

private:
    __aicore__ inline void BuildIndices()
    {
        // Lane (r,c), r<CHBMV_WIDE_ROWS,c<64. Direct DMA uses CHBMV_WIDE_ROWS pitch;
        // mirror DMA already has one output row per 64-complex block.
        CreateVecIndex(mirrorIndex_, 0, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        ShiftRight(xIndex_, mirrorIndex_, static_cast<int32_t>(6), CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        Muls(directIndex_, xIndex_, static_cast<int32_t>(CHBMV_WIDE_COLS), CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        Sub(directIndex_, mirrorIndex_, directIndex_, CHBMV_WIDE_LANES);
        Muls(mirrorIndex_, mirrorIndex_, 8, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        Muls(directIndex_, directIndex_, static_cast<int32_t>(CHBMV_WIDE_ROWS * 8U), CHBMV_WIDE_LANES);
        Muls(xIndex_, xIndex_, 8, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        Add(directIndex_, directIndex_, xIndex_, CHBMV_WIDE_LANES);
        Muls(xIndex_, xIndex_, static_cast<int32_t>(CHBMV_WIDE_COLS), CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        Sub(xIndex_, mirrorIndex_, xIndex_, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadPanel(int64_t row, uint32_t rows, int64_t first, uint32_t cols, bool mirror)
    {
        const int64_t origin = t_.upper ? t_.k : 0U;
        const int64_t source = mirror ? row * (static_cast<int64_t>(t_.lda) - 1) + origin + first :
                                        first * (static_cast<int64_t>(t_.lda) - 1) + origin + row;
        const uint32_t length = (mirror ? cols : rows) * 8U;
        const uint32_t dstPitch = (mirror ? CHBMV_WIDE_COLS : CHBMV_WIDE_ROWS) * 8U;
        const uint32_t rounded = (length + 31U) / 32U * 32U;
        const uint32_t srcStride =
            static_cast<uint32_t>((static_cast<uint64_t>(t_.lda) - 1U - (mirror ? cols : rows)) * 8U);
        const uint32_t dstStride = (dstPitch - rounded) / 32U;
        Sync<HardEvent::V_MTE2>();
        DataCopyPad(
            aIo_, a_[2 * source],
            DataCopyExtParams{static_cast<uint16_t>(mirror ? rows : cols), length, srcStride, dstStride, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
        Load(xIo_, x_, 2 * first, cols);
        Sync<HardEvent::MTE2_V>();
        const auto indices = (mirror ? mirrorIndex_ : directIndex_).ReinterpretCast<uint32_t>();
        Gather(ar_, aIo_, indices, 0U, rows * CHBMV_WIDE_COLS);
        Gather(ai_, aIo_, indices, 4U, rows * CHBMV_WIDE_COLS);
        Gather(xr_, xIo_, xIndex_.ReinterpretCast<uint32_t>(), 0U, rows * CHBMV_WIDE_COLS);
        Gather(xi_, xIo_, xIndex_.ReinterpretCast<uint32_t>(), 4U, rows * CHBMV_WIDE_COLS);
        PipeBarrier<PIPE_V>();
        if (mirror) {
            Muls(ai_, ai_, -1.0f, rows * CHBMV_WIDE_COLS);
            PipeBarrier<PIPE_V>();
        }
        // Tail lanes may contain arbitrary padding. Neither products nor reductions
        // below touch c>=cols, avoiding the invalid 0*Inf/NaN masking shortcut.
    }

    __aicore__ inline void MultiplyReduce(uint32_t rows, uint32_t cols)
    {
        const BinaryRepeatParams rep{1, 1, 1, 8, 8, 8};
        const uint64_t mask = cols;
        const uint8_t repeats = static_cast<uint8_t>(rows);
        Mul(pr_, ar_, xr_, mask, repeats, rep);
        Mul(pi_, ai_, xi_, mask, repeats, rep);
        PipeBarrier<PIPE_V>();
        Sub(pr_, pr_, pi_, mask, repeats, rep);
        Mul(pi_, ar_, xi_, mask, repeats, rep);
        PipeBarrier<PIPE_V>();
        Mul(ar_, ai_, xr_, mask, repeats, rep);
        PipeBarrier<PIPE_V>();
        Add(pi_, pi_, ar_, mask, repeats, rep);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(redR_, pr_, mask, repeats, 1, 1, 8);
        WholeReduceSum(redI_, pi_, mask, repeats, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Add(accR_, accR_, redR_, rows);
        Add(accI_, accI_, redI_, rows);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CommonInterior(int64_t row, uint32_t rows, int64_t first, int64_t end, bool mirror)
    {
        for (int64_t col = first; col < end; col += CHBMV_WIDE_COLS) {
            const uint32_t cols = static_cast<uint32_t>(Min(CHBMV_WIDE_COLS, end - col));
            LoadPanel(row, rows, col, cols, mirror);
            MultiplyReduce(rows, cols);
        }
    }

    __aicore__ inline bool FinishRow(int64_t row, float& real, float& imag)
    {
        if (ChbmvScalarRisk(t_) || !ChbmvWideFinite(real) || !ChbmvWideFinite(imag)) {
            if (ChbmvTryFiniteRow(a_, x_, y_, t_, row, real, imag)) {
                return true;
            }
        }
        if (!ChbmvWideFinite(real) || !ChbmvWideFinite(imag)) {
            fallback_.ProcessRange(row, row + 1);
            return false;
        }
        const float scaledReal = t_.alphaR * real - t_.alphaI * imag;
        const float scaledImag = t_.alphaR * imag + t_.alphaI * real;
        real = scaledReal;
        imag = scaledImag;
        const int64_t at = 2 * row;
        if (t_.betaR != 0.0f || t_.betaI != 0.0f) {
            const float yr = y_.GetValue(at);
            const float yi = y_.GetValue(at + 1);
            if (!ChbmvWideFinite(yr) || !ChbmvWideFinite(yi)) {
                fallback_.ProcessRange(row, row + 1);
                return false;
            }
            if (t_.betaR == 1.0f && t_.betaI == 0.0f) {
                real += yr;
                imag += yi;
            } else {
                real += t_.betaR * yr - t_.betaI * yi;
                imag += t_.betaR * yi + t_.betaI * yr;
            }
        }
        if (!ChbmvWideFinite(real) || !ChbmvWideFinite(imag)) {
            if (ChbmvTryFiniteRow(a_, x_, y_, t_, row, real, imag)) {
                return true;
            }
            fallback_.ProcessRange(row, row + 1);
            return false;
        }
        return true;
    }

    __aicore__ inline void CommitPanel(int64_t row, uint32_t rows, uint32_t completed)
    {
        const uint32_t fullMask = rows == 32U ? UINT32_MAX : ((1U << rows) - 1U);
        if (completed == fullMask) {
            // The common PF path commits one panel DMA, with no scalar GM cache writes.
            ChbmvCommitExact(y_, row, output_, rows);
        } else {
            // IEEE fallback rows have already committed themselves. Do not overwrite them.
            for (uint32_t i = 0; i < rows; ++i) {
                if ((completed & (1U << i)) != 0U) {
                    redR_.SetValue(0, output_.GetValue(2U * i));
                    redR_.SetValue(1, output_.GetValue(2U * i + 1U));
                    ChbmvCommitExact(y_, row + i, redR_, 1U);
                }
            }
        }
    }

    __aicore__ inline void ProcessPanel(int64_t row, uint32_t rows)
    {
        // alpha==0 retains the original no-read-A/x path, including beta special values.
        if (t_.alphaR == 0.0f && t_.alphaI == 0.0f) {
            fallback_.ProcessRange(row, row + rows);
            return;
        }
        Duplicate(accR_, 0.0f, rows);
        Duplicate(accI_, 0.0f, rows);
        PipeBarrier<PIPE_V>();
        const int64_t commonLo = Max(0, row + rows - 1 - static_cast<int64_t>(t_.effectiveK));
        const int64_t commonHi = Min(t_.n, row + t_.effectiveK + 1);
        CommonInterior(row, rows, commonLo, row, t_.upper != 0U);
        CommonInterior(row, rows, row + rows, commonHi, t_.upper == 0U);
        Wp2BoundaryPanels(row, rows, commonLo, commonHi);
        Sync<HardEvent::V_S>();
        uint32_t completed = 0U;
        for (uint32_t i = 0; i < rows; ++i) {
            const int64_t current = row + i;
            float real = accR_.GetValue(i);
            float imag = accI_.GetValue(i);
            if (FinishRow(current, real, imag)) {
                output_.SetValue(2U * i, real);
                output_.SetValue(2U * i + 1U, imag);
                completed |= 1U << i;
            }
        }
        CommitPanel(row, rows, completed);
    }

    ChbmvTilingData t_{};
    GlobalTensor<float> a_, x_, y_;
    TBuf<TPosition::VECCALC> aIoBuf_, xIoBuf_, planeBuf_, indexBuf_, accBuf_, outputBuf_;
    LocalTensor<float> aIo_, xIo_, ar_, ai_, xr_, xi_, pr_, pi_, accR_, accI_, redR_, redI_, output_;
    LocalTensor<int32_t> directIndex_, mirrorIndex_, xIndex_;
    ChbmvRowDot fallback_;

private:
    // Vectorize the central triangle and band boundaries with b32 lane masks.
    // Reuse shared row finalization, output, and exceptional-value fallback.

    __aicore__ inline void InitVectorBandMasks(TPipe& pipe)
    {
        pipe.InitBuffer(wp2DeltaBuf_, CHBMV_WIDE_LANES * 4U);
        pipe.InitBuffer(wp2ValidMaskBuf_, CHBMV_WIDE_LANES / 8U);
        pipe.InitBuffer(wp2UpperMaskBuf_, CHBMV_WIDE_LANES / 8U);
        wp2Delta_ = wp2DeltaBuf_.Get<float>();
        wp2ValidMask_ = wp2ValidMaskBuf_.Get<uint8_t>();
        wp2UpperMask_ = wp2UpperMaskBuf_.Get<uint8_t>();
    }

    __aicore__ inline void InitWp2(TPipe& pipe)
    {
        pipe.InitBuffer(wp2IndexBuf_, CHBMV_WIDE_LANES * 4U);
        pipe.InitBuffer(wp2MaskBuf_, CHBMV_WIDE_LANES / 8U);
        pipe.InitBuffer(wp2DiagBuf_, CHBMV_WIDE_ROWS * 4U);
        wp2Index_ = wp2IndexBuf_.Get<int32_t>();
        wp2ConjMask_ = wp2MaskBuf_.Get<uint8_t>();
        wp2DiagIndex_ = wp2DiagBuf_.Get<int32_t>();
        InitVectorBandMasks(pipe);
        auto hi = ar_.ReinterpretCast<int32_t>();
        auto lo = ai_.ReinterpretCast<int32_t>();
        // mirrorIndex=(r*64+c)*8, xIndex=c*8. Derive r*8 without SetValue loops.
        Sub(wp2Index_, mirrorIndex_, xIndex_, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        ShiftRight(wp2Index_, wp2Index_, static_cast<int32_t>(6), CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        Max(hi, wp2Index_, xIndex_, CHBMV_WIDE_LANES);
        Min(lo, wp2Index_, xIndex_, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        if (t_.upper) {
            Muls(wp2Index_, hi, static_cast<int32_t>(CHBMV_WIDE_ROWS), CHBMV_WIDE_LANES);
            PipeBarrier<PIPE_V>();
            Add(wp2Index_, wp2Index_, lo, CHBMV_WIDE_LANES);
        } else {
            Muls(wp2Index_, lo, static_cast<int32_t>(CHBMV_WIDE_ROWS), CHBMV_WIDE_LANES);
            PipeBarrier<PIPE_V>();
            Add(wp2Index_, wp2Index_, hi, CHBMV_WIDE_LANES);
        }
        PipeBarrier<PIPE_V>();
        // The central reflected half alone needs conjugation, not the whole rectangle.
        Sub(hi, mirrorIndex_, xIndex_, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        ShiftRight(hi, hi, static_cast<int32_t>(6), CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        Sub(hi, hi, xIndex_, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        Cast(wp2Delta_, hi, RoundMode::CAST_NONE, CHBMV_WIDE_LANES);
        PipeBarrier<PIPE_V>();
        if (t_.upper) {
            Compares(wp2ConjMask_, wp2Delta_, 0.0f, CMPMODE::GT, CHBMV_WIDE_LANES);
        } else {
            Compares(wp2ConjMask_, wp2Delta_, 0.0f, CMPMODE::LT, CHBMV_WIDE_LANES);
        }
        CreateVecIndex(wp2DiagIndex_, 0, CHBMV_WIDE_ROWS);
        PipeBarrier<PIPE_V>();
        Muls(wp2DiagIndex_, wp2DiagIndex_, static_cast<int32_t>((CHBMV_WIDE_ROWS + 1U) * 8U), CHBMV_WIDE_ROWS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float Wp2BandThreshold(int64_t value)
    {
        const int64_t limit = static_cast<int64_t>(CHBMV_WIDE_COLS);
        return static_cast<float>(Max(-limit, Min(limit, value)) * 8);
    }

    __aicore__ inline void BuildVectorBandMask(int64_t row, uint32_t rows, int64_t first, bool excludeDiagonal)
    {
        const uint32_t count = rows * CHBMV_WIDE_COLS;
        const int64_t offset = first - row;
        Compares(wp2ValidMask_, wp2Delta_, Wp2BandThreshold(offset - t_.effectiveK), CMPMODE::GE, count);
        Compares(wp2UpperMask_, wp2Delta_, Wp2BandThreshold(offset + t_.effectiveK), CMPMODE::LE, count);
        PipeBarrier<PIPE_V>();
        auto valid = wp2ValidMask_.ReinterpretCast<uint16_t>();
        auto scratch = wp2UpperMask_.ReinterpretCast<uint16_t>();
        And(valid, valid, scratch, count / 16U);
        PipeBarrier<PIPE_V>();
        if (excludeDiagonal) {
            Compares(wp2UpperMask_, wp2Delta_, Wp2BandThreshold(offset), CMPMODE::NE, count);
            PipeBarrier<PIPE_V>();
            And(valid, valid, scratch, count / 16U);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void NeutralizeOutsideBand(uint32_t count)
    {
        // Both operands must be zero before multiplication: zero times Inf is invalid.
        Duplicate(pr_, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Select(ar_, wp2ValidMask_, ar_, pr_, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Select(ai_, wp2ValidMask_, ai_, pr_, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Select(xr_, wp2ValidMask_, xr_, pr_, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Select(xi_, wp2ValidMask_, xi_, pr_, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Wp2MaskedReduce(
        int64_t row, uint32_t rows, int64_t first, uint32_t cols, bool excludeDiagonal)
    {
        BuildVectorBandMask(row, rows, first, excludeDiagonal);
        NeutralizeOutsideBand(rows * CHBMV_WIDE_COLS);
        // Preserve the original cols mask: arbitrary tail columns never enter arithmetic.
        MultiplyReduce(rows, cols);
    }

    __aicore__ inline void Wp2CentralLoad(int64_t row, uint32_t rows)
    {
        const int64_t origin = t_.upper ? t_.k : 0U;
        const int64_t first = row * t_.lda + origin;
        const uint32_t length = rows * 8U;
        const uint32_t rounded = (length + 31U) / 32U * 32U;
        Sync<HardEvent::V_MTE2>();
        DataCopyPad(
            aIo_, a_[2 * first],
            DataCopyExtParams{
                static_cast<uint16_t>(rows), length,
                static_cast<uint32_t>((static_cast<uint64_t>(t_.lda) - 1U - rows) * 8U),
                (CHBMV_WIDE_ROWS * 8U - rounded) / 32U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
        Load(xIo_, x_, 2 * row, rows);
        Sync<HardEvent::MTE2_V>();
        Gather(ar_, aIo_, wp2Index_.ReinterpretCast<uint32_t>(), 0U, rows * CHBMV_WIDE_COLS);
        Gather(ai_, aIo_, wp2Index_.ReinterpretCast<uint32_t>(), 4U, rows * CHBMV_WIDE_COLS);
        Gather(xr_, xIo_, xIndex_.ReinterpretCast<uint32_t>(), 0U, rows * CHBMV_WIDE_COLS);
        Gather(xi_, xIo_, xIndex_.ReinterpretCast<uint32_t>(), 4U, rows * CHBMV_WIDE_COLS);
        PipeBarrier<PIPE_V>();
        Muls(pr_, ai_, -1.0f, rows * CHBMV_WIDE_COLS);
        PipeBarrier<PIPE_V>();
        Select(ai_, wp2ConjMask_, pr_, ai_, SELMODE::VSEL_TENSOR_TENSOR_MODE, rows * CHBMV_WIDE_COLS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Wp2RealDiagonal(uint32_t rows)
    {
        // Central off-diagonal masks excluded i==j. Handle real diagonal separately:
        // an invented complex imaginary zero times Inf must never enter this product.
        Gather(ar_, aIo_, wp2DiagIndex_.ReinterpretCast<uint32_t>(), 0U, rows);
        Gather(xr_, xIo_, xIndex_.ReinterpretCast<uint32_t>(), 0U, rows);
        Gather(xi_, xIo_, xIndex_.ReinterpretCast<uint32_t>(), 4U, rows);
        PipeBarrier<PIPE_V>();
        Mul(pr_, ar_, xr_, rows);
        Mul(pi_, ar_, xi_, rows);
        PipeBarrier<PIPE_V>();
        Add(accR_, accR_, pr_, rows);
        Add(accI_, accI_, pi_, rows);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Wp2BoundaryPanels(int64_t row, uint32_t rows, int64_t commonLo, int64_t commonHi)
    {
        const int64_t edgeLo = Max(0, row - static_cast<int64_t>(t_.effectiveK));
        const int64_t edgeHi = Min(t_.n, row + rows + t_.effectiveK);
        if (edgeLo < commonLo) {
            const uint32_t cols = static_cast<uint32_t>(commonLo - edgeLo);
            LoadPanel(row, rows, edgeLo, cols, t_.upper != 0U);
            Wp2MaskedReduce(row, rows, edgeLo, cols, false);
        }
        Wp2CentralLoad(row, rows);
        Wp2MaskedReduce(row, rows, row, rows, true);
        Wp2RealDiagonal(rows);
        if (commonHi < edgeHi) {
            const uint32_t cols = static_cast<uint32_t>(edgeHi - commonHi);
            LoadPanel(row, rows, commonHi, cols, t_.upper == 0U);
            Wp2MaskedReduce(row, rows, commonHi, cols, false);
        }
    }

    TBuf<TPosition::VECCALC> wp2IndexBuf_, wp2MaskBuf_, wp2DiagBuf_;
    LocalTensor<int32_t> wp2Index_, wp2DiagIndex_;
    LocalTensor<uint8_t> wp2ConjMask_;
    TBuf<TPosition::VECCALC> wp2DeltaBuf_, wp2ValidMaskBuf_, wp2UpperMaskBuf_;
    LocalTensor<float> wp2Delta_;
    LocalTensor<uint8_t> wp2ValidMask_, wp2UpperMask_;
};

} // namespace

extern "C" __global__ __aicore__ void chbmv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, ChbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (tiling.path == CHBMV_PATH_DIAGONALS) {
        ChbmvDiagonals op;
        op.Init(a, x, y, tiling, pipe);
        op.Process();
    } else if (ChbmvUseWidePanel(tiling)) {
        ChbmvWidePanel op;
        op.Init(a, x, y, tiling, pipe);
        op.Process();
    } else {
        ChbmvRowDot op;
        op.Init(a, x, y, tiling, pipe);
        op.Process();
    }
}

void chbmv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint32_t blocks, const ChbmvTilingData& tiling, void* stream)
{
    chbmv_kernel<<<blocks, nullptr, stream>>>(a, x, y, tiling);
}
