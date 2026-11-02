/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "ctrmm_kernel_common.h"
#include "ctrmm_finish_layout.h"
#include "ctrmm_vector_fma.h"

namespace {
class CtrmmFinish {
    static constexpr uint32_t RECT_SIZE = 4096U;

public:
    __aicore__ inline void Init(GM_ADDR c, GM_ADDR workspace, const CtrmmTiling& t)
    {
        t_ = t;
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        c_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
        exceptional_ = Exceptional(workspace, t_);
        if (exceptional_) {
            const auto* flags = reinterpret_cast<const __gm__ uint32_t*>(workspace) + t.flagsOffset;
            uint32_t bits = 0;
            for (uint32_t i = 0; i < t.vectorCores; ++i) {
                bits |= flags[i];
            }
            vectorReference_ = (bits & kScalarFmaRequired) == 0;
        }
        if (!exceptional_ && t_.useCube) {
            InitRectangle();
            return;
        }
        InitVectorBuffers();
        InitOutputIndices();
    }
    __aicore__ inline void Run()
    {
        if (!exceptional_ && t_.useCube) {
            RunRectangle();
            return;
        }
        const uint32_t tile = vectorReference_ ? kReferenceColumns : RHS;
        const uint32_t rhsTiles = (t_.r + tile - 1U) / tile;
        const uint64_t tasks = static_cast<uint64_t>(t_.q) * rhsTiles;
        for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
            row_ = static_cast<uint32_t>(task / rhsTiles);
            col_ = static_cast<uint32_t>(task % rhsTiles) * tile;
            count_ = Min(tile, t_.r - col_);
            cached_ = kInvalidCachedRow;
            if (vectorReference_) {
                BatchedOrderedVector();
            } else if (exceptional_) {
                Reference();
            } else {
                Vector();
            }
            Store();
        }
    }

private:
    __aicore__ inline void VectorFma(LocalTensor<float> dst, float a, LocalTensor<float> b, LocalTensor<float> c)
    {
        ScalarVectorFma(dst, a, b, c, count_, fmaBuf_, fmaMaskBuf_);
    }
    __aicore__ inline void VectorVectorFma(
        LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float> c, uint32_t n)
    {
        BatchedVectorFma(dst, a, b, c, n, packed_, fmaBuf_, fmaMaskBuf_);
    }
    __aicore__ inline void AccumulateReference(
        ComplexValue alpha, ComplexValue a, ComplexValue factor, bool leftN, bool leftT)
    {
        for (uint32_t j = 0; j < count_; ++j) {
            ComplexValue x{xr_.GetValue(j), xi_.GetValue(j)};
            if (leftN && IsZero(x)) {
                continue;
            }
            ComplexValue term;
            if (leftN) {
                term = ComplexProduct(ComplexProduct(alpha, x, true), a);
            } else {
                term = ComplexProduct(factor, x, leftT);
            }
            yr_.SetValue(j, yr_.GetValue(j) + term.real);
            yi_.SetValue(j, yi_.GetValue(j) + term.imag);
        }
    }
    __aicore__ inline void SeedReference(ComplexValue alpha, ComplexValue diagonal, bool leftN, bool leftT, bool rightT)
    {
        LoadX(row_);
        Fence<HardEvent::MTE2_S>();
        ComplexValue d = t_.unit ? alpha : ComplexProduct(alpha, diagonal, true);
        for (uint32_t j = 0; j < count_; ++j) {
            const ComplexValue x{xr_.GetValue(j), xi_.GetValue(j)};
            ComplexValue y;
            if (leftN) {
                y = IsZero(x) ? x : ComplexProduct(alpha, x, true);
                if (!IsZero(x) && !t_.unit) {
                    y = ComplexProduct(y, diagonal);
                }
            } else if (leftT) {
                y = t_.unit ? x : ComplexProduct(x, diagonal);
            } else {
                y = rightT && IsOne(d) ? x : ComplexProduct(d, x);
            }
            yr_.SetValue(j, y.real);
            yi_.SetValue(j, y.imag);
        }
    }
    __aicore__ inline void ComputeBatch(uint32_t ks, bool leftN)
    {
        constexpr uint32_t COLS = kReferenceColumns;
        auto av = packed_[4U * RHS];
        auto iv = packed_[5U * RHS];
        const uint32_t n = ks * COLS;
        if (leftN && t_.alphaReal != 1.0f) {
            Muls(xr_, xr_, t_.alphaReal, n);
            Muls(xi_, xi_, t_.alphaReal, n);
            PipeBarrier<PIPE_V>();
        }
        Mul(tmp_[2U * RHS], xi_, iv, RHS);
        if (!t_.right) {
            Mul(tmp_[3U * RHS], xr_, iv, RHS);
        } else {
            Mul(tmp_[3U * RHS], xi_, av, RHS);
        }
        PipeBarrier<PIPE_V>();
        Muls(tmp_[2U * RHS], tmp_[2U * RHS], -1.0f, RHS);
        if (!t_.right) {
            Adds(iv, av, 0.0f, RHS);
        } else {
            Adds(xi_, xr_, 0.0f, RHS);
        }
        PipeBarrier<PIPE_V>();
        VectorVectorFma(tmp_, av, xr_, tmp_[2U * RHS], 2U * RHS);
    }
    __aicore__ inline void LoadBatchX(uint32_t begin, uint32_t ks)
    {
        constexpr uint32_t COLS = kReferenceColumns;
        const bool compact = t_.useCube;
        const uint32_t xpitch = (compact ? 2U : 1U) * t_.rp;
        const uint64_t offset = (compact ? kCubeAPlanes * t_.aPlane + 2U * t_.xPlane : kOriginalAPlanes * t_.aPlane) +
                                static_cast<uint64_t>(begin) * xpitch + col_;
        const DataCopyExtParams cp{
            static_cast<uint16_t>(ks), count_ * 4U, (xpitch - count_) * 4U, (COLS - (count_ + 7U) / 8U * 8U) / 8U, 0};
        DataCopyPad(xr_, w_[offset], cp, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        DataCopyPad(
            xi_, w_[offset + (compact ? 2U : 1U) * t_.xPlane], cp, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        Fence<HardEvent::MTE2_V>();
    }
    __aicore__ inline void LoadBatchCoefficients(uint32_t begin, uint32_t ks)
    {
        constexpr uint32_t COLS = kReferenceColumns;
        auto av = packed_[4U * RHS];
        auto iv = packed_[5U * RHS];
        Duplicate(av, 0.0f, 2U * RHS);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < ks; ++j) {
            auto a = Coefficient(begin + j);
            if (t_.right) {
                a.real *= t_.alphaReal;
                a.imag *= t_.alphaReal;
            }
            Duplicate(av[j * COLS], a.real, COLS);
            Duplicate(iv[j * COLS], a.imag, COLS);
        }
    }
    __aicore__ inline void SeedOrderedVector(bool leftN)
    {
        Duplicate(xr_, 0.0f, 2U * RHS);
        Fence<HardEvent::V_MTE2>();
        ComplexValue diagonal = Coefficient(row_);
        LoadX(row_);
        Fence<HardEvent::MTE2_V>();
        if (leftN && t_.alphaReal != 1.0f) {
            Muls(xr_, xr_, t_.alphaReal, count_);
            Muls(xi_, xi_, t_.alphaReal, count_);
            PipeBarrier<PIPE_V>();
        }
        if (t_.right) {
            diagonal.real *= t_.alphaReal;
            diagonal.imag *= t_.alphaReal;
        }
        if ((!t_.right && t_.unit) || (t_.right && IsOne(diagonal))) {
            Adds(yr_, xr_, 0.0f, count_);
            Adds(yi_, xi_, 0.0f, count_);
        } else {
            OrderedProduct(diagonal);
            Adds(yr_, tmp_, 0.0f, count_);
            Adds(yi_, tmp_[2U * RHS], 0.0f, count_);
        }
        Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline void InitVectorBuffers()
    {
        pipe_.InitBuffer(aBuf_, 2U * CHUNK * kFloatBytes);
        pipe_.InitBuffer(xBuf_, 2U * RHS * kFloatBytes);
        pipe_.InitBuffer(outBuf_, 2U * RHS * kFloatBytes);
        pipe_.InitBuffer(tmpBuf_, 4U * RHS * kFloatBytes);
        pipe_.InitBuffer(packBuf_, 8U * RHS * kFloatBytes);
        pipe_.InitBuffer(indexBuf_, (vectorReference_ ? 8U * kReferenceColumns : 8U * RHS) * kFloatBytes);
        if (vectorReference_) {
            pipe_.InitBuffer(fmaBuf_, 24U * RHS * kFloatBytes);
            pipe_.InitBuffer(fmaMaskBuf_, 8U * RHS / 8U);
        }
        ar_ = aBuf_.Get<float>();
        ai_ = ar_[CHUNK];
        xr_ = xBuf_.Get<float>();
        xi_ = xr_[RHS];
        yr_ = outBuf_.Get<float>();
        yi_ = yr_[RHS];
        tmp_ = tmpBuf_.Get<float>();
        packed_ = packBuf_.Get<float>();
        indices_ = indexBuf_.Get<uint32_t>();
    }
    __aicore__ inline void InitOutputIndices()
    {
        const uint32_t stride = t_.right ? 2U : 8U;
        const uint32_t size = stride * (vectorReference_ ? kReferenceColumns : RHS);
        CreateVecIndex(indices_.ReinterpretCast<int32_t>(), int32_t{0}, size);
        auto parity = packed_.ReinterpretCast<uint32_t>();
        Duplicate(parity.ReinterpretCast<uint16_t>(), uint16_t{1}, size * 2U);
        PipeBarrier<PIPE_V>();
        And(parity.ReinterpretCast<uint16_t>(), indices_.ReinterpretCast<uint16_t>(),
            parity.ReinterpretCast<uint16_t>(), size * 2U);
        ShiftRight(indices_, indices_, t_.right ? 1U : 3U, size);
        PipeBarrier<PIPE_V>();
        ShiftLeft(indices_, indices_, uint32_t{2}, size);
        ShiftLeft(parity, parity, kReferencePlaneByteShift, size); // RHS * sizeof(float)
        PipeBarrier<PIPE_V>();
        Add(indices_.ReinterpretCast<int32_t>(), indices_.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(),
            size);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void InitRectangle()
    {
        pipe_.InitBuffer(rectInBuf_, 4U * RECT_SIZE * kFloatBytes);
        pipe_.InitBuffer(rectOutBuf_, 2U * RECT_SIZE * kFloatBytes);
        pipe_.InitBuffer(rectPackBuf_, 2U * RECT_SIZE * kFloatBytes);
        pipe_.InitBuffer(rectIndexBuf_, 2U * RECT_SIZE * kFloatBytes);
        if (!t_.right) {
            return;
        }
        const bool wide = t_.right && t_.q <= kCubeTile && t_.r >= kWideOutputMinRhs && t_.rp % kWideOutputWidth == 0U;
        const uint32_t width = wide ? kWideOutputWidth : t_.rp % kCubeTile == 0U ? kCubeTile : kCubeNarrowTile;
        const uint32_t rectRows = wide ? 8U : 32U;
        auto index = rectIndexBuf_.Get<uint32_t>();
        auto scratch = rectInBuf_.Get<uint32_t>();
        InitRectangleIndices<RECT_SIZE>(index, scratch, rectRows, width, t_.right);
    }
    __aicore__ inline void RunRectangle()
    {
        const bool wide = t_.right && t_.q <= kCubeTile && t_.r >= kWideOutputMinRhs && t_.rp % kWideOutputWidth == 0U;
        const uint32_t width = wide ? kWideOutputWidth : t_.rp % kCubeTile == 0U ? kCubeTile : kCubeNarrowTile;
        const uint32_t rectRows = wide ? 8U : 32U;
        const uint32_t size = rectRows * width;
        auto input = rectInBuf_.Get<float>();
        auto output = rectOutBuf_.Get<float>();
        auto packed = rectPackBuf_.Get<float>();
        auto index = rectIndexBuf_.Get<uint32_t>();
        const uint32_t columns = (t_.r + width - 1U) / width;
        const uint64_t tasks = static_cast<uint64_t>((t_.q + rectRows - 1U) / rectRows) * columns;
        for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
            const uint32_t row = task / columns * rectRows;
            const uint32_t col = task % columns * width;
            const uint32_t rows = Min(rectRows, t_.q - row);
            const uint32_t cols = Min(width, t_.r - col);
            const uint64_t offset =
                kCubeAPlanes * t_.aPlane + kCubeOutputXPlane * t_.xPlane + static_cast<uint64_t>(row) * t_.rp + col;
            for (uint32_t p = 0; p < 2U; ++p) {
                DataCopyPad(
                    input[p * RECT_SIZE], w_[offset + p * t_.xPlane],
                    DataCopyExtParams{static_cast<uint16_t>(rectRows), width * 4U, (t_.rp - width) * 4U, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
            }
            Fence<HardEvent::MTE2_V>();
            auto result = ScaleRectangle<RECT_SIZE>(input, output, size, t_);
            PackRectangle<RECT_SIZE>(packed, result, index, rectRows, width, size, t_.right);
            Fence<HardEvent::V_MTE3>();
            const uint64_t dst =
                t_.right ? static_cast<uint64_t>(row) * t_.ldc + col : static_cast<uint64_t>(col) * t_.ldc + row;
            const uint32_t span = t_.right ? cols : rows;
            const uint32_t blocks = t_.right ? rows : cols;
            const uint32_t pitch = t_.right ? width : rectRows;
            DataCopyPad(
                c_[2U * dst], packed,
                DataCopyExtParams{
                    static_cast<uint16_t>(blocks), span * 8U, (pitch - (span + 3U) / 4U * 4U) / 4U,
                    (t_.ldc - span) * 8U, 0});
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::V_MTE2>();
        }
    }
    __aicore__ inline ComplexValue Coefficient(uint32_t k)
    {
        uint32_t base = k / CHUNK * CHUNK;
        if (cached_ != base) {
            const uint64_t offset = static_cast<uint64_t>(row_) * t_.qp + base;
            uint32_t size = Min(CHUNK, t_.qp - base);
            Read(ar_, w_[offset], size);
            Read(ai_, w_[t_.aPlane + offset], size);
            Fence<HardEvent::MTE2_S>();
            cached_ = base;
        }
        return {ar_.GetValue(k - base), ai_.GetValue(k - base)};
    }
    __aicore__ inline void LoadX(uint32_t k)
    {
        // Compact Cube operands already retain both original B components.
        // Reuse them for the fallback instead of storing a second snapshot.
        const bool compact = t_.useCube;
        const uint64_t offset =
            compact ? kCubeAPlanes * t_.aPlane + 2U * t_.xPlane + static_cast<uint64_t>(2U * k) * t_.rp + col_ :
                      kOriginalAPlanes * t_.aPlane + static_cast<uint64_t>(k) * t_.rp + col_;
        Read(xr_, w_[offset], count_);
        Read(xi_, w_[offset + (compact ? 2U : 1U) * t_.xPlane], count_);
    }
    __aicore__ inline void Scale()
    {
        if (t_.alphaReal == 1.0f && t_.alphaImag == 0.0f) {
            return;
        }
        Muls(tmp_, yr_, t_.alphaReal, count_);
        Muls(tmp_[RHS], yi_, t_.alphaImag, count_);
        Muls(tmp_[2U * RHS], yr_, t_.alphaImag, count_);
        Muls(tmp_[3U * RHS], yi_, t_.alphaReal, count_);
        PipeBarrier<PIPE_V>();
        Sub(yr_, tmp_, tmp_[RHS], count_);
        Add(yi_, tmp_[2U * RHS], tmp_[3U * RHS], count_);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Vector()
    {
        Duplicate(yr_, 0.0f, RHS);
        Duplicate(yi_, 0.0f, RHS);
        PipeBarrier<PIPE_V>();
        const uint32_t begin = t_.upper ? row_ : 0U;
        const uint32_t end = t_.upper ? t_.q : row_ + 1U;
        for (uint32_t k = begin; k < end; ++k) {
            const ComplexValue a = Coefficient(k);
            LoadX(k);
            Fence<HardEvent::MTE2_V>();
            Muls(tmp_, xr_, a.real, count_);
            Muls(tmp_[RHS], xi_, a.imag, count_);
            Muls(tmp_[2U * RHS], xi_, a.real, count_);
            Muls(tmp_[3U * RHS], xr_, a.imag, count_);
            PipeBarrier<PIPE_V>();
            Sub(tmp_, tmp_, tmp_[RHS], count_);
            Add(tmp_[2U * RHS], tmp_[2U * RHS], tmp_[3U * RHS], count_);
            PipeBarrier<PIPE_V>();
            Add(yr_, yr_, tmp_, count_);
            Add(yi_, yi_, tmp_[2U * RHS], count_);
            Fence<HardEvent::V_MTE2>();
        }
        Scale();
    }
    // Real alpha: each input component is zero or [2^-24,2^16],
    // and |alpha| is [2^-8,2^8]. Scaled operands are at least 2^-32 and
    // at most 2^24. At most one FMA operand is alpha-scaled, so products
    // lie on a 2^-102 lattice and midpoint adjustments on 2^-105, all
    // normal binary32. Split intermediates and sums cannot overflow.
    // Accumulators are not FMA operands; K order and alpha placement follow Netlib.
    // Batch independent complex products across K, then add each product in
    // the original Netlib order. This amortizes software-FMA instructions;
    // it does not use a tree reduction or change alpha placement.
    __aicore__ inline void BatchedOrderedVector()
    {
        constexpr uint32_t COLS = kReferenceColumns;
        constexpr uint32_t K = RHS / COLS;
        const bool leftN = !t_.right && t_.originalTrans == ACLBLAS_OP_N;
        const bool leftT = !t_.right && t_.originalTrans != ACLBLAS_OP_N;
        const bool rightT = t_.right && t_.originalTrans != ACLBLAS_OP_N;
        const bool descending = !t_.upper && (leftN || rightT);
        SeedOrderedVector(leftN);
        const uint32_t first = t_.upper ? row_ + 1U : 0U;
        const uint32_t last = t_.upper ? t_.q : row_;
        for (uint32_t step = 0; step < last - first; step += K) {
            const uint32_t ks = Min(K, last - first - step);
            const uint32_t begin = descending ? last - step - ks : first + step;
            LoadBatchCoefficients(begin, ks);
            LoadBatchX(begin, ks);
            ComputeBatch(ks, leftN);
            for (uint32_t j = 0; j < ks; ++j) {
                const uint32_t k = descending ? ks - 1U - j : j;
                Add(yr_, yr_, tmp_[k * COLS], count_);
                Add(yi_, yi_, tmp_[RHS + k * COLS], count_);
                PipeBarrier<PIPE_V>();
            }
            Fence<HardEvent::V_MTE2>();
        }
        if (leftT && t_.alphaReal != 1.0f) {
            Muls(yr_, yr_, t_.alphaReal, count_);
            Muls(yi_, yi_, t_.alphaReal, count_);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void OrderedProduct(ComplexValue a)
    {
        const uint32_t n = (count_ + kVectorRoundUp) / kVectorFloats * kVectorFloats;
        Muls(tmp_[RHS], xi_, -a.imag, n);
        PipeBarrier<PIPE_V>();
        VectorFma(tmp_, a.real, xr_, tmp_[RHS]);
        // LEFT's Netlib branch fuses ar*bi; RIGHT fuses ai*br.
        if (!t_.right) {
            Muls(tmp_[RHS], xr_, a.imag, n);
            PipeBarrier<PIPE_V>();
            VectorFma(tmp_[2U * RHS], a.real, xi_, tmp_[RHS]);
        } else {
            Muls(tmp_[RHS], xi_, a.real, n);
            PipeBarrier<PIPE_V>();
            VectorFma(tmp_[2U * RHS], a.imag, xr_, tmp_[RHS]);
        }
    }
    // Graillat/Muller (Numer. Math. 157, 2025), Algorithms 3/4/6/9.
    // Netlib branch order matters for non-finite values. In particular a UNIT
    // diagonal is an identity operation, not a complex multiply by (1, 0).
    __aicore__ inline void Reference()
    {
        const ComplexValue alpha{t_.alphaReal, t_.alphaImag};
        const ComplexValue diagonal = Coefficient(row_);
        const bool leftN = !t_.right && t_.originalTrans == ACLBLAS_OP_N;
        const bool leftT = !t_.right && t_.originalTrans != ACLBLAS_OP_N;
        const bool rightT = t_.right && t_.originalTrans != ACLBLAS_OP_N;
        SeedReference(alpha, diagonal, leftN, leftT, rightT);
        const bool descending = !t_.upper && (leftN || rightT);
        const uint32_t first = t_.upper ? row_ + 1U : 0U;
        const uint32_t last = t_.upper ? t_.q : row_;
        for (uint32_t step = 0; step < last - first; ++step) {
            const uint32_t k = descending ? last - 1U - step : first + step;
            const ComplexValue a = Coefficient(k);
            if (t_.right && IsZero(a)) {
                continue;
            }
            const ComplexValue factor = t_.right ? ComplexProduct(alpha, a, true) : a;
            LoadX(k);
            Fence<HardEvent::MTE2_S>();
            AccumulateReference(alpha, a, factor, leftN, leftT);
        }
        if (leftT) {
            for (uint32_t j = 0; j < count_; ++j) {
                ComplexValue y = ComplexProduct(alpha, {yr_.GetValue(j), yi_.GetValue(j)}, true);
                yr_.SetValue(j, y.real);
                yi_.SetValue(j, y.imag);
            }
        }
        Fence<HardEvent::S_V>();
    }
    __aicore__ inline void Store()
    {
        const uint32_t stride = t_.right ? 2U : 8U;
        Gather(packed_, yr_, indices_, 0U, count_ * stride);
        Fence<HardEvent::V_MTE3>();
        const uint64_t offset =
            t_.right ? static_cast<uint64_t>(row_) * t_.ldc + col_ : static_cast<uint64_t>(col_) * t_.ldc + row_;
        if (t_.right) {
            Write(c_[2U * offset], packed_, count_ * 2U);
        } else {
            DataCopyPad(
                c_[2U * offset], packed_,
                DataCopyExtParams{static_cast<uint16_t>(count_), 8U, 0, (t_.ldc - 1U) * 8U, 0});
        }
        Fence<HardEvent::MTE3_V>();
    }
    CtrmmTiling t_{};
    bool exceptional_ = false;
    bool vectorReference_ = false;
    uint32_t row_ = 0;
    uint32_t col_ = 0;
    uint32_t count_ = 0;
    uint32_t cached_ = 0;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> aBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> packBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> fmaBuf_;
    TBuf<TPosition::VECCALC> fmaMaskBuf_;
    TBuf<TPosition::VECCALC> rectInBuf_;
    TBuf<TPosition::VECCALC> rectOutBuf_;
    TBuf<TPosition::VECCALC> rectPackBuf_;
    TBuf<TPosition::VECCALC> rectIndexBuf_;
    GlobalTensor<float> w_;
    GlobalTensor<float> c_;
    LocalTensor<float> ar_;
    LocalTensor<float> ai_;
    LocalTensor<float> xr_;
    LocalTensor<float> xi_;
    LocalTensor<float> yr_;
    LocalTensor<float> yi_;
    LocalTensor<float> tmp_;
    LocalTensor<float> packed_;
    LocalTensor<uint32_t> indices_;
};

} // namespace
