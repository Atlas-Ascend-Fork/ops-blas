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

namespace {
// Small LEFT matrices fit in UB. Each AIV owns two (Q=8) or four
// adjacent output rows; packed columns use 32-byte UB slots.
template <uint32_t Q>
class CtrmmSmall {
    static constexpr uint32_t ROWS = Q == 8U ? 2U : 4U;
    static constexpr uint32_t PACK_ROWS = 4U;
    static constexpr uint32_t COLS = Q <= 16U ? 8U : Q <= 32U ? 16U : 32U;
    static constexpr uint32_t OUT = ROWS * COLS;
    static constexpr uint32_t U = Q * COLS;
    static constexpr uint32_t A_SIZE = ROWS * Q;
    static constexpr uint32_t LOG_Q = Q == 8 ? 3U : Q == 16 ? 4U : Q == 32 ? 5U : Q == 64 ? 6U : 7U;
    static constexpr uint32_t MASK_PITCH = U / 8U < 32U ? 32U : U / 8U;

public:
    struct Views {
        LocalTensor<float> ra;
        LocalTensor<float> rb;
        LocalTensor<float> ar;
        LocalTensor<float> ai;
        LocalTensor<float> br;
        LocalTensor<float> bi;
        LocalTensor<float> tmp;
        LocalTensor<uint32_t> index;
        LocalTensor<uint32_t> scratch;
        LocalTensor<float> out;
        LocalTensor<float> packed;
    };

    __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR c, const CtrmmTiling& t)
    {
        t_ = t;
        const uint32_t rowBlocks = (t_.q + ROWS - 1U) / ROWS;
        row_ = GetBlockIdx() % rowBlocks * ROWS;
        col_ = GetBlockIdx() / rowBlocks * COLS;
        rows_ = Min(ROWS, t_.q - row_);
        cols_ = Min(COLS, t_.r - col_);
        InitBuffers();
        const auto v = MakeViews();
        GlobalTensor<float> ag;
        GlobalTensor<float> bg;
        GlobalTensor<float> cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
        Duplicate(v.ra, 0.0f, 8U * Q * ROWS + 2U * U);
        Fence<HardEvent::V_MTE2>();
        LoadA(v, ag);
        LoadB(v, bg);
        SplitInputs(v);
        const bool sensitive = SensitiveInputs(v);
        bool directPacked = false;
        if (sensitive) {
            Reference(v.ar, v.ai, v.br, v.bi, v.out);
        } else {
            directPacked = ComputeOrdinary(v);
        }
        PackOutput(v, directPacked);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(
            cg[2U * (static_cast<uint64_t>(col_) * t_.ldc + row_)], v.packed,
            DataCopyExtParams{static_cast<uint16_t>(cols_), rows_ * 8U, 0, (t_.ldc - rows_) * 8U, 0});
        Fence<HardEvent::MTE3_V>();
    }

private:
    __aicore__ inline Views MakeViews()
    {
        auto ra = rawBuf_.Get<float>();
        auto rb = ra[8U * Q * ROWS];
        auto ar = dataBuf_.Get<float>();
        auto ai = ar[A_SIZE];
        auto br = ar[2U * A_SIZE];
        auto bi = ar[2U * A_SIZE + U];
        auto tmp = tmpBuf_.Get<float>();
        auto index = indexBuf_.Get<uint32_t>();
        auto scratch = index[U];
        auto out = outBuf_.Get<float>();
        auto packed = packBuf_.Get<float>();
        return {ra, rb, ar, ai, br, bi, tmp, index, scratch, out, packed};
    }

    __aicore__ inline void LoadA(const Views& v, GlobalTensor<float>& ag)
    {
        if (t_.transpose) {
            for (uint32_t rr = 0; rr < rows_; ++rr) {
                const uint32_t row = row_ + rr;
                const uint32_t begin = t_.upper ? row + t_.unit : 0U;
                const uint32_t end = t_.upper ? t_.q : row + 1U - t_.unit;
                if (end <= begin) {
                    continue;
                }
                const uint32_t left = (begin % 4U) * 2U;
                const uint32_t right = (8U - (left + (end - begin) * 2U) % 8U) % 8U;
                DataCopyPad(
                    v.ra[rr * Q * 2U + begin / 4U * 8U], ag[2U * (static_cast<uint64_t>(row) * t_.lda + begin)],
                    DataCopyExtParams{1, (end - begin) * 8U, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, static_cast<uint8_t>(left), static_cast<uint8_t>(right), 0.0f});
            }
        } else {
            // Entire off-diagonal columns contain four contiguous coefficients.
            const uint32_t begin = t_.upper ? row_ + rows_ : 0U;
            const uint32_t end = t_.upper ? t_.q : row_;
            if (end > begin) {
                DataCopyPad(
                    v.ra[begin * 8U], ag[2U * (static_cast<uint64_t>(begin) * t_.lda + row_)],
                    DataCopyExtParams{static_cast<uint16_t>(end - begin), rows_ * 8U, (t_.lda - rows_) * 8U, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((ROWS - rows_) * 2U), 0.0f});
            }
            for (uint32_t k = row_; k < row_ + rows_; ++k) {
                const uint32_t first = t_.upper ? 0U : k - row_ + t_.unit;
                const uint32_t last = t_.upper ? k - row_ + 1U - t_.unit : rows_;
                if (last <= first) {
                    continue;
                }
                DataCopyPad(
                    v.ra[k * 8U], ag[2U * (static_cast<uint64_t>(k) * t_.lda + row_ + first)],
                    DataCopyExtParams{1, (last - first) * 8U, 0, 0, 0},
                    DataCopyPadExtParams<float>{
                        true, static_cast<uint8_t>(first * 2U), static_cast<uint8_t>((ROWS - last) * 2U), 0.0f});
            }
        }
    }

    __aicore__ inline void LoadB(const Views& v, GlobalTensor<float>& bg)
    {
        const uint32_t size = Q * cols_;
        if (t_.q == Q && t_.ldb == Q) {
            Read(v.rb, bg[2U * static_cast<uint64_t>(col_) * t_.ldb], 2U * size);
        } else {
            DataCopyPad(
                v.rb, bg[2U * static_cast<uint64_t>(col_) * t_.ldb],
                DataCopyExtParams{
                    static_cast<uint16_t>(cols_), t_.q * 8U, (t_.ldb - t_.q) * 8U, (Q - (t_.q + 3U) / 4U * 4U) / 4U, 0},
                DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((4U - t_.q % 4U) % 4U * 2U), 0.0f});
        }
        Fence<HardEvent::MTE2_V>();
    }

    __aicore__ inline void SplitA(const Views& v, uint64_t& selected)
    {
        if (t_.transpose) {
            GatherMask(v.ar, v.ra, uint8_t{1}, true, 2U * A_SIZE, GatherMaskParams{1, 1, 8, 0}, selected);
            GatherMask(v.ai, v.ra, uint8_t{2}, true, 2U * A_SIZE, GatherMaskParams{1, 1, 8, 0}, selected);
        } else {
            CreateVecIndex(v.index.template ReinterpretCast<int32_t>(), int32_t{0}, A_SIZE);
            PipeBarrier<PIPE_V>();
            auto low = v.tmp.template ReinterpretCast<uint32_t>();
            Duplicate(low.template ReinterpretCast<int32_t>(), int32_t{Q - 1U}, A_SIZE);
            PipeBarrier<PIPE_V>();
            And(low.template ReinterpretCast<uint16_t>(), v.index.template ReinterpretCast<uint16_t>(),
                low.template ReinterpretCast<uint16_t>(), A_SIZE * 2U);
            ShiftRight(v.scratch, v.index, uint32_t{LOG_Q}, A_SIZE);
            PipeBarrier<PIPE_V>();
            ShiftLeft(low, low, uint32_t{5}, A_SIZE);
            ShiftLeft(v.scratch, v.scratch, uint32_t{3}, A_SIZE);
            PipeBarrier<PIPE_V>();
            Add(v.scratch.template ReinterpretCast<int32_t>(), v.scratch.template ReinterpretCast<int32_t>(),
                low.template ReinterpretCast<int32_t>(), A_SIZE);

            PipeBarrier<PIPE_V>();
            Gather(v.ar, v.ra, v.scratch, 0U, A_SIZE);
            Adds(
                v.scratch.template ReinterpretCast<int32_t>(), v.scratch.template ReinterpretCast<int32_t>(),
                int32_t{4}, A_SIZE);
            PipeBarrier<PIPE_V>();
            Gather(v.ai, v.ra, v.scratch, 0U, A_SIZE);
        }
    }

    __aicore__ inline void SplitInputs(const Views& v)
    {
        uint64_t selected = 0;
        SplitA(v, selected);
        const GatherMaskParams bParams{1, static_cast<uint16_t>(2U * U / kVectorFloats), 8, 0};
        GatherMask(v.br, v.rb, uint8_t{1}, false, 0U, bParams, selected);
        GatherMask(v.bi, v.rb, uint8_t{2}, false, 0U, bParams, selected);
        PipeBarrier<PIPE_V>();
        if (t_.conjugate) {
            Muls(v.ai, v.ai, -1.0f, A_SIZE);
            PipeBarrier<PIPE_V>();
        }
        if (t_.unit) {
            for (uint32_t rr = 0; rr < rows_; ++rr) {
                uint64_t diagonalMask[2] = {uint64_t{1} << ((row_ + rr) % kVectorFloats), 0U};
                Duplicate(v.ar[rr * Q + (row_ + rr) / kVectorFloats * kVectorFloats], 1.0f, diagonalMask, 1, 1, 8);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline bool SensitiveInputs(const Views& v)
    {
        bool sensitive = (t_.fastInputLimit == 0.0f) ||
                         ((ctrmm_ieee::BitsOf(t_.alphaReal) & ctrmm_ieee::kMagnitudeMask) > kFastAlphaLimitBits ||
                          (ctrmm_ieee::BitsOf(t_.alphaImag) & ctrmm_ieee::kMagnitudeMask) > kFastAlphaLimitBits);
        // All four input planes are adjacent and fully initialized. One
        // comparison covers A and B, avoiding four scalar/vector round trips.
        constexpr uint32_t checked = 2U * A_SIZE + 2U * U;
        Abs(v.tmp, v.ar, checked);
        PipeBarrier<PIPE_V>();
        auto mask = maskBuf_.Get<uint8_t>();
        constexpr uint32_t alignedChecked = (checked + kVectorRoundUp) / kVectorFloats * kVectorFloats;
        if constexpr (alignedChecked != checked) {
            Duplicate(v.tmp[checked], 0.0f, alignedChecked - checked);
            PipeBarrier<PIPE_V>();
        }
        CompareScalar(mask, v.tmp, t_.fastInputLimit, CMPMODE::LE, alignedChecked);
        PipeBarrier<PIPE_V>();
        auto maskValues = rawBuf_.Get<half>();
        Cast(maskValues, mask, RoundMode::CAST_NONE, checked / 8U);
        PipeBarrier<PIPE_V>();
        auto minimum = v.out.template ReinterpretCast<half>();
        ReduceMin(minimum, maskValues, maskValues[(checked / 8U + 31U) / 32U * 32U], checked / 8U, false);
        Fence<HardEvent::V_S>();
        sensitive |= static_cast<float>(minimum.GetValue(0)) != kFullByteMaskValue;
        if (!sensitive && CorrelatedComponents<true>(v.ar, v.ai, rawBuf_.Get<float>(), A_SIZE)) {
            sensitive = true;
        }
        return sensitive;
    }

    __aicore__ inline void ReduceRow(const Views& v, uint32_t rr, uint32_t kBegin, uint32_t kEnd, bool directPacked)
    {
        if (directPacked) {
            // dstRepStride counts float elements. Reduce directly to
            // column-major complex pairs, avoiding a final UB transpose.
            if constexpr (Q <= kVectorFloats) {
                WholeReduceSum(v.packed[rr * 2U], v.tmp, Q, cols_, PACK_ROWS * 2U, 1, Q / 8U);
                WholeReduceSum(v.packed[rr * 2U + 1U], v.tmp[2U * U], Q, cols_, PACK_ROWS * 2U, 1, Q / 8U);
            } else {
                auto partial = v.index.template ReinterpretCast<float>();
                WholeReduceSum(v.packed[rr * 2U], v.tmp[kBegin], kVectorFloats, cols_, PACK_ROWS * 2U, 1, Q / 8U);
                WholeReduceSum(
                    v.packed[rr * 2U + 1U], v.tmp[2U * U + kBegin], kVectorFloats, cols_, PACK_ROWS * 2U, 1, Q / 8U);
                if (kEnd - kBegin > kVectorFloats) {
                    WholeReduceSum(
                        partial[rr * 2U], v.tmp[kVectorFloats], kVectorFloats, cols_, PACK_ROWS * 2U, 1, Q / 8U);
                    WholeReduceSum(
                        partial[rr * 2U + 1U], v.tmp[2U * U + kVectorFloats], kVectorFloats, cols_, PACK_ROWS * 2U, 1,
                        Q / 8U);
                }
                PipeBarrier<PIPE_V>();
            }
        } else {
            if constexpr (Q <= kVectorFloats) {
                WholeReduceSum(v.out[rr * COLS], v.tmp, Q, cols_, 1, 1, Q / 8U);
                WholeReduceSum(v.out[OUT + rr * COLS], v.tmp[2U * U], Q, cols_, 1, 1, Q / 8U);
            } else {
                auto partial = v.index.template ReinterpretCast<float>();
                WholeReduceSum(v.out[rr * COLS], v.tmp[kBegin], kVectorFloats, cols_, 1, 1, Q / 8U);
                WholeReduceSum(v.out[OUT + rr * COLS], v.tmp[2U * U + kBegin], kVectorFloats, cols_, 1, 1, Q / 8U);
                if (kEnd - kBegin > kVectorFloats) {
                    WholeReduceSum(partial, v.tmp[kVectorFloats], kVectorFloats, cols_, 1, 1, Q / 8U);
                    WholeReduceSum(partial[COLS], v.tmp[2U * U + kVectorFloats], kVectorFloats, cols_, 1, 1, Q / 8U);
                    PipeBarrier<PIPE_V>();
                    Add(v.out[rr * COLS], v.out[rr * COLS], partial, cols_);
                    Add(v.out[OUT + rr * COLS], v.out[OUT + rr * COLS], partial[COLS], cols_);
                }
            }
        }
    }

    __aicore__ inline void MultiplyRow(
        const Views& v, uint32_t rr, uint32_t kBegin, uint32_t kEnd, LocalTensor<float> negativeAi,
        const BinaryRepeatParams& repeat)
    {
        const uint32_t size = Q * cols_;
        if constexpr (Q <= kVectorFloats) {
            Mul(v.tmp, v.ar[rr * Q], v.br, Q, cols_, repeat);
            Mul(v.tmp[U], v.ai[rr * Q], v.bi, Q, cols_, repeat);
            Mul(v.tmp[2U * U], v.ar[rr * Q], v.bi, Q, cols_, repeat);
            Mul(v.tmp[3U * U], v.ai[rr * Q], v.br, Q, cols_, repeat);
        } else {
            for (uint32_t half = kBegin; half < kEnd; half += kVectorFloats) {
                Mul(v.tmp[half], v.ar[rr * Q + half], v.br[half], kVectorFloats, cols_, repeat);
                Mul(v.tmp[2U * U + half], v.ar[rr * Q + half], v.bi[half], kVectorFloats, cols_, repeat);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t half = kBegin; half < kEnd; half += kVectorFloats) {
                MulAddDst(v.tmp[half], negativeAi[rr * Q + half], v.bi[half], kVectorFloats, cols_, repeat);
                MulAddDst(v.tmp[2U * U + half], v.ai[rr * Q + half], v.br[half], kVectorFloats, cols_, repeat);
            }
        }
        PipeBarrier<PIPE_V>();
        if constexpr (Q <= kVectorFloats) {
            Sub(v.tmp, v.tmp, v.tmp[U], size);
            Add(v.tmp[2U * U], v.tmp[2U * U], v.tmp[3U * U], size);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ScaleOutput(const Views& v, bool directPacked)
    {
        if (!directPacked) {
            Muls(v.tmp, v.out, t_.alphaReal, OUT);
            Muls(v.tmp[OUT], v.out[OUT], t_.alphaImag, OUT);
            Muls(v.tmp[2U * OUT], v.out, t_.alphaImag, OUT);
            Muls(v.tmp[3U * OUT], v.out[OUT], t_.alphaReal, OUT);
            PipeBarrier<PIPE_V>();
            Sub(v.out, v.tmp, v.tmp[OUT], OUT);
            Add(v.out[OUT], v.tmp[2U * OUT], v.tmp[3U * OUT], OUT);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline bool ComputeOrdinary(const Views& v)
    {
        bool directPacked;
        directPacked = t_.alphaReal == 1.0f && t_.alphaImag == 0.0f;
        if (!directPacked) {
            Duplicate(v.out, 0.0f, 2U * OUT);
            PipeBarrier<PIPE_V>();
        }
        auto negativeAi = rawBuf_.Get<float>();
        if constexpr (Q > kVectorFloats) {
            Muls(negativeAi, v.ai, -1.0f, A_SIZE);
            PipeBarrier<PIPE_V>();
        }
        const uint32_t kBegin = Q > kVectorFloats && t_.upper && row_ >= kVectorFloats ? kVectorFloats : 0U;
        const uint32_t kEnd = Q > kVectorFloats && !t_.upper && row_ + rows_ <= kVectorFloats ? kVectorFloats : Q;
        const BinaryRepeatParams repeat{1, 1, 1, Q / 8U, 0, Q / 8U};
        for (uint32_t rr = 0; rr < ROWS; ++rr) {
            if (rr >= rows_) {
                break;
            }
            MultiplyRow(v, rr, kBegin, kEnd, negativeAi, repeat);
            ReduceRow(v, rr, kBegin, kEnd, directPacked);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (Q > kVectorFloats) {
            if (directPacked && kEnd - kBegin > kVectorFloats) {
                Add(v.packed, v.packed, v.index.template ReinterpretCast<float>(), 2U * PACK_ROWS * cols_);
                PipeBarrier<PIPE_V>();
            }
        }
        ScaleOutput(v, directPacked);
        return directPacked;
    }

    __aicore__ inline void PackOutput(const Views& v, bool directPacked)
    {
        if (!directPacked) {
            const uint32_t packedSize = 2U * PACK_ROWS * cols_;
            CreateVecIndex(v.index.template ReinterpretCast<int32_t>(), int32_t{0}, packedSize);
            Duplicate(v.scratch.template ReinterpretCast<int32_t>(), int32_t{1}, packedSize);
            PipeBarrier<PIPE_V>();
            And(v.scratch.template ReinterpretCast<uint16_t>(), v.index.template ReinterpretCast<uint16_t>(),
                v.scratch.template ReinterpretCast<uint16_t>(), packedSize * 2U);
            ShiftRight(v.index, v.index, uint32_t{1}, packedSize);
            PipeBarrier<PIPE_V>();
            auto rowIndex = v.tmp.template ReinterpretCast<uint32_t>();
            Duplicate(rowIndex.template ReinterpretCast<int32_t>(), int32_t{ROWS - 1U}, packedSize);
            PipeBarrier<PIPE_V>();
            And(rowIndex.template ReinterpretCast<uint16_t>(), v.index.template ReinterpretCast<uint16_t>(),
                rowIndex.template ReinterpretCast<uint16_t>(), packedSize * 2U);
            ShiftRight(v.index, v.index, uint32_t{2}, packedSize);
            PipeBarrier<PIPE_V>();
            ShiftLeft(rowIndex, rowIndex, uint32_t{COLS == 8U ? 3U : COLS == 16U ? 4U : 5U}, packedSize);
            PipeBarrier<PIPE_V>();
            Add(v.index.template ReinterpretCast<int32_t>(), v.index.template ReinterpretCast<int32_t>(),
                rowIndex.template ReinterpretCast<int32_t>(), packedSize);
            PipeBarrier<PIPE_V>();
            ShiftLeft(v.index, v.index, uint32_t{2}, packedSize);
            ShiftLeft(
                v.scratch, v.scratch,
                uint32_t{
                    ROWS == 2U  ? 6U :
                    COLS == 8U  ? 7U :
                    COLS == 16U ? 8U :
                                  9U},
                packedSize);
            PipeBarrier<PIPE_V>();
            Add(v.index.template ReinterpretCast<int32_t>(), v.index.template ReinterpretCast<int32_t>(),
                v.scratch.template ReinterpretCast<int32_t>(), packedSize);
            PipeBarrier<PIPE_V>();
            Gather(v.packed, v.out, v.index, 0U, packedSize);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        pipe_.InitBuffer(rawBuf_, (8U * Q * ROWS + 2U * Q * COLS + kVectorFloats) * kFloatBytes);
        pipe_.InitBuffer(dataBuf_, (2U * A_SIZE + 2U * U) * kFloatBytes);
        pipe_.InitBuffer(tmpBuf_, 4U * U * kFloatBytes);
        pipe_.InitBuffer(indexBuf_, 2U * U * kFloatBytes);
        pipe_.InitBuffer(outBuf_, 2U * OUT * kFloatBytes);
        pipe_.InitBuffer(packBuf_, 2U * PACK_ROWS * COLS * kFloatBytes);
        pipe_.InitBuffer(maskBuf_, 4U * MASK_PITCH);
    }

    __aicore__ inline void Reference(
        LocalTensor<float> ar, LocalTensor<float> ai, LocalTensor<float> br, LocalTensor<float> bi,
        LocalTensor<float> out)
    {
        const ComplexValue alpha{t_.alphaReal, t_.alphaImag};

        const bool normal = t_.originalTrans == ACLBLAS_OP_N;
        for (uint32_t col = 0; col < cols_; ++col) {
            for (uint32_t rr = 0; rr < rows_; ++rr) {
                const uint32_t row = row_ + rr;
                const uint32_t aBase = rr * Q;
                const ComplexValue diagonal{ar.GetValue(aBase + row), ai.GetValue(aBase + row)};
                const uint32_t base = col * Q;
                const ComplexValue x{br.GetValue(base + row), bi.GetValue(base + row)};
                ComplexValue y;
                if (normal) {
                    y = IsZero(x) ? x : ComplexProduct(alpha, x, true);
                    if (!IsZero(x) && !t_.unit) {
                        y = ComplexProduct(y, diagonal);
                    }
                } else {
                    y = t_.unit ? x : ComplexProduct(x, diagonal);
                }
                const uint32_t begin = t_.upper ? row + 1U : 0U;
                const uint32_t end = t_.upper ? t_.q : row;
                for (uint32_t step = 0; step < end - begin; ++step) {
                    const uint32_t k = normal && !t_.upper ? end - 1U - step : begin + step;
                    ComplexValue b{br.GetValue(base + k), bi.GetValue(base + k)};
                    if (normal && IsZero(b)) {
                        continue;
                    }
                    ComplexValue a{ar.GetValue(aBase + k), ai.GetValue(aBase + k)};
                    ComplexValue term =
                        normal ? ComplexProduct(ComplexProduct(alpha, b, true), a) : ComplexProduct(a, b, true);
                    y.real += term.real;
                    y.imag += term.imag;
                }
                if (!normal) {
                    y = ComplexProduct(alpha, y, true);
                }
                out.SetValue(rr * COLS + col, y.real);
                out.SetValue(OUT + rr * COLS + col, y.imag);
            }
        }
        Fence<HardEvent::S_V>();
    }
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> packBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    CtrmmTiling t_{};
    uint32_t row_ = 0;
    uint32_t col_ = 0;
    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
};

} // namespace
