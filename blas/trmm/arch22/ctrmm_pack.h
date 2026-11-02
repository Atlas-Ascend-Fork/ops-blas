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
#include "ctrmm_input_check.h"

namespace {
template <uint32_t ShortQ = 0U>
class CtrmmPack {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR ws, const CtrmmTiling& t)
    {
        t_ = t;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
        pipe_.InitBuffer(rawBuf_, 2U * RECT_SIZE * kFloatBytes);
        pipe_.InitBuffer(realBuf_, CHUNK * kFloatBytes);
        pipe_.InitBuffer(imagBuf_, CHUNK * kFloatBytes);
        pipe_.InitBuffer(indexBuf_, CHUNK * 2U * kFloatBytes);
        pipe_.InitBuffer(absBuf_, kInputAbsBytes);
        pipe_.InitBuffer(maskBuf_, kInputMaskBytes);
        pipe_.InitBuffer(rectBuf_, 2U * RECT_SIZE * kFloatBytes);
        raw_ = rawBuf_.Get<float>();
        real_ = realBuf_.Get<float>();
        imag_ = imagBuf_.Get<float>();
        index_ = indexBuf_.Get<uint32_t>();
        Duplicate(maskBuf_.Get<uint16_t>()[kFastMaskHalfOffset], kFullHalfWordMask, kInputMaskHalfWords);
        Duplicate(maskBuf_.Get<uint16_t>()[kFmaMaskHalfOffset], kFullHalfWordMask, kInputMaskHalfWords);
        PipeBarrier<PIPE_V>();
        if (UseFourRows()) {
            SetFourRowIndices();
            SetExpandedIndices();
        } else {
            SetIndices(t_.transpose ? 3U : 5U, 0);
        }
        // LEFT uses native b32 transpose address lists.
    }
    __aicore__ inline void Run()
    {
        uint32_t bad = (t_.fastInputLimit == 0.0f || t_.q < kCorrelationMinElements) ? kReferenceRequired : 0U;
        if (UseFourRows()) {
            bad |= PackFourRows();
        } else {
            bad |= PackRows(bad);
        }
        bad |= PackX();
        if (!AllMaskBits(maskBuf_.Get<uint8_t>()[kFastMaskOffset], kInputMaskElements, absBuf_)) {
            bad |= kReferenceRequired;
        }
        const uint32_t alphaBits = ctrmm_ieee::BitsOf(t_.alphaReal) & ctrmm_ieee::kMagnitudeMask;
        if (t_.alphaImag != 0.0f || alphaBits < ctrmm_ieee::BitsOf(kVectorFmaMinAlpha) ||
            alphaBits > ctrmm_ieee::BitsOf(kVectorFmaMaxAlpha) ||
            !AllMaskBits(maskBuf_.Get<uint8_t>()[kFmaMaskOffset], kInputMaskElements, absBuf_)) {
            bad |= kScalarFmaRequired;
        }
        // Each producer writes one disjoint word. The next kernel launch
        // is the global visibility boundary for these flags and matrix data.
        auto flag = rawBuf_.Get<uint32_t>();
        flag.SetValue(0, bad);
        Fence<HardEvent::S_MTE3>();
        GlobalTensor<uint32_t> flags;
        flags.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ float*>(w_.GetPhyAddr())) + t_.flagsOffset);
        DataCopyPad(flags[GetBlockIdx()], flag, DataCopyExtParams{1, 4, 0, 0, 0});
        Fence<HardEvent::MTE3_S>();
    }

private:
    __aicore__ inline uint32_t CheckFinite(LocalTensor<float> x, uint32_t count)
    {
        return CheckInputFinite(x, count, raw_, maskBuf_, t_.fastInputLimit);
    }
    __aicore__ inline uint32_t PackRows(uint32_t bad)
    {
        for (uint32_t i = GetBlockIdx(); i < PaddedQ(); i += GetBlockNum()) {
            ZeroRow(i, PaddedQ(), 0, t_.aPlane);
            if (i < t_.q) {
                const uint32_t begin = t_.upper ? i + t_.unit : 0U;
                const uint32_t end = t_.upper ? t_.q : i + 1U - t_.unit;
                for (uint32_t k = begin; k < end; k += CHUNK) {
                    uint32_t count = Min(CHUNK, end - k);
                    const uint64_t src =
                        t_.transpose ? static_cast<uint64_t>(i) * t_.lda + k : static_cast<uint64_t>(k) * t_.lda + i;
                    LoadComplex(a_, src, t_.transpose ? 1U : t_.lda, count, 0);
                    bad |= CheckFinite(real_, count) | CheckFinite(imag_, count);
                    if (!(bad & kReferenceRequired) && CorrelatedComponents(real_, imag_, raw_, count)) {
                        bad |= kReferenceRequired;
                    }
                    if (t_.conjugate) {
                        Muls(imag_, imag_, -1.0f, count);
                        PipeBarrier<PIPE_V>();
                    }
                    StorePair(static_cast<uint64_t>(i) * PaddedQ() + k, t_.aPlane, count);
                }
                if (t_.unit) {
                    Duplicate(real_, 1.0f, 8);
                    Fence<HardEvent::V_MTE3>();
                    Write(w_[static_cast<uint64_t>(i) * PaddedQ() + i], real_, 1);
                    Fence<HardEvent::MTE3_V>();
                }
            }
        }
        return bad;
    }
    __aicore__ inline void LoadFourRows(uint32_t base, uint32_t rows)
    {
        if (t_.transpose) {
            for (uint32_t rr = 0; rr < rows; ++rr) {
                const uint32_t row = base + rr;
                const uint32_t begin = t_.upper ? row + t_.unit : 0U;
                const uint32_t end = t_.upper ? t_.q : row + 1U - t_.unit;
                if (end <= begin) {
                    continue;
                }
                const uint32_t left = begin % 4U * 2U;
                const uint32_t right = (8U - (left + (end - begin) * 2U) % 8U) % 8U;
                DataCopyPad(
                    raw_[rr * PaddedQ() * 2U + begin / 4U * 8U], a_[2U * (static_cast<uint64_t>(row) * t_.lda + begin)],
                    DataCopyExtParams{1, (end - begin) * 8U, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, static_cast<uint8_t>(left), static_cast<uint8_t>(right), 0.0f});
            }
        } else if (rows) {
            const uint32_t begin = t_.upper ? base + rows : 0U;
            const uint32_t end = t_.upper ? t_.q : base;
            if (end > begin) {
                DataCopyPad(
                    raw_[begin * 8U], a_[2U * (static_cast<uint64_t>(begin) * t_.lda + base)],
                    DataCopyExtParams{static_cast<uint16_t>(end - begin), rows * 8U, (t_.lda - rows) * 8U, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((4U - rows) * 2U), 0.0f});
            }
            for (uint32_t k = base; k < base + rows; ++k) {
                const uint32_t first = t_.upper ? 0U : k - base + t_.unit;
                const uint32_t last = t_.upper ? k - base + 1U - t_.unit : rows;
                if (last <= first) {
                    continue;
                }
                DataCopyPad(
                    raw_[k * 8U], a_[2U * (static_cast<uint64_t>(k) * t_.lda + base + first)],
                    DataCopyExtParams{1, (last - first) * 8U, 0, 0, 0},
                    DataCopyPadExtParams<float>{
                        true, static_cast<uint8_t>(first * 2U), static_cast<uint8_t>((4U - last) * 2U), 0.0f});
            }
        }
    }
    __aicore__ inline void LoadXRectangle(
        uint32_t row, uint32_t col, uint32_t rows, uint32_t cols, uint32_t width, uint32_t rectRows)
    {
        if (rows && cols) {
            const uint64_t offset =
                Right() ? static_cast<uint64_t>(row) * t_.ldb + col : static_cast<uint64_t>(col) * t_.ldb + row;
            const uint32_t span = Right() ? cols : rows;
            const uint32_t blocks = Right() ? rows : cols;
            const uint32_t pitch = Right() ? width : rectRows;
            DataCopyPad(
                raw_, b_[offset * 2U],
                DataCopyExtParams{
                    static_cast<uint16_t>(blocks), span * 8U, (t_.ldb - span) * 8U,
                    (pitch - (span + 3U) / 4U * 4U) / 4U, 0},
                DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((4U - span % 4U) % 4U * 2U), 0.0f});
        }
    }
    __aicore__ inline void SplitXRectangle(
        LocalTensor<float> re, LocalTensor<float> im, uint32_t size, uint32_t width, uint32_t rectRows)
    {
        if (Right()) {
            uint64_t selected = 0;
            const GatherMaskParams params{1, static_cast<uint16_t>(2U * size / 64U), 8, 0};
            GatherMask(re, raw_, uint8_t{1}, false, 0U, params, selected);
            GatherMask(im, raw_, uint8_t{2}, false, 0U, params, selected);
        } else {
            for (uint32_t rb = 0; rb < rectRows; rb += 4U) {
                uint64_t srcList[16];
                uint64_t dstList[16];
#pragma unroll
                for (uint32_t c = 0; c < 16U; ++c) {
                    srcList[c] = reinterpret_cast<uint64_t>(raw_[2U * c * rectRows + 2U * rb].GetPhyAddr());
                }
#pragma unroll
                for (uint32_t lane = 0; lane < 8U; ++lane) {
                    auto plane = (lane & 1U) ? im : re;
                    const uint32_t off = (rb + lane / 2U) * width;
                    dstList[2U * lane] = reinterpret_cast<uint64_t>(plane[off].GetPhyAddr());
                    dstList[2U * lane + 1U] = reinterpret_cast<uint64_t>(plane[off + 8U].GetPhyAddr());
                }
                TransDataTo5HD<float>(
                    dstList, srcList, TransDataTo5HDParams(false, false, width / 16U, 2U, 4U * rectRows));
            }
        }
    }
    __aicore__ inline void StoreXRectangle(
        LocalTensor<float> re, LocalTensor<float> im, uint32_t row, uint32_t col, uint32_t width, uint32_t rectRows,
        uint32_t size)
    {
        Fence<HardEvent::V_MTE3>();
        const uint64_t offset = kOriginalAPlanes * t_.aPlane + static_cast<uint64_t>(row) * t_.rp + col;
        const DataCopyExtParams copy{static_cast<uint16_t>(rectRows), width * 4U, 0, (t_.rp - width) * 4U, 0};
        if (!t_.useCube) {
            DataCopyPad(w_[offset], re, copy);
            DataCopyPad(w_[offset + t_.xPlane], im, copy);
        }
        if (t_.useCube) {
            Muls(raw_, im, -1.0f, size);
            Fence<HardEvent::V_MTE3>();
            const uint64_t br =
                kCubeAPlanes * t_.aPlane + 2U * t_.xPlane + static_cast<uint64_t>(2U * row) * t_.rp + col;
            const uint64_t bi = br + 2U * t_.xPlane;
            const DataCopyExtParams compact{
                static_cast<uint16_t>(rectRows), width * 4U, 0, (2U * t_.rp - width) * 4U, 0};
            DataCopyPad(w_[br], re, compact);
            DataCopyPad(w_[br + t_.rp], raw_, compact);
            DataCopyPad(w_[bi], im, compact);
            DataCopyPad(w_[bi + t_.rp], re, compact);
        }
        Fence<HardEvent::MTE3_V>();
    }
    __aicore__ inline uint32_t PaddedQ() const { return ShortQ ? ShortQ : t_.qp; }
    __aicore__ inline bool Right() const { return ShortQ ? true : t_.right != 0U; }
    // Four adjacent A rows share DMA requests and one pair of validation
    // reductions. The 64/128-column padded tiles fit the existing buffers.
    __aicore__ inline bool UseFourRows() const { return ShortQ || (t_.useCube && t_.q <= kCompactMaxOrder); }
    __aicore__ inline void SetFourRowIndices()
    {
        const uint32_t size = 4U * PaddedQ();
        auto scratch = absBuf_.Get<uint32_t>();
        CreateVecIndex(index_.ReinterpretCast<int32_t>(), int32_t{0}, size);
        PipeBarrier<PIPE_V>();
        if (t_.transpose) {
            ShiftLeft(index_, index_, uint32_t{3}, size);
        } else if (PaddedQ() == 192U) {
            for (uint32_t rr = 0; rr < 4U; ++rr) {
                CreateVecIndex(index_[rr * PaddedQ()].template ReinterpretCast<int32_t>(), int32_t{0}, PaddedQ());
                PipeBarrier<PIPE_V>();
                ShiftLeft(index_[rr * PaddedQ()], index_[rr * PaddedQ()], uint32_t{5}, PaddedQ());
                PipeBarrier<PIPE_V>();
                Adds(
                    index_[rr * PaddedQ()].template ReinterpretCast<int32_t>(),
                    index_[rr * PaddedQ()].template ReinterpretCast<int32_t>(), static_cast<int32_t>(rr * 8U),
                    PaddedQ());
            }
        } else {
            Duplicate(scratch.ReinterpretCast<int32_t>(), static_cast<int32_t>(PaddedQ() - 1U), size);
            PipeBarrier<PIPE_V>();
            And(scratch.ReinterpretCast<uint16_t>(), index_.ReinterpretCast<uint16_t>(),
                scratch.ReinterpretCast<uint16_t>(), size * 2U);
            ShiftRight(index_, index_, PaddedQ() == 64U ? 6U : PaddedQ() == 128U ? 7U : 8U, size);
            PipeBarrier<PIPE_V>();
            ShiftLeft(scratch, scratch, uint32_t{5}, size);
            ShiftLeft(index_, index_, uint32_t{3}, size);
            PipeBarrier<PIPE_V>();
            Add(index_.ReinterpretCast<int32_t>(), index_.ReinterpretCast<int32_t>(),
                scratch.ReinterpretCast<int32_t>(), size);
        }
        PipeBarrier<PIPE_V>();
        Adds(index_[CHUNK].ReinterpretCast<int32_t>(), index_.ReinterpretCast<int32_t>(), int32_t{4}, size);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void SetExpandedIndices()
    {
        const uint32_t size = 4U * PaddedQ();
        auto idx = rectBuf_.Get<uint32_t>()[RECT_SIZE];
        auto low = absBuf_.Get<uint32_t>();
        CreateVecIndex(idx.ReinterpretCast<int32_t>(), int32_t{0}, 2U * size);
        Duplicate(low.ReinterpretCast<int32_t>(), int32_t{1}, 2U * size);
        PipeBarrier<PIPE_V>();
        And(low.ReinterpretCast<uint16_t>(), idx.ReinterpretCast<uint16_t>(), low.ReinterpretCast<uint16_t>(),
            4U * size);
        ShiftRight(idx, idx, uint32_t{1}, 2U * size);
        PipeBarrier<PIPE_V>();
        if (PaddedQ() == 192U) {
            auto tmp = raw_;
            Cast(tmp, low.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, 2U * size);
            PipeBarrier<PIPE_V>();
            Muls(tmp, tmp, 768.0f, 2U * size);
            PipeBarrier<PIPE_V>();
            Cast(low.ReinterpretCast<int32_t>(), tmp, RoundMode::CAST_RINT, 2U * size);
        } else {
            ShiftLeft(low, low, PaddedQ() == 64U ? 8U : PaddedQ() == 128U ? 9U : 10U, 2U * size);
        }
        PipeBarrier<PIPE_V>();
        Add(idx.ReinterpretCast<int32_t>(), idx.ReinterpretCast<int32_t>(), low.ReinterpretCast<int32_t>(), 2U * size);
        PipeBarrier<PIPE_V>();
        ShiftLeft(idx, idx, uint32_t{2}, 2U * size);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StoreExpandedRows(uint32_t base)
    {
        const uint32_t size = 4U * PaddedQ();
        auto in = rectBuf_.Get<float>();
        auto idx = rectBuf_.Get<uint32_t>()[RECT_SIZE];
        DataCopy(in, real_, size);
        DataCopy(in[size], imag_, size);
        in.SetValue(2U * size, 0.0f);
        Fence<HardEvent::S_V>();
        PipeBarrier<PIPE_V>();
        Gather(raw_, in, idx, 0U, 2U * size);
        Fence<HardEvent::V_S>();
        const bool descending = !t_.upper && ((!Right() && t_.originalTrans == ACLBLAS_OP_N) ||
                                              (Right() && t_.originalTrans != ACLBLAS_OP_N));
        const bool separate = !t_.upper && !descending;
        if (separate) {
            for (uint32_t rr = 0; rr < 4U; ++rr) {
                raw_.SetValue(2U * (rr * PaddedQ() + base + rr), 0.0f);
                raw_.SetValue(2U * (rr * PaddedQ() + base + rr) + 1U, 0.0f);
            }
        }
        Fence<HardEvent::S_MTE3>();
        const uint64_t expanded = kOriginalAPlanes * t_.aPlane + 2U * t_.xPlane;
        Write(w_[expanded + 2U * static_cast<uint64_t>(base) * PaddedQ()], raw_, 2U * size);
        Fence<HardEvent::MTE3_V>();
        if (separate) {
            Duplicate(raw_, 0.0f, 4U * kDiagonalRowFloats);
            Fence<HardEvent::V_S>();
            for (uint32_t rr = 0; rr < 4U; ++rr) {
                const uint32_t off = rr * kDiagonalRowFloats + (base + rr) % 128U * 2U;
                raw_.SetValue(off, real_.GetValue(rr * PaddedQ() + base + rr));
                raw_.SetValue(off + 1U, imag_.GetValue(rr * PaddedQ() + base + rr));
            }
            Fence<HardEvent::S_MTE3>();
            const uint64_t diag = kCubeAPlanes * t_.aPlane + kCubeXPlanes * t_.xPlane;
            Write(w_[diag + static_cast<uint64_t>(base) * kDiagonalRowFloats], raw_, 4U * kDiagonalRowFloats);
            Fence<HardEvent::MTE3_V>();
        }
    }
    __aicore__ inline uint32_t PackFourRows()
    {
        uint32_t bad = 0;
        const uint32_t size = 4U * PaddedQ();
        for (uint32_t base = GetBlockIdx() * 4U; base < PaddedQ(); base += GetBlockNum() * 4U) {
            const uint32_t rows = base < t_.q ? Min(4U, t_.q - base) : 0U;
            Duplicate(raw_, 0.0f, 2U * size);
            Fence<HardEvent::V_MTE2>();
            LoadFourRows(base, rows);
            Fence<HardEvent::MTE2_S>();
            if (t_.unit) {
                for (uint32_t rr = 0; rr < rows; ++rr) {
                    raw_.SetValue(t_.transpose ? (rr * PaddedQ() + base + rr) * 2U : (base + rr) * 8U + rr * 2U, 1.0f);
                }
            }
            Fence<HardEvent::S_V>();
            Gather(real_, raw_, index_, 0U, size);
            Gather(imag_, raw_, index_[CHUNK], 0U, size);
            PipeBarrier<PIPE_V>();
            bad |= CheckFinite(real_, size) | CheckFinite(imag_, size);
            if (!(bad & kReferenceRequired)) {
                uint32_t active = 0U;
                for (uint32_t rr = 0; rr < rows; ++rr) {
                    active += t_.upper ? t_.q - (base + rr) - t_.unit : base + rr + 1U - t_.unit;
                }
                if (active >= kCorrelationMinActive && CorrelatedComponents(real_, imag_, raw_, size)) {
                    bad |= kReferenceRequired;
                }
            }
            if (t_.conjugate) {
                Muls(imag_, imag_, -1.0f, size);
                PipeBarrier<PIPE_V>();
            }
            StorePair(static_cast<uint64_t>(base) * PaddedQ(), t_.aPlane, size);
            StoreExpandedRows(base);
        }
        return bad;
    }
    __aicore__ inline void SetRectIndices()
    {
        if (Right()) {
            return; // Interleaved rows use the built-in even/odd masks.
        }
        const uint32_t width = t_.rp % kCubeTile == 0U ? kCubeTile : kCubeNarrowTile;
        const uint32_t size = RECT_ROWS * width;
        auto index = rectBuf_.Get<uint32_t>()[RECT_SIZE];
        auto scratch = rectBuf_.Get<uint32_t>();
        CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t{0}, size);
        PipeBarrier<PIPE_V>();
        if (Right()) {
            ShiftLeft(index, index, uint32_t{3}, size);
        } else {
            Duplicate(scratch.ReinterpretCast<int32_t>(), static_cast<int32_t>(width - 1U), size);
            PipeBarrier<PIPE_V>();
            And(scratch.ReinterpretCast<uint16_t>(), index.ReinterpretCast<uint16_t>(),
                scratch.ReinterpretCast<uint16_t>(), size * 2U);
            ShiftRight(index, index, width == 128U ? 7U : 6U, size);
            PipeBarrier<PIPE_V>();
            ShiftLeft(index, index, uint32_t{3}, size);
            ShiftLeft(scratch, scratch, uint32_t{8}, size);
            PipeBarrier<PIPE_V>();
            Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), scratch.ReinterpretCast<int32_t>(),
                size);
        }
        PipeBarrier<PIPE_V>();
        Adds(index[RECT_SIZE].ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t{4}, size);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline uint32_t PackX()
    {
        const bool wide = !ShortQ && Right() && t_.q > kCubeTile && t_.rp % kWideOutputWidth == 0U;
        const uint32_t width = wide ? kWideOutputWidth : t_.rp % kCubeTile == 0U ? kCubeTile : kCubeNarrowTile;
        const uint32_t rectRows = ShortQ || (Right() && t_.q <= kCubeTile) ? 16U : wide ? 32U : 64U;
        const uint32_t size = rectRows * width;
        auto re = rectBuf_.Get<float>();
        auto im = re[size];
        auto index = rectBuf_.Get<uint32_t>()[RECT_SIZE];
        uint32_t bad = 0;
        const uint32_t columns = t_.rp / width;
        const uint64_t tasks = static_cast<uint64_t>(PaddedQ() / rectRows) * columns;
        for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
            const uint32_t row = task / columns * rectRows;
            const uint32_t col = task % columns * width;
            const uint32_t rows = row < t_.q ? Min(rectRows, t_.q - row) : 0U;
            const uint32_t cols = col < t_.r ? Min(width, t_.r - col) : 0U;
            Duplicate(raw_, 0.0f, 2U * size);
            Fence<HardEvent::V_MTE2>();
            LoadXRectangle(row, col, rows, cols, width, rectRows);
            Fence<HardEvent::MTE2_V>();
            SplitXRectangle(re, im, size, width, rectRows);
            PipeBarrier<PIPE_V>();
            bad |= CheckFinite(re, 2U * size);
            StoreXRectangle(re, im, row, col, width, rectRows, size);
        }
        return bad;
    }
    __aicore__ inline void SetIndices(uint32_t shift, uint32_t base)
    {
        auto real = index_[base];
        auto imag = index_[base + CHUNK];
        CreateVecIndex(real.ReinterpretCast<int32_t>(), int32_t{0}, CHUNK);
        PipeBarrier<PIPE_V>();
        ShiftLeft(real, real, shift, CHUNK);
        PipeBarrier<PIPE_V>();
        Adds(imag.ReinterpretCast<int32_t>(), real.ReinterpretCast<int32_t>(), int32_t{4}, CHUNK);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadComplex(
        GlobalTensor<float> source, uint64_t offset, uint32_t stride, uint32_t count, uint32_t indexBase)
    {
        // raw_ is reused by Gather/reduction and the next MTE2 transfer.
        Fence<HardEvent::V_MTE2>();
        if (stride == 1) {
            Read(raw_, source[2U * offset], 2U * count);
        } else {
            DataCopyPad(
                raw_, source[2U * offset],
                DataCopyExtParams{static_cast<uint16_t>(count), 8U, (stride - 1U) * 8U, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        }
        Fence<HardEvent::MTE2_V>();
        Gather(real_, raw_, index_[indexBase], 0U, count);
        Gather(imag_, raw_, index_[indexBase + CHUNK], 0U, count);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StorePair(uint64_t offset, uint64_t plane, uint32_t count)
    {
        Fence<HardEvent::V_MTE3>();
        Write(w_[offset], real_, count);
        Write(w_[offset + plane], imag_, count);
        Fence<HardEvent::MTE3_V>();
    }
    __aicore__ inline void ZeroRow(uint32_t row, uint32_t width, uint64_t realBase, uint64_t imagBase)
    {
        Duplicate(real_, 0.0f, CHUNK);
        Fence<HardEvent::V_MTE3>();
        for (uint32_t j = 0; j < width; j += CHUNK) {
            uint32_t count = Min(CHUNK, width - j);
            Write(w_[realBase + static_cast<uint64_t>(row) * width + j], real_, count);
            Write(w_[imagBase + static_cast<uint64_t>(row) * width + j], real_, count);
        }
        Fence<HardEvent::MTE3_V>();
    }
    CtrmmTiling t_{};
    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> realBuf_;
    TBuf<TPosition::VECCALC> imagBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> absBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> rectBuf_;
    LocalTensor<float> raw_;
    LocalTensor<float> real_;
    LocalTensor<float> imag_;
    LocalTensor<uint32_t> index_;
    GlobalTensor<float> a_;
    GlobalTensor<float> b_;
    GlobalTensor<float> w_;
};

} // namespace
