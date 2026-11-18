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
#include "kernel_operator.h"
#include "chbmv_extended.h"
#include "chbmv_vector_layout.h"

// One vector lane owns one output row. Wide bands use independent partial
// sums, reduced within that lane; output ownership never crosses cores.
class ChbmvVector {
public:
    __aicore__ inline explicit ChbmvVector(const ChbmvTilingData& tiling) : t(tiling) {}

    __aicore__ inline void Run(GM_ADDR a, GM_ADDR x, GM_ADDR y)
    {
        AscendC::TPipe pipe;
        Init(a, x, y, &pipe);
        const uint32_t begin = AscendC::GetBlockIdx() * t.blockRows;
        const uint32_t end = begin + t.blockRows < t.n ? begin + t.blockRows : t.n;
        for (start = begin; start < end; start += t.vectorRows) {
            rows = end - start < t.vectorRows ? end - start : t.vectorRows;
            LoadY();
            if (t.alphaReal != 0 || t.alphaImag != 0) {
                LoadCaches();
                Duplicate(mr, 0.0f, 64);
                Duplicate(mi, 0.0f, 64);
                AscendC::PipeBarrier<PIPE_V>();
                if (t.vectorMode == 1) {
                    for (int32_t d = -static_cast<int32_t>(t.band); d < 0; ++d)
                        Pass(d, t.upper != 0);
                    Diagonal();
                    if (t.upper)
                        ApplyMirror();
                    for (int32_t d = 1; d <= static_cast<int32_t>(t.band); ++d)
                        Pass(d, t.upper == 0);
                    if (!t.upper)
                        ApplyMirror();
                } else {
                    // Short bands need fewer independent sums; longer bands
                    // use 32 to bound the length of each rounding chain.
                    Duplicate(partial, 0.0f, PartialCount() * 128);
                    AscendC::PipeBarrier<PIPE_V>();
                    MirrorTiled();
                    Diagonal();
                    DirectTiled();
                    ReduceWide();
                }
            }
            Store();
        }
        AscendC::ResetMask();
        pipe_ = nullptr;
    }

private:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        pipe->InitBuffer(aBuffer, t.aCacheBytes);
        pipe->InitBuffer(xBuffer, t.xCacheBytes);
        pipe->InitBuffer(scratch, CHBMV_VECTOR_SCRATCH_BYTES);
        aSpan = aBuffer.Get<float>();
        xSpan = xBuffer.Get<float>();
        auto s = scratch.Get<float>();
        ar = s[chbmv_layout::AR];
        ai = s[chbmv_layout::AI];
        xr = s[chbmv_layout::XR];
        xi = s[chbmv_layout::XI];
        yr = s[chbmv_layout::YR];
        yi = s[chbmv_layout::YI];
        mr = s[chbmv_layout::MR];
        mi = s[chbmv_layout::MI];
        vr = s[chbmv_layout::VR];
        vi = s[chbmv_layout::VI];
        p0 = s[chbmv_layout::P0];
        p1 = s[chbmv_layout::P1];
        p2 = s[chbmv_layout::P2];
        p3 = s[chbmv_layout::P3];
        offA = scratch.Get<uint32_t>()[chbmv_layout::A_OFFSETS];
        offX = scratch.Get<uint32_t>()[chbmv_layout::X_OFFSETS];
        packed = s[chbmv_layout::PACKED];
        phaseReal = s[chbmv_layout::PHASE_REAL];
        phaseImag = s[chbmv_layout::PHASE_IMAG];
        phaseOffsets = scratch.Get<uint32_t>()[chbmv_layout::PHASE_OFFSETS];
        partial = s[chbmv_layout::PARTIAL];
        phaseNegativeImag = s[chbmv_layout::PHASE_NEGATIVE_IMAG];
        CreateVecIndex(offX.ReinterpretCast<int32_t>(), int32_t(0), 64);
        if (t.vectorMode == 2)
            CreateVecIndex(phaseOffsets.ReinterpretCast<int32_t>(), int32_t(0), 128);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(offX.ReinterpretCast<int32_t>(), offX.ReinterpretCast<int32_t>(), int32_t(8), 64);
        if (t.vectorMode == 2)
            Muls(phaseOffsets.ReinterpretCast<int32_t>(), phaseOffsets.ReinterpretCast<int32_t>(), int32_t(8), 128);
        AscendC::PipeBarrier<PIPE_V>();
        if (t.vectorMode == 1) {
            Muls(offA.ReinterpretCast<int32_t>(), offX.ReinterpretCast<int32_t>(), static_cast<int32_t>(t.lda), 64);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    template <AscendC::HardEvent E>
    __aicore__ inline void Sync()
    {
        const auto id = static_cast<event_t>(pipe_->FetchEventID(E));
        AscendC::SetFlag<E>(id);
        AscendC::WaitFlag<E>(id);
    }
    __aicore__ inline void Scale(
        const AscendC::LocalTensor<float>& re, const AscendC::LocalTensor<float>& im, float sr, float si)
    {
        Muls(p0, re, sr, rows);
        Muls(p1, im, si, rows);
        Muls(p2, im, sr, rows);
        Muls(p3, re, si, rows);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(vr, p0, p1, rows);
        Add(vi, p2, p3, rows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Product(
        const AscendC::LocalTensor<float>& re, const AscendC::LocalTensor<float>& im, uint64_t bits,
        const AscendC::LocalTensor<float>& ar, const AscendC::LocalTensor<float>& ai)
    {
        uint64_t mask[2] = {bits, 0};
        Mul(p0, re, ar, mask, 1, {1, 1, 1, 8, 8, 8});
        Mul(p1, im, ai, mask, 1, {1, 1, 1, 8, 8, 8});
        Mul(p2, re, ai, mask, 1, {1, 1, 1, 8, 8, 8});
        Mul(p3, im, ar, mask, 1, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        Sub(vr, p0, p1, mask, 1, {1, 1, 1, 8, 8, 8});
        Add(vi, p2, p3, mask, 1, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadY();
    __aicore__ inline void CopyAt(
        const AscendC::LocalTensor<float>& dst, const AscendC::GlobalTensor<float>& src, uint64_t gmFloat,
        uint32_t ubFloat, uint32_t count)
    {
        const uint32_t aligned = ubFloat / 8 * 8;
        const uint8_t head = ubFloat - aligned;
        const uint8_t tail = (8 - (head + count) % 8) % 8;
        DataCopyPad(
            dst[aligned], src[gmFloat], AscendC::DataCopyExtParams{1, count * 4, 0, 0, 0},
            AscendC::DataCopyPadExtParams<float>{true, head, tail, 0.0f});
    }
    __aicore__ inline void LoadCaches()
    {
        const int64_t logical = static_cast<int64_t>(start) - t.band;
        const uint32_t first = logical > 0 ? logical : 0;
        const uint64_t last64 = static_cast<uint64_t>(start) + rows + t.band;
        const uint32_t last = last64 < t.n ? last64 : t.n;
        if (t.vectorMode == 1)
            Duplicate(aSpan, 0.0f, t.aCacheBytes / 4);
        Duplicate(xSpan, 0.0f, t.xCacheBytes / 4);
        Sync<AscendC::HardEvent::V_MTE2>();
        if (t.vectorMode == 1)
            CopyAt(
                aSpan, aGm, static_cast<uint64_t>(first) * t.lda * 2,
                static_cast<uint32_t>(first - logical) * t.lda * 2, (last - first) * t.lda * 2);
        if (t.incx == 1) {
            CopyAt(
                xSpan, xGm, static_cast<uint64_t>(first) * 2, static_cast<uint32_t>(first - logical) * 2,
                (last - first) * 2);
        } else {
            Sync<AscendC::HardEvent::V_S>();
            for (uint32_t c = first; c < last; ++c) {
                const int64_t at = t.xStart + static_cast<int64_t>(c) * t.incx;
                const uint32_t dest = static_cast<uint32_t>(c - logical) * 2;
                xSpan.SetValue(dest, xGm.GetValue(at * 2));
                xSpan.SetValue(dest + 1, xGm.GetValue(at * 2 + 1));
            }
            Sync<AscendC::HardEvent::S_V>();
        }
        Sync<AscendC::HardEvent::MTE2_V>();
        if (t.vectorMode == 2) {
            Sync<AscendC::HardEvent::MTE2_S>();
            if (t.alphaReal != 1 || t.alphaImag != 0) {
                Sync<AscendC::HardEvent::V_S>();
                for (uint32_t c = first; c < last; ++c) {
                    const uint32_t at = static_cast<uint32_t>(c - logical) * 2;
                    const float re = xSpan.GetValue(at);
                    const float im = xSpan.GetValue(at + 1);
                    xSpan.SetValue(at, t.alphaReal * re - t.alphaImag * im);
                    xSpan.SetValue(at + 1, t.alphaReal * im + t.alphaImag * re);
                }
                Sync<AscendC::HardEvent::S_V>();
            }
        }
    }
    __aicore__ inline uint64_t ValidMask(int32_t d)
    {
        const int64_t base = static_cast<int64_t>(start) + d;
        if (base >= t.n || base + rows <= 0)
            return 0;
        const uint32_t lo = base < 0 ? static_cast<uint32_t>(-base) : 0;
        const uint32_t hi = base + rows > t.n ? static_cast<uint32_t>(t.n - base) : rows;
        if (lo >= hi || lo >= 64)
            return 0;
        const uint64_t last = hi == 64 ? ~0ULL : ((1ULL << hi) - 1);
        return last & (~0ULL << lo);
    }
    __aicore__ inline void Pass(int32_t d, bool mirror)
    {
        const uint64_t bits = ValidMask(d);
        if (!bits)
            return;
        int64_t base = static_cast<int64_t>(t.band) * t.lda;
        if (mirror)
            base += t.upper ? static_cast<int64_t>(t.k) + d : d;
        else
            base += static_cast<int64_t>(d) * t.lda + (t.upper ? static_cast<int64_t>(t.k) - d : -d);
        Gather(ar, aSpan, offA, static_cast<uint32_t>(base * 8), rows);
        Gather(ai, aSpan, offA, static_cast<uint32_t>(base * 8 + 4), rows);
        const uint32_t xb = static_cast<uint32_t>(static_cast<int64_t>(t.band) + d) * 8;
        Gather(xr, xSpan, offX, xb, rows);
        Gather(xi, xSpan, offX, xb + 4, rows);
        AscendC::PipeBarrier<PIPE_V>();
        if (mirror) {
            Muls(ai, ai, -1.0f, rows);
            AscendC::PipeBarrier<PIPE_V>();
            Product(xr, xi, bits, ar, ai);
        } else {
            Scale(xr, xi, t.alphaReal, t.alphaImag);
            Product(vr, vi, bits, ar, ai);
        }
        uint64_t mask[2] = {bits, 0};
        Add(mirror ? mr : yr, mirror ? mr : yr, vr, mask, 1, {1, 1, 1, 8, 8, 8});
        Add(mirror ? mi : yi, mirror ? mi : yi, vi, mask, 1, {1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Diagonal()
    {
        if (t.vectorMode == 1)
            Gather(ar, aSpan, offA, (t.band * t.lda + (t.upper ? t.k : 0)) * 8, rows);
        else {
            Sync<AscendC::HardEvent::V_S>();
            for (uint32_t i = 0; i < rows; ++i)
                ar.SetValue(i, aGm.GetValue(2 * (static_cast<uint64_t>(start + i) * t.lda + (t.upper ? t.k : 0))));
            Sync<AscendC::HardEvent::S_V>();
        }
        Gather(xr, xSpan, offX, t.band * 8, rows);
        Gather(xi, xSpan, offX, t.band * 8 + 4, rows);
        AscendC::PipeBarrier<PIPE_V>();
        if (t.vectorMode == 2) {
            Mul(vr, xr, ar, rows);
            Mul(vi, xi, ar, rows);
            AscendC::PipeBarrier<PIPE_V>();
            Add(partial, partial, vr, rows);
            Add(partial[64], partial[64], vi, rows);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        Scale(xr, xi, t.alphaReal, t.alphaImag);
        Mul(vr, vr, ar, rows);
        Mul(vi, vi, ar, rows);
        AscendC::PipeBarrier<PIPE_V>();
        Add(yr, yr, vr, rows);
        Add(yi, yi, vi, rows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ApplyMirror()
    {
        Scale(mr, mi, t.alphaReal, t.alphaImag);
        Add(yr, yr, vr, rows);
        Add(yi, yi, vi, rows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline uint32_t PartialCount() const { return t.band <= 128 ? 8 : 32; }
    __aicore__ inline void ReduceWide()
    {
        for (uint32_t stride = PartialCount() / 2; stride != 0; stride /= 2) {
            for (uint32_t i = 0; i < stride; ++i) {
                Add(partial[i * 128], partial[i * 128], partial[(i + stride) * 128], rows);
                Add(partial[i * 128 + 64], partial[i * 128 + 64], partial[(i + stride) * 128 + 64], rows);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        Add(yr, yr, partial, rows);
        Add(yi, yi, partial[64], rows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void LoadMirrorTile(
        uint32_t pos, uint32_t count, const AscendC::LocalTensor<float>& real, const AscendC::LocalTensor<float>& imag)
    {
        constexpr uint32_t tile = CHBMV_MIRROR_TILE;
        if (count == tile) {
            // DMA overwrites every column of each valid row. Only rows
            // read by the final 16-row transpose group need zero fill.
            const uint32_t paddedRows = (rows + 15) / 16 * 16;
            if (paddedRows != rows)
                Duplicate(aSpan[rows * tile * 2], 0.0f, (paddedRows - rows) * tile * 2);
        } else {
            Duplicate(aSpan, 0.0f, chbmv_layout::MIRROR_INPUT_ELEMENTS);
        }
        Sync<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(
            aSpan, aGm[2 * (static_cast<uint64_t>(start) * t.lda + pos)],
            AscendC::DataCopyExtParams{
                static_cast<uint16_t>(rows), count * 8, (t.lda - count) * 8,
                (tile * 8 - (count * 8 + 31) / 32 * 32) / 32, 0},
            AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        Sync<AscendC::HardEvent::MTE2_V>();
        for (uint32_t group = 0; group < rows; group += 16) {
            uint64_t src[16];
            uint64_t dst[16];
#pragma unroll
            for (uint32_t i = 0; i < 16; ++i)
                src[i] = reinterpret_cast<uint64_t>(aSpan[(group + i) * tile * 2].GetPhyAddr());
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                dst[4 * i] = reinterpret_cast<uint64_t>(real[i * 64 + group].GetPhyAddr());
                dst[4 * i + 1] = reinterpret_cast<uint64_t>(real[i * 64 + group + 8].GetPhyAddr());
                dst[4 * i + 2] = reinterpret_cast<uint64_t>(imag[i * 64 + group].GetPhyAddr());
                dst[4 * i + 3] = reinterpret_cast<uint64_t>(imag[i * 64 + group + 8].GetPhyAddr());
            }
            // DAV-2201 single-repeat transpose requires zero repeat strides.
            const uint32_t repeats = (count + 3) / 4;
            AscendC::TransDataTo5HD<float>(
                dst, src,
                AscendC::TransDataTo5HDParams(false, false, repeats, repeats == 1 ? 0 : 32, repeats == 1 ? 0 : 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        Muls(imag, imag, -1.0f, static_cast<uint64_t>(rows), static_cast<uint8_t>(count), {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void AccumulateMirrorTile(
        int32_t base, uint32_t count, const AscendC::LocalTensor<float>& real, const AscendC::LocalTensor<float>& imag)
    {
        // Eight aligned phases cover every one-float shift. A phase
        // window is reused by eight diagonals; each vector input view
        // starts at a multiple of eight floats (32 bytes).
        for (uint32_t phase = 0; phase < 8 && phase < count; ++phase) {
            const uint32_t length = rows + count - phase - 1;
            const uint32_t xb = (static_cast<int64_t>(t.band) + base + phase) * 8;
            Gather(phaseReal[phase * 128], xSpan, phaseOffsets, xb, length);
            Gather(phaseImag[phase * 128], xSpan, phaseOffsets, xb + 4, length);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t phase = 0; phase < 8 && phase < count; ++phase)
            Muls(phaseNegativeImag[phase * 128], phaseImag[phase * 128], -1.0f, rows + count - phase - 1);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < count; ++j) {
            const int32_t d = base + j;
            const uint64_t bits = ValidMask(d);
            if (!bits)
                continue;
            const uint32_t phase = j % 8;
            const uint32_t at = phase * 128 + j - phase;
            uint64_t mask[2] = {bits, 0};
            auto sumReal = partial[(j & (PartialCount() - 1)) * 128];
            auto sumImag = partial[(j & (PartialCount() - 1)) * 128 + 64];
            MulAddDst(sumReal, real[j * 64], phaseReal[at], mask, 1, {1, 1, 1, 8, 8, 8});
            MulAddDst(sumImag, imag[j * 64], phaseReal[at], mask, 1, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            MulAddDst(sumReal, imag[j * 64], phaseNegativeImag[at], mask, 1, {1, 1, 1, 8, 8, 8});
            MulAddDst(sumImag, real[j * 64], phaseImag[at], mask, 1, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void MirrorTiled()
    {
        // Transpose 16 matrix columns at a time, splitting complex components
        // directly into contiguous output-row vectors. Source and destination
        // each occupy 32 KiB of the existing 64 KiB A cache.
        constexpr uint32_t tile = CHBMV_MIRROR_TILE;
        auto real = aSpan[chbmv_layout::MIRROR_REAL];
        auto imag = aSpan[chbmv_layout::MIRROR_IMAG];
        const int32_t first = t.upper ? -static_cast<int32_t>(t.band) : 1;
        const int32_t last = t.upper ? 0 : static_cast<int32_t>(t.band) + 1;
        for (int32_t base = first; base < last; base += tile) {
            const uint32_t count = last - base < tile ? last - base : tile;
            const uint32_t pos = t.upper ? static_cast<int64_t>(t.k) + base : base;
            LoadMirrorTile(pos, count, real, imag);
            AccumulateMirrorTile(base, count, real, imag);
        }
    }
    __aicore__ inline void DirectBounds(uint32_t c, uint32_t& lo, uint32_t& hi)
    {
        const uint32_t first = t.upper ? (c > t.band ? c - t.band : 0) : c + 1;
        const uint64_t last = t.upper ? c : static_cast<uint64_t>(c) + t.band + 1;
        lo = first > start ? first - start : 0;
        hi = last < start + rows ? last - start : rows;
    }
    __aicore__ inline uint64_t DirectIndex(uint32_t c, uint32_t lo)
    {
        const int64_t r = static_cast<int64_t>(start) + lo;
        return static_cast<uint64_t>(c) * t.lda + (t.upper ? static_cast<int64_t>(t.k) + r - c : r - c);
    }
    __aicore__ inline void LoadDirectTile(uint32_t base, uint32_t count)
    {
        uint32_t lo0;
        uint32_t hi0;
        uint32_t lo1;
        uint32_t hi1;
        DirectBounds(base, lo0, hi0);
        DirectBounds(base + count - 1, lo1, hi1);
        const bool full = lo0 == 0 && lo1 == 0 && hi0 == rows && hi1 == rows;
        Duplicate(aSpan, 0.0f, count * 64 * 2);
        Sync<AscendC::HardEvent::V_MTE2>();
        if (full) {
            DataCopyPad(
                aSpan, aGm[DirectIndex(base, 0) * 2],
                // At a fixed output row, the band row decreases by one
                // when advancing one matrix column (both UPPER and LOWER).
                AscendC::DataCopyExtParams{
                    static_cast<uint16_t>(count), rows * 8, (t.lda - 1 - rows) * 8,
                    (64 * 8 - (rows * 8 + 31) / 32 * 32) / 32, 0},
                AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
        } else {
            for (uint32_t j = 0; j < count; ++j) {
                uint32_t lo;
                uint32_t hi;
                DirectBounds(base + j, lo, hi);
                if (lo < hi)
                    CopyAt(aSpan, aGm, DirectIndex(base + j, lo) * 2, (j * 64 + lo) * 2, (hi - lo) * 2);
            }
        }
        Sync<AscendC::HardEvent::MTE2_V>();
    }
    __aicore__ inline void DirectTiled()
    {
        const uint32_t begin = t.upper ? start + 1 : (start > t.band ? start - t.band : 0);
        const uint64_t maxEnd = static_cast<uint64_t>(start) + rows + t.band;
        const uint32_t end = t.upper ? (maxEnd < t.n ? maxEnd : t.n) : start + rows - 1;
        for (uint32_t base = begin; base < end; base += CHBMV_DIRECT_TILE) {
            const uint32_t count = end - base < CHBMV_DIRECT_TILE ? end - base : CHBMV_DIRECT_TILE;
            LoadDirectTile(base, count);
            // Decode all direct columns together, then use aligned vector
            // views in the ordered accumulation loop (no per-column Gather).
            auto directReal = aSpan[chbmv_layout::DIRECT_REAL];
            auto directImag = aSpan[chbmv_layout::DIRECT_IMAG];
            uint64_t selected = 0;
            GatherMask(directReal, aSpan, 1, false, 0, {1, static_cast<uint16_t>(count * 2), 8, 8}, selected);
            GatherMask(directImag, aSpan, 2, false, 0, {1, static_cast<uint16_t>(count * 2), 8, 8}, selected);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < count; ++j) {
                const uint32_t c = base + j;
                uint32_t lo;
                uint32_t hi;
                DirectBounds(c, lo, hi);
                if (lo >= hi)
                    continue;
                const uint64_t bits = (hi == 64 ? ~0ULL : (1ULL << hi) - 1) & (~0ULL << lo);
                const uint32_t xiAt = (static_cast<int64_t>(c) - start + t.band) * 2;
                const float re = xSpan.GetValue(xiAt);
                const float im = xSpan.GetValue(xiAt + 1);
                auto ar = directReal[j * 64];
                auto ai = directImag[j * 64];
                uint64_t mask[2] = {bits, 0};
                auto sumReal = partial[(j & (PartialCount() - 1)) * 128];
                auto sumImag = partial[(j & (PartialCount() - 1)) * 128 + 64];
                Axpy(sumReal, ar, re, mask, 1, {1, 1, 8, 8});
                Axpy(sumImag, ai, re, mask, 1, {1, 1, 8, 8});
                AscendC::PipeBarrier<PIPE_V>();
                Axpy(sumReal, ai, -im, mask, 1, {1, 1, 8, 8});
                Axpy(sumImag, ar, im, mask, 1, {1, 1, 8, 8});
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline void Store();
    const ChbmvTilingData& t;
    uint32_t start, rows;
    AscendC::TPipe* pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuffer, xBuffer, scratch;
    AscendC::GlobalTensor<float> aGm, xGm, yGm;
    AscendC::LocalTensor<float> aSpan, xSpan, ar, ai, xr, xi, yr, yi, mr, mi, vr, vi, p0, p1, p2, p3, packed;
    AscendC::LocalTensor<uint32_t> offA, offX, phaseOffsets;
    AscendC::LocalTensor<float> phaseReal, phaseImag, phaseNegativeImag, partial;
};

#include "chbmv_vector_output.h"
