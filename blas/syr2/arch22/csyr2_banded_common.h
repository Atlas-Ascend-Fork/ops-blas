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
#include "csyr2_triangle_copy.h"
#include "csyr2_scalar_fallback.h"

namespace Csyr2Banded {
using namespace AscendC;
template <uint32_t Low, uint32_t High, uint32_t Cut>
__aicore__ inline uint64_t Prefix(uint32_t c)
{
    return uint64_t(c) * (c + 1ULL) / 2ULL + uint64_t(Low) * (c < Cut ? c : Cut) +
           uint64_t(High) * (c > Cut ? c - Cut : 0U);
}
template <uint32_t Low, uint32_t High, uint32_t Cut>
__aicore__ inline uint32_t Boundary(uint32_t part, uint32_t parts, uint32_t n)
{
    if (!part || !parts) {
        return 0;
    }
    if (part == parts) {
        return n;
    }
    const uint64_t target = Prefix<Low, High, Cut>(n) * part / parts;
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2U;
        if (Prefix<Low, High, Cut>(mid) < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}
template <uint32_t Low, uint32_t High, uint32_t Cut>
__aicore__ inline Csyr2Vector::ColumnRange Range(const Csyr2TilingData& t)
{
    const bool oneBandPerCore = t.n <= 4U * GetBlockNum();
    const uint32_t first =
        oneBandPerCore ? 4U * GetBlockIdx() : (Boundary<Low, High, Cut>(GetBlockIdx(), GetBlockNum(), t.n) + 3U) & ~3U;
    const uint32_t limit =
        oneBandPerCore ? first + 4U : (Boundary<Low, High, Cut>(GetBlockIdx() + 1U, GetBlockNum(), t.n) + 3U) & ~3U;
    return {first, limit < t.n ? limit : t.n};
}
__aicore__ inline void Compute(Csyr2Vector::Vectors& v, LocalTensor<float> cur, uint32_t col, uint32_t last)
{
    SetVectorMask<float, MaskMode::COUNTER>(0U, 2U * (col + 4U));
    if (col + 4U <= last) {
        Csyr2Vector::UpdateBandFour(cur, 2U * (col + 4U), v.xv, v.xr, v.yv, v.yr, 2U * col, 0U);
    } else {
#pragma unroll
        for (uint32_t local = 0; local < 4U; ++local) {
            const uint32_t c = col + local;
            if (c >= last) {
                break;
            }
            auto dst = cur[2U * (col + 4U) * local];
            const float xre = v.xv.GetValue(2U * c);
            const float xim = v.xv.GetValue(2U * c + 1U);
            const float yre = v.yv.GetValue(2U * c);
            const float yim = v.yv.GetValue(2U * c + 1U);
            Csyr2Vector::Update(dst, v.xv, v.xr, v.yv, v.yr, xre, xim, yre, yim);
        }
    }
}
struct Bands {
    Csyr2Vector::AtomicBuffers storage;
    Csyr2TilingData t;
    uint32_t first;
    uint32_t last;
    bool pending[2] = {false, false};
    uint32_t buffer = 0;

    template <bool Manual>
    __aicore__ inline void Init(
        TPipe& pipe, GM_ADDR x, GM_ADDR y, GM_ADDR a, const Csyr2TilingData& tiling, Csyr2Vector::ColumnRange range)
    {
        t = tiling;
        first = range.first;
        last = range.last;
        const uint32_t bytes = (2U * t.n * sizeof(float) + 255U) & ~255U;
        storage.Init(pipe, x, y, a, bytes, 4U);
        auto index = storage.ab1.Get<int32_t>();
        Csyr2Vector::InitRotationIndex(index, 2U * last);
        storage.v.Load(0U, last);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        if constexpr (Manual) {
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        }
        Csyr2Triangle::UpperIn(storage.a0, storage.v.a, first, t.lda, t.n - first < 4U ? t.n - first : 4U);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (Manual) {
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        } else {
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        }
    }
    __aicore__ inline void Precise()
    {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        const uint32_t rows = first + 4U;
        const uint32_t columns = last - first;
        for (uint32_t local = 0; local < columns; ++local) {
            const uint32_t c = first + local;
            Csyr2Vector::PreciseColumn(storage.a0[2U * rows * local], storage.v, 2U * c, 0U, c + 1U, 0U, t);
        }
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        Csyr2Triangle::UpperOut(storage.v.a, storage.a0, first, t.lda, columns);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Prefetch(uint32_t col, uint32_t next)
    {
        if (col + 4U >= last) {
            return;
        }
        if (pending[next]) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, next ? EVENT_ID3 : EVENT_ID2);
            pending[next] = false;
        }
        Csyr2Triangle::UpperIn(
            next ? storage.a1 : storage.a0, storage.v.a, col + 4U, t.lda, t.n - col - 4U < 4U ? t.n - col - 4U : 4U);
        set_flag(PIPE_MTE2, PIPE_V, next ? EVENT_ID1 : EVENT_ID0);
    }
    template <bool Manual>
    __aicore__ inline void Run()
    {
        PipeBarrier<PIPE_V>();
        Csyr2Vector::Rotate(storage.v, storage.ab1.Get<uint32_t>(), 2U * last);
        if constexpr (Manual) {
            PipeBarrier<PIPE_V>();
        }
        // Rotation indices share a1; release them before its first input DMA.
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        SetMaskCount();
        for (uint32_t col = first; col < last; col += 4U) {
            auto cur = buffer ? storage.a1 : storage.a0;
            wait_flag(PIPE_MTE2, PIPE_V, buffer ? EVENT_ID1 : EVENT_ID0);
            const uint32_t next = buffer ^ 1U;
            Prefetch(col, next);
            Compute(storage.v, cur, col, last);
            if constexpr (Manual) {
                set_flag(PIPE_V, PIPE_MTE3, buffer ? EVENT_ID5 : EVENT_ID4);
                wait_flag(PIPE_V, PIPE_MTE3, buffer ? EVENT_ID5 : EVENT_ID4);
            } else {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            }
            Csyr2Triangle::UpperOut(storage.v.a, cur, col, t.lda, t.n - col < 4U ? t.n - col : 4U);
            set_flag(PIPE_MTE3, PIPE_MTE2, buffer ? EVENT_ID3 : EVENT_ID2);
            pending[buffer] = true;
            buffer = next;
        }
        Csyr2Vector::Finish<true, false>(pending);
    }
};
} // namespace Csyr2Banded
