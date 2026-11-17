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
#include "common/helper/kernel_utils.h"

namespace Csyr2Vector {
using namespace AscendC;
struct Vectors {
    GlobalTensor<float> x;
    GlobalTensor<float> y;
    GlobalTensor<float> a;
    TBuf<TPosition::VECCALC> xb;
    TBuf<TPosition::VECCALC> yb;
    TBuf<TPosition::VECCALC> xrb;
    TBuf<TPosition::VECCALC> yrb;
    LocalTensor<float> xv;
    LocalTensor<float> yv;
    LocalTensor<float> xr;
    LocalTensor<float> yr;

    __aicore__ inline void InitGlobals(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm)
    {
        x.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xGm));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yGm));
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm));
    }
    __aicore__ inline void Allocate(TPipe& pipe, uint32_t bytes, uint32_t rotationBytes)
    {
        pipe.InitBuffer(xb, bytes);
        pipe.InitBuffer(yb, bytes);
        pipe.InitBuffer(xrb, rotationBytes);
        pipe.InitBuffer(yrb, rotationBytes);
    }
    __aicore__ inline void Allocate(TPipe& pipe, uint32_t bytes) { Allocate(pipe, bytes, bytes); }
    __aicore__ inline void Bind()
    {
        xv = xb.Get<float>();
        yv = yb.Get<float>();
        xr = xrb.Get<float>();
        yr = yrb.Get<float>();
    }
    __aicore__ inline void Load(uint32_t first, uint32_t rows)
    {
        gm_to_ub_align<ArchType::ASCEND_V220, float>(xv, x[2ULL * first], 0, 1, 8U * rows, 0, 0, 0, 0);
        gm_to_ub_align<ArchType::ASCEND_V220, float>(yv, y[2ULL * first], 0, 1, 8U * rows, 0, 0, 0, 0);
    }
    __aicore__ inline void Increment(LocalTensor<float> dst, uint32_t coefficient, uint32_t offset, uint32_t rows)
    {
        const float xre = xv.GetValue(coefficient);
        const float xim = xv.GetValue(coefficient + 1U);
        const float yre = yv.GetValue(coefficient);
        const float yim = yv.GetValue(coefficient + 1U);
        SetVectorMask<float, MaskMode::COUNTER>(0U, 2U * rows);
        Muls<float, false>(dst, xv[offset], yre, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(dst, xr[offset], yim, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(dst, yv[offset], xre, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(dst, yr[offset], xim, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    }
    __aicore__ inline void PairIncrement(LocalTensor<float> d0, LocalTensor<float> d1, uint32_t offset, uint32_t rows)
    {
        const float x0r = xv.GetValue(offset);
        const float x0i = xv.GetValue(offset + 1U);
        const float y0r = yv.GetValue(offset);
        const float y0i = yv.GetValue(offset + 1U);
        const float x1r = xv.GetValue(offset + 32U);
        const float x1i = xv.GetValue(offset + 33U);
        const float y1r = yv.GetValue(offset + 32U);
        const float y1i = yv.GetValue(offset + 33U);
        SetVectorMask<float, MaskMode::COUNTER>(0U, 2U * rows);
        Muls<float, false>(d0, xv[offset], y0r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        Muls<float, false>(d1, xv[offset], y1r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(d0, xr[offset], y0i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        Axpy<float, float, false>(d1, xr[offset], y1i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(d0, yv[offset], x0r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        Axpy<float, float, false>(d1, yv[offset], x1r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(d0, yr[offset], x0i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        Axpy<float, float, false>(d1, yr[offset], x1i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    }
};
struct AtomicBuffers {
    Vectors v;
    TBuf<TPosition::VECCALC> ab0;
    TBuf<TPosition::VECCALC> ab1;
    LocalTensor<float> a0;
    LocalTensor<float> a1;

    __aicore__ inline void Init(TPipe& pipe, GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, uint32_t bytes, uint32_t batch)
    {
        v.InitGlobals(xGm, yGm, aGm);
        v.Allocate(pipe, bytes);
        pipe.InitBuffer(ab0, batch * bytes);
        pipe.InitBuffer(ab1, batch * bytes);
        v.Bind();
        a0 = ab0.Get<float>();
        a1 = ab1.Get<float>();
    }
};
struct ColumnRange {
    uint32_t first;
    uint32_t last;
};
__aicore__ inline ColumnRange AtomicRange(uint32_t n, uint32_t cut1, uint32_t cut2)
{
    const uint32_t groups = GetBlockNum() / 16U;
    const uint32_t group = GetBlockIdx() / 16U;
    const uint32_t lane = GetBlockIdx() % 16U;
    const uint32_t start = group == 0U ? 0U : (group == 1U ? cut1 : cut2);
    const uint32_t stop = group + 1U == groups ? n : (group == 0U ? cut1 : cut2);
    const uint32_t first = ((start + 15U) & ~15U) + lane;
    const uint32_t limit = (stop + 15U) & ~15U;
    return {first, limit < n ? limit : n};
}
__aicore__ inline void InitOddRotationIndex(LocalTensor<int32_t> index, uint32_t count)
{
    uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0ULL};
    CreateVecIndex(index, int32_t(1), count);
    PipeBarrier<PIPE_V>();
    Adds(index, index, int32_t(-2), odd, uint8_t((count + 63U) / 64U), {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, int32_t(2), count);
}
template <bool ToMte2, bool Atomic>
__aicore__ inline void Finish(const bool (&pending)[2])
{
    SetMaskNorm();
    ResetMask();
    for (uint32_t b = 0; b < 2U; ++b) {
        if (pending[b]) {
            if constexpr (ToMte2) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, b ? EVENT_ID3 : EVENT_ID2);
            } else {
                wait_flag(PIPE_MTE3, PIPE_V, b ? EVENT_ID3 : EVENT_ID2);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (Atomic) {
        SetAtomicNone();
    }
}

// Build byte offsets for the (imaginary, real) companion vector.
__aicore__ inline void InitRotationIndex(LocalTensor<int32_t> index, uint32_t count)
{
    uint64_t even[2] = {0x5555555555555555ULL, 0ULL};
    uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0ULL};
    CreateVecIndex(index, int32_t(0), count);
    PipeBarrier<PIPE_V>();
    Adds(index, index, int32_t(1), even, uint8_t((count + 63U) / 64U), {1, 1, 8, 8});
    // Both masked instructions update the same index allocation.
    PipeBarrier<PIPE_V>();
    Adds(index, index, int32_t(-1), odd, uint8_t((count + 63U) / 64U), {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, int32_t(2), count);
}
__aicore__ inline void RotateVectors(
    LocalTensor<float> xr, LocalTensor<float> yr, LocalTensor<float> x, LocalTensor<float> y,
    LocalTensor<uint32_t> index, uint32_t count)
{
    uint64_t even[2] = {0x5555555555555555ULL, 0ULL};
    Gather(xr, x, index, 0U, count);
    Gather(yr, y, index, 0U, count);
    PipeBarrier<PIPE_V>();
    Muls(xr, xr, -1.0F, even, uint8_t((count + 63U) / 64U), {1, 1, 8, 8});
    Muls(yr, yr, -1.0F, even, uint8_t((count + 63U) / 64U), {1, 1, 8, 8});
}
__aicore__ inline void Rotate(Vectors& v, LocalTensor<uint32_t> index, uint32_t count)
{
    RotateVectors(v.xr, v.yr, v.xv, v.yv, index, count);
}
__aicore__ inline void InitHeadIndex(LocalTensor<int32_t> saved)
{
    CreateVecIndex(saved, int32_t(0), 8U);
    PipeBarrier<PIPE_V>();
    ShiftLeft(saved, saved, int32_t(2), 8U);
}
struct DiagonalBuffers {
    Vectors v;
    TBuf<TPosition::VECCALC> ab;
    TBuf<TPosition::VECCALC> ab1;
    TBuf<TPosition::VECCALC> preserved;
    LocalTensor<float> a0;
    LocalTensor<float> a1;
    LocalTensor<int32_t> saved;
    bool sharedHead;
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR x, GM_ADDR y, GM_ADDR a, uint32_t n)
    {
        v.InitGlobals(x, y, a);
        const uint32_t bytes = (8U * n + 255U) & ~255U;
        v.Allocate(pipe, bytes);
        sharedHead = bytes >= 16384U;
        pipe.InitBuffer(preserved, sharedHead ? 128U : 256U);
        pipe.InitBuffer(ab, 4U * bytes);
        pipe.InitBuffer(ab1, 4U * bytes - 128U);
        v.Bind();
        a0 = ab.Get<float>();
        a1 = ab1.Get<float>();
        saved = preserved.Get<int32_t>();
        InitHeadIndex(saved);
        if (!sharedHead) {
            InitHeadIndex(saved[32U]);
        }
    }
};
template <uint32_t Weight>
__aicore__ inline uint64_t LowerWork(uint32_t c, uint32_t n)
{
    const uint32_t leading = c < n / 4U ? c : n / 4U;
    return uint64_t(c) * (2ULL * n - c + 1ULL) / 2ULL + uint64_t(Weight) * c + 768ULL * leading;
}
template <uint32_t Weight>
__aicore__ inline uint32_t LowerBoundary(uint32_t part, uint32_t parts, uint32_t n)
{
    if (!part || !parts) {
        return 0;
    }
    if (part == parts) {
        return n;
    }
    const uint64_t target = LowerWork<Weight>(n, n) * part / parts;
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2U;
        if (LowerWork<Weight>(mid, n) < target) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return (lo + 3U) & ~3U;
}
template <uint32_t Weight>
__aicore__ inline ColumnRange LowerRange(uint32_t n)
{
    const bool oneBandPerCore = n <= 4U * GetBlockNum();
    const uint32_t first = oneBandPerCore ? 4U * GetBlockIdx() : LowerBoundary<Weight>(GetBlockIdx(), GetBlockNum(), n);
    const uint32_t limit = oneBandPerCore ? first + 4U : LowerBoundary<Weight>(GetBlockIdx() + 1U, GetBlockNum(), n);
    return {first, limit < n ? limit : n};
}
__aicore__ inline void Update(
    LocalTensor<float> dst, LocalTensor<float> x, LocalTensor<float> xr, LocalTensor<float> y, LocalTensor<float> yr,
    float xre, float xim, float yre, float yim)
{
    Axpy<float, float, false>(dst, x, yre, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Axpy<float, float, false>(dst, xr, yim, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Axpy<float, float, false>(dst, y, xre, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Axpy<float, float, false>(dst, yr, xim, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
}
// Four independent columns share one counter mask and three stage barriers.
__aicore__ inline void UpdateBandFour(
    LocalTensor<float> dst, uint32_t pitch, LocalTensor<float> x, LocalTensor<float> xr, LocalTensor<float> y,
    LocalTensor<float> yr, uint32_t coefficient, uint32_t row)
{
    const float x0r = x.GetValue(coefficient + 0U);
    const float x0i = x.GetValue(coefficient + 1U);
    const float y0r = y.GetValue(coefficient + 0U);
    const float y0i = y.GetValue(coefficient + 1U);
    const float x1r = x.GetValue(coefficient + 2U);
    const float x1i = x.GetValue(coefficient + 3U);
    const float y1r = y.GetValue(coefficient + 2U);
    const float y1i = y.GetValue(coefficient + 3U);
    const float x2r = x.GetValue(coefficient + 4U);
    const float x2i = x.GetValue(coefficient + 5U);
    const float y2r = y.GetValue(coefficient + 4U);
    const float y2i = y.GetValue(coefficient + 5U);
    const float x3r = x.GetValue(coefficient + 6U);
    const float x3i = x.GetValue(coefficient + 7U);
    const float y3r = y.GetValue(coefficient + 6U);
    const float y3i = y.GetValue(coefficient + 7U);
    Axpy<float, float, false>(dst[0U * pitch], x[row], y0r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[1U * pitch], x[row], y1r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[2U * pitch], x[row], y2r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[3U * pitch], x[row], y3r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Axpy<float, float, false>(dst[0U * pitch], xr[row], y0i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[1U * pitch], xr[row], y1i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[2U * pitch], xr[row], y2i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[3U * pitch], xr[row], y3i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Axpy<float, float, false>(dst[0U * pitch], y[row], x0r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[1U * pitch], y[row], x1r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[2U * pitch], y[row], x2r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[3U * pitch], y[row], x3r, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    Axpy<float, float, false>(dst[0U * pitch], yr[row], x0i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[1U * pitch], yr[row], x1i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[2U * pitch], yr[row], x2i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    Axpy<float, float, false>(dst[3U * pitch], yr[row], x3i, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
}
} // namespace Csyr2Vector
