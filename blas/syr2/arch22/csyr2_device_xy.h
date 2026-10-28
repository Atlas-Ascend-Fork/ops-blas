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

#include "csyr2_ascend_kernel.h"

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline bool Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::XyIlvFeasibleUnit()
{
    if constexpr (kWindowed || !kUnitPos) {
        return false;
    }
    if ((!xyReuse_) || td_.n <= 0 || (tileRow_ & 3) != 0) {
        return false;
    }
    const int32_t ilvRows = td_.n;
    const int32_t nIlv = ilvRows * 2;
    const int32_t xyPitch = Csyr2CeilAlign<8>(nIlv);
    const int32_t tmpCap = (ilvFloats_ * 2 > planeFloats_ * 3) ? (ilvFloats_ * 2) : (planeFloats_ * 3);
    if (ilvRows > tileRow_ || ilvRows > planeFloats_ || nIlv > ilvFloats_ || nIlv > planeFloats_ * 2 ||
        xyPitch + nIlv > tmpCap) {
        return false;
    }
    return true;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::IssueXyPreload(bool feasible)
{
    xyPreload_ = false;
    if (!feasible) {
        return;
    }
    const int32_t nIlv = td_.n * 2;
    xIlvHeld_ = aInQue_.AllocTensor<float>();
    CopyInFloats(xIlvHeld_, 0, nIlv, xGm_);
    aInQue_.EnQue(xIlvHeld_);
    yIlvHeld_ = aInQue_.AllocTensor<float>();
    CopyInFloats(yIlvHeld_, 0, nIlv, yGm_);
    aInQue_.EnQue(yIlvHeld_);
    xyPreload_ = true;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::BuildSwapTable()
{
    auto idx = idxBuf_.Get<int32_t>();
// One scalar-written block seeds byte offsets for swapping complex components.
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
        idx.SetValue(static_cast<uint32_t>(i), (i ^ 1) * static_cast<int32_t>(sizeof(float)));
    }
    SyncEvent<AscendC::HardEvent::S_V>();
    for (int32_t filled = 8; filled < ilvFloats_;) {
        const int32_t count = (ilvFloats_ - filled < filled) ? (ilvFloats_ - filled) : filled;
        AscendC::Adds(idx[static_cast<uint32_t>(filled)], idx, filled * 4, count);
        AscendC::PipeBarrier<PIPE_V>();
        filled += count;
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline __attribute__((always_inline)) void
Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::GatherRotatedComplex(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, AscendC::LocalTensor<uint32_t> table, uint32_t n2,
    uint8_t repeats)
{
    AscendC::Gather(dst, src, table, static_cast<uint32_t>(0), n2);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(0), 0x5555555555555555ULL);
    AscendC::Muls<float, false>(
        dst, dst, -1.0f, static_cast<uint64_t>(0), repeats, AscendC::UnaryRepeatParams(1, 1, 8, 8));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ResetMask();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::BuildXYDirectSwap()
{
    const uint32_t n2 = static_cast<uint32_t>(td_.n * 2);
    const int32_t pitch = (kUpper || kRotateLower) && td_.n >= 640 ?
                              Csyr2CeilAlign<128>(static_cast<int32_t>(n2)) :
                              Csyr2CeilAlign<8>(static_cast<int32_t>(n2)) + (td_.n >= 640 ? 16 : 0);
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp;
    auto ys = tmp[static_cast<uint32_t>(pitch)];
    auto table = idxBuf_.Get<uint32_t>();
    if (!xyPreload_)
        IssueXyPreload(true);
    const uint8_t repeats = static_cast<uint8_t>((n2 + 63U) / 64U);
    xIlvHeld_ = aInQue_.DeQue<float>();
    GatherRotatedComplex(xs, xIlvHeld_, table, n2, repeats);
    // Gather Y after the padded X sign transform, so any overlapping tail is overwritten.
    yIlvHeld_ = aInQue_.DeQue<float>();
    GatherRotatedComplex(ys, yIlvHeld_, table, n2, repeats);
    xyIlvRows_ = td_.n;
    xyIlvPitch_ = pitch;
    xyIlvReady_ = true;
    lowerHeld_ = !kUpper;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::BuildInterleaveTable()
{
    auto idx = idxBuf_.Get<int32_t>();
    auto tmp = tmpBuf_.Get<int32_t>();
    const int32_t n2 = ilvFloats_;
    constexpr int32_t kSeed = 64;
    const int32_t seed = (n2 < kSeed) ? n2 : kSeed;
    const uint32_t uSeed = static_cast<uint32_t>(seed);
    AscendC::CreateVecIndex(idx, static_cast<int32_t>(0), uSeed);
    AscendC::ShiftRight(tmp, idx, static_cast<int32_t>(1), uSeed);
    AscendC::ShiftLeft(idx, tmp, static_cast<int32_t>(1), uSeed);
    auto seq = tmpBuf_.Get<int32_t>()[static_cast<uint32_t>(ilvFloats_)];
    AscendC::CreateVecIndex(seq, static_cast<int32_t>(0), uSeed);
    AscendC::Sub(idx, seq, idx, uSeed);
    AscendC::Muls(idx, idx, static_cast<int32_t>(planeFloats_), uSeed);
    AscendC::Add(idx, idx, tmp, uSeed);
    AscendC::ShiftLeft(idx, idx, static_cast<int32_t>(2), uSeed);
    AscendC::PipeBarrier<PIPE_V>();
    int32_t s = seed;
    while (s < n2) {
        int32_t nxt = s * 2;
        if (nxt > n2) {
            nxt = n2;
        }
        auto dst = idx[static_cast<uint32_t>(s)];
        AscendC::Adds(dst, idx, s * 2, nxt - s);
        AscendC::PipeBarrier<PIPE_V>();
        s = nxt;
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::CopyInFloats(
    AscendC::LocalTensor<float> dst, int64_t gmFloatOff, int32_t floats, AscendC::GlobalTensor<float> src)
{
    if ((floats * static_cast<int32_t>(sizeof(float))) % 32 == 0) {
        AscendC::DataCopy(dst, src[gmFloatOff], static_cast<uint32_t>(floats));
        return;
    }
    AscendC::DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(floats) * sizeof(float);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    AscendC::DataCopyPadExtParams<float> pad(false, 0, 0, 0.0f);
    AscendC::DataCopyPad(dst, src[gmFloatOff], p, pad);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::CopyOutFloats(
    int64_t gmFloatOff, AscendC::LocalTensor<float> src, int32_t floats)
{
    if constexpr (!kUpper) {
        if ((floats * static_cast<int32_t>(sizeof(float))) % 32 == 0) {
            AscendC::DataCopy(aGm_[gmFloatOff], src, static_cast<uint32_t>(floats));
            return;
        }
    }
    AscendC::DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint32_t>(floats) * sizeof(float);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    AscendC::DataCopyPad(aGm_[gmFloatOff], src, p);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::CopyInViaQue(
    AscendC::LocalTensor<float> dst, int64_t gmFloatOff, int32_t floats, AscendC::GlobalTensor<float> src)
{
    auto t = aInQue_.AllocTensor<float>();
    CopyInFloats(t, gmFloatOff, floats, src);
    aInQue_.EnQue(t);
    t = aInQue_.DeQue<float>();
    AscendC::Adds(dst, t, 0.0f, static_cast<uint32_t>(floats));
    AscendC::PipeBarrier<PIPE_V>();
    aInQue_.FreeTensor(t);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline __attribute__((always_inline)) void
Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::PrepareInterleavedInput(
    AscendC::LocalTensor<float>& ilv, AscendC::LocalTensor<float>& held, AscendC::GlobalTensor<float> gm, int32_t nIlv)
{
    if constexpr (kUpper) {
        if (packB_ == 1 || upperPackedHeld_) {
            if (!xyPreload_) {
                held = aInQue_.AllocTensor<float>();
                CopyInFloats(held, 0, nIlv, gm);
                aInQue_.EnQue(held);
            }
            held = aInQue_.DeQue<float>();
            ilv = held;
        } else {
            CopyInViaQue(ilv, 0, nIlv, gm);
        }
    } else {
        if (!xyPreload_) {
            held = aInQue_.AllocTensor<float>();
            CopyInFloats(held, 0, nIlv, gm);
            aInQue_.EnQue(held);
        }
        held = aInQue_.DeQue<float>();
        ilv = held;
        lowerHeld_ = true;
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::PrepareXyIlv(
    int32_t ilvRows, int32_t nIlv, int32_t xyPitch)
{
    AscendC::LocalTensor<float> xIlv;
    AscendC::LocalTensor<float> yIlv;
    if constexpr (kUpper) {
        if (packB_ != 1) {
            xIlv = xyStageBuf_.Get<float>();
            yIlv = aPlaneBuf_.Get<float>();
        }
    } else {
        xIlv = xyStageBuf_.Get<float>();
        yIlv = aPlaneBuf_.Get<float>();
    }
    auto planes = xyPlaneBuf_.Get<float>();
    auto p0 = planes;
    auto p1 = planes[static_cast<uint32_t>(planeFloats_)];
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp;
    auto ys = tmp[static_cast<uint32_t>(xyPitch)];
    auto tbl = idxBuf_.Get<uint32_t>();
    const uint32_t u = static_cast<uint32_t>(ilvRows);
    const uint32_t n2 = static_cast<uint32_t>(nIlv);

    PrepareInterleavedInput(xIlv, xIlvHeld_, xGm_, nIlv);
    SplitPlanes(xIlv, p1, p0, ilvRows);
    AscendC::Muls(p0, p0, -1.0f, u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(xs, planes, tbl, static_cast<uint32_t>(0), n2);
    AscendC::PipeBarrier<PIPE_V>();

    PrepareInterleavedInput(yIlv, yIlvHeld_, yGm_, nIlv);
    SplitPlanes(yIlv, p1, p0, ilvRows);
    AscendC::Muls(p0, p0, -1.0f, u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(ys, planes, tbl, static_cast<uint32_t>(0), n2);
    AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::BuildXyIlv()
{
    xyIlvReady_ = false;
    xyIlvRows_ = 0;
    if constexpr (kWindowed) {
        return;
    }
    if constexpr (!kUnitPos) {
        return;
    }
    if ((!xyReuse_) || td_.n <= 0) {
        return;
    }
    if ((tileRow_ & 3) != 0) {
        return;
    }
    const int32_t ilvRows = td_.n;
    const int32_t nIlv = ilvRows * 2;
    const int32_t xyPitch = Csyr2CeilAlign<8>(nIlv);
    const int32_t tmpCap = (ilvFloats_ * 2 > planeFloats_ * 3) ? (ilvFloats_ * 2) : (planeFloats_ * 3);
    if (ilvRows > tileRow_ || ilvRows > planeFloats_ || nIlv > ilvFloats_ || nIlv > planeFloats_ * 2 ||
        xyPitch + nIlv > tmpCap) {
        return;
    }
    PrepareXyIlv(ilvRows, nIlv, xyPitch);
    xyIlvRows_ = ilvRows;
    xyIlvPitch_ = xyPitch;
    xyIlvReady_ = true;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::SplitPlanes(
    AscendC::LocalTensor<float> ilv, AscendC::LocalTensor<float> re, AscendC::LocalTensor<float> im, int32_t cur)
{
    const uint16_t rpt = static_cast<uint16_t>((cur * 2 + CSYR2_VEC_FLOATS - 1) / CSYR2_VEC_FLOATS);
    uint32_t mask = 0;
    uint64_t rsvdCnt = 0;
    AscendC::GatherMask<float>(re, ilv, 1, false, mask, {1, rpt, 8, 8}, rsvdCnt);
    AscendC::GatherMask<float>(im, ilv, 2, false, mask, {1, rpt, 8, 8}, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::LoadXYWindow(int32_t i0, int32_t cur)
{
    auto planes = xyPlaneBuf_.Get<float>();
    auto xr = planes;
    auto xi = planes[static_cast<uint32_t>(planeFloats_)];
    auto yr = planes[static_cast<uint32_t>(planeFloats_ * 2)];
    auto yi = planes[static_cast<uint32_t>(planeFloats_ * 3)];
    auto stage = xyStageBuf_.Get<float>();

    if constexpr (kUnitPos) {
        CopyInFloats(stage, static_cast<int64_t>(i0) * 2, cur * 2, xGm_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        SplitPlanes(stage, xr, xi, cur);
        CopyInFloats(stage, static_cast<int64_t>(i0) * 2, cur * 2, yGm_);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        SplitPlanes(stage, yr, yi, cur);
    } else {
        /* 跨步 / 负步长：逐元素取。只有精度用例走这里，三条性能用例都是 inc=1。 */
        for (int32_t k = 0; k < cur; ++k) {
            const int64_t xi0 = Csyr2VectorOffset(td_.n, td_.incx, i0 + k) * 2;
            const int64_t yi0 = Csyr2VectorOffset(td_.n, td_.incy, i0 + k) * 2;
            xr.SetValue(k, xGm_.GetValue(xi0));
            xi.SetValue(k, xGm_.GetValue(xi0 + 1));
            yr.SetValue(k, yGm_.GetValue(yi0));
            yi.SetValue(k, yGm_.GetValue(yi0 + 1));
        }
        SyncEvent<AscendC::HardEvent::S_V>();
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::LoadXYInterleavedWindow(
    int32_t i0, int32_t cur)
{
    const int32_t n2 = cur * 2;
    auto planes = xyPlaneBuf_.Get<float>();
    auto p0 = planes;
    auto p1 = planes[static_cast<uint32_t>(planeFloats_)];
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp;
    const int32_t xyPitch = Csyr2CeilAlign<8>(n2);
    auto ys = tmp[static_cast<uint32_t>(xyPitch)];
    auto tbl = idxBuf_.Get<uint32_t>();

    if (!xyPreload_ || i0 != 0) {
        xIlvHeld_ = aInQue_.AllocTensor<float>();
        CopyInFloats(xIlvHeld_, static_cast<int64_t>(i0) * 2, n2, xGm_);
        aInQue_.EnQue(xIlvHeld_);
    }
    xIlvHeld_ = aInQue_.DeQue<float>();
    SplitPlanes(xIlvHeld_, p1, p0, cur);
    AscendC::Muls(p0, p0, -1.0f, static_cast<uint32_t>(cur));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(xs, planes, tbl, static_cast<uint32_t>(0), static_cast<uint32_t>(n2));
    AscendC::PipeBarrier<PIPE_V>();

    if (!xyPreload_ || i0 != 0) {
        yIlvHeld_ = aInQue_.AllocTensor<float>();
        CopyInFloats(yIlvHeld_, static_cast<int64_t>(i0) * 2, n2, yGm_);
        aInQue_.EnQue(yIlvHeld_);
    }
    yIlvHeld_ = aInQue_.DeQue<float>();
    SplitPlanes(yIlvHeld_, p1, p0, cur);
    AscendC::Muls(p0, p0, -1.0f, static_cast<uint32_t>(cur));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(ys, planes, tbl, static_cast<uint32_t>(0), static_cast<uint32_t>(n2));
    AscendC::PipeBarrier<PIPE_V>();
    xyIlvRows_ = cur;
    xyIlvPitch_ = xyPitch;
}
