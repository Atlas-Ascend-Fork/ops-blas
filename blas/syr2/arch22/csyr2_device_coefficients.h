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
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::PrecomputeCoefIlv()
{
    if constexpr (kUnitAlpha) {
        coefReady_ = true;
        return;
    }
    auto t1 = xyPlaneBuf_.Get<float>();
    auto t2 = idxBuf_.Get<float>();
    AscendC::LocalTensor<float> xIlv;
    AscendC::LocalTensor<float> yIlv;
    if (kUpper || kWindowed || lowerHeld_) {
        xIlv = xIlvHeld_;
        yIlv = yIlvHeld_;
    } else {
        xIlv = xyStageBuf_.Get<float>();
        yIlv = aPlaneBuf_.Get<float>();
    }
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp;
    auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_)];
    const uint32_t n2 = static_cast<uint32_t>(xyIlvRows_ * 2);
    if constexpr (kFullMode != 0) {
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        Csyr2LayerMuls(t1, yIlv, td_.alphaReal, n2);
        Csyr2LayerMuls(t2, xIlv, td_.alphaReal, n2);
        AscendC::PipeBarrier<PIPE_V>();
        Csyr2LayerAxpy(t1, ys, td_.alphaImag, n2);
        Csyr2LayerAxpy(t2, xs, td_.alphaImag, n2);
        coefReady_ = true;
        return;
    }
    // Keep the exact logical length: generic windows need not have room
    // for a full padded repeat. The two destinations are separate buffers.
    Csyr2CountMaskBegin(n2);
    Csyr2NativeMulsCount(t1, yIlv, td_.alphaReal);
    Csyr2NativeMulsCount(t2, xIlv, td_.alphaReal);
    AscendC::PipeBarrier<PIPE_V>();
    Csyr2NativeAxpyCount(t1, ys, td_.alphaImag);
    Csyr2NativeAxpyCount(t2, xs, td_.alphaImag);
    AscendC::PipeBarrier<PIPE_V>();
    Csyr2CountMaskEnd();
    coefReady_ = true;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::LoadCoef(
    int32_t j, float* t1r, float* t1i, float* t2r, float* t2i)
{
    if constexpr (kFullMode == 0) {
        auto t1 = xyPlaneBuf_.Get<float>();
        auto t2 = idxBuf_.Get<float>();
        const uint32_t off = static_cast<uint32_t>(j * 2);
        *t1r = t1.GetValue(off);
        *t1i = t1.GetValue(off + 1);
        *t2r = t2.GetValue(off);
        *t2i = t2.GetValue(off + 1);

    } else {
        auto t1 = xyPlaneBuf_.Get<uint64_t>();
        auto t2 = idxBuf_.Get<uint64_t>();
        const uint64_t pair1 = t1.GetValue(static_cast<uint32_t>(j));
        const uint64_t pair2 = t2.GetValue(static_cast<uint32_t>(j));
        *t1r = __builtin_bit_cast(float, static_cast<uint32_t>(pair1));
        *t1i = __builtin_bit_cast(float, static_cast<uint32_t>(pair1 >> 32));
        *t2r = __builtin_bit_cast(float, static_cast<uint32_t>(pair2));
        *t2i = __builtin_bit_cast(float, static_cast<uint32_t>(pair2 >> 32));
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline bool Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::IsInfValue(float value) const
{
    const uint32_t bits = *reinterpret_cast<const uint32_t*>(&value);
    return (bits & 0x7fffffffU) == 0x7f800000U;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeGeneralCoefValues(
    float xr, float xi, float yr, float yi, float* t1r, float* t1i, float* t2r, float* t2i)
{
    // Keep each product separate from the following sum or difference.
    const volatile float t1r0 = yr * td_.alphaReal;
    const volatile float t1r1 = yi * td_.alphaImag;
    const volatile float t1i0 = yi * td_.alphaReal;
    const volatile float t1i1 = yr * td_.alphaImag;
    const volatile float t2r0 = xr * td_.alphaReal;
    const volatile float t2r1 = xi * td_.alphaImag;
    const volatile float t2i0 = xi * td_.alphaReal;
    const volatile float t2i1 = xr * td_.alphaImag;
    *t1r = t1r0 - t1r1;
    *t1i = t1i0 + t1i1;
    *t2r = t2r0 - t2r1;
    *t2i = t2i0 + t2i1;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeCoefValues(
    float xr, float xi, float yr, float yi, float* t1r, float* t1i, float* t2r, float* t2i)
{
    if constexpr (kAlphaOne) {
        /* 旧向量路径仍执行 1*x 与 0*x；仅 Inf 需要保留其中 0*Inf 的 NaN 语义。 */
        const bool hasInf = IsInfValue(xr) || IsInfValue(xi) || IsInfValue(yr) || IsInfValue(yi);
        if (__builtin_expect(hasInf, 0)) {
            ComputeGeneralCoefValues(xr, xi, yr, yi, t1r, t1i, t2r, t2i);
            return;
        }
        *t1r = yr;
        *t1i = yi;
        *t2r = xr;
        *t2i = xi;
        return;
    }
    /* 与原向量预计算一致：每项先单独 Muls，再 Sub/Add。volatile 防止标量复乘收缩成 FMA。 */
    ComputeGeneralCoefValues(xr, xi, yr, yi, t1r, t1i, t2r, t2i);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne, bool kXyKnownReady>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeCoef(
    int32_t j, float* t1r, float* t1i, float* t2r, float* t2i)
{
    float xr = 0.0f;
    float xi = 0.0f;
    float yr = 0.0f;
    float yi = 0.0f;
    if (kXyKnownReady || xyIlvReady_) {
        if (coefReady_) {
            LoadCoef(j, t1r, t1i, t2r, t2i);
            return;
        }
        /* BuildXyIlv 从逻辑元素 0 起缓存整段，j 即本地缓存中的元素偏移。 */
        const uint32_t off = static_cast<uint32_t>(j * 2);
        auto xIlv = upperPackedHeld_ ? xIlvHeld_ : xyStageBuf_.Get<float>();
        auto yIlv = upperPackedHeld_ ? yIlvHeld_ : aPlaneBuf_.Get<float>();
        xr = xIlv.GetValue(off);
        xi = xIlv.GetValue(off + 1);
        yr = yIlv.GetValue(off);
        yi = yIlv.GetValue(off + 1);
    } else {
        const int64_t xOff = Csyr2VectorOffset(td_.n, td_.incx, j) * 2;
        const int64_t yOff = Csyr2VectorOffset(td_.n, td_.incy, j) * 2;
        xr = xGm_.GetValue(xOff);
        xi = xGm_.GetValue(xOff + 1);
        yr = yGm_.GetValue(yOff);
        yi = yGm_.GetValue(yOff + 1);
    }
    ComputeCoefValues<kAlphaOne>(xr, xi, yr, yi, t1r, t1i, t2r, t2i);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool UseY, int Component>
__aicore__ inline float Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::UnitAlphaCoefficientBits(int32_t j)
{
    auto source = (UseY ? yIlvHeld_ : xIlvHeld_).ReinterpretCast<uint64_t>();
    const uint64_t pair = source.GetValue(static_cast<uint32_t>(j));
    uint32_t value = static_cast<uint32_t>(pair >> (Component * 32));
    const uint32_t cross = static_cast<uint32_t>(pair >> ((1 - Component) * 32));
    // alpha=(1,0): a nonfinite cross component contributes NaN through multiplication by zero.
    if ((cross & 0x7f800000u) == 0x7f800000u)
        value = 0x7fc00000u;
    return __builtin_bit_cast(float, value);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <AscendC::HardEvent Event>
__aicore__ inline __attribute__((always_inline)) void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::SyncEvent()
{
    const event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    AscendC::SetFlag<Event>(e);
    AscendC::WaitFlag<Event>(e);
}
