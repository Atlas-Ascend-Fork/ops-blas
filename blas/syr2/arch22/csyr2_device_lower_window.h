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
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeLowerWindowBatch(
    AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
    AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys, AscendC::LocalTensor<float> phaseCoef1,
    AscendC::LocalTensor<float> phaseCoef2, int32_t pitchFloats, int32_t j0, int32_t b)
{
    Csyr2CountMaskBegin(static_cast<uint32_t>(pitchFloats));
    /* op-major：同一轮内各列互不依赖，填满 V 流水。 */
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        Csyr2NativeMulsCount(
            aOut[static_cast<uint32_t>(kk * pitchFloats)], xIlv,
            phaseCoef1.GetValue(static_cast<uint32_t>((j0 + kk) * 2 + 0)));
    }
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        Csyr2NativeAxpyCount(
            aOut[static_cast<uint32_t>(kk * pitchFloats)], xs,
            phaseCoef1.GetValue(static_cast<uint32_t>((j0 + kk) * 2 + 1)));
    }
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        Csyr2NativeAxpyCount(
            aOut[static_cast<uint32_t>(kk * pitchFloats)], yIlv,
            phaseCoef2.GetValue(static_cast<uint32_t>((j0 + kk) * 2 + 0)));
    }
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        Csyr2NativeAxpyCount(
            aOut[static_cast<uint32_t>(kk * pitchFloats)], ys,
            phaseCoef2.GetValue(static_cast<uint32_t>((j0 + kk) * 2 + 1)));
    }
    AscendC::PipeBarrier<PIPE_V>();
    Csyr2CountMaskEnd();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ApplyLowerPrefixMask(
    AscendC::LocalTensor<float> aOut, uint64_t zmask, int32_t pitchFloats)
{
    if (zmask != 0) {
        uint64_t bitMask[2];
        bitMask[0] = zmask;
        bitMask[1] = 0;
        AscendC::Duplicate(
            aOut, 0.0f, bitMask, static_cast<uint8_t>(1), static_cast<uint16_t>(pitchFloats / 8),
            static_cast<uint8_t>(8));
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::MaskLowerWindowPrefix(
    AscendC::LocalTensor<float> aOut, int32_t j0, int32_t row0, int32_t pitchFloats, int32_t b)
{
    uint64_t zmask = 0;
    const int32_t prefixBase = j0 - row0; /* b=2 时可为 2，prefix 最大 4，掩码 8 bit 内 */
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        const int32_t prefix = prefixBase + kk;
        if (prefix > 0) {
            zmask |= ((1ULL << (prefix * 2)) - 1ULL) << (static_cast<uint32_t>(kk) * 8u);
        }
    }
    ApplyLowerPrefixMask(aOut, zmask, pitchFloats);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool Window>
__aicore__ inline int64_t Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::LowerColumnFloatOffset(
    int64_t base, int32_t row, int32_t kk, int32_t lda)
{
    if constexpr (Window) {
        return ((base + kk) * lda + row) * 2;
    } else {
        return (base + row + static_cast<int64_t>(kk) * td_.lda) * 2;
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch, bool Window, bool Rotated>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::StoreLowerBatch(
    AscendC::LocalTensor<float> storage, uint32_t rotation, int64_t base, int32_t row, int32_t lda, int32_t cur,
    int32_t pitchFloats, int32_t b)
{
    AscendC::PipeBarrier<PIPE_V>();
    aOutQue_.EnQue(storage);
    storage = aOutQue_.DeQue<float>();
    auto aOut = storage;
    if constexpr (Rotated) {
        aOut = storage[rotation];
    }
// Each column retains a separate contiguous GM copy.
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        CopyOutFloats(
            LowerColumnFloatOffset<Window>(base, row, kk, lda), aOut[static_cast<uint32_t>(kk * pitchFloats)], cur * 2);
    }
    aOutQue_.FreeTensor(storage);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessLowerWindowGroup()
{
    auto phaseCoef1 = kUnitAlpha ? yIlvHeld_ : xyPlaneBuf_.Get<float>();
    auto phaseCoef2 = kUnitAlpha ? xIlvHeld_ : idxBuf_.Get<float>();
    auto tmp = tmpBuf_.Get<float>();
    const int32_t n = td_.n;
    const int32_t lda = td_.lda;
    const int32_t bMax = Batch;
    for (int32_t wave = 0;; ++wave) {
        const bool reverse = td_.n >= 1536 && td_.n <= 2112 && (wave & 1) != 0;
        const int32_t lane = reverse ? step_ - 1 - jBegin_ : jBegin_;
        const int32_t j0 = (wave * step_ + lane) * bMax;
        if (j0 >= jEnd_)
            break;
        int32_t b = bMax;
        if (j0 + b > jEnd_) {
            b = jEnd_ - j0;
        }
        const int32_t row0 = j0 & ~3; /* 向量起始需 32B 对齐：行偏移取 4 的倍数 */
        const int32_t cur = n - row0;
        const int32_t pitchFloats = Csyr2CeilAlign<4>(cur) * 2;
        const uint32_t xyFloatOff = static_cast<uint32_t>(row0 * 2);
        auto xIlv = xIlvHeld_[xyFloatOff];
        auto yIlv = yIlvHeld_[xyFloatOff];
        auto xs = tmp[xyFloatOff];
        auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_) + xyFloatOff];

        auto aOut = aOutQue_.AllocTensor<float>();
        ComputeLowerWindowBatch<Batch>(aOut, xIlv, yIlv, xs, ys, phaseCoef1, phaseCoef2, pitchFloats, j0, b);
        MaskLowerWindowPrefix<Batch>(aOut, j0, row0, pitchFloats, b);
        StoreLowerBatch<Batch, true>(aOut, 0, j0, row0, lda, cur, pitchFloats, b);
    }
    aInQue_.FreeTensor(xIlvHeld_);
    aInQue_.FreeTensor(yIlvHeld_);
}
