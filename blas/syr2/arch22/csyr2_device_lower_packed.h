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
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeLowerPackedCount(
    AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
    AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys, AscendC::LocalTensor<float> phaseCoef1,
    AscendC::LocalTensor<float> phaseCoef2, int32_t pitchFloats, int32_t j0, int32_t b)
{
    const uint32_t n2 = static_cast<uint32_t>(pitchFloats);
    Csyr2CountMaskBegin(n2);
    /* op-major：同一轮内各列互不依赖，填满 V 流水。 */
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        Csyr2NativeMulsCount(
            aOut[static_cast<uint32_t>(kk * pitchFloats)], xIlv,
            (kUnitAlpha ? UnitAlphaCoefficientBits<true, 0>(j0 + kk) :
                          phaseCoef1.GetValue(static_cast<uint32_t>((j0 + kk) * 2 + 0))));
    }
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        Csyr2NativeAxpyCount(
            aOut[static_cast<uint32_t>(kk * pitchFloats)], xs,
            (kUnitAlpha ? UnitAlphaCoefficientBits<true, 1>(j0 + kk) :
                          phaseCoef1.GetValue(static_cast<uint32_t>((j0 + kk) * 2 + 1))));
    }
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        Csyr2NativeAxpyCount(
            aOut[static_cast<uint32_t>(kk * pitchFloats)], yIlv,
            (kUnitAlpha ? UnitAlphaCoefficientBits<false, 0>(j0 + kk) :
                          phaseCoef2.GetValue(static_cast<uint32_t>((j0 + kk) * 2 + 0))));
    }
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        Csyr2NativeAxpyCount(
            aOut[static_cast<uint32_t>(kk * pitchFloats)], ys,
            (kUnitAlpha ? UnitAlphaCoefficientBits<false, 1>(j0 + kk) :
                          phaseCoef2.GetValue(static_cast<uint32_t>((j0 + kk) * 2 + 1))));
    }
    AscendC::PipeBarrier<PIPE_V>();
    Csyr2CountMaskEnd();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline uint64_t Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::BuildLowerShortMask(
    int32_t j0, int32_t i0, int32_t cur, int32_t b)
{
    uint64_t zmask = 0;
    if (b == 4 && i0 == j0 && (j0 & 3) == 0) {
        zmask = 0x3f0f0300ULL;
    } else if (i0 < j0 + b) {
#pragma unroll
        for (int32_t kk = 0; kk < Batch; ++kk) {
            if (kk >= b)
                continue;
            int32_t prefix = (j0 + kk) - i0;
            if (prefix < 0) {
                prefix = 0;
            }
            if (prefix > cur) {
                prefix = cur;
            }
            if (prefix > 0 && prefix <= 4) {
                zmask |= ((1ULL << (prefix * 2)) - 1ULL) << (static_cast<uint32_t>(kk) * 8u);
            }
        }
    }
    return zmask;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ZeroLongLowerPrefixes(
    AscendC::LocalTensor<float> aOut, int32_t j0, int32_t i0, int32_t cur, int32_t pitchFloats, int32_t b)
{
// Three-column groups can begin three rows into an aligned block.
// Zero longer prefixes separately; the packed mask reserves only eight bits per column.
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        const int32_t prefix = j0 + kk - i0;
        if (kk < b && prefix > 4) {
            const int32_t count = prefix < cur ? prefix : cur;
            uint64_t mask[2] = {(1ULL << (count * 2)) - 1ULL, 0};
            AscendC::Duplicate(
                aOut[static_cast<uint32_t>(kk * pitchFloats)], 0.0f, mask, static_cast<uint8_t>(1),
                static_cast<uint16_t>(1), static_cast<uint8_t>(8));
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessLowerPackedGeneric(
    int32_t j0, int32_t b)
{
    auto phaseCoef1 = kUnitAlpha ? yIlvHeld_ : xyPlaneBuf_.Get<float>();
    auto phaseCoef2 = kUnitAlpha ? xIlvHeld_ : idxBuf_.Get<float>();
    const int32_t row0 = j0 & ~3;
    const int32_t len = td_.n - row0;
    const int64_t gmBase0 = static_cast<int64_t>(j0) * td_.lda + row0;
    auto tmp = tmpBuf_.Get<float>();
    for (int32_t off = 0; off < len; off += tileRow_) {
        const int32_t cur = (len - off > tileRow_) ? tileRow_ : (len - off);
        const int32_t i0 = row0 + off;
        const int32_t pitchFloats = Csyr2CeilAlign<4>(cur) * 2;
        const int32_t xyFloatOff = i0 * 2;
        auto xIlv = (lowerHeld_ ? xIlvHeld_ : xyStageBuf_.Get<float>())[static_cast<uint32_t>(xyFloatOff)];
        auto yIlv = (lowerHeld_ ? yIlvHeld_ : aPlaneBuf_.Get<float>())[static_cast<uint32_t>(xyFloatOff)];
        auto xs = tmp[static_cast<uint32_t>(xyFloatOff)];
        auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_ + xyFloatOff)];

        auto aOut = aOutQue_.AllocTensor<float>();
        ComputeLowerPackedCount<Batch>(aOut, xIlv, yIlv, xs, ys, phaseCoef1, phaseCoef2, pitchFloats, j0, b);
        const uint64_t zmask = BuildLowerShortMask<Batch>(j0, i0, cur, b);
        ZeroLongLowerPrefixes<Batch>(aOut, j0, i0, cur, pitchFloats, b);
        ApplyLowerPrefixMask(aOut, zmask, pitchFloats);
        StoreLowerBatch<Batch, false>(aOut, 0, gmBase0, off, td_.lda, cur, pitchFloats, b);
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::LoadLowerBatchCoefficients(
    int32_t j0, int32_t b, float (&c1r)[Batch], float (&c1i)[Batch], float (&c2r)[Batch], float (&c2i)[Batch])
{
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        if constexpr (kUnitAlpha) {
            const uint64_t yp = yIlvHeld_.ReinterpretCast<uint64_t>().GetValue(static_cast<uint32_t>(j0 + kk));
            const uint64_t xp = xIlvHeld_.ReinterpretCast<uint64_t>().GetValue(static_cast<uint32_t>(j0 + kk));
            const uint32_t yr = static_cast<uint32_t>(yp), yi = static_cast<uint32_t>(yp >> 32);
            const uint32_t xr = static_cast<uint32_t>(xp), xi = static_cast<uint32_t>(xp >> 32);
            c1r[kk] = __builtin_bit_cast(float, ((yi & 0x7f800000u) == 0x7f800000u) ? 0x7fc00000u : yr);
            c1i[kk] = __builtin_bit_cast(float, ((yr & 0x7f800000u) == 0x7f800000u) ? 0x7fc00000u : yi);
            c2r[kk] = __builtin_bit_cast(float, ((xi & 0x7f800000u) == 0x7f800000u) ? 0x7fc00000u : xr);
            c2i[kk] = __builtin_bit_cast(float, ((xr & 0x7f800000u) == 0x7f800000u) ? 0x7fc00000u : xi);
        } else {
            LoadCoef(j0 + kk, &c1r[kk], &c1i[kk], &c2r[kk], &c2i[kk]);
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeLowerPackedAxpyLayer(
    AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> src, const float (&coef)[Batch], int32_t pitchFloats,
    int32_t b, uint32_t n2)
{
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        if constexpr (kFullMode == 2)
            Csyr2LayerAxpy(aOut[static_cast<uint32_t>(kk * pitchFloats)], src, coef[kk], n2);
        else
            Csyr2NativeAxpyCount(aOut[static_cast<uint32_t>(kk * pitchFloats)], src, coef[kk]);
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeLowerPackedFull(
    AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
    AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys, const float (&c1r)[Batch],
    const float (&c1i)[Batch], const float (&c2r)[Batch], const float (&c2i)[Batch], int32_t pitchFloats, int32_t b)
{
    const uint32_t n2 = static_cast<uint32_t>(pitchFloats);
    if constexpr (kFullMode == 2) {
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    } else
        Csyr2CountMaskBegin(n2);
    /* op-major：同一轮内各列互不依赖，填满 V 流水。 */
    if (pitchFloats * (Batch - 1) < 2048)
        AscendC::PipeBarrier<PIPE_V>();
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        if constexpr (kFullMode == 2)
            Csyr2LayerMuls(aOut[static_cast<uint32_t>(kk * pitchFloats)], xIlv, c1r[kk], n2);
        else
            Csyr2NativeMulsCount(aOut[static_cast<uint32_t>(kk * pitchFloats)], xIlv, c1r[kk]);
    }
    ComputeLowerPackedAxpyLayer<Batch>(aOut, xs, c1i, pitchFloats, b, n2);
    ComputeLowerPackedAxpyLayer<Batch>(aOut, yIlv, c2r, pitchFloats, b, n2);
    ComputeLowerPackedAxpyLayer<Batch>(aOut, ys, c2i, pitchFloats, b, n2);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (kFullMode != 2)
        Csyr2CountMaskEnd();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::MaskLowerFullPrefix(
    AscendC::LocalTensor<float> aOut, int32_t j0, int32_t i0, int32_t cur, int32_t pitchFloats, int32_t b)
{
    uint64_t firstMask = 0;
    uint64_t secondMask = 0;
#pragma unroll
    for (int32_t kk = 0; kk < Batch; ++kk) {
        if (kk >= b)
            continue;
        int32_t prefix;
        if constexpr (kFullMode == 2 && Batch == 4) {
            // Full-tile groups begin at a multiple of four and have no row offset.
            prefix = kk;
        } else {
            prefix = j0 + kk - i0;
            if (prefix < 0)
                prefix = 0;
            if (prefix > cur)
                prefix = cur;
        }
        const int32_t first = prefix < 4 ? prefix : 4;
        const int32_t second = prefix > 4 ? prefix - 4 : 0;
        if (first)
            firstMask |= ((1ULL << (first * 2)) - 1ULL) << (kk * 8);
        if (second)
            secondMask |= ((1ULL << (second * 2)) - 1ULL) << (kk * 8);
    }
    if (firstMask) {
        uint64_t mask[2] = {firstMask, 0};
        AscendC::Duplicate(
            aOut, 0.0f, mask, static_cast<uint8_t>(1), static_cast<uint16_t>(pitchFloats / 8), static_cast<uint8_t>(8));
    }
    if (secondMask) {
        uint64_t mask[2] = {secondMask, 0};
        AscendC::Duplicate(
            aOut[8], 0.0f, mask, static_cast<uint8_t>(1), static_cast<uint16_t>(pitchFloats / 8),
            static_cast<uint8_t>(8));
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessLowerPackedFull(
    int32_t j0, int32_t b)
{
    float c1r[Batch], c1i[Batch], c2r[Batch], c2i[Batch];
    LoadLowerBatchCoefficients<Batch>(j0, b, c1r, c1i, c2r, c2i);
    const int32_t row0 = (kFullMode == 2 && Batch == 4) ? j0 : (j0 & ~3);
    const int32_t len = td_.n - row0;
    const int64_t gmBase0 = static_cast<int64_t>(j0) * td_.lda + row0;
    auto tmp = tmpBuf_.Get<float>();
    for (int32_t off = 0; off < len; off += (kFullMode == 2 ? len : tileRow_)) {
        const int32_t cur = kFullMode == 2 ? len : ((len - off > tileRow_) ? tileRow_ : (len - off));
        const int32_t i0 = row0 + off;
        const int32_t pitchFloats =
            kFullMode == 2 ? Csyr2CeilAlign < kRotateLower ? 128 : 64 > (cur * 2) : Csyr2CeilAlign<4>(cur) * 2;
        const int32_t xyFloatOff = i0 * 2;
        auto xIlv =
            ((kFullMode == 2 || lowerHeld_) ? xIlvHeld_ : xyStageBuf_.Get<float>())[static_cast<uint32_t>(xyFloatOff)];
        auto yIlv =
            ((kFullMode == 2 || lowerHeld_) ? yIlvHeld_ : aPlaneBuf_.Get<float>())[static_cast<uint32_t>(xyFloatOff)];
        auto xs = tmp[static_cast<uint32_t>(xyFloatOff)];
        auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_ + xyFloatOff)];

        auto storage = aOutQue_.AllocTensor<float>();
        const uint32_t rotation = kRotateLower ? (static_cast<uint32_t>(xyFloatOff) & 127U) : 0U;
        auto aOut = storage[rotation];
        ComputeLowerPackedFull<Batch>(aOut, xIlv, yIlv, xs, ys, c1r, c1i, c2r, c2i, pitchFloats, b);
        MaskLowerFullPrefix<Batch>(aOut, j0, i0, cur, pitchFloats, b);
        StoreLowerBatch<Batch, false, true>(storage, rotation, gmBase0, off, td_.lda, cur, pitchFloats, b);
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch, bool kCompleteGroup>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessLowerPackedOut(
    int32_t j0, int32_t b)
{
    if constexpr (kCompleteGroup)
        b = Batch;
    if constexpr (kFullMode == 0) {
        ProcessLowerPackedGeneric<Batch>(j0, b);
    } else {
        ProcessLowerPackedFull<Batch>(j0, b);
    }
}
