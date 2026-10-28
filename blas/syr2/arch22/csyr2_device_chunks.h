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
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::CopyInCols(
    AscendC::LocalTensor<float> dst, int64_t gmComplexOff, int32_t cur, int32_t b)
{
    // DMA strides are 32-bit bytes even though GM addresses are 64-bit.
    const uint64_t strideBytes = static_cast<uint64_t>(td_.lda - cur) * 8ULL;
    if (strideBytes > 0xffffffffULL) {
        const uint32_t pitch = (static_cast<uint32_t>(cur) * 2u + 7u) / 8u * 8u;
        for (int32_t col = 0; col < b; ++col) {
            const int64_t offset = (gmComplexOff + static_cast<int64_t>(col) * td_.lda) * 2;
            CopyInFloats(dst[static_cast<uint32_t>(col) * pitch], offset, cur * 2, aGm_);
        }
        return;
    }
    AscendC::DataCopyExtParams p;
    p.blockCount = static_cast<uint16_t>(b);
    p.blockLen = static_cast<uint32_t>(cur) * 8u;
    p.srcStride = static_cast<uint32_t>(strideBytes);
    p.dstStride = 0;
    p.rsv = 0;
    AscendC::DataCopyPadExtParams<float> pad(false, 0, 0, 0.0f);
    AscendC::DataCopyPad(dst, aGm_[gmComplexOff * 2], p, pad);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::CopyOutCols(
    int64_t gmComplexOff, AscendC::LocalTensor<float> src, int32_t cur, int32_t b)
{
    // DMA strides are 32-bit bytes even though GM addresses are 64-bit.
    const uint64_t strideBytes = static_cast<uint64_t>(td_.lda - cur) * 8ULL;
    if (strideBytes > 0xffffffffULL) {
        const uint32_t pitch = (static_cast<uint32_t>(cur) * 2u + 7u) / 8u * 8u;
        for (int32_t col = 0; col < b; ++col) {
            const int64_t offset = (gmComplexOff + static_cast<int64_t>(col) * td_.lda) * 2;
            CopyOutFloats(offset, src[static_cast<uint32_t>(col) * pitch], cur * 2);
        }
        return;
    }
    AscendC::DataCopyExtParams p;
    p.blockCount = static_cast<uint16_t>(b);
    p.blockLen = static_cast<uint32_t>(cur) * 8u;
    p.srcStride = 0;
    p.dstStride = static_cast<uint32_t>(strideBytes);
    p.rsv = 0;
    AscendC::DataCopyPad(aGm_[gmComplexOff * 2], src, p);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessGroup(int32_t j0, int32_t b)
{
    int32_t row0 = 0;
    int32_t len = 0;
    if constexpr (kUpper) {
        row0 = 0;
        len = j0 + b;
    } else {
        row0 = j0 & ~3;
        len = td_.n - row0;
    }
    const int64_t gmBase0 = static_cast<int64_t>(j0) * td_.lda + row0;
    for (int32_t off = 0; off < len; off += tileRow_) {
        const int32_t cur = (len - off > tileRow_) ? tileRow_ : (len - off);
        PackedChunk<kAlphaOne>(j0, b, row0 + off, cur, gmBase0 + off);
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::LoadPackedUpperCoefficients(
    int32_t j0, int32_t b, int32_t i0, int32_t cur, float (&c1r)[8], float (&c1i)[8], float (&c2r)[8], float (&c2i)[8],
    uint32_t (&nv)[8])
{
    for (int32_t kk = 0; kk < b; ++kk) {
        ComputeCoef<kAlphaOne>(j0 + kk, &c1r[kk], &c1i[kk], &c2r[kk], &c2i[kk]);
        int32_t valid = (j0 + kk + 1) - i0;
        if (valid > cur) {
            valid = cur;
        }
        nv[kk] = (valid > 0) ? static_cast<uint32_t>(valid * 2) : 0u;
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputePackedUpperLayer(
    AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> src, const float (&coef)[8], const uint32_t (&nv)[8],
    int32_t b, int32_t pitchFloats)
{
    for (int32_t kk = 0; kk < b; ++kk) {
        if (nv[kk] > 0) {
            AscendC::Axpy(aOut[static_cast<uint32_t>(kk * pitchFloats)], src, coef[kk], nv[kk]);
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::PackedUpperChunk(
    int32_t j0, int32_t b, int32_t i0, int32_t cur, int64_t gmComplexOff, int32_t pitchFloats, uint32_t nAll,
    AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv, AscendC::LocalTensor<float> xs,
    AscendC::LocalTensor<float> ys)
{
    float c1r[8];
    float c1i[8];
    float c2r[8];
    float c2i[8];
    uint32_t nv[8];
    LoadPackedUpperCoefficients<kAlphaOne>(j0, b, i0, cur, c1r, c1i, c2r, c2i, nv);
    auto aOut = aOutQue_.AllocTensor<float>();
    AscendC::Duplicate(aOut, 0.0f, static_cast<int32_t>(nAll));
    AscendC::PipeBarrier<PIPE_V>();
    /* op-major：同一轮内各列互不依赖，填满 V 流水。 */
    ComputePackedUpperLayer(aOut, xIlv, c1r, nv, b, pitchFloats);
    ComputePackedUpperLayer(aOut, xs, c1i, nv, b, pitchFloats);
    ComputePackedUpperLayer(aOut, yIlv, c2r, nv, b, pitchFloats);
    ComputePackedUpperLayer(aOut, ys, c2i, nv, b, pitchFloats);
    AscendC::PipeBarrier<PIPE_V>();
    aOutQue_.EnQue(aOut);
    aOut = aOutQue_.DeQue<float>();
    CopyOutCols(gmComplexOff, aOut, cur, b);
    aOutQue_.FreeTensor(aOut);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::PackedLowerChunk(
    int32_t j0, int32_t b, int32_t i0, int32_t cur, int64_t gmComplexOff, int32_t pitchFloats, uint32_t nAll,
    AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv, AscendC::LocalTensor<float> xs,
    AscendC::LocalTensor<float> ys)
{
    auto aIn = aInQue_.AllocTensor<float>();
    CopyInCols(aIn, gmComplexOff, cur, b);
    aInQue_.EnQue(aIn);
    aIn = aInQue_.DeQue<float>();

    auto aOut = aOutQue_.AllocTensor<float>();
    if ((static_cast<int32_t>(nAll) * static_cast<int32_t>(sizeof(float))) % 32 == 0) {
        AscendC::DataCopy(aOut, aIn, nAll);
    } else {
        AscendC::Adds(aOut, aIn, 0.0f, nAll);
    }
    AscendC::PipeBarrier<PIPE_V>();

    for (int32_t kk = 0; kk < b; ++kk) {
        float t1r = 0.0f;
        float t1i = 0.0f;
        float t2r = 0.0f;
        float t2i = 0.0f;
        ComputeCoef<kAlphaOne>(j0 + kk, &t1r, &t1i, &t2r, &t2i);
        auto seg = aOut[static_cast<uint32_t>(kk * pitchFloats)];
        int32_t prefix = (j0 + kk) - i0;
        if (prefix < 0) {
            prefix = 0;
        }
        if (prefix < cur) {
            const uint32_t nv = static_cast<uint32_t>(cur * 2);
            AscendC::Axpy(seg, xIlv, t1r, nv);
            AscendC::Axpy(seg, xs, t1i, nv);
            AscendC::Axpy(seg, yIlv, t2r, nv);
            AscendC::Axpy(seg, ys, t2i, nv);
            if (prefix > 0) {
                auto src = aIn[static_cast<uint32_t>(kk * pitchFloats)];
                AscendC::Adds(seg, src, 0.0f, static_cast<uint32_t>(prefix * 2));
            }
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
    aInQue_.FreeTensor(aIn);
    aOutQue_.EnQue(aOut);
    aOut = aOutQue_.DeQue<float>();
    CopyOutCols(gmComplexOff, aOut, cur, b);
    aOutQue_.FreeTensor(aOut);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::PackedChunk(
    int32_t j0, int32_t b, int32_t i0, int32_t cur, int64_t gmComplexOff)
{
    const int32_t pitch = Csyr2CeilAlign<4>(cur);
    const int32_t pitchFloats = pitch * 2;
    const uint32_t nAll = static_cast<uint32_t>(b * pitchFloats);
    const int32_t xyFloatOff = i0 * 2;
    auto xBase = upperPackedHeld_ ? xIlvHeld_ : xyStageBuf_.Get<float>();
    auto yBase = upperPackedHeld_ ? yIlvHeld_ : aPlaneBuf_.Get<float>();
    auto xIlv = xBase[static_cast<uint32_t>(xyFloatOff)];
    auto yIlv = yBase[static_cast<uint32_t>(xyFloatOff)];
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp[static_cast<uint32_t>(xyFloatOff)];
    auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_ + xyFloatOff)];

    if constexpr (kUpper) {
        PackedUpperChunk<kAlphaOne>(j0, b, i0, cur, gmComplexOff, pitchFloats, nAll, xIlv, yIlv, xs, ys);
        return;
    }

    PackedLowerChunk<kAlphaOne>(j0, b, i0, cur, gmComplexOff, pitchFloats, nAll, xIlv, yIlv, xs, ys);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::Accumulate(
    AscendC::LocalTensor<float> aRe, AscendC::LocalTensor<float> aIm, int32_t cur, float t1r, float t1i, float t2r,
    float t2i, int32_t xyOff)
{
    auto planes = xyPlaneBuf_.Get<float>();
    auto xr = planes[static_cast<uint32_t>(xyOff)];
    auto xi = planes[static_cast<uint32_t>(planeFloats_ + xyOff)];
    auto yr = planes[static_cast<uint32_t>(planeFloats_ * 2 + xyOff)];
    auto yi = planes[static_cast<uint32_t>(planeFloats_ * 3 + xyOff)];
    const uint32_t n = static_cast<uint32_t>(cur);
    AscendC::Axpy(aRe, xr, t1r, n);
    AscendC::Axpy(aRe, xi, -t1i, n);
    AscendC::Axpy(aRe, yr, t2r, n);
    AscendC::Axpy(aRe, yi, -t2i, n);
    AscendC::Axpy(aIm, xr, t1i, n);
    AscendC::Axpy(aIm, xi, t1r, n);
    AscendC::Axpy(aIm, yr, t2i, n);
    AscendC::Axpy(aIm, yi, t2r, n);
    AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::FinishInterleavedChunk(
    AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
    AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys, uint32_t n2, int64_t gmComplexOff, int32_t cur,
    float t1r, float t1i, float t2r, float t2i, int32_t prefix)
{
    AscendC::Muls(aOut, xIlv, t1r, n2);
    AscendC::Axpy(aOut, xs, t1i, n2);
    AscendC::Axpy(aOut, yIlv, t2r, n2);
    AscendC::Axpy(aOut, ys, t2i, n2);
    if (prefix > 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(aOut, 0.0f, prefix * 2);
    }
    AscendC::PipeBarrier<PIPE_V>();
    aOutQue_.EnQue(aOut);
    aOut = aOutQue_.DeQue<float>();
    CopyOutFloats(gmComplexOff * 2, aOut, cur * 2);
    aOutQue_.FreeTensor(aOut);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::WindowChunk(
    int64_t gmComplexOff, int32_t windowRowOff, int32_t cur, float t1r, float t1i, float t2r, float t2i, int32_t prefix)
{
    const uint32_t xyFloatOff = static_cast<uint32_t>(windowRowOff * 2);
    const uint32_t n2 = static_cast<uint32_t>(cur * 2);
    auto xIlv = xIlvHeld_[xyFloatOff];
    auto yIlv = yIlvHeld_[xyFloatOff];
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp[xyFloatOff];
    auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_) + xyFloatOff];
    auto aOut = aOutQue_.AllocTensor<float>();
    FinishInterleavedChunk(aOut, xIlv, yIlv, xs, ys, n2, gmComplexOff, cur, t1r, t1i, t2r, t2i, prefix);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::FinishPlanarChunk(
    AscendC::LocalTensor<float> aPlanes, AscendC::LocalTensor<float> aRe, AscendC::LocalTensor<float> aIm, int32_t i0,
    int32_t cur, int64_t gmComplexOff, float t1r, float t1i, float t2r, float t2i)
{
    int32_t xyOff = 0;
    if (xyReuse_) {
        xyOff = kUpper ? i0 : (i0 - jBegin_);
    }
    if ((!xyReuse_) || ((xyOff & 7) != 0)) {
        LoadXYWindow(i0, cur);
        xyOff = 0;
    }
    Accumulate(aRe, aIm, cur, t1r, t1i, t2r, t2i, xyOff);
    auto aOut = aOutQue_.AllocTensor<float>();
    auto tbl = idxBuf_.Get<uint32_t>();
    AscendC::Gather(aOut, aPlanes, tbl, static_cast<uint32_t>(0), static_cast<uint32_t>(cur * 2));
    AscendC::PipeBarrier<PIPE_V>();
    aOutQue_.EnQue(aOut);
    aOut = aOutQue_.DeQue<float>();
    CopyOutFloats(gmComplexOff * 2, aOut, cur * 2);
    aOutQue_.FreeTensor(aOut);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ChunkStep(
    int64_t gmComplexOff, int32_t i0, int32_t cur, float t1r, float t1i, float t2r, float t2i, int32_t prefix)
{
    if (xyIlvReady_) {
        const int32_t xyFloatOff = i0 * 2;
        auto aOut = aOutQue_.AllocTensor<float>();
        const uint32_t n2 = static_cast<uint32_t>(cur * 2);
        auto xIlv =
            ((kFullMode == 2 || lowerHeld_) ? xIlvHeld_ : xyStageBuf_.Get<float>())[static_cast<uint32_t>(xyFloatOff)];
        auto yIlv =
            ((kFullMode == 2 || lowerHeld_) ? yIlvHeld_ : aPlaneBuf_.Get<float>())[static_cast<uint32_t>(xyFloatOff)];
        auto tmp = tmpBuf_.Get<float>();
        auto xs = tmp[static_cast<uint32_t>(xyFloatOff)];
        auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_ + xyFloatOff)];
        FinishInterleavedChunk(aOut, xIlv, yIlv, xs, ys, n2, gmComplexOff, cur, t1r, t1i, t2r, t2i, prefix);
        return;
    }

    if constexpr (kUpper || TILING_KEY == CSYR2_TILING_KEY_LOWER_UNIT) {
        auto aPlanes = aPlaneBuf_.Get<float>();
        auto aRe = aPlanes;
        auto aIm = aPlanes[static_cast<uint32_t>(planeFloats_)];
        AscendC::Duplicate(aRe, 0.0f, planeFloats_);
        AscendC::Duplicate(aIm, 0.0f, planeFloats_);
        AscendC::PipeBarrier<PIPE_V>();
        FinishPlanarChunk(aPlanes, aRe, aIm, i0, cur, gmComplexOff, t1r, t1i, t2r, t2i);
        return;
    }

    auto aIn = aInQue_.AllocTensor<float>();
    CopyInFloats(aIn, gmComplexOff * 2, cur * 2, aGm_);
    aInQue_.EnQue(aIn);
    aIn = aInQue_.DeQue<float>();
    auto aPlanes = aPlaneBuf_.Get<float>();
    auto aRe = aPlanes;
    auto aIm = aPlanes[static_cast<uint32_t>(planeFloats_)];
    SplitPlanes(aIn, aRe, aIm, cur);
    aInQue_.FreeTensor(aIn);
    FinishPlanarChunk(aPlanes, aRe, aIm, i0, cur, gmComplexOff, t1r, t1i, t2r, t2i);
}
