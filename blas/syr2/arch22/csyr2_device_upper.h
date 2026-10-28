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
__aicore__ inline __attribute__((always_inline)) void
Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::WriteUpperSingleColumn(
    int32_t j, int32_t lda, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
    AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys)
{
    float t1r = 0.0f;
    float t1i = 0.0f;
    float t2r = 0.0f;
    float t2i = 0.0f;
    LoadCoef(j, &t1r, &t1i, &t2r, &t2i);
    auto aOut = aOutQue_.AllocTensor<float>();
    const uint32_t n2 = static_cast<uint32_t>((j + 1) * 2);
    Csyr2PaddedMuls(aOut, xIlv, t1r, n2);
    Csyr2PaddedAxpy(aOut, xs, t1i, n2);
    Csyr2PaddedAxpy(aOut, yIlv, t2r, n2);
    Csyr2PaddedAxpy(aOut, ys, t2i, n2);
    AscendC::PipeBarrier<PIPE_V>();
    aOutQue_.EnQue(aOut);
    aOut = aOutQue_.DeQue<float>();
    CopyOutFloats(static_cast<int64_t>(j) * lda * 2, aOut, static_cast<int32_t>(n2));
    aOutQue_.FreeTensor(aOut);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline __attribute__((always_inline)) void
Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::WriteUpperColumnPair(
    int32_t j, int32_t lda, uint32_t secondOff, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
    AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys)
{
    const int32_t j1 = j + step_;
    float t1r0 = 0.0f;
    float t1i0 = 0.0f;
    float t2r0 = 0.0f;
    float t2i0 = 0.0f;
    LoadCoef(j, &t1r0, &t1i0, &t2r0, &t2i0);
    float t1r1 = 0.0f;
    float t1i1 = 0.0f;
    float t2r1 = 0.0f;
    float t2i1 = 0.0f;
    LoadCoef(j1, &t1r1, &t1i1, &t2r1, &t2i1);
    auto aOut = aOutQue_.AllocTensor<float>();
    auto aOut1 = aOut[secondOff];
    const uint32_t n20 = static_cast<uint32_t>((j + 1) * 2);
    const uint32_t n21 = static_cast<uint32_t>((j1 + 1) * 2);
    Csyr2PaddedMuls(aOut, xIlv, t1r0, n20);
    Csyr2PaddedMuls(aOut1, xIlv, t1r1, n21);
    Csyr2PaddedAxpy(aOut, xs, t1i0, n20);
    Csyr2PaddedAxpy(aOut1, xs, t1i1, n21);
    Csyr2PaddedAxpy(aOut, yIlv, t2r0, n20);
    Csyr2PaddedAxpy(aOut1, yIlv, t2r1, n21);
    Csyr2PaddedAxpy(aOut, ys, t2i0, n20);
    Csyr2PaddedAxpy(aOut1, ys, t2i1, n21);
    AscendC::PipeBarrier<PIPE_V>();
    aOutQue_.EnQue(aOut);
    aOut = aOutQue_.DeQue<float>();
    CopyOutFloats(static_cast<int64_t>(j) * lda * 2, aOut, static_cast<int32_t>(n20));
    CopyOutFloats(static_cast<int64_t>(j1) * lda * 2, aOut[secondOff], static_cast<int32_t>(n21));
    aOutQue_.FreeTensor(aOut);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::LoadUpperBatchCoefficients(
    int32_t j, float (&c1r)[Batch], float (&c1i)[Batch], float (&c2r)[Batch], float (&c2i)[Batch],
    uint32_t (&n2)[Batch])
{
    if constexpr (kFullMode == 0) {
        for (int32_t q = 0; q < Batch; ++q) {
            const int32_t jq = j + step_ * q;
            LoadCoef(jq, &c1r[q], &c1i[q], &c2r[q], &c2i[q]);
            n2[q] = static_cast<uint32_t>((jq + 1) * 2);
        }
    } else {
#pragma unroll
        for (int32_t q = 0; q < Batch; ++q) {
            const int32_t jq = j + step_ * q;
            LoadCoef(jq, &c1r[q], &c1i[q], &c2r[q], &c2i[q]);
            n2[q] = static_cast<uint32_t>((jq + 1) * 2);
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool Add>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeUpperBatchElement(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float coef, uint32_t n2)
{
    if constexpr (Add) {
        Csyr2LayerAxpy(dst, src, coef, n2);
    } else {
        Csyr2LayerMuls(dst, src, coef, n2);
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch, bool Add>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ComputeUpperBatchLayer(
    AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> src, const float (&coef)[Batch],
    const uint32_t (&n2)[Batch], uint32_t segOff)
{
    if constexpr (kFullMode == 0) {
        for (int32_t q = 0; q < Batch; ++q) {
            ComputeUpperBatchElement<Add>(aOut[segOff * static_cast<uint32_t>(q)], src, coef[q], n2[q]);
        }
    } else {
#pragma unroll
        for (int32_t q = 0; q < Batch; ++q) {
            ComputeUpperBatchElement<Add>(aOut[segOff * static_cast<uint32_t>(q)], src, coef[q], n2[q]);
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::WriteUpperBatchColumns(
    int32_t j, int32_t lda, uint32_t segOff, AscendC::LocalTensor<float> aOut, const uint32_t (&n2)[Batch])
{
    if constexpr (kFullMode == 0) {
        for (int32_t q = 0; q < Batch; ++q) {
            CopyOutFloats(
                (static_cast<int64_t>(j) + step_ * q) * lda * 2, aOut[segOff * static_cast<uint32_t>(q)],
                static_cast<int32_t>(n2[q]));
        }
    } else {
#pragma unroll
        for (int32_t q = 0; q < Batch; ++q) {
            CopyOutFloats(
                (static_cast<int64_t>(j) + step_ * q) * lda * 2, aOut[segOff * static_cast<uint32_t>(q)],
                static_cast<int32_t>(n2[q]));
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::WriteUpperBatch(
    int32_t j, int32_t lda, uint32_t segOff, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
    AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys)
{
    float c1r[Batch];
    float c1i[Batch];
    float c2r[Batch];
    float c2i[Batch];
    uint32_t n2[Batch];
    LoadUpperBatchCoefficients<Batch>(j, c1r, c1i, c2r, c2i, n2);
    auto aOut = aOutQue_.AllocTensor<float>();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    AscendC::PipeBarrier<PIPE_V>();
    ComputeUpperBatchLayer<Batch, false>(aOut, xIlv, c1r, n2, segOff);
    if constexpr (kFullMode == 0) {
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
    AscendC::PipeBarrier<PIPE_V>();
    ComputeUpperBatchLayer<Batch, true>(aOut, xs, c1i, n2, segOff);
    if constexpr (kFullMode == 0) {
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
    AscendC::PipeBarrier<PIPE_V>();
    ComputeUpperBatchLayer<Batch, true>(aOut, yIlv, c2r, n2, segOff);
    if constexpr (kFullMode == 0) {
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
    AscendC::PipeBarrier<PIPE_V>();
    ComputeUpperBatchLayer<Batch, true>(aOut, ys, c2i, n2, segOff);
    AscendC::PipeBarrier<PIPE_V>();
    aOutQue_.EnQue(aOut);
    aOut = aOutQue_.DeQue<float>();
    WriteUpperBatchColumns<Batch>(j, lda, segOff, aOut, n2);
    aOutQue_.FreeTensor(aOut);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessUpperSingleTileQuad()
{
    auto xIlv = xIlvHeld_;
    auto yIlv = yIlvHeld_;
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp;
    auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_)];
    const int32_t lda = td_.lda;
    const uint32_t segOff = static_cast<uint32_t>(ilvFloats_);
    int32_t j = jBegin_;
    if (upperOct_) {
        for (; j + step_ * 7 < jEnd_; j += step_ * 8) {
            WriteUpperBatch<8>(j, lda, segOff, xIlv, yIlv, xs, ys);
        }
    }
    for (; j + step_ * 3 < jEnd_; j += step_ * 4) {
        WriteUpperBatch<4>(j, lda, segOff, xIlv, yIlv, xs, ys);
    }
    for (; j + step_ < jEnd_; j += step_ * 2) {
        WriteUpperColumnPair(j, lda, segOff, xIlv, yIlv, xs, ys);
    }
    if (j < jEnd_) {
        WriteUpperSingleColumn(j, lda, xIlv, yIlv, xs, ys);
    }
    aInQue_.FreeTensor(xIlvHeld_);
    aInQue_.FreeTensor(yIlvHeld_);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessUpperSingleTilePair()
{
    auto xIlv = xIlvHeld_;
    auto yIlv = yIlvHeld_;
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp;
    auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_)];
    const int32_t lda = td_.lda;
    const int32_t pairStep = step_ * 2;
    const uint32_t secondOff = static_cast<uint32_t>(ilvFloats_);
    int32_t j = jBegin_;
    for (; j + step_ < jEnd_; j += pairStep) {
        WriteUpperColumnPair(j, lda, secondOff, xIlv, yIlv, xs, ys);
    }
    if (j < jEnd_) {
        WriteUpperSingleColumn(j, lda, xIlv, yIlv, xs, ys);
    }
    aInQue_.FreeTensor(xIlvHeld_);
    aInQue_.FreeTensor(yIlvHeld_);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessUpperSingleTile()
{
    auto xIlv = xIlvHeld_;
    auto yIlv = yIlvHeld_;
    auto tmp = tmpBuf_.Get<float>();
    auto xs = tmp;
    auto ys = tmp[static_cast<uint32_t>(xyIlvPitch_)];
    const int32_t lda = td_.lda;
    for (int32_t j = jBegin_; j < jEnd_; j += step_) {
        WriteUpperSingleColumn(j, lda, xIlv, yIlv, xs, ys);
    }
    aInQue_.FreeTensor(xIlvHeld_);
    aInQue_.FreeTensor(yIlvHeld_);
}
