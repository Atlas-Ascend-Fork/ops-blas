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
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessLowerBalancedGroups()
{
    constexpr int32_t Batch = 4;
    const int32_t groups = jEnd_ / Batch;
    bool reverse = false;
    for (int32_t base = 0; base < groups; base += step_) {
        const int32_t core = reverse ? step_ - 1 - jBegin_ : jBegin_;
        const int32_t group = base + core;
        if (group < groups) {
            ProcessLowerPackedOut<Batch, true>(group * Batch, Batch);
        }
        reverse = !reverse;
    }
    const int32_t tail = jEnd_ - groups * Batch;
    if (tail != 0) {
        // Share the tail with the lighter end of the final full wave.
        const int32_t first = reverse && step_ >= Batch ? jBegin_ - (step_ - tail) : jBegin_;
        if (first >= 0) {
            for (int32_t offset = first; offset < tail; offset += step_) {
                ProcessLowerPackedOut<1, true>(groups * Batch + offset, 1);
            }
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessFullLower()
{
    BuildXYDirectSwap();
    PrecomputeCoefIlv();
    if constexpr (kUnitAlpha)
        SyncEvent<AscendC::HardEvent::MTE2_S>();
    else
        SyncEvent<AscendC::HardEvent::V_S>();
    ProcessLowerBalancedGroups();
    if (lowerHeld_) {
        aInQue_.FreeTensor(xIlvHeld_);
        aInQue_.FreeTensor(yIlvHeld_);
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ReleaseHeldXY()
{
    if constexpr (kUpper) {
        if (upperPackedHeld_) {
            aInQue_.FreeTensor(xIlvHeld_);
            aInQue_.FreeTensor(yIlvHeld_);
        }
    } else {
        if (lowerHeld_) {
            aInQue_.FreeTensor(xIlvHeld_);
            aInQue_.FreeTensor(yIlvHeld_);
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::Process()
{
    if (skip_ || jBegin_ >= jEnd_) {
        return;
    }
    if constexpr (kFullMode == 1) {
        static_assert(kUpper, "full-upper kernel requires an UPPER tiling key");
        BuildXYDirectSwap();
        PrecomputeCoefIlv();
        SyncEvent<AscendC::HardEvent::V_S>();
        ProcessUpperSingleTileQuad();
        return;
    }
    if constexpr (kFullMode == 2) {
        static_assert(!kUpper && !kWindowed, "full-lower kernel requires unit LOWER tiling");
        ProcessFullLower();
        return;
    }
    BuildXyIlv();
    if (xyIlvReady_) {
        PrecomputeCoefIlv();
        /* 向量写 xIlv/t1/t2；后续标量 GetValue 前需要 V->S。 */
        SyncEvent<AscendC::HardEvent::V_S>();
    }
    if constexpr (!kWindowed) {
        if (xyReuse_ && unionLen_ > 0 && !xyIlvReady_) {
            const int32_t i0 = kUpper ? 0 : jBegin_;
            LoadXYWindow(i0, unionLen_);
        }
    }
    if (td_.alphaReal == 1.0f && td_.alphaImag == 0.0f) {
        ProcessColumns<true>();
    } else {
        ProcessColumns<false>();
    }
    ReleaseHeldXY();
}

// The host selects this entry only for the existing full-window grouped branch.
template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <int Batch>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessFullLowerWindow()
{
    static_assert(!kUpper && kWindowed && kFullMode == 0 && !kUnitAlpha, "use the generic LOWER window layout");
    static_assert(Batch == 2 || Batch == 4, "full-window groups contain two or four columns");
    if (skip_ || jBegin_ >= jEnd_) {
        return;
    }
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    LoadXYInterleavedWindow(0, td_.n);
    PrecomputeCoefIlv();
    SyncEvent<AscendC::HardEvent::V_S>();
    // ProcessLowerWindowGroup also releases both held input queue tensors.
    ProcessLowerWindowGroup<Batch>();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessColumn(int32_t j)
{
    int32_t len = 0;
    int64_t gmBase = 0;
    int32_t iBase = 0;
    int32_t prefix = 0;
    if constexpr (kUpper) {
        len = j + 1;
        gmBase = static_cast<int64_t>(j) * td_.lda;
        iBase = 0;
    } else {
        len = td_.n - j;
        gmBase = static_cast<int64_t>(j) * td_.lda + j;
        iBase = j;
        if (xyIlvReady_) {
            const int32_t aligned = j & ~3;
            prefix = j - aligned;
            gmBase = static_cast<int64_t>(j) * td_.lda + aligned;
            iBase = aligned;
            len = td_.n - aligned;
        }
    }
    float t1r = 0.0f;
    float t1i = 0.0f;
    float t2r = 0.0f;
    float t2i = 0.0f;
    ComputeCoef<kAlphaOne>(j, &t1r, &t1i, &t2r, &t2i);
    for (int32_t off = 0; off < len; off += tileRow_) {
        const int32_t cur = (len - off > tileRow_) ? tileRow_ : (len - off);
        ChunkStep(gmBase + off, iBase + off, cur, t1r, t1i, t2r, t2i, prefix);
        prefix = 0;
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline bool Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::TryProcessLowerPacked(int32_t gStep)
{
    if constexpr (!kUpper) {
        if (xyIlvReady_ && packB_ == 4) {
            for (int32_t j0 = jBegin_ * 4; j0 < jEnd_; j0 += step_ * 4) {
                const int32_t count = (j0 + 4 > jEnd_) ? (jEnd_ - j0) : 4;
                ProcessLowerPackedOut<4>(j0, count);
            }
            return true;
        }
        if (xyIlvReady_ && packB_ > 1) {
            for (int32_t j0 = jBegin_ * packB_; j0 < jEnd_; j0 += gStep) {
                const int32_t b = (j0 + packB_ > jEnd_) ? (jEnd_ - j0) : packB_;
                if (packB_ == 3)
                    ProcessLowerPackedOut<3>(j0, b);
                else
                    ProcessLowerPackedOut<2>(j0, b);
            }
            return true;
        }
    }
    return false;
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessColumns()
{
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    if constexpr (kUpper) {
        if (packB_ == 1 && xyIlvReady_ && tileRow_ >= jEnd_) {
            if (upperQuad_) {
                ProcessUpperSingleTileQuad();
            } else if (upperPair_) {
                ProcessUpperSingleTilePair();
            } else {
                ProcessUpperSingleTile();
            }
            return;
        }
    }
    if constexpr (kWindowed) {
        ProcessWindowedColumns<kAlphaOne>();
        return;
    }
    const int32_t gStep = step_ * packB_;
    if (TryProcessLowerPacked(gStep)) {
        return;
    }
    for (int32_t j0 = jBegin_ * packB_; j0 < jEnd_; j0 += gStep) {
        int32_t b = packB_;
        if (j0 + b > jEnd_) {
            b = jEnd_ - j0;
        }
        if (xyIlvReady_ && packB_ > 1 && b > 1) {
            if constexpr (kUpper) {
                ProcessGroup<kAlphaOne>(j0, b);
                continue;
            }
        }
        for (int32_t kk = 0; kk < b; ++kk) {
            const int32_t j = j0 + kk;
            ProcessColumn<kAlphaOne>(j);
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessWindowColumns(
    int32_t row0, int32_t rowEnd)
{
    for (int32_t j = jBegin_; j < jEnd_; j += step_) {
        int32_t chunkRow = row0;
        int32_t chunkEnd = rowEnd;
        int32_t prefix = 0;
        if constexpr (kUpper) {
            if (j < row0) {
                continue;
            }
            if (j + 1 < chunkEnd) {
                chunkEnd = j + 1;
            }
        } else {
            if (j >= rowEnd) {
                continue;
            }
            if (j > row0) {
                chunkRow = row0 + ((j - row0) & ~7);
                prefix = j - chunkRow;
            }
        }
        const int32_t cur = chunkEnd - chunkRow;
        float t1r = 0.0f;
        float t1i = 0.0f;
        float t2r = 0.0f;
        float t2i = 0.0f;
        ComputeCoef<kAlphaOne>(j, &t1r, &t1i, &t2r, &t2i);
        WindowChunk(static_cast<int64_t>(j) * td_.lda + chunkRow, chunkRow - row0, cur, t1r, t1i, t2r, t2i, prefix);
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
template <bool kAlphaOne>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::ProcessWindowedColumns()
{
    const int32_t windowRows = tileRow_;
    for (int32_t row0 = 0; row0 < td_.n; row0 += windowRows) {
        const int32_t rowEnd = (row0 + windowRows < td_.n) ? (row0 + windowRows) : td_.n;
        LoadXYInterleavedWindow(row0, rowEnd - row0);
        if constexpr (kUpper) {
            if (row0 == 0 && rowEnd == td_.n) {
                PrecomputeCoefIlv();
                SyncEvent<AscendC::HardEvent::V_S>();
                if (upperQuad_) {
                    ProcessUpperSingleTileQuad();
                } else {
                    ProcessUpperSingleTilePair();
                }
                return;
            }
        } else {
            if (row0 == 0 && rowEnd == td_.n && winGroupB_ >= 2) {
                PrecomputeCoefIlv();
                SyncEvent<AscendC::HardEvent::V_S>();
                if (winGroupB_ == 4)
                    ProcessLowerWindowGroup<4>();
                else
                    ProcessLowerWindowGroup<2>();
                return;
            }
        }
        ProcessWindowColumns<kAlphaOne>(row0, rowEnd);
        aInQue_.FreeTensor(xIlvHeld_);
        aInQue_.FreeTensor(yIlvHeld_);
    }
}
