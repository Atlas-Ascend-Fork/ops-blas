/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cher_kernel.cpp
 * \brief Device-side SIMD AIV kernel for aclblasCher (arch22 / dav-2201).
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "cher_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t CHER_WORK_STRIDE = 2056U;
constexpr uint32_t CHER_BUF_FLOATS = 4160U;

__aicore__ inline void CherWaitEvent(HardEvent evt, TPipe* pipe)
{
    const event_t eid = static_cast<event_t>(pipe->FetchEventID(evt));
    if (evt == HardEvent::MTE2_V) {
        SetFlag<HardEvent::MTE2_V>(eid);
        WaitFlag<HardEvent::MTE2_V>(eid);
    } else if (evt == HardEvent::V_MTE2) {
        SetFlag<HardEvent::V_MTE2>(eid);
        WaitFlag<HardEvent::V_MTE2>(eid);
    } else if (evt == HardEvent::V_MTE3) {
        SetFlag<HardEvent::V_MTE3>(eid);
        WaitFlag<HardEvent::V_MTE3>(eid);
    } else if (evt == HardEvent::MTE3_V) {
        SetFlag<HardEvent::MTE3_V>(eid);
        WaitFlag<HardEvent::MTE3_V>(eid);
    } else if (evt == HardEvent::MTE2_S) {
        SetFlag<HardEvent::MTE2_S>(eid);
        WaitFlag<HardEvent::MTE2_S>(eid);
    } else if (evt == HardEvent::S_V) {
        SetFlag<HardEvent::S_V>(eid);
        WaitFlag<HardEvent::S_V>(eid);
    } else if (evt == HardEvent::V_S) {
        SetFlag<HardEvent::V_S>(eid);
        WaitFlag<HardEvent::V_S>(eid);
    } else if (evt == HardEvent::S_MTE3) {
        SetFlag<HardEvent::S_MTE3>(eid);
        WaitFlag<HardEvent::S_MTE3>(eid);
    } else if (evt == HardEvent::MTE3_MTE2) {
        SetFlag<HardEvent::MTE3_MTE2>(eid);
        WaitFlag<HardEvent::MTE3_MTE2>(eid);
    }
}

__aicore__ inline uint64_t CherXPhysIdx(uint32_t logical, uint32_t n, int64_t incx)
{
    if (incx >= 0) {
        return static_cast<uint64_t>(logical) * static_cast<uint64_t>(incx);
    }
    return static_cast<uint64_t>(n - 1U - logical) * static_cast<uint64_t>(-incx);
}

__aicore__ inline uint32_t CherColOf(uint32_t blockIdx, uint32_t ordinal, uint32_t numBlocks)
{
    const uint32_t base = ordinal * numBlocks;
    return base + ((ordinal & 1U) != 0U ? (numBlocks - 1U - blockIdx) : blockIdx);
}

__aicore__ inline void BuildCherIndexTables(
    const LocalTensor<int32_t>& shiftIdx, const LocalTensor<int32_t>& splitIdx,
    const LocalTensor<int32_t>& sprimePackIdx, const LocalTensor<int32_t>& tmpHalf, const LocalTensor<int32_t>& tmpOdd)
{
    CreateVecIndex(shiftIdx, 0, CHER_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    ShiftRight(tmpHalf, shiftIdx, 1, CHER_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Muls(tmpOdd, tmpHalf, 2, CHER_BUF_FLOATS);
    Sub(tmpOdd, shiftIdx, tmpOdd, CHER_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    // sprime[2k] = -xi_k (located at work[CHER_WORK_STRIDE + k]), sprime[2k+1] = xr_k (at work[k])
    Duplicate(sprimePackIdx, 1, CHER_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Sub(sprimePackIdx, sprimePackIdx, tmpOdd, CHER_BUF_FLOATS);
    Muls(tmpHalf, tmpHalf, static_cast<int32_t>(sizeof(float)), CHER_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Muls(sprimePackIdx, sprimePackIdx, static_cast<int32_t>(CHER_WORK_STRIDE * sizeof(float)), CHER_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Add(sprimePackIdx, tmpHalf, sprimePackIdx, CHER_BUF_FLOATS);
    Muls(shiftIdx, shiftIdx, static_cast<int32_t>(sizeof(float)), CHER_BUF_FLOATS);
    CreateVecIndex(splitIdx, 0, CHER_ARCH22_TILE_ROWS);
    PipeBarrier<PIPE_V>();
    Muls(splitIdx, splitIdx, 8, CHER_ARCH22_TILE_ROWS);
    PipeBarrier<PIPE_V>();
}

class CherKernel {
public:
    __aicore__ inline CherKernel() = default;

    __aicore__ inline void Init(GM_ADDR alphaPtr, GM_ADDR x, GM_ADDR A, const CherTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        active_ = (tiling_.numBlocks > 0U) && (blockIdx_ < tiling_.numBlocks) && (tiling_.n > 0U);
        if (!active_) {
            return;
        }
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alphaPtr));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));

        pipe_->InitBuffer(bufXraw_, CHER_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufXsprime_, CHER_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufA_, CHER_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufDelta_, CHER_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufWork_, CHER_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufShiftIdx_, CHER_BUF_FLOATS * sizeof(int32_t));
        pipe_->InitBuffer(bufSplitIdx_, CHER_ARCH22_TILE_ROWS * sizeof(int32_t));
        pipe_->InitBuffer(bufPackIdx_, CHER_BUF_FLOATS * sizeof(int32_t));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incx != 1) {
            pipe_->InitBuffer(bufWin_, CHER_ARCH22_WIN_MAX_CPLX * 2U * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        alphaVal_ = ReadAlpha();
        if (alphaVal_ == 0.0f) {
            return;
        }
        LocalTensor<int32_t> shiftIdx = bufShiftIdx_.Get<int32_t>();
        LocalTensor<int32_t> splitIdx = bufSplitIdx_.Get<int32_t>();
        LocalTensor<int32_t> packIdx = bufPackIdx_.Get<int32_t>();
        LocalTensor<int32_t> tmpHalf = bufA_.Get<int32_t>();
        LocalTensor<int32_t> tmpOdd = bufDelta_.Get<int32_t>();
        BuildCherIndexTables(shiftIdx, splitIdx, packIdx, tmpHalf, tmpOdd);

        const bool upper = (tiling_.uplo == ACLBLAS_UPPER);
        const uint32_t totalOrds = (tiling_.n + tiling_.numBlocks - 1U) / tiling_.numBlocks;
        for (uint32_t tileStart = 0U; tileStart < tiling_.n; tileStart += tiling_.tileRows) {
            const uint32_t tileEnd =
                (tileStart + tiling_.tileRows < tiling_.n) ? (tileStart + tiling_.tileRows) : tiling_.n;
            LoadXTileAndBuildSprime(tileStart, tileEnd - tileStart);
            ProcessColumnsForTile(upper, tileStart, tileEnd, totalOrds);
        }
    }

private:
    __aicore__ inline float ReadAlpha()
    {
        if (tiling_.alphaIsDevice == 0U) {
            return tiling_.alpha;
        }
        LocalTensor<float> ubScalar = bufScalar_.Get<float>();
        DataCopyExtParams cp{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubScalar, alphaGm_, cp, noPad);
        CherWaitEvent(HardEvent::MTE2_S, pipe_);
        return ubScalar.GetValue(0);
    }

    __aicore__ inline void ReadXScalar(uint32_t j, uint32_t tileStart, uint32_t tileEnd, float& xr, float& xi)
    {
        if (j >= tileStart && j < tileEnd) {
            LocalTensor<float> ubXraw = bufXraw_.Get<float>();
            const uint32_t localIdx = (j - tileStart) * 2U;
            xr = ubXraw.GetValue(localIdx);
            xi = ubXraw.GetValue(localIdx + 1U);
            return;
        }
        LocalTensor<float> ubScalar = bufScalar_.Get<float>();
        const uint64_t phys = CherXPhysIdx(j, tiling_.n, tiling_.incx);
        DataCopyExtParams cp{1U, 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubScalar, xGm_[2ULL * phys], cp, noPad);
        CherWaitEvent(HardEvent::MTE2_S, pipe_);
        xr = ubScalar.GetValue(0);
        xi = ubScalar.GetValue(1);
    }

    __aicore__ inline void LoadXTileStrided(LocalTensor<float>& ubXraw, uint32_t tileStart, uint32_t curLen)
    {
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        const uint64_t first = CherXPhysIdx(tileStart, tiling_.n, tiling_.incx);
        const uint64_t last = CherXPhysIdx(tileStart + curLen - 1U, tiling_.n, tiling_.incx);
        const uint64_t winLo = (first < last) ? first : last;
        const uint64_t winSpan = ((first < last) ? (last - first) : (first - last)) + 1ULL;
        if (winSpan <= CHER_ARCH22_WIN_MAX_CPLX) {
            LocalTensor<float> ubWin = bufWin_.Get<float>();
            DataCopyExtParams cpWin{1U, static_cast<uint32_t>(winSpan * 8ULL), 0U, 0U, 0U};
            DataCopyPad(ubWin, xGm_[2ULL * winLo], cpWin, noPad);
            CherWaitEvent(HardEvent::MTE2_S, pipe_);
            for (uint32_t i = 0U; i < curLen; ++i) {
                const uint64_t phys = CherXPhysIdx(tileStart + i, tiling_.n, tiling_.incx);
                const uint32_t wOff = static_cast<uint32_t>(phys - winLo) * 2U;
                ubXraw.SetValue(2U * i, ubWin.GetValue(wOff));
                ubXraw.SetValue(2U * i + 1U, ubWin.GetValue(wOff + 1U));
            }
        } else {
            LocalTensor<float> ubScalar = bufScalar_.Get<float>();
            for (uint32_t i = 0U; i < curLen; ++i) {
                const uint64_t phys = CherXPhysIdx(tileStart + i, tiling_.n, tiling_.incx);
                DataCopyExtParams cp1{1U, 8U, 0U, 0U, 0U};
                DataCopyPad(ubScalar, xGm_[2ULL * phys], cp1, noPad);
                CherWaitEvent(HardEvent::MTE2_S, pipe_);
                ubXraw.SetValue(2U * i, ubScalar.GetValue(0));
                ubXraw.SetValue(2U * i + 1U, ubScalar.GetValue(1));
            }
        }
        CherWaitEvent(HardEvent::S_V, pipe_);
    }

    __aicore__ inline void LoadXTileAndBuildSprime(uint32_t tileStart, uint32_t curLen)
    {
        LocalTensor<float> ubXraw = bufXraw_.Get<float>();
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        if (tiling_.incx == 1) {
            DataCopyExtParams cp{1U, curLen * 8U, 0U, 0U, 0U};
            DataCopyPad(ubXraw, xGm_[2ULL * tileStart], cp, noPad);
            CherWaitEvent(HardEvent::MTE2_S, pipe_);
            CherWaitEvent(HardEvent::MTE2_V, pipe_);
        } else {
            LoadXTileStrided(ubXraw, tileStart, curLen);
        }
        LocalTensor<float> ubWork = bufWork_.Get<float>();
        LocalTensor<uint32_t> splitIdx = bufSplitIdx_.Get<uint32_t>();
        LocalTensor<uint32_t> packIdx = bufPackIdx_.Get<uint32_t>();
        LocalTensor<float> ubXsprime = bufXsprime_.Get<float>();
        Gather(ubWork, ubXraw, splitIdx, 0U, curLen);
        Gather(ubWork[CHER_WORK_STRIDE], ubXraw, splitIdx, sizeof(float), curLen);
        PipeBarrier<PIPE_V>();
        Muls(ubWork[CHER_WORK_STRIDE], ubWork[CHER_WORK_STRIDE], -1.0f, static_cast<int32_t>(curLen));
        PipeBarrier<PIPE_V>();
        Gather(ubXsprime, ubWork, packIdx, 0U, 2U * curLen);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UpdateColumnSegment(
        uint32_t j, uint32_t tileStart, uint32_t rowLo, uint32_t count, float xjRe, float xjIm)
    {
        const float tr = alphaVal_ * xjRe;
        const float ti = -alphaVal_ * xjIm;
        LocalTensor<float> ubA = bufA_.Get<float>();
        LocalTensor<float> ubDelta = bufDelta_.Get<float>();
        LocalTensor<float> ubXraw = bufXraw_.Get<float>();
        LocalTensor<float> ubXsprime = bufXsprime_.Get<float>();
        const uint64_t aGmOff = 2ULL * (static_cast<uint64_t>(j) * tiling_.lda + rowLo);
        DataCopyExtParams cpA{1U, count * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubA, aGm_[aGmOff], cpA, noPad);

        const uint32_t localRow = rowLo - tileStart;
        const uint32_t h = localRow & 3U;
        if (h == 0U) {
            const uint8_t rep = static_cast<uint8_t>((2U * count + 63U) / 64U);
            Muls(ubDelta, ubXraw[2U * localRow], tr, static_cast<int32_t>(2U * count));
            PipeBarrier<PIPE_V>();
            Axpy<float, float, false>(
                ubDelta, ubXsprime[2U * localRow], ti, static_cast<uint64_t>(0), rep, UnaryRepeatParams(1, 1, 8, 8));
        } else {
            const uint32_t alignedLocal = localRow & ~3U;
            const uint32_t extCount = count + h;
            const uint8_t rep = static_cast<uint8_t>((2U * extCount + 63U) / 64U);
            LocalTensor<float> ubWork = bufWork_.Get<float>();
            LocalTensor<uint32_t> shiftIdx = bufShiftIdx_.Get<uint32_t>();
            Muls(ubWork, ubXraw[2U * alignedLocal], tr, static_cast<int32_t>(2U * extCount));
            PipeBarrier<PIPE_V>();
            Axpy<float, float, false>(
                ubWork, ubXsprime[2U * alignedLocal], ti, static_cast<uint64_t>(0), rep, UnaryRepeatParams(1, 1, 8, 8));
            PipeBarrier<PIPE_V>();
            Gather(ubDelta, ubWork, shiftIdx, static_cast<uint32_t>(2U * h * sizeof(float)), 2U * count);
        }
        CherWaitEvent(HardEvent::MTE2_V, pipe_);
        PipeBarrier<PIPE_V>();
        Add(ubA, ubA, ubDelta, static_cast<int32_t>(2U * count));
        CherWaitEvent(HardEvent::V_S, pipe_);
        if (j >= rowLo && j < rowLo + count) {
            const uint32_t dLocal = j - rowLo;
            // Read old diagonal real directly from GM (ignore old diagonal imag), force imag = 0.0f
            const float oldReal = aGm_.GetValue(aGmOff + 2ULL * static_cast<uint64_t>(dLocal));
            ubA.SetValue(2U * dLocal, oldReal + alphaVal_ * (xjRe * xjRe + xjIm * xjIm));
            ubA.SetValue(2U * dLocal + 1U, 0.0f);
        }
        CherWaitEvent(HardEvent::S_MTE3, pipe_);
        DataCopyPad(aGm_[aGmOff], ubA, cpA);
        CherWaitEvent(HardEvent::MTE3_MTE2, pipe_);
    }

    __aicore__ inline void ProcessColumnsForTile(bool upper, uint32_t tileStart, uint32_t tileEnd, uint32_t totalOrds)
    {
        for (uint32_t ord = 0U; ord < totalOrds; ++ord) {
            const uint32_t j = CherColOf(blockIdx_, ord, tiling_.numBlocks);
            if (j >= tiling_.n) {
                continue;
            }
            const uint32_t colLo = upper ? 0U : j;
            const uint32_t colHi = upper ? (j + 1U) : tiling_.n;
            const uint32_t rowLo = (colLo > tileStart) ? colLo : tileStart;
            const uint32_t rowHi = (colHi < tileEnd) ? colHi : tileEnd;
            if (rowLo >= rowHi) {
                continue;
            }
            float xjRe = 0.0f;
            float xjIm = 0.0f;
            ReadXScalar(j, tileStart, tileEnd, xjRe, xjIm);
            UpdateColumnSegment(j, tileStart, rowLo, rowHi - rowLo, xjRe, xjIm);
        }
    }

    TPipe* pipe_{nullptr};
    GlobalTensor<float> alphaGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> aGm_;
    TBuf<TPosition::VECCALC> bufXraw_;
    TBuf<TPosition::VECCALC> bufXsprime_;
    TBuf<TPosition::VECCALC> bufA_;
    TBuf<TPosition::VECCALC> bufDelta_;
    TBuf<TPosition::VECCALC> bufWork_;
    TBuf<TPosition::VECCALC> bufShiftIdx_;
    TBuf<TPosition::VECCALC> bufSplitIdx_;
    TBuf<TPosition::VECCALC> bufPackIdx_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
    CherTilingData tiling_{};
    uint32_t blockIdx_{0U};
    float alphaVal_{0.0f};
    bool active_{false};
};

} // namespace

extern "C" __global__ __aicore__ void cher_arch22_kernel(
    GM_ADDR alphaPtr, GM_ADDR x, GM_ADDR A, const CherTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CherKernel op;
    op.Init(alphaPtr, x, A, tiling, &pipe);
    op.Process();
}

void cher_kernel_do(
    const uint8_t* alphaPtr, const uint8_t* x, uint8_t* A, const CherTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cher_arch22_kernel<<<numBlocks, nullptr, stream>>>(
        const_cast<uint8_t*>(alphaPtr), const_cast<uint8_t*>(x), A, tiling);
}
