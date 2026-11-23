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
 * \file cher2_kernel.cpp
 * \brief Device-side SIMD AIV kernel for aclblasCher2 (arch22 / dav-2201).
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "cher2_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t CHER2_WORK_STRIDE = 1032U;
constexpr uint32_t CHER2_BUF_FLOATS = 2112U;

__aicore__ inline void Cher2WaitEvent(HardEvent evt, TPipe* pipe)
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

__aicore__ inline uint64_t Cher2VecPhysIdx(uint32_t logical, uint32_t n, int64_t inc)
{
    if (inc >= 0) {
        return static_cast<uint64_t>(logical) * static_cast<uint64_t>(inc);
    }
    return static_cast<uint64_t>(n - 1U - logical) * static_cast<uint64_t>(-inc);
}

__aicore__ inline uint32_t Cher2ColOf(uint32_t blockIdx, uint32_t ordinal, uint32_t numBlocks)
{
    const uint32_t base = ordinal * numBlocks;
    return base + ((ordinal & 1U) != 0U ? (numBlocks - 1U - blockIdx) : blockIdx);
}

__aicore__ inline void BuildCher2IndexTables(
    const LocalTensor<int32_t>& shiftIdx, const LocalTensor<int32_t>& splitIdx,
    const LocalTensor<int32_t>& sprimePackIdx, const LocalTensor<int32_t>& tmpHalf, const LocalTensor<int32_t>& tmpOdd)
{
    CreateVecIndex(shiftIdx, 0, CHER2_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    ShiftRight(tmpHalf, shiftIdx, 1, CHER2_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Muls(tmpOdd, tmpHalf, 2, CHER2_BUF_FLOATS);
    Sub(tmpOdd, shiftIdx, tmpOdd, CHER2_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Duplicate(sprimePackIdx, 1, CHER2_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Sub(sprimePackIdx, sprimePackIdx, tmpOdd, CHER2_BUF_FLOATS);
    Muls(tmpHalf, tmpHalf, static_cast<int32_t>(sizeof(float)), CHER2_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Muls(sprimePackIdx, sprimePackIdx, static_cast<int32_t>(CHER2_WORK_STRIDE * sizeof(float)), CHER2_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Add(sprimePackIdx, tmpHalf, sprimePackIdx, CHER2_BUF_FLOATS);
    Muls(shiftIdx, shiftIdx, static_cast<int32_t>(sizeof(float)), CHER2_BUF_FLOATS);
    CreateVecIndex(splitIdx, 0, CHER2_ARCH22_TILE_ROWS);
    PipeBarrier<PIPE_V>();
    Muls(splitIdx, splitIdx, 8, CHER2_ARCH22_TILE_ROWS);
    PipeBarrier<PIPE_V>();
}

class Cher2Kernel {
public:
    __aicore__ inline Cher2Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR A, const Cher2Arch22TilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        active_ = (tiling_.numBlocks > 0U) && (blockIdx_ < tiling_.numBlocks) && (tiling_.n > 0U);
        if (!active_) {
            return;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));

        pipe_->InitBuffer(bufXraw_, CHER2_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufXsprime_, CHER2_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufYraw_, CHER2_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufYsprime_, CHER2_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufA_, CHER2_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufD1_, CHER2_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufD2_, CHER2_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufWork_, CHER2_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufShiftIdx_, CHER2_BUF_FLOATS * sizeof(int32_t));
        pipe_->InitBuffer(bufSplitIdx_, CHER2_ARCH22_TILE_ROWS * sizeof(int32_t));
        pipe_->InitBuffer(bufPackIdx_, CHER2_BUF_FLOATS * sizeof(int32_t));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incx != 1 || tiling_.incy != 1) {
            pipe_->InitBuffer(bufWin_, CHER2_ARCH22_WIN_MAX_CPLX * 2U * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        LocalTensor<int32_t> shiftIdx = bufShiftIdx_.Get<int32_t>();
        LocalTensor<int32_t> splitIdx = bufSplitIdx_.Get<int32_t>();
        LocalTensor<int32_t> packIdx = bufPackIdx_.Get<int32_t>();
        LocalTensor<int32_t> tmpHalf = bufA_.Get<int32_t>();
        LocalTensor<int32_t> tmpOdd = bufD1_.Get<int32_t>();
        BuildCher2IndexTables(shiftIdx, splitIdx, packIdx, tmpHalf, tmpOdd);

        const bool upper = (tiling_.uplo == ACLBLAS_UPPER);
        const uint32_t tileRows = (tiling_.tileRows == 0U || tiling_.tileRows > CHER2_ARCH22_TILE_ROWS) ?
                                      CHER2_ARCH22_TILE_ROWS :
                                      tiling_.tileRows;
        const uint32_t totalOrds = (tiling_.n + tiling_.numBlocks - 1U) / tiling_.numBlocks;
        for (uint32_t tileStart = 0U; tileStart < tiling_.n; tileStart += tileRows) {
            const uint32_t tileEnd = (tileStart + tileRows < tiling_.n) ? (tileStart + tileRows) : tiling_.n;
            const uint32_t curLen = tileEnd - tileStart;
            LoadVectorTile(xGm_, tiling_.incx, tileStart, curLen, bufXraw_.Get<float>(), bufXsprime_.Get<float>());
            LoadVectorTile(yGm_, tiling_.incy, tileStart, curLen, bufYraw_.Get<float>(), bufYsprime_.Get<float>());
            ProcessColumnsForTile(upper, tileStart, tileEnd, totalOrds);
        }
    }

private:
    __aicore__ inline void LoadVectorTile(
        GlobalTensor<float>& vecGm, int64_t inc, uint32_t tileStart, uint32_t curLen, const LocalTensor<float>& ubRaw,
        const LocalTensor<float>& ubSprime)
    {
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        if (inc == 1) {
            DataCopyExtParams cp{1U, curLen * 8U, 0U, 0U, 0U};
            DataCopyPad(ubRaw, vecGm[2ULL * tileStart], cp, noPad);
            Cher2WaitEvent(HardEvent::MTE2_V, pipe_);
            Cher2WaitEvent(HardEvent::MTE2_S, pipe_);
        } else {
            const uint64_t first = Cher2VecPhysIdx(tileStart, tiling_.n, inc);
            const uint64_t last = Cher2VecPhysIdx(tileStart + curLen - 1U, tiling_.n, inc);
            const uint64_t winLo = (first < last) ? first : last;
            const uint64_t winSpan = ((first < last) ? (last - first) : (first - last)) + 1ULL;
            if (winSpan <= CHER2_ARCH22_WIN_MAX_CPLX) {
                LocalTensor<float> ubWin = bufWin_.Get<float>();
                DataCopyExtParams cpWin{1U, static_cast<uint32_t>(winSpan * 8ULL), 0U, 0U, 0U};
                DataCopyPad(ubWin, vecGm[2ULL * winLo], cpWin, noPad);
                Cher2WaitEvent(HardEvent::MTE2_S, pipe_);
                for (uint32_t i = 0U; i < curLen; ++i) {
                    const uint64_t phys = Cher2VecPhysIdx(tileStart + i, tiling_.n, inc);
                    const uint32_t wOff = static_cast<uint32_t>(phys - winLo) * 2U;
                    ubRaw.SetValue(2U * i, ubWin.GetValue(wOff));
                    ubRaw.SetValue(2U * i + 1U, ubWin.GetValue(wOff + 1U));
                }
            } else {
                LocalTensor<float> ubScalar = bufScalar_.Get<float>();
                for (uint32_t i = 0U; i < curLen; ++i) {
                    const uint64_t phys = Cher2VecPhysIdx(tileStart + i, tiling_.n, inc);
                    DataCopyExtParams cp1{1U, 8U, 0U, 0U, 0U};
                    DataCopyPad(ubScalar, vecGm[2ULL * phys], cp1, noPad);
                    Cher2WaitEvent(HardEvent::MTE2_S, pipe_);
                    ubRaw.SetValue(2U * i, ubScalar.GetValue(0));
                    ubRaw.SetValue(2U * i + 1U, ubScalar.GetValue(1));
                }
            }
            Cher2WaitEvent(HardEvent::S_V, pipe_);
        }
        LocalTensor<float> ubWork = bufWork_.Get<float>();
        LocalTensor<uint32_t> splitIdx = bufSplitIdx_.Get<uint32_t>();
        LocalTensor<uint32_t> packIdx = bufPackIdx_.Get<uint32_t>();
        Gather(ubWork, ubRaw, splitIdx, 0U, curLen);
        Gather(ubWork[CHER2_WORK_STRIDE], ubRaw, splitIdx, sizeof(float), curLen);
        PipeBarrier<PIPE_V>();
        Muls(ubWork[CHER2_WORK_STRIDE], ubWork[CHER2_WORK_STRIDE], -1.0f, static_cast<int32_t>(curLen));
        PipeBarrier<PIPE_V>();
        Gather(ubSprime, ubWork, packIdx, 0U, 2U * curLen);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReadVecScalar(
        GlobalTensor<float>& vecGm, int64_t inc, const LocalTensor<float>& ubRaw, uint32_t j, uint32_t tileStart,
        uint32_t tileEnd, float& vr, float& vi)
    {
        if (j >= tileStart && j < tileEnd) {
            const uint32_t idx = (j - tileStart) * 2U;
            vr = ubRaw.GetValue(idx);
            vi = ubRaw.GetValue(idx + 1U);
            return;
        }
        LocalTensor<float> ubScalar = bufScalar_.Get<float>();
        const uint64_t phys = Cher2VecPhysIdx(j, tiling_.n, inc);
        DataCopyExtParams cp{1U, 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubScalar, vecGm[2ULL * phys], cp, noPad);
        Cher2WaitEvent(HardEvent::MTE2_S, pipe_);
        vr = ubScalar.GetValue(0);
        vi = ubScalar.GetValue(1);
    }

    __aicore__ inline void ComputeInterleavedMul(
        const LocalTensor<float>& dst, const LocalTensor<float>& raw, const LocalTensor<float>& sprime,
        uint32_t localRow, uint32_t count, float tr, float ti)
    {
        const uint32_t h = localRow & 3U;
        if (h == 0U) {
            const uint8_t rep = static_cast<uint8_t>((2U * count + 63U) / 64U);
            Muls(dst, raw[2U * localRow], tr, static_cast<int32_t>(2U * count));
            PipeBarrier<PIPE_V>();
            Axpy<float, float, false>(
                dst, sprime[2U * localRow], ti, static_cast<uint64_t>(0), rep, UnaryRepeatParams(1, 1, 8, 8));
            return;
        }
        const uint32_t alignedLocal = localRow & ~3U;
        const uint32_t extCount = count + h;
        const uint8_t rep = static_cast<uint8_t>((2U * extCount + 63U) / 64U);
        LocalTensor<float> ubWork = bufWork_.Get<float>();
        LocalTensor<uint32_t> shiftIdx = bufShiftIdx_.Get<uint32_t>();
        Muls(ubWork, raw[2U * alignedLocal], tr, static_cast<int32_t>(2U * extCount));
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(
            ubWork, sprime[2U * alignedLocal], ti, static_cast<uint64_t>(0), rep, UnaryRepeatParams(1, 1, 8, 8));
        PipeBarrier<PIPE_V>();
        Gather(dst, ubWork, shiftIdx, static_cast<uint32_t>(2U * h * sizeof(float)), 2U * count);
    }

    __aicore__ inline void UpdateColumnSegment(
        uint32_t j, uint32_t tileStart, uint32_t rowLo, uint32_t count, float xjRe, float xjIm, float yjRe, float yjIm)
    {
        const uint64_t aGmOff = 2ULL * (static_cast<uint64_t>(j) * tiling_.lda + rowLo);
        if (xjRe == 0.0f && xjIm == 0.0f && yjRe == 0.0f && yjIm == 0.0f) {
            if (j >= rowLo && j < rowLo + count) {
                LocalTensor<float> ubScalar = bufScalar_.Get<float>();
                const uint64_t dGmOff = aGmOff + 2ULL * static_cast<uint64_t>(j - rowLo);
                ubScalar.SetValue(0, aGm_.GetValue(dGmOff));
                ubScalar.SetValue(1, 0.0f);
                Cher2WaitEvent(HardEvent::S_MTE3, pipe_);
                DataCopyExtParams cp1{1U, 8U, 0U, 0U, 0U};
                DataCopyPad(aGm_[dGmOff], ubScalar, cp1);
                Cher2WaitEvent(HardEvent::MTE3_MTE2, pipe_);
            }
            return;
        }

        const float t1r = tiling_.alphaReal * yjRe + tiling_.alphaImag * yjIm;
        const float t1i = tiling_.alphaImag * yjRe - tiling_.alphaReal * yjIm;
        const float axr = tiling_.alphaReal * xjRe - tiling_.alphaImag * xjIm;
        const float axi = tiling_.alphaReal * xjIm + tiling_.alphaImag * xjRe;
        const float t2r = axr;
        const float t2i = -axi;

        LocalTensor<float> ubA = bufA_.Get<float>();
        LocalTensor<float> ubD1 = bufD1_.Get<float>();
        LocalTensor<float> ubD2 = bufD2_.Get<float>();
        DataCopyExtParams cpA{1U, count * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubA, aGm_[aGmOff], cpA, noPad);

        const uint32_t localRow = rowLo - tileStart;
        ComputeInterleavedMul(ubD1, bufXraw_.Get<float>(), bufXsprime_.Get<float>(), localRow, count, t1r, t1i);
        ComputeInterleavedMul(ubD2, bufYraw_.Get<float>(), bufYsprime_.Get<float>(), localRow, count, t2r, t2i);
        Cher2WaitEvent(HardEvent::MTE2_V, pipe_);
        PipeBarrier<PIPE_V>();
        Add(ubA, ubA, ubD1, static_cast<int32_t>(2U * count));
        PipeBarrier<PIPE_V>();
        Add(ubA, ubA, ubD2, static_cast<int32_t>(2U * count));
        Cher2WaitEvent(HardEvent::V_S, pipe_);
        if (j >= rowLo && j < rowLo + count) {
            const uint32_t dLocal = j - rowLo;
            const float oldReal = aGm_.GetValue(aGmOff + 2ULL * static_cast<uint64_t>(dLocal));
            const float dRe = (xjRe * t1r - xjIm * t1i) + (yjRe * t2r - yjIm * t2i);
            ubA.SetValue(2U * dLocal, oldReal + dRe);
            ubA.SetValue(2U * dLocal + 1U, 0.0f);
        }
        Cher2WaitEvent(HardEvent::S_MTE3, pipe_);
        DataCopyPad(aGm_[aGmOff], ubA, cpA);
        Cher2WaitEvent(HardEvent::MTE3_MTE2, pipe_);
    }

    __aicore__ inline void ProcessColumnsForTile(bool upper, uint32_t tileStart, uint32_t tileEnd, uint32_t totalOrds)
    {
        LocalTensor<float> ubXraw = bufXraw_.Get<float>();
        LocalTensor<float> ubYraw = bufYraw_.Get<float>();
        for (uint32_t ord = 0U; ord < totalOrds; ++ord) {
            const uint32_t j = Cher2ColOf(blockIdx_, ord, tiling_.numBlocks);
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
            float yjRe = 0.0f;
            float yjIm = 0.0f;
            ReadVecScalar(xGm_, tiling_.incx, ubXraw, j, tileStart, tileEnd, xjRe, xjIm);
            ReadVecScalar(yGm_, tiling_.incy, ubYraw, j, tileStart, tileEnd, yjRe, yjIm);
            UpdateColumnSegment(j, tileStart, rowLo, rowHi - rowLo, xjRe, xjIm, yjRe, yjIm);
        }
    }

    TPipe* pipe_{nullptr};
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> aGm_;
    TBuf<TPosition::VECCALC> bufXraw_;
    TBuf<TPosition::VECCALC> bufXsprime_;
    TBuf<TPosition::VECCALC> bufYraw_;
    TBuf<TPosition::VECCALC> bufYsprime_;
    TBuf<TPosition::VECCALC> bufA_;
    TBuf<TPosition::VECCALC> bufD1_;
    TBuf<TPosition::VECCALC> bufD2_;
    TBuf<TPosition::VECCALC> bufWork_;
    TBuf<TPosition::VECCALC> bufShiftIdx_;
    TBuf<TPosition::VECCALC> bufSplitIdx_;
    TBuf<TPosition::VECCALC> bufPackIdx_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
    Cher2Arch22TilingData tiling_{};
    uint32_t blockIdx_{0U};
    bool active_{false};
};

} // namespace

extern "C" __global__ __aicore__ void cher2_arch22_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, const Cher2Arch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    Cher2Kernel op;
    op.Init(x, y, A, tiling, &pipe);
    op.Process();
}

void cher2_kernel_do(
    const uint8_t* x, const uint8_t* y, uint8_t* A, const Cher2Arch22TilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cher2_arch22_kernel<<<numBlocks, nullptr, stream>>>(const_cast<uint8_t*>(x), const_cast<uint8_t*>(y), A, tiling);
}
