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
 * \file snrm2_ex_kernel.cpp
 * \brief Device-side two-stage Ascend C SIMD kernel for aclblasSnrm2Ex (arch22 / dav-2201).
 */

#include <cstdint>
#include <type_traits>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "snrm2_ex_tiling_data.h"
#include "snrm2_ex_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t SNRM2_EX_RED_TMP = 512U;

__aicore__ inline void Snrm2ExWaitEvent(HardEvent evt, TPipe* pipe)
{
    const event_t eid = static_cast<event_t>(pipe->FetchEventID(evt));
    if (evt == HardEvent::MTE2_V) {
        SetFlag<HardEvent::MTE2_V>(eid);
        WaitFlag<HardEvent::MTE2_V>(eid);
    } else if (evt == HardEvent::V_MTE3) {
        SetFlag<HardEvent::V_MTE3>(eid);
        WaitFlag<HardEvent::V_MTE3>(eid);
    } else if (evt == HardEvent::MTE3_MTE2) {
        SetFlag<HardEvent::MTE3_MTE2>(eid);
        WaitFlag<HardEvent::MTE3_MTE2>(eid);
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
    }
}

class Snrm2ExKernel {
public:
    __aicore__ inline Snrm2ExKernel() = default;

    __aicore__ inline void InitPartial(GM_ADDR x, GM_ADDR workSpace, const Snrm2ExTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        active_ = (tiling_.numBlocks > 0U) && (blockIdx_ < tiling_.numBlocks) && (tiling_.totalN > 0U);
        if (!active_) {
            return;
        }
        xAddr_ = x;
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));

        if (blockIdx_ < tiling_.remainder) {
            myOffset_ = blockIdx_ * (tiling_.perCoreN + 1U);
            myCount_ = tiling_.perCoreN + 1U;
        } else {
            myOffset_ = blockIdx_ * tiling_.perCoreN + tiling_.remainder;
            myCount_ = tiling_.perCoreN;
        }
        active_ = (myCount_ > 0U);
        if (!active_) {
            return;
        }
        pipe_->InitBuffer(bufXIn_, SNRM2_EX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufX32_, SNRM2_EX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufWork_, SNRM2_EX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufRedOut_, 64U * sizeof(float));
        pipe_->InitBuffer(bufRedTmp_, SNRM2_EX_RED_TMP * sizeof(float));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incx != 1) {
            pipe_->InitBuffer(bufWin_, SNRM2_EX_ARCH22_WIN_MAX * sizeof(float));
        }
    }

    __aicore__ inline void ProcessPartial()
    {
        if (!active_) {
            return;
        }
        float scale = 0.0f;
        float ssq = 0.0f;
        if (tiling_.xType == static_cast<uint32_t>(ACL_FLOAT16)) {
            AccumulateTyped<half>(scale, ssq);
        } else {
            AccumulateTyped<float>(scale, ssq);
        }
        LocalTensor<float> ubSc = bufScalar_.Get<float>();
        ubSc.SetValue(0, scale);
        ubSc.SetValue(1, ssq);
        Snrm2ExWaitEvent(HardEvent::S_MTE3, pipe_);
        DataCopyExtParams cp2{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(wsGm_[2ULL * blockIdx_], ubSc, cp2);
    }

    __aicore__ inline void InitReduce(GM_ADDR workSpace, GM_ADDR result, const Snrm2ExTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        active_ = (GetBlockIdx() == 0U) && (tiling_.numBlocks > 0U);
        if (!active_) {
            return;
        }
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));
        resGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result));
        pipe_->InitBuffer(bufX32_, 512U * sizeof(float));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
    }

    __aicore__ inline void ProcessReduce()
    {
        if (!active_) {
            return;
        }
        LocalTensor<float> ubWs = bufX32_.Get<float>();
        DataCopyExtParams cpIn{1U, tiling_.numBlocks * 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubWs, wsGm_, cpIn, noPad);
        Snrm2ExWaitEvent(HardEvent::MTE2_S, pipe_);

        float maxScale = 0.0f;
        for (uint32_t i = 0U; i < tiling_.numBlocks; ++i) {
            const float s = ubWs.GetValue(2U * i);
            if (s > maxScale) {
                maxScale = s;
            }
        }
        LocalTensor<float> ubOut = bufScalar_.Get<float>();
        if (maxScale == 0.0f) {
            ubOut.SetValue(0, 0.0f);
            Snrm2ExWaitEvent(HardEvent::S_MTE3, pipe_);
            DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(resGm_, ubOut, cp1);
            return;
        }
        float totalSsq = 0.0f;
        for (uint32_t i = 0U; i < tiling_.numBlocks; ++i) {
            const float s = ubWs.GetValue(2U * i);
            const float q = ubWs.GetValue(2U * i + 1U);
            if (s > 0.0f) {
                const float r = s / maxScale;
                totalSsq += (r * r) * q;
            }
        }
        ubOut.SetValue(0, totalSsq);
        Snrm2ExWaitEvent(HardEvent::S_V, pipe_);
        Sqrt(ubOut, ubOut, 1U);
        PipeBarrier<PIPE_V>();
        Muls(ubOut, ubOut, maxScale, 1);
        Snrm2ExWaitEvent(HardEvent::V_MTE3, pipe_);
        DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(resGm_, ubOut, cp1);
    }

private:
    template <typename T>
    __aicore__ inline void UpdateTileScaled(uint32_t count, float& scale, float& ssq)
    {
        LocalTensor<float> x32 = bufX32_.Get<float>();
        LocalTensor<float> work = bufWork_.Get<float>();
        if (std::is_same<T, float>::value) {
            Muls(x32, bufXIn_.Get<float>(), 1.0f, static_cast<int32_t>(count));
        } else {
            Cast(x32, bufXIn_.Get<T>(), RoundMode::CAST_NONE, count);
        }
        PipeBarrier<PIPE_V>();
        Abs(work, x32, count);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> redOut = bufRedOut_.Get<float>();
        LocalTensor<float> redTmp = bufRedTmp_.Get<float>();
        ReduceMax(redOut, work, redTmp, static_cast<int32_t>(count));
        Snrm2ExWaitEvent(HardEvent::V_S, pipe_);
        const float tileMax = redOut.GetValue(0);
        if (tileMax == 0.0f) {
            return;
        }
        if (tileMax > scale) {
            if (scale == 0.0f) {
                scale = tileMax;
            } else {
                const float r = scale / tileMax;
                ssq *= (r * r);
                scale = tileMax;
            }
        }
        constexpr float TINY_SCALE_THRESHOLD = 1.0e-19f;
        constexpr float TINY_SCALE_BOOST = 18446744073709551616.0f; // 2^64, exact exponent shift
        if (scale < TINY_SCALE_THRESHOLD) {
            Muls(work, x32, TINY_SCALE_BOOST, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Muls(work, work, 1.0f / (scale * TINY_SCALE_BOOST), static_cast<int32_t>(count));
        } else {
            Muls(work, x32, 1.0f / scale, static_cast<int32_t>(count));
        }
        PipeBarrier<PIPE_V>();
        Mul(work, work, work, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(redOut, work, redTmp, static_cast<int32_t>(count));
        Snrm2ExWaitEvent(HardEvent::V_S, pipe_);
        ssq += redOut.GetValue(0);
    }

    template <typename T>
    __aicore__ inline void LoadStridedTile(
        GlobalTensor<T>& gm, int64_t startOff, int64_t inc, uint32_t tileStart, uint32_t count, LocalTensor<T>& ubDst)
    {
        DataCopyPadExtParams<T> noPad{false, 0U, 0U, static_cast<T>(0.0F)};
        const int64_t p0 = startOff + static_cast<int64_t>(tileStart) * inc;
        const int64_t p1 = startOff + static_cast<int64_t>(tileStart + count - 1U) * inc;
        const int64_t winLo = (p0 < p1) ? p0 : p1;
        const uint64_t span = static_cast<uint64_t>((p0 < p1) ? (p1 - p0) : (p0 - p1)) + 1ULL;
        if (span <= SNRM2_EX_ARCH22_WIN_MAX) {
            LocalTensor<T> ubWin = bufWin_.Get<T>();
            DataCopyExtParams cpW{1U, static_cast<uint32_t>(span * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubWin, gm[static_cast<uint64_t>(winLo)], cpW, noPad);
            Snrm2ExWaitEvent(HardEvent::MTE2_S, pipe_);
            for (uint32_t i = 0U; i < count; ++i) {
                const int64_t phys = startOff + static_cast<int64_t>(tileStart + i) * inc;
                ubDst.SetValue(i, ubWin.GetValue(static_cast<uint32_t>(phys - winLo)));
            }
        } else {
            LocalTensor<T> ubSc = bufScalar_.Get<T>();
            DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            for (uint32_t i = 0U; i < count; ++i) {
                const int64_t phys = startOff + static_cast<int64_t>(tileStart + i) * inc;
                DataCopyPad(ubSc, gm[static_cast<uint64_t>(phys)], cp1, noPad);
                Snrm2ExWaitEvent(HardEvent::MTE2_S, pipe_);
                ubDst.SetValue(i, ubSc.GetValue(0));
            }
        }
        Snrm2ExWaitEvent(HardEvent::S_V, pipe_);
    }

    __aicore__ inline uint32_t ComputeTileStep() const
    {
        const uint64_t sx = static_cast<uint64_t>((tiling_.incx < 0) ? -tiling_.incx : tiling_.incx);
        if (sx <= 1ULL) {
            return SNRM2_EX_ARCH22_TILE;
        }
        const uint32_t winCap = static_cast<uint32_t>((SNRM2_EX_ARCH22_WIN_MAX - 1ULL) / sx) + 1U;
        return (winCap < SNRM2_EX_ARCH22_TILE) ? winCap : SNRM2_EX_ARCH22_TILE;
    }

    template <typename T>
    __aicore__ inline void AccumulateTyped(float& scale, float& ssq)
    {
        GlobalTensor<T> xGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xAddr_));
        LocalTensor<T> xIn = bufXIn_.Get<T>();
        const bool contiguous = (tiling_.incx == 1);
        const uint32_t tileStep = contiguous ? SNRM2_EX_ARCH22_TILE : ComputeTileStep();
        DataCopyPadExtParams<T> noPad{false, 0U, 0U, static_cast<T>(0.0F)};
        for (uint32_t off = 0U; off < myCount_; off += tileStep) {
            const uint32_t cur = ((myCount_ - off) > tileStep) ? tileStep : (myCount_ - off);
            if (contiguous) {
                DataCopyExtParams cp{1U, static_cast<uint32_t>(cur * sizeof(T)), 0U, 0U, 0U};
                DataCopyPad(xIn, xGm[myOffset_ + off], cp, noPad);
                Snrm2ExWaitEvent(HardEvent::MTE2_V, pipe_);
            } else {
                LoadStridedTile<T>(xGm, tiling_.startOffset, tiling_.incx, myOffset_ + off, cur, xIn);
            }
            UpdateTileScaled<T>(cur, scale, ssq);
        }
    }

    TPipe* pipe_{nullptr};
    GM_ADDR xAddr_{nullptr};
    GlobalTensor<float> wsGm_;
    GlobalTensor<float> resGm_;
    TBuf<TPosition::VECCALC> bufXIn_;
    TBuf<TPosition::VECCALC> bufX32_;
    TBuf<TPosition::VECCALC> bufWork_;
    TBuf<TPosition::VECCALC> bufRedOut_;
    TBuf<TPosition::VECCALC> bufRedTmp_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
    Snrm2ExTilingData tiling_{};
    uint32_t blockIdx_{0U};
    uint32_t myOffset_{0U};
    uint32_t myCount_{0U};
    bool active_{false};
};

} // namespace

extern "C" __global__ __aicore__ void snrm2_ex_partial_arch22_kernel(
    GM_ADDR x, GM_ADDR workSpace, const Snrm2ExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    Snrm2ExKernel op;
    op.InitPartial(x, workSpace, tiling, &pipe);
    op.ProcessPartial();
}

extern "C" __global__ __aicore__ void snrm2_ex_reduce_arch22_kernel(
    GM_ADDR workSpace, GM_ADDR result, const Snrm2ExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    Snrm2ExKernel op;
    op.InitReduce(workSpace, result, tiling, &pipe);
    op.ProcessReduce();
}

void snrm2_ex_kernel_do(
    uint8_t* x, uint8_t* result, uint8_t* workSpace, const Snrm2ExTilingData& tiling, uint32_t numBlocks, void* stream)
{
    snrm2_ex_partial_arch22_kernel<<<numBlocks, nullptr, stream>>>(x, workSpace, tiling);
    snrm2_ex_reduce_arch22_kernel<<<1U, nullptr, stream>>>(workSpace, result, tiling);
}
