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
 * \file scalex_kernel.cpp
 * \brief Device-side Ascend C SIMD kernel for aclblasScalex (arch22 / dav-2201).
 */

#include <cstdint>
#include <type_traits>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "scalex_tiling_data.h"
#include "scalex_kernel.h"

using namespace AscendC;

namespace {

__aicore__ inline void ScalexWaitEvent(HardEvent evt, TPipe* pipe)
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
    } else if (evt == HardEvent::MTE3_S) {
        SetFlag<HardEvent::MTE3_S>(eid);
        WaitFlag<HardEvent::MTE3_S>(eid);
    } else if (evt == HardEvent::S_MTE2) {
        SetFlag<HardEvent::S_MTE2>(eid);
        WaitFlag<HardEvent::S_MTE2>(eid);
    }
}

class ScalexKernel {
public:
    __aicore__ inline ScalexKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR alpha, const ScalexTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        active_ = (tiling_.numBlocks > 0U) && (blockIdx_ < tiling_.numBlocks) && (tiling_.totalN > 0U);
        if (!active_) {
            return;
        }
        xAddr_ = x;
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
        pipe_->InitBuffer(bufXIn_, SCALEX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufX32_, SCALEX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incx != 1) {
            pipe_->InitBuffer(bufWin_, SCALEX_ARCH22_WIN_MAX * sizeof(float));
        }
        LoadAlpha(alpha);
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        if (tiling_.xType == static_cast<uint32_t>(ACL_FLOAT16)) {
            ProcessTyped<half>();
        } else if (tiling_.xType == static_cast<uint32_t>(ACL_BF16)) {
            ProcessTyped<bfloat16_t>();
        } else {
            ProcessTyped<float>();
        }
    }

private:
    __aicore__ inline void LoadAlpha(GM_ADDR alpha)
    {
        if (tiling_.alphaIsDevice == 0U || alpha == nullptr) {
            alphaVal_ = tiling_.alpha;
            return;
        }
        GlobalTensor<float> alphaGm;
        alphaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha), 1U);
        LocalTensor<float> ubAlpha = bufScalar_.Get<float>();
        DataCopyExtParams cp{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubAlpha, alphaGm, cp, noPad);
        ScalexWaitEvent(HardEvent::MTE2_S, pipe_);
        alphaVal_ = ubAlpha.GetValue(0);
    }

    template <typename T>
    __aicore__ inline void ComputeTile(uint32_t count)
    {
        LocalTensor<T> xIn = bufXIn_.Get<T>();
        LocalTensor<float> x32 = bufX32_.Get<float>();
        if (std::is_same<T, float>::value) {
            Muls(x32, bufXIn_.Get<float>(), alphaVal_, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Muls(bufXIn_.Get<float>(), x32, 1.0F, static_cast<int32_t>(count));
        } else {
            Cast(x32, xIn, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Muls(x32, x32, alphaVal_, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Cast(xIn, x32, RoundMode::CAST_RINT, count);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void LoadStridedTile(
        GlobalTensor<T>& gm, int64_t startOff, int64_t inc, uint32_t tileStart, uint32_t count, LocalTensor<T>& ubDst)
    {
        DataCopyPadExtParams<T> noPad{false, 0U, 0U, static_cast<T>(0.0F)};
        if (inc == 1) {
            DataCopyExtParams cp{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubDst, gm[static_cast<uint64_t>(startOff + tileStart)], cp, noPad);
            ScalexWaitEvent(HardEvent::MTE2_V, pipe_);
            return;
        }
        const int64_t p0 = startOff + static_cast<int64_t>(tileStart) * inc;
        const int64_t p1 = startOff + static_cast<int64_t>(tileStart + count - 1U) * inc;
        const int64_t winLo = (p0 < p1) ? p0 : p1;
        const uint64_t span = static_cast<uint64_t>((p0 < p1) ? (p1 - p0) : (p0 - p1)) + 1ULL;
        if (span <= SCALEX_ARCH22_WIN_MAX) {
            LocalTensor<T> ubWin = bufWin_.Get<T>();
            DataCopyExtParams cpW{1U, static_cast<uint32_t>(span * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubWin, gm[static_cast<uint64_t>(winLo)], cpW, noPad);
            ScalexWaitEvent(HardEvent::MTE2_S, pipe_);
            for (uint32_t i = 0U; i < count; ++i) {
                const int64_t phys = startOff + static_cast<int64_t>(tileStart + i) * inc;
                ubDst.SetValue(i, ubWin.GetValue(static_cast<uint32_t>(phys - winLo)));
            }
            ScalexWaitEvent(HardEvent::S_MTE2, pipe_);
        } else {
            LocalTensor<T> ubSc = bufScalar_.Get<T>();
            DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            for (uint32_t i = 0U; i < count; ++i) {
                const int64_t phys = startOff + static_cast<int64_t>(tileStart + i) * inc;
                DataCopyPad(ubSc, gm[static_cast<uint64_t>(phys)], cp1, noPad);
                ScalexWaitEvent(HardEvent::MTE2_S, pipe_);
                ubDst.SetValue(i, ubSc.GetValue(0));
                ScalexWaitEvent(HardEvent::S_MTE2, pipe_);
            }
        }
        ScalexWaitEvent(HardEvent::S_V, pipe_);
    }

    template <typename T>
    __aicore__ inline void StoreStridedTile(
        GlobalTensor<T>& gm, int64_t startOff, int64_t inc, uint32_t tileStart, uint32_t count, LocalTensor<T>& ubSrc)
    {
        if (inc == 1) {
            ScalexWaitEvent(HardEvent::V_MTE3, pipe_);
            DataCopyExtParams cp{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(gm[static_cast<uint64_t>(startOff + tileStart)], ubSrc, cp);
            ScalexWaitEvent(HardEvent::MTE3_MTE2, pipe_);
            return;
        }
        ScalexWaitEvent(HardEvent::V_S, pipe_);
        LocalTensor<T> ubSc = bufScalar_.Get<T>();
        DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        for (uint32_t i = 0U; i < count; ++i) {
            const int64_t phys = startOff + static_cast<int64_t>(tileStart + i) * inc;
            ubSc.SetValue(0, ubSrc.GetValue(i));
            ScalexWaitEvent(HardEvent::S_MTE3, pipe_);
            DataCopyPad(gm[static_cast<uint64_t>(phys)], ubSc, cp1);
            ScalexWaitEvent(HardEvent::MTE3_S, pipe_);
        }
    }

    template <typename T>
    __aicore__ inline void ProcessTyped()
    {
        const uint64_t absInc = static_cast<uint64_t>(tiling_.incx > 0 ? tiling_.incx : -tiling_.incx);
        const uint64_t gmSpan = 1ULL + static_cast<uint64_t>(tiling_.totalN - 1U) * absInc;
        GlobalTensor<T> xGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xAddr_), gmSpan);

        uint32_t tileStep = SCALEX_ARCH22_TILE;
        if (absInc > 1ULL && absInc < SCALEX_ARCH22_WIN_MAX) {
            const uint64_t winCount = (SCALEX_ARCH22_WIN_MAX - 1ULL) / absInc + 1ULL;
            if (winCount < tileStep) {
                tileStep = static_cast<uint32_t>(winCount);
            }
        }
        for (uint32_t done = 0U; done < myCount_; done += tileStep) {
            const uint32_t rem = myCount_ - done;
            const uint32_t count = (rem < tileStep) ? rem : tileStep;
            LocalTensor<T> xIn = bufXIn_.Get<T>();
            LoadStridedTile<T>(xGm, tiling_.startOffsetX, tiling_.incx, myOffset_ + done, count, xIn);
            ComputeTile<T>(count);
            StoreStridedTile<T>(xGm, tiling_.startOffsetX, tiling_.incx, myOffset_ + done, count, xIn);
        }
    }

    ScalexTilingData tiling_{};
    TPipe* pipe_ = nullptr;
    GM_ADDR xAddr_ = nullptr;
    uint32_t blockIdx_ = 0U;
    uint32_t myOffset_ = 0U;
    uint32_t myCount_ = 0U;
    float alphaVal_ = 0.0F;
    bool active_ = false;
    TBuf<TPosition::VECCALC> bufXIn_;
    TBuf<TPosition::VECCALC> bufX32_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
};

} // namespace

extern "C" __global__ __aicore__ void scalex_kernel_impl(GM_ADDR x, GM_ADDR alpha, ScalexTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ScalexKernel op;
    op.Init(x, alpha, tiling, &pipe);
    op.Process();
}

aclError scalex_kernel_do(uint8_t* x, void* alpha, const ScalexTilingData& tiling, uint32_t numBlocks, void* stream)
{
    // Drain any stale thread-level error so the post-launch check only reports
    // this launch's own failure (issue #1181 precedent).
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    scalex_kernel_impl<<<numBlocks, nullptr, stream>>>(x, static_cast<uint8_t*>(alpha), tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
