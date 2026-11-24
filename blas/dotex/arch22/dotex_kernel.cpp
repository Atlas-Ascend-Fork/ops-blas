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
 * \file dotex_kernel.cpp
 * \brief Device-side two-stage Ascend C SIMD kernel for aclblasDotEx (arch22 / dav-2201).
 */

#include <cstdint>
#include <type_traits>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "dotex_tiling_data.h"
#include "dotex_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t DOTEX_RED_TMP = 512U;

__aicore__ inline void DotexWaitEvent(HardEvent evt, TPipe* pipe)
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

class DotexKernel {
public:
    __aicore__ inline DotexKernel() = default;

    __aicore__ inline void InitPartial(
        GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const DotexTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        active_ = (tiling_.numBlocks > 0U) && (blockIdx_ < tiling_.numBlocks) && (tiling_.totalN > 0U);
        if (!active_) {
            return;
        }
        xAddr_ = x;
        yAddr_ = y;
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
        pipe_->InitBuffer(bufXIn_, DOTEX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufYIn_, DOTEX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufX32_, DOTEX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufY32_, DOTEX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufRedOut_, 64U * sizeof(float));
        pipe_->InitBuffer(bufRedTmp_, DOTEX_RED_TMP * sizeof(float));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incx != 1 || tiling_.incy != 1) {
            pipe_->InitBuffer(bufWin_, DOTEX_ARCH22_WIN_MAX * sizeof(float));
        }
    }

    __aicore__ inline void ProcessPartial()
    {
        if (!active_) {
            return;
        }
        float acc = 0.0f;
        if (tiling_.xType == static_cast<uint32_t>(ACL_FLOAT16)) {
            acc = AccumulateTyped<half>();
        } else if (tiling_.xType == static_cast<uint32_t>(ACL_BF16)) {
            acc = AccumulateTyped<bfloat16_t>();
        } else {
            acc = AccumulateTyped<float>();
        }
        LocalTensor<float> ubSc = bufScalar_.Get<float>();
        ubSc.SetValue(0, acc);
        DotexWaitEvent(HardEvent::S_MTE3, pipe_);
        DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(wsGm_[blockIdx_], ubSc, cp1);
    }

    __aicore__ inline void InitReduce(GM_ADDR workSpace, GM_ADDR result, const DotexTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        active_ = (GetBlockIdx() == 0U) && (tiling_.numBlocks > 0U);
        if (!active_) {
            return;
        }
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));
        resAddr_ = result;
        pipe_->InitBuffer(bufX32_, 256U * sizeof(float));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
    }

    __aicore__ inline void ProcessReduce()
    {
        if (!active_) {
            return;
        }
        LocalTensor<float> ubWs = bufX32_.Get<float>();
        DataCopyExtParams cpIn{1U, tiling_.numBlocks * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubWs, wsGm_, cpIn, noPad);
        DotexWaitEvent(HardEvent::MTE2_S, pipe_);
        float total = 0.0f;
        for (uint32_t i = 0U; i < tiling_.numBlocks; ++i) {
            total += ubWs.GetValue(i);
        }
        if (tiling_.xType == static_cast<uint32_t>(ACL_FLOAT16)) {
            StoreTypedResult<half>(total);
        } else if (tiling_.xType == static_cast<uint32_t>(ACL_BF16)) {
            StoreTypedResult<bfloat16_t>(total);
        } else {
            StoreTypedResult<float>(total);
        }
    }

private:
    template <typename T>
    __aicore__ inline void StoreTypedResult(float total)
    {
        GlobalTensor<T> resGm;
        resGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(resAddr_));
        LocalTensor<T> ubOut = bufScalar_.Get<T>();
        if (std::is_same<T, float>::value) {
            bufScalar_.Get<float>().SetValue(0, total);
            DotexWaitEvent(HardEvent::S_MTE3, pipe_);
        } else {
            LocalTensor<float> ubIn = bufX32_.Get<float>();
            ubIn.SetValue(0, total);
            DotexWaitEvent(HardEvent::S_V, pipe_);
            Cast(ubOut, ubIn, RoundMode::CAST_RINT, 1U);
            DotexWaitEvent(HardEvent::V_MTE3, pipe_);
        }
        DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(resGm, ubOut, cp1);
    }

    template <typename T>
    __aicore__ inline float ReduceProductTile(uint32_t count)
    {
        LocalTensor<float> x32 = bufX32_.Get<float>();
        LocalTensor<float> y32 = bufY32_.Get<float>();
        if (std::is_same<T, float>::value) {
            Mul(x32, bufXIn_.Get<float>(), bufYIn_.Get<float>(), count);
        } else {
            Cast(x32, bufXIn_.Get<T>(), RoundMode::CAST_NONE, count);
            Cast(y32, bufYIn_.Get<T>(), RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Mul(x32, x32, y32, count);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<float> redOut = bufRedOut_.Get<float>();
        LocalTensor<float> redTmp = bufRedTmp_.Get<float>();
        ReduceSum(redOut, x32, redTmp, static_cast<int32_t>(count));
        DotexWaitEvent(HardEvent::V_S, pipe_);
        return redOut.GetValue(0);
    }

    template <typename T>
    __aicore__ inline void LoadStridedTile(
        GlobalTensor<T>& gm, int64_t startOff, int64_t inc, uint32_t tileStart, uint32_t count, LocalTensor<T>& ubDst)
    {
        DataCopyPadExtParams<T> noPad{false, 0U, 0U, static_cast<T>(0.0F)};
        if (inc == 1) {
            DataCopyExtParams cp{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubDst, gm[static_cast<uint64_t>(startOff + tileStart)], cp, noPad);
            DotexWaitEvent(HardEvent::MTE2_V, pipe_);
            return;
        }
        const int64_t p0 = startOff + static_cast<int64_t>(tileStart) * inc;
        const int64_t p1 = startOff + static_cast<int64_t>(tileStart + count - 1U) * inc;
        const int64_t winLo = (p0 < p1) ? p0 : p1;
        const uint64_t span = static_cast<uint64_t>((p0 < p1) ? (p1 - p0) : (p0 - p1)) + 1ULL;
        if (span <= DOTEX_ARCH22_WIN_MAX) {
            LocalTensor<T> ubWin = bufWin_.Get<T>();
            DataCopyExtParams cpW{1U, static_cast<uint32_t>(span * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubWin, gm[static_cast<uint64_t>(winLo)], cpW, noPad);
            DotexWaitEvent(HardEvent::MTE2_S, pipe_);
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
                DotexWaitEvent(HardEvent::MTE2_S, pipe_);
                ubDst.SetValue(i, ubSc.GetValue(0));
            }
        }
        DotexWaitEvent(HardEvent::S_V, pipe_);
    }

    __aicore__ inline uint32_t ComputeTileStep(bool contiguous) const
    {
        if (contiguous) {
            return DOTEX_ARCH22_TILE;
        }
        const uint64_t sx = static_cast<uint64_t>((tiling_.incx < 0) ? -tiling_.incx : tiling_.incx);
        const uint64_t sy = static_cast<uint64_t>((tiling_.incy < 0) ? -tiling_.incy : tiling_.incy);
        const uint64_t absStride = (sx > sy) ? sx : sy;
        if (absStride <= 1ULL) {
            return DOTEX_ARCH22_TILE;
        }
        const uint32_t winCap = static_cast<uint32_t>((DOTEX_ARCH22_WIN_MAX - 1ULL) / absStride) + 1U;
        return (winCap < DOTEX_ARCH22_TILE) ? winCap : DOTEX_ARCH22_TILE;
    }

    template <typename T>
    __aicore__ inline float AccumulateTyped()
    {
        GlobalTensor<T> xGm;
        GlobalTensor<T> yGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xAddr_));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(yAddr_));
        LocalTensor<T> xIn = bufXIn_.Get<T>();
        LocalTensor<T> yIn = bufYIn_.Get<T>();

        const bool contiguous = (tiling_.incx == 1) && (tiling_.incy == 1);
        const uint32_t tileStep = ComputeTileStep(contiguous);
        DataCopyPadExtParams<T> noPad{false, 0U, 0U, static_cast<T>(0.0F)};
        float acc = 0.0f;
        for (uint32_t off = 0U; off < myCount_; off += tileStep) {
            const uint32_t cur = ((myCount_ - off) > tileStep) ? tileStep : (myCount_ - off);
            if (contiguous) {
                DataCopyExtParams cp{1U, static_cast<uint32_t>(cur * sizeof(T)), 0U, 0U, 0U};
                DataCopyPad(xIn, xGm[myOffset_ + off], cp, noPad);
                DataCopyPad(yIn, yGm[myOffset_ + off], cp, noPad);
                DotexWaitEvent(HardEvent::MTE2_V, pipe_);
            } else {
                LoadStridedTile<T>(xGm, tiling_.startOffsetX, tiling_.incx, myOffset_ + off, cur, xIn);
                LoadStridedTile<T>(yGm, tiling_.startOffsetY, tiling_.incy, myOffset_ + off, cur, yIn);
            }
            acc += ReduceProductTile<T>(cur);
        }
        return acc;
    }

    TPipe* pipe_{nullptr};
    GM_ADDR xAddr_{nullptr};
    GM_ADDR yAddr_{nullptr};
    GM_ADDR resAddr_{nullptr};
    GlobalTensor<float> wsGm_;
    TBuf<TPosition::VECCALC> bufXIn_;
    TBuf<TPosition::VECCALC> bufYIn_;
    TBuf<TPosition::VECCALC> bufX32_;
    TBuf<TPosition::VECCALC> bufY32_;
    TBuf<TPosition::VECCALC> bufRedOut_;
    TBuf<TPosition::VECCALC> bufRedTmp_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
    DotexTilingData tiling_{};
    uint32_t blockIdx_{0U};
    uint32_t myOffset_{0U};
    uint32_t myCount_{0U};
    bool active_{false};
};

} // namespace

extern "C" __global__ __aicore__ void dotex_partial_arch22_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const DotexTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    DotexKernel op;
    op.InitPartial(x, y, workSpace, tiling, &pipe);
    op.ProcessPartial();
}

extern "C" __global__ __aicore__ void dotex_reduce_arch22_kernel(
    GM_ADDR workSpace, GM_ADDR result, const DotexTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    DotexKernel op;
    op.InitReduce(workSpace, result, tiling, &pipe);
    op.ProcessReduce();
}

aclError dotex_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* result, const DotexTilingData& tiling, uint8_t* workSpace, uint32_t numBlocks,
    void* stream)
{
    // Drain any stale thread-level error so the post-launch check only reports
    // this pipeline's own launch failures (issue #1181 precedent).
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    dotex_partial_arch22_kernel<<<numBlocks, nullptr, stream>>>(x, y, workSpace, tiling);
    const aclError partialErr = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    if (partialErr != ACL_SUCCESS) {
        return partialErr;
    }
    dotex_reduce_arch22_kernel<<<1U, nullptr, stream>>>(workSpace, result, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
