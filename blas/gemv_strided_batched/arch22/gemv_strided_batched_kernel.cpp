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
 * \file gemv_strided_batched_kernel.cpp
 * \brief Device-side Ascend C kernel for aclblasGemvStridedBatched (arch22 / dav-2201).
 */

#include <cstdint>
#include <type_traits>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "gemv_strided_batched_tiling_data.h"
#include "gemv_strided_batched_kernel.h"

using namespace AscendC;

namespace {

__aicore__ inline void GemvSbWaitEvent(HardEvent evt, TPipe* pipe)
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
    } else if (evt == HardEvent::V_MTE2) {
        SetFlag<HardEvent::V_MTE2>(eid);
        WaitFlag<HardEvent::V_MTE2>(eid);
    }
}

class GemvStridedBatchedKernel {
public:
    __aicore__ inline GemvStridedBatchedKernel() = default;

    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR x, GM_ADDR y, const GemvStridedBatchedTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        active_ = (tiling_.usedCoreNum > 0U) && (blockIdx_ < tiling_.usedCoreNum) && (tiling_.batchCount > 0U);
        if (!active_) {
            return;
        }
        startBatch_ = blockIdx_ * tiling_.batchPerCore;
        myBatches_ = (blockIdx_ == tiling_.usedCoreNum - 1U) ? tiling_.batchTail : tiling_.batchPerCore;
        active_ = (myBatches_ > 0U);
        if (!active_) {
            return;
        }
        aAddr_ = a;
        xAddr_ = x;
        yAddr_ = y;

        pipe_->InitBuffer(bufAIn_, GEMV_SB_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufA32_, GEMV_SB_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufXIn_, GEMV_SB_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufX32_, GEMV_SB_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufYIn_, GEMV_SB_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufAcc_, GEMV_SB_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufWork_, GEMV_SB_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufRed_, 64U * sizeof(float));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incx != 1 || tiling_.incy != 1) {
            pipe_->InitBuffer(bufWin_, GEMV_SB_ARCH22_WIN_MAX * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        if (tiling_.dtype == 0U) {
            ProcessTyped<half, half>();
        } else if (tiling_.dtype == 2U) {
            ProcessTyped<half, float>();
        } else if (tiling_.dtype == 3U) {
            ProcessTyped<bfloat16_t, bfloat16_t>();
        } else if (tiling_.dtype == 4U) {
            ProcessTyped<bfloat16_t, float>();
        } else {
            ProcessTyped<float, float>();
        }
    }

private:
    template <typename T>
    __aicore__ inline void LoadStridedVec(
        GlobalTensor<T>& gm, int64_t startOff, int64_t inc, uint32_t vecStart, uint32_t count, LocalTensor<T>& ubDst)
    {
        DataCopyPadExtParams<T> noPad{false, 0U, 0U, static_cast<T>(0.0F)};
        if (inc == 1) {
            DataCopyExtParams cp{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubDst, gm[static_cast<uint64_t>(startOff + vecStart)], cp, noPad);
            GemvSbWaitEvent(HardEvent::MTE2_V, pipe_);
            return;
        }
        const int64_t p0 = startOff + static_cast<int64_t>(vecStart) * inc;
        const int64_t p1 = startOff + static_cast<int64_t>(vecStart + count - 1U) * inc;
        const int64_t winLo = (p0 < p1) ? p0 : p1;
        const uint64_t span = static_cast<uint64_t>((p0 < p1) ? (p1 - p0) : (p0 - p1)) + 1ULL;
        if (span <= GEMV_SB_ARCH22_WIN_MAX) {
            LocalTensor<T> ubWin = bufWin_.Get<T>();
            DataCopyExtParams cpW{1U, static_cast<uint32_t>(span * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubWin, gm[static_cast<uint64_t>(winLo)], cpW, noPad);
            GemvSbWaitEvent(HardEvent::MTE2_S, pipe_);
            for (uint32_t i = 0U; i < count; ++i) {
                const int64_t phys = startOff + static_cast<int64_t>(vecStart + i) * inc;
                ubDst.SetValue(i, ubWin.GetValue(static_cast<uint32_t>(phys - winLo)));
            }
        } else {
            LocalTensor<T> ubSc = bufScalar_.Get<T>();
            DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            for (uint32_t i = 0U; i < count; ++i) {
                const int64_t phys = startOff + static_cast<int64_t>(vecStart + i) * inc;
                DataCopyPad(ubSc, gm[static_cast<uint64_t>(phys)], cp1, noPad);
                GemvSbWaitEvent(HardEvent::MTE2_S, pipe_);
                ubDst.SetValue(i, ubSc.GetValue(0));
            }
        }
        GemvSbWaitEvent(HardEvent::S_V, pipe_);
    }

    template <typename T>
    __aicore__ inline void StoreStridedVec(
        GlobalTensor<T>& gm, int64_t startOff, int64_t inc, uint32_t vecStart, uint32_t count, LocalTensor<T>& ubSrc)
    {
        if (inc == 1) {
            GemvSbWaitEvent(HardEvent::V_MTE3, pipe_);
            DataCopyExtParams cp{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(gm[static_cast<uint64_t>(startOff + vecStart)], ubSrc, cp);
            GemvSbWaitEvent(HardEvent::MTE3_MTE2, pipe_);
            return;
        }
        GemvSbWaitEvent(HardEvent::V_S, pipe_);
        LocalTensor<T> ubSc = bufScalar_.Get<T>();
        DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        for (uint32_t i = 0U; i < count; ++i) {
            const int64_t phys = startOff + static_cast<int64_t>(vecStart + i) * inc;
            ubSc.SetValue(0, ubSrc.GetValue(i));
            GemvSbWaitEvent(HardEvent::S_MTE3, pipe_);
            DataCopyPad(gm[static_cast<uint64_t>(phys)], ubSc, cp1);
            GemvSbWaitEvent(HardEvent::MTE3_S, pipe_);
        }
    }

    template <typename T>
    __aicore__ inline void CastToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t count)
    {
        if (std::is_same<T, float>::value) {
            Muls(dst, src.template ReinterpretCast<float>(), 1.0F, static_cast<int32_t>(count));
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, count);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void CastFromFloat(LocalTensor<T>& dst, LocalTensor<float>& src, uint32_t count)
    {
        if (std::is_same<T, float>::value) {
            Muls(dst.template ReinterpretCast<float>(), src, 1.0F, static_cast<int32_t>(count));
        } else {
            Cast(dst, src, RoundMode::CAST_RINT, count);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename TIn, typename TOut>
    __aicore__ inline void InitYAcc(
        GlobalTensor<TOut>& yGm, int64_t startOffY, uint32_t offset, uint32_t count, LocalTensor<float>& acc)
    {
        if (tiling_.beta == 0.0F) {
            Duplicate(acc, 0.0F, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            return;
        }
        LocalTensor<TOut> yIn = bufYIn_.Get<TOut>();
        LoadStridedVec<TOut>(yGm, startOffY, tiling_.incy, offset, count, yIn);
        CastToFloat<TOut>(acc, yIn, count);
        Muls(acc, acc, tiling_.beta, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    template <typename TIn>
    __aicore__ inline void AccumulateNormalCols(
        GlobalTensor<TIn>& aGm, GlobalTensor<TIn>& xGm, int64_t baseA, int64_t startOffX, uint32_t mOff, uint32_t mCnt,
        LocalTensor<float>& acc)
    {
        const uint64_t uLda = static_cast<uint64_t>(tiling_.lda);
        DataCopyPadExtParams<TIn> noPadA{false, 0U, 0U, static_cast<TIn>(0.0F)};
        LocalTensor<TIn> aIn = bufAIn_.Get<TIn>();
        LocalTensor<float> a32 = bufA32_.Get<float>();
        LocalTensor<TIn> xIn = bufXIn_.Get<TIn>();
        LocalTensor<float> x32 = bufX32_.Get<float>();
        for (uint32_t xOff = 0U; xOff < tiling_.n; xOff += GEMV_SB_ARCH22_TILE) {
            const uint32_t remX = tiling_.n - xOff;
            const uint32_t xCnt = (remX < GEMV_SB_ARCH22_TILE) ? remX : GEMV_SB_ARCH22_TILE;
            LoadStridedVec<TIn>(xGm, startOffX, tiling_.incx, xOff, xCnt, xIn);
            CastToFloat<TIn>(x32, xIn, xCnt);
            GemvSbWaitEvent(HardEvent::V_S, pipe_);
            for (uint32_t c = 0U; c < xCnt; ++c) {
                const float scale = tiling_.alpha * x32.GetValue(c);
                const uint64_t colOff = static_cast<uint64_t>(baseA) + static_cast<uint64_t>(xOff + c) * uLda + mOff;
                DataCopyExtParams cpA{1U, static_cast<uint32_t>(mCnt * sizeof(TIn)), 0U, 0U, 0U};
                DataCopyPad(aIn, aGm[colOff], cpA, noPadA);
                GemvSbWaitEvent(HardEvent::MTE2_V, pipe_);
                CastToFloat<TIn>(a32, aIn, mCnt);
                GemvSbWaitEvent(HardEvent::V_MTE2, pipe_);
                Muls(a32, a32, scale, static_cast<int32_t>(mCnt));
                PipeBarrier<PIPE_V>();
                Add(acc, acc, a32, static_cast<int32_t>(mCnt));
                PipeBarrier<PIPE_V>();
            }
        }
    }

    template <typename TIn, typename TOut>
    __aicore__ inline void ProcessNormalBatch(
        GlobalTensor<TIn>& aGm, GlobalTensor<TIn>& xGm, GlobalTensor<TOut>& yGm, int64_t baseA, int64_t startOffX,
        int64_t startOffY)
    {
        LocalTensor<TOut> yIn = bufYIn_.Get<TOut>();
        LocalTensor<float> acc = bufAcc_.Get<float>();
        for (uint32_t mOff = 0U; mOff < tiling_.m; mOff += GEMV_SB_ARCH22_TILE) {
            const uint32_t remM = tiling_.m - mOff;
            const uint32_t mCnt = (remM < GEMV_SB_ARCH22_TILE) ? remM : GEMV_SB_ARCH22_TILE;
            InitYAcc<TIn, TOut>(yGm, startOffY, mOff, mCnt, acc);
            if (tiling_.alpha != 0.0F) {
                AccumulateNormalCols<TIn>(aGm, xGm, baseA, startOffX, mOff, mCnt, acc);
            }
            CastFromFloat<TOut>(yIn, acc, mCnt);
            StoreStridedVec<TOut>(yGm, startOffY, tiling_.incy, mOff, mCnt, yIn);
        }
    }

    template <typename TIn>
    __aicore__ inline float ComputeColumnDot(
        GlobalTensor<TIn>& aGm, GlobalTensor<TIn>& xGm, int64_t baseA, int64_t startOffX, uint32_t col)
    {
        const uint64_t colBase = static_cast<uint64_t>(baseA) + static_cast<uint64_t>(col) * tiling_.lda;
        DataCopyPadExtParams<TIn> noPadA{false, 0U, 0U, static_cast<TIn>(0.0F)};
        LocalTensor<TIn> aIn = bufAIn_.Get<TIn>();
        LocalTensor<float> a32 = bufA32_.Get<float>();
        LocalTensor<TIn> xIn = bufXIn_.Get<TIn>();
        LocalTensor<float> x32 = bufX32_.Get<float>();
        LocalTensor<float> work = bufWork_.Get<float>();
        LocalTensor<float> red = bufRed_.Get<float>();
        float dot = 0.0F;
        for (uint32_t rOff = 0U; rOff < tiling_.m; rOff += GEMV_SB_ARCH22_TILE) {
            const uint32_t remR = tiling_.m - rOff;
            const uint32_t rCnt = (remR < GEMV_SB_ARCH22_TILE) ? remR : GEMV_SB_ARCH22_TILE;
            DataCopyExtParams cpA{1U, static_cast<uint32_t>(rCnt * sizeof(TIn)), 0U, 0U, 0U};
            DataCopyPad(aIn, aGm[colBase + rOff], cpA, noPadA);
            GemvSbWaitEvent(HardEvent::MTE2_V, pipe_);
            LoadStridedVec<TIn>(xGm, startOffX, tiling_.incx, rOff, rCnt, xIn);
            CastToFloat<TIn>(a32, aIn, rCnt);
            CastToFloat<TIn>(x32, xIn, rCnt);
            Mul(a32, a32, x32, static_cast<int32_t>(rCnt));
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(red, a32, work, static_cast<int32_t>(rCnt));
            GemvSbWaitEvent(HardEvent::V_S, pipe_);
            dot += red.GetValue(0);
        }
        return dot;
    }

    template <typename TIn, typename TOut>
    __aicore__ inline void ProcessTransBatch(
        GlobalTensor<TIn>& aGm, GlobalTensor<TIn>& xGm, GlobalTensor<TOut>& yGm, int64_t baseA, int64_t startOffX,
        int64_t startOffY)
    {
        LocalTensor<TOut> yIn = bufYIn_.Get<TOut>();
        LocalTensor<float> acc = bufAcc_.Get<float>();
        for (uint32_t cOff = 0U; cOff < tiling_.n; cOff += GEMV_SB_ARCH22_TILE) {
            const uint32_t remC = tiling_.n - cOff;
            const uint32_t cCnt = (remC < GEMV_SB_ARCH22_TILE) ? remC : GEMV_SB_ARCH22_TILE;
            InitYAcc<TIn, TOut>(yGm, startOffY, cOff, cCnt, acc);
            if (tiling_.alpha != 0.0F) {
                GemvSbWaitEvent(HardEvent::V_S, pipe_);
                for (uint32_t c = 0U; c < cCnt; ++c) {
                    const float dot = ComputeColumnDot<TIn>(aGm, xGm, baseA, startOffX, cOff + c);
                    acc.SetValue(c, acc.GetValue(c) + tiling_.alpha * dot);
                }
                GemvSbWaitEvent(HardEvent::S_V, pipe_);
            }
            CastFromFloat<TOut>(yIn, acc, cCnt);
            StoreStridedVec<TOut>(yGm, startOffY, tiling_.incy, cOff, cCnt, yIn);
        }
    }

    template <typename TIn, typename TOut>
    __aicore__ inline void ProcessTyped()
    {
        const uint32_t xCount = (tiling_.trans == 0U) ? tiling_.n : tiling_.m;
        const uint32_t yCount = (tiling_.trans == 0U) ? tiling_.m : tiling_.n;
        const int64_t incx64 = static_cast<int64_t>(tiling_.incx);
        const int64_t incy64 = static_cast<int64_t>(tiling_.incy);
        const int64_t negOffX = (incx64 >= 0) ? 0LL : (1LL - static_cast<int64_t>(xCount)) * incx64;
        const int64_t negOffY = (incy64 >= 0) ? 0LL : (1LL - static_cast<int64_t>(yCount)) * incy64;

        GlobalTensor<TIn> aGm;
        GlobalTensor<TIn> xGm;
        GlobalTensor<TOut> yGm;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(aAddr_));
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ TIn*>(xAddr_));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ TOut*>(yAddr_));

        for (uint32_t b = 0U; b < myBatches_; ++b) {
            const int64_t idx = static_cast<int64_t>(startBatch_ + b);
            const int64_t baseA = idx * tiling_.strideA;
            const int64_t startOffX = idx * tiling_.stridex + negOffX;
            const int64_t startOffY = idx * tiling_.stridey + negOffY;
            if (tiling_.trans == 0U) {
                ProcessNormalBatch<TIn, TOut>(aGm, xGm, yGm, baseA, startOffX, startOffY);
            } else {
                ProcessTransBatch<TIn, TOut>(aGm, xGm, yGm, baseA, startOffX, startOffY);
            }
        }
    }

    GemvStridedBatchedTilingData tiling_{};
    TPipe* pipe_ = nullptr;
    GM_ADDR aAddr_ = nullptr;
    GM_ADDR xAddr_ = nullptr;
    GM_ADDR yAddr_ = nullptr;
    uint32_t blockIdx_ = 0U;
    uint32_t startBatch_ = 0U;
    uint32_t myBatches_ = 0U;
    bool active_ = false;
    TBuf<TPosition::VECCALC> bufAIn_;
    TBuf<TPosition::VECCALC> bufA32_;
    TBuf<TPosition::VECCALC> bufXIn_;
    TBuf<TPosition::VECCALC> bufX32_;
    TBuf<TPosition::VECCALC> bufYIn_;
    TBuf<TPosition::VECCALC> bufAcc_;
    TBuf<TPosition::VECCALC> bufWork_;
    TBuf<TPosition::VECCALC> bufRed_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
};

} // namespace

extern "C" __global__ __aicore__ void gemv_strided_batched_kernel_impl(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GemvStridedBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GemvStridedBatchedKernel op;
    op.Init(a, x, y, tiling, &pipe);
    op.Process();
}

aclError gemv_strided_batched_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, const GemvStridedBatchedTilingData& tiling, uint32_t numBlocks, void* stream)
{
    // Drain any stale thread-level error so the post-launch check only reports
    // this pipeline's own launch failures (issue #1181 precedent).
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    gemv_strided_batched_kernel_impl<<<numBlocks, nullptr, stream>>>(a, x, y, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
