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
 * \file sdgmm_kernel.cpp
 * \brief Device-side SIMD AIV kernel for aclblasSdgmm (arch22 / dav-2201).
 */

#include <cstdint>
#include "kernel_operator.h"
#include "sdgmm_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t ALIGN_ELEMS = 8U;

__aicore__ inline uint64_t CalcXIndex(uint32_t logicalIdx, uint32_t xLen, int32_t incx)
{
    if (incx > 0) {
        return static_cast<uint64_t>(logicalIdx) * static_cast<uint64_t>(incx);
    }
    const uint64_t absIncx = static_cast<uint64_t>(-static_cast<int64_t>(incx));
    return static_cast<uint64_t>(xLen - 1U - logicalIdx) * absIncx;
}

__aicore__ inline void WaitEvent(HardEvent evt, TPipe* pipe)
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
    }
}

__aicore__ inline void LoadXSegment(
    GlobalTensor<float>& xGm, LocalTensor<float>& ubX, LocalTensor<float>& ubWin, const SdgmmTilingData& tiling,
    uint32_t rowOffset, uint32_t curM, TPipe* pipe)
{
    DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
    if (tiling.incx == 1) {
        DataCopyExtParams cpX{1U, curM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(ubX, xGm[rowOffset], cpX, noPad);
        WaitEvent(HardEvent::MTE2_V, pipe);
        return;
    }
    const uint64_t idxFirst = CalcXIndex(rowOffset, tiling.m, tiling.incx);
    const uint64_t idxLast = CalcXIndex(rowOffset + curM - 1U, tiling.m, tiling.incx);
    const uint64_t winStart = (idxFirst < idxLast) ? idxFirst : idxLast;
    const uint64_t winSpan = ((idxFirst < idxLast) ? (idxLast - idxFirst) : (idxFirst - idxLast)) + 1ULL;
    if (winSpan <= SDGMM_ARCH22_WIN_MAX_ELEMS) {
        DataCopyExtParams cpWin{1U, static_cast<uint32_t>(winSpan * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(ubWin, xGm[winStart], cpWin, noPad);
        WaitEvent(HardEvent::MTE2_S, pipe);
        for (uint32_t i = 0U; i < curM; ++i) {
            const uint64_t phys = CalcXIndex(rowOffset + i, tiling.m, tiling.incx);
            ubX.SetValue(i, ubWin.GetValue(static_cast<uint32_t>(phys - winStart)));
        }
        WaitEvent(HardEvent::S_V, pipe);
        return;
    }
    for (uint32_t i = 0U; i < curM; ++i) {
        ubX.SetValue(i, xGm.GetValue(CalcXIndex(rowOffset + i, tiling.m, tiling.incx)));
    }
    WaitEvent(HardEvent::S_V, pipe);
}

class SdgmmKernel {
public:
    __aicore__ inline SdgmmKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR A, GM_ADDR C, const SdgmmTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        const uint32_t blockIdx = GetBlockIdx();
        if (tiling_.mBlocks == 0U || tiling_.colBlocks == 0U) {
            active_ = false;
            return;
        }
        const uint32_t colBlockIdx = blockIdx / tiling_.mBlocks;
        const uint32_t mBlockIdx = blockIdx % tiling_.mBlocks;
        if (colBlockIdx >= tiling_.colBlocks) {
            active_ = false;
            return;
        }
        startCol_ =
            colBlockIdx * tiling_.perCoreN + ((colBlockIdx < tiling_.remainder) ? colBlockIdx : tiling_.remainder);
        const uint32_t colCount = tiling_.perCoreN + ((colBlockIdx < tiling_.remainder) ? 1U : 0U);
        endCol_ = startCol_ + colCount;

        const uint32_t startTile = mBlockIdx * tiling_.perCoreMTile +
                                   ((mBlockIdx < tiling_.mTileRemainder) ? mBlockIdx : tiling_.mTileRemainder);
        const uint32_t tileCount = tiling_.perCoreMTile + ((mBlockIdx < tiling_.mTileRemainder) ? 1U : 0U);
        startM_ = startTile * tiling_.tileM;
        endM_ = startM_ + tileCount * tiling_.tileM;
        if (endM_ > tiling_.m) {
            endM_ = tiling_.m;
        }
        active_ = (startCol_ < endCol_) && (startM_ < endM_);
        if (!active_) {
            return;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C));

        const uint32_t bufBytes = tiling_.tileM * static_cast<uint32_t>(sizeof(float));
        pipe_->InitBuffer(bufA_, bufBytes);
        pipe_->InitBuffer(bufC_, bufBytes);
        if (tiling_.mode == SDGMM_ARCH22_MODE_LEFT) {
            pipe_->InitBuffer(bufX_, bufBytes);
            pipe_->InitBuffer(bufWin_, SDGMM_ARCH22_WIN_MAX_ELEMS * static_cast<uint32_t>(sizeof(float)));
        }
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        if (tiling_.mode == SDGMM_ARCH22_MODE_LEFT) {
            ProcessLeft();
        } else {
            ProcessRight();
        }
    }

private:
    __aicore__ inline void ProcessRight()
    {
        LocalTensor<float> ubA = bufA_.Get<float>();
        LocalTensor<float> ubC = bufC_.Get<float>();
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        bool firstIter = true;
        for (uint32_t j = startCol_; j < endCol_; ++j) {
            const float xj = xGm_.GetValue(CalcXIndex(j, tiling_.n, tiling_.incx));
            for (uint32_t row = startM_; row < endM_; row += tiling_.tileM) {
                const uint32_t curM = ((endM_ - row) > tiling_.tileM) ? tiling_.tileM : (endM_ - row);
                if (!firstIter) {
                    WaitEvent(HardEvent::V_MTE2, pipe_);
                    WaitEvent(HardEvent::MTE3_V, pipe_);
                }
                const uint64_t aOff = static_cast<uint64_t>(j) * tiling_.lda + row;
                const uint64_t cOff = static_cast<uint64_t>(j) * tiling_.ldc + row;
                DataCopyExtParams cp{1U, curM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(ubA, aGm_[aOff], cp, noPad);
                WaitEvent(HardEvent::MTE2_V, pipe_);
                WaitEvent(HardEvent::S_V, pipe_);
                Muls(ubC, ubA, xj, static_cast<int32_t>(curM));
                WaitEvent(HardEvent::V_MTE3, pipe_);
                DataCopyPad(cGm_[cOff], ubC, cp);
                firstIter = false;
            }
        }
    }

    __aicore__ inline void ProcessLeft()
    {
        LocalTensor<float> ubA = bufA_.Get<float>();
        LocalTensor<float> ubC = bufC_.Get<float>();
        LocalTensor<float> ubX = bufX_.Get<float>();
        LocalTensor<float> ubWin = bufWin_.Get<float>();
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        bool firstIter = true;
        for (uint32_t row = startM_; row < endM_; row += tiling_.tileM) {
            const uint32_t curM = ((endM_ - row) > tiling_.tileM) ? tiling_.tileM : (endM_ - row);
            if (!firstIter) {
                WaitEvent(HardEvent::V_MTE2, pipe_);
                WaitEvent(HardEvent::MTE3_V, pipe_);
            }
            LoadXSegment(xGm_, ubX, ubWin, tiling_, row, curM, pipe_);
            for (uint32_t j = startCol_; j < endCol_; ++j) {
                if (j > startCol_) {
                    WaitEvent(HardEvent::V_MTE2, pipe_);
                    WaitEvent(HardEvent::MTE3_V, pipe_);
                }
                const uint64_t aOff = static_cast<uint64_t>(j) * tiling_.lda + row;
                const uint64_t cOff = static_cast<uint64_t>(j) * tiling_.ldc + row;
                DataCopyExtParams cp{1U, curM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                DataCopyPad(ubA, aGm_[aOff], cp, noPad);
                WaitEvent(HardEvent::MTE2_V, pipe_);
                Mul(ubC, ubA, ubX, static_cast<int32_t>(curM));
                WaitEvent(HardEvent::V_MTE3, pipe_);
                DataCopyPad(cGm_[cOff], ubC, cp);
                firstIter = false;
            }
        }
    }

    TPipe* pipe_{nullptr};
    GlobalTensor<float> xGm_;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> cGm_;
    TBuf<TPosition::VECCALC> bufA_;
    TBuf<TPosition::VECCALC> bufC_;
    TBuf<TPosition::VECCALC> bufX_;
    TBuf<TPosition::VECCALC> bufWin_;
    SdgmmTilingData tiling_{};
    uint32_t startCol_{0U};
    uint32_t endCol_{0U};
    uint32_t startM_{0U};
    uint32_t endM_{0U};
    bool active_{false};
};

} // namespace

extern "C" __global__ __aicore__ void sdgmm_arch22_kernel(GM_ADDR x, GM_ADDR A, GM_ADDR C, const SdgmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SdgmmKernel op;
    op.Init(x, A, C, tiling, &pipe);
    op.Process();
}

aclError sdgmm_kernel_do(
    const uint8_t* x, const uint8_t* A, uint8_t* C, const SdgmmTilingData& tiling, uint32_t numBlocks, void* stream)
{
    // Drain any stale thread-level error so the post-launch check only reports
    // this pipeline's own launch failures (issue #1181 precedent).
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    sdgmm_arch22_kernel<<<numBlocks, nullptr, stream>>>(const_cast<uint8_t*>(x), const_cast<uint8_t*>(A), C, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
