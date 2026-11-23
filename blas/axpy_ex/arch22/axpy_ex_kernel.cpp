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
 * \file axpy_ex_kernel.cpp
 * \brief Device-side Ascend C SIMD kernel for aclblasAxpyEx (arch22 / dav-2201).
 */

#include <cstdint>
#include <type_traits>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "axpy_ex_tiling_data.h"
#include "axpy_ex_kernel.h"

using namespace AscendC;

namespace {

__aicore__ inline void AxpyExWaitEvent(HardEvent evt, TPipe* pipe)
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
    }
}

class AxpyExKernel {
public:
    __aicore__ inline AxpyExKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR alpha, const AxpyExTilingData& tiling, TPipe* pipe)
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
        alphaVal_ = (tiling_.alphaIsDevice != 0U) ? *reinterpret_cast<__gm__ float*>(alpha) : tiling_.alpha;

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

        pipe_->InitBuffer(bufXIn_, AXPY_EX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufYIn_, AXPY_EX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufX32_, AXPY_EX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufY32_, AXPY_EX_ARCH22_TILE * sizeof(float));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incx != 1 || tiling_.incy != 1) {
            pipe_->InitBuffer(bufWin_, AXPY_EX_ARCH22_WIN_MAX * sizeof(float));
        }
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
    template <typename T>
    __aicore__ inline void ComputeTile(uint32_t count)
    {
        LocalTensor<T> xIn = bufXIn_.Get<T>();
        LocalTensor<T> yIn = bufYIn_.Get<T>();
        LocalTensor<float> x32 = bufX32_.Get<float>();
        if (std::is_same<T, float>::value) {
            LocalTensor<float> xF = bufXIn_.Get<float>();
            LocalTensor<float> yF = bufYIn_.Get<float>();
            Muls(x32, xF, alphaVal_, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Add(yF, yF, x32, count);
            PipeBarrier<PIPE_V>();
            return;
        }
        LocalTensor<float> y32 = bufY32_.Get<float>();
        Cast(x32, xIn, RoundMode::CAST_NONE, count);
        Cast(y32, yIn, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(x32, x32, alphaVal_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Add(y32, y32, x32, count);
        PipeBarrier<PIPE_V>();
        Cast(yIn, y32, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void LoadStridedTile(
        GlobalTensor<T>& gm, int64_t startOff, int64_t inc, uint32_t tileStart, uint32_t count,
        LocalTensor<T>& ubDst)
    {
        DataCopyPadExtParams<T> noPad{false, 0U, 0U, static_cast<T>(0.0F)};
        if (inc == 1) {
            DataCopyExtParams cp{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubDst, gm[static_cast<uint64_t>(startOff + tileStart)], cp, noPad);
            AxpyExWaitEvent(HardEvent::MTE2_V, pipe_);
            return;
        }
        const int64_t p0 = startOff + static_cast<int64_t>(tileStart) * inc;
        const int64_t p1 = startOff + static_cast<int64_t>(tileStart + count - 1U) * inc;
        const int64_t winLo = (p0 < p1) ? p0 : p1;
        const uint64_t span = static_cast<uint64_t>((p0 < p1) ? (p1 - p0) : (p0 - p1)) + 1ULL;
        if (span <= AXPY_EX_ARCH22_WIN_MAX) {
            LocalTensor<T> ubWin = bufWin_.Get<T>();
            DataCopyExtParams cpW{1U, static_cast<uint32_t>(span * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(ubWin, gm[static_cast<uint64_t>(winLo)], cpW, noPad);
            AxpyExWaitEvent(HardEvent::MTE2_S, pipe_);
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
                AxpyExWaitEvent(HardEvent::MTE2_S, pipe_);
                ubDst.SetValue(i, ubSc.GetValue(0));
            }
        }
        AxpyExWaitEvent(HardEvent::S_V, pipe_);
    }

    template <typename T>
    __aicore__ inline void StoreStridedTile(
        GlobalTensor<T>& gm, int64_t startOff, int64_t inc, uint32_t tileStart, uint32_t count,
        LocalTensor<T>& ubSrc)
    {
        if (inc == 1) {
            AxpyExWaitEvent(HardEvent::V_MTE3, pipe_);
            DataCopyExtParams cp{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(gm[static_cast<uint64_t>(startOff + tileStart)], ubSrc, cp);
            AxpyExWaitEvent(HardEvent::MTE3_MTE2, pipe_);
            return;
        }
        AxpyExWaitEvent(HardEvent::V_S, pipe_);
        LocalTensor<T> ubSc = bufScalar_.Get<T>();
        DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        for (uint32_t i = 0U; i < count; ++i) {
            const int64_t phys = startOff + static_cast<int64_t>(tileStart + i) * inc;
            ubSc.SetValue(0, ubSrc.GetValue(i));
            AxpyExWaitEvent(HardEvent::S_MTE3, pipe_);
            DataCopyPad(gm[static_cast<uint64_t>(phys)], ubSc, cp1);
            AxpyExWaitEvent(HardEvent::MTE3_S, pipe_);
        }
    }

    __aicore__ inline uint32_t ComputeTileStep() const
    {
        const uint64_t sx = static_cast<uint64_t>((tiling_.incx < 0) ? -tiling_.incx : tiling_.incx);
        const uint64_t sy = static_cast<uint64_t>((tiling_.incy < 0) ? -tiling_.incy : tiling_.incy);
        const uint64_t maxStride = (sx > sy) ? sx : sy;
        if (maxStride <= 1ULL) {
            return tiling_.tileSize;
        }
        const uint32_t winCap = static_cast<uint32_t>((AXPY_EX_ARCH22_WIN_MAX - 1ULL) / maxStride) + 1U;
        return (winCap < tiling_.tileSize) ? winCap : tiling_.tileSize;
    }

    template <typename T>
    __aicore__ inline void ProcessTyped()
    {
        GlobalTensor<T> xGm;
        GlobalTensor<T> yGm;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xAddr_));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(yAddr_));
        LocalTensor<T> xIn = bufXIn_.Get<T>();
        LocalTensor<T> yIn = bufYIn_.Get<T>();

        const bool contiguous = (tiling_.incx == 1) && (tiling_.incy == 1);
        const uint32_t tileStep = contiguous ? tiling_.tileSize : ComputeTileStep();
        DataCopyPadExtParams<T> noPad{false, 0U, 0U, static_cast<T>(0.0F)};
        for (uint32_t off = 0U; off < myCount_; off += tileStep) {
            const uint32_t cur = ((myCount_ - off) > tileStep) ? tileStep : (myCount_ - off);
            if (contiguous) {
                DataCopyExtParams cp{1U, static_cast<uint32_t>(cur * sizeof(T)), 0U, 0U, 0U};
                DataCopyPad(xIn, xGm[myOffset_ + off], cp, noPad);
                DataCopyPad(yIn, yGm[myOffset_ + off], cp, noPad);
                AxpyExWaitEvent(HardEvent::MTE2_V, pipe_);
                ComputeTile<T>(cur);
                AxpyExWaitEvent(HardEvent::V_MTE3, pipe_);
                DataCopyPad(yGm[myOffset_ + off], yIn, cp);
                AxpyExWaitEvent(HardEvent::MTE3_MTE2, pipe_);
            } else {
                LoadStridedTile<T>(xGm, tiling_.startOffsetX, tiling_.incx, myOffset_ + off, cur, xIn);
                LoadStridedTile<T>(yGm, tiling_.startOffsetY, tiling_.incy, myOffset_ + off, cur, yIn);
                ComputeTile<T>(cur);
                StoreStridedTile<T>(yGm, tiling_.startOffsetY, tiling_.incy, myOffset_ + off, cur, yIn);
            }
        }
    }

    TPipe* pipe_{nullptr};
    GM_ADDR xAddr_{nullptr};
    GM_ADDR yAddr_{nullptr};
    TBuf<TPosition::VECCALC> bufXIn_;
    TBuf<TPosition::VECCALC> bufYIn_;
    TBuf<TPosition::VECCALC> bufX32_;
    TBuf<TPosition::VECCALC> bufY32_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
    AxpyExTilingData tiling_{};
    float alphaVal_{0.0f};
    uint32_t blockIdx_{0U};
    uint32_t myOffset_{0U};
    uint32_t myCount_{0U};
    bool active_{false};
};

} // namespace

extern "C" __global__ __aicore__ void axpy_ex_arch22_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR alpha, const AxpyExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AxpyExKernel op;
    op.Init(x, y, alpha, tiling, &pipe);
    op.Process();
}

void axpy_ex_kernel_do(
    uint8_t* x, uint8_t* y, void* alpha, const AxpyExTilingData& tiling, uint32_t numBlocks, void* stream)
{
    axpy_ex_arch22_kernel<<<numBlocks, nullptr, stream>>>(x, y, static_cast<uint8_t*>(alpha), tiling);
}
