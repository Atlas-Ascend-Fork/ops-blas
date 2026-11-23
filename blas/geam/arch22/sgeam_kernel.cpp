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
 * \file sgeam_kernel.cpp
 * \brief Device-side Ascend C SIMD kernel for aclblasSgeam (arch22 / dav-2201).
 */

#include <cstdint>
#include "kernel_operator.h"
#include "sgeam_tiling_data.h"
#include "sgeam_kernel.h"
#include "geam_kernel_common.h"

using namespace AscendC;

namespace {

__aicore__ inline void SgeamWaitEvent(HardEvent evt, TPipe* pipe)
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
    } else if (evt == HardEvent::S_MTE2) {
        SetFlag<HardEvent::S_MTE2>(eid);
        WaitFlag<HardEvent::S_MTE2>(eid);
    }
}

class SgeamKernel {
public:
    __aicore__ inline SgeamKernel() = default;

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const SgeamTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        geam_arch22::InitGeamKernelContext(ctx_, a, b, c, tiling_, pipe, 1U, SGEAM_ARCH22_TILE, SGEAM_ARCH22_WIN_MAX);
    }

    __aicore__ inline void Process()
    {
        if (!ctx_.active) {
            return;
        }
        for (uint32_t col = ctx_.colStart; col < ctx_.colEnd; ++col) {
            for (uint32_t rowStart = 0U; rowStart < tiling_.m; rowStart += SGEAM_ARCH22_TILE) {
                const uint32_t rem = tiling_.m - rowStart;
                const uint32_t curM = (rem < SGEAM_ARCH22_TILE) ? rem : SGEAM_ARCH22_TILE;
                ProcessTile(col, rowStart, curM);
            }
        }
    }

private:
    __aicore__ inline void LoadMatrixColTile(
        GlobalTensor<float>& gm, aclblasOperation_t op, uint32_t ld, uint32_t col, uint32_t rowStart, uint32_t curM,
        LocalTensor<float>& ubDst)
    {
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        if (op == ACLBLAS_OP_N) {
            const uint64_t off = static_cast<uint64_t>(col) * ld + rowStart;
            DataCopyExtParams cp{1U, static_cast<uint32_t>(curM * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(ubDst, gm[off], cp, noPad);
            SgeamWaitEvent(HardEvent::MTE2_V, ctx_.pipe);
            return;
        }
        const uint64_t baseOff = static_cast<uint64_t>(col) + static_cast<uint64_t>(rowStart) * ld;
        const uint64_t span = 1ULL + static_cast<uint64_t>(curM - 1U) * ld;
        if (span <= SGEAM_ARCH22_WIN_MAX) {
            LocalTensor<float> ubWin = ctx_.bufWin.Get<float>();
            DataCopyExtParams cpW{1U, static_cast<uint32_t>(span * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(ubWin, gm[baseOff], cpW, noPad);
            SgeamWaitEvent(HardEvent::MTE2_S, ctx_.pipe);
            for (uint32_t i = 0U; i < curM; ++i) {
                ubDst.SetValue(i, ubWin.GetValue(i * ld));
            }
            SgeamWaitEvent(HardEvent::S_MTE2, ctx_.pipe);
        } else {
            LocalTensor<float> ubSc = ctx_.bufScalar.Get<float>();
            DataCopyExtParams cp1{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            for (uint32_t i = 0U; i < curM; ++i) {
                DataCopyPad(ubSc, gm[baseOff + static_cast<uint64_t>(i) * ld], cp1, noPad);
                SgeamWaitEvent(HardEvent::MTE2_S, ctx_.pipe);
                ubDst.SetValue(i, ubSc.GetValue(0));
                SgeamWaitEvent(HardEvent::S_MTE2, ctx_.pipe);
            }
        }
        SgeamWaitEvent(HardEvent::S_V, ctx_.pipe);
    }

    __aicore__ inline void ProcessTile(uint32_t col, uint32_t rowStart, uint32_t curM)
    {
        LocalTensor<float> ubA = ctx_.bufA.Get<float>();
        LocalTensor<float> ubB = ctx_.bufB.Get<float>();
        LocalTensor<float> ubC = ctx_.bufC.Get<float>();

        if (tiling_.alphaIsZero == 0U) {
            LoadMatrixColTile(ctx_.aGm, tiling_.opA, tiling_.lda, col, rowStart, curM, ubA);
        }
        if (tiling_.betaIsZero == 0U) {
            LoadMatrixColTile(ctx_.bGm, tiling_.opB, tiling_.ldb, col, rowStart, curM, ubB);
        }
        if (tiling_.alphaIsZero != 0U && tiling_.betaIsZero != 0U) {
            Duplicate<float>(ubC, 0.0F, static_cast<int32_t>(curM));
        } else if (tiling_.alphaIsZero != 0U) {
            Muls<float>(ubC, ubB, tiling_.beta, static_cast<int32_t>(curM));
        } else if (tiling_.betaIsZero != 0U) {
            Muls<float>(ubC, ubA, tiling_.alpha, static_cast<int32_t>(curM));
        } else {
            Muls<float>(ubC, ubA, tiling_.alpha, static_cast<int32_t>(curM));
            Muls<float>(ubB, ubB, tiling_.beta, static_cast<int32_t>(curM));
            PipeBarrier<PIPE_V>();
            Add<float>(ubC, ubC, ubB, static_cast<int32_t>(curM));
        }
        SgeamWaitEvent(HardEvent::V_MTE3, ctx_.pipe);
        const uint64_t outOff = static_cast<uint64_t>(col) * tiling_.ldc + rowStart;
        DataCopyExtParams cpOut{1U, static_cast<uint32_t>(curM * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(ctx_.cGm[outOff], ubC, cpOut);
        SgeamWaitEvent(HardEvent::MTE3_MTE2, ctx_.pipe);
    }

    SgeamTilingData tiling_{};
    geam_arch22::GeamKernelContext ctx_{};
};

} // namespace

extern "C" __global__ __aicore__ void sgeam_kernel_impl(GM_ADDR a, GM_ADDR b, GM_ADDR c, SgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SgeamKernel op;
    op.Init(a, b, c, tiling, &pipe);
    op.Process();
}

void sgeam_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, const SgeamTilingData& tiling, uint32_t numBlocks, void* stream)
{
    sgeam_kernel_impl<<<numBlocks, nullptr, stream>>>(a, b, c, tiling);
}
