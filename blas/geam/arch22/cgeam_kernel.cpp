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
 * \file cgeam_kernel.cpp
 * \brief Device-side Ascend C kernel for aclblasCgeam (arch22 / dav-2201).
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cgeam_tiling_data.h"
#include "cgeam_kernel.h"
#include "geam_kernel_common.h"

using namespace AscendC;

namespace {

__aicore__ inline void CgeamWaitEvent(HardEvent evt, TPipe* pipe)
{
    const event_t eid = static_cast<event_t>(pipe->FetchEventID(evt));
    if (evt == HardEvent::MTE2_S) {
        SetFlag<HardEvent::MTE2_S>(eid);
        WaitFlag<HardEvent::MTE2_S>(eid);
    } else if (evt == HardEvent::S_MTE3) {
        SetFlag<HardEvent::S_MTE3>(eid);
        WaitFlag<HardEvent::S_MTE3>(eid);
    } else if (evt == HardEvent::MTE3_MTE2) {
        SetFlag<HardEvent::MTE3_MTE2>(eid);
        WaitFlag<HardEvent::MTE3_MTE2>(eid);
    } else if (evt == HardEvent::S_MTE2) {
        SetFlag<HardEvent::S_MTE2>(eid);
        WaitFlag<HardEvent::S_MTE2>(eid);
    }
}

class CgeamKernel {
public:
    __aicore__ inline CgeamKernel() = default;

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgeamTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        geam_arch22::InitGeamKernelContext(ctx_, a, b, c, tiling_, pipe, 2U, CGEAM_ARCH22_TILE, CGEAM_ARCH22_WIN_MAX);
    }

    __aicore__ inline void Process()
    {
        if (!ctx_.active) {
            return;
        }
        for (uint32_t col = ctx_.colStart; col < ctx_.colEnd; ++col) {
            for (uint32_t rowStart = 0U; rowStart < tiling_.m; rowStart += CGEAM_ARCH22_TILE) {
                const uint32_t rem = tiling_.m - rowStart;
                const uint32_t curM = (rem < CGEAM_ARCH22_TILE) ? rem : CGEAM_ARCH22_TILE;
                ProcessTile(col, rowStart, curM);
            }
        }
    }

private:
    __aicore__ inline void LoadComplexColTile(
        GlobalTensor<float>& gm, aclblasOperation_t op, uint32_t ld, uint32_t col, uint32_t rowStart, uint32_t curM,
        LocalTensor<float>& ubDst)
    {
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        if (op == ACLBLAS_OP_N) {
            const uint64_t off = (static_cast<uint64_t>(col) * ld + rowStart) * 2ULL;
            DataCopyExtParams cp{1U, static_cast<uint32_t>(curM * 2U * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(ubDst, gm[off], cp, noPad);
            CgeamWaitEvent(HardEvent::MTE2_S, ctx_.pipe);
            return;
        }
        const uint64_t baseOff = (static_cast<uint64_t>(col) + static_cast<uint64_t>(rowStart) * ld) * 2ULL;
        const uint64_t span = 1ULL + static_cast<uint64_t>(curM - 1U) * ld;
        if (span <= CGEAM_ARCH22_WIN_MAX) {
            LocalTensor<float> ubWin = ctx_.bufWin.Get<float>();
            DataCopyExtParams cpW{1U, static_cast<uint32_t>(span * 2U * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(ubWin, gm[baseOff], cpW, noPad);
            CgeamWaitEvent(HardEvent::MTE2_S, ctx_.pipe);
            for (uint32_t i = 0U; i < curM; ++i) {
                const uint32_t srcIdx = i * ld * 2U;
                ubDst.SetValue(i * 2U, ubWin.GetValue(srcIdx));
                ubDst.SetValue(i * 2U + 1U, ubWin.GetValue(srcIdx + 1U));
            }
            CgeamWaitEvent(HardEvent::S_MTE2, ctx_.pipe);
        } else {
            LocalTensor<float> ubSc = ctx_.bufScalar.Get<float>();
            DataCopyExtParams cp1{1U, static_cast<uint32_t>(2U * sizeof(float)), 0U, 0U, 0U};
            for (uint32_t i = 0U; i < curM; ++i) {
                DataCopyPad(ubSc, gm[baseOff + static_cast<uint64_t>(i) * ld * 2ULL], cp1, noPad);
                CgeamWaitEvent(HardEvent::MTE2_S, ctx_.pipe);
                ubDst.SetValue(i * 2U, ubSc.GetValue(0));
                ubDst.SetValue(i * 2U + 1U, ubSc.GetValue(1));
                CgeamWaitEvent(HardEvent::S_MTE2, ctx_.pipe);
            }
        }
    }

    __aicore__ inline void ProcessTile(uint32_t col, uint32_t rowStart, uint32_t curM)
    {
        LocalTensor<float> ubA = ctx_.bufA.Get<float>();
        LocalTensor<float> ubB = ctx_.bufB.Get<float>();
        LocalTensor<float> ubC = ctx_.bufC.Get<float>();
        if (tiling_.alphaIsZero == 0U) {
            LoadComplexColTile(ctx_.aGm, tiling_.opA, tiling_.lda, col, rowStart, curM, ubA);
        }
        if (tiling_.betaIsZero == 0U) {
            LoadComplexColTile(ctx_.bGm, tiling_.opB, tiling_.ldb, col, rowStart, curM, ubB);
        }
        const float conjA = (tiling_.opA == ACLBLAS_OP_C) ? -1.0F : 1.0F;
        const float conjB = (tiling_.opB == ACLBLAS_OP_C) ? -1.0F : 1.0F;
        for (uint32_t i = 0U; i < curM; ++i) {
            float cr = 0.0F;
            float ci = 0.0F;
            if (tiling_.alphaIsZero == 0U) {
                const float ar = ubA.GetValue(i * 2U);
                const float ai = conjA * ubA.GetValue(i * 2U + 1U);
                cr += tiling_.alphaR * ar - tiling_.alphaI * ai;
                ci += tiling_.alphaR * ai + tiling_.alphaI * ar;
            }
            if (tiling_.betaIsZero == 0U) {
                const float br = ubB.GetValue(i * 2U);
                const float bi = conjB * ubB.GetValue(i * 2U + 1U);
                cr += tiling_.betaR * br - tiling_.betaI * bi;
                ci += tiling_.betaR * bi + tiling_.betaI * br;
            }
            ubC.SetValue(i * 2U, cr);
            ubC.SetValue(i * 2U + 1U, ci);
        }
        CgeamWaitEvent(HardEvent::S_MTE3, ctx_.pipe);
        const uint64_t outOff = (static_cast<uint64_t>(col) * tiling_.ldc + rowStart) * 2ULL;
        DataCopyExtParams cpOut{1U, static_cast<uint32_t>(curM * 2U * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(ctx_.cGm[outOff], ubC, cpOut);
        CgeamWaitEvent(HardEvent::MTE3_MTE2, ctx_.pipe);
    }

    CgeamTilingData tiling_{};
    geam_arch22::GeamKernelContext ctx_{};
};

} // namespace

extern "C" __global__ __aicore__ void cgeam_kernel_impl(GM_ADDR a, GM_ADDR b, GM_ADDR c, CgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgeamKernel op;
    op.Init(a, b, c, tiling, &pipe);
    op.Process();
}

void cgeam_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, const CgeamTilingData& tiling, uint32_t numBlocks, void* stream)
{
    cgeam_kernel_impl<<<numBlocks, nullptr, stream>>>(a, b, c, tiling);
}
