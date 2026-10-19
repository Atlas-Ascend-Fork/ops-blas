/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "ctbmv_kernel_common.h"

#if __DAV_C220_VEC__
namespace Ctbmv {

class CtbmvPanelProducts {
protected:
    __aicore__ inline void MultiplyProducts(uint32_t count, uint32_t trans)
    {
        Mul(pr, ar, xr, count);
        Mul(pi, ar, xi, count);
        Mul(cross, ai, xi, count);
        PipeBarrier<PIPE_V>();
        if (trans == ACLBLAS_OP_C) {
            Add(pr, pr, cross, count);
        } else {
            Sub(pr, pr, cross, count);
        }
        PipeBarrier<PIPE_V>();
        Mul(cross, ai, xr, count);
        PipeBarrier<PIPE_V>();
        if (trans == ACLBLAS_OP_C) {
            Sub(pi, pi, cross, count);
        } else {
            Add(pi, pi, cross, count);
        }
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> ar;
    LocalTensor<float> ai;
    LocalTensor<float> xr;
    LocalTensor<float> xi;
    LocalTensor<float> pr;
    LocalTensor<float> pi;
    LocalTensor<float> cross;
};

class CtbmvTinyVectorAiv : public CtbmvPanelProducts {
public:
    __aicore__ inline bool Process(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        InitInput(a, x, workspace, tiling);
        InitBuffers();
        LoadData();
        if (!CheckFinite()) {
            return false;
        }
        ComputeOutputs(tiling.trans);
        return true;
    }

private:
    GlobalTensor<float> aGm;
    GlobalTensor<float> xGm;
    GlobalTensor<uint32_t> offsetsGm;

    __aicore__ inline void ComputeOutputs(uint32_t trans)
    {
        aSlab.SetValue(aCount + 2U, 1.0F);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        SetMaskNorm();
        ResetMask();
        const uint32_t count = n * 8U;
        Gather(ar, aSlab, offsets, 0U, count);
        Gather(ai, aSlab, offsets[128U], 0U, count);
        Gather(xr, xSlab, offsets[256U], 0U, count);
        Gather(xi, xSlab, offsets[384U], 0U, count);
        PipeBarrier<PIPE_V>();
        MultiplyProducts(count, trans);
        WholeReduceSum<float, true>(yr, pr, 8U, n, 1U, 1U, 1U);
        WholeReduceSum<float, true>(yi, pi, 8U, n, 1U, 1U, 1U);
        PipeBarrier<PIPE_V>();
        Gather(result, yr, offsets[512U], 0U, n * 2U);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(xGm, result, 0, 1, n * 8U, 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool CheckFinite()
    {
        Abs(magnitude, aSlab, 256U);
        PipeBarrier<PIPE_V>();
        CompareScalar(maskA, magnitude, 1.0e10F, CMPMODE::LE, 256U);
        PipeBarrier<PIPE_V>();
        Abs(magnitude, xSlab, 64U);
        PipeBarrier<PIPE_V>();
        CompareScalar(maskX, magnitude, 1.0e10F, CMPMODE::LE, 64U);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto aBits = maskA.ReinterpretCast<uint64_t>();
        auto xBits = maskX.ReinterpretCast<uint64_t>();
        for (uint32_t word = 0U; word < 4U; ++word) {
            if (aBits.GetValue(word) != ~0ULL) {
                return false;
            }
        }
        if (xBits.GetValue(0U) != ~0ULL) {
            return false;
        }
        return true;
    }

    __aicore__ inline void LoadData()
    {
        Duplicate(aSlab, 0.0F, 256U);
        Duplicate(xSlab, 0.0F, 64U);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams acopy{1U, aCount * 4U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> apad{true, 0U, static_cast<uint8_t>((8U - aCount % 8U) % 8U), 0.0F};
        DataCopyPad(aSlab, aGm, acopy, apad);
        DataCopyExtParams xcopy{1U, n * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> xpad{true, 0U, static_cast<uint8_t>((8U - (n * 2U) % 8U) % 8U), 0.0F};
        DataCopyPad(xSlab, xGm, xcopy, xpad);
        gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(offsets, offsetsGm, 0, 1, 544U * 4U, 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void InitInput(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        n = tiling.n;
        aCount = n * tiling.lda * 2U;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), aCount);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), n * 2U);
        offsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace), 544U);
    }

    uint32_t n;
    uint32_t aCount;

    __aicore__ inline void InitBuffers()
    {
        aSlab = LocalTensor<float>(TPosition::VECCALC, 0U, 256U);
        xSlab = LocalTensor<float>(TPosition::VECCALC, 1024U, 64U);
        magnitude = LocalTensor<float>(TPosition::VECCALC, 1280U, 256U);
        maskA = LocalTensor<uint8_t>(TPosition::VECCALC, 2304U, 32U);
        maskX = LocalTensor<uint8_t>(TPosition::VECCALC, 2336U, 32U);
        ar = LocalTensor<float>(TPosition::VECCALC, 2368U, 128U);
        ai = LocalTensor<float>(TPosition::VECCALC, 2880U, 128U);
        xr = LocalTensor<float>(TPosition::VECCALC, 3392U, 128U);
        xi = LocalTensor<float>(TPosition::VECCALC, 3904U, 128U);
        pr = LocalTensor<float>(TPosition::VECCALC, 4416U, 128U);
        pi = LocalTensor<float>(TPosition::VECCALC, 4928U, 128U);
        cross = LocalTensor<float>(TPosition::VECCALC, 5440U, 128U);
        yr = LocalTensor<float>(TPosition::VECCALC, 5952U, 16U);
        yi = LocalTensor<float>(TPosition::VECCALC, 6016U, 16U);
        result = LocalTensor<float>(TPosition::VECCALC, 6080U, 32U);
        offsets = LocalTensor<uint32_t>(TPosition::VECCALC, 6208U, 544U);
    }

    LocalTensor<float> aSlab;
    LocalTensor<float> xSlab;
    LocalTensor<float> magnitude;
    LocalTensor<uint8_t> maskA;
    LocalTensor<uint8_t> maskX;
    LocalTensor<float> yr;
    LocalTensor<float> yi;
    LocalTensor<float> result;
    LocalTensor<uint32_t> offsets;
};

class CtbmvSmallPanelAiv : public CtbmvPanelProducts {
public:
    __aicore__ inline void Process(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& t)
    {
        InitInput(a, x, workspace, t);
        InitBuffers();
        LoadData();
        for (uint32_t first = core * 8U; first < n; first += t.useCoreNum * 8U) {
            ProcessBatch(first, t.trans);
        }
    }

private:
    GlobalTensor<float> aGm;
    GlobalTensor<float> xGm;
    GlobalTensor<float> outGm;
    GlobalTensor<uint32_t> offsetsGm;

    __aicore__ inline void ProcessBatch(uint32_t first, uint32_t trans)
    {
        const uint32_t batch = Min(8U, n - first);
        const uint32_t count = batch * width;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        for (uint32_t field = 0U; field < 4U; ++field) {
            gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
                offsets[field * 512U], offsetsGm[field * entries + first * width], 0, 1, count * 4U, 0, 0, 0, 0);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Gather(ar, aSlab, offsets, 0U, count);
        Gather(ai, aSlab, offsets[512U], 0U, count);
        Gather(xr, xSlab, offsets[1024U], 0U, count);
        Gather(xi, xSlab, offsets[1536U], 0U, count);
        PipeBarrier<PIPE_V>();
        MultiplyProducts(count, trans);
        WholeReduceSum<float, true>(yr, pr, width, batch, 1U, 1U, width / 8U);
        WholeReduceSum<float, true>(yi, pi, width, batch, 1U, 1U, width / 8U);
        PipeBarrier<PIPE_V>();
        Gather(result, yr, offsets[2048U], 0U, batch * 2U);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(outGm[first * 2U], result, 0, 1, batch * 8U, 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadData()
    {
        Duplicate(aSlab, 0.0F, (aCount + 15U) & ~7U);
        Duplicate(xSlab, 0.0F, (n * 2U + 15U) & ~7U);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams acopy{1U, aCount * 4U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> apad{true, 0U, static_cast<uint8_t>((8U - aCount % 8U) % 8U), 0.0F};
        DataCopyPad(aSlab, aGm, acopy, apad);
        DataCopyExtParams xcopy{1U, n * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> xpad{true, 0U, static_cast<uint8_t>((8U - n * 2U % 8U) % 8U), 0.0F};
        DataCopyPad(xSlab, xGm, xcopy, xpad);
        gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(offsets[2048U], offsetsGm[entries * 4U], 0, 1, 64U, 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        aSlab.SetValue(aCount + 2U, 1.0F);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void InitInput(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& t)
    {
        core = GetBlockIdx();
        n = t.n;
        width = t.k < 8U ? 8U : (t.k < 16U ? 16U : (t.k < 32U ? 32U : 64U));
        entries = ((n + 7U) & ~7U) * width;
        aCount = n * t.lda * 2U;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), aCount);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), n * 2U);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), n * 2U);
        offsetsGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint32_t*>(workspace + ((n * 8U + 31U) & ~31U)), entries * 4U + 16U);
    }

    uint32_t core;
    uint32_t n;
    uint32_t width;
    uint32_t entries;
    uint32_t aCount;

    __aicore__ inline void InitBuffers()
    {
        aSlab = LocalTensor<float>(TPosition::VECCALC, 0U, 16392U);
        xSlab = LocalTensor<float>(TPosition::VECCALC, 65568U, 264U);
        offsets = LocalTensor<uint32_t>(TPosition::VECCALC, 66624U, 2064U);
        ar = LocalTensor<float>(TPosition::VECCALC, 74880U, 512U);
        ai = LocalTensor<float>(TPosition::VECCALC, 76928U, 512U);
        xr = LocalTensor<float>(TPosition::VECCALC, 78976U, 512U);
        xi = LocalTensor<float>(TPosition::VECCALC, 81024U, 512U);
        pr = LocalTensor<float>(TPosition::VECCALC, 83072U, 512U);
        pi = LocalTensor<float>(TPosition::VECCALC, 85120U, 512U);
        cross = LocalTensor<float>(TPosition::VECCALC, 87168U, 512U);
        yr = LocalTensor<float>(TPosition::VECCALC, 89216U, 8U);
        yi = LocalTensor<float>(TPosition::VECCALC, 89248U, 8U);
        result = LocalTensor<float>(TPosition::VECCALC, 89280U, 16U);
    }

    LocalTensor<float> aSlab;
    LocalTensor<float> xSlab;
    LocalTensor<uint32_t> offsets;
    LocalTensor<float> yr;
    LocalTensor<float> yi;
    LocalTensor<float> result;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
};

} // namespace Ctbmv
#endif
