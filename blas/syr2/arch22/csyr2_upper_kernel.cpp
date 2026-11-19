/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "csyr2_kernel.h"
#include "csyr2_vector_common.h"
#include "csyr2_banded_common.h"
#include "common/helper/kernel_utils.h"
#include "csyr2_atomic_common.h"
using namespace AscendC;

// Tiny matrices: one column per core removes the four-column serial loop.
// Fixed buffers cover n <= 16; only the referenced prefix is transferred.
__global__ __aicore__ __vector__ void csyr2_small_upper(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    Csyr2Vector::Vectors v;
    v.InitGlobals(xGm, yGm, aGm);
    TBuf<TPosition::VECCALC> ab;
    TBuf<TPosition::VECCALC> ib;
    v.Allocate(pipe, 256U);
    pipe.InitBuffer(ab, 256U);
    pipe.InitBuffer(ib, 256U);
    v.Bind();
    auto av = ab.Get<float>();
    auto index = ib.Get<int32_t>();
    for (uint32_t col = GetBlockIdx(); col < t.n; col += GetBlockNum()) {
        const uint32_t count = 2U * (col + 1U);
        Csyr2Vector::InitRotationIndex(index, count);
        DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0F};
        DataCopyPad(v.xv, v.x, copy, pad);
        DataCopyPad(v.yv, v.y, copy, pad);
        DataCopyPad(av, v.a[2ULL * col * t.lda], copy, pad);
        PipeBarrier<PIPE_ALL>();
        Csyr2Vector::Rotate(v, index.ReinterpretCast<uint32_t>(), count);
        PipeBarrier<PIPE_V>();
        const float xre = v.xv.GetValue(2U * col);
        const float xim = v.xv.GetValue(2U * col + 1U);
        const float yre = v.yv.GetValue(2U * col);
        const float yim = v.yv.GetValue(2U * col + 1U);
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0U, count);
        Csyr2Vector::Update(av, v.xv, v.xr, v.yv, v.yr, xre, xim, yre, yim);
        SetMaskNorm();
        ResetMask();
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(v.a[2ULL * col * t.lda], av, copy);
        PipeBarrier<PIPE_ALL>();
    }
}

__global__ __aicore__ __vector__ void csyr2_tiny_scalar(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> x;
    GlobalTensor<float> y;
    GlobalTensor<float> a;
    x.SetGlobalBuffer((__gm__ float*)xGm);
    y.SetGlobalBuffer((__gm__ float*)yGm);
    a.SetGlobalBuffer((__gm__ float*)aGm);
    for (uint32_t c = 0; c < t.n; ++c) {
        const float xcR = x.GetValue(2U * c);
        const float xcI = x.GetValue(2U * c + 1U);
        const float ycR = y.GetValue(2U * c);
        const float ycI = y.GetValue(2U * c + 1U);
        for (uint32_t r = 0; r <= c; ++r) {
            const float xr = x.GetValue(2U * r);
            const float xi = x.GetValue(2U * r + 1U);
            const float yr = y.GetValue(2U * r);
            const float yi = y.GetValue(2U * r + 1U);
            const uint64_t pos = 2ULL * (uint64_t(c) * t.lda + r);
            a.SetValue(pos, a.GetValue(pos) + ((xcR * yr - xcI * yi) + (ycR * xr - ycI * xi)));
            a.SetValue(pos + 1U, a.GetValue(pos + 1U) + ((xcR * yi + xcI * yr) + (ycR * xi + ycI * xr)));
        }
    }
}

void csyr2_small_upper_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    if (t.n <= 4U) {
        csyr2_tiny_scalar<<<1U, nullptr, stream>>>(x, y, a, t);
    } else {
        csyr2_small_upper<<<blocks, nullptr, stream>>>(x, y, a, t);
    }
}

template <uint32_t Low, uint32_t High, uint32_t Cut>
__global__ __aicore__ __vector__ void csyr2_banded_vector_manual(
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto range = Csyr2Banded::Range<Low, High, Cut>(t);
    if (range.first >= range.last) {
        return;
    }
    TPipe pipe;
    Csyr2Banded::Bands bands;
    bands.Init<true>(pipe, xGm, yGm, aGm, t, range);
    bands.Run<true>();
}
// Selected from the 16-device search. The larger late-column cost gives
// the long-column region more cores. The manual variant signals the
// MTE2->S semaphore before the first A band prefetch (UpperIn) so the
// rotation-index loads settle ahead of the scalar-dependent steps, and it
// runs without the RequiresPrecise probe used by the plain path.
void csyr2_banded_vector_manual_do(
    uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    csyr2_banded_vector_manual<512U, 960U, 1536U><<<blocks, nullptr, stream>>>(x, y, a, t);
}

template <uint32_t Low, uint32_t High, uint32_t Cut>
__global__ __aicore__ __vector__ void csyr2_banded_vector(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto range = Csyr2Banded::Range<Low, High, Cut>(t);
    if (range.first >= range.last) {
        return;
    }
    TPipe pipe;
    Csyr2Banded::Bands bands;
    bands.Init<false>(pipe, xGm, yGm, aGm, t, range);
    if (Csyr2Vector::RequiresPrecise(bands.storage.v, t.n, range.last)) {
        bands.Precise();
        return;
    }
    bands.Run<false>();
}
// Routes shapes above the banded threshold to the manual variant; smaller
// shapes run the plain kernel, which probes RequiresPrecise before computing.
void csyr2_banded_vector_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    if (t.n > 32U) {
        csyr2_banded_vector_manual_do(blocks, stream, x, y, a, t);
    } else {
        csyr2_banded_vector<512U, 960U, 1536U><<<blocks, nullptr, stream>>>(x, y, a, t);
    }
}

// Compute the rank-two increment in UB, then perform A += delta in GM.
// Each matrix element has one owner; no cross-core floating-point reduction.
template <uint32_t Batch>
__global__ __aicore__ __vector__ void csyr2_upper_atomic_manual(
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t, uint32_t cut1, uint32_t cut2)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csyr2Atomic::Run<true, Batch>(xGm, yGm, aGm, t, cut1, cut2);
}

static uint32_t UpperAtomicBoundary(uint32_t part, uint32_t parts, uint32_t n)
{
    if (parts == 0U) {
        return 0U;
    }
    const uint64_t target = (uint64_t(n) * (n + 1U) / 2U + 512ULL * n) * part / parts;
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2U;
        if (uint64_t(mid) * (mid + 1U) / 2U + 512ULL * mid < target) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}
void csyr2_upper_atomic_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    const uint32_t cores = ((blocks > 48U ? 48U : blocks) / 16U) * 16U;
    const uint32_t cut1 = UpperAtomicBoundary(1U, cores / 16U, t.n);
    const uint32_t cut2 = UpperAtomicBoundary(2U, cores / 16U, t.n);
    if (t.n <= 2048U) {
        csyr2_upper_atomic_manual<4U><<<cores, nullptr, stream>>>(x, y, a, t, cut1, cut2);
    } else {
        csyr2_upper_atomic_manual<2U><<<cores, nullptr, stream>>>(x, y, a, t, cut1, cut2);
    }
}
