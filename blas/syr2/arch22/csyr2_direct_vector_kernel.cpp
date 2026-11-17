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
#include "common/helper/kernel_utils.h"
#include "csyr2_kernel.h"
#include "csyr2_precise_scalar.h"
#include "csyr2_vector_common.h"
using namespace AscendC;
namespace {
template <uint32_t kRows>
__aicore__ inline uint64_t Prefix(uint32_t c)
{
    const uint64_t q = c / kRows;
    const uint64_t rem = c % kRows;
    const uint64_t transfers = kRows * q * (q + 1ULL) / 2ULL + rem * (q + 1ULL);
    return uint64_t(c) * (c + 1ULL) / 2ULL + 1024ULL * c + 512ULL * transfers;
}
template <uint32_t kRows>
__aicore__ inline uint32_t Boundary(uint32_t part, uint32_t parts, uint32_t n)
{
    if (!part || !parts) {
        return 0;
    }
    if (part == parts) {
        return n;
    }
    const uint64_t target = Prefix<kRows>(n) * part / parts;
    uint32_t lo = 0;
    uint32_t hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2U;
        if (Prefix<kRows>(mid) < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}
} // namespace
template <uint32_t kRows>
struct DirectWindow {
    Csyr2Vector::Vectors v;
    TBuf<TPosition::VECCALC> ab0;
    TBuf<TPosition::VECCALC> ab1;
    TBuf<TPosition::VECCALC> indexb;
    LocalTensor<float> a0;
    LocalTensor<float> a1;
    LocalTensor<int32_t> index;
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, uint32_t n, uint32_t last)
    {
        v.InitGlobals(xGm, yGm, aGm);
        const uint32_t vectorBytes = (2U * n * sizeof(float) + 31U) & ~31U;
        v.Allocate(pipe, vectorBytes, 2U * kRows * sizeof(float));
        pipe.InitBuffer(ab0, 2U * kRows * sizeof(float));
        pipe.InitBuffer(ab1, 2U * kRows * sizeof(float));
        pipe.InitBuffer(indexb, 2U * kRows * sizeof(int32_t));
        v.Bind();
        a0 = ab0.Get<float>();
        a1 = ab1.Get<float>();
        index = indexb.Get<int32_t>();
        gm_to_ub_align<ArchType::ASCEND_V220, float>(v.xv, v.x, 0, 1, 2U * last * sizeof(float), 0, 0, 0, 0);
        gm_to_ub_align<ArchType::ASCEND_V220, float>(v.yv, v.y, 0, 1, 2U * last * sizeof(float), 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline bool HasNonfinite(uint32_t last)
    {
        for (uint32_t offset = 0; offset < 2U * last; offset += 2U * kRows) {
            const uint32_t count = 2U * last - offset < 2U * kRows ? 2U * last - offset : 2U * kRows;
            Add(a0, v.xv[offset], v.yv[offset], count);
            PipeBarrier<PIPE_V>();
            ReduceSum(a1, a0, index.ReinterpretCast<float>(), count);
            PipeBarrier<PIPE_ALL>();
            const uint32_t bits = a1.ReinterpretCast<uint32_t>().GetValue(0);
            if ((bits & 0x7f800000U) == 0x7f800000U) {
                return true;
            }
        }
        return false;
    }
    __aicore__ inline void Precise(uint32_t first, uint32_t last, const Csyr2TilingData& t)
    {
        // Preserve the full complex expression, including 0*Inf, and group
        // each complex product before combining the two rank-one terms.
        for (uint32_t col = first; col < last; ++col) {
            const float xcR = v.xv.GetValue(2U * col);
            const float xcI = v.xv.GetValue(2U * col + 1U);
            const float ycR = v.yv.GetValue(2U * col);
            const float ycI = v.yv.GetValue(2U * col + 1U);
            for (uint32_t row = 0; row <= col; row += kRows) {
                const uint32_t nr = col + 1U - row < kRows ? col + 1U - row : kRows;
                const uint64_t offset = (uint64_t(col) * t.lda + row) * 2ULL;
                gm_to_ub_align<ArchType::ASCEND_V220, float>(a0, v.a[offset], 0, 1, nr * 8U, 0, 0, 0, 0);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t i = 0; i < nr; ++i) {
                    const uint32_t r = 2U * (row + i);
                    const float xR = v.xv.GetValue(r);
                    const float xI = v.xv.GetValue(r + 1U);
                    const float yR = v.yv.GetValue(r);
                    const float yI = v.yv.GetValue(r + 1U);
                    float ar = a0.GetValue(2U * i);
                    float ai = a0.GetValue(2U * i + 1U);
                    Csyr2Precise::Update(ar, ai, t.ar, t.ai, xcR, xcI, ycR, ycI, xR, xI, yR, yI);
                    a0.SetValue(2U * i, ar);
                    a0.SetValue(2U * i + 1U, ai);
                }
                PipeBarrier<PIPE_ALL>();
                ub_to_gm_align<ArchType::ASCEND_V220, float>(v.a[offset], a0, 0, 1, nr * 8U, 0, 0, 0, 0);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    __aicore__ inline void Increment(LocalTensor<float> cur, uint32_t col, uint32_t row, uint32_t nr)
    {
        const float xre = v.xv.GetValue(2U * col);
        const float xim = v.xv.GetValue(2U * col + 1U);
        const float yre = v.yv.GetValue(2U * col);
        const float yim = v.yv.GetValue(2U * col + 1U);
        SetVectorMask<float, MaskMode::COUNTER>(0U, 2U * nr);
        Muls<float, false>(cur, v.xv[2U * row], yre, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(cur, v.xr, yim, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(cur, v.yv[2U * row], xre, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Axpy<float, float, false>(cur, v.yr, xim, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
    }
    __aicore__ inline void RunRow(
        uint32_t first, uint32_t last, uint32_t row, uint32_t rowCount, const Csyr2TilingData& t,
        bool (&storePending)[2])
    {
        const uint32_t begin = first > row ? first : row;
        if (begin >= last) {
            return;
        }
        uint32_t buffer = 0;
        uint32_t nr = begin + 1U - row < rowCount ? begin + 1U - row : rowCount;
        SetMaskCount();
        for (uint32_t col = begin; col < last; ++col) {
            auto cur = buffer ? a1 : a0;
            if (storePending[buffer]) {
                wait_flag(PIPE_MTE3, PIPE_V, buffer ? EVENT_ID3 : EVENT_ID2);
                storePending[buffer] = false;
            }
            const uint32_t next = buffer ^ 1U;
            const uint32_t nextNr = col + 2U - row < rowCount ? col + 2U - row : rowCount;
            Increment(cur, col, row, nr);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            ub_to_gm_align<ArchType::ASCEND_V220, float>(
                v.a[(uint64_t(col) * t.lda + row) * 2ULL], cur, 0, 1, 2U * nr * sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, buffer ? EVENT_ID3 : EVENT_ID2);
            storePending[buffer] = true;
            nr = nextNr;
            buffer = next;
        }
        SetMaskNorm();
        ResetMask();
    }
    __aicore__ inline void Run(uint32_t first, uint32_t last, const Csyr2TilingData& t)
    {
        Csyr2Vector::InitRotationIndex(index, 2U * kRows);
        PipeBarrier<PIPE_V>();
        SetAtomicAdd<float>();
        bool storePending[2] = {false, false};
        for (uint32_t row = 0; row < last; row += kRows) {
            const uint32_t rowCount = last - row < kRows ? last - row : kRows;
            Csyr2Vector::RotateVectors(
                v.xr, v.yr, v.xv[2U * row], v.yv[2U * row], index.ReinterpretCast<uint32_t>(), 2U * rowCount);
            PipeBarrier<PIPE_V>();
            RunRow(first, last, row, rowCount, t, storePending);
        }
        for (uint32_t b = 0; b < 2U; ++b)
            if (storePending[b]) {
                wait_flag(PIPE_MTE3, PIPE_V, b ? EVENT_ID3 : EVENT_ID2);
            }
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
    }
};
template <uint32_t kRows>
__global__ __aicore__ __vector__ void csyr2_window_checked(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t first = Boundary<kRows>(GetBlockIdx(), GetBlockNum(), t.n);
    const uint32_t last = Boundary<kRows>(GetBlockIdx() + 1U, GetBlockNum(), t.n);
    if (first == last) {
        return;
    }
    TPipe pipe;
    DirectWindow<kRows> window;
    window.Init(pipe, xGm, yGm, aGm, t.n, last);
    if (window.HasNonfinite(last)) {
        window.Precise(first, last, t);
        return;
    }
    window.Run(first, last, t);
}
void csyr2_direct_vector_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    // Two resident vectors and five row-tile buffers must fit in 192 KiB UB.
    const uint32_t vectorBytes = (8U * t.n + 31U) & ~31U;
    if (2U * vectorBytes + 5U * 8U * 3072U <= 192U * 1024U) {
        csyr2_window_checked<3072U><<<blocks, nullptr, stream>>>(x, y, a, t);
    } else {
        csyr2_window_checked<2048U><<<blocks, nullptr, stream>>>(x, y, a, t);
    }
}
