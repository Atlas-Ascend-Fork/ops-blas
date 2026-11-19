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
#include "csyr2_vector_common.h"
namespace Csyr2Atomic {
using namespace AscendC;
template <bool Upper, uint32_t Batch>
__aicore__ inline void Compute(
    Csyr2Vector::Vectors& v, LocalTensor<float> av, uint32_t pitch, uint32_t col, uint32_t first, uint32_t last,
    uint32_t n)
{
    for (uint32_t j = 0; j < Batch && col + 16U * j < last; ++j) {
        const uint32_t c = col + 16U * j;
        if constexpr (Upper) {
            v.Increment(av[pitch * j], 2U * c, 0U, c + 1U);
        } else {
            const uint32_t offset = 2U * (c - first);
            v.Increment(av[pitch * j], offset, offset, n - c);
        }
    }
}
template <bool Upper, uint32_t Batch>
__aicore__ inline void Publish(
    GlobalTensor<float> a, LocalTensor<float> av, uint32_t pitch, uint32_t col, uint32_t last, const Csyr2TilingData& t)
{
    for (uint32_t j = 0; j < Batch && col + 16U * j < last; ++j) {
        const uint32_t c = col + 16U * j;
        const uint32_t row = Upper ? 0U : c;
        const uint32_t rows = Upper ? c + 1U : t.n - c;
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            a[2ULL * (uint64_t(c) * t.lda + row)], av[pitch * j], 0, 1, 8U * rows, 0, 0, 0, 0);
    }
}
__aicore__ inline LocalTensor<float> AcquireOutputBuffer(
    Csyr2Vector::AtomicBuffers& storage, uint32_t buffer, const bool (&pending)[2])
{
    auto output = buffer ? storage.a1 : storage.a0;
    if (pending[buffer]) {
        wait_flag(PIPE_MTE3, PIPE_V, buffer ? EVENT_ID3 : EVENT_ID2);
    }
    return output;
}

template <bool Upper, uint32_t Batch>
__aicore__ inline void Run(
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, const Csyr2TilingData& t, uint32_t cut1, uint32_t cut2)
{
    static_assert(Batch == 2U || Batch == 4U, "Supported column batches");
    const Csyr2Vector::ColumnRange range = Csyr2Vector::AtomicRange(t.n, cut1, cut2);
    const uint32_t first = range.first;
    const uint32_t last = range.last;
    if (first >= last) {
        return;
    }
    TPipe pipe;
    const uint32_t rows0 = Upper ? last : t.n - first;
    const uint32_t bytes = (8U * rows0 + 255U) & ~255U;
    Csyr2Vector::AtomicBuffers storage;
    storage.Init(pipe, xGm, yGm, aGm, bytes, Batch);
    auto index = storage.ab0.Get<int32_t>();
    storage.v.Load(Upper ? 0U : first, rows0);
    if constexpr (Upper) {
        Csyr2Vector::InitRotationIndex(index, 2U * rows0);
    } else {
        Csyr2Vector::InitRotationIndex(index, 2U * rows0);
    }
    PipeBarrier<PIPE_ALL>();
    Csyr2Vector::Rotate(storage.v, index.ReinterpretCast<uint32_t>(), 2U * rows0);
    PipeBarrier<PIPE_ALL>();
    const uint32_t pitch = bytes / sizeof(float);
    bool pending[2] = {false, false};
    uint32_t buffer = 0;
    SetAtomicAdd<float>();
    SetMaskCount();
    for (uint32_t col = first; col < last; col += 16U * Batch) {
        auto av = Csyr2Atomic::AcquireOutputBuffer(storage, buffer, pending);
        Compute<Upper, Batch>(storage.v, av, pitch, col, first, last, t.n);
        set_flag(PIPE_V, PIPE_MTE3, buffer ? EVENT_ID1 : EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, buffer ? EVENT_ID1 : EVENT_ID0);
        Publish<Upper, Batch>(storage.v.a, av, pitch, col, last, t);
        set_flag(PIPE_MTE3, PIPE_V, buffer ? EVENT_ID3 : EVENT_ID2);
        pending[buffer] = true;
        buffer ^= 1U;
    }
    Csyr2Vector::Finish<false, true>(pending);
}
} // namespace Csyr2Atomic
