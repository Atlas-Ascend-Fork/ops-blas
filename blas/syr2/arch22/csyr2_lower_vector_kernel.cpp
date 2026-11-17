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
#include "csyr2_scalar_fallback.h"
#include "csyr2_triangle_copy.h"
#include "csyr2_vector_common.h"
using namespace AscendC;

__aicore__ inline void PreciseLower(Csyr2Vector::DiagonalBuffers& storage, uint32_t first, const Csyr2TilingData& t)
{
    const uint32_t rows = t.n - first;
    const uint32_t columns = rows < 4U ? rows : 4U;
    const uint32_t pitch = (2U * rows + 7U) & ~7U;
    Csyr2Triangle::LowerIn(storage.a0, storage.v.a, first, first, rows, columns, t.lda);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t j = 0; j < columns; ++j) {
        const uint32_t c = first + j;
        Csyr2Vector::PreciseColumn(storage.a0[pitch * j], storage.v, 2U * j, c, t.n, first, t);
    }
    PipeBarrier<PIPE_ALL>();
    Csyr2Triangle::LowerOut(
        storage.v.a, storage.a0, storage.saved.ReinterpretCast<float>(), first, first, rows, columns, t.lda,
        storage.sharedHead);
    PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void UpdateLowerBand(
    Csyr2Vector::Vectors& v, LocalTensor<float> av, uint32_t pitch, uint32_t col, uint32_t first, uint32_t rows,
    uint32_t columns)
{
    SetMaskCount();
    for (uint32_t j = 0; j < columns; ++j) {
        const uint32_t c = col + j;
        const float xre = v.xv.GetValue(2U * (c - first));
        const float xim = v.xv.GetValue(2U * (c - first) + 1U);
        const float yre = v.yv.GetValue(2U * (c - first));
        const float yim = v.yv.GetValue(2U * (c - first) + 1U);
        SetVectorMask<float, MaskMode::COUNTER>(0U, 2U * rows);
        Csyr2Vector::Update(
            av[pitch * j], v.xv[2U * (col - first)], v.xr[2U * (col - first)], v.yv[2U * (col - first)],
            v.yr[2U * (col - first)], xre, xim, yre, yim);
    }
}
__aicore__ inline void RunLower(
    Csyr2Vector::DiagonalBuffers& storage, uint32_t first, uint32_t last, const Csyr2TilingData& t)
{
    bool pending[2] = {false, false};
    uint32_t buffer = 0;
    Csyr2Triangle::LowerIn(
        storage.a0, storage.v.a, first, first, t.n - first, t.n - first < 4U ? t.n - first : 4U, t.lda);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    for (uint32_t col = first; col < last; col += 4U) {
        const uint32_t rows = t.n - col;
        const uint32_t pitch = (2U * rows + 7U) & ~7U;
        const uint32_t columns = rows < 4U ? rows : 4U;
        auto av = buffer ? storage.a1 : storage.a0;
        wait_flag(PIPE_MTE2, PIPE_V, buffer ? EVENT_ID1 : EVENT_ID0);
        const uint32_t next = buffer ^ 1U;
        if (col + 4U < last) {
            if (pending[next]) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, next ? EVENT_ID3 : EVENT_ID2);
                pending[next] = false;
            }
            const uint32_t nextCol = col + 4U;
            const uint32_t nextRows = t.n - nextCol;
            Csyr2Triangle::LowerIn(
                next ? storage.a1 : storage.a0, storage.v.a, nextCol, nextCol, nextRows, nextRows < 4U ? nextRows : 4U,
                t.lda);
            set_flag(PIPE_MTE2, PIPE_V, next ? EVENT_ID1 : EVENT_ID0);
        }
        UpdateLowerBand(storage.v, av, pitch, col, first, rows, columns);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        Csyr2Triangle::LowerOut(
            storage.v.a, av, storage.saved.ReinterpretCast<float>()[storage.sharedHead ? 0U : 32U * buffer], col, col,
            rows, columns, t.lda, storage.sharedHead);
        set_flag(PIPE_MTE3, PIPE_MTE2, buffer ? EVENT_ID3 : EVENT_ID2);
        pending[buffer] = true;
        buffer = next;
    }
    for (uint32_t b = 0; b < 2U; ++b)
        if (pending[b]) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, b ? EVENT_ID3 : EVENT_ID2);
        }
    PipeBarrier<PIPE_ALL>();
}
template <uint32_t Weight>
__global__ __aicore__ __vector__ void csyr2_lower_vector(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto range = Csyr2Vector::LowerRange<Weight>(t.n);
    if (range.first >= range.last) {
        return;
    }
    TPipe pipe;
    Csyr2Vector::DiagonalBuffers storage;
    storage.Init(pipe, xGm, yGm, aGm, t.n);
    auto index = storage.ab.Get<int32_t>();
    const uint32_t vectorRows = t.n - range.first;
    Csyr2Vector::InitRotationIndex(index, 2U * vectorRows);
    storage.v.Load(range.first, vectorRows);
    PipeBarrier<PIPE_ALL>();
    if (Csyr2Vector::RequiresPrecise(storage.v, t.n, vectorRows)) {
        PreciseLower(storage, range.first, t);
        return;
    }
    Csyr2Vector::Rotate(storage.v, index.ReinterpretCast<uint32_t>(), 2U * vectorRows);
    PipeBarrier<PIPE_ALL>();
    RunLower(storage, range.first, range.last, t);
}
void csyr2_lower_vector_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    csyr2_lower_vector<1024U><<<blocks, nullptr, stream>>>(x, y, a, t);
}

// One exact suffix transfer per column; left padding aligns all four
// columns to the same row origin. No access above the diagonal.
__aicore__ inline void ContiguousLowerIn(
    LocalTensor<float> dst, GlobalTensor<float> a, uint32_t col, uint32_t rows, uint32_t columns, uint32_t lda)
{
    const uint32_t pitch = (2U * rows + 7U) & ~7U;
#pragma unroll
    for (uint32_t j = 0; j < 4U; ++j) {
        if (j >= columns) {
            break;
        }
        DataCopyExtParams copy{1, 8U * (rows - j), 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, static_cast<uint8_t>(2U * j), 0, 0.0F};
        DataCopyPad(dst[pitch * j], a[2ULL * (uint64_t(col + j) * lda + col + j)], copy, pad);
    }
}

__aicore__ inline void RunContiguous(
    Csyr2Vector::DiagonalBuffers& storage, uint32_t first, uint32_t last, const Csyr2TilingData& t)
{
    bool pending[2] = {false, false};
    uint32_t buffer = 0;
    SetAtomicAdd<float>();
    for (uint32_t col = first; col < last; col += 4U) {
        const uint32_t rows = t.n - col;
        const uint32_t pitch = (2U * rows + 7U) & ~7U;
        const uint32_t columns = rows < 4U ? rows : 4U;
        auto av = buffer ? storage.a1 : storage.a0;
        if (pending[buffer]) {
            wait_flag(PIPE_MTE3, PIPE_V, buffer ? EVENT_ID3 : EVENT_ID2);
            pending[buffer] = false;
        }
        const uint32_t next = buffer ^ 1U;
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0U, 2U * rows);
#pragma unroll
        for (uint32_t j = 0; j < 4U; ++j) {
            if (j >= columns) {
                break;
            }
            const uint32_t c = col + j;
            storage.v.Increment(av[pitch * j], 2U * (c - first), 2U * (col - first), rows);
        }
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        Csyr2Triangle::LowerOut(
            storage.v.a, av, storage.saved.ReinterpretCast<float>()[storage.sharedHead ? 0U : 32U * buffer], col, col,
            rows, columns, t.lda, storage.sharedHead, buffer ? EVENT_ID5 : EVENT_ID4);
        set_flag(PIPE_MTE3, PIPE_V, buffer ? EVENT_ID3 : EVENT_ID2);
        pending[buffer] = true;
        buffer = next;
    }
    for (uint32_t b = 0; b < 2U; ++b)
        if (pending[b]) {
            wait_flag(PIPE_MTE3, PIPE_V, b ? EVENT_ID3 : EVENT_ID2);
        }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
}
template <uint32_t Weight>
__global__ __aicore__ __vector__ void csyr2_lower_contiguous_atomic(
    GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto range = Csyr2Vector::LowerRange<Weight>(t.n);
    if (range.first >= range.last) {
        return;
    }
    TPipe pipe;
    Csyr2Vector::DiagonalBuffers storage;
    storage.Init(pipe, xGm, yGm, aGm, t.n);
    auto index = storage.ab1.Get<int32_t>();
    const uint32_t vectorRows = t.n - range.first;
    Csyr2Vector::InitOddRotationIndex(index, 2U * vectorRows);
    storage.v.Load(range.first, vectorRows);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    PipeBarrier<PIPE_V>();
    Csyr2Vector::Rotate(storage.v, index.ReinterpretCast<uint32_t>(), 2U * vectorRows);
    PipeBarrier<PIPE_V>();
    RunContiguous(storage, range.first, range.last, t);
}
void csyr2_lower_contiguous_do(
    uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    csyr2_lower_contiguous_atomic<1024U><<<blocks, nullptr, stream>>>(x, y, a, t);
}

struct GeneralTiles {
    Csyr2Vector::Vectors v;
    TBuf<TPosition::VECCALC> ab;
    TBuf<TPosition::VECCALC> preserved;
    LocalTensor<float> a0;
    LocalTensor<float> a1;
    LocalTensor<int32_t> saved;
    uint32_t tileRows;
    __aicore__ inline void Init(
        TPipe& pipe, GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, uint32_t first, const Csyr2TilingData& t)
    {
        v.InitGlobals(xGm, yGm, aGm);
        const uint32_t vectorRows = t.n - first;
        const uint32_t bytes = (8U * vectorRows + 255U) & ~255U;
        // Keep only this core's suffix resident; use the remaining UB for two four-column tiles.
        const uint32_t maxTileBytes = ((192U * 1024U - 4U * bytes - 256U) / 8U) & ~255U;
        const uint32_t tileBytes = bytes < maxTileBytes ? bytes : maxTileBytes;
        tileRows = tileBytes / 8U;
        v.Allocate(pipe, bytes);
        pipe.InitBuffer(preserved, 256U);
        pipe.InitBuffer(ab, 8U * tileBytes);
        v.Bind();
        a0 = ab.Get<float>();
        a1 = a0[tileBytes];
        // The combined A allocation also holds the full gather index during setup.
        auto index = ab.Get<int32_t>();
        saved = preserved.Get<int32_t>();
        Csyr2Vector::InitHeadIndex(saved);
        Csyr2Vector::InitHeadIndex(saved[32U]);
        Csyr2Vector::InitRotationIndex(index, 2U * vectorRows);
        v.Load(first, vectorRows);
        PipeBarrier<PIPE_ALL>();
        Csyr2Vector::Rotate(v, index.ReinterpretCast<uint32_t>(), 2U * vectorRows);
        PipeBarrier<PIPE_ALL>();
    }
};
__aicore__ inline void UpdateGeneralTile(
    Csyr2Vector::Vectors& v, LocalTensor<float> av, uint32_t pitch, uint32_t col, uint32_t row, uint32_t first,
    uint32_t rows, uint32_t columns)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0ULL, 2U * rows);
    for (uint32_t j = 0; j < columns; ++j) {
        const uint32_t c = col + j;
        Csyr2Vector::Update(
            av[pitch * j], v.xv[2U * (row - first)], v.xr[2U * (row - first)], v.yv[2U * (row - first)],
            v.yr[2U * (row - first)], v.xv.GetValue(2U * (c - first)), v.xv.GetValue(2U * (c - first) + 1U),
            v.yv.GetValue(2U * (c - first)), v.yv.GetValue(2U * (c - first) + 1U));
    }
}
__aicore__ inline void PrefetchGeneral(
    GeneralTiles& storage, bool (&pending)[2], uint32_t next, uint32_t nextCol, uint32_t nextRow, uint32_t last,
    const Csyr2TilingData& t)
{
    if (nextCol < last) {
        if (pending[next]) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, next ? EVENT_ID3 : EVENT_ID2);
            pending[next] = false;
        }
        const uint32_t nextRows = t.n - nextRow < storage.tileRows ? t.n - nextRow : storage.tileRows;
        const uint32_t nextCols = t.n - nextCol < 4U ? t.n - nextCol : 4U;
        Csyr2Triangle::LowerIn(
            next ? storage.a1 : storage.a0, storage.v.a, nextCol, nextRow, nextRows, nextCols, t.lda);
        set_flag(PIPE_MTE2, PIPE_V, next ? EVENT_ID1 : EVENT_ID0);
    }
}
__aicore__ inline void RunGeneral(GeneralTiles& storage, uint32_t first, uint32_t last, const Csyr2TilingData& t)
{
    bool pending[2] = {false, false};
    uint32_t buffer = 0;
    uint32_t col = first;
    uint32_t row = first;
    const uint32_t firstRows = t.n - first < storage.tileRows ? t.n - first : storage.tileRows;
    const uint32_t firstCols = t.n - first < 4U ? t.n - first : 4U;
    Csyr2Triangle::LowerIn(storage.a0, storage.v.a, col, row, firstRows, firstCols, t.lda);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    while (col < last) {
        const uint32_t rows = t.n - row < storage.tileRows ? t.n - row : storage.tileRows;
        const uint32_t columns = t.n - col < 4U ? t.n - col : 4U;
        const uint32_t pitch = (2U * rows + 7U) & ~7U;
        auto av = buffer ? storage.a1 : storage.a0;
        wait_flag(PIPE_MTE2, PIPE_V, buffer ? EVENT_ID1 : EVENT_ID0);
        const uint32_t nextCol = row + rows < t.n ? col : col + 4U;
        const uint32_t nextRow = row + rows < t.n ? row + rows : nextCol;
        const uint32_t next = buffer ^ 1U;
        PrefetchGeneral(storage, pending, next, nextCol, nextRow, last, t);
        UpdateGeneralTile(storage.v, av, pitch, col, row, first, rows, columns);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        Csyr2Triangle::LowerOut(
            storage.v.a, av, storage.saved.ReinterpretCast<float>()[32U * buffer], col, row, rows, columns, t.lda,
            false);
        set_flag(PIPE_MTE3, PIPE_MTE2, buffer ? EVENT_ID3 : EVENT_ID2);
        pending[buffer] = true;
        buffer = next;
        col = nextCol;
        row = nextRow;
    }
    for (uint32_t b = 0; b < 2U; ++b)
        if (pending[b]) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, b ? EVENT_ID3 : EVENT_ID2);
        }
    PipeBarrier<PIPE_ALL>();
}
// Dedicated unit-stride path for matrices above 2048 and up to 4096 rows.
// Each UB column is rounded to 32 bytes; GM transfers retain the exact length.
__global__ __aicore__ __vector__ void csyr2_lower_general(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t first = Csyr2Vector::LowerBoundary<512U>(GetBlockIdx(), GetBlockNum(), t.n);
    const uint32_t limit = Csyr2Vector::LowerBoundary<512U>(GetBlockIdx() + 1U, GetBlockNum(), t.n);
    const uint32_t last = limit < t.n ? limit : t.n;
    if (first >= last) {
        return;
    }
    TPipe pipe;
    GeneralTiles storage;
    storage.Init(pipe, xGm, yGm, aGm, first, t);
    RunGeneral(storage, first, last, t);
}
void csyr2_lower_general_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    csyr2_lower_general<<<blocks, nullptr, stream>>>(x, y, a, t);
}
