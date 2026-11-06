/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "kernel_operator.h"
#include "gemm_ex_dma_copy.h"
#include "cgemm_ex_kernel.h"
#include "gemm_ex_tiling.h"
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "lib/matmul_intf.h"

using namespace AscendC;
namespace {
constexpr uint32_t TILE = 1024;
__aicore__ inline void Fence() { PipeBarrier<PIPE_ALL>(); }
__aicore__ inline void MakeIndices(
    LocalTensor<uint32_t> idx, LocalTensor<uint32_t> inter, LocalTensor<uint32_t> scratch, uint32_t count)
{
    CreateVecIndex(idx.ReinterpretCast<int32_t>(), 0, count);
    CreateVecIndex(inter.ReinterpretCast<int32_t>(), 0, 2 * count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(idx, idx, 3U, count);
    ShiftRight(scratch, inter, 1U, 2 * count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(scratch, scratch, 1U, 2 * count);
    PipeBarrier<PIPE_V>();
    Sub(inter.ReinterpretCast<int32_t>(), inter.ReinterpretCast<int32_t>(), scratch.ReinterpretCast<int32_t>(),
        2 * count);
    ShiftRight(scratch, scratch, 1U, 2 * count);
    PipeBarrier<PIPE_V>();
    Muls(inter.ReinterpretCast<int32_t>(), inter.ReinterpretCast<int32_t>(), static_cast<int32_t>(count), 2 * count);
    PipeBarrier<PIPE_V>();
    Add(inter.ReinterpretCast<int32_t>(), inter.ReinterpretCast<int32_t>(), scratch.ReinterpretCast<int32_t>(),
        2 * count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(inter, inter, 2U, 2 * count);
    PipeBarrier<PIPE_V>();
}
} // namespace

extern "C" __global__ __aicore__ void cgemm_ex_split(
    GM_ADDR input, GM_ADDR real, GM_ADDR imag, uint64_t count, float sign)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    TBuf<TPosition::VECCALC> indices;
    pipe.InitBuffer(buffer, 4 * TILE * sizeof(float));
    pipe.InitBuffer(indices, TILE * sizeof(uint32_t));
    auto raw = buffer.Get<float>();
    auto re = raw[2 * TILE];
    auto im = raw[3 * TILE];
    auto idx = indices.Get<uint32_t>();
    for (uint32_t j = 0; j < TILE; ++j) {
        idx.SetValue(j, 2 * j * sizeof(float));
    }
    Fence();
    auto src = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(input));
    auto rg = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(real));
    auto ig = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(imag));
    for (uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * TILE; start < count;
         start += static_cast<uint64_t>(GetBlockNum()) * TILE) {
        uint32_t len = static_cast<uint32_t>(count - start < TILE ? count - start : TILE);
        DataCopyExtParams in{1, len * 8U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(raw, src[2 * start], in, pad);
        Fence();
        Gather(re, raw, idx, 0U, len);
        Gather(im, raw[1], idx, 0U, len);
        Fence();
        Muls(im, im, sign, len);
        Fence();
        DataCopyExtParams out{1, len * 4U, 0, 0, 0};
        DataCopyPad(rg[start], re, out);
        DataCopyPad(ig[start], im, out);
        Fence();
    }
}

// Batch adjacent complex columns so short rows do not incur one GM roundtrip
// and a complete pipeline fence per column. Only the identity epilogue uses
// this path; general alpha/beta handling remains below.
__aicore__ inline void CopyExpandedIdentity(GM_ADDR product, GM_ADDR output, CgemmExLayout t)
{
    constexpr uint32_t CAP = 6144;
    const uint32_t mp = t.mode == 2 ? t.packedLd / 2 : t.packedLd;
    const uint32_t width = (2 * t.m + 7) / 8 * 8;
    const uint32_t batch = CAP / (2 * mp);
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indexBuffer;
    pipe.InitBuffer(values, 2 * CAP * sizeof(float));
    pipe.InitBuffer(indexBuffer, 4 * CAP * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto out = raw[CAP];
    auto indices = indexBuffer.Get<uint32_t>();
    auto base = indices[CAP];
    auto scratch = indices[2 * CAP];
    if (t.mode == 3) {
        MakeIndices(scratch, base, scratch[CAP], mp);
        for (uint32_t col = 0; col < batch; ++col) {
            Adds(
                indices[col * width].ReinterpretCast<int32_t>(), base.ReinterpretCast<int32_t>(),
                static_cast<int32_t>(col * 2 * mp * sizeof(float)), width);
        }
        PipeBarrier<PIPE_V>();
    }
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(product));
    auto destination = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * batch; first < t.n;
         first += static_cast<uint64_t>(GetBlockNum()) * batch) {
        const uint32_t count = min(batch, static_cast<uint32_t>(t.n - first));
        DataCopy(raw, source[first * 2 * mp], count * 2 * mp);
        Fence();
        if (t.mode == 3) {
            for (uint32_t col = 0; col < count; ++col) {
                Muls(raw[col * 2 * mp + mp], raw[col * 2 * mp + mp], -1.0f, mp);
            }
            PipeBarrier<PIPE_V>();
            Gather(out, raw, indices, 0U, count * width);
            Fence();
        }
        const uint32_t sourceStride = t.mode == 2 ? (2 * mp - width) / 8 : 0;
        DataCopyExtParams copy{
            static_cast<uint16_t>(count), static_cast<uint32_t>(t.m) * 8U, sourceStride,
            static_cast<uint32_t>(t.ldc - t.m) * 8U, 0};
        DataCopyPad(destination[first * 2 * t.ldc], t.mode == 2 ? raw : out, copy);
        Fence();
    }
}

struct ComplexCombineTile {
    CgemmExLayout t;
    uint32_t tile;
    LocalTensor<float> re;
    LocalTensor<float> im;
    LocalTensor<float> tmp;
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> raw;
    LocalTensor<float> planar;
    LocalTensor<uint32_t> idx;
    LocalTensor<uint32_t> inter;
    GlobalTensor<float> rg;
    GlobalTensor<float> ig;
    GlobalTensor<float> cg;

    __aicore__ inline void CopyIdentity()
    {
        const bool flat = t.packedLd == 2 * t.m && t.ldc == t.m;
        const uint64_t rows = flat ? static_cast<uint64_t>(t.m) * t.n : t.m;
        const uint64_t columns = flat ? 1 : t.n;
        const uint64_t pieces = (rows + tile - 1) / tile;
        for (uint64_t task = GetBlockIdx(); task < pieces * columns; task += GetBlockNum()) {
            const uint64_t col = task / pieces;
            const uint64_t row = task % pieces * tile;
            const uint32_t len = static_cast<uint32_t>(min(static_cast<uint64_t>(tile), rows - row));
            DataCopyPad(raw, rg[col * t.packedLd + 2 * row], {1, len * 8U, 0, 0, 0}, {false, 0, 0, 0});
            Fence();
            DataCopyPad(cg[2 * (col * t.ldc + row)], raw, {1, len * 8U, 0, 0, 0});
            Fence();
        }
    }
    __aicore__ inline void LoadProduct(uint64_t row, uint64_t col, uint32_t len)
    {
        const uint64_t packed = col * t.m + row;
        DataCopyExtParams one{1, len * 4U, 0, 0, 0};
        DataCopyExtParams two{1, len * 8U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        if (t.mode == 3) {
            DataCopyPad(re, rg[2 * col * t.packedLd + row], one, pad);
            DataCopyPad(im, rg[(2 * col + 1) * t.packedLd + row], one, pad);
            Fence();
            Muls(im, im, -1.0f, len);
            Fence();
        } else if (t.mode == 2) {
            DataCopyPad(raw, rg[col * t.packedLd + 2 * row], two, pad);
            Fence();
            Gather(re, raw, idx, 0U, len);
            Gather(im, raw[1], idx, 0U, len);
            Fence();
        } else {
            DataCopyPad(re, rg[packed], one, pad);
            DataCopyPad(im, ig[packed], one, pad);
            Fence();
        }
    }
    __aicore__ inline void ScaleProduct(uint32_t len)
    {
        if (t.ar == 1.0f && t.ai == 0.0f) {
            DataCopy(planar, re, tile);
            DataCopy(planar[tile], im, tile);
        } else {
            Muls(planar, re, t.ar, len);
            Muls(tmp, im, t.ai, len);
            Fence();
            Sub(planar, planar, tmp, len);
            Muls(planar[tile], im, t.ar, len);
            Muls(tmp, re, t.ai, len);
            Fence();
            Add(planar[tile], planar[tile], tmp, len);
        }
    }
    __aicore__ inline void AccumulateC(uint64_t pos, uint32_t len)
    {
        DataCopyExtParams two{1, len * 8U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(raw, cg[pos * 2], two, pad);
        Fence();
        Gather(cr, raw, idx, 0U, len);
        Gather(ci, raw[1], idx, 0U, len);
        Fence();
        Muls(re, cr, t.br, len);
        Muls(im, ci, t.bi, len);
        Fence();
        Sub(re, re, im, len);
        Muls(im, ci, t.br, len);
        Muls(tmp, cr, t.bi, len);
        Fence();
        Add(im, im, tmp, len);
        Fence();
        if (t.mode == 1) {
            DataCopy(planar, re, tile);
            DataCopy(planar[tile], im, tile);
        } else {
            Add(planar, planar, re, len);
            Add(planar[tile], planar[tile], im, len);
        }
    }
    __aicore__ inline void Run(uint64_t row, uint64_t col, uint32_t len)
    {
        const uint64_t pos = col * t.ldc + row;
        if (t.mode != 1) {
            LoadProduct(row, col, len);
            ScaleProduct(len);
        } else {
            Duplicate(planar, 0.0f, 2 * tile);
        }
        Fence();
        if (t.br != 0.0f || t.bi != 0.0f) {
            AccumulateC(pos, len);
        }
        Fence();
        Gather(raw, planar, inter, 0U, 2 * len);
        Fence();
        DataCopyExtParams two{1, len * 8U, 0, 0, 0};
        DataCopyPad(cg[pos * 2], raw, two);
        Fence();
    }
};

extern "C" __global__ __aicore__ void cgemm_ex_combine(GM_ADDR real, GM_ADDR imag, GM_ADDR output, CgemmExLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ((t.mode == 2 || t.mode == 3) && t.m <= 3072 && t.ar == 1.0f && t.ai == 0.0f && t.br == 0.0f && t.bi == 0.0f) {
        CopyExpandedIdentity(real, output, t);
        return;
    }
    const uint32_t TILE = min(2048U, (static_cast<uint32_t>(t.m) + 15) / 16 * 16);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    TBuf<TPosition::VECCALC> indices;
    pipe.InitBuffer(buffer, 10 * TILE * sizeof(float));
    pipe.InitBuffer(indices, 5 * TILE * sizeof(uint32_t));
    auto all = buffer.Get<float>();
    auto re = all;
    auto im = all[TILE];
    auto tmp = all[2 * TILE];
    auto cr = all[3 * TILE];
    auto ci = all[4 * TILE];
    auto raw = all[5 * TILE];
    auto planar = all[7 * TILE];
    auto idx = indices.Get<uint32_t>();
    auto inter = idx[TILE];
    auto rg = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(real));
    auto ig = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(imag));
    auto cg = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    ComplexCombineTile operation{t, TILE, re, im, tmp, cr, ci, raw, planar, idx, inter, rg, ig, cg};
    if (t.mode == 2 && t.ar == 1.0f && t.ai == 0.0f && t.br == 0.0f && t.bi == 0.0f) {
        operation.CopyIdentity();
        return;
    }
    MakeIndices(idx, inter, idx[3 * TILE], TILE);
    uint64_t tiles = (static_cast<uint64_t>(t.m) + TILE - 1) / TILE;
    for (uint64_t task = GetBlockIdx(); task < tiles * t.n; task += GetBlockNum()) {
        uint64_t col = task / tiles;
        uint64_t row = (task % tiles) * TILE;
        uint32_t len = static_cast<uint32_t>(t.m - row < TILE ? t.m - row : TILE);
        operation.Run(row, col, len);
    }
}

void cgemm_ex_split_do(
    uint32_t blocks, void* stream, uint8_t* input, uint8_t* real, uint8_t* imag, uint64_t count, float sign)
{
    cgemm_ex_split<<<blocks, nullptr, stream>>>(input, real, imag, count, sign);
}
__aicore__ inline void CopyExpandedInterleaved(GM_ADDR product, GM_ADDR output, CgemmExLayout t)
{
    constexpr uint32_t CAP = 49152;
    const uint32_t mp = t.packedLd / 2;
    const uint32_t width = (2 * t.m + 7) / 8 * 8;
    const uint32_t batch = min(CAP / (2 * mp), static_cast<uint32_t>((t.n + GetBlockNum() - 1) / GetBlockNum()));
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, CAP * sizeof(float));
    auto local = buffer.Get<float>();
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(product));
    auto destination = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * batch; first < t.n;
         first += static_cast<uint64_t>(GetBlockNum()) * batch) {
        const uint32_t count = min(batch, static_cast<uint32_t>(t.n - first));
        DataCopy(local, source[first * 2 * mp], count * 2 * mp);
        Fence();
        DataCopyExtParams cp{
            static_cast<uint16_t>(count), static_cast<uint32_t>(t.m) * 8U, (2 * mp - width) / 8U,
            static_cast<uint32_t>(t.ldc - t.m) * 8U, 0};
        DataCopyPad(destination[first * 2 * t.ldc], local, cp);
        Fence();
    }
}

class CompactComplexCopy : private gemm_ex::DmaCopyParams, private gemm_ex::DmaCopyBuffers {
public:
    __aicore__ inline CompactComplexCopy(gemm_ex::DmaCopyParams parameters, gemm_ex::DmaCopyBuffers buffers)
        : DmaCopyParams(parameters), DmaCopyBuffers(buffers)
    {}
    __aicore__ inline void Run()
    {
        Preload();
        for (uint32_t wave = 0; wave < waves; ++wave) {
            const uint32_t col = first + wave * batch;
            const uint32_t count = min(batch, end - col);
            const bool slot = wave & 1U;
            auto local = slot ? local1 : local0;
            Compact(slot, local, count);
            Prefetch(wave, slot, col);
            Store(slot, local, col, count);
        }
        gemm_ex::FinishCopyReuse(waves);
    }

private:
    __aicore__ inline void Preload()
    {
        const uint32_t count0 = min(batch, end - first);
        DataCopy(local0, source[static_cast<uint64_t>(first) * sourceLd], count0 * sourceLd);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (waves >= 2) {
            const uint32_t count1 = min(batch, end - first - batch);
            DataCopy(local1, source[static_cast<uint64_t>(first + batch) * sourceLd], count1 * sourceLd);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
    }
    __aicore__ inline void Compact(bool slot, LocalTensor<float> local, uint32_t count)
    {
        if (slot) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        } else {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        if ASCEND_IS_AIV {
            SetMaskCount();
            set_vector_mask(0, width);
            auto* data = reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
            vreducev2(
                data, data, data + 1024, static_cast<uint16_t>(count), 1, 7, static_cast<uint16_t>(sourceLd / 8), 0);
            SetMaskNorm();
        }
        if (slot) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        } else {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
    }
    __aicore__ inline void Prefetch(uint32_t wave, bool slot, uint32_t col)
    {
        if (wave >= 1 && wave + 1 < waves) {
            const uint32_t next = col + batch;
            const uint32_t nextCount = min(batch, end - next);
            if (slot) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                DataCopy(local0, source[static_cast<uint64_t>(next) * sourceLd], nextCount * sourceLd);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                DataCopy(local1, source[static_cast<uint64_t>(next) * sourceLd], nextCount * sourceLd);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
    }
    __aicore__ inline void Store(bool slot, LocalTensor<float> local, uint32_t col, uint32_t count)
    {
        if (slot) {
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        } else {
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        DataCopyPad(dest[static_cast<uint64_t>(col) * destLd], local, {1, count * width * 4U, 0, 0, 0});
        if (slot) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        } else {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
};

extern "C" __global__ __aicore__ void cgemm_ex_copy_interleaved(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n, uint32_t ldc, uint32_t sourceLd, uint32_t per,
    uint32_t batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetMaskNorm();
    const uint32_t width = 2U * m;
    const uint32_t destLd = 2U * ldc;
    constexpr uint32_t CAP = 24576;
    const uint32_t first = GetBlockIdx() * per;
    const uint32_t end = min(first + per, n);
    if (first >= end) {
        return;
    }
    LocalTensor<float> local0(TPosition::VECCALC, 0, CAP);
    LocalTensor<float> local1(TPosition::VECCALC, CAP * sizeof(float), CAP);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(inputGm));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outputGm));
    const uint32_t dma = (width + 7) / 8 * 8;
    const bool compactRows = destLd == width && sourceLd >= dma && sourceLd % 8 == 0 && sourceLd <= CAP;
    if (compactRows) {
        batch = min(batch, CAP / sourceLd);
    }
    const uint32_t waves = (end - first + batch - 1) / batch;
    const gemm_ex::DmaCopyParams parameters{width, destLd, sourceLd, first, end, batch, waves};
    const gemm_ex::DmaCopyBuffers buffers{local0, local1, source, dest};
    if (compactRows) {
        CompactComplexCopy(parameters, buffers).Run();
        return;
    }
    gemm_ex::StridedDmaCopy<true>(parameters, buffers).Run();
}

template <uint32_t CAP>
__global__ __aicore__ void cgemm_ex_copy_mode3(GM_ADDR product, GM_ADDR output, CgemmExLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t mp = t.packedLd;
    const uint32_t width = (2 * t.m + 7) / 8 * 8;
    const uint32_t batch = min(CAP / (2 * mp), static_cast<uint32_t>((t.n + GetBlockNum() - 1) / GetBlockNum()));
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indexBuffer;
    pipe.InitBuffer(values, 3 * CAP * sizeof(float));
    pipe.InitBuffer(indexBuffer, 2 * CAP * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto out = raw[CAP];
    auto sign = raw[2 * CAP];
    auto indices = indexBuffer.Get<uint32_t>();
    auto base = indices[CAP];
    MakeIndices(raw.ReinterpretCast<uint32_t>(), base, out.ReinterpretCast<uint32_t>(), mp);
    for (uint32_t col = 0; col < batch; ++col) {
        Adds(
            indices[col * width].ReinterpretCast<int32_t>(), base.ReinterpretCast<int32_t>(),
            static_cast<int32_t>(col * 2 * mp * sizeof(float)), width);
    }
    PipeBarrier<PIPE_V>();
    // raw/out were vector scratch above. Do not let MTE2 overwrite either early.
    Fence();
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(product));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * batch; first < t.n;
         first += static_cast<uint64_t>(GetBlockNum()) * batch) {
        const uint32_t count = min(batch, static_cast<uint32_t>(t.n - first));
        DataCopy(raw, source[first * 2 * mp], count * 2 * mp);
        Fence();
        Gather(out, raw, indices, 0U, count * width);
        PipeBarrier<PIPE_V>();
        // Odd interleaved elements are the negated imaginary plane. A masked
        // scalar multiply avoids constructing and loading a sign vector.
        SetMaskNorm();
        uint64_t oddMask[2] = {0xAAAAAAAAAAAAAAAAULL, 0};
        Muls(out, out, -1.0f, oddMask, static_cast<uint8_t>((count * width + 63) / 64), UnaryRepeatParams{1, 1, 8, 8});
        Fence();
        DataCopyExtParams cp{
            static_cast<uint16_t>(count), static_cast<uint32_t>(t.m) * 8U, 0, static_cast<uint32_t>(t.ldc - t.m) * 8U,
            0};
        DataCopyPad(dest[first * 2 * t.ldc], out, cp);
        Fence();
    }
}

// Tile the physical transpose over both axes. Each output K slice is512B;
// this path requires kp%64==0 and leaves every padded coordinate zero.
__aicore__ inline void MakeCanonicalTileIndices(
    LocalTensor<uint32_t> i, LocalTensor<uint32_t> offset, LocalTensor<uint32_t> tmp)
{
    constexpr uint32_t CAP = 8192;
    constexpr uint32_t RAW_PITCH = 136;
    CreateVecIndex(i.ReinterpretCast<int32_t>(), 0, CAP);
    PipeBarrier<PIPE_V>();
    ShiftRight(offset, i, 7U, CAP);
    PipeBarrier<PIPE_V>();
    ShiftLeft(offset, offset, 1U, CAP);
    ShiftLeft(tmp, i, 25U, CAP);
    PipeBarrier<PIPE_V>();
    ShiftRight(tmp, tmp, 26U, CAP);
    PipeBarrier<PIPE_V>();
    Muls(tmp.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(RAW_PITCH), CAP);
    PipeBarrier<PIPE_V>();
    Add(offset.ReinterpretCast<int32_t>(), offset.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), CAP);
    PipeBarrier<PIPE_V>();
    ShiftLeft(tmp, i, 31U, CAP);
    PipeBarrier<PIPE_V>();
    ShiftRight(tmp, tmp, 31U, CAP);
    PipeBarrier<PIPE_V>();
    Add(offset.ReinterpretCast<int32_t>(), offset.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), CAP);
    PipeBarrier<PIPE_V>();
    ShiftLeft(offset, offset, 2U, CAP);
}

__aicore__ inline void LoadCanonicalTile(
    LocalTensor<float> raw, GlobalTensor<float> source, CgemmExpandedLayout t, uint32_t firstN, uint32_t firstK,
    uint32_t nc)
{
    constexpr uint32_t KC = 64;
    constexpr uint32_t RAW_PITCH = 136;
    if (firstN < t.n && firstK < t.k) {
        const uint32_t vn = min(nc, static_cast<uint32_t>(t.n) - firstN);
        const uint32_t vk = min(KC, static_cast<uint32_t>(t.k) - firstK);
        const uint32_t dma = (2 * vn + 7) / 8 * 8;
        DataCopyPad(
            raw, source[2 * (static_cast<uint64_t>(firstK) * t.ldb + firstN)],
            {static_cast<uint16_t>(vk), vn * 8U, static_cast<uint32_t>(t.ldb - vn) * 8U, (RAW_PITCH - dma) / 8, 0},
            {true, 0, static_cast<uint8_t>(dma - 2 * vn), 0.0f});
    }
}

__aicore__ inline void ExpandCanonicalTiled107(GM_ADDR input, GM_ADDR output, CgemmExpandedLayout t)
{
    constexpr uint32_t NC = 64;
    constexpr uint32_t KC = 64;
    constexpr uint32_t CAP = 2 * NC * KC;
    constexpr uint32_t RAW_PITCH = 2 * NC + 8;
    constexpr uint32_t RAW = RAW_PITCH * KC;
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indices;
    pipe.InitBuffer(values, (RAW + CAP) * sizeof(float));
    pipe.InitBuffer(indices, 3 * CAP * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto out = raw[RAW];
    auto i = indices.Get<uint32_t>();
    auto offset = i[CAP];
    auto tmp = i[2 * CAP];
    MakeCanonicalTileIndices(i, offset, tmp);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(input));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    const uint32_t kt = t.kp / KC;
    const uint32_t nt = (t.np + NC - 1) / NC;
    for (uint32_t job = GetBlockIdx(); job < kt * nt; job += GetBlockNum()) {
        const uint32_t firstN = (job / kt) * NC;
        const uint32_t firstK = (job % kt) * KC;
        const uint32_t nc = min(NC, static_cast<uint32_t>(t.np) - firstN);
        Duplicate(raw, 0.0f, RAW);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        LoadCanonicalTile(raw, source, t, firstN, firstK, nc);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Gather(out, raw, offset, 0U, CAP);
        PipeBarrier<PIPE_V>();
        if (t.tb == 2) {
            uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0};
            Muls(out, out, -1.0f, odd, uint8_t(CAP / 64), {1, 1, 8, 8});
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(
            dest[2 * (static_cast<uint64_t>(firstN) * t.kp + firstK)], out,
            {static_cast<uint16_t>(nc), KC * 8U, 0, static_cast<uint32_t>(t.kp - KC) * 8U, 0});
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

extern "C" __global__ __aicore__ void cgemm_ex_copy_smalln107(GM_ADDR input, GM_ADDR output, CgemmExLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetMaskNorm();
    constexpr uint32_t CAP = 6144;
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indices;
    pipe.InitBuffer(values, 2 * CAP * sizeof(float));
    pipe.InitBuffer(indices, 3 * CAP * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto out = raw[CAP];
    auto i = indices.Get<uint32_t>();
    auto index = i[CAP];
    auto tmp = i[2 * CAP];
    const uint32_t mp = t.packedLd;
    const uint32_t used = 2 * mp;
    CreateVecIndex(i.ReinterpretCast<int32_t>(), 0, used);
    PipeBarrier<PIPE_V>();
    ShiftRight(index, i, 1U, used);
    ShiftLeft(tmp, i, 31U, used);
    PipeBarrier<PIPE_V>();
    ShiftRight(tmp, tmp, 31U, used);
    PipeBarrier<PIPE_V>();
    Muls(tmp.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), static_cast<int32_t>(mp), used);
    PipeBarrier<PIPE_V>();
    Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), used);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, 2U, used);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(input));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    for (uint32_t col = GetBlockIdx(); col < t.n; col += GetBlockNum()) {
        DataCopy(raw, source[static_cast<uint64_t>(col) * used], used);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Muls(raw[mp], raw[mp], -1.0f, mp);
        PipeBarrier<PIPE_V>();
        Gather(out, raw, index, 0U, 2 * t.m);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dest[static_cast<uint64_t>(col) * 2 * t.ldc], out, {1, static_cast<uint32_t>(t.m) * 8U, 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

bool LaunchPlanarIdentityCopy(
    uint32_t blocks, void* stream, uint8_t* real, uint8_t* output, const CgemmExLayout& t, bool identity)
{
    if (t.mode != 3 || !identity) {
        return false;
    }
    if (t.m <= 3072 && t.n <= 64) {
        cgemm_ex_copy_smalln107<<<blocks, nullptr, stream>>>(real, output, t);
    } else if (t.m <= 128 && t.n <= 256) {
        cgemm_ex_copy_mode3<2048><<<blocks, nullptr, stream>>>(real, output, t);
    } else if (t.n >= 512 && t.m <= 3072) {
        cgemm_ex_copy_mode3<6144><<<blocks, nullptr, stream>>>(real, output, t);
    } else {
        return false;
    }
    return true;
}
void cgemm_ex_combine_do(uint32_t blocks, void* stream, uint8_t* real, uint8_t* imag, uint8_t* output, CgemmExLayout t)
{
    if (blocks == 0 || t.m <= 0 || t.n <= 0) {
        return;
    }
    const bool identity = t.ar == 1.0f && t.ai == 0.0f && t.br == 0.0f && t.bi == 0.0f;

    if (LaunchPlanarIdentityCopy(blocks, stream, real, output, t, identity)) {
        return;
    }
    if (t.mode == 2 && t.m <= 3072 && identity) {
        const uint32_t nominal = (static_cast<uint32_t>(t.n) + blocks - 1) / blocks;
        const uint32_t aligned = (nominal + 7) / 8 * 8;
        uint32_t per = nominal;
        uint32_t use = blocks;
        uint32_t batch =
            std::min<uint32_t>(24576U / ((2U * static_cast<uint32_t>(t.m) + 7U) / 8U * 8U), (nominal + 1U) / 2U);
        if ((2U * static_cast<uint32_t>(t.m)) % 8 != 0 && nominal >= 8 && t.packedLd >= 512 &&
            static_cast<uint64_t>(aligned) * t.packedLd <= 2ULL * 24576) {
            per = aligned;
            batch = aligned / 2;
            use = std::min(blocks, (static_cast<uint32_t>(t.n) + per - 1) / per);
        }
        cgemm_ex_copy_interleaved<<<use, nullptr, stream>>>(real, output, t.m, t.n, t.ldc, t.packedLd, per, batch);
    } else {
        cgemm_ex_combine<<<blocks, nullptr, stream>>>(real, imag, output, t);
    }
}

struct ComplexDirectTile {
    CgemmExDirectLayout t;
    LocalTensor<float> raw;
    LocalTensor<float> ar;
    LocalTensor<float> ai;
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> tmp;
    LocalTensor<float> tmp2;
    LocalTensor<float> ca;
    LocalTensor<float> cb;
    LocalTensor<uint32_t> idx;
    LocalTensor<uint32_t> inter;
    LocalTensor<uint32_t> inputIdx;
    GlobalTensor<float> ag;
    GlobalTensor<float> bg;
    GlobalTensor<float> cg;
    bool cached;
    static constexpr uint32_t TILE = 256;

    __aicore__ inline void ApplyBeta(uint64_t pos, uint32_t len)
    {
        DataCopyExtParams cp{1, len * 8U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(raw, cg[pos], cp, pad);
        Fence();
        Gather(ar, raw, idx, 0U, len);
        Gather(ai, raw[1], idx, 0U, len);
        Fence();
        Muls(cr, ar, t.br, len);
        Muls(ci, ai, t.br, len);
        Muls(tmp, ai, t.bi, len);
        Muls(tmp2, ar, t.bi, len);
        Fence();
        Sub(cr, cr, tmp, len);
        Add(ci, ci, tmp2, len);
        Fence();
    }
    __aicore__ inline void LoadA(uint64_t row, int l, uint32_t len)
    {
        DataCopyExtParams cp{1, len * 8U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        if (cached) {
            uint32_t base = static_cast<uint32_t>(l * (t.ta ? 1 : t.lda) * 8);
            Gather(ar, ca, inputIdx, base, len);
            Gather(ai, ca, inputIdx, base + 4, len);
            PipeBarrier<PIPE_V>();
        } else {
            if (t.ta == 0) {
                DataCopyPad(raw, ag[2 * (static_cast<uint64_t>(l) * t.lda + row)], cp, pad);
            } else {
                for (uint32_t i = 0; i < len; ++i) {
                    uint64_t ap = 2 * ((row + i) * t.lda + l);
                    raw.SetValue(2 * i, ag.GetValue(ap));
                    raw.SetValue(2 * i + 1, ag.GetValue(ap + 1));
                }
            }
            Fence();
            Gather(ar, raw, idx, 0U, len);
            Gather(ai, raw[1], idx, 0U, len);
            Fence();
        }
    }
    __aicore__ inline void Reduce(uint64_t row, uint64_t col, uint32_t len)
    {
        for (int l = 0; l < t.k; ++l) {
            uint64_t bp = 2 * (t.tb == 0 ? col * t.ldb + l : static_cast<uint64_t>(l) * t.ldb + col);
            float br = cached ? cb.GetValue(bp) : bg.GetValue(bp);
            float bi = (cached ? cb.GetValue(bp + 1) : bg.GetValue(bp + 1)) * (t.tb == 2 ? -1.0f : 1.0f);
            float tr = t.ar * br - t.ai * bi;
            float ti = t.ar * bi + t.ai * br;
            LoadA(row, l, len);
            if (t.ta == 2) {
                Muls(ai, ai, -1.0f, len);
                PipeBarrier<PIPE_V>();
            }
            Axpy(cr, ar, tr, len);
            Axpy(ci, ai, tr, len);
            PipeBarrier<PIPE_V>();
            Axpy(cr, ai, -ti, len);
            Axpy(ci, ar, ti, len);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Run(uint64_t row, uint64_t col, uint32_t len)
    {
        const uint64_t pos = (col * t.ldc + row) * 2;
        for (uint32_t i = 0; cached && i < len; ++i) {
            inputIdx.SetValue(i, static_cast<uint32_t>((t.ta ? (row + i) * t.lda : row + i) * 8));
        }
        Fence();
        Duplicate(cr, 0.0f, 2 * TILE);
        Fence();
        if (t.br != 0.0f || t.bi != 0.0f) {
            ApplyBeta(pos, len);
        }
        Reduce(row, col, len);
        Gather(raw, cr, inter, 0U, 2 * len);
        Fence();
        DataCopyExtParams cp{1, len * 8U, 0, 0, 0};
        DataCopyPad(cg[pos], raw, cp);
        Fence();
    }
};

__aicore__ inline void MakeDirectComplexIndices(LocalTensor<uint32_t> idx, LocalTensor<uint32_t> inter, uint32_t rows)
{
    constexpr uint32_t TILE = 256;
    for (uint32_t i = 0; i < min(TILE, rows); ++i) {
        idx.SetValue(i, i * 8);
        inter.SetValue(2 * i, i * 4);
        inter.SetValue(2 * i + 1, (TILE + i) * 4);
    }
    Fence();
}

extern "C" __global__ __aicore__ void cgemm_ex_direct(GM_ADDR a, GM_ADDR b, GM_ADDR c, CgemmExDirectLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t TILE = 256;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    TBuf<TPosition::VECCALC> indices;
    TBuf<TPosition::VECCALC> cacheBuffer;
    pipe.InitBuffer(buffer, 8 * TILE * sizeof(float));
    pipe.InitBuffer(indices, 4 * TILE * sizeof(uint32_t));
    pipe.InitBuffer(cacheBuffer, 8192 * sizeof(float));
    auto raw = buffer.Get<float>();
    auto ar = raw[2 * TILE];
    auto ai = raw[3 * TILE];
    auto cr = raw[4 * TILE];
    auto ci = raw[5 * TILE];
    auto tmp = raw[6 * TILE];
    auto tmp2 = raw[7 * TILE];
    auto idx = indices.Get<uint32_t>();
    auto inter = idx[TILE];
    auto inputIdx = idx[3 * TILE];
    MakeDirectComplexIndices(idx, inter, static_cast<uint32_t>(t.m));
    auto ag = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(a));
    auto bg = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(b));
    auto cg = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(c));
    const uint64_t na = 2ULL * t.lda * (t.ta ? t.m : t.k);
    const uint64_t nb = 2ULL * t.ldb * (t.tb ? t.k : t.n);
    const uint32_t offset = static_cast<uint32_t>((na + 7) / 8 * 8);
    const bool cached = na + nb + 16 <= 8192;
    auto ca = cacheBuffer.Get<float>();
    auto cb = ca[offset < 8192 ? offset : 0];
    if (cached) {
        DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        DataCopyPad(ca, ag, {1, static_cast<uint32_t>(na) * 4U, 0, 0, 0}, padding);
        DataCopyPad(cb, bg, {1, static_cast<uint32_t>(nb) * 4U, 0, 0, 0}, padding);
        Fence();
    }
    ComplexDirectTile operation{t, raw, ar, ai, cr, ci, tmp, tmp2, ca, cb, idx, inter, inputIdx, ag, bg, cg, cached};
    uint64_t tiles = (static_cast<uint64_t>(t.m) + TILE - 1) / TILE;
    for (uint64_t task = GetBlockIdx(); task < tiles * t.n; task += GetBlockNum()) {
        const uint64_t col = task / tiles;
        const uint64_t row = (task % tiles) * TILE;
        const uint32_t len = static_cast<uint32_t>(t.m - row < TILE ? t.m - row : TILE);
        operation.Run(row, col, len);
    }
}

void cgemm_ex_direct_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, CgemmExDirectLayout t)
{
    cgemm_ex_direct<<<blocks, nullptr, stream>>>(a, b, c, t);
}

// Exact sign/permute expansion of complex columns. Batch contiguous columns;
// explicit events retain every producer/consumer and buffer-reuse dependency.
__aicore__ inline void NegateParity107(LocalTensor<float> data, bool imaginary, uint32_t count)
{
    const uint64_t pattern = imaginary ? 0xaaaaaaaaaaaaaaaaULL : 0x5555555555555555ULL;
    uint64_t mask[2] = {pattern, 0};
    const uint32_t repeats = count / 64;
    const uint32_t tail = count % 64;
    if (repeats) {
        Muls(data, data, -1.0f, mask, static_cast<uint8_t>(repeats), {1, 1, 8, 8});
    }
    if (tail) {
        mask[0] = pattern & ((1ULL << tail) - 1);
        Muls(data[repeats * 64], data[repeats * 64], -1.0f, mask, uint8_t(1), {1, 1, 8, 8});
    }
}
__aicore__ inline void MakeSwapIndices(LocalTensor<uint32_t> index, uint32_t indexCount)
{
    CreateVecIndex(index.ReinterpretCast<int32_t>(), 0, indexCount);
    PipeBarrier<PIPE_V>();
    uint64_t even[2] = {0x5555555555555555ULL, 0ULL};
    uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0ULL};
    Adds(
        index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(1), even,
        static_cast<uint8_t>(indexCount / 64), {1, 1, 8, 8});
    Adds(
        index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), int32_t(-1), odd,
        static_cast<uint8_t>(indexCount / 64), {1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, 2U, indexCount);
}

__aicore__ inline void LoadPaddedComplexBatch(
    LocalTensor<float> raw, GlobalTensor<float> source, uint64_t first, uint32_t count, uint32_t cols, uint32_t rows,
    uint32_t ld, uint32_t paddedRows, uint32_t dmaWidth)
{
    if (first < cols) {
        const uint32_t valid = min(count, static_cast<uint32_t>(cols - first));
        DataCopyPad(
            raw, source[first * 2 * ld],
            {static_cast<uint16_t>(valid), rows * 8U, (ld - rows) * 8U, (2 * paddedRows - dmaWidth) / 8U, 0},
            {true, 0, static_cast<uint8_t>(dmaWidth - 2 * rows), 0.0f});
    }
}

__aicore__ inline void RotateComplexBatch(
    LocalTensor<float> raw, LocalTensor<float> other, LocalTensor<uint32_t> index, float imagSign, bool otherImaginary,
    uint32_t elements)
{
    if (imagSign == -1.0f) {
        NegateParity107(raw, true, elements);
        PipeBarrier<PIPE_V>();
    }
    Gather(other, raw, index, 0U, elements);
    PipeBarrier<PIPE_V>();
    NegateParity107(other, otherImaginary, elements);
}

__aicore__ inline void ExpandRealFast107(
    GM_ADDR input, GM_ADDR output, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t paddedRows, uint32_t paddedCols,
    float imagSign, bool otherImaginary = false)
{
    constexpr uint32_t CAP = 8192;
    static_assert(3 * CAP * sizeof(float) <= 192 * 1024);
    const uint32_t batch = min(CAP / (2 * paddedRows), (paddedCols + GetBlockNum() - 1) / GetBlockNum());
    const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * batch;
    if (begin >= paddedCols) {
        return;
    }
    SetMaskNorm();
    const uint32_t used = batch * 2 * paddedRows;
    const uint32_t indexCount = (used + 63) / 64 * 64;
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indices;
    pipe.InitBuffer(values, 2 * CAP * sizeof(float));
    pipe.InitBuffer(indices, CAP * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto other = raw[CAP];
    auto index = indices.Get<uint32_t>();
    MakeSwapIndices(index, indexCount);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(input));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    const uint32_t dmaWidth = (rows * 2 + 7) / 8 * 8;
    for (uint64_t first = begin; first < paddedCols; first += static_cast<uint64_t>(GetBlockNum()) * batch) {
        const uint32_t count = min(batch, static_cast<uint32_t>(paddedCols - first));
        const uint32_t elements = count * 2 * paddedRows;
        Duplicate(raw, 0.0f, elements);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        LoadPaddedComplexBatch(raw, source, first, count, cols, rows, ld, paddedRows, dmaWidth);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        RotateComplexBatch(raw, other, index, imagSign, otherImaginary, elements);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const DataCopyExtParams out{static_cast<uint16_t>(count), paddedRows * 8U, 0, paddedRows * 8U, 0};
        DataCopyPad(dest[first * 4 * paddedRows], raw, out);
        DataCopyPad(dest[(first * 4 + 2) * paddedRows], other, out);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

// Expand several physical columns in one DMA/vector batch. Mode 0 emits R(A),
// mode 1 emits adjacent real/imaginary columns, and mode 2 keeps interleaving.
__aicore__ inline void MakePairSigns(
    LocalTensor<uint32_t> swap, LocalTensor<uint32_t> scratch, LocalTensor<uint32_t> parity, LocalTensor<float> sign,
    LocalTensor<float> negative, uint32_t used)
{
    CreateVecIndex(swap.ReinterpretCast<int32_t>(), 0, used);
    PipeBarrier<PIPE_V>();
    ShiftRight(scratch, swap, 1U, used);
    PipeBarrier<PIPE_V>();
    ShiftLeft(scratch, scratch, 1U, used);
    PipeBarrier<PIPE_V>();
    Sub(parity.ReinterpretCast<int32_t>(), swap.ReinterpretCast<int32_t>(), scratch.ReinterpretCast<int32_t>(), used);
    PipeBarrier<PIPE_V>();
    Muls(parity.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), -2, used);
    PipeBarrier<PIPE_V>();
    Adds(parity.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), 1, used);
    PipeBarrier<PIPE_V>();
    Add(swap.ReinterpretCast<int32_t>(), swap.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), used);
    Cast(sign, parity.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, used);
    PipeBarrier<PIPE_V>();
    ShiftLeft(swap, swap, 2U, used);
    Muls(negative, sign, -1.0f, used);
}

__aicore__ inline void ReadComplexColumnBatch(
    LocalTensor<float> raw, GlobalTensor<float> source, uint64_t first, uint32_t count, uint32_t cols, uint32_t rows,
    uint32_t ld, uint32_t paddedRows, uint32_t dmaWidth)
{
    const uint32_t elements = count * 2 * paddedRows;
    Duplicate(raw, 0.0f, elements);
    Fence();
    if (first < cols) {
        const uint32_t valid = min(count, static_cast<uint32_t>(cols - first));
        DataCopyExtParams cp{
            static_cast<uint16_t>(valid), rows * 8U, (ld - rows) * 8U, (2 * paddedRows - dmaWidth) / 8U, 0};
        DataCopyPad(raw, source[first * 2 * ld], cp, {true, 0, static_cast<uint8_t>(dmaWidth - 2 * rows), 0.0f});
        Fence();
    }
}

struct ColumnExpansionBatch {
    uint32_t rows;
    uint32_t cols;
    uint32_t ld;
    uint32_t paddedRows;
    uint32_t paddedCols;
    uint32_t batch;
    uint32_t mode;
    uint32_t dmaWidth;
    float imagSign;
    LocalTensor<float> raw;
    LocalTensor<float> other;
    LocalTensor<float> planar;
    LocalTensor<float> sign;
    LocalTensor<float> negative;
    LocalTensor<uint32_t> swap;
    LocalTensor<uint32_t> scratch;
    GlobalTensor<float> source;
    GlobalTensor<float> destination;
    static constexpr uint32_t CAP = 4096;
    __aicore__ inline void Store(uint64_t first, uint32_t count, uint32_t elements)
    {
        if (mode == 1) {
            Gather(planar, raw, scratch, 0U, elements / 2);
            Gather(planar[CAP / 2], raw[1], scratch, 0U, elements / 2);
            PipeBarrier<PIPE_V>();
            Muls(planar[CAP / 2], planar[CAP / 2], imagSign, elements / 2);
            Fence();
            DataCopyExtParams cp{static_cast<uint16_t>(count), paddedRows * 4U, 0, paddedRows * 4U, 0};
            DataCopyPad(destination[first * 2 * paddedRows], planar, cp);
            DataCopyPad(destination[(first * 2 + 1) * paddedRows], planar[CAP / 2], cp);
        } else {
            if (imagSign == -1.0f) {
                Mul(raw, raw, sign, elements);
                PipeBarrier<PIPE_V>();
            }
            if (mode == 0) {
                Gather(other, raw, swap, 0U, elements);
                PipeBarrier<PIPE_V>();
                Mul(other, other, negative, elements);
                Fence();
                DataCopyExtParams cp{static_cast<uint16_t>(count), paddedRows * 8U, 0, paddedRows * 8U, 0};
                DataCopyPad(destination[first * 4 * paddedRows], raw, cp);
                DataCopyPad(destination[(first * 4 + 2) * paddedRows], other, cp);
            } else {
                Fence();
                DataCopy(destination[first * 2 * paddedRows], raw, elements);
            }
        }
        Fence();
    }
    __aicore__ inline void Run(uint64_t first)
    {
        const uint32_t count = min(batch, static_cast<uint32_t>(paddedCols - first));
        const uint32_t elements = count * 2 * paddedRows;
        ReadComplexColumnBatch(raw, source, first, count, cols, rows, ld, paddedRows, dmaWidth);
        Store(first, count, elements);
    }
};

__aicore__ inline void ExpandColumns(
    GM_ADDR input, GM_ADDR output, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t paddedRows, uint32_t paddedCols,
    uint32_t mode, float imagSign)
{
    if (mode == 0) {
        ExpandRealFast107(input, output, rows, cols, ld, paddedRows, paddedCols, imagSign);
        return;
    }
    constexpr uint32_t CAP = 4096;
    const uint32_t batch =
        min(CAP / (2 * paddedRows), static_cast<uint32_t>((paddedCols + GetBlockNum() - 1) / GetBlockNum()));
    const uint32_t used = batch * 2 * paddedRows;
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indexBuffer;
    pipe.InitBuffer(values, 5 * CAP * sizeof(float));
    pipe.InitBuffer(indexBuffer, 3 * CAP * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto other = raw[CAP];
    auto planar = raw[2 * CAP];
    auto sign = raw[3 * CAP];
    auto negative = raw[4 * CAP];
    auto swap = indexBuffer.Get<uint32_t>();
    auto scratch = swap[CAP];
    auto parity = swap[2 * CAP];
    if (mode == 0 || (mode == 2 && imagSign == -1.0f)) {
        MakePairSigns(swap, scratch, parity, sign, negative, used);
    }
    if (mode == 1) {
        CreateVecIndex(scratch.ReinterpretCast<int32_t>(), 0, used / 2);
        PipeBarrier<PIPE_V>();
        ShiftLeft(scratch, scratch, 3U, used / 2);
        Fence();
    }
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(input));
    auto destination = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    const uint32_t dmaWidth = (2 * rows + 7) / 8 * 8;
    ColumnExpansionBatch operation{rows, cols,  ld,     paddedRows, paddedCols, batch, mode,    dmaWidth, imagSign,
                                   raw,  other, planar, sign,       negative,   swap,  scratch, source,   destination};
    for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * batch; first < paddedCols;
         first += static_cast<uint64_t>(GetBlockNum()) * batch) {
        operation.Run(first);
    }
}

// Overlap independent columns; queues enforce MTE2 -> V -> MTE3 and buffer reuse.
struct PipelinedColumnExpansion {
    uint32_t rows;
    uint32_t cols;
    uint32_t ld;
    uint32_t paddedRows;
    uint32_t paddedCols;
    uint32_t batch;
    uint32_t dmaWidth;
    uint64_t start;
    uint64_t stride;
    float imagSign;
    event_t zeroReady;
    TQue<QuePosition::VECIN, 2>& inputs;
    TQue<QuePosition::VECOUT, 2>& outputs;
    LocalTensor<float> sign;
    LocalTensor<float> negative;
    LocalTensor<uint32_t> swap;
    GlobalTensor<float> source;
    GlobalTensor<float> dest;
    static constexpr uint32_t CAP = 4096;
    __aicore__ inline void Load(uint32_t step)
    {
        const uint64_t first = start + step * stride;
        const uint32_t count = min(batch, static_cast<uint32_t>(paddedCols - first));
        auto raw = inputs.AllocTensor<float>();
        Duplicate(raw, 0.0f, count * 2 * paddedRows);
        SetFlag<HardEvent::V_MTE2>(zeroReady);
        WaitFlag<HardEvent::V_MTE2>(zeroReady);
        if (first < cols) {
            const uint32_t valid = min(count, static_cast<uint32_t>(cols - first));
            DataCopyExtParams cp{
                static_cast<uint16_t>(valid), rows * 8U, (ld - rows) * 8U, (2 * paddedRows - dmaWidth) / 8U, 0};
            DataCopyPad(raw, source[first * 2 * ld], cp, {true, 0, static_cast<uint8_t>(dmaWidth - 2 * rows), 0.0f});
        }
        inputs.EnQue(raw);
    }
    __aicore__ inline void Compute(uint32_t step)
    {
        const uint64_t first = start + (step - 1) * stride;
        const uint32_t count = min(batch, static_cast<uint32_t>(paddedCols - first));
        const uint32_t elements = count * 2 * paddedRows;
        auto raw = inputs.DeQue<float>();
        auto out = outputs.AllocTensor<float>();
        DataCopy(out, raw, elements);
        PipeBarrier<PIPE_V>();
        inputs.FreeTensor(raw);
        if (imagSign == -1.0f) {
            Mul(out, out, sign, elements);
            PipeBarrier<PIPE_V>();
        }
        Gather(out[CAP], out, swap, 0U, elements);
        PipeBarrier<PIPE_V>();
        Mul(out[CAP], out[CAP], negative, elements);
        outputs.EnQue(out);
    }
    __aicore__ inline void Store(uint32_t step)
    {
        const uint64_t first = start + (step - 2) * stride;
        const uint32_t count = min(batch, static_cast<uint32_t>(paddedCols - first));
        auto out = outputs.DeQue<float>();
        DataCopyExtParams cp{static_cast<uint16_t>(count), paddedRows * 8U, 0, paddedRows * 8U, 0};
        DataCopyPad(dest[first * 4 * paddedRows], out, cp);
        DataCopyPad(dest[(first * 4 + 2) * paddedRows], out[CAP], cp);
        outputs.FreeTensor(out);
    }
};

__aicore__ inline void ExpandRealBlockPipeline(
    GM_ADDR input, GM_ADDR output, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t paddedRows, uint32_t paddedCols,
    float imagSign)
{
    constexpr uint32_t CAP = 4096;
    const uint32_t batch =
        min(CAP / (2 * paddedRows), static_cast<uint32_t>((paddedCols + GetBlockNum() - 1) / GetBlockNum()));
    const uint32_t used = batch * 2 * paddedRows;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputs;
    TQue<QuePosition::VECOUT, 2> outputs;
    TBuf<TPosition::VECCALC> signs;
    TBuf<TPosition::VECCALC> indexBuffer;
    pipe.InitBuffer(inputs, 2, CAP * sizeof(float));
    pipe.InitBuffer(outputs, 2, 2 * CAP * sizeof(float));
    pipe.InitBuffer(signs, 2 * CAP * sizeof(float));
    pipe.InitBuffer(indexBuffer, 3 * CAP * sizeof(uint32_t));
    auto sign = signs.Get<float>();
    auto negative = sign[CAP];
    auto swap = indexBuffer.Get<uint32_t>();
    auto scratch = swap[CAP];
    auto parity = swap[2 * CAP];
    MakePairSigns(swap, scratch, parity, sign, negative, used);
    PipeBarrier<PIPE_V>();
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(input));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    const uint32_t dmaWidth = (2 * rows + 7) / 8 * 8;
    const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * batch;
    const uint64_t stride = static_cast<uint64_t>(GetBlockNum()) * batch;
    const uint32_t iterations =
        start < paddedCols ? static_cast<uint32_t>((paddedCols - start + stride - 1) / stride) : 0;
    const auto zeroReady = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    PipelinedColumnExpansion operation{rows,     cols,  ld,       paddedRows, paddedCols, batch,
                                       dmaWidth, start, stride,   imagSign,   zeroReady,  inputs,
                                       outputs,  sign,  negative, swap,       source,     dest};
    for (uint32_t step = 0; step < iterations + 2; ++step) {
        if (step < iterations) {
            operation.Load(step);
        }
        if (step >= 1 && step - 1 < iterations) {
            operation.Compute(step);
        }
        if (step >= 2) {
            operation.Store(step);
        }
    }
}
__aicore__ inline void ExpandRealBlockAdaptive(
    GM_ADDR input, GM_ADDR output, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t paddedRows, uint32_t paddedCols,
    float imagSign)
{
    const uint32_t batch = 4096 / (2 * paddedRows);
    if (paddedCols > GetBlockNum() * batch * 2) {
        ExpandRealBlockPipeline(input, output, rows, cols, ld, paddedRows, paddedCols, imagSign);
    } else {
        ExpandColumns(input, output, rows, cols, ld, paddedRows, paddedCols, 0, imagSign);
    }
}

__aicore__ inline void ConjugateInterleaved(LocalTensor<float> raw, LocalTensor<float> sign, uint32_t elements)
{
    uint32_t offset = 0;
    while (offset + 64 <= elements) {
        const uint8_t repeats = min((elements - offset) / 64, 255U);
        Mul(raw[offset], raw[offset], sign, uint64_t(64), repeats, {1, 1, 0, 8, 8, 0});
        offset += static_cast<uint32_t>(repeats) * 64;
    }
    if (offset < elements) {
        Mul(raw[offset], raw[offset], sign, static_cast<uint64_t>(elements - offset), 1, {1, 1, 0, 8, 8, 0});
    }
    Fence();
}

template <bool Conjugate>
__aicore__ inline void ExpandInterleavedColumns(
    GM_ADDR input, GM_ADDR output, uint32_t rows, uint32_t cols, uint32_t ld, uint32_t paddedRows, uint32_t paddedCols)
{
    constexpr uint32_t CAP = 32768;
    const uint32_t batch =
        min(CAP / (2 * paddedRows), static_cast<uint32_t>((paddedCols + GetBlockNum() - 1) / GetBlockNum()));
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> signBuffer;
    pipe.InitBuffer(values, CAP * sizeof(float));
    auto raw = values.Get<float>();
    if constexpr (Conjugate) {
        pipe.InitBuffer(signBuffer, 64 * sizeof(float));
        auto bits = signBuffer.Get<uint32_t>();
        CreateVecIndex(bits.ReinterpretCast<int32_t>(), 0, 64);
        PipeBarrier<PIPE_V>();
        ShiftLeft(bits, bits, 31U, 64);
        PipeBarrier<PIPE_V>();
        Adds(bits.ReinterpretCast<int32_t>(), bits.ReinterpretCast<int32_t>(), 1065353216, 64);
        Fence();
    }
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(input));
    auto destination = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    const uint32_t dmaWidth = (2 * rows + 7) / 8 * 8;
    for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * batch; first < paddedCols;
         first += static_cast<uint64_t>(GetBlockNum()) * batch) {
        const uint32_t count = min(batch, static_cast<uint32_t>(paddedCols - first));
        const uint32_t elements = count * 2 * paddedRows;
        ReadComplexColumnBatch(raw, source, first, count, cols, rows, ld, paddedRows, dmaWidth);
        if constexpr (Conjugate) {
            auto sign = signBuffer.Get<float>();
            ConjugateInterleaved(raw, sign, elements);
        }
        DataCopy(destination[first * 2 * paddedRows], raw, elements);
        Fence();
    }
}

__aicore__ inline bool ExpandBatched(GM_ADDR a, GM_ADDR b, GM_ADDR ea, GM_ADDR eb, const CgemmExpandedLayout& t)
{
    const uint32_t ar = t.ta == 0 ? t.m : t.k;
    const uint32_t ac = t.ta == 0 ? t.k : t.m;
    const uint32_t arp = t.ta == 0 ? t.mp : t.kp;
    const uint32_t acp = t.ta == 0 ? t.kp : t.mp;
    const uint32_t brp = t.tb == 0 ? t.kp : t.np;
    if (arp > 2048 || brp > 2048) {
        return false;
    }
    if (t.tb == 0) {
        ExpandColumns(a, ea, ar, ac, t.lda, arp, acp, 0, t.ta == 1 ? -1.0f : 1.0f);
        ExpandColumns(b, eb, t.k, t.n, t.ldb, t.kp, t.np, 2, 1.0f);
    } else {
        ExpandColumns(a, ea, ar, ac, t.lda, arp, acp, t.ta == 0 ? 1 : 2, t.ta == 2 ? 1.0f : -1.0f);
        ExpandColumns(b, eb, t.n, t.k, t.ldb, t.np, t.kp, 0, t.tb == 1 ? -1.0f : 1.0f);
    }
    return true;
}

// R(A) maps complex multiplication to one real GEMM with an interleaved
// reduction: [[ar,-ai],[ai,ar]] per complex element. B remains a complex
// column interpreted as a real vector. R(A)^T represents A^H, so plain
// transpose conjugates A before expansion; conjugate-transpose does not.
struct ComplexExpansionStages {
    CgemmExpandedLayout t;
    uint32_t S;
    LocalTensor<float> raw;
    LocalTensor<float> re;
    LocalTensor<float> im;
    LocalTensor<float> tmp;
    LocalTensor<uint32_t> idx;
    LocalTensor<uint32_t> inter;
    GlobalTensor<float> ag;
    GlobalTensor<float> bg;
    GlobalTensor<float> ax;
    GlobalTensor<float> bx;
    __aicore__ inline void TransposedA()
    {
        uint32_t rowsA = t.ta == 0 ? t.m : t.k;
        uint32_t colsA = t.ta == 0 ? t.k : t.m;
        uint32_t rpA = t.ta == 0 ? t.mp : t.kp;
        uint32_t cpA = t.ta == 0 ? t.kp : t.mp;
        uint32_t tilesA = (rpA + S - 1) / S;
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        for (uint64_t task = GetBlockIdx(); task < static_cast<uint64_t>(tilesA) * cpA; task += GetBlockNum()) {
            uint32_t col = task / tilesA;
            uint32_t row = (task % tilesA) * S;
            uint32_t len = min(S, rpA - row);
            Duplicate(re, 0.0f, 2 * S);
            Fence();
            if (col < colsA && row < rowsA) {
                uint32_t valid = min(len, rowsA - row);
                DataCopyPad(raw, ag[2 * (static_cast<uint64_t>(col) * t.lda + row)], {1, valid * 8U, 0, 0, 0}, pad);
                Fence();
                Gather(re, raw, idx, 0U, valid);
                Gather(im, raw[1], idx, 0U, valid);
                Fence();
                Muls(im, im, t.ta == 2 ? 1.0f : -1.0f, valid);
                Fence();
            }
            if (t.ta == 0) {
                DataCopyPad(ax[static_cast<uint64_t>(2 * col) * t.mp + row], re, {1, len * 4U, 0, 0, 0});
                DataCopyPad(ax[static_cast<uint64_t>(2 * col + 1) * t.mp + row], im, {1, len * 4U, 0, 0, 0});
            } else {
                Gather(raw, re, inter, 0U, 2 * len);
                Fence();
                DataCopyPad(ax[static_cast<uint64_t>(col) * 2 * t.kp + 2 * row], raw, {1, len * 8U, 0, 0, 0});
            }
            Fence();
        }
    }
    __aicore__ inline void TransposedB()
    {
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        uint32_t tilesB = (t.np + S - 1) / S;
        for (uint64_t task = GetBlockIdx(); task < static_cast<uint64_t>(tilesB) * t.kp; task += GetBlockNum()) {
            uint32_t col = task / tilesB;
            uint32_t row = (task % tilesB) * S;
            uint32_t len = min(S, static_cast<uint32_t>(t.np) - row);
            Duplicate(re, 0.0f, 2 * S);
            Fence();
            if (col < static_cast<uint32_t>(t.k) && row < static_cast<uint32_t>(t.n)) {
                uint32_t valid = min(len, static_cast<uint32_t>(t.n) - row);
                DataCopyPad(raw, bg[2 * (static_cast<uint64_t>(col) * t.ldb + row)], {1, valid * 8U, 0, 0, 0}, pad);
                Fence();
                Gather(re, raw, idx, 0U, valid);
                Gather(im, raw[1], idx, 0U, valid);
                Fence();
                if (t.tb == 1) {
                    Muls(im, im, -1.0f, valid);
                    Fence();
                }
            }
            Gather(raw, re, inter, 0U, 2 * len);
            Fence();
            DataCopyPad(bx[static_cast<uint64_t>(2 * col) * 2 * t.np + 2 * row], raw, {1, len * 8U, 0, 0, 0});
            Fence();
            Muls(tmp, im, -1.0f, len);
            DataCopy(tmp[S], re, S);
            Fence();
            Gather(raw, tmp, inter, 0U, 2 * len);
            Fence();
            DataCopyPad(bx[static_cast<uint64_t>(2 * col + 1) * 2 * t.np + 2 * row], raw, {1, len * 8U, 0, 0, 0});
            Fence();
        }
    }
    __aicore__ inline void NormalA()
    {
        uint32_t rows = t.ta == 0 ? t.m : t.k;
        uint32_t cols = t.ta == 0 ? t.k : t.m;
        uint32_t rp = t.ta == 0 ? t.mp : t.kp;
        uint32_t cp = t.ta == 0 ? t.kp : t.mp;
        uint32_t tiles = (rp + S - 1) / S;
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        for (uint64_t task = GetBlockIdx(); task < static_cast<uint64_t>(tiles) * cp; task += GetBlockNum()) {
            uint32_t col = task / tiles;
            uint32_t row = (task % tiles) * S;
            uint32_t len = min(S, rp - row);
            Duplicate(re, 0.0f, 2 * S);
            Fence();
            if (col < cols && row < rows) {
                uint32_t valid = min(len, rows - row);
                DataCopyPad(raw, ag[2 * (static_cast<uint64_t>(col) * t.lda + row)], {1, valid * 8U, 0, 0, 0}, pad);
                Fence();
                Gather(re, raw, idx, 0U, valid);
                Gather(im, raw[1], idx, 0U, valid);
                Fence();
                if (t.ta == 1) {
                    Muls(im, im, -1.0f, valid);
                    Fence();
                }
            }
            Gather(raw, re, inter, 0U, 2 * len);
            Fence();
            DataCopyPad(ax[static_cast<uint64_t>(2 * col) * 2 * rp + 2 * row], raw, {1, len * 8U, 0, 0, 0});
            Fence();
            Muls(tmp, im, -1.0f, len);
            DataCopy(tmp[S], re, S);
            Fence();
            Gather(raw, tmp, inter, 0U, 2 * len);
            Fence();
            DataCopyPad(ax[static_cast<uint64_t>(2 * col + 1) * 2 * rp + 2 * row], raw, {1, len * 8U, 0, 0, 0});
            Fence();
        }
    }
    __aicore__ inline void NormalB()
    {
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        uint32_t tiles = (t.kp + S - 1) / S;
        for (uint64_t task = GetBlockIdx(); task < static_cast<uint64_t>(tiles) * t.np; task += GetBlockNum()) {
            uint32_t col = task / tiles;
            uint32_t row = (task % tiles) * S;
            uint32_t len = min(S, static_cast<uint32_t>(t.kp) - row);
            Duplicate(raw, 0.0f, 2 * S);
            Fence();
            if (col < static_cast<uint32_t>(t.n) && row < static_cast<uint32_t>(t.k)) {
                uint32_t valid = min(len, static_cast<uint32_t>(t.k) - row);
                DataCopyPad(raw, bg[2 * (static_cast<uint64_t>(col) * t.ldb + row)], {1, valid * 8U, 0, 0, 0}, pad);
                Fence();
            }
            DataCopyPad(bx[static_cast<uint64_t>(col) * 2 * t.kp + 2 * row], raw, {1, len * 8U, 0, 0, 0});
            Fence();
        }
    }
};

extern "C" __global__ __aicore__ void cgemm_ex_expand(
    GM_ADDR a, GM_ADDR b, GM_ADDR ea, GM_ADDR eb, CgemmExpandedLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (ExpandBatched(a, b, ea, eb, t)) {
        return;
    }
    const uint32_t S = min(2048U, static_cast<uint32_t>(max(max(t.mp, t.np), t.kp)));
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    TBuf<TPosition::VECCALC> indices;
    pipe.InitBuffer(buffer, 6 * S * sizeof(float));
    pipe.InitBuffer(indices, 5 * S * sizeof(uint32_t));
    auto raw = buffer.Get<float>();
    auto re = raw[2 * S];
    auto im = raw[3 * S];
    auto tmp = raw[4 * S];
    auto idx = indices.Get<uint32_t>();
    auto inter = idx[S];
    MakeIndices(idx, inter, idx[3 * S], S);
    auto ag = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(a));
    auto bg = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(b));
    auto ax = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(ea));
    auto bx = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(eb));
    ComplexExpansionStages stages{t, S, raw, re, im, tmp, idx, inter, ag, bg, ax, bx};
    if (t.tb != 0) {
        stages.TransposedA();
        stages.TransposedB();
    } else {
        stages.NormalA();
        stages.NormalB();
    }
}

template <bool InterleavedA, bool Conjugate, bool Pipeline = false>
__aicore__ inline void RunCompactInterleaved(GM_ADDR a, GM_ADDR b, GM_ADDR ea, GM_ADDR eb, CgemmExpandedLayout t)
{
    const uint32_t ar = t.ta == 0 ? t.m : t.k;
    const uint32_t ac = t.ta == 0 ? t.k : t.m;
    const uint32_t arp = t.ta == 0 ? t.mp : t.kp;
    const uint32_t acp = t.ta == 0 ? t.kp : t.mp;
    if constexpr (InterleavedA) {
        ExpandInterleavedColumns<Conjugate>(a, ea, ar, ac, t.lda, arp, acp);
        if constexpr (Pipeline) {
            ExpandRealBlockPipeline(b, eb, t.n, t.k, t.ldb, t.np, t.kp, t.tb == 1 ? -1.0f : 1.0f);
        } else {
            ExpandColumns(b, eb, t.n, t.k, t.ldb, t.np, t.kp, 0, t.tb == 1 ? -1.0f : 1.0f);
        }
    } else {
        if constexpr (Pipeline) {
            ExpandRealBlockPipeline(a, ea, ar, ac, t.lda, arp, acp, t.ta == 1 ? -1.0f : 1.0f);
        } else {
            ExpandColumns(a, ea, ar, ac, t.lda, arp, acp, 0, t.ta == 1 ? -1.0f : 1.0f);
        }
        ExpandInterleavedColumns<false>(b, eb, t.k, t.n, t.ldb, t.kp, t.np);
    }
}

struct CompactComplexExpand {
    uint32_t mn;
    uint32_t kmp;
    uint32_t npkp;
    int32_t lda;
    int32_t ldb;
    uint32_t flags;
};
static_assert(sizeof(CompactComplexExpand) == 24);

__aicore__ inline void MakeRotatedPairIndices(
    LocalTensor<uint32_t> index, LocalTensor<uint32_t> parity, LocalTensor<float> sign, uint32_t widthA)
{
    CreateVecIndex(index.ReinterpretCast<int32_t>(), 0, widthA);
    PipeBarrier<PIPE_V>();
    ShiftRight(parity, index, 1U, widthA);
    PipeBarrier<PIPE_V>();
    ShiftLeft(parity, parity, 1U, widthA);
    PipeBarrier<PIPE_V>();
    Sub(parity.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), widthA);
    PipeBarrier<PIPE_V>();
    Muls(parity.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), static_cast<int32_t>(-2), widthA);
    PipeBarrier<PIPE_V>();
    Adds(parity.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), static_cast<int32_t>(1), widthA);
    PipeBarrier<PIPE_V>();
    Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), widthA);
    Cast(sign, parity.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, widthA);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, 2U, widthA);
    Muls(sign, sign, -1.0f, widthA);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void LoadSmallNnInputs(
    LocalTensor<float> rawA, LocalTensor<float> rawB, GlobalTensor<float> a, GlobalTensor<float> b,
    CompactComplexExpand q, uint32_t core, uint32_t k, uint32_t m, uint32_t first, uint32_t valid, uint32_t kp)
{
    if (core < k) {
        const uint32_t dma = (2 * m + 7) / 8 * 8;
        DataCopyPad(
            rawA, a[static_cast<uint64_t>(core) * 2 * q.lda], {1, m * 8U, 0, 0, 0},
            {true, 0, static_cast<uint8_t>(dma - 2 * m), 0.0f});
    }
    if (valid) {
        const uint32_t dma = (2 * k + 7) / 8 * 8;
        DataCopyPad(
            rawB, b[static_cast<uint64_t>(first) * 2 * q.ldb],
            {static_cast<uint16_t>(valid), k * 8U, static_cast<uint32_t>(q.ldb - k) * 8U, (2 * kp - dma) / 8U, 0},
            {true, 0, static_cast<uint8_t>(dma - 2 * k), 0.0f});
    }
}

__aicore__ inline void FinishSmallNnExpansion(
    LocalTensor<float> rawA, LocalTensor<float> rotatedA, LocalTensor<float> rawB, LocalTensor<uint32_t> index,
    LocalTensor<uint32_t> parity, LocalTensor<float> sign, GlobalTensor<float> ea, GlobalTensor<float> eb,
    uint32_t core, uint32_t k, uint32_t kp, uint32_t mp, uint32_t widthA, uint32_t first, uint32_t count)
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    // These index/sign buffers are disjoint from both in-flight input buffers.
    if (core < k) {
        MakeRotatedPairIndices(index, parity, sign, widthA);
    }
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (core < k) {
        Gather(rotatedA, rawA, index, 0U, widthA);
        PipeBarrier<PIPE_V>();
        Mul(rotatedA, rotatedA, sign, widthA);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if (core < kp) {
        DataCopy(ea[static_cast<uint64_t>(core) * 4 * mp], rawA, widthA);
        DataCopy(ea[(static_cast<uint64_t>(core) * 4 + 2) * mp], rotatedA, widthA);
    }
    if (count) {
        DataCopy(eb[static_cast<uint64_t>(first) * 2 * kp], rawB, count * 2 * kp);
    }
}

extern "C" __global__ __aicore__ void cgemm_ex_expand_small_nn(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR eaGm, GM_ADDR ebGm, CompactComplexExpand q)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SetMaskNorm();
    constexpr uint32_t CAP = 4096;
    const uint32_t m = q.mn & 65535U;
    const uint32_t n = q.mn >> 16;
    const uint32_t k = q.kmp & 65535U;
    const uint32_t mp = q.kmp >> 16;
    const uint32_t np = q.npkp & 65535U;
    const uint32_t kp = q.npkp >> 16;
    const uint32_t core = GetBlockIdx();
    const uint32_t widthA = 2 * mp;
    const uint32_t per = (np + GetBlockNum() - 1) / GetBlockNum();
    const uint32_t first = core * per;
    const uint32_t count = first < np ? min(per, np - first) : 0;
    const uint32_t valid = first < n ? min(count, n - first) : 0;
    LocalTensor<float> rawA(TPosition::VECCALC, 0, CAP);
    LocalTensor<float> rotatedA(TPosition::VECCALC, CAP * 4, CAP);
    LocalTensor<float> rawB(TPosition::VECCALC, CAP * 8, CAP);
    LocalTensor<float> sign(TPosition::VECCALC, CAP * 20, CAP);
    LocalTensor<uint32_t> index(TPosition::VECCALC, CAP * 12, CAP);
    LocalTensor<uint32_t> parity(TPosition::VECCALC, CAP * 16, CAP);
    auto a = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(aGm));
    auto b = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(bGm));
    auto ea = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(eaGm));
    auto eb = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(ebGm));
    if (core < kp) {
        Duplicate(rawA, 0.0f, widthA);
        Duplicate(rotatedA, 0.0f, widthA);
    }
    if (count) {
        Duplicate(rawB, 0.0f, count * 2 * kp);
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    LoadSmallNnInputs(rawA, rawB, a, b, q, core, k, m, first, valid, kp);
    FinishSmallNnExpansion(rawA, rotatedA, rawB, index, parity, sign, ea, eb, core, k, kp, mp, widthA, first, count);
    PipeBarrier<PIPE_ALL>();
    SetMaskNorm();
}

template <bool IA, bool CJ, bool PP = false>
__global__ __aicore__ void cgemm_ex_expand_interleaved(
    GM_ADDR a, GM_ADDR b, GM_ADDR ea, GM_ADDR eb, CgemmExpandedLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunCompactInterleaved<IA, CJ, PP>(a, b, ea, eb, t);
}
template <bool IA, bool CJ, bool PP = false>
__global__ __aicore__ void cgemm_ex_expand_compact(GM_ADDR a, GM_ADDR b, GM_ADDR ea, GM_ADDR eb, CompactComplexExpand q)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemmExpandedLayout t{
        static_cast<int32_t>(q.mn & 65535),
        static_cast<int32_t>(q.mn >> 16),
        static_cast<int32_t>(q.kmp & 65535),
        static_cast<int32_t>(q.kmp >> 16),
        static_cast<int32_t>(q.npkp & 65535),
        static_cast<int32_t>(q.npkp >> 16),
        q.lda,
        q.ldb,
        static_cast<int32_t>(q.flags & 3),
        static_cast<int32_t>((q.flags >> 2) & 3),
        0};
    RunCompactInterleaved<IA, CJ, PP>(a, b, ea, eb, t);
}
__aicore__ inline void MakeCanonicalIndices(
    LocalTensor<uint32_t> index, LocalTensor<int32_t> column, LocalTensor<int32_t> rem, LocalTensor<int32_t> reduction,
    LocalTensor<int32_t> parity, LocalTensor<float> raw, LocalTensor<float> sign, CgemmExpandedLayout t, uint32_t batch,
    uint32_t used)
{
    CreateVecIndex(index.ReinterpretCast<int32_t>(), 0, used);
    PipeBarrier<PIPE_V>();
    Cast(raw, index.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, used);
    PipeBarrier<PIPE_V>();
    Adds(raw, raw, 0.5f, used);
    PipeBarrier<PIPE_V>();
    Muls(raw, raw, 1.0f / static_cast<float>(2 * t.kp), used);
    PipeBarrier<PIPE_V>();
    Cast(column, raw, RoundMode::CAST_FLOOR, used);
    PipeBarrier<PIPE_V>();
    Muls(rem, column, 2 * t.kp, used);
    PipeBarrier<PIPE_V>();
    Sub(rem, index.ReinterpretCast<int32_t>(), rem, used);
    PipeBarrier<PIPE_V>();
    ShiftRight(reduction.ReinterpretCast<uint32_t>(), rem.ReinterpretCast<uint32_t>(), 1U, used);
    PipeBarrier<PIPE_V>();
    Muls(parity, reduction, 2, used);
    PipeBarrier<PIPE_V>();
    Sub(parity, rem, parity, used);
    Muls(reduction, reduction, static_cast<int32_t>(2 * batch + 8), used);
    Muls(column, column, 2, used);
    PipeBarrier<PIPE_V>();
    Add(index.ReinterpretCast<int32_t>(), reduction, column, used);
    PipeBarrier<PIPE_V>();
    Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), parity, used);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, 2U, used);
    if (t.tb == 2) {
        Cast(sign, parity, RoundMode::CAST_NONE, used);
        PipeBarrier<PIPE_V>();
        Muls(sign, sign, -2.0f, used);
        PipeBarrier<PIPE_V>();
        Adds(sign, sign, 1.0f, used);
    }
}

__aicore__ inline void ExpandCanonicalB(GM_ADDR input, GM_ADDR output, const CgemmExpandedLayout& t)
{
    if (t.kp % 64 == 0 && t.kp <= 512) {
        ExpandCanonicalTiled107(input, output, t);
        return;
    }
    constexpr uint32_t CAP = 4096;
    const uint32_t batch = min(32U, (CAP / t.kp - 8U) / 2U) / 4U * 4U;
    const uint32_t used = batch * 2U * t.kp;
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indexBuffer;
    pipe.InitBuffer(values, 3 * CAP * sizeof(float));
    pipe.InitBuffer(indexBuffer, 5 * CAP * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto out = raw[CAP];
    auto sign = raw[2 * CAP];
    auto index = indexBuffer.Get<uint32_t>();
    auto column = index[CAP].ReinterpretCast<int32_t>();
    auto rem = index[2 * CAP].ReinterpretCast<int32_t>();
    auto reduction = index[3 * CAP].ReinterpretCast<int32_t>();
    auto parity = index[4 * CAP].ReinterpretCast<int32_t>();
    MakeCanonicalIndices(index, column, rem, reduction, parity, raw, sign, t, batch, used);
    // raw held vector scratch above; order its reuse by MTE2 explicitly.
    Fence();
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(input));
    auto destination = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(output));
    for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * batch; first < t.np;
         first += static_cast<uint64_t>(GetBlockNum()) * batch) {
        const uint32_t count = min(batch, static_cast<uint32_t>(t.np - first));
        Duplicate(raw, 0.0f, t.kp * (2 * batch + 8));
        Fence();
        if (first < t.n) {
            const uint32_t valid = min(count, static_cast<uint32_t>(t.n - first));
            const uint32_t dmaWidth = (2 * valid + 7) / 8 * 8;
            DataCopyExtParams cp{
                static_cast<uint16_t>(t.k), valid * 8U, static_cast<uint32_t>(t.ldb - valid) * 8U,
                (2 * batch + 8 - dmaWidth) / 8U, 0};
            DataCopyPad(raw, source[2 * first], cp, {true, 0, static_cast<uint8_t>(dmaWidth - 2 * valid), 0.0f});
            Fence();
        }
        Gather(out, raw, index, 0U, count * 2 * t.kp);
        PipeBarrier<PIPE_V>();
        if (t.tb == 2) {
            Mul(out, out, sign, count * 2 * t.kp);
        }
        Fence();
        DataCopy(destination[first * 2 * t.kp], out, count * 2 * t.kp);
        Fence();
    }
}

extern "C" __global__ __aicore__ void cgemm_ex_expand_canonical(
    GM_ADDR a, GM_ADDR b, GM_ADDR ea, GM_ADDR eb, CgemmExpandedLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t ar = t.ta == 0 ? t.m : t.k;
    const uint32_t ac = t.ta == 0 ? t.k : t.m;
    const uint32_t arp = t.ta == 0 ? t.mp : t.kp;
    const uint32_t acp = t.ta == 0 ? t.kp : t.mp;
    ExpandColumns(a, ea, ar, ac, t.lda, arp, acp, 0, t.ta == 1 ? -1.0f : 1.0f);
    ExpandCanonicalB(b, eb, t);
}

template <bool IA, bool CJ, bool Pipeline>
void LaunchCompactExpansion(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* ea, uint8_t* eb, CgemmExpandedLayout t,
    CompactComplexExpand q, bool compact)
{
    if (compact) {
        cgemm_ex_expand_compact<IA, CJ, Pipeline><<<blocks, nullptr, stream>>>(a, b, ea, eb, q);
    } else {
        cgemm_ex_expand_interleaved<IA, CJ, Pipeline><<<blocks, nullptr, stream>>>(a, b, ea, eb, t);
    }
}
template <bool IA, bool CJ>
void LaunchExpansionPipeline(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* ea, uint8_t* eb, CgemmExpandedLayout t,
    CompactComplexExpand q, bool compact, bool pipeline)
{
    if (pipeline) {
        LaunchCompactExpansion<IA, CJ, true>(blocks, stream, a, b, ea, eb, t, q, compact);
    } else {
        LaunchCompactExpansion<IA, CJ, false>(blocks, stream, a, b, ea, eb, t, q, compact);
    }
}
bool IsCompactExpansion(const CgemmExpandedLayout& t)
{
    return t.m <= 65535 && t.n <= 65535 && t.k <= 65535 && t.mp <= 65535 && t.np <= 65535 && t.kp <= 65535;
}
bool IsSmallNnExpansion(const CgemmExpandedLayout& t, uint32_t blocks, bool compact)
{
    if (blocks == 0) {
        return false;
    }
    return compact && t.ta == 0 && t.tb == 0 && t.k <= 32 && static_cast<uint32_t>(t.kp) <= blocks &&
           2U * static_cast<uint32_t>(t.mp) <= 4096U &&
           static_cast<uint64_t>((static_cast<uint32_t>(t.np) + blocks - 1) / blocks) * 2U * t.kp <= 4096U;
}
void cgemm_ex_expand_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* ea, uint8_t* eb, CgemmExpandedLayout t)
{
    // The public host validates nonempty dimensions and available vector cores.
    if (blocks == 0 || t.mp <= 0 || t.np <= 0 || t.kp <= 0) {
        return;
    }
    const bool compact = IsCompactExpansion(t);
    const CompactComplexExpand q{
        static_cast<uint32_t>(t.m) | (static_cast<uint32_t>(t.n) << 16),
        static_cast<uint32_t>(t.k) | (static_cast<uint32_t>(t.mp) << 16),
        static_cast<uint32_t>(t.np) | (static_cast<uint32_t>(t.kp) << 16),
        t.lda,
        t.ldb,
        static_cast<uint32_t>(t.ta) | (static_cast<uint32_t>(t.tb) << 2)};
    if (IsSmallNnExpansion(t, blocks, compact)) {
        cgemm_ex_expand_small_nn<<<blocks, nullptr, stream>>>(a, b, ea, eb, q);
        return;
    }
    const int arp = t.ta == 0 ? t.mp : t.kp;
    const int brp = t.tb == 0 ? t.kp : t.np;
    const uint32_t mode0Rows = t.tb == 0 ? arp : t.np;
    const uint32_t mode0Cols = t.tb == 0 ? (t.ta == 0 ? t.kp : t.mp) : t.kp;
    const bool pipeline = mode0Rows <= 2048 && mode0Cols > blocks * (4096 / (2 * mode0Rows)) * 2;
    const bool batched = arp <= 2048 && brp <= 2048;
    if (t.canonicalB) {
        cgemm_ex_expand_canonical<<<blocks, nullptr, stream>>>(a, b, ea, eb, t);
    } else if (batched && t.tb == 0) {
        LaunchExpansionPipeline<false, false>(blocks, stream, a, b, ea, eb, t, q, compact, pipeline);
    } else if (batched && t.ta == 1) {
        LaunchExpansionPipeline<true, true>(blocks, stream, a, b, ea, eb, t, q, compact, pipeline);
    } else if (batched && t.ta == 2) {
        LaunchExpansionPipeline<true, false>(blocks, stream, a, b, ea, eb, t, q, compact, pipeline);
    } else {
        cgemm_ex_expand<<<blocks, nullptr, stream>>>(a, b, ea, eb, t);
    }
}

extern "C" __global__ __aicore__ void cgemm_ex_rawb114_kernel(GM_ADDR input, GM_ADDR output, CgemmExpandedLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ExpandRealFast107(input, output, t.n, t.k, t.ldb, t.np, t.kp, t.tb == 1 ? -1.0f : 1.0f, t.ta == 1);
}

void cgemm_ex_rawb114_do(uint32_t blocks, void* stream, uint8_t* b, uint8_t* bx, CgemmExpandedLayout t)
{
    cgemm_ex_rawb114_kernel<<<blocks, nullptr, stream>>>(b, bx, t);
}

// NN alternative: raw A is a2M-by-K real matrix. Only small B is planarized
// into K-by-2N. The real GEMM produces all four products, the finish forms C.
extern "C" __global__ __aicore__ void cgemm_ex_planarb115_kernel(GM_ADDR bGm, GM_ADDR bxGm, CgemmExpandedLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetMaskNorm();
    constexpr uint32_t CAP = 8192;
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indices;
    pipe.InitBuffer(values, 2 * CAP * sizeof(float));
    pipe.InitBuffer(indices, CAP / 2 * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto real = raw[CAP];
    auto imag = real[CAP / 2];
    auto index = indices.Get<uint32_t>();
    const uint32_t batch =
        min(CAP / (2 * static_cast<uint32_t>(t.kp)), (static_cast<uint32_t>(t.np) + GetBlockNum() - 1) / GetBlockNum());
    const uint32_t capacity = batch * static_cast<uint32_t>(t.kp);
    const uint32_t dma = (2 * static_cast<uint32_t>(t.k) + 7) / 8 * 8;
    CreateVecIndex(index.ReinterpretCast<int32_t>(), 0, capacity);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, 3U, capacity);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(bGm));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(bxGm));
    for (uint32_t first = GetBlockIdx() * batch; first < t.np; first += GetBlockNum() * batch) {
        const uint32_t count = min(batch, static_cast<uint32_t>(t.np) - first);
        const uint32_t used = count * static_cast<uint32_t>(t.kp);
        Duplicate(raw, 0.0f, 2 * used);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        LoadPaddedComplexBatch(raw, source, first, count, t.n, t.k, t.ldb, t.kp, dma);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Gather(real, raw, index, 0U, used);
        Gather(imag, raw[1], index, 0U, used);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const DataCopyExtParams out{
            static_cast<uint16_t>(count), static_cast<uint32_t>(t.kp) * 4U, 0, static_cast<uint32_t>(t.kp) * 4U, 0};
        DataCopyPad(dest[static_cast<uint64_t>(2 * first) * t.kp], real, out);
        DataCopyPad(dest[static_cast<uint64_t>(2 * first + 1) * t.kp], imag, out);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}
extern "C" __global__ __aicore__ void cgemm_ex_cross115_finish(GM_ADDR productGm, GM_ADDR outputGm, CgemmExLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetMaskNorm();
    constexpr uint32_t CAP = 16384;
    const uint32_t first = 2 * GetBlockIdx();
    if (first >= t.n) {
        return;
    }
    const uint32_t count = min(2U, static_cast<uint32_t>(t.n) - first);
    const uint32_t ld = 2 * static_cast<uint32_t>(t.packedLd);
    const uint32_t width = 2 * static_cast<uint32_t>(t.m);
    TPipe pipe;
    TBuf<TPosition::VECCALC> values;
    TBuf<TPosition::VECCALC> indices;
    pipe.InitBuffer(values, 2 * CAP * sizeof(float));
    pipe.InitBuffer(indices, 4096 * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto out = raw[CAP];
    auto swapped = out[8192];
    auto index = indices.Get<uint32_t>();
    const uint32_t vectorCount = (ld + 63) / 64 * 64;
    MakeSwapIndices(index, vectorCount);
    auto product = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(productGm));
    auto output = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outputGm));
    DataCopy(raw, product[static_cast<uint64_t>(2 * first) * ld], 2 * count * ld);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t col = 0; col < count; ++col) {
        Gather(swapped, raw[(2 * col + 1) * ld], index, 0U, ld);
        PipeBarrier<PIPE_V>();
        NegateParity107(swapped, false, ld);
        PipeBarrier<PIPE_V>();
        Add(out[col * ld], raw[2 * col * ld], swapped, ld);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const uint32_t dma = (width + 7) / 8 * 8;
    DataCopyPad(
        output[static_cast<uint64_t>(first) * 2 * t.ldc], out,
        {static_cast<uint16_t>(count), width * 4U, (ld - dma) / 8U, static_cast<uint32_t>(t.ldc - t.m) * 8U, 0});
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
void cgemm_ex_cross115_do(
    uint32_t cubes, uint32_t vectors, void* stream, uint8_t* a, uint8_t* b, uint8_t* bx, uint8_t* product,
    uint8_t* output, CgemmExpandedLayout t, CgemmExLayout finish)
{
    cgemm_ex_planarb115_kernel<<<vectors, nullptr, stream>>>(b, bx, t);
    GemmExTiling real{};
    real.m = 2 * t.m;
    real.n = 2 * t.n;
    real.k = t.k;
    real.lda = 2 * t.lda;
    real.ldb = t.kp;
    real.ldc = 2 * t.mp;
    real.transA = 0;
    real.transB = 0;
    real.alpha = 1.0f;
    real.beta = 0.0f;
    cgemm_ex_cross_compute_do(cubes, stream, a, bx, product, real);
    cgemm_ex_cross115_finish<<<(t.n + 1) / 2, nullptr, stream>>>(product, output, finish);
}

// Dedicated short complex NN compute. Preserve padded input/output layout;
// consume the entire <=64 real K range in one L0 loading step.
namespace cgemm_short347 {
using namespace matmul;
using Fp32 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulShapeParams shape{112, 384, 64, 112, 192, 64};
constexpr MatmulFuncParams func{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                true,  true};
constexpr MatmulBiasParams bias{false};
constexpr MatmulConfig config = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shape, func, bias);
constexpr MatmulApiStaticTiling staticTiling = GetMatmulApiTiling<Fp32, Fp32, Fp32, Fp32>(config);
using Compute = MatmulImpl<Fp32, Fp32, Fp32, Fp32, staticTiling>;
} // namespace cgemm_short347
extern "C" __global__ __aicore__ __cube__ void cgemm_short347_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GemmExTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetHF32Mode(HF32Mode::DISABLE);
    const uint32_t nt = (t.n + 111) / 112;
    const uint32_t idx = GetBlockIdx();
    const int nb = (idx % nt) * 112;
    const int mb = (idx / nt) * 384;
    const int nr = min(112, t.n - nb);
    const int mc = min(384, t.m - mb);
    auto left = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(bGm) + static_cast<uint64_t>(nb) * t.ldb);
    auto right = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(aGm) + mb);
    auto out = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(cGm) + static_cast<uint64_t>(nb) * t.ldc + mb);
    TPipe pipe;
    cgemm_short347::Compute mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetOrgShape(nr, t.lda, t.k, t.ldb, t.ldc);
    mm.SetSingleShape(nr, mc, t.k);
    mm.SetTensorA(left, false);
    mm.SetTensorB(right, false);
    mm.IterateAll(out, 0);
    mm.End();
}
void cgemm_short347_do(uint32_t, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    const uint32_t grid = ((t.m + 383) / 384) * ((t.n + 111) / 112);
    cgemm_short347_kernel<<<grid, nullptr, stream>>>(a, b, c, t);
}
