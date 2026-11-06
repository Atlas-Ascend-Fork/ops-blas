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
#include "sgemm_ex_direct_kernel.h"
using namespace AscendC;
namespace {
// Combine the rounded product and its residual with the prior accumulator.
__aicore__ inline void AccumulateProduct(
    LocalTensor<float> result, LocalTensor<float> p, LocalTensor<float> e, LocalTensor<float> q, LocalTensor<float> sum,
    LocalTensor<float> z, uint32_t len)
{
    Add(sum, p, result, len);
    PipeBarrier<PIPE_ALL>();
    Sub(z, sum, p, len);
    PipeBarrier<PIPE_ALL>();
    Sub(q, sum, z, len);
    Sub(z, result, z, len);
    PipeBarrier<PIPE_ALL>();
    Sub(q, p, q, len);
    PipeBarrier<PIPE_ALL>();
    Add(q, q, z, len);
    PipeBarrier<PIPE_ALL>();
    Add(e, e, q, len);
    PipeBarrier<PIPE_ALL>();
    Add(result, sum, e, len);
    PipeBarrier<PIPE_ALL>();
}
// Emulate a single-rounded FP32 multiply-add with error-free product/sum
// transforms on vector units whose multiply-add rounds the product first.
__aicore__ inline void AccurateUpdate(
    LocalTensor<float> result, LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float> w, uint32_t len)
{
    constexpr uint32_t T = 128;
    auto ah = w;
    auto al = w[T];
    auto bh = w[2 * T];
    auto bl = w[3 * T];
    auto p = w[4 * T];
    auto e = w[5 * T];
    auto q = w[6 * T];
    auto sum = w[7 * T];
    auto z = w[8 * T];
    Muls(ah, a, 4097.0f, len);
    Muls(bh, b, 4097.0f, len);
    PipeBarrier<PIPE_ALL>();
    Sub(al, ah, a, len);
    Sub(bl, bh, b, len);
    PipeBarrier<PIPE_ALL>();
    Sub(ah, ah, al, len);
    Sub(bh, bh, bl, len);
    PipeBarrier<PIPE_ALL>();
    Sub(al, a, ah, len);
    Sub(bl, b, bh, len);
    Mul(p, a, b, len);
    PipeBarrier<PIPE_ALL>();
    Mul(e, ah, bh, len);
    PipeBarrier<PIPE_ALL>();
    Sub(e, e, p, len);
    Mul(q, ah, bl, len);
    PipeBarrier<PIPE_ALL>();
    Add(e, e, q, len);
    Mul(q, al, bh, len);
    PipeBarrier<PIPE_ALL>();
    Add(e, e, q, len);
    Mul(q, al, bl, len);
    PipeBarrier<PIPE_ALL>();
    Add(e, e, q, len);
    AccumulateProduct(result, p, e, q, sum, z, len);
}
} // namespace
namespace {
struct DirectRealTile {
    SgemmExDirectLayout t;
    LocalTensor<float> av;
    LocalTensor<float> result;
    LocalTensor<float> old;
    LocalTensor<float> scalar;
    LocalTensor<float> ca;
    LocalTensor<float> cb;
    LocalTensor<uint32_t> indices;
    GlobalTensor<float> ag;
    GlobalTensor<float> bg;
    GlobalTensor<float> cg;
    bool cached;
    static constexpr uint32_t TILE = 128;
    __aicore__ inline void LoadA(uint64_t row, int l, uint32_t len)
    {
        DataCopyExtParams cp{1, len * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        if (cached) {
            Gather(av, ca, indices, static_cast<uint32_t>(l * (t.ta ? 1 : t.lda) * 4), len);
        } else if (t.ta == 0) {
            DataCopyPad(av, ag[static_cast<uint64_t>(l) * t.lda + row], cp, pad);
        } else {
            for (uint32_t i = 0; i < len; ++i) {
                av.SetValue(i, ag.GetValue((row + i) * t.lda + l));
            }
        }
    }
    __aicore__ inline void Accumulate(uint64_t row, uint64_t col, uint32_t len)
    {
        for (int l = 0; l < t.k; ++l) {
            uint64_t bp = t.tb == 0 ? col * t.ldb + l : static_cast<uint64_t>(l) * t.ldb + col;
            float bv = cached ? cb.GetValue(bp) : bg.GetValue(bp);
            if (t.ta == 0 && bv == 0.0f) {
                continue;
            }
            LoadA(row, l, len);
            PipeBarrier<PIPE_ALL>();
            Duplicate(scalar, t.ta == 0 ? t.alpha * bv : bv, len);
            PipeBarrier<PIPE_ALL>();
            if (t.alpha != 1.0f || t.k > 32) {
                AccurateUpdate(result, av, scalar, av[4 * TILE], len);
            } else {
                FusedMulAdd(av, scalar, result, len);
                PipeBarrier<PIPE_ALL>();
                DataCopy(result, av, TILE);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Finish(uint64_t pos, uint32_t len)
    {
        DataCopyExtParams cp{1, len * 4U, 0, 0, 0};
        if (t.ta != 0) {
            Muls(result, result, t.alpha, len);
            PipeBarrier<PIPE_ALL>();
            if (t.beta != 0.0f) {
                Axpy(result, old, t.beta, len);
                PipeBarrier<PIPE_ALL>();
            }
        }
        DataCopyPad(cg[pos], result, cp);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Run(uint64_t row, uint64_t col, uint64_t pos, uint32_t len)
    {
        for (uint32_t i = 0; cached && i < len; ++i) {
            indices.SetValue(i, static_cast<uint32_t>((t.ta ? (row + i) * t.lda : row + i) * 4));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cp{1, len * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        Duplicate(result, 0.0f, TILE);
        PipeBarrier<PIPE_ALL>();
        if (t.beta != 0.0f) {
            DataCopyPad(old, cg[pos], cp, pad);
            PipeBarrier<PIPE_ALL>();
        }
        // Netlib scales B before the row update for non-transposed A.
        if (t.ta == 0 && t.beta != 0.0f) {
            Muls(result, old, t.beta, len);
            PipeBarrier<PIPE_ALL>();
        }
        if (t.alpha != 0.0f) {
            Accumulate(row, col, len);
        }
        Finish(pos, len);
    }
};
} // namespace
extern "C" __global__ __aicore__ void sgemm_ex_direct(GM_ADDR a, GM_ADDR b, GM_ADDR c, SgemmExDirectLayout t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t TILE = 128;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    TBuf<TPosition::VECCALC> cacheBuffer;
    TBuf<TPosition::VECCALC> indexBuffer;
    pipe.InitBuffer(buffer, 13 * TILE * sizeof(float));
    pipe.InitBuffer(cacheBuffer, 8192 * sizeof(float));
    pipe.InitBuffer(indexBuffer, TILE * sizeof(uint32_t));
    auto indices = indexBuffer.Get<uint32_t>();
    auto av = buffer.Get<float>();
    auto result = av[TILE];
    auto old = av[2 * TILE];
    auto scalar = av[3 * TILE];
    GlobalTensor<float> ag;
    GlobalTensor<float> bg;
    GlobalTensor<float> cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    const uint64_t na = static_cast<uint64_t>(t.lda) * (t.ta ? t.m : t.k);
    const uint64_t nb = static_cast<uint64_t>(t.ldb) * (t.tb ? t.k : t.n);
    const uint32_t offset = static_cast<uint32_t>((na + 7) / 8 * 8);
    const bool cached = t.alpha != 0.0f && t.k > 0 && na + nb + 16 <= 8192;
    auto ca = cacheBuffer.Get<float>();
    auto cb = ca[offset < 8192 ? offset : 0];
    if (cached) {
        DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        DataCopyPad(ca, ag, {1, static_cast<uint32_t>(na) * 4U, 0, 0, 0}, padding);
        DataCopyPad(cb, bg, {1, static_cast<uint32_t>(nb) * 4U, 0, 0, 0}, padding);
        PipeBarrier<PIPE_ALL>();
    }
    DirectRealTile tile{t, av, result, old, scalar, ca, cb, indices, ag, bg, cg, cached};
    uint64_t tiles = (static_cast<uint64_t>(t.m) + TILE - 1) / TILE;
    for (uint64_t task = GetBlockIdx(); task < tiles * t.n; task += GetBlockNum()) {
        const uint64_t col = task / tiles;
        const uint64_t row = (task % tiles) * TILE;
        const uint64_t pos = col * t.ldc + row;
        const uint32_t len = static_cast<uint32_t>(t.m - row < TILE ? t.m - row : TILE);
        tile.Run(row, col, pos, len);
    }
}

void sgemm_ex_direct_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, SgemmExDirectLayout t)
{
    sgemm_ex_direct<<<blocks, nullptr, stream>>>(a, b, c, t);
}
