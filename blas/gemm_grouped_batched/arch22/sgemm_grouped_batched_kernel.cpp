/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Ascend C arch22 correctness path. No arch35 MicroAPI dependencies.
#include "kernel_operator.h"
#include "sgemm_gb_packet.h"
using namespace AscendC;
namespace {
class VectorGemm {
public:
    static constexpr uint32_t kColumns = 8;
    __aicore__ inline void Init(bool transposeA, bool transposeB, TPipe* pipe)
    {
        pipe->InitBuffer(aBuf_, sgemm_gb::kVectorRows * sizeof(float));
        pipe->InitBuffer(cBuf_, kColumns * sgemm_gb::kVectorRows * sizeof(float));
        pipe->InitBuffer(tBuf_, sgemm_gb::kVectorRows * sizeof(float));
        pipe->InitBuffer(bBuf_, kColumns * sgemm_gb::kVectorRows * sizeof(float));
        if (transposeA || transposeB) {
            pipe->InitBuffer(stridedBuf_, sgemm_gb::kVectorRows * 32);
            pipe->InitBuffer(offsetBuf_, sgemm_gb::kVectorRows * sizeof(uint32_t));
            auto offsets = offsetBuf_.Get<uint32_t>();
            for (uint32_t r = 0; r < sgemm_gb::kVectorRows; ++r)
                offsets.SetValue(r, r * 32);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Run(const sgemm_gb::Packet& p)
    {
        const auto& g = p.g;
        const uint64_t rows = (uint64_t(g.m) + sgemm_gb::kVectorRows - 1) / sgemm_gb::kVectorRows;
        // Retain single-column tasks when a wider tile would reduce active cores.
        const uint32_t columns = rows * g.n * p.count >= uint64_t(GetBlockNum()) * kColumns ? kColumns : 1;
        const uint64_t perBatch = rows * ((uint64_t(g.n) + columns - 1) / columns);
        const uint64_t total = perBatch * p.count;
        for (uint64_t t = GetBlockIdx(); t < total; t += GetBlockNum()) {
            const uint32_t batch = static_cast<uint32_t>(t / perBatch);
            const uint64_t within = t % perBatch;
            const uint32_t col = static_cast<uint32_t>(within / rows) * columns;
            const uint32_t row = static_cast<uint32_t>(within % rows) * sgemm_gb::kVectorRows;
            const uint32_t len = g.m - row < sgemm_gb::kVectorRows ? g.m - row : sgemm_gb::kVectorRows;
            const uint32_t width = g.n - col < columns ? g.n - col : columns;
            a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.a[batch]));
            b_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.b[batch]));
            c_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.c[batch]));
            Tile(g, row, col, len, width);
        }
    }

private:
    __aicore__ inline void Load(LocalTensor<float>& dst, GlobalTensor<float>& src, uint64_t offset, uint32_t len)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(dst, src[offset], cp, pad);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadStrided(
        LocalTensor<float>& dst, GlobalTensor<float>& src, uint64_t offset, uint64_t stride, uint32_t len)
    {
        auto strided = stridedBuf_.Get<float>();
        auto offsets = offsetBuf_.Get<uint32_t>();
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        if (stride <= UINT32_MAX) {
            DataCopyExtParams cp{static_cast<uint16_t>(len), sizeof(float), static_cast<uint32_t>(stride), 0, 0};
            DataCopyPad(strided, src[offset], cp, pad);
        } else {
            // DMA stride is 32-bit. Keep 64-bit GM offsets and load each value
            // into a separate aligned UB block before the common gather.
            DataCopyExtParams cp{1, sizeof(float), 0, 0, 0};
            const uint64_t step = stride / sizeof(float) + 1;
            for (uint32_t r = 0; r < len; ++r)
                DataCopyPad(strided[r * 8], src[offset + uint64_t(r) * step], cp, pad);
        }
        PipeBarrier<PIPE_ALL>();
        Gather(dst, strided, offsets, 0, len);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void InitOutput(
        const sgemm_gb::Group& g, uint32_t row, uint32_t col, uint32_t len, uint32_t width, bool scaleOnly,
        LocalTensor<float>& allC)
    {
        for (uint32_t c = 0; c < width; ++c) {
            auto cv = allC[c * sgemm_gb::kVectorRows];
            if ((scaleOnly || !g.ta) && g.beta != 0.0f) {
                Load(cv, c_, uint64_t(col + c) * g.ldc + row, len);
                if (g.beta != 1.0f)
                    Muls(cv, cv, g.beta, len);
            } else {
                Duplicate(cv, 0.0f, len);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void CacheB(
        const sgemm_gb::Group& g, uint32_t col, uint32_t k, uint32_t width, uint64_t bStride, LocalTensor<float>& allB)
    {
        const uint32_t remaining = g.k - k;
        const uint32_t count = remaining < sgemm_gb::kVectorRows ? remaining : sgemm_gb::kVectorRows;
        for (uint32_t c = 0; c < width; ++c) {
            auto cachedB = allB[c * sgemm_gb::kVectorRows];
            if (g.tb)
                LoadStrided(cachedB, b_, uint64_t(k) * g.ldb + col + c, bStride, count);
            else
                Load(cachedB, b_, uint64_t(col + c) * g.ldb + k, count);
        }
    }
    __aicore__ inline void LoadA(
        const sgemm_gb::Group& g, uint32_t row, uint32_t k, uint32_t len, uint64_t aStride, LocalTensor<float>& av)
    {
        if (!g.ta) {
            Load(av, a_, uint64_t(k) * g.lda + row, len);
        } else {
            LoadStrided(av, a_, uint64_t(row) * g.lda + k, aStride, len);
        }
    }
    __aicore__ inline void Accumulate(
        const sgemm_gb::Group& g, uint32_t width, uint32_t localK, uint32_t len, LocalTensor<float>& av,
        LocalTensor<float>& allB, LocalTensor<float>& allC, LocalTensor<float>& tmp)
    {
        for (uint32_t c = 0; c < width; ++c) {
            auto cv = allC[c * sgemm_gb::kVectorRows];
            const float bv = allB.GetValue(c * sgemm_gb::kVectorRows + localK);
            Duplicate(tmp, g.ta ? bv : g.alpha * bv, len);
            PipeBarrier<PIPE_ALL>();
            MulAddDst(cv, av, tmp, len);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void RunK(
        const sgemm_gb::Group& g, uint32_t row, uint32_t col, uint32_t len, uint32_t width, LocalTensor<float>& av,
        LocalTensor<float>& allB, LocalTensor<float>& allC, LocalTensor<float>& tmp)
    {
        const uint64_t aStride = (uint64_t(g.lda) - 1) * sizeof(float);
        const uint64_t bStride = (uint64_t(g.ldb) - 1) * sizeof(float);
        for (uint32_t k = 0; k < g.k; ++k) {
            const uint32_t localK = k % sgemm_gb::kVectorRows;
            if (localK == 0)
                CacheB(g, col, k, width, bStride, allB);
            // Each A tile is shared by independent output columns. Every output
            // retains the original ascending-K FP32 accumulation order.
            LoadA(g, row, k, len, aStride, av);
            Accumulate(g, width, localK, len, av, allB, allC, tmp);
        }
    }
    __aicore__ inline void StoreOutput(
        const sgemm_gb::Group& g, uint32_t row, uint32_t col, uint32_t len, uint32_t width, bool scaleOnly,
        LocalTensor<float>& allC, LocalTensor<float>& tmp)
    {
        for (uint32_t c = 0; c < width; ++c) {
            auto cv = allC[c * sgemm_gb::kVectorRows];
            const uint64_t cbase = uint64_t(col + c) * g.ldc + row;
            if (!scaleOnly && g.ta) {
                Muls(cv, cv, g.alpha, len);
                PipeBarrier<PIPE_ALL>();
                if (g.beta != 0.0f) {
                    Load(tmp, c_, cbase, len);
                    Muls(tmp, tmp, g.beta, len);
                    PipeBarrier<PIPE_ALL>();
                    Add(cv, cv, tmp, len);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
            DataCopyPad(c_[cbase], cv, cp);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Tile(const sgemm_gb::Group& g, uint32_t row, uint32_t col, uint32_t len, uint32_t width)
    {
        auto av = aBuf_.Get<float>();
        auto allC = cBuf_.Get<float>();
        auto allB = bBuf_.Get<float>();
        auto tmp = tBuf_.Get<float>();
        const bool scaleOnly = g.k == 0 || g.alpha == 0.0f;
        if (scaleOnly && g.beta == 1.0f)
            return;
        InitOutput(g, row, col, len, width, scaleOnly, allC);
        if (!scaleOnly)
            RunK(g, row, col, len, width, av, allB, allC, tmp);
        StoreOutput(g, row, col, len, width, scaleOnly, allC, tmp);
    }
    TBuf<TPosition::VECCALC> aBuf_, cBuf_, tBuf_, bBuf_;
    TBuf<TPosition::VECCALC> stridedBuf_, offsetBuf_;
    GlobalTensor<float> a_, b_, c_;
};
} // namespace
extern "C" __global__ __aicore__ void sgemm_gb_vector(sgemm_gb::Packet packet)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    VectorGemm op;
    op.Init(packet.g.ta != 0, packet.g.tb != 0, &pipe);
    op.Run(packet);
}
void sgemm_gb_vector_do(sgemm_gb::Packet packet, uint32_t blocks, void* stream)
{
    sgemm_gb_vector<<<blocks, nullptr, stream>>>(packet);
}

// One AIV block computes one complete 8x8 matrix. The pointer-list and
// strided packets differ only in the three matrix addresses.
namespace {
__aicore__ inline uint64_t Small8A(const sgemm_gb::Small8Packet& p, uint32_t i) { return p.a[i]; }
__aicore__ inline uint64_t Small8B(const sgemm_gb::Small8Packet& p, uint32_t i) { return p.b[i]; }
__aicore__ inline uint64_t Small8C(const sgemm_gb::Small8Packet& p, uint32_t i) { return p.c[i]; }
__aicore__ inline uint64_t Small8A(const sgemm_gb::Small8StridedPacket& p, uint32_t i)
{
    return p.baseA + uint64_t(i) * p.strideA;
}
__aicore__ inline uint64_t Small8B(const sgemm_gb::Small8StridedPacket& p, uint32_t i)
{
    return p.baseB + uint64_t(i) * p.strideB;
}
__aicore__ inline uint64_t Small8C(const sgemm_gb::Small8StridedPacket& p, uint32_t i)
{
    return p.baseC + uint64_t(i) * p.strideC;
}
__aicore__ inline void Small8Multiply(LocalTensor<float>& av, LocalTensor<float>& bv, LocalTensor<float>& cv)
{
#pragma unroll
    for (uint32_t col = 0; col < 8; ++col) {
#pragma unroll
        for (uint32_t row = 0; row < 8; ++row) {
            float sum = 0.0f;
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k)
                sum += av.GetValue(k * 8 + row) * bv.GetValue(col * 8 + k);
            cv.SetValue(col * 8 + row, sum); // beta=0: never read C.
        }
    }
}
template <class P>
__aicore__ inline void Small8Run(const P& packet)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> aBuf, bBuf, cBuf;
    pipe.InitBuffer(aBuf, 64 * sizeof(float));
    pipe.InitBuffer(bBuf, 64 * sizeof(float));
    pipe.InitBuffer(cBuf, 64 * sizeof(float));
    auto av = aBuf.Get<float>();
    auto bv = bBuf.Get<float>();
    auto cv = cBuf.Get<float>();
    for (uint32_t batch = GetBlockIdx(); batch < packet.count; batch += GetBlockNum()) {
        GlobalTensor<float> a, b, c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Small8A(packet, batch)));
        b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Small8B(packet, batch)));
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Small8C(packet, batch)));
        DataCopyExtParams cp{1, 64 * sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(av, a[0], cp, pad);
        DataCopyPad(bv, b[0], cp, pad);
        PipeBarrier<PIPE_ALL>();
        Small8Multiply(av, bv, cv);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(c[0], cv, cp);
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace
extern "C" __global__ __aicore__ void sgemm_gb_small8(sgemm_gb::Small8Packet packet)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Small8Run(packet);
}
void sgemm_gb_small8_do(sgemm_gb::Small8Packet packet, uint32_t blocks, void* stream)
{
    sgemm_gb_small8<<<blocks, nullptr, stream>>>(packet);
}
extern "C" __global__ __aicore__ void sgemm_gb_small8_strided(sgemm_gb::Small8StridedPacket packet)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Small8Run(packet);
}
void sgemm_gb_small8_strided_do(sgemm_gb::Small8StridedPacket packet, uint32_t blocks, void* stream)
{
    sgemm_gb_small8_strided<<<blocks, nullptr, stream>>>(packet);
}
