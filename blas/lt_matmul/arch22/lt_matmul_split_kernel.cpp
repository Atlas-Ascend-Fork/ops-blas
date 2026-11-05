/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Experimental deterministic multi-core split-K. Existing H0/H1/H2 are unchanged.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lt_matmul_kernel.h"
using namespace AscendC;
using namespace matmul;
namespace {
constexpr MatmulFuncParams F{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                             true,  true};
constexpr MatmulBiasParams B{false};
template <uint32_t M, uint32_t N>
constexpr MatmulShapeParams shape{M, N, 512, M, N, M == 32 ? 32U : 64U};
template <uint32_t M, uint32_t N>
constexpr auto config = GetMMConfig < M == 32 ? MatmulConfigMode::CONFIG_NORM :
                                                MatmulConfigMode::CONFIG_MDL > (shape<M, N>, F, B);
template <bool T>
using I = MatmulType<TPosition::GM, CubeFormat::ND, float, T>;
using O = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>;
template <bool TA, bool TB, uint32_t M, uint32_t N>
constexpr auto tiling = GetMatmulApiTiling<I<TA>, I<TB>, O, O>(config<M, N>);
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
template <HardEvent E>
__aicore__ inline void Sync()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}
template <bool TA, bool TB, uint32_t M, uint32_t N>
__aicore__ inline void Cube(GM_ADDR a, GM_ADDR b, GM_ADDR p, const LtMatmul22Tiling& t, uint32_t splits, TPipe& pipe)
{
    MatmulImpl<I<TA>, I<TB>, O, O, tiling<TA, TB, M, N>, CB> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> ag, bg, pg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    pg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p));
    const uint32_t rows = (t.m + M - 1) / M, cols = (t.n + N - 1) / N;
    // 8-element K boundaries retain natural FP32 Cube input block alignment.
    const uint32_t chunk = (t.k + 8 * splits - 1) / (8 * splits) * 8;
    for (uint32_t job = GetBlockIdx(); job < rows * cols * splits; job += GetBlockNum()) {
        const uint32_t s = job / (rows * cols), tile = job % (rows * cols);
        const uint32_t row = tile / cols * M, col = tile % cols * N, start = s * chunk;
        if (start >= t.k)
            continue;
        mm.SetOrgShape(TA ? t.lda : t.m, TB ? t.n : t.ldb, TA ? t.k : t.lda, TB ? t.ldb : t.k, t.productLd);
        mm.SetSingleShape(Min(M, t.m - row), Min(N, t.n - col), Min(chunk, t.k - start));
        const uint64_t ao = TA ? uint64_t(start) * t.lda + row : uint64_t(row) * t.lda + start;
        const uint64_t bo = TB ? uint64_t(col) * t.ldb + start : uint64_t(start) * t.ldb + col;
        mm.SetTensorA(ag[ao], TA);
        mm.SetTensorB(bg[bo], TB);
        mm.IterateAll(pg[(uint64_t(s) * t.m + row) * t.productLd + col], false);
        mm.End();
    }
}
template <uint32_t M, uint32_t N>
__aicore__ inline void Dispatch(
    GM_ADDR a, GM_ADDR b, GM_ADDR p, const LtMatmul22Tiling& t, uint32_t splits, TPipe& pipe)
{
    if (t.transA) {
        if (t.transB)
            Cube<true, true, M, N>(a, b, p, t, splits, pipe);
        else
            Cube<true, false, M, N>(a, b, p, t, splits, pipe);
    } else if (t.transB)
        Cube<false, true, M, N>(a, b, p, t, splits, pipe);
    else
        Cube<false, false, M, N>(a, b, p, t, splits, pipe);
}
__aicore__ inline void Read(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint64_t off, uint32_t rows, uint32_t n, uint32_t ld)
{
    const uint64_t gap = uint64_t(ld - n) * 4;
    const uint32_t pitch = (n + 7) / 8 * 8;
    const DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(pitch - n), 0.0f};
    if (gap <= 0xffffffffULL) {
        const DataCopyExtParams copy{static_cast<uint16_t>(rows), n * 4, static_cast<uint32_t>(gap), 0, 0};
        DataCopyPad(dst, src[off], copy, pad);
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            const DataCopyExtParams copy{1, n * 4, 0, 0, 0};
            DataCopyPad(dst[r * pitch], src[off + uint64_t(r) * ld], copy, pad);
        }
    }
}
__aicore__ inline void Reduce(GM_ADDR p, GM_ADDR c, GM_ADDR d, const LtMatmul22Tiling& t, uint32_t splits, TPipe& pipe)
{
    // N<=128: three 8x128 FP32 planes, total UB=12288 bytes.
    TBuf<TPosition::VECCALC> ab, bb, cb;
    pipe.InitBuffer(ab, 4096);
    pipe.InitBuffer(bb, 4096);
    pipe.InitBuffer(cb, 4096);
    auto sum = ab.Get<float>(), tmp = bb.Get<float>(), old = cb.Get<float>();
    GlobalTensor<float> pg, cg, dg;
    pg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    dg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    const uint32_t chunk = (t.k + 8 * splits - 1) / (8 * splits) * 8;
    const uint32_t active = (t.k + chunk - 1) / chunk;
    for (uint32_t row = GetBlockIdx() * 8; row < t.m; row += GetBlockNum() * 8) {
        const uint32_t rows = Min(8, t.m - row), count = rows * ((t.n + 7) / 8 * 8);
        Read(sum, pg, uint64_t(row) * t.productLd, rows, t.n, t.productLd);
        Sync<HardEvent::MTE2_V>();
        for (uint32_t s = 1; s < active; ++s) {
            Read(tmp, pg, (uint64_t(s) * t.m + row) * t.productLd, rows, t.n, t.productLd);
            Sync<HardEvent::MTE2_V>();
            Add(sum, sum, tmp, count);
            Sync<HardEvent::V_MTE2>();
        }
        if (t.alpha != 1.0f) {
            Muls(sum, sum, t.alpha, count);
            PipeBarrier<PIPE_V>();
        }
        if (t.beta != 0.0f) {
            Read(old, cg, uint64_t(row) * t.ldc, rows, t.n, t.ldc);
            Sync<HardEvent::MTE2_V>();
            Muls(old, old, t.beta, count);
            PipeBarrier<PIPE_V>();
            Add(sum, sum, old, count);
        }
        Sync<HardEvent::V_MTE3>();
        const uint64_t gap = uint64_t(t.ldd - t.n) * 4;
        if (gap <= 0xffffffffULL) {
            const DataCopyExtParams copy{static_cast<uint16_t>(rows), t.n * 4, 0, static_cast<uint32_t>(gap), 0};
            DataCopyPad(dg[uint64_t(row) * t.ldd], sum, copy);
        } else {
            for (uint32_t r = 0; r < rows; ++r) {
                const DataCopyExtParams copy{1, t.n * 4, 0, 0, 0};
                DataCopyPad(dg[uint64_t(row + r) * t.ldd], sum[r * ((t.n + 7) / 8 * 8)], copy);
            }
        }
        Sync<HardEvent::MTE3_MTE2>();
    }
}
} // namespace
extern "C" __global__ __aicore__ __cube__ void lt_matmul22_split_cube(
    GM_ADDR a, GM_ADDR b, GM_ADDR p, LtMatmul22Tiling t, uint32_t splits)
{
    TPipe pipe;
    if (t.cubeTile == 32)
        Dispatch<32, 64>(a, b, p, t, splits, pipe);
    else
        Dispatch<64, 128>(a, b, p, t, splits, pipe);
}
extern "C" __global__ __aicore__ __vector__ void lt_matmul22_split_reduce(
    GM_ADDR p, GM_ADDR c, GM_ADDR d, LtMatmul22Tiling t, uint32_t splits)
{
    TPipe pipe;
    Reduce(p, c, d, t, splits, pipe);
}
void LtMatmul22SplitCube(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* p, LtMatmul22Tiling t, uint32_t splits)
{
    lt_matmul22_split_cube<<<blocks, nullptr, stream>>>(a, b, p, t, splits);
}
void LtMatmul22SplitReduce(
    uint32_t blocks, void* stream, uint8_t* p, uint8_t* c, uint8_t* d, LtMatmul22Tiling t, uint32_t splits)
{
    lt_matmul22_split_reduce<<<blocks, nullptr, stream>>>(p, c, d, t, splits);
}
