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
#include "lib/matmul_intf.h"
#include "lt_matmul_kernel.h"
using namespace AscendC;
using namespace matmul;
namespace {
constexpr MatmulFuncParams F{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                             true,  true};
constexpr MatmulBiasParams B{false};
template <int M, int N>
constexpr MatmulShapeParams shape{M, N, 512, M, N, N == 128 ? 64U : 32U};
template <int M, int N>
constexpr auto config = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shape<M, N>, F, B);
template <bool T, bool NZ>
using I = MatmulType<TPosition::GM, NZ ? CubeFormat::NZ : CubeFormat::ND, float, T>;
using O = MatmulType<TPosition::GM, CubeFormat::ND, float>;
template <bool TA, bool TB, bool ANZ, int M, int N>
constexpr auto tiling = GetMatmulApiTiling<I<TA, ANZ>, I<TB, !ANZ>, O, O>(config<M, N>);
__aicore__ inline uint64_t Offset(uint32_t rows, uint32_t row, uint32_t col)
{
    const uint64_t h = (uint64_t(rows) + 15) / 16 * 16;
    return uint64_t(col / 8) * h * 8 + uint64_t(row) * 8 + col % 8;
}
template <bool TA, bool TB, bool ANZ, int M, int N>
__aicore__ inline void Cube(GM_ADDR a, GM_ADDR b, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    MatmulImpl<I<TA, ANZ>, I<TB, !ANZ>, O, O, tiling<TA, TB, ANZ, M, N>> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> ag, bg, dg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    dg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    const uint32_t cols = (t.n + N - 1) / N, rows = (t.m + M - 1) / M;
    mm.SetOrgShape(
        ANZ ? t.m : (TA ? t.lda : t.m), ANZ ? (TB ? t.n : t.ldb) : t.n, ANZ ? t.k : (TA ? t.k : t.lda),
        ANZ ? (TB ? t.ldb : t.k) : t.k, t.productLd);
    for (uint32_t job = GetBlockIdx(); job < rows * cols; job += GetBlockNum()) {
        const uint32_t row = job / cols * M, col = job % cols * N;
        mm.SetSingleShape(t.m - row < M ? t.m - row : M, t.n - col < N ? t.n - col : N, t.k);
        const uint64_t ao = ANZ ? (TA ? Offset(t.k, 0, row) : Offset(t.m, row, 0)) : (TA ? row : uint64_t(row) * t.lda);
        const uint64_t bo = ANZ ? (TB ? uint64_t(col) * t.ldb : col) : (TB ? Offset(t.n, col, 0) : Offset(t.k, 0, col));
        mm.SetTensorA(ag[ao], TA);
        mm.SetTensorB(bg[bo], TB);
        mm.IterateAll(dg[uint64_t(row) * t.productLd + col], false);
        mm.End();
    }
}
template <bool ANZ, int M, int N>
__aicore__ inline void Dispatch(GM_ADDR a, GM_ADDR b, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    if (t.transA) {
        if (t.transB)
            Cube<true, true, ANZ, M, N>(a, b, d, t, pipe);
        else
            Cube<true, false, ANZ, M, N>(a, b, d, t, pipe);
    } else if (t.transB)
        Cube<false, true, ANZ, M, N>(a, b, d, t, pipe);
    else
        Cube<false, false, ANZ, M, N>(a, b, d, t, pipe);
}
template <bool ANZ>
__aicore__ inline void Tile(GM_ADDR a, GM_ADDR b, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    if (t.cubeTile == 32)
        Dispatch<ANZ, 32, 128>(a, b, d, t, pipe);
    else if (t.cubeTile == 48)
        Dispatch<ANZ, 48, 96>(a, b, d, t, pipe);
    else
        Dispatch<ANZ, 64, 128>(a, b, d, t, pipe);
}
} // namespace
extern "C" __global__ __aicore__ __cube__ void lt_matmul22_mixed_nz_cube(
    GM_ADDR a, GM_ADDR b, GM_ADDR d, LtMatmul22Tiling t, uint32_t matrix)
{
    TPipe pipe;
    if (matrix == 0)
        Tile<true>(a, b, d, t, pipe);
    else
        Tile<false>(a, b, d, t, pipe);
}
void LtMatmul22MixedNZCube(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* d, LtMatmul22Tiling t, uint32_t matrix)
{
    lt_matmul22_mixed_nz_cube<<<blocks, nullptr, stream>>>(a, b, d, t, matrix);
}
