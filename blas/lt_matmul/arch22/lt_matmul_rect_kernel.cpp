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
#include "lt_matmul_tiling.h"

using namespace AscendC;
using namespace matmul;

namespace {
template <bool T>
using Input = MatmulType<TPosition::GM, CubeFormat::ND, float, T>;
using Output = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulFuncParams F{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                             true,  true};
constexpr MatmulBiasParams B{false};
template <uint32_t M, uint32_t N, uint32_t K>
constexpr MatmulShapeParams SHAPE{M, N, K, M, N, 32U};
template <uint32_t M, uint32_t N, uint32_t K>
constexpr auto CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(SHAPE<M, N, K>, F, B);
template <bool TA, bool TB, uint32_t M, uint32_t N, uint32_t K>
constexpr auto TILING = GetMatmulApiTiling<Input<TA>, Input<TB>, Output, Output>(CONFIG<M, N, K>);
template <bool TA, bool TB, uint32_t M, uint32_t N, uint32_t K>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    MatmulImpl<Input<TA>, Input<TB>, Output, Output, TILING<TA, TB, M, N, K>> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> ag, bg, dg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    dg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    const uint32_t nc = (t.n + N - 1U) / N, nr = (t.m + M - 1U) / M;
    for (uint64_t tile = GetBlockIdx(); tile < static_cast<uint64_t>(nc) * nr; tile += GetBlockNum()) {
        const uint32_t row = tile / nc * M, col = tile % nc * N;
        mm.SetOrgShape(TA ? t.lda : t.m, TB ? t.n : t.ldb, TA ? t.k : t.lda, TB ? t.ldb : t.k, t.productLd);
        mm.SetSingleShape(t.m - row < M ? t.m - row : M, t.n - col < N ? t.n - col : N, t.k);
        mm.SetTensorA(ag[TA ? row : static_cast<uint64_t>(row) * t.lda], TA);
        mm.SetTensorB(bg[TB ? static_cast<uint64_t>(col) * t.ldb : col], TB);
        mm.IterateAll(dg[static_cast<uint64_t>(row) * t.productLd + col], false);
        mm.End();
    }
}
template <uint32_t M, uint32_t N, uint32_t K>
__aicore__ inline void Dispatch(GM_ADDR a, GM_ADDR b, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    if (t.transA) {
        if (t.transB)
            Run<true, true, M, N, K>(a, b, d, t, pipe);
        else
            Run<true, false, M, N, K>(a, b, d, t, pipe);
    } else {
        if (t.transB)
            Run<false, true, M, N, K>(a, b, d, t, pipe);
        else
            Run<false, false, M, N, K>(a, b, d, t, pipe);
    }
}
} // namespace
extern "C" __global__ __aicore__ __cube__ void lt_matmul22_rect_short(
    GM_ADDR a, GM_ADDR b, GM_ADDR d, LtMatmul22Tiling t)
{
    TPipe pipe;
    if (t.cubeTile == 160U)
        Run<true, true, 160U, 128U, 96U>(a, b, d, t, pipe);
    else
        Run<false, false, 128U, 192U, 160U>(a, b, d, t, pipe);
}
void LtMatmul22RectShort(uint32_t groups, void* stream, uint8_t* a, uint8_t* b, uint8_t* d, LtMatmul22Tiling t)
{
    lt_matmul22_rect_short<<<groups, nullptr, stream>>>(a, b, d, t);
}

namespace {
// Preserve the native path for aligned widths. For an N tail, copy full
// 32-column NZ groups as ND batches and finish the exact tail separately.
// On A2 srcNdStride is measured in1024B; dstNdStride is in float elements.
__aicore__ inline void Columns32Out(
    const __gm__ void* gm, const LocalTensor<int8_t>& src, const void* opaque, uint64_t, uint64_t)
{
    const auto& p = *reinterpret_cast<const DataCopyOutParams*>(opaque);
    const uint16_t m = p.burstLen / 2U;
    const uint16_t stride = (m + 15U) / 16U * 16U;
    const auto floats = src.ReinterpretCast<float>();
    auto* dst = (__gm__ float*)gm;
    auto* cc = (__cc__ float*)floats.GetPhyAddr();
    const uint8_t unit = static_cast<uint8_t>(p.enUnitFlag ? 3U : 0U);
    if (p.oriNSize % 16U == 0U) {
        SetFixpipeNz2ndFlag(1U, 0U, 0U);
        copy_matrix_cc_to_gm(
            dst, cc, 0, p.oriNSize, m, p.dstStride, stride, unit, QuantMode_t::NoQuant, uint8_t(0), false, true);
        return;
    }
    const uint16_t count = p.oriNSize / 32U;
    const uint16_t head = count * 32U;
    if (count != 0U) {
        SetFixpipeNz2ndFlag(count, 2U * (stride / 16U), 32U);
        copy_matrix_cc_to_gm(
            dst, cc, 0, 32U, m, p.dstStride, stride, unit, QuantMode_t::NoQuant, uint8_t(0), false, true);
    }
    if (head != p.oriNSize) {
        if (count != 0U)
            PipeBarrier<PIPE_FIX>();
        SetFixpipeNz2ndFlag(1U, 0U, 0U);
        copy_matrix_cc_to_gm(
            dst + head, cc + static_cast<uint64_t>(head) * stride, 0, p.oriNSize - head, m, p.dstStride, stride, unit,
            QuantMode_t::NoQuant, uint8_t(0), false, true);
    }
}

template <bool TRANS>
using RowInput = MatmulType<TPosition::GM, CubeFormat::ND, float, TRANS>;
using RowOutput = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulFuncParams ROW_FUNC{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                    true,  true};
constexpr MatmulBiasParams ROW_BIAS{false};
template <uint32_t M>
constexpr MatmulShapeParams ROW_SHAPE{M, 160U, 128U, M, 160U, 64U};
template <uint32_t M>
constexpr auto ROW_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(ROW_SHAPE<M>, ROW_FUNC, ROW_BIAS);
template <bool TA, bool TB, uint32_t M>
constexpr auto ROW_TILING = GetMatmulApiTiling<RowInput<TA>, RowInput<TB>, RowOutput, RowOutput>(ROW_CONFIG<M>);

template <bool TA, bool TB, uint32_t M>
__aicore__ inline void Rows(
    GM_ADDR a, GM_ADDR b, GM_ADDR product, const LtMatmul22Tiling& t, uint32_t first, uint32_t rows, TPipe& pipe)
{
    MatmulImpl<
        RowInput<TA>, RowInput<TB>, RowOutput, RowOutput, ROW_TILING<TA, TB, M>,
        MatmulCallBackFunc<Columns32Out, nullptr, nullptr>>
        mm;
    mm.SetSubBlockIdx(0U);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> ag, bg, pg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    pg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(product));
    mm.SetOrgShape(TA ? t.lda : t.m, TB ? t.n : t.ldb, TA ? t.k : t.lda, TB ? t.ldb : t.k, t.productLd);
    mm.SetSingleShape(rows, t.n, t.k);
    mm.SetTensorA(ag[TA ? first : static_cast<uint64_t>(first) * t.lda], TA);
    mm.SetTensorB(bg, TB);
    mm.IterateAll(pg[static_cast<uint64_t>(first) * t.productLd], false);
    mm.End();
}

template <uint32_t M>
__aicore__ inline void TransposeRows(
    GM_ADDR a, GM_ADDR b, GM_ADDR product, const LtMatmul22Tiling& t, uint32_t first, uint32_t rows, TPipe& pipe)
{
    if (t.transA) {
        if (t.transB)
            Rows<true, true, M>(a, b, product, t, first, rows, pipe);
        else
            Rows<true, false, M>(a, b, product, t, first, rows, pipe);
    } else {
        if (t.transB)
            Rows<false, true, M>(a, b, product, t, first, rows, pipe);
        else
            Rows<false, false, M>(a, b, product, t, first, rows, pipe);
    }
}
} // namespace

extern "C" __global__ __aicore__ __cube__ void lt_matmul22_cube_rows160(
    GM_ADDR a, GM_ADDR b, GM_ADDR product, LtMatmul22Tiling t)
{
    const uint64_t chunks = (static_cast<uint64_t>(t.m) + 7U) / 8U;
    const uint32_t first = chunks * GetBlockIdx() / GetBlockNum() * 8U;
    const uint32_t last = chunks * (GetBlockIdx() + 1U) / GetBlockNum() * 8U;
    const uint32_t end = last < t.m ? last : t.m;
    if (first >= end)
        return;
    TPipe pipe;
    // The host bounds every owned row range by the H5 capsule's 96 rows.
    // SetSingleShape retains the actual extent, including short final bands.
    TransposeRows<96>(a, b, product, t, first, end - first, pipe);
}

void LtMatmul22Rows160(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* product, LtMatmul22Tiling t)
{
    lt_matmul22_cube_rows160<<<blocks, nullptr, stream>>>(a, b, product, t);
}

namespace {
template <bool TRANS>
using NZInput = MatmulType<TPosition::GM, CubeFormat::NZ, float, TRANS>;
using NDOutput = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulFuncParams NZ_FUNC{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                   true,  true};
constexpr MatmulBiasParams NZ_BIAS{false};
template <uint32_t M, uint32_t N, uint32_t KC, uint32_t BK = (N == 128U ? 64U : 32U)>
constexpr MatmulShapeParams NZ_SHAPE{M, N, KC, M, N, BK};
template <uint32_t M, uint32_t N, uint32_t KC, uint32_t BK = (N == 128U ? 64U : 32U)>
constexpr MatmulConfig NZ_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(NZ_SHAPE<M, N, KC, BK>, NZ_FUNC, NZ_BIAS);
template <bool TA, bool TB, uint32_t M, uint32_t N, uint32_t KC, uint32_t BK = (N == 128U ? 64U : 32U)>
constexpr MatmulApiStaticTiling NZ_TILING =
    GetMatmulApiTiling<NZInput<TA>, NZInput<TB>, NDOutput, NDOutput>(NZ_CONFIG<M, N, KC, BK>);
template <bool TA, bool TB, uint32_t M, uint32_t N, uint32_t KC, uint32_t BK = (N == 128U ? 64U : 32U)>
using NZMatmul = MatmulImpl<NZInput<TA>, NZInput<TB>, NDOutput, NDOutput, NZ_TILING<TA, TB, M, N, KC, BK>>;

__aicore__ inline uint64_t NZOffset(uint32_t physicalRows, uint32_t row, uint32_t column)
{
    const uint64_t height = (static_cast<uint64_t>(physicalRows) + 15U) / 16U * 16U;
    return static_cast<uint64_t>(column / 8U) * height * 8U + static_cast<uint64_t>(row) * 8U + column % 8U;
}

template <bool TA, bool TB, uint32_t M, uint32_t N, uint32_t KC, uint32_t BK = (N == 128U ? 64U : 32U)>
__aicore__ inline void CubeNZ(GM_ADDR a, GM_ADDR b, GM_ADDR product, const LtMatmul22Tiling& t, TPipe& pipe)
{
    NZMatmul<TA, TB, M, N, KC, BK> mm;
    mm.SetSubBlockIdx(0U);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> aGm, bGm, productGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    productGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(product));
    const uint64_t columns = (static_cast<uint64_t>(t.n) + N - 1U) / N;
    const uint64_t rows = (static_cast<uint64_t>(t.m) + M - 1U) / M;
    // Both inputs are NZ: original logical shapes, rather than original ND
    // row pitches, determine global NZ column-slab strides inside the library.
    mm.SetOrgShape(t.m, t.n, t.k, t.k, t.productLd);
    for (uint64_t tile = GetBlockIdx(); tile < rows * columns; tile += GetBlockNum()) {
        const uint32_t row = static_cast<uint32_t>(tile / columns * M);
        const uint32_t column = static_cast<uint32_t>(tile % columns * N);
        const uint32_t actualM = t.m - row < M ? t.m - row : M;
        const uint32_t actualN = t.n - column < N ? t.n - column : N;
        const uint64_t aOffset = TA ? NZOffset(t.k, 0U, row) : NZOffset(t.m, row, 0U);
        const uint64_t bOffset = TB ? NZOffset(t.n, column, 0U) : NZOffset(t.k, 0U, column);
        mm.SetSingleShape(actualM, actualN, t.k);
        mm.SetTensorA(aGm[aOffset], TA);
        mm.SetTensorB(bGm[bOffset], TB);
        mm.IterateAll(productGm[static_cast<uint64_t>(row) * t.productLd + column], false);
        mm.End();
    }
}

template <uint32_t M, uint32_t N, uint32_t KC, uint32_t BK = (N == 128U ? 64U : 32U)>
__aicore__ inline void DispatchNZ(GM_ADDR a, GM_ADDR b, GM_ADDR product, const LtMatmul22Tiling& t, TPipe& pipe)
{
    if (t.transA) {
        if (t.transB)
            CubeNZ<true, true, M, N, KC, BK>(a, b, product, t, pipe);
        else
            CubeNZ<true, false, M, N, KC, BK>(a, b, product, t, pipe);
    } else {
        if (t.transB)
            CubeNZ<false, true, M, N, KC, BK>(a, b, product, t, pipe);
        else
            CubeNZ<false, false, M, N, KC, BK>(a, b, product, t, pipe);
    }
}
template <uint32_t KC>
__aicore__ inline void DispatchNZTile(GM_ADDR a, GM_ADDR b, GM_ADDR product, const LtMatmul22Tiling& t, TPipe& pipe)
{
    if (t.cubeTile == 32U)
        DispatchNZ<32, 64, KC>(a, b, product, t, pipe);
    else if (t.cubeTile == 128U)
        DispatchNZ<128, 128, KC>(a, b, product, t, pipe);
    else
        DispatchNZ<64, 128, KC>(a, b, product, t, pipe);
}
template <bool TA, bool TB>
constexpr bool FITS_N96 =
    NZ_TILING<TA, TB, 64, 96, 2048, 64>.baseM == 64 && NZ_TILING<TA, TB, 64, 96, 2048, 64>.baseN == 96 &&
    NZ_TILING<TA, TB, 64, 96, 2048, 64>.baseK == 64 && NZ_TILING<TA, TB, 64, 96, 2048, 64>.shareL1Size <= 459776 &&
    NZ_TILING<TA, TB, 64, 96, 2048, 64>.shareL0CSize <= 65536 && NZ_TILING<TA, TB, 64, 96, 2048, 64>.shareUbSize == 0 &&
    64U * 64U * 4U * NZ_TILING<TA, TB, 64, 96, 2048, 64>.dbL0A <= 65536U &&
    96U * 64U * 4U * NZ_TILING<TA, TB, 64, 96, 2048, 64>.dbL0B <= 65536U;
static_assert(
    FITS_N96<false, false> && FITS_N96<false, true> && FITS_N96<true, false> && FITS_N96<true, true>,
    "N96 must fit existing H2 resource gate");
} // namespace

extern "C" __global__ __aicore__ __cube__ void lt_matmul22_cube_nz_inputs(
    GM_ADDR aNz, GM_ADDR bNz, GM_ADDR product, LtMatmul22Tiling t)
{
    TPipe pipe;
    // Select a static logical K bound that contains the complete reduction.
    // Smaller bounds allow the SDK to retain a full short reduction in L1.
    if (t.k <= 64U) {
        DispatchNZTile<64>(aNz, bNz, product, t, pipe);
    } else if (t.k <= 512U) {
        DispatchNZTile<512>(aNz, bNz, product, t, pipe);
    } else if (t.k <= 2048U && t.cubeTile == 64U && t.cubeTileN == 96U) {
        DispatchNZ<64, 96, 2048, 64>(aNz, bNz, product, t, pipe);
    } else if (t.k <= 1024U && t.cubeTile == 64U && t.cubeTileN == 64U) {
        DispatchNZ<64, 64, 1024, 64>(aNz, bNz, product, t, pipe);
    } else if (
        t.k <= 2048U && t.cubeTile == 32U && t.cubeTileN == 64U && t.m <= 32U && t.n >= 512U && !t.transA &&
        !t.transB) {
        DispatchNZ<32, 64, 2048, 64>(aNz, bNz, product, t, pipe);
    } else if (t.k <= 2048U && t.cubeTile == 32U && t.cubeTileN == 128U) {
        DispatchNZ<32, 128, 2048>(aNz, bNz, product, t, pipe);
    } else {
        DispatchNZTile<4096>(aNz, bNz, product, t, pipe);
    }
}

void LtMatmul22CubeNZ(uint32_t blocks, void* stream, uint8_t* aNz, uint8_t* bNz, uint8_t* product, LtMatmul22Tiling t)
{
    // Host validates the H2 capacity and routes ordered/zero-product cases
    // before reaching this full-N, single-reduction Cube path.
    lt_matmul22_cube_nz_inputs<<<blocks, nullptr, stream>>>(aNz, bNz, product, t);
}
