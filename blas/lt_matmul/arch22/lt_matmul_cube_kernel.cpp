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

// Shared static Matmul implementation; each device translation unit owns its instances.
namespace {
using namespace AscendC;
using namespace matmul;
constexpr uint32_t BM = LT_MATMUL22_TILE_M;
constexpr uint32_t BN = LT_MATMUL22_TILE_N;
constexpr uint32_t HALF_M = BM / 2;
constexpr uint32_t K_PANEL = 4096;
// One static MatmulImpl, no KFC registration or system device workspace.
template <uint32_t TM, uint32_t TN, bool SHORT>
constexpr MatmulShapeParams SHAPE{
    TM,
    TN,
    SHORT ? 512U : K_PANEL,
    TM == 256 ? 64U : TM,
    TN,
    SHORT ? (TM == 256 || (TM == 32 && TN == 64) ? 32U : 64U) : (TN == 128 || (TM == 128 && TN == 112) ? 64U : 32U)};
constexpr MatmulFuncParams FUNC{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                true,  true};
constexpr MatmulBiasParams BIAS{false};
template <uint32_t TM, uint32_t TN, bool SHORT>
constexpr MatmulConfig CONFIG = GetMMConfig < SHORT && TM != 64U ?
                                    MatmulConfigMode::CONFIG_NORM :
                                    MatmulConfigMode::CONFIG_MDL > (SHAPE<TM, TN, SHORT>, FUNC, BIAS);

template <bool TRANS>
using Input = MatmulType<TPosition::GM, CubeFormat::ND, float, TRANS>;
using Output = MatmulType<TPosition::GM, CubeFormat::ND, float>;
// Native ND copy-out preserves the logical N tail; ND_ALIGN is the
// separate output format that pads rows. Keep the high-level API contract.
using NativeOutputCallback = MatmulCallBackFunc<nullptr, nullptr, nullptr>;

template <bool TA, bool TB, uint32_t TM, uint32_t TN, bool SHORT>
constexpr MatmulApiStaticTiling TILING =
    GetMatmulApiTiling<Input<TA>, Input<TB>, Output, Output>(CONFIG<TM, TN, SHORT>);
// The narrower long-TN tile stays inside the original H1 capsule.
template <bool TA, bool TB>
constexpr bool FITS_H1_N112 =
    TILING<TA, TB, 128, 112, false>.baseM == 128 && TILING<TA, TB, 128, 112, false>.baseN == 112 &&
    TILING<TA, TB, 128, 112, false>.baseK == 64 && TILING<TA, TB, 128, 112, false>.shareL1Size <= 459776 &&
    TILING<TA, TB, 128, 112, false>.shareL0CSize <= 65536 && TILING<TA, TB, 128, 112, false>.shareUbSize == 0 &&
    128U * 64U * 4U * TILING<TA, TB, 128, 112, false>.dbL0A <= 65536U &&
    112U * 64U * 4U * TILING<TA, TB, 128, 112, false>.dbL0B <= 65536U;
static_assert(
    FITS_H1_N112<false, false> && FITS_H1_N112<false, true> && FITS_H1_N112<true, false> && FITS_H1_N112<true, true>,
    "N112 must fit the unchanged H1 resources");
template <bool TA, bool TB, uint32_t TM, uint32_t TN, bool MIXED, bool SHORT>
using MatrixMultiply =
    MatmulImpl<Input<TA>, Input<TB>, Output, Output, TILING<TA, TB, TM, TN, SHORT>, NativeOutputCallback>;

// Slots 0..7 are reserved for Matmul, 11..15 for runtime/library sync.
// mode 2 synchronizes only the AIC and its two AIVs, with no global barrier.
constexpr uint16_t SNAPSHOT_READY = 8;
constexpr uint16_t PRODUCT_READY = 9;
constexpr uint16_t EPILOGUE_DONE = 10;

__aicore__ inline uint32_t MinCount(uint32_t a, uint32_t b) { return a < b ? a : b; }

template <bool TA, bool TB, uint32_t TM, uint32_t TN, bool MIXED, bool SHORT>
__aicore__ inline void ProductTile(
    MatrixMultiply<TA, TB, TM, TN, MIXED, SHORT>& mm, const GlobalTensor<float>& ag, const GlobalTensor<float>& bg,
    const GlobalTensor<float>& pg, const LtMatmul22Tiling& t, uint32_t row, uint32_t col)
{
    // SetOrgShape carries physical pitches; SetSingleShape carries logical extents.
    mm.SetOrgShape(TA ? t.lda : t.m, TB ? t.n : t.ldb, TA ? t.k : t.lda, TB ? t.ldb : t.k, t.productLd);
    const uint32_t curM = MinCount(TM, t.m - row);
    const uint32_t curN = MinCount(TN, t.nCube - col);
    // The public dispatcher sends K>4096 to the ordered FP32 fallback.
    // A Cube tile therefore has one full reduction and never atomic panels.
    mm.SetSingleShape(curM, curN, t.k);
    const uint64_t aOffset = TA ? row : static_cast<uint64_t>(row) * t.lda;
    const uint64_t bOffset = TB ? static_cast<uint64_t>(col) * t.ldb : col;
    mm.SetTensorA(ag[aOffset], TA);
    mm.SetTensorB(bg[bOffset], TB);
    mm.IterateAll(pg[static_cast<uint64_t>(row) * t.productLd + col], false);
    mm.End();
}

template <bool TA, bool TB, bool MIXED, uint32_t TM, uint32_t TN, bool SHORT>
__aicore__ inline void CubeMain(GM_ADDR a, GM_ADDR b, GM_ADDR product, const LtMatmul22Tiling& t, TPipe& pipe)
{
    MatrixMultiply<TA, TB, TM, TN, MIXED, SHORT> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> ag, bg, pg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    pg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(product));
    const uint64_t cols = (static_cast<uint64_t>(t.nCube) + TN - 1) / TN;
    const uint64_t rows = (static_cast<uint64_t>(t.m) + TM - 1) / TM;
    for (uint64_t tile = GetBlockIdx(); tile < rows * cols; tile += GetBlockNum()) {
        const uint32_t row = tile / cols * TM;
        const uint32_t col = tile % cols * TN;
        if constexpr (MIXED) {
            CrossCoreWaitFlag<2>(SNAPSHOT_READY);
        }
        ProductTile<TA, TB, TM, TN, MIXED, SHORT>(mm, ag, bg, pg, t, row, col);
        if constexpr (MIXED) {
            CrossCoreSetFlag<2, PIPE_FIX>(PRODUCT_READY);
            CrossCoreWaitFlag<2>(EPILOGUE_DONE);
        }
    }
}

} // namespace

namespace {
__aicore__ inline void ReadRows(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint64_t offset, uint32_t rows, uint32_t cols,
    uint32_t ld)
{
    const uint64_t gap = static_cast<uint64_t>(ld - cols) * 4U;
    const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    if (gap <= 0xFFFFFFFFULL) {
        const DataCopyExtParams copy{static_cast<uint16_t>(rows), cols * 4U, static_cast<uint32_t>(gap), 0, 0};
        DataCopyPad(dst, src[offset], copy, pad);
    } else {
        const DataCopyExtParams copy{1, cols * 4U, 0, 0, 0};
        for (uint32_t row = 0; row < rows; ++row) {
            DataCopyPad(dst[row * cols], src[offset + static_cast<uint64_t>(row) * ld], copy, pad);
        }
    }
}

__aicore__ inline void WriteRows(
    const GlobalTensor<float>& dst, const LocalTensor<float>& src, uint64_t offset, uint32_t rows, uint32_t cols,
    uint32_t ld)
{
    const uint64_t gap = static_cast<uint64_t>(ld - cols) * 4U;
    if (gap <= 0xFFFFFFFFULL) {
        const DataCopyExtParams copy{static_cast<uint16_t>(rows), cols * 4U, 0, static_cast<uint32_t>(gap), 0};
        DataCopyPad(dst[offset], src, copy);
    } else {
        const DataCopyExtParams copy{1, cols * 4U, 0, 0, 0};
        for (uint32_t row = 0; row < rows; ++row) {
            DataCopyPad(dst[offset + static_cast<uint64_t>(row) * ld], src[row * cols], copy);
        }
    }
}

__aicore__ inline void InplaceEpilogue(GM_ADDR c, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    // Each AIV snapshots at most 32*64 FP32 elements, with one product buffer.
    TBuf<TPosition::VECCALC> oldBuffer, productBuffer;
    pipe.InitBuffer(oldBuffer, HALF_M * BN * sizeof(float));
    pipe.InitBuffer(productBuffer, HALF_M * BN * sizeof(float));
    const LocalTensor<float> old = oldBuffer.Get<float>();
    const LocalTensor<float> product = productBuffer.Get<float>();
    GlobalTensor<float> cg, dg;
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    dg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    const uint64_t cols = (static_cast<uint64_t>(t.nCube) + BN - 1) / BN;
    const uint64_t rows = (static_cast<uint64_t>(t.m) + BM - 1) / BM;
    const uint32_t half = GetBlockIdx() % 2;
    const uint32_t group = GetBlockIdx() / 2;
    for (uint64_t tile = group; tile < rows * cols; tile += GetBlockNum()) {
        const uint32_t row = tile / cols * BM + half * HALF_M;
        const uint32_t col = tile % cols * BN;
        const uint32_t curM = row < t.m ? MinCount(HALF_M, t.m - row) : 0;
        const uint32_t curN = MinCount(BN, t.nCube - col);
        const uint32_t count = curM * curN;
        if (count != 0) {
            ReadRows(old, cg, static_cast<uint64_t>(row) * t.ldc + col, curM, curN, t.ldc);
        }
        // Empty halves still participate in all three handshakes.
        CrossCoreSetFlag<2, PIPE_MTE2>(SNAPSHOT_READY);
        CrossCoreWaitFlag<2>(PRODUCT_READY);
        if (count != 0) {
            ReadRows(product, dg, static_cast<uint64_t>(row) * t.ldd + col, curM, curN, t.ldd);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if (t.alpha != 1.0f) {
                Muls(product, product, t.alpha, count);
            }
            Muls(old, old, t.beta, count);
            PipeBarrier<PIPE_V>();
            Add(product, product, old, count);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WriteRows(dg, product, static_cast<uint64_t>(row) * t.ldd + col, curM, curN, t.ldd);
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(EPILOGUE_DONE);
        // Prevent product UB reuse before MTE3 has consumed the previous tile.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

template <bool MIXED, uint32_t TM, uint32_t TN, bool SHORT = false>
__aicore__ inline void DispatchCube(GM_ADDR a, GM_ADDR b, GM_ADDR product, const LtMatmul22Tiling& t, TPipe& pipe)
{
    if (t.transA) {
        if (t.transB) {
            CubeMain<true, true, MIXED, TM, TN, SHORT>(a, b, product, t, pipe);
        } else {
            CubeMain<true, false, MIXED, TM, TN, SHORT>(a, b, product, t, pipe);
        }
    } else if (t.transB) {
        CubeMain<false, true, MIXED, TM, TN, SHORT>(a, b, product, t, pipe);
    } else {
        CubeMain<false, false, MIXED, TM, TN, SHORT>(a, b, product, t, pipe);
    }
}
} // namespace

extern "C" __global__ __aicore__ __cube__ void lt_matmul22_cube(
    GM_ADDR a, GM_ADDR b, GM_ADDR product, LtMatmul22Tiling t)
{
    TPipe pipe;
    if (t.shortK) {
        if (t.cubeTile == 256U) {
            DispatchCube<false, 256, 128, true>(a, b, product, t, pipe);
        } else if (t.cubeTile == 32U) {
            DispatchCube<false, 32, 64, true>(a, b, product, t, pipe);
        } else if (t.cubeTile == 64U && t.cubeTileN == 128U) {
            DispatchCube<false, 64, 128, true>(a, b, product, t, pipe);
        } else if (t.cubeTile == 48U) {
            DispatchCube<false, 48, 96, true>(a, b, product, t, pipe);
        } else if (t.cubeTile == 80U) {
            DispatchCube<false, 80, 96, true>(a, b, product, t, pipe);
        } else if (t.cubeTile == 96U) {
            DispatchCube<false, 96, 128, true>(a, b, product, t, pipe);
        } else if (t.cubeTile == 128U) {
            DispatchCube<false, 128, 128, true>(a, b, product, t, pipe);
        } else {
            DispatchCube<false, 64, 64, true>(a, b, product, t, pipe);
        }
    } else if (t.cubeTile == 32U && t.cubeTileN == 64U) {
        DispatchCube<false, 32, 64>(a, b, product, t, pipe);
    } else if (t.cubeTile == 32U) {
        DispatchCube<false, 32, 128>(a, b, product, t, pipe);
    } else if (t.cubeTile == 128U && t.cubeTileN == 112U) {
        DispatchCube<false, 128, 112>(a, b, product, t, pipe);
    } else if (t.cubeTile == LT_MATMUL22_H1_TILE_M) {
        DispatchCube<false, 128, 128>(a, b, product, t, pipe);
    } else if (t.cubeTileN == LT_MATMUL22_H1_TILE_N) {
        DispatchCube<false, 64, 128>(a, b, product, t, pipe);
    } else {
        DispatchCube<false, 64, 64>(a, b, product, t, pipe);
    }
}

extern "C" __global__ __aicore__ void lt_matmul22_mix(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, LtMatmul22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if ASCEND_IS_AIC {
        DispatchCube<true, 64, 64>(a, b, d, t, pipe);
    } else {
        InplaceEpilogue(c, d, t, pipe);
    }
}

void LtMatmul22Cube(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* product, LtMatmul22Tiling t)
{
    lt_matmul22_cube<<<blocks, nullptr, stream>>>(a, b, product, t);
}

void LtMatmul22Mix(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, LtMatmul22Tiling t)
{
    lt_matmul22_mix<<<blocks, nullptr, stream>>>(a, b, c, d, t);
}

namespace {
constexpr auto TN64X192_TILING = TILING<true, false, 64, 192, false>;
static_assert(
    TN64X192_TILING.baseM == 64U && TN64X192_TILING.baseN == 192U && TN64X192_TILING.baseK == 32U,
    "The long-TN instance must retain its verified full-K H1 reduction grain");
static_assert(
    TN64X192_TILING.shareL1Size <= 459776U && TN64X192_TILING.shareL0CSize <= 65536U &&
        TN64X192_TILING.shareUbSize == 0U && 64U * 32U * sizeof(float) * TN64X192_TILING.dbL0A <= 65536U &&
        192U * 32U * sizeof(float) * TN64X192_TILING.dbL0B <= 65536U,
    "The long-TN instance must fit the unchanged H1 resources");
} // namespace

extern "C" __global__ __aicore__ __cube__ void lt_matmul22_cube_tn_64x192(
    GM_ADDR a, GM_ADDR b, GM_ADDR product, LtMatmul22Tiling t)
{
    TPipe pipe;
    CubeMain<true, false, false, 64, 192, false>(a, b, product, t, pipe);
}

void LtMatmul22CubeTN64x192(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* product, LtMatmul22Tiling t)
{
    lt_matmul22_cube_tn_64x192<<<blocks, nullptr, stream>>>(a, b, product, t);
}
