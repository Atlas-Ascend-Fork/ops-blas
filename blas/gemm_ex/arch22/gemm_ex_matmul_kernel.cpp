/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CATLASS-backed arch22 implementation; dependency revision is pinned by CMake.
#ifdef GEMMEX_CATLASS_ENABLED
#include "gemm_ex_device_common.h"

using SgemmNativeArgs = gemm_ex::NativeArgs;
static_assert(sizeof(SgemmNativeArgs) == 40);
// Each Cube owns complete groups of public C columns. Group boundaries are
// 32-byte aligned even for an unaligned public leading dimension. There are
// no adjacent N tiles writing the same partial output cache line from other
// Cores. ROW_MAJOR Fixpipe supports exact nSize and element-valued dstStride.
template <class LA, class LB, uint32_t TM, uint32_t TN, uint32_t TK, bool SMALL>
__global__ __aicore__ __cube__ void gemmex_fallback_column_owner_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, SgemmNativeArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunColumnOwner<LA, LB, TM, TN, TK, SMALL>(a, b, c, t);
}
template <uint32_t TM, uint32_t TN, uint32_t TK, bool SMALL>
void LaunchColumnOwner(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    const uint32_t groups = (static_cast<uint32_t>(t.n) + TM - 1) / TM;
    blocks = blocks < groups ? blocks : groups;
    SgemmNativeArgs args{t.m, t.n, t.k, t.lda, t.ldb, t.ldc, t.transA, t.transB, 0, 0};
    using R = Catlass::layout::RowMajor;
    using C = Catlass::layout::ColumnMajor;
    if (t.transB == 0) {
        if (t.transA == 0) {
            gemmex_fallback_column_owner_kernel<R, R, TM, TN, TK, SMALL><<<blocks, nullptr, stream>>>(a, b, c, args);
        } else {
            gemmex_fallback_column_owner_kernel<R, C, TM, TN, TK, SMALL><<<blocks, nullptr, stream>>>(a, b, c, args);
        }
    } else {
        if (t.transA == 0) {
            gemmex_fallback_column_owner_kernel<C, R, TM, TN, TK, SMALL><<<blocks, nullptr, stream>>>(a, b, c, args);
        } else {
            gemmex_fallback_column_owner_kernel<C, C, TM, TN, TK, SMALL><<<blocks, nullptr, stream>>>(a, b, c, args);
        }
    }
}

void gemmex_fallback_column_owner_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    if (t.k <= 32) {
        LaunchColumnOwner<64, 512, 32, true>(blocks, stream, a, b, c, t);
    } else if (t.transA == 0 && t.transB == 0 && t.k <= 128) {
        LaunchColumnOwner<16, 512, 128, true>(blocks, stream, a, b, c, t);
    } else if (t.m <= 64 && t.k <= 256) {
        LaunchColumnOwner<96, 256, 128, false>(blocks, stream, a, b, c, t);
    } else if (t.m <= 32) {
        LaunchColumnOwner<48, 32, 128, false>(blocks, stream, a, b, c, t);
    } else if (t.m <= 128) {
        LaunchColumnOwner<16, 128, 128, false>(blocks, stream, a, b, c, t);
    } else {
        LaunchColumnOwner<32, 512, 128, true>(blocks, stream, a, b, c, t);
    }
}

template <class LayoutA, class LayoutB, uint32_t TM, uint32_t TN>
__global__ __aicore__ __cube__ void gemmex_fallback_catlass_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, SgemmNativeArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunCatlass<LayoutA, LayoutB, TM, TN, 128>(a, b, c, t);
}
template <uint32_t TM, uint32_t TN>
void LaunchNative(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling original)
{
    const auto t = gemm_ex::MakeNativeArgs<TM, TN>(original);
    blocks = gemm_ex::NativeBlockCount<TM, TN>(blocks, original);
    using R = Catlass::layout::RowMajor;
    using C = Catlass::layout::ColumnMajor;
    if (t.transB == 0) {
        if (t.transA == 0) {
            gemmex_fallback_catlass_kernel<R, R, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        } else {
            gemmex_fallback_catlass_kernel<R, C, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        }
    } else {
        if (t.transA == 0) {
            gemmex_fallback_catlass_kernel<C, R, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        } else {
            gemmex_fallback_catlass_kernel<C, C, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        }
    }
}
template <class LayoutA, class LayoutB, uint32_t TM, uint32_t TN>
__global__ __aicore__ __cube__ void gemmex_fallback_cold256_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, SgemmNativeArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunCatlass<LayoutA, LayoutB, TM, TN, 256>(a, b, c, t);
}
template <uint32_t TM, uint32_t TN>
void LaunchCold256(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling original)
{
    const auto t = gemm_ex::MakeNativeArgs<TM, TN>(original);
    blocks = gemm_ex::NativeBlockCount<TM, TN>(blocks, original);
    using R = Catlass::layout::RowMajor;
    using C = Catlass::layout::ColumnMajor;
    if (t.transB == 0) {
        if (t.transA == 0) {
            gemmex_fallback_cold256_kernel<R, R, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        } else {
            gemmex_fallback_cold256_kernel<R, C, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        }
    } else {
        if (t.transA == 0) {
            gemmex_fallback_cold256_kernel<C, R, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        } else {
            gemmex_fallback_cold256_kernel<C, C, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        }
    }
}

void gemmex_fallback_catlass_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    if (t.n <= 128) {
        LaunchNative<96, 128>(blocks, stream, a, b, c, t);
    } else if (t.m <= 128 && t.k <= 256) {
        LaunchNative<96, 64>(blocks, stream, a, b, c, t);
    } else if (t.m <= 128 && t.n >= 512) {
        LaunchNative<64, 128>(blocks, stream, a, b, c, t);
    } else if (t.m <= 128) {
        LaunchNative<64, 32>(blocks, stream, a, b, c, t);
    } else if (t.m <= 512 && t.n <= 512 && t.k > 128) {
        LaunchNative<128, 64>(blocks, stream, a, b, c, t);
    } else if (t.transA != 0 && t.transB == 0 && t.m > 1024 && t.n > 256 && t.k > 32 && t.k <= 128) {
        LaunchCold256<96, 128>(blocks, stream, a, b, c, t);
    } else {
        LaunchNative<128, 256>(blocks, stream, a, b, c, t);
    }
}

using namespace AscendC;
extern "C" __global__ __aicore__ void gemmex_fallback_catlass_copy_kernel(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n, uint32_t ldc, uint32_t sourceLd, uint32_t per,
    uint32_t batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunCatlassCopy(inputGm, outputGm, m, n, ldc, sourceLd, per, batch);
}
void gemmex_fallback_catlass_copy_do(
    uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t, uint32_t sourceLd)
{
    if (!blocks) {
        return;
    }
    gemmex_fallback_catlass_copy_kernel<<<blocks, nullptr, stream>>>(
        input, output, t.m, t.n, t.ldc, sourceLd,
        (t.m % 8 == 0 ? (static_cast<uint32_t>(t.n) + blocks - 1) / blocks :
                        ((static_cast<uint32_t>(t.n) + blocks * 8 - 1) / (blocks * 8)) * 8),
        (49152 / sourceLd / 8) * 8);
}
struct SgemmRowPackArgs {
    uint32_t rows;
    uint32_t width;
    uint32_t outputLd;
    uint32_t perAndBatch;
};
struct SgemmPackArgs {
    SgemmRowPackArgs a;
    SgemmRowPackArgs b;
};
static_assert(sizeof(SgemmPackArgs) == 32);
namespace {} // namespace
extern "C" __global__ __aicore__ void gemmex_fallback_catlass_pack_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR paGm, GM_ADDR pbGm, SgemmPackArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunCatlassPack(aGm, bGm, paGm, pbGm, t);
}
void gemmex_fallback_catlass_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR pa, GM_ADDR pb, GemmExTiling t, GemmExTiling padded)
{
    if (!blocks) {
        return;
    }
    const auto args = gemm_ex::MakePackArgs<SgemmPackArgs>(blocks, t, padded);
    gemmex_fallback_catlass_pack_kernel<<<blocks, nullptr, stream>>>(a, b, pa, pb, args);
}
// Compact NN input packing: keep the aligned GM leading dimensions consumed
// by the unchanged Cube, but transfer only logical elements through UB.
extern "C" __global__ __aicore__ void gemmex_fallback_dma_pack_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR paGm, GM_ADDR pbGm, SgemmPackArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunDmaPack(aGm, bGm, paGm, pbGm, t);
}
void gemmex_fallback_dma_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR pa, GM_ADDR pb, GemmExTiling t, GemmExTiling padded)
{
    if (!blocks) {
        return;
    }
    const uint32_t rowsA = static_cast<uint32_t>(t.transA ? t.m : t.k);
    const uint32_t rowsB = static_cast<uint32_t>(t.transB ? t.k : t.n);
    const SgemmPackArgs args{
        {rowsA, static_cast<uint32_t>(t.transA ? t.k : t.m), static_cast<uint32_t>(padded.lda),
         (rowsA + blocks - 1) / blocks},
        {rowsB, static_cast<uint32_t>(t.transB ? t.n : t.k), static_cast<uint32_t>(padded.ldb),
         (rowsB + blocks - 1) / blocks}};
    gemmex_fallback_dma_pack_kernel<<<blocks, nullptr, stream>>>(a, b, pa, pb, args);
}

using ShortNnArgs = gemm_ex::ShortArgs;

__global__ __aicore__ __cube__ void gemmex_fallback_tt_short_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR cGm, ShortNnArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetHF32Mode(HF32Mode::DISABLE);
    gemm_ex::ShortTtTile<ShortNnArgs, false> tile(t);
    tile.Run(a, b, cGm);
}
void gemmex_fallback_tt_short_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 127) / 128;
    const uint32_t grid = ((static_cast<uint32_t>(t.n) + 191) / 192) * columns;
    const auto q = gemm_ex::MakeShortArgs(t, static_cast<uint32_t>(t.ldc), columns);
    const uint32_t activeBlocks = blocks < grid ? blocks : grid;
    gemmex_fallback_tt_short_kernel<<<activeBlocks, nullptr, stream>>>(a, b, c, q);
}

__global__ __aicore__ __cube__ void gemmex_fallback_short_native_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR cGm, ShortNnArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunShortNative(a, b, cGm, t);
}
void gemmex_fallback_short_native_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t, uint32_t ldc)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 383) / 384;
    const auto q = gemm_ex::MakeShortArgs(t, ldc, columns);
    gemmex_fallback_short_native_kernel<<<blocks, nullptr, stream>>>(a, b, c, q);
}
template <uint32_t MODE>
__global__ __aicore__ __cube__ void gemmex_fallback_full117_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR cGm, ShortNnArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetHF32Mode(HF32Mode::DISABLE);
    using namespace Catlass;
    using LA = layout::RowMajor;
    using LB = layout::RowMajor;
    constexpr uint32_t TM = 64;
    constexpr uint32_t TN = 368;
    const uint32_t core = GetBlockIdx();
    using Types = gemm_ex::TileTypes<LA, LB>;
    const uint32_t row = gemm_ex::TileRow(core, t.columns, t.reciprocal);
    const uint32_t col = core - row * t.columns;
    const uint32_t mb = row * TM;
    const uint32_t nb = col * TN;
    const uint32_t am = min(TM, t.n - mb);
    const uint32_t an = min(TN, t.m - nb);
    const uint32_t mr = RoundUp<Gemm::helper::L1AlignHelper<float, LA>::M_ALIGNED>(am);
    const uint32_t nr = ((an + 15) / 16 * 16);
    LocalTensor<float> l1a(TPosition::A1, 0, TM * 32);
    LocalTensor<float> l1b(TPosition::A1, TM * 32 * 4, 32 * TN);
    LocalTensor<float> l0a(TPosition::A2, 0, TM * 32);
    LocalTensor<float> l0b(TPosition::B2, 0, 32 * TN);
    LocalTensor<float> acc(TPosition::CO1, 0, TM * TN);
    auto right = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(a));
    auto scratch = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(cGm) + static_cast<uint64_t>(core) * TM * TN);
    LA la(t.n, t.k, t.ldb);
    LB lb(t.k, t.m, t.lda);
    auto layoutL1A = Types::L1A::template MakeLayout<float>(TM, 32);
    auto layoutL1B = Types::L1B::template MakeLayout<float>(32, TN);
    typename Types::Copy::CopyGmToL1B gmToB;
    auto leftBits = gemm_ex::BindGlobal(reinterpret_cast<__gm__ half*>(b));
    if constexpr (MODE == 4 || MODE == 6) {
        leftBits.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    }
    if constexpr (MODE == 5 || MODE == 6) {
        right.SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
    }
    DataCopy(
        l1a.ReinterpretCast<half>(), leftBits[la.GetOffset(MatrixCoord(mb, 0)) * 2],
        Nd2NzParams{
            1, static_cast<uint16_t>(am), static_cast<uint16_t>(t.k * 2), 0, static_cast<uint16_t>(t.ldb * 2), TM, 1,
            0});
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    gmToB(l1b, right[lb.GetOffset(MatrixCoord(0, nb))], layoutL1B, lb.GetTileLayout(MakeCoord(t.k, an)));
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    gemm_ex::ComputeLoadedTile<Types>({l1a, l1b, l0a, l0b, acc}, layoutL1A, layoutL1B, scratch, mr, nr, t.k, TM * 2);
}
void gemmex_fallback_short_nz_native_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t, uint32_t ldc)
{
    constexpr uint32_t tn = 368;
    const uint32_t columns = (static_cast<uint32_t>(t.m) + tn - 1) / tn;
    const auto q = gemm_ex::MakeShortArgs(t, ldc, columns);
    gemmex_fallback_full117_kernel<1><<<blocks, nullptr, stream>>>(a, b, c, q);
}

// Controlled no-count variant of the pinned SDK vreducev2 implementation.
// This compaction has a known result length, so no scalar count is consumed.
template <uint32_t COLUMNS>
__global__ __aicore__ void gemmex_fallback_drain_copy101_kernel(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunShortDrain<COLUMNS, 0>(inputGm, outputGm, m, n);
}
void gemmex_fallback_nz_bitcopy_do(uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t)
{
    const uint32_t use = std::min(blocks, (static_cast<uint32_t>(t.n) + 15) / 16);
    const uint32_t cols = (static_cast<uint32_t>(t.m) + 367) / 368;
    if (cols == 1) {
        gemmex_fallback_drain_copy101_kernel<1><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 2) {
        gemmex_fallback_drain_copy101_kernel<2><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 3) {
        gemmex_fallback_drain_copy101_kernel<3><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 4) {
        gemmex_fallback_drain_copy101_kernel<4><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else {
        gemmex_fallback_drain_copy101_kernel<5><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    }
}
// Resident compact NN, 32 < K <= 128. No input packing.
// Tile copies follow pinned CATLASS 14d5536; L0C is ordinary float NZ C0=16.
__global__ __aicore__ __cube__ void gemmex_fallback_moderate_nz_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR cGm, ShortNnArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunModerateNz(a, b, cGm, t);
}
void gemmex_fallback_moderate_nz_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 255) / 256;
    const auto q = gemm_ex::MakeShortArgs(t, 0, columns);
    gemmex_fallback_moderate_nz_kernel<<<blocks, nullptr, stream>>>(a, b, c, q);
}

extern "C" __global__ __aicore__ void gemmex_fallback_nz_fourrow_copy_kernel(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunNzFourrowCopy(inputGm, outputGm, m, n);
}
void gemmex_fallback_nz_fourrow_copy_do(uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t)
{
    gemmex_fallback_nz_fourrow_copy_kernel<<<blocks, nullptr, stream>>>(input, output, t.m, t.n);
}

// Bounded two-slot NZ-to-public-ND bitwise transfer.
// Each AIV owns 8 complete public columns; no partial cacheline is shared.
extern "C" __global__ __aicore__ void gemmex_fallback_nz_pipeline40_kernel(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunNzPipeline40(inputGm, outputGm, m, n);
}
void gemmex_fallback_nz_loop_copy_do(uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t)
{
    gemmex_fallback_nz_pipeline40_kernel<<<blocks, nullptr, stream>>>(input, output, t.m, t.n);
}

// A short output grid is expanded over K. Every partial is FP32 and reduced
// by an explicit, measured AIV kernel; there are no atomic output races.
template <uint32_t TM, uint32_t TK, uint32_t SPLITS>
__global__ __aicore__ __cube__ void gemmex_fallback_split_nt_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outGm, SgemmNativeArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunSplitNt<TM, TK, SPLITS>(aGm, bGm, outGm, t);
}
template <uint32_t TM, uint32_t TK, uint32_t SPLITS>
void LaunchSplitNt(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling original)
{
    const uint32_t groups = (static_cast<uint32_t>(original.n) + TM - 1) / TM;
    const uint32_t grid = groups * SPLITS;
    const auto t = gemm_ex::MakeSplitArgs(original, groups);
    gemmex_fallback_split_nt_kernel<TM, TK, SPLITS><<<std::min(blocks, grid), nullptr, stream>>>(a, b, c, t);
}
void gemmex_fallback_split_nt_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    LaunchSplitNt<128, 128, 8>(blocks, stream, a, b, c, t);
}
extern "C" __global__ __aicore__ void gemmex_fallback_split_sum_kernel(
    GM_ADDR sourceGm, GM_ADDR destGm, uint32_t m, uint32_t n, uint32_t sourceLd, uint32_t splits)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunSplitSum(sourceGm, destGm, m, n, sourceLd, splits);
}
void gemmex_fallback_split_sum_do(
    uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t, uint32_t sourceLd, uint32_t splits)
{
    gemmex_fallback_split_sum_kernel<<<blocks, nullptr, stream>>>(input, output, t.m, t.n, sourceLd, splits);
}
#endif
