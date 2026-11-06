/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef GEMMEX_CATLASS_ENABLED
#include "gemm_ex_device_common.h"

// CATLASS-backed arch22 implementation; dependency revision is pinned by CMake.

using SgemmExPrivateSgemmNativeArgs = gemm_ex::NativeArgs;
static_assert(sizeof(SgemmExPrivateSgemmNativeArgs) == 40);
// Each Cube owns complete groups of public C columns. Group boundaries are
// 32-byte aligned even for an unaligned public leading dimension. There are
// no adjacent N tiles writing the same partial output cache line from other
// Cores. ROW_MAJOR Fixpipe supports exact nSize and element-valued dstStride.
template <class LA, class LB, uint32_t TM, uint32_t TN, uint32_t TK, bool SMALL>
__global__ __aicore__ __cube__ void sgemm_ex_arch22_column_owner_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, SgemmExPrivateSgemmNativeArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunColumnOwner<LA, LB, TM, TN, TK, SMALL>(a, b, c, t);
}
template <uint32_t TM, uint32_t TN, uint32_t TK, bool SMALL>
void SgemmExPrivateLaunchColumnOwner(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    const uint32_t groups = (static_cast<uint32_t>(t.n) + TM - 1) / TM;
    blocks = blocks < groups ? blocks : groups;
    SgemmExPrivateSgemmNativeArgs args{t.m, t.n, t.k, t.lda, t.ldb, t.ldc, t.transA, t.transB, 0, 0};
    using R = Catlass::layout::RowMajor;
    using C = Catlass::layout::ColumnMajor;
    if (t.transB == 0) {
        if (t.transA == 0) {
            sgemm_ex_arch22_column_owner_kernel<R, R, TM, TN, TK, SMALL><<<blocks, nullptr, stream>>>(a, b, c, args);
        } else {
            sgemm_ex_arch22_column_owner_kernel<R, C, TM, TN, TK, SMALL><<<blocks, nullptr, stream>>>(a, b, c, args);
        }
    } else {
        if (t.transA == 0) {
            sgemm_ex_arch22_column_owner_kernel<C, R, TM, TN, TK, SMALL><<<blocks, nullptr, stream>>>(a, b, c, args);
        } else {
            sgemm_ex_arch22_column_owner_kernel<C, C, TM, TN, TK, SMALL><<<blocks, nullptr, stream>>>(a, b, c, args);
        }
    }
}

void sgemm_ex_arch22_column_owner_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    if (t.k <= 32) {
        SgemmExPrivateLaunchColumnOwner<64, 512, 32, true>(blocks, stream, a, b, c, t);
    } else if (t.transA == 0 && t.transB == 0 && t.k <= 128) {
        SgemmExPrivateLaunchColumnOwner<16, 512, 128, true>(blocks, stream, a, b, c, t);
    } else if (t.m <= 64 && t.k <= 256) {
        SgemmExPrivateLaunchColumnOwner<96, 256, 128, false>(blocks, stream, a, b, c, t);
    } else if (t.m <= 32) {
        SgemmExPrivateLaunchColumnOwner<48, 32, 128, false>(blocks, stream, a, b, c, t);
    } else if (t.m <= 128) {
        SgemmExPrivateLaunchColumnOwner<16, 128, 128, false>(blocks, stream, a, b, c, t);
    } else {
        SgemmExPrivateLaunchColumnOwner<32, 512, 128, true>(blocks, stream, a, b, c, t);
    }
}

template <class LayoutA, class LayoutB, uint32_t TM, uint32_t TN>
__global__ __aicore__ __cube__ void sgemm_ex_arch22_catlass_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, SgemmExPrivateSgemmNativeArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunCatlass<LayoutA, LayoutB, TM, TN, 128>(a, b, c, t);
}
template <uint32_t TM, uint32_t TN>
void SgemmExPrivateLaunchNative(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling original)
{
    const auto t = gemm_ex::MakeNativeArgs<TM, TN>(original);
    blocks = gemm_ex::NativeBlockCount<TM, TN>(blocks, original);
    using R = Catlass::layout::RowMajor;
    using C = Catlass::layout::ColumnMajor;
    if (t.transB == 0) {
        if (t.transA == 0) {
            sgemm_ex_arch22_catlass_kernel<R, R, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        } else {
            sgemm_ex_arch22_catlass_kernel<R, C, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        }
    } else {
        if (t.transA == 0) {
            sgemm_ex_arch22_catlass_kernel<C, R, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        } else {
            sgemm_ex_arch22_catlass_kernel<C, C, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        }
    }
}
template <class LayoutA, class LayoutB, uint32_t TM, uint32_t TN>
__global__ __aicore__ __cube__ void sgemm_ex_arch22_cold256_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, SgemmExPrivateSgemmNativeArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunCatlass<LayoutA, LayoutB, TM, TN, 256>(a, b, c, t);
}
template <uint32_t TM, uint32_t TN>
void SgemmExPrivateLaunchCold256(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling original)
{
    const auto t = gemm_ex::MakeNativeArgs<TM, TN>(original);
    blocks = gemm_ex::NativeBlockCount<TM, TN>(blocks, original);
    using R = Catlass::layout::RowMajor;
    using C = Catlass::layout::ColumnMajor;
    if (t.transB == 0) {
        if (t.transA == 0) {
            sgemm_ex_arch22_cold256_kernel<R, R, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        } else {
            sgemm_ex_arch22_cold256_kernel<R, C, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        }
    } else {
        if (t.transA == 0) {
            sgemm_ex_arch22_cold256_kernel<C, R, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        } else {
            sgemm_ex_arch22_cold256_kernel<C, C, TM, TN><<<blocks, nullptr, stream>>>(a, b, c, t);
        }
    }
}

void sgemm_ex_arch22_catlass_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    if (t.n <= 128) {
        SgemmExPrivateLaunchNative<96, 128>(blocks, stream, a, b, c, t);
    } else if (t.m <= 128 && t.k <= 256) {
        SgemmExPrivateLaunchNative<96, 64>(blocks, stream, a, b, c, t);
    } else if (t.m <= 128 && t.n >= 512) {
        SgemmExPrivateLaunchNative<64, 128>(blocks, stream, a, b, c, t);
    } else if (t.m <= 128) {
        SgemmExPrivateLaunchNative<64, 32>(blocks, stream, a, b, c, t);
    } else if (t.m <= 512 && t.n <= 512 && t.k > 128) {
        SgemmExPrivateLaunchNative<128, 64>(blocks, stream, a, b, c, t);
    } else if (t.transA != 0 && t.transB == 0 && t.m > 1024 && t.n > 256 && t.k > 32 && t.k <= 128) {
        SgemmExPrivateLaunchCold256<96, 128>(blocks, stream, a, b, c, t);
    } else {
        SgemmExPrivateLaunchNative<128, 256>(blocks, stream, a, b, c, t);
    }
}

using namespace AscendC;
extern "C" __global__ __aicore__ void sgemm_ex_arch22_catlass_copy_kernel(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n, uint32_t ldc, uint32_t sourceLd, uint32_t per,
    uint32_t batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunCatlassCopy(inputGm, outputGm, m, n, ldc, sourceLd, per, batch);
}
void sgemm_ex_arch22_catlass_copy_do(
    uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t, uint32_t sourceLd)
{
    if (!blocks) {
        return;
    }
    sgemm_ex_arch22_catlass_copy_kernel<<<blocks, nullptr, stream>>>(
        input, output, t.m, t.n, t.ldc, sourceLd,
        (t.m % 8 == 0 ? (static_cast<uint32_t>(t.n) + blocks - 1) / blocks :
                        ((static_cast<uint32_t>(t.n) + blocks * 8 - 1) / (blocks * 8)) * 8),
        (49152 / sourceLd / 8) * 8);
}
struct SgemmExPrivateSgemmRowPackArgs {
    uint32_t rows;
    uint32_t width;
    uint32_t outputLd;
    uint32_t perAndBatch;
};
struct SgemmExPrivateSgemmPackArgs {
    SgemmExPrivateSgemmRowPackArgs a;
    SgemmExPrivateSgemmRowPackArgs b;
};
static_assert(sizeof(SgemmExPrivateSgemmPackArgs) == 32);
namespace {} // namespace
extern "C" __global__ __aicore__ void sgemm_ex_arch22_catlass_pack_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR paGm, GM_ADDR pbGm, SgemmExPrivateSgemmPackArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunCatlassPack(aGm, bGm, paGm, pbGm, t);
}
void sgemm_ex_arch22_catlass_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR pa, GM_ADDR pb, GemmExTiling t, GemmExTiling padded)
{
    if (!blocks) {
        return;
    }
    const auto args = gemm_ex::MakePackArgs<SgemmExPrivateSgemmPackArgs>(blocks, t, padded);
    sgemm_ex_arch22_catlass_pack_kernel<<<blocks, nullptr, stream>>>(a, b, pa, pb, args);
}
// Compact NN input packing: keep the aligned GM leading dimensions consumed
// by the unchanged Cube, but transfer only logical elements through UB.
extern "C" __global__ __aicore__ void sgemm_ex_arch22_dma_pack_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR paGm, GM_ADDR pbGm, SgemmExPrivateSgemmPackArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunDmaPack(aGm, bGm, paGm, pbGm, t);
}
void sgemm_ex_arch22_dma_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR pa, GM_ADDR pb, GemmExTiling t, GemmExTiling padded)
{
    if (!blocks) {
        return;
    }
    const uint32_t rowsA = static_cast<uint32_t>(t.transA ? t.m : t.k);
    const uint32_t rowsB = static_cast<uint32_t>(t.transB ? t.k : t.n);
    const SgemmExPrivateSgemmPackArgs args{
        {rowsA, static_cast<uint32_t>(t.transA ? t.k : t.m), static_cast<uint32_t>(padded.lda),
         (rowsA + blocks - 1) / blocks},
        {rowsB, static_cast<uint32_t>(t.transB ? t.n : t.k), static_cast<uint32_t>(padded.ldb),
         (rowsB + blocks - 1) / blocks}};
    sgemm_ex_arch22_dma_pack_kernel<<<blocks, nullptr, stream>>>(a, b, pa, pb, args);
}

using SgemmExPrivateShortNnArgs = gemm_ex::ShortArgs;

__global__ __aicore__ __cube__ void sgemm_ex_arch22_short_native_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR cGm, SgemmExPrivateShortNnArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunShortNative(a, b, cGm, t);
}
void sgemm_ex_arch22_short_native_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t, uint32_t ldc)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 383) / 384;
    const auto q = gemm_ex::MakeShortArgs(t, ldc, columns);
    sgemm_ex_arch22_short_native_kernel<<<blocks, nullptr, stream>>>(a, b, c, q);
}

// Resident compact NN, 32 < K <= 128. No input packing.
// Tile copies follow pinned CATLASS 14d5536; L0C is ordinary float NZ C0=16.
__global__ __aicore__ __cube__ void sgemm_ex_arch22_moderate_nz_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR cGm, SgemmExPrivateShortNnArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunModerateNz(a, b, cGm, t);
}
void sgemm_ex_arch22_moderate_nz_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 255) / 256;
    const auto q = gemm_ex::MakeShortArgs(t, 0, columns);
    sgemm_ex_arch22_moderate_nz_kernel<<<blocks, nullptr, stream>>>(a, b, c, q);
}

extern "C" __global__ __aicore__ void sgemm_ex_arch22_nz_fourrow_copy_kernel(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunNzFourrowCopy(inputGm, outputGm, m, n);
}
void sgemm_ex_arch22_nz_fourrow_copy_do(uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t)
{
    sgemm_ex_arch22_nz_fourrow_copy_kernel<<<blocks, nullptr, stream>>>(input, output, t.m, t.n);
}

// Bounded two-slot NZ-to-public-ND bitwise transfer.
// Each AIV owns 8 complete public columns; no partial cacheline is shared.
extern "C" __global__ __aicore__ void sgemm_ex_arch22_nz_pipeline40_kernel(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunNzPipeline40(inputGm, outputGm, m, n);
}
void sgemm_ex_arch22_nz_loop_copy_do(uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t)
{
    sgemm_ex_arch22_nz_pipeline40_kernel<<<blocks, nullptr, stream>>>(input, output, t.m, t.n);
}

// A short output grid is expanded over K. Every partial is FP32 and reduced
// by an explicit, measured AIV kernel; there are no atomic output races.
template <uint32_t TM, uint32_t TK, uint32_t SPLITS>
__global__ __aicore__ __cube__ void sgemm_ex_arch22_split_nt_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outGm, SgemmExPrivateSgemmNativeArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunSplitNt<TM, TK, SPLITS>(aGm, bGm, outGm, t);
}
template <uint32_t TM, uint32_t TK, uint32_t SPLITS>
void SgemmExPrivateLaunchSplitNt(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling original)
{
    const uint32_t groups = (static_cast<uint32_t>(original.n) + TM - 1) / TM;
    const uint32_t grid = groups * SPLITS;
    const auto t = gemm_ex::MakeSplitArgs(original, groups);
    sgemm_ex_arch22_split_nt_kernel<TM, TK, SPLITS><<<std::min(blocks, grid), nullptr, stream>>>(a, b, c, t);
}
void sgemm_ex_arch22_split_nt_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    SgemmExPrivateLaunchSplitNt<128, 128, 8>(blocks, stream, a, b, c, t);
}
extern "C" __global__ __aicore__ void sgemm_ex_arch22_split_sum_kernel(
    GM_ADDR sourceGm, GM_ADDR destGm, uint32_t m, uint32_t n, uint32_t sourceLd, uint32_t splits)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunSplitSum(sourceGm, destGm, m, n, sourceLd, splits);
}
void sgemm_ex_arch22_split_sum_do(
    uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t, uint32_t sourceLd, uint32_t splits)
{
    sgemm_ex_arch22_split_sum_kernel<<<blocks, nullptr, stream>>>(input, output, t.m, t.n, sourceLd, splits);
}

using SmallTtArgs259 = gemm_ex::NativeArgs;
static_assert(sizeof(SmallTtArgs259) == 40);
template <class LayoutA, class LayoutB, uint32_t TM, uint32_t TN>
__global__ __aicore__ __cube__ void sgemm_ex_smalltt259_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, SmallTtArgs259 t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    gemm_ex::RunCatlass<LayoutA, LayoutB, TM, TN, 128>(a, b, c, t);
}
void SgemmExSmallTt259Do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 63) / 64;
    const uint32_t grid = ((static_cast<uint32_t>(t.n) + 31) / 32) * columns;
    const uint32_t reciprocal = columns == 1 ? 0 : static_cast<uint32_t>(((1ULL << 32) + columns - 1) / columns);
    SmallTtArgs259 args{t.m, t.n, t.k, t.lda, t.ldb, t.ldc, t.transA, t.transB, grid | (columns << 16), reciprocal};
    using C = Catlass::layout::ColumnMajor;
    const uint32_t activeBlocks = blocks < grid ? blocks : grid;
    sgemm_ex_smalltt259_kernel<C, C, 32, 64><<<activeBlocks, nullptr, stream>>>(a, b, c, args);
}

// CATLASS-backed arch22 implementation; dependency revision is pinned by CMake.

using namespace AscendC;
using ShortNnArgs = gemm_ex::ShortArgs;
__global__ __aicore__ __cube__ void gemmex_fallback_tt_stagger159_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR cGm, ShortNnArgs t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetHF32Mode(HF32Mode::DISABLE);
    gemm_ex::ShortTtTile<ShortNnArgs, true> tile(t);
    tile.Run(a, b, cGm);
}
void gemmex_fallback_tt_stagger159_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 127) / 128;
    const uint32_t grid = ((static_cast<uint32_t>(t.n) + 191) / 192) * columns;
    const auto q = gemm_ex::MakeShortArgs(t, static_cast<uint32_t>(t.ldc), columns);
    const uint32_t activeBlocks = blocks < grid ? blocks : grid;
    gemmex_fallback_tt_stagger159_kernel<<<activeBlocks, nullptr, stream>>>(a, b, c, q);
}

#endif
