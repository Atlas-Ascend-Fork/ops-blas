/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "cann_ops_blas_common.h"
#include "cher_tiling_data.h"
#include "common/helper/kernel_constant.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace {

// Fast-path shape/thread knobs aligned with aclblasCsyr arch35.
constexpr uint32_t CHER_FAST_MAX_N = 4096;
constexpr uint32_t CHER_SMALL_MATRIX_MAX_N = 512;
constexpr uint32_t CHER_SMALL_THREADS = 512;
constexpr uint32_t CHER_LARGE_THREADS = 1024;
constexpr uint32_t CHER_FAST_MAX_THREADS = CHER_LARGE_THREADS;
constexpr uint32_t CHER_TUNED_AIV_BLOCKS = 56;
// Tiny/UB path: keep reachable via rowsPerBlock>0. Cap at 16 on 950 — n=32 UB loses
// PF_1010 (GPU/NPU < 0.4) vs cyclic; Inf/NaN (TC_FL_108) still matches on cyclic.
constexpr uint32_t CHER_UB_MIN_N = 16;
constexpr uint32_t CHER_UB_COMPLEX = 4096;
constexpr uint32_t CHER_GROUPED_MAX_N = 1024;
constexpr uint32_t CHER_PAIRED_MAX_N = 2048;
constexpr uint32_t CHER_PAIR_ALIGNMENT = 8;

__aicore__ inline CherTilingData ResolveCherAlphaTiling(const CherTilingData& tiling, GM_ADDR alpha)
{
    CherTilingData local = tiling;
    if (local.alphaIsDevice != 0u && alpha != nullptr) {
        local.alpha = *reinterpret_cast<__gm__ const float*>(alpha);
        local.alphaIsDevice = 0u;
    }
    return local;
}
constexpr uint32_t CHER_TILE_SIDE_SHIFT = 6;
constexpr uint32_t CHER_TILE_SIDE = 1u << CHER_TILE_SIDE_SHIFT;
constexpr uint32_t CHER_TILE_COLUMN_SHIFT = 4;
constexpr uint32_t CHER_TILE_COLUMN_STRIDE = 1u << CHER_TILE_COLUMN_SHIFT;

template <bool UPLO_IS_UPPER>
__simt_callee__ inline void GetCherColRange(uint32_t row, uint32_t n, uint32_t& colStart, uint32_t& colEnd)
{
    if constexpr (UPLO_IS_UPPER) {
        colStart = row;
        colEnd = n;
    } else {
        colStart = 0;
        colEnd = row + 1;
    }
}

// A += alpha * x_row * conj(x_col); diagonal imag forced to 0.
__simt_callee__ inline __attribute__((always_inline)) void ApplyCherUpdate(
    uint32_t row, uint32_t col, float axReal, float axImag, float2 xCol, float2& aValue)
{
    aValue.x += (__fma(axReal, xCol.x, 0.0f) + __fma(axImag, xCol.y, 0.0f));
    aValue.y += (__fma(axImag, xCol.x, 0.0f) - __fma(axReal, xCol.y, 0.0f));
    if (row == col) {
        aValue.y = 0.0f;
    }
}

template <typename MatrixIndex>
__simt_callee__ inline __attribute__((always_inline)) void UpdateCherPair(
    uint32_t row, uint32_t col0, uint32_t col1, uint32_t lda, float axReal, float axImag,
    __gm__ const float2* __restrict xComplexGm, __gm__ float2* __restrict aComplexGm)
{
    float2 xCol0 = xComplexGm[col0];
    float2 xCol1 = xComplexGm[col1];
    MatrixIndex aIndex0 = static_cast<MatrixIndex>(col0) * static_cast<MatrixIndex>(lda) + row;
    MatrixIndex aIndex1 = static_cast<MatrixIndex>(col1) * static_cast<MatrixIndex>(lda) + row;
    float2 aValue0 = aComplexGm[aIndex0];
    float2 aValue1 = aComplexGm[aIndex1];
    ApplyCherUpdate(row, col0, axReal, axImag, xCol0, aValue0);
    ApplyCherUpdate(row, col1, axReal, axImag, xCol1, aValue1);
    aComplexGm[aIndex0] = aValue0;
    aComplexGm[aIndex1] = aValue1;
}

__simt_callee__ inline uint64_t XFloatIdx(uint32_t i, uint32_t n, int64_t incx)
{
    if (incx >= 0) {
        return static_cast<uint64_t>(i) * static_cast<uint64_t>(incx) * 2ULL;
    }
    return static_cast<uint64_t>(n - 1U - i) * static_cast<uint64_t>(-incx) * 2ULL;
}

// Keep alpha*(xr*yr+…) form so Inf/NaN matches cblas (TC_FL_108).
__simt_callee__ inline __attribute__((always_inline)) void ApplyCherElement(
    uint32_t row, uint32_t col, uint32_t lda, float alpha, float xr, float xi, float yr, float yi, __gm__ float* aGm)
{
    const uint64_t aBase = (static_cast<uint64_t>(col) * lda + row) * 2ULL;
    if (row == col) {
        aGm[aBase] = aGm[aBase] + alpha * (xr * xr + xi * xi);
        aGm[aBase + 1] = 0.0f;
    } else {
        aGm[aBase] = aGm[aBase] + alpha * (xr * yr + xi * yi);
        aGm[aBase + 1] = aGm[aBase + 1] + alpha * (xi * yr - xr * yi);
    }
}

// Scalar GM path.
template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CherGm(
    uint32_t n, uint32_t lda, float alpha, int64_t incx, __gm__ const float* xGm, __gm__ float* aGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        const uint64_t xRowBase = XFloatIdx(row, n, incx);
        const float xr = xGm[xRowBase];
        const float xi = xGm[xRowBase + 1];

        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        GetCherColRange<UPLO_IS_UPPER>(row, n, colStart, colEnd);
        for (uint32_t col = colStart; col < colEnd; ++col) {
            const uint64_t xColBase = XFloatIdx(col, n, incx);
            ApplyCherElement(row, col, lda, alpha, xr, xi, xGm[xColBase], xGm[xColBase + 1], aGm);
        }
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CherUb(
    uint32_t n, uint32_t lda, float alpha, __gm__ const float* xGm, __gm__ float* aGm, uint32_t rowStart,
    uint32_t rowEnd, uint32_t xBase, uint32_t xLen)
{
    __ubuf__ float xUb[CHER_UB_COMPLEX * 2];
    const uint32_t floatLen = xLen * 2;
    for (uint32_t i = threadIdx.x; i < floatLen; i += blockDim.x) {
        xUb[i] = xGm[static_cast<uint64_t>(xBase) * 2ULL + i];
    }
    asc_syncthreads();

    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        const uint32_t rowOff = (row - xBase) * 2;
        const float xr = xUb[rowOff];
        const float xi = xUb[rowOff + 1];

        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        GetCherColRange<UPLO_IS_UPPER>(row, n, colStart, colEnd);
        for (uint32_t col = colStart; col < colEnd; ++col) {
            const uint32_t colOff = (col - xBase) * 2;
            ApplyCherElement(row, col, lda, alpha, xr, xi, xUb[colOff], xUb[colOff + 1], aGm);
        }
    }
}

template <bool UPLO_IS_UPPER, typename MatrixIndex>
__simt_callee__ inline __attribute__((always_inline)) void UpdateCyclicRow(
    uint32_t n, uint32_t lda, uint32_t row, uint32_t coreCount, float2 axRow,
    __gm__ const float2* __restrict xComplexGm, __gm__ float2* __restrict aComplexGm)
{
    uint32_t firstColumn = blockIdx.x;
    if constexpr (UPLO_IS_UPPER) {
        if (row > firstColumn) {
            firstColumn += ((row - firstColumn + coreCount - 1u) / coreCount) * coreCount;
        }
    }
    uint32_t endColumn = UPLO_IS_UPPER ? n : row + 1u;
    uint32_t col = firstColumn;
    for (; col + coreCount < endColumn; col += 2u * coreCount) {
        UpdateCherPair<MatrixIndex>(row, col, col + coreCount, lda, axRow.x, axRow.y, xComplexGm, aComplexGm);
    }
    if (col < endColumn) {
        float2 xCol = xComplexGm[col];
        MatrixIndex aIndex = static_cast<MatrixIndex>(col) * static_cast<MatrixIndex>(lda) + row;
        float2 aValue = aComplexGm[aIndex];
        ApplyCherUpdate(row, col, axRow.x, axRow.y, xCol, aValue);
        aComplexGm[aIndex] = aValue;
    }
}

template <
    bool UPLO_IS_UPPER, bool ALPHA_IS_ONE, typename MatrixIndex = uint64_t, uint32_t FIXED_CORES = 0,
    uint32_t FIXED_THREADS = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHER_FAST_MAX_THREADS) inline void CherCyclicColumns(
    uint32_t n, uint32_t lda, float alpha, __gm__ const float* __restrict xGm, __gm__ float* __restrict aGm)
{
    const uint32_t rowStride = FIXED_THREADS != 0 ? FIXED_THREADS : blockDim.x;
    const uint32_t coreCount = FIXED_CORES != 0 ? FIXED_CORES : gridDim.x;
    auto* __restrict xComplexGm = reinterpret_cast<__gm__ const float2*>(xGm);
    auto* __restrict aComplexGm = reinterpret_cast<__gm__ float2*>(aGm);

    for (uint32_t row = threadIdx.x; row < n; row += rowStride) {
        float2 xRow = xComplexGm[row];
        float2 axRow;
        if constexpr (ALPHA_IS_ONE) {
            axRow = xRow;
        } else {
            axRow = make_float2(__fma(alpha, xRow.x, 0.0f), __fma(alpha, xRow.y, 0.0f));
        }
        UpdateCyclicRow<UPLO_IS_UPPER, MatrixIndex>(n, lda, row, coreCount, axRow, xComplexGm, aComplexGm);
    }
}

template <bool UPLO_IS_UPPER, bool ALPHA_IS_ONE>
__simt_callee__ inline __attribute__((always_inline)) void UpdateTileColumns(
    uint32_t n, uint32_t lda, uint32_t row, uint32_t colBase, uint32_t columnLane, bool diagonal, float axReal,
    float axImag, __gm__ const float2* __restrict xComplexGm, __gm__ float2* __restrict aComplexGm)
{
    uint32_t colEnd = colBase + CHER_TILE_SIDE < n ? colBase + CHER_TILE_SIDE : n;
    if constexpr (ALPHA_IS_ONE) {
        uint32_t col = colBase + columnLane;
        if constexpr (UPLO_IS_UPPER) {
            if (row > col) {
                col += ((row - col + CHER_TILE_COLUMN_STRIDE - 1u) >> CHER_TILE_COLUMN_SHIFT) << CHER_TILE_COLUMN_SHIFT;
            }
        } else {
            colEnd = colEnd < row + 1u ? colEnd : row + 1u;
        }
        for (; col + CHER_TILE_COLUMN_STRIDE < colEnd; col += 2u * CHER_TILE_COLUMN_STRIDE) {
            UpdateCherPair<uint32_t>(
                row, col, col + CHER_TILE_COLUMN_STRIDE, lda, axReal, axImag, xComplexGm, aComplexGm);
        }
        if (col < colEnd) {
            float2 xCol = xComplexGm[col];
            uint32_t aIndex = col * lda + row;
            float2 value = aComplexGm[aIndex];
            ApplyCherUpdate(row, col, axReal, axImag, xCol, value);
            aComplexGm[aIndex] = value;
        }
    } else {
        uint32_t col = colBase + columnLane;
        if (!diagonal) {
            for (; col + CHER_TILE_COLUMN_STRIDE < colEnd; col += 2u * CHER_TILE_COLUMN_STRIDE) {
                UpdateCherPair<uint32_t>(
                    row, col, col + CHER_TILE_COLUMN_STRIDE, lda, axReal, axImag, xComplexGm, aComplexGm);
            }
        }
        for (; col < colEnd; col += CHER_TILE_COLUMN_STRIDE) {
            if (UPLO_IS_UPPER ? row <= col : row >= col) {
                float2 xCol = xComplexGm[col];
                uint32_t aIndex = col * lda + row;
                float2 value = aComplexGm[aIndex];
                ApplyCherUpdate(row, col, axReal, axImag, xCol, value);
                aComplexGm[aIndex] = value;
            }
        }
    }
}

template <bool UPLO_IS_UPPER, bool ALPHA_IS_ONE>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHER_FAST_MAX_THREADS) inline void CherTriangleTiles(
    uint32_t n, uint32_t lda, uint32_t tileCount, float alpha, __gm__ const float* __restrict xGm,
    __gm__ float* __restrict aGm)
{
    auto* __restrict xComplexGm = reinterpret_cast<__gm__ const float2*>(xGm);
    auto* __restrict aComplexGm = reinterpret_cast<__gm__ float2*>(aGm);
    uint32_t rowLane = threadIdx.x & (CHER_TILE_SIDE - 1u);
    uint32_t columnLane = threadIdx.x >> CHER_TILE_SIDE_SHIFT;
    for (uint32_t tile = blockIdx.x; tile < tileCount; tile += CHER_TUNED_AIV_BLOCKS) {
        uint32_t tileCol = static_cast<uint32_t>((__sqrtf(static_cast<float>(8u * tile + 1u)) - 1.0f) * 0.5f);
        uint32_t triangleStart = tileCol * (tileCol + 1u) / 2u;
        if (triangleStart > tile) {
            --tileCol;
            triangleStart -= tileCol + 1u;
        }
        uint32_t tileRow = tile - triangleStart;
        uint32_t rowBase = (UPLO_IS_UPPER ? tileRow : tileCol) << CHER_TILE_SIDE_SHIFT;
        uint32_t colBase = (UPLO_IS_UPPER ? tileCol : tileRow) << CHER_TILE_SIDE_SHIFT;
        uint32_t row = rowBase + rowLane;
        if (row >= n) {
            continue;
        }
        float2 xRow = xComplexGm[row];
        float axReal;
        float axImag;
        if constexpr (ALPHA_IS_ONE) {
            axReal = xRow.x;
            axImag = xRow.y;
        } else {
            axReal = __fma(alpha, xRow.x, 0.0f);
            axImag = __fma(alpha, xRow.y, 0.0f);
        }
        UpdateTileColumns<UPLO_IS_UPPER, ALPHA_IS_ONE>(
            n, lda, row, colBase, columnLane, tileRow == tileCol, axReal, axImag, xComplexGm, aComplexGm);
    }
}

__aicore__ inline void CherFallbackToGm(
    const CherTilingData& tiling, __gm__ const float* xGm, __gm__ float* aGm)
{
    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<CherGm<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, tiling.incx, xGm, aGm);
    } else {
        asc_vf_call<CherGm<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, tiling.incx, xGm, aGm);
    }
}

template <typename MatrixIndex, uint32_t CORES = 0, uint32_t THREADS = 0>
__aicore__ inline __attribute__((always_inline)) void DispatchCherFast(
    const CherTilingData& tiling, __gm__ const float* xGm, __gm__ float* aGm)
{
    const bool alphaIsOne = tiling.n >= CHER_SMALL_MATRIX_MAX_N && tiling.alpha == 1.0f;
    if (tiling.uplo == ACLBLAS_UPPER) {
        if (alphaIsOne) {
            asc_vf_call<CherCyclicColumns<true, true, MatrixIndex, CORES, THREADS>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, xGm, aGm);
        } else {
            asc_vf_call<CherCyclicColumns<true, false, MatrixIndex, CORES, THREADS>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, xGm, aGm);
        }
    } else if (alphaIsOne) {
        asc_vf_call<CherCyclicColumns<false, true, MatrixIndex, CORES, THREADS>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, xGm, aGm);
    } else {
        asc_vf_call<CherCyclicColumns<false, false, MatrixIndex, CORES, THREADS>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, xGm, aGm);
    }
}

} // namespace

__aicore__ inline void CherDispatchUb(
    const CherTilingData& tiling, __gm__ const float* xGm, __gm__ float* aGm)
{
    const int32_t blkIdx = AscendC::GetBlockIdx();
    const uint32_t rowsPerBlk = tiling.rowsPerBlock;
    const uint32_t rowStart = static_cast<uint32_t>(blkIdx) * rowsPerBlk;
    const uint32_t rowEnd = (rowStart + rowsPerBlk < tiling.n) ? (rowStart + rowsPerBlk) : tiling.n;
    if (rowStart >= rowEnd) {
        return;
    }
    const uint32_t xBase = (tiling.uplo == ACLBLAS_UPPER) ? rowStart : 0;
    const uint32_t xLen = (tiling.uplo == ACLBLAS_UPPER) ? (tiling.n - rowStart) : rowEnd;
    if (xLen > CHER_UB_COMPLEX) {
        CherFallbackToGm(tiling, xGm, aGm);
        return;
    }
    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<CherUb<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, xGm, aGm, rowStart, rowEnd, xBase,
            xLen);
    } else {
        asc_vf_call<CherUb<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, xGm, aGm, rowStart, rowEnd, xBase,
            xLen);
    }
}

extern "C" __global__ __aicore__ void cher_kernel(GM_ADDR alpha, GM_ADDR x, GM_ADDR A, const CherTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const CherTilingData local = ResolveCherAlphaTiling(tiling, alpha);
    auto* __restrict xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* __restrict aGm = reinterpret_cast<__gm__ float*>(A);
    const bool pairAligned =
        ((reinterpret_cast<uint64_t>(xGm) | reinterpret_cast<uint64_t>(aGm)) & (CHER_PAIR_ALIGNMENT - 1u)) == 0u;

    // CherGm: non-unit stride / unaligned / out-of-range. Else cyclic (+ tiles via cher_kernel_do).
    // Extreme Inf/NaN vs cblas companion-component diffs are accepted in ST (TC_FL_108).
    if (!pairAligned || local.incx != 1 || local.n == 0 || local.n > CHER_FAST_MAX_N) {
        CherFallbackToGm(local, xGm, aGm);
        return;
    }
    if (local.n <= CHER_UB_MIN_N && local.rowsPerBlock > 0) {
        CherDispatchUb(local, xGm, aGm);
        return;
    }

    if (local.lda <= CHER_FAST_MAX_N && AscendC::GetBlockNum() == CHER_TUNED_AIV_BLOCKS &&
        local.numThreads == CHER_FAST_MAX_THREADS) {
        DispatchCherFast<uint32_t, CHER_TUNED_AIV_BLOCKS, CHER_FAST_MAX_THREADS>(local, xGm, aGm);
    } else if (local.lda <= CHER_FAST_MAX_N) {
        DispatchCherFast<uint32_t>(local, xGm, aGm);
    } else {
        DispatchCherFast<uint64_t>(local, xGm, aGm);
    }
}

extern "C" __global__ __aicore__ void cher_triangle_tiles(
    GM_ADDR alpha, GM_ADDR x, GM_ADDR A, const CherTilingData tiling, uint32_t tileCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const CherTilingData local = ResolveCherAlphaTiling(tiling, alpha);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* aGm = reinterpret_cast<__gm__ float*>(A);
    const bool unit = local.alpha == 1.0f;
    if (local.uplo == ACLBLAS_UPPER) {
        if (unit) {
            asc_vf_call<CherTriangleTiles<true, true>>(
                dim3{CHER_FAST_MAX_THREADS, 1, 1}, local.n, local.lda, tileCount, local.alpha, xGm, aGm);
        } else {
            asc_vf_call<CherTriangleTiles<true, false>>(
                dim3{CHER_FAST_MAX_THREADS, 1, 1}, local.n, local.lda, tileCount, local.alpha, xGm, aGm);
        }
    } else {
        if (unit) {
            asc_vf_call<CherTriangleTiles<false, true>>(
                dim3{CHER_FAST_MAX_THREADS, 1, 1}, local.n, local.lda, tileCount, local.alpha, xGm, aGm);
        } else {
            asc_vf_call<CherTriangleTiles<false, false>>(
                dim3{CHER_FAST_MAX_THREADS, 1, 1}, local.n, local.lda, tileCount, local.alpha, xGm, aGm);
        }
    }
}

void cher_kernel_do(
    uint8_t* alpha, uint8_t* x, uint8_t* A, const CherTilingData& tiling, uint32_t numBlocks, void* stream)
{
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(A)) & (CHER_PAIR_ALIGNMENT - 1u)) == 0u;
    // Tiles: (1024,1600] (helps TC_PF_1028) and (2048,4096]. (1600,2048] stays cyclic
    // (TC_PF_1136 LOWER n=2030 is faster there).
    const bool useTriangleTiles =
        (tiling.n > CHER_GROUPED_MAX_N && tiling.n <= 1600u) || (tiling.n > CHER_PAIRED_MAX_N);
    // Host cannot peek Device alpha without syncing; allow tile path when alphaIsDevice.
    const bool alphaMaybeNonzero = tiling.alphaIsDevice != 0u || tiling.alpha != 0.0f;
    if (aligned && tiling.incx == 1 && useTriangleTiles && tiling.n <= CHER_FAST_MAX_N &&
        tiling.lda <= CHER_FAST_MAX_N && numBlocks == CHER_TUNED_AIV_BLOCKS &&
        tiling.numThreads == CHER_FAST_MAX_THREADS && alphaMaybeNonzero) {
        uint32_t tileSide = (tiling.n + CHER_TILE_SIDE - 1u) >> CHER_TILE_SIDE_SHIFT;
        uint32_t tileCount = tileSide * (tileSide + 1u) / 2u;
        cher_triangle_tiles<<<numBlocks, nullptr, stream>>>(alpha, x, A, tiling, tileCount);
        return;
    }
    cher_kernel<<<numBlocks, nullptr, stream>>>(alpha, x, A, tiling);
}
