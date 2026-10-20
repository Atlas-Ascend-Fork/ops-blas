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
#include "common/helper/kernel_constant.h"
#include "csyr_kernel.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace {

// Keep the products independently rounded to FP32 before their sum/difference.
// A contracted a*b +/- c*d changes Inf/NaN propagation when finite products overflow.
// Write __fma(a, b, 0.0f) at the expression site to avoid helper overhead in Debug.

// Fast-path tuning range. Data is accessed in GM; this is not a UB allocation
// or an API limit. With both n and lda <= 4096, col * lda + row fits uint32_t.
constexpr uint32_t CSYR_FAST_MAX_N = 4096;
constexpr uint32_t CSYR_FLOATS_PER_COMPLEX = 2;
constexpr uint32_t CSYR_FAST_MAX_THREADS = CSYR_LARGE_THREADS;

// The Ascend 950PR tuning configuration used 56 AIV (vector) blocks. This is
// a specialization target, not the core count of every supported device.
// Host launches min(n, GetAivCoreCount()); both fixed-stride dispatch sites
// below require the actual grid to match. Other grids use runtime strides.
constexpr uint32_t CSYR_TUNED_AIV_BLOCKS = 56;

// Tiny matrices flatten a 32-row rectangle: low 5 bits are the row index.
constexpr uint32_t CSYR_TINY_ROW_SHIFT = 5;
constexpr uint32_t CSYR_TINY_ROW_COUNT = 1u << CSYR_TINY_ROW_SHIFT;
// A 64 x 64 tile uses 1024 threads: 64 row lanes x 16 column lanes.
// Each column lane visits columns 16 apart; paired loads advance by 32.
constexpr uint32_t CSYR_TILE_SIDE_SHIFT = 6;
constexpr uint32_t CSYR_TILE_SIDE = 1u << CSYR_TILE_SIDE_SHIFT;
constexpr uint32_t CSYR_TILE_COLUMN_SHIFT = 4;
constexpr uint32_t CSYR_TILE_COLUMN_STRIDE = 1u << CSYR_TILE_COLUMN_SHIFT;
static_assert(CSYR_TILE_SIDE * CSYR_TILE_COLUMN_STRIDE == CSYR_FAST_MAX_THREADS);

// Column-group tuning thresholds; larger shapes use the cyclic-column path.
constexpr uint32_t CSYR_GROUPED_MAX_N = 1024;
constexpr uint32_t CSYR_PAIRED_MAX_N = 2048;
constexpr uint32_t CSYR_DEFAULT_COLUMN_GROUPS = 4;
constexpr uint32_t CSYR_PAIRED_COLUMN_GROUPS = 2;
// float2 loads/stores need 8-byte alignment; the public complex ABI needs 4.
constexpr uint32_t CSYR_PAIR_ALIGNMENT = 8;

template <bool UPLO_IS_UPPER>
__simt_callee__ inline void GetCsyrColRange(uint32_t row, uint32_t n, uint32_t& colStart, uint32_t& colEnd)
{
    if constexpr (UPLO_IS_UPPER) {
        colStart = row;
        colEnd = n;
    } else {
        colStart = 0;
        colEnd = row + 1;
    }
}

__simt_callee__ inline void LoadComplex(__gm__ const float* source, int64_t complexIndex, float& real, float& imag)
{
    int64_t scalarIndex = complexIndex * CSYR_FLOATS_PER_COMPLEX;
    real = source[scalarIndex];
    imag = source[scalarIndex + 1];
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CsyrGm(
    uint32_t n, uint32_t lda, float alphaReal, float alphaImag, int64_t incx, __gm__ const float* alphaGm,
    uint32_t alphaIsDevice, __gm__ const float* xGm, __gm__ float* aGm)
{
    if (alphaIsDevice != 0u) {
        alphaReal = alphaGm[0];
        alphaImag = alphaGm[1];
    }
    if (alphaReal == 0.0f && alphaImag == 0.0f) {
        return;
    }

    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        int64_t rowIndex = (incx >= 0) ? static_cast<int64_t>(row) * incx : static_cast<int64_t>(n - 1 - row) * (-incx);
        float xRowReal;
        float xRowImag;
        LoadComplex(xGm, rowIndex, xRowReal, xRowImag);

        float axRowReal = (__fma(alphaReal, xRowReal, 0.0f) - __fma(alphaImag, xRowImag, 0.0f));
        float axRowImag = (__fma(alphaReal, xRowImag, 0.0f) + __fma(alphaImag, xRowReal, 0.0f));

        uint32_t colStart;
        uint32_t colEnd;
        GetCsyrColRange<UPLO_IS_UPPER>(row, n, colStart, colEnd);
        for (uint32_t col = colStart; col < colEnd; ++col) {
            int64_t colIndex =
                (incx >= 0) ? static_cast<int64_t>(col) * incx : static_cast<int64_t>(n - 1 - col) * (-incx);
            float xColReal;
            float xColImag;
            LoadComplex(xGm, colIndex, xColReal, xColImag);

            float updateReal = (__fma(axRowReal, xColReal, 0.0f) - __fma(axRowImag, xColImag, 0.0f));
            float updateImag = (__fma(axRowReal, xColImag, 0.0f) + __fma(axRowImag, xColReal, 0.0f));
            uint64_t aIndex = (static_cast<uint64_t>(col) * static_cast<uint64_t>(lda) + row) * CSYR_FLOATS_PER_COMPLEX;
            aGm[aIndex] += updateReal;
            aGm[aIndex + 1] += updateImag;
        }
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(CSYR_FAST_MAX_THREADS) inline void CsyrTinyUnitTriangle(
    uint32_t n, __gm__ const float* __restrict xGm, __gm__ float* __restrict aGm)
{
    auto* __restrict xComplexGm = reinterpret_cast<__gm__ const float2*>(xGm);
    auto* __restrict aComplexGm = reinterpret_cast<__gm__ float2*>(aGm);
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = index & (CSYR_TINY_ROW_COUNT - 1u);
    uint32_t col = index >> CSYR_TINY_ROW_SHIFT;
    if (row < n && col < n && (UPLO_IS_UPPER ? row <= col : row >= col)) {
        float2 xRow = xComplexGm[row];
        float2 xCol = xComplexGm[col];
        uint32_t aIndex = col * n + row;
        float2 aValue = aComplexGm[aIndex];
        float axReal = xRow.x - 0.0f * xRow.y;
        float axImag = xRow.y + 0.0f * xRow.x;
        aValue.x += (__fma(axReal, xCol.x, 0.0f) - __fma(axImag, xCol.y, 0.0f));
        aValue.y += (__fma(axReal, xCol.y, 0.0f) + __fma(axImag, xCol.x, 0.0f));
        aComplexGm[aIndex] = aValue;
    }
}

template <typename MatrixIndex>
__simt_callee__ inline __attribute__((always_inline)) void UpdateCsyrPair(
    uint32_t row, uint32_t col, uint32_t nextCol, uint32_t lda, float axReal, float axImag,
    __gm__ const float2* __restrict xComplexGm, __gm__ float2* __restrict aComplexGm)
{
    float2 xCol0 = xComplexGm[col];
    float2 xCol1 = xComplexGm[nextCol];
    MatrixIndex aIndex0 = static_cast<MatrixIndex>(col) * static_cast<MatrixIndex>(lda) + row;
    MatrixIndex aIndex1 = static_cast<MatrixIndex>(nextCol) * static_cast<MatrixIndex>(lda) + row;
    float2 aValue0 = aComplexGm[aIndex0];
    float2 aValue1 = aComplexGm[aIndex1];
    aValue0.x += __fma(axReal, xCol0.x, 0.0f) - __fma(axImag, xCol0.y, 0.0f);
    aValue0.y += __fma(axReal, xCol0.y, 0.0f) + __fma(axImag, xCol0.x, 0.0f);
    aValue1.x += __fma(axReal, xCol1.x, 0.0f) - __fma(axImag, xCol1.y, 0.0f);
    aValue1.y += __fma(axReal, xCol1.y, 0.0f) + __fma(axImag, xCol1.x, 0.0f);
    aComplexGm[aIndex0] = aValue0;
    aComplexGm[aIndex1] = aValue1;
}

template <bool UPLO_IS_UPPER, bool ALPHA_IS_ONE>
__simt_callee__ inline __attribute__((always_inline)) void UpdateTileColumns(
    uint32_t n, uint32_t lda, uint32_t row, uint32_t colBase, uint32_t columnLane, bool diagonal, float axReal,
    float axImag, __gm__ const float2* __restrict xComplexGm, __gm__ float2* __restrict aComplexGm)
{
    uint32_t colEnd = colBase + CSYR_TILE_SIDE < n ? colBase + CSYR_TILE_SIDE : n;
    if constexpr (ALPHA_IS_ONE) {
        uint32_t col = colBase + columnLane;
        if constexpr (UPLO_IS_UPPER) {
            if (row > col) {
                col += ((row - col + CSYR_TILE_COLUMN_STRIDE - 1u) >> CSYR_TILE_COLUMN_SHIFT) << CSYR_TILE_COLUMN_SHIFT;
            }
        } else {
            colEnd = colEnd < row + 1u ? colEnd : row + 1u;
        }
        {
            for (; col + CSYR_TILE_COLUMN_STRIDE < colEnd; col += 2u * CSYR_TILE_COLUMN_STRIDE) {
                UpdateCsyrPair<uint32_t>(
                    row, col, col + CSYR_TILE_COLUMN_STRIDE, lda, axReal, axImag, xComplexGm, aComplexGm);
            }
        }
        if (col < colEnd) {
            {
                float2 xCol = xComplexGm[col];
                uint32_t aIndex = col * lda + row;
                float2 value = aComplexGm[aIndex];
                value.x += __fma(axReal, xCol.x, 0.0f) - __fma(axImag, xCol.y, 0.0f);
                value.y += __fma(axReal, xCol.y, 0.0f) + __fma(axImag, xCol.x, 0.0f);
                aComplexGm[aIndex] = value;
            }
        }
    } else {
        uint32_t col = colBase + columnLane;
        if (!diagonal) {
            for (; col + CSYR_TILE_COLUMN_STRIDE < colEnd; col += 2u * CSYR_TILE_COLUMN_STRIDE) {
                UpdateCsyrPair<uint32_t>(
                    row, col, col + CSYR_TILE_COLUMN_STRIDE, lda, axReal, axImag, xComplexGm, aComplexGm);
            }
        }
        for (; col < colEnd; col += CSYR_TILE_COLUMN_STRIDE) {
            if (UPLO_IS_UPPER ? row <= col : row >= col) {
                float2 xCol = xComplexGm[col];
                uint32_t aIndex = col * lda + row;
                float2 value = aComplexGm[aIndex];
                value.x += __fma(axReal, xCol.x, 0.0f) - __fma(axImag, xCol.y, 0.0f);
                value.y += __fma(axReal, xCol.y, 0.0f) + __fma(axImag, xCol.x, 0.0f);
                aComplexGm[aIndex] = value;
            }
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
    // Fetch two independent columns before arithmetic to overlap the
    // GM reads. Every element retains the original FP32 operation order.
    for (; col + coreCount < endColumn; col += 2u * coreCount) {
        UpdateCsyrPair<MatrixIndex>(row, col, col + coreCount, lda, axRow.x, axRow.y, xComplexGm, aComplexGm);
    }
    if (col < endColumn) {
        float2 xCol = xComplexGm[col];
        MatrixIndex aIndex = static_cast<MatrixIndex>(col) * static_cast<MatrixIndex>(lda) + row;
        float2 aValue = aComplexGm[aIndex];
        aValue.x += (__fma(axRow.x, xCol.x, 0.0f) - __fma(axRow.y, xCol.y, 0.0f));
        aValue.y += (__fma(axRow.x, xCol.y, 0.0f) + __fma(axRow.y, xCol.x, 0.0f));
        aComplexGm[aIndex] = aValue;
    }
}

template <bool UPLO_IS_UPPER, bool ALPHA_IS_ONE>
__simt_vf__ __aicore__ LAUNCH_BOUND(CSYR_FAST_MAX_THREADS) inline void CsyrTriangleTiles(
    uint32_t n, uint32_t lda, uint32_t tileCount, float alphaReal, float alphaImag, __gm__ const float* __restrict xGm,
    __gm__ float* __restrict aGm)
{
    auto* __restrict xComplexGm = reinterpret_cast<__gm__ const float2*>(xGm);
    auto* __restrict aComplexGm = reinterpret_cast<__gm__ float2*>(aGm);
    uint32_t rowLane = threadIdx.x & (CSYR_TILE_SIDE - 1u);
    uint32_t columnLane = threadIdx.x >> CSYR_TILE_SIDE_SHIFT;
    for (uint32_t tile = blockIdx.x; tile < tileCount; tile += CSYR_TUNED_AIV_BLOCKS) {
        // Invert the triangular tile number; correction makes the integer
        // mapping exact even if the floating square root rounds upward.
        uint32_t tileCol = static_cast<uint32_t>((__sqrtf(static_cast<float>(8u * tile + 1u)) - 1.0f) * 0.5f);
        uint32_t triangleStart = tileCol * (tileCol + 1u) / 2u;
        if (triangleStart > tile) {
            --tileCol;
            triangleStart -= tileCol + 1u;
        }
        uint32_t tileRow = tile - triangleStart;
        uint32_t rowBase = (UPLO_IS_UPPER ? tileRow : tileCol) << CSYR_TILE_SIDE_SHIFT;
        uint32_t colBase = (UPLO_IS_UPPER ? tileCol : tileRow) << CSYR_TILE_SIDE_SHIFT;
        uint32_t row = rowBase + rowLane;
        if (row >= n) {
            continue;
        }
        float2 xRow = xComplexGm[row];
        float axReal;
        float axImag;
        if constexpr (ALPHA_IS_ONE) {
            axReal = xRow.x - 0.0f * xRow.y;
            axImag = xRow.y + 0.0f * xRow.x;
        } else {
            axReal = __fma(alphaReal, xRow.x, 0.0f) - __fma(alphaImag, xRow.y, 0.0f);
            axImag = __fma(alphaReal, xRow.y, 0.0f) + __fma(alphaImag, xRow.x, 0.0f);
        }
        UpdateTileColumns<UPLO_IS_UPPER, ALPHA_IS_ONE>(
            n, lda, row, colBase, columnLane, tileRow == tileCol, axReal, axImag, xComplexGm, aComplexGm);
    }
}

template <
    bool UPLO_IS_UPPER, bool ALPHA_IS_ONE, typename MatrixIndex = uint64_t, uint32_t FIXED_CORES = 0,
    uint32_t FIXED_THREADS = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(CSYR_FAST_MAX_THREADS) inline void CsyrCyclicColumns(
    uint32_t n, uint32_t lda, float alphaReal, float alphaImag, __gm__ const float* alphaGm, uint32_t alphaIsDevice,
    __gm__ const float* __restrict xGm, __gm__ float* __restrict aGm)
{
    const uint32_t rowStride = FIXED_THREADS != 0 ? FIXED_THREADS : blockDim.x;
    const uint32_t coreCount = FIXED_CORES != 0 ? FIXED_CORES : gridDim.x;
    auto* __restrict xComplexGm = reinterpret_cast<__gm__ const float2*>(xGm);
    auto* __restrict aComplexGm = reinterpret_cast<__gm__ float2*>(aGm);
    if constexpr (!ALPHA_IS_ONE) {
        if (alphaIsDevice != 0u) {
            alphaReal = alphaGm[0];
            alphaImag = alphaGm[1];
        }
        if (alphaReal == 0.0f && alphaImag == 0.0f) {
            return;
        }
    }

    // Each core owns the same cyclic columns as before. Reuse a row value
    // across those columns instead of reloading alpha*x for every matrix element.
    for (uint32_t row = threadIdx.x; row < n; row += rowStride) {
        float2 axRow;
        if constexpr (ALPHA_IS_ONE) {
            float2 xRow = xComplexGm[row];
            axRow = make_float2(xRow.x - 0.0f * xRow.y, xRow.y + 0.0f * xRow.x);
        } else {
            float2 xRow = xComplexGm[row];
            axRow = make_float2(
                __fma(alphaReal, xRow.x, 0.0f) - __fma(alphaImag, xRow.y, 0.0f),
                __fma(alphaReal, xRow.y, 0.0f) + __fma(alphaImag, xRow.x, 0.0f));
        }
        UpdateCyclicRow<UPLO_IS_UPPER, MatrixIndex>(n, lda, row, coreCount, axRow, xComplexGm, aComplexGm);
    }
}

template <bool UPLO_IS_UPPER, uint32_t COLUMN_GROUPS, typename MatrixIndex = uint64_t, uint32_t FIXED_CORES = 0>
__simt_vf__ __aicore__ LAUNCH_BOUND(CSYR_FAST_MAX_THREADS) inline void CsyrUnitAlphaGroupedColumns(
    uint32_t n, uint32_t lda, __gm__ const float* __restrict xGm, __gm__ float* __restrict aGm)
{
    const uint32_t coreCount = FIXED_CORES != 0 ? FIXED_CORES : gridDim.x;
    constexpr uint32_t groupThreads = CSYR_FAST_MAX_THREADS / COLUMN_GROUPS;
    auto* __restrict xComplexGm = reinterpret_cast<__gm__ const float2*>(xGm);
    auto* __restrict aComplexGm = reinterpret_cast<__gm__ float2*>(aGm);
    uint32_t columnGroup = threadIdx.x / groupThreads;
    uint32_t rowLane = threadIdx.x - columnGroup * groupThreads;
    uint32_t columnStride = coreCount * COLUMN_GROUPS;

    // Keep each lane group's cyclic column ownership, but retain alpha*x
    // in registers while visiting all columns for the lane's current row.
    for (uint32_t row = rowLane; row < n; row += groupThreads) {
        float2 xRow = xComplexGm[row];
        float axReal = xRow.x - 0.0f * xRow.y;
        float axImag = xRow.y + 0.0f * xRow.x;
        uint32_t firstColumn = blockIdx.x * COLUMN_GROUPS + columnGroup;
        if constexpr (UPLO_IS_UPPER) {
            if (row > firstColumn) {
                firstColumn += ((row - firstColumn + columnStride - 1u) / columnStride) * columnStride;
            }
        }
        uint32_t endColumn = UPLO_IS_UPPER ? n : row + 1u;
        uint32_t col = firstColumn;
        // Fetch two independent columns before arithmetic to overlap the
        // GM reads. Every element retains the original FP32 operation order.
        for (; col + columnStride < endColumn; col += 2u * columnStride) {
            uint32_t nextCol = col + columnStride;
            float2 xCol0 = xComplexGm[col];
            float2 xCol1 = xComplexGm[nextCol];
            MatrixIndex aIndex0 = static_cast<MatrixIndex>(col) * static_cast<MatrixIndex>(lda) + row;
            MatrixIndex aIndex1 = static_cast<MatrixIndex>(nextCol) * static_cast<MatrixIndex>(lda) + row;
            float2 aValue0 = aComplexGm[aIndex0];
            float2 aValue1 = aComplexGm[aIndex1];
            aValue0.x += (__fma(axReal, xCol0.x, 0.0f) - __fma(axImag, xCol0.y, 0.0f));
            aValue0.y += (__fma(axReal, xCol0.y, 0.0f) + __fma(axImag, xCol0.x, 0.0f));
            aValue1.x += (__fma(axReal, xCol1.x, 0.0f) - __fma(axImag, xCol1.y, 0.0f));
            aValue1.y += (__fma(axReal, xCol1.y, 0.0f) + __fma(axImag, xCol1.x, 0.0f));
            aComplexGm[aIndex0] = aValue0;
            aComplexGm[aIndex1] = aValue1;
        }
        if (col < endColumn) {
            float2 xCol = xComplexGm[col];
            MatrixIndex aIndex = static_cast<MatrixIndex>(col) * static_cast<MatrixIndex>(lda) + row;
            float2 aValue = aComplexGm[aIndex];
            aValue.x += (__fma(axReal, xCol.x, 0.0f) - __fma(axImag, xCol.y, 0.0f));
            aValue.y += (__fma(axReal, xCol.y, 0.0f) + __fma(axImag, xCol.x, 0.0f));
            aComplexGm[aIndex] = aValue;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CSYR_FAST_MAX_THREADS) inline void CsyrUnitAlphaLowerTriangle1024(
    __gm__ const float* __restrict xGm, __gm__ float* __restrict aGm)
{
    constexpr uint32_t matrixShift = 10; // log2(1024), the exact-size lower-triangle specialization.
    constexpr uint32_t matrixSize = 1u << matrixShift;
    constexpr uint32_t halfSize = matrixSize / 2;
    constexpr uint32_t rectangleSize = matrixSize * halfSize;
    auto* __restrict xComplexGm = reinterpret_cast<__gm__ const float2*>(xGm);
    auto* __restrict aComplexGm = reinterpret_cast<__gm__ float2*>(aGm);
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < rectangleSize; index += stride) {
        uint32_t col = index >> matrixShift;
        uint32_t row = index & (matrixSize - 1);
        if (row < col) {
            row = matrixSize - 1 - row;
            col = matrixSize - 1 - col;
        }
        float2 xCol = xComplexGm[col];
        float2 xRow = xComplexGm[row];
        uint32_t aIndex = (col << matrixShift) + row;
        float2 aValue = aComplexGm[aIndex];
        float axReal = xRow.x - 0.0f * xRow.y;
        float axImag = xRow.y + 0.0f * xRow.x;
        aValue.x += (__fma(axReal, xCol.x, 0.0f) - __fma(axImag, xCol.y, 0.0f));
        aValue.y += (__fma(axReal, xCol.y, 0.0f) + __fma(axImag, xCol.x, 0.0f));
        aComplexGm[aIndex] = aValue;
    }
    uint32_t diagonal = blockIdx.x + threadIdx.x * gridDim.x;
    if (diagonal < halfSize) {
        uint32_t row = halfSize + diagonal;
        float2 xRow = xComplexGm[row];
        uint32_t aIndex = (row << matrixShift) + row;
        float2 aValue = aComplexGm[aIndex];
        float axReal = xRow.x - 0.0f * xRow.y;
        float axImag = xRow.y + 0.0f * xRow.x;
        aValue.x += (__fma(axReal, xRow.x, 0.0f) - __fma(axImag, xRow.y, 0.0f));
        aValue.y += (__fma(axReal, xRow.y, 0.0f) + __fma(axImag, xRow.x, 0.0f));
        aComplexGm[aIndex] = aValue;
    }
}

template <uint32_t SIZE_SHIFT>
__simt_vf__ __aicore__ LAUNCH_BOUND(CSYR_FAST_MAX_THREADS) inline void CsyrUnitAlphaUpperTriangle(
    __gm__ const float* __restrict xGm, __gm__ float* __restrict aGm)
{
    constexpr uint32_t matrixSize = 1u << SIZE_SHIFT;
    constexpr uint32_t halfSize = matrixSize / 2;
    constexpr uint32_t rectangleSize = matrixSize * halfSize;
    auto* __restrict xComplexGm = reinterpret_cast<__gm__ const float2*>(xGm);
    auto* __restrict aComplexGm = reinterpret_cast<__gm__ float2*>(aGm);
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < rectangleSize; index += stride) {
        uint32_t col = halfSize + (index >> SIZE_SHIFT);
        uint32_t row = index & (matrixSize - 1);
        if (row > col) {
            row = matrixSize - 1 - row;
            col = matrixSize - 1 - col;
        }
        float2 xCol = xComplexGm[col];
        float2 xRow = xComplexGm[row];
        uint32_t aIndex = (col << SIZE_SHIFT) + row;
        float2 aValue = aComplexGm[aIndex];
        float axReal = xRow.x - 0.0f * xRow.y;
        float axImag = xRow.y + 0.0f * xRow.x;
        aValue.x += (__fma(axReal, xCol.x, 0.0f) - __fma(axImag, xCol.y, 0.0f));
        aValue.y += (__fma(axReal, xCol.y, 0.0f) + __fma(axImag, xCol.x, 0.0f));
        aComplexGm[aIndex] = aValue;
    }
    uint32_t row = blockIdx.x + threadIdx.x * gridDim.x;
    if (row < halfSize) {
        float2 xRow = xComplexGm[row];
        uint32_t aIndex = (row << SIZE_SHIFT) + row;
        float2 aValue = aComplexGm[aIndex];
        float axReal = xRow.x - 0.0f * xRow.y;
        float axImag = xRow.y + 0.0f * xRow.x;
        aValue.x += (__fma(axReal, xRow.x, 0.0f) - __fma(axImag, xRow.y, 0.0f));
        aValue.y += (__fma(axReal, xRow.y, 0.0f) + __fma(axImag, xRow.x, 0.0f));
        aComplexGm[aIndex] = aValue;
    }
}

__aicore__ inline void CsyrFallbackToGm(
    const CsyrTilingData& tiling, __gm__ const float* alphaGm, __gm__ const float* xGm, __gm__ float* aGm)
{
    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<CsyrGm<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, tiling.incx,
            alphaGm, tiling.alphaIsDevice, xGm, aGm);
    } else {
        asc_vf_call<CsyrGm<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, tiling.incx,
            alphaGm, tiling.alphaIsDevice, xGm, aGm);
    }
}

template <bool UPPER, typename MatrixIndex, uint32_t CORES, uint32_t THREADS, uint32_t GROUPS>
__aicore__ inline __attribute__((always_inline)) void DispatchCsyrColumns(
    const CsyrTilingData& tiling, bool alphaIsOne, bool useGroupedColumns, bool usePairedColumns,
    __gm__ const float* alphaGm, __gm__ const float* xGm, __gm__ float* aGm)
{
    if (useGroupedColumns) {
        asc_vf_call<CsyrUnitAlphaGroupedColumns<UPPER, GROUPS, MatrixIndex, CORES>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, xGm, aGm);
    } else if (usePairedColumns) {
        asc_vf_call<CsyrUnitAlphaGroupedColumns<UPPER, CSYR_PAIRED_COLUMN_GROUPS, MatrixIndex, CORES>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, xGm, aGm);
    } else if (alphaIsOne) {
        asc_vf_call<CsyrCyclicColumns<UPPER, true, MatrixIndex, CORES, THREADS>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, alphaGm,
            tiling.alphaIsDevice, xGm, aGm);
    } else {
        asc_vf_call<CsyrCyclicColumns<UPPER, false, MatrixIndex, CORES, THREADS>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, alphaGm,
            tiling.alphaIsDevice, xGm, aGm);
    }
}

template <typename MatrixIndex, uint32_t CORES = 0, uint32_t THREADS = 0, uint32_t GROUPS = CSYR_DEFAULT_COLUMN_GROUPS>
__aicore__ inline __attribute__((always_inline)) void DispatchCsyrFast(
    const CsyrTilingData& tiling, __gm__ const float* alphaGm, __gm__ const float* xGm, __gm__ float* aGm)
{
    const bool alphaIsOne = tiling.n >= CSYR_SMALL_MATRIX_MAX_N && tiling.alphaIsDevice == 0u &&
                            tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f;
    const bool useGroupedColumns = alphaIsOne && tiling.n > CSYR_SMALL_MATRIX_MAX_N && tiling.n <= CSYR_GROUPED_MAX_N;
    const bool usePairedColumns = alphaIsOne & (tiling.n > CSYR_GROUPED_MAX_N) & (tiling.n <= CSYR_PAIRED_MAX_N);
    if (tiling.uplo == ACLBLAS_UPPER) {
        // Exact packed shapes: template shifts 9 and 11 mean n = lda = 512 and 2048.
        if (alphaIsOne && ((tiling.n == CSYR_SMALL_MATRIX_MAX_N) & (tiling.lda == CSYR_SMALL_MATRIX_MAX_N))) {
            asc_vf_call<CsyrUnitAlphaUpperTriangle<9>>(dim3{tiling.numThreads, 1, 1}, xGm, aGm);
        } else if (usePairedColumns && ((tiling.n == CSYR_PAIRED_MAX_N) & (tiling.lda == CSYR_PAIRED_MAX_N))) {
            asc_vf_call<CsyrUnitAlphaUpperTriangle<11>>(dim3{tiling.numThreads, 1, 1}, xGm, aGm);
        } else {
            DispatchCsyrColumns<true, MatrixIndex, CORES, THREADS, GROUPS>(
                tiling, alphaIsOne, useGroupedColumns, usePairedColumns, alphaGm, xGm, aGm);
        }
    } else if (useGroupedColumns && ((tiling.n == CSYR_GROUPED_MAX_N) & (tiling.lda == CSYR_GROUPED_MAX_N))) {
        asc_vf_call<CsyrUnitAlphaLowerTriangle1024>(dim3{tiling.numThreads, 1, 1}, xGm, aGm);
    } else {
        DispatchCsyrColumns<false, MatrixIndex, CORES, THREADS, GROUPS>(
            tiling, alphaIsOne, useGroupedColumns, usePairedColumns, alphaGm, xGm, aGm);
    }
}

} // namespace

extern "C" __global__ __aicore__ void csyr_kernel(GM_ADDR alpha, GM_ADDR x, GM_ADDR A, const CsyrTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* alphaGm = reinterpret_cast<__gm__ const float*>(alpha);
    auto* __restrict xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* __restrict aGm = reinterpret_cast<__gm__ float*>(A);
    // float2 accesses require 8-byte alignment; the public complex type needs only 4.
    const bool pairAligned =
        ((reinterpret_cast<uint64_t>(xGm) | reinterpret_cast<uint64_t>(aGm)) & (CSYR_PAIR_ALIGNMENT - 1u)) == 0u;
    if (pairAligned && tiling.n > 0 && tiling.n < CSYR_TINY_ROW_COUNT && tiling.lda == tiling.n && tiling.incx == 1 &&
        tiling.alphaIsDevice == 0u && tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f) {
        if (tiling.uplo == ACLBLAS_UPPER) {
            asc_vf_call<CsyrTinyUnitTriangle<true>>(dim3{tiling.numThreads, 1, 1}, tiling.n, xGm, aGm);
        } else {
            asc_vf_call<CsyrTinyUnitTriangle<false>>(dim3{tiling.numThreads, 1, 1}, tiling.n, xGm, aGm);
        }
        return;
    }
    if (!pairAligned || tiling.incx != 1 || tiling.n < CSYR_TINY_ROW_COUNT || tiling.n > CSYR_FAST_MAX_N) {
        CsyrFallbackToGm(tiling, alphaGm, xGm, aGm);
        return;
    }
    // Fast dispatch guarantees n <= 4096; bound lda before narrowing indices.
    // Use the tuned AIV grid only when the actual blocks and threads match.
    // The generic instantiations below read gridDim.x and blockDim.x at runtime.
    if (tiling.lda <= CSYR_FAST_MAX_N && AscendC::GetBlockNum() == CSYR_TUNED_AIV_BLOCKS &&
        tiling.numThreads == CSYR_FAST_MAX_THREADS) {
        DispatchCsyrFast<uint32_t, CSYR_TUNED_AIV_BLOCKS, CSYR_FAST_MAX_THREADS, CSYR_PAIRED_COLUMN_GROUPS>(
            tiling, alphaGm, xGm, aGm);
    } else if (tiling.lda <= CSYR_FAST_MAX_N) {
        DispatchCsyrFast<uint32_t>(tiling, alphaGm, xGm, aGm);
    } else {
        DispatchCsyrFast<uint64_t>(tiling, alphaGm, xGm, aGm);
    }
}

extern "C" __global__ __aicore__ void csyr_triangle_tiles(
    GM_ADDR x, GM_ADDR A, const CsyrTilingData tiling, uint32_t tileCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* aGm = reinterpret_cast<__gm__ float*>(A);
    bool unit = tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f;
    if (tiling.uplo == ACLBLAS_UPPER) {
        if (unit) {
            asc_vf_call<CsyrTriangleTiles<true, true>>(
                dim3{CSYR_FAST_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, tileCount, tiling.alphaReal, tiling.alphaImag,
                xGm, aGm);
        } else {
            asc_vf_call<CsyrTriangleTiles<true, false>>(
                dim3{CSYR_FAST_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, tileCount, tiling.alphaReal, tiling.alphaImag,
                xGm, aGm);
        }
    } else {
        if (unit) {
            asc_vf_call<CsyrTriangleTiles<false, true>>(
                dim3{CSYR_FAST_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, tileCount, tiling.alphaReal, tiling.alphaImag,
                xGm, aGm);
        } else {
            asc_vf_call<CsyrTriangleTiles<false, false>>(
                dim3{CSYR_FAST_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, tileCount, tiling.alphaReal, tiling.alphaImag,
                xGm, aGm);
        }
    }
}

void csyr_kernel_do(
    uint8_t* alpha, uint8_t* x, uint8_t* A, const CsyrTilingData& tiling, uint32_t numBlocks, void* stream)
{
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(A)) & (CSYR_PAIR_ALIGNMENT - 1u)) == 0u;
    // The tile loop uses CSYR_TUNED_AIV_BLOCKS as its stride; keep this
    // launch guard paired with it. Other grids fall back to csyr_kernel.
    if (aligned && tiling.incx == 1 && tiling.n > CSYR_GROUPED_MAX_N && tiling.n <= CSYR_FAST_MAX_N &&
        tiling.lda <= CSYR_FAST_MAX_N && tiling.alphaIsDevice == 0u && numBlocks == CSYR_TUNED_AIV_BLOCKS &&
        tiling.numThreads == CSYR_FAST_MAX_THREADS && !(tiling.alphaReal == 0.0f && tiling.alphaImag == 0.0f)) {
        uint32_t tileSide = (tiling.n + CSYR_TILE_SIDE - 1u) >> CSYR_TILE_SIDE_SHIFT;
        uint32_t tileCount = tileSide * (tileSide + 1u) / 2u;
        csyr_triangle_tiles<<<numBlocks, nullptr, stream>>>(x, A, tiling, tileCount);
        return;
    }
    csyr_kernel<<<numBlocks, nullptr, stream>>>(alpha, x, A, tiling);
}
