/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cher2_kernel.h"
#include "cher2_aiv_kernel.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace {
constexpr uint32_t TILE_SIDE = 64;
constexpr uint32_t THREADS = 1024;
constexpr uint32_t COLUMN_LANES = THREADS / TILE_SIDE;

__simt_callee__ inline int64_t VectorIndex(uint32_t i, uint32_t n, int64_t incx)
{
    return incx > 0 ? static_cast<int64_t>(i) * incx : static_cast<int64_t>(n - 1 - i) * (-incx);
}

template <bool PAIRED, typename Index>
__simt_callee__ inline __attribute__((always_inline)) float2 LoadComplex(__gm__ const float* p, Index offset)
{
    if constexpr (PAIRED) {
        return reinterpret_cast<__gm__ const float2*>(p)[offset / 2];
    } else {
        return {p[offset], p[offset + 1]};
    }
}

template <bool PAIRED, typename Index>
__simt_callee__ inline __attribute__((always_inline)) void StoreComplex(__gm__ float* p, Index offset, float2 value)
{
    if constexpr (PAIRED) {
        reinterpret_cast<__gm__ float2*>(p)[offset / 2] = value;
    } else {
        p[offset] = value.x;
        p[offset + 1] = value.y;
    }
}

template <bool UPPER, typename Index, bool UNIT_STRIDE, bool PAIRED, bool DIAGONAL>
__simt_callee__ inline __attribute__((always_inline)) void UpdateColumns(
    uint32_t n, uint32_t lda, float ar, float ai, int64_t incx, int64_t incy, uint32_t row, uint32_t colBase,
    uint32_t colLane, __gm__ const float* x, __gm__ const float* y, __gm__ float* a)
{
    const Index xi = UNIT_STRIDE ? 2 * row : static_cast<Index>(2 * VectorIndex(row, n, incx));
    const float2 xRow = LoadComplex<PAIRED>(x, xi);
    const float xr = xRow.x, xx = xRow.y;
    const Index yiRow = UNIT_STRIDE ? 2 * row : static_cast<Index>(2 * VectorIndex(row, n, incy));
    const float2 yRow = LoadComplex<PAIRED>(y, yiRow);
    const float yr = yRow.x, yy = yRow.y;
    for (uint32_t lane = colLane; lane < TILE_SIDE; lane += COLUMN_LANES) {
        const uint32_t col = colBase + lane;
        if (col >= n || (DIAGONAL && (UPPER ? row > col : row < col))) {
            continue;
        }
        const Index xj = UNIT_STRIDE ? 2 * col : static_cast<Index>(2 * VectorIndex(col, n, incx));
        const Index yj = UNIT_STRIDE ? 2 * col : static_cast<Index>(2 * VectorIndex(col, n, incy));
        const float2 xCol = LoadComplex<PAIRED>(x, xj);
        const float2 yCol = LoadComplex<PAIRED>(y, yj);
        const float cxr = xCol.x, cxi = xCol.y;
        const float cyr = yCol.x, cyi = yCol.y;
        const Index offset = 2 * (static_cast<Index>(col) * lda + row);
        // Preserve Netlib's zero-column branch and FP32 operation order.
        if (cxr != 0.0f || cxi != 0.0f || cyr != 0.0f || cyi != 0.0f) {
            // temp1 = alpha * conj(y[j]); temp2 = conj(alpha * x[j]).
            const float t1r = __fma(ar, cyr, 0.0f) - __fma(ai, -cyi, 0.0f);
            const float t1i = __fma(ar, -cyi, 0.0f) + __fma(ai, cyr, 0.0f);
            const float t2r = __fma(ar, cxr, 0.0f) - __fma(ai, cxi, 0.0f);
            const float t2i = -(__fma(ar, cxi, 0.0f) + __fma(ai, cxr, 0.0f));
            const float r1 = __fma(xr, t1r, 0.0f) - __fma(xx, t1i, 0.0f);
            const float r2 = __fma(yr, t2r, 0.0f) - __fma(yy, t2i, 0.0f);
            if (DIAGONAL && row == col) {
                a[offset] = a[offset] + (r1 + r2);
            } else {
                float2 value = LoadComplex<PAIRED>(a, offset);
                value.x = (value.x + r1) + r2;
                const float i1 = __fma(xr, t1i, 0.0f) + __fma(xx, t1r, 0.0f);
                const float i2 = __fma(yr, t2i, 0.0f) + __fma(yy, t2r, 0.0f);
                value.y = (value.y + i1) + i2;
                StoreComplex<PAIRED>(a, offset, value);
            }
        }
        if (DIAGONAL && row == col) {
            a[offset + 1] = 0.0f;
        }
    }
}

template <bool UPPER, typename Index, bool UNIT_STRIDE, bool PAIRED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void Cher2Tiles(
    uint32_t n, uint32_t lda, float ar, float ai, int64_t incx, int64_t incy, Index tileCount, __gm__ const float* x,
    __gm__ const float* y, __gm__ float* a)
{
    const uint32_t rowLane = threadIdx.x & (TILE_SIDE - 1);
    const uint32_t colLane = threadIdx.x / TILE_SIDE;
    for (Index tile = blockIdx.x; tile < tileCount; tile += gridDim.x) {
        // Invert the triangular tile number, correcting FP32 sqrt rounding.
        Index major = static_cast<Index>((__sqrtf(8.0f * static_cast<float>(tile) + 1.0f) - 1.0f) * 0.5f);
        while (major * (major + 1) / 2 > tile) {
            --major;
        }
        while ((major + 1) * (major + 2) / 2 <= tile) {
            ++major;
        }
        const Index minor = tile - major * (major + 1) / 2;
        const uint32_t row = (UPPER ? minor : major) * TILE_SIDE + rowLane;
        const uint32_t colBase = (UPPER ? major : minor) * TILE_SIDE;
        if (row >= n) {
            continue;
        }
        // Only diagonal tiles need a triangle mask and special diagonal stores.
        if (major == minor) {
            UpdateColumns<UPPER, Index, UNIT_STRIDE, PAIRED, true>(
                n, lda, ar, ai, incx, incy, row, colBase, colLane, x, y, a);
        } else {
            UpdateColumns<UPPER, Index, UNIT_STRIDE, PAIRED, false>(
                n, lda, ar, ai, incx, incy, row, colBase, colLane, x, y, a);
        }
    }
}
template <typename Index, bool UNIT_STRIDE, bool PAIRED>
__aicore__ inline void Dispatch(const Cher2TilingData& t, GM_ADDR x, GM_ADDR y, GM_ADDR a)
{
    const Index side = (static_cast<Index>(t.n) + TILE_SIDE - 1) / TILE_SIDE;
    const Index tiles = side * (side + 1) / 2;
    if (t.upper) {
        asc_vf_call<Cher2Tiles<true, Index, UNIT_STRIDE, PAIRED>>(
            dim3{THREADS, 1, 1}, t.n, t.lda, t.alphaReal, t.alphaImag, t.incx, t.incy, tiles,
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ const float*>(y),
            reinterpret_cast<__gm__ float*>(a));
    } else {
        asc_vf_call<Cher2Tiles<false, Index, UNIT_STRIDE, PAIRED>>(
            dim3{THREADS, 1, 1}, t.n, t.lda, t.alphaReal, t.alphaImag, t.incx, t.incy, tiles,
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ const float*>(y),
            reinterpret_cast<__gm__ float*>(a));
    }
}
} // namespace

extern "C" __global__ __aicore__ void cher2_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR a, Cher2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.n >= 1024 && tiling.n <= 4096 && tiling.incx == 1 && tiling.incy == 1 &&
        ((reinterpret_cast<uint64_t>(x) | reinterpret_cast<uint64_t>(y) | reinterpret_cast<uint64_t>(a)) & 7u) == 0) {
        cher2_detail::Cher2AivKernel kernel;
        kernel.Run(x, y, a, tiling);
        return;
    }
    // Bound all float offsets before selecting 32-bit indexing. Other shapes
    // and signed strides retain the full-width path.
    if (tiling.incx == 1 && tiling.incy == 1 && static_cast<uint64_t>(tiling.n) * tiling.lda <= 0x7fffffffu) {
        const bool aligned =
            ((reinterpret_cast<uint64_t>(x) | reinterpret_cast<uint64_t>(y) | reinterpret_cast<uint64_t>(a)) & 7u) == 0;
        if (aligned) {
            Dispatch<uint32_t, true, true>(tiling, x, y, a);
        } else {
            Dispatch<uint32_t, true, false>(tiling, x, y, a);
        }
    } else {
        Dispatch<uint64_t, false, false>(tiling, x, y, a);
    }
}

void cher2_kernel_do(uint8_t* x, uint8_t* y, uint8_t* a, const Cher2TilingData& tiling, uint32_t blocks, void* stream)
{
    cher2_kernel<<<blocks, nullptr, stream>>>(x, y, a, tiling);
}
