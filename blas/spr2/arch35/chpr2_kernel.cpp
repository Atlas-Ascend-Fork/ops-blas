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
#include "simt_api/asc_simt.h"
#include "cann_ops_blas_common.h"
#include "chpr2_kernel.h"

namespace {

template <bool UPPER>
__simt_callee__ inline uint64_t ColumnBase(uint32_t col, uint32_t n)
{
    if constexpr (UPPER) {
        return static_cast<uint64_t>(col) * (static_cast<uint64_t>(col) + 1) / 2;
    }
    return static_cast<uint64_t>(col) * (2 * static_cast<uint64_t>(n) - col + 1) / 2;
}

template <bool UPPER>
__simt_callee__ inline uint32_t FindColumn(uint64_t packedIndex, uint32_t n)
{
    uint32_t left = 0;
    uint32_t right = n;
    while (left + 1 < right) {
        const uint32_t middle = left + (right - left) / 2;
        if (ColumnBase<UPPER>(middle, n) <= packedIndex) {
            left = middle;
        } else {
            right = middle;
        }
    }
    return left;
}

__simt_callee__ inline void ComplexProduct(float ar, float ai, float br, float bi, float& real, float& imag)
{
    real = ar * br - ai * bi;
    imag = ar * bi + ai * br;
}

__simt_callee__ inline void StoreActiveElement(
    __gm__ float* ap, uint64_t apIndex, float ur, float ui, float vr, float vi, bool diagonal)
{
    if (diagonal) {
        // Netlib ignores the old imaginary component on the diagonal.
        ap[apIndex] = ap[apIndex] + (ur + vr);
    } else {
        ap[apIndex] = (ap[apIndex] + ur) + vr;
        ap[apIndex + 1] = (ap[apIndex + 1] + ui) + vi;
    }
}

template <bool UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHPR2_THREADS) inline void Chpr2Packed(
    __gm__ const float* x, __gm__ const float* y, __gm__ float* ap, uint32_t n, float alphaReal, float alphaImag,
    int64_t incx, int64_t incy, int64_t xStart, int64_t yStart, uint64_t begin, uint64_t end)
{
    uint64_t p = begin + threadIdx.x;
    if (p >= end) {
        return;
    }
    // One exact integer search per lane. Subsequent iterations advance through
    // columns, avoiding a square-root mapping and its large-index rounding errors.
    uint32_t col = FindColumn<UPPER>(p, n);
    uint64_t colBase = ColumnBase<UPPER>(col, n);
    uint64_t nextBase = ColumnBase<UPPER>(col + 1, n);
    uint32_t cachedCol = n;
    float t1r = 0.0f;
    float t1i = 0.0f;
    float t2r = 0.0f;
    float t2i = 0.0f;
    bool activeColumn = false;
    for (; p < end; p += blockDim.x) {
        while (p >= nextBase) {
            ++col;
            colBase = nextBase;
            nextBase = ColumnBase<UPPER>(col + 1, n);
        }
        if (cachedCol != col) {
            const uint64_t ix = static_cast<uint64_t>(xStart + static_cast<int64_t>(col) * incx) * 2;
            const uint64_t iy = static_cast<uint64_t>(yStart + static_cast<int64_t>(col) * incy) * 2;
            const float xr = x[ix];
            const float xi = x[ix + 1];
            const float yr = y[iy];
            const float yi = y[iy + 1];
            activeColumn = xr != 0.0f || xi != 0.0f || yr != 0.0f || yi != 0.0f;
            if (activeColumn) {
                ComplexProduct(alphaReal, alphaImag, yr, -yi, t1r, t1i);
                ComplexProduct(alphaReal, alphaImag, xr, xi, t2r, t2i);
                t2i = -t2i;
            }
            cachedCol = col;
        }
        const uint32_t row = static_cast<uint32_t>(p - colBase) + (UPPER ? 0 : col);
        const uint64_t apIndex = p * 2;
        if (activeColumn) {
            const uint64_t ix = static_cast<uint64_t>(xStart + static_cast<int64_t>(row) * incx) * 2;
            const uint64_t iy = static_cast<uint64_t>(yStart + static_cast<int64_t>(row) * incy) * 2;
            float ur, ui, vr, vi;
            ComplexProduct(x[ix], x[ix + 1], t1r, t1i, ur, ui);
            ComplexProduct(y[iy], y[iy + 1], t2r, t2i, vr, vi);
            StoreActiveElement(ap, apIndex, ur, ui, vr, vi, row == col);
        }
        if (row == col) {
            ap[apIndex + 1] = 0.0f;
        }
    }
}

} // namespace

__global__ __aicore__ void chpr2_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR ap, const Chpr2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint64_t begin = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tiling.elementsPerBlock;
    uint64_t end = begin + tiling.elementsPerBlock;
    if (end > tiling.packedLength) {
        end = tiling.packedLength;
    }
    if (begin >= end) {
        return;
    }
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ const float*>(y);
    auto* apGm = reinterpret_cast<__gm__ float*>(ap);
    // Aligned contiguous packed intervals give balanced work and coalesced
    // stores. SIMT word stores handle all tails without padded GM accesses.
    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<Chpr2Packed<true>>(
            dim3{CHPR2_THREADS, 1, 1}, xGm, yGm, apGm, tiling.n, tiling.alphaReal, tiling.alphaImag, tiling.incx,
            tiling.incy, tiling.xStart, tiling.yStart, begin, end);
    } else {
        asc_vf_call<Chpr2Packed<false>>(
            dim3{CHPR2_THREADS, 1, 1}, xGm, yGm, apGm, tiling.n, tiling.alphaReal, tiling.alphaImag, tiling.incx,
            tiling.incy, tiling.xStart, tiling.yStart, begin, end);
    }
}

void chpr2_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR ap, const Chpr2TilingData& tiling, uint32_t numBlocks, void* stream)
{
    chpr2_kernel<<<numBlocks, nullptr, stream>>>(x, y, ap, tiling);
}
