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
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "chpr_kernel.h"

using namespace AscendC;

constexpr uint32_t CHPR_LANES = 32U;
constexpr uint32_t CHPR_THREADS = 128U;
constexpr uint32_t CHPR_CACHE_COMPLEX = 4096U;
constexpr uint32_t ACLBLAS_UPPER_VALUE = 121U;
// Complex values use interleaved real and imaginary float components.
constexpr uint32_t CHPR_COMPLEX_COMPONENTS = 2U;
constexpr uint32_t CHPR_IMAG_OFFSET = 1U;

template <bool UPLO_IS_UPPER>
__simt_callee__ inline uint64_t ChprColumnBase(uint32_t col, uint32_t n)
{
    if constexpr (UPLO_IS_UPPER) {
        return static_cast<uint64_t>(col) * (col + 1U) / 2U;
    } else {
        return static_cast<uint64_t>(col) * (2ULL * n - col + 1U) / 2U;
    }
}

__simt_callee__ inline uint64_t XPhysicalIndex(uint32_t n, uint32_t logical, int64_t incx)
{
    if (incx > 0) {
        return static_cast<uint64_t>(static_cast<int64_t>(logical) * incx);
    }
    return static_cast<uint64_t>((static_cast<int64_t>(n) - 1 - logical) * (-incx));
}

template <bool UPLO_IS_UPPER, bool CACHE_X>
__simt_callee__ inline void ChprUpdateColumn(
    __gm__ const float* x, __gm__ float* ap, __ubuf__ const float* xCache, uint32_t n, int64_t incx, uint32_t col,
    float alpha)
{
    const uint32_t lane = threadIdx.x % CHPR_LANES;
    float colReal;
    float colImag;
    if constexpr (CACHE_X) {
        colReal = xCache[CHPR_COMPLEX_COMPONENTS * col];
        colImag = xCache[CHPR_COMPLEX_COMPONENTS * col + CHPR_IMAG_OFFSET];
    } else {
        const uint64_t physical = CHPR_COMPLEX_COMPONENTS * XPhysicalIndex(n, col, incx);
        colReal = x[physical];
        colImag = x[physical + CHPR_IMAG_OFFSET];
    }
    const uint64_t colBase = ChprColumnBase<UPLO_IS_UPPER>(col, n);
    // Netlib skips a zero column of x, including when alpha or other x elements are nonfinite.
    if (colReal == 0.0f && colImag == 0.0f) {
        if (lane == 0U) {
            const uint64_t diagonal = UPLO_IS_UPPER ? colBase + col : colBase;
            ap[CHPR_COMPLEX_COMPONENTS * diagonal + CHPR_IMAG_OFFSET] = 0.0f;
        }
        return;
    }
    const float tempReal = alpha * colReal;
    const float tempImag = -alpha * colImag;
    const uint32_t rowBegin = UPLO_IS_UPPER ? 0U : col;
    const uint32_t rowEnd = UPLO_IS_UPPER ? col + 1U : n;
    for (uint32_t row = rowBegin + lane; row < rowEnd; row += CHPR_LANES) {
        float rowReal;
        float rowImag;
        if constexpr (CACHE_X) {
            rowReal = xCache[CHPR_COMPLEX_COMPONENTS * row];
            rowImag = xCache[CHPR_COMPLEX_COMPONENTS * row + CHPR_IMAG_OFFSET];
        } else {
            const uint64_t physical = CHPR_COMPLEX_COMPONENTS * XPhysicalIndex(n, row, incx);
            rowReal = x[physical];
            rowImag = x[physical + CHPR_IMAG_OFFSET];
        }
        const uint64_t packed = colBase + row - rowBegin;
        const float updateReal = rowReal * tempReal - rowImag * tempImag;
        ap[CHPR_COMPLEX_COMPONENTS * packed] = ap[CHPR_COMPLEX_COMPONENTS * packed] + updateReal;
        if (row == col) {
            ap[CHPR_COMPLEX_COMPONENTS * packed + CHPR_IMAG_OFFSET] = 0.0f;
        } else {
            const float updateImag = rowReal * tempImag + rowImag * tempReal;
            ap[CHPR_COMPLEX_COMPONENTS * packed + CHPR_IMAG_OFFSET] =
                ap[CHPR_COMPLEX_COMPONENTS * packed + CHPR_IMAG_OFFSET] + updateImag;
        }
    }
}

template <bool UPLO_IS_UPPER, bool CACHE_X>
__simt_vf__ __aicore__ LAUNCH_BOUND(CHPR_THREADS) inline void ChprUpdate(
    __gm__ const float* x, __gm__ float* ap, uint32_t n, int64_t incx, float alpha)
{
    __ubuf__ float xCache[CHPR_COMPLEX_COMPONENTS * CHPR_CACHE_COMPLEX];
    if constexpr (CACHE_X) {
        for (uint32_t logical = threadIdx.x; logical < n; logical += blockDim.x) {
            const uint64_t physical = CHPR_COMPLEX_COMPONENTS * XPhysicalIndex(n, logical, incx);
            xCache[CHPR_COMPLEX_COMPONENTS * logical] = x[physical];
            xCache[CHPR_COMPLEX_COMPONENTS * logical + CHPR_IMAG_OFFSET] = x[physical + CHPR_IMAG_OFFSET];
        }
        asc_syncthreads();
    }

    const uint32_t columnsPerBlock = blockDim.x / CHPR_LANES;
    const uint32_t firstCol = blockIdx.x * columnsPerBlock + threadIdx.x / CHPR_LANES;
    const uint32_t colStep = gridDim.x * columnsPerBlock;
    for (uint32_t col = firstCol; col < n; col += colStep) {
        ChprUpdateColumn<UPLO_IS_UPPER, CACHE_X>(x, ap, xCache, n, incx, col, alpha);
    }
}

__global__ __aicore__ void chpr_kernel(GM_ADDR x, GM_ADDR ap, const ChprTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* apGm = reinterpret_cast<__gm__ float*>(ap);
    if (tiling.uplo == ACLBLAS_UPPER_VALUE) {
        if (tiling.n <= CHPR_CACHE_COMPLEX) {
            asc_vf_call<ChprUpdate<true, true>>(
                dim3{CHPR_THREADS, 1, 1}, xGm, apGm, tiling.n, tiling.incx, tiling.alpha);
        } else {
            asc_vf_call<ChprUpdate<true, false>>(
                dim3{CHPR_THREADS, 1, 1}, xGm, apGm, tiling.n, tiling.incx, tiling.alpha);
        }
    } else {
        if (tiling.n <= CHPR_CACHE_COMPLEX) {
            asc_vf_call<ChprUpdate<false, true>>(
                dim3{CHPR_THREADS, 1, 1}, xGm, apGm, tiling.n, tiling.incx, tiling.alpha);
        } else {
            asc_vf_call<ChprUpdate<false, false>>(
                dim3{CHPR_THREADS, 1, 1}, xGm, apGm, tiling.n, tiling.incx, tiling.alpha);
        }
    }
}

void chpr_kernel_do(GM_ADDR x, GM_ADDR ap, const ChprTilingData& tiling, uint32_t numBlocks, void* stream)
{
    chpr_kernel<<<numBlocks, nullptr, stream>>>(x, ap, tiling);
}
