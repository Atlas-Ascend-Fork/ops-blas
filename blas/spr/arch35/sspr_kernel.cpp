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
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cann_ops_blas_common.h"
#include "sspr_tiling_data.h"
#include "common/helper/kernel_constant.h"

// Column base offset: first element index of column j in packed array.
// UPPER: sum_{k=0}^{j-1} (k+1) = j*(j+1)/2
// LOWER: sum_{k=0}^{j-1} (n-k) = j*(2*n-j+1)/2
template <bool UPLO_IS_UPPER>
__simt_callee__ inline uint64_t SsprColumnBase(uint32_t col, uint32_t n)
{
    if constexpr (UPLO_IS_UPPER) {
        return static_cast<uint64_t>(col) * (col + 1) / 2;
    } else {
        return static_cast<uint64_t>(col) * (2 * n - col + 1) / 2;
    }
}

// GM path: all data accessed directly from GM, grid-stride loop over columns.
// Column-major packed format: within each column, elements are stored contiguously (stride=1).
// Per-column hoist: x[col] read once, multiplied by alpha, reused for all rows in the column.
template <bool INCX_POSITIVE>
__simt_callee__ inline void SsprGmInner(
    uint32_t rowStart, uint32_t rowEnd, uint32_t n, int64_t incx, float axCol, uint64_t colBase, uint32_t apOffset,
    __gm__ const float* __restrict xGm, __gm__ float* __restrict apGm)
{
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        float xRow;
        if constexpr (INCX_POSITIVE) {
            xRow = xGm[static_cast<int64_t>(row) * incx];
        } else {
            xRow = xGm[(static_cast<int64_t>(n) - 1 - static_cast<int64_t>(row)) * (-incx)];
        }
        uint64_t apIdx = colBase + row - apOffset;
        apGm[apIdx] = apGm[apIdx] + axCol * xRow;
    }
}

template <bool UPLO_IS_UPPER, bool INCX_POSITIVE>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsprGmImpl(
    uint32_t n, float alpha, int64_t incx, __gm__ const float* __restrict xGm, __gm__ float* __restrict apGm)
{
    for (uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x) {
        float xCol;
        if constexpr (INCX_POSITIVE) {
            xCol = xGm[static_cast<int64_t>(col) * incx];
        } else {
            xCol = xGm[(static_cast<int64_t>(n) - 1 - static_cast<int64_t>(col)) * (-incx)];
        }
        float axCol = alpha * xCol;

        uint64_t colBase = SsprColumnBase<UPLO_IS_UPPER>(col, n);

        if constexpr (UPLO_IS_UPPER) {
            SsprGmInner<INCX_POSITIVE>(0, col + 1, n, incx, axCol, colBase, 0, xGm, apGm);
        } else {
            SsprGmInner<INCX_POSITIVE>(col, n, n, incx, axCol, colBase, col, xGm, apGm);
        }
    }
}

// UB-x path: cache x vector in __ubuf__ so all columns in the block share it.
// Precondition: incx == 1 (guaranteed by the dispatcher in the kernel entry).
// Each thread handles columns in [colStart, colEnd) with block-internal stride.
template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsprUb(
    uint32_t n, float alpha, __gm__ const float* __restrict xGm, __gm__ float* __restrict apGm, uint32_t colStart,
    uint32_t colEnd, uint32_t xBase, uint32_t xLen)
{
    __ubuf__ float xUb[UB_X_FLOATS];

    // Collaborative load x into UB (incx == 1, so contiguous)
    for (uint32_t i = threadIdx.x; i < xLen; i += blockDim.x) {
        xUb[i] = xGm[xBase + i];
    }
    asc_syncthreads();

    // Block-internal stride over columns assigned to this block
    for (uint32_t col = colStart + threadIdx.x; col < colEnd; col += blockDim.x) {
        // x[col] is column-level constant: read once from UB, hoisted outside inner loop
        float axCol = alpha * xUb[col - xBase];

        uint64_t colBase = SsprColumnBase<UPLO_IS_UPPER>(col, n);

        if constexpr (UPLO_IS_UPPER) {
            // Column col: rows 0..col, AP[colBase + 0..col] contiguous with stride=1
            // x[0..col] read from UB sequentially, stride=1
            for (uint32_t row = 0; row <= col; ++row) {
                float xRow = xUb[row - xBase];
                uint64_t apIdx = colBase + row;
                apGm[apIdx] = apGm[apIdx] + axCol * xRow;
            }
        } else {
            // Column col: rows col..n-1, AP[colBase + 0..n-1-col] contiguous with stride=1
            // x[col..n-1] read from UB sequentially, stride=1
            for (uint32_t row = col; row < n; ++row) {
                float xRow = xUb[row - xBase];
                uint64_t apIdx = colBase + (row - col);
                apGm[apIdx] = apGm[apIdx] + axCol * xRow;
            }
        }
    }
}

// Fallback helper: launch the GM path when UB-x conditions are not met.
__aicore__ inline void SsprFallbackToGm(
    const SsprTilingData& tiling, __gm__ float* __restrict xGm, __gm__ float* __restrict apGm)
{
    if (tiling.uplo == ACLBLAS_UPPER) {
        if (tiling.incx >= 0) {
            asc_vf_call<SsprGmImpl<true, true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.alpha, tiling.incx, xGm, apGm);
        } else {
            asc_vf_call<SsprGmImpl<true, false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.alpha, tiling.incx, xGm, apGm);
        }
    } else {
        if (tiling.incx >= 0) {
            asc_vf_call<SsprGmImpl<false, true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.alpha, tiling.incx, xGm, apGm);
        } else {
            asc_vf_call<SsprGmImpl<false, false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.alpha, tiling.incx, xGm, apGm);
        }
    }
}

__global__ __aicore__ void sspr_kernel(GM_ADDR x, GM_ADDR ap, const SsprTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* __restrict xGm = reinterpret_cast<__gm__ float* __restrict>(x);
    auto* __restrict apGm = reinterpret_cast<__gm__ float* __restrict>(ap);

    if (tiling.incx != 1 || tiling.n < UB_THRESHOLD || tiling.n > UB_X_FLOATS) {
        SsprFallbackToGm(tiling, xGm, apGm);
        return;
    }

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t colsPerBlk = tiling.columnsPerBlock;
    uint32_t colStart = static_cast<uint32_t>(blkIdx) * colsPerBlk;
    uint32_t colEnd = (colStart + colsPerBlk < tiling.n) ? (colStart + colsPerBlk) : tiling.n;
    if (colStart >= colEnd) {
        return;
    }

    uint32_t xBase = (tiling.uplo == ACLBLAS_UPPER) ? 0 : colStart;
    uint32_t xLen = (tiling.uplo == ACLBLAS_UPPER) ? colEnd : (tiling.n - colStart);

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<SsprUb<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.alpha, xGm, apGm, colStart, colEnd, xBase, xLen);
    } else {
        asc_vf_call<SsprUb<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.alpha, xGm, apGm, colStart, colEnd, xBase, xLen);
    }
}

aclError sspr_kernel_do(GM_ADDR x, GM_ADDR ap, const SsprTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    sspr_kernel<<<numBlocks, nullptr, stream>>>(x, ap, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
