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
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "syr_tiling_data.h"
#include "common/helper/kernel_constant.h"

static constexpr uint32_t UB_FLOATS = 8192;

template <bool UPLO_IS_UPPER>
__simt_callee__ inline void GetColRange(uint32_t row, uint32_t n, uint32_t& colStart, uint32_t& colEnd)
{
    if constexpr (UPLO_IS_UPPER) {
        colStart = row;
        colEnd = n;
    } else {
        colStart = 0;
        colEnd = row + 1;
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsyrGm(
    uint32_t n, uint32_t lda, float alpha, int64_t incx, __gm__ const float* xGm, __gm__ float* aGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        float xRow = (incx >= 0) ? xGm[row * incx] : xGm[(n - 1 - row) * (-incx)];
        float axRow = alpha * xRow;

        uint32_t colStart, colEnd;
        GetColRange<UPLO_IS_UPPER>(row, n, colStart, colEnd);

        for (uint32_t col = colStart; col < colEnd; ++col) {
            float xCol = (incx >= 0) ? xGm[col * incx] : xGm[(n - 1 - col) * (-incx)];
            uint64_t aIdx = static_cast<uint64_t>(col) * lda + row;
            aGm[aIdx] = aGm[aIdx] + axRow * xCol;
        }
    }
}

__aicore__ inline void FallbackToGm(
    const SyrTilingData& tiling, __gm__ float* __restrict xGm, __gm__ float* __restrict aGm)
{
    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<SsyrGm<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, tiling.incx, xGm, aGm);
    } else {
        asc_vf_call<SsyrGm<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, tiling.incx, xGm, aGm);
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsyrUb(
    uint32_t n, uint32_t lda, float alpha, __gm__ const float* xGm, __gm__ float* aGm, uint32_t rowStart,
    uint32_t rowEnd, uint32_t xBase, uint32_t xLen)
{
    __ubuf__ float xUb[UB_FLOATS];

    for (uint32_t i = threadIdx.x; i < xLen; i += blockDim.x)
        xUb[i] = xGm[xBase + i];

    asc_syncthreads();

    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        float xRow = xUb[row - xBase];
        float axRow = alpha * xRow;

        uint32_t colStart, colEnd;
        GetColRange<UPLO_IS_UPPER>(row, n, colStart, colEnd);

        for (uint32_t col = colStart; col < colEnd; ++col) {
            float xCol = xUb[col - xBase];
            uint64_t aIdx = static_cast<uint64_t>(col) * lda + row;
            aGm[aIdx] = aGm[aIdx] + axRow * xCol;
        }
    }
}

__global__ __aicore__ void syr_kernel(GM_ADDR x, GM_ADDR A, const SyrTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* __restrict xGm = reinterpret_cast<__gm__ float* __restrict>(x);
    auto* __restrict aGm = reinterpret_cast<__gm__ float* __restrict>(A);

    if (tiling.incx != 1 || tiling.n < 32) {
        FallbackToGm(tiling, xGm, aGm);
        return;
    }

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t rowsPerBlk = tiling.rowsPerBlock;
    uint32_t rowStart = static_cast<uint32_t>(blkIdx) * rowsPerBlk;
    uint32_t rowEnd = (rowStart + rowsPerBlk < tiling.n) ? (rowStart + rowsPerBlk) : tiling.n;
    if (rowStart >= rowEnd) {
        return;
    }

    uint32_t xBase = (tiling.uplo == ACLBLAS_UPPER) ? rowStart : 0;
    uint32_t xLen = (tiling.uplo == ACLBLAS_UPPER) ? (tiling.n - rowStart) : rowEnd;

    if (xLen > UB_FLOATS) {
        FallbackToGm(tiling, xGm, aGm);
        return;
    }

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<SsyrUb<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, xGm, aGm, rowStart, rowEnd, xBase, xLen);
    } else {
        asc_vf_call<SsyrUb<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alpha, xGm, aGm, rowStart, rowEnd, xBase, xLen);
    }
}

aclError syr_kernel_do(GM_ADDR x, GM_ADDR A, const SyrTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    syr_kernel<<<numBlocks, nullptr, stream>>>(x, A, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
