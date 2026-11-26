/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ssbmv_kernel.cpp
 * \brief Single-precision symmetric banded matrix-vector multiply.
 */

#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "ssbmv_tiling_data.h"
#include "common/helper/kernel_constant.h"

static constexpr uint32_t UB_X_FLOATS = 16384; // 64 KB __ubuf__ for x tile

// ==========================================================================
//  GM path — grid-stride  (incx != 1 or small n)
// ==========================================================================
template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsbmvGm(
    uint32_t n, uint32_t k, uint32_t lda, float alpha, float beta, int64_t incx, int64_t incy, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm)
{
    // Grid-stride loop: each thread handles rows {tid, tid+step, tid+2*step, ...}
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        float acc = 0.0f;

        // Column range limited by band width k
        uint32_t colStart = (row >= k) ? (row - k) : 0;
        uint32_t colEnd = (n < row + k + 1) ? n : (row + k + 1);

        for (uint32_t col = colStart; col < colEnd; ++col) {
            // Compute band-storage GM index (LAPACK format).
            // UPPER: if row<=col read from column col, else use symmetry
            // LOWER: if row>=col read from column col, else use symmetry
            int64_t gmIdx;
            if constexpr (UPLO_IS_UPPER) {
                if (row <= col)
                    gmIdx = (k + row - col) + static_cast<int64_t>(lda) * col;
                else
                    gmIdx = (k + col - row) + static_cast<int64_t>(lda) * row;
            } else {
                if (row >= col)
                    gmIdx = (row - col) + static_cast<int64_t>(lda) * col;
                else
                    gmIdx = (col - row) + static_cast<int64_t>(lda) * row;
            }

            // x access with stride: incx>0 forward, incx<0 backward
            float xVal = (incx >= 0) ? xGm[col * incx] : xGm[(n - 1 - col) * (-incx)];
            acc += aGm[gmIdx] * xVal;
        }

        // y access with stride, then write back
        int64_t yIdx = (incy >= 0) ? (row * incy) : ((n - 1 - row) * (-incy));
        yGm[yIdx] = alpha * acc + beta * yGm[yIdx];
    }
}

// ==========================================================================
//  UB path — x cached in __ubuf__ shared memory  (incx == 1)
// ==========================================================================
template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsbmvUb(
    uint32_t n, uint32_t k, uint32_t lda, float alpha, float beta, int64_t incy, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm, uint32_t rowStart, uint32_t rowEnd, int32_t xBase, uint32_t xLen)
{
    // UB shared memory
    __ubuf__ float xUb[UB_X_FLOATS];

    // Phase 1 — cooperative load: threads jointly copy x[ xBase .. xBase+xLen )
    // from global memory into shared UB. Each thread loads a strided subset.
    for (uint32_t i = threadIdx.x; i < xLen; i += blockDim.x)
        xUb[i] = xGm[static_cast<uint32_t>(xBase) + i];
    asc_syncthreads(); // ensure all writes to xUb are visible

    // Phase 2 — compute: each thread handles its assigned rows.
    // Row range [rowStart, rowEnd) is contiguous within this block,
    // so x values are spatially local (good for __ubuf__ hit rate).
    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        float acc = 0.0f;

        uint32_t colStart = (row >= k) ? (row - k) : 0;
        uint32_t colEnd = (n < row + k + 1) ? n : (row + k + 1);

        for (uint32_t col = colStart; col < colEnd; ++col) {
            // A index — same band-storage formula as GM path
            int64_t gmIdx;
            if constexpr (UPLO_IS_UPPER) {
                if (row <= col)
                    gmIdx = (k + row - col) + static_cast<int64_t>(lda) * col;
                else
                    gmIdx = (k + col - row) + static_cast<int64_t>(lda) * row;
            } else {
                if (row >= col)
                    gmIdx = (row - col) + static_cast<int64_t>(lda) * col;
                else
                    gmIdx = (col - row) + static_cast<int64_t>(lda) * row;
            }

            // x from shared memory — xBase offset maps column to UB index
            acc += aGm[gmIdx] * xUb[col - xBase];
        }

        int64_t yIdx = (incy >= 0) ? (row * incy) : ((n - 1 - row) * (-incy));
        yGm[yIdx] = alpha * acc + beta * yGm[yIdx];
    }
}

// ==========================================================================
//  Kernel dispatcher — choose GM or UB path per launch
// ==========================================================================
__global__ __aicore__ void ssbmv_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* aGm = reinterpret_cast<__gm__ float*>(a);
    auto* xGm = reinterpret_cast<__gm__ float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);

    if (tiling.incx != 1 || tiling.n < 32) {
        if (tiling.uplo == ACLBLAS_UPPER) {
            asc_vf_call<SsbmvGm<true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alpha, tiling.beta, tiling.incx,
                tiling.incy, aGm, xGm, yGm);
        } else {
            asc_vf_call<SsbmvGm<false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alpha, tiling.beta, tiling.incx,
                tiling.incy, aGm, xGm, yGm);
        }
        return;
    }

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t rowsPerBlk = tiling.rowsPerBlock;
    uint32_t rowStart = static_cast<uint32_t>(blkIdx) * rowsPerBlk;
    uint32_t rowEnd = (rowStart + rowsPerBlk < tiling.n) ? (rowStart + rowsPerBlk) : tiling.n;
    if (rowStart >= rowEnd) {
        return;
    }

    int32_t xBase = (rowStart >= tiling.k) ? static_cast<int32_t>(rowStart - tiling.k) : 0;
    uint32_t xEnd = (tiling.n < rowEnd + tiling.k) ? tiling.n : (rowEnd + tiling.k);
    uint32_t xLen = xEnd - static_cast<uint32_t>(xBase);

    if (xLen == 0 || xLen > UB_X_FLOATS) {
        if (tiling.uplo == ACLBLAS_UPPER) {
            asc_vf_call<SsbmvGm<true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alpha, tiling.beta, tiling.incx,
                tiling.incy, aGm, xGm, yGm);
        } else {
            asc_vf_call<SsbmvGm<false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alpha, tiling.beta, tiling.incx,
                tiling.incy, aGm, xGm, yGm);
        }
        return;
    }

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<SsbmvUb<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alpha, tiling.beta, tiling.incy, aGm,
            xGm, yGm, rowStart, rowEnd, xBase, xLen);
    } else {
        asc_vf_call<SsbmvUb<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alpha, tiling.beta, tiling.incy, aGm,
            xGm, yGm, rowStart, rowEnd, xBase, xLen);
    }
}

aclError ssbmv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, uint32_t numBlocks, const SsbmvTilingData& tiling, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    ssbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
