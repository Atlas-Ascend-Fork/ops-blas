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
#include "sger_tiling_data.h"
#include "common/helper/kernel_constant.h"

// GM path: all data accessed from GM, grid-stride loop over columns.
// Column-major addressing: A[i][j] = A[i + j * lda].
// Each thread handles multiple columns via grid-stride loop;
// inner loop iterates over all rows of a column (sequential access in column-major).
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgerGm(
    uint32_t m, uint32_t n, uint32_t lda, float alpha, int incx, int incy, __gm__ const float* xGm,
    __gm__ const float* yGm, __gm__ float* aGm)
{
    for (uint64_t col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x) {
        // Read y[col] with support for negative incy — keep in sync with SgerUbX
        float yVal = (incy >= 0) ? yGm[static_cast<int64_t>(col) * incy] :
                                   yGm[(static_cast<int64_t>(n) - 1 - static_cast<int64_t>(col)) * (-incy)];
        float ayVal = alpha * yVal;

        // Traverse all rows of this column — A[i + col*lda] is sequential in column-major
        for (uint64_t row = 0; row < m; ++row) {
            // Read x[row] with support for negative incx
            float xVal = (incx >= 0) ? xGm[static_cast<int64_t>(row) * incx] :
                                       xGm[(static_cast<int64_t>(m) - 1 - static_cast<int64_t>(row)) * (-incx)];

            // Column-major: A[i][j] = A[row + col * lda]
            uint64_t aIdx = row + col * lda;
            aGm[aIdx] = aGm[aIdx] + xVal * ayVal;
        }
    }
}

// UB-x path: cache x vector in __ubuf__ so all columns in the block share it.
// Precondition: incx == 1 (guaranteed by the dispatcher).
// Each thread handles columns in [colStart, colEnd) with block-internal stride.
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgerUbX(
    uint32_t m, uint32_t n, uint32_t lda, float alpha, int incy, __gm__ const float* xGm, __gm__ const float* yGm,
    __gm__ float* aGm, uint32_t colStart, uint32_t colEnd)
{
    __ubuf__ float xUb[UB_X_FLOATS];

    // Collaborative load x into UB (incx == 1, so contiguous)
    for (uint64_t i = threadIdx.x; i < m; i += blockDim.x) {
        xUb[i] = xGm[i];
    }
    asc_syncthreads();

    // Block-internal stride over columns assigned to this block
    for (uint64_t col = colStart + threadIdx.x; col < colEnd; col += blockDim.x) {
        // keep in sync with SgerGm
        float yVal = (incy >= 0) ? yGm[static_cast<int64_t>(col) * incy] :
                                   yGm[(static_cast<int64_t>(n) - 1 - static_cast<int64_t>(col)) * (-incy)];
        float ayVal = alpha * yVal;

        for (uint64_t row = 0; row < m; ++row) {
            uint64_t aIdx = row + col * lda;
            aGm[aIdx] = aGm[aIdx] + xUb[row] * ayVal;
        }
    }
}

__global__ __aicore__ void sger_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR A, const SgerTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ const float*>(y);
    auto* aGm = reinterpret_cast<__gm__ float*>(A);

    if ((tiling.incx == 1) && (tiling.m <= UB_X_FLOATS)) {
        // UB-x path: cache x in UB, shared across all columns in the block
        int32_t blkIdx = AscendC::GetBlockIdx();
        uint64_t colStart = static_cast<uint64_t>(blkIdx) * tiling.colsPerBlock;
        uint64_t colEndCandidate = colStart + tiling.colsPerBlock;
        uint64_t colEnd = colEndCandidate < tiling.n ? colEndCandidate : tiling.n;
        if (colStart < colEnd) {
            asc_vf_call<SgerUbX>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.incy, xGm, yGm, aGm,
                colStart, colEnd);
        }
        return;
    }
    // GM path: grid-stride over all columns, all data from GM
    asc_vf_call<SgerGm>(
        dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.incx, tiling.incy, xGm, yGm,
        aGm);
}

aclError sger_arch35_kernel_do(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, const SgerTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    sger_kernel<<<numBlocks, nullptr, stream>>>(x, y, A, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
