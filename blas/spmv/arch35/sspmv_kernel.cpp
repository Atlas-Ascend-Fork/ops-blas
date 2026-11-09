/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sspmv_kernel.cpp
 * \brief SSPMV Kernel for ascend950 (DAV_3510)
 */

#include "acl/acl_rt.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "cann_ops_blas_common.h"
#include "sspmv_tiling_data.h"

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SspmvSimtCompute(
    uint32_t n, float alpha, float beta, int64_t incx, int64_t incy, __gm__ const float* aGm, __gm__ const float* xGm,
    __gm__ float* yGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        // Column-start for the symmetry path, computed once per row.
        int64_t symStart;
        if constexpr (UPLO_IS_UPPER) {
            symStart = static_cast<int64_t>(row) * (row + 1) / 2;
        } else {
            symStart = static_cast<int64_t>(row) * n - static_cast<int64_t>(row) * (row - 1) / 2;
        }

        float acc = 0.0f;
        int64_t colStart = 0;

        for (uint32_t jCol = 0; jCol < n; ++jCol) {
            int64_t gmIdx;
            if constexpr (UPLO_IS_UPPER) {
                gmIdx = (row <= jCol) ? colStart + row : symStart + jCol;
                colStart += (jCol + 1);
            } else {
                gmIdx = (row >= jCol) ? colStart + (row - jCol) : symStart + (jCol - row);
                colStart += (n - jCol);
            }

            // Conditional expression for x stride — simpler than if-else.
            float xVal = (incx >= 0) ? xGm[jCol * incx] : xGm[(n - 1 - jCol) * (-incx)];
            acc += aGm[gmIdx] * xVal;
        }

        int64_t yIdx = (incy >= 0) ? (row * incy) : ((n - 1 - row) * (-incy));
        float yVal = yGm[yIdx];
        yGm[yIdx] = alpha * acc + beta * yVal;
    }
}

__global__ __aicore__ void sspmv_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SspmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<SspmvSimtCompute<true>>(
            dim3{tiling.nthreads, 1, 1}, tiling.n, tiling.alpha, tiling.beta, tiling.incx, tiling.incy,
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(y));
    } else {
        asc_vf_call<SspmvSimtCompute<false>>(
            dim3{tiling.nthreads, 1, 1}, tiling.n, tiling.alpha, tiling.beta, tiling.incx, tiling.incy,
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(y));
    }
}

aclError sspmv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, uint32_t numBlocks, const SspmvTilingData& tiling, void* stream)
{
    sspmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
    // issue #862: 返回发射阶段错误码，host 据此返回 EXECUTION_FAILED（对齐 chpmv 口径）
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
