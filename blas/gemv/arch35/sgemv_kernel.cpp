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
 * \file sgemv_kernel.cpp
 * \brief Single-precision general matrix-vector multiply (SGEMV) kernel for arch35 (DAV_3510).
 *
 * SIMT implementation.
 * y = alpha * op(A) * x + beta * y
 * trans=N: op(A) = A   (m x n), y has m elements, x has n elements
 * trans=T: op(A) = A^T (n x m), y has n elements, x has m elements
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "sgemv_tiling_data.h"
#include "common/helper/kernel_constant.h"

// Slab fast-path launcher (sgemv_slab_kernel.cpp).
void SgemvLaunchSlab(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgemvTilingData& tiling, uint32_t numBlocks,
    void* stream);

static constexpr uint32_t UB_X_FLOATS = 16384; // 64 KB __ubuf__ for x vector

// ==========================================================================
//  GM path — trans=N: grid-stride over output rows
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvNGm(
    uint32_t m, uint32_t n, uint32_t lda, float alpha, float beta, int64_t incx, int64_t incy, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm)
{
    int64_t lda64 = static_cast<int64_t>(lda);

    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < m; row += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        int64_t row64 = static_cast<int64_t>(row);

        for (uint32_t col = 0; col < n; ++col) {
            int64_t col64 = static_cast<int64_t>(col);
            // A stored column-major: A[row,col] = a[row + col*lda]
            int64_t aIdx = row64 + col64 * lda64;
            float aVal = aGm[aIdx];

            // x access with stride
            int64_t xIdx = (incx >= 0) ? (col64 * incx) : ((static_cast<int64_t>(n) - 1 - col64) * (-incx));
            float xVal = xGm[xIdx];

            acc += aVal * xVal;
        }

        // y access with stride
        int64_t yIdx = (incy >= 0) ? (row64 * incy) : ((static_cast<int64_t>(m) - 1 - row64) * (-incy));
        float yVal = yGm[yIdx];
        yGm[yIdx] = alpha * acc + beta * yVal;
    }
}

// ==========================================================================
//  GM path — trans=T: grid-stride over output columns
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvTGm(
    uint32_t m, uint32_t n, uint32_t lda, float alpha, float beta, int64_t incx, int64_t incy, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm)
{
    int64_t lda64 = static_cast<int64_t>(lda);

    for (uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        int64_t col64 = static_cast<int64_t>(col);

        for (uint32_t row = 0; row < m; ++row) {
            int64_t row64 = static_cast<int64_t>(row);
            // A stored column-major: A[row,col] = a[row + col*lda]
            int64_t aIdx = row64 + col64 * lda64;
            float aVal = aGm[aIdx];

            // x access with stride (x has m elements for trans=T)
            int64_t xIdx = (incx >= 0) ? (row64 * incx) : ((static_cast<int64_t>(m) - 1 - row64) * (-incx));
            float xVal = xGm[xIdx];

            acc += aVal * xVal;
        }

        // y access with stride (y has n elements for trans=T)
        int64_t yIdx = (incy >= 0) ? (col64 * incy) : ((static_cast<int64_t>(n) - 1 - col64) * (-incy));
        float yVal = yGm[yIdx];
        yGm[yIdx] = alpha * acc + beta * yVal;
    }
}

// ==========================================================================
//  UB path — trans=N: x cached in __ubuf__ shared memory (incx == 1)
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvNUb(
    uint32_t m, uint32_t n, uint32_t lda, float alpha, float beta, int64_t incy, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm)
{
    __ubuf__ float xUb[UB_X_FLOATS];
    int64_t lda64 = static_cast<int64_t>(lda);

    // Cooperative load: threads jointly load x[0..n-1] into UB
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        xUb[i] = xGm[i];
    }
    asc_syncthreads();

    // Compute: each thread handles assigned rows via grid-stride
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < m; row += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        int64_t row64 = static_cast<int64_t>(row);

        for (uint32_t col = 0; col < n; ++col) {
            int64_t aIdx = row64 + static_cast<int64_t>(col) * lda64;
            acc += aGm[aIdx] * xUb[col];
        }

        int64_t yIdx = (incy >= 0) ? (row64 * incy) : ((static_cast<int64_t>(m) - 1 - row64) * (-incy));
        float yVal = yGm[yIdx];
        yGm[yIdx] = alpha * acc + beta * yVal;
    }
}

// ==========================================================================
//  UB path — trans=T: x cached in __ubuf__ shared memory (incx == 1)
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvTUb(
    uint32_t m, uint32_t n, uint32_t lda, float alpha, float beta, int64_t incy, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* yGm)
{
    __ubuf__ float xUb[UB_X_FLOATS];
    int64_t lda64 = static_cast<int64_t>(lda);

    // Cooperative load: threads jointly load x[0..m-1] into UB
    for (uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
        xUb[i] = xGm[i];
    }
    asc_syncthreads();

    // Compute: each thread handles assigned columns via grid-stride
    for (uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        int64_t col64 = static_cast<int64_t>(col);

        for (uint32_t row = 0; row < m; ++row) {
            int64_t aIdx = static_cast<int64_t>(row) + col64 * lda64;
            acc += aGm[aIdx] * xUb[row];
        }

        int64_t yIdx = (incy >= 0) ? (col64 * incy) : ((static_cast<int64_t>(n) - 1 - col64) * (-incy));
        float yVal = yGm[yIdx];
        yGm[yIdx] = alpha * acc + beta * yVal;
    }
}

// ==========================================================================
//  Kernel dispatcher — choose GM or UB path per launch
// ==========================================================================
__global__ __aicore__ void sgemv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;

    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);

    uint32_t outDim = (tiling.trans == 0) ? tiling.m : tiling.n;
    uint32_t xDim = (tiling.trans == 0) ? tiling.n : tiling.m;

    // UB path requires incx==1 and x vector fits in UB and enough work
    bool useUb = (tiling.incx == 1) && (xDim <= UB_X_FLOATS) && (outDim >= 32);

    if (tiling.trans == 0) {
        if (useUb) {
            asc_vf_call<SgemvNUb>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.beta, tiling.incy,
                aGm, xGm, yGm);
        } else {
            asc_vf_call<SgemvNGm>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.beta, tiling.incx,
                tiling.incy, aGm, xGm, yGm);
        }
    } else {
        if (useUb) {
            asc_vf_call<SgemvTUb>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.beta, tiling.incy,
                aGm, xGm, yGm);
        } else {
            asc_vf_call<SgemvTGm>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.beta, tiling.incx,
                tiling.incy, aGm, xGm, yGm);
        }
    }
}

void sgemv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SgemvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    // Column-chunked slab fast paths (see sgemv_slab_kernel.cpp).
    if (tiling.useSlab != 0) {
        SgemvLaunchSlab(a, x, y, workSpace, tiling, numBlocks, stream);
        return;
    }
    sgemv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
}
