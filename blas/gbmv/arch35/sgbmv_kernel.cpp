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
 * \file sgbmv_kernel.cpp
 * \brief Single-precision general banded matrix-vector multiply.
 */

#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "sgbmv_tiling_data.h"
#include "common/helper/kernel_constant.h"

static constexpr uint32_t UB_X_FLOATS = 16384; // 64 KB __ubuf__ for x tile

__simt_callee__ __aicore__ inline void WriteY(
    uint32_t outIdx, uint32_t dim, int64_t incy, float alpha, float beta, float acc, __gm__ float* yGm)
{
    int64_t yIdx = (incy >= 0) ? (outIdx * incy) : ((dim - 1 - outIdx) * (-incy));
    float alphaTerm = (alpha == 0.0f) ? 0.0f : (alpha * acc);
    yGm[yIdx] = (beta == 0.0f) ? alphaTerm : (alphaTerm + beta * yGm[yIdx]);
}

// ==========================================================================
//  GM path — grid-stride
// ==========================================================================
template <bool TRANS_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgbmvGm(
    uint32_t m, uint32_t n, uint32_t kl, uint32_t ku, uint32_t lda, float alpha, float beta, int64_t incx, int64_t incy,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    uint32_t outDim = TRANS_T ? n : m;
    uint32_t xDim = TRANS_T ? m : n;

    for (uint32_t outIdx = blockIdx.x * blockDim.x + threadIdx.x; outIdx < outDim; outIdx += gridDim.x * blockDim.x) {
        float acc = 0.0f;

        if constexpr (!TRANS_T) {
            uint32_t row = outIdx;
            uint32_t colStart = (row >= kl) ? (row - kl) : 0;
            uint32_t colEnd = (row + ku + 1 < n) ? (row + ku + 1) : n;
            for (uint32_t col = colStart; col < colEnd; ++col) {
                int64_t aIdx = (ku + row - col) + static_cast<int64_t>(lda) * col;
                float xVal = (incx >= 0) ? xGm[col * incx] : xGm[(xDim - 1 - col) * (-incx)];
                acc += aGm[aIdx] * xVal;
            }
            WriteY(row, m, incy, alpha, beta, acc, yGm);
        } else {
            uint32_t col = outIdx;
            uint32_t rowStart = (col >= ku) ? (col - ku) : 0;
            uint32_t rowEnd = (col + kl + 1 < m) ? (col + kl + 1) : m;
            for (uint32_t row = rowStart; row < rowEnd; ++row) {
                int64_t aIdx = (ku + row - col) + static_cast<int64_t>(lda) * col;
                float xVal = (incx >= 0) ? xGm[row * incx] : xGm[(xDim - 1 - row) * (-incx)];
                acc += aGm[aIdx] * xVal;
            }
            WriteY(col, n, incy, alpha, beta, acc, yGm);
        }
    }
}

// ==========================================================================
//  UB path — x cached in __ubuf__ shared memory  (incx == 1)
// ==========================================================================
template <bool TRANS_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgbmvUb(
    uint32_t m, uint32_t n, uint32_t kl, uint32_t ku, uint32_t lda, float alpha, float beta, int64_t incy,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm, uint32_t outStart, uint32_t outEnd,
    int32_t xBase, uint32_t xLen)
{
    __ubuf__ float xUb[UB_X_FLOATS];

    for (uint32_t i = threadIdx.x; i < xLen; i += blockDim.x) {
        xUb[i] = xGm[static_cast<uint32_t>(xBase) + i];
    }
    asc_syncthreads();

    for (uint32_t outIdx = outStart + threadIdx.x; outIdx < outEnd; outIdx += blockDim.x) {
        float acc = 0.0f;

        if constexpr (!TRANS_T) {
            uint32_t row = outIdx;
            uint32_t colStart = (row >= kl) ? (row - kl) : 0;
            uint32_t colEnd = (row + ku + 1 < n) ? (row + ku + 1) : n;
            for (uint32_t col = colStart; col < colEnd; ++col) {
                int64_t aIdx = (ku + row - col) + static_cast<int64_t>(lda) * col;
                acc += aGm[aIdx] * xUb[col - xBase];
            }
            WriteY(row, m, incy, alpha, beta, acc, yGm);
        } else {
            uint32_t col = outIdx;
            uint32_t rowStart = (col >= ku) ? (col - ku) : 0;
            uint32_t rowEnd = (col + kl + 1 < m) ? (col + kl + 1) : m;
            for (uint32_t row = rowStart; row < rowEnd; ++row) {
                int64_t aIdx = (ku + row - col) + static_cast<int64_t>(lda) * col;
                acc += aGm[aIdx] * xUb[row - xBase];
            }
            WriteY(col, n, incy, alpha, beta, acc, yGm);
        }
    }
}

// ==========================================================================
//  Kernel dispatcher — shared by trans=N and trans=T
// ==========================================================================
template <bool TRANS_T>
__aicore__ inline void SgbmvDispatch(
    const SgbmvTilingData& tiling, __gm__ float* aGm, __gm__ float* xGm, __gm__ float* yGm)
{
    uint32_t outDim = TRANS_T ? tiling.n : tiling.m;
    uint32_t xDim = TRANS_T ? tiling.m : tiling.n;

    if (tiling.incx != 1 || outDim < 32) {
        asc_vf_call<SgbmvGm<TRANS_T>>(
            dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.kl, tiling.ku, tiling.lda, tiling.alpha,
            tiling.beta, tiling.incx, tiling.incy, aGm, xGm, yGm);
        return;
    }

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t rowsPerBlk = tiling.rowsPerBlock;
    uint32_t outStart = static_cast<uint32_t>(blkIdx) * rowsPerBlk;
    uint32_t outEnd = (outStart + rowsPerBlk < outDim) ? (outStart + rowsPerBlk) : outDim;
    if (outStart >= outEnd) {
        return;
    }

    int32_t xBase;
    uint32_t xEnd;
    if constexpr (!TRANS_T) {
        xBase = (outStart >= tiling.kl) ? static_cast<int32_t>(outStart - tiling.kl) : 0;
        xEnd = (xDim < outEnd + tiling.ku) ? xDim : (outEnd + tiling.ku);
    } else {
        xBase = (outStart >= tiling.ku) ? static_cast<int32_t>(outStart - tiling.ku) : 0;
        xEnd = (xDim < outEnd + tiling.kl) ? xDim : (outEnd + tiling.kl);
    }
    uint32_t xLen = xEnd - static_cast<uint32_t>(xBase);

    if (xLen == 0 || xLen > UB_X_FLOATS) {
        asc_vf_call<SgbmvGm<TRANS_T>>(
            dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.kl, tiling.ku, tiling.lda, tiling.alpha,
            tiling.beta, tiling.incx, tiling.incy, aGm, xGm, yGm);
        return;
    }

    asc_vf_call<SgbmvUb<TRANS_T>>(
        dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.kl, tiling.ku, tiling.lda, tiling.alpha, tiling.beta,
        tiling.incy, aGm, xGm, yGm, outStart, outEnd, xBase, xLen);
}

__global__ __aicore__ void sgbmv_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SgbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;

    auto* aGm = reinterpret_cast<__gm__ float*>(a);
    auto* xGm = reinterpret_cast<__gm__ float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);

    if (tiling.trans == 0) {
        SgbmvDispatch<false>(tiling, aGm, xGm, yGm);
    } else {
        SgbmvDispatch<true>(tiling, aGm, xGm, yGm);
    }
}

aclError sgbmv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SgbmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    sgbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
