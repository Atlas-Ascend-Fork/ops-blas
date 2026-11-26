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
#include "sgeqrf_batched_tiling_data.h"

using namespace AscendC;

constexpr uint32_t GEQRF_SIMT_THREADS = 2048;
constexpr uint32_t GEQRF_SMEM_SIZE = 2048;

__ubuf__ float g_smem[GEQRF_SMEM_SIZE];

__simt_callee__ __aicore__ inline float WarpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        val += asc_shfl_down(val, offset, 32);
    }
    return val;
}

__simt_callee__ __aicore__ inline float BlockReduceSum(float localVal, __ubuf__ float* smem)
{
    uint32_t tid = threadIdx.x;
    uint32_t warpId = tid / 32;
    uint32_t laneId = tid % 32;
    uint32_t numWarps = (blockDim.x + 31) / 32;

    float warpSum = WarpReduceSum(localVal);
    if (laneId == 0) {
        smem[warpId] = warpSum;
    }
    asc_syncthreads();

    float result = 0.0f;
    if (tid < 32) {
        float val = (tid < numWarps) ? smem[tid] : 0.0f;
        result = WarpReduceSum(val);
    }
    asc_syncthreads();
    return result;
}

__simt_callee__ __aicore__ inline void ComputeCoefficients(
    uint32_t m, uint32_t n, uint64_t lda, uint32_t i, float tau, __gm__ float* A, __ubuf__ float* smem)
{
    uint32_t tid = threadIdx.x;
    uint32_t numWarps = (blockDim.x + 31) / 32;

    for (uint32_t c = i + 1; c < n; c++) {
        float local_dot = (tid == 0) ? A[i + c * lda] : 0.0f;
        for (uint32_t row = i + 1 + tid; row < m; row += blockDim.x) {
            local_dot += A[row + i * lda] * A[row + c * lda];
        }
        float dot = BlockReduceSum(local_dot, smem);
        if (tid == 0) {
            smem[numWarps + (c - i - 1)] = tau * dot;
        }
        asc_syncthreads();
    }
}

__simt_callee__ __aicore__ inline void ApplyUpdate(
    uint32_t m, uint32_t n, uint64_t lda, uint32_t i, __gm__ float* A, __ubuf__ float* smem)
{
    uint32_t tid = threadIdx.x;
    uint32_t numWarps = (blockDim.x + 31) / 32;

    for (uint32_t c = i + 1; c < n; c++) {
        float coeff = smem[numWarps + (c - i - 1)];
        if (tid == 0) {
            smem[0] = A[i + c * lda];
        }
        asc_syncthreads();
        float aiOrig = smem[0];
        for (uint32_t row = i + 1 + tid; row < m; row += blockDim.x) {
            A[row + c * lda] -= coeff * A[row + i * lda];
        }
        if (tid == 0) {
            A[i + c * lda] = aiOrig - coeff;
        }
        asc_syncthreads();
    }
}

__simt_callee__ __aicore__ inline void ApplyRank1Update(
    uint32_t m, uint32_t n, uint64_t lda, uint32_t i, float tau, __gm__ float* A, __ubuf__ float* smem)
{
    ComputeCoefficients(m, n, lda, i, tau, A, smem);
    ApplyUpdate(m, n, lda, i, A, smem);
}

__simt_callee__ __aicore__ inline float ComputeSigma(
    uint32_t m, uint64_t lda, uint32_t i, __gm__ float* A, __ubuf__ float* smem)
{
    uint32_t tid = threadIdx.x;

    float local_sigma = 0.0f;
    for (uint32_t row = i + 1 + tid; row < m; row += blockDim.x) {
        float val = A[row + i * lda];
        local_sigma += val * val;
    }

    float sigma = BlockReduceSum(local_sigma, smem);
    if (tid == 0) {
        smem[0] = sigma;
    }
    asc_syncthreads();
    return smem[0];
}

__simt_callee__ __aicore__ inline void ComputeTauAndNormalize(
    uint32_t m, uint64_t lda, uint32_t i, float sigma, __gm__ float* A, __gm__ float* Tau, __ubuf__ float* smem)
{
    uint32_t tid = threadIdx.x;

    float x1 = A[i + i * lda];
    float tau;
    float alpha;
    if (sigma == 0.0f) {
        tau = 0.0f;
        alpha = x1;
    } else {
        float normX = sqrtf(sigma + x1 * x1);
        alpha = (x1 >= 0.0f) ? -normX : normX;
        tau = (alpha - x1) / alpha;
    }
    if (tid == 0) {
        smem[0] = alpha;
        smem[1] = tau;
    }
    asc_syncthreads();
    alpha = smem[0];
    tau = smem[1];
    asc_syncthreads();

    if (tau == 0.0f) {
        if (tid == 0) {
            A[i + i * lda] = alpha;
            Tau[i] = tau;
        }
        asc_syncthreads();
        return;
    }

    float vScale = x1 - alpha;
    for (uint32_t row = i + 1 + tid; row < m; row += blockDim.x) {
        A[row + i * lda] = A[row + i * lda] / vScale;
    }
    if (tid == 0) {
        A[i + i * lda] = alpha;
        Tau[i] = tau;
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(GEQRF_SIMT_THREADS) inline void GeqrfSimtFp32(
    uint32_t m, uint32_t n, uint64_t lda, uint32_t numB, uint32_t startB, __gm__ float* aarrayBase,
    __gm__ float* tauarrayBase, __ubuf__ float* smem)
{
    __gm__ uintptr_t* aPtrAddrs = reinterpret_cast<__gm__ uintptr_t*>(aarrayBase);
    __gm__ uintptr_t* tauPtrAddrs = reinterpret_cast<__gm__ uintptr_t*>(tauarrayBase);
    uint32_t k = (m < n) ? m : n;

    for (uint32_t b = 0; b < numB; b++) {
        __gm__ float* A = reinterpret_cast<__gm__ float*>(aPtrAddrs[startB + b]);
        __gm__ float* Tau = reinterpret_cast<__gm__ float*>(tauPtrAddrs[startB + b]);

        for (uint32_t i = 0; i < k; i++) {
            float sigma = ComputeSigma(m, lda, i, A, smem);
            ComputeTauAndNormalize(m, lda, i, sigma, A, Tau, smem);
            float tau = smem[1];
            if (tau != 0.0f) {
                ApplyRank1Update(m, n, lda, i, tau, A, smem);
            }
        }
    }
}

__global__ __aicore__ void sgeqrf_batched(GM_ADDR aarrayPtr, GM_ADDR tauarrayPtr, GeqrfBatchedTilingData td)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t coreIdx = GetBlockIdx();
    uint32_t startB = coreIdx * td.batchPerCore;
    uint32_t numB = (coreIdx == td.usedCoreNum - 1) ? td.batchTail : td.batchPerCore;
    if (numB == 0) {
        return;
    }

    __gm__ float* aarrayBase = reinterpret_cast<__gm__ float*>(aarrayPtr);
    __gm__ float* tauarrayBase = reinterpret_cast<__gm__ float*>(tauarrayPtr);

    uint32_t warpSize = 32;
    uint32_t numThreads = ((td.m + warpSize - 1) / warpSize) * warpSize;
    if (numThreads < warpSize) {
        numThreads = warpSize;
    }
    if (numThreads > GEQRF_SIMT_THREADS) {
        numThreads = GEQRF_SIMT_THREADS;
    }

    asc_vf_call<GeqrfSimtFp32>(
        dim3{numThreads, 1, 1}, td.m, td.n, static_cast<uint64_t>(td.lda), numB, startB, aarrayBase, tauarrayBase,
        g_smem);
}

aclError sgeqrf_batched_kernel_do(
    GM_ADDR aarrayPtr, GM_ADDR tauarrayPtr, const GeqrfBatchedTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    sgeqrf_batched<<<numBlocks, 0, stream>>>(aarrayPtr, tauarrayPtr, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
