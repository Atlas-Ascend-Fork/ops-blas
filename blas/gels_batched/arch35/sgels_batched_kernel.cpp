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
 * \file sgels_batched_kernel.cpp
 * \brief SIMT flat implementation for batched least-squares solver.
 *
 * Architecture:
 *   - Single kernel launch processes ALL batches in a loop.
 *   - SIMT programming model: __simt_vf__ + asc_vf_call + threadIdx + asc_syncthreads.
 *   - Shared __simt_callee__ helpers for parallel reduce, norm, Householder, reflection.
 *   - Multi-core distribution via batchPerCore/batchTail (R4 compliant).
 *
 * Algorithm paths:
 *   m >= n: QR decomposition -> Apply Q^T to C -> Back-substitution R*X = C'
 *   m <  n: LQ decomposition -> Forward-substitution L*Y = C -> Zero C[m:n,:] -> Apply Q^T
 *   OP_T: in-kernel transpose of A before decomposition (no extra kernel launch)
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "sgels_batched_tiling_data.h"

using namespace AscendC;

static constexpr float HOUSEHOLDER_EPS = 1.0e-30f;
static constexpr uint32_t UB_PARTIALS_SIZE = 128;
static constexpr uint32_t UB_SHARED_SIZE = 16;

// ============================================================================
// Shared __simt_callee__ helpers
// ============================================================================
__simt_callee__ __aicore__ inline float ParallelReduceSum(__ubuf__ float* partialSums, float val)
{
    partialSums[threadIdx.x] = val;
    asc_syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + s];
        }
        asc_syncthreads();
    }
    float result = partialSums[0];
    asc_syncthreads();
    return result;
}

__simt_callee__ __aicore__ inline float ParallelNorm(
    __gm__ float* src, uint32_t tailLen, uint32_t stride, __ubuf__ float* partialSums, float akk, __gm__ float* tauGm,
    uint32_t k)
{
    float partial = 0.0f;
    for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
        float v = src[i * stride];
        partial += v * v;
    }
    float xnorm = sqrtf(ParallelReduceSum(partialSums, partial));
    if (xnorm < HOUSEHOLDER_EPS && akk >= 0.0f) {
        if (threadIdx.x == 0) {
            tauGm[k] = 0.0f;
        }
        asc_syncthreads();
        return -1.0f;
    }
    return xnorm;
}

__simt_callee__ __aicore__ inline void ComputeHouseholderParams(
    float akk, float xnorm, __gm__ float* tauGm, uint32_t k, __gm__ float* aGm, int32_t lda, __ubuf__ float* sharedBuf)
{
    if (threadIdx.x == 0) {
        float na = fabsf(akk);
        float sc = (na > xnorm) ? na : xnorm;
        float sa = akk / sc;
        float sx = xnorm / sc;
        float nm = sc * sqrtf(sa * sa + sx * sx);
        float beta = (akk >= 0.0f) ? -nm : nm;
        float tau = (beta - akk) / beta;
        float scaleV = 1.0f / (akk - beta);
        tauGm[k] = tau;
        aGm[k * lda + k] = beta;
        sharedBuf[0] = scaleV;
        sharedBuf[1] = tau;
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void ApplyColReflection(
    uint32_t tailLen, int32_t lda, __gm__ float* vCol, __gm__ float* aCol, float tau, __ubuf__ float* partialSums)
{
    float dotPartial = 0.0f;
    for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
        dotPartial += vCol[i] * aCol[i];
    }
    float w = ParallelReduceSum(partialSums, dotPartial) + aCol[-1];
    asc_syncthreads();
    if (threadIdx.x == 0) {
        aCol[-1] -= tau * w;
    }
    float alpha = -tau * w;
    for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
        aCol[i] += alpha * vCol[i];
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void ApplyRowReflection(
    uint32_t tailLen, int32_t lda, __gm__ float* vRow, __gm__ float* aRow, float tau, __ubuf__ float* partialSums)
{
    float dotPartial = 0.0f;
    for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
        dotPartial += vRow[i * lda] * aRow[i * lda];
    }
    float w = ParallelReduceSum(partialSums, dotPartial) + aRow[-lda];
    asc_syncthreads();
    if (threadIdx.x == 0) {
        aRow[-lda] -= tau * w;
    }
    float alpha = -tau * w;
    for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
        aRow[i * lda] += alpha * vRow[i * lda];
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void ApplyQTransposeQR(
    uint32_t minMN, uint32_t m, uint32_t nrhs, int32_t lda, int32_t ldc, __gm__ float* aGm, __gm__ float* cGm,
    __gm__ float* tauGm, __ubuf__ float* partialSums)
{
    for (uint32_t k = 0; k < minMN; k++) {
        float tau = tauGm[k];
        if (tau == 0.0f)
            continue;
        uint32_t tailLen = m - k - 1;
        __gm__ float* vCol = aGm + k * lda + k + 1;
        for (uint32_t j = 0; j < nrhs; j++) {
            __gm__ float* cCol = cGm + j * ldc + k + 1;
            float dotPartial = 0.0f;
            for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
                dotPartial += vCol[i] * cCol[i];
            }
            float w = ParallelReduceSum(partialSums, dotPartial) + cCol[-1];
            asc_syncthreads();
            if (threadIdx.x == 0) {
                cCol[-1] -= tau * w;
            }
            float alpha = -tau * w;
            for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
                cCol[i] += alpha * vCol[i];
            }
            asc_syncthreads();
        }
    }
}

__simt_callee__ __aicore__ inline void ApplyQTransposeLQ(
    uint32_t minMN, uint32_t n, uint32_t nrhs, int32_t lda, int32_t ldc, __gm__ float* aGm, __gm__ float* cGm,
    __gm__ float* tauGm, __ubuf__ float* partialSums)
{
    for (int32_t kk = static_cast<int32_t>(minMN) - 1; kk >= 0; kk--) {
        uint32_t k = static_cast<uint32_t>(kk);
        float tau = tauGm[k];
        if (tau == 0.0f)
            continue;
        uint32_t tailLen = n - k - 1;
        __gm__ float* vRow = aGm + (k + 1) * lda + k;
        for (uint32_t j = 0; j < nrhs; j++) {
            float dotPartial = 0.0f;
            for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
                dotPartial += vRow[i * lda] * cGm[j * ldc + k + 1 + i];
            }
            float w = ParallelReduceSum(partialSums, dotPartial) + cGm[j * ldc + k];
            asc_syncthreads();
            if (threadIdx.x == 0) {
                cGm[j * ldc + k] -= tau * w;
            }
            float alpha = -tau * w;
            for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
                cGm[j * ldc + k + 1 + i] += alpha * vRow[i * lda];
            }
            asc_syncthreads();
        }
    }
}
__simt_callee__ __aicore__ inline void BackSubstitution(
    uint32_t n, uint32_t nrhs, int32_t lda, int32_t ldc, __gm__ float* aGm, __gm__ float* cGm, __gm__ int* devInfoBatch,
    __ubuf__ float* partialSums)
{
    for (uint32_t j = 0; j < nrhs; j++) {
        for (int32_t ii = static_cast<int32_t>(n) - 1; ii >= 0; ii--) {
            uint32_t i = static_cast<uint32_t>(ii);
            float rii = aGm[i * lda + i];
            if (rii == 0.0f) {
                if (threadIdx.x == 0) {
                    *devInfoBatch = static_cast<int32_t>(i + 1);
                }
                asc_syncthreads();
                return;
            }
            uint32_t dotLen = n - i - 1;
            float dotPartial = 0.0f;
            for (uint32_t p = threadIdx.x; p < dotLen; p += blockDim.x) {
                dotPartial += aGm[(i + 1 + p) * lda + i] * cGm[j * ldc + i + 1 + p];
            }
            float sum = ParallelReduceSum(partialSums, dotPartial);
            if (threadIdx.x == 0) {
                cGm[j * ldc + i] = (cGm[j * ldc + i] - sum) / rii;
            }
            asc_syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        *devInfoBatch = 0;
    }
}

__simt_callee__ __aicore__ inline void ForwardSubstitution(
    uint32_t m, uint32_t nrhs, int32_t lda, int32_t ldc, __gm__ float* aGm, __gm__ float* cGm, __gm__ int* devInfoBatch,
    __ubuf__ float* partialSums)
{
    for (uint32_t j = 0; j < nrhs; j++) {
        for (uint32_t i = 0; i < m; i++) {
            float lii = aGm[i * lda + i];
            if (lii == 0.0f) {
                if (threadIdx.x == 0) {
                    *devInfoBatch = static_cast<int32_t>(i + 1);
                }
                asc_syncthreads();
                return;
            }
            float dotPartial = 0.0f;
            for (uint32_t p = threadIdx.x; p < i; p += blockDim.x) {
                dotPartial += aGm[p * lda + i] * cGm[j * ldc + p];
            }
            float sum = ParallelReduceSum(partialSums, dotPartial);
            if (threadIdx.x == 0) {
                cGm[j * ldc + i] = (cGm[j * ldc + i] - sum) / lii;
            }
            asc_syncthreads();
        }
    }
}

__simt_callee__ __aicore__ inline void ZeroTailRows(
    uint32_t m, uint32_t n, uint32_t nrhs, int32_t ldc, __gm__ float* cGm)
{
    for (uint32_t j = 0; j < nrhs; j++) {
        for (uint32_t i = m + threadIdx.x; i < n; i += blockDim.x) {
            cGm[j * ldc + i] = 0.0f;
        }
    }
    asc_syncthreads();
}

// ============================================================================
// SIMT VF: In-kernel transpose A[origM x origN, lda=origLda] -> AT[origN x origM, lda=origN]
// Uses tempA as scratch space (sized origLda * origN floats).
// ============================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgelsTransposeSimt(
    uint32_t origM, uint32_t origN, int32_t origLda, __gm__ float* aGm, __gm__ float* tempA)
{
    // issue #864: lda*origN 在 32 位域回绕会截断拷贝规模，乘法与循环下标统一 64 位
    uint64_t copySize = static_cast<uint64_t>(static_cast<uint32_t>(origLda)) * origN;
    for (uint64_t i = threadIdx.x; i < copySize; i += blockDim.x) {
        tempA[i] = aGm[i];
    }
    asc_syncthreads();

    for (uint32_t col = 0; col < origM; col++) {
        for (uint32_t row = threadIdx.x; row < origN; row += blockDim.x) {
            aGm[static_cast<uint64_t>(col) * origN + row] = tempA[static_cast<uint64_t>(row) * origLda + col];
        }
    }
    asc_syncthreads();
}

// ============================================================================
// SIMT VF: QR decomposition + Apply Q^T + Back-substitution (m >= n)
// ============================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgelsQRSolveSimt(
    uint32_t m, uint32_t n, uint32_t nrhs, int32_t lda, int32_t ldc, __gm__ float* aGm, __gm__ float* cGm,
    __gm__ float* tauGm, __gm__ int* devInfoBatch)
{
    __ubuf__ float partialSums[UB_PARTIALS_SIZE];
    __ubuf__ float sharedBuf[UB_SHARED_SIZE];

    // Step 1: QR Decomposition (Householder, column-based)
    for (uint32_t k = 0; k < n; k++) {
        uint32_t tailLen = m - k - 1;
        __gm__ float* vCol = aGm + k * lda + k + 1;
        float akk = aGm[k * lda + k];
        float xnorm = ParallelNorm(vCol, tailLen, 1, partialSums, akk, tauGm, k);
        if (xnorm < 0.0f)
            continue;
        ComputeHouseholderParams(akk, xnorm, tauGm, k, aGm, lda, sharedBuf);
        float scaleV = sharedBuf[0];
        float tau = sharedBuf[1];
        for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
            vCol[i] *= scaleV;
        }
        asc_syncthreads();
        for (uint32_t j = k + 1; j < n; j++) {
            ApplyColReflection(tailLen, lda, vCol, aGm + j * lda + k + 1, tau, partialSums);
        }
    }

    // Step 2: Apply Q^T to C
    ApplyQTransposeQR(n, m, nrhs, lda, ldc, aGm, cGm, tauGm, partialSums);

    // Step 3: Back-substitution R * X = C'
    BackSubstitution(n, nrhs, lda, ldc, aGm, cGm, devInfoBatch, partialSums);
}

// ============================================================================
// SIMT VF: LQ decomposition + Forward-sub + Zero + Apply Q^T (m < n)
// ============================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgelsLQSolveSimt(
    uint32_t m, uint32_t n, uint32_t nrhs, int32_t lda, int32_t ldc, __gm__ float* aGm, __gm__ float* cGm,
    __gm__ float* tauGm, __gm__ int* devInfoBatch)
{
    __ubuf__ float partialSums[UB_PARTIALS_SIZE];
    __ubuf__ float sharedBuf[UB_SHARED_SIZE];

    // Step 1: LQ Decomposition (Householder, row-based)
    for (uint32_t k = 0; k < m; k++) {
        uint32_t tailLen = n - k - 1;
        __gm__ float* vRow = aGm + (k + 1) * lda + k;
        float akk = aGm[k * lda + k];
        float xnorm = ParallelNorm(vRow, tailLen, static_cast<uint32_t>(lda), partialSums, akk, tauGm, k);
        if (xnorm < 0.0f)
            continue;
        ComputeHouseholderParams(akk, xnorm, tauGm, k, aGm, lda, sharedBuf);
        float scaleV = sharedBuf[0];
        float tau = sharedBuf[1];
        for (uint32_t i = threadIdx.x; i < tailLen; i += blockDim.x) {
            vRow[i * lda] *= scaleV;
        }
        asc_syncthreads();
        for (uint32_t ii = k + 1; ii < m; ii++) {
            ApplyRowReflection(tailLen, lda, vRow, aGm + (k + 1) * lda + ii, tau, partialSums);
        }
    }

    // Step 2: Forward-substitution L * Y = C
    if (threadIdx.x == 0) {
        *devInfoBatch = 0;
    }
    asc_syncthreads();
    ForwardSubstitution(m, nrhs, lda, ldc, aGm, cGm, devInfoBatch, partialSums);

    asc_syncthreads();
    if (*devInfoBatch != 0) {
        return;
    }

    // Step 3: Zero C rows [m, n)
    ZeroTailRows(m, n, nrhs, ldc, cGm);

    // Step 4: Apply Q^T to extended C
    ApplyQTransposeLQ(m, n, nrhs, lda, ldc, aGm, cGm, tauGm, partialSums);
}

// ============================================================================
// Global kernel entry: dispatch batches to SIMT VF functions
// ============================================================================
__global__ __aicore__ void sgels_batched_simt_kernel(
    GM_ADDR aArray, GM_ADDR cArray, GM_ADDR workspace, GM_ADDR devInfo, const SgelsBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t blkIdx = GetBlockIdx();
    uint32_t startBatch = blkIdx * tiling.batchPerCore;
    uint32_t numBatch = (blkIdx == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;
    if (numBatch == 0) {
        return;
    }

    auto* aAddrArray = reinterpret_cast<__gm__ uint64_t*>(aArray);
    auto* cAddrArray = reinterpret_cast<__gm__ uint64_t*>(cArray);
    auto* workspaceGm = reinterpret_cast<__gm__ float*>(workspace);
    auto* devInfoGm = reinterpret_cast<__gm__ int*>(devInfo);
    // issue #864: 每 core tempA 尺寸与 host 分配口径一致（origLda * origN，origN=tiling.m），
    // 此前按 tiling.n 计算步长，n>m 时核间 tempA 交叠
    uint64_t tempAPerCoreFloats = static_cast<uint64_t>(static_cast<uint32_t>(tiling.origLda)) * tiling.m;
    __gm__ float* tempA = workspaceGm + static_cast<uint64_t>(tiling.batchSize) * tiling.minMN +
                          static_cast<uint64_t>(blkIdx) * tempAPerCoreFloats;

    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchId = startBatch + b;
        __gm__ float* aGm = reinterpret_cast<__gm__ float*>(aAddrArray[batchId]);
        __gm__ float* cGm = reinterpret_cast<__gm__ float*>(cAddrArray[batchId]);
        __gm__ float* tauGm = workspaceGm + static_cast<uint64_t>(batchId) * tiling.minMN;
        __gm__ int* devInfoBatch = &devInfoGm[batchId];

        if (tiling.trans) {
            asc_vf_call<SgelsTransposeSimt>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.m, tiling.origLda, aGm, tempA);
        }

        if (tiling.m >= tiling.n) {
            asc_vf_call<SgelsQRSolveSimt>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.nrhs, tiling.lda, tiling.ldc, aGm, cGm, tauGm,
                devInfoBatch);
        } else {
            asc_vf_call<SgelsLQSolveSimt>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.nrhs, tiling.lda, tiling.ldc, aGm, cGm, tauGm,
                devInfoBatch);
        }
    }
}

void sgels_decompose_kernel_do(
    GM_ADDR aArray, GM_ADDR cArray, GM_ADDR workspace, GM_ADDR devInfo, const SgelsBatchedTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    sgels_batched_simt_kernel<<<numBlocks, nullptr, stream>>>(aArray, cArray, workspace, devInfo, tiling);
}
