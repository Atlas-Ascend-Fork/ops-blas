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
 * \file sgetrf_batched_kernel.cpp
 * \brief Kernel-side implementation for batched single-precision LU factorization (SIMT).
 */

#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "sgetrf_batched_tiling_data.h"

using namespace AscendC;

constexpr uint32_t GETRF_SIMT_MAX_THREADS = 256;
static_assert(
    (GETRF_SIMT_MAX_THREADS & (GETRF_SIMT_MAX_THREADS - 1)) == 0,
    "GETRF_SIMT_MAX_THREADS must be a power of 2 for tree reduction correctness");

// Helper: read a device-side float pointer from the pointer array stored in GM.
// The pointer array is stored as raw bytes (GM_ADDR) to avoid __gm__ pointer-to-pointer issues.
__simt_callee__ __aicore__ inline __gm__ float* ReadPtrFromArray(GM_ADDR aarrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(aarrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ float*>(rawAddr);
}

__simt_callee__ __aicore__ inline void Rank1Update(__gm__ float* A, uint32_t lda, uint32_t col, uint32_t updateSize)
{
    if (updateSize > 0) {
        for (uint32_t localR = threadIdx.x; localR < updateSize; localR += blockDim.x) {
            uint32_t row = col + 1 + localR;
            float multiplier = A[row + col * lda];
            for (uint32_t localC = 0; localC < updateSize; localC++) {
                uint32_t cc = col + 1 + localC;
                A[row + cc * lda] -= multiplier * A[col + cc * lda];
            }
        }
    }
}

__simt_callee__ __aicore__ inline int FindPivotRow(
    __gm__ float* A, uint32_t lda, uint32_t col, uint32_t n, __ubuf__ float* partialVals, __ubuf__ int* partialIdx)
{
    float localMax = 0.0f;
    int localRow = static_cast<int>(col);
    for (uint32_t row = col + threadIdx.x; row < n; row += blockDim.x) {
        float val = A[row + col * lda];
        float absVal = (val >= 0.0f) ? val : -val;
        if (absVal > localMax) {
            localMax = absVal;
            localRow = static_cast<int>(row);
        }
    }
    partialVals[threadIdx.x] = localMax;
    partialIdx[threadIdx.x] = localRow;
    asc_syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (partialVals[threadIdx.x + stride] > partialVals[threadIdx.x]) {
                partialVals[threadIdx.x] = partialVals[threadIdx.x + stride];
                partialIdx[threadIdx.x] = partialIdx[threadIdx.x + stride];
            }
        }
        asc_syncthreads();
    }
    return partialIdx[0];
}

__simt_callee__ __aicore__ inline void SwapRows(__gm__ float* A, uint32_t lda, uint32_t row1, uint32_t row2, uint32_t n)
{
    if (row1 != row2) {
        for (uint32_t c = threadIdx.x; c < n; c += blockDim.x) {
            float tmp = A[row1 + c * lda];
            A[row1 + c * lda] = A[row2 + c * lda];
            A[row2 + c * lda] = tmp;
        }
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void ComputeMultipliers(__gm__ float* A, uint32_t lda, uint32_t col, uint32_t n)
{
    float diagVal = A[col + col * lda];
    if (diagVal == 0.0f) {
        asc_syncthreads();
        return;
    }
    for (uint32_t row = col + 1 + threadIdx.x; row < n; row += blockDim.x) {
        A[row + col * lda] /= diagVal;
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(GETRF_SIMT_MAX_THREADS) inline void GetrfSimtPivot(
    uint32_t n, uint32_t lda, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase, __gm__ int* pivotArray,
    __gm__ int* infoArray)
{
    __ubuf__ float partialVals[GETRF_SIMT_MAX_THREADS];
    __ubuf__ int partialIdx[GETRF_SIMT_MAX_THREADS];
    __ubuf__ int sharedInfoBuf[1];

    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        __gm__ float* A = ReadPtrFromArray(aarrayBase, batchIdx);
        __gm__ int* piv = pivotArray + batchIdx * n;

        if (threadIdx.x == 0) {
            sharedInfoBuf[0] = 0;
        }
        asc_syncthreads();

        for (uint32_t col = 0; col < n; col++) {
            int pivotRow = FindPivotRow(A, lda, col, n, partialVals, partialIdx);
            float pivotVal = A[static_cast<uint32_t>(pivotRow) + col * lda];

            if (threadIdx.x == 0) {
                piv[col] = pivotRow + 1;
                if (pivotVal == 0.0f) {
                    sharedInfoBuf[0] = static_cast<int>(col) + 1;
                }
            }
            asc_syncthreads();
            if (sharedInfoBuf[0] != 0) {
                break;
            }

            SwapRows(A, lda, col, static_cast<uint32_t>(pivotRow), n);
            ComputeMultipliers(A, lda, col, n);
            Rank1Update(A, lda, col, n - col - 1);
            asc_syncthreads();
        }

        if (threadIdx.x == 0) {
            infoArray[batchIdx] = sharedInfoBuf[0];
        }
        asc_syncthreads();
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(GETRF_SIMT_MAX_THREADS) inline void GetrfSimtNoPivot(
    uint32_t n, uint32_t lda, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase, __gm__ int* infoArray)
{
    __ubuf__ int sharedInfoBuf[1];

    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        __gm__ float* A = ReadPtrFromArray(aarrayBase, batchIdx);

        if (threadIdx.x == 0) {
            sharedInfoBuf[0] = 0;
        }
        asc_syncthreads();

        for (uint32_t col = 0; col < n; col++) {
            float diagVal = A[col + col * lda];

            // Check singularity
            if (threadIdx.x == 0) {
                if (diagVal == 0.0f) {
                    sharedInfoBuf[0] = static_cast<int>(col) + 1;
                }
            }
            asc_syncthreads();
            if (sharedInfoBuf[0] != 0) {
                break;
            }

            // Multiplier computation
            for (uint32_t row = col + 1 + threadIdx.x; row < n; row += blockDim.x) {
                A[row + col * lda] /= diagVal;
            }
            asc_syncthreads();

            // Rank-1 update
            Rank1Update(A, lda, col, n - col - 1);
            asc_syncthreads();
        }

        // Write infoArray only if provided (may be nullptr in non-pivot mode)
        if (threadIdx.x == 0 && infoArray != nullptr) {
            infoArray[batchIdx] = sharedInfoBuf[0];
        }
        asc_syncthreads();
    }
}

extern "C" __global__ __aicore__ void sgetrf_batched_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, const SgetrfBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t blockId = GetBlockIdx();
    uint32_t startBatch = blockId * tiling.batchPerCore;
    uint32_t numBatch = (blockId == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;
    if (numBatch == 0) {
        return;
    }

    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);

    if (tiling.usePivot) {
        asc_vf_call<GetrfSimtPivot>(
            dim3{GETRF_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, numBatch, startBatch, aarray, pivots, infos);
    } else {
        asc_vf_call<GetrfSimtNoPivot>(
            dim3{GETRF_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, numBatch, startBatch, aarray, infos);
    }
}

// issue #1180: launch through aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs
// so the runtime status of the launch is returned to the caller: the <<<>>>
// statement form discards it. Same pattern as the gemm_batched arch35 launchers.
aclError sgetrf_batched_kernel_do(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, const SgetrfBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(sgetrf_batched_kernel), &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        GM_ADDR aarray;
        GM_ADDR pivotArray;
        GM_ADDR infoArray;
        SgetrfBatchedTilingData tiling;
    } args{aarray, pivotArray, infoArray, tiling};
    static_assert(offsetof(Arguments, tiling) == 3U * sizeof(uint8_t*), "kernel ABI mismatch");
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}
