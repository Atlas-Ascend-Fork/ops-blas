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
 * \file getri_batched_kernel.cpp
 * \brief Kernel-side implementation for batched single-precision matrix inversion (SIMT).
 */

#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "getri_batched_tiling_data.h"

using namespace AscendC;

constexpr uint32_t GETRI_SIMT_MAX_THREADS = 256;
static_assert(
    (GETRI_SIMT_MAX_THREADS & (GETRI_SIMT_MAX_THREADS - 1)) == 0, "GETRI_SIMT_MAX_THREADS must be a power of 2");

constexpr uint32_t GETRI_REG_TILE_N = 8;

// Helper: read a device-side float pointer from the pointer array stored in GM.
__simt_callee__ __aicore__ inline __gm__ float* ReadPtrFromArray(GM_ADDR arrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(arrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ float*>(rawAddr);
}

// Initialize C as an identity matrix (column-major).
__simt_callee__ __aicore__ inline void InitIdentity(__gm__ float* C, uint32_t ldc, uint32_t n)
{
    for (uint32_t col = threadIdx.x; col < n; col += blockDim.x) {
        for (uint32_t row = 0; row < n; row++) {
            C[row + col * ldc] = (row == col) ? 1.0f : 0.0f;
        }
    }
    asc_syncthreads();
}

// Apply permutation P to C: swap rows according to pivot array (1-indexed LAPACK convention).
// For n <= GETRI_REG_TILE_N, keeps entire column in a register array across all k-iterations to eliminate GM RAW
// hazards.
__simt_callee__ __aicore__ inline void ApplyPermutation(__gm__ float* C, uint32_t ldc, uint32_t n, __gm__ int* piv)
{
    if (n <= GETRI_REG_TILE_N) {
        if (threadIdx.x < n) {
            uint32_t col = threadIdx.x;
            float regs[GETRI_REG_TILE_N] = {};
            for (uint32_t i = 0; i < n; i++) {
                regs[i] = C[i + col * ldc];
            }
            for (uint32_t k = 0; k < n; k++) {
                int pivotRow = piv[k] - 1;
                if (pivotRow >= 0 && pivotRow < static_cast<int>(n) && pivotRow != static_cast<int>(k)) {
                    uint32_t pr = static_cast<uint32_t>(pivotRow);
                    float tmp = regs[k];
                    regs[k] = regs[pr];
                    regs[pr] = tmp;
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                C[i + col * ldc] = regs[i];
            }
        }
        asc_syncthreads();
    } else {
        for (uint32_t k = 0; k < n; k++) {
            int pivotRow = piv[k] - 1;
            if (pivotRow >= 0 && pivotRow < static_cast<int>(n) && pivotRow != static_cast<int>(k)) {
                for (uint32_t col = threadIdx.x; col < n; col += blockDim.x) {
                    float tmp = C[k + col * ldc];
                    C[k + col * ldc] = C[pivotRow + col * ldc];
                    C[pivotRow + col * ldc] = tmp;
                }
            }
            asc_syncthreads();
        }
    }
}

// Forward solve L * X = C in-place, overwriting C with X.
// L is unit lower triangular, stored in A below diagonal (A[i,j] for i > j).
// Column-parallel: each thread handles specific columns, processing all rows sequentially.
// For n <= GETRI_REG_TILE_N, reads column into register array ONCE before the j-loop, performs all j-iterations
// purely in registers, and writes back ONCE after the loop.
__simt_callee__ __aicore__ inline void ForwardSolve(
    __gm__ float* A, uint32_t lda, __gm__ float* C, uint32_t ldc, uint32_t n)
{
    if (n <= GETRI_REG_TILE_N) {
        if (threadIdx.x < n) {
            uint32_t col = threadIdx.x;
            float regs[GETRI_REG_TILE_N] = {};
            for (uint32_t i = 0; i < n; i++) {
                regs[i] = C[i + col * ldc];
            }
            for (uint32_t j = 0; j < n; j++) {
                float cJC = regs[j];
                for (uint32_t i = j + 1; i < n; i++) {
                    regs[i] -= A[i + j * lda] * cJC;
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                C[i + col * ldc] = regs[i];
            }
        }
        asc_syncthreads();
    } else {
        for (uint32_t j = 0; j < n; j++) {
            for (uint32_t col = threadIdx.x; col < n; col += blockDim.x) {
                float cJC = C[j + col * ldc];
                asc_syncthreads();
                for (uint32_t i = j + 1; i < n; i++) {
                    C[i + col * ldc] -= A[i + j * lda] * cJC;
                }
            }
            asc_syncthreads();
        }
    }
}

// Backward solve for n <= GETRI_REG_TILE_N: register-based approach
__simt_callee__ __aicore__ inline void BackwardSolveReg(
    __gm__ float* A, uint32_t lda, __gm__ float* C, uint32_t ldc, uint32_t n, __ubuf__ int* sharedInfoBuf)
{
    float regs[GETRI_REG_TILE_N] = {};
    if (threadIdx.x < n) {
        uint32_t col = threadIdx.x;
        for (uint32_t i = 0; i < n; i++) {
            regs[i] = C[i + col * ldc];
        }
    }

    for (int32_t j = static_cast<int32_t>(n) - 1; j >= 0; j--) {
        uint32_t uj = static_cast<uint32_t>(j);
        float ujj = A[uj + uj * lda];

        if (ujj == 0.0f) {
            sharedInfoBuf[0] = j + 1;
        }
        asc_syncthreads();
        if (sharedInfoBuf[0] != 0)
            break;

        if (threadIdx.x < n) {
            float cJC = regs[uj] / ujj;
            regs[uj] = cJC;
            for (uint32_t i = 0; i < uj; i++) {
                regs[i] -= A[i + uj * lda] * cJC;
            }
        }
        asc_syncthreads();
    }

    if (threadIdx.x < n && sharedInfoBuf[0] == 0) {
        uint32_t col = threadIdx.x;
        for (uint32_t i = 0; i < n; i++) {
            C[i + col * ldc] = regs[i];
        }
    }
    asc_syncthreads();
}

// Backward solve for n > GETRI_REG_TILE_N: general GM-based approach
__simt_callee__ __aicore__ inline void BackwardSolveGeneral(
    __gm__ float* A, uint32_t lda, __gm__ float* C, uint32_t ldc, uint32_t n, __ubuf__ int* sharedInfoBuf)
{
    for (int32_t j = static_cast<int32_t>(n) - 1; j >= 0; j--) {
        uint32_t uj = static_cast<uint32_t>(j);
        float ujj = A[uj + uj * lda];

        if (ujj == 0.0f) {
            sharedInfoBuf[0] = j + 1;
        }
        asc_syncthreads();
        if (sharedInfoBuf[0] != 0)
            break;

        for (uint32_t col = threadIdx.x; col < n; col += blockDim.x) {
            float cJC = C[uj + col * ldc] / ujj;
            C[uj + col * ldc] = cJC;
            for (uint32_t i = 0; i < uj; i++) {
                C[i + col * ldc] -= A[i + uj * lda] * cJC;
            }
        }
        asc_syncthreads();
    }
}

// Backward solve U * inv(A) = X in-place, overwriting C (currently X) with inv(A).
// U is upper triangular, stored in A on and above diagonal.
__simt_callee__ __aicore__ inline void BackwardSolve(
    __gm__ float* A, uint32_t lda, __gm__ float* C, uint32_t ldc, uint32_t n, __ubuf__ int* sharedInfoBuf)
{
    if (n <= GETRI_REG_TILE_N) {
        BackwardSolveReg(A, lda, C, ldc, n, sharedInfoBuf);
    } else {
        BackwardSolveGeneral(A, lda, C, ldc, n, sharedInfoBuf);
    }
}

// SIMT VF function: pivot mode (with row permutation).
__simt_vf__ __aicore__ LAUNCH_BOUND(GETRI_SIMT_MAX_THREADS) inline void GetriSimtPivot(
    uint32_t n, uint32_t lda, uint32_t ldc, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase,
    GM_ADDR carrayBase, __gm__ int* pivotArray, __gm__ int* infoArray)
{
    __ubuf__ int sharedInfoBuf[1];

    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        __gm__ float* A = ReadPtrFromArray(aarrayBase, batchIdx);
        __gm__ float* C = ReadPtrFromArray(carrayBase, batchIdx);
        __gm__ int* piv = pivotArray + batchIdx * n;

        // All threads initialize sharedInfoBuf to ensure correct state
        // (guards against stale __ubuf__ data from previous kernel executions)
        sharedInfoBuf[0] = 0;
        asc_syncthreads();

        // Step 1: Initialize C = I
        InitIdentity(C, ldc, n);

        // Step 2: Apply permutation P to C
        ApplyPermutation(C, ldc, n, piv);

        // Step 3: Forward solve L * X = C (in-place, overwriting C with X)
        ForwardSolve(A, lda, C, ldc, n);

        // Step 4: Backward solve U * inv(A) = X (in-place, overwriting C with inv(A))
        BackwardSolve(A, lda, C, ldc, n, sharedInfoBuf);

        // Step 5: Write info
        if (threadIdx.x == 0) {
            infoArray[batchIdx] = sharedInfoBuf[0];
        }
        asc_syncthreads();
    }
}

// SIMT VF function: no-pivot mode (P = I, skip permutation).
__simt_vf__ __aicore__ LAUNCH_BOUND(GETRI_SIMT_MAX_THREADS) inline void GetriSimtNoPivot(
    uint32_t n, uint32_t lda, uint32_t ldc, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase,
    GM_ADDR carrayBase, __gm__ int* infoArray)
{
    __ubuf__ int sharedInfoBuf[1];

    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        __gm__ float* A = ReadPtrFromArray(aarrayBase, batchIdx);
        __gm__ float* C = ReadPtrFromArray(carrayBase, batchIdx);

        // All threads initialize sharedInfoBuf to ensure correct state
        sharedInfoBuf[0] = 0;
        asc_syncthreads();

        InitIdentity(C, ldc, n);
        ForwardSolve(A, lda, C, ldc, n);
        BackwardSolve(A, lda, C, ldc, n, sharedInfoBuf);

        // Defensive: host guarantees infoArray != nullptr via ValidateGetriBatchedParams
        if (threadIdx.x == 0 && infoArray != nullptr) {
            infoArray[batchIdx] = sharedInfoBuf[0];
        }
        asc_syncthreads();
    }
}

extern "C" __global__ __aicore__ void sgetri_batched_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR carray, GM_ADDR infoArray, const SgetriBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.usedCoreNum == 0) {
        return;
    }

    uint32_t blockId = GetBlockIdx();
    if (blockId >= tiling.usedCoreNum) {
        return;
    }

    uint32_t startBatch = blockId * tiling.batchPerCore;
    uint32_t numBatch = (blockId == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;
    if (numBatch == 0) {
        return;
    }

    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);

    if (tiling.usePivot) {
        asc_vf_call<GetriSimtPivot>(
            dim3{GETRI_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, tiling.ldc, numBatch, startBatch, aarray, carray,
            pivots, infos);
    } else {
        asc_vf_call<GetriSimtNoPivot>(
            dim3{GETRI_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, tiling.ldc, numBatch, startBatch, aarray, carray,
            infos);
    }
}

// issue #1180: launch through aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs
// so the runtime status of the launch is returned to the caller: the <<<>>>
// statement form discards it. Same pattern as the gemm_batched arch35 launchers.
aclError sgetri_batched_kernel_do(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR carray, GM_ADDR infoArray, const SgetriBatchedTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(sgetri_batched_kernel), &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        GM_ADDR aarray;
        GM_ADDR pivotArray;
        GM_ADDR carray;
        GM_ADDR infoArray;
        SgetriBatchedTilingData tiling;
    } args{aarray, pivotArray, carray, infoArray, tiling};
    static_assert(offsetof(Arguments, tiling) == 4U * sizeof(uint8_t*), "kernel ABI mismatch");
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}
