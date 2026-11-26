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
 * \file sgetrs_batched_kernel.cpp
 * \brief Kernel-side implementation for batched single-precision triangular solve (SIMT).
 */

#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "sgetrs_batched_tiling_data.h"

using namespace AscendC;

constexpr uint32_t GETRS_SIMT_MAX_THREADS = 256;
static_assert(
    (GETRS_SIMT_MAX_THREADS & (GETRS_SIMT_MAX_THREADS - 1)) == 0, "GETRS_SIMT_MAX_THREADS must be a power of 2");

constexpr uint32_t GETRS_REG_TILE_N = 8;

// Helper: read a device-side float pointer from the pointer array stored in GM.
// Caller must ensure all pointer array elements are valid (non-null) device addresses.
__simt_callee__ __aicore__ inline __gm__ float* ReadPtrFromArray(GM_ADDR arrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(arrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    if (rawAddr == 0) {
        return nullptr;
    }
    return reinterpret_cast<__gm__ float*>(rawAddr);
}

// Helper: load B columns into register array (shared by all Reg solve functions).
__simt_callee__ __aicore__ inline void LoadBToRegs(
    __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs, float regs[GETRS_REG_TILE_N])
{
    if (threadIdx.x < nrhs) {
        uint32_t col = threadIdx.x;
        for (uint32_t i = 0; i < n; i++) {
            regs[i] = B[i + col * ldb];
        }
    }
}

// Helper: store register array back to B columns (shared by all Reg solve functions).
__simt_callee__ __aicore__ inline void StoreRegsToB(
    __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs, const float regs[GETRS_REG_TILE_N])
{
    if (threadIdx.x < nrhs) {
        uint32_t col = threadIdx.x;
        for (uint32_t i = 0; i < n; i++) {
            B[i + col * ldb] = regs[i];
        }
    }
    asc_syncthreads();
}

// Helper: read A, B pointers and pivot offset for one batch (shared by all VF functions).
__simt_callee__ __aicore__ inline void ReadBatchPtrs(
    GM_ADDR aarrayBase, GM_ADDR barrayBase, uint32_t batchIdx, __gm__ float*& A, __gm__ float*& B)
{
    A = ReadPtrFromArray(aarrayBase, batchIdx);
    B = ReadPtrFromArray(barrayBase, batchIdx);
}

__simt_callee__ __aicore__ inline __gm__ int* ReadPivotPtr(__gm__ int* pivotArray, uint32_t batchIdx, uint32_t n)
{
    return pivotArray + static_cast<int64_t>(batchIdx) * n;
}

// Apply permutation P to B: swap rows according to pivot array (1-indexed LAPACK convention).
// Supports nrhs columns (each thread handles one column when threadIdx.x < nrhs).
__simt_callee__ __aicore__ inline void ApplyPermutationNrhs(
    __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs, __gm__ int* piv)
{
    if (n <= GETRS_REG_TILE_N) {
        if (threadIdx.x < nrhs) {
            uint32_t col = threadIdx.x;
            float regs[GETRS_REG_TILE_N] = {};
            for (uint32_t i = 0; i < n; i++) {
                regs[i] = B[i + col * ldb];
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
                B[i + col * ldb] = regs[i];
            }
        }
        asc_syncthreads();
    } else {
        for (uint32_t k = 0; k < n; k++) {
            int pivotRow = piv[k] - 1;
            if (pivotRow >= 0 && pivotRow < static_cast<int>(n) && pivotRow != static_cast<int>(k)) {
                if (threadIdx.x < nrhs) {
                    float tmp = B[k + threadIdx.x * ldb];
                    B[k + threadIdx.x * ldb] = B[static_cast<uint32_t>(pivotRow) + threadIdx.x * ldb];
                    B[static_cast<uint32_t>(pivotRow) + threadIdx.x * ldb] = tmp;
                }
            }
            asc_syncthreads();
        }
    }
}

// Apply inverse permutation P^T to B: reverse-order row swaps (for trans=T/C path).
__simt_callee__ __aicore__ inline void ApplyInversePermutationNrhs(
    __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs, __gm__ int* piv)
{
    if (n <= GETRS_REG_TILE_N) {
        if (threadIdx.x < nrhs) {
            uint32_t col = threadIdx.x;
            float regs[GETRS_REG_TILE_N] = {};
            for (uint32_t i = 0; i < n; i++) {
                regs[i] = B[i + col * ldb];
            }
            for (int32_t k = static_cast<int32_t>(n) - 1; k >= 0; k--) {
                int pivotRow = piv[k] - 1;
                if (pivotRow >= 0 && pivotRow < static_cast<int>(n) && pivotRow != k) {
                    uint32_t pr = static_cast<uint32_t>(pivotRow);
                    uint32_t uk = static_cast<uint32_t>(k);
                    float tmp = regs[uk];
                    regs[uk] = regs[pr];
                    regs[pr] = tmp;
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                B[i + col * ldb] = regs[i];
            }
        }
        asc_syncthreads();
    } else {
        for (int32_t k = static_cast<int32_t>(n) - 1; k >= 0; k--) {
            int pivotRow = piv[k] - 1;
            if (pivotRow >= 0 && pivotRow < static_cast<int>(n) && pivotRow != k) {
                uint32_t pr = static_cast<uint32_t>(pivotRow);
                uint32_t uk = static_cast<uint32_t>(k);
                if (threadIdx.x < nrhs) {
                    float tmp = B[uk + threadIdx.x * ldb];
                    B[uk + threadIdx.x * ldb] = B[pr + threadIdx.x * ldb];
                    B[pr + threadIdx.x * ldb] = tmp;
                }
            }
            asc_syncthreads();
        }
    }
}

// Forward solve L * Y = B in-place (L is unit lower triangular).
// Each thread handles one column (threadIdx.x < nrhs).
__simt_callee__ __aicore__ inline void ForwardSolveNrhs(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    if (n <= GETRS_REG_TILE_N) {
        if (threadIdx.x < nrhs) {
            uint32_t col = threadIdx.x;
            float regs[GETRS_REG_TILE_N] = {};
            for (uint32_t i = 0; i < n; i++) {
                regs[i] = B[i + col * ldb];
            }
            for (uint32_t j = 0; j < n; j++) {
                float bJC = regs[j];
                for (uint32_t i = j + 1; i < n; i++) {
                    regs[i] -= A[i + j * lda] * bJC;
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                B[i + col * ldb] = regs[i];
            }
        }
        asc_syncthreads();
    } else {
        for (uint32_t j = 0; j < n; j++) {
            float bJC = 0.0f;
            if (threadIdx.x < nrhs) {
                bJC = B[j + threadIdx.x * ldb];
            }
            if (threadIdx.x < nrhs) {
                for (uint32_t i = j + 1; i < n; i++) {
                    B[i + threadIdx.x * ldb] -= A[i + j * lda] * bJC;
                }
            }
            asc_syncthreads();
        }
    }
}

// Backward solve U * X = Y in-place (register-based, n <= GETRS_REG_TILE_N).
__simt_callee__ __aicore__ inline void BackwardSolveNrhsReg(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    float regs[GETRS_REG_TILE_N] = {};
    LoadBToRegs(B, ldb, n, nrhs, regs);
    for (int32_t j = static_cast<int32_t>(n) - 1; j >= 0; j--) {
        uint32_t uj = static_cast<uint32_t>(j);
        float ujj = A[uj + uj * lda];
        if (threadIdx.x < nrhs) {
            float bJC = regs[uj] / ujj;
            regs[uj] = bJC;
            for (uint32_t i = 0; i < uj; i++) {
                regs[i] -= A[i + uj * lda] * bJC;
            }
        }
        asc_syncthreads();
    }
    StoreRegsToB(B, ldb, n, nrhs, regs);
}

// Backward solve U * X = Y in-place (GM-based, n > GETRS_REG_TILE_N).
__simt_callee__ __aicore__ inline void BackwardSolveNrhsGeneral(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    for (int32_t j = static_cast<int32_t>(n) - 1; j >= 0; j--) {
        uint32_t uj = static_cast<uint32_t>(j);
        float ujj = A[uj + uj * lda];
        float bJC = 0.0f;
        if (threadIdx.x < nrhs) {
            bJC = B[uj + threadIdx.x * ldb] / ujj;
            B[uj + threadIdx.x * ldb] = bJC;
        }
        asc_syncthreads();
        if (threadIdx.x < nrhs) {
            for (uint32_t i = 0; i < uj; i++) {
                B[i + threadIdx.x * ldb] -= A[i + uj * lda] * bJC;
            }
        }
        asc_syncthreads();
    }
}

// Backward solve U * X = Y wrapper (dispatches by n).
__simt_callee__ __aicore__ inline void BackwardSolveNrhs(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    if (n <= GETRS_REG_TILE_N) {
        BackwardSolveNrhsReg(A, lda, B, ldb, n, nrhs);
    } else {
        BackwardSolveNrhsGeneral(A, lda, B, ldb, n, nrhs);
    }
}

// Solve U^T * Y = B in-place (register-based, n <= GETRS_REG_TILE_N).
// U^T is lower triangular; forward solve.
__simt_callee__ __aicore__ inline void BackwardSolveTNrhsReg(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    float regs[GETRS_REG_TILE_N] = {};
    LoadBToRegs(B, ldb, n, nrhs, regs);
    for (uint32_t j = 0; j < n; j++) {
        float ujj = A[j + j * lda];
        if (threadIdx.x < nrhs) {
            float bJC = regs[j] / ujj;
            regs[j] = bJC;
            for (uint32_t i = j + 1; i < n; i++) {
                regs[i] -= A[j + i * lda] * bJC;
            }
        }
        asc_syncthreads();
    }
    StoreRegsToB(B, ldb, n, nrhs, regs);
}

// Solve U^T * Y = B in-place (GM-based, n > GETRS_REG_TILE_N).
__simt_callee__ __aicore__ inline void BackwardSolveTNrhsGeneral(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    for (uint32_t j = 0; j < n; j++) {
        float ujj = A[j + j * lda];
        float bJC = 0.0f;
        if (threadIdx.x < nrhs) {
            bJC = B[j + threadIdx.x * ldb] / ujj;
            B[j + threadIdx.x * ldb] = bJC;
        }
        asc_syncthreads();
        if (threadIdx.x < nrhs) {
            for (uint32_t i = j + 1; i < n; i++) {
                B[i + threadIdx.x * ldb] -= A[j + i * lda] * bJC;
            }
        }
        asc_syncthreads();
    }
}

// Solve U^T * Y = B wrapper (dispatches by n).
__simt_callee__ __aicore__ inline void BackwardSolveTNrhs(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    if (n <= GETRS_REG_TILE_N) {
        BackwardSolveTNrhsReg(A, lda, B, ldb, n, nrhs);
    } else {
        BackwardSolveTNrhsGeneral(A, lda, B, ldb, n, nrhs);
    }
}

// Solve L^T * Z = Y in-place (register-based, n <= GETRS_REG_TILE_N).
// L^T is unit upper triangular; backward solve.
__simt_callee__ __aicore__ inline void ForwardSolveTNrhsReg(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    float regs[GETRS_REG_TILE_N] = {};
    LoadBToRegs(B, ldb, n, nrhs, regs);
    for (int32_t j = static_cast<int32_t>(n) - 1; j >= 0; j--) {
        uint32_t uj = static_cast<uint32_t>(j);
        if (threadIdx.x < nrhs) {
            for (uint32_t i = uj + 1; i < n; i++) {
                regs[uj] -= A[i + uj * lda] * regs[i];
            }
        }
        asc_syncthreads();
    }
    StoreRegsToB(B, ldb, n, nrhs, regs);
}

// Solve L^T * Z = Y in-place (GM-based, n > GETRS_REG_TILE_N).
__simt_callee__ __aicore__ inline void ForwardSolveTNrhsGeneral(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    for (int32_t j = static_cast<int32_t>(n) - 1; j >= 0; j--) {
        uint32_t uj = static_cast<uint32_t>(j);
        if (threadIdx.x < nrhs) {
            float bJC = B[uj + threadIdx.x * ldb];
            for (uint32_t i = uj + 1; i < n; i++) {
                bJC -= A[i + uj * lda] * B[i + threadIdx.x * ldb];
            }
            B[uj + threadIdx.x * ldb] = bJC;
        }
        asc_syncthreads();
    }
}

// Solve L^T * Z = Y wrapper (dispatches by n).
__simt_callee__ __aicore__ inline void ForwardSolveTNrhs(
    __gm__ float* A, uint32_t lda, __gm__ float* B, uint32_t ldb, uint32_t n, uint32_t nrhs)
{
    if (n <= GETRS_REG_TILE_N) {
        ForwardSolveTNrhsReg(A, lda, B, ldb, n, nrhs);
    } else {
        ForwardSolveTNrhsGeneral(A, lda, B, ldb, n, nrhs);
    }
}

// SIMT VF: with pivot (dispatches N vs T/C inside batch loop).
__simt_vf__ __aicore__ LAUNCH_BOUND(GETRS_SIMT_MAX_THREADS) inline void GetrsSimtPivot(
    uint32_t n, uint32_t nrhs, uint32_t lda, uint32_t ldb, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase,
    GM_ADDR barrayBase, __gm__ int* pivotArray, uint32_t trans)
{
    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        __gm__ float* A;
        __gm__ float* B;
        ReadBatchPtrs(aarrayBase, barrayBase, batchIdx, A, B);
        __gm__ int* piv = ReadPivotPtr(pivotArray, batchIdx, n);

        if (trans == GETRS_TRANS_N) {
            ApplyPermutationNrhs(B, ldb, n, nrhs, piv);
            ForwardSolveNrhs(A, lda, B, ldb, n, nrhs);
            BackwardSolveNrhs(A, lda, B, ldb, n, nrhs);
        } else {
            BackwardSolveTNrhs(A, lda, B, ldb, n, nrhs);
            ForwardSolveTNrhs(A, lda, B, ldb, n, nrhs);
            ApplyInversePermutationNrhs(B, ldb, n, nrhs, piv);
        }
    }
}

// SIMT VF: without pivot (dispatches N vs T/C inside batch loop).
__simt_vf__ __aicore__ LAUNCH_BOUND(GETRS_SIMT_MAX_THREADS) inline void GetrsSimtNoPivot(
    uint32_t n, uint32_t nrhs, uint32_t lda, uint32_t ldb, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase,
    GM_ADDR barrayBase, uint32_t trans)
{
    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        __gm__ float* A;
        __gm__ float* B;
        ReadBatchPtrs(aarrayBase, barrayBase, batchIdx, A, B);

        if (trans == GETRS_TRANS_N) {
            ForwardSolveNrhs(A, lda, B, ldb, n, nrhs);
            BackwardSolveNrhs(A, lda, B, ldb, n, nrhs);
        } else {
            BackwardSolveTNrhs(A, lda, B, ldb, n, nrhs);
            ForwardSolveTNrhs(A, lda, B, ldb, n, nrhs);
        }
    }
}

extern "C" __global__ __aicore__ void sgetrs_batched_kernel(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR ipiv, const SgetrsBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.usedCoreNum == 0) {
        return;
    }

    uint32_t blockId = GetBlockIdx();
    if (blockId >= tiling.usedCoreNum) {
        return;
    }

    uint32_t startBatch = static_cast<uint32_t>(static_cast<int64_t>(blockId) * tiling.batchPerCore);
    uint32_t numBatch = (blockId == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;
    if (numBatch == 0) {
        return;
    }

    auto* pivots = reinterpret_cast<__gm__ int*>(ipiv);

    if (tiling.usePivot) {
        asc_vf_call<GetrsSimtPivot>(
            dim3{GETRS_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.nrhs, tiling.lda, tiling.ldb, numBatch, startBatch,
            aarray, barray, pivots, tiling.trans);
    } else {
        asc_vf_call<GetrsSimtNoPivot>(
            dim3{GETRS_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.nrhs, tiling.lda, tiling.ldb, numBatch, startBatch,
            aarray, barray, tiling.trans);
    }
}

aclError sgetrs_batched_kernel_do(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR ipiv, const SgetrsBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    sgetrs_batched_kernel<<<numBlocks, nullptr, stream>>>(aarray, barray, ipiv, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
