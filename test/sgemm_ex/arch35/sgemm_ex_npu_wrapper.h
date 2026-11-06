/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "sgemm_ex_param.h"

// ── Helper: allocate device memory and copy host→device ──
// On success *devPtr points to allocated device memory.
// On failure *devPtr is set to nullptr and an error status is returned.
// If hostPtr is nullptr or bytes is 0, succeeds immediately with *devPtr = nullptr.
inline aclblasStatus_t SgemmExAllocAndCopy(void** devPtr, const void* hostPtr, size_t bytes)
{
    *devPtr = nullptr;
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        *devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(*devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*devPtr);
        *devPtr = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ── Helper: free up to three device buffers ──
inline void SgemmExCleanup(void* dA, void* dB, void* dC)
{
    if (dA != nullptr) {
        aclrtFree(dA);
    }
    if (dB != nullptr) {
        aclrtFree(dB);
    }
    if (dC != nullptr) {
        aclrtFree(dC);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NPU wrapper for aclblasSgemmEx
//
// Handles device memory allocation, H2D/D2H transfers, and kernel invocation.
// NULLPTR inputs are passed through without device allocation (for error-path
// and edge-case testing).  Every ACL call is checked; on failure the wrapper
// frees all allocated device buffers and returns an error status.
// ═══════════════════════════════════════════════════════════════════════════════

inline aclblasStatus_t aclblasSgemmEx_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc,
    aclblasGemmAlgo_t algo)
{
    // 1. Fast path: handle == nullptr or m <= 0 or n <= 0 → pass through directly
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasSgemmEx(
            handle, transA, transB, m, n, k, alpha, A, ACL_FLOAT, lda, B, ACL_FLOAT, ldb, beta, C, ACL_FLOAT, ldc);
    }

    // 2. Compute device buffer sizes (column-major, float = 4 bytes)
    const int physColsA = (transA == ACLBLAS_OP_N) ? k : m;
    const int physColsB = (transB == ACLBLAS_OP_N) ? n : k;
    const int allocLda = std::max(1, lda);
    const int allocLdb = std::max(1, ldb);
    const int allocLdc = std::max(1, ldc);
    const size_t aBytes = static_cast<size_t>(allocLda) * std::max(1, physColsA) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(allocLdb) * std::max(1, physColsB) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(allocLdc) * std::max(1, n) * sizeof(float);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;

    // 3. Allocate and copy A (only when A is non-null and k > 0)
    if (A != nullptr && k > 0) {
        aclblasStatus_t allocRet = SgemmExAllocAndCopy(&dA, A, aBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) {
            SgemmExCleanup(dA, dB, dC);
            return allocRet;
        }
    }

    // 4. Allocate and copy B (only when B is non-null and k > 0)
    if (B != nullptr && k > 0) {
        aclblasStatus_t allocRet = SgemmExAllocAndCopy(&dB, B, bBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) {
            SgemmExCleanup(dA, dB, dC);
            return allocRet;
        }
    }

    // 5. Allocate and copy C (only when C is non-null)
    if (C != nullptr) {
        aclblasStatus_t allocRet = SgemmExAllocAndCopy(&dC, C, cBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) {
            SgemmExCleanup(dA, dB, dC);
            return allocRet;
        }
    }

    // 6. Execute kernel (pass device pointers; fall back to host ptr for nullptr)
    aclblasStatus_t ret = aclblasSgemmEx(
        handle, transA, transB, m, n, k, alpha, dA ? static_cast<const float*>(dA) : A, ACL_FLOAT, lda,
        dB ? static_cast<const float*>(dB) : B, ACL_FLOAT, ldb, beta, dC ? static_cast<float*>(dC) : C, ACL_FLOAT, ldc);

    // 7. Synchronize device (must check return value)
    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        SgemmExCleanup(dA, dB, dC);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // 8. Copy result back from device to host (only on success, when C is non-null)
    if (ret == ACLBLAS_STATUS_SUCCESS && C != nullptr && dC != nullptr) {
        aclError copyRet = aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_SUCCESS) {
            SgemmExCleanup(dA, dB, dC);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    // 9. Free device memory and return
    SgemmExCleanup(dA, dB, dC);
    return ret;
}
