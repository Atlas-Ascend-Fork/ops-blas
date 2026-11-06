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
#include <cstdint>
#include <cstring>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "gemm_ex_param.h"

// ── Helper: allocate device memory and copy host→device ──
// On success, *devPtr points to allocated device memory.
// On failure, *devPtr is set to nullptr and an error status is returned.
// If hostPtr is nullptr or bytes is 0, succeeds immediately with *devPtr = nullptr.
inline aclblasStatus_t AllocateAndCopyDevice(void** devPtr, const void* hostPtr, size_t bytes)
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
inline void CleanupDeviceBuffers(void* dA, void* dB, void* dC)
{
    if (dA)
        aclrtFree(dA);
    if (dB)
        aclrtFree(dB);
    if (dC)
        aclrtFree(dC);
}

// ── Helper: compute device buffer sizes for A, B, C ──
struct DeviceBufferSizes {
    size_t aBytes;
    size_t bBytes;
    size_t cBytes;
};

inline DeviceBufferSizes ComputeBufferSizes(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, int lda, int ldb, int ldc,
    aclDataType Atype, aclDataType Btype, aclDataType Ctype)
{
    const int physColsA = (transA == ACLBLAS_OP_N) ? k : m;
    const int physColsB = (transB == ACLBLAS_OP_N) ? n : k;
    const int allocLda = std::max(1, lda);
    const int allocLdb = std::max(1, ldb);
    const int allocLdc = std::max(1, ldc);
    const int elemSizeA = aclDataTypeSize(Atype);
    const int elemSizeB = aclDataTypeSize(Btype);
    const int elemSizeC = aclDataTypeSize(Ctype);
    DeviceBufferSizes sizes;
    sizes.aBytes = static_cast<size_t>(allocLda) * std::max(1, physColsA) * elemSizeA;
    sizes.bBytes = static_cast<size_t>(allocLdb) * std::max(1, physColsB) * elemSizeB;
    sizes.cBytes = static_cast<size_t>(allocLdc) * std::max(1, n) * elemSizeC;
    return sizes;
}

// ── Helper: copy result from device back to host ──
inline aclblasStatus_t CopyResultBack(void* hostC, void* devC, size_t cBytes, aclblasStatus_t callRet)
{
    if (hostC == nullptr || devC == nullptr || callRet != ACLBLAS_STATUS_SUCCESS) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError aclRet = aclrtMemcpy(hostC, cBytes, devC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (aclRet != ACL_SUCCESS) ? ACLBLAS_STATUS_INTERNAL_ERROR : ACLBLAS_STATUS_SUCCESS;
}

// ── NPU wrapper for aclblasGemmEx ──
// Handles device memory allocation, H2D/D2H transfers, and kernel invocation.
// NULLPTR inputs are passed through without device allocation (for error-path testing).
// aclblasGemmEx is declared in cann_ops_blas.h with C linkage; link via libops_blas.so.
inline aclblasStatus_t aclblasGemmEx_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const void* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb,
    const void* beta, void* C, aclDataType Ctype, int ldc, aclblasComputeType_t computeType)
{
    // Pass through for error-path or empty-matrix testing
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasGemmEx(
            handle, transA, transB, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType,
            ACLBLAS_GEMM_DEFAULT);
    }

    auto sizes = ComputeBufferSizes(transA, transB, m, n, k, lda, ldb, ldc, Atype, Btype, Ctype);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;

    // Allocate and copy A (only when A is non-null and k > 0)
    if (A != nullptr && k > 0) {
        aclblasStatus_t allocRet = AllocateAndCopyDevice(&dA, A, sizes.aBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) {
            CleanupDeviceBuffers(dA, dB, dC);
            return allocRet;
        }
    }

    // Allocate and copy B (only when B is non-null and k > 0)
    if (B != nullptr && k > 0) {
        aclblasStatus_t allocRet = AllocateAndCopyDevice(&dB, B, sizes.bBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) {
            CleanupDeviceBuffers(dA, dB, dC);
            return allocRet;
        }
    }

    // Allocate and copy C (only when C is non-null)
    if (C != nullptr) {
        aclblasStatus_t allocRet = AllocateAndCopyDevice(&dC, C, sizes.cBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) {
            CleanupDeviceBuffers(dA, dB, dC);
            return allocRet;
        }
    }

    // Execute kernel
    aclblasStatus_t ret = aclblasGemmEx(
        handle, transA, transB, m, n, k, alpha, dA ? dA : A, Atype, lda, dB ? dB : B, Btype, ldb, beta, dC ? dC : C,
        Ctype, ldc, computeType, ACLBLAS_GEMM_DEFAULT);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        // issue #825: sync 失败同样走 CleanupDeviceBuffers 释放 dA/dB/dC，
        // 与本函数其他失败路径及 strsm 包装器范式一致，避免设备内存泄漏
        CleanupDeviceBuffers(dA, dB, dC);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // Copy result back from device to host
    aclblasStatus_t copyRet = CopyResultBack(C, dC, sizes.cBytes, ret);
    if (copyRet != ACLBLAS_STATUS_SUCCESS) {
        CleanupDeviceBuffers(dA, dB, dC);
        return copyRet;
    }

    CleanupDeviceBuffers(dA, dB, dC);
    return ret;
}
