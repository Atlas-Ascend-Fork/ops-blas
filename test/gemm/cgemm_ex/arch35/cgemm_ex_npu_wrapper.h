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
#include "cgemm_ex_param.h"

inline size_t CgemmExElemSize(aclblasCgemmExType_t t)
{
    constexpr size_t kBytes[] = {4, 8, 2, 4};
    uint32_t idx = static_cast<uint32_t>(t);
    if (idx > 3U) {
        return 4;
    }
    return kBytes[idx];
}

inline aclblasStatus_t CgemmExAllocAndCopy(void** devPtr, const void* hostPtr, size_t bytes)
{
    constexpr size_t kMaxBytes = static_cast<size_t>(1) << 32;
    *devPtr = nullptr;
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (bytes > kMaxBytes) {
        return ACLBLAS_STATUS_INVALID_VALUE;
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

inline void CgemmExFree(void* dA, void* dB, void* dC)
{
    if (dA) {
        aclrtFree(dA);
    }
    if (dB) {
        aclrtFree(dB);
    }
    if (dC) {
        aclrtFree(dC);
    }
}

inline aclblasStatus_t CgemmExAllocABC(
    const void* A, size_t aBytes, const void* B, size_t bBytes, const void* C, size_t cBytes, void** dA, void** dB,
    void** dC)
{
    *dA = nullptr;
    *dB = nullptr;
    *dC = nullptr;
    aclblasStatus_t st = CgemmExAllocAndCopy(dA, A, aBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CgemmExAllocAndCopy(dB, B, bBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        CgemmExFree(*dA, *dB, *dC);
        *dA = nullptr;
        return st;
    }
    st = CgemmExAllocAndCopy(dC, C, cBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        CgemmExFree(*dA, *dB, *dC);
        *dA = nullptr;
        *dB = nullptr;
        return st;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CgemmExCopyBackC(void* C, void* dC, size_t cBytes)
{
    if (dC == nullptr || C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCgemmEx_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const void* A, aclblasCgemmExType_t typeA, int lda, const void* B, aclblasCgemmExType_t typeB,
    int ldb, const aclblasComplex* beta, void* C, aclblasCgemmExType_t typeC, int ldc)
{
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasCgemmEx(
            handle, transA, transB, m, n, k, alpha, A, typeA, lda, B, typeB, ldb, beta, C, typeC, ldc);
    }
    int physColsA = (transA == ACLBLAS_OP_N) ? k : m;
    int physColsB = (transB == ACLBLAS_OP_N) ? n : k;
    size_t aBytes =
        static_cast<size_t>(std::max(1, lda)) * static_cast<size_t>(std::max(1, physColsA)) * CgemmExElemSize(typeA);
    size_t bBytes =
        static_cast<size_t>(std::max(1, ldb)) * static_cast<size_t>(std::max(1, physColsB)) * CgemmExElemSize(typeB);
    size_t cBytes =
        static_cast<size_t>(std::max(1, ldc)) * static_cast<size_t>(std::max(1, n)) * CgemmExElemSize(typeC);
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    aclblasStatus_t st = CgemmExAllocABC(A, aBytes, B, bBytes, C, cBytes, &dA, &dB, &dC);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    aclblasStatus_t ret = aclblasCgemmEx(
        handle, transA, transB, m, n, k, alpha, dA != nullptr ? dA : A, typeA, lda, dB != nullptr ? dB : B, typeB, ldb,
        beta, dC != nullptr ? dC : C, typeC, ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeDevice() == ACL_SUCCESS) {
        ret = CgemmExCopyBackC(C, dC, cBytes);
    } else if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    CgemmExFree(dA, dB, dC);
    return ret;
}
