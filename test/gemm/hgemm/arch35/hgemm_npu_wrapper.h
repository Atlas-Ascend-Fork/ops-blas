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
#include "hgemm_param.h"

inline aclblasStatus_t HgemmAllocAndCopy(void** devPtr, const void* hostPtr, size_t bytes)
{
    *devPtr = nullptr;
    if (hostPtr == nullptr || bytes == 0)
        return ACLBLAS_STATUS_SUCCESS;
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

inline void HgemmFreeBuffers(void* dA, void* dB, void* dC)
{
    if (dA)
        aclrtFree(dA);
    if (dB)
        aclrtFree(dB);
    if (dC)
        aclrtFree(dC);
}

struct HgemmDeviceBuf {
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
};

inline aclblasStatus_t HgemmAllocDevice(
    HgemmDeviceBuf& dev, int k, const void* A, const void* B, void* C, size_t aBytes, size_t bBytes, size_t cBytes)
{
    // A/B staging must not require k > 0: the k == 0 cases feed small legal
    // matrices so the "A/B present but unreferenced" path is actually taken
    // (bytes collapse to a minimum footprint for those shapes). Negative k
    // is a validation-injection row - skip staging entirely, the operator
    // rejects it before touching any operand memory.
    if (A != nullptr && k >= 0) {
        aclblasStatus_t ret = HgemmAllocAndCopy(&dev.dA, A, aBytes);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
            return ret;
        }
    }
    if (B != nullptr && k >= 0) {
        aclblasStatus_t ret = HgemmAllocAndCopy(&dev.dB, B, bBytes);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
            return ret;
        }
    }
    if (C != nullptr) {
        aclblasStatus_t ret = HgemmAllocAndCopy(&dev.dC, C, cBytes);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
            return ret;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Post-launch tail: synchronize the handle stream (stream-scoped, not the
// whole device, so the accuracy cases prove the kernels really landed on the
// bound stream), then D2H-copy the whole ldc*n C buffer. C==nullptr with a
// successful call means the operator never wrote C, so nothing is copied back.
inline aclblasStatus_t HgemmSyncAndCopyBack(
    aclblasHandle_t handle, HgemmDeviceBuf& dev, void* C, size_t cBytes, aclblasStatus_t apiRet)
{
    if (apiRet != ACLBLAS_STATUS_SUCCESS) {
        HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
        return apiRet;
    }
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || stream == nullptr) {
        HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (dev.dC && C) {
        aclError aclRet = aclrtMemcpy(C, cBytes, dev.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    HgemmFreeBuffers(dev.dA, dev.dB, dev.dC);
    return apiRet;
}

// Mirrors aclblasSgemm_npu: stage host FLOAT16 buffers on device, invoke the
// library API through the handle stream, then copy C back.
inline aclblasStatus_t aclblasHgemm_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasHalf* alpha, const aclblasHalf* A, int lda, const aclblasHalf* B, int ldb, const aclblasHalf* beta,
    aclblasHalf* C, int ldc)
{
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasHgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    size_t physColsA = (transA == ACLBLAS_OP_N) ? k : m;
    size_t physColsB = (transB == ACLBLAS_OP_N) ? n : k;
    size_t aBytes = static_cast<size_t>(std::max(1, lda)) * std::max(size_t(1), physColsA) * sizeof(aclblasHalf);
    size_t bBytes = static_cast<size_t>(std::max(1, ldb)) * std::max(size_t(1), physColsB) * sizeof(aclblasHalf);
    size_t cBytes = static_cast<size_t>(std::max(1, ldc)) * std::max(1, n) * sizeof(aclblasHalf);

    HgemmDeviceBuf dev;
    aclblasStatus_t st = HgemmAllocDevice(dev, k, A, B, C, aBytes, bBytes, cBytes);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    aclblasStatus_t ret = aclblasHgemm(
        handle, transA, transB, m, n, k, alpha, dev.dA ? static_cast<const aclblasHalf*>(dev.dA) : A, lda,
        dev.dB ? static_cast<const aclblasHalf*>(dev.dB) : B, ldb, beta, dev.dC ? static_cast<aclblasHalf*>(dev.dC) : C,
        ldc);

    return HgemmSyncAndCopyBack(handle, dev, C, cBytes, ret);
}
