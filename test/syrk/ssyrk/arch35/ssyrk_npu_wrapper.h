/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSYRK_NPU_WRAPPER_H
#define SSYRK_NPU_WRAPPER_H

#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct SsyrkDeviceBuffers {
    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;

    ~SsyrkDeviceBuffers()
    {
        if (dAlpha) aclrtFree(dAlpha);
        if (dA) aclrtFree(dA);
        if (dBeta) aclrtFree(dBeta);
        if (dC) aclrtFree(dC);
    }
};

static aclblasStatus_t AllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes,
    SsyrkDeviceBuffers& bufs)
{
    aclError aclRet = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t TryAllocAndCopy(
    const void* hostPtr, size_t bytes, void*& devPtr, SsyrkDeviceBuffers& bufs)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return AllocAndCopy(devPtr, hostPtr, bytes, bufs);
}

static const float* DevOrHost(const void* dev, const float* host)
{
    return dev ? static_cast<const float*>(dev) : host;
}

inline aclblasStatus_t aclblasSsyrk_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const float* A, int lda,
    const float* beta, float* C, int ldc)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    }

    const int aCols = (trans == ACLBLAS_OP_N) ? k : n;
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(aCols) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);
    constexpr size_t scalarBytes = sizeof(float);

    SsyrkDeviceBuffers bufs;
    aclblasStatus_t st;

    st = TryAllocAndCopy(alpha, scalarBytes, bufs.dAlpha, bufs);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = TryAllocAndCopy(A, aBytes, bufs.dA, bufs);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = TryAllocAndCopy(beta, scalarBytes, bufs.dBeta, bufs);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = TryAllocAndCopy(C, cBytes, bufs.dC, bufs);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    aclblasStatus_t ret = aclblasSsyrk(
        handle, uplo, trans, n, k,
        DevOrHost(bufs.dAlpha, alpha),
        DevOrHost(bufs.dA, A),
        lda,
        DevOrHost(bufs.dBeta, beta),
        bufs.dC ? static_cast<float*>(bufs.dC) : C,
        ldc);

    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (C != nullptr && bufs.dC != nullptr) {
        aclRet = aclrtMemcpy(C, cBytes, bufs.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    return ret;
}

#endif // SSYRK_NPU_WRAPPER_H
