/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct Ssyr2kDeviceBuffers {
    void* dAlpha = nullptr;
    void* dBeta = nullptr;
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;

    ~Ssyr2kDeviceBuffers()
    {
        if (dAlpha) aclrtFree(dAlpha);
        if (dBeta) aclrtFree(dBeta);
        if (dA) aclrtFree(dA);
        if (dB) aclrtFree(dB);
        if (dC) aclrtFree(dC);
    }
};

static aclblasStatus_t Ssyr2kAllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
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

static const float* Ssyr2kDevOrHost(const void* dev, const float* host)
{
    return dev ? static_cast<const float*>(dev) : host;
}

static bool Ssyr2kNeedsPassthrough(aclblasHandle handle, int n, int k,
    const float* alpha, const float* beta,
    aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    return handle == nullptr || n <= 0 || k < 0 ||
           alpha == nullptr || beta == nullptr ||
           (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) ||
           (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C);
}

static aclblasStatus_t Ssyr2kAllocAllDeviceBuffers(
    Ssyr2kDeviceBuffers& bufs, const float* alpha, const float* A, const float* B,
    const float* beta, float* C, size_t aBytes, size_t bBytes, size_t cBytes)
{
    constexpr size_t scalarBytes = sizeof(float);
    aclblasStatus_t st;
    st = Ssyr2kAllocAndCopy(bufs.dAlpha, alpha, scalarBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = Ssyr2kAllocAndCopy(bufs.dA, A, aBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = Ssyr2kAllocAndCopy(bufs.dB, B, bBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = Ssyr2kAllocAndCopy(bufs.dBeta, beta, scalarBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = Ssyr2kAllocAndCopy(bufs.dC, C, cBytes);
    return st;
}

static aclblasStatus_t Ssyr2kSyncAndCopyBack(void* dC, float* C, size_t cBytes)
{
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (C != nullptr && dC != nullptr) {
        if (aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSsyr2k_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc)
{
    if (Ssyr2kNeedsPassthrough(handle, n, k, alpha, beta, uplo, trans)) {
        return aclblasSsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    int aCols = (trans == ACLBLAS_OP_N) ? k : n;
    size_t aBytes = (aCols > 0) ? static_cast<size_t>(lda) * static_cast<size_t>(aCols) * sizeof(float) : 0;
    size_t bBytes = (aCols > 0) ? static_cast<size_t>(ldb) * static_cast<size_t>(aCols) * sizeof(float) : 0;
    size_t cBytes = static_cast<size_t>(n) * static_cast<size_t>(ldc) * sizeof(float);

    Ssyr2kDeviceBuffers bufs;
    aclblasStatus_t st = Ssyr2kAllocAllDeviceBuffers(
        bufs, alpha, A, B, beta, C, aBytes, bBytes, cBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    aclblasStatus_t ret = aclblasSsyr2k(
        handle, uplo, trans, n, k,
        Ssyr2kDevOrHost(bufs.dAlpha, alpha),
        Ssyr2kDevOrHost(bufs.dA, A),
        lda,
        Ssyr2kDevOrHost(bufs.dB, B),
        ldb,
        Ssyr2kDevOrHost(bufs.dBeta, beta),
        bufs.dC ? static_cast<float*>(bufs.dC) : C,
        ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = Ssyr2kSyncAndCopyBack(bufs.dC, C, cBytes);
    }

    return ret;
}

