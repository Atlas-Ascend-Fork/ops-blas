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

#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct CherkDeviceBuffers {
    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;

    ~CherkDeviceBuffers()
    {
        if (dAlpha != nullptr) {
            aclrtFree(dAlpha);
        }
        if (dA != nullptr) {
            aclrtFree(dA);
        }
        if (dBeta != nullptr) {
            aclrtFree(dBeta);
        }
        if (dC != nullptr) {
            aclrtFree(dC);
        }
    }
};

static aclblasStatus_t CherkAllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    aclError aclRet = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    return aclRet == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

// Keep null/empty inputs null so validation cases reach the operator unchanged.
static aclblasStatus_t CherkTryAllocAndCopy(const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return CherkAllocAndCopy(devPtr, hostPtr, bytes);
}

struct CherkInputLayout {
    const float* alpha;
    size_t aBytes;
    size_t cBytes;
};

static CherkInputLayout CherkDescribeInputs(
    aclblasOperation_t trans, int n, int k, int lda, int ldc, const float* alpha, bool nullAlpha)
{
    const int aCols = trans == ACLBLAS_OP_N ? k : n;
    return {
        nullAlpha ? nullptr : alpha, static_cast<size_t>(lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex),
        static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex)};
}

static aclblasStatus_t CherkUploadInputs(
    const CherkInputLayout& inputs, const aclblasComplex* A, const float* beta, const aclblasComplex* C,
    CherkDeviceBuffers& bufs)
{
    aclblasStatus_t st = CherkTryAllocAndCopy(inputs.alpha, sizeof(float), bufs.dAlpha);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CherkTryAllocAndCopy(A, inputs.aBytes, bufs.dA);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CherkTryAllocAndCopy(beta, sizeof(float), bufs.dBeta);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return CherkTryAllocAndCopy(C, inputs.cBytes, bufs.dC);
}

static aclblasStatus_t CherkDownloadResult(aclblasComplex* C, const void* dC, size_t cBytes)
{
    if (C == nullptr || dC == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclError aclRet = aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return aclRet == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}
