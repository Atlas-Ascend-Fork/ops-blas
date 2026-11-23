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
#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct CherArch22NpuOptions {
    bool alphaOnDevice = false;
};

struct CherArch22DeviceBuffers {
    void* dAlpha = nullptr;
    void* dX = nullptr;
    void* dA = nullptr;

    ~CherArch22DeviceBuffers()
    {
        if (dAlpha != nullptr) {
            aclrtFree(dAlpha);
        }
        if (dX != nullptr) {
            aclrtFree(dX);
        }
        if (dA != nullptr) {
            aclrtFree(dA);
        }
    }
};

static inline aclblasStatus_t CherArch22AllocCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    aclError ret = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t CherArch22TryAllocCopy(const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return CherArch22AllocCopy(devPtr, hostPtr, bytes);
}

inline aclblasStatus_t aclblasCher_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* A, int lda, const CherArch22NpuOptions& options = {})
{
    if (handle == nullptr || n <= 0 || incx == 0 || incx == INT_MIN) {
        return aclblasCher(handle, uplo, n, alpha, x, incx, A, lda);
    }

    const int absIncx = std::abs(incx);
    const size_t xBytes =
        (x != nullptr) ? (static_cast<size_t>(n - 1) * static_cast<size_t>(absIncx) + 1U) * sizeof(aclblasComplex)
                       : 0U;
    const int allocLda = std::max(1, lda);
    const size_t aBytes =
        (A != nullptr) ? static_cast<size_t>(allocLda) * static_cast<size_t>(n) * sizeof(aclblasComplex)
                                         : 0U;

    CherArch22DeviceBuffers bufs;
    if (options.alphaOnDevice && alpha != nullptr) {
        aclblasStatus_t alphaSt = CherArch22AllocCopy(bufs.dAlpha, alpha, sizeof(float));
        if (alphaSt != ACLBLAS_STATUS_SUCCESS) {
            return alphaSt;
        }
    }
    aclblasStatus_t st = CherArch22TryAllocCopy(x, xBytes, bufs.dX);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CherArch22TryAllocCopy(A, aBytes, bufs.dA);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    const float* alphaArg = (bufs.dAlpha != nullptr) ? static_cast<const float*>(bufs.dAlpha) : alpha;
    st = aclblasCher(
        handle, uplo, n, alphaArg, (bufs.dX != nullptr) ? static_cast<const aclblasComplex*>(bufs.dX) : x, incx,
        (bufs.dA != nullptr) ? static_cast<aclblasComplex*>(bufs.dA) : A, lda);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (A != nullptr && bufs.dA != nullptr) {
        if (aclrtMemcpy(A, aBytes, bufs.dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
