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

struct CherNpuOptions {
    // When true, copy alpha to Device and pass the Device pointer so Host
    // CheckPtrLocation + D2H (issue #510) is exercised.
    bool alphaOnDevice = false;
};

struct CherDeviceBuffers {
    void* dAlpha = nullptr;
    void* dX = nullptr;
    void* dA = nullptr;

    ~CherDeviceBuffers()
    {
        if (dAlpha) {
            aclrtFree(dAlpha);
        }
        if (dX) {
            aclrtFree(dX);
        }
        if (dA) {
            aclrtFree(dA);
        }
    }
};

static aclblasStatus_t CherAllocCopy(void*& devPtr, const void* hostPtr, size_t bytes)
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

static aclblasStatus_t CherTryAllocCopy(const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return CherAllocCopy(devPtr, hostPtr, bytes);
}

static void CherBufferBytes(int n, int incx, int lda, const aclblasComplex* x, const aclblasComplex* A,
                            size_t& xBytes, size_t& aBytes)
{
    xBytes = 0;
    aBytes = 0;
    const int absIncx = std::abs(incx);
    if (x != nullptr && n > 0 && absIncx > 0) {
        xBytes = static_cast<size_t>((n - 1) * absIncx + 1) * sizeof(aclblasComplex);
    }
    if (A != nullptr && n > 0) {
        const int allocLda = std::max(1, std::max(n, lda));
        aBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    }
}

static aclblasStatus_t CherPrepareDeviceBuffers(
    const float* alpha, const aclblasComplex* x, aclblasComplex* A, size_t xBytes, size_t aBytes,
    const CherNpuOptions& options, CherDeviceBuffers& bufs)
{
    if (options.alphaOnDevice && alpha != nullptr) {
        aclblasStatus_t alphaSt = CherAllocCopy(bufs.dAlpha, alpha, sizeof(float));
        if (alphaSt != ACLBLAS_STATUS_SUCCESS) {
            return alphaSt;
        }
    }
    aclblasStatus_t st = CherTryAllocCopy(x, xBytes, bufs.dX);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return CherTryAllocCopy(A, aBytes, bufs.dA);
}

static aclblasStatus_t CherFinalizeNpu(
    aclblasStatus_t status, aclblasComplex* A, size_t aBytes, const CherDeviceBuffers& bufs)
{
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (A != nullptr && bufs.dA != nullptr) {
        if (aclrtMemcpy(A, aBytes, bufs.dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return status;
}

inline aclblasStatus_t aclblasCher_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* A, int lda, const CherNpuOptions& options = {})
{
    // Validation-only paths (null handle / empty / unsafe stride): do not allocate.
    if (handle == nullptr || n <= 0 || incx == 0 || incx == INT_MIN) {
        return aclblasCher(handle, uplo, n, alpha, x, incx, A, lda);
    }

    size_t xBytes = 0;
    size_t aBytes = 0;
    CherBufferBytes(n, incx, lda, x, A, xBytes, aBytes);

    CherDeviceBuffers bufs;
    aclblasStatus_t st = CherPrepareDeviceBuffers(alpha, x, A, xBytes, aBytes, options, bufs);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    const float* alphaArg = bufs.dAlpha ? static_cast<const float*>(bufs.dAlpha) : alpha;
    st = aclblasCher(
        handle, uplo, n, alphaArg, bufs.dX ? static_cast<const aclblasComplex*>(bufs.dX) : x, incx,
        bufs.dA ? static_cast<aclblasComplex*>(bufs.dA) : A, lda);
    return CherFinalizeNpu(st, A, aBytes, bufs);
}
