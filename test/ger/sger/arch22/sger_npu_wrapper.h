/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGER_NPU_WRAPPER_H
#define SGER_NPU_WRAPPER_H

#include <algorithm>
#include <climits>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Short-circuit pre-checks: null handle, dim errors, no-op cases, and INT_MIN.
// INT_MIN must be caught here because `-incx` overflow on INT_MIN is signed UB
// in the byte-count math below.
static inline bool NeedPassThrough(aclblasHandle_t handle, int64_t m, int64_t n, int64_t incx, int64_t incy)
{
    return handle == nullptr || m < 0 || n < 0 || m == 0 || n == 0 || incx == INT_MIN || incy == INT_MIN || incx == 0 ||
           incy == 0;
}

// Allocate device memory and copy host data. On success sets dPtr and returns SUCCESS.
// On failure frees any partial allocation, sets dPtr=nullptr, and returns the aclError.
// A null host pointer is treated as "no allocation needed" and returns SUCCESS with dPtr=nullptr,
// so error-path tests can pass nullptr through to the kernel for validation.
static inline aclError AllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr || bytes == 0)
        return ACL_SUCCESS;
    aclError ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    ret = aclrtMemcpy(dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(dPtr);
        dPtr = nullptr;
    }
    return ret;
}

inline aclblasStatus_t aclblasSger_npu(
    aclblasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, float* y,
    int64_t incy, float* A, int64_t lda)
{
    if (NeedPassThrough(handle, m, n, incx, incy)) {
        if (handle == nullptr)
            return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
        if (m < 0 || n < 0 || incx == INT_MIN || incy == INT_MIN || incx == 0 || incy == 0) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        return ACLBLAS_STATUS_SUCCESS; // m == 0 or n == 0
    }
    if (lda < std::max<int64_t>(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const int64_t absIncx = (incx < 0) ? -incx : incx;
    const int64_t absIncy = (incy < 0) ? -incy : incy;

    const size_t xBytes = static_cast<size_t>((m - 1) * absIncx + 1) * sizeof(float);
    const size_t yBytes = static_cast<size_t>((n - 1) * absIncy + 1) * sizeof(float);
    // Column-major A: total size = lda rows * n cols.
    const size_t aBytes = static_cast<size_t>(n) * lda * sizeof(float);

    void* dX = nullptr;
    void* dY = nullptr;
    void* dA = nullptr;

    aclError aclRet = AllocCopyH2D(dX, x, xBytes);
    if (aclRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;

    aclRet = AllocCopyH2D(dY, y, yBytes);
    if (aclRet != ACL_SUCCESS) {
        if (dX)
            aclrtFree(dX);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = AllocCopyH2D(dA, A, aBytes);
    if (aclRet != ACL_SUCCESS) {
        if (dX)
            aclrtFree(dX);
        if (dY)
            aclrtFree(dY);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = aclblasSger(
        handle, m, n, alpha, static_cast<const float*>(dX), incx, static_cast<float*>(dY), incy,
        static_cast<float*>(dA), lda);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (ret == ACLBLAS_STATUS_SUCCESS && A != nullptr && dA != nullptr) {
        if (aclrtMemcpy(A, aBytes, dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (dA)
        aclrtFree(dA);
    if (dY)
        aclrtFree(dY);
    if (dX)
        aclrtFree(dX);

    return ret;
}

#endif // SGER_NPU_WRAPPER_H
