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

#include <climits>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Check if the call should pass through directly to the API (null handle, error cases,
// no-op cases, or INT_MIN to avoid -incx/-incy UB).
static inline bool NeedPassThrough(
    aclblasHandle_t handle, int m, int n, int incx, int incy)
{
    return handle == nullptr || m < 0 || n < 0 || m == 0 || n == 0 ||
           incx == INT_MIN || incy == INT_MIN;
}

// Allocate device memory and copy host data. On success sets dPtr and returns SUCCESS.
// On failure frees any partial allocation, sets dPtr=nullptr, and returns the aclError.
static inline aclError AllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr) return ACL_SUCCESS;
    aclError ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtMemcpy(dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(dPtr); dPtr = nullptr; }
    return ret;
}

// NPU wrapper for aclblasSger.
// nullptr inputs are passed through without device allocation (for error-path testing).
inline aclblasStatus_t aclblasSger_npu(
    aclblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx,
    const float* y, int incy, float* A, int lda)
{
    if (NeedPassThrough(handle, m, n, incx, incy)) {
        return aclblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
    }

    const int absIncX = (incx >= 0) ? incx : -incx;
    const int absIncY = (incy >= 0) ? incy : -incy;
    const size_t xBytes = static_cast<size_t>(m) * static_cast<size_t>(absIncX) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(n) * static_cast<size_t>(absIncY) * sizeof(float);
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float);

    // alpha: host-side scalar (cuBLAS convention), passed directly — no Device copy
    void* dX = nullptr;
    void* dY = nullptr;
    void* dA = nullptr;

    aclError aclRet = AllocCopyH2D(dX, x, xBytes);
    if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;

    aclRet = AllocCopyH2D(dY, y, yBytes);
    if (aclRet != ACL_SUCCESS) {
        if (dX) aclrtFree(dX);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = AllocCopyH2D(dA, A, aBytes);
    if (aclRet != ACL_SUCCESS) {
        if (dX) aclrtFree(dX);
        if (dY) aclrtFree(dY);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = aclblasSger(handle, m, n, alpha,
        static_cast<const float*>(dX), incx,
        static_cast<const float*>(dY), incy,
        static_cast<float*>(dA), lda);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        if (dX) aclrtFree(dX);
        if (dY) aclrtFree(dY);
        if (dA) aclrtFree(dA);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS && A != nullptr) {
        if (aclrtMemcpy(A, aBytes, dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (dX) aclrtFree(dX);
    if (dY) aclrtFree(dY);
    if (dA) aclrtFree(dA);
    return ret;
}

