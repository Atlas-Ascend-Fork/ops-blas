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
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "fill.h"

static inline bool SdgmmNeedPassThrough(aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, int incx)
{
    return handle == nullptr || m <= 0 || n <= 0 || incx == INT_MIN ||
           (mode != ACLBLAS_SIDE_LEFT && mode != ACLBLAS_SIDE_RIGHT);
}

static inline aclError SdgmmAllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr) {
        return ACL_SUCCESS;
    }
    aclError ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(dPtr);
        dPtr = nullptr;
    }
    return ret;
}

static inline void SdgmmFreeAll(void* dX, void* dA, void* dC)
{
    if (dX != nullptr) {
        aclrtFree(dX);
    }
    if (dA != nullptr) {
        aclrtFree(dA);
    }
    if (dC != nullptr) {
        aclrtFree(dC);
    }
}

static inline aclError SdgmmAllocAndFillC(void*& dC, float* C, size_t cBytes)
{
    dC = nullptr;
    if (C == nullptr) {
        return ACL_SUCCESS;
    }
    aclError ret = aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    std::vector<float> sentinelBuf(cBytes / sizeof(float), kBlasSentinel);
    ret = aclrtMemcpy(dC, cBytes, sentinelBuf.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(dC);
        dC = nullptr;
    }
    return ret;
}

inline aclblasStatus_t aclblasSdgmm_npu(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx,
    float* C, int ldc)
{
    if (SdgmmNeedPassThrough(handle, mode, m, n, incx)) {
        return aclblasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    }

    const int xLen = (mode == ACLBLAS_SIDE_LEFT) ? m : n;
    const int64_t absIncx = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const size_t xTotalEl = static_cast<size_t>(xLen - 1) * static_cast<size_t>(absIncx) + 1;
    const size_t xBytes = xTotalEl * sizeof(float);
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);

    void* dX = nullptr;
    void* dA = nullptr;
    void* dC = nullptr;
    if (SdgmmAllocCopyH2D(dX, x, xBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (SdgmmAllocCopyH2D(dA, A, aBytes) != ACL_SUCCESS) {
        SdgmmFreeAll(dX, dA, dC);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (SdgmmAllocAndFillC(dC, C, cBytes) != ACL_SUCCESS) {
        SdgmmFreeAll(dX, dA, dC);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = aclblasSdgmm(
        handle, mode, m, n, static_cast<const float*>(dA), lda, static_cast<const float*>(dX), incx,
        static_cast<float*>(dC), ldc);
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        SdgmmFreeAll(dX, dA, dC);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ret == ACLBLAS_STATUS_SUCCESS && C != nullptr && dC != nullptr) {
        if (aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            SdgmmFreeAll(dX, dA, dC);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    SdgmmFreeAll(dX, dA, dC);
    return ret;
}
