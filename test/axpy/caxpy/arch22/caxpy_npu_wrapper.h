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

#include <cstdint>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "fill.h"

static inline bool CaxpyNeedPassThrough(aclblasHandle_t handle, int n) { return handle == nullptr || n <= 0; }

static inline aclError CaxpyAllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr || bytes == 0) {
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

static inline void CaxpyFreeAll(void* dX, void* dY)
{
    if (dX) {
        aclrtFree(dX);
    }
    if (dY) {
        aclrtFree(dY);
    }
}

// Physical storage span (in complex elements) covered by a length-n vector
// with the given (possibly negative) increment.
static inline size_t CaxpySpanElements(int n, int increment)
{
    if (n <= 0) {
        return 0;
    }
    const uint64_t absIncrement = increment >= 0 ? static_cast<uint64_t>(increment) :
                                                   static_cast<uint64_t>(-(static_cast<int64_t>(increment) + 1)) + 1;
    const uint64_t logicalDistance = static_cast<uint64_t>(n - 1);
    if (absIncrement != 0 &&
        logicalDistance > (std::numeric_limits<size_t>::max() / sizeof(aclblasComplex) - 1) / absIncrement) {
        return 0;
    }
    return static_cast<size_t>(logicalDistance * absIncrement + 1);
}

inline aclblasStatus_t aclblasCaxpy_npu(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, aclblasComplex* y,
    int incy)
{
    if (CaxpyNeedPassThrough(handle, n)) {
        return aclblasCaxpy(handle, n, alpha, x, incx, y, incy);
    }

    const size_t xElements = CaxpySpanElements(n, incx);
    const size_t yElements = CaxpySpanElements(n, incy);
    if ((n > 0 && xElements == 0) || (n > 0 && yElements == 0)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const size_t xBytes = xElements * sizeof(aclblasComplex);
    const size_t yBytes = yElements * sizeof(aclblasComplex);

    void* dX = nullptr;
    void* dY = nullptr;

    if (CaxpyAllocCopyH2D(dX, x, xBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (CaxpyAllocCopyH2D(dY, y, yBytes) != ACL_SUCCESS) {
        CaxpyFreeAll(dX, dY);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = aclblasCaxpy(
        handle, n, alpha, static_cast<const aclblasComplex*>(dX), incx, static_cast<aclblasComplex*>(dY), incy);

    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        CaxpyFreeAll(dX, dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS && y != nullptr && dY != nullptr) {
        if (aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            CaxpyFreeAll(dX, dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    CaxpyFreeAll(dX, dY);
    return ret;
}
