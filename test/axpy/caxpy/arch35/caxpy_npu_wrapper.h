/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAXPY_NPU_WRAPPER_H
#define CAXPY_NPU_WRAPPER_H

#include <cstddef>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline bool caxpyGetBufferBytes(int n, int increment, size_t& bytes)
{
    uint64_t absIncrement = increment >= 0 ? static_cast<uint64_t>(increment) : -static_cast<int64_t>(increment);
    uint64_t elementCount = absIncrement * (static_cast<uint64_t>(n) - 1U) + 1U;
    if (elementCount > std::numeric_limits<size_t>::max() / sizeof(aclblasComplex)) {
        return false;
    }
    bytes = static_cast<size_t>(elementCount) * sizeof(aclblasComplex);
    return true;
}

inline void caxpyFreeDevice(void*& pointer)
{
    if (pointer != nullptr) {
        aclrtFree(pointer);
        pointer = nullptr;
    }
}

inline aclblasStatus_t aclblasCaxpy_npu(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* y, int incy)
{
    if (handle == nullptr || n <= 0 || alpha == nullptr || x == nullptr || y == nullptr || incx == 0 || incy == 0) {
        return aclblasCaxpy(handle, n, alpha, x, incx, y, incy);
    }

    size_t xBytes = 0;
    size_t yBytes = 0;
    if (!caxpyGetBufferBytes(n, incx, xBytes) || !caxpyGetBufferBytes(n, incy, yBytes)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    void* deviceX = nullptr;
    void* deviceY = nullptr;
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    if (aclrtMalloc(&deviceX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&deviceY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_ALLOC_FAILED;
    } else if (aclrtMemcpy(deviceX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
               aclrtMemcpy(deviceY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else {
        status = aclblasCaxpy(
            handle, n, alpha, static_cast<const aclblasComplex*>(deviceX), incx,
            static_cast<aclblasComplex*>(deviceY), incy);
        if (status == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeDevice() != ACL_SUCCESS) {
            status = ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        if (status == ACLBLAS_STATUS_SUCCESS &&
            aclrtMemcpy(y, yBytes, deviceY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            status = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    caxpyFreeDevice(deviceX);
    caxpyFreeDevice(deviceY);
    return status;
}

#endif // CAXPY_NPU_WRAPPER_H
