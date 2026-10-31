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

#include "acl/acl.h"
#include "cann_ops_blas.h"

// NPU wrapper — same signature as aclblasSswap.
// nullptr inputs are passed through without device allocation (for error-path testing).
// n<=0 cases are passed through directly to the API (returns SUCCESS without further validation).

static inline aclblasStatus_t CopyToDevice(void** dPtr, const void* hPtr, size_t bytes)
{
    if (hPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(*dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*dPtr);
        *dPtr = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t CopyFromDevice(void* hPtr, const void* dPtr, size_t bytes)
{
    if (hPtr == nullptr || dPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMemcpy(hPtr, bytes, dPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

static inline void FreeDevice(void* dX, void* dY)
{
    if (dX)
        aclrtFree(dX);
    if (dY)
        aclrtFree(dY);
}

inline aclblasStatus_t aclblasSswap_npu(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSswap(handle, n, x, incx, y, incy);
    }

    const auto storageBytes = [n](int inc) {
        const int64_t absInc = inc < 0 ? -static_cast<int64_t>(inc) : inc;
        return (static_cast<size_t>(1 + (n - 1) * absInc)) * sizeof(float);
    };
    const size_t xBytes = storageBytes(incx);
    const size_t yBytes = storageBytes(incy);

    void* dX = nullptr;
    void* dY = nullptr;

    aclblasStatus_t status = CopyToDevice(&dX, x, xBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = CopyToDevice(&dY, y, yBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        FreeDevice(dX, dY);
        return status;
    }

    aclblasStatus_t ret = aclblasSswap(handle, n, static_cast<float*>(dX), incx, static_cast<float*>(dY), incy);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        FreeDevice(dX, dY);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    status = CopyFromDevice(x, dX, xBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        FreeDevice(dX, dY);
        return status;
    }
    status = CopyFromDevice(y, dY, yBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        FreeDevice(dX, dY);
        return status;
    }

    FreeDevice(dX, dY);
    return ret;
}
