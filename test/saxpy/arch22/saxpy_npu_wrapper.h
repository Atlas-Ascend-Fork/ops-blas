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
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace saxpy_arch22_detail {
constexpr size_t SAXPY_DUMMY_ELEMENTS = 64U;

inline size_t SpanElements(int n, int increment)
{
    if (n <= 0) {
        return 0U;
    }
    const int64_t absIncrement = (increment < 0) ? -static_cast<int64_t>(increment) : static_cast<int64_t>(increment);
    return static_cast<size_t>(static_cast<int64_t>(n - 1) * absIncrement + 1);
}

inline aclError AllocCopyH2D(void*& devicePtr, const void* hostPtr, size_t bytes)
{
    devicePtr = nullptr;
    if (hostPtr == nullptr || bytes == 0U) {
        return ACL_SUCCESS;
    }
    aclError ret = aclrtMalloc(&devicePtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(devicePtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(devicePtr);
        devicePtr = nullptr;
    }
    return ret;
}

inline void FreeAll(void* first, void* second)
{
    if (first != nullptr) {
        aclrtFree(first);
    }
    if (second != nullptr) {
        aclrtFree(second);
    }
}
} // namespace saxpy_arch22_detail

inline aclblasStatus_t AclblasSaxpyOnDevice(
    aclblasHandle_t handle, int n, const float* alpha, const std::vector<float>& xHost, int incx,
    std::vector<float>& yHost, int incy)
{
    using namespace saxpy_arch22_detail;
    void* deviceX = nullptr;
    void* deviceY = nullptr;
    const size_t xBytes = xHost.size() * sizeof(float);
    const size_t yBytes = yHost.size() * sizeof(float);

    if (AllocCopyH2D(deviceX, xHost.data(), xBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (AllocCopyH2D(deviceY, yHost.data(), yBytes) != ACL_SUCCESS) {
        FreeAll(deviceX, deviceY);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t ret =
        aclblasSaxpy(handle, n, alpha, static_cast<float*>(deviceX), incx, static_cast<float*>(deviceY), incy);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        FreeAll(deviceX, deviceY);
        return ret;
    }

    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || stream == nullptr) {
        FreeAll(deviceX, deviceY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        FreeAll(deviceX, deviceY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (!yHost.empty() && deviceY != nullptr) {
        if (aclrtMemcpy(yHost.data(), yBytes, deviceY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            FreeAll(deviceX, deviceY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    FreeAll(deviceX, deviceY);
    return ret;
}

// Skip allocation for negative or overflowing spans.
inline aclblasStatus_t AclblasSaxpyValidation(
    aclblasHandle_t handle, int n, const float* alpha, bool xNull, int incx, bool yNull, int incy)
{
    using namespace saxpy_arch22_detail;
    if (n <= 0) {
        float dummy = 0.0F;
        return aclblasSaxpy(handle, n, alpha, xNull ? nullptr : &dummy, incx, yNull ? nullptr : &dummy, incy);
    }

    float dummyHost[SAXPY_DUMMY_ELEMENTS] = {0.0F};
    void* deviceX = nullptr;
    void* deviceY = nullptr;
    if (!xNull && AllocCopyH2D(deviceX, dummyHost, sizeof(dummyHost)) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (!yNull && AllocCopyH2D(deviceY, dummyHost, sizeof(dummyHost)) != ACL_SUCCESS) {
        FreeAll(deviceX, deviceY);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t ret =
        aclblasSaxpy(handle, n, alpha, static_cast<float*>(deviceX), incx, static_cast<float*>(deviceY), incy);
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        FreeAll(deviceX, deviceY);
        return (ret == ACLBLAS_STATUS_SUCCESS) ? ACLBLAS_STATUS_INTERNAL_ERROR : ret;
    }

    FreeAll(deviceX, deviceY);
    return ret;
}
