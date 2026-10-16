/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSWAP_NPU_WRAPPER_H
#define CSWAP_NPU_WRAPPER_H

#include <cstddef>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline bool cswap_get_buffer_layout(
    int n, int increment, size_t offsetElements, size_t& offsetBytes, size_t& totalBytes)
{
    uint64_t absIncrement =
        increment > 0 ? static_cast<uint64_t>(increment) : static_cast<uint64_t>(-static_cast<int64_t>(increment));
    uint64_t spanElements = absIncrement * (static_cast<uint64_t>(n) - 1) + 1;
    constexpr size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
    if (spanElements > maxElements) {
        return false;
    }
    size_t span = static_cast<size_t>(spanElements);
    if (offsetElements > maxElements - span) {
        return false;
    }
    offsetBytes = offsetElements * sizeof(aclblasComplex);
    totalBytes = (span + offsetElements) * sizeof(aclblasComplex);
    return true;
}

inline aclblasStatus_t cswap_copy_to_device(const void* host, size_t bytes, void** device)
{
    if (host == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline void cswap_free_device(void*& device)
{
    if (device != nullptr) {
        aclrtFree(device);
        device = nullptr;
    }
}

inline aclblasStatus_t aclblasCswap_npu(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, size_t xOffsetElements = 0,
    size_t yOffsetElements = 0)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCswap(handle, n, x, incx, y, incy);
    }

    size_t xOffsetBytes = 0;
    size_t yOffsetBytes = 0;
    size_t xBytes = 0;
    size_t yBytes = 0;
    if (!cswap_get_buffer_layout(n, incx, xOffsetElements, xOffsetBytes, xBytes) ||
        !cswap_get_buffer_layout(n, incy, yOffsetElements, yOffsetBytes, yBytes)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    void* xHostBase = x == nullptr ? nullptr : reinterpret_cast<uint8_t*>(x) - xOffsetBytes;
    void* yHostBase = y == nullptr ? nullptr : reinterpret_cast<uint8_t*>(y) - yOffsetBytes;
    void* deviceX = nullptr;
    void* deviceY = nullptr;
    aclblasStatus_t status = cswap_copy_to_device(xHostBase, xBytes, &deviceX);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = cswap_copy_to_device(yHostBase, yBytes, &deviceY);
    }
    if (status != ACLBLAS_STATUS_SUCCESS) {
        cswap_free_device(deviceX);
        cswap_free_device(deviceY);
        return status;
    }

    auto* logicalDeviceX =
        deviceX == nullptr ? nullptr : reinterpret_cast<aclblasComplex*>(static_cast<uint8_t*>(deviceX) + xOffsetBytes);
    auto* logicalDeviceY =
        deviceY == nullptr ? nullptr : reinterpret_cast<aclblasComplex*>(static_cast<uint8_t*>(deviceY) + yOffsetBytes);
    status = aclblasCswap(handle, n, logicalDeviceX, incx, logicalDeviceY, incy);
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeDevice() != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (status == ACLBLAS_STATUS_SUCCESS &&
        (aclrtMemcpy(xHostBase, xBytes, deviceX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
         aclrtMemcpy(yHostBase, yBytes, deviceY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    cswap_free_device(deviceX);
    cswap_free_device(deviceY);
    return status;
}

#endif // CSWAP_NPU_WRAPPER_H
