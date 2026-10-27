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
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace cgeru_test_detail {

inline aclError AllocCopyH2D(void*& device, const void* host, size_t bytes)
{
    device = nullptr;
    if (host == nullptr || bytes == 0)
        return ACL_SUCCESS;
    aclError ret = aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    ret = aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(device);
        device = nullptr;
    }
    return ret;
}

inline bool MustPassThrough(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, const aclblasComplex* A, int lda)
{
    if (handle == nullptr || m <= 0 || n <= 0 || alpha == nullptr || incx == 0 || incy == 0 || lda < std::max(1, m)) {
        return true;
    }
    if (alpha->real == 0.0f && alpha->imag == 0.0f)
        return true;
    return x == nullptr || y == nullptr || A == nullptr;
}

inline bool ComplexVectorBytes(int length, int increment, size_t& bytes)
{
    const uint64_t stride =
        (increment < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(increment)) : static_cast<uint64_t>(increment);
    const uint64_t count = 1 + static_cast<uint64_t>(length - 1) * stride;
    if (count > std::numeric_limits<size_t>::max() / sizeof(aclblasComplex))
        return false;
    bytes = static_cast<size_t>(count) * sizeof(aclblasComplex);
    return true;
}

inline bool ComplexMatrixBytes(int lda, int n, size_t& bytes)
{
    const uint64_t count = static_cast<uint64_t>(lda) * static_cast<uint64_t>(n);
    if (count > std::numeric_limits<size_t>::max() / sizeof(aclblasComplex))
        return false;
    bytes = static_cast<size_t>(count) * sizeof(aclblasComplex);
    return true;
}

inline void FreeDevice(void* device, aclError& firstError)
{
    if (device == nullptr)
        return;
    const aclError ret = aclrtFree(device);
    if (firstError == ACL_SUCCESS && ret != ACL_SUCCESS)
        firstError = ret;
}

} // namespace cgeru_test_detail

inline aclblasStatus_t aclblasCgeru_npu(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    using namespace cgeru_test_detail;
    if (MustPassThrough(handle, m, n, alpha, x, incx, y, incy, A, lda)) {
        return aclblasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
    }

    size_t xBytes = 0;
    size_t yBytes = 0;
    size_t aBytes = 0;
    if (!ComplexVectorBytes(m, incx, xBytes) || !ComplexVectorBytes(n, incy, yBytes) ||
        !ComplexMatrixBytes(lda, n, aBytes)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    void* deviceX = nullptr;
    void* deviceY = nullptr;
    void* deviceA = nullptr;

    aclError aclRet = AllocCopyH2D(deviceX, x, xBytes);
    if (aclRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    aclRet = AllocCopyH2D(deviceY, y, yBytes);
    if (aclRet != ACL_SUCCESS) {
        aclError cleanupRet = ACL_SUCCESS;
        FreeDevice(deviceX, cleanupRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = AllocCopyH2D(deviceA, A, aBytes);
    if (aclRet != ACL_SUCCESS) {
        aclError cleanupRet = ACL_SUCCESS;
        FreeDevice(deviceX, cleanupRet);
        FreeDevice(deviceY, cleanupRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t status = aclblasCgeru(
        handle, m, n, alpha, static_cast<const aclblasComplex*>(deviceX), incx,
        static_cast<const aclblasComplex*>(deviceY), incy, static_cast<aclblasComplex*>(deviceA), lda);
    aclError runtimeRet = ACL_SUCCESS;
    if (status == ACLBLAS_STATUS_SUCCESS) {
        runtimeRet = aclrtSynchronizeDevice();
        if (runtimeRet == ACL_SUCCESS) {
            runtimeRet = aclrtMemcpy(A, aBytes, deviceA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }

    aclError cleanupRet = ACL_SUCCESS;
    FreeDevice(deviceX, cleanupRet);
    FreeDevice(deviceY, cleanupRet);
    FreeDevice(deviceA, cleanupRet);
    if (runtimeRet != ACL_SUCCESS || cleanupRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    return status;
}
