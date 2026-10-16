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
#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t SsprAllocCopyH2D(void*& device, const float* host, size_t bytes)
{
    device = nullptr;
    if (host == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(device);
        device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// NPU wrapper for aclblasSspr — same signature as the API.
// For n <= 0 or null handle, passes through directly to hit host-side
// parameter validation / early-return paths.
// Otherwise, allocates device memory, copies H2D, invokes kernel,
// synchronises, copies D2H, and frees.
inline aclblasStatus_t aclblasSspr_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* ap)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSspr(handle, uplo, n, alpha, x, incx, ap);
    }

    const int allocN = std::max(1, n);
    const int absIncx = std::abs(incx);
    const size_t xBytes = static_cast<size_t>((allocN - 1) * absIncx + 1) * sizeof(float);
    const size_t apBytes = static_cast<size_t>(allocN) * (allocN + 1) / 2 * sizeof(float);

    void* dX = nullptr;
    void* dAP = nullptr;
    aclblasStatus_t copyStatus = SsprAllocCopyH2D(dX, x, xBytes);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    copyStatus = SsprAllocCopyH2D(dAP, ap, apBytes);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        if (dX) {
            aclrtFree(dX);
        }
        return copyStatus;
    }

    aclblasStatus_t ret =
        aclblasSspr(handle, uplo, n, alpha, static_cast<const float*>(dX), incx, static_cast<float*>(dAP));

    aclrtSynchronizeDevice();
    if (ret == ACLBLAS_STATUS_SUCCESS && ap != nullptr) {
        aclrtMemcpy(ap, apBytes, dAP, apBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (dX) {
        aclrtFree(dX);
    }
    if (dAP) {
        aclrtFree(dAP);
    }
    return ret;
}
