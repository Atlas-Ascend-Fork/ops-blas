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

// NPU wrapper for aclblasSspr — same signature as the API.
// For n <= 0 or null handle, passes through directly to hit host-side
// parameter validation / early-return paths.
// Otherwise, allocates device memory, copies H2D, invokes kernel,
// synchronises, copies D2H, and frees.
inline aclblasStatus_t aclblasSspr_npu(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const float* alpha,
    const float* x,
    int incx,
    float* ap)
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
    aclError aclRet;

    if (x != nullptr) {
        aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { aclrtFree(dX); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (ap != nullptr) {
        aclRet = aclrtMalloc(&dAP, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { if (dX) aclrtFree(dX); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(dAP, apBytes, ap, apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { if (dX) aclrtFree(dX); aclrtFree(dAP); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }

    aclblasStatus_t ret = aclblasSspr(handle, uplo, n, alpha,
        static_cast<const float*>(dX), incx, static_cast<float*>(dAP));

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (ret == ACLBLAS_STATUS_SUCCESS && ap != nullptr) {
        if (aclrtMemcpy(ap, apBytes, dAP, apBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (dX) aclrtFree(dX);
    if (dAP) aclrtFree(dAP);
    return ret;
}
