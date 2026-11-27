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
#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Host -> device staging for aclblasStpsv (arch22 host side takes device pointers).
// AP device length is n*(n+1)/2, x device length is 1 + (n-1)*|incx|.
inline aclblasStatus_t aclblasStpsv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* ap, float* x, int incx)
{
    // Anything that cannot be staged is forwarded so the host validates and reports it.
    if (handle == nullptr || n <= 0 || ap == nullptr || x == nullptr || incx == 0) {
        return aclblasStpsv(handle, uplo, trans, diag, n, ap, x, incx);
    }

    const int absIncx = std::abs(incx);
    const size_t apBytes = static_cast<size_t>(n) * (n + 1) / 2 * sizeof(float);
    const size_t xBytes = static_cast<size_t>((n - 1) * absIncx + 1) * sizeof(float);

    void* dAp = nullptr;
    void* dX = nullptr;

    if (aclrtMalloc(&dAp, apBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(dAp, apBytes, ap, apBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(dAp);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        aclrtFree(dAp);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(dAp);
        aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret =
        aclblasStpsv(handle, uplo, trans, diag, n, static_cast<const float*>(dAp), static_cast<float*>(dX), incx);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (ret == ACLBLAS_STATUS_SUCCESS) {
        if (aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    aclrtFree(dAp);
    aclrtFree(dX);
    return ret;
}
