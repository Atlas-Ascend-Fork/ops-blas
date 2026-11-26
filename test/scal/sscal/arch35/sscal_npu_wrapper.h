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
#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasSscal_npu(
    aclblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    if (handle == nullptr || n <= 0 || x == nullptr) {
        return aclblasSscal(handle, n, alpha, x, incx);
    }

    int absInc = std::abs(incx);
    size_t xElements = static_cast<size_t>((n - 1) * absInc + 1);
    const size_t xBytes = xElements * sizeof(float);

    void* dX = nullptr;
    aclError aclRet;

    aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret = aclblasSscal(handle, n, alpha, static_cast<float*>(dX), incx);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (ret == ACLBLAS_STATUS_SUCCESS) {
        if (aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    aclrtFree(dX);

    return ret;
}

