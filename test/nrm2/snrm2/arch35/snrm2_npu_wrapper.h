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

#include <cmath>
#include <cstdint>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

inline std::unique_ptr<DeviceBuffer> tryAllocAndCopySnrm2(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr)
        return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline aclblasStatus_t aclblasSnrm2_npu(
    aclblasHandle_t handle, const int64_t n, const float* x, const int64_t incx, float* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const size_t resultBytes = sizeof(float);

    // If result is nullptr, pass nullptr to ACL for parameter validation testing.
    // Preserve the API's own return value for negative-test assertions; only fall
    // back to INTERNAL_ERROR when validation passed but async execution failed.
    if (result == nullptr) {
        aclblasStatus_t ret = aclblasSnrm2(handle, static_cast<int>(n), nullptr, static_cast<int>(incx), nullptr);
        if (ret == ACL_SUCCESS && aclrtSynchronizeDevice() != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return ret;
    }

    std::unique_ptr<DeviceBuffer> dX;
    if (n > 0 && incx > 0) {
        const size_t dataBytes = static_cast<size_t>((n - 1) * std::abs(incx) + 1) * sizeof(float);
        dX = tryAllocAndCopySnrm2(x, dataBytes);
    }

    auto dResult = std::make_unique<DeviceBuffer>(resultBytes);

    aclblasStatus_t ret = aclblasSnrm2(
        handle, static_cast<int>(n), dX ? static_cast<const float*>(dX->ptr()) : nullptr, static_cast<int>(incx),
        static_cast<float*>(dResult->ptr()));

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    dResult->copyToHost(result, resultBytes);

    return ret;
}
