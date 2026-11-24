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
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "device.h"

inline std::unique_ptr<DeviceBuffer> tryAllocAndCopyScasum(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr)
        return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

// Host -> Device wrapper. Input length follows the physical span (n-1)*incx+1;
// span-overflow cases use a single dummy complex element.
inline aclblasStatus_t aclblasScasum_npu(
    aclblasHandle_t handle, int64_t n, const aclblasComplex* x, int64_t incx, float* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const bool quickReturn = n <= 0 || incx <= 0;
    const bool spanOverflow = !quickReturn &&
                              static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx) * 2 >
                                  static_cast<int64_t>(UINT32_MAX);
    const size_t complexLen = quickReturn ? 0 : (spanOverflow ? 1 : static_cast<size_t>(n - 1) * incx + 1);
    const size_t xBytes = complexLen * 2 * sizeof(float);
    const size_t resultBytes = sizeof(float);

    auto dX = quickReturn ? nullptr : tryAllocAndCopyScasum(x, xBytes);

    if (result == nullptr) {
        aclblasStatus_t ret = aclblasScasum(
            handle, static_cast<int>(n), dX ? reinterpret_cast<const aclblasComplex*>(dX->ptr()) : nullptr,
            static_cast<int>(incx), nullptr);
        if (ret == ACL_SUCCESS && aclrtSynchronizeDevice() != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return ret;
    }

    auto dResult = std::make_unique<DeviceBuffer>(resultBytes);

    aclblasStatus_t ret = aclblasScasum(
        handle, static_cast<int>(n), dX ? reinterpret_cast<const aclblasComplex*>(dX->ptr()) : nullptr,
        static_cast<int>(incx), reinterpret_cast<float*>(dResult->ptr()));

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    dResult->copyToHost(result, resultBytes);

    return ret;
}
