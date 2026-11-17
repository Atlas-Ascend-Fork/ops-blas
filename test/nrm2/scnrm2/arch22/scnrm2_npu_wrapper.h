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
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

inline std::unique_ptr<DeviceBuffer> tryAllocAndCopyScnrm2(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr) {
        return nullptr;
    }
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline aclblasStatus_t aclblasScnrm2_npu(
    aclblasHandle_t handle, int64_t n, const aclblasComplex* x, int64_t incx, float* result)
{
    // arch22 aclblasScnrm2 缺少 handle/result 判空守卫，空指针直接透传会在
    // host 侧解引用空 handle / 内核写空 result 指针；测试封装先行拦下，
    // 语义对齐 aclblasSnrm2 的参数校验口径（issue #920）。
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    if (result == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclrtStream stream = nullptr;
    aclblasStatus_t streamStatus = aclblasGetStream(handle, &stream);
    if (streamStatus != ACLBLAS_STATUS_SUCCESS) {
        return streamStatus;
    }

    std::unique_ptr<DeviceBuffer> dX;
    if (n > 0 && x != nullptr) {
        int64_t absInc = (incx > 0) ? incx : -incx;
        size_t complexCount = static_cast<size_t>(1 + (n - 1) * absInc);
        dX = tryAllocAndCopyScnrm2(x, complexCount * sizeof(aclblasComplex));
    }

    auto dResult = std::make_unique<DeviceBuffer>(sizeof(float));
    aclblasStatus_t ret = aclblasScnrm2(
        handle, static_cast<int>(n), dX ? static_cast<const aclblasComplex*>(dX->ptr()) : nullptr,
        static_cast<int>(incx), static_cast<float*>(dResult->ptr()));

    aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        dResult->copyToHost(result, sizeof(float));
    }
    return ret;
}
