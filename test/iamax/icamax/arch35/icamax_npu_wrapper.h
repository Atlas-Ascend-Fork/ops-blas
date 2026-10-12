/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

inline aclblasStatus_t aclblasIcamax_npu(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    if (handle == nullptr) {
        return aclblasIcamax(nullptr, n, nullptr, incx, nullptr);
    }

    std::unique_ptr<DeviceBuffer> deviceX;
    if (n > 0 && incx > 0 && x != nullptr) {
        const size_t xLength = static_cast<size_t>(n - 1) * static_cast<size_t>(incx) + 1U;
        const size_t xBytes = xLength * sizeof(aclblasComplex);
        deviceX = std::make_unique<DeviceBuffer>(xBytes);
        deviceX->copyFromHost(x, xBytes);
    }

    if (result == nullptr) {
        return aclblasIcamax(
            handle, n, deviceX ? static_cast<const aclblasComplex*>(deviceX->ptr()) : nullptr, incx, nullptr);
    }

    auto deviceResult = std::make_unique<DeviceBuffer>(sizeof(int));
    const aclblasStatus_t status = aclblasIcamax(
        handle, n, deviceX ? static_cast<const aclblasComplex*>(deviceX->ptr()) : nullptr, incx,
        static_cast<int*>(deviceResult->ptr()));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    deviceResult->copyToHost(result, sizeof(int));
    return status;
}
