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

#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

inline aclblasStatus_t aclblasSnrm2Ex_npu(
    aclblasHandle_t handle, aclDataType xtype, const void* hX, int64_t n, int64_t incx, float* hResult)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    const size_t elemSize = (xtype == ACL_FLOAT16) ? sizeof(uint16_t) : sizeof(float);
    size_t xBytes = 0;
    if (hX != nullptr && n > 0 && incx != 0 && incx != static_cast<int64_t>(INT32_MIN) && incx != INT64_MIN) {
        const int64_t absInc = std::abs(incx);
        const int64_t nMinus1 = n - 1;
        if (nMinus1 > 0 && absInc > (INT64_MAX - 1) / nMinus1) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        xBytes = static_cast<size_t>(nMinus1 * absInc + 1) * elemSize;
    }

    std::unique_ptr<DeviceBuffer> dX;
    if (hX != nullptr && xBytes > 0) {
        dX = std::make_unique<DeviceBuffer>(xBytes);
        dX->copyFromHost(hX, xBytes);
    }

    std::unique_ptr<DeviceBuffer> dResult;
    if (hResult != nullptr) {
        dResult = std::make_unique<DeviceBuffer>(sizeof(float));
    }

    const aclblasStatus_t ret =
        aclblasSnrm2Ex(handle, xtype, dX ? dX->ptr() : nullptr, n, incx, dResult ? dResult->ptr() : nullptr);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (hResult != nullptr && dResult) {
        dResult->copyToHost(hResult, sizeof(float));
    }
    return ret;
}
