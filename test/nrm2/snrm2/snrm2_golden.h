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

#include "cblas_compat.h"

// Reference implementation — same signature as aclblasSnrm2
inline aclblasStatus_t aclblasSnrm2_cpu(
    aclblasHandle_t handle, const int64_t n, const float* x, const int64_t incx, float* result)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (x == nullptr || result == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n <= 0 || incx <= 0) {
        *result = 0.0f;
        return ACLBLAS_STATUS_SUCCESS;
    }

    *result = cblas_snrm2(static_cast<int>(n), x, static_cast<int>(incx));
    return ACLBLAS_STATUS_SUCCESS;
}
