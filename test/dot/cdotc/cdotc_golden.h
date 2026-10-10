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

#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t aclblasCdotc_cpu(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    aclblasComplex* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        if (result == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        *result = aclblasComplex{0.0f, 0.0f};
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (incx == 0 || incy == 0 || x == nullptr || y == nullptr || result == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    cblas_cdotc_sub(n, x, incx, y, incy, result);
    return ACLBLAS_STATUS_SUCCESS;
}
