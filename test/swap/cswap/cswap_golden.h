/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSWAP_GOLDEN_H
#define CSWAP_GOLDEN_H

#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t aclblasCswap_cpu(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy)
{
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (x == nullptr || y == nullptr || incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    cblas_cswap(n, static_cast<void*>(x), incx, static_cast<void*>(y), incy);
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CSWAP_GOLDEN_H
