/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAXPY_GOLDEN_H
#define CAXPY_GOLDEN_H

#include <cstdint>

#include "cann_ops_blas.h"

inline int64_t caxpyCpuOffset(int n, int increment, int logicalIndex)
{
    int64_t absIncrement = increment >= 0 ? static_cast<int64_t>(increment) : -static_cast<int64_t>(increment);
    return increment >= 0 ? static_cast<int64_t>(logicalIndex) * absIncrement :
                            static_cast<int64_t>(n - 1 - logicalIndex) * absIncrement;
}

inline aclblasStatus_t aclblasCaxpy_cpu(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (alpha == nullptr || x == nullptr || y == nullptr || incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    for (int i = 0; i < n; ++i) {
        const aclblasComplex& xValue = x[caxpyCpuOffset(n, incx, i)];
        aclblasComplex& yValue = y[caxpyCpuOffset(n, incy, i)];
        float yReal = alpha->real * xValue.real - alpha->imag * xValue.imag + yValue.real;
        float yImag = alpha->real * xValue.imag + alpha->imag * xValue.real + yValue.imag;
        yValue.real = yReal;
        yValue.imag = yImag;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CAXPY_GOLDEN_H
