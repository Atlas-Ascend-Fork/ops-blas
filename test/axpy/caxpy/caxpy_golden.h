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

#include "cann_ops_blas.h"

inline aclblasStatus_t CaxpyValidateParams(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy)
{
    if (handle == nullptr || alpha == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0 || incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && (x == nullptr || y == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasComplex CaxpyComplexMac(aclblasComplex alpha, aclblasComplex x, aclblasComplex y)
{
    return {alpha.real * x.real - alpha.imag * x.imag + y.real, alpha.real * x.imag + alpha.imag * x.real + y.imag};
}

// Logical-index -> physical storage index for a strided complex vector.
// Positive stride reads forward from the base; negative stride is BLAS
// convention: the last logical element is physically first.
inline int64_t CaxpyStorageIndex(int64_t logicalIndex, int64_t n, int64_t increment)
{
    const uint64_t absIncrement = increment > 0 ? static_cast<uint64_t>(increment) : static_cast<uint64_t>(-increment);
    return increment > 0 ? logicalIndex * static_cast<int64_t>(absIncrement) :
                           (n - 1 - logicalIndex) * static_cast<int64_t>(absIncrement);
}

// y = alpha * x + y, complex, arbitrary (possibly negative) incx/incy.
// x and y are the caller-owned base storage pointers (not offset by any guard region).
inline aclblasStatus_t aclblasCaxpy_cpu(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, aclblasComplex* y,
    int incy)
{
    aclblasStatus_t st = CaxpyValidateParams(handle, n, alpha, x, incx, y, incy);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    for (int64_t i = 0; i < n; i++) {
        const int64_t xIdx = CaxpyStorageIndex(i, n, incx);
        const int64_t yIdx = CaxpyStorageIndex(i, n, incy);
        y[yIdx] = CaxpyComplexMac(*alpha, x[xIdx], y[yIdx]);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
