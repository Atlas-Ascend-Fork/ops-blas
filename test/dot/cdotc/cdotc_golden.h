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

#include <cstddef>

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

    // 参考实现改用 double 累加：大 n 下 FP32 顺序累加的舍入误差会超过被测
    // kernel 与真值的偏差（kernel 侧 Kahan/ordered 路径的累计误差 ~1e-3 量级，
    // 而 FP32 golden 自身误差在大 n 时可达 1e-1 量级），使 verifyScalar 把更
    // 接近真值的 kernel 输出误判为 FAIL。double 累加保证 golden 的精度显著
    // 高于被测实现与容差门限，参考值仅受输入 FP32 量化影响。
    double real = 0.0;
    double imag = 0.0;
    const int64_t absIncX = incx >= 0 ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t absIncY = incy >= 0 ? static_cast<int64_t>(incy) : -static_cast<int64_t>(incy);
    for (int i = 0; i < n; ++i) {
        const std::size_t xIndex = static_cast<std::size_t>(
            incx > 0 ? static_cast<int64_t>(i) * absIncX : static_cast<int64_t>(n - 1 - i) * absIncX);
        const std::size_t yIndex = static_cast<std::size_t>(
            incy > 0 ? static_cast<int64_t>(i) * absIncY : static_cast<int64_t>(n - 1 - i) * absIncY);
        const double xr = static_cast<double>(x[xIndex].real);
        const double xi = static_cast<double>(x[xIndex].imag);
        const double yr = static_cast<double>(y[yIndex].real);
        const double yi = static_cast<double>(y[yIndex].imag);
        real += xr * yr + xi * yi;
        imag += xr * yi - xi * yr;
    }
    *result = aclblasComplex{static_cast<float>(real), static_cast<float>(imag)};
    return ACLBLAS_STATUS_SUCCESS;
}
