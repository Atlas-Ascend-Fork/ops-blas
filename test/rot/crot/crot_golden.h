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

#include "acl/acl.h"
#include "cann_ops_blas.h"

// CPU golden for aclblasCrot. The signature matches the API and the formulas below are the
// ones stated in the task book for aclblasCrot 950 (section 2.1.1). Section 2.1.5 notes that
// reference BLAS has no crot with a complex s and that the cited Netlib/LAPACK reference is an
// archived LAPACK auxiliary routine, so this file is the golden definition used for acceptance.
//     x'[k] = c*x[k] + s*y[j]
//     y'[j] = c*y[j] - conj(s)*x[k]      (pre-rotation x[k])
// The rotation is expanded directly here with the same real/imag order as the device kernel,
// because neither reference BLAS nor cblas exposes a crot entry point.
// Parameter validation mirrors the NPU operator.
//   - handle == nullptr            -> ACLBLAS_STATUS_HANDLE_IS_NULLPTR
//   - n <= 0                       -> ACLBLAS_STATUS_SUCCESS (no-op, x/y untouched)
//   - x/y/c/s nullptr (n > 0)      -> ACLBLAS_STATUS_INVALID_VALUE
//   - zero / negative strides      -> legal inputs, evaluated in reference order
// Zero stride: the loop recomputes the address every iteration and reads the current buffer
// contents, so repeated iterations on element 0 accumulate serially, as required by the task
// book section 2.1.4 (reference BLAS rejects a zero increment instead).
// Complex expansion using float arithmetic only.
//   s*Y      : re = sr*yr - si*yi , im = sr*yi + si*yr
//   conj(s)*X: re = sr*xr + si*xi , im = sr*xi - si*xr
//   x' re = c*xr + (sr*yr - si*yi)
//   x' im = c*xi + (sr*yi + si*yr)
//   y' re = c*yr - (sr*xr + si*xi)
//   y' im = c*yi - (sr*xi - si*xr)
// Each product goes through a volatile temporary, exactly like CrotRotate in
// crot_kernel.cpp. That forbids FMA contraction on both sides, so kernel and
// golden round identically even when huge c/s make cancellation amplify a
// single fused-rounding difference past the FP32 tolerance.
inline aclblasStatus_t aclblasCrot_cpu(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const aclblasComplex* s)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n <= 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (c == nullptr || s == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    const float cc = *c;
    const float sr = s->real;
    const float si = s->imag;

    // |inc| computed in int64 so INT32_MIN does not overflow on negation.
    const int64_t absIncX = (incx < 0) ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx);
    const int64_t absIncY = (incy < 0) ? -static_cast<int64_t>(incy) : static_cast<int64_t>(incy);

    for (int i = 0; i < n; i++) {
        const int64_t jx = (incx >= 0) ? static_cast<int64_t>(i) * incx : static_cast<int64_t>(n - 1 - i) * absIncX;
        const int64_t jy = (incy >= 0) ? static_cast<int64_t>(i) * incy : static_cast<int64_t>(n - 1 - i) * absIncY;
        const float xr = x[jx].real;
        const float xi = x[jx].imag;
        const float yr = y[jy].real;
        const float yi = y[jy].imag;

        volatile float srYr = sr * yr;
        volatile float siYi = si * yi;
        volatile float srYi = sr * yi;
        volatile float siYr = si * yr;
        volatile float srXr = sr * xr;
        volatile float siXi = si * xi;
        volatile float srXi = sr * xi;
        volatile float siXr = si * xr;
        volatile float cXr = cc * xr;
        volatile float cXi = cc * xi;
        volatile float cYr = cc * yr;
        volatile float cYi = cc * yi;
        volatile float xRePart = srYr - siYi;
        volatile float xImPart = srYi + siYr;
        volatile float yRePart = srXr + siXi;
        volatile float yImPart = srXi - siXr;
        const float newXr = cXr + xRePart;
        const float newXi = cXi + xImPart;
        const float newYr = cYr - yRePart;
        const float newYi = cYi - yImPart;

        x[jx].real = newXr;
        x[jx].imag = newXi;
        y[jy].real = newYr;
        y[jy].imag = newYi;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
