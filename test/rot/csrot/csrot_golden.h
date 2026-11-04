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
#include "cblas_compat.h"

// CPU golden for aclblasCsrot — signature identical to the API (c / s passed by
// pointer, aligned with cublasCsrot), returns aclblasStatus_t.
// Parameter validation is aligned with the NPU operator AND netlib csrot.f:
//   - handle == nullptr            -> ACLBLAS_STATUS_HANDLE_IS_NULLPTR
//   - n <= 0                       -> ACLBLAS_STATUS_SUCCESS (no-op, x/y untouched)
//   - x == nullptr || y == nullptr -> ACLBLAS_STATUS_INVALID_VALUE (n > 0)
//   - c == nullptr || s == nullptr -> ACLBLAS_STATUS_INVALID_VALUE
//   - incx == 0 / incy == 0        -> NOT rejected (netlib reuses element 0 N times,
//                                     serial accumulation)
//   - incx < 0 / incy < 0          -> NOT rejected (netlib walks from the tail end)
//   - c == 1 && s == 0             -> NOT short-circuited (netlib runs the rotation)
// Reference arithmetic: the netlib csrot.f rotation formula, evaluated per float
// lane with every product rounded exactly once (volatile materialization, same
// discipline as the NPU kernel):
//   x' = c*x + s*y ;  y' = c*y - s*x      (real / imag lanes independent — c, s real)
// This is the cblas_csrot / OpenBLAS semantic and matches the NPU kernel
// bit-for-bit, including the overflow (Inf/NaN) behaviour of the extreme cases.
// Why the system BLAS is not called directly on this environment:
//   1. The netlib cblas.h shipped here (via /etc/alternatives) does NOT declare
//      cblas_csrot, and libblas does not export the cblas_csrot symbol.
//   2. Calling the Fortran csrot_ entry directly is NOT equivalent: gfortran
//      compiles the COMPLEX arithmetic as (a*c - b*d, a*d + b*c) with b == 0, so
//      once one lane reaches ±Inf the spurious cross term 0*Inf = NaN poisons
//      the other lane (verified: x.re = -Inf forces x.im = NaN). That is a
//      compiler codegen artifact of this reference-BLAS build, not netlib
//      semantics — cblas_csrot / OpenBLAS evaluate the lanes independently —
//      so the reference formula is implemented explicitly here.
inline aclblasStatus_t aclblasCsrot_cpu(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const float* s)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n <= 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (c == nullptr || s == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // Stride semantics of netlib csrot.f: inc == 0 holds the index at element 0
    // (serial accumulation over the reused address); inc < 0 starts at the tail
    // and walks toward the front (logical i -> physical (n-1-i)*|inc|).
    const int64_t absIncX = (incx < 0) ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx);
    const int64_t absIncY = (incy < 0) ? -static_cast<int64_t>(incy) : static_cast<int64_t>(incy);
    for (int i = 0; i < n; ++i) {
        const int64_t xIdx = (incx == 0) ? 0 : ((incx < 0) ? (static_cast<int64_t>(n) - 1 - i) : i) * absIncX;
        const int64_t yIdx = (incy == 0) ? 0 : ((incy < 0) ? (static_cast<int64_t>(n) - 1 - i) : i) * absIncY;
        const float xr = x[xIdx].real;
        const float xi = x[xIdx].imag;
        const float yr = y[yIdx].real;
        const float yi = y[yIdx].imag;
        // Each product rounded once (volatile), then added/subtracted — the same
        // per-step rounding as the NPU kernel and the reference implementation.
        volatile float cxr = *c * xr;
        volatile float syr = *s * yr;
        volatile float cxi = *c * xi;
        volatile float syi = *s * yi;
        volatile float cyr = *c * yr;
        volatile float sxr = *s * xr;
        volatile float cyi = *c * yi;
        volatile float sxi = *s * xi;
        x[xIdx].real = cxr + syr;
        x[xIdx].imag = cxi + syi;
        y[yIdx].real = cyr - sxr;
        y[yIdx].imag = cyi - sxi;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
