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

#include <cmath>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "fill.h"

// CPU reference for aclblasCdotu: result = Σ x[k]*y[j] (unconjugated complex dot).
// Accumulated in double precision so the golden is near-exact (independent of
// summation order). For large n the float32-sequential reference error would
// otherwise dominate the scalar diff vs the 32*ULP max_abs_error_limit; an
// accurate float32 tree-reduce kernel should sit well inside the task's bounds.
inline void aclblasCdotu_cpu(
    int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* result)
{
    double real = 0.0;
    double imag = 0.0;
    for (int i = 0; i < n; i++) {
        int xi = (incx > 0) ? (i * incx) : ((n - 1 - i) * (-incx));
        int yi = (incy > 0) ? (i * incy) : ((n - 1 - i) * (-incy));
        const aclblasComplex& xc = x[xi];
        const aclblasComplex& yc = y[yi];
        real += static_cast<double>(xc.real) * yc.real - static_cast<double>(xc.imag) * yc.imag;
        imag += static_cast<double>(xc.real) * yc.imag + static_cast<double>(xc.imag) * yc.real;
    }
    result->real = static_cast<float>(real);
    result->imag = static_cast<float>(imag);
}

// Float32-sequential reference (cblas / Netlib cdotu accumulation semantics).
// Used for extreme-value cases where float32 products overflow to Inf/NaN and the
// double-accumulated golden would stay finite — both references must agree on the
// overflow behavior per task §3.2 "Inf 一致性" handling.
inline void aclblasCdotu_cpu_f32(
    int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* result)
{
    float real = 0.0f;
    float imag = 0.0f;
    for (int i = 0; i < n; i++) {
        int xi = (incx > 0) ? (i * incx) : ((n - 1 - i) * (-incx));
        int yi = (incy > 0) ? (i * incy) : ((n - 1 - i) * (-incy));
        const aclblasComplex& xc = x[xi];
        const aclblasComplex& yc = y[yi];
        real += xc.real * yc.real - xc.imag * yc.imag;
        imag += xc.real * yc.imag + xc.imag * yc.real;
    }
    result->real = real;
    result->imag = imag;
}

// Complex strided vector builder matching Netlib cdotu index semantics:
// element i (0-based) of a vector with stride inc maps to complex slot
//   inc > 0 → i*inc ;  inc < 0 → (count-1-i)*|inc|.
// Real/imag parts use independent seeds (real: seed, imag: seed+1000) so the
// ranges match while avoiding degenerate correlation.
inline std::vector<aclblasComplex> makeBlasComplexStrided(
    int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    const int absInc = std::abs(inc);
    const size_t size = (count > 0) ? static_cast<size_t>(count - 1) * static_cast<size_t>(absInc) + 1 : 1;
    std::vector<aclblasComplex> data(size, aclblasComplex{0.0f, 0.0f});
    if (count <= 0) {
        return data;
    }

    std::vector<float> realPart = makeBlasArray(count, fill, seed);
    std::vector<float> imagPart = makeBlasArray(count, fill, seed + 1000U);
    for (int i = 0; i < count; i++) {
        int idx = (inc > 0) ? (i * inc) : ((count - 1 - i) * absInc);
        data[idx] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}