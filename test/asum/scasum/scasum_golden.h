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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"

// CPU reference for aclblasScasum: result = Σ (|Re(x_i)| + |Im(x_i)|).
// Accumulated in double precision (near-exact golden) so a correct float32
// reduction kernel passes the task's §3.2 rtol=2^-10 / 32*ULP scalar bounds.
inline aclblasStatus_t aclblasScasum_cpu(
    aclblasHandle_t handle, int64_t n, const aclblasComplex* x, int64_t incx, float* result)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (result == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n <= 0 || incx <= 0) {
        *result = 0.0f;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // Accumulate in double precision so the golden is near-exact (independent of
    // summation order). The reduction error of a float32 sequential reference would
    // otherwise dominate and make the 32*ULP max_abs_error_limit unachievable for
    // large n. An accurate float32 tree/vector-reduce kernel should then sit well
    // inside the task's rtol=2^-10 / max_abs_error_limit=32*ULP bounds.
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        aclblasComplex c = x[i * incx];
        sum += std::fabs(static_cast<double>(c.real)) + std::fabs(static_cast<double>(c.imag));
    }
    *result = static_cast<float>(sum);
    return ACLBLAS_STATUS_SUCCESS;
}