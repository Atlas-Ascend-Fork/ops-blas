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
#include <initializer_list>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

enum class SrotGolden { Arch35, Arch22 };

namespace srot_golden_detail {
inline bool InvalidArch22Strides(int n, int incx, int incy)
{
    for (int inc : {incx, incy}) {
        if (inc == std::numeric_limits<int>::min())
            return true;
        const int64_t step = inc < 0 ? -static_cast<int64_t>(inc) : inc;
        if (static_cast<int64_t>(n - 1) * step > std::numeric_limits<int32_t>::max())
            return true;
    }
    return false;
}

inline void Netlib310ZeroStride(int n, float* x, int incx, float* y, int incy, float c, float s)
{
    int64_t ix = incx < 0 ? (1 - static_cast<int64_t>(n)) * incx : 0;
    int64_t iy = incy < 0 ? (1 - static_cast<int64_t>(n)) * incy : 0;
    for (int i = 0; i < n; ++i) {
        const float oldX = x[ix];
        const float oldY = y[iy];
        // Match the task's Netlib 3.10 reference built with -ffp-contract=off.
        // Repeated zero-stride updates amplify FMA differences in other BLAS builds.
        volatile float cx = c * oldX;
        volatile float sy = s * oldY;
        volatile float cy = c * oldY;
        volatile float sx = s * oldX;
        const float nextX = cx + sy;
        const float nextY = cy - sx;
        y[iy] = nextY;
        x[ix] = nextX;
        ix += incx;
        iy += incy;
    }
}
} // namespace srot_golden_detail

// Arch35 retains its existing validation and delegates all valid inputs to cblas_srot.
// Arch22 keeps n < 0, INT_MIN and overflowing spans invalid, permits zero strides,
// and fixes only their serial reference arithmetic to the task's Netlib 3.10 build.
// Nonzero-stride inputs still use the linked BLAS implementation under either policy.
template <SrotGolden policy = SrotGolden::Arch35>
inline aclblasStatus_t aclblasSrot_cpu(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if constexpr (policy == SrotGolden::Arch22) {
        if (n < 0)
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (c == nullptr || s == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if constexpr (policy == SrotGolden::Arch22) {
        if (srot_golden_detail::InvalidArch22Strides(n, incx, incy))
            return ACLBLAS_STATUS_INVALID_VALUE;
        if (incx == 0 || incy == 0) {
            srot_golden_detail::Netlib310ZeroStride(n, x, incx, y, incy, *c, *s);
            return ACLBLAS_STATUS_SUCCESS;
        }
    }
    cblas_srot(n, x, incx, y, incy, *c, *s);
    return ACLBLAS_STATUS_SUCCESS;
}
