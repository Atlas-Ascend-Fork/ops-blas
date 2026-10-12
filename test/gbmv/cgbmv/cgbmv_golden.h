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

// Argument checks of aclblasCgbmv that only involve scalars (enforced even for the m == 0 / n == 0 no-op).
inline aclblasStatus_t CgbmvCpuCheckScalars(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, const aclblasComplex* alpha,
    const aclblasComplex* beta)
{
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || kl < 0 || ku < 0 || static_cast<int64_t>(lda) < static_cast<int64_t>(kl) + ku + 1 ||
        incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return (alpha == nullptr || beta == nullptr) ? ACLBLAS_STATUS_INVALID_VALUE : ACLBLAS_STATUS_SUCCESS;
}

// CPU golden: Netlib reference BLAS cblas_cgbmv. Parameter checks mirror aclblasCgbmv so that
// the reference library is never called with arguments it would reject or dereference as null.
inline aclblasStatus_t aclblasCgbmv_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    aclblasStatus_t st = CgbmvCpuCheckScalars(trans, m, n, kl, ku, lda, incx, incy, alpha, beta);
    if (st != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return st;
    }
    if (kl > m - 1 || ku > n - 1 || a == nullptr || x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    cblas_cgbmv(CblasColMajor, ToCblasOp(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    return ACLBLAS_STATUS_SUCCESS;
}
