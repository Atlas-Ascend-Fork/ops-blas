/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// aclblasCgbmv CPU golden：参数校验镜像 NPU 侧语义（no-op / alpha=0 / beta=0 /
// 参与计算判空），随后直调参考 cblas_cgbmv（列主序，与任务书 golden 口径一致）。
// aclblasComplex 与 CBLAS 复数布局二进制兼容（cherk golden 同款 reinterpret）。

#pragma once

#include <cblas.h>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

// Sentinel：校验通过
static const int CGBMV_CPU_VALID_OK = 0x7FFFFFFF;

static int ValidateCgbmvCpuParams(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, const aclblasComplex* alpha,
    const aclblasComplex* beta, const aclblasComplex* A, const aclblasComplex* x, const aclblasComplex* y)
{
    const bool opOk = trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
    if (alpha == nullptr || beta == nullptr || !opOk || m < 0 || n < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (m == 0 || n == 0) {
        return CGBMV_CPU_VALID_OK;
    }
    const bool shapeOk = kl >= 0 && kl <= m - 1 && ku >= 0 && ku <= n - 1 && lda >= kl + ku + 1 && incx != 0 &&
                         incy != 0 && y != nullptr;
    if (!shapeOk) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    const bool alphaZero = (alpha->real == 0.0F && alpha->imag == 0.0F);
    if (!alphaZero && (A == nullptr || x == nullptr)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CGBMV_CPU_VALID_OK;
}

inline aclblasStatus_t aclblasCgbmv_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy)
{
    (void)handle;
    const int st = ValidateCgbmvCpuParams(trans, m, n, kl, ku, lda, incx, incy, alpha, beta, A, x, y);
    if (st != CGBMV_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS; // y 保持原值
    }
    // alpha==0 且 beta==1：netlib quick return，y 逐位保持
    if (alpha->real == 0.0F && alpha->imag == 0.0F && beta->real == 1.0F && beta->imag == 0.0F) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // cblas 语义与 netlib gbmv.f 一致：alpha==0 时不读 A/x，beta==0 时不读 y
    cblas_cgbmv(
        CblasColMajor, ToCblasOp(trans), m, n, kl, ku, alpha, reinterpret_cast<const void*>(A), lda,
        reinterpret_cast<const void*>(x), incx, reinterpret_cast<const void*>(beta), reinterpret_cast<void*>(y), incy);
    return ACLBLAS_STATUS_SUCCESS;
}
