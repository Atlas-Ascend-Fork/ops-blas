/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: CPU golden（与芯片无关）。文件落地为 test/<family>/{{op}}/{{op}}_golden.h
// - 签名与 BLAS API 完全一致，返回 aclblasStatus_t
// - 保留参数校验（与 NPU 算子一致；CBLAS 对空指针会崩溃，必须校验）
// - 校验通过后调用参考库函数；用 cblas_compat.h 的 ToCblasOp/ToCblasUplo/ToCblasDiag 转换枚举
// - 使用 CblasColMajor（列主序），与 BLAS 标准一致

#pragma once

#include <algorithm>
#include <climits>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

// ---- 变体 A：CBLAS 算子（Level-1/2/3），以 gemv 为例 ----
inline aclblasStatus_t aclblas
{
    {
        Op
    }
}
_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || lda < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    cblas_sgemv(CblasColMajor, ToCblasOp(trans), m, n, *alpha, a, lda, x, incx, *beta, y, incy);
    return ACLBLAS_STATUS_SUCCESS;
}

// ---- 变体 B：LAPACK 算子（geqrf/getrf/tpttr/trttp），以 tpttr 为例 ----
// Fortran 函数声明已在 cblas_compat.h 中，直接调用即可。
//
// inline aclblasStatus_t aclblas{{Op}}_cpu(
//     aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* ap, float* a, int lda)
// {
//     if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
//     if (n < 0 || lda < std::max(1, n)) return ACLBLAS_STATUS_INVALID_VALUE;
//     if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) return ACLBLAS_STATUS_INVALID_VALUE;
//     if (ap == nullptr || a == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
//     if (n == 0) return ACLBLAS_STATUS_SUCCESS;
//
//     char uploChar = (uplo == ACLBLAS_UPPER) ? 'U' : 'L';
//     int info = 0;
//     stpttr_(&uploChar, &n, ap, a, &lda, &info, 1);
//     return ACLBLAS_STATUS_SUCCESS;
// }
