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

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

// issue #775: golden 校验按 strsm_host.cpp 的 aclblasStrsm/ValidateStrsmParams 顺序分段；
// 拆分校验 helper 以满足圈复杂度门限（!824 检视意见）
// 本函数对齐入口 side/m/n 负值与 ValidateStrsmParams 的枚举段
static inline bool StrsmGoldenCheckEnums(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n)
{
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT)
        return false;
    if (m < 0 || n < 0)
        return false;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER)
        return false;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return false;
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT)
        return false;
    return true;
}

// 本函数对齐 ValidateStrsmParams 的 B/A/lda/ldb 段（quick-return 之后生效）
static inline bool StrsmGoldenCheckArgs(
    aclblasSideMode_t side, int m, int n, const float* alpha, const float* A, const float* B, int lda, int ldb)
{
    if (B == nullptr)
        return false;
    if (*alpha != 0.0f && A == nullptr)
        return false;
    int k = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (lda < std::max(1, k))
        return false;
    if (ldb < std::max(1, m))
        return false;
    return true;
}

inline aclblasStatus_t aclblasStrsm_cpu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb)
{
    // issue #775: 校验顺序对齐 strsm_host.cpp 的 aclblasStrsm/ValidateStrsmParams：
    // 句柄 → 枚举与维度 → alpha → m==0/n==0 quick-return → B/A → lda/ldb 下界
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!StrsmGoldenCheckEnums(side, uplo, trans, diag, m, n) || alpha == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (!StrsmGoldenCheckArgs(side, m, n, alpha, A, B, lda, ldb))
        return ACLBLAS_STATUS_INVALID_VALUE;

    CBLAS_SIDE cblasSide = (side == ACLBLAS_SIDE_LEFT) ? CblasLeft : CblasRight;
    cblas_strsm(
        CblasColMajor, cblasSide, ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag), m, n, *alpha, A, lda, B, ldb);
    return ACLBLAS_STATUS_SUCCESS;
}
