/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

static const int SSYRK_CPU_VALID_OK = 0x7FFFFFFF;

static bool IsValidUplo(aclblasFillMode_t uplo) { return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER; }

static bool IsValidTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

static int ValidateSsyrkCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldc,
    const float* alpha, const float* beta, const float* A, const float* C)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (!IsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (!IsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (n == 0) {
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
    }
    bool isTrans = (trans != ACLBLAS_OP_N);
    int minLda = isTrans ? std::max(1, k) : std::max(1, n);
    if (lda < minLda) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (A == nullptr && n > 0 && k > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (C == nullptr && n > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return SSYRK_CPU_VALID_OK;
}

// ssyrk only updates the uplo triangle (diagonal included); the non-uplo
// triangle keeps the caller's C old values, matching cblas_ssyrk semantics.
static void ClearMatrix(float* C, int n, int ldc, aclblasFillMode_t uplo)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            const bool isUplo = (uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                C[static_cast<size_t>(j) * ldc + i] = 0.0f;
            }
        }
    }
}

static void ScaleMatrix(float* C, int n, int ldc, float betaVal, aclblasFillMode_t uplo)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            const bool isUplo = (uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                size_t idx = static_cast<size_t>(j) * ldc + i;
                C[idx] = betaVal * C[idx];
            }
        }
    }
}

inline aclblasStatus_t aclblasSsyrk_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const float* A, int lda, const float* beta, float* C, int ldc)
{
    int st = ValidateSsyrkCpuParams(handle, uplo, trans, n, k, lda, ldc, alpha, beta, A, C);
    if (st != SSYRK_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    float alphaVal = *alpha;
    float betaVal = *beta;

    if (alphaVal == 0.0f && betaVal == 0.0f) {
        ClearMatrix(C, n, ldc, uplo);
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (k == 0) {
        if (betaVal == 0.0f) {
            ClearMatrix(C, n, ldc, uplo);
        } else {
            ScaleMatrix(C, n, ldc, betaVal, uplo);
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_ssyrk(CblasColMajor, ToCblasUplo(uplo), ToCblasOp(trans), n, k, alphaVal, A, lda, betaVal, C, ldc);

    return ACLBLAS_STATUS_SUCCESS;
}
