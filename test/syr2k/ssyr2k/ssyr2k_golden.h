/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSYR2K_GOLDEN_H
#define SSYR2K_GOLDEN_H

#include <algorithm>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

static const int SSYR2K_CPU_VALID_OK = 0x7FFFFFFF;

static bool Ssyr2kIsValidUplo(aclblasFillMode_t uplo)
{
    return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER;
}

static bool Ssyr2kIsValidTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

static int ValidateSsyr2kCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, int lda, int ldb, int ldc,
    const float* alpha, const float* A, const float* B, const float* beta, const float* C)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (!Ssyr2kIsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (!Ssyr2kIsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (n == 0) {
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
    }
    int minLd = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < minLd || ldb < minLd) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (C == nullptr && n > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (k > 0 && (A == nullptr || B == nullptr)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return SSYR2K_CPU_VALID_OK;
}

inline void Ssyr2kCpuKZero(aclblasFillMode_t uplo, int n, float beta, float* C, int ldc)
{
    bool isUpper = (uplo == ACLBLAS_UPPER);
    for (int j = 0; j < n; j++) {
        for (int i = isUpper ? 0 : j; i <= (isUpper ? j : n - 1); i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc);
            C[idx] = (beta == 0.0f) ? 0.0f : beta * C[idx];
        }
    }
}

inline aclblasStatus_t aclblasSsyr2k_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc)
{
    int st = ValidateSsyr2kCpuParams(handle, uplo, trans, n, k, lda, ldb, ldc,
        alpha, A, B, beta, C);
    if (st != SSYR2K_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    float betaVal = *beta;

    if (k == 0) {
        Ssyr2kCpuKZero(uplo, n, betaVal, C, ldc);
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_ssyr2k(CblasColMajor,
                  ToCblasUplo(uplo), ToCblasOp(trans),
                  n, k,
                  *alpha, A, lda, B, ldb,
                  betaVal, C, ldc);
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SSYR2K_GOLDEN_H
