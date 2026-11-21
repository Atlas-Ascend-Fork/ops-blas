/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYR2K_GOLDEN_H
#define CSYR2K_GOLDEN_H

#include <algorithm>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

static constexpr int CSYR2K_CPU_VALID_OK = 0x7FFFFFFF;

inline int ValidateCsyr2kCpuParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta,
    const aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    int minLd = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < minLd || ldb < minLd || ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (alpha == nullptr || beta == nullptr || C == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (k > 0 && (A == nullptr || B == nullptr)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CSYR2K_CPU_VALID_OK;
}

inline void Csyr2kScaleTriangle(aclblasFillMode_t uplo, int n, const aclblasComplex& beta, aclblasComplex* C, int ldc)
{
    bool upper = uplo == ACLBLAS_UPPER;
    bool betaZero = beta.real == 0.0f && beta.imag == 0.0f;
    for (int col = 0; col < n; ++col) {
        int rowBegin = upper ? 0 : col;
        int rowEnd = upper ? col + 1 : n;
        for (int row = rowBegin; row < rowEnd; ++row) {
            size_t offset = static_cast<size_t>(col) * ldc + row;
            if (betaZero) {
                C[offset] = {0.0f, 0.0f};
            } else {
                aclblasComplex old = C[offset];
                C[offset] = {beta.real * old.real - beta.imag * old.imag, beta.real * old.imag + beta.imag * old.real};
            }
        }
    }
}

inline aclblasStatus_t aclblasCsyr2k_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C,
    int ldc)
{
    int status = ValidateCsyr2kCpuParams(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (status != CSYR2K_CPU_VALID_OK)
        return static_cast<aclblasStatus_t>(status);
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    bool alphaZero = alpha->real == 0.0f && alpha->imag == 0.0f;
    if (alphaZero || k == 0) {
        Csyr2kScaleTriangle(uplo, n, *beta, C, ldc);
        return ACLBLAS_STATUS_SUCCESS;
    }

    CBLAS_TRANSPOSE cblasTrans = (trans == ACLBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    cblas_csyr2k(
        CblasColMajor, ToCblasUplo(uplo), cblasTrans, n, k, static_cast<const void*>(alpha),
        static_cast<const void*>(A), lda, static_cast<const void*>(B), ldb, static_cast<const void*>(beta),
        static_cast<void*>(C), ldc);
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CSYR2K_GOLDEN_H
