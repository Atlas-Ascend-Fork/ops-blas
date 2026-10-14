/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline CBLAS_SIDE CtrsmToCblasSide(aclblasSideMode_t side)
{
    return side == ACLBLAS_SIDE_LEFT ? CblasLeft : CblasRight;
}

inline bool CtrsmAlphaIsZero(const aclblasComplex& alpha) { return alpha.real == 0.0f && alpha.imag == 0.0f; }

inline bool CtrsmValidEnums(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    return (side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT) &&
           (uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER) &&
           (trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C) &&
           (diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT);
}

inline aclblasStatus_t aclblasCtrsmCpu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* a, int lda,
    aclblasComplex* b, int ldb)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!CtrsmValidEnums(side, uplo, trans, diag)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const int order = side == ACLBLAS_SIDE_LEFT ? m : n;
    if (lda < std::max(1, order) || ldb < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (b == nullptr || (!CtrsmAlphaIsZero(*alpha) && a == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (CtrsmAlphaIsZero(*alpha)) {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                b[static_cast<size_t>(col) * ldb + row] = aclblasComplex{0.0f, 0.0f};
            }
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_ctrsm(
        CblasColMajor, CtrsmToCblasSide(side), ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag), m, n, alpha, a,
        lda, b, ldb);
    return ACLBLAS_STATUS_SUCCESS;
}
