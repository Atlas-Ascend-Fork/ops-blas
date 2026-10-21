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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

// Index helpers shared by both architecture tests.
inline size_t CtbmvVectorIndex(int logicalIndex, int n, int incx)
{
    const size_t absIncx = static_cast<size_t>(incx >= 0 ? incx : -static_cast<int64_t>(incx));
    return incx >= 0 ? static_cast<size_t>(logicalIndex) * absIncx :
                       static_cast<size_t>(n - 1 - logicalIndex) * absIncx;
}

inline size_t CtbmvBandIndex(aclblasFillMode_t uplo, int k, int lda, int row, int col)
{
    const int bandRow = uplo == ACLBLAS_UPPER ? k + row - col : row - col;
    return static_cast<size_t>(bandRow) + static_cast<size_t>(col) * lda;
}

// Golden for buffers that already look the way cblas expects them.
inline void CtbmvGolden(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const std::vector<aclblasComplex>& a, int lda, std::vector<aclblasComplex>& x, int incx)
{
    cblas_ctbmv(
        CblasColMajor, ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag), n, k, static_cast<const void*>(a.data()),
        lda, static_cast<void*>(x.data()), incx);
}

// aclblasComplex = {float real; float imag} is binary-compatible with OpenBLAS/cblas
// complex layout, so a plain void* reinterpret is safe.
// Validating variant: mirrors the aclblas entry-point checks, so callers can assert
// on the status before comparing numbers.
inline aclblasStatus_t aclblasCtbmv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const aclblasComplex* a, int lda, aclblasComplex* x, int incx)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < k + 1) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (a == nullptr || x == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_ctbmv(CblasColMajor, ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag), n, k,
                static_cast<const void*>(a), lda, static_cast<void*>(x), incx);
    return ACLBLAS_STATUS_SUCCESS;
}
