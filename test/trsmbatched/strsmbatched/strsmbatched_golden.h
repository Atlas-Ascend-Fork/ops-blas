/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRSMBATCHED_GOLDEN_H
#define STRSMBATCHED_GOLDEN_H

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline CBLAS_SIDE ToCblasSide(aclblasSideMode_t side)
{
    return (side == ACLBLAS_SIDE_LEFT) ? CblasLeft : CblasRight;
}

inline aclblasStatus_t ValidateStrsmbatchedGoldenEnums(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) return ACLBLAS_STATUS_INVALID_VALUE;
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ValidateStrsmbatchedGoldenParams(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha, const float* const A[],
    const float* const B[], int lda, int ldb, int batchCount)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    aclblasStatus_t st = ValidateStrsmbatchedGoldenEnums(side, uplo, trans, diag);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    if (m < 0 || n < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    // Mirror the NPU contract (cuBLAS batched-family convention): negative
    // batchCount is invalid; zero batches is a legal no-op. Like the NPU
    // entry, every scalar check (enums, m/n, alpha, batchCount, lda/ldb)
    // runs unconditionally before the no-op gate, and the gate sits right
    // before the pointer-array checks below, so an invalid ld combined with
    // batchCount == 0 still reports INVALID_VALUE while null A/B arrays are
    // acceptable when there is nothing to compute.
    if (batchCount < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (lda < std::max(1, aDim)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldb < std::max(1, m)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (batchCount == 0) return ACLBLAS_STATUS_SUCCESS;
    if (B == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (*alpha != 0.0f && A == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline void StrsmbatchedGoldenZeroB(float* const B[], int b, int m, int n, int ldb)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            B[b][static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldb)] = 0.0f;
        }
    }
}

inline aclblasStatus_t aclblasStrsmBatched_cpu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* const A[], int lda,
    float* const B[], int ldb,
    int batchCount)
{
    aclblasStatus_t st = ValidateStrsmbatchedGoldenParams(handle, side, uplo, trans, diag,
        m, n, alpha, A, B, lda, ldb, batchCount);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    for (int b = 0; b < batchCount; b++) {
        if (*alpha == 0.0f) {
            StrsmbatchedGoldenZeroB(B, b, m, n, ldb);
        } else {
            cblas_strsm(CblasColMajor, ToCblasSide(side), ToCblasUplo(uplo),
                        ToCblasOp(trans), ToCblasDiag(diag),
                        m, n, *alpha, A[b], lda, B[b], ldb);
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // STRSMBATCHED_GOLDEN_H
