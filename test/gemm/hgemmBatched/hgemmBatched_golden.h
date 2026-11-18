/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HGEMM_BATCHED_GOLDEN_H
#define HGEMM_BATCHED_GOLDEN_H

#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "dtype_cast.h"
#include "hgemmBatched_param.h"

// Reference implementation for a single batch.
// Operates on FLOAT32 host buffers that were dequantized from FLOAT16 inputs.
// The mathematical model is C_i = alpha * op(A_i) * op(B_i) + beta * C_i,
// evaluated with FLOAT32 accumulation through Netlib cblas_sgemm per batch.
// The caller is responsible for rounding the FLOAT32 result back to FLOAT16
// (IEEE 754 round-to-nearest-even) before comparison; that rounding is applied
// at the granularity that mirrors the NPU output storage (FLOAT16).
inline void HgemmBatchedCpuReference(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc)
{
    if (m <= 0 || n <= 0) {
        return;
    }
    if (k == 0 || alpha == 0.0f) {
        // Degenerate path: C = beta * C.
        // beta == 1 leaves C untouched; beta == 0 zero-fills (C need not be a
        // valid input, so NaN/Inf old contents must not propagate); otherwise
        // scale in place.
        if (beta == 1.0f) {
            return;
        }
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < m; row++) {
                float& v = C[static_cast<int64_t>(col) * ldc + row];
                v = (beta == 0.0f) ? 0.0f : v * beta;
            }
        }
        return;
    }

    cblas_sgemm(
        CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

#endif // HGEMM_BATCHED_GOLDEN_H
