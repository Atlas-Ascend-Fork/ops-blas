/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CGEMM_BATCHED_GOLDEN_H
#define CGEMM_BATCHED_GOLDEN_H

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "cgemm_batched_param.h"

inline void ApplyComplexAlphaZeroGolden(aclblasComplex* c, int m, int n, int ldc, const aclblasComplex& beta)
{
    if (beta.real == 1.0f && beta.imag == 0.0f)
        return;
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            int64_t idx = static_cast<int64_t>(col) * ldc + row;
            if (beta.real == 0.0f && beta.imag == 0.0f) {
                c[idx] = {0.0f, 0.0f};
                continue;
            }
            float cr = c[idx].real;
            float ci = c[idx].imag;
            // Match the specified sequence of FP32 operations, without FMA contraction.
            volatile float rr = beta.real * cr;
            volatile float ii = beta.imag * ci;
            volatile float ri = beta.real * ci;
            volatile float ir = beta.imag * cr;
            c[idx].real = rr - ii;
            c[idx].imag = ri + ir;
        }
    }
}

inline CGEMM_BATCHED_SIGNATURE(aclblasCgemmBatched_cpu)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || batchCount < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (batchCount == 0 || m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    aclblasComplex alphaVal = *alpha;
    aclblasComplex betaVal = *beta;

    CBLAS_TRANSPOSE cTransA = ToCblasOp(transA);
    CBLAS_TRANSPOSE cTransB = ToCblasOp(transB);

    for (int batch = 0; batch < batchCount; batch++) {
        const aclblasComplex* aData = Aarray ? Aarray[batch] : nullptr;
        const aclblasComplex* bData = Barray ? Barray[batch] : nullptr;
        aclblasComplex* cData = Carray ? Carray[batch] : nullptr;
        if (cData == nullptr)
            continue;

        bool alphaZero = (alphaVal.real == 0.0f && alphaVal.imag == 0.0f);
        if (alphaZero || k == 0) {
            ApplyComplexAlphaZeroGolden(cData, m, n, ldc, betaVal);
            continue;
        }

        cblas_cgemm(
            CblasColMajor, cTransA, cTransB, m, n, k, reinterpret_cast<const void*>(&alphaVal),
            reinterpret_cast<const void*>(aData), lda, reinterpret_cast<const void*>(bData), ldb,
            reinterpret_cast<const void*>(&betaVal), reinterpret_cast<void*>(cData), ldc);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CGEMM_BATCHED_GOLDEN_H
