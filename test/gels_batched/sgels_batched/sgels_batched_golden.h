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
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

extern "C" {
void sgels_(const char* trans, const int* m, const int* n, const int* nrhs, float* a, const int* lda, float* b,
            const int* ldb, float* work, const int* lwork, int* info);
}

static inline aclblasStatus_t validateGelsCpuParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda,
    float* const Carray[], int ldc, int* devInfo, int batchSize)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0 || nrhs < 0 || batchSize < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max({1, m, n}))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (batchSize > 0 && (Aarray == nullptr || Carray == nullptr))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (devInfo == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSgelsBatched_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda,
    float* const Carray[], int ldc, int* devInfo, int batchSize)
{
    aclblasStatus_t validRet =
        validateGelsCpuParams(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, devInfo, batchSize);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return validRet;

    if (m == 0 || n == 0 || nrhs == 0 || batchSize == 0) {
        *devInfo = 0;
        return ACLBLAS_STATUS_SUCCESS;
    }

    int globalInfo = 0;
    const char transChar = (trans == ACLBLAS_OP_N) ? 'N' : 'T';
    const int solRows = (trans == ACLBLAS_OP_N) ? n : m;
    const int rhsRows = (trans == ACLBLAS_OP_N) ? m : n;

    for (int b = 0; b < batchSize; b++) {
        std::vector<float> Awork(static_cast<size_t>(lda) * n);
        std::vector<float> Cwork(static_cast<size_t>(ldc) * nrhs, 0.0f);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                Awork[i + j * lda] = Aarray[b][i + j * lda];

        for (int j = 0; j < nrhs; j++)
            for (int i = 0; i < rhsRows; i++)
                Cwork[i + j * ldc] = Carray[b][i + j * ldc];

        int info = 0;
        int lwork = -1;
        float workQuery;
        sgels_(&transChar, &m, &n, &nrhs, Awork.data(), &lda, Cwork.data(), &ldc, &workQuery, &lwork, &info);

        if (info == 0) {
            lwork = static_cast<int>(workQuery);
            std::vector<float> work(lwork);
            sgels_(&transChar, &m, &n, &nrhs, Awork.data(), &lda, Cwork.data(), &ldc, work.data(), &lwork, &info);
        }

        for (int j = 0; j < nrhs; j++)
            for (int i = 0; i < solRows; i++)
                Carray[b][i + j * ldc] = Cwork[i + j * ldc];

        if (info != 0 && globalInfo == 0)
            globalInfo = info;
    }

    *devInfo = globalInfo;
    return ACLBLAS_STATUS_SUCCESS;
}

