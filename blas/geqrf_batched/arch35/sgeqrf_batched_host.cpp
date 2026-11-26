/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "acl/acl.h"
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "sgeqrf_batched_tiling_data.h"

aclError sgeqrf_batched_kernel_do(
    uint8_t* aarray, uint8_t* tauArray, const GeqrfBatchedTilingData& tiling, uint32_t numBlocks, void* stream);

static const char* const TAG = "aclblasSgeqrfBatched";

static aclblasStatus_t ValidateGeqrfBatchedParams(
    int m, int n, float* const Aarray[], int lda, float* const TauArray[], int batchSize, int* info)
{
    if (m < 0) {
        OP_LOGE(TAG, "m must be >= 0, got %d", m);
        *info = ACLBLAS_LAPACK_INFO_ARG_1;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE(TAG, "n must be >= 0, got %d", n);
        *info = ACLBLAS_LAPACK_INFO_ARG_2;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) {
        OP_LOGE(TAG, "lda must be >= max(1, m), got lda=%d, m=%d", lda, m);
        *info = ACLBLAS_LAPACK_INFO_ARG_4;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        OP_LOGE(TAG, "batchSize must be >= 0, got %d", batchSize);
        *info = ACLBLAS_LAPACK_INFO_ARG_6;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize > 0 && Aarray == nullptr) {
        OP_LOGE(TAG, "Aarray must not be nullptr when batchSize > 0");
        *info = ACLBLAS_LAPACK_INFO_ARG_3;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize > 0 && m > 0 && n > 0 && TauArray == nullptr) {
        OP_LOGE(TAG, "TauArray must not be nullptr when batchSize > 0 and min(m,n) > 0");
        *info = ACLBLAS_LAPACK_INFO_ARG_5;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static GeqrfBatchedTilingData CalGeqrfBatchedTilingData(
    uint32_t m, uint32_t n, int32_t lda, uint32_t batchSize, uint32_t coreNum)
{
    GeqrfBatchedTilingData td = {};
    td.m = m;
    td.n = n;
    td.lda = lda;
    td.batchSize = batchSize;
    td.coreNum = coreNum;

    if (coreNum == 0) {
        return td;
    }
    uint32_t batchPerCore = (batchSize + coreNum - 1) / coreNum;
    uint32_t usedCoreNum = (batchSize + batchPerCore - 1) / batchPerCore;
    uint32_t batchTail = batchSize - (usedCoreNum - 1) * batchPerCore;

    td.batchPerCore = batchPerCore;
    td.usedCoreNum = usedCoreNum;
    td.batchTail = batchTail;

    return td;
}

static aclblasStatus_t LaunchGeqrfBatchedKernel(
    float* const Aarray[], float* const TauArray[], const GeqrfBatchedTilingData& tiling, aclrtStream stream)
{
    const aclError launchErr = sgeqrf_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float**>(Aarray)),
        reinterpret_cast<uint8_t*>(const_cast<float**>(TauArray)), tiling, tiling.usedCoreNum, stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasSgeqrf_batched", "sgeqrf_batched_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSgeqrfBatched(
    aclblasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info,
    int batchSize)
{
    if (handle == nullptr) {
        OP_LOGE(TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (info == nullptr) {
        OP_LOGE(TAG, "info must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasStatus_t validRet = ValidateGeqrfBatchedParams(m, n, Aarray, lda, TauArray, batchSize, info);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }
    if (m == 0 || n == 0 || batchSize == 0) {
        *info = ACLBLAS_LAPACK_INFO_OK;
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE(TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    auto* h = handle;
    GeqrfBatchedTilingData tiling = CalGeqrfBatchedTilingData(
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<int32_t>(lda), static_cast<uint32_t>(batchSize),
        coreNum);

    OP_LOGD(
        TAG, "tiling: m=%u n=%u lda=%d batchSize=%u coreNum=%u usedCoreNum=%u batchPerCore=%u batchTail=%u", tiling.m,
        tiling.n, tiling.lda, tiling.batchSize, tiling.coreNum, tiling.usedCoreNum, tiling.batchPerCore,
        tiling.batchTail);
    OP_LOGI(TAG, "launching kernel: blocks=%u", tiling.usedCoreNum);

    aclblasStatus_t launchRet = LaunchGeqrfBatchedKernel(Aarray, TauArray, tiling, h->stream);
    if (launchRet != ACLBLAS_STATUS_SUCCESS) {
        return launchRet;
    }

    OP_LOGI(TAG, "completed: m=%d, n=%d, batchSize=%d", m, n, batchSize);
    *info = ACLBLAS_LAPACK_INFO_OK;
    return ACLBLAS_STATUS_SUCCESS;
}
