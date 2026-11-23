/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sgetrf_batched_host.cpp
 * \brief Host-side API for batched single-precision LU factorization: getrfBatched.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "sgetrf_batched_tiling_data.h"
#include "sgetrf_batched_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static aclblasStatus_t ValidateGetrfBatchedParams(
    int n, float* const Aarray[], int lda, int* pivotArray, int* infoArray, int batchSize)
{
    if (n < 0) {
        OP_LOGE("aclblasSgetrfBatched", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        OP_LOGE("aclblasSgetrfBatched", "batchSize must be >= 0, got %d", batchSize);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // LAPACK convention: validate lda before the n==0/batchSize==0 no-op check.
    // For n=0, max(1, n)=1, so lda must be >= 1.
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasSgetrfBatched", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // No-op: when n==0 or batchSize==0, no data pointers are accessed.
    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (Aarray == nullptr) {
        OP_LOGE("aclblasSgetrfBatched", "Aarray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // infoArray is required only when PivotArray is provided (pivot mode).
    // PivotArray == NULL && infoArray == NULL is legal (non-pivot mode without info output).
    if (pivotArray != nullptr && infoArray == nullptr) {
        OP_LOGE("aclblasSgetrfBatched", "infoArray must not be nullptr when PivotArray is provided");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSgetrfBatched(
    aclblasHandle_t handle, int n, float* const Aarray[], int lda, int* PivotArray, int* infoArray, int batchSize)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSgetrfBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    auto* h = handle;

    aclblasStatus_t validRet = ValidateGetrfBatchedParams(n, Aarray, lda, PivotArray, infoArray, batchSize);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Get core count and compute batch distribution
    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasSgetrfBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t uBatchSize = static_cast<uint32_t>(batchSize);
    uint32_t batchPerCore = (uBatchSize - 1) / coreNum + 1;
    uint32_t usedCoreNum = (uBatchSize - 1) / batchPerCore + 1;
    if (usedCoreNum == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Build TilingData
    SgetrfBatchedTilingData tiling = {};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.usedCoreNum = usedCoreNum;
    tiling.batchPerCore = batchPerCore;
    tiling.batchTail = uBatchSize - (usedCoreNum - 1) * batchPerCore;
    tiling.usePivot = (PivotArray != nullptr) ? 1u : 0u;

    OP_LOGD(
        "aclblasSgetrfBatched", "tiling: n=%u lda=%u usedCoreNum=%u batchPerCore=%u batchTail=%u usePivot=%u", tiling.n,
        tiling.lda, tiling.usedCoreNum, tiling.batchPerCore, tiling.batchTail, tiling.usePivot);
    OP_LOGI("aclblasSgetrfBatched", "launching kernel: blocks=%u, cores=%u", usedCoreNum, coreNum);

    // issue #1180: a failed launch must not be reported as SUCCESS.
    const aclError launch = sgetrf_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float**>(Aarray)), reinterpret_cast<uint8_t*>(PivotArray),
        reinterpret_cast<uint8_t*>(infoArray), tiling, usedCoreNum, h->stream);
    CHECK_RET(launch == ACL_SUCCESS, OP_LOGE("aclblasSgetrfBatched", "sgetrf_batched_kernel_do failed: %d", launch);
              return ACLBLAS_STATUS_EXECUTION_FAILED);

    return ACLBLAS_STATUS_SUCCESS;
}
