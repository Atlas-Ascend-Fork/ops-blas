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
 * \file cgetrf_batched_host.cpp
 * \brief Host-side API for batched single-precision complex LU factorization.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "cgetrf_batched_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

aclError cgetrf_batched_kernel_do(
    uint8_t* aarray, uint8_t* pivotArray, uint8_t* infoArray, const CgetrfBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream);

static aclblasStatus_t ValidateGetrfBatchedParams(
    int n, aclblasComplex* const Aarray[], int lda, int* pivotArray, int* infoArray, int batchSize)
{
    if (n < 0) {
        OP_LOGE("aclblasCgetrfBatched", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        OP_LOGE("aclblasCgetrfBatched", "batchSize must be >= 0, got %d", batchSize);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // LAPACK convention: validate lda before the n==0/batchSize==0 no-op check.
    // For n=0, max(1, n)=1, so lda must be >= 1.
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasCgetrfBatched", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // No-op: when n==0 or batchSize==0, no data pointers are accessed.
    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (Aarray == nullptr) {
        OP_LOGE("aclblasCgetrfBatched", "Aarray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // infoArray is required only when PivotArray is provided (pivot mode).
    // PivotArray == NULL && infoArray == NULL is legal (non-pivot mode without info output).
    if (pivotArray != nullptr && infoArray == nullptr) {
        OP_LOGE("aclblasCgetrfBatched", "infoArray must not be nullptr when PivotArray is provided");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCgetrfBatched(
    aclblasHandle_t handle, int n, aclblasComplex* const Aarray[], int lda, int* PivotArray, int* infoArray,
    int batchSize)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgetrfBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t validRet = ValidateGetrfBatchedParams(n, Aarray, lda, PivotArray, infoArray, batchSize);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // In-place factorization needs 0 extra bytes from the handle workspace.
    const aclblasStatus_t workspaceRet = EnsureDefaultWorkspace(handle, 0);
    if (workspaceRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCgetrfBatched", "workspace ensure failed, ret=%d", static_cast<int>(workspaceRet));
        return workspaceRet;
    }

    // Get core count and compute batch distribution
    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasCgetrfBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t uBatchSize = static_cast<uint32_t>(batchSize);
    uint32_t batchPerCore = (uBatchSize - 1) / coreNum + 1;
    uint32_t usedCoreNum = (uBatchSize - 1) / batchPerCore + 1;

    // Build TilingData
    CgetrfBatchedTilingData tiling = {};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.usedCoreNum = usedCoreNum;
    tiling.batchPerCore = batchPerCore;
    tiling.batchTail = uBatchSize - (usedCoreNum - 1) * batchPerCore;
    tiling.usePivot = (PivotArray != nullptr) ? 1u : 0u;

    OP_LOGD(
        "aclblasCgetrfBatched", "tiling: n=%u lda=%u usedCoreNum=%u batchPerCore=%u batchTail=%u usePivot=%u", tiling.n,
        tiling.lda, tiling.usedCoreNum, tiling.batchPerCore, tiling.batchTail, tiling.usePivot);
    OP_LOGI("aclblasCgetrfBatched", "launching kernel: blocks=%u, cores=%u", usedCoreNum, coreNum);

    aclError launchStatus = cgetrf_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex**>(Aarray)), reinterpret_cast<uint8_t*>(PivotArray),
        reinterpret_cast<uint8_t*>(infoArray), tiling, usedCoreNum, handle->stream);
    if (launchStatus != ACL_SUCCESS) {
        OP_LOGE("aclblasCgetrfBatched", "kernel launch failed, error=%d", static_cast<int>(launchStatus));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
