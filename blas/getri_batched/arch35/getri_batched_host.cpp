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
 * \file getri_batched_host.cpp
 * \brief Host-side API for batched single-precision matrix inversion: aclblasSgetriBatched.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "getri_batched_tiling_data.h"
#include "getri_batched_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static aclblasStatus_t ValidateGetriBatchedParams(
    int n, int lda, int ldc, int batchSize, const float* const Aarray[], float* const Carray[], int* infoArray)
{
    if (n < 0) {
        OP_LOGE("aclblasSgetriBatched", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        OP_LOGE("aclblasSgetriBatched", "batchSize must be >= 0, got %d", batchSize);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasSgetriBatched", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, n)) {
        OP_LOGE("aclblasSgetriBatched", "ldc must be >= max(1, n), got ldc=%d, n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // No-op: when n==0 or batchSize==0, no data pointers are accessed.
    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (Aarray == nullptr) {
        OP_LOGE("aclblasSgetriBatched", "Aarray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Carray == nullptr) {
        OP_LOGE("aclblasSgetriBatched", "Carray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (infoArray == nullptr) {
        OP_LOGE("aclblasSgetriBatched", "infoArray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // PivotArray == nullptr is legal (no-pivot mode, P = I).
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchGetriBatchedKernel(
    aclblasHandle_t handle, int n, int lda, int ldc, int batchSize, const int* PivotArray, const float* const Aarray[],
    float* const Carray[], int* infoArray)
{
    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasSgetriBatched", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t uBatchSize = static_cast<uint32_t>(batchSize);
    uint32_t batchPerCore = (uBatchSize - 1) / coreNum + 1;
    uint32_t usedCoreNum = (uBatchSize - 1) / batchPerCore + 1;

    SgetriBatchedTilingData tiling = {};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.usedCoreNum = usedCoreNum;
    tiling.batchPerCore = batchPerCore;
    tiling.batchTail = uBatchSize - (usedCoreNum - 1) * batchPerCore;
    tiling.usePivot = (PivotArray != nullptr) ? 1u : 0u;

    OP_LOGD(
        "aclblasSgetriBatched", "tiling: n=%u lda=%u ldc=%u usedCoreNum=%u batchPerCore=%u batchTail=%u usePivot=%u",
        tiling.n, tiling.lda, tiling.ldc, tiling.usedCoreNum, tiling.batchPerCore, tiling.batchTail, tiling.usePivot);
    OP_LOGI("aclblasSgetriBatched", "launching kernel: blocks=%u, cores=%u", usedCoreNum, coreNum);

    auto* h = handle;
    // issue #1180: a failed launch must not be reported as SUCCESS.
    const aclError launch = sgetri_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<const float**>(Aarray)),
        reinterpret_cast<uint8_t*>(const_cast<int*>(PivotArray)),
        reinterpret_cast<uint8_t*>(const_cast<float**>(Carray)), reinterpret_cast<uint8_t*>(infoArray), tiling,
        usedCoreNum, h->stream);
    CHECK_RET(launch == ACL_SUCCESS, OP_LOGE("aclblasSgetriBatched", "sgetri_batched_kernel_do failed: %d", launch);
              return ACLBLAS_STATUS_EXECUTION_FAILED);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSgetriBatched(
    aclblasHandle_t handle, int n, const float* const Aarray[], int lda, const int* PivotArray, float* const Carray[],
    int ldc, int* infoArray, int batchSize)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSgetriBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t validRet = ValidateGetriBatchedParams(n, lda, ldc, batchSize, Aarray, Carray, infoArray);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    OP_LOGI(
        "aclblasSgetriBatched", "n=%d, lda=%d, ldc=%d, batchSize=%d, usePivot=%d", n, lda, ldc, batchSize,
        (PivotArray != nullptr) ? 1 : 0);

    return LaunchGetriBatchedKernel(handle, n, lda, ldc, batchSize, PivotArray, Aarray, Carray, infoArray);
}
