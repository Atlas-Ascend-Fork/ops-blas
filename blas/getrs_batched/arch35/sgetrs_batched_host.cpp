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
 * \file sgetrs_batched_host.cpp
 * \brief Host-side API for batched single-precision triangular solve: aclblasSgetrsBatched.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "sgetrs_batched_tiling_data.h"
#include "sgetrs_batched_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static constexpr uint32_t GETRS_SIMT_MAX_THREADS_HOST = 256;

// Parameter position indices (LAPACK convention):
//   aclblasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchCount)
//   Position:                 1       2    3   4     5       6     7       8       9    10    11
// info = -PARAM_XXX when the corresponding parameter is invalid.
static constexpr int GETRS_PARAM_TRANS = 2;
static constexpr int GETRS_PARAM_N = 3;
static constexpr int GETRS_PARAM_NRHS = 4;
static constexpr int GETRS_PARAM_AARRAY = 5;
static constexpr int GETRS_PARAM_LDA = 6;
static constexpr int GETRS_PARAM_BARRAY = 8;
static constexpr int GETRS_PARAM_LDB = 9;
static constexpr int GETRS_PARAM_BATCH_COUNT = 11;

static void SetInfo(int* info, int val)
{
    if (info != nullptr) {
        *info = val;
    }
}

static aclblasStatus_t ValidateSgetrsBatchedParams(
    aclblasOperation_t trans, int n, int nrhs, int lda, int ldb, int batchCount, const float* const Aarray[],
    float* const Barray[], int* info)
{
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE(
            "aclblasSgetrsBatched", "trans must be OP_N(111), OP_T(112) or OP_C(113), got %d", static_cast<int>(trans));
        SetInfo(info, -GETRS_PARAM_TRANS);
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0) {
        OP_LOGE("aclblasSgetrsBatched", "n must be >= 0, got %d", n);
        SetInfo(info, -GETRS_PARAM_N);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (nrhs < 0 || nrhs > static_cast<int>(GETRS_SIMT_MAX_THREADS_HOST)) {
        OP_LOGE("aclblasSgetrsBatched", "nrhs must be in [0, %u], got %d", GETRS_SIMT_MAX_THREADS_HOST, nrhs);
        SetInfo(info, -GETRS_PARAM_NRHS);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && nrhs > 0 && batchCount > 0 && Aarray == nullptr) {
        OP_LOGE("aclblasSgetrsBatched", "Aarray must not be nullptr");
        SetInfo(info, -GETRS_PARAM_AARRAY);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasSgetrsBatched", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        SetInfo(info, -GETRS_PARAM_LDA);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && nrhs > 0 && batchCount > 0 && Barray == nullptr) {
        OP_LOGE("aclblasSgetrsBatched", "Barray must not be nullptr");
        SetInfo(info, -GETRS_PARAM_BARRAY);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldb < std::max(1, n)) {
        OP_LOGE("aclblasSgetrsBatched", "ldb must be >= max(1, n), got ldb=%d, n=%d", ldb, n);
        SetInfo(info, -GETRS_PARAM_LDB);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchCount < 0) {
        OP_LOGE("aclblasSgetrsBatched", "batchCount must be >= 0, got %d", batchCount);
        SetInfo(info, -GETRS_PARAM_BATCH_COUNT);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static SgetrsBatchedTilingData ComputeTiling(
    aclblasOperation_t trans, int n, int nrhs, int lda, int ldb, const int* devIpiv, int batchCount, uint32_t coreNum)
{
    SgetrsBatchedTilingData tiling = {};
    if (coreNum == 0) {
        return tiling;
    }
    uint32_t uBatchCount = static_cast<uint32_t>(batchCount);
    uint32_t batchPerCore = (uBatchCount - 1) / coreNum + 1;
    uint32_t usedCoreNum = (uBatchCount - 1) / batchPerCore + 1;

    tiling.n = static_cast<uint32_t>(n);
    tiling.nrhs = static_cast<uint32_t>(nrhs);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.usedCoreNum = usedCoreNum;
    tiling.batchPerCore = batchPerCore;
    tiling.batchTail = uBatchCount - (usedCoreNum - 1) * batchPerCore;
    tiling.usePivot = (devIpiv != nullptr) ? 1u : 0u;
    switch (trans) {
        case ACLBLAS_OP_N:
            tiling.trans = GETRS_TRANS_N;
            break;
        case ACLBLAS_OP_T:
            tiling.trans = GETRS_TRANS_T;
            break;
        case ACLBLAS_OP_C:
            tiling.trans = GETRS_TRANS_C;
            break;
        default:
            tiling.trans = GETRS_TRANS_N;
            break;
    }
    return tiling;
}

static aclblasStatus_t LaunchSgetrsBatchedKernel(
    aclblasHandle_t handle, aclblasOperation_t trans, int n, int nrhs, int lda, int ldb, const float* const Aarray[],
    const int* devIpiv, float* const Barray[], int batchCount)
{
    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasSgetrsBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    SgetrsBatchedTilingData tiling = ComputeTiling(trans, n, nrhs, lda, ldb, devIpiv, batchCount, coreNum);

    OP_LOGD(
        "aclblasSgetrsBatched",
        "tiling: n=%u nrhs=%u lda=%u ldb=%u usedCoreNum=%u batchPerCore=%u batchTail=%u usePivot=%u trans=%u", tiling.n,
        tiling.nrhs, tiling.lda, tiling.ldb, tiling.usedCoreNum, tiling.batchPerCore, tiling.batchTail, tiling.usePivot,
        tiling.trans);
    OP_LOGI(
        "aclblasSgetrsBatched",
        "launching kernel: blocks=%u, cores=%u, n=%u, nrhs=%u, trans=%u, usePivot=%u, batchCount=%d",
        tiling.usedCoreNum, coreNum, tiling.n, tiling.nrhs, tiling.trans, tiling.usePivot, batchCount);

    auto* h = handle;
    // Pointer arrays are read-only on device side; casts match kernel_do's uint8_t* signature
    const aclError launchErr = sgetrs_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<const float**>(Aarray)),
        reinterpret_cast<uint8_t*>(const_cast<float**>(Barray)), reinterpret_cast<uint8_t*>(const_cast<int*>(devIpiv)),
        tiling, tiling.usedCoreNum, h->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasSgetrsBatched", "sgetrs_batched_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSgetrsBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda,
    const int* devIpiv, float* const Barray[], int ldb, int* info, int batchCount)
{
    OP_LOGD(
        "aclblasSgetrsBatched",
        "entry: trans=%d, n=%d, nrhs=%d, lda=%d, ldb=%d, batchCount=%d, "
        "Aarray=%p, devIpiv=%p, Barray=%p, info=%p",
        static_cast<int>(trans), n, nrhs, lda, ldb, batchCount, Aarray, devIpiv, Barray, info);

    if (handle == nullptr) {
        OP_LOGE("aclblasSgetrsBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t validRet = ValidateSgetrsBatchedParams(trans, n, nrhs, lda, ldb, batchCount, Aarray, Barray, info);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    SetInfo(info, 0);

    if (n == 0 || nrhs == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchSgetrsBatchedKernel(handle, trans, n, nrhs, lda, ldb, Aarray, devIpiv, Barray, batchCount);
}
