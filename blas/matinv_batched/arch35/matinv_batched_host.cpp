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
 * \file matinv_batched_host.cpp
 * \brief Host-side API for batched single-precision matrix inversion: aclblasSmatinvBatched.
 *
 * Five-step orchestration on the same stream (all asynchronous):
 *   Step 0 - InitPtrArray: build workspace pointer array (matTmpPtrArray).
 *   Step 1 - Copy A[i] -> Ainv[i] (pointer array semantics).
 *   Step 2 - getrf in-place LU decomposition on Ainv[i].
 *   Step 3 - Copy LU factors from Ainv[i] to workspace flat blocks (via pointer array).
 *   Step 4 - getri: read LU from workspace, write inverse to Ainv[i].
 */

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include "log/log.h"
#include "cann_ops_blas.h"

#include "matinv_batched_tiling_data.h"
#include "copy_atoa_kernel.h"
#include "init_ptr_array_kernel.h"

#include "../../getrf_batched/arch35/sgetrf_batched_kernel.h"
#include "../../getrf_batched/arch35/sgetrf_batched_tiling_data.h"
#include "../../getri_batched/arch35/getri_batched_kernel.h"
#include "../../getri_batched/arch35/getri_batched_tiling_data.h"

#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

namespace {

static const char* const TAG = "aclblasSmatinvBatched";

static aclblasStatus_t ValidateSmatinvBatchedParams(
    int n, int lda, int lda_inv, int batchSize, const float* const A[], float* const Ainv[], int* info)
{
    if (n < 0) {
        OP_LOGE(TAG, "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 32) {
        OP_LOGE(TAG, "n must be <= 32, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        OP_LOGE(TAG, "batchSize must be >= 0, got %d", batchSize);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE(TAG, "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda_inv < std::max(1, n)) {
        OP_LOGE(TAG, "lda_inv must be >= max(1, n), got lda_inv=%d, n=%d", lda_inv, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // For empty operations (n==0 or batchSize==0), pointer arguments are not dereferenced,
    // so skip nullptr checks to allow caller to pass null arrays.
    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (A == nullptr) {
        OP_LOGE(TAG, "A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Ainv == nullptr) {
        OP_LOGE(TAG, "Ainv must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (info == nullptr) {
        OP_LOGE(TAG, "info must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

struct MatinvWorkspacePartitions {
    uint8_t* matPtrArray;
    uint8_t* matTmpBuf;
    uint8_t* pivBuf;
    size_t totalWsBytes;
};

static aclblasStatus_t CheckAndPartitionWorkspace(
    _aclblas_handle* h, int n, int batchSize, MatinvWorkspacePartitions& out)
{
    out.totalWsBytes = 0;
    size_t ptrArrayBytes = static_cast<size_t>(batchSize) * sizeof(float*);
    size_t matTmpBytes = static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batchSize) *
                         sizeof(float);
    size_t pivBytes = static_cast<size_t>(n) * static_cast<size_t>(batchSize) * sizeof(int);
    out.totalWsBytes = ptrArrayBytes + matTmpBytes + pivBytes;

    size_t availableBytes = GetEffectiveWorkspaceSize(h);
    if (out.totalWsBytes > availableBytes) {
        OP_LOGE(TAG,
                "workspace required %zu bytes (ptrArray=%zu + matTmp=%zu + piv=%zu), "
                "but only %zu available. Call aclblasSetWorkspace(h, ws, %zu).",
                out.totalWsBytes, ptrArrayBytes, matTmpBytes, pivBytes, availableBytes, out.totalWsBytes);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    auto* ws = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    out.matPtrArray = ws;
    out.matTmpBuf = ws + ptrArrayBytes;
    out.pivBuf = ws + ptrArrayBytes + matTmpBytes;
    OP_LOGD(TAG, "workspace: ptrArray=%zu matTmp=%zu piv=%zu bytes", ptrArrayBytes, matTmpBytes, pivBytes);
    return ACLBLAS_STATUS_SUCCESS;
}

struct MatinvTilingBundle {
SmatinvBatchedTilingData base{};
SmatinvBatchedTilingData step3{};
SgetrfBatchedTilingData getrf{};
SgetriBatchedTilingData getri{};
};

static void BuildAllTiling(
    uint32_t uN, uint32_t lda, uint32_t ldaInv, uint32_t usedCoreNum, uint32_t batchPerCore,
    uint32_t batchTail, uint32_t uBatchSize, MatinvTilingBundle& out)
{
    out.base = {};
    out.base.n = uN;
    out.base.lda = lda;
    out.base.ldaInv = ldaInv;
    out.base.usedCoreNum = usedCoreNum;
    out.base.batchPerCore = batchPerCore;
    out.base.batchTail = batchTail;
    out.base.batchSize = uBatchSize;

    out.step3 = out.base;
    out.step3.lda = out.base.ldaInv;
    out.step3.ldaInv = uN;

    out.getrf = {};
    out.getrf.n = uN;
    out.getrf.lda = ldaInv;
    out.getrf.usedCoreNum = usedCoreNum;
    out.getrf.batchPerCore = batchPerCore;
    out.getrf.batchTail = batchTail;
    out.getrf.usePivot = 1u;

    out.getri = {};
    out.getri.n = uN;
    out.getri.lda = uN;
    out.getri.ldc = ldaInv;
    out.getri.usedCoreNum = usedCoreNum;
    out.getri.batchPerCore = batchPerCore;
    out.getri.batchTail = batchTail;
    out.getri.usePivot = 1u;
}

static void ExecuteFiveStepKernel(
    _aclblas_handle* h, uint32_t usedCoreNum,
    const float* const A[], float* const Ainv[], int* info,
    const MatinvWorkspacePartitions& ws, const MatinvTilingBundle& tiling,
    uint32_t uN, uint32_t uBatchSize)
{
    OP_LOGD(TAG, "Step 0: init_ptr_array_kernel_do, blocks=%u", usedCoreNum);
    init_ptr_array_kernel_do(ws.matPtrArray, ws.matTmpBuf, uN, uBatchSize, usedCoreNum, h->stream);

    OP_LOGD(TAG, "Step 1: copy_kernel_do A->Ainv, blocks=%u", usedCoreNum);
    copy_kernel_do(reinterpret_cast<const uint8_t*>(A),
                   reinterpret_cast<uint8_t*>(const_cast<void*>(static_cast<const void*>(Ainv))), tiling.base,
                   usedCoreNum, h->stream);

    OP_LOGD(TAG, "Step 2: sgetrf_batched_kernel_do on Ainv, blocks=%u", usedCoreNum);
    sgetrf_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<void*>(static_cast<const void*>(Ainv))),
        reinterpret_cast<uint8_t*>(ws.pivBuf), reinterpret_cast<uint8_t*>(info), tiling.getrf,
        usedCoreNum, h->stream);

    OP_LOGD(TAG, "Step 3: copy_kernel_do Ainv->matTmpBuf, blocks=%u", usedCoreNum);
    copy_kernel_do(
        reinterpret_cast<const uint8_t*>(Ainv),
        reinterpret_cast<uint8_t*>(ws.matPtrArray), tiling.step3, usedCoreNum, h->stream);

    OP_LOGD(TAG, "Step 4: sgetri_batched_kernel_do, blocks=%u", usedCoreNum);
    sgetri_batched_kernel_do(
        reinterpret_cast<uint8_t*>(ws.matPtrArray), reinterpret_cast<uint8_t*>(ws.pivBuf),
        reinterpret_cast<uint8_t*>(const_cast<void*>(static_cast<const void*>(Ainv))),
        reinterpret_cast<uint8_t*>(info), tiling.getri, usedCoreNum, h->stream);
}

static aclblasStatus_t LaunchSmatinvBatchedKernel(
    aclblasHandle_t handle, int n, int lda, int lda_inv, int batchSize, const float* const A[], float* const Ainv[],
    int* info)
{
    auto* h = handle;

    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE(TAG, "GetAivCoreCount failed (returned 0)");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t uBatchSize = static_cast<uint32_t>(batchSize);
    uint32_t batchPerCore = (uBatchSize - 1) / coreNum + 1;
    uint32_t usedCoreNum = (uBatchSize - 1) / batchPerCore + 1;
    uint32_t batchTail = uBatchSize - (usedCoreNum - 1) * batchPerCore;

    MatinvWorkspacePartitions ws{};
    aclblasStatus_t wsStatus = CheckAndPartitionWorkspace(h, n, batchSize, ws);
    if (wsStatus != ACLBLAS_STATUS_SUCCESS) {
        return wsStatus;
    }

    uint32_t uN = static_cast<uint32_t>(n);
    MatinvTilingBundle tiling{};
    BuildAllTiling(uN, static_cast<uint32_t>(lda), static_cast<uint32_t>(lda_inv), usedCoreNum, batchPerCore,
                   batchTail, uBatchSize, tiling);

    OP_LOGI(TAG, "launching: n=%u batchSize=%u cores=%u wsBytes=%zu", uN, uBatchSize, usedCoreNum, ws.totalWsBytes);
    OP_LOGD(TAG, "tiling base: n=%u lda=%u ldaInv=%u usedCoreNum=%u batchPerCore=%u batchTail=%u", tiling.base.n,
            tiling.base.lda, tiling.base.ldaInv, tiling.base.usedCoreNum, tiling.base.batchPerCore,
            tiling.base.batchTail);

    ExecuteFiveStepKernel(h, usedCoreNum, A, Ainv, info, ws, tiling, uN, uBatchSize);
    return ACLBLAS_STATUS_SUCCESS;
}

}  // namespace

aclblasStatus_t aclblasSmatinvBatched(
    aclblasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info,
    int batchSize)
{
    if (handle == nullptr) {
        OP_LOGE(TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // Full parameter validation (must be done BEFORE any early returns to detect invalid arguments)
    aclblasStatus_t validationStatus =
        ValidateSmatinvBatchedParams(n, lda, lda_inv, batchSize, A, Ainv, info);
    if (validationStatus != ACLBLAS_STATUS_SUCCESS) {
        return validationStatus;
    }

    // Empty operation: all parameters are valid, but n==0 or batchSize==0 means no work to do
    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchSmatinvBatchedKernel(handle, n, lda, lda_inv, batchSize, A, Ainv, info);
}
