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
 * \file sgels_batched_host.cpp
 * \brief Host-side API for batched least-squares solver: aclblasSgelsBatched.
 *
 * SIMT model: single kernel launch handles all batches internally.
 * OP_T is converted to OP_N via in-kernel transpose of A (no extra kernel launch).
 * Kernel reads per-batch addresses directly from device pointer arrays (no flatten).
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "sgels_batched_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

struct SgelsBatchedTilingData;

void sgels_decompose_kernel_do(
    uint8_t* aArray, uint8_t* cArray, uint8_t* workspace, uint8_t* devInfo, const SgelsBatchedTilingData& tiling,
    uint32_t numBlocks, void* stream);

static const char* const TAG = "aclblasSgelsBatched";
static constexpr uint32_t SIMT_OPTIMAL_THREADS = 128;

static aclblasStatus_t ValidateSgelsBatchedParams(
    aclblasOperation_t trans, int m, int n, int nrhs, float* const aArray[], int lda, float* const cArray[], int ldc,
    int* devInfo, int batchSize)
{
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T) {
        OP_LOGE(TAG, "trans must be OP_N(111) or OP_T(112), got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0) {
        OP_LOGE(TAG, "m must be >= 0, got %d", m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE(TAG, "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (nrhs < 0) {
        OP_LOGE(TAG, "nrhs must be >= 0, got %d", nrhs);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        OP_LOGE(TAG, "batchSize must be >= 0, got %d", batchSize);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) {
        OP_LOGE(TAG, "lda must be >= max(1, m), got lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int maxMN = std::max(m, n);
    if (ldc < std::max(1, maxMN)) {
        OP_LOGE(TAG, "ldc must be >= max(1, m, n), got ldc=%d, m=%d, n=%d", ldc, m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize > 0 && aArray == nullptr) {
        OP_LOGE(TAG, "Aarray must not be nullptr when batchSize > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize > 0 && cArray == nullptr) {
        OP_LOGE(TAG, "Carray must not be nullptr when batchSize > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (devInfo == nullptr) {
        OP_LOGE(TAG, "devInfo must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static SgelsBatchedTilingData BuildTilingData(
    aclblasOperation_t trans, int m, int n, int nrhs, int lda, int ldc, int batchSize, uint32_t coreNum)
{
    uint32_t batchSizeU32 = static_cast<uint32_t>(batchSize);
    uint32_t usedCoreNum = std::max(1u, std::min(batchSizeU32, coreNum));
    uint32_t batchPerCore = (batchSizeU32 + usedCoreNum - 1) / usedCoreNum;
    uint32_t batchTail = batchSizeU32 - (usedCoreNum - 1) * batchPerCore;

    const bool doTranspose = (trans == ACLBLAS_OP_T);
    const int effM = doTranspose ? n : m;
    const int effN = doTranspose ? m : n;
    const int effLda = doTranspose ? n : lda;
    uint32_t minMN = static_cast<uint32_t>(std::min(effM, effN));
    uint32_t maxMN = static_cast<uint32_t>(std::max(effM, effN));

    SgelsBatchedTilingData tiling = {};
    tiling.m = static_cast<uint32_t>(effM);
    tiling.n = static_cast<uint32_t>(effN);
    tiling.nrhs = static_cast<uint32_t>(nrhs);
    tiling.lda = static_cast<int32_t>(effLda);
    tiling.ldc = static_cast<int32_t>(ldc);
    tiling.batchSize = batchSizeU32;
    tiling.batchPerCore = batchPerCore;
    tiling.batchTail = batchTail;
    tiling.usedCoreNum = usedCoreNum;
    tiling.numThreads = SIMT_OPTIMAL_THREADS;
    tiling.minMN = minMN;
    tiling.maxMN = maxMN;
    tiling.trans = doTranspose ? 1u : 0u;
    tiling.origLda = static_cast<int32_t>(lda);
    return tiling;
}

static aclblasStatus_t LaunchGelsKernel(
    _aclblas_handle* h, float* const aArray[], float* const cArray[], int* devInfo,
    const SgelsBatchedTilingData& tiling)
{
    size_t tauBytes = static_cast<size_t>(tiling.batchSize) * tiling.minMN * sizeof(float);
    // issue #864: OP_T 时 kernel 侧转置按 origLda * origN（= 用户 lda * 用户 n，origN=tiling.m）
    // 复制 A，tempA 每 core 尺寸必须与 kernel 的 tempAPerCoreFloats 一致；此前按 tiling.n
    // （= 用户 m）分配，n>m 时各核 tempA 交叠且越出 workspace 末尾
    size_t tempAPerCoreBytes = tiling.trans ? static_cast<size_t>(tiling.origLda) * tiling.m * sizeof(float) : 0;
    size_t tempABytes = tempAPerCoreBytes * tiling.usedCoreNum;
    size_t workspaceBytes = tauBytes + tempABytes;

    size_t availableBytes = GetEffectiveWorkspaceSize(h);
    if (workspaceBytes > availableBytes) {
        OP_LOGE(
            TAG,
            "workspace required %zu bytes, but only %zu bytes available. "
            "Please call aclblasSetWorkspace with size >= %zu bytes",
            workspaceBytes, availableBytes, workspaceBytes);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint8_t* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    OP_LOGI(
        TAG, "launching SIMT kernel: cores=%u, batches=%u, threads=%u", tiling.usedCoreNum, tiling.batchSize,
        tiling.numThreads);

    sgels_decompose_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float**>(aArray)),
        reinterpret_cast<uint8_t*>(const_cast<float**>(cArray)), workspace, reinterpret_cast<uint8_t*>(devInfo), tiling,
        tiling.usedCoreNum, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSgelsBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, float* const aArray[], int lda,
    float* const cArray[], int ldc, int* devInfo, int batchSize)
{
    if (handle == nullptr) {
        OP_LOGE(TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    auto* h = handle;

    aclblasStatus_t validRet =
        ValidateSgelsBatchedParams(trans, m, n, nrhs, aArray, lda, cArray, ldc, devInfo, batchSize);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    if (m == 0 || n == 0 || nrhs == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE(TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    SgelsBatchedTilingData tiling = BuildTilingData(trans, m, n, nrhs, lda, ldc, batchSize, coreNum);
    if (tiling.usedCoreNum == 0) {
        OP_LOGE(TAG, "usedCoreNum is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    OP_LOGD(
        TAG,
        "tiling: m=%u n=%u nrhs=%u lda=%d ldc=%d batchSize=%u "
        "batchPerCore=%u batchTail=%u usedCoreNum=%u minMN=%u maxMN=%u trans=%u origLda=%d",
        tiling.m, tiling.n, tiling.nrhs, tiling.lda, tiling.ldc, tiling.batchSize, tiling.batchPerCore,
        tiling.batchTail, tiling.usedCoreNum, tiling.minMN, tiling.maxMN, tiling.trans, tiling.origLda);

    return LaunchGelsKernel(h, aArray, cArray, devInfo, tiling);
}
