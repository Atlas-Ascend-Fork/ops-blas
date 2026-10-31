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
#include <cstdio>
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "stpttr_tiling_data.h"

void stpttr_kernel_do(
    uint8_t* aPacked, uint8_t* aFull, uint8_t* workSpace, const TpttrTilingData& tiling, uint32_t numBlocks,
    void* stream);

// ---------------------------------------------------------------------------
// Parameter validation — returns non-success if invalid
// ---------------------------------------------------------------------------
static aclblasStatus_t ValidateParams(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, int lda, const float* AP, const float* A)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasStpttr", "handle is nullptr");
        // issue #722: 对齐 api_list.md 契约（docs/zh/api_list.md:40）
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasStpttr", "n=%d, expected >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) {
        OP_LOGE("aclblasStpttr", "uplo=%d, expected 121(UPPER) or 122(LOWER)", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (lda < n) {
        OP_LOGE("aclblasStpttr", "lda=%d, expected >= n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (AP == nullptr) {
        OP_LOGE("aclblasStpttr", "AP must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasStpttr", "A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Tiling data computation
// ---------------------------------------------------------------------------
static aclblasStatus_t CalTilingData(TpttrTilingData* tiling, int n, int lda, int uploVal)
{
    uint32_t vectorCoreNum = GetAivCoreCount();
    if (vectorCoreNum == 0) {
        OP_LOGE("aclblasStpttr", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint32_t useCoreNum = vectorCoreNum;
    if (useCoreNum > static_cast<uint32_t>(n)) {
        useCoreNum = static_cast<uint32_t>(n);
    }
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    tiling->uplo = uploVal;
    tiling->n = n;
    tiling->lda = lda;
    tiling->useCoreNum = useCoreNum;
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Kernel execution & cleanup
// ---------------------------------------------------------------------------
static aclblasStatus_t ExecuteKernel(const float* AP, float* A, const TpttrTilingData& tiling, aclrtStream stream)
{
    stpttr_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(AP)), reinterpret_cast<uint8_t*>(A), nullptr, tiling,
        tiling.useCoreNum, stream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
aclblasStatus_t aclblasStpttr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* AP, float* A, int lda)
{
    aclblasStatus_t status = ValidateParams(handle, n, uplo, lda, AP, A);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    auto* h = handle;
    int uploVal = (uplo == ACLBLAS_LOWER) ? 0 : 1;

    TpttrTilingData tiling{};
    status = CalTilingData(&tiling, n, lda, uploVal);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    return ExecuteKernel(AP, A, tiling, h->stream);
}
