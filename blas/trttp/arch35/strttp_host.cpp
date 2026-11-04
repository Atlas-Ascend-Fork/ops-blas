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
#include "strttp_tiling_data.h"

void strttp_kernel_do(uint8_t* a, uint8_t* ap, const TrttpTilingData& tiling, uint32_t numBlocks, void* stream);

// ---------------------------------------------------------------------------
// Parameter validation — returns non-success if invalid
// ---------------------------------------------------------------------------
static aclblasStatus_t ValidateParams(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, int lda, const float* A, const float* AP)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasStrttp", "handle is nullptr");
        // issue #765: 对齐 api_list.md 契约（docs/zh/api_list.md:40），同 stpttr（#722）
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasStrttp", "n=%d, expected >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) {
        OP_LOGE("aclblasStrttp", "uplo=%d, expected 121(UPPER) or 122(LOWER)", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasStrttp", "A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (AP == nullptr) {
        OP_LOGE("aclblasStrttp", "AP must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasStrttp", "lda=%d, expected >= max(1,n)=%d", lda, std::max(1, n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Tiling data computation
// ---------------------------------------------------------------------------
static aclblasStatus_t CalTilingData(TrttpTilingData* tiling, uint32_t n, uint32_t lda, uint32_t uploVal)
{
    uint32_t vecCoreNum = GetAivCoreCount();
    if (vecCoreNum == 0) {
        OP_LOGE("aclblasStrttp", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint32_t useCoreNum = vecCoreNum;
    if (useCoreNum > n) {
        useCoreNum = n;
    }
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    tiling->n = n;
    tiling->lda = lda;
    tiling->uplo = uploVal;
    tiling->useCoreNum = useCoreNum;
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Kernel execution & cleanup
// ---------------------------------------------------------------------------
static aclblasStatus_t ExecuteKernel(const float* A, float* AP, const TrttpTilingData& tiling, aclrtStream stream)
{
    strttp_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(AP), tiling, tiling.useCoreNum,
        stream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
aclblasStatus_t aclblasStrttp(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* A, int lda, float* AP)
{
    aclblasStatus_t status = ValidateParams(handle, n, uplo, lda, A, AP);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    auto* h = handle;
    uint32_t uploVal = (uplo == ACLBLAS_LOWER) ? 0 : 1;
    uint32_t uN = static_cast<uint32_t>(n);
    uint32_t uLda = static_cast<uint32_t>(lda);

    TrttpTilingData tiling{};
    status = CalTilingData(&tiling, uN, uLda, uploVal);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    return ExecuteKernel(A, AP, tiling, h->stream);
}
