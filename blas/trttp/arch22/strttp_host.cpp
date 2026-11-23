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
 * \file strttp_host.cpp
 * \brief Host-side implementation for aclblasStrttp (arch22 / dav-2201).
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "strttp_tiling_data.h"
#include "strttp_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasStrttp";

aclblasStatus_t ValidateStrttpParams(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, int lda, const float* a, const float* ap)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE(OP_TAG, "n=%d, expected >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) {
        OP_LOGE(OP_TAG, "uplo=%d, expected 121(UPPER) or 122(LOWER)", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || ap == nullptr) {
        OP_LOGE(OP_TAG, "A/AP must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < n) {
        OP_LOGE(OP_TAG, "lda=%d, expected >= n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t CalStrttpTilingData(StrttpTilingData* tiling, uint32_t n, uint32_t lda, uint32_t uploVal)
{
    const uint32_t vecCoreNum = GetAivCoreCount();
    if (vecCoreNum == 0U) {
        OP_LOGE(OP_TAG, "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useCoreNum = std::min(vecCoreNum, n);
    if (useCoreNum == 0U) {
        useCoreNum = 1U;
    }
    tiling->n = n;
    tiling->lda = lda;
    tiling->uplo = uploVal;
    tiling->useCoreNum = useCoreNum;
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasStrttp(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* A, int lda, float* AP)
{
    const aclblasStatus_t status = ValidateStrttpParams(handle, n, uplo, lda, A, AP);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    StrttpTilingData tiling{};
    const uint32_t uploVal = (uplo == ACLBLAS_LOWER) ? 0U : 1U;
    const aclblasStatus_t tilingStatus =
        CalStrttpTilingData(&tiling, static_cast<uint32_t>(n), static_cast<uint32_t>(lda), uploVal);
    if (tilingStatus != ACLBLAS_STATUS_SUCCESS) {
        return tilingStatus;
    }

    const aclError launchErr = strttp_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(AP), tiling, tiling.useCoreNum,
        handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "strttp_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
