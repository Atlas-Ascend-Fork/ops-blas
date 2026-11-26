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
 * \file stpttr_host.cpp
 * \brief Host-side implementation for aclblasStpttr (arch22 / dav-2201).
 */

#include <algorithm>
#include "acl/acl.h"
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "stpttr_tiling_data.h"
#include "stpttr_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasStpttr";

aclblasStatus_t ValidateStpttrParams(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, int lda, const float* ap, const float* a)
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
    if (lda < std::max(1, n)) {
        OP_LOGE(OP_TAG, "lda=%d, expected >= max(1, n)=%d", lda, std::max(1, n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (ap == nullptr || a == nullptr) {
        OP_LOGE(OP_TAG, "AP/A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t CalStpttrTilingData(StpttrTilingData* tiling, int n, int lda, int uploVal)
{
    const uint32_t vectorCoreNum = GetAivCoreCount();
    if (vectorCoreNum == 0U) {
        OP_LOGE(OP_TAG, "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useCoreNum = std::min(vectorCoreNum, static_cast<uint32_t>(n));
    if (useCoreNum == 0U) {
        useCoreNum = 1U;
    }
    tiling->uplo = uploVal;
    tiling->n = n;
    tiling->lda = lda;
    tiling->useCoreNum = useCoreNum;
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasStpttr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* AP, float* A, int lda)
{
    const aclblasStatus_t status = ValidateStpttrParams(handle, n, uplo, lda, AP, A);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return status;
    }

    StpttrTilingData tiling{};
    const int uploVal = (uplo == ACLBLAS_LOWER) ? 0 : 1;
    const aclblasStatus_t tilingStatus = CalStpttrTilingData(&tiling, n, lda, uploVal);
    if (tilingStatus != ACLBLAS_STATUS_SUCCESS) {
        return tilingStatus;
    }

    const aclError launchErr = stpttr_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(AP)), reinterpret_cast<uint8_t*>(A), tiling, tiling.useCoreNum,
        handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "stpttr_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
