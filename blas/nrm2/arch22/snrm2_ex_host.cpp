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
 * \file snrm2_ex_host.cpp
 * \brief Host-side implementation for aclblasSnrm2Ex (arch22 / dav-2201).
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "snrm2_ex_tiling_data.h"
#include "snrm2_ex_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasSnrm2Ex";
constexpr uint32_t SNRM2_EX_MIN_ELEMS_PER_CORE = 256U;

aclblasStatus_t ValidateSnrm2ExParams(
    aclblasHandle_t handle, aclDataType xtype, int64_t n, int64_t incx, const void* x, const void* result)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || n > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(OP_TAG, "n invalid: %lld", static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (incx == 0 || incx == static_cast<int64_t>(INT32_MIN) || incx == INT64_MIN) {
        OP_LOGE(OP_TAG, "incx invalid: %lld", static_cast<long long>(incx));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint64_t absInc = static_cast<uint64_t>((incx > 0) ? incx : -incx);
    if (static_cast<uint64_t>(n - 1) > UINT64_MAX / absInc) {
        OP_LOGE(OP_TAG, "(n-1)*abs(incx) exceeds UINT64_MAX");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (xtype != ACL_FLOAT && xtype != ACL_FLOAT16) {
        OP_LOGE(OP_TAG, "invalid xtype=%d", static_cast<int>(xtype));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr || x == nullptr) {
        OP_LOGE(OP_TAG, "x/result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

Snrm2ExTilingData CalcSnrm2ExTilingData(int64_t n, int64_t incx, aclDataType xtype, uint32_t numBlocks)
{
    Snrm2ExTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.numBlocks = numBlocks;
    tiling.incx = incx;
    tiling.startOffset = (incx >= 0) ? 0LL : (1LL - n) * incx;
    tiling.xType = static_cast<uint32_t>(xtype);
    if (numBlocks == 0U) {
        return tiling;
    }
    tiling.perCoreN = tiling.totalN / numBlocks;
    tiling.remainder = tiling.totalN % numBlocks;
    return tiling;
}

} // namespace

aclblasStatus_t aclblasSnrm2Ex(
    aclblasHandle_t handle, aclDataType xtype, const void* x, const int64_t n, const int64_t incx, void* result)
{
    const aclblasStatus_t status = ValidateSnrm2ExParams(handle, xtype, n, incx, x, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n == 0) {
        if (result == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        const float zero = 0.0f;
        if (aclrtMemcpy(result, sizeof(float), &zero, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            OP_LOGE(OP_TAG, "aclrtMemcpy zero result failed");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint32_t totalN = static_cast<uint32_t>(n);
    const uint32_t numBlocks = std::max(1U, std::min(CeilDiv(totalN, SNRM2_EX_MIN_ELEMS_PER_CORE), aivCoreNum));
    const Snrm2ExTilingData tiling = CalcSnrm2ExTilingData(n, incx, xtype, numBlocks);

    const size_t wsBytes = static_cast<size_t>(numBlocks) * 2U * sizeof(float);
    if (wsBytes > GetEffectiveWorkspaceSize(handle)) {
        OP_LOGE(OP_TAG, "workspace %zu > handle %zu", wsBytes, GetEffectiveWorkspaceSize(handle));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    auto* wsDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    OP_LOGD(
        OP_TAG, "launching kernel: n=%u blocks=%u incx=%lld xtype=%u", totalN, numBlocks, static_cast<long long>(incx),
        tiling.xType);
    const aclError launchErr = snrm2_ex_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<void*>(x)), reinterpret_cast<uint8_t*>(result), wsDevice, tiling,
        numBlocks, handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "snrm2_ex_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
