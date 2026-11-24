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
 * \file dotex_host.cpp
 * \brief Host-side implementation for aclblasDotEx (arch22 / dav-2201).
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "dotex_tiling_data.h"
#include "dotex_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasDotEx";
constexpr uint32_t DOTEX_MIN_ELEMS_PER_CORE = 256U;

aclblasStatus_t ValidateDotexTypes(
    aclDataType xType, aclDataType yType, aclDataType resultType, aclDataType executionType)
{
    if (xType != yType) {
        OP_LOGE(OP_TAG, "xType %d != yType %d", xType, yType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (xType != ACL_FLOAT && xType != ACL_FLOAT16 && xType != ACL_BF16) {
        OP_LOGE(OP_TAG, "xType %d not supported (only FP32/FP16/BF16)", xType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (resultType != xType) {
        OP_LOGE(OP_TAG, "resultType %d must match xType %d", resultType, xType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (executionType != ACL_FLOAT) {
        OP_LOGE(OP_TAG, "executionType must be ACL_FLOAT, got %d", executionType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateDotexSpan(int n, int incx, int incy)
{
    (void)n;
    (void)incx;
    (void)incy;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateDotexParams(
    int n, const void* x, aclDataType xType, int incx, const void* y, aclDataType yType, int incy, void* result,
    aclDataType resultType, aclDataType executionType)
{
    if (n < 0) {
        OP_LOGE(OP_TAG, "n=%d must be >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN) {
        OP_LOGE(OP_TAG, "incx/incy invalid: incx=%d incy=%d", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr || (n > 0 && (x == nullptr || y == nullptr))) {
        OP_LOGE(OP_TAG, "result must not be nullptr, and x/y must not be nullptr when n>0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const aclblasStatus_t typeRet = ValidateDotexTypes(xType, yType, resultType, executionType);
    if (typeRet != ACLBLAS_STATUS_SUCCESS) {
        return typeRet;
    }
    return ValidateDotexSpan(n, incx, incy);
}

DotexTilingData CalDotexTilingData(int n, aclDataType xType, int incx, int incy, uint32_t numBlocks)
{
    DotexTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.numBlocks = numBlocks;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.startOffsetX = (incx >= 0) ? 0LL : (1LL - static_cast<int64_t>(n)) * static_cast<int64_t>(incx);
    tiling.startOffsetY = (incy >= 0) ? 0LL : (1LL - static_cast<int64_t>(n)) * static_cast<int64_t>(incy);
    tiling.xType = static_cast<uint32_t>(xType);
    if (numBlocks == 0U) {
        return tiling;
    }
    tiling.perCoreN = tiling.totalN / numBlocks;
    tiling.remainder = tiling.totalN % numBlocks;
    return tiling;
}

} // namespace

aclblasStatus_t aclblasDotEx(
    aclblasHandle_t handle, int n, const void* x, aclDataType xType, int incx, const void* y, aclDataType yType,
    int incy, void* result, aclDataType resultType, aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const aclblasStatus_t status =
        ValidateDotexParams(n, x, xType, incx, y, yType, incy, result, resultType, executionType);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n == 0) {
        const size_t sz = (resultType == ACL_FLOAT16 || resultType == ACL_BF16) ? sizeof(uint16_t) : sizeof(float);
        if (aclrtMemset(result, sz, 0, sz) != ACL_SUCCESS) {
            OP_LOGE(OP_TAG, "aclrtMemset failed");
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint32_t totalN = static_cast<uint32_t>(n);
    const uint32_t numBlocks = std::max(1U, std::min(CeilDiv(totalN, DOTEX_MIN_ELEMS_PER_CORE), aivCoreNum));
    const DotexTilingData tiling = CalDotexTilingData(n, xType, incx, incy, numBlocks);

    const size_t wsSize = static_cast<size_t>(numBlocks) * sizeof(float);
    if (wsSize > GetEffectiveWorkspaceSize(handle)) {
        OP_LOGE(OP_TAG, "workspace %zu > handle %zu", wsSize, GetEffectiveWorkspaceSize(handle));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint8_t* wsDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    OP_LOGD(
        OP_TAG, "launching kernel: n=%u blocks=%u incx=%d incy=%d xType=%u", totalN, numBlocks, incx, incy,
        tiling.xType);
    const aclError launchErr = dotex_kernel_do(
        const_cast<uint8_t*>(static_cast<const uint8_t*>(x)), const_cast<uint8_t*>(static_cast<const uint8_t*>(y)),
        static_cast<uint8_t*>(result), tiling, wsDevice, numBlocks, handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "dotex_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
