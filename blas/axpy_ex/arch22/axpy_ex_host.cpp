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
 * \file axpy_ex_host.cpp
 * \brief Host-side implementation for aclblasAxpyEx (arch22 / dav-2201): y = alpha * x + y.
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "axpy_ex_tiling_data.h"
#include "axpy_ex_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasAxpyEx";
constexpr uint32_t AXPY_EX_MIN_ELEMS_PER_CORE = 256U;

bool IsSupportedAxpyExVecType(aclDataType dtype)
{
    return (dtype == ACL_FLOAT16) || (dtype == ACL_BF16) || (dtype == ACL_FLOAT);
}

aclblasStatus_t ValidateAxpyExTypes(
    aclDataType alphaType, aclDataType xType, aclDataType yType, aclDataType executionType)
{
    if (alphaType != ACL_FLOAT || executionType != ACL_FLOAT) {
        OP_LOGE(OP_TAG, "alphaType/executionType must be ACL_FLOAT");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedAxpyExVecType(xType) || !IsSupportedAxpyExVecType(yType) || xType != yType) {
        OP_LOGE(OP_TAG, "xType(%d)/yType(%d) unsupported or mismatch", static_cast<int>(xType),
            static_cast<int>(yType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateAxpyExParams(
    int n, const void* alpha, aclDataType alphaType, const void* x, aclDataType xType, int incx, void* y,
    aclDataType yType, int incy, aclDataType executionType)
{
    const aclblasStatus_t typeSt = ValidateAxpyExTypes(alphaType, xType, yType, executionType);
    if (typeSt != ACLBLAS_STATUS_SUCCESS) {
        return typeSt;
    }
    if (alpha == nullptr || x == nullptr || y == nullptr) {
        OP_LOGE(OP_TAG, "alpha/x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN) {
        OP_LOGE(OP_TAG, "incx/incy invalid: incx=%d incy=%d", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

AxpyExTilingData CalAxpyExTilingData(
    int n, int incx, int incy, uint32_t numBlocks, uint32_t xType, float alphaHost, bool alphaIsDevice)
{
    AxpyExTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.tileSize = AXPY_EX_ARCH22_TILE;
    tiling.numBlocks = numBlocks;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.startOffsetX = (incx >= 0) ? 0LL : (1LL - static_cast<int64_t>(n)) * static_cast<int64_t>(incx);
    tiling.startOffsetY = (incy >= 0) ? 0LL : (1LL - static_cast<int64_t>(n)) * static_cast<int64_t>(incy);
    tiling.alpha = alphaIsDevice ? 0.0f : alphaHost;
    tiling.alphaIsDevice = alphaIsDevice ? 1U : 0U;
    tiling.xType = xType;
    if (numBlocks == 0U) {
        return tiling;
    }
    tiling.perCoreN = tiling.totalN / numBlocks;
    tiling.remainder = tiling.totalN % numBlocks;
    return tiling;
}

} // namespace

aclblasStatus_t aclblasAxpyEx(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, const void* x, aclDataType xType,
    int incx, void* y, aclDataType yType, int incy, aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE(OP_TAG, "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t status = ValidateAxpyExParams(
        n, alpha, alphaType, x, xType, incx, y, yType, incy, executionType);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    aclrtPtrAttributes ptrAttr{};
    const aclError aclRet = aclrtPointerGetAttributes(alpha, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtPointerGetAttributes for alpha failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const bool alphaIsDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    const float alphaHost = alphaIsDevice ? 0.0f : *reinterpret_cast<const float*>(alpha);

    const uint32_t totalN = static_cast<uint32_t>(n);
    const uint32_t numBlocks = std::max(1U, std::min(CeilDiv(totalN, AXPY_EX_MIN_ELEMS_PER_CORE), aivCoreNum));
    const AxpyExTilingData tiling = CalAxpyExTilingData(
        n, incx, incy, numBlocks, static_cast<uint32_t>(xType), alphaHost, alphaIsDevice);

    OP_LOGD(OP_TAG, "launching kernel: n=%u blocks=%u incx=%d incy=%d xType=%u", totalN, numBlocks, incx, incy,
        tiling.xType);
    void* alphaPtr = alphaIsDevice ? const_cast<void*>(alpha) : nullptr;
    axpy_ex_kernel_do(
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(x)), reinterpret_cast<uint8_t*>(y), alphaPtr, tiling,
        numBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
