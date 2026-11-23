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
 * \file scalex_host.cpp
 * \brief Host-side implementation for aclblasScalex (arch22 / dav-2201): x = alpha * x.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "scalex_tiling_data.h"
#include "scalex_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasScalex";
constexpr uint32_t SCALEX_MIN_ELEMS_PER_CORE = 256U;

bool IsSupportedScalexType(aclDataType dtype)
{
    return (dtype == ACL_FLOAT16) || (dtype == ACL_BF16) || (dtype == ACL_FLOAT);
}

aclblasStatus_t ValidateScalexParams(
    int n, const void* alpha, aclDataType alphaType, void* x, aclDataType xType, int incx, aclDataType executionType)
{
    if (alphaType != ACL_FLOAT || executionType != ACL_FLOAT) {
        OP_LOGE(OP_TAG, "alphaType/executionType must be ACL_FLOAT");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedScalexType(xType)) {
        OP_LOGE(OP_TAG, "xType(%d) not supported", static_cast<int>(xType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (alpha == nullptr || x == nullptr) {
        OP_LOGE(OP_TAG, "alpha/x must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint64_t spanX = 1ULL + static_cast<uint64_t>(n - 1) * static_cast<uint64_t>(incx);
    if (spanX > static_cast<uint64_t>(UINT32_MAX)) {
        OP_LOGE(OP_TAG, "strided vector span exceeds uint32 domain");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

ScalexTilingData CalScalexTilingData(
    int n, int incx, uint32_t numBlocks, uint32_t xType, float alphaHost, bool alphaIsDevice)
{
    ScalexTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.tileSize = SCALEX_ARCH22_TILE;
    tiling.numBlocks = numBlocks;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.startOffsetX = 0LL;
    tiling.alpha = alphaIsDevice ? 0.0F : alphaHost;
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

aclblasStatus_t aclblasScalex(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, void* x, aclDataType xType, int incx,
    aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (incx == 0) {
        OP_LOGE(OP_TAG, "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx < 0) {
        OP_LOGE(OP_TAG, "negative incx=%d not supported", incx);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    const aclblasStatus_t st = ValidateScalexParams(n, alpha, alphaType, x, xType, incx, executionType);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
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
    const float alphaHost = alphaIsDevice ? 0.0F : *reinterpret_cast<const float*>(alpha);

    const uint32_t maxUseful = CeilDiv<uint32_t>(static_cast<uint32_t>(n), SCALEX_MIN_ELEMS_PER_CORE);
    const uint32_t numBlocks = std::min(aivCoreNum, std::max(1U, maxUseful));
    const ScalexTilingData tiling = CalScalexTilingData(
        n, incx, numBlocks, static_cast<uint32_t>(xType), alphaHost, alphaIsDevice);
    void* alphaPtr = alphaIsDevice ? const_cast<void*>(alpha) : nullptr;
    scalex_kernel_do(reinterpret_cast<uint8_t*>(x), alphaPtr, tiling, numBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
