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
 * \file rotex_host.cpp
 * \brief Host-side implementation for aclblasRotEx (arch22 / dav-2201).
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/dtype_cast.h"
#include "common/helper/host_utils.h"
#include "rotex_tiling_data.h"
#include "rotex_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasRotEx";
constexpr uint32_t ROTEX_MIN_ELEMS_PER_CORE = 256U;

bool IsSupportedRotExType(aclDataType dtype)
{
    return (dtype == ACL_FLOAT) || (dtype == ACL_FLOAT16) || (dtype == ACL_BF16);
}

aclblasStatus_t CheckRotExTypeSupport(
    aclDataType executionType, aclDataType xType, aclDataType yType, aclDataType csType)
{
    if (executionType != ACL_FLOAT) {
        OP_LOGE(OP_TAG, "executionType=%d not supported", static_cast<int>(executionType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (!IsSupportedRotExType(xType) || !IsSupportedRotExType(yType) || !IsSupportedRotExType(csType)) {
        OP_LOGE(OP_TAG, "xType/yType/csType must be FP32/FP16/BF16");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (xType != yType || xType != csType) {
        OP_LOGE(OP_TAG, "only symmetric types supported (xType==yType==csType)");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateRotExParams(
    aclblasHandle_t handle, int n, const void* x, aclDataType xType, int incx, const void* y, aclDataType yType,
    int incy, const void* c, const void* s, aclDataType csType, aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE(OP_TAG, "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN) {
        OP_LOGE(OP_TAG, "incx/incy invalid: incx=%d incy=%d", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        OP_LOGE(OP_TAG, "x/y/c/s must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return CheckRotExTypeSupport(executionType, xType, yType, csType);
}

aclblasStatus_t ReadScalarValue(const void* ptr, aclDataType csType, aclrtStream stream, float& outVal)
{
    if (csType == ACL_FLOAT) {
        return ReadScalarFromHostOrDevice(static_cast<const float*>(ptr), stream, outVal);
    }
    uint16_t raw = 0U;
    const aclblasStatus_t st = ReadScalarFromHostOrDevice(
        static_cast<const uint16_t*>(ptr), stream, raw);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (csType == ACL_FLOAT16) {
        outVal = blas_common::HalfToFloat(raw);
    } else {
        outVal = blas_common::Bf16ToFloat(raw);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

RotExTilingData SetupRotExTiling(
    int n, int incx, int incy, aclDataType xType, float cReal, float sReal, uint32_t numBlocks)
{
    RotExTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.numBlocks = numBlocks;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.startOffsetX = (incx >= 0) ? 0LL : (1LL - static_cast<int64_t>(n)) * static_cast<int64_t>(incx);
    tiling.startOffsetY = (incy >= 0) ? 0LL : (1LL - static_cast<int64_t>(n)) * static_cast<int64_t>(incy);
    tiling.cReal = cReal;
    tiling.sReal = sReal;
    tiling.xType = static_cast<uint32_t>(xType);
    if (numBlocks == 0U) {
        return tiling;
    }
    tiling.perCoreN = tiling.totalN / numBlocks;
    tiling.remainder = tiling.totalN % numBlocks;
    return tiling;
}

} // namespace

aclblasStatus_t aclblasRotEx(
    aclblasHandle_t handle, int n, void* x, aclDataType xType, int incx, void* y, aclDataType yType, int incy,
    const void* c, const void* s, aclDataType csType, aclDataType executionType)
{
    const aclblasStatus_t st = ValidateRotExParams(
        handle, n, x, xType, incx, y, yType, incy, c, s, csType, executionType);
    if (st != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return st;
    }
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    float cReal = 0.0f;
    float sReal = 0.0f;
    aclblasStatus_t ret = ReadScalarValue(c, csType, handle->stream, cReal);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = ReadScalarValue(s, csType, handle->stream, sReal);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    const uint32_t totalN = static_cast<uint32_t>(n);
    const uint32_t numBlocks = std::max(1U, std::min(CeilDiv(totalN, ROTEX_MIN_ELEMS_PER_CORE), aivCoreNum));
    const RotExTilingData tiling = SetupRotExTiling(n, incx, incy, xType, cReal, sReal, numBlocks);
    OP_LOGD(OP_TAG, "launching kernel: n=%u blocks=%u incx=%d incy=%d xType=%u", totalN, numBlocks, incx, incy,
        tiling.xType);
    const aclError err = rotex_kernel_do(
        static_cast<uint8_t*>(x), static_cast<uint8_t*>(y), tiling, numBlocks, handle->stream);
    if (err != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "rotex_kernel_do launch failed: %d", err);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
