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
 * \brief Mixed-precision vector scaling: x = alpha * x
 *        Arch35 (ascend950) host-side implementation.
 *        Dual-path: AIV SIMD for incx==1, SIMT for incx!=1.
 *        Supports FP32, FP16, BF16 xTypes with FP32 compute precision.
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "scalex_tiling_data.h"
#include "scalex_kernel.h"

static uint32_t GetBytePerElement(uint32_t xType)
{
    constexpr uint32_t kFp16Bf16Size = 2; // sizeof(half) == sizeof(bfloat16_t)
    constexpr uint32_t kFloatSize = sizeof(float);

    if (xType == static_cast<uint32_t>(ACL_FLOAT)) {
        return 2 * kFloatSize; // inQueue + outQueue
    }
    return 2 * kFp16Bf16Size + kFloatSize; // inQueue + outQueue + midBuf(cast to float)
}

static uint32_t GetAlignUnit(uint32_t xType)
{
    if (xType == static_cast<uint32_t>(ACL_FLOAT)) {
        return 32 / sizeof(float);
    }
    return 32 / 2;
}

static aclblasStatus_t ValidateScalexParams(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, void* x, aclDataType xType, int incx,
    aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasScalex", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n > 0 && alpha == nullptr) {
        OP_LOGE("aclblasScalex", "alpha must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && x == nullptr) {
        OP_LOGE("aclblasScalex", "x must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alphaType != ACL_FLOAT) {
        OP_LOGE("aclblasScalex", "alphaType must be ACL_FLOAT(0), got %d", static_cast<int>(alphaType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (executionType != ACL_FLOAT) {
        OP_LOGE("aclblasScalex", "executionType must be ACL_FLOAT(0), got %d", static_cast<int>(executionType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (xType != ACL_FLOAT16 && xType != ACL_BF16 && xType != ACL_FLOAT) {
        OP_LOGE(
            "aclblasScalex", "xType must be ACL_FLOAT16(1), ACL_FLOAT(0) or ACL_BF16(27), got %d",
            static_cast<int>(xType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static ScalexTilingData CalScalexTilingDataContiguous(uint32_t totalN, uint32_t coreNum, uint32_t xType)
{
    ScalexTilingData tiling{};
    tiling.totalN = totalN;
    tiling.incx = 1;
    tiling.xType = xType;

    uint32_t alignUnit = GetAlignUnit(xType);
    if (coreNum == 0 || alignUnit == 0) {
        OP_LOGE("aclblasScalex", "invalid coreNum=%u or alignUnit=%u", coreNum, alignUnit);
        return tiling;
    }
    uint32_t rawPerCore = totalN / coreNum;
    tiling.perCoreN = (rawPerCore / alignUnit) * alignUnit;
    tiling.remainder = totalN - tiling.perCoreN * coreNum;

    uint32_t maxElements = UB_SIZE / GetBytePerElement(xType);
    tiling.tileSize = (maxElements / alignUnit) * alignUnit;

    return tiling;
}

static ScalexTilingData CalScalexTilingDataStrided(int64_t n, int64_t incx, uint32_t numBlocks, uint32_t xType)
{
    ScalexTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.incx = incx;
    tiling.xType = xType;
    tiling.numBlocks = numBlocks;

    uint32_t avgElements = CeilDiv<uint32_t>(static_cast<uint32_t>(n), numBlocks);
    tiling.nthreads = std::min(CeilAlign<uint32_t>(avgElements, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);

    return tiling;
}

static aclblasStatus_t LaunchScalexKernel(
    int n, int incx, uint32_t aivCoreNum, const void* alpha, bool alphaIsDevice, uint32_t xType, void* x,
    aclrtStream stream)
{
    ScalexTilingData tiling;
    uint32_t numBlocks;

    if (incx == 1) {
        uint32_t totalN = static_cast<uint32_t>(n);
        numBlocks = (totalN < aivCoreNum) ? totalN : aivCoreNum;
        tiling = CalScalexTilingDataContiguous(totalN, numBlocks, xType);
    } else {
        numBlocks = std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM), aivCoreNum);
        tiling = CalScalexTilingDataStrided(n, incx, numBlocks, xType);
    }

    tiling.alphaIsDevice = alphaIsDevice ? 1u : 0u;
    if (!alphaIsDevice) {
        tiling.alpha = *reinterpret_cast<const float*>(alpha);
    } else {
        tiling.alpha = 0.0f;
    }

    OP_LOGI(
        "aclblasScalex", "launching kernel: blocks=%u, cores=%u, incx=%d, alphaIsDevice=%u", numBlocks, aivCoreNum,
        incx, tiling.alphaIsDevice);

    void* alphaPtr = alphaIsDevice ? const_cast<void*>(alpha) : nullptr;
    scalex_kernel_do(reinterpret_cast<uint8_t*>(x), alphaPtr, tiling, numBlocks, stream);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasScalex(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, void* x, aclDataType xType, int incx,
    aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasScalex", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // issue #749: n<=0 快速返回保留；incx==0 为非法步长显式拒绝；
    // incx<0 为合法 BLAS 语义但当前实现未支持，显式返回 NOT_SUPPORTED
    // （不再与 n<=0 混合静默假成功）
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (incx == 0) {
        OP_LOGE("aclblasScalex", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx < 0) {
        OP_LOGE("aclblasScalex", "negative incx=%d not supported", incx);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    aclblasStatus_t status = ValidateScalexParams(handle, n, alpha, alphaType, x, xType, incx, executionType);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasScalex", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    auto* h = handle;

    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(alpha, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasScalex", "aclrtPointerGetAttributes for alpha failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    bool alphaIsDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);

    return LaunchScalexKernel(n, incx, aivCoreNum, alpha, alphaIsDevice, static_cast<uint32_t>(xType), x, h->stream);
}
