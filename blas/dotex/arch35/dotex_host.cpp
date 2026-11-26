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
 * \brief Extended dot product: result = sum(x[i*incx] * y[i*incy])
 *        incx==incy → SIMD regbase,  incx!=incy → SIMT grid-stride.
 *        Input: FP16 / BF16 / FP32, compute in FP32, result matches input type.
 *        Arch35 (ascend950) host-side implementation.
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "dotex_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

void dotex_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* result, const DotexTilingData& tilingData, uint8_t* workSpace, uint32_t numBlocks,
    void* stream);

static uint32_t MapXType(aclDataType xType)
{
    switch (xType) {
        case ACL_FLOAT:
            return DOTEX_XTYPE_FP32;
        case ACL_FLOAT16:
            return DOTEX_XTYPE_FP16;
        case ACL_BF16:
            return DOTEX_XTYPE_BF16;
        default:
            return DOTEX_XTYPE_FP32;
    }
}

// 对齐 arch22 e676ddc3 口径：步长 span 超出 uint32 域时显式拒绝，两架构
// 对同一入参返回码一致（arch35 kernel 使用 int64 索引无内存安全问题，
// 但跨架构口径不一致会让移植调用方踩到返回码差异）。
static aclblasStatus_t ValidateDotexSpan(int n, int incx, int incy)
{
    if (n <= 1) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint64_t absIncx =
        (incx < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(incx)) : static_cast<uint64_t>(incx);
    const uint64_t absIncy =
        (incy < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(incy)) : static_cast<uint64_t>(incy);
    const uint64_t spanX = 1ULL + static_cast<uint64_t>(n - 1) * absIncx;
    const uint64_t spanY = 1ULL + static_cast<uint64_t>(n - 1) * absIncy;
    if (spanX > static_cast<uint64_t>(UINT32_MAX) || spanY > static_cast<uint64_t>(UINT32_MAX)) {
        OP_LOGE("aclblasDotEx", "strided vector span exceeds uint32 domain");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateDotexParams(
    int n, const void* x, aclDataType xType, int incx, const void* y, aclDataType yType, int incy, void* result,
    aclDataType resultType, aclDataType executionType)
{
    if (n < 0) {
        OP_LOGE("aclblasDotEx", "n=%d must be >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        OP_LOGE("aclblasDotEx", "incx and incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasDotEx", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasDotEx", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasDotEx", "y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (xType != yType) {
        OP_LOGE("aclblasDotEx", "xType %d != yType %d", xType, yType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (xType != ACL_FLOAT && xType != ACL_FLOAT16 && xType != ACL_BF16) {
        OP_LOGE("aclblasDotEx", "xType %d not supported (only FP32/FP16/BF16)", xType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (resultType != xType) {
        OP_LOGE("aclblasDotEx", "resultType %d must match xType %d", resultType, xType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (executionType != ACL_FLOAT) {
        OP_LOGE("aclblasDotEx", "executionType must be ACL_FLOAT, got %d", executionType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    return ValidateDotexSpan(n, incx, incy);
}

static DotexTilingData CalDotexTilingData(int n, aclDataType xType, int incx, int incy, uint32_t vectorCoreNum)
{
    DotexTilingData tiling{};
    tiling.n = n;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.srcType = MapXType(xType);

    uint32_t useCoreNum = std::min(vectorCoreNum, static_cast<uint32_t>(n));
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    tiling.useCoreNum = useCoreNum;

    // SIMT stride offsets
    tiling.kx = (incx >= 0) ? 0 : (1LL - static_cast<int64_t>(n)) * incx;
    tiling.ky = (incy >= 0) ? 0 : (1LL - static_cast<int64_t>(n)) * incy;

    if (incx == incy) {
        tiling.numThreads = 1;
    } else {
        tiling.numThreads = std::min(
            CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), useCoreNum), SIMT_MIN_THREAD_NUM),
            SIMT_MAX_THREAD_NUM);
    }

    return tiling;
}

static aclblasStatus_t LaunchDotexKernel(
    _aclblas_handle* h, const DotexTilingData& tilingData, const void* x, const void* y, void* result)
{
    size_t workspaceSize = tilingData.useCoreNum * (tilingData.numThreads + 1) * sizeof(float);
    if (workspaceSize > GetEffectiveWorkspaceSize(h)) {
        OP_LOGE("aclblasDotEx", "workspace %zu > handle %zu", workspaceSize, GetEffectiveWorkspaceSize(h));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    dotex_kernel_do(
        const_cast<uint8_t*>(static_cast<const uint8_t*>(x)), const_cast<uint8_t*>(static_cast<const uint8_t*>(y)),
        static_cast<uint8_t*>(result), tilingData, workspaceDevice, tilingData.useCoreNum, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasDotEx(
    aclblasHandle_t handle, int n, const void* x, aclDataType xType, int incx, const void* y, aclDataType yType,
    int incy, void* result, aclDataType resultType, aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasDotEx", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n == 0) {
        // issue #966: n==0 快速路径不读 x/y/inc，保留既有快速成功返回契约（对齐
        // crot QuickReturnSkipsDataPointerValidation 口径）；仅校验写 result 所需
        // 子集：result 空指针与 resultType 元素宽度——防止按 4 字节宽度向 1 字节
        // 元素宽度的 resultType 缓冲（如 ACL_INT8）越界写，result 空指针时不再以
        // EXECUTION_FAILED 暴露 memset 失败位置。
        if (result == nullptr) {
            OP_LOGE("aclblasDotEx", "result must not be nullptr");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (resultType != ACL_FLOAT && resultType != ACL_FLOAT16 && resultType != ACL_BF16) {
            OP_LOGE("aclblasDotEx", "resultType %d not supported (only FP32/FP16/BF16)", static_cast<int>(resultType));
            return ACLBLAS_STATUS_NOT_SUPPORTED;
        }
        size_t sz = (resultType == ACL_FLOAT16 || resultType == ACL_BF16) ? sizeof(uint16_t) : sizeof(float);
        aclError ret = aclrtMemset(result, sz, 0, sz);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasDotEx", "aclrtMemset failed");
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t status = ValidateDotexParams(n, x, xType, incx, y, yType, incy, result, resultType, executionType);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    uint32_t vectorCoreNum = GetAivCoreCount();
    if (vectorCoreNum == 0) {
        OP_LOGE("aclblasDotEx", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    DotexTilingData tilingData = CalDotexTilingData(n, xType, incx, incy, vectorCoreNum);
    OP_LOGD(
        "aclblasDotEx", "n=%d inc=%d incy=%d cores=%u threads=%u", tilingData.n, tilingData.incx, tilingData.incy,
        tilingData.useCoreNum, tilingData.numThreads);

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    return LaunchDotexKernel(h, tilingData, x, y, result);
}
