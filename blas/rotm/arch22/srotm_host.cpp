/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "srotm_tiling_data.h"

struct SrotmTilingData;

void srotm_kernel_do(const SrotmTilingData& tiling, uint32_t numBlocks, void* stream);

constexpr uint32_t MAX_CORE_NUM = 50;
constexpr uint32_t MIN_ELEMENTS_PER_CORE = 2048;
constexpr uint32_t MAX_SELECTED_CORES = 8;
constexpr uint32_t DEFAULT_LAUNCH_BLOCKS = 8;

static uint64_t AbsStride(int64_t inc)
{
    return inc >= 0 ? static_cast<uint64_t>(inc) : static_cast<uint64_t>(-(inc + 1)) + 1U;
}

static uint64_t GetVectorStorageSize(int64_t n, int64_t inc)
{
    if (n <= 0) {
        return 0;
    }
    return 1U + static_cast<uint64_t>(n - 1) * AbsStride(inc);
}

static uint32_t SelectCoreNum(uint32_t n)
{
    uint32_t coreNum = (n + MIN_ELEMENTS_PER_CORE - 1) / MIN_ELEMENTS_PER_CORE;
    if (coreNum == 0) {
        coreNum = 1;
    }
    return std::min(coreNum, std::min(MAX_SELECTED_CORES, MAX_CORE_NUM));
}

static aclblasStatus_t CheckSrotmParams(
    aclblasHandle handle, float* x, float* y, const float* sparam, int64_t n, int64_t incx, int64_t incy)
{
    // BLAS 约定 n<=0 时无需执行计算，直接按成功返回。
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // 输入向量和参数表都必须有效，否则无法继续构造旋转计算。
    if (x == nullptr || y == nullptr || sparam == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // 步长不能为 0，否则逻辑向量元素会重复映射到同一地址。
    if (incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // 当前实现将 n 下发为 uint32_t，超出范围时拒绝继续执行。
    if (n > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    aclrtContext currentCtx = nullptr;
    aclError contextRet = aclrtGetCurrentContext(&currentCtx);
    // 运行时上下文未就绪时，后续内存与流操作都不安全。
    if (contextRet != ACL_SUCCESS || currentCtx == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    // handle 为空时无法取得执行流与运行配置。
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static void FillSrotmCoefficients(SrotmTilingData& tiling, const float* sparam)
{
    tiling.sflag = sparam[0];
    tiling.h11 = 1.0f;
    tiling.h12 = 0.0f;
    tiling.h21 = 0.0f;
    tiling.h22 = 1.0f;
    if (tiling.sflag < 0.0f) {
        tiling.h11 = sparam[1];
        tiling.h21 = sparam[2];
        tiling.h12 = sparam[3];
        tiling.h22 = sparam[4];
    } else if (tiling.sflag == 0.0f) {
        tiling.h12 = sparam[3];
        tiling.h21 = sparam[2];
    } else {
        tiling.h11 = sparam[1];
        tiling.h12 = 1.0f;
        tiling.h21 = -1.0f;
        tiling.h22 = sparam[4];
    }
}

// issue #869: param 数组位置探测 + device 时整组 D2H 读回（与 arch35 ResolveSrotmParam
// 同口径），host 不再直接解引用可能位于 device 侧的 sparam
static aclblasStatus_t ResolveSrotmParam(const float* param, float (&paramHost)[5])
{
    bool paramDevice = false;
    const aclblasStatus_t locStatus = CheckPtrLocation(param, &paramDevice);
    if (locStatus != ACLBLAS_STATUS_SUCCESS) {
        return locStatus;
    }
    if (paramDevice) {
        const aclError copyRet =
            aclrtMemcpy(paramHost, sizeof(paramHost), param, sizeof(paramHost), ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else {
        for (int i = 0; i < 5; ++i) {
            paramHost[i] = param[i];
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static SrotmTilingData BuildSrotmTilingData(
    float* x, float* y, const float* sparam, int64_t n, int64_t incx, int64_t incy)
{
    SrotmTilingData tiling = {};
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.y = reinterpret_cast<uint64_t>(y);
    tiling.xStorageSize = GetVectorStorageSize(n, incx);
    tiling.yStorageSize = GetVectorStorageSize(n, incy);
    tiling.n = static_cast<uint32_t>(n);
    // 非 unit-stride 场景难以安全进行多核对齐切分，保守退化为单核。
    tiling.useCoreNum = (incx == 1 && incy == 1) ? SelectCoreNum(static_cast<uint32_t>(n)) : 1U;
    tiling.incx = incx;
    tiling.incy = incy;
    FillSrotmCoefficients(tiling, sparam);
    return tiling;
}

static aclblasStatus_t LaunchSrotmKernel(const SrotmTilingData& tiling, aclrtStream useStream)
{
    srotm_kernel_do(tiling, std::max<uint32_t>(DEFAULT_LAUNCH_BLOCKS, tiling.useCoreNum), useStream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSrotm(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* sparam)
{
    aclblasStatus_t status = CheckSrotmParams(handle, x, y, sparam, n, incx, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    // BLAS 约定 n<=0 时无需执行计算，直接按成功返回；此路径不触碰 sparam。
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // issue #869: param 数组按 rot 族惯例探测位置（与 arch35 一致），device 侧整组 D2H
    // 后在 host 读取控制标量，避免主机直接访问设备地址
    float paramHost[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const aclblasStatus_t paramStatus = ResolveSrotmParam(sparam, paramHost);
    if (paramStatus != ACLBLAS_STATUS_SUCCESS) {
        return paramStatus;
    }
    if (paramHost[0] == -2.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;
    SrotmTilingData tiling = BuildSrotmTilingData(x, y, paramHost, n, incx, incy);

    return LaunchSrotmKernel(tiling, useStream);
}
