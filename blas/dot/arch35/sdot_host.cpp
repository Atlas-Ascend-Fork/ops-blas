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
 * \file sdot_host.cpp
 * \brief Single-precision dot product (SIMD AIV): result = sum(x[i] * y[i])
 *        Arch35 (ascend950) host-side implementation.
 */

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "sdot_tiling_data.h"

aclError sdot_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* result, uint8_t* workSpace, uint32_t numBlocks, const SdotTilingData& tiling,
    void* stream);

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------
static aclblasStatus_t ValidateSdotParams(int incx, int incy, const float* x, const float* y, const float* result)
{
    // issue #1325: INT32_MIN 步长无合法使用场景，对齐 arch22（#780）与 snrm2_ex arch35
    // 的显式拒绝口径，消除跨架构行为分裂；INT64_MIN 经公共 API（int 步长）不可表达
    if (incx == 0 || incx == INT32_MIN) {
        OP_LOGE("aclblasSdot", "incx must not be zero or INT32_MIN");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0 || incy == INT32_MIN) {
        OP_LOGE("aclblasSdot", "incy must not be zero or INT32_MIN");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasSdot", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasSdot", "y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasSdot", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Tiling data computation
// ---------------------------------------------------------------------------
static SdotTilingData CalSdotTilingData(int64_t n, int64_t incx, int64_t incy, uint32_t vectorCoreNum)
{
    SdotTilingData tiling{};
    tiling.n = n;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.useCoreNum = std::min(vectorCoreNum, static_cast<uint32_t>(n));
    if (tiling.useCoreNum == 0) {
        tiling.useCoreNum = 1;
    }
    return tiling;
}

// ---------------------------------------------------------------------------
// Kernel launch helper
// ---------------------------------------------------------------------------
static aclblasStatus_t LaunchSdotKernel(
    _aclblas_handle* h, const SdotTilingData& tiling, const float* x, const float* y, float* result)
{
    size_t workspaceNeed = tiling.useCoreNum * sizeof(float);
    CHECK_RET(workspaceNeed <= GetEffectiveWorkspaceSize(h),
              OP_LOGE("aclblasSdot", "workspace %zu > handle %zu", workspaceNeed, GetEffectiveWorkspaceSize(h));
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    OP_LOGI("aclblasSdot", "launching kernel with %u cores", tiling.useCoreNum);

    const aclError launchErr = sdot_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(const_cast<float*>(y)),
        reinterpret_cast<uint8_t*>(result), workspaceDevice, tiling.useCoreNum, tiling, h->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasSdot", "sdot_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public API: aclblasSdot
// ---------------------------------------------------------------------------
// 签名与 include/cann_ops_blas.h:360 公共声明（int n/incx/incy）对齐。此前实现误用
// int64_t 形参，C++ 名 mangling 下定义的是另一符号，导致头文件声明的 aclblasSdot
// 无定义（arch35 --ops=dot 构建链接失败：undefined reference to `aclblasSdot'）。
aclblasStatus_t aclblasSdot(
    aclblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSdot", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        if (result != nullptr) {
            aclError ret = aclrtMemset(result, sizeof(float), 0, sizeof(float));
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasSdot", "aclrtMemset failed, ret=%d", ret);
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t status = ValidateSdotParams(incx, incy, x, y, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    auto* h = handle;

    uint32_t vectorCoreNum = GetAivCoreCount();
    if (vectorCoreNum == 0) {
        OP_LOGE("aclblasSdot", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    // kernel 侧 n/incx/incy 均为 int64 tiling 域，int 入参安全升宽；
    // #1081 的 uint32 n 越界拒绝在 int 公共 API 下不可达（n<=INT_MAX），随签名对齐移除
    SdotTilingData tiling = CalSdotTilingData(n, incx, incy, vectorCoreNum);

    OP_LOGD(
        "aclblasSdot", "tiling: n=%ld incx=%ld incy=%ld useCoreNum=%u", tiling.n, tiling.incx, tiling.incy,
        tiling.useCoreNum);

    return LaunchSdotKernel(h, tiling, x, y, result);
}
