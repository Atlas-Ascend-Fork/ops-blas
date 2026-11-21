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
#include <limits>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "sdot_tiling_data.h"

void sdot_kernel_do(uint8_t* x, uint8_t* y, uint8_t* result, uint8_t* workSpace,
                    uint32_t numBlocks, const SdotTilingData& tiling, void *stream);

// ---------------------------------------------------------------------------
// Parameter validation
// ---------------------------------------------------------------------------
static aclblasStatus_t ValidateSdotParams(
    int64_t incx, int64_t incy, const float* x, const float* y, const float* result)
{
    CHECK_RET(incx != 0, OP_LOGE("aclblasSdot", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSdot", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSdot", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSdot", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        result != nullptr, OP_LOGE("aclblasSdot", "result must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
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
    CHECK_RET(
        workspaceNeed <= GetEffectiveWorkspaceSize(h),
        OP_LOGE("aclblasSdot", "workspace %zu > handle %zu", workspaceNeed, GetEffectiveWorkspaceSize(h));
        return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    OP_LOGI("aclblasSdot", "launching kernel with %u cores", tiling.useCoreNum);

    sdot_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(const_cast<float*>(y)),
        reinterpret_cast<uint8_t*>(result), workspaceDevice, tiling.useCoreNum, tiling, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Public API: aclblasSdot
// ---------------------------------------------------------------------------
aclblasStatus_t aclblasSdot(
    aclblasHandle_t handle, const int64_t n, const float* x, const int64_t incx, const float* y, const int64_t incy,
    float* result)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSdot", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
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

    // issue #1081: 当前实现将 n 下发为 uint32_t（sdot_kernel.cpp），超出范围时
    // 拒绝继续执行，对齐 srotm 口径；否则 int64 n 静默截断导致 SUCCESS 下结果错误
    CHECK_RET(n <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
              OP_LOGE("aclblasSdot", "n=%ld exceeds uint32 range", static_cast<long>(n));
              return ACLBLAS_STATUS_NOT_SUPPORTED);

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

    SdotTilingData tiling = CalSdotTilingData(n, incx, incy, vectorCoreNum);

    OP_LOGD(
        "aclblasSdot", "tiling: n=%ld incx=%ld incy=%ld useCoreNum=%u", tiling.n, tiling.incx, tiling.incy,
        tiling.useCoreNum);

    return LaunchSdotKernel(h, tiling, x, y, result);
}
