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
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "chpr_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static aclblasStatus_t ValidateChprParams(
    aclblasFillMode_t uplo, int incx, const float* alpha, const aclblasComplex* x, const aclblasComplex* ap)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasChpr", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0 && incx != INT_MIN,
              OP_LOGE("aclblasChpr", "incx must be nonzero and not INT_MIN, got %d", incx);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasChpr", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasChpr", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ap != nullptr, OP_LOGE("aclblasChpr", "ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// issue #849: 对齐 csyr/cher 的 Host/Device 双支持口径（同 ssyr/sspr 的
// ReadHostScalarFromDevice 模式——tiling 携带 host 标量）：Host 指针直接读取；
// Device 指针经 D2H 回读，避免对设备地址做 Host 解引用的非法访问/偶发崩溃。
static aclblasStatus_t ResolveChprAlpha(aclblasHandle_t handle, const float* alpha, float& alphaVal, bool& quickReturn)
{
    bool alphaIsDevice = false;
    aclblasStatus_t st = CheckPtrLocation(alpha, &alphaIsDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (!alphaIsDevice) {
        alphaVal = *alpha;
        quickReturn = (alphaVal == 0.0f);
        return ACLBLAS_STATUS_SUCCESS;
    }
    st = ReadHostScalarFromDevice(alpha, alphaVal, handle->stream, "aclblasChpr");
    if (st == ACLBLAS_STATUS_SUCCESS) {
        quickReturn = (alphaVal == 0.0f);
    }
    return st;
}

static aclblasStatus_t LaunchChprKernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, float alpha, const aclblasComplex* x, int incx,
    aclblasComplex* ap)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum != 0, OP_LOGE("aclblasChpr", "vector core count is zero");
              return ACLBLAS_STATUS_EXECUTION_FAILED);

    // Each block updates four interleaved columns; neighbouring lanes update consecutive packed rows.
    constexpr uint32_t COLUMNS_PER_BLOCK = 4U;
    ChprTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.useCoreNum = std::min((tiling.n + COLUMNS_PER_BLOCK - 1U) / COLUMNS_PER_BLOCK, aivCoreNum);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.alpha = alpha;
    tiling.incx = static_cast<int64_t>(incx);

    chpr_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(ap), tiling,
        tiling.useCoreNum, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* ap)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasChpr", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasChpr", "n must be nonnegative, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t status = ValidateChprParams(uplo, incx, alpha, x, ap);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    // issue #849: alpha 位置经 CheckPtrLocation 判定，Device 指针 D2H 回读；
    // 零系数快速返回判定同样基于回读值。
    float alphaValue = 0.0f;
    bool alphaQuickReturn = false;
    const aclblasStatus_t alphaStatus = ResolveChprAlpha(handle, alpha, alphaValue, alphaQuickReturn);
    if (alphaStatus != ACLBLAS_STATUS_SUCCESS) {
        return alphaStatus;
    }
    if (alphaQuickReturn) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return LaunchChprKernel(handle, uplo, n, alphaValue, x, incx, ap);
}
