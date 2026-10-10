/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Single-precision complex conjugate dot product for Arch35 (Ascend 950).

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "cdotc_kernel.h"

namespace {

constexpr uint32_t WORKSPACE_ALIGNMENT_FLOATS = 8;

static uint32_t AlignWorkspace(uint32_t value)
{
    return (value + WORKSPACE_ALIGNMENT_FLOATS - 1) / WORKSPACE_ALIGNMENT_FLOATS * WORKSPACE_ALIGNMENT_FLOATS;
}

static aclblasStatus_t ValidateCdotcParams(
    int incx, int incy, const aclblasComplex* x, const aclblasComplex* y, const aclblasComplex* result)
{
    if (incx == 0) {
        OP_LOGE("aclblasCdotc", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasCdotc", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CHECK_RET(x != nullptr, OP_LOGE("aclblasCdotc", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasCdotc", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(result != nullptr, OP_LOGE("aclblasCdotc", "result must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static CdotcTilingData CalCdotcTilingData(int n, int incx, int incy, uint32_t vectorCoreNum)
{
    CdotcTilingData tiling{};
    tiling.n = n;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.useCoreNum = std::min(std::min(vectorCoreNum, static_cast<uint32_t>(n)), CDOTC_MAX_PARALLEL_CORE_NUM);
    return tiling;
}

static aclblasStatus_t LaunchCdotcKernel(
    _aclblas_handle* h, const CdotcTilingData& tiling, const aclblasComplex* x, const aclblasComplex* y,
    aclblasComplex* result)
{
    const uint32_t workspaceStride = AlignWorkspace(tiling.useCoreNum);
    const size_t workspaceNeed = static_cast<size_t>(2) * workspaceStride * sizeof(float);
    const size_t workspaceSize = GetEffectiveWorkspaceSize(h);
    if (workspaceNeed > workspaceSize) {
        OP_LOGE("aclblasCdotc", "workspace %zu > handle %zu", workspaceNeed, workspaceSize);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    CHECK_RET(workspaceDevice != nullptr, OP_LOGE("aclblasCdotc", "effective workspace is nullptr");
              return ACLBLAS_STATUS_EXECUTION_FAILED);

    OP_LOGD("aclblasCdotc", "launching packed RegBase kernels, cores=%u", tiling.useCoreNum);
    CdotcKernelDo(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(result), workspaceDevice,
        tiling.useCoreNum, tiling, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCdotc(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    aclblasComplex* result)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCdotc", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasCdotc", "n must not be negative, n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);

    if (n == 0) {
        CHECK_RET(result != nullptr, OP_LOGE("aclblasCdotc", "result must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        aclError ret = aclrtMemsetAsync(result, sizeof(aclblasComplex), 0, sizeof(aclblasComplex), handle->stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCdotc", "aclrtMemsetAsync failed, ret=%d", ret);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t status = ValidateCdotcParams(incx, incy, x, y, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    const uint32_t vectorCoreNum = GetAivCoreCount();
    if (vectorCoreNum == 0) {
        OP_LOGE("aclblasCdotc", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const CdotcTilingData tiling = CalCdotcTilingData(n, incx, incy, vectorCoreNum);
    OP_LOGD(
        "aclblasCdotc", "tiling: n=%ld incx=%ld incy=%ld useCoreNum=%u", tiling.n, tiling.incx, tiling.incy,
        tiling.useCoreNum);
    return LaunchCdotcKernel(handle, tiling, x, y, result);
}
