/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "icamax_kernel.h"
#include "log/log.h"

namespace {

constexpr uint32_t WORKSPACE_WORDS_PER_CORE = 2U;
constexpr uint32_t WORKSPACE_ALIGN_WORDS = 64U;

static aclblasStatus_t ValidateIcamaxParams(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result, bool& quickReturn)
{
    quickReturn = false;
    if (handle == nullptr) {
        OP_LOGE("aclblasIcamax", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasIcamax", "n must be non-negative, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasIcamax", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || incx < 1) {
        quickReturn = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasIcamax", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t WriteIcamaxQuickReturn(aclblasHandle_t handle, int* result)
{
    const aclError status = aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream);
    if (status != ACL_SUCCESS) {
        OP_LOGE("aclblasIcamax", "failed to write quick-return result, ret=%d", status);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t CalcIcamaxTiling(int n, uint32_t numBlocks, int incx, IcamaxTilingData& tiling)
{
    if (numBlocks == 0U) {
        OP_LOGE("aclblasIcamax", "numBlocks must be positive");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    tiling.totalN = static_cast<uint32_t>(n);
    if (incx == 1 && tiling.totalN >= numBlocks * ICAMAX_SIMD_VECTOR_COMPLEX_COUNT) {
        const uint32_t vectorGroupCount = tiling.totalN / ICAMAX_SIMD_VECTOR_COMPLEX_COUNT;
        tiling.perCoreN = vectorGroupCount / numBlocks * ICAMAX_SIMD_VECTOR_COMPLEX_COUNT;
        tiling.remainder = vectorGroupCount % numBlocks;
        tiling.vectorAligned = 1U;
        tiling.scalarTail = tiling.totalN % ICAMAX_SIMD_VECTOR_COMPLEX_COUNT;
    } else {
        tiling.perCoreN = tiling.totalN / numBlocks;
        tiling.remainder = tiling.totalN % numBlocks;
    }
    tiling.useCoreNum = numBlocks;
    tiling.tileSize = ICAMAX_SIMD_TILE_COMPLEX_COUNT;
    tiling.incx = static_cast<uint32_t>(incx);
    const uint32_t remainderUnit = tiling.vectorAligned != 0U ? ICAMAX_SIMD_VECTOR_COMPLEX_COUNT : 1U;
    const uint32_t maxCoreN = tiling.perCoreN + (tiling.remainder == 0U ? 0U : remainderUnit) + tiling.scalarTail;
    tiling.nthreads = std::min(CeilAlign<uint32_t>(maxCoreN, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchIcamaxKernel(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE("aclblasIcamax", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    IcamaxTilingData tiling{};
    const aclblasStatus_t tilingStatus = CalcIcamaxTiling(n, numBlocks, incx, tiling);
    if (tilingStatus != ACLBLAS_STATUS_SUCCESS) {
        return tilingStatus;
    }

    const uint32_t workspaceWords = numBlocks * WORKSPACE_WORDS_PER_CORE;
    const uint32_t alignedWords =
        (workspaceWords + WORKSPACE_ALIGN_WORDS - 1U) / WORKSPACE_ALIGN_WORDS * WORKSPACE_ALIGN_WORDS;
    const size_t requiredWorkspace = static_cast<size_t>(alignedWords) * sizeof(uint32_t);
    const size_t workspaceSize = GetEffectiveWorkspaceSize(handle);
    if (workspaceSize < requiredWorkspace) {
        OP_LOGE("aclblasIcamax", "workspace too small: need %zu, got %zu", requiredWorkspace, workspaceSize);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    OP_LOGD(
        "aclblasIcamax",
        "tiling: totalN=%u perCoreN=%u remainder=%u useCoreNum=%u tileSize=%u nthreads=%u incx=%u "
        "vectorAligned=%u scalarTail=%u",
        tiling.totalN, tiling.perCoreN, tiling.remainder, tiling.useCoreNum, tiling.tileSize, tiling.nthreads,
        tiling.incx, tiling.vectorAligned, tiling.scalarTail);
    OP_LOGI("aclblasIcamax", "launching kernel: blocks=%u, nthreads=%u, incx=%d", numBlocks, tiling.nthreads, incx);

    icamax_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(x)), reinterpret_cast<GM_ADDR>(result),
        reinterpret_cast<GM_ADDR>(GetEffectiveWorkspace(handle)), tiling, numBlocks,
        reinterpret_cast<void*>(handle->stream));
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasIcamax(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    bool quickReturn = false;
    const aclblasStatus_t status = ValidateIcamaxParams(handle, n, x, incx, result, quickReturn);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (quickReturn) {
        return WriteIcamaxQuickReturn(handle, result);
    }
    return LaunchIcamaxKernel(handle, n, x, incx, result);
}
