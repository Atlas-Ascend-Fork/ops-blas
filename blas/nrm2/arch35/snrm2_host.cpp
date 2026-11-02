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
 * \file nrm2_host.cpp
 * \brief aclblasSnrm2 Host-side dispatch for ascend950 (arch35).
 *        Dual-path: SIMD membase for incx==1, SIMT for incx!=1.
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "snrm2_kernel.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

static aclblasStatus_t ValidateSnrm2Params(aclblasHandle_t handle, int n, int incx, const float* x, float* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSnrm2", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasSnrm2", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if ((n > 0 && incx > 0) && x == nullptr) {
        OP_LOGE("aclblasSnrm2", "x must not be nullptr when n > 0 and incx > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t CalcSnrm2LaunchConfig(
    int n, int incx, uint32_t aivCoreNum, uint32_t* numBlocks, uint32_t* nthreads)
{
    *numBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    if (*numBlocks > SNRM2_MAX_CORE_NUM) {
        *numBlocks = SNRM2_MAX_CORE_NUM;
    }

    if (incx != 1) {
        *nthreads = std::min(
            CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), *numBlocks), SIMT_MIN_THREAD_NUM),
            SIMT_MAX_THREAD_NUM);
    } else {
        *nthreads = 0;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static Snrm2TilingData CalcSnrm2TilingData(int n, int incx, uint32_t numBlocks, uint32_t nthreads)
{
    Snrm2TilingData tiling{};
    tiling.n = n;
    tiling.incx = incx;
    uint32_t blocks = (numBlocks == 0) ? 1 : numBlocks;
    tiling.useCoreNum = blocks;
    tiling.perCoreN = static_cast<uint32_t>(n) / blocks;
    tiling.remainder = static_cast<uint32_t>(n) % blocks;
    tiling.nthreads = nthreads;
    return tiling;
}

static aclblasStatus_t Snrm2ExecuteKernel(
    const float* x, float* result, uint32_t numBlocks, const Snrm2TilingData& tiling, void* workspace,
    aclrtStream stream)
{
    OP_LOGI("aclblasSnrm2", "launching kernel: blocks=%u, incx=%d, workspace=%p", numBlocks, tiling.incx, workspace);

    snrm2_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(result),
        reinterpret_cast<uint8_t*>(workspace), tiling, numBlocks, stream);

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasSnrm2(aclblasHandle_t handle, int n, const float* x, int incx, float* result)
{
    OP_LOGD("aclblasSnrm2", "entry: n=%d, incx=%d", n, incx);

    aclblasStatus_t status = ValidateSnrm2Params(handle, n, incx, x, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Quick return: n <= 0 or incx <= 0 -- write 0.0f to device result
    if (n <= 0 || incx <= 0) {
        float zero = 0.0f;
        aclError memRet = aclrtMemcpy(result, sizeof(float), &zero, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (memRet != ACL_SUCCESS) {
            OP_LOGE("aclblasSnrm2", "aclrtMemcpy for early-return zero failed: %d", memRet);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = handle;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSnrm2", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t numBlocks = 0;
    uint32_t nthreads = 0;
    status = CalcSnrm2LaunchConfig(n, incx, aivCoreNum, &numBlocks, &nthreads);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    Snrm2TilingData tiling = CalcSnrm2TilingData(n, incx, numBlocks, nthreads);

    OP_LOGD(
        "aclblasSnrm2",
        "tiling: n=%d incx=%d useCoreNum=%u perCoreN=%u "
        "remainder=%u nthreads=%u",
        tiling.n, tiling.incx, tiling.useCoreNum, tiling.perCoreN, tiling.remainder, tiling.nthreads);

    // Reduce rounds the partial-sum count up to one UB block; size the check to that
    // padded count so the host-side memset and the kernel's reduce reads stay within
    // the validated region.
    uint32_t paddedCoreNum = (tiling.useCoreNum + SNRM2_WORKSPACE_ALIGN_FLOATS - 1) / SNRM2_WORKSPACE_ALIGN_FLOATS *
                             SNRM2_WORKSPACE_ALIGN_FLOATS;
    size_t requiredBytes = static_cast<size_t>(paddedCoreNum) * sizeof(float);
    CHECK_RET(requiredBytes <= GetEffectiveWorkspaceSize(h),
              OP_LOGE("aclblasSnrm2", "workspace %zu > handle %zu", requiredBytes, GetEffectiveWorkspaceSize(h));
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    void* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    // Zero-init workspace on the host side: compute cores write only their own
    // workspace[blockIdx] slot, but the reduce kernel sums paddedCount (useCoreNum
    // rounded up to one UB block), so the unwritten tail slots must read as zero.
    // Done here rather than inside kernel_do so memset failures propagate via the
    // aclblasStatus_t return value instead of being silently swallowed.
    aclError memsetRet = aclrtMemsetAsync(workspace, requiredBytes, 0, requiredBytes, h->stream);
    if (memsetRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSnrm2", "aclrtMemsetAsync workspace failed: %d", memsetRet);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    status = Snrm2ExecuteKernel(x, result, numBlocks, tiling, workspace, h->stream);
    OP_LOGD("aclblasSnrm2", "exit: status=%d", static_cast<int>(status));
    return status;
}
