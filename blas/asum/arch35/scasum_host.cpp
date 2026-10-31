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
 * \file scasum_host.cpp
 * \brief scasum Host-side dispatch for ascend950 (complex64 asum)
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "scasum_tiling_data.h"

void scasum_kernel_do(
    uint8_t* inGM, uint8_t* outGM, uint8_t* workSpace, const ScasumTilingData& tiling, uint32_t numBlocks,
    void* stream);

namespace {

// AIV kernel handles 32B-alignment via maxDataCount tiling (float view);
// SIMT kernel accesses elements individually — no block alignment needed.
// R4: per-core ranges are described by total/perCore/remainder; each block
// derives its own offset/count kernel-side (offset(i) = i * perCoreN +
// min(i, remainder), count(i) = perCoreN + (i < remainder ? 1 : 0)),
// which is the same distribution the host previously serialised into
// per-core arrays.
static ScasumTilingData CalcScasumTilingData(int64_t totalComplexNum, uint32_t vecCoreNum)
{
    ScasumTilingData tiling;
    tiling.n = totalComplexNum;
    tiling.useCoreNum = 0;

    uint32_t useCoreNum = std::min(vecCoreNum, static_cast<uint32_t>(totalComplexNum));
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    tiling.useCoreNum = useCoreNum;
    tiling.perCoreN = static_cast<uint32_t>(totalComplexNum) / useCoreNum;
    tiling.remainder = static_cast<uint32_t>(totalComplexNum) % useCoreNum;

    return tiling;
}

static aclblasStatus_t ValidateScasumParams(
    aclblasHandle_t handle, int n, int incx, const aclblasComplex* x, float* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasScasum", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasScasum", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && incx > 0 && x == nullptr) {
        OP_LOGE("aclblasScasum", "x must not be nullptr when n > 0 and incx > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t CalcScasumLaunchConfig(int n, int incx, uint32_t* numBlocks, uint32_t* nthreads)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasScasum", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    *numBlocks = (static_cast<uint32_t>(n) < aivCoreNum) ? static_cast<uint32_t>(n) : aivCoreNum;
    if (*numBlocks > SCASUM_MAX_CORE_NUM) {
        *numBlocks = SCASUM_MAX_CORE_NUM;
    }
    if (*numBlocks == 0) {
        *numBlocks = 1;
    }

    // More threads help hide RVECLD memory latency for large vectors, but the
    // per-thread work becomes too small (thread launch dominates) for small n.
    uint32_t maxThreads = (static_cast<uint32_t>(n) >= 2097152) ? 2048u : 1024u;
    *nthreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), *numBlocks), SIMT_MIN_THREAD_NUM), maxThreads);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ScasumExecuteKernel(
    _aclblas_handle* h, const aclblasComplex* x, float* result, uint32_t numBlocks, const ScasumTilingData& tiling)
{
    // SIMT single-kernel path always needs workspace for per-core partials.
    size_t workspaceBytes = static_cast<size_t>(numBlocks) * sizeof(float);
    CHECK_RET(workspaceBytes <= GetEffectiveWorkspaceSize(h),
              OP_LOGE("aclblasScasum", "workspace %zu > handle %zu", workspaceBytes, GetEffectiveWorkspaceSize(h));
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    scasum_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(result), workspaceDevice,
        tiling, numBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result)
{
    aclblasStatus_t status = ValidateScasumParams(handle, n, incx, x, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Match cuBLAS semantics: n <= 0 or incx <= 0 produces zero. result is a
    // device output for this API, so do not dereference it from host code.
    if (n <= 0 || incx <= 0) {
        float zero = 0.0f;
        aclError memRet = aclrtMemcpy(result, sizeof(float), &zero, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (memRet != ACL_SUCCESS) {
            OP_LOGE("aclblasScasum", "aclrtMemcpy for early-return zero failed: %d", memRet);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t numBlocks;
    uint32_t nthreads;
    status = CalcScasumLaunchConfig(n, incx, &numBlocks, &nthreads);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    ScasumTilingData tiling = CalcScasumTilingData(n, numBlocks);
    tiling.incx = incx;
    tiling.nthreads = nthreads;

    OP_LOGD(
        "aclblasScasum", "tiling: n=%ld incx=%ld useCoreNum=%u numBlocks=%u nthreads=%u", tiling.n, tiling.incx,
        tiling.useCoreNum, numBlocks, tiling.nthreads);

    return ScasumExecuteKernel(handle, x, result, numBlocks, tiling);
}
