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
 * \file stpmv_host.cpp
 * \brief Single-precision tpmv host-side implementation.
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "stpmv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

void stpmv_arch35_kernel_do(
    uint8_t* aP, uint8_t* x, uint8_t* y, const StpmvTilingData& tiling, uint32_t numBlocks, void* stream);

void stpmv_arch35_scatter_do(
    uint8_t* dst, uint8_t* src, const StpmvTilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateStpmvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, const float* AP, float* x, int incx)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasStpmv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasStpmv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
              OP_LOGE("aclblasStpmv", "invalid diag=%d", static_cast<int>(diag));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasStpmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(AP != nullptr, OP_LOGE("aclblasStpmv", "AP must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasStpmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static StpmvTilingData CalStpmvTilingData(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t nU32, int incx,
    uint32_t numThreads)
{
    StpmvTilingData tiling{};
    tiling.n = nU32;
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = incx;
    tiling.numThreads = numThreads;
    return tiling;
}

static aclblasStatus_t AllocateWorkspace(int n, uint8_t** workspaceOut)
{
    *workspaceOut = nullptr;
    const size_t workspaceSize = static_cast<size_t>(n) * sizeof(float);
    aclError aclRet = aclrtMalloc(reinterpret_cast<void**>(workspaceOut), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, OP_LOGE("aclblasStpmv", "aclrtMalloc failed, ret=%d", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ExecuteKernel(
    const float* AP, float* x, uint8_t* workspace, const StpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    stpmv_arch35_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(AP)), reinterpret_cast<uint8_t*>(x), workspace, tiling, numBlocks,
        stream);

    aclError aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, OP_LOGE("aclblasStpmv", "aclrtSynchronizeStream failed, ret=%d", aclRet);
              return ACLBLAS_STATUS_INTERNAL_ERROR);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t WriteBackResult(
    float* x, uint8_t* workspace, int n, int incx, const StpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    const size_t workspaceSize = static_cast<size_t>(n) * sizeof(float);
    if (incx == 1) {
        aclError aclRet = aclrtMemcpy(x, workspaceSize, workspace, workspaceSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS,
                  OP_LOGE("aclblasStpmv", "aclrtMemcpy D2D (workspace->x) failed, ret=%d", aclRet);
                  return ACLBLAS_STATUS_INTERNAL_ERROR);
    } else {
        stpmv_arch35_scatter_do(reinterpret_cast<uint8_t*>(x), workspace, tiling, numBlocks, stream);
        aclError aclRet = aclrtSynchronizeStream(stream);
        CHECK_RET(aclRet == ACL_SUCCESS, OP_LOGE("aclblasStpmv", "scatter sync failed, ret=%d", aclRet);
                  return ACLBLAS_STATUS_INTERNAL_ERROR);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasStpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasStpmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasStpmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    // issue #748: 校验顺序对齐 ctpmv 契约（枚举/incx 校验先于 n==0 快速返回），
    // 使 n=0 且 incx=0 与 ctpmv 同返回 INVALID_VALUE
    aclblasStatus_t st = ValidateStpmvParams(uplo, trans, diag, AP, x, incx);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0, OP_LOGE("aclblasStpmv", "GetAivCoreCount failed");
              return ACLBLAS_STATUS_EXECUTION_FAILED);

    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t useNumBlocks = std::min(nU32, aivCoreNum);
    useNumBlocks = std::max<uint32_t>(useNumBlocks, 1);
    uint32_t numThreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(nU32, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);

    StpmvTilingData tiling = CalStpmvTilingData(uplo, trans, diag, nU32, incx, numThreads);

    OP_LOGD(
        "aclblasStpmv", "tiling: n=%u uplo=%u trans=%u diag=%u incx=%lld numThreads=%u numBlocks=%u", tiling.n,
        tiling.uplo, tiling.trans, tiling.diag, tiling.incx, tiling.numThreads, useNumBlocks);
    OP_LOGI("aclblasStpmv", "launching kernel");

    uint8_t* workspaceDevice = nullptr;
    st = AllocateWorkspace(n, &workspaceDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    st = ExecuteKernel(AP, x, workspaceDevice, tiling, useNumBlocks, h->stream);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(workspaceDevice);
        return st;
    }

    st = WriteBackResult(x, workspaceDevice, n, incx, tiling, useNumBlocks, h->stream);
    aclrtFree(workspaceDevice);
    return st;
}
