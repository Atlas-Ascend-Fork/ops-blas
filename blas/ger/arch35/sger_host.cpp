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
#include "sger_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

aclError sger_arch35_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* A, const SgerTilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateSgerParams(
    int m, int n, int lda, int incx, int incy, const float* alpha, const float* x, const float* y, const float* A)
{
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSger", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    if (m > 0 && n > 0) {
        CHECK_RET(x != nullptr, OP_LOGE("aclblasSger", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(y != nullptr, OP_LOGE("aclblasSger", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(A != nullptr, OP_LOGE("aclblasSger", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(lda >= std::max(1, m), OP_LOGE("aclblasSger", "lda must be >= max(1,m), got lda=%d, m=%d", lda, m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSger", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSger", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE("aclblasSger", "incx must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != INT_MIN, OP_LOGE("aclblasSger", "incy must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SgerTilingData CalSgerTilingData(uint32_t useNumBlocks, int m, int n, int lda, float alpha, int incx, int incy)
{
    SgerTilingData tilingData{};
    uint32_t colsPerBlock = CeilDiv<uint32_t>(static_cast<uint32_t>(n), useNumBlocks);
    tilingData.numThreads = std::min(CeilAlign<uint32_t>(colsPerBlock, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.colsPerBlock = colsPerBlock;
    tilingData.m = static_cast<uint32_t>(m);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.alpha = alpha;
    tilingData.incx = incx;
    tilingData.incy = incy;
    return tilingData;
}

static aclblasStatus_t LaunchSgerKernel(
    _aclblas_handle* h, int m, int n, int lda, float alphaVal, int incx, int incy, const float* x, const float* y,
    float* A)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSger", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM), aivCoreNum);
    if (useNumBlocks == 0) {
        useNumBlocks = 1;
    }
    SgerTilingData tiling = CalSgerTilingData(useNumBlocks, m, n, lda, alphaVal, incx, incy);
    OP_LOGD(
        "aclblasSger",
        "tiling: m=%lu n=%lu lda=%lu incx=%d incy=%d numThreads=%u colsPerBlock=%u numBlocks=%u alpha=%f", tiling.m,
        tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.numThreads, tiling.colsPerBlock, useNumBlocks,
        tiling.alpha);
    OP_LOGI(
        "aclblasSger", "launching kernel: blocks=%u, cores=%u, path=%s", useNumBlocks, aivCoreNum,
        (incx == 1 && static_cast<uint64_t>(m) <= UB_X_FLOATS) ? "UB-x" : "GM");
    const aclError launchErr = sger_arch35_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(const_cast<float*>(y)),
        reinterpret_cast<uint8_t*>(A), tiling, useNumBlocks, h->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasSger", "sger_arch35_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSger(
    aclblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy,
    float* A, int lda)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSger", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(m >= 0, OP_LOGE("aclblasSger", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSger", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateSgerParams(m, n, lda, incx, incy, alpha, x, y, A);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    float alphaVal = *alpha;
    if (alphaVal == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = handle;
    return LaunchSgerKernel(h, m, n, lda, alphaVal, incx, incy, x, y, A);
}
