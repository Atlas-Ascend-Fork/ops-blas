/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sgbmv_host.cpp
 * \brief single-precision gbmv host-side implementation
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "sgbmv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

void sgbmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgbmvTilingData& tiling, uint32_t numBlocks,
    void* stream);

static aclblasStatus_t ValidateSgbmvParams(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, const float* alpha,
    const float* beta, const float* A, const float* x, const float* y)
{
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasSgbmv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(kl >= 0 && ku >= 0, OP_LOGE("aclblasSgbmv", "invalid kl=%d or ku=%d", kl, ku);
              return ACLBLAS_STATUS_INVALID_VALUE);
    // issue #739: int 加法在 kl+ku 极大时回绕为负导致 lda 校验穿透；对齐 cgbmv 用 int64 中间量
    CHECK_RET(static_cast<int64_t>(lda) >= static_cast<int64_t>(kl) + ku + 1,
              OP_LOGE("aclblasSgbmv", "invalid lda=%d, kl=%d, ku=%d", lda, kl, ku);
              return ACLBLAS_STATUS_INVALID_VALUE);
    // issue #739: kl/ku 超过矩阵半带宽上限时 tiling 按虚假带宽下发，kernel 访问 A 越界；
    // 对齐 cgbmv（ValidateCgbmvParams）的上界检查
    CHECK_RET(kl <= m - 1 && ku <= n - 1,
              OP_LOGE("aclblasSgbmv", "kl=%d must be <= m-1=%d, ku=%d must be <= n-1=%d", kl, m - 1, ku, n - 1);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSgbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSgbmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSgbmv", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasSgbmv", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasSgbmv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSgbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSgbmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SgbmvTilingData CalSgbmvTilingData(
    uint32_t useNumBlocks, int m, int n, int kl, int ku, int lda, aclblasOperation_t trans, float alpha, float beta,
    int incx, int incy)
{
    SgbmvTilingData tilingData{};
    bool isTransT = (trans != ACLBLAS_OP_N);
    uint32_t outDim = isTransT ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    tilingData.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(outDim, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.rowsPerBlock =
        static_cast<uint32_t>((outDim + static_cast<int>(useNumBlocks) - 1) / static_cast<int>(useNumBlocks));
    tilingData.m = static_cast<uint32_t>(m);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.kl = static_cast<uint32_t>(kl);
    tilingData.ku = static_cast<uint32_t>(ku);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.trans = isTransT ? 1U : 0U;
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);

    return tilingData;
}

aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A,
    int lda, const float* x, int incx, const float* beta, float* y, int incy)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSgbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(m >= 0 && n >= 0, OP_LOGE("aclblasSgbmv", "invalid m=%d or n=%d", m, n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateSgbmvParams(trans, m, n, kl, ku, lda, incx, incy, alpha, beta, A, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgbmv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    bool isTransT = (trans != ACLBLAS_OP_N);
    uint32_t outDim = isTransT ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(outDim, SIMT_MIN_THREAD_NUM), aivCoreNum);
    SgbmvTilingData tiling = CalSgbmvTilingData(useNumBlocks, m, n, kl, ku, lda, trans, *alpha, *beta, incx, incy);

    OP_LOGD(
        "aclblasSgbmv", "tiling: m=%u n=%u kl=%u ku=%u lda=%u trans=%u numBlocks=%u numThreads=%u", tiling.m, tiling.n,
        tiling.kl, tiling.ku, tiling.lda, tiling.trans, useNumBlocks, tiling.numThreads);
    OP_LOGI("aclblasSgbmv", "launching kernel");

    sgbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(y), nullptr, tiling, useNumBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
