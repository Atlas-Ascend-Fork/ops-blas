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
 * \file sspmv_host.cpp
 * \brief SSPMV Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "sspmv_tiling_data.h"

void sspmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks, const SspmvTilingData& tiling,
    void* stream);

static aclblasStatus_t ValidateSspmvParams(
    aclblasFillMode_t uplo, int incx, int incy, const float* alpha, const float* beta, const float* ap, const float* x,
    const float* y)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasSspmv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSspmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSspmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSspmv", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasSspmv", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ap != nullptr, OP_LOGE("aclblasSspmv", "ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSspmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSspmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SspmvTilingData CalSspmvTilingData(
    uint32_t useNumBlocks, int n, aclblasFillMode_t uplo, float alpha, float beta, int incx, int incy)
{
    SspmvTilingData tilingData{};
    tilingData.nthreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);
    return tilingData;
}

aclblasStatus_t aclblasSspmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* ap, const float* x,
    int incx, const float* beta, float* y, int incy)
{
    // issue #765: handle 判空先于参数校验，对齐 api_list.md 契约与 stpttr（#722）口径
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSspmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSspmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateSspmvParams(uplo, incx, incy, alpha, beta, ap, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSspmv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);

    SspmvTilingData tiling = CalSspmvTilingData(useNumBlocks, n, uplo, *alpha, *beta, incx, incy);

    OP_LOGD(
        "aclblasSspmv", "tiling: n=%u uplo=%u nthreads=%u numBlocks=%u", tiling.n, tiling.uplo, tiling.nthreads,
        useNumBlocks);
    OP_LOGI("aclblasSspmv", "launching kernel");

    sspmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(ap)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(y), nullptr, useNumBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
