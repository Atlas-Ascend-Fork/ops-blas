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
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "ssbmv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

void ssbmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks, const SsbmvTilingData& tiling,
    void* stream);

static aclblasStatus_t ValidateSbmvParams(
    aclblasFillMode_t uplo, int k, int lda, int incx, int incy, const float* alpha, const float* beta, const float* A,
    const float* x, const float* y)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasSsbmv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasSsbmv", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    // 写成 lda > k 而不是 lda >= k + 1：k == INT_MAX 时 k + 1 在 int 域溢出（UB）；int 域
    // lda 不可能大于 INT_MAX，该比较等价地拒绝这种不存在合法 lda 的带宽场景。
    CHECK_RET(lda > k, OP_LOGE("aclblasSsbmv", "invalid lda=%d, k=%d", lda, k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSsbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSsbmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSsbmv", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasSsbmv", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasSsbmv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSsbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSsbmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SsbmvTilingData CalSsbmvTilingData(
    uint32_t useNumBlocks, int n, int k, int lda, aclblasFillMode_t uplo, float alpha, float beta, int incx, int incy)
{
    SsbmvTilingData tilingData{};
    tilingData.numThreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.numBlocks = useNumBlocks;
    tilingData.rowsPerBlock =
        static_cast<uint32_t>((n + static_cast<int>(useNumBlocks) - 1) / static_cast<int>(useNumBlocks));
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.k = static_cast<uint32_t>(k);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);

    return tilingData;
}

aclblasStatus_t aclblasSsbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    CHECK_RET(n >= 0, OP_LOGE("aclblasSsbmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSsbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateSbmvParams(uplo, k, lda, incx, incy, alpha, beta, A, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSsbmv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);

    SsbmvTilingData tiling = CalSsbmvTilingData(useNumBlocks, n, k, lda, uplo, *alpha, *beta, incx, incy);

    OP_LOGD(
        "aclblasSsbmv", "tiling: n=%u k=%u lda=%u uplo=%u numBlocks=%u numThreads=%u", tiling.n, tiling.k, tiling.lda,
        tiling.uplo, tiling.numBlocks, tiling.numThreads);
    OP_LOGI("aclblasSsbmv", "launching kernel");

    ssbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(y), nullptr, useNumBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
