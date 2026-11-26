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
#include "syr2_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

struct Syr2TilingData;

aclError syr2_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* A, const Syr2TilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateSyr2Params(
    aclblasFillMode_t uplo, int n, int lda, int incx, int incy, const float* alpha, const float* x, const float* y,
    const float* A)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasSsyr2", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda >= std::max(1, n), OP_LOGE("aclblasSsyr2", "invalid lda=%d, n=%d", lda, n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSsyr2", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSsyr2", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSsyr2", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSsyr2", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSsyr2", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasSsyr2", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static Syr2TilingData CalSyr2TilingData(
    uint32_t useNumBlocks, int n, int lda, aclblasFillMode_t uplo, float alpha, int incx, int incy)
{
    Syr2TilingData tilingData{};
    tilingData.numThreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.rowsPerBlock =
        static_cast<uint32_t>((n + static_cast<int>(useNumBlocks) - 1) / static_cast<int>(useNumBlocks));
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);

    return tilingData;
}

aclblasStatus_t aclblasSsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y,
    int incy, float* A, int lda)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSsyr2", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasSsyr2", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateSyr2Params(uplo, n, lda, incx, incy, alpha, x, y, A);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSsyr2", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);

    Syr2TilingData tiling = CalSyr2TilingData(useNumBlocks, n, lda, uplo, *alpha, incx, incy);

    OP_LOGD(
        "aclblasSsyr2", "tiling: n=%u lda=%u uplo=%u incx=%ld incy=%ld numThreads=%u rowsPerBlock=%u numBlocks=%u",
        tiling.n, tiling.lda, tiling.uplo, tiling.incx, tiling.incy, tiling.numThreads, tiling.rowsPerBlock,
        useNumBlocks);
    OP_LOGI("aclblasSsyr2", "launching kernel: blocks=%u, cores=%u", useNumBlocks, aivCoreNum);

    const aclError launchErr = syr2_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(const_cast<float*>(y)),
        reinterpret_cast<uint8_t*>(A), tiling, useNumBlocks, h->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasSsyr2", "syr2_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
