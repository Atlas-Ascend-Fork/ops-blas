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
#include "syr_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

struct SyrTilingData;

void syr_kernel_do(uint8_t* x, uint8_t* A, const SyrTilingData &tiling,
                   uint32_t numBlocks, void *stream);

static aclblasStatus_t ValidateSyrParams(
    aclblasFillMode_t uplo, int n, int lda, int incx, const float* alpha, const float* x, const float* A)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSsyr", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        lda >= std::max(1, n), OP_LOGE("aclblasSsyr", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSsyr", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSsyr", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSsyr", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasSsyr", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SyrTilingData CalSyrTilingData(
    uint32_t useNumBlocks, int n, int lda, aclblasFillMode_t uplo, float alpha, int incx)
{
    SyrTilingData tilingData{};
    tilingData.numThreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.rowsPerBlock =
        static_cast<uint32_t>((n + static_cast<int>(useNumBlocks) - 1) / static_cast<int>(useNumBlocks));
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    tilingData.incx = static_cast<int64_t>(incx);

    return tilingData;
}

aclblasStatus_t aclblasSsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const float* alpha, const float* x, const int incx,
    float* A, const int lda)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSsyr", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasSsyr", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateSyrParams(uplo, n, lda, incx, alpha, x, A);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    aclrtStream stream = h->stream;

    // issue #510: alpha 允许 Host/Device 指针；Device 指针须先经 handle stream D2H 读回，
    // 不能在 Host 直接解引用，否则会崩溃或读到错误标量。
    bool alphaIsDevice = false;
    aclblasStatus_t alphaLocSt = CheckPtrLocation(alpha, &alphaIsDevice);
    if (alphaLocSt != ACLBLAS_STATUS_SUCCESS) {
        return alphaLocSt;
    }
    float alphaVal = 0.0f;
    if (alphaIsDevice) {
        aclblasStatus_t readSt = ReadHostScalarFromDevice(alpha, alphaVal, stream, "aclblasSsyr");
        if (readSt != ACLBLAS_STATUS_SUCCESS) {
            return readSt;
        }
    } else {
        alphaVal = *alpha;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSsyr", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);

    SyrTilingData tiling = CalSyrTilingData(useNumBlocks, n, lda, uplo, alphaVal, incx);

    OP_LOGD(
        "aclblasSsyr", "tiling: n=%u lda=%u uplo=%u incx=%ld numThreads=%u rowsPerBlock=%u numBlocks=%u", tiling.n,
        tiling.lda, tiling.uplo, tiling.incx, tiling.numThreads, tiling.rowsPerBlock, useNumBlocks);
    OP_LOGI("aclblasSsyr", "launching kernel: blocks=%u, cores=%u", useNumBlocks, aivCoreNum);

    syr_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(A), tiling, useNumBlocks, stream);

    return ACLBLAS_STATUS_SUCCESS;
}
