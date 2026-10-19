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
#include "sspr_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

// ---- kernel entry (local declaration, replacing aclblas_kernel_do.h) ----
void sspr_kernel_do(uint8_t* x, uint8_t* ap, const SsprTilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateSsprParams(
    aclblasFillMode_t uplo, int incx, const float* alpha, const float* x, const float* ap)
{
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSspr", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSspr", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSspr", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE("aclblasSspr", "incx must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSspr", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ap != nullptr, OP_LOGE("aclblasSspr", "ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SsprTilingData CalSsprTilingData(uint32_t useNumBlocks, int n, aclblasFillMode_t uplo, float alpha, int incx)
{
    SsprTilingData tilingData{};
    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t colsPerBlk = CeilDiv<uint32_t>(nU32, useNumBlocks);
    tilingData.numThreads = std::min(CeilAlign<uint32_t>(colsPerBlk, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.columnsPerBlock = colsPerBlk;
    tilingData.n = nU32;
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    tilingData.incx = static_cast<int64_t>(incx);
    return tilingData;
}

aclblasStatus_t aclblasSspr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* ap)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSspr", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasSspr", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateSsprParams(uplo, incx, alpha, x, ap);
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
        aclblasStatus_t readSt = ReadHostScalarFromDevice(alpha, alphaVal, stream, "aclblasSspr");
        if (readSt != ACLBLAS_STATUS_SUCCESS) {
            return readSt;
        }
    } else {
        alphaVal = *alpha;
    }
    if (alphaVal == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSspr", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);

    SsprTilingData tiling = CalSsprTilingData(useNumBlocks, n, uplo, alphaVal, incx);

    OP_LOGI(
        "aclblasSspr", "launching kernel: blocks=%u, cores=%u, ubEligible=%d", useNumBlocks, aivCoreNum,
        (incx == 1 && n >= static_cast<int>(UB_THRESHOLD) && static_cast<uint32_t>(n) <= UB_X_FLOATS));

    sspr_kernel_do((uint8_t*)const_cast<float*>(x), (uint8_t*)ap, tiling, useNumBlocks, stream);

    return ACLBLAS_STATUS_SUCCESS;
}
