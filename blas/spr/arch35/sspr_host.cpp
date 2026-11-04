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
#include "common/helper/sspr_host_common.h"

// ---- kernel entry (local declaration, replacing aclblas_kernel_do.h) ----
void sspr_kernel_do(uint8_t* x, uint8_t* ap, const SsprTilingData& tiling, uint32_t numBlocks, void* stream);

// !831 检视意见：与 arch22（#781）口径收敛——alpha == 0 是 BLAS no-op（x/ap 不被引用），
// x/ap 判空仅在 alpha != 0 时执行（公共序列见 sspr_host_common.h 的 SsprPostAlphaChecks），
// 消除同一公有接口跨架构返回码分裂。本架构参数校验契约：alpha 判空先行、uplo 非法返回
// INVALID_VALUE（与 arch22 的 INVALID_ENUM 有意不同），故校验器保留在本文件经函数指针
// 传入公共入口序列。
static aclblasStatus_t ValidateSsprParams(aclblasFillMode_t uplo, int incx, const float* alpha)
{
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSspr", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasSspr", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSspr", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE("aclblasSspr", "incx must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static SsprTilingData CalSsprTilingData(uint32_t useNumBlocks, int n, aclblasFillMode_t uplo, float alpha, int incx);

aclblasStatus_t aclblasSspr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* ap)
{
    SsprEntryCtx ctx;
    if (!SsprEntryProceed(handle, uplo, n, incx, alpha, ValidateSsprParams, &ctx)) {
        return ctx.st;
    }

    aclrtStream stream = ctx.h->stream;

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
    uint32_t aivCoreNum = 0;
    if (!SsprPostAlphaChecks(alphaVal, x, ap, &ctx.st, &aivCoreNum)) {
        return ctx.st;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);

    SsprTilingData tiling = CalSsprTilingData(useNumBlocks, n, uplo, alphaVal, incx);

    OP_LOGI(
        "aclblasSspr", "launching kernel: blocks=%u, cores=%u, ubEligible=%d", useNumBlocks, aivCoreNum,
        (incx == 1 && n >= static_cast<int>(UB_THRESHOLD) && static_cast<uint32_t>(n) <= UB_X_FLOATS));

    sspr_kernel_do((uint8_t*)const_cast<float*>(x), (uint8_t*)ap, tiling, useNumBlocks, stream);

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
