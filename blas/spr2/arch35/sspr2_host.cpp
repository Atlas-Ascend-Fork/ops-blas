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
#include "sspr2_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

void sspr2_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* ap, const Sspr2TilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateSspr2Params(aclblasFillMode_t uplo, int incx, int incy, const float* alpha)
{
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSspr2", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasSspr2", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSspr2", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE("aclblasSspr2", "incx must not be INT_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSspr2", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != INT_MIN, OP_LOGE("aclblasSspr2", "incy must not be INT_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// !831 检视意见：与 arch22（#781）口径收敛——alpha == 0 是 BLAS no-op（x/y/ap 不被引用），
// 指针判空仅在 alpha != 0 时执行，消除同一公有接口跨架构返回码分裂
static aclblasStatus_t ValidateSspr2DataPtrs(const float* x, const float* y, const float* ap)
{
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSspr2", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSspr2", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ap != nullptr, OP_LOGE("aclblasSspr2", "ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static Sspr2TilingData CalSspr2TilingData(int n, aclblasFillMode_t uplo, float alpha, int incx, int incy)
{
    Sspr2TilingData tilingData{};
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.incy = static_cast<int64_t>(incy);
    return tilingData;
}

aclblasStatus_t aclblasSspr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y,
    int incy, float* ap)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSspr2", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasSspr2", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateSspr2Params(uplo, incx, incy, alpha);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    float alphaVal = *alpha;
    if (alphaVal == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    st = ValidateSspr2DataPtrs(x, y, ap);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    aclrtStream stream = h->stream;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSspr2", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(n, SIMT_MIN_THREAD_NUM), aivCoreNum);

    Sspr2TilingData tiling = CalSspr2TilingData(n, uplo, alphaVal, incx, incy);

    sspr2_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(const_cast<float*>(y)),
        reinterpret_cast<uint8_t*>(ap), tiling, useNumBlocks, stream);

    return ACLBLAS_STATUS_SUCCESS;
}
