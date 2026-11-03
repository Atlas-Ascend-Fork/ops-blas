/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "stbsv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void stbsv_kernel_do(const StbsvTilingData& tiling, void* stream);

static uint64_t GetTbsvUbBudgetBytes()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGW("aclblasStbsv", "platform unavailable, use scalar fallback");
        return 0;
    }
    uint64_t ubSizeBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeBytes);
    // Divide first to avoid overflow; align down before sharing the budget
    // with both host dispatch and device-side buffer allocation.
    uint64_t budgetBytes = ubSizeBytes / STBSV_UB_BUDGET_DENOMINATOR * STBSV_UB_BUDGET_NUMERATOR;
    // InitBuffer takes a 32-bit byte count; this is an API representation
    // limit, not an assumed hardware capacity.
    if (budgetBytes > UINT32_MAX) {
        budgetBytes = UINT32_MAX;
    }
    return budgetBytes / STBSV_UB_ALIGNMENT_BYTES * STBSV_UB_ALIGNMENT_BYTES;
}

static aclblasStatus_t ValidateTbsvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, int lda, int incx,
    const float* A, const float* x)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasStbsv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasStbsv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
              OP_LOGE("aclblasStbsv", "invalid diag=%d", static_cast<int>(diag));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda > k, OP_LOGE("aclblasStbsv", "invalid lda=%d, k=%d", lda, k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasStbsv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE("aclblasStbsv", "incx must not be INT_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    if (n > 0) {
        CHECK_RET(A != nullptr, OP_LOGE("aclblasStbsv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(x != nullptr, OP_LOGE("aclblasStbsv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasStbsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const float* A, int lda, float* x, int incx)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasStbsv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasStbsv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasStbsv", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);

    aclblasStatus_t st = ValidateTbsvParams(uplo, trans, diag, n, k, lda, incx, A, x);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (k == 0 && diag == ACLBLAS_UNIT) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    StbsvTilingData tiling{};
    tiling.a = reinterpret_cast<uint64_t>(A);
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = incx;
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ubSizeBytes = GetTbsvUbBudgetBytes();
    OP_LOGD(
        "aclblasStbsv", "tiling: n=%u k=%u uplo=%u trans=%u diag=%u incx=%d lda=%u", tiling.n, tiling.k, tiling.uplo,
        tiling.trans, tiling.diag, tiling.incx, tiling.lda);
    OP_LOGI("aclblasStbsv", "launching kernel");

    stbsv_kernel_do(tiling, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
