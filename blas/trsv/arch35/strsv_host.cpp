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
 * \file strsv_host.cpp
 * \brief Single-precision trsv host-side implementation.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "strsv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

struct StrsvTilingData;

void strsv_kernel_do(uint8_t* gmAddrA, uint8_t* gmAddrX, const StrsvTilingData& tiling, void* stream);

static constexpr uint32_t SIMT_OPTIMAL_THREADS = 128;

static aclblasStatus_t ValidateTrsvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int incx, int lda, const float* A,
    const float* x)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasStrsv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasStrsv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
              OP_LOGE("aclblasStrsv", "invalid diag=%d", static_cast<int>(diag));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasStrsv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda >= std::max<int>(1, n), OP_LOGE("aclblasStrsv", "invalid lda=%d, n=%d", lda, n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasStrsv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasStrsv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    // issue #744: 对齐同 arch ctrsv 的跨度守卫——kernel 侧偏移按 uint32 寻址，
    // n*lda 或 (n-1)*|incx| 超出 uint32 时地址回绕产生越界读写；超界显式拒绝
    const uint64_t stride = incx < 0 ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx);
    const uint64_t extent = static_cast<uint64_t>(n - 1) * stride + 1;
    const uint64_t limit = UINT32_MAX;
    if (static_cast<uint64_t>(n) * static_cast<uint64_t>(lda) > limit || extent > limit) {
        OP_LOGE("aclblasStrsv", "matrix or vector extent exceeds uint32 range: n=%d lda=%d incx=%d", n, lda, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasStrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasStrsv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasStrsv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateTrsvParams(uplo, trans, diag, n, incx, lda, A, x);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t numThreads = SIMT_OPTIMAL_THREADS;

    StrsvTilingData tiling;
    tiling.n = static_cast<uint32_t>(n);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = static_cast<int32_t>(incx);
    tiling.lda = static_cast<int32_t>(lda);
    tiling.numThreads = numThreads;

    OP_LOGD(
        "aclblasStrsv", "tiling: n=%u uplo=%u trans=%u diag=%u incx=%d lda=%d numThreads=%u", tiling.n, tiling.uplo,
        tiling.trans, tiling.diag, tiling.incx, tiling.lda, tiling.numThreads);
    OP_LOGI("aclblasStrsv", "launching kernel");

    strsv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(x), tiling, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
