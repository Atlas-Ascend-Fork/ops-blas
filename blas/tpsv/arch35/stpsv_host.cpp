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
 * \file stpsv_host.cpp
 * \brief Single-precision tpsv host-side implementation.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "stpsv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

struct StpsvTilingData;

void stpsv_kernel_do(const StpsvTilingData& tiling, void* stream);

static constexpr uint32_t SIMT_THRESHOLD = 128;

static aclblasStatus_t ValidateTpsvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int incx, const float* AP,
    const float* x)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasStpsv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasStpsv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
              OP_LOGE("aclblasStpsv", "invalid diag=%d", static_cast<int>(diag));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasStpsv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    // The per-element x offset is i * |incx| and the device GM view length is (n-1)*|incx| + 1.
    // Both are computed in uint32 on the device (stpsv_kernel.cpp Init/XOffset). Reject shapes whose
    // span (n-1)*|incx| reaches or exceeds UINT32_MAX: at span == UINT32_MAX the per-element offset
    // still fits uint32, but the GM view length span+1 wraps to 0 in uint32 (kernel Init), so we
    // must refuse at the boundary. The CSV suite keeps |incx| <= 3 and never hits this, but a
    // hand-built caller could (e.g. n=4, incx=1431655765 -> (4-1)*1431655765 = 4294967295 =
    // UINT32_MAX), which would otherwise wrap xCount and silently corrupt x. Same guard as the
    // arch22 stpsv_host.cpp.
    const int64_t absIncx64 = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const uint64_t span = static_cast<uint64_t>(n > 0 ? (n - 1) : 0) * static_cast<uint64_t>(absIncx64);
    CHECK_RET(span < 0xFFFFFFFFULL,
              OP_LOGE("aclblasStpsv", "x index span %llu overflows uint32", (unsigned long long)span);
              return ACLBLAS_STATUS_NOT_SUPPORTED);
    CHECK_RET(AP != nullptr, OP_LOGE("aclblasStpsv", "AP must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasStpsv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasStpsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasStpsv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasStpsv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateTpsvParams(uplo, trans, diag, n, incx, AP, x);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Determine execution path: numThreads=0 → scalar, numThreads>0 → SIMT
    uint32_t numThreads = 0;
    if (static_cast<uint32_t>(n) >= SIMT_THRESHOLD) {
        numThreads = std::min(CeilAlign<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    }

    StpsvTilingData tiling;
    tiling.ap = reinterpret_cast<uint64_t>(AP);
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.n = static_cast<uint32_t>(n);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = static_cast<int64_t>(incx);
    tiling.numThreads = numThreads;

    OP_LOGD(
        "aclblasStpsv", "tiling: n=%u uplo=%u trans=%u diag=%u incx=%ld numThreads=%u", tiling.n, tiling.uplo,
        tiling.trans, tiling.diag, tiling.incx, tiling.numThreads);
    OP_LOGI("aclblasStpsv", "launching kernel");

    stpsv_kernel_do(tiling, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
