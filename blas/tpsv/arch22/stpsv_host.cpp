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
 * \brief Single-precision triangular packed solver host-side entry (arch22, Atlas A2/A3).
 */

#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "stpsv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"

void stpsv_kernel_do(const StpsvTilingData& tiling, void* stream);

static bool IsValidFillMode(aclblasFillMode_t uplo) { return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER; }

static bool IsValidOperation(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

static bool IsValidDiagType(aclblasDiagType_t diag) { return diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT; }

static aclblasStatus_t ValidateStpsvArgs(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasStpsv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasStpsv", "invalid n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // The solve keeps the whole solution vector resident in UB, which bounds the order.
    // Rejecting up front is far better than launching a kernel that would overrun UB.
    if (static_cast<uint32_t>(n) > ACLBLAS_STPSV_MAX_N) {
        OP_LOGE("aclblasStpsv", "n=%d exceeds the supported maximum %u", n, ACLBLAS_STPSV_MAX_N);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    // Enum arguments are rejected with INVALID_VALUE, consistent with the rest of ops-blas.
    if (!IsValidFillMode(uplo)) {
        OP_LOGE("aclblasStpsv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsValidOperation(trans)) {
        OP_LOGE("aclblasStpsv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsValidDiagType(diag)) {
        OP_LOGE("aclblasStpsv", "invalid diag=%d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasStpsv", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // The per-element x offset is i * |incx| and the device GM view length is (n-1)*|incx| + 1.
    // Both are computed in uint32 on the device. Reject shapes whose span (n-1)*|incx| reaches or
    // exceeds UINT32_MAX: at span == UINT32_MAX the per-element offset still fits uint32, but the
    // GM view length span+1 wraps to 0 in uint32 (kernel Init), so we must refuse at the boundary.
    // The CSV suite keeps |incx| <= 3 and never hits this, but a hand-built caller could (e.g.
    // n=4, incx=1431655765 -> (4-1)*1431655765 = 4294967295 = UINT32_MAX), which would otherwise
    // wrap xLen and silently corrupt x.
    const int64_t absIncx64 = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const uint64_t span = static_cast<uint64_t>(n > 0 ? (n - 1) : 0) * static_cast<uint64_t>(absIncx64);
    if (span >= 0xFFFFFFFFULL) {
        OP_LOGE("aclblasStpsv", "x index span %llu overflows uint32", (unsigned long long)span);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (AP == nullptr) {
        OP_LOGE("aclblasStpsv", "AP must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasStpsv", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasStpsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx)
{
    // The handle is validated first, before any quick-return shortcut. A null handle must be
    // reported as HANDLE_IS_NULLPTR even when n == 0, matching the arch35 path and the
    // reference (cuBLAS / Netlib) behaviour. Otherwise arch22 and the 950 (arch35) path would
    // disagree on the return value for the (n == 0 && handle == nullptr) combination.
    if (handle == nullptr) {
        OP_LOGE("aclblasStpsv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    // n == 0 is a legal quick return: AP and x are neither read nor written.
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateStpsvArgs(handle, uplo, trans, diag, n, AP, x, incx);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = handle;
    StpsvTilingData tiling{};
    tiling.ap = reinterpret_cast<uint64_t>(AP);
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.n = static_cast<uint32_t>(n);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = static_cast<int64_t>(incx);

    OP_LOGD(
        "aclblasStpsv", "tiling: n=%u uplo=%u trans=%u diag=%u incx=%ld", tiling.n, tiling.uplo, tiling.trans,
        tiling.diag, tiling.incx);

    stpsv_kernel_do(tiling, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
