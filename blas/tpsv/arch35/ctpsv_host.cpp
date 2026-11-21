/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ctpsv_host.cpp
 * \brief Single-precision complex tpsv host-side implementation.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "ctpsv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

struct CtpsvTilingData;

void ctpsv_kernel_do(const CtpsvTilingData& tiling, void* stream);

static constexpr uint32_t SIMT_THRESHOLD = 128;
static constexpr uint32_t SIMT_MAX_BLOCKS = 8;

// The AIV core count is a device property and never changes during a process; querying it
// through PlatformAscendCManager on every call adds measurable per-launch host overhead, so
// cache it on first use.
static uint32_t CtpsvCoreCount()
{
    static uint32_t cores = GetAivCoreCount();
    return (cores == 0) ? 1U : cores;
}

// The SIMT tree reduction halves the thread count every step, so the block size must be a power of two.
static constexpr uint32_t CtpsvNextPow2(uint32_t v)
{
    uint32_t r = 1;
    while (r < v) {
        r <<= 1;
    }
    return r;
}

static aclblasStatus_t ValidateCtpsvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int incx, const aclblasComplex* AP,
    const aclblasComplex* x)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasCtpsv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasCtpsv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
        OP_LOGE("aclblasCtpsv", "invalid diag=%d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasCtpsv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(AP != nullptr, OP_LOGE("aclblasCtpsv", "AP must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasCtpsv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCtpsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasCtpsv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasCtpsv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateCtpsvParams(uplo, trans, diag, n, incx, AP, x);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Determine execution path: numThreads=0 → scalar, numThreads>0 → SIMT
    uint32_t numThreads = 0;
    uint32_t numBlocks = 1;
    if (static_cast<uint32_t>(n) >= SIMT_THRESHOLD) {
        numBlocks = std::min(CtpsvCoreCount(), SIMT_MAX_BLOCKS);
        numBlocks = std::min(numBlocks, static_cast<uint32_t>(n));
        // Defensive guard: min(CtpsvCoreCount(), SIMT_MAX_BLOCKS, n) is already >= 1 (CtpsvCoreCount()
        // returns >= 1 and n >= SIMT_THRESHOLD); this branch is unreachable and only silences static analysis.
        if (numBlocks == 0U) {
            numBlocks = 1U;
        }
        const uint32_t rowsPerBlock = (static_cast<uint32_t>(n) + numBlocks - 1) / numBlocks;
        numThreads = std::min(CtpsvNextPow2(rowsPerBlock), SIMT_MAX_THREAD_NUM);
        numThreads = std::max(numThreads, 32U);
    }

    CtpsvTilingData tiling{};
    tiling.ap = reinterpret_cast<uint64_t>(AP);
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.n = static_cast<uint32_t>(n);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.incx = static_cast<int64_t>(incx);
    tiling.numThreads = numThreads;
    tiling.numBlocks = numBlocks;

    OP_LOGD(
        "aclblasCtpsv", "tiling: n=%u uplo=%u trans=%u diag=%u incx=%ld numThreads=%u numBlocks=%u", tiling.n,
        tiling.uplo, tiling.trans, tiling.diag, tiling.incx, tiling.numThreads, tiling.numBlocks);
    OP_LOGI("aclblasCtpsv", "launching kernel");

    ctpsv_kernel_do(tiling, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
