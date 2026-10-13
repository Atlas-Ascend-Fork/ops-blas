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
 * \file ctrmv_host.cpp
 * \brief single-precision complex triangular matrix-vector multiply host implementation for ascend950
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "ctrmv_common.h"

static aclblasStatus_t ValidateCtrmvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const aclblasComplex* a, int lda,
    const aclblasComplex* x, int incx)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCtrmv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE("aclblasCtrmv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) {
        OP_LOGE("aclblasCtrmv", "invalid diag=%d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0) {
        OP_LOGE("aclblasCtrmv", "invalid n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasCtrmv", "invalid lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasCtrmv", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (a == nullptr) {
        OP_LOGE("aclblasCtrmv", "a must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasCtrmv", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Largest power of two <= x (x >= 1). The warp-shuffle reduction width must be
// a power of two, so the per-row thread count is rounded down to one.
inline uint32_t CtrmvRoundDownPow2(uint32_t x) { return 1u << (31u - __builtin_clz(x)); }

static CtrmvTilingData CalCtrmvTilingData(
    uint32_t useNumBlocks, int n, int lda, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int incx)
{
    CtrmvTilingData tilingData{};
    const uint32_t rowsPerCore = CeilDiv<uint32_t>(static_cast<uint32_t>(n), useNumBlocks);
    // Each row is cooperatively accumulated by up to CTRMV_THREADS_PER_ROW
    // threads (column split + intra-group warp-shuffle reduction). The factor is
    // capped by the row count and by the warp-shuffle width (<= 32).
    constexpr uint32_t CTRMV_THREADS_PER_ROW = 8;
    uint32_t threadsPerRow = std::min<uint32_t>(CTRMV_THREADS_PER_ROW, std::max<uint32_t>(n, 1));
    threadsPerRow = CtrmvRoundDownPow2(threadsPerRow);
    // 64-bit intermediate: rowsPerCore * threadsPerRow can overflow uint32 for
    // huge n (the interface puts no upper bound on n), so compute the block
    // size exactly before capping it at SIMT_MAX_THREAD_NUM.
    const uint64_t lanes = static_cast<uint64_t>(rowsPerCore) * threadsPerRow;
    uint32_t numThreads =
        CeilAlign<uint32_t>(static_cast<uint32_t>(std::min<uint64_t>(lanes, SIMT_MAX_THREAD_NUM)), SIMT_MIN_THREAD_NUM);
    numThreads = std::min(numThreads, SIMT_MAX_THREAD_NUM);
    tilingData.numThreads = numThreads;
    tilingData.rowsPerCore = rowsPerCore;
    tilingData.threadsPerRow = threadsPerRow;
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.trans = static_cast<uint32_t>(trans);
    tilingData.diag = static_cast<uint32_t>(diag);
    tilingData.incx = incx;
    return tilingData;
}

aclblasStatus_t aclblasCtrmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
{
    if (n < 0) {
        OP_LOGE("aclblasCtrmv", "invalid n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (handle == nullptr) {
        OP_LOGE("aclblasCtrmv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateCtrmvParams(uplo, trans, diag, n, A, lda, x, incx);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCtrmv", "parameter validation failed, status=%d", static_cast<int>(st));
        return st;
    }

    aclrtStream useStream = handle->stream;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCtrmv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // Use all AIV cores; each core processes rowsPerCore rows with a thread block
    // whose threads cooperate per row (column split), so medium n still uses the
    // full multi-core width.
    uint32_t useNumBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    useNumBlocks = std::max<uint32_t>(useNumBlocks, 1);

    CtrmvTilingData tilingData = CalCtrmvTilingData(useNumBlocks, n, lda, uplo, trans, diag, incx);
    // Fused single-launch path: the whole x is staged into UB as native float2,
    // so it requires a contiguous x (incx == 1) and n within the UB cache
    // capacity. The kernel handles every uplo/trans/diag combination; anything
    // beyond the capacity uses the two-stage path.
    if (n <= static_cast<int>(CTRMV_SMALL_N_MAX) && incx == 1) {
        OP_LOGD("aclblasCtrmv", "small fused path: n=%u numThreads=%u", tilingData.n, tilingData.numThreads);
        ctrmv_arch35_small_kernel_do(A, x, tilingData, useStream);
        return ACLBLAS_STATUS_SUCCESS;
    }

    OP_LOGD(
        "aclblasCtrmv", "tiling: n=%u lda=%u uplo=%u trans=%u diag=%u incx=%d numThreads=%u numBlocks=%u", tilingData.n,
        tilingData.lda, tilingData.uplo, tilingData.trans, tilingData.diag, tilingData.incx, tilingData.numThreads,
        useNumBlocks);
    OP_LOGI("aclblasCtrmv", "launching kernel");

    const size_t workspaceNeed = static_cast<size_t>(n) * sizeof(aclblasComplex);
    if (workspaceNeed > GetEffectiveWorkspaceSize(handle)) {
        OP_LOGE("aclblasCtrmv", "workspace %zu > handle %zu", workspaceNeed, GetEffectiveWorkspaceSize(handle));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle));

    ctrmv_arch35_kernel_do(A, x, workspaceDevice, tilingData, useNumBlocks, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
