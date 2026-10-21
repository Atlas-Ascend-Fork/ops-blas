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
 * \file ctbmv_host.cpp
 * \brief single-precision complex triangular band matrix-vector multiply host implementation for ascend950
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "ctbmv_common.h"

static aclblasStatus_t ValidateCtbmvParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasCtbmv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasCtbmv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT,
        OP_LOGE("aclblasCtbmv", "invalid diag=%d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(n >= 0, OP_LOGE("aclblasCtbmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasCtbmv", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    // k + 1 can overflow when k is INT_MAX; for non-negative k, lda > k is equivalent.
    CHECK_RET(
        lda > k, OP_LOGE("aclblasCtbmv", "invalid lda=%d, k=%d", lda, k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasCtbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(a != nullptr, OP_LOGE("aclblasCtbmv", "a must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasCtbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static CtbmvTilingData CalCtbmvTilingData(
    uint32_t useNumBlocks, int n, int k, int lda, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int incx)
{
    CtbmvTilingData tilingData{};
    uint32_t nValue = static_cast<uint32_t>(n);
    uint32_t kValue = static_cast<uint32_t>(k);
    tilingData.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(nValue, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tilingData.numBlocks = useNumBlocks;
    tilingData.rowsPerBlock = CeilDiv<uint32_t>(nValue, useNumBlocks);
    // Unit-stride small vectors also fit in UB; use the parallel small-shape
    // kernel instead of the serial Netlib fallback.
    tilingData.useUb = (incx == 1 && nValue <= CTBMV_UB_X_ELEMS &&
        tilingData.rowsPerBlock + kValue <= CTBMV_UB_X_ELEMS) ? 1U : 0U;
    tilingData.n = nValue;
    tilingData.k = kValue;
    tilingData.lda = static_cast<uint32_t>(lda);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.trans = static_cast<uint32_t>(trans);
    tilingData.diag = static_cast<uint32_t>(diag);
    tilingData.incx = static_cast<int64_t>(incx);
    return tilingData;
}

// Chooses the grid for the requested shape and returns the matching tiling.
// The initial split spreads rows over the AIV cores. The refinements below then
// either pin narrow-band shapes that fit in UB to a single block, so the whole x
// snapshot stays resident, or widen the split for wide bands, where every row is
// a long serial chain that only more resident warps can hide. Each change to the
// block count rebuilds the tiling so the new split gets its own thread count
// instead of inheriting the previous partition.
static CtbmvTilingData PlanCtbmvTiling(
    uint32_t aivCoreNum, int n, int k, int lda, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int incx, uint32_t& useNumBlocks)
{
    useNumBlocks =
        std::max<uint32_t>(std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM), aivCoreNum), 1);
    CtbmvTilingData tilingData = CalCtbmvTilingData(useNumBlocks, n, k, lda, uplo, trans, diag, incx);

    // Narrow bands only: a wide band leaves every row a long serial chain, which
    // one block cannot hide (see the wide-band spread below).
    if (tilingData.useUb != 0U && tilingData.n <= CTBMV_SINGLE_BLOCK_UB_N &&
        tilingData.k < CTBMV_WIDE_BAND_MIN_K && useNumBlocks != 1U) {
        useNumBlocks = 1U;
        tilingData = CalCtbmvTilingData(useNumBlocks, n, k, lda, uplo, trans, diag, incx);
    }
    // A small wide-band shape pinned to one block would leave every row's chain
    // on a single core while the other cores stay idle, so spread the rows with
    // a narrow window instead. At or below CTBMV_NETLIB_SMALL_N the kernel
    // dispatches to the serial Netlib path and ignores the block count, so the
    // spread only applies above that size.
    if (tilingData.useUb != 0U && tilingData.n <= CTBMV_SINGLE_BLOCK_UB_N &&
        tilingData.n > CTBMV_NETLIB_SMALL_N && tilingData.k >= CTBMV_WIDE_BAND_MIN_K) {
        const uint32_t smallWideBlocks = std::max<uint32_t>(
            std::min(CeilDiv<uint32_t>(tilingData.n, CTBMV_SMALL_WIDEBAND_ROWS_PER_BLOCK), aivCoreNum), 1U);
        if (smallWideBlocks > useNumBlocks) {
            useNumBlocks = smallWideBlocks;
            tilingData = CalCtbmvTilingData(useNumBlocks, n, k, lda, uplo, trans, diag, incx);
        }
    }
    // Wide bands do more serial work per row. Spread the UB row windows over
    // more cores without changing VF launch alignment or each row's reduction
    // order. The window steps down as the band grows; a bigger window would
    // leave too few resident warps per core to hide the longer chains.
    uint32_t wideBandRowsPerBlock = CTBMV_WIDE_BAND_ROWS_PER_BLOCK;
    if (k >= CTBMV_WIDE_BAND_EXTREME_MIN_K) {
        wideBandRowsPerBlock = CTBMV_WIDE_BAND_EXTREME_ROWS_PER_BLOCK;
    } else if (k >= CTBMV_WIDE_BAND_MEDIUM_MIN_K) {
        wideBandRowsPerBlock = CTBMV_WIDE_BAND_MEDIUM_ROWS_PER_BLOCK;
    }
    if (tilingData.useUb != 0U && tilingData.n > wideBandRowsPerBlock &&
        tilingData.k >= CTBMV_WIDE_BAND_MIN_K) {
        const uint32_t wideBandBlocks =
            std::min(CeilDiv<uint32_t>(tilingData.n, wideBandRowsPerBlock), aivCoreNum);
        if (wideBandBlocks > useNumBlocks) {
            useNumBlocks = wideBandBlocks;
            tilingData = CalCtbmvTilingData(useNumBlocks, n, k, lda, uplo, trans, diag, incx);
        }
    }

    // One warp reduces one output row for wide bands. Keep sixteen warps per
    // block to hide GM-load and shuffle latency in short (k=64) wide rows.
    if (tilingData.useUb != 0U && tilingData.k >= CTBMV_WIDE_BAND_MIN_K) {
        tilingData.numThreads = std::max(tilingData.numThreads, CTBMV_WARP_REDUCTION_THREADS);
    }
    return tilingData;
}

static aclblasStatus_t LaunchCtbmvKernel(
    _aclblas_handle* h, const aclblasComplex* A, aclblasComplex* x, uint8_t* workspaceDevice,
    const CtbmvTilingData& tilingData, uint32_t useNumBlocks)
{
    OP_LOGD(
        "aclblasCtbmv",
        "tiling: n=%u k=%u lda=%u uplo=%u trans=%u diag=%u incx=%ld numBlocks=%u numThreads=%u",
        tilingData.n, tilingData.k, tilingData.lda, tilingData.uplo, tilingData.trans, tilingData.diag,
        tilingData.incx, tilingData.numBlocks, tilingData.numThreads);
    ctbmv_arch35_kernel_do(A, x, workspaceDevice, tilingData, useNumBlocks, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCtbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
{
    CHECK_RET(n >= 0, OP_LOGE("aclblasCtbmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCtbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateCtbmvParams(uplo, trans, diag, n, k, A, lda, x, incx);
    CHECK_RET(st == ACLBLAS_STATUS_SUCCESS, return st);
    if (k == 0 && diag == ACLBLAS_UNIT) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = handle;
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(
        aivCoreNum > 0, OP_LOGE("aclblasCtbmv", "GetAivCoreCount failed"); return ACLBLAS_STATUS_EXECUTION_FAILED);

    uint32_t useNumBlocks = 0;
    CtbmvTilingData tilingData = PlanCtbmvTiling(aivCoreNum, n, k, lda, uplo, trans, diag, incx, useNumBlocks);

    size_t workspaceSize = (k > 0) ? static_cast<size_t>(n) * sizeof(aclblasComplex) : 0;
    uint8_t* workspaceDevice = nullptr;
    if (k > 0) {
        CHECK_RET(
            workspaceSize <= GetEffectiveWorkspaceSize(h),
            OP_LOGE("aclblasCtbmv", "workspace %zu > handle %zu", workspaceSize, GetEffectiveWorkspaceSize(h));
            return ACLBLAS_STATUS_EXECUTION_FAILED);
        workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    }
    return LaunchCtbmvKernel(h, A, x, workspaceDevice, tilingData, useNumBlocks);
}

