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
 * \file sgbmv_host.cpp
 * \brief single-precision gbmv host-side implementation for Atlas A2/A3 (arch22)
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "sgbmv_kernel.h"
#include "sgbmv_tiling_data.h"

constexpr uint32_t CORE_SPLIT_CHUNK_SIZE = 32U;
constexpr uint32_t MIN_CHUNKS_FOR_MULTI_CORE = 2U;

// ---- L1 column-split plan (mirrors the kernel-side gating; see sgbmv_kernel) ----
constexpr uint32_t SGBMV_L1_WIN_MAX = 8192U;     // per-core y window cap (floats)
constexpr uint32_t SGBMV_L1_BAND_MAX = 6120U;    // band slot cap (staging slot is 6144 floats)
constexpr uint32_t SGBMV_L1_CORES_MAX = 64U;     // reduction tree width
constexpr uint32_t SGBMV_L1_X_ELEMS = 2048U;     // per-core x column window (floats)
constexpr uint64_t SGBMV_L1_WS_MAX = 512ULL * 1024ULL * 1024ULL;
constexpr uint32_t SGBMV_FAST_TILE_M = 128U;     // mirrors FAST_TILE_M in the kernel
constexpr uint32_t SGBMV_FAST_A_BYTES = 160000U; // mirrors MAX_UB_A_BYTES in the kernel

// Band-storage extent in elements. kl and ku are each bounded only by m/n
// (kl < m, ku < n), so their sum can exceed INT_MAX for extreme shapes;
// lda comparisons against it go through here so the 32-bit sum never wraps.
static int64_t SgbmvBandExtent(int kl, int ku)
{
    return static_cast<int64_t>(kl) + static_cast<int64_t>(ku) + 1;
}

// Mirrors IsFastpathN in the kernel: shapes owned by the tiled fast path must
// not be redirected to the column-split path (they keep their own core plan).
static bool SgbmvFastNCompatible(int kl, int ku, int lda)
{
    if (static_cast<int64_t>(lda) < SgbmvBandExtent(kl, ku) || lda > 256) {
        return false;
    }
    uint64_t span = (static_cast<uint64_t>(SGBMV_FAST_TILE_M) + static_cast<uint64_t>(kl) + static_cast<uint64_t>(ku) +
                     63U) / 32U * 32U;
    return static_cast<uint64_t>(lda) * span * 4ULL <= SGBMV_FAST_A_BYTES;
}

static bool SgbmvL1Plan(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, float alpha,
    uint32_t aivCoreNum, uint32_t& outCores, uint64_t& outBytes)
{
    if (trans != ACLBLAS_OP_N || incx != 1 || incy != 1 || alpha == 0.0f || m <= 0 || n <= 0) {
        return false;
    }
    if (static_cast<int64_t>(lda) != SgbmvBandExtent(kl, ku)) {
        return false; // compact band storage only (columns contiguous in GM)
    }
    if (SgbmvFastNCompatible(kl, ku, lda)) {
        return false;
    }
    uint32_t band = static_cast<uint32_t>(lda);
    if (band > SGBMV_L1_BAND_MAX) {
        return false;
    }
    uint32_t mm = static_cast<uint32_t>(m);
    uint32_t nn = static_cast<uint32_t>(n);
    uint32_t cores = std::min(aivCoreNum, nn);
    // Small-n shapes (e.g. n=77, band=831): every extra core costs reduction
    // traffic (K workspace slices read+folded) and per-core zero-init while
    // owning only a handful of streamed columns — cap at ceil(n/4) cores
    // (floor 8) so each core still owns at least ~4 columns.
    uint32_t coresByCols = (nn + 3U) / 4U;
    if (coresByCols >= 8U) {
        cores = std::min(cores, coresByCols);
    }
    if (cores == 0U) {
        return false;
    }
    if (cores > SGBMV_L1_CORES_MAX) {
        cores = SGBMV_L1_CORES_MAX;
    }
    uint32_t colsPerCore = (nn + cores - 1U) / cores;
    if (colsPerCore > SGBMV_L1_X_ELEMS) {
        return false;
    }
    uint32_t worstWin = std::min<uint32_t>(mm, colsPerCore + band - 1U);
    if (worstWin > SGBMV_L1_WIN_MAX) {
        return false;
    }
    uint32_t stride = (mm + 7U) / 8U * 8U;
    uint64_t bytes = static_cast<uint64_t>(cores) * stride * 4ULL;
    if (bytes > SGBMV_L1_WS_MAX) {
        return false;
    }
    outCores = cores;
    outBytes = bytes;
    return true;
}

static aclblasStatus_t ValidateSgbmvDims(int m, int n, int kl, int ku, int lda)
{
    CHECK_RET(m >= 0, OP_LOGE("aclblasSgbmv", "invalid m=%d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSgbmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(kl >= 0, OP_LOGE("aclblasSgbmv", "invalid kl=%d", kl); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ku >= 0, OP_LOGE("aclblasSgbmv", "invalid ku=%d", ku); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        m <= 0 || kl < m, OP_LOGE("aclblasSgbmv", "kl=%d must be < m=%d", kl, m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        n <= 0 || ku < n, OP_LOGE("aclblasSgbmv", "ku=%d must be < n=%d", ku, n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        static_cast<int64_t>(lda) >= SgbmvBandExtent(kl, ku),
        OP_LOGE("aclblasSgbmv", "invalid lda=%d, kl=%d, ku=%d", lda, kl, ku);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateSgbmvScalars(int incx, int incy, const float* alpha, const float* beta)
{
    CHECK_RET(incx != 0, OP_LOGE("aclblasSgbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSgbmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSgbmv", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasSgbmv", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateSgbmvArrays(int m, int n, float alpha, const float* A, const float* x, const float* y)
{
    if (m <= 0 || n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (alpha != 0.0f) {
        CHECK_RET(A != nullptr, OP_LOGE("aclblasSgbmv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(x != nullptr, OP_LOGE("aclblasSgbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSgbmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateSgbmvParams(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, const float* alpha,
    const float* beta, const float* A, const float* x, const float* y)
{
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasSgbmv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM);
    aclblasStatus_t st = ValidateSgbmvDims(m, n, kl, ku, lda);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateSgbmvScalars(incx, incy, alpha, beta);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return ValidateSgbmvArrays(m, n, *alpha, A, x, y);
}

static SgbmvTilingData CalSgbmvTilingData(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, float alpha, float beta,
    uint32_t aivCoreNum)
{
    bool isTransT = (trans != ACLBLAS_OP_N);
    uint32_t outDim = isTransT ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);

    // Core-count chunk granularity for the TVec path:
    // mirrors the kernel-side gate (sgbmv_kernel: alpha != 0 && trans != N &&
    // unit strides && band > MAX_GATHER_BANDWIDTH(129) && (IsFastpathT ||
    // UseVecFallbackT)) — those paths split output columns at 8-float
    // boundaries, so the launch grid must be chunked at 8 as well.
    uint32_t chunkSize = CORE_SPLIT_CHUNK_SIZE;
    if (isTransT && incx == 1 && incy == 1 && alpha != 0.0f) {
        const uint32_t band = static_cast<uint32_t>(kl) + static_cast<uint32_t>(ku) + 1U;
        if (band > 129U) {
            bool tvecGate = (static_cast<int64_t>(lda) >= SgbmvBandExtent(kl, ku) && lda <= 256); // IsFastpathT / MAX_FAST_LDA
            if (!tvecGate) { // UseVecFallbackT: rowP/XWIN budget (kernel constants)
                uint32_t band8 = (band + 7U) / 8U * 8U;
                uint32_t rowP = 8U;
                while (rowP < band8) {
                    rowP *= 2U;
                }
                tvecGate = rowP <= 4096U && band + 256U + rowP + 8U <= 8512U;
            }
            if (tvecGate) {
                chunkSize = 8U;
            }
        }
    }

    uint32_t useNumBlocks = 1U;
    if (incy == 1) {
        uint32_t numChunks = outDim / chunkSize;
        if (numChunks >= MIN_CHUNKS_FOR_MULTI_CORE) {
            useNumBlocks = std::min(numChunks, aivCoreNum);
        }
    }

    SgbmvTilingData tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.kl = static_cast<uint32_t>(kl);
    tiling.ku = static_cast<uint32_t>(ku);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.trans = isTransT ? 1U : 0U;
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.useCoreNum = useNumBlocks;
    return tiling;
}

aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A,
    int lda, const float* x, int incx, const float* beta, float* y, int incy)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSgbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    aclblasStatus_t st = ValidateSgbmvParams(trans, m, n, kl, ku, lda, incx, incy, alpha, beta, A, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (*alpha == 0.0f && *beta == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0U, OP_LOGE("aclblasSgbmv", "GetAivCoreCount failed"); return ACLBLAS_STATUS_INTERNAL_ERROR);

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    SgbmvTilingData tiling = CalSgbmvTilingData(trans, m, n, kl, ku, lda, incx, incy, *alpha, *beta, aivCoreNum);

    // L1 column-split plan (must mirror the kernel-side gating exactly): the
    // N path with unit strides and compact band storage (lda == kl+ku+1)
    // streams A columns contiguously per core; partials go through the
    // handle workspace and a cooperative reduction.
    uint8_t* workSpace = nullptr;
    uint32_t l1Cores = 0U;
    uint64_t l1Bytes = 0U;
    if (SgbmvL1Plan(trans, m, n, kl, ku, lda, incx, incy, *alpha, aivCoreNum, l1Cores, l1Bytes) &&
        EnsureDefaultWorkspace(h, l1Bytes) == ACLBLAS_STATUS_SUCCESS) {
        tiling.colSplit = 1U;
        tiling.useCoreNum = l1Cores;
        workSpace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
        sgbmv_kernel_l1_do(
            reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
            reinterpret_cast<uint8_t*>(y), workSpace, tiling, tiling.useCoreNum, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }

    sgbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(y), nullptr, tiling, tiling.useCoreNum, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
