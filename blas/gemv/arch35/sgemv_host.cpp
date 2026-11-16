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
 * \file sgemv_host.cpp
 * \brief Single-precision sgemv host-side implementation (SIMT).
 *
 * y = alpha * op(A) * x + beta * y
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "sgemv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

void sgemv_kernel_do(
    uint8_t* A, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgemvTilingData& tiling, uint32_t numBlocks,
    void* stream);

namespace {

// ==========================================================================
//  Slab fast-path configuration (kernels in sgemv_slab_kernel.cpp)
// ==========================================================================
constexpr uint32_t SGEMV_SLAB_MIN_DIM = 64;         // smaller products stay generic
constexpr uint32_t SGEMV_SLAB_MIN_COLS = 8;         // columns per chunk minimum
constexpr uint32_t SGEMV_T_UB_MAX_ROWS = 16384;     // x fits in the 64 KB __ubuf__ cache
constexpr uint32_t SGEMV_T_TILE_ROWS = 8192;        // row tile for larger x (must match kernel)
constexpr uint32_t SGEMV_N_MTE_ROWS = 1024;         // MTE tile rows (must match kernel)
constexpr uint32_t SGEMV_N_MTE_MAX_SLAB_COLS = 128; // MTE x-cache columns (must match kernel)
constexpr uint32_t SGEMV_T_MIN_SEG_ROWS = 512;      // segments shorter than this stream inefficiently
constexpr uint32_t SGEMV_T_WIDE_SLAB_COLS = 16;     // wide slabs only benefit from long segments
constexpr uint32_t SGEMV_T_WIDE_MIN_SEG_ROWS = 1024;
constexpr uint32_t SGEMV_T_MAX_SEG_SHIFT = 3;
constexpr uint32_t SGEMV_T_MAX_ITEMS = 512; // (column, segment) items per block (must match kernel)
// T columns shorter than this give every warp enough independent work that
// extra co-resident blocks add nothing (measured: 1024-row columns regress).
constexpr uint32_t SGEMV_T_MIN_WIDEN_ROWS = 4096;
// Matrices whose footprint fits the L2 keep whole columns per warp (long
// sequential runs stream better from L2); larger ones stream from DRAM,
// where the shorter, more numerous segment runs win.
constexpr size_t SGEMV_T_SEG_L2_BYTES = 112ULL << 20; // arch35 L2 (950PR PCIE 112 MB, server 128 MB)

// T slab: log2 of the row segments per column. Columns dealt round-robin to
// the 64 warps quantize badly when the chunk holds only 1-3 columns per
// warp, so the columns are sliced into 2^k row segments until the item
// count approaches the block's part[] budget; segments also fill idle warps
// in narrow slabs. Long segments still stream better, so slicing stops at
// the measured minimum rows per segment.
inline uint32_t CalcSgemvTSegShift(uint32_t m, uint32_t chunkLen)
{
    uint32_t minSegRows = (chunkLen >= SGEMV_T_WIDE_SLAB_COLS) ? SGEMV_T_WIDE_MIN_SEG_ROWS : SGEMV_T_MIN_SEG_ROWS;
    uint32_t shift = 0;
    while (shift < SGEMV_T_MAX_SEG_SHIFT) {
        uint32_t nextSegs = 1U << (shift + 1);
        if (chunkLen * nextSegs > SGEMV_T_MAX_ITEMS || (m >> (shift + 1)) < minSegRows) {
            break;
        }
        ++shift;
    }
    return shift;
}

// Whether the (column, segment) slicing pays off for this slab. Narrow
// slabs segment unconditionally (filling idle warps dominates); wide slabs
// only when the matrix streams from DRAM.
inline bool SgemvTSegPays(uint32_t n, uint32_t lda, uint32_t chunkLen)
{
    if (chunkLen < SGEMV_T_WIDE_SLAB_COLS) {
        return true;
    }
    return static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float) > SGEMV_T_SEG_L2_BYTES;
}

bool CanUseSgemvSlab(int m, int n, int lda, int incx, int incy, float alpha)
{
    // Slab kernels index A in 32-bit units and use warp-tree / two-launch
    // reduction topologies; restrict those to the unit-stride contract.
    // Small products stay generic so every component multiplication keeps
    // the explicit binary32 rounding boundary of the reference order.
    return (incx == 1) && (incy == 1) && (alpha != 0.0f) && (m >= static_cast<int>(SGEMV_SLAB_MIN_DIM)) &&
           (n >= static_cast<int>(SGEMV_SLAB_MIN_DIM)) &&
           (static_cast<size_t>(lda) * static_cast<size_t>(n) <= 0xFFFFFFFFULL);
}

SgemvTilingData MakeSgemvTiling(
    int m, int n, int lda, aclblasOperation_t trans, float alpha, float beta, int incx, int incy)
{
    SgemvTilingData tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.trans = (trans != ACLBLAS_OP_N) ? 1U : 0U;
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    return tiling;
}

uint32_t ConfigureSgemvGeneric(uint32_t outDim, uint32_t aivCoreNum, SgemvTilingData& tiling)
{
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(outDim, SIMT_MIN_THREAD_NUM), aivCoreNum);
    if (useNumBlocks == 0) {
        useNumBlocks = 1;
    }
    tiling.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(outDim, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tiling.rowsPerBlock = CeilDiv<uint32_t>(outDim, useNumBlocks);
    return useNumBlocks;
}

// Tiling fields every slab configuration shares (chunk geometry and the
// epilogue contract); branch-specific fields follow in each configurator.
inline void SetSgemvSlabCommon(SgemvTilingData& tiling, uint32_t chunkLen, uint32_t numChunks)
{
    tiling.useSlab = 1U;
    tiling.betaIsZero = (tiling.beta == 0.0f) ? 1U : 0U;
    tiling.usePairs = (tiling.m % 2U == 0U && tiling.lda % 2U == 0U) ? 1U : 0U;
    tiling.chunkLen = chunkLen;
    tiling.colChunks = numChunks;
}

// trans=N slab: one row (pair) per thread, balanced across row tiles. The
// pair kernel covers two rows per thread, so pairable matrices need half
// the threads per tile. Returns the block count, or 0 to fall back.
uint32_t ConfigureSgemvNSlab(
    aclblasHandle_t handle, SgemvTilingData& tiling, uint32_t& numChunks, uint32_t& chunkLen, uint8_t*& wsArg)
{
    uint32_t rowsPerThread = ((tiling.m % 2U == 0U) && (tiling.lda % 2U == 0U)) ? 2U : 1U;
    uint32_t rowTiles = CeilDiv<uint32_t>(tiling.m, SIMT_MAX_THREAD_NUM * rowsPerThread);
    uint32_t threadsPerTile = CeilDiv<uint32_t>(CeilDiv<uint32_t>(tiling.m, rowTiles), rowsPerThread);
    tiling.numThreads = std::min(CeilAlign<uint32_t>(threadsPerTile, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    rowTiles = CeilDiv<uint32_t>(tiling.m, tiling.numThreads * rowsPerThread);
    // Starved grid (m too small to fill the warp slots): widen the column
    // chunking past the core count so the MTE tile transport can stream
    // wide matrices with the DMA engine instead of latency-bound loads.
    if (tiling.m >= SGEMV_N_MTE_ROWS && threadsPerTile < SIMT_MAX_THREAD_NUM) {
        uint32_t wideChunks = CeilDiv<uint32_t>(tiling.n, SGEMV_N_MTE_MAX_SLAB_COLS);
        if (wideChunks > numChunks) {
            numChunks = wideChunks;
            chunkLen = CeilDiv<uint32_t>(tiling.n, numChunks);
            numChunks = CeilDiv<uint32_t>(tiling.n, chunkLen);
        }
    }
    wsArg = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    size_t wsNeed = static_cast<size_t>(numChunks) * tiling.m * sizeof(float);
    while (wsArg != nullptr && GetEffectiveWorkspaceSize(handle) < wsNeed && numChunks > 1U) {
        numChunks /= 2U;
        chunkLen = CeilDiv<uint32_t>(tiling.n, numChunks);
        numChunks = CeilDiv<uint32_t>(tiling.n, chunkLen);
        wsNeed = static_cast<size_t>(numChunks) * tiling.m * sizeof(float);
    }
    if (wsArg == nullptr || GetEffectiveWorkspaceSize(handle) < wsNeed) {
        OP_LOGD("aclblasSgemv", "workspace %zu insufficient, fallback", wsNeed);
        wsArg = nullptr;
        return 0;
    }
    SetSgemvSlabCommon(tiling, chunkLen, numChunks);
    tiling.rowTiles = rowTiles;
    return numChunks * rowTiles;
}

// trans=T with x cacheable in UB: one block per column chunk, x staged in
// __ubuf__. Two blocks per core: the co-resident pair interleaves their
// warps on the core, hiding load latency behind the issue-bound column
// recurrence (x is re-staged per block, a small re-read; a third block
// measurably over-subscribes the core, so exactly two). Only pays when the
// columns are long enough to be latency-bound; short columns already give
// every warp independent work. Returns the block count.
uint32_t ConfigureSgemvTSlabSmallM(
    uint32_t aivCoreNum, SgemvTilingData& tiling, uint32_t& numChunks, uint32_t& chunkLen)
{
    uint32_t tChunks = std::min(2U * aivCoreNum, CeilDiv<uint32_t>(tiling.n, SGEMV_SLAB_MIN_COLS));
    if (tChunks > numChunks && tiling.m >= SGEMV_T_MIN_WIDEN_ROWS) {
        numChunks = tChunks;
        chunkLen = CeilDiv<uint32_t>(tiling.n, numChunks);
        numChunks = CeilDiv<uint32_t>(tiling.n, chunkLen);
    }
    tiling.numThreads = SIMT_MAX_THREAD_NUM;
    SetSgemvSlabCommon(tiling, chunkLen, numChunks);
    tiling.segShift = SgemvTSegPays(tiling.n, tiling.lda, chunkLen) ? CalcSgemvTSegShift(tiling.m, chunkLen) : 0U;
    tiling.rowTiles = 0U;
    return numChunks;
}

// trans=T with x too large for UB: the matrix is cut into row tiles whose
// x segment is staged in UB (workspace partials over row tiles). Returns
// the block count, or 0 to fall back.
uint32_t ConfigureSgemvTSlabLargeM(
    aclblasHandle_t handle, uint32_t aivCoreNum, SgemvTilingData& tiling, uint32_t& numChunks, uint32_t& chunkLen,
    uint8_t*& wsArg)
{
    uint32_t rowTiles = CeilDiv<uint32_t>(tiling.m, SGEMV_T_TILE_ROWS);
    uint32_t targetChunks = CeilDiv<uint32_t>(aivCoreNum, rowTiles);
    uint32_t maxChunks = std::min(aivCoreNum, numChunks);
    if (targetChunks < 1U) {
        targetChunks = 1U;
    }
    if (targetChunks > maxChunks) {
        targetChunks = maxChunks;
    }
    chunkLen = CeilDiv<uint32_t>(tiling.n, targetChunks);
    numChunks = CeilDiv<uint32_t>(tiling.n, chunkLen);
    wsArg = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    size_t wsNeed = static_cast<size_t>(rowTiles) * tiling.n * sizeof(float);
    if (wsArg == nullptr || GetEffectiveWorkspaceSize(handle) < wsNeed) {
        OP_LOGD("aclblasSgemv", "workspace %zu insufficient, fallback", wsNeed);
        wsArg = nullptr;
        return 0;
    }
    tiling.numThreads = SIMT_MAX_THREAD_NUM;
    SetSgemvSlabCommon(tiling, chunkLen, numChunks);
    // Slice the tile's columns into row segments so the warp round-robin
    // quantizes smoothly (same gating as the small-m path, over tile rows).
    tiling.segShift =
        SgemvTSegPays(tiling.n, tiling.lda, chunkLen) ? CalcSgemvTSegShift(SGEMV_T_TILE_ROWS, chunkLen) : 0U;
    tiling.rowTiles = rowTiles;
    return numChunks * rowTiles;
}

// Returns the slab block count, or 0 to fall back to the generic path.
uint32_t ConfigureSgemvSlab(
    aclblasHandle_t handle, bool isTransN, uint32_t aivCoreNum, SgemvTilingData& tiling, uint8_t*& wsArg)
{
    uint32_t numChunks = std::min(aivCoreNum, CeilDiv<uint32_t>(tiling.n, SGEMV_SLAB_MIN_COLS));
    if (numChunks == 0) {
        numChunks = 1;
    }
    uint32_t chunkLen = CeilDiv<uint32_t>(tiling.n, numChunks);
    numChunks = CeilDiv<uint32_t>(tiling.n, chunkLen);

    if (isTransN) {
        return ConfigureSgemvNSlab(handle, tiling, numChunks, chunkLen, wsArg);
    }
    if (tiling.m <= SGEMV_T_UB_MAX_ROWS) {
        return ConfigureSgemvTSlabSmallM(aivCoreNum, tiling, numChunks, chunkLen);
    }
    return ConfigureSgemvTSlabLargeM(handle, aivCoreNum, tiling, numChunks, chunkLen, wsArg);
}

} // namespace

static aclblasStatus_t ValidateSgemvParams(
    aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, const float* alpha, const float* beta,
    const float* a, const float* x, const float* y)
{
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasSgemv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasSgemv", "invalid m=%d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSgemv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda >= std::max(1, m), OP_LOGE("aclblasSgemv", "invalid lda=%d, m=%d", lda, m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSgemv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT32_MIN, OP_LOGE("aclblasSgemv", "incx must not be INT32_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSgemv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != INT32_MIN, OP_LOGE("aclblasSgemv", "incy must not be INT32_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSgemv", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasSgemv", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    if (m > 0 && n > 0) {
        CHECK_RET(a != nullptr, OP_LOGE("aclblasSgemv", "a must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    // x dimension: trans=N ? n : m
    uint32_t xDim = (trans == ACLBLAS_OP_N) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    if (xDim > 0) {
        CHECK_RET(x != nullptr, OP_LOGE("aclblasSgemv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    // y dimension: trans=N ? m : n
    uint32_t yDim = (trans == ACLBLAS_OP_N) ? static_cast<uint32_t>(m) : static_cast<uint32_t>(n);
    if (yDim > 0) {
        CHECK_RET(y != nullptr, OP_LOGE("aclblasSgemv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSgemv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    // Validate parameters
    aclblasStatus_t st = ValidateSgemvParams(trans, m, n, lda, incx, incy, alpha, beta, a, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Early exit for empty matrix
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    bool isTransN = (trans == ACLBLAS_OP_N);
    uint32_t outDim = isTransN ? static_cast<uint32_t>(m) : static_cast<uint32_t>(n);

    SgemvTilingData tiling = MakeSgemvTiling(m, n, lda, trans, *alpha, *beta, incx, incy);
    uint32_t useNumBlocks = 0;
    uint8_t* wsArg = nullptr;
    if (CanUseSgemvSlab(m, n, lda, incx, incy, *alpha)) {
        useNumBlocks = ConfigureSgemvSlab(h, isTransN, aivCoreNum, tiling, wsArg);
    }
    if (useNumBlocks == 0) {
        useNumBlocks = ConfigureSgemvGeneric(outDim, aivCoreNum, tiling);
    }

    OP_LOGD(
        "aclblasSgemv",
        "tiling: m=%u n=%u lda=%u trans=%u useSlab=%u chunkLen=%u colChunks=%u segShift=%u rowTiles=%u numBlocks=%u "
        "numThreads=%u",
        tiling.m, tiling.n, tiling.lda, tiling.trans, tiling.useSlab, tiling.chunkLen, tiling.colChunks,
        tiling.segShift, tiling.rowTiles, useNumBlocks, tiling.numThreads);
    OP_LOGI("aclblasSgemv", "launching kernel");

    // The T small-m slab path never reads the workspace; keep the kernel
    // descriptor valid by substituting y.
    uint8_t* wsFinal = (wsArg != nullptr) ? wsArg : reinterpret_cast<uint8_t*>(y);
    sgemv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(a)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(y), wsFinal, tiling, useNumBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
