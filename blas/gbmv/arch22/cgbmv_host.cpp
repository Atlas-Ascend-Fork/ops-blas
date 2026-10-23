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
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "cgbmv_tiling_data.h"
#include "cgbmv_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

namespace {

constexpr uint32_t CGBMV_TINY_MAX_OUT = 8U;
constexpr uint64_t CGBMV_TINY_MAX_NNZ = 1024U;
constexpr uint32_t DIAGONAL_VECTOR_CHUNK = 256U;
constexpr uint32_t SEGMENT_GRAIN = 4U;
constexpr uint32_t TALL_MATRIX_COLUMN_LIMIT = 256U;
constexpr uint32_t ROW_DOT_MIN_BAND = 128U;
constexpr uint32_t ROW_DOT_MIN_N = 1500U;
constexpr uint32_t ROW_DOT_WINDOW_LIMIT = 1200U;

aclblasStatus_t ValidateCgbmvParams(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, const aclblasComplex* A,
    const aclblasComplex* x, const aclblasComplex* y, bool alphaZero)
{
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasCgbmv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        m >= 0 && n >= 0, OP_LOGE("aclblasCgbmv", "invalid m=%d or n=%d", m, n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(
        kl >= 0 && kl <= m - 1, OP_LOGE("aclblasCgbmv", "invalid kl=%d, m=%d", kl, m);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ku >= 0 && ku <= n - 1, OP_LOGE("aclblasCgbmv", "invalid ku=%d, n=%d", ku, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        lda >= kl + ku + 1, OP_LOGE("aclblasCgbmv", "invalid lda=%d, kl=%d, ku=%d", lda, kl, ku);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasCgbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasCgbmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (!alphaZero) {
        CHECK_RET(A != nullptr, OP_LOGE("aclblasCgbmv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(x != nullptr, OP_LOGE("aclblasCgbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(y != nullptr, OP_LOGE("aclblasCgbmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

uint64_t CountBandElements(uint32_t m, uint32_t n, uint32_t kl, uint32_t ku)
{
    uint64_t nnz = 0;
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t lo = (j > ku) ? (j - ku) : 0U;
        const uint32_t hi = (m - 1U < j + kl) ? (m - 1U) : (j + kl);
        if (lo <= hi) {
            nnz += hi - lo + 1U;
        }
    }
    return nnz;
}

constexpr uint32_t TILE_ROWS_BASE = 32;
constexpr uint32_t SWEEP_BAND_LIMIT = 128U;
constexpr uint32_t TILE_ROWS_WIDE_MIN = 48;
constexpr uint32_t TILE_ROWS_WIDE_MAX = 64;
constexpr int64_t PANEL_LANES = 64;
constexpr int64_t PANELS_PER_CORE_BALANCE = 8;

constexpr uint64_t COST_PANEL_COVERED_ROW = 14U;
constexpr uint64_t COST_PANEL_COVERED_BASE = 14U;
constexpr uint64_t COST_PANEL_MASKED_ROW = 8U;
constexpr uint64_t COST_PANEL_TAIL_ROW = 3U;
constexpr uint64_t COST_BETA_IDENT = 6U;
constexpr uint64_t COST_BETA_SCALED = 9U;
constexpr uint64_t COST_CORE_BASE_ROW = 10U;
constexpr uint64_t COST_CORE_BASE_GROUP = 7U;

int64_t RowsPerCoreAtWidth(int64_t length, uint32_t width, uint32_t cores)
{
    const int64_t stride = static_cast<int64_t>(width) * cores;
    return ((length - 1) / stride + 1) * width;
}

uint32_t BalanceWidth(int64_t length, uint32_t cores)
{
    if (cores == 0) {
        return TILE_ROWS_BASE;
    }
    uint32_t width = TILE_ROWS_BASE;
    int64_t rows = RowsPerCoreAtWidth(length, width, cores);
    for (uint32_t candidate = TILE_ROWS_WIDE_MIN; candidate <= TILE_ROWS_WIDE_MAX;
         candidate += TILE_ROWS_WIDE_MAX - TILE_ROWS_WIDE_MIN) {
        if (RowsPerCoreAtWidth(length, candidate, cores) <= rows) {
            width = candidate;
            rows = RowsPerCoreAtWidth(length, candidate, cores);
        }
    }
    return width;
}

int64_t CoveredPanelSpan(int64_t begin, int64_t end, int64_t row, int64_t active, int64_t kl, int64_t ku)
{
    const int64_t spanLow = std::max(begin, row + active - 1 - kl);
    const int64_t spanHigh = std::min(end - PANEL_LANES, row + ku - PANEL_LANES + 1);
    const int64_t firstCovered = (spanLow - begin + PANEL_LANES - 1) / PANEL_LANES;
    const int64_t lastCovered = (spanHigh - begin) / PANEL_LANES;
    return lastCovered < firstCovered ? 0 : lastCovered - firstCovered + 1;
}

uint64_t ScoreOutputTile(const CgbmvTilingData& t, int64_t row, uint32_t rows)
{
    const int64_t count = std::min<int64_t>(rows, t.lenY - row);
    const uint64_t maskWordLoad = (rows + 32U - 1U) / 32U;
    const uint64_t laneGroupLoads = static_cast<uint64_t>((2 * rows + PANEL_LANES - 1) / PANEL_LANES) +
                                    static_cast<uint64_t>((2 * count + PANEL_LANES - 1) / PANEL_LANES);
    uint64_t score = laneGroupLoads;
    if (t.betaR != 0.0F || t.betaI != 0.0F) {
        score += (t.betaR == 1.0F && t.betaI == 0.0F) ? COST_BETA_IDENT : COST_BETA_SCALED;
    }
    const int64_t active = std::min<int64_t>(count, t.n + t.kl - row);
    if (active <= 0) {
        return score;
    }
    const int64_t begin = std::max<int64_t>(0, row - t.kl);
    const int64_t end = std::min<int64_t>(t.n, row + active + t.ku);
    if (begin >= end) {
        return score;
    }
    const uint64_t panelCount = static_cast<uint64_t>((end - begin + PANEL_LANES - 1) / PANEL_LANES);
    const uint64_t coveredCount = static_cast<uint64_t>(CoveredPanelSpan(begin, end, row, active, t.kl, t.ku));
    score += panelCount * (COST_PANEL_COVERED_ROW * rows + COST_PANEL_COVERED_BASE) +
             (panelCount - coveredCount) * (COST_PANEL_MASKED_ROW * rows + maskWordLoad);
    if ((end - begin) % PANEL_LANES != 0) {
        score += COST_PANEL_TAIL_ROW * rows + maskWordLoad;
    }
    return score;
}

uint64_t CriticalCoreScore(const CgbmvTilingData& t, uint32_t rows, uint32_t cores)
{
    if (rows == 0 || cores == 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    if (t.lenY <= 0) {
        return 0;
    }
    const int64_t tiles = (t.lenY + rows - 1) / rows;
    const uint32_t blocks = static_cast<uint32_t>(std::min<int64_t>(cores, tiles));
    uint64_t critical = 0;
    for (uint32_t core = 0; core < blocks; ++core) {
        uint64_t score = COST_CORE_BASE_ROW * rows +
                         COST_CORE_BASE_GROUP * static_cast<uint64_t>((2 * rows + PANEL_LANES - 1) / PANEL_LANES);
        for (int64_t tile = core; tile < tiles; tile += blocks) {
            score += ScoreOutputTile(t, tile * rows, rows);
        }
        critical = std::max(critical, score);
    }
    return critical;
}

uint32_t PickNPanelWidth(const CgbmvTilingData& t, uint32_t cores)
{
    if (cores == 0) {
        return TILE_ROWS_BASE;
    }
    const int64_t tilesAtBase = (t.lenY + TILE_ROWS_BASE - 1) / TILE_ROWS_BASE;
    if (tilesAtBase > PANELS_PER_CORE_BALANCE * static_cast<int64_t>(cores)) {
        return BalanceWidth(t.lenY, cores);
    }
    uint32_t width = TILE_ROWS_BASE;
    uint64_t best = CriticalCoreScore(t, width, cores);
    for (uint32_t candidate = TILE_ROWS_WIDE_MIN; candidate <= TILE_ROWS_WIDE_MAX; candidate += 16U) {
        const uint64_t score = CriticalCoreScore(t, candidate, cores);
        if (score < best) {
            width = candidate;
            best = score;
        }
    }
    OP_LOGD(
        "aclblasCgbmv", "N panel rows=%u worstCost=%llu (cores=%u)", width, static_cast<unsigned long long>(best),
        cores);
    return width;
}
uint32_t PickSegmentLen(uint32_t lenY, uint32_t cores)
{
    if (cores <= 1U) {
        return CeilAlign<uint32_t>(lenY, SEGMENT_GRAIN);
    }
    return CeilAlign<uint32_t>(CeilDiv<uint32_t>(lenY, cores), SEGMENT_GRAIN);
}

uint32_t RouteOutputTile(const CgbmvTilingData& t, uint32_t cores)
{
    if (t.kl == 0 && t.ku == 0) {
        return std::min<uint32_t>(DIAGONAL_VECTOR_CHUNK, (t.lenY + 7U) / 8U * 8U);
    }
    const uint32_t bandW = t.kl + t.ku + 1U;
    if (t.mode == 0U && bandW > SWEEP_BAND_LIMIT) {
        return PickNPanelWidth(t, cores);
    }
    return PickSegmentLen(t.lenY, cores);
}

} // namespace

void LaunchCgbmv(
    const aclblasComplex* A, const aclblasComplex* x, aclblasComplex* y, const CgbmvTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    cgbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), tiling, numBlocks,
        stream);
}

void FillTiling(
    CgbmvTilingData& tiling, const aclblasComplex* alpha, const aclblasComplex* beta, const aclblasComplex* A,
    const aclblasComplex* x, aclblasComplex* y, int m, int n, int kl, int ku, int lda, aclblasOperation_t trans,
    int incx, int incy)
{
    tiling.a = reinterpret_cast<uint64_t>(A);
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.y = reinterpret_cast<uint64_t>(y);
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.kl = static_cast<uint32_t>(kl);
    tiling.ku = static_cast<uint32_t>(ku);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.trans = (trans == ACLBLAS_OP_N) ? 0U : ((trans == ACLBLAS_OP_T) ? 1U : 2U);
    tiling.alphaR = alpha->real;
    tiling.alphaI = alpha->imag;
    tiling.betaR = beta->real;
    tiling.betaI = beta->imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
}

bool ShouldRouteRowDot(const CgbmvTilingData& tiling, uint32_t aivCoreNum, uint32_t bandW, bool isTrans)
{
    return isTrans && bandW >= ROW_DOT_MIN_BAND && tiling.n >= ROW_DOT_MIN_N &&
           bandW + CeilAlign<uint32_t>(CeilDiv<uint32_t>(tiling.n, aivCoreNum), 8U) <= ROW_DOT_WINDOW_LIMIT;
}

bool TryFastRoute(
    CgbmvTilingData& tiling, const aclblasComplex* A, const aclblasComplex* x, aclblasComplex* y, void* stream,
    bool isTrans, bool alphaZero, uint32_t outLen, uint32_t aivCoreNum)
{
    const uint32_t bandW = tiling.kl + tiling.ku + 1U;
    const bool strided = (tiling.incx != 1) || (tiling.incy != 1);
    if (alphaZero) {
        OP_LOGD("aclblasCgbmv", "route: beta-scale m=%u n=%u trans=%u", tiling.m, tiling.n, tiling.trans);
        LaunchCgbmv(A, x, y, tiling, 1U, stream);
        return true;
    }
    if (!strided && outLen <= CGBMV_TINY_MAX_OUT &&
        CountBandElements(tiling.m, tiling.n, tiling.kl, tiling.ku) <= CGBMV_TINY_MAX_NNZ) {
        OP_LOGD("aclblasCgbmv", "route: tiny m=%u n=%u kl=%u ku=%u", tiling.m, tiling.n, tiling.kl, tiling.ku);
        LaunchCgbmv(A, x, y, tiling, 1U, stream);
        return true;
    }
    if (!strided && ShouldRouteRowDot(tiling, aivCoreNum, bandW, isTrans)) {
        uint32_t coreNum = std::min(aivCoreNum, CeilDiv<uint32_t>(outLen, 8U));
        tiling.path = CGBMV_PATH_ROW_DOT;
        tiling.outPerCore = CeilAlign<uint32_t>(CeilDiv<uint32_t>(outLen, coreNum), 8U);
        tiling.coreNum = CeilDiv<uint32_t>(outLen, tiling.outPerCore);
        OP_LOGD("aclblasCgbmv", "route: row-dot lenY=%u cores=%u", outLen, tiling.coreNum);
        LaunchCgbmv(A, x, y, tiling, tiling.coreNum, stream);
        return true;
    }
    if (strided) {
        OP_LOGD("aclblasCgbmv", "route: scalar strided m=%u n=%u", tiling.m, tiling.n);
        LaunchCgbmv(A, x, y, tiling, 1U, stream);
        return true;
    }
    return false;
}

void LaunchSweepPath(
    CgbmvTilingData& tiling, const aclblasComplex* y, uint32_t aivCoreNum, uint32_t outLen, bool isTrans)
{
    tiling.mode = isTrans ? 1U : 0U;
    tiling.lenY = outLen;
    const bool yContigAligned = ((reinterpret_cast<uintptr_t>(y) & 31U) == 0U);
    uint32_t cores = yContigAligned ? aivCoreNum : 1U;
    if (tiling.mode == 0U && tiling.n <= TALL_MATRIX_COLUMN_LIMIT && tiling.lenY > 1024) {
        cores = std::min<uint32_t>(cores, aivCoreNum / 2U);
    }
    tiling.outputTile = RouteOutputTile(tiling, cores);
    tiling.blockCount = CeilDiv<uint32_t>(outLen, tiling.outputTile);
}

aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasCgbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasCgbmv", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasCgbmv", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    const bool alphaZero = (alpha->real == 0.0f && alpha->imag == 0.0f);
    const bool betaOne = (beta->real == 1.0f && beta->imag == 0.0f);

    aclblasStatus_t st = ValidateCgbmvParams(trans, m, n, kl, ku, lda, incx, incy, A, x, y, alphaZero);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (alphaZero && betaOne) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgbmv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const bool isTrans = (trans != ACLBLAS_OP_N);
    const uint32_t outLen = isTrans ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);

    CgbmvTilingData tiling{};
    FillTiling(tiling, alpha, beta, A, x, y, m, n, kl, ku, lda, trans, incx, incy);

    if (TryFastRoute(tiling, A, x, y, h->stream, isTrans, alphaZero, outLen, aivCoreNum)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    LaunchSweepPath(tiling, y, aivCoreNum, outLen, isTrans);
    OP_LOGD(
        "aclblasCgbmv", "route: sweep mode=%u segLen=%u blocks=%u lenY=%u", tiling.mode, tiling.outputTile,
        tiling.blockCount, outLen);
    LaunchCgbmv(A, x, y, tiling, tiling.blockCount, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
