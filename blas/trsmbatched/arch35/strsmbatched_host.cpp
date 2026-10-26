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
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "strsmbatched_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

namespace {
constexpr uint32_t AUX_MIN_ELEMS_PER_BLOCK = 1024;
constexpr uint32_t BLOCKED_THRESHOLD = 128;
constexpr uint32_t SIMT_BLOCKED_N_THRESHOLD = 256;

static uint32_t CalcAuxNumBlocks(uint64_t totalElems, uint32_t aivCoreNum)
{
    if (totalElems == 0) return 1;
    uint64_t blocks = (totalElems + AUX_MIN_ELEMS_PER_BLOCK - 1) / AUX_MIN_ELEMS_PER_BLOCK;
    if (blocks > aivCoreNum) blocks = aivCoreNum;
    if (blocks == 0) blocks = 1;
    return static_cast<uint32_t>(blocks);
}

// Panel size must not exceed MAX_BLOCK_NB (128) in kernel.cpp.
// 64 for small matrices (fits in single UB block), 128 for larger (blocked path).
static uint32_t ChoosePanelSize(uint32_t m)
{
    if (m <= 64) return 64;
    return 128;
}

// K chunk size for GEMM L1 double-buffer. Aligned to FP32_C0=8 (L0 load granularity).
// Smaller chunks reduce L1 pressure; larger chunks improve L0 reuse.
static uint32_t ChooseTileK(uint32_t k)
{
    if (k <= 8) return 8;
    if (k <= 32) return 32;
    if (k <= 64) return 64;
    return 128;
}

// GEMM tile MN selection: pick the largest tile that yields >= aicCoreNum tiles
// (to saturate all Cube cores), falling back to the tile with most tiles otherwise.
static void ChooseGemmTileMN(uint32_t gemmM, uint32_t gemmN, uint32_t aicCoreNum,
    uint32_t& tileM, uint32_t& tileN)
{
    const uint32_t candidates[4] = {128, 64, 32, 16};
    uint32_t bestM = 16;
    uint32_t bestN = 16;
    uint64_t bestTiles = 0;
    bool found = false;
    for (uint32_t i = 0; i < 4; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t tm = candidates[i];
            uint32_t tn = candidates[j];
            uint64_t tiles = CeilDiv<uint64_t>(gemmM, tm) * CeilDiv<uint64_t>(gemmN, tn);
            if (tiles >= aicCoreNum) {
                if (!found || static_cast<uint64_t>(tm) * tn > static_cast<uint64_t>(bestM) * bestN) {
                    bestM = tm;
                    bestN = tn;
                    found = true;
                }
            } else if (!found && tiles > bestTiles) {
                bestTiles = tiles;
                bestM = tm;
                bestN = tn;
            }
        }
    }
    tileM = bestM;
    tileN = bestN;
}

static TrsmbatchedGemmTilingData CalcGemmTiling(uint32_t gemmM, uint32_t gemmN, uint32_t bs,
    uint32_t lda, uint32_t ldb, uint64_t aOffset, uint64_t bOffset, uint32_t tempRowStride,
    uint32_t aicCoreNum)
{
    TrsmbatchedGemmTilingData t{};
    t.m = gemmM;
    t.n = gemmN;
    t.k = bs;
    t.lda = lda;
    t.ldb = ldb;
    t.aOffset = aOffset;
    t.bOffset = bOffset;
    ChooseGemmTileMN(gemmM, gemmN, aicCoreNum, t.tileM, t.tileN);
    t.tileKChunk = ChooseTileK(bs);

    constexpr uint32_t L1_SIZE_BYTES = 512 * 1024;
    constexpr uint32_t L1_BUF_NUM_LOCAL = 2;
    constexpr uint32_t FRACTAL_LOCAL = 16;
    constexpr uint32_t FP32_C0_LOCAL = 8;
    // Reserve 10% of L1 as safety margin for alignment overhead and metadata.
    constexpr uint32_t L1_SAFE_RATIO_NUM = 9;
    constexpr uint32_t L1_SAFE_RATIO_DEN = 10;
    uint32_t aSideL1 = CeilAlign<uint32_t>(t.tileM, FRACTAL_LOCAL)
        * CeilAlign<uint32_t>(t.tileKChunk, FP32_C0_LOCAL) * 4u;
    uint32_t bSideL1 = CeilAlign<uint32_t>(t.tileKChunk, FP32_C0_LOCAL)
        * CeilAlign<uint32_t>(t.tileN, FRACTAL_LOCAL) * 4u;
    while (L1_BUF_NUM_LOCAL * (aSideL1 + bSideL1) > L1_SIZE_BYTES * L1_SAFE_RATIO_NUM / L1_SAFE_RATIO_DEN && t.tileKChunk > FP32_C0_LOCAL) {
        t.tileKChunk /= 2;
        t.tileKChunk = CeilAlign<uint32_t>(std::max<uint32_t>(t.tileKChunk, FP32_C0_LOCAL), FP32_C0_LOCAL);
        aSideL1 = CeilAlign<uint32_t>(t.tileM, FRACTAL_LOCAL)
            * CeilAlign<uint32_t>(t.tileKChunk, FP32_C0_LOCAL) * 4u;
        bSideL1 = CeilAlign<uint32_t>(t.tileKChunk, FP32_C0_LOCAL)
            * CeilAlign<uint32_t>(t.tileN, FRACTAL_LOCAL) * 4u;
    }

    t.tempRowStride = tempRowStride;
    return t;
}

static aclblasStatus_t ValidateTrsmbatchedEnums(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) {
        OP_LOGE("aclblasStrsmBatched", "side must be LEFT(141) or RIGHT(142), got %d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasStrsmBatched", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE("aclblasStrsmBatched", "trans must be OP_N(111), OP_T(112) or OP_C(113), got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) {
        OP_LOGE("aclblasStrsmBatched", "diag must be NON_UNIT(131) or UNIT(132), got %d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Scalar-parameter validation only (enums, m/n, alpha, batchCount, lda/ldb):
// every check here is a pure host-scalar condition with no dereference of the
// A/B device pointer arrays, so it must run even when batchCount == 0 (cuBLAS
// error-condition semantics are unconditional). The A/B array pointer checks
// live in the entry after the batchCount == 0 no-op gate.
static aclblasStatus_t ValidateTrsmbatchedParams(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha, int lda, int ldb, int batchCount)
{
    aclblasStatus_t st = ValidateTrsmbatchedEnums(side, uplo, trans, diag);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    if (m < 0 || n < 0) {
        OP_LOGE("aclblasStrsmBatched", "m and n must be >= 0, got m=%d, n=%d", m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasStrsmBatched", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchCount < 0) {
        OP_LOGE("aclblasStrsmBatched", "batchCount must be >= 0, got %d", batchCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (static_cast<size_t>(batchCount) > (SIZE_MAX / sizeof(float*))) {
        OP_LOGE("aclblasStrsmBatched", "batchCount too large: %d", batchCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int k = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (lda < std::max(1, k)) {
        OP_LOGE("aclblasStrsmBatched", "lda must be >= max(1, k), got lda=%d, k=%d", lda, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldb < std::max(1, m)) {
        OP_LOGE("aclblasStrsmBatched", "ldb must be >= max(1, m), got ldb=%d, m=%d", ldb, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ---------- Common helpers ----------

// Scale B by alpha. Uses exact comparison: alpha is a user-provided scalar,
// not a computed result, so exact == 1.0f is the correct check to skip the no-op case.
static void ScaleBByAlpha(_aclblas_handle* h, float* B, float alpha,
    uint32_t mU32, uint32_t nU32, int32_t ldb, uint32_t aivCoreNum)
{
    if (alpha == 1.0f) return;
    uint32_t scaleBlocks = std::min(aivCoreNum, nU32);
    if (scaleBlocks == 0) scaleBlocks = 1;
    OP_LOGD("aclblasStrsmBatched", "scale_kernel: m=%u n=%u alpha=%f blocks=%u",
        mU32, nU32, alpha, scaleBlocks);
    trsmbatched_scale_kernel_do(reinterpret_cast<uint8_t*>(B), alpha, mU32, nU32,
        ldb, scaleBlocks, h->stream);
}

// Launch transpose kernel with standard block calculation.
static void LaunchTranspose(_aclblas_handle* h, float* src, float* dst,
    uint32_t rows, uint32_t cols, int32_t ldIn, int32_t ldOut, uint32_t aivCoreNum)
{
    uint32_t transBlocks = std::min(aivCoreNum, rows);
    if (transBlocks == 0) transBlocks = 1;
    OP_LOGD("aclblasStrsmBatched", "transpose: rows=%u cols=%u blocks=%u", rows, cols, transBlocks);
    trsmbatched_transpose_kernel_do(reinterpret_cast<uint8_t*>(src), reinterpret_cast<uint8_t*>(dst),
        rows, cols, ldIn, ldOut, transBlocks, h->stream);
}

// ---------- SIMT-only path helpers ----------

static aclblasStatus_t TrsmbatchedLeftSimtPath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, int lda, int ldb, float alpha, const float* A, float* B,
    uint32_t aivCoreNum)
{
    uint32_t mU32 = static_cast<uint32_t>(m);
    uint32_t nU32 = static_cast<uint32_t>(n);

    ScaleBByAlpha(h, B, alpha, mU32, nU32, static_cast<int32_t>(ldb), aivCoreNum);

    // Use panel kernel with panelStart=0, panelSize=m for SIMT-only full solve
    TrsmbatchedPanelTilingData pt{};
    pt.uplo = static_cast<uint32_t>(uplo);
    pt.trans = static_cast<uint32_t>(trans);
    pt.diag = static_cast<uint32_t>(diag);
    pt.m = mU32;
    pt.n = nU32;
    pt.lda = static_cast<int32_t>(lda);
    pt.ldb = static_cast<int32_t>(ldb);
    pt.panelStart = 0;
    pt.panelSize = mU32;
    uint32_t panelBlocks = std::min(aivCoreNum, nU32);
    if (panelBlocks == 0) panelBlocks = 1;
    OP_LOGD("aclblasStrsmBatched", "launching panel_kernel (simt path): m=%u n=%u uplo=%d trans=%d diag=%d blocks=%u",
        mU32, nU32, static_cast<int>(uplo), static_cast<int>(trans), static_cast<int>(diag), panelBlocks);
    trsmbatched_panel_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(B),
        pt, panelBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t TrsmbatchedRightSimtPath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, int lda, int ldb, float alpha, const float* A, float* B,
    uint32_t aivCoreNum)
{
    // Scale B by alpha first (before transpose, since scalar mult is layout-independent)
    ScaleBByAlpha(h, B, alpha, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<int32_t>(ldb), aivCoreNum);

    // Transpose B -> Bt
    int newLdb = std::max(1, n);
    size_t btBytes = static_cast<size_t>(newLdb) * static_cast<size_t>(m) * sizeof(float);
    if (btBytes > GetEffectiveWorkspaceSize(h)) {
        aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, btBytes);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            OP_LOGE("aclblasStrsmBatched", "workspace not enough for right simt path: need=%zu, have=%zu",
                btBytes, GetEffectiveWorkspaceSize(h));
            return wsRet;
        }
    }
    uint8_t* btDev = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    LaunchTranspose(h, B, reinterpret_cast<float*>(btDev),
        static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<int32_t>(ldb), static_cast<int32_t>(newLdb), aivCoreNum);

    // Solve left-side: op(A') * Bt = Bt with alpha=1 (already applied)
    // Right TRSM: X * op(A) = alpha*B => transpose => op(A)^T * X^T = alpha*B^T
    aclblasOperation_t fTrans = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_T : ACLBLAS_OP_N;
    OP_LOGD("aclblasStrsmBatched", "right simt path: left solve with uplo=%d trans=%d m=%d n=%d",
        static_cast<int>(uplo), static_cast<int>(fTrans), n, m);
    aclblasStatus_t solveRet = TrsmbatchedLeftSimtPath(h, uplo, fTrans, diag,
        n, m, lda, newLdb, 1.0f, A, reinterpret_cast<float*>(btDev), aivCoreNum);
    if (solveRet != ACLBLAS_STATUS_SUCCESS) {
        return solveRet;
    }

    // Transpose Bt -> B
    LaunchTranspose(h, reinterpret_cast<float*>(btDev), B,
        static_cast<uint32_t>(n), static_cast<uint32_t>(m),
        static_cast<int32_t>(newLdb), static_cast<int32_t>(ldb), aivCoreNum);

    return ACLBLAS_STATUS_SUCCESS;
}

// ---------- Blocked path helpers ----------

struct TrsmbatchedBlockedCtx {
    uint32_t mU32;
    uint32_t nU32;
    uint32_t ldaU32;
    uint32_t ldbU32;
    uint32_t panelBs;
    uint32_t aicCoreNum;
    uint32_t noTransTempStride;
    uint32_t transTempStride;
    uint32_t nAligned;
    uint8_t* tempDev;
    uint8_t* aWsDev;
    uint8_t* bWsDev;
    bool isUpper;
    bool isTrans;
};

static void TrsmbatchedTrailingUpdateNoTrans(
    _aclblas_handle* h, uint32_t nU32, uint32_t k, uint32_t bs, uint32_t mC,
    uint64_t aOffset, uint64_t axpyOffset, uint32_t ldaU32, uint32_t ldbU32,
    uint32_t noTransTempStride, uint32_t aicCoreNum, uint32_t aivCoreNum,
    const float* A, float* B, uint8_t* tempDev)
{
    TrsmbatchedGemmTilingData gt = CalcGemmTiling(nU32, mC, bs, ldbU32, ldaU32,
        k, aOffset, noTransTempStride, aicCoreNum);
    uint64_t gemmTileCount = static_cast<uint64_t>((nU32 + gt.tileM - 1) / gt.tileM)
                           * static_cast<uint64_t>((mC + gt.tileN - 1) / gt.tileN);
    uint32_t gemmBlocks = static_cast<uint32_t>(std::min<uint64_t>(gemmTileCount, aicCoreNum));
    if (gemmBlocks == 0) gemmBlocks = 1;
    OP_LOGD("aclblasStrsmBatched", "NoTrans GEMM: m=%u n=%u k=%u tiles=%lu blocks=%u",
        gt.m, gt.n, gt.k, static_cast<unsigned long>(gemmTileCount), gemmBlocks);
    trsmbatched_gemm_kernel_do(reinterpret_cast<uint8_t*>(B),
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), tempDev, gt, gemmBlocks, h->stream);

    TrsmbatchedAxpyTilingData at{};
    at.m = mC;
    at.n = nU32;
    at.ldb = ldbU32;
    at.tempRowStride = noTransTempStride;
    at.bOffset = axpyOffset;
    OP_LOGD("aclblasStrsmBatched", "NoTrans AXPY_TRANS: m=%u n=%u bOffset=%lu", at.m, at.n,
        static_cast<unsigned long>(at.bOffset));
    trsmbatched_axpy_trans_kernel_do(reinterpret_cast<uint8_t*>(B), tempDev, at,
        CalcAuxNumBlocks(static_cast<uint64_t>(mC) * nU32, aivCoreNum), h->stream);
}

static void TrsmbatchedTrailingUpdateTrans(
    _aclblas_handle* h, uint32_t nU32, uint32_t k, uint32_t bs, uint32_t mC,
    uint64_t aOffset, uint64_t axpyOffset, uint32_t ldaU32, uint32_t ldbU32,
    uint32_t transTempStride, uint32_t nAligned, uint32_t aicCoreNum, uint32_t aivCoreNum,
    const float* A, float* B, uint8_t* tempDev, uint8_t* aWsDev, uint8_t* bWsDev)
{
    uint32_t bsAligned = CeilAlign<uint32_t>(bs, 8u);
    OP_LOGD("aclblasStrsmBatched", "Trans extract_a: mC=%u bs=%u bsAligned=%u lda=%u aOffset=%lu",
        mC, bs, bsAligned, ldaU32, static_cast<unsigned long>(aOffset));
    trsmbatched_extract_a_kernel_do(reinterpret_cast<uint8_t*>(const_cast<float*>(A)), aWsDev,
        mC, bs, bsAligned, ldaU32, aOffset,
        CalcAuxNumBlocks(static_cast<uint64_t>(mC) * bs, aivCoreNum), h->stream);

    OP_LOGD("aclblasStrsmBatched", "Trans extract_b: bs=%u n=%u nAligned=%u ldb=%u bOffset=%lu",
        bs, nU32, nAligned, ldbU32, static_cast<unsigned long>(k));
    trsmbatched_extract_b_kernel_do(reinterpret_cast<uint8_t*>(B), bWsDev,
        bs, nU32, nAligned, ldbU32, k,
        CalcAuxNumBlocks(static_cast<uint64_t>(bs) * nU32, aivCoreNum), h->stream);

    TrsmbatchedGemmTilingData gt = CalcGemmTiling(mC, nU32, bs, bsAligned, nAligned,
        0, 0, transTempStride, aicCoreNum);
    uint64_t gemmTileCount = static_cast<uint64_t>((mC + gt.tileM - 1) / gt.tileM)
                           * static_cast<uint64_t>((nU32 + gt.tileN - 1) / gt.tileN);
    uint32_t gemmBlocks = static_cast<uint32_t>(std::min<uint64_t>(gemmTileCount, aicCoreNum));
    if (gemmBlocks == 0) gemmBlocks = 1;
    OP_LOGD("aclblasStrsmBatched", "Trans GEMM: m=%u n=%u k=%u blocks=%u", gt.m, gt.n, gt.k, gemmBlocks);
    trsmbatched_gemm_kernel_do(aWsDev, bWsDev, tempDev, gt, gemmBlocks, h->stream);

    TrsmbatchedAxpyTilingData at{};
    at.m = mC;
    at.n = nU32;
    at.ldb = ldbU32;
    at.tempRowStride = gt.tempRowStride;
    at.bOffset = axpyOffset;
    OP_LOGD("aclblasStrsmBatched", "Trans AXPY: m=%u n=%u bOffset=%lu", at.m, at.n,
        static_cast<unsigned long>(at.bOffset));
    trsmbatched_axpy_kernel_do(reinterpret_cast<uint8_t*>(B), tempDev, at,
        CalcAuxNumBlocks(static_cast<uint64_t>(mC) * nU32, aivCoreNum), h->stream);
}

static void TrsmbatchedBlockedStep(
    _aclblas_handle* h, const TrsmbatchedBlockedCtx& ctx, uint32_t step, uint32_t numSteps,
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int lda, int ldb, const float* A, float* B, uint32_t aivCoreNum)
{
    bool forward = (!ctx.isTrans && !ctx.isUpper) || (ctx.isTrans && ctx.isUpper);
    uint32_t k = forward ? step * ctx.panelBs : (numSteps - 1 - step) * ctx.panelBs;
    uint32_t bs = std::min(ctx.panelBs, ctx.mU32 - k);

    // Step 1: Panel solve
    TrsmbatchedPanelTilingData pt{};
    pt.uplo = static_cast<uint32_t>(uplo);
    pt.trans = static_cast<uint32_t>(trans);
    pt.diag = static_cast<uint32_t>(diag);
    pt.m = ctx.mU32;
    pt.n = ctx.nU32;
    pt.lda = static_cast<int32_t>(lda);
    pt.ldb = static_cast<int32_t>(ldb);
    pt.panelStart = k;
    pt.panelSize = bs;
    uint32_t panelBlocks = std::min(aivCoreNum, ctx.nU32);
    if (panelBlocks == 0) panelBlocks = 1;
    OP_LOGD("aclblasStrsmBatched", "Panel: step=%u k=%u bs=%u blocks=%u", step, k, bs, panelBlocks);
    trsmbatched_panel_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(B),
        pt, panelBlocks, h->stream);

    // Step 2+3: Trailing update
    uint32_t mC = 0;
    uint64_t aOffset = 0;
    uint64_t axpyOffset = 0;
    if (forward) {
        mC = ctx.mU32 - k - bs;
        if (mC == 0) return;
        // forward: trailing matrix is rows [k+bs, m). aOffset points to the A sub-block.
        // NoTrans: A[k+bs, k] (row-major: row=k+bs, col=k => offset = (k+bs) + k*lda)
        // Trans:   A[k, k+bs] (row-major: row=k, col=k+bs => offset = k + (k+bs)*lda)
        aOffset = ctx.isTrans ? (static_cast<uint64_t>(k) + static_cast<uint64_t>(k + bs) * ctx.ldaU32)
                              : (static_cast<uint64_t>(k + bs) + static_cast<uint64_t>(k) * ctx.ldaU32);
        axpyOffset = k + bs;
    } else {
        mC = k;
        if (mC == 0) return;
        // backward: trailing matrix is rows [0, k). aOffset points to the A sub-block.
        // NoTrans: A[0, k] (row-major: row=0, col=k => offset = 0 + k*lda = k*lda)
        // Trans:   A[k, 0] (row-major: row=k, col=0 => offset = k + 0*lda = k)
        aOffset = ctx.isTrans ? static_cast<uint64_t>(k) : static_cast<uint64_t>(k) * ctx.ldaU32;
        axpyOffset = 0;
    }

    if (!ctx.isTrans) {
        TrsmbatchedTrailingUpdateNoTrans(h, ctx.nU32, k, bs, mC, aOffset, axpyOffset,
            ctx.ldaU32, ctx.ldbU32, ctx.noTransTempStride, ctx.aicCoreNum, aivCoreNum,
            A, B, ctx.tempDev);
    } else {
        TrsmbatchedTrailingUpdateTrans(h, ctx.nU32, k, bs, mC, aOffset, axpyOffset,
            ctx.ldaU32, ctx.ldbU32, ctx.transTempStride, ctx.nAligned, ctx.aicCoreNum, aivCoreNum,
            A, B, ctx.tempDev, ctx.aWsDev, ctx.bWsDev);
    }
}

static aclblasStatus_t TrsmbatchedLeftBlockedPath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, int lda, int ldb, float alpha, const float* A, float* B,
    uint32_t aivCoreNum, uint32_t aicCoreNum, uint8_t* wsOverride = nullptr)
{
    uint32_t mU32 = static_cast<uint32_t>(m);
    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t panelBs = ChoosePanelSize(mU32);

    // Step 0: B *= alpha
    ScaleBByAlpha(h, B, alpha, mU32, nU32, static_cast<int32_t>(ldb), aivCoreNum);

    TrsmbatchedBlockedCtx ctx{};
    ctx.mU32 = mU32;
    ctx.nU32 = nU32;
    ctx.ldaU32 = static_cast<uint32_t>(lda);
    ctx.ldbU32 = static_cast<uint32_t>(ldb);
    ctx.panelBs = panelBs;
    ctx.aicCoreNum = aicCoreNum;
    ctx.isUpper = (uplo == ACLBLAS_UPPER);
    ctx.isTrans = (trans != ACLBLAS_OP_N);
    ctx.noTransTempStride = CeilAlign<uint32_t>(mU32, 8u);
    ctx.transTempStride = CeilAlign<uint32_t>(nU32, 8u);
    ctx.nAligned = CeilAlign<uint32_t>(nU32, 8u);

    size_t tempWs = ctx.isTrans
        ? static_cast<size_t>(mU32) * ctx.transTempStride * sizeof(float)
        : static_cast<size_t>(nU32) * ctx.noTransTempStride * sizeof(float);
    size_t aWs = static_cast<size_t>(mU32) * panelBs * sizeof(float);
    size_t bWs = static_cast<size_t>(panelBs) * ctx.nAligned * sizeof(float);
    size_t totalWsMax = ctx.isTrans ? (tempWs + aWs + bWs) : tempWs;
    if (wsOverride == nullptr) {
        if (totalWsMax > GetEffectiveWorkspaceSize(h)) {
            aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, totalWsMax);
            if (wsRet != ACLBLAS_STATUS_SUCCESS) {
                OP_LOGE("aclblasStrsmBatched", "workspace not enough: need=%zu, have=%zu",
                    totalWsMax, GetEffectiveWorkspaceSize(h));
                return wsRet;
            }
        }
        ctx.tempDev = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    } else {
        ctx.tempDev = wsOverride;
    }
    if (ctx.isTrans) {
        ctx.aWsDev = ctx.tempDev + tempWs;
        ctx.bWsDev = ctx.aWsDev + aWs;
    }

    uint32_t numSteps = CeilDiv<uint32_t>(mU32, panelBs);
    OP_LOGI("aclblasStrsmBatched", "blocked path: m=%u n=%u panelBs=%u steps=%u isTrans=%d",
        mU32, nU32, panelBs, numSteps, static_cast<int>(ctx.isTrans));
    for (uint32_t step = 0; step < numSteps; ++step) {
        TrsmbatchedBlockedStep(h, ctx, step, numSteps, uplo, trans, diag, lda, ldb, A, B, aivCoreNum);
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// Calculate workspace needed for the right-device path's blocked sub-solve.
// Returns total blocked workspace size (also the offset where Bt buffer starts).
static size_t CalcRightBlockedWorkspace(uint32_t mU32, uint32_t nU32, uint32_t panelBs,
    bool isTransAfter)
{
    uint32_t noTransTempStride = CeilAlign<uint32_t>(nU32, 8u);
    uint32_t transTempStride = CeilAlign<uint32_t>(mU32, 8u);
    size_t blockedTempWs = isTransAfter
        ? static_cast<size_t>(nU32) * transTempStride * sizeof(float)
        : static_cast<size_t>(mU32) * noTransTempStride * sizeof(float);
    size_t aWs = static_cast<size_t>(nU32) * panelBs * sizeof(float);
    uint32_t mAligned = CeilAlign<uint32_t>(mU32, 8u);
    size_t bWs = static_cast<size_t>(panelBs) * mAligned * sizeof(float);
    return isTransAfter ? (blockedTempWs + aWs + bWs) : blockedTempWs;
}

static aclblasStatus_t TrsmbatchedRightDevicePath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, int lda, int ldb, float alpha, const float* A, float* B,
    uint32_t aivCoreNum, uint32_t aicCoreNum)
{
    // Step 0: B *= alpha (before transpose, since scalar mult is layout-independent)
    ScaleBByAlpha(h, B, alpha, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<int32_t>(ldb), aivCoreNum);

    // Transpose B -> Bt
    int newLdb = std::max(1, n);
    size_t btBytes = static_cast<size_t>(newLdb) * static_cast<size_t>(m) * sizeof(float);
    // Right path workspace layout: [blocked_ws] [Bt]
    // The blocked path must NOT overlap with Bt since it reads/writes Bt during the solve.
    uint32_t mU32 = static_cast<uint32_t>(m);
    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t panelBs = ChoosePanelSize(nU32);

    // After transpose: left-side solve with mEff=n, nEff=m
    bool isTransAfter = (trans == ACLBLAS_OP_N);
    size_t blockedWsMax = CalcRightBlockedWorkspace(mU32, nU32, panelBs, isTransAfter);
    size_t totalWsMax = blockedWsMax + btBytes;
    if (totalWsMax > GetEffectiveWorkspaceSize(h)) {
        aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, totalWsMax);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            OP_LOGE("aclblasStrsmBatched", "workspace not enough for right path: need=%zu, have=%zu",
                totalWsMax, GetEffectiveWorkspaceSize(h));
            return wsRet;
        }
    }
    uint8_t* wsBase = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* blockedWsPtr = wsBase;
    uint8_t* btDev = wsBase + blockedWsMax;

    LaunchTranspose(h, B, reinterpret_cast<float*>(btDev),
        mU32, nU32, static_cast<int32_t>(ldb), static_cast<int32_t>(newLdb), aivCoreNum);

    // Solve left-side: op(A') * Bt = Bt with alpha=1 (already applied)
    // Right TRSM: X * op(A) = alpha*B => transpose => op(A)^T * X^T = alpha*B^T
    // The left solve uses original uplo with flipped trans (not flipped uplo).
    // Pass blockedWsPtr to avoid workspace overlap with Bt.
    aclblasOperation_t fTrans = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_T : ACLBLAS_OP_N;
    OP_LOGD("aclblasStrsmBatched", "right path: left solve with uplo=%d trans=%d m=%d n=%d",
        static_cast<int>(uplo), static_cast<int>(fTrans), n, m);
    aclblasStatus_t solveRet = TrsmbatchedLeftBlockedPath(h, uplo, fTrans, diag,
        n, m, lda, newLdb, 1.0f, A, reinterpret_cast<float*>(btDev),
        aivCoreNum, aicCoreNum, blockedWsPtr);
    if (solveRet != ACLBLAS_STATUS_SUCCESS) {
        return solveRet;
    }

    // Transpose Bt -> B
    LaunchTranspose(h, reinterpret_cast<float*>(btDev), B,
        nU32, mU32, static_cast<int32_t>(newLdb), static_cast<int32_t>(ldb), aivCoreNum);

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchTrsmbatchedKernel(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, int lda, int ldb, float alpha,
    const float* A, float* B, uint32_t aivCoreNum, uint32_t aicCoreNum)
{
    // alpha==0 exact comparison is intentional: 0 is a sentinel value from user input,
    // not a computed result, so exact comparison correctly identifies the fast-path case.
    if (alpha == 0.0f) {
        uint32_t nU32 = static_cast<uint32_t>(n);
        uint32_t numBlocks = std::min(aivCoreNum, nU32);
        if (numBlocks == 0) numBlocks = 1;
        OP_LOGI("aclblasStrsmBatched", "alpha==0 fast path, m=%d, n=%d, blocks=%u", m, n, numBlocks);
        trsmbatched_zero_kernel_do(reinterpret_cast<uint8_t*>(B),
            static_cast<uint32_t>(m), nU32, static_cast<int32_t>(ldb), numBlocks, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t mU32 = static_cast<uint32_t>(m);

    if (side == ACLBLAS_SIDE_RIGHT) {
        // Right path: check if n (the effective m for right-side) qualifies for blocked
        uint32_t nU32 = static_cast<uint32_t>(n);
        if (nU32 <= BLOCKED_THRESHOLD) {
            OP_LOGI("aclblasStrsmBatched", "right simt path selected: m=%d, n=%d", m, n);
            return TrsmbatchedRightSimtPath(h, uplo, trans, diag, m, n, lda, ldb, alpha, A, B, aivCoreNum);
        }
        return TrsmbatchedRightDevicePath(h, uplo, trans, diag, m, n, lda, ldb, alpha, A, B,
            aivCoreNum, aicCoreNum);
    }

    // side=Left
    // Path selection logic (same as strsm):
    // - M <= BLOCKED_THRESHOLD && N >= SIMT_BLOCKED_N_THRESHOLD => Blocked path
    // - Otherwise => SIMT-only path
    bool useBlockedPath = (mU32 > BLOCKED_THRESHOLD) ||
                          (mU32 <= BLOCKED_THRESHOLD &&
                           static_cast<uint32_t>(n) >= SIMT_BLOCKED_N_THRESHOLD);
    if (!useBlockedPath) {
        OP_LOGI("aclblasStrsmBatched", "left simt path selected: m=%d, n=%d", m, n);
        return TrsmbatchedLeftSimtPath(h, uplo, trans, diag, m, n, lda, ldb, alpha, A, B, aivCoreNum);
    }

    OP_LOGI("aclblasStrsmBatched", "left blocked path selected: m=%d, n=%d", m, n);
    return TrsmbatchedLeftBlockedPath(h, uplo, trans, diag, m, n, lda, ldb, alpha, A, B,
        aivCoreNum, aicCoreNum);
}

static aclblasStatus_t CopyPtrArraysD2H(
    const float* const A[], float* const B[], int batchCount, bool needA,
    std::vector<const float*>& hAPtrs, std::vector<float*>& hBPtrs)
{
    size_t ptrArrayBytes = static_cast<size_t>(batchCount) * sizeof(float*);
    hAPtrs.assign(batchCount, nullptr);
    hBPtrs.assign(batchCount, nullptr);

    if (needA) {
        aclError aclRet = aclrtMemcpy(hAPtrs.data(), ptrArrayBytes, A, ptrArrayBytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasStrsmBatched", "aclrtMemcpy A D2H failed, ret=%d", static_cast<int>(aclRet));
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    aclError aclRet = aclrtMemcpy(hBPtrs.data(), ptrArrayBytes, B, ptrArrayBytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasStrsmBatched", "aclrtMemcpy B D2H failed, ret=%d", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t RunTrsmbatchedBatchLoop(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, int lda, int ldb, float alpha,
    const std::vector<const float*>& hAPtrs, const std::vector<float*>& hBPtrs,
    bool needA, int batchCount, uint32_t aivCoreNum, uint32_t aicCoreNum)
{
    for (int i = 0; i < batchCount; ++i) {
        const float* A = needA ? hAPtrs[i] : nullptr;
        float* B = hBPtrs[i];
        if (B == nullptr) {
            OP_LOGE("aclblasStrsmBatched", "B[%d] contains nullptr", i);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (needA && A == nullptr) {
            OP_LOGE("aclblasStrsmBatched", "A[%d] contains nullptr when alpha != 0", i);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        OP_LOGD("aclblasStrsmBatched", "batch %d/%d: A=%p B=%p", i, batchCount,
            static_cast<const void*>(A), static_cast<void*>(B));
        aclblasStatus_t ret = LaunchTrsmbatchedKernel(h, side, uplo, trans, diag, m, n, lda, ldb,
            alpha, A, B, aivCoreNum, aicCoreNum);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            OP_LOGE("aclblasStrsmBatched", "batch %d failed with status %d", i, static_cast<int>(ret));
            return ret;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* const A[], int lda,
    float* const B[], int ldb,
    int batchCount)
{
    auto* h = handle;
    CHECK_RET(h != nullptr,
        OP_LOGE("aclblasStrsmBatched", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    // Scalar-parameter validation first (enums, m/n, alpha, batchCount,
    // lda/ldb): cuBLAS error-condition semantics are unconditional, so an
    // invalid uplo/trans/diag/m/n/alpha/ld combined with batchCount == 0
    // still reports INVALID_VALUE.
    aclblasStatus_t st = ValidateTrsmbatchedParams(side, uplo, trans, diag, m, n, alpha,
        lda, ldb, batchCount);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // cuBLAS batched-family convention (gemvBatched/gemvStridedBatched treat
    // batchCount < 0 as the only batchCount error condition): zero batches is
    // a legal no-op. All scalar validation above has already run; the gate
    // sits at the last moment before the A/B device pointer arrays would be
    // checked or dereferenced, so they may be null when there is nothing to
    // compute.
    if (batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (B == nullptr) {
        OP_LOGE("aclblasStrsmBatched", "B must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (*alpha != 0.0f && A == nullptr) {
        OP_LOGE("aclblasStrsmBatched", "A must not be nullptr when alpha != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0,
        OP_LOGE("aclblasStrsmBatched", "GetAivCoreCount failed"); return ACLBLAS_STATUS_INTERNAL_ERROR);
    uint32_t aicCoreNum = GetAicCoreCount();
    CHECK_RET(aicCoreNum > 0,
        OP_LOGE("aclblasStrsmBatched", "GetAicCoreCount returned 0"); return ACLBLAS_STATUS_INTERNAL_ERROR);

    OP_LOGI("aclblasStrsmBatched",
        "enter: side=%d uplo=%d trans=%d diag=%d m=%d n=%d lda=%d ldb=%d batchCount=%d aivCores=%u aicCores=%u",
        static_cast<int>(side), static_cast<int>(uplo), static_cast<int>(trans), static_cast<int>(diag),
        m, n, lda, ldb, batchCount, aivCoreNum, aicCoreNum);

    bool needA = (A != nullptr && *alpha != 0.0f);
    std::vector<const float*> hAPtrs;
    std::vector<float*> hBPtrs;
    st = CopyPtrArraysD2H(A, B, batchCount, needA, hAPtrs, hBPtrs);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    return RunTrsmbatchedBatchLoop(h, side, uplo, trans, diag, m, n, lda, ldb,
        *alpha, hAPtrs, hBPtrs, needA, batchCount, aivCoreNum, aicCoreNum);
}
