/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cherk_host.cpp
 * \brief CHERK Host implementation for ascend950 (DAV_3510)
 *
 *        Phase 0 (AIV): Deinterleave complex A → Ar, Ai via cherk_deinterleave_kernel_do
 *        Phase 1 (AIC × 4): 4 real GEMM launches via gemm_kernel_do (4M decomposition)
 *        Phase 2 (AIV): Combine/Scale/Hermitian via cherk_combine_kernel_do
 *
 *        4M decomposition (trans='N'): C = α·A·A^H + β·C
 *          A·A^H = (Ar·Ar^T + Ai·Ai^T) + i·(Ai·Ar^T - Ar·Ai^T)
 *          t1=Ar·Ar^T, t2=Ai·Ai^T, t3=Ai·Ar^T, t4=Ar·Ai^T
 *          Cr = t1 + t2, Ci = t3 - t4
 *
 *        4M decomposition (trans='T'/'C'): C = α·A^H·A + β·C
 *          A^H·A = (Ar^T·Ar + Ai^T·Ai) + i·(Ar^T·Ai - Ai^T·Ar)
 *          t1=Ar^T·Ar, t2=Ai^T·Ai, t3=Ar^T·Ai, t4=Ai^T·Ar
 *          Cr = t1 + t2, Ci = t3 - t4
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cherk_kernel.h"
#include "gemm/arch35/gemm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/syrk_host_utils.h"

static const char* OP_TAG = "aclblasCherk";

// ==========================================================================
//  Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateCherkParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha, const aclblasComplex* A,
    int lda, const float* beta, const aclblasComplex* C, int ldc)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_C,
              OP_LOGE(OP_TAG, "trans must be OP_N(111) or OP_C(113), got %d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    if (trans == ACLBLAS_OP_N) {
        CHECK_RET(lda >= std::max(1, n),
                  OP_LOGE(OP_TAG, "lda must be >= max(1,n) for trans=N, got lda=%d n=%d", lda, n);
                  return ACLBLAS_STATUS_INVALID_VALUE);
    } else {
        CHECK_RET(lda >= std::max(1, k),
                  OP_LOGE(OP_TAG, "lda must be >= max(1,k) for trans=T/C, got lda=%d k=%d", lda, k);
                  return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(ldc >= std::max(1, n), OP_LOGE(OP_TAG, "ldc must be >= max(1,n), got ldc=%d n=%d", ldc, n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(A == nullptr && n > 0 && k > 0), OP_LOGE(OP_TAG, "A must not be nullptr when n>0 and k>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(C == nullptr && n > 0), OP_LOGE(OP_TAG, "C must not be nullptr when n>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  GEMM tiling computation (reuses gemm pattern)
// ==========================================================================
static GemmTilingData CalCherkGemmTilingData(
    int n, int k, int lda, int ldb, int tempLdc, aclblasOperation_t transa, aclblasOperation_t transb)
{
    GemmTilingData tiling{};
    tiling.m = n; // HERK: m = n (C is n×n)
    tiling.n = n; // HERK: n = n
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = tempLdc;
    tiling.cLdc = tempLdc;
    tiling.baseM = GEMM_BASE_M;
    tiling.baseN = GEMM_BASE_N;
    tiling.baseK = GEMM_BASE_K;
    tiling.tileKChunk = GEMM_TILE_K_CHUNK;
    tiling.c0Size = GEMM_C0_SIZE;
    tiling.isTransA = (transa != ACLBLAS_OP_N) ? 1 : 0;
    tiling.isTransB = (transb != ACLBLAS_OP_N) ? 1 : 0;
    // alpha/beta are real for HERK; GEMM kernel doesn't use them (no post-process)
    tiling.alphaReal = 1.0f;
    tiling.alphaImag = 0.0f;
    tiling.betaReal = 0.0f;
    tiling.betaImag = 0.0f;
    tiling.hasBeta = 0;
    return tiling;
}

static void ApplyColMajorSwap(GemmTilingData& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
}

static void CalcMultiCorePartition(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    int32_t mTiles = (tiling.m + tiling.baseM - 1) / tiling.baseM;
    int32_t nTiles = (tiling.n + tiling.baseN - 1) / tiling.baseN;
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = 1;
    int32_t bestUtilization = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= maxCores; mb++) {
        int32_t nb = std::min(nTiles, maxCores / mb);
        if (nb < 1)
            nb = 1;
        int32_t utilization = mb * nb;
        if (utilization > bestUtilization && utilization <= maxCores) {
            bestUtilization = utilization;
            bestMBlocks = mb;
            bestNBlocks = nb;
        }
    }
    if (bestUtilization == 0) {
        bestMBlocks = 1;
        bestNBlocks = 1;
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    tiling.usedCoreNum = bestMBlocks * bestNBlocks;
}

static void PrepareCubeTiling(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    ApplyColMajorSwap(tiling);
    CalcMultiCorePartition(tiling, cubeCoreNum);
}

// ==========================================================================
//  Combine tiling computation
// ==========================================================================
static CherkCombineTilingData CalCombineTilingData(
    uint32_t usedAivCoreNum, uint32_t n, uint32_t ldc, uint32_t tempLdc, uint8_t isAlphaZero, uint8_t isKZero,
    uint8_t isBetaZero, uint8_t uploMode, float alphaVal, float betaVal)
{
    CherkCombineTilingData tiling{};
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempLdc = tempLdc;
    tiling.rowsPerCore = CeilDiv<uint32_t>(n, usedAivCoreNum);
    tiling.alphaVal = alphaVal;
    tiling.betaVal = betaVal;
    tiling.uploMode = uploMode;
    tiling.isAlphaZero = isAlphaZero;
    tiling.isKZero = isKZero;
    tiling.isBetaZero = isBetaZero;
    return tiling;
}

// ==========================================================================
//  PrepareCherkParams — read alpha/beta, compute flags
// ==========================================================================
static aclblasStatus_t ReadCherkScalarsFromDevice(
    const float* alpha, const float* beta, float& alphaVal, float& betaVal, aclrtStream stream)
{
    alphaVal = 0.0f;
    betaVal = 0.0f;
    aclblasStatus_t readRet = ReadAlphaBetaFromDevice(alpha, beta, alphaVal, betaVal, stream, OP_TAG);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t PrepareCherkParams(
    _aclblas_handle* h, aclblasOperation_t& trans, uint32_t k, const float* alpha, const float* beta, float& alphaVal,
    float& betaVal, bool& isAlphaZero, bool& isKZero, bool& isBetaZero, bool& skipTemp)
{
    aclrtStream stream = h->stream;

    // Map trans='C' → trans='T' (HERK: both compute A^H·A)
    if (trans == ACLBLAS_OP_C) {
        trans = ACLBLAS_OP_T;
    }

    // Resolve alpha/beta: accept host or device pointers, reject mixing.
    // Issue #628: previous implementation assumed device pointers and always
    // issued D2H copies, reading garbage when callers passed host scalars.
    bool alphaIsDevice = false;
    bool betaIsDevice = false;
    aclblasStatus_t locRet = CheckPtrLocation(alpha, &alphaIsDevice);
    if (locRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "alpha pointer location query failed, st=%d", static_cast<int>(locRet));
        return locRet;
    }
    locRet = CheckPtrLocation(beta, &betaIsDevice);
    if (locRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "beta pointer location query failed, st=%d", static_cast<int>(locRet));
        return locRet;
    }
    if (alphaIsDevice != betaIsDevice) {
        OP_LOGE(OP_TAG, "alpha and beta must be both host or both device pointers");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alphaIsDevice) {
        aclblasStatus_t readRet = ReadCherkScalarsFromDevice(alpha, beta, alphaVal, betaVal, stream);
        if (readRet != ACLBLAS_STATUS_SUCCESS) {
            return readRet;
        }
    } else {
        alphaVal = *alpha;
        betaVal = *beta;
    }

    isAlphaZero = (alphaVal == 0.0f);
    isKZero = (k == 0);
    isBetaZero = (betaVal == 0.0f);
    skipTemp = isAlphaZero || isKZero;

    OP_LOGD(
        OP_TAG, "alpha=%.6f beta=%.6f isAlphaZero=%d isKZero=%d isBetaZero=%d", alphaVal, betaVal, isAlphaZero, isKZero,
        isBetaZero);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  PrepareCherkWorkspace — compute workspace size and ensure allocation
// ==========================================================================
static aclblasStatus_t PrepareCherkWorkspace(
    _aclblas_handle* h, uint32_t n, uint32_t k, aclblasOperation_t trans, bool skipTemp, uint32_t& aicCoreNum,
    uint32_t& aivCoreNum, uint32_t& usedAivCoreNum, uint32_t& tempLdc, uint32_t& logicalRows, uint32_t& physColsA,
    size_t& arBytes, size_t& tempBytes)
{
    // Get core counts
    aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE(OP_TAG, "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE(OP_TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    usedAivCoreNum = GetUsedAivCoreNum(n, aivCoreNum);
    tempLdc = CeilAlign<uint32_t>(n, GEMM_FRACTAL);

    // Compute workspace size: dAr/dAi (logicalRows×physCols×4B) + dT1..4 (tempLdc×n×4B)
    uint32_t physRowsA = (trans == ACLBLAS_OP_N) ? n : k;
    physColsA = (trans == ACLBLAS_OP_N) ? k : n;
    logicalRows = physRowsA; // no transpose in deinterleave
    arBytes = static_cast<size_t>(logicalRows) * physColsA * sizeof(float);
    tempBytes = static_cast<size_t>(tempLdc) * n * sizeof(float);
    size_t workspaceNeed = 0;
    if (!skipTemp) {
        workspaceNeed = arBytes * 2 + tempBytes * 4;
    }
    // Ensure 512B alignment for workspace
    constexpr size_t GM_ALIGN = 512;
    workspaceNeed = (workspaceNeed + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;

    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, workspaceNeed);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "workspace ensure failed, required=%zu, ret=%d", workspaceNeed, wsRet);
        return wsRet;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  RunCherkGemmPhase — Phase 0+1: AIV deinterleave + 4 GEMM launches
// ==========================================================================
static aclblasStatus_t RunCherkGemmPhase(
    _aclblas_handle* h, const aclblasComplex* A, uint32_t lda, uint32_t physColsA, uint32_t logicalRows, size_t arBytes,
    uint32_t n, uint32_t k, aclblasOperation_t trans, uint32_t aicCoreNum, uint32_t aivCoreNum, uint32_t tempLdc,
    uint8_t* wsBase, size_t tempBytes, uint8_t* dT1, uint8_t* dT2, uint8_t* dT3, uint8_t* dT4)
{
    aclrtStream stream = h->stream;

    // Ar, Ai allocated in workspace after temp area
    uint8_t* dAr = wsBase + tempBytes * 4;
    uint8_t* dAi = dAr + arBytes;

    // Phase 0: Deinterleave complex A → Ar, Ai via AIV kernel
    CherkDeinterleaveTilingData deintTiling{};
    deintTiling.rows = logicalRows;
    deintTiling.cols = physColsA;
    deintTiling.lda = lda;
    deintTiling.rowsPerCore = CeilDiv<uint32_t>(logicalRows, aivCoreNum);
    uint32_t deintBlocks = aivCoreNum;

    OP_LOGI(OP_TAG, "launching deinterleave kernel: aivBlocks=%u", deintBlocks);
    cherk_deinterleave_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)), reinterpret_cast<GM_ADDR>(dAr),
        reinterpret_cast<GM_ADDR>(dAi), deintTiling, deintBlocks, stream);

    // Phase 1: 4 real GEMM launches (transa/transb derived from HERK trans)
    aclblasOperation_t transaGemm = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_N : ACLBLAS_OP_T;
    aclblasOperation_t transbGemm = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_T : ACLBLAS_OP_N;

    GemmTilingData cubeTiling = CalCherkGemmTilingData(n, k, logicalRows, logicalRows, tempLdc, transaGemm, transbGemm);
    PrepareCubeTiling(cubeTiling, aicCoreNum);
    cubeTiling.ldc = static_cast<int32_t>(tempLdc);

    uint32_t numBlocks = static_cast<uint32_t>(cubeTiling.usedCoreNum);
    OP_LOGI(OP_TAG, "launching 4 gemm kernels: aicBlocks=%u", numBlocks);

    // 4M GEMM: t1=Ar·Ar^T, t2=Ai·Ai^T (symmetric, transpose-safe)
    // t3/t4 operand order swapped to compensate for fixpipe transpose
    // gemm_kernel_do(a,b,c) stores (a·b)^T, so passing (Ar,Ai,dT3) stores Ai·Ar^T
    gemm_kernel_do(numBlocks, stream, dAr, dAr, dT1, cubeTiling);
    gemm_kernel_do(numBlocks, stream, dAi, dAi, dT2, cubeTiling);
    if (trans == ACLBLAS_OP_N) {
        gemm_kernel_do(numBlocks, stream, dAr, dAi, dT3, cubeTiling); // t3=Ai·Ar^T
        gemm_kernel_do(numBlocks, stream, dAi, dAr, dT4, cubeTiling); // t4=Ar·Ai^T
    } else {
        gemm_kernel_do(numBlocks, stream, dAi, dAr, dT3, cubeTiling); // t3=Ar^T·Ai
        gemm_kernel_do(numBlocks, stream, dAr, dAi, dT4, cubeTiling); // t4=Ai^T·Ar
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  LaunchCherkKernel — full pipeline (orchestrates 3 sub-functions)
// ==========================================================================
static aclblasStatus_t LaunchCherkKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k, const float* alpha,
    const aclblasComplex* A, uint32_t lda, const float* beta, aclblasComplex* C, uint32_t ldc)
{
    float alphaVal, betaVal;
    bool isAlphaZero, isKZero, isBetaZero, skipTemp;
    aclblasStatus_t st =
        PrepareCherkParams(h, trans, k, alpha, beta, alphaVal, betaVal, isAlphaZero, isKZero, isBetaZero, skipTemp);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Fast path: alpha==0 or k==0 (skipTemp) and beta==1 → C unchanged.
    // The beta==0 case falls through to the combine kernel, which zeroes the
    // uplo triangle and restores the non-uplo part from C_old.
    if (skipTemp && betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aicCoreNum, aivCoreNum, usedAivCoreNum, tempLdc, logicalRows, physColsA;
    size_t arBytes, tempBytes;
    st = PrepareCherkWorkspace(
        h, n, k, trans, skipTemp, aicCoreNum, aivCoreNum, usedAivCoreNum, tempLdc, logicalRows, physColsA, arBytes,
        tempBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint8_t* wsBase = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* dT1 = wsBase;
    uint8_t* dT2 = dT1 + tempBytes;
    uint8_t* dT3 = dT2 + tempBytes;
    uint8_t* dT4 = dT3 + tempBytes;

    if (!skipTemp) {
        st = RunCherkGemmPhase(
            h, A, lda, physColsA, logicalRows, arBytes, n, k, trans, aicCoreNum, aivCoreNum, tempLdc, wsBase, tempBytes,
            dT1, dT2, dT3, dT4);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }

    CherkCombineTilingData combineTiling = CalCombineTilingData(
        usedAivCoreNum, n, ldc, tempLdc, static_cast<uint8_t>(isAlphaZero ? 1 : 0),
        static_cast<uint8_t>(isKZero ? 1 : 0), static_cast<uint8_t>(isBetaZero ? 1 : 0), static_cast<uint8_t>(uplo),
        alphaVal, betaVal);

    OP_LOGI(OP_TAG, "launching combine kernel: aivCores=%u", usedAivCoreNum);
    aclrtStream stream = h->stream;
    cherk_combine_kernel_do(
        reinterpret_cast<GM_ADDR>(dT1), reinterpret_cast<GM_ADDR>(dT2), reinterpret_cast<GM_ADDR>(dT3),
        reinterpret_cast<GM_ADDR>(dT4), reinterpret_cast<GM_ADDR>(C), combineTiling, usedAivCoreNum, stream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Public API entry
// ==========================================================================
extern "C" aclblasStatus_t aclblasCherk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const aclblasComplex* A, int lda, const float* beta, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);

    aclblasStatus_t st = ValidateCherkParams(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);

    // n=0 fast path
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchCherkKernel(
        h, uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha, A, static_cast<uint32_t>(lda), beta,
        C, static_cast<uint32_t>(ldc));
}
