/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cherkx_host.cpp
 * \brief CHERKX Host implementation for ascend950 (DAV_3510)
 *
 *        Phase 0 (AIV × 2): Deinterleave complex A → Ar, Ai and B → Br, Bi
 *        Phase 1 (AIC × 4): 4 real GEMM launches via gemm_kernel_do (4M decomposition)
 *        Phase 2 (AIV): Combine/Scale/Hermitian via cherkx_combine_kernel_do
 *
 *        4M decomposition (trans='N'): C = α·A·B^H + β·C, α complex (cuBLAS contract)
 *          B^H = Br^T - i·Bi^T
 *          A·B^H = (Ar·Br^T + Ai·Bi^T) + i·(Ai·Br^T - Ar·Bi^T)
 *          t1=Ar·Br^T, t2=Ai·Bi^T, t3=Ai·Br^T, t4=Ar·Bi^T
 *          P_r = t1 + t2, P_i = t3 - t4
 *          Cr = αr·P_r − αi·P_i, Ci = αr·P_i + αi·P_r  (αi==0 keeps Cr=αr·P_r, Ci=αr·P_i)
 *
 *        4M decomposition (trans='T'/'C'): C = α·A^H·B + β·C
 *          A^H = Ar^T - i·Ai^T
 *          A^H·B = (Ar^T·Br + Ai^T·Bi) + i·(Ar^T·Bi - Ai^T·Br)
 *          t1=Ar^T·Br, t2=Ai^T·Bi, t3=Ar^T·Bi, t4=Ai^T·Br
 *          P_r/P_i and complex-alpha scaling as above
 *
 *        Key differences from cherk:
 *        - A != B (dual-matrix input): Phase 0 calls deinterleave twice
 *        - alpha is complex (cherk has real alpha): the combine kernel applies
 *          the full complex cross-scaling when alphaImag != 0
 *        - Workspace has 4 real matrices (Ar, Ai, Br, Bi) instead of 2
 *        - GEMM operands come from different sources (A's parts and B's parts)
 *        - Diagonal imaginary part is theoretically non-zero (A != B), must
 *          be explicitly forced to 0 by ZeroDiagonal in combine kernel
 *
 *        Validation order and scalar handling follow the aclblasSsymm terminal
 *        pattern: early enum/dim checks → n==0 quick return → ld and
 *        scalar-pointer validation → pointer-location probe with mixed-mode
 *        rejection → host dereference or device D2H read-back → conditional
 *        A/B/C non-null checks based on the resolved scalar true values.
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cherkx_kernel.h"
#include "gemm/arch35/gemm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/syrk_host_utils.h"

static const char* OP_TAG = "aclblasCherkx";

// ==========================================================================
//  Parameter validation order: handle → enums → dims, then after the
//  n==0 quick return: ld → scalar pointers
// ==========================================================================
static aclblasStatus_t CheckCherkxEarlyParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k)
{
    CHECK_RET(
        handle != nullptr, OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE(OP_TAG, "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_C,
        OP_LOGE(OP_TAG, "trans must be OP_N(111) or OP_C(113), got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Leading-dimension checks (step 3a of the validation order): trans=N requires
// lda/ldb >= max(1,n), trans=C requires max(1,k); ldc >= max(1,n) always.
// Extracted from ValidateCherkxParams only to keep the NBNC within limits;
// the per-check order and return codes are bit-identical to the inline form.
static aclblasStatus_t ValidateCherkxLeadingDims(
    aclblasOperation_t trans, int n, int k,
    int lda, int ldb, int ldc)
{
    if (trans == ACLBLAS_OP_N) {
        CHECK_RET(
            lda >= std::max(1, n),
            OP_LOGE(OP_TAG, "lda must be >= max(1,n) for trans=N, got lda=%d n=%d", lda, n);
            return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(
            ldb >= std::max(1, n),
            OP_LOGE(OP_TAG, "ldb must be >= max(1,n) for trans=N, got ldb=%d n=%d", ldb, n);
            return ACLBLAS_STATUS_INVALID_VALUE);
    } else {
        CHECK_RET(
            lda >= std::max(1, k),
            OP_LOGE(OP_TAG, "lda must be >= max(1,k) for trans=C, got lda=%d k=%d", lda, k);
            return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(
            ldb >= std::max(1, k),
            OP_LOGE(OP_TAG, "ldb must be >= max(1,k) for trans=C, got ldb=%d k=%d", ldb, k);
            return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(
        ldc >= std::max(1, n),
        OP_LOGE(OP_TAG, "ldc must be >= max(1,n), got ldc=%d n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Validate leading dimensions and scalar pointers (after the n==0 quick return,
// so n > 0 is guaranteed here). A/B/C are checked conditionally in the entry
// after the scalar true values are resolved (BLAS: alpha==0 or k==0 frees A/B,
// beta==0 frees C).
static aclblasStatus_t ValidateCherkxParams(
    aclblasOperation_t trans, int n, int k,
    int lda, int ldb, int ldc,
    const aclblasComplex* alpha, const float* beta)
{
    aclblasStatus_t ldSt = ValidateCherkxLeadingDims(trans, n, k, lda, ldb, ldc);
    if (ldSt != ACLBLAS_STATUS_SUCCESS) {
        return ldSt;
    }
    CHECK_RET(
        alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Scalar resolution (three-step pattern)
// ==========================================================================

// Staging read of alpha (complex, 8B) and beta (real, 4B) from device memory:
// two async D2H copies followed by a single stream synchronization, keeping
// the "two memcpyAsync + one sync" sequence mandated for cherkx. Mixed types
// (complex alpha + real beta) rule out the CBlas3ReadScalarsFromDevice<T>
// template (T=aclblasComplex would read 8B from a 4B beta scalar; T=float
// would truncate alpha's imaginary part), so this operator-local wrapper
// mirrors the ReadCher2kScalars signature precedent.
static aclblasStatus_t ReadCherkxScalarsFromDevice(
    const aclblasComplex* alpha, const float* beta,
    aclblasComplex& alphaVal, float& betaVal,
    aclrtStream stream)
{
    aclError aclRet = aclrtMemcpyAsync(&alphaVal, sizeof(aclblasComplex), alpha,
        sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtMemcpyAsync alpha D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtMemcpyAsync(&betaVal, sizeof(float), beta, sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtMemcpyAsync beta D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtSynchronizeStream for D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ResolveCherkxScalars(
    const aclblasComplex* alpha, const float* beta,
    aclblasComplex& alphaVal, float& betaVal,
    aclrtStream stream)
{
    bool alphaIsDevice = false;
    bool betaIsDevice = false;
    aclblasStatus_t st = CheckPtrLocation(alpha, &alphaIsDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "alpha pointer location query failed, st=%d", static_cast<int>(st));
        return st;
    }
    st = CheckPtrLocation(beta, &betaIsDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "beta pointer location query failed, st=%d", static_cast<int>(st));
        return st;
    }
    if (alphaIsDevice != betaIsDevice) {
        OP_LOGE(OP_TAG, "alpha and beta must be both host or both device pointers");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alphaIsDevice) {
        return ReadCherkxScalarsFromDevice(alpha, beta, alphaVal, betaVal, stream);
    }
    alphaVal = *alpha;
    betaVal = *beta;
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  GEMM tiling computation (reuses gemm pattern)
//  cherkx: lda and ldb come from A and B respectively. After deinterleave,
//  both Ar/Ai and Br/Bi are tightly packed with stride = logicalRows (= physRows),
//  so the GEMM uses logicalRows for both lda and ldb.
// ==========================================================================
static GemmTilingData CalCherkxGemmTilingData(
    int n, int k, int lda, int ldb, int tempLdc,
    aclblasOperation_t transa, aclblasOperation_t transb)
{
    GemmTilingData tiling{};
    tiling.m = n;  // CHERKX: m = n (C is n×n)
    tiling.n = n;  // CHERKX: n = n
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
    // The GEMM kernel doesn't consume alpha/beta (no post-process); all
    // scaling (including the complex-alpha cross-scaling) happens in combine.
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
        if (nb < 1) nb = 1;
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
static CherkxCombineTilingData CalCherkxCombineTilingData(
    uint32_t usedAivCoreNum, uint32_t n, uint32_t ldc, uint32_t tempLdc,
    uint8_t isAlphaZero, uint8_t isKZero, uint8_t isBetaZero,
    uint8_t uploMode, float alphaReal, float alphaImag, float betaVal)
{
    CherkxCombineTilingData tiling{};
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempLdc = tempLdc;
    tiling.rowsPerCore = CeilDiv<uint32_t>(n, usedAivCoreNum);
    tiling.alphaReal = alphaReal;
    tiling.alphaImag = alphaImag;
    tiling.betaVal = betaVal;
    tiling.uploMode = uploMode;
    tiling.isAlphaZero = isAlphaZero;
    tiling.isKZero = isKZero;
    tiling.isBetaZero = isBetaZero;
    return tiling;
}

// ==========================================================================
//  PrepareCherkxWorkspace — compute workspace size and ensure allocation
//  cherkx: workspace = 4 temp + 4 real (Ar, Ai, Br, Bi)
// ==========================================================================
static aclblasStatus_t PrepareCherkxWorkspace(
    _aclblas_handle* h, uint32_t n, uint32_t k,
    aclblasOperation_t trans, bool skipTemp,
    uint32_t& aicCoreNum, uint32_t& aivCoreNum, uint32_t& usedAivCoreNum,
    uint32_t& tempLdc, uint32_t& logicalRows, uint32_t& physCols,
    size_t& realBytes, size_t& tempBytes)
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

    // Compute workspace size:
    //   4 temp (tempLdc×n×4B each) + 4 real (logicalRows×physCols×4B each)
    // A and B have the same dimensions, so all 4 real matrices have the same size.
    uint32_t physRows = (trans == ACLBLAS_OP_N) ? n : k;
    physCols = (trans == ACLBLAS_OP_N) ? k : n;
    logicalRows = physRows;  // no transpose in deinterleave
    realBytes = static_cast<size_t>(logicalRows) * physCols * sizeof(float);
    tempBytes = static_cast<size_t>(tempLdc) * n * sizeof(float);
    size_t workspaceNeed = 0;
    if (!skipTemp) {
        // Pre-multiply guard: a single quarter over the limit means the
        // realBytes*4 + tempBytes*4 sum would exceed (or wrap around) the
        // 2 GiB ceiling — reject before the multiplication can wrap.
        constexpr size_t kWsQuarter = ACLBLAS_MAX_WORKSPACE_SIZE / 4;
        if (realBytes > kWsQuarter || tempBytes > kWsQuarter) {
            OP_LOGE(OP_TAG, "workspace need exceeds limit, realBytes=%zu, tempBytes=%zu", realBytes, tempBytes);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        workspaceNeed = realBytes * 4 + tempBytes * 4;  // 4 real + 4 temp
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
//  RunCherkxGemmPhase — Phase 0+1: 2× AIV deinterleave + 4 GEMM launches
//  cherkx key differences from cherk:
//    - Two deinterleave calls (A → Ar, Ai and B → Br, Bi)
//    - GEMM operands from different sources (A's parts vs B's parts)
// ==========================================================================
static void RunCherkxGemmPhase(
    _aclblas_handle* h,
    const aclblasComplex* A, uint32_t lda,
    const aclblasComplex* B, uint32_t ldb,
    uint32_t physCols, uint32_t logicalRows, size_t realBytes,
    uint32_t n, uint32_t k,
    aclblasOperation_t trans, uint32_t aicCoreNum, uint32_t aivCoreNum, uint32_t tempLdc,
    uint8_t* wsBase, size_t tempBytes,
    uint8_t* dT1, uint8_t* dT2, uint8_t* dT3, uint8_t* dT4)
{
    aclrtStream stream = h->stream;

    // Ar, Ai, Br, Bi allocated in workspace after temp area
    uint8_t* dAr = wsBase + tempBytes * 4;
    uint8_t* dAi = dAr + realBytes;
    uint8_t* dBr = dAi + realBytes;
    uint8_t* dBi = dBr + realBytes;

    // Phase 0: Deinterleave complex A → Ar, Ai and B → Br, Bi via AIV kernel
    CherkxDeinterleaveTilingData deintTilingA{};
    deintTilingA.rows = logicalRows;
    deintTilingA.cols = physCols;
    deintTilingA.lda = lda;
    deintTilingA.rowsPerCore = CeilDiv<uint32_t>(logicalRows, aivCoreNum);
    uint32_t deintBlocks = aivCoreNum;

    CherkxDeinterleaveTilingData deintTilingB{};
    deintTilingB.rows = logicalRows;
    deintTilingB.cols = physCols;
    deintTilingB.lda = ldb;
    deintTilingB.rowsPerCore = CeilDiv<uint32_t>(logicalRows, aivCoreNum);

    OP_LOGI(OP_TAG, "launching deinterleave kernels for A and B: aivBlocks=%u", deintBlocks);
    cherkx_deinterleave_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<GM_ADDR>(dAr),
        reinterpret_cast<GM_ADDR>(dAi), deintTilingA, deintBlocks, stream);

    cherkx_deinterleave_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)),
        reinterpret_cast<GM_ADDR>(dBr),
        reinterpret_cast<GM_ADDR>(dBi), deintTilingB, deintBlocks, stream);

    // Phase 1: 4 real GEMM launches (transa/transb derived from CHERKX trans)
    // gemm_kernel_do(numBlocks, stream, b_ptr, a_ptr, c_ptr, tiling)
    //   computes a·b^T (trans='N') or b^T·a (trans='T')
    //   where b_ptr is the first GM arg, a_ptr is the second.
    aclblasOperation_t transaGemm = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_N : ACLBLAS_OP_T;
    aclblasOperation_t transbGemm = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_T : ACLBLAS_OP_N;

    // After deinterleave, Ar/Ai and Br/Bi are tightly packed with stride = logicalRows.
    // GEMM lda = ldb = logicalRows (same for both A and B sources).
    GemmTilingData cubeTiling = CalCherkxGemmTilingData(
        n, k, logicalRows, logicalRows, tempLdc, transaGemm, transbGemm);
    PrepareCubeTiling(cubeTiling, aicCoreNum);
    cubeTiling.ldc = static_cast<int32_t>(tempLdc);

    uint32_t numBlocks = static_cast<uint32_t>(cubeTiling.usedCoreNum);
    OP_LOGI(OP_TAG, "launching 4 gemm kernels: aicBlocks=%u", numBlocks);

    // 4M GEMM: left matrix from A, right matrix from B (cherkx: A != B)
    // GEMM-1/2: t1=Ar·Br^T, t2=Ai·Bi^T (same for trans='N' and trans='T')
    gemm_kernel_do(numBlocks, stream, dBr, dAr, dT1, cubeTiling);
    gemm_kernel_do(numBlocks, stream, dBi, dAi, dT2, cubeTiling);

    // GEMM-3/4 (P_i): a/b exchanged between trans='N' and trans='T'
    if (trans == ACLBLAS_OP_N) {
        gemm_kernel_do(numBlocks, stream, dBr, dAi, dT3, cubeTiling);  // t3=Ai·Br^T
        gemm_kernel_do(numBlocks, stream, dBi, dAr, dT4, cubeTiling);  // t4=Ar·Bi^T
    } else {
        // trans='T' (mapped from 'C'): t3/t4 a/b swapped
        gemm_kernel_do(numBlocks, stream, dBi, dAr, dT3, cubeTiling);  // t3=Ar^T·Bi
        gemm_kernel_do(numBlocks, stream, dBr, dAi, dT4, cubeTiling);  // t4=Ai^T·Br
    }
}

// ==========================================================================
//  LaunchCherkxKernel — full pipeline (orchestrates 3 sub-functions).
//  Receives the RESOLVED scalar true values (host dereference or device D2H
//  read-back already done by the caller), so all branch decisions here are
//  based on true values, never on pointer-mode guesses.
// ==========================================================================
static aclblasStatus_t LaunchCherkxKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans,
    uint32_t n, uint32_t k,
    float alphaReal, float alphaImag, float betaVal,
    const aclblasComplex* A, uint32_t lda,
    const aclblasComplex* B, uint32_t ldb,
    aclblasComplex* C, uint32_t ldc)
{
    // Map trans='C' → trans='T' (CHERKX: both compute A^H·B)
    if (trans == ACLBLAS_OP_C) {
        trans = ACLBLAS_OP_T;
    }

    // Complex-zero test: a pure-imaginary alpha is NOT zero.
    bool isAlphaZero = (alphaReal == 0.0f && alphaImag == 0.0f);
    bool isKZero = (k == 0);
    bool isBetaZero = (betaVal == 0.0f);
    bool skipTemp = isAlphaZero || isKZero;

    OP_LOGD(OP_TAG, "alpha=(%.6f,%.6f) beta=%.6f isAlphaZero=%d isKZero=%d isBetaZero=%d",
        alphaReal, alphaImag, betaVal, isAlphaZero, isKZero, isBetaZero);

    // cherkx: no skipTemp && beta==1 fast path. Unlike cherk (where A==B and
    // the anti-symmetry guarantees diagonal imag = 0), cherkx has A!=B so the
    // Hermitian constraint requires the diagonal imaginary part to be explicitly
    // zeroed via ZeroDiagonal in the combine kernel — even when C is otherwise
    // unchanged (alpha=0/k=0 and beta=1). The combine kernel handles this by
    // always executing ZeroDiagonal regardless of skipTemp. Also, skipTemp with
    // beta==0 must NOT be replaced by a host-side full-matrix memset: cherkx
    // only writes the uplo triangle, so a memset would destroy the non-uplo
    // region; the kernel's skipTemp&&beta==0 branch writes zeros per uplo
    // column and restores the rest from C_old.

    uint32_t aicCoreNum, aivCoreNum, usedAivCoreNum, tempLdc, logicalRows, physCols;
    size_t realBytes, tempBytes;
    aclblasStatus_t st = PrepareCherkxWorkspace(h, n, k, trans, skipTemp,
        aicCoreNum, aivCoreNum, usedAivCoreNum, tempLdc, logicalRows, physCols,
        realBytes, tempBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // skipTemp: no deinterleave/GEMM, no workspace use — dT1..dT4 stay null and
    // the combine kernel only stores the pointers (its skipTemp paths never
    // read them). Computing dT2..dT4 offsets unconditionally would be pointer
    // arithmetic on a possibly-null workspace base (UB).
    uint8_t* dT1 = nullptr;
    uint8_t* dT2 = nullptr;
    uint8_t* dT3 = nullptr;
    uint8_t* dT4 = nullptr;

    if (!skipTemp) {
        uint8_t* wsBase = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
        dT1 = wsBase;
        dT2 = dT1 + tempBytes;
        dT3 = dT2 + tempBytes;
        dT4 = dT3 + tempBytes;
        RunCherkxGemmPhase(h, A, lda, B, ldb, physCols, logicalRows, realBytes,
            n, k, trans, aicCoreNum, aivCoreNum, tempLdc, wsBase, tempBytes,
            dT1, dT2, dT3, dT4);
    }

    CherkxCombineTilingData combineTiling = CalCherkxCombineTilingData(
        usedAivCoreNum, n, ldc, tempLdc,
        static_cast<uint8_t>(isAlphaZero ? 1 : 0),
        static_cast<uint8_t>(isKZero ? 1 : 0),
        static_cast<uint8_t>(isBetaZero ? 1 : 0),
        static_cast<uint8_t>(uplo), alphaReal, alphaImag, betaVal);

    OP_LOGI(OP_TAG, "launching combine kernel: aivCores=%u", usedAivCoreNum);
    aclrtStream stream = h->stream;
    cherkx_combine_kernel_do(
        reinterpret_cast<GM_ADDR>(dT1), reinterpret_cast<GM_ADDR>(dT2),
        reinterpret_cast<GM_ADDR>(dT3), reinterpret_cast<GM_ADDR>(dT4),
        reinterpret_cast<GM_ADDR>(C),
        combineTiling, usedAivCoreNum, stream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Public API entry
// ==========================================================================
extern "C" aclblasStatus_t aclblasCherkx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc)
{
    // 1. Early checks: handle, enums (INVALID_ENUM), non-negative dims.
    aclblasStatus_t st = CheckCherkxEarlyParams(handle, uplo, trans, n, k);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // 2. Quick return: n==0 — no pointer/ld validation afterwards (BLAS standard).
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 3. Validate leading dimensions and scalar pointers.
    st = ValidateCherkxParams(trans, n, k, lda, ldb, ldc, alpha, beta);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // 4. Resolve the scalar true values before any value-dependent branching
    //    (host dereference, or device D2H read-back; mixed mode rejected).
    auto* h = static_cast<_aclblas_handle*>(handle);
    aclblasComplex alphaVal = {0.0f, 0.0f};
    float betaVal = 0.0f;
    st = ResolveCherkxScalars(alpha, beta, alphaVal, betaVal, h->stream);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // 5. Complex-zero test on alpha (αr==0 && αi==0; pure-imaginary is NOT zero).
    bool isAlphaZero = (alphaVal.real == 0.0f && alphaVal.imag == 0.0f);

    // 6. Conditional non-null checks on the resolved true values.
    // BLAS: if alpha == 0 or k == 0 then A and B need not be valid inputs
    // (n > 0 is guaranteed by the quick return above).
    if (!isAlphaZero && k > 0) {
        CHECK_RET(A != nullptr,
            OP_LOGE(OP_TAG, "A must not be nullptr when alpha != 0 and k > 0");
            return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(B != nullptr,
            OP_LOGE(OP_TAG, "B must not be nullptr when alpha != 0 and k > 0");
            return ACLBLAS_STATUS_INVALID_VALUE);
    }
    // BLAS: if beta == 0 then C does not have to be a valid input.
    if (betaVal != 0.0f) {
        CHECK_RET(C != nullptr,
            OP_LOGE(OP_TAG, "C must not be nullptr when beta != 0");
            return ACLBLAS_STATUS_INVALID_VALUE);
    }
    // beta == 0 and C == nullptr: no output destination, nothing to compute.
    // Returning here avoids launching kernels that would write to a null GM
    // address, which would poison the stream for subsequent synchronizations.
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 7. Full pipeline with the resolved scalars.
    return LaunchCherkxKernel(h, uplo, trans,
        static_cast<uint32_t>(n), static_cast<uint32_t>(k),
        alphaVal.real, alphaVal.imag, betaVal,
        A, static_cast<uint32_t>(lda),
        B, static_cast<uint32_t>(ldb),
        C, static_cast<uint32_t>(ldc));
}
