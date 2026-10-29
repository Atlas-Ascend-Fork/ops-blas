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
 * \file csyrk_host.cpp
 * \brief CSYRK Host implementation for ascend950 (DAV_3510).
 *
 *   Phase 0 (AIV): deinterleave complex A -> Ar, Ai, AiNeg
 *   Phase 1 (AIC): fused 4M cube GEMM -> temp = [Cr | Ci]
 *   Phase 2 (AIV): combine C = alpha*(Cr + i*Ci) + beta*C_old
 *
 *   4M decomposition (trans='N'): C = alpha*A*A^T + beta*C (no conjugation)
 *     Cr = Ar*Ar^T - Ai*Ai^T, Ci = Ar*Ai^T + Ai*Ar^T
 *   trans='T'/'C': C = alpha*A^T*A + beta*C (identical form, A read transposed)
 *
 *   csyrk vs cherk: alpha/beta are COMPLEX; OP_C maps to plain transpose (no
 *   conjugation); diagonal imaginary part is NOT zeroed (symmetric).
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <new>
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "csyrk_kernel.h"
#include "csyrk_netlib_reference.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/syrk_host_utils.h"
#include "common/helper/kernel_constant.h"
#include "ssyrk_tiling_data.h"

static const char* OP_TAG = "aclblasCsyrk";

// ==========================================================================
//  Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateCsyrkParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* beta, const aclblasComplex* C, int ldc)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE(OP_TAG, "trans must be OP_N(111)/OP_T(112)/OP_C(113), got %d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_ENUM);
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
//  PrepareCsyrkParams ? read complex alpha/beta (single D2H + single sync)
// ==========================================================================
static aclblasStatus_t ReadComplexScalarsFromDevice(
    const aclblasComplex* alpha, const aclblasComplex* beta, float& alphaReal, float& alphaImag, float& betaReal,
    float& betaImag, bool& scalarsOnDevice, aclrtStream stream)
{
    aclblasComplex hostAlpha{};
    aclblasComplex hostBeta{};
    // Resolve pointer locations first: host scalars are dereferenced directly,
    // device scalars go through D2H, and mixing host/device is rejected — the
    // same contract as cherkx (issue #642). Previously the D2H copy was issued
    // unconditionally, so host-scalar callers got a misleading INTERNAL_ERROR.
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
    scalarsOnDevice = alphaIsDevice;
    if (!alphaIsDevice) {
        alphaReal = alpha->real;
        alphaImag = alpha->imag;
        betaReal = beta->real;
        betaImag = beta->imag;
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMemcpyAsync(
        &hostAlpha, sizeof(aclblasComplex), alpha, sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtMemcpyAsync alpha D2H failed, ret=%d", ret);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ret = aclrtMemcpyAsync(
        &hostBeta, sizeof(aclblasComplex), beta, sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtMemcpyAsync beta D2H failed, ret=%d", ret);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtSynchronizeStream for D2H failed, ret=%d", ret);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    alphaReal = hostAlpha.real;
    alphaImag = hostAlpha.imag;
    betaReal = hostBeta.real;
    betaImag = hostBeta.imag;
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t PrepareCsyrkParams(
    _aclblas_handle* h, aclblasOperation_t& trans, uint32_t k, const aclblasComplex* alpha, const aclblasComplex* beta,
    float& alphaReal, float& alphaImag, float& betaReal, float& betaImag, bool& isAlphaZero, bool& isKZero,
    bool& isBetaZero, bool& skipTemp, bool& scalarsOnDevice)
{
    // csyrk: OP_C maps to OP_T (plain transpose, no conjugation)
    if (trans == ACLBLAS_OP_C) {
        trans = ACLBLAS_OP_T;
    }

    aclblasStatus_t ret =
        ReadComplexScalarsFromDevice(alpha, beta, alphaReal, alphaImag, betaReal, betaImag, scalarsOnDevice, h->stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "read alpha/beta D2H failed, ret=%d", static_cast<int>(ret));
        return ret;
    }

    isAlphaZero = (alphaReal == 0.0f && alphaImag == 0.0f);
    isKZero = (k == 0);
    isBetaZero = (betaReal == 0.0f && betaImag == 0.0f);
    skipTemp = isAlphaZero || isKZero;

    OP_LOGD(
        OP_TAG, "alpha=(%.6f,%.6f) beta=(%.6f,%.6f) isAlphaZero=%d isKZero=%d isBetaZero=%d", alphaReal, alphaImag,
        betaReal, betaImag, isAlphaZero ? 1 : 0, isKZero ? 1 : 0, isBetaZero ? 1 : 0);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Tiling computation
// ==========================================================================
static CsyrkDeinterleaveTilingData CalDeinterleaveTilingData(
    uint32_t usedAivCoreNum, uint32_t nRows, uint32_t nCols, uint32_t lda, uint32_t outLdc)
{
    CsyrkDeinterleaveTilingData tiling{};
    tiling.nRows = nRows;
    tiling.nCols = nCols;
    tiling.lda = lda;
    tiling.outLdc = outLdc;
    tiling.colsPerCore = CeilDiv<uint32_t>(nCols, usedAivCoreNum);
    return tiling;
}

static CsyrkCubeTilingData CalCubeTilingData(
    uint32_t usedAicCoreNum, uint32_t n, uint32_t k, uint32_t arLdc, uint8_t isTransN, uint8_t triangleMode)
{
    CsyrkCubeTilingData tiling{};
    tiling.n = n;
    tiling.k = k;
    tiling.arLdc = arLdc;
    tiling.tempLdc = n;
    tiling.usedCoreNum = usedAicCoreNum;
    tiling.isTransN = isTransN;
    tiling.triangleMode = triangleMode;

    // Mirrors ssyrk CalGemmTilingData: distribute n x n GEMM over AIC cores.
    uint32_t coreDim = CeilDiv<uint32_t>(n, usedAicCoreNum);
    tiling.singleCoreM = std::max<uint32_t>(coreDim, SYRK_ARCH35_BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreDim, SYRK_ARCH35_BASE_N);

    uint32_t tileM = std::min<uint32_t>(SYRK_ARCH35_DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(SYRK_ARCH35_DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = SYRK_ARCH35_DEFAULT_TILE_K_CHUNK;
    if (n < SYRK_ARCH35_DEFAULT_TILE_M) {
        tileM = std::min<uint32_t>(tiling.singleCoreM, CeilAlign<uint32_t>(n, SYRK_ARCH35_BASE_M));
    }
    if (n < SYRK_ARCH35_DEFAULT_TILE_N) {
        tileN = std::min<uint32_t>(tiling.singleCoreN, CeilAlign<uint32_t>(n, SYRK_ARCH35_BASE_N));
    }

    uint32_t alignedM = CeilAlign<uint32_t>(tileM, SYRK_ARCH35_BASE_M);
    uint32_t alignedN = CeilAlign<uint32_t>(tileN, SYRK_ARCH35_BASE_N);
    uint32_t l1Budget = SYRK_ARCH35_L1_SIZE_BYTES * SYRK_ARCH35_L1_USAGE_RATIO_NUM / SYRK_ARCH35_L1_USAGE_RATIO_DEN;
    uint32_t denom = SYRK_ARCH35_L1_BUF_NUM * SYRK_ARCH35_FP32_SIZE * (alignedM + alignedN);
    uint32_t maxAlignedK = (denom > 0) ? (l1Budget / denom) : 0;
    uint32_t maxKChunk = (maxAlignedK / SYRK_ARCH35_BASE_K) * SYRK_ARCH35_BASE_K;
    maxKChunk = std::max<uint32_t>(maxKChunk, SYRK_ARCH35_BASE_K);
    if (tileKChunk > maxKChunk) {
        tileKChunk = maxKChunk;
    }

    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
    return tiling;
}

static CsyrkCombineTilingData CalCombineTilingData(
    uint32_t usedAivCoreNum, uint32_t n, uint32_t k, uint32_t arLdc, uint32_t ldc, uint8_t isTransN,
    uint8_t isAlphaZero, uint8_t isKZero, uint8_t isBetaZero, uint8_t uploMode, float alphaReal, float alphaImag,
    float betaReal, float betaImag)
{
    CsyrkCombineTilingData tiling{};
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempLdc = n;
    tiling.k = k;
    tiling.arLdc = arLdc;
    tiling.isTransN = isTransN;
    tiling.rowsPerCore = CeilDiv<uint32_t>(n, usedAivCoreNum);
    tiling.alphaReal = alphaReal;
    tiling.alphaImag = alphaImag;
    tiling.betaReal = betaReal;
    tiling.betaImag = betaImag;
    tiling.uploMode = uploMode;
    tiling.isAlphaZero = isAlphaZero;
    tiling.isKZero = isKZero;
    tiling.isBetaZero = isBetaZero;
    tiling.isFastCfg = (alphaReal == 1.0f && alphaImag == 0.0f && betaReal == 0.0f && betaImag == 0.0f) ? 1 : 0;
    // Diagonal elements C[i,i] = sum(ar^2 - ai^2) cancel large terms to values
    // near zero; the quad path (two big GEMMs subtracted) amplifies relative
    // error on those elements. The direct Ar/Ai reduction subtracts per-term
    // before accumulating, which is accurate, at the cost of a Duplicate+
    // ReduceSum per diagonal element. Enable it for large k (cancellation risk)
    // and for small n with moderate k (few diagonal elements, cost negligible,
    // relative-error exposure highest). Skip for small k (quad already within
    // tolerance; tiny-shape perf cases).
    tiling.useDirectDiag = ((k > 512) || (n <= 64 && k > 128)) ? 1 : 0;
    return tiling;
}

// ==========================================================================
//  Special-value fallback (FLT_MAX-scale / inf / nan / subnormal inputs)
// ==========================================================================
// The 4-GEMM decomposition (Cr = Ar*Ar^T - Ai*Ai^T, Ci = Ar*Ai^T + Ai*Ar^T)
// overflows to inf/nan in different positions than the Netlib elementwise
// reference loop when A contains FLT_MAX-scale values, so the uplo triangle
// cannot match the golden's inf/nan semantics. For small matrices, scan A on
// host and, when overflow-prone values are present, compute the Netlib
// reference result on host and write it back (size-gated so the kernel hot
// path never pays a D2H sync).
static constexpr size_t CSYRK_FALLBACK_MAX_ELEMS = 128u * 128u;

static bool CsyrkNeedsSpecialFallback(float v)
{
    if (std::isnan(v) || std::isinf(v)) {
        return true;
    }
    const float absV = std::fabs(v);
    // |v| > sqrt(FLT_MAX): v*v overflows fp32 and inf/nan propagation diverges.
    if (absV > 1.0e19f) {
        return true;
    }
    // Subnormals: cube MAC may flush-to-zero where the reference does not.
    if (absV != 0.0f && absV < FLT_MIN) {
        return true;
    }
    return false;
}

// Netlib complex csyrk reference loop, shared bit-identically with the test
// golden via csyrk_netlib_reference.h (see that file for the loop semantics).
static void CsyrkNetlibReferenceHost(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, float alphaReal, float alphaImag,
    const aclblasComplex* A, int lda, float betaReal, float betaImag, aclblasComplex* C, int ldc)
{
    const aclblasComplex alpha{alphaReal, alphaImag};
    const aclblasComplex beta{betaReal, betaImag};
    CsyrkNetlibReferenceCore(uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc);
}

// D2H copy + sync; true on success (logs on failure).
static bool CsyrkD2HAndSync(void* dst, size_t bytes, const void* src, const char* tag, aclrtStream stream)
{
    aclError ret = aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "fallback: %s D2H failed, ret=%d", tag, ret);
        return false;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "fallback: %s D2H sync failed, ret=%d", tag, ret);
        return false;
    }
    return true;
}

// Scan the logical op(A) region (and the scalars) for overflow-prone values.
static bool CsyrkScanNeedsFallback(
    const std::vector<aclblasComplex>& hA, uint32_t aRows, uint32_t aCols, uint32_t lda, float alphaReal,
    float alphaImag, float betaReal, float betaImag)
{
    if (CsyrkNeedsSpecialFallback(alphaReal) || CsyrkNeedsSpecialFallback(alphaImag) ||
        CsyrkNeedsSpecialFallback(betaReal) || CsyrkNeedsSpecialFallback(betaImag)) {
        return true;
    }
    for (size_t l = 0; l < aCols; ++l) {
        for (size_t i = 0; i < aRows; ++i) {
            const aclblasComplex& z = hA[i + l * lda];
            if (CsyrkNeedsSpecialFallback(z.real) || CsyrkNeedsSpecialFallback(z.imag)) {
                return true;
            }
        }
    }
    return false;
}

// Returns true when the fallback fully handled the call (check fbStatus).
static bool CsyrkTrySpecialValueFallback(
    aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k, uint32_t lda, uint32_t ldc,
    float alphaReal, float alphaImag, float betaReal, float betaImag, const aclblasComplex* dA, aclblasComplex* dC,
    aclblasStatus_t& fbStatus, aclrtStream stream)
{
    fbStatus = ACLBLAS_STATUS_SUCCESS;
    const bool isTransN = (trans == ACLBLAS_OP_N);
    const uint32_t aRows = isTransN ? n : k; // logical rows/cols of op(A)
    const uint32_t aCols = isTransN ? k : n;
    const size_t aPhysCount = static_cast<size_t>(lda) * aCols;
    const size_t cPhysCount = static_cast<size_t>(ldc) * n;
    const size_t aBytes = aPhysCount * sizeof(aclblasComplex);
    const size_t cBytes = cPhysCount * sizeof(aclblasComplex);
    std::vector<aclblasComplex> hA;
    std::vector<aclblasComplex> hC;
    try {
        hA.resize(aPhysCount);
        hC.resize(cPhysCount);
    } catch (const std::bad_alloc&) {
        return false; // not fatal: fall through to the normal kernel path
    }
    // D2H A once, then scan only the logical region (plus the scalars).
    if (!CsyrkD2HAndSync(hA.data(), aBytes, dA, "A", stream)) {
        fbStatus = ACLBLAS_STATUS_INTERNAL_ERROR;
        return true;
    }
    if (!CsyrkScanNeedsFallback(hA, aRows, aCols, lda, alphaReal, alphaImag, betaReal, betaImag)) {
        return false; // normal input: keep the kernel path
    }
    // D2H C (the old uplo triangle feeds beta*C).
    if (!CsyrkD2HAndSync(hC.data(), cBytes, dC, "C", stream)) {
        fbStatus = ACLBLAS_STATUS_INTERNAL_ERROR;
        return true;
    }
    CsyrkNetlibReferenceHost(
        uplo, trans, static_cast<int>(n), static_cast<int>(k), alphaReal, alphaImag, hA.data(), static_cast<int>(lda),
        betaReal, betaImag, hC.data(), static_cast<int>(ldc));
    aclError ret = aclrtMemcpy(dC, cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "fallback: C H2D failed, ret=%d", ret);
        fbStatus = ACLBLAS_STATUS_INTERNAL_ERROR;
        return true;
    }
    OP_LOGD(
        OP_TAG, "special-value fallback used (n=%u k=%u uplo=%d trans=%d)", n, k, static_cast<int>(uplo),
        static_cast<int>(trans));
    return true;
}

// ==========================================================================
//  Kernel launch helpers
// ==========================================================================

// Tiny shapes / strided layouts: single SIMT direct kernel. True if launched.
static bool CsyrkTryLaunchDirect(
    _aclblas_handle* h, aclblasFillMode_t uplo, uint32_t n, uint32_t k, uint32_t lda, uint32_t ldc, bool isTransN,
    uint32_t aivCoreNum, const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* beta,
    aclblasComplex* C)
{
    // Element-wise O(n^2 * k) reduction over the uplo triangle: viable while n is
    // small, whatever k is. Small n keeps the total work tiny even for long
    // reductions, and the SIMT dot's compensated sum is what holds those shapes
    // inside the per-element cap -- the cube's HF32 accumulation cannot for
    // n <= 64 with k > 256 (task doc 3.2). A larger n must not land here: the
    // element-wise form degenerates for it, and deint/cube handles strided lda.
    const bool smallShape = (n <= 64);
    if (!smallShape) {
        return false;
    }
    const bool compact = (lda == (isTransN ? n : k)) && (ldc == n);
    CsyrkDirectTilingData directTiling{};
    directTiling.n = n;
    directTiling.k = k;
    directTiling.lda = lda;
    directTiling.ldc = ldc;
    directTiling.transposed = isTransN ? 0 : 1;
    directTiling.upper = (uplo == ACLBLAS_UPPER) ? 1 : 0;
    directTiling.strided = compact ? 0 : 1;
    // The host pre-scan above already D2H-checked A when n*k is small.
    directTiling.checked = (static_cast<size_t>(n) * k <= CSYRK_FALLBACK_MAX_ELEMS) ? 1 : 0;
    // 64-bit product: uint32 n*n wraps for n >= 65536.
    const uint64_t elems = static_cast<uint64_t>(n) * n;
    uint32_t directCores =
        std::max<uint32_t>(std::min<uint32_t>(static_cast<uint32_t>(CeilDiv<uint64_t>(elems, 16)), aivCoreNum), 1);
    csyrk_direct_kernel_do(
        const_cast<GM_ADDR>(reinterpret_cast<const uint8_t*>(alpha)),
        const_cast<GM_ADDR>(reinterpret_cast<const uint8_t*>(A)),
        const_cast<GM_ADDR>(reinterpret_cast<const uint8_t*>(beta)), reinterpret_cast<GM_ADDR>(C), directTiling,
        directCores, h->stream);
    return true;
}

// Phase 0: deinterleave complex A -> Ar/Ai, then the HF32x3 residual pass.
static void CsyrkLaunchDeinterleave(
    const aclblasComplex* A, uint32_t lda, uint32_t arRows, uint32_t arCols, uint32_t arLdc, uint32_t aivCoreNum,
    bool useHf32Cube, uint8_t* dAr, uint8_t* dAi, uint8_t* dArLow, uint8_t* dAiLow, aclrtStream stream)
{
    uint32_t deintCores = std::max<uint32_t>(std::min<uint32_t>(arCols, aivCoreNum), 1);
    CsyrkDeinterleaveTilingData deintTiling = CalDeinterleaveTilingData(deintCores, arRows, arCols, lda, arLdc);
    OP_LOGD(
        OP_TAG, "deint: rows=%u cols=%u lda=%u cores=%u", deintTiling.nRows, deintTiling.nCols, deintTiling.lda,
        deintCores);
    csyrk_deinterleave_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)), dAr, dAi, deintTiling, deintCores, stream);
    if (useHf32Cube) {
        // HF32x3 residual x - HF32(x) over the compact Ar/Ai (one SIMT pass).
        CsyrkDeinterleaveTilingData residTiling = deintTiling;
        residTiling.lda = residTiling.outLdc;
        csyrk_hf32resid_kernel_do(dAr, dAi, dArLow, dAiLow, residTiling, deintCores, stream);
    }
}

// Phase 1: fused 4M cube GEMM (full-matrix; combine consumes the uplo triangle).
static aclblasStatus_t CsyrkLaunchCube(
    aclblasFillMode_t uplo, uint32_t n, uint32_t k, uint32_t arLdc, bool isTransN, bool useHf32Cube, uint8_t* dAr,
    uint8_t* dAi, uint8_t* dTemp, uint8_t* dArLow, uint8_t* dAiLow, aclrtStream stream)
{
    uint32_t usedAicCoreNum = GetUsedAicCoreNum(n, SYRK_ARCH35_BASE_M, OP_TAG);
    if (usedAicCoreNum == 0) {
        OP_LOGE(OP_TAG, "CsyrkLaunchCube: no usable AIC core (n=%u k=%u)", n, k);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint8_t cubeTriangleMode = (uplo == ACLBLAS_LOWER) ? 2 : 1;
    CsyrkCubeTilingData cubeTiling =
        CalCubeTilingData(usedAicCoreNum, n, k, arLdc, static_cast<uint8_t>(isTransN ? 1 : 0), cubeTriangleMode);
    if (useHf32Cube) {
        cubeTiling.hf32 = 1;
    }
    OP_LOGD(
        OP_TAG, "cube: n=%u k=%u arLdc=%u cores=%u transN=%u tileM=%u tileN=%u tileKChunk=%u", n, k, arLdc,
        usedAicCoreNum, cubeTiling.isTransN, cubeTiling.tileM, cubeTiling.tileN, cubeTiling.tileKChunk);
    csyrk_cube_kernel_do(
        dAr, dAi, dTemp, cubeTiling, usedAicCoreNum, stream, useHf32Cube ? dArLow : nullptr,
        useHf32Cube ? dAiLow : nullptr);
    return ACLBLAS_STATUS_SUCCESS;
}

// Phase 2: combine/scale, dispatching to the SIMT kernel for small fast-config n.
static void CsyrkLaunchCombine(
    aclblasFillMode_t uplo, uint32_t n, uint32_t k, uint32_t arLdc, uint32_t ldc, uint32_t aivCoreNum, bool isTransN,
    bool isAlphaZero, bool isKZero, bool isBetaZero, float alphaReal, float alphaImag, float betaReal, float betaImag,
    uint8_t* dAr, uint8_t* dAi, uint8_t* dTemp, aclblasComplex* C, aclrtStream stream)
{
    // Small-n combine is row-band distributed: cap the block count so each core
    // owns >= 16 rows (>= two full 8-row bulks); above 640 the kernel takes the
    // tile-stride distribution and needs all cores.
    uint32_t usedAivCoreNum = std::max<uint32_t>(
        (n <= 640) ? std::min<uint32_t>(CeilDiv<uint32_t>(n, 16), aivCoreNum) : std::min<uint32_t>(n, aivCoreNum), 1);
    CsyrkCombineTilingData combineTiling = CalCombineTilingData(
        usedAivCoreNum, n, k, arLdc, ldc, static_cast<uint8_t>(isTransN ? 1 : 0),
        static_cast<uint8_t>(isAlphaZero ? 1 : 0), static_cast<uint8_t>(isKZero ? 1 : 0),
        static_cast<uint8_t>(isBetaZero ? 1 : 0), static_cast<uint8_t>(uplo), alphaReal, alphaImag, betaReal, betaImag);
    // Small-n fast config (alpha==(1,0), beta==0): one thread per uplo element,
    // no TPipe/TBuf barriers. Capped at n<=256 and only when the diagonal needs
    // no per-element k-loop recompute (that would serialize on wide shapes).
    const uint32_t simtMaxN = 256;
    if (combineTiling.isFastCfg && n > 0 && n <= simtMaxN && combineTiling.useDirectDiag == 0) {
        combineTiling.useSimt = 1;
        uint32_t simtCores = std::max<uint32_t>(std::min<uint32_t>(CeilDiv<uint32_t>(n, 4), aivCoreNum), 1);
        usedAivCoreNum = simtCores;
        combineTiling.rowsPerCore = CeilDiv<uint32_t>(n, simtCores);
        uint32_t workPerCore = CeilDiv<uint32_t>(n, usedAivCoreNum) * n;
        // Round the per-core thread count up to a 128-lane floor so each thread
        // handles ~1 element (too few threads serializes the band).
        combineTiling.simtThreads = CeilAlign<uint32_t>(std::min<uint32_t>(workPerCore, SIMT_MAX_THREAD_NUM), 128);
        csyrk_combine_simt_kernel_do(
            dAr, dAi, dTemp, reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(C)), combineTiling, usedAivCoreNum,
            stream);
        return;
    }
    csyrk_combine_kernel_do(
        dAr, dAi, dTemp, reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(C)), combineTiling, usedAivCoreNum,
        stream);
}

// Reads alpha/beta, applies the C-unchanged short-circuit and the special-value
// fallback. Returns true when the caller must return `status` immediately.
static bool CsyrkPrelude(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k, uint32_t lda,
    uint32_t ldc, const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* beta, aclblasComplex* C,
    float& alphaReal, float& alphaImag, float& betaReal, float& betaImag, bool& isAlphaZero, bool& isKZero,
    bool& isBetaZero, bool& skipTemp, bool& scalarsOnDevice, aclblasStatus_t& status)
{
    aclblasStatus_t st = PrepareCsyrkParams(
        h, trans, k, alpha, beta, alphaReal, alphaImag, betaReal, betaImag, isAlphaZero, isKZero, isBetaZero, skipTemp,
        scalarsOnDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        status = st;
        return true;
    }
    // Fast path: alpha==0 or k==0 and beta==(1,0) -> C unchanged.
    if (skipTemp && betaReal == 1.0f && betaImag == 0.0f) {
        status = ACLBLAS_STATUS_SUCCESS;
        return true;
    }
    // Special-value safety net (FLT_MAX-scale A overflows the 4-GEMM and diverges
    // from the Netlib golden): size-gated so the hot path never syncs.
    if (!skipTemp && static_cast<size_t>(n) * k <= CSYRK_FALLBACK_MAX_ELEMS) {
        aclblasStatus_t fbStatus = ACLBLAS_STATUS_SUCCESS;
        bool fbDone = CsyrkTrySpecialValueFallback(
            uplo, trans, n, k, lda, ldc, alphaReal, alphaImag, betaReal, betaImag, A, C, fbStatus, h->stream);
        if (fbDone) {
            status = fbStatus;
            return true;
        }
    }
    return false;
}

// Overflow-checked byte-size math. These operands are driven by caller-supplied
// n/k (uint32_t, unbounded above) and lda/ldc, so a plain size_t multiply can
// wrap to a small value and slip past the 2 GiB cap: EnsureDefaultWorkspace only
// ever sees the already-wrapped result, not the true requirement.
static bool CsyrkMulU64(uint64_t a, uint64_t b, uint64_t& out)
{
    if (a != 0 && b > UINT64_MAX / a) {
        return false;
    }
    out = a * b;
    return true;
}

static bool CsyrkAddU64(uint64_t a, uint64_t b, uint64_t& out)
{
    if (a > UINT64_MAX - b) {
        return false;
    }
    out = a + b;
    return true;
}

// Ensures the workspace holds Ar, Ai, temp (n x 4n) and, when useHf32Cube, the
// two HF32x3 residual buffers; outputs the per-matrix Ar/Ai byte size.
static aclblasStatus_t CsyrkEnsureWorkspace(
    _aclblas_handle* h, uint32_t n, uint32_t arLdc, uint32_t tempLdc, uint32_t arCols, bool useHf32Cube,
    size_t& arMatBytes)
{
    constexpr size_t GM_ALIGN = 32;
    uint64_t arBytes = 0;
    uint64_t tempBytes = 0;
    uint64_t total = 0;
    // arBytes = arLdc * arCols * 4 ; tempBytes = n * tempLdc * 4 * 4
    // total    = (useHf32Cube ? 4 : 2) * arBytes + tempBytes, then 32B-aligned.
    const bool ok = CsyrkMulU64(arLdc, static_cast<uint64_t>(arCols) * sizeof(float), arBytes) &&
                    CsyrkMulU64(n, static_cast<uint64_t>(tempLdc) * 4 * sizeof(float), tempBytes) &&
                    CsyrkMulU64(arBytes, useHf32Cube ? 4 : 2, total) && CsyrkAddU64(total, tempBytes, total) &&
                    CsyrkAddU64(total, GM_ALIGN - 1, total);
    if (!ok) {
        OP_LOGE(OP_TAG, "workspace size overflow: n=%u arLdc=%u tempLdc=%u arCols=%u", n, arLdc, tempLdc, arCols);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    total = total / GM_ALIGN * GM_ALIGN;
    if (total > ACLBLAS_MAX_WORKSPACE_SIZE) {
        OP_LOGE(
            OP_TAG, "workspace required %zu bytes exceeds maximum limit %zu bytes", static_cast<size_t>(total),
            ACLBLAS_MAX_WORKSPACE_SIZE);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    arMatBytes = static_cast<size_t>(arBytes);
    aclblasStatus_t ret = EnsureDefaultWorkspace(h, static_cast<size_t>(total));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "workspace ensure failed, required=%zu, ret=%d", static_cast<size_t>(total), ret);
    }
    return ret;
}

// ==========================================================================
//  LaunchCsyrkKernel - full pipeline
// ==========================================================================

// Runs the deint -> cube -> combine stages over the ensured workspace.
struct CsyrkGeom {
    uint32_t n, k, lda, ldc, arRows, arCols, arLdc, tempLdc;
    bool isTransN, useHf32Cube;
    bool isAlphaZero, isKZero, isBetaZero, skipTemp;
    // alpha/beta resolved from device-resident pointers: only then may the raw
    // pointers be handed to the direct SIMT kernel as GM addresses (!722).
    bool scalarsOnDevice;
    float alphaReal, alphaImag, betaReal, betaImag;
};

static aclblasStatus_t CsyrkRunStages(
    _aclblas_handle* h, aclblasFillMode_t uplo, const CsyrkGeom& g, const aclblasComplex* A, aclblasComplex* C,
    uint32_t aivCoreNum)
{
    size_t arMatBytes = 0;
    aclblasStatus_t wsRet = CsyrkEnsureWorkspace(h, g.n, g.arLdc, g.tempLdc, g.arCols, g.useHf32Cube, arMatBytes);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }
    uint8_t* ws = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* dAr = ws;
    uint8_t* dAi = dAr + arMatBytes;
    uint8_t* dArLow = dAi + arMatBytes; // valid only when useHf32Cube
    uint8_t* dAiLow = dArLow + (g.useHf32Cube ? arMatBytes : 0);
    uint8_t* dTemp = dAiLow + (g.useHf32Cube ? arMatBytes : 0);
    aclrtStream stream = h->stream;
    if (!g.skipTemp) {
        CsyrkLaunchDeinterleave(
            A, g.lda, g.arRows, g.arCols, g.arLdc, aivCoreNum, g.useHf32Cube, dAr, dAi, dArLow, dAiLow, stream);
        aclblasStatus_t cubeSt = CsyrkLaunchCube(
            uplo, g.n, g.k, g.arLdc, g.isTransN, g.useHf32Cube, dAr, dAi, dTemp, dArLow, dAiLow, stream);
        if (cubeSt != ACLBLAS_STATUS_SUCCESS) {
            return cubeSt;
        }
    }
    CsyrkLaunchCombine(
        uplo, g.n, g.k, g.arLdc, g.ldc, aivCoreNum, g.isTransN, g.isAlphaZero, g.isKZero, g.isBetaZero, g.alphaReal,
        g.alphaImag, g.betaReal, g.betaImag, dAr, dAi, dTemp, C, stream);
    return ACLBLAS_STATUS_SUCCESS;
}

// beta-only combine (alpha==0 or k==0, beta!=(1,0)). The kernel's betaOnly branch
// reads only cGM_, so the Ar/Ai/temp pointers are never dereferenced and no
// workspace is needed.
static aclblasStatus_t CsyrkLaunchCombineOnly(
    _aclblas_handle* h, aclblasFillMode_t uplo, const CsyrkGeom& g, aclblasComplex* C, uint32_t aivCoreNum)
{
    CsyrkLaunchCombine(
        uplo, g.n, g.k, g.arLdc, g.ldc, aivCoreNum, g.isTransN, g.isAlphaZero, g.isKZero, g.isBetaZero, g.alphaReal,
        g.alphaImag, g.betaReal, g.betaImag, nullptr, nullptr, nullptr, C, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCsyrkKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* alpha, const aclblasComplex* A, uint32_t lda, const aclblasComplex* beta, aclblasComplex* C,
    uint32_t ldc)
{
    CsyrkGeom g{};
    g.n = n;
    g.k = k;
    g.lda = lda;
    g.ldc = ldc;
    aclblasStatus_t preStatus = ACLBLAS_STATUS_SUCCESS;
    if (CsyrkPrelude(
            h, uplo, trans, n, k, lda, ldc, alpha, A, beta, C, g.alphaReal, g.alphaImag, g.betaReal, g.betaImag,
            g.isAlphaZero, g.isKZero, g.isBetaZero, g.skipTemp, g.scalarsOnDevice, preStatus)) {
        return preStatus;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    uint32_t aicCoreNum = GetAicCoreCount();
    if (aivCoreNum == 0 || aicCoreNum == 0) {
        OP_LOGE(OP_TAG, "core count is 0 (aiv=%u aic=%u)", aivCoreNum, aicCoreNum);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    g.isTransN = (trans == ACLBLAS_OP_N);
    // The direct kernel reads alpha/beta from GM addresses (!722 review): host
    // scalars must not be handed down as pointers, so only the device-scalar
    // case may take it; host scalars go through the tiling-carrying paths below.
    if (g.scalarsOnDevice && CsyrkTryLaunchDirect(h, uplo, n, k, lda, ldc, g.isTransN, aivCoreNum, alpha, A, beta, C)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // HF32x3 (compensated HF32 cube) is the large-n cube path; small n/k keep
    // native fp32 (HF32's 10-bit mantissa would need compensation even there at
    // no throughput benefit, matching the precise = (n<=256 && k<=512) gate).
    g.useHf32Cube = !(n <= 256 && k <= 512);
    g.arRows = g.isTransN ? n : k;
    g.arCols = g.isTransN ? k : n;
    g.arLdc = CeilAlign<uint32_t>(g.arRows, CSYRK_ARCH35_ELEMENTS_PER_BLOCK);
    g.tempLdc = CeilAlign<uint32_t>(n, CSYRK_ARCH35_ELEMENTS_PER_BLOCK);
    // Pure beta*C scale: no deinterleave, no cube, no Ar/Ai/temp workspace. Binding
    // the n x 4n temp slab here would fail large valid alpha==0 calls.
    if (g.skipTemp) {
        return CsyrkLaunchCombineOnly(h, uplo, g, C, aivCoreNum);
    }
    return CsyrkRunStages(h, uplo, g, A, C, aivCoreNum);
}

// ==========================================================================
//  Public API entry
// ==========================================================================
extern "C" aclblasStatus_t aclblasCsyrk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);

    aclblasStatus_t st = ValidateCsyrkParams(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);

    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchCsyrkKernel(
        h, uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha, A, static_cast<uint32_t>(lda), beta,
        C, static_cast<uint32_t>(ldc));
}
