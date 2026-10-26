/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file ssyr2k_host.cpp
 * \brief SSYR2K Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "ssyr2k_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/syrk_host_utils.h"

static inline GM_ADDR ToGmAddr(const float* p)
{
    return reinterpret_cast<GM_ADDR>(const_cast<float*>(p));
}

static aclblasStatus_t ValidateSsyr2kParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k,
    int lda, int ldb, int ldc,
    const float* alpha, const float* A, const float* B,
    const float* beta, float* C)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSsyr2k", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasSsyr2k", "trans must be OP_N(111) or OP_T(112) or OP_C(113), got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE);

    int minLda = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    int minLdb = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    CHECK_RET(
        lda >= minLda,
        OP_LOGE("aclblasSsyr2k", "lda must be >= %d, got lda=%d", minLda, lda);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldb >= minLdb,
        OP_LOGE("aclblasSsyr2k", "ldb must be >= %d, got ldb=%d", minLdb, ldb);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldc >= std::max(1, n),
        OP_LOGE("aclblasSsyr2k", "ldc must be >= %d, got ldc=%d", std::max(1, n), ldc);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSsyr2k", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasSsyr2k", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    // A/B non-null criterion unified with ssyrk/ssyrkx: only required when
    // both n > 0 and k > 0 (n == 0 or k == 0 means A/B are never referenced).
    CHECK_RET(
        !(A == nullptr && n > 0 && k > 0),
        OP_LOGE("aclblasSsyr2k", "A must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        !(B == nullptr && n > 0 && k > 0),
        OP_LOGE("aclblasSsyr2k", "B must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(C != nullptr || n == 0, OP_LOGE("aclblasSsyr2k", "C must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static Ssyr2kGemmTilingData CalGemmTilingData(
    uint32_t usedAicCoreNum, uint32_t n, uint32_t k,
    uint32_t leftLd, uint32_t rightLd, uint8_t isTransN,
    uint32_t tempRowStride)
{
    Ssyr2kGemmTilingData tiling{};
    tiling.n = n;
    tiling.k = k;
    tiling.leftLd = leftLd;
    tiling.rightLd = rightLd;
    tiling.isTransN = isTransN;

    uint32_t coreDim = CeilDiv<uint32_t>(n, usedAicCoreNum);
    tiling.singleCoreM = std::max<uint32_t>(coreDim, SSYR2K_ARCH35_BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreDim, SSYR2K_ARCH35_BASE_N);

    uint32_t tileM = std::min<uint32_t>(SSYR2K_ARCH35_DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(SSYR2K_ARCH35_DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = SSYR2K_ARCH35_DEFAULT_TILE_K_CHUNK;

    if (n < SSYR2K_ARCH35_DEFAULT_TILE_M) {
        tileM = std::min<uint32_t>(tiling.singleCoreM,
            CeilAlign<uint32_t>(n, SSYR2K_ARCH35_BASE_M));
    }
    if (n < SSYR2K_ARCH35_DEFAULT_TILE_N) {
        tileN = std::min<uint32_t>(tiling.singleCoreN,
            CeilAlign<uint32_t>(n, SSYR2K_ARCH35_BASE_N));
    }

    uint32_t alignedM = CeilAlign<uint32_t>(tileM, SSYR2K_ARCH35_BASE_M);
    uint32_t alignedN = CeilAlign<uint32_t>(tileN, SSYR2K_ARCH35_BASE_N);
    uint32_t l1Budget = SSYR2K_ARCH35_L1_SIZE_BYTES * SSYR2K_ARCH35_L1_USAGE_RATIO_NUM
                       / SSYR2K_ARCH35_L1_USAGE_RATIO_DEN;
    uint32_t denom = SSYR2K_ARCH35_L1_BUF_NUM * SSYR2K_ARCH35_FP32_SIZE * (alignedM + alignedN);
    uint32_t maxAlignedK = (denom > 0) ? (l1Budget / denom) : 0;
    uint32_t maxKChunk = (maxAlignedK / SSYR2K_ARCH35_BASE_K) * SSYR2K_ARCH35_BASE_K;
    maxKChunk = std::max<uint32_t>(maxKChunk, SSYR2K_ARCH35_BASE_K);
    if (tileKChunk > maxKChunk) {
        tileKChunk = maxKChunk;
    }

    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
    tiling.tempRowStride = tempRowStride;

    return tiling;
}

static Ssyr2kScaleTilingData CalScaleTilingData(
    uint32_t usedAivCoreNum, uint32_t n, uint32_t ldc,
    uint8_t isAlphaZero, uint8_t isKZero, uint8_t isBetaZero, uint32_t tempRowStride,
    uint8_t uploMode, float alphaVal, float betaVal)
{
    return ::CalScaleTilingData<Ssyr2kScaleTilingData>(
        usedAivCoreNum, n, ldc, isAlphaZero, isKZero, isBetaZero,
        tempRowStride, uploMode, alphaVal, betaVal);
}

static aclblasStatus_t LaunchGemmKernels(
    _aclblas_handle* h, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, aclblasOperation_t trans,
    const float* A, const float* B,
    uint32_t usedAicCoreNum, uint32_t tempRowStride,
    uint8_t* temp1Device, uint8_t* temp2Device)
{
    uint8_t isTransN = (trans == ACLBLAS_OP_N) ? 1 : 0;

    // GEMM1: op(A) * op(B)^T -> temp1  (left=A, right=B)
    Ssyr2kGemmTilingData gemm1Tiling = CalGemmTilingData(
        usedAicCoreNum, n, k, lda, ldb, isTransN, tempRowStride);
    OP_LOGD("aclblasSsyr2k",
        "gemm1 tiling: n=%u k=%u leftLd=%u rightLd=%u isTransN=%u aicCores=%u "
        "singleCoreM=%u singleCoreN=%u tileM=%u tileN=%u tileKChunk=%u tempRowStride=%u",
        gemm1Tiling.n, gemm1Tiling.k, gemm1Tiling.leftLd, gemm1Tiling.rightLd,
        gemm1Tiling.isTransN, usedAicCoreNum,
        gemm1Tiling.singleCoreM, gemm1Tiling.singleCoreN,
        gemm1Tiling.tileM, gemm1Tiling.tileN, gemm1Tiling.tileKChunk,
        gemm1Tiling.tempRowStride);
    OP_LOGI("aclblasSsyr2k", "launching gemm1 kernel: aicCores=%u", usedAicCoreNum);
    ssyr2k_gemm_kernel_do(ToGmAddr(A), ToGmAddr(B), temp1Device,
        gemm1Tiling, usedAicCoreNum, h->stream);

    // GEMM2: op(B) * op(A)^T -> temp2  (left=B, right=A)
    Ssyr2kGemmTilingData gemm2Tiling = CalGemmTilingData(
        usedAicCoreNum, n, k, ldb, lda, isTransN, tempRowStride);
    OP_LOGI("aclblasSsyr2k", "launching gemm2 kernel: aicCores=%u", usedAicCoreNum);
    ssyr2k_gemm_kernel_do(ToGmAddr(B), ToGmAddr(A), temp2Device,
        gemm2Tiling, usedAicCoreNum, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

static void LaunchScaleKernel(
    _aclblas_handle* h, uint32_t usedAivCoreNum, uint32_t n, uint32_t ldc,
    bool isAlphaZero, bool isKZero, bool isBetaZero,
    aclblasFillMode_t uplo, float alphaVal, float betaVal,
    uint32_t tempRowStride, uint8_t* temp1Device, uint8_t* temp2Device, float* C)
{
    Ssyr2kScaleTilingData scaleTiling = CalScaleTilingData(
        usedAivCoreNum, n, ldc,
        static_cast<uint8_t>(isAlphaZero ? 1 : 0),
        static_cast<uint8_t>(isKZero ? 1 : 0),
        static_cast<uint8_t>(isBetaZero ? 1 : 0),
        tempRowStride,
        static_cast<uint8_t>(uplo), alphaVal, betaVal);

    OP_LOGD("aclblasSsyr2k",
        "scale tiling: n=%u ldc=%u aivCores=%u isAlphaZero=%u isKZero=%u isBetaZero=%u uplo=%u",
        scaleTiling.n, scaleTiling.ldc, usedAivCoreNum,
        static_cast<uint32_t>(scaleTiling.isAlphaZero),
        static_cast<uint32_t>(scaleTiling.isKZero),
        static_cast<uint32_t>(scaleTiling.isBetaZero),
        static_cast<uint32_t>(scaleTiling.uploMode));
    OP_LOGI("aclblasSsyr2k", "launching scale kernel: aivCores=%u", usedAivCoreNum);

    ssyr2k_scale_kernel_do(
        temp1Device, temp2Device, reinterpret_cast<uint8_t*>(C),
        scaleTiling, usedAivCoreNum, h->stream);
}

static aclblasStatus_t LaunchSsyr2kKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans,
    uint32_t n, uint32_t k,
    const float* alpha, const float* A, uint32_t lda,
    const float* B, uint32_t ldb,
    const float* beta, float* C, uint32_t ldc)
{
    SyrkLaunchCtx ctx;
    aclblasStatus_t prepRet = PrepareSyrkLaunch(
        h->stream, n, k, trans, alpha, beta, SSYR2K_ARCH35_BASE_M, SSYR2K_ARCH35_FIXPIPE_N_ALIGN,
        "aclblasSsyr2k", ctx);
    if (prepRet != ACLBLAS_STATUS_SUCCESS) {
        return prepRet;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    size_t requiredBytes = 2 * ctx.tempAligned;
    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, requiredBytes);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSsyr2k", "workspace ensure failed, required=%zu, ret=%d", requiredBytes, wsRet);
        return wsRet;
    }

    uint8_t* wsBase = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* temp1Device = wsBase;
    uint8_t* temp2Device = temp1Device + ctx.tempAligned;

    if (!ctx.isAlphaZero && !ctx.isKZero) {
        aclblasStatus_t gemmRet = LaunchGemmKernels(h, n, k, lda, ldb, trans, A, B,
            ctx.usedAicCoreNum, ctx.tempRowStride, temp1Device, temp2Device);
        if (gemmRet != ACLBLAS_STATUS_SUCCESS) {
            return gemmRet;
        }
    }

    LaunchScaleKernel(h, ctx.usedAivCoreNum, n, ldc,
        ctx.isAlphaZero, ctx.isKZero, ctx.isBetaZero, uplo, ctx.alphaVal, ctx.betaVal,
        ctx.tempRowStride, temp1Device, temp2Device, C);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSsyr2k(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSsyr2k", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    CHECK_RET(n >= 0,
        OP_LOGE("aclblasSsyr2k", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0,
        OP_LOGE("aclblasSsyr2k", "k must be >= 0, got %d", k);
        return ACLBLAS_STATUS_INVALID_VALUE);

    aclblasStatus_t st = ValidateSsyr2kParams(
        uplo, trans, n, k, lda, ldb, ldc, alpha, A, B, beta, C);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);

    return LaunchSsyr2kKernel(h, uplo, trans,
        static_cast<uint32_t>(n), static_cast<uint32_t>(k),
        alpha, A, static_cast<uint32_t>(lda),
        B, static_cast<uint32_t>(ldb),
        beta, C, static_cast<uint32_t>(ldc));
}
