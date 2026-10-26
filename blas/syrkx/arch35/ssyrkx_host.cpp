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
 * \file ssyrkx_host.cpp
 * \brief SSYRKX Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "ssyrkx_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/syrk_host_utils.h"

static inline GM_ADDR ToGmAddr(const float* p)
{
    return reinterpret_cast<GM_ADDR>(const_cast<float*>(p));
}

static aclblasStatus_t ValidateUploAndTrans(
    aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSsyrkx", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasSsyrkx", "trans must be OP_N, OP_T or OP_C, got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateLeadingDims(
    aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    if (trans == ACLBLAS_OP_N) {
        CHECK_RET(lda >= std::max(1, n),
            OP_LOGE("aclblasSsyrkx", "lda must be >= max(1,n) for trans=N, got lda=%d n=%d", lda, n);
            return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(ldb >= std::max(1, n),
            OP_LOGE("aclblasSsyrkx", "ldb must be >= max(1,n) for trans=N, got ldb=%d n=%d", ldb, n);
            return ACLBLAS_STATUS_INVALID_VALUE);
    } else {
        CHECK_RET(lda >= std::max(1, k),
            OP_LOGE("aclblasSsyrkx", "lda must be >= max(1,k) for trans=T, got lda=%d k=%d", lda, k);
            return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(ldb >= std::max(1, k),
            OP_LOGE("aclblasSsyrkx", "ldb must be >= max(1,k) for trans=T, got ldb=%d k=%d", ldb, k);
            return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(ldc >= std::max(1, n),
        OP_LOGE("aclblasSsyrkx", "ldc must be >= max(1,n), got ldc=%d n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidatePointers(
    const float* alpha, const float* A, const float* B,
    const float* beta, float* C, int n, int k)
{
    CHECK_RET(alpha != nullptr,
        OP_LOGE("aclblasSsyrkx", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr,
        OP_LOGE("aclblasSsyrkx", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(A == nullptr && n > 0 && k > 0),
        OP_LOGE("aclblasSsyrkx", "A must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(B == nullptr && n > 0 && k > 0),
        OP_LOGE("aclblasSsyrkx", "B must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(C == nullptr && n > 0),
        OP_LOGE("aclblasSsyrkx", "C must not be nullptr when n>0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateSsyrkxParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc)
{
    aclblasStatus_t st = ValidateUploAndTrans(uplo, trans);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = ValidateLeadingDims(trans, n, k, lda, ldb, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    return ValidatePointers(alpha, A, B, beta, C, n, k);
}

static SsyrkxGemmTilingData CalGemmTilingData(
    uint32_t usedAicCoreNum, uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb,
    uint8_t isTransN, uint32_t tempRowStride)
{
    SsyrkxGemmTilingData tiling{};
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.isTransN = isTransN;

    uint32_t coreDim = CeilDiv<uint32_t>(n, usedAicCoreNum);
    tiling.singleCoreM = std::max<uint32_t>(coreDim, SYRKX_ARCH35_BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreDim, SYRKX_ARCH35_BASE_N);

    uint32_t tileM = std::min<uint32_t>(SYRKX_ARCH35_DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(SYRKX_ARCH35_DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = SYRKX_ARCH35_DEFAULT_TILE_K_CHUNK;

    if (n < SYRKX_ARCH35_DEFAULT_TILE_M) {
        tileM = std::min<uint32_t>(tiling.singleCoreM,
            CeilAlign<uint32_t>(n, SYRKX_ARCH35_BASE_M));
    }
    if (n < SYRKX_ARCH35_DEFAULT_TILE_N) {
        tileN = std::min<uint32_t>(tiling.singleCoreN,
            CeilAlign<uint32_t>(n, SYRKX_ARCH35_BASE_N));
    }

    uint32_t alignedM = CeilAlign<uint32_t>(tileM, SYRKX_ARCH35_BASE_M);
    uint32_t alignedN = CeilAlign<uint32_t>(tileN, SYRKX_ARCH35_BASE_N);
    uint32_t l1Budget = SYRKX_ARCH35_L1_SIZE_BYTES * SYRKX_ARCH35_L1_USAGE_RATIO_NUM
                       / SYRKX_ARCH35_L1_USAGE_RATIO_DEN;
    uint32_t denom = SYRKX_ARCH35_L1_BUF_NUM * SYRKX_ARCH35_FP32_SIZE * (alignedM + alignedN);
    uint32_t maxAlignedK = (denom > 0) ? (l1Budget / denom) : 0;
    uint32_t maxKChunk = (maxAlignedK / SYRKX_ARCH35_BASE_K) * SYRKX_ARCH35_BASE_K;
    maxKChunk = std::max<uint32_t>(maxKChunk, SYRKX_ARCH35_BASE_K);
    if (tileKChunk > maxKChunk) {
        tileKChunk = maxKChunk;
    }

    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
    tiling.tempRowStride = tempRowStride;

    return tiling;
}

static aclblasStatus_t LaunchGemmKernel(
    _aclblas_handle* h, uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb,
    aclblasOperation_t trans, const float* A, const float* B,
    uint32_t usedAicCoreNum, uint8_t* tempDevice, uint32_t tempRowStride)
{
    uint8_t isTransN = (trans == ACLBLAS_OP_N) ? 1 : 0;
    SsyrkxGemmTilingData gemmTiling = CalGemmTilingData(
        usedAicCoreNum, n, k, lda, ldb, isTransN, tempRowStride);
    uint32_t divM = CeilDiv<uint32_t>(n, gemmTiling.singleCoreM);
    uint32_t divN = CeilDiv<uint32_t>(n, gemmTiling.singleCoreN);
    uint64_t totalTiles = static_cast<uint64_t>(divM) * static_cast<uint64_t>(divN);
    uint32_t actualAicCores = std::max<uint32_t>(
        std::min<uint64_t>(totalTiles, usedAicCoreNum), 1);
    OP_LOGD("aclblasSsyrkx",
        "gemm tiling: n=%u k=%u lda=%u ldb=%u isTransN=%u aicCores=%u singleCoreM=%u singleCoreN=%u "
        "tileM=%u tileN=%u tileKChunk=%u tempRowStride=%u",
        gemmTiling.n, gemmTiling.k, gemmTiling.lda, gemmTiling.ldb, gemmTiling.isTransN, actualAicCores,
        gemmTiling.singleCoreM, gemmTiling.singleCoreN,
        gemmTiling.tileM, gemmTiling.tileN, gemmTiling.tileKChunk,
        gemmTiling.tempRowStride);
    OP_LOGI("aclblasSsyrkx", "launching gemm kernel: n=%u k=%u aicCores=%u", n, k, actualAicCores);

    // GEMM computes temp_T = op(B) * op(A)^T: left=gmB, right=gmA
    ssyrkx_gemm_kernel_do(
        ToGmAddr(B), ToGmAddr(A), tempDevice,
        gemmTiling, actualAicCores, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchSsyrkxKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans,
    uint32_t n, uint32_t k,
    const float* alpha, const float* A, uint32_t lda,
    const float* B, uint32_t ldb,
    const float* beta, float* C, uint32_t ldc)
{
    SyrkLaunchCtx ctx;
    aclblasStatus_t prepRet = PrepareSyrkLaunch(
        h->stream, n, k, trans, alpha, beta, SYRKX_ARCH35_BASE_M, SYRKX_ARCH35_FIXPIPE_N_ALIGN,
        "aclblasSsyrkx", ctx);
    if (prepRet != ACLBLAS_STATUS_SUCCESS) {
        return prepRet;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, ctx.tempAligned);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSsyrkx", "workspace ensure failed, required=%zu, ret=%d", ctx.tempAligned, wsRet);
        return wsRet;
    }

    uint8_t* tempDevice = static_cast<uint8_t*>(GetEffectiveWorkspace(h));

    if (!ctx.isAlphaZero && !ctx.isKZero) {
        aclblasStatus_t gemmRet = LaunchGemmKernel(h, n, k, lda, ldb, trans, A, B,
            ctx.usedAicCoreNum, tempDevice, ctx.tempRowStride);
        if (gemmRet != ACLBLAS_STATUS_SUCCESS) {
            return gemmRet;
        }
    }

    SsyrkxScaleTilingData scaleTiling = CalScaleTilingData<SsyrkxScaleTilingData>(
        ctx.usedAivCoreNum, n, ldc,
        static_cast<uint8_t>(ctx.isAlphaZero ? 1 : 0),
        static_cast<uint8_t>(ctx.isKZero ? 1 : 0),
        static_cast<uint8_t>(ctx.isBetaZero ? 1 : 0),
        ctx.tempRowStride,
        static_cast<uint8_t>(uplo), ctx.alphaVal, ctx.betaVal);
    OP_LOGD("aclblasSsyrkx",
        "scale tiling: n=%u ldc=%u tempRowStride=%u rowsPerCore=%u uplo=%u "
        "isAlphaZero=%u isKZero=%u isBetaZero=%u",
        scaleTiling.n, scaleTiling.ldc, scaleTiling.tempRowStride, scaleTiling.rowsPerCore,
        scaleTiling.uploMode, scaleTiling.isAlphaZero, scaleTiling.isKZero, scaleTiling.isBetaZero);
    OP_LOGI("aclblasSsyrkx", "launching scale kernel: n=%u aivCores=%u", n, ctx.usedAivCoreNum);
    ssyrkx_scale_kernel_do(
        tempDevice, ToGmAddr(C),
        scaleTiling, ctx.usedAivCoreNum, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSsyrkx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSsyrkx", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    CHECK_RET(n >= 0,
        OP_LOGE("aclblasSsyrkx", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0,
        OP_LOGE("aclblasSsyrkx", "k must be >= 0, got %d", k);
        return ACLBLAS_STATUS_INVALID_VALUE);

    aclblasStatus_t st = ValidateSsyrkxParams(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);

    return LaunchSsyrkxKernel(h, uplo, trans,
        static_cast<uint32_t>(n), static_cast<uint32_t>(k),
        alpha, A, static_cast<uint32_t>(lda),
        B, static_cast<uint32_t>(ldb),
        beta, C, static_cast<uint32_t>(ldc));
}
