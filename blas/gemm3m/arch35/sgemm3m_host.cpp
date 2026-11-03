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
 * \file sgemm3m_host.cpp
 * \brief GEMM3M host-side implementation (tensor_api cube + SIMD vector).
 *
 * C = alpha * (A1*B1 + A2*B2 + A3*B3) + beta * C (column-major, FP32)
 * A contains A1/A2/A3 merged along K; B contains B1/B2/B3 merged along K.
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "sgemm3m_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// ============================================================================
// Parameter validation
// ============================================================================

static aclblasStatus_t ValidateDimensions(aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k)
{
    if (transA != ACLBLAS_OP_N && transA != ACLBLAS_OP_T && transA != ACLBLAS_OP_C) {
        OP_LOGE("aclblasSgemm3m", "invalid transA=%d", static_cast<int>(transA));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (transB != ACLBLAS_OP_N && transB != ACLBLAS_OP_T && transB != ACLBLAS_OP_C) {
        OP_LOGE("aclblasSgemm3m", "invalid transB=%d", static_cast<int>(transB));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0) {
        OP_LOGE("aclblasSgemm3m", "invalid dimensions: m=%d, n=%d, k=%d", m, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateLeadingDims(bool isTransA, bool isTransB, int m, int n, int k, int lda, int ldb, int ldc)
{
    // transA=N: A is M×(3K), lda >= M
    // transA=T: A is (3K)×M, lda >= 3K
    int64_t expectedLda = isTransA ? std::max<int64_t>(1, static_cast<int64_t>(GEMM3M_NUM_PAIRS) * k) : std::max(1, m);
    if (static_cast<int64_t>(lda) < expectedLda) {
        OP_LOGE("aclblasSgemm3m", "invalid lda=%d, expected=%lld", lda, static_cast<long long>(expectedLda));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // transB=N: B is (3K)×N, ldb >= 3K
    // transB=T: B is N×(3K), ldb >= N
    int64_t expectedLdb = isTransB ? std::max(1, n) : std::max<int64_t>(1, static_cast<int64_t>(GEMM3M_NUM_PAIRS) * k);
    if (static_cast<int64_t>(ldb) < expectedLdb) {
        OP_LOGE("aclblasSgemm3m", "invalid ldb=%d, expected=%lld", ldb, static_cast<long long>(expectedLdb));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("aclblasSgemm3m", "invalid ldc=%d, expected=%d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidatePointers(
    const float* alpha, const float* beta, const float* A, const float* B, float* C, int m, int n, int k,
    float alphaVal)
{
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclblasSgemm3m", "alpha and beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && k > 0 && alphaVal != 0.0f) {
        if (A == nullptr || B == nullptr) {
            OP_LOGE("aclblasSgemm3m", "A/B matrices must not be nullptr when m,n,k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (m > 0 && n > 0 && C == nullptr) {
        OP_LOGE("aclblasSgemm3m", "C must not be nullptr when m,n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGemm3mParams(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, int lda, int ldb,
    int ldc, const float* alpha, const float* beta, const float* A, const float* B, float* C)
{
    auto* h = static_cast<_aclblas_handle*>(handle);
    if (h == nullptr) {
        OP_LOGE("aclblasSgemm3m", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    aclblasStatus_t st = ValidateDimensions(transA, transB, m, n, k);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    bool isTransA = (transA != ACLBLAS_OP_N);
    bool isTransB = (transB != ACLBLAS_OP_N);
    st = ValidateLeadingDims(isTransA, isTransB, m, n, k, lda, ldb, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    float alphaVal = (alpha != nullptr) ? *alpha : 0.0f;
    return ValidatePointers(alpha, beta, A, B, C, m, n, k, alphaVal);
}

// ============================================================================
// Tiling computation
// ============================================================================

static void InitTilingParams(
    Gemm3MTilingData& t, int m, int n, int k, int lda, int ldb, int ldc, bool isTransA, bool isTransB, float alpha,
    float beta)
{
    t.m = m;
    t.n = n;
    t.k = k;
    t.lda = lda;
    t.ldb = ldb;
    t.ldc = ldc;
    t.ldcOrig = ldc;
    t.isTransA = isTransA ? 1 : 0;
    t.isTransB = isTransB ? 1 : 0;
    t.alpha = alpha;
    t.beta = beta;
    t.hasBeta = (beta != 0.0f) ? 1 : 0;
    t.needPostProcess = (alpha != 1.0f || beta != 0.0f) ? 1 : 0;
}

static void CalcMultiCorePartition(Gemm3MTilingData& t, uint32_t cubeCoreNum)
{
    int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    int32_t mTiles = static_cast<int32_t>(CeilDiv<int64_t>(t.m, GEMM3M_BASE_M));
    int32_t nTiles = static_cast<int32_t>(CeilDiv<int64_t>(t.n, GEMM3M_BASE_N));
    int32_t bestMb = 1, bestNb = 1, bestUtil = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= maxCores; ++mb) {
        int32_t nb = std::min(nTiles, maxCores / mb);
        if (nb < 1) {
            nb = 1;
        }
        int32_t util = mb * nb;
        if (util > bestUtil && util <= maxCores) {
            bestUtil = util;
            bestMb = mb;
            bestNb = nb;
        }
    }
    if (bestUtil == 0) {
        bestMb = 1;
        bestNb = 1;
    }
    t.mBlocks = bestMb;
    t.nBlocks = bestNb;
    t.usedCoreNum = bestMb * bestNb;
}

static void CalcPerCoreWorkload(Gemm3MTilingData& t)
{
    t.singleCoreM = static_cast<int32_t>(CeilDiv<int64_t>(t.m, t.mBlocks));
    t.singleCoreM = static_cast<int32_t>(CeilDiv<int64_t>(t.singleCoreM, GEMM3M_BASE_M) * GEMM3M_BASE_M);
    if (t.singleCoreM > t.m) {
        t.singleCoreM = t.m;
    }
    t.singleCoreN = static_cast<int32_t>(CeilDiv<int64_t>(t.n, t.nBlocks));
    t.singleCoreN = static_cast<int32_t>(CeilDiv<int64_t>(t.singleCoreN, GEMM3M_BASE_N) * GEMM3M_BASE_N);
    if (t.singleCoreN > t.n) {
        t.singleCoreN = t.n;
    }
}

static Gemm3MTilingData CalcGemm3mTiling(
    uint32_t cubeCoreNum, int m, int n, int k, int lda, int ldb, int ldc, bool isTransA, bool isTransB, float alpha,
    float beta)
{
    Gemm3MTilingData t{};
    InitTilingParams(t, m, n, k, lda, ldb, ldc, isTransA, isTransB, alpha, beta);
    CalcMultiCorePartition(t, cubeCoreNum);
    CalcPerCoreWorkload(t);
    OP_LOGD(
        "aclblasSgemm3m",
        "tiling: M=%d, N=%d, K=%d, baseM=%d, baseN=%d, baseK=%d, "
        "singleCoreM=%d, singleCoreN=%d, mBlocks=%d, nBlocks=%d, cores=%d",
        m, n, k, GEMM3M_BASE_M, GEMM3M_BASE_N, GEMM3M_BASE_K, t.singleCoreM, t.singleCoreN, t.mBlocks, t.nBlocks,
        t.usedCoreNum);
    return t;
}

// ============================================================================
// Boundary handlers
// ============================================================================

static aclblasStatus_t HandleKZero(float* C, int m, int ldc, int n, float betaVal)
{
    // issue #742: 按 m×n 逻辑足迹逐列处理——ldc > m 时 ldc*n 全量区间含
    // 列间填充/子块外相邻数据，memset/缩放会越界并破坏相邻数据
    if (betaVal == 0.0f) {
        const size_t colBytes = static_cast<size_t>(m) * sizeof(float);
        for (int j = 0; j < n; ++j) {
            aclError ret = aclrtMemset(C + static_cast<size_t>(j) * ldc, colBytes, 0, colBytes);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasSgemm3m", "aclrtMemset failed (k=0), ret=%d", ret);
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    } else if (betaVal != 1.0f) {
        std::vector<float> colHost(static_cast<size_t>(m));
        const size_t colBytes = static_cast<size_t>(m) * sizeof(float);
        for (int j = 0; j < n; ++j) {
            float* col = C + static_cast<size_t>(j) * ldc;
            aclError ret = aclrtMemcpy(colHost.data(), colBytes, col, colBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasSgemm3m", "aclrtMemcpy D2H failed (k=0), ret=%d", ret);
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
            for (size_t i = 0; i < colHost.size(); ++i) {
                colHost[i] *= betaVal;
            }
            ret = aclrtMemcpy(col, colBytes, colHost.data(), colBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasSgemm3m", "aclrtMemcpy H2D failed (k=0), ret=%d", ret);
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t HandleAlphaZero(float* C, int m, int ldc, int n, float betaVal)
{
    if (betaVal == 0.0f) {
        const size_t colBytes = static_cast<size_t>(m) * sizeof(float);
        aclError ret = ACL_SUCCESS;
        for (int j = 0; j < n; ++j) {
            ret = aclrtMemset(C + static_cast<size_t>(j) * ldc, colBytes, 0, colBytes);
            if (ret != ACL_SUCCESS) {
                break;
            }
        }
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasSgemm3m", "aclrtMemset failed (alpha=0), ret=%d", ret);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else if (betaVal != 1.0f) {
        return HandleKZero(C, m, ldc, n, betaVal);
    }
    // beta==1.0f: C unchanged (alpha*0 + 1*C = C), no operation needed
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Kernel launch
// ============================================================================

// Allocate temp buffer from workspace for the cube kernel's intermediate result.
static aclblasStatus_t AllocateTempBuffer(const _aclblas_handle* h, int m, int n, uint8_t*& tempDevice)
{
    size_t tempBytes = static_cast<size_t>(m) * n * sizeof(float);
    if (!CheckEffectiveWorkspaceSize(h, tempBytes)) {
        OP_LOGE("aclblasSgemm3m", "workspace not enough for temp buffer: need=%zu bytes", tempBytes);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    tempDevice = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    return ACLBLAS_STATUS_SUCCESS;
}

// Launch alpha/beta vector kernel for post-processing C = alpha*temp + beta*C.
// This is an AIV-only kernel; on arch35 AIC:AIV = 1:2, so use GetAivCoreCount()
// as the upper bound to utilize twice as many vector cores as the cube kernel.
// A per-core workload cap avoids launching excessive cores for small matrices.
static void LaunchAlphaBetaKernel(
    const _aclblas_handle* h, uint8_t* tempDevice, float* C, const Gemm3MTilingData& abTilingData, int m, int n,
    int ldc, float alphaVal, float betaVal, uint32_t aivCoreNum)
{
    constexpr int64_t ELEMENTS_PER_CORE = 16384;
    int64_t totalElements = static_cast<int64_t>(m) * n;
    uint32_t alphaBetaCores = static_cast<uint32_t>(std::min<int64_t>(
        static_cast<int64_t>(aivCoreNum), std::max<int64_t>(1, CeilDiv<int64_t>(totalElements, ELEMENTS_PER_CORE))));
    OP_LOGI(
        "aclblasSgemm3m", "launching alpha/beta kernel: alpha=%.4f, beta=%.4f, m=%d, n=%d, ldcOrig=%d, blocks=%u",
        alphaVal, betaVal, m, n, ldc, alphaBetaCores);
    uint8_t* cOrigPtr = reinterpret_cast<uint8_t*>(C);
    gemm3m_alpha_beta_do(alphaBetaCores, h->stream, tempDevice, cOrigPtr, cOrigPtr, abTilingData);
}

static aclblasStatus_t LaunchGemm3mKernel(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const float* A,
    int lda, const float* B, int ldb, float* C, int ldc, float alphaVal, float betaVal, uint32_t cubeCoreNum,
    uint32_t aivCoreNum)
{
    Gemm3MTilingData tilingData = CalcGemm3mTiling(
        cubeCoreNum, m, n, k, lda, ldb, ldc, transA != ACLBLAS_OP_N, transB != ACLBLAS_OP_N, alphaVal, betaVal);
    if (tilingData.usedCoreNum == 0) {
        OP_LOGE("aclblasSgemm3m", "Invalid tiling data, usedCoreNum=0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    bool needPostProcess = (alphaVal != 1.0f) || (betaVal != 0.0f);
    tilingData.ldcOrig = ldc;
    // Temp buffer stores C^T (N rows x M cols) in row-major (NDExtLayoutPtn):
    //   element (i,j) at buffer[i*ldc + j], so max index = (N-1)*M + (M-1) = M*N-1.
    // Setting ldc=m (original M) gives row stride = M, which exactly fits the
    // M*N temp buffer for all M,N combinations. Using ldc=n would overflow
    // when M < N because max index becomes (N-1)*N + (M-1) > M*N-1.
    if (needPostProcess) {
        tilingData.ldc = m;
    }
    Gemm3MTilingData abTilingData = tilingData;
    std::swap(tilingData.m, tilingData.n);
    std::swap(tilingData.lda, tilingData.ldb);
    std::swap(tilingData.isTransA, tilingData.isTransB);
    std::swap(tilingData.singleCoreM, tilingData.singleCoreN);
    std::swap(tilingData.mBlocks, tilingData.nBlocks);
    // After swap + rounding, mBlocks*singleCoreM may exceed m (round-up to baseM).
    // Recompute usedCoreNum to only count cores with actual work, avoiding
    // out-of-bounds GM access from idle cores with mOff >= m or nOff >= n.
    int32_t effMBlocks = static_cast<int32_t>(CeilDiv<int64_t>(tilingData.m, tilingData.singleCoreM));
    int32_t effNBlocks = static_cast<int32_t>(CeilDiv<int64_t>(tilingData.n, tilingData.singleCoreN));
    tilingData.mBlocks = effMBlocks;
    tilingData.nBlocks = effNBlocks;
    tilingData.usedCoreNum = effMBlocks * effNBlocks;
    auto* h = static_cast<_aclblas_handle*>(handle);
    uint8_t* tempDevice = nullptr;
    if (needPostProcess) {
        aclblasStatus_t st = AllocateTempBuffer(h, m, n, tempDevice);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    uint8_t* aPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
    uint8_t* bPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(B));
    uint8_t* cPtr = needPostProcess ? tempDevice : reinterpret_cast<uint8_t*>(C);
    OP_LOGI(
        "aclblasSgemm3m", "launching cube kernel: blocks=%u, needPostProcess=%d, lda=%d, ldb=%d, ldc=%d",
        tilingData.usedCoreNum, needPostProcess, tilingData.lda, tilingData.ldb, tilingData.ldc);
    gemm3m_kernel_do(tilingData.usedCoreNum, h->stream, aPtr, bPtr, cPtr, tilingData);
    if (needPostProcess) {
        LaunchAlphaBetaKernel(h, tempDevice, C, abTilingData, m, n, ldc, alphaVal, betaVal, aivCoreNum);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// API entry
// ============================================================================

aclblasStatus_t aclblasSgemm3m(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    OP_LOGI(
        "aclblasSgemm3m", "entry: transA=%d, transB=%d, m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d",
        static_cast<int>(transA), static_cast<int>(transB), m, n, k, lda, ldb, ldc);

    aclblasStatus_t st = ValidateGemm3mParams(handle, transA, transB, m, n, k, lda, ldb, ldc, alpha, beta, A, B, C);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSgemm3m", "parameter validation failed, status=%d", static_cast<int>(st));
        return st;
    }

    // For real (FP32) dtype, OP_C is equivalent to OP_T (conjugate transpose of real = transpose)
    if (transA == ACLBLAS_OP_C) {
        OP_LOGD("aclblasSgemm3m", "transA=C treated as T for FP32 real dtype");
    }
    if (transB == ACLBLAS_OP_C) {
        OP_LOGD("aclblasSgemm3m", "transB=C treated as T for FP32 real dtype");
    }

    if (m == 0 || n == 0) {
        OP_LOGI("aclblasSgemm3m", "m=0 or n=0, skip");
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (k == 0) {
        OP_LOGI("aclblasSgemm3m", "k=0, handle C=beta*C");
        return HandleKZero(C, m, ldc, n, *beta);
    }

    uint32_t cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasSgemm3m", "Failed to get cube core count");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemm3m", "Failed to get aiv core count");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    float alphaVal = *alpha;
    float betaVal = *beta;

    if (alphaVal == 0.0f) {
        OP_LOGI("aclblasSgemm3m", "alpha=0, beta=%.4f, skipping matmul", betaVal);
        return HandleAlphaZero(C, m, ldc, n, betaVal);
    }

    return LaunchGemm3mKernel(
        handle, transA, transB, m, n, k, A, lda, B, ldb, C, ldc, alphaVal, betaVal, cubeCoreNum, aivCoreNum);
}
