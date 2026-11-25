/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "hgemm_kernel.h"
#include "gemm_host_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/dtype_cast.h"

// ==========================================================================
//  Tiling constants - must mirror hgemm_kernel.cpp
// ==========================================================================
constexpr int32_t HGEMM_BASE_M = 128;
constexpr int32_t HGEMM_BASE_K = 64;
constexpr int32_t HGEMM_BASE_N = 128;
constexpr int32_t HGEMM_TILE_K_CHUNK = 128;
constexpr int32_t HGEMM_C0_SIZE = 16;
constexpr int32_t HGEMM_FRACTAL = 16;
constexpr int32_t HGEMM_GROUP_M = 2;
constexpr int32_t HGEMM_GROUP_N = 2;

// Multi-core partition with a group-level cost model. Each core processes
// its tile block as a sequence of (GROUP_M x GROUP_N) tile groups; every
// group sweeps the full K axis. A group of (mG x nG) tiles costs, per K
// element: MAC cycles = mG * nG * BM * BN / 4096 (cube rate), BW cycles =
// (mG * BM + nG * BN) * 2 / 64 (MTE1/MTE2 staging), plus a fixed
// pipeline-restart cost per group. Degenerate groups (a single tile in one
// dimension) are bandwidth-bound and cost up to 2x per MAC, so the model
// prefers partitions that keep full 2x2 groups on every core even at a
// slightly worse raw tile makespan (e.g. 2936^3: (2,14) with 6 full groups
// per core beats (1,23) with 12 degenerate 2x1 groups).
static int64_t HgemmCoreGroupCost(int32_t mTiles, int32_t nTiles, int32_t kTotal)
{
    constexpr int64_t restartCycles = 1000;
    constexpr int64_t macPerTile = static_cast<int64_t>(HGEMM_BASE_M) * HGEMM_BASE_N / 4096; // 4
    constexpr int64_t bwPerTileDim = static_cast<int64_t>(HGEMM_BASE_M) * 2 / 64;            // 4
    const int32_t mGroups = CeilDiv(mTiles, HGEMM_GROUP_M);
    const int32_t nGroups = CeilDiv(nTiles, HGEMM_GROUP_N);
    int64_t cost = 0;
    for (int32_t mg = 0; mg < mGroups; mg++) {
        int32_t mG = (mg + 1 < mGroups) ? HGEMM_GROUP_M : (mTiles - mg * HGEMM_GROUP_M);
        for (int32_t ng = 0; ng < nGroups; ng++) {
            int32_t nG = (ng + 1 < nGroups) ? HGEMM_GROUP_N : (nTiles - ng * HGEMM_GROUP_N);
            int64_t macCycles = static_cast<int64_t>(mG) * nG * macPerTile * kTotal;
            int64_t bwCycles = (static_cast<int64_t>(mG) + nG) * bwPerTileDim * kTotal;
            cost += (macCycles > bwCycles ? macCycles : bwCycles) + restartCycles;
        }
    }
    return cost;
}

static void CalcHgemmMultiCorePartition(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    const int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    const int32_t mTiles = CeilDiv<int32_t>(tiling.m, tiling.baseM);
    const int32_t nTiles = CeilDiv<int32_t>(tiling.n, tiling.baseN);
    const int32_t kTotal = tiling.k;
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = std::max(1, std::min(nTiles, maxCores));
    int64_t bestCost = std::numeric_limits<int64_t>::max();
    int32_t bestCores = 0;
    for (int32_t mBlocks = 1; mBlocks <= mTiles && mBlocks <= maxCores; ++mBlocks) {
        int32_t nBlocks = std::max(1, std::min(nTiles, maxCores / mBlocks));
        int64_t cost = HgemmCoreGroupCost(CeilDiv(mTiles, mBlocks), CeilDiv(nTiles, nBlocks), kTotal);
        int32_t cores = mBlocks * nBlocks;
        if (cost < bestCost || (cost == bestCost && cores > bestCores)) {
            bestCost = cost;
            bestMBlocks = mBlocks;
            bestNBlocks = nBlocks;
            bestCores = cores;
        }
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    tiling.usedCoreNum = bestMBlocks * bestNBlocks;
}

static void PrepareHgemmCubeTiling(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    ApplyGemmColMajorSwap(tiling);
    CalcHgemmMultiCorePartition(tiling, cubeCoreNum);
}

// ==========================================================================
//  Parameter validation (order per task spec sec. 2.5: alpha/beta null checks
//  precede the m/n short-circuit; A/B/C null checks follow it)
// ==========================================================================
static aclblasStatus_t ValidateHgemmParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, int lda, int ldb, const void* beta, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasHgemm", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) {
        OP_LOGE("aclblasHgemm", "invalid transa: %d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C) {
        OP_LOGE("aclblasHgemm", "invalid transb: %d", static_cast<int>(transb));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0) {
        OP_LOGE("aclblasHgemm", "m, n, k must be non-negative: m=%d, n=%d, k=%d", m, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int ldaMin = isTransA ? std::max(1, k) : std::max(1, m);
    if (lda < ldaMin) {
        OP_LOGE("aclblasHgemm", "lda=%d must be >= %d", lda, ldaMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int ldbMin = isTransB ? std::max(1, n) : std::max(1, k);
    if (ldb < ldbMin) {
        OP_LOGE("aclblasHgemm", "ldb=%d must be >= %d", ldb, ldbMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("aclblasHgemm", "ldc=%d must be >= %d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasHgemm", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclblasHgemm", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateHgemmPointers(
    int k, float alphaAbs, const void* A, const void* B, float betaAbs, const void* C)
{
    if (k > 0 && alphaAbs != 0.0f) {
        if (A == nullptr) {
            OP_LOGE("aclblasHgemm", "A must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (B == nullptr) {
            OP_LOGE("aclblasHgemm", "B must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (k > 0) {
        if (C == nullptr) {
            OP_LOGE("aclblasHgemm", "C must not be nullptr when k > 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (betaAbs != 0.0f && C == nullptr) {
        OP_LOGE("aclblasHgemm", "C must not be nullptr when beta != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  HandleHgemmAlphaZero - C = beta * C when k == 0 or alpha == 0
// ==========================================================================
static aclblasStatus_t HandleHgemmAlphaZero(aclblasHandle_t handle, int m, int n, int ldc, float betaVal, void* C)
{
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 0.0f) {
        // Zero the m*n logical region only: BLAS leaves the ldc padding rows
        // untouched (beta == 0 must not read C, which may hold NaN garbage).
        auto* hZero = reinterpret_cast<_aclblas_handle*>(handle);
        if (ldc == m) {
            size_t cBytes = static_cast<size_t>(ldc) * n * sizeof(aclblasHalf);
            // Async memset ordered on the handle stream keeps the BLAS
            // asynchronous contract: in-flight kernels queued earlier on the
            // same stream complete before the zeroing, so a prior GEMM cannot
            // overwrite the zeroed result.
            aclError ret = aclrtMemsetAsync(C, cBytes, 0, cBytes, hZero->stream);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasHgemm", "aclrtMemsetAsync failed, ret=%d", ret);
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
            return ACLBLAS_STATUS_SUCCESS;
        }
        uint32_t aivCoreNumZero = GetAivCoreCount();
        if (aivCoreNumZero == 0) {
            OP_LOGE("aclblasHgemm", "GetAivCoreCount failed");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        uint32_t vecBlocksZero = std::min(aivCoreNumZero, static_cast<uint32_t>(n));
        hgemm_scale_do(vecBlocksZero, hZero->stream, reinterpret_cast<uint8_t*>(C), m, n, ldc, 0.0f);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasHgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    hgemm_scale_do(vecBlocks, h->stream, reinterpret_cast<uint8_t*>(C), m, n, ldc, betaVal);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Launch path
// ==========================================================================
static aclblasStatus_t PrepareHgemmWorkspace(_aclblas_handle* h, int m, int n, uint8_t*& tempABDevice)
{
    tempABDevice = nullptr;
    size_t workspaceNeed = static_cast<size_t>(CeilAlign(m, HGEMM_FRACTAL)) * n * sizeof(float);
    aclblasStatus_t st = EnsureDefaultWorkspace(h, workspaceNeed);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasHgemm", "workspace need %zu bytes", workspaceNeed);
        return st;
    }
    tempABDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (reinterpret_cast<uintptr_t>(tempABDevice) % 512 != 0) {
        OP_LOGW("aclblasHgemm", "workspace address not 512B aligned: %p", tempABDevice);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchHgemmAlphaBetaKernel(
    aclrtStream stream, uint8_t* tempABDevice, void* C, int n, const GemmTilingData& abTiling)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasHgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    GemmTilingData tiling = abTiling;
    tiling.usedCoreNum = static_cast<int32_t>(vecBlocks);
    hgemm_alpha_beta_do(
        vecBlocks, stream, tempABDevice, reinterpret_cast<uint8_t*>(C), reinterpret_cast<uint8_t*>(C), tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchHgemmKernel(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, float alphaVal,
    const aclblasHalf* A, int lda, const aclblasHalf* B, int ldb, float betaVal, aclblasHalf* C, int ldc)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    uint32_t cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasHgemm", "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (k == 0 || alphaVal == 0.0f) {
        return HandleHgemmAlphaZero(handle, m, n, ldc, betaVal, C);
    }

    bool needPostProcess = (alphaVal != 1.0f) || (betaVal != 0.0f);

    GemmTilingData tiling = BuildGemmTilingData(m, n, k, lda, ldb, ldc, transa, transb, alphaVal, 0.0f, betaVal, 0.0f);
    tiling.baseM = HGEMM_BASE_M;
    tiling.baseN = HGEMM_BASE_N;
    tiling.baseK = HGEMM_BASE_K;
    tiling.tileKChunk = HGEMM_TILE_K_CHUNK;
    tiling.c0Size = HGEMM_C0_SIZE;

    if (needPostProcess) {
        tiling.ldc = static_cast<int32_t>(CeilAlign(m, HGEMM_FRACTAL));
    }
    GemmTilingData abTiling = tiling;

    PrepareHgemmCubeTiling(tiling, cubeCoreNum);

    uint32_t numBlocks = static_cast<uint32_t>(tiling.usedCoreNum);
    uint8_t* aDevicePtr = reinterpret_cast<uint8_t*>(const_cast<aclblasHalf*>(B));
    uint8_t* bDevicePtr = reinterpret_cast<uint8_t*>(const_cast<aclblasHalf*>(A));

    uint8_t* tempABDevice = nullptr;
    if (needPostProcess) {
        aclblasStatus_t st = PrepareHgemmWorkspace(h, m, n, tempABDevice);
        if (st != ACLBLAS_STATUS_SUCCESS)
            return st;
    }
    uint8_t* cDevicePtr = needPostProcess ? tempABDevice : reinterpret_cast<uint8_t*>(C);

    hgemm_kernel_do(numBlocks, stream, aDevicePtr, bDevicePtr, cDevicePtr, needPostProcess ? 1 : 0, tiling);

    if (needPostProcess) {
        return LaunchHgemmAlphaBetaKernel(stream, tempABDevice, C, n, abTiling);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  aclblasHgemm - public API entry (FLOAT16)
// ==========================================================================
extern "C" aclblasStatus_t aclblasHgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasHalf* alpha, const aclblasHalf* A, int lda, const aclblasHalf* B, int ldb, const aclblasHalf* beta,
    aclblasHalf* C, int ldc)
{
    OP_LOGI(
        "aclblasHgemm", "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k);

    aclblasStatus_t st = ValidateHgemmParams(handle, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    float alphaVal = blas_common::HalfToFloat(*alpha);
    float betaVal = blas_common::HalfToFloat(*beta);

    float alphaAbs = std::abs(alphaVal);
    st = ValidateHgemmPointers(k, alphaAbs, A, B, betaVal, C);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    return LaunchHgemmKernel(handle, transa, transb, m, n, k, alphaVal, A, lda, B, ldb, betaVal, C, ldc);
}

#endif
