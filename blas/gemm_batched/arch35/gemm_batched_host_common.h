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
 * \file gemm_batched_host_common.h
 * \brief Shared host helpers for the GEMM-batched family (Sgemm/Hgemm/Bf16 ...).
 *
 * These helpers are dtype-agnostic: the element type only affects the tiling
 * dtypeCase and the log tag, both of which are passed in by the caller.  Keeping
 * them in one place avoids the large cross-file duplication between
 * gemm_batched_host.cpp and hgemmBatched_host.cpp.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "common/helper/host_utils.h"
#include "common/helper/aclblas_handle_internal.h"
#include "gemm_batched_tiling_data.h"

// Kernel launchers (defined in gemm_batched_kernel.cpp)
void gemm_batched_gemm_kernel_do(
    uint32_t numBlocks, void* stream, const uint8_t* aarray, const uint8_t* barray, uint8_t* carray,
    const GemmBatchedGemmTilingData& tilingData);

void gemm_batched_alpha_beta_kernel_do(
    uint32_t numBlocks, void* stream, const uint8_t* tempAB, uint8_t* carray,
    const GemmBatchedAlphaBetaTilingData& tilingData);

void gemm_batched_early_exit_kernel_do(
    uint32_t numBlocks, void* stream, uint8_t* carray, const GemmBatchedAlphaBetaTilingData& tilingData);

// Minimum elements per AIV core to amortize launch overhead for post-process kernels.
constexpr int64_t GEMM_BATCHED_AIV_ELEMENTS_PER_CORE = 16384;

// ============================================================================
// Workspace management
// ============================================================================
static inline aclblasStatus_t GemmBatchedEnsureWorkspace(_aclblas_handle* h, size_t requiredSize)
{
    return EnsureDefaultWorkspace(h, requiredSize);
}

// ============================================================================
// Tiling computation
// ============================================================================
static inline int32_t GemmBatchedGreatestCommonDivisor(int32_t lhs, int32_t rhs)
{
    while (rhs != 0) {
        const int32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

// Balance the trailing wave when the batch count alone already fills the Cube
// cores: pick the smallest N split (1..cores) that makes batchCount*nBlocks land
// closest to a whole number of core waves (fewest idle cores in the last wave),
// subject to the matrix being wide enough to split without an over-narrow tile.
static inline int32_t GemmBatchedPickTrailingWaveNBlocks(int n, int batchCount, uint32_t cubeCoreNum)
{
    const int maxNBlocks = std::max(1, n / 128);
    int bestN = 1;
    int64_t bestIdle = (static_cast<int64_t>(cubeCoreNum) - batchCount % static_cast<int64_t>(cubeCoreNum)) %
                       static_cast<int64_t>(cubeCoreNum);
    for (int nb = 2; nb <= maxNBlocks; nb++) {
        const int64_t total = static_cast<int64_t>(batchCount) * nb;
        const int64_t idle = (static_cast<int64_t>(cubeCoreNum) - (total % static_cast<int64_t>(cubeCoreNum))) %
                             static_cast<int64_t>(cubeCoreNum);
        if (idle < bestIdle) {
            bestIdle = idle;
            bestN = nb;
        }
    }
    return bestN;
}

static inline void GemmBatchedFindBestSpatialSplit(
    int m, int n, int32_t mTiles, int32_t nTiles, int32_t spatialCoreTarget, uint32_t baseM, uint32_t baseN,
    int32_t& bestMBlocks, int32_t& bestNBlocks)
{
    bestMBlocks = 1;
    bestNBlocks = 1;
    int32_t bestUtil = 0;
    uint64_t bestTileLoops = UINT64_MAX;
    uint64_t bestAspectDelta = UINT64_MAX;
    for (int32_t mb = 1; mb <= mTiles && mb <= spatialCoreTarget; mb++) {
        int32_t nb = std::min(nTiles, spatialCoreTarget / mb);
        if (nb < 1)
            nb = 1;
        int32_t util = mb * nb;
        const uint32_t blockM = CeilAlign(static_cast<uint32_t>((m + mb - 1) / mb), baseM);
        const uint32_t blockN = CeilAlign(static_cast<uint32_t>((n + nb - 1) / nb), baseN);
        const uint64_t tileLoops =
            static_cast<uint64_t>((blockM + GEMM_BATCHED_DEFAULT_TILE_M - 1) / GEMM_BATCHED_DEFAULT_TILE_M) *
            ((blockN + GEMM_BATCHED_DEFAULT_TILE_N - 1) / GEMM_BATCHED_DEFAULT_TILE_N);
        const uint64_t scaledM = static_cast<uint64_t>(blockM) * GEMM_BATCHED_DEFAULT_TILE_N;
        const uint64_t scaledN = static_cast<uint64_t>(blockN) * GEMM_BATCHED_DEFAULT_TILE_M;
        const uint64_t aspectDelta = scaledM > scaledN ? scaledM - scaledN : scaledN - scaledM;
        const bool betterShape =
            tileLoops < bestTileLoops || (tileLoops == bestTileLoops && tileLoops > 1 && aspectDelta < bestAspectDelta);
        if (util <= spatialCoreTarget && (util > bestUtil || (util == bestUtil && betterShape))) {
            bestUtil = util;
            bestMBlocks = mb;
            bestNBlocks = nb;
            bestTileLoops = tileLoops;
            bestAspectDelta = aspectDelta;
        }
    }
    if (bestUtil == 0) {
        bestMBlocks = 1;
        bestNBlocks = 1;
    }
}

// useTrailingWaveBalance selects the batch-bound scheduling strategy once the
// batch count alone can fill the Cube cores: when true, split only along N to
// minimise trailing-wave idle cores (Hgemm); when false, keep the historical
// "smallest split that lands on a whole number of waves" search (Sgemm).
static inline void GemmBatchedCalcMnBlockSplit(
    int m, int n, int batchCount, uint32_t cubeCoreNum, bool useTrailingWaveBalance, int32_t& bestMBlocks,
    int32_t& bestNBlocks)
{
    uint32_t baseM = GEMM_BATCHED_BASE_M;
    uint32_t baseN = GEMM_BATCHED_BASE_N;
    // When the batch count alone already fills (or overfills) the Cube cores,
    // one task per batch is optimal: splitting a matrix spatially only adds
    // more, smaller tasks whose per-task fixed latency (L1 staging + pipeline
    // drain) hurts overall throughput.  Only split spatially when too few
    // batches are available to saturate the cores.
    if (useTrailingWaveBalance && batchCount >= static_cast<int>(cubeCoreNum)) {
        bestMBlocks = 1;
        bestNBlocks = GemmBatchedPickTrailingWaveNBlocks(n, batchCount, cubeCoreNum);
        return;
    }
    int32_t mTiles = (m + baseM - 1) / baseM;
    int32_t nTiles = (n + baseN - 1) / baseN;
    const int32_t batchCoreGcd = GemmBatchedGreatestCommonDivisor(static_cast<int32_t>(cubeCoreNum), batchCount);
    const int32_t spatialCoreTarget =
        std::max<int32_t>(1, static_cast<int32_t>(cubeCoreNum) / std::max<int32_t>(1, batchCoreGcd));
    GemmBatchedFindBestSpatialSplit(
        m, n, mTiles, nTiles, spatialCoreTarget, baseM, baseN, bestMBlocks, bestNBlocks);
}

static inline aclblasStatus_t GemmBatchedCalcGemmTiling(
    const char* opName, int dtypeCase, bool useTrailingWaveBalance, int m, int n, int k, int batchCount, int lda,
    int ldb, int ldc, bool isTransA, bool isTransB, uint32_t cubeCoreNum, GemmBatchedGemmTilingData& tiling)
{
    tiling = GemmBatchedGemmTilingData{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.isTransA = isTransA ? 1 : 0;
    tiling.isTransB = isTransB ? 1 : 0;
    tiling.batchCount = static_cast<uint32_t>(batchCount);
    tiling.dtypeCase = dtypeCase;

    tiling.tileM = GEMM_BATCHED_DEFAULT_TILE_M;
    tiling.tileN = GEMM_BATCHED_DEFAULT_TILE_N;
    tiling.tileKChunk = GEMM_BATCHED_DEFAULT_TILE_K_CHUNK;

    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = 1;
    GemmBatchedCalcMnBlockSplit(
        m, n, batchCount, cubeCoreNum, useTrailingWaveBalance, bestMBlocks, bestNBlocks);

    uint32_t baseM = GEMM_BATCHED_BASE_M;
    uint32_t baseN = GEMM_BATCHED_BASE_N;
    tiling.mBlocks = static_cast<uint32_t>(bestMBlocks);
    tiling.nBlocks = static_cast<uint32_t>(bestNBlocks);
    tiling.singleCoreM = static_cast<uint32_t>(((m + bestMBlocks - 1) / bestMBlocks + baseM - 1) / baseM * baseM);
    tiling.singleCoreN = static_cast<uint32_t>(((n + bestNBlocks - 1) / bestNBlocks + baseN - 1) / baseN * baseN);
    if (tiling.singleCoreM > static_cast<uint32_t>(m))
        tiling.singleCoreM = m;
    if (tiling.singleCoreN > static_cast<uint32_t>(n))
        tiling.singleCoreN = n;

    int64_t totalTasks64 = static_cast<int64_t>(batchCount) * bestMBlocks * bestNBlocks;
    if (totalTasks64 > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(opName, "totalTasks overflow: %lld", static_cast<long long>(totalTasks64));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling.totalTasks = static_cast<uint32_t>(totalTasks64);
    tiling.usedAicCoreNum = static_cast<uint32_t>(std::min(static_cast<int64_t>(cubeCoreNum), totalTasks64));

    return ACLBLAS_STATUS_SUCCESS;
}

static inline void GemmBatchedApplyColumnMajorSwap(GemmBatchedGemmTilingData& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
    std::swap(tiling.singleCoreM, tiling.singleCoreN);
    std::swap(tiling.mBlocks, tiling.nBlocks);
}

// ============================================================================
// Workspace helpers
// ============================================================================
static inline aclblasStatus_t GemmBatchedPrepareTempWorkspace(
    const char* opName, _aclblas_handle* h, int batchCount, int origN, uint32_t tempRowStride,
    GemmBatchedGemmTilingData& tilingData, uint8_t*& workspace, size_t& alignedPtrArrayBytes)
{
    tilingData.ldc = tempRowStride;
    // The Cube kernel always writes the FP32 accumulator temporary, regardless of
    // the input dtype, so the temporary buffer size uses sizeof(float).
    size_t perBatchTempBytes = static_cast<size_t>(origN) * tempRowStride * sizeof(float);
    size_t totalTempBytes = static_cast<size_t>(batchCount) * perBatchTempBytes;

    constexpr size_t TEMP_DATA_ALIGN = 64;
    size_t ptrArrayBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    alignedPtrArrayBytes = (ptrArrayBytes + TEMP_DATA_ALIGN - 1) & ~(TEMP_DATA_ALIGN - 1);
    size_t totalWorkspaceBytes = alignedPtrArrayBytes + totalTempBytes;

    aclblasStatus_t wsRet = GemmBatchedEnsureWorkspace(h, totalWorkspaceBytes);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(opName, "workspace ensure failed, required=%zu", totalWorkspaceBytes);
        return wsRet;
    }
    workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(h));

    std::vector<uint64_t> tempPtrs(batchCount);
    uint64_t tempDataBase = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workspace + alignedPtrArrayBytes));
    for (int i = 0; i < batchCount; i++) {
        tempPtrs[i] = tempDataBase + static_cast<uint64_t>(i) * perBatchTempBytes;
    }
    aclError aclRet = aclrtMemcpy(workspace, ptrArrayBytes, tempPtrs.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opName, "Failed to copy temp ptr array, err=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t GemmBatchedValidateCoreCounts(
    const char* opName, uint32_t& cubeCoreNum, uint32_t& aivCoreNum)
{
    cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE(opName, "cube core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE(opName, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Enlarge the output tile for medium/large operands to cut redundant A/B L1
// reloads.  tiling.m/n are post-swap, matching the kernel's Cube view.
static inline void GemmBatchedEnlargeOutputTileForLargeOperands(GemmBatchedGemmTilingData& tiling)
{
    if (tiling.m >= 256 && tiling.n >= 256) {
        // Large: whole-L0C 256x256 output tile (single accumulator bank).  This
        // minimizes the A/B reload factor that dominates MTE2 traffic.
        tiling.tileM = 256;
        tiling.tileN = 256;
        if (tiling.k >= 512) {
            tiling.tileKChunk = 256;
        }
    } else if (tiling.m >= 128 && tiling.n >= 128) {
        // 128-wide operands that still exceed one tile: use a 256x128 strip
        // (double-buffered L0C) to halve the reload factor versus 128x128.
        if (tiling.m >= tiling.n) {
            tiling.tileM = 256;
            tiling.tileN = 128;
        } else {
            tiling.tileM = 128;
            tiling.tileN = 256;
        }
    }
}

// Dtype-agnostic alpha/beta post-process launch: the C pointer array is passed
// as a raw byte pointer and reinterpreted inside the launcher.
static inline void GemmBatchedLaunchAlphaBetaPostProcess(
    const char* opName, int dtypeCase, _aclblas_handle* h, int origM, int origN, int origLdc, uint32_t tempRowStride,
    float alphaVal, float betaVal, int batchCount, uint32_t aivCoreNum, uint8_t* workspace,
    size_t alignedPtrArrayBytes, uint8_t* carrayPtrAB)
{
    GemmBatchedAlphaBetaTilingData abTiling{};
    abTiling.m = origM;
    abTiling.n = origN;
    abTiling.ldc = origLdc;
    abTiling.tempRowStride = static_cast<int32_t>(tempRowStride);
    abTiling.alpha = alphaVal;
    abTiling.beta = betaVal;
    abTiling.hasBeta = (betaVal != 0.0f) ? 1 : 0;
    abTiling.batchCount = batchCount;
    abTiling.dtypeCase = dtypeCase;
    abTiling.totalCols = static_cast<int64_t>(batchCount) * origN;

    int64_t totalElements = static_cast<int64_t>(batchCount) * origM * origN;
    abTiling.usedAivCoreNum = static_cast<int32_t>(std::min(
        static_cast<int64_t>(aivCoreNum),
        std::max(
            static_cast<int64_t>(1),
            (totalElements + GEMM_BATCHED_AIV_ELEMENTS_PER_CORE - 1) / GEMM_BATCHED_AIV_ELEMENTS_PER_CORE)));

    const uint8_t* tempABData = workspace + alignedPtrArrayBytes;

    OP_LOGI(
        opName, "launching alpha/beta kernel: cores=%d, alpha=%.4f, beta=%.4f, batch=%d, m=%d, n=%d",
        abTiling.usedAivCoreNum, alphaVal, betaVal, batchCount, origM, origN);

    gemm_batched_alpha_beta_kernel_do(
        static_cast<uint32_t>(abTiling.usedAivCoreNum), h->stream, tempABData, carrayPtrAB, abTiling);
}

// Early-exit launch for the k==0 / alpha==0 degenerate case (C = beta*C).
static inline aclblasStatus_t GemmBatchedLaunchEarlyExit(
    const char* opName, int dtypeCase, _aclblas_handle* h, int m, int n, int ldc, int batchCount, float betaVal,
    uint8_t* carrayPtr)
{
    if (betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS; // C unchanged
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE(opName, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    GemmBatchedAlphaBetaTilingData eeTiling{};
    eeTiling.m = m;
    eeTiling.n = n;
    eeTiling.ldc = ldc;
    eeTiling.tempRowStride = ldc;
    eeTiling.alpha = 0.0f;
    eeTiling.beta = betaVal;
    eeTiling.hasBeta = (betaVal == 0.0f) ? 0 : 1;
    eeTiling.batchCount = batchCount;
    eeTiling.dtypeCase = dtypeCase;
    eeTiling.totalCols = static_cast<int64_t>(batchCount) * n;

    int64_t totalElements = static_cast<int64_t>(batchCount) * m * n;
    eeTiling.usedAivCoreNum = static_cast<int32_t>(std::min(
        static_cast<int64_t>(aivCoreNum),
        std::max(
            static_cast<int64_t>(1),
            (totalElements + GEMM_BATCHED_AIV_ELEMENTS_PER_CORE - 1) / GEMM_BATCHED_AIV_ELEMENTS_PER_CORE)));

    OP_LOGI(
        opName, "early exit: beta=%.4f, batch=%d, m=%d, n=%d, cores=%d", betaVal, batchCount, m, n,
        eeTiling.usedAivCoreNum);

    gemm_batched_early_exit_kernel_do(static_cast<uint32_t>(eeTiling.usedAivCoreNum), h->stream, carrayPtr, eeTiling);

    return ACLBLAS_STATUS_SUCCESS;
}
