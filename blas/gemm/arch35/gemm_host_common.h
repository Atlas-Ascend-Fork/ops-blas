/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include "cann_ops_blas.h"
#include "gemm_tiling_data.h"

inline void CalcGemmMultiCorePartition(GemmTilingData& tiling, uint32_t cubeCoreNum, int m, int n)
{
    const int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    const int32_t mTiles =
        static_cast<int32_t>((static_cast<int64_t>(m) + static_cast<int64_t>(tiling.baseM) - 1) / tiling.baseM);
    const int32_t nTiles =
        static_cast<int32_t>((static_cast<int64_t>(n) + static_cast<int64_t>(tiling.baseN) - 1) / tiling.baseN);
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = 1;
    int32_t bestUtilization = 0;
    for (int32_t mBlocks = 1; mBlocks <= mTiles && mBlocks <= maxCores; ++mBlocks) {
        const int32_t nBlocks = std::max(1, std::min(nTiles, maxCores / mBlocks));
        const int32_t utilization = mBlocks * nBlocks;
        if (utilization > bestUtilization && utilization <= maxCores) {
            bestUtilization = utilization;
            bestMBlocks = mBlocks;
            bestNBlocks = nBlocks;
        }
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    tiling.usedCoreNum = bestMBlocks * bestNBlocks;
}

inline GemmTilingData BuildGemmTilingData(
    int m, int n, int k, int lda, int ldb, int ldc, aclblasOperation_t transa, aclblasOperation_t transb,
    float alphaReal, float alphaImag, float betaReal, float betaImag)
{
    GemmTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.cLdc = ldc;
    tiling.baseM = GEMM_BASE_M;
    tiling.baseN = GEMM_BASE_N;
    tiling.baseK = GEMM_BASE_K;
    tiling.tileKChunk = GEMM_TILE_K_CHUNK;
    tiling.c0Size = GEMM_C0_SIZE;
    tiling.isTransA = transa != ACLBLAS_OP_N ? 1 : 0;
    tiling.isTransB = transb != ACLBLAS_OP_N ? 1 : 0;
    tiling.alphaReal = alphaReal;
    tiling.alphaImag = alphaImag;
    tiling.betaReal = betaReal;
    tiling.betaImag = betaImag;
    tiling.hasBeta = betaReal != 0.0f || betaImag != 0.0f ? 1 : 0;
    return tiling;
}

inline void ApplyGemmColMajorSwap(GemmTilingData& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
}

inline void PrepareGemmCubeTiling(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    ApplyGemmColMajorSwap(tiling);
    CalcGemmMultiCorePartition(tiling, cubeCoreNum, tiling.m, tiling.n);
}

// Internal entry, after public argument validation.

aclblasStatus_t LaunchCgemmThreeProducts(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex& alpha, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb,
    const aclblasComplex& beta, aclblasComplex* c, int ldc);

aclblasStatus_t LaunchCgemm3mUnitBatches(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const aclblasComplex* a,
    int lda, int64_t strideA, const aclblasComplex* b, int ldb, int64_t strideB, aclblasComplex* c, int ldc,
    int64_t strideC, int batch);

aclblasStatus_t LaunchCgemm3mOnchipBatches(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const aclblasComplex* a,
    int lda, int64_t strideA, const aclblasComplex* b, int ldb, int64_t strideB, aclblasComplex* c, int ldc,
    int64_t strideC, int batch);
