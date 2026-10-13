/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file cgeam_host.cpp
 * @brief Host-side implementation for Cgeam operator (complex)
 *        Handles parameter validation, tiling calculation, and kernel launch
 */

#include "cann_ops_blas.h"
#include "cgeam_kernel.h"
#include "geam_host_common.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include <algorithm>
#include <limits>

namespace {

using CgeamConstants::BLOCK_BYTES;
using CgeamConstants::L2_CAPACITY_BYTES;
using CgeamConstants::WORKSET_MATRIX_COUNT;

constexpr uint32_t COMPLEX_PER_BLOCK = BLOCK_BYTES / sizeof(aclblasComplex);
// The packed path uses two 64 KiB queue slots for A/C and one 64 KiB slot for B.
// Derive the element count from the per-slot allocation instead of a matrix shape.
constexpr uint32_t FAST_PATH_BUFFER_BYTES = 64U * 1024U;
constexpr uint32_t FAST_PATH_TILE_COMPLEX = FAST_PATH_BUFFER_BYTES / sizeof(aclblasComplex);
// Keep very small jobs on one AIV core to avoid multicore scheduling overhead.
constexpr uint32_t NN_MULTICORE_MIN_WORK_BYTES = 16U * 1024U;
constexpr uint32_t NN_MULTICORE_MIN_COMPLEX = NN_MULTICORE_MIN_WORK_BYTES / sizeof(aclblasComplex);
// Keep a constant 4096-complex tile area. Capacity-range transpose inputs use
// a taller tile so C stores remain contiguous over a longer address range.
constexpr uint32_t TRANSPOSE_TILE_ROWS = 64;
constexpr uint32_t TRANSPOSE_TILE_COLS = 64;
constexpr uint32_t TALL_TRANSPOSE_TILE_ROWS = 128;
constexpr uint32_t TALL_TRANSPOSE_TILE_COLS = 32;
constexpr uint32_t SIMT_THREAD_GRANULARITY = 128;
constexpr uint32_t SIMT_SCHEDULING_THREAD_CAP = 1024;

bool HasUnitScalars(float alphaR, float alphaI, float betaR, float betaI)
{
    return alphaR == 1.0f && alphaI == 0.0f && betaR == 1.0f && betaI == 0.0f;
}

void SelectTransposeTileShape(
    uint32_t m, uint32_t n, uint32_t& tileRows, uint32_t& tileCols)
{
    // Let S be one matrix's size. Tall tiles cover the middle two capacity tiers:
    // 1) 3S < L2; 2) 3S >= L2 and 2S < L2; 3) 2S >= L2 and S <= L2; 4) S > L2.
    const uint64_t matrixBytes = static_cast<uint64_t>(m) * n * sizeof(aclblasComplex);
    // Called only for valid transpose inputs. Capacity tiers 2-3 use taller tiles.
    const bool tall = WORKSET_MATRIX_COUNT * matrixBytes >= L2_CAPACITY_BYTES && matrixBytes <= L2_CAPACITY_BYTES;
    tileRows = tall ? TALL_TRANSPOSE_TILE_ROWS : TRANSPOSE_TILE_ROWS;
    tileCols = tall ? TALL_TRANSPOSE_TILE_COLS : TRANSPOSE_TILE_COLS;
}

bool IsNNFastPath(const CgeamTilingData& tiling)
{
    const bool copyA = tiling.alphaR == 1.0f && tiling.alphaI == 0.0f && tiling.betaIsZero != 0U;
    const bool addAB = tiling.opB == ACLBLAS_OP_N && tiling.ldb == tiling.m &&
        HasUnitScalars(tiling.alphaR, tiling.alphaI, tiling.betaR, tiling.betaI);
    return tiling.opA == ACLBLAS_OP_N && tiling.lda == tiling.m && tiling.ldc == tiling.m &&
        (copyA || addAB);
}

void ConfigureNNFastPath(uint32_t aivCoreNum, CgeamTilingData& tiling)
{
    const uint64_t totalComplex = static_cast<uint64_t>(tiling.m) * tiling.n;
    const uint64_t alignedTaskCount = (totalComplex + COMPLEX_PER_BLOCK - 1U) / COMPLEX_PER_BLOCK;
    const uint32_t numBlocks = totalComplex < NN_MULTICORE_MIN_COMPLEX ? 1U :
        static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum, alignedTaskCount));
    const uint64_t rawPerCore = totalComplex / numBlocks;
    tiling.fastPath = 1U;
    tiling.fastPerCoreComplex = (rawPerCore / COMPLEX_PER_BLOCK) * COMPLEX_PER_BLOCK;
    const uint64_t leftover = totalComplex - tiling.fastPerCoreComplex * numBlocks;
    tiling.fastExtraBlockCores = static_cast<uint32_t>(leftover / COMPLEX_PER_BLOCK);
    tiling.fastTailComplex = static_cast<uint32_t>(leftover % COMPLEX_PER_BLOCK);
    tiling.fastTileComplex = FAST_PATH_TILE_COMPLEX;

    // Launch exactly numBlocks AIV blocks.
    tiling.blockCount = numBlocks;
}

bool HasCompactTransposeInputs(const CgeamTilingData& tiling)
{
    const bool anyTransposed = tiling.opA != ACLBLAS_OP_N || tiling.opB != ACLBLAS_OP_N;
    const bool compactA = tiling.lda == (tiling.opA == ACLBLAS_OP_N ? tiling.m : tiling.n);
    const bool compactB = tiling.ldb == (tiling.opB == ACLBLAS_OP_N ? tiling.m : tiling.n);
    const bool copyA = tiling.opA != ACLBLAS_OP_N && tiling.alphaR == 1.0f && tiling.alphaI == 0.0f &&
        tiling.betaIsZero != 0U;
    const bool addAB = anyTransposed && compactB &&
        HasUnitScalars(tiling.alphaR, tiling.alphaI, tiling.betaR, tiling.betaI);
    return compactA && tiling.ldc == tiling.m && (copyA || addAB);
}

bool ConfigureTransposeTilePath(uint32_t aivCoreNum, CgeamTilingData& tiling)
{
    if (!HasCompactTransposeInputs(tiling)) {
        return false;
    }
    const uint32_t m = tiling.m;
    const uint32_t n = tiling.n;
    uint32_t tileRows = 0U;
    uint32_t tileCols = 0U;
    SelectTransposeTileShape(m, n, tileRows, tileCols);
    const uint64_t totalComplex = static_cast<uint64_t>(m) * n;
    const uint64_t rowTiles = (static_cast<uint64_t>(m) + tileRows - 1U) / tileRows;
    uint64_t colTiles = (static_cast<uint64_t>(n) + tileCols - 1U) / tileCols;
    if (totalComplex < tileRows * tileCols || rowTiles * colTiles > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    // Merge adjacent input columns only for matching transposed operations with
    // unaligned row strides. Aligned inputs and mixed conjugation retain the
    // standard tiling. Four 60 KiB buffers preserve two pipeline stages.
    constexpr uint32_t MERGED_TILE_COLS = 60U;
    constexpr uint32_t INPUT_BURST_ALIGNMENT_BYTES = 128U;
    const bool sameTranspose = tiling.opA != ACLBLAS_OP_N && tiling.opA == tiling.opB;
    const bool merged = tiling.betaIsZero == 0U && sameTranspose && tileRows == TALL_TRANSPOSE_TILE_ROWS &&
        (static_cast<uint64_t>(n) * sizeof(aclblasComplex)) % INPUT_BURST_ALIGNMENT_BYTES != 0U;
    if (merged) {
        tileCols = MERGED_TILE_COLS;
        // A wider tile can only decrease the column count already checked above.
        colTiles = (static_cast<uint64_t>(n) + tileCols - 1U) / tileCols;
    }
    const uint32_t totalTiles = static_cast<uint32_t>(rowTiles * colTiles);
    const uint32_t numBlocks = std::min(aivCoreNum, totalTiles);
    tiling.transposeTilePath = merged ? CgeamTransposePath::MERGED : CgeamTransposePath::STANDARD;
    tiling.transposeTileRows = tileRows;
    tiling.transposeTileCols = tileCols;
    tiling.transposeRowTiles = static_cast<uint32_t>(rowTiles);
    tiling.transposeTotalTiles = totalTiles;
    tiling.blockCount = numBlocks;
    return true;
}

void ConfigureGeneralSimtPath(uint32_t aivCoreNum, CgeamTilingData& tiling)
{
    const uint64_t totalComplex = static_cast<uint64_t>(tiling.m) * tiling.n;
    const uint32_t numBlocks = static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum, totalComplex));
    const uint64_t perCore = (totalComplex + numBlocks - 1U) / numBlocks;
    const uint64_t alignedThreads =
        (perCore + SIMT_THREAD_GRANULARITY - 1U) / SIMT_THREAD_GRANULARITY * SIMT_THREAD_GRANULARITY;
    tiling.simtThreads =
        static_cast<uint32_t>(std::min<uint64_t>(alignedThreads, SIMT_SCHEDULING_THREAD_CAP));
    tiling.blockCount = numBlocks;
}

CgeamTilingData CalCgeamTilingData(
    aclblasOperation_t transa, aclblasOperation_t transb, uint32_t m, uint32_t n, float alphaR, float alphaI,
    float betaR, float betaI, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t alphaIsZero, uint32_t betaIsZero,
    uint32_t aivCoreNum)
{
    CgeamTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.alphaR = alphaR;
    tiling.alphaI = alphaI;
    tiling.betaR = betaR;
    tiling.betaI = betaI;
    tiling.alphaIsZero = alphaIsZero;
    tiling.betaIsZero = betaIsZero;
    tiling.opA = transa;
    tiling.opB = transb;

    if (IsNNFastPath(tiling)) {
        ConfigureNNFastPath(aivCoreNum, tiling);
        return tiling;
    }
    if (ConfigureTransposeTilePath(aivCoreNum, tiling)) {
        return tiling;
    }
    // All remaining valid non-empty inputs use the general path.
    ConfigureGeneralSimtPath(aivCoreNum, tiling);
    return tiling;
}

aclblasStatus_t LaunchCgeamKernel(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* beta, const aclblasComplex* B,
    int ldb, aclblasComplex* C, int ldc)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgeam", "Failed to get AIV core count");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    float alphaR = alpha->real;
    float alphaI = alpha->imag;
    float betaR = beta->real;
    float betaI = beta->imag;
    uint32_t alphaIsZero = (alphaR == 0.0f && alphaI == 0.0f) ? 1u : 0u;
    uint32_t betaIsZero = (betaR == 0.0f && betaI == 0.0f) ? 1u : 0u;

    CgeamTilingData tiling = CalCgeamTilingData(
        transa, transb, static_cast<uint32_t>(m), static_cast<uint32_t>(n), alphaR, alphaI, betaR, betaI,
        static_cast<uint32_t>(lda), static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc), alphaIsZero, betaIsZero,
        aivCoreNum);
    // Validated non-empty input and nonzero AIV count guarantee a nonzero grid.
    const uint32_t numBlocks = tiling.blockCount;
    OP_LOGD(
        "aclblasCgeam",
        "Tiling: fastPath=%u, transposeTilePath=%u, simtThreads=%u, fastTile=%u, "
        "totalBlocks=%u",
        tiling.fastPath, static_cast<uint32_t>(tiling.transposeTilePath),
        tiling.simtThreads, tiling.fastTileComplex, numBlocks);

    // Convert pointers to GM_ADDR (complex data is stored as interleaved float pairs).
    // When alpha/beta value is 0, A/B may be nullptr; use C as a dummy valid GM address
    // to avoid hardware issues with null descriptors.
    // (Kernel skips reading A/B based on alphaIsZero/betaIsZero tiling flags.)
    GM_ADDR gmC = reinterpret_cast<GM_ADDR>(C);
    GM_ADDR gmA = (A != nullptr) ? reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)) : gmC;
    GM_ADDR gmB = (B != nullptr) ? reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)) : gmC;

    cgeam_kernel_do(gmA, gmB, gmC, tiling, numBlocks, handle->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCgeam(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* beta, const aclblasComplex* B,
    int ldb, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgeam", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    if ((m == 0 || n == 0) && m >= 0 && n >= 0) {
        return ValidateGeamTranspose(transa, transb, "aclblasCgeam");
    }

    aclblasStatus_t status =
        ValidateGeamParams("aclblasCgeam", transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    return LaunchCgeamKernel(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
