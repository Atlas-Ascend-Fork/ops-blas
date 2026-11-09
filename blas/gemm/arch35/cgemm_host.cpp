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
#include <complex>
#include <limits>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "gemm_kernel.h"
#include "gemm_host_common.h"
#include "cgemm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// 950 Cube tiles are multiples of 16. Two FP32 accumulator tiles share L0C to overlap compute and output.
constexpr uint32_t CGEMM_MIN_TILE = GEMM_FRACTAL;
constexpr uint32_t CGEMM_MAX_TILE = CGEMM_MAX_SQUARE_TILE;
constexpr uint32_t CGEMM3M_PARTITION_K = 256;
constexpr uint32_t CGEMM_TILE_SIZE_RATIO = 2; // Adjacent supported sizes double each dimension.
static_assert(CGEMM_OUTPUT_STAGES * CGEMM_MAX_TILE * CGEMM_MAX_TILE * sizeof(float) <= CGEMM_L0C_BYTES);
constexpr size_t CGEMM_WORKSPACE_ALIGNMENT = CGEMM_CUBE_FRACTAL * CGEMM_DATA_BLOCK_BYTES;
static aclblasStatus_t HandleCgemmAlphaZero(aclblasHandle_t, int, int, int, std::complex<float>, aclblasComplex*);
static bool CheckedMulSize(size_t, size_t, size_t&);
static bool AddWorkspaceRegion(size_t, size_t&, size_t&);

struct CgemmLaunchContext {
    aclblasHandle_t handle;
    _aclblas_handle* internal;
    aclrtStream stream;
    aclblasOperation_t transa;
    aclblasOperation_t transb;
    int m;
    int n;
    int k;
    std::complex<float> alpha;
    const aclblasComplex* a;
    int lda;
    const aclblasComplex* b;
    int ldb;
    std::complex<float> beta;
    aclblasComplex* c;
    int ldc;
    uint32_t cubeCores;
    uint32_t vectorCores;
    bool forceThreeProducts;
};

struct CgemmWorkspaceLayout {
    int aRows;
    int aCols;
    int bRows;
    int bCols;
    size_t tempLdc;
    size_t aElements;
    size_t bElements;
    size_t aBytes;
    size_t bBytes;
    size_t tempBytes;
    uint32_t fusedTile;
    uint32_t productTile;
    bool threeProducts;
    uint32_t kPartitions;
    uint32_t kSpan;
    size_t partElements;
    uint8_t* realA;
    uint8_t* imagA;
    uint8_t* realB;
    uint8_t* imagB;
    uint8_t* auxiliaryA;
    uint8_t* auxiliaryB;
    uint8_t* temp1;
    uint8_t* temp2;
    uint8_t* temp3;
    uint8_t* temp4;
};

static aclblasStatus_t InitCgemmCores(CgemmLaunchContext& context)
{
    context.cubeCores = GetAicCoreCount();
    if (context.cubeCores == 0) {
        OP_LOGE("aclblasCgemm", "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    context.vectorCores = GetAivCoreCount();
    if (context.vectorCores == 0) {
        OP_LOGE("aclblasCgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ReserveCgemmWorkspace(CgemmLaunchContext& context, size_t bytes, const char* path)
{
    const aclblasStatus_t status = EnsureDefaultWorkspace(context.internal, bytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCgemm", "%s requires %zu workspace bytes, ret=%d", path, bytes, status);
    }
    return status;
}

static uint32_t SelectCgemmAccumulationTile(const CgemmLaunchContext& context)
{
    uint32_t tile = CGEMM_MAX_TILE;
    while (tile > CGEMM_MIN_TILE) {
        const uint64_t tasks = static_cast<uint64_t>(CGEMM_FOUR_PRODUCTS) * CeilDiv(context.m, static_cast<int>(tile)) *
                               CeilDiv(context.n, static_cast<int>(tile));
        if (tasks >= context.cubeCores) {
            break;
        }
        tile /= CGEMM_TILE_SIZE_RATIO;
    }
    return tile;
}

// Bound the work assigned to the busiest core, including an incomplete final wave.
static uint64_t EstimateCgemmCoreWork(
    const CgemmLaunchContext& context, uint32_t tile, uint32_t products, uint32_t partitions)
{
    const uint64_t rowTiles = CeilDiv<uint64_t>(context.m, tile);
    const uint64_t colTiles = CeilDiv<uint64_t>(context.n, tile);
    const uint64_t waves = CeilDiv<uint64_t>(rowTiles * colTiles * products * partitions, context.cubeCores);
    const uint64_t rows = CeilAlign(std::min<uint32_t>(context.m, tile), CGEMM_CUBE_FRACTAL);
    const uint64_t cols = CeilAlign(std::min<uint32_t>(context.n, tile), CGEMM_CUBE_FRACTAL);
    // Count aligned matrix blocks to keep the estimate in range for large inputs.
    return waves * (rows / CGEMM_CUBE_FRACTAL) * (cols / CGEMM_CUBE_FRACTAL);
}

static uint32_t SelectCgemmProductTile(
    const CgemmLaunchContext& context, uint32_t defaultTile, uint32_t products, uint32_t partitions)
{
    // An occupancy-driven reduction already trades input reuse for more tasks.
    // Do not apply a second reduction using only the output-work estimate.
    if (defaultTile < CGEMM_MAX_TILE) {
        return defaultTile;
    }
    // Compare adjacent supported sizes; keep the larger tile when work is equal
    // because it reuses each input across more output elements.
    const uint32_t smallerTile = defaultTile / CGEMM_TILE_SIZE_RATIO;
    return EstimateCgemmCoreWork(context, smallerTile, products, partitions) <
                   EstimateCgemmCoreWork(context, defaultTile, products, partitions) ?
               smallerTile :
               defaultTile;
}

static uint32_t SelectCgemmFusedTile(const CgemmLaunchContext& context)
{
    const bool unitOutput =
        context.alpha == std::complex<float>(1.0f, 0.0f) && context.beta == std::complex<float>(0.0f, 0.0f);
    if (!unitOutput) {
        return 0;
    }
    const uint32_t singleLoadDepth = CGEMM_HALF_L0_FLOAT_ELEMENTS / CGEMM_FUSED_TILE;
    if (context.k <= static_cast<int>(singleLoadDepth)) {
        return CGEMM_FUSED_TILE;
    }
    const uint64_t largeTiles = static_cast<uint64_t>(CeilDiv(context.m, static_cast<int>(CGEMM_FUSED_MAX_TILE))) *
                                CeilDiv(context.n, static_cast<int>(CGEMM_FUSED_MAX_TILE));
    // Otherwise expose the three products as independent tasks to fill the device.
    // Fusion starts once output tiles alone cover all worker/product pairs.
    const uint64_t workerProducts = static_cast<uint64_t>(context.cubeCores) * CGEMM_THREE_PRODUCTS;
    return largeTiles >= workerProducts ? CGEMM_FUSED_MAX_TILE : 0;
}

static bool CalculateCgemmWorkspaceSizes(const CgemmLaunchContext& context, CgemmWorkspaceLayout& layout)
{
    layout.fusedTile = SelectCgemmFusedTile(context);
    layout.threeProducts = context.forceThreeProducts || context.alpha == std::complex<float>(1.0f, 0.0f) &&
                                                             context.beta == std::complex<float>(0.0f, 0.0f);
    layout.aRows = context.transa == ACLBLAS_OP_N ? context.m : context.k;
    layout.aCols = context.transa == ACLBLAS_OP_N ? context.k : context.m;
    layout.bRows = context.transb == ACLBLAS_OP_N ? context.k : context.n;
    layout.bCols = context.transb == ACLBLAS_OP_N ? context.n : context.k;
    layout.tempLdc = static_cast<size_t>(CeilAlign(context.m, GEMM_FRACTAL));
    // A balanced two-way merge shortens accumulations once K exceeds one L1 panel.
    // This rule uses input extent and buffer capacity, not a list of matrix shapes.
    const uint32_t accumulationTile = SelectCgemmAccumulationTile(context);
    const uint32_t panelDepth = CGEMM_L1_OPERAND_STAGE_ELEMENTS / accumulationTile;
    layout.kPartitions = context.k > static_cast<int>(panelDepth) ? 2 : 1;
    if (context.forceThreeProducts && context.alpha == std::complex<float>(1.0f, 0.0f) &&
        context.beta == std::complex<float>(0.0f, 0.0f)) {
        layout.kPartitions = CeilDiv<uint32_t>(context.k, CGEMM3M_PARTITION_K);
    }
    const uint32_t products = layout.threeProducts ? CGEMM_THREE_PRODUCTS : CGEMM_FOUR_PRODUCTS;
    layout.productTile = SelectCgemmProductTile(context, accumulationTile, products, layout.kPartitions);
    layout.kSpan = CeilDiv(CeilDiv(context.k, static_cast<int>(GEMM_FRACTAL)), static_cast<int>(layout.kPartitions)) *
                   GEMM_FRACTAL;
    size_t tempElements = 0;
    return CheckedMulSize(static_cast<size_t>(layout.aRows), static_cast<size_t>(layout.aCols), layout.aElements) &&
           CheckedMulSize(static_cast<size_t>(layout.bRows), static_cast<size_t>(layout.bCols), layout.bElements) &&
           CheckedMulSize(
               layout.tempLdc, static_cast<size_t>(CeilAlign(context.n, GEMM_FRACTAL)), layout.partElements) &&
           CheckedMulSize(layout.partElements, layout.kPartitions, tempElements) &&
           CheckedMulSize(layout.aElements, sizeof(float), layout.aBytes) &&
           CheckedMulSize(layout.bElements, sizeof(float), layout.bBytes) &&
           CheckedMulSize(layout.fusedTile != 0 ? 0 : tempElements, sizeof(float), layout.tempBytes);
}

static aclblasStatus_t AllocateCgemmWorkspace(CgemmLaunchContext& context, CgemmWorkspaceLayout& layout)
{
    enum Region {
        REAL_A,
        IMAG_A,
        REAL_B,
        IMAG_B,
        AUXILIARY_A,
        AUXILIARY_B,
        PRODUCT_1,
        PRODUCT_2,
        PRODUCT_3,
        PRODUCT_4,
        REGION_COUNT
    };
    size_t total = 0;
    size_t offsets[REGION_COUNT]{};
    const size_t sizes[REGION_COUNT] = {
        layout.aBytes,
        layout.aBytes,
        layout.bBytes,
        layout.bBytes,
        layout.threeProducts ? layout.aBytes : 0,
        layout.threeProducts ? layout.bBytes : 0,
        layout.tempBytes,
        layout.tempBytes,
        layout.tempBytes,
        layout.tempBytes};
    for (size_t index = 0; index < REGION_COUNT; ++index) {
        if (!AddWorkspaceRegion(sizes[index], total, offsets[index])) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    const aclblasStatus_t status = ReserveCgemmWorkspace(context, total, "operation");
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    uint8_t* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(context.internal));
    layout.realA = workspace + offsets[REAL_A];
    layout.imagA = workspace + offsets[IMAG_A];
    layout.realB = workspace + offsets[REAL_B];
    layout.imagB = workspace + offsets[IMAG_B];
    layout.auxiliaryA = workspace + offsets[AUXILIARY_A];
    layout.auxiliaryB = workspace + offsets[AUXILIARY_B];
    layout.temp1 = workspace + offsets[PRODUCT_1];
    layout.temp2 = workspace + offsets[PRODUCT_2];
    layout.temp3 = workspace + offsets[PRODUCT_3];
    layout.temp4 = workspace + offsets[PRODUCT_4];
    return ACLBLAS_STATUS_SUCCESS;
}

static void LaunchCgemmDeinterleave(const CgemmLaunchContext& context, const CgemmWorkspaceLayout& layout)
{
    const uint32_t blocks = std::min<uint64_t>(
        context.vectorCores, CeilDiv<uint64_t>(layout.aElements + layout.bElements, CGEMM_SIMT_THREADS));
    CgemmDeinterleaveTilingData tiling{};
    tiling.aRows = static_cast<uint32_t>(layout.aRows);
    tiling.aCols = static_cast<uint32_t>(layout.aCols);
    tiling.lda = static_cast<uint32_t>(context.lda);
    tiling.aOutRows = static_cast<uint32_t>(layout.aRows);
    tiling.bRows = static_cast<uint32_t>(layout.bRows);
    tiling.bCols = static_cast<uint32_t>(layout.bCols);
    tiling.ldb = static_cast<uint32_t>(context.ldb);
    tiling.bOutRows = static_cast<uint32_t>(layout.bRows);
    tiling.aRowsPerCore = CeilDiv<uint32_t>(tiling.aRows, blocks);
    tiling.bRowsPerCore = CeilDiv<uint32_t>(tiling.bRows, blocks);
    tiling.threeProducts = layout.threeProducts;
    tiling.conjugateA = context.transa == ACLBLAS_OP_C ? 1 : 0;
    tiling.conjugateB = context.transb == ACLBLAS_OP_C ? 1 : 0;
    gemm_cgemm_deinterleave_do(
        blocks, context.stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(context.a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(context.b)), layout.realA, layout.imagA, layout.realB,
        layout.imagB, layout.auxiliaryA, layout.auxiliaryB, tiling);
}

static GemmTilingData PrepareCgemmTiling(const CgemmLaunchContext& context, const CgemmWorkspaceLayout& layout)
{
    GemmTilingData tiling = BuildGemmTilingData(
        context.m, context.n, context.k, layout.aRows, layout.bRows, static_cast<int>(layout.tempLdc),
        context.transa == ACLBLAS_OP_N ? ACLBLAS_OP_N : ACLBLAS_OP_T,
        context.transb == ACLBLAS_OP_N ? ACLBLAS_OP_N : ACLBLAS_OP_T, context.alpha.real(), context.alpha.imag(),
        context.beta.real(), context.beta.imag());
    ApplyGemmColMajorSwap(tiling);
    const uint32_t tile = layout.productTile;
    const uint32_t products = layout.threeProducts ? CGEMM_THREE_PRODUCTS : CGEMM_FOUR_PRODUCTS;
    const uint64_t tasks = static_cast<uint64_t>(products) * layout.kPartitions *
                           CeilDiv(tiling.m, static_cast<int>(tile)) * CeilDiv(tiling.n, static_cast<int>(tile));
    tiling.cgemmThreeProducts = layout.threeProducts;
    tiling.cgemmKPartitions = layout.kPartitions;
    tiling.cgemmKSpan = layout.kSpan;
    tiling.baseM = tile;
    tiling.baseN = tile;
    tiling.ldc = layout.tempLdc;
    tiling.usedCoreNum = static_cast<int32_t>(std::min<uint64_t>(tasks, context.cubeCores));
    return tiling;
}

static void LaunchCgemmEpilogue(const CgemmLaunchContext& context, const CgemmWorkspaceLayout& layout)
{
    const uint64_t outputElements = static_cast<uint64_t>(context.m) * context.n;
    const uint32_t blocks =
        std::min<uint64_t>(context.vectorCores, CeilDiv<uint64_t>(outputElements, CGEMM_SIMT_THREADS));
    CgemmEpilogueTilingData tiling{};
    tiling.m = static_cast<uint32_t>(context.m);
    tiling.n = static_cast<uint32_t>(context.n);
    tiling.ldc = static_cast<uint32_t>(context.ldc);
    tiling.tempLdc = static_cast<uint32_t>(layout.tempLdc);
    tiling.kPartitions = layout.kPartitions;
    tiling.partStride = layout.partElements;
    tiling.threeProducts = layout.threeProducts;
    tiling.rowsPerCore = CeilDiv<uint32_t>(tiling.m, blocks);
    tiling.alphaReal = context.alpha.real();
    tiling.alphaImag = context.alpha.imag();
    tiling.betaReal = context.beta.real();
    tiling.betaImag = context.beta.imag();
    tiling.betaZero = context.beta == std::complex<float>(0.0f, 0.0f) ? 1 : 0;
    if (context.forceThreeProducts && (context.alpha != std::complex<float>(1.0f, 0.0f) || !tiling.betaZero)) {
        tiling.referenceK = context.k;
        tiling.referenceLda = context.lda;
        tiling.referenceLdb = context.ldb;
        tiling.referenceTransA = static_cast<uint8_t>(context.transa);
        tiling.referenceTransB = static_cast<uint8_t>(context.transb);
    }
    gemm_cgemm_epilogue_do(
        blocks, context.stream, layout.temp1, layout.temp2, layout.temp3, layout.temp4,
        reinterpret_cast<uint8_t*>(context.c), tiling,
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(context.a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(context.b)));
}

static bool UseCgemmOnchipInput(const CgemmLaunchContext& context)
{
    const uint32_t shortDepth = CGEMM_HALF_L0_FLOAT_ELEMENTS / CGEMM_FUSED_TILE;
    return context.alpha == std::complex<float>(1.0f, 0.0f) && context.beta == std::complex<float>(0.0f, 0.0f) &&
           context.k <= static_cast<int>(shortDepth);
}

static void LaunchCgemmOnchipInput(const CgemmLaunchContext& context)
{
    CgemmOnchipTilingData t{
        static_cast<uint32_t>(context.m),
        static_cast<uint32_t>(context.n),
        static_cast<uint32_t>(context.k),
        static_cast<uint32_t>(context.lda),
        static_cast<uint32_t>(context.ldb),
        static_cast<uint32_t>(context.ldc),
        static_cast<uint8_t>(context.transa != ACLBLAS_OP_N),
        static_cast<uint8_t>(context.transb != ACLBLAS_OP_N),
        static_cast<uint8_t>(context.transa == ACLBLAS_OP_C),
        static_cast<uint8_t>(context.transb == ACLBLAS_OP_C)};
    const uint64_t tiles = static_cast<uint64_t>(CeilDiv(context.m, static_cast<int>(CGEMM_FUSED_TILE))) *
                           CeilDiv(context.n, static_cast<int>(CGEMM_FUSED_TILE));
    const uint32_t blocks = std::min<uint64_t>(context.cubeCores, tiles);
    gemm_cgemm_onchip_do(
        blocks, context.stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(context.a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(context.b)), reinterpret_cast<uint8_t*>(context.c), t);
}

static aclblasStatus_t LaunchGeneralCgemm(CgemmLaunchContext& context)
{
    if (!context.forceThreeProducts && UseCgemmOnchipInput(context)) {
        LaunchCgemmOnchipInput(context);
        return ACLBLAS_STATUS_SUCCESS;
    }
    CgemmWorkspaceLayout layout{};
    if (!CalculateCgemmWorkspaceSizes(context, layout)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const aclblasStatus_t status = AllocateCgemmWorkspace(context, layout);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    LaunchCgemmDeinterleave(context, layout);
    GemmTilingData tiling = PrepareCgemmTiling(context, layout);
    if (layout.fusedTile != 0) {
        tiling.baseM = layout.fusedTile;
        tiling.baseN = layout.fusedTile;
        const uint64_t tiles = static_cast<uint64_t>(CeilDiv(context.m, static_cast<int>(layout.fusedTile))) *
                               CeilDiv(context.n, static_cast<int>(layout.fusedTile));
        tiling.usedCoreNum = std::min<uint64_t>(tiles, context.cubeCores);
        // The output keeps the caller's column stride, including padding.
        tiling.ldc = context.ldc;
        gemm_cgemm_fused_do(
            tiling.usedCoreNum, context.stream, layout.realA, layout.imagA, layout.realB, layout.imagB,
            layout.auxiliaryA, layout.auxiliaryB, reinterpret_cast<uint8_t*>(context.c), tiling);
        return ACLBLAS_STATUS_SUCCESS;
    }
    gemm_cgemm_products_do(
        tiling.usedCoreNum, context.stream, layout.realA, layout.imagA, layout.realB, layout.imagB, layout.auxiliaryA,
        layout.auxiliaryB, layout.temp1, layout.temp2, layout.temp3, layout.temp4, tiling);
    LaunchCgemmEpilogue(context, layout);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCgemmKernel(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    std::complex<float> alphaVal, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    std::complex<float> betaVal, aclblasComplex* C, int ldc, bool forceThreeProducts = false)
{
    auto* internal = reinterpret_cast<_aclblas_handle*>(handle);
    CgemmLaunchContext context{
        handle, internal, internal->stream,  transa, transb, m, n, k, alphaVal, A, lda, B, ldb, betaVal, C, ldc,
        0,      0,        forceThreeProducts};
    const aclblasStatus_t coreStatus = InitCgemmCores(context);
    if (coreStatus != ACLBLAS_STATUS_SUCCESS) {
        return coreStatus;
    }
    if (k == 0 || alphaVal == std::complex<float>(0.0f, 0.0f)) {
        return HandleCgemmAlphaZero(handle, m, n, ldc, betaVal, C);
    }
    return LaunchGeneralCgemm(context);
}

static aclblasStatus_t ValidateCgemmParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, int lda, int ldb, const void* beta, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgemm", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) {
        OP_LOGE("aclblasCgemm", "invalid transa: %d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C) {
        OP_LOGE("aclblasCgemm", "invalid transb: %d", static_cast<int>(transb));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0) {
        OP_LOGE("aclblasCgemm", "m, n, k must be non-negative: m=%d, n=%d, k=%d", m, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int ldaMin = isTransA ? std::max(1, k) : std::max(1, m);
    if (lda < ldaMin) {
        OP_LOGE("aclblasCgemm", "lda=%d must be >= %d", lda, ldaMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int ldbMin = isTransB ? std::max(1, n) : std::max(1, k);
    if (ldb < ldbMin) {
        OP_LOGE("aclblasCgemm", "ldb=%d must be >= %d", ldb, ldbMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("aclblasCgemm", "ldc=%d must be >= %d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasCgemm", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclblasCgemm", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t HandleCgemmAlphaZero(
    aclblasHandle_t handle, int m, int n, int ldc, std::complex<float> betaVal, aclblasComplex* C)
{
    if (betaVal == std::complex<float>(1.0f, 0.0f)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint64_t outputElements = static_cast<uint64_t>(m) * n;
    uint32_t vecBlocks = std::min<uint64_t>(aivCoreNum, CeilDiv<uint64_t>(outputElements, CGEMM_SIMT_THREADS));
    CgemmEpilogueTilingData tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.tempLdc = static_cast<uint32_t>(CeilAlign(m, GEMM_FRACTAL));
    tiling.rowsPerCore = CeilDiv<uint32_t>(static_cast<uint32_t>(m), vecBlocks);
    tiling.betaReal = betaVal.real();
    tiling.betaImag = betaVal.imag();
    tiling.alphaZero = 1;
    tiling.betaZero = betaVal == std::complex<float>(0.0f, 0.0f) ? 1 : 0;
    gemm_cgemm_epilogue_do(
        vecBlocks, h->stream, nullptr, nullptr, nullptr, nullptr, reinterpret_cast<uint8_t*>(C), tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static bool CheckedMulSize(size_t lhs, size_t rhs, size_t& out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

static bool AddWorkspaceRegion(size_t bytes, size_t& total, size_t& offset)
{
    if (total > std::numeric_limits<size_t>::max() - (CGEMM_WORKSPACE_ALIGNMENT - 1)) {
        return false;
    }
    offset = (total + CGEMM_WORKSPACE_ALIGNMENT - 1) / CGEMM_WORKSPACE_ALIGNMENT * CGEMM_WORKSPACE_ALIGNMENT;
    if (bytes > std::numeric_limits<size_t>::max() - offset) {
        return false;
    }
    total = offset + bytes;
    return true;
}

static aclblasStatus_t ValidateCgemmPointers(
    int k, const aclblasComplex* A, const aclblasComplex* B, const aclblasComplex* C)
{
    if (C == nullptr) {
        OP_LOGE("aclblasCgemm", "C must not be nullptr for a non-empty output");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k > 0 && (A == nullptr || B == nullptr)) {
        OP_LOGE("aclblasCgemm", "A and B must not be nullptr when k > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

extern "C" aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    OP_LOGI(
        "aclblasCgemm", "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k);

    aclblasStatus_t st = ValidateCgemmParams(handle, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    std::complex<float> alphaVal(alpha->real, alpha->imag);
    std::complex<float> betaVal(beta->real, beta->imag);

    st = ValidateCgemmPointers(k, A, B, C);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    return LaunchCgemmKernel(handle, transa, transb, m, n, k, alphaVal, A, lda, B, ldb, betaVal, C, ldc);
}

aclblasStatus_t LaunchCgemmThreeProducts(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex& alpha, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb,
    const aclblasComplex& beta, aclblasComplex* c, int ldc)
{
    return LaunchCgemmKernel(
        handle, transa, transb, m, n, k, {alpha.real, alpha.imag}, a, lda, b, ldb, {beta.real, beta.imag}, c, ldc,
        true);
}
aclblasStatus_t LaunchCgemm3mOnchipBatches(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const aclblasComplex* a,
    int lda, int64_t strideA, const aclblasComplex* b, int ldb, int64_t strideB, aclblasComplex* c, int ldc,
    int64_t strideC, int batch)
{
    CgemmOnchipTilingData t{
        uint32_t(m),
        uint32_t(n),
        uint32_t(k),
        uint32_t(lda),
        uint32_t(ldb),
        uint32_t(ldc),
        uint8_t(ta != ACLBLAS_OP_N),
        uint8_t(tb != ACLBLAS_OP_N),
        uint8_t(ta == ACLBLAS_OP_C),
        uint8_t(tb == ACLBLAS_OP_C)};
    // Netlib's non-transposed A path accumulates directly into C; shorter
    // parts bound the cancellation error relative to that ordered reduction.
    t.reductionSpan = k > 128 ? (ta == ACLBLAS_OP_N ? 128 : 256) : 0;
    t.batchCount = batch;
    t.strideA = strideA;
    t.strideB = strideB;
    t.strideC = strideC;
    const uint32_t cores = GetAicCoreCount();
    if (cores == 0) {
        OP_LOGE("aclblasCgemm3mStridedBatched", "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint64_t tiles = uint64_t(CeilDiv(m, int(CGEMM_FUSED_TILE))) * CeilDiv(n, int(CGEMM_FUSED_TILE)) * batch;
    gemm_cgemm_onchip_do(
        std::min<uint64_t>(cores, tiles), handle->stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b)), reinterpret_cast<uint8_t*>(c), t);
    return ACLBLAS_STATUS_SUCCESS;
}
#endif
