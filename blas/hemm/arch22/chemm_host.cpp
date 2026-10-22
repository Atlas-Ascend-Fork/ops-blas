/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chemm (arch22) host entry: parameter/dimension/leading-dim validation,
// Hermitian-aware tiling and workspace planning, and selection of the
// OrderedAiv / SimpleAiv / Mixed launch strategy for aclblasChemm.

#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "common/helper/aclblas_handle_internal.h"
#include "cann_ops_blas.h"
#include "chemm_kernel.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

bool IsValidSide(aclblasSideMode_t side) { return side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT; }

bool IsValidUplo(aclblasFillMode_t uplo) { return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER; }

bool IsSimpleChemm(aclblasComplex alpha, aclblasComplex beta)
{
    return alpha.real == 1.0f && alpha.imag == 0.0f && beta.real == 0.0f && beta.imag == 0.0f;
}

bool IsIdentityChemm(aclblasComplex alpha, aclblasComplex beta)
{
    return alpha.real == 0.0f && alpha.imag == 0.0f && beta.real == 1.0f && beta.imag == 0.0f;
}

uint32_t GetAicBlockCount()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return platform == nullptr ? 1U : std::max<uint32_t>(1U, platform->GetCoreNumAic());
}

uint32_t GetAivBlockCount()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return platform == nullptr ? 1U : std::max<uint32_t>(1U, platform->GetCoreNumAiv());
}

bool UseAivPath(aclblasSideMode_t side, const ChemmAivTilingData& tiling)
{
    const bool tiny = tiling.m <= static_cast<int32_t>(CHEMM_MICRO_TILE) &&
                      tiling.n <= static_cast<int32_t>(CHEMM_MICRO_TILE) &&
                      tiling.dimA <= static_cast<int32_t>(CHEMM_MICRO_TILE);
    if (tiny) {
        return true;
    }
    const uint64_t feeds = (static_cast<uint64_t>(tiling.dimA) + CHEMM_MICRO_TILE - 1U) / CHEMM_MICRO_TILE;
    const uint64_t gridWork = static_cast<uint64_t>(tiling.tileRows) * tiling.tileCols;
    const bool oneWave = tiling.blockDim == gridWork;
    const bool footprintFits =
        side == ACLBLAS_SIDE_RIGHT ? feeds <= CHEMM_AIV_CACHE_FEEDS : feeds <= CHEMM_AIV_LEFT_MAX_FEEDS;
    return oneWave && footprintFits;
}

bool MulFitsUint64(uint64_t lhs, uint64_t rhs, uint64_t* result)
{
    if (rhs != 0U && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool AddFitsUint64(uint64_t lhs, uint64_t rhs, uint64_t* result)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool RoundUpCubeDimension(int value, uint32_t panel, uint32_t* result)
{
    if (panel == 0U) {
        return false;
    }
    const uint64_t logical = static_cast<uint64_t>(value);
    const uint64_t minimum = std::max<uint64_t>(logical, panel);
    if (minimum > std::numeric_limits<uint32_t>::max() - (panel - 1U)) {
        return false;
    }
    *result = static_cast<uint32_t>(((minimum + panel - 1U) / panel) * panel);
    return true;
}

uint32_t SelectPanelRowTile(aclblasSideMode_t side, uint32_t rows, uint32_t dimA)
{
    if (side != ACLBLAS_SIDE_RIGHT) {
        return CHEMM_TILE_M;
    }
    if (dimA > CHEMM_MICRO_TILE && dimA <= CHEMM_NARROW_K) {
        return CHEMM_NARROW_TILE_M;
    }
    const uint32_t tail = rows & (CHEMM_MAX_MIX_TILE_M - 1U);
    return tail == 0U || tail > CHEMM_TILE_M ? CHEMM_MAX_MIX_TILE_M : CHEMM_TILE_M;
}

aclblasStatus_t MakeAivTiling(
    aclblasSideMode_t side, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc, aclblasComplex alpha,
    aclblasComplex beta, ChemmAivTilingData* tiling)
{
    tiling->m = static_cast<int32_t>(m);
    tiling->n = static_cast<int32_t>(n);
    tiling->dimA = static_cast<int32_t>(side == ACLBLAS_SIDE_LEFT ? m : n);
    tiling->lda = static_cast<int32_t>(lda);
    tiling->ldb = static_cast<int32_t>(ldb);
    tiling->ldc = static_cast<int32_t>(ldc);
    tiling->alphaReal = alpha.real;
    tiling->alphaImag = alpha.imag;
    tiling->betaReal = beta.real;
    tiling->betaImag = beta.imag;

    const uint64_t cols = (static_cast<uint64_t>(n) + CHEMM_MICRO_TILE - 1U) / CHEMM_MICRO_TILE;
    const uint64_t feeds = (static_cast<uint64_t>(tiling->dimA) + CHEMM_MICRO_TILE - 1U) / CHEMM_MICRO_TILE;
    const uint64_t rows16 = (static_cast<uint64_t>(m) + CHEMM_MICRO_TILE - 1U) / CHEMM_MICRO_TILE;
    const uint32_t cores = GetAivBlockCount();
    const uint64_t rows8 = (static_cast<uint64_t>(m) + CHEMM_MICRO_TILE / 2U - 1U) / (CHEMM_MICRO_TILE / 2U);
    const uint32_t rowTile = m > static_cast<int64_t>(CHEMM_MICRO_TILE) && rows8 * cols <= cores && feeds >= 2U ?
                                 CHEMM_MICRO_TILE / 2U :
                                 CHEMM_MICRO_TILE;
    const uint64_t rows = (static_cast<uint64_t>(m) + rowTile - 1U) / rowTile;
    const bool tiny = m <= CHEMM_MICRO_TILE && n <= CHEMM_MICRO_TILE && tiling->dimA <= CHEMM_MICRO_TILE;
    const uint64_t work = tiny ? (static_cast<uint64_t>(m) + 7U) / 8U : rows * cols;
    if (rows > std::numeric_limits<uint32_t>::max() || cols > std::numeric_limits<uint32_t>::max() || work == 0U ||
        work > std::numeric_limits<uint32_t>::max()) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling->tileRows = static_cast<uint32_t>(rows);
    tiling->tileCols = static_cast<uint32_t>(cols);
    tiling->blockDim = std::min<uint32_t>(cores, static_cast<uint32_t>(work));
    tiling->tileBase = 0U;
    tiling->rowTile = rowTile;
    return ACLBLAS_STATUS_SUCCESS;
}

struct MixedRoute {
    bool panel;
    bool direct;
    bool directOnePanel;
    bool reuse;
    uint32_t cores;
    uint32_t work;
};

void InitializeMixedTiling(
    aclblasSideMode_t side, int m, int n, int lda, int ldb, int ldc, aclblasComplex alpha, aclblasComplex beta,
    ChemmMixedTilingData* tiling)
{
    tiling->m = m;
    tiling->n = n;
    tiling->dimA = side == ACLBLAS_SIDE_LEFT ? m : n;
    tiling->lda = lda;
    tiling->ldb = ldb;
    tiling->ldc = ldc;
    tiling->alphaReal = alpha.real;
    tiling->alphaImag = alpha.imag;
    tiling->betaReal = beta.real;
    tiling->betaImag = beta.imag;
}

aclblasStatus_t ConfigureMixedGrid(bool simple, uint32_t workM, ChemmMixedTilingData* tiling, uint32_t* work)
{
    if (workM == 0U) {
        OP_LOGE("aclblasChemm", "MIX row tile must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t workN = simple ? CHEMM_WORK_N : CHEMM_TILE_N;
    uint32_t paddedM = 0U;
    uint32_t paddedN = 0U;
    uint32_t paddedK = 0U;
    if (!RoundUpCubeDimension(tiling->m, workM, &paddedM) || !RoundUpCubeDimension(tiling->n, workN, &paddedN) ||
        !RoundUpCubeDimension(tiling->dimA, CHEMM_K_FEED, &paddedK)) {
        OP_LOGE(
            "aclblasChemm", "MIX dimension alignment overflow, m=%d, n=%d, dimA=%d", tiling->m, tiling->n,
            tiling->dimA);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling->tileRowBase = 0U;
    tiling->tileColBase = 0U;
    tiling->tileRows = paddedM / workM;
    tiling->tileCols = paddedN / workN;
    tiling->kFeeds = paddedK / CHEMM_K_FEED;
    const uint64_t tileCount = static_cast<uint64_t>(tiling->tileRows) * tiling->tileCols;
    if (tileCount == 0U || tileCount > std::numeric_limits<uint32_t>::max() || tiling->kFeeds == 0U) {
        OP_LOGE(
            "aclblasChemm", "invalid MIX grid, tiles=%llu, kFeeds=%u", static_cast<unsigned long long>(tileCount),
            tiling->kFeeds);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *work = static_cast<uint32_t>(tileCount);
    return ACLBLAS_STATUS_SUCCESS;
}

bool CanUseDirectMixed(bool simple, int lda, int ldb, int ldc, const ChemmMixedTilingData& tiling)
{
    const uint32_t ldcBlocks = static_cast<uint32_t>(ldc) / 4U;
    return simple && static_cast<uint32_t>(tiling.dimA) % CHEMM_K_FEED == 0U &&
           (static_cast<uint32_t>(lda) & 3U) == 0U && (static_cast<uint32_t>(ldb) & 3U) == 0U &&
           (static_cast<uint32_t>(ldc) & 3U) == 0U && ldcBlocks >= CHEMM_TILE_M / 4U &&
           ldcBlocks - CHEMM_TILE_M / 4U <= 65535U;
}

MixedRoute SelectMixedRoute(
    aclblasSideMode_t side, aclblasFillMode_t uplo, int lda, int ldb, int ldc, bool simple, bool panel, uint32_t cores,
    uint32_t work, const ChemmMixedTilingData& tiling)
{
    const bool direct = CanUseDirectMixed(simple, lda, ldb, ldc, tiling);
    const bool directOnePanel = direct && !panel && static_cast<uint32_t>(tiling.n) <= CHEMM_TILE_N;
    return {panel, direct, directOnePanel, panel || tiling.tileRows > 1U, cores, work};
}

uint32_t SelectMixedBlockDim(const MixedRoute& route)
{
    constexpr uint32_t panelCores = 16U;
    const uint32_t panelWork = std::min<uint32_t>({route.cores, route.work, panelCores});
    return route.panel ? panelWork : std::min<uint32_t>(route.cores, route.work);
}

uint32_t SelectMixedCacheSlots(const MixedRoute& route, const ChemmMixedTilingData& tiling, uint64_t regionSlots)
{
    return route.panel ? tiling.kFeeds :
           route.reuse ? static_cast<uint32_t>(route.directOnePanel ? 2ULL * tiling.kFeeds : regionSlots) :
                         0U;
}

aclblasStatus_t ConfigureMixedSchedule(const MixedRoute& route, int ldb, ChemmMixedTilingData* tiling)
{
    tiling->blockDim = SelectMixedBlockDim(route);
    tiling->workspaceCores = tiling->blockDim;
    tiling->cachePanels = route.panel ? 1U : 2U;
    tiling->cacheFeeds = tiling->kFeeds;
    const uint64_t regionSlots = 4ULL * tiling->kFeeds;
    if (route.reuse && regionSlots > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE("aclblasChemm", "MIX cache slot count overflow, kFeeds=%u", tiling->kFeeds);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling->cacheSlots = SelectMixedCacheSlots(route, *tiling, regionSlots);
    tiling->narrowMode = CHEMM_NARROW_STREAM;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t AddMixedInputWorkspace(const MixedRoute& route, ChemmMixedTilingData* tiling, uint64_t* workspace)
{
    tiling->inputSlotStride = route.panel ?
                                  2ULL * tiling->rowTile * CHEMM_EXP_K :
                                  static_cast<uint64_t>(route.reuse ? 1U : 3U) * CHEMM_INPUT_MATRIX_ELEMENTS;
    tiling->inputCoreStride = static_cast<uint64_t>(CHEMM_INPUT_RING) * tiling->inputSlotStride;
    uint64_t inputFloats = 0U;
    if (!MulFitsUint64(tiling->workspaceCores, tiling->inputCoreStride, &inputFloats) ||
        !AddFitsUint64(*workspace, inputFloats, workspace)) {
        OP_LOGE("aclblasChemm", "MIX input workspace size overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!route.reuse) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    uint64_t cacheCoreFloats = 0U;
    uint64_t cacheFloats = 0U;
    if (!MulFitsUint64(tiling->cacheSlots, CHEMM_INPUT_MATRIX_ELEMENTS, &cacheCoreFloats) ||
        !MulFitsUint64(tiling->workspaceCores, cacheCoreFloats, &cacheFloats) ||
        !AddFitsUint64(*workspace, cacheFloats, workspace)) {
        OP_LOGE("aclblasChemm", "MIX cache workspace size overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t AddMixedProductWorkspace(const MixedRoute& route, ChemmMixedTilingData* tiling, uint64_t* workspace)
{
    tiling->productOffset = *workspace;
    const uint64_t regularProduct = std::max<uint32_t>(CHEMM_PRODUCT_ELEMENTS, CHEMM_PLANAR_PRODUCT_ELEMENTS);
    tiling->productSlotStride = regularProduct;
    const uint64_t productRing = static_cast<uint64_t>(CHEMM_PRODUCT_RING) * tiling->productSlotStride;
    tiling->productCoreStride = route.direct ? 0U : route.panel ? 0U : productRing;
    uint64_t productFloats = 0U;
    if (!MulFitsUint64(tiling->workspaceCores, tiling->productCoreStride, &productFloats) ||
        !AddFitsUint64(*workspace, productFloats, workspace)) {
        OP_LOGE("aclblasChemm", "MIX product workspace size overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t MakeMixedTiling(
    aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, int lda, int ldb, int ldc, aclblasComplex alpha,
    aclblasComplex beta, ChemmMixedTilingData* tiling)
{
    InitializeMixedTiling(side, m, n, lda, ldb, ldc, alpha, beta, tiling);
    const bool simple = IsSimpleChemm(alpha, beta);
    const uint32_t cores = GetAicBlockCount();
    const uint32_t feeds = (static_cast<uint32_t>(tiling->dimA) + CHEMM_K_FEED - 1U) / CHEMM_K_FEED;
    const uint32_t panelM = n <= 16 ? 16U : 32U;
    const bool panel =
        simple && m > 32 && n <= 32 && static_cast<uint64_t>(feeds) * panelM * CHEMM_EXP_K <= CHEMM_PANEL_L1_ELEMENTS;
    tiling->rowTile =
        panel ? SelectPanelRowTile(side, static_cast<uint32_t>(m), static_cast<uint32_t>(tiling->dimA)) : 0U;
    const uint32_t workM = panel ? tiling->rowTile : CHEMM_TILE_M;
    uint32_t work = 0U;
    aclblasStatus_t status = ConfigureMixedGrid(simple, workM, tiling, &work);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const MixedRoute route = SelectMixedRoute(side, uplo, lda, ldb, ldc, simple, panel, cores, work, *tiling);
    status = ConfigureMixedSchedule(route, ldb, tiling);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    uint64_t workspaceFloats = 0U;
    status = AddMixedInputWorkspace(route, tiling, &workspaceFloats);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return AddMixedProductWorkspace(route, tiling, &workspaceFloats);
}

aclblasStatus_t AcquireWorkspace(_aclblas_handle* handle, size_t bytes, uint8_t** workspace)
{
    aclblasStatus_t status = EnsureDefaultWorkspace(handle, bytes);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        *workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    }
    return status;
}

aclblasStatus_t ValidateChemmDimensions(
    aclblasSideMode_t side, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc)
{
    if (m < 0 || n < 0) {
        OP_LOGE(
            "aclblasChemm", "m and n must be nonnegative, m=%lld, n=%lld", static_cast<long long>(m),
            static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    constexpr int64_t maxKernelDimension = std::numeric_limits<int32_t>::max();
    if (m > maxKernelDimension || n > maxKernelDimension || lda > maxKernelDimension || ldb > maxKernelDimension ||
        ldc > maxKernelDimension) {
        OP_LOGE(
            "aclblasChemm", "dimensions exceed INT32_MAX, m=%lld, n=%lld, lda=%lld, ldb=%lld, ldc=%lld",
            static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(lda),
            static_cast<long long>(ldb), static_cast<long long>(ldc));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int64_t dimA = side == ACLBLAS_SIDE_LEFT ? m : n;
    if (lda < std::max<int64_t>(1, dimA) || ldb < std::max<int64_t>(1, m) || ldc < std::max<int64_t>(1, m)) {
        OP_LOGE(
            "aclblasChemm", "invalid leading dimensions, lda=%lld, ldb=%lld, ldc=%lld, dimA=%lld, m=%lld",
            static_cast<long long>(lda), static_cast<long long>(ldb), static_cast<long long>(ldc),
            static_cast<long long>(dimA), static_cast<long long>(m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool HasNullChemmPointer(
    const aclblasComplex* alpha, const aclblasComplex* a, const aclblasComplex* b, const aclblasComplex* beta,
    const aclblasComplex* c)
{
    return alpha == nullptr || a == nullptr || b == nullptr || beta == nullptr || c == nullptr;
}

aclblasStatus_t ValidateChemmArgs(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* a, int64_t lda, const aclblasComplex* b, int64_t ldb,
    const aclblasComplex* beta, const aclblasComplex* c, int64_t ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasChemm", "handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!IsValidSide(side) || !IsValidUplo(uplo)) {
        OP_LOGE("aclblasChemm", "invalid side=%d or uplo=%d", static_cast<int>(side), static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    const aclblasStatus_t dimensionStatus = ValidateChemmDimensions(side, m, n, lda, ldb, ldc);
    if (dimensionStatus != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return dimensionStatus;
    }
    if (HasNullChemmPointer(alpha, a, b, beta, c)) {
        OP_LOGE("aclblasChemm", "alpha, A, B, beta and C must not be nullptr for nonempty matrices");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchOrderedAiv(
    aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc,
    aclblasComplex alpha, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex beta, aclblasComplex* c,
    _aclblas_handle* handle)
{
    ChemmAivTilingData tiling{};
    aclblasStatus_t status = MakeAivTiling(side, m, n, lda, ldb, ldc, alpha, beta, &tiling);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasChemm", "AIV tiling failed, m=%lld, n=%lld, status=%d", static_cast<long long>(m),
            static_cast<long long>(n), static_cast<int>(status));
        return status;
    }
    tiling.rowTile = CHEMM_MICRO_TILE;
    tiling.tileRows = (static_cast<uint32_t>(m) + CHEMM_MICRO_TILE - 1U) / CHEMM_MICRO_TILE;
    tiling.tileCols = (static_cast<uint32_t>(n) + CHEMM_MICRO_TILE - 1U) / CHEMM_MICRO_TILE;
    const uint64_t work = static_cast<uint64_t>(tiling.tileRows) * tiling.tileCols;
    const uint32_t cores = GetAivBlockCount();
    for (uint64_t base = 0U; base < work; base += cores) {
        tiling.tileBase = static_cast<uint32_t>(base);
        tiling.blockDim = static_cast<uint32_t>(std::min<uint64_t>(cores, work - base));
        chemm_ordered_aiv_kernel_do(
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b)), reinterpret_cast<uint8_t*>(c), tiling,
            side == ACLBLAS_SIDE_RIGHT, uplo == ACLBLAS_LOWER, handle->stream);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool TryLaunchSimpleAiv(
    aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc,
    aclblasComplex alpha, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex beta, aclblasComplex* c,
    _aclblas_handle* handle)
{
    ChemmAivTilingData tiling{};
    if (MakeAivTiling(side, m, n, lda, ldb, ldc, alpha, beta, &tiling) != ACLBLAS_STATUS_SUCCESS ||
        !UseAivPath(side, tiling)) {
        return false;
    }
    chemm_aiv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b)), reinterpret_cast<uint8_t*>(c), tiling,
        side == ACLBLAS_SIDE_RIGHT, uplo == ACLBLAS_LOWER, handle->stream);
    return true;
}

aclblasStatus_t LaunchMixedChemm(
    aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc,
    aclblasComplex alpha, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex beta, aclblasComplex* c,
    _aclblas_handle* handle)
{
    ChemmMixedTilingData tiling{};
    aclblasStatus_t status = MakeMixedTiling(
        side, uplo, static_cast<int>(m), static_cast<int>(n), static_cast<int>(lda), static_cast<int>(ldb),
        static_cast<int>(ldc), alpha, beta, &tiling);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    uint64_t workspaceFloats = tiling.productOffset;
    uint64_t productFloats = 0U;
    if (!MulFitsUint64(tiling.workspaceCores, tiling.productCoreStride, &productFloats) ||
        !AddFitsUint64(workspaceFloats, productFloats, &workspaceFloats) ||
        workspaceFloats > std::numeric_limits<size_t>::max() / sizeof(float)) {
        OP_LOGE("aclblasChemm", "MIX workspace byte count overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    uint8_t* workspace = nullptr;
    status = AcquireWorkspace(handle, static_cast<size_t>(workspaceFloats) * sizeof(float), &workspace);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasChemm", "MIX workspace acquisition failed, required=%zu bytes, status=%d",
            static_cast<size_t>(workspaceFloats) * sizeof(float), static_cast<int>(status));
        return status;
    }
    chemm_mixed_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b)), reinterpret_cast<uint8_t*>(c), workspace, tiling,
        side == ACLBLAS_SIDE_RIGHT, uplo == ACLBLAS_LOWER, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasChemm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* a, int64_t lda, const aclblasComplex* b, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* c, int64_t ldc)
{
    static_assert(sizeof(aclblasComplex) == sizeof(float) * 2, "complex64 ABI must be two FP32 values");

    aclblasStatus_t status = ValidateChemmArgs(handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return status;
    }

    const aclblasComplex alphaHost = *alpha;
    const aclblasComplex betaHost = *beta;
    if (IsIdentityChemm(alphaHost, betaHost)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* internalHandle = static_cast<_aclblas_handle*>(handle);
    const int64_t dimA = side == ACLBLAS_SIDE_LEFT ? m : n;
    const bool simple = IsSimpleChemm(alphaHost, betaHost);
    if (!simple && dimA > CHEMM_MICRO_TILE) {
        return LaunchOrderedAiv(side, uplo, m, n, lda, ldb, ldc, alphaHost, a, b, betaHost, c, internalHandle);
    }
    if (simple && TryLaunchSimpleAiv(side, uplo, m, n, lda, ldb, ldc, alphaHost, a, b, betaHost, c, internalHandle)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return LaunchMixedChemm(side, uplo, m, n, lda, ldb, ldc, alphaHost, a, b, betaHost, c, internalHandle);
}
