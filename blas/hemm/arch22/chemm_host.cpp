/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <limits>

#include "common/helper/aclblas_handle_internal.h"
#include "cann_ops_blas.h"
#include "chemm_tiling_data.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

template <bool kAic>
uint32_t CoreCount()
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return platform == nullptr ? 1U :
           kAic                ? std::max<uint32_t>(1U, platform->GetCoreNumAic()) :
                                 std::max<uint32_t>(1U, platform->GetCoreNumAiv());
}

aclblasStatus_t ValidateDimensions(aclblasSideMode_t side, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc)
{
    if (m < 0 || n < 0) {
        OP_LOGE("aclblasChemm", "negative matrix dimension");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    constexpr int64_t maxDimension = std::numeric_limits<int32_t>::max();
    const int64_t k = side == ACLBLAS_SIDE_LEFT ? m : n;
    if (m > maxDimension || n > maxDimension || lda > maxDimension || ldb > maxDimension || ldc > maxDimension ||
        lda < k || ldb < m || ldc < m) {
        OP_LOGE("aclblasChemm", "unsupported dimensions or invalid leading dimension");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateArguments(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* a, int64_t lda, const aclblasComplex* b, int64_t ldb,
    const aclblasComplex* beta, const aclblasComplex* c, int64_t ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasChemm", "handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if ((side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) || (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER)) {
        OP_LOGE("aclblasChemm", "invalid side or uplo");
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    const aclblasStatus_t status = ValidateDimensions(side, m, n, lda, ldb, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return status;
    }
    if (alpha == nullptr || a == nullptr || b == nullptr || beta == nullptr || c == nullptr) {
        OP_LOGE("aclblasChemm", "nonempty matrices require nonnull operands and coefficients");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

ChemmAivTilingData MakeVectorTiling(
    aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc,
    aclblasComplex alpha, aclblasComplex beta, bool simple)
{
    ChemmAivTilingData t{};
    t.m = static_cast<int32_t>(m);
    t.n = static_cast<int32_t>(n);
    t.dimA = static_cast<int32_t>(side == ACLBLAS_SIDE_LEFT ? m : n);
    t.lda = static_cast<int32_t>(lda);
    t.ldb = static_cast<int32_t>(ldb);
    t.ldc = static_cast<int32_t>(ldc);
    t.alphaReal = alpha.real;
    t.alphaImag = alpha.imag;
    t.betaReal = beta.real;
    t.betaImag = beta.imag;
    t.kernelFlags =
        (side == ACLBLAS_SIDE_RIGHT ? CHEMM_FLAG_RIGHT : 0U) | (uplo == ACLBLAS_LOWER ? CHEMM_FLAG_LOWER : 0U);
    const uint64_t cols = (static_cast<uint64_t>(n) + 15U) / 16U;
    const uint64_t rows8 = (static_cast<uint64_t>(m) + 7U) / 8U;
    const uint32_t cores = CoreCount<false>();
    t.rowTile = simple && m > 16 && t.dimA > 16 && rows8 * cols <= cores ? 8U : 16U;
    t.tileRows = static_cast<uint32_t>((static_cast<uint64_t>(m) + t.rowTile - 1U) / t.rowTile);
    t.tileCols = static_cast<uint32_t>(cols);
    t.mode = !simple ? ChemmVectorMode::ORDERED : m <= 16 && n <= 16 ? ChemmVectorMode::LOCAL : ChemmVectorMode::CACHED;
    const uint64_t work = t.mode == ChemmVectorMode::LOCAL ? rows8 : static_cast<uint64_t>(t.tileRows) * t.tileCols;
    t.blockDim = static_cast<uint32_t>(std::min<uint64_t>(cores, work));
    t.reductionBanks = 1U;
    while (t.reductionBanks < static_cast<uint32_t>(t.dimA) && t.reductionBanks < 16U) {
        t.reductionBanks *= 2U;
    }
    if (m == 16 && n == 16 && (lda & 3U) == 0U && (ldb & 3U) == 0U && (ldc & 3U) == 0U) {
        t.kernelFlags |= CHEMM_FLAG_FULL;
    }
    return t;
}

bool UseVector(const ChemmAivTilingData& t)
{
    if (t.mode != ChemmVectorMode::CACHED) {
        return true;
    }
    const uint32_t feeds = (static_cast<uint32_t>(t.dimA) + 15U) / 16U;
    const uint32_t capacity =
        (t.kernelFlags & CHEMM_FLAG_RIGHT) != 0U ? CHEMM_AIV_CACHE_FEEDS : CHEMM_AIV_LEFT_MAX_FEEDS;
    return static_cast<uint64_t>(t.tileRows) * t.tileCols == t.blockDim && feeds <= capacity;
}

aclblasStatus_t MakeMatrixGrid(ChemmMixedTilingData& t)
{
    if (t.panelM == 0U || t.expandedK == 0U || t.rawK >= t.expandedK) {
        OP_LOGE("aclblasChemm", "invalid matrix panel geometry");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    t.tileCols = (static_cast<uint32_t>(t.n) + t.panelM - 1U) / t.panelM;
    if (t.tileCols == 0U) {
        OP_LOGE("aclblasChemm", "empty matrix column grid");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t cores = CoreCount<true>();
    const uint32_t rowGroups = std::max<uint32_t>(1U, cores / t.tileCols);
    const bool stationary = (t.kernelFlags & CHEMM_FLAG_RIGHT) != 0U && t.kFeeds == 1U;
    const uint32_t rows =
        stationary ? static_cast<uint32_t>(t.m) : (static_cast<uint32_t>(t.m) + rowGroups - 1U) / rowGroups;
    const uint32_t balancedRows = (rows + 31U) & ~31U;
    const uint32_t inputK = stationary ? t.expandedK : t.expandedK - t.rawK;
    const uint32_t maxRows =
        std::min<uint32_t>({CHEMM_MAX_ROWS, CHEMM_L0_ELEMENTS / (2U * inputK), CHEMM_L0_ELEMENTS / (2U * t.panelM)}) &
        ~31U;
    t.rowTile = std::min<uint32_t>(maxRows, std::max<uint32_t>(32U, balancedRows));
    if (t.rowTile == 0U) {
        OP_LOGE("aclblasChemm", "matrix row tile exceeds local capacity");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    t.tileRows = (static_cast<uint32_t>(t.m) + t.rowTile - 1U) / t.rowTile;
    const uint64_t tiles = static_cast<uint64_t>(t.tileRows) * t.tileCols;
    if (tiles > std::numeric_limits<uint32_t>::max() || tiles * t.kFeeds > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE("aclblasChemm", "matrix work grid exceeds kernel indexing range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    t.rowOwned = stationary && tiles > 1U;
    const uint64_t work = t.rowOwned != 0U ? t.tileRows : tiles;
    t.workTiles = static_cast<uint32_t>(work);
    t.blockDim = static_cast<uint32_t>(std::min<uint64_t>(cores, work));
    if (t.blockDim == 0U) {
        OP_LOGE("aclblasChemm", "matrix grid has no active core");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    t.tilesPerCore = static_cast<uint32_t>(work / t.blockDim);
    t.extraTiles = static_cast<uint32_t>(work % t.blockDim);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t MakeMatrixTiling(const ChemmAivTilingData& v, ChemmMixedTilingData& t)
{
    t.kernelFlags = v.kernelFlags;
    t.m = v.m;
    t.n = v.n;
    t.dimA = v.dimA;
    t.lda = v.lda;
    t.ldb = v.ldb;
    t.ldc = v.ldc;
    t.expandedK = 2U * std::min<uint32_t>(CHEMM_K_FEED, (static_cast<uint32_t>(t.dimA) + 7U) & ~7U);
    t.kFeeds = (static_cast<uint32_t>(t.dimA) + CHEMM_K_FEED - 1U) / CHEMM_K_FEED;
    t.panelM = std::min<uint32_t>(CHEMM_MAX_PANEL, (static_cast<uint32_t>(t.n) + 15U) & ~15U);
    const uint32_t lastK = static_cast<uint32_t>(t.dimA) - (t.kFeeds - 1U) * CHEMM_K_FEED;
    t.rawK =
        (t.kernelFlags & CHEMM_FLAG_RIGHT) != 0U && t.kFeeds > 1U && lastK >= CHEMM_MICRO_TILE ? CHEMM_MICRO_TILE : 0U;
    const aclblasStatus_t status = MakeMatrixGrid(t);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    t.pElements = t.panelM * t.expandedK;
    t.qElements = 2U * t.rowTile * t.expandedK;
    t.l0BankMask = t.rowOwned == 0U && 2U * t.qElements <= CHEMM_L0_ELEMENTS ? 1U : 0U;
    const uint64_t pCache = static_cast<uint64_t>(t.kFeeds) * t.pElements;
    const uint64_t gmCache = t.rowOwned != 0U ? 2U * CHEMM_MICRO_TILE * t.expandedK : pCache;
    t.residentP = pCache + 2U * t.qElements <= CHEMM_L1_ELEMENTS;
    t.inputSlotStride = 2U * t.rowTile * (t.expandedK - t.rawK);
    t.pCacheOffset = CHEMM_INPUT_RING * t.inputSlotStride;
    if (gmCache > std::numeric_limits<uint32_t>::max() - t.pCacheOffset) {
        OP_LOGE("aclblasChemm", "stationary operand cache exceeds kernel indexing range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    t.inputCoreStride = t.pCacheOffset + static_cast<uint32_t>(gmCache);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchChemm(
    const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c, const ChemmAivTilingData& tiling,
    _aclblas_handle* handle)
{
    auto* gmA = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a));
    auto* gmB = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b));
    auto* gmC = reinterpret_cast<uint8_t*>(c);
    if (UseVector(tiling)) {
        chemm_aiv_kernel_do(gmA, gmB, gmC, tiling, handle->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    ChemmMixedTilingData matrix{};
    aclblasStatus_t status = MakeMatrixTiling(tiling, matrix);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const uint64_t bytes = static_cast<uint64_t>(matrix.blockDim) * matrix.inputCoreStride * sizeof(float);
    if (bytes > std::numeric_limits<size_t>::max()) {
        OP_LOGE("aclblasChemm", "workspace byte count overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    status = EnsureDefaultWorkspace(handle, static_cast<size_t>(bytes));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasChemm", "cannot acquire workspace, status=%d", static_cast<int>(status));
        return status;
    }
    chemm_mixed_kernel_do(gmA, gmB, gmC, static_cast<uint8_t*>(GetEffectiveWorkspace(handle)), matrix, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasChemm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* a, int64_t lda, const aclblasComplex* b, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* c, int64_t ldc)
{
    static_assert(sizeof(aclblasComplex) == 2U * sizeof(float), "complex64 ABI requires two FP32 components");
    const aclblasStatus_t status = ValidateArguments(handle, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return status;
    }
    if (alpha->real == 0.0f && alpha->imag == 0.0f && beta->real == 1.0f && beta->imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const bool simple = alpha->real == 1.0f && alpha->imag == 0.0f && beta->real == 0.0f && beta->imag == 0.0f;
    return LaunchChemm(
        a, b, c, MakeVectorTiling(side, uplo, m, n, lda, ldb, ldc, *alpha, *beta, simple),
        static_cast<_aclblas_handle*>(handle));
}
