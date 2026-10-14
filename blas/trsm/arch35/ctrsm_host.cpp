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
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "ctrsm_kernel.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {

constexpr uint32_t CTRSM_SIMT_THREADS = 64;
constexpr uint64_t CTRSM_ZERO_ELEMS_PER_BLOCK = 1024;
constexpr int CTRSM_BLOCK_MIN_ORDER = 64;
constexpr int CTRSM_BLOCK_MIN_RHS = 32;
constexpr size_t CTRSM_MATMUL_SYSTEM_WORKSPACE = 16U * 1024U * 1024U;
constexpr size_t CTRSM_CUBE_TILING_BUFFER_SIZE = 1024;
constexpr size_t CTRSM_CUBE_TILING_CACHE_SIZE = 64;
// Fixed odd multiplier: multiplication permutes the 64 cache buckets instead of
// discarding low bits as an even multiplier would. Collisions are allowed: each
// hit checks the full (m, n, k, coreCount) key. Retain this value for stable mapping.
constexpr uint64_t CTRSM_CUBE_TILING_HASH_MULTIPLIER = 1315423911ULL;

struct CtrsmCubeTilingCacheEntry {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t coreCount = 0;
    size_t dataSize = 0;
    bool valid = false;
    alignas(8) std::array<uint8_t, CTRSM_CUBE_TILING_BUFFER_SIZE> data{};
};

inline bool IsZero(const aclblasComplex& value)
{
    // Either sign of zero is valid for each component of the zero-alpha fast path.
    return value.real == 0.0f && value.imag == 0.0f;
}

CtrsmTilingData CalcCtrsmTiling(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    int lda, int ldb, const aclblasComplex& alpha, uint32_t coreNum)
{
    CtrsmTilingData tiling{};
    tiling.side = static_cast<uint32_t>(side);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint64_t>(lda);
    tiling.ldb = static_cast<uint64_t>(ldb);
    tiling.alphaReal = alpha.real;
    tiling.alphaImag = alpha.imag;
    tiling.numThreads = CTRSM_SIMT_THREADS;
    tiling.systemCount = side == ACLBLAS_SIDE_LEFT ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);

    uint32_t usedCores = std::min(coreNum, tiling.systemCount);
    if (usedCores == 0) {
        usedCores = 1;
    }
    tiling.perCoreSystems = tiling.systemCount / usedCores;
    tiling.coreRemainder = tiling.systemCount % usedCores;
    return tiling;
}

aclblasStatus_t ValidateCtrsmParams(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb)
{
    CHECK_RET(side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
              OP_LOGE("aclblasCtrsm", "invalid side=%d", static_cast<int>(side));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasCtrsm", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasCtrsm", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
              OP_LOGE("aclblasCtrsm", "invalid diag=%d", static_cast<int>(diag));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0 && n >= 0, OP_LOGE("aclblasCtrsm", "m and n must be >= 0, got m=%d, n=%d", m, n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasCtrsm", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);

    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const int k = side == ACLBLAS_SIDE_LEFT ? m : n;
    CHECK_RET(lda >= std::max(1, k), OP_LOGE("aclblasCtrsm", "invalid lda=%d, k=%d", lda, k);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= std::max(1, m), OP_LOGE("aclblasCtrsm", "invalid ldb=%d, m=%d", ldb, m);
              return ACLBLAS_STATUS_INVALID_VALUE);

    CHECK_RET(B != nullptr, OP_LOGE("aclblasCtrsm", "B must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    constexpr uint64_t maxElements = std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
    if (static_cast<uint64_t>(ldb) * static_cast<uint64_t>(n) > maxElements) {
        OP_LOGE("aclblasCtrsm", "B storage size overflow: ldb=%d, n=%d", ldb, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsZero(*alpha)) {
        CHECK_RET(A != nullptr, OP_LOGE("aclblasCtrsm", "A must not be nullptr when alpha is nonzero");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        if (static_cast<uint64_t>(lda) * static_cast<uint64_t>(k) > maxElements) {
            OP_LOGE("aclblasCtrsm", "A storage size overflow: lda=%d, order=%d", lda, k);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool AddWorkspaceMatrix(uint64_t rows, uint64_t cols, size_t& bytes, size_t& total)
{
    constexpr size_t alignment = 32;
    constexpr size_t maxBytes = std::numeric_limits<size_t>::max();
    if (cols == 0 || rows > (maxBytes - alignment + 1) / sizeof(float) / cols) {
        return false;
    }
    const size_t rawBytes = static_cast<size_t>(rows * cols * sizeof(float));
    bytes = (rawBytes + alignment - 1) / alignment * alignment;
    if (bytes > maxBytes - total) {
        return false;
    }
    total += bytes;
    return true;
}

size_t CtrsmCubeTilingCacheIndex(uint32_t m, uint32_t n, uint32_t k, uint32_t coreCount)
{
    uint64_t hash = m;
    hash = hash * CTRSM_CUBE_TILING_HASH_MULTIPLIER + n;
    hash = hash * CTRSM_CUBE_TILING_HASH_MULTIPLIER + k;
    hash = hash * CTRSM_CUBE_TILING_HASH_MULTIPLIER + coreCount;
    return static_cast<size_t>(hash % CTRSM_CUBE_TILING_CACHE_SIZE);
}

bool GetCtrsmCubeTiling(uint32_t m, uint32_t n, uint32_t k, uint32_t coreCount, CtrsmCubeTilingCacheEntry& result)
{
    static std::array<CtrsmCubeTilingCacheEntry, CTRSM_CUBE_TILING_CACHE_SIZE> cache{};
    static std::mutex cacheMutex;
    std::lock_guard<std::mutex> lock(cacheMutex);
    CtrsmCubeTilingCacheEntry& entry = cache[CtrsmCubeTilingCacheIndex(m, n, k, coreCount)];
    if (entry.valid && entry.m == m && entry.n == n && entry.k == k && entry.coreCount == coreCount) {
        result = entry;
        return true;
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
    if (platform == nullptr) {
        return false;
    }
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*platform);
    tilingApi.SetDim(coreCount);
    tilingApi.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tilingApi.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetOrgShape(m, n, k);
    tilingApi.SetShape(m, n, k);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);
    optiling::TCubeTiling cubeTiling{};
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        return false;
    }
    const size_t dataSize = cubeTiling.GetDataSize();
    if (dataSize > entry.data.size()) {
        return false;
    }

    entry = {};
    entry.m = m;
    entry.n = n;
    entry.k = k;
    entry.coreCount = coreCount;
    entry.dataSize = dataSize;
    entry.valid = true;
    cubeTiling.SaveToBuffer(entry.data.data(), entry.dataSize);
    result = entry;
    return true;
}

} // namespace

namespace {
aclblasStatus_t CalcCtrsmBlockedTiling(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex& alpha, int lda, int ldb, CtrsmTilingData& t)
{
    t.side = side;
    t.trans = trans;
    t.diag = diag;
    t.originalOrder = side == ACLBLAS_SIDE_LEFT ? m : n;
    const uint64_t paddedOrder =
        (static_cast<uint64_t>(t.originalOrder) + CTRSM_CONVERT_TILE - 1) / CTRSM_CONVERT_TILE * CTRSM_CONVERT_TILE;
    t.rhsCount = side == ACLBLAS_SIDE_LEFT ? n : m;
    const bool transpose = (trans != ACLBLAS_OP_N) != (side == ACLBLAS_SIDE_RIGHT);
    t.effectiveUpper = (uplo == ACLBLAS_UPPER) != transpose;
    t.lda = lda;
    t.ldb = ldb;
    t.alphaReal = alpha.real;
    t.alphaImag = alpha.imag;
    t.panelSize = CTRSM_PANEL_SIZE;
    t.groupPanels = CTRSM_GROUP_PANELS;
    t.splitCols = CTRSM_SPLIT_COLS;
    t.splitCount = (static_cast<uint64_t>(t.rhsCount) + t.splitCols - 1) / t.splitCols;
    t.combinedCols = 2ULL * t.splitCount * t.splitCols;
    // Matmul SetOrgShape takes signed int strides. Reject before allocation or launch.
    constexpr uint64_t maxMatmulDim = std::numeric_limits<int32_t>::max();
    if (paddedOrder > maxMatmulDim / 2 || t.combinedCols > maxMatmulDim) {
        OP_LOGE("aclblasCtrsm", "Packed layout exceeds Matmul signed-int stride range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    t.order = static_cast<uint32_t>(paddedOrder);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t CtrsmBlockedPath(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex& alpha, const aclblasComplex* a, int lda,
    aclblasComplex* b, int ldb, uint32_t cores)
{
    CtrsmTilingData t = CalcCtrsmTiling(side, uplo, trans, diag, m, n, lda, ldb, alpha, cores);
    t.prepareCores = cores;
    const auto tilingStatus = CalcCtrsmBlockedTiling(side, uplo, trans, diag, m, n, alpha, lda, ldb, t);
    if (tilingStatus != ACLBLAS_STATUS_SUCCESS) {
        return tilingStatus;
    }
    size_t aBytes = 0;
    size_t bBytes = 0;
    size_t xBytes = 0;
    const size_t riskBytes =
        (static_cast<size_t>(cores) + t.order / CTRSM_CONVERT_TILE) * CTRSM_RISK_WORDS * sizeof(uint32_t);
    size_t total = CTRSM_MATMUL_SYSTEM_WORKSPACE + riskBytes;
    if (!AddWorkspaceMatrix(t.order, 2ULL * t.order, aBytes, total) ||
        !AddWorkspaceMatrix(t.order, t.combinedCols, bBytes, total) ||
        !AddWorkspaceMatrix(t.splitCount, 4ULL * t.groupPanels * t.panelSize * t.splitCols, xBytes, total)) {
        OP_LOGE("aclblasCtrsm", "Blocked workspace size overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (total > GetEffectiveWorkspaceSize(h)) {
        const auto status = EnsureDefaultWorkspace(h, total);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    auto* packedA = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    auto* packedB = packedA + aBytes;
    auto* packedX = packedB + bBytes;
    t.riskOffset = aBytes + bBytes + xBytes;
    auto* system = packedX + xBytes + riskBytes;
    CtrsmCubeTilingCacheEntry cube{};
    if (!GetCtrsmCubeTiling(t.order, 2 * t.splitCols, 2 * t.groupPanels * t.panelSize, 1, cube)) {
        OP_LOGE("aclblasCtrsm", "Cannot obtain Cube tiling for blocked solve");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ctrsm_block_prepare_dma_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)), reinterpret_cast<uint8_t*>(b), packedA, packedB, t,
        cores, h->stream);
    ctrsm_mix_block_kernel_do(packedA, packedB, packedX, system, t, cube.data.data(), t.splitCount, h->stream);
    ctrsm_block_finish_dma_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)), packedA, packedB, reinterpret_cast<uint8_t*>(b), t,
        cores, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

aclblasStatus_t aclblasCtrsm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    aclblasComplex* B, int ldb)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasCtrsm", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    aclblasStatus_t status = ValidateCtrsmParams(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0, OP_LOGE("aclblasCtrsm", "GetAivCoreCount failed"); return ACLBLAS_STATUS_INTERNAL_ERROR);

    if (IsZero(*alpha)) {
        const uint64_t totalElements = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
        uint64_t requestedBlocks = (totalElements + CTRSM_ZERO_ELEMS_PER_BLOCK - 1) / CTRSM_ZERO_ELEMS_PER_BLOCK;
        uint32_t numBlocks = static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum, requestedBlocks));
        if (numBlocks == 0) {
            numBlocks = 1;
        }
        ctrsm_zero_kernel_do(
            reinterpret_cast<uint8_t*>(B), static_cast<uint32_t>(m), static_cast<uint32_t>(n),
            static_cast<int32_t>(ldb), numBlocks, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }

    const int order = side == ACLBLAS_SIDE_LEFT ? m : n;
    const int rhsCount = side == ACLBLAS_SIDE_LEFT ? n : m;
    if (order >= CTRSM_BLOCK_MIN_ORDER && rhsCount >= CTRSM_BLOCK_MIN_RHS) {
        return CtrsmBlockedPath(h, side, uplo, trans, diag, m, n, *alpha, A, lda, B, ldb, aivCoreNum);
    }

    CtrsmTilingData tiling = CalcCtrsmTiling(side, uplo, trans, diag, m, n, lda, ldb, *alpha, aivCoreNum);
    uint32_t numBlocks = std::min(aivCoreNum, tiling.systemCount);
    if (numBlocks == 0) {
        numBlocks = 1;
    }
    ctrsm_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), reinterpret_cast<uint8_t*>(B), tiling, numBlocks,
        h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
