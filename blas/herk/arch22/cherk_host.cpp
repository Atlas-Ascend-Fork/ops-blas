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
#include <cstdint>
#include <limits>
#include <vector>

#include "cann_ops_blas.h"
#include "log/log.h"
#include "cherk_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

namespace {
constexpr const char* OP_TAG = "aclblasCherk";
constexpr uint32_t CUBE_M = CHERK_ARCH22_CUBE_M_ALIGN;
constexpr uint32_t CUBE_N = 128U;
constexpr uint32_t DIRECT_TILE = 128U;
constexpr uint32_t COMPACT_TILE = 128U;

void FillInterleaveMetadata(std::vector<uint32_t>& metadata, uint32_t offset, uint32_t count, uint32_t planeElements)
{
    for (uint32_t index = 0U; index < count; ++index) {
        const uint32_t planar = index >> 1U;
        const uint32_t plane = (index & 1U) == 0U ? 0U : planeElements;
        metadata[offset + index] = (plane + planar) * sizeof(float);
    }
}

void FillTransposeMetadata(
    std::vector<uint32_t>& metadata, uint32_t offset, uint32_t count, uint32_t rows, uint32_t columnStride)
{
    if (rows == 0U) {
        return;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        const uint32_t row = index / rows;
        const uint32_t column = index % rows;
        metadata[offset + index] = (column * columnStride + row) * sizeof(float);
    }
}

void FillTriangleMetadata(std::vector<uint32_t>& metadata)
{
    auto* diagonal = reinterpret_cast<uint8_t*>(metadata.data() + CHERK_ARCH22_META_DIAGONAL_OFFSET);
    auto* upperTriangle = reinterpret_cast<uint8_t*>(metadata.data() + CHERK_ARCH22_META_UPPER_OFFSET);
    auto* lowerTriangle = reinterpret_cast<uint8_t*>(metadata.data() + CHERK_ARCH22_META_LOWER_OFFSET);
    for (uint32_t row = 0U; row < 64U; ++row) {
        for (uint32_t column = 0U; column < 64U; ++column) {
            const uint32_t index = row * 64U + column;
            const uint8_t bit = static_cast<uint8_t>(1U << (index & 7U));
            if (row == column) {
                diagonal[index >> 3U] |= bit;
            }
            if (row <= column) {
                upperTriangle[index >> 3U] |= bit;
            }
            if (row >= column) {
                lowerTriangle[index >> 3U] |= bit;
            }
        }
    }
}

GM_ADDR UploadCherkMetadata(_aclblas_handle* handle, const std::vector<uint32_t>& metadata)
{
    void* deviceMetadata = nullptr;
    const size_t bytes = metadata.size() * sizeof(uint32_t);
    aclError ret = aclrtMalloc(&deviceMetadata, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "failed to allocate persistent arch22 metadata");
        return nullptr;
    }
    ret = aclrtMemcpy(deviceMetadata, bytes, metadata.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(deviceMetadata);
        OP_LOGE(OP_TAG, "failed to upload persistent arch22 metadata");
        return nullptr;
    }
    handle->cherk_arch22_metadata = deviceMetadata;
    return reinterpret_cast<GM_ADDR>(deviceMetadata);
}

GM_ADDR GetCherkArch22Metadata(_aclblas_handle* handle)
{
    if (handle->cherk_arch22_metadata != nullptr) {
        return reinterpret_cast<GM_ADDR>(handle->cherk_arch22_metadata);
    }
    std::vector<uint32_t> metadata(CHERK_ARCH22_META_WORDS, 0U);
    FillInterleaveMetadata(
        metadata, CHERK_ARCH22_META_INTERLEAVE_OFFSET, CHERK_ARCH22_META_INTERLEAVE_COUNT, 64U * 64U);
    FillTransposeMetadata(metadata, CHERK_ARCH22_META_TRANSPOSE_OFFSET, CHERK_ARCH22_META_TRANSPOSE_COUNT, 64U, 64U);
    FillTransposeMetadata(metadata, CHERK_ARCH22_META_PACK_C_OFFSET, CHERK_ARCH22_META_PACK_C_COUNT, 64U, 32U);
    FillTransposeMetadata(metadata, CHERK_ARCH22_META_PACK_C128_OFFSET, CHERK_ARCH22_META_PACK_C128_COUNT, 128U, 32U);
    FillTransposeMetadata(metadata, CHERK_ARCH22_META_TINY8_TRANSPOSE_OFFSET, 8U * 8U, 8U, 8U);
    FillTransposeMetadata(metadata, CHERK_ARCH22_META_TINY16_TRANSPOSE_OFFSET, 16U * 16U, 16U, 16U);
    FillTransposeMetadata(metadata, CHERK_ARCH22_META_TINY_TRANSPOSE_OFFSET, 32U * 32U, 32U, 32U);
    FillInterleaveMetadata(
        metadata, CHERK_ARCH22_META_TINY_INTERLEAVE_OFFSET, CHERK_ARCH22_META_TINY_INTERLEAVE_COUNT, 32U);
    FillInterleaveMetadata(
        metadata, CHERK_ARCH22_META_TINY64_INTERLEAVE_OFFSET, CHERK_ARCH22_META_TINY64_INTERLEAVE_COUNT, 64U);
    FillInterleaveMetadata(
        metadata, CHERK_ARCH22_META_OUTER_K2_INTERLEAVE_OFFSET, CHERK_ARCH22_META_OUTER_K2_INTERLEAVE_COUNT, 128U * 8U);
    FillTriangleMetadata(metadata);
    return UploadCherkMetadata(handle, metadata);
}

aclblasStatus_t ValidateEnums(aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE(OP_TAG, "ValidateEnums: invalid uplo enum: %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (trans == ACLBLAS_OP_T) {
        OP_LOGE(OP_TAG, "ValidateEnums: OP_T is valid but unsupported by CHERK");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_C) {
        OP_LOGE(OP_TAG, "ValidateEnums: invalid trans enum: %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateValues(
    aclblasOperation_t trans, int n, int k, const float* alpha, const aclblasComplex* a, int lda, const float* beta,
    const aclblasComplex* c, int ldc)
{
    if (n < 0 || k < 0) {
        OP_LOGE(OP_TAG, "ValidateValues: invalid n=%d or k=%d", n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE(OP_TAG, "ValidateValues: alpha and beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int minLda = trans == ACLBLAS_OP_N ? std::max(1, n) : std::max(1, k);
    if (lda < minLda) {
        OP_LOGE(OP_TAG, "ValidateValues: invalid lda=%d, expected >= %d", lda, minLda);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int minLdc = std::max(1, n);
    if (ldc < minLdc) {
        OP_LOGE(OP_TAG, "ValidateValues: invalid ldc=%d, expected >= %d", ldc, minLdc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (a == nullptr && n > 0 && k > 0) {
        OP_LOGE(OP_TAG, "ValidateValues: A must not be nullptr when n=%d and k=%d", n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (c == nullptr && n > 0) {
        OP_LOGE(OP_TAG, "ValidateValues: C must not be nullptr when n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool WorkspaceSizeIsSafe(uint32_t n, uint32_t k, bool directPath, bool compactPath)
{
    const uint64_t nAlign = directPath || compactPath ? DIRECT_TILE : CHERK_ARCH22_CUBE_M_ALIGN;
    const uint64_t cubeN = (static_cast<uint64_t>(n) + nAlign - 1U) / nAlign * nAlign;
    const uint64_t cubeK = (static_cast<uint64_t>(k) + CHERK_ARCH22_CUBE_K_ALIGN - 1U) / CHERK_ARCH22_CUBE_K_ALIGN *
                           CHERK_ARCH22_CUBE_K_ALIGN;
    const uint64_t nk = cubeN * cubeK;
    const uint64_t nn = cubeN * cubeN;
    constexpr uint64_t maxFloats = static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / sizeof(float);
    const uint64_t inputPlanes = directPath ? 3U : 2U;
    return nk <= maxFloats / inputPlanes && nn <= (maxFloats - inputPlanes * nk) / 2U;
}

struct CherkPathSelection {
    bool directPath;
    bool compactPath;
    bool tinyVector;
    bool tinyTranspose;
    bool tiny1Direct;
    bool square64Mix;
    bool square128Mix;
    bool outerK2;
};

void ConfigureCherkCoreCounts(CherkArch22TilingData& tiling, uint32_t n, uint32_t aivCores, uint32_t aicCores)
{
    tiling.aivCoreNum = std::max<uint32_t>(1U, std::min<uint32_t>(aivCores, n));
    if (tiling.smallPath == 2U || tiling.outerPath != 0U) {
        tiling.aivCoreNum = std::max<uint32_t>(1U, std::min<uint32_t>(aivCores, (n + 7U) / 8U));
    }
    const bool fusedLowKMix = tiling.directPath != 0U && tiling.cubeK == 64U && (n < 512U || n >= 2048U);
    const bool stridedLowKPostPackMix = tiling.directPath != 0U && tiling.cubeK == 64U && n > 1024U && n < 2048U;
    const bool stridedPostPackMix =
        stridedLowKPostPackMix || (tiling.directPath != 0U && tiling.cubeK > 64U && tiling.cubeK <= 192U && n >= 2048U);
    tiling.directMixMode = fusedLowKMix ? 1U : (stridedPostPackMix ? 3U : 0U);
    const uint64_t tileAxis = static_cast<uint64_t>(tiling.cubeN) / DIRECT_TILE;
    const uint64_t tileCount =
        tiling.directPath != 0U ?
            tileAxis * (tileAxis + 1ULL) / 2ULL :
            static_cast<uint64_t>(tiling.cubeN / (tiling.compactPath != 0U ? COMPACT_TILE : CUBE_M)) *
                (tiling.cubeN / CUBE_N);
    const uint64_t compactAxis = tiling.compactPath != 0U ? static_cast<uint64_t>(tiling.cubeN / COMPACT_TILE) : 0ULL;
    const uint64_t compactRealJobs = compactAxis * (compactAxis + 1ULL) / 2ULL;
    const uint64_t schedulerJobs = tiling.compactPath != 0U ? tileCount + compactRealJobs : tileCount;
    tiling.aicCoreNum = std::max<uint32_t>(
        1U, static_cast<uint32_t>(std::min<uint64_t>(aicCores, std::max<uint64_t>(schedulerJobs, 1U))));
    if (tiling.directMixMode == 1U) {
        const uint64_t workUnits = 2ULL * tileCount;
        while (tiling.aicCoreNum > 1U && workUnits % (2ULL * tiling.aicCoreNum) != 0ULL) {
            --tiling.aicCoreNum;
        }
    }
}

uint8_t SelectCherkSmallPath(const CherkPathSelection& paths, uint32_t n, uint32_t k)
{
    if (paths.tiny1Direct) {
        return 6U;
    }
    if (n <= 4U && k <= 4U) {
        return 1U;
    }
    if (paths.tinyVector) {
        return 2U;
    }
    if (paths.tinyTranspose) {
        return 7U;
    }
    if (paths.square64Mix) {
        return 3U;
    }
    return paths.square128Mix ? 5U : 0U;
}

bool CherkNeedsNoWorkspace(const CherkArch22TilingData& tiling)
{
    return tiling.skipProduct != 0U || tiling.smallPath == 1U || tiling.smallPath == 2U || tiling.smallPath == 4U ||
           tiling.smallPath == 6U || tiling.smallPath == 7U || tiling.outerPath != 0U;
}

aclblasStatus_t InitializeCherkTiling(
    CherkArch22TilingData& tiling, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldc, const CherkPathSelection& paths)
{
    tiling = {};
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldc = ldc;
    tiling.directPath = static_cast<uint8_t>(paths.directPath);
    tiling.compactPath = static_cast<uint8_t>(paths.compactPath);
    const uint32_t nAlign =
        tiling.directPath != 0U || tiling.compactPath != 0U ? COMPACT_TILE : CHERK_ARCH22_CUBE_M_ALIGN;
    tiling.cubeN = (n + nAlign - 1U) / nAlign * nAlign;
    tiling.cubeK = (k + CHERK_ARCH22_CUBE_K_ALIGN - 1U) / CHERK_ARCH22_CUBE_K_ALIGN * CHERK_ARCH22_CUBE_K_ALIGN;
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.skipProduct = static_cast<uint8_t>(k == 0U);
    tiling.outerPath = static_cast<uint8_t>(paths.outerK2);
    tiling.smallPath = SelectCherkSmallPath(paths, n, k);
    return ACLBLAS_STATUS_SUCCESS;
}

void ConfigureCherkPackRows(CherkArch22TilingData& tiling)
{
    uint64_t ubBytes = 192ULL * 1024ULL;
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform != nullptr) {
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    }
    constexpr uint64_t packUbSafetyBytes = 256ULL;
    const uint64_t packBytesPerRow = static_cast<uint64_t>(tiling.cubeN) * 5ULL * sizeof(float);
    const uint64_t packRowsByUb =
        ubBytes > packUbSafetyBytes && packBytesPerRow != 0ULL ? (ubBytes - packUbSafetyBytes) / packBytesPerRow : 1ULL;
    tiling.directPackBatchRows = static_cast<uint8_t>(
        packRowsByUb >= 8ULL ? 8U : (packRowsByUb >= 4ULL ? 4U : (packRowsByUb >= 2ULL ? 2U : 1U)));
}

aclblasStatus_t FinalizeCherkTiling(
    _aclblas_handle* handle, uint32_t n, uint32_t k, CherkArch22TilingData& tiling, uint32_t aivCores,
    uint32_t aicCores)
{
    ConfigureCherkPackRows(tiling);
    ConfigureCherkCoreCounts(tiling, n, aivCores, aicCores);
    if (CherkNeedsNoWorkspace(tiling)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!WorkspaceSizeIsSafe(n, k, tiling.directPath != 0U, tiling.compactPath != 0U)) {
        OP_LOGE(OP_TAG, "FinalizeCherkTiling: workspace size overflow for n=%u, k=%u", n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const size_t workspaceBytes = tiling.directPath != 0U ?
                                      CherkArch22DirectWorkspaceBytes(tiling.cubeN, tiling.cubeK) :
                                      CherkArch22WorkspaceBytes(tiling.cubeN, tiling.cubeK);
    return EnsureDefaultWorkspace(handle, workspaceBytes);
}

CherkPathSelection SelectCherkPaths(aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k)
{
    const bool compactFamily = n >= 128U && n < 768U && k > 64U;
    const bool directFamily = !compactFamily && n >= 64U && (n <= 512U || k <= 300U);
    const bool directN = trans == ACLBLAS_OP_N && k != 0U && directFamily;
    const bool directC = trans == ACLBLAS_OP_C && k != 0U && directFamily && !(n < 300U && k > 3500U);
    return {
        directN || directC,
        compactFamily && !(directN || directC),
        // Fixed-size-only kernels do not define a general tail contract.
        // Use the generic small/direct/compact paths for these inputs.
        false,
        false,
        false,
        false,
        false,
        uplo == ACLBLAS_UPPER && trans == ACLBLAS_OP_N && CherkArch22OuterGeometry(n, k),
    };
}

aclblasStatus_t PrepareTiling(
    _aclblas_handle* handle, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k, uint32_t lda,
    uint32_t ldc, CherkArch22TilingData& tiling)
{
    const uint32_t aivCores = GetAivCoreCount();
    const uint32_t aicCores = GetAicCoreCount();
    if (aivCores == 0U || aicCores == 0U) {
        OP_LOGE(OP_TAG, "PrepareTiling: failed to query AIV/AIC core count");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const CherkPathSelection paths = SelectCherkPaths(uplo, trans, n, k);
    InitializeCherkTiling(tiling, uplo, trans, n, k, lda, ldc, paths);
    return FinalizeCherkTiling(handle, n, k, tiling, aivCores, aicCores);
}
} // namespace

extern "C" aclblasStatus_t aclblasCherk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha,
    const aclblasComplex* a, int lda, const float* beta, aclblasComplex* c, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "aclblasCherk: handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    aclblasStatus_t status = ValidateEnums(uplo, trans);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateValues(trans, n, k, alpha, a, lda, beta, c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* internalHandle = static_cast<_aclblas_handle*>(handle);
    CherkArch22TilingData tiling{};
    status = PrepareTiling(
        internalHandle, uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(lda),
        static_cast<uint32_t>(ldc), tiling);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    void* workspace = GetEffectiveWorkspace(internalHandle);
    GM_ADDR metadata = nullptr;
    if (tiling.smallPath != 1U) {
        metadata = GetCherkArch22Metadata(internalHandle);
        if (metadata == nullptr) {
            OP_LOGE(OP_TAG, "aclblasCherk: failed to prepare persistent arch22 metadata");
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    cherk_arch22_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float*>(alpha)), reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(a)),
        reinterpret_cast<GM_ADDR>(const_cast<float*>(beta)), reinterpret_cast<GM_ADDR>(c),
        reinterpret_cast<GM_ADDR>(workspace), metadata, tiling, internalHandle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
