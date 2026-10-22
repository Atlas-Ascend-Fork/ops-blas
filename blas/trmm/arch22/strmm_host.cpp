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
 * \file strmm_host.cpp
 * \brief Atlas A2/A3 STRMM implementation (arch22).
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "strmm_tiling_data.h"

void strmm_pack_kernel_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* packed,
                          uint8_t* b, uint8_t* packedB, StrmmPackTilingData tiling);
void strmm_gemm_kernel_do(uint32_t blocks, void* stream, uint8_t* left, uint8_t* right, uint8_t* temp,
                          StrmmGemmTilingData tiling);
void strmm_scale_kernel_do(uint32_t blocks, void* stream, uint8_t* temp, uint8_t* c, StrmmScaleTilingData tiling);
void strmm_direct_kernel_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c,
                            StrmmDirectTilingData tiling);
void strmm_reference_order_kernel_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* temp,
                                     StrmmReferenceTilingData tiling);
uint32_t strmm_gemm_single_k();

namespace {
constexpr const char* OP_TAG = "aclblasStrmm";

// Preserve the tuned path-selection thresholds.
constexpr uint32_t kAlignBMinNExclusive = 64U;
constexpr uint32_t kAlignBMaxN = 128U;
constexpr uint32_t kAlignBMinM = 512U;
constexpr uint32_t kDirectGemmMaxM = 128U;
constexpr uint32_t kDirectGemmMinN = 64U;
constexpr float kReferenceAlphaAbs = 2.0F;
constexpr int kDirectMaxDim = 32;
// Eight float32 elements occupy one 32-byte GM update granule.
constexpr int kDirectRowTile = 8;
constexpr int kDirectLowerMinM = 16;
constexpr int kDirectLowerMaxM = 32;
constexpr int kDirectColWidth = 16;
// A float32 leading dimension must fit a uint32_t DMA byte stride.
constexpr int kMaxDmaLeadingDim = 0x3FFFFFFF;

aclblasStatus_t ReadAlpha(const float* alpha, aclrtStream stream, float& value)
{
    bool isDevice = false;
    aclblasStatus_t location = CheckPtrLocation(alpha, &isDevice);
    if (location != ACLBLAS_STATUS_SUCCESS) {
        return location;
    }
    if (!isDevice) {
        value = *alpha;
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMemcpyAsync(&value, sizeof(value), alpha, sizeof(value), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "device alpha read failed: %d", ret);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "device alpha stream sync failed: %d", ret);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ZeroOutput(_aclblas_handle* h, uint32_t m, uint32_t n, uint32_t ldc, float* C)
{
    const size_t colBytes = static_cast<size_t>(m) * sizeof(float);
    if (ldc == m) {
        const size_t bytes = colBytes * n;
        if (aclrtMemsetAsync(C, bytes, 0, bytes, h->stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    for (uint32_t col = 0; col < n; ++col) {
        if (aclrtMemsetAsync(C + static_cast<size_t>(col) * ldc, colBytes, 0, colBytes, h->stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Keep the checked workspace layout together for sizing and launch address calculation.
struct StrmmWorkspaceLayout {
    uint32_t packedLd;
    uint32_t tempLd;
    uint64_t alignedABytes = 0U;
    uint64_t tempBytes = 0U;
    uint64_t totalBytes = 0U;
    bool alignB;
};

aclblasStatus_t PrepareStrmmWorkspace(_aclblas_handle* h, aclblasSideMode_t side, uint32_t m, uint32_t n,
                                      uint32_t dim, uint32_t ldb, StrmmWorkspaceLayout& layout)
{
    const uint64_t packedLd64 = (static_cast<uint64_t>(dim) + CBLAS3_EXPAND_TILE - 1U) /
                                CBLAS3_EXPAND_TILE * CBLAS3_EXPAND_TILE;
    const uint64_t tempLd64 = (static_cast<uint64_t>(m) + CBLAS3_TEMP_ALIGN - 1U) /
                              CBLAS3_TEMP_ALIGN * CBLAS3_TEMP_ALIGN;
    if (packedLd64 > std::numeric_limits<uint32_t>::max() ||
        tempLd64 > std::numeric_limits<uint32_t>::max()) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    layout.packedLd = static_cast<uint32_t>(packedLd64);
    layout.tempLd = static_cast<uint32_t>(tempLd64);

    uint64_t aBytes = 0U;
    if (!CBlas3TryMulU64(packedLd64, dim, &aBytes) ||
        !CBlas3TryMulU64(aBytes, sizeof(float), &aBytes) ||
        !CBlas3TryMulU64(tempLd64, n, &layout.tempBytes) ||
        !CBlas3TryMulU64(layout.tempBytes, sizeof(float), &layout.tempBytes) ||
        !CBlas3TryAlignGm(aBytes, &layout.alignedABytes) ||
        !CBlas3TryAddU64(layout.alignedABytes, layout.tempBytes, &layout.totalBytes) ||
        layout.totalBytes > ACLBLAS_MAX_WORKSPACE_SIZE) {
        OP_LOGE(OP_TAG, "workspace size overflow or exceeds the library limit");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    // A small right-side triangle repeatedly gathers rows of B with its long
    // physical stride. Align that stride once in the existing pack launch.
    layout.alignB = side == ACLBLAS_SIDE_RIGHT && n > kAlignBMinNExclusive && n <= kAlignBMaxN &&
                    m >= kAlignBMinM && ldb % kDirectRowTile != 0U;
    if (layout.alignB) {
        uint64_t expandedBytes = 0U;
        if (CBlas3TryAddU64(layout.totalBytes, layout.tempBytes, &expandedBytes) &&
            expandedBytes <= ACLBLAS_MAX_WORKSPACE_SIZE) {
            layout.totalBytes = expandedBytes;
        } else {
            layout.alignB = false; // Optional optimization must not reduce the supported workspace range.
        }
    }
    return EnsureDefaultWorkspace(h, static_cast<size_t>(layout.totalBytes));
}

bool CanWriteStrmmGemmDirectly(uint32_t m, uint32_t n, float alpha, const float* B,
                              float* C, uint32_t ldb, uint32_t ldc)
{
    return alpha == 1.0F && B != C && m <= kDirectGemmMaxM && n >= kDirectGemmMinN &&
           ldb == m && ldc == m;
}

CBlas3GemmTilingData MakeStrmmGemmTiling(aclblasSideMode_t side, bool cubeTransposeA,
    uint32_t m, uint32_t n, uint32_t ldb, uint32_t ldc, bool directGemmOutput,
    const StrmmWorkspaceLayout& layout)
{
    CBlas3GemmTilingData gemm{};
    gemm.m = n;
    gemm.n = m;
    gemm.k = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    gemm.ldc = directGemmOutput ? ldc : layout.tempLd;
    gemm.mBlocks = CeilDiv<uint32_t>(n, CBLAS3_TILE_M);
    gemm.nBlocks = CeilDiv<uint32_t>(m, CBLAS3_TILE_N);
    gemm.uploMode = CBLAS3_UPLO_NONE;
    gemm.transMode = cubeTransposeA
                         ? ((side == ACLBLAS_SIDE_LEFT) ? CBLAS3_GEMM_TRANS_RIGHT : CBLAS3_GEMM_TRANS_LEFT)
                         : CBLAS3_GEMM_TRANS_NONE;
    const uint32_t effectiveLdb = layout.alignB ? layout.tempLd : ldb;
    gemm.ldA = (side == ACLBLAS_SIDE_LEFT) ? effectiveLdb : layout.packedLd;
    gemm.ldB = (side == ACLBLAS_SIDE_LEFT) ? layout.packedLd : effectiveLdb;
    gemm.packedLd = cubeTransposeA ? layout.packedLd : gemm.ldA;
    return gemm;
}

void LaunchStrmmGemm(_aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, uint32_t gemmBlocks, const CBlas3GemmTilingData& gemm,
    uint8_t* left, uint8_t* right, uint8_t* gemmOutput)
{
    const uint32_t triangleOnRight = side == ACLBLAS_SIDE_LEFT;
    const uint32_t lowerTriangle = (uplo == ACLBLAS_LOWER) == (trans == ACLBLAS_OP_T);
    auto launch = [triangleOnRight, lowerTriangle](uint32_t blocks, void* stream,
        uint8_t* l, uint8_t* r, uint8_t* output, CBlas3GemmTilingData chunk) {
        strmm_gemm_kernel_do(blocks, stream, l, r, output,
                            StrmmGemmTilingData{chunk, triangleOnRight, lowerTriangle});
    };
    CBlas3LaunchGemmChunked(launch, gemmBlocks, h->stream, left, right, gemmOutput,
                            gemm, strmm_gemm_single_k());
}

aclblasStatus_t RunStrmm(_aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo,
                         aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t m, uint32_t n,
                         float alpha, const float* A, uint32_t lda, const float* B, uint32_t ldb,
                         float* C, uint32_t ldc)
{
    const uint32_t dim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    StrmmWorkspaceLayout layout{};
    aclblasStatus_t workspaceStatus = PrepareStrmmWorkspace(h, side, m, n, dim, ldb, layout);
    if (workspaceStatus != ACLBLAS_STATUS_SUCCESS) {
        return workspaceStatus;
    }
    auto* workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    auto* packed = workspace;
    auto* temp = workspace + layout.alignedABytes;
    auto* packedB = layout.alignB ? temp + layout.tempBytes : nullptr;

    const uint32_t aivCores = GetAivCoreCount();
    const uint32_t aicCores = GetAicCoreCount();
    if (aivCores == 0U || aicCores == 0U) {
        OP_LOGE(OP_TAG, "AIV or AIC core count is zero");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // Keep A in its source orientation and let Cube transpose the A operand.
    // The STRMM Cube kernel uses independent A/B strides, including padded B.
    const bool cubeTransposeA = (trans == ACLBLAS_OP_T);
    StrmmPackTilingData pack{dim, lda, layout.packedLd, static_cast<uint32_t>(uplo),
                             static_cast<uint32_t>(cubeTransposeA ? ACLBLAS_OP_N : trans),
                             static_cast<uint32_t>(diag), layout.alignB ? m : 0U, n, ldb, layout.tempLd};
    strmm_pack_kernel_do(std::min(dim, aivCores), h->stream,
                         reinterpret_cast<uint8_t*>(const_cast<float*>(A)), packed,
                         reinterpret_cast<uint8_t*>(const_cast<float*>(B)), packedB, pack);

    // A column-major m x n buffer has the same bytes as a row-major n x m
    // buffer. Compute C^T: B^T * op(A)^T for LEFT, op(A)^T * B^T for RIGHT.
    const bool directGemmOutput = CanWriteStrmmGemmDirectly(m, n, alpha, B, C, ldb, ldc);
    const auto gemm = MakeStrmmGemmTiling(side, cubeTransposeA, m, n, ldb, ldc, directGemmOutput, layout);
    const uint32_t gemmBlocks = std::min(gemm.mBlocks * gemm.nBlocks, aicCores);
    auto* bBytes = layout.alignB ? packedB : reinterpret_cast<uint8_t*>(const_cast<float*>(B));
    auto* left = (side == ACLBLAS_SIDE_LEFT) ? bBytes : packed;
    auto* right = (side == ACLBLAS_SIDE_LEFT) ? packed : bBytes;
    auto* gemmOutput = directGemmOutput ? reinterpret_cast<uint8_t*>(C) : temp;
    LaunchStrmmGemm(h, side, uplo, trans, gemmBlocks, gemm, left, right, gemmOutput);

    if (!directGemmOutput) {
        StrmmScaleTilingData scale{m, n, ldc, layout.tempLd, alpha};
        strmm_scale_kernel_do(std::min(n, aivCores), h->stream, temp, reinterpret_cast<uint8_t*>(C), scale);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t RunStrmmReferenceOrder(_aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo,
                                      aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t m, uint32_t n, float alpha,
                                      const float* A, uint32_t lda, const float* B, uint32_t ldb,
                                      float* C, uint32_t ldc)
{
    const uint64_t tempLd64 = (static_cast<uint64_t>(m) + CBLAS3_TEMP_ALIGN - 1U) /
                              CBLAS3_TEMP_ALIGN * CBLAS3_TEMP_ALIGN;
    uint64_t bytes = 0U;
    if (tempLd64 > std::numeric_limits<uint32_t>::max() ||
        !CBlas3TryMulU64(tempLd64, n, &bytes) ||
        !CBlas3TryMulU64(bytes, sizeof(float), &bytes) ||
        bytes > ACLBLAS_MAX_WORKSPACE_SIZE) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclblasStatus_t workspaceStatus = EnsureDefaultWorkspace(h, static_cast<size_t>(bytes));
    if (workspaceStatus != ACLBLAS_STATUS_SUCCESS) {
        return workspaceStatus;
    }
    auto* temp = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0U) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    StrmmReferenceTilingData ref{m, n, lda, ldb, static_cast<uint32_t>(tempLd64),
                                  static_cast<uint32_t>(side), static_cast<uint32_t>(uplo),
                                  static_cast<uint32_t>(trans),
                                  static_cast<uint32_t>(diag), alpha};
    strmm_reference_order_kernel_do(std::min(n, cores), h->stream,
                                    reinterpret_cast<uint8_t*>(const_cast<float*>(A)),
                                    reinterpret_cast<uint8_t*>(const_cast<float*>(B)), temp, ref);
    StrmmScaleTilingData copy{m, n, ldc, static_cast<uint32_t>(tempLd64), 1.0F};
    strmm_scale_kernel_do(std::min(n, cores), h->stream, temp, reinterpret_cast<uint8_t*>(C), copy);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t RunStrmmDirect(_aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, float alphaValue,
    const float* A, int lda, const float* B, int ldb, float* C, int ldc)
{
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0U) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    StrmmDirectTilingData direct{static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                                 static_cast<uint32_t>(lda), static_cast<uint32_t>(ldb),
                                 static_cast<uint32_t>(ldc), static_cast<uint32_t>(side),
                                 static_cast<uint32_t>(uplo), static_cast<uint32_t>(trans),
                                 static_cast<uint32_t>(diag), alphaValue};
    // MTE3 writes full aligned output columns in parallel. Short or padded
    // columns stay on one block to avoid overlapping GM update granules.
    // The tiled DMA uses 32-bit byte strides. Larger valid leading
    // dimensions retain the scalar 64-bit-address, single-block path.
    const bool parallelColumns = B != C && m >= kDirectRowTile && (m % kDirectRowTile) == 0 &&
                                 ldc == m && n > 1 && lda <= kMaxDmaLeadingDim && ldb <= kMaxDmaLeadingDim;
    const bool lowerTiles = parallelColumns && side == ACLBLAS_SIDE_LEFT && uplo == ACLBLAS_LOWER &&
                            trans == ACLBLAS_OP_N && (m == kDirectLowerMinM || m == kDirectLowerMaxM) &&
                            alphaValue == alphaValue;
    const uint32_t tasks = lowerTiles ?
        static_cast<uint32_t>((m / kDirectRowTile) * ((n + kDirectColWidth - 1) / kDirectColWidth)) :
        static_cast<uint32_t>(n);
    const uint32_t blocks = parallelColumns ? std::min(cores, tasks) : 1U;
    strmm_direct_kernel_do(blocks, h->stream,
                           reinterpret_cast<uint8_t*>(const_cast<float*>(A)),
                           reinterpret_cast<uint8_t*>(const_cast<float*>(B)),
                           reinterpret_cast<uint8_t*>(C), direct);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t CheckStrmmBasicParameters(aclblasHandle_t handle, aclblasSideMode_t side,
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n)
{
    CHECK_RET(handle != nullptr,
              OP_LOGE(OP_TAG, "handle is null"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
              OP_LOGE(OP_TAG, "invalid side=%d", static_cast<int>(side)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "invalid uplo=%d", static_cast<int>(uplo)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T,
              OP_LOGE(OP_TAG, "invalid trans=%d", static_cast<int>(trans)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT,
              OP_LOGE(OP_TAG, "invalid diag=%d", static_cast<int>(diag)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0 && n >= 0,
              OP_LOGE(OP_TAG, "negative matrix dimension m=%d n=%d", m, n); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

aclblasStatus_t aclblasStrmm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
                             aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
                             const float* alpha, const float* A, int lda,
                             const float* B, int ldb, float* C, int ldc)
{
    const aclblasStatus_t paramStatus = CheckStrmmBasicParameters(handle, side, uplo, trans, diag, m, n);
    if (paramStatus != ACLBLAS_STATUS_SUCCESS) {
        return paramStatus;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const int dim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    CHECK_RET(lda >= dim && ldb >= m && ldc >= m,
              OP_LOGE(OP_TAG, "invalid leading dimension lda=%d ldb=%d ldc=%d", lda, ldb, ldc);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr && C != nullptr,
              OP_LOGE(OP_TAG, "alpha or C is null"); return ACLBLAS_STATUS_INVALID_VALUE);

    auto* h = static_cast<_aclblas_handle*>(handle);
    float alphaValue = 0.0F;
    aclblasStatus_t alphaStatus = ReadAlpha(alpha, h->stream, alphaValue);
    if (alphaStatus != ACLBLAS_STATUS_SUCCESS) {
        return alphaStatus;
    }
    if (alphaValue == 0.0F) {
        return ZeroOutput(h, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(ldc), C);
    }
    CHECK_RET(A != nullptr && B != nullptr,
              OP_LOGE(OP_TAG, "A or B is null"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (alphaValue >= kReferenceAlphaAbs || alphaValue <= -kReferenceAlphaAbs) {
        return RunStrmmReferenceOrder(h, side, uplo, trans, diag, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                                      alphaValue, A, static_cast<uint32_t>(lda), B, static_cast<uint32_t>(ldb),
                                      C, static_cast<uint32_t>(ldc));
    }
    if (m <= kDirectMaxDim && n <= kDirectMaxDim) {
        return RunStrmmDirect(h, side, uplo, trans, diag, m, n, alphaValue, A, lda, B, ldb, C, ldc);
    }
    return RunStrmm(h, side, uplo, trans, diag, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                    alphaValue, A, static_cast<uint32_t>(lda), B, static_cast<uint32_t>(ldb), C,
                    static_cast<uint32_t>(ldc));
}
