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
#include <cmath>
#include <complex>
#include <limits>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemm/arch35/cgemm_kernel.h"
#include "gemm/arch35/gemm_host_common.h"
#include "csyrkx_kernel.h"

namespace {
constexpr uint32_t THREADS = 256;
constexpr size_t ALIGNMENT = 512;

// Checked byte spans also protect allocation-free ordered/scale-only paths.
// Dimensions are positive here. This checks representability, not allocation
// ownership: callers must still provide sufficiently large device buffers.
bool AddressableSpan(const aclblasComplex* pointer, uint32_t rows, uint32_t cols, uint32_t ld)
{
    const uint64_t elements = (static_cast<uint64_t>(cols) - 1) * ld + rows;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(aclblasComplex)) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(pointer);
    const uint64_t lastByte = elements * sizeof(aclblasComplex) - 1;
    return lastByte <= std::numeric_limits<uintptr_t>::max() - base;
}

// All intermediate regions are bounded by the handle's workspace limit before
// addition, so even maximal int dimensions cannot wrap an allocation size.
bool AppendRegion(uint64_t elements, size_t& bytes, size_t& offset)
{
    if (elements > ACLBLAS_MAX_WORKSPACE_SIZE / sizeof(float)) {
        return false;
    }
    const size_t region = (static_cast<size_t>(elements) * sizeof(float) + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    if (region > ACLBLAS_MAX_WORKSPACE_SIZE - bytes) {
        return false;
    }
    offset = bytes;
    bytes += region;
    return true;
}

GemmTilingData BuildProductTiling(const CsyrkxTilingData& t, uint32_t rows, uint32_t cubes)
{
    GemmTilingData product = BuildGemmTilingData(
        t.n, t.n, t.k, rows, rows, t.productLd, t.trans ? ACLBLAS_OP_T : ACLBLAS_OP_N,
        t.trans ? ACLBLAS_OP_N : ACLBLAS_OP_T, 1.0f, 0.0f, 0.0f, 0.0f);
    ApplyGemmColMajorSwap(product);
    uint32_t tile = 128;
    while (tile > 16 && 2ULL * t.parts * CeilDiv<uint64_t>(t.n, tile) * (CeilDiv<uint64_t>(t.n, tile) + 1) < cubes) {
        tile /= 2;
    }
    product.baseM = product.baseN = tile;
    product.cgemmKPartitions = t.parts;
    product.cgemmKSpan = CeilDiv<uint64_t>(CeilDiv<uint64_t>(t.k, 16), t.parts) * 16;
    const uint64_t tileGroups = CeilDiv<uint64_t>(t.n, tile);
    product.usedCoreNum = std::min<uint64_t>(cubes, 2ULL * t.parts * tileGroups * (tileGroups + 1));
    return product;
}

aclblasStatus_t AllocateAlignedWorkspace(aclblasHandle_t handle, size_t bytes, uint8_t*& workspace)
{
    const uintptr_t original = reinterpret_cast<uintptr_t>(GetEffectiveWorkspace(handle));
    size_t padding = (ALIGNMENT - original % ALIGNMENT) % ALIGNMENT;
    if (bytes > ACLBLAS_MAX_WORKSPACE_SIZE - padding) {
        OP_LOGE("aclblasCsyrkx", "aligned workspace exceeds supported size");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    size_t required = bytes + padding;
    // A library workspace expansion can return a different base address.
    // Reserve space for the worst alignment in that case, capped at the limit.
    if (handle->workspace_owner == AclblasWorkspaceOwner::Library && required > GetEffectiveWorkspaceSize(handle)) {
        required =
            bytes > ACLBLAS_MAX_WORKSPACE_SIZE - (ALIGNMENT - 1) ? ACLBLAS_MAX_WORKSPACE_SIZE : bytes + ALIGNMENT - 1;
    }
    const auto status = EnsureDefaultWorkspace(handle, required);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCsyrkx", "workspace allocation failed, status=%d", static_cast<int>(status));
        return status;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(GetEffectiveWorkspace(handle));
    padding = (ALIGNMENT - base % ALIGNMENT) % ALIGNMENT;
    if (base == 0 || bytes > GetEffectiveWorkspaceSize(handle) || padding > GetEffectiveWorkspaceSize(handle) - bytes ||
        base > std::numeric_limits<uintptr_t>::max() - padding ||
        base + padding > std::numeric_limits<uintptr_t>::max() - (bytes - 1)) {
        OP_LOGE("aclblasCsyrkx", "aligned workspace address or capacity is invalid");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    workspace = reinterpret_cast<uint8_t*>(base + padding);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchProducts(
    aclblasHandle_t handle, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c, CsyrkxTilingData t,
    uint32_t vectors, uint32_t cubes)
{
    const uint32_t rows = t.trans ? t.k : t.n;
    const uint32_t cols = t.trans ? t.n : t.k;
    const uint64_t elements = static_cast<uint64_t>(rows) * cols;
    const uint64_t paddedN = (static_cast<uint64_t>(t.n) + 15) / 16 * 16;
    if (paddedN > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE("aclblasCsyrkx", "aligned n exceeds supported range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    t.productLd = paddedN;
    t.partStride = paddedN * paddedN;
    // Fixed, deterministic two-way partition bounds long FP32 accumulations.
    t.parts = t.k > 256 ? 2 : 1;
    const uint32_t splitBlocks = std::min<uint64_t>(vectors, CeilDiv<uint64_t>(2 * elements, 512));
    t.riskBlocks = splitBlocks;
    static_assert(sizeof(uint32_t) == sizeof(float), "Risk metadata uses 32-bit workspace elements");
    size_t offsets[9]{};
    size_t bytes = 0;
    for (uint32_t i = 0; i < 9; ++i) {
        const uint64_t regionElements =
            i < 4 ? elements : (i < 8 ? t.partStride * t.parts : uint64_t(t.riskBlocks) * CSYRKX_RISK_SLOT_WORDS);
        if (!AppendRegion(regionElements, bytes, offsets[i])) {
            OP_LOGE("aclblasCsyrkx", "workspace exceeds supported size for region %u", i);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    uint8_t* workspace = nullptr;
    const auto status = AllocateAlignedWorkspace(handle, bytes, workspace);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    uint8_t* planes[8];
    for (uint32_t i = 0; i < 8; ++i) {
        planes[i] = workspace + offsets[i];
    }
    auto* riskFlags = workspace + offsets[8];
    // Independent deinterleave also bounds every active A/B component. All
    // blocks write their own flag, so workspace reuse needs no flag memset.
    // Both transpose modes deliberately leave imaginary components unchanged.
    csyrkx_split_do(
        splitBlocks, handle->stream, reinterpret_cast<const uint8_t*>(a), reinterpret_cast<const uint8_t*>(b),
        planes[0], planes[1], planes[2], planes[3], riskFlags, t);

    const GemmTilingData product = BuildProductTiling(t, rows, cubes);
    // Col-major swapping transposes the row-major product coordinates; only the
    // opposite tile triangle is needed. Diagonal tiles are masked by epilogue.
    gemm_cgemm_triangular_products_do(
        product.usedCoreNum, handle->stream, planes[0], planes[1], planes[2], planes[3], nullptr, nullptr, planes[4],
        planes[5], planes[6], planes[7], product, !t.upper);
    const uint32_t outBlocks =
        std::min<uint64_t>(vectors, CeilDiv<uint64_t>(static_cast<uint64_t>(t.n) * t.n, THREADS));
    csyrkx_epilogue_do(
        outBlocks, handle->stream, reinterpret_cast<const uint8_t*>(a), reinterpret_cast<const uint8_t*>(b), planes[4],
        planes[5], planes[6], planes[7], riskFlags, reinterpret_cast<uint8_t*>(c), t);
    return ACLBLAS_STATUS_SUCCESS;
}
aclblasStatus_t ValidateCsyrkxArguments(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    int lda, int ldb, const aclblasComplex* beta, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCsyrkx", "handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if ((uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) ||
        (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) || n < 0 || k < 0) {
        OP_LOGE(
            "aclblasCsyrkx", "invalid uplo=%d trans=%d n=%d k=%d", static_cast<int>(uplo), static_cast<int>(trans), n,
            k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int rows = trans == ACLBLAS_OP_N ? n : k;
    if (lda < std::max(1, rows) || ldb < std::max(1, rows) || ldc < std::max(1, n) || alpha == nullptr ||
        beta == nullptr) {
        OP_LOGE("aclblasCsyrkx", "invalid leading dimensions or null scalar: lda=%d ldb=%d ldc=%d", lda, ldb, ldc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

CsyrkxTilingData BuildCsyrkxTiling(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc,
    const std::complex<float>& av, const std::complex<float>& bv, bool scaleOnly)
{
    CsyrkxTilingData t{};
    t.n = n;
    t.k = k;
    t.lda = lda;
    t.ldb = ldb;
    t.ldc = ldc;
    t.ar = av.real();
    t.ai = av.imag();
    t.br = bv.real();
    t.bi = bv.imag();
    t.upper = uplo == ACLBLAS_UPPER;
    t.trans = trans != ACLBLAS_OP_N;
    t.scaleOnly = scaleOnly;
    t.betaZero = bv == std::complex<float>(0.0f, 0.0f);
    return t;
}

aclblasStatus_t DispatchCsyrkx(
    aclblasHandle_t handle, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c, int n, int k,
    bool scaleOnly, uint32_t vectors, const CsyrkxTilingData& t)
{
    // Unsafe scalar magnitudes require the ordered arithmetic: finite
    // reassociated results can conceal an intermediate overflow in Netlib.
    // With components bounded by 2^20, N terms are at most 2^62;
    // FP32 accumulation stagnates well before 2^88. T dot/alpha and beta*C
    // are bounded likewise, so reassociation cannot conceal overflow. The
    // split scan and pre-write C scan enforce the same bound on matrix data.
    constexpr float safeMagnitude = 1048576.0f; // 2^20; NaN also fails <=.
    const bool scalarRisk = !(std::fabs(t.ar) <= safeMagnitude) || !(std::fabs(t.ai) <= safeMagnitude) ||
                            !(std::fabs(t.br) <= safeMagnitude) || !(std::fabs(t.bi) <= safeMagnitude);
    const uint32_t blocks = std::min<uint64_t>(vectors, CeilDiv<uint64_t>(static_cast<uint64_t>(n) * n, THREADS));
    if (scaleOnly || (!scalarRisk && n <= 8 && k <= 32)) {
        csyrkx_direct_do(
            blocks, handle->stream, reinterpret_cast<const uint8_t*>(a), reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<uint8_t*>(c), t);
        return ACLBLAS_STATUS_SUCCESS;
    }
    // Small matrices benefit from a single ordered SIMD launch instead of
    // deinterleave + Cube + epilogue, without SIMT's scalar temporary traffic.
    // Large bounded products use Cube; their finite rounding need not match
    // serial FP32 bit-for-bit. Device input/C risk checks protect exceptions.
    if ((n <= 64 && k <= 64) || scalarRisk) {
        const uint64_t groups = CeilDiv<uint64_t>(n, 64);
        const uint32_t orderedBlocks = std::min<uint64_t>(vectors, groups * (groups + 1) / 2 * 16);
        csyrkx_ordered_do(
            orderedBlocks, handle->stream, reinterpret_cast<const uint8_t*>(a), reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<uint8_t*>(c), t);
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t cubes = GetAicCoreCount();
    if (cubes == 0) {
        OP_LOGE("aclblasCsyrkx", "no AIC cores available");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return LaunchProducts(handle, a, b, c, t, vectors, cubes);
}

} // namespace

aclblasStatus_t aclblasCsyrkx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const aclblasComplex* beta, aclblasComplex* c,
    int ldc)
{
    const auto validation = ValidateCsyrkxArguments(handle, uplo, trans, n, k, alpha, lda, ldb, beta, ldc);
    if (validation != ACLBLAS_STATUS_SUCCESS) {
        return validation;
    }
    const int rows = trans == ACLBLAS_OP_N ? n : k;
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const std::complex<float> av(alpha->real, alpha->imag);
    const std::complex<float> bv(beta->real, beta->imag);
    const bool scaleOnly = k == 0 || av == std::complex<float>(0.0f, 0.0f);
    if (c == nullptr || (!scaleOnly && (a == nullptr || b == nullptr))) {
        OP_LOGE("aclblasCsyrkx", "C and referenced A/B matrices must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (scaleOnly && bv == std::complex<float>(1.0f, 0.0f)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t inputCols = trans == ACLBLAS_OP_N ? k : n;
    if (!AddressableSpan(c, n, n, ldc) ||
        (!scaleOnly && (!AddressableSpan(a, rows, inputCols, lda) || !AddressableSpan(b, rows, inputCols, ldb)))) {
        OP_LOGE("aclblasCsyrkx", "matrix address span exceeds supported range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t vectors = GetAivCoreCount();
    if (vectors == 0) {
        OP_LOGE("aclblasCsyrkx", "no AIV cores available");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const CsyrkxTilingData t = BuildCsyrkxTiling(uplo, trans, n, k, lda, ldb, ldc, av, bv, scaleOnly);
    return DispatchCsyrkx(handle, a, b, c, n, k, scaleOnly, vectors, t);
}
#endif
