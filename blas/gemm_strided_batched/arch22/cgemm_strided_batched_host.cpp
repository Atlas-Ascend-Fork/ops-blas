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
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cgemm_strided_batched_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

namespace {
constexpr const char* kName = "aclblasCgemmStridedBatched";

bool ValidTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

bool IsZero(const aclblasComplex& value) { return value.real == 0.0f && value.imag == 0.0f; }
constexpr size_t kBatchWorkspaceBudget = 128U * 1024U * 1024U;

bool ValidDimensions(
    aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb, int ldc)
{
    return lda >= std::max(1, transa == ACLBLAS_OP_N ? m : k) && ldb >= std::max(1, transb == ACLBLAS_OP_N ? k : n) &&
           ldc >= std::max(1, m);
}

bool UseCube(const CgemmSbTiling& t, bool small, bool alphaZero)
{
    // Called only after the zero-dimension quick return.
    const bool stridesFit = t.paddedM <= 65535U && t.paddedN <= 65535U && t.paddedK <= 65535U;
    // Cube packing uses 32-bit byte strides. Large leading dimensions must use
    // the direct path, which forms each GM address in 64 bits and splits DMA.
    constexpr uint32_t maxDmaLeadingDimension = std::numeric_limits<uint32_t>::max() / sizeof(aclblasComplex);
    const bool dmaStridesFit =
        t.lda <= maxDmaLeadingDimension && t.ldb <= maxDmaLeadingDimension && t.ldc <= maxDmaLeadingDimension;
    return !small && !alphaZero && stridesFit && dmaStridesFit &&
           static_cast<uint64_t>(t.m) * t.n >= (512ULL + t.k - 1U) / t.k && t.paddedM >= t.m && t.paddedN >= t.n &&
           t.paddedK >= t.k;
}

CgemmSbTiling MakeTiling(
    aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const aclblasComplex* alpha, int lda,
    int64_t strideA, int ldb, int64_t strideB, const aclblasComplex* beta, int ldc, int64_t strideC)
{
    CgemmSbTiling tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.paddedM = (tiling.m + 63U) / 64U * 64U;
    tiling.paddedN = (tiling.n + 63U) / 64U * 64U;
    tiling.paddedK = (tiling.k + 63U) / 64U * 64U;
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.transA = static_cast<uint32_t>(transa);
    tiling.transB = static_cast<uint32_t>(transb);
    tiling.alphaReal = alpha->real;
    tiling.alphaImag = alpha->imag;
    tiling.betaReal = beta->real;
    tiling.betaImag = beta->imag;
    tiling.batchCount = 1U;
    tiling.strideA = strideA;
    tiling.strideB = strideB;
    tiling.strideC = strideC;

    if (m <= 32 && n <= 32 && k <= 32) {
        tiling.paddedM = 32U;
        tiling.paddedN = 32U;
        tiling.paddedK = 32U;
    }
    return tiling;
}

bool IndependentOutputBatches(const CgemmSbTiling& t) { return t.strideC >= static_cast<int64_t>(t.ldc) * t.n; }

aclblasStatus_t PrepareWorkspace(_aclblas_handle* h, CgemmSbTiling& tiling, int batchCount)
{
    const uint64_t limit = ACLBLAS_MAX_WORKSPACE_SIZE / sizeof(float);
    const uint64_t mk = static_cast<uint64_t>(tiling.paddedM) * tiling.paddedK;
    const uint64_t kn = static_cast<uint64_t>(tiling.paddedK) * tiling.paddedN;
    const uint64_t mn = static_cast<uint64_t>(tiling.paddedM) * tiling.paddedN;
    if (mk > limit / 2U || kn > limit / 2U || mn > limit / 4U || 2U * mk + 2U * kn > limit - 4U * mn) {
        OP_LOGE(kName, "workspace exceeds handle limit");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const uint64_t floats = CgemmSbWorkspaceFloats(tiling.paddedM, tiling.paddedN, tiling.paddedK);
    tiling.workspaceStride = floats;
    if (floats > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    size_t requested = static_cast<size_t>(floats * sizeof(float));
    if (h->workspace_owner == AclblasWorkspaceOwner::Library && IndependentOutputBatches(tiling)) {
        const size_t slots = std::max<size_t>(1U, std::min<size_t>(batchCount, kBatchWorkspaceBudget / requested));
        requested *= slots;
    }
    return EnsureDefaultWorkspace(h, requested);
}

uint32_t GroupSize(_aclblas_handle* h, const CgemmSbTiling& tiling, int batchCount)
{
    const bool dense = tiling.ldc == tiling.m && tiling.strideC == static_cast<int64_t>(tiling.m) * tiling.n;
    const bool square = tiling.m == tiling.n && tiling.n == tiling.k && tiling.m <= 192U;
    const bool needsAlignedGroup = !dense || square;
    const uint64_t budget = std::min<size_t>(GetEffectiveWorkspaceSize(h), kBatchWorkspaceBudget);
    uint32_t groupSize =
        static_cast<uint32_t>(std::min<uint64_t>(batchCount, budget / (tiling.workspaceStride * sizeof(float))));
    groupSize = std::max(1U, groupSize);
    while (groupSize < static_cast<uint32_t>(batchCount) && needsAlignedGroup && groupSize > 1U &&
           (static_cast<int64_t>(groupSize) * tiling.strideC) % 8 != 0) {
        --groupSize;
    }
    return groupSize;
}

// README: with k == 0 the multiply is skipped and each batch applies C_i = beta * C_i.
// The direct vector kernel already implements this: it skips the k loop and stores
// alpha * 0 + beta * C. Launching it here avoids the cube-path k==0 tiling corner.
aclblasStatus_t ScaleOnlyBatches(
    _aclblas_handle* h, CgemmSbTiling tiling, int batchCount, const aclblasComplex* A, const aclblasComplex* B,
    aclblasComplex* C)
{
    const uint32_t aivCores = GetAivCoreCount();
    if (aivCores == 0) {
        OP_LOGE(kName, "core count query failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    tiling.batchCount = static_cast<uint32_t>(batchCount);
    const bool independent = tiling.strideC >= static_cast<int64_t>(tiling.ldc) * tiling.n;
    const bool parallelC = tiling.m % 8 == 0 && tiling.ldc % 8 == 0 && tiling.strideC % 8 == 0 &&
                           reinterpret_cast<uintptr_t>(C) % 64U == 0U;
    const uint32_t blocks = parallelC && independent ? aivCores : 1U;
    cgemm_sb_direct_batched_launch(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)), reinterpret_cast<GM_ADDR>(C), tiling, blocks,
        h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ExecuteCgemm(
    _aclblas_handle* h, CgemmSbTiling tiling, int batchCount, const aclblasComplex* A, const aclblasComplex* B,
    aclblasComplex* C)
{
    const uint32_t aivCores = GetAivCoreCount();
    const uint32_t aicCores = GetAicCoreCount();
    if (aivCores == 0 || aicCores == 0) {
        OP_LOGE(kName, "core count query failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const bool alphaZero = tiling.alphaReal == 0.0f && tiling.alphaImag == 0.0f;
    const bool small = !(tiling.m == 16U && batchCount >= 16) && CgemmSbSmallShape(tiling) &&
                       reinterpret_cast<uintptr_t>(C) % 64U == 0U;
    const bool useCube = UseCube(tiling, small, alphaZero);
    void* workspace = nullptr;
    if (useCube) {
        const aclblasStatus_t status = PrepareWorkspace(h, tiling, batchCount);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        workspace = GetEffectiveWorkspace(h);
    }
    // Kernels iterate over batches; Cube stages partition workspace within each group.
    // Library storage uses a 128 MiB grouping budget; user storage is not enlarged.
    const bool parallelC = tiling.m % 8 == 0 && tiling.ldc % 8 == 0 && tiling.strideC % 8 == 0 &&
                           reinterpret_cast<uintptr_t>(C) % 64U == 0U;
    uint32_t groupSize = useCube ? 1U : static_cast<uint32_t>(batchCount);
    if (useCube && IndependentOutputBatches(tiling)) {
        groupSize = GroupSize(h, tiling, batchCount);
    }
    // Only workspace-limited Cube groups or overlapping Cube outputs need repeated
    // launches. Stream ordering protects workspace reuse and sequential C updates.
    for (int i = 0; i < batchCount; i += static_cast<int>(tiling.batchCount)) {
        tiling.batchCount = std::min(groupSize, static_cast<uint32_t>(batchCount - i));
        auto* ai = alphaZero ? nullptr : const_cast<aclblasComplex*>(A + static_cast<int64_t>(i) * tiling.strideA);
        auto* bi = alphaZero ? nullptr : const_cast<aclblasComplex*>(B + static_cast<int64_t>(i) * tiling.strideB);
        auto* ci = C + static_cast<int64_t>(i) * tiling.strideC;
        cgemm_sb_launch(
            reinterpret_cast<GM_ADDR>(ai), reinterpret_cast<GM_ADDR>(bi), reinterpret_cast<GM_ADDR>(ci),
            reinterpret_cast<GM_ADDR>(workspace), tiling, aivCores, aicCores, useCube, parallelC, h->stream);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Shape/stride sign check shared by the entry validator (issue #683).
bool ValidShape(int m, int n, int k, int batchCount, int64_t strideA, int64_t strideB, int64_t strideC)
{
    return m >= 0 && n >= 0 && k >= 0 && batchCount >= 0 && strideA >= 0 && strideB >= 0 && strideC >= 0;
}

aclblasStatus_t ValidateEntry(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, int64_t strideA, int64_t strideB, int64_t strideC, int batchCount, const aclblasComplex* alpha,
    const aclblasComplex* beta)
{
    if (handle == nullptr) {
        OP_LOGE(kName, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!ValidTrans(transa) || !ValidTrans(transb) || !ValidShape(m, n, k, batchCount, strideA, strideB, strideC)) {
        OP_LOGE(kName, "invalid operation, dimension, batch count, stride or scalar pointer");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE(kName, "scalar pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!ValidDimensions(transa, transb, m, n, k, lda, ldb, ldc)) {
        OP_LOGE(kName, "invalid leading dimension");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

aclblasStatus_t aclblasCgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, int64_t strideA, const aclblasComplex* B, int ldb,
    int64_t strideB, const aclblasComplex* beta, aclblasComplex* C, int ldc, int64_t strideC, int batchCount)
{
    const aclblasStatus_t entry = ValidateEntry(
        handle, transa, transb, m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchCount, alpha, beta);
    if (entry != ACLBLAS_STATUS_SUCCESS) {
        return entry;
    }
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (k == 0) {
        // A/B are never read when k == 0, matching the arch35 entry contract.
        if (beta->real == 1.0f && beta->imag == 0.0f) {
            return ACLBLAS_STATUS_SUCCESS;
        }
        if (C == nullptr) {
            OP_LOGE(kName, "C is nullptr");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        auto* h = reinterpret_cast<_aclblas_handle*>(handle);
        CgemmSbTiling tiling =
            MakeTiling(transa, transb, m, n, k, alpha, lda, strideA, ldb, strideB, beta, ldc, strideC);
        return ScaleOnlyBatches(h, tiling, batchCount, A, B, C);
    }
    if (C == nullptr || (!IsZero(*alpha) && (A == nullptr || B == nullptr))) {
        OP_LOGE(kName, "required matrix pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    CgemmSbTiling tiling{};
    tiling = MakeTiling(transa, transb, m, n, k, alpha, lda, strideA, ldb, strideB, beta, ldc, strideC);
    return ExecuteCgemm(h, tiling, batchCount, A, B, C);
}
