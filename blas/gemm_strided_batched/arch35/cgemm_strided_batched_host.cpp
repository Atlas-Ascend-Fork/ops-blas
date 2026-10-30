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
#include "cann_ops_blas.h"
#include "cgemm_strided_batched_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

namespace {
// Existing arch35 dispatch/packing cutoffs; these names do not retune them.
constexpr uint32_t LOCAL_PATH_MAX_DIM = 112;
constexpr uint32_t LOCAL_UNSLICED_MAX_DIM = 96;
constexpr uint32_t LOCAL_LARGE_SLICE_ROWS = 56;
constexpr uint32_t LOCAL_PADDING_MIN_DIM = 24;
constexpr uint32_t LOCAL_MERGE_MAX_DIM = 16;
constexpr uint32_t LOCAL_MERGE_MIN_BATCH = 128;
constexpr uint32_t LOCAL_SMALL_GROUP_MAX_DIM = 32;
constexpr uint32_t LOCAL_SINGLE_GROUP_MIN_DIM = 64;
constexpr uint32_t LOCAL_SLICE_MAX_BATCH = 4;
constexpr uint32_t LOCAL_INPUT_PAD_FLOATS = 4;
constexpr uint32_t LOCAL_MAX_BATCH_GROUP = 8U;
constexpr uint32_t LOCAL_MERGED_BATCH_GROUP = 4;
constexpr uint32_t LOCAL_SCRATCH_FLOATS = 256;
constexpr uint32_t LOCAL_WORKSPACE_BYTES_PER_AIV = 64;
constexpr uint32_t DEFAULT_CUBE_TILE_DIM = 128;
constexpr uint32_t WIDE_CUBE_TILE_DIM = 256;
constexpr uint32_t DEFAULT_CUBE_BASE_K = 64;
constexpr uint32_t WIDE_CUBE_BASE_K = 32;
constexpr uint32_t COMPENSATED_MIN_K = 64;
constexpr uint32_t DENSE_MERGE_MIN_BATCH = 64;
constexpr uint32_t DENSE_MERGED_BATCH_GROUP = 4;
constexpr uint32_t DENSE_MERGE_MAX_DIM = 32;
constexpr uint32_t DENSE_MERGE_K = 16;
constexpr uint32_t FUSED_MAX_DIM = 256;
constexpr uint32_t FUSED_MAX_K = 512;
constexpr uint32_t ORDERED_M_ALIGNMENT = 128;
constexpr uint32_t ORDERED_N_ALIGNMENT = 64;
constexpr uint32_t ORDERED_K_ALIGNMENT = 4;
constexpr uint32_t VECTOR_OUTPUTS_PER_BLOCK = 128;
constexpr uint32_t DIRECT_MAX_WORK = 65536;
constexpr uint32_t DIRECT_MAX_DIM = 32;
constexpr uint32_t DIRECT_THIN_MAX_DIM = 4;
constexpr uint32_t TINY_MAX_DIM = 8;

bool ValidTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

bool WorkspaceLayout(CgemmSbTiling& t, size_t& bytes)
{
    constexpr uint64_t ALIGN = 128;
    constexpr uint64_t LIMIT = (uint64_t{2} << 30) / sizeof(float);
    auto plane = [](uint64_t x, uint64_t y, uint64_t batch, uint64_t& result) {
        if (batch == 0 || x * y > LIMIT / batch) {
            return false;
        }
        result = ((x * y * batch + ALIGN - 1) / ALIGN) * ALIGN;
        return true;
    };
    if (!plane(t.mp, t.kp, t.batchesA, t.aSize) || !plane(uint64_t(t.np) * t.batchGroup, t.kp, t.batchesB, t.bSize) ||
        !plane(t.mp, t.np, t.batch, t.cSize)) {
        return false;
    }
    uint32_t inputPlanes =
        t.compensatedHalf ? CGEMM_SB_HALF_PLANES : (t.threeM ? CGEMM_SB_THREE_M_PLANES : CGEMM_SB_COMPONENT_PLANES);
    t.bOffset = inputPlanes * t.aSize;
    t.cOffset = t.bOffset + inputPlanes * t.bSize;
    t.flagOffset = t.cOffset + CGEMM_SB_GENERAL_PRODUCTS * t.cSize;
    uint64_t total = t.flagOffset + ((uint64_t(t.flagCores) * CGEMM_SB_FLAG_STRIDE + ALIGN - 1) / ALIGN) * ALIGN;
    if (t.normalizeOrdered) {
        uint64_t as = 0;
        uint64_t bs = 0;
        if (!plane(t.orderedM, t.orderedK, t.batchesA, as) || !plane(t.orderedK, t.orderedN, t.batchesB, bs)) {
            return false;
        }
        t.orderedAOffset = total;
        t.orderedBOffset = total + CGEMM_SB_COMPONENT_PLANES * as;
        total = t.orderedBOffset + CGEMM_SB_COMPONENT_PLANES * bs;
    }
    if (total > LIMIT) {
        return false;
    }
    bytes = total * sizeof(float);
    return true;
}

bool InvalidShape(int m, int n, int k, int batchCount) { return m < 0 || n < 0 || k < 0 || batchCount < 0; }

aclblasStatus_t ValidateLeadingDimensions(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, int lda, int ldb, int ldc)
{
    if (lda < std::max(1, transA == ACLBLAS_OP_N ? m : k)) {
        OP_LOGE(
            "ValidateLeadingDimensions", "invalid lda=%d, expected >= %d", lda,
            std::max(1, transA == ACLBLAS_OP_N ? m : k));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldb < std::max(1, transB == ACLBLAS_OP_N ? k : n)) {
        OP_LOGE(
            "ValidateLeadingDimensions", "invalid ldb=%d, expected >= %d", ldb,
            std::max(1, transB == ACLBLAS_OP_N ? k : n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("ValidateLeadingDimensions", "invalid ldc=%d, expected >= %d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// issue #683: negative strides would form out-of-bounds GM addresses (A + i*strideA with
// i > 0); the FP32 sibling validates them the same way before any launch.
aclblasStatus_t ValidateStrides(int64_t strideA, int64_t strideB, int64_t strideC)
{
    CHECK_RET(strideA >= 0, OP_LOGE("aclblasCgemmStridedBatched", "invalid strideA=%ld, must be >= 0", strideA);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(strideB >= 0, OP_LOGE("aclblasCgemmStridedBatched", "invalid strideB=%ld, must be >= 0", strideB);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(strideC >= 0, OP_LOGE("aclblasCgemmStridedBatched", "invalid strideC=%ld, must be >= 0", strideC);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateBasicArguments(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, const aclblasComplex* alpha, const aclblasComplex* beta)
{
    if (handle == nullptr) {
        OP_LOGE("ValidateBasicArguments", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!ValidTrans(transA) || !ValidTrans(transB)) {
        OP_LOGE(
            "ValidateBasicArguments",
            "transA and transB must be OP_N(%d), OP_T(%d), or OP_C(%d), got transA=%d, transB=%d",
            static_cast<int>(ACLBLAS_OP_N), static_cast<int>(ACLBLAS_OP_T), static_cast<int>(ACLBLAS_OP_C),
            static_cast<int>(transA), static_cast<int>(transB));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (InvalidShape(m, n, k, batchCount)) {
        OP_LOGE("ValidateBasicArguments", "invalid dimensions: m=%d, n=%d, k=%d, batchCount=%d", m, n, k, batchCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto status = ValidateLeadingDimensions(transA, transB, m, n, k, lda, ldb, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (alpha == nullptr) {
        OP_LOGE("ValidateBasicArguments", "alpha is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("ValidateBasicArguments", "beta is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateStorageArguments(
    int m, int n, int k, int batchCount, const aclblasComplex* A, const aclblasComplex* B, aclblasComplex* C)
{
    if (batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    uint64_t outputElements = uint64_t(m) * n;
    if (outputElements > std::numeric_limits<size_t>::max() / sizeof(aclblasComplex) / batchCount) {
        OP_LOGE(
            "ValidateStorageArguments", "output size overflows size_t: m=%d, n=%d, batchCount=%d", m, n, batchCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k > 0 && A == nullptr) {
        OP_LOGE("ValidateStorageArguments", "A is nullptr when k=%d", k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k > 0 && B == nullptr) {
        OP_LOGE("ValidateStorageArguments", "B is nullptr when k=%d", k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (C == nullptr) {
        OP_LOGE("ValidateStorageArguments", "C is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

CgemmSbTiling MakeTiling(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, int lda, int ldb, int ldc,
    int64_t strideA, int64_t strideB, int64_t strideC, const aclblasComplex& alpha, const aclblasComplex& beta,
    int batchCount)
{
    CgemmSbTiling t{};
    t.m = m;
    t.n = n;
    t.k = k;
    t.batch = batchCount;
    t.lda = lda;
    t.ldb = ldb;
    t.ldc = ldc;
    t.transA = static_cast<uint32_t>(transA);
    t.transB = static_cast<uint32_t>(transB);
    t.strideA = strideA;
    t.strideB = strideB;
    t.strideC = strideC;
    t.ar = alpha.real;
    t.ai = alpha.imag;
    t.br = beta.real;
    t.bi = beta.imag;
    return t;
}

bool MergeLocalBatches(const CgemmSbTiling& t)
{
    return t.m == t.mp && t.n == t.np && t.m <= LOCAL_MERGE_MAX_DIM && t.n <= LOCAL_MERGE_MAX_DIM &&
           t.k <= LOCAL_MERGE_MAX_DIM && t.ldc == t.m && t.strideC == int64_t(t.m) * t.n &&
           t.batch >= LOCAL_MERGE_MIN_BATCH;
}

bool PadLocalInput(const CgemmSbTiling& t)
{
    return t.mp >= LOCAL_PADDING_MIN_DIM || t.np >= LOCAL_PADDING_MIN_DIM || t.kp >= LOCAL_PADDING_MIN_DIM;
}

void ConfigureLocal(CgemmSbTiling& t, uint32_t aiv)
{
    if (aiv == 0) {
        aiv = 1;
    }
    t.mp = CeilAlign<uint32_t>(t.m, CGEMM_SB_FP32_PER_BLOCK);
    t.localSliceRows = CGEMM_SB_LOCAL_SLICE_ROWS;
    // Row slices bound UB and L0 operands when a full local matrix no longer fits.
    if (t.m > LOCAL_UNSLICED_MAX_DIM || t.n > LOCAL_UNSLICED_MAX_DIM || t.k > LOCAL_UNSLICED_MAX_DIM) {
        t.batchGroup = 1;
        t.localInputPad = LOCAL_INPUT_PAD_FLOATS;
        t.localSliceRows = LOCAL_LARGE_SLICE_ROWS;
        t.localSlices = (uint32_t(t.m) + t.localSliceRows - 1) / t.localSliceRows;
        return;
    }
    uint32_t as = CGEMM_SB_LOCAL_A_STORAGE_FLOATS * t.mp * t.kp;
    uint32_t bs = CGEMM_SB_COMPONENT_PLANES * t.np * t.kp;
    uint32_t maxIn = std::max(as / 2, bs);
    // Local A high/residual packing occupies twice its interleaved FP32 input size.
    uint32_t maxOut = std::max(as, bs);
    uint32_t capacity = std::min(
        (CGEMM_SB_UB_BUDGET_BYTES / CGEMM_SB_FP32_BYTES - LOCAL_SCRATCH_FLOATS) / (maxIn + maxOut),
        std::max(1U, CGEMM_SB_LOCAL_LARGE_SLOT_BYTES / (CGEMM_SB_FP32_BYTES * (as + bs))));
    t.batchGroup = std::min(std::min(capacity, LOCAL_MAX_BATCH_GROUP), (uint32_t(t.batch) + aiv - 1) / aiv);
    uint32_t smallGroup =
        t.mp <= LOCAL_SMALL_GROUP_MAX_DIM && t.np <= LOCAL_SMALL_GROUP_MAX_DIM && t.kp <= LOCAL_SMALL_GROUP_MAX_DIM ?
            2U :
            1U;
    t.batchGroup = std::min(
        t.batchGroup, t.m >= LOCAL_SINGLE_GROUP_MIN_DIM || t.n >= LOCAL_SINGLE_GROUP_MIN_DIM ? 1U : smallGroup);
    t.localBatchMerge = MergeLocalBatches(t);
    if (t.localBatchMerge) {
        t.batchGroup = LOCAL_MERGED_BATCH_GROUP;
    }
    t.localInputPad = PadLocalInput(t) ? LOCAL_INPUT_PAD_FLOATS : 0;
    t.localSlices =
        t.batch <= LOCAL_SLICE_MAX_BATCH && t.n >= LOCAL_SINGLE_GROUP_MIN_DIM && t.m >= LOCAL_SINGLE_GROUP_MIN_DIM ?
            (uint32_t(t.m) + CGEMM_SB_LOCAL_SLICE_ROWS - 1) / CGEMM_SB_LOCAL_SLICE_ROWS :
            0;
    if (t.localSlices) {
        t.localInputPad = LOCAL_INPUT_PAD_FLOATS;
    }
    uint32_t merged = t.localBatchMerge ? t.batchGroup : 1;
    t.baseK = std::min(
        2 * t.kp * merged, CGEMM_SB_L0_HALF_BANK_ELEMENTS / std::max(t.np * merged, 2 * t.mp) / CGEMM_SB_FRACTAL_EDGE *
                               CGEMM_SB_FRACTAL_EDGE);
}

aclblasStatus_t LaunchLocal(
    aclblasHandle_t handle, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, CgemmSbTiling t, uint32_t aiv)
{
    ConfigureLocal(t, aiv);
    const size_t workspaceBytes = static_cast<size_t>(aiv) * LOCAL_WORKSPACE_BYTES_PER_AIV;
    auto status = EnsureDefaultWorkspace(handle, workspaceBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCgemmStridedBatched", "workspace ensure failed, required=%zu, ret=%d", workspaceBytes,
            static_cast<int>(status));
        return status;
    }
    uint32_t blocks = GetAicCoreCount();
    if (blocks == 0) {
        OP_LOGE("aclblasCgemmStridedBatched", "GetAicCoreCount failed (returned 0) for local path");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (t.localSlices) {
        blocks = static_cast<uint32_t>(std::min<uint64_t>(blocks, uint64_t(t.batch) * t.localSlices));
    }
    cgemm_sb_local_do(
        blocks, handle->stream, gmA, gmB, gmC, reinterpret_cast<GM_ADDR>(GetEffectiveWorkspace(handle)), t);
    return ACLBLAS_STATUS_SUCCESS;
}

void ConfigureDenseLayouts(CgemmSbTiling& t, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC)
{
    t.densePack = t.m == t.mp && t.n == t.np && t.k == t.kp && t.lda == (t.transA == ACLBLAS_OP_N ? t.m : t.k) &&
                  t.ldb == (t.transB == ACLBLAS_OP_N ? t.k : t.n) &&
                  reinterpret_cast<uintptr_t>(gmA) % CGEMM_SB_DMA_BLOCK_BYTES == 0 &&
                  reinterpret_cast<uintptr_t>(gmB) % CGEMM_SB_DMA_BLOCK_BYTES == 0 &&
                  t.strideA % CGEMM_SB_COMPLEX_PER_BLOCK == 0 && t.strideB % CGEMM_SB_COMPLEX_PER_BLOCK == 0;
    t.denseCombine = t.m == t.mp && t.n == t.np && t.ldc == t.m &&
                     reinterpret_cast<uintptr_t>(gmC) % CGEMM_SB_DMA_BLOCK_BYTES == 0 &&
                     t.strideC % CGEMM_SB_COMPLEX_PER_BLOCK == 0;
}

uint32_t CompensatedBaseK(const CgemmSbTiling& t)
{
    if (t.mp <= DEFAULT_CUBE_TILE_DIM && t.np <= DEFAULT_CUBE_TILE_DIM && t.kp <= DEFAULT_CUBE_TILE_DIM) {
        return t.kp;
    }
    return t.tileM <= DEFAULT_CUBE_TILE_DIM && t.tileN <= DEFAULT_CUBE_TILE_DIM ? DEFAULT_CUBE_TILE_DIM :
                                                                                  DEFAULT_CUBE_BASE_K;
}

void ConfigureComplexTiling(CgemmSbTiling& t)
{
    t.threeM = t.ar == 1 && t.ai == 0 && t.br == 0 && t.bi == 0;

    // Split each FP32 operand into FP16 high and residual parts for compensated Cube multiplication.
    t.compensatedHalf = t.threeM && t.k >= COMPENSATED_MIN_K;
    if (t.compensatedHalf) {
        t.baseK = CompensatedBaseK(t);
    }
    t.batchGroup = 1;
    if (t.threeM && t.densePack && t.transA == ACLBLAS_OP_N && t.transB == ACLBLAS_OP_N && t.m <= DENSE_MERGE_MAX_DIM &&
        t.n <= DENSE_MERGE_MAX_DIM && t.k == DENSE_MERGE_K && t.strideA != 0 && t.strideB != 0 &&
        t.batch >= DENSE_MERGE_MIN_BATCH) {
        t.batchGroup = DENSE_MERGED_BATCH_GROUP;
        if (t.batch % t.batchGroup != 0) {
            t.batchGroup = 1;
        }
    }
}

void ConfigureOrderedTiling(CgemmSbTiling& t)
{
    t.orderedM = CeilAlign<uint32_t>(t.m, ORDERED_M_ALIGNMENT);
    t.orderedN = CeilAlign<uint32_t>(t.n, ORDERED_N_ALIGNMENT);
    t.orderedK = CeilAlign<uint32_t>(t.k, ORDERED_K_ALIGNMENT);
    t.normalizeOrdered = t.threeM && t.k >= CGEMM_SB_ORDERED_MIN_K &&
                         !(t.transA == ACLBLAS_OP_N && t.transB == ACLBLAS_OP_N && t.m % ORDERED_M_ALIGNMENT == 0 &&
                           t.n % ORDERED_N_ALIGNMENT == 0 && t.k % ORDERED_K_ALIGNMENT == 0);
}

void ConfigureGlobal(CgemmSbTiling& t, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC)
{
    t.batchesA = t.strideA == 0 ? 1 : t.batch;
    t.batchesB = t.strideB == 0 ? 1 : t.batch;
    t.tileM = DEFAULT_CUBE_TILE_DIM;
    t.tileN = DEFAULT_CUBE_TILE_DIM;
    t.baseK = DEFAULT_CUBE_BASE_K;
    if (t.mp <= WIDE_CUBE_TILE_DIM && t.np <= WIDE_CUBE_TILE_DIM &&
        (t.mp > DEFAULT_CUBE_TILE_DIM || t.np > DEFAULT_CUBE_TILE_DIM)) {
        t.tileM = t.np;
        t.tileN = t.mp;
        t.baseK = WIDE_CUBE_BASE_K;
    }
    ConfigureDenseLayouts(t, gmA, gmB, gmC);
    ConfigureComplexTiling(t);
    ConfigureOrderedTiling(t);
}

aclblasStatus_t LaunchGlobal(
    aclblasHandle_t handle, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, CgemmSbTiling t, uint32_t aiv, uint32_t vectorBlocks)
{
    ConfigureGlobal(t, gmA, gmB, gmC);
    t.flagCores = aiv;
    size_t workspaceBytes = 0;
    if (!WorkspaceLayout(t, workspaceBytes)) {
        // Padding very thin, long-K batches can make packed storage exceed the
        // workspace budget. The direct path needs no temporary allocation.
        cgemm_sb_vector_do(vectorBlocks, handle->stream, gmA, gmB, gmC, nullptr, t, CGEMM_SB_MODE_DIRECT);
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto status = EnsureDefaultWorkspace(handle, workspaceBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCgemmStridedBatched", "workspace ensure failed, required=%zu, ret=%d", workspaceBytes,
            static_cast<int>(status));
        return status;
    }
    auto workspace = reinterpret_cast<GM_ADDR>(GetEffectiveWorkspace(handle));
    uint32_t aic = GetAicCoreCount();
    if (aic == 0) {
        OP_LOGE("aclblasCgemmStridedBatched", "GetAicCoreCount failed (returned 0) for global path");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (t.m <= FUSED_MAX_DIM && t.n <= FUSED_MAX_DIM && t.k <= FUSED_MAX_K) {
        cgemm_sb_fused_do(aic, handle->stream, gmA, gmB, gmC, workspace, t);
        return ACLBLAS_STATUS_SUCCESS;
    }
    cgemm_sb_vector_do(aiv, handle->stream, gmA, gmB, gmC, workspace, t, CGEMM_SB_MODE_PACK);
    if (t.threeM && t.k >= CGEMM_SB_ORDERED_MIN_K) {
        if (t.normalizeOrdered) {
            cgemm_sb_normalize_do(aiv, handle->stream, gmA, gmB, workspace, t);
        }
        cgemm_sb_hybrid_do(aic, handle->stream, gmA, gmB, gmC, workspace, t);
    } else {
        cgemm_sb_cube_do(aic, handle->stream, workspace, t);
    }
    cgemm_sb_vector_do(vectorBlocks, handle->stream, gmA, gmB, gmC, workspace, t, CGEMM_SB_MODE_COMBINE);

    return ACLBLAS_STATUS_SUCCESS;
}

bool TinyMatrix(const CgemmSbTiling& t) { return t.m <= TINY_MAX_DIM && t.n <= TINY_MAX_DIM && t.k <= TINY_MAX_DIM; }

bool SmallDirectMatrix(const CgemmSbTiling& t, uint64_t elements)
{
    // Called only after excluding k=0, so the reduction extent is nonzero.
    return (t.m <= DIRECT_MAX_DIM && t.n <= DIRECT_MAX_DIM && elements <= DIRECT_MAX_WORK / uint64_t(t.k)) ||
           (t.m <= DIRECT_THIN_MAX_DIM && t.n <= DIRECT_THIN_MAX_DIM);
}

aclblasStatus_t LaunchOperation(aclblasHandle_t handle, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, CgemmSbTiling t)
{
    bool scaleOnly = t.k == 0 || (t.ar == 0 && t.ai == 0);
    if (scaleOnly && t.br == 1 && t.bi == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    uint32_t aiv = GetAivCoreCount();
    if (aiv == 0) {
        OP_LOGE("aclblasCgemmStridedBatched", "GetAivCoreCount failed (returned 0)");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint64_t elements = uint64_t(t.m) * t.n * t.batch;
    uint32_t vectorBlocks = static_cast<uint32_t>(
        std::min<uint64_t>(aiv, (elements + VECTOR_OUTPUTS_PER_BLOCK - 1) / VECTOR_OUTPUTS_PER_BLOCK));
    bool generalScalars = t.ar != 1 || t.ai != 0 || t.br != 0 || t.bi != 0;
    if (!scaleOnly && !generalScalars && TinyMatrix(t)) {
        cgemm_sb_vector_do(aiv, handle->stream, gmA, gmB, gmC, nullptr, t, CGEMM_SB_MODE_TINY);
        return ACLBLAS_STATUS_SUCCESS;
    }
    // General scalars retain Netlib's scaling and accumulation order.
    bool direct = scaleOnly || generalScalars || SmallDirectMatrix(t, elements);
    if (direct) {
        cgemm_sb_vector_do(
            vectorBlocks, handle->stream, gmA, gmB, gmC, nullptr, t,
            scaleOnly ? CGEMM_SB_MODE_SCALE : CGEMM_SB_MODE_DIRECT);
        return ACLBLAS_STATUS_SUCCESS;
    }
    t.mp = CeilAlign<uint32_t>(t.m, CGEMM_SB_FRACTAL_EDGE);
    t.np = CeilAlign<uint32_t>(t.n, CGEMM_SB_FRACTAL_EDGE);
    t.kp = CeilAlign<uint32_t>(t.k, CGEMM_SB_FRACTAL_EDGE);
    if (t.m <= LOCAL_PATH_MAX_DIM && t.n <= LOCAL_PATH_MAX_DIM && t.k <= LOCAL_PATH_MAX_DIM) {
        return LaunchLocal(handle, gmA, gmB, gmC, t, aiv);
    }
    return LaunchGlobal(handle, gmA, gmB, gmC, t, aiv, vectorBlocks);
}
} // namespace

aclblasStatus_t aclblasCgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, int64_t strideA, const aclblasComplex* B, int ldb,
    int64_t strideB, const aclblasComplex* beta, aclblasComplex* C, int ldc, int64_t strideC, int batchCount)
{
    auto status = ValidateBasicArguments(handle, transA, transB, m, n, k, lda, ldb, ldc, batchCount, alpha, beta);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateStrides(strideA, strideB, strideC);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    status = ValidateStorageArguments(m, n, k, batchCount, A, B, C);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    auto t = MakeTiling(transA, transB, m, n, k, lda, ldb, ldc, strideA, strideB, strideC, *alpha, *beta, batchCount);
    auto gmA = reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A));
    auto gmB = reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B));
    auto gmC = reinterpret_cast<GM_ADDR>(C);
    return LaunchOperation(handle, gmA, gmB, gmC, t);
}
