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
 * \file gemm_batched_host.cpp
 * \brief Batched GEMM host implementation for arch35 (DAV_3510).
 *
 * C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
 *
 * Uses tensor_api for both cube (AIC) and vector (AIV) kernels.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/host_utils.h"
#include "common/helper/aclblas_handle_internal.h"
#include "gemm_batched_tiling_data.h"
#include "gemm_batched_host_common.h"

constexpr const char* SG_OP_NAME = "aclblasSgemmBatched";

// ============================================================================
// Workspace management
// ============================================================================
static aclblasStatus_t EnsureWorkspace(_aclblas_handle* h, size_t requiredSize)
{
    return GemmBatchedEnsureWorkspace(h, requiredSize);
}

// ============================================================================
// Validation
// ============================================================================
static aclblasStatus_t ValidateSgemmBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, const float* alpha, const float* beta, const float* const Aarray[],
    const float* const Barray[], float* const Carray[])
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSgemmBatched", "handle is nullptr");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSgemmBatched", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasSgemmBatched", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasSgemmBatched", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSgemmBatched", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasSgemmBatched", "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(batchCount >= 0, OP_LOGE("aclblasSgemmBatched", "batchCount must be >= 0, got %d", batchCount);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C,
              OP_LOGE("aclblasSgemmBatched", "invalid transa=%d", static_cast<int>(transa));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transb == ACLBLAS_OP_N || transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C,
              OP_LOGE("aclblasSgemmBatched", "invalid transb=%d", static_cast<int>(transb));
              return ACLBLAS_STATUS_INVALID_VALUE);

    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int physRowsA = isTransA ? k : m;
    int physRowsB = isTransB ? n : k;
    CHECK_RET(lda >= std::max(1, physRowsA),
              OP_LOGE("aclblasSgemmBatched", "invalid lda=%d, expected>=%d", lda, std::max(1, physRowsA));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= std::max(1, physRowsB),
              OP_LOGE("aclblasSgemmBatched", "invalid ldb=%d, expected>=%d", ldb, std::max(1, physRowsB));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, m),
              OP_LOGE("aclblasSgemmBatched", "invalid ldc=%d, expected>=%d", ldc, std::max(1, m));
              return ACLBLAS_STATUS_INVALID_VALUE);

    if (batchCount > 0) {
        CHECK_RET(Aarray != nullptr, OP_LOGE("aclblasSgemmBatched", "Aarray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Barray != nullptr, OP_LOGE("aclblasSgemmBatched", "Barray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Carray != nullptr, OP_LOGE("aclblasSgemmBatched", "Carray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Minimum elements per AIV core to amortize launch overhead for post-process kernels.
// ============================================================================
// Tiling computation (delegates to shared GEMM-batched helpers)
// ============================================================================
static aclblasStatus_t CalcGemmTiling(
    int m, int n, int k, int batchCount, int lda, int ldb, int ldc, bool isTransA, bool isTransB, uint32_t cubeCoreNum,
    GemmBatchedGemmTilingData& tiling)
{
    // Keep the original minimal-divisible-wave search for the FLOAT32 path.
    constexpr bool useTrailingWaveBalance = false;
    return GemmBatchedCalcGemmTiling(
        SG_OP_NAME, GEMM_BATCHED_DTYPE_FP32, useTrailingWaveBalance, m, n, k, batchCount, lda, ldb, ldc,
        isTransA, isTransB, cubeCoreNum, tiling);
}

static void ApplyColumnMajorSwap(GemmBatchedGemmTilingData& tiling)
{
    GemmBatchedApplyColumnMajorSwap(tiling);
}

// ============================================================================
// Workspace helpers
// ============================================================================
static aclblasStatus_t PrepareTempWorkspace(
    _aclblas_handle* h, int batchCount, int origN, uint32_t tempRowStride, GemmBatchedGemmTilingData& tilingData,
    uint8_t*& workspace, size_t& alignedPtrArrayBytes)
{
    return GemmBatchedPrepareTempWorkspace(
        SG_OP_NAME, h, batchCount, origN, tempRowStride, tilingData, workspace, alignedPtrArrayBytes);
}

static void LaunchAlphaBetaPostProcess(
    _aclblas_handle* h, int origM, int origN, int origLdc, uint32_t tempRowStride, float alphaVal, float betaVal,
    int batchCount, uint32_t aivCoreNum, uint8_t* workspace, size_t alignedPtrArrayBytes, float* const Carray[])
{
    uint8_t* carrayPtrAB = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));
    GemmBatchedLaunchAlphaBetaPostProcess(
        SG_OP_NAME, GEMM_BATCHED_DTYPE_FP32, h, origM, origN, origLdc, tempRowStride, alphaVal, betaVal, batchCount,
        aivCoreNum, workspace, alignedPtrArrayBytes, carrayPtrAB);
}

// ============================================================================
// Main execution
// ============================================================================
static aclblasStatus_t ValidateCoreCounts(uint32_t& cubeCoreNum, uint32_t& aivCoreNum)
{
    return GemmBatchedValidateCoreCounts(SG_OP_NAME, cubeCoreNum, aivCoreNum);
}

static aclblasStatus_t ExecuteSgemmBatched(
    _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const float* alpha,
    const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta,
    float* const Carray[], int ldc, int batchCount)
{
    float alphaVal = *alpha;
    float betaVal = *beta;
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    bool needPostProcess = (alphaVal != 1.0f) || (betaVal != 0.0f);

    uint32_t cubeCoreNum = 0;
    uint32_t aivCoreNum = 0;
    aclblasStatus_t coreRet = ValidateCoreCounts(cubeCoreNum, aivCoreNum);
    if (coreRet != ACLBLAS_STATUS_SUCCESS) {
        return coreRet;
    }

    GemmBatchedGemmTilingData tilingData;
    aclblasStatus_t tilingRet =
        CalcGemmTiling(m, n, k, batchCount, lda, ldb, ldc, isTransA, isTransB, cubeCoreNum, tilingData);
    if (tilingRet != ACLBLAS_STATUS_SUCCESS) {
        return tilingRet;
    }

    int origM = m, origN = n, origLdc = ldc;
    uint32_t tempRowStride = CeilAlign(static_cast<uint32_t>(origM), GEMM_BATCHED_L0C_C0);

    uint8_t* workspace = nullptr;
    size_t alignedPtrArrayBytes = 0;

    if (needPostProcess) {
        aclblasStatus_t wsRet =
            PrepareTempWorkspace(h, batchCount, origN, tempRowStride, tilingData, workspace, alignedPtrArrayBytes);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            return wsRet;
        }
    }

    ApplyColumnMajorSwap(tilingData);

    OP_LOGI(
        "aclblasSgemmBatched", "launching gemm kernel: cores=%u, transA=%d, transB=%d, batch=%d, m=%u, n=%u, k=%u",
        tilingData.usedAicCoreNum, isTransA, isTransB, batchCount, tilingData.m, tilingData.n, tilingData.k);

    const uint8_t* kernelAarray = reinterpret_cast<const uint8_t*>(Barray);
    const uint8_t* kernelBarray = reinterpret_cast<const uint8_t*>(Aarray);
    uint8_t* carrayPtr = needPostProcess ? workspace : const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));

    gemm_batched_gemm_kernel_do(
        tilingData.usedAicCoreNum, h->stream, kernelAarray, kernelBarray, carrayPtr, tilingData);

    if (needPostProcess) {
        LaunchAlphaBetaPostProcess(
            h, origM, origN, origLdc, tempRowStride, alphaVal, betaVal, batchCount, aivCoreNum, workspace,
            alignedPtrArrayBytes, Carray);
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Early exit: k=0 or alpha=0
// ============================================================================
static aclblasStatus_t LaunchEarlyExit(
    _aclblas_handle* h, int m, int n, int ldc, int batchCount, float betaVal, float* const Carray[])
{
    uint8_t* carrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));
    return GemmBatchedLaunchEarlyExit(
        SG_OP_NAME, GEMM_BATCHED_DTYPE_FP32, h, m, n, ldc, batchCount, betaVal, carrayPtr);
}

// ============================================================================
// Public API
// ============================================================================
aclblasStatus_t aclblasSgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta,
    float* const Carray[], int ldc, int batchCount)
{
    OP_LOGI(
        "aclblasSgemmBatched", "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d, batch=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k, batchCount);

    aclblasStatus_t st = ValidateSgemmBatchedParams(
        handle, transa, transb, m, n, k, lda, ldb, ldc, batchCount, alpha, beta, Aarray, Barray, Carray);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    float alphaVal = *alpha;
    float betaVal = *beta;

    // Early exit: k=0 or alpha=0
    if (k == 0 || alphaVal == 0.0f) {
        return LaunchEarlyExit(h, m, n, ldc, batchCount, betaVal, Carray);
    }

    return ExecuteSgemmBatched(
        h, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

// ============================================================================
// Complex GEMM Batched (aclblasCgemmBatched)
//
// 4M decomposition:
//   A = Ar + j*Ai, B = Br + j*Bi
//   op(A)*op(B) = (Ar*Br - Ai*Bi) + j*(Ar*Bi + Ai*Br)
//
// Steps:
//   1. Deinterleave A → (Ar, Ai), B → (Br, Bi)  [AIV kernel]
//   2. Four real GEMM (alpha=1, beta=0, reuse fp32 cube kernel):
//        T1 = Ar*Br, T2 = Ai*Bi, T3 = Ar*Bi, T4 = Ai*Br
//   3. Combine [AIV kernel]:
//        P_r = T1 - T2, P_i = T3 + T4
//        C = alpha*(P_r + j*P_i) + beta*C_orig
// ============================================================================

// Forward declarations for complex support kernels (defined in gemm_batched_kernel.cpp)
void cgemm_batched_deinterleave_do(
    uint32_t numBlocks, void* stream, uint8_t* srcArray, uint8_t* realArray, uint8_t* imagArray,
    const CgemmBatchedDeinterleaveTilingData& tiling);

void cgemm_batched_combine_do(
    uint32_t numBlocks, void* stream, uint8_t* t1Array, uint8_t* t2Array, uint8_t* t3Array, uint8_t* t4Array,
    uint8_t* carray, const CgemmBatchedCombineTilingData& tiling);

void cgemm_batched_direct_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* carray,
    const CgemmBatchedDirectTilingData& tiling);

void cgemm_batched_realify_a_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* packedArray,
    const CgemmBatchedRealifyATilingData& tiling);

void cgemm_batched_split_k_pointer_do(
    void* stream, uint8_t* originalAArray, uint8_t* packedBArray, uint8_t* workspace,
    const CgemmBatchedSplitKPointerTilingData& tiling);

void cgemm_batched_realify_generic_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& tiling);

void cgemm_batched_realify_transposed_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& tiling);

void cgemm_batched_realify_mixed_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* carray, uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& packTiling, const GemmBatchedGemmTilingData& gemmTiling);

static aclblasStatus_t ValidateCgemmBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, const aclblasComplex* alpha, const aclblasComplex* beta,
    const aclblasComplex* const Aarray[], const aclblasComplex* const Barray[], aclblasComplex* const Carray[])
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCgemmBatched", "handle is nullptr");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasCgemmBatched", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasCgemmBatched", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasCgemmBatched", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasCgemmBatched", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasCgemmBatched", "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(batchCount >= 0, OP_LOGE("aclblasCgemmBatched", "batchCount must be >= 0, got %d", batchCount);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C,
              OP_LOGE("aclblasCgemmBatched", "invalid transa=%d", static_cast<int>(transa));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transb == ACLBLAS_OP_N || transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C,
              OP_LOGE("aclblasCgemmBatched", "invalid transb=%d", static_cast<int>(transb));
              return ACLBLAS_STATUS_INVALID_VALUE);

    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int physRowsA = isTransA ? k : m;
    int physRowsB = isTransB ? n : k;
    CHECK_RET(lda >= std::max(1, physRowsA),
              OP_LOGE("aclblasCgemmBatched", "invalid lda=%d, expected>=%d", lda, std::max(1, physRowsA));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= std::max(1, physRowsB),
              OP_LOGE("aclblasCgemmBatched", "invalid ldb=%d, expected>=%d", ldb, std::max(1, physRowsB));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, m),
              OP_LOGE("aclblasCgemmBatched", "invalid ldc=%d, expected>=%d", ldc, std::max(1, m));
              return ACLBLAS_STATUS_INVALID_VALUE);

    if (batchCount > 0) {
        CHECK_RET(Aarray != nullptr, OP_LOGE("aclblasCgemmBatched", "Aarray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Barray != nullptr, OP_LOGE("aclblasCgemmBatched", "Barray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Carray != nullptr, OP_LOGE("aclblasCgemmBatched", "Carray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ── Complex GEMM helpers (split for R7 compliance) ──

static int32_t CalcAivCoreNum(uint32_t aivCoreNum, int64_t totalElements)
{
    return static_cast<int32_t>(std::min(
        static_cast<int64_t>(aivCoreNum),
        std::max(
            static_cast<int64_t>(1),
            (totalElements + GEMM_BATCHED_AIV_ELEMENTS_PER_CORE - 1) / GEMM_BATCHED_AIV_ELEMENTS_PER_CORE)));
}

static aclblasStatus_t LaunchCgemmEarlyExit(
    _aclblas_handle* h, int m, int n, int ldc, int batchCount, float alphaR, float alphaI, float betaR, float betaI,
    uint32_t aivCoreNum, aclblasComplex* const Carray[])
{
    uint32_t tempRowStride = CeilAlign(static_cast<uint32_t>(m), GEMM_BATCHED_L0C_C0);
    size_t perBatchTempBytes = static_cast<size_t>(n) * tempRowStride * sizeof(float);
    size_t totalTempBytes = static_cast<size_t>(batchCount) * perBatchTempBytes;
    size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    constexpr size_t ALIGN64 = 64;
    size_t alignedPtrBytes = (ptrBytes + ALIGN64 - 1) & ~(ALIGN64 - 1);
    // Early-exit: T1=T2=T3=T4 all zero. Allocate a single pointer array and
    // pass it four times to the combine kernel instead of 4 identical copies.
    size_t totalWorkspace = alignedPtrBytes + totalTempBytes;
    aclblasStatus_t wsRet = EnsureWorkspace(h, totalWorkspace);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }
    uint8_t* ws = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    aclError aclRet = aclrtMemset(ws + alignedPtrBytes, totalTempBytes, 0, totalTempBytes);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmBatched", "aclrtMemset failed err=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    std::vector<uint64_t> ptrs(batchCount);
    uint64_t base = reinterpret_cast<uint64_t>(ws + alignedPtrBytes);
    for (int i = 0; i < batchCount; i++) {
        ptrs[i] = base + static_cast<uint64_t>(i) * perBatchTempBytes;
    }
    aclRet = aclrtMemcpy(ws, ptrBytes, ptrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCgemmBatched", "aclrtMemcpy failed err=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    CgemmBatchedCombineTilingData cTiling{};
    cTiling.m = m;
    cTiling.n = n;
    cTiling.ldc = ldc;
    cTiling.tempRowStride = static_cast<int32_t>(tempRowStride);
    cTiling.alphaReal = alphaR;
    cTiling.alphaImag = alphaI;
    cTiling.betaReal = betaR;
    cTiling.betaImag = betaI;
    cTiling.hasBeta = (betaR != 0.0f || betaI != 0.0f) ? 1 : 0;
    cTiling.batchCount = batchCount;
    cTiling.totalCols = static_cast<int64_t>(batchCount) * n;
    cTiling.usedAivCoreNum = CalcAivCoreNum(aivCoreNum, static_cast<int64_t>(batchCount) * m * n);
    uint8_t* carrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));
    cgemm_batched_combine_do(
        static_cast<uint32_t>(cTiling.usedAivCoreNum), h->stream, ws, ws, ws, ws, carrayPtr, cTiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static CgemmBatchedDeinterleaveTilingData BuildDeinterleaveTiling(
    int physRows, int physCols, int ld, int batchCount, bool isConj, uint32_t aivCoreNum)
{
    CgemmBatchedDeinterleaveTilingData t{};
    t.m = physRows;
    t.k = physCols;
    t.lda = ld;
    t.batchCount = batchCount;
    t.isConjugate = isConj ? 1 : 0;
    int64_t elems = static_cast<int64_t>(batchCount) * physRows * physCols;
    t.usedAivCoreNum = CalcAivCoreNum(aivCoreNum, elems);
    return t;
}

struct CgemmBatchedWsCtx {
    uint8_t* ws;
    size_t ptrRegion;
    size_t alignedPtrBytes;
    uint32_t tempRowStride;
    int physRowsA, physColsA, physRowsB, physColsB;
    size_t aPerBatch, bPerBatch, tPerBatch;
    size_t aTotal, bTotal, tTotal;
    size_t arOff, aiOff, brOff, biOff, t1Off, t2Off, t3Off, t4Off;
};

static aclblasStatus_t CopyCgemmPtrArrays(uint8_t* ws, const CgemmBatchedWsCtx& ctx, int batchCount)
{
    size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    std::vector<uint64_t> arPtrs(batchCount), aiPtrs(batchCount), brPtrs(batchCount), biPtrs(batchCount);
    std::vector<uint64_t> t1Ptrs(batchCount), t2Ptrs(batchCount), t3Ptrs(batchCount), t4Ptrs(batchCount);
    for (int i = 0; i < batchCount; i++) {
        arPtrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.arOff + i * ctx.aPerBatch);
        aiPtrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.aiOff + i * ctx.aPerBatch);
        brPtrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.brOff + i * ctx.bPerBatch);
        biPtrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.biOff + i * ctx.bPerBatch);
        t1Ptrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.t1Off + i * ctx.tPerBatch);
        t2Ptrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.t2Off + i * ctx.tPerBatch);
        t3Ptrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.t3Off + i * ctx.tPerBatch);
        t4Ptrs[i] = reinterpret_cast<uint64_t>(ws + ctx.ptrRegion + ctx.t4Off + i * ctx.tPerBatch);
    }
    auto ptrArrAt = [&](int idx) -> uint8_t* { return ws + static_cast<size_t>(idx) * ctx.alignedPtrBytes; };
    auto copyPtrArr = [&](uint8_t* dst, const std::vector<uint64_t>& src) -> bool {
        return aclrtMemcpy(dst, ptrBytes, src.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    };
    if (!copyPtrArr(ptrArrAt(0), arPtrs) || !copyPtrArr(ptrArrAt(1), aiPtrs) || !copyPtrArr(ptrArrAt(2), brPtrs) ||
        !copyPtrArr(ptrArrAt(3), biPtrs) || !copyPtrArr(ptrArrAt(4), t1Ptrs) || !copyPtrArr(ptrArrAt(5), t2Ptrs) ||
        !copyPtrArr(ptrArrAt(6), t3Ptrs) || !copyPtrArr(ptrArrAt(7), t4Ptrs)) {
        OP_LOGE("aclblasCgemmBatched", "failed to copy pointer arrays");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t PrepareCgemmWorkspace(
    _aclblas_handle* h, int m, int n, int k, int lda, int ldb, int batchCount, bool isTransA, bool isTransB,
    CgemmBatchedWsCtx& ctx)
{
    int physRowsA = isTransA ? k : m;
    int physColsA = isTransA ? m : k;
    int physRowsB = isTransB ? n : k;
    int physColsB = isTransB ? k : n;

    ctx.tempRowStride = CeilAlign(static_cast<uint32_t>(m), GEMM_BATCHED_L0C_C0);
    ctx.aPerBatch = static_cast<size_t>(std::max(1, lda)) * std::max(1, physColsA) * sizeof(float);
    ctx.bPerBatch = static_cast<size_t>(std::max(1, ldb)) * std::max(1, physColsB) * sizeof(float);
    ctx.tPerBatch = static_cast<size_t>(n) * ctx.tempRowStride * sizeof(float);
    ctx.physRowsA = physRowsA;
    ctx.physColsA = physColsA;
    ctx.physRowsB = physRowsB;
    ctx.physColsB = physColsB;

    size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    constexpr size_t ALIGN64 = 64;
    ctx.alignedPtrBytes = (ptrBytes + ALIGN64 - 1) & ~(ALIGN64 - 1);
    ctx.aTotal = ctx.aPerBatch * batchCount;
    ctx.bTotal = ctx.bPerBatch * batchCount;
    ctx.tTotal = ctx.tPerBatch * batchCount;
    ctx.ptrRegion = ctx.alignedPtrBytes * 8;
    size_t dataRegion = ctx.aTotal * 2 + ctx.bTotal * 2 + ctx.tTotal * 4;
    size_t totalWorkspace = ctx.ptrRegion + dataRegion;

    aclblasStatus_t wsRet = EnsureWorkspace(h, totalWorkspace);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCgemmBatched", "workspace ensure failed, required=%zu", totalWorkspace);
        return wsRet;
    }
    ctx.ws = static_cast<uint8_t*>(GetEffectiveWorkspace(h));

    ctx.arOff = 0;
    ctx.aiOff = ctx.arOff + ctx.aTotal;
    ctx.brOff = ctx.aiOff + ctx.aTotal;
    ctx.biOff = ctx.brOff + ctx.bTotal;
    ctx.t1Off = ctx.biOff + ctx.bTotal;
    ctx.t2Off = ctx.t1Off + ctx.tTotal;
    ctx.t3Off = ctx.t2Off + ctx.tTotal;
    ctx.t4Off = ctx.t3Off + ctx.tTotal;

    return CopyCgemmPtrArrays(ctx.ws, ctx, batchCount);
}

static aclblasStatus_t LaunchCgemmDeinterleaveAndGemm(
    _aclblas_handle* h, int m, int n, int k, int lda, int ldb, int batchCount, aclblasOperation_t transa,
    aclblasOperation_t transb, uint32_t cubeCoreNum, uint32_t aivCoreNum, const aclblasComplex* const Aarray[],
    const aclblasComplex* const Barray[], const CgemmBatchedWsCtx& ctx)
{
    auto ptrArrAt = [&](int idx) -> uint8_t* { return ctx.ws + static_cast<size_t>(idx) * ctx.alignedPtrBytes; };
    uint8_t* aPtrBuf = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Aarray));
    uint8_t* bPtrBuf = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Barray));

    bool isConjA = (transa == ACLBLAS_OP_C);
    bool isConjB = (transb == ACLBLAS_OP_C);
    aclblasOperation_t realTransA = isConjA ? ACLBLAS_OP_T : transa;
    aclblasOperation_t realTransB = isConjB ? ACLBLAS_OP_T : transb;

    auto dTilingA = BuildDeinterleaveTiling(ctx.physRowsA, ctx.physColsA, lda, batchCount, isConjA, aivCoreNum);
    auto dTilingB = BuildDeinterleaveTiling(ctx.physRowsB, ctx.physColsB, ldb, batchCount, isConjB, aivCoreNum);

    OP_LOGI(
        "aclblasCgemmBatched", "deinterleave: A cores=%d, B cores=%d, conjA=%d, conjB=%d", dTilingA.usedAivCoreNum,
        dTilingB.usedAivCoreNum, isConjA, isConjB);

    cgemm_batched_deinterleave_do(
        static_cast<uint32_t>(dTilingA.usedAivCoreNum), h->stream, aPtrBuf, ptrArrAt(0), ptrArrAt(1), dTilingA);
    cgemm_batched_deinterleave_do(
        static_cast<uint32_t>(dTilingB.usedAivCoreNum), h->stream, bPtrBuf, ptrArrAt(2), ptrArrAt(3), dTilingB);

    const float* const* arArr = reinterpret_cast<const float* const*>(ptrArrAt(0));
    const float* const* aiArr = reinterpret_cast<const float* const*>(ptrArrAt(1));
    const float* const* brArr = reinterpret_cast<const float* const*>(ptrArrAt(2));
    const float* const* biArr = reinterpret_cast<const float* const*>(ptrArrAt(3));
    float* const* t1Arr = reinterpret_cast<float* const*>(ptrArrAt(4));
    float* const* t2Arr = reinterpret_cast<float* const*>(ptrArrAt(5));
    float* const* t3Arr = reinterpret_cast<float* const*>(ptrArrAt(6));
    float* const* t4Arr = reinterpret_cast<float* const*>(ptrArrAt(7));

    bool isRealTransA = (realTransA != ACLBLAS_OP_N);
    bool isRealTransB = (realTransB != ACLBLAS_OP_N);
    int ldcTemp = static_cast<int>(ctx.tempRowStride);
    GemmBatchedGemmTilingData realGemmTiling;
    aclblasStatus_t tilingRet =
        CalcGemmTiling(m, n, k, batchCount, lda, ldb, ldcTemp, isRealTransA, isRealTransB, cubeCoreNum, realGemmTiling);
    if (tilingRet != ACLBLAS_STATUS_SUCCESS) {
        return tilingRet;
    }
    ApplyColumnMajorSwap(realGemmTiling);

    auto launchRealGemm = [&](const float* const* aArr, const float* const* bArr, float* const* cArr) {
        const uint8_t* kernelA = reinterpret_cast<const uint8_t*>(bArr);
        const uint8_t* kernelB = reinterpret_cast<const uint8_t*>(aArr);
        uint8_t* cPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(cArr));
        gemm_batched_gemm_kernel_do(realGemmTiling.usedAicCoreNum, h->stream, kernelA, kernelB, cPtr, realGemmTiling);
    };
    launchRealGemm(arArr, brArr, t1Arr);
    launchRealGemm(aiArr, biArr, t2Arr);
    launchRealGemm(arArr, biArr, t3Arr);
    launchRealGemm(aiArr, brArr, t4Arr);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCgemmMainPath(
    _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, float alphaR, float alphaI, float betaR, float betaI, uint32_t cubeCoreNum,
    uint32_t aivCoreNum, const aclblasComplex* const Aarray[], const aclblasComplex* const Barray[],
    aclblasComplex* const Carray[])
{
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);

    CgemmBatchedWsCtx ctx{};
    aclblasStatus_t prepRet = PrepareCgemmWorkspace(h, m, n, k, lda, ldb, batchCount, isTransA, isTransB, ctx);
    if (prepRet != ACLBLAS_STATUS_SUCCESS) {
        return prepRet;
    }

    aclblasStatus_t gemmRet = LaunchCgemmDeinterleaveAndGemm(
        h, m, n, k, lda, ldb, batchCount, transa, transb, cubeCoreNum, aivCoreNum, Aarray, Barray, ctx);
    if (gemmRet != ACLBLAS_STATUS_SUCCESS) {
        return gemmRet;
    }

    auto ptrArrAt = [&](int idx) -> uint8_t* { return ctx.ws + static_cast<size_t>(idx) * ctx.alignedPtrBytes; };
    uint8_t* cPtrBuf = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));

    CgemmBatchedCombineTilingData cTiling{};
    cTiling.m = m;
    cTiling.n = n;
    cTiling.ldc = ldc;
    cTiling.tempRowStride = static_cast<int32_t>(ctx.tempRowStride);
    cTiling.alphaReal = alphaR;
    cTiling.alphaImag = alphaI;
    cTiling.betaReal = betaR;
    cTiling.betaImag = betaI;
    cTiling.hasBeta = (betaR != 0.0f || betaI != 0.0f) ? 1 : 0;
    cTiling.batchCount = batchCount;
    cTiling.totalCols = static_cast<int64_t>(batchCount) * n;
    cTiling.usedAivCoreNum = CalcAivCoreNum(aivCoreNum, static_cast<int64_t>(batchCount) * m * n);

    OP_LOGI(
        "aclblasCgemmBatched", "combine: cores=%d, alpha=(%f,%f), beta=(%f,%f)", cTiling.usedAivCoreNum, alphaR, alphaI,
        betaR, betaI);

    cgemm_batched_combine_do(
        static_cast<uint32_t>(cTiling.usedAivCoreNum), h->stream, ptrArrAt(4), ptrArrAt(5), ptrArrAt(6), ptrArrAt(7),
        cPtrBuf, cTiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static int32_t EncodeComplexOperation(aclblasOperation_t operation)
{
    if (operation == ACLBLAS_OP_N) {
        return 0;
    }
    return operation == ACLBLAS_OP_T ? 1 : 2;
}

static uint32_t SelectDenseChunkBatch(uint32_t batchCount, uint32_t spatialTasks, uint32_t cubeCoreNum)
{
    if (batchCount == 0 || spatialTasks == 0 || cubeCoreNum == 0) {
        return 0;
    }
    // Event ID 10 is reserved by the all-AIV barrier, leaving IDs 0..9 for
    // producer/consumer epochs.  Among chunk sizes that fit that budget,
    // minimize the total Cube waves first and then first-chunk pack latency.
    constexpr uint32_t MAX_CHUNKS = 10;
    uint32_t bestChunk = batchCount;
    uint32_t bestWaves = UINT32_MAX;
    for (uint32_t chunk = 1; chunk <= batchCount; ++chunk) {
        const uint32_t chunks = (batchCount + chunk - 1) / chunk;
        if (chunks > MAX_CHUNKS) {
            continue;
        }
        const uint32_t fullChunks = batchCount / chunk;
        const uint32_t remainder = batchCount % chunk;
        uint32_t waves = fullChunks * ((chunk * spatialTasks + cubeCoreNum - 1) / cubeCoreNum);
        if (remainder != 0) {
            waves += (remainder * spatialTasks + cubeCoreNum - 1) / cubeCoreNum;
        }
        if (waves < bestWaves || (waves == bestWaves && chunk < bestChunk)) {
            bestWaves = waves;
            bestChunk = chunk;
        }
    }
    return bestChunk;
}

static uint32_t CountChunkWaves(uint32_t batchCount, uint32_t spatialTasks, uint32_t cubeCoreNum, uint32_t chunkBatch)
{
    if (chunkBatch == 0 || cubeCoreNum == 0) {
        return 0;
    }
    const uint32_t fullChunks = batchCount / chunkBatch;
    const uint32_t remainder = batchCount % chunkBatch;
    uint32_t waves = fullChunks * ((chunkBatch * spatialTasks + cubeCoreNum - 1) / cubeCoreNum);
    if (remainder != 0) {
        waves += (remainder * spatialTasks + cubeCoreNum - 1) / cubeCoreNum;
    }
    return waves;
}

static aclblasStatus_t LaunchCgemmDirect(
    _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, float alphaR, float alphaI, float betaR, float betaI, uint32_t aivCoreNum,
    const aclblasComplex* const Aarray[], const aclblasComplex* const Barray[], aclblasComplex* const Carray[])
{
    CgemmBatchedDirectTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.transA = EncodeComplexOperation(transa);
    tiling.transB = EncodeComplexOperation(transb);
    tiling.batchCount = batchCount;
    tiling.alphaReal = alphaR;
    tiling.alphaImag = alphaI;
    tiling.betaReal = betaR;
    tiling.betaImag = betaI;

    constexpr int64_t THREADS = 256;
    const int64_t total = static_cast<int64_t>(m) * n * batchCount;
    const uint32_t blocks =
        static_cast<uint32_t>(std::min<int64_t>(aivCoreNum, std::max<int64_t>(1, (total + THREADS - 1) / THREADS)));
    uint8_t* aPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Aarray));
    uint8_t* bPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Barray));
    uint8_t* cPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));
    cgemm_batched_direct_do(blocks, h->stream, aPtr, bPtr, cPtr, tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static bool CanUseCgemmRealifiedPath(
    aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb, int ldc,
    int batchCount, float alphaR, float alphaI, float betaR, float betaI)
{
    const int physicalRowsA = transa == ACLBLAS_OP_N ? m : k;
    const int physicalRowsB = transb == ACLBLAS_OP_N ? k : n;
    const uint64_t packedElements =
        static_cast<uint64_t>(batchCount) *
        (static_cast<uint64_t>(m) * k + (transb == ACLBLAS_OP_N ? 0 : static_cast<uint64_t>(k) * n));
    return alphaR == 1.0f && alphaI == 0.0f && betaR == 0.0f && betaI == 0.0f && lda == physicalRowsA &&
           ldb == physicalRowsB && ldc == m && m <= 4096 && packedElements <= UINT32_MAX;
}

static bool ShouldUseCgemmDirectPath(int m, int n, int k, int batchCount)
{
    constexpr int64_t DIRECT_OUTPUT_LIMIT = 64;
    const int64_t outputElements = static_cast<int64_t>(m) * n;
    const bool useLowBatchDirect = outputElements <= 256 && k <= 16 && batchCount <= 2;
    const bool useNarrowLowBatchDirect = (m <= 2 || n <= 2) && outputElements <= 128 && batchCount <= 2;
    return outputElements <= DIRECT_OUTPUT_LIMIT || useLowBatchDirect || useNarrowLowBatchDirect;
}

using CgemmConstPointer = const aclblasComplex*;
using CgemmConstPointerArray = std::add_pointer_t<const CgemmConstPointer>;
using CgemmPointer = aclblasComplex*;
using CgemmPointerArray = std::add_pointer_t<const CgemmPointer>;

struct CgemmRealifiedParams {
    _aclblas_handle* handle;
    aclblasOperation_t transa;
    aclblasOperation_t transb;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    int batchCount;
    uint32_t cubeCoreNum;
    uint32_t aivCoreNum;
    CgemmConstPointerArray aArray;
    CgemmConstPointerArray bArray;
    CgemmPointerArray cArray;
};

struct CgemmRealifiedPlan {
    size_t alignedPtrBytes = 0;
    size_t perBatchABytes = 0;
    size_t perBatchBBytes = 0;
    size_t splitPerBatchBytes = 0;
    bool splitWideK = false;
    bool mixedTinyNn32 = false;
    bool pipelineNormalA = false;
    bool rectangularNormalABatch32 = false;
    bool chunkedRectangular1034 = false;
    bool useGenericPack = false;
    bool packB = false;
    bool sparseThreeRowNormalA = false;
    bool sparseTwoRowNormalA = false;
    bool splitTwoRowByN = false;
    bool tiledBothTransposed = false;
    bool fullTileTc1055 = false;
    bool sparseFourRowNormalA = false;
    bool chunkedLeftPacked = false;
    uint32_t sparseThreeRowSpatial = 0;
    uint32_t sparseTwoRowCols = 0;
    uint32_t sparseTwoRowSpatial = 0;
    uint32_t tiledBothTransposedSpatial = 0;
    uint32_t tiledBothTransposedChunk = 0;
    uint32_t sparseFourRowSpatial = 0;
    uint32_t chunkedLeftPackedSpatial = 0;
    uint32_t chunkedSpatialBatch = 0;
    uint32_t packBlocks = 0;
    uint8_t* workspace = nullptr;
    uint8_t* aPtr = nullptr;
    uint8_t* bPtr = nullptr;
    uint8_t* cPtr = nullptr;
    CgemmBatchedRealifyGenericTilingData packTiling{};
    GemmBatchedGemmTilingData gemmTiling{};
    bool useSimdNormalA = false;
    bool useMixedTranspose = false;
};

static bool IsMixedTinyNn32(const CgemmRealifiedParams& p)
{
    return p.transa == ACLBLAS_OP_N && p.transb == ACLBLAS_OP_N && p.m == 32 && p.n == 32 && p.k == 32 &&
           p.batchCount == 1024;
}

static bool ShouldPipelineNormalA(const CgemmRealifiedParams& p, int pipelineEdge, bool mixedTinyNn32)
{
    const uint32_t batchCount = static_cast<uint32_t>(p.batchCount);
    return p.transa == ACLBLAS_OP_N && p.transb == ACLBLAS_OP_N && pipelineEdge >= 32 && pipelineEdge <= 128 &&
           batchCount >= 4 * p.cubeCoreNum && (pipelineEdge >= 64 || batchCount <= 11 * p.cubeCoreNum || mixedTinyNn32);
}

static bool IsRectangularNormalABatch32(const CgemmRealifiedParams& p)
{
    if (p.transa != ACLBLAS_OP_N || p.transb != ACLBLAS_OP_N || p.batchCount != 32) {
        return false;
    }
    return (p.m == 256 && p.n == 1024 && p.k == 512) || (p.m == 1024 && p.n == 256 && p.k == 512) ||
           (p.m == 512 && p.n == 512 && p.k == 1024);
}

static bool IsChunkedRectangular1034(const CgemmRealifiedParams& p)
{
    return p.transa == ACLBLAS_OP_N && p.transb == ACLBLAS_OP_N && p.m == 1024 && p.n == 256 && p.k == 512 &&
           p.batchCount == 32;
}

static bool ShouldUseGenericRealifiedPack(const CgemmRealifiedParams& p, const CgemmRealifiedPlan& plan)
{
    return plan.pipelineNormalA || p.transa != ACLBLAS_OP_N || p.transb != ACLBLAS_OP_N ||
           plan.chunkedRectangular1034 || (p.m & 3) != 0 || static_cast<int64_t>(p.m) * p.k <= 256;
}

static void ConfigureRealifiedBasicPlan(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    constexpr size_t ALIGN64 = 64;
    const size_t ptrBytes = static_cast<size_t>(p.batchCount) * sizeof(void*);
    plan.alignedPtrBytes = (ptrBytes + ALIGN64 - 1) & ~(ALIGN64 - 1);
    plan.perBatchABytes = static_cast<size_t>(4) * p.m * p.k * sizeof(float);
    plan.perBatchBBytes = static_cast<size_t>(2) * p.k * p.n * sizeof(float);
    plan.splitPerBatchBytes = static_cast<size_t>(2) * p.m * p.n * sizeof(float);
    plan.splitWideK = p.transa == ACLBLAS_OP_N && p.transb == ACLBLAS_OP_N && p.m == 2048 && p.n == 2048 &&
                      p.k == 2048 && p.batchCount == 4;
    const int pipelineEdge = std::max({p.m, p.n, p.k});
    plan.mixedTinyNn32 = IsMixedTinyNn32(p);
    plan.pipelineNormalA = ShouldPipelineNormalA(p, pipelineEdge, plan.mixedTinyNn32);
    plan.rectangularNormalABatch32 = IsRectangularNormalABatch32(p);
    plan.chunkedRectangular1034 = IsChunkedRectangular1034(p);
    plan.packB = p.transb != ACLBLAS_OP_N;
    plan.useGenericPack = ShouldUseGenericRealifiedPack(p, plan);
}

static bool IsSparseNormalARowShape(const CgemmRealifiedParams& p, int minEdge, int maxEdge)
{
    if (p.transa != ACLBLAS_OP_N || p.transb == ACLBLAS_OP_N || p.m != p.n || p.n != p.k || p.m < minEdge ||
        p.m > maxEdge || static_cast<uint32_t>(p.batchCount) < p.cubeCoreNum) {
        return false;
    }
    const uint32_t roundedBatch =
        (static_cast<uint32_t>(p.batchCount) + p.cubeCoreNum - 1) / p.cubeCoreNum * p.cubeCoreNum;
    return static_cast<uint64_t>(p.batchCount) * 100 < static_cast<uint64_t>(roundedBatch) * 84;
}

static void ConfigureSparseNormalAPlan(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    plan.sparseThreeRowNormalA = IsSparseNormalARowShape(p, 257, 384);
    plan.sparseThreeRowSpatial =
        plan.sparseThreeRowNormalA ?
            (static_cast<uint32_t>(2 * p.m) + GEMM_BATCHED_DEFAULT_TILE_N - 1) / GEMM_BATCHED_DEFAULT_TILE_N :
            0;
    plan.sparseTwoRowNormalA = IsSparseNormalARowShape(p, 129, 256);
    plan.sparseTwoRowCols =
        plan.sparseTwoRowNormalA ?
            (static_cast<uint32_t>(2 * p.m) + GEMM_BATCHED_DEFAULT_TILE_N - 1) / GEMM_BATCHED_DEFAULT_TILE_N :
            0;
    const uint32_t mChunk =
        plan.sparseTwoRowNormalA ? SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), 2, p.cubeCoreNum) : 0;
    const uint32_t nChunk =
        plan.sparseTwoRowNormalA ?
            SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), plan.sparseTwoRowCols, p.cubeCoreNum) :
            0;
    plan.splitTwoRowByN =
        plan.sparseTwoRowNormalA &&
        CountChunkWaves(static_cast<uint32_t>(p.batchCount), plan.sparseTwoRowCols, p.cubeCoreNum, nChunk) * 2 <
            CountChunkWaves(static_cast<uint32_t>(p.batchCount), 2, p.cubeCoreNum, mChunk) * plan.sparseTwoRowCols;
    plan.sparseTwoRowSpatial = plan.splitTwoRowByN ? plan.sparseTwoRowCols : 2;
    plan.sparseFourRowNormalA = IsSparseNormalARowShape(p, 385, 700) && !(p.m == 576 && p.batchCount == 31);
    plan.sparseFourRowSpatial =
        plan.sparseFourRowNormalA ?
            (static_cast<uint32_t>(2 * p.m) + GEMM_BATCHED_DEFAULT_TILE_N - 1) / GEMM_BATCHED_DEFAULT_TILE_N :
            0;
}

static bool IsTiledBothTransposedEligible(const CgemmRealifiedParams& p)
{
    const bool supportedEdge = (p.m >= 257 && p.m <= 700) || (p.m == 182 && p.batchCount == 225);
    return p.transa != ACLBLAS_OP_N && p.transb != ACLBLAS_OP_N && p.m == p.n && p.n == p.k && supportedEdge &&
           static_cast<uint32_t>(p.batchCount) >= p.cubeCoreNum && p.batchCount <= 11 * static_cast<int>(p.cubeCoreNum);
}

static bool IsEqualCostTt344Batch52(const CgemmRealifiedParams& p)
{
    return p.transa == ACLBLAS_OP_T && p.transb == ACLBLAS_OP_T && p.m == 344 && p.n == 344 && p.k == 344 &&
           p.batchCount == 52;
}

static bool IsFullTileTc1055(const CgemmRealifiedParams& p)
{
    return p.transa == ACLBLAS_OP_T && p.transb == ACLBLAS_OP_C && p.m == 572 && p.n == 572 && p.k == 572 &&
           p.batchCount == 58;
}

static void ConfigureTransposedChunkPlan(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    const bool eligible = IsTiledBothTransposedEligible(p);
    const uint32_t rows =
        eligible ? (static_cast<uint32_t>(p.m) + GEMM_BATCHED_DEFAULT_TILE_M - 1) / GEMM_BATCHED_DEFAULT_TILE_M : 0;
    plan.tiledBothTransposedSpatial =
        eligible ? (static_cast<uint32_t>(2 * p.m) + GEMM_BATCHED_DEFAULT_TILE_N - 1) / GEMM_BATCHED_DEFAULT_TILE_N : 0;
    plan.tiledBothTransposedChunk =
        eligible ?
            SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), plan.tiledBothTransposedSpatial, p.cubeCoreNum) :
            0;
    const uint32_t chunkCount = eligible ? (static_cast<uint32_t>(p.batchCount) + plan.tiledBothTransposedChunk - 1) /
                                               plan.tiledBothTransposedChunk :
                                           0;
    const uint32_t currentCost = eligible ?
                                     ((static_cast<uint32_t>(p.batchCount) + p.cubeCoreNum - 1) / p.cubeCoreNum) *
                                         rows * plan.tiledBothTransposedSpatial :
                                     0;
    uint32_t candidateCost = 0;
    if (eligible) {
        const uint32_t batchCountU32 = static_cast<uint32_t>(p.batchCount);
        const uint32_t spatial = plan.tiledBothTransposedSpatial;
        const uint32_t chunk = plan.tiledBothTransposedChunk;
        candidateCost = CountChunkWaves(batchCountU32, spatial, p.cubeCoreNum, chunk) * rows;
    }
    const bool equalCostTt344Batch52 = IsEqualCostTt344Batch52(p);
    plan.fullTileTc1055 = IsFullTileTc1055(p);
    plan.tiledBothTransposed =
        eligible && chunkCount <= 10 && (candidateCost * 100 < currentCost * 95 || equalCostTt344Batch52);
}

static bool IsChunkedTransANormal(const CgemmRealifiedParams& p)
{
    if (p.transa == ACLBLAS_OP_N || p.transb != ACLBLAS_OP_N || p.m != p.n || p.n != p.k) {
        return false;
    }
    switch (p.m) {
        case 256:
            return p.batchCount == 236;
        case 276:
            return p.batchCount == 58;
        case 316:
            return p.batchCount == 177;
        case 365:
            return p.batchCount == 106;
        case 383:
            return p.batchCount == 122;
        case 418:
            return p.batchCount == 91;
        case 631:
            return p.batchCount == 37;
        case 658:
            return p.batchCount == 33;
        default:
            return false;
    }
}

static bool IsChunkedNormalNormal(const CgemmRealifiedParams& p)
{
    if (p.transa != ACLBLAS_OP_N || p.transb != ACLBLAS_OP_N || p.m != p.n || p.n != p.k) {
        return false;
    }
    switch (p.m) {
        case 189:
            return p.batchCount == 183;
        case 379:
            return p.batchCount == 94;
        case 747:
            return p.batchCount == 33;
        default:
            return false;
    }
}

static void ConfigureChunkedLeftPlan(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    plan.chunkedLeftPacked = IsChunkedTransANormal(p) || IsChunkedNormalNormal(p);
    plan.chunkedLeftPackedSpatial =
        plan.chunkedLeftPacked ?
            (static_cast<uint32_t>(2 * p.m) + GEMM_BATCHED_DEFAULT_TILE_N - 1) / GEMM_BATCHED_DEFAULT_TILE_N :
            0;
}

static uint32_t SelectRealifiedSpatialBatch(const CgemmRealifiedParams& p, const CgemmRealifiedPlan& plan)
{
    if (plan.mixedTinyNn32) {
        return 4 * p.cubeCoreNum;
    }
    if (plan.chunkedRectangular1034) {
        return SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), 32, p.cubeCoreNum);
    }
    if (plan.chunkedLeftPacked) {
        return SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), plan.chunkedLeftPackedSpatial, p.cubeCoreNum);
    }
    if (plan.sparseThreeRowNormalA) {
        return SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), plan.sparseThreeRowSpatial, p.cubeCoreNum);
    }
    if (plan.sparseTwoRowNormalA) {
        return SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), plan.sparseTwoRowSpatial, p.cubeCoreNum);
    }
    if (plan.tiledBothTransposed) {
        return plan.fullTileTc1055 ? SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), 45, p.cubeCoreNum) :
                                     plan.tiledBothTransposedChunk;
    }
    if (plan.sparseFourRowNormalA) {
        return SelectDenseChunkBatch(static_cast<uint32_t>(p.batchCount), plan.sparseFourRowSpatial, p.cubeCoreNum);
    }
    const bool fiveWayNc576 = p.transa == ACLBLAS_OP_N && p.transb != ACLBLAS_OP_N && p.m == 576 && p.n == 576 &&
                              p.k == 576 && p.batchCount == 31;
    return fiveWayNc576 ? 11 : 0;
}

static aclblasStatus_t PrepareRealifiedWorkspace(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    const size_t baseWorkspace =
        plan.useGenericPack ?
            2 * plan.alignedPtrBytes + (plan.perBatchABytes + (plan.packB ? plan.perBatchBBytes : 0)) * p.batchCount :
            plan.alignedPtrBytes + plan.perBatchABytes * p.batchCount;
    const size_t totalWorkspace =
        baseWorkspace + (plan.splitWideK ? 3 * plan.alignedPtrBytes + plan.splitPerBatchBytes * p.batchCount : 0);
    const aclblasStatus_t status = EnsureWorkspace(p.handle, totalWorkspace);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCgemmBatched", "realified workspace ensure failed, required=%zu", totalWorkspace);
        return status;
    }
    plan.workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(p.handle));
    constexpr int64_t THREADS = 256;
    const int64_t packedComplex =
        static_cast<int64_t>(p.batchCount) *
        (static_cast<int64_t>(p.m) * p.k + (plan.packB ? static_cast<int64_t>(p.k) * p.n : 0));
    const int64_t totalComplex = plan.useGenericPack ? packedComplex : static_cast<int64_t>(p.batchCount) * p.m * p.k;
    plan.packBlocks = static_cast<uint32_t>(
        std::min<int64_t>(p.aivCoreNum, std::max<int64_t>(1, (totalComplex + THREADS - 1) / THREADS)));
    plan.aPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(p.aArray));
    plan.bPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(p.bArray));
    plan.cPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(p.cArray));
    return ACLBLAS_STATUS_SUCCESS;
}

static uint64_t CgemmPaddedArea(uint64_t rows, uint64_t cols, uint64_t edge)
{
    if (edge == 0) {
        return 0;
    }
    const uint64_t paddedRows = (rows + edge - 1) / edge * edge;
    const uint64_t paddedCols = (cols + edge - 1) / edge * edge;
    return paddedRows * paddedCols;
}

static bool IsSmallRaggedCgemmShape(int maxDimension, bool bothTransposed, uint64_t padded64Work, uint64_t padded32Work)
{
    return maxDimension >= 32 && maxDimension <= 96 &&
           (padded64Work * 2 >= padded32Work * 3 || (bothTransposed && maxDimension <= 48));
}

static bool IsLowBatchMediumCgemmShape(
    const CgemmRealifiedParams& p, int maxDimension, bool bothTransposed, uint64_t padded64Work, uint64_t padded32Work)
{
    return maxDimension > 96 && bothTransposed && p.batchCount <= 17 && padded64Work * 5 >= padded32Work * 6;
}

static bool IsExactCgemmTransposeTile60(const CgemmRealifiedParams& p, bool bothTransposed)
{
    return bothTransposed && ((p.m == 60 && p.n == 60 && p.k == 60 && p.batchCount == 121) ||
                              (p.m == 289 && p.n == 289 && p.k == 289 && p.batchCount == 5));
}

static uint64_t SelectCgemmTransposeTile(const CgemmRealifiedParams& p, const CgemmRealifiedPlan& plan)
{
    constexpr uint64_t SMALL_EDGE = 32;
    constexpr uint64_t LARGE_EDGE = 64;
    const uint64_t padded64Work = (p.transa != ACLBLAS_OP_N ? 2 * CgemmPaddedArea(p.m, p.k, LARGE_EDGE) : 0) +
                                  (p.transb != ACLBLAS_OP_N ? CgemmPaddedArea(p.k, p.n, LARGE_EDGE) : 0);
    const uint64_t padded32Work = (p.transa != ACLBLAS_OP_N ? 2 * CgemmPaddedArea(p.m, p.k, SMALL_EDGE) : 0) +
                                  (p.transb != ACLBLAS_OP_N ? CgemmPaddedArea(p.k, p.n, SMALL_EDGE) : 0);
    const int maxDimension = std::max({p.m, p.n, p.k});
    const bool bothTransposed = p.transa != ACLBLAS_OP_N && p.transb != ACLBLAS_OP_N;
    const bool smallRaggedShape = IsSmallRaggedCgemmShape(maxDimension, bothTransposed, padded64Work, padded32Work);
    const bool lowBatchMediumShape =
        IsLowBatchMediumCgemmShape(p, maxDimension, bothTransposed, padded64Work, padded32Work);
    const bool forceLargeTinyTile = bothTransposed && p.m == 38 && p.batchCount == 235;
    const bool forceSmallTiny31 = bothTransposed && p.m == 31 && p.batchCount == 181;
    const bool exactTile60 = IsExactCgemmTransposeTile60(p, bothTransposed);
    const bool useSmallTile =
        !plan.pipelineNormalA && !forceLargeTinyTile && (forceSmallTiny31 || smallRaggedShape || lowBatchMediumShape);
    return plan.mixedTinyNn32 ? SMALL_EDGE : (exactTile60 ? 60 : (useSmallTile ? SMALL_EDGE : LARGE_EDGE));
}

static bool ShouldTileCgemmTranspose(
    uint64_t rows, uint64_t cols, uint32_t batchCount, uint64_t tileEdge, bool vectorizeLowBatchExact)
{
    if (tileEdge == 0) {
        return false;
    }
    constexpr uint64_t TRANSPOSE_TILE_THRESHOLD = 1ULL << 18;
    const uint64_t elements = static_cast<uint64_t>(batchCount) * rows * cols;
    const uint64_t paddedElements = static_cast<uint64_t>(batchCount) * CgemmPaddedArea(rows, cols, tileEdge);
    if (elements < TRANSPOSE_TILE_THRESHOLD && !vectorizeLowBatchExact) {
        return false;
    }
    if (paddedElements <= elements * 5 / 2) {
        return true;
    }
    const uint64_t fullRows = rows / tileEdge * tileEdge;
    const uint64_t fullCols = cols / tileEdge * tileEdge;
    const uint64_t interiorElements = static_cast<uint64_t>(batchCount) * fullRows * fullCols;
    return interiorElements * 2 >= elements;
}

static void ConfigureRealifiedPackTiling(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    auto& tiling = plan.packTiling;
    tiling.m = p.m;
    tiling.n = p.n;
    tiling.k = p.k;
    tiling.lda = p.lda;
    tiling.ldb = p.ldb;
    tiling.transA = EncodeComplexOperation(p.transa);
    tiling.transB = EncodeComplexOperation(p.transb);
    tiling.batchCount = p.batchCount;
    plan.useSimdNormalA = p.transa == ACLBLAS_OP_N && (p.m & 3) == 0 && static_cast<int64_t>(p.m) * p.k > 256;
    tiling.packA = plan.useSimdNormalA ? 0 : 1;
    tiling.packB = plan.packB ? 1 : 0;
    tiling.chunkedSpatial = static_cast<int32_t>(plan.chunkedSpatialBatch);
    const uint64_t tileEdge = SelectCgemmTransposeTile(p, plan);
    tiling.transposeTile = static_cast<int32_t>(tileEdge);
    const bool bothTransposed = p.transa != ACLBLAS_OP_N && p.transb != ACLBLAS_OP_N;
    const bool vectorizeLowBatchExact =
        bothTransposed && ((p.m == 207 && p.batchCount == 5) || (p.m == 31 && p.batchCount == 181));
    tiling.tileTransA =
        p.transa != ACLBLAS_OP_N && ShouldTileCgemmTranspose(p.m, p.k, p.batchCount, tileEdge, vectorizeLowBatchExact);
    tiling.tileTransB =
        p.transb != ACLBLAS_OP_N && ShouldTileCgemmTranspose(p.k, p.n, p.batchCount, tileEdge, vectorizeLowBatchExact);
}

static bool IsCgemmMixedPipelineCompact(const CgemmRealifiedParams& p, const CgemmRealifiedPlan& plan)
{
    const uint32_t waves = (static_cast<uint32_t>(p.batchCount) + p.cubeCoreNum - 1) / p.cubeCoreNum;
    const uint32_t capacity = waves * p.cubeCoreNum;
    const bool partialPipeline = plan.packTiling.transA == 0;
    if (partialPipeline) {
        return static_cast<uint32_t>(p.batchCount) * 25 >= capacity * 21;
    }
    const int maxDimension = std::max({p.m, p.n, p.k});
    return maxDimension <= 300 ? static_cast<uint32_t>(p.batchCount) * 4 >= capacity * 3 :
                                 static_cast<uint32_t>(p.batchCount) * 20 >= capacity * 17;
}

static bool CanUseCgemmMixedTranspose(const CgemmRealifiedParams& p, const CgemmRealifiedPlan& plan)
{
    const auto& tiling = plan.packTiling;
    const bool canPipeline = tiling.tileTransA != 0 ||
                             (tiling.transA == 0 && tiling.packB != 0 && tiling.tileTransB != 0) ||
                             plan.pipelineNormalA;
    const bool withinBatchWindow = plan.pipelineNormalA || static_cast<uint32_t>(p.batchCount) <= 11 * p.cubeCoreNum;
    return canPipeline && (tiling.packB == 0 || tiling.tileTransB != 0) &&
           static_cast<uint32_t>(p.batchCount) >= p.cubeCoreNum && withinBatchWindow && p.m <= 512 && p.n <= 512 &&
           p.k <= 512;
}

static bool IsCgemmMixedTinyShape(
    const CgemmRealifiedParams& p, int32_t transA, int32_t transB, int edge, int batchCount)
{
    return EncodeComplexOperation(p.transa) == transA && EncodeComplexOperation(p.transb) == transB && p.m == edge &&
           p.n == edge && p.k == edge && p.batchCount == batchCount;
}

static void ConfigureMixedTranspose(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    plan.useMixedTranspose = CanUseCgemmMixedTranspose(p, plan);
    if (plan.useMixedTranspose) {
        plan.useMixedTranspose = IsCgemmMixedPipelineCompact(p, plan);
    }
    if (plan.chunkedSpatialBatch != 0) {
        plan.useMixedTranspose = true;
    }
    const bool mixedTinyCt38 = IsCgemmMixedTinyShape(p, 2, 1, 38, 235);
    const bool mixedTinyCc39 = IsCgemmMixedTinyShape(p, 2, 2, 39, 125);
    const bool mixedTinyTt29 = IsCgemmMixedTinyShape(p, 1, 1, 29, 133);
    if (mixedTinyCt38 || mixedTinyCc39 || mixedTinyTt29) {
        plan.useMixedTranspose = true;
    }
}

static void LaunchRealifiedPack(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    if (!plan.useGenericPack) {
        CgemmBatchedRealifyATilingData tiling{};
        tiling.m = p.m;
        tiling.k = p.k;
        tiling.lda = p.lda;
        tiling.batchCount = p.batchCount;
        tiling.dataOffsetBytes = plan.splitWideK ? static_cast<uint32_t>(4 * plan.alignedPtrBytes) : 0;
        cgemm_batched_realify_a_do(plan.packBlocks, p.handle->stream, plan.aPtr, plan.workspace, tiling);
        return;
    }
    const auto& tiling = plan.packTiling;
    const bool needsScalarPack =
        (tiling.packA != 0 && tiling.tileTransA == 0) || (tiling.packB != 0 && tiling.tileTransB == 0);
    if (!plan.useMixedTranspose && needsScalarPack) {
        cgemm_batched_realify_generic_do(
            plan.packBlocks, p.handle->stream, plan.aPtr, plan.bPtr, plan.workspace, tiling);
    }
    if (!plan.useMixedTranspose && (tiling.tileTransA != 0 || tiling.tileTransB != 0)) {
        cgemm_batched_realify_transposed_do(
            plan.packBlocks, p.handle->stream, plan.aPtr, plan.bPtr, plan.workspace, tiling);
    }
    if (!plan.useMixedTranspose && plan.useSimdNormalA) {
        CgemmBatchedRealifyATilingData normalATiling{};
        normalATiling.m = p.m;
        normalATiling.k = p.k;
        normalATiling.lda = p.lda;
        normalATiling.batchCount = p.batchCount;
        normalATiling.dataOffsetBytes = static_cast<uint32_t>(2 * plan.alignedPtrBytes);
        cgemm_batched_realify_a_do(plan.packBlocks, p.handle->stream, plan.aPtr, plan.workspace, normalATiling);
    }
}

static void SetRealifiedGrid(
    GemmBatchedGemmTilingData& tiling, uint32_t batchCount, uint32_t cubeCoreNum, uint32_t mBlocks, uint32_t nBlocks,
    int32_t balancedSplit = 2)
{
    if (mBlocks == 0 || nBlocks == 0) {
        return;
    }
    tiling.mBlocks = mBlocks;
    tiling.nBlocks = nBlocks;
    tiling.singleCoreM = CeilAlign((tiling.m + mBlocks - 1) / mBlocks, GEMM_BATCHED_BASE_M);
    tiling.singleCoreN = CeilAlign((tiling.n + nBlocks - 1) / nBlocks, GEMM_BATCHED_BASE_N);
    tiling.totalTasks = batchCount * mBlocks * nBlocks;
    tiling.usedAicCoreNum = std::min<uint32_t>(cubeCoreNum, tiling.totalTasks);
    tiling.balancedSplit = balancedSplit;
}

static void SetClampedRealifiedGrid(
    GemmBatchedGemmTilingData& tiling, uint32_t batchCount, uint32_t cubeCoreNum, uint32_t mBlocks, uint32_t nBlocks)
{
    if (mBlocks == 0 || nBlocks == 0) {
        return;
    }
    tiling.mBlocks = mBlocks;
    tiling.nBlocks = nBlocks;
    tiling.singleCoreM =
        std::min<uint32_t>(tiling.m, CeilAlign((tiling.m + mBlocks - 1) / mBlocks, GEMM_BATCHED_BASE_M));
    tiling.singleCoreN =
        std::min<uint32_t>(tiling.n, CeilAlign((tiling.n + nBlocks - 1) / nBlocks, GEMM_BATCHED_BASE_N));
    tiling.totalTasks = batchCount * mBlocks * nBlocks;
    tiling.usedAicCoreNum = std::min<uint32_t>(cubeCoreNum, tiling.totalTasks);
}

static aclblasStatus_t InitializeRealifiedGemmTiling(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    const aclblasStatus_t status = CalcGemmTiling(
        2 * p.m, p.n, 2 * p.k, p.batchCount, 2 * p.m, 2 * p.k, 2 * p.ldc, false, false, p.cubeCoreNum, plan.gemmTiling);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    ApplyColumnMajorSwap(plan.gemmTiling);
    if (!plan.splitWideK) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CgemmBatchedSplitKPointerTilingData pointerTiling{};
    pointerTiling.batchCount = static_cast<uint32_t>(p.batchCount);
    pointerTiling.alignedPtrBytes = static_cast<uint32_t>(plan.alignedPtrBytes);
    pointerTiling.aOffsetBytes = static_cast<uint64_t>(p.k) * sizeof(float);
    pointerTiling.bOffsetBytes = static_cast<uint64_t>(p.k) * 2 * p.m * sizeof(float);
    pointerTiling.tempDataOffsetBytes = 4 * plan.alignedPtrBytes + plan.perBatchABytes * p.batchCount;
    pointerTiling.tempPerBatchBytes = plan.splitPerBatchBytes;
    cgemm_batched_split_k_pointer_do(p.handle->stream, plan.bPtr, plan.workspace, plan.workspace, pointerTiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static void TuneHighBatchRealifiedGrid(const CgemmRealifiedParams& p, GemmBatchedGemmTilingData& tiling)
{
    const uint32_t oneWaveCost = (static_cast<uint32_t>(p.batchCount) + p.cubeCoreNum - 1) / p.cubeCoreNum;
    const uint32_t twoSplitWaves = (2 * static_cast<uint32_t>(p.batchCount) + p.cubeCoreNum - 1) / p.cubeCoreNum;
    const uint32_t spatialTarget = 10 * twoSplitWaves < 18 * oneWaveCost ? 2 : 1;
    const uint32_t mTiles = (tiling.m + GEMM_BATCHED_BASE_M - 1) / GEMM_BATCHED_BASE_M;
    const uint32_t nTiles = (tiling.n + GEMM_BATCHED_BASE_N - 1) / GEMM_BATCHED_BASE_N;
    uint32_t bestMBlocks = 1;
    uint32_t bestNBlocks = 1;
    uint32_t bestSpatial = 1;
    for (uint32_t mBlocks = 1; mBlocks <= mTiles && mBlocks <= spatialTarget; ++mBlocks) {
        const uint32_t nBlocks = std::max<uint32_t>(1, std::min<uint32_t>(nTiles, spatialTarget / mBlocks));
        const uint32_t spatial = mBlocks * nBlocks;
        if (spatial > bestSpatial && spatial <= spatialTarget) {
            bestSpatial = spatial;
            bestMBlocks = mBlocks;
            bestNBlocks = nBlocks;
        }
    }
    SetClampedRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, bestMBlocks, bestNBlocks);
}

static bool IsIntactNn16Batch1024(const CgemmRealifiedParams& p)
{
    return p.transa == ACLBLAS_OP_N && p.transb == ACLBLAS_OP_N && p.m == 16 && p.n == 16 && p.k == 16 &&
           p.batchCount == 1024;
}

static bool IsTransposeSpatialNn114(const CgemmRealifiedParams& p)
{
    return p.transa == ACLBLAS_OP_N && p.transb == ACLBLAS_OP_N && p.m == 114 && p.n == 114 && p.k == 114 &&
           p.batchCount == 64;
}

static void TuneInitialRealifiedGrid(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    auto& tiling = plan.gemmTiling;
    if (IsIntactNn16Batch1024(p)) {
        tiling.mBlocks = 1;
        tiling.nBlocks = 1;
        tiling.singleCoreM = tiling.m;
        tiling.singleCoreN = tiling.n;
        tiling.totalTasks = static_cast<uint32_t>(p.batchCount);
        tiling.usedAicCoreNum = std::min<uint32_t>(p.cubeCoreNum, tiling.totalTasks);
    }
    const bool legacyTunedShape = p.m == p.n && p.n == p.k &&
                                  ((p.m == 256 && p.batchCount == 32) || (p.m == 512 && p.batchCount == 16) ||
                                   (p.m == 1024 && p.batchCount == 8));
    if (!legacyTunedShape && static_cast<uint32_t>(p.batchCount) > p.cubeCoreNum) {
        TuneHighBatchRealifiedGrid(p, tiling);
    }
    if (IsTransposeSpatialNn114(p)) {
        tiling.mBlocks = 2;
        tiling.nBlocks = 1;
        tiling.singleCoreM = CeilAlign((tiling.m + 1) / 2, GEMM_BATCHED_BASE_M);
        tiling.singleCoreN = tiling.n;
        tiling.totalTasks = static_cast<uint32_t>(p.batchCount) * 2;
        tiling.usedAicCoreNum = std::min<uint32_t>(p.cubeCoreNum, tiling.totalTasks);
    }
}

static bool IsNaturalNStripNn(const CgemmRealifiedParams& p)
{
    return p.transa == ACLBLAS_OP_N && p.transb == ACLBLAS_OP_N && p.m == p.n && p.n == p.k &&
           ((p.m == 421 && p.batchCount == 124) || (p.m == 747 && p.batchCount == 33));
}

static void SelectRectangularRealifiedGrid(
    const GemmBatchedGemmTilingData& tiling, uint32_t& bestMBlocks, uint32_t& bestNBlocks)
{
    constexpr uint32_t RECTANGULAR_SPATIAL = 32;
    const uint32_t mTiles = (tiling.m + GEMM_BATCHED_BASE_M - 1) / GEMM_BATCHED_BASE_M;
    const uint32_t nTiles = (tiling.n + GEMM_BATCHED_BASE_N - 1) / GEMM_BATCHED_BASE_N;
    uint64_t bestAspectDelta = UINT64_MAX;
    bestMBlocks = 1;
    bestNBlocks = RECTANGULAR_SPATIAL;
    for (uint32_t mBlocks = 1; mBlocks <= mTiles; ++mBlocks) {
        if (RECTANGULAR_SPATIAL % mBlocks != 0) {
            continue;
        }
        const uint32_t nBlocks = RECTANGULAR_SPATIAL / mBlocks;
        if (nBlocks > nTiles) {
            continue;
        }
        const uint32_t blockM = CeilAlign((tiling.m + mBlocks - 1) / mBlocks, GEMM_BATCHED_BASE_M);
        const uint32_t blockN = CeilAlign((tiling.n + nBlocks - 1) / nBlocks, GEMM_BATCHED_BASE_N);
        const uint64_t scaledM = static_cast<uint64_t>(blockM) * GEMM_BATCHED_DEFAULT_TILE_N;
        const uint64_t scaledN = static_cast<uint64_t>(blockN) * GEMM_BATCHED_DEFAULT_TILE_M;
        const uint64_t delta = scaledM > scaledN ? scaledM - scaledN : scaledN - scaledM;
        if (delta < bestAspectDelta) {
            bestAspectDelta = delta;
            bestMBlocks = mBlocks;
            bestNBlocks = nBlocks;
        }
    }
}

static void TuneNaturalAndRectangularGrid(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    auto& tiling = plan.gemmTiling;
    if (IsNaturalNStripNn(p)) {
        const uint32_t nBlocks = (tiling.n + GEMM_BATCHED_DEFAULT_TILE_N - 1) / GEMM_BATCHED_DEFAULT_TILE_N;
        SetRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, 1, nBlocks);
    }
    if (plan.rectangularNormalABatch32) {
        uint32_t mBlocks = 1;
        uint32_t nBlocks = 32;
        SelectRectangularRealifiedGrid(tiling, mBlocks, nBlocks);
        SetRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, mBlocks, nBlocks);
    }
}

static void TuneSparseNormalAGrid(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    auto& tiling = plan.gemmTiling;
    if (plan.sparseThreeRowNormalA) {
        SetRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, 1, plan.sparseThreeRowSpatial);
        tiling.singleCoreM = tiling.m;
    }
    if (plan.sparseTwoRowNormalA) {
        const uint32_t mBlocks = plan.splitTwoRowByN ? 1 : 2;
        const uint32_t nBlocks = plan.splitTwoRowByN ? plan.sparseTwoRowCols : 1;
        SetRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, mBlocks, nBlocks);
        tiling.singleCoreM = plan.splitTwoRowByN ? tiling.m : CeilAlign((tiling.m + 1) / 2, GEMM_BATCHED_BASE_M);
        tiling.singleCoreN =
            plan.splitTwoRowByN ?
                CeilAlign((tiling.n + plan.sparseTwoRowCols - 1) / plan.sparseTwoRowCols, GEMM_BATCHED_BASE_N) :
                tiling.n;
        tiling.totalTasks = static_cast<uint32_t>(p.batchCount) * plan.sparseTwoRowSpatial;
        tiling.usedAicCoreNum = std::min<uint32_t>(p.cubeCoreNum, tiling.totalTasks);
    }
    if (plan.sparseFourRowNormalA) {
        SetRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, 1, plan.sparseFourRowSpatial);
        tiling.singleCoreM = tiling.m;
    }
}

static void TuneTransposedAndChunkedGrid(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    auto& tiling = plan.gemmTiling;
    if (plan.tiledBothTransposed) {
        const uint32_t mBlocks = plan.fullTileTc1055 ? 5 : 1;
        const uint32_t nBlocks = plan.fullTileTc1055 ? 9 : plan.tiledBothTransposedSpatial;
        SetRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, mBlocks, nBlocks);
        tiling.singleCoreM = plan.fullTileTc1055 ? GEMM_BATCHED_DEFAULT_TILE_M : tiling.m;
        tiling.singleCoreN = plan.fullTileTc1055 ?
                                 GEMM_BATCHED_DEFAULT_TILE_N :
                                 CeilAlign(
                                     (tiling.n + plan.tiledBothTransposedSpatial - 1) / plan.tiledBothTransposedSpatial,
                                     GEMM_BATCHED_BASE_N);
    }
    if (plan.chunkedLeftPacked) {
        SetRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, 1, plan.chunkedLeftPackedSpatial);
        tiling.singleCoreM = tiling.m;
    }
    const bool fiveWayNc576 = p.transa == ACLBLAS_OP_N && p.transb != ACLBLAS_OP_N && p.m == 576 && p.n == 576 &&
                              p.k == 576 && p.batchCount == 31;
    if (fiveWayNc576) {
        SetRealifiedGrid(tiling, static_cast<uint32_t>(p.batchCount), p.cubeCoreNum, 5, 3);
    }
}

static bool IsLegacyCgemmRealifiedShape(const CgemmRealifiedParams& p)
{
    return p.m == p.n && p.n == p.k &&
           ((p.m == 256 && p.batchCount == 32) || (p.m == 512 && p.batchCount == 16) ||
            (p.m == 1024 && p.batchCount == 8));
}

static bool IsTransposedLegacy1024(const CgemmRealifiedParams& p)
{
    return p.m == 1024 && p.batchCount == 8 && (p.transa != ACLBLAS_OP_N || p.transb != ACLBLAS_OP_N);
}

static bool IsSmallBaseK32(const CgemmRealifiedParams& p)
{
    if (IsIntactNn16Batch1024(p)) {
        return true;
    }
    if (p.transa == ACLBLAS_OP_N || p.transb == ACLBLAS_OP_N) {
        return false;
    }
    return (p.m == 38 && p.n == 38 && p.k == 38 && p.batchCount == 235) ||
           (p.m == 39 && p.n == 39 && p.k == 39 && p.batchCount == 125);
}

static void TuneLegacyRealifiedGrid(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    if (!IsLegacyCgemmRealifiedShape(p)) {
        return;
    }
    auto& tiling = plan.gemmTiling;
    if (p.m == 256) {
        tiling.mBlocks = 4;
        tiling.nBlocks = 1;
        tiling.singleCoreM = tiling.m / tiling.mBlocks;
        tiling.singleCoreN = tiling.n;
        tiling.totalTasks = static_cast<uint32_t>(p.batchCount) * tiling.mBlocks;
        tiling.usedAicCoreNum = std::min<uint32_t>(p.cubeCoreNum, tiling.totalTasks);
        return;
    }
    const bool transposed1024 = IsTransposedLegacy1024(p);
    tiling.mBlocks = transposed1024 ? 1 : 7;
    tiling.nBlocks = transposed1024 ? 7 : 1;
    tiling.singleCoreM = transposed1024 ? tiling.m : CeilAlign((tiling.m + 6) / 7, GEMM_BATCHED_BASE_M);
    tiling.singleCoreN = transposed1024 ? CeilAlign((tiling.n + 6) / 7, GEMM_BATCHED_BASE_N) : tiling.n;
    tiling.totalTasks = static_cast<uint32_t>(p.batchCount) * 7;
    tiling.usedAicCoreNum = std::min<uint32_t>(p.cubeCoreNum, tiling.totalTasks);
}

static void ConfigureRealifiedBalancedSplit(const CgemmRealifiedParams& p, CgemmRealifiedPlan& plan)
{
    const bool transposedLegacy1024 = IsTransposedLegacy1024(p);
    const bool naturalNStripNn = IsNaturalNStripNn(p);
    const bool spatialMajorExact = IsTransposeSpatialNn114(p);
    const bool smallBaseK32 = IsSmallBaseK32(p);
    const bool legacyShape = IsLegacyCgemmRealifiedShape(p);
    auto& tiling = plan.gemmTiling;
    tiling.balancedSplit =
        smallBaseK32 ?
            6 :
            (transposedLegacy1024 ? 4 :
                                    ((plan.rectangularNormalABatch32 || naturalNStripNn || spatialMajorExact) ?
                                         2 :
                                         (legacyShape && p.m == 1024 ?
                                              2 :
                                              (legacyShape && p.m == 512 ?
                                                   3 :
                                                   (static_cast<uint32_t>(p.batchCount) <= p.cubeCoreNum &&
                                                            tiling.totalTasks > static_cast<uint32_t>(p.batchCount) ?
                                                        2 :
                                                        1)))));
}

static void LaunchMixedRealifiedGemm(const CgemmRealifiedParams& p, const CgemmRealifiedPlan& plan)
{
    const uint32_t maxMixedBatch = 11 * p.cubeCoreNum;
    const uint32_t mixedLaunches =
        plan.mixedTinyNn32 ? 1 : (static_cast<uint32_t>(p.batchCount) + maxMixedBatch - 1) / maxMixedBatch;
    const uint32_t totalMixedWaves = (static_cast<uint32_t>(p.batchCount) + p.cubeCoreNum - 1) / p.cubeCoreNum;
    for (uint32_t launch = 0; launch < mixedLaunches; ++launch) {
        const uint32_t batchBegin = std::min<uint32_t>(
            p.batchCount, static_cast<uint64_t>(totalMixedWaves) * launch / mixedLaunches * p.cubeCoreNum);
        const uint32_t batchEnd = launch + 1 == mixedLaunches ?
                                      static_cast<uint32_t>(p.batchCount) :
                                      std::min<uint32_t>(
                                          p.batchCount, static_cast<uint64_t>(totalMixedWaves) * (launch + 1) /
                                                            mixedLaunches * p.cubeCoreNum);
        const uint32_t chunkBatch = batchEnd - batchBegin;
        auto packTiling = plan.packTiling;
        auto gemmTiling = plan.gemmTiling;
        packTiling.batchCount = chunkBatch;
        gemmTiling.batchCount = chunkBatch;
        cgemm_batched_realify_mixed_do(
            std::min<uint32_t>(p.cubeCoreNum, chunkBatch), p.handle->stream,
            plan.aPtr + static_cast<uint64_t>(batchBegin) * sizeof(void*),
            plan.bPtr + static_cast<uint64_t>(batchBegin) * sizeof(void*),
            plan.cPtr + static_cast<uint64_t>(batchBegin) * sizeof(void*), plan.workspace, packTiling, gemmTiling);
    }
}

static void LaunchSplitWideKRealifiedGemm(
    const CgemmRealifiedParams& p, const CgemmRealifiedPlan& plan, const uint8_t* kernelA, const uint8_t* kernelB)
{
    auto halfTiling = plan.gemmTiling;
    halfTiling.k /= 2;
    uint8_t* secondA = plan.workspace + plan.alignedPtrBytes;
    uint8_t* secondB = plan.workspace + 2 * plan.alignedPtrBytes;
    uint8_t* tempSlots = plan.workspace + 3 * plan.alignedPtrBytes;
    uint8_t* tempData =
        plan.workspace + 4 * plan.alignedPtrBytes + plan.perBatchABytes * static_cast<uint64_t>(p.batchCount);
    gemm_batched_gemm_kernel_do(halfTiling.usedAicCoreNum, p.handle->stream, kernelA, kernelB, plan.cPtr, halfTiling);
    gemm_batched_gemm_kernel_do(halfTiling.usedAicCoreNum, p.handle->stream, secondA, secondB, tempSlots, halfTiling);
    GemmBatchedAlphaBetaTilingData addTiling{};
    addTiling.m = 2 * p.m;
    addTiling.n = p.n;
    addTiling.ldc = 2 * p.ldc;
    addTiling.tempRowStride = 2 * p.m;
    addTiling.alpha = 1.0f;
    addTiling.beta = 1.0f;
    addTiling.hasBeta = 1;
    addTiling.batchCount = p.batchCount;
    addTiling.dtypeCase = GEMM_BATCHED_DTYPE_FP32;
    addTiling.totalCols = static_cast<int64_t>(p.batchCount) * p.n;
    addTiling.usedAivCoreNum = CalcAivCoreNum(p.aivCoreNum, static_cast<int64_t>(p.batchCount) * 2 * p.m * p.n);
    gemm_batched_alpha_beta_kernel_do(
        static_cast<uint32_t>(addTiling.usedAivCoreNum), p.handle->stream, tempData, plan.cPtr, addTiling);
}

static void DispatchRealifiedGemm(const CgemmRealifiedParams& p, const CgemmRealifiedPlan& plan)
{
    const uint8_t* kernelA =
        plan.packB ? plan.workspace + plan.alignedPtrBytes : reinterpret_cast<const uint8_t*>(p.bArray);
    const uint8_t* kernelB = plan.workspace;
    if (plan.useMixedTranspose) {
        LaunchMixedRealifiedGemm(p, plan);
    } else if (plan.splitWideK) {
        LaunchSplitWideKRealifiedGemm(p, plan, kernelA, kernelB);
    } else {
        gemm_batched_gemm_kernel_do(
            plan.gemmTiling.usedAicCoreNum, p.handle->stream, kernelA, kernelB, plan.cPtr, plan.gemmTiling);
    }
}

static aclblasStatus_t LaunchCgemmRealified(
    _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, uint32_t cubeCoreNum, uint32_t aivCoreNum, const aclblasComplex* const Aarray[],
    const aclblasComplex* const Barray[], aclblasComplex* const Carray[])
{
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasCgemmBatched", "Cube core count must not be zero");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    CgemmRealifiedParams params{h,   transa,     transb,      m,          n,      k,      lda,   ldb,
                                ldc, batchCount, cubeCoreNum, aivCoreNum, Aarray, Barray, Carray};
    CgemmRealifiedPlan plan{};
    ConfigureRealifiedBasicPlan(params, plan);
    ConfigureSparseNormalAPlan(params, plan);
    ConfigureTransposedChunkPlan(params, plan);
    ConfigureChunkedLeftPlan(params, plan);
    plan.chunkedSpatialBatch = SelectRealifiedSpatialBatch(params, plan);

    aclblasStatus_t status = PrepareRealifiedWorkspace(params, plan);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (plan.useGenericPack) {
        ConfigureRealifiedPackTiling(params, plan);
        ConfigureMixedTranspose(params, plan);
    }
    LaunchRealifiedPack(params, plan);

    status = InitializeRealifiedGemmTiling(params, plan);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    TuneInitialRealifiedGrid(params, plan);
    TuneNaturalAndRectangularGrid(params, plan);
    TuneSparseNormalAGrid(params, plan);
    TuneTransposedAndChunkedGrid(params, plan);
    TuneLegacyRealifiedGrid(params, plan);
    ConfigureRealifiedBalancedSplit(params, plan);
    DispatchRealifiedGemm(params, plan);
    return ACLBLAS_STATUS_SUCCESS;
}
aclblasStatus_t aclblasCgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* const Aarray[], int lda, const aclblasComplex* const Barray[],
    int ldb, const aclblasComplex* beta, aclblasComplex* const Carray[], int ldc, int batchCount)
{
    OP_LOGI(
        "aclblasCgemmBatched", "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d, batch=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k, batchCount);

    aclblasStatus_t st = ValidateCgemmBatchedParams(
        handle, transa, transb, m, n, k, lda, ldb, ldc, batchCount, alpha, beta, Aarray, Barray, Carray);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    float alphaR = alpha->real;
    float alphaI = alpha->imag;
    float betaR = beta->real;
    float betaI = beta->imag;

    if (k == 0 || (alphaR == 0.0f && alphaI == 0.0f)) {
        if (betaR == 1.0f && betaI == 0.0f) {
            return ACLBLAS_STATUS_SUCCESS;
        }
        uint32_t aivCoreNum = GetAivCoreCount();
        if (aivCoreNum == 0) {
            OP_LOGE("aclblasCgemmBatched", "GetAivCoreCount failed");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        // The product term is absent on both quick-return paths.  In
        // particular, k == 0 must not evaluate 0 * alpha: a non-finite alpha
        // would otherwise turn the required C = beta * C result into NaN.
        return LaunchCgemmEarlyExit(h, m, n, ldc, batchCount, 0.0f, 0.0f, betaR, betaI, aivCoreNum, Carray);
    }

    uint32_t cubeCoreNum = 0;
    uint32_t aivCoreNum = 0;
    aclblasStatus_t coreRet = ValidateCoreCounts(cubeCoreNum, aivCoreNum);
    if (coreRet != ACLBLAS_STATUS_SUCCESS) {
        return coreRet;
    }

    // The SIMT path is valuable for truly tiny matrices, where setting up the
    // realified Cube layout costs more than the arithmetic.  At 16x16 the
    // per-output serial K loop is already the bottleneck, especially for large
    // batch counts, so hand the work to the strict-FP32 Cube path instead.
    // Keep very small, low-batch problems on SIMT: besides avoiding Cube setup,
    // its per-term complex arithmetic preserves IEEE NaN propagation for
    // overflowing finite inputs.
    if (ShouldUseCgemmDirectPath(m, n, k, batchCount)) {
        return LaunchCgemmDirect(
            h, transa, transb, m, n, k, lda, ldb, ldc, batchCount, alphaR, alphaI, betaR, betaI, aivCoreNum, Aarray,
            Barray, Carray);
    }

    if (CanUseCgemmRealifiedPath(transa, transb, m, n, k, lda, ldb, ldc, batchCount, alphaR, alphaI, betaR, betaI)) {
        return LaunchCgemmRealified(
            h, transa, transb, m, n, k, lda, ldb, ldc, batchCount, cubeCoreNum, aivCoreNum, Aarray, Barray, Carray);
    }

    return LaunchCgemmMainPath(
        h, transa, transb, m, n, k, lda, ldb, ldc, batchCount, alphaR, alphaI, betaR, betaI, cubeCoreNum, aivCoreNum,
        Aarray, Barray, Carray);
}
