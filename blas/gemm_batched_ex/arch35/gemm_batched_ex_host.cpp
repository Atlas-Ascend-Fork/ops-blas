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
 * \file gemm_batched_ex_host.cpp
 * \brief Batched GEMM host-side implementation (SIMD membase).
 *
 * C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
 *
 * Supported dtype combinations (7 total):
 * 1. FP16 pure: COMPUTE_16F + FP16/FP16/FP16 (alpha/beta as float, see README #1155)
 * 2. FP16 mixed: COMPUTE_32F + FP16/FP16/FP16
 * 3. BF16: COMPUTE_32F + BF16/BF16/BF16
 * 4-7. FP8: COMPUTE_32F + FP8(E4M3/E5M2) input + FP16 output
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"

#include "cann_ops_blas_common.h"
#include "gemm_batched_ex_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// ============================================================================
// Aggregated parameter structures (reduce function parameter count)
// ============================================================================

// Problem description (immutable GEMM parameters from aclblasGemmBatchedEx)
struct GemmBatchedProblem {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    int batchCount;
    aclDataType aType;
    aclDataType bType;
    aclDataType cType;
    aclblasComputeType_t computeType;
    aclblasGemmAlgo_t algo;
    aclblasOperation_t transA;
    aclblasOperation_t transB;
    float alpha;
    float beta;
};

// Runtime context (device/pointer/stream, built during execution phase)
struct GemmBatchedContext {
    void* stream;
    uint32_t cubeCoreNum;
    // G.FMT.14 requires a space between '*' and 'const'; clang-format (PointerAlignment: Left)
    // would rejoin them, so the three array members below keep manual formatting.
    // clang-format off
    const void* const *aarray;
    const void* const *barray;
    void* const *carray;
    // clang-format on
};

void gemm_batched_ex_kernel_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* carray,
    const GemmBatchedExTilingData& tilingData, bool isTransA, bool isTransB, GemmBatchedDTypeCase dtypeCase);

void gemm_batched_ex_alpha_beta_do(
    uint32_t numBlocks, void* stream, uint8_t* tempAB, uint8_t* carray, const GemmBatchedExTilingData& tilingData,
    GemmBatchedDTypeCase dtypeCase, bool useFP32Temp);

void gemm_batched_ex_early_exit_do(
    uint32_t numBlocks, void* stream, uint8_t* carray, const GemmBatchedExTilingData& tilingData,
    GemmBatchedDTypeCase dtypeCase);

// Read alpha/beta as float. issue #1155: the scalar type is float for every
// computeType (including COMPUTE_16F) — computeType does not change the scalar
// decoding, it only selects the accumulation precision.
static void ReadAlphaBeta(
    const void* alpha, const void* beta, aclblasComputeType_t computeType, float* alphaVal, float* betaVal)
{
    (void)computeType;
    *alphaVal = *static_cast<const float*>(alpha);
    *betaVal = *static_cast<const float*>(beta);
}

// Supported Atype/Btype/Ctype/computeType combinations, see README supported-dtype table
struct GemmBatchedDtypeCombo {
    aclDataType aType;
    aclDataType bType;
    aclDataType cType;
    aclblasComputeType_t computeType;
};

constexpr GemmBatchedDtypeCombo SUPPORTED_DTYPE_COMBOS[] = {
    {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACLBLAS_COMPUTE_16F},
    {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACLBLAS_COMPUTE_32F},
    {ACL_BF16, ACL_BF16, ACL_BF16, ACLBLAS_COMPUTE_32F},
    {ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, ACLBLAS_COMPUTE_32F},
    {ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT16, ACLBLAS_COMPUTE_32F},
    {ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT16, ACLBLAS_COMPUTE_32F},
    {ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, ACLBLAS_COMPUTE_32F},
};

static bool IsValidDtypeCombination(const GemmBatchedProblem& prob)
{
    for (const auto& combo : SUPPORTED_DTYPE_COMBOS) {
        if (prob.aType == combo.aType && prob.bType == combo.bType && prob.cType == combo.cType &&
            prob.computeType == combo.computeType) {
            return true;
        }
    }
    return false;
}

static bool IsFP8Type(aclDataType dtype) { return (dtype == ACL_FLOAT8_E4M3FN || dtype == ACL_FLOAT8_E5M2); }

// Dtype triple → kernel dtype case mapping, see GetDtypeCase fallthrough for unknown combos
struct GemmBatchedDtypeCaseEntry {
    aclDataType aType;
    aclDataType bType;
    aclDataType cType;
    GemmBatchedDTypeCase dtypeCase;
};

constexpr GemmBatchedDtypeCaseEntry DTYPE_CASE_TABLE[] = {
    {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, GEMM_BATCHED_DTYPE_FP16},
    {ACL_BF16, ACL_BF16, ACL_BF16, GEMM_BATCHED_DTYPE_BF16},
    {ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, GEMM_BATCHED_DTYPE_FP8_E4M3},
    {ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT16, GEMM_BATCHED_DTYPE_FP8_E5M2},
    {ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT16, GEMM_BATCHED_DTYPE_FP8_E5M2_E4M3},
    {ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT16, GEMM_BATCHED_DTYPE_FP8_E4M3_E5M2},
};

static GemmBatchedDTypeCase GetDtypeCase(const GemmBatchedProblem& prob)
{
    for (const auto& entry : DTYPE_CASE_TABLE) {
        if (prob.aType == entry.aType && prob.bType == entry.bType && prob.cType == entry.cType) {
            return entry.dtypeCase;
        }
    }
    return GEMM_BATCHED_DTYPE_FP16_OUT_F32;
}

// ============================================================================
// Validation sub-functions
// ============================================================================

static aclblasStatus_t ValidateBatchedDimensions(const GemmBatchedProblem& prob)
{
    CHECK_RET(prob.m >= 0, OP_LOGE("aclblasGemmBatchedEx", "invalid m=%d", prob.m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(prob.n >= 0, OP_LOGE("aclblasGemmBatchedEx", "invalid n=%d", prob.n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(prob.k >= 0, OP_LOGE("aclblasGemmBatchedEx", "invalid k=%d", prob.k);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(prob.transA == ACLBLAS_OP_N || prob.transA == ACLBLAS_OP_T || prob.transA == ACLBLAS_OP_C,
              OP_LOGE("aclblasGemmBatchedEx", "invalid transa=%d", static_cast<int>(prob.transA));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(prob.transB == ACLBLAS_OP_N || prob.transB == ACLBLAS_OP_T || prob.transB == ACLBLAS_OP_C,
              OP_LOGE("aclblasGemmBatchedEx", "invalid transb=%d", static_cast<int>(prob.transB));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(prob.batchCount >= 0, OP_LOGE("aclblasGemmBatchedEx", "invalid batchCount=%d", prob.batchCount);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateBatchedLeadingDims(const GemmBatchedProblem& prob)
{
    bool isTransA = (prob.transA != ACLBLAS_OP_N);
    bool isTransB = (prob.transB != ACLBLAS_OP_N);
    int expectedLda = isTransA ? std::max(1, prob.k) : std::max(1, prob.m);
    CHECK_RET(prob.lda >= expectedLda,
              OP_LOGE("aclblasGemmBatchedEx", "invalid lda=%d, expected=%d", prob.lda, expectedLda);
              return ACLBLAS_STATUS_INVALID_VALUE);
    int expectedLdb = isTransB ? std::max(1, prob.n) : std::max(1, prob.k);
    CHECK_RET(prob.ldb >= expectedLdb,
              OP_LOGE("aclblasGemmBatchedEx", "invalid ldb=%d, expected=%d", prob.ldb, expectedLdb);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(prob.ldc >= std::max(1, prob.m),
              OP_LOGE("aclblasGemmBatchedEx", "invalid ldc=%d, expected max(1, m=%d)", prob.ldc, prob.m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateBatchedDtypeAndPointers(
    const GemmBatchedProblem& prob, const void* const aarray[], const void* const barray[], void* const carray[])
{
    CHECK_RET(prob.algo == ACLBLAS_GEMM_DEFAULT,
              OP_LOGE("aclblasGemmBatchedEx", "unsupported algo=%d", static_cast<int>(prob.algo));
              return ACLBLAS_STATUS_NOT_SUPPORTED);
    CHECK_RET(IsValidDtypeCombination(prob),
              OP_LOGE(
                  "aclblasGemmBatchedEx", "invalid dtype combination: Atype=%d, Btype=%d, Ctype=%d",
                  static_cast<int>(prob.aType), static_cast<int>(prob.bType), static_cast<int>(prob.cType));
              return ACLBLAS_STATUS_NOT_SUPPORTED);
    if (IsFP8Type(prob.aType) || IsFP8Type(prob.bType)) {
        CHECK_RET(prob.computeType == ACLBLAS_COMPUTE_32F,
                  OP_LOGE("aclblasGemmBatchedEx", "FP8 input must use ACLBLAS_COMPUTE_32F");
                  return ACLBLAS_STATUS_NOT_SUPPORTED);
    }
    if (prob.batchCount > 0) {
        CHECK_RET(aarray != nullptr, OP_LOGE("aclblasGemmBatchedEx", "Aarray must not be nullptr when batchCount > 0");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(barray != nullptr, OP_LOGE("aclblasGemmBatchedEx", "Barray must not be nullptr when batchCount > 0");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(carray != nullptr, OP_LOGE("aclblasGemmBatchedEx", "Carray must not be nullptr when batchCount > 0");
                  return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGemmBatchedParams(
    aclblasHandle_t handle, const GemmBatchedProblem& prob, const void* alpha, const void* beta,
    const void* const aarray[], const void* const barray[], void* const carray[])
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasGemmBatchedEx", "handle is nullptr");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasGemmBatchedEx", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasGemmBatchedEx", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    aclblasStatus_t st = ValidateBatchedDimensions(prob);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateBatchedLeadingDims(prob);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return ValidateBatchedDtypeAndPointers(prob, aarray, barray, carray);
}

// ============================================================================
// Tiling computation sub-functions
// ============================================================================

static void InitBatchedTilingParams(GemmBatchedExTilingData& tiling, const GemmBatchedProblem& prob)
{
    tiling.m = prob.m;
    tiling.n = prob.n;
    tiling.k = prob.k;
    tiling.lda = prob.lda;
    tiling.ldb = prob.ldb;
    tiling.ldc = prob.ldc;
    tiling.isTransA = (prob.transA != ACLBLAS_OP_N) ? 1 : 0;
    tiling.isTransB = (prob.transB != ACLBLAS_OP_N) ? 1 : 0;
    tiling.alpha = prob.alpha;
    tiling.beta = prob.beta;
    tiling.hasBeta = (prob.beta != 0.0f) ? 1 : 0;
    tiling.batchCount = prob.batchCount;
    bool isFP8Input = IsFP8Type(prob.aType) || IsFP8Type(prob.bType);
    if (isFP8Input) {
        tiling.baseM = 32;
        tiling.baseN = 16;
        tiling.baseK = 32;
        tiling.c0Size = 32;
    } else {
        tiling.baseM = 128;
        tiling.baseN = 128;
        tiling.baseK = 16;
        tiling.c0Size = 16;
    }
    GemmBatchedDTypeCase dtypeCase = GetDtypeCase(prob);
    bool needPostProcess = (prob.alpha != 1.0f) || (prob.beta != 0.0f);
    bool useFP32Output =
        needPostProcess && (dtypeCase == GEMM_BATCHED_DTYPE_FP16 || dtypeCase == GEMM_BATCHED_DTYPE_BF16);
    tiling.outputFp32 = useFP32Output ? 1 : 0;
    tiling.cElemSize = (prob.cType == ACL_FLOAT) ? 4 : 2;
}

static void CalcBatchedMultiCorePartition(
    GemmBatchedExTilingData& tiling, const GemmBatchedProblem& prob, uint32_t cubeCoreNum)
{
    int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    int32_t mTiles = (prob.m + tiling.baseM - 1) / tiling.baseM;
    int32_t nTiles = (prob.n + tiling.baseN - 1) / tiling.baseN;
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = 1;
    int32_t bestUtilization = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= maxCores; mb++) {
        int32_t nb = std::min(nTiles, maxCores / mb);
        if (nb < 1) {
            nb = 1;
        }
        int32_t utilization = mb * nb;
        if (utilization > bestUtilization && utilization <= maxCores) {
            bestUtilization = utilization;
            bestMBlocks = mb;
            bestNBlocks = nb;
        }
    }
    if (bestUtilization == 0) {
        bestMBlocks = 1;
        bestNBlocks = 1;
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    int64_t totalTasks64 = static_cast<int64_t>(prob.batchCount) * bestMBlocks * bestNBlocks;
    tiling.usedCoreNum = static_cast<int32_t>(std::min(static_cast<int64_t>(cubeCoreNum), totalTasks64));
    tiling.totalTasks = static_cast<int32_t>(totalTasks64);
}

static void CalcBatchedPerCoreWorkload(GemmBatchedExTilingData& tiling, const GemmBatchedProblem& prob)
{
    tiling.singleCoreM = (prob.m + tiling.mBlocks - 1) / tiling.mBlocks;
    tiling.singleCoreM = ((tiling.singleCoreM + tiling.baseM - 1) / tiling.baseM) * tiling.baseM;
    if (tiling.singleCoreM > prob.m) {
        tiling.singleCoreM = prob.m;
    }
    tiling.singleCoreN = (prob.n + tiling.nBlocks - 1) / tiling.nBlocks;
    tiling.singleCoreN = ((tiling.singleCoreN + tiling.baseN - 1) / tiling.baseN) * tiling.baseN;
    if (tiling.singleCoreN > prob.n) {
        tiling.singleCoreN = prob.n;
    }
    OP_LOGD(
        "aclblasGemmBatchedEx",
        "tiling: M=%d, N=%d, K=%d, batch=%d, baseM=%d, baseN=%d, baseK=%d, "
        "singleCoreM=%d, singleCoreN=%d, mBlocks=%d, nBlocks=%d, totalTasks=%d, usedCores=%d",
        prob.m, prob.n, tiling.k, tiling.batchCount, tiling.baseM, tiling.baseN, tiling.baseK, tiling.singleCoreM,
        tiling.singleCoreN, tiling.mBlocks, tiling.nBlocks, tiling.totalTasks, tiling.usedCoreNum);
}

static GemmBatchedExTilingData CalcBatchedBlockMmadTiling(const GemmBatchedProblem& prob, uint32_t cubeCoreNum)
{
    GemmBatchedExTilingData tiling{};
    InitBatchedTilingParams(tiling, prob);
    CalcBatchedMultiCorePartition(tiling, prob, cubeCoreNum);
    CalcBatchedPerCoreWorkload(tiling, prob);
    return tiling;
}

// ── Tiling and execution helpers ──

static aclblasStatus_t PrepareTilingData(
    const GemmBatchedProblem& prob, uint32_t cubeCoreNum, GemmBatchedExTilingData& tilingData, bool& needPostProcess,
    GemmBatchedDTypeCase& dtypeCase, bool& useFP32Output)
{
    tilingData = CalcBatchedBlockMmadTiling(prob, cubeCoreNum);
    if (tilingData.usedCoreNum == 0) {
        OP_LOGE("aclblasGemmBatchedEx", "Invalid tiling data, usedCoreNum=0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    needPostProcess = (prob.alpha != 1.0f) || (prob.beta != 0.0f);
    dtypeCase = GetDtypeCase(prob);
    useFP32Output = needPostProcess && (dtypeCase == GEMM_BATCHED_DTYPE_FP16 || dtypeCase == GEMM_BATCHED_DTYPE_BF16);
    // ldc override and column-major swap are done by caller
    return ACLBLAS_STATUS_SUCCESS;
}

// ── Column-major swap, dtype resolution, and post-process helpers ──

static void ApplyColumnMajorSwap(GemmBatchedExTilingData& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
    std::swap(tiling.singleCoreM, tiling.singleCoreN);
    std::swap(tiling.mBlocks, tiling.nBlocks);
}

static GemmBatchedDTypeCase ResolveKernelDtypeCase(GemmBatchedDTypeCase dtypeCase, bool useFP32Output)
{
    if (useFP32Output) {
        return (dtypeCase == GEMM_BATCHED_DTYPE_BF16) ? GEMM_BATCHED_DTYPE_BF16_OUT_F32 :
                                                        GEMM_BATCHED_DTYPE_FP16_OUT_F32;
    }
    return dtypeCase;
}

static void LaunchAlphaBetaPostProcess(
    void* stream, uint8_t* workspace, size_t ptrArrayBytes, const GemmBatchedProblem& prob,
    const GemmBatchedExTilingData& abTilingData, GemmBatchedDTypeCase dtypeCase, bool useFP32Output,
    uint32_t cubeCoreNum, void* const carray[])
{
    uint8_t* tempABData = workspace + ptrArrayBytes;
    uint8_t* origCarrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(carray));

    int64_t totalElements = static_cast<int64_t>(prob.batchCount) * prob.m * prob.n;
    constexpr int64_t ELEMENTS_PER_CORE = 16384;
    uint32_t alphaBetaCores = static_cast<uint32_t>(std::min(
        static_cast<int64_t>(cubeCoreNum),
        std::max(static_cast<int64_t>(1), (totalElements + ELEMENTS_PER_CORE - 1) / ELEMENTS_PER_CORE)));

    OP_LOGI(
        "aclblasGemmBatchedEx", "launching alpha/beta kernel: cores=%d, alpha=%.4f, beta=%.4f, batch=%d, m=%d, n=%d",
        alphaBetaCores, prob.alpha, prob.beta, prob.batchCount, prob.m, prob.n);
    gemm_batched_ex_alpha_beta_do(
        alphaBetaCores, stream, tempABData, origCarrayPtr, abTilingData, dtypeCase, useFP32Output);
}

// Main execution path: tiling → execute → alpha/beta post-process
// Workspace layout (when needPostProcess):
//   [0 .. ptrArrayBytes)                                      : temp C pointer array (H2D)
//   [ptrArrayBytes .. ptrArrayBytes + batchCount*perBatchBytes): temp AB data
// Launch the (swapped) main kernel, then the alpha/beta post-process when needed
static void LaunchBatchedMainKernel(
    const GemmBatchedProblem& prob, const GemmBatchedContext& ctx, const GemmBatchedExTilingData& tilingData,
    const GemmBatchedExTilingData& abTilingData, GemmBatchedDTypeCase dtypeCase, GemmBatchedDTypeCase kernelDtypeCase,
    bool needPostProcess, bool useFP32Output, uint8_t* workspace, size_t ptrArrayBytes)
{
    // Determine pointer arguments (column-major swap: aarray↔barray)
    uint8_t* aarrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(ctx.barray));
    uint8_t* barrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(ctx.aarray));
    uint8_t* carrayPtr;
    if (needPostProcess) {
        carrayPtr = workspace; // Temp pointer array at workspace start (set up by init_workspace)
    } else {
        carrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(ctx.carray));
    }

    bool isTransA = (prob.transA != ACLBLAS_OP_N);
    bool isTransB = (prob.transB != ACLBLAS_OP_N);

    OP_LOGI(
        "aclblasGemmBatchedEx",
        "launching kernel: blocks=%d, transA=%d, transB=%d, dtypeCase=%d, "
        "needPostProcess=%d, batch=%d",
        tilingData.usedCoreNum, isTransA, isTransB, static_cast<int>(kernelDtypeCase), needPostProcess,
        prob.batchCount);
    gemm_batched_ex_kernel_do(
        tilingData.usedCoreNum, ctx.stream, aarrayPtr, barrayPtr, carrayPtr, tilingData, isTransA, isTransB,
        kernelDtypeCase);

    if (needPostProcess) {
        LaunchAlphaBetaPostProcess(
            ctx.stream, workspace, ptrArrayBytes, prob, abTilingData, dtypeCase, useFP32Output, ctx.cubeCoreNum,
            ctx.carray);
    }
}

static aclblasStatus_t ExecuteGemmBatchedMainPath(
    aclblasHandle_t handle, const GemmBatchedProblem& prob, GemmBatchedContext& ctx, bool needPostProcess,
    uint8_t* workspace, size_t ptrArrayBytes)
{
    auto* h = handle;
    ctx.stream = h->stream;
    ctx.cubeCoreNum = GetAicCoreCount();
    if (ctx.cubeCoreNum == 0) {
        OP_LOGE("aclblasGemmBatchedEx", "Failed to get cube core count");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    GemmBatchedExTilingData tilingData{};
    GemmBatchedDTypeCase dtypeCase = GEMM_BATCHED_DTYPE_FP16;
    bool useFP32Output = false;
    aclblasStatus_t st =
        PrepareTilingData(prob, ctx.cubeCoreNum, tilingData, needPostProcess, dtypeCase, useFP32Output);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Save pre-swap tiling for alpha/beta kernel (original m, n, ldc)
    GemmBatchedExTilingData abTilingData = tilingData;

    if (needPostProcess) {
        tilingData.ldc = prob.m; // Packed temp output (stride = m)
    }

    ApplyColumnMajorSwap(tilingData);

    GemmBatchedDTypeCase kernelDtypeCase = ResolveKernelDtypeCase(dtypeCase, useFP32Output);
    LaunchBatchedMainKernel(
        prob, ctx, tilingData, abTilingData, dtypeCase, kernelDtypeCase, needPostProcess, useFP32Output, workspace,
        ptrArrayBytes);

    return ACLBLAS_STATUS_SUCCESS;
}

// ── Early-exit and workspace setup helpers ──

static aclblasStatus_t LaunchEarlyExitKernel(
    aclblasHandle_t handle, const GemmBatchedProblem& prob, void* const carray[])
{
    if (prob.beta == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS; // C unchanged
    }
    OP_LOGI(
        "aclblasGemmBatchedEx", "k=%d, alpha=%.4f, beta=%.4f, early exit via device kernel", prob.k, prob.alpha,
        prob.beta);
    GemmBatchedExTilingData earlyTiling{};
    earlyTiling.m = prob.m;
    earlyTiling.n = prob.n;
    earlyTiling.ldc = prob.ldc;
    earlyTiling.beta = prob.beta;
    earlyTiling.hasBeta = (prob.beta != 0.0f) ? 1 : 0;
    earlyTiling.batchCount = prob.batchCount;
    earlyTiling.cElemSize = (prob.cType == ACL_FLOAT) ? 4 : 2;

    auto* h = handle;
    constexpr int64_t ELEMENTS_PER_CORE = 16384;
    int64_t totalElements = static_cast<int64_t>(prob.batchCount) * prob.m * prob.n;
    uint32_t numBlocks = static_cast<uint32_t>(std::min(
        static_cast<int64_t>(GetAicCoreCount()),
        std::max(static_cast<int64_t>(1), (totalElements + ELEMENTS_PER_CORE - 1) / ELEMENTS_PER_CORE)));

    uint8_t* carrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(carray));
    GemmBatchedDTypeCase dtypeCase = GetDtypeCase(prob);
    gemm_batched_ex_early_exit_do(numBlocks, h->stream, carrayPtr, earlyTiling, dtypeCase);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t SetupPostProcessWorkspace(
    aclblasHandle_t handle, const GemmBatchedProblem& prob, bool useFP32Output, uint8_t*& workspace,
    size_t& ptrArrayBytes, size_t& perBatchBytes)
{
    auto* h = handle;
    size_t tempElemSize = useFP32Output ? sizeof(float) : sizeof(uint16_t);
    perBatchBytes = static_cast<size_t>(prob.m) * prob.n * tempElemSize;
    ptrArrayBytes = static_cast<size_t>(prob.batchCount) * sizeof(void*);
    // Workspace layout: [tempPtrArray | tempABData]
    size_t totalWorkspaceBytes = ptrArrayBytes + static_cast<size_t>(prob.batchCount) * perBatchBytes;

    if (!CheckEffectiveWorkspaceSize(h, totalWorkspaceBytes)) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    // Build temp C pointer array on host, then copy to workspace
    std::vector<uint64_t> tempPtrs(prob.batchCount);
    uint64_t tempABBase = reinterpret_cast<uint64_t>(workspace + ptrArrayBytes);
    for (int i = 0; i < prob.batchCount; i++) {
        tempPtrs[i] = tempABBase + static_cast<uint64_t>(i) * perBatchBytes;
    }
    aclError aclRet = aclrtMemcpy(workspace, ptrArrayBytes, tempPtrs.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasGemmBatchedEx", "Failed to copy temp ptr array, err=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* const aarray[], aclDataType aType, int lda, const void* const barray[],
    aclDataType bType, int ldb, const void* beta, void* const carray[], aclDataType cType, int ldc, int batchCount,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
{
    OP_LOGI(
        "aclblasGemmBatchedEx",
        "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d, batch=%d, "
        "Atype=%d, Btype=%d, Ctype=%d, computeType=%d",
        static_cast<int>(transa), static_cast<int>(transb), m, n, k, batchCount, static_cast<int>(aType),
        static_cast<int>(bType), static_cast<int>(cType), static_cast<int>(computeType));
    GemmBatchedProblem prob{m,     n,     k,           lda,  ldb,    ldc,    batchCount, aType,
                            bType, cType, computeType, algo, transa, transb, 0.0f,       0.0f};
    aclblasStatus_t st = ValidateGemmBatchedParams(handle, prob, alpha, beta, aarray, barray, carray);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasGemmBatchedEx", "parameter validation failed, status=%d", static_cast<int>(st));
        return st;
    }
    ReadAlphaBeta(alpha, beta, computeType, &prob.alpha, &prob.beta);
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Early-exit paths (k=0 or alpha=0): handle via device kernel
    if (k == 0 || prob.alpha == 0.0f) {
        return LaunchEarlyExitKernel(handle, prob, carray);
    }

    // Main execution path: set up workspace if needed, then launch kernels
    bool needPostProcess = (prob.alpha != 1.0f) || (prob.beta != 0.0f);
    GemmBatchedDTypeCase dtypeCase = GetDtypeCase(prob);
    bool useFP32Output =
        needPostProcess && (dtypeCase == GEMM_BATCHED_DTYPE_FP16 || dtypeCase == GEMM_BATCHED_DTYPE_BF16);

    uint8_t* workspace = nullptr;
    size_t ptrArrayBytes = 0;
    size_t perBatchBytes = 0;
    if (needPostProcess) {
        st = SetupPostProcessWorkspace(handle, prob, useFP32Output, workspace, ptrArrayBytes, perBatchBytes);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }

    GemmBatchedContext ctx{};
    ctx.aarray = aarray;
    ctx.barray = barray;
    ctx.carray = carray;

    return ExecuteGemmBatchedMainPath(handle, prob, ctx, needPostProcess, workspace, ptrArrayBytes);
}
