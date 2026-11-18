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
 * \file hgemm_batched_host.cpp
 * \brief Batched half-precision GEMM host implementation for arch35 (DAV_3510).
 *
 * C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],  FLOAT16 in/out.
 *
 * Reuses the same-family GEMM batch kernel (gemm_batched_*) defined in
 * gemm_batched_kernel.cpp: the Cube kernel accumulates FLOAT16 inputs into an
 * FP32 L0C and writes an FP32 temporary, and the AIV alpha/beta kernel applies
 * the scalar epilogue and casts back to FLOAT16.  This host layer only differs
 * from aclblasSgemmBatched in the element type (FLOAT16 pointers/scalars) and
 * the tiling dtype case.
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/dtype_cast.h"
#include "common/helper/host_utils.h"
#include "common/helper/aclblas_handle_internal.h"
#include "gemm_batched_tiling_data.h"
#include "gemm_batched_host_common.h"

constexpr const char* HB_OP_NAME = "aclblasHgemmBatched";

// Kernel launchers (defined in gemm_batched_kernel.cpp)
// FLOAT16-input / FLOAT16-output GEMM: casts the FP32 L0C accumulator to half
// directly on copy-out (no FP32 temporary + epilogue).
void gemm_batched_gemm_kernel_halfout_do(
    uint32_t numBlocks, void* stream, const uint8_t* aarray, const uint8_t* barray, uint8_t* carray,
    const GemmBatchedGemmTilingData& tilingData);

// ============================================================================
// Validation
// ============================================================================
static aclblasStatus_t ValidateHgemmBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, const aclblasHalf* alpha, const aclblasHalf* beta, const aclblasHalf* const Aarray[],
    const aclblasHalf* const Barray[], aclblasHalf* const Carray[])
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasHgemmBatched", "handle is nullptr");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasHgemmBatched", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasHgemmBatched", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasHgemmBatched", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasHgemmBatched", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasHgemmBatched", "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(batchCount >= 0, OP_LOGE("aclblasHgemmBatched", "batchCount must be >= 0, got %d", batchCount);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C,
              OP_LOGE("aclblasHgemmBatched", "invalid transa=%d", static_cast<int>(transa));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(transb == ACLBLAS_OP_N || transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C,
              OP_LOGE("aclblasHgemmBatched", "invalid transb=%d", static_cast<int>(transb));
              return ACLBLAS_STATUS_INVALID_ENUM);

    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int physRowsA = isTransA ? k : m;
    int physRowsB = isTransB ? n : k;
    CHECK_RET(lda >= std::max(1, physRowsA),
              OP_LOGE("aclblasHgemmBatched", "invalid lda=%d, expected>=%d", lda, std::max(1, physRowsA));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= std::max(1, physRowsB),
              OP_LOGE("aclblasHgemmBatched", "invalid ldb=%d, expected>=%d", ldb, std::max(1, physRowsB));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, m),
              OP_LOGE("aclblasHgemmBatched", "invalid ldc=%d, expected>=%d", ldc, std::max(1, m));
              return ACLBLAS_STATUS_INVALID_VALUE);

    if (batchCount > 0) {
        CHECK_RET(Aarray != nullptr, OP_LOGE("aclblasHgemmBatched", "Aarray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Barray != nullptr, OP_LOGE("aclblasHgemmBatched", "Barray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Carray != nullptr, OP_LOGE("aclblasHgemmBatched", "Carray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

struct HgemmLaunchContext {
    GemmBatchedGemmTilingData tilingData;
    uint32_t aivCoreNum = 0;
    int origM = 0;
    int origN = 0;
    int origLdc = 0;
    uint32_t tempRowStride = 0;
    uint8_t* workspace = nullptr;
    size_t alignedPtrArrayBytes = 0;
};

// Validate core counts, compute tiling, prepare the FP32 temporary workspace
// (only for the post-process path), then apply the column-major swap and output
// tile enlargement.  Fills the launch context used by ExecuteHgemmBatched.
static aclblasStatus_t PrepareHgemmLaunchContext(
    _aclblas_handle* h, bool needPostProcess, bool isTransA, bool isTransB, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, HgemmLaunchContext& ctx)
{
    uint32_t cubeCoreNum = 0;
    aclblasStatus_t coreRet = GemmBatchedValidateCoreCounts(HB_OP_NAME, cubeCoreNum, ctx.aivCoreNum);
    if (coreRet != ACLBLAS_STATUS_SUCCESS) {
        return coreRet;
    }
    // Keep the trailing-wave N-split balancing for the FLOAT16 path.
    constexpr bool useTrailingWaveBalance = true;
    aclblasStatus_t tilingRet = GemmBatchedCalcGemmTiling(
        HB_OP_NAME, GEMM_BATCHED_DTYPE_FP16, useTrailingWaveBalance, m, n, k, batchCount, lda, ldb, ldc,
        isTransA, isTransB, cubeCoreNum, ctx.tilingData);
    if (tilingRet != ACLBLAS_STATUS_SUCCESS) {
        return tilingRet;
    }
    ctx.origM = m;
    ctx.origN = n;
    ctx.origLdc = ldc;
    // The FP32 temporary uses the same L0C-column-aligned row stride as FP32 GEMM.
    ctx.tempRowStride = CeilAlign(static_cast<uint32_t>(ctx.origM), GEMM_BATCHED_L0C_C0);
    if (needPostProcess) {
        aclblasStatus_t wsRet = GemmBatchedPrepareTempWorkspace(
            HB_OP_NAME, h, batchCount, ctx.origN, ctx.tempRowStride, ctx.tilingData, ctx.workspace,
            ctx.alignedPtrArrayBytes);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            return wsRet;
        }
    }
    GemmBatchedApplyColumnMajorSwap(ctx.tilingData);
    GemmBatchedEnlargeOutputTileForLargeOperands(ctx.tilingData);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ExecuteHgemmBatched(
    _aclblas_handle* h, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasHalf* alpha, const aclblasHalf* const Aarray[], int lda, const aclblasHalf* const Barray[], int ldb,
    const aclblasHalf* beta, aclblasHalf* const Carray[], int ldc, int batchCount)
{
    float alphaVal = blas_common::HalfToFloat(*alpha);
    float betaVal = blas_common::HalfToFloat(*beta);
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    // Fast path: alpha == 1 and beta == 0.  The GEMM kernel writes FLOAT16
    // directly from the FP32 accumulator via the fixpipe cast, so no FP32
    // temporary and no separate epilogue pass are needed.
    // Otherwise the FP32 temporary + AIV alpha/beta (with FLOAT16 cast) is used.
    bool fastDirectOut = (alphaVal == 1.0f) && (betaVal == 0.0f);
    bool needPostProcess = !fastDirectOut;

    HgemmLaunchContext ctx;
    aclblasStatus_t prepRet = PrepareHgemmLaunchContext(
        h, needPostProcess, isTransA, isTransB, m, n, k, lda, ldb, ldc, batchCount, ctx);
    if (prepRet != ACLBLAS_STATUS_SUCCESS) {
        return prepRet;
    }
    GemmBatchedGemmTilingData& tilingData = ctx.tilingData;
    const uint32_t aivCoreNum = ctx.aivCoreNum;
    const int origM = ctx.origM;
    const int origN = ctx.origN;
    const int origLdc = ctx.origLdc;
    const uint32_t tempRowStride = ctx.tempRowStride;
    uint8_t* workspace = ctx.workspace;
    const size_t alignedPtrArrayBytes = ctx.alignedPtrArrayBytes;

    OP_LOGI(
        "aclblasHgemmBatched", "launching gemm kernel: cores=%u, transA=%d, transB=%d, batch=%d, m=%u, n=%u, k=%u",
        tilingData.usedAicCoreNum, isTransA, isTransB, batchCount, tilingData.m, tilingData.n, tilingData.k);

    const uint8_t* kernelAarray = reinterpret_cast<const uint8_t*>(Barray);
    const uint8_t* kernelBarray = reinterpret_cast<const uint8_t*>(Aarray);
    uint8_t* carrayPtr = needPostProcess ? workspace : const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));

    if (fastDirectOut) {
        gemm_batched_gemm_kernel_halfout_do(
            tilingData.usedAicCoreNum, h->stream, kernelAarray, kernelBarray, carrayPtr, tilingData);
    } else {
        gemm_batched_gemm_kernel_do(
            tilingData.usedAicCoreNum, h->stream, kernelAarray, kernelBarray, carrayPtr, tilingData);
    }

    if (needPostProcess) {
        uint8_t* carrayPtrAB = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));
        GemmBatchedLaunchAlphaBetaPostProcess(
            HB_OP_NAME, GEMM_BATCHED_DTYPE_FP16, h, origM, origN, origLdc, tempRowStride, alphaVal, betaVal,
            batchCount, aivCoreNum, workspace, alignedPtrArrayBytes, carrayPtrAB);
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Early exit: k=0 or alpha=0
// ============================================================================
static aclblasStatus_t LaunchEarlyExit(
    _aclblas_handle* h, int m, int n, int ldc, int batchCount, float betaVal, aclblasHalf* const Carray[])
{
    uint8_t* carrayPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(Carray));
    return GemmBatchedLaunchEarlyExit(
        HB_OP_NAME, GEMM_BATCHED_DTYPE_FP16, h, m, n, ldc, batchCount, betaVal, carrayPtr);
}

// ============================================================================
// Public API
// ============================================================================
aclblasStatus_t aclblasHgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasHalf* alpha, const aclblasHalf* const Aarray[], int lda, const aclblasHalf* const Barray[], int ldb,
    const aclblasHalf* beta, aclblasHalf* const Carray[], int ldc, int batchCount)
{
    OP_LOGI(
        "aclblasHgemmBatched", "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d, batch=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k, batchCount);

    aclblasStatus_t st = ValidateHgemmBatchedParams(
        handle, transa, transb, m, n, k, lda, ldb, ldc, batchCount, alpha, beta, Aarray, Barray, Carray);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    float alphaVal = blas_common::HalfToFloat(*alpha);
    float betaVal = blas_common::HalfToFloat(*beta);

    // Early exit: k=0 or alpha=0 degenerates to C = beta * C
    if (k == 0 || alphaVal == 0.0f) {
        return LaunchEarlyExit(h, m, n, ldc, batchCount, betaVal, Carray);
    }

    return ExecuteHgemmBatched(
        h, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
