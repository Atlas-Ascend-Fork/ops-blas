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
 * \file sgemm_ex_host.cpp
 * \brief FP32 GEMM host-side implementation (SIMD membase, BlockMmad).
 *
 * C = alpha * op(A) * op(B) + beta * C
 *
 * Column-major trick: swap A<->B, M<->N, transA<->transB so the Cube kernel
 * computes C'[n*m] = op(B')[n*k] * op(A')[k*m] in row-major (native Cube order).
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "sgemm_ex_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// ============================================================================
// Tiling computation
// ============================================================================

static void InitSgemmExTilingParams(
    SgemmExTilingData& tiling, int m, int n, int k, int lda, int ldb, int ldc, bool isTransA, bool isTransB,
    float alpha, float beta)
{
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.cLdc = ldc;
    tiling.isTransA = isTransA ? 1 : 0;
    tiling.isTransB = isTransB ? 1 : 0;
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.hasBeta = (beta != 0.0f) ? 1 : 0;
    tiling.baseM = 32;
    tiling.baseN = 16;
    tiling.baseK = 8;
}

static void CalcSgemmExMultiCorePartition(SgemmExTilingData& tiling, uint32_t cubeCoreNum, int m, int n)
{
    int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    int32_t mTiles = (m + tiling.baseM - 1) / tiling.baseM;
    int32_t nTiles = (n + tiling.baseN - 1) / tiling.baseN;
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
    tiling.usedCoreNum = bestMBlocks * bestNBlocks;
}

static void CalcSgemmExPerCoreWorkload(SgemmExTilingData& tiling, int m, int n)
{
    tiling.singleCoreM = (m + tiling.mBlocks - 1) / tiling.mBlocks;
    tiling.singleCoreM = ((tiling.singleCoreM + tiling.baseM - 1) / tiling.baseM) * tiling.baseM;
    if (tiling.singleCoreM > m) {
        tiling.singleCoreM = m;
    }
    tiling.singleCoreN = (n + tiling.nBlocks - 1) / tiling.nBlocks;
    tiling.singleCoreN = ((tiling.singleCoreN + tiling.baseN - 1) / tiling.baseN) * tiling.baseN;
    if (tiling.singleCoreN > n) {
        tiling.singleCoreN = n;
    }
}

static SgemmExTilingData CalcSgemmExTiling(
    uint32_t cubeCoreNum, int m, int n, int k, int lda, int ldb, int ldc, bool isTransA, bool isTransB, float alpha,
    float beta)
{
    SgemmExTilingData tiling{};
    InitSgemmExTilingParams(tiling, m, n, k, lda, ldb, ldc, isTransA, isTransB, alpha, beta);
    CalcSgemmExMultiCorePartition(tiling, cubeCoreNum, m, n);
    CalcSgemmExPerCoreWorkload(tiling, m, n);
    OP_LOGD(
        "aclblasSgemmEx",
        "tiling: M=%d, N=%d, K=%d, baseM=%d, baseN=%d, baseK=%d, "
        "singleCoreM=%d, singleCoreN=%d, mBlocks=%d, nBlocks=%d, usedCores=%d",
        m, n, k, tiling.baseM, tiling.baseN, tiling.baseK, tiling.singleCoreM, tiling.singleCoreN, tiling.mBlocks,
        tiling.nBlocks, tiling.usedCoreNum);
    return tiling;
}

// ============================================================================
// Edge case handling (k==0 or alpha==0: result = beta * C)
// ============================================================================

static bool HandleSgemmExEdgeCases(
    int k, float alphaVal, float betaVal, float* C, int ldc, int n, aclblasStatus_t& outStatus)
{
    if (k == 0 || alphaVal == 0.0f) {
        OP_LOGD("aclblasSgemmEx", "edge case: k=%d, alpha=%.4f, beta=%.4f", k, alphaVal, betaVal);
        if (C == nullptr) {
            OP_LOGD("aclblasSgemmEx", "C is nullptr, skipping computation");
            outStatus = ACLBLAS_STATUS_SUCCESS;
            return true;
        }
        if (betaVal == 0.0f) {
            size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);
            aclError ret = aclrtMemset(C, cBytes, 0, cBytes);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasSgemmEx", "aclrtMemset failed, ret=%d", ret);
                outStatus = ACLBLAS_STATUS_EXECUTION_FAILED;
            } else {
                outStatus = ACLBLAS_STATUS_SUCCESS;
            }
            return true;
        }
        if (betaVal == 1.0f) {
            outStatus = ACLBLAS_STATUS_SUCCESS;
            return true;
        }
        // beta != 0 && beta != 1: fall through to device kernel path
        // (alpha_beta kernel with alpha=0 computes C = 0*tempAB + beta*C = beta*C)
        return false;
    }
    if (C == nullptr) {
        OP_LOGD("aclblasSgemmEx", "C is nullptr with beta=0, skipping GEMM computation");
        outStatus = ACLBLAS_STATUS_SUCCESS;
        return true;
    }
    return false;
}

static aclblasStatus_t PrepareSgemmExWorkspace(
    _aclblas_handle* h, SgemmExTilingData& tilingData, bool needPostProcess, int m, int n, uint8_t*& tempABDevice)
{
    if (!needPostProcess) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    size_t tempABBytes = static_cast<size_t>(CeilAlign(m, 16)) * static_cast<size_t>(n) * sizeof(float);
    OP_LOGD(
        "aclblasSgemmEx", "workspace: tempABBytes=%zu, needPostProcess=%d, m=%d, n=%d, alignedM=%d", tempABBytes,
        needPostProcess, m, n, CeilAlign(m, 16));
    if (!CheckEffectiveWorkspaceSize(h, tempABBytes)) {
        OP_LOGE(
            "aclblasSgemmEx", "workspace not enough: need=%zu, have=%zu", tempABBytes, GetEffectiveWorkspaceSize(h));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    tempABDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    tilingData.ldc = static_cast<int32_t>(CeilAlign(m, 16));
    return ACLBLAS_STATUS_SUCCESS;
}

static void ApplyColumnMajorSwap(SgemmExTilingData& tilingData)
{
    std::swap(tilingData.m, tilingData.n);
    std::swap(tilingData.lda, tilingData.ldb);
    std::swap(tilingData.isTransA, tilingData.isTransB);
    std::swap(tilingData.singleCoreM, tilingData.singleCoreN);
    std::swap(tilingData.mBlocks, tilingData.nBlocks);
}

static void LaunchAlphaBetaKernel(
    _aclblas_handle* h, uint8_t* tempABDevice, float* C, const SgemmExTilingData& abTilingData, int m, int n,
    float alphaVal, float betaVal, uint32_t cubeCoreNum)
{
    uint8_t* cOrigPtr = reinterpret_cast<uint8_t*>(C);
    uint8_t* cOutPtr = reinterpret_cast<uint8_t*>(C);
    // Minimum elements per AIV core to amortize launch overhead.
    // 16384 = 64 * AB_TILE_SIZE(256): each core processes at least 64 tiles,
    // balancing parallelism (more cores for large matrices) vs launch cost
    // (avoid over-parallelizing small matrices).
    constexpr int64_t ELEMENTS_PER_CORE = 16384;
    int32_t neededCores =
        static_cast<int32_t>((static_cast<int64_t>(m) * n + ELEMENTS_PER_CORE - 1) / ELEMENTS_PER_CORE);
    if (neededCores < 1) {
        neededCores = 1;
    }
    uint32_t aivCores = static_cast<uint32_t>(std::min(static_cast<int32_t>(cubeCoreNum), neededCores));
    OP_LOGI(
        "aclblasSgemmEx", "launching alpha/beta kernel: alpha=%.4f, beta=%.4f, m=%d, n=%d, blocks=%u", alphaVal,
        betaVal, m, n, aivCores);
    sgemm_ex_alpha_beta_do(aivCores, h->stream, tempABDevice, cOrigPtr, cOutPtr, abTilingData);
}

static aclblasStatus_t LaunchSgemmExKernel(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    float alphaVal = *alpha;
    float betaVal = *beta;

    aclblasStatus_t edgeStatus;
    if (HandleSgemmExEdgeCases(k, alphaVal, betaVal, C, ldc, n, edgeStatus)) {
        return edgeStatus;
    }

    // beta-only path (k==0 or alpha==0, beta!=0 && beta!=1): skip cube kernel,
    // compute C = beta*C via alpha_beta kernel with alpha=0.
    bool isBetaOnly = (k == 0 || alphaVal == 0.0f);
    float alphaForPostProcess = isBetaOnly ? 0.0f : alphaVal;

    uint32_t cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasSgemmEx", "GetAicCoreCount failed, cube core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    bool isTransA = (transA != ACLBLAS_OP_N);
    bool isTransB = (transB != ACLBLAS_OP_N);
    SgemmExTilingData tilingData =
        CalcSgemmExTiling(cubeCoreNum, m, n, k, lda, ldb, ldc, isTransA, isTransB, alphaVal, betaVal);
    if (tilingData.usedCoreNum == 0) {
        OP_LOGE("aclblasSgemmEx", "invalid tiling data, usedCoreNum=0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    bool needPostProcess = (alphaVal != 1.0f) || (betaVal != 0.0f);
    uint8_t* tempABDevice = nullptr;
    aclblasStatus_t wsRet = PrepareSgemmExWorkspace(h, tilingData, needPostProcess, m, n, tempABDevice);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }

    SgemmExTilingData abTilingData = tilingData;
    ApplyColumnMajorSwap(tilingData);

    if (!isBetaOnly) {
        uint8_t* aDevicePtr = reinterpret_cast<uint8_t*>(const_cast<float*>(B));
        uint8_t* bDevicePtr = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
        uint8_t* cDevicePtr = needPostProcess ? tempABDevice : reinterpret_cast<uint8_t*>(C);

        OP_LOGI(
            "aclblasSgemmEx",
            "launching cube kernel: blocks=%d, transA=%d, transB=%d, alpha=%.4f, beta=%.4f, "
            "needPostProcess=%d, lda=%d, ldb=%d, ldc=%d",
            tilingData.usedCoreNum, isTransA, isTransB, alphaVal, betaVal, needPostProcess, tilingData.lda,
            tilingData.ldb, tilingData.ldc);

        sgemm_ex_kernel_do(
            static_cast<uint32_t>(tilingData.usedCoreNum), h->stream, aDevicePtr, bDevicePtr, cDevicePtr, tilingData);
    } else {
        // Zero tempAB to avoid NaN propagation (0 * NaN = NaN in IEEE 754)
        size_t tempABBytes = static_cast<size_t>(CeilAlign(m, 16)) * static_cast<size_t>(n) * sizeof(float);
        aclError ret = aclrtMemset(tempABDevice, tempABBytes, 0, tempABBytes);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasSgemmEx", "aclrtMemset tempAB failed, ret=%d", ret);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    if (needPostProcess) {
        abTilingData.alpha = alphaForPostProcess;
        LaunchAlphaBetaKernel(h, tempABDevice, C, abTilingData, m, n, alphaForPostProcess, betaVal, cubeCoreNum);
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Public API entry
// ============================================================================

#include "../gemm_ex_validation.h"
extern "C" aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const float* alpha,
    const void* a, aclDataType at, int lda, const void* b, aclDataType bt, int ldb, const float* beta, void* c,
    aclDataType ct, int ldc)
{
    const auto status = gemm_ex::Validate(
        handle, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc, ACL_FLOAT, "aclblasSgemmEx");
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0 || !c)
        return status;
    return LaunchSgemmExKernel(
        handle, ta, tb, m, n, k, alpha, static_cast<const float*>(a), lda, static_cast<const float*>(b), ldb, beta,
        static_cast<float*>(c), ldc);
}
