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
#include <cstdint>
#include <complex>
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "gemm_kernel.h"
#include "gemm_host_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// ==========================================================================
//  Tiling computation (mirrors gemm_ex for FP32)
// ==========================================================================

// ==========================================================================
//  ValidateGemmParams — common parameter validation
// ==========================================================================
static aclblasStatus_t ValidateGemmParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, int lda, int ldb, const void* beta, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasGemm", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) {
        OP_LOGE("aclblasGemm", "invalid transa: %d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C) {
        OP_LOGE("aclblasGemm", "invalid transb: %d", static_cast<int>(transb));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0) {
        OP_LOGE("aclblasGemm", "m, n, k must be non-negative: m=%d, n=%d, k=%d", m, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    int ldaMin = isTransA ? std::max(1, k) : std::max(1, m);
    if (lda < ldaMin) {
        OP_LOGE("aclblasGemm", "lda=%d must be >= %d", lda, ldaMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int ldbMin = isTransB ? std::max(1, n) : std::max(1, k);
    if (ldb < ldbMin) {
        OP_LOGE("aclblasGemm", "ldb=%d must be >= %d", ldb, ldbMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("aclblasGemm", "ldc=%d must be >= %d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasGemm", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclblasGemm", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGemmPointers(
    int k, float alphaAbs, const void* A, const void* B, float betaAbs, const void* C)
{
    if (k > 0 && alphaAbs != 0.0f) {
        if (A == nullptr) {
            OP_LOGE("aclblasGemm", "A must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (B == nullptr) {
            OP_LOGE("aclblasGemm", "B must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (k > 0) {
        if (C == nullptr) {
            OP_LOGE("aclblasGemm", "C must not be nullptr when k > 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (betaAbs != 0.0f && C == nullptr) {
        OP_LOGE("aclblasGemm", "C must not be nullptr when beta != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  HandleSgemmAlphaZero — C = beta * C when k == 0 or alpha == 0
// ==========================================================================
static aclblasStatus_t HandleSgemmAlphaZero(aclblasHandle_t handle, int m, int n, int ldc, float betaVal, float* C)
{
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 0.0f) {
        // issue #1154: 只清零逻辑 m 行（BLAS 语义仅定义 op(C) 的 m×n 区域），
        // 行区间 [m, ldc) 是调用方 C 缓冲的 padding 数据，必须保持原值
        // （对齐 gemm_strided_batched ZeroAllBatch 口径）。
        auto* cBase = reinterpret_cast<uint8_t*>(C);
        const size_t colStrideBytes = static_cast<size_t>(ldc) * sizeof(float);
        if (ldc == m) {
            size_t cBytes = colStrideBytes * n;
            aclError ret = aclrtMemset(C, cBytes, 0, cBytes);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasSgemm", "aclrtMemset failed, ret=%d", ret);
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
            return ACLBLAS_STATUS_SUCCESS;
        }
        const size_t rowBytes = static_cast<size_t>(m) * sizeof(float);
        for (int col = 0; col < n; col++) {
            uint8_t* colPtr = cBase + static_cast<size_t>(col) * colStrideBytes;
            aclError ret = aclrtMemset(colPtr, rowBytes, 0, rowBytes);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasSgemm", "aclrtMemset failed, col=%d, ret=%d", col, ret);
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    gemm_scale_do(vecBlocks, h->stream, reinterpret_cast<uint8_t*>(C), m, n, ldc, betaVal, 0.0f, 0);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  LaunchSgemmKernel — FP32 device kernel launch
// ==========================================================================
static aclblasStatus_t PrepareSgemmWorkspace(
    _aclblas_handle* h, bool needPostProcess, int m, int n, uint8_t*& tempABDevice)
{
    tempABDevice = nullptr;
    if (!needPostProcess) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    size_t workspaceNeed = static_cast<size_t>(CeilAlign(m, GEMM_FRACTAL)) * n * sizeof(float);
    if (!CheckEffectiveWorkspaceSize(h, workspaceNeed)) {
        OP_LOGE("aclblasSgemm", "workspace need %zu bytes", workspaceNeed);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    tempABDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (reinterpret_cast<uintptr_t>(tempABDevice) % 512 != 0) {
        OP_LOGW("aclblasSgemm", "workspace address not 512B aligned: %p", tempABDevice);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchAlphaBetaKernel(
    aclrtStream stream, uint8_t* tempABDevice, float* C, int n, const GemmTilingData& abTiling)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    GemmTilingData tiling = abTiling;
    tiling.usedCoreNum = static_cast<int32_t>(vecBlocks);
    OP_LOGI("aclblasSgemm", "launching alpha_beta kernel: aivBlocks=%u", vecBlocks);
    gemm_alpha_beta_do(
        vecBlocks, stream, tempABDevice, reinterpret_cast<uint8_t*>(C), reinterpret_cast<uint8_t*>(C), tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchSgemmKernel(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, float alphaVal,
    const float* A, int lda, const float* B, int ldb, float betaVal, float* C, int ldc)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    uint32_t cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasSgemm", "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (k == 0 || alphaVal == 0.0f) {
        return HandleSgemmAlphaZero(handle, m, n, ldc, betaVal, C);
    }

    bool needPostProcess = (alphaVal != 1.0f) || (betaVal != 0.0f);

    GemmTilingData tiling = BuildGemmTilingData(m, n, k, lda, ldb, ldc, transa, transb, alphaVal, 0.0f, betaVal, 0.0f);

    if (needPostProcess) {
        tiling.ldc = static_cast<int32_t>(CeilAlign(m, GEMM_FRACTAL));
    }
    GemmTilingData abTiling = tiling;

    PrepareGemmCubeTiling(tiling, cubeCoreNum);

    uint32_t numBlocks = static_cast<uint32_t>(tiling.usedCoreNum);
    uint8_t* aDevicePtr = reinterpret_cast<uint8_t*>(const_cast<float*>(B));
    uint8_t* bDevicePtr = reinterpret_cast<uint8_t*>(const_cast<float*>(A));

    uint8_t* tempABDevice = nullptr;
    aclblasStatus_t st = PrepareSgemmWorkspace(h, needPostProcess, m, n, tempABDevice);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    uint8_t* cDevicePtr = needPostProcess ? tempABDevice : reinterpret_cast<uint8_t*>(C);

    OP_LOGI("aclblasSgemm", "launching cube kernel: blocks=%u", numBlocks);
    gemm_kernel_do(numBlocks, stream, aDevicePtr, bDevicePtr, cDevicePtr, tiling);

    if (needPostProcess) {
        return LaunchAlphaBetaKernel(stream, tempABDevice, C, n, abTiling);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
// ==========================================================================
//  aclblasSgemm — public API entry (FP32)
// ==========================================================================
extern "C" aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    OP_LOGI(
        "aclblasSgemm", "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k);

    aclblasStatus_t st = ValidateGemmParams(handle, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    float alphaVal = *alpha;
    float betaVal = *beta;

    float alphaAbs = std::abs(alphaVal);
    st = ValidateGemmPointers(k, alphaAbs, A, B, betaVal, C);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    return LaunchSgemmKernel(handle, transa, transb, m, n, k, alphaVal, A, lda, B, ldb, betaVal, C, ldc);
}

#endif
