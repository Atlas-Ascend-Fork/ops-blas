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
 * \file gemm_ex_host.cpp
 * \brief General matrix multiply (GEMM) host-side implementation (SIMD membase).
 *
 * Phase 2: BlockMmad low-level API. Manual tiling computation (no MultiCoreMatmulTiling).
 * C = alpha * op(A) * op(B) + beta * C
 *
 * Supported dtype combinations:
 * 1. FP16 pure: COMPUTE_16F + FP16/FP16/FP16
 * 2. FP16 mixed: COMPUTE_32F + FP16/FP16/FP16
 * 3. BF16 pure: COMPUTE_32F + BF16/BF16/BF16
 * 5-8. FP8: COMPUTE_32F + FP8(E4M3/E5M2) input + FP16 output
 */

#include <algorithm>
#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "gemm_ex_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/dtype_cast.h"

void gemm_ex_kernel_do(
    uint32_t numBlocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, const GemmExTilingData& tilingData,
    bool isTransA, bool isTransB, GemmDTypeCase dtypeCase);

void gemm_ex_alpha_beta_do(
    uint32_t numBlocks, void* stream, uint8_t* tempAB, uint8_t* cOrig, uint8_t* cOut,
    const GemmExTilingData& tilingData, GemmDTypeCase dtypeCase, bool useFP32Temp);

static bool IsValidDtypeCombination(
    aclDataType Atype, aclDataType Btype, aclDataType Ctype, aclblasComputeType_t computeType)
{
    switch (computeType) {
        case ACLBLAS_COMPUTE_16F:
            return (Atype == ACL_FLOAT16 && Btype == ACL_FLOAT16 && Ctype == ACL_FLOAT16);
        case ACLBLAS_COMPUTE_32F:
            switch (Atype) {
                case ACL_FLOAT16:
                    return (Btype == ACL_FLOAT16 && Ctype == ACL_FLOAT16);
                case ACL_BF16:
                    return (Btype == ACL_BF16 && Ctype == ACL_BF16);
                case ACL_FLOAT:
                    return (Btype == ACL_FLOAT && Ctype == ACL_FLOAT);
                case ACL_FLOAT8_E4M3FN:
                    return ((Btype == ACL_FLOAT8_E4M3FN || Btype == ACL_FLOAT8_E5M2) && Ctype == ACL_FLOAT16);
                case ACL_FLOAT8_E5M2:
                    return ((Btype == ACL_FLOAT8_E5M2 || Btype == ACL_FLOAT8_E4M3FN) && Ctype == ACL_FLOAT16);
                default:
                    return false;
            }
        default:
            return false;
    }
}

static bool IsFP8Type(aclDataType dtype) { return (dtype == ACL_FLOAT8_E4M3FN || dtype == ACL_FLOAT8_E5M2); }

static GemmDTypeCase GetDtypeCase(aclDataType Atype, aclDataType Btype, aclDataType Ctype)
{
    switch (Atype) {
        case ACL_FLOAT16:
            return (Btype == ACL_FLOAT16 && Ctype == ACL_FLOAT16) ? GEMM_DTYPE_FP16 : GEMM_DTYPE_INVALID;
        case ACL_BF16:
            return (Btype == ACL_BF16 && Ctype == ACL_BF16) ? GEMM_DTYPE_BF16 : GEMM_DTYPE_INVALID;
        case ACL_FLOAT:
            return (Btype == ACL_FLOAT && Ctype == ACL_FLOAT) ? GEMM_DTYPE_FP32 : GEMM_DTYPE_INVALID;
        case ACL_FLOAT8_E4M3FN:
            if (Btype == ACL_FLOAT8_E4M3FN && Ctype == ACL_FLOAT16) {
                return GEMM_DTYPE_FP8_E4M3;
            } else if (Btype == ACL_FLOAT8_E5M2 && Ctype == ACL_FLOAT16) {
                return GEMM_DTYPE_FP8_E5M2_E4M3;
            }
            return GEMM_DTYPE_INVALID;
        case ACL_FLOAT8_E5M2:
            if (Btype == ACL_FLOAT8_E5M2 && Ctype == ACL_FLOAT16) {
                return GEMM_DTYPE_FP8_E5M2;
            } else if (Btype == ACL_FLOAT8_E4M3FN && Ctype == ACL_FLOAT16) {
                return GEMM_DTYPE_FP8_E4M3_E5M2;
            }
            return GEMM_DTYPE_INVALID;
        default:
            return GEMM_DTYPE_INVALID;
    }
}

// ============================================================================
// Validation sub-functions (P1-5: split ValidateGemmParams)
// ============================================================================

static aclblasStatus_t ValidateDimensions(aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k)
{
    CHECK_RET(m >= 0, OP_LOGE("aclblasGemmEx", "invalid m=%d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasGemmEx", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasGemmEx", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C,
              OP_LOGE("aclblasGemmEx", "invalid transa=%d", static_cast<int>(transa));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transb == ACLBLAS_OP_N || transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C,
              OP_LOGE("aclblasGemmEx", "invalid transb=%d", static_cast<int>(transb));
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateLeadingDims(bool isTransA, bool isTransB, int m, int n, int k, int lda, int ldb, int ldc)
{
    int expectedLda = isTransA ? std::max(1, k) : std::max(1, m);
    CHECK_RET(lda >= expectedLda, OP_LOGE("aclblasGemmEx", "invalid lda=%d, expected=%d", lda, expectedLda);
              return ACLBLAS_STATUS_INVALID_VALUE);
    int expectedLdb = isTransB ? std::max(1, n) : std::max(1, k);
    CHECK_RET(ldb >= expectedLdb, OP_LOGE("aclblasGemmEx", "invalid ldb=%d, expected=%d", ldb, expectedLdb);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, m), OP_LOGE("aclblasGemmEx", "invalid ldc=%d, expected max(1, m=%d)", ldc, m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateDtypeAndPointers(
    aclDataType Atype, aclDataType Btype, aclDataType Ctype, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo,
    const void* alpha, const void* beta, int m, int n, int k, const void* A, const void* B, void* C)
{
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasGemmEx", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasGemmEx", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(algo == ACLBLAS_GEMM_DEFAULT, OP_LOGE("aclblasGemmEx", "unsupported algo=%d", static_cast<int>(algo));
              return ACLBLAS_STATUS_NOT_SUPPORTED);
    CHECK_RET(IsValidDtypeCombination(Atype, Btype, Ctype, computeType),
              OP_LOGE(
                  "aclblasGemmEx", "invalid dtype combination: Atype=%d, Btype=%d, Ctype=%d", static_cast<int>(Atype),
                  static_cast<int>(Btype), static_cast<int>(Ctype));
              return ACLBLAS_STATUS_NOT_SUPPORTED);
    if (IsFP8Type(Atype) || IsFP8Type(Btype)) {
        CHECK_RET(computeType == ACLBLAS_COMPUTE_32F,
                  OP_LOGE("aclblasGemmEx", "FP8 input must use ACLBLAS_COMPUTE_32F");
                  return ACLBLAS_STATUS_NOT_SUPPORTED);
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const float betaVal = *static_cast<const float*>(beta);
    const float alphaVal = *static_cast<const float*>(alpha);
    if (k > 0 && alphaVal != 0.0f) {
        CHECK_RET(A != nullptr, OP_LOGE("aclblasGemmEx", "A must not be nullptr when k > 0 and alpha != 0");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(B != nullptr, OP_LOGE("aclblasGemmEx", "B must not be nullptr when k > 0 and alpha != 0");
                  return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(C != nullptr || betaVal == 0.0f, OP_LOGE("aclblasGemmEx", "C must not be nullptr when beta != 0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGemmParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, const void* alpha, const void* beta, aclDataType Atype, aclDataType Btype, aclDataType Ctype,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo, const void* A, const void* B, void* C)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasGemmEx", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    aclblasStatus_t st = ValidateDimensions(transa, transb, m, n, k);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    st = ValidateLeadingDims(isTransA, isTransB, m, n, k, lda, ldb, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return ValidateDtypeAndPointers(Atype, Btype, Ctype, computeType, algo, alpha, beta, m, n, k, A, B, C);
}

// ============================================================================
// Tiling computation sub-functions (P1-4: split CalcBlockMmadTiling)
// ============================================================================

static void InitTilingParams(
    GemmExTilingData& tiling, int m, int n, int k, int lda, int ldb, int ldc, aclDataType Atype, aclDataType Btype,
    aclDataType Ctype, bool isTransA, bool isTransB, float alpha, float beta)
{
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.isTransA = isTransA ? 1 : 0;
    tiling.isTransB = isTransB ? 1 : 0;
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.hasBeta = (beta != 0.0f) ? 1 : 0;
    bool isFP32Input = (Atype == ACL_FLOAT && Btype == ACL_FLOAT);
    bool isFP8Input = IsFP8Type(Atype) || IsFP8Type(Btype);
    if (isFP32Input) {
        tiling.baseM = 32;
        tiling.baseN = 16;
        tiling.baseK = 8;
        tiling.c0Size = 8;
    } else if (isFP8Input) {
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
    GemmDTypeCase dtypeCase = GetDtypeCase(Atype, Btype, Ctype);
    bool needPostProcess = (alpha != 1.0f) || (beta != 0.0f);
    bool useFP32Output = needPostProcess && (dtypeCase == GEMM_DTYPE_FP16 || dtypeCase == GEMM_DTYPE_BF16);
    tiling.outputFp32 = useFP32Output ? 1 : 0;
}

static void CalcMultiCorePartition(GemmExTilingData& tiling, uint32_t cubeCoreNum, int m, int n)
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

static void CalcPerCoreWorkload(GemmExTilingData& tiling, int m, int n)
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
    OP_LOGD(
        "aclblasGemmEx",
        "BlockMmad tiling: M=%d, N=%d, K=%d, baseM=%d, baseN=%d, baseK=%d, "
        "singleCoreM=%d, singleCoreN=%d, mBlocks=%d, nBlocks=%d, usedCores=%d",
        m, n, tiling.k, tiling.baseM, tiling.baseN, tiling.baseK, tiling.singleCoreM, tiling.singleCoreN,
        tiling.mBlocks, tiling.nBlocks, tiling.usedCoreNum);
}

// Manual tiling computation for BlockMmad low-level API
static GemmExTilingData CalcBlockMmadTiling(
    uint32_t cubeCoreNum, int m, int n, int k, int lda, int ldb, int ldc, aclDataType Atype, aclDataType Btype,
    aclDataType Ctype, bool isTransA, bool isTransB, float alpha, float beta)
{
    GemmExTilingData tiling{};
    InitTilingParams(tiling, m, n, k, lda, ldb, ldc, Atype, Btype, Ctype, isTransA, isTransB, alpha, beta);
    CalcMultiCorePartition(tiling, cubeCoreNum, m, n);
    CalcPerCoreWorkload(tiling, m, n);
    return tiling;
}

// ============================================================================
// Kernel launch sub-functions (P0-3: split aclblasGemmEx)
// ============================================================================

static aclblasStatus_t HandleKZero(void* C, int ldc, int n, aclDataType Ctype, float betaVal)
{
    // k == 0 且 beta == 0 时契约允许 C 为 nullptr：此时无需写 C，直接成功返回，
    // 避免对空指针执行 aclrtMemset/aclrtMemcpy（issue #355）
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 0.0f) {
        size_t elemSizeC = (Ctype == ACL_FLOAT) ? 4 : 2;
        size_t cBytes = static_cast<size_t>(ldc) * n * elemSizeC;
        aclError aclRet = aclrtMemset(C, cBytes, 0, cBytes);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasGemmEx", "aclrtMemset failed for C (k=0, beta=0), ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else if (betaVal != 1.0f) {
        size_t elemSizeC = (Ctype == ACL_FLOAT) ? 4 : 2;
        size_t cBytes = static_cast<size_t>(ldc) * n * elemSizeC;
        size_t cCount = static_cast<size_t>(ldc) * n;
        std::vector<uint8_t> cHost(cBytes);
        aclError aclRet = aclrtMemcpy(cHost.data(), cBytes, C, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasGemmEx", "aclrtMemcpy failed for C (k=0), ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        if (Ctype == ACL_FLOAT) {
            float* cData = reinterpret_cast<float*>(cHost.data());
            for (size_t i = 0; i < cCount; i++) {
                cData[i] *= betaVal;
            }
        } else {
            uint16_t* cData = reinterpret_cast<uint16_t*>(cHost.data());
            for (size_t i = 0; i < cCount; i++) {
                float cVal =
                    (Ctype == ACL_FLOAT16) ? blas_common::HalfToFloat(cData[i]) : blas_common::Bf16ToFloat(cData[i]);
                cVal *= betaVal;
                cData[i] = (Ctype == ACL_FLOAT16) ? blas_common::FloatToHalf(cVal) : blas_common::FloatToBf16(cVal);
            }
        }
        aclRet = aclrtMemcpy(C, cBytes, cHost.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasGemmEx", "aclrtMemcpy H2D failed for C (k=0), ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static bool HandleAlphaZero(void* C, int ldc, int n, aclDataType Ctype, float betaVal, aclblasStatus_t& outStatus)
{
    if (betaVal == 0.0f) {
        size_t cBytes = static_cast<size_t>(ldc) * n * (Ctype == ACL_FLOAT ? 4 : 2);
        aclError aclRet = aclrtMemset(C, cBytes, 0, cBytes);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasGemmEx", "aclrtMemset failed for C, ret=%d", aclRet);
            outStatus = ACLBLAS_STATUS_INTERNAL_ERROR;
            return true;
        }
    }
    if (betaVal == 0.0f || betaVal == 1.0f) {
        OP_LOGI("aclblasGemmEx", "alpha=0, beta=%.1f, skipping matmul", betaVal);
        outStatus = ACLBLAS_STATUS_SUCCESS;
        return true;
    }
    return false;
}

static aclblasStatus_t CheckWorkspaceSize(bool useFP32Output, aclDataType Ctype, int m, int n, _aclblas_handle* h)
{
    size_t tempElemSize = useFP32Output ? 4 : ((Ctype == ACL_FLOAT) ? 4 : 2);
    size_t tempABBytes = static_cast<size_t>(m) * n * tempElemSize;
    size_t availableBytes = GetEffectiveWorkspaceSize(h);
    if (tempABBytes > availableBytes) {
        OP_LOGE(
            "aclblasGemmEx",
            "workspace required %zu bytes, but only %zu bytes available. "
            "Please call aclblasSetWorkspace with size >= %zu bytes",
            tempABBytes, availableBytes, tempABBytes);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static void LaunchAlphaBetaKernel(
    _aclblas_handle* h, uint8_t* tempABDevice, void* C, const GemmExTilingData& abTilingData, int m, int n, int ldc,
    GemmDTypeCase dtypeCase, bool useFP32Output, float alphaVal, float betaVal, uint32_t cubeCoreNum)
{
    uint8_t* cOrigPtr = reinterpret_cast<uint8_t*>(const_cast<void*>(C));
    uint8_t* cOutPtr = reinterpret_cast<uint8_t*>(C);
    OP_LOGI(
        "aclblasGemmEx",
        "launching alpha/beta kernel: alpha=%.4f, beta=%.4f, "
        "m=%d, n=%d, ldc=%d, dtypeCase=%d, useFP32Temp=%d",
        alphaVal, betaVal, m, n, ldc, dtypeCase, useFP32Output ? 1 : 0);
    constexpr int64_t ELEMENTS_PER_CORE = 16384;
    uint32_t alphaBetaCores = static_cast<uint32_t>(std::min(
        static_cast<int32_t>(cubeCoreNum),
        std::max(1, static_cast<int32_t>((static_cast<int64_t>(m) * n + ELEMENTS_PER_CORE - 1) / ELEMENTS_PER_CORE))));
    gemm_ex_alpha_beta_do(
        alphaBetaCores, h->stream, tempABDevice, cOrigPtr, cOutPtr, abTilingData, dtypeCase, useFP32Output);
}

static aclblasStatus_t AllocateAndLaunchKernel(
    aclblasHandle_t handle, const void* A, const void* B, void* C, aclDataType Atype, aclDataType Btype,
    aclDataType Ctype, bool isTransA, bool isTransB, float alphaVal, float betaVal, int m, int n, int k, int lda,
    int ldb, int ldc, uint32_t cubeCoreNum)
{
    GemmExTilingData tilingData = CalcBlockMmadTiling(
        cubeCoreNum, m, n, k, lda, ldb, ldc, Atype, Btype, Ctype, isTransA, isTransB, alphaVal, betaVal);
    OP_CHECK_IF(
        tilingData.usedCoreNum == 0, OP_LOGE("aclblasGemmEx", "Invalid tiling data, usedCoreNum=0"),
        return ACLBLAS_STATUS_EXECUTION_FAILED);
    bool needPostProcess = (alphaVal != 1.0f) || (betaVal != 0.0f);
    GemmDTypeCase dtypeCase = GetDtypeCase(Atype, Btype, Ctype);
    bool useFP32Output = needPostProcess && (dtypeCase == GEMM_DTYPE_FP16 || dtypeCase == GEMM_DTYPE_BF16);
    bool needTempBuffer = needPostProcess;
    if (needTempBuffer) {
        tilingData.ldc = m;
    }
    GemmExTilingData abTilingData = tilingData;
    // Column-major trick: swap A↔B, M↔N
    std::swap(tilingData.m, tilingData.n);
    std::swap(tilingData.lda, tilingData.ldb);
    std::swap(tilingData.isTransA, tilingData.isTransB);
    std::swap(tilingData.singleCoreM, tilingData.singleCoreN);
    std::swap(tilingData.mBlocks, tilingData.nBlocks);

    auto* h = handle;

    uint8_t* tempABDevice = nullptr;
    if (needTempBuffer) {
        aclblasStatus_t st = CheckWorkspaceSize(useFP32Output, Ctype, m, n, h);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    tempABDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    uint8_t* aDevicePtr = reinterpret_cast<uint8_t*>(const_cast<void*>(B));
    uint8_t* bDevicePtr = reinterpret_cast<uint8_t*>(const_cast<void*>(A));
    uint8_t* cDevicePtr = needTempBuffer ? tempABDevice : reinterpret_cast<uint8_t*>(C);
    OP_LOGI(
        "aclblasGemmEx",
        "launching kernel: blocks=%u, transA=%d, transB=%d, dtypeCase=%d, "
        "needPostProcess=%d, lda=%d, ldb=%d, ldc=%d",
        tilingData.usedCoreNum, isTransA, isTransB, dtypeCase, needPostProcess, lda, ldb, tilingData.ldc);
    GemmDTypeCase kernelDtypeCase = dtypeCase;
    if (useFP32Output) {
        kernelDtypeCase = (dtypeCase == GEMM_DTYPE_BF16) ? GEMM_DTYPE_BF16_OUT_F32 : GEMM_DTYPE_FP16_OUT_F32;
    }
    gemm_ex_kernel_do(
        tilingData.usedCoreNum, h->stream, aDevicePtr, bDevicePtr, cDevicePtr, tilingData, isTransA, isTransB,
        kernelDtypeCase);
    if (needTempBuffer) {
        LaunchAlphaBetaKernel(
            h, tempABDevice, C, abTilingData, m, n, ldc, dtypeCase, useFP32Output, alphaVal, betaVal, cubeCoreNum);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasGemmEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb,
    const void* beta, void* C, aclDataType Ctype, int ldc, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
{
    OP_LOGI(
        "aclblasGemmEx",
        "entry: transa=%d, transb=%d, m=%d, n=%d, k=%d, "
        "Atype=%d, Btype=%d, Ctype=%d, computeType=%d, algo=%d",
        static_cast<int>(transa), static_cast<int>(transb), m, n, k, static_cast<int>(Atype), static_cast<int>(Btype),
        static_cast<int>(Ctype), static_cast<int>(computeType), static_cast<int>(algo));
    aclblasStatus_t st = ValidateGemmParams(
        handle, transa, transb, m, n, k, lda, ldb, ldc, alpha, beta, Atype, Btype, Ctype, computeType, algo, A, B, C);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasGemmEx", "parameter validation failed, status=%d", static_cast<int>(st));
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (k == 0) {
        return HandleKZero(C, ldc, n, Ctype, *static_cast<const float*>(beta));
    }
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    uint32_t cubeCoreNum = GetAicCoreCount();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasGemmEx", "Failed to get cube core count");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    const float alphaVal = *static_cast<const float*>(alpha);
    const float betaVal = *static_cast<const float*>(beta);
    if (alphaVal == 0.0f) {
        aclblasStatus_t alphaSt;
        if (HandleAlphaZero(C, ldc, n, Ctype, betaVal, alphaSt)) {
            return alphaSt;
        }
    }
    return AllocateAndLaunchKernel(
        handle, A, B, C, Atype, Btype, Ctype, isTransA, isTransB, alphaVal, betaVal, m, n, k, lda, ldb, ldc,
        cubeCoreNum);
}
