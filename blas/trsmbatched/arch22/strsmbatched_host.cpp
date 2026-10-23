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
#include <vector>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "strsmbatched_kernel.h"
#include "strsmbatched_tiling_data.h"

namespace {

constexpr int MAX_ELEM_CHECK_BATCH = 64;

GM_ADDR AsGmAddr(const void* ptr) { return reinterpret_cast<GM_ADDR>(const_cast<void*>(ptr)); }

aclblasStatus_t ValidateEnumsAndDims(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa,
    aclblasDiagType_t diag, int m, int n, int lda, int ldb, int batchCount)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasStrsmBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) {
        OP_LOGE("aclblasStrsmBatched", "invalid side=%d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasStrsmBatched", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T) {
        OP_LOGE("aclblasStrsmBatched", "invalid transa=%d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) {
        OP_LOGE("aclblasStrsmBatched", "invalid diag=%d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || batchCount < 0) {
        OP_LOGE("aclblasStrsmBatched", "negative dimension: m=%d, n=%d, batchCount=%d", m, n, batchCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (lda < aDim || ldb < m) {
        OP_LOGE("aclblasStrsmBatched", "invalid lda=%d (min %d) or ldb=%d (min %d)", lda, aDim, ldb, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ReadScalarAlpha(const float* alpha, float& alphaVal)
{
    if (alpha == nullptr) {
        OP_LOGE("aclblasStrsmBatched", "alpha is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    bool isDevice = false;
    aclrtPtrAttributes attr{};
    if (aclrtPointerGetAttributes(alpha, &attr) == ACL_SUCCESS) {
        isDevice = (attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    }
    if (!isDevice) {
        alphaVal = *alpha;
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclError ret = aclrtMemcpy(&alphaVal, sizeof(float), alpha, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasStrsmBatched", "failed to copy device alpha, aclRet=%d", static_cast<int>(ret));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateBatchPtrArray(const void* ptrArray, int batchCount)
{
    if (ptrArray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchCount <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    bool isDevice = true;
    aclrtPtrAttributes attr{};
    if (aclrtPointerGetAttributes(ptrArray, &attr) == ACL_SUCCESS) {
        isDevice = (attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    }
    if (!isDevice) {
        auto hostPtrs = reinterpret_cast<const void* const*>(ptrArray);
        for (int i = 0; i < batchCount; ++i) {
            if (hostPtrs[i] == nullptr) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    const void* buf[MAX_ELEM_CHECK_BATCH] = {nullptr};
    int remaining = batchCount;
    auto devPtrBytes = reinterpret_cast<const uint8_t*>(ptrArray);
    while (remaining > 0) {
        const int curCount = std::min(remaining, MAX_ELEM_CHECK_BATCH);
        const size_t copyBytes = static_cast<size_t>(curCount) * sizeof(void*);
        if (aclrtMemcpy(buf, copyBytes, devPtrBytes, copyBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        for (int i = 0; i < curCount; ++i) {
            if (buf[i] == nullptr) {
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
        }
        devPtrBytes += copyBytes;
        remaining -= curCount;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t SelectGroupPanels(uint32_t aDim)
{
    if (aDim > 256U) {
        return 8U;
    }
    if (aDim > 128U) {
        return 4U;
    }
    return 1U;
}

StrsmbatchedTilingData BuildTilingData(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa, aclblasDiagType_t diag, int m, int n,
    int lda, int ldb, int batchCount, float alphaVal)
{
    StrsmbatchedTilingData t{};
    t.side = (side == ACLBLAS_SIDE_LEFT) ? 0U : 1U;
    t.uplo = (uplo == ACLBLAS_UPPER) ? 0U : 1U;
    t.trans = (transa == ACLBLAS_OP_N) ? 0U : 1U;
    t.diag = (diag == ACLBLAS_NON_UNIT) ? 0U : 1U;
    t.m = static_cast<uint32_t>(m);
    t.n = static_cast<uint32_t>(n);
    t.lda = static_cast<uint32_t>(lda);
    t.ldb = static_cast<uint32_t>(ldb);
    t.batchCount = static_cast<uint32_t>(batchCount);
    const uint32_t aDim = (t.side == 0U) ? t.m : t.n;
    t.panelNb = STRSMBATCHED_PANEL_NB;
    t.groupPanels = SelectGroupPanels(aDim);
    t.tempLd = std::max(64U, CeilAlign<uint32_t>(t.m, 16U));
    const uint32_t maxCols = std::max(64U, CeilAlign<uint32_t>(t.n, 16U));
    t.tempBatchStride = (aDim <= t.panelNb) ? 0ULL : (static_cast<uint64_t>(maxCols) * t.tempLd);
    t.alpha = alphaVal;
    t.reserved = 0U;
    return t;
}

static aclblasStatus_t ValidateBatchPointers(float alphaVal, const float* const A[], float* const B[], int batchCount)
{
    if (ValidateBatchPtrArray(B, batchCount) != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasStrsmBatched", "B or B[i] is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alphaVal != 0.0f && ValidateBatchPtrArray(A, batchCount) != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasStrsmBatched", "A or A[i] is nullptr when alpha != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static void DispatchZeroKernel(
    aclblasHandle_t handle, const StrsmbatchedTilingData& tiling, float* const B[], uint32_t aivCores)
{
    const uint32_t tiles = tiling.batchCount * CeilDiv<uint32_t>(tiling.m, 64U) * CeilDiv<uint32_t>(tiling.n, 64U);
    const uint32_t blocks = std::max(1U, std::min(aivCores, tiles));
    strsmbatched_zero_kernel_do(AsGmAddr(B), tiling, blocks, handle->stream);
}

static bool TryDispatchSmallAivKernel(
    aclblasHandle_t handle, const StrsmbatchedTilingData& tiling, const float* const A[], float* const B[],
    uint32_t aivCores)
{
    const uint32_t aDim = (tiling.side == 0U) ? tiling.m : tiling.n;
    const uint32_t rhsLen = (tiling.side == 0U) ? tiling.n : tiling.m;
    const uint32_t numChunks = CeilDiv<uint32_t>(rhsLen, STRSMBATCHED_PANEL_NB);
    const bool usePanelGemm = (aDim >= 24U) && (numChunks >= 3U) && (tiling.batchCount * numChunks >= 96U);
    const bool useMultiBatchSmall = (tiling.side == 0U) && (tiling.m == tiling.n) &&
                                    (tiling.m == 8U || tiling.m == 16U || tiling.m == 32U) &&
                                    (tiling.lda == tiling.m) && (tiling.ldb == tiling.m) && (tiling.batchCount >= 16U);
    if (aDim <= tiling.panelNb && !usePanelGemm) {
        if (useMultiBatchSmall) {
            const uint32_t effG = (tiling.m == 8U) ? 32U : (64U / tiling.m);
            uint32_t tasks = CeilDiv<uint32_t>(tiling.batchCount, effG);
            if (tiling.m == 8U && tiling.batchCount <= 256U) {
                tasks = std::max(tasks, 20U);
            }
            const uint32_t blocks = std::max(1U, std::min(aivCores, tasks));
            strsmbatched_small_aiv_kernel_do(AsGmAddr(A), AsGmAddr(B), tiling, blocks, handle->stream);
            return true;
        }
        const uint32_t tasks = tiling.batchCount * numChunks;
        const uint32_t blocks = std::max(1U, std::min(aivCores, tasks));
        strsmbatched_fallback_aiv_kernel_do(AsGmAddr(A), AsGmAddr(B), tiling, blocks, handle->stream);
        return true;
    }
    return false;
}

static aclblasStatus_t DispatchBlockedMixKernel(
    aclblasHandle_t handle, const StrsmbatchedTilingData& tiling, const float* const A[], float* const B[])
{
    const uint32_t aicCores = GetAicCoreCount();
    if (aicCores == 0U) {
        OP_LOGE("aclblasStrsmBatched", "GetAicCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint32_t aDim = (tiling.side == 0U) ? tiling.m : tiling.n;
    uint64_t totalWorkspaceElems = static_cast<uint64_t>(tiling.batchCount) * (tiling.tempBatchStride + 8192ULL);
    if (aDim > tiling.panelNb && (((tiling.lda & 7U) != 0U) || ((tiling.ldb & 7U) != 0U))) {
        const uint32_t ldaAligned = CeilAlign<uint32_t>(aDim, 16U);
        const uint32_t ldbAligned = CeilAlign<uint32_t>(tiling.m, 16U);
        if ((tiling.lda & 7U) != 0U) {
            totalWorkspaceElems += static_cast<uint64_t>(tiling.batchCount) * ldaAligned * aDim;
        }
        if ((tiling.ldb & 7U) != 0U) {
            totalWorkspaceElems += static_cast<uint64_t>(tiling.batchCount) * ldbAligned * tiling.n;
        }
        totalWorkspaceElems += static_cast<uint64_t>(tiling.batchCount) * 4ULL + 64ULL;
    }
    if (totalWorkspaceElems > ACLBLAS_MAX_WORKSPACE_SIZE / sizeof(float)) {
        OP_LOGE(
            "aclblasStrsmBatched", "workspace elements (%llu) exceeds 2 GiB limit",
            static_cast<unsigned long long>(totalWorkspaceElems));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const size_t requiredBytes = static_cast<size_t>(totalWorkspaceElems) * sizeof(float);
    const aclblasStatus_t wsStatus = EnsureDefaultWorkspace(handle, requiredBytes);
    if (wsStatus != ACLBLAS_STATUS_SUCCESS) {
        return wsStatus;
    }
    strsmbatched_blocked_mix_kernel_do(
        AsGmAddr(A), AsGmAddr(B), AsGmAddr(GetEffectiveWorkspace(handle)), tiling, aicCores, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[],
    int ldb, int batchCount)
{
    const aclblasStatus_t paramStatus =
        ValidateEnumsAndDims(handle, side, uplo, transa, diag, m, n, lda, ldb, batchCount);
    if (paramStatus != ACLBLAS_STATUS_SUCCESS) {
        return paramStatus;
    }
    float alphaVal = 0.0f;
    const aclblasStatus_t alphaStatus = ReadScalarAlpha(alpha, alphaVal);
    if (alphaStatus != ACLBLAS_STATUS_SUCCESS) {
        return alphaStatus;
    }
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t ptrStatus = ValidateBatchPointers(alphaVal, A, B, batchCount);
    if (ptrStatus != ACLBLAS_STATUS_SUCCESS) {
        return ptrStatus;
    }
    const StrsmbatchedTilingData tiling =
        BuildTilingData(side, uplo, transa, diag, m, n, lda, ldb, batchCount, alphaVal);
    const uint32_t aivCores = GetAivCoreCount();
    if (aivCores == 0U) {
        OP_LOGE("aclblasStrsmBatched", "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (alphaVal == 0.0f) {
        DispatchZeroKernel(handle, tiling, B, aivCores);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (TryDispatchSmallAivKernel(handle, tiling, A, B, aivCores)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return DispatchBlockedMixKernel(handle, tiling, A, B);
}
