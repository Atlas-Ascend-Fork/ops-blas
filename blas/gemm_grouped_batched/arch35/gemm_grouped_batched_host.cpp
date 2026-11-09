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
#include <vector>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "gemm_grouped_batched_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// ---- kernel entry (local declaration, replacing aclblas_kernel_do.h) ----
void gemm_grouped_batched_kernel_do(
    uint8_t* tilingGm, uint8_t* groupParamsGm, uint8_t* aPtrArrayGm, uint8_t* bPtrArrayGm, uint8_t* cPtrArrayGm,
    uint32_t numBlocks, void* stream);

constexpr uint32_t K_TILE_MAX = 256;
constexpr uint32_t FP32_ALIGN = GEMM_GROUPED_FP32_ALIGN;
constexpr uint32_t UB_BYTES = GEMM_GROUPED_UB_SIZE;
constexpr uint32_t BUF_ALIGN = 256u;

static uint32_t MinGemmTileBufSize() { return CeilAlign<uint32_t>(FP32_ALIGN * FP32_ALIGN * sizeof(float), BUF_ALIGN); }

static uint32_t CeilAlignTileMNBuf(uint32_t tM_a, uint32_t tN_a)
{
    return CeilAlign<uint32_t>(tM_a * tN_a * sizeof(float), BUF_ALIGN);
}

static void AssignGroupParamBufSizes(
    GroupParam& gp, uint32_t szA, uint32_t szB, uint32_t szC, uint32_t szCIn, uint32_t szMul, uint32_t szVec)
{
    gp.bufSizeA = szA;
    gp.bufSizeB = szB;
    gp.bufSizeC = szC;
    gp.bufSizeCIn = szCIn;
    gp.bufSizeMulTmp = szMul;
    gp.bufSizeVecTmp = szVec;
}

static void SetGroupParamTiles(GroupParam& gp, uint32_t tileM, uint32_t tileK, uint32_t tileN)
{
    gp.tileM = tileM;
    gp.tileK = tileK;
    gp.tileN = tileN;
    gp.tileM_aligned = CeilAlign<uint32_t>(gp.tileM, FP32_ALIGN);
    gp.tileK_aligned = CeilAlign<uint32_t>(gp.tileK, FP32_ALIGN);
    gp.tileN_aligned = CeilAlign<uint32_t>(gp.tileN, FP32_ALIGN);
}

static void SetInactiveGroupParamTilesAndBufs(GroupParam& gp)
{
    SetGroupParamTiles(gp, FP32_ALIGN, FP32_ALIGN, FP32_ALIGN);
    const uint32_t minBuf = MinGemmTileBufSize();
    AssignGroupParamBufSizes(
        gp, minBuf, minBuf, minBuf, minBuf, minBuf, CeilAlign<uint32_t>(FP32_ALIGN * sizeof(float), BUF_ALIGN));
}

static aclblasStatus_t ValidateOneGroupTranspose(int g, aclblasOperation_t trans, const char* transName)
{
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T) {
        OP_LOGE(
            "aclblasGemmGroupedBatched", "trans must be OP_N(111) or OP_T(112), got %s=%d, group=%d", transName,
            static_cast<int>(trans), g);
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t CopyScratchHostToDevice(
    void* dst, size_t dstSize, const void* src, size_t srcSize, const char* name)
{
    aclError aclRet = aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasGemmGroupedBatched", "aclrtMemcpy %s H2D failed, ret=%d", name, aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename FitFn>
static void SearchTileSizesInUb(uint32_t mVal, uint32_t nVal, FitFn&& fits, uint32_t& bestTM, uint32_t& bestTN)
{
    bestTM = 0;
    bestTN = 0;
    for (uint32_t tM = CeilAlign<uint32_t>(mVal, FP32_ALIGN); tM >= FP32_ALIGN; tM -= FP32_ALIGN) {
        for (uint32_t tN = CeilAlign<uint32_t>(nVal, FP32_ALIGN); tN >= FP32_ALIGN; tN -= FP32_ALIGN) {
            if (fits(tM, tN)) {
                bestTM = tM;
                bestTN = tN;
                break;
            }
        }
        if (bestTM != 0) {
            break;
        }
    }
    if (bestTM == 0) {
        bestTM = FP32_ALIGN;
        bestTN = FP32_ALIGN;
    }
}

// API contract: Aarray/Barray/Carray are host-resident pointer arrays; each element
// is a device pointer to the corresponding matrix. The runtime cannot reliably
// distinguish every pageable host allocation, so the implementation follows this
// documented contract instead of probing pointer-array locations.

struct GemmGroupedScratchLayout {
    size_t tilingOff;
    size_t groupParamsOff;
    size_t aPtrOff;
    size_t bPtrOff;
    size_t cPtrOff;
    size_t totalBytes;
};

static GemmGroupedScratchLayout ComputeScratchLayout(size_t groupParamCount, uint32_t totalB)
{
    GemmGroupedScratchLayout layout{};
    size_t off = 0;
    layout.tilingOff = off;
    off += sizeof(GemmGroupedBatchedTilingData);
    off = CeilAlign<size_t>(off, 256u);
    layout.groupParamsOff = off;
    off += groupParamCount * sizeof(GroupParam);
    off = CeilAlign<size_t>(off, 256u);
    layout.aPtrOff = off;
    off += static_cast<size_t>(totalB) * sizeof(uint64_t);
    layout.bPtrOff = off;
    off += static_cast<size_t>(totalB) * sizeof(uint64_t);
    layout.cPtrOff = off;
    off += static_cast<size_t>(totalB) * sizeof(uint64_t);
    layout.totalBytes = off;
    return layout;
}

static aclblasStatus_t ValidateCommonParams(
    aclblasHandle_t handle, int groupCount, const aclblasOperation_t* transaArray,
    const aclblasOperation_t* transbArray, const int* mArray, const int* nArray, const int* kArray,
    const void* alphaArray, const void* Aarray, const int* ldaArray, const void* Barray, const int* ldbArray,
    const void* betaArray, const void* Carray, const int* ldcArray, const int* groupSizeArray)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasGemmGroupedBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (groupCount < 0) {
        OP_LOGE("aclblasGemmGroupedBatched", "groupCount must be >= 0, got %d", groupCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (groupCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (transaArray == nullptr || transbArray == nullptr) {
        OP_LOGE("aclblasGemmGroupedBatched", "transaArray/transbArray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (mArray == nullptr || nArray == nullptr || kArray == nullptr) {
        OP_LOGE("aclblasGemmGroupedBatched", "mArray/nArray/kArray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alphaArray == nullptr || betaArray == nullptr) {
        OP_LOGE("aclblasGemmGroupedBatched", "alphaArray/betaArray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldaArray == nullptr || ldbArray == nullptr || ldcArray == nullptr) {
        OP_LOGE("aclblasGemmGroupedBatched", "ldaArray/ldbArray/ldcArray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (groupSizeArray == nullptr) {
        OP_LOGE("aclblasGemmGroupedBatched", "groupSizeArray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Aarray == nullptr || Barray == nullptr || Carray == nullptr) {
        OP_LOGE("aclblasGemmGroupedBatched", "Aarray/Barray/Carray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGroupDimensions(int g, int m, int n, int k, int groupSize)
{
    if (m < 0 || n < 0 || k < 0) {
        OP_LOGE("aclblasGemmGroupedBatched", "m/n/k must be >= 0, got group[%d]: m=%d n=%d k=%d", g, m, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (groupSize < 0) {
        OP_LOGE("aclblasGemmGroupedBatched", "groupSize must be >= 0, got group[%d]: groupSize=%d", g, groupSize);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGroupTranspose(int g, aclblasOperation_t transa, aclblasOperation_t transb)
{
    aclblasStatus_t ret = ValidateOneGroupTranspose(g, transa, "transa");
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    return ValidateOneGroupTranspose(g, transb, "transb");
}

static aclblasStatus_t ValidateGroupLeadingDims(
    int g, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb, int ldc)
{
    // For transpose operations: op(A) is k×m, so A is k×m column-major with lda≥k
    // For trans=N: op(A) is m×k, so A is m×k column-major with lda≥m
    bool transaIsTranspose = (transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C);
    bool transbIsTranspose = (transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C);
    int ldaMin = transaIsTranspose ? std::max(1, k) : std::max(1, m);
    int ldbMin = transbIsTranspose ? std::max(1, n) : std::max(1, k);
    if (lda < ldaMin) {
        OP_LOGE("aclblasGemmGroupedBatched", "lda must be >= %d, got lda=%d, group=%d", ldaMin, lda, g);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldb < ldbMin) {
        OP_LOGE("aclblasGemmGroupedBatched", "ldb must be >= %d, got ldb=%d, group=%d", ldbMin, ldb, g);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("aclblasGemmGroupedBatched", "ldc must be >= max(1, m), got ldc=%d, m=%d, group=%d", ldc, m, g);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateGroupParams(
    int groupCount, const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray, const int* mArray,
    const int* nArray, const int* kArray, const int* ldaArray, const int* ldbArray, const int* ldcArray,
    const int* groupSizeArray, int64_t& totalBatchCount)
{
    // issue #870: 总批次以 uint64 累加并封顶 uint32 可承载上限（对齐 Ex 版 AddWithinUint32），
    // 防止 int 累加溢出为负后经 size_t 无符号放大触发 resize 异常、进程终止
    uint64_t totalBatches = 0;
    for (int g = 0; g < groupCount; g++) {
        aclblasStatus_t ret = ValidateGroupDimensions(g, mArray[g], nArray[g], kArray[g], groupSizeArray[g]);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
        if (groupSizeArray[g] == 0) {
            continue;
        }
        ret = ValidateGroupTranspose(g, transaArray[g], transbArray[g]);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
        ret = ValidateGroupLeadingDims(
            g, transaArray[g], transbArray[g], mArray[g], nArray[g], kArray[g], ldaArray[g], ldbArray[g], ldcArray[g]);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
        totalBatches += static_cast<uint64_t>(groupSizeArray[g]);
        if (totalBatches > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            OP_LOGE(
                "aclblasGemmGroupedBatched", "total batch count exceeds uint32 range: %llu",
                static_cast<unsigned long long>(totalBatches));
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    totalBatchCount = static_cast<int64_t>(totalBatches);
    return ACLBLAS_STATUS_SUCCESS;
}

static uint32_t ComputeBufSizes(
    uint32_t tM, uint32_t tK, uint32_t tN, uint32_t& szA, uint32_t& szB, uint32_t& szC, uint32_t& szCIn,
    uint32_t& szMul, uint32_t& szVec)
{
    uint32_t tM_a = CeilAlign<uint32_t>(tM, FP32_ALIGN);
    uint32_t tK_a = CeilAlign<uint32_t>(tK, FP32_ALIGN);
    uint32_t tN_a = CeilAlign<uint32_t>(tN, FP32_ALIGN);
    szA = CeilAlign<uint32_t>(tM_a * tK_a * sizeof(float), BUF_ALIGN);
    szB = CeilAlign<uint32_t>(tK_a * tN_a * sizeof(float), BUF_ALIGN);
    szC = CeilAlignTileMNBuf(tM_a, tN_a);
    szCIn = CeilAlignTileMNBuf(tM_a, tN_a);
    szMul = CeilAlignTileMNBuf(tM_a, tN_a);
    szVec = CeilAlign<uint32_t>(std::max(tM_a, tK_a) * sizeof(float), BUF_ALIGN);
    return szA + szB + szC + szCIn + szMul + szVec;
}

static GroupParam InitGroupParamCommonFields(
    int g, const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray, const int* mArray,
    const int* nArray, const int* kArray, const float* alphaArray, const float* betaArray, const int* ldaArray,
    const int* ldbArray, const int* ldcArray, const int* groupSizeArray, int groupOffset)
{
    GroupParam gp = {};
    gp.transa = (transaArray[g] == ACLBLAS_OP_T) ? 1 : 0;
    gp.transb = (transbArray[g] == ACLBLAS_OP_T) ? 1 : 0;
    gp.m = static_cast<uint32_t>(mArray[g]);
    gp.n = static_cast<uint32_t>(nArray[g]);
    gp.k = static_cast<uint32_t>(kArray[g]);
    gp.lda = static_cast<int32_t>(ldaArray[g]);
    gp.ldb = static_cast<int32_t>(ldbArray[g]);
    gp.ldc = static_cast<int32_t>(ldcArray[g]);
    gp.groupSize = static_cast<uint32_t>(groupSizeArray[g]);
    gp.groupOffset = static_cast<uint32_t>(groupOffset);
    gp.alphaReal = alphaArray[g];
    gp.alphaImag = 0.0f;
    gp.betaReal = betaArray[g];
    gp.betaImag = 0.0f;
    return gp;
}

static void SearchBetaOnlyTileSizes(uint32_t mVal, uint32_t nVal, uint32_t& bestTM, uint32_t& bestTN)
{
    const uint32_t minBuf = MinGemmTileBufSize();
    SearchTileSizesInUb(
        mVal, nVal,
        [&](uint32_t tM, uint32_t tN) {
            uint32_t tM_a = CeilAlign<uint32_t>(tM, FP32_ALIGN);
            uint32_t tN_a = CeilAlign<uint32_t>(tN, FP32_ALIGN);
            uint32_t mnBuf = CeilAlignTileMNBuf(tM_a, tN_a);
            uint32_t szVec = CeilAlign<uint32_t>(tM_a * sizeof(float), BUF_ALIGN);
            return minBuf + minBuf + mnBuf + mnBuf + mnBuf + szVec <= UB_BYTES;
        },
        bestTM, bestTN);
}

static void ConfigureBetaOnlyGroupParam(GroupParam& gp, uint32_t mVal, uint32_t nVal)
{
    uint32_t bestTM = 0;
    uint32_t bestTN = 0;
    SearchBetaOnlyTileSizes(mVal, nVal, bestTM, bestTN);
    SetGroupParamTiles(gp, std::min(bestTM, mVal), 0, std::min(bestTN, nVal));
    gp.tileK_aligned = 0;
    const uint32_t minBuf = MinGemmTileBufSize();
    uint32_t szA = minBuf;
    uint32_t szB = minBuf;
    uint32_t szC = 0;
    uint32_t szCIn = 0;
    uint32_t szMul = 0;
    uint32_t szVec = 0;
    ComputeBufSizes(gp.tileM, gp.tileK, gp.tileN, szA, szB, szC, szCIn, szMul, szVec);
    AssignGroupParamBufSizes(gp, minBuf, minBuf, szC, szCIn, szMul, szVec);
}

static void SearchGemmTileSizes(
    uint32_t mVal, uint32_t nVal, uint32_t kVal, uint32_t& bestTM, uint32_t& bestTK, uint32_t& bestTN)
{
    bestTK = 0;
    SearchTileSizesInUb(
        mVal, nVal,
        [&](uint32_t tM, uint32_t tN) {
            uint32_t tK = CeilAlign<uint32_t>(std::min(kVal, K_TILE_MAX), FP32_ALIGN);
            uint32_t szA = 0;
            uint32_t szB = 0;
            uint32_t szC = 0;
            uint32_t szCIn = 0;
            uint32_t szMul = 0;
            uint32_t szVec = 0;
            if (ComputeBufSizes(tM, tK, tN, szA, szB, szC, szCIn, szMul, szVec) <= UB_BYTES) {
                bestTK = tK;
                return true;
            }
            return false;
        },
        bestTM, bestTN);
    if (bestTK == 0) {
        bestTK = FP32_ALIGN;
    }
}

static void ConfigureGemmGroupParam(GroupParam& gp, uint32_t mVal, uint32_t nVal, uint32_t kVal)
{
    uint32_t bestTM = 0;
    uint32_t bestTK = 0;
    uint32_t bestTN = 0;
    SearchGemmTileSizes(mVal, nVal, kVal, bestTM, bestTK, bestTN);
    SetGroupParamTiles(gp, std::min(bestTM, mVal), std::min(bestTK, kVal), std::min(bestTN, nVal));
    uint32_t szA = 0;
    uint32_t szB = 0;
    uint32_t szC = 0;
    uint32_t szCIn = 0;
    uint32_t szMul = 0;
    uint32_t szVec = 0;
    ComputeBufSizes(gp.tileM, gp.tileK, gp.tileN, szA, szB, szC, szCIn, szMul, szVec);
    AssignGroupParamBufSizes(gp, szA, szB, szC, szCIn, szMul, szVec);
}

static GroupParam BuildGroupParam(
    int g, const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray, const int* mArray,
    const int* nArray, const int* kArray, const float* alphaArray, const float* betaArray, const int* ldaArray,
    const int* ldbArray, const int* ldcArray, const int* groupSizeArray, int groupOffset, bool inactive)
{
    GroupParam gp = InitGroupParamCommonFields(
        g, transaArray, transbArray, mArray, nArray, kArray, alphaArray, betaArray, ldaArray, ldbArray, ldcArray,
        groupSizeArray, groupOffset);
    if (inactive) {
        SetInactiveGroupParamTilesAndBufs(gp);
    } else if (gp.k == 0 || (gp.alphaReal == 0.0f && gp.alphaImag == 0.0f)) {
        ConfigureBetaOnlyGroupParam(gp, gp.m, gp.n);
    } else {
        ConfigureGemmGroupParam(gp, gp.m, gp.n, gp.k);
    }
    return gp;
}

struct GemmGroupedMaxBufSizes {
    uint32_t maxBufSizeA = 0;
    uint32_t maxBufSizeB = 0;
    uint32_t maxBufSizeC = 0;
    uint32_t maxBufSizeCIn = 0;
    uint32_t maxBufSizeMulTmp = 0;
    uint32_t maxBufSizeVecTmp = 0;
};

struct GemmGroupedCoreDistribution {
    uint32_t coreNum = 0;
    uint32_t batchPerCore = 0;
    uint32_t usedCoreNum = 0;
    uint32_t batchTail = 0;
};

struct GemmGroupedHostPtrArrays {
    std::vector<uint64_t> aPtr;
    std::vector<uint64_t> bPtr;
    std::vector<uint64_t> cPtr;
};

static aclblasStatus_t ValidateGemmGroupedBatchedInputs(
    aclblasHandle_t handle, int groupCount, const aclblasOperation_t* transaArray,
    const aclblasOperation_t* transbArray, const int* mArray, const int* nArray, const int* kArray,
    const void* alphaArray, const void* Aarray, const int* ldaArray, const void* Barray, const int* ldbArray,
    const void* betaArray, const void* Carray, const int* ldcArray, const int* groupSizeArray, int64_t& totalBatchCount)
{
    aclblasStatus_t ret = ValidateCommonParams(
        handle, groupCount, transaArray, transbArray, mArray, nArray, kArray, alphaArray, Aarray, ldaArray, Barray,
        ldbArray, betaArray, Carray, ldcArray, groupSizeArray);
    if (ret != ACLBLAS_STATUS_SUCCESS || groupCount == 0) {
        return ret;
    }

    totalBatchCount = 0;
    ret = ValidateGroupParams(
        groupCount, transaArray, transbArray, mArray, nArray, kArray, ldaArray, ldbArray, ldcArray, groupSizeArray,
        totalBatchCount);
    if (ret != ACLBLAS_STATUS_SUCCESS || totalBatchCount == 0) {
        return ret;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static std::vector<GroupParam> BuildAllGroupParams(
    int groupCount, const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray, const int* mArray,
    const int* nArray, const int* kArray, const float* alphaArray, const float* betaArray, const int* ldaArray,
    const int* ldbArray, const int* ldcArray, const int* groupSizeArray)
{
    std::vector<GroupParam> groupParams;
    groupParams.reserve(groupCount);
    int batchSlotOffset = 0;
    for (int g = 0; g < groupCount; g++) {
        const bool inactive = (groupSizeArray[g] == 0 || mArray[g] == 0 || nArray[g] == 0);
        groupParams.push_back(BuildGroupParam(
            g, transaArray, transbArray, mArray, nArray, kArray, alphaArray, betaArray, ldaArray, ldbArray, ldcArray,
            groupSizeArray, batchSlotOffset, inactive));
        batchSlotOffset += groupSizeArray[g];
    }
    return groupParams;
}

static GemmGroupedMaxBufSizes ComputeMaxGroupBufSizes(const std::vector<GroupParam>& groupParams)
{
    GemmGroupedMaxBufSizes maxBuf{};
    for (const GroupParam& gp : groupParams) {
        maxBuf.maxBufSizeA = std::max(maxBuf.maxBufSizeA, gp.bufSizeA);
        maxBuf.maxBufSizeB = std::max(maxBuf.maxBufSizeB, gp.bufSizeB);
        maxBuf.maxBufSizeC = std::max(maxBuf.maxBufSizeC, gp.bufSizeC);
        maxBuf.maxBufSizeCIn = std::max(maxBuf.maxBufSizeCIn, gp.bufSizeCIn);
        maxBuf.maxBufSizeMulTmp = std::max(maxBuf.maxBufSizeMulTmp, gp.bufSizeMulTmp);
        maxBuf.maxBufSizeVecTmp = std::max(maxBuf.maxBufSizeVecTmp, gp.bufSizeVecTmp);
    }
    return maxBuf;
}

template <typename T>
static GemmGroupedHostPtrArrays FlattenHostPtrArrays(
    int64_t totalBatchCount, const T* const* Aarray, const T* const* Barray, T* const* Carray)
{
    GemmGroupedHostPtrArrays hostPtrs;
    hostPtrs.aPtr.resize(static_cast<size_t>(totalBatchCount));
    hostPtrs.bPtr.resize(static_cast<size_t>(totalBatchCount));
    hostPtrs.cPtr.resize(static_cast<size_t>(totalBatchCount));
    for (int64_t b = 0; b < totalBatchCount; b++) {
        hostPtrs.aPtr[b] = reinterpret_cast<uint64_t>(const_cast<T*>(Aarray[b]));
        hostPtrs.bPtr[b] = reinterpret_cast<uint64_t>(const_cast<T*>(Barray[b]));
        hostPtrs.cPtr[b] = reinterpret_cast<uint64_t>(Carray[b]);
    }
    return hostPtrs;
}

static aclblasStatus_t ComputeGemmGroupedCoreDistribution(uint32_t totalB, GemmGroupedCoreDistribution& cores)
{
    cores.coreNum = GetAivCoreCount();
    if (cores.coreNum == 0) {
        OP_LOGE("aclblasGemmGroupedBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    cores.batchPerCore = CeilDiv<uint32_t>(totalB, cores.coreNum);
    cores.usedCoreNum = CeilDiv<uint32_t>(totalB, cores.batchPerCore);
    cores.batchTail = totalB - (cores.usedCoreNum - 1) * cores.batchPerCore;
    return ACLBLAS_STATUS_SUCCESS;
}

static GemmGroupedBatchedTilingData BuildGemmGroupedTilingData(
    int groupCount, uint32_t totalB, uint32_t dtype, const GemmGroupedCoreDistribution& cores,
    const GemmGroupedMaxBufSizes& maxBuf)
{
    GemmGroupedBatchedTilingData tiling = {};
    tiling.groupCount = static_cast<uint32_t>(groupCount);
    tiling.totalBatchCount = totalB;
    tiling.dtype = dtype;
    tiling.coreNum = cores.coreNum;
    tiling.usedCoreNum = cores.usedCoreNum;
    tiling.batchPerCore = cores.batchPerCore;
    tiling.batchTail = cores.batchTail;
    tiling.maxBufSizeA = maxBuf.maxBufSizeA;
    tiling.maxBufSizeB = maxBuf.maxBufSizeB;
    tiling.maxBufSizeC = maxBuf.maxBufSizeC;
    tiling.maxBufSizeCIn = maxBuf.maxBufSizeCIn;
    tiling.maxBufSizeMulTmp = maxBuf.maxBufSizeMulTmp;
    tiling.maxBufSizeVecTmp = maxBuf.maxBufSizeVecTmp;

    OP_LOGD(
        "aclblasGemmGroupedBatched",
        "tiling: groupCount=%u totalBatch=%u dtype=%u "
        "coreNum=%u usedCoreNum=%u batchPerCore=%u batchTail=%u",
        tiling.groupCount, tiling.totalBatchCount, tiling.dtype, tiling.coreNum, tiling.usedCoreNum,
        tiling.batchPerCore, tiling.batchTail);
    OP_LOGI("aclblasGemmGroupedBatched", "launching kernel: blocks=%u, cores=%u", cores.usedCoreNum, cores.coreNum);
    return tiling;
}

struct GemmGroupedScratchDevicePtrs {
    uint8_t* devTiling = nullptr;
    uint8_t* devGroupParams = nullptr;
    uint8_t* devAPtrArray = nullptr;
    uint8_t* devBPtrArray = nullptr;
    uint8_t* devCPtrArray = nullptr;
};

static aclblasStatus_t CopyGemmGroupedScratchToDevice(
    const GemmGroupedScratchDevicePtrs& devPtrs, GemmGroupedBatchedTilingData& tiling,
    const std::vector<GroupParam>& groupParams, uint32_t totalB, const GemmGroupedHostPtrArrays& hostPtrs)
{
    aclblasStatus_t ret = CopyScratchHostToDevice(
        devPtrs.devTiling, sizeof(GemmGroupedBatchedTilingData), &tiling, sizeof(GemmGroupedBatchedTilingData),
        "tiling");
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = CopyScratchHostToDevice(
        devPtrs.devGroupParams, groupParams.size() * sizeof(GroupParam), groupParams.data(),
        groupParams.size() * sizeof(GroupParam), "groupParams");
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = CopyScratchHostToDevice(
        devPtrs.devAPtrArray, totalB * sizeof(uint64_t), hostPtrs.aPtr.data(), totalB * sizeof(uint64_t), "aPtrArray");
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = CopyScratchHostToDevice(
        devPtrs.devBPtrArray, totalB * sizeof(uint64_t), hostPtrs.bPtr.data(), totalB * sizeof(uint64_t), "bPtrArray");
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    return CopyScratchHostToDevice(
        devPtrs.devCPtrArray, totalB * sizeof(uint64_t), hostPtrs.cPtr.data(), totalB * sizeof(uint64_t), "cPtrArray");
}

static aclblasStatus_t UploadScratchAndLaunchKernel(
    _aclblas_handle* h, GemmGroupedBatchedTilingData& tiling, const std::vector<GroupParam>& groupParams,
    uint32_t totalB, const GemmGroupedHostPtrArrays& hostPtrs, uint32_t usedCoreNum)
{
    GemmGroupedScratchLayout scratch = ComputeScratchLayout(groupParams.size(), totalB);
    size_t availableBytes = GetEffectiveWorkspaceSize(h);
    if (scratch.totalBytes > availableBytes) {
        OP_LOGE(
            "aclblasGemmGroupedBatched",
            "scratch required %zu bytes, but only %zu bytes available. "
            "Please call aclblasSetWorkspace with size >= %zu bytes",
            scratch.totalBytes, availableBytes, scratch.totalBytes);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint8_t* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    GemmGroupedScratchDevicePtrs devPtrs{};
    devPtrs.devTiling = workspace + scratch.tilingOff;
    devPtrs.devGroupParams = workspace + scratch.groupParamsOff;
    devPtrs.devAPtrArray = workspace + scratch.aPtrOff;
    devPtrs.devBPtrArray = workspace + scratch.bPtrOff;
    devPtrs.devCPtrArray = workspace + scratch.cPtrOff;

    tiling.groupParamsGmAddr = reinterpret_cast<uint64_t>(devPtrs.devGroupParams);
    tiling.aPtrArrayGmAddr = reinterpret_cast<uint64_t>(devPtrs.devAPtrArray);
    tiling.bPtrArrayGmAddr = reinterpret_cast<uint64_t>(devPtrs.devBPtrArray);
    tiling.cPtrArrayGmAddr = reinterpret_cast<uint64_t>(devPtrs.devCPtrArray);

    aclblasStatus_t ret = CopyGemmGroupedScratchToDevice(devPtrs, tiling, groupParams, totalB, hostPtrs);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    gemm_grouped_batched_kernel_do(
        devPtrs.devTiling, devPtrs.devGroupParams, devPtrs.devAPtrArray, devPtrs.devBPtrArray, devPtrs.devCPtrArray,
        usedCoreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
static aclblasStatus_t GemmGroupedBatchedImpl(
    aclblasHandle_t handle, int groupCount, const aclblasOperation_t* transaArray,
    const aclblasOperation_t* transbArray, const int* mArray, const int* nArray, const int* kArray, const T* alphaArray,
    const T* const* Aarray, const int* ldaArray, const T* const* Barray, const int* ldbArray, const T* betaArray,
    T* const* Carray, const int* ldcArray, const int* groupSizeArray, uint32_t dtype)
{
    int64_t totalBatchCount = 0;
    aclblasStatus_t ret = ValidateGemmGroupedBatchedInputs(
        handle, groupCount, transaArray, transbArray, mArray, nArray, kArray, alphaArray, Aarray, ldaArray, Barray,
        ldbArray, betaArray, Carray, ldcArray, groupSizeArray, totalBatchCount);
    if (ret != ACLBLAS_STATUS_SUCCESS || totalBatchCount == 0) {
        return ret;
    }

    // issue #870: 对齐 arch22/Ex 版本，host 侧构建与发射路径的异常不再 terminate 进程
    try {
        std::vector<GroupParam> groupParams = BuildAllGroupParams(
            groupCount, transaArray, transbArray, mArray, nArray, kArray, alphaArray, betaArray, ldaArray, ldbArray,
            ldcArray, groupSizeArray);
        GemmGroupedMaxBufSizes maxBuf = ComputeMaxGroupBufSizes(groupParams);

        uint32_t totalB = static_cast<uint32_t>(totalBatchCount);
        GemmGroupedHostPtrArrays hostPtrs = FlattenHostPtrArrays(totalBatchCount, Aarray, Barray, Carray);

        GemmGroupedCoreDistribution cores{};
        ret = ComputeGemmGroupedCoreDistribution(totalB, cores);
        if (ret != ACLBLAS_STATUS_SUCCESS || cores.usedCoreNum == 0) {
            return ret;
        }

        GemmGroupedBatchedTilingData tiling = BuildGemmGroupedTilingData(groupCount, totalB, dtype, cores, maxBuf);
        auto* h = handle;
        return UploadScratchAndLaunchKernel(h, tiling, groupParams, totalB, hostPtrs, cores.usedCoreNum);
    } catch (const std::exception&) {
        OP_LOGE("aclblasGemmGroupedBatched", "host-side execution failed with an exception");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
}

aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount, const aclblasOperation_t* transaArray,
    const aclblasOperation_t* transbArray, const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* const* Aarray, const int* ldaArray, const float* const* Barray,
    const int* ldbArray, const float* betaArray, float* const* Carray, const int* ldcArray, const int* groupSizeArray)
{
    return GemmGroupedBatchedImpl<float>(
        handle, groupCount, transaArray, transbArray, mArray, nArray, kArray, alphaArray, Aarray, ldaArray, Barray,
        ldbArray, betaArray, Carray, ldcArray, groupSizeArray, 0);
}
