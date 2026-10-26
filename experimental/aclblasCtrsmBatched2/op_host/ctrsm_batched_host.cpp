/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <complex>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "../../aclblas_minimal.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "ctrsm_batched_tiling_data.h"
#include "ctrsm_batched_kernel_do.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

constexpr int32_t MIX_NB_SMALL = 16;
constexpr int32_t MIX_NB_LARGE = 32;

static const char* GetSocVersion() { return aclrtGetSocName(); }

static void GenerateCubeTiling(int32_t maxMN, uint8_t* tilingBuf, uint32_t* tilingSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(GetSocVersion());
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    tilingApi.SetDim(1);
    tilingApi.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tilingApi.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetOrgShape(maxMN, maxMN, maxMN);
    tilingApi.SetShape(maxMN, maxMN, maxMN);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);
    optiling::TCubeTiling cubeTiling;
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        *tilingSize = 0;
        return;
    }
    *tilingSize = cubeTiling.GetDataSize();
    cubeTiling.SaveToBuffer(tilingBuf, *tilingSize);
}

static bool IsValidSide(aclblasSideMode_t s) { return s == ACLBLAS_SIDE_LEFT || s == ACLBLAS_SIDE_RIGHT; }
static bool IsValidUplo(aclblasFillMode_t u) { return u == ACLBLAS_UPPER || u == ACLBLAS_LOWER; }
static bool IsValidTrans(aclblasOperation_t t) { return t == ACLBLAS_OP_N || t == ACLBLAS_OP_T || t == ACLBLAS_OP_C; }
static bool IsValidDiag(aclblasDiagType_t d) { return d == ACLBLAS_NON_UNIT || d == ACLBLAS_UNIT; }

// 计算基本 tiling 参数（side/uplo/transa/diag 映射、useOrigA、nb 等）
static void FillCtrsmTilingBasicParams(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa, aclblasDiagType_t diag, int64_t m,
    int64_t n, int64_t lda, int64_t ldb, int64_t batchCount, float alphaReal, float alphaImag,
    CtrsmBatchedTilingData& td, int32_t& kDim, int32_t& nCols, int32_t& kDimAligned, int32_t& nColsAligned,
    bool& sideRight, bool& useDualAiv)
{
    kDim = (side == ACLBLAS_SIDE_LEFT) ? (int32_t)m : (int32_t)n;
    nCols = (side == ACLBLAS_SIDE_LEFT) ? (int32_t)n : (int32_t)m;
    kDimAligned = CEIL_ALIGN(kDim, FLOAT_ALIGN);
    nColsAligned = CEIL_ALIGN(nCols, FLOAT_ALIGN);

    sideRight = (side == ACLBLAS_SIDE_RIGHT);
    bool isTransN = (transa == ACLBLAS_OP_N);
    bool isTransT = (transa == ACLBLAS_OP_T);
    bool isTransC = (transa == ACLBLAS_OP_C);
    bool needTA = ((isTransT || isTransC) && !sideRight) || (isTransN && sideRight);
    bool padOn = (kDim != kDimAligned) || (nCols != nColsAligned);
    bool conjA = isTransC;
    bool useOrigA = (!needTA && !padOn && !conjA);

    td.m = (int32_t)m;
    td.n = (int32_t)n;
    td.lda = (int32_t)lda;
    td.ldb = (int32_t)ldb;
    td.batchCount = (int32_t)batchCount;
    td.side = sideRight ? SIDE_RIGHT : SIDE_LEFT;
    td.uplo = (uplo == ACLBLAS_UPPER) ? UPLO_UPPER : UPLO_LOWER;
    td.transa = isTransN ? TRANS_N : (isTransT ? TRANS_T : TRANS_C);
    td.diag = (diag == ACLBLAS_NON_UNIT) ? DIAG_NONUNIT : DIAG_UNIT;
    td.alphaReal = alphaReal;
    td.alphaImag = alphaImag;
    td.useOrigA = useOrigA ? 1 : 0;
    td.aEffStride = useOrigA ? ((int32_t)lda * 2) : (kDimAligned * 2);
    useDualAiv = (kDim >= 128 && nColsAligned >= 128);
    td.dualAivMode = useDualAiv ? 1 : 0;
    td.nb = (kDim > 1024) ? MIX_NB_LARGE : MIX_NB_SMALL;
}

// 计算 workspace 大小和 numBlocks
static void ComputeWorkspaceAndBlocks(
    CtrsmBatchedTilingData& td, int32_t kDimAligned, int32_t nColsAligned, int32_t numSplits, int32_t splitNColsAligned,
    int32_t lastNColsAligned, bool sideRight, bool useDualAiv, int64_t batchCount, int32_t& maxMN, uint32_t& numBlocks,
    int32_t& effectiveBatch)
{
    int32_t wsNCols = (numSplits > 1) ?
                          ((splitNColsAligned > lastNColsAligned) ? splitNColsAligned : lastNColsAligned) :
                          nColsAligned;
    int64_t aSize = (int64_t)kDimAligned * kDimAligned * 2;
    int64_t bSize = (int64_t)kDimAligned * wsNCols * 2;
    int64_t xnegSize = (int64_t)2 * 16 * 2 * td.nb * wsNCols * 2;
    int64_t rightTempSize = sideRight ? ((int64_t)wsNCols * kDimAligned * 2) : 0;
    int64_t gemmAreaSize = xnegSize;
    if (rightTempSize > gemmAreaSize)
        gemmAreaSize = rightTempSize;
    int64_t splitBGemmSize = (bSize + gemmAreaSize);
    td.splitBGemmSize = splitBGemmSize * sizeof(float);
    if (numSplits > 1) {
        td.workspaceOffset = (aSize + splitBGemmSize * numSplits) * sizeof(float);
    } else {
        td.workspaceOffset = (aSize + splitBGemmSize) * sizeof(float);
    }
    maxMN = (kDimAligned > wsNCols) ? kDimAligned : wsNCols;
    int32_t maxK = 2 * kDimAligned;
    if (maxK > maxMN)
        maxMN = maxK;
    if (numSplits > 1) {
        numBlocks = (uint32_t)((int32_t)batchCount * numSplits);
        effectiveBatch = (int32_t)batchCount;
    } else if (useDualAiv) {
        numBlocks = (uint32_t)batchCount;
        effectiveBatch = (int32_t)batchCount;
    } else {
        effectiveBatch = (int32_t)batchCount;
        if (effectiveBatch % 2 != 0)
            effectiveBatch++;
        numBlocks = (uint32_t)(effectiveBatch / 2);
    }
}

// 计算 split/numBlocks/workspace 参数
static void FillCtrsmTilingSplitAndBlocks(
    CtrsmBatchedTilingData& td, int32_t kDim, int32_t nCols, int32_t kDimAligned, int32_t nColsAligned, bool sideRight,
    bool useDualAiv, int64_t batchCount, int32_t& maxMN, uint32_t& numBlocks, int32_t& effectiveBatch)
{
    constexpr int32_t SPLIT_ALIGN = 16;
    constexpr int32_t TOTAL_AICORES = 20;
    int32_t numSplits = 1;
    if (useDualAiv && (int32_t)batchCount <= TOTAL_AICORES / 2) {
        int32_t maxSplits = TOTAL_AICORES / (int32_t)batchCount;
        int32_t minSplitNCols = (td.nb > 64) ? td.nb : 64;
        int32_t maxBySize = nCols / minSplitNCols;
        if (maxSplits > maxBySize)
            maxSplits = maxBySize;
        if ((int32_t)batchCount * maxSplits > TOTAL_AICORES)
            maxSplits = TOTAL_AICORES / (int32_t)batchCount;
        if (maxSplits > 1)
            numSplits = maxSplits;
    }
    int32_t splitNCols = nCols;
    int32_t splitNColsAligned = nColsAligned;
    int32_t lastNCols = nCols;
    int32_t lastNColsAligned = nColsAligned;
    if (numSplits > 1) {
        splitNCols = (nCols / numSplits) & ~(SPLIT_ALIGN - 1);
        splitNColsAligned = CEIL_ALIGN(splitNCols, FLOAT_ALIGN);
        lastNCols = nCols - splitNCols * (numSplits - 1);
        lastNColsAligned = CEIL_ALIGN(lastNCols, FLOAT_ALIGN);
    }
    td.numSplits = numSplits;
    td.splitNCols = splitNCols;
    td.splitNColsAligned = splitNColsAligned;
    td.lastNCols = lastNCols;
    td.lastNColsAligned = lastNColsAligned;
    ComputeWorkspaceAndBlocks(
        td, kDimAligned, nColsAligned, numSplits, splitNColsAligned, lastNColsAligned, sideRight, useDualAiv,
        batchCount, maxMN, numBlocks, effectiveBatch);
}

// 由 side/uplo/transa/diag 及尺寸推导 tiling 参数，并计算 kernel 启动参数（maxMN/numBlocks/effectiveBatch）
static void FillCtrsmTiling(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa, aclblasDiagType_t diag, int64_t m,
    int64_t n, int64_t lda, int64_t ldb, int64_t batchCount, float alphaReal, float alphaImag,
    CtrsmBatchedTilingData& td, int32_t& maxMN, uint32_t& numBlocks, int32_t& effectiveBatch)
{
    int32_t kDim, nCols, kDimAligned, nColsAligned;
    bool sideRight, useDualAiv;
    FillCtrsmTilingBasicParams(
        side, uplo, transa, diag, m, n, lda, ldb, batchCount, alphaReal, alphaImag, td, kDim, nCols, kDimAligned,
        nColsAligned, sideRight, useDualAiv);
    FillCtrsmTilingSplitAndBlocks(
        td, kDim, nCols, kDimAligned, nColsAligned, sideRight, useDualAiv, batchCount, maxMN, numBlocks,
        effectiveBatch);
}

// 分配 device 内存、拷贝入参与 cube tiling、启动 kernel 并同步、释放资源
// !692 检视意见修复：任一失败路径统一走 cleanup 释放全部已分配设备缓冲，不再泄漏
static aclblasStatus_t LaunchCtrsmKernel(
    const CtrsmBatchedTilingData& td, int32_t maxMN, uint32_t numBlocks, int32_t effectiveBatch,
    const std::complex<float>* const aArray[], int64_t lda, std::complex<float>* const bArray[], int64_t batchCount,
    aclrtStream stream)
{
    size_t ptrArraySize = (size_t)batchCount * sizeof(void*);
    uint8_t *tilingDevice = nullptr, *aArrayDevice = nullptr, *bArrayDevice = nullptr;
    uint8_t *cubeTilingDevice = nullptr, *gemmWsDevice = nullptr, *sysWsDevice = nullptr;
    auto platformInst = platform_ascendc::PlatformAscendCManager::GetInstance(GetSocVersion());
    size_t sysWsSize = static_cast<size_t>(platformInst->GetLibApiWorkSpaceSize());
    if (sysWsSize < 16 * 1024 * 1024)
        sysWsSize = 16 * 1024 * 1024;

    aclblasStatus_t status = ACLBLAS_STATUS_INTERNAL_ERROR;
    do {
        if (aclrtMalloc((void**)&tilingDevice, sizeof(td), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc((void**)&aArrayDevice, ptrArraySize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMalloc((void**)&bArrayDevice, ptrArraySize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            // issue #587: 拷贝失败时 tiling/指针数组保持未初始化，内核照常启动会读垃圾数据，
            // 对齐 aclblasTrsmBatched 的写法补充返回值检查，失败即返回不下发内核
            aclrtMemcpy(tilingDevice, sizeof(td), &td, sizeof(td), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(aArrayDevice, ptrArraySize, aArray, ptrArraySize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(bArrayDevice, ptrArraySize, bArray, ptrArraySize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            break;
        }

        uint8_t cubeTilingBuf[2048];
        uint32_t cubeTilingSize = 0;
        GenerateCubeTiling(maxMN, cubeTilingBuf, &cubeTilingSize);
        if (aclrtMalloc((void**)&cubeTilingDevice, sizeof(cubeTilingBuf), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMemcpy(
                cubeTilingDevice, sizeof(cubeTilingBuf), cubeTilingBuf, sizeof(cubeTilingBuf),
                ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            break;
        }

        size_t gemmWsSize = (size_t)effectiveBatch * td.workspaceOffset;
        if (aclrtMalloc((void**)&gemmWsDevice, gemmWsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            aclrtMemset(gemmWsDevice, gemmWsSize, 0, gemmWsSize) != ACL_SUCCESS ||
            aclrtMalloc((void**)&sysWsDevice, sysWsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            break;
        }

        ctrsm_batched_mix12_kernel_do(
            aArrayDevice, bArrayDevice, tilingDevice, cubeTilingDevice, gemmWsDevice, sysWsDevice, numBlocks, stream);
        status =
            (aclrtSynchronizeStream(stream) == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
    } while (0);

    if (sysWsDevice != nullptr)
        aclrtFree(sysWsDevice);
    if (gemmWsDevice != nullptr)
        aclrtFree(gemmWsDevice);
    if (cubeTilingDevice != nullptr)
        aclrtFree(cubeTilingDevice);
    if (bArrayDevice != nullptr)
        aclrtFree(bArrayDevice);
    if (aArrayDevice != nullptr)
        aclrtFree(aArrayDevice);
    if (tilingDevice != nullptr)
        aclrtFree(tilingDevice);
    return status;
}

// 参数合法性校验：句柄、枚举、维度、lda/ldb。返回 SUCCESS 表示校验通过可继续
static aclblasStatus_t ValidateCtrsmArgs(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa,
    aclblasDiagType_t diag, int64_t m, int64_t n, const std::complex<float>* alpha,
    const std::complex<float>* const aArray[], int64_t lda, std::complex<float>* const bArray[], int64_t ldb,
    int64_t batchCount)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!IsValidSide(side) || !IsValidUplo(uplo) || !IsValidTrans(transa) || !IsValidDiag(diag)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || batchCount < 0 || alpha == nullptr || aArray == nullptr || bArray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int64_t minLda = (side == ACLBLAS_SIDE_LEFT) ? std::max((int64_t)1, m) : std::max((int64_t)1, n);
    if (lda < minLda || ldb < std::max((int64_t)1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa,
    aclblasDiagType_t diag, int64_t m, int64_t n, const std::complex<float>* alpha,
    const std::complex<float>* const aArray[], int64_t lda, std::complex<float>* const bArray[], int64_t ldb,
    int64_t batchCount)
{
    aclblasStatus_t vst =
        ValidateCtrsmArgs(handle, side, uplo, transa, diag, m, n, alpha, aArray, lda, bArray, ldb, batchCount);
    if (vst != ACLBLAS_STATUS_SUCCESS) {
        return vst;
    }
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    float alphaReal = alpha->real();
    float alphaImag = alpha->imag();
    if (alphaReal == 0.0f && alphaImag == 0.0f) {
        for (int64_t i = 0; i < batchCount; i++) {
            size_t bytes = (size_t)m * ldb * 2 * sizeof(float);
            CHECK_RET(aclrtMemset(bArray[i], bytes, 0, bytes) == ACL_SUCCESS, return ACLBLAS_STATUS_INTERNAL_ERROR);
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    CtrsmBatchedTilingData td;
    int32_t maxMN;
    uint32_t numBlocks;
    int32_t effectiveBatch;
    FillCtrsmTiling(
        side, uplo, transa, diag, m, n, lda, ldb, batchCount, alphaReal, alphaImag, td, maxMN, numBlocks,
        effectiveBatch);

    return LaunchCtrsmKernel(td, maxMN, numBlocks, effectiveBatch, aArray, lda, bArray, batchCount, stream);
}
