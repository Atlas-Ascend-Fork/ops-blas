/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trsm_batched_host.cpp
 * \brief StrsmBatched operator host: validation, tiling, workspace allocation and kernel launch.
 *        Solves op(A) * X = alpha * B (side=L) or X * op(A) = alpha * B (side=R) for a batch.
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "../../aclblas_minimal.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "trsm_batched_kernel_do.h"
#include "trsm_batched_tiling_data.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static int32_t ChooseNb(int32_t kDim)
{
    if (kDim >= 4096)
        return 64;
    if (kDim >= 256)
        return 32;
    return 16;
}

static const char* GetSocVersion() { return aclrtGetSocName(); }

static bool IsValidSide(aclblasSideMode_t side) { return (side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT); }

static bool IsValidUplo(aclblasFillMode_t uplo) { return (uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER); }

static bool IsValidRealTrans(aclblasOperation_t transa)
{
    // Strsm is real-valued: conjugate transpose (OP_C) is not meaningful, only N/T are valid.
    return (transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T);
}

static bool IsValidDiag(aclblasDiagType_t diag) { return (diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT); }

static void GenerateCubeTiling(
    int32_t maxM, int32_t nCols, int32_t nb, bool isTransA, uint8_t* tilingBuf, uint32_t* tilingSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(GetSocVersion());
    matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    tilingApi.SetDim(1);
    tilingApi.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, isTransA);
    tilingApi.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetOrgShape(maxM, nCols, nb);
    tilingApi.SetShape(maxM, nCols, nb);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling cubeTiling;
    if (tilingApi.GetTiling(cubeTiling) == -1) {
        LOG_PRINT("ERROR: cube tiling generation failed\n");
        *tilingSize = 0;
        return;
    }
    *tilingSize = cubeTiling.GetDataSize();
    cubeTiling.SaveToBuffer(tilingBuf, *tilingSize);
}

// Compute splitFactor, workspace offset, and numBlocks from derived dimensions
static void ComputeSplitParams(
    int32_t kDim, int32_t nColsAligned, int32_t kDimAligned, int32_t maxMN, int64_t batchCount, int64_t cubeCoreNum,
    bool coopMode, TrsmBatchedTilingData& tiling, uint32_t& numBlocks)
{
    int32_t splitFactor = 1;
    if (coopMode && kDim >= 768 && batchCount <= cubeCoreNum / 2) {
        splitFactor = (int32_t)cubeCoreNum / (int32_t)batchCount;
        if (splitFactor > nColsAligned / 256) {
            splitFactor = nColsAligned / 256;
        }
        if (splitFactor > 8)
            splitFactor = 8;
        if (splitFactor < 1)
            splitFactor = 1;
    }
    tiling.splitFactor = splitFactor;
    int32_t splitNcols = nColsAligned;
    if (splitFactor > 1) {
        splitNcols = CEIL_ALIGN(nColsAligned / splitFactor, FLOAT_ALIGN);
    }
    tiling.splitNcolsMax = splitNcols;
    tiling.workspaceOffset = ((int64_t)kDimAligned * kDimAligned * splitFactor + (int64_t)kDimAligned * nColsAligned +
                              2 * (int64_t)maxMN * splitNcols * splitFactor) *
                             sizeof(float);
    tiling.coopMode = coopMode ? 1 : 0;
    if (coopMode) {
        numBlocks = (uint32_t)batchCount * (uint32_t)splitFactor;
    } else {
        numBlocks = ((uint32_t)batchCount + 1) / 2;
        int32_t numPanels = (kDim + tiling.nb - 1) / tiling.nb;
        if (numPanels <= 1 && numBlocks > (uint32_t)cubeCoreNum) {
            numBlocks = (uint32_t)cubeCoreNum;
        }
    }
}

// 查询核数、由尺寸推导 tiling 参数，并计算 maxMN 与 numBlocks（协作模式判定在内）
static void FillTrsmTiling(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa, aclblasDiagType_t diag, int64_t m,
    int64_t n, int64_t lda, int64_t ldb, int64_t batchCount, float alpha, TrsmBatchedTilingData& tiling, int32_t& maxMN,
    uint32_t& numBlocks)
{
    int32_t deviceId = 0;
    aclrtGetDevice(&deviceId);
    int64_t vecCoreNum = 0;
    int64_t cubeCoreNum = 0;
    aclrtGetDeviceInfo((uint32_t)deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vecCoreNum);
    aclrtGetDeviceInfo((uint32_t)deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &cubeCoreNum);
    if (vecCoreNum <= 0) {
        vecCoreNum = 40;
    }
    if (cubeCoreNum <= 0) {
        cubeCoreNum = 20;
    }

    int32_t kDim = (side == ACLBLAS_SIDE_LEFT) ? (int32_t)m : (int32_t)n;
    int32_t nCols = (side == ACLBLAS_SIDE_LEFT) ? (int32_t)n : (int32_t)m;
    int32_t kDimAligned = CEIL_ALIGN(kDim, FLOAT_ALIGN);
    int32_t nColsAligned = CEIL_ALIGN(nCols, FLOAT_ALIGN);
    maxMN = (kDimAligned > nColsAligned ? kDimAligned : nColsAligned);
    int32_t maxMNAligned = CEIL_ALIGN(maxMN, FLOAT_ALIGN);

    tiling.m = (int32_t)m;
    tiling.n = (int32_t)n;
    tiling.lda = (int32_t)lda;
    tiling.ldb = (int32_t)ldb;
    tiling.batchCount = (int32_t)batchCount;
    tiling.nb = ChooseNb(kDim);
    tiling.totalCores = std::min((int32_t)vecCoreNum, (int32_t)batchCount);
    if (tiling.totalCores < 1) {
        tiling.totalCores = 1;
    }
    tiling.side = (side == ACLBLAS_SIDE_LEFT) ? SIDE_LEFT : SIDE_RIGHT;
    tiling.uplo = (uplo == ACLBLAS_UPPER) ? UPLO_UPPER : UPLO_LOWER;
    tiling.transa = (transa == ACLBLAS_OP_N) ? TRANS_N : TRANS_T;
    tiling.diag = (diag == ACLBLAS_NON_UNIT) ? DIAG_NONUNIT : DIAG_UNIT;
    tiling.alphaReal = alpha;

    bool coopMode = (batchCount < cubeCoreNum);
    ComputeSplitParams(kDim, nColsAligned, kDimAligned, maxMN, batchCount, cubeCoreNum, coopMode, tiling, numBlocks);
}

// 分配 device 内存、拷贝入参/cube tiling/单位矩阵，启动 kernel 并同步、释放资源
// 构建 maxMN×maxMN 单位矩阵到 device（GEMM 尾块更新所需）
static aclblasStatus_t BuildIdentityDevice(int32_t maxMN, uint8_t** idDevice)
{
    size_t idSize = (size_t)maxMN * maxMN * sizeof(float);
    std::vector<float> idHost((size_t)maxMN * maxMN, 0.0f);
    for (int32_t i = 0; i < maxMN; i++) {
        idHost[(size_t)i * maxMN + i] = 1.0f;
    }
    CHECK_RET(
        aclrtMalloc((void**)idDevice, idSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    CHECK_RET(
        aclrtMemcpy(*idDevice, idSize, idHost.data(), idSize, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    return ACLBLAS_STATUS_SUCCESS;
}

// 分配并拷贝 tiling / A、B 指针数组 / cube tiling / gemm 与 sys 工作区到 device
static aclblasStatus_t AllocTrsmDeviceInputs(
    const TrsmBatchedTilingData& tiling, int32_t maxMN, const float* const aArray[], float* const bArray[],
    int64_t batchCount, uint8_t** tilingDevice, uint8_t** aArrayDevice, uint8_t** bArrayDevice,
    uint8_t** cubeTilingDevice, uint8_t** gemmWsDevice, uint8_t** sysWsDevice)
{
    size_t ptrArraySize = (size_t)batchCount * sizeof(void*);
    CHECK_RET(
        aclrtMalloc((void**)tilingDevice, sizeof(TrsmBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    CHECK_RET(
        aclrtMalloc((void**)aArrayDevice, ptrArraySize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    CHECK_RET(
        aclrtMalloc((void**)bArrayDevice, ptrArraySize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    CHECK_RET(
        aclrtMemcpy(
            *tilingDevice, sizeof(TrsmBatchedTilingData), &tiling, sizeof(TrsmBatchedTilingData),
            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    CHECK_RET(
        aclrtMemcpy(*aArrayDevice, ptrArraySize, aArray, ptrArraySize, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    CHECK_RET(
        aclrtMemcpy(*bArrayDevice, ptrArraySize, bArray, ptrArraySize, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    uint8_t cubeTilingBuf[2048];
    uint32_t cubeTilingSize = 0;
    int32_t cubeN = maxMN;
    if (tiling.splitFactor > 1) {
        int32_t nCols = (tiling.side == SIDE_LEFT) ? tiling.n : tiling.m;
        int32_t nColsAligned = CEIL_ALIGN(nCols, FLOAT_ALIGN);
        int32_t splitNcols = CEIL_ALIGN(nColsAligned / tiling.splitFactor, FLOAT_ALIGN);
        if (splitNcols < maxMN)
            cubeN = splitNcols;
    }
    GenerateCubeTiling(maxMN, cubeN, maxMN, false, cubeTilingBuf, &cubeTilingSize);
    // issue #612: tiling 结构膨胀超过缓冲容量时编译期无感知，必须显式校验；
    // 拷贝长度用实际 tiling 大小，避免把缓冲尾部未初始化区域一并下发设备
    CHECK_RET(cubeTilingSize <= sizeof(cubeTilingBuf), return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        aclrtMalloc((void**)cubeTilingDevice, sizeof(cubeTilingBuf), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    CHECK_RET(
        aclrtMemcpy(
            *cubeTilingDevice, sizeof(cubeTilingBuf), cubeTilingBuf, cubeTilingSize, ACL_MEMCPY_HOST_TO_DEVICE) ==
            ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    size_t gemmWsSize = (size_t)batchCount * tiling.workspaceOffset;
    CHECK_RET(
        aclrtMalloc((void**)gemmWsDevice, gemmWsSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    CHECK_RET(
        aclrtMemset(*gemmWsDevice, gemmWsSize, 0, gemmWsSize) == ACL_SUCCESS, return ACLBLAS_STATUS_INTERNAL_ERROR);

    auto platformInst = platform_ascendc::PlatformAscendCManager::GetInstance(GetSocVersion());
    size_t sysWsSize = static_cast<size_t>(platformInst->GetLibApiWorkSpaceSize());
    if (sysWsSize < 16 * 1024 * 1024) {
        sysWsSize = 16 * 1024 * 1024;
    }
    CHECK_RET(
        aclrtMalloc((void**)sysWsDevice, sysWsSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
        return ACLBLAS_STATUS_INTERNAL_ERROR);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchTrsmKernel(
    const TrsmBatchedTilingData& tiling, int32_t maxMN, uint32_t numBlocks, const float* const aArray[], int64_t lda,
    float* const bArray[], int64_t batchCount, aclrtStream stream)
{
    uint8_t* tilingDevice = nullptr;
    uint8_t* aArrayDevice = nullptr;
    uint8_t* bArrayDevice = nullptr;
    uint8_t* cubeTilingDevice = nullptr;
    uint8_t* gemmWsDevice = nullptr;
    uint8_t* sysWsDevice = nullptr;
    uint8_t* idDevice = nullptr;
    aclblasStatus_t status = AllocTrsmDeviceInputs(
        tiling, maxMN, aArray, bArray, batchCount, &tilingDevice, &aArrayDevice, &bArrayDevice, &cubeTilingDevice,
        &gemmWsDevice, &sysWsDevice);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = BuildIdentityDevice(maxMN, &idDevice);
    }
    if (status != ACLBLAS_STATUS_SUCCESS) {
        if (idDevice != nullptr) {
            aclrtFree(idDevice);
        }
        if (sysWsDevice != nullptr) {
            aclrtFree(sysWsDevice);
        }
        if (gemmWsDevice != nullptr) {
            aclrtFree(gemmWsDevice);
        }
        if (cubeTilingDevice != nullptr) {
            aclrtFree(cubeTilingDevice);
        }
        if (bArrayDevice != nullptr) {
            aclrtFree(bArrayDevice);
        }
        if (aArrayDevice != nullptr) {
            aclrtFree(aArrayDevice);
        }
        if (tilingDevice != nullptr) {
            aclrtFree(tilingDevice);
        }
        return status;
    }

    trsm_batched_mix12_kernel_do(
        aArrayDevice, bArrayDevice, tilingDevice, cubeTilingDevice, gemmWsDevice, sysWsDevice, idDevice, numBlocks,
        stream);
    aclError syncRet = aclrtSynchronizeStream(stream);

    aclrtFree(idDevice);
    aclrtFree(sysWsDevice);
    aclrtFree(gemmWsDevice);
    aclrtFree(cubeTilingDevice);
    aclrtFree(bArrayDevice);
    aclrtFree(aArrayDevice);
    aclrtFree(tilingDevice);
    CHECK_RET(syncRet == ACL_SUCCESS, return ACLBLAS_STATUS_INTERNAL_ERROR);
    return ACLBLAS_STATUS_SUCCESS;
}

extern "C" aclblasStatus_t aclblasStrsmBatchedEx(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t transa,
    aclblasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const aArray[], int64_t lda,
    float* const bArray[], int64_t ldb, int64_t batchCount)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!IsValidSide(side) || !IsValidUplo(uplo) || !IsValidRealTrans(transa) || !IsValidDiag(diag)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || batchCount < 0 || alpha == nullptr || aArray == nullptr || bArray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int64_t minLda = (side == ACLBLAS_SIDE_LEFT) ? std::max((int64_t)1, m) : std::max((int64_t)1, n);
    if (lda < minLda || ldb < std::max((int64_t)1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (*alpha == 0.0f) {
        for (int64_t i = 0; i < batchCount; i++) {
            size_t bytes = (size_t)m * ldb * sizeof(float);
            CHECK_RET(aclrtMemset(bArray[i], bytes, 0, bytes) == ACL_SUCCESS, return ACLBLAS_STATUS_INTERNAL_ERROR);
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    TrsmBatchedTilingData tiling;
    int32_t maxMN;
    uint32_t numBlocks;
    FillTrsmTiling(side, uplo, transa, diag, m, n, lda, ldb, batchCount, *alpha, tiling, maxMN, numBlocks);

    return LaunchTrsmKernel(tiling, maxMN, numBlocks, aArray, lda, bArray, batchCount, stream);
}
