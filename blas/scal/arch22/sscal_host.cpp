/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sscal_host.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

struct SscalTilingData;
// arch35-style: tiling passed by value
void sscal_kernel_do(uint8_t* x, uint8_t* workSpace, const SscalTilingData& tiling, uint32_t numBlocks, void* stream);
// arch22-style: tiling passed as GM pointer (for backward compatibility)
void sscal_kernel_do(uint8_t* x, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;

struct SscalTilingData {
    uint32_t n;
    uint32_t useCoreNum;
    uint32_t startOffset[40];
    uint32_t calNum[40];
    float alpha;
};

SscalTilingData CalTilingData(uint32_t totalEleNum, uint32_t vecCoreNum, float alpha)
{
    SscalTilingData tilingData;
    tilingData.n = totalEleNum;
    tilingData.useCoreNum = 0;
    tilingData.alpha = alpha;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t totalBlockNum = totalEleNum / ELEMENTS_PER_BLOCK_TILING;
    uint32_t remainNum = totalEleNum % ELEMENTS_PER_BLOCK_TILING;

    if (totalBlockNum == 0) {
        tilingData.calNum[0] = remainNum;
        tilingData.useCoreNum = 1;
    } else if (totalBlockNum <= vecCoreNum) {
        for (uint32_t i = 0; i < totalBlockNum; i++) {
            tilingData.startOffset[i] = ELEMENTS_PER_BLOCK_TILING * i;
            tilingData.calNum[i] = ELEMENTS_PER_BLOCK_TILING;
        }
        tilingData.calNum[totalBlockNum - 1] += remainNum;
        tilingData.useCoreNum = totalBlockNum;
    } else {
        uint64_t blockNumEachCore;
        uint32_t remainBlock;

        blockNumEachCore = totalBlockNum / vecCoreNum;
        remainBlock = totalBlockNum % vecCoreNum;

        uint64_t currOffset = 0;
        uint64_t currCalNum = 0;

        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainBlock) {
                currCalNum = (blockNumEachCore + 1) * ELEMENTS_PER_BLOCK_TILING;
            } else {
                currCalNum = blockNumEachCore * ELEMENTS_PER_BLOCK_TILING;
            }
            tilingData.startOffset[i] = currOffset;
            tilingData.calNum[i] = currCalNum;
            currOffset += currCalNum;
        }
        tilingData.calNum[vecCoreNum - 1] += remainNum;
        tilingData.useCoreNum = vecCoreNum;
    }
    return tilingData;
}

aclblasStatus_t aclblasSscal(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    if (handle == nullptr) {
        LOG_PRINT("aclblasSscal: handle is nullptr\n");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (alpha == nullptr) {
        LOG_PRINT("aclblasSscal: alpha must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        LOG_PRINT("aclblasSscal: x must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;

    SscalTilingData tiling = CalTilingData(n, numBlocks, *alpha);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(SscalTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SscalTilingData), &tiling, sizeof(SscalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    sscal_kernel_do(reinterpret_cast<uint8_t*>(x), nullptr, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCsscal(
    aclblasHandle_t handle, int n, const float* alpha, aclblasComplex* x, int incx)
{
    if (alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;
    uint32_t totalFloatNum = static_cast<uint32_t>(n) * 2;

    SscalTilingData tiling = CalTilingData(totalFloatNum, numBlocks, *alpha);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(SscalTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SscalTilingData), &tiling, sizeof(SscalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    sscal_kernel_do(reinterpret_cast<uint8_t*>(x), nullptr, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}