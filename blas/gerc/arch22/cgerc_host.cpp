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
 * \file cgerc_host.cpp
 * \brief Host side implementation for cgerc operator
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void cgerc_kernel_do(
    uint8_t* d_x, uint8_t* d_y, uint8_t* d_offset, uint8_t* d_A, uint8_t* work_space, uint8_t* tiling_gm,
    uint32_t num_blocks, void* stream);

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;
constexpr uint32_t MAX_DATA_COUNT = 32 * 1024 / 4;
constexpr uint32_t NUM_FLAG = 2;
constexpr uint32_t NUM_INBYTES = 4;

// Tiling data structure
struct CgercTilingData {
    uint32_t m;
    uint32_t n;
    float alphaReal;
    float alphaImag;
    uint64_t startOffset[40];
    uint64_t calNum[40];
};

// Tiling calculation
CgercTilingData CalCgercTilingData(uint32_t m, uint32_t n, float alphaReal, float alphaImag, uint32_t vecCoreNum)
{
    CgercTilingData tilingData;
    memset(&tilingData, 0, sizeof(CgercTilingData));

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    // 按复数作tiling，以下单位都是复数
    uint64_t rowNumEachCore = m / vecCoreNum;
    uint64_t remainRowNum = m % vecCoreNum;

    if (rowNumEachCore == 0) {
        for (uint64_t i = 0; i < remainRowNum; i++) {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i;
        }
    } else {
        uint64_t currOffset = 0;
        uint64_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainRowNum) {
                currNum = rowNumEachCore + 1;
            } else {
                currNum = rowNumEachCore;
            }
            tilingData.calNum[i] = currNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currNum;
        }
    }

    tilingData.m = m;
    tilingData.n = n;
    tilingData.alphaReal = alphaReal;
    tilingData.alphaImag = alphaImag;

    return tilingData;
}

// Create gather offset for cgerc
std::vector<uint32_t> CreateCgercOffset()
{
    uint32_t gatherOffsetSize = MAX_DATA_COUNT;
    std::vector<uint32_t> offsetData(gatherOffsetSize);

    for (uint32_t i = 0; i < gatherOffsetSize / NUM_FLAG; i++) {
        offsetData[NUM_FLAG * i] = NUM_INBYTES * i;
        offsetData[NUM_FLAG * i + 1] = NUM_INBYTES * (i + gatherOffsetSize / NUM_FLAG);
    }

    return offsetData;
}

aclblasStatus_t aclblasCgerc(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* h = handle;
    aclrtStream useStream = h->stream;

    float alphaReal = alpha->real;
    float alphaImag = alpha->imag;

    uint32_t numBlocks = 8;

    CgercTilingData tiling = CalCgercTilingData(m, n, alphaReal, alphaImag, numBlocks);

    std::vector<uint32_t> offset = CreateCgercOffset();
    size_t offsetByteSize = offset.size() * sizeof(uint32_t);
    size_t workspaceSize = 1024;

    uint8_t* offsetDevice = nullptr;
    uint8_t* workspaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&offsetDevice, offsetByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(offsetDevice);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CgercTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workspaceDevice);
              aclrtFree(offsetDevice); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(offsetDevice, offsetByteSize, offset.data(), offsetByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              aclrtFree(workspaceDevice); aclrtFree(offsetDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CgercTilingData), &tiling, sizeof(CgercTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              aclrtFree(workspaceDevice); aclrtFree(offsetDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    cgerc_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), offsetDevice, reinterpret_cast<uint8_t*>(A),
        workspaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(tilingDevice); aclrtFree(workspaceDevice); aclrtFree(offsetDevice);
              return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(offsetDevice);
    aclrtFree(workspaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
