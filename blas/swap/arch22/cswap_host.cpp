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
 * \file cswap_host.cpp
 * \brief
 */

#include <climits>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void cswap_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;
constexpr uint32_t CACUL_TWO = 2;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;

struct CswapTilingData {
    uint32_t n;
    uint32_t coreNum;
    uint32_t startOffset[40];
    uint32_t calNum[40];
    int32_t incx;
    int32_t incy;
};

static CswapTilingData CalCswapTilingData(uint32_t n, uint32_t vecCoreNum, int32_t incx, int32_t incy)
{
    CswapTilingData tilingData;
    tilingData.n = n;
    tilingData.coreNum = 0;
    tilingData.incx = incx;
    tilingData.incy = incy;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t complexNum = n;
    uint32_t numPerCore = complexNum / vecCoreNum;
    uint32_t remainNum = complexNum % vecCoreNum;

    if (numPerCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            tilingData.calNum[i] = CACUL_TWO;
            tilingData.startOffset[i] = CACUL_TWO * i;
        }
        tilingData.coreNum = remainNum;
    } else {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currCalNum = (numPerCore + 1) * CACUL_TWO;
            } else {
                currCalNum = numPerCore * CACUL_TWO;
            }
            tilingData.calNum[i] = currCalNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currCalNum;
        }
        tilingData.coreNum = vecCoreNum;
    }

    return tilingData;
}

aclblasStatus_t aclblasCswap(aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        LOG_PRINT("aclblasCswap: handle is nullptr\n");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        LOG_PRINT("aclblasCswap: x and y must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        LOG_PRINT("aclblasCswap: n must be non-negative, got n=%d\n", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incx == INT_MIN || incy == 0 || incy == INT_MIN) {
        LOG_PRINT("aclblasCswap: incx and incy must be non-zero and not INT_MIN, got incx=%d, incy=%d\n", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;

    CswapTilingData tiling = CalCswapTilingData(static_cast<uint32_t>(n), numBlocks, incx, incy);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CswapTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CswapTilingData), &tiling, sizeof(CswapTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              return ACLBLAS_STATUS_INTERNAL_ERROR);

    cswap_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), nullptr, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(tilingDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}