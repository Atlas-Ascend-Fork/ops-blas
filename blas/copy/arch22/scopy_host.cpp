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
 * \file scopy.asc
 * \brief
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void scopy_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;

struct CopyTilingData {
    uint32_t n;
    uint32_t useCoreNum;
    uint32_t startOffset[40];
    uint32_t calNum[40];
};

CopyTilingData CalTilingData(uint32_t totalEleNum, uint32_t vecCoreNum)
{
    CopyTilingData tilingData;
    tilingData.n = totalEleNum;
    tilingData.useCoreNum = 0;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }

    // Set zero for startOffset and calNum
    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }
    // Num of blocks
    uint32_t totalBlockNum = totalEleNum / ELEMENTS_PER_BLOCK_TILING;
    // Remain elements num
    uint32_t remainNum = totalEleNum % ELEMENTS_PER_BLOCK_TILING;

    if (totalBlockNum == 0) {
        // Use only 1 AIV core.
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

aclblasStatus_t aclblasScopy_legacy(
    aclblasHandle_t handle, uint8_t* x, uint8_t* y, const int64_t n, const int64_t incx, const int64_t incy)
{
    // issue #779: 对齐全库 handle 判空口径（#765），空 handle 不再静默取空流继续执行
    CHECK_RET(handle != nullptr, LOG_PRINT("aclblasScopy_legacy: handle is nullptr\n");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;
    CopyTilingData tiling = CalTilingData(n, numBlocks);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CopyTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CopyTilingData), &tiling, sizeof(CopyTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              return ACLBLAS_STATUS_INTERNAL_ERROR);

    scopy_kernel_do(x, y, nullptr, tilingDevice, numBlocks, useStream);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCcopy(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, aclblasComplex* y, int incy)
{
    // issue #779: 对齐全库 handle 判空口径（#765），空 handle 不再静默取空流继续执行
    CHECK_RET(handle != nullptr, LOG_PRINT("aclblasCcopy: handle is nullptr\n");
              return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // issue #734: 非连续步长在 arch22 kernel 向量化数据通路不支持步进（strided
    // 路径存在设备任务异常 507035），显式拒绝消除静默错算（对齐 caxpy #692/cgerc
    // #732 口径）。完整步长支持需 kernel 数据通路改造，已在 issue 留痕。
    if (incx != 1 || incy != 1) {
        LOG_PRINT("aclblasCcopy: arch22 kernel requires contiguous layout, got incx=%d incy=%d\n", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;
    // issue #734: 非连续步长走按复数分核的 strided 路径；连续步长保留原快路径
    const bool strided = (incx != 1 || incy != 1);
    CopyTilingData tiling = CalTilingData(static_cast<uint64_t>(n) * 2, numBlocks);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CopyTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CopyTilingData), &tiling, sizeof(CopyTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              return ACLBLAS_STATUS_INTERNAL_ERROR);

    scopy_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), nullptr,
        tilingDevice, numBlocks, useStream);

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
