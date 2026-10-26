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
 * \file caxpy_host.cpp
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

void caxpy_kernel_do(
    uint8_t* x, uint8_t* maskBuf, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint32_t COMPLEX_NUM = 2;
constexpr uint32_t K_FACTOR_4 = 4;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t MAX_DATA_COUNT = 38 * 1024 / sizeof(float);

struct CaxpyTilingData {
    uint32_t n;
    float alphaReal;
    float alphaImag;
    uint32_t startOffset[40];
    uint32_t calNum[40];
};

CaxpyTilingData CalTilingData(uint32_t n, uint32_t vecCoreNum, float alphaReal, float alphaImag)
{
    CaxpyTilingData tilingData;
    tilingData.n = n * COMPLEX_NUM;
    tilingData.alphaReal = alphaReal;
    tilingData.alphaImag = alphaImag;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t rowNumEachCore = n / vecCoreNum;
    uint32_t remainRowNum = n % vecCoreNum;

    if (rowNumEachCore == 0) {
        for (uint32_t i = 0; i < remainRowNum; i++) {
            tilingData.calNum[i] = COMPLEX_NUM;
            tilingData.startOffset[i] = i * COMPLEX_NUM;
        }
    } else {
        uint32_t currOffset = 0;
        uint32_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainRowNum) {
                currNum = rowNumEachCore + 1;
            } else {
                currNum = rowNumEachCore;
            }
            tilingData.calNum[i] = currNum * COMPLEX_NUM;
            tilingData.startOffset[i] = currOffset;
            currOffset += currNum * COMPLEX_NUM;
        }
    }

    return tilingData;
}

void GenMaskData(uint32_t* maskData)
{
    uint32_t offsetNum = COMPLEX_NUM;
    uint32_t complexCount = MAX_DATA_COUNT / COMPLEX_NUM;
    for (uint32_t i = 0; i < complexCount; i++) {
        maskData[offsetNum * i] = K_FACTOR_4 * i;
        maskData[offsetNum * i + 1] = K_FACTOR_4 * (i + complexCount);
    }
}

aclblasStatus_t aclblasCaxpy(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, aclblasComplex* y,
    int incy)
{
    if (handle == nullptr) {
        LOG_PRINT("aclblasCaxpy: handle is nullptr\n");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // !692 检视意见修复：n<0 转 uint32 后回绕为超大值，tiling 按约 43 亿元素切分导致设备端
    // 越界读写；对齐 arch35 拒绝负 n 并对 n=0 快速返回
    if (n < 0) {
        LOG_PRINT("aclblasCaxpy: n must be non-negative, got %d\n", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // !692 检视意见修复：x/y 为空时指针原样传入内核造成设备侧空指针访问
    if (x == nullptr || y == nullptr) {
        LOG_PRINT("aclblasCaxpy: x and y must not be nullptr when n is positive\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // !692 检视意见修复：arch22 内核仅按连续布局切分（tiling 不含 inc 参数），incx/incy
    // 非 1 时此前被静默按连续布局处理、结果错误，显式拒绝
    if (incx != 1 || incy != 1) {
        LOG_PRINT("aclblasCaxpy: arch22 kernel only supports contiguous layout, got incx=%d incy=%d\n", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = DEFAULT_VECTOR_NUM;

    float alphaReal = alpha->real;
    float alphaImag = alpha->imag;

    CaxpyTilingData tiling = CalTilingData(static_cast<uint32_t>(n), numBlocks, alphaReal, alphaImag);

    uint32_t maskSize = MAX_DATA_COUNT * sizeof(uint32_t) * COMPLEX_NUM;

    uint8_t* maskDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    uint32_t* maskHost = new uint32_t[MAX_DATA_COUNT * COMPLEX_NUM];
    GenMaskData(maskHost);

    aclError aclRet = aclrtMalloc((void**)&maskDevice, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); delete[] maskHost;
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CaxpyTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(maskDevice);
              delete[] maskHost; return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              aclrtFree(maskDevice); delete[] maskHost; return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CaxpyTilingData), &tiling, sizeof(CaxpyTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              aclrtFree(maskDevice); delete[] maskHost; return ACLBLAS_STATUS_INTERNAL_ERROR);

    caxpy_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), maskDevice, reinterpret_cast<uint8_t*>(y), nullptr,
        tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(tilingDevice); aclrtFree(maskDevice); delete[] maskHost; return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(maskDevice);
    aclrtFree(tilingDevice);
    delete[] maskHost;

    return ACLBLAS_STATUS_SUCCESS;
}