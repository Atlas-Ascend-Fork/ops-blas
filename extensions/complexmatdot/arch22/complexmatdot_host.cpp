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
 * \file complexmatdot_host.cpp
 * \brief Complex matrix dot product host implementation (extensions)
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "complexmatdot_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

constexpr uint32_t COMPLEX_NUM = 2;

constexpr uint32_t MAX_DATA_COUNT = 27 * 1024 / sizeof(float);

constexpr uint32_t MUL_NUM = 2;

constexpr uint32_t FOUR_NUM = 4;
struct ComplexMatDotTilingData {
    uint32_t m;
    uint32_t n;
    uint64_t startOffset[40];
    uint32_t calNum[40]; // num in FP32 format
};

static void CalTilingData(ComplexMatDotTilingData& tilingData, uint32_t m, uint32_t n, uint32_t vecCoreNum)
{
    tilingData.m = m;
    tilingData.n = n;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }

    // Initialize arrays
    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t numEachCore = m * n / vecCoreNum; // num of complex
    uint32_t remainNum = m * n - vecCoreNum * numEachCore;

    if (numEachCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i * COMPLEX_NUM; // each complex has 2 FP32 elements
        }
    } else {
        uint64_t currOffset = 0;
        uint64_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currNum = numEachCore + 1;
            } else {
                currNum = numEachCore;
            }
            tilingData.calNum[i] = currNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currNum * COMPLEX_NUM;
        }
    }
}

static std::vector<uint32_t> CreateAugComplexMatDot()
{
    uint32_t complexCount = MAX_DATA_COUNT / 2;

    std::vector<uint32_t> augData(MAX_DATA_COUNT);

    for (uint32_t i = 0; i < complexCount; i++) {
        augData[MUL_NUM * i] = FOUR_NUM * i;
        augData[MUL_NUM * i + 1] = FOUR_NUM * (i + complexCount);
    }
    return augData;
}

// issue #779: 设备缓冲分配与拷贝抽出为独立函数，控制 aclblasComplexMatDot 函数体规模
static aclblasStatus_t CopyTilingAndAugToDevice(
    const ComplexMatDotTilingData& tiling, const std::vector<uint32_t>& aug, uint8_t** augDevice,
    uint8_t** tilingDevice)
{
    const size_t augByteSize = aug.size() * sizeof(uint32_t);

    aclError aclRet = aclrtMalloc((void**)augDevice, augByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMalloc((void**)tilingDevice, sizeof(ComplexMatDotTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        aclrtFree(*augDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet = aclrtMemcpy(*augDevice, augByteSize, aug.data(), augByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet);
        aclrtFree(*tilingDevice);
        aclrtFree(*augDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(
        *tilingDevice, sizeof(ComplexMatDotTilingData), &tiling, sizeof(ComplexMatDotTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet);
        aclrtFree(*tilingDevice);
        aclrtFree(*augDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasComplexMatDot(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* matx, const aclblasComplex* maty,
    aclblasComplex* result)
{
    // issue #581: handle 未判空即解引用会主机侧段错误
    if (handle == nullptr) {
        LOG_PRINT("aclblasComplexMatDot: handle is nullptr\n");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    // issue #581: m/n 必须为正值，且 m*n 需在 uint32 内可表示——负数转 uint32 后变成超大值、
    // 乘积回绕会使 tiling 覆盖的元素数与真实矩阵不符，结果静默错误
    if (m <= 0 || n <= 0) {
        LOG_PRINT("aclblasComplexMatDot: invalid m=%d, n=%d, both must be positive\n", m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (static_cast<uint64_t>(m) * static_cast<uint64_t>(n) > 0xFFFFFFFFULL) {
        LOG_PRINT(
            "aclblasComplexMatDot: m*n=%lld exceeds uint32 range\n",
            static_cast<long long>(static_cast<uint64_t>(m) * static_cast<uint64_t>(n)));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // issue #711: m/n 为正时 matx/maty/result 任一为空会把空指针作为 GM 地址下发内核，
    // 造成设备侧非法访问；对齐 #581 修复的 caxpy 指针判空口径
    if (matx == nullptr || maty == nullptr || result == nullptr) {
        LOG_PRINT("aclblasComplexMatDot: matx, maty and result must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* h = handle;
    aclrtStream useStream = h->stream;

    // issue #716: 核数不得硬编码
    uint32_t numBlocks = GetAivCoreCount();

    ComplexMatDotTilingData tiling;
    CalTilingData(tiling, m, n, numBlocks);
    std::vector<uint32_t> aug = CreateAugComplexMatDot();

    uint8_t* augDevice = nullptr;
    uint8_t* tilingDevice = nullptr;
    const aclblasStatus_t copyRet = CopyTilingAndAugToDevice(tiling, aug, &augDevice, &tilingDevice);
    if (copyRet != ACLBLAS_STATUS_SUCCESS) {
        return copyRet;
    }

    complex_mat_dot_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(matx)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(maty)), augDevice, reinterpret_cast<uint8_t*>(result),
        tilingDevice, numBlocks, useStream);
    aclError aclRet = aclrtSynchronizeStream(useStream);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
        aclrtFree(tilingDevice);
        aclrtFree(augDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(augDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
