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
 * \file sdot_host.cpp
 * \brief Real vector dot product: result = sum(x[i] * y[i])
 */

#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void sdot_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* result, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;

struct SdotTilingData {
    uint32_t n;
    uint32_t coreNum;
    uint32_t isconj;
    uint32_t startOffset[DEFAULT_VECTOR_NUM];
    uint32_t calNum[DEFAULT_VECTOR_NUM];
};

SdotTilingData CalSdotTilingData(uint32_t n, uint32_t vecCoreNum)
{
    SdotTilingData tilingData;
    tilingData.n = n;
    tilingData.isconj = 0;
    tilingData.coreNum = 0;

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    if (vecCoreNum > DEFAULT_VECTOR_NUM) {
        vecCoreNum = DEFAULT_VECTOR_NUM;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t numPerCore = n / vecCoreNum;
    uint32_t remainNum = n % vecCoreNum;

    if (numPerCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i;
        }
        tilingData.coreNum = remainNum;
    } else {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currCalNum = numPerCore + 1;
            } else {
                currCalNum = numPerCore;
            }
            tilingData.calNum[i] = currCalNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currCalNum;
        }
        tilingData.coreNum = vecCoreNum;
    }
    return tilingData;
}

aclblasStatus_t aclblasSdot(
    aclblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy,
    float* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (result == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        float zero = 0.0f;
        aclError memRet = aclrtMemcpy(result, sizeof(float), &zero, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        return (memRet == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (incx != 1 || incy != 1) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;
    size_t workspaceBytes = 1024;
    size_t tilingBytes = sizeof(SdotTilingData);

    CHECK_RET(
        workspaceBytes + tilingBytes <= GetEffectiveWorkspaceSize(h),
        LOG_PRINT("workspace need %zu, available %zu\n", workspaceBytes + tilingBytes, GetEffectiveWorkspaceSize(h));
        return ACLBLAS_STATUS_EXECUTION_FAILED);

    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* tilingDevice = workspaceDevice + workspaceBytes;

    SdotTilingData tiling = CalSdotTilingData(static_cast<uint32_t>(n), numBlocks);

    aclError aclRet = aclrtMemcpy(tilingDevice, tilingBytes, &tiling, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    sdot_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(const_cast<float*>(y)),
        reinterpret_cast<uint8_t*>(result), workspaceDevice, tilingDevice, numBlocks, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}