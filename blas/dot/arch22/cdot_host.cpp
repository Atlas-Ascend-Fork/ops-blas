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
 * \file cdot_host.cpp
 * \brief Complex dot product: cdotu (unconjugated) and cdotc (conjugated)
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

void cdot_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* result, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;
constexpr uint32_t COMPLEX_NUM = 2;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t IS_NOT_CONJ = 0;
constexpr uint32_t IS_CONJ = 1;

struct CdotTilingData {
    uint32_t n;
    uint32_t coreNum;
    uint32_t isConj;
    uint32_t startOffset[DEFAULT_VECTOR_NUM];
    uint32_t calNum[DEFAULT_VECTOR_NUM];
};

CdotTilingData CalCdotTilingData(uint32_t n, uint32_t vecCoreNum, uint32_t isConj)
{
    CdotTilingData tilingData{};
    tilingData.n = n;
    tilingData.isConj = isConj;
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

    uint32_t complexNum = n / COMPLEX_NUM;
    uint32_t numPerCore = complexNum / vecCoreNum;
    uint32_t remainNum = complexNum % vecCoreNum;

    if (numPerCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            tilingData.calNum[i] = COMPLEX_NUM;
            tilingData.startOffset[i] = COMPLEX_NUM * i;
        }
        tilingData.coreNum = remainNum;
    } else {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currCalNum = (numPerCore + 1) * COMPLEX_NUM;
            } else {
                currCalNum = numPerCore * COMPLEX_NUM;
            }
            tilingData.calNum[i] = currCalNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currCalNum;
        }
        tilingData.coreNum = vecCoreNum;
    }
    return tilingData;
}

static aclblasStatus_t LaunchCdot(
    bool isConj, _aclblas_handle* h, const int64_t n, const aclblasComplex* x, const aclblasComplex* y,
    aclblasComplex* result)
{
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;
    size_t workspaceBytes = 1024;
    size_t tilingBytes = sizeof(CdotTilingData);

    CHECK_RET(
        workspaceBytes + tilingBytes <= GetEffectiveWorkspaceSize(h),
        LOG_PRINT("workspace need %zu, available %zu\n", workspaceBytes + tilingBytes, GetEffectiveWorkspaceSize(h));
        return ACLBLAS_STATUS_EXECUTION_FAILED);

    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* tilingDevice = workspaceDevice + workspaceBytes;

    CdotTilingData tiling = CalCdotTilingData(static_cast<uint32_t>(n * 2), numBlocks, isConj ? IS_CONJ : IS_NOT_CONJ);

    aclError aclRet = aclrtMemcpy(tilingDevice, tilingBytes, &tiling, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    cdot_kernel_do(reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
                    reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(result),
                    workspaceDevice, tilingDevice, numBlocks, useStream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCdotu(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y,
    int incy, aclblasComplex* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (result == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        aclblasComplex zero = {0.0f, 0.0f};
        aclError memRet = aclrtMemcpy(result, sizeof(aclblasComplex), &zero, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
        return (memRet == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (incx != 1 || incy != 1) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return LaunchCdot(false, handle, static_cast<int64_t>(n), x, y, result);
}

aclblasStatus_t aclblasCdotc(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y,
    int incy, aclblasComplex* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (result == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        aclblasComplex zero = {0.0f, 0.0f};
        aclError memRet = aclrtMemcpy(result, sizeof(aclblasComplex), &zero, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
        return (memRet == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (incx != 1 || incy != 1) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || y == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return LaunchCdot(true, handle, static_cast<int64_t>(n), x, y, result);
}