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
 * \file cgemv_batched_host.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <memory>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cgemv_batched_plan.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void cgemv_batched_kernel_do(uint8_t* A, uint8_t* x, uint8_t* mask, uint8_t* y,
                             uint8_t* workSpace, uint8_t* tilingGm,
                             uint32_t numBlocks, void *stream);

constexpr uint32_t MAX_CORE_CNT = 40;
constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

struct CgemvBatchedTilingData {
    uint32_t dtype;
    uint32_t trans;
    uint32_t m;
    uint32_t n;
    uint32_t maxMatNum;
    uint32_t calMatNum[40];
    uint32_t startMatId[40];
};

static CgemvBatchedTilingData CalTilingData(
    uint32_t batchCount, uint32_t m, uint32_t n, uint32_t dtype, uint32_t trans, uint32_t vecCoreNum)
{
    CgemvBatchedTilingData tilingData;
    tilingData.dtype = dtype;
    tilingData.trans = trans;
    tilingData.m = m;
    tilingData.n = n;

    // Calculate maxMatNum based on available UB size
    // For simplicity, we set maxMatNum to a reasonable value
    uint32_t maxMatNum = 1;                   // Can be adjusted based on actual UB size
    bool isTrans = trans == 0 ? false : true; // 0: ACLBLAS_OP_N
    bool dataType = (dtype == (uint32_t)aclDataType_t::ACL_C_64F) ? true : false;
    maxMatNum = CalMaxMatNum(isTrans, dataType, static_cast<uint32_t>(m));

    tilingData.maxMatNum = maxMatNum > 0 ? maxMatNum : 1;

    // Initialize arrays
    for (uint32_t i = 0; i < MAX_CORE_CNT; i++) {
        tilingData.startMatId[i] = 0;
        tilingData.calMatNum[i] = 0;
    }

    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > MAX_CORE_CNT ? MAX_CORE_CNT : vecCoreNum;

    // Distribute batch across cores
    uint32_t baseNum = batchCount / vecCoreNum;
    uint32_t remainNum = batchCount % vecCoreNum;

    uint32_t curMatId = 0;
    for (uint32_t i = 0; i < vecCoreNum; i++) {
        uint32_t curMatNum = 0;
        if (i < remainNum) {
            curMatNum = baseNum + 1;
        } else {
            curMatNum = baseNum;
        }
        tilingData.startMatId[i] = curMatId;
        tilingData.calMatNum[i] = curMatNum;
        curMatId += curMatNum;
    }

    return tilingData;
}

aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy, int batchCount)
{
    if (alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;

    uint32_t dtype = 1; // 1 = float
    uint32_t dtypeSize = sizeof(float);
    uint32_t transUint = (trans == ACLBLAS_OP_N) ? 0 : 1;

    CgemvBatchedTilingData tiling = CalTilingData(batchCount, m, n, dtype, transUint, numBlocks);
    // CreateCgemvBatchedMask 以 new uint32_t[] 分配主机侧 mask，交给 unique_ptr 管理，
    // 保证成功与各错误返回路径都会释放（此前全函数无 delete[]）。
    std::unique_ptr<uint32_t[]> mask(CreateCgemvBatchedMask(m, dtype, transUint));
    size_t maskSize = tiling.maxMatNum * 32 * 2 * sizeof(uint32_t);

    size_t workSpaceSize = WORKSPACE_SIZE;

    uint8_t* maskDevice = nullptr;
    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&maskDevice, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&workSpaceDevice, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(maskDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CgemvBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workSpaceDevice);
        aclrtFree(maskDevice); return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(maskDevice, maskSize, mask.get(), maskSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workSpaceDevice); aclrtFree(maskDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclRet = aclrtMemcpy(
        tilingDevice, sizeof(CgemvBatchedTilingData), &tiling, sizeof(CgemvBatchedTilingData),
        ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workSpaceDevice); aclrtFree(maskDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    cgemv_batched_kernel_do(reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), maskDevice,
                            reinterpret_cast<uint8_t*>(y), workSpaceDevice, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
        aclrtFree(workSpaceDevice); aclrtFree(maskDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(maskDevice);
    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
