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
 * \file nrm2_host.cpp
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

void snrm2_kernel_do(
    uint8_t* x, uint8_t* result, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;

struct Nrm2TilingData {
    uint32_t n;
    uint32_t useCoreNum;
    uint32_t startOffset[40];
    uint32_t calNum[40];
};

static Nrm2TilingData CalTilingData(uint32_t totalEleNum, uint32_t vecCoreNum)
{
    Nrm2TilingData tilingData{};
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

static aclblasStatus_t Nrm2LaunchKernel(
    const Nrm2TilingData& tiling, uint8_t* xDevice, uint8_t* resultDevice, uint32_t numBlocks, aclrtStream stream)
{
    constexpr size_t workSpaceSize = 16 * 1024 * 1024;

    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&workSpaceDevice, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(Nrm2TilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workSpaceDevice);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(Nrm2TilingData), &tiling, sizeof(Nrm2TilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              aclrtFree(workSpaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    snrm2_kernel_do(xDevice, resultDevice, workSpaceDevice, tilingDevice, numBlocks, stream);
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(tilingDevice); aclrtFree(workSpaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSnrm2(aclblasHandle_t handle, int n, const float* x, int incx, float* result)
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
    // issue #1150: the arch22 tiling/kernel carry no incx field and the kernel computes
    // on contiguous floats, so any stride other than 1 would silently evaluate on
    // x[0..n-1] instead of x[0], x[incx], ... Reject it explicitly instead; incx <= 0
    // no longer falls into the write-zero early return either. arch35 keeps its full
    // stride support (SIMD/SIMT dual path).
    if (incx != 1) {
        OP_LOGE("aclblasSnrm2", "arch22 supports incx = 1 only, got incx=%d", incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;

    Nrm2TilingData tiling = CalTilingData(static_cast<uint32_t>(n), numBlocks);

    return Nrm2LaunchKernel(
        tiling, reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(result), numBlocks,
        useStream);
}

aclblasStatus_t aclblasScnrm2(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result)
{
    // issue #1149: the complex entry had no validation at all while the sibling
    // aclblasSnrm2 above enforces the full chain; mirror that order so illegal
    // inputs return status codes instead of crashing or reaching the kernel.
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
    // issue #1150: same contiguous-memory restriction as aclblasSnrm2 (see above).
    if (incx != 1) {
        OP_LOGE("aclblasScnrm2", "arch22 supports incx = 1 only, got incx=%d", incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasScnrm2", "x must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // The tiling is fed 2*n float elements (complex64 interleaved); reject n beyond
    // the int32 half range so the element count cannot overflow (same guard as crot).
    if (n > INT32_MAX / 2) {
        OP_LOGE("aclblasScnrm2", "n too large: n=%d exceeds INT32_MAX/2", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;

    Nrm2TilingData tiling = CalTilingData(static_cast<uint32_t>(n) * 2, numBlocks);

    return Nrm2LaunchKernel(
        tiling, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(result),
        numBlocks, useStream);
}
