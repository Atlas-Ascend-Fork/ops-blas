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

// ACL 返回码辅助：非 ACL_SUCCESS 时打印（G.FMT.04 一行一条语句合规形态）
static inline bool ExpectAclOk(aclError got, const char* what)
{
    if (got == ACL_SUCCESS) {
        return true;
    }
    LOG_PRINT("aclblasSscal: %s failed. ERROR: %d\n", what, static_cast<int>(got));
    return false;
}

SscalTilingData CalTilingData(uint32_t totalEleNum, uint32_t vecCoreNum, float alpha)
{
    SscalTilingData tilingData{};
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
    if (n < 0) {
        LOG_PRINT("aclblasSscal: n must not be negative\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
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

    // issue #1324: alpha 支持 Host/Device 双位置（对齐 arch35 #1111 同族口径，
    // ReadScalarFromHostOrDevice 为 #652 既有 helper）：host 指针直接解引用，
    // device 指针异步 D2H + stream 同步后参与 tiling，消除 host 侧对 device
    // 指针的裸解引用（段错误/垃圾值风险）
    float alphaVal = 0.0f;
    aclblasStatus_t readSt = ReadScalarFromHostOrDevice(alpha, useStream, alphaVal);
    if (readSt != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasSscal: read alpha failed, status=%d\n", readSt);
        return readSt;
    }

    SscalTilingData tiling = CalTilingData(n, numBlocks, alphaVal);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(SscalTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ExpectAclOk(aclRet, "aclrtMalloc"), return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SscalTilingData), &tiling, sizeof(SscalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (!ExpectAclOk(aclRet, "aclrtMemcpy")) {
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    sscal_kernel_do(reinterpret_cast<uint8_t*>(x), nullptr, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    if (!ExpectAclOk(aclRet, "aclrtSynchronizeStream")) {
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCsscal(aclblasHandle_t handle, int n, const float* alpha, aclblasComplex* x, int incx)
{
    // issue #1323: 对齐同文件 aclblasSscal 校验序列与 README arch22 契约，
    // 空 handle 不再解引用空指针，负 n 不再经 uint32 回绕成巨量元素
    if (handle == nullptr) {
        LOG_PRINT("aclblasCsscal: handle is nullptr\n");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        LOG_PRINT("aclblasCsscal: n must not be negative\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (alpha == nullptr) {
        LOG_PRINT("aclblasCsscal: alpha must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        LOG_PRINT("aclblasCsscal: x must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;

    // issue #1324: alpha 支持 Host/Device 双位置（对齐 arch35 #1111 同族口径），
    // 消除 host 侧对 device 指针的裸解引用（段错误/垃圾值风险）
    float alphaVal = 0.0f;
    aclblasStatus_t readSt = ReadScalarFromHostOrDevice(alpha, useStream, alphaVal);
    if (readSt != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasCsscal: read alpha failed, status=%d\n", readSt);
        return readSt;
    }

    uint32_t totalFloatNum = static_cast<uint32_t>(n) * 2;

    SscalTilingData tiling = CalTilingData(totalFloatNum, numBlocks, alphaVal);

    uint8_t* tilingDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(SscalTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ExpectAclOk(aclRet, "aclrtMalloc"), return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SscalTilingData), &tiling, sizeof(SscalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (!ExpectAclOk(aclRet, "aclrtMemcpy")) {
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    sscal_kernel_do(reinterpret_cast<uint8_t*>(x), nullptr, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    if (!ExpectAclOk(aclRet, "aclrtSynchronizeStream")) {
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
