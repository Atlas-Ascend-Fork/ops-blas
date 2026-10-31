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
 * \file cscal_host.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <new>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void cscal_kernel_do(
    uint8_t* x, uint8_t* maskBuf, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint32_t MAX_LENG_PER_UB_PROC = 6144;
constexpr uint32_t ELEMENTS_EACH_COMPLEX64 = 2;
constexpr uint32_t PING_PONG_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t COMPLEX_DATA_NUM_PER_BLOCK = 4;

struct CscalTilingData {
    int32_t n;
    int32_t incx;
    float alphaReal;
    float alphaImag;
};

CscalTilingData CalTilingData(int32_t n, int32_t incx, float alphaReal, float alphaImag)
{
    CscalTilingData tilingData;
    tilingData.n = n;
    tilingData.incx = incx;
    tilingData.alphaReal = alphaReal;
    tilingData.alphaImag = alphaImag;
    return tilingData;
}

void CreateMaskData(uint32_t* maskData)
{
    uint32_t imagBaseAddr = MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(float) +
                            MAX_LENG_PER_UB_PROC * sizeof(float) * ELEMENTS_EACH_COMPLEX64 * PING_PONG_NUM;
    uint32_t realBaseAddr = imagBaseAddr + MAX_LENG_PER_UB_PROC * sizeof(uint32_t);

    int k = 0;
    for (uint32_t i = 0; i < MAX_LENG_PER_UB_PROC; i++) {
        maskData[k++] = realBaseAddr + i * sizeof(float);
        maskData[k++] = imagBaseAddr + i * sizeof(float);
    }
}

aclblasStatus_t aclblasCscal(aclblasHandle_t handle, int n, const aclblasComplex* alpha, aclblasComplex* x, int incx)
{
    if (handle == nullptr) {
        LOG_PRINT("aclblasCscal: handle is nullptr\n");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (alpha == nullptr || x == nullptr) {
        LOG_PRINT("aclblasCscal: alpha and x must not be nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        // README 契约：n<=0 为 no-op（同时消除 kernel 侧 n=0 除零的源头）
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (incx <= 0) {
        // README 契约：incx<=0 为 no-op
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 40;

    float alphaReal = alpha->real;
    float alphaImag = alpha->imag;

    CscalTilingData tiling = CalTilingData(static_cast<int32_t>(n), incx, alphaReal, alphaImag);

    uint8_t* tilingDevice = nullptr;
    uint8_t* maskDevice = nullptr;

    // issue #712: 默认 new[] 失败抛出的 std::bad_alloc 会穿越 extern "C" ABI 边界，
    // 改为 nothrow + 判空映射 ACLBLAS_STATUS_ALLOC_FAILED
    uint32_t* maskHost = new (std::nothrow) uint32_t[MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64];
    if (maskHost == nullptr) {
        LOG_PRINT("aclblasCscal: alloc mask host buffer failed\n");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    CreateMaskData(maskHost);

    aclError aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(CscalTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); delete[] maskHost;
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc(
        (void**)&maskDevice, MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(uint32_t),
        ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              delete[] maskHost; return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(CscalTilingData), &tiling, sizeof(CscalTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(maskDevice);
              aclrtFree(tilingDevice); delete[] maskHost; return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclRet = aclrtMemcpy(
        maskDevice, MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(uint32_t), maskHost,
        MAX_LENG_PER_UB_PROC * ELEMENTS_EACH_COMPLEX64 * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(maskDevice);
              aclrtFree(tilingDevice); delete[] maskHost; return ACLBLAS_STATUS_INTERNAL_ERROR);

    cscal_kernel_do(reinterpret_cast<uint8_t*>(x), maskDevice, nullptr, tilingDevice, numBlocks, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(maskDevice); aclrtFree(tilingDevice); delete[] maskHost; return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(tilingDevice);
    aclrtFree(maskDevice);
    delete[] maskHost;

    return ACLBLAS_STATUS_SUCCESS;
}