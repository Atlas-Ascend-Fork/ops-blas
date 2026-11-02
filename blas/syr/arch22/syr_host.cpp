/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void ssyr_kernel_do(
    uint8_t* gm_x, uint8_t* gm_A, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t WORKSPACE_SIZE = 1024;

struct SsyrTilingData {
    uint32_t uplo;
    uint32_t n;
    float alpha;
    uint32_t coreNum;
};

aclblasStatus_t aclblasSsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const float* alpha, const float* x, const int incx,
    float* A, const int lda)
{
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (n < 0 || lda < std::max(1, n) || incx == 0 || alpha == nullptr || x == nullptr || A == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    // issue #736: alpha 无条件 D2H 读回对 host 指针会 memcpy 失败并误报 INTERNAL_ERROR；
    // 改用位置探测共享 helper（host 直接解引用，device 异步 D2H），对齐 csyr/#510 ssyr 口径
    float alphaVal = 0.0f;
    const aclblasStatus_t alphaStatus = ReadScalarFromHostOrDevice(alpha, useStream, alphaVal);
    if (alphaStatus != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasSsyr: resolve alpha failed. STATUS: %d\n", static_cast<int>(alphaStatus));
        return alphaStatus;
    }

    uint32_t vecCoreNum = DEFAULT_VECTOR_NUM;

    SsyrTilingData tiling;
    tiling.uplo = (uplo == ACLBLAS_UPPER) ? 1 : 0;
    tiling.n = static_cast<uint32_t>(n);
    tiling.alpha = alphaVal;
    tiling.coreNum = vecCoreNum;

    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&workSpaceDevice, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(SsyrTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workSpaceDevice);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(SsyrTilingData), &tiling, sizeof(SsyrTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              aclrtFree(workSpaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    ssyr_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(A), workSpaceDevice, tilingDevice,
        vecCoreNum, useStream);

    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(tilingDevice); aclrtFree(workSpaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
