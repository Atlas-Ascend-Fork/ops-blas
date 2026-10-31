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
#include <algorithm>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void ssyr2_kernel_do(uint8_t* gm_x, uint8_t* gm_y, uint8_t* gm_A, uint8_t* workSpace, uint8_t* tilingGm,
                     uint32_t numBlocks, void *stream);

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t WORKSPACE_SIZE = 1024;

struct Ssyr2TilingData {
    uint32_t uplo;
    uint32_t n;
    float alpha;
    uint32_t coreNum;
};

aclblasStatus_t aclblasSsyr2(aclblasHandle_t handle,
                             aclblasFillMode_t uplo,
                             const int n,
                             const float *alpha,
                             const float *x, const int incx,
                             const float *y, const int incy,
                             float *A, const int lda)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0 || lda < std::max(1, n) || incx == 0 || incy == 0 || alpha == nullptr || x == nullptr ||
        y == nullptr || A == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t vecCoreNum = DEFAULT_VECTOR_NUM;

    Ssyr2TilingData tiling;
    tiling.uplo = (uplo == ACLBLAS_UPPER) ? 1 : 0;
    tiling.n = n;
    tiling.alpha = *alpha;
    tiling.coreNum = vecCoreNum;

    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&workSpaceDevice, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(Ssyr2TilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workSpaceDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMemcpy(tilingDevice, sizeof(Ssyr2TilingData), &tiling, sizeof(Ssyr2TilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice); aclrtFree(workSpaceDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    ssyr2_kernel_do(reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
                    reinterpret_cast<uint8_t*>(const_cast<float*>(y)),
                    reinterpret_cast<uint8_t*>(A),
                    workSpaceDevice, tilingDevice, vecCoreNum, useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice); aclrtFree(workSpaceDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}