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
 * \file csrot_host.cpp
 * \brief Complex vector rotation host implementation
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

void csrot_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workSpace, uint8_t* tilingGm, uint32_t numBlocks, void* stream);

// Tiling data structure from sip
struct RotTilingData {
    int32_t elementCount;
    float cosValue;
    float sinValue;
};

aclblasStatus_t aclblasCsrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const float* s)
{
    if (c == nullptr || s == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t numBlocks = 8;
    size_t workSpaceSize = 32;

    RotTilingData tiling;
    // kernel 按交织复数的 float 分量逐分量旋转（x'_r = c*x_r + s*y_r 等），
    // elementCount 必须是 float 分量数 2*n；传 n 会导致仅前一半复数元素
    // 被旋转、后一半保持原值（issue #358 测试缓冲修正后暴露）
    tiling.elementCount = static_cast<int32_t>(n) * 2;
    tiling.cosValue = *c;
    tiling.sinValue = *s;

    uint8_t* workSpaceDevice = nullptr;
    uint8_t* tilingDevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&workSpaceDevice, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet = aclrtMalloc((void**)&tilingDevice, sizeof(RotTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", aclRet); aclrtFree(workSpaceDevice);
              return ACLBLAS_STATUS_ALLOC_FAILED);

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(RotTilingData), &tiling, sizeof(RotTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", aclRet); aclrtFree(tilingDevice);
              aclrtFree(workSpaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    csrot_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), workSpaceDevice, tilingDevice, numBlocks,
        useStream);
    aclRet = aclrtSynchronizeStream(useStream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(tilingDevice); aclrtFree(workSpaceDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    aclrtFree(workSpaceDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
