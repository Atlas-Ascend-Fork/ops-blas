/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

/**
 * aclblasSspr 测试调度包装：Host 侧打包向量拷贝至 Device，调用 aclblasSspr
 * （x / ap 为 Device 内存，alpha 为 Host 内存 float32 指针），同步后拷回 AP。
 */
inline aclblasStatus_t SsprAllocCopyToDevice(void** dst, const void* src, size_t bytes)
{
    if (src == nullptr || bytes == 0) {
        *dst = nullptr;
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError aclRet = aclrtMalloc(dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(*dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dst);
        *dst = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSspr_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const float* alpha, const float* x, int incx, float* ap)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSspr(handle, uplo, n, alpha, x, incx, ap);
    }

    const int absIncx = std::abs(incx);
    const size_t xBytes = (static_cast<size_t>(n - 1) * static_cast<size_t>(absIncx) + 1) * sizeof(float);
    const size_t apBytes = static_cast<size_t>(n) * (n + 1) / 2 * sizeof(float);

    void* dX = nullptr;
    void* dAP = nullptr;

    aclblasStatus_t st = SsprAllocCopyToDevice(&dX, x, xBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = SsprAllocCopyToDevice(&dAP, ap, apBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        if (dX != nullptr) {
            aclrtFree(dX);
        }
        return st;
    }

    aclblasStatus_t ret = aclblasSspr(handle, uplo, n, alpha, static_cast<const float*>(dX), incx,
        static_cast<float*>(dAP));

    aclrtSynchronizeDevice();
    if (ret == ACLBLAS_STATUS_SUCCESS && ap != nullptr) {
        aclrtMemcpy(ap, apBytes, dAP, apBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (dX != nullptr) {
        aclrtFree(dX);
    }
    if (dAP != nullptr) {
        aclrtFree(dAP);
    }
    return ret;
}
