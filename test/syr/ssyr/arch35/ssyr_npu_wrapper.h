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
#include <cmath>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline void PrepareVector(
    const float* host, int n, int inc, std::vector<float>& buf, const float*& upload, int& useInc, size_t& bytes)
{
    int absInc = std::abs(inc);
    if (host == nullptr) {
        upload = nullptr;
        bytes = 0;
        useInc = inc;
        return;
    }
    if (inc < 0) {
        buf.resize(n);
        for (int i = 0; i < n; i++)
            buf[i] = host[(n - 1 - i) * absInc];
        upload = buf.data();
        useInc = 1;
        bytes = static_cast<size_t>(n) * sizeof(float);
        return;
    }
    upload = host;
    useInc = inc;
    bytes = static_cast<size_t>((n - 1) * absInc + 1) * sizeof(float);
}

inline aclError CopyToDevice(void** devPtr, const void* hostPtr, size_t bytes)
{
    aclError ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    return aclrtMemcpy(*devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

inline aclblasStatus_t aclblasSsyr_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, float alpha, const float* x, int incx, float* A, int lda,
    bool alphaOnDevice = false)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSsyr(handle, uplo, n, &alpha, x, incx, A, lda);
    }

    std::vector<float> xBuf;
    const float* xUpload = nullptr;
    int useIncx = incx;
    size_t xBytes = 0;
    PrepareVector(x, n, incx, xBuf, xUpload, useIncx, xBytes);

    int allocLda = std::max(1, std::max(n, lda));
    size_t aBytes = static_cast<size_t>(allocLda) * n * sizeof(float);

    void* dX = nullptr;
    void* dA = nullptr;
    if (xUpload != nullptr && CopyToDevice(&dX, xUpload, xBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (A != nullptr && CopyToDevice(&dA, A, aBytes) != ACL_SUCCESS) {
        if (dX)
            aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // issue #510: alphaOnDevice 时按设备指针传入，覆盖 Device 标量指针路径
    void* dAlpha = nullptr;
    const float* alphaArg = &alpha;
    if (alphaOnDevice) {
        if (CopyToDevice(&dAlpha, &alpha, sizeof(float)) != ACL_SUCCESS) {
            if (dX)
                aclrtFree(dX);
            if (dA)
                aclrtFree(dA);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        alphaArg = static_cast<const float*>(dAlpha);
    }

    aclblasStatus_t ret =
        aclblasSsyr(handle, uplo, n, alphaArg, static_cast<const float*>(dX), useIncx, static_cast<float*>(dA), lda);

    aclrtSynchronizeDevice();
    if (A != nullptr) {
        aclrtMemcpy(A, aBytes, dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (dAlpha)
        aclrtFree(dAlpha);
    if (dX)
        aclrtFree(dX);
    if (dA)
        aclrtFree(dA);
    return ret;
}

