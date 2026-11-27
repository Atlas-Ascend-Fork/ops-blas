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

inline void CleanupDeviceBuffers(void* dX, void* dY, void* dA)
{
    if (dX)
        aclrtFree(dX);
    if (dY)
        aclrtFree(dY);
    if (dA)
        aclrtFree(dA);
}

inline aclblasStatus_t aclblasSsyr2_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, float alpha, const float* x, int incx, const float* y,
    int incy, float* A, int lda)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSsyr2(handle, uplo, n, &alpha, x, incx, y, incy, A, lda);
    }

    std::vector<float> xBuf, yBuf;
    const float *xUpload = nullptr, *yUpload = nullptr;
    int useIncx = incx, useIncy = incy;
    size_t xBytes = 0, yBytes = 0;
    PrepareVector(x, n, incx, xBuf, xUpload, useIncx, xBytes);
    PrepareVector(y, n, incy, yBuf, yUpload, useIncy, yBytes);

    int allocLda = std::max(1, std::max(n, lda));
    size_t aBytes = static_cast<size_t>(allocLda) * n * sizeof(float);

    void *dX = nullptr, *dY = nullptr, *dA = nullptr;
    if (xUpload != nullptr && CopyToDevice(&dX, xUpload, xBytes) != ACL_SUCCESS) {
        CleanupDeviceBuffers(dX, dY, dA);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (yUpload != nullptr && CopyToDevice(&dY, yUpload, yBytes) != ACL_SUCCESS) {
        CleanupDeviceBuffers(dX, dY, dA);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (A != nullptr && CopyToDevice(&dA, A, aBytes) != ACL_SUCCESS) {
        CleanupDeviceBuffers(dX, dY, dA);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret = aclblasSsyr2(
        handle, uplo, n, &alpha, static_cast<const float*>(dX), useIncx, static_cast<const float*>(dY), useIncy,
        static_cast<float*>(dA), lda);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (ret == ACLBLAS_STATUS_SUCCESS && A != nullptr) {
        if (aclrtMemcpy(A, aBytes, dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    CleanupDeviceBuffers(dX, dY, dA);
    return ret;
}

