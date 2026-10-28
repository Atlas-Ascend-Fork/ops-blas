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
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline void Csyr2PrepareVector(
    const aclblasComplex* host, int n, int inc, std::vector<aclblasComplex>& buf, const aclblasComplex*& upload,
    int& useInc, size_t& bytes)
{
    const int absInc = std::abs(inc);
    if (host == nullptr) {
        upload = nullptr;
        bytes = 0;
        useInc = inc;
        return;
    }
    /* 负步长原样上传，让 kernel Gather 打到 TC_INC。不要收成 inc=1。 */
    (void)buf;
    upload = host;
    useInc = inc;
    bytes = static_cast<size_t>((n - 1) * absInc + 1) * sizeof(aclblasComplex);
}

inline aclError Csyr2CopyToDevice(void** devPtr, const void* hostPtr, size_t bytes)
{
    aclError ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    return aclrtMemcpy(*devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

inline void Csyr2CleanupDevice(void* dX, void* dY, void* dA)
{
    if (dX) {
        aclrtFree(dX);
    }
    if (dY) {
        aclrtFree(dY);
    }
    if (dA) {
        aclrtFree(dA);
    }
}

struct Csyr2DeviceBuf {
    void* dX = nullptr;
    void* dY = nullptr;
    void* dA = nullptr;
    size_t aBytes = 0;
    int useIncx = 1;
    int useIncy = 1;

    ~Csyr2DeviceBuf() { Csyr2CleanupDevice(dX, dY, dA); }
};

inline aclblasStatus_t Csyr2Upload(
    Csyr2DeviceBuf& dev, int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    const aclblasComplex* A, int lda)
{
    std::vector<aclblasComplex> xBuf;
    std::vector<aclblasComplex> yBuf;
    const aclblasComplex* xUpload = nullptr;
    const aclblasComplex* yUpload = nullptr;
    size_t xBytes = 0;
    size_t yBytes = 0;
    Csyr2PrepareVector(x, n, incx, xBuf, xUpload, dev.useIncx, xBytes);
    Csyr2PrepareVector(y, n, incy, yBuf, yUpload, dev.useIncy, yBytes);

    const int allocLda = std::max(1, std::max(n, lda));
    dev.aBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(std::max(n, 1)) * sizeof(aclblasComplex);

    if (xUpload != nullptr && Csyr2CopyToDevice(&dev.dX, xUpload, xBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (yUpload != nullptr && Csyr2CopyToDevice(&dev.dY, yUpload, yBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (A != nullptr && Csyr2CopyToDevice(&dev.dA, A, dev.aBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t Csyr2Launch(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, Csyr2DeviceBuf& dev, int lda)
{
    return aclblasCsyr2(
        handle, uplo, n, alpha, static_cast<const aclblasComplex*>(dev.dX), dev.useIncx,
        static_cast<const aclblasComplex*>(dev.dY), dev.useIncy, static_cast<aclblasComplex*>(dev.dA), lda);
}

inline aclblasStatus_t aclblasCsyr2_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    }

    Csyr2DeviceBuf dev;
    const aclblasStatus_t up = Csyr2Upload(dev, n, x, incx, y, incy, A, lda);
    if (up != ACLBLAS_STATUS_SUCCESS) {
        return up;
    }
    const aclblasStatus_t ret = Csyr2Launch(handle, uplo, n, alpha, dev, lda);
    aclrtSynchronizeDevice();
    if (A != nullptr && dev.dA != nullptr) {
        aclrtMemcpy(A, dev.aBytes, dev.dA, dev.aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    return ret;
}
