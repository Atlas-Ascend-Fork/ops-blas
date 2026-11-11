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

#include <cstddef>
#include <cstdlib>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

inline std::unique_ptr<DeviceBuffer> CsymvCopyToDevice(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr || bytes == 0) {
        return nullptr;
    }
    auto buffer = std::make_unique<DeviceBuffer>(bytes);
    buffer->copyFromHost(hostPtr, bytes);
    return buffer;
}

struct CsymvDeviceBuffers {
    std::unique_ptr<DeviceBuffer> alpha;
    std::unique_ptr<DeviceBuffer> A;
    std::unique_ptr<DeviceBuffer> x;
    std::unique_ptr<DeviceBuffer> beta;
    std::unique_ptr<DeviceBuffer> y;

    const aclblasComplex* alphaArg = nullptr;
    const aclblasComplex* aArg = nullptr;
    const aclblasComplex* xArg = nullptr;
    const aclblasComplex* betaArg = nullptr;
    aclblasComplex* yArg = nullptr;
    size_t yBytes = 0;

    void Prepare(
        int n, const aclblasComplex* alphaHost, const aclblasComplex* aHost, int lda, const aclblasComplex* xHost,
        int incx, const aclblasComplex* betaHost, aclblasComplex* yHost, int incy, bool deviceScalars)
    {
        const size_t aBytes = n > 0 && lda > 0 ? static_cast<size_t>(lda) * n * sizeof(aclblasComplex) : 0;
        const size_t xBytes = n > 0 ? (1U + static_cast<size_t>(n - 1) * std::abs(incx)) * sizeof(aclblasComplex) : 0;
        yBytes = n > 0 ? (1U + static_cast<size_t>(n - 1) * std::abs(incy)) * sizeof(aclblasComplex) : 0;

        if (deviceScalars) {
            alpha = CsymvCopyToDevice(alphaHost, sizeof(aclblasComplex));
            beta = CsymvCopyToDevice(betaHost, sizeof(aclblasComplex));
        }
        A = CsymvCopyToDevice(aHost, aBytes);
        x = CsymvCopyToDevice(xHost, xBytes);
        y = CsymvCopyToDevice(yHost, yBytes);

        alphaArg = alpha ? static_cast<const aclblasComplex*>(alpha->ptr()) : alphaHost;
        aArg = A ? static_cast<const aclblasComplex*>(A->ptr()) : nullptr;
        xArg = x ? static_cast<const aclblasComplex*>(x->ptr()) : nullptr;
        betaArg = beta ? static_cast<const aclblasComplex*>(beta->ptr()) : betaHost;
        yArg = y ? static_cast<aclblasComplex*>(y->ptr()) : nullptr;
    }

    void CopyYToHost(aclblasComplex* yHost)
    {
        if (y && yHost != nullptr && yBytes != 0) {
            y->copyToHost(yHost, yBytes);
        }
    }
};

inline aclblasStatus_t aclblasCsymv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy,
    bool deviceScalars)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    CsymvDeviceBuffers buffers;
    buffers.Prepare(n, alpha, A, lda, x, incx, beta, y, incy, deviceScalars);
    aclblasStatus_t status = aclblasCsymv(
        handle, uplo, n, buffers.alphaArg, buffers.aArg, lda, buffers.xArg, incx, buffers.betaArg, buffers.yArg, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || stream == nullptr) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    buffers.CopyYToHost(y);
    return ACLBLAS_STATUS_SUCCESS;
}
