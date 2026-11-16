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
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

// sgemv npu wrapper: allocate device buffer and copy host data (nullable)
inline std::unique_ptr<DeviceBuffer> tryAllocAndCopy(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr)
        return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

// Performance measurement helper: warmup (5x) + event-wrapped sampling (100x)
// on the handle stream. Returns the steady-state average kernel time in
// milliseconds via *kernelMs; H2D/D2H copies stay outside the measured window.
inline aclblasStatus_t SgemvMeasureKernelMs(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* dA, int lda,
    const float* dX, int incx, const float* beta, float* dY, int incy, float* kernelMs)
{
    constexpr int kSgemvWarmup = 5;
    constexpr int kSgemvSamples = 100;

    for (int i = 0; i < kSgemvWarmup; i++) {
        aclblasStatus_t ret = aclblasSgemv(handle, trans, m, n, alpha, dA, lda, dX, incx, beta, dY, incy);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
    }

    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclrtEvent startEvent = nullptr;
    aclrtEvent stopEvent = nullptr;
    if (aclrtCreateEvent(&startEvent) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtCreateEvent(&stopEvent) != ACL_SUCCESS) {
        (void)aclrtDestroyEvent(startEvent);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtRecordEvent(startEvent, stream);
    for (int i = 0; i < kSgemvSamples; i++) {
        aclblasStatus_t ret = aclblasSgemv(handle, trans, m, n, alpha, dA, lda, dX, incx, beta, dY, incy);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            (void)aclrtDestroyEvent(startEvent);
            (void)aclrtDestroyEvent(stopEvent);
            return ret;
        }
    }
    aclrtRecordEvent(stopEvent, stream);

    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    float totalMs = 0.0f;
    if (aclrtSynchronizeEvent(stopEvent) != ACL_SUCCESS ||
        aclrtEventElapsedTime(&totalMs, startEvent, stopEvent) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else {
        *kernelMs = totalMs / static_cast<float>(kSgemvSamples);
    }
    (void)aclrtDestroyEvent(startEvent);
    (void)aclrtDestroyEvent(stopEvent);
    return status;
}

inline aclblasStatus_t aclblasSgemv_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy, float* kernelMs = nullptr)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m <= 0 || n <= 0) {
        return aclblasSgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    const int allocLda = std::max(lda, 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t aBytes = static_cast<size_t>(allocLda) * std::max(1, n) * sizeof(float);
    const size_t xBytes = static_cast<size_t>((xCount - 1) * std::abs(incx) + 1) * sizeof(float);
    const size_t yBytes = static_cast<size_t>((yCount - 1) * std::abs(incy) + 1) * sizeof(float);

    auto dA = tryAllocAndCopy(a, aBytes);
    auto dX = tryAllocAndCopy(x, xBytes);
    auto dY = tryAllocAndCopy(y, yBytes);
    const float* dAPtr = dA ? static_cast<const float*>(dA->ptr()) : nullptr;
    const float* dXPtr = dX ? static_cast<const float*>(dX->ptr()) : nullptr;
    float* dYPtr = dY ? static_cast<float*>(dY->ptr()) : nullptr;

    // Performance cases (TC_PF_*): warmup + event-wrapped sampling on the device
    // buffers, average kernel time reported via *kernelMs.
    aclblasStatus_t ret =
        (kernelMs != nullptr) ?
            SgemvMeasureKernelMs(handle, trans, m, n, alpha, dAPtr, lda, dXPtr, incx, beta, dYPtr, incy, kernelMs) :
            aclblasSgemv(handle, trans, m, n, alpha, dAPtr, lda, dXPtr, incx, beta, dYPtr, incy);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    if (y != nullptr && dY) {
        dY->copyToHost(y, yBytes);
    }

    return ret;
}
