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
#include <chrono>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct CtpmvDeviceBuffers {
    void* ap = nullptr;
    void* x = nullptr;

    void Release()
    {
        if (ap != nullptr)
            aclrtFree(ap);
        if (x != nullptr)
            aclrtFree(x);
        ap = nullptr;
        x = nullptr;
    }
};

inline aclblasStatus_t PrepareCtpmvBuffers(
    int n, int incx, const aclblasComplex* ap, const aclblasComplex* x, CtpmvDeviceBuffers& buffers)
{
    if (n <= 0)
        return ACLBLAS_STATUS_SUCCESS;
    const size_t apCount = static_cast<size_t>(n) * (n + 1U) / 2U;
    const size_t xCount = static_cast<size_t>(n - 1) * std::abs(incx) + 1U;
    if (ap != nullptr) {
        if (aclrtMalloc(&buffers.ap, apCount * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (aclrtMemcpy(
                buffers.ap, apCount * sizeof(aclblasComplex), ap, apCount * sizeof(aclblasComplex),
                ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            buffers.Release();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    if (x != nullptr) {
        if (aclrtMalloc(&buffers.x, xCount * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            buffers.Release();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (aclrtMemcpy(
                buffers.x, xCount * sizeof(aclblasComplex), x, xCount * sizeof(aclblasComplex),
                ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            buffers.Release();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCtpmv_npu(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, const aclblasComplex* ap, aclblasComplex* x, int incx)
{
    if (handle == nullptr || n <= 0 || incx == 0 || (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) ||
        (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) ||
        (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT)) {
        return aclblasCtpmv(handle, uplo, trans, diag, n, ap, x, incx);
    }
    CtpmvDeviceBuffers buffers;
    aclblasStatus_t status = PrepareCtpmvBuffers(n, incx, ap, x, buffers);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = aclblasCtpmv(
        handle, uplo, trans, diag, n, static_cast<const aclblasComplex*>(buffers.ap),
        static_cast<aclblasComplex*>(buffers.x), incx);
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && x != nullptr) {
        const size_t xCount = static_cast<size_t>(n - 1) * std::abs(incx) + 1U;
        if (aclrtMemcpy(
                x, xCount * sizeof(aclblasComplex), buffers.x, xCount * sizeof(aclblasComplex),
                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            status = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    buffers.Release();
    return status;
}

inline aclblasStatus_t PrepareCtpmvBenchmarkInputs(
    const aclblasComplex* x, size_t xCount, size_t slotCount, size_t slotStride, CtpmvDeviceBuffers& buffers)
{
    // Each in-place invocation starts from the original input, without copies
    // or synchronization between timed launches.
    std::vector<aclblasComplex> inputs(slotCount * slotStride);
    for (size_t i = 0; i < slotCount; ++i) {
        std::copy_n(x, xCount, inputs.data() + i * slotStride);
    }
    const size_t bytes = inputs.size() * sizeof(aclblasComplex);
    if (aclrtMalloc(&buffers.x, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(buffers.x, bytes, inputs.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ReadCtpmvBenchmarkOutput(
    aclblasComplex* x, size_t xCount, size_t slotCount, size_t slotStride, const CtpmvDeviceBuffers& buffers)
{
    {
        const auto* last = static_cast<const aclblasComplex*>(buffers.x) + (slotCount - 1U) * slotStride;
        if (aclrtMemcpy(
                x, xCount * sizeof(aclblasComplex), last, xCount * sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCtpmv_npu_benchmark(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, const aclblasComplex* ap, aclblasComplex* x, int incx, int warmup, int iterations,
    float* averageUs)
{
    if (averageUs == nullptr || warmup < 0 || iterations <= 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    CtpmvDeviceBuffers buffers;
    aclblasStatus_t status = PrepareCtpmvBuffers(n, incx, ap, nullptr, buffers);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    const size_t xCount = static_cast<size_t>(n - 1) * std::abs(incx) + 1U;
    const size_t slotCount = static_cast<size_t>(warmup) + static_cast<size_t>(iterations);
    const size_t slotStride = (xCount + 63U) & ~size_t(63U);
    status = PrepareCtpmvBenchmarkInputs(x, xCount, slotCount, slotStride, buffers);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        buffers.Release();
        return status;
    }
    size_t slot = 0U;
    const auto launch = [&]() {
        auto* input = static_cast<aclblasComplex*>(buffers.x) + slot++ * slotStride;
        return aclblasCtpmv(handle, uplo, trans, diag, n, static_cast<const aclblasComplex*>(buffers.ap), input, incx);
    };
    for (int i = 0; i < warmup && status == ACLBLAS_STATUS_SUCCESS; ++i)
        status = launch();
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations && status == ACLBLAS_STATUS_SUCCESS; ++i) {
        status = launch();
    }
    const aclError syncRet = aclrtSynchronizeStream(stream);
    const auto end = std::chrono::steady_clock::now();
    if (status == ACLBLAS_STATUS_SUCCESS && syncRet != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = ReadCtpmvBenchmarkOutput(x, xCount, slotCount, slotStride, buffers);
    }
    buffers.Release();
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    const double elapsedUs = std::chrono::duration<double, std::micro>(end - begin).count();
    *averageUs = static_cast<float>(elapsedUs / static_cast<double>(iterations));
    return ACLBLAS_STATUS_SUCCESS;
}
