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
#include <chrono>
#include <cstdlib>
#include <iostream>
#include "acl/acl.h"
#include "cann_ops_blas.h"

struct CtrsvDeviceBuffers {
    void* a = nullptr;
    void* x = nullptr;

    ~CtrsvDeviceBuffers()
    {
        if (a != nullptr)
            aclrtFree(a);
        if (x != nullptr)
            aclrtFree(x);
    }
};

inline aclblasStatus_t CtrsvAllocAndCopy(void*& device, const void* host, size_t bytes)
{
    if (host == nullptr || bytes == 0U)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCtrsv_npu(
    aclblasHandle_t h, aclblasFillMode_t u, aclblasOperation_t t, aclblasDiagType_t d, int64_t n,
    const aclblasComplex* A, int64_t lda, aclblasComplex* x, int64_t incx)
{
    if (h == nullptr || n <= 0 || A == nullptr || x == nullptr || incx == 0 || lda <= 0)
        return aclblasCtrsv(h, u, t, d, (int)n, A, (int)lda, x, (int)incx);
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t xCount = static_cast<size_t>(n - 1) * static_cast<size_t>(std::abs(incx)) + 1U;
    const size_t xBytes = xCount * sizeof(aclblasComplex);
    void* dA = nullptr;
    void* dX = nullptr;
    if (aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    if (aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        aclrtFree(dA);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclError e = aclrtMemcpy(dA, aBytes, A, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (e == ACL_SUCCESS)
        e = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (e != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclblasStatus_t s =
        aclblasCtrsv(h, u, t, d, (int)n, (const aclblasComplex*)dA, (int)lda, (aclblasComplex*)dX, (int)incx);
    if (s == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeDevice() != ACL_SUCCESS) {
        std::cerr << "[Ctrsv] device synchronize failed: " << aclGetRecentErrMsg() << std::endl;
        s = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    e = aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(dA);
    aclrtFree(dX);
    return e == ACL_SUCCESS ? s : ACLBLAS_STATUS_INTERNAL_ERROR;
}

// Time consecutive public API calls and one final stream synchronization.
// Allocation and the single H2D preparation stay outside the timed interval.
inline aclblasStatus_t aclblasCtrsv_npu_benchmark(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, int warmup,
    int iterations, double* averageUs)
{
    if (handle == nullptr || stream == nullptr || a == nullptr || x == nullptr || n <= 0 || lda < n || incx == 0 ||
        warmup < 0 || iterations <= 0 || averageUs == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t xCount = static_cast<size_t>(n - 1) * static_cast<size_t>(std::abs(incx)) + 1U;
    const size_t xBytes = xCount * sizeof(aclblasComplex);
    CtrsvDeviceBuffers buffers;
    aclblasStatus_t status = CtrsvAllocAndCopy(buffers.a, a, aBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = CtrsvAllocAndCopy(buffers.x, x, xBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    const auto invoke = [&]() {
        return aclblasCtrsv(
            handle, uplo, trans, diag, n, static_cast<const aclblasComplex*>(buffers.a), lda,
            static_cast<aclblasComplex*>(buffers.x), incx);
    };
    // Warm up without introducing a synchronization point between launches.
    // One fence before measurement ensures all warmup work is complete.
    for (int i = 0; i < warmup; ++i) {
        status = invoke();
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "[Ctrsv] warmup synchronize failed: " << aclGetRecentErrMsg() << std::endl;
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        status = invoke();
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        std::cerr << "[Ctrsv] measured synchronize failed: " << aclGetRecentErrMsg() << std::endl;
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsedUs = std::chrono::duration<double, std::micro>(end - begin).count();
    *averageUs = elapsedUs / static_cast<double>(iterations);
    return ACLBLAS_STATUS_SUCCESS;
}
