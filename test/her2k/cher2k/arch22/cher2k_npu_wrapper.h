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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../../../syr2/csyr2/arch22/device_matrix_snapshot.h"

struct Cher2kDeviceBuffers {
    void* alpha = nullptr;
    void* beta = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    ~Cher2kDeviceBuffers()
    {
        if (alpha)
            aclrtFree(alpha);
        if (beta)
            aclrtFree(beta);
        if (a)
            aclrtFree(a);
        if (b)
            aclrtFree(b);
        if (c)
            aclrtFree(c);
    }
};

struct Cher2kEventPair {
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    ~Cher2kEventPair()
    {
        if (start)
            aclrtDestroyEvent(start);
        if (end)
            aclrtDestroyEvent(end);
    }
};

inline aclblasStatus_t Cher2kAllocAndCopy(void*& device, const void* host, size_t bytes)
{
    if (host == nullptr || bytes == 0U)
        return ACLBLAS_STATUS_SUCCESS;
    if (aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t Cher2kPrepareDeviceBuffers(
    Cher2kDeviceBuffers& buffers, const aclblasComplex* alpha, const aclblasComplex* a, size_t aBytes,
    const aclblasComplex* b, size_t bBytes, const float* beta, aclblasComplex* c, size_t cBytes)
{
    aclblasStatus_t status = Cher2kAllocAndCopy(buffers.alpha, alpha, sizeof(aclblasComplex));
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = Cher2kAllocAndCopy(buffers.beta, beta, sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = Cher2kAllocAndCopy(buffers.a, a, aBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = Cher2kAllocAndCopy(buffers.b, b, bBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    return Cher2kAllocAndCopy(buffers.c, c, cBytes);
}

inline aclblasStatus_t Cher2kWaitForPriorWork(aclrtStream stream)
{
    aclrtEvent event = nullptr;
    if (aclrtCreateEvent(&event) != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    const aclError recordRet = aclrtRecordEvent(event, stream);
    const aclError waitRet = recordRet == ACL_SUCCESS ? aclrtSynchronizeEvent(event) : recordRet;
    const aclError destroyRet = aclrtDestroyEvent(event);
    return waitRet == ACL_SUCCESS && destroyRet == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS :
                                                                 ACLBLAS_STATUS_EXECUTION_FAILED;
}

inline bool Cher2kNeedsPassthrough(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const float* beta)
{
    return handle == nullptr || n <= 0 || k < 0 || alpha == nullptr || beta == nullptr ||
           (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) || (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_C);
}

inline aclblasStatus_t aclblasCher2k_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const float* beta, aclblasComplex* c, int ldc,
    bool verifySnapshot = false)
{
    if (Cher2kNeedsPassthrough(handle, uplo, trans, n, k, alpha, beta)) {
        return aclblasCher2k(handle, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    const int columns = trans == ACLBLAS_OP_N ? k : n;
    const size_t aBytes = columns > 0 ? static_cast<size_t>(lda) * columns * sizeof(aclblasComplex) : 0U;
    const size_t bBytes = columns > 0 ? static_cast<size_t>(ldb) * columns * sizeof(aclblasComplex) : 0U;
    const size_t cBytes = static_cast<size_t>(ldc) * n * sizeof(aclblasComplex);

    Cher2kDeviceBuffers buffers;
    aclblasStatus_t status = Cher2kPrepareDeviceBuffers(buffers, alpha, a, aBytes, b, bBytes, beta, c, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    if (verifySnapshot) {
        status = triangle_test::VerifyDeviceMatrixSnapshot(buffers.c, c, cBytes);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }

    status = aclblasCher2k(
        handle, uplo, trans, n, k, buffers.alpha ? static_cast<const aclblasComplex*>(buffers.alpha) : alpha,
        buffers.a ? static_cast<const aclblasComplex*>(buffers.a) : a, lda,
        buffers.b ? static_cast<const aclblasComplex*>(buffers.b) : b, ldb,
        buffers.beta ? static_cast<const float*>(buffers.beta) : beta,
        buffers.c ? static_cast<aclblasComplex*>(buffers.c) : c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    aclrtStream stream = nullptr;
    status = aclblasGetStream(handle, &stream);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = Cher2kWaitForPriorWork(stream);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if (c != nullptr && buffers.c != nullptr &&
        aclrtMemcpy(c, cBytes, buffers.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Measures one end-to-end batch: all API dispatch and all queued device work
// are inside the wall-clock interval. A single final synchronization drains
// the stream without adding one test-harness synchronization per sample.
// Allocation, copies, warmup, golden generation, and validation stay outside.
template <typename LaunchFn>
inline aclblasStatus_t Cher2kMeasureBatches(
    LaunchFn& launch, aclrtStream stream, int warmup, int iterations, float* averageUs)
{
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    for (int i = 0; i < warmup; ++i) {
        status = launch();
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
    }
    Cher2kEventPair events;
    if (aclrtCreateEvent(&events.start) != ACL_SUCCESS || aclrtCreateEvent(&events.end) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtRecordEvent(events.start, stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    for (int i = 0; i < iterations; ++i) {
        status = launch();
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
    }
    if (aclrtRecordEvent(events.end, stream) != ACL_SUCCESS || aclrtSynchronizeEvent(events.end) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    float elapsedMs = 0.0f;
    if (aclrtEventElapsedTime(&elapsedMs, events.start, events.end) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    *averageUs = elapsedMs * 1000.0f / static_cast<float>(iterations);
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCher2k_npu_benchmark(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const float* beta,
    aclblasComplex* c, int ldc, int warmup, int iterations, float* averageUs)
{
    if (averageUs == nullptr || warmup < 0 || iterations <= 0 ||
        Cher2kNeedsPassthrough(handle, uplo, trans, n, k, alpha, beta)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int columns = trans == ACLBLAS_OP_N ? k : n;
    const size_t aBytes = columns > 0 ? static_cast<size_t>(lda) * columns * sizeof(aclblasComplex) : 0U;
    const size_t bBytes = columns > 0 ? static_cast<size_t>(ldb) * columns * sizeof(aclblasComplex) : 0U;
    const size_t cBytes = static_cast<size_t>(ldc) * n * sizeof(aclblasComplex);
    Cher2kDeviceBuffers buffers;
    aclblasStatus_t status = Cher2kPrepareDeviceBuffers(buffers, alpha, a, aBytes, b, bBytes, beta, c, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    const auto launch = [&]() {
        return aclblasCher2k(
            handle, uplo, trans, n, k, static_cast<const aclblasComplex*>(buffers.alpha),
            static_cast<const aclblasComplex*>(buffers.a), lda, static_cast<const aclblasComplex*>(buffers.b), ldb,
            static_cast<const float*>(buffers.beta), static_cast<aclblasComplex*>(buffers.c), ldc);
    };
    return Cher2kMeasureBatches(launch, stream, warmup, iterations, averageUs);
}
