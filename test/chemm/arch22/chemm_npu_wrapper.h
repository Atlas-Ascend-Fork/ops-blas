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

#include <cstdint>
#include <mutex>
#include <unordered_set>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../../../blas/common/helper/aclblas_handle_internal.h"
#include "../../../blas/hemm/arch22/chemm_tiling_data.h"

inline std::mutex& ChemmTimeoutMutex()
{
    static std::mutex mutex;
    return mutex;
}

inline std::unordered_set<aclblasHandle_t>& ChemmTimedOutHandles()
{
    static std::unordered_set<aclblasHandle_t> timedOutHandles;
    return timedOutHandles;
}

inline bool ChemmIsHandleTimedOut(aclblasHandle_t handle)
{
    std::lock_guard<std::mutex> lock(ChemmTimeoutMutex());
    return ChemmTimedOutHandles().count(handle) != 0U;
}

inline void ChemmMarkHandleTimedOut(aclblasHandle_t handle)
{
    std::lock_guard<std::mutex> lock(ChemmTimeoutMutex());
    (void)ChemmTimedOutHandles().insert(handle);
}

struct ChemmDeviceBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
};

inline void ChemmReleaseDeviceBuffers(ChemmDeviceBuffers& buffers)
{
    (void)aclrtFree(buffers.a);
    (void)aclrtFree(buffers.b);
    (void)aclrtFree(buffers.c);
    buffers = {};
}

inline aclblasStatus_t ChemmAllocAndCopy(void** device, size_t bytes, const void* host)
{
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        (void)aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ChemmPrepareDeviceBuffers(
    ChemmDeviceBuffers& buffers, const aclblasComplex* a, size_t aBytes, const aclblasComplex* b, size_t bBytes,
    const aclblasComplex* c, size_t cBytes)
{
    aclblasStatus_t status = ChemmAllocAndCopy(&buffers.a, aBytes, a);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = ChemmAllocAndCopy(&buffers.b, bBytes, b);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = ChemmAllocAndCopy(&buffers.c, cBytes, c);
    }
    if (status != ACLBLAS_STATUS_SUCCESS) {
        ChemmReleaseDeviceBuffers(buffers);
    }
    return status;
}

inline aclblasStatus_t ChemmRestoreTimeout(uint32_t savedTimeout, bool& changed)
{
    if (!changed) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtSetOpExecuteTimeOutWithMs(savedTimeout) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    changed = false;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ChemmRunDeviceOperation(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc, size_t aBytes, size_t bBytes, size_t cBytes)
{
    ChemmDeviceBuffers buffers;
    aclblasStatus_t status = ChemmPrepareDeviceBuffers(buffers, A, aBytes, B, bBytes, C, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Keep the handle in Library workspace mode.  aclblasChemm derives the
    // exact workspace from its current tiling and can grow it as required.
    uint32_t savedTimeout = 0U;
    bool timeoutChanged = false;
    if (aclrtGetOpExecuteTimeout(&savedTimeout) != ACL_SUCCESS ||
        aclrtSetOpExecuteTimeOutWithMs(5000U) != ACL_SUCCESS) {
        ChemmReleaseDeviceBuffers(buffers);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    timeoutChanged = true;
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS) {
        (void)ChemmRestoreTimeout(savedTimeout, timeoutChanged);
        ChemmReleaseDeviceBuffers(buffers);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclblasStatus_t ret = aclblasChemm(
        handle, side, uplo, m, n, alpha, static_cast<const aclblasComplex*>(buffers.a), lda,
        static_cast<const aclblasComplex*>(buffers.b), ldb, beta, static_cast<aclblasComplex*>(buffers.c), ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeStreamWithTimeout(stream, 10000) == ACL_SUCCESS) {
        if (aclrtMemcpy(C, cBytes, buffers.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else if (ret == ACLBLAS_STATUS_SUCCESS) {
        ChemmMarkHandleTimedOut(handle);
        ret = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    aclblasStatus_t restore = ChemmRestoreTimeout(savedTimeout, timeoutChanged);
    ChemmReleaseDeviceBuffers(buffers);
    if (restore != ACLBLAS_STATUS_SUCCESS) {
        return restore;
    }
    return ret;
}

inline aclblasStatus_t aclblasChemm_npu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    if (handle != nullptr && ChemmIsHandleTimedOut(handle)) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (handle == nullptr || m <= 0 || n <= 0 || A == nullptr || B == nullptr || C == nullptr) {
        return aclblasChemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t aElemCount = static_cast<size_t>(aDim) * static_cast<size_t>(lda);
    const size_t bElemCount = static_cast<size_t>(n) * static_cast<size_t>(ldb);
    const size_t cElemCount = static_cast<size_t>(n) * static_cast<size_t>(ldc);
    const size_t aBytes = aElemCount * 2U * sizeof(float);
    const size_t bBytes = bElemCount * 2U * sizeof(float);
    const size_t cBytes = cElemCount * 2U * sizeof(float);

    return ChemmRunDeviceOperation(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, aBytes, bBytes, cBytes);
}

template <typename Launch>
inline aclblasStatus_t ChemmWarmupLaunches(Launch& launch, aclrtStream stream, int warmup)
{
    for (int i = 0; i < warmup; ++i) {
        const aclblasStatus_t status = launch();
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return aclrtSynchronizeStream(stream) == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
}

inline void ChemmDestroyEvents(aclrtEvent start, aclrtEvent end)
{
    if (start != nullptr) {
        (void)aclrtDestroyEvent(start);
    }
    if (end != nullptr) {
        (void)aclrtDestroyEvent(end);
    }
}

template <typename Launch>
inline aclblasStatus_t ChemmTimedLaunches(Launch& launch, aclrtStream stream, int iterations, float& elapsedMs)
{
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    if (aclrtCreateEvent(&start) != ACL_SUCCESS || aclrtCreateEvent(&end) != ACL_SUCCESS) {
        ChemmDestroyEvents(start, end);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclblasStatus_t status =
        aclrtRecordEvent(start, stream) == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
    for (int i = 0; i < iterations && status == ACLBLAS_STATUS_SUCCESS; ++i) {
        status = launch();
    }
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtRecordEvent(end, stream) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeEvent(end) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtEventElapsedTime(&elapsedMs, start, end) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ChemmDestroyEvents(start, end);
    return status;
}

inline aclblasStatus_t aclblasChemm_perf(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc, int warmup, int iterations, float& averageUs)
{
    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(n) * static_cast<size_t>(ldb) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(n) * static_cast<size_t>(ldc) * sizeof(aclblasComplex);
    ChemmDeviceBuffers buffers;
    aclblasStatus_t status = ChemmPrepareDeviceBuffers(buffers, A, aBytes, B, bBytes, C, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Performance launches must use the same tiling-owned Library workspace.
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS) {
        ChemmReleaseDeviceBuffers(buffers);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    auto launch = [&]() {
        return aclblasChemm(
            handle, side, uplo, m, n, alpha, static_cast<const aclblasComplex*>(buffers.a), lda,
            static_cast<const aclblasComplex*>(buffers.b), ldb, beta, static_cast<aclblasComplex*>(buffers.c), ldc);
    };
    float elapsedMs = 0.0f;
    status = ChemmWarmupLaunches(launch, stream, warmup);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = ChemmTimedLaunches(launch, stream, iterations, elapsedMs);
    }
    ChemmReleaseDeviceBuffers(buffers);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        averageUs = elapsedMs * 1000.0f / static_cast<float>(iterations);
    }
    return status;
}
