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
#include <securec.h>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct _aclblas_handle {
    aclrtStream stream;
    void* workspace;
    size_t workspaceSize;
};

/**
 * NPU wrapper for aclblasCgetrfBatched.
 *
 * The API expects device-side pointer arrays and device memory.
 * This wrapper handles:
 *   1. Allocate device memory for each matrix and copy H2D
 *   2. Create device pointer array
 *   3. Allocate device memory for PivotArray and infoArray
 *   4. Call aclblasCgetrfBatched
 *   5. Sync and copy results D2H
 *   6. Free all device memory
 *
 * If handle == nullptr or n <= 0 or batchSize <= 0, pass through directly.
 * If Aarray == nullptr or infoArray == nullptr, pass through for error-path testing.
 */

struct GetrfDeviceBuffers {
    std::vector<void*> dMatrices;
    void* dPtrArray = nullptr;
    void* dPivot = nullptr;
    void* dInfo = nullptr;

    void Cleanup()
    {
        for (auto& dm : dMatrices) {
            if (dm)
                aclrtFree(dm);
        }
        if (dPtrArray)
            aclrtFree(dPtrArray);
        if (dPivot)
            aclrtFree(dPivot);
        if (dInfo)
            aclrtFree(dInfo);
    }
};

static aclblasStatus_t AllocateAndCopyMatrices(
    GetrfDeviceBuffers& bufs, aclblasComplex* const Aarray[], int batchSize, size_t matBytes)
{
    bufs.dMatrices.resize(batchSize, nullptr);
    for (int b = 0; b < batchSize; b++) {
        aclError aclRet = aclrtMalloc(&bufs.dMatrices[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (Aarray[b] != nullptr) {
            aclRet = aclrtMemcpy(bufs.dMatrices[b], matBytes, Aarray[b], matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (aclRet != ACL_SUCCESS) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t CreateDevicePtrArray(
    GetrfDeviceBuffers& bufs, const std::vector<void*>& dMatrices, int batchSize, size_t ptrArrayBytes)
{
    std::vector<aclblasComplex*> hPtrArray(batchSize);
    for (int b = 0; b < batchSize; b++) {
        hPtrArray[b] = static_cast<aclblasComplex*>(dMatrices[b]);
    }

    aclError aclRet = aclrtMalloc(&bufs.dPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(bufs.dPtrArray, ptrArrayBytes, hPtrArray.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t AllocatePivotAndInfo(
    GetrfDeviceBuffers& bufs, int* PivotArray, size_t pivotBytes, size_t infoBytes)
{
    if (PivotArray != nullptr) {
        aclError aclRet = aclrtMalloc(&bufs.dPivot, pivotBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        std::vector<int> zeroPivot(pivotBytes / sizeof(int), 0);
        aclRet = aclrtMemcpy(bufs.dPivot, pivotBytes, zeroPivot.data(), pivotBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    aclError aclRet = aclrtMalloc(&bufs.dInfo, infoBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static void CopyResultsD2H(
    aclblasComplex* const Aarray[], int* PivotArray, int* infoArray, const GetrfDeviceBuffers& bufs, int batchSize,
    size_t matBytes, size_t pivotBytes, size_t infoBytes)
{
    for (int b = 0; b < batchSize; b++) {
        if (Aarray[b] != nullptr) {
            aclrtMemcpy(Aarray[b], matBytes, bufs.dMatrices[b], matBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    if (PivotArray != nullptr && bufs.dPivot != nullptr) {
        aclrtMemcpy(PivotArray, pivotBytes, bufs.dPivot, pivotBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    aclrtMemcpy(infoArray, infoBytes, bufs.dInfo, infoBytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

inline aclblasStatus_t aclblasCgetrfBatched_npu(
    aclblasHandle_t handle, int n, aclblasComplex* const Aarray[], int lda, int* PivotArray, int* infoArray,
    int batchSize)
{
    if (handle == nullptr || n <= 0 || batchSize <= 0 || Aarray == nullptr || infoArray == nullptr) {
        return aclblasCgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
    }

    const size_t matBytes = static_cast<size_t>(lda) * n * sizeof(aclblasComplex);
    const size_t pivotBytes = static_cast<size_t>(n) * batchSize * sizeof(int);
    const size_t infoBytes = static_cast<size_t>(batchSize) * sizeof(int);
    const size_t ptrArrayBytes = static_cast<size_t>(batchSize) * sizeof(aclblasComplex*);

    GetrfDeviceBuffers bufs;

    aclblasStatus_t status = AllocateAndCopyMatrices(bufs, Aarray, batchSize, matBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    status = CreateDevicePtrArray(bufs, bufs.dMatrices, batchSize, ptrArrayBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    status = AllocatePivotAndInfo(bufs, PivotArray, pivotBytes, infoBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    aclblasStatus_t ret = aclblasCgetrfBatched(
        handle, n, reinterpret_cast<aclblasComplex* const*>(bufs.dPtrArray), lda, static_cast<int*>(bufs.dPivot),
        static_cast<int*>(bufs.dInfo), batchSize);

    aclrtSynchronizeStream(handle->stream);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        CopyResultsD2H(Aarray, PivotArray, infoArray, bufs, batchSize, matBytes, pivotBytes, infoBytes);
    }

    bufs.Cleanup();
    return ret;
}

struct CgetrfBenchmarkBuffers {
    void* dMatrices = nullptr;
    void* dPtrArray = nullptr;
    void* dPivot = nullptr;
    void* dInfo = nullptr;

    void Cleanup()
    {
        if (dMatrices)
            aclrtFree(dMatrices);
        if (dPtrArray)
            aclrtFree(dPtrArray);
        if (dPivot)
            aclrtFree(dPivot);
        if (dInfo)
            aclrtFree(dInfo);
    }
};

inline aclblasStatus_t AllocateBenchmarkBuffers(
    CgetrfBenchmarkBuffers& bufs, size_t matrixElements, int n, int batchSize, bool usePivot)
{
    const size_t matrixBytes = matrixElements * batchSize * sizeof(aclblasComplex);
    const size_t pointerBytes = static_cast<size_t>(batchSize) * sizeof(aclblasComplex*);
    const size_t pivotBytes = static_cast<size_t>(n) * batchSize * sizeof(int);
    const size_t infoBytes = static_cast<size_t>(batchSize) * sizeof(int);
    if (aclrtMalloc(&bufs.dMatrices, matrixBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&bufs.dPtrArray, pointerBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&bufs.dInfo, infoBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        (usePivot && aclrtMalloc(&bufs.dPivot, pivotBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    std::vector<aclblasComplex*> hostPointers(batchSize);
    auto* matrixBase = static_cast<aclblasComplex*>(bufs.dMatrices);
    for (int b = 0; b < batchSize; b++) {
        hostPointers[b] = matrixBase + static_cast<size_t>(b) * matrixElements;
    }
    if (aclrtMemcpy(bufs.dPtrArray, pointerBytes, hostPointers.data(), pointerBytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

template <class Restore, class Invoke>
aclblasStatus_t WarmupCgetrfBenchmark(Restore& restoreInput, Invoke& invoke, aclrtStream stream, int warmupCount)
{
    for (int i = 0; i < warmupCount; i++) {
        if (restoreInput() != ACL_SUCCESS || invoke() != ACLBLAS_STATUS_SUCCESS ||
            aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        std::cerr << "CGETRF_WARMUP_PROGRESS," << i + 1 << ',' << warmupCount << std::endl;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

template <class Restore, class Invoke>
aclblasStatus_t SampleCgetrfBenchmark(
    Restore& restoreInput, Invoke& invoke, aclrtStream stream, aclrtEvent start, aclrtEvent end, int sampleCount,
    std::vector<double>& sampleUs)
{
    sampleUs.clear();
    sampleUs.reserve(sampleCount);
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    for (int i = 0; i < sampleCount; i++) {
        float elapsedMs = 0.0f;
        if (restoreInput() != ACL_SUCCESS || aclrtRecordEvent(start, stream) != ACL_SUCCESS ||
            invoke() != ACLBLAS_STATUS_SUCCESS || aclrtRecordEvent(end, stream) != ACL_SUCCESS ||
            aclrtSynchronizeEvent(end) != ACL_SUCCESS || aclrtEventElapsedTime(&elapsedMs, start, end) != ACL_SUCCESS) {
            status = ACLBLAS_STATUS_EXECUTION_FAILED;
            break;
        }
        sampleUs.push_back(static_cast<double>(elapsedMs) * 1000.0);
        std::cerr << "CGETRF_SAMPLE_PROGRESS," << i + 1 << ',' << sampleCount << ','
                  << static_cast<double>(elapsedMs) * 1000.0 << std::endl;
    }

    return status;
}

inline aclblasStatus_t BenchmarkCgetrfBatched(
    aclblasHandle_t handle, int n, aclblasComplex* const Aarray[], int lda, bool usePivot, int batchSize,
    int warmupCount, int sampleCount, std::vector<double>& sampleUs)
{
    const size_t matrixElements = static_cast<size_t>(lda) * n;
    const size_t totalElements = matrixElements * batchSize;
    const size_t matrixBytes = totalElements * sizeof(aclblasComplex);

    std::vector<aclblasComplex> hostMatrices(totalElements);
    for (int b = 0; b < batchSize; b++) {
        const size_t offset = static_cast<size_t>(b) * matrixElements;
        const auto copyStatus = memcpy_s(
            hostMatrices.data() + offset, (totalElements - offset) * sizeof(aclblasComplex), Aarray[b],
            matrixElements * sizeof(aclblasComplex));
        if (copyStatus != EOK) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    CgetrfBenchmarkBuffers bufs;
    auto allocation = AllocateBenchmarkBuffers(bufs, matrixElements, n, batchSize, usePivot);
    if (allocation != ACLBLAS_STATUS_SUCCESS)
        return allocation;

    auto invoke = [&]() {
        return aclblasCgetrfBatched(
            handle, n, reinterpret_cast<aclblasComplex* const*>(bufs.dPtrArray), lda,
            usePivot ? static_cast<int*>(bufs.dPivot) : nullptr, static_cast<int*>(bufs.dInfo), batchSize);
    };
    auto restoreInput = [&]() {
        return aclrtMemcpy(bufs.dMatrices, matrixBytes, hostMatrices.data(), matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    };
    if (WarmupCgetrfBenchmark(restoreInput, invoke, handle->stream, warmupCount) != ACLBLAS_STATUS_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    if (aclrtCreateEvent(&start) != ACL_SUCCESS || aclrtCreateEvent(&end) != ACL_SUCCESS) {
        if (start)
            aclrtDestroyEvent(start);
        if (end)
            aclrtDestroyEvent(end);
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const auto status = SampleCgetrfBenchmark(restoreInput, invoke, handle->stream, start, end, sampleCount, sampleUs);

    aclrtDestroyEvent(start);
    aclrtDestroyEvent(end);
    bufs.Cleanup();
    return status;
}
