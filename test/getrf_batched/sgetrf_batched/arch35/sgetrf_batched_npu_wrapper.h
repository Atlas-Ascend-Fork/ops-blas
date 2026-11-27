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
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct _aclblas_handle {
    aclrtStream stream;
    void* workspace;
    size_t workspaceSize;
};

/**
 * NPU wrapper for aclblasSgetrfBatched.
 *
 * The API expects device-side pointer arrays and device memory.
 * This wrapper handles:
 *   1. Allocate device memory for each matrix and copy H2D
 *   2. Create device pointer array
 *   3. Allocate device memory for PivotArray and infoArray
 *   4. Call aclblasSgetrfBatched
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
    GetrfDeviceBuffers& bufs, float* const Aarray[], int batchSize, size_t matBytes)
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
    std::vector<float*> hPtrArray(batchSize);
    for (int b = 0; b < batchSize; b++) {
        hPtrArray[b] = static_cast<float*>(dMatrices[b]);
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

static bool CopyResultsD2H(
    float* const Aarray[], int* PivotArray, int* infoArray, const GetrfDeviceBuffers& bufs, int batchSize,
    size_t matBytes, size_t pivotBytes, size_t infoBytes)
{
    for (int b = 0; b < batchSize; b++) {
        if (Aarray[b] != nullptr &&
            aclrtMemcpy(Aarray[b], matBytes, bufs.dMatrices[b], matBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return false;
        }
    }
    if (PivotArray != nullptr && bufs.dPivot != nullptr &&
        aclrtMemcpy(PivotArray, pivotBytes, bufs.dPivot, pivotBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    return aclrtMemcpy(infoArray, infoBytes, bufs.dInfo, infoBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

inline aclblasStatus_t aclblasSgetrfBatched_npu(
    aclblasHandle_t handle, int n, float* const Aarray[], int lda, int* PivotArray, int* infoArray, int batchSize)
{
    if (handle == nullptr || n <= 0 || batchSize <= 0 || Aarray == nullptr || infoArray == nullptr) {
        return aclblasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
    }

    const size_t matBytes = static_cast<size_t>(lda) * n * sizeof(float);
    const size_t pivotBytes = static_cast<size_t>(n) * batchSize * sizeof(int);
    const size_t infoBytes = static_cast<size_t>(batchSize) * sizeof(int);
    const size_t ptrArrayBytes = static_cast<size_t>(batchSize) * sizeof(float*);

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

    aclblasStatus_t ret = aclblasSgetrfBatched(
        handle, n, reinterpret_cast<float* const*>(bufs.dPtrArray), lda, static_cast<int*>(bufs.dPivot),
        static_cast<int*>(bufs.dInfo), batchSize);

    auto* h = handle;
    if (aclrtSynchronizeStream(h->stream) != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (ret == ACLBLAS_STATUS_SUCCESS &&
               !CopyResultsD2H(Aarray, PivotArray, infoArray, bufs, batchSize, matBytes, pivotBytes, infoBytes)) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    bufs.Cleanup();
    return ret;
}

