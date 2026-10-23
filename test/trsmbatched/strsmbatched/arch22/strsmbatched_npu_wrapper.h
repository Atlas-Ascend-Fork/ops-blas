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
#include <cstring>
#include <vector>

#include <securec.h>
#include "acl/acl.h"
#include "cann_ops_blas.h"

struct TrsmbatchedDeviceBuffers {
    std::vector<void*> dA;
    std::vector<void*> dB;
    void* dASlab = nullptr;
    void* dBSlab = nullptr;
    size_t bStride = 0;
    void* dAPtrArray = nullptr;
    void* dBPtrArray = nullptr;

    void Cleanup()
    {
        if (dASlab != nullptr) {
            aclrtFree(dASlab);
        }
        if (dBSlab != nullptr) {
            aclrtFree(dBSlab);
        }
        if (dAPtrArray != nullptr) {
            aclrtFree(dAPtrArray);
        }
        if (dBPtrArray != nullptr) {
            aclrtFree(dBPtrArray);
        }
        dA.clear();
        dB.clear();
        dASlab = nullptr;
        dBSlab = nullptr;
        bStride = 0;
        dAPtrArray = nullptr;
        dBPtrArray = nullptr;
    }
};

inline size_t TrsmbatchedAlignedBatchStride(size_t elemBytes)
{
    const size_t minBytes = std::max<size_t>(elemBytes, sizeof(float));
    size_t aligned = (minBytes + 255U) & ~static_cast<size_t>(255U);
    if (elemBytes > 4096U && (aligned & 4095U) == 0U) {
        aligned += 256U;
    }
    return aligned;
}

inline aclblasStatus_t TrsmbatchedAllocAndCopyA(
    TrsmbatchedDeviceBuffers& bufs, const float* const A[], int batchCount, size_t aBytes)
{
    bufs.dA.assign(batchCount, nullptr);
    const size_t stride = TrsmbatchedAlignedBatchStride(aBytes);
    const size_t totalBytes = static_cast<size_t>(batchCount) * stride;
    aclError aclRet = aclrtMalloc(&bufs.dASlab, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    std::vector<uint8_t> stage;
    if (aBytes > 0) {
        stage.assign(totalBytes, 0U);
    }
    uint8_t* basePtr = static_cast<uint8_t*>(bufs.dASlab);
    for (int b = 0; b < batchCount; b++) {
        if (A[b] == nullptr) {
            bufs.dA[b] = nullptr;
            continue;
        }
        const size_t off = static_cast<size_t>(b) * stride;
        bufs.dA[b] = basePtr + off;
        if (aBytes > 0) {
            const errno_t err = memcpy_s(stage.data() + off, totalBytes - off, A[b], aBytes);
            if (err != EOK) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    if (aBytes > 0) {
        aclRet = aclrtMemcpy(bufs.dASlab, totalBytes, stage.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t TrsmbatchedAllocAndCopyB(
    TrsmbatchedDeviceBuffers& bufs, float* const B[], int batchCount, size_t bBytes)
{
    bufs.dB.assign(batchCount, nullptr);
    const size_t stride = TrsmbatchedAlignedBatchStride(bBytes);
    bufs.bStride = stride;
    const size_t totalBytes = static_cast<size_t>(batchCount) * stride;
    aclError aclRet = aclrtMalloc(&bufs.dBSlab, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    std::vector<uint8_t> stage;
    if (bBytes > 0) {
        stage.assign(totalBytes, 0U);
    }
    uint8_t* basePtr = static_cast<uint8_t*>(bufs.dBSlab);
    for (int b = 0; b < batchCount; b++) {
        if (B[b] == nullptr) {
            bufs.dB[b] = nullptr;
            continue;
        }
        const size_t off = static_cast<size_t>(b) * stride;
        bufs.dB[b] = basePtr + off;
        if (bBytes > 0) {
            const errno_t err = memcpy_s(stage.data() + off, totalBytes - off, B[b], bBytes);
            if (err != EOK) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    if (bBytes > 0) {
        aclRet = aclrtMemcpy(bufs.dBSlab, totalBytes, stage.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t TrsmbatchedCreatePtrArrays(
    TrsmbatchedDeviceBuffers& bufs, int batchCount, bool hasA, size_t ptrArrayBytes)
{
    if (hasA) {
        std::vector<float*> hAPtrArray(batchCount);
        for (int b = 0; b < batchCount; b++) {
            hAPtrArray[b] = static_cast<float*>(bufs.dA[b]);
        }
        aclError aclRet = aclrtMalloc(&bufs.dAPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet =
            aclrtMemcpy(bufs.dAPtrArray, ptrArrayBytes, hAPtrArray.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    std::vector<float*> hBPtrArray(batchCount);
    for (int b = 0; b < batchCount; b++) {
        hBPtrArray[b] = static_cast<float*>(bufs.dB[b]);
    }
    aclError aclRet = aclrtMalloc(&bufs.dBPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(bufs.dBPtrArray, ptrArrayBytes, hBPtrArray.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t TrsmbatchedCopyResultsD2H(
    TrsmbatchedDeviceBuffers& bufs, float* const B[], int batchCount, size_t bBytes)
{
    if (bBytes == 0 || bufs.dBSlab == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const size_t totalBytes = static_cast<size_t>(batchCount) * bufs.bStride;
    std::vector<uint8_t> stage(totalBytes);
    aclError d2hRet = aclrtMemcpy(stage.data(), totalBytes, bufs.dBSlab, totalBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (d2hRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    for (int b = 0; b < batchCount; b++) {
        if (B[b] != nullptr && bufs.dB[b] != nullptr) {
            const errno_t err = memcpy_s(B[b], bBytes, stage.data() + static_cast<size_t>(b) * bufs.bStride, bBytes);
            if (err != EOK) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasStrsmBatched_npu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[],
    int ldb, int batchCount)
{
    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (handle == nullptr || m <= 0 || n <= 0 || batchCount <= 0 || alpha == nullptr || B == nullptr ||
        lda < std::max(1, aDim) || ldb < std::max(1, m)) {
        return aclblasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
    }

    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(aDim) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(ldb) * static_cast<size_t>(n) * sizeof(float);
    const size_t ptrArrayBytes = static_cast<size_t>(batchCount) * sizeof(float*);

    TrsmbatchedDeviceBuffers bufs;
    const bool hasA = (A != nullptr);

    if (hasA) {
        aclblasStatus_t st = TrsmbatchedAllocAndCopyA(bufs, A, batchCount, aBytes);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }

    aclblasStatus_t st = TrsmbatchedAllocAndCopyB(bufs, B, batchCount, bBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    st = TrsmbatchedCreatePtrArrays(bufs, batchCount, hasA, ptrArrayBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto dAPtrs = hasA ? static_cast<const float* const*>(bufs.dAPtrArray) : nullptr;
    auto dBPtrs = static_cast<float* const*>(bufs.dBPtrArray);

    aclblasStatus_t ret =
        aclblasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, dAPtrs, lda, dBPtrs, ldb, batchCount);
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclrtStream stream = nullptr;
        aclblasGetStream(handle, &stream);
        aclError syncRet = aclrtSynchronizeStream(stream);
        if (syncRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        st = TrsmbatchedCopyResultsD2H(bufs, B, batchCount, bBytes);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }

    bufs.Cleanup();
    return ret;
}
