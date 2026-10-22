/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRMM_NPU_H
#define STRMM_NPU_H

#include "acl/acl.h"
#include "cann_ops_blas.h"

static inline aclError AllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr) return ACL_SUCCESS;
    aclError ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;
    ret = aclrtMemcpy(dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(dPtr); dPtr = nullptr; }
    return ret;
}

static inline void FreeDev(void* dPtr)
{
    if (dPtr) aclrtFree(dPtr);
}

// Column-major B→C copy: B and C are both ldb/ldc × n (column stride × n cols).
// When ldb==ldc the entire ldb*n block is copied in one D2D transfer.
// Otherwise each column (m elements) is copied individually.
static inline aclblasStatus_t StrmmCopyB2C(void* dC, const void* dB,
    int m, int n, int ldb, int ldc, size_t cBytes, size_t bBytes, aclrtStream stream)
{
    if (ldb == ldc) {
        if (aclrtMemcpyAsync(dC, cBytes, dB, bBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMemsetAsync(dC, cBytes, 0, cBytes, stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    size_t colBytes = static_cast<size_t>(m) * sizeof(float);
    for (int j = 0; j < n; ++j) {
        size_t offset = static_cast<size_t>(j) * static_cast<size_t>(ldc) * sizeof(float);
        if (aclrtMemcpyAsync(static_cast<float*>(dC) + static_cast<size_t>(j) * ldc,
                cBytes - offset,
                static_cast<const float*>(dB) + static_cast<size_t>(j) * ldb,
                colBytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t StrmmSyncAndCopyD2H(aclblasHandle handle,
    const void* dC, float* C, size_t cBytes)
{
    aclrtStream stream = nullptr;
    aclblasGetStream(handle, &stream);
    if (stream != nullptr) {
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    if (aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Device buffer bundle for strmm: A, B, C and optional device-side alpha.
struct StrmmDevBuffers {
    void* dA;
    void* dB;
    void* dC;
    void* dAlpha;
};

// Allocates and H2D-copies A/B, allocates C, and optionally copies alpha to
// device. On any failure all previously allocated buffers are freed and
// ACLBLAS_STATUS_ALLOC_FAILED is returned.
static inline aclblasStatus_t AllocStrmmDevBuffers(
    StrmmDevBuffers& bufs, const float* A, const float* B,
    size_t aBytes, size_t bBytes, size_t cBytes,
    bool alphaOnDevice, const float* alpha)
{
    bufs.dA = nullptr;
    bufs.dB = nullptr;
    bufs.dC = nullptr;
    bufs.dAlpha = nullptr;
    if (AllocCopyH2D(bufs.dA, A, aBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (AllocCopyH2D(bufs.dB, B, bBytes) != ACL_SUCCESS) {
        FreeDev(bufs.dA);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMalloc(&bufs.dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        FreeDev(bufs.dA);
        FreeDev(bufs.dB);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (alphaOnDevice) {
        if (AllocCopyH2D(bufs.dAlpha, alpha, sizeof(float)) != ACL_SUCCESS) {
            FreeDev(bufs.dA);
            FreeDev(bufs.dB);
            FreeDev(bufs.dC);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline void FreeStrmmDevBuffers(StrmmDevBuffers& bufs)
{
    FreeDev(bufs.dA);
    FreeDev(bufs.dB);
    FreeDev(bufs.dC);
    FreeDev(bufs.dAlpha);
}

inline aclblasStatus_t aclblasStrmm_npu(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float* C,
    int ldc,
    bool alphaOnDevice = false)
{
    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (handle == nullptr || m <= 0 || n <= 0 ||
        alpha == nullptr || A == nullptr || B == nullptr || C == nullptr ||
        (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) ||
        lda < aDim || ldb < m || ldc < m) {
        return aclblasStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    }

    // Column-major buffer sizes: A is lda×aDim, B is ldb×n, C is ldc×n.
    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(ldb) * static_cast<size_t>(n) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);

    StrmmDevBuffers bufs;
    aclblasStatus_t allocSt = AllocStrmmDevBuffers(
        bufs, A, B, aBytes, bBytes, cBytes, alphaOnDevice, alpha);
    if (allocSt != ACLBLAS_STATUS_SUCCESS) {
        return allocSt;
    }

    aclrtStream stream = nullptr;
    aclblasStatus_t getStreamRet = aclblasGetStream(handle, &stream);
    if (getStreamRet != ACLBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[WARN] strmm: aclblasGetStream failed, ret=%d, fallback to default stream\n",
                static_cast<int>(getStreamRet));
    }
    aclblasStatus_t cRet = StrmmCopyB2C(bufs.dC, bufs.dB, m, n, ldb, ldc, cBytes, bBytes, stream);
    if (cRet != ACLBLAS_STATUS_SUCCESS) {
        FreeStrmmDevBuffers(bufs);
        return cRet;
    }

    const float* alphaArg = alphaOnDevice ? static_cast<const float*>(bufs.dAlpha) : alpha;
    aclblasStatus_t ret = aclblasStrmm(
        handle, side, uplo, trans, diag, m, n,
        alphaArg,
        static_cast<const float*>(bufs.dA), lda,
        static_cast<const float*>(bufs.dB), ldb,
        static_cast<float*>(bufs.dC), ldc);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = StrmmSyncAndCopyD2H(handle, bufs.dC, C, cBytes);
    }

    FreeStrmmDevBuffers(bufs);
    return ret;
}

#endif // STRMM_NPU_H
