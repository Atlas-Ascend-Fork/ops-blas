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

static inline aclError CtrmmAllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr) {
        return ACL_SUCCESS;
    }
    aclError ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(dPtr);
        dPtr = nullptr;
    }
    return ret;
}

static inline void CtrmmFreeDev(void* dPtr)
{
    if (dPtr) {
        aclrtFree(dPtr);
    }
}

// Column-major B -> C pre-copy (complex64): C and B are ldb/ldc x n. When the
// strides coincide the whole block moves in one D2D transfer; otherwise the m
// active rows of each column are copied individually and the padding rows of
// C are zeroed (the NPU kernel never writes rows m..ldc-1, mirroring the
// golden's per-column copy).
static inline aclblasStatus_t CtrmmCopyB2C(
    void* dC, const void* dB, int m, int n, int ldb, int ldc, size_t cBytes, aclrtStream stream)
{
    if (ldb == ldc) {
        if (aclrtMemcpyAsync(dC, cBytes, dB, cBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMemsetAsync(dC, cBytes, 0, cBytes, stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    size_t colBytes = static_cast<size_t>(m) * sizeof(aclblasComplex);
    for (int j = 0; j < n; ++j) {
        size_t offset = static_cast<size_t>(j) * static_cast<size_t>(ldc) * sizeof(aclblasComplex);
        void* dst = reinterpret_cast<uint8_t*>(dC) + offset;
        const void* src = reinterpret_cast<const uint8_t*>(dB) +
                          static_cast<size_t>(j) * static_cast<size_t>(ldb) * sizeof(aclblasComplex);
        if (aclrtMemcpyAsync(dst, cBytes - offset, src, colBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream) != ACL_SUCCESS) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t CtrmmSyncAndCopyD2H(
    aclblasHandle handle, const void* dC, aclblasComplex* C, size_t cBytes)
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

// Device buffer bundle: A, B, C and optional device-side alpha.
struct CtrmmDevBuffers {
    void* dA;
    void* dB;
    void* dC;
    void* dAlpha;
};

static inline void FreeCtrmmDevBuffers(CtrmmDevBuffers& bufs)
{
    CtrmmFreeDev(bufs.dA);
    CtrmmFreeDev(bufs.dB);
    if (bufs.dC != bufs.dB) { // alias mode shares the B buffer as C
        CtrmmFreeDev(bufs.dC);
    }
    CtrmmFreeDev(bufs.dAlpha);
}

static inline aclblasStatus_t AllocCtrmmDevBuffers(
    CtrmmDevBuffers& bufs, const aclblasComplex* A, const aclblasComplex* B, size_t aBytes, size_t bBytes,
    size_t cBytes, bool alphaOnDevice, const aclblasComplex* alpha)
{
    // All-null init up front: on any failure the single null-safe
    // FreeCtrmmDevBuffers unwind covers the partially-allocated set.
    bufs.dA = nullptr;
    bufs.dB = nullptr;
    bufs.dC = nullptr;
    bufs.dAlpha = nullptr;
    if (CtrmmAllocCopyH2D(bufs.dA, A, aBytes) != ACL_SUCCESS || CtrmmAllocCopyH2D(bufs.dB, B, bBytes) != ACL_SUCCESS ||
        aclrtMalloc(&bufs.dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        FreeCtrmmDevBuffers(bufs);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (alphaOnDevice && CtrmmAllocCopyH2D(bufs.dAlpha, alpha, sizeof(aclblasComplex)) != ACL_SUCCESS) {
        FreeCtrmmDevBuffers(bufs);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Prepare C from the uploaded B for a driver call: in alias mode C shares
// B's buffer (exercising the operator's in-place path); otherwise B is
// pre-copied into C. Extracted verbatim from the driver body.
static inline aclblasStatus_t PrepareCtrmmNpuBuffers(
    CtrmmDevBuffers& bufs, int m, int n, int ldb, int ldc, size_t bBytes, size_t cBytes, bool aliasBC,
    aclrtStream stream)
{
    if (aliasBC && cBytes <= bBytes) {
        if (bufs.dC != nullptr && bufs.dC != bufs.dB) {
            CtrmmFreeDev(bufs.dC);
        }
        bufs.dC = bufs.dB;
        return ACLBLAS_STATUS_SUCCESS;
    }
    return CtrmmCopyB2C(bufs.dC, bufs.dB, m, n, ldb, ldc, cBytes, stream);
}

// Host-buffer convenience driver: uploads A/B, pre-copies B->C on device,
// calls aclblasCtrmm, syncs and downloads C. Invalid/quick-return shapes are
// forwarded to the API unchanged (the operator itself owns validation).
inline aclblasStatus_t aclblasCtrmm_npu(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc, bool alphaOnDevice = false, bool aliasBC = false)
{
    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (handle == nullptr || m <= 0 || n <= 0 || alpha == nullptr || A == nullptr || B == nullptr || C == nullptr ||
        (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) || lda < aDim || ldb < m || ldc < m) {
        return aclblasCtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    }

    // Column-major buffer sizes (complex64): A is lda x aDim, B is ldb x n,
    // C is ldc x n.
    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(ldb) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);

    CtrmmDevBuffers bufs;
    aclblasStatus_t allocSt = AllocCtrmmDevBuffers(bufs, A, B, aBytes, bBytes, cBytes, alphaOnDevice, alpha);
    if (allocSt != ACLBLAS_STATUS_SUCCESS) {
        return allocSt;
    }

    aclrtStream stream = nullptr;
    aclblasStatus_t getStreamRet = aclblasGetStream(handle, &stream);
    if (getStreamRet != ACLBLAS_STATUS_SUCCESS) {
        fprintf(
            stderr, "[WARN] ctrmm: aclblasGetStream failed, ret=%d, fallback to default stream\n",
            static_cast<int>(getStreamRet));
    }
    aclblasStatus_t prepSt = PrepareCtrmmNpuBuffers(bufs, m, n, ldb, ldc, bBytes, cBytes, aliasBC, stream);
    if (prepSt != ACLBLAS_STATUS_SUCCESS) {
        FreeCtrmmDevBuffers(bufs);
        return prepSt;
    }

    const aclblasComplex* alphaArg = alphaOnDevice ? static_cast<const aclblasComplex*>(bufs.dAlpha) : alpha;
    aclblasStatus_t ret = aclblasCtrmm(
        handle, side, uplo, trans, diag, m, n, alphaArg, static_cast<const aclblasComplex*>(bufs.dA), lda,
        static_cast<const aclblasComplex*>(bufs.dB), ldb, static_cast<aclblasComplex*>(bufs.dC), ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = CtrmmSyncAndCopyD2H(handle, bufs.dC, C, cBytes);
    }

    FreeCtrmmDevBuffers(bufs);
    return ret;
}
