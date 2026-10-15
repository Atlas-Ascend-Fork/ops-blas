/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"

// Unified release chain for the five device buffers (shared by the wrapper
// destructor and the perf.h error/normal paths, replacing three structurally
// identical releases across files that triggered a codecheck duplicate-code warning).
static inline void Cher2kFreeDeviceBuffers(void* dAlpha, void* dA, void* dB, void* dBeta, void* dC)
{
    if (dAlpha)
        aclrtFree(dAlpha);
    if (dA)
        aclrtFree(dA);
    if (dB)
        aclrtFree(dB);
    if (dBeta)
        aclrtFree(dBeta);
    if (dC)
        aclrtFree(dC);
}

struct Cher2kDeviceBuffers {
    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dB = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;

    ~Cher2kDeviceBuffers() { Cher2kFreeDeviceBuffers(dAlpha, dA, dB, dBeta, dC); }
};

static aclblasStatus_t Cher2kAllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    aclError aclRet = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("Cher2kAllocAndCopy", "aclrtMalloc failed, ret=%d", aclRet);
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("Cher2kAllocAndCopy", "aclrtMemcpy H2D failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Allocate+copy only when hostPtr is non-null and bytes > 0; leaves devPtr null
// otherwise so the caller can pass nullptr straight through to the API — this
// is how the nullAlpha / nullBeta / nullA / nullB / nullC cases reach the
// operator's own validation paths.
static aclblasStatus_t Cher2kTryAllocAndCopy(const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return Cher2kAllocAndCopy(devPtr, hostPtr, bytes);
}

// [codecheck #12/#13] Tail after a successful call: "synchronize + copy C back".
// Synchronize the device first, then D2H-copy when both host C and the device
// buffer exist; ret is passed through unchanged. The failure path does not enter
// this function (the main function has already returned early, so no sync happens).
static aclblasStatus_t Cher2kSyncAndCopyBack(aclblasStatus_t ret, void* dC, void* C, size_t cBytes)
{
    // C == nullptr with beta == 0 returns SUCCESS without writing: the wrapper
    // has no device buffer to sync in that case, so the canary check in the
    // host test verifies the host buffer stayed untouched.
    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCher2k_npu", "aclrtSynchronizeDevice failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (C != nullptr && dC != nullptr) {
        aclRet = aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasCher2k_npu", "aclrtMemcpy D2H failed, ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ret;
}

// [codecheck #12/#13] Ptr-selection helper: "prefer the device buffer, otherwise
// pass the host pointer through". In the nullAlpha/nullBeta/nullC passthrough
// cases the wrapper has no device buffer (dev == null), so the caller's host
// pointer is handed to the operator as-is, reaching the operator's own
// validation / quick-return path.
template <typename T>
static T* Cher2kPickDevicePtr(void* dev, T* host)
{
    return dev != nullptr ? static_cast<T*>(dev) : host;
}

// [codecheck #12/#13] Upload stage for the five device buffers: TryAllocAndCopy
// each segment in the fixed order alpha -> A -> B -> beta -> C; any failure
// returns the error code immediately (the RAII destructor of bufs frees the
// already-uploaded prefix segments).
static aclblasStatus_t Cher2kUploadBuffers(
    Cher2kDeviceBuffers& bufs, const aclblasComplex* effAlpha, const aclblasComplex* A, size_t aBytes,
    const aclblasComplex* B, size_t bBytes, const float* effBeta, const void* C, size_t cBytes)
{
    constexpr size_t alphaBytes = sizeof(aclblasComplex); // 2 floats
    constexpr size_t betaBytes = sizeof(float);           // 1 float
    aclblasStatus_t st = Cher2kTryAllocAndCopy(effAlpha, alphaBytes, bufs.dAlpha);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k_npu", "upload alpha failed, st=%d", static_cast<int>(st));
        return st;
    }
    st = Cher2kTryAllocAndCopy(A, aBytes, bufs.dA);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k_npu", "upload A failed, st=%d", static_cast<int>(st));
        return st;
    }
    st = Cher2kTryAllocAndCopy(B, bBytes, bufs.dB);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k_npu", "upload B failed, st=%d", static_cast<int>(st));
        return st;
    }
    st = Cher2kTryAllocAndCopy(effBeta, betaBytes, bufs.dBeta);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k_npu", "upload beta failed, st=%d", static_cast<int>(st));
        return st;
    }
    return Cher2kTryAllocAndCopy(C, cBytes, bufs.dC);
}

// NPU wrapper for aclblasCher2k. Extends the cherk wrapper with:
//   - dB upload (rank-2k takes two operand matrices of identical shape)
//   - nullAlpha / nullBeta passthrough (device scalar pointers forced to null)
//   - nullC passthrough: C == nullptr with n > 0 exercises the C-pointer semantics
//     (beta != 0 -> INVALID_VALUE, beta == 0 -> SUCCESS without write), so the
//     wrapper deliberately passes C through and only syncs/copies back when a
//     real host buffer exists.
// When handle is null or n <= 0 it forwards directly so the operator's own
// validation / quick-return paths are exercised.
inline aclblasStatus_t aclblasCher2k_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const float* beta, aclblasComplex* C, int ldc,
    bool nullAlpha = false, bool nullBeta = false)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCher2k(
            handle, uplo, trans, n, k, (nullAlpha ? nullptr : alpha), A, lda, B, ldb, (nullBeta ? nullptr : beta), C,
            ldc);
    }

    const int abCols = (trans == ACLBLAS_OP_N) ? k : n;
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(ldb) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);

    const aclblasComplex* effAlpha = nullAlpha ? nullptr : alpha;
    const float* effBeta = nullBeta ? nullptr : beta;

    Cher2kDeviceBuffers bufs;
    aclblasStatus_t st = Cher2kUploadBuffers(bufs, effAlpha, A, aBytes, B, bBytes, effBeta, C, cBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k_npu", "Cher2kUploadBuffers failed, st=%d", static_cast<int>(st));
        return st;
    }

    aclblasStatus_t ret = aclblasCher2k(
        handle, uplo, trans, n, k, Cher2kPickDevicePtr(bufs.dAlpha, effAlpha), Cher2kPickDevicePtr(bufs.dA, A), lda,
        Cher2kPickDevicePtr(bufs.dB, B), ldb, Cher2kPickDevicePtr(bufs.dBeta, effBeta), Cher2kPickDevicePtr(bufs.dC, C),
        ldc);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCher2k_npu", "aclblasCher2k returned st=%d", static_cast<int>(ret));
        return ret;
    }
    return Cher2kSyncAndCopyBack(ret, bufs.dC, C, cBytes);
}
