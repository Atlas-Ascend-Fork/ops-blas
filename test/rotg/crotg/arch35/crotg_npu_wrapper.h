/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of CANN Open Software
 * License Agreement Version 2.0 (the "License"). Please refer to
 * the License for details. You may not use this file except in
 * compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
 * IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See LICENSE in the root of the software
 * repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// ==========================================================================
// NPU wrapper helpers — each returns ACLBLAS_STATUS_SUCCESS or an error
// code, keeping the main pipeline simple and reducing cyclomatic
// complexity.
// ==========================================================================

// Allocate device memory. Returns status; *ptr is set on success,
// unchanged on failure.
inline aclblasStatus_t NpuAllocDevice(size_t bytes, void** ptr)
{
    if (aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Free device memory if allocated (safe to call with nullptr).
inline void NpuFreeDevice(void* ptr)
{
    if (ptr != nullptr) {
        aclrtFree(ptr);
    }
}

// H2D memcpy: copies host -> device. Returns status.
inline aclblasStatus_t NpuH2D(void* dst, const void* src, size_t bytes)
{
    if (aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE)
        != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Sync device then D2H memcpy: copies device -> host. Returns status.
inline aclblasStatus_t NpuSyncAndD2H(void* dst, const void* src,
                                     size_t bytes)
{
    if (aclrtMemcpy(dst, bytes, src, bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Allocate all four device buffers (a, b, c, s). Returns status.
inline aclblasStatus_t NpuAllocAll(void** dA, void** dB, void** dC,
                                     void** dS, size_t complexBytes,
                                     size_t floatBytes)
{
    // On any allocation failure return immediately WITHOUT freeing the
    // already-allocated buffers here: cleanup is centralized in the caller
    // via NpuFreeAll (null-safe). Freeing here as well would double-free.
    // A failed NpuAllocDevice leaves *ptr unchanged (stays nullptr as set by
    // the caller), so NpuFreeAll safely skips it.
    aclblasStatus_t ret = NpuAllocDevice(complexBytes, dA);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = NpuAllocDevice(complexBytes, dB);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = NpuAllocDevice(floatBytes, dC);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = NpuAllocDevice(complexBytes, dS);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Free all four device buffers.
inline void NpuFreeAll(void* dA, void* dB, void* dC, void* dS)
{
    NpuFreeDevice(dS);
    NpuFreeDevice(dC);
    NpuFreeDevice(dB);
    NpuFreeDevice(dA);
}

// Snapshot host inputs and copy to device buffers. Returns status.
inline aclblasStatus_t NpuH2DInputs(aclblasComplex* a,
                                     aclblasComplex* b, void* dA,
                                     void* dB, size_t complexBytes)
{
    aclblasComplex hA = *a;
    aclblasComplex hB = *b;
    aclblasStatus_t ret = NpuH2D(dA, &hA, complexBytes);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    return NpuH2D(dB, &hB, complexBytes);
}

// ==========================================================================
// NPU wrapper — same signature as aclblasCrotg.
// If any pointer is nullptr it is forwarded directly to the API
// (for NullHandle / NullA/B/C/S error-path tests); otherwise:
// allocate device memory -> H2D copy a/b -> call ACL -> D2H copy
// a/c/s (b is read-only, not overwritten). Resource cleanup is
// centralized in NpuFreeAll so no failure path leaks device memory.
// ==========================================================================
inline aclblasStatus_t aclblasCrotg_npu(
    aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b,
    float* c, aclblasComplex* s)
{
    if (handle == nullptr || a == nullptr || b == nullptr ||
        c == nullptr || s == nullptr) {
        return aclblasCrotg(handle, a, b, c, s);
    }

    const size_t complexBytes = sizeof(aclblasComplex); // 8
    const size_t floatBytes = sizeof(float);            // 4

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    void* dS = nullptr;

    aclblasStatus_t ret = NpuAllocAll(&dA, &dB, &dC, &dS,
                                      complexBytes, floatBytes);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        NpuFreeAll(dA, dB, dC, dS);
        return ret;
    }

    ret = NpuH2DInputs(a, b, dA, dB, complexBytes);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        NpuFreeAll(dA, dB, dC, dS);
        return ret;
    }

    ret = aclblasCrotg(handle,
                       static_cast<aclblasComplex*>(dA),
                       static_cast<aclblasComplex*>(dB),
                       static_cast<float*>(dC),
                       static_cast<aclblasComplex*>(dS));
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        } else {
            ret = NpuSyncAndD2H(a, dA, complexBytes);
            if (ret == ACLBLAS_STATUS_SUCCESS) {
                ret = NpuSyncAndD2H(c, dC, floatBytes);
            }
            if (ret == ACLBLAS_STATUS_SUCCESS) {
                ret = NpuSyncAndD2H(s, dS, complexBytes);
            }
        }
    }

    NpuFreeAll(dA, dB, dC, dS);
    return ret;
}
