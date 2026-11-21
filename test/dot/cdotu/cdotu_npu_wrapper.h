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
#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Host-side device buffer pair plus the result scalar, with a single cleanup path.
namespace cdotu_npu_detail {

struct CdotuDeviceBuffers {
    void* dX = nullptr;
    void* dY = nullptr;
    void* dResult = nullptr;
};

inline size_t CdotuVectorBytes(int n, int inc)
{
    return (n > 0) ? (static_cast<size_t>(n - 1) * static_cast<size_t>(std::abs(inc)) + 1) * sizeof(aclblasComplex)
                   : sizeof(aclblasComplex);
}

inline aclblasStatus_t CdotuCopyIn(void** dst, size_t bytes, const void* src)
{
    if (src == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(*dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*dst);
        *dst = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CdotuCopyOut(void* hostDst, size_t bytes, void* deviceSrc)
{
    aclError ret = aclrtMemcpy(hostDst, bytes, deviceSrc, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

inline void CdotuFreeBuffers(CdotuDeviceBuffers& bufs)
{
    if (bufs.dX != nullptr) {
        aclrtFree(bufs.dX);
        bufs.dX = nullptr;
    }
    if (bufs.dY != nullptr) {
        aclrtFree(bufs.dY);
        bufs.dY = nullptr;
    }
    if (bufs.dResult != nullptr) {
        aclrtFree(bufs.dResult);
        bufs.dResult = nullptr;
    }
}

// Performance measurement helper: warmup (5×) + event-wrapped sampling (100×).
// Returns the steady-state average kernel time in milliseconds via *kernelMs.
inline aclblasStatus_t CdotuMeasurePerf(aclblasHandle_t handle, int n, const aclblasComplex* dx, int incx,
                                        const aclblasComplex* dy, int incy, aclblasComplex* dr,
                                        aclrtStream stream, float* kernelMs, CdotuDeviceBuffers& bufs)
{
    constexpr int kCdotuWarmup = 5;
    constexpr int kCdotuSamples = 100;

    for (int i = 0; i < kCdotuWarmup; i++) {
        aclblasStatus_t ret = aclblasCdotu(handle, n, dx, incx, dy, incy, dr);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            CdotuFreeBuffers(bufs);
            return ret;
        }
    }

    aclrtEvent startEvent = nullptr, stopEvent = nullptr;
    aclError retEvent = aclrtCreateEvent(&startEvent);
    if (retEvent != ACL_SUCCESS) {
        CdotuFreeBuffers(bufs);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    retEvent = aclrtCreateEvent(&stopEvent);
    if (retEvent != ACL_SUCCESS) {
        (void)aclrtDestroyEvent(startEvent);
        CdotuFreeBuffers(bufs);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtRecordEvent(startEvent, stream);
    for (int i = 0; i < kCdotuSamples; i++) {
        aclblasStatus_t ret = aclblasCdotu(handle, n, dx, incx, dy, incy, dr);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            (void)aclrtDestroyEvent(startEvent);
            (void)aclrtDestroyEvent(stopEvent);
            CdotuFreeBuffers(bufs);
            return ret;
        }
    }
    aclrtRecordEvent(stopEvent, stream);
    aclrtSynchronizeEvent(stopEvent);

    float totalMs = 0.0f;
    aclrtEventElapsedTime(&totalMs, startEvent, stopEvent);
    *kernelMs = totalMs / static_cast<float>(kCdotuSamples);

    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(stopEvent);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace cdotu_npu_detail

// aclblasCdotu_npu: wraps the device API for GTest. x/y are complex vectors of
// physical length (n>0 ? 1+(n-1)*|inc| : 0) complex elements.
//
// kernelMs (optional): when non-null, receives the pure kernel execution time in
// milliseconds measured with aclrt events on the handle's stream, wrapping only
// the cdotu kernel launch (host copy-in/copy-out excluded). This is the kernel ms
// from which verify_performance.py derives the per-call NPU latency.

inline aclblasStatus_t aclblasCdotu_npu(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    aclblasComplex* result, float* kernelMs = nullptr)
{
    if (handle == nullptr) {
        return aclblasCdotu(handle, n, x, incx, y, incy, result);
    }

    using namespace cdotu_npu_detail;
    const bool quickReturn = (n <= 0);
    CdotuDeviceBuffers bufs;
    size_t xBytes = CdotuVectorBytes(n, incx);
    size_t yBytes = CdotuVectorBytes(n, incy);

    aclblasStatus_t st = CdotuCopyIn(&bufs.dX, xBytes, (quickReturn) ? nullptr : x);
    if (st == ACLBLAS_STATUS_SUCCESS) {
        st = CdotuCopyIn(&bufs.dY, yBytes, (quickReturn) ? nullptr : y);
    }
    if (st == ACLBLAS_STATUS_SUCCESS) {
        aclError ret = aclrtMalloc(&bufs.dResult, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
        st = (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (st != ACLBLAS_STATUS_SUCCESS) {
        CdotuFreeBuffers(bufs);
        return st;
    }

    aclrtStream stream = nullptr;
    aclblasGetStream(handle, &stream);

    const aclblasComplex* dx = static_cast<const aclblasComplex*>(bufs.dX);
    const aclblasComplex* dy = static_cast<const aclblasComplex*>(bufs.dY);
    aclblasComplex* dr = (result != nullptr) ? static_cast<aclblasComplex*>(bufs.dResult) : nullptr;

    aclblasStatus_t ret = (kernelMs != nullptr)
        ? CdotuMeasurePerf(handle, n, dx, incx, dy, incy, dr, stream, kernelMs, bufs)
        : aclblasCdotu(handle, n, dx, incx, dy, incy, dr);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        CdotuFreeBuffers(bufs);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS && result != nullptr) {
        ret = CdotuCopyOut(result, sizeof(aclblasComplex), bufs.dResult);
    }

    CdotuFreeBuffers(bufs);
    return ret;
}
