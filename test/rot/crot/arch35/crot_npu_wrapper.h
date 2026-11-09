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
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// NPU wrapper — c / s accepted by value, plus an optional csPtrMode that controls
// where the scalar pointers are materialized before the operator call.
//   csPtrMode == "host"   : operator invoked with &c / &s (host stack scalars).
//   csPtrMode == "device" : c and s each allocated on device (aclrtMalloc + H2D),
//                           operator invoked with the device pointers.
//   csPtrMode == "mixed"  : c on host, s on device.
// c and s are resolved independently by the operator via aclrtPointerGetAttributes.
// cIsNull / sIsNull force the corresponding argument to nullptr (negative cases
// from the CSV); the operator then returns ACLBLAS_STATUS_INVALID_VALUE.
// Fast paths skip device allocation entirely.
//   - handle == nullptr                    -> HANDLE_IS_NULLPTR
//   - n <= 0                               -> SUCCESS (no-op)
//   - x == nullptr || y == nullptr         -> INVALID_VALUE
//   - cIsNull || sIsNull                   -> INVALID_VALUE
// Normal path: allocate device buffers covering the stride access span
//   (element span = (n-1)*|inc|+1 for |inc|>=1; n for inc==0, matching the host
//   buffer allocation in the test), H2D copy x/y, optionally H2D copy the
//   device-side scalar(s), call the operator, synchronize, D2H copy back x/y
//   (in-place), then free. Every ACL call is checked; allocated memory is freed
//   on any failure.

// Element span covered by stride access. |inc|>=1: (n-1)*|inc|+1. inc==0: n
// elements (the test allocates n so the H2D span is well-defined even though
// the operator only reuses element 0).
static inline size_t CrotBufElems(int n, int inc)
{
    if (n <= 0)
        return 0;
    const int64_t absInc = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    if (absInc == 0)
        return static_cast<size_t>(n);
    return static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1;
}

// Resolve csPtrMode ("host"/"device"/"mixed") into per-scalar on-device flags.
static inline void CrotResolveCsPtrMode(const std::string& csPtrMode, bool& cOnDevice, bool& sOnDevice)
{
    cOnDevice = (csPtrMode == "device");
    sOnDevice = (csPtrMode == "device") || (csPtrMode == "mixed");
}

// Allocate a device float scalar and H2D-copy the host value in.
static inline aclblasStatus_t CrotPrepareDeviceScalar(float hostVal, void*& dScalar, const float*& scalarArg)
{
    scalarArg = nullptr; // filled on success
    if (aclrtMalloc(&dScalar, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(dScalar, sizeof(float), &hostVal, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(dScalar);
        dScalar = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    scalarArg = static_cast<const float*>(dScalar);
    return ACLBLAS_STATUS_SUCCESS;
}

// Allocate a device complex64 scalar (two interleaved floats) and H2D-copy it in.
static inline aclblasStatus_t CrotPrepareDeviceComplexScalar(
    float re, float im, void*& dScalar, const aclblasComplex*& scalarArg)
{
    scalarArg = nullptr; // filled on success
    if (aclrtMalloc(&dScalar, 2 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const float hostVal[2] = {re, im};
    if (aclrtMemcpy(dScalar, 2 * sizeof(float), hostVal, 2 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(dScalar);
        dScalar = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    scalarArg = static_cast<const aclblasComplex*>(dScalar);
    return ACLBLAS_STATUS_SUCCESS;
}

// Device buffers owned by one aclblasCrot_npu call. Release() frees whatever
// has been allocated so far and is safe to call on any error path.
struct CrotDeviceBufs {
    void* dX = nullptr;
    void* dY = nullptr;
    void* dC = nullptr;
    void* dS = nullptr;
    size_t xBytes = 0;
    size_t yBytes = 0;

    void Release()
    {
        if (dX)
            aclrtFree(dX);
        if (dY)
            aclrtFree(dY);
        if (dC)
            aclrtFree(dC);
        if (dS)
            aclrtFree(dS);
        dX = nullptr;
        dY = nullptr;
        dC = nullptr;
        dS = nullptr;
    }
};

// Allocate the device x/y spans and upload the host data.
static inline aclblasStatus_t CrotAllocAndUpload(
    const aclblasComplex* x, int n, int incx, const aclblasComplex* y, int incy, CrotDeviceBufs& bufs)
{
    bufs.xBytes = CrotBufElems(n, incx) * sizeof(aclblasComplex);
    bufs.yBytes = CrotBufElems(n, incy) * sizeof(aclblasComplex);

    if (aclrtMalloc(&bufs.dX, bufs.xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMalloc(&bufs.dY, bufs.yBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(bufs.dX, bufs.xBytes, x, bufs.xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(bufs.dY, bufs.yBytes, y, bufs.yBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Construct c / s pointers on the side indicated by csPtrMode.
static inline aclblasStatus_t CrotResolveScalarPtrs(
    const float& c, const aclblasComplex& s, bool cOnDevice, bool sOnDevice, CrotDeviceBufs& bufs, const float*& cArg,
    const aclblasComplex*& sArg)
{
    cArg = &c;
    sArg = &s;
    if (cOnDevice) {
        aclblasStatus_t st = CrotPrepareDeviceScalar(c, bufs.dC, cArg);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    if (sOnDevice) {
        aclblasStatus_t st = CrotPrepareDeviceComplexScalar(s.real, s.imag, bufs.dS, sArg);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Wait for the operator, copy x/y back in place, then free the device buffers.
static inline aclblasStatus_t CrotDownloadAndFree(
    aclblasComplex* x, aclblasComplex* y, CrotDeviceBufs& bufs, aclrtStream stream)
{
    // Wait on the stream the operator was launched on. A device-wide sync would also
    // wait for unrelated work and would mask a bug where the kernel is enqueued on a
    // different stream than handle->stream.
    const aclError syncRet = (stream != nullptr) ? aclrtSynchronizeStream(stream) : aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        bufs.Release();
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (aclrtMemcpy(x, bufs.xBytes, bufs.dX, bufs.xBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        bufs.Release();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(y, bufs.yBytes, bufs.dY, bufs.yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        bufs.Release();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    bufs.Release();
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCrot_npu(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float c,
    const aclblasComplex s, const std::string& csPtrMode = "host", bool cIsNull = false, bool sIsNull = false,
    aclrtStream stream = nullptr)
{
    // Fast path: no device buffers needed. c/s are passed as host pointers here
    // (or nullptr for the negative cases); the operator short-circuits before
    // touching them for handle/n/x/y cases.
    if (handle == nullptr || n <= 0 || x == nullptr || y == nullptr || cIsNull || sIsNull) {
        const float* cArg = cIsNull ? nullptr : &c;
        const aclblasComplex* sArg = sIsNull ? nullptr : &s;
        return aclblasCrot(handle, n, x, incx, y, incy, cArg, sArg);
    }

    bool cOnDevice = false;
    bool sOnDevice = false;
    CrotResolveCsPtrMode(csPtrMode, cOnDevice, sOnDevice);

    CrotDeviceBufs bufs;
    aclblasStatus_t ret = CrotAllocAndUpload(x, n, incx, y, incy, bufs);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        bufs.Release();
        return ret;
    }

    const float* cArg = nullptr;
    const aclblasComplex* sArg = nullptr;
    ret = CrotResolveScalarPtrs(c, s, cOnDevice, sOnDevice, bufs, cArg, sArg);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        bufs.Release();
        return ret;
    }

    ret = aclblasCrot(
        handle, n, static_cast<aclblasComplex*>(bufs.dX), incx, static_cast<aclblasComplex*>(bufs.dY), incy, cArg,
        sArg);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        bufs.Release();
        return ret;
    }
    return CrotDownloadAndFree(x, y, bufs, stream);
}
