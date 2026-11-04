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
// where the c / s scalar pointers are materialized before being passed to the operator:
//   csPtrMode == "host"   : operator invoked with &c / &s (host stack scalars).
//   csPtrMode == "device" : c and s each allocated on device (aclrtMalloc + H2D copy).
//   csPtrMode == "mixed"  : c stays on host (&c), s allocated on device.
// The operator auto-determines each pointer's location via aclrtPointerGetAttributes,
// so the wrapper only constructs the pointer at the desired side.
// cIsNull / sIsNull carry the negative CSV cases (c / s column == "null"): the wrapper
// must forward a REAL nullptr to the operator, which a by-value parameter cannot
// express. When set, the corresponding scalar is forwarded as nullptr and the device
// scalar allocation for that side is skipped; the operator rejects the call in its
// parameter-validation stage (ACLBLAS_STATUS_INVALID_VALUE) before any kernel launch,
// so the device x/y buffers are simply freed untouched.
// Fast paths (passed through without device allocation):
//   - handle == nullptr  : operator returns ACLBLAS_STATUS_HANDLE_IS_NULLPTR
//   - n <= 0             : operator returns ACLBLAS_STATUS_SUCCESS (no-op)
//   - x == nullptr or y == nullptr (n > 0): operator returns ACLBLAS_STATUS_INVALID_VALUE
// Normal path: malloc device buffers covering the stride access range
//   (element span = (n-1)*|inc|+1 complex elements for both x and y), H2D copy x/y,
//   optionally H2D copy the device-side c / s scalar(s), call the operator,
//   synchronize, D2H copy back x/y (in-place modification), then free.
// Every ACL call is checked; on any failure already-allocated device memory is
// freed via the freeAll lambda before returning a structured error code (no leak).

// Element span covered by stride access: (n-1)*|inc| + 1 complex elements (>= n when |inc|>=1).
static inline size_t CsrotBufElems(int n, int inc)
{
    if (n <= 0)
        return 0;
    int absInc = (inc < 0) ? -inc : inc;
    if (absInc == 0)
        absInc = 1; // inc==0 reuses element 0; a single element span suffices.
    return static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1;
}

// Resolve csPtrMode ("host"/"device"/"mixed") into per-scalar on-device flags.
static inline void CsrotResolveCsPtrMode(const std::string& csPtrMode, bool& cOnDevice, bool& sOnDevice)
{
    cOnDevice = (csPtrMode == "device");
    sOnDevice = (csPtrMode == "device") || (csPtrMode == "mixed");
}

// Allocate a device scalar (sizeof(float)) and H2D-copy the host value in.
static inline aclblasStatus_t CsrotPrepareDeviceScalar(float hostVal, void*& dScalar, const float*& scalarArg)
{
    scalarArg = nullptr;
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

// Device resources owned by one wrapper invocation. Free() is null-safe and
// idempotent, so every early-return path can release whatever was staged
// without leaks.
struct CsrotDeviceResources {
    void* x = nullptr;
    void* y = nullptr;
    void* c = nullptr;
    void* s = nullptr;

    void Free()
    {
        if (x != nullptr) {
            aclrtFree(x);
            x = nullptr;
        }
        if (y != nullptr) {
            aclrtFree(y);
            y = nullptr;
        }
        if (c != nullptr) {
            aclrtFree(c);
            c = nullptr;
        }
        if (s != nullptr) {
            aclrtFree(s);
            s = nullptr;
        }
    }
};

// Allocate device buffers covering the stride access range of x/y and stage the
// host inputs (H2D). On failure everything allocated so far is released.
static inline aclblasStatus_t CsrotStageInputs(
    int n, int incx, int incy, const aclblasComplex* x, const aclblasComplex* y, CsrotDeviceResources& res)
{
    const size_t xBytes = CsrotBufElems(n, incx) * sizeof(aclblasComplex);
    const size_t yBytes = CsrotBufElems(n, incy) * sizeof(aclblasComplex);
    if (aclrtMalloc(&res.x, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&res.y, yBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        res.Free();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(res.x, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(res.y, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        res.Free();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Resolve the c / s pointer sides per csPtrMode: host side -> &c / &s (the
// caller's scalars, which outlive this helper — c/s are taken by const
// reference so the pointers stay valid after return), device side -> allocated
// + H2D-staged scalar, null side -> nullptr (no allocation; the operator rejects
// it in validation). On failure everything is released.
static inline aclblasStatus_t CsrotStageScalars(
    const float& c, const float& s, bool cOnDevice, bool sOnDevice, bool cIsNull, bool sIsNull,
    CsrotDeviceResources& res, const float*& cArg, const float*& sArg)
{
    cArg = cIsNull ? nullptr : &c;
    sArg = sIsNull ? nullptr : &s;
    if (cOnDevice && !cIsNull) {
        aclblasStatus_t st = CsrotPrepareDeviceScalar(c, res.c, cArg);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            res.Free();
            return st;
        }
    }
    if (sOnDevice && !sIsNull) {
        aclblasStatus_t st = CsrotPrepareDeviceScalar(s, res.s, sArg);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            res.Free();
            return st;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Run the operator on the staged device buffers, synchronize, and copy the
// in-place results back to host (D2H). Releases the device resources on every
// exit path.
static inline aclblasStatus_t CsrotRunAndFetch(
    aclblasHandle_t handle, int n, int incx, int incy, aclblasComplex* x, aclblasComplex* y, CsrotDeviceResources& res,
    const float* cArg, const float* sArg)
{
    const size_t xBytes = CsrotBufElems(n, incx) * sizeof(aclblasComplex);
    const size_t yBytes = CsrotBufElems(n, incy) * sizeof(aclblasComplex);
    aclblasStatus_t ret = aclblasCsrot(
        handle, n, static_cast<aclblasComplex*>(res.x), incx, static_cast<aclblasComplex*>(res.y), incy, cArg, sArg);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        res.Free();
        return ret;
    }
    // 仅同步算子绑定流（而非整设备）：若算子误用默认流而非 handle->stream，
    // SynchronizeStream 无法"蒙混过关"，从而暴露流序/错流问题。
    aclrtStream runStream = nullptr;
    if (aclblasGetStream(handle, &runStream) != ACLBLAS_STATUS_SUCCESS) {
        res.Free();
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (aclrtSynchronizeStream(runStream) != ACL_SUCCESS) {
        res.Free();
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (aclrtMemcpy(x, xBytes, res.x, xBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(y, yBytes, res.y, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        res.Free();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    res.Free();
    return ret;
}

inline aclblasStatus_t aclblasCsrot_npu(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float c,
    const float s, const std::string& csPtrMode = "host", bool cIsNull = false, bool sIsNull = false)
{
    // Fast path: no device buffers needed (handle null / n<=0 / nullptr x or y).
    // A null c / s is a pure parameter-validation case: the operator rejects it
    // before any kernel launch, so the host x/y pointers are never dereferenced.
    if (handle == nullptr || n <= 0 || x == nullptr || y == nullptr) {
        return aclblasCsrot(handle, n, x, incx, y, incy, cIsNull ? nullptr : &c, sIsNull ? nullptr : &s);
    }

    // Stride-overflow fast path: when |(n-1)*inc| exceeds INT32_MAX the operator
    // rejects the call (CalCsrotTilingStrided -> ACLBLAS_STATUS_INVALID_VALUE) before
    // any kernel launch, so x/y are never dereferenced. Short-circuiting here avoids
    // an impossible ~16GB device allocation (and host-vector overflow) for these
    // negative cases; the test feeds a tiny dummy buffer and this path forwards it
    // directly to the operator purely for parameter validation.
    if (n > 0 && x != nullptr && y != nullptr) {
        const int64_t aIncx = (incx < 0) ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx);
        const int64_t aIncy = (incy < 0) ? -static_cast<int64_t>(incy) : static_cast<int64_t>(incy);
        const int64_t offX = static_cast<int64_t>(n - 1) * aIncx;
        const int64_t offY = static_cast<int64_t>(n - 1) * aIncy;
        if (offX > INT32_MAX || offX < -static_cast<int64_t>(INT32_MAX) || offY > INT32_MAX ||
            offY < -static_cast<int64_t>(INT32_MAX)) {
            return aclblasCsrot(handle, n, x, incx, y, incy, cIsNull ? nullptr : &c, sIsNull ? nullptr : &s);
        }
    }

    bool cOnDevice = false;
    bool sOnDevice = false;
    CsrotResolveCsPtrMode(csPtrMode, cOnDevice, sOnDevice);

    CsrotDeviceResources res;
    aclblasStatus_t st = CsrotStageInputs(n, incx, incy, x, y, res);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const float* cArg = nullptr;
    const float* sArg = nullptr;
    st = CsrotStageScalars(c, s, cOnDevice, sOnDevice, cIsNull, sIsNull, res, cArg, sArg);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return CsrotRunAndFetch(handle, n, incx, incy, x, y, res, cArg, sArg);
}
