/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file crot_host.cpp
 * \brief Host-side API for BLAS Level-1 complex Givens plane rotation: aclblasCrot.
 *        Arch35 (ascend950) implementation.
 *
 *        Applies the rotation to complex64 vectors x and y in place:
 *            x[k] = c*x[k] + s*y[j]
 *            y[j] = c*y[j] - conj(s)*x[k]   (pre-rotation x[k])
 *
 *        incx==incy==1 dispatches to the SIMD membase kernel (bulk GM<->UB tiles);
 *        every other stride combination goes to the SIMT kernel, which keeps a
 *        dedicated serial path for zero strides.
 *
 *        Boundary contract follows the task book for aclblasCrot 950 (sections 2.4 and
 *        2.5) instead of a reference-BLAS routine, since reference BLAS has no crot with a
 *        complex s: handle==nullptr returns HANDLE_IS_NULLPTR, n<=0 short-circuits to
 *        SUCCESS without touching x/y, and zero / negative strides are legal inputs.
 */

#include <cstdint>
#include <climits>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "crot_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

static aclblasStatus_t ValidateCrotParams(
    const aclblasComplex* x, const aclblasComplex* y, const float* c, const aclblasComplex* s)
{
    if (x == nullptr || y == nullptr) {
        OP_LOGE("aclblasCrot", "x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (c == nullptr || s == nullptr) {
        OP_LOGE("aclblasCrot", "c/s must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Pointer location query: determine whether a c/s scalar pointer lives on the
// host or on the device (NPU HBM). On query failure the call is rejected,
// consistent with aclblasSrot.
// ==========================================================================
static aclblasStatus_t CrotCheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCrot", "aclrtPointerGetAttributes failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Resolve the c/s source; c and s are handled independently.
//   host pointer   -> dereference once, store the scalar in tiling, forward nullptr
//   device pointer -> store 0 placeholder, forward the pointer verbatim
// s is complex64, so the device path forwards &s->real (two interleaved floats).
// ==========================================================================
static aclblasStatus_t PrepareCrotCsSource(
    const float* c, const aclblasComplex* s, CrotTilingData& tiling, float** cPtr, float** sPtr)
{
    bool cIsDevice = false;
    bool sIsDevice = false;
    aclblasStatus_t cLocSt = CrotCheckPtrLocation(c, &cIsDevice);
    if (cLocSt != ACLBLAS_STATUS_SUCCESS) {
        return cLocSt;
    }
    aclblasStatus_t sLocSt = CrotCheckPtrLocation(s, &sIsDevice);
    if (sLocSt != ACLBLAS_STATUS_SUCCESS) {
        return sLocSt;
    }

    tiling.cIsDevice = cIsDevice ? 1u : 0u;
    tiling.sIsDevice = sIsDevice ? 1u : 0u;
    tiling.cosValue = cIsDevice ? 0.0f : (*c);
    tiling.sinRe = sIsDevice ? 0.0f : s->real;
    tiling.sinIm = sIsDevice ? 0.0f : s->imag;
    *cPtr = cIsDevice ? const_cast<float*>(c) : nullptr;
    *sPtr = sIsDevice ? const_cast<float*>(reinterpret_cast<const float*>(s)) : nullptr;
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Tiling: the element span (n-1)*|inc| is computed in int64 and range-checked
// before narrowing, since the SIMT kernel indexes with uint32_t offsets. SIMT
// threads per block are sized from the per-block element count; the contiguous
// SIMD path does not use them.
// ==========================================================================
static aclblasStatus_t CalCrotTiling(uint32_t n, int incx, int incy, uint32_t numBlocks, CrotTilingData& tiling)
{
    int64_t absIncX = (incx < 0) ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx);
    int64_t absIncY = (incy < 0) ? -static_cast<int64_t>(incy) : static_cast<int64_t>(incy);
    int64_t lastX = static_cast<int64_t>(n - 1) * absIncX;
    int64_t lastY = static_cast<int64_t>(n - 1) * absIncY;
    if (lastX > INT32_MAX || lastY > INT32_MAX) {
        OP_LOGE(
            "aclblasCrot", "strided offset overflow: n=%u incx=%d incy=%d lastX=%lld lastY=%lld",
            static_cast<unsigned>(n), incx, incy, static_cast<long long>(lastX), static_cast<long long>(lastY));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    tiling.incx = incx;
    tiling.incy = incy;
    if (tiling.tilingKey == 1U) {
        tiling.nthreads =
            std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Launch — tiling computation + asynchronous kernel launch
// ==========================================================================
static aclblasStatus_t LaunchCrotKernel(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const aclblasComplex* s)
{
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCrot", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t totalN = static_cast<uint32_t>(n);

    CrotTilingData tiling{};
    tiling.totalN = totalN;
    // Aliased buffers with different strides cannot be split across threads: iteration i
    // writes an element that iteration j reads, so the outcome would depend on scheduling.
    // The same-pointer case is detected here and routed to the single-thread serial path.
    // Partially overlapping buffers cannot be detected cheaply and stay out of contract.
    tiling.serialOnly = ((x == y) && (incx != incy)) ? 1U : 0U;

    // Contiguous (incx == incy == 1) takes the SIMD membase path: one core per
    // 1/numBlocks slice, walked in UB tiles. Any other stride (including 0 and
    // negative) stays on the SIMT kernel.
    uint32_t numBlocks;
    aclblasStatus_t tilingSt;
    if (incx == 1 && incy == 1) {
        tiling.tilingKey = 0;
        numBlocks = (totalN < aivCoreNum) ? totalN : aivCoreNum;
        tiling.perCoreN = totalN / numBlocks;
        tiling.remainder = totalN % numBlocks;
        tilingSt = CalCrotTiling(totalN, incx, incy, numBlocks, tiling);
    } else {
        tiling.tilingKey = 1;
        // Blocks: enough to cover the work at SIMT_MIN_THREAD_NUM granularity, capped
        // by the AIV core count. n>0 here, so the result is always >= 1.
        uint32_t blocksByThreads = CeilDiv<uint32_t>(totalN, SIMT_MIN_THREAD_NUM);
        numBlocks = (blocksByThreads < aivCoreNum) ? blocksByThreads : aivCoreNum;
        if (numBlocks == 0) {
            numBlocks = 1;
        }
        tilingSt = CalCrotTiling(totalN, incx, incy, numBlocks, tiling);
    }
    if (tilingSt != ACLBLAS_STATUS_SUCCESS) {
        return tilingSt;
    }

    // Resolve c/s source (host scalar vs device GM pointer); fills
    // cosValue/sinRe/sinIm/cIsDevice/sIsDevice and yields forwarded pointers.
    float* cPtr = nullptr;
    float* sPtr = nullptr;
    aclblasStatus_t csSt = PrepareCrotCsSource(c, s, tiling, &cPtr, &sPtr);
    if (csSt != ACLBLAS_STATUS_SUCCESS) {
        return csSt;
    }

    OP_LOGD(
        "aclblasCrot",
        "tiling: key=%u totalN=%u incx=%d incy=%d c=%f sr=%f si=%f cIsDevice=%u sIsDevice=%u numBlocks=%u "
        "nthreads=%u serialOnly=%u perCoreN=%u remainder=%u",
        tiling.tilingKey, tiling.totalN, incx, incy, tiling.cosValue, tiling.sinRe, tiling.sinIm, tiling.cIsDevice,
        tiling.sIsDevice, numBlocks, tiling.nthreads, tiling.serialOnly, tiling.perCoreN, tiling.remainder);
    OP_LOGI(
        "aclblasCrot", "launching kernel: key=%u blocks=%u cIsDevice=%u sIsDevice=%u", tiling.tilingKey, numBlocks,
        tiling.cIsDevice, tiling.sIsDevice);

    // In-place operation: no workspace needed.
    crot_kernel_do(reinterpret_cast<float*>(x), reinterpret_cast<float*>(y), cPtr, sPtr, numBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Public API entry — dispatch only (validate then launch)
// ==========================================================================
extern "C" aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const aclblasComplex* s)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCrot", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateCrotParams(x, y, c, s);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    return LaunchCrotKernel(handle, n, x, incx, y, incy, c, s);
}
