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
 *        Arch22 (Atlas A2/A3) implementation.
 *
 *        Applies the rotation to complex64 vectors x and y in place:
 *            x[k] = c*x[k] + s*y[j]
 *            y[j] = c*y[j] - conj(s)*x[k]   (pre-rotation x[k])
 *
 *        incx==incy==1 dispatches to the contiguous SIMD kernel; every other
 *        stride combination (|inc|>1 or negative) goes to the strided scalar kernel.
 *
 *        Boundary contract follows the task book (sections 2.4 and 2.5) for the
 *        arch22 family: handle==nullptr returns HANDLE_IS_NULLPTR, n<0 returns
 *        INVALID_VALUE, n==0 short-circuits to SUCCESS without touching x/y, and
 *        zero / INT32_MIN strides are rejected (INVALID_VALUE). This is the declared
 *        per-arch difference from the arch35 (950PR) implementation, which accepts
 *        n<=0 and zero strides (issue #827).
 */

#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "crot_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void crot_kernel_do(
    uint8_t* x, uint8_t* y, int32_t elementCount, int32_t incx, int32_t incy, float cosValue, float sinRe, float sinIm,
    uint32_t numBlocks, void* stream);

// Validation order is part of the observable contract (task book section 2.5):
//   handle -> n<0 -> n==0 (quick return) -> x/y/c/s null -> incx/incy -> span overflow.
// The handle check must precede the n==0 quick return (issue #765/#861).
static aclblasStatus_t ValidateCrotInputs(
    aclblasHandle_t handle, int n, const aclblasComplex* x, const aclblasComplex* y, const float* c,
    const aclblasComplex* s, int incx, int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCrot", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasCrot", "n must be non-negative, got n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        OP_LOGE("aclblasCrot", "x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (c == nullptr || s == nullptr) {
        OP_LOGE("aclblasCrot", "c/s must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // issue #827: zero / INT32_MIN strides are rejected on arch22 (declared per-arch
    // difference from arch35, which accepts them).
    if (incx == 0 || incx == INT_MIN || incy == 0 || incy == INT_MIN) {
        OP_LOGE("aclblasCrot", "incx/incy must be non-zero and not INT_MIN, got incx=%d incy=%d", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // issue #866: (n-1)*|inc| must fit in int32, checked in int64 on the host so the
    // kernel-side address offset cannot overflow.
    const int64_t absIncX = (incx < 0) ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx);
    const int64_t absIncY = (incy < 0) ? -static_cast<int64_t>(incy) : static_cast<int64_t>(incy);
    const int64_t lastX = static_cast<int64_t>(n - 1) * absIncX;
    const int64_t lastY = static_cast<int64_t>(n - 1) * absIncY;
    if (lastX > INT32_MAX || lastY > INT32_MAX) {
        OP_LOGE(
            "aclblasCrot", "strided offset overflow: n=%d incx=%d incy=%d lastX=%lld lastY=%lld", n, incx, incy,
            static_cast<long long>(lastX), static_cast<long long>(lastY));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // The kernel works in float units: the element count is 2*n (complex64 = 2 floats) and the
    // largest float index is 2*n-1. Reject n > INT32_MAX/2 here so 2*n cannot overflow the
    // int32 elementCount handed to the kernel; without this the kernel would see a negative
    // count, take the "nothing to do" early return and report SUCCESS while leaving x/y
    // unrotated.
    if (n > INT32_MAX / 2) {
        OP_LOGE("aclblasCrot", "n too large: n=%d exceeds INT32_MAX/2", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

extern "C" aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const aclblasComplex* s)
{
    const aclblasStatus_t inputStatus = ValidateCrotInputs(handle, n, x, y, c, s, incx, incy);
    if (inputStatus != ACLBLAS_STATUS_SUCCESS) {
        return inputStatus;
    }
    if (n == 0) {
        // Quick return: no allocation or launch.
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    // Resolve c and s. c is a float scalar; s is complex64. Both may be host or device
    // pointers, queried via aclrtPointerGetAttributes (host dereference / async D2H).
    float cHost = 0.0f;
    aclblasStatus_t readSt = ReadScalarFromHostOrDevice(c, useStream, cHost);
    if (readSt != ACLBLAS_STATUS_SUCCESS) {
        return readSt;
    }
    aclblasComplex sHost{0.0f, 0.0f};
    readSt = ReadScalarFromHostOrDevice(s, useStream, sHost);
    if (readSt != ACLBLAS_STATUS_SUCCESS) {
        return readSt;
    }

    // The kernel addresses interleaved floats, so the element count is 2*n (the same
    // convention as the sibling arch22 csrot). The scalars travel as kernel parameters,
    // so no tiling GM buffer (and no per-call malloc / memcpy / free) is needed.
    const int32_t elementCount = static_cast<int32_t>(n) * 2;

    uint32_t numBlocks = GetAivCoreCount();
    if (numBlocks == 0) {
        numBlocks = 8;
    }

    // Asynchronous, like the other BLAS level-1 entry points: the kernel is enqueued on the
    // handle stream and the caller synchronizes before reading results. No stream sync here.
    crot_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), elementCount, incx, incy, cHost, sHost.real,
        sHost.imag, numBlocks, useStream);
    return ACLBLAS_STATUS_SUCCESS;
}
