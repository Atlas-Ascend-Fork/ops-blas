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
#include <cstdlib>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// RAII owner for a device buffer allocated by CsyrAllocCopyToDevice / aclrtMalloc:
// aclrtFree runs at scope exit, so every early-return path (including fatal
// assertion failures) frees what has already been allocated. A null owner frees
// nothing.
using CsyrDeviceBuf = std::unique_ptr<void, aclError (*)(void*)>;

static aclblasStatus_t CsyrAllocCopyToDevice(void** dst, const void* src, size_t bytes)
{
    if (src == nullptr || bytes == 0) {
        *dst = nullptr;
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclError aclRet = aclrtMalloc(dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(*dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dst);
        *dst = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Convenience wrapper for host-side test data: copies x and A to device, invokes
// aclblasCsyr, and copies the updated A back to the host buffer.
// Pass-through rule (inherited from the previous project): calls that can be served
// purely by the host-side validation chain (null handle / n<=0 / null alpha / null x /
// null A / incx==0) are forwarded verbatim with host pointers - the operator must
// return before touching any tensor. Everything else goes through device buffers, so
// even a mis-launching implementation reads valid device memory instead of host
// pointers. `xLenElements` allows a short x buffer (x_len column, K-dim mismatch).
inline aclblasStatus_t aclblasCsyr_npu(aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* x, int incx, aclblasComplex* A, int lda,
    size_t xLenElements)
{
    if (handle == nullptr || n <= 0 || alpha == nullptr || x == nullptr || A == nullptr || incx == 0) {
        return aclblasCsyr(handle, uplo, n, alpha, x, incx, A, lda);
    }

    // K-dim mismatch (spec bc8, x_len column): the caller-provided physical x buffer must be
    // able to hold the logical sequence 1 + (n-1)*|incx| complex elements; a short buffer is
    // reported as INVALID_VALUE before any device transfer (L2_005/L2_006).
    const size_t xLogical = static_cast<size_t>(n > 0 ? 1 + (n - 1) * std::abs(incx) : 0);
    if (xLenElements < xLogical) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const size_t xBytes = xLenElements * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(lda) * n * sizeof(aclblasComplex);

    void* dX = nullptr;
    void* dA = nullptr;

    aclblasStatus_t st = CsyrAllocCopyToDevice(&dX, x, xBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    st = CsyrAllocCopyToDevice(&dA, A, aBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        if (dX) {
            aclrtFree(dX);
        }
        return st;
    }

    aclblasStatus_t ret = aclblasCsyr(
        handle, uplo, n, alpha, static_cast<const aclblasComplex*>(dX), incx, static_cast<aclblasComplex*>(dA), lda);

    aclrtSynchronizeDevice();
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclrtMemcpy(A, aBytes, dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (dX) {
        aclrtFree(dX);
    }
    if (dA) {
        aclrtFree(dA);
    }
    return ret;
}
