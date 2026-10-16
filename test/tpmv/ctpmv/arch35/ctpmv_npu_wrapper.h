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

#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Host-buffer test adapter: returns only after this handle's stream completes and x is copied back.
// The production aclblasCtpmv API accepts device buffers and remains asynchronous.
// nullptr inputs are passed through without device allocation (for error-path testing).
// n <= 0: fast return via direct API call.

// Allocate a device buffer and copy `bytes` from host; on copy failure the buffer is
// released and *dst reset to nullptr. A null src succeeds with *dst left nullptr.
inline aclblasStatus_t CtpmvAllocCopyH2D(void** dst, const void* src, size_t bytes)
{
    *dst = nullptr;
    if (src == nullptr) {
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

inline void CtpmvFreeDev(void* p, void* q)
{
    if (p != nullptr) {
        aclrtFree(p);
    }
    if (q != nullptr) {
        aclrtFree(q);
    }
}

inline aclblasStatus_t aclblasCtpmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx)
{
    // Fast path: null handle or n <= 0, pass through directly
    if (handle == nullptr || n <= 0) {
        return aclblasCtpmv(handle, uplo, trans, diag, n, AP, x, incx);
    }

    aclrtStream stream = nullptr;
    const aclblasStatus_t streamRet = aclblasGetStream(handle, &stream);
    if (streamRet != ACLBLAS_STATUS_SUCCESS) {
        return streamRet;
    }

    const int absIncx = std::abs(incx);
    const size_t apSize = static_cast<size_t>(n) * (n + 1) / 2;
    const size_t xSize = static_cast<size_t>((n - 1) * absIncx + 1);

    const size_t apBytes = apSize * sizeof(aclblasComplex);
    const size_t xBytes = xSize * sizeof(aclblasComplex);

    void* dAP = nullptr;
    void* dX = nullptr;
    aclblasStatus_t st = CtpmvAllocCopyH2D(&dAP, AP, apBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CtpmvAllocCopyH2D(&dX, x, xBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        CtpmvFreeDev(dAP, nullptr);
        return st;
    }

    // Call kernel (in-place: x is both input and output)
    aclblasStatus_t ret = aclblasCtpmv(
        handle, uplo, trans, diag, n, static_cast<const aclblasComplex*>(dAP), static_cast<aclblasComplex*>(dX), incx);

    // A failed submission can leave earlier work queued; drain before releasing either device buffer.
    const aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        // Completion is unproven: leave buffers to context teardown instead of risking use-after-free.
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ret == ACLBLAS_STATUS_SUCCESS && x != nullptr && dX != nullptr) {
        const aclError d2hRet = aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (d2hRet != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    CtpmvFreeDev(dAP, dX);
    return ret;
}
