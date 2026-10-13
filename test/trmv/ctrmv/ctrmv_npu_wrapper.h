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
#include <cstdio>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Allocate a device buffer of `bytes` and copy `host` into it.
inline aclblasStatus_t CtrmvCopyH2D(const void* host, size_t bytes, void** device)
{
    aclError aclRet = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        printf("[aclblasCtrmv_npu] aclrtMalloc failed, ret=%d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        printf("[aclblasCtrmv_npu] aclrtMemcpy H2D failed, ret=%d\n", static_cast<int>(aclRet));
        aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Synchronize the handle's stream (not the whole device) and copy `device` back to `host`.
inline aclblasStatus_t CtrmvCopyD2H(aclblasHandle_t handle, void* host, const void* device, size_t bytes)
{
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || stream == nullptr) {
        printf("[aclblasCtrmv_npu] aclblasGetStream failed\n");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        printf("[aclblasCtrmv_npu] aclrtSynchronizeStream failed, ret=%d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtMemcpy(host, bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        printf("[aclblasCtrmv_npu] aclrtMemcpy D2H failed, ret=%d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Host wrapper: copies A/x to device memory, calls aclblasCtrmv in place on the
// device buffer, then copies x back. When `a` or `x` is null (NULLPTR fill for
// negative test cases), the pointer is forwarded as-is so the API's own null
// checks are exercised.
inline aclblasStatus_t aclblasCtrmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* a, int lda, aclblasComplex* x, int incx)
{
    if (handle == nullptr || n <= 0 || a == nullptr || x == nullptr) {
        return aclblasCtrmv(handle, uplo, trans, diag, n, a, lda, x, incx);
    }

    const size_t aBytes = static_cast<size_t>(lda) * n * sizeof(aclblasComplex);
    const int absIncx = std::abs(incx);
    const size_t xBytes = static_cast<size_t>((n - 1) * absIncx + 1) * sizeof(aclblasComplex);

    void* dA = nullptr;
    void* dX = nullptr;
    aclblasStatus_t st = CtrmvCopyH2D(a, aBytes, &dA);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CtrmvCopyH2D(x, xBytes, &dX);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(dA);
        return st;
    }

    aclblasStatus_t ret = aclblasCtrmv(
        handle, uplo, trans, diag, n, static_cast<const aclblasComplex*>(dA), lda, static_cast<aclblasComplex*>(dX),
        incx);

    st = CtrmvCopyD2H(handle, x, dX, xBytes);
    aclrtFree(dA);
    aclrtFree(dX);
    return (st == ACLBLAS_STATUS_SUCCESS) ? ret : st;
}
