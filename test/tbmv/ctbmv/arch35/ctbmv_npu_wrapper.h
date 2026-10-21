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

inline aclblasStatus_t aclblasCtbmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const aclblasComplex* a, int lda, aclblasComplex* x, int incx)
{
    // For invalid inputs (null handle, n<=0, null buffers, or invalid enums) call the
    // operator directly so it can return the proper status (HANDLE_IS_NULLPTR / INVALID_VALUE /
    // INVALID_ENUM) without attempting a device-side copy of invalid host pointers.
    if (handle == nullptr || n <= 0 || a == nullptr || x == nullptr) {
        return aclblasCtbmv(handle, uplo, trans, diag, n, k, a, lda, x, incx);
    }

    const size_t aBytes = static_cast<size_t>(lda) * n * sizeof(aclblasComplex);
    const int absIncx = std::abs(incx);
    const size_t xBytes = (static_cast<size_t>(n - 1) * absIncx + 1) * sizeof(aclblasComplex);

    void* dA = nullptr;
    void* dX = nullptr;
    aclError aclRet;

    aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dA, aBytes, a, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret = aclblasCtbmv(
        handle, uplo, trans, diag, n, k,
        static_cast<const aclblasComplex*>(dA), lda, static_cast<aclblasComplex*>(dX), incx);

    aclrtSynchronizeDevice();
    aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dA);
    aclrtFree(dX);
    return ret;
}

