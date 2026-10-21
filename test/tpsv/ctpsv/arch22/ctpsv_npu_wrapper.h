/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasCtpsv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* ap, aclblasComplex* x, int incx)
{
    if (handle == nullptr || n <= 0 || ap == nullptr || x == nullptr) {
        return aclblasCtpsv(handle, uplo, trans, diag, n, ap, x, incx);
    }

    const size_t apBytes = static_cast<size_t>(n) * (n + 1) / 2 * sizeof(aclblasComplex);
    const int absIncx = std::abs(incx);
    const size_t xBytes = static_cast<size_t>((n - 1) * absIncx + 1) * sizeof(aclblasComplex);

    void* dAp = nullptr;
    void* dX = nullptr;
    aclError aclRet;

    aclRet = aclrtMalloc(&dAp, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dAp, apBytes, ap, apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAp);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAp);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAp);
        aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret = aclblasCtpsv(
        handle, uplo, trans, diag, n, static_cast<const aclblasComplex*>(dAp), static_cast<aclblasComplex*>(dX), incx);

    aclrtSynchronizeDevice();
    aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dAp);
    aclrtFree(dX);
    return ret;
}
