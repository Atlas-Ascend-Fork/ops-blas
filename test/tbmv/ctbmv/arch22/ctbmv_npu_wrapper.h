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

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t CtbmvCopyInputToDevice(void** device, const void* host, size_t bytes)
{
    if (host == nullptr || bytes == 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCtbmv_npu(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int n, int k, const aclblasComplex* a, int lda, aclblasComplex* x, int incx)
{
    const bool hasStorage = n > 0 && lda > 0;
    const size_t aBytes = hasStorage ? static_cast<size_t>(lda) * n * sizeof(aclblasComplex) : 0U;
    const int64_t absIncx = incx >= 0 ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const size_t xCount = n > 0 && absIncx > 0 ? static_cast<size_t>(n - 1) * absIncx + 1U : 0U;
    const size_t xBytes = xCount * sizeof(aclblasComplex);

    void* deviceA = nullptr;
    void* deviceX = nullptr;
    aclblasStatus_t status = CtbmvCopyInputToDevice(&deviceA, a, aBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = CtbmvCopyInputToDevice(&deviceX, x, xBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        if (deviceA != nullptr) {
            aclrtFree(deviceA);
        }
        return status;
    }

    const aclblasComplex* aArg = a == nullptr ? nullptr : static_cast<const aclblasComplex*>(deviceA);
    aclblasComplex* xArg = x == nullptr ? nullptr : static_cast<aclblasComplex*>(deviceX);
    status = aclblasCtbmv(handle, uplo, trans, diag, n, k, aArg, lda, xArg, incx);
    if (status == ACLBLAS_STATUS_SUCCESS && n > 0) {
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
            aclrtMemcpy(x, xBytes, deviceX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            status = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (deviceA != nullptr) {
        aclrtFree(deviceA);
    }
    if (deviceX != nullptr) {
        aclrtFree(deviceX);
    }
    return status;
}
