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

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t strttpCopyToDevice(void** dPtr, const void* hPtr, size_t allocBytes, size_t copyBytes)
{
    if (hPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(dPtr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (copyBytes > 0U) {
        ret = aclrtMemcpy(*dPtr, allocBytes, hPtr, copyBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            aclrtFree(*dPtr);
            *dPtr = nullptr;
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t strttpCopyFromDevice(void* hPtr, const void* dPtr, size_t bytes)
{
    if (hPtr == nullptr || dPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclError ret = aclrtMemcpy(hPtr, bytes, dPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

inline void strttpFreeDevice(void* dA, void* dP)
{
    if (dA != nullptr) {
        aclrtFree(dA);
    }
    if (dP != nullptr) {
        aclrtFree(dP);
    }
}

inline aclblasStatus_t aclblasStrttp_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* a, int lda, float* ap)
{
    if (handle == nullptr || n <= 0) {
        return aclblasStrttp(handle, uplo, n, a, lda, ap);
    }

    const int allocN = std::max(1, n);
    const int allocLda = std::max(allocN, lda);
    const size_t aAllocBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(allocN) * sizeof(float);
    const size_t aHostBytes = (lda > 0) ? (static_cast<size_t>(lda) * static_cast<size_t>(allocN) * sizeof(float)) : 0U;
    const size_t apBytes = static_cast<size_t>(allocN) * static_cast<size_t>(allocN + 1) / 2U * sizeof(float);

    void* dA = nullptr;
    void* dP = nullptr;

    aclblasStatus_t status = strttpCopyToDevice(&dA, a, aAllocBytes, aHostBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = strttpCopyToDevice(&dP, ap, apBytes, apBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        strttpFreeDevice(dA, dP);
        return status;
    }

    const aclblasStatus_t ret =
        aclblasStrttp(handle, uplo, n, static_cast<const float*>(dA), lda, static_cast<float*>(dP));
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        strttpFreeDevice(dA, dP);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        status = strttpCopyFromDevice(ap, dP, apBytes);
    }
    strttpFreeDevice(dA, dP);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return ret;
}
