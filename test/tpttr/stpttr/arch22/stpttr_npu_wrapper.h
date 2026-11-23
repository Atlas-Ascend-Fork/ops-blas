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

inline aclblasStatus_t stpttrCopyToDevice(void** dPtr, const void* hPtr, size_t allocBytes, size_t copyBytes)
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

inline aclblasStatus_t stpttrCopyFromDevice(void* hPtr, const void* dPtr, size_t bytes)
{
    if (hPtr == nullptr || dPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclError ret = aclrtMemcpy(hPtr, bytes, dPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

inline void stpttrFreeDevice(void* dP, void* dA)
{
    if (dP != nullptr) {
        aclrtFree(dP);
    }
    if (dA != nullptr) {
        aclrtFree(dA);
    }
}

inline aclblasStatus_t aclblasStpttr_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* ap, float* a, int lda)
{
    if (handle == nullptr || n <= 0) {
        return aclblasStpttr(handle, uplo, n, ap, a, lda);
    }

    const int allocN = std::max(1, n);
    const int allocLda = std::max(allocN, lda);
    const size_t apBytes = static_cast<size_t>(allocN) * static_cast<size_t>(allocN + 1) / 2U * sizeof(float);
    const size_t aAllocBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(allocN) * sizeof(float);
    const size_t aHostBytes = (lda > 0) ? (static_cast<size_t>(lda) * static_cast<size_t>(allocN) * sizeof(float)) : 0U;

    void* dP = nullptr;
    void* dA = nullptr;

    aclblasStatus_t status = stpttrCopyToDevice(&dP, ap, apBytes, apBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = stpttrCopyToDevice(&dA, a, aAllocBytes, aHostBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        stpttrFreeDevice(dP, dA);
        return status;
    }

    const aclblasStatus_t ret =
        aclblasStpttr(handle, uplo, n, static_cast<const float*>(dP), static_cast<float*>(dA), lda);
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        stpttrFreeDevice(dP, dA);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS && aHostBytes > 0U) {
        status = stpttrCopyFromDevice(a, dA, aHostBytes);
    }
    stpttrFreeDevice(dP, dA);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return ret;
}
