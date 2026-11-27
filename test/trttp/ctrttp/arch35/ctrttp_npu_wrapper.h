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

// Wraps the ACL lifetime of one aclblasCtrttp call: malloc -> H2D -> operator -> sync ->
// D2H -> free. The diagnostic matrix AP is copied with its trailing guard region included so
// the caller can prove that aclblasCtrttp never writes past n(n+1)/2 packed elements.
static inline aclblasStatus_t CtrttpCopyToDevice(void** dPtr, const void* hPtr, size_t bytes)
{
    if (hPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(*dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*dPtr);
        *dPtr = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t CtrttpCopyFromDevice(void* hPtr, const void* dPtr, size_t bytes)
{
    if (hPtr == nullptr || dPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMemcpy(hPtr, bytes, dPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

// apCapacity is the element count of the host AP buffer, packed length plus tail guard.
inline aclblasStatus_t aclblasCtrttp_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* a, int lda, aclblasComplex* ap,
    int64_t apCapacity)
{
    // n <= 0 never touches A / AP, so the device buffers can be skipped entirely.
    if (handle == nullptr || n <= 0) {
        return aclblasCtrttp(handle, uplo, n, nullptr, lda, nullptr);
    }

    const int64_t allocLda = std::max(1, lda);
    const size_t aBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t apBytes = static_cast<size_t>(apCapacity) * sizeof(aclblasComplex);

    void* dA = nullptr;
    void* dP = nullptr;
    aclblasStatus_t status = CtrttpCopyToDevice(&dA, a, aBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = CtrttpCopyToDevice(&dP, ap, apBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        if (dA != nullptr) {
            aclrtFree(dA);
        }
        return status;
    }

    aclblasStatus_t ret = aclblasCtrttp(
        handle, uplo, n, static_cast<const aclblasComplex*>(dA), lda, static_cast<aclblasComplex*>(dP));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        if (dA != nullptr) {
            aclrtFree(dA);
        }
        if (dP != nullptr) {
            aclrtFree(dP);
        }
        return ret;
    }

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        if (dA != nullptr) {
            aclrtFree(dA);
        }
        if (dP != nullptr) {
            aclrtFree(dP);
        }
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    status = CtrttpCopyFromDevice(ap, dP, apBytes);
    if (dA != nullptr) {
        aclrtFree(dA);
    }
    if (dP != nullptr) {
        aclrtFree(dP);
    }
    return status;
}

