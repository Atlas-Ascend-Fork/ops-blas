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

#include <algorithm>
#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline bool CtrsmCanLaunch(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* a, int lda,
    const aclblasComplex* b, int ldb)
{
    if (handle == nullptr || (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) ||
        (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) ||
        (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) ||
        (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) || m < 0 || n < 0 || alpha == nullptr) {
        return false;
    }
    const int order = side == ACLBLAS_SIDE_LEFT ? m : n;
    if (lda < std::max(1, order) || ldb < std::max(1, m) || m == 0 || n == 0 || b == nullptr) {
        return false;
    }
    const bool alphaZero = alpha->real == 0.0f && alpha->imag == 0.0f;
    return alphaZero || a != nullptr;
}

inline aclError CtrsmAllocCopy(void** device, const void* host, size_t bytes)
{
    *device = nullptr;
    if (host == nullptr || bytes == 0) {
        return ACL_SUCCESS;
    }
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
    }
    return ret;
}

inline aclblasStatus_t aclblasCtrsmNpu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* a, int lda,
    aclblasComplex* b, int ldb)
{
    if (!CtrsmCanLaunch(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)) {
        return aclblasCtrsm(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb);
    }

    const int order = side == ACLBLAS_SIDE_LEFT ? m : n;
    const size_t aBytes = static_cast<size_t>(lda) * order * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(ldb) * n * sizeof(aclblasComplex);
    const bool alphaZero = alpha->real == 0.0f && alpha->imag == 0.0f;
    void* deviceA = nullptr;
    void* deviceB = nullptr;

    if (!alphaZero && CtrsmAllocCopy(&deviceA, a, aBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (CtrsmAllocCopy(&deviceB, b, bBytes) != ACL_SUCCESS) {
        if (deviceA != nullptr) {
            aclrtFree(deviceA);
        }
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t status = aclblasCtrsm(
        handle, side, uplo, trans, diag, m, n, alpha, static_cast<const aclblasComplex*>(deviceA), lda,
        static_cast<aclblasComplex*>(deviceB), ldb);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        aclrtStream stream = nullptr;
        status = aclblasGetStream(handle, &stream);
        if (status == ACLBLAS_STATUS_SUCCESS && stream != nullptr && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            status = ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        if (status == ACLBLAS_STATUS_SUCCESS &&
            aclrtMemcpy(b, bBytes, deviceB, bBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            status = ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    if (deviceA != nullptr) {
        aclrtFree(deviceA);
    }
    if (deviceB != nullptr) {
        aclrtFree(deviceB);
    }
    return status;
}
