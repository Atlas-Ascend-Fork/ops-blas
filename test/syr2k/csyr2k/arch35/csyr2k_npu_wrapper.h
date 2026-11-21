/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYR2K_NPU_WRAPPER_H
#define CSYR2K_NPU_WRAPPER_H

#include <algorithm>
#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct Csyr2kDeviceBuffers {
    void* alpha = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    void* beta = nullptr;
    void* c = nullptr;

    ~Csyr2kDeviceBuffers()
    {
        if (alpha) {
            aclrtFree(alpha);
        }
        if (a) {
            aclrtFree(a);
        }
        if (b) {
            aclrtFree(b);
        }
        if (beta) {
            aclrtFree(beta);
        }
        if (c) {
            aclrtFree(c);
        }
    }
};

inline aclblasStatus_t Csyr2kAllocAndCopy(void*& device, const void* host, size_t bytes)
{
    if (host == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        device = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool Csyr2kCanUseDeviceWrapper(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta,
    const aclblasComplex* C, int ldc)
{
    if (handle == nullptr || n <= 0 || k < 0 || alpha == nullptr || beta == nullptr || C == nullptr) {
        return false;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return false;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return false;
    }
    int minLd = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < minLd || ldb < minLd || ldc < std::max(1, n)) {
        return false;
    }
    if (k > 0 && (A == nullptr || B == nullptr)) {
        return false;
    }
    return true;
}

inline aclblasStatus_t aclblasCsyr2k_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C,
    int ldc)
{
    if (!Csyr2kCanUseDeviceWrapper(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)) {
        return aclblasCsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    int matrixCols = (trans == ACLBLAS_OP_N) ? k : n;
    size_t aBytes = static_cast<size_t>(lda) * matrixCols * sizeof(aclblasComplex);
    size_t bBytes = static_cast<size_t>(ldb) * matrixCols * sizeof(aclblasComplex);
    size_t cBytes = static_cast<size_t>(ldc) * n * sizeof(aclblasComplex);

    Csyr2kDeviceBuffers buffers;
    aclblasStatus_t status = Csyr2kAllocAndCopy(buffers.alpha, alpha, sizeof(aclblasComplex));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = Csyr2kAllocAndCopy(buffers.a, A, aBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = Csyr2kAllocAndCopy(buffers.b, B, bBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = Csyr2kAllocAndCopy(buffers.beta, beta, sizeof(aclblasComplex));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = Csyr2kAllocAndCopy(buffers.c, C, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    aclblasStatus_t ret = aclblasCsyr2k(
        handle, uplo, trans, n, k, static_cast<const aclblasComplex*>(buffers.alpha),
        static_cast<const aclblasComplex*>(buffers.a), lda, static_cast<const aclblasComplex*>(buffers.b), ldb,
        static_cast<const aclblasComplex*>(buffers.beta), static_cast<aclblasComplex*>(buffers.c), ldc);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (aclrtMemcpy(C, cBytes, buffers.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CSYR2K_NPU_WRAPPER_H
