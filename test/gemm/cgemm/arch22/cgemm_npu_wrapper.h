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
#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// arch22 copy of the device-pointer wrapper. The arch directories are compiled in
// isolation (see ops_blas_get_test_target_sources), so the arch35 wrapper is not
// visible here; the two are intentionally identical in behaviour.
// Unlike the arch35 wrapper this one also uploads alpha and beta, because the
// arch22 implementation reads them back through the host-or-device helper and
// the test wants to exercise the device-scalar branch as well as the host one.
struct CgemmArch22DeviceBuffers {
    void* alpha = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    void* beta = nullptr;
    void* c = nullptr;

    ~CgemmArch22DeviceBuffers()
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

inline aclblasStatus_t CgemmArch22AllocAndCopy(void*& device, const void* host, size_t bytes)
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

inline bool CgemmArch22CanUseDeviceWrapper(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb,
    const aclblasComplex* beta, const aclblasComplex* c, int ldc)
{
    if (handle == nullptr || m <= 0 || n <= 0 || k < 0 || alpha == nullptr || beta == nullptr || c == nullptr) {
        return false;
    }
    if (transA != ACLBLAS_OP_N && transA != ACLBLAS_OP_T && transA != ACLBLAS_OP_C) {
        return false;
    }
    if (transB != ACLBLAS_OP_N && transB != ACLBLAS_OP_T && transB != ACLBLAS_OP_C) {
        return false;
    }
    const int minimumLda = transA == ACLBLAS_OP_N ? std::max(1, m) : std::max(1, k);
    const int minimumLdb = transB == ACLBLAS_OP_N ? std::max(1, k) : std::max(1, n);
    if (lda < minimumLda || ldb < minimumLdb || ldc < std::max(1, m)) {
        return false;
    }
    return !(k > 0 && (a == nullptr || b == nullptr));
}

// Uploads alpha, A, B, beta and C into device buffers held by `buffers`.
aclblasStatus_t CgemmArch22UploadBuffers(
    CgemmArch22DeviceBuffers& buffers, const aclblasComplex* alpha, const aclblasComplex* a, const aclblasComplex* b,
    const aclblasComplex* beta, aclblasComplex* c, size_t aBytes, size_t bBytes, size_t cBytes)
{
    aclblasStatus_t status = CgemmArch22AllocAndCopy(buffers.alpha, alpha, sizeof(aclblasComplex));
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = CgemmArch22AllocAndCopy(buffers.a, a, aBytes);
    }
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = CgemmArch22AllocAndCopy(buffers.b, b, bBytes);
    }
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = CgemmArch22AllocAndCopy(buffers.beta, beta, sizeof(aclblasComplex));
    }
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = CgemmArch22AllocAndCopy(buffers.c, c, cBytes);
    }
    return status;
}

inline aclblasStatus_t aclblasCgemm_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb,
    const aclblasComplex* beta, aclblasComplex* c, int ldc)
{
    if (!CgemmArch22CanUseDeviceWrapper(handle, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
        return aclblasCgemm(handle, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    const int aColumns = transA == ACLBLAS_OP_N ? k : m;
    const int bColumns = transB == ACLBLAS_OP_N ? n : k;
    const size_t aBytes =
        static_cast<size_t>(std::max(1, lda)) * static_cast<size_t>(std::max(1, aColumns)) * sizeof(aclblasComplex);
    const size_t bBytes =
        static_cast<size_t>(std::max(1, ldb)) * static_cast<size_t>(std::max(1, bColumns)) * sizeof(aclblasComplex);
    const size_t cBytes =
        static_cast<size_t>(std::max(1, ldc)) * static_cast<size_t>(std::max(1, n)) * sizeof(aclblasComplex);

    CgemmArch22DeviceBuffers buffers;
    aclblasStatus_t status = CgemmArch22UploadBuffers(buffers, alpha, a, b, beta, c, aBytes, bBytes, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    const aclblasStatus_t ret = aclblasCgemm(
        handle, transA, transB, m, n, k, static_cast<const aclblasComplex*>(buffers.alpha),
        static_cast<const aclblasComplex*>(buffers.a), lda, static_cast<const aclblasComplex*>(buffers.b), ldb,
        static_cast<const aclblasComplex*>(buffers.beta), static_cast<aclblasComplex*>(buffers.c), ldc);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (aclrtMemcpy(c, cBytes, buffers.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
