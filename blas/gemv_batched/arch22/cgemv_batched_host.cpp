/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgemv_batched_host.cpp
 * \brief Host-side API for batched complex GEMV with pointer-array form.
 *        Aarray/xarray/yarray are device-side pointer arrays, matching cuBLAS
 *        cublasCgemvBatched semantics.
 *
 *        The arch22 legacy batched kernel (cgemv_batched_kernel_impl.h) addresses
 *        the operands as one contiguous batched buffer and its tiling data
 *        carries no alpha/beta/lda/incx/incy, so it cannot be reused for the
 *        pointer-array form: forwarding the array base address would make the
 *        kernel read Aarray[0]'s address value as the first matrix element and
 *        write y back by contiguous offset (silent memory corruption). This
 *        path therefore reads the pointer arrays back to host and dispatches
 *        each batch to the single-matrix arch22 cgemv implementation
 *        (blas/gemv/arch22/cgemv_host.cpp), which already handles
 *        alpha/beta/lda/incx/incy correctly.
 */

#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// Validate the public API arguments in the fixed order required by the task spec.
// Returns the first failing status code, or ACLBLAS_STATUS_SUCCESS.
static aclblasStatus_t ValidateCgemvBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, int batchCount,
    const aclblasComplex* const* Aarray, const aclblasComplex* const* xarray, const aclblasComplex* const* yarray,
    const aclblasComplex* alpha, const aclblasComplex* beta)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m <= 0 || n <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 0 ? m : 1)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0 || batchCount <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Aarray == nullptr || xarray == nullptr || yarray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// The pointer arrays live in device memory; copy them back so each batch can be
// handed to the single-matrix cgemv host entry point.  The copy is submitted on
// the handle stream as an async D2H and then synchronized, so it sees every
// prior write enqueued on that stream (e.g. an async H2D fill or a producer
// kernel).  A plain aclrtMemcpy would use the default stream and could race
// ahead of those writes, reading stale or null pointers.
static aclblasStatus_t ReadPtrArrayToHost(
    const void* devArray, int batchCount, std::vector<aclblasComplex*>& hostPtrs, aclrtStream stream)
{
    hostPtrs.assign(static_cast<size_t>(batchCount), nullptr);
    if (batchCount <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(aclblasComplex*);
    aclError aclRet = aclrtMemcpyAsync(
        hostPtrs.data(), ptrBytes, const_cast<void*>(devArray), ptrBytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* const Aarray[], int lda, const aclblasComplex* const xarray[], int incx,
    const aclblasComplex* beta, aclblasComplex* const yarray[], int incy, int batchCount)
{
    // Argument validation runs first so callers keep getting the precise contract
    // error for malformed input, independent of the dispatch path below.
    aclblasStatus_t validRet = ValidateCgemvBatchedParams(
        handle, trans, m, n, lda, incx, incy, batchCount, Aarray, xarray, yarray, alpha, beta);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    std::vector<aclblasComplex*> hostA;
    std::vector<aclblasComplex*> hostX;
    std::vector<aclblasComplex*> hostY;
    aclblasStatus_t ret = ReadPtrArrayToHost(Aarray, batchCount, hostA, stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = ReadPtrArrayToHost(xarray, batchCount, hostX, stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    ret = ReadPtrArrayToHost(yarray, batchCount, hostY, stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    for (int b = 0; b < batchCount; b++) {
        if (hostA[b] == nullptr || hostX[b] == nullptr || hostY[b] == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        aclblasStatus_t batchRet =
            aclblasCgemv(handle, trans, m, n, alpha, hostA[b], lda, hostX[b], incx, beta, hostY[b], incy);
        if (batchRet != ACLBLAS_STATUS_SUCCESS) {
            return batchRet;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
