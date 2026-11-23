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

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclError gemvSbCopyToDevice(void** devPtr, const void* hostPtr, size_t bytes)
{
    aclError ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    const aclError copyRet = aclrtMemcpy(*devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != ACL_SUCCESS) {
        aclrtFree(*devPtr);
        *devPtr = nullptr;
        return copyRet;
    }
    return ACL_SUCCESS;
}

inline void gemvSbCleanupDevice(void* dA, void* dX, void* dY)
{
    if (dA != nullptr) {
        aclrtFree(dA);
    }
    if (dX != nullptr) {
        aclrtFree(dX);
    }
    if (dY != nullptr) {
        aclrtFree(dY);
    }
}

template <typename TIn, typename TOut, typename ApiFn>
inline aclblasStatus_t GemvStridedBatchedNpuImpl(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const TIn* a, int lda,
    int64_t strideA, const TIn* x, int incx, int64_t stridex, const float* beta, TOut* y, int incy, int64_t stridey,
    int batchCount, ApiFn apiFn)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m <= 0 || n <= 0 || batchCount <= 0) {
        return apiFn(
            handle, trans, m, n, alpha, static_cast<const TIn*>(nullptr), lda, strideA,
            static_cast<const TIn*>(nullptr), incx, stridex, beta, static_cast<TOut*>(nullptr), incy, stridey,
            batchCount);
    }
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr || strideA <= 0 ||
        stridex <= 0 || stridey <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const size_t aBytes = static_cast<size_t>(batchCount) * static_cast<size_t>(strideA) * sizeof(TIn);
    const size_t xBytes = static_cast<size_t>(batchCount) * static_cast<size_t>(stridex) * sizeof(TIn);
    const size_t yBytes = static_cast<size_t>(batchCount) * static_cast<size_t>(stridey) * sizeof(TOut);
    void* dA = nullptr;
    void* dX = nullptr;
    void* dY = nullptr;
    if (gemvSbCopyToDevice(&dA, a, aBytes) != ACL_SUCCESS || gemvSbCopyToDevice(&dX, x, xBytes) != ACL_SUCCESS ||
        gemvSbCopyToDevice(&dY, y, yBytes) != ACL_SUCCESS) {
        gemvSbCleanupDevice(dA, dX, dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const aclblasStatus_t ret = apiFn(
        handle, trans, m, n, alpha, static_cast<const TIn*>(dA), lda, strideA, static_cast<const TIn*>(dX), incx,
        stridex, beta, static_cast<TOut*>(dY), incy, stridey, batchCount);
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        gemvSbCleanupDevice(dA, dX, dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ret == ACLBLAS_STATUS_SUCCESS && aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        gemvSbCleanupDevice(dA, dX, dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    gemvSbCleanupDevice(dA, dX, dY);
    return ret;
}

inline aclblasStatus_t aclblasSgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    int64_t strideA, const float* x, int incx, int64_t stridex, const float* beta, float* y, int incy, int64_t stridey,
    int batchCount)
{
    return GemvStridedBatchedNpuImpl<float, float>(
        handle, trans, m, n, alpha, a, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount,
        aclblasSgemvStridedBatched);
}

inline aclblasStatus_t aclblasHSHgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    int64_t strideA, const uint16_t* x, int incx, int64_t stridex, const float* beta, uint16_t* y, int incy,
    int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, a, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount,
        aclblasHSHgemvStridedBatched);
}

inline aclblasStatus_t aclblasHSSgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    int64_t strideA, const uint16_t* x, int incx, int64_t stridex, const float* beta, float* y, int incy,
    int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<uint16_t, float>(
        handle, trans, m, n, alpha, a, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount,
        aclblasHSSgemvStridedBatched);
}

inline aclblasStatus_t aclblasTSTgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    int64_t strideA, const uint16_t* x, int incx, int64_t stridex, const float* beta, uint16_t* y, int incy,
    int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, a, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount,
        aclblasTSTgemvStridedBatched);
}

inline aclblasStatus_t aclblasTSSgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    int64_t strideA, const uint16_t* x, int incx, int64_t stridex, const float* beta, float* y, int incy,
    int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<uint16_t, float>(
        handle, trans, m, n, alpha, a, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount,
        aclblasTSSgemvStridedBatched);
}
