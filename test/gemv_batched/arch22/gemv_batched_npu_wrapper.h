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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

template <typename T>
inline DeviceBuffer UploadBatchPtrArray(
    const T* hostBase, size_t stride, size_t perBatchBytes, int batchCount, std::vector<DeviceBuffer>& bufs)
{
    bufs.reserve(batchCount);
    std::vector<void*> hPtrs(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        bufs.emplace_back(perBatchBytes);
        bufs.back().copyFromHost(hostBase + static_cast<size_t>(b) * stride, perBatchBytes);
        hPtrs[b] = bufs.back().ptr();
    }
    const size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    DeviceBuffer ptrArr(ptrBytes);
    ptrArr.copyFromHost(hPtrs.data(), ptrBytes);
    return ptrArr;
}

template <typename TIn, typename TOut, typename ApiFn>
inline aclblasStatus_t GemvBatchedPtrArrayImpl(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const TIn* a, int lda,
    const TIn* x, int incx, const float* beta, TOut* y, int incy, int batchCount, ApiFn apiFn)
{
    const size_t absIncx = static_cast<size_t>(std::abs(incx));
    const size_t absIncy = static_cast<size_t>(std::abs(incy));
    const size_t aStride = static_cast<size_t>(lda) * static_cast<size_t>(n);
    const size_t xLen = static_cast<size_t>((trans == ACLBLAS_OP_N) ? n : m);
    const size_t yLen = static_cast<size_t>((trans == ACLBLAS_OP_N) ? m : n);
    const size_t xStride = (xLen - 1U) * absIncx + 1U;
    const size_t yStride = (yLen - 1U) * absIncy + 1U;
    const size_t perBatchABytes = aStride * sizeof(TIn);
    const size_t perBatchXBytes = xStride * sizeof(TIn);
    const size_t perBatchYBytes = yStride * sizeof(TOut);

    std::vector<DeviceBuffer> dABufs;
    std::vector<DeviceBuffer> dXBufs;
    std::vector<DeviceBuffer> dYBufs;
    DeviceBuffer dAPtrArr = UploadBatchPtrArray(a, aStride, perBatchABytes, batchCount, dABufs);
    DeviceBuffer dXPtrArr = UploadBatchPtrArray(x, xStride, perBatchXBytes, batchCount, dXBufs);
    DeviceBuffer dYPtrArr = UploadBatchPtrArray(y, yStride, perBatchYBytes, batchCount, dYBufs);
    const aclblasStatus_t ret = apiFn(
        handle, trans, m, n, alpha, reinterpret_cast<const TIn* const*>(dAPtrArr.ptr()), lda,
        reinterpret_cast<const TIn* const*>(dXPtrArr.ptr()), incx, beta, reinterpret_cast<TOut* const*>(dYPtrArr.ptr()),
        incy, batchCount);
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        for (int b = 0; b < batchCount; ++b) {
            dYBufs[b].copyToHost(y + static_cast<size_t>(b) * yStride, perBatchYBytes);
        }
    }
    return ret;
}

inline bool HasNullGemvArgs(
    aclblasHandle_t handle, const void* a, const void* x, const void* y, const float* alpha, const float* beta)
{
    return (handle == nullptr) || (a == nullptr) || (x == nullptr) || (y == nullptr) || (alpha == nullptr) ||
           (beta == nullptr);
}

inline bool HasInvalidGemvDims(aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, int batchCount)
{
    const bool badTrans = (trans != ACLBLAS_OP_N) && (trans != ACLBLAS_OP_T);
    return badTrans || (m <= 0) || (n <= 0) || (batchCount <= 0) || (incx == 0) || (incy == 0) || (lda < m);
}

template <typename TIn, typename TOut, typename ApiFn>
inline aclblasStatus_t GemvBatchedNpuEntry(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const TIn* a, int lda,
    const TIn* x, int incx, const float* beta, TOut* y, int incy, int batchCount, ApiFn apiFn)
{
    if (HasNullGemvArgs(handle, a, x, y, alpha, beta) || HasInvalidGemvDims(trans, m, n, lda, incx, incy, batchCount)) {
        using ConstInPtr = const TIn*;
        using OutPtr = TOut*;
        const ConstInPtr* aArr = (a == nullptr) ? nullptr : &a;
        const ConstInPtr* xArr = (x == nullptr) ? nullptr : &x;
        const OutPtr* yArr = (y == nullptr) ? nullptr : &y;
        return apiFn(handle, trans, m, n, alpha, aArr, lda, xArr, incx, beta, yArr, incy, batchCount);
    }
    return GemvBatchedPtrArrayImpl<TIn, TOut>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, apiFn);
}

inline aclblasStatus_t aclblasGemvBatchedS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    return GemvBatchedNpuEntry<float, float>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasSgemvBatched);
}

inline aclblasStatus_t aclblasGemvBatchedHSH_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, uint16_t* y, int incy, int batchCount)
{
    return GemvBatchedNpuEntry<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasHSHgemvBatched);
}

inline aclblasStatus_t aclblasGemvBatchedHSS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    return GemvBatchedNpuEntry<uint16_t, float>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasHSSgemvBatched);
}

inline aclblasStatus_t aclblasGemvBatchedTST_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, uint16_t* y, int incy, int batchCount)
{
    return GemvBatchedNpuEntry<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasTSTgemvBatched);
}

inline aclblasStatus_t aclblasGemvBatchedTSS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    return GemvBatchedNpuEntry<uint16_t, float>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasTSSgemvBatched);
}
