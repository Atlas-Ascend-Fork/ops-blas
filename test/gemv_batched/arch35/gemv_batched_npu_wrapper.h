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

template <typename T_IN, typename T_OUT, typename ApiFn>
inline aclblasStatus_t GemvBatchedPtrArrayImpl(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
    const float* alpha, const T_IN* a, int lda,
    const T_IN* x, int incx, const float* beta, T_OUT* y, int incy,
    int batchCount, ApiFn apiFn)
{
    const int absIncx = std::abs(incx);
    const int absIncy = std::abs(incy);
    const size_t aStride = (size_t)lda * n;
    const size_t xStride = (trans == ACLBLAS_OP_N)
        ? (size_t)((n - 1) * absIncx + 1) : (size_t)((m - 1) * absIncx + 1);
    const size_t yStride = (trans == ACLBLAS_OP_N)
        ? (size_t)((m - 1) * absIncy + 1) : (size_t)((n - 1) * absIncy + 1);
    const size_t perBatchABytes = aStride * sizeof(T_IN);
    const size_t perBatchXBytes = xStride * sizeof(T_IN);
    const size_t perBatchYBytes = yStride * sizeof(T_OUT);
    std::vector<DeviceBuffer> dABufs, dXBufs, dYBufs;
    dABufs.reserve(batchCount); dXBufs.reserve(batchCount); dYBufs.reserve(batchCount);
    std::vector<const T_IN*> hAPtrs(batchCount);
    std::vector<const T_IN*> hXPtrs(batchCount);
    std::vector<T_OUT*> hYPtrs(batchCount);
    for (int b = 0; b < batchCount; b++) {
        dABufs.emplace_back(perBatchABytes);
        dXBufs.emplace_back(perBatchXBytes);
        dYBufs.emplace_back(perBatchYBytes);
        dABufs.back().copyFromHost(a + b * aStride, perBatchABytes);
        dXBufs.back().copyFromHost(x + b * xStride, perBatchXBytes);
        dYBufs.back().copyFromHost(y + b * yStride, perBatchYBytes);
        hAPtrs[b] = static_cast<const T_IN*>(dABufs.back().ptr());
        hXPtrs[b] = static_cast<const T_IN*>(dXBufs.back().ptr());
        hYPtrs[b] = static_cast<T_OUT*>(dYBufs.back().ptr());
    }
    const size_t ptrBytes = (size_t)batchCount * sizeof(void*);
    DeviceBuffer dAPtrArr(ptrBytes), dXPtrArr(ptrBytes), dYPtrArr(ptrBytes);
    dAPtrArr.copyFromHost(hAPtrs.data(), ptrBytes);
    dXPtrArr.copyFromHost(hXPtrs.data(), ptrBytes);
    dYPtrArr.copyFromHost(hYPtrs.data(), ptrBytes);
    aclblasStatus_t ret = apiFn(handle, trans, m, n, alpha,
        reinterpret_cast<const T_IN *const *>(dAPtrArr.ptr()), lda,
        reinterpret_cast<const T_IN *const *>(dXPtrArr.ptr()), incx,
        beta,
        reinterpret_cast<T_OUT *const *>(dYPtrArr.ptr()), incy, batchCount);
    aclrtSynchronizeDevice();
    for (int b = 0; b < batchCount; b++)
        dYBufs[b].copyToHost(y + b * yStride, perBatchYBytes);
    return ret;
}

// ---- dtype 1 (S): float in, float out ----
inline aclblasStatus_t aclblasGemvBatchedS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m <= 0 || n <= 0 || batchCount <= 0)
        return aclblasSgemvBatched(handle, trans, m, n, alpha, nullptr, lda, nullptr, incx, beta, nullptr, incy, batchCount);
    return GemvBatchedPtrArrayImpl<float, float>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasSgemvBatched);
}

// ---- dtype 0 (HSH): uint16_t in, uint16_t out ----
inline aclblasStatus_t aclblasGemvBatchedHSH_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, uint16_t* y, int incy, int batchCount)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m <= 0 || n <= 0 || batchCount <= 0)
        return aclblasHSHgemvBatched(handle, trans, m, n, alpha, nullptr, lda, nullptr, incx, beta, nullptr, incy, batchCount);
    return GemvBatchedPtrArrayImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasHSHgemvBatched);
}

// ---- dtype 2 (HSS): uint16_t in, float out ----
inline aclblasStatus_t aclblasGemvBatchedHSS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m <= 0 || n <= 0 || batchCount <= 0)
        return aclblasHSSgemvBatched(handle, trans, m, n, alpha, nullptr, lda, nullptr, incx, beta, nullptr, incy, batchCount);
    return GemvBatchedPtrArrayImpl<uint16_t, float>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasHSSgemvBatched);
}

// ---- dtype 3 (TST): uint16_t in, uint16_t out (bf16) ----
inline aclblasStatus_t aclblasGemvBatchedTST_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, uint16_t* y, int incy, int batchCount)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m <= 0 || n <= 0 || batchCount <= 0)
        return aclblasTSTgemvBatched(handle, trans, m, n, alpha, nullptr, lda, nullptr, incx, beta, nullptr, incy, batchCount);
    return GemvBatchedPtrArrayImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasTSTgemvBatched);
}

// ---- dtype 4 (TSS): uint16_t in, float out (bf16) ----
inline aclblasStatus_t aclblasGemvBatchedTSS_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* a, int lda,
    const uint16_t* x, int incx, const float* beta, float* y, int incy, int batchCount)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (a == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m <= 0 || n <= 0 || batchCount <= 0)
        return aclblasTSSgemvBatched(handle, trans, m, n, alpha, nullptr, lda, nullptr, incx, beta, nullptr, incy, batchCount);
    return GemvBatchedPtrArrayImpl<uint16_t, float>(
        handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, batchCount, aclblasTSSgemvBatched);
}
