/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cgemv_batched_npu_wrapper.h
 * \brief NPU wrapper that splits contiguous host complex buffers into a
 *        device-side pointer array and calls aclblasCgemvBatched. The caller
 *        passes the stream the operator must be bound to; the wrapper waits on
 *        that stream only (not the whole device) before downloading results.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

namespace cgemv_batched_npu_wrapper {

// Per-batch strides of the three operand buffers for a cuBLAS-style layout.
struct CgemvBatchedStrides {
    size_t aStride;
    size_t xStride;
    size_t yStride;
};

inline CgemvBatchedStrides ComputeStrides(aclblasOperation_t trans, int m, int n, int lda, int incx, int incy)
{
    const int absIncx = std::abs(incx);
    const int absIncy = std::abs(incy);
    const bool isTransN = (trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? n : m;
    const int yCount = isTransN ? m : n;
    CgemvBatchedStrides s;
    s.aStride = static_cast<size_t>(lda) * n;
    s.xStride = static_cast<size_t>((xCount - 1) * absIncx + 1);
    s.yStride = static_cast<size_t>((yCount - 1) * absIncy + 1);
    return s;
}

// Allocate one device buffer per batch and upload the corresponding host slice.
// Two overloads keep the read-only and writable operand groups type-safe while
// sharing the same upload loop.
inline void UploadReadOnlyBatches(
    const aclblasComplex* hostBase, size_t stride, size_t perBatchBytes, int batchCount,
    std::vector<DeviceBuffer>& bufs, std::vector<const aclblasComplex*>& hostPtrs)
{
    bufs.reserve(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        bufs.emplace_back(perBatchBytes);
        bufs.back().copyFromHost(hostBase + b * stride, perBatchBytes);
        hostPtrs[b] = static_cast<const aclblasComplex*>(bufs.back().ptr());
    }
}

inline void UploadWritableBatches(
    aclblasComplex* hostBase, size_t stride, size_t perBatchBytes, int batchCount, std::vector<DeviceBuffer>& bufs,
    std::vector<aclblasComplex*>& hostPtrs)
{
    bufs.reserve(batchCount);
    for (int b = 0; b < batchCount; ++b) {
        bufs.emplace_back(perBatchBytes);
        bufs.back().copyFromHost(hostBase + b * stride, perBatchBytes);
        hostPtrs[b] = static_cast<aclblasComplex*>(bufs.back().ptr());
    }
}

// Build the device-side pointer array that the API consumes (cuBLAS form).
inline DeviceBuffer BuildReadOnlyPtrArray(const std::vector<const aclblasComplex*>& hostPtrs, int batchCount)
{
    const size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    DeviceBuffer arr(ptrBytes);
    if (batchCount > 0) {
        arr.copyFromHost(hostPtrs.data(), ptrBytes);
    }
    return arr;
}

inline DeviceBuffer BuildWritablePtrArray(const std::vector<aclblasComplex*>& hostPtrs, int batchCount)
{
    const size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    DeviceBuffer arr(ptrBytes);
    if (batchCount > 0) {
        arr.copyFromHost(hostPtrs.data(), ptrBytes);
    }
    return arr;
}

// Download every batch output back into the contiguous host buffer.
inline void DownloadBatches(
    aclblasComplex* hostBase, size_t stride, size_t perBatchBytes, int batchCount, std::vector<DeviceBuffer>& bufs)
{
    for (int b = 0; b < batchCount; ++b) {
        bufs[b].copyToHost(hostBase + b * stride, perBatchBytes);
    }
}

} // namespace cgemv_batched_npu_wrapper

inline aclblasStatus_t aclblasCgemvBatched_npu(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy, int batchCount)
{
    using namespace cgemv_batched_npu_wrapper;

    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // For error cases with null arrays, non-positive dims, or invalid strides,
    // pass nullptr to exercise the host-side validation path directly.
    if (a == nullptr || x == nullptr || y == nullptr || m <= 0 || n <= 0 || batchCount <= 0 || incx == 0 || incy == 0 ||
        incx == INT_MIN || incy == INT_MIN) {
        return aclblasCgemvBatched(
            handle, trans, m, n, alpha, reinterpret_cast<const aclblasComplex* const*>(a), lda,
            reinterpret_cast<const aclblasComplex* const*>(x), incx, beta, reinterpret_cast<aclblasComplex* const*>(y),
            incy, batchCount);
    }

    const CgemvBatchedStrides s = ComputeStrides(trans, m, n, lda, incx, incy);
    const size_t perBatchABytes = s.aStride * sizeof(aclblasComplex);
    const size_t perBatchXBytes = s.xStride * sizeof(aclblasComplex);
    const size_t perBatchYBytes = s.yStride * sizeof(aclblasComplex);

    std::vector<DeviceBuffer> dABufs;
    std::vector<DeviceBuffer> dXBufs;
    std::vector<DeviceBuffer> dYBufs;
    std::vector<const aclblasComplex*> hAPtrs(batchCount);
    std::vector<const aclblasComplex*> hXPtrs(batchCount);
    std::vector<aclblasComplex*> hYPtrs(batchCount);
    UploadReadOnlyBatches(a, s.aStride, perBatchABytes, batchCount, dABufs, hAPtrs);
    UploadReadOnlyBatches(x, s.xStride, perBatchXBytes, batchCount, dXBufs, hXPtrs);
    UploadWritableBatches(y, s.yStride, perBatchYBytes, batchCount, dYBufs, hYPtrs);

    DeviceBuffer dAPtrArr = BuildReadOnlyPtrArray(hAPtrs, batchCount);
    DeviceBuffer dXPtrArr = BuildReadOnlyPtrArray(hXPtrs, batchCount);
    DeviceBuffer dYPtrArr = BuildWritablePtrArray(hYPtrs, batchCount);

    aclblasStatus_t ret = aclblasCgemvBatched(
        handle, trans, m, n, alpha, reinterpret_cast<const aclblasComplex* const*>(dAPtrArr.ptr()), lda,
        reinterpret_cast<const aclblasComplex* const*>(dXPtrArr.ptr()), incx, beta,
        reinterpret_cast<aclblasComplex* const*>(dYPtrArr.ptr()), incy, batchCount);
    // Wait only on the stream the operator was launched on. A device-wide sync
    // would also cover kernels enqueued on other streams, which would mask a
    // stream-binding defect in the operator under test.
    aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    DownloadBatches(y, s.yStride, perBatchYBytes, batchCount, dYBufs);
    return ret;
}
