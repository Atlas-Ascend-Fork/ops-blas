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
#include <climits>
#include <cstdint>
#include <limits>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

namespace sgemv_arch22 {

inline uint64_t AbsStride(int64_t stride)
{
    return stride < 0 ? static_cast<uint64_t>(-stride) : static_cast<uint64_t>(stride);
}

inline uint64_t StridedExtent(int count, int stride)
{
    if (count <= 0) {
        return 0U;
    }
    return (static_cast<uint64_t>(count - 1) * AbsStride(stride)) + 1U;
}

inline bool ValidSgemvLayout(aclblasOperation_t trans, int m, int n, int lda, int incx, int incy)
{
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return false;
    }
    if (m < 0 || n < 0 || lda < std::max(1, m)) {
        return false;
    }
    if (incx == 0 || incx == INT32_MIN || incy == 0 || incy == INT32_MIN) {
        return false;
    }
    return true;
}

inline bool ValidSgemvVector(int count, int stride, const float* data)
{
    if (count <= 0) {
        return true;
    }
    return data != nullptr && StridedExtent(count, stride) - 1U <= static_cast<uint64_t>(INT32_MAX);
}

inline bool ValidSgemvDeviceCall(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr) {
        return false;
    }
    if (!ValidSgemvLayout(trans, m, n, lda, incx, incy)) {
        return false;
    }
    if (alpha == nullptr || beta == nullptr) {
        return false;
    }
    if (m > 0 && n > 0 && a == nullptr) {
        return false;
    }

    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    return ValidSgemvVector(xCount, incx, x) && ValidSgemvVector(yCount, incy, y);
}

inline std::unique_ptr<DeviceBuffer> CopyIfPresent(const float* src, size_t bytes)
{
    if (src == nullptr) {
        return nullptr;
    }
    auto buffer = std::make_unique<DeviceBuffer>(std::max(bytes, sizeof(float)));
    if (bytes > 0U) {
        buffer->copyFromHost(src, bytes);
    }
    return buffer;
}

struct SgemvDeviceData {
    std::unique_ptr<DeviceBuffer> a;
    std::unique_ptr<DeviceBuffer> x;
    std::unique_ptr<DeviceBuffer> y;
    size_t yBytes = 0;
};

inline aclblasStatus_t PrepareDeviceData(
    aclblasOperation_t trans, int m, int n, const float* a, int lda, const float* x, int incx, const float* y, int incy,
    SgemvDeviceData& data)
{
    if (lda > 0 && n > 0 &&
        static_cast<uint64_t>(n) > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(lda)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const uint64_t aElements = static_cast<uint64_t>(lda) * static_cast<uint64_t>(n);
    if (aElements > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const uint64_t aBytes = aElements * sizeof(float);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const uint64_t xBytes = StridedExtent(xCount, incx) * sizeof(float);
    data.yBytes = static_cast<size_t>(StridedExtent(yCount, incy) * sizeof(float));

    if (aBytes > static_cast<uint64_t>(SIZE_MAX) || xBytes > static_cast<uint64_t>(SIZE_MAX)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    data.a = CopyIfPresent(a, static_cast<size_t>(aBytes));
    data.x = CopyIfPresent(x, static_cast<size_t>(xBytes));
    data.y = CopyIfPresent(y, data.yBytes);
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t LaunchSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy, const SgemvDeviceData& data)
{
    return aclblasSgemv(
        handle, trans, m, n, alpha, data.a ? static_cast<const float*>(data.a->ptr()) : nullptr, lda,
        data.x ? static_cast<const float*>(data.x->ptr()) : nullptr, incx, beta,
        data.y ? static_cast<float*>(data.y->ptr()) : nullptr, incy);
}

inline aclblasStatus_t aclblasSgemv_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr) {
        return aclblasSgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    if (!ValidSgemvDeviceCall(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)) {
        auto dA = CopyIfPresent(a, sizeof(float));
        auto dX = CopyIfPresent(x, sizeof(float));
        auto dY = CopyIfPresent(y, sizeof(float));
        const aclblasStatus_t status = aclblasSgemv(
            handle, trans, m, n, alpha, dA ? static_cast<const float*>(dA->ptr()) : nullptr, lda,
            dX ? static_cast<const float*>(dX->ptr()) : nullptr, incx, beta,
            dY ? static_cast<float*>(dY->ptr()) : nullptr, incy);
        if (status == ACLBLAS_STATUS_SUCCESS && dY && aclrtSynchronizeDevice() != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return status;
    }

    SgemvDeviceData data;
    if (m == 0 || n == 0) {
        const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
        data.yBytes = static_cast<size_t>(StridedExtent(yCount, incy) * sizeof(float));
        data.a = CopyIfPresent(a, sizeof(float));
        data.x = CopyIfPresent(x, sizeof(float));
        data.y = CopyIfPresent(y, data.yBytes);
    } else {
        const aclblasStatus_t prepStatus = PrepareDeviceData(trans, m, n, a, lda, x, incx, y, incy, data);
        if (prepStatus != ACLBLAS_STATUS_SUCCESS) {
            return prepStatus;
        }
    }

    const aclblasStatus_t status = LaunchSgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, data);
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && y != nullptr && data.y && data.yBytes > 0U) {
        data.y->copyToHost(y, data.yBytes);
    }
    return status;
}

struct SgemvPerfResult {
    aclblasStatus_t status = ACLBLAS_STATUS_INTERNAL_ERROR;
    double avgUs = 0.0;
    int samples = 0;
};

inline SgemvPerfResult aclblasSgemv_npu_perf(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float* a, int lda, const float* x, int incx, const float* beta, float* y, int incy, int warmup, int samples)
{
    SgemvPerfResult result;
    SgemvDeviceData data;
    result.status = PrepareDeviceData(trans, m, n, a, lda, x, incx, y, incy, data);
    if (result.status != ACLBLAS_STATUS_SUCCESS) {
        return result;
    }

    for (int i = 0; i < warmup; ++i) {
        result.status = LaunchSgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, data);
        if (result.status != ACLBLAS_STATUS_SUCCESS) {
            return result;
        }
    }

    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    if (aclrtCreateEvent(&start) != ACL_SUCCESS || aclrtCreateEvent(&stop) != ACL_SUCCESS ||
        aclrtRecordEvent(start, stream) != ACL_SUCCESS) {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (stop != nullptr) {
            aclrtDestroyEvent(stop);
        }
        result.status = ACLBLAS_STATUS_INTERNAL_ERROR;
        return result;
    }

    for (int i = 0; i < samples; ++i) {
        result.status = LaunchSgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, data);
        if (result.status != ACLBLAS_STATUS_SUCCESS) {
            break;
        }
    }
    if (result.status == ACLBLAS_STATUS_SUCCESS && aclrtRecordEvent(stop, stream) == ACL_SUCCESS &&
        aclrtSynchronizeStream(stream) == ACL_SUCCESS) {
        float elapsedMs = 0.0F;
        if (aclrtEventElapsedTime(&elapsedMs, start, stop) == ACL_SUCCESS && samples > 0) {
            result.avgUs = static_cast<double>(elapsedMs) * 1000.0 / static_cast<double>(samples);
            result.samples = samples;
        } else {
            result.status = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else if (result.status == ACLBLAS_STATUS_SUCCESS) {
        result.status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(stop);
    return result;
}

} // namespace sgemv_arch22
