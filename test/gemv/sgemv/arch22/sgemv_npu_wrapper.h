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
#include <cmath>
#include <cstdint>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

namespace sgemv_arch22_detail {

inline std::unique_ptr<DeviceBuffer> AllocSgemvDevBuf(const void* hostData, size_t byteCount)
{
    if (hostData == nullptr || byteCount == 0U) {
        return nullptr;
    }
    auto devBuf = std::make_unique<DeviceBuffer>(byteCount);
    devBuf->copyFromHost(hostData, byteCount);
    return devBuf;
}

inline bool IsSgemvValidPositiveCall(
    aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda, const float* x, int incx,
    const float* beta, const float* y, int incy)
{
    const bool validTrans = (trans == ACLBLAS_OP_N) || (trans == ACLBLAS_OP_T) || (trans == ACLBLAS_OP_C);
    if (!validTrans || m <= 0 || n <= 0 || lda < m) {
        return false;
    }
    if (incx == 0 || incx == INT32_MIN || incy == 0 || incy == INT32_MIN) {
        return false;
    }
    if (alpha == nullptr || beta == nullptr || a == nullptr || x == nullptr || y == nullptr) {
        return false;
    }
    return true;
}

inline aclError CreateSgemvTimelineEvent(aclrtEvent* ev)
{
    aclError ret = aclrtCreateEventExWithFlag(ev, ACL_EVENT_TIME_LINE);
    if (ret != ACL_SUCCESS) {
        ret = aclrtCreateEvent(ev);
    }
    return ret;
}

inline aclblasStatus_t RunSgemvTimedLoop(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float* dA, int lda, const float* dX, int incx, const float* beta, float* dY, int incy, double* avgUsOut)
{
    constexpr int warmupCount = 5;
    const int timedCount = (static_cast<int64_t>(m) * static_cast<int64_t>(n) >= 4194304LL) ? 20 : 50;
    for (int i = 0; i < warmupCount; ++i) {
        aclblasStatus_t st = aclblasSgemv(handle, trans, m, n, alpha, dA, lda, dX, incx, beta, dY, incy);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclrtEvent startEv = nullptr;
    aclrtEvent stopEv = nullptr;
    if (CreateSgemvTimelineEvent(&startEv) != ACL_SUCCESS || CreateSgemvTimelineEvent(&stopEv) != ACL_SUCCESS) {
        if (startEv != nullptr) {
            (void)aclrtDestroyEvent(startEv);
        }
        if (stopEv != nullptr) {
            (void)aclrtDestroyEvent(stopEv);
        }
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    (void)aclrtRecordEvent(startEv, stream);
    for (int i = 0; i < timedCount; ++i) {
        (void)aclblasSgemv(handle, trans, m, n, alpha, dA, lda, dX, incx, beta, dY, incy);
    }
    (void)aclrtRecordEvent(stopEv, stream);
    aclError syncRet = aclrtSynchronizeStream(stream);
    float elapsedMs = 0.0f;
    aclError timeRet = aclrtEventElapsedTime(&elapsedMs, startEv, stopEv);
    (void)aclrtDestroyEvent(startEv);
    (void)aclrtDestroyEvent(stopEv);
    if (syncRet != ACL_SUCCESS || timeRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (avgUsOut != nullptr && timedCount > 0) {
        *avgUsOut = (static_cast<double>(elapsedMs) * 1000.0) / static_cast<double>(timedCount);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace sgemv_arch22_detail

inline aclblasStatus_t aclblasSgemv_npu_perf(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float* a, int lda, const float* x, int incx, const float* beta, float* y, int incy, double* avgUsOut)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!sgemv_arch22_detail::IsSgemvValidPositiveCall(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)) {
        auto dDummyA = sgemv_arch22_detail::AllocSgemvDevBuf(a, sizeof(float));
        auto dDummyX = sgemv_arch22_detail::AllocSgemvDevBuf(x, sizeof(float));
        auto dDummyY = sgemv_arch22_detail::AllocSgemvDevBuf(y, sizeof(float));
        return aclblasSgemv(
            handle, trans, m, n, alpha, dDummyA ? static_cast<const float*>(dDummyA->ptr()) : nullptr, lda,
            dDummyX ? static_cast<const float*>(dDummyX->ptr()) : nullptr, incx, beta,
            dDummyY ? static_cast<float*>(dDummyY->ptr()) : nullptr, incy);
    }

    const int64_t xLen = (trans == ACLBLAS_OP_N) ? static_cast<int64_t>(n) : static_cast<int64_t>(m);
    const int64_t yLen = (trans == ACLBLAS_OP_N) ? static_cast<int64_t>(m) : static_cast<int64_t>(n);
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float);
    const size_t xBytes = static_cast<size_t>((xLen - 1) * std::abs(static_cast<int64_t>(incx)) + 1) * sizeof(float);
    const size_t yBytes = static_cast<size_t>((yLen - 1) * std::abs(static_cast<int64_t>(incy)) + 1) * sizeof(float);

    auto devA = sgemv_arch22_detail::AllocSgemvDevBuf(a, aBytes);
    auto devX = sgemv_arch22_detail::AllocSgemvDevBuf(x, xBytes);
    auto devY = sgemv_arch22_detail::AllocSgemvDevBuf(y, yBytes);
    const float* ptrA = devA ? static_cast<const float*>(devA->ptr()) : nullptr;
    const float* ptrX = devX ? static_cast<const float*>(devX->ptr()) : nullptr;
    float* ptrY = devY ? static_cast<float*>(devY->ptr()) : nullptr;

    aclblasStatus_t status = aclblasSgemv(handle, trans, m, n, alpha, ptrA, lda, ptrX, incx, beta, ptrY, incy);
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && y != nullptr && devY != nullptr) {
        devY->copyToHost(y, yBytes);
    }
    if (status == ACLBLAS_STATUS_SUCCESS && avgUsOut != nullptr && stream != nullptr) {
        return sgemv_arch22_detail::RunSgemvTimedLoop(
            handle, stream, trans, m, n, alpha, ptrA, lda, ptrX, incx, beta, ptrY, incy, avgUsOut);
    }
    return status;
}

inline aclblasStatus_t aclblasSgemv_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    return aclblasSgemv_npu_perf(handle, nullptr, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, nullptr);
}
