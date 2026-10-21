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
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

inline std::unique_ptr<DeviceBuffer> sgbmvTryAllocAndCopy(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr) {
        return nullptr;
    }
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline size_t SgbmvCalcVecBytes(int count, size_t step)
{
    if (count <= 0) {
        return sizeof(float);
    }
    return (static_cast<size_t>(count - 1) * step + 1U) * sizeof(float);
}

inline const float* SgbmvGetConstDevicePtr(const std::unique_ptr<DeviceBuffer>& buf)
{
    if (buf == nullptr) {
        return nullptr;
    }
    return static_cast<const float*>(buf->ptr());
}

inline float* SgbmvGetDevicePtr(const std::unique_ptr<DeviceBuffer>& buf)
{
    if (buf == nullptr) {
        return nullptr;
    }
    return static_cast<float*>(buf->ptr());
}

inline aclblasStatus_t aclblasSgbmv_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* a,
    int lda, const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m <= 0 || n <= 0) {
        return aclblasSgbmv(handle, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }

    const int allocN = std::max(1, n);
    const int64_t allocLda = std::max<int64_t>(lda, static_cast<int64_t>(kl) + static_cast<int64_t>(ku) + 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t absIncx = static_cast<size_t>(std::abs(incx));
    const size_t absIncy = static_cast<size_t>(std::abs(incy));

    const size_t aBytes = static_cast<size_t>(allocLda) * allocN * sizeof(float);
    const size_t xBytes = SgbmvCalcVecBytes(xCount, absIncx);
    const size_t yBytes = SgbmvCalcVecBytes(yCount, absIncy);

    auto dA = sgbmvTryAllocAndCopy(a, aBytes);
    auto dX = sgbmvTryAllocAndCopy(x, xBytes);
    auto dY = sgbmvTryAllocAndCopy(y, yBytes);

    aclblasStatus_t ret = aclblasSgbmv(
        handle, trans, m, n, kl, ku, alpha, SgbmvGetConstDevicePtr(dA), lda, SgbmvGetConstDevicePtr(dX), incx, beta,
        SgbmvGetDevicePtr(dY), incy);

    aclrtStream stream = nullptr;
    if (handle != nullptr) {
        aclblasGetStream(handle, &stream);
    }
    aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (y != nullptr && dY != nullptr) {
        dY->copyToHost(y, yBytes);
    }

    return ret;
}
