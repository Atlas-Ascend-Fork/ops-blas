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
 * \brief 测试用的算子调用封装（family 级共享）。
 *
 * 封装体只依赖算子的 C 接口，与具体架构无关，因此 arch22 与 arch35 共用这一份实现，
 * 两个架构目录下的 ssbmv_npu_wrapper.h 只是各自的入口。测试参数解析（ssbmv_param.h）
 * 与 golden（ssbmv_golden.h）同样放在 family 级，理由相同。
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

/*! \brief |stride| in 64-bit: negating INT_MIN as an int is undefined behaviour. */
inline int64_t ssbmvAbsStride(int stride)
{
    const int64_t value = stride;
    return value > 0 ? value : -value;
}

inline std::unique_ptr<DeviceBuffer> ssbmvTryAllocAndCopy(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr)
        return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline aclblasStatus_t aclblasSsbmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return aclblasSsbmv(handle, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }

    const size_t aBytes = static_cast<size_t>(lda) * n * sizeof(float);
    /* 跨度按 64 位算：n=2^20、|inc|=2^12 时 (n-1)*|inc| 已超出 INT32_MAX，int 乘法会溢出。 */
    const int64_t xCount = (static_cast<int64_t>(n) - 1) * ssbmvAbsStride(incx) + 1;
    const int64_t yCount = (static_cast<int64_t>(n) - 1) * ssbmvAbsStride(incy) + 1;
    constexpr int64_t MAX_ELEMS = static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(float));
    if (xCount > MAX_ELEMS || yCount > MAX_ELEMS) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const size_t xBytes = static_cast<size_t>(xCount) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(yCount) * sizeof(float);

    auto dA = ssbmvTryAllocAndCopy(a, aBytes);
    auto dX = ssbmvTryAllocAndCopy(x, xBytes);
    auto dY = ssbmvTryAllocAndCopy(y, yBytes);

    aclblasStatus_t ret = aclblasSsbmv(
        handle, uplo, n, k, alpha, dA ? static_cast<const float*>(dA->ptr()) : nullptr, lda,
        dX ? static_cast<const float*>(dX->ptr()) : nullptr, incx, beta, dY ? static_cast<float*>(dY->ptr()) : nullptr,
        incy);

    /* 只等算子真正下发的那条流：全设备同步会把「下发到错误的流」掩盖成通过。 */
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || stream == nullptr) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    if (y != nullptr && dY) {
        dY->copyToHost(y, yBytes);
    }

    return ret;
}
