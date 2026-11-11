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

inline std::unique_ptr<DeviceBuffer> chbmvTryAllocAndCopy(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr) {
        return nullptr;
    }
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline aclblasStatus_t aclblasChbmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta,
    aclblasComplex* y, int incy)
{
    if (n <= 0 || handle == nullptr || uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER ||
        k < 0 || lda < k + 1 || incx == 0 || incy == 0 ||
        alpha == nullptr || beta == nullptr || a == nullptr || x == nullptr || y == nullptr) {
        return aclblasChbmv(handle, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }

    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t xBytes = static_cast<size_t>((n - 1) * std::abs(incx) + 1) * sizeof(aclblasComplex);
    const size_t yBytes = static_cast<size_t>((n - 1) * std::abs(incy) + 1) * sizeof(aclblasComplex);

    auto dA = chbmvTryAllocAndCopy(a, aBytes);
    auto dX = chbmvTryAllocAndCopy(x, xBytes);
    auto dY = chbmvTryAllocAndCopy(y, yBytes);

    aclblasStatus_t ret = aclblasChbmv(
        handle, uplo, n, k, alpha, dA ? static_cast<const aclblasComplex*>(dA->ptr()) : nullptr, lda,
        dX ? static_cast<const aclblasComplex*>(dX->ptr()) : nullptr, incx, beta,
        dY ? static_cast<aclblasComplex*>(dY->ptr()) : nullptr, incy);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (y != nullptr && dY) {
        dY->copyToHost(y, yBytes);
    }

    return ret;
}
