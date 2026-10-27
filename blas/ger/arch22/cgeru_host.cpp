/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgeru_host.cpp
 * \brief Host side implementation for cgeru (arch22): parameter validation,
 *        column-split tiling and asynchronous kernel launch.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "cann_ops_blas.h"
#include "cgeru_tiling_data.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void cgeru_kernel_do(
    const uint8_t* x, const uint8_t* y, uint8_t* A, const CgeruTilingData& tiling, uint32_t numBlocks, void* stream);

namespace {

static_assert(sizeof(aclblasComplex) == 2 * sizeof(float), "aclblasComplex must contain two FP32 values");

uint64_t AbsStride(int64_t stride) { return static_cast<uint64_t>(stride < 0 ? -stride : stride); }

bool IsComplexVectorAddressable(uint32_t length, int64_t stride)
{
    if (length == 0) {
        return true;
    }
    const uint64_t absStride = AbsStride(stride);
    const uint64_t lastComplex = static_cast<uint64_t>(length - 1) * absStride;
    constexpr uint64_t maxFloatIndex = std::numeric_limits<size_t>::max() / sizeof(float);
    return lastComplex <= (maxFloatIndex - 1) / 2;
}

bool IsComplexMatrixAddressable(uint32_t m, uint32_t n, uint32_t lda)
{
    if (m == 0 || n == 0) {
        return true;
    }
    const uint64_t lastComplex =
        static_cast<uint64_t>(n - 1) * static_cast<uint64_t>(lda) + static_cast<uint64_t>(m - 1);
    constexpr uint64_t maxFloatIndex = std::numeric_limits<size_t>::max() / sizeof(float);
    return lastComplex <= (maxFloatIndex - 1) / 2;
}

aclblasStatus_t ValidateCgeruShape(int m, int n, const aclblasComplex* alpha, int incx, int incy, int lda)
{
    if (m < 0) {
        OP_LOGE("aclblasCgeru", "m must be >= 0, got %d", m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE("aclblasCgeru", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasCgeru", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // 与 950PR 合入版测试契约一致：只拒绝 0；INT_MIN 作负步长由寻址范围校验兜底
    if (incx == 0) {
        OP_LOGE("aclblasCgeru", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasCgeru", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // A is column-major: lda is the physical column stride and must cover m rows.
    if (lda < std::max(1, m)) {
        OP_LOGE("aclblasCgeru", "lda must be >= max(1,m), got lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateCgeruAddressRange(int m, int n, int lda, int incx, int incy)
{
    if (!IsComplexVectorAddressable(static_cast<uint32_t>(m), incx)) {
        OP_LOGE("aclblasCgeru", "x address range overflows size_t");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsComplexVectorAddressable(static_cast<uint32_t>(n), incy)) {
        OP_LOGE("aclblasCgeru", "y address range overflows size_t");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsComplexMatrixAddressable(static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(lda))) {
        OP_LOGE("aclblasCgeru", "A address range overflows size_t");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

CgeruTilingData CalCgeruTilingData(
    int m, int n, int lda, int incx, int incy, const aclblasComplex& alpha, uint32_t aivCoreNum)
{
    CgeruTilingData tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.alphaReal = alpha.real;
    tiling.alphaImag = alpha.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.xStart = (incx < 0) ? static_cast<uint64_t>(m - 1) * AbsStride(incx) : 0;
    tiling.yStart = (incy < 0) ? static_cast<uint64_t>(n - 1) * AbsStride(incy) : 0;

    // Even column split across AIV cores (kernel derives colStart/colCount).
    const uint32_t cores = std::min(static_cast<uint32_t>(n), aivCoreNum);
    tiling.colsPerCore = (cores == 0U) ? 0U : ((n + cores - 1U) / cores);
    return tiling;
}

aclblasStatus_t LaunchCgeru(
    aclblasHandle_t handle, int m, int n, const aclblasComplex& alphaValue, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgeru", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const CgeruTilingData tiling = CalCgeruTilingData(m, n, lda, incx, incy, alphaValue, aivCoreNum);
    const uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    OP_LOGD(
        "aclblasCgeru", "tiling: m=%u n=%u lda=%u incx=%lld incy=%lld colsPerCore=%u numBlocks=%u", tiling.m, tiling.n,
        tiling.lda, static_cast<long long>(tiling.incx), static_cast<long long>(tiling.incy), tiling.colsPerCore,
        numBlocks);

    // All paths use per-core UB scratch only; no GM workspace is needed.
    cgeru_kernel_do(
        reinterpret_cast<const uint8_t*>(x), reinterpret_cast<const uint8_t*>(y), reinterpret_cast<uint8_t*>(A), tiling,
        numBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCgeru(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgeru", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t status = ValidateCgeruShape(m, n, alpha, incx, incy, lda);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasComplex alphaValue = *alpha;
    if (alphaValue.real == 0.0f && alphaValue.imag == 0.0f) {
        // Netlib cgeru quick return: alpha == (0,0) leaves A untouched.
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (x == nullptr) {
        OP_LOGE("aclblasCgeru", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasCgeru", "y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasCgeru", "A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    status = ValidateCgeruAddressRange(m, n, lda, incx, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    if (handle->stream == nullptr) {
        OP_LOGE("aclblasCgeru", "stream not initialized");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    return LaunchCgeru(handle, m, n, alphaValue, x, incx, y, incy, A, lda);
}
