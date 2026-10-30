/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "sger_tiling.h"

void sger_kernel_do(const SgerTilingData& tiling, uint32_t numBlocks, void* stream);

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t QUEUE_NUM = 3;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_SIZE = 192 * 1024;

static uint32_t CalSgerBlockNum(int n, int m)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        aivCoreNum = 1;
    }

    return std::min(static_cast<uint32_t>(n), aivCoreNum);
}

static SgerTilingData CalSgerTilingData(int m, int n, int lda, int incx, int incy, uint32_t coreNum, float alpha)
{
    SgerTilingData tiling = {};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.alpha = alpha;

    if (coreNum == 0) {
        coreNum = 1;
    }

    tiling.colsPerBlock = static_cast<uint32_t>(n + coreNum - 1) / coreNum;
    tiling.useCoreNum = coreNum;

    return tiling;
}

aclblasStatus_t aclblasSger(
    aclblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy,
    float* A, int lda)
{
    if (handle == nullptr) {
        LOG_PRINT("aclblasSger: handle is nullptr\n");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (m < 0 || n < 0) {
        LOG_PRINT("aclblasSger: matrix dimensions m and n must be non-negative, got m=%d, n=%d\n", m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (incx == 0 || incx == INT_MIN) {
        LOG_PRINT("aclblasSger: incx must be non-zero and not INT_MIN, got %d\n", incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (incy == 0 || incy == INT_MIN) {
        LOG_PRINT("aclblasSger: incy must be non-zero and not INT_MIN, got %d\n", incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (alpha == nullptr) {
        LOG_PRINT("aclblasSger: alpha is nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        LOG_PRINT("aclblasSger: x is nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        LOG_PRINT("aclblasSger: y is nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        LOG_PRINT("aclblasSger: A is nullptr\n");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (lda < std::max(1, m)) {
        LOG_PRINT("aclblasSger: lda must be >= max(1, m), got lda=%d, m=%d\n", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    if (useStream == nullptr) {
        LOG_PRINT("aclblasSger: stream not initialized\n");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    uint32_t numBlocks = CalSgerBlockNum(n, m);

    float alphaHost = 0.0f;
    aclblasStatus_t readSt = ReadScalarFromHostOrDevice(alpha, useStream, alphaHost);
    if (readSt != ACLBLAS_STATUS_SUCCESS) {
        return readSt;
    }
    SgerTilingData tiling = CalSgerTilingData(m, n, lda, incx, incy, numBlocks, alphaHost);
    tiling.A = reinterpret_cast<uint64_t>(A);
    tiling.x = reinterpret_cast<uint64_t>(x);
    tiling.y = reinterpret_cast<uint64_t>(y);

    sger_kernel_do(tiling, tiling.useCoreNum, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
