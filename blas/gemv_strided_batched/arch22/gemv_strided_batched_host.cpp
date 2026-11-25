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
 * \file gemv_strided_batched_host.cpp
 * \brief Host-side API for strided-batched real GEMV (arch22 / dav-2201):
 *        y[i] = alpha * op(A[i]) * x[i] + beta * y[i].
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemv_strided_batched_tiling_data.h"
#include "gemv_strided_batched_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasGemvStridedBatched";

aclblasStatus_t ValidateGemvSbParams(
    aclblasOperation_t trans, int m, int n, int lda, int64_t strideA, int incx, int incy, int64_t stridex,
    int64_t stridey, const float* alpha, const float* beta, const void* a, const void* x, const void* y)
{
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE(OP_TAG, "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < std::max(1, m) || strideA <= 0 || stridex <= 0 || stridey <= 0) {
        OP_LOGE(OP_TAG, "invalid lda/stride");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incx == INT_MIN || incy == 0 || incy == INT_MIN) {
        OP_LOGE(OP_TAG, "invalid incx=%d or incy=%d", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr || a == nullptr || x == nullptr || y == nullptr) {
        OP_LOGE(OP_TAG, "pointer argument must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename TIn, typename TOut>
aclblasStatus_t LaunchGemvSbKernel(
    const TIn* a, const TIn* x, TOut* y, const GemvStridedBatchedTilingData& tiling, uint32_t numBlocks, void* stream)
{
    const aclError launchErr = gemv_strided_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<TIn*>(a)), reinterpret_cast<uint8_t*>(const_cast<TIn*>(x)),
        reinterpret_cast<uint8_t*>(y), tiling, numBlocks, stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "gemv_strided_batched_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename TIn, typename TOut>
aclblasStatus_t GemvStridedBatchedImpl(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const TIn* a, int lda,
    int64_t strideA, const TIn* x, int incx, int64_t stridex, const float* beta, TOut* y, int incy, int64_t stridey,
    int batchCount, uint32_t dtype)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m < 0 || n < 0 || batchCount < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t st =
        ValidateGemvSbParams(trans, m, n, lda, strideA, incx, incy, stridex, stridey, alpha, beta, a, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint32_t uBatch = static_cast<uint32_t>(batchCount);
    const uint32_t maxPerCore = (uBatch + coreNum - 1U) / coreNum;
    if (maxPerCore == 0U) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t usedCoreNum = (uBatch + maxPerCore - 1U) / maxPerCore;
    GemvStridedBatchedTilingData tiling{};
    tiling.dtype = dtype;
    tiling.trans = (trans == ACLBLAS_OP_N) ? 0U : 1U;
    tiling.alpha = *alpha;
    tiling.beta = *beta;
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.usedCoreNum = usedCoreNum;
    tiling.batchCount = uBatch;
    tiling.batchPerCore = maxPerCore;
    tiling.batchTail = uBatch - (usedCoreNum - 1U) * maxPerCore;
    tiling.lda = lda;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.strideA = strideA;
    tiling.stridex = stridex;
    tiling.stridey = stridey;

    return LaunchGemvSbKernel(a, x, y, tiling, usedCoreNum, handle->stream);
}

} // namespace

aclblasStatus_t aclblasSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda,
    int64_t strideA, const float* x, int incx, int64_t stridex, const float* beta, float* y, int incy, int64_t stridey,
    int batchCount)
{
    return GemvStridedBatchedImpl<float, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 1U);
}

aclblasStatus_t aclblasHSHgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* A, int lda,
    int64_t strideA, const uint16_t* x, int incx, int64_t stridex, const float* beta, uint16_t* y, int incy,
    int64_t stridey, int batchCount)
{
    return GemvStridedBatchedImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 0U);
}

aclblasStatus_t aclblasHSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* A, int lda,
    int64_t strideA, const uint16_t* x, int incx, int64_t stridex, const float* beta, float* y, int incy,
    int64_t stridey, int batchCount)
{
    return GemvStridedBatchedImpl<uint16_t, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 2U);
}

aclblasStatus_t aclblasTSTgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* A, int lda,
    int64_t strideA, const uint16_t* x, int incx, int64_t stridex, const float* beta, uint16_t* y, int incy,
    int64_t stridey, int batchCount)
{
    return GemvStridedBatchedImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 3U);
}

aclblasStatus_t aclblasTSSgemvStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* A, int lda,
    int64_t strideA, const uint16_t* x, int incx, int64_t stridex, const float* beta, float* y, int incy,
    int64_t stridey, int batchCount)
{
    return GemvStridedBatchedImpl<uint16_t, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount, 4U);
}
