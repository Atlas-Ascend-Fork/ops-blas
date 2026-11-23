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
 * \file gemv_batched_host.cpp
 * \brief Host-side API for batched real GEMV (arch22 / dav-2201):
 *        y[i] = alpha * op(A[i]) * x[i] + beta * y[i].
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/gemv_batched_span_check.h"
#include "gemv_batched_tiling_data.h"
#include "gemv_batched_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasGemvBatched";

template <typename TIn, typename TOut>
aclblasStatus_t ValidateGemvBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const TIn* const Aarray[],
    int lda, const TIn* const xarray[], int incx, const float* beta, TOut* const yarray[], int incy, int batchCount)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m <= 0 || n <= 0 || lda < m || incx == 0 || incy == 0 || batchCount <= 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Aarray == nullptr || xarray == nullptr || yarray == nullptr || alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (GemvBatchedSpanExceedsInt32(trans == ACLBLAS_OP_N, 1, m, n, lda, incx, incy, batchCount)) {
        OP_LOGE(OP_TAG, "shape span exceeds kernel 32-bit index domain");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename TIn, typename TOut>
aclblasStatus_t GemvBatchedImpl(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const TIn* const Aarray[],
    int lda, const TIn* const xarray[], int incx, const float* beta, TOut* const yarray[], int incy, int batchCount,
    uint32_t dtype)
{
    const aclblasStatus_t validRet = ValidateGemvBatchedParams(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
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
    GemvBatchedTilingData tiling{};
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

    const aclError launchRet = gemv_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<TIn**>(Aarray)),
        reinterpret_cast<uint8_t*>(const_cast<TIn**>(xarray)),
        reinterpret_cast<uint8_t*>(const_cast<TOut**>(yarray)),
        tiling, usedCoreNum, handle->stream);
    if (launchRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "kernel launch failed, ACL error %d.", static_cast<int>(launchRet));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* const Aarray[],
    int lda, const float* const xarray[], int incx, const float* beta, float* const yarray[], int incy, int batchCount)
{
    return GemvBatchedImpl<float, float>(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 1U);
}

aclblasStatus_t aclblasHSHgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* const Aarray[],
    int lda, const uint16_t* const xarray[], int incx, const float* beta, uint16_t* const yarray[], int incy,
    int batchCount)
{
    return GemvBatchedImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 0U);
}

aclblasStatus_t aclblasHSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* const Aarray[],
    int lda, const uint16_t* const xarray[], int incx, const float* beta, float* const yarray[], int incy,
    int batchCount)
{
    return GemvBatchedImpl<uint16_t, float>(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 2U);
}

aclblasStatus_t aclblasTSTgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* const Aarray[],
    int lda, const uint16_t* const xarray[], int incx, const float* beta, uint16_t* const yarray[], int incy,
    int batchCount)
{
    return GemvBatchedImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 3U);
}

aclblasStatus_t aclblasTSSgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const uint16_t* const Aarray[],
    int lda, const uint16_t* const xarray[], int incx, const float* beta, float* const yarray[], int incy,
    int batchCount)
{
    return GemvBatchedImpl<uint16_t, float>(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, 4U);
}
