/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "sgemv_tiling_data.h"

namespace {
constexpr int32_t SPLIT_N_MAX_M = 32;
constexpr int32_t SPLIT_N_MIN_N = 4096;
constexpr uint32_t SPLIT_N_COL_GRANULARITY = 256U;
constexpr uint32_t ROW_BLOCK_ALIGN = 8U;
constexpr uint32_t ROW_ALIGN_THRESHOLD = 64U;

aclblasStatus_t ValidateSgemvInputs(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
                                    const float *alpha, const float *a, int lda, const float *x, int incx,
                                    const float *beta, const float *y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const bool validOp = (trans == ACLBLAS_OP_N) || (trans == ACLBLAS_OP_T) || (trans == ACLBLAS_OP_C);
    if (!validOp) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int32_t minLda = (m > 1) ? m : 1;
    if (m < 0 || n < 0 || lda < minLda) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const bool badStride = (incx == 0) || (incx == INT32_MIN) || (incy == 0) || (incy == INT32_MIN);
    if (badStride || alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int64_t xLen = (trans == ACLBLAS_OP_N) ? static_cast<int64_t>(n) : static_cast<int64_t>(m);
    const int64_t yLen = (trans == ACLBLAS_OP_N) ? static_cast<int64_t>(m) : static_cast<int64_t>(n);
    const int64_t absIncx = (incx > 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t absIncy = (incy > 0) ? static_cast<int64_t>(incy) : -static_cast<int64_t>(incy);
    if ((xLen > 1 && (xLen - 1) * absIncx > INT32_MAX) || (yLen > 1 && (yLen - 1) * absIncy > INT32_MAX)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const bool missingPtr = (m > 0 && n > 0 && a == nullptr) || (xLen > 0 && x == nullptr) ||
                            (yLen > 0 && y == nullptr);
    if (missingPtr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t DivCeilU32(uint32_t num, uint32_t den)
{
    if (den == 0U) {
        return 1U;
    }
    return (num + den - 1U) / den;
}

void ConfigureSgemvTiling(aclblasOperation_t trans, int m, int n, int lda, int incx, int incy,
                          float alphaVal, float betaVal, uint32_t maxCores, SgemvTilingData &tiling)
{
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.alpha = alphaVal;
    tiling.beta = betaVal;

    const bool useSplitN = (trans == ACLBLAS_OP_N) && (m <= SPLIT_N_MAX_M) && (n >= SPLIT_N_MIN_N) &&
                           (incx == 1) && (incy == 1) && (alphaVal == 1.0f) && (betaVal == 0.0f);
    if (useSplitN) {
        const uint32_t colGroups = DivCeilU32(static_cast<uint32_t>(n), SPLIT_N_COL_GRANULARITY);
        const uint32_t activeCores = std::min(colGroups, maxCores);
        const uint32_t groupsPerCore = DivCeilU32(colGroups, activeCores);
        tiling.kernelMode = SGEMV_MODE_NO_TRANS_SPLIT_N;
        tiling.usedCoreNum = DivCeilU32(colGroups, groupsPerCore);
        tiling.workPerCore = groupsPerCore * SPLIT_N_COL_GRANULARITY;
        return;
    }

    if (trans == ACLBLAS_OP_N) {
        const uint32_t totalRows = static_cast<uint32_t>(m);
        tiling.kernelMode = SGEMV_MODE_NO_TRANS;
        if (incy == 1 && totalRows >= ROW_ALIGN_THRESHOLD) {
            const uint32_t rowBlocks = DivCeilU32(totalRows, ROW_BLOCK_ALIGN);
            const uint32_t activeCores = std::min(rowBlocks, maxCores);
            const uint32_t blocksPerCore = DivCeilU32(rowBlocks, activeCores);
            tiling.usedCoreNum = DivCeilU32(rowBlocks, blocksPerCore);
            tiling.workPerCore = blocksPerCore * ROW_BLOCK_ALIGN;
        } else {
            const uint32_t activeCores = std::min(totalRows, maxCores);
            const uint32_t rowsPerCore = DivCeilU32(totalRows, activeCores);
            tiling.usedCoreNum = DivCeilU32(totalRows, rowsPerCore);
            tiling.workPerCore = rowsPerCore;
        }
        return;
    }

    const uint32_t totalCols = static_cast<uint32_t>(n);
    const uint32_t activeCores = std::min(totalCols, maxCores);
    const uint32_t colsPerCore = DivCeilU32(totalCols, activeCores);
    tiling.kernelMode = SGEMV_MODE_TRANS;
    tiling.usedCoreNum = DivCeilU32(totalCols, colsPerCore);
    tiling.workPerCore = colsPerCore;
}
} // namespace

aclblasStatus_t aclblasSgemv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n,
                             const float *alpha, const float *a, int lda, const float *x, int incx,
                             const float *beta, float *y, int incy)
{
    aclblasStatus_t checkStatus = ValidateSgemvInputs(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    if (checkStatus != ACLBLAS_STATUS_SUCCESS) {
        return checkStatus;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint32_t maxCores = GetAivCoreCount();
    if (maxCores == 0U) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    SgemvTilingData tiling{};
    ConfigureSgemvTiling(trans, m, n, lda, incx, incy, *alpha, *beta, maxCores, tiling);

    if (tiling.kernelMode == SGEMV_MODE_NO_TRANS_SPLIT_N) {
        const size_t yBytes = static_cast<size_t>(m) * sizeof(float);
        aclError memRet = aclrtMemsetAsync(y, yBytes, 0, yBytes, handle->stream);
        if (memRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    aclblasSgemv_do(tiling.usedCoreNum, handle->stream,
                    reinterpret_cast<uint8_t *>(const_cast<float *>(a)),
                    reinterpret_cast<uint8_t *>(const_cast<float *>(x)),
                    reinterpret_cast<uint8_t *>(y), &tiling);

    return ACLBLAS_STATUS_SUCCESS;
}
