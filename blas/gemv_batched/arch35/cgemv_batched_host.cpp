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
 * \file cgemv_batched_host.cpp
 * \brief Host-side API for batched complex GEMV on arch35 (Ascend 950PR).
 *        y[b] = alpha * op(A[b]) * x[b] + beta * y[b],  op = N / T / C
 *        Aarray/xarray/yarray are device-side pointer arrays (cuBLAS form).
 */

#include <algorithm>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cgemv_batched_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

void cgemv_batched_kernel_do(
    uint8_t* Aarray, uint8_t* xarray, uint8_t* yarray, const CgemvBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream);

namespace {

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0f && v.imag == 0.0f; }

// Validate the public API arguments in the fixed order required by the task spec.
// Returns the first failing status code, or ACLBLAS_STATUS_SUCCESS.
aclblasStatus_t ValidateCgemvBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, int batchCount,
    const aclblasComplex* const* Aarray, const aclblasComplex* const* xarray, const aclblasComplex* const* yarray,
    const aclblasComplex* alpha, const aclblasComplex* beta)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgemvBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m <= 0 || n <= 0) {
        OP_LOGE("aclblasCgemvBatched", "invalid m=%d or n=%d", m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) {
        OP_LOGE("aclblasCgemvBatched", "invalid lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN || batchCount <= 0) {
        OP_LOGE("aclblasCgemvBatched", "invalid incx=%d incy=%d batchCount=%d", incx, incy, batchCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Aarray == nullptr || xarray == nullptr || yarray == nullptr) {
        OP_LOGE("aclblasCgemvBatched", "array pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclblasCgemvBatched", "alpha or beta is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE("aclblasCgemvBatched", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Output-level parallelism: spread the total outputs over all cores, and when
// the per-batch output is small align the per-core slice to whole batches so
// each core never straddles a batch boundary.
uint32_t CalcBatchPerCore(uint32_t coreNum, uint32_t batchCount, uint32_t outSizeCalc)
{
    // The caller already rejects coreNum == 0, but the divisor guard is kept
    // local so the static analyzer sees it before every division below.
    if (coreNum == 0u) {
        coreNum = 1u;
    }
    uint32_t totalWork = batchCount * outSizeCalc;
    uint32_t maxPerCore = (totalWork + coreNum - 1) / coreNum;
    if (outSizeCalc <= 64u && batchCount >= 8u) {
        uint32_t bPerCore = (batchCount + coreNum - 1u) / coreNum;
        if (bPerCore == 0u) {
            bPerCore = 1u;
        }
        uint32_t aligned = bPerCore * outSizeCalc;
        if (aligned > maxPerCore) {
            maxPerCore = (aligned > totalWork) ? totalWork : aligned;
        }
    }
    return maxPerCore;
}

CgemvBatchedTilingData CalTilingData(
    uint32_t batchCount, uint32_t m, uint32_t n, uint32_t trans, const aclblasComplex& alpha,
    const aclblasComplex& beta, int32_t lda, int32_t incx, int32_t incy, uint32_t coreNum, uint32_t usedCoreNum,
    uint32_t maxPerCore, uint32_t totalWork)
{
    CgemvBatchedTilingData td = {};
    td.trans = trans;
    td.alphaR = alpha.real;
    td.alphaI = alpha.imag;
    td.betaR = beta.real;
    td.betaI = beta.imag;
    td.m = m;
    td.n = n;
    td.lda = static_cast<uint32_t>(lda);
    td.incx = incx;
    td.incy = incy;

    td.outSize = (trans != 0) ? n : m;
    td.dotSize = (trans != 0) ? m : n;

    td.tileRed = td.dotSize;
    td.alphaIsZero = IsComplexZero(alpha) ? 1u : 0u;
    td.betaIsZero = IsComplexZero(beta) ? 1u : 0u;

    td.coreNum = coreNum;
    td.usedCoreNum = usedCoreNum;
    td.batchCount = batchCount;
    td.batchPerCore = maxPerCore;
    td.batchTail = totalWork - (usedCoreNum - 1u) * maxPerCore;

    OP_LOGD(
        "aclblasCgemvBatched",
        "tiling: trans=%u alpha=(%.3f,%.3f) beta=(%.3f,%.3f) m=%u n=%u lda=%u incx=%d incy=%d "
        "outSize=%u dotSize=%u tileRed=%u batchCount=%u coreNum=%u usedCoreNum=%u batchPerCore=%u batchTail=%u",
        td.trans, td.alphaR, td.alphaI, td.betaR, td.betaI, td.m, td.n, td.lda, td.incx, td.incy, td.outSize,
        td.dotSize, td.tileRed, td.batchCount, td.coreNum, td.usedCoreNum, td.batchPerCore, td.batchTail);
    return td;
}

// 2D column split for trans=N: row-partitioning gives every core only a thin
// band of rows, so the strided column reads are short and slow. When the output
// fits one padded row tile and the matrix is tall-and-skinny (few rows per
// core), split the reduction dimension instead: each core then reads long
// contiguous column runs and the slices are merged with GM atomics. Restricted
// to beta==0 so y needs no read-modify-write.
void ApplyColSplit2D(CgemvBatchedTilingData& tiling, uint32_t coreNum, uint32_t batchCount)
{
    tiling.colSplits = 1u;
    tiling.colPerSplit = tiling.dotSize;
    if (tiling.trans != 0u || tiling.incx != 1 || tiling.incy != 1 || tiling.betaIsZero != 1u ||
        tiling.lda != tiling.m || batchCount == 0u) {
        return;
    }
    uint32_t rowsPerCore = (tiling.usedCoreNum > 0u) ?
                               ((batchCount * tiling.outSize + tiling.usedCoreNum - 1u) / tiling.usedCoreNum) :
                               batchCount * tiling.outSize;
    uint32_t cs = coreNum / batchCount;
    uint32_t grid = cs * batchCount;
    if (cs < 2u || grid > coreNum || rowsPerCore >= 64u) {
        return;
    }
    uint32_t colPer = (tiling.dotSize + cs - 1u) / cs;
    if (colPer < 64u) {
        return;
    }
    tiling.colSplits = cs;
    tiling.colPerSplit = colPer;
    tiling.usedCoreNum = grid;
}

} // namespace

aclblasStatus_t aclblasCgemvBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* const Aarray[], int lda, const aclblasComplex* const xarray[], int incx,
    const aclblasComplex* beta, aclblasComplex* const yarray[], int incy, int batchCount)
{
    aclblasStatus_t validRet = ValidateCgemvBatchedParams(
        handle, trans, m, n, lda, incx, incy, batchCount, Aarray, xarray, yarray, alpha, beta);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasCgemvBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t transUint = (trans == ACLBLAS_OP_N) ? 0u : ((trans == ACLBLAS_OP_C) ? 2u : 1u);
    uint32_t outSizeCalc = (transUint != 0u) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    uint32_t dotSizeCalc = (transUint != 0u) ? static_cast<uint32_t>(m) : static_cast<uint32_t>(n);
    (void)dotSizeCalc;

    uint32_t batchCountU = static_cast<uint32_t>(batchCount);
    uint32_t totalWork = batchCountU * outSizeCalc;
    uint32_t maxPerCore = CalcBatchPerCore(coreNum, batchCountU, outSizeCalc);
    uint32_t usedCoreNum = (maxPerCore > 0) ? ((totalWork + maxPerCore - 1) / maxPerCore) : 0;

    CgemvBatchedTilingData tiling = CalTilingData(
        batchCountU, static_cast<uint32_t>(m), static_cast<uint32_t>(n), transUint, *alpha, *beta,
        static_cast<int32_t>(lda), incx, incy, coreNum, usedCoreNum, maxPerCore, totalWork);

    ApplyColSplit2D(tiling, coreNum, batchCountU);
    usedCoreNum = tiling.usedCoreNum;

    cgemv_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex**>(Aarray)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex**>(xarray)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex**>(yarray)), tiling, usedCoreNum, useStream);
    return ACLBLAS_STATUS_SUCCESS;
}
