/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file sgeam_host.cpp
 * @brief Host-side implementation for Sgeam operator (float)
 *        Handles parameter validation, tiling calculation, and kernel launch
 */

#include "cann_ops_blas.h"
#include "geam_host_common.h"

#include "sgeam_kernel.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include <algorithm>

namespace {

constexpr uint32_t UB_SIZE = 248 * 1024; // 248 KB
constexpr uint32_t UB_RESERVE = 256;
constexpr uint32_t ALIGN_UNIT = 8;    // 32 bytes / sizeof(float)
constexpr uint32_t SGEAM_BUFFERS = 5; // 2*bufA(TQue num=2) + 2*bufC(TQue num=2) + bufB(TBuf)
// Hardware limit for DataCopyPadExt blockCount parameter.
// When curM (rows processed per core) exceeds this, DataCopyPad blockCount
// exceeds hardware limit and behavior is undefined.
constexpr uint32_t MAX_BLOCK_COUNT = 4095;

SgeamTilingData CalSgeamTilingData(
    aclblasOperation_t transa, aclblasOperation_t transb, uint32_t m, uint32_t n, float alpha, float beta, uint32_t lda,
    uint32_t ldb, uint32_t ldc, uint32_t aivCoreNum)
{
    SgeamTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.alphaIsZero = (alpha == 0.0f) ? 1u : 0u;
    tiling.betaIsZero = (beta == 0.0f) ? 1u : 0u;
    tiling.opA = transa;
    tiling.opB = transb;

    // Calculate tileM based on UB capacity, capped to MAX_BLOCK_COUNT
    // to avoid triggering the kernel batch-loop error path when curM > 4095.
    uint32_t maxTileM = (UB_SIZE - UB_RESERVE) / (SGEAM_BUFFERS * sizeof(float));
    uint32_t tileM = (maxTileM / ALIGN_UNIT) * ALIGN_UNIT;
    tileM = std::min(tileM, MAX_BLOCK_COUNT);
    tiling.tileM = tileM;

    // NN multi-column: max columns per inner iteration that fit in UB
    // UB budget: SGEAM_BUFFERS × colsIter × tileM × sizeof(float) ≤ UB_SIZE - UB_RESERVE
    uint32_t colsIter = (UB_SIZE - UB_RESERVE) / (SGEAM_BUFFERS * static_cast<uint32_t>(sizeof(float)) * tileM);
    if (colsIter < 1u) {
        colsIter = 1u;
    }
    tiling.colsIter = colsIter;

    // 2D block decomposition: colBlocks x mBlocks
    uint32_t colBlocks = std::min(n, aivCoreNum);
    if (colBlocks == 0) {
        colBlocks = 1;
    }
    tiling.colBlocks = colBlocks;
    tiling.perCoreN = n / colBlocks;
    tiling.remainder = n % colBlocks;

    // Calculate m-blocks
    uint32_t totalMTiles = (tiling.tileM > 0) ? ((m + tiling.tileM - 1) / tiling.tileM) : 1;
    uint32_t mBlocks = 1;
    if (colBlocks < aivCoreNum && totalMTiles > 1) {
        uint32_t maxMBlocks = aivCoreNum / colBlocks;
        mBlocks = std::min(maxMBlocks, totalMTiles);
    }
    if (mBlocks == 0) // to suppress 'maybe divide by zero' warning
        __builtin_unreachable();
    tiling.mBlocks = mBlocks;
    tiling.perCoreMTile = totalMTiles / mBlocks;
    tiling.mTileRemainder = totalMTiles % mBlocks;

    return tiling;
}

aclblasStatus_t LaunchSgeamKernel(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, const float* alpha,
    const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc)
{
    // Get core count
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgeam", "Failed to get AIV core count");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    float alphaVal = *alpha;
    float betaVal = *beta;

    // Calculate tiling
    SgeamTilingData tiling = CalSgeamTilingData(
        transa, transb, static_cast<uint32_t>(m), static_cast<uint32_t>(n), alphaVal, betaVal, static_cast<uint32_t>(lda),
        static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc), aivCoreNum);

    // Calculate total blocks
    uint32_t numBlocks = tiling.colBlocks * tiling.mBlocks;
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    OP_LOGD(
        "aclblasSgeam", "Tiling: tileM=%u, colBlocks=%u, mBlocks=%u, totalBlocks=%u", tiling.tileM, tiling.colBlocks,
        tiling.mBlocks, numBlocks);

    // Convert pointers to GM_ADDR. When alpha/beta value is 0, A/B may be nullptr;
    // use C as a dummy valid GM address to avoid hardware issues with null descriptors.
    // (Kernel skips reading A/B based on alphaIsZero/betaIsZero tiling flags.)
    GM_ADDR gmC = reinterpret_cast<GM_ADDR>(C);
    GM_ADDR gmA = (A != nullptr) ? reinterpret_cast<GM_ADDR>(const_cast<float*>(A)) : gmC;
    GM_ADDR gmB = (B != nullptr) ? reinterpret_cast<GM_ADDR>(const_cast<float*>(B)) : gmC;

    // Launch kernel
    sgeam_kernel_do(gmA, gmB, gmC, tiling, numBlocks, handle->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasSgeam(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, const float* alpha,
    const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSgeam", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t status =
        ValidateGeamParams("aclblasSgeam", transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchSgeamKernel(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
