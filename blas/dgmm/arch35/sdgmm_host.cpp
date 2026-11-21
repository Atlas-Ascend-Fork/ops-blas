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
 * \file sdgmm_host.cpp
 * \brief Host-side implementation of aclblasSdgmm (arch35).
 *        Diagonal matrix-matrix multiplication: C = diag(x) * A (mode=LEFT) or
 *        C = A * diag(x) (mode=RIGHT). Column-major storage (BLAS convention).
 *        Single-path tensor API dispatch: all mode/incx combinations go through
 *        the same SIMD kernel (GetValue for strided scalar access).
 */

#include <cstdint>
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "sdgmm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

static aclblasStatus_t ValidateSdgmmParams(
    aclblasSideMode_t mode, int m, int n,
    const float* A, int lda, const float* x, int incx,
    float* C, int ldc)
{
    if (mode != ACLBLAS_SIDE_LEFT && mode != ACLBLAS_SIDE_RIGHT) {
        OP_LOGE("aclblasSdgmm", "mode must be SIDE_LEFT(141) or SIDE_RIGHT(142), got %d",
                static_cast<int>(mode));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0) {
        OP_LOGE("aclblasSdgmm", "m must be >= 0, got %d", m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE("aclblasSdgmm", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) {
        OP_LOGE("aclblasSdgmm", "lda must be >= max(1, m), got lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("aclblasSdgmm", "ldc must be >= max(1, m), got ldc=%d, m=%d", ldc, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && (x == nullptr || A == nullptr || C == nullptr)) {
        OP_LOGE("aclblasSdgmm", "A/x/C must not be nullptr when m>0 and n>0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == C && lda != ldc) {
        OP_LOGE("aclblasSdgmm", "in-place execution (A==C) requires lda==ldc, got lda=%d, ldc=%d", lda, ldc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Tiling: 2D block decomposition — colBlocks × mBlocks.
// colBlocks = min(n, aivCoreNum) splits along n (columns).
// mBlocks > 1 only when n < aivCoreNum (tall-skinny): splits along m to
// utilize idle cores. Each block handles [startCol,endCol) × [startM,endM).
static SdgmmTilingData CalSdgmmTilingData(
    uint32_t mode, uint32_t m, uint32_t n, int32_t incx,
    uint32_t lda, uint32_t ldc, uint32_t aivCoreNum)
{
    SdgmmTilingData tiling{};
    tiling.mode = mode;
    tiling.m = m;
    tiling.n = n;
    tiling.incx = incx;
    tiling.lda = lda;
    tiling.ldc = ldc;

    constexpr uint32_t ALIGN_UNIT = 32 / sizeof(float);  // = 8
    constexpr uint32_t UB_RESERVE = 256;

    // Both modes use 3 UB buffers (A + C + X). mode=R batch-copies an x segment
    // to UB; mode=L keeps an x row-segment in UB.
    constexpr uint32_t buffersPerTile = 3;
    uint32_t maxTileM = (UB_SIZE - UB_RESERVE) / (buffersPerTile * sizeof(float));
    tiling.tileM = (maxTileM / ALIGN_UNIT) * ALIGN_UNIT;

    // Column split: colBlocks = min(n, aivCoreNum)
    uint32_t colBlocks = std::min(n, aivCoreNum);
    if (colBlocks == 0) {
        colBlocks = 1;
    }
    tiling.perCoreN = n / colBlocks;
    tiling.remainder = n - tiling.perCoreN * colBlocks;

    // M-dimension split: activate when n < aivCoreNum (idle cores available)
    // and m spans multiple tiles.
    uint32_t mTiles = (tiling.tileM > 0) ? (m + tiling.tileM - 1) / tiling.tileM : 1;
    if (mTiles == 0) {
        mTiles = 1;
    }
    uint32_t mBlocks = 1;
    if (colBlocks < aivCoreNum && mTiles > 1) {
        uint32_t maxMBlocks = aivCoreNum / colBlocks;
        mBlocks = std::min(maxMBlocks, mTiles);
    }
    tiling.mBlocks = mBlocks;
    if (mBlocks == 0) {
        mBlocks = 1;
    }
    tiling.perCoreMTile = mTiles / mBlocks;
    tiling.mTileRemainder = mTiles - tiling.perCoreMTile * mBlocks;

    return tiling;
}

static aclblasStatus_t LaunchSdgmmKernel(
    aclblasSideMode_t mode, int m, int n,
    const float* A, int lda, const float* x, int incx,
    float* C, int ldc, uint32_t aivCoreNum, aclrtStream stream)
{
    uint32_t modeNorm = (mode == ACLBLAS_SIDE_LEFT) ? SDGMM_MODE_LEFT : SDGMM_MODE_RIGHT;

    SdgmmTilingData tiling = CalSdgmmTilingData(
        modeNorm, static_cast<uint32_t>(m), static_cast<uint32_t>(n), incx,
        static_cast<uint32_t>(lda), static_cast<uint32_t>(ldc), aivCoreNum);

    if (tiling.tileM == 0) {
        OP_LOGE("aclblasSdgmm",
                "tileM is 0 (UB too small for %u buffers), cannot compute",
                3u);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t colBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    if (colBlocks == 0) {
        colBlocks = 1;
    }
    uint32_t numBlocks = colBlocks * tiling.mBlocks;

    OP_LOGD("aclblasSdgmm",
            "params: mode=%d, m=%d, n=%d, incx=%d, lda=%d, ldc=%d, numBlocks=%u",
            static_cast<int>(mode), m, n, incx, lda, ldc, numBlocks);

    OP_LOGD("aclblasSdgmm", "tiling: mode=%u, perCoreN=%u, remainder=%u, mBlocks=%u, tileM=%u",
            tiling.mode, tiling.perCoreN, tiling.remainder, tiling.mBlocks, tiling.tileM);

    OP_LOGI("aclblasSdgmm",
            "launch kernel: m=%d, n=%d, incx=%d, numBlocks=%u, aivCoreNum=%u",
            m, n, incx, numBlocks, aivCoreNum);

    // The kernel handles negative incx internally by computing non-negative
    // indices (same formula as the CPU golden), so we pass the original x
    // pointer without pre-offsetting. This avoids tensor negative-index access
    // and the -incx overflow risk that the pre-offset approach had.

    sdgmm_kernel_do(reinterpret_cast<const GM_ADDR>(x),
                    reinterpret_cast<const GM_ADDR>(A),
                    reinterpret_cast<GM_ADDR>(C),
                    tiling, numBlocks, stream);
    return ACLBLAS_STATUS_SUCCESS;
}

extern "C" aclblasStatus_t aclblasSdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode,
    int m, int n, const float* A, int lda,
    const float* x, int incx, float* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSdgmm", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // Quick return: m==0 or n==0 — no computation, skip all further checks.
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateSdgmmParams(mode, m, n, A, lda, x, incx, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSdgmm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);
    return LaunchSdgmmKernel(mode, m, n, A, lda, x, incx, C, ldc, aivCoreNum, h->stream);
}
