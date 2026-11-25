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
 * \brief Host-side implementation of aclblasSdgmm (arch22 / dav-2201).
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "sdgmm_kernel.h"

namespace {

constexpr const char* OP_TAG = "aclblasSdgmm";

aclblasStatus_t ValidateSdgmmDimsAndMode(aclblasSideMode_t mode, int m, int n)
{
    if (mode != ACLBLAS_SIDE_LEFT && mode != ACLBLAS_SIDE_RIGHT) {
        OP_LOGE(OP_TAG, "mode must be SIDE_LEFT(141) or SIDE_RIGHT(142), got %d", static_cast<int>(mode));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0) {
        OP_LOGE(OP_TAG, "m and n must be >= 0, got m=%d, n=%d", m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateSdgmmBodyParams(
    aclblasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, const float* C, int ldc)
{
    if (incx == INT_MIN || incx == 0) {
        OP_LOGE(OP_TAG, "incx must not be 0 or INT_MIN");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m) || ldc < std::max(1, m)) {
        OP_LOGE(OP_TAG, "lda and ldc must be >= max(1, m), got lda=%d ldc=%d m=%d", lda, ldc, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr || A == nullptr || C == nullptr) {
        OP_LOGE(OP_TAG, "A/x/C must not be nullptr when m>0 and n>0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == C && lda != ldc) {
        OP_LOGE(OP_TAG, "in-place execution (A==C) requires lda==ldc, got lda=%d ldc=%d", lda, ldc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int64_t aSpan = static_cast<int64_t>(lda) * static_cast<int64_t>(n);
    const int64_t cSpan = static_cast<int64_t>(ldc) * static_cast<int64_t>(n);
    const int64_t xLen = (mode == ACLBLAS_SIDE_LEFT) ? static_cast<int64_t>(m) : static_cast<int64_t>(n);
    const int64_t absIncx = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t xSpan = (xLen - 1) * absIncx + 1;
    if (aSpan > static_cast<int64_t>(UINT32_MAX) || cSpan > static_cast<int64_t>(UINT32_MAX) ||
        xSpan > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(OP_TAG, "matrix or vector span exceeds uint32 addressable index domain");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

SdgmmTilingData CalSdgmmTilingData(
    uint32_t mode, uint32_t m, uint32_t n, int32_t incx, uint32_t lda, uint32_t ldc, uint32_t aivCoreNum)
{
    SdgmmTilingData tiling{};
    tiling.mode = mode;
    tiling.m = m;
    tiling.n = n;
    tiling.incx = incx;
    tiling.lda = lda;
    tiling.ldc = ldc;
    tiling.tileM = SDGMM_ARCH22_TILE_M;

    const uint32_t colBlocks = std::max(1U, std::min(n, aivCoreNum));
    if (colBlocks == 0U) {
        return tiling;
    }
    tiling.colBlocks = colBlocks;
    tiling.perCoreN = n / colBlocks;
    tiling.remainder = n % colBlocks;

    const uint32_t mTiles = std::max(1U, CeilDiv<uint32_t>(m, tiling.tileM));
    uint32_t mBlocks = 1U;
    if (colBlocks < aivCoreNum && mTiles > 1U) {
        mBlocks = std::min(aivCoreNum / colBlocks, mTiles);
    }
    if (mBlocks == 0U) {
        return tiling;
    }
    tiling.mBlocks = mBlocks;
    tiling.perCoreMTile = mTiles / mBlocks;
    tiling.mTileRemainder = mTiles % mBlocks;
    return tiling;
}

aclblasStatus_t LaunchSdgmmKernel(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx,
    float* C, int ldc)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint32_t modeNorm = (mode == ACLBLAS_SIDE_LEFT) ? SDGMM_ARCH22_MODE_LEFT : SDGMM_ARCH22_MODE_RIGHT;
    const SdgmmTilingData tiling = CalSdgmmTilingData(
        modeNorm, static_cast<uint32_t>(m), static_cast<uint32_t>(n), incx, static_cast<uint32_t>(lda),
        static_cast<uint32_t>(ldc), aivCoreNum);
    const uint32_t numBlocks = tiling.colBlocks * tiling.mBlocks;
    OP_LOGD(
        OP_TAG, "tiling: mode=%u m=%u n=%u colBlocks=%u mBlocks=%u tileM=%u", tiling.mode, tiling.m, tiling.n,
        tiling.colBlocks, tiling.mBlocks, tiling.tileM);
    OP_LOGI(OP_TAG, "launching kernel: numBlocks=%u, aivCoreNum=%u", numBlocks, aivCoreNum);
    const aclError launchErr = sdgmm_kernel_do(
        reinterpret_cast<const uint8_t*>(x), reinterpret_cast<const uint8_t*>(A), reinterpret_cast<uint8_t*>(C), tiling,
        numBlocks, handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "sdgmm_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasSdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx,
    float* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    aclblasStatus_t st = ValidateSdgmmDimsAndMode(mode, m, n);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    st = ValidateSdgmmBodyParams(mode, m, n, A, lda, x, incx, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return LaunchSdgmmKernel(handle, mode, m, n, A, lda, x, incx, C, ldc);
}
