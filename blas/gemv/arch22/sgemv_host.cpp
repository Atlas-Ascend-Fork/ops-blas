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
 * \file sgemv_host.cpp
 * \brief Single-precision sgemv host entry for Atlas A2/A3.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "sgemv_kernel.h"
#include "sgemv_tiling_data.h"

namespace {
constexpr uint32_t SGEMMV_MAX_CORE_NUM = 40;
constexpr uint32_t SGEMMV_CUBE_MIN_DIM = 256;
constexpr uint32_t SGEMMV_CUBE_MIN_K = 64;
constexpr uint32_t SGEMMV_CUBE_MAX_K = 8192;

uint64_t AbsInc(int64_t inc) { return static_cast<uint64_t>(inc > 0 ? inc : -inc); }

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return (value + divisor - 1U) / divisor;
}

uint32_t AlignUp(uint32_t value, uint32_t align) { return CeilDiv(value, align) * align; }

uint64_t StrideSpan(uint32_t length, int64_t inc)
{
    if (length == 0U) {
        return 0U;
    }
    return static_cast<uint64_t>(length - 1U) * AbsInc(inc);
}

aclblasStatus_t ValidateSgemvParams(
    aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, const float* alpha, const float* beta,
    const float* a, const float* x, float* y)
{
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasSgemv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasSgemv", "invalid m=%d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSgemv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(lda >= std::max(1, m), OP_LOGE("aclblasSgemv", "invalid lda=%d, m=%d", lda, m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSgemv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != std::numeric_limits<int32_t>::min(), OP_LOGE("aclblasSgemv", "incx must not be INT32_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSgemv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != std::numeric_limits<int32_t>::min(), OP_LOGE("aclblasSgemv", "incy must not be INT32_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSgemv", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasSgemv", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);

    const uint32_t xLength = (trans == ACLBLAS_OP_N) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    const uint32_t yLength = (trans == ACLBLAS_OP_N) ? static_cast<uint32_t>(m) : static_cast<uint32_t>(n);
    if (m > 0 && n > 0) {
        CHECK_RET(a != nullptr, OP_LOGE("aclblasSgemv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (xLength > 0U) {
        CHECK_RET(x != nullptr, OP_LOGE("aclblasSgemv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (yLength > 0U) {
        CHECK_RET(y != nullptr, OP_LOGE("aclblasSgemv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }

    const uint64_t xSpan = StrideSpan(xLength, incx);
    const uint64_t ySpan = StrideSpan(yLength, incy);
    constexpr uint64_t maxSpan = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    CHECK_RET(xSpan <= maxSpan,
              OP_LOGE("aclblasSgemv", "incx span=%llu exceeds int32", static_cast<unsigned long long>(xSpan));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ySpan <= maxSpan,
              OP_LOGE("aclblasSgemv", "incy span=%llu exceeds int32", static_cast<unsigned long long>(ySpan));
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

SgemvPath SelectPath(aclblasOperation_t trans, int m, int n, int incx, int incy, float alpha, float beta)
{
    const bool unitY = (incy == 1);
    if (trans == ACLBLAS_OP_N) {
        if (alpha == 1.0f && beta == 0.0f && unitY && incx == 1 && static_cast<uint32_t>(m) <= SGEMMV_SMALL_M_LIMIT &&
            n >= 256) {
            return SgemvPath::SMALL_M_N;
        }
        return SgemvPath::NORMAL_N;
    }
    if (static_cast<uint32_t>(m) <= SGEMMV_T_SMALL_M_LIMIT) {
        return SgemvPath::SMALL_M_T;
    }
    return SgemvPath::NORMAL_T;
}

void CalculateParallelism(
    SgemvPath path, uint32_t outDim, int incy, uint32_t availableCores, uint32_t& numBlocks, uint32_t& outPerBlock)
{
    if (incy != 1 || availableCores <= 1U) {
        numBlocks = 1U;
        outPerBlock = outDim;
        return;
    }
    const bool transpose = path == SgemvPath::NORMAL_T || path == SgemvPath::SMALL_M_T;
    const uint32_t usefulBlocks = std::min(availableCores, CeilDiv(outDim, transpose ? 16U : 64U));
    numBlocks = std::max(1U, usefulBlocks);
    const uint32_t initialPerBlock = CeilDiv(outDim, numBlocks);
    outPerBlock = AlignUp(initialPerBlock, SGEMMV_PARTITION_ALIGN);
    numBlocks = transpose ? usefulBlocks : CeilDiv(outDim, outPerBlock);
}
SgemvTilingData MakeSgemvTiling(
    aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, float alpha, float beta,
    uint32_t availableCores)
{
    SgemvTilingData tiling{};
    const SgemvPath path = SelectPath(trans, m, n, incx, incy, alpha, beta);
    const uint32_t outDim = (path == SgemvPath::NORMAL_N || path == SgemvPath::SMALL_M_N) ? static_cast<uint32_t>(m) :
                                                                                            static_cast<uint32_t>(n);
    const uint32_t partitionDim = (path == SgemvPath::SMALL_M_N) ? static_cast<uint32_t>(n) : outDim;
    uint32_t numBlocks = 1U;
    uint32_t outPerBlock = partitionDim;
    CalculateParallelism(path, partitionDim, incy, availableCores, numBlocks, outPerBlock);

    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.outPerBlock = outPerBlock;
    tiling.numBlocks = numBlocks;
    tiling.path = static_cast<uint32_t>(path);
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    return tiling;
}

bool UseCubePath(aclblasOperation_t trans, const SgemvTilingData& tiling)
{
    return tiling.alpha == 1.0f && tiling.beta == 0.0f && tiling.incx == 1 && tiling.incy == 1 &&
           trans == ACLBLAS_OP_N && tiling.lda == tiling.m && tiling.m >= SGEMMV_CUBE_MIN_DIM &&
           tiling.n >= SGEMMV_CUBE_MIN_K && tiling.n <= SGEMMV_CUBE_MAX_K;
}

struct CubePlan {
    uint32_t outBlocks;
    uint32_t splitK;
    uint32_t kPerSlice;
};

CubePlan FindCubePlan(uint32_t cubeOutDim, uint32_t cubeInDim)
{
    CubePlan plan{};
    const uint32_t cubeOutBlocks =
        std::min<uint32_t>(SGEMMV_CUBE_MAX_BLOCKS, CeilDiv(cubeOutDim, SGEMMV_CUBE_TARGET_COLS));
    plan.outBlocks = cubeOutBlocks;
    plan.splitK = 1U;
    plan.kPerSlice = cubeInDim;
    uint32_t bestWaveNumerator = CeilDiv(cubeOutBlocks, SGEMMV_CUBE_MAX_BLOCKS);
    uint32_t bestWaves = bestWaveNumerator;
    const uint32_t minOutBlocks = CeilDiv(cubeOutDim, SGEMMV_CUBE_MAX_COLS);
    for (uint32_t outBlocks = minOutBlocks; outBlocks <= cubeOutBlocks; ++outBlocks) {
        const uint32_t colsPerBlock = AlignUp(CeilDiv(cubeOutDim, outBlocks), SGEMMV_ALIGN);
        const uint32_t nTiles = CeilDiv(colsPerBlock, SGEMMV_CUBE_TARGET_COLS);
        for (uint32_t candidate = 1U; candidate <= SGEMMV_CUBE_SPLIT_MAX; ++candidate) {
            if (candidate > 1U && cubeInDim < candidate * SGEMMV_CUBE_TARGET_COLS) {
                continue;
            }
            const uint32_t candidatePerSlice =
                AlignUp(CeilDiv(cubeInDim, candidate), static_cast<uint32_t>(SGEMMV_ALIGN * sizeof(float)));
            const uint32_t actualSplit = CeilDiv(cubeInDim, candidatePerSlice);
            const uint32_t tasks = outBlocks * actualSplit;
            const uint32_t waves = CeilDiv(tasks, SGEMMV_CUBE_MAX_BLOCKS);
            const uint32_t waveNumerator = waves * nTiles;
            const bool lowerCost = waveNumerator * plan.splitK < bestWaveNumerator * actualSplit;
            const bool lowerWavesAtEqualCost =
                waveNumerator * plan.splitK == bestWaveNumerator * actualSplit && waves < bestWaves;
            if (lowerCost || lowerWavesAtEqualCost) {
                plan.outBlocks = outBlocks;
                plan.splitK = actualSplit;
                plan.kPerSlice = candidatePerSlice;
                bestWaveNumerator = waveNumerator;
                bestWaves = waves;
            }
        }
    }
    return plan;
}

void ConfigureSquareCube(aclblasOperation_t trans, uint32_t cubeOutDim, uint32_t cubeInDim, CubePlan& plan)
{
    if (cubeOutDim == cubeInDim) {
        if (cubeOutDim >= 3840U) {
            plan.outBlocks = 4U;
            plan.splitK = 5U;
            plan.kPerSlice =
                AlignUp(CeilDiv(cubeInDim, plan.splitK), static_cast<uint32_t>(SGEMMV_ALIGN * sizeof(float)));
        } else if (cubeOutDim >= 3072U) {
            plan.outBlocks = CeilDiv(cubeOutDim, SGEMMV_CUBE_BASE_COLS);
            plan.splitK = std::max(1U, SGEMMV_CUBE_MAX_BLOCKS / plan.outBlocks);
            plan.kPerSlice =
                AlignUp(CeilDiv(cubeInDim, plan.splitK), static_cast<uint32_t>(SGEMMV_ALIGN * sizeof(float)));
        } else if (cubeOutDim >= 1600U) {
            plan.outBlocks = 3U;
            plan.splitK = 5U;
            plan.kPerSlice =
                AlignUp(CeilDiv(cubeInDim, plan.splitK), static_cast<uint32_t>(SGEMMV_ALIGN * sizeof(float)));
        } else if (cubeOutDim >= 1024U) {
            plan.outBlocks = 4U;
            plan.splitK = 5U;
            plan.kPerSlice =
                AlignUp(CeilDiv(cubeInDim, plan.splitK), static_cast<uint32_t>(SGEMMV_ALIGN * sizeof(float)));
        } else if (trans == ACLBLAS_OP_N && cubeOutDim >= 256U && cubeOutDim % SGEMMV_ALIGN != 0U) {
            plan.outBlocks = cubeOutDim < 400U ? 4U : 8U;
            plan.splitK = 1U;
            plan.kPerSlice = cubeInDim;
        }
    }
}

aclblasStatus_t LaunchCube(
    aclblasHandle_t handle, aclblasOperation_t trans, uint8_t* aPtr, uint8_t* xPtr, uint8_t* yPtr,
    SgemvTilingData& tiling, uint32_t availableCores)
{
    const uint32_t cubeOutDim = trans == ACLBLAS_OP_N ? tiling.m : tiling.n;
    const uint32_t cubeInDim = trans == ACLBLAS_OP_N ? tiling.n : tiling.m;
    CubePlan plan = FindCubePlan(cubeOutDim, cubeInDim);
    ConfigureSquareCube(trans, cubeOutDim, cubeInDim, plan);
    const uint32_t maxCols = cubeOutDim >= 3840U ? SGEMMV_CUBE_ODD_COLS :
                                                   (cubeOutDim < 3072U ? SGEMMV_CUBE_BASE_COLS : SGEMMV_CUBE_MAX_COLS);
    plan.outBlocks = std::max(plan.outBlocks, CeilDiv(cubeOutDim, maxCols));
    tiling.cubeKPerSlice = plan.kPerSlice;
    tiling.cubeSplitK = plan.splitK;
    tiling.numBlocks = plan.outBlocks * plan.splitK;
    if (plan.splitK == 1U) {
        sgemv_cube_kernel_do(aPtr, xPtr, yPtr, tiling, handle->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint64_t workspaceBytes = static_cast<uint64_t>(plan.splitK) * cubeOutDim * sizeof(float);
    CHECK_RET(workspaceBytes <= std::numeric_limits<size_t>::max(),
              OP_LOGE("aclblasSgemv", "split-K workspace exceeds size_t");
              return ACLBLAS_STATUS_ALLOC_FAILED);
    const aclblasStatus_t workspaceStatus = EnsureDefaultWorkspace(handle, static_cast<size_t>(workspaceBytes));
    if (workspaceStatus != ACLBLAS_STATUS_SUCCESS) {
        return workspaceStatus;
    }
    auto* partialPtr = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    const uint32_t reduceBlocks =
        std::max(1U, std::min(availableCores, CeilDiv(cubeOutDim, SGEMMV_CUBE_SPLIT_REDUCE_TILE)));
    sgemv_cube_kernel_do(aPtr, xPtr, partialPtr, tiling, handle->stream);
    sgemv_split_reduce_kernel_do(partialPtr, yPtr, tiling, reduceBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSgemv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    const aclblasStatus_t status = ValidateSgemvParams(trans, m, n, lda, incx, incy, alpha, beta, A, x, y);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint32_t availableCores = std::min(GetAivCoreCount(), SGEMMV_MAX_CORE_NUM);
    if (availableCores == 0U) {
        OP_LOGE("aclblasSgemv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    SgemvTilingData tiling = MakeSgemvTiling(trans, m, n, lda, incx, incy, *alpha, *beta, availableCores);
    auto* aPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(A));
    auto* xPtr = reinterpret_cast<uint8_t*>(const_cast<float*>(x));
    auto* yPtr = reinterpret_cast<uint8_t*>(y);
    if (UseCubePath(trans, tiling)) {
        return LaunchCube(handle, trans, aPtr, xPtr, yPtr, tiling, availableCores);
    }
    if (tiling.path == static_cast<uint32_t>(SgemvPath::SMALL_M_N)) {
        sgemv_init_kernel_do(yPtr, tiling, handle->stream);
        sgemv_small_m_kernel_do(aPtr, xPtr, yPtr, tiling, tiling.numBlocks, handle->stream);
    } else {
        sgemv_kernel_do(aPtr, xPtr, yPtr, tiling, tiling.numBlocks, handle->stream);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
