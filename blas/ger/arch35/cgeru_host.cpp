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
#include <cstddef>
#include <cstdint>
#include <limits>

#include "cann_ops_blas.h"
#include "cgeru_tiling_data.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void cgeru_arch35_kernel_do(
    const uint8_t* x, const uint8_t* y, uint8_t* A, const CgeruTilingData& tiling, uint32_t numBlocks, void* stream);

namespace {

static_assert(sizeof(aclblasComplex) == 2 * sizeof(float), "aclblasComplex must contain two FP32 values");

uint64_t AbsStride(int stride)
{
    const int64_t wideStride = static_cast<int64_t>(stride);
    return static_cast<uint64_t>(wideStride < 0 ? -wideStride : wideStride);
}

bool IsComplexVectorAddressable(uint32_t length, int stride)
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
    if (incx == 0) {
        OP_LOGE("aclblasCgeru", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasCgeru", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // INT_MIN 步长跨度校验与 arch22 ValidateCgeruScalars 对齐，同在标量校验阶段拦截：
    // |INT_MIN| = 2^31 的步长仅在向量长度 <=1（零维或单元素）时可寻址，长度 >1 时在此
    // 统一拒绝，保证 alpha==0 与零维 quick return 路径上双架构状态码一致（issue #1031）。
    if (incx == std::numeric_limits<int>::min() && m > 1) {
        OP_LOGE("aclblasCgeru", "incx span only addressable for vector length <= 1, got incx=%d, m=%d", incx, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == std::numeric_limits<int>::min() && n > 1) {
        OP_LOGE("aclblasCgeru", "incy span only addressable for vector length <= 1, got incy=%d, n=%d", incy, n);
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
    const uint32_t shapeM = static_cast<uint32_t>(m);
    const uint32_t shapeN = static_cast<uint32_t>(n);
    if (!IsComplexVectorAddressable(shapeM, incx)) {
        OP_LOGE("aclblasCgeru", "x address range overflows size_t");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsComplexVectorAddressable(shapeN, incy)) {
        OP_LOGE("aclblasCgeru", "y address range overflows size_t");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsComplexMatrixAddressable(shapeM, shapeN, static_cast<uint32_t>(lda))) {
        OP_LOGE("aclblasCgeru", "A address range overflows size_t");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return value / divisor + ((value % divisor) != 0U ? 1U : 0U);
}

struct CgeruBlockGrid {
    uint32_t rowBlocks;
    uint32_t colBlocks;
    uint32_t blockCount;
    uint64_t maxTileElements;
    uint64_t cacheCopyCost;
};

bool IsBetterBlockGrid(const CgeruBlockGrid& candidate, const CgeruBlockGrid& best)
{
    return candidate.blockCount > best.blockCount ||
           (candidate.blockCount == best.blockCount && candidate.maxTileElements < best.maxTileElements) ||
           (candidate.blockCount == best.blockCount && candidate.maxTileElements == best.maxTileElements &&
            candidate.cacheCopyCost < best.cacheCopyCost) ||
           (candidate.blockCount == best.blockCount && candidate.maxTileElements == best.maxTileElements &&
            candidate.cacheCopyCost == best.cacheCopyCost && candidate.colBlocks > best.colBlocks);
}

CgeruBlockGrid SelectBlockGrid(uint32_t m, uint32_t n, uint32_t candidateBlocks, uint64_t totalElements)
{
    CgeruBlockGrid best{1U, 1U, 1U, totalElements, 8ULL * m + 12ULL * n};
    const uint32_t maxRowBlocks = std::min(m, candidateBlocks);
    for (uint32_t rowBlocks = 1; rowBlocks <= maxRowBlocks; ++rowBlocks) {
        const uint32_t maxColBlocks = std::min(n, candidateBlocks / rowBlocks);
        for (uint32_t colBlocks = 1; colBlocks <= maxColBlocks; ++colBlocks) {
            const CgeruBlockGrid candidate{
                rowBlocks, colBlocks, rowBlocks * colBlocks,
                static_cast<uint64_t>(CeilDiv(m, rowBlocks)) * CeilDiv(n, colBlocks),
                8ULL * m * colBlocks + 12ULL * n * rowBlocks};
            if (IsBetterBlockGrid(candidate, best)) {
                best = candidate;
            }
        }
    }
    return best;
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

    const uint64_t totalElements = static_cast<uint64_t>(tiling.m) * tiling.n;
    uint64_t candidateBlocks = (totalElements + CGERU_MIN_ELEMENTS_PER_BLOCK - 1) / CGERU_MIN_ELEMENTS_PER_BLOCK;
    candidateBlocks = std::max<uint64_t>(1, std::min<uint64_t>(candidateBlocks, aivCoreNum));
    const CgeruBlockGrid grid =
        SelectBlockGrid(tiling.m, tiling.n, static_cast<uint32_t>(candidateBlocks), totalElements);
    tiling.rowBlocks = grid.rowBlocks;
    tiling.colBlocks = grid.colBlocks;

    const bool useContiguousReg =
        incx == 1 && incy == 1 && lda == m && tiling.m <= CGERU_VECTOR_MAX_ROWS && totalElements > 32768ULL;
    if (useContiguousReg) {
        // RegBase copies x once and slides over whole columns, never row slices.
        tiling.rowBlocks = 1;
        tiling.colBlocks = std::min(tiling.n, static_cast<uint32_t>(candidateBlocks));
        tiling.tilingKey = CGERU_TILING_CONTIGUOUS_REG;
        return tiling;
    }

    const uint32_t maxRows = tiling.m / tiling.rowBlocks + ((tiling.m % tiling.rowBlocks) != 0U ? 1U : 0U);
    const uint32_t maxCols = tiling.n / tiling.colBlocks + ((tiling.n % tiling.colBlocks) != 0U ? 1U : 0U);
    tiling.tilingKey = (maxRows <= CGERU_CACHE_COMPLEX && maxCols <= CGERU_CACHE_COMPLEX) ? CGERU_TILING_CACHE_X_AND_P :
                                                                                            CGERU_TILING_DIRECT_GM;
    return tiling;
}

} // namespace

extern "C" aclblasStatus_t aclblasCgeru(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCgeru", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    aclblasStatus_t status = ValidateCgeruShape(m, n, alpha, incx, incy, lda);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasComplex alphaValue = *alpha;
    if (alphaValue.real == 0.0f && alphaValue.imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    CHECK_RET(x != nullptr, OP_LOGE("aclblasCgeru", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasCgeru", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr, OP_LOGE("aclblasCgeru", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    status = ValidateCgeruAddressRange(m, n, lda, incx, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgeru", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const CgeruTilingData tiling = CalCgeruTilingData(m, n, lda, incx, incy, alphaValue, aivCoreNum);
    const uint32_t numBlocks = tiling.rowBlocks * tiling.colBlocks;
    OP_LOGD(
        "aclblasCgeru", "tiling: m=%u n=%u lda=%u incx=%ld incy=%ld rowBlocks=%u colBlocks=%u numBlocks=%u path=%s",
        tiling.m, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.rowBlocks, tiling.colBlocks, numBlocks,
        tiling.tilingKey == CGERU_TILING_CACHE_X_AND_P ?
            "CACHE_X_AND_P" :
            (tiling.tilingKey == CGERU_TILING_CONTIGUOUS_REG ? "CONTIGUOUS_REG" : "DIRECT_GM"));

    // All three paths use per-core UB/register scratch only; extra GM workspace is 0 bytes.
    cgeru_arch35_kernel_do(
        reinterpret_cast<const uint8_t*>(x), reinterpret_cast<const uint8_t*>(y), reinterpret_cast<uint8_t*>(A), tiling,
        numBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
