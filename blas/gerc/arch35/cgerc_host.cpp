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
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "log/log.h"
#include "cgerc_kernel.h"
#include "cgerc_tiling_data.h"

namespace {

constexpr const char* OP_NAME = "aclblasCgerc";
constexpr uint32_t THREAD_COUNT_GROWTH_FACTOR = 2U;

bool IsComplexIndexRangeSafe(uint64_t maxIndex)
{
    constexpr uint64_t elementBytes = sizeof(aclblasComplex);
    return maxIndex <= (std::numeric_limits<uint64_t>::max() - (elementBytes - 1U)) / elementBytes;
}

bool AreAddressRangesSafe(int m, int n, int lda, int incx, int incy)
{
    if (m == 0 || n == 0) {
        return true;
    }
    const uint64_t absIncx =
        incx > 0 ? static_cast<uint64_t>(incx) : static_cast<uint64_t>(-static_cast<int64_t>(incx));
    const uint64_t absIncy =
        incy > 0 ? static_cast<uint64_t>(incy) : static_cast<uint64_t>(-static_cast<int64_t>(incy));
    const uint64_t xMaxIndex = static_cast<uint64_t>(m - 1) * absIncx;
    const uint64_t yMaxIndex = static_cast<uint64_t>(n - 1) * absIncy;
    const uint64_t aMaxIndex = static_cast<uint64_t>(n - 1) * static_cast<uint64_t>(lda) + static_cast<uint64_t>(m - 1);
    return IsComplexIndexRangeSafe(xMaxIndex) && IsComplexIndexRangeSafe(yMaxIndex) &&
           IsComplexIndexRangeSafe(aMaxIndex);
}

aclblasStatus_t ValidateCgercParams(
    int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    const aclblasComplex* a, int lda)
{
    // GERC accepts negative BLAS strides; only a zero stride is invalid. A is column-major with m rows,
    // so its leading dimension is bounded by m rather than by the number of columns n.
    CHECK_RET(m >= 0, OP_LOGE(OP_NAME, "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE(OP_NAME, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_NAME, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE(OP_NAME, "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE(OP_NAME, "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (lda < std::max(1, m)) {
        OP_LOGE(OP_NAME, "lda must be >= max(1,m), got lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0) {
        CHECK_RET(x != nullptr, OP_LOGE(OP_NAME, "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(y != nullptr, OP_LOGE(OP_NAME, "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(a != nullptr, OP_LOGE(OP_NAME, "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(AreAddressRangesSafe(m, n, lda, incx, incy),
                  OP_LOGE(OP_NAME, "input dimensions and strides exceed the 64-bit byte address range");
                  return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t SelectThreadCount(uint32_t m)
{
    uint32_t threads = SIMT_MIN_THREAD_NUM;
    while (threads < m && threads < SIMT_MAX_THREAD_NUM) {
        threads *= THREAD_COUNT_GROWTH_FACTOR;
    }
    return threads;
}

bool MakeVectorTiling(CgercTilingData& tiling)
{
    using namespace CgercConfig;
    if (tiling.contiguous == 0U || tiling.alphaOne == 0U || tiling.rowBlocks != 1U || tiling.m < FP32_VECTOR_LANES) {
        return false;
    }
    const uint64_t alignedRows = CeilDiv<uint64_t>(tiling.m, FP32_VECTOR_LANES) * FP32_VECTOR_LANES;
    const uint64_t xBytes = alignedRows * sizeof(aclblasComplex);
    if (xBytes + Y_ALIGNMENT_MARGIN >= UB_SIZE) {
        return false;
    }
    const uint64_t bytesPerTileColumn = A_BUFFER_COUNT * xBytes + sizeof(aclblasComplex);
    const uint64_t tileColumns = (UB_SIZE - xBytes - Y_ALIGNMENT_MARGIN) / bytesPerTileColumn;
    if (tileColumns == 0U) {
        return false;
    }
    tiling.vectorAlignedRows = static_cast<uint32_t>(alignedRows);
    tiling.vectorTileColumns = static_cast<uint32_t>(tileColumns);
    return true;
}

CgercTilingData MakeTiling(
    int m, int n, int lda, const aclblasComplex& alpha, int incx, int incy, uint32_t coreCount, uint32_t& numBlocks)
{
    CgercTilingData tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.numThreads = SelectThreadCount(tiling.m);

    const uint32_t maxRowBlocks = CeilDiv<uint32_t>(tiling.m, tiling.numThreads);
    const uint32_t preferredColBlocks = std::min(tiling.n, coreCount);
    const uint32_t availableRowBlocks = std::max(1U, coreCount / std::max(1U, preferredColBlocks));
    tiling.rowBlocks = std::max(1U, std::min(maxRowBlocks, availableRowBlocks));
    tiling.colBlocks = std::max(1U, std::min(tiling.n, coreCount / tiling.rowBlocks));
    numBlocks = tiling.rowBlocks * tiling.colBlocks;

    tiling.incx = incx;
    tiling.incy = incy;
    tiling.alphaReal = alpha.real;
    tiling.alphaImag = alpha.imag;
    tiling.contiguous = (incx == 1 && incy == 1 && lda == m) ? 1U : 0U;
    tiling.alphaOne = (alpha.real == 1.0F && alpha.imag == 0.0F) ? 1U : 0U;
    tiling.vectorPath = MakeVectorTiling(tiling) ? 1U : 0U;
    return tiling;
}

aclblasStatus_t LaunchCgercKernel(
    aclblasHandle_t handle, int m, int n, const aclblasComplex& alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* a, int lda)
{
    const uint32_t aivCoreCount = GetAivCoreCount();
    if (aivCoreCount == 0U) {
        OP_LOGE(OP_NAME, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t numBlocks = 1U;
    const CgercTilingData tiling = MakeTiling(m, n, lda, alpha, incx, incy, aivCoreCount, numBlocks);
    OP_LOGD(
        OP_NAME, "tiling: m=%u n=%u lda=%u incx=%d incy=%d threads=%u rowBlocks=%u colBlocks=%u blocks=%u path=%s",
        tiling.m, tiling.n, tiling.lda, tiling.incx, tiling.incy, tiling.numThreads, tiling.rowBlocks, tiling.colBlocks,
        numBlocks, tiling.contiguous ? "contiguous" : "strided");
    OP_LOGI(OP_NAME, "launching kernel: blocks=%u cores=%u", numBlocks, aivCoreCount);

    const aclError launchErr = cgerc_arch35_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(a), tiling, numBlocks,
        handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasCgerc", "cgerc_arch35_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCgerc(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* a, int lda)
{
    // The handle error takes precedence; all remaining argument checks run before any quick return.
    CHECK_RET(handle != nullptr, OP_LOGE(OP_NAME, "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    aclblasStatus_t status = ValidateCgercParams(m, n, alpha, x, incx, y, incy, a, lda);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m == 0 || n == 0 || (alpha->real == 0.0F && alpha->imag == 0.0F)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchCgercKernel(handle, m, n, *alpha, x, incx, y, incy, a, lda);
}
