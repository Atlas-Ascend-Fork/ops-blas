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
 * \file chpr2_host.cpp
 * \brief Host-side implementation of aclblasChpr2 (arch22 / dav-2201).
 */

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "chpr2_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"

namespace {

constexpr const char* OP_TAG = "aclblasChpr2";

aclblasStatus_t ValidateChpr2Body(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, const aclblasComplex* ap)
{
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha is nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE(OP_TAG, "uplo must be UPPER or LOWER"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        incx != 0 && incx != INT_MIN && incy != 0 && incy != INT_MIN,
        OP_LOGE(OP_TAG, "incx/incy must be nonzero and not INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        x != nullptr && y != nullptr && ap != nullptr,
        OP_LOGE(OP_TAG, "x, y and ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    const uint64_t maxElements = std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
    const uint64_t absIncx = (incx < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(incx))
                                        : static_cast<uint64_t>(incx);
    const uint64_t absIncy = (incy < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(incy))
                                        : static_cast<uint64_t>(incy);
    const uint64_t lenX = static_cast<uint64_t>(n - 1) * absIncx + 1ULL;
    const uint64_t lenY = static_cast<uint64_t>(n - 1) * absIncy + 1ULL;
    const uint64_t lenAP = static_cast<uint64_t>(n) * (static_cast<uint64_t>(n) + 1ULL) / 2ULL;
    CHECK_RET(
        lenX <= maxElements && lenY <= maxElements && lenAP <= maxElements,
        OP_LOGE(OP_TAG, "buffer byte length overflows size_t"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchChpr2Kernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex& alphaVal, const aclblasComplex* x,
    int incx, const aclblasComplex* y, int incy, aclblasComplex* ap)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(
        aivCoreNum != 0U, OP_LOGE(OP_TAG, "GetAivCoreCount failed"); return ACLBLAS_STATUS_EXECUTION_FAILED);
    const uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    Chpr2TilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.numBlocks = numBlocks;
    tiling.alphaReal = alphaVal.real;
    tiling.alphaImag = alphaVal.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);

    OP_LOGD(
        OP_TAG, "tiling: n=%u uplo=%u incx=%ld incy=%ld numBlocks=%u", tiling.n, tiling.uplo, tiling.incx, tiling.incy,
        numBlocks);
    OP_LOGI(OP_TAG, "launching kernel: numBlocks=%u, aivCoreNum=%u", numBlocks, aivCoreNum);
    chpr2_kernel_do(
        reinterpret_cast<const uint8_t*>(x), reinterpret_cast<const uint8_t*>(y), reinterpret_cast<uint8_t*>(ap),
        tiling, numBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* alpha, const aclblasComplex* x,
    const int incx, const aclblasComplex* y, const int incy, aclblasComplex* ap)
{
    CHECK_RET(handle != nullptr, OP_LOGE(OP_TAG, "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be nonnegative"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t st = ValidateChpr2Body(uplo, n, alpha, x, incx, y, incy, ap);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const aclblasComplex alphaVal = *alpha;
    if (alphaVal.real == 0.0f && alphaVal.imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return LaunchChpr2Kernel(handle, uplo, n, alphaVal, x, incx, y, incy, ap);
}
