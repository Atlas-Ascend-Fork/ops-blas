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
 * \file cher2_host.cpp
 * \brief Host-side implementation of aclblasCher2 (arch22 / dav-2201).
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cher2_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"

namespace {

constexpr const char* OP_TAG = "aclblasCher2";

aclblasStatus_t ValidateCher2Structure(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, int incx, int incy, int lda)
{
    if (n < 0) {
        OP_LOGE(OP_TAG, "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE(OP_TAG, "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN || lda < std::max(1, n) || alpha == nullptr) {
        OP_LOGE(OP_TAG, "invalid stride/lda/alpha parameter");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchCher2Kernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, const aclblasComplex* y, int incy, aclblasComplex* a, int lda)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    Cher2Arch22TilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.numBlocks = numBlocks;
    tiling.tileRows = CHER2_ARCH22_TILE_ROWS;
    tiling.alphaReal = alpha->real;
    tiling.alphaImag = alpha->imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);

    OP_LOGD(
        OP_TAG, "tiling: n=%u lda=%u uplo=%u incx=%ld incy=%ld numBlocks=%u", tiling.n, tiling.lda, tiling.uplo,
        tiling.incx, tiling.incy, numBlocks);
    OP_LOGI(OP_TAG, "launching kernel: numBlocks=%u, aivCoreNum=%u", numBlocks, aivCoreNum);
    const aclError launchErr = cher2_kernel_do(
        reinterpret_cast<const uint8_t*>(x), reinterpret_cast<const uint8_t*>(y), reinterpret_cast<uint8_t*>(a), tiling,
        numBlocks, handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "cher2_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCher2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* alpha, const aclblasComplex* x,
    const int incx, const aclblasComplex* y, const int incy, aclblasComplex* a, const int lda)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    aclblasStatus_t st = ValidateCher2Structure(uplo, n, alpha, incx, incy, lda);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (n == 0 || (alpha->real == 0.0f && alpha->imag == 0.0f)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || a == nullptr) {
        OP_LOGE(OP_TAG, "x, y and A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint64_t maxElements = std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
    if (static_cast<uint64_t>(n) * static_cast<uint64_t>(lda) > maxElements) {
        OP_LOGE(OP_TAG, "matrix byte span overflows size_t");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return LaunchCher2Kernel(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
}
