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
 * \file cher_host.cpp
 * \brief Host-side implementation of aclblasCher (arch22 / dav-2201).
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cher_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"

namespace {

constexpr const char* OP_TAG = "aclblasCher";

aclblasStatus_t ValidateCherParams(aclblasFillMode_t uplo, int n, int lda, int incx, const float* alpha)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(lda >= std::max(1, n), OP_LOGE(OP_TAG, "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    const int64_t elemSpan = static_cast<int64_t>(lda) * static_cast<int64_t>(n);
    if (elemSpan > (INT64_MAX / static_cast<int64_t>(sizeof(aclblasComplex))) ||
        elemSpan > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(OP_TAG, "lda*n span overflows index domain, lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CHECK_RET(incx != 0 && incx != INT_MIN, OP_LOGE(OP_TAG, "invalid incx=%d", incx);
              return ACLBLAS_STATUS_INVALID_VALUE);
    const int64_t absIncx = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t xSpan = static_cast<int64_t>(n - 1) * absIncx + 1;
    if (xSpan > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(OP_TAG, "x span overflows uint32 index domain, n=%d, incx=%d", n, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ResolveCherAlpha(
    aclblasHandle_t handle, const float* alpha, const aclblasComplex* x, const aclblasComplex* A, float& alphaVal,
    bool& alphaIsDevice, bool& quickReturn)
{
    aclblasStatus_t st = CheckPtrLocation(alpha, &alphaIsDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (!alphaIsDevice) {
        alphaVal = *alpha;
        quickReturn = (alphaVal == 0.0f);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x != nullptr && A != nullptr) {
        alphaVal = 0.0f;
        quickReturn = false;
        return ACLBLAS_STATUS_SUCCESS;
    }
    st = ReadHostScalarFromDevice(alpha, alphaVal, handle->stream, OP_TAG);
    if (st == ACLBLAS_STATUS_SUCCESS) {
        quickReturn = (alphaVal == 0.0f);
    }
    return st;
}

aclblasStatus_t LaunchCherKernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* A, int lda, float alphaVal, bool alphaIsDevice)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    CherTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.alphaIsDevice = alphaIsDevice ? 1U : 0U;
    tiling.alpha = alphaVal;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.numBlocks = numBlocks;
    tiling.tileRows = CHER_ARCH22_TILE_ROWS;

    OP_LOGD(
        OP_TAG, "tiling: n=%u lda=%u uplo=%u incx=%ld numBlocks=%u alphaIsDevice=%u", tiling.n, tiling.lda, tiling.uplo,
        tiling.incx, tiling.numBlocks, tiling.alphaIsDevice);
    OP_LOGI(OP_TAG, "launching kernel: numBlocks=%u, aivCoreNum=%u", numBlocks, aivCoreNum);
    const uint8_t* alphaPtr = alphaIsDevice ? reinterpret_cast<const uint8_t*>(alpha) : nullptr;
    const aclError launchErr = cher_kernel_do(
        alphaPtr, reinterpret_cast<const uint8_t*>(x), reinterpret_cast<uint8_t*>(A), tiling, numBlocks,
        handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "cher_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const float* alpha, const aclblasComplex* x,
    const int incx, aclblasComplex* A, const int lda)
{
    CHECK_RET(handle != nullptr, OP_LOGE(OP_TAG, "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateCherParams(uplo, n, lda, incx, alpha);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    float alphaVal = 0.0f;
    bool alphaIsDevice = false;
    bool quickReturn = false;
    st = ResolveCherAlpha(handle, alpha, x, A, alphaVal, alphaIsDevice, quickReturn);
    if (st != ACLBLAS_STATUS_SUCCESS || quickReturn) {
        return st;
    }
    CHECK_RET(x != nullptr && A != nullptr, OP_LOGE(OP_TAG, "x and A must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return LaunchCherKernel(handle, uplo, n, alpha, x, incx, A, lda, alphaVal, alphaIsDevice);
}
