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
 * \file chpmv_host.cpp
 * \brief Host-side implementation of aclblasChpmv (arch22 / dav-2201).
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "chpmv_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"

namespace {

constexpr const char* OP_TAG = "aclblasChpmv";

aclblasStatus_t ValidateChpmvInput(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* ap,
    const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr.");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE(OP_TAG, "n must be nonnegative, got %d.", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE(OP_TAG, "uplo is invalid, got %d.", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN || alpha == nullptr || beta == nullptr ||
        ap == nullptr || x == nullptr || y == nullptr) {
        OP_LOGE(OP_TAG, "invalid strides or null operand pointers.");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchChpmvScaleOnly(
    aclblasHandle_t handle, int n, const aclblasComplex& betaVal, aclblasComplex* y, int incy, uint32_t aivCoreNum)
{
    const uint32_t size = static_cast<uint32_t>(n);
    const uint32_t reduceBlocks = std::min(CeilDiv<uint32_t>(size, CHPMV_ARCH22_TILE_ROWS), aivCoreNum);
    ChpmvTilingData tiling{};
    tiling.n = size;
    tiling.nPad = CeilAlign<uint32_t>(size, 8U);
    tiling.reduceBlocks = reduceBlocks;
    tiling.tileRows = CHPMV_ARCH22_TILE_ROWS;
    tiling.betaReal = betaVal.real;
    tiling.betaImag = betaVal.imag;
    tiling.incy = static_cast<int64_t>(incy);

    OP_LOGI(OP_TAG, "launching scale-only kernel: blocks=%u", reduceBlocks);
    chpmv_scale_kernel_do(reinterpret_cast<uint8_t*>(y), tiling, reduceBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchChpmvTwoStage(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex& alphaVal, const aclblasComplex* ap,
    const aclblasComplex* x, int incx, const aclblasComplex& betaVal, aclblasComplex* y, int incy, uint32_t aivCoreNum)
{
    const uint32_t size = static_cast<uint32_t>(n);
    const uint32_t nPad = CeilAlign<uint32_t>(size, 8U);
    const size_t bytesPerCore = static_cast<size_t>(nPad) * 2U * sizeof(float);
    uint8_t* wsPtr = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    const size_t wsSize = GetEffectiveWorkspaceSize(handle);
    if (wsPtr == nullptr || bytesPerCore > wsSize) {
        OP_LOGE(OP_TAG, "effective workspace (%zu B) insufficient for one core (%zu B)", wsSize, bytesPerCore);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint32_t maxWsCores = static_cast<uint32_t>(wsSize / bytesPerCore);
    const uint32_t accumBlocks = std::min({size, aivCoreNum, maxWsCores});
    const uint32_t reduceBlocks = std::min(CeilDiv<uint32_t>(size, CHPMV_ARCH22_TILE_ROWS), aivCoreNum);

    ChpmvTilingData tiling{};
    tiling.n = size;
    tiling.nPad = nPad;
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.accumBlocks = accumBlocks;
    tiling.reduceBlocks = reduceBlocks;
    tiling.tileRows = CHPMV_ARCH22_TILE_ROWS;
    tiling.alphaReal = alphaVal.real;
    tiling.alphaImag = alphaVal.imag;
    tiling.betaReal = betaVal.real;
    tiling.betaImag = betaVal.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);

    OP_LOGD(
        OP_TAG, "tiling: n=%u nPad=%u uplo=%u accumBlocks=%u reduceBlocks=%u", tiling.n, tiling.nPad, tiling.uplo,
        accumBlocks, reduceBlocks);
    OP_LOGI(OP_TAG, "launching two-stage kernels: accum=%u reduce=%u", accumBlocks, reduceBlocks);
    chpmv_accum_kernel_do(
        reinterpret_cast<const uint8_t*>(ap), reinterpret_cast<const uint8_t*>(x), wsPtr, tiling, accumBlocks,
        handle->stream);
    chpmv_reduce_kernel_do(wsPtr, reinterpret_cast<uint8_t*>(y), tiling, reduceBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* ap,
    const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    const aclblasStatus_t st = ValidateChpmvInput(handle, uplo, n, alpha, ap, x, incx, beta, y, incy);
    if (st != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return st;
    }
    const aclblasComplex alphaVal = *alpha;
    const aclblasComplex betaVal = *beta;
    if (alphaVal.real == 0.0f && alphaVal.imag == 0.0f && betaVal.real == 1.0f && betaVal.imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (alphaVal.real == 0.0f && alphaVal.imag == 0.0f) {
        return LaunchChpmvScaleOnly(handle, n, betaVal, y, incy, aivCoreNum);
    }
    return LaunchChpmvTwoStage(handle, uplo, n, alphaVal, ap, x, incx, betaVal, y, incy, aivCoreNum);
}
