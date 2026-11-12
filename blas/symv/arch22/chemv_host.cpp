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
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "chemv_kernel.h"
#include "chemv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"

namespace {

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0f && v.imag == 0.0f; }

inline bool IsComplexOne(const aclblasComplex& v) { return v.real == 1.0f && v.imag == 0.0f; }

inline bool IsComplexFinite(const aclblasComplex& v) { return std::isfinite(v.real) && std::isfinite(v.imag); }

aclblasStatus_t ValidateStaticParams(
    aclblasFillMode_t uplo, int n, int lda, int incx, int incy, const aclblasComplex* alpha, const aclblasComplex* beta)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasChemv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasChemv", "invalid lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0 || incx == INT32_MIN || incy == INT32_MIN) {
        OP_LOGE("aclblasChemv", "invalid stride incx=%d, incy=%d", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int64_t incxAbs = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t incyAbs = (incy >= 0) ? static_cast<int64_t>(incy) : -static_cast<int64_t>(incy);
    const int64_t spanX = static_cast<int64_t>(n - 1) * incxAbs;
    const int64_t spanY = static_cast<int64_t>(n - 1) * incyAbs;
    constexpr int64_t kMaxSpan = static_cast<int64_t>(UINT32_MAX);
    if (spanX > kMaxSpan || spanY > kMaxSpan) {
        OP_LOGE(
            "aclblasChemv", "strided offset overflow: n=%d incx=%d incy=%d spanX=%lld spanY=%lld", n, incx, incy,
            static_cast<long long>(spanX), static_cast<long long>(spanY));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclblasChemv", "alpha and beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ResolveChemvScalarLocations(
    aclblasHandle_t handle, const aclblasComplex* alpha, const aclblasComplex* beta, bool& alphaDevice, bool& betaDevice)
{
    if (handle->cher2k_scalar_location_valid && handle->cher2k_scalar_alpha_ptr == alpha &&
        handle->cher2k_scalar_beta_ptr == beta) {
        alphaDevice = handle->cher2k_scalar_alpha_device;
        betaDevice = handle->cher2k_scalar_beta_device;
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = CheckPtrLocation(alpha, &alphaDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CheckPtrLocation(beta, &betaDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    handle->cher2k_scalar_location_valid = true;
    handle->cher2k_scalar_alpha_ptr = alpha;
    handle->cher2k_scalar_beta_ptr = beta;
    handle->cher2k_scalar_alpha_device = alphaDevice;
    handle->cher2k_scalar_beta_device = betaDevice;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ReadChemvScalars(
    aclblasHandle_t handle, const aclblasComplex* alpha, const aclblasComplex* beta, aclblasComplex& alphaVal,
    aclblasComplex& betaVal)
{
    bool alphaDevice = false;
    bool betaDevice = false;
    const aclblasStatus_t locSt = ResolveChemvScalarLocations(handle, alpha, beta, alphaDevice, betaDevice);
    if (locSt != ACLBLAS_STATUS_SUCCESS) {
        return locSt;
    }
    if (!alphaDevice) {
        alphaVal = *alpha;
    } else if (
        aclrtMemcpyAsync(
            &alphaVal, sizeof(alphaVal), alpha, sizeof(alphaVal), ACL_MEMCPY_DEVICE_TO_HOST, handle->stream) !=
        ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (!betaDevice) {
        betaVal = *beta;
    } else if (
        aclrtMemcpyAsync(&betaVal, sizeof(betaVal), beta, sizeof(betaVal), ACL_MEMCPY_DEVICE_TO_HOST, handle->stream) !=
        ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if ((alphaDevice || betaDevice) && aclrtSynchronizeStream(handle->stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t GetCachedAivCoreCount(aclblasHandle_t handle)
{
    if (handle->cgemm_aiv_cores != 0U) {
        return handle->cgemm_aiv_cores;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores != 0U) {
        handle->cgemm_aiv_cores = cores;
    }
    return cores;
}

uint32_t SelectColSplitCores(uint32_t n, uint32_t aivCoreNum)
{
    const uint32_t cappedAiv = std::min(aivCoreNum, CHEMV_MAX_CORES);
    uint32_t cores = 1U;
    if (n <= 32U) {
        cores = std::min(cappedAiv, 8U);
    } else if (n <= 64U) {
        cores = std::min(cappedAiv, 12U);
    } else if (n <= 128U) {
        cores = std::min(cappedAiv, 16U);
    } else if (n <= 256U) {
        cores = std::min(cappedAiv, 24U);
    } else if (n <= 512U) {
        cores = std::min(cappedAiv, 32U);
    } else {
        cores = cappedAiv;
    }
    cores = std::min(cores, CeilDiv<uint32_t>(n, 2U));
    return cores == 0U ? 1U : cores;
}

ChemvTilingData BuildBaseTiling(
    aclblasFillMode_t uplo, int n, int lda, const aclblasComplex& alphaVal, const aclblasComplex& betaVal, int incx,
    int incy, uint32_t cores, uint32_t mode, uint32_t strideFloats)
{
    ChemvTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.useCoreNum = cores;
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.strideFloats = strideFloats;
    tiling.mode = mode;
    tiling.alphaReal = alphaVal.real;
    tiling.alphaImag = alphaVal.imag;
    tiling.betaReal = betaVal.real;
    tiling.betaImag = betaVal.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    return tiling;
}

bool TryDispatchColSplitKernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int lda, const aclblasComplex& alphaVal,
    const aclblasComplex& betaVal, int incx, int incy, uint32_t aivCoreNum, uint8_t* aPtr, uint8_t* xPtr, uint8_t* yPtr)
{
    const uint32_t nU32 = static_cast<uint32_t>(n);
    const bool canUseColSplit = (incx == 1 && incy == 1 && nU32 > CHEMV_COLSPLIT_MIN_N &&
                                 nU32 <= CHEMV_COLSPLIT_MAX_N && IsComplexFinite(alphaVal) && IsComplexFinite(betaVal));
    if (!canUseColSplit) {
        return false;
    }
    const uint32_t cores = SelectColSplitCores(nU32, aivCoreNum);
    const uint32_t strideFloats = CeilAlign<uint32_t>(nU32, 64U);
    const size_t wsBytes = static_cast<size_t>(cores) * 2U * static_cast<size_t>(strideFloats) * sizeof(float);
    bool wsReady = GetEffectiveWorkspaceSize(handle) >= wsBytes;
    if (!wsReady && handle->workspace_owner == AclblasWorkspaceOwner::Library) {
        wsReady = (EnsureDefaultWorkspace(handle, wsBytes) == ACLBLAS_STATUS_SUCCESS);
    }
    if (!wsReady || GetEffectiveWorkspace(handle) == nullptr) {
        return false;
    }
    const ChemvTilingData tiling =
        BuildBaseTiling(uplo, n, lda, alphaVal, betaVal, incx, incy, cores, CHEMV_MODE_COLSPLIT, strideFloats);
    chemv_colsplit_kernel_do(
        aPtr, xPtr, yPtr, reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle)), tiling, cores, handle->stream);
    return true;
}

aclblasStatus_t DispatchChemvKernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex& alphaVal, const aclblasComplex* a,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex& betaVal, aclblasComplex* y, int incy)
{
    const uint32_t aivCoreNum = GetCachedAivCoreCount(handle);
    if (aivCoreNum == 0U) {
        OP_LOGE("aclblasChemv", "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint32_t nU32 = static_cast<uint32_t>(n);
    aclrtStream stream = handle->stream;
    auto* yPtr = reinterpret_cast<uint8_t*>(y);

    if (IsComplexZero(alphaVal)) {
        const uint32_t cores = std::max(1U, std::min(aivCoreNum, CeilDiv<uint32_t>(nU32, 256U)));
        const ChemvTilingData tiling =
            BuildBaseTiling(uplo, n, lda, alphaVal, betaVal, incx, incy, cores, CHEMV_MODE_SCALE, 0U);
        chemv_scale_kernel_do(yPtr, tiling, cores, stream);
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* aPtr = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a));
    auto* xPtr = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x));
    if (TryDispatchColSplitKernel(handle, uplo, n, lda, alphaVal, betaVal, incx, incy, aivCoreNum, aPtr, xPtr, yPtr)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint32_t generalCores =
        (nU32 <= CHEMV_COLSPLIT_MIN_N && incx == 1 && incy == 1) ?
            1U :
            ((nU32 <= 32U) ? 1U : std::max(1U, std::min(aivCoreNum, CeilDiv<uint32_t>(nU32, 16U))));
    const ChemvTilingData tiling =
        BuildBaseTiling(uplo, n, lda, alphaVal, betaVal, incx, incy, generalCores, CHEMV_MODE_GENERAL, 0U);
    chemv_general_kernel_do(aPtr, xPtr, yPtr, tiling, generalCores, stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasChemv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasChemv", "invalid n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t status = ValidateStaticParams(uplo, n, lda, incx, incy, alpha, beta);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    aclblasComplex alphaVal{0.0f, 0.0f};
    aclblasComplex betaVal{0.0f, 0.0f};
    status = ReadChemvScalars(handle, alpha, beta, alphaVal, betaVal);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (IsComplexZero(alphaVal) && IsComplexOne(betaVal)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!IsComplexZero(alphaVal) && (A == nullptr || x == nullptr)) {
        OP_LOGE("aclblasChemv", "A and x must not be nullptr when alpha is nonzero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasChemv", "y must not be nullptr when output is updated");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    return DispatchChemvKernel(handle, uplo, n, alphaVal, A, lda, x, incx, betaVal, y, incy);
}
