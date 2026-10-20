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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "csyr_kernel.h"
#include "log/log.h"

namespace {

static_assert(sizeof(aclblasComplex) == 2 * sizeof(float), "aclblasComplex must be an interleaved pair of FP32 values");

inline bool IsZero(const aclblasComplex& value)
{
    return value.real == 0.0f && value.imag == 0.0f;
}

aclblasStatus_t FreeHostAlpha(aclblasComplex* hostValue)
{
    aclError ret = aclrtFreeHost(hostValue);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCsyr", "aclrtFreeHost for alpha D2H failed, ret=%d", static_cast<int>(ret));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ReadDeviceAlpha(const aclblasComplex* alpha, aclblasComplex& value, aclrtStream stream)
{
    aclblasComplex* hostValue = nullptr;
    aclError ret = aclrtMallocHost(reinterpret_cast<void**>(&hostValue), sizeof(*hostValue));
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCsyr", "aclrtMallocHost for alpha D2H failed, ret=%d", static_cast<int>(ret));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    ret = aclrtMemcpyAsync(hostValue, sizeof(*hostValue), alpha, sizeof(*hostValue), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCsyr", "aclrtMemcpyAsync alpha D2H failed, ret=%d", static_cast<int>(ret));
        FreeHostAlpha(hostValue);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCsyr", "aclrtSynchronizeStream for alpha D2H failed, ret=%d", static_cast<int>(ret));
        FreeHostAlpha(hostValue);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    value = *hostValue;
    return FreeHostAlpha(hostValue);
}

CsyrTilingData CalCsyrTilingData(
    int n, int lda, aclblasFillMode_t uplo, int incx, const aclblasComplex& alpha, bool alphaIsDevice)
{
    CsyrTilingData tiling{};
    tiling.numThreads = (n <= CSYR_SMALL_MATRIX_MAX_N) ? CSYR_SMALL_THREADS : CSYR_LARGE_THREADS;
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.alphaReal = alpha.real;
    tiling.alphaImag = alpha.imag;
    tiling.alphaIsDevice = alphaIsDevice ? 1u : 0u;
    tiling.incx = static_cast<int64_t>(incx);
    return tiling;
}

aclblasStatus_t ValidateCsyrParams(aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, int incx, int lda)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCsyr", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasCsyr", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasCsyr", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasCsyr", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ResolveCsyrAlpha(
    aclblasHandle_t handle, const aclblasComplex* alpha, const aclblasComplex* x, const aclblasComplex* A,
    aclblasComplex& value, bool& isDevice, bool& quickReturn)
{
    aclblasStatus_t status = CheckPtrLocation(alpha, &isDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (!isDevice) {
        value = *alpha;
        quickReturn = IsZero(value);
        return ACLBLAS_STATUS_SUCCESS;
    }
    // Keep the normal device-scalar path asynchronous. Only null data pointers
    // require a host read to distinguish a legal zero-alpha quick return.
    if (x != nullptr && A != nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    status = ReadDeviceAlpha(alpha, value, handle->stream);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        quickReturn = IsZero(value);
    }
    return status;
}

aclblasStatus_t LaunchCsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, aclblasComplex* A, int lda, const aclblasComplex& alphaValue, bool alphaIsDevice)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCsyr", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // Query the device rather than assuming a fixed core count. The kernel
    // selects a fixed-stride specialization only for a matching launch grid.
    uint32_t useNumBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    CsyrTilingData tiling = CalCsyrTilingData(n, lda, uplo, incx, alphaValue, alphaIsDevice);
    OP_LOGD(
        "aclblasCsyr", "tiling: n=%u lda=%u uplo=%u incx=%ld threads=%u blocks=%u alphaIsDevice=%u", tiling.n,
        tiling.lda, tiling.uplo, tiling.incx, tiling.numThreads, useNumBlocks, tiling.alphaIsDevice);

    void* alphaPtr = alphaIsDevice ? const_cast<aclblasComplex*>(alpha) : nullptr;
    csyr_kernel_do(
        static_cast<uint8_t*>(alphaPtr), reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(A), tiling, useNumBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* alpha, const aclblasComplex* x,
    const int incx, aclblasComplex* A, const int lda)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCsyr", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasCsyr", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t status = ValidateCsyrParams(uplo, n, alpha, incx, lda);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    aclblasComplex alphaValue{};
    bool alphaIsDevice = false;
    bool quickReturn = false;
    status = ResolveCsyrAlpha(handle, alpha, x, A, alphaValue, alphaIsDevice, quickReturn);
    if (status != ACLBLAS_STATUS_SUCCESS || quickReturn) {
        return status;
    }

    if (x == nullptr) {
        OP_LOGE("aclblasCsyr", "x must not be nullptr when alpha is nonzero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasCsyr", "A must not be nullptr when alpha is nonzero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    return LaunchCsyr(handle, uplo, n, alpha, x, incx, A, lda, alphaValue, alphaIsDevice);
}
