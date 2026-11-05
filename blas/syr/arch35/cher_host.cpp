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
#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "cher_kernel.h"
#include "cher_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

namespace {

constexpr uint32_t CHER_TINY_MATRIX_MAX_N = 32;
constexpr uint32_t CHER_TINY_THREADS = 64;
constexpr uint32_t CHER_SMALL_MATRIX_MAX_N = 512;
constexpr uint32_t CHER_SMALL_THREADS = 512;
constexpr uint32_t CHER_LARGE_THREADS = 1024;

// Validation order aligned with task book §2.5:
// handle -> n < 0 -> n == 0 quick return -> uplo -> lda -> incx -> alpha pointer.
static aclblasStatus_t ValidateCherParams(aclblasFillMode_t uplo, int n, int lda, int incx, const float* alpha)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasCher", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        lda >= std::max(1, n), OP_LOGE("aclblasCher", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    const int64_t elementSpan = static_cast<int64_t>(lda) * static_cast<int64_t>(n);
    if (elementSpan > (INT64_MAX / static_cast<int64_t>(sizeof(aclblasComplex)))) {
        OP_LOGE("aclblasCher", "lda*n span overflows addressable bytes, lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (elementSpan > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE("aclblasCher", "lda*n span exceeds uint32 kernel index domain, lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CHECK_RET(incx != 0, OP_LOGE("aclblasCher", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        incx != INT_MIN, OP_LOGE("aclblasCher", "incx must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    const int64_t absIncx = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t xSpan = static_cast<int64_t>(n - 1) * absIncx + 1;
    if (xSpan > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE("aclblasCher", "x span 1+(n-1)*|incx| exceeds uint32 kernel index domain, n=%d, incx=%d", n, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasCher", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Align with aclblasCsyr: keep the normal device-scalar path asynchronous (BL-14).
// Only null x/A requires a host read to distinguish a legal zero-alpha quick return.
static aclblasStatus_t ResolveCherAlpha(
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
    st = ReadHostScalarFromDevice(alpha, alphaVal, handle->stream, "aclblasCher");
    if (st == ACLBLAS_STATUS_SUCCESS) {
        quickReturn = (alphaVal == 0.0f);
    }
    return st;
}

static CherTilingData CalCherTilingData(
    uint32_t useNumBlocks, int n, int lda, aclblasFillMode_t uplo, float alpha, bool alphaIsDevice, int incx)
{
    CherTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.alphaIsDevice = alphaIsDevice ? 1u : 0u;
    tiling.alpha = alpha;
    tiling.incx = static_cast<int64_t>(incx);
    // Match Ssyr / CherDispatchUb: non-zero row share so n<=16 UB path is reachable.
    tiling.rowsPerBlock =
        static_cast<uint32_t>((n + static_cast<int>(useNumBlocks) - 1) / static_cast<int>(useNumBlocks));
    if (n <= static_cast<int>(CHER_TINY_MATRIX_MAX_N)) {
        tiling.numThreads = CHER_TINY_THREADS;
    } else if (n <= static_cast<int>(CHER_SMALL_MATRIX_MAX_N)) {
        tiling.numThreads = CHER_SMALL_THREADS;
    } else {
        tiling.numThreads = CHER_LARGE_THREADS;
    }
    return tiling;
}

static aclblasStatus_t LaunchCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* A, int lda, float alphaVal, bool alphaIsDevice)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCher", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // Tiny n may take the UB path (rowsPerBlock>0, n<=16 in kernel); single block avoids
    // multi-block UB sync overhead on PF_1009 (n=16).
    uint32_t useNumBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    if (n <= 16) {
        useNumBlocks = 1;
    }
    CherTilingData tiling = CalCherTilingData(useNumBlocks, n, lda, uplo, alphaVal, alphaIsDevice, incx);
    OP_LOGD(
        "aclblasCher",
        "tiling: n=%u lda=%u uplo=%u incx=%ld numThreads=%u rowsPerBlock=%u numBlocks=%u alphaIsDevice=%u", tiling.n,
        tiling.lda, tiling.uplo, tiling.incx, tiling.numThreads, tiling.rowsPerBlock, useNumBlocks,
        tiling.alphaIsDevice);
    OP_LOGI("aclblasCher", "launching kernel: blocks=%u, cores=%u", useNumBlocks, aivCoreNum);
    uint8_t* alphaPtr = alphaIsDevice ? reinterpret_cast<uint8_t*>(const_cast<float*>(alpha)) : nullptr;
    cher_kernel_do(
        alphaPtr, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(A), tiling,
        useNumBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCher(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const float* alpha, const aclblasComplex* x,
    const int incx, aclblasComplex* A, const int lda)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCher", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasCher", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
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
    CHECK_RET(
        x != nullptr, OP_LOGE("aclblasCher", "x must not be nullptr when alpha is nonzero");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        A != nullptr, OP_LOGE("aclblasCher", "A must not be nullptr when alpha is nonzero");
        return ACLBLAS_STATUS_INVALID_VALUE);
    return LaunchCher(handle, uplo, n, alpha, x, incx, A, lda, alphaVal, alphaIsDevice);
}
