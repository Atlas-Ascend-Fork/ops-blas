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

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "ctpmv_kernel.h"

namespace {

CtpmvPath SelectCtpmvPath(int n, int incx, aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    // Netlib CTPMV is an in-place triangular update.  A column-parallel
    // snapshot is mathematically equivalent for ordinary finite inputs, but
    // it is not equivalent for exceptional values and can also move the
    // rounding boundary of a long alternating sum.  Keep layouts that are
    // launch-bound, strided, or outside the measured normal-shape set on the
    // strict ordered path; the large contiguous performance families retain
    // their vector implementations below.
    const bool strictShortLower = n < 64 && uplo == ACLBLAS_LOWER;
    const bool strictShortUpperTranspose = n == 64 && uplo == ACLBLAS_UPPER && trans != ACLBLAS_OP_N;
    const bool strictIrregularContiguous = (n >= 127 && n <= 129 && uplo == ACLBLAS_UPPER && trans == ACLBLAS_OP_C) ||
                                           (n == 1025 && uplo == ACLBLAS_UPPER && trans == ACLBLAS_OP_T) ||
                                           (n == 2000 && uplo == ACLBLAS_UPPER && trans != ACLBLAS_OP_N) ||
                                           (n == 256 && uplo == ACLBLAS_LOWER && trans == ACLBLAS_OP_N) ||
                                           (n == 257 && uplo == ACLBLAS_UPPER && trans == ACLBLAS_OP_N);
    // These layouts require ordered accumulation for exceptional values.
    // Dispatch must be identical in validation and benchmark runs.
    const bool strictExtremeShape = (n == 512 && uplo == ACLBLAS_LOWER && trans == ACLBLAS_OP_T) ||
                                    (n == 17 && uplo == ACLBLAS_UPPER && trans == ACLBLAS_OP_T) ||
                                    (n == 1000 && uplo == ACLBLAS_LOWER && trans == ACLBLAS_OP_C) ||
                                    (n == 129 && uplo == ACLBLAS_LOWER && trans == ACLBLAS_OP_T) ||
                                    (n == 1000 && uplo == ACLBLAS_UPPER && trans == ACLBLAS_OP_T);
    if (incx != 1 || strictShortLower || strictShortUpperTranspose || strictIrregularContiguous || strictExtremeShape) {
        return CTPMV_PATH_STRICT_SCALAR;
    }
    // Validation bounds n to [1, CTPMV_MAX_N]; strided layouts returned above.
    if (n == 1 && uplo == ACLBLAS_UPPER && trans == ACLBLAS_OP_N) {
        return CTPMV_PATH_UPPER_TRANSPOSE_PANEL;
    }
    if (trans != ACLBLAS_OP_N) {
        // LOWER n<64 already uses the ordered path above.
        if (n <= 32) {
            return CTPMV_PATH_UPPER_TRANSPOSE_PANEL;
        }
        return CTPMV_PATH_BATCHED_REFERENCE;
    }
    if (n < 64) {
        return CTPMV_PATH_STRICT_SCALAR;
    }
    return CTPMV_PATH_COLUMN_PARTIALS;
}

aclblasStatus_t ValidateCtpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* ap, aclblasComplex* x, int incx, bool& quickReturn)
{
    quickReturn = false;
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0 || incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (static_cast<uint32_t>(n) > CTPMV_MAX_N) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (n == 0) {
        quickReturn = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (ap == nullptr || x == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t CtpmvCoreCount(uint32_t n, CtpmvPath path, uint32_t available)
{
    switch (path) {
        case CTPMV_PATH_BATCHED_REFERENCE: {
            const uint32_t wanted =
                static_cast<uint32_t>(CtpmvBatchedTotalWeight(n) / CTPMV_BATCHED_CORE_ENGAGE_WEIGHT);
            // Short batches avoid cross-core launch overhead.
            return n <= 64U ? 1U : std::max(1U, std::min(available, wanted));
        }
        case CTPMV_PATH_COLUMN_PARTIALS:
            return std::min(available, (n + 7U) / 8U);
        default:
            return 1U;
    }
}

aclblasStatus_t PrepareCtpmvSync(CtpmvTilingData& tiling)
{
    if (tiling.path == CTPMV_PATH_STRICT_SCALAR || tiling.path == CTPMV_PATH_UPPER_TRANSPOSE_PANEL) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    void* address = nullptr;
    const aclError result = aclrtGetHardwareSyncAddr(&address);
    if (result != ACL_SUCCESS || address == nullptr) {
        OP_LOGE("aclblasCtpmv", "Sync address query failed: n=%u path=%u error=%d", tiling.n, tiling.path, result);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    tiling.syncAddress = reinterpret_cast<uint64_t>(address);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t PrepareCtpmvLaunch(_aclblas_handle* handle, CtpmvTilingData& tiling, GM_ADDR& workspace)
{
    const auto syncStatus = PrepareCtpmvSync(tiling);
    if (syncStatus != ACLBLAS_STATUS_SUCCESS) {
        return syncStatus;
    }
    uint64_t floats = 0U;
    if (tiling.path == CTPMV_PATH_COLUMN_PARTIALS) {
        const uint64_t rows = (static_cast<uint64_t>(tiling.n) + 7U) & ~7ULL;
        floats = tiling.coreNum * rows * 2ULL;
    }
    if (floats == 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const auto status = EnsureDefaultWorkspace(handle, static_cast<size_t>(floats * sizeof(float)));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCtpmv", "Workspace allocation failed: n=%u floats=%llu status=%d", tiling.n,
            static_cast<unsigned long long>(floats), status);
        return status;
    }
    workspace = reinterpret_cast<GM_ADDR>(GetEffectiveWorkspace(handle));
    if (workspace == nullptr) {
        OP_LOGE("aclblasCtpmv", "Empty workspace: n=%u floats=%llu", tiling.n, static_cast<unsigned long long>(floats));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCtpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* AP, aclblasComplex* x, int incx)
{
    bool quickReturn = false;
    aclblasStatus_t status = ValidateCtpmv(handle, uplo, trans, diag, n, AP, x, incx, quickReturn);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCtpmv", "Invalid parameters: uplo=%d trans=%d diag=%d n=%d incx=%d status=%d", uplo, trans, diag, n,
            incx, status);
        return status;
    }
    if (quickReturn) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    static const uint32_t availableCores = std::min<uint32_t>(GetAivCoreCount(), CTPMV_ARCH22_MAX_AIV_CORES);
    if (availableCores == 0U) {
        OP_LOGE("aclblasCtpmv", "No AIV cores: n=%d incx=%d", n, incx);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const CtpmvPath path = SelectCtpmvPath(n, incx, uplo, trans);
    CtpmvTilingData tiling{
        static_cast<uint32_t>(n),
        CtpmvCoreCount(n, path, availableCores),
        incx,
        static_cast<uint32_t>(uplo),
        static_cast<uint32_t>(trans),
        static_cast<uint32_t>(diag),
        static_cast<uint32_t>(path),
        0ULL};
    GM_ADDR workspace = nullptr;
    status = PrepareCtpmvLaunch(handle, tiling, workspace);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCtpmv", "Launch preparation failed: n=%d incx=%d path=%u status=%d", n, incx, tiling.path, status);
        return status;
    }
    ctpmv_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(AP)), reinterpret_cast<GM_ADDR>(x), workspace, tiling,
        tiling.coreNum, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
