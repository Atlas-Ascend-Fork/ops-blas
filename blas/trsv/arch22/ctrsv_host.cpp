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
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "ctrsv_kernel.h"

namespace {
constexpr uint32_t TILE_SIZE = 64U;

bool ValidCtrsvEnums(aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    return (uplo == ACLBLAS_LOWER || uplo == ACLBLAS_UPPER) &&
           (trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C) &&
           (diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT);
}

size_t ConfigurePreparedWorkspace(CtrsvPreparedArgs& args, uint32_t paddedN)
{
    size_t offset = CTRSV_TILING_REGION_BYTES;
    const auto reserve = [&offset](uint32_t& field, size_t bytes) {
        offset = (offset + CTRSV_WORKSPACE_ALIGNMENT - 1U) & ~(static_cast<size_t>(CTRSV_WORKSPACE_ALIGNMENT) - 1U);
        field = static_cast<uint32_t>(offset);
        offset += bytes;
    };
    const size_t vectorBytes = static_cast<size_t>(paddedN) * 2U * sizeof(float);
    reserve(args.flags, static_cast<size_t>(args.blocks) * CTRSV_FLAG_SLOT_BYTES);
    reserve(args.ping, vectorBytes);
    const size_t panelBytes = static_cast<size_t>(paddedN) * TILE_SIZE * sizeof(float);
    reserve(args.diagonalReal, panelBytes);
    reserve(args.diagonalImag, panelBytes);
    return (offset + CTRSV_WORKSPACE_ALIGNMENT - 1U) & ~(static_cast<size_t>(CTRSV_WORKSPACE_ALIGNMENT) - 1U);
}
} // namespace

static aclblasStatus_t ValidateCtrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCtrsv", "handle is null, n=%d lda=%d incx=%d", n, lda, incx);
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0 || incx == 0 || incx == std::numeric_limits<int>::min() || lda < std::max(1, n) ||
        !ValidCtrsvEnums(uplo, trans, diag)) {
        OP_LOGE(
            "aclblasCtrsv", "invalid n=%d lda=%d incx=%d uplo=%d trans=%d diag=%d", n, lda, incx,
            static_cast<int>(uplo), static_cast<int>(trans), static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (A == nullptr || x == nullptr) {
        OP_LOGE("aclblasCtrsv", "null input A=%p x=%p n=%d lda=%d incx=%d", A, x, n, lda, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 1 && diag == ACLBLAS_UNIT)
        return ACLBLAS_STATUS_SUCCESS;

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCtrsvWorkspace(
    _aclblas_handle* h, uint8_t* matrix, uint8_t* vector, const CtrsvTilingData& td, bool contiguous)
{
    const int32_t n = td.n, lda = td.lda, uplo = td.uplo, trans = td.trans, diag = td.diag;
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0U || GetAicCoreCount() == 0U) {
        OP_LOGE("aclblasCtrsv", "failed to query core counts, n=%d lda=%d", n, lda);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint32_t panels = (static_cast<uint32_t>(n) + TILE_SIZE - 1U) / TILE_SIZE;
    const bool tail = n % TILE_SIZE != 0U;
    // UNIT contiguous inputs have already returned above. Every prepared
    // owner, including aligned shapes, has capacity for at most four panels.
    const bool prepared = contiguous && n > 64 && n <= 4096 && cores >= (panels + 3U) / 4U;
    CtrsvPreparedArgs args{};
    args.n = static_cast<uint32_t>(n);
    args.lda = static_cast<uint32_t>(lda);
    args.uplo = uplo;
    args.blocks = prepared ? std::min(panels, cores) : 1U;
    const size_t requiredBytes = prepared ? ConfigurePreparedWorkspace(args, panels * TILE_SIZE) : sizeof(td);
    const auto status = EnsureDefaultWorkspace(h, requiredBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            "aclblasCtrsv", "workspace allocation failed, bytes=%zu n=%d status=%d", requiredBytes, n,
            static_cast<int>(status));
        return status;
    }
    void* workspace = GetEffectiveWorkspace(h);
    if (workspace == nullptr || GetEffectiveWorkspaceSize(h) < requiredBytes) {
        OP_LOGE(
            "aclblasCtrsv", "invalid workspace=%p bytes=%zu required=%zu n=%d", workspace, GetEffectiveWorkspaceSize(h),
            requiredBytes, n);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (!prepared &&
        aclrtMemcpyAsync(workspace, sizeof(td), &td, sizeof(td), ACL_MEMCPY_HOST_TO_DEVICE, h->stream) != ACL_SUCCESS) {
        OP_LOGE("aclblasCtrsv", "tiling copy failed, n=%d lda=%d trans=%d diag=%d", n, lda, trans, diag);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ctrsv_kernel_do(
        matrix, vector, reinterpret_cast<uint8_t*>(workspace), trans, diag, prepared, tail, args, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCtrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
{
    const auto validation = ValidateCtrsv(handle, uplo, trans, diag, n, A, lda, x, incx);
    if (validation != ACLBLAS_STATUS_SUCCESS)
        return validation;
    if (n == 0 || (n == 1 && diag == ACLBLAS_UNIT))
        return ACLBLAS_STATUS_SUCCESS;
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    auto* matrix = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A));
    auto* vector = reinterpret_cast<uint8_t*>(x);
    const bool contiguous = lda == n && incx == 1;
    if (contiguous && n <= 64) {
        ctrsv_tiny_do(matrix, vector, static_cast<uint32_t>(n), uplo, trans, diag, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (contiguous && diag == ACLBLAS_UNIT && n <= 4096) {
        ctrsv_ordered_unit_do(matrix, vector, static_cast<uint32_t>(n), uplo, trans, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchCtrsvWorkspace(h, matrix, vector, {n, lda, incx, uplo, trans, diag}, contiguous);
}
