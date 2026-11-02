/*
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
#include "ctrsmbatched_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
namespace {
aclblasStatus_t ValidateCtrsmBatched(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* alpha, int lda, int ldb, int batches)
{
    if ((side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) || (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) ||
        (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) ||
        (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) || m < 0 || n < 0 || std::min(lda, ldb) < 0 || batches < 1 ||
        !alpha ||
        ((m > 0 && n > 0) && (lda < std::max(1, side == ACLBLAS_SIDE_LEFT ? m : n) || ldb < std::max(1, m)))) {
        OP_LOGE(
            "aclblasCtrsmBatched", "invalid args side=%d uplo=%d trans=%d diag=%d m=%d n=%d lda=%d ldb=%d batch=%d",
            side, uplo, trans, diag, m, n, lda, ldb, batches);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
aclblasStatus_t ReadCtrsmAlpha(const aclblasComplex* alpha, aclrtStream stream, aclblasComplex& value)
{
    bool device = false;
    const auto status = CheckPtrLocation(alpha, &device);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if (!device) {
        value = *alpha;
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto ret = aclrtMemcpyAsync(&value, sizeof(value), alpha, sizeof(value), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCtrsmBatched", "device alpha read failed: %d", ret);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCtrsmBatched", "device alpha stream sync failed: %d", ret);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
int SelectCtrsmPath(
    bool square, bool small, bool one, int k, int rhs, int batches, int vectors, int right, int tr, int unit)
{
    const bool shortPanel =
        k > 48 && k <= 96 && rhs <= 64 && (batches <= vectors * 2 || (!right && !tr && !unit && k % 32 == 0));
    const bool blocked = one && (k > 64 || shortPanel) && k <= 4096;
    const int path = square ? 1 : small ? 2 : blocked ? 3 : one && k >= 4 && k <= 64 ? 4 : 0;
    return path;
}
size_t CtrsmWorkspaceSize(int path, int batches, int k, int rhs, int right)
{
    size_t bytes = 0;
    if (path == 1) {
        const size_t pairs = (size_t(batches) + 1) / 2;
        bytes = (pairs * 3 * 4096 + (pairs + 7) / 8 * 8) * sizeof(float);
    } else if (path == 2) {
        bytes = size_t(batches) * (48 * 64 + ((uint64_t(rhs) + 207) / 208) * 2 * 48 * 208) * sizeof(float);
    } else if (path == 3) {
        const size_t blocks = (size_t(k) + 63) / 64, dim = blocks * 128, rp = (size_t(rhs) + 63) / 64 * 64;
        bytes = size_t(batches) * (blocks * 16384 + dim * dim + (right ? rp * dim : 0) + blocks * 4096) * sizeof(float);
    } else if (path == 4) {
        const size_t pad = k <= 32 ? 32 : k <= 48 ? 48 : 64;
        const size_t nr = pad <= 32 ? 256 : pad <= 48 ? 160 : 128;
        const size_t packedRhs = right ? (size_t(rhs) + nr - 1) / nr * 4 * nr * pad : 0;
        bytes = size_t(batches) * (4 * pad * pad + packedRhs) * sizeof(float);
    }
    return bytes;
}
void LaunchCtrsmPath(
    aclblasHandle_t h, int path, void* aa, void* bb, int batches, int k, int rhs, int lda, int ldb, int right,
    int upper, int tr, int unit, int cores)
{
    void* w = GetEffectiveWorkspace(h);
    if (path == 1)
        LaunchCtrsmBatchedHalfDiagonalSmall(aa, bb, w, batches, upper, cores, h->stream);
    else if (path == 2) {
        const int mode = (!tr ? 1 : tr == 2 ? 2 : 0) + ((upper == !tr) ? 3 : 0);
        LaunchCtrsmBatchedSmallInverse(aa, bb, w, batches, k, rhs, lda, ldb, cores, h->stream, mode);
    } else {
        const auto launch = path == 3 ? LaunchCtrsmBatchedBlockInverse : LaunchCtrsmBatchedInverseCube;
        launch(aa, bb, w, batches, k, rhs, lda, ldb, right, upper, tr, unit, cores, h->stream);
    }
}
bool IsCompactSquare(int right, int tr, int unit, bool one, int m, int n, int lda, int ldb)
{
    return !right && !tr && !unit && one && m == n && lda == m && ldb == m && (m == 8 || m == 16 || m == 32);
}
bool IsDirectRhs(int right, int k, int rhs, int batches, int vectors, int unit, bool one)
{
    return right && k >= 4 && k <= 44 && rhs <= 24 && batches >= vectors * 4 && unit && one;
}
aclblasStatus_t DispatchCtrsmBatched(
    aclblasHandle_t h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, aclblasComplex value, const aclblasComplex* const a[], int lda, aclblasComplex* const b[], int ldb,
    int batches)
{
    auto* aa = const_cast<void*>(static_cast<const void*>(a));
    auto* bb = const_cast<void*>(static_cast<const void*>(b));
    const int right = side == ACLBLAS_SIDE_RIGHT, upper = uplo == ACLBLAS_UPPER, unit = diag == ACLBLAS_UNIT,
              tr = trans == ACLBLAS_OP_N ? 0 :
                   trans == ACLBLAS_OP_C ? 2 :
                                           1,
              k = right ? n : m, rhs = right ? m : n, cores = GetAicCoreCount(), vectors = GetAivCoreCount();
    const bool one = value.real == 1.f && value.imag == 0.f;
    auto fallback = [&]() {
        LaunchCtrsmBatchedVectorFallback(
            aa, bb, batches, k, rhs, lda, ldb, right, upper, tr, unit, value.real, value.imag, vectors, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    };
    if (!cores || !vectors) {
        OP_LOGE("aclblasCtrsmBatched", "hardware core count is zero");
        return ACLBLAS_STATUS_ARCH_MISMATCH;
    }
    const bool square = IsCompactSquare(right, tr, unit, one, m, n, lda, ldb);
    if (square && k < 32) {
        LaunchCtrsmBatchedPlanarSquare(aa, bb, k, batches, upper, vectors, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    const bool small = !right && unit && k <= 24 && one;
    if (!small && IsDirectRhs(right, k, rhs, batches, vectors, unit, one)) {
        LaunchCtrsmBatchedDirectRhs(aa, bb, batches, k, rhs, lda, ldb, right, upper, tr, unit, vectors, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    const int path = SelectCtrsmPath(square, small, one, k, rhs, batches, vectors, right, tr, unit);
    const size_t bytes = CtrsmWorkspaceSize(path, batches, k, rhs, right);
    if (!path || bytes > size_t(2) * 1024 * 1024 * 1024 - 16 * 1024 * 1024 ||
        (h->workspace_owner == AclblasWorkspaceOwner::User && bytes > h->workspace_size))
        return fallback();
    const auto status = EnsureDefaultWorkspace(h, bytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCtrsmBatched", "workspace allocation failed, bytes=%zu status=%d", bytes, int(status));
        return status;
    }
    LaunchCtrsmPath(h, path, aa, bb, batches, k, rhs, lda, ldb, right, upper, tr, unit, cores);
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace
aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const aclblasComplex* alpha, const aclblasComplex* const a[], int lda, aclblasComplex* const b[],
    int ldb, int batches)
{
    if (!h) {
        OP_LOGE("aclblasCtrsmBatched", "handle is null");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    auto status = ValidateCtrsmBatched(side, uplo, trans, diag, m, n, alpha, lda, ldb, batches);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0)
        return status;
    aclblasComplex value{};
    status = ReadCtrsmAlpha(alpha, h->stream, value);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    const bool zero = value.real == 0 && value.imag == 0;
    if (!b || (!zero && !a)) {
        OP_LOGE("aclblasCtrsmBatched", "pointer array is null m=%d n=%d batch=%d", m, n, batches);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (zero) {
        const auto cores = GetAivCoreCount();
        if (!cores) {
            OP_LOGE("aclblasCtrsmBatched", "Vector core count is zero");
            return ACLBLAS_STATUS_ARCH_MISMATCH;
        }
        LaunchCtrsmBatchedZero(const_cast<void*>(static_cast<const void*>(b)), m, n, ldb, batches, cores, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    return DispatchCtrsmBatched(h, side, uplo, trans, diag, m, n, value, a, lda, b, ldb, batches);
}
