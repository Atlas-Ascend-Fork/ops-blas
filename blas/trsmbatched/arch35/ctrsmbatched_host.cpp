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
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "ctrsmbatched_kernel.h"

void ctrsm_batched_kernel_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream);
void ctrsm_batched_cube_do(
    uint8_t* a, uint8_t* b, uint8_t* workspace, const CtrsmBatchedTilingData& t, uint32_t kp, uint32_t rp, uint32_t aiv,
    uint32_t aic, void* stream);

void ctrsm_batched_square_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream);

namespace {
// Rounded division uses uint64_t for products of batch and RHS counts.
uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

bool IsSquareFastPath(const CtrsmBatchedTilingData& t)
{
    return t.m == t.n && (t.m == 8 || t.m == 16 || t.m == 32) && t.side == ACLBLAS_SIDE_LEFT &&
           t.trans == ACLBLAS_OP_N && t.diag == ACLBLAS_NON_UNIT && t.alpha.real == 1 && t.alpha.imag == 0;
}

uint64_t VectorWork(const CtrsmBatchedTilingData& t)
{
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const uint64_t rhs = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    if (t.diag == ACLBLAS_UNIT) {
        const uint32_t tile = k <= 40 && rhs > 128 ? 256 : 128;
        return CeilDiv(rhs, tile) * t.batchCount;
    }
    if (t.alpha.real == 0 && t.alpha.imag == 0) {
        return CeilDiv(rhs * t.batchCount, 128);
    }
    if (k >= 128) {
        return CeilDiv(rhs, 64) * t.batchCount;
    }
    const auto plan = GetCtrsmBatchedVectorPlan(t);
    return CeilDiv(CeilDiv(rhs, plan.width / plan.groups) * t.batchCount, plan.groups);
}

// ALLOC_FAILED means the caller can use the workspace-free vector solve.
// Other runtime errors must be returned to the user unchanged.
aclblasStatus_t TryLaunchCube(
    aclblasHandle_t handle, uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t aiv)
{
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const uint64_t rhs = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    const bool candidate = (k >= 64 && !(k == 64 && rhs == 64)) || (k >= 16 && rhs >= 128);
    if (!candidate || t.diag == ACLBLAS_UNIT || (t.alpha.real == 0 && t.alpha.imag == 0)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const uint64_t kp = CeilDiv(k, 8) * 8;
    const uint64_t rp = CeilDiv(rhs, 16) * 16;
    const uint64_t factor = 8 * (2 * kp + rp);
    if (kp > ACLBLAS_MAX_WORKSPACE_SIZE / factor) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const uint64_t perBatch = kp * factor;
    if (t.batchCount > ACLBLAS_MAX_WORKSPACE_SIZE / perBatch || GetAicCoreCount() == 0) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const auto status = EnsureDefaultWorkspace(handle, perBatch * t.batchCount);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    ctrsm_batched_cube_do(
        a, b, reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle)), t, kp, rp, aiv, GetAicCoreCount(),
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

aclblasStatus_t aclblasCtrsmBatched(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* const A[], int lda,
    aclblasComplex* const B[], int ldb, int batchCount)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const auto status = ValidateCtrsmBatched(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return status;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    CtrsmBatchedTilingData t{
        static_cast<uint32_t>(m),
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(lda),
        static_cast<uint32_t>(ldb),
        static_cast<uint32_t>(batchCount),
        side,
        uplo,
        trans,
        diag,
        *alpha};
    auto a = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex**>(A));
    auto b = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex**>(B));
    if (IsSquareFastPath(t)) {
        const uint32_t groups = m == 8 ? (batchCount <= 256 ? 8 : 32) : (m == 16 ? 8 : 4);
        const uint32_t blocks = std::min<uint64_t>(cores, CeilDiv(t.batchCount, groups));
        ctrsm_batched_square_do(a, b, t, blocks, handle->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    const auto cubeStatus = TryLaunchCube(handle, a, b, t, cores);
    if (cubeStatus != ACLBLAS_STATUS_ALLOC_FAILED) {
        return cubeStatus;
    }
    const uint32_t blocks = std::min<uint64_t>(cores, VectorWork(t));
    ctrsm_batched_kernel_do(a, b, t, blocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
