/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HGEMM_BATCHED_NPU_WRAPPER_H
#define HGEMM_BATCHED_NPU_WRAPPER_H

#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "../../../gemm_batched/gemm_batched_npu_common.h"
#include "hgemmBatched_param.h"

// Test-side adapter: the CSV/param layer works with host FLOAT16 buffers, so the
// wrapper copies each batch to device, calls the library API on device pointer
// arrays, synchronizes, and copies C back to host.

inline GemmBatchedBufferSizes ComputeHgemmBatchedBufferSizes(
    aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, int lda, int ldb, int ldc)
{
    int physColsA = hgemmBatchedPhysCols(m, k, transA);
    int physColsB = hgemmBatchedPhysCols(k, n, transB);
    constexpr size_t kHalfBytes = 2;
    GemmBatchedBufferSizes sizes;
    sizes.aBytes = static_cast<size_t>(std::max(1, lda)) * std::max(1, physColsA) * kHalfBytes;
    sizes.bBytes = static_cast<size_t>(std::max(1, ldb)) * std::max(1, physColsB) * kHalfBytes;
    sizes.cBytes = static_cast<size_t>(std::max(1, ldc)) * std::max(1, n) * kHalfBytes;
    return sizes;
}

inline aclblasStatus_t AllocateHgemmBatchedDevice(
    GemmBatchedDeviceCtx& ctx,
    const aclblasHalf* const Aarray[], const aclblasHalf* const Barray[], aclblasHalf* const Carray[],
    int batchCount, int k,
    aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int lda, int ldb, int ldc,
    const aclblasHalf* alpha)
{
    ctx.Init(batchCount);
    auto sizes = ComputeHgemmBatchedBufferSizes(transA, transB, m, n, k, lda, ldb, ldc);

    // A/B are only needed when the GEMM path actually consumes them.
    bool needAB = (k > 0) && (alpha == nullptr || *alpha != 0);

    for (int i = 0; i < batchCount; i++) {
        aclblasStatus_t s = AllocateBatchBuffersTpl<aclblasHalf>(
            ctx, i, Aarray, Barray, Carray, needAB, sizes);
        if (s != ACLBLAS_STATUS_SUCCESS) { return s; }
    }
    return AllocatePtrArrays(ctx, needAB,
        Aarray != nullptr, Barray != nullptr, Carray != nullptr, batchCount);
}

inline HGEMM_BATCHED_SIGNATURE(aclblasHgemmBatched_npu)
{
    if (handle == nullptr || batchCount <= 0 || m <= 0 || n <= 0) {
        return aclblasHgemmBatched(
            handle, transA, transB, m, n, k,
            alpha, Aarray, lda, Barray, ldb,
            beta, Carray, ldc, batchCount);
    }

    GemmBatchedDeviceCtx ctx;
    aclblasStatus_t allocRet = AllocateHgemmBatchedDevice(
        ctx, Aarray, Barray, Carray, batchCount, k,
        transA, transB, m, n, lda, ldb, ldc, alpha);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        ctx.Cleanup();
        return allocRet;
    }

    const aclblasHalf* const* dAPtr =
        ctx.dAPtrArray ? reinterpret_cast<const aclblasHalf* const*>(ctx.dAPtrArray) : nullptr;
    const aclblasHalf* const* dBPtr =
        ctx.dBPtrArray ? reinterpret_cast<const aclblasHalf* const*>(ctx.dBPtrArray) : nullptr;
    aclblasHalf* const* dCPtr = ctx.dCPtrArray ? reinterpret_cast<aclblasHalf* const*>(ctx.dCPtrArray) : nullptr;

    auto sizes = ComputeHgemmBatchedBufferSizes(transA, transB, m, n, k, lda, ldb, ldc);
    return RunBatchedSyncAndCopy<aclblasHalf>(ctx, Carray, batchCount, sizes.cBytes, [&]() {
        return aclblasHgemmBatched(
            handle, transA, transB, m, n, k,
            alpha, dAPtr ? dAPtr : Aarray, lda,
            dBPtr ? dBPtr : Barray, ldb,
            beta, dCPtr ? dCPtr : Carray, ldc, batchCount);
    });
}

#endif // HGEMM_BATCHED_NPU_WRAPPER_H
