/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../../gemm_batched_npu_common.h"
#include "cgemm_batched_param.h"

inline GemmBatchedBufferSizes ComputeCgemmBatchedBufferSizes(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, int lda, int ldb, int ldc)
{
    int physColsA = cgemmBatchedPhysCols(m, k, transA);
    int physColsB = cgemmBatchedPhysCols(k, n, transB);
    GemmBatchedBufferSizes sizes;
    sizes.aBytes = static_cast<size_t>(std::max(1, lda)) * std::max(1, physColsA) * sizeof(aclblasComplex);
    sizes.bBytes = static_cast<size_t>(std::max(1, ldb)) * std::max(1, physColsB) * sizeof(aclblasComplex);
    sizes.cBytes = static_cast<size_t>(std::max(1, ldc)) * std::max(1, n) * sizeof(aclblasComplex);
    return sizes;
}

inline aclblasStatus_t AllocateCgemmBatchedDevice(
    GemmBatchedDeviceCtx& ctx, const aclblasComplex* const Aarray[], const aclblasComplex* const Barray[],
    aclblasComplex* const Carray[], int batchCount, int k, aclblasOperation_t transA, aclblasOperation_t transB, int m,
    int n, int lda, int ldb, int ldc, const aclblasComplex* alpha)
{
    ctx.Init(batchCount);
    auto sizes = ComputeCgemmBatchedBufferSizes(transA, transB, m, n, k, lda, ldb, ldc);

    bool needAB = (k > 0) && (alpha == nullptr || !(alpha->real == 0.0f && alpha->imag == 0.0f));

    for (int i = 0; i < batchCount; i++) {
        aclblasStatus_t s = AllocateBatchBuffersTpl<aclblasComplex>(ctx, i, Aarray, Barray, Carray, needAB, sizes);
        if (s != ACLBLAS_STATUS_SUCCESS) {
            return s;
        }
    }
    return AllocatePtrArrays(ctx, needAB, Aarray != nullptr, Barray != nullptr, Carray != nullptr, batchCount);
}

inline CGEMM_BATCHED_SIGNATURE(aclblasCgemmBatched_npu)
{
    if (handle == nullptr || batchCount <= 0 || m <= 0 || n <= 0) {
        return aclblasCgemmBatched(
            handle, transA, transB, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    }

    GemmBatchedDeviceCtx ctx;
    aclblasStatus_t allocRet = AllocateCgemmBatchedDevice(
        ctx, Aarray, Barray, Carray, batchCount, k, transA, transB, m, n, lda, ldb, ldc, alpha);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        ctx.Cleanup();
        return allocRet;
    }

    const aclblasComplex* const* dAPtr =
        ctx.dAPtrArray ? reinterpret_cast<const aclblasComplex* const*>(ctx.dAPtrArray) : nullptr;
    const aclblasComplex* const* dBPtr =
        ctx.dBPtrArray ? reinterpret_cast<const aclblasComplex* const*>(ctx.dBPtrArray) : nullptr;
    aclblasComplex* const* dCPtr = ctx.dCPtrArray ? reinterpret_cast<aclblasComplex* const*>(ctx.dCPtrArray) : nullptr;

    auto sizes = ComputeCgemmBatchedBufferSizes(transA, transB, m, n, k, lda, ldb, ldc);
    return RunBatchedSyncAndCopy<aclblasComplex>(ctx, Carray, batchCount, sizes.cBytes, [&]() {
        return aclblasCgemmBatched(
            handle, transA, transB, m, n, k, alpha, dAPtr ? dAPtr : Aarray, lda, dBPtr ? dBPtr : Barray, ldb, beta,
            dCPtr ? dCPtr : Carray, ldc, batchCount);
    });
}
