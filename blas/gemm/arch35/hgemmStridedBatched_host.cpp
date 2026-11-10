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
#include <limits>
#include "cann_ops_blas.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/dtype_cast.h"
#include "common/helper/host_utils.h"
#include "hgemmStridedBatched_kernel.h"

namespace {
constexpr char OP_NAME[] = "aclblasHgemmStridedBatched";
bool ValidOp(aclblasOperation_t op) { return op == ACLBLAS_OP_N || op == ACLBLAS_OP_T || op == ACLBLAS_OP_C; }

// All operands have already been checked nonnegative. Include the final byte,
// not merely the first byte of the last element, in the address overflow check.
bool ValidSpan(const void* ptr, int rows, int cols, int ld, int64_t stride, int batches)
{
    if (rows == 0 || cols == 0)
        return true;
    const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / sizeof(aclblasHalf);
    const uint64_t matrix = static_cast<uint64_t>(cols - 1) * ld + rows;
    if (matrix > limit)
        return false;
    if (batches > 1 && static_cast<uint64_t>(stride) > (limit - matrix) / (batches - 1))
        return false;
    const uint64_t bytes = sizeof(aclblasHalf) * (static_cast<uint64_t>(batches - 1) * stride + matrix);
    return bytes - 1 <= std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(ptr);
}

uint32_t VectorBlocks(const HgemmBatchEpilogue& e, uint32_t cores)
{
    const uint64_t count = static_cast<uint64_t>(e.m) * e.n * e.batchCount;
    return static_cast<uint32_t>(
        std::min<uint64_t>(cores, (count + HGEMM_EPILOGUE_ELEMENTS_PER_CORE - 1) / HGEMM_EPILOGUE_ELEMENTS_PER_CORE));
}

template <uint64_t TILE>
uint64_t PaddedArea(int m, int n)
{
    static_assert(TILE > 0, "Tile size must be positive");
    return ((static_cast<uint64_t>(m) + TILE - 1) / TILE * TILE) *
           ((static_cast<uint64_t>(n) + TILE - 1) / TILE * TILE);
}

GemmStridedBatchedExTilingData MakeTiling(
    int m, int n, int k, int lda, int ldb, int ldc, int64_t strideA, int64_t strideB, int64_t outputStride, int batches,
    bool ta, bool tb, uint32_t cores)
{
    GemmStridedBatchedExTilingData t{};
    t.m = n;
    t.n = m;
    t.k = k;
    t.lda = ldb;
    t.ldb = lda;
    t.ldc = ldc;
    t.logicalM = m;
    t.logicalN = n;
    t.isTransA = tb;
    t.isTransB = ta;
    t.strideA = strideB;
    t.strideB = strideA;
    t.workspaceBatchStride = outputStride;
    t.batchCount = batches;
    t.splitK = 1;
    t.baseM = (m <= HGEMM_TILE_SMALL && n <= HGEMM_TILE_SMALL) ? HGEMM_TILE_SMALL : HGEMM_TILE_STANDARD;
    if (m >= HGEMM_TILE_STANDARD && n >= HGEMM_TILE_STANDARD && k >= HGEMM_REDUCTION_STANDARD) {
        // Prefer the larger tile unless its padded output area grows by more
        // than 50%. Near a 256 boundary, 128 tiles avoid oversized L0 loads.
        if (HGEMM_PADDED_AREA_DENOMINATOR * PaddedArea<HGEMM_TILE_LARGE>(m, n) <=
            HGEMM_PADDED_AREA_NUMERATOR * PaddedArea<HGEMM_TILE_STANDARD>(m, n))
            t.baseM = HGEMM_TILE_LARGE;
    }
    t.baseN = t.baseM;
    t.baseK = t.baseM == HGEMM_TILE_STANDARD ? HGEMM_REDUCTION_STANDARD : HGEMM_REDUCTION_SHALLOW;
    t.c0Size = HGEMM_CUBE_C0_ELEMENTS;
    t.singleCoreM = t.baseM;
    t.singleCoreN = t.baseN;
    t.mBlocks = (static_cast<int64_t>(n) + t.baseM - 1) / t.baseM;
    t.nBlocks = (static_cast<int64_t>(m) + t.baseN - 1) / t.baseN;
    t.totalTasks = static_cast<uint64_t>(batches) * t.mBlocks * t.nBlocks;
    t.usedCoreNum = std::min<uint64_t>(cores, t.totalTasks);
    return t;
}

struct HgemmRequest {
    aclblasHandle_t handle;
    aclblasOperation_t transa, transb;
    int m, n, k;
    const aclblasHalf* alpha;
    const aclblasHalf* A;
    int lda;
    int64_t strideA;
    const aclblasHalf* B;
    int ldb;
    int64_t strideB;
    const aclblasHalf* beta;
    aclblasHalf* C;
    int ldc;
    int64_t strideC;
    int batchCount;
    bool ta = false, tb = false;
    float av = 0, bv = 0;
    _aclblas_handle* h = nullptr;
    HgemmBatchEpilogue e{};
    uint32_t aic = 0, aiv = 0;
    int64_t tempLd = 0;
    size_t colBytes = 0, available = 0;
    uint8_t* workspace = nullptr;
    int strip = 0, group = 0;

    aclblasStatus_t ValidateOutputStride() const
    {
        // Nonempty output batches must occupy disjoint column-major spans.
        // int32 dimensions and ld fit in this uint64 calculation, including padding.
        if (m > 0 && n > 0 && batchCount > 1) {
            const uint64_t outputSpan = static_cast<uint64_t>(n - 1) * ldc + m;
            if (static_cast<uint64_t>(strideC) < outputSpan) {
                OP_LOGE(OP_NAME, "output batches overlap");
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t ValidateShape() const
    {
        if (handle == nullptr) {
            OP_LOGE(OP_NAME, "handle is null");
            return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
        }
        if (!ValidOp(transa) || !ValidOp(transb)) {
            OP_LOGE(OP_NAME, "invalid transpose operation");
            return ACLBLAS_STATUS_INVALID_ENUM;
        }
        if (m < 0 || n < 0 || k < 0 || batchCount < 0) {
            OP_LOGE(OP_NAME, "negative shape or batch count");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (lda < std::max(1, ta ? k : m) || ldb < std::max(1, tb ? n : k) || ldc < std::max(1, m) || strideA < 0 ||
            strideB < 0 || strideC < 0 || alpha == nullptr || beta == nullptr) {
            OP_LOGE(OP_NAME, "invalid leading dimension, stride or scalar pointer");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        return ValidateOutputStride();
    }

    aclblasStatus_t ValidateBuffers() const
    {
        if ((k > 0 && (A == nullptr || B == nullptr)) || ((k > 0 || bv != 0) && C == nullptr)) {
            OP_LOGE(OP_NAME, "null storage for a nonempty operation");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (C == nullptr)
            return ACLBLAS_STATUS_SUCCESS; // k=0,beta=0, per task contract
        if (!ValidSpan(C, m, n, ldc, strideC, batchCount) ||
            (k > 0 && (!ValidSpan(A, ta ? k : m, ta ? m : k, lda, strideA, batchCount) ||
                       !ValidSpan(B, tb ? n : k, tb ? k : n, ldb, strideB, batchCount)))) {
            OP_LOGE(OP_NAME, "matrix address span overflows");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        const uint64_t maxElements = std::numeric_limits<int64_t>::max();
        const uint64_t matrixElements = static_cast<uint64_t>(m) * n;
        if (matrixElements > maxElements / batchCount) {
            OP_LOGE(OP_NAME, "logical output element count overflows");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }

        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t ScaleOnly()
    {
        if (bv == 1.0f)
            return ACLBLAS_STATUS_SUCCESS;
        const uint32_t aiv = GetAivCoreCount();
        if (!aiv) {
            OP_LOGE(OP_NAME, "vector core count is zero");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        HgemmBatchCombine(VectorBlocks(e, aiv), h->stream, nullptr, reinterpret_cast<uint8_t*>(C), e);
        return ACLBLAS_STATUS_SUCCESS;
    }

    bool IsGrouped() const
    {
        return m == n && n == k && IsHgemmGroupedSize(m) && av == 1.0f && bv == 0.0f && lda == m && ldb == m &&
               ldc == m && strideA == m * m && strideB == m * m && strideC == m * m &&
               reinterpret_cast<uintptr_t>(A) % HGEMM_DMA_ALIGNMENT_BYTES == 0 &&
               reinterpret_cast<uintptr_t>(B) % HGEMM_DMA_ALIGNMENT_BYTES == 0 &&
               reinterpret_cast<uintptr_t>(C) % HGEMM_DMA_ALIGNMENT_BYTES == 0;
    }

    aclblasStatus_t RunGrouped()
    {
        auto t = MakeTiling(m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchCount, ta, tb, aic);
        HgemmBatchGrouped(
            std::min<uint32_t>(aic, batchCount), h->stream, reinterpret_cast<uint8_t*>(const_cast<aclblasHalf*>(B)),
            reinterpret_cast<uint8_t*>(const_cast<aclblasHalf*>(A)), reinterpret_cast<uint8_t*>(C), t);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunDirect()
    {
        auto t = MakeTiling(m, n, k, lda, ldb, ldc, strideA, strideB, strideC, batchCount, ta, tb, aic);
        if (static_cast<uint64_t>(t.mBlocks) * t.nBlocks > UINT32_MAX) {
            OP_LOGE(OP_NAME, "direct tiling task count overflows");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        HgemmBatchCube(
            t.usedCoreNum, h->stream, reinterpret_cast<uint8_t*>(const_cast<aclblasHalf*>(B)),
            reinterpret_cast<uint8_t*>(const_cast<aclblasHalf*>(A)), reinterpret_cast<uint8_t*>(C), t, true);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t PrepareWorkspace()
    {
        // Bound temporary storage by the caller's effective workspace. Group batches
        // and strip C columns; stream ordering permits safe reuse without allocation.
        tempLd = (static_cast<int64_t>(m) + HGEMM_WORKSPACE_COLUMN_ALIGNMENT - 1) / HGEMM_WORKSPACE_COLUMN_ALIGNMENT *
                 HGEMM_WORKSPACE_COLUMN_ALIGNMENT;
        colBytes = tempLd * sizeof(float);
        if (tempLd > INT32_MAX || colBytes > ACLBLAS_MAX_WORKSPACE_SIZE) {
            OP_LOGE(OP_NAME, "workspace column exceeds supported capacity");
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (h->workspace_owner == AclblasWorkspaceOwner::Library) {
            const uint64_t columns = static_cast<uint64_t>(n) * batchCount;
            const size_t required =
                columns > ACLBLAS_MAX_WORKSPACE_SIZE / colBytes ? ACLBLAS_MAX_WORKSPACE_SIZE : columns * colBytes;
            const aclblasStatus_t status = EnsureDefaultWorkspace(h, required);
            if (status != ACLBLAS_STATUS_SUCCESS) {
                OP_LOGE(OP_NAME, "workspace allocation failed, status=%d", static_cast<int>(status));
                return status;
            }
        }
        // Growth may synchronize and replace the library buffer. Read both fields
        // only after it succeeds; user-owned buffers retain bounded strip/group use.
        available = GetEffectiveWorkspaceSize(h);
        workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
        if (!workspace) {
            OP_LOGE(OP_NAME, "workspace pointer is null");
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        const size_t padding =
            (HGEMM_DMA_ALIGNMENT_BYTES - reinterpret_cast<uintptr_t>(workspace) % HGEMM_DMA_ALIGNMENT_BYTES) %
            HGEMM_DMA_ALIGNMENT_BYTES;
        if (available < padding || reinterpret_cast<uintptr_t>(workspace) > UINTPTR_MAX - padding) {
            OP_LOGE(OP_NAME, "workspace alignment exceeds available address span");
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        workspace += padding;
        available -= padding;
        if (available < colBytes) {
            OP_LOGE(OP_NAME, "workspace cannot hold one aligned output column");
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        strip = std::min<uint64_t>(n, available / colBytes);
        group = std::min<uint64_t>(batchCount, available / (colBytes * strip));
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunWorkspace()
    {
        for (int b0 = 0; b0 < batchCount;) {
            const int count = std::min(group, batchCount - b0);
            for (int c0 = 0; c0 < n;) {
                const int columns = std::min(strip, n - c0);
                const int64_t tempStride = tempLd * columns;
                auto t = MakeTiling(m, columns, k, lda, ldb, tempLd, strideA, strideB, tempStride, count, ta, tb, aic);
                if (static_cast<uint64_t>(t.mBlocks) * t.nBlocks > UINT32_MAX) {
                    OP_LOGE(OP_NAME, "workspace tiling task count overflows");
                    return ACLBLAS_STATUS_INVALID_VALUE;
                }
                const auto* ab = A + static_cast<int64_t>(b0) * strideA;
                const auto* bb = B + static_cast<int64_t>(b0) * strideB + (tb ? c0 : static_cast<int64_t>(c0) * ldb);
                auto* cb = C + static_cast<int64_t>(b0) * strideC + static_cast<int64_t>(c0) * ldc;
                HgemmBatchCube(
                    t.usedCoreNum, h->stream, reinterpret_cast<uint8_t*>(const_cast<aclblasHalf*>(bb)),
                    reinterpret_cast<uint8_t*>(const_cast<aclblasHalf*>(ab)), workspace, t, false);
                e = {m, columns, ldc, static_cast<int32_t>(tempLd), count, strideC, tempStride, av, bv, 0};
                HgemmBatchCombine(VectorBlocks(e, aiv), h->stream, workspace, reinterpret_cast<uint8_t*>(cb), e);
                c0 += columns;
            }
            b0 += count;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t Run()
    {
        ta = transa != ACLBLAS_OP_N;
        tb = transb != ACLBLAS_OP_N;
        auto status = ValidateShape();
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
        if (m == 0 || n == 0 || batchCount == 0)
            return ACLBLAS_STATUS_SUCCESS;
        av = blas_common::HalfToFloat(*alpha);
        bv = blas_common::HalfToFloat(*beta);
        status = ValidateBuffers();
        if (status != ACLBLAS_STATUS_SUCCESS || C == nullptr)
            return status;
        h = reinterpret_cast<_aclblas_handle*>(handle);
        e = {m, n, ldc, 0, batchCount, strideC, 0, av, bv, 1};
        if (k == 0 || av == 0.0f)
            return ScaleOnly();
        aic = GetAicCoreCount();
        aiv = GetAivCoreCount();
        if (!aic || !aiv) {
            OP_LOGE(OP_NAME, "cube or vector core count is zero");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        if (IsGrouped())
            return RunGrouped();
        // Exact alpha=1,beta=0 can write half results without workspace.
        if (av == 1.0f && bv == 0.0f)
            return RunDirect();
        status = PrepareWorkspace();
        return status == ACLBLAS_STATUS_SUCCESS ? RunWorkspace() : status;
    }
};
} // namespace

extern "C" aclblasStatus_t aclblasHgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasHalf* alpha, const aclblasHalf* A, int lda, int64_t strideA, const aclblasHalf* B, int ldb,
    int64_t strideB, const aclblasHalf* beta, aclblasHalf* C, int ldc, int64_t strideC, int batchCount)
{
    HgemmRequest request{handle,  transa, transb, m,       n,    k, alpha, A,       lda,
                         strideA, B,      ldb,    strideB, beta, C, ldc,   strideC, batchCount};
    return request.Run();
}
