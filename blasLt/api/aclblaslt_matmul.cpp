/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclblaslt_matmul.cpp
 * \brief Public C API: aclblasLtMatmul — validates inputs, builds the MatmulProblem, dispatches to
 *        the matmul engine.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_handle_impl.h"
#include "aclblaslt_layout_impl.h"
#include "aclblaslt_logger_impl.h"
#include "aclblaslt_matmul_problem.h"
#include "matmul_engine.h"

#include <cstdint>
#include <limits>

namespace {

bool IsRepresentableAsUint32(uint64_t value)
{
    return value <= std::numeric_limits<uint32_t>::max();
}

bool IsLayoutCompatible(const aclblasLtMatrixLayoutImpl* layout, uint64_t rows, uint64_t cols)
{
    if (layout == nullptr || layout->rows != rows || layout->cols != cols || layout->ld < 0) {
        return false;
    }
    const uint64_t ld = static_cast<uint64_t>(layout->ld);
    if (!IsRepresentableAsUint32(ld)) {
        return false;
    }
    if (ld == 0) {
        return true;
    }
    uint64_t minLd = (layout->order == ACLBLASLT_ORDER_ROW) ? cols : rows;
    if (layout->type == ACL_FLOAT4_E2M1) {
        minLd = minLd / 2U + minLd % 2U;
    }
    return ld >= minLd;
}

bool HasSupportedTransposeModes(const aclblasLtMatmulDescImpl* desc)
{
    const bool transAValid =
        desc->transA == ACLBLAS_OP_N || desc->transA == ACLBLAS_OP_T || desc->transA == ACLBLAS_OP_C;
    const bool transBValid =
        desc->transB == ACLBLAS_OP_N || desc->transB == ACLBLAS_OP_T || desc->transB == ACLBLAS_OP_C;
    return transAValid && transBValid;
}

bool HasSingleNonStridedBatch(const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b,
    const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d)
{
    return a->batchCount == 1 && b->batchCount == 1 && c->batchCount == 1 && d->batchCount == 1 &&
           a->stridedBatchOffset == 0 && b->stridedBatchOffset == 0 && c->stridedBatchOffset == 0 &&
           d->stridedBatchOffset == 0;
}

bool AreOperandLayoutsCompatible(const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a,
    const aclblasLtMatrixLayoutImpl* b, uint64_t m, uint64_t n, uint64_t k)
{
    const uint64_t aRows = (desc->transA == ACLBLAS_OP_N) ? m : k;
    const uint64_t aCols = (desc->transA == ACLBLAS_OP_N) ? k : m;
    const uint64_t bRows = (desc->transB == ACLBLAS_OP_N) ? k : n;
    const uint64_t bCols = (desc->transB == ACLBLAS_OP_N) ? n : k;
    return IsLayoutCompatible(a, aRows, aCols) && IsLayoutCompatible(b, bRows, bCols);
}

bool AreOutputLayoutsCompatible(const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d,
    uint64_t m, uint64_t n)
{
    return IsLayoutCompatible(c, m, n) && IsLayoutCompatible(d, m, n);
}

bool IsMatmulLayoutValid(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b,
    const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d, uint64_t m, uint64_t n, uint64_t k)
{
    if (!IsRepresentableAsUint32(m) || !IsRepresentableAsUint32(n) || !IsRepresentableAsUint32(k)) {
        return false;
    }
    return HasSupportedTransposeModes(desc) && HasSingleNonStridedBatch(a, b, c, d) &&
           AreOperandLayoutsCompatible(desc, a, b, m, n, k) && AreOutputLayoutsCompatible(c, d, m, n);
}

} // namespace

namespace {

bool IsSupportedPointerMode(int32_t pointerMode)
{
    return pointerMode == ACLBLASLT_POINTER_MODE_HOST || pointerMode == ACLBLASLT_POINTER_MODE_DEVICE;
}

} // namespace

// issue #492: alpha/beta pointer mode values (matching aclblasLtPointerMode_t).
static constexpr int32_t ACLBLASLT_POINTER_MODE_DEVICE_VECTOR = 2;

extern "C" {

aclblasStatus_t aclblasLtMatmul(
    aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t computeDesc, const void* alpha, const void* A,
    aclblasLtMatrixLayout_t Adesc, const void* B, aclblasLtMatrixLayout_t Bdesc, const void* beta, const void* C,
    aclblasLtMatrixLayout_t Cdesc, void* D, aclblasLtMatrixLayout_t Ddesc, const aclblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes, aclrtStream stream)
{
    // Validate lightHandle
    if (lightHandle == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "lightHandle is null");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    // Validate descriptors
    if (computeDesc == nullptr || Adesc == nullptr || Bdesc == nullptr || Cdesc == nullptr || Ddesc == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "one or more descriptors are null");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Validate pointers
    if (alpha == nullptr || beta == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "alpha or beta is null");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Get layout info
    auto* ALayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
    auto* BLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
    auto* CLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Cdesc);
    auto* DLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Ddesc);
    auto* desc = reinterpret_cast<aclblasLtMatmulDescImpl*>(computeDesc);
    if (!IsSupportedPointerMode(desc->pointerMode)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Get dimensions
    uint64_t m = DLayout->rows;
    uint64_t n = DLayout->cols;
    uint64_t k = (desc->transA == ACLBLAS_OP_N) ? ALayout->cols : ALayout->rows;

    AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_TRACE, "aclblasLtMatmul",
        "m=%lu, n=%lu, k=%lu", static_cast<unsigned long>(m), static_cast<unsigned long>(n),
        static_cast<unsigned long>(k));

    // BLAS/cuBLAS convention: m=0 or n=0 is a no-op, succeed without touching matrices.
    if (m == 0U || n == 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (!IsMatmulLayoutValid(desc, ALayout, BLayout, CLayout, DLayout, m, n, k)) {
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "matrix dimensions, leading dimensions, batch metadata, or integer ranges are invalid");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (A == nullptr || B == nullptr || D == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "A, B, or D pointer is null");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // issue #491: the matmul kernels interpret matrices in row-major order only.
    // The layout descriptor defaults to ACLBLASLT_ORDER_COL, so an unset or
    // unsupported order must be rejected explicitly instead of being silently
    // interpreted as row-major (which would compute on misinterpreted memory).
    const bool rowOrder = ALayout->order == ACLBLASLT_ORDER_ROW && BLayout->order == ACLBLASLT_ORDER_ROW &&
                          CLayout->order == ACLBLASLT_ORDER_ROW && DLayout->order == ACLBLASLT_ORDER_ROW;
    if (!rowOrder) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "only ACLBLASLT_ORDER_ROW layouts are supported (A=%d, B=%d, C=%d, D=%d)", static_cast<int>(ALayout->order),
            static_cast<int>(BLayout->order), static_cast<int>(CLayout->order), static_cast<int>(DLayout->order));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    // Validate workspace alignment (must be 16B aligned)
    if (workspace != nullptr && (reinterpret_cast<uintptr_t>(workspace) & 0xF) != 0) {
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "workspace is not 16B aligned");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Validate workspace size
    if (algo != nullptr && workspaceSizeInBytes < algo->max_workspace_bytes) {
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "workspaceSizeInBytes (%zu) < algo->max_workspace_bytes (%zu)",
            workspaceSizeInBytes, algo->max_workspace_bytes);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* handleImpl = lightHandle;
    const uint32_t numBlocks = QueryCubeCoreNum(handleImpl->deviceId);

    // issue #492: alpha/beta are passed through unchanged for both supported modes:
    // HOST pointers are dereferenced on host by the engine, DEVICE scalars are read by
    // the arch35 epilogue kernel on device (a host-side D2H copy-back would hand the
    // engine a host stack address as a GM pointer). Unsupported modes were already
    // rejected by the IsSupportedPointerMode check above.
    const void* alphaArg = alpha;
    const void* betaArg = beta;

    MatmulProblem problem;
    BuildMatmulProblem(
        desc, ALayout, BLayout, CLayout, DLayout, m, n, k, alphaArg, betaArg, A, B, C, D, algo, workspace,
        workspaceSizeInBytes, numBlocks, stream, problem);

    return MatmulLaunch(problem);
}

} // extern "C"
