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

#include <acl/acl.h>
#include <cstdint>

// issue #492: alpha/beta pointer mode values (matching aclblasLtPointerMode_t).
static constexpr int32_t ACLBLASLT_POINTER_MODE_HOST = 0;
static constexpr int32_t ACLBLASLT_POINTER_MODE_DEVICE = 1;
static constexpr int32_t ACLBLASLT_POINTER_MODE_DEVICE_VECTOR = 2;

// issue #492: copy one float scalar from a device pointer back to host. Uses the
// caller stream (async + sync) so ordering with preceding work is preserved;
// falls back to the synchronous API when no stream is provided.
static aclblasStatus_t ReadScalarFromDevice(const void* devicePtr, float& hostVal, aclrtStream stream)
{
    hostVal = 0.0f;
    aclError aclRet;
    if (stream != nullptr) {
        aclRet = aclrtMemcpyAsync(&hostVal, sizeof(float), devicePtr, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (aclRet == ACL_SUCCESS) {
            aclRet = aclrtSynchronizeStream(stream);
        }
    } else {
        aclRet = aclrtMemcpy(&hostVal, sizeof(float), devicePtr, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (aclRet != ACL_SUCCESS) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "alpha/beta scalar D2H copy failed, ret=%d",
            static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

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
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "one or more descriptors are null");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Validate pointers
    if (alpha == nullptr || beta == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "alpha or beta is null");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Get layout info
    auto* ALayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
    auto* BLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
    auto* CLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Cdesc);
    auto* DLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Ddesc);
    auto* desc = reinterpret_cast<aclblasLtMatmulDescImpl*>(computeDesc);

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

    // issue #492: resolve alpha/beta according to the descriptor pointer mode.
    // HOST (default) dereferences directly; DEVICE scalars must be copied back
    // to host through the stream — the host must never dereference a device
    // address; DEVICE_VECTOR (one scalar per batch) is not supported here.
    const void* alphaArg = alpha;
    const void* betaArg = beta;
    float alphaDeviceValue = 0.0f;
    float betaDeviceValue = 0.0f;
    if (desc->pointerMode == ACLBLASLT_POINTER_MODE_HOST) {
        // pass through: alpha/beta are host pointers
    } else if (desc->pointerMode == ACLBLASLT_POINTER_MODE_DEVICE) {
        aclblasStatus_t readSt = ReadScalarFromDevice(alpha, alphaDeviceValue, stream);
        if (readSt != ACLBLAS_STATUS_SUCCESS) {
            return readSt;
        }
        readSt = ReadScalarFromDevice(beta, betaDeviceValue, stream);
        if (readSt != ACLBLAS_STATUS_SUCCESS) {
            return readSt;
        }
        alphaArg = &alphaDeviceValue;
        betaArg = &betaDeviceValue;
    } else if (desc->pointerMode == ACLBLASLT_POINTER_MODE_DEVICE_VECTOR) {
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "device vector pointer mode is not supported");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    } else {
        AclBlasLt::LoggerManager::GetInstance().Log(ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "invalid pointer mode %d", static_cast<int>(desc->pointerMode));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    MatmulProblem problem;
    BuildMatmulProblem(
        desc, ALayout, BLayout, CLayout, DLayout, m, n, k, alphaArg, betaArg, A, B, C, D, algo, workspace,
        workspaceSizeInBytes, numBlocks, stream, problem);

    return MatmulLaunch(problem);
}

} // extern "C"
