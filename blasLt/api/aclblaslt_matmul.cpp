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
#include "host_utils.h"
#include "matmul_engine.h"

#include <cstdint>
#include <limits>

namespace {

bool IsRepresentableAsUint32(uint64_t value) { return value <= std::numeric_limits<uint32_t>::max(); }

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

// issue #595: 校验链暂不接受 ACLBLAS_OP_C。内核将 trans 编码为布尔（N/非 N），
// OP_C 与 OP_T 走同一路径且无共轭实现，放行会静默返回 A^T 而非 A^H；
// 待补齐共轭转置路径后再放开。
bool HasSupportedTransposeModes(const aclblasLtMatmulDescImpl* desc)
{
    const bool transAValid = desc->transA == ACLBLAS_OP_N || desc->transA == ACLBLAS_OP_T;
    const bool transBValid = desc->transB == ACLBLAS_OP_N || desc->transB == ACLBLAS_OP_T;
    return transAValid && transBValid;
}

bool HasSingleNonStridedBatch(
    const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b, const aclblasLtMatrixLayoutImpl* c,
    const aclblasLtMatrixLayoutImpl* d)
{
    return a->batchCount == 1 && b->batchCount == 1 && c->batchCount == 1 && d->batchCount == 1 &&
           a->stridedBatchOffset == 0 && b->stridedBatchOffset == 0 && c->stridedBatchOffset == 0 &&
           d->stridedBatchOffset == 0;
}

bool AreOperandLayoutsCompatible(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b,
    uint64_t m, uint64_t n, uint64_t k)
{
    const uint64_t aRows = (desc->transA == ACLBLAS_OP_N) ? m : k;
    const uint64_t aCols = (desc->transA == ACLBLAS_OP_N) ? k : m;
    const uint64_t bRows = (desc->transB == ACLBLAS_OP_N) ? k : n;
    const uint64_t bCols = (desc->transB == ACLBLAS_OP_N) ? n : k;
    return IsLayoutCompatible(a, aRows, aCols) && IsLayoutCompatible(b, bRows, bCols);
}

bool AreOutputLayoutsCompatible(
    const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d, uint64_t m, uint64_t n)
{
    return IsLayoutCompatible(c, m, n) && IsLayoutCompatible(d, m, n);
}

// issue #582: 内核写出宽度由 A/B dtype 路径决定，C/D dtype 必须落在对应路径实际支持的白名单内，
// 否则设备端按错误元素宽度读写 D 造成越界或静默错误结果：
//   FP32 路径  ：MMAD 按 float 写原始乘积；HOST 模式 alpha=1 且 beta=0 时不启动 epilogue，
//                D 必须与原始乘积同宽（FP32），其余场景 epilogue 支持 FP32/BF16 输出；
//   MXFP8 路径 ：内核固定按 bfloat16 写出，D 须为 BF16；
//   MXFP4 路径 ：内核按 D dtype 选择 bf16/float 实例，D 须为二者之一。
// C 由 epilogue 读取，仅支持 FP32/BF16 解释。
bool HasSupportedDtypes(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b,
    const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d, const void* alpha, const void* beta)
{
    const aclDataType dtypeA = a->type;
    const aclDataType dtypeB = b->type;
    const aclDataType dtypeC = c->type;
    const aclDataType dtypeD = d->type;
    if (dtypeC != ACL_FLOAT && dtypeC != ACL_BF16) {
        return false;
    }
    if (IsMxfp8Type(dtypeA) && IsMxfp8Type(dtypeB)) {
        return dtypeD == ACL_BF16;
    }
    if (IsMxfp4Type(dtypeA) && IsMxfp4Type(dtypeB)) {
        return dtypeD == ACL_BF16 || dtypeD == ACL_FLOAT;
    }
    if (dtypeA == ACL_FLOAT && dtypeB == ACL_FLOAT) {
        const bool epilogueRuns = desc->pointerMode == ACLBLASLT_POINTER_MODE_DEVICE ||
                                  (*static_cast<const float*>(alpha) != 1.0f) ||
                                  (*static_cast<const float*>(beta) != 0.0f);
        return epilogueRuns ? (dtypeD == ACL_FLOAT || dtypeD == ACL_BF16) : dtypeD == ACL_FLOAT;
    }
    return false;
}

bool IsMatmulLayoutValid(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b,
    const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d, uint64_t m, uint64_t n, uint64_t k)
{
    if (!IsRepresentableAsUint32(m) || !IsRepresentableAsUint32(n) || !IsRepresentableAsUint32(k)) {
        return false;
    }
    return HasSingleNonStridedBatch(a, b, c, d) && AreOperandLayoutsCompatible(desc, a, b, m, n, k) &&
           AreOutputLayoutsCompatible(c, d, m, n);
}

} // namespace

namespace {

bool IsSupportedPointerMode(int32_t pointerMode)
{
    return pointerMode == ACLBLASLT_POINTER_MODE_HOST || pointerMode == ACLBLASLT_POINTER_MODE_DEVICE;
}

} // namespace

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

    // issue #594: HOST 指针模式下 alpha/beta 按 float 解引用、DEVICE 模式下 epilogue 内核按
    // float 读取标量，scaleType 非 ACL_FLOAT 时继续会按 4 字节误读 2 字节标量（主机越界读 +
    // 错误系数），此处显式拒绝。
    if (desc->scaleType != ACL_FLOAT) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "scaleType %d is not supported, only ACL_FLOAT scalars are parsed", static_cast<int>(desc->scaleType));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // issue #595: OP_C 无共轭实现，静默按普通转置计算会返回错误结果，显式拒绝。
    if (!HasSupportedTransposeModes(desc)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "transA/transB ACLBLAS_OP_C is not supported (no conjugate path)");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    // Get dimensions
    uint64_t m = DLayout->rows;
    uint64_t n = DLayout->cols;
    uint64_t k = (desc->transA == ACLBLAS_OP_N) ? ALayout->cols : ALayout->rows;

    AclBlasLt::LoggerManager::GetInstance().Log(
        ACLBLASLT_LOG_MASK_TRACE, "aclblasLtMatmul", "m=%lu, n=%lu, k=%lu", static_cast<unsigned long>(m),
        static_cast<unsigned long>(n), static_cast<unsigned long>(k));

    // BLAS/cuBLAS convention: m=0 or n=0 is a no-op, succeed without touching matrices.
    if (m == 0U || n == 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (!IsMatmulLayoutValid(desc, ALayout, BLayout, CLayout, DLayout, m, n, k)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "matrix dimensions, leading dimensions, batch metadata, or integer ranges are invalid");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // issue #582: C/D dtype 白名单校验，内核实际不支持的组合不下发。
    if (!HasSupportedDtypes(desc, ALayout, BLayout, CLayout, DLayout, alpha, beta)) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul",
            "dtype combination A=%d B=%d C=%d D=%d is not supported by any matmul kernel",
            static_cast<int>(ALayout->type), static_cast<int>(BLayout->type), static_cast<int>(CLayout->type),
            static_cast<int>(DLayout->type));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    if (A == nullptr || B == nullptr || D == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "A, B, or D pointer is null");
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
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "workspace is not 16B aligned");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Validate workspace size
    if (algo != nullptr && workspaceSizeInBytes < algo->max_workspace_bytes) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatmul", "workspaceSizeInBytes (%zu) < algo->max_workspace_bytes (%zu)",
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
