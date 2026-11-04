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
 * \file host_utils.h
 * \brief Shared host-side utility functions for blasLt.
 */

#pragma once

#include "aclblaslt_layout_impl.h"
#include "cann_ops_blasLt.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <securec.h>
#include <type_traits>

inline aclblasStatus_t CheckedMemcpyS(void* dest, size_t destMax, const void* src, size_t count)
{
    if (memcpy_s(dest, destMax, src, count) != EOK) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CheckedMemsetS(void* dest, size_t destMax, size_t count)
{
    if (memset_s(dest, destMax, 0, count) != EOK) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool MemcpySSucceeds(void* dest, size_t destMax, const void* src, size_t count)
{
    return memcpy_s(dest, destMax, src, count) == EOK;
}

// Copy implBytes from impl into capsule (capacity capsuleBytes) and zero the trailing reserved bytes.
// Shared by every descriptor Create/Init/SetAttribute path to avoid re-implementing the pack step.
inline aclblasStatus_t PackImplIntoCapsule(void* capsule, size_t capsuleBytes, const void* impl, size_t implBytes)
{
    aclblasStatus_t copyStatus = CheckedMemcpyS(capsule, capsuleBytes, impl, implBytes);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (capsuleBytes > implBytes) {
        copyStatus = CheckedMemsetS(
            reinterpret_cast<char*>(capsule) + implBytes, capsuleBytes - implBytes, capsuleBytes - implBytes);
    }
    return copyStatus;
}

// Shared GetAttribute out-parameter tail: validate the caller buffer size, copy requiredSize bytes
// from src, and report the written size. On undersized buffers, sizeWritten still reports requiredSize.
inline aclblasStatus_t CopyAttributeOut(
    void* buf, size_t sizeInBytes, const void* src, size_t requiredSize, size_t* sizeWritten)
{
    if (sizeWritten != nullptr) {
        *sizeWritten = requiredSize;
    }
    // Probe mode: sizeInBytes == 0, user queries required buffer size.
    if (sizeInBytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // Error: buf is null with nonzero size, or buffer too small.
    if (buf == nullptr || sizeInBytes < requiredSize) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return CheckedMemcpyS(buf, sizeInBytes, src, requiredSize);
}

template <typename T>
inline aclblasStatus_t AllocHandle(T** out)
{
    if (out == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *out = nullptr;
    T* p = new (std::nothrow) T();
    if (p == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    *out = p;
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
inline aclblasStatus_t FreeHandle(T* p)
{
    if (p == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    delete p;
    return ACLBLAS_STATUS_SUCCESS;
}

inline size_t GetTypeSize(aclDataType dt)
{
    switch (dt) {
        case ACL_FLOAT16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
            return 4;
        case ACL_INT8:
            return 1;
        default:
            return 0;
    }
}

inline bool IsDataTypeSupported(aclDataType dt)
{
    return GetTypeSize(dt) != 0;
}

inline bool IsMxfp8Type(aclDataType dt)
{
    return dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2;
}

inline bool IsMxfp4Type(aclDataType dt)
{
    return dt == ACL_FLOAT4_E2M1;
}

inline bool CheckComputeTypeCompatibility(aclblasComputeType_t ct, aclDataType typeA, aclDataType typeB)
{
    if (typeA == ACL_FLOAT && typeB == ACL_FLOAT) {
        return ct == ACLBLAS_COMPUTE_32F || ct == ACLBLAS_COMPUTE_32F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F_FAST_TF32;
    }
    if (IsMxfp8Type(typeA) && IsMxfp8Type(typeB)) {
        return ct == ACLBLAS_COMPUTE_32F;
    }
    if (IsMxfp4Type(typeA) && IsMxfp4Type(typeB)) {
        return ct == ACLBLAS_COMPUTE_32F;
    }
    if (typeA == ACL_FLOAT16 && typeB == ACL_FLOAT16) {
        return ct == ACLBLAS_COMPUTE_16F || ct == ACLBLAS_COMPUTE_16F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F ||
               ct == ACLBLAS_COMPUTE_32F_PEDANTIC || ct == ACLBLAS_COMPUTE_32F_FAST_16F;
    }
    if (typeA == ACL_INT8 && typeB == ACL_INT8) {
        return ct == ACLBLAS_COMPUTE_32I || ct == ACLBLAS_COMPUTE_32I_PEDANTIC;
    }
    return false;
}

template <typename T1, typename T2>
inline uint64_t CeilDivU64(T1 a, T2 b)
{
    const uint64_t ua = static_cast<uint64_t>(a);
    const uint64_t ub = static_cast<uint64_t>(b);
    if (ua == 0U || ub == 0U) {
        return 0U;
    }
    return (ua - 1U) / ub + 1U;
}

template <typename R = uint32_t, typename T1, typename T2>
inline R CeilDiv(T1 a, T2 b)
{
    static_assert(std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
                  "CeilDiv arguments must be arithmetic types");
    static_assert(std::is_arithmetic<R>::value, "CeilDiv return type must be arithmetic");
    return static_cast<R>(CeilDivU64(a, b));
}

template <typename R = uint32_t, typename T1, typename T2>
inline R CeilAlign(T1 val, T2 align)
{
    return CeilDiv<R>(val, align) * static_cast<R>(align);
}

// Host-side matmul workload validation predicates shared by aclblasLtMatmul() and
// aclblasLtMatmulAlgoCheck() (these were previously ~110 lines of file-local copies
// duplicated across the two translation units and kept in sync by hand via "Mirror of
// <file>:<line>" annotations -- one anchor had already drifted within a single commit).
// Bodies moved verbatim from the former aclblaslt_matmul_algo.cpp mirrors.

inline bool IsRepresentableAsUint32(uint64_t value) { return value <= std::numeric_limits<uint32_t>::max(); }

inline bool IsSupportedPointerMode(int32_t pointerMode)
{
    return pointerMode == ACLBLASLT_POINTER_MODE_HOST || pointerMode == ACLBLASLT_POINTER_MODE_DEVICE;
}

// 校验链暂不接受 ACLBLAS_OP_C。内核将 trans 编码为布尔（N/非 N），
// OP_C 与 OP_T 走同一路径且无共轭实现，放行会静默返回 A^T 而非 A^H；
// 待补齐共轭转置路径后再放开。
inline bool HasSupportedTransposeModes(const aclblasLtMatmulDescImpl* desc)
{
    const bool transAValid = desc->transA == ACLBLAS_OP_N || desc->transA == ACLBLAS_OP_T;
    const bool transBValid = desc->transB == ACLBLAS_OP_N || desc->transB == ACLBLAS_OP_T;
    return transAValid && transBValid;
}

// Details: exact rows/cols equality, negative ld rejected, uint32 representability of ld,
// ld == 0 allowed through, minLd picked per layout->order (ROW -> cols, otherwise rows), and the
// half-byte fold minLd = minLd/2 + minLd%2 for ACL_FLOAT4_E2M1.
inline bool IsLayoutCompatible(const aclblasLtMatrixLayoutImpl* layout, uint64_t rows, uint64_t cols)
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

inline bool HasSingleNonStridedBatch(
    const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b, const aclblasLtMatrixLayoutImpl* c,
    const aclblasLtMatrixLayoutImpl* d)
{
    return a->batchCount == 1 && b->batchCount == 1 && c->batchCount == 1 && d->batchCount == 1 &&
           a->stridedBatchOffset == 0 && b->stridedBatchOffset == 0 && c->stridedBatchOffset == 0 &&
           d->stridedBatchOffset == 0;
}

inline bool AreOperandLayoutsCompatible(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b,
    uint64_t m, uint64_t n, uint64_t k)
{
    const uint64_t aRows = (desc->transA == ACLBLAS_OP_N) ? m : k;
    const uint64_t aCols = (desc->transA == ACLBLAS_OP_N) ? k : m;
    const uint64_t bRows = (desc->transB == ACLBLAS_OP_N) ? k : n;
    const uint64_t bCols = (desc->transB == ACLBLAS_OP_N) ? n : k;
    return IsLayoutCompatible(a, aRows, aCols) && IsLayoutCompatible(b, bRows, bCols);
}

inline bool AreOutputLayoutsCompatible(
    const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d, uint64_t m, uint64_t n)
{
    return IsLayoutCompatible(c, m, n) && IsLayoutCompatible(d, m, n);
}

// uint32 overflow first, then batch metadata, then operand/output layout shapes.
inline bool IsMatmulLayoutValid(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b,
    const aclblasLtMatrixLayoutImpl* c, const aclblasLtMatrixLayoutImpl* d, uint64_t m, uint64_t n, uint64_t k)
{
    if (!IsRepresentableAsUint32(m) || !IsRepresentableAsUint32(n) || !IsRepresentableAsUint32(k)) {
        return false;
    }
    return HasSingleNonStridedBatch(a, b, c, d) && AreOperandLayoutsCompatible(desc, a, b, m, n, k) &&
           AreOutputLayoutsCompatible(c, d, m, n);
}

// The matmul kernels interpret matrices in row-major order only.
// The layout descriptor defaults to ACLBLASLT_ORDER_COL, so an unset or
// unsupported order must be rejected explicitly instead of being silently
// interpreted as row-major (which would compute on misinterpreted memory).
inline bool HasRowMajorLayouts(
    const aclblasLtMatrixLayoutImpl* a, const aclblasLtMatrixLayoutImpl* b, const aclblasLtMatrixLayoutImpl* c,
    const aclblasLtMatrixLayoutImpl* d)
{
    return a->order == ACLBLASLT_ORDER_ROW && b->order == ACLBLASLT_ORDER_ROW && c->order == ACLBLASLT_ORDER_ROW &&
           d->order == ACLBLASLT_ORDER_ROW;
}

// Verdict of the shared C-dtype / MXFP output-dtype rules (the common prefix of
// aclblasLtMatmul()'s HasSupportedDtypes and AlgoCheck's HasSupportedDtypesForAlgoCheck;
// extracted to remove duplicate code). kNotMxPath: A/B is not an MXFP pair and the
// caller applies its own A/B-path rules -- the FP32 branch intentionally diverges (matmul sees
// runtime alpha/beta; AlgoCheck applies the strict pointer-mode rule).
enum class AclblasMxDtypeRule {
    kSupported,
    kUnsupported,
    kNotMxPath,
};

// C dtype must be FP32/BF16 (epilogue reads C); D per A/B path: MXFP8 -> BF16 only,
// MXFP4 -> {BF16, FP32}. See the callers for the full rationale.
inline AclblasMxDtypeRule CheckCAndMxDtypeRules(
    aclDataType dtypeA, aclDataType dtypeB, aclDataType dtypeC, aclDataType dtypeD)
{
    if (dtypeC != ACL_FLOAT && dtypeC != ACL_BF16) {
        return AclblasMxDtypeRule::kUnsupported;
    }
    if (IsMxfp8Type(dtypeA) && IsMxfp8Type(dtypeB)) {
        return (dtypeD == ACL_BF16) ? AclblasMxDtypeRule::kSupported : AclblasMxDtypeRule::kUnsupported;
    }
    if (IsMxfp4Type(dtypeA) && IsMxfp4Type(dtypeB)) {
        return (dtypeD == ACL_BF16 || dtypeD == ACL_FLOAT) ? AclblasMxDtypeRule::kSupported
                                                           : AclblasMxDtypeRule::kUnsupported;
    }
    return AclblasMxDtypeRule::kNotMxPath;
}
