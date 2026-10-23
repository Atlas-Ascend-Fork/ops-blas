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
 * \file aclblaslt_matmul_desc.cpp
 * \brief Public C API: matmul compute descriptor CRUD.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_layout_impl.h"
#include "host_utils.h"

#include <cstdint>
#include <new>

namespace {

aclblasStatus_t InitMatmulDescCapsule(
    aclblasLtMatmulDescOpaque_t* capsule, aclblasComputeType_t computeType, aclDataType scaleType)
{
    aclblasLtMatmulDescImpl impl;
    impl.magic = ACLBLASLT_DESC_MAGIC;
    impl.computeType = computeType;
    impl.scaleType = scaleType;

    static_assert(sizeof(impl) <= sizeof(*capsule), "aclblasLtMatmulDescImpl too large, not fit in capsule!");
    return PackImplIntoCapsule(capsule, sizeof(*capsule), &impl, sizeof(impl));
}

// ---- SetAttribute per-attribute handlers ----
// Each handler owns its own size validation and copy so the dispatcher switch stays branch-light,
// keeping aclblasLtMatmulDescSetAttribute() under the cyclomatic-complexity / function-length limits.

aclblasStatus_t SetDescEpilogue(aclblasLtMatmulDescImpl& impl, const void* buf, size_t sizeInBytes)
{
    if (sizeInBytes != sizeof(aclblasLtEpilogue_t) && sizeInBytes != sizeof(uint32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    uint32_t v = 0;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(uint32_t));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    impl.epilogue = static_cast<aclblasLtEpilogue_t>(v);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t SetDescBiasPointer(aclblasLtMatmulDescImpl& impl, const void* buf, size_t sizeInBytes)
{
    if (sizeInBytes != BIAS_PTR_STORAGE_BYTES) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    void* biasPtr = nullptr;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&biasPtr, BIAS_PTR_STORAGE_BYTES, buf, BIAS_PTR_STORAGE_BYTES);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    impl.bias = biasPtr;
    return ACLBLAS_STATUS_SUCCESS;
}

// Read a strict int32 payload and store it into an enum-typed field (transA/transB/biasDataType).
template <typename E>
aclblasStatus_t SetDescEnumFromI32(E& field, const void* buf, size_t sizeInBytes)
{
    if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int32_t v = 0;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(int32_t));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    field = static_cast<E>(v);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t SetDescScalePointer(const void*& field, const void* buf, size_t sizeInBytes)
{
    if (sizeInBytes != sizeof(void*)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const void* scalePtr = nullptr;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&scalePtr, sizeof(void*), buf, sizeof(void*));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    field = scalePtr;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t SetDescPointerMode(aclblasLtMatmulDescImpl& impl, const void* buf, size_t sizeInBytes)
{
    if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int32_t mode = 0;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&mode, sizeof(mode), buf, sizeof(mode));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (mode != ACLBLASLT_POINTER_MODE_HOST && mode != ACLBLASLT_POINTER_MODE_DEVICE) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    impl.pointerMode = mode;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ApplyMatmulDescSetAttr(
    aclblasLtMatmulDescImpl& impl, aclblasLtMatmulDescAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    switch (attr) {
        case ACLBLASLT_MATMUL_DESC_EPILOGUE:
            return SetDescEpilogue(impl, buf, sizeInBytes);
        case ACLBLASLT_MATMUL_DESC_BIAS_POINTER:
            return SetDescBiasPointer(impl, buf, sizeInBytes);
        case ACLBLASLT_MATMUL_DESC_TRANSA:
            return SetDescEnumFromI32(impl.transA, buf, sizeInBytes);
        case ACLBLASLT_MATMUL_DESC_TRANSB:
            return SetDescEnumFromI32(impl.transB, buf, sizeInBytes);
        case ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
            return SetDescEnumFromI32(impl.biasDataType, buf, sizeInBytes);
        case ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER:
            return SetDescScalePointer(impl.scaleA, buf, sizeInBytes);
        case ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER:
            return SetDescScalePointer(impl.scaleB, buf, sizeInBytes);
        case ACLBLASLT_MATMUL_DESC_POINTER_MODE:
            return SetDescPointerMode(impl, buf, sizeInBytes);
        case ACLBLASLT_MATMUL_DESC_A_SCALE_MODE:
        case ACLBLASLT_MATMUL_DESC_B_SCALE_MODE:
            return ACLBLAS_STATUS_SUCCESS;
        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
}

} // namespace

extern "C" {

aclblasStatus_t aclblasLtMatmulDescInit(
    aclblasLtMatmulDesc_t matmulDesc, aclblasComputeType_t computeType, aclDataType scaleType)
{
    if (matmulDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return InitMatmulDescCapsule(matmulDesc, computeType, scaleType);
}

aclblasStatus_t aclblasLtMatmulDescCreate(
    aclblasLtMatmulDesc_t* desc, aclblasComputeType_t computeType, aclDataType scaleType)
{
    if (desc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *desc = nullptr;

    auto* capsule = new (std::nothrow) aclblasLtMatmulDescOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t st = InitMatmulDescCapsule(capsule, computeType, scaleType);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return st;
    }

    *desc = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulDescDestroy(const aclblasLtMatmulDesc_t desc)
{
    if (desc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* capsule = reinterpret_cast<aclblasLtMatmulDescOpaque_t*>(desc);
    delete capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatmulDescSetAttribute(
    aclblasLtMatmulDesc_t desc, aclblasLtMatmulDescAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    if (desc == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulDescImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), desc, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    const aclblasStatus_t setStatus = ApplyMatmulDescSetAttr(impl, attr, buf, sizeInBytes);
    if (setStatus != ACLBLAS_STATUS_SUCCESS) {
        return setStatus;
    }

    return PackImplIntoCapsule(desc, sizeof(*desc), &impl, sizeof(impl));
}

aclblasStatus_t aclblasLtMatmulDescGetAttribute(
    aclblasLtMatmulDesc_t desc, aclblasLtMatmulDescAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten)
{
    if (desc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatmulDescImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), desc, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    size_t requiredSize = 0;
    const void* srcPtr = nullptr;

    switch (attr) {
        case ACLBLASLT_MATMUL_DESC_EPILOGUE:
            requiredSize = sizeof(impl.epilogue);
            srcPtr = &impl.epilogue;
            break;

        case ACLBLASLT_MATMUL_DESC_BIAS_POINTER:
            requiredSize = BIAS_PTR_STORAGE_BYTES;
            srcPtr = &impl.bias;
            break;

        case ACLBLASLT_MATMUL_DESC_TRANSA:
            requiredSize = sizeof(impl.transA);
            srcPtr = &impl.transA;
            break;

        case ACLBLASLT_MATMUL_DESC_TRANSB:
            requiredSize = sizeof(impl.transB);
            srcPtr = &impl.transB;
            break;

        case ACLBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
            requiredSize = sizeof(impl.biasDataType);
            srcPtr = &impl.biasDataType;
            break;

        case ACLBLASLT_MATMUL_DESC_A_SCALE_POINTER:
            requiredSize = sizeof(impl.scaleA);
            srcPtr = &impl.scaleA;
            break;

        case ACLBLASLT_MATMUL_DESC_B_SCALE_POINTER:
            requiredSize = sizeof(impl.scaleB);
            srcPtr = &impl.scaleB;
            break;

        case ACLBLASLT_MATMUL_DESC_POINTER_MODE:
            requiredSize = sizeof(impl.pointerMode);
            srcPtr = &impl.pointerMode;
            break;

        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    return CopyAttributeOut(buf, sizeInBytes, srcPtr, requiredSize, sizeWritten);
}

} // extern "C"
