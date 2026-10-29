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
 * \file aclblaslt_transform_desc.cpp
 * \brief Public C API: matrix transform descriptor CRUD.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_capsule.h"
#include "host_utils.h"
#include "matrix_transform_engine.h"

#include <cstdint>
#include <new>

namespace {

// Shared Set/Get prologue: reject null args, copy the capsule into a stack impl, and reject a
// corrupted / foreign descriptor via the magic check. Get 路径不要求 buf 非空——探测模式
// （sizeInBytes==0、buf 为空、sizeWritten 回写所需大小）由 CopyAttributeOut 统一处理（issue #664）。
aclblasStatus_t UnpackTransformDesc(
    const aclblasLtMatrixTransformDesc_t transformDesc, const void* buf, aclblasLtMatrixTransformDescImpl& impl,
    bool requireBuf)
{
    if (transformDesc == nullptr || (requireBuf && buf == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), transformDesc, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (impl.magic != ACLBLASLT_TRANSFORM_DESC_MAGIC) {
        return ACLBLAS_STATUS_INVALID_VALUE; // corrupted / foreign descriptor
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" {

aclblasStatus_t aclblasLtMatrixTransformDescCreate(aclblasLtMatrixTransformDesc_t* transformDesc, aclDataType scaleType)
{
    if (transformDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *transformDesc = nullptr;

    auto* capsule = new (std::nothrow) aclblasLtMatrixTransformDescOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasLtMatrixTransformDescImpl impl;
    impl.magic = ACLBLASLT_TRANSFORM_DESC_MAGIC;
    impl.scaleType = scaleType;

    aclblasStatus_t copyStatus = MatPackTransformImpl(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }

    *transformDesc = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixTransformDescDestroy(const aclblasLtMatrixTransformDesc_t transformDesc)
{
    if (transformDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* capsule = reinterpret_cast<aclblasLtMatrixTransformDescOpaque_t*>(transformDesc);
    delete capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixTransformDescSetAttribute(
    aclblasLtMatrixTransformDesc_t transformDesc, aclblasLtMatrixTransformDescAttribute_t attr, const void* buf,
    size_t sizeInBytes)
{
    aclblasLtMatrixTransformDescImpl impl;
    aclblasStatus_t copyStatus = UnpackTransformDesc(transformDesc, buf, impl, true);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (sizeInBytes != sizeof(int32_t)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int32_t v = 0;
    copyStatus = CheckedMemcpyS(&v, sizeof(v), buf, sizeof(int32_t));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    switch (attr) {
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
            impl.scaleType = static_cast<aclDataType>(v);
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
            impl.pointerMode = v;
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
            impl.transA = static_cast<aclblasOperation_t>(v);
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
            impl.transB = static_cast<aclblasOperation_t>(v);
            break;
        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    return MatPackTransformImpl(transformDesc, sizeof(*transformDesc), &impl, sizeof(impl));
}

aclblasStatus_t aclblasLtMatrixTransformDescGetAttribute(
    aclblasLtMatrixTransformDesc_t transformDesc, aclblasLtMatrixTransformDescAttribute_t attr, void* buf,
    size_t sizeInBytes, size_t* sizeWritten)
{
    aclblasLtMatrixTransformDescImpl impl;
    aclblasStatus_t copyStatus = UnpackTransformDesc(transformDesc, buf, impl, false);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    const void* srcPtr = nullptr;
    switch (attr) {
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
            srcPtr = &impl.scaleType;
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
            srcPtr = &impl.pointerMode;
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
            srcPtr = &impl.transA;
            break;
        case ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
            srcPtr = &impl.transB;
            break;
        default:
            return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    const size_t requiredSize = sizeof(int32_t);
    return CopyAttributeOut(buf, sizeInBytes, srcPtr, requiredSize, sizeWritten);
}

} // extern "C"
