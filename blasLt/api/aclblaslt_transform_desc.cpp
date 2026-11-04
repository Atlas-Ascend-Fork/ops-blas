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
#include "aclblaslt_logger_impl.h"

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

// In-place initializer of a caller-allocated transform desc capsule. Mirrors the Create path's
// Impl packing (magic + scaleType + in-place trans), with the NSDMI defaults (pointerMode = host)
// picked up identically, so an Init capsule is byte-equivalent to a Create + SetAttribute one.
aclblasStatus_t InitTransformDescCapsule(aclblasLtMatrixTransformDescOpaque_t* capsule, aclDataType scaleType)
{
    // magic and scaleType have no NSDMI and must be assigned explicitly; pointerMode and
    // transA/transB keep their NSDMI defaults (host / N), byte-equivalent to the Create path.
    aclblasLtMatrixTransformDescImpl impl;
    impl.magic = ACLBLASLT_TRANSFORM_DESC_MAGIC;
    impl.scaleType = scaleType;
    static_assert(sizeof(impl) <= sizeof(*capsule),
                  "aclblasLtMatrixTransformDescImpl too large, not fit in capsule!");
    return PackImplIntoCapsule(capsule, sizeof(*capsule), &impl, sizeof(impl));
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
    impl.heapOwned = 1U;  // heap provenance: this capsule may be Destroy()ed

    aclblasStatus_t copyStatus = MatPackTransformImpl(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }

    *transformDesc = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

// Parameter face mirrors cuBLASLt 13.x cublasLtMatrixTransformDescInit(transformDesc, scaleType):
// by value, two parameters. transa/transb are NOT Init parameters there (nor here) -- they are set
// afterwards via aclblasLtMatrixTransformDescSetAttribute(TRANSA/TRANSB), the same path Create()
// users take. The handle is passed by value like the repo's aclblasLtMatmulDescInit /
// aclblasLtMatmulPreferenceInit (all mirroring cuBLASLt's *Init convention); a null handle is a
// NULL pointer, so a single check covers it.
aclblasStatus_t aclblasLtMatrixTransformDescInit(
    aclblasLtMatrixTransformDesc_t transformDesc, aclDataType scaleType)
{
    if (transformDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return InitTransformDescCapsule(transformDesc, scaleType);
}

aclblasStatus_t aclblasLtMatrixTransformDescDestroy(const aclblasLtMatrixTransformDesc_t transformDesc)
{
    if (transformDesc == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* capsule = reinterpret_cast<aclblasLtMatrixTransformDescOpaque_t*>(transformDesc);
    // Provenance guard: an Init() product lives in caller storage (stack / static / member);
    // deleting such a pointer is UB (heap metadata corruption / SIGABRT) and previously the
    // crash happened only after this function had already returned SUCCESS. The heapOwned flag
    // set by Create() distinguishes the two provenances and round-trips SetAttribute. Also
    // rejects corrupted / foreign capsules via the magic check (mirrors UnpackTransformDesc).
    aclblasLtMatrixTransformDescImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), capsule, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }
    if (impl.magic != ACLBLASLT_TRANSFORM_DESC_MAGIC) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatrixTransformDescDestroy",
            "descriptor failed the magic check (corrupted / foreign capsule)");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (impl.heapOwned == 0U) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatrixTransformDescDestroy",
            "capsule was not allocated by aclblasLtMatrixTransformDescCreate (heapOwned=0); "
            "Init-produced descriptors must not be passed to Destroy");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
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
