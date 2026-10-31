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
 * \file aclblaslt_layout.cpp
 * \brief Public C API: matrix layout descriptor CRUD.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_layout_impl.h"
#include "host_utils.h"

#include <cstdint>
#include <new>

namespace {

// Validate that the caller buffer is exactly sizeof(T), then load it into field. Collapsing the
// per-attribute size check into one helper keeps the Set/Get switches free of nested branches.
template <typename T>
inline aclblasStatus_t LoadLayoutField(const void* buf, size_t sizeInBytes, T& field)
{
    if (sizeInBytes != sizeof(T)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    field = *reinterpret_cast<const T*>(buf);
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
inline aclblasStatus_t StoreLayoutFieldOut(void* buf, size_t sizeInBytes, const T& field, size_t* sizeWritten)
{
    const size_t requiredSize = sizeof(T);
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
    *reinterpret_cast<T*>(buf) = field;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ApplyLayoutSetAttr(
    aclblasLtMatrixLayoutImpl& impl, aclblasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    switch (attr) {
        case ACLBLASLT_MATRIX_LAYOUT_TYPE:
            return LoadLayoutField(buf, sizeInBytes, impl.type);
        case ACLBLASLT_MATRIX_LAYOUT_ROWS:
            return LoadLayoutField(buf, sizeInBytes, impl.rows);
        case ACLBLASLT_MATRIX_LAYOUT_COLS:
            return LoadLayoutField(buf, sizeInBytes, impl.cols);
        case ACLBLASLT_MATRIX_LAYOUT_LD:
            return LoadLayoutField(buf, sizeInBytes, impl.ld);
        case ACLBLASLT_MATRIX_LAYOUT_ORDER:
            return LoadLayoutField(buf, sizeInBytes, impl.order);
        case ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            return LoadLayoutField(buf, sizeInBytes, impl.batchCount);
        case ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            return LoadLayoutField(buf, sizeInBytes, impl.stridedBatchOffset);
        default:
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
}

aclblasStatus_t ReadLayoutGetAttr(
    const aclblasLtMatrixLayoutImpl& impl, aclblasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes,
    size_t* sizeWritten)
{
    switch (attr) {
        case ACLBLASLT_MATRIX_LAYOUT_TYPE:
            return StoreLayoutFieldOut(buf, sizeInBytes, impl.type, sizeWritten);
        case ACLBLASLT_MATRIX_LAYOUT_ROWS:
            return StoreLayoutFieldOut(buf, sizeInBytes, impl.rows, sizeWritten);
        case ACLBLASLT_MATRIX_LAYOUT_COLS:
            return StoreLayoutFieldOut(buf, sizeInBytes, impl.cols, sizeWritten);
        case ACLBLASLT_MATRIX_LAYOUT_LD:
            return StoreLayoutFieldOut(buf, sizeInBytes, impl.ld, sizeWritten);
        case ACLBLASLT_MATRIX_LAYOUT_ORDER:
            return StoreLayoutFieldOut(buf, sizeInBytes, impl.order, sizeWritten);
        case ACLBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            return StoreLayoutFieldOut(buf, sizeInBytes, impl.batchCount, sizeWritten);
        case ACLBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            return StoreLayoutFieldOut(buf, sizeInBytes, impl.stridedBatchOffset, sizeWritten);
        default:
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
}

} // namespace

extern "C" {

aclblasStatus_t aclblasLtMatrixLayoutCreate(
    aclblasLtMatrixLayout_t* layout, aclDataType type, uint64_t rows, uint64_t cols, int64_t ld)
{
    // Step 1: validate inputs. BLAS/cuBLAS permit empty matrices (m=0 or n=0); reject invalid ld only.
    if (layout == nullptr || ld < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *layout = nullptr;
    // Step 2: allocate the opaque capsule on the heap.
    auto* capsule = new (std::nothrow) aclblasLtMatrixLayoutOpaque_t();
    if (capsule == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    // Step 3: build the impl on the stack and copy it into the capsule.
    aclblasLtMatrixLayoutImpl impl;
    impl.magic = ACLBLASLT_LAYOUT_MAGIC;
    impl.type = type;
    impl.rows = rows;
    impl.cols = cols;
    impl.ld = (ld == 0) ? static_cast<int64_t>(rows) : ld;
    // Step 4: copy impl into capsule and zero the remaining bytes.
    static_assert(sizeof(impl) <= sizeof(*capsule), "aclblasLtMatrixLayoutImpl too large, not fit in capsule!");
    aclblasStatus_t copyStatus = PackImplIntoCapsule(capsule, sizeof(*capsule), &impl, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        delete capsule;
        return copyStatus;
    }
    // Step 5: return the opaque layout handle.
    *layout = capsule;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixLayoutDestroy(const aclblasLtMatrixLayout_t layout)
{
    if (layout == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* capsule = reinterpret_cast<aclblasLtMatrixLayoutOpaque_t*>(layout);
    delete capsule;

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtMatrixLayoutSetAttribute(
    aclblasLtMatrixLayout_t layout, aclblasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes)
{
    if (layout == nullptr || buf == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Unpack the opaque capsule into a stack-side impl for mutation.
    aclblasLtMatrixLayoutImpl impl;
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), layout, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    const aclblasStatus_t setStatus = ApplyLayoutSetAttr(impl, attr, buf, sizeInBytes);
    if (setStatus != ACLBLAS_STATUS_SUCCESS) {
        return setStatus;
    }

    // Pack the updated impl back into the opaque capsule.
    return PackImplIntoCapsule(layout, sizeof(*layout), &impl, sizeof(impl));
}

aclblasStatus_t aclblasLtMatrixLayoutGetAttribute(
    const aclblasLtMatrixLayout_t layout, aclblasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes,
    size_t* sizeWritten)
{
    if (layout == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasLtMatrixLayoutImpl impl;
    static_assert(sizeof(impl) <= sizeof(*layout), "aclblasLtMatrixLayoutImpl too large for capsule");
    aclblasStatus_t copyStatus = CheckedMemcpyS(&impl, sizeof(impl), layout, sizeof(impl));
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        return copyStatus;
    }

    return ReadLayoutGetAttr(impl, attr, buf, sizeInBytes, sizeWritten);
}

} // extern "C"
