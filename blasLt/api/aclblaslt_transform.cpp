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
 * \file aclblaslt_transform.cpp
 * \brief Public C API: aclblasLtMatrixTransform — validates inputs, packs layouts, dispatches to
 *        the transform engine.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_handle_impl.h"
#include "aclblaslt_layout_impl.h"
#include "aclblaslt_logger_impl.h"
#include "matrix_transform_engine.h"

#include <acl/acl.h>
#include <cstdint>

namespace {

// Pack the transform-relevant fields of a matrix layout capsule into the plain parameter struct
// consumed by the transform operator unit.
MatTransformLayout MatPackTransformLayout(const aclblasLtMatrixLayoutImpl* layout)
{
    MatTransformLayout packed;
    packed.type = layout->type;
    packed.rows = layout->rows;
    packed.cols = layout->cols;
    packed.ld = layout->ld;
    packed.order = layout->order;
    packed.batchCount = layout->batchCount;
    return packed;
}

} // namespace

extern "C" {

aclblasStatus_t aclblasLtMatrixTransform(
    aclblasLtHandle_t lightHandle, aclblasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A,
    aclblasLtMatrixLayout_t Adesc, const void* beta, const void* B, aclblasLtMatrixLayout_t Bdesc, void* C,
    aclblasLtMatrixLayout_t Cdesc, aclrtStream stream)
{
    if (lightHandle == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatrixTransform", "lightHandle is null");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (transformDesc == nullptr || Adesc == nullptr || Cdesc == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatrixTransform", "transformDesc, Adesc, or Cdesc is null");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatrixTransform", "alpha is null");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* desc = reinterpret_cast<aclblasLtMatrixTransformDescImpl*>(transformDesc);
    auto* ALayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Adesc);
    auto* CLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Cdesc);
    if (desc->magic != ACLBLASLT_TRANSFORM_DESC_MAGIC || ALayout->magic != ACLBLASLT_LAYOUT_MAGIC ||
        CLayout->magic != ACLBLASLT_LAYOUT_MAGIC) {
        AclBlasLt::LoggerManager::GetInstance().Log(
            ACLBLASLT_LOG_MASK_ERROR, "aclblasLtMatrixTransform",
            "corrupted or foreign descriptor/layout (magic mismatch)");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* handleImpl = lightHandle;

    const uint64_t rows = CLayout->rows;
    const uint64_t cols = CLayout->cols;
    AclBlasLt::LoggerManager::GetInstance().Log(
        ACLBLASLT_LOG_MASK_TRACE, "aclblasLtMatrixTransform", "rows=%lu, cols=%lu", static_cast<unsigned long>(rows),
        static_cast<unsigned long>(cols));
    if (rows == 0U || cols == 0U) {
        return ACLBLAS_STATUS_SUCCESS; // empty matrix, no-op (checked before B presence)
    }

    const MatTransformLayout aPacked = MatPackTransformLayout(ALayout);
    const MatTransformLayout cPacked = MatPackTransformLayout(CLayout);
    auto* BLayout = reinterpret_cast<aclblasLtMatrixLayoutImpl*>(Bdesc);
    const MatTransformLayout bPacked = (BLayout != nullptr) ? MatPackTransformLayout(BLayout) : MatTransformLayout{};
    const bool bLayoutValid = (BLayout != nullptr) && BLayout->magic == ACLBLASLT_LAYOUT_MAGIC;

#ifdef OPS_BLASLT_ARCH22
    // MatrixTransform has no arch22 device implementation.
    return ACLBLAS_STATUS_NOT_SUPPORTED;
#else
    return MatTransformLaunch(
        handleImpl->deviceId, desc, alpha, A, &aPacked, beta, B, (BLayout != nullptr) ? &bPacked : nullptr,
        bLayoutValid, C, &cPacked, rows, cols, stream);
#endif
}

} // extern "C"
