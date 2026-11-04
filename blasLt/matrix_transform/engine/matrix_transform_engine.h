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
 * \file matrix_transform_engine.h
 * \brief Host-side contract of aclblasLtMatrixTransform: the transform descriptor impl with its
 *        capsule magic, the plain matrix-layout parameter struct the public API layer packs for
 *        each operand, and the validate-and-launch engine entry. Only this minimal surface is
 *        shared with the public API layer; all other transform helpers stay in the operator unit.
 */

#pragma once

#include "cann_ops_blasLt.h"

#include <acl/acl.h>
#include <cstdint>

constexpr uint32_t ACLBLASLT_TRANSFORM_DESC_MAGIC = 0xACBE1234;

struct aclblasLtMatrixTransformDescImpl {
    uint32_t magic;
    aclDataType scaleType;
    int32_t pointerMode = 0;  // 0 = host
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    // Provenance flag: 1 for capsules allocated by aclblasLtMatrixTransformDescCreate() (heap,
    // may be Destroy()ed); 0 for in-place Init() products (caller storage -- Destroy() rejects
    // them with INVALID_VALUE instead of deleting a non-heap pointer, which is UB). Occupies
    // capsule bytes [20,24), previously unused. Round-trips Set/GetAttribute
    // (unpack-modify-repack preserves it).
    uint32_t heapOwned = 0;
};
static_assert(
    sizeof(aclblasLtMatrixTransformDescImpl) <= sizeof(aclblasLtMatrixTransformDescOpaque_t),
    "Impl of aclblasLtMatrixTransformDesc must fit in capsule!");

// Plain matrix-layout parameters of one transform operand, packed by the public API layer from its
// validated layout capsule before entering the operator unit.
struct MatTransformLayout {
    aclDataType type = ACL_DT_UNDEFINED;
    uint64_t rows = 0;
    uint64_t cols = 0;
    int64_t ld = 0;
    aclblasLtOrder_t order = ACLBLASLT_ORDER_COL;
    int32_t batchCount = 1;
};

// Validate-and-launch engine entry of aclblasLtMatrixTransform: resolves the scale path and the
// alpha/beta scalars, checks operand pointers and layouts, then dispatches the device pipeline
// (FP4 staged pipeline or the templated engine). Caller has already verified the handle, the
// descriptor / A / C capsules, packed the layouts (bLayout is null iff no B layout was supplied,
// bLayoutValid carries its capsule check), and short-circuited the empty (rows==0 || cols==0) no-op.
aclblasStatus_t MatTransformLaunch(
    int32_t deviceId, const aclblasLtMatrixTransformDescImpl* desc, const void* alpha, const void* A,
    const MatTransformLayout* aLayout, const void* beta, const void* B, const MatTransformLayout* bLayout,
    bool bLayoutValid, void* C, const MatTransformLayout* cLayout, uint64_t rows, uint64_t cols, aclrtStream stream);
