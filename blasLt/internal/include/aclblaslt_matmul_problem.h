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
 * \file aclblaslt_matmul_problem.h
 * \brief Plain problem description consumed by the matmul engine. The API layer fills it from the
 *        validated capsules so the engine works on a POD struct (easy to unit test / mock).
 */

#pragma once

#include "aclblaslt_layout_impl.h"

#include "cann_ops_blasLt.h"

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

struct MatmulProblem {
    // dimensions
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    bool transA = false;
    bool transB = false;

    // types
    aclDataType dtypeA = ACL_DT_UNDEFINED;
    aclDataType dtypeB = ACL_DT_UNDEFINED;
    aclDataType dtypeC = ACL_DT_UNDEFINED;
    aclDataType dtypeD = ACL_DT_UNDEFINED;

    // scalars
    float alpha = 1.0f;
    float beta = 0.0f;

    // device pointers
    const void* A = nullptr;
    const void* B = nullptr;
    const void* C = nullptr;
    void* D = nullptr;
    void* scaleA = nullptr;
    void* scaleB = nullptr;
    void* workspace = nullptr;
    size_t workspaceSize = 0;

    // layout
    uint32_t lda = 0;
    uint32_t ldb = 0;
    uint32_t ldc = 0;
    uint32_t ldd = 0;
    aclblasLtOrder_t orderA = ACLBLASLT_ORDER_COL;
    aclblasLtOrder_t orderB = ACLBLASLT_ORDER_COL;
    aclblasLtOrder_t orderC = ACLBLASLT_ORDER_COL;
    aclblasLtOrder_t orderD = ACLBLASLT_ORDER_COL;

    // execution context
    uint32_t numBlocks = 0;
    aclrtStream stream = nullptr;

    // optional caller-supplied algo (raw capsule, decoded by the engine)
    const aclblasLtMatmulAlgo_t* algo = nullptr;
};

// Assemble a MatmulProblem from already-validated descriptor / layout impls and call parameters.
// Dimensions (m, n, k) and the cube-core count (numBlocks) are computed by the caller and passed in.
void BuildMatmulProblem(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* aLayout,
    const aclblasLtMatrixLayoutImpl* bLayout, const aclblasLtMatrixLayoutImpl* cLayout,
    const aclblasLtMatrixLayoutImpl* dLayout, uint64_t m, uint64_t n, uint64_t k, const void* alpha, const void* beta,
    const void* A, const void* B, const void* C, void* D, const aclblasLtMatmulAlgo_t* algo, void* workspace,
    size_t workspaceSize, uint32_t numBlocks, aclrtStream stream, MatmulProblem& out);
