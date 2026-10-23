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
 * \file aclblaslt_matmul_problem.cpp
 * \brief Assemble a MatmulProblem from validated descriptor / layout impls.
 */

#include "aclblaslt_matmul_problem.h"

#include <cstdint>

void BuildMatmulProblem(
    const aclblasLtMatmulDescImpl* desc, const aclblasLtMatrixLayoutImpl* aLayout,
    const aclblasLtMatrixLayoutImpl* bLayout, const aclblasLtMatrixLayoutImpl* cLayout,
    const aclblasLtMatrixLayoutImpl* dLayout, uint64_t m, uint64_t n, uint64_t k, const void* alpha, const void* beta,
    const void* A, const void* B, const void* C, void* D, const aclblasLtMatmulAlgo_t* algo, void* workspace,
    size_t workspaceSize, uint32_t numBlocks, aclrtStream stream, MatmulProblem& out)
{
    out = MatmulProblem{};
    out.m = m;
    out.n = n;
    out.k = k;
    out.transA = (desc->transA != ACLBLAS_OP_N);
    out.transB = (desc->transB != ACLBLAS_OP_N);

    out.dtypeA = aLayout->type;
    out.dtypeB = bLayout->type;
    out.dtypeC = cLayout->type;
    out.dtypeD = dLayout->type;

    out.devicePointerMode = (desc->pointerMode == ACLBLASLT_POINTER_MODE_DEVICE);
    if (out.devicePointerMode) {
        out.alphaDevice = alpha;
        out.betaDevice = beta;
    } else {
        out.alpha = *reinterpret_cast<const float*>(alpha);
        out.beta = *reinterpret_cast<const float*>(beta);
    }

    out.A = A;
    out.B = B;
    out.C = C;
    out.D = D;
    out.scaleA = const_cast<void*>(desc->scaleA);
    out.scaleB = const_cast<void*>(desc->scaleB);
    out.workspace = workspace;
    out.workspaceSize = workspaceSize;

    out.lda = static_cast<uint32_t>(aLayout->ld);
    out.ldb = static_cast<uint32_t>(bLayout->ld);
    out.ldc = static_cast<uint32_t>(cLayout->ld > 0 ? cLayout->ld : static_cast<int64_t>(n));
    out.ldd = static_cast<uint32_t>(dLayout->ld > 0 ? dLayout->ld : static_cast<int64_t>(n));
    out.orderA = aLayout->order;
    out.orderB = bLayout->order;
    out.orderC = cLayout->order;
    out.orderD = dLayout->order;

    out.numBlocks = numBlocks;
    out.stream = stream;
    out.algo = algo;
}
