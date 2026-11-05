/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "aclblaslt_matmul_problem.h"
#include "cann_ops_blasLt.h"

constexpr uint32_t LT_MATMUL22_ALGO_H0 = 0x22000000U;
constexpr uint32_t LT_MATMUL22_ALGO_H1 = 0x22000001U;
constexpr uint32_t LT_MATMUL22_ALGO_H2 = 0x22000002U;
constexpr uint32_t LT_MATMUL22_ALGO_H3 = 0x22000003U;
constexpr uint32_t LT_MATMUL22_ALGO_H4 = 0x22000004U;
constexpr uint32_t LT_MATMUL22_ALGO_H10 = 0x2200000aU;
constexpr uint32_t LT_MATMUL22_ALGO_H11 = 0x2200000bU;
constexpr uint32_t LT_MATMUL22_ALGO_H12 = 0x2200000cU;
constexpr uint32_t LT_MATMUL22_ALGO_H9 = 0x22000009U;
constexpr uint32_t LT_MATMUL22_ALGO_H5 = 0x22000005U;

aclblasStatus_t LtMatmul22Api(
    aclblasLtHandle_t handle, aclblasLtMatmulDesc_t computeDesc, const void* alpha, const void* a,
    aclblasLtMatrixLayout_t aDesc, const void* b, aclblasLtMatrixLayout_t bDesc, const void* beta, const void* c,
    aclblasLtMatrixLayout_t cDesc, void* d, aclblasLtMatrixLayout_t dDesc, const aclblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceBytes, aclrtStream stream);
aclblasStatus_t LtMatmul22Heuristic(
    aclblasLtHandle_t handle, aclblasLtMatmulDesc_t computeDesc, aclblasLtMatrixLayout_t aDesc,
    aclblasLtMatrixLayout_t bDesc, aclblasLtMatrixLayout_t cDesc, aclblasLtMatrixLayout_t dDesc,
    aclblasLtMatmulPreference_t preference, int requested, aclblasLtMatmulHeuristicResult_t* results, int* count);
aclblasStatus_t LtMatmul22AlgoInit(int id, aclblasLtMatmulAlgo_t& algo);
aclblasStatus_t LtMatmul22Launch(const MatmulProblem& problem);
