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
#include <cstdint>

constexpr uint32_t GEMM_GROUPED22_CAPACITY = 16;
constexpr uint32_t GEMM_GROUPED22_TILE = 128;
constexpr uint32_t GEMM_GROUPED22_K_CHUNK = 1024;
struct GemmGrouped22Group {
    int32_t m, n, k, lda, ldb, ldc, transA, transB;
    uint64_t instanceStart, instances, cubeStart, cubeTasks, vectorStart, vectorTasks;
    uint64_t workspaceOffset;
    uint64_t alphaAddress, betaAddress;
    float alpha, beta;
    uint32_t scalarDevice, cube, direct, tempLd, repairSmall, cubeTileN;
};
struct GemmGrouped22Batch {
    GemmGrouped22Group groups[GEMM_GROUPED22_CAPACITY];
    uint64_t cubeTasks, vectorTasks;
    uint32_t count;
};
static_assert(sizeof(GemmGrouped22Batch) < 3500, "kernel argument payload must remain bounded");
