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

constexpr uint32_t CGEMM_COMPLEX_COMPONENTS = 2;
constexpr uint32_t CGEMM_DATA_BLOCK_BYTES = 32; // 950 data-copy block size.
constexpr uint32_t CGEMM_FOUR_PRODUCTS = 4;
constexpr uint32_t CGEMM_THREE_PRODUCTS = 3;
constexpr uint32_t CGEMM_SIMT_THREADS = 512;
constexpr uint64_t CGEMM_UINT32_VALUE_COUNT = 1ULL << 32;
constexpr uint64_t CGEMM_UINT32_MAX_VALUE = CGEMM_UINT32_VALUE_COUNT - 1;
constexpr uint32_t CGEMM_KIB = 1024;
constexpr uint32_t CGEMM_L0_OPERAND_BYTES = 64 * CGEMM_KIB;
constexpr uint32_t CGEMM_L0C_BYTES = 256 * CGEMM_KIB;
constexpr uint32_t CGEMM_L1_BYTES = 512 * CGEMM_KIB;
constexpr uint32_t CGEMM_L0_FLOAT_ELEMENTS = CGEMM_L0_OPERAND_BYTES / sizeof(float);
constexpr uint32_t CGEMM_HALF_L0_FLOAT_ELEMENTS = CGEMM_L0_FLOAT_ELEMENTS / 2;
// One L1 ping-pong stage holds two FP32 operands.
constexpr uint32_t CGEMM_L1_PING_PONG_STAGES = 2;
constexpr uint32_t CGEMM_L1_OPERANDS_PER_STAGE = 2;
constexpr uint32_t CGEMM_L1_OPERAND_STAGE_ELEMENTS =
    CGEMM_L1_BYTES / (CGEMM_L1_PING_PONG_STAGES * CGEMM_L1_OPERANDS_PER_STAGE * sizeof(float));

constexpr uint32_t CGEMM_CUBE_FRACTAL = 16; // 950 Cube matrix-row granularity.
constexpr uint32_t CGEMM_OUTPUT_STAGES = 2; // Overlap storing one product tile with computing the next.
constexpr uint32_t CgemmMaxSquareTile()
{
    uint32_t tile = CGEMM_CUBE_FRACTAL;
    constexpr uint32_t growth = 2; // Double each tile dimension at the next step.
    while (growth * growth * CGEMM_OUTPUT_STAGES * tile * tile * sizeof(float) <= CGEMM_L0C_BYTES) {
        tile *= growth;
    }
    return tile;
}
constexpr uint32_t CGEMM_MAX_SQUARE_TILE = CgemmMaxSquareTile();
// Four products share the per-tile accumulator budget in the fused path.
constexpr uint32_t CGEMM_FUSED_TILE = CGEMM_MAX_SQUARE_TILE / CGEMM_COMPLEX_COMPONENTS;
constexpr uint32_t CGEMM_VECTOR_BYTES = 256;                   // 950 vector register width.
constexpr uint32_t CGEMM_UB_WORKSPACE_BYTES = 192 * CGEMM_KIB; // Conservative per-AIV budget.
constexpr uint32_t CGEMM_FUSED_MAX_TILE = CGEMM_MAX_SQUARE_TILE;

struct CgemmDeinterleaveTilingData {
    uint32_t aRows;
    uint32_t aCols;
    uint32_t lda;
    uint32_t aOutRows;
    uint32_t bRows;
    uint32_t bCols;
    uint32_t ldb;
    uint32_t bOutRows;
    uint32_t aRowsPerCore;
    uint32_t bRowsPerCore;
    uint8_t conjugateA;
    uint8_t conjugateB;
    uint8_t threeProducts;
};

struct CgemmEpilogueTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;
    uint32_t tempLdc;
    uint32_t rowsPerCore;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    uint8_t alphaZero;
    uint8_t betaZero;
    uint8_t threeProducts;
    uint32_t kPartitions;
    uint64_t partStride;
    // Nonzero only for the forced-3m scalar epilogue that can require ordered correction.
    uint32_t referenceK = 0, referenceLda = 0, referenceLdb = 0;
    uint8_t referenceTransA = 0, referenceTransB = 0;
};

struct CgemmOnchipTilingData {
    uint32_t m, n, k, lda, ldb, ldc;
    uint8_t transA, transB, conjugateA, conjugateB;
    uint32_t batchCount = 1;
    int64_t strideA = 0, strideB = 0, strideC = 0;
    uint32_t reductionSpan = 0;
};
