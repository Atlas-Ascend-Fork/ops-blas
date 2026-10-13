/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file cgeam_tiling_data.h
 * @brief Tiling data structure for Cgeam operator (host/device shared)
 *        C = alpha * op(A) + beta * op(B)  (complex)
 */

#pragma once

#include <cstdint>
#include "cann_ops_blas_common.h"

namespace CgeamConstants {
constexpr uint32_t BLOCK_BYTES = 32U;
// Cache policy counts two input matrices (A, B) and the full working set (A, B, C).
constexpr uint64_t INPUT_MATRIX_COUNT = 2ULL;
constexpr uint64_t WORKSET_MATRIX_COUNT = 3ULL;
// Ascend 950PR L2 capacity, shared by host tiling and device traversal decisions.
constexpr uint64_t L2_CAPACITY_BYTES = 128ULL * 1024ULL * 1024ULL;
} // namespace CgeamConstants

enum class CgeamTransposePath : uint32_t { NONE, STANDARD, MERGED };

struct CgeamTilingData {
    uint32_t m;             // Number of rows in output matrix C
    uint32_t n;             // Number of columns in output matrix C
    float alphaR;           // Real part of scalar alpha
    float alphaI;           // Imaginary part of scalar alpha
    uint32_t alphaIsZero;   // Optimized flag indicating alpha == 0
    aclblasOperation_t opA; // Transpose type for matrix A
    uint32_t lda;           // Leading dimension of matrix A
    float betaR;            // Real part of scalar beta
    float betaI;            // Imaginary part of scalar beta
    uint32_t betaIsZero;    // Optimized flag indicating beta == 0
    aclblasOperation_t opB; // Transpose type for matrix B
    uint32_t ldb;           // Leading dimension of matrix B
    uint32_t ldc;           // Leading dimension of matrix C
    // Common one-dimensional launch grid.
    uint32_t blockCount;      // Launched AIV blocks
    // Contiguous fast path: alpha=1, beta=0/1, compact active inputs and C.
    uint32_t fastPath;             // Whether the packed-complex flat path is enabled
    uint32_t fastTileComplex;      // Complex elements processed per UB tile
    uint64_t fastPerCoreComplex;   // 32-byte-aligned base workload per core
    uint32_t fastExtraBlockCores;  // Cores receiving one extra 32-byte block
    uint32_t fastTailComplex;      // Final unaligned complex elements
    // Tiled transpose path: alpha=1, beta=0/1, compact active inputs and C.
    CgeamTransposePath transposeTilePath; // Explicit transpose strategy selected by the host
    uint32_t transposeTileRows;    // Output rows processed by one tile
    uint32_t transposeTileCols;    // Output columns processed by one tile
    uint32_t transposeRowTiles;    // Number of tiles along the output-row dimension
    uint32_t transposeTotalTiles;  // Total number of transpose tiles
    // Elementwise fallback for arbitrary scalars, leading dimensions, and transpose modes.
    uint32_t simtThreads;          // Threads launched on each AIV core
};
