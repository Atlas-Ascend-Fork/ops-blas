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

#include <cstddef>
#include <cstdint>

struct CherkArch22TilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldc;
    uint32_t cubeN;
    uint32_t cubeK;
    uint32_t trans;
    uint32_t uplo;
    uint32_t aivCoreNum;
    uint32_t aicCoreNum;
    uint8_t skipProduct;
    uint8_t smallPath;
    uint8_t directPath;
    uint8_t outerPath;
    uint8_t compactPath;
    // 0: standalone, 1: fused contiguous MIX, 2: fused strided MIX,
    // 3: standalone pack followed by strided producer-consumer MIX.
    uint8_t directMixMode;
    // Compile-time OP_N pack specialization selected from the runtime UB
    // budget. Valid values are 1, 2, 4 and 8 rows.
    uint8_t directPackBatchRows;
};

constexpr uint32_t CHERK_ARCH22_CUBE_M_ALIGN = 256U;
constexpr uint32_t CHERK_ARCH22_CUBE_K_ALIGN = 64U;
constexpr uint32_t CHERK_ARCH22_OUTER_ROWS = 128U;
constexpr uint32_t CHERK_ARCH22_OUTER_RANK = 2U;
constexpr uint32_t CHERK_ARCH22_OUTER_PANEL_COLS = 8U;

__host__ __aicore__ inline bool CherkArch22OuterGeometry(uint32_t n, uint32_t k)
{
    return n > 0U && n <= CHERK_ARCH22_OUTER_ROWS && n % CHERK_ARCH22_OUTER_PANEL_COLS == 0U &&
           k == CHERK_ARCH22_OUTER_RANK;
}

// Persistent metadata is one uint32_t array. The final 128 words hold the
// 4096-bit diagonal mask and can also be viewed as bytes on device.
constexpr uint32_t CHERK_ARCH22_META_INTERLEAVE_OFFSET = 0U;
constexpr uint32_t CHERK_ARCH22_META_INTERLEAVE_COUNT = 64U * 64U * 2U;
constexpr uint32_t CHERK_ARCH22_META_TRANSPOSE_OFFSET =
    CHERK_ARCH22_META_INTERLEAVE_OFFSET + CHERK_ARCH22_META_INTERLEAVE_COUNT;
constexpr uint32_t CHERK_ARCH22_META_TRANSPOSE_COUNT = 64U * 64U;
constexpr uint32_t CHERK_ARCH22_META_PACK_C_OFFSET =
    CHERK_ARCH22_META_TRANSPOSE_OFFSET + CHERK_ARCH22_META_TRANSPOSE_COUNT;
constexpr uint32_t CHERK_ARCH22_META_PACK_C_COUNT = 32U * 64U;
constexpr uint32_t CHERK_ARCH22_META_PACK_C128_OFFSET =
    CHERK_ARCH22_META_PACK_C_OFFSET + CHERK_ARCH22_META_PACK_C_COUNT;
constexpr uint32_t CHERK_ARCH22_META_PACK_C128_COUNT = 32U * 128U;
constexpr uint32_t CHERK_ARCH22_META_TINY8_TRANSPOSE_OFFSET =
    CHERK_ARCH22_META_PACK_C128_OFFSET + CHERK_ARCH22_META_PACK_C128_COUNT;
constexpr uint32_t CHERK_ARCH22_META_TINY8_TRANSPOSE_COUNT = 8U * 8U;
constexpr uint32_t CHERK_ARCH22_META_TINY16_TRANSPOSE_OFFSET =
    CHERK_ARCH22_META_TINY8_TRANSPOSE_OFFSET + CHERK_ARCH22_META_TINY8_TRANSPOSE_COUNT;
constexpr uint32_t CHERK_ARCH22_META_TINY16_TRANSPOSE_COUNT = 16U * 16U;
constexpr uint32_t CHERK_ARCH22_META_TINY_TRANSPOSE_OFFSET =
    CHERK_ARCH22_META_TINY16_TRANSPOSE_OFFSET + CHERK_ARCH22_META_TINY16_TRANSPOSE_COUNT;
constexpr uint32_t CHERK_ARCH22_META_TINY_TRANSPOSE_COUNT = 32U * 32U;
constexpr uint32_t CHERK_ARCH22_META_TINY_INTERLEAVE_OFFSET =
    CHERK_ARCH22_META_TINY_TRANSPOSE_OFFSET + CHERK_ARCH22_META_TINY_TRANSPOSE_COUNT;
constexpr uint32_t CHERK_ARCH22_META_TINY_INTERLEAVE_COUNT = 32U * 2U;
constexpr uint32_t CHERK_ARCH22_META_TINY64_INTERLEAVE_OFFSET =
    CHERK_ARCH22_META_TINY_INTERLEAVE_OFFSET + CHERK_ARCH22_META_TINY_INTERLEAVE_COUNT;
constexpr uint32_t CHERK_ARCH22_META_TINY64_INTERLEAVE_COUNT = 64U * 2U;
constexpr uint32_t CHERK_ARCH22_META_OUTER_K2_INTERLEAVE_OFFSET =
    CHERK_ARCH22_META_TINY64_INTERLEAVE_OFFSET + CHERK_ARCH22_META_TINY64_INTERLEAVE_COUNT;
constexpr uint32_t CHERK_ARCH22_META_OUTER_K2_INTERLEAVE_COUNT = 128U * 8U * 2U;
constexpr uint32_t CHERK_ARCH22_META_DIAGONAL_OFFSET =
    CHERK_ARCH22_META_OUTER_K2_INTERLEAVE_OFFSET + CHERK_ARCH22_META_OUTER_K2_INTERLEAVE_COUNT;
constexpr uint32_t CHERK_ARCH22_META_DIAGONAL_WORDS = (64U * 64U / 8U) / sizeof(uint32_t);
constexpr uint32_t CHERK_ARCH22_META_TRIANGLE_WORDS = (64U * 64U / 8U) / sizeof(uint32_t);
constexpr uint32_t CHERK_ARCH22_META_UPPER_OFFSET =
    CHERK_ARCH22_META_DIAGONAL_OFFSET + CHERK_ARCH22_META_DIAGONAL_WORDS;
constexpr uint32_t CHERK_ARCH22_META_LOWER_OFFSET = CHERK_ARCH22_META_UPPER_OFFSET + CHERK_ARCH22_META_TRIANGLE_WORDS;
constexpr uint32_t CHERK_ARCH22_META_WORDS = CHERK_ARCH22_META_LOWER_OFFSET + CHERK_ARCH22_META_TRIANGLE_WORDS;

// Workspace is row-major SoA. Both operations publish Kp x Np planes: OP_N
// packs source columns directly and OP_C transposes tiled source columns.
// Cube therefore uses one proven transposed-A/normal-B contract.
__host__ __aicore__ inline uint64_t CherkArch22MatrixElements(uint32_t cubeN, uint32_t cubeK)
{
    return static_cast<uint64_t>(cubeN) * cubeK;
}

__host__ __aicore__ inline uint64_t CherkArch22OffsetRealInput(uint32_t n, uint32_t k) { return 0; }
__host__ __aicore__ inline uint64_t CherkArch22OffsetImagInput(uint32_t n, uint32_t k)
{
    return CherkArch22MatrixElements(n, k);
}
__host__ __aicore__ inline uint64_t CherkArch22OffsetRealOutput(uint32_t n, uint32_t k)
{
    return 2ULL * CherkArch22MatrixElements(n, k);
}
__host__ __aicore__ inline uint64_t CherkArch22OffsetCrossOutput(uint32_t cubeN, uint32_t cubeK)
{
    return CherkArch22OffsetRealOutput(cubeN, cubeK) + static_cast<uint64_t>(cubeN) * cubeN;
}
__host__ __aicore__ inline uint64_t CherkArch22WorkspaceFloats(uint32_t cubeN, uint32_t cubeK)
{
    return CherkArch22OffsetCrossOutput(cubeN, cubeK) + static_cast<uint64_t>(cubeN) * cubeN;
}
static inline size_t CherkArch22WorkspaceBytes(uint32_t cubeN, uint32_t cubeK)
{
    const uint64_t bytes = CherkArch22WorkspaceFloats(cubeN, cubeK) * sizeof(float);
    return static_cast<size_t>((bytes + 511U) & ~511ULL);
}

// Low-rank direct-H workspace: three KxN input planes and two final NxN
// Hermitian output planes. The producer writes only the physical triangle.
__host__ __aicore__ inline uint64_t CherkArch22DirectInputElements(uint32_t cubeN, uint32_t cubeK)
{
    return static_cast<uint64_t>(cubeN) * cubeK;
}
__host__ __aicore__ inline uint64_t CherkArch22DirectOffsetRealInput(uint32_t cubeN, uint32_t cubeK) { return 0ULL; }
__host__ __aicore__ inline uint64_t CherkArch22DirectOffsetImagInput(uint32_t cubeN, uint32_t cubeK)
{
    return CherkArch22DirectInputElements(cubeN, cubeK);
}
__host__ __aicore__ inline uint64_t CherkArch22DirectOffsetNegImagInput(uint32_t cubeN, uint32_t cubeK)
{
    return 2ULL * CherkArch22DirectInputElements(cubeN, cubeK);
}
__host__ __aicore__ inline uint64_t CherkArch22DirectOffsetRealOutput(uint32_t cubeN, uint32_t cubeK)
{
    return 3ULL * CherkArch22DirectInputElements(cubeN, cubeK);
}
__host__ __aicore__ inline uint64_t CherkArch22DirectOffsetImagOutput(uint32_t cubeN, uint32_t cubeK)
{
    return CherkArch22DirectOffsetRealOutput(cubeN, cubeK) + static_cast<uint64_t>(cubeN) * cubeN;
}
__host__ __aicore__ inline uint64_t CherkArch22DirectWorkspaceFloats(uint32_t cubeN, uint32_t cubeK)
{
    return CherkArch22DirectOffsetImagOutput(cubeN, cubeK) + static_cast<uint64_t>(cubeN) * cubeN;
}
static inline size_t CherkArch22DirectWorkspaceBytes(uint32_t cubeN, uint32_t cubeK)
{
    const uint64_t bytes = CherkArch22DirectWorkspaceFloats(cubeN, cubeK) * sizeof(float);
    return static_cast<size_t>((bytes + 511U) & ~511ULL);
}

// Shared triangular task layout for both Cube epilogues.
struct CherkTriangleTaskCursor {
    uint32_t rowTile;
    uint64_t rowTaskBase;
    uint32_t tileAxis;
    bool upper;
};

__host__ __aicore__ inline CherkTriangleTaskCursor InitCherkTriangleCursor(
    uint64_t firstTask, uint32_t tileAxis, bool upper)
{
    CherkTriangleTaskCursor cursor{0U, 0ULL, tileAxis, upper};
    while (cursor.rowTile < tileAxis) {
        const uint32_t rowTasks = upper ? tileAxis - cursor.rowTile : cursor.rowTile + 1U;
        if (firstTask < cursor.rowTaskBase + rowTasks) {
            break;
        }
        cursor.rowTaskBase += rowTasks;
        ++cursor.rowTile;
    }
    return cursor;
}

__host__ __aicore__ inline uint32_t ResolveCherkTriangleCol(CherkTriangleTaskCursor& cursor, uint64_t task)
{
    uint32_t rowTasks = cursor.upper ? cursor.tileAxis - cursor.rowTile : cursor.rowTile + 1U;
    while (task >= cursor.rowTaskBase + rowTasks) {
        cursor.rowTaskBase += rowTasks;
        ++cursor.rowTile;
        rowTasks = cursor.upper ? cursor.tileAxis - cursor.rowTile : cursor.rowTile + 1U;
    }
    const uint32_t localTask = static_cast<uint32_t>(task - cursor.rowTaskBase);
    return cursor.upper ? cursor.rowTile + localTask : localTask;
}
