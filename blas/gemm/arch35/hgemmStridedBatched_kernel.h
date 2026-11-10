/*
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
#include "common/helper/gemm_strided_tiling_arch35.h"

// Shared Host/device policy. Values preserve the validated 950PR schedule.
constexpr uint32_t HGEMM_CUBE_C0_ELEMENTS = 16; // FP16 NZ/ZN inner block.
constexpr uint32_t HGEMM_DMA_ALIGNMENT_BYTES = 32;
constexpr uint32_t HGEMM_WORKSPACE_COLUMN_ALIGNMENT = HGEMM_DMA_ALIGNMENT_BYTES / sizeof(float);
constexpr uint32_t HGEMM_TILE_SMALL = 64;
constexpr uint32_t HGEMM_TILE_STANDARD = 128;
constexpr uint32_t HGEMM_TILE_LARGE = 256;
constexpr uint32_t HGEMM_REDUCTION_SHALLOW = 64;
constexpr uint32_t HGEMM_REDUCTION_STANDARD = 128;
constexpr uint32_t HGEMM_K_PANEL_ELEMENTS = 256;
// Large tiles are selected only when padded area is at most 3/2 of standard tiles.
constexpr uint32_t HGEMM_PADDED_AREA_NUMERATOR = 3;
constexpr uint32_t HGEMM_PADDED_AREA_DENOMINATOR = 2;
constexpr uint32_t HGEMM_EPILOGUE_THREADS = 256;
constexpr uint32_t HGEMM_EPILOGUE_ELEMENTS_PER_CORE = 1024;
constexpr uint32_t HGEMM_DOUBLE_BUFFER_SLOTS = 2;
constexpr uint64_t HGEMM_L1_CAPACITY_BYTES = 512 * 1024;
constexpr uint64_t HGEMM_L1_SLOT_BYTES = HGEMM_L1_CAPACITY_BYTES / HGEMM_DOUBLE_BUFFER_SLOTS;
constexpr uint64_t HGEMM_L0_AB_CAPACITY_BYTES = 64 * 1024;
constexpr uint64_t HGEMM_L0_AB_HALF_BYTES = HGEMM_L0_AB_CAPACITY_BYTES / HGEMM_DOUBLE_BUFFER_SLOTS;
constexpr uint64_t HGEMM_L0_C_CAPACITY_BYTES = 256 * 1024;
constexpr uint32_t HGEMM_SMALL_TILE_C_SLOTS = 8;
constexpr uint32_t HGEMM_STANDARD_TILE_C_SLOTS = 4;
// Mmad accumulation needs an explicit PIPE_M dependency below ten C0-by-C0 blocks.
constexpr uint32_t HGEMM_MMAD_DEPENDENCY_BLOCKS = 10;
static_assert((HGEMM_DOUBLE_BUFFER_SLOTS & (HGEMM_DOUBLE_BUFFER_SLOTS - 1)) == 0, "Slot ring must be a power of two");

constexpr uint32_t HGEMM_GROUPED_SIZES[] = {8, 16, 32, 64};
static_assert(
    sizeof(HGEMM_GROUPED_SIZES) / sizeof(HGEMM_GROUPED_SIZES[0]) == 4,
    "Update grouped kernel dispatch when adding or removing a size");

inline bool IsHgemmGroupedSize(int size)
{
    for (const auto candidate : HGEMM_GROUPED_SIZES) {
        if (size == static_cast<int>(candidate))
            return true;
    }
    return false;
}

struct HgemmBatchEpilogue {
    int32_t m, n, ldc, workspaceLdc, batchCount;
    int64_t strideC, workspaceStride;
    float alpha, beta;
    int32_t scaleOnly;
};

void HgemmBatchCube(
    uint32_t blocks, void* stream, uint8_t* left, uint8_t* right, uint8_t* out,
    const GemmStridedBatchedExTilingData& tiling, bool direct);
void HgemmBatchCombine(uint32_t blocks, void* stream, uint8_t* tmp, uint8_t* c, const HgemmBatchEpilogue& tiling);

void HgemmBatchGrouped(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, const GemmStridedBatchedExTilingData& tiling);
