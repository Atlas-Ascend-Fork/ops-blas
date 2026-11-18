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
#include "chbmv_tiling_data.h"

namespace chbmv_layout {
// Offsets and capacities are in 32-bit elements, for both float and uint32_t.
// Each region begins at the end of the previous reservation; no regions alias.
constexpr uint32_t LANE_ELEMENTS = CHBMV_VECTOR_ROWS;
constexpr uint32_t PACKED_ELEMENTS = CHBMV_VECTOR_ROWS * (CHBMV_OUTPUT_BYTES / sizeof(float));
constexpr uint32_t PHASE_COUNT = 8;
constexpr uint32_t PHASE_STRIDE = 2 * CHBMV_VECTOR_ROWS;
constexpr uint32_t PHASE_ELEMENTS = PHASE_COUNT * PHASE_STRIDE;
constexpr uint32_t OFFSET_ELEMENTS = PHASE_STRIDE;
constexpr uint32_t OFFSET_RESERVED_ELEMENTS = PACKED_ELEMENTS;
constexpr uint32_t MAX_PARTIALS = 32;
constexpr uint32_t PARTIAL_ELEMENTS = MAX_PARTIALS * PHASE_STRIDE;

constexpr uint32_t AR = 0;
constexpr uint32_t AI = AR + LANE_ELEMENTS;
constexpr uint32_t XR = AI + LANE_ELEMENTS;
constexpr uint32_t XI = XR + LANE_ELEMENTS;
constexpr uint32_t YR = XI + LANE_ELEMENTS;
constexpr uint32_t YI = YR + LANE_ELEMENTS;
constexpr uint32_t MR = YI + LANE_ELEMENTS;
constexpr uint32_t MI = MR + LANE_ELEMENTS;
constexpr uint32_t VR = MI + LANE_ELEMENTS;
constexpr uint32_t VI = VR + LANE_ELEMENTS;
constexpr uint32_t P0 = VI + LANE_ELEMENTS;
constexpr uint32_t P1 = P0 + LANE_ELEMENTS;
constexpr uint32_t P2 = P1 + LANE_ELEMENTS;
constexpr uint32_t P3 = P2 + LANE_ELEMENTS;
constexpr uint32_t A_OFFSETS = P3 + LANE_ELEMENTS;
constexpr uint32_t X_OFFSETS = A_OFFSETS + LANE_ELEMENTS;
constexpr uint32_t PACKED = X_OFFSETS + LANE_ELEMENTS;
constexpr uint32_t PHASE_REAL = PACKED + PACKED_ELEMENTS;
constexpr uint32_t PHASE_IMAG = PHASE_REAL + PHASE_ELEMENTS;
constexpr uint32_t PHASE_OFFSETS = PHASE_IMAG + PHASE_ELEMENTS;
constexpr uint32_t PARTIAL = PHASE_OFFSETS + OFFSET_RESERVED_ELEMENTS;
constexpr uint32_t PHASE_NEGATIVE_IMAG = PARTIAL + PARTIAL_ELEMENTS;
constexpr uint32_t SCRATCH_ELEMENTS = PHASE_NEGATIVE_IMAG + PHASE_ELEMENTS;

static_assert(sizeof(float) == sizeof(uint32_t), "Scratch views must use the same element size");
static_assert(LANE_ELEMENTS * sizeof(float) % CHBMV_OUTPUT_BYTES == 0, "Lane regions must be DMA aligned");
static_assert(
    PACKED_ELEMENTS % LANE_ELEMENTS == 0 && PHASE_ELEMENTS % LANE_ELEMENTS == 0 &&
        OFFSET_RESERVED_ELEMENTS % LANE_ELEMENTS == 0 && PARTIAL_ELEMENTS % LANE_ELEMENTS == 0,
    "All region boundaries must preserve lane alignment");
static_assert(OFFSET_ELEMENTS <= OFFSET_RESERVED_ELEMENTS, "Phase offsets must not overlap partial sums");
static_assert(CHBMV_VECTOR_ROWS + CHBMV_MIRROR_TILE - 1 <= PHASE_STRIDE, "Phase windows must not overlap");
static_assert(SCRATCH_ELEMENTS * sizeof(float) <= CHBMV_VECTOR_SCRATCH_BYTES, "Scratch exceeds reserved UB");

// The A cache is reused by mirror and direct tiles, which execute sequentially.
constexpr uint32_t MIRROR_COMPONENT_ELEMENTS = CHBMV_VECTOR_ROWS * CHBMV_MIRROR_TILE;
constexpr uint32_t MIRROR_INPUT_ELEMENTS = 2 * MIRROR_COMPONENT_ELEMENTS;
constexpr uint32_t MIRROR_REAL = MIRROR_INPUT_ELEMENTS;
constexpr uint32_t MIRROR_IMAG = MIRROR_REAL + MIRROR_COMPONENT_ELEMENTS;
constexpr uint32_t MATRIX_CACHE_ELEMENTS = MIRROR_IMAG + MIRROR_COMPONENT_ELEMENTS;
constexpr uint32_t DIRECT_COMPONENT_ELEMENTS = CHBMV_VECTOR_ROWS * CHBMV_DIRECT_TILE;
constexpr uint32_t DIRECT_REAL = 2 * DIRECT_COMPONENT_ELEMENTS;
constexpr uint32_t DIRECT_IMAG = DIRECT_REAL + DIRECT_COMPONENT_ELEMENTS;
static_assert(DIRECT_IMAG + DIRECT_COMPONENT_ELEMENTS <= MATRIX_CACHE_ELEMENTS, "Direct tile exceeds A cache");
} // namespace chbmv_layout
