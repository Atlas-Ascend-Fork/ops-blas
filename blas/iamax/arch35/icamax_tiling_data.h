/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "common/helper/kernel_constant.h"

constexpr uint32_t ICAMAX_COMPLEX_COMPONENT_COUNT = 2U;
constexpr uint32_t ICAMAX_COMPONENT_BYTES = sizeof(float);
constexpr uint32_t ICAMAX_COMPLEX_BYTES = ICAMAX_COMPLEX_COMPONENT_COUNT * ICAMAX_COMPONENT_BYTES;
constexpr uint32_t ICAMAX_SIMD_VECTOR_COMPLEX_COUNT = 64U;
constexpr uint32_t ICAMAX_SIMD_TILE_COMPLEX_COUNT =
    UB_SIZE / ICAMAX_COMPLEX_BYTES / ICAMAX_SIMD_VECTOR_COMPLEX_COUNT * ICAMAX_SIMD_VECTOR_COMPLEX_COUNT;
constexpr uint32_t ICAMAX_SIMD_QUAD_MIN_COMPLEX_COUNT = 2U * ICAMAX_SIMD_TILE_COMPLEX_COUNT;

static_assert(ICAMAX_SIMD_TILE_COMPLEX_COUNT > 0U, "Icamax SIMD tile must not be empty");
static_assert(ICAMAX_SIMD_TILE_COMPLEX_COUNT * ICAMAX_COMPLEX_BYTES <= UB_SIZE, "Icamax SIMD tile exceeds UB budget");

struct IcamaxTilingData {
    uint32_t totalN;
    uint32_t perCoreN;
    uint32_t remainder;
    uint32_t useCoreNum;
    uint32_t tileSize;
    uint32_t nthreads;
    uint32_t incx;
    uint32_t vectorAligned;
    uint32_t scalarTail;
};
