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

enum class IcaminLayout : uint32_t { CONTIGUOUS, DENSE_STRIDED, SPARSE_STRIDED };
enum class IcaminReduction : uint32_t { SINGLE_CORE, STAGED, FUSED };

constexpr uint32_t ICAMIN_SLOT_FLOATS = 8;
constexpr uint32_t ICAMIN_GROUP_TILES = 32;
constexpr uint32_t ICAMIN_CONTIGUOUS_TILE = 7680;
constexpr uint32_t ICAMIN_STRIDED_TILE = 2048;
constexpr uint32_t ICAMIN_LARGE_TILE = 14336;
constexpr uint32_t ICAMIN_FIRST_NORM_FLOATS = 64;
// Includes a full 64-lane fetch from the last Gather offset view.
constexpr uint32_t ICAMIN_METADATA_FLOATS = 288;
// DataCopyPad's A2/A3 source gap is a uint32_t byte count.
constexpr uint32_t ICAMIN_MAX_DMA_INCX = 536870912;

struct IcaminArch22Tiling {
    uint32_t n;
    uint32_t incx;
    uint32_t cores;
    uint32_t perCore;
    uint32_t tile;
    uint32_t inputFloats;
    uint32_t buffers;
    uint32_t slots;
    IcaminLayout layout;
    IcaminReduction reduction;
};
