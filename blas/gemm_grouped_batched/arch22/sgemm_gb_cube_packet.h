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
#include "sgemm_gb_packet.h"
namespace sgemm_gb {
// SDK-independent wire format. The SDK's TCubeTiling object is copied byte-for-byte into this area.
// Both translation units assert size/alignment at the local CANN compilation stage.
constexpr uint32_t kCubeTilingWords = 50;
struct CubePacket {
    Packet matrices;
    alignas(8) uint32_t tiling[kCubeTilingWords];
    uint32_t tileN;
    uint32_t tileM;
};
static_assert(sizeof(CubePacket) == 2032, "Cube kernel argument ABI changed");
static_assert(sizeof(CubePacket) <= 2048, "Kernel argument budget exceeded");
// A/C/B allocations often lie within a 4 GiB span per group. Keep the base
// addresses at full width, then pack each independent matrix as three byte
// offsets. This permits 147 matrices within the same by-value ABI budget.
constexpr uint32_t kCubeOffsetBatches = 147;
struct OffsetMatrices {
    Group g;
    uint32_t count;
    uint32_t reserved;
    uint64_t baseA, baseB, baseC;
    uint32_t a[kCubeOffsetBatches];
    uint32_t b[kCubeOffsetBatches];
    uint32_t c[kCubeOffsetBatches];
};
struct CubeOffsetPacket {
    OffsetMatrices matrices;
    alignas(8) uint32_t tiling[kCubeTilingWords];
    uint32_t tileN;
    uint32_t tileM;
};
static_assert(sizeof(OffsetMatrices) == 1840, "Offset matrix ABI changed");
static_assert(sizeof(CubeOffsetPacket) == 2048, "Offset kernel argument ABI changed");
constexpr uint32_t kSmallCubeGroups = 4;
constexpr uint32_t kSmallCubeMatrices = 32;
struct CubeGroupPacket {
    uint32_t groups, count;
    Group g[kSmallCubeGroups];
    uint32_t starts[kSmallCubeGroups + 1];
    alignas(8) uint32_t tiling[kSmallCubeGroups][kCubeTilingWords];
    uint64_t a[kSmallCubeMatrices], b[kSmallCubeMatrices], c[kSmallCubeMatrices];
};
static_assert(sizeof(CubeGroupPacket) <= 2048, "Small group kernel argument budget exceeded");
struct SingleMatrices {
    Group g;
    uint32_t count, reserved;
    uint64_t a[1], b[1], c[1];
};
struct CubeSinglePacket {
    SingleMatrices matrices;
    alignas(8) uint32_t tiling[kCubeTilingWords];
    uint32_t tileN, tileM;
};
} // namespace sgemm_gb
