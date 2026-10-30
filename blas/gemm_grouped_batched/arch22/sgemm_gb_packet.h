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
namespace sgemm_gb {
// Immutable kernel arguments; no shared GM metadata or stream-unsafe host workspace.
// CANN 9.1 arch22 TCubeTiling is 200 bytes. With its exact wire size below,
// 74 pointer triplets fit within the 2 KiB by-value kernel argument budget.
constexpr uint32_t kPacketBatches = 74;
// AIV-only 8x8 specialization carries more pointers because it needs no
// Cube tiling payload. At 80 matrices the kernel argument remains under 2 KiB.
constexpr uint32_t kSmall8Batches = 80;
// Keep the vector tile bounded for both A_N and A_T.  A_N uses this value
// directly while A_T selects its smaller gather tile in the kernel.
constexpr uint32_t kVectorRows = 512;
// Use enough output tasks to occupy the AIC cores on 910B3.  512 would leave
// a 1024x1024 case with only four tasks and serialize most of the device.
constexpr uint32_t kCubeTile = 512;
struct Group {
    uint32_t m, n, k, lda, ldb, ldc;
    uint32_t ta, tb;
    float alpha, beta;
};
struct Packet {
    Group g;
    uint32_t count;
    uint32_t reserved;
    uint64_t a[kPacketBatches];
    uint64_t b[kPacketBatches];
    uint64_t c[kPacketBatches];
};
static_assert(sizeof(Packet) == 1824, "Host/device packet ABI changed");
struct Small8Packet {
    Group g;
    uint32_t count;
    uint32_t reserved;
    uint64_t a[kSmall8Batches];
    uint64_t b[kSmall8Batches];
    uint64_t c[kSmall8Batches];
};
static_assert(sizeof(Small8Packet) == 1968, "Small AIV packet ABI changed");
static_assert(sizeof(Small8Packet) <= 2048, "Small AIV kernel argument budget exceeded");
struct Small8StridedPacket {
    Group g;
    uint32_t count;
    uint32_t reserved;
    uint64_t baseA, baseB, baseC;
    uint64_t strideA, strideB, strideC;
};
static_assert(sizeof(Small8StridedPacket) == 96, "Strided AIV packet ABI changed");
static_assert(sizeof(void*) == 8, "This implementation requires 64-bit addresses");
// Device code uses the same non-overflowing division formula, without STL headers.
constexpr uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    if (y == 0)
        return 0;
    return x / y + (x % y != 0);
}
} // namespace sgemm_gb
