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
// Host-side Cube tiling and launch integration for arch22.
#include "sgemm_gb_plan.h"
#include "sgemm_gb_cube_packet.h"
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
void sgemm_gb_cube_do(sgemm_gb::CubePacket packet, uint32_t blocks, void* stream);
void sgemm_gb_cube_offset_do(sgemm_gb::CubeOffsetPacket packet, uint32_t blocks, void* stream);
void sgemm_gb_cube_groups_do(sgemm_gb::CubeGroupPacket packet, uint32_t blocks, void* stream);
namespace sgemm_gb {
struct CubePlan {
    bool enabled = false;
    uint32_t blocks = 0;
    uint32_t tileN = 0;
    uint32_t tileM = 0;
    uint32_t cores = 0;
    alignas(8) uint32_t tiling[kCubeTilingWords]{};
};
struct CubeTileSelection {
    uint32_t tileN = 0, tileM = 0, thinBlock = 0;
    bool wholeSmall = false, shortStrip = false, thinShort = false;
};
inline void SelectInitialCubeTile(const Packet& packet, uint32_t cores, CubeTileSelection& choice)
{
    const auto& g = packet.g;
    auto& tileN = choice.tileN;
    auto& tileM = choice.tileM;
    uint32_t tile = g.m % kCubeTile == 0 && g.n % kCubeTile == 0 ? kCubeTile :
                    g.m % 256 == 0 && g.n % 256 == 0             ? 256 :
                    g.m % 128 == 0 && g.n % 128 == 0             ? 128 :
                    g.m % 64 == 0 && g.n % 64 == 0               ? 64 :
                    g.m % 32 == 0 && g.n % 32 == 0               ? 32 :
                                                                   16;
    // A full 512/256 tile can leave most AIC cores idle for a small batch.
    // Use 128 tiles when needed to expose at least one output tile per core;
    // retain the existing large-tile path once that parallelism is present.
    while (tile > 128 && uint64_t(g.m / tile) * (g.n / tile) * packet.count < cores)
        tile /= 2;
    tileN = tile;
    tileM = tile;
    // Narrow matrices otherwise launch thousands of tiny square tasks.
    // Keep full, aligned tiles in each dimension without crossing its GM edge.
    if (std::max(g.m, g.n) >= 256 && uint64_t(std::max(g.m, g.n)) >= 4ULL * std::min(g.m, g.n)) {
        tileN = 512;
        tileM = 512;
        while (g.n % tileN != 0)
            tileN /= 2;
        while (g.m % tileM != 0)
            tileM /= 2;
        // 96 is a valid 16-aligned output extent. Three 32-wide tiles
        // otherwise repeat the long-K setup for each narrow matrix.
        if (g.n < 512 && g.n % 96 == 0 && tileN < 96)
            tileN = 96;
        if (g.m < 512 && g.m % 96 == 0 && tileM < 96)
            tileM = 96;
    }
}
inline void TuneKHeavyCubeTile(const Packet& packet, uint32_t cores, CubeTileSelection& choice)
{
    const auto& g = packet.g;
    auto& tileN = choice.tileN;
    auto& tileM = choice.tileM;
    (void)cores;
    // Mid-sized K-heavy groups otherwise pay for many 32/128-wide
    // output tasks. Keep at least one task per core at the group level.
    if (g.k >= 512 && g.m % 192 == 0 && g.n % 192 == 0 && g.m >= 384 && g.n >= 384 && g.m <= 768 && g.n <= 768) {
        tileM = 192;
        tileN = 192;
    } else if (g.k >= 512 && g.m % 96 == 0 && g.n % 96 == 0 && std::min(g.m, g.n) <= 192 && std::max(g.m, g.n) <= 384) {
        tileM = 96;
        tileN = 96;
    }
}
inline void TuneMidBatchCubeTile(const Packet& packet, uint32_t cores, CubeTileSelection& choice)
{
    const auto& g = packet.g;
    auto& tileN = choice.tileN;
    auto& tileM = choice.tileM;
    (void)cores;
    // A full mid-size matrix per task keeps one batch on one AIC when
    // there are already enough independent matrices to occupy the device.
    if (g.k >= 512 && g.m >= 256 && g.n >= 256 && g.m <= 384 && g.n <= 384 && packet.count >= 32) {
        tileM = g.m;
        tileN = g.n;
    }
    if (g.k >= 384 && g.m == 192 && g.n == 192 && packet.count >= 32) {
        tileM = 192;
        tileN = 192;
    }
    if (g.k >= 128 && g.m >= 256 && g.m <= 512 && g.n == 96 && packet.count >= 32) {
        tileM = g.m;
        tileN = 96;
    }
}
inline void TuneShortBatchCubeTile(const Packet& packet, uint32_t cores, CubeTileSelection& choice)
{
    const auto& g = packet.g;
    auto& tileN = choice.tileN;
    auto& tileM = choice.tileM;
    // For short reductions, independent batches already occupy the AICs.
    // Avoid paying the input/setup cost twice for one wide output.
    if (g.k <= 128 && g.m >= 512 && g.n >= 256 && g.m <= 1024 && g.n <= 512 && packet.count >= 32) {
        tileM = g.m;
        tileN = g.n;
    }
    if (g.k <= 128 && g.m >= 1536 && g.n >= 384 && g.m % 768 == 0 && g.n % 384 == 0 && packet.count <= 8 &&
        uint64_t(g.m / 768) * (g.n / 384) * packet.count >= cores / 2) {
        tileM = 768;
        tileN = 384;
    }
}
inline void TuneShortWideCubeTile(const Packet& packet, uint32_t cores, CubeTileSelection& choice)
{
    const auto& g = packet.g;
    auto& tileN = choice.tileN;
    auto& tileM = choice.tileM;
    (void)cores;
    if (g.k <= 256 && g.m >= 512 && g.m % 512 == 0 && g.n == 192 && packet.count >= 32) {
        tileM = 512;
        tileN = 192;
    }
    if (g.k <= 32 && g.m >= 64 && g.m <= 96 && g.n >= 1024 && g.n % 1024 == 0 && packet.count >= 32) {
        tileM = g.m;
        tileN = 1024;
    }
}
inline void TuneShortThinCubeTile(const Packet& packet, uint32_t cores, CubeTileSelection& choice)
{
    const auto& g = packet.g;
    auto& tileN = choice.tileN;
    auto& tileM = choice.tileM;
    (void)cores;
    if (std::min(g.m, g.n) >= 128 && std::min(g.m, g.n) <= 192 && std::max(g.m, g.n) <= 1024 && g.k <= 256 &&
        std::max(g.m, g.n) >= 4 * std::min(g.m, g.n) && packet.count >= 32) {
        tileM = g.m;
        tileN = g.n;
    }
}
inline void TuneLargeCubeTile(const Packet& packet, uint32_t cores, CubeTileSelection& choice)
{
    const auto& g = packet.g;
    auto& tileN = choice.tileN;
    auto& tileM = choice.tileM;
    // Reuse the full matrix on one AIC when enough independent batches
    // occupy the device; short reductions keep their separate tuning.
    if (g.k >= 64 && packet.count >= cores) {
        tileM = g.m;
        tileN = g.n;
    }
    // With 64 large matrices and 20 AICs, a whole-matrix task takes four
    // waves, the last with only four active cores. Split only when this
    // reduces the busiest core's estimated work by at least five percent.
    if (g.k >= 2048 && g.m >= 2048 && g.n >= 2048 && packet.count >= cores) {
        uint32_t parts = 1;
        uint64_t waves = CeilDiv(uint64_t(packet.count), cores);
        for (uint32_t candidate : {2U, 4U}) {
            if (g.m % candidate != 0 || g.m / candidate < 1024)
                continue;
            const uint64_t next = CeilDiv(uint64_t(packet.count) * candidate, cores);
            if (next * parts * 100 < waves * candidate * 95) {
                parts = candidate;
                waves = next;
            }
        }
        tileM = g.m / parts;
        tileN = g.n;
    }
}
inline void FinalizeCubeTile(const Packet& packet, uint32_t cores, CubeTileSelection& choice)
{
    const auto& g = packet.g;
    auto& tileN = choice.tileN;
    auto& tileM = choice.tileM;
    auto& wholeSmall = choice.wholeSmall;
    auto& shortStrip = choice.shortStrip;
    auto& thinShort = choice.thinShort;
    auto& thinBlock = choice.thinBlock;
    wholeSmall = g.m >= 96 && g.n >= 96 && g.m <= 256 && g.n <= 256 && g.k <= 256 && packet.count < cores;
    if (wholeSmall) {
        tileM = g.m;
        tileN = g.n;
    }
    // In the transposed Cube view these outputs have only 16/32 rows.
    // Widen the contiguous store to avoid four separate 128-float writes.
    // Enough independent batches allow one complete matrix per AIC task.
    shortStrip = g.k <= 32 && g.n <= 32 && g.m >= 512;
    if (shortStrip && packet.count >= cores) {
        tileM = g.m;
        tileN = g.n;
    }
    thinShort = g.k <= 32 && std::min(g.m, g.n) >= 64 && std::min(g.m, g.n) <= 384 && std::max(g.m, g.n) >= 512 &&
                g.m % 32 == 0 && g.n % 32 == 0 && packet.count >= cores;
    if (thinShort) {
        tileM = g.m;
        tileN = g.n;
    }
    const uint32_t thinSide = std::min(g.m, g.n);
    thinBlock = thinSide % 128 == 0 ? 128 : (thinSide % 96 == 0 ? 96 : 64);
}
inline CubeTileSelection SelectCubeTile(const Packet& packet, uint32_t cores)
{
    CubeTileSelection choice{};
    SelectInitialCubeTile(packet, cores, choice);
    TuneKHeavyCubeTile(packet, cores, choice);
    TuneMidBatchCubeTile(packet, cores, choice);
    TuneShortBatchCubeTile(packet, cores, choice);
    TuneShortWideCubeTile(packet, cores, choice);
    TuneShortThinCubeTile(packet, cores, choice);
    TuneLargeCubeTile(packet, cores, choice);
    FinalizeCubeTile(packet, cores, choice);
    return choice;
}
template <class Platform>
inline CubePlan BuildCubePlan(const Packet& packet, uint32_t cores, Platform& platform, const CubeTileSelection& choice)
{
    CubePlan plan{};
    const auto& g = packet.g;
    const uint32_t tileN = choice.tileN, tileM = choice.tileM, thinBlock = choice.thinBlock;
    const bool wholeSmall = choice.wholeSmall, shortStrip = choice.shortStrip, thinShort = choice.thinShort;
    matmul_tiling::MatmulApiTiling tiler(platform);
    // C_col = A_col * B_col becomes C_row^T = op(B_col)^T * op(A_col)^T.
    // Transpose flags therefore SWAP with the operands; do not toggle either flag.
    tiler.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, g.tb != 0);
    tiler.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, g.ta != 0);
    tiler.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiler.SetBias(false);
    tiler.SetOrgShape(g.n, g.m, g.k);
    // Each kernel task owns one explicit output tile.  The internal
    // matmul split must fit A2's on-chip storage independently of that
    // output tile size.
    const int32_t partialK = g.k >= 2048 && g.k % 1024 == 0 ? 512 : g.k;
    tiler.SetShape(tileN, tileM, partialK);
    // CANN 9.1 SetFixSplit documents baseK=-1 as the supported automatic mode.
    const uint32_t baseM =
        thinShort ? (g.n <= 384 ? thinBlock : 256u) : (wholeSmall ? tileN : (tileN <= 96 ? tileN : 128));
    const uint32_t baseN =
        thinShort ?
            (g.n <= 384 ? 256u : thinBlock) :
            (shortStrip ? std::min(tileM, 512u) :
                          (wholeSmall ? (tileN * tileM > 32768 ? tileM / 2 : tileM) : (tileM <= 96 ? tileM : 128)));
    if (tiler.SetFixSplit(baseM, baseN, wholeSmall || thinShort || tileN <= 96 || tileM <= 96 ? -1 : 128) != 0)
        throw std::runtime_error("FP32 Cube fixed split rejected");
    tiler.SetBufferSpace(-1, -1, -1);
    AscendC::tiling::TCubeTiling tc{};
    static_assert(sizeof(tc) <= sizeof(plan.tiling), "Increase tiling wire area on both sides");
    static_assert(alignof(decltype(tc)) <= 8, "TCubeTiling alignment exceeds wire format");
    if (tiler.GetTiling(tc) != 0)
        throw std::runtime_error("FP32 Cube tiling failed; no silent AIV perf fallback");
    std::copy_n(reinterpret_cast<const unsigned char*>(&tc), sizeof(tc), reinterpret_cast<unsigned char*>(plan.tiling));
    const uint64_t tasks = uint64_t(g.n / tileN) * (g.m / tileM) * packet.count;
    plan.blocks = static_cast<uint32_t>(std::min<uint64_t>(cores, tasks));
    plan.cores = cores;
    plan.tileN = tileN;
    plan.tileM = tileM;
    plan.enabled = true;
    return plan;
}
inline std::vector<CubePlan> PrepareCubePlans(const std::vector<Packet>& packets)
{
    std::vector<CubePlan> plans(packets.size());
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform)
        throw std::runtime_error("Ascend platform unavailable for Cube tiling");
    const uint32_t cores = platform->GetCoreNumAic();
    if (!cores)
        throw std::runtime_error("zero AIC core count");
    for (size_t i = 0; i < packets.size(); ++i) {
        if (!CubeEligible(packets[i]))
            continue;
        const auto choice = SelectCubeTile(packets[i], cores);
        plans[i] = BuildCubePlan(packets[i], cores, *platform, choice);
    }
    return plans;
}
inline bool SmallGroupsOverlap(
    const CubeGroupPacket& bundle, const uint64_t (&sizeA)[kSmallCubeMatrices],
    const uint64_t (&sizeB)[kSmallCubeMatrices], const uint64_t (&sizeC)[kSmallCubeMatrices])
{
    const auto overlaps = [](uint64_t x, uint64_t xs, uint64_t y, uint64_t ys) {
        // Subtraction avoids overflowing either endpoint.
        return x <= y ? y - x < xs : x - y < ys;
    };
    for (uint32_t i = 0; i < bundle.count; ++i)
        for (uint32_t j = 0; j < bundle.count; ++j)
            if ((i != j && overlaps(bundle.c[i], sizeC[i], bundle.c[j], sizeC[j])) ||
                overlaps(bundle.c[i], sizeC[i], bundle.a[j], sizeA[j]) ||
                overlaps(bundle.c[i], sizeC[i], bundle.b[j], sizeB[j]))
                return true;
    return false;
}
inline size_t LaunchSmallGroups(
    const std::vector<Packet>& packets, const std::vector<CubePlan>& plans, size_t begin, void* stream)
{
    CubeGroupPacket bundle{};
    uint64_t sizeA[kSmallCubeMatrices]{}, sizeB[kSmallCubeMatrices]{}, sizeC[kSmallCubeMatrices]{};
    for (size_t i = begin; i < packets.size() && bundle.groups < kSmallCubeGroups; ++i) {
        const auto& p = packets[i];
        const auto& plan = plans[i];
        if (!plan.enabled || p.count > 8 || !p.count || p.g.m > 256 || p.g.n > 256 || p.g.k > 256 ||
            plan.tileM != p.g.m || plan.tileN != p.g.n || bundle.count + p.count > kSmallCubeMatrices)
            break;
        const uint32_t group = bundle.groups++;
        bundle.g[group] = p.g;
        bundle.starts[group] = bundle.count;
        std::copy_n(plan.tiling, kCubeTilingWords, bundle.tiling[group]);
        for (uint32_t j = 0; j < p.count; ++j) {
            const uint32_t at = bundle.count++;
            bundle.a[at] = p.a[j];
            bundle.b[at] = p.b[j];
            bundle.c[at] = p.c[j];
            sizeA[at] = uint64_t(p.g.m) * p.g.k * sizeof(float);
            sizeB[at] = uint64_t(p.g.n) * p.g.k * sizeof(float);
            sizeC[at] = uint64_t(p.g.m) * p.g.n * sizeof(float);
        }
    }
    if (bundle.groups < 2)
        return 0;
    bundle.starts[bundle.groups] = bundle.count;
    if (SmallGroupsOverlap(bundle, sizeA, sizeB, sizeC))
        return 0;
    sgemm_gb_cube_groups_do(bundle, std::min(plans[begin].cores, bundle.count), stream);
    return bundle.groups;
}
inline void LaunchCube(const Packet& matrices, const CubePlan& plan, void* stream)
{
    CubePacket p{};
    p.matrices = matrices;
    p.tileN = plan.tileN;
    p.tileM = plan.tileM;
    std::copy_n(plan.tiling, kCubeTilingWords, p.tiling);
    sgemm_gb_cube_do(p, plan.blocks, stream);
}
inline bool LaunchCombinedCube(
    const Packet& first, const Packet& second, const CubePlan& plan, const CubePlan& next, void* stream)
{
    if (!plan.enabled || !next.enabled || first.reserved != second.reserved ||
        first.count + second.count > kCubeOffsetBatches || plan.tileM != next.tileM || plan.tileN != next.tileN ||
        std::memcmp(&first.g, &second.g, sizeof(Group)) != 0 ||
        std::memcmp(plan.tiling, next.tiling, sizeof(plan.tiling)) != 0)
        return false;
    CubeOffsetPacket p{};
    p.matrices.g = first.g;
    p.matrices.count = first.count + second.count;
    p.matrices.reserved = first.reserved;
    p.tileM = plan.tileM;
    p.tileN = plan.tileN;
    std::copy_n(plan.tiling, kCubeTilingWords, p.tiling);
    uint64_t baseA = std::numeric_limits<uint64_t>::max();
    uint64_t baseB = std::numeric_limits<uint64_t>::max();
    uint64_t baseC = std::numeric_limits<uint64_t>::max();
    for (const Packet* part : {&first, &second})
        for (uint32_t i = 0; i < part->count; ++i) {
            baseA = std::min(baseA, part->a[i]);
            baseB = std::min(baseB, part->b[i]);
            baseC = std::min(baseC, part->c[i]);
        }
    uint32_t dest = 0;
    for (const Packet* part : {&first, &second})
        for (uint32_t i = 0; i < part->count; ++i) {
            const uint64_t da = part->a[i] - baseA;
            const uint64_t db = part->b[i] - baseB;
            const uint64_t dc = part->c[i] - baseC;
            if (da > std::numeric_limits<uint32_t>::max() || db > std::numeric_limits<uint32_t>::max() ||
                dc > std::numeric_limits<uint32_t>::max())
                return false;
            p.matrices.a[dest] = static_cast<uint32_t>(da);
            p.matrices.b[dest] = static_cast<uint32_t>(db);
            p.matrices.c[dest] = static_cast<uint32_t>(dc);
            ++dest;
        }
    p.matrices.baseA = baseA;
    p.matrices.baseB = baseB;
    p.matrices.baseC = baseC;
    const uint64_t tasks = uint64_t(p.matrices.count) * (first.g.n / plan.tileN) * (first.g.m / plan.tileM);
    sgemm_gb_cube_offset_do(p, static_cast<uint32_t>(std::min<uint64_t>(plan.cores, tasks)), stream);
    return true;
}
} // namespace sgemm_gb
