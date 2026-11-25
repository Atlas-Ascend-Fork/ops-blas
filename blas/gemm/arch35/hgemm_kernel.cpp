/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "hgemm_kernel.h"
#define KERNEL_UTILS_LITE
#include "common/arch/hardware.h"
#include "common/helper/kernel_utils.h"
#include "tensor_api/tensor.h"

namespace te = AscendC::Te;

// ============================================================================
// Hgemm tile constants (FLOAT16 operands, FLOAT32 accumulation in L0C).
// A/B use NZ/ZN fractal layouts with C0 = 16 halves (32 bytes). One L0C
// group holds a (GROUP_M x GROUP_N) block of BM x BN tiles (256 x 256
// accumulator elements); the A/B L0 tiles are shared by every tile of the
// group, giving 64 MAC per staged byte at this group shape, the
// compute/bandwidth balance point for the 4096 MAC/cycle cube. Each L1
// pipeline stage holds one full K-chunk for the whole group (128KB: 256x128
// A + 128x256 B), arranged as a 4-deep ring across the 512KB L1 so several
// GM->L1 DMAs stay in flight and L2-miss latency on panels larger than the
// 128MB L2 (e.g. 8192^3) is absorbed. L0A/L0B hold per-subtile slots for
// one BK=64 Mmad k-step each (2-deep ring): one Mmad of (128 x 128 x 64)
// per tile per k-step keeps the instruction and L1->L0 request rate low.
// ============================================================================
constexpr uint32_t HGEMM_BASE_M = 128;
constexpr uint32_t HGEMM_BASE_K = 64;
constexpr uint32_t HGEMM_BASE_N = 128;
constexpr uint32_t HGEMM_TILE_K_CHUNK = 128;
constexpr uint32_t HGEMM_GROUP_M = 2;
constexpr uint32_t HGEMM_GROUP_N = 2;
// Ascend 950 (dav-3510) L1 is 512KB (CANN kernel_utils_constants.h
// TOTAL_L1_SIZE); the shared hardware.h value is stale for this arch.
constexpr uint64_t HGEMM_L1_SIZE = 512 * 1024;
// One L1 stage holds the whole group's K-chunk operands:
//   A: (GROUP_M*BM) x K_CHUNK halves, B: K_CHUNK x (GROUP_N*BN) halves.
constexpr uint64_t HGEMM_L1_STAGE_BYTES = static_cast<uint64_t>(HGEMM_TILE_K_CHUNK) *
                                          (HGEMM_GROUP_M * HGEMM_BASE_M + HGEMM_GROUP_N * HGEMM_BASE_N) * sizeof(half);
// Deeper L1 rings give MTE2 more slack to absorb L2-miss latency on panels
// that exceed the 128MB L2 (e.g. 8192^3: A+B = 256MB).
constexpr uint32_t HGEMM_L1_STAGES = static_cast<uint32_t>(HGEMM_L1_SIZE / HGEMM_L1_STAGE_BYTES);
constexpr uint32_t HGEMM_L0_SUB_K = 64;
// L0 ring: one stage holds the group's A and B subtiles for one BK k-step
// (GROUP_M slots of BM x BK + GROUP_N slots of BK x BN halves = 32KB per
// side). A 2-deep ring keeps the Mmad stream of k-step i overlapping the
// L1->L0 copies of k-step i+1.
constexpr uint32_t HGEMM_L0_STAGES = 2;
constexpr uint32_t HGEMM_FP16_C0 = 16; // 32B / sizeof(half)
constexpr uint32_t HGEMM_L0C_C0 = 16;  // L0C FLOAT32 accumulator fractal width
constexpr uint32_t HGEMM_L0C_TILE_BYTES = HGEMM_BASE_M * HGEMM_BASE_N * sizeof(float);

constexpr uint32_t HGEMM_L0C_FINAL = 3;
constexpr uint32_t HGEMM_L0C_NON_FINAL = 2;
constexpr uint32_t HGEMM_BUFFER_COUNT = 2;
constexpr uint64_t HGEMM_BUFFER_MASK = HGEMM_BUFFER_COUNT - 1;

struct HgemmCubeState {
    GemmTilingData tiling;
    uint32_t mBlockIdx;
    uint32_t nBlockIdx;
    uint64_t mOffset;
    uint64_t nOffset;
    uint32_t actualM;
    uint32_t actualN;
    uint32_t baseMCount;
    uint32_t tailM;
    uint32_t baseNCount;
    uint32_t tailN;
    uint32_t mLoopCount;
    uint32_t nLoopCount;
    uint32_t mGroupSpan;
    uint32_t mGroupCount;
    uint32_t nGroupSpan;
    uint32_t nGroupCount;
};

__aicore__ inline void HgemmComputeBalancedAxis(
    uint32_t length, uint32_t baseTile, uint32_t blockCount, uint32_t blockIdx, uint64_t& elementOffset,
    uint32_t& actualLength)
{
    if (baseTile == 0 || blockCount == 0) {
        elementOffset = 0;
        actualLength = 0;
        return;
    }
    uint32_t tileCount = CeilDiv(length, baseTile);
    uint32_t tilesPerBlock = tileCount / blockCount;
    uint32_t extraTileBlocks = tileCount % blockCount;
    uint32_t currentTileCount = tilesPerBlock + (blockIdx < extraTileBlocks ? 1U : 0U);
    uint32_t startTile = blockIdx * tilesPerBlock + (blockIdx < extraTileBlocks ? blockIdx : extraTileBlocks);
    elementOffset = static_cast<uint64_t>(startTile) * baseTile;
    uint32_t remaining = length - static_cast<uint32_t>(elementOffset);
    uint32_t assignedLength = currentTileCount * baseTile;
    actualLength = assignedLength < remaining ? assignedLength : remaining;
}

template <uint32_t BM, uint32_t BK, uint32_t BN, uint32_t GROUP_M, uint32_t GROUP_N>
__aicore__ inline bool InitHgemmState(HgemmCubeState& st, const GemmTilingData& tiling)
{
    st.tiling = tiling;
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(st.tiling.usedCoreNum)) {
        return false;
    }
    if (st.tiling.mBlocks == 0 || st.tiling.nBlocks == 0) {
        return false;
    }
    st.mBlockIdx = AscendC::GetBlockIdx() % static_cast<uint32_t>(st.tiling.mBlocks);
    st.nBlockIdx = AscendC::GetBlockIdx() / static_cast<uint32_t>(st.tiling.mBlocks);
    HgemmComputeBalancedAxis(
        static_cast<uint32_t>(st.tiling.m), BM, static_cast<uint32_t>(st.tiling.mBlocks), st.mBlockIdx, st.mOffset,
        st.actualM);
    HgemmComputeBalancedAxis(
        static_cast<uint32_t>(st.tiling.n), BN, static_cast<uint32_t>(st.tiling.nBlocks), st.nBlockIdx, st.nOffset,
        st.actualN);
    if (st.actualM == 0 || st.actualN == 0) {
        return false;
    }
    st.baseMCount = st.actualM / BM;
    st.tailM = st.actualM % BM;
    st.baseNCount = st.actualN / BN;
    st.tailN = st.actualN % BN;
    st.mLoopCount = CeilDiv(st.actualM, BM);
    st.nLoopCount = CeilDiv(st.actualN, BN);
    st.mGroupSpan = st.mLoopCount < GROUP_M ? st.mLoopCount : GROUP_M;
    st.mGroupCount = CeilDiv(st.mLoopCount, st.mGroupSpan);
    st.nGroupSpan = st.nLoopCount < GROUP_N ? st.nLoopCount : GROUP_N;
    st.nGroupCount = CeilDiv(st.nLoopCount, st.nGroupSpan);
    return true;
}

// Per-tile logical extent helpers (tile idx 0..loopCount-1; the tile at
// baseCount carries the tail when actual is not a multiple of the base).
__aicore__ inline uint32_t HgemmTileLen(uint32_t tileIdx, uint32_t baseCount, uint32_t tail, uint32_t baseLen)
{
    return (tileIdx != baseCount) ? baseLen : tail;
}

// ============================================================================
// tensor_api data movement and compute primitives (FLOAT16 operands)
// ============================================================================

template <uint32_t BM, uint32_t BK, typename GM_TENSOR_A>
__aicore__ inline void HgemmCopyGM2L1A(
    GM_TENSOR_A& gmATensor, uint64_t l1OffsetBytes, uint64_t mPos, uint32_t groupMRows, uint32_t kOffset,
    uint32_t curKChunk)
{
    using T = half;
    auto gmBlock =
        gmATensor.Slice(te::MakeCoord(mPos, static_cast<uint64_t>(kOffset)), te::MakeShape(groupMRows, curKChunk));
    auto tensorAL1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1OffsetBytes),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<HGEMM_FP16_C0>>(groupMRows, curKChunk));
    te::Copy(te::MakeCopy(te::CopyGM2L1{}), tensorAL1, gmBlock);
}

template <uint32_t BK, uint32_t BN, typename GM_TENSOR_B>
__aicore__ inline void HgemmCopyGM2L1B(
    GM_TENSOR_B& gmBTensor, uint64_t l1OffsetBytes, uint64_t nPos, uint32_t groupNCols, uint32_t kOffset,
    uint32_t curKChunk)
{
    using T = half;
    auto gmBlock =
        gmBTensor.Slice(te::MakeCoord(static_cast<uint64_t>(kOffset), nPos), te::MakeShape(curKChunk, groupNCols));
    auto tensorBL1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1OffsetBytes),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<HGEMM_FP16_C0>>(curKChunk, groupNCols));
    te::Copy(te::MakeCopy(te::CopyGM2L1{}), tensorBL1, gmBlock);
}

template <uint32_t BM, uint32_t BK>
__aicore__ inline void HgemmCopyL1ToL0TileA(
    uint64_t l1OffsetBytes, uint64_t l0OffsetBytes, uint32_t groupMRows, uint32_t curKChunk, uint32_t kSub,
    uint32_t subK, uint32_t curM, uint32_t miSub)
{
    using T = half;
    auto tensorAL1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1OffsetBytes),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<HGEMM_FP16_C0>>(groupMRows, curKChunk));
    auto tensorAL1Tile = tensorAL1.Slice(
        te::MakeCoord(static_cast<uint64_t>(miSub) * BM, static_cast<uint64_t>(kSub)), te::MakeShape(curM, subK));
    // Destination layout mirrors the Mmad operand exactly (same shape at the
    // same base), which keeps the NZ fractal strides consistent.
    auto tensorAL0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0A, T>(l0OffsetBytes),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<HGEMM_FP16_C0>>(curM, subK));
    te::Copy(te::MakeCopy(te::CopyL12L0A{}), tensorAL0, tensorAL1Tile);
}

template <uint32_t BK, uint32_t BN>
__aicore__ inline void HgemmCopyL1ToL0TileB(
    uint64_t l1OffsetBytes, uint64_t l0OffsetBytes, uint32_t curKChunk, uint32_t groupNCols, uint32_t kSub,
    uint32_t subK, uint32_t curN, uint32_t niSub)
{
    using T = half;
    auto tensorBL1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1OffsetBytes),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<HGEMM_FP16_C0>>(curKChunk, groupNCols));
    auto tensorBL1Tile = tensorBL1.Slice(
        te::MakeCoord(static_cast<uint64_t>(kSub), static_cast<uint64_t>(niSub) * BN), te::MakeShape(subK, curN));
    // Destination layout mirrors the Mmad operand exactly (same shape at the
    // same base), which keeps the ZN fractal strides consistent.
    auto tensorBL0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0B, T>(l0OffsetBytes),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<HGEMM_FP16_C0>>(subK, curN));
    te::Copy(te::MakeCopy(te::CopyL12L0B{}), tensorBL0, tensorBL1Tile);
}

template <uint32_t BM, uint32_t BK, uint32_t BN>
__aicore__ inline void HgemmMmadTile(
    uint64_t l0OffsetA, uint64_t l0OffsetB, uint64_t l0cOffset, uint32_t curM, uint32_t curK, uint32_t curN,
    bool isFirstK, bool isLastK)
{
    using T = half;
    auto tensorAL0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0A, T>(l0OffsetA),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<HGEMM_FP16_C0>>(curM, curK));
    auto tensorBL0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0B, T>(l0OffsetB),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<HGEMM_FP16_C0>>(curK, curN));
    auto tensorL0C = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, float>(l0cOffset),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<HGEMM_L0C_C0>>(curM, curN));

    uint8_t unitFlag = isLastK ? HGEMM_L0C_FINAL : HGEMM_L0C_NON_FINAL;
    te::MmadParams mmadParams{
        static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(curK), unitFlag, isFirstK};
    te::Mmad(te::MmadAtom<te::MmadTraits<te::MmadOperation>>{}.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
}

template <uint32_t BM, uint32_t BN, typename GM_TENSOR_C>
__aicore__ inline void HgemmWriteFixpipeBlock(
    const HgemmCubeState& st, GM_TENSOR_C& gmCTensor, uint32_t mi, uint32_t ni, uint64_t l0cOffset)
{
    uint32_t curM = HgemmTileLen(mi, st.baseMCount, st.tailM, BM);
    uint32_t curN = HgemmTileLen(ni, st.baseNCount, st.tailN, BN);
    uint64_t mOff = st.mOffset + static_cast<uint64_t>(mi) * BM;
    uint64_t nOff = st.nOffset + static_cast<uint64_t>(ni) * BN;
    auto gmBlockC = gmCTensor.Slice(te::MakeCoord(mOff, nOff), te::MakeShape(curM, curN));
    auto tensorL0C = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, float>(l0cOffset),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<HGEMM_L0C_C0>>(curM, curN));
    te::MakeCopy(te::CopyL0C2GM{}).Call(gmBlockC, tensorL0C, te::FixpipeParams{HGEMM_L0C_FINAL});
}

// Fixpipe an accumulated group out to GM: drains the Mmad queue first so the
// L0C slots are stable, then issues the per-tile L0C->GM copies.
template <uint32_t BM, uint32_t BN, uint32_t GROUP_N, typename TensorC>
__aicore__ inline void HgemmFlushGroup(
    const HgemmCubeState& st, TensorC& gmCTensor, uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd)
{
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
    for (uint32_t miSub = mStart; miSub < mEnd; miSub++) {
        for (uint32_t niSub = nStart; niSub < nEnd; niSub++) {
            uint64_t l0cOffset =
                static_cast<uint64_t>((miSub - mStart) * GROUP_N + (niSub - nStart)) * HGEMM_L0C_TILE_BYTES;
            HgemmWriteFixpipeBlock<BM, BN>(st, gmCTensor, miSub, niSub, l0cOffset);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
}

// ============================================================================
// L0 staging of one k-step: copies the group's A and B subtiles from L1 into
// the ping-pong ring slots, then issues the (GROUP_M x GROUP_N) Mmad sweep
// over the staged tiles. Each copy destination mirrors the matching Mmad
// operand (same layout, same base) so the fractal strides agree.
// ============================================================================
template <uint32_t BM, uint32_t BK, uint32_t BN, uint32_t GROUP_N>
__aicore__ inline void HgemmStageL0AndCompute(
    const HgemmCubeState& st, uint64_t l1OffsetA, uint64_t l1OffsetB, uint64_t l0BaseA, uint64_t l0BaseB,
    uint32_t mStart, uint32_t nStart, uint32_t groupMTiles, uint32_t groupNTiles, uint32_t groupMRows,
    uint32_t groupNCols, uint32_t curKChunk, uint32_t kSub, uint32_t subK, bool isFirstK, bool isLastK,
    bool& fixpipeWaited)
{
    constexpr uint64_t L0_TILE_BYTES = BM * BK * sizeof(half); // per-subtile L0 slot
    uint16_t l0Stage =
        static_cast<uint16_t>(l0BaseA / (HardwareInfo<ArchType::ASCEND_V350>::l0ASize / HGEMM_L0_STAGES));

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Stage);
    for (uint32_t miSub = 0; miSub < groupMTiles; miSub++) {
        uint32_t curM = HgemmTileLen(mStart + miSub, st.baseMCount, st.tailM, BM);
        HgemmCopyL1ToL0TileA<BM, BK>(
            l1OffsetA, l0BaseA + static_cast<uint64_t>(miSub) * L0_TILE_BYTES, groupMRows, curKChunk, kSub, subK, curM,
            miSub);
    }
    for (uint32_t niSub = 0; niSub < groupNTiles; niSub++) {
        uint32_t curN = HgemmTileLen(nStart + niSub, st.baseNCount, st.tailN, BN);
        HgemmCopyL1ToL0TileB<BK, BN>(
            l1OffsetB, l0BaseB + static_cast<uint64_t>(niSub) * L0_TILE_BYTES, curKChunk, groupNCols, kSub, subK, curN,
            niSub);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Stage);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Stage);

    for (uint32_t miSub = 0; miSub < groupMTiles; miSub++) {
        uint32_t curM = HgemmTileLen(mStart + miSub, st.baseMCount, st.tailM, BM);
        uint64_t l0OffsetA = l0BaseA + static_cast<uint64_t>(miSub) * L0_TILE_BYTES;
        for (uint32_t niSub = 0; niSub < groupNTiles; niSub++) {
            uint32_t curN = HgemmTileLen(nStart + niSub, st.baseNCount, st.tailN, BN);
            uint64_t l0OffsetB = l0BaseB + static_cast<uint64_t>(niSub) * L0_TILE_BYTES;
            uint64_t l0cOffset = static_cast<uint64_t>(miSub * GROUP_N + niSub) * HGEMM_L0C_TILE_BYTES;
            // The previous group's Fixpipe drains while this group's first K
            // chunk stages through L1/L0; only the first Mmad of the group
            // needs to wait for it.
            if (!fixpipeWaited) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
                fixpipeWaited = true;
            }
            HgemmMmadTile<BM, BK, BN>(l0OffsetA, l0OffsetB, l0cOffset, curM, subK, curN, isFirstK, isLastK);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Stage);
}

// ============================================================================
// One L0C group: a (GROUP_M x GROUP_N) block of BM x BN tiles accumulated
// over the full K axis. L0A/L0B subtiles are shared by every tile of the
// group; the L1 ring buffers stage K chunks ahead of the Mmad stream. The
// group's own Fixpipe is deferred to the next group's first K chunk (see
// HgemmFlushGroup) so it overlaps the next staging phase.
// ============================================================================
template <
    uint32_t BM, uint32_t BK, uint32_t BN, uint32_t GROUP_M, uint32_t GROUP_N, bool IsTransA, bool IsTransB,
    typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void ProcessHgemmGroup(
    const HgemmCubeState& st, TensorA gmATensor, TensorB gmBTensor, TensorC gmCTensor, uint32_t mStart, uint32_t mEnd,
    uint32_t nStart, uint32_t nEnd, bool hasPrevGroup, uint32_t prevMStart, uint32_t prevMEnd, uint32_t prevNStart,
    uint32_t prevNEnd, uint64_t& l1PingPong, uint64_t& l0PingPong)
{
    constexpr uint64_t L1_STAGE_BYTES = HGEMM_L1_STAGE_BYTES;
    constexpr uint64_t L0A_STAGE_BYTES = HardwareInfo<ArchType::ASCEND_V350>::l0ASize / HGEMM_L0_STAGES;
    constexpr uint64_t L0B_STAGE_BYTES = HardwareInfo<ArchType::ASCEND_V350>::l0BSize / HGEMM_L0_STAGES;

    uint32_t kTotal = static_cast<uint32_t>(st.tiling.k);
    uint32_t groupMTiles = mEnd - mStart;
    uint32_t groupNTiles = nEnd - nStart;
    uint32_t lastCurM = HgemmTileLen(mEnd - 1, st.baseMCount, st.tailM, BM);
    uint32_t lastCurN = HgemmTileLen(nEnd - 1, st.baseNCount, st.tailN, BN);
    uint32_t groupMRows = (groupMTiles - 1) * BM + lastCurM;
    uint32_t groupNCols = (groupNTiles - 1) * BN + lastCurN;
    uint64_t mPos = st.mOffset + static_cast<uint64_t>(mStart) * BM;
    uint64_t nPos = st.nOffset + static_cast<uint64_t>(nStart) * BN;
    bool fixpipeWaited = false;
    uint32_t kChunkCount = CeilDiv(kTotal, HGEMM_TILE_K_CHUNK);

    for (uint32_t chunkIdx = 0; chunkIdx < kChunkCount; chunkIdx++) {
        uint32_t kOffset = chunkIdx * HGEMM_TILE_K_CHUNK;
        uint32_t remainingK = kTotal - kOffset;
        uint32_t curKChunk = remainingK < HGEMM_TILE_K_CHUNK ? remainingK : HGEMM_TILE_K_CHUNK;
        uint64_t l1Stage = l1PingPong % HGEMM_L1_STAGES;
        uint64_t l1OffsetA = l1Stage * L1_STAGE_BYTES;
        uint64_t l1OffsetB = l1OffsetA + GROUP_M * BM * HGEMM_TILE_K_CHUNK * sizeof(half);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1Stage));
        HgemmCopyGM2L1A<BM, BK>(gmATensor, l1OffsetA, mPos, groupMRows, kOffset, curKChunk);
        HgemmCopyGM2L1B<BK, BN>(gmBTensor, l1OffsetB, nPos, groupNCols, kOffset, curKChunk);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1Stage));

        // Flush the previous group's Fixpipe while this group's first K-chunk
        // DMA is in flight: the barrier drains the previous Mmad tail, the
        // Fixpipe then overlaps the DMA + L0 staging instead of stalling the
        // group boundary.
        if (hasPrevGroup && chunkIdx == 0) {
            HgemmFlushGroup<BM, BN, GROUP_N>(st, gmCTensor, prevMStart, prevMEnd, prevNStart, prevNEnd);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1Stage));

        for (uint32_t kSub = 0; kSub < curKChunk; kSub += HGEMM_L0_SUB_K) {
            uint32_t remainingSubK = curKChunk - kSub;
            uint32_t subK = remainingSubK < HGEMM_L0_SUB_K ? remainingSubK : HGEMM_L0_SUB_K;
            uint64_t l0Stage = l0PingPong % HGEMM_L0_STAGES;
            bool isFirstK = (chunkIdx == 0 && kSub == 0);
            bool isLastK = (chunkIdx == kChunkCount - 1 && kSub + subK == curKChunk);
            HgemmStageL0AndCompute<BM, BK, BN, GROUP_N>(
                st, l1OffsetA, l1OffsetB, l0Stage * L0A_STAGE_BYTES, l0Stage * L0B_STAGE_BYTES, mStart, nStart,
                groupMTiles, groupNTiles, groupMRows, groupNCols, curKChunk, kSub, subK, isFirstK, isLastK,
                fixpipeWaited);
            l0PingPong++;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1Stage));
        l1PingPong++;
    }
}

template <bool IsTransA, bool IsTransB, typename T, typename OutT>
__aicore__ inline auto HgemmMakeGmTensors(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, const HgemmCubeState& st)
{
    using LayoutGM_A = AscendC::Std::conditional_t<IsTransA, te::DNExtLayoutPtn, te::NDExtLayoutPtn>;
    using LayoutGM_B = AscendC::Std::conditional_t<IsTransB, te::DNExtLayoutPtn, te::NDExtLayoutPtn>;

    uint64_t aDim0 = IsTransA ? static_cast<uint64_t>(st.tiling.lda) : static_cast<uint64_t>(st.tiling.m);
    uint64_t aDim1 = IsTransA ? static_cast<uint64_t>(st.tiling.k) : static_cast<uint64_t>(st.tiling.lda);
    uint64_t bDim0 = IsTransB ? static_cast<uint64_t>(st.tiling.ldb) : static_cast<uint64_t>(st.tiling.k);
    uint64_t bDim1 = IsTransB ? static_cast<uint64_t>(st.tiling.n) : static_cast<uint64_t>(st.tiling.ldb);

    auto gmATensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ T*>(a)),
        te::MakeFrameLayout<LayoutGM_A>(aDim0, aDim1));
    auto gmBTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ T*>(b)),
        te::MakeFrameLayout<LayoutGM_B>(bDim0, bDim1));
    auto gmCTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ OutT*>(c)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(st.tiling.m), static_cast<uint64_t>(st.tiling.ldc)));
    return AscendC::Std::make_tuple(gmATensor, gmBTensor, gmCTensor);
}

__aicore__ inline void HgemmSetRingFlags()
{
    for (uint32_t stage = 0; stage < HGEMM_L1_STAGES; stage++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(stage));
    }
    for (uint32_t stage = 0; stage < HGEMM_L0_STAGES; stage++) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(stage));
    }
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
}

__aicore__ inline void HgemmWaitRingFlags()
{
    for (uint32_t stage = 0; stage < HGEMM_L1_STAGES; stage++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(stage));
    }
    for (uint32_t stage = 0; stage < HGEMM_L0_STAGES; stage++) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(stage));
    }
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
}

template <
    uint32_t BM, uint32_t BK, uint32_t BN, uint32_t GROUP_M, uint32_t GROUP_N, bool IsTransA, bool IsTransB,
    typename OutT>
__aicore__ inline void HgemmCubeKernelImpl(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, GemmTilingData& tiling)
{
    using T = half;
    HgemmCubeState st{};
    if (!InitHgemmState<BM, BK, BN, GROUP_M, GROUP_N>(st, tiling)) {
        return;
    }

    auto tensors = HgemmMakeGmTensors<IsTransA, IsTransB, T, OutT>(a, b, c, st);
    auto gmATensor = AscendC::Std::get<0>(tensors);
    auto gmBTensor = AscendC::Std::get<1>(tensors);
    auto gmCTensor = AscendC::Std::get<2>(tensors);

    HgemmSetRingFlags();
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;
    bool hasPrevGroup = false;
    uint32_t prevMStart = 0;
    uint32_t prevMEnd = 0;
    uint32_t prevNStart = 0;
    uint32_t prevNEnd = 0;

    for (uint32_t mg = 0; mg < st.mGroupCount; mg++) {
        uint32_t mStart = mg * st.mGroupSpan;
        uint32_t mEnd = (mStart + st.mGroupSpan < st.mLoopCount) ? mStart + st.mGroupSpan : st.mLoopCount;
        for (uint32_t ng = 0; ng < st.nGroupCount; ng++) {
            uint32_t nStart = ng * st.nGroupSpan;
            uint32_t nEnd = (nStart + st.nGroupSpan < st.nLoopCount) ? nStart + st.nGroupSpan : st.nLoopCount;
            ProcessHgemmGroup<BM, BK, BN, GROUP_M, GROUP_N, IsTransA, IsTransB>(
                st, gmATensor, gmBTensor, gmCTensor, mStart, mEnd, nStart, nEnd, hasPrevGroup, prevMStart, prevMEnd,
                prevNStart, prevNEnd, l1PingPong, l0PingPong);
            hasPrevGroup = true;
            prevMStart = mStart;
            prevMEnd = mEnd;
            prevNStart = nStart;
            prevNEnd = nEnd;
        }
    }
    if (hasPrevGroup) {
        HgemmFlushGroup<BM, BN, GROUP_N>(st, gmCTensor, prevMStart, prevMEnd, prevNStart, prevNEnd);
    }
    HgemmWaitRingFlags();
}

extern "C" __global__ __cube__ void hgemm_kernel_fp16(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, int32_t outputFp32, GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    bool isTransA = (tiling.isTransA != 0);
    bool isTransB = (tiling.isTransB != 0);
    if (outputFp32 != 0) {
        if (isTransA && isTransB) {
            HgemmCubeKernelImpl<
                HGEMM_BASE_M, HGEMM_BASE_K, HGEMM_BASE_N, HGEMM_GROUP_M, HGEMM_GROUP_N, true, true, float>(
                a, b, c, tiling);
        } else if (isTransA && !isTransB) {
            HgemmCubeKernelImpl<
                HGEMM_BASE_M, HGEMM_BASE_K, HGEMM_BASE_N, HGEMM_GROUP_M, HGEMM_GROUP_N, true, false, float>(
                a, b, c, tiling);
        } else if (!isTransA && isTransB) {
            HgemmCubeKernelImpl<
                HGEMM_BASE_M, HGEMM_BASE_K, HGEMM_BASE_N, HGEMM_GROUP_M, HGEMM_GROUP_N, false, true, float>(
                a, b, c, tiling);
        } else {
            HgemmCubeKernelImpl<
                HGEMM_BASE_M, HGEMM_BASE_K, HGEMM_BASE_N, HGEMM_GROUP_M, HGEMM_GROUP_N, false, false, float>(
                a, b, c, tiling);
        }
    } else {
        if (isTransA && isTransB) {
            HgemmCubeKernelImpl<
                HGEMM_BASE_M, HGEMM_BASE_K, HGEMM_BASE_N, HGEMM_GROUP_M, HGEMM_GROUP_N, true, true, half>(
                a, b, c, tiling);
        } else if (isTransA && !isTransB) {
            HgemmCubeKernelImpl<
                HGEMM_BASE_M, HGEMM_BASE_K, HGEMM_BASE_N, HGEMM_GROUP_M, HGEMM_GROUP_N, true, false, half>(
                a, b, c, tiling);
        } else if (!isTransA && isTransB) {
            HgemmCubeKernelImpl<
                HGEMM_BASE_M, HGEMM_BASE_K, HGEMM_BASE_N, HGEMM_GROUP_M, HGEMM_GROUP_N, false, true, half>(
                a, b, c, tiling);
        } else {
            HgemmCubeKernelImpl<
                HGEMM_BASE_M, HGEMM_BASE_K, HGEMM_BASE_N, HGEMM_GROUP_M, HGEMM_GROUP_N, false, false, half>(
                a, b, c, tiling);
        }
    }
}

// ============================================================================
// Alpha/Beta vector kernel (FLOAT16 output)
//   C = alpha * tempAB + beta * C_orig, computed in FLOAT32 and rounded to
//   FLOAT16 (round-to-nearest-even), matching the golden's fp32-then-fp16 path.
//   tempAB holds the cube result as FLOAT32 with row stride tiling.ldc
//   (CeilAlign(m, 16)); column j of C maps to row j of tempAB.
// ============================================================================
namespace hgemm_ab_kernel {
using namespace AscendC;

constexpr int32_t AB_TILE_SIZE = 256;
constexpr int32_t AB_BUF_NUM = 2;

class HgemmAlphaBetaKernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmTilingData tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = tiling.m;
        n_ = tiling.n;
        ldc_ = tiling.ldc;
        cLdc_ = tiling.cLdc;
        alpha_ = tiling.alphaReal;
        beta_ = tiling.betaReal;
        hasBeta_ = tiling.hasBeta;

        tempGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tempAB), static_cast<uint64_t>(ldc_) * n_);
        cOrigGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cOrig), static_cast<uint64_t>(cLdc_) * n_);
        cOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cOut), static_cast<uint64_t>(cLdc_) * n_);

        pipe_->InitBuffer(tempQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(cOrigQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(half));
        pipe_->InitBuffer(outQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(half));
        pipe_->InitBuffer(calcBuf_, AB_TILE_SIZE * 2 * sizeof(float));

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockNum > 1) {
            int32_t colsPerCore = (n_ + blockNum - 1) / blockNum;
            startCol_ = blockIdx * colsPerCore;
            endCol_ = startCol_ + colsPerCore;
            if (endCol_ > n_)
                endCol_ = n_;
            if (startCol_ >= n_) {
                startCol_ = 0;
                endCol_ = 0;
            }
        } else {
            startCol_ = 0;
            endCol_ = n_;
        }
    }

    __aicore__ inline void Process()
    {
        for (int32_t col = startCol_; col < endCol_; col++) {
            int32_t rowOffset = 0;
            while (rowOffset < m_) {
                int32_t count = m_ - rowOffset;
                if (count > AB_TILE_SIZE)
                    count = AB_TILE_SIZE;
                ProcessTile(col, rowOffset, count);
                rowOffset += count;
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t tempBytes = static_cast<uint32_t>(count) * sizeof(float);
        DataCopyExtParams extTemp{1, tempBytes, 0, 0, 0};
        DataCopyPadExtParams<float> padTemp{false, 0, 0, 0};
        uint32_t halfBytes = static_cast<uint32_t>(count) * sizeof(half);
        DataCopyExtParams extHalf{1, halfBytes, 0, 0, 0};
        DataCopyPadExtParams<half> padHalf{false, 0, 0, 0};

        uint64_t tempOffset = static_cast<uint64_t>(col) * ldc_ + rowOffset;
        LocalTensor<float> tempTile = tempQue_.AllocTensor<float>();
        DataCopyPad(tempTile, tempGlobal_[tempOffset], extTemp, padTemp);
        tempQue_.EnQue(tempTile);
        LocalTensor<float> tempLocal = tempQue_.DeQue<float>();

        LocalTensor<float> accF = calcBuf_.Get<float>();
        Muls(accF, tempLocal, alpha_, count);

        if (hasBeta_ != 0) {
            uint64_t cOrigOffset = static_cast<uint64_t>(col) * cLdc_ + rowOffset;
            LocalTensor<half> cOrigTile = cOrigQue_.AllocTensor<half>();
            DataCopyPad(cOrigTile, cOrigGlobal_[cOrigOffset], extHalf, padHalf);
            cOrigQue_.EnQue(cOrigTile);
            LocalTensor<half> cOrigLocal = cOrigQue_.DeQue<half>();

            LocalTensor<float> scaledF = calcBuf_.Get<float>()[AB_TILE_SIZE];
            Cast(scaledF, cOrigLocal, RoundMode::CAST_NONE, count);
            Muls(scaledF, scaledF, beta_, count);
            Add(accF, accF, scaledF, count);
            cOrigQue_.FreeTensor(cOrigLocal);
        }

        uint64_t cOutOffset = static_cast<uint64_t>(col) * cLdc_ + rowOffset;
        LocalTensor<half> outTile = outQue_.AllocTensor<half>();
        Cast(outTile, accF, RoundMode::CAST_RINT, count);
        outQue_.EnQue(outTile);
        LocalTensor<half> outLocal = outQue_.DeQue<half>();
        DataCopyPad(cOutGlobal_[cOutOffset], outLocal, extHalf);
        outQue_.FreeTensor(outLocal);

        tempQue_.FreeTensor(tempLocal);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, AB_BUF_NUM> tempQue_;
    TQue<QuePosition::VECIN, AB_BUF_NUM> cOrigQue_;
    TQue<QuePosition::VECOUT, AB_BUF_NUM> outQue_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    GlobalTensor<float> tempGlobal_;
    GlobalTensor<half> cOrigGlobal_;
    GlobalTensor<half> cOutGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int32_t cLdc_ = 0;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    int32_t hasBeta_ = 0;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
};
} // namespace hgemm_ab_kernel

extern "C" __global__ __aicore__ void hgemm_alpha_beta_kernel(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    hgemm_ab_kernel::HgemmAlphaBetaKernel op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

// ============================================================================
// Scale kernel - C = beta * C (in-place FLOAT16, FLOAT32 math), for the
// alpha == 0 / k == 0 fast path
// ============================================================================
namespace hgemm_scale_kernel {
using namespace AscendC;

constexpr int32_t SCALE_TILE_SIZE = 256;
constexpr int32_t SCALE_BUF_NUM = 2;

class HgemmScaleKernel {
public:
    __aicore__ inline void Init(__gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc, float betaReal, TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = m;
        n_ = n;
        ldc_ = ldc;
        beta_ = betaReal;
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cInOut), static_cast<uint64_t>(ldc) * n);
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        colsPerCore_ = (n_ + blockNum - 1) / blockNum;
        startCol_ = blockIdx * colsPerCore_;
        endCol_ = startCol_ + colsPerCore_;
        if (endCol_ > n_)
            endCol_ = n_;
        if (startCol_ >= n_) {
            startCol_ = 0;
            endCol_ = 0;
        }
    }

    __aicore__ inline void Process()
    {
        if (startCol_ >= endCol_)
            return;
        pipe_->InitBuffer(inQue_, SCALE_BUF_NUM, SCALE_TILE_SIZE * sizeof(half));
        pipe_->InitBuffer(outQue_, SCALE_BUF_NUM, SCALE_TILE_SIZE * sizeof(half));
        pipe_->InitBuffer(calcBuf_, SCALE_TILE_SIZE * sizeof(float));

        for (int32_t col = startCol_; col < endCol_; col++) {
            int32_t rowOffset = 0;
            while (rowOffset < m_) {
                int32_t count = m_ - rowOffset;
                if (count > SCALE_TILE_SIZE)
                    count = SCALE_TILE_SIZE;
                ProcessTile(col, rowOffset, count);
                rowOffset += count;
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t halfBytes = static_cast<uint32_t>(count) * sizeof(half);
        DataCopyExtParams extHalf{1, halfBytes, 0, 0, 0};
        DataCopyPadExtParams<half> padHalf{false, 0, 0, 0};

        uint64_t cOffset = static_cast<uint64_t>(col) * ldc_ + rowOffset;
        LocalTensor<half> outTile = outQue_.AllocTensor<half>();
        if (beta_ == 0.0f) {
            // beta == 0: C is not a valid input per BLAS semantics (may hold
            // NaN/Inf garbage) - write zeros without reading it at all.
            Duplicate(outTile, static_cast<half>(0), count);
            outQue_.EnQue(outTile);
            LocalTensor<half> outLocal = outQue_.DeQue<half>();
            DataCopyPad(cGlobal_[cOffset], outLocal, extHalf);
            outQue_.FreeTensor(outLocal);
            return;
        }
        LocalTensor<half> inTile = inQue_.AllocTensor<half>();
        DataCopyPad(inTile, cGlobal_[cOffset], extHalf, padHalf);
        inQue_.EnQue(inTile);
        LocalTensor<half> inLocal = inQue_.DeQue<half>();

        LocalTensor<float> inF = calcBuf_.Get<float>();
        Cast(inF, inLocal, RoundMode::CAST_NONE, count);
        Muls(inF, inF, beta_, count);
        Cast(outTile, inF, RoundMode::CAST_RINT, count);
        outQue_.EnQue(outTile);
        LocalTensor<half> outLocal = outQue_.DeQue<half>();
        DataCopyPad(cGlobal_[cOffset], outLocal, extHalf);
        outQue_.FreeTensor(outLocal);
        inQue_.FreeTensor(inLocal);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, SCALE_BUF_NUM> inQue_;
    TQue<QuePosition::VECOUT, SCALE_BUF_NUM> outQue_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    GlobalTensor<half> cGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    float beta_ = 1.0f;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
    int32_t colsPerCore_ = 0;
};
} // namespace hgemm_scale_kernel

extern "C" __global__ __aicore__ void hgemm_scale_kernel_fp16(
    __gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc, float betaReal)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    hgemm_scale_kernel::HgemmScaleKernel op;
    op.Init(cInOut, m, n, ldc, betaReal, &pipe);
    op.Process();
}

// ============================================================================
// Kernel launchers
// ============================================================================

void hgemm_kernel_do(
    uint32_t numBlocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, int32_t outputFp32,
    const GemmTilingData& tilingData)
{
    hgemm_kernel_fp16<<<numBlocks, nullptr, stream>>>(a, b, c, outputFp32, tilingData);
}

void hgemm_alpha_beta_do(
    uint32_t numBlocks, void* stream, GM_ADDR tempAB, GM_ADDR cOrig, GM_ADDR cOut, const GemmTilingData& tilingData)
{
    hgemm_alpha_beta_kernel<<<numBlocks, nullptr, stream>>>(tempAB, cOrig, cOut, tilingData);
}

void hgemm_scale_do(uint32_t numBlocks, void* stream, GM_ADDR cInOut, int32_t m, int32_t n, int32_t ldc, float betaReal)
{
    hgemm_scale_kernel_fp16<<<numBlocks, nullptr, stream>>>(cInOut, m, n, ldc, betaReal);
}

#endif
