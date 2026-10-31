/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_cube_arch35.h
 * \brief Shared tensor_api cube-GEMM infrastructure for the arch35 GEMM family.
 *
 * csymm_cube_kernel.cpp started as a fork of gemm_kernel.cpp and carried a
 * near-identical copy of the cube helpers, which tripped the duplicate-code
 * scan. The parts that are byte-for-byte identical (constants, the per-block
 * state, the M/N partitioning, the L1/L0 data-movement and Mmad helpers) live
 * here so both kernels share one definition.
 *
 * The parts that genuinely diverged are intentionally NOT shared:
 *   * ProcessAllTiles / ProcessMNTile / ProcessKChunk: gemm walks K in
 *     tileKChunk steps, while csymm prefetches and splits K into two segments.
 *   * the FIX_M drain at the end of ProcessAllTiles (csymm releases the last
 *     L1 bank explicitly, because the fused 3-GEMM launch reuses the flag
 *     state across GEMMs).
 */

#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#ifndef KERNEL_UTILS_LITE
#define KERNEL_UTILS_LITE
#endif
#include "common/helper/kernel_utils.h"
#include "common/arch/hardware.h"
#include "gemm/arch35/gemm_tiling_struct.h"

namespace te = AscendC::Te;

constexpr uint32_t GEMM_FP32_C0 = 8;

constexpr uint32_t GEMM_L0C_C0 = 16;
constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;
constexpr uint16_t GEMM_ZERO_FLAG = 0;
constexpr uint16_t GEMM_FIRST_FLAG = 1;
constexpr uint64_t GEMM_BUFFER_COUNT = 2;
constexpr uint64_t GEMM_BUFFER_MASK = GEMM_BUFFER_COUNT - 1;

// ============================================================================
// Per-block GEMM state
// ============================================================================
struct GemmCubeState {
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
    uint32_t mBlockSize;
    uint32_t nBlockSize;
    uint32_t mBlockCount;
    uint32_t nBlockCount;
};

struct GemmTileState {
    uint32_t mi;
    uint32_t ni;
    uint32_t mStart;
    uint32_t nStart;
    uint32_t curM;
    uint32_t curN;
};

__aicore__ inline void ComputeBalancedAxis(
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
    if (blockIdx >= blockCount) {
        // A launch index beyond this axis' block count (possible when a fused kernel
        // splits the launch index differently from mBlocks * nBlocks) must yield an
        // empty slice; letting it through would wrap `remaining` below to a huge value.
        elementOffset = 0;
        actualLength = 0;
        return;
    }
    uint32_t currentTileCount = tilesPerBlock + (blockIdx < extraTileBlocks ? 1U : 0U);
    uint32_t startTile = blockIdx * tilesPerBlock + (blockIdx < extraTileBlocks ? blockIdx : extraTileBlocks);
    elementOffset = static_cast<uint64_t>(startTile) * baseTile;
    uint32_t remaining = length - static_cast<uint32_t>(elementOffset);
    uint32_t assignedLength = currentTileCount * baseTile;
    actualLength = assignedLength < remaining ? assignedLength : remaining;
}

template <uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void Compute2DBlocking(GemmCubeState& st)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    // L0C single-buffer assumption: uses full l0CSize for tile accumulation.
    // If L0C double-buffer or multi-tile accumulation is enabled later, reserve half: l0CSize / (2 * C0_TILE_BYTES).
    constexpr uint32_t MAX_C0_TILES = HardwareInfo<ArchType::ASCEND_V350>::l0CSize / C0_TILE_BYTES;
    // A tile configuration whose bytes exceed L0C yields MAX_C0_TILES == 0 and the
    // halving loop below can never terminate; reject it at compile time. The check
    // uses 64-bit arithmetic so a wrapping C0_TILE_BYTES cannot sneak through.
    static_assert(static_cast<uint64_t>(BASE_M) * BASE_N * sizeof(float) <=
            HardwareInfo<ArchType::ASCEND_V350>::l0CSize,
        "Compute2DBlocking: BASE_M*BASE_N tile bytes exceed L0C size, MAX_C0_TILES would be 0");
    st.mBlockSize = st.mLoopCount;
    st.nBlockSize = st.nLoopCount;
    // 64-bit product: loop-count squares can wrap uint32_t.
    while (static_cast<uint64_t>(st.mBlockSize) * st.nBlockSize > MAX_C0_TILES) {
        if (st.mBlockSize >= st.nBlockSize) {
            st.mBlockSize = (st.mBlockSize + 1) / 2;
        } else {
            st.nBlockSize = (st.nBlockSize + 1) / 2;
        }
    }
    st.mBlockCount = CeilDiv(st.mLoopCount, st.mBlockSize);
    st.nBlockCount = CeilDiv(st.nLoopCount, st.nBlockSize);
}

// blockIdx is passed in so the same per-block logic can be reused by the fused
// 3-GEMM csymm kernel, where the launch block index is split into
// (gemmId, in-gemm block); gemm passes AscendC::GetBlockIdx().
template <uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline bool InitGemmState(GemmCubeState& st, const GemmTilingData& tiling, uint32_t blockIdx)
{
    st.tiling = tiling;
    if (blockIdx >= static_cast<uint32_t>(st.tiling.usedCoreNum)) {
        return false;
    }
    if (st.tiling.mBlocks == 0 || st.tiling.nBlocks == 0) {
        return false;
    }
    st.mBlockIdx = blockIdx % static_cast<uint32_t>(st.tiling.mBlocks);
    st.nBlockIdx = blockIdx / static_cast<uint32_t>(st.tiling.mBlocks);
    ComputeBalancedAxis(
        static_cast<uint32_t>(st.tiling.m), BASE_M, static_cast<uint32_t>(st.tiling.mBlocks), st.mBlockIdx, st.mOffset,
        st.actualM);
    ComputeBalancedAxis(
        static_cast<uint32_t>(st.tiling.n), BASE_N, static_cast<uint32_t>(st.tiling.nBlocks), st.nBlockIdx, st.nOffset,
        st.actualN);
    if (st.actualM == 0 || st.actualN == 0) {
        return false;
    }
    st.baseMCount = st.actualM / BASE_M;
    st.tailM = st.actualM % BASE_M;
    st.baseNCount = st.actualN / BASE_N;
    st.tailN = st.actualN % BASE_N;
    st.mLoopCount = CeilDiv(st.actualM, BASE_M);
    st.nLoopCount = CeilDiv(st.actualN, BASE_N);
    Compute2DBlocking<BASE_M, BASE_N>(st);
    return true;
}

template <uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline GemmTileState MakeGemmTileState(
    const GemmCubeState& st, uint32_t mi, uint32_t ni, uint32_t mStart, uint32_t nStart)
{
    GemmTileState tile{
        mi, ni, mStart, nStart, (mi != st.baseMCount) ? BASE_M : st.tailM, (ni != st.baseNCount) ? BASE_N : st.tailN};
    return tile;
}

__aicore__ inline void MakeGemmBlockRange(
    const GemmCubeState& st, uint32_t mb, uint32_t nb, uint32_t& mStart, uint32_t& mEnd, uint32_t& nStart,
    uint32_t& nEnd)
{
    mStart = mb * st.mBlockSize;
    mEnd = (mStart + st.mBlockSize < st.mLoopCount) ? mStart + st.mBlockSize : st.mLoopCount;
    nStart = nb * st.nBlockSize;
    nEnd = (nStart + st.nBlockSize < st.nLoopCount) ? nStart + st.nBlockSize : st.nLoopCount;
}

__aicore__ inline void InitGemmPipelineFlags()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
}

// ============================================================================
// GM operand tensors
// ============================================================================
template <typename TensorA, typename TensorB, typename TensorC>
struct GemmOperandTensors {
    TensorA a;
    TensorB b;
    TensorC c;
};

template <bool IsTransA, bool IsTransB>
__aicore__ inline auto MakeGemmOperandTensors(
    const GemmCubeState& st, __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c)
{
    using T = float;
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
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ T*>(c)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(st.tiling.m), static_cast<uint64_t>(st.tiling.ldc)));

    return GemmOperandTensors<decltype(gmATensor), decltype(gmBTensor), decltype(gmCTensor)>{
        gmATensor, gmBTensor, gmCTensor};
}

// ============================================================================
// tensor_api based data movement and computation
// ============================================================================
template <uint32_t BASE_M, uint32_t BASE_K, typename GM_TENSOR_A>
__aicore__ inline void TensorCopyGM2L1A(
    const GemmCubeState& st, GM_TENSOR_A& gmATensor, uint64_t l1OffsetBytes, uint32_t mi, uint32_t kOffset,
    uint32_t curM, uint32_t curKChunk)
{
    using T = float;
    uint64_t mPos = st.mOffset + static_cast<uint64_t>(mi) * BASE_M;
    auto gmBlock = gmATensor.Slice(te::MakeCoord(mPos, static_cast<uint64_t>(kOffset)), te::MakeShape(curM, curKChunk));
    auto tensorAL1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1OffsetBytes),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<GEMM_FP32_C0>>(curM, curKChunk));
    te::Copy(te::MakeCopy(te::CopyGM2L1{}), tensorAL1, gmBlock);
}

template <uint32_t BASE_M, uint32_t BASE_K>
__aicore__ inline void TensorCopyL1ToL0A(
    uint64_t l1OffsetBytes, uint64_t l0OffsetBytes, uint32_t curM, uint32_t curKChunk, uint32_t kInnerOffset,
    uint32_t curK)
{
    using T = float;
    auto tensorAL1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1OffsetBytes),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<GEMM_FP32_C0>>(curM, curKChunk));
    auto tensorAL1Tile =
        tensorAL1.Slice(te::MakeCoord(0L, static_cast<uint64_t>(kInnerOffset)), te::MakeShape(curM, curK));
    auto tensorAL0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0A, T>(l0OffsetBytes),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<GEMM_FP32_C0>>(curM, curK));
    te::Copy(te::MakeCopy(te::CopyL12L0A{}), tensorAL0, tensorAL1Tile);
}

template <uint32_t BASE_K, uint32_t BASE_N, typename GM_TENSOR_B>
__aicore__ inline void TensorCopyGM2L1B(
    const GemmCubeState& st, GM_TENSOR_B& gmBTensor, uint64_t l1OffsetBytes, uint32_t ni, uint32_t kOffset,
    uint32_t curKChunk, uint32_t curN)
{
    using T = float;
    uint64_t nPos = st.nOffset + static_cast<uint64_t>(ni) * BASE_N;
    auto gmBlock = gmBTensor.Slice(te::MakeCoord(static_cast<uint64_t>(kOffset), nPos), te::MakeShape(curKChunk, curN));
    auto tensorBL1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1OffsetBytes),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<GEMM_FP32_C0>>(curKChunk, curN));
    te::Copy(te::MakeCopy(te::CopyGM2L1{}), tensorBL1, gmBlock);
}

template <uint32_t BASE_K, uint32_t BASE_N>
__aicore__ inline void TensorCopyL1ToL0B(
    uint64_t l1OffsetBytes, uint64_t l0OffsetBytes, uint32_t curKChunk, uint32_t curN, uint32_t kInnerOffset,
    uint32_t curK)
{
    using T = float;
    auto tensorBL1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1OffsetBytes),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<GEMM_FP32_C0>>(curKChunk, curN));
    auto tensorBL1Tile =
        tensorBL1.Slice(te::MakeCoord(static_cast<uint64_t>(kInnerOffset), 0L), te::MakeShape(curK, curN));
    auto tensorBL0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0B, T>(l0OffsetBytes),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<GEMM_FP32_C0>>(curK, curN));
    te::Copy(te::MakeCopy(te::CopyL12L0B{}), tensorBL0, tensorBL1Tile);
}

template <uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void TensorMmadTile(
    const GemmCubeState& st, uint64_t l0OffsetA, uint64_t l0OffsetB, uint32_t mi, uint32_t ni, uint32_t mStart,
    uint32_t nStart, uint32_t curM, uint32_t curK, uint32_t curN, bool isFirstK, bool isLastK)
{
    using T = float;
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
    uint64_t l0cOffset = static_cast<uint64_t>(cTileIdx) * C0_TILE_BYTES;

    auto tensorAL0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0A, T>(l0OffsetA),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<GEMM_FP32_C0>>(curM, curK));
    auto tensorBL0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0B, T>(l0OffsetB),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<GEMM_FP32_C0>>(curK, curN));
    auto tensorL0C = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, float>(l0cOffset),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<GEMM_L0C_C0>>(curM, curN));

    uint8_t unitFlag = isLastK ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
    te::MmadParams mmadParams{
        static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(curK), unitFlag, isFirstK};
    te::Mmad(te::MmadAtom<te::MmadTraits<te::MmadOperation>>{}.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
}

template <uint32_t BASE_M, uint32_t BASE_N, typename GM_TENSOR_C>
__aicore__ inline void TensorWriteFixpipeBlock(
    const GemmCubeState& st, GM_TENSOR_C& gmCTensor, uint32_t mi, uint32_t ni, uint32_t mStart, uint32_t nStart)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
    uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
    uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
    uint64_t l0cOffset = static_cast<uint64_t>(cTileIdx) * C0_TILE_BYTES;
    uint64_t mOff = st.mOffset + static_cast<uint64_t>(mi) * BASE_M;
    uint64_t nOff = st.nOffset + static_cast<uint64_t>(ni) * BASE_N;
    auto gmBlockC = gmCTensor.Slice(te::MakeCoord(mOff, nOff), te::MakeShape(curM, curN));
    auto tensorL0C = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, float>(l0cOffset),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<GEMM_L0C_C0>>(curM, curN));
    te::MakeCopy(te::CopyL0C2GM{}).Call(gmBlockC, tensorL0C, te::FixpipeParams{FINAL_ACCUMULATION});
}

// One BK-sized L1->L0A/L0B stage of a K chunk plus its Mmad: both kernels run
// the identical L0 ping-pong pipeline, they only disagree above this level on
// how the K axis is chopped (gemm: tileKChunk; csymm: two prefetched segments).
template <uint32_t BM, uint32_t BK, uint32_t BN>
__aicore__ inline void RunL1ToL0Mmad(
    const GemmCubeState& st, const GemmTileState& tile, uint64_t l1OffsetA, uint64_t l1OffsetB, uint64_t l0BufId,
    uint32_t kInnerOffset, uint32_t curKChunk, uint32_t curK, bool isFirstK, bool isLastK)
{
    uint64_t l0OffsetA = l0BufId * (HardwareInfo<ArchType::ASCEND_V350>::l0ASize / GEMM_BUFFER_COUNT);
    uint64_t l0OffsetB = l0BufId * (HardwareInfo<ArchType::ASCEND_V350>::l0BSize / GEMM_BUFFER_COUNT);

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
    TensorCopyL1ToL0A<BM, BK>(l1OffsetA, l0OffsetA, tile.curM, curKChunk, kInnerOffset, curK);
    TensorCopyL1ToL0B<BK, BN>(l1OffsetB, l0OffsetB, curKChunk, tile.curN, kInnerOffset, curK);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

    TensorMmadTile<BM, BN>(
        st, l0OffsetA, l0OffsetB, tile.mi, tile.ni, tile.mStart, tile.nStart, tile.curM, curK, tile.curN, isFirstK,
        isLastK);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
}
