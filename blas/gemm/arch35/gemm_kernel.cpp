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
#include "gemm_kernel.h"
#define KERNEL_UTILS_LITE
#include "common/arch/hardware.h"
#include "common/helper/kernel_utils.h"
#include "tensor_api/tensor.h"

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
// GemmCubeState
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
    st.mBlockSize = st.mLoopCount;
    st.nBlockSize = st.nLoopCount;
    while (st.mBlockSize * st.nBlockSize > MAX_C0_TILES) {
        if (st.mBlockSize >= st.nBlockSize) {
            st.mBlockSize = (st.mBlockSize + 1) / 2;
        } else {
            st.nBlockSize = (st.nBlockSize + 1) / 2;
        }
    }
    st.mBlockCount = CeilDiv(st.mLoopCount, st.mBlockSize);
    st.nBlockCount = CeilDiv(st.nLoopCount, st.nBlockSize);
}

template <uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline bool InitGemmState(GemmCubeState& st, const GemmTilingData& tiling)
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

// ============================================================================
// GEMM Cube Kernel — tensor_api based
// ============================================================================
template <uint32_t BM, uint32_t BK, uint32_t BN, typename TensorA, typename TensorB>
__aicore__ inline void ProcessKChunk(
    const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor, const GemmTileState& tile, uint64_t l1OffsetA,
    uint64_t l1OffsetB, uint32_t kOffset, uint32_t curKChunk, uint64_t l1BufId, uint64_t& l0PingPong)
{
    TensorCopyGM2L1A<BM, BK>(st, gmATensor, l1OffsetA, tile.mi, kOffset, tile.curM, curKChunk);
    TensorCopyGM2L1B<BK, BN>(st, gmBTensor, l1OffsetB, tile.ni, kOffset, curKChunk, tile.curN);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

    for (uint32_t kInnerOffset = 0; kInnerOffset < curKChunk; kInnerOffset += BK) {
        uint32_t remainingK = curKChunk - kInnerOffset;
        uint32_t curK = remainingK < BK ? remainingK : BK;
        uint64_t l0BufId = l0PingPong & GEMM_BUFFER_MASK;
        uint64_t l0OffsetA = l0BufId * (HardwareInfo<ArchType::ASCEND_V350>::l0ASize / GEMM_BUFFER_COUNT);
        uint64_t l0OffsetB = l0BufId * (HardwareInfo<ArchType::ASCEND_V350>::l0BSize / GEMM_BUFFER_COUNT);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        TensorCopyL1ToL0A<BM, BK>(l1OffsetA, l0OffsetA, tile.curM, curKChunk, kInnerOffset, curK);
        TensorCopyL1ToL0B<BK, BN>(l1OffsetB, l0OffsetB, curKChunk, tile.curN, kInnerOffset, curK);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

        uint32_t globalKOffset = kOffset + kInnerOffset;
        bool isFirstK = (globalKOffset == 0);
        bool isLastK = (globalKOffset + curK == static_cast<uint32_t>(st.tiling.k));
        TensorMmadTile<BM, BN>(
            st, l0OffsetA, l0OffsetB, tile.mi, tile.ni, tile.mStart, tile.nStart, tile.curM, curK, tile.curN, isFirstK,
            isLastK);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <uint32_t BM, uint32_t BK, uint32_t BN, typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void ProcessMNTile(
    const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor, TensorC gmCTensor, const GemmTileState& tile,
    uint64_t l1BufferBytes, uint64_t& l1PingPong, uint64_t& l0PingPong)
{
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    for (uint32_t kOffset = 0; kOffset < static_cast<uint32_t>(st.tiling.k);
         kOffset += static_cast<uint32_t>(st.tiling.tileKChunk)) {
        uint32_t remainingK = static_cast<uint32_t>(st.tiling.k) - kOffset;
        uint32_t curKChunk = remainingK < static_cast<uint32_t>(st.tiling.tileKChunk) ?
                                 remainingK :
                                 static_cast<uint32_t>(st.tiling.tileKChunk);
        uint64_t l1BufId = l1PingPong & GEMM_BUFFER_MASK;
        uint64_t l1OffsetA = l1BufId * l1BufferBytes;
        uint64_t l1OffsetB =
            l1OffsetA + static_cast<uint64_t>(BM) * static_cast<uint64_t>(st.tiling.tileKChunk) * sizeof(float);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        ProcessKChunk<BM, BK, BN>(
            st, gmATensor, gmBTensor, tile, l1OffsetA, l1OffsetB, kOffset, curKChunk, l1BufId, l0PingPong);
        l1PingPong++;
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(GEMM_ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(GEMM_ZERO_FLAG);
    TensorWriteFixpipeBlock<BM, BN>(st, gmCTensor, tile.mi, tile.ni, tile.mStart, tile.nStart);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
}

template <
    uint32_t BM, uint32_t BK, uint32_t BN, uint32_t C0_VAL, bool IsTransA, bool IsTransB, typename TensorA,
    typename TensorB, typename TensorC>
__aicore__ inline void ProcessMNBlock(
    const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor, TensorC gmCTensor, uint64_t l1BufferBytes,
    uint64_t& l1PingPong, uint64_t& l0PingPong, uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd)
{
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            GemmTileState tile{
                mi, ni, mStart, nStart, (mi != st.baseMCount) ? BM : st.tailM, (ni != st.baseNCount) ? BN : st.tailN};
            ProcessMNTile<BM, BK, BN>(st, gmATensor, gmBTensor, gmCTensor, tile, l1BufferBytes, l1PingPong, l0PingPong);
        }
    }
}

template <
    uint32_t BM, uint32_t BK, uint32_t BN, uint32_t C0_VAL, bool IsTransA, bool IsTransB, typename TensorA,
    typename TensorB, typename TensorC>
__aicore__ inline void ProcessAllTiles(const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor, TensorC gmCTensor)
{
    constexpr uint64_t L1_BUFFER_BYTES = HardwareInfo<ArchType::ASCEND_V350>::l1Size / GEMM_BUFFER_COUNT;
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;

    for (uint32_t mb = 0; mb < st.mBlockCount; mb++) {
        for (uint32_t nb = 0; nb < st.nBlockCount; nb++) {
            uint32_t mStart = mb * st.mBlockSize;
            uint32_t mEnd = (mStart + st.mBlockSize < st.mLoopCount) ? mStart + st.mBlockSize : st.mLoopCount;
            uint32_t nStart = nb * st.nBlockSize;
            uint32_t nEnd = (nStart + st.nBlockSize < st.nLoopCount) ? nStart + st.nBlockSize : st.nLoopCount;
            ProcessMNBlock<BM, BK, BN, C0_VAL, IsTransA, IsTransB>(
                st, gmATensor, gmBTensor, gmCTensor, L1_BUFFER_BYTES, l1PingPong, l0PingPong, mStart, mEnd, nStart,
                nEnd);
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
}

template <uint32_t BM, uint32_t BK, uint32_t BN, uint32_t C0_VAL, bool IsTransA, bool IsTransB>
__aicore__ inline void GemmCubeKernelImpl(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, GemmTilingData& tiling)
{
    using T = float;
    GemmCubeState st{};
    if (!InitGemmState<BM, BK, BN, C0_VAL>(st, tiling)) {
        return;
    }

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

    ProcessAllTiles<BM, BK, BN, C0_VAL, IsTransA, IsTransB>(st, gmATensor, gmBTensor, gmCTensor);
}

extern "C" __global__ __cube__ void gemm_kernel_fp32(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    bool isTransA = (tiling.isTransA != 0);
    bool isTransB = (tiling.isTransB != 0);
    if (isTransA && isTransB) {
        GemmCubeKernelImpl<GEMM_BASE_M, GEMM_BASE_K, GEMM_BASE_N, GEMM_C0_SIZE, true, true>(a, b, c, tiling);
    } else if (isTransA && !isTransB) {
        GemmCubeKernelImpl<GEMM_BASE_M, GEMM_BASE_K, GEMM_BASE_N, GEMM_C0_SIZE, true, false>(a, b, c, tiling);
    } else if (!isTransA && isTransB) {
        GemmCubeKernelImpl<GEMM_BASE_M, GEMM_BASE_K, GEMM_BASE_N, GEMM_C0_SIZE, false, true>(a, b, c, tiling);
    } else {
        GemmCubeKernelImpl<GEMM_BASE_M, GEMM_BASE_K, GEMM_BASE_N, GEMM_C0_SIZE, false, false>(a, b, c, tiling);
    }
}

// ============================================================================
// Alpha/Beta Vector kernel (real FP32)
// ============================================================================
namespace ab_kernel {
using namespace AscendC;

constexpr int32_t AB_TILE_SIZE = 256;
constexpr int32_t AB_BUF_NUM = 2;

class AlphaBetaFp32Kernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmTilingData tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = tiling.m;
        n_ = tiling.n;
        ldc_ = tiling.ldc;   // tempAB row stride (= CeilAlign(m, GEMM_FRACTAL))
                             // Note: After ApplyColMajorSwap, tempAB stores C^T (transposed).
                             // col index 0..n-1 maps to original M dimension, row index maps to N.
        cLdc_ = tiling.cLdc; // C matrix original ldc
        alpha_ = tiling.alphaReal;
        beta_ = tiling.betaReal;
        hasBeta_ = tiling.hasBeta;

        tempGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tempAB), static_cast<uint64_t>(ldc_) * n_);
        cOrigGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cOrig), static_cast<uint64_t>(cLdc_) * n_);
        cOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cOut), static_cast<uint64_t>(cLdc_) * n_);

        pipe_->InitBuffer(tempQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(cOrigQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(outQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(calcBuf_, AB_TILE_SIZE * sizeof(float));

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
        uint32_t nbytes = static_cast<uint32_t>(count) * sizeof(float);
        DataCopyExtParams ext{1, nbytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

        // 1. Load tempAB tile (row stride = ldc_ = CeilAlign(m, GEMM_FRACTAL))
        uint64_t tempOffset = static_cast<uint64_t>(col) * ldc_ + rowOffset;
        LocalTensor<float> tempTile = tempQue_.AllocTensor<float>();
        DataCopyPad(tempTile, tempGlobal_[tempOffset], ext, padParams);
        tempQue_.EnQue(tempTile);
        LocalTensor<float> tempLocal = tempQue_.DeQue<float>();

        // 2. result = alpha * tempAB
        LocalTensor<float> alphaResult = outQue_.AllocTensor<float>();
        Muls(alphaResult, tempLocal, alpha_, count);

        // 3. result += beta * C_orig (if hasBeta)
        if (hasBeta_) {
            uint64_t cOrigOffset = static_cast<uint64_t>(col) * cLdc_ + rowOffset;
            LocalTensor<float> cOrigTile = cOrigQue_.AllocTensor<float>();
            DataCopyPad(cOrigTile, cOrigGlobal_[cOrigOffset], ext, padParams);
            cOrigQue_.EnQue(cOrigTile);
            LocalTensor<float> cOrigLocal = cOrigQue_.DeQue<float>();

            LocalTensor<float> scaledC = calcBuf_.Get<float>();
            Muls(scaledC, cOrigLocal, beta_, count);
            Add(alphaResult, alphaResult, scaledC, count);
            cOrigQue_.FreeTensor(cOrigLocal);
        }

        // 4. Write result back to GM (row stride = cLdc_ = original ldc)
        uint64_t cOutOffset = static_cast<uint64_t>(col) * cLdc_ + rowOffset;
        outQue_.EnQue(alphaResult);
        LocalTensor<float> resultTile = outQue_.DeQue<float>();
        DataCopyPad(cOutGlobal_[cOutOffset], resultTile, ext);
        outQue_.FreeTensor(resultTile);

        tempQue_.FreeTensor(tempLocal);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, AB_BUF_NUM> tempQue_;
    TQue<QuePosition::VECIN, AB_BUF_NUM> cOrigQue_;
    TQue<QuePosition::VECOUT, AB_BUF_NUM> outQue_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    GlobalTensor<float> tempGlobal_;
    GlobalTensor<float> cOrigGlobal_;
    GlobalTensor<float> cOutGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;  // tempAB row stride (= CeilAlign(m, 16))
    int32_t cLdc_ = 0; // C matrix original ldc
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    int32_t hasBeta_ = 0;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
};

extern "C" __global__ __aicore__ void gemm_alpha_beta_kernel_fp32(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AlphaBetaFp32Kernel op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

} // namespace ab_kernel

// ============================================================================
// Scale kernel — C = beta * C (in-place), for alpha==0 / k==0 fast path
// ============================================================================
namespace scale_kernel {
using namespace AscendC;

constexpr int32_t SCALE_TILE_SIZE = 256;
constexpr int32_t SCALE_BUF_NUM = 2;

class ScaleFp32Kernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc, float betaReal, float betaImag, int32_t isComplex,
        TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = m;
        n_ = n;
        ldc_ = ldc;
        betaReal_ = betaReal;
        betaImag_ = betaImag;
        isComplex_ = isComplex;
        elemPerCol_ = isComplex ? (m * 2) : m;
        cGlobal_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(cInOut), static_cast<uint64_t>(ldc) * n * (isComplex ? 2 : 1));
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
        pipe_->InitBuffer(inQue_, SCALE_BUF_NUM, SCALE_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(outQue_, SCALE_BUF_NUM, SCALE_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(calcBuf_, SCALE_TILE_SIZE * sizeof(float));

        for (int32_t col = startCol_; col < endCol_; col++) {
            int32_t rowOffset = 0;
            while (rowOffset < elemPerCol_) {
                int32_t count = elemPerCol_ - rowOffset;
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
        uint32_t nbytes = static_cast<uint32_t>(count) * sizeof(float);
        DataCopyExtParams ext{1, nbytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

        uint64_t cOffset = static_cast<uint64_t>(col) * ldc_ * (isComplex_ ? 2 : 1) + rowOffset;
        LocalTensor<float> inTile = inQue_.AllocTensor<float>();
        DataCopyPad(inTile, cGlobal_[cOffset], ext, padParams);
        inQue_.EnQue(inTile);
        LocalTensor<float> inLocal = inQue_.DeQue<float>();

        LocalTensor<float> outTile = outQue_.AllocTensor<float>();
        if (isComplex_ && betaImag_ != 0.0f) {
            // Complex multiply: (br+bi*i)*(cr+ci*i) = (br*cr-bi*ci) + (br*ci+bi*cr)*i
            // count is even (2*m), process real/imag interleaved
            int32_t halfCount = count / 2;
            LocalTensor<float> tmp = calcBuf_.Get<float>();
            // tmp = bi * input
            Muls(tmp, inLocal, betaImag_, count);
            // out[even] = br*in[even] - bi*in[odd] = br*in[even] - tmp[odd]
            // out[odd]  = br*in[odd] + bi*in[even] = br*in[odd] + tmp[even]
            Muls(outTile, inLocal, betaReal_, count);
            for (int32_t i = 0; i < halfCount; i++) {
                int32_t re = i * 2;
                int32_t im = re + 1;
                outTile.SetValue(re, outTile.GetValue(re) - tmp.GetValue(im));
                outTile.SetValue(im, outTile.GetValue(im) + tmp.GetValue(re));
            }
        } else {
            Muls(outTile, inLocal, betaReal_, count);
        }
        outQue_.EnQue(outTile);
        LocalTensor<float> outLocal = outQue_.DeQue<float>();
        DataCopyPad(cGlobal_[cOffset], outLocal, ext);
        outQue_.FreeTensor(outLocal);
        inQue_.FreeTensor(inLocal);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, SCALE_BUF_NUM> inQue_;
    TQue<QuePosition::VECOUT, SCALE_BUF_NUM> outQue_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    GlobalTensor<float> cGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int32_t elemPerCol_ = 0;
    float betaReal_ = 1.0f;
    float betaImag_ = 0.0f;
    int32_t isComplex_ = 0;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
    int32_t colsPerCore_ = 0;
};

} // namespace scale_kernel

extern "C" __global__ __aicore__ void gemm_scale_kernel_fp32(
    __gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc, float betaReal, float betaImag, int32_t isComplex)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    scale_kernel::ScaleFp32Kernel op;
    op.Init(cInOut, m, n, ldc, betaReal, betaImag, isComplex, &pipe);
    op.Process();
}

// ============================================================================
// Cgemm combine kernel — merge 4 FP32 GEMM results into complex C
// C = alpha*(t1-t2) + i*alpha*(t3+t4) + beta*C
// ============================================================================
namespace cgemm_combine {
using namespace AscendC;

constexpr int32_t COMBINE_TILE = 256;
constexpr int32_t COMBINE_BUF_NUM = 2;

class CgemmCombineKernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* t1, __gm__ uint8_t* t2, __gm__ uint8_t* t3, __gm__ uint8_t* t4, int32_t tempLdc,
        __gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc, float ar, float ai, float br, float bi,
        AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = m;
        n_ = n;
        ldc_ = ldc;
        tempLdc_ = tempLdc;
        ar_ = ar;
        ai_ = ai;
        br_ = br;
        bi_ = bi;
        t1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t1), static_cast<uint64_t>(tempLdc) * n);
        t2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t2), static_cast<uint64_t>(tempLdc) * n);
        t3Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t3), static_cast<uint64_t>(tempLdc) * n);
        t4Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t4), static_cast<uint64_t>(tempLdc) * n);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cInOut), static_cast<uint64_t>(ldc) * n * 2);
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
        pipe_->InitBuffer(t1Que_, COMBINE_BUF_NUM, COMBINE_TILE * sizeof(float));
        pipe_->InitBuffer(t2Que_, COMBINE_BUF_NUM, COMBINE_TILE * sizeof(float));
        // ProcessTile 的交织复数 C 路径按 2*count 个 float 读写 cQue_/outQue_（count 最大为
        // COMBINE_TILE），缓冲区必须按两倍 tile 尺寸分配，否则 count>COMBINE_TILE/2（即
        // m>128）时 DataCopyPad/SetValue 越界（issue #458）
        pipe_->InitBuffer(cQue_, COMBINE_BUF_NUM, COMBINE_TILE * 2 * sizeof(float));
        pipe_->InitBuffer(outQue_, COMBINE_BUF_NUM, COMBINE_TILE * 2 * sizeof(float));

        for (int32_t col = startCol_; col < endCol_; col++) {
            int32_t rowOffset = 0;
            while (rowOffset < m_) {
                int32_t count = m_ - rowOffset;
                if (count > COMBINE_TILE)
                    count = COMBINE_TILE;
                ProcessTile(col, rowOffset, count);
                rowOffset += count;
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t nbytes = static_cast<uint32_t>(count) * sizeof(float);
        DataCopyExtParams ext{1, nbytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

        uint64_t tempOffset = static_cast<uint64_t>(col) * tempLdc_ + rowOffset;

        LocalTensor<float> t1Tile = t1Que_.AllocTensor<float>();
        DataCopyPad(t1Tile, t1Global_[tempOffset], ext, padParams);
        t1Que_.EnQue(t1Tile);
        LocalTensor<float> t1Local = t1Que_.DeQue<float>();

        LocalTensor<float> t2Tile = t2Que_.AllocTensor<float>();
        DataCopyPad(t2Tile, t2Global_[tempOffset], ext, padParams);
        t2Que_.EnQue(t2Tile);
        LocalTensor<float> t2Local = t2Que_.DeQue<float>();

        LocalTensor<float> t3Tile = t1Que_.AllocTensor<float>();
        DataCopyPad(t3Tile, t3Global_[tempOffset], ext, padParams);
        t1Que_.EnQue(t3Tile);
        LocalTensor<float> t3Local = t1Que_.DeQue<float>();

        LocalTensor<float> t4Tile = t2Que_.AllocTensor<float>();
        DataCopyPad(t4Tile, t4Global_[tempOffset], ext, padParams);
        t2Que_.EnQue(t4Tile);
        LocalTensor<float> t4Local = t2Que_.DeQue<float>();

        // Load C_orig (interleaved complex: 2*count floats)
        uint32_t cplxBytes = static_cast<uint32_t>(count) * 2 * sizeof(float);
        DataCopyExtParams cplxExt{1, cplxBytes, 0, 0, 0};
        uint64_t cOffset = static_cast<uint64_t>(col) * ldc_ * 2 + rowOffset * 2;
        LocalTensor<float> cTile = cQue_.AllocTensor<float>();
        DataCopyPad(cTile, cGlobal_[cOffset], cplxExt, padParams);
        cQue_.EnQue(cTile);
        LocalTensor<float> cLocal = cQue_.DeQue<float>();

        // Combine: C = alpha*(t1-t2) + i*alpha*(t3+t4) + beta*C
        LocalTensor<float> outTile = outQue_.AllocTensor<float>();
        for (int32_t i = 0; i < count; i++) {
            float ab_r = t1Local.GetValue(i) - t2Local.GetValue(i);
            float ab_i = t3Local.GetValue(i) + t4Local.GetValue(i);
            float c_r = cLocal.GetValue(i * 2);
            float c_i = cLocal.GetValue(i * 2 + 1);
            float out_r = ar_ * ab_r - ai_ * ab_i + br_ * c_r - bi_ * c_i;
            float out_i = ar_ * ab_i + ai_ * ab_r + br_ * c_i + bi_ * c_r;
            outTile.SetValue(i * 2, out_r);
            outTile.SetValue(i * 2 + 1, out_i);
        }

        outQue_.EnQue(outTile);
        LocalTensor<float> outLocal = outQue_.DeQue<float>();
        DataCopyPad(cGlobal_[cOffset], outLocal, cplxExt);
        outQue_.FreeTensor(outLocal);

        t1Que_.FreeTensor(t1Local);
        t1Que_.FreeTensor(t3Local);
        t2Que_.FreeTensor(t2Local);
        t2Que_.FreeTensor(t4Local);
        cQue_.FreeTensor(cLocal);
    }

    AscendC::TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, COMBINE_BUF_NUM> t1Que_;
    TQue<QuePosition::VECIN, COMBINE_BUF_NUM> t2Que_;
    TQue<QuePosition::VECIN, COMBINE_BUF_NUM> cQue_;
    TQue<QuePosition::VECOUT, COMBINE_BUF_NUM> outQue_;
    GlobalTensor<float> t1Global_;
    GlobalTensor<float> t2Global_;
    GlobalTensor<float> t3Global_;
    GlobalTensor<float> t4Global_;
    GlobalTensor<float> cGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int32_t tempLdc_ = 0;
    float ar_ = 1.0f;
    float ai_ = 0.0f;
    float br_ = 0.0f;
    float bi_ = 0.0f;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
    int32_t colsPerCore_ = 0;
};

} // namespace cgemm_combine

extern "C" __global__ __aicore__ void gemm_cgemm_combine_kernel(
    __gm__ uint8_t* t1, __gm__ uint8_t* t2, __gm__ uint8_t* t3, __gm__ uint8_t* t4, int32_t tempLdc,
    __gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc, float ar, float ai, float br, float bi)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    cgemm_combine::CgemmCombineKernel op;
    op.Init(t1, t2, t3, t4, tempLdc, cInOut, m, n, ldc, ar, ai, br, bi, &pipe);
    op.Process();
}

// ============================================================================
// Kernel launchers
// ============================================================================

void gemm_kernel_do(uint32_t numBlocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmTilingData& tilingData)
{
    gemm_kernel_fp32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
}

void gemm_alpha_beta_do(
    uint32_t numBlocks, void* stream, GM_ADDR tempAB, GM_ADDR cOrig, GM_ADDR cOut, const GemmTilingData& tilingData)
{
    ab_kernel::gemm_alpha_beta_kernel_fp32<<<numBlocks, nullptr, stream>>>(tempAB, cOrig, cOut, tilingData);
}

void gemm_scale_do(
    uint32_t numBlocks, void* stream, GM_ADDR cInOut, int32_t m, int32_t n, int32_t ldc, float betaReal, float betaImag,
    int32_t isComplex)
{
    gemm_scale_kernel_fp32<<<numBlocks, nullptr, stream>>>(cInOut, m, n, ldc, betaReal, betaImag, isComplex);
}

void gemm_cgemm_combine_do(
    uint32_t numBlocks, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, int32_t tempLdc, GM_ADDR cInOut,
    int32_t m, int32_t n, int32_t ldc, float ar, float ai, float br, float bi)
{
    gemm_cgemm_combine_kernel<<<numBlocks, nullptr, stream>>>(
        t1, t2, t3, t4, tempLdc, cInOut, m, n, ldc, ar, ai, br, bi);
}

#endif
