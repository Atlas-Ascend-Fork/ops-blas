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
#include "kernel_operator.h"
#include "common/helper/gemm_strided_tiling_arch35.h"
#include "common/arch/hardware.h"
#include "common/helper/kernel_utils.h"

// ============================================================================
// GemmStridedCubeState — POD runtime state for strided cube kernel
// ============================================================================
struct GemmStridedCubeState {
    GemmStridedBatchedExTilingData tiling;
    uint32_t actualM;
    uint32_t actualN;
    uint32_t baseMCount;
    uint32_t mnTasks;
    uint32_t mBlockIdx;
    uint32_t nBlockIdx;
    uint32_t tailM;
    uint32_t tailMAlign;
    uint32_t baseNCount;
    uint32_t tailN;
    uint32_t tailNAlign;
    uint32_t mLoopCount;
    uint32_t nLoopCount;
    uint32_t kLoopCount;
    uint32_t batchIdx;
    uint32_t splitIdx;
    uint32_t kStart;
    uint32_t kLength;
    uint64_t aBaseOffset;
    uint64_t bBaseOffset;
    uint64_t cBaseOffset;
    uint32_t mBlockSize;
    uint32_t nBlockSize;
    uint32_t mBlockCount;
    uint32_t nBlockCount;
};

// ============================================================================
// Standalone __aicore__ helper functions
// ============================================================================

__aicore__ inline bool InitStridedTilingBase(GemmStridedCubeState& st, GemmStridedBatchedExTilingData tiling)
{
    st.tiling = tiling;
    if (st.tiling.mBlocks <= 0 || st.tiling.nBlocks <= 0) {
        return false;
    }
    const uint64_t mnTasks = static_cast<uint64_t>(st.tiling.mBlocks) * static_cast<uint64_t>(st.tiling.nBlocks);
    if (mnTasks > UINT32_MAX) {
        return false;
    }
    st.mnTasks = static_cast<uint32_t>(mnTasks);
    return true;
}

__aicore__ inline void ComputeStridedBaseOffsets(GemmStridedCubeState& st)
{
    if (!st.tiling.isTransB) {
        st.bBaseOffset = static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN;
    } else {
        st.bBaseOffset = static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN * st.tiling.ldb;
    }
    if (!st.tiling.isTransA) {
        st.aBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.singleCoreM * st.tiling.lda;
    } else {
        st.aBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.singleCoreM;
    }
    st.cBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.ldc * st.tiling.singleCoreM +
                     static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN;
}

template <uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void ComputeStrided2DBlocking(GemmStridedCubeState& st)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    constexpr uint32_t MAX_C0_TILES = HardwareInfo<ArchType::ASCEND_V350>::l0CSize / C0_TILE_BYTES;
    static_assert(MAX_C0_TILES > 0, "L0C is too small for one output tile");
    st.mBlockSize = st.mLoopCount;
    st.nBlockSize = st.nLoopCount;
    while (static_cast<uint64_t>(st.mBlockSize) * st.nBlockSize > MAX_C0_TILES) {
        if (st.mBlockSize < st.nBlockSize) {
            st.nBlockSize = (st.nBlockSize + 1) / 2;
        } else {
            st.mBlockSize = (st.mBlockSize + 1) / 2;
        }
    }
    // R8: mBlockSize/nBlockSize >= 1 guaranteed (mLoopCount/nLoopCount >= 1 when actualM/N > 0)
    if (st.nBlockSize == 0) {
        st.nBlockSize = 1;
    }
    if (st.mBlockSize == 0) {
        st.mBlockSize = 1;
    }
    st.mBlockCount = CeilDiv(st.mLoopCount, st.mBlockSize);
    st.nBlockCount = CeilDiv(st.nLoopCount, st.nBlockSize);
}

__aicore__ inline void DecodeStridedTask(GemmStridedCubeState& st, uint64_t taskId, uint32_t& mnTask)
{
    const uint32_t batchCount = static_cast<uint32_t>(st.tiling.batchCount);
    const uint32_t splitK = static_cast<uint32_t>(st.tiling.splitK);
    const uint64_t batchSplit = static_cast<uint64_t>(batchCount) * splitK;
    uint64_t batchSplitTask = 0;
    switch (st.tiling.selectedAlgo) {
        case GEMM_STRIDED_KERNEL_BATCH_FIRST:
        case GEMM_STRIDED_KERNEL_PERSISTENT:
            batchSplitTask = taskId % batchSplit;
            mnTask = static_cast<uint32_t>(taskId / batchSplit);
            break;
        default:
            mnTask = static_cast<uint32_t>(taskId % st.mnTasks);
            batchSplitTask = taskId / st.mnTasks;
            break;
    }
    st.batchIdx = static_cast<uint32_t>(batchSplitTask / splitK);
    st.splitIdx = static_cast<uint32_t>(batchSplitTask % splitK);

    if (st.tiling.selectedAlgo == GEMM_STRIDED_KERNEL_M_MAJOR) {
        st.mBlockIdx = mnTask / static_cast<uint32_t>(st.tiling.nBlocks);
        st.nBlockIdx = mnTask % static_cast<uint32_t>(st.tiling.nBlocks);
    } else {
        st.mBlockIdx = mnTask % static_cast<uint32_t>(st.tiling.mBlocks);
        st.nBlockIdx = mnTask / static_cast<uint32_t>(st.tiling.mBlocks);
    }
}

template <uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void InitStridedTaskState(GemmStridedCubeState& st, uint64_t taskId)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint32_t mnTask = 0;
    DecodeStridedTask(st, taskId, mnTask);
    st.actualM = static_cast<uint32_t>(st.tiling.m) - st.mBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreM);
    if (st.actualM > static_cast<uint32_t>(st.tiling.singleCoreM)) {
        st.actualM = static_cast<uint32_t>(st.tiling.singleCoreM);
    }
    st.actualN = static_cast<uint32_t>(st.tiling.n) - st.nBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreN);
    if (st.actualN > static_cast<uint32_t>(st.tiling.singleCoreN)) {
        st.actualN = static_cast<uint32_t>(st.tiling.singleCoreN);
    }
    st.mLoopCount = CeilDiv(st.actualM, BASE_M);
    st.nLoopCount = CeilDiv(st.actualN, BASE_N);
    const uint32_t fullK = static_cast<uint32_t>(st.tiling.k);
    st.kStart =
        static_cast<uint32_t>((static_cast<uint64_t>(fullK) * st.splitIdx) / static_cast<uint32_t>(st.tiling.splitK));
    const uint32_t kEnd = static_cast<uint32_t>(
        (static_cast<uint64_t>(fullK) * (st.splitIdx + 1U)) / static_cast<uint32_t>(st.tiling.splitK));
    st.kLength = kEnd - st.kStart;
    st.kLoopCount = CeilDiv(st.kLength, BASE_K);
    st.baseMCount = st.actualM / BASE_M;
    st.tailM = st.actualM % BASE_M;
    st.tailMAlign = RoundUp(st.tailM, CUBE_BLOCK);
    st.baseNCount = st.actualN / BASE_N;
    st.tailN = st.actualN % BASE_N;
    st.tailNAlign = RoundUp(st.tailN, CUBE_BLOCK);
    ComputeStridedBaseOffsets(st);
    ComputeStrided2DBlocking<BASE_M, BASE_N>(st);
}

template <typename A_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t C0_VAL>
__aicore__ inline void LoadAStridedTile(
    const GemmStridedCubeState& st, AscendC::GlobalTensor<A_TYPE>& aGM, AscendC::LocalTensor<A_TYPE>& a1,
    AscendC::LocalTensor<A_TYPE>& a2, uint32_t mi, uint32_t kIdx, uint32_t curM, uint32_t curK, uint32_t curMAlign)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint64_t aOffset;
    AscendC::Nd2NzParams ndA;
    ndA.ndNum = 1;
    ndA.dstNzNStride = 1;
    ndA.dstNzMatrixStride = 0;
    if (!st.tiling.isTransA) {
        aOffset = st.aBaseOffset + static_cast<uint64_t>(mi) * BASE_M * st.tiling.lda +
                  static_cast<uint64_t>(st.kStart + kIdx * BASE_K);
        ndA.nValue = curM;
        ndA.dValue = curK;
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(curM, CUBE_BLOCK);
    } else {
        aOffset = st.aBaseOffset + static_cast<uint64_t>(st.kStart + kIdx * BASE_K) * st.tiling.lda +
                  static_cast<uint64_t>(mi) * BASE_M;
        ndA.nValue = curK;
        ndA.dValue = curM;
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(BASE_K, CUBE_BLOCK);
    }
    AscendC::DataCopy(a1, aGM[aOffset], ndA);
    // GM -> L1 must finish before MTE1 reads this tile.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::LoadData2DParamsV2 ldA;
    ldA.mStartPosition = 0;
    ldA.kStartPosition = 0;
    ldA.sid = 0;
    if (st.tiling.isTransA) {
        ldA.mStep = CeilDiv(BASE_K, CUBE_BLOCK);
        ldA.kStep = RoundUp(CeilDiv(curMAlign, C0_VAL), 2u);
        ldA.srcStride = CeilDiv(BASE_K, CUBE_BLOCK);
        ldA.dstStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.ifTranspose = true;
    } else {
        ldA.mStep = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.kStep = CeilDiv(BASE_K, C0_VAL);
        ldA.srcStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.dstStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.ifTranspose = false;
    }
    AscendC::LoadData(a2, a1, ldA);
    // Protect both L0 consumption and the next refill of this L1 tile.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
}

template <typename B_TYPE, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void LoadBStridedTile(
    const GemmStridedCubeState& st, AscendC::GlobalTensor<B_TYPE>& bGM, AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<B_TYPE>& b2, uint32_t ni, uint32_t kIdx, uint32_t curK, uint32_t curN, uint32_t curNAlign)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint64_t bOffset;
    AscendC::Nd2NzParams ndB;
    ndB.ndNum = 1;
    ndB.dstNzNStride = 1;
    ndB.dstNzMatrixStride = 0;
    if (!st.tiling.isTransB) {
        bOffset = st.bBaseOffset + static_cast<uint64_t>(st.kStart + kIdx * BASE_K) * st.tiling.ldb +
                  static_cast<uint64_t>(ni) * BASE_N;
        ndB.nValue = curK;
        ndB.dValue = curN;
        ndB.srcNdMatrixStride = 0;
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
        ndB.dstNzC0Stride = RoundUp(BASE_K, CUBE_BLOCK);
    } else {
        bOffset = st.bBaseOffset + static_cast<uint64_t>(ni) * BASE_N * st.tiling.ldb +
                  static_cast<uint64_t>(st.kStart + kIdx * BASE_K);
        ndB.nValue = curN;
        ndB.dValue = curK;
        ndB.srcNdMatrixStride = 0;
        ndB.dstNzC0Stride = RoundUp(curN, CUBE_BLOCK);
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
    }
    AscendC::DataCopy(b1, bGM[bOffset], ndB);
    // GM -> L1 must finish before MTE1 reads this tile.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::LoadData2DParamsV2 ldB;
    ldB.mStartPosition = 0;
    ldB.kStartPosition = 0;
    ldB.sid = 0;
    if (st.tiling.isTransB) {
        ldB.mStep = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.kStep = CeilDiv(BASE_K, C0_VAL);
        ldB.srcStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.dstStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.ifTranspose = false;
    } else {
        ldB.mStep = CeilDiv(BASE_K, CUBE_BLOCK);
        ldB.kStep = CeilDiv(static_cast<uint32_t>(curNAlign * sizeof(B_TYPE)), 32u);
        ldB.srcStride = CeilDiv(BASE_K, CUBE_BLOCK);
        ldB.dstStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.ifTranspose = true;
    }
    AscendC::LoadData(b2, b1, ldB);
    // Protect both L0 consumption and the next refill of this L1 tile.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
}

template <
    typename A_TYPE, typename B_TYPE, typename ACC_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N,
    uint32_t C0_VAL>
__aicore__ inline void ProcessStridedNTile(
    const GemmStridedCubeState& st, AscendC::GlobalTensor<B_TYPE>& bGM, AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b1, AscendC::LocalTensor<B_TYPE>& b2, uint32_t mi, uint32_t ni, uint32_t kIdx,
    uint32_t mStart, uint32_t nStart, uint32_t curM, uint32_t curK)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    if (ni != nStart) {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    }
    uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
    uint32_t curNAlign = (ni != st.baseNCount) ? BASE_N : st.tailNAlign;
    LoadBStridedTile<B_TYPE, BASE_K, BASE_N, C0_VAL>(st, bGM, b1, b2, ni, kIdx, curK, curN, curNAlign);
    uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
    AscendC::LocalTensor<ACC_TYPE> c0(AscendC::TPosition::CO1, cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
    AscendC::MmadParams mp{};
    mp.m = curM;
    mp.n = curN;
    mp.k = curK;
    mp.cmatrixInitVal = (kIdx == 0);
    AscendC::Mmad(c0, a2, b2, mp);
    // A2/B2 are reused by the next MTE1 load, including across M/N/K tiles.
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
}

template <
    typename C_GM_TYPE, typename ACC_TYPE, uint32_t BASE_M, uint32_t BASE_N, QuantMode_t QUANT_MODE,
    bool WAIT_MTE2 = true>
__aicore__ inline void WriteStridedFixpipe(
    const GemmStridedCubeState& st, AscendC::GlobalTensor<C_GM_TYPE>& cGM, uint32_t mStart, uint32_t mEnd,
    uint32_t nStart, uint32_t nEnd, uint64_t cBaseBytes = 0)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
        uint32_t curMAlign = (mi != st.baseMCount) ? BASE_M : st.tailMAlign;
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
            uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
            AscendC::LocalTensor<ACC_TYPE> c0(
                AscendC::TPosition::CO1, cBaseBytes + cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
            uint64_t cOffset = st.cBaseOffset + static_cast<uint64_t>(mi) * BASE_M * st.tiling.ldc +
                               static_cast<uint64_t>(ni) * BASE_N;
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.mSize = curM;
            fp.nSize = curN;
            fp.srcStride = curMAlign;
            fp.dstStride = static_cast<uint32_t>(st.tiling.ldc);
            fp.quantPre = QUANT_MODE;
            AscendC::Fixpipe(cGM[cOffset], c0, fp);
            if constexpr (WAIT_MTE2) {
                // Do not reuse L0C for the next Mmad until Fixpipe has read it.
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
            }
        }
    }
}

// ============================================================================
// RunKSlice — Process one K slice across M×N tiles (depth 2: mi→ni)
// ============================================================================
template <
    typename A_TYPE, typename B_TYPE, typename ACC_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N,
    uint32_t C0_VAL>
__aicore__ inline void RunKSlice(
    GemmStridedCubeState& st, AscendC::GlobalTensor<A_TYPE>& aGM, AscendC::GlobalTensor<B_TYPE>& bGM,
    AscendC::LocalTensor<A_TYPE>& a1, AscendC::LocalTensor<B_TYPE>& b1, AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b2, uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd, uint32_t kIdx,
    uint32_t curK, bool& firstMmad)
{
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        if (!firstMmad) {
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        }
        firstMmad = false;
        uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
        uint32_t curMAlign = (mi != st.baseMCount) ? BASE_M : st.tailMAlign;
        LoadAStridedTile<A_TYPE, BASE_M, BASE_K, C0_VAL>(st, aGM, a1, a2, mi, kIdx, curM, curK, curMAlign);
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            ProcessStridedNTile<A_TYPE, B_TYPE, ACC_TYPE, BASE_M, BASE_K, BASE_N, C0_VAL>(
                st, bGM, a2, b1, b2, mi, ni, kIdx, mStart, nStart, curM, curK);
        }
    }
}

// ============================================================================
// RunStridedTask — Inner computation for one task (depth 3: mb→nb→kIdx)
// ============================================================================
template <
    typename A_TYPE, typename B_TYPE, typename C_GM_TYPE, typename ACC_TYPE, uint32_t BASE_M, uint32_t BASE_K,
    uint32_t BASE_N, uint32_t C0_VAL, QuantMode_t QMODE>
__aicore__ inline void RunStridedTask(
    GemmStridedCubeState& st, AscendC::GlobalTensor<A_TYPE>& aGM, AscendC::GlobalTensor<B_TYPE>& bGM,
    AscendC::GlobalTensor<C_GM_TYPE>& cGM, AscendC::LocalTensor<A_TYPE>& a1, AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<A_TYPE>& a1Alt, AscendC::LocalTensor<B_TYPE>& b1Alt, AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b2)
{
    bool firstMmad = true;
    for (uint32_t mb = 0; mb < st.mBlockCount; mb++) {
        for (uint32_t nb = 0; nb < st.nBlockCount; nb++) {
            uint32_t mStart = mb * st.mBlockSize;
            uint32_t mEnd = (mStart + st.mBlockSize < st.mLoopCount) ? mStart + st.mBlockSize : st.mLoopCount;
            uint32_t nStart = nb * st.nBlockSize;
            uint32_t nEnd = (nStart + st.nBlockSize < st.nLoopCount) ? nStart + st.nBlockSize : st.nLoopCount;
            for (uint32_t kIdx = 0; kIdx < st.kLoopCount; kIdx++) {
                uint32_t curK = (kIdx == st.kLoopCount - 1) ? (st.kLength - kIdx * BASE_K) : BASE_K;
                if (st.tiling.selectedAlgo == GEMM_STRIDED_KERNEL_DEEP_K && (kIdx & 1U) != 0) {
                    RunKSlice<A_TYPE, B_TYPE, ACC_TYPE, BASE_M, BASE_K, BASE_N, C0_VAL>(
                        st, aGM, bGM, a1Alt, b1Alt, a2, b2, mStart, mEnd, nStart, nEnd, kIdx, curK, firstMmad);
                } else {
                    RunKSlice<A_TYPE, B_TYPE, ACC_TYPE, BASE_M, BASE_K, BASE_N, C0_VAL>(
                        st, aGM, bGM, a1, b1, a2, b2, mStart, mEnd, nStart, nEnd, kIdx, curK, firstMmad);
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            WriteStridedFixpipe<C_GM_TYPE, ACC_TYPE, BASE_M, BASE_N, QMODE>(st, cGM, mStart, mEnd, nStart, nEnd);
            firstMmad = true;
        }
    }
}

// ============================================================================
// GEMM_STRIDED_CUBE_KERNEL — Macro generating all strided cube kernel variants
//
// Supports separate A_TYPE and B_TYPE for FP8 mixed-type kernels.
// Task loop iterates over batch × mBlocks × nTasks with grid-stride.
//
// Parameters:
//   FUNC_NAME  - kernel function name
//   A_TYPE     - input A element type
//   B_TYPE     - input B element type
//   C_GM_TYPE  - output GM element type
//   ACC_TYPE   - MMAD accumulator type (float or int32_t)
//   BM, BK, BN - base tile dimensions
//   C0_VAL     - C0 parameter for LoadData
//   QUANT_MODE - Fixpipe quantization mode
// ============================================================================
#define GEMM_STRIDED_CUBE_KERNEL(FUNC_NAME, A_TYPE, B_TYPE, C_GM_TYPE, ACC_TYPE, BM, BK, BN, C0_VAL, QUANT_MODE) \
    __cube__ __global__ void FUNC_NAME(                                                                          \
        __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling)          \
    {                                                                                                            \
        AscendC::InitSocState();                                                                                 \
        GemmStridedCubeState st{};                                                                               \
        if (!InitStridedTilingBase(st, tiling)) {                                                                \
            return;                                                                                              \
        }                                                                                                        \
        constexpr uint32_t L1_SIZE = HardwareInfo<ArchType::ASCEND_V350>::l1Size;                                \
        constexpr uint32_t L1_HALF = L1_SIZE / 2;                                                                \
        /* DAV-3510 uses a 512-byte L1 offset to avoid A/B ping-pong bank conflicts. */                          \
        constexpr uint32_t L1_BANK_SKEW = 512;                                                                   \
        static_assert(L1_HALF > L1_BANK_SKEW, "L1 buffer must leave room for the bank skew");                    \
        constexpr uint32_t L1_SKEWED_BUFFER = L1_HALF - L1_BANK_SKEW;                                            \
        AscendC::LocalTensor<A_TYPE> a1(AscendC::TPosition::A1, 0, L1_SKEWED_BUFFER);                            \
        AscendC::LocalTensor<A_TYPE> a1Alt(AscendC::TPosition::A1, L1_HALF, L1_SKEWED_BUFFER);                   \
        AscendC::LocalTensor<B_TYPE> b1(AscendC::TPosition::B1, L1_SIZE + L1_BANK_SKEW, L1_SKEWED_BUFFER);       \
        AscendC::LocalTensor<B_TYPE> b1Alt(                                                                      \
            AscendC::TPosition::B1, L1_SIZE + L1_HALF + L1_BANK_SKEW, L1_SKEWED_BUFFER);                         \
        AscendC::LocalTensor<A_TYPE> a2(AscendC::TPosition::A2, 0, (BM) * (BK));                                 \
        AscendC::LocalTensor<B_TYPE> b2(AscendC::TPosition::B2, 0, (BK) * (BN));                                 \
        const uint64_t blockIdxVal = AscendC::GetBlockIdx();                                                     \
        const uint64_t gridDim = AscendC::GetBlockNum();                                                         \
        for (uint64_t taskId = blockIdxVal; taskId < st.tiling.totalTasks; taskId += gridDim) {                  \
            InitStridedTaskState<(BM), (BK), (BN), (C0_VAL)>(st, taskId);                                        \
            AscendC::GlobalTensor<A_TYPE> aGM;                                                                   \
            AscendC::GlobalTensor<B_TYPE> bGM;                                                                   \
            AscendC::GlobalTensor<C_GM_TYPE> cGM;                                                                \
            const int64_t aBatchOffset = static_cast<int64_t>(st.batchIdx) * st.tiling.strideA;                  \
            const int64_t bBatchOffset = static_cast<int64_t>(st.batchIdx) * st.tiling.strideB;                  \
            const int64_t cBatchOffset = static_cast<int64_t>(st.batchIdx) * st.tiling.workspaceBatchStride +    \
                                         static_cast<int64_t>(st.splitIdx) * st.tiling.workspaceSplitStride;     \
            aGM.SetGlobalBuffer(reinterpret_cast<__gm__ A_TYPE*>(a) + aBatchOffset);                             \
            bGM.SetGlobalBuffer(reinterpret_cast<__gm__ B_TYPE*>(b) + bBatchOffset);                             \
            cGM.SetGlobalBuffer(reinterpret_cast<__gm__ C_GM_TYPE*>(c) + cBatchOffset);                          \
            RunStridedTask<A_TYPE, B_TYPE, C_GM_TYPE, ACC_TYPE, (BM), (BK), (BN), (C0_VAL), QUANT_MODE>(         \
                st, aGM, bGM, cGM, a1, b1, a1Alt, b1Alt, a2, b2);                                                \
        }                                                                                                        \
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);                                                  \
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);                                                 \
    }
