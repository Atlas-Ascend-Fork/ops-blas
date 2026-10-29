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
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "cgemm_strided_batched_kernel.h"

namespace cgemm_sb {
using AscendC::Add;
using AscendC::Adds;
using AscendC::Cast;
using AscendC::CreateVecIndex;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
using AscendC::Duplicate;
using AscendC::Fixpipe;
using AscendC::FixpipeParamsV220;
using AscendC::Gather;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::HF32Mode;
using AscendC::InitSocState;
using AscendC::LoadData;
using AscendC::LoadData3DParamsV2;
using AscendC::LocalTensor;
using AscendC::Mmad;
using AscendC::MmadParams;
using AscendC::Muls;
using AscendC::Nd2NzParams;
using AscendC::PipeBarrier;
using AscendC::RoundMode;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicNone;
using AscendC::SetFlag;
using AscendC::SetHF32Mode;
using AscendC::SetMaskNorm;
using AscendC::ShiftRight;
using AscendC::Sub;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::WaitFlag;
} // namespace cgemm_sb

// ---- vector implementation ----

namespace cgemm_sb {

// Seed byte offsets for [real[0], imag[0], real[1], imag[1], ...].
__aicore__ inline void SbInterleavedSeed(LocalTensor<int32_t> indices, uint32_t planeElements)
{
    constexpr uint32_t seedElements = 16U;
    CreateVecIndex(indices, 0, seedElements);
    ShiftRight(indices, indices, 1, seedElements);
    Muls(indices, indices, static_cast<int32_t>(sizeof(float)), seedElements);
    uint64_t oddLanes[2] = {0xaaaaULL, 0U}; // Select the imaginary component in each pair.
    Adds(indices, indices, static_cast<int32_t>(planeElements * sizeof(float)), oddLanes, 1U, {1, 1, 8, 8});
}

__aicore__ inline uint64_t SbCompactWorkspaceStride(uint32_t m)
{
    const uint32_t padded = m <= 32U ? 32U : (m + 63U) / 64U * 64U;
    return 8ULL * padded * padded;
}

__aicore__ inline void SbGather(
    LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<uint32_t> offsets, uint32_t base, uint32_t count)
{
    const uint32_t full = count / 64U;
    if (full != 0U) {
        Gather(dst, src, offsets, base, 64ULL, static_cast<uint8_t>(full), 8U);
    }
    if (count % 64U != 0U) {
        Gather(dst[full * 64U], src, offsets[full * 64U], base, static_cast<uint64_t>(count % 64U), 1U, 8U);
    }
}

// Fixed normal-mask operations avoid changing CTRL for every index-building step.
__aicore__ inline void SbIndex(LocalTensor<int32_t> dst, int32_t first, uint32_t count)
{
    CreateVecIndex(dst, first, static_cast<uint64_t>(count), 1U, 1U, 8U);
}

__aicore__ inline void SbIndexAdd(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, int32_t value, uint32_t count)
{
    const uint32_t full = count / 64U;
    if (full != 0U) {
        Adds(dst, src, value, 64ULL, static_cast<uint8_t>(full), {1, 1, 8, 8});
    }
    if (count % 64U) {
        Adds(dst[full * 64U], src[full * 64U], value, static_cast<uint64_t>(count % 64U), 1U, {1, 1, 8, 8});
    }
}

__aicore__ inline void SbIndexMul(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, int32_t value, uint32_t count)
{
    Muls(dst, src, value, static_cast<uint64_t>(count), 1U, {1, 1, 8, 8});
}

__aicore__ inline void SbIndexShift(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, int32_t value, uint32_t count)
{
    ShiftRight(dst, src, value, static_cast<uint64_t>(count), 1U, {1, 1, 8, 8});
}

__aicore__ inline void SbFill(LocalTensor<float> dst, float value, uint32_t count)
{
    const uint32_t full = count / 64U;
    if (full != 0U) {
        Duplicate(dst, value, 64ULL, static_cast<uint8_t>(full), 1U, 8U);
    }
    if (count % 64U) {
        Duplicate(dst[full * 64U], value, static_cast<uint64_t>(count % 64U), 1U, 1U, 8U);
    }
}
template <bool left, bool initializeMask = false>
__aicore__ inline void SbNzSeed(LocalTensor<int32_t> ix, uint32_t trans, uint32_t pitch)
{
    if constexpr (initializeMask) {
        SetMaskNorm();
    }
    SbIndex(ix, 0, 8U);
    if (trans == ACLBLAS_OP_N) {
        SbIndexMul(ix, ix, 4, 8U);
        if constexpr (left) {
            SbIndex(ix[8U], 1, 8U);
            uint64_t odd[2] = {0xaaULL, 0U};
            Adds(ix[8U], ix[8U], -2, odd, 1U, {1, 1, 8, 8});
            SbIndexMul(ix[8U], ix[8U], 4, 8U);
        }
    } else {
        SbIndexShift(ix, ix, 1, 8U);
        SbIndexMul(ix, ix, pitch * 4, 8U);
        if constexpr (left) {
            SbIndexAdd(ix[8U], ix, 0, 8U);
        }
        uint64_t odd[2] = {0xaaULL, 0U};
        uint64_t even[2] = {0x55ULL, 0U};
        Adds(ix, ix, 4, odd, 1U, {1, 1, 8, 8});
        if constexpr (left) {
            Adds(ix[8U], ix[8U], 4, even, 1U, {1, 1, 8, 8});
        }
    }
}
__aicore__ inline void SbInitLoadParams(
    LoadData3DParamsV2<float>& p, uint32_t width, uint32_t channels, uint32_t k, uint32_t m, bool transpose)
{
    p.l1H = 1U;
    p.l1W = width;
    p.channelSize = channels;
    p.kExtension = k;
    p.mExtension = m;
    p.strideW = 1U;
    p.strideH = 1U;
    p.filterW = 1U;
    p.filterH = 1U;
    p.dilationFilterW = 1U;
    p.dilationFilterH = 1U;
    p.filterSizeW = false;
    p.filterSizeH = false;
    p.enTranspose = transpose;
    p.fMatrixCtrl = false;
}
template <uint32_t area>
__aicore__ inline void SbZeroOutput(GlobalTensor<float> cg, LocalTensor<float> output, uint64_t total, uint32_t cores)
{
    for (uint64_t pos = GetBlockIdx() * area; pos < total; pos += cores * 2U * area) {
        const uint32_t n = total - pos < area ? total - pos : area;
        DataCopyExtParams z{1U, n * 4U, 0U, 0U, 0U};
        DataCopyPad(cg[pos], output, z);
    }
    CrossCoreSetFlag<0, PIPE_MTE3>(10U);
    CrossCoreWaitFlag<0>(10U);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}
template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbExpandNzMatrix(LocalTensor<int32_t> ix, uint32_t rows, uint32_t sourcePitch)
{
    for (uint32_t r = left ? 2U : 1U; r < rows; r *= 2U) {
        const uint32_t part = rows - r < r ? rows - r : r;
        const uint32_t step = (left ? r / 2U : r) * (trans == ACLBLAS_OP_N ? sourcePitch * 4U : 8U);
        SbIndexAdd(ix[r * 8U], ix, step, part * 8U);
    }
    for (uint32_t blocks = 1U; blocks < width / 4U; blocks *= 2U) {
        const uint32_t part = width / 4U - blocks < blocks ? width / 4U - blocks : blocks;
        const uint32_t step = blocks * (trans == ACLBLAS_OP_N ? 32U : sourcePitch * 16U);
        SbIndexAdd(ix[blocks * rows * 8U], ix, step, part * rows * 8U);
    }
}
__aicore__ inline void SbSignalPackedRound(uint32_t rounds)
{
    if (rounds % 2U == 0U) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    } else {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(8U);
}

template <bool waitFirst>
__aicore__ inline void SbDrainPackedOutput(uint32_t rounds)
{
    if constexpr (waitFirst) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    if (rounds != 0U) {
        CrossCoreWaitFlag<2>(9U);
    }
}

template <uint32_t size>
__aicore__ inline void SbStoreAugmentedGroup(
    GlobalTensor<float> ws, LocalTensor<float> output, uint64_t workspaceStride, uint32_t first, uint32_t nb)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams oc{
        static_cast<uint16_t>(nb), size * sizeof(float), 0U,
        static_cast<uint32_t>((workspaceStride - size) * sizeof(float)), 0U};
    DataCopyPad(ws[first * workspaceStride], output, oc);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

} // namespace cgemm_sb

// ---- augmented mmad implementation ----
namespace cgemm_sb {
template <uint32_t kCgemmSbCubeBaseK, uint32_t baseM, uint32_t baseN, bool outputNZ = false, bool doubleCo1 = false>
class CgemmSbAugmentedMmad {
public:
    __aicore__ inline void Init(
        __gm__ float* left, __gm__ float* right, __gm__ float* out, uint32_t m, uint32_t n, uint32_t k,
        uint32_t leftStride, uint32_t outStride, uint32_t rightStride = 0U)
    {
        ar_.SetGlobalBuffer(left);
        br_.SetGlobalBuffer(right);
        rr_.SetGlobalBuffer(out);
        m_ = m;
        n_ = n;
        k_ = k;
        leftStride_ = leftStride;
        outStride_ = outStride;
        rightStride_ = rightStride == 0U ? n : rightStride;
    }

    __aicore__ inline void PrepareCo1() { SetFlag<HardEvent::FIX_M>(EVENT_ID0); }

    // Transfer several independent matrices with one ND2NZ command per operand.
    __aicore__ inline void Preload(uint32_t count, uint32_t leftBatchStride, uint32_t rightBatchStride)
    {
        constexpr uint32_t as = baseM * kCgemmSbCubeBaseK;
        constexpr uint32_t bs = kCgemmSbCubeBaseK * baseN;
        LocalTensor<float> left(TPosition::A1, 0U, as * count);
        LocalTensor<float> right(TPosition::B1, 128U * 1024U, bs * count);
        Nd2NzParams ap{
            static_cast<uint16_t>(count),
            static_cast<uint16_t>(m_),
            static_cast<uint16_t>(kCgemmSbCubeBaseK),
            static_cast<uint16_t>(leftBatchStride),
            static_cast<uint16_t>(leftStride_),
            static_cast<uint16_t>(baseM),
            1U,
            static_cast<uint16_t>(as)};
        Nd2NzParams bp{
            static_cast<uint16_t>(count),
            static_cast<uint16_t>(kCgemmSbCubeBaseK),
            static_cast<uint16_t>(n_),
            static_cast<uint16_t>(rightBatchStride),
            static_cast<uint16_t>(rightStride_),
            static_cast<uint16_t>(kCgemmSbCubeBaseK),
            1U,
            static_cast<uint16_t>(bs)};
        DataCopy(left, ar_, ap);
        DataCopy(right, br_, bp);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    // Inputs are already in the native L1 NZ layout, so no ND conversion is needed.
    __aicore__ inline void PreloadNz(uint32_t count, uint32_t batchStride)
    {
        constexpr uint32_t as = baseM * kCgemmSbCubeBaseK;
        constexpr uint32_t bs = kCgemmSbCubeBaseK * baseN;
        LocalTensor<float> left(TPosition::A1, 0U, count * as);
        LocalTensor<float> right(TPosition::B1, 128U * 1024U, count * bs);
        DataCopyParams ap{
            static_cast<uint16_t>(count), static_cast<uint16_t>(as / 8U),
            static_cast<uint16_t>((batchStride - as) / 8U), 0U};
        DataCopyParams bp{
            static_cast<uint16_t>(count), static_cast<uint16_t>(bs / 8U),
            static_cast<uint16_t>((batchStride - bs) / 8U), 0U};
        DataCopy(left, ar_, ap);
        DataCopy(right, br_, bp);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void PreloadDiagonalNz(uint32_t count, uint32_t batchStride, uint32_t tileM, uint32_t tileN)
    {
        constexpr uint32_t as = baseM * kCgemmSbCubeBaseK;
        constexpr uint32_t bs = kCgemmSbCubeBaseK * baseN;
        LocalTensor<float> left(TPosition::A1, 0U, as);
        LocalTensor<float> right(TPosition::B1, 128U * 1024U, bs);
        DataCopyParams ap{
            static_cast<uint16_t>(kCgemmSbCubeBaseK / 8U), static_cast<uint16_t>(tileM), 0U,
            static_cast<uint16_t>(baseM - tileM)};
        for (uint32_t i = 0U; i < count; ++i) {
            DataCopy(left[i * tileM * 8U], ar_[i * batchStride], ap);
        }
        const uint32_t batchB = kCgemmSbCubeBaseK * tileN;
        DataCopyParams bp{
            static_cast<uint16_t>(count), static_cast<uint16_t>(batchB / 8U),
            static_cast<uint16_t>((batchStride - batchB) / 8U), 0U};
        DataCopy(right, br_, bp);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11>
    __aicore__ inline void SbStoreAugmentedTile(
        Arg0& diagonalTile, Arg1& diagonalCount, Arg2& diagonalMStride, Arg3& diagonalNStride, Arg4& co1Slot, Arg5& rr1,
        Arg6& mBase, Arg7& nBase, Arg8& cols, Arg9& activeM, Arg10& activeN, Arg11& outParams)
    {
        outParams.nSize = outputNZ ? activeN : cols;
        outParams.srcStride = activeM;
        outParams.dstStride = outStride_;
        outParams.ndNum = 1;
        outParams.srcNdStride = 0;
        outParams.dstNdStride = 0;
        const uint64_t outOffset = static_cast<uint64_t>(mBase) * outStride_ + nBase;
        if constexpr (outputNZ) {
            Fixpipe<float, float, AscendC::CFG_NZ>(rr_[outOffset], rr1, outParams);
        } else if (diagonalTile == 0U) {
            Fixpipe(rr_[outOffset], rr1, outParams);
        } else {
            outParams.mSize = diagonalTile;
            outParams.nSize = diagonalTile * 2U;
            for (uint32_t batch = 0U; batch < diagonalCount; ++batch) {
                const uint32_t source =
                    batch * (diagonalNStride == 0U ? diagonalTile * 2U : diagonalNStride) * activeM +
                    batch * (diagonalMStride == 0U ? diagonalTile : diagonalMStride) * 16U;
                Fixpipe(rr_[static_cast<uint64_t>(batch) * diagonalTile * outStride_], rr1[source], outParams);
            }
        }
        if constexpr (doubleCo1) {
            if (co1Slot == 0U) {
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::FIX_M>(EVENT_ID1);
            }
        } else {
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8, typename Arg9>
    __aicore__ inline void SbAccumulateAugmentedTile(
        Arg0& co1Slot, Arg1& br1, Arg2& ar2, Arg3& br2, Arg4& rr1, Arg5& activeM, Arg6& activeN, Arg7& kBase,
        Arg8& activeK, Arg9& loadB)
    {
        LoadData(br2, br1, loadB);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        MmadParams params = {};
        params.m = activeM;
        params.n = activeN;
        params.k = activeK;
        params.cmatrixInitVal = kBase == 0U;
        params.cmatrixSource = false;
        params.kDirectionAlign = false;
        if (kBase == 0U) {
            WaitOutput(co1Slot);
        }
        Mmad(rr1, ar2, br2, params);
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8>
    __aicore__ inline void SbCopyAugmentedOperands(
        Arg0& preloaded, Arg1& ar1, Arg2& br1, Arg3& mBase, Arg4& nBase, Arg5& rows, Arg6& cols, Arg7& kBase,
        Arg8& activeK)
    {
        if (kBase != 0U) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        }
        if (!preloaded) {
            Nd2NzParams aParams;
            aParams.ndNum = 1;
            aParams.nValue = rows;
            aParams.dValue = activeK;
            aParams.srcNdMatrixStride = 0;
            aParams.srcDValue = leftStride_;
            aParams.dstNzC0Stride = baseM;
            aParams.dstNzNStride = 1;
            aParams.dstNzMatrixStride = 0;
            const uint64_t aOffset = static_cast<uint64_t>(mBase) * leftStride_ + kBase;
            DataCopy(ar1, ar_[aOffset], aParams);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

            Nd2NzParams bParams;
            bParams.ndNum = 1;
            bParams.nValue = activeK;
            bParams.dValue = cols;
            bParams.srcNdMatrixStride = 0;
            bParams.srcDValue = rightStride_;
            bParams.dstNzC0Stride = kCgemmSbCubeBaseK;
            bParams.dstNzNStride = 1;
            bParams.dstNzMatrixStride = 0;
            const uint64_t bOffset = static_cast<uint64_t>(kBase) * rightStride_ + nBase;
            DataCopy(br1, br_[bOffset], bParams);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        }
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8, typename Arg9, typename Arg10>
    __aicore__ inline void SbLoadAugmentedOperands(
        Arg0& preloaded, Arg1& ar1, Arg2& br1, Arg3& ar2, Arg4& mBase, Arg5& nBase, Arg6& rows, Arg7& cols,
        Arg8& activeM, Arg9& kBase, Arg10& activeK)
    {
        SbCopyAugmentedOperands(preloaded, ar1, br1, mBase, nBase, rows, cols, kBase, activeK);
        if (!preloaded) {
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        }
        if (kBase != 0U) {
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        }
        LoadData3DParamsV2<float> loadA;
        cgemm_sb::SbInitLoadParams(loadA, baseM, kCgemmSbCubeBaseK, activeK, activeM, false);
        LoadData(ar2, ar1, loadA);

        if (!preloaded) {
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        }
    }

    template <
        typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
        typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12, typename Arg13>
    __aicore__ inline void SbComputeAugmentedTile(
        Arg0& preloaded, Arg1& diagonalTile, Arg2& diagonalCount, Arg3& diagonalMStride, Arg4& diagonalNStride,
        Arg5& co1Slot, Arg6& ar1, Arg7& br1, Arg8& ar2, Arg9& br2, Arg10& rr1, Arg11& nTiles, Arg12& tile, Arg13& mBase)
    {
        const uint32_t nBase = static_cast<uint32_t>((tile % nTiles) * baseN);
        const uint32_t rows = m_ - mBase < baseM ? m_ - mBase : baseM;
        const uint32_t cols = n_ - nBase < baseN ? n_ - nBase : baseN;
        const uint32_t activeM = (rows + 15U) / 16U * 16U;
        const uint32_t activeN = (cols + 15U) / 16U * 16U;
        for (uint32_t kBase = 0U; kBase < k_; kBase += kCgemmSbCubeBaseK) {
            const uint32_t activeK =
                k_ - kBase < kCgemmSbCubeBaseK ? (k_ - kBase + 15U) / 16U * 16U : kCgemmSbCubeBaseK;
            SbLoadAugmentedOperands(preloaded, ar1, br1, ar2, mBase, nBase, rows, cols, activeM, kBase, activeK);
            LoadData3DParamsV2<float> loadB;
            cgemm_sb::SbInitLoadParams(loadB, kCgemmSbCubeBaseK, baseN, activeN, activeK, true);
            SbAccumulateAugmentedTile(co1Slot, br1, ar2, br2, rr1, activeM, activeN, kBase, activeK, loadB);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 outParams = {};
        outParams.mSize = rows;
        SbStoreAugmentedTile(
            diagonalTile, diagonalCount, diagonalMStride, diagonalNStride, co1Slot, rr1, mBase, nBase, cols, activeM,
            activeN, outParams);
    }

    __aicore__ inline void Process(
        uint32_t tileIndex, uint32_t tileStride, bool keepState = false, bool preloaded = false,
        uint32_t batchSlot = 0U, uint32_t diagonalTile = 0U, uint32_t diagonalCount = 0U, uint32_t diagonalMStride = 0U,
        uint32_t diagonalNStride = 0U, uint32_t co1Slot = 0U)
    {
        constexpr uint32_t aSize = baseM * kCgemmSbCubeBaseK;
        constexpr uint32_t bSize = kCgemmSbCubeBaseK * baseN;
        constexpr uint32_t cSize = baseM * baseN;
        LocalTensor<float> ar1(TPosition::A1, preloaded ? batchSlot * aSize * sizeof(float) : 0U, aSize);
        LocalTensor<float> br1(
            TPosition::B1, preloaded ? 128U * 1024U + batchSlot * bSize * sizeof(float) : 2U * aSize * sizeof(float),
            bSize);
        LocalTensor<float> ar2(TPosition::A2, 0U, aSize);
        LocalTensor<float> br2(TPosition::B2, 0U, bSize);
        LocalTensor<float> rr1(TPosition::CO1, (doubleCo1 ? co1Slot * cSize * sizeof(float) : 0U), cSize);
        const uint32_t nTiles = (n_ + baseN - 1U) / baseN;
        const uint64_t tileCount = static_cast<uint64_t>((m_ + baseM - 1U) / baseM) * nTiles;
        for (uint64_t tile = tileIndex; tile < tileCount; tile += tileStride) {
            const uint32_t mBase = static_cast<uint32_t>((tile / nTiles) * baseM);
            SbComputeAugmentedTile(
                preloaded, diagonalTile, diagonalCount, diagonalMStride, diagonalNStride, co1Slot, ar1, br1, ar2, br2,
                rr1, nTiles, tile, mBase);
        }
        if (!keepState) {
            WaitOutput(co1Slot);
        }
    }

private:
    __aicore__ inline void WaitOutput(uint32_t co1Slot)
    {
        if constexpr (doubleCo1) {
            if (co1Slot != 0U) {
                WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
                return;
            }
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    GlobalTensor<float> ar_;
    GlobalTensor<float> br_;
    GlobalTensor<float> rr_;
    uint32_t m_;
    uint32_t n_;
    uint32_t k_;
    uint32_t leftStride_, outStride_, rightStride_;
};

} // namespace cgemm_sb
