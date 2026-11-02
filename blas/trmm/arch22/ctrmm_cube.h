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
#include "ctrmm_kernel_common.h"

namespace {
constexpr uint32_t kCtrmmCubeBaseM = kCubeTile;
constexpr uint32_t kCtrmmCubeBaseK = 8U;
constexpr uint32_t kCtrmmBatchK = kCubeTile;
constexpr IsResetLoad3dConfig kKeepLoad3d{false, false};

// Real and imaginary outputs share one CO1 tile. A has 2 or 8 slots per
// complex coefficient; B stores two useful rows and is expanded only in L1.
// Long reductions retain one complex term per MMAD and the Netlib K direction.
template <uint32_t kCtrmmCubeBaseN>
class CtrmmCube {
public:
    __aicore__ inline void Init(
        __gm__ float* ar, __gm__ float* br, __gm__ float* bi, __gm__ float* rr, __gm__ float* ri, uint32_t m,
        uint32_t n, uint32_t k, bool upper, bool descending, __gm__ float* diagonal)
    {
        ar_.SetGlobalBuffer(ar, static_cast<uint64_t>(m) * k);
        br_.SetGlobalBuffer(br, static_cast<uint64_t>(k) * n);
        bi_.SetGlobalBuffer(bi, static_cast<uint64_t>(k) * n);
        rr_.SetGlobalBuffer(rr, static_cast<uint64_t>(m) * n);
        ri_.SetGlobalBuffer(ri, static_cast<uint64_t>(m) * n);
        m_ = m;
        n_ = n;
        k_ = k;
        componentStride_ = m <= kCompactMaxOrder ? kCompactComponents : kExpandedComponents;
        upper_ = upper;
        descending_ = descending;
        diagonal_.SetGlobalBuffer(diagonal);
    }

    __aicore__ inline void PrepareCo1() { SetFlag<HardEvent::FIX_M>(EVENT_ID0); }

    __aicore__ inline void Process(uint32_t tileIndex, uint32_t tileStride)
    {
        // FP32 Cube format: c0=8, fractalNum=2, fractalSize=16*8.
        // M tails use 16-row granularity; N is 64/128 complex outputs.
        // L1 and L0 double buffering preserves each output's accumulation order.
        const uint32_t mTiles = (m_ + kCtrmmCubeBaseM - 1U) / kCtrmmCubeBaseM;
        const uint32_t nTiles = n_ / kCtrmmCubeBaseN;
        const uint64_t tileCount = static_cast<uint64_t>(mTiles) * nTiles;

        clearB_.repeatTimes = 1;
        clearB_.blockNum = 2U * bSizeAlignL1 / 8U;
        clearB_.dstGap = 0;
        clearB_.initValue = 0.0f;
        InitConstValue(zeroB0_, clearB_);
        InitConstValue(zeroB1_, clearB_);
        PipeBarrier<PIPE_ALL>();
        SetHF32Mode(HF32Mode::DISABLE);
        for (uint64_t tile = tileIndex; tile < tileCount; tile += tileStride) {
            ProcessTile(tile, nTiles);
        }
        // Drain the final Fixpipe completion token. Without this wait, a
        // later kernel invocation can block while reinitializing FIX_M.
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

private:
    static constexpr uint32_t aSizeAlignL1 = kCtrmmCubeBaseM * kCtrmmBatchK;
    static constexpr uint32_t bSizeAlignL1 = kCtrmmBatchK * kCtrmmCubeBaseN;
    static constexpr uint32_t aSizeAlignL0 = kCtrmmCubeBaseM * kCubeL0K;
    static constexpr uint32_t bSizeAlignL0 = kCubeL0K * kCtrmmCubeBaseN;
    static constexpr uint32_t cSizeAlignL0 = kCtrmmCubeBaseM * kCtrmmCubeBaseN;

    __aicore__ inline void ComputeGroup(uint32_t l0Slot, bool separateDiagonal, uint32_t batchStep, uint32_t groupStep)
    {
        constexpr uint32_t groupK = kCubeGroupK;
        for (uint32_t step = 0; step < groupK / kCtrmmCubeBaseK; ++step) {
            const uint32_t slice = descending_ ? groupK / kCtrmmCubeBaseK - 1U - step : step;
            auto aSlice = ar2_[(l0Slot * groupK + slice * kCtrmmCubeBaseK) * kCtrmmCubeBaseM];
            auto bSlice = br2_[(l0Slot * groupK + slice * kCtrmmCubeBaseK) * 2U * kCtrmmCubeBaseN];
            Compute(
                accumulator_, aSlice, bSlice, !separateDiagonal && batchStep == 0U && groupStep == 0U && step == 0U,
                componentStride_ <= 4U ? 8U : 2U);
        }
    }

    __aicore__ inline void PrefetchGroup(
        uint32_t groupStep, uint32_t groups, uint32_t l0Slot, LocalTensor<float>& aCurrent,
        LocalTensor<float>& bCurrent)
    {
        constexpr uint32_t groupK = kCubeGroupK;
        if (groupStep + 1U < groups) {
            const uint32_t nextSlot = 1U - l0Slot;
            const event_t nextReady = nextSlot == 0U ? EVENT_ID0 : EVENT_ID1;
            if (groupStep != 0U) {
                WaitFlag<HardEvent::M_MTE1>(nextReady);
            }
            auto aNext = ar2_[nextSlot * kCtrmmCubeBaseM * groupK];
            auto bNext = br2_[nextSlot * 2U * kCtrmmCubeBaseN * groupK];
            const uint32_t nextGroup = descending_ ? groups - 2U - groupStep : groupStep + 1U;
            LoadGroup(aNext, bNext, aCurrent, bCurrent, nextGroup);
            SetFlag<HardEvent::MTE1_M>(nextReady);
        }
    }

    __aicore__ inline void RunGroups(
        LocalTensor<float>& aCurrent, LocalTensor<float>& bCurrent, bool separateDiagonal, uint32_t batchStep)
    {
        constexpr uint32_t groupK = kCubeGroupK;
        const uint32_t groups = kCtrmmBatchK / groupK;
        const uint32_t firstGroup = descending_ ? groups - 1U : 0U;
        LoadGroup(ar2_, br2_, aCurrent, bCurrent, firstGroup);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        for (uint32_t groupStep = 0; groupStep < groups; ++groupStep) {
            const uint32_t l0Slot = groupStep % 2U;
            const event_t l0Ready = l0Slot == 0U ? EVENT_ID0 : EVENT_ID1;
            WaitFlag<HardEvent::MTE1_M>(l0Ready);
            PrefetchGroup(groupStep, groups, l0Slot, aCurrent, bCurrent);
            ComputeGroup(l0Slot, separateDiagonal, batchStep, groupStep);
            SetFlag<HardEvent::M_MTE1>(l0Ready);
        }
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    }

    __aicore__ inline void PrefetchBatch(
        uint32_t batchStep, uint32_t batches, uint32_t slot, uint32_t begin, uint32_t end, uint32_t mBase,
        uint32_t nBase)
    {
        constexpr uint32_t l1Span = aSizeAlignL1 + 2U * bSizeAlignL1;
        if (batchStep + 1U < batches) {
            const uint32_t nextSlot = 1U - slot;
            const event_t nextReady = nextSlot == 0U ? EVENT_ID0 : EVENT_ID1;
            if (batchStep != 0U) {
                WaitFlag<HardEvent::MTE1_MTE2>(nextReady);
            }
            LocalTensor<float> aNext(TPosition::A1, nextSlot * l1Span * kFloatBytes, aSizeAlignL1);
            LocalTensor<float> brNext(TPosition::B1, (nextSlot * l1Span + aSizeAlignL1) * kFloatBytes, bSizeAlignL1);
            LocalTensor<float> biNext(
                TPosition::B1, (nextSlot * l1Span + aSizeAlignL1 + bSizeAlignL1) * kFloatBytes, bSizeAlignL1);
            const uint32_t nextBatch =
                descending_ ? end - kCtrmmBatchK * (batchStep + 2U) : begin + kCtrmmBatchK * (batchStep + 1U);
            CopyToL1(aNext, brNext, ar_, br_, mBase, nBase, nextBatch, true);
            CopyToL1(aNext, biNext, ar_, bi_, mBase, nBase, nextBatch, false);
            SetFlag<HardEvent::MTE2_MTE1>(nextReady);
        }
    }

    __aicore__ inline void RunBatches(
        uint32_t begin, uint32_t end, uint32_t mBase, uint32_t nBase, bool separateDiagonal)
    {
        const uint32_t batches = (end - begin) / kCtrmmBatchK;
        constexpr uint32_t l1Span = aSizeAlignL1 + 2U * bSizeAlignL1;
        const uint32_t firstBatch = descending_ ? end - kCtrmmBatchK : begin;
        CopyToL1(ar1_, br1_, ar_, br_, mBase, nBase, firstBatch, true);
        CopyToL1(ar1_, bi1_, ar_, bi_, mBase, nBase, firstBatch, false);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        for (uint32_t batchStep = 0; batchStep < batches; ++batchStep) {
            const uint32_t slot = batchStep % 2U;
            const event_t ready = slot == 0U ? EVENT_ID0 : EVENT_ID1;
            WaitFlag<HardEvent::MTE2_MTE1>(ready);
            LocalTensor<float> aCurrent(TPosition::A1, slot * l1Span * kFloatBytes, aSizeAlignL1);
            LocalTensor<float> bCurrent(TPosition::B1, (slot * l1Span + aSizeAlignL1) * kFloatBytes, 2U * bSizeAlignL1);
            PrefetchBatch(batchStep, batches, slot, begin, end, mBase, nBase);
            RunGroups(aCurrent, bCurrent, separateDiagonal, batchStep);
            SetFlag<HardEvent::MTE1_MTE2>(ready);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        if (batches > 1U) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline void SeedDiagonal(uint32_t mBase, uint32_t nBase, bool separateDiagonal)
    {
        // A separate seed is needed only when the k loop reaches the diagonal last.
        for (uint32_t dk = 0; separateDiagonal && dk < 2U * mTile_; dk += kCtrmmBatchK) {
            Fence<HardEvent::MTE1_MTE2>();
            CopyToL1(ar1_, br1_, diagonal_, br_, mBase, nBase, dk, true, kDiagonalRowFloats, 2U * mBase + dk, true);
            CopyToL1(ar1_, bi1_, diagonal_, bi_, mBase, nBase, dk, false, kDiagonalRowFloats, 2U * mBase + dk, true);
            Fence<HardEvent::MTE2_MTE1>();
            for (uint32_t dkPart = 0; dkPart < kCtrmmBatchK; dkPart += kCubeL0K) {
                LoadToL0(ar1_, br1_, ar2_, br2_, dkPart, true, kCubeL0K);
                Fence<HardEvent::MTE1_M>();
                Compute(accumulator_, ar2_, br2_, dk == 0U && dkPart == 0U, kCubeL0K);
                PipeBarrier<PIPE_ALL>();
            }
        }
        if (separateDiagonal && componentStride_ != 2U) {
            InitConstValue(zeroB0_, clearB_);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessTile(uint64_t tile, uint32_t nTiles)
    {
        if (nTiles == 0U) {
            return;
        }
        const uint32_t mBase = static_cast<uint32_t>((tile / nTiles) * kCtrmmCubeBaseM);
        const uint32_t nBase = static_cast<uint32_t>((tile % nTiles) * kCtrmmCubeBaseN);
        mTile_ = Min(kCtrmmCubeBaseM, m_ - mBase);
        auto ri = accumulator_[mTile_ * kCtrmmCubeBaseN];
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        const uint32_t begin = upper_ ? componentStride_ * mBase : 0U;
        const uint32_t end =
            upper_ ? k_ : (componentStride_ * (mBase + mTile_) + kCtrmmBatchK - 1U) / kCtrmmBatchK * kCtrmmBatchK;
        const bool separateDiagonal = !upper_ && !descending_;
        SeedDiagonal(mBase, nBase, separateDiagonal);
        RunBatches(begin, end, mBase, nBase, separateDiagonal);
        Fence<HardEvent::M_FIX>();
        StoreToGm(accumulator_, rr_, mBase, nBase);

        StoreToGm(ri, ri_, mBase, nBase);

        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void CopyToL1(
        LocalTensor<float>& a1, LocalTensor<float>& b1, GlobalTensor<float>& left, GlobalTensor<float>& right,
        uint32_t mBase, uint32_t nBase, uint32_t kBase, bool copyA, uint32_t leftStride = 0U,
        uint32_t rightBase = kInvalidCachedRow, bool compactB = false)
    {
        Nd2NzParams aParams;
        aParams.ndNum = 1;
        aParams.nValue = mTile_;
        aParams.dValue = kCtrmmBatchK;
        aParams.srcNdMatrixStride = 0;
        const uint32_t aStride = leftStride ? leftStride : k_;
        aParams.srcDValue = aStride;
        aParams.dstNzC0Stride = kCtrmmCubeBaseM;
        aParams.dstNzNStride = 1;
        aParams.dstNzMatrixStride = 0;
        if (copyA) {
            DataCopy(a1, left[static_cast<uint64_t>(mBase) * aStride + kBase], aParams);
        }

        Nd2NzParams bParams{};
        bParams.ndNum = kCtrmmBatchK / componentStride_;
        bParams.nValue = 2U;
        bParams.dValue = kCtrmmCubeBaseN;
        bParams.srcNdMatrixStride = 2U * n_;
        bParams.srcDValue = n_;
        bParams.dstNzC0Stride = kCtrmmBatchK;
        bParams.dstNzNStride = 1U;
        bParams.dstNzMatrixStride = 8U * componentStride_;
        uint32_t sourceK = rightBase == kInvalidCachedRow ? kBase : rightBase;
        if (componentStride_ == 2U || compactB) {
            bParams.ndNum = 1U;
            bParams.nValue = kCtrmmBatchK;
            bParams.srcNdMatrixStride = 0U;
            bParams.dstNzMatrixStride = 0U;
        } else {
            sourceK /= componentStride_ / 2U;
        }
        DataCopy(b1, right[static_cast<uint64_t>(sourceK) * n_ + nBase], bParams);
    }

    __aicore__ inline void LoadAToL0(
        LocalTensor<float>& a1, LocalTensor<float>& a2, uint32_t offset, bool copyA, uint32_t physicalK)
    {
        // FP32 NZ -> ZZ, equivalent to the official LoadData3DV2
        // false/false KernelMmad path. For 64x64: L1/L0=4096 floats.
        LoadData3DParamsV2<float> aParams;
        aParams.l1H = 1U;
        aParams.l1W = kCtrmmCubeBaseM;
        aParams.channelSize = kCtrmmBatchK;
        aParams.kStartPt = offset;
        aParams.kExtension = physicalK;
        aParams.mExtension = mTile_;
        aParams.strideW = 1U;
        aParams.strideH = 1U;
        aParams.filterW = 1U;
        aParams.filterH = 1U;
        aParams.dilationFilterW = 1U;
        aParams.dilationFilterH = 1U;
        aParams.filterSizeW = false;
        aParams.filterSizeH = false;
        aParams.enTranspose = false;
        aParams.fMatrixCtrl = false;
        if (copyA) {
            if (physicalK == kCtrmmCubeBaseK) {
                // A single K8 slice has one contiguous column of 16x8
                // fractals in both L1 NZ and L0A ZZ, so no transpose is needed.
                LoadData2DParams slice{};
                slice.startIndex = offset / kCtrmmCubeBaseK * (kCtrmmCubeBaseM / kCubeFractalRows);
                slice.repeatTimes = kCtrmmCubeBaseM / kCubeFractalRows;
                slice.srcStride = 1;
                slice.dstGap = 0;
                slice.ifTranspose = false;
                LoadData(a2, a1, slice);
            } else {
                LoadData(a2, a1, aParams);
            }
        }
    }

    __aicore__ inline void LoadToL0(
        LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2, uint32_t offset,
        bool copyA, uint32_t physicalK = kCtrmmCubeBaseK)
    {
        LoadAToL0(a1, a2, offset, copyA, physicalK);

        // FP32 NZ -> ZN. For K=N=64 this uses four K fractals and
        // eight N c0 repeats; the hardware transpose supplies ZN layout.
        LoadData3DParamsV2<float> bParams;
        bParams.l1H = 1U;
        bParams.l1W = kCtrmmBatchK;
        bParams.mStartPt = offset;
        bParams.channelSize = 2U * kCtrmmCubeBaseN;
        bParams.kExtension = 2U * kCtrmmCubeBaseN;
        bParams.mExtension = physicalK;
        bParams.strideW = 1U;
        bParams.strideH = 1U;
        bParams.filterW = 1U;
        bParams.filterH = 1U;
        bParams.dilationFilterW = 1U;
        bParams.dilationFilterH = 1U;
        bParams.filterSizeW = false;
        bParams.filterSizeH = false;
        bParams.enTranspose = true;
        bParams.fMatrixCtrl = false;
        // Full-K diagonal loads reset FMatrix after A's 3D load. Their
        // last B load leaves exactly the same FMatrix needed by the entire
        // K8 loop. A's 2D load does not change that configuration.
        if (!bConfigReady_ || (physicalK != kCtrmmCubeBaseK && copyA)) {
            LoadData(b2, b1, bParams);
        } else {
            LoadData<float, kKeepLoad3d>(b2, b1, bParams);
        }
        bConfigReady_ = true;
    }

    __aicore__ inline void LoadGroup(
        LocalTensor<float>& aDest, LocalTensor<float>& bDest, LocalTensor<float>& aSource, LocalTensor<float>& bSource,
        uint32_t group)
    {
        constexpr uint32_t groupK = kCubeGroupK;
        LoadData2DParams aBatch{};
        aBatch.startIndex = group * (groupK / kCtrmmCubeBaseK) * (kCtrmmCubeBaseM / kCubeFractalRows);
        aBatch.repeatTimes = (groupK / kCtrmmCubeBaseK) * (kCtrmmCubeBaseM / kCubeFractalRows);
        aBatch.srcStride = 1;
        aBatch.dstGap = 0;
        aBatch.ifTranspose = false;
        LoadData(aDest, aSource, aBatch);
        LoadToL0(aSource, bSource, aDest, bDest, group * groupK, false, groupK);
    }

    __aicore__ inline void Compute(
        LocalTensor<float>& c1, LocalTensor<float>& a2, LocalTensor<float>& b2, bool initialize, uint32_t logicalK = 2U)
    {
        MmadParams params = {};
        params.m = mTile_;
        params.n = 2U * kCtrmmCubeBaseN;
        params.k = logicalK;
        params.cmatrixInitVal = initialize;
        params.cmatrixSource = false;
        params.kDirectionAlign = false;
        Mmad(c1, a2, b2, params);
    }

    __aicore__ inline void StoreToGm(
        LocalTensor<float>& c1, GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        FixpipeParamsV220 params = {};
        params.mSize = mTile_;
        params.nSize = kCtrmmCubeBaseN;
        params.srcStride = mTile_;
        params.dstStride = n_;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        Fixpipe(output[static_cast<uint64_t>(mBase) * n_ + nBase], c1, params);
    }

    LocalTensor<float> ar1_{TPosition::A1, 0U, aSizeAlignL1};
    LocalTensor<float> br1_{TPosition::B1, aSizeAlignL1 * kFloatBytes, bSizeAlignL1};
    LocalTensor<float> bi1_{TPosition::B1, (aSizeAlignL1 + bSizeAlignL1) * kFloatBytes, bSizeAlignL1};
    LocalTensor<float> ar2_{TPosition::A2, 0U, aSizeAlignL0};
    LocalTensor<float> br2_{TPosition::B2, 0U, 2U * bSizeAlignL0};
    LocalTensor<float> accumulator_{TPosition::CO1, 0U, 2U * cSizeAlignL0};
    LocalTensor<float> zeroB0_{TPosition::B1, aSizeAlignL1 * kFloatBytes, 2U * bSizeAlignL1};
    LocalTensor<float> zeroB1_{TPosition::B1, (2U * aSizeAlignL1 + 2U * bSizeAlignL1) * kFloatBytes, 2U * bSizeAlignL1};
    InitConstValueParams<float> clearB_{};
    bool upper_ = false;
    bool descending_ = false;
    bool bConfigReady_ = false;
    GlobalTensor<float> diagonal_;
    GlobalTensor<float> ar_;
    GlobalTensor<float> br_;
    GlobalTensor<float> bi_;
    GlobalTensor<float> rr_;
    GlobalTensor<float> ri_;
    uint32_t m_;
    uint32_t mTile_ = 0;
    uint32_t componentStride_ = kExpandedComponents;
    uint32_t n_;
    uint32_t k_;
};

template <uint32_t N>
__aicore__ inline void RunCube(GM_ADDR ws, const CtrmmTiling& t)
{
    auto* w = reinterpret_cast<__gm__ float*>(ws);
    const uint64_t base = kCubeAPlanes * t.aPlane + kCubeOutputXPlane * t.xPlane;
    CtrmmCube<N> cube;
    auto* input = w + kOriginalAPlanes * t.aPlane + 2U * t.xPlane;
    cube.Init(
        input, input + 8U * t.aPlane, input + 8U * t.aPlane + 2U * t.xPlane, w + base, w + base + t.xPlane,
        (t.q + kCubeFractalRows - 1U) / kCubeFractalRows * kCubeFractalRows, t.rp,
        (t.qp <= kCompactMaxOrder ? kCompactComponents : kExpandedComponents) * t.qp, t.upper != 0,
        !t.upper && ((!t.right && t.originalTrans == ACLBLAS_OP_N) || (t.right && t.originalTrans != ACLBLAS_OP_N)),
        w + kCubeAPlanes * t.aPlane + kCubeXPlanes * t.xPlane);
    cube.PrepareCo1();
    cube.Process(GetBlockIdx(), GetBlockNum());
}

} // namespace
