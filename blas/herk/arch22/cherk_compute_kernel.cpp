/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/cube_load_params_arch22.h"
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_utils.h"
#include "common/compute/simd.h"
#include "cherk_kernel.h"
#include "common/helper/triangle_store_arch22.h"

namespace {
constexpr uint32_t CUBE_M = 256U;
constexpr uint32_t CUBE_N = 128U;
constexpr uint32_t CUBE_K = 64U;
constexpr uint32_t CUBE_PANEL_K = 256U;
__aicore__ inline uint32_t CherkNormalizeBlockCount(uint32_t blocks) { return blocks == 0U ? 1U : blocks; }

__aicore__ inline uint64_t WorkspaceCount(const CherkArch22TilingData& tiling)
{
    return CherkArch22WorkspaceFloats(tiling.cubeN, tiling.cubeK);
}

} // namespace

// General planar Cube matrix-product device class.
template <uint32_t TILE_M>
class CherkArch22Mmad {
public:
    __aicore__ inline void Init(
        __gm__ float* realInput, __gm__ float* imagInput, __gm__ float* realOutput, __gm__ float* crossOutput,
        uint32_t n, uint32_t k, bool transA, bool transB, uint32_t uplo, uint32_t validK = 0U, uint32_t validN = 0U)
    {
        const uint64_t inputCount = static_cast<uint64_t>(n) * k;
        const uint64_t outputCount = static_cast<uint64_t>(n) * n;
        realInput_.SetGlobalBuffer(realInput, inputCount);
        imagInput_.SetGlobalBuffer(imagInput, inputCount);
        realOutput_.SetGlobalBuffer(realOutput, outputCount);
        crossOutput_.SetGlobalBuffer(crossOutput, outputCount);
        n_ = n;
        validN_ = validN == 0U ? n : validN;
        k_ = k;
        computeK_ = k == CUBE_K && validK != 0U ? (validK + 15U) / 16U * 16U : CUBE_K;
        transA_ = transA;
        transB_ = transB;
        uplo_ = uplo;
        InitCopyAndLoadParams();
    }

    __aicore__ inline void SetDirectImaginary(__gm__ float* negativeImag)
    {
        negativeImagInput_.SetGlobalBuffer(negativeImag, uint64_t(n_) * k_);
        directImaginary_ = true;
    }

    struct TileBuffers {
        AscendC::LocalTensor<float> a1;
        AscendC::LocalTensor<float> b1;
        AscendC::LocalTensor<float> a2;
        AscendC::LocalTensor<float> b2;
        AscendC::LocalTensor<float> c1;
    };
    __aicore__ __attribute__((always_inline)) inline void Process(uint32_t firstTile, uint32_t tileStride)
    {
        tileStride = tileStride == 0U ? 1U : tileStride;
        constexpr uint32_t aCount = TILE_M * CUBE_PANEL_K;
        constexpr uint32_t bCount = CUBE_PANEL_K * CUBE_N;
        constexpr uint32_t aL0Count = TILE_M * CUBE_K;
        constexpr uint32_t bL0Count = CUBE_K * CUBE_N;
        constexpr uint32_t cCount = TILE_M * CUBE_N;
        AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, 0U, aCount);
        AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, aCount * sizeof(float), bCount);
        AscendC::LocalTensor<float> a2(AscendC::TPosition::A2, 0U, aL0Count);
        AscendC::LocalTensor<float> b2(AscendC::TPosition::B2, 0U, bL0Count);
        AscendC::LocalTensor<float> c1(AscendC::TPosition::CO1, 0U, cCount);
        const uint32_t mTiles = (validN_ + TILE_M - 1U) / TILE_M;
        const uint32_t nTiles = (validN_ + CUBE_N - 1U) / CUBE_N;
        const uint64_t tileCount = static_cast<uint64_t>(mTiles) * nTiles;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
        // Real tiles cost two MMAD products while cross tiles cost one. The
        // former tile-owned schedule assigned either three products or one to
        // a core and produced severe triangular load skew. Publish the two
        // planes as independent job domains instead. Real jobs begin on the
        // cores that receive one fewer cross tile, making both domains
        // complementary without changing either output layout.
        TileBuffers b{a1, b1, a2, b2, c1};
        if (directImaginary_) {
            RunDirectJobs(b, firstTile, tileStride, tileCount, nTiles);
        } else {
            RunPlaneJobs(b, firstTile, tileStride, tileCount, nTiles);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    }

private:
    __aicore__ inline void RunDirectJobs(
        TileBuffers& b, uint32_t firstTile, uint32_t tileStride, uint64_t tileCount, uint32_t nTiles)
    {
        uint32_t job = 0U;
        for (uint64_t tile = 0U; tile < tileCount; ++tile) {
            const uint32_t mBase = static_cast<uint32_t>(tile / nTiles) * TILE_M;
            const uint32_t nBase = static_cast<uint32_t>(tile % nTiles) * CUBE_N;
            const bool selected = uplo_ == ACLBLAS_UPPER ? mBase + TILE_M > nBase : mBase < nBase + CUBE_N;
            if (!selected) {
                continue;
            }
            const uint32_t owner = (job * 2U) % tileStride;
            if (owner == firstTile) {
                ConfigureTile(mBase, nBase);
                ProductPair(
                    b.a1, b.b1, b.a2, b.b2, b.c1, realInput_, realInput_, imagInput_, imagInput_, realOutput_, mBase,
                    nBase);
            }
            if ((owner + 1U) % tileStride == firstTile) {
                ConfigureTile(mBase, nBase);
                // Stored [col,row] is the desired OP_C imaginary output:
                // I^T R - R^T I. Both terms share the same C1 owner.
                ProductPair(
                    b.a1, b.b1, b.a2, b.b2, b.c1, imagInput_, realInput_, realInput_, negativeImagInput_, crossOutput_,
                    mBase, nBase);
            }
            ++job;
        }
    }
    __aicore__ inline void RunPlaneJobs(
        TileBuffers& b, uint32_t firstTile, uint32_t tileStride, uint64_t tileCount, uint32_t nTiles)
    {
        const uint32_t crossRemainder = static_cast<uint32_t>(tileCount % tileStride);
        uint32_t realJob = 0U;
        for (uint64_t tile = 0U; tile < tileCount; ++tile) {
            const uint32_t mBase = static_cast<uint32_t>(tile / nTiles) * TILE_M;
            const uint32_t nBase = static_cast<uint32_t>(tile % nTiles) * CUBE_N;

            // The consumer reads the symmetric result through [column,row].
            // Produce only the physical half that maps to the selected C half.
            const bool selectedReal = uplo_ == ACLBLAS_UPPER ? mBase + TILE_M > nBase : mBase < nBase + CUBE_N;
            if (selectedReal) {
                const uint32_t owner = (realJob + crossRemainder) % tileStride;
                if (owner == firstTile) {
                    ConfigureTile(mBase, nBase);
                    ProductPair(
                        b.a1, b.b1, b.a2, b.b2, b.c1, realInput_, realInput_, imagInput_, imagInput_, realOutput_,
                        mBase, nBase);
                }
                ++realJob;
            }
        }

        for (uint64_t tile = firstTile; tile < tileCount; tile += tileStride) {
            const uint32_t mBase = static_cast<uint32_t>(tile / nTiles) * TILE_M;
            const uint32_t nBase = static_cast<uint32_t>(tile % nTiles) * CUBE_N;
            // Q = R^T I. Its transpose is consumed from the mirrored tile.
            ConfigureTile(mBase, nBase);
            ProductSingle(b.a1, b.b1, b.a2, b.b2, b.c1, realInput_, imagInput_, crossOutput_, mBase, nBase);
        }
    }

    __aicore__ inline void ConfigureTile(uint32_t mBase, uint32_t nBase)
    {
        const uint32_t rows = validN_ - mBase < TILE_M ? validN_ - mBase : TILE_M;
        const uint32_t cols = validN_ - nBase < CUBE_N ? validN_ - nBase : CUBE_N;
        tileM_ = (rows + 15U) / 16U * 16U;
        tileN_ = (cols + 15U) / 16U * 16U;
        aLoadParams_.kExtension = tileM_;
        bLoadParams_.kExtension = tileN_;
        aPanelLoadParams_.kExtension = tileM_;
        bPanelLoadParams_.kExtension = tileN_;
        // Keep NZ pitches padded, but copy only channels consumed by LoadData.
        if (transA_) {
            aCopyParams_.dValue = tileM_;
            aPanelCopyParams_.dValue = tileM_;
            aCopyParams_.nValue = k_ == CUBE_K ? computeK_ : CUBE_K;
        } else {
            aCopyParams_.nValue = tileM_;
            aPanelCopyParams_.nValue = tileM_;
        }
        if (!transB_) {
            bCopyParams_.dValue = tileN_;
            bPanelCopyParams_.dValue = tileN_;
            bCopyParams_.nValue = k_ == CUBE_K ? computeK_ : CUBE_K;
        } else {
            bCopyParams_.nValue = tileN_;
            bPanelCopyParams_.nValue = tileN_;
        }
    }

    __aicore__ inline void InitCopyAndLoadParams()
    {
        InitCopyParams();
        InitLoadParams();
        InitPanelParams();
    }

    __aicore__ inline void InitCopyParams()
    {
        aCopyParams_ = {};
        aCopyParams_.ndNum = 1;
        aCopyParams_.nValue = transA_ ? CUBE_K : TILE_M;
        aCopyParams_.dValue = transA_ ? TILE_M : CUBE_K;
        aCopyParams_.srcDValue = transA_ ? n_ : k_;
        aCopyParams_.dstNzC0Stride = transA_ ? CUBE_K : TILE_M;
        aCopyParams_.dstNzNStride = 1;

        bCopyParams_ = {};
        bCopyParams_.ndNum = 1;
        bCopyParams_.nValue = transB_ ? CUBE_N : CUBE_K;
        bCopyParams_.dValue = transB_ ? CUBE_K : CUBE_N;
        bCopyParams_.srcDValue = transB_ ? k_ : n_;
        bCopyParams_.dstNzC0Stride = transB_ ? CUBE_N : CUBE_K;
        bCopyParams_.dstNzNStride = 1;
    }

    __aicore__ inline void InitLoadParams()
    {
        aLoadParams_ = {};
        aLoadParams_.l1H = 1U;
        aLoadParams_.l1W = transA_ ? CUBE_K : TILE_M;
        aLoadParams_.channelSize = transA_ ? TILE_M : CUBE_K;
        aLoadParams_.kExtension = transA_ ? TILE_M : CUBE_K;
        aLoadParams_.mExtension = transA_ ? CUBE_K : TILE_M;
        cube_load::InitUnitFilter(aLoadParams_);
        aLoadParams_.enTranspose = transA_;
        aLoadParams_.fMatrixCtrl = false;

        bLoadParams_ = {};
        bLoadParams_.l1H = 1U;
        bLoadParams_.l1W = transB_ ? CUBE_N : CUBE_K;
        bLoadParams_.channelSize = transB_ ? CUBE_K : CUBE_N;
        bLoadParams_.kExtension = transB_ ? CUBE_K : CUBE_N;
        bLoadParams_.mExtension = transB_ ? CUBE_N : CUBE_K;
        cube_load::InitUnitFilter(bLoadParams_);
        // L0B consumes its matrix orientation through the B-side path; this
        // matches the proven CHER2K/CHEMM arch22 contract for both layouts.
        bLoadParams_.enTranspose = true;
        bLoadParams_.fMatrixCtrl = false;
        if (transA_ && !transB_ && k_ == CUBE_K) {
            aLoadParams_.mExtension = computeK_;
            bLoadParams_.mExtension = computeK_;
        }
    }

    __aicore__ inline void InitPanelParams()
    {
        aPanelCopyParams_ = aCopyParams_;
        aPanelCopyParams_.nValue = transA_ ? CUBE_PANEL_K : TILE_M;
        aPanelCopyParams_.dValue = transA_ ? TILE_M : CUBE_PANEL_K;
        aPanelCopyParams_.dstNzC0Stride = transA_ ? CUBE_PANEL_K : TILE_M;
        bPanelCopyParams_ = bCopyParams_;
        bPanelCopyParams_.nValue = transB_ ? CUBE_N : CUBE_PANEL_K;
        bPanelCopyParams_.dValue = transB_ ? CUBE_PANEL_K : CUBE_N;
        bPanelCopyParams_.dstNzC0Stride = transB_ ? CUBE_N : CUBE_PANEL_K;

        aPanelLoadParams_ = aLoadParams_;
        aPanelLoadParams_.l1W = transA_ ? CUBE_PANEL_K : TILE_M;
        aPanelLoadParams_.channelSize = transA_ ? TILE_M : CUBE_PANEL_K;
        bPanelLoadParams_ = bLoadParams_;
        bPanelLoadParams_.l1W = transB_ ? CUBE_N : CUBE_PANEL_K;
    }

    __aicore__ inline void CopyToL1(
        AscendC::LocalTensor<float>& a1, AscendC::LocalTensor<float>& b1, AscendC::GlobalTensor<float>& left,
        AscendC::GlobalTensor<float>& right, uint32_t mBase, uint32_t nBase, uint32_t kBase)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        const uint64_t aOffset =
            transA_ ? static_cast<uint64_t>(kBase) * n_ + mBase : static_cast<uint64_t>(mBase) * k_ + kBase;
        const uint64_t bOffset =
            transB_ ? static_cast<uint64_t>(nBase) * k_ + kBase : static_cast<uint64_t>(kBase) * n_ + nBase;
        AscendC::DataCopy(a1, left[aOffset], aCopyParams_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::DataCopy(b1, right[bOffset], bCopyParams_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
    }

    __aicore__ inline void LoadToL0(
        AscendC::LocalTensor<float>& a1, AscendC::LocalTensor<float>& b1, AscendC::LocalTensor<float>& a2,
        AscendC::LocalTensor<float>& b2)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::LoadData(a2, a1, aLoadParams_);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        AscendC::LoadData(b2, b1, bLoadParams_);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    }

    __aicore__ inline void Compute(
        AscendC::LocalTensor<float>& c1, AscendC::LocalTensor<float>& a2, AscendC::LocalTensor<float>& b2,
        bool initialize)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::MmadParams params{};
        params.m = tileM_;
        params.n = tileN_;
        params.k = computeK_;
        params.cmatrixInitVal = initialize;
        params.cmatrixSource = false;
        params.kDirectionAlign = false;
        AscendC::Mmad(c1, a2, b2, params);
        if ((tileM_ / 16U) * (tileN_ / 16U) < 10U) {
            AscendC::PipeBarrier<PIPE_M>();
        }
        // M_MTE1 releases the L0 operands after MMAD; M_FIX protects C1.
        // L1 prefetch can proceed independently on MTE2.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void CopyPanelToL1(
        AscendC::LocalTensor<float>& a1, AscendC::LocalTensor<float>& b1, AscendC::GlobalTensor<float>& left,
        AscendC::GlobalTensor<float>& right, uint32_t mBase, uint32_t nBase, uint32_t panelBase)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        const uint64_t aOffset =
            transA_ ? static_cast<uint64_t>(panelBase) * n_ + mBase : static_cast<uint64_t>(mBase) * k_ + panelBase;
        const uint64_t bOffset =
            transB_ ? static_cast<uint64_t>(nBase) * k_ + panelBase : static_cast<uint64_t>(panelBase) * n_ + nBase;
        AscendC::DataCopy(a1, left[aOffset], aPanelCopyParams_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::DataCopy(b1, right[bOffset], bPanelCopyParams_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
    }

    __aicore__ inline void LoadPanelSliceToL0(
        AscendC::LocalTensor<float>& a1, AscendC::LocalTensor<float>& b1, AscendC::LocalTensor<float>& a2,
        AscendC::LocalTensor<float>& b2, uint32_t slice, bool firstSlice, bool lastSlice)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        if (firstSlice) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        }
        const uint32_t panelK = slice * CUBE_K;
        aPanelLoadParams_.mStartPt = transA_ ? panelK : 0U;
        aPanelLoadParams_.kStartPt = transA_ ? 0U : panelK;
        bPanelLoadParams_.mStartPt = transB_ ? 0U : panelK;
        bPanelLoadParams_.kStartPt = transB_ ? panelK : 0U;
        AscendC::LoadData(a2, a1, aPanelLoadParams_);
        AscendC::LoadData(b2, b1, bPanelLoadParams_);
        if (lastSlice) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    }

    __aicore__ inline void AccumulateTerm(
        AscendC::LocalTensor<float>& a1, AscendC::LocalTensor<float>& b1, AscendC::LocalTensor<float>& a2,
        AscendC::LocalTensor<float>& b2, AscendC::LocalTensor<float>& c1, AscendC::GlobalTensor<float>& left,
        AscendC::GlobalTensor<float>& right, uint32_t mBase, uint32_t nBase, bool initialize, uint32_t begin = 0U,
        uint32_t end = 0U)
    {
        const uint32_t limit = end == 0U ? k_ : end;
        uint32_t kBase = begin;
        constexpr uint32_t slicesPerPanel = CUBE_PANEL_K / CUBE_K;
        for (; kBase + CUBE_PANEL_K <= limit; kBase += CUBE_PANEL_K) {
            CopyPanelToL1(a1, b1, left, right, mBase, nBase, kBase);
            for (uint32_t slice = 0U; slice < slicesPerPanel; ++slice) {
                LoadPanelSliceToL0(a1, b1, a2, b2, slice, slice == 0U, slice + 1U == slicesPerPanel);
                Compute(c1, a2, b2, initialize && kBase == begin && slice == 0U);
            }
        }
        for (; kBase < limit; kBase += CUBE_K) {
            CopyToL1(a1, b1, left, right, mBase, nBase, kBase);
            LoadToL0(a1, b1, a2, b2);
            Compute(c1, a2, b2, initialize && kBase == begin);
        }
    }

    __aicore__ inline void Store(
        AscendC::LocalTensor<float>& c1, AscendC::GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::FixpipeParamsV220 params{};
        params.mSize = tileM_;
        params.nSize = tileN_;
        params.srcStride = tileM_;
        params.dstStride = n_;
        params.ndNum = 1;
        AscendC::Fixpipe(output[static_cast<uint64_t>(mBase) * n_ + nBase], c1, params);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void ProductPair(
        AscendC::LocalTensor<float>& a1, AscendC::LocalTensor<float>& b1, AscendC::LocalTensor<float>& a2,
        AscendC::LocalTensor<float>& b2, AscendC::LocalTensor<float>& c1, AscendC::GlobalTensor<float>& left0,
        AscendC::GlobalTensor<float>& right0, AscendC::GlobalTensor<float>& left1, AscendC::GlobalTensor<float>& right1,
        AscendC::GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        // Accumulate bounded K panels in the owned workspace. Atomic mode
        // is used only for this private product plane, never for user C.
        for (uint32_t begin = 0U; begin < k_; begin += CUBE_PANEL_K) {
            const uint32_t end = begin + CUBE_PANEL_K < k_ ? begin + CUBE_PANEL_K : k_;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AccumulateTerm(a1, b1, a2, b2, c1, left0, right0, mBase, nBase, true, begin, end);
            AccumulateTerm(a1, b1, a2, b2, c1, left1, right1, mBase, nBase, false, begin, end);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            if (begin != 0U) {
                AscendC::SetAtomicAdd<float>();
            }
            Store(c1, output, mBase, nBase);
            AscendC::SetAtomicNone();
        }
    }

    __aicore__ inline void ProductSingle(
        AscendC::LocalTensor<float>& a1, AscendC::LocalTensor<float>& b1, AscendC::LocalTensor<float>& a2,
        AscendC::LocalTensor<float>& b2, AscendC::LocalTensor<float>& c1, AscendC::GlobalTensor<float>& left,
        AscendC::GlobalTensor<float>& right, AscendC::GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        for (uint32_t begin = 0U; begin < k_; begin += CUBE_PANEL_K) {
            const uint32_t end = begin + CUBE_PANEL_K < k_ ? begin + CUBE_PANEL_K : k_;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AccumulateTerm(a1, b1, a2, b2, c1, left, right, mBase, nBase, true, begin, end);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            if (begin != 0U) {
                AscendC::SetAtomicAdd<float>();
            }
            Store(c1, output, mBase, nBase);
            AscendC::SetAtomicNone();
        }
    }

    AscendC::GlobalTensor<float> realInput_;
    AscendC::GlobalTensor<float> imagInput_;
    AscendC::GlobalTensor<float> negativeImagInput_;
    bool directImaginary_ = false;
    AscendC::GlobalTensor<float> realOutput_;
    AscendC::GlobalTensor<float> crossOutput_;
    uint32_t n_ = 0U;
    uint32_t validN_ = 0U;
    uint32_t tileM_ = TILE_M;
    uint32_t tileN_ = CUBE_N;
    uint32_t k_ = 0U;
    uint32_t computeK_ = CUBE_K;
    bool transA_ = false;
    bool transB_ = false;
    uint32_t uplo_ = ACLBLAS_UPPER;
    AscendC::Nd2NzParams aCopyParams_;
    AscendC::Nd2NzParams bCopyParams_;
    AscendC::Nd2NzParams aPanelCopyParams_;
    AscendC::Nd2NzParams bPanelCopyParams_;
    AscendC::LoadData3DParamsV2<float> aLoadParams_;
    AscendC::LoadData3DParamsV2<float> bLoadParams_;
    AscendC::LoadData3DParamsV2<float> aPanelLoadParams_;
    AscendC::LoadData3DParamsV2<float> bPanelLoadParams_;
};

// General product scaling and triangular output device helpers.
namespace {
constexpr uint32_t EPILOGUE_TILE = 64U;
constexpr uint32_t EPILOGUE_TILE_ELEMENTS = EPILOGUE_TILE * EPILOGUE_TILE;
constexpr uint32_t EPILOGUE_AOS_ELEMENTS = EPILOGUE_TILE_ELEMENTS * 2U;
} // namespace

struct CherkEpilogueBuffers {
    AscendC::LocalTensor<float> real;
    AscendC::LocalTensor<float> imag;
    AscendC::LocalTensor<float> crossRows;
    AscendC::LocalTensor<float> crossTransposed;
    AscendC::LocalTensor<float> aos;
    AscendC::LocalTensor<uint32_t> transposeOffsets;
    AscendC::LocalTensor<uint32_t> interleaveOffsets;
    AscendC::LocalTensor<uint8_t> diagonalMask;
};

__aicore__ inline void LoadEpilogueMetadata(GM_ADDR metadataGm, CherkEpilogueBuffers& b)
{
    AscendC::GlobalTensor<uint32_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
    AscendC::GlobalTensor<uint8_t> metadataBytes;
    metadataBytes.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(metadataGm), CHERK_ARCH22_META_WORDS * sizeof(uint32_t));
    AscendC::DataCopy(
        b.transposeOffsets, metadata[CHERK_ARCH22_META_TRANSPOSE_OFFSET], CHERK_ARCH22_META_TRANSPOSE_COUNT);
    AscendC::DataCopy(b.interleaveOffsets, metadata[CHERK_ARCH22_META_INTERLEAVE_OFFSET], EPILOGUE_AOS_ELEMENTS);
    AscendC::DataCopy(
        b.diagonalMask, metadataBytes[CHERK_ARCH22_META_DIAGONAL_OFFSET * sizeof(uint32_t)],
        EPILOGUE_TILE_ELEMENTS / 8U);
    const event_t metadataToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
}

struct CherkEpilogueStorage {
    AscendC::TBuf<AscendC::TPosition::VECCALC> planarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transposeOffsetBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> interleaveOffsetBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diagonalMaskBuf;
    CherkEpilogueBuffers tensors;

    __aicore__ inline void Init(AscendC::TPipe& pipe, GM_ADDR metadataGm)
    {
        pipe.InitBuffer(planarBuf, EPILOGUE_TILE_ELEMENTS * 4U * sizeof(float));
        pipe.InitBuffer(aosBuf, EPILOGUE_AOS_ELEMENTS * sizeof(float));
        pipe.InitBuffer(transposeOffsetBuf, EPILOGUE_TILE_ELEMENTS * sizeof(uint32_t));
        pipe.InitBuffer(interleaveOffsetBuf, EPILOGUE_AOS_ELEMENTS * sizeof(uint32_t));
        pipe.InitBuffer(diagonalMaskBuf, EPILOGUE_TILE_ELEMENTS / 8U);
        AscendC::LocalTensor<float> planar = planarBuf.Get<float>();
        tensors = {
            planar,
            planar[EPILOGUE_TILE_ELEMENTS],
            planar[EPILOGUE_TILE_ELEMENTS * 2U],
            planar[EPILOGUE_TILE_ELEMENTS * 3U],
            aosBuf.Get<float>(),
            transposeOffsetBuf.Get<uint32_t>(),
            interleaveOffsetBuf.Get<uint32_t>(),
            diagonalMaskBuf.Get<uint8_t>()};
        LoadEpilogueMetadata(metadataGm, tensors);
    }
};

__aicore__ inline void LoadCrossProductRows(
    AscendC::GlobalTensor<float>& workspace, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, CherkEpilogueBuffers& b)
{
    const uint32_t rowBytes = validCols * sizeof(float);
    const uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams rowCopy{
        static_cast<uint16_t>(validRows), rowBytes, static_cast<uint32_t>((tiling.cubeN - validCols) * sizeof(float)),
        static_cast<uint32_t>((EPILOGUE_TILE * sizeof(float) - alignedRowBytes) / 32U), 0U};
    const AscendC::DataCopyPadExtParams<float> rowPad{
        alignedRowBytes != rowBytes, 0U, static_cast<uint8_t>((alignedRowBytes - rowBytes) / sizeof(float)), 0.0f};
    const uint64_t rowOffset = static_cast<uint64_t>(rowBase) * tiling.cubeN + colBase;
    AscendC::DataCopyPad(
        b.crossRows, workspace[CherkArch22OffsetCrossOutput(tiling.cubeN, tiling.cubeK) + rowOffset], rowCopy, rowPad);
}

__aicore__ inline void FinalizeProductTile(
    const CherkArch22TilingData& tiling, float alpha, bool diagonalTile, CherkEpilogueBuffers& b)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Gather(
        b.crossTransposed, diagonalTile ? b.imag : b.crossRows, b.transposeOffsets, 0U, EPILOGUE_TILE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    if (tiling.trans == ACLBLAS_OP_N) {
        // H = A*A^H: imag(H) = Q^T - Q.
        AscendC::Sub(b.imag, b.imag, b.crossTransposed, EPILOGUE_TILE_ELEMENTS);
    } else {
        // H = A^H*A: imag(H) = Q - Q^T.
        AscendC::Sub(b.imag, b.crossTransposed, b.imag, EPILOGUE_TILE_ELEMENTS);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(b.real, b.real, alpha, EPILOGUE_TILE_ELEMENTS);
    AscendC::Muls(b.imag, b.imag, alpha, EPILOGUE_TILE_ELEMENTS);
}

__aicore__ inline void LoadProductTile(
    AscendC::GlobalTensor<float>& workspace, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, float alpha, bool diagonalTile, CherkEpilogueBuffers& b)
{
    AscendC::Duplicate(b.real, 0.0f, EPILOGUE_TILE_ELEMENTS);
    AscendC::Duplicate(b.imag, 0.0f, EPILOGUE_TILE_ELEMENTS);
    AscendC::Duplicate(b.crossRows, 0.0f, EPILOGUE_TILE_ELEMENTS);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

    const uint32_t columnBytes = validRows * sizeof(float);
    const uint32_t alignedColumnBytes = (columnBytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams columnCopy{
        static_cast<uint16_t>(validCols), columnBytes,
        static_cast<uint32_t>((tiling.cubeN - validRows) * sizeof(float)),
        static_cast<uint32_t>((EPILOGUE_TILE * sizeof(float) - alignedColumnBytes) / 32U), 0U};
    const AscendC::DataCopyPadExtParams<float> columnPad{
        alignedColumnBytes != columnBytes, 0U, static_cast<uint8_t>((alignedColumnBytes - columnBytes) / sizeof(float)),
        0.0f};
    const uint64_t transposedOffset = static_cast<uint64_t>(colBase) * tiling.cubeN + rowBase;
    AscendC::DataCopyPad(
        b.real, workspace[CherkArch22OffsetRealOutput(tiling.cubeN, tiling.cubeK) + transposedOffset], columnCopy,
        columnPad);
    AscendC::DataCopyPad(
        b.imag, workspace[CherkArch22OffsetCrossOutput(tiling.cubeN, tiling.cubeK) + transposedOffset], columnCopy,
        columnPad);

    if (!diagonalTile) {
        LoadCrossProductRows(workspace, tiling, rowBase, colBase, validRows, validCols, b);
    }
    FinalizeProductTile(tiling, alpha, diagonalTile, b);
}

__aicore__ inline void LoadAndApplyBeta(
    AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, float beta, CherkEpilogueBuffers& b)
{
    const uint32_t blockBytes = validRows * 2U * sizeof(float);
    const uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(validCols), blockBytes,
        static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)),
        static_cast<uint32_t>((EPILOGUE_TILE * 2U * sizeof(float) - alignedBlockBytes) / 32U), 0U};
    const AscendC::DataCopyPadExtParams<float> pad{
        alignedBlockBytes != blockBytes, 0U, static_cast<uint8_t>((alignedBlockBytes - blockBytes) / sizeof(float)),
        0.0f};
    const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
    if (rowBase == colBase) {
        triangle_store::LoadComplexSelected<EPILOGUE_TILE>(
            b.aos, c, tiling.ldc, rowBase, colBase, validRows, validCols, tiling.uplo == ACLBLAS_UPPER, EVENT_ID0);
    } else {
        AscendC::DataCopyPad(b.aos, c[cOffset], copy, pad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = EPILOGUE_AOS_ELEMENTS / 64U;
    AscendC::GatherMask<float>(b.crossRows, b.aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(b.crossTransposed, b.aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(b.crossRows, b.crossRows, beta, EPILOGUE_TILE_ELEMENTS);
    AscendC::Muls(b.crossTransposed, b.crossTransposed, beta, EPILOGUE_TILE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(b.real, b.real, b.crossRows, EPILOGUE_TILE_ELEMENTS);
    AscendC::Add(b.imag, b.imag, b.crossTransposed, EPILOGUE_TILE_ELEMENTS);
}

__aicore__ inline void ZeroDiagonal(
    uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols, CherkEpilogueBuffers& b)
{
    if (rowBase != colBase || validRows == 0U || validCols == 0U) {
        return;
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(b.crossRows, 0.0f, EPILOGUE_TILE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select(
        b.imag, b.diagonalMask, b.crossRows, b.imag, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, EPILOGUE_TILE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void InterleaveTile(
    const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols,
    CherkEpilogueBuffers& b)
{
    const bool fullTile =
        tiling.uplo == ACLBLAS_UPPER ? colBase >= rowBase + validRows : rowBase >= colBase + validCols;
    if (fullTile) {
        AscendC::Gather(b.aos, b.real, b.interleaveOffsets, 0U, EPILOGUE_AOS_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    for (uint32_t column = 0U; column < validCols; ++column) {
        const uint32_t absoluteColumn = colBase + column;
        const uint32_t firstRow = upper || absoluteColumn <= rowBase ? 0U : absoluteColumn - rowBase;
        const uint32_t count = upper ? Min<uint32_t>(validRows, absoluteColumn - rowBase + 1U) : validRows - firstRow;
        AscendC::Gather(
            b.aos[column * EPILOGUE_TILE * 2U], b.real[column * EPILOGUE_TILE + firstRow], b.interleaveOffsets, 0U,
            count * 2U);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void StoreSelectedTile(
    AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, CherkEpilogueBuffers& b)
{
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    const bool fullTile = upper ? colBase >= rowBase + validRows : rowBase >= colBase + validCols;
    if (fullTile) {
        const uint32_t blockBytes = validRows * 2U * sizeof(float);
        const uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
        const AscendC::DataCopyExtParams copy{
            static_cast<uint16_t>(validCols), blockBytes,
            static_cast<uint32_t>((EPILOGUE_TILE * 2U * sizeof(float) - alignedBlockBytes) / 32U),
            static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)), 0U};
        const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
        AscendC::DataCopyPad(c[cOffset], b.aos, copy);
        return;
    }

    // Only diagonal tiles are partial. Per-column writes preserve the opposite
    // triangle without a read-modify-write of values that BLAS does not own.
    for (uint32_t column = 0U; column < validCols; ++column) {
        const uint32_t absoluteColumn = colBase + column;
        const uint32_t firstRow = upper ? 0U : absoluteColumn - rowBase;
        const uint32_t count = upper ? Min<uint32_t>(validRows, absoluteColumn - rowBase + 1U) : validRows - firstRow;
        if (count == 0U) {
            continue;
        }
        const uint64_t cOffset = (static_cast<uint64_t>(absoluteColumn) * tiling.ldc + rowBase + firstRow) * 2ULL;
        const AscendC::DataCopyExtParams copy{1U, static_cast<uint32_t>(count * 2U * sizeof(float)), 0U, 0U, 0U};
        AscendC::DataCopyPad(c[cOffset], b.aos[column * EPILOGUE_TILE * 2U], copy);
    }
}

struct CherkEpilogueContext {
    float alpha;
    float beta;
    bool hasProduct;
    AscendC::GlobalTensor<float>& c;
    AscendC::GlobalTensor<float>& workspace;
    const CherkArch22TilingData& tiling;
    CherkEpilogueBuffers& buffers;
};

__aicore__ inline void ProcessEpilogueTile(CherkEpilogueContext& ctx, uint32_t rowBase, uint32_t colBase)
{
    const uint32_t validRows = Min<uint32_t>(EPILOGUE_TILE, ctx.tiling.n - rowBase);
    const uint32_t validCols = Min<uint32_t>(EPILOGUE_TILE, ctx.tiling.n - colBase);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    if (ctx.hasProduct) {
        LoadProductTile(
            ctx.workspace, ctx.tiling, rowBase, colBase, validRows, validCols, ctx.alpha, rowBase == colBase,
            ctx.buffers);
    } else {
        AscendC::Duplicate(ctx.buffers.real, 0.0f, EPILOGUE_TILE_ELEMENTS);
        AscendC::Duplicate(ctx.buffers.imag, 0.0f, EPILOGUE_TILE_ELEMENTS);
    }
    if (ctx.beta != 0.0f) {
        AscendC::PipeBarrier<PIPE_V>();
        LoadAndApplyBeta(ctx.c, ctx.tiling, rowBase, colBase, validRows, validCols, ctx.beta, ctx.buffers);
    }
    ZeroDiagonal(rowBase, colBase, validRows, validCols, ctx.buffers);
    InterleaveTile(ctx.tiling, rowBase, colBase, validRows, validCols, ctx.buffers);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    StoreSelectedTile(ctx.c, ctx.tiling, rowBase, colBase, validRows, validCols, ctx.buffers);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void RunEpilogueTasks(CherkEpilogueContext& ctx, uint32_t blockIdx, uint32_t blockNum)
{
    blockNum = blockNum == 0U ? 1U : blockNum;
    const uint32_t tileAxis = (ctx.tiling.n + EPILOGUE_TILE - 1U) / EPILOGUE_TILE;
    const uint64_t taskCount = static_cast<uint64_t>(tileAxis) * (tileAxis + 1U) / 2ULL;
    const uint64_t taskBegin = static_cast<uint64_t>(blockIdx) * taskCount / blockNum;
    const uint64_t taskEnd = static_cast<uint64_t>(blockIdx + 1U) * taskCount / blockNum;
    const bool upper = ctx.tiling.uplo == ACLBLAS_UPPER;
    CherkTriangleTaskCursor cursor = InitCherkTriangleCursor(taskBegin, tileAxis, upper);
    for (uint64_t task = taskBegin; task < taskEnd; ++task) {
        const uint32_t colTile = ResolveCherkTriangleCol(cursor, task);
        ProcessEpilogueTile(ctx, cursor.rowTile * EPILOGUE_TILE, colTile * EPILOGUE_TILE);
    }
}

extern "C" __global__ __aicore__ void cherk_arch22_mmad_kernel(GM_ADDR workspaceGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    uint32_t blockNum = AscendC::GetBlockNum();
    blockNum = CherkNormalizeBlockCount(blockNum);
    __gm__ float* base = reinterpret_cast<__gm__ float*>(workspaceGm);
    __gm__ float* realInput = base + CherkArch22OffsetRealInput(tiling.cubeN, tiling.cubeK);
    __gm__ float* imagInput = base + CherkArch22OffsetImagInput(tiling.cubeN, tiling.cubeK);
    __gm__ float* realOutput = base + CherkArch22OffsetRealOutput(tiling.cubeN, tiling.cubeK);
    __gm__ float* crossOutput = base + CherkArch22OffsetCrossOutput(tiling.cubeN, tiling.cubeK);
    const bool directImag = CherkArch22DirectImaginary(tiling);
    if (tiling.compactPath != 0U || directImag) {
        CherkArch22Mmad<128U> mm;
        mm.Init(
            realInput, imagInput, realOutput, crossOutput, tiling.cubeN, tiling.cubeK, true, false, tiling.uplo,
            tiling.k, tiling.n);
        if (directImag) {
            mm.SetDirectImaginary(base + CherkArch22OffsetNegativeImagInput(tiling.cubeN, tiling.cubeK));
        }
        mm.Process(AscendC::GetBlockIdx(), blockNum);
    } else {
        CherkArch22Mmad<CUBE_M> mm;
        mm.Init(
            realInput, imagInput, realOutput, crossOutput, tiling.cubeN, tiling.cubeK, true, false, tiling.uplo,
            tiling.k, tiling.n);
        mm.Process(AscendC::GetBlockIdx(), blockNum);
    }
}
extern "C" __global__ __aicore__ void cherk_arch22_epilogue_kernel(
    GM_ADDR alphaGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    const bool hasProduct = tiling.skipProduct == 0U && alpha != 0.0f;
    if (!hasProduct && beta == 1.0f) {
        return;
    }

    AscendC::TPipe pipe;
    CherkEpilogueStorage storage;
    storage.Init(pipe, metadataGm);
    CherkEpilogueBuffers& b = storage.tensors;

    AscendC::GlobalTensor<float> c;
    AscendC::GlobalTensor<float> workspace;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2U);
    if (hasProduct) {
        workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm), WorkspaceCount(tiling));
    }

    uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    blockNum = CherkNormalizeBlockCount(blockNum);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    CherkEpilogueContext context{alpha, beta, hasProduct, c, workspace, tiling, b};
    RunEpilogueTasks(context, static_cast<uint32_t>(AscendC::GetBlockIdx()), blockNum);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
}
// A scaled long reduction can exceed the absolute error budget even when
// the Cube result is close to the exact dot product. Keep a reference-order
// path for amplified products: each rounded complex term is added in K order.
// Only beta=0 uses this path, so no old-C snapshot or restoration is needed.
struct ReferenceProductTile {
    static constexpr uint32_t Rows = 64U;
    static constexpr uint32_t Layers = 32U;
    static constexpr uint32_t CacheBase = Rows;
    static constexpr uint32_t CoeffBase = CacheBase + Rows * Layers;
    static_assert(CoeffBase + Layers * 8U <= 4096U);
    AscendC::LocalTensor<float> real;
    AscendC::LocalTensor<float> imag;
    AscendC::LocalTensor<float> scratch;
    AscendC::GlobalTensor<float>& workspace;
    const CherkArch22TilingData& t;
    float alpha;

    __aicore__ inline void LoadPlane(
        AscendC::LocalTensor<float> plane, uint64_t base, uint32_t col, uint32_t row, uint32_t valid, uint32_t layer,
        uint32_t count)
    {
        const uint32_t bytes = valid * sizeof(float);
        const uint32_t aligned = (bytes + 31U) & ~31U;
        const AscendC::DataCopyExtParams rows{
            static_cast<uint16_t>(count), bytes, (t.cubeN - valid) * 4U, (Rows * 4U - aligned) / 32U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{
            aligned != bytes, 0U, static_cast<uint8_t>((aligned - bytes) / 4U), 0.0f};
        const uint64_t start = base + uint64_t(layer) * t.cubeN;
        AscendC::DataCopyPad(plane[CacheBase], workspace[start + row], rows, pad);
        const AscendC::DataCopyExtParams columns{static_cast<uint16_t>(count), 4U, (t.cubeN - 1U) * 4U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> coefficientPad{true, 0U, 7U, 0.0f};
        AscendC::DataCopyPad(plane[CoeffBase], workspace[start + col], columns, coefficientPad);
    }
    __aicore__ inline void LoadBatch(uint32_t col, uint32_t row, uint32_t valid, uint32_t layer, uint32_t count)
    {
        LoadPlane(real, CherkArch22OffsetRealInput(t.cubeN, t.cubeK), col, row, valid, layer, count);
        LoadPlane(imag, CherkArch22OffsetImagInput(t.cubeN, t.cubeK), col, row, valid, layer, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }
    __aicore__ inline void AddLayer(uint32_t layer)
    {
        auto ar = real[CacheBase + layer * Rows];
        auto ai = imag[CacheBase + layer * Rows];
        auto rr = scratch;
        auto ii = scratch[Rows];
        auto ri = scratch[2U * Rows];
        auto ir = scratch[3U * Rows];
        const float cr = alpha * real.GetValue(CoeffBase + layer * 8U);
        const float ci = alpha * imag.GetValue(CoeffBase + layer * 8U);
        AscendC::Muls(rr, ar, cr, Rows);
        AscendC::Muls(ii, ai, ci, Rows);
        AscendC::Muls(ri, ai, cr, Rows);
        AscendC::Muls(ir, ar, ci, Rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(rr, rr, ii, Rows);
        AscendC::Sub(ri, ri, ir, Rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(real, real, rr, Rows);
        AscendC::Add(imag, imag, ri, Rows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Accumulate(uint32_t col, uint32_t row, uint32_t valid)
    {
        // Only the first Rows values hold the result. The remaining plane
        // capacity caches inputs; output gathers address only the result prefix.
        AscendC::Duplicate(real, 0.0f, Rows);
        AscendC::Duplicate(imag, 0.0f, Rows);
        for (uint32_t layer = 0U; layer < t.k; layer += Layers) {
            const uint32_t count = Min<uint32_t>(Layers, t.k - layer);
            LoadBatch(col, row, valid, layer, count);
            for (uint32_t local = 0U; local < count; ++local)
                AddLayer(local);
            // Release cached inputs only after every vector reader finishes.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
    }
};
__aicore__ inline void ReferenceOrderProduct(
    AscendC::LocalTensor<float> real, AscendC::LocalTensor<float> imag, AscendC::LocalTensor<float> scratch,
    AscendC::LocalTensor<float> output, AscendC::LocalTensor<uint32_t> interleave,
    AscendC::LocalTensor<uint32_t> indices, AscendC::GlobalTensor<float>& workspace, AscendC::GlobalTensor<float>& c,
    const CherkArch22TilingData& t, float alpha, uint32_t blocks)
{
    if (blocks == 0U) {
        return;
    }
    constexpr uint32_t rowsPerJob = 64U;
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t rowTiles = (t.n + rowsPerJob - 1U) / rowsPerJob;
    ReferenceProductTile tile{real, imag, scratch, workspace, t, alpha};
    uint32_t prefix = 0U;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    for (uint32_t col = 0U; col < t.n; ++col) {
        const bool upper = t.uplo == ACLBLAS_UPPER;
        const uint32_t first = upper ? 0U : col / rowsPerJob;
        const uint32_t end = upper ? col / rowsPerJob + 1U : rowTiles;
        const uint32_t jobs = end - first;
        for (uint32_t job = (block + blocks - prefix % blocks) % blocks; job < jobs; job += blocks) {
            const uint32_t row = (first + job) * rowsPerJob;
            const uint32_t valid = Min<uint32_t>(rowsPerJob, t.n - row);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            tile.Accumulate(col, row, valid);
            if (col >= row && col < row + valid) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                imag.SetValue(col - row, 0.0f);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            }
            const uint32_t begin = upper ? 0U : (col > row ? col - row : 0U);
            const uint32_t count = upper ? Min<uint32_t>(valid, col - row + 1U) : valid - begin;
            AscendC::Adds(
                indices.ReinterpretCast<int32_t>(), interleave.ReinterpretCast<int32_t>(),
                static_cast<int32_t>(begin * 4U), count * 2U);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(output, real, indices, 0U, count * 2U);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            const AscendC::DataCopyExtParams cp{1U, count * 8U, 0U, 0U, 0U};
            AscendC::DataCopyPad(c[(uint64_t(col) * t.ldc + row + begin) * 2ULL], output, cp);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        prefix += jobs;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
}

template <uint32_t ROWS, uint32_t COLS, uint32_t TRANSPOSE_OFFSET>
class CherkRectEpilogue {
public:
    static constexpr uint32_t RECT_ROWS = ROWS;
    static constexpr uint32_t RECT_COLS = COLS;
    static constexpr uint32_t RECT_SIZE = ROWS * COLS;
    struct RectBuffers {
        AscendC::LocalTensor<float> real;
        AscendC::LocalTensor<float> imag;
        AscendC::LocalTensor<float> cross;
        AscendC::LocalTensor<float> trans;
        AscendC::LocalTensor<float> aos;
        AscendC::LocalTensor<uint32_t> transpose;
        AscendC::LocalTensor<uint32_t> interleave;
    };
    static __aicore__ __attribute__((always_inline)) inline void RectLoad(
        AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float>& gm, uint64_t base, uint32_t rows, uint32_t cols,
        uint32_t pitch, uint32_t stride, uint32_t lanes = 1U)
    {
        const uint32_t bytes = rows * lanes * 4U;
        const uint32_t aligned = (bytes + 31U) & ~31U;
        const AscendC::DataCopyExtParams copy{
            static_cast<uint16_t>(cols), bytes, (stride - rows) * lanes * 4U, (pitch * lanes * 4U - aligned) / 32U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{
            aligned != bytes, 0U, static_cast<uint8_t>((aligned - bytes) / 4U), 0.0f};
        AscendC::DataCopyPad(dst, gm[base], copy, pad);
    }
    static __aicore__ __attribute__((always_inline)) inline void RectProduct(
        RectBuffers& b, AscendC::GlobalTensor<float>& workspace, const CherkArch22TilingData& t, uint32_t row,
        uint32_t col, uint32_t rows, uint32_t cols, float alpha)
    {
        if (alpha != 0.0f && t.k != 0U) {
            RectLoad(
                b.real, workspace, CherkArch22OffsetRealOutput(t.cubeN, t.cubeK) + uint64_t(col) * t.cubeN + row, rows,
                cols, RECT_ROWS, t.cubeN);
            RectLoad(
                b.imag, workspace, CherkArch22OffsetCrossOutput(t.cubeN, t.cubeK) + uint64_t(col) * t.cubeN + row, rows,
                cols, RECT_ROWS, t.cubeN);
            const bool directImag = CherkArch22DirectImaginary(t);
            if (!directImag) {
                RectLoad(
                    b.cross, workspace, CherkArch22OffsetCrossOutput(t.cubeN, t.cubeK) + uint64_t(row) * t.cubeN + col,
                    cols, rows, RECT_COLS, t.cubeN);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if (!directImag) {
                AscendC::Gather(b.trans, b.cross, b.transpose, 0U, RECT_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                if (t.trans == ACLBLAS_OP_N) {
                    AscendC::Sub(b.imag, b.imag, b.trans, RECT_SIZE);
                } else {
                    AscendC::Sub(b.imag, b.trans, b.imag, RECT_SIZE);
                }
            }
            if (alpha != 1.0f) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(b.real, b.real, alpha, RECT_SIZE);
                AscendC::Muls(b.imag, b.imag, alpha, RECT_SIZE);
            }
        } else {
            AscendC::Duplicate(b.real, 0.0f, RECT_SIZE);
            AscendC::Duplicate(b.imag, 0.0f, RECT_SIZE);
        }
    }
    static __aicore__ __attribute__((always_inline)) inline void RectBeta(
        RectBuffers& b, AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& t, uint32_t row, uint32_t col,
        uint32_t rows, uint32_t cols, float beta)
    {
        if (beta != 0.0f) {
            const bool selected = t.uplo == ACLBLAS_UPPER ? col >= row + rows : row >= col + cols;
            if (selected) {
                RectLoad(b.aos, c, (uint64_t(col) * t.ldc + row) * 2ULL, rows, cols, RECT_ROWS, t.ldc, 2U);
            } else {
                triangle_store::LoadComplexSelected<RECT_ROWS>(
                    b.aos, c, t.ldc, row, col, rows, cols, t.uplo == ACLBLAS_UPPER, EVENT_ID0);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            uint64_t reserved = 0;
            AscendC::GatherMask(b.cross, b.aos, 1U, false, 0U, {1U, 128U, 8U, 8U}, reserved);
            AscendC::GatherMask(b.trans, b.aos, 2U, false, 0U, {1U, 128U, 8U, 8U}, reserved);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(b.cross, b.cross, beta, RECT_SIZE);
            AscendC::Muls(b.trans, b.trans, beta, RECT_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(b.real, b.real, b.cross, RECT_SIZE);
            AscendC::Add(b.imag, b.imag, b.trans, RECT_SIZE);
        }
    }
    static __aicore__ __attribute__((always_inline)) inline void RectProcess(
        RectBuffers& b, AscendC::GlobalTensor<float>& workspace, AscendC::GlobalTensor<float>& c,
        const CherkArch22TilingData& t, uint32_t row, uint32_t col, float alpha, float beta)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        if (beta != 0.0f || t.uplo == ACLBLAS_LOWER) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        const uint32_t rows = Min<uint32_t>(RECT_ROWS, t.n - row);
        const uint32_t cols = Min<uint32_t>(RECT_COLS, t.n - col);
        RectProduct(b, workspace, t, row, col, rows, cols, alpha);
        RectBeta(b, c, t, row, col, rows, cols, beta);
        AscendC::PipeBarrier<PIPE_V>();
        const bool full = t.uplo == ACLBLAS_UPPER ? col >= row + rows : row >= col + cols;
        if (!full) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (uint32_t j = 0; j < cols; ++j)
                if (col + j >= row && col + j < row + rows) {
                    b.imag.SetValue(j * RECT_ROWS + col + j - row, 0.0f);
                }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        }
        if (beta == 0.0f && t.uplo == ACLBLAS_UPPER) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::Gather(b.aos, b.real, b.interleave, 0U, RECT_SIZE * 2U);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        if (full) {
            const uint32_t bytes = rows * 8U;
            const uint32_t aligned = (bytes + 31U) & ~31U;
            const AscendC::DataCopyExtParams cp{
                static_cast<uint16_t>(cols), bytes, (RECT_ROWS * 8U - aligned) / 32U, (t.ldc - rows) * 8U, 0U};
            AscendC::DataCopyPad(c[(uint64_t(col) * t.ldc + row) * 2ULL], b.aos, cp);
        } else {
            triangle_store::StoreComplexDiagonal<RECT_ROWS>(
                c, b.aos, b.real, b.cross.template ReinterpretCast<int32_t>(), t.ldc, row, col, rows, cols,
                t.uplo == ACLBLAS_UPPER, EVENT_ID0);
        }
        if (beta != 0.0f || t.uplo == ACLBLAS_LOWER) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
    static __aicore__ inline void RunJobs(
        RectBuffers& b, AscendC::GlobalTensor<float>& w, AscendC::GlobalTensor<float>& c,
        const CherkArch22TilingData& t, float alpha, float beta, uint32_t blocks)
    {
        const uint32_t block = AscendC::GetBlockIdx();
        const uint32_t rowTiles = (t.n + RECT_ROWS - 1U) / RECT_ROWS;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        uint32_t prefix = 0U;
        const uint32_t partialJobs = (t.n + RECT_COLS - 1U) / RECT_COLS;
        for (uint32_t col = 0; col < t.n; col += RECT_COLS) {
            const bool upper = t.uplo == ACLBLAS_UPPER;
            const uint32_t partialRow = col / RECT_ROWS;
            // Partial stores cost more than full 2D bursts. Distribute the two
            // domains independently, with full work filling the lighter cores.
            if ((col / RECT_COLS) % blocks == block) {
                RectProcess(b, w, c, t, partialRow * RECT_ROWS, col, alpha, beta);
            }
            const uint32_t first = upper ? 0U : partialRow + 1U;
            const uint32_t end = upper ? partialRow : rowTiles;
            const uint32_t count = end - first;
            for (uint32_t local = (block + blocks - (prefix + partialJobs) % blocks) % blocks; local < count;
                 local += blocks)
                RectProcess(b, w, c, t, (first + local) * RECT_ROWS, col, alpha, beta);
            prefix += count;
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
    static __aicore__ __attribute__((always_inline)) inline void Run(
        GM_ADDR alphaGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
        const CherkArch22TilingData t)
    {
        const float alpha = *reinterpret_cast<__gm__ float*>(alphaGm);
        const float beta = *reinterpret_cast<__gm__ float*>(betaGm);
        if ((alpha == 0.0f || t.k == 0U) && beta == 1.0f) {
            return;
        }
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> planar;
        AscendC::TBuf<AscendC::TPosition::VECCALC> aos;
        AscendC::TBuf<AscendC::TPosition::VECCALC> indices;
        pipe.InitBuffer(planar, RECT_SIZE * 4U * 4U);
        pipe.InitBuffer(aos, RECT_SIZE * 2U * 4U);
        pipe.InitBuffer(indices, RECT_SIZE * 3U * 4U);
        auto p = planar.Get<float>();
        auto idx = indices.Get<uint32_t>();
        RectBuffers b{p, p[RECT_SIZE], p[2U * RECT_SIZE], p[3U * RECT_SIZE], aos.Get<float>(), idx, idx[RECT_SIZE]};
        AscendC::GlobalTensor<uint32_t> meta;
        meta.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
        if (!CherkArch22DirectImaginary(t)) {
            AscendC::DataCopy(b.transpose, meta[TRANSPOSE_OFFSET], RECT_SIZE);
        }
        AscendC::DataCopy(b.interleave, meta[CHERK_ARCH22_META_INTERLEAVE_OFFSET], RECT_SIZE * 2U);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::GlobalTensor<float> w;
        AscendC::GlobalTensor<float> c;
        w.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm));
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));
        const uint32_t blocks = AscendC::GetBlockNum();
        const float magnitude = alpha < 0.0f ? -alpha : alpha;
        if (__builtin_isfinite(alpha) && magnitude > 1.0f && beta == 0.0f && t.trans == ACLBLAS_OP_N &&
            t.k > CUBE_PANEL_K) {
            ReferenceOrderProduct(b.real, b.imag, b.cross, b.aos, b.interleave, b.transpose, w, c, t, alpha, blocks);
            return;
        }

        RunJobs(b, w, c, t, alpha, beta, blocks);
    }
};

extern "C" __global__ __aicore__ void cherk_arch22_rect_epilogue_kernel(
    GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CherkRectEpilogue<128U, 32U, CHERK_ARCH22_META_RECT_OFFSET>::Run(alpha, beta, c, workspace, metadata, t);
}
extern "C" __global__ __aicore__ void cherk_arch22_wide_rect_epilogue_kernel(
    GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CherkRectEpilogue<256U, 16U, CHERK_ARCH22_META_WIDE_OFFSET>::Run(alpha, beta, c, workspace, metadata, t);
}
__host__ __aicore__ inline bool UseWideRect(const CherkArch22TilingData& t)
{
    // Wider output bursts amortize DMA cost when every AIV owns >=16 tiles.
    return uint64_t(t.n) * t.n >= uint64_t(t.aivCoreNum) * 16ULL * 8192ULL;
}
void cherk_arch22_compute_do(
    GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData& tiling,
    void* stream)
{
    if (tiling.skipProduct == 0U) {
        cherk_arch22_mmad_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(workspace, tiling);
    }
    if (tiling.n >= 128U && UseWideRect(tiling)) {
        cherk_arch22_wide_rect_epilogue_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
            alpha, beta, c, workspace, metadata, tiling);
    } else if (tiling.n >= 128U) {
        cherk_arch22_rect_epilogue_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
            alpha, beta, c, workspace, metadata, tiling);
    } else {
        cherk_arch22_epilogue_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
            alpha, beta, c, workspace, metadata, tiling);
    }
}

/** Single-launch H3 path for UPPER/OP_N/n=k=128. */
