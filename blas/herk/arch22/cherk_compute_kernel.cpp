/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_utils.h"
#include "common/compute/simd.h"
#include "cherk_kernel.h"

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
        uint32_t n, uint32_t k, bool transA, bool transB, uint32_t uplo)
    {
        const uint64_t inputCount = static_cast<uint64_t>(n) * k;
        const uint64_t outputCount = static_cast<uint64_t>(n) * n;
        realInput_.SetGlobalBuffer(realInput, inputCount);
        imagInput_.SetGlobalBuffer(imagInput, inputCount);
        realOutput_.SetGlobalBuffer(realOutput, outputCount);
        crossOutput_.SetGlobalBuffer(crossOutput, outputCount);
        n_ = n;
        k_ = k;
        transA_ = transA;
        transB_ = transB;
        uplo_ = uplo;
        InitCopyAndLoadParams();
    }

    __aicore__ inline void Process(uint32_t firstTile, uint32_t tileStride)
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
        const uint32_t mTiles = n_ / TILE_M;
        const uint32_t nTiles = n_ / CUBE_N;
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
                    ProductPair(
                        a1, b1, a2, b2, c1, realInput_, realInput_, imagInput_, imagInput_, realOutput_, mBase, nBase);
                }
                ++realJob;
            }
        }

        for (uint64_t tile = firstTile; tile < tileCount; tile += tileStride) {
            const uint32_t mBase = static_cast<uint32_t>(tile / nTiles) * TILE_M;
            const uint32_t nBase = static_cast<uint32_t>(tile % nTiles) * CUBE_N;
            // Q = R^T I. Its transpose is consumed from the mirrored tile.
            ProductSingle(a1, b1, a2, b2, c1, realInput_, imagInput_, crossOutput_, mBase, nBase);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    }

private:
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
        aLoadParams_.strideW = 1U;
        aLoadParams_.strideH = 1U;
        aLoadParams_.filterW = 1U;
        aLoadParams_.filterH = 1U;
        aLoadParams_.dilationFilterW = 1U;
        aLoadParams_.dilationFilterH = 1U;
        aLoadParams_.filterSizeW = false;
        aLoadParams_.filterSizeH = false;
        aLoadParams_.enTranspose = transA_;
        aLoadParams_.fMatrixCtrl = false;

        bLoadParams_ = {};
        bLoadParams_.l1H = 1U;
        bLoadParams_.l1W = transB_ ? CUBE_N : CUBE_K;
        bLoadParams_.channelSize = transB_ ? CUBE_K : CUBE_N;
        bLoadParams_.kExtension = transB_ ? CUBE_K : CUBE_N;
        bLoadParams_.mExtension = transB_ ? CUBE_N : CUBE_K;
        bLoadParams_.strideW = 1U;
        bLoadParams_.strideH = 1U;
        bLoadParams_.filterW = 1U;
        bLoadParams_.filterH = 1U;
        bLoadParams_.dilationFilterW = 1U;
        bLoadParams_.dilationFilterH = 1U;
        bLoadParams_.filterSizeW = false;
        bLoadParams_.filterSizeH = false;
        // L0B consumes its matrix orientation through the B-side path; this
        // matches the proven CHER2K/CHEMM arch22 contract for both layouts.
        bLoadParams_.enTranspose = true;
        bLoadParams_.fMatrixCtrl = false;
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
        params.m = TILE_M;
        params.n = CUBE_N;
        params.k = CUBE_K;
        params.cmatrixInitVal = initialize;
        params.cmatrixSource = false;
        params.kDirectionAlign = false;
        AscendC::Mmad(c1, a2, b2, params);
        AscendC::PipeBarrier<PIPE_ALL>();
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
        AscendC::GlobalTensor<float>& right, uint32_t mBase, uint32_t nBase, bool initialize)
    {
        if (k_ >= CUBE_PANEL_K && k_ % CUBE_PANEL_K == 0U) {
            constexpr uint32_t slicesPerPanel = CUBE_PANEL_K / CUBE_K;
            for (uint32_t panelBase = 0U; panelBase < k_; panelBase += CUBE_PANEL_K) {
                CopyPanelToL1(a1, b1, left, right, mBase, nBase, panelBase);
                for (uint32_t slice = 0U; slice < slicesPerPanel; ++slice) {
                    LoadPanelSliceToL0(a1, b1, a2, b2, slice, slice == 0U, slice + 1U == slicesPerPanel);
                    Compute(c1, a2, b2, initialize && panelBase == 0U && slice == 0U);
                }
            }
            return;
        }
        for (uint32_t kBase = 0U; kBase < k_; kBase += CUBE_K) {
            CopyToL1(a1, b1, left, right, mBase, nBase, kBase);
            LoadToL0(a1, b1, a2, b2);
            Compute(c1, a2, b2, initialize && kBase == 0U);
        }
    }

    __aicore__ inline void Store(
        AscendC::LocalTensor<float>& c1, AscendC::GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::FixpipeParamsV220 params{};
        params.mSize = TILE_M;
        params.nSize = CUBE_N;
        params.srcStride = TILE_M;
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
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AccumulateTerm(a1, b1, a2, b2, c1, left0, right0, mBase, nBase, true);
        AccumulateTerm(a1, b1, a2, b2, c1, left1, right1, mBase, nBase, false);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        Store(c1, output, mBase, nBase);
    }

    __aicore__ inline void ProductSingle(
        AscendC::LocalTensor<float>& a1, AscendC::LocalTensor<float>& b1, AscendC::LocalTensor<float>& a2,
        AscendC::LocalTensor<float>& b2, AscendC::LocalTensor<float>& c1, AscendC::GlobalTensor<float>& left,
        AscendC::GlobalTensor<float>& right, AscendC::GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AccumulateTerm(a1, b1, a2, b2, c1, left, right, mBase, nBase, true);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        Store(c1, output, mBase, nBase);
    }

    AscendC::GlobalTensor<float> realInput_, imagInput_;
    AscendC::GlobalTensor<float> realOutput_, crossOutput_;
    uint32_t n_ = 0U;
    uint32_t k_ = 0U;
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
    AscendC::DataCopy(b.interleaveOffsets, metadata[CHERK_ARCH22_META_INTERLEAVE_OFFSET], EPILOGUE_TILE * 2U);
    AscendC::DataCopy(
        b.diagonalMask, metadataBytes[CHERK_ARCH22_META_DIAGONAL_OFFSET * sizeof(uint32_t)],
        EPILOGUE_TILE_ELEMENTS / 8U);
    const event_t metadataToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
}

struct CherkEpilogueStorage {
    AscendC::TBuf<AscendC::TPosition::VECCALC> planarBuf, aosBuf, transposeOffsetBuf, interleaveOffsetBuf,
        diagonalMaskBuf;
    CherkEpilogueBuffers tensors;

    __aicore__ inline void Init(AscendC::TPipe& pipe, GM_ADDR metadataGm)
    {
        pipe.InitBuffer(planarBuf, EPILOGUE_TILE_ELEMENTS * 4U * sizeof(float));
        pipe.InitBuffer(aosBuf, EPILOGUE_AOS_ELEMENTS * sizeof(float));
        pipe.InitBuffer(transposeOffsetBuf, EPILOGUE_TILE_ELEMENTS * sizeof(uint32_t));
        pipe.InitBuffer(interleaveOffsetBuf, EPILOGUE_TILE * 2U * sizeof(uint32_t));
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
    AscendC::DataCopyPad(b.aos, c[cOffset], copy, pad);
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
    if (tiling.compactPath != 0U) {
        CherkArch22Mmad<128U> mm;
        mm.Init(realInput, imagInput, realOutput, crossOutput, tiling.cubeN, tiling.cubeK, true, false, tiling.uplo);
        mm.Process(AscendC::GetBlockIdx(), blockNum);
    } else {
        CherkArch22Mmad<CUBE_M> mm;
        mm.Init(realInput, imagInput, realOutput, crossOutput, tiling.cubeN, tiling.cubeK, true, false, tiling.uplo);
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
void cherk_arch22_compute_do(
    GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData& tiling,
    void* stream)
{
    if (tiling.skipProduct == 0U) {
        cherk_arch22_mmad_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(workspace, tiling);
    }
    cherk_arch22_epilogue_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(alpha, beta, c, workspace, metadata, tiling);
}

/** Single-launch H3 path for UPPER/OP_N/n=k=128. */
namespace {
constexpr uint32_t SQUARE128_N = 128U;
constexpr uint32_t SQUARE128_PACK_ROWS = 32U;
constexpr uint32_t SQUARE128_PACK_ELEMENTS = SQUARE128_PACK_ROWS * SQUARE128_N;

__aicore__ inline void StoreSquare128Packed(
    AscendC::GlobalTensor<float>& workspace, AscendC::LocalTensor<float>& real, AscendC::LocalTensor<float>& imag,
    uint32_t packedRowBase, event_t vToMte3, event_t mte3ToV)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    const uint64_t outputOffset = static_cast<uint64_t>(packedRowBase) * SQUARE128_N;
    AscendC::DataCopy(
        workspace[CherkArch22OffsetRealInput(SQUARE128_N, SQUARE128_N) + outputOffset], real, SQUARE128_PACK_ELEMENTS);
    AscendC::DataCopy(
        workspace[CherkArch22OffsetImagInput(SQUARE128_N, SQUARE128_N) + outputOffset], imag, SQUARE128_PACK_ELEMENTS);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
}

__aicore__ inline void CherkArch22Square128Pack(GM_ADDR aGm, GM_ADDR workspaceGm, const CherkArch22TilingData tiling)
{
    // In MIX 1:2, AscendC::GetBlockIdx() is already the global AIV ordinal (0..3).
    const uint32_t globalAiv = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t packedRowBase = globalAiv * SQUARE128_PACK_ROWS;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf, planarBuf;
    pipe.InitBuffer(aosBuf, SQUARE128_PACK_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(planarBuf, SQUARE128_PACK_ELEMENTS * 2U * sizeof(float));
    AscendC::LocalTensor<float> aos = aosBuf.Get<float>();
    AscendC::LocalTensor<float> real = planarBuf.Get<float>();
    AscendC::LocalTensor<float> imag = real[SQUARE128_PACK_ELEMENTS];

    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> workspace;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm), static_cast<uint64_t>(tiling.lda) * SQUARE128_N * 2ULL);
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm), WorkspaceCount(tiling));
    const AscendC::DataCopyExtParams inputCopy{
        SQUARE128_PACK_ROWS, SQUARE128_N * 2U * sizeof(float),
        static_cast<uint32_t>((tiling.lda - SQUARE128_N) * 2U * sizeof(float)), 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0f};
    AscendC::DataCopyPad(aos, a[static_cast<uint64_t>(packedRowBase) * tiling.lda * 2ULL], inputCopy, noPad);
    const event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    const event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    const event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = SQUARE128_PACK_ELEMENTS * 2U / 64U;
    AscendC::GatherMask(real, aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask(imag, aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    StoreSquare128Packed(workspace, real, imag, packedRowBase, vToMte3, mte3ToV);
    pipe.Reset();
}

__aicore__ inline void ProcessSquare128ConsumeTile(
    uint32_t task, float alpha, float beta, bool hasProduct, AscendC::GlobalTensor<float>& c,
    AscendC::GlobalTensor<float>& workspace, const CherkArch22TilingData& tiling, CherkEpilogueBuffers& b)
{
    constexpr uint32_t rowTile[3] = {0U, 0U, 1U};
    constexpr uint32_t colTile[3] = {0U, 1U, 1U};
    const uint32_t rowBase = rowTile[task] * EPILOGUE_TILE;
    const uint32_t colBase = colTile[task] * EPILOGUE_TILE;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    if (hasProduct) {
        LoadProductTile(
            workspace, tiling, rowBase, colBase, EPILOGUE_TILE, EPILOGUE_TILE, alpha, rowBase == colBase, b);
    } else {
        AscendC::Duplicate(b.real, 0.0f, EPILOGUE_TILE_ELEMENTS);
        AscendC::Duplicate(b.imag, 0.0f, EPILOGUE_TILE_ELEMENTS);
    }
    if (beta != 0.0f) {
        AscendC::PipeBarrier<PIPE_V>();
        LoadAndApplyBeta(c, tiling, rowBase, colBase, EPILOGUE_TILE, EPILOGUE_TILE, beta, b);
    }
    ZeroDiagonal(rowBase, colBase, EPILOGUE_TILE, EPILOGUE_TILE, b);
    InterleaveTile(tiling, rowBase, colBase, EPILOGUE_TILE, EPILOGUE_TILE, b);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    StoreSelectedTile(c, tiling, rowBase, colBase, EPILOGUE_TILE, EPILOGUE_TILE, b);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void CherkArch22Square128Consume(
    GM_ADDR alphaGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    const bool hasProduct = alpha != 0.0f;
    AscendC::TPipe pipe;
    CherkEpilogueStorage storage;
    storage.Init(pipe, metadataGm);
    CherkEpilogueBuffers& b = storage.tensors;

    AscendC::GlobalTensor<float> c;
    AscendC::GlobalTensor<float> workspace;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm), static_cast<uint64_t>(tiling.ldc) * SQUARE128_N * 2ULL);
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm), WorkspaceCount(tiling));
    // AIC0 publishes Q while AIC1 publishes the accumulated real plane. The
    // second phase barrier joins both domains after metadata has moved to UB.
    AscendC::SyncAll<false>();

    const uint32_t globalAiv = static_cast<uint32_t>(AscendC::GetBlockIdx());
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    for (uint32_t task = globalAiv; task < 3U; task += 4U) {
        ProcessSquare128ConsumeTile(task, alpha, beta, hasProduct, c, workspace, tiling, b);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
}
} // namespace

extern "C" __global__ __aicore__ __schedmode__(1) void cherk_arch22_square128_mix_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        CherkArch22Square128Pack(aGm, workspaceGm, tiling);
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIC {
        AscendC::InitSocState();
        __gm__ float* base = reinterpret_cast<__gm__ float*>(workspaceGm);
        CherkArch22Mmad<128U> mm;
        mm.Init(
            base + CherkArch22OffsetRealInput(SQUARE128_N, SQUARE128_N),
            base + CherkArch22OffsetImagInput(SQUARE128_N, SQUARE128_N),
            base + CherkArch22OffsetRealOutput(SQUARE128_N, SQUARE128_N),
            base + CherkArch22OffsetCrossOutput(SQUARE128_N, SQUARE128_N), SQUARE128_N, SQUARE128_N, true, false,
            ACLBLAS_UPPER);
        mm.Process(static_cast<uint32_t>(AscendC::GetBlockIdx()), 2U);
        AscendC::SyncAll<false>();
        return;
    }
    if ASCEND_IS_AIV {
        CherkArch22Square128Consume(alphaGm, betaGm, cGm, workspaceGm, metadataGm, tiling);
    }
}
void cherk_arch22_square128_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream)
{
    cherk_arch22_square128_mix_kernel<<<2U, nullptr, stream>>>(alpha, a, beta, c, workspace, metadata, tiling);
}
