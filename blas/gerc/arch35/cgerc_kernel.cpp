/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "cgerc_kernel.h"

namespace {

using namespace AscendC;

constexpr uint32_t COMPLEX_COMPONENT_COUNT = 2U;
using CgercConfig::A_BUFFER_COUNT;
using CgercConfig::UB_BLOCK_BYTES;
constexpr uint32_t FUSED_COLUMN_COUNT = 3U;
constexpr uint8_t FIRST_BUFFER_INDEX = 0U;
constexpr uint8_t SECOND_BUFFER_INDEX = 1U;
static_assert(VECTOR_REG_WIDTH == CgercConfig::VECTOR_REGISTER_BYTES, "Cgerc Host/Kernel register widths differ");

__aicore__ inline void MatchCblasAlphaOneSpecialValue(float& yReal, float& yImag)
{
    // Self-inequality detects NaN without classifying either infinity as NaN.
    if (yReal != yReal || yImag != yImag) {
        const float originalReal = yReal;
        const float originalImag = yImag;
        yReal = originalReal + 0.0F * originalImag;
        yImag = originalImag - 0.0F * originalReal;
    }
}

template <typename Register, typename Mask>
__simd_callee__ inline void RoundedAxpy(Register& destination, Register& source, float scalar, Mask& mask)
{
    // Explicit register Axpy is fused even with -ffp-contract=off. Separate instructions preserve the
    // FP32 multiplication boundary, including overflow followed by opposite-signed infinity addition.
    Register product;
    AscendC::MicroAPI::Muls(product, source, scalar, mask);
    AscendC::MicroAPI::Add(destination, destination, product, mask);
}

__simd_vf__ inline void ComplexAxpyInterleaved(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, float yReal, float yImag, uint16_t loopNum, uint32_t tailRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> xReal;
    RegTensor<float> xImag;
    RegTensor<float> aReal;
    RegTensor<float> aImag;
    auto fullMask = CreateMask<float, MaskPattern::ALL>();
    const uint16_t loops = loopNum;

    for (uint16_t i = 0; i < loops; ++i) {
        uint32_t activeRows = (i + 1U == loops && tailRows != 0U) ? tailRows : VL;
        auto mask = activeRows < VL ? UpdateMask<float>(activeRows) : fullMask;
        const uint32_t realOffset = static_cast<uint32_t>(i) * VL;
        const uint32_t packedOffset = COMPLEX_COMPONENT_COUNT * realOffset;
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(xReal, xImag, xAddr + packedOffset);
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(aReal, aImag, aAddr + packedOffset);
        RoundedAxpy(aReal, xReal, yReal, mask);
        RoundedAxpy(aReal, xImag, yImag, mask);
        RoundedAxpy(aImag, xImag, yReal, mask);
        RoundedAxpy(aImag, xReal, -yImag, mask);
        DataCopy<float, StoreDist::DIST_INTLV_B32>(aAddr + packedOffset, aReal, aImag, mask);
    }
}

__simd_vf__ inline void ComplexAxpyInterleaved3(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr0, __ubuf__ float* aAddr1, __ubuf__ float* aAddr2, float yReal0,
    float yImag0, float yReal1, float yImag1, float yReal2, float yImag2, uint16_t loopNum, uint32_t tailRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> xReal;
    RegTensor<float> xImag;
    RegTensor<float> aReal0;
    RegTensor<float> aImag0;
    RegTensor<float> aReal1;
    RegTensor<float> aImag1;
    RegTensor<float> aReal2;
    RegTensor<float> aImag2;
    auto fullMask = CreateMask<float, MaskPattern::ALL>();
    const uint16_t loops = loopNum;

    for (uint16_t i = 0; i < loops; ++i) {
        uint32_t activeRows = (i + 1U == loops && tailRows != 0U) ? tailRows : VL;
        auto mask = activeRows < VL ? UpdateMask<float>(activeRows) : fullMask;
        const uint32_t packedOffset = COMPLEX_COMPONENT_COUNT * static_cast<uint32_t>(i) * VL;
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(xReal, xImag, xAddr + packedOffset);
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(aReal0, aImag0, aAddr0 + packedOffset);
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(aReal1, aImag1, aAddr1 + packedOffset);
        DataCopy<float, LoadDist::DIST_DINTLV_B32>(aReal2, aImag2, aAddr2 + packedOffset);

        RoundedAxpy(aReal0, xReal, yReal0, mask);
        RoundedAxpy(aReal1, xReal, yReal1, mask);
        RoundedAxpy(aReal2, xReal, yReal2, mask);
        RoundedAxpy(aReal0, xImag, yImag0, mask);
        RoundedAxpy(aReal1, xImag, yImag1, mask);
        RoundedAxpy(aReal2, xImag, yImag2, mask);
        RoundedAxpy(aImag0, xImag, yReal0, mask);
        RoundedAxpy(aImag1, xImag, yReal1, mask);
        RoundedAxpy(aImag2, xImag, yReal2, mask);
        RoundedAxpy(aImag0, xReal, -yImag0, mask);
        RoundedAxpy(aImag1, xReal, -yImag1, mask);
        RoundedAxpy(aImag2, xReal, -yImag2, mask);

        DataCopy<float, StoreDist::DIST_INTLV_B32>(aAddr0 + packedOffset, aReal0, aImag0, mask);
        DataCopy<float, StoreDist::DIST_INTLV_B32>(aAddr1 + packedOffset, aReal1, aImag1, mask);
        DataCopy<float, StoreDist::DIST_INTLV_B32>(aAddr2 + packedOffset, aReal2, aImag2, mask);
    }
}

class CgercKernel {
public:
    __aicore__ inline bool Init(GM_ADDR x, GM_ADDR y, GM_ADDR A, const CgercTilingData& tiling, TPipe* pipe)
    {
        m_ = tiling.m;
        alignedM_ = tiling.vectorAlignedRows;
        n_ = tiling.n;
        colBlocks_ = tiling.colBlocks;
        const uint32_t colBlock = GetBlockIdx();
        if (colBlocks_ == 0U || colBlock >= colBlocks_) {
            return false;
        }
        const uint32_t perCoreN = n_ / colBlocks_;
        const uint32_t remainder = n_ % colBlocks_;
        if (colBlock < remainder) {
            colStart_ = colBlock * (perCoreN + 1U);
            colEnd_ = colStart_ + perCoreN + 1U;
        } else {
            colStart_ = colBlock * perCoreN + remainder;
            colEnd_ = colStart_ + perCoreN;
        }
        colEnd_ = colEnd_ < n_ ? colEnd_ : n_;
        if (colStart_ >= colEnd_ || !InitBuffers(tiling.vectorTileColumns, pipe)) {
            return false;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
        xGm_.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
        yGm_.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
        aGm_.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
        yReadyEventId_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        return true;
    }

    __aicore__ inline void Process()
    {
        if (colStart_ >= colEnd_ || tileCols_ == 0U) {
            return;
        }
        LocalTensor<float> xInterleaved = xInterleavedBuf_.Get<float>();
        LocalTensor<float> yInterleaved = yInterleavedBuf_.Get<float>();
        LocalTensor<float> aInterleaved0 = aInterleavedBuf0_.Get<float>();
        LocalTensor<float> aInterleaved1 = aInterleavedBuf1_.Get<float>();
        LocalTensor<float> aInterleaved2 = aInterleavedBuf2_.Get<float>();
        CopyInX(xInterleaved);
        __ubuf__ float* xAddr = reinterpret_cast<__ubuf__ float*>(xInterleaved.GetPhyAddr());
        constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
        const uint16_t vectorLoops = static_cast<uint16_t>((m_ + VL - 1U) / VL);
        const uint32_t tailRows = m_ % VL;
        const uint32_t tileCount = (colEnd_ - colStart_ + tileCols_ - 1U) / tileCols_;
        const uint32_t firstCols = (colEnd_ - colStart_) < tileCols_ ? (colEnd_ - colStart_) : tileCols_;
        const uint64_t firstOffset = COMPLEX_COMPONENT_COUNT * static_cast<uint64_t>(colStart_) * m_;
        CopyInColumns(aInterleaved0, firstOffset, firstCols);
        SetFlag<HardEvent::MTE2_V>(FIRST_BUFFER_INDEX);

        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            ProcessTile(
                tile, tileCount, xAddr, yInterleaved, aInterleaved0, aInterleaved1, aInterleaved2, vectorLoops,
                tailRows);
        }
        const uint8_t lastEventId = static_cast<uint8_t>((tileCount - 1U) % A_BUFFER_COUNT);
        WaitFlag<HardEvent::MTE3_MTE2>(lastEventId);
    }

private:
    __aicore__ inline bool InitBuffers(uint32_t capacityColumns, TPipe* pipe)
    {
        constexpr uint32_t VL = CgercConfig::FP32_VECTOR_LANES;
        if (capacityColumns == 0U || alignedM_ < m_ || alignedM_ % VL != 0U) {
            return false;
        }
        const uint32_t columnsPerBuffer = (colEnd_ - colStart_ + A_BUFFER_COUNT - 1U) / A_BUFFER_COUNT;
        tileCols_ = capacityColumns < columnsPerBuffer ? capacityColumns : columnsPerBuffer;
        const uint64_t xBytes = COMPLEX_COMPONENT_COUNT * static_cast<uint64_t>(alignedM_) * sizeof(float);
        const uint64_t aBytes = tileCols_ * xBytes;
        const uint64_t rawYBytes = COMPLEX_COMPONENT_COUNT * static_cast<uint64_t>(tileCols_) * sizeof(float);
        const uint64_t yBytes = (rawYBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
        if (xBytes + A_BUFFER_COUNT * aBytes + yBytes > UB_SIZE) {
            return false;
        }
        return pipe->InitBuffer(xInterleavedBuf_, static_cast<uint32_t>(xBytes)) &&
               pipe->InitBuffer(yInterleavedBuf_, static_cast<uint32_t>(yBytes)) &&
               pipe->InitBuffer(aInterleavedBuf0_, static_cast<uint32_t>(aBytes)) &&
               pipe->InitBuffer(aInterleavedBuf1_, static_cast<uint32_t>(aBytes)) &&
               pipe->InitBuffer(aInterleavedBuf2_, static_cast<uint32_t>(aBytes));
    }

    __aicore__ inline void CopyInX(LocalTensor<float> destination)
    {
        const uint32_t columnBytes = COMPLEX_COMPONENT_COUNT * m_ * sizeof(float);
        DataCopyPad(
            destination, xGm_[0], DataCopyExtParams{1U, columnBytes, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
        SetFlag<HardEvent::MTE2_V>(FIRST_BUFFER_INDEX);
        WaitFlag<HardEvent::MTE2_V>(FIRST_BUFFER_INDEX);
    }

    __aicore__ inline LocalTensor<float> SelectABuffer(
        uint8_t bufferIndex, LocalTensor<float> first, LocalTensor<float> second, LocalTensor<float> third)
    {
        if (bufferIndex == FIRST_BUFFER_INDEX) {
            return first;
        }
        return bufferIndex == SECOND_BUFFER_INDEX ? second : third;
    }

    __aicore__ inline void CopyInYColumns(LocalTensor<float> destination, uint32_t columnOffset, uint32_t columns)
    {
        const uint32_t copyBytes = COMPLEX_COMPONENT_COUNT * columns * sizeof(float);
        const uint64_t sourceOffset = COMPLEX_COMPONENT_COUNT * static_cast<uint64_t>(columnOffset);
        DataCopyPad(
            destination, yGm_[sourceOffset], DataCopyExtParams{1U, copyBytes, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
    }

    __aicore__ inline void PrefetchNextTile(
        uint32_t tile, uint32_t tileCount, uint32_t colBase, LocalTensor<float> first, LocalTensor<float> second,
        LocalTensor<float> third)
    {
        if (tile + 1U >= tileCount) {
            return;
        }
        const uint8_t nextEventId = static_cast<uint8_t>((tile + 1U) % A_BUFFER_COUNT);
        LocalTensor<float> next = SelectABuffer(nextEventId, first, second, third);
        if (tile >= A_BUFFER_COUNT - 1U) {
            WaitFlag<HardEvent::MTE3_MTE2>(nextEventId);
        }
        const uint32_t nextColBase = colBase + tileCols_;
        const uint32_t nextCols = (colEnd_ - nextColBase) < tileCols_ ? (colEnd_ - nextColBase) : tileCols_;
        const uint64_t nextOffset = COMPLEX_COMPONENT_COUNT * static_cast<uint64_t>(nextColBase) * m_;
        CopyInColumns(next, nextOffset, nextCols);
        SetFlag<HardEvent::MTE2_V>(nextEventId);
    }

    __aicore__ inline void LoadY(__ubuf__ const float* yAddr, uint32_t localColumn, float& yReal, float& yImag)
    {
        const uint32_t offset = COMPLEX_COMPONENT_COUNT * localColumn;
        yReal = yAddr[offset];
        yImag = yAddr[offset + 1U];
        MatchCblasAlphaOneSpecialValue(yReal, yImag);
    }

    __aicore__ inline void ProcessFusedColumns(
        __ubuf__ float* xAddr, __ubuf__ const float* yAddr, __ubuf__ float* aAddr, uint32_t localColumn,
        uint16_t vectorLoops, uint32_t tailRows)
    {
        float yReal0;
        float yImag0;
        float yReal1;
        float yImag1;
        float yReal2;
        float yImag2;
        LoadY(yAddr, localColumn, yReal0, yImag0);
        LoadY(yAddr, localColumn + 1U, yReal1, yImag1);
        LoadY(yAddr, localColumn + FUSED_COLUMN_COUNT - 1U, yReal2, yImag2);
        const bool nonzero0 = yReal0 != 0.0F || yImag0 != 0.0F;
        const bool nonzero1 = yReal1 != 0.0F || yImag1 != 0.0F;
        const bool nonzero2 = yReal2 != 0.0F || yImag2 != 0.0F;
        if (nonzero0 && nonzero1 && nonzero2) {
            ComplexAxpyInterleaved3(
                xAddr, aAddr + COMPLEX_COMPONENT_COUNT * localColumn * alignedM_,
                aAddr + COMPLEX_COMPONENT_COUNT * (localColumn + 1U) * alignedM_,
                aAddr + COMPLEX_COMPONENT_COUNT * (localColumn + FUSED_COLUMN_COUNT - 1U) * alignedM_, yReal0, yImag0,
                yReal1, yImag1, yReal2, yImag2, vectorLoops, tailRows);
            return;
        }
        ProcessColumnIfNonzero(xAddr, aAddr, localColumn, yReal0, yImag0, vectorLoops, tailRows);
        ProcessColumnIfNonzero(xAddr, aAddr, localColumn + 1U, yReal1, yImag1, vectorLoops, tailRows);
        ProcessColumnIfNonzero(
            xAddr, aAddr, localColumn + FUSED_COLUMN_COUNT - 1U, yReal2, yImag2, vectorLoops, tailRows);
    }

    __aicore__ inline void ProcessColumnIfNonzero(
        __ubuf__ float* xAddr, __ubuf__ float* aAddr, uint32_t localColumn, float yReal, float yImag,
        uint16_t vectorLoops, uint32_t tailRows)
    {
        if (yReal == 0.0F && yImag == 0.0F) {
            return;
        }
        ComplexAxpyInterleaved(
            xAddr, aAddr + COMPLEX_COMPONENT_COUNT * localColumn * alignedM_, yReal, yImag, vectorLoops, tailRows);
    }

    __aicore__ inline void ComputeTile(
        __ubuf__ float* xAddr, LocalTensor<float> yInterleaved, LocalTensor<float> aInterleaved, uint32_t columns,
        uint16_t vectorLoops, uint32_t tailRows)
    {
        __ubuf__ const float* yAddr = reinterpret_cast<__ubuf__ const float*>(yInterleaved.GetPhyAddr());
        __ubuf__ float* aAddr = reinterpret_cast<__ubuf__ float*>(aInterleaved.GetPhyAddr());
        uint32_t localColumn = 0U;
        for (; localColumn + FUSED_COLUMN_COUNT - 1U < columns; localColumn += FUSED_COLUMN_COUNT) {
            ProcessFusedColumns(xAddr, yAddr, aAddr, localColumn, vectorLoops, tailRows);
        }
        for (; localColumn < columns; ++localColumn) {
            float yReal;
            float yImag;
            LoadY(yAddr, localColumn, yReal, yImag);
            ProcessColumnIfNonzero(xAddr, aAddr, localColumn, yReal, yImag, vectorLoops, tailRows);
        }
    }

    __aicore__ inline void ProcessTile(
        uint32_t tile, uint32_t tileCount, __ubuf__ float* xAddr, LocalTensor<float> yInterleaved,
        LocalTensor<float> first, LocalTensor<float> second, LocalTensor<float> third, uint16_t vectorLoops,
        uint32_t tailRows)
    {
        const uint8_t eventId = static_cast<uint8_t>(tile % A_BUFFER_COUNT);
        LocalTensor<float> current = SelectABuffer(eventId, first, second, third);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        const uint32_t colBase = colStart_ + tile * tileCols_;
        const uint32_t curCols = (colEnd_ - colBase) < tileCols_ ? (colEnd_ - colBase) : tileCols_;
        CopyInYColumns(yInterleaved, colBase, curCols);
        SetFlag<HardEvent::MTE2_S>(yReadyEventId_);
        WaitFlag<HardEvent::MTE2_S>(yReadyEventId_);
        PrefetchNextTile(tile, tileCount, colBase, first, second, third);
        ComputeTile(xAddr, yInterleaved, current, curCols, vectorLoops, tailRows);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        const uint64_t aOffset = COMPLEX_COMPONENT_COUNT * static_cast<uint64_t>(colBase) * m_;
        CopyOutColumns(aOffset, current, curCols);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    __aicore__ inline uint32_t LocalColumnStrideBlocks() const
    {
        const uint32_t columnBytes = COMPLEX_COMPONENT_COUNT * m_ * sizeof(float);
        const uint32_t copiedBytes = (columnBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
        const uint32_t localColumnBytes = COMPLEX_COMPONENT_COUNT * alignedM_ * sizeof(float);
        return (localColumnBytes - copiedBytes) / UB_BLOCK_BYTES;
    }

    __aicore__ inline void CopyInColumns(LocalTensor<float> destination, uint64_t sourceOffset, uint32_t columns)
    {
        const uint32_t columnBytes = COMPLEX_COMPONENT_COUNT * m_ * sizeof(float);
        DataCopyExtParams copyParams{static_cast<uint16_t>(columns), columnBytes, 0U, LocalColumnStrideBlocks(), 0U};
        DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
        DataCopyPad(destination, aGm_[sourceOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyOutColumns(uint64_t destinationOffset, LocalTensor<float> source, uint32_t columns)
    {
        const uint32_t columnBytes = COMPLEX_COMPONENT_COUNT * m_ * sizeof(float);
        DataCopyExtParams copyParams{static_cast<uint16_t>(columns), columnBytes, LocalColumnStrideBlocks(), 0U, 0U};
        DataCopyPad(aGm_[destinationOffset], source, copyParams);
    }

    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> aGm_;
    TBuf<TPosition::VECCALC> xInterleavedBuf_;
    TBuf<TPosition::VECCALC> yInterleavedBuf_;
    TBuf<TPosition::VECCALC> aInterleavedBuf0_;
    TBuf<TPosition::VECCALC> aInterleavedBuf1_;
    TBuf<TPosition::VECCALC> aInterleavedBuf2_;
    uint32_t m_ = 0;
    uint32_t alignedM_ = 0;
    uint32_t n_ = 0;
    uint32_t colBlocks_ = 1;
    uint32_t colStart_ = 0;
    uint32_t colEnd_ = 0;
    uint32_t tileCols_ = 1;
    event_t yReadyEventId_{};
};

template <bool ALPHA_ONE>
__simt_callee__ __aicore__ inline float2 ConjugateAndScaleY(float2 yValue, float alphaReal, float alphaImag)
{
    float2 result;
    if constexpr (ALPHA_ONE) {
        // Keep infinities intact; only true NaNs need component propagation.
        if (yValue.x != yValue.x || yValue.y != yValue.y) {
            const float originalReal = yValue.x;
            const float originalImag = yValue.y;
            yValue.x = originalReal + 0.0F * originalImag;
            yValue.y = originalImag - 0.0F * originalReal;
        }
        result.x = yValue.x;
        result.y = -yValue.y;
    } else {
        result.x = alphaReal * yValue.x + alphaImag * yValue.y;
        result.y = alphaImag * yValue.x - alphaReal * yValue.y;
    }
    return result;
}

__simt_callee__ __aicore__ inline void AccumulateComplexOuterProduct(
    float2 xValue, float2 scaledConjugateY, __gm__ float2* aGm, uint64_t aIndex)
{
    float2 aValue = aGm[aIndex];
    // Materialize each FP32 product before the component sum/difference to preserve non-fused evaluation,
    // including overflow and Inf/NaN propagation. Keep these SIMT rounding boundaries in addition to the
    // file-level -ffp-contract=off option; this file has no fp-contract pragma covering SIMT callees.
    volatile float realTerm0 = xValue.x * scaledConjugateY.x;
    volatile float realTerm1 = xValue.y * scaledConjugateY.y;
    volatile float imagTerm0 = xValue.x * scaledConjugateY.y;
    volatile float imagTerm1 = xValue.y * scaledConjugateY.x;
    aValue.x += realTerm0 - realTerm1;
    aValue.y += imagTerm0 + imagTerm1;
    aGm[aIndex] = aValue;
}

template <bool ALPHA_ONE>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgercContiguous(
    uint32_t m, uint32_t n, uint32_t rowBlocks, uint32_t colBlocks, float alphaReal, float alphaImag,
    __gm__ const float2* xGm, __gm__ const float2* yGm, __gm__ float2* aGm)
{
    const uint32_t block = blockIdx.x;
    const uint32_t rowBlock = block % rowBlocks;
    const uint32_t colBlock = block / rowBlocks;
    if (colBlock >= colBlocks) {
        return;
    }
    const uint64_t rowStep = static_cast<uint64_t>(rowBlocks) * blockDim.x;

    for (uint64_t row = static_cast<uint64_t>(rowBlock) * blockDim.x + threadIdx.x; row < m; row += rowStep) {
        const float2 xVal = xGm[row];

        for (uint64_t col = colBlock; col < n; col += colBlocks) {
            const float2 yValue = yGm[col];
            if (yValue.x == 0.0F && yValue.y == 0.0F) {
                continue;
            }
            const uint64_t aIndex = row + col * static_cast<uint64_t>(m);
            const float2 scaledConjugateY = ConjugateAndScaleY<ALPHA_ONE>(yValue, alphaReal, alphaImag);
            AccumulateComplexOuterProduct(xVal, scaledConjugateY, aGm, aIndex);
        }
    }
}

template <bool ALPHA_ONE>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgercStrided(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t rowBlocks, uint32_t colBlocks, int32_t incx, int32_t incy,
    float alphaReal, float alphaImag, __gm__ const float2* xGm, __gm__ const float2* yGm, __gm__ float2* aGm)
{
    const uint32_t block = blockIdx.x;
    const uint32_t rowBlock = block % rowBlocks;
    const uint32_t colBlock = block / rowBlocks;
    if (colBlock >= colBlocks) {
        return;
    }
    const uint64_t rowStep = static_cast<uint64_t>(rowBlocks) * blockDim.x;
    const int64_t absIncx = incx > 0 ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t absIncy = incy > 0 ? static_cast<int64_t>(incy) : -static_cast<int64_t>(incy);

    for (uint64_t row = static_cast<uint64_t>(rowBlock) * blockDim.x + threadIdx.x; row < m; row += rowStep) {
        const uint64_t xIndex = incx > 0 ? row * absIncx : (static_cast<uint64_t>(m) - 1U - row) * absIncx;
        const float2 xVal = xGm[xIndex];

        for (uint64_t col = colBlock; col < n; col += colBlocks) {
            const uint64_t yIndex = incy > 0 ? col * absIncy : (static_cast<uint64_t>(n) - 1U - col) * absIncy;
            const float2 yValue = yGm[yIndex];
            if (yValue.x == 0.0F && yValue.y == 0.0F) {
                continue;
            }
            const uint64_t aIndex = row + col * static_cast<uint64_t>(lda);
            const float2 scaledConjugateY = ConjugateAndScaleY<ALPHA_ONE>(yValue, alphaReal, alphaImag);
            AccumulateComplexOuterProduct(xVal, scaledConjugateY, aGm, aIndex);
        }
    }
}

} // namespace

extern "C" __global__ __aicore__ void cgerc_arch35_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR A, const CgercTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.m == 0U || tiling.n == 0U || tiling.rowBlocks == 0U || tiling.colBlocks == 0U ||
        static_cast<uint64_t>(GetBlockIdx()) >= static_cast<uint64_t>(tiling.rowBlocks) * tiling.colBlocks) {
        return;
    }
    if (tiling.vectorPath != 0U) {
        TPipe pipe;
        CgercKernel op;
        if (op.Init(x, y, A, tiling, &pipe)) {
            op.Process();
            return;
        }
    }

    auto* xGm = reinterpret_cast<__gm__ const float2*>(x);
    auto* yGm = reinterpret_cast<__gm__ const float2*>(y);
    auto* aGm = reinterpret_cast<__gm__ float2*>(A);
    const dim3 threads{tiling.numThreads, 1, 1};

    if (tiling.contiguous != 0U) {
        if (tiling.alphaOne != 0U) {
            asc_vf_call<CgercContiguous<true>>(
                threads, tiling.m, tiling.n, tiling.rowBlocks, tiling.colBlocks, tiling.alphaReal, tiling.alphaImag,
                xGm, yGm, aGm);
        } else {
            asc_vf_call<CgercContiguous<false>>(
                threads, tiling.m, tiling.n, tiling.rowBlocks, tiling.colBlocks, tiling.alphaReal, tiling.alphaImag,
                xGm, yGm, aGm);
        }
        return;
    }

    if (tiling.alphaOne != 0U) {
        asc_vf_call<CgercStrided<true>>(
            threads, tiling.m, tiling.n, tiling.lda, tiling.rowBlocks, tiling.colBlocks, tiling.incx, tiling.incy,
            tiling.alphaReal, tiling.alphaImag, xGm, yGm, aGm);
    } else {
        asc_vf_call<CgercStrided<false>>(
            threads, tiling.m, tiling.n, tiling.lda, tiling.rowBlocks, tiling.colBlocks, tiling.incx, tiling.incy,
            tiling.alphaReal, tiling.alphaImag, xGm, yGm, aGm);
    }
}

aclError cgerc_arch35_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* a, const CgercTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    cgerc_arch35_kernel<<<numBlocks, nullptr, stream>>>(x, y, a, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
