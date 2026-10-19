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

#include "ctbmv_kernel_common.h"

#if __DAV_C220_VEC__
namespace Ctbmv {

class CtbmvCompactPanelAiv : public CtbmvShape {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        InitShape(tiling);
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2U);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(n_) * 2U);
        xSnapshotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), static_cast<uint64_t>(n_) * 2U);
        const uint64_t panelTableOffsetBytes = (static_cast<uint64_t>(n_) * 2U * sizeof(float) + 31U) & ~31ULL;
        panelOffsetsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint32_t*>(workspace) + panelTableOffsetBytes / sizeof(uint32_t),
            CTBMV_PANEL_TABLE_WORDS);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= useCoreNum_ || blockIdx_ * rowsPerCore_ >= n_) {
            return;
        }

        InitBuffers();
        InitWindow();
        LoadPanel();
        ComputeBands();
        Gather(interleaved, yReal, interleaveOffsets, 0U, count * 2U);
        PipeBarrier<PIPE_ALL>();
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            xGm_[static_cast<uint64_t>(outputBegin) * 2U], interleaved, 0, 1, count * 2U * sizeof(float), 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void ComputeBands()
    {
        // In a left-halo layout only the first panel has a triangular prefix.
        // Handle that small boundary in scalar UB accesses so every subsequent
        // vector destination remains 32-byte aligned.
        const uint32_t scalarBoundaryCount = !rightHalo && outputBegin == 0U ? Min(count, (k_ + 7U) & ~7U) : 0U;
        ComputeScalarBoundary(scalarBoundaryCount);
        const uint32_t maxBandDistance = rightHalo ? Min(k_, n_ - outputBegin - 1U) : Min(k_, outputBegin + count - 1U);
        for (uint32_t bandDistance = 0U; bandDistance <= maxBandDistance; ++bandDistance) {
            const uint32_t validOutputBegin =
                rightHalo ? outputBegin : Max(outputBegin + scalarBoundaryCount, bandDistance);
            const uint32_t outputOffset = validOutputBegin - outputBegin;
            const uint32_t validCount = rightHalo ? Min(count, n_ - outputBegin - bandDistance) : count - outputOffset;
            const uint32_t xLogicalBegin = rightHalo ? outputBegin + bandDistance : validOutputBegin - bandDistance;
            const uint32_t xRelativeBegin = xLogicalBegin - xBegin;
            const uint32_t xView = xRelativeBegin & (CTBMV_PANEL_X_VIEWS - 1U);
            const uint32_t xViewOffset = xView * CTBMV_PANEL_MAX_X + xRelativeBegin - xView;

            if (diag_ == ACLBLAS_UNIT && bandDistance == 0U) {
                Add(yReal[outputOffset], yReal[outputOffset], xReal[xViewOffset], validCount);
                Add(yImag[outputOffset], yImag[outputOffset], xImag[xViewOffset], validCount);
                PipeBarrier<PIPE_V>();
                continue;
            }

            ComputeVectorBand(bandDistance, outputOffset, xRelativeBegin, xViewOffset, validCount);
        }
    }

    __aicore__ inline void ComputeVectorBand(
        uint32_t bandDistance, uint32_t outputOffset, uint32_t xRelativeBegin, uint32_t xViewOffset,
        uint32_t validCount)
    {
        uint32_t bandBaseElements = 0U;
        if (noTrans && uplo_ == ACLBLAS_UPPER) {
            bandBaseElements = bandDistance * lda_ + k_ - bandDistance;
        } else if (noTrans) {
            bandBaseElements = xRelativeBegin * lda_ + bandDistance;
        } else if (uplo_ == ACLBLAS_LOWER) {
            bandBaseElements = bandDistance;
        } else {
            bandBaseElements = outputOffset * lda_ + k_ - bandDistance;
        }
        const uint32_t bandBaseBytes = bandBaseElements * 2U * sizeof(float);
        const uint32_t realPhase = bandBaseBytes % CTBMV_COMPACT_GATHER_ALIGNMENT / sizeof(float);
        const uint32_t imagBaseBytes = bandBaseBytes + sizeof(float);
        const uint32_t imagPhase = imagBaseBytes % CTBMV_COMPACT_GATHER_ALIGNMENT / sizeof(float);
        LocalTensor<uint32_t> realColumnOffsets = panelOffsets[CTBMV_PANEL_TILE * (2U + realPhase)];
        LocalTensor<uint32_t> imagColumnOffsets = panelOffsets[CTBMV_PANEL_TILE * (2U + imagPhase)];
        Gather(aReal, aSlab, realColumnOffsets, bandBaseBytes & ~(CTBMV_COMPACT_GATHER_ALIGNMENT - 1U), validCount);
        Gather(aImag, aSlab, imagColumnOffsets, imagBaseBytes & ~(CTBMV_COMPACT_GATHER_ALIGNMENT - 1U), validCount);
        PipeBarrier<PIPE_V>();

        Mul(productReal, aReal, xReal[xViewOffset], validCount);
        Mul(productImag, aReal, xImag[xViewOffset], validCount);
        Mul(crossReal, aImag, xImag[xViewOffset], validCount);
        Mul(crossImag, aImag, xReal[xViewOffset], validCount);
        PipeBarrier<PIPE_V>();

        if (trans_ == ACLBLAS_OP_C) {
            Add(productReal, productReal, crossReal, validCount);
            Sub(productImag, productImag, crossImag, validCount);
        } else {
            Sub(productReal, productReal, crossReal, validCount);
            Add(productImag, productImag, crossImag, validCount);
        }
        PipeBarrier<PIPE_V>();
        Add(yReal[outputOffset], yReal[outputOffset], productReal, validCount);
        Add(yImag[outputOffset], yImag[outputOffset], productImag, validCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeScalarBoundary(uint32_t scalarBoundaryCount)
    {
        if (scalarBoundaryCount == 0U) {
            return;
        }
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (uint32_t output = 0U; output < scalarBoundaryCount; ++output) {
            float sumReal = 0.0F;
            float sumImag = 0.0F;
            const uint32_t outputBandCount = Min(k_, output) + 1U;
            for (uint32_t bandDistance = 0U; bandDistance < outputBandCount; ++bandDistance) {
                const uint32_t xRelative = output - bandDistance;
                const uint32_t xView = xRelative & (CTBMV_PANEL_X_VIEWS - 1U);
                const uint32_t xOffset = xView * CTBMV_PANEL_MAX_X + xRelative - xView;
                const float xRealValue = xReal.GetValue(xOffset);
                const float xImagValue = xImag.GetValue(xOffset);
                if (diag_ == ACLBLAS_UNIT && bandDistance == 0U) {
                    sumReal += xRealValue;
                    sumImag += xImagValue;
                    continue;
                }
                const uint32_t aElement = noTrans ? xRelative * lda_ + bandDistance : output * lda_ + k_ - bandDistance;
                const float aRealValue = aSlab.GetValue(aElement * 2U);
                const float aImagValue = aSlab.GetValue(aElement * 2U + 1U);
                if (trans_ == ACLBLAS_OP_C) {
                    sumReal += aRealValue * xRealValue + aImagValue * xImagValue;
                    sumImag += aRealValue * xImagValue - aImagValue * xRealValue;
                } else {
                    sumReal += aRealValue * xRealValue - aImagValue * xImagValue;
                    sumImag += aRealValue * xImagValue + aImagValue * xRealValue;
                }
            }
            yReal.SetValue(output, sumReal);
            yImag.SetValue(output, sumImag);
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadPanel()
    {
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            aSlab, aGm_[slabGlobalFloatIndex], 0, 1, slabColumnCount * lda_ * 2U * sizeof(float), 0, 0, 0, 0);
        gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
            panelOffsets, panelOffsetsGm_, 0, 1, CTBMV_PANEL_TABLE_WORDS * sizeof(uint32_t), 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();

        // One GM transfer and two gathers produce all phase-aligned views.
        Duplicate(xAos, 0.0F, CTBMV_PANEL_MAX_X * 2U);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        DataCopyExtParams xCopy{1U, xCount * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> xPad{true, 0U, static_cast<uint8_t>(((4U - xCount % 4U) % 4U) * 2U), 0.0F};
        DataCopyPad(xAos, xSnapshotGm_[static_cast<uint64_t>(xBegin) * 2U], xCopy, xPad);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Gather(xReal, xAos, panelOffsets[CTBMV_PANEL_BASE_WORDS], 0U, CTBMV_PANEL_WINDOW_WORDS);
        Gather(
            xImag, xAos, panelOffsets[CTBMV_PANEL_BASE_WORDS + CTBMV_PANEL_WINDOW_WORDS], 0U, CTBMV_PANEL_WINDOW_WORDS);
        PipeBarrier<PIPE_V>();

        Duplicate(yReal, 0.0F, count);
        Duplicate(yImag, 0.0F, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitWindow()
    {
        outputBegin = blockIdx_ * rowsPerCore_;
        count = Min(rowsPerCore_, n_ - outputBegin);
        noTrans = trans_ == ACLBLAS_OP_N;
        rightHalo = (noTrans && uplo_ == ACLBLAS_UPPER) || (!noTrans && uplo_ == ACLBLAS_LOWER);
        leftHalo = rightHalo ? 0U : Min(k_, outputBegin);
        xBegin = outputBegin - leftHalo;
        xCount = rightHalo ? Min(rowsPerCore_ + k_, n_ - outputBegin) : count + leftHalo;
        aColumnBegin = noTrans && uplo_ == ACLBLAS_LOWER ? xBegin : outputBegin;
        slabColumnCount = noTrans ? xCount : count;
        slabGlobalFloatIndex = static_cast<uint64_t>(aColumnBegin) * lda_ * 2U;
    }

    uint32_t outputBegin;
    uint32_t count;
    bool noTrans;
    bool rightHalo;
    uint32_t leftHalo;
    uint32_t xBegin;
    uint32_t xCount;
    uint32_t aColumnBegin;
    uint32_t slabColumnCount;
    uint64_t slabGlobalFloatIndex;

    __aicore__ inline void InitBuffers()
    {
        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        xAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xAosOffset);
        xReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xRealOffset);
        xImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xImagOffset);
        aSlab = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aSlabOffset);
        aReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aRealOffset);
        aImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aImagOffset);
        productReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(productRealOffset);
        productImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(productImagOffset);
        crossReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(crossRealOffset);
        crossImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(crossImagOffset);
        yReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(yRealOffset);
        yImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(yImagOffset);
        interleaved = buffer.GetBuffer<BufferType::ASCEND_UB, float>(interleavedOffset);
        panelOffsets = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(panelOffsetsOffset);
        interleaveOffsets = panelOffsets;
    }

    static constexpr uint32_t xAosOffset = 0U;
    static constexpr uint32_t xRealOffset = xAosOffset + CTBMV_PANEL_MAX_X * 2U * sizeof(float);
    static constexpr uint32_t xImagOffset = xRealOffset + CTBMV_PANEL_X_VIEWS * CTBMV_PANEL_MAX_X * sizeof(float);
    static constexpr uint32_t aSlabOffset = xImagOffset + CTBMV_PANEL_X_VIEWS * CTBMV_PANEL_MAX_X * sizeof(float);
    static constexpr uint32_t aRealOffset = aSlabOffset + CTBMV_PANEL_MAX_A_FLOATS * sizeof(float);
    static constexpr uint32_t aImagOffset = aRealOffset + CTBMV_PANEL_TILE * sizeof(float);
    static constexpr uint32_t productRealOffset = aImagOffset + CTBMV_PANEL_TILE * sizeof(float);
    static constexpr uint32_t productImagOffset = productRealOffset + CTBMV_PANEL_TILE * sizeof(float);
    static constexpr uint32_t crossRealOffset = productImagOffset + CTBMV_PANEL_TILE * sizeof(float);
    static constexpr uint32_t crossImagOffset = crossRealOffset + CTBMV_PANEL_TILE * sizeof(float);
    static constexpr uint32_t yRealOffset = crossImagOffset + CTBMV_PANEL_TILE * sizeof(float);
    static constexpr uint32_t yImagOffset = yRealOffset + CTBMV_PANEL_TILE * sizeof(float);
    static constexpr uint32_t interleavedOffset = yImagOffset + CTBMV_PANEL_TILE * sizeof(float);
    static constexpr uint32_t panelOffsetsOffset = interleavedOffset + CTBMV_PANEL_TILE * 2U * sizeof(float);

    LocalTensor<float> xAos;
    LocalTensor<float> xReal;
    LocalTensor<float> xImag;
    LocalTensor<float> aSlab;
    LocalTensor<float> aReal;
    LocalTensor<float> aImag;
    LocalTensor<float> productReal;
    LocalTensor<float> productImag;
    LocalTensor<float> crossReal;
    LocalTensor<float> crossImag;
    LocalTensor<float> yReal;
    LocalTensor<float> yImag;
    LocalTensor<float> interleaved;
    LocalTensor<uint32_t> panelOffsets;
    LocalTensor<uint32_t> interleaveOffsets;

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t Max(uint32_t lhs, uint32_t rhs) const { return lhs > rhs ? lhs : rhs; }

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> xSnapshotGm_;
    GlobalTensor<uint32_t> panelOffsetsGm_;
};

class CtbmvSmallScalarAiv : public CtbmvShape {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, const CtbmvTilingData& tiling)
    {
        InitShape(tiling);
        incx_ = tiling.incx;
        const uint64_t absIncx = incx_ < 0 ? static_cast<uint64_t>(-incx_) : static_cast<uint64_t>(incx_);
        xCount_ = n_ == 0U ? 0U : static_cast<uint64_t>(n_ - 1U) * absIncx + 1U;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2U);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), xCount_ * 2U);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ != 0U) {
            return;
        }
        InitBuffers();
        const bool cacheA = lda_ <= 64U;
        if (cacheA) {
            gm_to_ub_align<ArchType::ASCEND_V220, float>(aCached, aGm_, 0, 1, n_ * lda_ * 8U, 0, 0, 0, 0);
        }
        gm_to_ub_align<ArchType::ASCEND_V220, float>(snapshot, xGm_, 0, 1, xCount_ * 2U * sizeof(float), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        if (incx_ != 1 && incx_ != -1) {
            for (uint64_t i = 0U; i < xCount_ * 2U; ++i) {
                output.SetValue(i, snapshot.GetValue(i));
            }
        }

        for (uint32_t row = 0U; row < n_; ++row) {
            ComputeRow(row, cacheA);
        }

        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(xGm_, output, 0, 1, xCount_ * 2U * sizeof(float), 0, 0, 0, 0);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    __aicore__ inline void AccumulateElement(uint32_t row, uint32_t col, bool cacheA, float& sumReal, float& sumImag)
    {
        float aReal = 1.0F;
        float aImag = 0.0F;
        if (!(diag_ == ACLBLAS_UNIT && row == col)) {
            const uint32_t matrixRow = trans_ == ACLBLAS_OP_N ? row : col;
            const uint32_t matrixCol = trans_ == ACLBLAS_OP_N ? col : row;
            const uint32_t bandRow = uplo_ == ACLBLAS_UPPER ? k_ + matrixRow - matrixCol : matrixRow - matrixCol;
            const uint64_t index = (static_cast<uint64_t>(matrixCol) * lda_ + bandRow) * 2U;
            aReal = cacheA ? aCached.GetValue(index) : aGm_.GetValue(index);
            aImag = cacheA ? aCached.GetValue(index + 1U) : aGm_.GetValue(index + 1U);
            if (trans_ == ACLBLAS_OP_C) {
                aImag = -aImag;
            }
        }
        const uint64_t xIndex = VectorIndex(col);
        const float xReal = snapshot.GetValue(xIndex * 2U);
        const float xImag = snapshot.GetValue(xIndex * 2U + 1U);
        // One conservative finite-range test covers NaN, Inf and
        // overflow-sensitive inputs. NaN makes <= false as required.
        const float magnitude = Abs(aReal) + Abs(aImag) + Abs(xReal) + Abs(xImag);
        const bool specialValues = !(magnitude <= 1.0e30F);
        if (specialValues) {
            const float productReal = SafeAdd(SafeMultiply(aReal, xReal), -SafeMultiply(aImag, xImag));
            const float productImag = SafeAdd(SafeMultiply(aReal, xImag), SafeMultiply(aImag, xReal));
            sumReal = SafeAdd(sumReal, productReal);
            sumImag = SafeAdd(sumImag, productImag);
        } else {
            sumReal += aReal * xReal - aImag * xImag;
            sumImag += aReal * xImag + aImag * xReal;
        }
    }

    __aicore__ inline void ComputeRow(uint32_t row, bool cacheA)
    {
        uint32_t colBegin = 0U;
        uint32_t colEnd = 0U;
        GetRowRange(row, colBegin, colEnd);

        float sumReal = 0.0F;
        float sumImag = 0.0F;
        for (uint32_t col = colBegin; col <= colEnd; ++col) {
            AccumulateElement(row, col, cacheA, sumReal, sumImag);
        }
        const uint64_t outputIndex = VectorIndex(row);
        output.SetValue(outputIndex * 2U, sumReal);
        output.SetValue(outputIndex * 2U + 1U, sumImag);
    }

    __aicore__ inline void InitBuffers()
    {
        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        snapshot = buffer.GetBuffer<BufferType::ASCEND_UB, float>(snapshotOffset);
        output = buffer.GetBuffer<BufferType::ASCEND_UB, float>(outputOffset);
        aCached = buffer.GetBuffer<BufferType::ASCEND_UB, float>(4096U);
    }

    static constexpr uint32_t snapshotOffset = 0U;
    static constexpr uint32_t outputOffset = 512U * sizeof(float);

    LocalTensor<float> snapshot;
    LocalTensor<float> output;
    LocalTensor<float> aCached;

    __aicore__ inline uint32_t FloatBits(float value) const
    {
        union {
            float number;
            uint32_t bits;
        } converter;
        converter.number = value;
        return converter.bits;
    }

    __aicore__ inline float BitsFloat(uint32_t bits) const
    {
        union {
            float number;
            uint32_t bits;
        } converter;
        converter.bits = bits;
        return converter.number;
    }

    __aicore__ inline bool IsNan(float value) const { return (FloatBits(value) & 0x7FFFFFFFU) > 0x7F800000U; }

    __aicore__ inline bool IsInf(float value) const { return (FloatBits(value) & 0x7FFFFFFFU) == 0x7F800000U; }

    __aicore__ inline float Abs(float value) const { return value < 0.0F ? -value : value; }

    __aicore__ inline float MakeNan() const { return BitsFloat(0x7FC00000U); }

    __aicore__ inline float MakeInf(bool negative) const { return BitsFloat(negative ? 0xFF800000U : 0x7F800000U); }

    __aicore__ inline float SafeMultiply(float lhs, float rhs) const
    {
        constexpr float maxFloat = 3.402823466e+38F;
        if (IsNan(lhs) || IsNan(rhs)) {
            return MakeNan();
        }
        const bool negative = ((FloatBits(lhs) ^ FloatBits(rhs)) & 0x80000000U) != 0U;
        const bool lhsInf = IsInf(lhs);
        const bool rhsInf = IsInf(rhs);
        if ((lhsInf && rhs == 0.0F) || (rhsInf && lhs == 0.0F)) {
            return MakeNan();
        }
        if (lhsInf || rhsInf) {
            return MakeInf(negative);
        }
        if (lhs == 0.0F || rhs == 0.0F) {
            return lhs * rhs;
        }
        if (Abs(lhs) > maxFloat / Abs(rhs)) {
            return MakeInf(negative);
        }
        return lhs * rhs;
    }

    __aicore__ inline float SafeAdd(float lhs, float rhs) const
    {
        constexpr float maxFloat = 3.402823466e+38F;
        if (IsNan(lhs) || IsNan(rhs)) {
            return MakeNan();
        }
        const bool lhsInf = IsInf(lhs);
        const bool rhsInf = IsInf(rhs);
        if (lhsInf && rhsInf && ((FloatBits(lhs) ^ FloatBits(rhs)) & 0x80000000U) != 0U) {
            return MakeNan();
        }
        if (lhsInf) {
            return lhs;
        }
        if (rhsInf) {
            return rhs;
        }
        if (rhs > 0.0F && lhs > maxFloat - rhs) {
            return MakeInf(false);
        }
        if (rhs < 0.0F && lhs < -maxFloat - rhs) {
            return MakeInf(true);
        }
        return lhs + rhs;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t SaturatingSub(uint32_t lhs, uint32_t rhs) const { return lhs > rhs ? lhs - rhs : 0U; }

    __aicore__ inline uint64_t VectorIndex(uint32_t logicalIndex) const
    {
        const uint64_t absIncx = incx_ < 0 ? static_cast<uint64_t>(-incx_) : static_cast<uint64_t>(incx_);
        return incx_ >= 0 ? static_cast<uint64_t>(logicalIndex) * absIncx :
                            static_cast<uint64_t>(n_ - 1U - logicalIndex) * absIncx;
    }

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;

    int64_t incx_ = 1;
    uint64_t xCount_ = 0U;
};

} // namespace Ctbmv
#endif
