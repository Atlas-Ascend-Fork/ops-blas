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

class CtbmvTransposeFastAiv : public CtbmvVectorBuffers {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        InitShape(tiling);
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2U);
        const bool directNoTranspose = tiling.path == CtbmvPath::Row || tiling.path == CtbmvPath::Transpose;
        xGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(directNoTranspose ? workspace : x), static_cast<uint64_t>(n_) * 2U);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(directNoTranspose ? x : workspace), static_cast<uint64_t>(n_) * 2U);
    }

    __aicore__ inline void Process()
    {
        ProcessRange(blockIdx_ * rowsPerCore_, Min((blockIdx_ + 1U) * rowsPerCore_, n_));
    }

    __aicore__ inline void ProcessRange(uint32_t rowBegin, uint32_t rowEnd)
    {
        if (useCoreNum_ == 0U || blockIdx_ >= useCoreNum_ || rowBegin >= rowEnd) {
            return;
        }
        SetMaskNorm();
        ResetMask();

        InitBuffers();
        for (uint32_t i = 0U; i < CTBMV_COMPACT_GATHER_ALIGNMENT / sizeof(uint32_t); ++i) {
            aRealPattern.SetValue(i, 0x01010101U);
            aImagPattern.SetValue(i, 0x02020202U);
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t chunkBegin = rowBegin; chunkBegin < rowEnd; chunkBegin += resultTileComplex) {
            const uint32_t chunkCount = Min(resultTileComplex, rowEnd - chunkBegin);
            for (uint32_t localRow = 0U; localRow < chunkCount; ++localRow) {
                ComputeRow(chunkBegin, localRow);
            }

            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
            ub_to_gm_align<ArchType::ASCEND_V220, float>(
                outputGm_[static_cast<uint64_t>(chunkBegin) * 2U], resultAos, 0, 1, chunkCount * 2U * sizeof(float), 0,
                0, 0, 0);
            PipeBarrier<PIPE_ALL>();
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(outputGm_);
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void ComputeRow(uint32_t chunkBegin, uint32_t localRow)
    {
        DescribeRow(chunkBegin, localRow);
        LoadRow();
        ApplyDiagonal();
        SplitRow();
        MultiplyRow();
        ReduceRow(localRow);
    }

    __aicore__ inline void ReduceRow(uint32_t localRow)
    {
        ReduceVectorPartials(reduceReal, outReal, count);
        PipeBarrier<PIPE_V>();
        ReduceVectorPartials(reduceImag, outImag, count);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        const uint32_t fullRepeats = count / 64U;
        const bool hasTail = count % 64U != 0U;
        float sumReal = hasTail ? reduceReal.GetValue(24U) : 0.0F;
        float sumImag = hasTail ? reduceImag.GetValue(24U) : 0.0F;
        for (uint32_t i = 0U; i < fullRepeats; ++i) {
            sumReal += reduceReal.GetValue(i);
            sumImag += reduceImag.GetValue(i);
        }
        resultAos.SetValue(localRow * 2U, sumReal);
        resultAos.SetValue(localRow * 2U + 1U, sumImag);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void MultiplyRow()
    {
        const uint32_t repeats = (count + 63U) / 64U;
        mul_v<ArchType::ASCEND_V220, float>(outReal, aReal, xReal, repeats, 1, 1, 1, 8, 8, 8);
        mul_v<ArchType::ASCEND_V220, float>(tmpReal, aImag, xImag, repeats, 1, 1, 1, 8, 8, 8);
        mul_v<ArchType::ASCEND_V220, float>(outImag, aReal, xImag, repeats, 1, 1, 1, 8, 8, 8);
        mul_v<ArchType::ASCEND_V220, float>(tmpImag, aImag, xReal, repeats, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        if (trans_ != ACLBLAS_OP_C) {
            sub_v<ArchType::ASCEND_V220, float>(outReal, outReal, tmpReal, repeats, 1, 1, 1, 8, 8, 8);
            add_v<ArchType::ASCEND_V220, float>(outImag, outImag, tmpImag, repeats, 1, 1, 1, 8, 8, 8);
        } else {
            add_v<ArchType::ASCEND_V220, float>(outReal, outReal, tmpReal, repeats, 1, 1, 1, 8, 8, 8);
            sub_v<ArchType::ASCEND_V220, float>(outImag, outImag, tmpImag, repeats, 1, 1, 1, 8, 8, 8);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SplitRow()
    {
        if (noTrans) {
            const uint16_t gatherRepeats = static_cast<uint16_t>((count + 7U) / 8U);
            uint64_t gatheredCount = 0U;
            GatherMask(aReal, aAos, aRealPattern, true, 64U, {1U, gatherRepeats, 8U, 0U}, gatheredCount);
            gatheredCount = 0U;
            GatherMask(aImag, aAos, aImagPattern, true, 64U, {1U, gatherRepeats, 8U, 0U}, gatheredCount);
            PipeBarrier<PIPE_V>();
            // GatherMask leaves its selection mask active.  The raw
            // mul_v calls below require the normal full vector mask.
            SetMaskNorm();
            ResetMask();
        } else {
            SplitComplexVector(aReal, aImag, aAos, count);
        }
        SplitComplexVector(xReal, xImag, xAos, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyDiagonal()
    {
        if (diag_ == ACLBLAS_UNIT) {
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
            const uint32_t diagIndex =
                noTrans ? (uplo_ == ACLBLAS_UPPER ? 0U : count - 1U) : (uplo_ == ACLBLAS_UPPER ? count - 1U : 0U);
            const uint32_t diagOffset = diagIndex * (noTrans ? CTBMV_COMPACT_PADDED_COMPLEX_FLOATS : 2U);
            aAos.SetValue(diagOffset, 1.0F);
            aAos.SetValue(diagOffset + 1U, 0.0F);
            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        } else {
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        }
    }

    __aicore__ inline void LoadRow()
    {
        if (noTrans) {
            const uint32_t sourceStrideBytes = lda_ > 1U ? (lda_ - 2U) * 2U * sizeof(float) : 0U;
            DataCopyExtParams aCopyParams{
                static_cast<uint16_t>(count), static_cast<uint32_t>(2U * sizeof(float)), sourceStrideBytes, 0U, 0U};
            DataCopyPadExtParams<float> aPadParams{
                true, 0U, static_cast<uint8_t>(CTBMV_COMPACT_PADDED_COMPLEX_FLOATS - 2U), 0.0F};
            DataCopyPad(aAos, aGm_[aOffset], aCopyParams, aPadParams);
        } else {
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                aAos, aGm_[aOffset], 0, 1, count * 2U * sizeof(float), 0, 0, 0, 0);
        }
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            xAos, xGm_[static_cast<uint64_t>(colBegin) * 2U], 0, 1, count * 2U * sizeof(float), 0, 0, 0, 0);
    }

    __aicore__ inline void DescribeRow(uint32_t chunkBegin, uint32_t localRow)
    {
        row = chunkBegin + localRow;
        noTrans = trans_ == ACLBLAS_OP_N;
        colBegin = noTrans ? (uplo_ == ACLBLAS_UPPER ? row : SaturatingSub(row, k_)) :
                             (uplo_ == ACLBLAS_UPPER ? SaturatingSub(row, k_) : row);
        colEnd = noTrans ? (uplo_ == ACLBLAS_UPPER ? Min(n_ - 1U, row + k_) : row) :
                           (uplo_ == ACLBLAS_UPPER ? row : Min(n_ - 1U, row + k_));
        count = colEnd - colBegin + 1U;
        bandRowBegin = noTrans ? (uplo_ == ACLBLAS_UPPER ? k_ : row - colBegin) :
                                 (uplo_ == ACLBLAS_UPPER ? k_ - (row - colBegin) : 0U);
        aOffset = noTrans ? (static_cast<uint64_t>(colBegin) * lda_ + bandRowBegin) * 2U :
                            (static_cast<uint64_t>(row) * lda_ + bandRowBegin) * 2U;
    }

    uint32_t row;
    bool noTrans;
    uint32_t colBegin;
    uint32_t colEnd;
    uint32_t count;
    uint32_t bandRowBegin;
    uint64_t aOffset;

    __aicore__ inline void InitBuffers()
    {
        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        aAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aAosOffset);
        xAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xAosOffset);
        aReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aRealOffset);
        aImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aImagOffset);
        xReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xRealOffset);
        xImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xImagOffset);
        outReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(outRealOffset);
        outImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(outImagOffset);
        tmpReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(tmpRealOffset);
        tmpImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(tmpImagOffset);
        reduceReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(reduceRealOffset);
        reduceImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(reduceImagOffset);
        resultAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(resultOffset);
        aRealPattern = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(aRealPatternOffset);
        aImagPattern = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(aImagPatternOffset);
    }

    static constexpr uint32_t resultTileComplex = 256U;
    static constexpr uint32_t aAosOffset = 0U;
    static constexpr uint32_t xAosOffset =
        aAosOffset + CTBMV_FAST_TILE_COMPLEX * CTBMV_COMPACT_PADDED_COMPLEX_FLOATS * sizeof(float);
    static constexpr uint32_t aRealOffset = xAosOffset + CTBMV_FAST_TILE_COMPLEX * 2U * sizeof(float);
    static constexpr uint32_t aImagOffset = aRealOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t xRealOffset = aImagOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t xImagOffset = xRealOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t outRealOffset = xImagOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t outImagOffset = outRealOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t tmpRealOffset = outImagOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t tmpImagOffset = tmpRealOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t reduceSlotFloats = 32U;
    static constexpr uint32_t reduceRealOffset = tmpImagOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t reduceImagOffset = reduceRealOffset + reduceSlotFloats * sizeof(float);
    static constexpr uint32_t resultOffset = reduceImagOffset + reduceSlotFloats * sizeof(float);
    static constexpr uint32_t aRealPatternOffset = resultOffset + resultTileComplex * 2U * sizeof(float);
    static constexpr uint32_t aImagPatternOffset = aRealPatternOffset + CTBMV_COMPACT_GATHER_ALIGNMENT;

    LocalTensor<float> reduceReal;
    LocalTensor<float> reduceImag;
    LocalTensor<float> resultAos;
    LocalTensor<uint32_t> aRealPattern;
    LocalTensor<uint32_t> aImagPattern;

    __aicore__ inline void ReduceVectorPartials(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count) const
    {
        const uint32_t fullRepeats = count / 64U;
        const uint32_t tail = count % 64U;
        if (fullRepeats != 0U) {
            WholeReduceSum<float, true>(dst, src, 64U, fullRepeats, 1U, 1U, 8U);
        }
        if (tail != 0U) {
            WholeReduceSum<float, true>(dst[24U], src[fullRepeats * 64U], tail, 1U, 1U, 1U, 8U);
        }
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t SaturatingSub(uint32_t lhs, uint32_t rhs) const { return lhs > rhs ? lhs - rhs : 0U; }

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> outputGm_;
};

} // namespace Ctbmv
#endif
