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

class CtbmvTransposeShared : public CtbmvShape {
protected:
    template <typename Implementation>
    __aicore__ inline bool ProcessTiles(Implementation& implementation)
    {
        if (blockIdx_ >= useCoreNum_) {
            return false;
        }
        implementation.InitBuffers();
        implementation.InitWindow();
        for (uint32_t firstColumn = blockIdx_ * outputTile; firstColumn < n_; firstColumn += useCoreNum_ * outputTile) {
            implementation.ProcessTile(firstColumn);
        }
        return true;
    }

    template <uint32_t Entries, uint32_t MaxX>
    __aicore__ inline void InitInputBuffers(AsdopsBuffer<ArchType::ASCEND_V220>& buffer)
    {
        constexpr uint32_t realOffset = Entries * 8U;
        constexpr uint32_t imagOffset = realOffset + Entries * 4U;
        constexpr uint32_t aosOffset = imagOffset + Entries * 4U;
        constexpr uint32_t xRealOffset = aosOffset + MaxX * 8U;
        constexpr uint32_t xImagOffset = xRealOffset + Entries * 4U;
        aAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        aReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(realOffset);
        aImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(imagOffset);
        xAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aosOffset);
        xReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xRealOffset);
        xImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xImagOffset);
    }

    __aicore__ inline void LoadColumns(uint32_t firstColumn)
    {
        batch = Min(outputTile, n_ - firstColumn);
        prefix = uplo_ == ACLBLAS_UPPER && firstColumn < k_ ? k_ - firstColumn : 0U;
        xBegin = uplo_ == ACLBLAS_UPPER ? (firstColumn >= k_ ? firstColumn - k_ : 0U) : firstColumn;
        xCount = Min(innerCount + batch - 1U - prefix, n_ - xBegin);
        paddedXCount = alignedInner + batch - 1U;
        blockAlignedInner = (innerCount + 3U) & ~3U;
        xLeft = prefix % 4U;
        xRight = (4U - (xCount + xLeft) % 4U) % 4U;
        // Tail lanes are masked in the multiply/reduction loop; only
        // logical A entries and triangular boundary slots are consumed.
        Duplicate(xAos, 0.0F, (paddedXCount + 3U) / 4U * 8U);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams aCopyParams{
            static_cast<uint16_t>(batch), innerCount * 8U, (lda_ - innerCount) * 8U,
            (alignedInner - blockAlignedInner) / 4U, 0U};
        DataCopyPadExtParams<float> aPadParams{
            true, 0U, static_cast<uint8_t>((blockAlignedInner - innerCount) * 2U), 0.0F};
        DataCopyPad(aAos, aGm_[static_cast<uint64_t>(firstColumn) * lda_ * 2U], aCopyParams, aPadParams);
        DataCopyExtParams xCopyParams{1U, xCount * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> xPadParams{
            true, static_cast<uint8_t>(xLeft * 2U), static_cast<uint8_t>(xRight * 2U), 0.0F};
        DataCopyPad(xAos[prefix / 4U * 8U], xGm_[static_cast<uint64_t>(xBegin) * 2U], xCopyParams, xPadParams);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    }

    __aicore__ inline void ClearTriangle(uint32_t firstColumn)
    {
        // Band storage contains unused triangular slots. Clear those A
        // slots, including any NaN sentinels, before complex arithmetic.
        if (uplo_ == ACLBLAS_UPPER && firstColumn < k_) {
            for (uint32_t b = 0U; b < batch && firstColumn + b < k_; ++b) {
                const uint32_t invalid = k_ - firstColumn - b;
                Duplicate(aReal[b * alignedInner], 0.0F, invalid);
                Duplicate(aImag[b * alignedInner], 0.0F, invalid);
            }
        } else if (uplo_ == ACLBLAS_LOWER && firstColumn + batch + k_ > n_) {
            for (uint32_t b = 0U; b < batch; ++b) {
                const uint32_t valid = Min(innerCount, n_ - firstColumn - b);
                if (valid == innerCount) {
                    continue;
                }
                const uint32_t head = valid / 64U * 64U;
                const uint32_t tail = valid % 64U;
                uint64_t mask[2] = {~0ULL << tail, 0ULL};
                Duplicate(aReal[b * alignedInner + head], 0.0F, mask, 1U, 1U, 8U);
                Duplicate(aImag[b * alignedInner + head], 0.0F, mask, 1U, 1U, 8U);
                if (head + 64U < alignedInner) {
                    Duplicate(aReal[b * alignedInner + head + 64U], 0.0F, alignedInner - head - 64U);
                    Duplicate(aImag[b * alignedInner + head + 64U], 0.0F, alignedInner - head - 64U);
                }
            }
        }
        PipeBarrier<PIPE_V>();
        if (diag_ == ACLBLAS_UNIT) {
            const uint32_t diagonal = uplo_ == ACLBLAS_UPPER ? k_ : 0U;
            uint64_t mask[2] = {1ULL << (diagonal % 64U), 0ULL};
            Duplicate(aReal[diagonal / 64U * 64U], 1.0F, mask, batch, 1U, alignedInner / 8U);
            Duplicate(aImag[diagonal / 64U * 64U], 0.0F, mask, batch, 1U, alignedInner / 8U);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void AccumulateDot()
    {
        Muls(xNeg, trans_ == ACLBLAS_OP_C ? xReal : xImag, -1.0F, expandedCount);
        Duplicate(productReal, 0.0F, batch * 64U);
        Duplicate(productImag, 0.0F, batch * 64U);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        const uint32_t stride = alignedInner / 8U;
        for (uint32_t inner = 0U; inner < innerCount; inner += 64U) {
            const uint32_t lanes = Min(64U, innerCount - inner);
            SetVectorMask<float>(0ULL, lanes == 64U ? ~0ULL : (1ULL << lanes) - 1ULL);
            vmla(
                reinterpret_cast<__ubuf__ float*>(productReal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(xReal[inner].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(aReal[inner].GetPhyAddr()), batch, 1, 1, 1, 8, stride, stride);
            vmla(
                reinterpret_cast<__ubuf__ float*>(productImag.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(xImag[inner].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(aReal[inner].GetPhyAddr()), batch, 1, 1, 1, 8, stride, stride);
            PipeBarrier<PIPE_V>();
            auto realImagTerm = trans_ == ACLBLAS_OP_C ? xImag[inner] : xNeg[inner];
            auto imagImagTerm = trans_ == ACLBLAS_OP_C ? xNeg[inner] : xReal[inner];
            vmla(
                reinterpret_cast<__ubuf__ float*>(productReal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(realImagTerm.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(aImag[inner].GetPhyAddr()), batch, 1, 1, 1, 8, stride, stride);
            vmla(
                reinterpret_cast<__ubuf__ float*>(productImag.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(imagImagTerm.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(aImag[inner].GetPhyAddr()), batch, 1, 1, 1, 8, stride, stride);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ReduceDot()
    {
        ResetMask();
        cadd_v<ArchType::ASCEND_V220, float>(xAos, productReal, batch, 1, 1, 8);
        cadd_v<ArchType::ASCEND_V220, float>(xAos[64U], productImag, batch, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Gather(outputAos, xAos, interleave, 0U, batch * 2U);
    }

    uint32_t expandedCount;
    uint32_t batch;
    uint32_t prefix;
    uint32_t xBegin;
    uint32_t xCount;
    uint32_t paddedXCount;
    uint32_t blockAlignedInner;
    uint32_t xLeft;
    uint32_t xRight;
    uint32_t innerCount;
    uint32_t alignedInner;
    uint32_t fairRows;
    uint32_t outputTile;
    uint32_t usedEntries;
    LocalTensor<float> aAos;
    LocalTensor<float> aReal;
    LocalTensor<float> aImag;
    LocalTensor<float> xAos;
    LocalTensor<float> xReal;
    LocalTensor<float> xImag;
    LocalTensor<float> xNeg;
    LocalTensor<float> productReal;
    LocalTensor<float> productImag;
    LocalTensor<uint32_t> indexReal;
    LocalTensor<uint32_t> interleave;
    LocalTensor<float> outputAos;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<uint32_t> offsetsGm_;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
};

class CtbmvTransposeNarrowAiv : public CtbmvTransposeShared {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        InitShape(tiling);
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2U);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), n_ * 2U);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), n_ * 2U);
        const uint32_t offset = (n_ * 8U + 31U) & ~31U;
        offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace + offset), 8320U);
    }

    __aicore__ inline void Process()
    {
        if (!ProcessTiles(*this)) {
            return;
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(outputGm_);
        PipeBarrier<PIPE_ALL>();
    }

private:
    friend class CtbmvTransposeShared;

    __aicore__ inline void ProcessTile(uint32_t firstColumn)
    {
        LoadColumns(firstColumn);
        SplitColumns();
        ClearTriangle(firstColumn);
        AccumulateDot();
        ReduceDot();
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            outputGm_[static_cast<uint64_t>(firstColumn) * 2U], outputAos, 0, 1, batch * 8U, 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SplitColumns()
    {
        expandedCount = batch * alignedInner;
        SplitComplexVector(aReal, aImag, aAos, expandedCount);
        Gather(xReal, xAos, indexReal, 0U, expandedCount);
        Gather(xImag, xAos, indexImag, 0U, expandedCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitWindow()
    {
        innerCount = k_ + 1U;
        alignedInner = (innerCount + 63U) & ~63U;
        fairRows = ((n_ + useCoreNum_ - 1U) / useCoreNum_ + 3U) & ~3U;
        outputTile = Min((entries / alignedInner) & ~3U, fairRows);
        usedEntries = outputTile * alignedInner;
        if (usedEntries <= 1024U) {
            gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(indexReal, offsetsGm_, 0, 1, usedEntries * 4U, 0, 0, 0, 0);
            gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
                indexImag, offsetsGm_[entries], 0, 1, usedEntries * 4U, 0, 0, 0, 0);
            gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
                interleave, offsetsGm_[entries * 2U], 0, 1, outputTile * 8U, 0, 0, 0, 0);
        } else {
            gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
                indexReal, offsetsGm_, 0, 1, entries * 8U + 512U, 0, 0, 0, 0);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitBuffers()
    {
        static_assert(outputOffset + 512U <= 192U * 1024U, "transpose UB budget");
        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        InitInputBuffers<entries, maxX>(buffer);
        xNeg = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xNegOffset);
        productReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(productRealOffset);
        productImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(productImagOffset);
        indexReal = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(indexRealOffset);
        indexImag = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(indexImagOffset);
        interleave = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(interleaveOffset);
        outputAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(outputOffset);
    }

    static constexpr uint32_t entries = 4096U;
    static constexpr uint32_t maxX = 1152U;
    static constexpr uint32_t aRealOffset = entries * 8U;
    static constexpr uint32_t aImagOffset = aRealOffset + entries * 4U;
    static constexpr uint32_t xAosOffset = aImagOffset + entries * 4U;
    static constexpr uint32_t xRealOffset = xAosOffset + maxX * 8U;
    static constexpr uint32_t xImagOffset = xRealOffset + entries * 4U;
    static constexpr uint32_t xNegOffset = xImagOffset + entries * 4U;
    static constexpr uint32_t productRealOffset = xNegOffset + entries * 4U;
    static constexpr uint32_t productImagOffset = productRealOffset + entries * 4U;
    static constexpr uint32_t indexRealOffset = productImagOffset + entries * 4U;
    static constexpr uint32_t indexImagOffset = indexRealOffset + entries * 4U;
    static constexpr uint32_t interleaveOffset = indexImagOffset + entries * 4U;
    static constexpr uint32_t outputOffset = interleaveOffset + 512U;

    LocalTensor<uint32_t> indexImag;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
};

class CtbmvTransposeBatchAiv : public CtbmvTransposeShared {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        InitShape(tiling);
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2U);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), n_ * 2U);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), n_ * 2U);
        const uint32_t offset = (n_ * 8U + 31U) & ~31U;
        offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace + offset), 6272U);
    }

    __aicore__ inline void Process() { ProcessTiles(*this); }

private:
    friend class CtbmvTransposeShared;

    __aicore__ inline void ProcessTile(uint32_t firstColumn)
    {
        LoadColumns(firstColumn);
        SplitColumns();
        ClearTriangle(firstColumn);
        if (alignedInner <= 512U) {
            BlockDot();
        } else {
            AccumulateDot();
            ReduceDot();
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            outputGm_[static_cast<uint64_t>(firstColumn) * 2U], outputAos, 0, 1, batch * 8U, 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BlockDot()
    {
        // Materialize contiguous products, reduce eight lanes first,
        // then one <=64-lane reduction per output. This removes the
        // loop-carried accumulation dependency across inner chunks.
        const uint32_t tail = innerCount % 64U;
        if (tail != 0U) {
            uint64_t mask[2] = {~0ULL << tail, 0ULL};
            Duplicate(aReal[innerCount / 64U * 64U], 0.0F, mask, batch, 1U, alignedInner / 8U);
            Duplicate(aImag[innerCount / 64U * 64U], 0.0F, mask, batch, 1U, alignedInner / 8U);
            PipeBarrier<PIPE_V>();
        }
        auto qr = aAos;
        auto qi = aAos[entries];
        Mul(qr, xReal, aReal, expandedCount);
        Mul(qi, xImag, aReal, expandedCount);
        PipeBarrier<PIPE_V>();
        Mul(aReal, xImag, aImag, expandedCount);
        PipeBarrier<PIPE_V>();
        if (trans_ == ACLBLAS_OP_C) {
            Add(qr, qr, aReal, expandedCount);
        } else {
            Sub(qr, qr, aReal, expandedCount);
        }
        Mul(xImag, xReal, aImag, expandedCount);
        PipeBarrier<PIPE_V>();
        if (trans_ == ACLBLAS_OP_C) {
            Sub(qi, qi, xImag, expandedCount);
        } else {
            Add(qi, qi, xImag, expandedCount);
        }
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float>(xAos, qr, expandedCount / 64U, 64U, 1U, 1U, 8U);
        BlockReduceSum<float>(xAos[1024U], qi, expandedCount / 64U, 64U, 1U, 1U, 8U);
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float, true>(aAos, xAos, alignedInner / 8U, batch, 1U, 1U, alignedInner / 64U);
        WholeReduceSum<float, true>(aAos[64U], xAos[1024U], alignedInner / 8U, batch, 1U, 1U, alignedInner / 64U);
        PipeBarrier<PIPE_V>();
        Gather(outputAos, aAos, interleave, 0U, batch * 2U);
    }

    __aicore__ inline void SplitColumns()
    {
        expandedCount = batch * alignedInner;
        SplitComplexVector(aReal, aImag, aAos, expandedCount);
        PipeBarrier<PIPE_V>();
        SplitComplexVector(xWindowReal, xWindowImag, xAos, (paddedXCount + 31U) & ~31U);
        PipeBarrier<PIPE_V>();
        Gather(xReal, xWindowReal, indexReal, 0U, expandedCount);
        Gather(xImag, xWindowImag, indexReal, 0U, expandedCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitWindow()
    {
        innerCount = k_ + 1U;
        alignedInner = (innerCount + 63U) & ~63U;
        fairRows = ((n_ + useCoreNum_ - 1U) / useCoreNum_ + 3U) & ~3U;
        outputTile = Min(64U, Min((entries / alignedInner) & ~3U, fairRows));
        usedEntries = outputTile * alignedInner;
        // A AoS storage is dead after the split. Reuse it first for the x
        // planar window, then for negated x and both reduction accumulators.
        xWindowReal = aAos;
        xWindowImag = aAos[maxX];
        xNeg = aAos;
        productReal = aAos[usedEntries];
        productImag = aAos[usedEntries + outputTile * 64U];
        gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(indexReal, offsetsGm_, 0, 1, usedEntries * 4U, 0, 0, 0, 0);
        gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
            interleave, offsetsGm_[entries], 0, 1, outputTile * 8U, 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
    }

    LocalTensor<float> xWindowReal;
    LocalTensor<float> xWindowImag;

    __aicore__ inline void InitBuffers()
    {
        static_assert(outputOffset + 512U <= 192U * 1024U, "transpose UB budget");
        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        InitInputBuffers<entries, maxX>(buffer);
        indexReal = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(indexRealOffset);
        interleave = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(interleaveOffset);
        outputAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(outputOffset);
    }

    static constexpr uint32_t entries = 6144U;
    static constexpr uint32_t maxX = 1152U;
    static constexpr uint32_t aRealOffset = entries * 8U;
    static constexpr uint32_t aImagOffset = aRealOffset + entries * 4U;
    static constexpr uint32_t xAosOffset = aImagOffset + entries * 4U;
    static constexpr uint32_t xRealOffset = xAosOffset + maxX * 8U;
    static constexpr uint32_t xImagOffset = xRealOffset + entries * 4U;
    static constexpr uint32_t indexRealOffset = xImagOffset + entries * 4U;
    static constexpr uint32_t interleaveOffset = indexRealOffset + entries * 4U;
    static constexpr uint32_t outputOffset = interleaveOffset + 512U;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
};

} // namespace Ctbmv
#endif
