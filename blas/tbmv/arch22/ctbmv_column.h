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

class CtbmvColumnFastAiv : public CtbmvVectorBuffers {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        InitShape(tiling);
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2U);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(n_) * 2U);
        const uint64_t workspaceOffset = static_cast<uint64_t>(blockIdx_) * n_ * 2U;
        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace) + workspaceOffset, static_cast<uint64_t>(n_) * 2U);
    }

    __aicore__ inline void Process()
    {
        if (useCoreNum_ == 0U || blockIdx_ >= useCoreNum_) {
            return;
        }

        InitBuffers();
        for (uint32_t i = 0; i < CTBMV_FAST_TILE_COMPLEX; ++i) {
            mask.SetValue(i * 2U, outRealOffset + i * sizeof(float));
            mask.SetValue(i * 2U + 1U, outImagOffset + i * sizeof(float));
            aMask.SetValue(i, i * CTBMV_COMPACT_PADDED_COMPLEX_FLOATS * sizeof(float));
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<float>();

        const uint32_t maxBandDistance = Min(k_, n_ - 1U);
        for (uint32_t bandDistance = blockIdx_; bandDistance <= maxBandDistance; bandDistance += useCoreNum_) {
            ProcessBand(bandDistance);
        }
        SetAtomicNone();
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(yGm_);
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void ProcessBand(uint32_t bandDistance)
    {
        const uint32_t elementCount = n_ - bandDistance;
        const uint32_t columnBegin = uplo_ == ACLBLAS_UPPER ? bandDistance : 0U;
        const uint32_t rowBegin = uplo_ == ACLBLAS_UPPER ? 0U : bandDistance;
        const uint32_t bandRow = uplo_ == ACLBLAS_UPPER ? k_ - bandDistance : bandDistance;

        for (uint32_t offset = 0U; offset < elementCount; offset += CTBMV_FAST_TILE_COMPLEX) {
            ProcessColumnTile(bandDistance, elementCount, columnBegin, rowBegin, bandRow, offset);
        }
    }

    __aicore__ inline void ProcessColumnTile(
        uint32_t bandDistance, uint32_t elementCount, uint32_t columnBegin, uint32_t rowBegin, uint32_t bandRow,
        uint32_t offset)
    {
        const uint32_t count = Min(CTBMV_FAST_TILE_COMPLEX, elementCount - offset);
        const uint64_t aOffset = (static_cast<uint64_t>(columnBegin + offset) * lda_ + bandRow) * 2U;
        DataCopyExtParams aCopyParams{
            static_cast<uint16_t>(count), static_cast<uint32_t>(2U * sizeof(float)),
            static_cast<uint32_t>((lda_ - 1U) * 2U * sizeof(float)), 0U, 0U};
        DataCopyPadExtParams<float> aPadParams{
            true, 0U, static_cast<uint8_t>(CTBMV_COMPACT_PADDED_COMPLEX_FLOATS - 2U), 0.0F};
        DataCopyPad(aAos, aGm_[aOffset], aCopyParams, aPadParams);
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            xAos, xGm_[static_cast<uint64_t>(columnBegin + offset) * 2U], 0, 1, count * 2U * sizeof(float), 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();

        if (diag_ == ACLBLAS_UNIT && bandDistance == 0U) {
            ub_to_gm_align<ArchType::ASCEND_V220, float>(
                yGm_[static_cast<uint64_t>(rowBegin + offset) * 2U], xAos, 0, 1, count * 2U * sizeof(float), 0, 0, 0,
                0);
            PipeBarrier<PIPE_ALL>();
            return;
        }

        ComputeColumnProduct(count);
        const uint32_t gatherCount = ((count * 2U + 63U) / 64U) * 64U;
        Gather(interleaved, ubBase, mask, static_cast<uint32_t>(0), gatherCount);
        PipeBarrier<PIPE_ALL>();
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            yGm_[static_cast<uint64_t>(rowBegin + offset) * 2U], interleaved, 0, 1, count * 2U * sizeof(float), 0, 0, 0,
            0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeColumnProduct(uint32_t count)
    {
        Gather(aReal, aAos, aMask, static_cast<uint32_t>(0), count);
        Gather(aImag, aAos[1U], aMask, static_cast<uint32_t>(0), count);
        PipeBarrier<PIPE_V>();
        // Gather's count-based mask must not leak into the following repeat-
        // based deinterleave and SIMD primitives.
        ResetMask();
        SplitComplexVector(xReal, xImag, xAos, count);
        PipeBarrier<PIPE_V>();
        const uint32_t repeats = (count + 63U) / 64U;
        mul_v<ArchType::ASCEND_V220, float>(outReal, aReal, xReal, repeats, 1, 1, 1, 8, 8, 8);
        mul_v<ArchType::ASCEND_V220, float>(tmpReal, aImag, xImag, repeats, 1, 1, 1, 8, 8, 8);
        mul_v<ArchType::ASCEND_V220, float>(outImag, aReal, xImag, repeats, 1, 1, 1, 8, 8, 8);
        mul_v<ArchType::ASCEND_V220, float>(tmpImag, aImag, xReal, repeats, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        sub_v<ArchType::ASCEND_V220, float>(outReal, outReal, tmpReal, repeats, 1, 1, 1, 8, 8, 8);
        add_v<ArchType::ASCEND_V220, float>(outImag, outImag, tmpImag, repeats, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitBuffers()
    {
        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        ubBase = buffer.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        aAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aosOffset);
        xAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xAosOffset);
        aReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(realOffset);
        aImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(imagOffset);
        xReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xRealOffset);
        xImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xImagOffset);
        outReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(outRealOffset);
        outImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(outImagOffset);
        tmpReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(tmpRealOffset);
        tmpImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(tmpImagOffset);
        interleaved = buffer.GetBuffer<BufferType::ASCEND_UB, float>(interleavedOffset);
        mask = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(maskOffset);
        aMask = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(aMaskOffset);
    }

    static constexpr uint32_t aosOffset = 0U;
    static constexpr uint32_t xAosOffset =
        CTBMV_FAST_TILE_COMPLEX * CTBMV_COMPACT_PADDED_COMPLEX_FLOATS * sizeof(float);
    static constexpr uint32_t realOffset = xAosOffset + CTBMV_FAST_TILE_COMPLEX * 2U * sizeof(float);
    static constexpr uint32_t imagOffset = realOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t xRealOffset = imagOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t xImagOffset = xRealOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t outRealOffset = xImagOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t outImagOffset = outRealOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t tmpRealOffset = outImagOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t tmpImagOffset = tmpRealOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t interleavedOffset = tmpImagOffset + CTBMV_FAST_TILE_COMPLEX * sizeof(float);
    static constexpr uint32_t maskOffset = interleavedOffset + CTBMV_FAST_TILE_COMPLEX * 2U * sizeof(float);
    static constexpr uint32_t aMaskOffset = maskOffset + CTBMV_FAST_TILE_COMPLEX * 2U * sizeof(uint32_t);

    LocalTensor<float> ubBase;
    LocalTensor<float> interleaved;
    LocalTensor<uint32_t> mask;
    LocalTensor<uint32_t> aMask;

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
};

} // namespace Ctbmv
#endif

namespace Ctbmv {

class CtbmvAiv : public CtbmvShape {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        InitShape(tiling);
        incx_ = tiling.incx;
        rowOwnedNoTranspose_ = tiling.path == CtbmvPath::Row;
        absIncx_ = incx_ >= 0 ? incx_ : -incx_;

        const uint64_t aFloatCount = static_cast<uint64_t>(lda_) * n_ * 2U;
        const uint64_t xFloatCount = (static_cast<uint64_t>(n_ - 1U) * absIncx_ + 1U) * 2U;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), aFloatCount);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), xFloatCount);
        const uint64_t workspaceFloatCount =
            static_cast<uint64_t>(n_) * 2U *
            ((incx_ == 1 && trans_ == ACLBLAS_OP_N && !rowOwnedNoTranspose_) ? useCoreNum_ : 1U);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), workspaceFloatCount);
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(outputGm_);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= useCoreNum_) {
            return;
        }

        const uint32_t rowBegin = blockIdx_ * rowsPerCore_;
        const uint32_t rowEnd = Min(rowBegin + rowsPerCore_, n_);
        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            ComputeRow(row);
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(outputGm_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyBack()
    {
        if (blockIdx_ >= useCoreNum_) {
            return;
        }
#if __DAV_C220_VEC__
        if (incx_ == 1 && trans_ == ACLBLAS_OP_N && !rowOwnedNoTranspose_) {
            ReducePrivateSlices();
            return;
        }
        if (incx_ == 1) {
            CopyContiguous();
            return;
        }
#endif
        CopyStrided();
    }

private:
    __aicore__ inline void CopyStrided()
    {
        const bool reducePrivateSlices = incx_ == 1 && trans_ == ACLBLAS_OP_N;
        const uint32_t rowBegin = reducePrivateSlices ? 0U : blockIdx_ * rowsPerCore_;
        const uint32_t rowEnd = reducePrivateSlices ? n_ : Min(rowBegin + rowsPerCore_, n_);
        if (reducePrivateSlices) {
            for (uint32_t row = rowBegin; row < rowEnd; ++row) {
                float real = 0.0F;
                float imag = 0.0F;
                for (uint32_t core = 0; core < useCoreNum_; ++core) {
                    const uint64_t outputOffset = (static_cast<uint64_t>(core) * n_ + row) * 2U;
                    real += outputGm_.GetValue(outputOffset);
                    imag += outputGm_.GetValue(outputOffset + 1U);
                }
                xGm_.SetValue(static_cast<uint64_t>(row) * 2U, real);
                xGm_.SetValue(static_cast<uint64_t>(row) * 2U + 1U, imag);
            }
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(xGm_);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            const uint64_t outputOffset = static_cast<uint64_t>(row) * 2U;
            const uint64_t xOffset = VectorOffset(row);
            xGm_.SetValue(xOffset, outputGm_.GetValue(outputOffset));
            xGm_.SetValue(xOffset + 1U, outputGm_.GetValue(outputOffset + 1U));
        }
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(xGm_);
        PipeBarrier<PIPE_ALL>();
    }

#if __DAV_C220_VEC__
    __aicore__ inline void CopyContiguous()
    {
        constexpr uint32_t copyTileComplex = 1024U;
        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        LocalTensor<float> contiguous = buffer.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        for (uint32_t row = 0U; row < n_; row += copyTileComplex) {
            const uint32_t count = Min(copyTileComplex, n_ - row);
            const uint32_t copyBytes = count * 2U * sizeof(float);
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                contiguous, outputGm_[static_cast<uint64_t>(row) * 2U], 0, 1, copyBytes, 0, 0, 0, 0);
            PipeBarrier<PIPE_ALL>();
            ub_to_gm_align<ArchType::ASCEND_V220, float>(
                xGm_[static_cast<uint64_t>(row) * 2U], contiguous, 0, 1, copyBytes, 0, 0, 0, 0);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ReducePrivateSlices()
    {
        constexpr uint32_t copyTileComplex = 1024U;
        constexpr uint32_t tempOffset = copyTileComplex * 2U * sizeof(float);
        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        LocalTensor<float> accumulated = buffer.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        LocalTensor<float> temporary = buffer.GetBuffer<BufferType::ASCEND_UB, float>(tempOffset);

        for (uint32_t row = 0U; row < n_; row += copyTileComplex) {
            const uint32_t count = Min(copyTileComplex, n_ - row);
            const uint32_t floatCount = count * 2U;
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                accumulated, outputGm_[static_cast<uint64_t>(row) * 2U], 0, 1, floatCount * sizeof(float), 0, 0, 0, 0);
            PipeBarrier<PIPE_ALL>();

            const uint32_t repeats = (floatCount + 63U) / 64U;
            for (uint32_t core = 1U; core < useCoreNum_; ++core) {
                const uint64_t outputOffset = (static_cast<uint64_t>(core) * n_ + row) * 2U;
                gm_to_ub_align<ArchType::ASCEND_V220, float>(
                    temporary, outputGm_[outputOffset], 0, 1, floatCount * sizeof(float), 0, 0, 0, 0);
                PipeBarrier<PIPE_ALL>();
                add_v<ArchType::ASCEND_V220, float>(accumulated, accumulated, temporary, repeats, 1, 1, 1, 8, 8, 8);
                // Finish vector reads before reloading temporary, and finish
                // accumulated writes before the final MTE3 copy.
                PipeBarrier<PIPE_ALL>();
            }

            ub_to_gm_align<ArchType::ASCEND_V220, float>(
                xGm_[static_cast<uint64_t>(row) * 2U], accumulated, 0, 1, floatCount * sizeof(float), 0, 0, 0, 0);
            PipeBarrier<PIPE_ALL>();
        }
    }

#endif

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t SaturatingSub(uint32_t lhs, uint32_t rhs) const { return lhs > rhs ? lhs - rhs : 0U; }

    __aicore__ inline uint64_t VectorOffset(uint32_t logicalIndex) const
    {
        const uint64_t physicalIndex = incx_ >= 0 ? static_cast<uint64_t>(logicalIndex) * absIncx_ :
                                                    static_cast<uint64_t>(n_ - 1U - logicalIndex) * absIncx_;
        return physicalIndex * 2U;
    }

    __aicore__ inline uint64_t MatrixOffset(uint32_t row, uint32_t col) const
    {
        const int64_t bandRow = uplo_ == ACLBLAS_UPPER ? static_cast<int64_t>(k_) + static_cast<int64_t>(row) - col :
                                                         static_cast<int64_t>(row) - col;
        return (static_cast<uint64_t>(col) * lda_ + static_cast<uint64_t>(bandRow)) * 2U;
    }

    __aicore__ inline void ComputeRow(uint32_t row)
    {
#if __DAV_C220_VEC__
        ComputeRowDma(row);
#else
        uint32_t colBegin = 0U;
        uint32_t colEnd = 0U;
        GetRowRange(row, colBegin, colEnd);

        float accReal = 0.0F;
        float accImag = 0.0F;
        for (uint32_t col = colBegin; col <= colEnd; ++col) {
            float aReal = 1.0F;
            float aImag = 0.0F;
            if (!(diag_ == ACLBLAS_UNIT && col == row)) {
                const uint32_t matrixRow = trans_ == ACLBLAS_OP_N ? row : col;
                const uint32_t matrixCol = trans_ == ACLBLAS_OP_N ? col : row;
                const uint64_t aOffset = MatrixOffset(matrixRow, matrixCol);
                aReal = aGm_.GetValue(aOffset);
                aImag = aGm_.GetValue(aOffset + 1U);
                if (trans_ == ACLBLAS_OP_C) {
                    aImag = -aImag;
                }
            }

            const uint64_t xOffset = VectorOffset(col);
            const float xReal = xGm_.GetValue(xOffset);
            const float xImag = xGm_.GetValue(xOffset + 1U);
            accReal += aReal * xReal - aImag * xImag;
            accImag += aReal * xImag + aImag * xReal;
        }

        const uint64_t outputOffset = static_cast<uint64_t>(row) * 2U;
        outputGm_.SetValue(outputOffset, accReal);
        outputGm_.SetValue(outputOffset + 1U, accImag);
#endif
    }

#if __DAV_C220_VEC__
    // Each complex value occupies one UB block. Only the two valid GM floats
    // are transferred; padding never reads adjacent GM memory.
    __aicore__ inline void LoadTile(
        LocalTensor<float> dst, GlobalTensor<float> src, uint64_t offset, uint64_t step, uint32_t count)
    {
        DataCopyPadExtParams<float> pad{true, 0U, 6U, 0.0F};
        if (count == 1U || (step >= 2U && (step - 2U) <= 0xffffffffU / sizeof(float))) {
            DataCopyExtParams copy{
                static_cast<uint16_t>(count), 8U, count == 1U ? 0U : static_cast<uint32_t>((step - 2U) * sizeof(float)),
                0U, 0U};
            DataCopyPad(dst, src[offset], copy, pad);
        } else {
            DataCopyExtParams copy{1U, 8U, 0U, 0U, 0U};
            for (uint32_t i = 0U; i < count; ++i) {
                DataCopyPad(dst[i * 8U], src[offset + i * step], copy, pad);
            }
        }
    }

    __aicore__ inline void ComputeTileDma(
        uint32_t row, uint32_t col, uint32_t count, bool unit, float& real, float& imag)
    {
        LocalTensor<float> a(TPosition::VECCALC, 0U, 512U);
        LocalTensor<float> x(TPosition::VECCALC, 2048U, 512U);
        LocalTensor<float> ar(TPosition::VECCALC, 4096U, 64U);
        LocalTensor<float> ai(TPosition::VECCALC, 4352U, 64U);
        LocalTensor<float> xr(TPosition::VECCALC, 4608U, 64U);
        LocalTensor<float> xi(TPosition::VECCALC, 4864U, 64U);
        LocalTensor<float> pr(TPosition::VECCALC, 5120U, 64U);
        LocalTensor<float> pi(TPosition::VECCALC, 5376U, 64U);
        LocalTensor<float> tmp(TPosition::VECCALC, 5632U, 64U);
        LocalTensor<uint32_t> am(TPosition::VECCALC, 5888U, 64U);
        LocalTensor<uint32_t> xm(TPosition::VECCALC, 6144U, 64U);
        LoadTile(x, xGm_, VectorOffset(incx_ > 0 ? col : col + count - 1U), absIncx_ * 2U, count);
        if (!unit) {
            const uint64_t offset = trans_ == ACLBLAS_OP_N ? MatrixOffset(row, col) : MatrixOffset(col, row);
            LoadTile(a, aGm_, offset, trans_ == ACLBLAS_OP_N ? (static_cast<uint64_t>(lda_) - 1U) * 2U : 2U, count);
        }
        for (uint32_t i = 0U; i < count; ++i) {
            am.SetValue(i, i * 32U);
            xm.SetValue(i, (incx_ > 0 ? i : count - 1U - i) * 32U);
        }
        PipeBarrier<PIPE_ALL>();
        Gather(xr, x, xm, static_cast<uint32_t>(0), count);
        Gather(xi, x[1U], xm, static_cast<uint32_t>(0), count);
        if (unit) {
            Duplicate(ar, 1.0F, count);
            Duplicate(ai, 0.0F, count);
        } else {
            Gather(ar, a, am, static_cast<uint32_t>(0), count);
            Gather(ai, a[1U], am, static_cast<uint32_t>(0), count);
        }
        PipeBarrier<PIPE_V>();
        if (trans_ == ACLBLAS_OP_C) {
            Muls(ai, ai, -1.0F, count);
            PipeBarrier<PIPE_V>();
        }
        AccumulateTile(ar, ai, xr, xi, pr, pi, tmp, count, real, imag);
    }

    __aicore__ inline void AccumulateTile(
        LocalTensor<float> ar, LocalTensor<float> ai, LocalTensor<float> xr, LocalTensor<float> xi,
        LocalTensor<float> pr, LocalTensor<float> pi, LocalTensor<float> tmp, uint32_t count, float& real, float& imag)
    {
        Mul(pr, ar, xr, count);
        Mul(tmp, ai, xi, count);
        Mul(pi, ar, xi, count);
        PipeBarrier<PIPE_V>();
        Sub(pr, pr, tmp, count);
        PipeBarrier<PIPE_V>();
        Mul(tmp, ai, xr, count);
        PipeBarrier<PIPE_V>();
        Add(pi, pi, tmp, count);
        PipeBarrier<PIPE_ALL>();
        // Preserve increasing-column accumulation order rather than introducing
        // a different reduction tree in the generic accuracy fallback.
        for (uint32_t i = 0U; i < count; ++i) {
            real += pr.GetValue(i);
            imag += pi.GetValue(i);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeRowDma(uint32_t row)
    {
        uint32_t begin = 0U;
        uint32_t end = 0U;
        GetRowRange(row, begin, end);
        float real = 0.0F;
        float imag = 0.0F;
        for (uint32_t col = begin; col <= end;) {
            const bool unit = diag_ == ACLBLAS_UNIT && col == row;
            uint32_t count = unit ? 1U : Min(64U, end - col + 1U);
            if (diag_ == ACLBLAS_UNIT && col < row) {
                count = Min(count, row - col);
            }
            ComputeTileDma(row, col, count, unit, real, imag);
            col += count;
        }
        LocalTensor<float> result(TPosition::VECCALC, 0U, 8U);
        result.SetValue(0U, real);
        result.SetValue(1U, imag);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams store{1U, 8U, 0U, 0U, 0U};
        DataCopyPad(outputGm_[static_cast<uint64_t>(row) * 2U], result, store);
        PipeBarrier<PIPE_ALL>();
    }
#endif

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> outputGm_;

    int64_t incx_ = 1;
    int64_t absIncx_ = 1;
    bool rowOwnedNoTranspose_ = false;
};

} // namespace Ctbmv
