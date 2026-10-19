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

class CtbmvWideColumnSliceAiv : public CtbmvShape {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& tiling)
    {
        InitShape(tiling);
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(lda_) * n_ * 2U);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), static_cast<uint64_t>(n_) * 2U);
        const uint64_t alignedSnapshotBytes = (static_cast<uint64_t>(n_) * 2U * sizeof(float) + 31U) & ~31ULL;
        const uint64_t partialOffset = alignedSnapshotBytes + 9216U + static_cast<uint64_t>(blockIdx_) * n_ * 8U;
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + partialOffset), n_ * 2U);
        atomicMaskGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint32_t*>(workspace) + alignedSnapshotBytes / sizeof(uint32_t), 1152U * 2U);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= useCoreNum_) {
            return;
        }

        InitBuffers();
        InitWindow();
        LoadInput();
        PrefetchColumn(columnBegin, aAos);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        for (uint32_t localColumn = 0U; localColumn < columnCount; ++localColumn) {
            ProcessColumn(localColumn);
        }

        StorePartial();
    }

private:
    __aicore__ inline void AccumulateColumn(
        uint32_t localColumn, uint32_t column, uint32_t rowBegin, bool useLeftPadding, uint32_t leftPaddingRows,
        uint32_t vectorCount, float xReal, float xImag)
    {
        const uint32_t vectorGlobalRowBegin = useLeftPadding ? rowBegin - leftPaddingRows : rowBegin;
        auto currentA = (localColumn & 1U) ? highPadAAos : aAos;
        SplitComplexVector(aReal, aImag, currentA, vectorCount);
        PipeBarrier<PIPE_V>();
        // Stage the next column while vector arithmetic consumes this one.
        if (localColumn + 1U < columnCount) {
            auto nextA = (localColumn & 1U) ? aAos : highPadAAos;
            PrefetchColumn(column + 1U, nextA);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }
        if (diag_ == ACLBLAS_UNIT) {
            const uint32_t diagonal = column - vectorGlobalRowBegin;
            uint64_t diagonalMask[2] = {1ULL << (diagonal % 64U), 0ULL};
            Duplicate(aReal[diagonal / 64U * 64U], 1.0F, diagonalMask, 1U, 1U, 8U);
            Duplicate(aImag[diagonal / 64U * 64U], 0.0F, diagonalMask, 1U, 1U, 8U);
            PipeBarrier<PIPE_V>();
        }
        AscendC::Axpy(yReal[vectorGlobalRowBegin], aReal, xReal, static_cast<int32_t>(vectorCount));
        AscendC::Axpy(yImag[vectorGlobalRowBegin], aReal, xImag, static_cast<int32_t>(vectorCount));
        PipeBarrier<PIPE_V>();
        AscendC::Axpy(yImag[vectorGlobalRowBegin], aImag, xReal, static_cast<int32_t>(vectorCount));
        AscendC::Axpy(yReal[vectorGlobalRowBegin], aImag, -xImag, static_cast<int32_t>(vectorCount));
    }

    __aicore__ inline void StorePartial()
    {
        {
            AsdopsBuffer<ArchType::ASCEND_V220> buffer;
            constexpr uint32_t interleavedOffset = xAosOffset;
            constexpr uint32_t maskOffset = interleavedOffset + maxAtomicRows * 2U * sizeof(float);
            LocalTensor<float> ubBase = buffer.GetBuffer<BufferType::ASCEND_UB, float>(0U);
            LocalTensor<float> interleaved = buffer.GetBuffer<BufferType::ASCEND_UB, float>(interleavedOffset);
            LocalTensor<uint32_t> mask = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(maskOffset);
            gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
                mask, atomicMaskGm_, 0, 1, atomicCount * 2U * sizeof(uint32_t), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            // The output staging area reuses xAos after the last scalar read.
            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            Gather(interleaved, yReal[atomicBegin], mask, static_cast<uint32_t>(0), atomicCount * 2U);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            ub_to_gm_align<ArchType::ASCEND_V220, float>(
                outputGm_[static_cast<uint64_t>(atomicBegin) * 2U], interleaved, 0, 1, atomicCount * 2U * sizeof(float),
                0, 0, 0, 0);
            PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void ProcessColumn(uint32_t localColumn)
    {
        const uint32_t column = columnBegin + localColumn;
        const uint32_t rowBegin = uplo_ == ACLBLAS_UPPER ? SaturatingSub(column, k_) : column;
        const uint32_t rowEnd = uplo_ == ACLBLAS_UPPER ? column : Min(n_ - 1U, column + k_);
        const uint32_t count = rowEnd - rowBegin + 1U;

        const uint32_t misalignment = rowBegin & 7U;
        const bool useLeftPadding = misalignment > 0U;
        const uint32_t leftPaddingRows = useLeftPadding ? misalignment : 0U;
        const uint32_t copyMatrixCount = count;
        const uint32_t vectorCount = copyMatrixCount + leftPaddingRows;

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        PipeBarrier<PIPE_V>();

        const uint32_t xOffset = localColumn * 2U;
        const float xReal = xAos.GetValue(xOffset);
        const float xImag = xAos.GetValue(xOffset + 1U);
        AccumulateColumn(localColumn, column, rowBegin, useLeftPadding, leftPaddingRows, vectorCount, xReal, xImag);
    }

    __aicore__ inline void LoadInput()
    {
        Duplicate(yReal[zeroBegin], 0.0F, zeroCount);
        Duplicate(yImag[zeroBegin], 0.0F, zeroCount);
        // DataCopyPad can add at most eight float elements on the left.
        // Keep an aligned eight-float zero prefix in a second A buffer so
        // row phases 5/6/7 only need another 2/4/6 padded floats.
        Duplicate(highPadAAos, 0.0F, 8U);
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            xAos, xGm_[static_cast<uint64_t>(columnBegin) * 2U], 0, 1, columnCount * 2U * sizeof(float), 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitWindow()
    {
        baseColumns = n_ / useCoreNum_;
        extraColumns = n_ % useCoreNum_;
        columnBegin = blockIdx_ * baseColumns + Min(blockIdx_, extraColumns);
        columnCount = baseColumns + (blockIdx_ < extraColumns ? 1U : 0U);
        lastColumn = columnBegin + columnCount - 1U;
        outputBegin = uplo_ == ACLBLAS_UPPER ? SaturatingSub(columnBegin, k_) : columnBegin;
        outputEnd = uplo_ == ACLBLAS_UPPER ? lastColumn : Min(n_ - 1U, lastColumn + k_);
        outputCount = outputEnd - outputBegin + 1U;
        atomicBegin = outputBegin & ~3U;
        atomicEnd = Min(n_, (outputEnd + 4U) & ~3U);
        atomicCount = atomicEnd - atomicBegin;
        zeroBegin = outputBegin & ~7U;
        zeroEnd = Min(n_, (outputEnd + 8U) & ~7U);
        zeroCount = zeroEnd - zeroBegin;
    }

    uint32_t baseColumns;
    uint32_t extraColumns;
    uint32_t lastColumn;
    uint32_t outputBegin;
    uint32_t outputEnd;
    uint32_t outputCount;
    uint32_t atomicBegin;
    uint32_t atomicEnd;
    uint32_t atomicCount;
    uint32_t zeroBegin;
    uint32_t zeroEnd;
    uint32_t zeroCount;

    uint32_t columnBegin;
    uint32_t columnCount;

    __aicore__ inline void InitBuffers()
    {
        // vreducev2 writes 32 planar values per repeat.  A fully padded
        // k=1023 column can contain 1031 values, so reserve the rounded-up
        // 1056-value destination instead of letting the final repeat overlap
        // the following UB view.

        AsdopsBuffer<ArchType::ASCEND_V220> buffer;
        yReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(yRealOffset);
        yImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(yImagOffset);
        aAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aAosOffset);
        highPadAAos = aAos[maxBandVector * 2U];
        aReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aRealOffset);
        aImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(aImagOffset);
        xAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(xAosOffset);
    }

    static constexpr uint32_t maxN = 4096U;
    // vreducev2 reads full 64-float repeats. Reserve 2112 AoS floats
    // for each source buffer so the padded second source cannot overlap
    // aReal while the real/imaginary extraction instructions execute.
    static constexpr uint32_t maxBandVector = 1056U;
    static constexpr uint32_t maxBandPlanar = 1056U;
    static constexpr uint32_t maxAtomicRows = 1152U;
    static constexpr uint32_t yRealOffset = 0U;
    static constexpr uint32_t yImagOffset = yRealOffset + maxN * sizeof(float);
    static constexpr uint32_t aAosOffset = yImagOffset + maxN * sizeof(float);
    static constexpr uint32_t aRealOffset = aAosOffset + maxBandVector * 4U * sizeof(float);
    static constexpr uint32_t aImagOffset = aRealOffset + maxBandPlanar * sizeof(float);
    static constexpr uint32_t xAosOffset = aImagOffset + maxBandPlanar * sizeof(float);

    LocalTensor<float> yReal;
    LocalTensor<float> yImag;
    LocalTensor<float> aAos;
    LocalTensor<float> highPadAAos;
    LocalTensor<float> aReal;
    LocalTensor<float> aImag;
    LocalTensor<float> xAos;

    __aicore__ inline void PrefetchColumn(uint32_t column, LocalTensor<float> dst) const
    {
        const uint32_t begin = uplo_ == ACLBLAS_UPPER ? SaturatingSub(column, k_) : column;
        const uint32_t end = uplo_ == ACLBLAS_UPPER ? column : Min(n_ - 1U, column + k_);
        const uint32_t count = end - begin + 1U;
        const uint32_t band = uplo_ == ACLBLAS_UPPER ? k_ + begin - column : 0U;
        const uint64_t offset = (static_cast<uint64_t>(column) * lda_ + band) * 2U;
        const uint32_t phase = begin & 7U;
        const bool high = phase > 4U;
        const uint32_t left = (high ? phase - 4U : phase) * 2U;
        if (high) {
            Duplicate(dst, 0.0F, 8U);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        DataCopyExtParams copy{1U, count * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{
            true, static_cast<uint8_t>(left), static_cast<uint8_t>((8U - (left + count * 2U) % 8U) % 8U), 0.0F};
        DataCopyPad(dst[high ? 8U : 0U], aGm_[offset], copy, pad);
    }
    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t SaturatingSub(uint32_t lhs, uint32_t rhs) const { return lhs > rhs ? lhs - rhs : 0U; }

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<uint32_t> atomicMaskGm_;
};

class CtbmvWideGroupAiv {
public:
    __aicore__ inline void Process(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& t)
    {
        const uint32_t core = GetBlockIdx();
        if (core >= t.useCoreNum) {
            return;
        }
        InitInput(a, x, workspace, t, core);
        InitBuffers();
        InitWindow(t);
        LoadInput();
        for (uint32_t local = 0U; local < columns; local += 8U) {
            ProcessBatch(local, t);
        }
        StorePartial();
    }

private:
    __aicore__ inline void LoadBatch(uint32_t local, uint32_t column, uint32_t batch, const CtbmvTilingData& t)
    {
        Duplicate(pr, 0.0F, 8U * width + 8U);
        Duplicate(pi, 0.0F, 8U * width + 8U);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        const uint32_t blockWidth = (k + 4U) & ~3U;
        DataCopyExtParams aCopy{
            static_cast<uint16_t>(batch), (k + 1U) * 8U, (t.lda - k - 1U) * 8U, (width - blockWidth) / 4U, 0U};
        DataCopyPadExtParams<float> aPad{true, 0U, static_cast<uint8_t>((blockWidth - k - 1U) * 2U), 0.0F};
        DataCopyPad(aa, aGm[static_cast<uint64_t>(column) * t.lda * 2U], aCopy, aPad);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        SplitComplexVector(ar, ai, aa, batch * width);
        Brcb(br, xr[local], 1U, {1U, 8U});
        Brcb(bi, xi[local], 1U, {1U, 8U});
        PipeBarrier<PIPE_V>();
        Muls(neg, bi, -1.0F, 64U);
        if (t.diag == ACLBLAS_UNIT) {
            const uint32_t diagonal = upper ? k : 0U;
            uint64_t mask[2] = {1ULL << (diagonal % 64U), 0ULL};
            Duplicate(ar[diagonal / 64U * 64U], 1.0F, mask, batch, 1U, width / 8U);
            Duplicate(ai[diagonal / 64U * 64U], 0.0F, mask, batch, 1U, width / 8U);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MultiplyBatch(uint32_t batch)
    {
        SetMaskNorm();
        ResetMask();
        for (uint32_t inner = 0U; inner <= k; inner += 64U) {
            const uint32_t lanes = Min(64U, k + 1U - inner);
            SetVectorMask<float>(0ULL, lanes == 64U ? ~0ULL : (1ULL << lanes) - 1ULL);
            mul_v<ArchType::ASCEND_V220, float>(
                pr[inner], ar[inner], br, batch, 1U, 1U, 0U, width / 8U, width / 8U, 1U);
            mul_v<ArchType::ASCEND_V220, float>(
                pi[inner], ar[inner], bi, batch, 1U, 1U, 0U, width / 8U, width / 8U, 1U);
            PipeBarrier<PIPE_V>();
            vmla(
                reinterpret_cast<__ubuf__ float*>(pr[inner].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ai[inner].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(neg.GetPhyAddr()), batch, 1U, 1U, 0U, width / 8U, width / 8U, 1U);
            vmla(
                reinterpret_cast<__ubuf__ float*>(pi[inner].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ai[inner].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(br.GetPhyAddr()), batch, 1U, 1U, 0U, width / 8U, width / 8U, 1U);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessBatch(uint32_t local, const CtbmvTilingData& t)
    {
        const uint32_t batch = Min(8U, columns - local);
        const uint32_t column = first + local;
        const int32_t rawBegin = static_cast<int32_t>(column) - static_cast<int32_t>((upper ? k : 0U) + phase);
        const uint32_t skip = rawBegin < 0 ? static_cast<uint32_t>(-rawBegin) : 0U;
        const uint32_t outputBegin = rawBegin < 0 ? 0U : static_cast<uint32_t>(rawBegin);
        const uint32_t count = Min(n - outputBegin, k + batch + phase - skip);
        const uint32_t reduceRows = (count + 7U) & ~7U;
        LoadBatch(local, column, batch, t);
        MultiplyBatch(batch);
        ResetMask();
        Gather(aa, pr, indices[skip * 8U], 0U, reduceRows * 8U);
        Gather(aa[maxRows * 8U], pi, indices[skip * 8U], 0U, reduceRows * 8U);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float>(partialR, aa, reduceRows / 8U, 64U, 1U, 1U, 8U);
        BlockReduceSum<float>(partialI, aa[maxRows * 8U], reduceRows / 8U, 64U, 1U, 1U, 8U);
        PipeBarrier<PIPE_V>();
        Add(yr[outputBegin], yr[outputBegin], partialR, count);
        Add(yi[outputBegin], yi[outputBegin], partialI, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StorePartial()
    {
        const uint32_t atomBegin = logicalBegin & ~3U;
        const uint32_t atomEnd = Min(n, (logicalEnd + 3U) & ~3U);
        const uint32_t atomCount = atomEnd - atomBegin;
        gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(atomMask, maskGm, 0, 1, atomCount * 8U, 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Gather(xa, yr[atomBegin], atomMask, 0U, atomCount * 2U);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        ub_to_gm_align<ArchType::ASCEND_V220, float>(
            outGm[static_cast<uint64_t>(atomBegin) * 2U], xa, 0, 1, atomCount * 8U, 0, 0, 0, 0);
        PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void LoadInput()
    {
        Duplicate(yr[zeroBegin], 0.0F, zeroEnd - zeroBegin);
        Duplicate(yi[zeroBegin], 0.0F, zeroEnd - zeroBegin);
        Duplicate(xa, 0.0F, 256U);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams xCopy{1U, columns * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> xPad{true, 0U, static_cast<uint8_t>((4U - columns % 4U) % 4U * 2U), 0.0F};
        DataCopyPad(xa, xGm[static_cast<uint64_t>(first) * 2U], xCopy, xPad);
        phase = (first - (upper ? k : 0U)) & 7U;
        gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(
            indices, groupGm[phase * tableRows * 8U], 0, 1, tableRows * 32U, 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        SplitComplexVector(xr, xi, xa, (columns + 7U) & ~7U);
        PipeBarrier<PIPE_V>();
    }

    uint32_t phase;

    __aicore__ inline void InitWindow(const CtbmvTilingData& t)
    {
        upper = t.uplo == ACLBLAS_UPPER;
        logicalBegin = upper ? (first > k ? first - k : 0U) : first;
        zeroBegin = logicalBegin & ~7U;
        logicalEnd = Min(n, first + columns + (upper ? 0U : k));
        zeroEnd = Min(n, (logicalEnd + 7U) & ~7U);
    }

    bool upper;
    uint32_t logicalBegin;
    uint32_t zeroBegin;
    uint32_t logicalEnd;
    uint32_t zeroEnd;

    __aicore__ inline void InitInput(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CtbmvTilingData& t, uint32_t core)
    {
        n = t.n;
        k = t.k;
        width = (k + 64U) & ~63U;
        tableRows = (k + 22U) & ~7U;
        first = core * (n / t.useCoreNum) + Min(core, n % t.useCoreNum);
        columns = n / t.useCoreNum + (core < n % t.useCoreNum ? 1U : 0U);
        alignedSnapshot = (n * 8U + 31U) & ~31U;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(n) * t.lda * 2U);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), n * 2U);
        const uint64_t partialOffset =
            alignedSnapshot + 9216U + 8U * tableRows * 32U + static_cast<uint64_t>(core) * n * 8U;
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + partialOffset), n * 2U);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace + alignedSnapshot), 2304U);
        groupGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint32_t*>(workspace + alignedSnapshot + 9216U), 8U * tableRows * 8U);
    }

    GlobalTensor<float> aGm;
    GlobalTensor<float> xGm;
    GlobalTensor<float> outGm;
    GlobalTensor<uint32_t> maskGm;
    GlobalTensor<uint32_t> groupGm;

    uint32_t n;
    uint32_t k;
    uint32_t width;
    uint32_t tableRows;
    uint32_t first;
    uint32_t columns;
    uint32_t alignedSnapshot;

    __aicore__ inline void InitBuffers()
    {
        yr = LocalTensor<float>(TPosition::VECCALC, 0U, 4096U);
        yi = LocalTensor<float>(TPosition::VECCALC, 16384U, 4096U);
        aa = LocalTensor<float>(TPosition::VECCALC, 32768U, maxEntries * 2U);
        ar = LocalTensor<float>(TPosition::VECCALC, 69632U, maxEntries);
        ai = LocalTensor<float>(TPosition::VECCALC, 88064U, maxEntries);
        pr = LocalTensor<float>(TPosition::VECCALC, 106496U, maxEntries + 8U);
        pi = LocalTensor<float>(TPosition::VECCALC, 124960U, maxEntries + 8U);
        // The x staging area becomes interleaved output scratch after reduction.
        xa = LocalTensor<float>(TPosition::VECCALC, 143424U, 2304U);
        xr = LocalTensor<float>(TPosition::VECCALC, 144448U, 128U);
        xi = LocalTensor<float>(TPosition::VECCALC, 144960U, 128U);
        br = LocalTensor<float>(TPosition::VECCALC, 145472U, 64U);
        bi = LocalTensor<float>(TPosition::VECCALC, 145728U, 64U);
        neg = LocalTensor<float>(TPosition::VECCALC, 145984U, 64U);
        partialR = LocalTensor<float>(TPosition::VECCALC, 146240U, maxRows);
        partialI = LocalTensor<float>(TPosition::VECCALC, 148352U, maxRows);
        indices = LocalTensor<uint32_t>(TPosition::VECCALC, 150464U, maxRows * 8U);
        atomMask = LocalTensor<uint32_t>(TPosition::VECCALC, 167360U, 2304U);
    }

    static constexpr uint32_t maxEntries = 8U * 576U;
    static constexpr uint32_t maxRows = 528U;

    LocalTensor<float> yr;
    LocalTensor<float> yi;
    LocalTensor<float> aa;
    LocalTensor<float> ar;
    LocalTensor<float> ai;
    LocalTensor<float> pr;
    LocalTensor<float> pi;
    LocalTensor<float> xa;
    LocalTensor<float> xr;
    LocalTensor<float> xi;
    LocalTensor<float> br;
    LocalTensor<float> bi;
    LocalTensor<float> neg;
    LocalTensor<float> partialR;
    LocalTensor<float> partialI;
    LocalTensor<uint32_t> indices;
    LocalTensor<uint32_t> atomMask;

    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const { return a < b ? a : b; }
};

} // namespace Ctbmv
#endif
