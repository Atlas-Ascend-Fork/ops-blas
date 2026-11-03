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
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "stbsv_tiling_data.h"
#include "stbsv_fma.h"

using namespace AscendC;

enum class TbsvUplo { UPPER, LOWER };
enum class TbsvTrans { NO_TRANS, TRANS };
enum class TbsvDiag { UNIT, NON_UNIT };
constexpr uint32_t TBSV_ELEMENTS_PER_BLOCK = 8;

__aicore__ inline int64_t TbsvBandIdxUpper(uint32_t row, uint32_t col, uint32_t k, uint32_t lda)
{
    return static_cast<int64_t>(k) + static_cast<int64_t>(row) - static_cast<int64_t>(col) +
           static_cast<int64_t>(col) * static_cast<int64_t>(lda);
}

__aicore__ inline int64_t TbsvBandIdxLower(uint32_t row, uint32_t col, uint32_t lda)
{
    return static_cast<int64_t>(row) - static_cast<int64_t>(col) +
           static_cast<int64_t>(col) * static_cast<int64_t>(lda);
}

__aicore__ inline int64_t TbsvXOffset(uint32_t idx, uint32_t n, int32_t incx)
{
    if (incx >= 0) {
        return static_cast<int64_t>(idx) * static_cast<int64_t>(incx);
    } else {
        return static_cast<int64_t>(n - 1 - idx) * (-static_cast<int64_t>(incx));
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
class StbsvKernel {
public:
    __aicore__ inline StbsvKernel() {}
    __aicore__ inline void Init(const StbsvTilingData& tiling);
    template <bool PARALLEL = false, bool WINDOW = false>
    __aicore__ inline void Process(TPipe& pipe);

private:
    __aicore__ inline void LoadX(LocalTensor<float>& xUb);
    __aicore__ inline void StoreX(LocalTensor<float>& xUb);
    __aicore__ inline void ProcessGlobal();
    template <bool WINDOW>
    __aicore__ inline void ProcessNoTrans(LocalTensor<float>& xUb, LocalTensor<float>& aUb);
    template <bool WINDOW, bool CACHED_DIAGONAL>
    __aicore__ inline void ProcessTrans(LocalTensor<float>& xUb, LocalTensor<float>& aUb);
    __aicore__ inline void ProcessOwnedRanges(LocalTensor<float>& xUb, LocalTensor<float>& aUb);
    __aicore__ inline void ApplyOwnedUpdate(
        LocalTensor<float>& xUb, LocalTensor<float>& aUb, uint32_t pivot, uint32_t begin, uint32_t end, float xValue);
    template <bool OWNED = false>
    __aicore__ inline void EnsureTransCache(
        uint32_t pivot, uint32_t firstCol, uint32_t lastCol, uint32_t ownedBegin = 0, uint32_t ownedEnd = 0);
    __aicore__ inline void EnsureNoTransCache(uint32_t pivot);
    __aicore__ inline void WaitTransWindowReaders(uint32_t firstCol, uint32_t lastCol);
    template <bool WINDOW>
    __aicore__ inline void PrepareTransWindow(uint32_t pivot, uint32_t firstCol, uint32_t lastCol);
    __aicore__ inline int64_t CachedBandOffset(int64_t globalOffset) const;
    __aicore__ inline uint32_t CachedBandStride() const;
    __aicore__ inline LocalTensor<float> CachedBandSource() const;
    template <bool OWNED = false>
    __aicore__ inline void UpdateSegment(
        LocalTensor<float>& xUb, LocalTensor<float>& aUb, uint32_t xStart, int64_t aOffset, uint32_t count,
        float alpha);
    __aicore__ inline void UpdateVector(
        LocalTensor<float>& xUb, LocalTensor<float>& aUb, uint32_t xStart, int64_t aOffset, uint32_t count,
        float alpha);
    __aicore__ inline void LoadBand(LocalTensor<float>& aUb, int64_t offset, uint32_t count, uint32_t leftPadding = 0);
    __aicore__ inline float GetDiag(uint32_t row) const;
    __aicore__ inline float GetSingleDiag(uint32_t row) const;
    template <bool WINDOW, bool CACHED_DIAGONAL>
    __aicore__ inline float GetTransDiag(uint32_t row) const;

    template <bool WINDOW>
    __aicore__ inline bool PublishNoTransPivot(uint32_t col, float& value);
    __aicore__ inline void InitVectorBuffers(
        TPipe& pipe, TBuf<TPosition::VECCALC>& xBuf, TBuf<TPosition::VECCALC>& outputBuf,
        TBuf<TPosition::VECCALC>& aBuf, uint32_t xElements, uint32_t bandElements);
    __aicore__ inline void InitUnitMatrixCache(
        TPipe& pipe, TBuf<TPosition::VECCALC>& cacheBuf, uint32_t effectiveBand, uint32_t xElements,
        uint32_t bandCopies, uint32_t bandElements);
    __aicore__ inline void InitUnitColumnCache(
        TPipe& pipe, TBuf<TPosition::VECCALC>& cacheBuf, uint32_t effectiveBand, uint32_t xElements,
        uint32_t bandCopies, uint32_t bandElements);
    __aicore__ inline void InitMatrixCache(
        TPipe& pipe, TBuf<TPosition::VECCALC>& cacheBuf, uint32_t xElements, uint32_t bandCopies,
        uint32_t bandElements);
    __aicore__ inline uint32_t SelectOwnedTransRows(
        uint32_t effectiveBand, uint64_t available, uint32_t& diagonalStageElements) const;
    __aicore__ inline uint32_t SelectSingleTransRows(
        uint32_t effectiveBand, uint64_t available, uint32_t& diagonalStageElements) const;
    __aicore__ inline LocalTensor<float> LoadDiagonalCache(
        TPipe& pipe, TBuf<TPosition::VECCALC>& diagonalBuf, TBuf<TPosition::VECCALC>& diagonalGatherBuf,
        LocalTensor<float>& xUb, uint32_t xElements, uint32_t diagonalStageElements);
    __aicore__ inline void InitNoTransWindow(LocalTensor<float>& gathered, uint32_t available, uint32_t bandElements);
    __aicore__ inline void InitNarrowWindow(
        TPipe& pipe, TBuf<TPosition::VECCALC>& cacheBuf, uint32_t effectiveBand, uint32_t xElements,
        uint32_t bandCopies, uint32_t bandElements);
    __aicore__ inline uint32_t SelectFallbackTransRows(uint32_t effectiveBand, uint32_t diagonalStageElements) const;
    template <bool PARALLEL>
    __aicore__ inline void InitTransWindow(
        LocalTensor<float>& gathered, uint32_t effectiveBand, uint32_t diagonalStageElements, uint32_t ownedTransRows,
        uint32_t singleTransRows);
    template <bool PARALLEL>
    __aicore__ inline void InitDiagonalCache(
        TPipe& pipe, TBuf<TPosition::VECCALC>& diagonalBuf, TBuf<TPosition::VECCALC>& diagonalGatherBuf,
        LocalTensor<float>& xUb, uint32_t effectiveBand, uint32_t xElements, uint32_t bandCopies,
        uint32_t bandElements);
    __aicore__ inline void InitScratch(TPipe& pipe, TBuf<TPosition::VECCALC> (&scratchBuf)[11], uint32_t bandElements);
    template <bool PARALLEL, bool WINDOW>
    __aicore__ inline void RunCachedSolve(
        TPipe& pipe, LocalTensor<float>& xUb, LocalTensor<float>& aUb, TBuf<TPosition::VECCALC>& gatherBuf,
        uint32_t effectiveBand, uint32_t bandElements);

    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;
    LocalTensor<float> scratch[11];
    uint32_t splitMaskElements = 0;
    bool splitMaskReady = false;
    LocalTensor<float> gatherBand;
    uint32_t gatherIndexSpan = 0;
    uint32_t gatherIndexStride = 0;
    bool gatherIndicesReady = false;
    LocalTensor<float> aCache;
    LocalTensor<float> diagonalCache;
    LocalTensor<float> outputX;
    bool xInputDirty = false;
    bool cacheA = false;
    bool cacheDiagonal = false;
    uint32_t unitCachePitch = 0;
    // Reuse the diagonal DMA staging buffer after its compact diagonal has
    // been extracted. GM offsets stay global throughout the update path.
    LocalTensor<float> transCache;
    uint32_t transCacheCols = 0;
    uint32_t transCacheBase = 0;
    uint32_t transCacheEnd = 0;
    LocalTensor<float> noTransCache;
    LocalTensor<uint32_t> noTransIndices;
    uint32_t noTransCacheCols = 0;
    uint32_t noTransCacheBase = 0;
    uint32_t noTransCacheEnd = 0;
    // Reuse the per-row strided DMA staging for eight adjacent pivot rows.
    // Each target column occupies one aligned 32B block in this layout.
    bool transTileEnabled = false;
    bool transTileActive = false;
    bool transTileSeparateIndices = false;
    LocalTensor<float> transTileStorage;
    uint32_t transTileRows = 8;
    uint32_t transTilePitch = 8;
    uint32_t transTileFirstRow = 0;
    uint32_t transTileFirstCol = 0;
    uint32_t transTilePivot = 0;
    uint32_t transTileLoadedRow = 0xffffffff;
    uint32_t currentBandColumn = 0;

    uint32_t n;
    uint32_t k;
    uint32_t lda;
    int32_t incx;
    uint32_t ubElements = 0;
};

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::Init(const StbsvTilingData& tiling)
{
    this->n = tiling.n;
    this->k = tiling.k;
    this->lda = tiling.lda;
    this->incx = tiling.incx;
    this->ubElements = static_cast<uint32_t>(tiling.ubSizeBytes / sizeof(float));

    int64_t aCount = static_cast<int64_t>(tiling.lda) * static_cast<int64_t>(n);
    int64_t absIncx = incx >= 0 ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    int64_t xCount = (n > 0) ? (absIncx * static_cast<int64_t>(n - 1) + 1) : 0;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.a), aCount);
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x), xCount);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline float StbsvKernel<UPLO, TRANS, DIAG>::GetDiag(uint32_t row) const
{
    if (cacheDiagonal) {
        return diagonalCache.GetValue(row);
    }
    if constexpr (UPLO == TbsvUplo::UPPER) {
        int64_t offset = static_cast<int64_t>(k) + static_cast<int64_t>(row) * static_cast<int64_t>(lda);
        return cacheA ? aCache.GetValue(offset) : aGM.GetValue(offset);
    } else {
        int64_t offset = static_cast<int64_t>(row) * static_cast<int64_t>(lda);
        return cacheA ? aCache.GetValue(offset) : aGM.GetValue(offset);
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline float StbsvKernel<UPLO, TRANS, DIAG>::GetSingleDiag(uint32_t row) const
{
    if (cacheDiagonal) {
        return diagonalCache.GetValue(row);
    }
    int64_t offset = static_cast<int64_t>(row) * lda;
    if constexpr (UPLO == TbsvUplo::UPPER) {
        offset += k;
    }
    if (cacheA) {
        return aCache.GetValue(offset);
    }
    if constexpr (TRANS == TbsvTrans::NO_TRANS && DIAG == TbsvDiag::NON_UNIT) {
        if (noTransCacheCols != 0 && row >= noTransCacheBase && row < noTransCacheEnd) {
            return noTransCache.GetValue(8 + offset - static_cast<int64_t>(noTransCacheBase) * lda);
        }
    }
    if constexpr (TRANS == TbsvTrans::TRANS && DIAG == TbsvDiag::NON_UNIT) {
        if (transCacheCols != 0 && row >= transCacheBase && row < transCacheEnd) {
            return transCache.GetValue(offset - static_cast<int64_t>(transCacheBase) * lda);
        }
    }
    return aGM.GetValue(offset);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool WINDOW, bool CACHED_DIAGONAL>
__aicore__ inline float StbsvKernel<UPLO, TRANS, DIAG>::GetTransDiag(uint32_t row) const
{
    return CACHED_DIAGONAL ? diagonalCache.GetValue(row) : (WINDOW ? GetSingleDiag(row) : GetDiag(row));
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::LoadX(LocalTensor<float>& xUb)
{
    if (incx == 1) {
        DataCopyExtParams params{1, n * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(xUb, xGM, params, pad);
        event_t toScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(toScalar);
        WaitFlag<HardEvent::MTE2_S>(toScalar);
        event_t toVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(toVector);
        WaitFlag<HardEvent::MTE2_V>(toVector);
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        xUb.SetValue(i, xGM.GetValue(TbsvXOffset(i, n, incx)));
    }
    event_t scalarToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(scalarToVector);
    WaitFlag<HardEvent::S_V>(scalarToVector);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::StoreX(LocalTensor<float>& xUb)
{
    if (incx == 1) {
        event_t fromScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(fromScalar);
        WaitFlag<HardEvent::S_MTE3>(fromScalar);
        event_t fromVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(fromVector);
        WaitFlag<HardEvent::V_MTE3>(fromVector);
        DataCopyExtParams params{1, n * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(xGM, xUb, params);
        event_t toScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(toScalar);
        WaitFlag<HardEvent::MTE3_S>(toScalar);
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        xGM.SetValue(TbsvXOffset(i, n, incx), xUb.GetValue(i));
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::LoadBand(
    LocalTensor<float>& aUb, int64_t offset, uint32_t count, uint32_t leftPadding)
{
    if constexpr (TRANS == TbsvTrans::NO_TRANS) {
        if (noTransCacheCols != 0) {
            // Eight leading floats permit an inactive x alignment prefix even
            // when the pivot is the first column in a newly loaded window.
            int64_t local = CachedBandOffset(offset) - leftPadding;
            uint32_t alignedCount = (count + leftPadding + 7) / 8 * 8;
            if (local % TBSV_ELEMENTS_PER_BLOCK == 0) {
                DataCopy(aUb, noTransCache[local], alignedCount);
            } else {
                Gather(
                    aUb, noTransCache, noTransIndices, static_cast<uint32_t>(local * sizeof(float)),
                    count + leftPadding);
            }
            PipeBarrier<PIPE_V>();
            return;
        }
    }
    if (cacheA || transCacheCols != 0 || transTileActive) {
        int64_t localOffset = CachedBandOffset(offset);
        if constexpr (TRANS == TbsvTrans::NO_TRANS) {
            if (localOffset >= leftPadding && (localOffset - leftPadding) % TBSV_ELEMENTS_PER_BLOCK == 0) {
                // Matching source/x block alignment permits a plain UB copy.
                // Extra prefix/suffix lanes never participate in the solve.
                uint32_t alignedCount = (count + leftPadding + 7) / 8 * 8;
                DataCopy(aUb, aCache[localOffset - leftPadding], alignedCount);
                PipeBarrier<PIPE_V>();
                return;
            }
        }
        uint32_t stride = CachedBandStride();
        if constexpr (TRANS == TbsvTrans::TRANS) {
            // Whole-column/full-A caches and separately staged row tiles do not use
            // gatherBand as coefficient storage. Reuse its existing eight
            // band-sized regions for the eight possible x alignment prefixes.
            // Gather's byte base offset supplies the changing source origin.
            if (cacheA || transCacheCols != 0 || (transTileActive && transTileSeparateIndices)) {
                if (!gatherIndicesReady || gatherIndexStride != stride) {
                    for (uint32_t padding = 0; padding < 8; ++padding) {
                        auto table = gatherBand.template ReinterpretCast<int32_t>()[padding * gatherIndexSpan];
                        ArithProgression(
                            table, -static_cast<int32_t>(padding * stride * sizeof(float)),
                            static_cast<int32_t>(stride * sizeof(float)), gatherIndexSpan);
                    }
                    PipeBarrier<PIPE_V>();
                    for (uint32_t padding = 0; padding < 8; ++padding) {
                        auto table = gatherBand.template ReinterpretCast<int32_t>()[padding * gatherIndexSpan];
                        Maxs(table, table, int32_t{0}, gatherIndexSpan);
                    }
                    PipeBarrier<PIPE_V>();
                    gatherIndexStride = stride;
                    gatherIndicesReady = true;
                }
                auto table = gatherBand.template ReinterpretCast<uint32_t>()[leftPadding * gatherIndexSpan];
                auto source = CachedBandSource();
                Gather(aUb, source, table, static_cast<uint32_t>(localOffset * sizeof(float)), count + leftPadding);
                PipeBarrier<PIPE_V>();
                return;
            }
        }
        auto offsets = scratch[6].template ReinterpretCast<int32_t>();
        int32_t first = static_cast<int32_t>(localOffset * sizeof(float));
        ArithProgression(
            offsets, first - static_cast<int32_t>(leftPadding * stride * sizeof(float)),
            static_cast<int32_t>(stride * sizeof(float)), count + leftPadding);
        PipeBarrier<PIPE_V>();
        Maxs(offsets, offsets, first, count + leftPadding);
        PipeBarrier<PIPE_V>();
        auto source = CachedBandSource();
        Gather(aUb, source, offsets.template ReinterpretCast<uint32_t>(), uint32_t{0}, count + leftPadding);
        // Gather's ordinary consumer is Vector. Scalar fallback reads the
        // cache directly; publish to Scalar only in the uncached caller below.
        PipeBarrier<PIPE_V>();
        return;
    }
    if constexpr (TRANS == TbsvTrans::TRANS) {
        // A partial edge group uses the original strided-DMA staging and
        // overwrites index storage. Rebuild tables if cached processing resumes.
        gatherIndicesReady = false;
        DataCopyExtParams strided{
            static_cast<uint16_t>(count), sizeof(float), (lda - 2) * static_cast<uint32_t>(sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(gatherBand, aGM[offset], strided, pad);
        event_t copyToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(copyToVector);
        WaitFlag<HardEvent::MTE2_V>(copyToVector);
        auto offsets = scratch[6].template ReinterpretCast<int32_t>();
        ArithProgression(offsets, -static_cast<int32_t>(leftPadding * 32), int32_t{32}, count + leftPadding);
        PipeBarrier<PIPE_V>();
        Maxs(offsets, offsets, int32_t{0}, count + leftPadding);
        PipeBarrier<PIPE_V>();
        Gather(aUb, gatherBand, offsets.template ReinterpretCast<uint32_t>(), uint32_t{0}, count + leftPadding);
        PipeBarrier<PIPE_V>();
        return;
    }
    DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    uint32_t total = leftPadding + count;
    uint32_t rightPadding = (TBSV_ELEMENTS_PER_BLOCK - (total % TBSV_ELEMENTS_PER_BLOCK)) % TBSV_ELEMENTS_PER_BLOCK;
    DataCopyPadExtParams<float> pad{
        leftPadding != 0 || rightPadding != 0, static_cast<uint8_t>(leftPadding), static_cast<uint8_t>(rightPadding),
        1.0f};
    DataCopyPad(aUb, aGM[offset], params, pad);
    event_t mte2ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(mte2ToVector);
    WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool OWNED>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::UpdateSegment(
    LocalTensor<float>& xUb, LocalTensor<float>& aUb, uint32_t xStart, int64_t aOffset, uint32_t count, float alpha)
{
    if (__builtin_isnan(alpha)) {
        // NaN updates affect only the active lanes. A bit mask avoids scalar
        // save/restore of the inactive alignment prefix and never reads A.
        uint32_t aligned = xStart / TBSV_ELEMENTS_PER_BLOCK * TBSV_ELEMENTS_PER_BLOCK;
        uint32_t left = xStart - aligned;
        uint32_t length = left + count;
        event_t scalarToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
        for (uint32_t block = 0; block < length; block += 64) {
            uint32_t lanes = length - block < 64 ? length - block : 64;
            uint64_t active = lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
            if (block == 0) {
                active &= ~((uint64_t{1} << left) - 1);
            }
            // Duplicate's bit-mask array is {low 64 bits, high 64 bits}.
            uint64_t mask[2] = {active, 0};
            Duplicate(xUb[aligned + block], alpha, mask, uint8_t{1}, uint16_t{1}, uint8_t{8});
        }
        ResetMask();
        event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
        return;
    }
    // The first matrix column is already known from the update interval;
    // do not recover it through a 64-bit division in a hot cache lookup.
    if constexpr (TRANS == TbsvTrans::TRANS) {
        currentBandColumn = xStart;
    } else if constexpr (DIAG == TbsvDiag::UNIT) {
        currentBandColumn = UPLO == TbsvUplo::UPPER ? xStart + count : xStart - 1;
    }
    if (!__builtin_isfinite(alpha)) {
        // With a non-finite multiplier the separate product/add has the same
        // IEEE result class as FMA. Preserve the inactive alignment prefix.
        uint32_t aligned = xStart / 8 * 8;
        uint32_t left = xStart - aligned;
        float prefix[7];
        for (uint32_t i = 0; i < left; ++i) {
            prefix[i] = xUb.GetValue(aligned + i);
        }
        event_t scalarToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
        LoadBand(aUb, aOffset, count, left);
        Axpy(xUb[aligned], aUb, alpha, left + count);
        event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
        for (uint32_t i = 0; i < left; ++i) {
            xUb.SetValue(aligned + i, prefix[i]);
        }
        xInputDirty = left != 0;
        return;
    }
    if (__builtin_fabsf(alpha) >= 0x1p-40f && count >= (OWNED ? 4 : 16)) {
        UpdateVector(xUb, aUb, xStart, aOffset, count, alpha);
        return;
    }
    if (cacheA || transCacheCols != 0 || transTileActive || noTransCacheCols != 0) {
        uint32_t stride = CachedBandStride();
        int64_t localOffset = CachedBandOffset(aOffset);
        auto source = CachedBandSource();
        for (uint32_t i = 0; i < count; ++i) {
            float value = StbsvArithmetic::Fma(
                alpha, source.GetValue(localOffset + static_cast<int64_t>(i) * stride), xUb.GetValue(xStart + i));
            xUb.SetValue(xStart + i, value);
        }
        xInputDirty = true;
        return;
    }
    LoadBand(aUb, aOffset, count);
    if constexpr (TRANS == TbsvTrans::TRANS) {
        // The uncached transpose coefficients were compacted by Gather,
        // so MTE2_S alone cannot publish the actual Scalar input.
        event_t gatherToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(gatherToScalar);
        WaitFlag<HardEvent::V_S>(gatherToScalar);
    }
    event_t copyToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(copyToScalar);
    WaitFlag<HardEvent::MTE2_S>(copyToScalar);
    for (uint32_t i = 0; i < count; ++i) {
        float value = StbsvArithmetic::Fma(alpha, aUb.GetValue(i), xUb.GetValue(xStart + i));
        xUb.SetValue(xStart + i, value);
    }
    xInputDirty = true;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::UpdateVector(
    LocalTensor<float>& xUb, LocalTensor<float>& aUb, uint32_t xStart, int64_t aOffset, uint32_t count, float alpha)
{
    uint32_t alignedStart = xStart / 8 * 8;
    uint32_t left = xStart - alignedStart;
    uint32_t length = left + count;
    LoadBand(aUb, aOffset, count, left);
    event_t scalarToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    if (xInputDirty) {
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
        xInputDirty = false;
    }
    float originalAlpha = alpha;
    float outputScale = 1.0f;
    bool scaled = __builtin_fabsf(alpha) > 0x1p90f;
    auto input = xUb[alignedStart];
    if (scaled) {
        // Normalized input temporarily replaces the persistent split mask.
        splitMaskReady = false;
        // Internal power-of-two normalization only: the original A/b and the
        // sequential FMA recurrence are unchanged. Safe lanes stay normal;
        // exceptional lanes are restored from exact unscaled integer FMA.
        StbsvArithmetic::FloatBits bits{alpha};
        uint32_t shift = ((bits.bits >> 23) & 255) - 127 - 60;
        bits.bits -= shift << 23;
        alpha = bits.value;
        StbsvArithmetic::FloatBits inputScaleBits;
        StbsvArithmetic::FloatBits outputScaleBits;
        inputScaleBits.bits = (127 - shift) << 23;
        outputScaleBits.bits = (127 + shift) << 23;
        outputScale = outputScaleBits.value;
        input = scratch[9];
        Muls(input, xUb[alignedStart], inputScaleBits.value, length);
        PipeBarrier<PIPE_V>();
    }
    StbsvArithmetic::FloatBits alphaHi{alpha};
    alphaHi.bits &= 0xfffff000;
    float alphaLo = alpha - alphaHi.value;
    auto hi = scratch[0];
    auto lo = scratch[1];
    auto product = scratch[2];
    auto residual = scratch[3];
    auto temp = scratch[4];
    auto sum = scratch[5];
    auto bp = scratch[6];
    auto error = scratch[7];
    auto tail = scratch[8];
    auto mask = bp.template ReinterpretCast<uint32_t>();
    if (scaled) {
        Duplicate(mask, uint32_t{0xfffff000}, length);
        PipeBarrier<PIPE_V>();
    } else {
        mask = scratch[9].template ReinterpretCast<uint32_t>();
        if (!splitMaskReady) {
            // Restore the entire allocation, not only this update's short
            // prefix: a subsequent unscaled segment may be longer.
            Duplicate(mask, uint32_t{0xfffff000}, splitMaskElements);
            PipeBarrier<PIPE_V>();
            splitMaskReady = true;
        }
    }
    And(hi.template ReinterpretCast<uint16_t>(), aUb.template ReinterpretCast<uint16_t>(),
        mask.template ReinterpretCast<uint16_t>(), length * 2);
    PipeBarrier<PIPE_V>();
    Sub(lo, aUb, hi, length);
    Muls(product, aUb, alpha, length);
    Muls(residual, hi, alphaHi.value, length);
    PipeBarrier<PIPE_V>();
    Sub(residual, residual, product, length);
    PipeBarrier<PIPE_V>();
    // Experimental shortening of the compensation chain. The A2 composite
    // multiply-add can round differently from separate Muls/Add; retain all
    // guard/repair logic and validate this branch independently against Netlib.
    Axpy(residual, hi, alphaLo, length);
    PipeBarrier<PIPE_V>();
    Axpy(residual, lo, alphaHi.value, length);
    PipeBarrier<PIPE_V>();
    Axpy(residual, lo, alphaLo, length);
    Add(sum, product, input, length);
    PipeBarrier<PIPE_V>();
    Sub(bp, sum, product, length);
    PipeBarrier<PIPE_V>();
    Sub(temp, sum, bp, length);
    PipeBarrier<PIPE_V>();
    Sub(error, product, temp, length);
    Sub(temp, input, bp, length);
    PipeBarrier<PIPE_V>();
    Add(error, error, temp, length);
    PipeBarrier<PIPE_V>();
    Add(tail, error, residual, length);
    PipeBarrier<PIPE_V>();
    uint32_t maskWords = (length + 15) / 16;
    // On dav-2201 the calCount comparison API requires 256B source length.
    // Its mask is reserved: a short count does not represent a masked tail.
    // All scratch buffers include initialized padding; only active bits are used.
    uint32_t compareCount = (length + 63) / 64 * 64;
    // Half-ULP and lower-binade quarter-ULP boundaries are normal powers of
    // two in the guarded exponent range. Conservatively mark every such tail,
    // without constructing an ULP threshold for each lane. Extra marked lanes
    // are repaired exactly, never approximately accepted.
    // The ordinary Dekker/TwoSum compensation remains in the same source order.
    // Marked lanes use the exact integer FMA rather than rounded-tail addition.
    auto midpoint = scratch[10].template ReinterpretCast<uint8_t>();
    // For a 32-bit lane, (bits << 9) == 0 exactly when its 23 mantissa
    // bits are zero. This is the same conservative predicate with one op.
    ShiftLeft(hi.template ReinterpretCast<uint32_t>(), tail.template ReinterpretCast<uint32_t>(), uint32_t{9}, length);
    PipeBarrier<PIPE_V>();
    CompareScalar(midpoint, hi.template ReinterpretCast<int32_t>(), int32_t{0}, CMPMODE::EQ, compareCount);
    PipeBarrier<PIPE_V>();
    Abs(lo, aUb, length);
    Abs(product, input, length);
    PipeBarrier<PIPE_V>();
    CompareScalar(bp.template ReinterpretCast<uint8_t>(), lo, 0x1p-20f, CMPMODE::GE, compareCount);
    CompareScalar(error.template ReinterpretCast<uint8_t>(), lo, 0x1p20f, CMPMODE::LE, compareCount);
    CompareScalar(temp.template ReinterpretCast<uint8_t>(), product, 0x1p110f, CMPMODE::LE, compareCount);
    PipeBarrier<PIPE_V>();
    And(bp.template ReinterpretCast<uint16_t>(), bp.template ReinterpretCast<uint16_t>(),
        error.template ReinterpretCast<uint16_t>(), maskWords);
    PipeBarrier<PIPE_V>();
    // Unscaled normal products have exponent >= -60 and exact-product quantum
    // >= 2^-106. A smaller addend cannot create a subnormal cancellation;
    // exact half-way products still take integer FMA via the power-tail mask.
    // Scaled products have an even higher minimum exponent (+40). If input
    // normalization loses a tiny original addend, it can only decide a product
    // midpoint; every scaled power tail, including zero, still repairs with
    // original unscaled operands. No midpoint/overflow repair is removed.
    And(bp.template ReinterpretCast<uint16_t>(), bp.template ReinterpretCast<uint16_t>(),
        temp.template ReinterpretCast<uint16_t>(), maskWords);
    event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(vectorToScalar);
    WaitFlag<HardEvent::V_S>(vectorToScalar);
    // Visit only exceptional lanes. In the usual all-safe block the scalar
    // controller reads two 64-bit packed masks, rather than scanning each lane.
    bool repairedVectorInput = false;
    bool scaledRepairs = false;
    for (uint32_t block = 0; block < length; block += 64) {
        uint32_t lanes = length - block < 64 ? length - block : 64;
        uint64_t active = lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
        if (block == 0) {
            active &= ~((uint64_t{1} << left) - 1);
        }
        uint64_t midpointBits = scratch[10].template ReinterpretCast<uint64_t>().GetValue(block / 64);
        uint64_t exceptional = (~bp.template ReinterpretCast<uint64_t>().GetValue(block / 64) | midpointBits) & active;
        while (exceptional != 0) {
            uint32_t bit = 63 - __builtin_clzll(exceptional);
            exceptional &= ~(uint64_t{1} << bit);
            uint32_t i = block + bit;
            float original = xUb.GetValue(alignedStart + i);
            float av = aUb.GetValue(i);
            float aa = __builtin_fabsf(av);
            float ac = __builtin_fabsf(input.GetValue(i));
            if (scaled) {
                hi.SetValue(i, StbsvArithmetic::Fma(originalAlpha, av, original));
                scaledRepairs = true;
            } else if (
                ((midpointBits & (uint64_t{1} << bit)) != 0 && tail.GetValue(i) != 0.0f) ||
                !(aa >= 0x1p-20f && aa <= 0x1p20f && ac <= 0x1p110f)) {
                sum.SetValue(i, StbsvArithmetic::Fma(alpha, av, original));
                tail.SetValue(i, 0.0f);
                repairedVectorInput = true;
            }
        }
    }
    // Scaled repairs are stored in hi and read only by Scalar after the final
    // V_S wait. Only ordinary sum/tail repairs introduce a new S->V dependency.
    if (repairedVectorInput) {
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
    }
    constexpr bool forward = (UPLO == TbsvUplo::LOWER && TRANS == TbsvTrans::NO_TRANS) ||
                             (UPLO == TbsvUplo::UPPER && TRANS == TbsvTrans::TRANS);
    if (forward || left == 0) {
        // The forward alignment prefix contains only already consumed pivots.
        // Their solutions live in outputX, so working-x prefix writes are dead.
        Add(xUb[alignedStart], sum, tail, length);
    } else {
        // Only the first block can contain an unsolved inactive prefix.
        // All subsequent lanes are contiguous and can share one counter call.
        uint32_t lanes = length < 64 ? length : 64;
        uint64_t active = (lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1) & ~((uint64_t{1} << left) - 1);
        uint64_t writeMask[2] = {active, 0};
        Add(xUb[alignedStart], sum, tail, writeMask, uint8_t{1}, BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        ResetMask();
        if (length > 64) {
            Add(xUb[alignedStart + 64], sum[64], tail[64], length - 64);
        }
    }
    if (scaled) {
        PipeBarrier<PIPE_V>();
        if (forward || left == 0) {
            Muls(xUb[alignedStart], xUb[alignedStart], outputScale, length);
        } else {
            // Preserve an unsolved alignment prefix in its original domain.
            // Mask both Add and rescaling; no Scalar prefix save/restore is needed.
            uint32_t lanes = length < 64 ? length : 64;
            uint64_t active = (lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1) & ~((uint64_t{1} << left) - 1);
            uint64_t writeMask[2] = {active, 0};
            Muls(
                xUb[alignedStart], xUb[alignedStart], outputScale, writeMask, uint8_t{1},
                UnaryRepeatParams{1, 1, 8, 8});
            ResetMask();
            if (length > 64) {
                Muls(xUb[alignedStart + 64], xUb[alignedStart + 64], outputScale, length - 64);
            }
        }
    }
    SetFlag<HardEvent::V_S>(vectorToScalar);
    WaitFlag<HardEvent::V_S>(vectorToScalar);
    if (scaledRepairs) {
        // Publish exact Scalar working-x repairs and hi scratch before Vector
        // reuse. When no lane was repaired, neither has a Scalar dependency.
        xInputDirty = true;
        for (uint32_t block = 0; block < length; block += 64) {
            uint32_t lanes = length - block < 64 ? length - block : 64;
            uint64_t active = lanes == 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
            if (block == 0) {
                active &= ~((uint64_t{1} << left) - 1);
            }
            uint64_t exceptional = (~bp.template ReinterpretCast<uint64_t>().GetValue(block / 64) |
                                    scratch[10].template ReinterpretCast<uint64_t>().GetValue(block / 64)) &
                                   active;
            while (exceptional != 0) {
                uint32_t bit = 63 - __builtin_clzll(exceptional);
                exceptional &= ~(uint64_t{1} << bit);
                xUb.SetValue(alignedStart + block + bit, hi.GetValue(block + bit));
            }
        }
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool WINDOW>
__aicore__ inline bool StbsvKernel<UPLO, TRANS, DIAG>::PublishNoTransPivot(uint32_t col, float& value)
{
    // Netlib STBSV tests X(J) != 0 before the NO_TRANS diagonal divide.
    if (value == 0.0f) {
        outputX.SetValue(col, value);
        return false;
    }
    if constexpr (DIAG == TbsvDiag::NON_UNIT) {
        if (!__builtin_isnan(value)) {
            value /= WINDOW ? GetSingleDiag(col) : GetDiag(col);
        }
    }
    outputX.SetValue(col, value);
    return true;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool WINDOW>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::ProcessNoTrans(LocalTensor<float>& xUb, LocalTensor<float>& aUb)
{
    if constexpr (UPLO == TbsvUplo::LOWER) {
        for (uint32_t col = 0; col < n; ++col) {
            float xValue = xUb.GetValue(col);
            if (k > 0 && __builtin_isnan(xValue)) {
                // Every following row is reachable through the first band.
                // NaN * A + x is NaN even when that coefficient is zero.
                UpdateSegment(outputX, aUb, col, 0, n - col, xValue);
                return;
            }
            if (!PublishNoTransPivot<WINDOW>(col, xValue)) {
                continue;
            }

            uint32_t count = (col + k + 1 < n) ? k : (n - col - 1);
            if (count > 0) {
                EnsureNoTransCache(col);
                int64_t aOffset = static_cast<int64_t>(col) * lda + 1;
                UpdateSegment(xUb, aUb, col + 1, aOffset, count, -xValue);
            }
        }
    } else {
        for (uint32_t col = n; col-- > 0;) {
            float xValue = xUb.GetValue(col);
            if (k > 0 && __builtin_isnan(xValue)) {
                UpdateSegment(outputX, aUb, 0, 0, col + 1, xValue);
                return;
            }
            if (!PublishNoTransPivot<WINDOW>(col, xValue)) {
                continue;
            }

            uint32_t rowStart = (col > k) ? (col - k) : 0;
            uint32_t count = col - rowStart;
            if (count > 0) {
                EnsureNoTransCache(col);
                int64_t aOffset = static_cast<int64_t>(col) * lda + k - count;
                UpdateSegment(xUb, aUb, rowStart, aOffset, count, -xValue);
            }
        }
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline int64_t StbsvKernel<UPLO, TRANS, DIAG>::CachedBandOffset(int64_t globalOffset) const
{
    if constexpr (TRANS == TbsvTrans::TRANS) {
        if (transTileActive) {
            return static_cast<int64_t>(currentBandColumn - transTileFirstCol) * transTilePitch + transTilePivot -
                   transTileFirstRow;
        }
    }
    if constexpr (TRANS == TbsvTrans::NO_TRANS) {
        if (noTransCacheCols != 0) {
            if constexpr (DIAG == TbsvDiag::UNIT) {
                int64_t col = currentBandColumn;
                uint32_t effectiveBand = k < n ? k : n - 1;
                int64_t bandRow = globalOffset - col * lda;
                return 8 + (col - noTransCacheBase) * unitCachePitch + bandRow -
                       (UPLO == TbsvUplo::UPPER ? k - effectiveBand : 1);
            }
            return 8 + globalOffset - static_cast<int64_t>(noTransCacheBase) * lda;
        }
    }
    if constexpr (DIAG == TbsvDiag::UNIT) {
        if (unitCachePitch != 0) {
            uint32_t effectiveBand = k < n ? k : n - 1;
            int64_t col = currentBandColumn;
            int64_t bandRow = globalOffset - col * lda;
            return col * unitCachePitch + bandRow - (UPLO == TbsvUplo::UPPER ? k - effectiveBand : 1);
        }
    }
    if constexpr (TRANS == TbsvTrans::TRANS) {
        return transCacheCols != 0 ? globalOffset - static_cast<int64_t>(transCacheBase) * lda : globalOffset;
    }
    return globalOffset;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline uint32_t StbsvKernel<UPLO, TRANS, DIAG>::CachedBandStride() const
{
    if constexpr (TRANS == TbsvTrans::NO_TRANS) {
        return 1;
    }
    if (transTileActive) {
        return transTilePitch;
    }
    if constexpr (DIAG == TbsvDiag::UNIT) {
        if (unitCachePitch != 0) {
            return unitCachePitch - 1;
        }
    }
    return lda - 1;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline LocalTensor<float> StbsvKernel<UPLO, TRANS, DIAG>::CachedBandSource() const
{
    if constexpr (TRANS == TbsvTrans::TRANS) {
        return transTileActive ? transTileStorage : (transCacheCols != 0 ? transCache : aCache);
    }
    return noTransCacheCols != 0 ? noTransCache : aCache;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::EnsureNoTransCache(uint32_t pivot)
{
    if (noTransCacheCols == 0 || (pivot >= noTransCacheBase && pivot < noTransCacheEnd)) {
        return;
    }
    if constexpr (UPLO == TbsvUplo::LOWER) {
        noTransCacheBase = pivot;
    } else {
        noTransCacheBase = pivot + 1 > noTransCacheCols ? pivot + 1 - noTransCacheCols : 0;
    }
    uint32_t columns = noTransCacheCols < n - noTransCacheBase ? noTransCacheCols : n - noTransCacheBase;
    noTransCacheEnd = noTransCacheBase + columns;
    event_t beforeVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(beforeVector);
    WaitFlag<HardEvent::V_MTE2>(beforeVector);
    event_t beforeScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(beforeScalar);
    WaitFlag<HardEvent::S_MTE2>(beforeScalar);
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    if constexpr (DIAG == TbsvDiag::UNIT) {
        uint32_t effectiveBand = k < n ? k : n - 1;
        uint32_t firstBandRow = UPLO == TbsvUplo::UPPER ? k - effectiveBand : 1;
        // Never load a UNIT diagonal, including padded physical columns.
        // Each off-diagonal block is padded to unitCachePitch in UB.
        DataCopyExtParams params{
            static_cast<uint16_t>(columns), effectiveBand * static_cast<uint32_t>(sizeof(float)),
            (lda - effectiveBand) * static_cast<uint32_t>(sizeof(float)), 0, 0};
        DataCopyPad(noTransCache[8], aGM[static_cast<int64_t>(noTransCacheBase) * lda + firstBandRow], params, pad);
    } else {
        DataCopyExtParams params{1, columns * lda * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(noTransCache[8], aGM[static_cast<int64_t>(noTransCacheBase) * lda], params, pad);
    }
    event_t toVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(toVector);
    WaitFlag<HardEvent::MTE2_V>(toVector);
    event_t toScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(toScalar);
    WaitFlag<HardEvent::MTE2_S>(toScalar);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool OWNED>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::EnsureTransCache(
    uint32_t pivot, uint32_t firstCol, uint32_t lastCol, uint32_t ownedBegin, uint32_t ownedEnd)
{
    if (transTileEnabled) {
        transTilePivot = pivot;
        // All budget-selected tile widths are powers of two.
        uint32_t firstRow = pivot & ~(transTileRows - 1);
        // Partial final groups read only their actual rows from GM and retain
        // the fixed UB pitch with per-block padding and dstStride below.
        uint32_t validRows = transTileRows < n - firstRow ? transTileRows : n - firstRow;
        transTileActive = true;
        if (transTileLoadedRow == firstRow) {
            return;
        }
        transTileFirstRow = firstRow;
        uint32_t lastRow = firstRow + validRows - 1;
        uint32_t lastTarget;
        int64_t sourceOffset;
        if constexpr (UPLO == TbsvUplo::UPPER) {
            transTileFirstCol = firstRow + 1;
            lastTarget = k < n - lastRow - 1 ? lastRow + k : n - 1;
            sourceOffset = static_cast<int64_t>(k) + firstRow + static_cast<int64_t>(transTileFirstCol) * (lda - 1);
        } else {
            transTileFirstCol = firstRow > k ? firstRow - k : 0;
            lastTarget = lastRow - 1;
            sourceOffset = firstRow + static_cast<int64_t>(transTileFirstCol) * (lda - 1);
        }
        if constexpr (OWNED) {
            // Each core updates only its fixed target partition. Cache the
            // union of this pivot tile intersected with that partition, not
            // every other core's coefficients. The partition is constant for
            // all pivots sharing transTileLoadedRow.
            if (transTileFirstCol < ownedBegin) {
                transTileFirstCol = ownedBegin;
            }
            if (lastTarget >= ownedEnd) {
                lastTarget = ownedEnd - 1;
            }
            sourceOffset = static_cast<int64_t>(firstRow) + static_cast<int64_t>(transTileFirstCol) * (lda - 1) +
                           (UPLO == TbsvUplo::UPPER ? k : 0);
        }
        uint32_t columns = lastTarget - transTileFirstCol + 1;
        // All referenced terms are in the union of the eight pivot bands.
        // Unused edge entries may cross a column but remain inside lda*n.
        // Only NON_UNIT uses this cache, so diagonal prefetch is permitted.
        event_t beforeCopy = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(beforeCopy);
        WaitFlag<HardEvent::V_MTE2>(beforeCopy);
        event_t beforeScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(beforeScalar);
        WaitFlag<HardEvent::S_MTE2>(beforeScalar);
        // GM uses the actual final row count; UB retains the fixed row pitch.
        // DataCopyPad rounds each UB block to 8 floats, then dstStride adds
        // the remaining complete 32B blocks. Gather never reads those gaps.
        uint32_t paddedRows = (validRows + 7) / 8 * 8;
        DataCopyExtParams params{
            static_cast<uint16_t>(columns), validRows * static_cast<uint32_t>(sizeof(float)),
            (lda - 1 - validRows) * static_cast<uint32_t>(sizeof(float)), (transTilePitch - paddedRows) / 8, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(transTileStorage, aGM[sourceOffset], params, pad);
        event_t afterCopy = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(afterCopy);
        WaitFlag<HardEvent::MTE2_V>(afterCopy);
        event_t afterScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(afterScalar);
        WaitFlag<HardEvent::MTE2_S>(afterScalar);
        transTileLoadedRow = firstRow;
        return;
    }
    if (transCacheCols == 0 || (firstCol >= transCacheBase && lastCol < transCacheEnd)) {
        return;
    }
    if constexpr (UPLO == TbsvUplo::UPPER) {
        transCacheBase = firstCol;
    } else {
        transCacheBase = lastCol + 1 > transCacheCols ? lastCol + 1 - transCacheCols : 0;
    }
    uint32_t columns = transCacheCols < n - transCacheBase ? transCacheCols : n - transCacheBase;
    transCacheEnd = transCacheBase + columns;
    // A single block has only one final 32-byte padding region: the cache
    // column pitch is exactly lda, unlike per-column non-aligned DMA blocks.
    // Capacity is derived from an already allocated buffer, including padding.
    event_t beforeCopy = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(beforeCopy);
    WaitFlag<HardEvent::V_MTE2>(beforeCopy);
    DataCopyExtParams params{1, columns * lda * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(transCache, aGM[static_cast<int64_t>(transCacheBase) * lda], params, pad);
    event_t afterCopy = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(afterCopy);
    WaitFlag<HardEvent::MTE2_V>(afterCopy);
    event_t afterScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(afterScalar);
    WaitFlag<HardEvent::MTE2_S>(afterScalar);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool WINDOW, bool CACHED_DIAGONAL>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::ProcessTrans(LocalTensor<float>& xUb, LocalTensor<float>& aUb)
{
    // A row update of A^T visits each target's terms in exactly the same order
    // as the Netlib transposed row solve, but updates independent targets together.
    // Unlike NO_TRANS, the transposed reference also divides zero pivots.
    if constexpr (UPLO == TbsvUplo::UPPER) {
        for (uint32_t col = 0; col < n; ++col) {
            float xValue = xUb.GetValue(col);
            if (k > 0 && __builtin_isnan(xValue)) {
                UpdateSegment(outputX, aUb, col, 0, n - col, xValue);
                return;
            }
            if constexpr (DIAG == TbsvDiag::NON_UNIT) {
                if (!__builtin_isnan(xValue)) {
                    xValue /= GetTransDiag<WINDOW, CACHED_DIAGONAL>(col);
                }
            }
            outputX.SetValue(col, xValue);
            uint32_t count = k < n - col - 1 ? k : n - col - 1;
            if (count > 0) {
                PrepareTransWindow<WINDOW>(col, col + 1, col + count);
                int64_t offset = k + static_cast<int64_t>(col) + static_cast<int64_t>(col + 1) * (lda - 1);
                UpdateSegment(xUb, aUb, col + 1, offset, count, -xValue);
            }
        }
    } else {
        for (uint32_t col = n; col-- > 0;) {
            float xValue = xUb.GetValue(col);
            if (k > 0 && __builtin_isnan(xValue)) {
                UpdateSegment(outputX, aUb, 0, 0, col + 1, xValue);
                return;
            }
            if constexpr (DIAG == TbsvDiag::NON_UNIT) {
                if (!__builtin_isnan(xValue)) {
                    xValue /= GetTransDiag<WINDOW, CACHED_DIAGONAL>(col);
                }
            }
            outputX.SetValue(col, xValue);
            uint32_t start = col > k ? col - k : 0;
            uint32_t count = col - start;
            if (count > 0) {
                PrepareTransWindow<WINDOW>(col, start, col - 1);
                int64_t offset = static_cast<int64_t>(col) + static_cast<int64_t>(start) * (lda - 1);
                UpdateSegment(xUb, aUb, start, offset, count, -xValue);
            }
        }
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::ProcessGlobal()
{
    // Constant-memory fallback: do not impose an undocumented n/k UB limit.
    constexpr bool forward = (UPLO == TbsvUplo::LOWER && TRANS == TbsvTrans::NO_TRANS) ||
                             (UPLO == TbsvUplo::UPPER && TRANS == TbsvTrans::TRANS);
    for (uint32_t step = 0; step < n; ++step) {
        uint32_t row = forward ? step : n - 1 - step;
        float value = xGM.GetValue(TbsvXOffset(row, n, incx));
        uint32_t available = forward ? row : n - row - 1;
        uint32_t count = k < available ? k : available;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t previous = forward ? row - count + i : row + count - i;
            float solved = xGM.GetValue(TbsvXOffset(previous, n, incx));
            if constexpr (TRANS == TbsvTrans::NO_TRANS) {
                if (solved == 0.0f) {
                    continue;
                }
            }
            uint32_t aRow = TRANS == TbsvTrans::NO_TRANS ? row : previous;
            uint32_t aCol = TRANS == TbsvTrans::NO_TRANS ? previous : row;
            int64_t offset;
            if constexpr (UPLO == TbsvUplo::UPPER) {
                offset = TbsvBandIdxUpper(aRow, aCol, k, lda);
            } else {
                offset = TbsvBandIdxLower(aRow, aCol, lda);
            }
            value = StbsvArithmetic::Fma(-aGM.GetValue(offset), solved, value);
        }
        // Netlib guards only the NO_TRANS divide against a zero pivot.
        if constexpr (DIAG == TbsvDiag::NON_UNIT) {
            if constexpr (TRANS == TbsvTrans::TRANS) {
                value /= GetDiag(row);
            } else if (value != 0.0f) {
                value /= GetDiag(row);
            }
        }
        xGM.SetValue(TbsvXOffset(row, n, incx), value);
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::ApplyOwnedUpdate(
    LocalTensor<float>& xUb, LocalTensor<float>& aUb, uint32_t pivot, uint32_t begin, uint32_t end, float xValue)
{
    if constexpr (TRANS == TbsvTrans::NO_TRANS) {
        if (xValue == 0.0f) {
            return;
        }
    }
    constexpr bool forward = (UPLO == TbsvUplo::LOWER && TRANS == TbsvTrans::NO_TRANS) ||
                             (UPLO == TbsvUplo::UPPER && TRANS == TbsvTrans::TRANS);
    uint32_t start = forward ? pivot + 1 : (pivot > k ? pivot - k : 0);
    uint32_t stop = forward ? (k < n - pivot - 1 ? pivot + k + 1 : n) : pivot;
    if (start < begin) {
        start = begin;
    }
    if (stop > end) {
        stop = end;
    }
    if (start >= stop) {
        return;
    }
    int64_t offset;
    if constexpr (TRANS == TbsvTrans::NO_TRANS) {
        EnsureNoTransCache(pivot);
        offset =
            static_cast<int64_t>(pivot) * lda + static_cast<int64_t>(start) - pivot + (UPLO == TbsvUplo::UPPER ? k : 0);
    } else {
        EnsureTransCache<true>(pivot, start, stop - 1, begin, end);
        offset =
            static_cast<int64_t>(pivot) + static_cast<int64_t>(start) * (lda - 1) + (UPLO == TbsvUplo::UPPER ? k : 0);
    }
    UpdateSegment<true>(xUb, aUb, start, offset, stop - start, -xValue);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::ProcessOwnedRanges(
    LocalTensor<float>& xUb, LocalTensor<float>& aUb)
{
    constexpr uint32_t panelWidth = 32;
    uint32_t peers = GetBlockNum();
    uint32_t span = ((n + peers * panelWidth - 1) / (peers * panelWidth)) * panelWidth;
    uint32_t id = GetBlockIdx();
    uint32_t begin = id * span < n ? id * span : n;
    uint32_t end = span < n - begin ? begin + span : n;
    constexpr bool forward = (UPLO == TbsvUplo::LOWER && TRANS == TbsvTrans::NO_TRANS) ||
                             (UPLO == TbsvUplo::UPPER && TRANS == TbsvTrans::TRANS);
    // A panel never straddles the aligned ownership boundary. Its owner solves
    // its pivots sequentially, updating only its own targets after each pivot.
    // The other cores apply the published pivots in the same order to their
    // disjoint targets. There is no reduction, reassociation or approximate solve.
    SyncAll<true>();
    for (uint32_t step = 0; step < n;) {
        uint32_t remaining = n - step;
        uint32_t panelCount = remaining < panelWidth ? remaining : panelWidth;
        if constexpr (!forward) {
            // Consume the partial high-end block first so subsequent UB source
            // addresses are 32B aligned, including every ownership boundary.
            if (remaining % panelWidth != 0) {
                panelCount = remaining % panelWidth;
            }
        }
        uint32_t panelStart = forward ? step : remaining - panelCount;
        uint32_t owner = panelStart / span;
        bool panelHasNaN = false;
        uint32_t nanPivot = 0;
        float nanValue = 0.0f;
        if (id == owner) {
            for (uint32_t lane = 0; lane < panelCount; ++lane) {
                uint32_t pivot = forward ? panelStart + lane : panelStart + panelCount - 1 - lane;
                float xValue = xUb.GetValue(pivot);
                // Keep Netlib's distinct zero-pivot order for N versus T/C.
                if (!__builtin_isnan(xValue) && (TRANS == TbsvTrans::TRANS || xValue != 0.0f)) {
                    xValue /= GetDiag(pivot);
                }
                outputX.SetValue(pivot, xValue);
                if (__builtin_isnan(xValue)) {
                    panelHasNaN = true;
                    nanPivot = pivot;
                    nanValue = xValue;
                    // Fill unsolved payload slots so every peer sees the same
                    // termination point without a reused flag/mailbox race.
                    for (uint32_t rest = lane + 1; rest < panelCount; ++rest) {
                        uint32_t row = forward ? panelStart + rest : panelStart + panelCount - 1 - rest;
                        outputX.SetValue(row, xValue);
                    }
                    break;
                }
                ApplyOwnedUpdate(xUb, aUb, pivot, begin, end, xValue);
            }
            event_t publish = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(publish);
            WaitFlag<HardEvent::S_MTE3>(publish);
            DataCopyExtParams params{1, panelCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(xGM[panelStart], outputX[panelStart], params);
            event_t finish = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(finish);
            WaitFlag<HardEvent::MTE3_S>(finish);
        }
        SyncAll<true>();
        if (id != owner) {
            DataCopyExtParams params{1, panelCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            // Non-owned output slots are dead locally and can hold the payload.
            // UpdateSegment only writes working x and scratch, not outputX.
            DataCopyPad(outputX[panelStart], xGM[panelStart], params, pad);
            event_t receive = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(receive);
            WaitFlag<HardEvent::MTE2_S>(receive);
            for (uint32_t lane = 0; lane < panelCount; ++lane) {
                uint32_t pivot = forward ? panelStart + lane : panelStart + panelCount - 1 - lane;
                float xValue = outputX.GetValue(pivot);
                if (__builtin_isnan(xValue)) {
                    panelHasNaN = true;
                    nanPivot = pivot;
                    nanValue = xValue;
                    break;
                }
                ApplyOwnedUpdate(xUb, aUb, pivot, begin, end, xValue);
            }
        }
        if (panelHasNaN) {
            // Rounded partitions can leave idle peers. They still take part in
            // every synchronization, but must never issue a zero-length DMA.
            if (begin == end) {
                return;
            }
            uint32_t nanBegin = forward ? (nanPivot > begin ? nanPivot : begin) : begin;
            uint32_t nanEnd = forward ? end : (nanPivot + 1 < end ? nanPivot + 1 : end);
            for (uint32_t row = nanBegin; row < nanEnd; ++row) {
                outputX.SetValue(row, nanValue);
            }
            event_t publish = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(publish);
            WaitFlag<HardEvent::S_MTE3>(publish);
            DataCopyExtParams params{1, (end - begin) * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(xGM[begin], outputX[begin], params);
            event_t finish = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(finish);
            WaitFlag<HardEvent::MTE3_S>(finish);
            return;
        }
        step += panelCount;
    }
    // Every solved pivot is already published in xGM. Do not store another
    // core's stale working/output slots over those final values.
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitVectorBuffers(
    TPipe& pipe, TBuf<TPosition::VECCALC>& xBuf, TBuf<TPosition::VECCALC>& outputBuf, TBuf<TPosition::VECCALC>& aBuf,
    uint32_t xElements, uint32_t bandElements)
{
    pipe.InitBuffer(xBuf, xElements * sizeof(float));
    pipe.InitBuffer(outputBuf, xElements * sizeof(float));
    outputX = outputBuf.Get<float>();
    pipe.InitBuffer(aBuf, bandElements * sizeof(float));
    auto aUb = aBuf.Get<float>();
    Duplicate(aUb, 0.0f, bandElements);
    event_t initializeBand = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(initializeBand);
    WaitFlag<HardEvent::V_MTE2>(initializeBand);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitUnitMatrixCache(
    TPipe& pipe, TBuf<TPosition::VECCALC>& cacheBuf, uint32_t effectiveBand, uint32_t xElements, uint32_t bandCopies,
    uint32_t bandElements)
{
    uint32_t pitch = (effectiveBand + 7) / 8 * 8;
    uint64_t offDiagonalElements = static_cast<uint64_t>(pitch) * n;
    if (!(effectiveBand != 0 && lda <= 0x3fffffff &&
          offDiagonalElements + uint64_t{2} * xElements + bandCopies * bandElements + 64 <= ubElements)) {
        return;
    }
    // Cache the off-diagonal band only; DMA padding determines the actual pitch.
    cacheA = true;
    unitCachePitch = pitch;
    pipe.InitBuffer(cacheBuf, offDiagonalElements * sizeof(float));
    aCache = cacheBuf.Get<float>();
    uint32_t firstBandRow = UPLO == TbsvUplo::UPPER ? k - effectiveBand : 1;
    DataCopyExtParams params{
        static_cast<uint16_t>(n), effectiveBand * static_cast<uint32_t>(sizeof(float)),
        (lda - effectiveBand) * static_cast<uint32_t>(sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(aCache, aGM[firstBandRow], params, pad);
    event_t toScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(toScalar);
    WaitFlag<HardEvent::MTE2_S>(toScalar);
    event_t toVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(toVector);
    WaitFlag<HardEvent::MTE2_V>(toVector);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitUnitColumnCache(
    TPipe& pipe, TBuf<TPosition::VECCALC>& cacheBuf, uint32_t effectiveBand, uint32_t xElements, uint32_t bandCopies,
    uint32_t bandElements)
{
    if (cacheA || effectiveBand < 16 || lda > 0x3fffffff) {
        return;
    }
    uint32_t pitch = (effectiveBand + 7) / 8 * 8;
    uint64_t used = uint64_t{2} * xElements + uint64_t{bandCopies} * bandElements + 64;
    uint64_t available = used < ubElements ? ubElements - used : 0;
    if (available <= bandElements + 16) {
        return;
    }
    uint32_t columns = (available - bandElements - 16) / pitch;
    if (columns > n) {
        columns = n;
    }
    if (columns < 2) {
        return;
    }
    unitCachePitch = pitch;
    noTransCacheCols = columns;
    uint32_t elements = bandElements + 16 + columns * pitch;
    pipe.InitBuffer(cacheBuf, elements * sizeof(float));
    auto storage = cacheBuf.Get<float>();
    noTransIndices = storage.template ReinterpretCast<uint32_t>();
    noTransCache = storage[bandElements];
    ArithProgression(noTransIndices.template ReinterpretCast<int32_t>(), int32_t{0}, int32_t{4}, bandElements);
    Duplicate(noTransCache, 0.0f, uint32_t{8});
    PipeBarrier<PIPE_V>();
    noTransCacheBase = n;
    noTransCacheEnd = n;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitMatrixCache(
    TPipe& pipe, TBuf<TPosition::VECCALC>& cacheBuf, uint32_t xElements, uint32_t bandCopies, uint32_t bandElements)
{
    uint64_t matrixElements = static_cast<uint64_t>(lda) * n;
    if (!(DIAG == TbsvDiag::NON_UNIT &&
          matrixElements + uint64_t{2} * xElements + bandCopies * bandElements + 64 <= ubElements)) {
        return;
    }
    cacheA = true;
    pipe.InitBuffer(cacheBuf, ((matrixElements + 7) / 8 * 8) * sizeof(float));
    aCache = cacheBuf.Get<float>();
    DataCopyExtParams params{1, static_cast<uint32_t>(matrixElements * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(aCache, aGM, params, pad);
    event_t copyToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(copyToScalar);
    WaitFlag<HardEvent::MTE2_S>(copyToScalar);
    event_t copyToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(copyToVector);
    WaitFlag<HardEvent::MTE2_V>(copyToVector);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline uint32_t StbsvKernel<UPLO, TRANS, DIAG>::SelectOwnedTransRows(
    uint32_t effectiveBand, uint64_t available, uint32_t& diagonalStageElements) const
{
    uint32_t peers = GetBlockNum();
    uint32_t ownedSpan = ((n + peers * 32 - 1) / (peers * 32)) * 32;
    for (uint32_t rows = 128; rows >= 8; rows /= 2) {
        uint32_t columns = effectiveBand + rows - 1 < n ? effectiveBand + rows - 1 : n;
        if (columns > ownedSpan) {
            columns = ownedSpan;
        }
        // Logical rows and UB pitch remain distinct; keep the 32B layout spacing.
        uint32_t tileElements = (rows + 8) * columns;
        if (lda > rows && tileElements <= available) {
            if (tileElements > diagonalStageElements) {
                diagonalStageElements = tileElements;
            }
            return rows;
        }
    }
    return 0;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline uint32_t StbsvKernel<UPLO, TRANS, DIAG>::SelectSingleTransRows(
    uint32_t effectiveBand, uint64_t available, uint32_t& diagonalStageElements) const
{
    uint32_t oldColumns = diagonalStageElements / lda;
    if (!(effectiveBand >= 16 && oldColumns < 2 * effectiveBand + TBSV_ELEMENTS_PER_BLOCK)) {
        return 0;
    }
    for (uint32_t rows = 128; rows >= 8; rows /= 2) {
        uint32_t columns = effectiveBand + rows - 1 < n ? effectiveBand + rows - 1 : n;
        uint32_t tileElements = (rows + 8) * columns;
        if (n >= rows && lda > rows && tileElements <= available) {
            if (tileElements > diagonalStageElements) {
                diagonalStageElements = tileElements;
            }
            return rows;
        }
    }
    return 0;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline LocalTensor<float> StbsvKernel<UPLO, TRANS, DIAG>::LoadDiagonalCache(
    TPipe& pipe, TBuf<TPosition::VECCALC>& diagonalBuf, TBuf<TPosition::VECCALC>& diagonalGatherBuf,
    LocalTensor<float>& xUb, uint32_t xElements, uint32_t diagonalStageElements)
{
    pipe.InitBuffer(diagonalBuf, xElements * sizeof(float));
    pipe.InitBuffer(diagonalGatherBuf, diagonalStageElements * sizeof(float));
    diagonalCache = diagonalBuf.Get<float>();
    auto gathered = diagonalGatherBuf.Get<float>();
    uint32_t diagonalRow = UPLO == TbsvUplo::UPPER ? k : 0;
    DataCopyExtParams params{
        static_cast<uint16_t>(n), sizeof(float), (lda - 1) * static_cast<uint32_t>(sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(gathered, aGM[diagonalRow], params, pad);
    event_t copyToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(copyToVector);
    WaitFlag<HardEvent::MTE2_V>(copyToVector);
    // x has not been loaded yet: reuse its storage for diagonal gather indices.
    ArithProgression(xUb.template ReinterpretCast<int32_t>(), int32_t{0}, int32_t{32}, n);
    PipeBarrier<PIPE_V>();
    Gather(diagonalCache, gathered, xUb.template ReinterpretCast<uint32_t>(), uint32_t{0}, n);
    event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(vectorToScalar);
    WaitFlag<HardEvent::V_S>(vectorToScalar);
    event_t vectorToCopy = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vectorToCopy);
    WaitFlag<HardEvent::V_MTE2>(vectorToCopy);
    return gathered;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitNoTransWindow(
    LocalTensor<float>& gathered, uint32_t available, uint32_t bandElements)
{
    if (available <= bandElements + 16) {
        return;
    }
    noTransCacheCols = (available - bandElements - 16) / lda;
    if (noTransCacheCols < 2) {
        noTransCacheCols = 0;
    }
    if (noTransCacheCols == 0) {
        return;
    }
    if (noTransCacheCols > n) {
        noTransCacheCols = n;
    }
    noTransCache = gathered[bandElements];
    noTransIndices = gathered.template ReinterpretCast<uint32_t>();
    ArithProgression(noTransIndices.template ReinterpretCast<int32_t>(), int32_t{0}, int32_t{4}, bandElements);
    Duplicate(noTransCache, 0.0f, uint32_t{8});
    PipeBarrier<PIPE_V>();
    noTransCacheBase = n;
    noTransCacheEnd = n;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline uint32_t StbsvKernel<UPLO, TRANS, DIAG>::SelectFallbackTransRows(
    uint32_t effectiveBand, uint32_t diagonalStageElements) const
{
    // Preserve the 32/16/8-row priority and every original eligibility condition.
    uint32_t union32 = effectiveBand + 31 < n ? effectiveBand + 31 : n;
    uint32_t unionCols = effectiveBand + 15 < n ? effectiveBand + 15 : n;
    if (effectiveBand >= 16 && n >= 32 && lda >= 33 && uint64_t{32} * union32 <= diagonalStageElements) {
        return 32;
    }
    if (effectiveBand >= 16 && uint64_t{16} * unionCols <= diagonalStageElements) {
        return 16;
    }
    uint32_t eightUnionCols = effectiveBand + 7 < n ? effectiveBand + 7 : n;
    if (effectiveBand >= 16 && uint64_t{8} * eightUnionCols <= diagonalStageElements) {
        return 8;
    }
    return 0;
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool PARALLEL>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitTransWindow(
    LocalTensor<float>& gathered, uint32_t effectiveBand, uint32_t diagonalStageElements, uint32_t ownedTransRows,
    uint32_t singleTransRows)
{
    uint32_t availableCols = diagonalStageElements / lda;
    if constexpr (PARALLEL) {
        transTileStorage = gathered;
        transTileRows = ownedTransRows;
        transTilePitch = ownedTransRows + 8;
        transTileSeparateIndices = true;
    } else if (singleTransRows != 0) {
        transTileStorage = gathered;
        transTileRows = singleTransRows;
        transTilePitch = singleTransRows + 8;
        transTileSeparateIndices = true;
    } else if (availableCols >= 2 * effectiveBand + TBSV_ELEMENTS_PER_BLOCK) {
        transCache = gathered;
        transCacheCols = availableCols < n ? availableCols : n;
        transCacheBase = n;
        transCacheEnd = n;
    } else {
        uint32_t rows = SelectFallbackTransRows(effectiveBand, diagonalStageElements);
        if (rows != 0) {
            transTileStorage = gathered;
            transTileRows = rows;
            transTilePitch = rows;
            transTileSeparateIndices = true;
        }
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool PARALLEL>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitDiagonalCache(
    TPipe& pipe, TBuf<TPosition::VECCALC>& diagonalBuf, TBuf<TPosition::VECCALC>& diagonalGatherBuf,
    LocalTensor<float>& xUb, uint32_t effectiveBand, uint32_t xElements, uint32_t bandCopies, uint32_t bandElements)
{
    if (!(DIAG == TbsvDiag::NON_UNIT && !cacheA && lda <= 0x3fffffff &&
          uint64_t{11} * xElements + static_cast<uint64_t>(bandCopies) * bandElements + 64 <= ubElements)) {
        return;
    }
    cacheDiagonal = true;
    uint32_t diagonalStageElements = xElements * 8;
    uint32_t ownedTransRows = 0;
    uint32_t singleTransRows = 0;
    if constexpr (TRANS == TbsvTrans::TRANS) {
        uint64_t available = ubElements - uint64_t{3} * xElements - uint64_t{bandCopies} * bandElements - 64;
        if constexpr (PARALLEL) {
            ownedTransRows = SelectOwnedTransRows(effectiveBand, available, diagonalStageElements);
        } else {
            singleTransRows = SelectSingleTransRows(effectiveBand, available, diagonalStageElements);
        }
    }
    auto gathered = LoadDiagonalCache(pipe, diagonalBuf, diagonalGatherBuf, xUb, xElements, diagonalStageElements);
    if constexpr (TRANS == TbsvTrans::NO_TRANS) {
        if (effectiveBand >= 16) {
            InitNoTransWindow(gathered, xElements * 8, bandElements);
        }
    } else {
        InitTransWindow<PARALLEL>(gathered, effectiveBand, diagonalStageElements, ownedTransRows, singleTransRows);
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool WINDOW>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::PrepareTransWindow(
    uint32_t pivot, uint32_t firstCol, uint32_t lastCol)
{
    if constexpr (WINDOW) {
        WaitTransWindowReaders(firstCol, lastCol);
    }
    EnsureTransCache(pivot, firstCol, lastCol);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::WaitTransWindowReaders(uint32_t firstCol, uint32_t lastCol)
{
    if constexpr (DIAG == TbsvDiag::NON_UNIT) {
        if (k >= 16 || transCacheCols == 0 || (firstCol >= transCacheBase && lastCol < transCacheEnd)) {
            return;
        }
        // Narrow windows have Scalar coefficient/diagonal consumers as well.
        event_t beforeScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(beforeScalar);
        WaitFlag<HardEvent::S_MTE2>(beforeScalar);
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitNarrowWindow(
    TPipe& pipe, TBuf<TPosition::VECCALC>& cacheBuf, uint32_t effectiveBand, uint32_t xElements, uint32_t bandCopies,
    uint32_t bandElements)
{
    if constexpr (DIAG == TbsvDiag::NON_UNIT) {
        if (cacheA || cacheDiagonal || noTransCacheCols != 0 || transCacheCols != 0 || incx != 1 ||
            effectiveBand == 0 || effectiveBand >= 16 || lda > 0x3fffffff || lda > 2 * (effectiveBand + 1)) {
            return;
        }
        // Only previously uncached narrow inputs use this bounded 32-KiB window.
        uint64_t used = uint64_t{2} * xElements + uint64_t{bandCopies} * bandElements + 64;
        uint32_t available = used < ubElements ? static_cast<uint32_t>(ubElements - used) : 0;
        if (available > STBSV_MAX_NARROW_WINDOW_ELEMENTS) {
            available = STBSV_MAX_NARROW_WINDOW_ELEMENTS;
        }
        available = available / TBSV_ELEMENTS_PER_BLOCK * TBSV_ELEMENTS_PER_BLOCK;
        uint32_t minimumCols = TRANS == TbsvTrans::NO_TRANS ? 2 : 2 * effectiveBand + 8;
        uint32_t reserved = TRANS == TbsvTrans::NO_TRANS ? bandElements + 16 : 16;
        if (uint64_t{reserved} + uint64_t{minimumCols} * lda > available) {
            return;
        }
        pipe.InitBuffer(cacheBuf, available * sizeof(float));
        auto gathered = cacheBuf.Get<float>();
        if constexpr (TRANS == TbsvTrans::NO_TRANS) {
            InitNoTransWindow(gathered, available, bandElements);
        } else {
            InitTransWindow<false>(gathered, effectiveBand, available - 16, 0, 0);
        }
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::InitScratch(
    TPipe& pipe, TBuf<TPosition::VECCALC> (&scratchBuf)[11], uint32_t bandElements)
{
    for (uint32_t i = 0; i < 11; ++i) {
        pipe.InitBuffer(scratchBuf[i], bandElements * sizeof(float));
        scratch[i] = scratchBuf[i].Get<float>();
        if (i == 9) {
            Duplicate(scratch[i].template ReinterpretCast<uint32_t>(), uint32_t{0xfffff000}, bandElements);
        } else {
            Duplicate(scratch[i], 0.0f, bandElements);
        }
    }
    splitMaskElements = bandElements;
    splitMaskReady = true;
    PipeBarrier<PIPE_V>();
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool PARALLEL, bool WINDOW>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::RunCachedSolve(
    TPipe& pipe, LocalTensor<float>& xUb, LocalTensor<float>& aUb, TBuf<TPosition::VECCALC>& gatherBuf,
    uint32_t effectiveBand, uint32_t bandElements)
{
    if constexpr (TRANS == TbsvTrans::NO_TRANS) {
        if constexpr (PARALLEL) {
            ProcessOwnedRanges(xUb, aUb);
        } else {
            ProcessNoTrans<WINDOW>(xUb, aUb);
        }
    } else {
        pipe.InitBuffer(gatherBuf, bandElements * 8 * sizeof(float));
        gatherBand = gatherBuf.Get<float>();
        gatherIndexSpan = bandElements;
        if (!transTileSeparateIndices) {
            transTileStorage = gatherBand;
        }
        // The original live gather allocation and all eligibility bounds remain unchanged.
        transTileEnabled =
            DIAG == TbsvDiag::NON_UNIT && !cacheA && transCacheCols == 0 && effectiveBand >= 16 && lda <= 0x3fffffff;
        if constexpr (PARALLEL) {
            ProcessOwnedRanges(xUb, aUb);
        } else if constexpr (WINDOW) {
            ProcessTrans<true, false>(xUb, aUb);
        } else if (cacheDiagonal) {
            ProcessTrans<false, true>(xUb, aUb);
        } else {
            ProcessTrans<false, false>(xUb, aUb);
        }
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
template <bool PARALLEL, bool WINDOW>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::Process(TPipe& pipe)
{
    // Tiny or over-budget inputs retain the constant-memory ordered recurrence.
    if (n <= 4) {
        if constexpr (!PARALLEL) {
            ProcessGlobal();
        } else if (GetBlockIdx() == 0) {
            ProcessGlobal();
        }
        return;
    }
    uint32_t effectiveBand = k < n ? k : n - 1;
    uint32_t bandCopies = TRANS == TbsvTrans::TRANS ? 20 : 12;
    uint32_t bandElements = (effectiveBand + 7 + 63) / 64 * 64;
    if (uint64_t{2} * ((n + 7) / 8 * 8) + static_cast<uint64_t>(bandCopies) * bandElements + 16 > ubElements ||
        (TRANS == TbsvTrans::TRANS && lda > 0x3fffffff)) {
        if constexpr (!PARALLEL) {
            ProcessGlobal();
        } else if (GetBlockIdx() == 0) {
            ProcessGlobal();
        }
        return;
    }
    uint32_t xElements = ((n + TBSV_ELEMENTS_PER_BLOCK - 1) / TBSV_ELEMENTS_PER_BLOCK) * TBSV_ELEMENTS_PER_BLOCK;
    // All buffer owners remain alive through the solve. Helpers only borrow them.
    TBuf<TPosition::VECCALC> xBuf, outputBuf, aBuf, cacheBuf, diagonalBuf, diagonalGatherBuf, gatherBuf;
    TBuf<TPosition::VECCALC> scratchBuf[11];
    InitVectorBuffers(pipe, xBuf, outputBuf, aBuf, xElements, bandElements);
    auto xUb = xBuf.Get<float>();
    auto aUb = aBuf.Get<float>();
    if constexpr (DIAG == TbsvDiag::UNIT) {
        InitUnitMatrixCache(pipe, cacheBuf, effectiveBand, xElements, bandCopies, bandElements);
        if constexpr (TRANS == TbsvTrans::NO_TRANS) {
            InitUnitColumnCache(pipe, cacheBuf, effectiveBand, xElements, bandCopies, bandElements);
        }
    }
    InitMatrixCache(pipe, cacheBuf, xElements, bandCopies, bandElements);
    InitDiagonalCache<PARALLEL>(
        pipe, diagonalBuf, diagonalGatherBuf, xUb, effectiveBand, xElements, bandCopies, bandElements);
    if constexpr (WINDOW) {
        InitNarrowWindow(pipe, cacheBuf, effectiveBand, xElements, bandCopies, bandElements);
    }
    LoadX(xUb);
    InitScratch(pipe, scratchBuf, bandElements);
    RunCachedSolve<PARALLEL, WINDOW>(pipe, xUb, aUb, gatherBuf, effectiveBand, bandElements);
    // Solved pivots must not race with vector prefix reads or other cores' slots.
    if constexpr (!PARALLEL) {
        StoreX(outputX);
    }
}

#define DEFINE_TBSV_KERNEL(uplo, trans, diag, name)                       \
    __global__ __aicore__ void name(StbsvTilingData tiling)               \
    {                                                                     \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                   \
        TPipe pipe;                                                       \
        StbsvKernel<TbsvUplo::uplo, TbsvTrans::trans, TbsvDiag::diag> op; \
        op.Init(tiling);                                                  \
        op.Process(pipe);                                                 \
    }

DEFINE_TBSV_KERNEL(LOWER, NO_TRANS, NON_UNIT, stbsv_kernel_lower_no_trans_non_unit)
DEFINE_TBSV_KERNEL(LOWER, NO_TRANS, UNIT, stbsv_kernel_lower_no_trans_unit)
DEFINE_TBSV_KERNEL(UPPER, NO_TRANS, NON_UNIT, stbsv_kernel_upper_no_trans_non_unit)
DEFINE_TBSV_KERNEL(UPPER, NO_TRANS, UNIT, stbsv_kernel_upper_no_trans_unit)
DEFINE_TBSV_KERNEL(LOWER, TRANS, NON_UNIT, stbsv_kernel_lower_trans_non_unit)
DEFINE_TBSV_KERNEL(LOWER, TRANS, UNIT, stbsv_kernel_lower_trans_unit)
DEFINE_TBSV_KERNEL(UPPER, TRANS, NON_UNIT, stbsv_kernel_upper_trans_non_unit)
DEFINE_TBSV_KERNEL(UPPER, TRANS, UNIT, stbsv_kernel_upper_trans_unit)

#undef DEFINE_TBSV_KERNEL

#define DEFINE_TBSV_PARALLEL_KERNEL(uplo, trans, name)                          \
    __schedmode__(1) __global__ __mix__(0, 1) void name(StbsvTilingData tiling) \
    {                                                                           \
        TPipe pipe;                                                             \
        StbsvKernel<TbsvUplo::uplo, TbsvTrans::trans, TbsvDiag::NON_UNIT> op;   \
        op.Init(tiling);                                                        \
        op.Process<true>(pipe);                                                 \
    }

DEFINE_TBSV_PARALLEL_KERNEL(LOWER, NO_TRANS, stbsv_kernel_lower_no_trans_non_unit_owned)
DEFINE_TBSV_PARALLEL_KERNEL(UPPER, NO_TRANS, stbsv_kernel_upper_no_trans_non_unit_owned)
DEFINE_TBSV_PARALLEL_KERNEL(LOWER, TRANS, stbsv_kernel_lower_trans_non_unit_owned)
DEFINE_TBSV_PARALLEL_KERNEL(UPPER, TRANS, stbsv_kernel_upper_trans_non_unit_owned)
#undef DEFINE_TBSV_PARALLEL_KERNEL

#define DEFINE_TBSV_WINDOW_KERNEL(uplo, trans, name)                          \
    __global__ __aicore__ void name(StbsvTilingData tiling)                   \
    {                                                                         \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                       \
        TPipe pipe;                                                           \
        StbsvKernel<TbsvUplo::uplo, TbsvTrans::trans, TbsvDiag::NON_UNIT> op; \
        op.Init(tiling);                                                      \
        op.Process<false, true>(pipe);                                        \
    }
DEFINE_TBSV_WINDOW_KERNEL(LOWER, NO_TRANS, stbsv_kernel_lower_no_trans_non_unit_window)
DEFINE_TBSV_WINDOW_KERNEL(UPPER, NO_TRANS, stbsv_kernel_upper_no_trans_non_unit_window)
DEFINE_TBSV_WINDOW_KERNEL(LOWER, TRANS, stbsv_kernel_lower_trans_non_unit_window)
DEFINE_TBSV_WINDOW_KERNEL(UPPER, TRANS, stbsv_kernel_upper_trans_non_unit_window)
#undef DEFINE_TBSV_WINDOW_KERNEL

static bool UseNarrowWindowKernel(const StbsvTilingData& tiling)
{
    const uint64_t ubElements = tiling.ubSizeBytes / sizeof(float);
    if (tiling.n <= 4 || tiling.diag != ACLBLAS_NON_UNIT || tiling.incx != 1) {
        return false;
    }
    uint32_t effective = tiling.k < tiling.n ? tiling.k : tiling.n - 1;
    if (effective == 0 || effective >= 16 || tiling.lda > 2 * (effective + 1)) {
        return false;
    }
    uint32_t xElements = (tiling.n + 7) / 8 * 8;
    uint32_t bandElements = (effective + 70) / 64 * 64;
    uint32_t copies = tiling.trans == ACLBLAS_OP_N ? 12 : 20;
    uint64_t used = uint64_t{2} * xElements + uint64_t{copies} * bandElements + 64;
    if (uint64_t{tiling.lda} * tiling.n + used <= ubElements ||
        uint64_t{11} * xElements + uint64_t{copies} * bandElements + 64 <= ubElements) {
        return false;
    }
    uint32_t available = used < ubElements ? static_cast<uint32_t>(ubElements - used) : 0;
    if (available > STBSV_MAX_NARROW_WINDOW_ELEMENTS) {
        available = STBSV_MAX_NARROW_WINDOW_ELEMENTS;
    }
    available = available / TBSV_ELEMENTS_PER_BLOCK * TBSV_ELEMENTS_PER_BLOCK;
    uint32_t minimumCols = tiling.trans == ACLBLAS_OP_N ? 2 : 2 * effective + 8;
    uint32_t reserved = tiling.trans == ACLBLAS_OP_N ? bandElements + 16 : 16;
    return uint64_t{reserved} + uint64_t{minimumCols} * tiling.lda <= available;
}

static void LaunchNarrowWindowKernel(const StbsvTilingData& tiling, void* stream)
{
    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            stbsv_kernel_lower_no_trans_non_unit_window<<<1, nullptr, stream>>>(tiling);
        } else {
            stbsv_kernel_lower_trans_non_unit_window<<<1, nullptr, stream>>>(tiling);
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            stbsv_kernel_upper_no_trans_non_unit_window<<<1, nullptr, stream>>>(tiling);
        } else {
            stbsv_kernel_upper_trans_non_unit_window<<<1, nullptr, stream>>>(tiling);
        }
    }
}

static bool UseParallelKernel(const StbsvTilingData& tiling)
{
    const uint64_t ubElements = tiling.ubSizeBytes / sizeof(float);
    uint32_t effective = tiling.k < tiling.n ? tiling.k : tiling.n - 1;
    uint32_t xElements = (tiling.n + 7) / 8 * 8;
    uint32_t bandElements = (effective + 70) / 64 * 64;
    uint32_t copies = tiling.trans == ACLBLAS_OP_N ? 12 : 20;
    return tiling.diag == ACLBLAS_NON_UNIT && tiling.incx == 1 && tiling.n >= 512 && effective >= 128 &&
           uint64_t{4} * effective >= tiling.n &&
           (tiling.trans == ACLBLAS_OP_N || tiling.n >= 1024 || uint64_t{2} * effective >= tiling.n) &&
           uint64_t{2} * xElements + uint64_t{copies} * bandElements + 16 <= ubElements &&
           (tiling.trans == ACLBLAS_OP_N || tiling.lda <= 0x3fffffff);
}

static void LaunchParallelKernel(const StbsvTilingData& tiling, void* stream)
{
    uint32_t blocks = tiling.trans != ACLBLAS_OP_N && tiling.n < 1024 ? 4 : 8;
    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            stbsv_kernel_lower_no_trans_non_unit_owned<<<blocks, nullptr, stream>>>(tiling);
        } else {
            stbsv_kernel_lower_trans_non_unit_owned<<<blocks, nullptr, stream>>>(tiling);
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            stbsv_kernel_upper_no_trans_non_unit_owned<<<blocks, nullptr, stream>>>(tiling);
        } else {
            stbsv_kernel_upper_trans_non_unit_owned<<<blocks, nullptr, stream>>>(tiling);
        }
    }
    return;
}

static void LaunchSingleKernel(const StbsvTilingData& tiling, void* stream)
{
    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stbsv_kernel_lower_no_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stbsv_kernel_lower_no_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stbsv_kernel_lower_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stbsv_kernel_lower_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stbsv_kernel_upper_no_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stbsv_kernel_upper_no_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stbsv_kernel_upper_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stbsv_kernel_upper_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        }
    }
}

void stbsv_kernel_do(const StbsvTilingData& tiling, void* stream)
{
    if (UseParallelKernel(tiling)) {
        LaunchParallelKernel(tiling, stream);
        return;
    }
    if (UseNarrowWindowKernel(tiling)) {
        LaunchNarrowWindowKernel(tiling, stream);
        return;
    }
    LaunchSingleKernel(tiling, stream);
}
