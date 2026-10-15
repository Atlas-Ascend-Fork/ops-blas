/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "chemm_kernel.h"

namespace {

template <bool kFull, event_t kVToMte2Event>
__aicore__ inline __attribute__((always_inline)) void LoadAivComplexTile(
    __gm__ float* source, uint32_t ld, uint32_t rows, uint32_t cols, __ubuf__ float* complexTile)
{
    if constexpr (!kFull) {
        vector_dup(complexTile, 0.0f, 2U * kVectorRepeats, 1U, 1U, 8U, 0U);
    }
    set_flag(PIPE_V, PIPE_MTE2, kVToMte2Event);
    wait_flag(PIPE_V, PIPE_MTE2, kVToMte2Event);
    if constexpr (kFull) {
        copy_gm_to_ubuf_align_b32(
            complexTile, source, 0U, kMicro, kComplexColumnFloats * sizeof(float), 0U, 0U,
            (ld - kMicro) * 2U * sizeof(float), 0U);
        return;
    }
    const uint32_t floats = 2U * rows;
    const uint32_t alignedFloats = (floats + 7U) & ~7U;
    copy_gm_to_ubuf_align_b32(
        complexTile, source, 0U, static_cast<uint16_t>(cols), floats * sizeof(float), 0U,
        static_cast<uint8_t>(alignedFloats - floats), (2U * ld - floats) * sizeof(float),
        (kComplexColumnFloats - alignedFloats) / 8U);
}

template <bool kFull>
__aicore__ inline __attribute__((always_inline)) void PrepareGridTile(
    uint32_t row, uint32_t col, uint32_t ld, uint32_t rows, uint32_t cols, __ubuf__ float* raw)
{
    if constexpr (!kFull) {
        const uint32_t validRows = row < rows ? (rows - row < kMicro ? rows - row : kMicro) : 0U;
        const uint32_t validCols = col < cols ? (cols - col < kMicro ? cols - col : kMicro) : 0U;
        if (validRows != kMicro || validCols != kMicro || (ld & 3U) != 0U) {
            vector_dup(raw, 0.0f, 2U * kVectorRepeats, 1U, 1U, 8U, 0U);
        }
    }
}

template <bool kFull>
__aicore__ inline __attribute__((always_inline)) void CopyGridTile(
    __gm__ float* source, uint32_t row, uint32_t col, uint32_t ld, uint32_t rows, uint32_t cols, __ubuf__ float* raw)
{
    if constexpr (kFull) {
        copy_gm_to_ubuf_align_b32(
            raw, source + 2U * ElementIndex(row, col, ld), 0U, kMicro, kComplexColumnFloats * sizeof(float), 0U, 0U,
            (ld - kMicro) * 2U * sizeof(float), 0U);
        return;
    }
    const uint32_t validRows = row < rows ? (rows - row < kMicro ? rows - row : kMicro) : 0U;
    const uint32_t validCols = col < cols ? (cols - col < kMicro ? cols - col : kMicro) : 0U;
    if (validRows == 0U || validCols == 0U) {
        return;
    }
    if (validRows == kMicro && validCols == kMicro && (ld & 3U) == 0U) {
        copy_gm_to_ubuf_align_b32(
            raw, source + 2U * ElementIndex(row, col, ld), 0U, kMicro, kComplexColumnFloats * sizeof(float), 0U, 0U,
            (ld - kMicro) * 2U * sizeof(float), 0U);
        return;
    }
    const uint32_t floats = 2U * validRows;
    const uint32_t alignedFloats = (floats + 7U) & ~7U;
    copy_gm_to_ubuf_align_b32(
        raw, source + 2U * ElementIndex(row, col, ld), 0U, static_cast<uint16_t>(validCols), floats * sizeof(float), 0U,
        static_cast<uint8_t>(alignedFloats - floats), (2U * ld - floats) * sizeof(float),
        (kComplexColumnFloats - alignedFloats) / 8U);
}

template <bool kLower>
__aicore__ inline __attribute__((always_inline)) void GridHermitianSource(
    uint32_t row, uint32_t col, uint32_t& sourceRow, uint32_t& sourceCol)
{
    const bool reflected = kLower ? row < col : row > col;
    sourceRow = reflected ? col : row;
    sourceCol = reflected ? row : col;
}

template <bool kRight, bool kLower, bool kFull, bool kSkipY, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void PrepareGridPair(
    const Tiling& tiling, uint32_t xRow, uint32_t xCol, uint32_t yRow, uint32_t yCol, __ubuf__ float* xRaw,
    __ubuf__ float* yRaw)
{
    if constexpr (kRight) {
        PrepareGridTile<kFull>(
            xRow, xCol, static_cast<uint32_t>(tiling.ldb), static_cast<uint32_t>(tiling.m),
            static_cast<uint32_t>(tiling.n), xRaw);
        if constexpr (!kSkipY) {
            uint32_t sourceRow;
            uint32_t sourceCol;
            GridHermitianSource<kLower>(yRow, yCol, sourceRow, sourceCol);
            PrepareGridTile<kFull>(
                sourceRow, sourceCol, static_cast<uint32_t>(tiling.lda), static_cast<uint32_t>(tiling.dimA),
                static_cast<uint32_t>(tiling.dimA), yRaw);
        }
    } else {
        uint32_t sourceRow;
        uint32_t sourceCol;
        GridHermitianSource<kLower>(xRow, xCol, sourceRow, sourceCol);
        PrepareGridTile<kFull>(
            sourceRow, sourceCol, static_cast<uint32_t>(tiling.lda), static_cast<uint32_t>(tiling.dimA),
            static_cast<uint32_t>(tiling.dimA), xRaw);
        if constexpr (!kSkipY) {
            PrepareGridTile<kFull>(
                yRow, yCol, static_cast<uint32_t>(tiling.ldb), static_cast<uint32_t>(tiling.m),
                static_cast<uint32_t>(tiling.n), yRaw);
        }
    }
}

template <bool kRight, bool kLower, bool kFull, bool kSkipY, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void CopyGridPair(
    __gm__ float* a, __gm__ float* b, const Tiling& tiling, uint32_t xRow, uint32_t xCol, uint32_t yRow, uint32_t yCol,
    __ubuf__ float* xRaw, __ubuf__ float* yRaw)
{
    if constexpr (kRight) {
        CopyGridTile<kFull>(
            b, xRow, xCol, static_cast<uint32_t>(tiling.ldb), static_cast<uint32_t>(tiling.m),
            static_cast<uint32_t>(tiling.n), xRaw);
        if constexpr (!kSkipY) {
            uint32_t sourceRow;
            uint32_t sourceCol;
            GridHermitianSource<kLower>(yRow, yCol, sourceRow, sourceCol);
            CopyGridTile<kFull>(
                a, sourceRow, sourceCol, static_cast<uint32_t>(tiling.lda), static_cast<uint32_t>(tiling.dimA),
                static_cast<uint32_t>(tiling.dimA), yRaw);
        }
    } else {
        uint32_t sourceRow;
        uint32_t sourceCol;
        GridHermitianSource<kLower>(xRow, xCol, sourceRow, sourceCol);
        CopyGridTile<kFull>(
            a, sourceRow, sourceCol, static_cast<uint32_t>(tiling.lda), static_cast<uint32_t>(tiling.dimA),
            static_cast<uint32_t>(tiling.dimA), xRaw);
        if constexpr (!kSkipY) {
            CopyGridTile<kFull>(
                b, yRow, yCol, static_cast<uint32_t>(tiling.ldb), static_cast<uint32_t>(tiling.m),
                static_cast<uint32_t>(tiling.n), yRaw);
        }
    }
}

enum class GridPhase { ISSUE, COMMIT, CONSUME };

template <GridPhase kPhase, event_t kEvent>
__aicore__ inline __attribute__((always_inline)) void SyncGridPair()
{
    if constexpr (kPhase == GridPhase::ISSUE) {
        set_flag(PIPE_V, PIPE_MTE2, kEvent);
        wait_flag(PIPE_V, PIPE_MTE2, kEvent);
    } else if constexpr (kPhase == GridPhase::COMMIT) {
        set_flag(PIPE_MTE2, PIPE_V, kEvent);
    } else {
        wait_flag(PIPE_MTE2, PIPE_V, kEvent);
    }
}

template <bool kRight, bool kLower, uint32_t kBank, bool kSkipY>
__aicore__ inline __attribute__((always_inline)) void DecodeGridPair(
    uint32_t xRow, uint32_t xCol, uint32_t yRow, uint32_t yCol, __ubuf__ float* xRaw, __ubuf__ float* yRaw,
    __ubuf__ float* leftReal, __ubuf__ float* leftImag, __ubuf__ float* rightReal, __ubuf__ float* rightImag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    if constexpr (kRight) {
        CInterleavedToPlanes(xRaw, leftReal, leftImag);
        if constexpr (!kSkipY) {
            DecodeHermitianPlanes16<kLower>(
                yRow, yCol, yRaw, rightReal, rightImag, transReal, transImag, triangleMask, diagonalMask);
        }
    } else {
        DecodeHermitianPlanes16<kLower>(
            xRow, xCol, xRaw, leftReal, leftImag, transReal, transImag, triangleMask, diagonalMask);
        if constexpr (!kSkipY) {
            CInterleavedToPlanes(yRaw, rightReal, rightImag);
        }
    }
}

template <bool kLower, bool kFull>
__aicore__ inline __attribute__((always_inline)) void LoadAivHermitianPlanes(
    __gm__ float* a, __gm__ uint8_t* maskGm, const ChemmAivTilingData& tiling, __ubuf__ float* complexTile,
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t dimA = static_cast<uint32_t>(tiling.dimA);
    LoadAivComplexTile<kFull, EVENT_ID2>(a, static_cast<uint32_t>(tiling.lda), dimA, dimA, complexTile);
    constexpr uint64_t kTriangleOffset = kLower ? 32U : 0U;
    constexpr uint32_t kMaskBytes = kLower ? 64U : 96U;
    copy_gm_to_ubuf_align_b8(triangleMask, maskGm + kTriangleOffset, 0U, 1U, kMaskBytes, 0U, 0U, 0U, 0U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(transReal), reinterpret_cast<__ubuf__ uint32_t*>(complexTile), nullptr, 8U,
        1U, 1U, 8U, 8U);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(transImag), reinterpret_cast<__ubuf__ uint32_t*>(complexTile), nullptr, 8U,
        1U, 2U, 8U, 8U);
    pipe_barrier(PIPE_V);
    TransposePlane16(transReal, real);
    TransposePlane16(transImag, imag);
    pipe_barrier(PIPE_V);
    vmuls(transImag, transImag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
    ApplyHermitianMasks(real, imag, transReal, transImag, triangleMask, diagonalMask);
}

template <bool kFull, bool kColumns, event_t kReuseEvent, event_t kReadyEvent>
__aicore__ inline __attribute__((always_inline)) void LoadAivDensePlanes(
    __gm__ float* source, const ChemmAivTilingData& tiling, uint32_t ld, __ubuf__ float* raw, __ubuf__ float* real,
    __ubuf__ float* imag)
{
    LoadAivComplexTile<kFull, kReuseEvent>(
        source, ld, static_cast<uint32_t>(tiling.m), static_cast<uint32_t>(tiling.n), raw);
    set_flag(PIPE_MTE2, PIPE_V, kReadyEvent);
    wait_flag(PIPE_MTE2, PIPE_V, kReadyEvent);
    if constexpr (kColumns) {
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(real), reinterpret_cast<__ubuf__ uint32_t*>(raw), nullptr, 8U, 1U, 1U,
            8U, 8U);
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(imag), reinterpret_cast<__ubuf__ uint32_t*>(raw), nullptr, 8U, 1U, 2U,
            8U, 8U);
    } else {
        CInterleavedToPlanes(raw, real, imag);
    }
}

template <bool kRight, bool kLower, bool kFull>
__aicore__ inline __attribute__((always_inline)) void PrepareAivTinyOperands(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, __ubuf__ float* raw, __ubuf__ float* rowReal,
    __ubuf__ float* rowImag, __ubuf__ float* auxReal, __ubuf__ float* auxImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    LoadAivHermitianPlanes<kLower, kFull>(
        a, reinterpret_cast<__gm__ uint8_t*>(g_packMasks), tiling, raw, rowReal, rowImag, auxReal, auxImag,
        triangleMask, diagonalMask);
    pipe_barrier(PIPE_V);
    if constexpr (!kRight) {
        TransposePlane16(rowReal, auxReal);
        TransposePlane16(rowImag, auxImag);
    }
    LoadAivDensePlanes<kFull, kRight, EVENT_ID3, EVENT_ID1>(
        b, tiling, static_cast<uint32_t>(tiling.ldb), raw, kRight ? auxReal : rowReal, kRight ? auxImag : rowImag);
    pipe_barrier(PIPE_V);
}

// A partial K bank contains eight rows, each with sixteen output columns.
constexpr uint32_t kLocalBankElements = kAivRowsPerVectorCore * kMicro;
constexpr uint32_t kLocalPartElements = kMicro * kLocalBankElements;
constexpr uint32_t kLocalBroadcastElements = kMicro * kAivRowsPerVectorCore * kFp32DataBlockElements;

__aicore__ inline void BroadcastPlanes(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag,
    uint8_t repeats)
{
    vbrcb(
        reinterpret_cast<__ubuf__ uint32_t*>(broadcastReal), reinterpret_cast<__ubuf__ uint32_t*>(real), 1U, 8U,
        repeats);
    vbrcb(
        reinterpret_cast<__ubuf__ uint32_t*>(broadcastImag), reinterpret_cast<__ubuf__ uint32_t*>(imag), 1U, 8U,
        repeats);
    pipe_barrier(PIPE_V);
}

__aicore__ inline void BroadcastLocalRows(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* broadcasts, __ubuf__ float* temporary, uint32_t rowBase,
    uint32_t validK)
{
    // Compact [K,16] -> [K,8]; BRCB advances by eight source FP32 per repeat.
    copy_ubuf_to_ubuf(temporary, real + rowBase, 0U, static_cast<uint16_t>(validK), 1U, 1U, 0U);
    copy_ubuf_to_ubuf(temporary + kLocalBankElements, imag + rowBase, 0U, static_cast<uint16_t>(validK), 1U, 1U, 0U);
    pipe_barrier(PIPE_V);
    BroadcastPlanes(
        temporary, temporary + kLocalBankElements, broadcasts, broadcasts + kLocalBroadcastElements,
        static_cast<uint8_t>(validK));
}

template <bool kInit>
__aicore__ inline void AccumulateVectorRows(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* sums, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negative, uint32_t validK, uint8_t broadcastStride)
{
    const uint8_t repeats = static_cast<uint8_t>(validK);
    __ubuf__ float* sumImag = sums + kLocalPartElements;
    if constexpr (kInit) {
        vmul(sums, real, broadcastReal, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
        vmul(sums + 8U, real + 8U, broadcastReal, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
        vmul(sumImag, imag, broadcastReal, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
        vmul(sumImag + 8U, imag + 8U, broadcastReal, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
    } else {
        vmla(sums, real, broadcastReal, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
        vmla(sums + 8U, real + 8U, broadcastReal, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
        vmla(sumImag, imag, broadcastReal, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
        vmla(sumImag + 8U, imag + 8U, broadcastReal, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
    }
    pipe_barrier(PIPE_V);
    vmla(sums, negative, broadcastImag, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
    vmla(sums + 8U, negative + 8U, broadcastImag, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
    vmla(sumImag, real, broadcastImag, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
    vmla(sumImag + 8U, real + 8U, broadcastImag, repeats, 2U, 0U, 1U, 16U, 2U, broadcastStride);
}

__aicore__ inline void ClearUnusedBanks(__ubuf__ float* sums, uint32_t validK, uint32_t banks)
{
    if (banks != validK) {
        const uint8_t repeats = static_cast<uint8_t>(2U * (banks - validK));
        vector_dup(sums + validK * kLocalBankElements, 0.0f, repeats, 1U, 1U, 8U, 0U);
        vector_dup(sums + kLocalPartElements + validK * kLocalBankElements, 0.0f, repeats, 1U, 1U, 8U, 0U);
    }
}

template <uint32_t kStride>
__aicore__ inline void ReduceLocalLevel(__ubuf__ float* sums, uint32_t banks)
{
    const uint8_t repeats = static_cast<uint8_t>(banks / (2U * kStride));
    if (repeats == 0U) {
        return;
    }
    // The last level has one repeat; no uint8 stride truncation is allowed.
    constexpr uint8_t repeatStride = static_cast<uint8_t>(kStride == 8U ? 0U : 32U * kStride);
    constexpr uint32_t offset = kStride * kLocalBankElements;
    __ubuf__ float* imag = sums + kLocalPartElements;
    pipe_barrier(PIPE_V);
    vadd(sums, sums, sums + offset, repeats, 2U, 2U, 2U, repeatStride, repeatStride, repeatStride);
    vadd(sums + 8U, sums + 8U, sums + offset + 8U, repeats, 2U, 2U, 2U, repeatStride, repeatStride, repeatStride);
    vadd(imag, imag, imag + offset, repeats, 2U, 2U, 2U, repeatStride, repeatStride, repeatStride);
    vadd(imag + 8U, imag + 8U, imag + offset + 8U, repeats, 2U, 2U, 2U, repeatStride, repeatStride, repeatStride);
}

__aicore__ inline void ReduceLocalRows(__ubuf__ float* sums, uint32_t validK, uint32_t banks)
{
    ClearUnusedBanks(sums, validK, banks);
    ReduceLocalLevel<1U>(sums, banks);
    ReduceLocalLevel<2U>(sums, banks);
    ReduceLocalLevel<4U>(sums, banks);
    ReduceLocalLevel<8U>(sums, banks);
    pipe_barrier(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void StoreAivTinyResult(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t rowBase, uint32_t cols, uint8_t rows,
    __ubuf__ float* productReal, __ubuf__ float* productImag, __ubuf__ float* raw)
{
    PlanesToCInterleaved<1U>(productReal, productImag, raw, kComplexColumnFloats);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    StoreComplexTail<kComplexColumnFloats>(c + 2U * rowBase, raw, cols, rows, static_cast<uint32_t>(tiling.ldc));
    pipe_barrier(PIPE_MTE3);
}

template <bool kRight, bool kLower>
__aicore__ inline void RunAivTiny(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* raw,
    __ubuf__ float* rowPlanes, __ubuf__ float* auxPlanes, __ubuf__ float* accumulators, __ubuf__ float* broadcasts,
    __ubuf__ float* negativeVectorImag, __ubuf__ uint8_t* masks)
{
    __ubuf__ float* rowReal = rowPlanes;
    __ubuf__ float* rowImag = rowPlanes + kMicroElements;
    __ubuf__ float* auxReal = auxPlanes;
    __ubuf__ float* auxImag = auxPlanes + kMicroElements;
    __ubuf__ uint8_t* triangleMask = masks + (kLower ? 32U : 0U);
    __ubuf__ uint8_t* diagonalMask = masks + 64U;
    if ((tiling.kernelFlags & CHEMM_FLAG_FULL) != 0U) {
        PrepareAivTinyOperands<kRight, kLower, true>(
            a, b, tiling, raw, rowReal, rowImag, auxReal, auxImag, triangleMask, diagonalMask);
    } else {
        PrepareAivTinyOperands<kRight, kLower, false>(
            a, b, tiling, raw, rowReal, rowImag, auxReal, auxImag, triangleMask, diagonalMask);
    }
    const uint32_t rowBase = static_cast<uint32_t>(get_block_idx()) * kAivRowsPerVectorCore;
    const uint8_t rows = static_cast<uint8_t>(
        static_cast<uint32_t>(tiling.m) - rowBase < kAivRowsPerVectorCore ? static_cast<uint32_t>(tiling.m) - rowBase :
                                                                            kAivRowsPerVectorCore);
    const uint32_t cols = static_cast<uint32_t>(tiling.n);
    const uint32_t validK = static_cast<uint32_t>(tiling.dimA);
    BroadcastLocalRows(auxReal, auxImag, broadcasts, raw, rowBase, validK);
    set_vector_mask(0ULL, kFullFp32Mask);
    vmuls(negativeVectorImag, rowImag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
    AccumulateVectorRows<true>(
        rowReal, rowImag, accumulators, broadcasts, broadcasts + kLocalBroadcastElements, negativeVectorImag, validK,
        kAivRowsPerVectorCore);
    ReduceLocalRows(accumulators, validK, tiling.reductionBanks);
    StoreAivTinyResult(c, tiling, rowBase, cols, rows, accumulators, accumulators + kLocalPartElements, raw);
}

__aicore__ inline void LoadDenseVectorCache(
    __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t col, uint32_t validCols, uint32_t kFeeds,
    __ubuf__ float* broadcasts, __ubuf__ float* yCacheReal, __ubuf__ float* yCacheImag)
{
    const uint32_t stageRows = kFeeds * kMicro;
    const uint32_t validFloats = 2U * static_cast<uint32_t>(tiling.dimA);
    const uint32_t alignedFloats = (validFloats + 7U) & ~7U;
    const uint32_t rawStride = 2U * stageRows;
    copy_gm_to_ubuf_align_b32(
        broadcasts, b + 2U * ElementIndex(0U, col, static_cast<uint32_t>(tiling.ldb)), 0U,
        static_cast<uint16_t>(validCols), validFloats * sizeof(float), 0U,
        static_cast<uint8_t>(alignedFloats - validFloats),
        (2U * static_cast<uint32_t>(tiling.ldb) - validFloats) * sizeof(float), (rawStride - alignedFloats) / 8U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    const uint32_t tailFloats = rawStride - validFloats;
    if (tailFloats != 0U) {
        const uint32_t prefix = validFloats - (validFloats & ~7U);
        set_mask_norm();
        set_vector_mask(0ULL, ((1ULL << tailFloats) - 1ULL) << prefix);
        vector_dup(
            broadcasts + (validFloats & ~7U), 0.0f, static_cast<uint8_t>(validCols), 1U, 1U,
            static_cast<uint8_t>(rawStride / 8U), 0U);
        set_vector_mask(0ULL, kFullFp32Mask);
    }
    CInterleavedToPlanes(
        broadcasts, yCacheReal, yCacheImag, kMicro, rawStride,
        static_cast<uint8_t>(stageRows / kAivRowsPerTransposeRepeat));
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void LoadAivGridCache(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t col, uint32_t validCols,
    uint32_t kFeeds, __ubuf__ float* raw, __ubuf__ float* broadcasts, __ubuf__ float* yCacheReal,
    __ubuf__ float* yCacheImag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask, __ubuf__ uint8_t* masks)
{
    copy_gm_to_ubuf_align_b8(masks, reinterpret_cast<__gm__ uint8_t*>(g_packMasks), 0U, 1U, 96U, 0U, 0U, 0U, 0U);
    if constexpr (kRight) {
        for (uint32_t feed = 0U; feed < kFeeds; ++feed) {
            const uint32_t kBase = feed * kMicro;
            uint32_t sourceRow;
            uint32_t sourceCol;
            GridHermitianSource<kLower>(kBase, col, sourceRow, sourceCol);
            PrepareGridTile<false>(
                sourceRow, sourceCol, static_cast<uint32_t>(tiling.lda), static_cast<uint32_t>(tiling.dimA),
                static_cast<uint32_t>(tiling.dimA), raw);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
            CopyGridTile<false>(
                a, sourceRow, sourceCol, static_cast<uint32_t>(tiling.lda), static_cast<uint32_t>(tiling.dimA),
                static_cast<uint32_t>(tiling.dimA), raw);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
            const uint32_t offset = kBase * kMicro;
            DecodeHermitianPlanes16<kLower>(
                kBase, col, raw, yCacheReal + offset, yCacheImag + offset, transReal, transImag, triangleMask,
                diagonalMask);
        }

    } else {
        LoadDenseVectorCache(b, tiling, col, validCols, kFeeds, broadcasts, yCacheReal, yCacheImag);
    }
}

template <bool kRight, bool kLower>
__aicore__ inline void DecodeAivGridBank(
    uint32_t bank, uint32_t alignedRow, uint32_t kBase, __ubuf__ float* raw, __ubuf__ float* leftReal,
    __ubuf__ float* leftImag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    if (bank == 0U) {
        SyncGridPair<GridPhase::CONSUME, EVENT_ID2>();
    } else {
        SyncGridPair<GridPhase::CONSUME, EVENT_ID5>();
    }
    if constexpr (kRight) {
        CInterleavedToPlanes(raw, leftReal, leftImag);
    } else {
        DecodeHermitianPlanes16<kLower>(
            alignedRow, kBase, raw, leftReal, leftImag, transReal, transImag, triangleMask, diagonalMask);
    }
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void SubmitNextAivGridBank(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t bank, uint32_t alignedRow,
    uint32_t nextK, uint32_t col, __ubuf__ float* nextRaw)
{
    PrepareGridPair<kRight, kLower, false, true>(
        tiling, alignedRow, nextK, nextK, col, nextRaw, nextRaw + kComplexTileFloats);
    if (bank == 0U) {
        SyncGridPair<GridPhase::ISSUE, EVENT_ID2>();
        CopyGridPair<kRight, kLower, false, true>(
            a, b, tiling, alignedRow, nextK, nextK, col, nextRaw, nextRaw + kComplexTileFloats);
        SyncGridPair<GridPhase::COMMIT, EVENT_ID2>();
    } else {
        SyncGridPair<GridPhase::ISSUE, EVENT_ID5>();
        CopyGridPair<kRight, kLower, false, true>(
            a, b, tiling, alignedRow, nextK, nextK, col, nextRaw, nextRaw + kComplexTileFloats);
        SyncGridPair<GridPhase::COMMIT, EVENT_ID5>();
    }
}

template <bool kInit, uint32_t kRows>
__aicore__ inline void AccumulateGridRows(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* sums, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negative, uint32_t validK)
{
    AccumulateVectorRows<kInit>(real, imag, sums, broadcastReal, broadcastImag, negative, validK, kRows);
    if constexpr (kInit) {
        ClearUnusedBanks(sums, validK, kMicro);
    }
    if constexpr (kRows == kMicro) {
        __ubuf__ float* next = sums + 2U * kLocalPartElements;
        AccumulateVectorRows<kInit>(
            real, imag, next, broadcastReal + 64U, broadcastImag + 64U, negative, validK, kRows);
        if constexpr (kInit) {
            ClearUnusedBanks(next, validK, kMicro);
        }
    }
}

template <bool kRight, bool kLower, uint32_t kRows>
__aicore__ inline void AccumulateAivGridFeed(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t alignedRow, uint32_t rowOffset,
    uint32_t col, uint32_t feed, uint32_t kFeeds, __ubuf__ float* rawSlots, __ubuf__ float* leftReal,
    __ubuf__ float* leftImag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ float* accumulators,
    __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag, __ubuf__ float* negativeImag,
    __ubuf__ float* yCacheReal, __ubuf__ float* yCacheImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    set_mask_norm();
    set_vector_mask(0ULL, kFullFp32Mask);
    const uint32_t kBase = feed * kMicro;
    const uint32_t bank = feed & 1U;
    const uint32_t validK =
        static_cast<uint32_t>(tiling.dimA) - kBase < kMicro ? static_cast<uint32_t>(tiling.dimA) - kBase : kMicro;
    __ubuf__ float* raw = rawSlots + bank * 2U * kComplexTileFloats;
    DecodeAivGridBank<kRight, kLower>(
        bank, alignedRow, kBase, raw, leftReal, leftImag, transReal, transImag, triangleMask, diagonalMask);
    if (feed + 1U < kFeeds) {
        const uint32_t nextBank = bank ^ 1U;
        SubmitNextAivGridBank<kRight, kLower>(
            a, b, tiling, nextBank, alignedRow, kBase + kMicro, col, rawSlots + nextBank * 2U * kComplexTileFloats);
    }
    pipe_barrier(PIPE_V);
    TransposePlane16(leftReal, transReal);
    TransposePlane16(leftImag, transImag);
    pipe_barrier(PIPE_V);
    if constexpr (kRows == kAivRowsPerVectorCore) {
        BroadcastLocalRows(transReal, transImag, broadcastReal, raw, rowOffset, validK);
    } else {
        BroadcastPlanes(transReal, transImag, broadcastReal, broadcastImag, static_cast<uint8_t>(2U * validK));
    }
    __ubuf__ float* vectorReal = yCacheReal + kBase * kMicro;
    __ubuf__ float* vectorImag = yCacheImag + kBase * kMicro;
    vmuls(negativeImag, vectorImag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
    if (feed == 0U) {
        AccumulateGridRows<true, kRows>(
            vectorReal, vectorImag, accumulators, broadcastReal, broadcastImag, negativeImag, validK);
    } else {
        AccumulateGridRows<false, kRows>(
            vectorReal, vectorImag, accumulators, broadcastReal, broadcastImag, negativeImag, validK);
    }
    pipe_barrier(PIPE_V);
}

template <bool kRight, bool kLower, uint32_t kRows>
__aicore__ inline void AccumulateAivGridFeeds(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t alignedRow, uint32_t rowOffset,
    uint32_t col, uint32_t kFeeds, __ubuf__ float* rawSlots, __ubuf__ float* leftReal, __ubuf__ float* leftImag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ float* accumulators, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negativeImag, __ubuf__ float* yCacheReal, __ubuf__ float* yCacheImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    PrepareGridPair<kRight, kLower, false, true>(
        tiling, alignedRow, 0U, 0U, col, rawSlots, rawSlots + kComplexTileFloats);
    SyncGridPair<GridPhase::ISSUE, EVENT_ID2>();
    CopyGridPair<kRight, kLower, false, true>(
        a, b, tiling, alignedRow, 0U, 0U, col, rawSlots, rawSlots + kComplexTileFloats);
    SyncGridPair<GridPhase::COMMIT, EVENT_ID2>();
    for (uint32_t feed = 0U; feed < kFeeds; ++feed) {
        AccumulateAivGridFeed<kRight, kLower, kRows>(
            a, b, tiling, alignedRow, rowOffset, col, feed, kFeeds, rawSlots, leftReal, leftImag, transReal, transImag,
            accumulators, broadcastReal, broadcastImag, negativeImag, yCacheReal, yCacheImag, triangleMask,
            diagonalMask);
    }
}

template <uint32_t kRows>
__aicore__ inline void StoreAivGridResult(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validRows,
    uint32_t validCols, __ubuf__ float* accumulators, __ubuf__ float* outputTile)
{
    ReduceLocalRows(accumulators, kMicro, kMicro);
    if constexpr (kRows == kMicro) {
        __ubuf__ float* next = accumulators + 2U * kLocalPartElements;
        ReduceLocalRows(next, kMicro, kMicro);
        // Both trees have finished. Bank1 is dead and joins the second row group.
        copy_ubuf_to_ubuf(accumulators + kLocalBankElements, next, 0U, 1U, 16U, 0U, 0U);
        copy_ubuf_to_ubuf(
            accumulators + kLocalPartElements + kLocalBankElements, next + kLocalPartElements, 0U, 1U, 16U, 0U, 0U);
        pipe_barrier(PIPE_V);
    }
    PlanesToCInterleaved<kRows / kAivRowsPerVectorCore>(
        accumulators, accumulators + kLocalPartElements, outputTile, kComplexColumnFloats);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    StoreComplexTail<kComplexColumnFloats>(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(tiling.ldc)), outputTile, validCols, validRows,
        static_cast<uint32_t>(tiling.ldc));
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

template <bool kRight, bool kLower, uint32_t kRows = kMicro>
__aicore__ inline void RunAivGrid(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* rawSlots,
    __ubuf__ float* leftPlanes, __ubuf__ float* leftTranspose, __ubuf__ float* accumulators, __ubuf__ float* broadcasts,
    __ubuf__ float* negativeVectorImag, __ubuf__ float* outputTile, __ubuf__ float* yCache, __ubuf__ uint8_t* masks)
{
    static_assert(
        kRows == kMicro || kRows == kAivRowsPerVectorCore, "AIV Grid rows must match a vector broadcast batch");
    constexpr uint32_t kBroadcastElements = kRows * kFp32DataBlockElements;

    __ubuf__ float* leftReal = leftPlanes;
    __ubuf__ float* leftImag = leftPlanes + kMicroElements;
    __ubuf__ float* leftTransReal = leftTranspose;
    __ubuf__ float* leftTransImag = leftTranspose + kMicroElements;
    __ubuf__ float* broadcastReal = broadcasts;
    __ubuf__ float* broadcastImag = broadcasts + kMicro * kBroadcastElements;
    __ubuf__ float* yCacheReal = yCache;
    __ubuf__ float* yCacheImag = yCache + kAivYCachePlaneElements;
    __ubuf__ uint8_t* triangleMask = masks + (kLower ? 32U : 0U);
    __ubuf__ uint8_t* diagonalMask = masks + 64U;

    const uint32_t core = static_cast<uint32_t>(get_block_idx());
    const uint32_t tileCol = core / tiling.tileRows;
    const uint32_t tileRow = core - tileCol * tiling.tileRows;
    const uint32_t row = tileRow * kRows;
    const uint32_t col = tileCol * kMicro;
    const uint32_t alignedRow = row & ~(kMicro - 1U);
    const uint32_t rowOffset = row & (kMicro - 1U);
    const uint32_t validRows =
        static_cast<uint32_t>(tiling.m) - row < kRows ? static_cast<uint32_t>(tiling.m) - row : kRows;
    const uint32_t validCols =
        static_cast<uint32_t>(tiling.n) - col < kMicro ? static_cast<uint32_t>(tiling.n) - col : kMicro;
    const uint32_t kFeeds = (static_cast<uint32_t>(tiling.dimA) + kMicro - 1U) / kMicro;
    LoadAivGridCache<kRight, kLower>(
        a, b, tiling, col, validCols, kFeeds, rawSlots, broadcasts, yCacheReal, yCacheImag, leftTransReal,
        leftTransImag, triangleMask, diagonalMask, masks);
    AccumulateAivGridFeeds<kRight, kLower, kRows>(
        a, b, tiling, alignedRow, rowOffset, col, kFeeds, rawSlots, leftReal, leftImag, leftTransReal, leftTransImag,
        accumulators, broadcastReal, broadcastImag, negativeVectorImag, yCacheReal, yCacheImag, triangleMask,
        diagonalMask);
    StoreAivGridResult<kRows>(c, tiling, row, col, validRows, validCols, accumulators, outputTile);
}

__aicore__ inline __attribute__((always_inline)) void SplitOrderedFmaOperands(
    __ubuf__ float* src00, __ubuf__ float* src01, __ubuf__ float* src10, __ubuf__ float* src11,
    __ubuf__ float* product0, __ubuf__ float* product1, __ubuf__ float* highA0, __ubuf__ float* highA1,
    __ubuf__ float* lowA0, __ubuf__ float* lowA1, __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0,
    __ubuf__ float* lowB1, __ubuf__ float* temporary0, __ubuf__ float* temporary1, uint8_t rows,
    uint8_t src0BlockStride, uint8_t src1BlockStride, uint16_t src0RepeatStride, uint16_t src1RepeatStride)
{
    constexpr float kSplitter = 4097.0f;
    vmul(product0, src00, src10, rows, 1U, src0BlockStride, src1BlockStride, 2U, src0RepeatStride, src1RepeatStride);
    vmul(product1, src01, src11, rows, 1U, src0BlockStride, src1BlockStride, 2U, src0RepeatStride, src1RepeatStride);
    vmuls(highA0, src00, static_cast<float>(kSplitter), rows, 1U, src0BlockStride, 2U, src0RepeatStride);
    vmuls(highA1, src01, static_cast<float>(kSplitter), rows, 1U, src0BlockStride, 2U, src0RepeatStride);
    vmuls(highB0, src10, static_cast<float>(kSplitter), rows, 1U, src1BlockStride, 2U, src1RepeatStride);
    vmuls(highB1, src11, static_cast<float>(kSplitter), rows, 1U, src1BlockStride, 2U, src1RepeatStride);
    pipe_barrier(PIPE_V);

    vsub(temporary0, highA0, src00, rows, 1U, 1U, src0BlockStride, 2U, 2U, src0RepeatStride);
    vsub(temporary1, highA1, src01, rows, 1U, 1U, src0BlockStride, 2U, 2U, src0RepeatStride);
    vsub(lowA0, highB0, src10, rows, 1U, 1U, src1BlockStride, 2U, 2U, src1RepeatStride);
    vsub(lowA1, highB1, src11, rows, 1U, 1U, src1BlockStride, 2U, 2U, src1RepeatStride);
    pipe_barrier(PIPE_V);

    vsub(highA0, highA0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(highA1, highA1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(highB0, highB0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(highB1, highB1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);

    vsub(lowA0, src00, highA0, rows, 1U, src0BlockStride, 1U, 2U, src0RepeatStride, 2U);
    vsub(lowA1, src01, highA1, rows, 1U, src0BlockStride, 1U, 2U, src0RepeatStride, 2U);
    vsub(lowB0, src10, highB0, rows, 1U, src1BlockStride, 1U, 2U, src1RepeatStride, 2U);
    vsub(lowB1, src11, highB1, rows, 1U, src1BlockStride, 1U, 2U, src1RepeatStride, 2U);
    pipe_barrier(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void ComputeOrderedFmaResidual(
    __ubuf__ float* product0, __ubuf__ float* product1, __ubuf__ float* highA0, __ubuf__ float* highA1,
    __ubuf__ float* lowA0, __ubuf__ float* lowA1, __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0,
    __ubuf__ float* lowB1, __ubuf__ float* error0, __ubuf__ float* error1, __ubuf__ float* temporary0,
    __ubuf__ float* temporary1, uint8_t rows)
{
    vmul(error0, highA0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vmul(error1, highA1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vsub(error0, error0, product0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(error1, error1, product1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vmul(temporary0, highA0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vmul(temporary1, highA1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vadd(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vadd(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vmul(temporary0, lowA0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vmul(temporary1, lowA1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vadd(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vadd(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vmul(temporary0, lowA0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vmul(temporary1, lowA1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vadd(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vadd(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void AddOrderedFmaProduct(
    __ubuf__ float* dst0, __ubuf__ float* dst1, __ubuf__ float* product0, __ubuf__ float* product1,
    __ubuf__ float* highA0, __ubuf__ float* highA1, __ubuf__ float* lowA0, __ubuf__ float* lowA1,
    __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0, __ubuf__ float* lowB1,
    __ubuf__ float* error0, __ubuf__ float* error1, __ubuf__ float* temporary0, __ubuf__ float* temporary1,
    uint8_t rows)
{
    vadd(highA0, dst0, product0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vadd(highA1, dst1, product1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vsub(lowA0, highA0, dst0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(lowA1, highA1, dst1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vsub(highB0, highA0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(highB1, highA1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vsub(lowB0, dst0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(lowB1, dst1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(temporary0, product0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vsub(temporary1, product1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vadd(lowB0, lowB0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vadd(lowB1, lowB1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vadd(temporary0, error0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vadd(temporary1, error1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
    vadd(dst0, highA0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vadd(dst1, highA1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void OrderedSoftwareFmaPair(
    __ubuf__ float* dst0, __ubuf__ float* dst1, __ubuf__ float* src00, __ubuf__ float* src01, __ubuf__ float* src10,
    __ubuf__ float* src11, __ubuf__ float* scratch, uint8_t rows, uint8_t src0BlockStride, uint8_t src1BlockStride,
    uint16_t src0RepeatStride, uint16_t src1RepeatStride)
{
    __ubuf__ float* product0 = scratch;
    __ubuf__ float* product1 = product0 + kMicroElements;
    __ubuf__ float* highA0 = product1 + kMicroElements;
    __ubuf__ float* highA1 = highA0 + kMicroElements;
    __ubuf__ float* lowA0 = highA1 + kMicroElements;
    __ubuf__ float* lowA1 = lowA0 + kMicroElements;
    __ubuf__ float* highB0 = lowA1 + kMicroElements;
    __ubuf__ float* highB1 = highB0 + kMicroElements;
    __ubuf__ float* lowB0 = highB1 + kMicroElements;
    __ubuf__ float* lowB1 = lowB0 + kMicroElements;
    __ubuf__ float* error0 = lowB1 + kMicroElements;
    __ubuf__ float* error1 = error0 + kMicroElements;
    __ubuf__ float* temporary0 = error1 + kMicroElements;
    __ubuf__ float* temporary1 = temporary0 + kMicroElements;
    SplitOrderedFmaOperands(
        src00, src01, src10, src11, product0, product1, highA0, highA1, lowA0, lowA1, highB0, highB1, lowB0, lowB1,
        temporary0, temporary1, rows, src0BlockStride, src1BlockStride, src0RepeatStride, src1RepeatStride);
    ComputeOrderedFmaResidual(
        product0, product1, highA0, highA1, lowA0, lowA1, highB0, highB1, lowB0, lowB1, error0, error1, temporary0,
        temporary1, rows);
    AddOrderedFmaProduct(
        dst0, dst1, product0, product1, highA0, highA1, lowA0, lowA1, highB0, highB1, lowB0, lowB1, error0, error1,
        temporary0, temporary1, rows);
}

struct OrderedAivBuffers {
    __ubuf__ float* raw;
    __ubuf__ float* xReal;
    __ubuf__ float* xImag;
    __ubuf__ float* yReal;
    __ubuf__ float* yImag;
    __ubuf__ float* transReal;
    __ubuf__ float* transImag;
    __ubuf__ float* broadcastReal;
    __ubuf__ float* broadcastImag;
    __ubuf__ float* sumReal;
    __ubuf__ float* sumImag;
    __ubuf__ float* outReal;
    __ubuf__ float* outImag;
    __ubuf__ float* output;
    __ubuf__ float* fma;
    __ubuf__ float* productReal;
    __ubuf__ float* productImag;
    __ubuf__ float* scaledReal;
    __ubuf__ float* scaledImag;
    __ubuf__ float* coefficient;
    __ubuf__ uint8_t* masks;
};

// Netlib first rounds the cross product, then fuses the other product.
// Scaling a complex value and multiplying two operands use different
// imaginary-product orders in the reference BLAS implementation.
__aicore__ inline __attribute__((always_inline)) void ScaleOrderedPlanes(
    __ubuf__ float* dstReal, __ubuf__ float* dstImag, __ubuf__ float* srcReal, __ubuf__ float* srcImag, float real,
    float imag, uint8_t rows, uint16_t stride, OrderedAivBuffers& buffers)
{
    vmuls(dstReal, srcImag, -imag, rows, 1U, 1U, 2U, stride);
    vmuls(dstImag, srcReal, imag, rows, 1U, 1U, 2U, stride);
    vector_dup(buffers.coefficient, real, 1U, 1U, 1U, 2U, 0U);
    pipe_barrier(PIPE_V);
    OrderedSoftwareFmaPair(
        dstReal, dstImag, srcReal, srcImag, buffers.coefficient, buffers.coefficient, buffers.fma, rows, 1U, 1U, stride,
        0U);
}

template <bool kPrefix>
__aicore__ inline __attribute__((always_inline)) void MultiplyOrderedPlanes(
    __ubuf__ float* srcReal, __ubuf__ float* srcImag, __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag,
    uint32_t first, uint8_t rows, OrderedAivBuffers& buffers)
{
    __ubuf__ float* real = buffers.productReal + first * kMicro;
    __ubuf__ float* imag = buffers.productImag + first * kMicro;
    vmul(real, srcImag, broadcastImag, rows, 1U, 1U, 0U, 2U, 0U, 1U);
    if constexpr (kPrefix) {
        vmul(imag, srcImag, broadcastReal, rows, 1U, 1U, 0U, 2U, 0U, 1U);
    } else {
        vmul(imag, srcReal, broadcastImag, rows, 1U, 1U, 0U, 2U, 0U, 1U);
    }
    pipe_barrier(PIPE_V);
    vmuls(real, real, -1.0f, rows, 1U, 1U, 2U, 2U);
    pipe_barrier(PIPE_V);
    OrderedSoftwareFmaPair(
        real, imag, srcReal, kPrefix ? srcReal : srcImag, broadcastReal, kPrefix ? broadcastImag : broadcastReal,
        buffers.fma, rows, 1U, 0U, 0U, 1U);
}

__aicore__ inline __attribute__((always_inline)) void AddOrderedProduct(
    __ubuf__ float* real, __ubuf__ float* imag, uint32_t first, uint8_t rows, OrderedAivBuffers& buffers)
{
    const uint32_t offset = first * kMicro;
    vadd(real + offset, real + offset, buffers.productReal + offset, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    vadd(imag + offset, imag + offset, buffers.productImag + offset, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    pipe_barrier(PIPE_V);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void LoadOrderedAivFeed(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t feed,
    uint32_t validCols, OrderedAivBuffers& buffers)
{
    set_vector_mask(0ULL, kFullFp32Mask);
    PrepareGridPair<kRight, kLower, false, false>(
        tiling, row, feed, feed, col, buffers.raw, buffers.raw + kComplexTileFloats);
    SyncGridPair<GridPhase::ISSUE, EVENT_ID2>();
    CopyGridPair<kRight, kLower, false, false>(
        a, b, tiling, row, feed, feed, col, buffers.raw, buffers.raw + kComplexTileFloats);
    SyncGridPair<GridPhase::COMMIT, EVENT_ID2>();
    SyncGridPair<GridPhase::CONSUME, EVENT_ID2>();
    DecodeGridPair<kRight, kLower, 0U, false>(
        row, feed, feed, col, buffers.raw, buffers.raw + kComplexTileFloats, buffers.xReal, buffers.xImag,
        buffers.yReal, buffers.yImag, buffers.transReal, buffers.transImag, buffers.masks + (kLower ? 32U : 0U),
        buffers.masks + 64U);
    pipe_barrier(PIPE_V);
    TransposePlane16(buffers.xReal, buffers.transReal);
    TransposePlane16(buffers.xImag, buffers.transImag);
    pipe_barrier(PIPE_V);
    BroadcastPlanes(buffers.transReal, buffers.transImag, buffers.broadcastReal, buffers.broadcastImag, 2U * kMicro);
    set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
}

template <bool kPrefix>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedLeftK(
    const ChemmAivTilingData& tiling, uint32_t k, uint32_t first, uint8_t rows, OrderedAivBuffers& buffers)
{
    __ubuf__ float* real = buffers.yReal + k * kMicro;
    __ubuf__ float* imag = buffers.yImag + k * kMicro;
    if constexpr (!kPrefix) {
        ScaleOrderedPlanes(
            buffers.scaledReal, buffers.scaledImag, real, imag, tiling.alphaReal, tiling.alphaImag, 1U, 0U, buffers);
        real = buffers.scaledReal;
        imag = buffers.scaledImag;
    }
    const uint32_t broadcast = k * kMicro * kFp32DataBlockElements + first * kFp32DataBlockElements;
    MultiplyOrderedPlanes<kPrefix>(
        real, imag, buffers.broadcastReal + broadcast, buffers.broadcastImag + broadcast, first, rows, buffers);
    AddOrderedProduct(
        kPrefix ? buffers.sumReal : buffers.outReal, kPrefix ? buffers.sumImag : buffers.outImag, first, rows, buffers);
}

template <bool kLower, bool kPrefix, bool kMiddle>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedLeftFeed(
    const ChemmAivTilingData& tiling, uint32_t feed, uint32_t validRows, OrderedAivBuffers& buffers)
{
    const uint32_t remaining = static_cast<uint32_t>(tiling.dimA) - feed;
    const uint32_t count = remaining < kMicro ? remaining : kMicro;
    for (uint32_t index = 0U; index < count; ++index) {
        const uint32_t k = kLower && !kPrefix ? count - 1U - index : index;
        uint32_t first = 0U;
        uint32_t rows = validRows;
        if constexpr (kMiddle) {
            if constexpr (kPrefix != kLower) {
                first = k + 1U;
                rows = first < validRows ? validRows - first : 0U;
            } else {
                rows = k < validRows ? k : validRows;
            }
        }
        if (rows != 0U) {
            AccumulateOrderedLeftK<kPrefix>(tiling, k, first, static_cast<uint8_t>(rows), buffers);
        }
    }
}

__aicore__ inline __attribute__((always_inline)) void InitializeOrderedLeftDiagonal(
    const ChemmAivTilingData& tiling, uint32_t validRows, OrderedAivBuffers& buffers)
{
    for (uint32_t k = 0U; k < validRows; ++k) {
        const uint32_t offset = k * kMicro;
        ScaleOrderedPlanes(
            buffers.scaledReal, buffers.scaledImag, buffers.yReal + offset, buffers.yImag + offset, tiling.alphaReal,
            tiling.alphaImag, 1U, 0U, buffers);
        __ubuf__ float* diagonal =
            buffers.broadcastReal + k * kMicro * kFp32DataBlockElements + k * kFp32DataBlockElements;
        vmul(buffers.productReal + offset, buffers.scaledReal, diagonal, 1U, 1U, 1U, 0U, 2U, 0U, 1U);
        vmul(buffers.productImag + offset, buffers.scaledImag, diagonal, 1U, 1U, 1U, 0U, 2U, 0U, 1U);
        pipe_barrier(PIPE_V);
        AddOrderedProduct(buffers.outReal, buffers.outImag, k, 1U, buffers);
        ScaleOrderedPlanes(
            buffers.productReal + offset, buffers.productImag + offset, buffers.sumReal + offset,
            buffers.sumImag + offset, tiling.alphaReal, tiling.alphaImag, 1U, 2U, buffers);
        AddOrderedProduct(buffers.outReal, buffers.outImag, k, 1U, buffers);
    }
}

template <bool kLower>
__aicore__ inline __attribute__((always_inline)) void ComputeOrderedLeft(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validRows,
    uint32_t validCols, OrderedAivBuffers& buffers)
{
    const uint32_t dimA = static_cast<uint32_t>(tiling.dimA);
    if constexpr (kLower) {
        LoadOrderedAivFeed<false, kLower>(a, b, tiling, row, col, row, validCols, buffers);
        AccumulateOrderedLeftFeed<kLower, true, true>(tiling, row, validRows, buffers);
    } else {
        for (uint32_t feed = 0U; feed < row; feed += kMicro) {
            LoadOrderedAivFeed<false, kLower>(a, b, tiling, row, col, feed, validCols, buffers);
            AccumulateOrderedLeftFeed<kLower, true, false>(tiling, feed, validRows, buffers);
        }
    }
    if constexpr (kLower) {
        for (uint32_t feed = row + kMicro; feed < dimA; feed += kMicro) {
            LoadOrderedAivFeed<false, kLower>(a, b, tiling, row, col, feed, validCols, buffers);
            AccumulateOrderedLeftFeed<kLower, true, false>(tiling, feed, validRows, buffers);
        }
    }
    LoadOrderedAivFeed<false, kLower>(a, b, tiling, row, col, row, validCols, buffers);
    if constexpr (!kLower) {
        AccumulateOrderedLeftFeed<kLower, true, true>(tiling, row, validRows, buffers);
    }
    InitializeOrderedLeftDiagonal(tiling, validRows, buffers);
    AccumulateOrderedLeftFeed<kLower, false, true>(tiling, row, validRows, buffers);
    if constexpr (kLower) {
        for (uint32_t end = row; end != 0U; end -= kMicro) {
            const uint32_t feed = end - kMicro;
            LoadOrderedAivFeed<false, kLower>(a, b, tiling, row, col, feed, validCols, buffers);
            AccumulateOrderedLeftFeed<kLower, false, false>(tiling, feed, validRows, buffers);
        }
    } else {
        for (uint32_t feed = row + kMicro; feed < dimA; feed += kMicro) {
            LoadOrderedAivFeed<false, kLower>(a, b, tiling, row, col, feed, validCols, buffers);
            AccumulateOrderedLeftFeed<kLower, false, false>(tiling, feed, validRows, buffers);
        }
    }
}

__aicore__ inline __attribute__((always_inline)) void InitializeOrderedRightDiagonal(
    const ChemmAivTilingData& tiling, uint8_t rows, uint32_t validCols, OrderedAivBuffers& buffers)
{
    float alphaReal = tiling.alphaReal;
    float alphaImag = tiling.alphaImag;
    for (uint32_t k = 0U; k < validCols; ++k) {
        set_vector_mask(0ULL, 1ULL << k);
        __ubuf__ float* diagonal = buffers.yReal + k * kMicro;
        vmuls(buffers.scaledReal, diagonal, alphaReal, 1U, 1U, 1U, 2U, 0U);
        vmuls(buffers.scaledImag, diagonal, alphaImag, 1U, 1U, 1U, 2U, 0U);
        pipe_barrier(PIPE_V);
        const uint32_t broadcast = k * kMicro * kFp32DataBlockElements;
        MultiplyOrderedPlanes<false>(
            buffers.scaledReal, buffers.scaledImag, buffers.broadcastReal + broadcast,
            buffers.broadcastImag + broadcast, 0U, rows, buffers);
        AddOrderedProduct(buffers.outReal, buffers.outImag, 0U, rows, buffers);
    }
    set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
}

__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedRightFeed(
    const ChemmAivTilingData& tiling, uint32_t feed, uint32_t col, uint8_t rows, uint32_t validCols,
    OrderedAivBuffers& buffers)
{
    const uint32_t remaining = static_cast<uint32_t>(tiling.dimA) - feed;
    const uint32_t count = remaining < kMicro ? remaining : kMicro;
    const uint64_t mask = (1ULL << validCols) - 1ULL;
    for (uint32_t k = 0U; k < count; ++k) {
        const uint64_t active = feed == col ? mask & ~(1ULL << k) : mask;
        if (active == 0U) {
            continue;
        }
        set_vector_mask(0ULL, active);
        const uint32_t offset = k * kMicro;
        ScaleOrderedPlanes(
            buffers.scaledReal, buffers.scaledImag, buffers.yReal + offset, buffers.yImag + offset, tiling.alphaReal,
            tiling.alphaImag, 1U, 0U, buffers);
        const uint32_t broadcast = k * kMicro * kFp32DataBlockElements;
        MultiplyOrderedPlanes<false>(
            buffers.scaledReal, buffers.scaledImag, buffers.broadcastReal + broadcast,
            buffers.broadcastImag + broadcast, 0U, rows, buffers);
        AddOrderedProduct(buffers.outReal, buffers.outImag, 0U, rows, buffers);
    }
}

template <bool kLower>
__aicore__ inline __attribute__((always_inline)) void ComputeOrderedRight(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint8_t rows,
    uint32_t validCols, OrderedAivBuffers& buffers)
{
    LoadOrderedAivFeed<true, kLower>(a, b, tiling, row, col, col, validCols, buffers);
    InitializeOrderedRightDiagonal(tiling, rows, validCols, buffers);
    for (uint32_t feed = 0U; feed < static_cast<uint32_t>(tiling.dimA); feed += kMicro) {
        LoadOrderedAivFeed<true, kLower>(a, b, tiling, row, col, feed, validCols, buffers);
        AccumulateOrderedRightFeed(tiling, feed, col, rows, validCols, buffers);
    }
}

__aicore__ inline __attribute__((always_inline)) void PrepareOrderedAivOutput(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validRows,
    uint32_t validCols, OrderedAivBuffers& buffers)
{
    set_mask_norm();
    set_vector_mask(0ULL, kFullFp32Mask);
    copy_gm_to_ubuf_align_b8(
        buffers.masks, reinterpret_cast<__gm__ uint8_t*>(g_packMasks), 0U, 1U, 96U, 0U, 0U, 0U, 0U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    if (tiling.betaReal != 0.0f || tiling.betaImag != 0.0f) {
        LoadOutputTile16<false>(
            c, row, col, static_cast<uint32_t>(tiling.ldc), static_cast<uint32_t>(tiling.m),
            static_cast<uint32_t>(tiling.n), buffers.output);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        CInterleavedToPlanes(buffers.output, buffers.xReal, buffers.xImag);
        pipe_barrier(PIPE_V);
        set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
        ScaleOrderedPlanes(
            buffers.outReal, buffers.outImag, buffers.xReal, buffers.xImag, tiling.betaReal, tiling.betaImag,
            static_cast<uint8_t>(validRows), 2U, buffers);
    } else {
        set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
        vector_dup(buffers.outReal, 0.0f, validRows, 1U, 1U, 2U, 0U);
        vector_dup(buffers.outImag, 0.0f, validRows, 1U, 1U, 2U, 0U);
    }
    vector_dup(buffers.sumReal, 0.0f, validRows, 1U, 1U, 2U, 0U);
    vector_dup(buffers.sumImag, 0.0f, validRows, 1U, 1U, 2U, 0U);
    pipe_barrier(PIPE_V);
}

__aicore__ inline OrderedAivBuffers OrderedBuffers(__ubuf__ float* scratch)
{
    constexpr uint32_t plane = kMicroElements;
    return {
        scratch,
        scratch + 4U * plane,
        scratch + 5U * plane,
        scratch + 6U * plane,
        scratch + 7U * plane,
        scratch + 8U * plane,
        scratch + 9U * plane,
        scratch + 10U * plane,
        scratch + 18U * plane,
        scratch + 26U * plane,
        scratch + 27U * plane,
        scratch + 28U * plane,
        scratch + 29U * plane,
        scratch + 30U * plane,
        scratch + 32U * plane,
        scratch + 46U * plane,
        scratch + 47U * plane,
        scratch + 48U * plane,
        scratch + 49U * plane,
        scratch + 50U * plane,
        reinterpret_cast<__ubuf__ uint8_t*>(scratch + 51U * plane)};
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void RunOrderedAiv(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t tile,
    __ubuf__ float* scratch)
{
    OrderedAivBuffers buffers = OrderedBuffers(scratch);
    const uint32_t tileCol = tile / tiling.tileRows;
    const uint32_t row = (tile - tileCol * tiling.tileRows) * kMicro;
    const uint32_t col = tileCol * kMicro;
    const uint32_t validRows =
        static_cast<uint32_t>(tiling.m) - row < kMicro ? static_cast<uint32_t>(tiling.m) - row : kMicro;
    const uint32_t validCols =
        static_cast<uint32_t>(tiling.n) - col < kMicro ? static_cast<uint32_t>(tiling.n) - col : kMicro;
    PrepareOrderedAivOutput(c, tiling, row, col, validRows, validCols, buffers);
    if (tiling.alphaReal != 0.0f || tiling.alphaImag != 0.0f) {
        if constexpr (kRight) {
            ComputeOrderedRight<kLower>(a, b, tiling, row, col, static_cast<uint8_t>(validRows), validCols, buffers);
        } else {
            ComputeOrderedLeft<kLower>(a, b, tiling, row, col, validRows, validCols, buffers);
        }
    }
    set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
    PlanesToCInterleaved(buffers.outReal, buffers.outImag, buffers.output, kComplexColumnFloats);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    StoreComplexTail<kComplexColumnFloats>(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(tiling.ldc)), buffers.output, validCols, validRows,
        static_cast<uint32_t>(tiling.ldc));
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}
} // namespace

__aicore__ inline void ComputeUnit(__gm__ float* a, __gm__ float* b, __gm__ float* c, __ubuf__ float* scratch)
{
    __ubuf__ float* aBlock = scratch;
    __ubuf__ float* value = scratch + kFp32DataBlockElements;
    __ubuf__ float* broadcast = scratch + 2U * kFp32DataBlockElements;

    set_vector_mask(0ULL, 3ULL);
    copy_gm_to_ubuf_align_b32(aBlock, a, 0U, 1U, 2U * sizeof(float), 0U, 0U, 0U, 0U);
    copy_gm_to_ubuf_align_b32(value, b, 0U, 1U, 2U * sizeof(float), 0U, 0U, 0U, 0U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // Only the first BRCB datablock is consumed; lanes loaded past A(0,0)
    // remain dummy and never enter the two-lane VMUL.
    vbrcb(reinterpret_cast<__ubuf__ uint32_t*>(broadcast), reinterpret_cast<__ubuf__ uint32_t*>(aBlock), 1U, 8U, 1U);
    // VMUL reads the BRCB result through UB and therefore requires this
    // producer-consumer handoff.
    pipe_barrier(PIPE_V);
    vmul(value, value, broadcast, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm_align_b32(c, value, 0U, 1U, 2U * sizeof(float), 0U, 0U, 0U, 0U);
    pipe_barrier(PIPE_MTE3);
}

template <bool kRight, bool kLower>
__aicore__ inline void RunAivGridMode(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* rawSlots,
    __ubuf__ float* leftPlanes, __ubuf__ float* leftTranspose, __ubuf__ float* accumulators, __ubuf__ float* broadcasts,
    __ubuf__ float* negativeVectorImag, __ubuf__ float* outputTile, __ubuf__ float* yCache, __ubuf__ uint8_t* masks)
{
    if (tiling.rowTile == kAivRowsPerVectorCore) {
        RunAivGrid<kRight, kLower, kAivRowsPerVectorCore>(
            a, b, c, tiling, rawSlots, leftPlanes, leftTranspose, accumulators, broadcasts, negativeVectorImag,
            outputTile, yCache, masks);
    } else {
        RunAivGrid<kRight, kLower, kMicro>(
            a, b, c, tiling, rawSlots, leftPlanes, leftTranspose, accumulators, broadcasts, negativeVectorImag,
            outputTile, yCache, masks);
    }
}

template <bool kRight, bool kLower>
__aicore__ inline void RunVector(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* scratch)
{
    if (tiling.mode == ChemmVectorMode::ORDERED) {
        const uint32_t count = tiling.tileRows * tiling.tileCols;
        for (uint64_t tile = static_cast<uint32_t>(get_block_idx()); tile < count; tile += tiling.blockDim) {
            RunOrderedAiv<kRight, kLower>(a, b, c, tiling, static_cast<uint32_t>(tile), scratch);
        }
        return;
    }
    __ubuf__ float* left = scratch + 4U * kComplexTileFloats;
    __ubuf__ float* right = left + 2U * kMicroElements;
    __ubuf__ float* transpose = right + 2U * kMicroElements;
    __ubuf__ float* sums = transpose + 2U * kMicroElements;
    __ubuf__ float* broadcasts = sums + kAivAccumulatorParts * kAivAccumulatorBanks * kMicroElements;
    __ubuf__ float* negative = broadcasts + 2U * kMicro * kMicro * kFp32DataBlockElements;
    __ubuf__ float* output = negative + kMicroElements;
    __ubuf__ float* cache = output + kComplexTileFloats;
    __ubuf__ uint8_t* masks = reinterpret_cast<__ubuf__ uint8_t*>(cache + 2U * kAivYCachePlaneElements);
    if (tiling.mode == ChemmVectorMode::LOCAL) {
        RunAivTiny<kRight, kLower>(a, b, c, tiling, scratch, left, right, sums, broadcasts, negative, masks);
        return;
    }
    RunAivGridMode<kRight, kLower>(
        a, b, c, tiling, scratch, left, transpose, sums, broadcasts, negative, output, cache, masks);
}

__global__ __aicore__ __vector__ void chemm_aiv_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, ChemmAivTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    set_atomic_none();
    set_mask_norm();
    set_vector_mask(0ULL, kFullFp32Mask);
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    constexpr uint32_t scratchElements =
        4U * kComplexTileFloats + (9U + kAivAccumulatorParts * kAivAccumulatorBanks) * kMicroElements +
        2U * kMicro * kMicro * kFp32DataBlockElements + 2U * kAivYCachePlaneElements + 24U;
    __ubuf__ float scratch[scratchElements] __attribute__((aligned(32)));
    if (tiling.mode == ChemmVectorMode::LOCAL && tiling.m == 1 && tiling.n == 1) {
        ComputeUnit(a, b, c, scratch);
        return;
    }
    const bool right = (tiling.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        RunVector<true, true>(a, b, c, tiling, scratch);
    } else if (right) {
        RunVector<true, false>(a, b, c, tiling, scratch);
    } else if (lower) {
        RunVector<false, true>(a, b, c, tiling, scratch);
    } else {
        RunVector<false, false>(a, b, c, tiling, scratch);
    }
}

void chemm_aiv_kernel_do(uint8_t* a, uint8_t* b, uint8_t* c, const ChemmAivTilingData& tiling, void* stream)
{
    chemm_aiv_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, tiling);
}
