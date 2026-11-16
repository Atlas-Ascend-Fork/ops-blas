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

#include "chemm_kernel_common.h"

namespace {

constexpr uint32_t kAivAccumulatorBanks = 16U;
constexpr uint32_t kAivAccumulatorParts = 2U;

// chemm (arch22) AIV (vector) compute path.

// Tiny and unit vector paths.
template <uint32_t kK, uint32_t kEnd, uint32_t kRowCapacity>
__aicore__ inline __attribute__((always_inline)) void BroadcastAivRange(
    __ubuf__ float* scalarReal, __ubuf__ float* scalarImag, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, uint32_t rowBase)
{
    constexpr uint32_t kRowOffset = kK * kMicro;
    constexpr uint32_t kBroadcastElements = kRowCapacity * kFp32DataBlockElements;
    __ubuf__ float* real = broadcastReal + kK * kBroadcastElements;
    __ubuf__ float* imag = broadcastImag + kK * kBroadcastElements;
    constexpr uint8_t kRepeats = static_cast<uint8_t>(kRowCapacity / kAivRowsPerBroadcastRepeat);
    // The arithmetic source block stride is zero, so one repeated FP32
    // datablock per row feeds both halves of the 16-column output vector.
    asc_brcb(
        reinterpret_cast<__ubuf__ uint32_t*>(real),
        reinterpret_cast<__ubuf__ uint32_t*>(scalarReal + kRowOffset + rowBase), 1U, 8U, kRepeats);
    asc_brcb(
        reinterpret_cast<__ubuf__ uint32_t*>(imag),
        reinterpret_cast<__ubuf__ uint32_t*>(scalarImag + kRowOffset + rowBase), 1U, 8U, kRepeats);
    if constexpr (kK + 1U < kEnd) {
        BroadcastAivRange<kK + 1U, kEnd, kRowCapacity>(scalarReal, scalarImag, broadcastReal, broadcastImag, rowBase);
    }
}

template <bool kInit, uint32_t kK, uint32_t kBanks, uint32_t kBroadcastElements>
__aicore__ inline __attribute__((always_inline)) void AccumulateAivMainK(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* accumulators, __ubuf__ float* broadcastReal,
    uint32_t rowBase, uint8_t rows)
{
    constexpr uint32_t kRowOffset = kK * kMicro;
    constexpr uint32_t kBank = kK % kBanks;
    constexpr bool kInitialize = kInit && kK < kBanks;
    constexpr uint32_t kBankStride = kMicroElements;
    constexpr uint32_t kPartStride = kBanks * kBankStride;
    __ubuf__ float* sumReal = accumulators + kBank * kBankStride + rowBase * kMicro;
    __ubuf__ float* sumImag = accumulators + kPartStride + kBank * kBankStride + rowBase * kMicro;
    __ubuf__ float* srcReal = vectorReal + kRowOffset;
    __ubuf__ float* srcImag = vectorImag + kRowOffset;
    __ubuf__ float* br = broadcastReal + kK * kBroadcastElements;
    if constexpr (kInitialize) {
        asc_mul(sumReal, srcReal, br, rows, 1U, 1U, 0U, 2U, 0U, 1U);
        asc_mul(sumImag, srcImag, br, rows, 1U, 1U, 0U, 2U, 0U, 1U);
    } else {
        asc_fma(sumReal, srcReal, br, rows, 1U, 1U, 0U, 2U, 0U, 1U);
        asc_fma(sumImag, srcImag, br, rows, 1U, 1U, 0U, 2U, 0U, 1U);
    }
}

template <bool kInit, uint32_t kK, uint32_t kEnd, uint32_t kBanks, uint32_t kBroadcastElements>
__aicore__ inline __attribute__((always_inline)) void AccumulateAivMainRange(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* accumulators, __ubuf__ float* broadcastReal,
    uint32_t rowBase, uint8_t rows)
{
    AccumulateAivMainK<kInit, kK, kBanks, kBroadcastElements>(
        vectorReal, vectorImag, accumulators, broadcastReal, rowBase, rows);
    if constexpr (kK + 1U < kEnd) {
        AccumulateAivMainRange<kInit, kK + 1U, kEnd, kBanks, kBroadcastElements>(
            vectorReal, vectorImag, accumulators, broadcastReal, rowBase, rows);
    }
}

template <uint32_t kK, uint32_t kBanks, uint32_t kBroadcastElements>
__aicore__ inline __attribute__((always_inline)) void AccumulateAivCrossK(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* accumulators, __ubuf__ float* broadcastImag,
    __ubuf__ float* negativeVectorImag, uint32_t rowBase, uint8_t rows)
{
    constexpr uint32_t kRowOffset = kK * kMicro;
    constexpr uint32_t kBank = kK % kBanks;
    constexpr uint32_t kPartStride = kBanks * kMicroElements;
    __ubuf__ float* sumReal = accumulators + kBank * kMicroElements + rowBase * kMicro;
    __ubuf__ float* sumImag = accumulators + kPartStride + kBank * kMicroElements + rowBase * kMicro;
    asc_fma(
        sumReal, negativeVectorImag + kRowOffset, broadcastImag + kK * kBroadcastElements, rows, 1U, 1U, 0U, 2U, 0U,
        1U);
    asc_fma(sumImag, vectorReal + kRowOffset, broadcastImag + kK * kBroadcastElements, rows, 1U, 1U, 0U, 2U, 0U, 1U);
}

template <uint32_t kK, uint32_t kEnd, uint32_t kBanks, uint32_t kBroadcastElements>
__aicore__ inline __attribute__((always_inline)) void AccumulateAivCrossRange(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* accumulators, __ubuf__ float* broadcastImag,
    __ubuf__ float* negativeVectorImag, uint32_t rowBase, uint8_t rows)
{
    AccumulateAivCrossK<kK, kBanks, kBroadcastElements>(
        vectorReal, vectorImag, accumulators, broadcastImag, negativeVectorImag, rowBase, rows);
    if constexpr (kK + 1U < kEnd) {
        AccumulateAivCrossRange<kK + 1U, kEnd, kBanks, kBroadcastElements>(
            vectorReal, vectorImag, accumulators, broadcastImag, negativeVectorImag, rowBase, rows);
    }
}

template <bool kInit, uint32_t kK, uint32_t kEnd, uint32_t kBanks, uint32_t kBroadcastElements>
__aicore__ inline __attribute__((always_inline)) void AccumulateAivRange(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* accumulators, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negativeVectorImag, uint32_t rowBase, uint8_t rows)
{
    AccumulateAivMainRange<kInit, kK, kEnd, kBanks, kBroadcastElements>(
        vectorReal, vectorImag, accumulators, broadcastReal, rowBase, rows);
    // Cross terms update the same complex accumulators through UB.
    asc_sync_pipe(PIPE_V);
    AccumulateAivCrossRange<kK, kEnd, kBanks, kBroadcastElements>(
        vectorReal, vectorImag, accumulators, broadcastImag, negativeVectorImag, rowBase, rows);
}

template <uint32_t kPart, uint32_t kBanks, uint32_t kStride, uint32_t kBank = 0U>
__aicore__ inline __attribute__((always_inline)) void ReduceAivBankLevel(
    __ubuf__ float* accumulators, uint32_t rowBase, uint8_t rows)
{
    constexpr uint32_t kPartStride = kBanks * kMicroElements;
    const uint32_t offset = kBank * kMicroElements + rowBase * kMicro;
    __ubuf__ float* dst = accumulators + kPart * kPartStride + offset;
    asc_add(dst, dst, dst + kStride * kMicroElements, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    if constexpr (kBank + 2U * kStride < kBanks) {
        ReduceAivBankLevel<kPart, kBanks, kStride, kBank + 2U * kStride>(accumulators, rowBase, rows);
    }
}

template <uint32_t kBanks>
__aicore__ inline __attribute__((always_inline)) void ReduceAivAccumulators(
    __ubuf__ float* accumulators, uint32_t rowBase, uint8_t rows, __ubuf__ float*& productReal,
    __ubuf__ float*& productImag)
{
    constexpr uint32_t kPartStride = kBanks * kMicroElements;
    static_assert(
        kBanks == 1U || kBanks == 2U || kBanks == 4U || kBanks == 8U || kBanks == 16U,
        "AIV bank tree requires a power of two");
    if constexpr (kBanks >= 2U) {
        ReduceAivBankLevel<0U, kBanks, 1U>(accumulators, rowBase, rows);
        ReduceAivBankLevel<1U, kBanks, 1U>(accumulators, rowBase, rows);
    }
    if constexpr (kBanks >= 4U) {
        asc_sync_pipe(PIPE_V);
        ReduceAivBankLevel<0U, kBanks, 2U>(accumulators, rowBase, rows);
        ReduceAivBankLevel<1U, kBanks, 2U>(accumulators, rowBase, rows);
    }
    if constexpr (kBanks >= 8U) {
        asc_sync_pipe(PIPE_V);
        ReduceAivBankLevel<0U, kBanks, 4U>(accumulators, rowBase, rows);
        ReduceAivBankLevel<1U, kBanks, 4U>(accumulators, rowBase, rows);
    }
    if constexpr (kBanks == 16U) {
        asc_sync_pipe(PIPE_V);
        ReduceAivBankLevel<0U, kBanks, 8U>(accumulators, rowBase, rows);
        ReduceAivBankLevel<1U, kBanks, 8U>(accumulators, rowBase, rows);
    }
    productReal = accumulators;
    productImag = accumulators + kPartStride;
}

template <bool kFull>
__aicore__ inline __attribute__((always_inline)) void LoadAivCPlanes(
    __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* raw, __ubuf__ float* real, __ubuf__ float* imag)
{
    LoadAivComplexTile<kFull, EVENT_ID5>(
        c, static_cast<uint32_t>(tiling.ldc), static_cast<uint32_t>(tiling.m), static_cast<uint32_t>(tiling.n), raw);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    CInterleavedToPlanes(raw, real, imag);
}

__aicore__ inline __attribute__((always_inline)) void ApplyAivCoefficients(
    const ChemmAivTilingData& tiling, __ubuf__ float* productReal, __ubuf__ float* productImag, __ubuf__ float* cReal,
    __ubuf__ float* cImag, __ubuf__ float* outReal, __ubuf__ float* outImag, uint32_t rowBase, uint8_t rows)
{
    const uint32_t offset = rowBase * kMicro;
    productReal += offset;
    productImag += offset;
    cReal += offset;
    cImag += offset;
    outReal += offset;
    outImag += offset;
    asc_mul_scalar(outReal, productReal, tiling.alphaReal, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(outImag, productImag, tiling.alphaReal, rows, 1U, 1U, 2U, 2U);
    // Each following pair reads the preceding pair's destinations through UB.
    asc_sync_pipe(PIPE_V);
    asc_axpy(outReal, productImag, -tiling.alphaImag, rows, 1U, 1U, 2U, 2U);
    asc_axpy(outImag, productReal, tiling.alphaImag, rows, 1U, 1U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_axpy(outReal, cReal, tiling.betaReal, rows, 1U, 1U, 2U, 2U);
    asc_axpy(outImag, cImag, tiling.betaReal, rows, 1U, 1U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_axpy(outReal, cImag, -tiling.betaImag, rows, 1U, 1U, 2U, 2U);
    asc_axpy(outImag, cReal, tiling.betaImag, rows, 1U, 1U, 2U, 2U);
}

__aicore__ inline void RunAivUnit(__gm__ float* a, __gm__ float* b, __gm__ float* c)
{
    __ubuf__ float aBlock[8U] __attribute__((aligned(32)));
    __ubuf__ float value[8U] __attribute__((aligned(32)));
    __ubuf__ float broadcast[64U] __attribute__((aligned(32)));

    asc_set_vector_mask(0ULL, 3ULL);
    asc_copy_gm2ub_align(aBlock, a, 2U * sizeof(float));
    asc_copy_gm2ub_align(value, b, 2U * sizeof(float));
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // Only the first BRCB datablock is consumed; lanes loaded past A(0,0)
    // remain dummy and never enter the two-lane VMUL.
    asc_brcb(reinterpret_cast<__ubuf__ uint32_t*>(broadcast), reinterpret_cast<__ubuf__ uint32_t*>(aBlock), 1U, 8U, 1U);
    // VMUL reads the BRCB result through UB and therefore requires this
    // producer-consumer handoff.
    asc_sync_pipe(PIPE_V);
    asc_mul(value, value, broadcast, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm_align(c, value, 2U * sizeof(float));
    asc_sync_pipe(PIPE_MTE3);
}

template <bool kRight, bool kLower, bool kFull>
__aicore__ inline __attribute__((always_inline)) void PrepareAivTinyOperands(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, __ubuf__ float* raw, __ubuf__ float* rowReal,
    __ubuf__ float* rowImag, __ubuf__ float* auxReal, __ubuf__ float* auxImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    LoadAivHermitianPlanes<kLower, kFull>(
        a, MaskGm(tiling), tiling, raw, rowReal, rowImag, auxReal, auxImag, triangleMask, diagonalMask);
    asc_sync_pipe(PIPE_V);
    if constexpr (!kRight) {
        TransposePlane16(rowReal, auxReal);
        TransposePlane16(rowImag, auxImag);
    }
    LoadAivBPlanes<kRight, kFull>(b, tiling, raw, rowReal, rowImag, auxReal, auxImag);
    asc_sync_pipe(PIPE_V);
}

template <bool kFull, bool kSimpleCoeff, uint32_t kRowCapacity>
__aicore__ inline __attribute__((always_inline)) void StoreAivTinyResult(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t rowBase, uint32_t cols, uint8_t rows,
    __ubuf__ float* productReal, __ubuf__ float* productImag, __ubuf__ float* raw, __ubuf__ float* rowReal,
    __ubuf__ float* rowImag, __ubuf__ float* auxReal, __ubuf__ float* auxImag)
{
    __ubuf__ float* finalReal = productReal;
    __ubuf__ float* finalImag = productImag;
    if constexpr (!kSimpleCoeff) {
        LoadAivCPlanes<kFull>(c, tiling, raw, rowReal, rowImag);
        asc_sync_pipe(PIPE_V);
        ApplyAivCoefficients(tiling, productReal, productImag, rowReal, rowImag, auxReal, auxImag, rowBase, rows);
        finalReal = auxReal;
        finalImag = auxImag;
    }
    if constexpr (kRowCapacity == kAivRowsPerVectorCore) {
        PlanesToCInterleavedRows8(
            finalReal + rowBase * kMicro, finalImag + rowBase * kMicro, raw, kComplexColumnFloats);
    } else {
        PlanesToCInterleavedStride(finalReal, finalImag, raw, kComplexColumnFloats);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    StoreComplexTail<kComplexColumnFloats>(c + 2U * rowBase, raw, cols, rows, static_cast<uint32_t>(tiling.ldc));
    asc_sync_pipe(PIPE_MTE3);
}

template <bool kRight, bool kLower, bool kFull, uint32_t kAivK, uint32_t kRowCapacity, bool kSimpleCoeff>
__aicore__ inline void RunAivTiny(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* raw,
    __ubuf__ float* rowPlanes, __ubuf__ float* auxPlanes, __ubuf__ float* accumulators, __ubuf__ float* broadcasts,
    __ubuf__ float* negativeVectorImag, __ubuf__ uint8_t* masks)
{
    constexpr uint32_t kBanks = kAivK < kAivAccumulatorBanks ? kAivK : kAivAccumulatorBanks;
    constexpr uint32_t kBroadcastElements = kRowCapacity * kFp32DataBlockElements;

    __ubuf__ float* rowReal = rowPlanes;
    __ubuf__ float* rowImag = rowPlanes + kMicroElements;
    __ubuf__ float* auxReal = auxPlanes;
    __ubuf__ float* auxImag = auxPlanes + kMicroElements;
    __ubuf__ float* broadcastReal = broadcasts;
    __ubuf__ float* broadcastImag = broadcasts + kAivK * kBroadcastElements;
    __ubuf__ uint8_t* triangleMask = masks + (kLower ? 32U : 0U);
    __ubuf__ uint8_t* diagonalMask = masks + 64U;
    PrepareAivTinyOperands<kRight, kLower, kFull>(
        a, b, tiling, raw, rowReal, rowImag, auxReal, auxImag, triangleMask, diagonalMask);
    __ubuf__ float* scalarReal = auxReal;
    __ubuf__ float* scalarImag = auxImag;
    __ubuf__ float* vectorReal = rowReal;
    __ubuf__ float* vectorImag = rowImag;
    static_assert(
        kRowCapacity == kAivRowsPerVectorCore || kRowCapacity == kMicro,
        "pure AIV rows must match an arch22 vector batch");
    // BRCB consumes 32-byte source datablocks, so multi-core row bases stay
    // on eight-FP32 boundaries; finer row splits would require repacking.
    const uint32_t rowBase = static_cast<uint32_t>(AscendC::GetBlockIdx()) * kRowCapacity;
    const uint8_t rows = static_cast<uint8_t>(
        static_cast<uint32_t>(tiling.m) - rowBase < kRowCapacity ? static_cast<uint32_t>(tiling.m) - rowBase :
                                                                   kRowCapacity);

    const uint32_t cols = static_cast<uint32_t>(tiling.n);
    __ubuf__ float* productReal;
    __ubuf__ float* productImag;
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    BroadcastAivRange<0U, kAivK, kRowCapacity>(scalarReal, scalarImag, broadcastReal, broadcastImag, rowBase);
    asc_sync_pipe(PIPE_V);
    constexpr uint8_t kNegativeRepeats = static_cast<uint8_t>((kAivK * kMicro + 63U) / 64U);
    asc_mul_scalar(negativeVectorImag, vectorImag, -1.0f, kNegativeRepeats, 1U, 1U, 8U, 8U);
    asc_set_vector_mask(0ULL, (1ULL << cols) - 1ULL);
    AccumulateAivRange<true, 0U, kAivK, kBanks, kBroadcastElements>(
        vectorReal, vectorImag, accumulators, broadcastReal, broadcastImag, negativeVectorImag, rowBase, rows);
    // Reduction reads the accumulator banks produced by vmul/vmla.
    asc_sync_pipe(PIPE_V);
    ReduceAivAccumulators<kBanks>(accumulators, rowBase, rows, productReal, productImag);
    asc_sync_pipe(PIPE_V);
    StoreAivTinyResult<kFull, kSimpleCoeff, kRowCapacity>(
        c, tiling, rowBase, cols, rows, productReal, productImag, raw, rowReal, rowImag, auxReal, auxImag);
}

// chemm (arch22) AIV grid-tiled variant.

// Grid vector path.
__aicore__ inline __attribute__((always_inline)) void LoadLeftAivGridCache(
    __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t col, uint32_t validCols, uint32_t kFeeds,
    __ubuf__ float* broadcasts, __ubuf__ float* yCacheReal, __ubuf__ float* yCacheImag)
{
    const uint32_t stageRows = kFeeds * kMicro;
    const uint32_t validFloats = 2U * static_cast<uint32_t>(tiling.dimA);
    const uint32_t alignedFloats = (validFloats + 7U) & ~7U;
    const uint32_t rawStride = 2U * stageRows;
    asc_copy_gm2ub_align(
        broadcasts, b + 2U * ElementIndex(0U, col, static_cast<uint32_t>(tiling.ldb)), static_cast<uint16_t>(validCols),
        validFloats * sizeof(float), 0U, static_cast<uint8_t>(alignedFloats - validFloats),
        (2U * static_cast<uint32_t>(tiling.ldb) - validFloats) * sizeof(float), (rawStride - alignedFloats) / 8U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    const uint32_t tailFloats = rawStride - validFloats;
    if (tailFloats != 0U) {
        const uint32_t prefix = validFloats - (validFloats & ~7U);
        asc_set_mask_norm();
        asc_set_vector_mask(0ULL, ((1ULL << tailFloats) - 1ULL) << prefix);
        asc_duplicate(
            broadcasts + (validFloats & ~7U), 0.0f, static_cast<uint8_t>(validCols), 1U,
            static_cast<uint8_t>(rawStride / 8U));
        asc_set_vector_mask(0ULL, kFullFp32Mask);
    }
    CInterleavedToPlanesStride(
        broadcasts, yCacheReal, yCacheImag, kMicro, rawStride,
        static_cast<uint8_t>(stageRows / kAivRowsPerTransposeRepeat));
}

template <bool kLower>
__aicore__ inline __attribute__((always_inline)) void LoadRightAivGridCache(
    __gm__ float* a, const ChemmAivTilingData& tiling, uint32_t col, uint32_t kFeeds, __ubuf__ float* raw,
    __ubuf__ float* yCacheReal, __ubuf__ float* yCacheImag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
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
        DecodeHermitianPlanes16<kLower, 0U, false>(
            tiling, kBase, col, raw, yCacheReal + offset, yCacheImag + offset, transReal, transImag, triangleMask,
            diagonalMask);
    }
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void LoadAivGridCache(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t col, uint32_t validCols,
    uint32_t kFeeds, __ubuf__ float* raw, __ubuf__ float* broadcasts, __ubuf__ float* yCacheReal,
    __ubuf__ float* yCacheImag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask, __ubuf__ uint8_t* masks)
{
    asc_copy_gm2ub_align(masks, MaskGm(tiling), 96U);
    if constexpr (kRight) {
        LoadRightAivGridCache<kLower>(
            a, tiling, col, kFeeds, raw, yCacheReal, yCacheImag, transReal, transImag, triangleMask, diagonalMask);
    } else {
        LoadLeftAivGridCache(b, tiling, col, validCols, kFeeds, broadcasts, yCacheReal, yCacheImag);
    }
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void DecodeAivGridBank(
    const ChemmAivTilingData& tiling, uint32_t bank, uint32_t alignedRow, uint32_t kBase, uint32_t col,
    __ubuf__ float* raw, __ubuf__ float* leftReal, __ubuf__ float* leftImag, __ubuf__ float* rightReal,
    __ubuf__ float* rightImag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    if (bank == 0U) {
        WaitGridPair<0U>();
        DecodeGridPair<kRight, kLower, 0U, true>(
            tiling, alignedRow, kBase, kBase, col, raw, raw + kComplexTileFloats, leftReal, leftImag, rightReal,
            rightImag, transReal, transImag, triangleMask, diagonalMask);
    } else {
        WaitGridPair<1U>();
        DecodeGridPair<kRight, kLower, 1U, true>(
            tiling, alignedRow, kBase, kBase, col, raw, raw + kComplexTileFloats, leftReal, leftImag, rightReal,
            rightImag, transReal, transImag, triangleMask, diagonalMask);
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
        SubmitGridPair<0U>();
        CopyGridPair<kRight, kLower, false, true>(
            a, b, tiling, alignedRow, nextK, nextK, col, nextRaw, nextRaw + kComplexTileFloats);
        CommitGridPair<0U>();
    } else {
        SubmitGridPair<1U>();
        CopyGridPair<kRight, kLower, false, true>(
            a, b, tiling, alignedRow, nextK, nextK, col, nextRaw, nextRaw + kComplexTileFloats);
        CommitGridPair<1U>();
    }
}

template <bool kRight, bool kLower, uint32_t kRows>
__aicore__ inline __attribute__((always_inline)) void AccumulateAivGridFeed(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t alignedRow, uint32_t rowOffset,
    uint32_t col, uint32_t validRows, uint32_t validCols, uint32_t feed, uint32_t kFeeds, __ubuf__ float* rawSlots,
    __ubuf__ float* leftReal, __ubuf__ float* leftImag, __ubuf__ float* rightReal, __ubuf__ float* rightImag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ float* accumulators, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negativeImag, __ubuf__ float* yCacheReal, __ubuf__ float* yCacheImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    constexpr uint32_t kBroadcastElements = kRows * kFp32DataBlockElements;
    asc_set_mask_norm();
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    const uint32_t kBase = feed * kMicro;
    const uint32_t bank = feed & 1U;
    __ubuf__ float* raw = rawSlots + bank * 2U * kComplexTileFloats;
    DecodeAivGridBank<kRight, kLower>(
        tiling, bank, alignedRow, kBase, col, raw, leftReal, leftImag, rightReal, rightImag, transReal, transImag,
        triangleMask, diagonalMask);
    if (feed + 1U < kFeeds) {
        const uint32_t nextBank = bank ^ 1U;
        SubmitNextAivGridBank<kRight, kLower>(
            a, b, tiling, nextBank, alignedRow, kBase + kMicro, col, rawSlots + nextBank * 2U * kComplexTileFloats);
    }
    asc_sync_pipe(PIPE_V);
    TransposePlane16(leftReal, transReal);
    TransposePlane16(leftImag, transImag);
    asc_sync_pipe(PIPE_V);
    BroadcastAivRange<0U, kMicro, kRows>(transReal, transImag, broadcastReal, broadcastImag, rowOffset);
    asc_sync_pipe(PIPE_V);
    __ubuf__ float* vectorReal = yCacheReal + kBase * kMicro;
    __ubuf__ float* vectorImag = yCacheImag + kBase * kMicro;
    asc_mul_scalar(negativeImag, vectorImag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
    asc_set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
    if (feed == 0U) {
        AccumulateAivRange<true, 0U, kMicro, kAivAccumulatorBanks, kBroadcastElements>(
            vectorReal, vectorImag, accumulators, broadcastReal, broadcastImag, negativeImag, 0U,
            static_cast<uint8_t>(validRows));
    } else {
        AccumulateAivRange<false, 0U, kMicro, kAivAccumulatorBanks, kBroadcastElements>(
            vectorReal, vectorImag, accumulators, broadcastReal, broadcastImag, negativeImag, 0U,
            static_cast<uint8_t>(validRows));
    }
    asc_sync_pipe(PIPE_V);
}

template <bool kRight, bool kLower, uint32_t kRows>
__aicore__ inline __attribute__((always_inline)) void AccumulateAivGridFeeds(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t alignedRow, uint32_t rowOffset,
    uint32_t col, uint32_t validRows, uint32_t validCols, uint32_t kFeeds, __ubuf__ float* rawSlots,
    __ubuf__ float* leftReal, __ubuf__ float* leftImag, __ubuf__ float* rightReal, __ubuf__ float* rightImag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ float* accumulators, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negativeImag, __ubuf__ float* yCacheReal, __ubuf__ float* yCacheImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    PrepareGridPair<kRight, kLower, false, true>(
        tiling, alignedRow, 0U, 0U, col, rawSlots, rawSlots + kComplexTileFloats);
    SubmitGridPair<0U>();
    CopyGridPair<kRight, kLower, false, true>(
        a, b, tiling, alignedRow, 0U, 0U, col, rawSlots, rawSlots + kComplexTileFloats);
    CommitGridPair<0U>();
    for (uint32_t feed = 0U; feed < kFeeds; ++feed) {
        AccumulateAivGridFeed<kRight, kLower, kRows>(
            a, b, tiling, alignedRow, rowOffset, col, validRows, validCols, feed, kFeeds, rawSlots, leftReal, leftImag,
            rightReal, rightImag, transReal, transImag, accumulators, broadcastReal, broadcastImag, negativeImag,
            yCacheReal, yCacheImag, triangleMask, diagonalMask);
    }
}

template <bool kSimpleCoeff>
__aicore__ inline __attribute__((always_inline)) void StoreAivGridResult(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validRows,
    uint32_t validCols, __ubuf__ float* accumulators, __ubuf__ float* outputTile, __ubuf__ float* rightReal,
    __ubuf__ float* rightImag, __ubuf__ float* leftReal, __ubuf__ float* leftImag)
{
    __ubuf__ float* productReal;
    __ubuf__ float* productImag;
    ReduceAivAccumulators<kAivAccumulatorBanks>(
        accumulators, 0U, static_cast<uint8_t>(validRows), productReal, productImag);
    asc_sync_pipe(PIPE_V);
    __ubuf__ float* finalReal = productReal;
    __ubuf__ float* finalImag = productImag;
    if constexpr (!kSimpleCoeff) {
        LoadOutputTile16<false>(
            c, row, col, static_cast<uint32_t>(tiling.ldc), static_cast<uint32_t>(tiling.m),
            static_cast<uint32_t>(tiling.n), outputTile);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        CInterleavedToPlanes(outputTile, rightReal, rightImag);
        ApplyAivCoefficients(
            tiling, productReal, productImag, rightReal, rightImag, leftReal, leftImag, 0U,
            static_cast<uint8_t>(validRows));
        asc_sync_pipe(PIPE_V);
        finalReal = leftReal;
        finalImag = leftImag;
    }
    PlanesToCInterleavedStride(finalReal, finalImag, outputTile, kComplexColumnFloats);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    StoreComplexTail<kComplexColumnFloats>(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(tiling.ldc)), outputTile, validCols, validRows,
        static_cast<uint32_t>(tiling.ldc));
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

template <bool kRight, bool kLower, bool kSimpleCoeff, uint32_t kRows = kMicro>
__aicore__ inline void RunAivGrid(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* rawSlots,
    __ubuf__ float* leftPlanes, __ubuf__ float* rightPlanes, __ubuf__ float* leftTranspose,
    __ubuf__ float* accumulators, __ubuf__ float* broadcasts, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* outputTile, __ubuf__ float* yCache, __ubuf__ uint8_t* masks)
{
    static_assert(
        kRows == kMicro || kRows == kAivRowsPerVectorCore, "AIV Grid rows must match a vector broadcast batch");
    constexpr uint32_t kBroadcastElements = kRows * kFp32DataBlockElements;

    __ubuf__ float* leftReal = leftPlanes;
    __ubuf__ float* leftImag = leftPlanes + kMicroElements;
    __ubuf__ float* rightReal = rightPlanes;
    __ubuf__ float* rightImag = rightPlanes + kMicroElements;
    __ubuf__ float* leftTransReal = leftTranspose;
    __ubuf__ float* leftTransImag = leftTranspose + kMicroElements;
    __ubuf__ float* broadcastReal = broadcasts;
    __ubuf__ float* broadcastImag = broadcasts + kMicro * kBroadcastElements;
    __ubuf__ float* yCacheReal = yCache;
    __ubuf__ float* yCacheImag = yCache + kAivYCachePlaneElements;
    __ubuf__ uint8_t* triangleMask = masks + (kLower ? 32U : 0U);
    __ubuf__ uint8_t* diagonalMask = masks + 64U;

    const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
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
        a, b, tiling, alignedRow, rowOffset, col, validRows, validCols, kFeeds, rawSlots, leftReal, leftImag, rightReal,
        rightImag, leftTransReal, leftTransImag, accumulators, broadcastReal, broadcastImag, negativeVectorImag,
        yCacheReal, yCacheImag, triangleMask, diagonalMask);
    StoreAivGridResult<kSimpleCoeff>(
        c, tiling, row, col, validRows, validCols, accumulators, outputTile, rightReal, rightImag, leftReal, leftImag);
}

} // namespace
