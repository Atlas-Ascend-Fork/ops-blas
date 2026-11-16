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

// chemm (arch22) output epilogue: alpha/beta scaling and store-back.

// Vector result combination and stores.
__aicore__ inline __attribute__((always_inline)) void AccumulateBeta(
    bool unitBeta, __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* oldReal, __ubuf__ float* oldImag,
    __ubuf__ float* tmp0, __ubuf__ float* tmp1, __ubuf__ float* betaReal, __ubuf__ float* betaImag)
{
    if (unitBeta) {
        asc_add(outReal, outReal, oldReal, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        asc_add(outImag, outImag, oldImag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    } else {
        asc_mul(tmp0, oldReal, betaReal, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        asc_mul(tmp1, oldImag, betaImag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        asc_sync_pipe(PIPE_V);
        asc_sub(tmp0, tmp0, tmp1, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        asc_sync_pipe(PIPE_V);
        asc_add(outReal, outReal, tmp0, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        asc_mul(tmp0, oldImag, betaReal, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        asc_mul(tmp1, oldReal, betaImag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        asc_sync_pipe(PIPE_V);
        asc_add(tmp0, tmp0, tmp1, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        asc_sync_pipe(PIPE_V);
        asc_add(outImag, outImag, tmp0, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    }
    asc_sync_pipe(PIPE_V);
}
template <bool kFull>
__aicore__ inline __attribute__((always_inline)) bool MicroTileExtent(
    const ChemmMixedTilingData& tiling, uint32_t rowBase, uint32_t colBase, uint32_t& validRows, uint32_t& validCols)
{
    validRows = kMicro;
    validCols = kMicro;
    if constexpr (!kFull) {
        validRows = rowBase < static_cast<uint32_t>(tiling.m) ? (static_cast<uint32_t>(tiling.m) - rowBase < kMicro ?
                                                                     static_cast<uint32_t>(tiling.m) - rowBase :
                                                                     kMicro) :
                                                                0U;
        validCols = colBase < static_cast<uint32_t>(tiling.n) ? (static_cast<uint32_t>(tiling.n) - colBase < kMicro ?
                                                                     static_cast<uint32_t>(tiling.n) - colBase :
                                                                     kMicro) :
                                                                0U;
    }
    return validRows != 0U && validCols != 0U;
}

template <bool kFull>
__aicore__ inline __attribute__((always_inline)) bool OutputHalfExtent(
    const ChemmMixedTilingData& tiling, uint32_t tileRow, uint32_t tileCol, uint32_t& validRows, uint32_t& validCols)
{
    validRows = CHEMM_TILE_M;
    validCols = CHEMM_TILE_N / 2U;
    if constexpr (!kFull) {
        validRows =
            tileRow < static_cast<uint32_t>(tiling.m) ?
                (static_cast<uint32_t>(tiling.m) - tileRow < CHEMM_TILE_M ? static_cast<uint32_t>(tiling.m) - tileRow :
                                                                            CHEMM_TILE_M) :
                0U;
        validCols = tileCol < static_cast<uint32_t>(tiling.n) ?
                        (static_cast<uint32_t>(tiling.n) - tileCol < CHEMM_TILE_N / 2U ?
                             static_cast<uint32_t>(tiling.n) - tileCol :
                             CHEMM_TILE_N / 2U) :
                        0U;
    }
    return validRows != 0U && validCols != 0U;
}

template <bool kFull, bool kSimpleCoeff>
__aicore__ inline __attribute__((always_inline)) void CombineTile16(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t rowBase, uint32_t colBase, __ubuf__ float* product,
    __ubuf__ float* stage, __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* oldReal,
    __ubuf__ float* oldImag, __ubuf__ float* tmp0, __ubuf__ float* tmp1, __ubuf__ float* complexTile,
    __ubuf__ float* betaReal, __ubuf__ float* betaImag)
{
    uint32_t validRows;
    uint32_t validCols;
    if (!MicroTileExtent<kFull>(tiling, rowBase, colBase, validRows, validCols)) {
        return;
    }
    const uint32_t productCol = colBase & ((CHEMM_TILE_N / 2U) - 1U);
    const uint32_t productRow = rowBase & (CHEMM_TILE_M - 1U);
    __ubuf__ float* productBase = product + productCol * CHEMM_EXP_N + 2U * productRow;
    CInterleavedToPlanesStride(productBase, outReal, outImag, kMicro, CHEMM_EXP_N);
    asc_sync_pipe(PIPE_V);

    if constexpr (!kSimpleCoeff) {
        const bool zeroBeta = tiling.betaReal == 0.0f && tiling.betaImag == 0.0f;
        const bool unitBeta = tiling.betaReal == 1.0f && tiling.betaImag == 0.0f;
        if (!zeroBeta) {
            LoadOutputTile16<kFull>(
                c, rowBase, colBase, static_cast<uint32_t>(tiling.ldc), static_cast<uint32_t>(tiling.m),
                static_cast<uint32_t>(tiling.n), complexTile);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            CInterleavedToPlanes(complexTile, oldReal, oldImag);
            asc_sync_pipe(PIPE_V);
            AccumulateBeta(unitBeta, outReal, outImag, oldReal, oldImag, tmp0, tmp1, betaReal, betaImag);
        }
    }

    const uint32_t localCol = colBase & ((CHEMM_TILE_N / 2U) - 1U);
    const uint32_t localRow = rowBase & (CHEMM_TILE_M - 1U);
    PlanesToCInterleavedStride(
        outReal, outImag, stage + localCol * (CHEMM_TILE_M * 2U) + localRow * 2U, CHEMM_TILE_M * 2U);
    // Finish the transform before the next micro-tile reuses outReal/outImag.
    asc_sync_pipe(PIPE_V);
}

template <bool kFull>
__aicore__ inline void StoreOutputHalf(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t tileRow, uint32_t tileCol, uint32_t validRows,
    uint32_t validCols, __ubuf__ float* stage)
{
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    const uint64_t offset = 2U * ElementIndex(tileRow, tileCol, static_cast<uint32_t>(tiling.ldc));
    if constexpr (kFull) {
        constexpr uint16_t blocks = CHEMM_TILE_M / 4U;
        asc_copy_ub2gm(
            c + offset, stage, CHEMM_TILE_N / 2U, blocks, 0U,
            static_cast<uint16_t>(static_cast<uint32_t>(tiling.ldc) / 4U - blocks));
    } else if (
        (validRows & 3U) == 0U && (static_cast<uint32_t>(tiling.ldc) & 3U) == 0U &&
        static_cast<uint32_t>(tiling.ldc) / 4U >= validRows / 4U &&
        static_cast<uint32_t>(tiling.ldc) / 4U - validRows / 4U <= 65535U) {
        const uint16_t blocks = static_cast<uint16_t>(validRows / 4U);
        asc_copy_ub2gm(
            c + offset, stage, static_cast<uint16_t>(validCols), blocks, static_cast<uint16_t>(16U - blocks),
            static_cast<uint16_t>(static_cast<uint32_t>(tiling.ldc) / 4U - blocks));
    } else {
        StoreComplexTail<2U * CHEMM_TILE_M>(c + offset, stage, validCols, validRows, static_cast<uint32_t>(tiling.ldc));
    }
    // Retire the store before V overwrites stage for the next output tile.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
}

template <uint32_t kOutput, bool kFull, bool kSimpleCoeff>
__aicore__ inline bool ConsumeHalf(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t tile, bool stagePending,
    __ubuf__ float* product, __ubuf__ float* stage, __ubuf__ float* outReal, __ubuf__ float* outImag,
    __ubuf__ float* oldReal, __ubuf__ float* oldImag, __ubuf__ float* tmp0, __ubuf__ float* tmp1,
    __ubuf__ float* complexTile, __ubuf__ float* betaReal, __ubuf__ float* betaImag)
{
    const uint32_t tileRow = TileRow(tiling, tile) * CHEMM_TILE_M;
    const uint32_t tileCol = TileCol(tiling, tile) * CHEMM_WORK_N + kOutput * CHEMM_TILE_N + half * (CHEMM_TILE_N / 2U);
    uint32_t validRows;
    uint32_t validCols;
    if (!OutputHalfExtent<kFull>(tiling, tileRow, tileCol, validRows, validCols)) {
        return stagePending;
    }
    // Only a real MTE3 store produces the stage-reuse event. Empty boundary
    // halves carry the pending state forward instead of waiting for an event
    // they never emitted.
    if (stagePending) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    for (uint32_t col = 0U; col < CHEMM_TILE_N / 2U; col += kMicro) {
        for (uint32_t row = 0U; row < CHEMM_TILE_M; row += kMicro) {
            CombineTile16<kFull, kSimpleCoeff>(
                c, tiling, tileRow + row, tileCol + col, product, stage, outReal, outImag, oldReal, oldImag, tmp0, tmp1,
                complexTile, betaReal, betaImag);
        }
    }
    StoreOutputHalf<kFull>(c, tiling, tileRow, tileCol, validRows, validCols, stage);
    return true;
}

template <bool kFull, bool kSimpleCoeff>
__aicore__ inline bool ConsumePair(
    __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t half,
    uint32_t ordinal, uint32_t tileCount, uint32_t tile, bool stagePending, __ubuf__ float* product,
    __ubuf__ float* stage, __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* oldReal,
    __ubuf__ float* oldImag, __ubuf__ float* tmp0, __ubuf__ float* tmp1, __ubuf__ float* complexTile,
    __ubuf__ float* betaReal, __ubuf__ float* betaImag)
{
    const uint32_t group = ordinal & 1U;
    CHEMM_WAIT(kProductReady + group);
    if (ordinal != 0U) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    }
    __gm__ float* source =
        ProductSlot(workspace, tiling, core, group * CHEMM_PRODUCTS_PER_GROUP) + half * kHalfProductElements;
    asc_copy_gm2ub(product, source, kHalfProductElements * sizeof(float));
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    stagePending = ConsumeHalf<0U, kFull, kSimpleCoeff>(
        c, tiling, half, tile, stagePending, product, stage, outReal, outImag, oldReal, oldImag, tmp0, tmp1,
        complexTile, betaReal, betaImag);

    // Output 1 reuses the same 32 KiB product UB after output 0 retires.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    source = ProductSlot(workspace, tiling, core, group * CHEMM_PRODUCTS_PER_GROUP + 1U) + half * kHalfProductElements;
    asc_copy_gm2ub(product, source, kHalfProductElements * sizeof(float));
    if (ordinal + CHEMM_PRODUCT_GROUPS < tileCount) {
        CHEMM_NOTIFY(PIPE_MTE2, kProductFree + group);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    stagePending = ConsumeHalf<1U, kFull, kSimpleCoeff>(
        c, tiling, half, tile, stagePending, product, stage, outReal, outImag, oldReal, oldImag, tmp0, tmp1,
        complexTile, betaReal, betaImag);
    if (ordinal + 1U < tileCount) {
        // Keep exactly one V->MTE2 ownership token for the next pair.
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    }
    return stagePending;
}

template <bool kFull>
__aicore__ inline __attribute__((always_inline)) void CombinePlanarTile16(
    const ChemmMixedTilingData& tiling, uint32_t rowBase, uint32_t colBase, uint32_t oldCColBase,
    __ubuf__ float* product, __ubuf__ float* stage, __ubuf__ float* oldCStage, __ubuf__ float* outReal,
    __ubuf__ float* outImag, __ubuf__ float* oldReal, __ubuf__ float* oldImag, __ubuf__ float* tmp0,
    __ubuf__ float* tmp1, __ubuf__ float* betaReal, __ubuf__ float* betaImag)
{
    uint32_t validRows;
    uint32_t validCols;
    if (!MicroTileExtent<kFull>(tiling, rowBase, colBase, validRows, validCols)) {
        return;
    }

    const uint32_t productCol = colBase & ((CHEMM_TILE_N / 2U) - 1U);
    const uint32_t productRow = rowBase & (CHEMM_TILE_M - 1U);
    const uint32_t offset = productCol * CHEMM_PLANE_N + productRow;
    TransposePlane16Strides(product + offset, outReal, CHEMM_PLANE_N, kMicro);
    TransposePlane16Strides(product + kPlanarProductHalfPlaneElements + offset, oldReal, CHEMM_PLANE_N, kMicro);
    TransposePlane16Strides(product + 2U * kPlanarProductHalfPlaneElements + offset, outImag, CHEMM_PLANE_N, kMicro);
    TransposePlane16Strides(product + 3U * kPlanarProductHalfPlaneElements + offset, oldImag, CHEMM_PLANE_N, kMicro);
    asc_sync_pipe(PIPE_V);
    asc_sub(outReal, outReal, oldReal, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    asc_add(outImag, outImag, oldImag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    asc_sync_pipe(PIPE_V);

    const uint32_t localCol = colBase & ((CHEMM_TILE_N / 2U) - 1U);
    const uint32_t localRow = rowBase & (CHEMM_TILE_M - 1U);
    const bool zeroBeta = tiling.betaReal == 0.0f && tiling.betaImag == 0.0f;
    if (!zeroBeta) {
        CInterleavedToPlanesStride(
            oldCStage + (localCol - oldCColBase) * (CHEMM_TILE_M * 2U) + localRow * 2U, oldReal, oldImag, kMicro,
            CHEMM_TILE_M * 2U);
        asc_sync_pipe(PIPE_V);
        AccumulateBeta(
            tiling.betaReal == 1.0f && tiling.betaImag == 0.0f, outReal, outImag, oldReal, oldImag, tmp0, tmp1,
            betaReal, betaImag);
    }

    PlanesToCInterleavedStride(
        outReal, outImag, stage + localCol * (CHEMM_TILE_M * 2U) + localRow * 2U, CHEMM_TILE_M * 2U);
    // Finish the transform before the next micro-tile reuses outReal/outImag.
    asc_sync_pipe(PIPE_V);
}

template <bool kFull>
__aicore__ inline bool ConsumePlanarHalf(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t tile, bool stagePending,
    __ubuf__ float* product, __ubuf__ float* stage, __ubuf__ float* oldCStage, __ubuf__ float* outReal,
    __ubuf__ float* outImag, __ubuf__ float* oldReal, __ubuf__ float* oldImag, __ubuf__ float* tmp0,
    __ubuf__ float* tmp1, __ubuf__ float* betaReal, __ubuf__ float* betaImag)
{
    const uint32_t tileRow = TileRow(tiling, tile) * CHEMM_TILE_M;
    const uint32_t tileCol = TileCol(tiling, tile) * CHEMM_TILE_N + half * (CHEMM_TILE_N / 2U);
    uint32_t validRows;
    uint32_t validCols;
    if (!OutputHalfExtent<kFull>(tiling, tileRow, tileCol, validRows, validCols)) {
        return stagePending;
    }
    if (stagePending) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    for (uint32_t panel = 0U; panel < CHEMM_TILE_N / 2U; panel += kStagePanelCols) {
        if (panel >= validCols) {
            continue;
        }
        const uint32_t panelCols = validCols - panel < kStagePanelCols ? validCols - panel : kStagePanelCols;
        if (tiling.betaReal != 0.0f || tiling.betaImag != 0.0f) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
            LoadOutputPanel<kFull>(
                c, tileRow, tileCol + panel, static_cast<uint32_t>(tiling.ldc), validRows, panelCols, oldCStage);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        }
        for (uint32_t col = 0U; col < kStagePanelCols; col += kMicro) {
            for (uint32_t row = 0U; row < CHEMM_TILE_M; row += kMicro) {
                CombinePlanarTile16<kFull>(
                    tiling, tileRow + row, tileCol + panel + col, panel, product, stage, oldCStage, outReal, outImag,
                    oldReal, oldImag, tmp0, tmp1, betaReal, betaImag);
            }
        }
    }
    StoreOutputHalf<kFull>(c, tiling, tileRow, tileCol, validRows, validCols, stage);
    return true;
}

// AIV input production and output consumption.
__aicore__ inline __attribute__((always_inline)) void WaitInputBank(uint32_t job, uint32_t ringSlot, uint32_t ubBank)
{
    if (job >= CHEMM_INPUT_RING) {
        CHEMM_WAIT(kInputFree + ringSlot);
    }
    if (job >= 2U) {
        if (ubBank == 0U) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        } else {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }
    }
}

struct AivPackBuffers {
    __ubuf__ float* input;
    __ubuf__ float* tiles;
    __ubuf__ float* bulk;
    __ubuf__ float* real;
    __ubuf__ float* imag;
    __ubuf__ float* transReal;
    __ubuf__ float* transImag;
    __ubuf__ uint8_t* triangleMask;
    __ubuf__ uint8_t* diagonalMask;
};

__aicore__ inline __attribute__((always_inline)) void BindAivPackBuffers(
    __ubuf__ float* scratch, uint32_t inputSpan, uint32_t bulkSpan, uint32_t planeElements, AivPackBuffers& buffers)
{
    buffers.input = scratch;
    buffers.tiles = buffers.input + inputSpan;
    buffers.bulk = buffers.tiles + 2U * kComplexTileFloats;
    buffers.real = buffers.bulk + bulkSpan;
    buffers.imag = buffers.real + planeElements;
    buffers.transReal = buffers.imag + planeElements;
    buffers.transImag = buffers.transReal + kMicroElements;
    buffers.triangleMask = reinterpret_cast<__ubuf__ uint8_t*>(buffers.transImag + kMicroElements);
    buffers.diagonalMask = buffers.triangleMask + 32U;
}

__aicore__ inline __attribute__((always_inline)) bool BeginAivTileRange(
    uint32_t tileBegin, uint32_t tileEnd, uint32_t& tileCount)
{
    asc_set_mask_norm();
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    tileCount = tileEnd - tileBegin;
    return tileCount != 0U;
}

template <bool kRight, bool kLower, bool kFull>
__aicore__ inline __attribute__((always_inline)) void PackPlanarJob(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t job, __ubuf__ float* inputUb, __ubuf__ float* complexTile,
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ float* alphaReal, __ubuf__ float* alphaImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t ordinal = job / tiling.kFeeds;
    const uint32_t feed = job - ordinal * tiling.kFeeds;
    const uint32_t tile = tileBegin + ordinal;
    const bool reuseCache = ReuseCache(tiling);
    const bool packCache = !reuseCache || ordinal == 0U || CacheGroup(tiling, tile) != CacheGroup(tiling, tile - 1U);
    const uint32_t ringSlot = job % CHEMM_INPUT_RING;
    const uint32_t ubBank = job & 1U;
    WaitInputBank(job, ringSlot, ubBank);
    PackPlanarInputHalf<kRight, kLower, kFull>(
        a, b, workspace, tiling, core, half, tile, feed, ringSlot, ubBank, packCache, inputUb, complexTile, real, imag,
        transReal, transImag, alphaReal, alphaImag, triangleMask, diagonalMask);
}

__aicore__ inline __attribute__((always_inline)) void InitializePlanarCoefficients(
    const ChemmMixedTilingData& tiling, __ubuf__ float* alphaReal, __ubuf__ float* alphaImag, __ubuf__ float* betaReal,
    __ubuf__ float* betaImag)
{
    asc_set_mask_norm();
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    asc_duplicate(alphaReal, tiling.alphaReal, kVectorRepeats, 1U, 8U);
    asc_duplicate(alphaImag, tiling.alphaImag, kVectorRepeats, 1U, 8U);
    asc_duplicate(betaReal, tiling.betaReal, kVectorRepeats, 1U, 8U);
    asc_duplicate(betaImag, tiling.betaImag, kVectorRepeats, 1U, 8U);
    asc_sync_pipe(PIPE_V);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void PackPlanarFeedRange(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t ordinal, uint32_t feedBegin, uint32_t feedEnd, __ubuf__ float* inputUb,
    __ubuf__ float* inputTiles, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal,
    __ubuf__ float* transImag, __ubuf__ float* alphaReal, __ubuf__ float* alphaImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    for (uint32_t feed = feedBegin; feed < feedEnd; ++feed) {
        const uint32_t job = ordinal * tiling.kFeeds + feed;
        PackPlanarJob<kRight, kLower, false>(
            a, b, workspace, tiling, core, half, tileBegin, job, inputUb, inputTiles, real, imag, transReal, transImag,
            alphaReal, alphaImag, triangleMask, diagonalMask);
    }
}

__aicore__ inline __attribute__((always_inline)) void LoadPlanarProduct(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t half, uint32_t ordinal,
    bool releaseProduct, __ubuf__ float* product, __ubuf__ float* oldCStage)
{
    const uint32_t group = ordinal & 1U;
    CHEMM_WAIT(kProductReady + group);
    if (ordinal != 0U) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    }
    __gm__ float* source = ProductSlot(workspace, tiling, core, 2U * group);
    asc_copy_gm2ub(
        product, source + half * kPlanarProductHalfPlaneElements, 4U, kPlanarProductHalfPlaneElements / 8U,
        kPlanarProductHalfPlaneElements / 8U, 0U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    if (tiling.kFeeds > 1U) {
        source = ProductSlot(workspace, tiling, core, 2U * group + 1U);
        constexpr uint8_t repeats = static_cast<uint8_t>(kPlanarProductHalfPlaneElements / 64U);
        for (uint32_t plane = 0U; plane < 4U; ++plane) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
            asc_copy_gm2ub(
                oldCStage,
                source + plane * CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS + half * kPlanarProductHalfPlaneElements,
                kPlanarProductHalfPlaneElements * sizeof(float));
            if (plane == 3U && releaseProduct) {
                CHEMM_NOTIFY(PIPE_MTE2, kProductFree + group);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
            asc_add(
                product + plane * kPlanarProductHalfPlaneElements, product + plane * kPlanarProductHalfPlaneElements,
                oldCStage, repeats, 1U, 1U, 1U, 8U, 8U, 8U);
        }
        asc_sync_pipe(PIPE_V);
    }
    if (tiling.kFeeds == 1U && releaseProduct) {
        CHEMM_NOTIFY(PIPE_MTE2, kProductFree + group);
    }
}

__aicore__ inline __attribute__((always_inline)) void DrainAivOutput(
    uint32_t tileCount, uint32_t kFeeds, bool stagePending)
{
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if (tileCount * kFeeds > 1U) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    if (stagePending) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    WaitComplexTile<0U>();
    WaitComplexTile<1U>();
}

// chemm (arch22) pipelined path overlapping load and compute stages.

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void RunAivPlanarLoop(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling,
    uint32_t core, uint32_t half, uint32_t tileBegin, uint32_t tileCount, __ubuf__ float* inputUb,
    __ubuf__ float* product, __ubuf__ float* stage, __ubuf__ float* oldCStage, __ubuf__ float* inputTiles,
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* oldReal, __ubuf__ float* oldImag,
    __ubuf__ float* tmp0, __ubuf__ float* tmp1, __ubuf__ float* alphaReal, __ubuf__ float* alphaImag,
    __ubuf__ float* betaReal, __ubuf__ float* betaImag, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    ReleaseComplexTile<0U>();
    ReleaseComplexTile<1U>();
    const uint32_t initial = tileCount < CHEMM_PRODUCT_GROUPS ? tileCount : CHEMM_PRODUCT_GROUPS;
    for (uint32_t ordinal = 0U; ordinal < initial; ++ordinal) {
        PackPlanarFeedRange<kRight, kLower>(
            a, b, workspace, tiling, core, half, tileBegin, ordinal, 0U, tiling.kFeeds, inputUb, inputTiles, real, imag,
            transReal, transImag, alphaReal, alphaImag, triangleMask, diagonalMask);
    }
    bool stagePending = false;
    for (uint32_t ordinal = 0U; ordinal < tileCount; ++ordinal) {
        const uint32_t future = ordinal + CHEMM_PRODUCT_GROUPS;
        const uint32_t split = (tiling.kFeeds + 1U) / 2U;
        if (future < tileCount) {
            PackPlanarFeedRange<kRight, kLower>(
                a, b, workspace, tiling, core, half, tileBegin, future, 0U, split, inputUb, inputTiles, real, imag,
                transReal, transImag, alphaReal, alphaImag, triangleMask, diagonalMask);
        }
        LoadPlanarProduct(workspace, tiling, core, half, ordinal, future < tileCount, product, oldCStage);
        stagePending = ConsumePlanarHalf<false>(
            c, tiling, half, tileBegin + ordinal, stagePending, product, stage, oldCStage, outReal, outImag, oldReal,
            oldImag, tmp0, tmp1, betaReal, betaImag);
        if (ordinal + 1U < tileCount) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        }
        if (future < tileCount) {
            PackPlanarFeedRange<kRight, kLower>(
                a, b, workspace, tiling, core, half, tileBegin, future, split, tiling.kFeeds, inputUb, inputTiles, real,
                imag, transReal, transImag, alphaReal, alphaImag, triangleMask, diagonalMask);
        }
    }
    DrainAivOutput(tileCount, tiling.kFeeds, stagePending);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void RunAivPlanar(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling,
    uint32_t core, uint32_t half, uint32_t tileBegin, uint32_t tileEnd, __ubuf__ float* scratch)
{
    __ubuf__ float* inputUb = scratch;
    __ubuf__ float* product = inputUb + 2U * kPlanarInputBankElements;
    __ubuf__ float* stage = product + kPlanarProductHalfElements;
    __ubuf__ float* oldCStage = stage + kStageElements;
    __ubuf__ float* inputTiles = oldCStage + kStagePanelElements;
    __ubuf__ float* real = inputTiles + 2U * kComplexTileFloats;
    __ubuf__ float* imag = real + kMicroElements;
    __ubuf__ float* transReal = imag + kMicroElements;
    __ubuf__ float* transImag = transReal + kMicroElements;
    __ubuf__ float* outReal = transImag + kMicroElements;
    __ubuf__ float* outImag = outReal + kMicroElements;
    __ubuf__ float* oldReal = outImag + kMicroElements;
    __ubuf__ float* oldImag = oldReal + kMicroElements;
    __ubuf__ float* tmp0 = oldImag + kMicroElements;
    __ubuf__ float* tmp1 = tmp0 + kMicroElements;
    __ubuf__ float* alphaReal = tmp1 + kMicroElements;
    __ubuf__ float* alphaImag = alphaReal + kMicroElements;
    __ubuf__ float* betaReal = alphaImag + kMicroElements;
    __ubuf__ float* betaImag = betaReal + kMicroElements;
    __ubuf__ uint8_t* triangleMask = reinterpret_cast<__ubuf__ uint8_t*>(betaImag + kMicroElements);
    __ubuf__ uint8_t* diagonalMask = triangleMask + 32U;

    InitializePlanarCoefficients(tiling, alphaReal, alphaImag, betaReal, betaImag);
    const uint32_t tileCount = tileEnd - tileBegin;
    if (tileCount == 0U) {
        return;
    }
    RunAivPlanarLoop<kRight, kLower>(
        a, b, c, workspace, tiling, core, half, tileBegin, tileCount, inputUb, product, stage, oldCStage, inputTiles,
        real, imag, transReal, transImag, outReal, outImag, oldReal, oldImag, tmp0, tmp1, alphaReal, alphaImag,
        betaReal, betaImag, triangleMask, diagonalMask);
}

template <
    bool kRight, bool kLower, bool kFull, bool kSimpleCoeff, bool kOnePanel = false, bool kKeepDummy = false,
    uint32_t kTileM = CHEMM_TILE_M>
__aicore__ inline void PackTile(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t ordinal, uint32_t tile, __ubuf__ float* inputUb, __ubuf__ float* complexTile,
    __ubuf__ float* bulk, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal,
    __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    const bool reuseCache = ReuseCache(tiling);
    const uint32_t rowBase = TileRow(tiling, tile) * kTileM;
    const uint32_t colBase = TileCol(tiling, tile) * CHEMM_WORK_N;
    const uint32_t cacheGroup = colBase / CHEMM_WORK_N;
    const uint32_t cacheBank = cacheGroup & 1U;
    const bool sharedCache = kOnePanel && reuseCache && tiling.tileColBase == 0U && tiling.blockDim >= 8U;
    const uint32_t previousCacheGroup = ordinal == 0U ? cacheGroup : CacheGroup(tiling, tile - 1U);
    const bool packCache =
        !reuseCache || (!sharedCache || core == 0U) && (ordinal == 0U || cacheGroup != previousCacheGroup);
    const bool syncSharedCache = sharedCache && ordinal == 0U;
    uint32_t job = ordinal * tiling.kFeeds;
    uint32_t ringSlot = job % CHEMM_INPUT_RING;
    uint32_t ubBank = job & 1U;
    __gm__ float* slot = InputSlot(workspace, tiling, core, ringSlot);
    for (uint32_t feed = 0U; feed < tiling.kFeeds; ++feed) {
        WaitInputBank(job, ringSlot, ubBank);
        PackInputHalf<kRight, kLower, kFull, kSimpleCoeff, kOnePanel, kKeepDummy, kTileM>(
            a, b, workspace, tiling, core, half, feed, ringSlot, rowBase, colBase, cacheBank, slot, ubBank, packCache,
            sharedCache, syncSharedCache, inputUb, complexTile, bulk, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
        ++job;
        ++ringSlot;
        if (ringSlot == CHEMM_INPUT_RING) {
            ringSlot = 0U;
            slot = InputSlot(workspace, tiling, core, 0U);
        } else {
            slot += tiling.inputSlotStride;
        }
        ubBank ^= 1U;
    }
}

template <bool kRight, bool kLower, bool kFull, bool kOnePanel, bool kKeepDummy>
__aicore__ __attribute__((always_inline)) inline void RunAivDirectBody(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t tileEnd, __ubuf__ float* scratch)
{
    constexpr uint32_t halfRows = CHEMM_TILE_M / 2U;
    constexpr uint32_t paddedRows = (halfRows + kMicro - 1U) / kMicro * kMicro;
    constexpr uint32_t qMicroBlocks = 2U * paddedRows / kMicro;
    constexpr uint32_t bulkElements = 2U * paddedRows * CHEMM_K_FEED;
    uint32_t inputBankElements = kAivInputBankElements;
    if constexpr (kOnePanel) {
        inputBankElements = AivInputBankElements<true, CHEMM_TILE_M>(tiling.panelM);
    }
    AivPackBuffers buffers;
    BindAivPackBuffers(scratch, 2U * inputBankElements, 2U * bulkElements, qMicroBlocks * kMicroElements, buffers);
    uint32_t tileCount;
    if (!BeginAivTileRange(tileBegin, tileEnd, tileCount)) {
        return;
    }
    ReleaseComplexTile<0U>();
    ReleaseComplexTile<1U>();
    for (uint32_t ordinal = 0U; ordinal < tileCount; ++ordinal) {
        PackTile<kRight, kLower, kFull, true, kOnePanel, kKeepDummy>(
            a, b, workspace, tiling, core, half, ordinal, tileBegin + ordinal, buffers.input, buffers.tiles,
            buffers.bulk, buffers.real, buffers.imag, buffers.transReal, buffers.transImag, buffers.triangleMask,
            buffers.diagonalMask);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if (tileCount * tiling.kFeeds > 1U) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    WaitComplexTile<0U>();
    WaitComplexTile<1U>();
}

template <bool kRight, bool kLower>
__aicore__ __attribute__((always_inline)) inline void RunAivDirect(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t tileEnd, __ubuf__ float* scratch)
{
    const uint32_t flags = tiling.kernelFlags;
    const bool full = (flags & CHEMM_FLAG_FULL) != 0U;
    const bool onePanel = (flags & CHEMM_FLAG_ONE_PANEL) != 0U;
    const bool keepDummy = (flags & CHEMM_FLAG_KEEP_DUMMY) != 0U;
    if (onePanel) {
        if (full) {
            RunAivDirectBody<kRight, kLower, true, true, true>(
                a, b, workspace, tiling, core, half, tileBegin, tileEnd, scratch);
        } else if (keepDummy) {
            RunAivDirectBody<kRight, kLower, false, true, true>(
                a, b, workspace, tiling, core, half, tileBegin, tileEnd, scratch);
        } else {
            RunAivDirectBody<kRight, kLower, false, true, false>(
                a, b, workspace, tiling, core, half, tileBegin, tileEnd, scratch);
        }
    } else if (full) {
        RunAivDirectBody<kRight, kLower, true, false, false>(
            a, b, workspace, tiling, core, half, tileBegin, tileEnd, scratch);
    } else if (keepDummy) {
        RunAivDirectBody<kRight, kLower, false, false, true>(
            a, b, workspace, tiling, core, half, tileBegin, tileEnd, scratch);
    } else {
        RunAivDirectBody<kRight, kLower, false, false, false>(
            a, b, workspace, tiling, core, half, tileBegin, tileEnd, scratch);
    }
}

// Shared-panel production and narrow result combination.
__aicore__ __attribute__((always_inline)) inline void RunAicFullPanel(
    __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t begin,
    uint32_t end, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    RunAic<true, true, CHEMM_EXP_M>(c, workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
}

__aicore__ __attribute__((always_inline)) inline bool UseSharedPanelCache(const ChemmMixedTilingData& tiling)
{
    return tiling.tileColBase == 0U && tiling.blockDim >= 8U;
}

__aicore__ __attribute__((always_inline)) inline __gm__ float* PrepareAivPanelCache(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, bool sharedCache)
{
    ReleaseComplexTile<0U>();
    ReleaseComplexTile<1U>();
    const uint32_t cacheCore = sharedCache ? 0U : core;
    const uint32_t cacheBank = tiling.tileColBase & 1U;
    return CacheSlot(workspace, tiling, cacheCore, cacheBank, 0U, 0U);
}

// Lightweight producer for the resident-panel MIX kernel. Every P/Q element
// written to GM is produced by a vector operation, so no padding clear is
// needed. P is packed only for the first output tile; Q keeps the two-UB/three-
// slot schedule used by the AIC consumer.
template <bool kRight, bool kLower, uint32_t kTileM>
__aicore__ __attribute__((always_inline)) inline void PackAivPanelQ(
    __gm__ float* a, __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t rowBase,
    uint32_t kBase, __ubuf__ float* q, __ubuf__ float* bulk, __ubuf__ float* inputTiles, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    constexpr bool kBulkQ = kRight && kTileM % (2U * kMicro) == 0U;
    if constexpr (kBulkQ) {
        if (rowBase + kTileM <= static_cast<uint32_t>(tiling.m)) {
            const uint32_t validK = static_cast<uint32_t>(tiling.dimA) - kBase < CHEMM_K_FEED ?
                                        static_cast<uint32_t>(tiling.dimA) - kBase :
                                        CHEMM_K_FEED;
            PackBulkQRightHalf<kTileM>(b, tiling, half, rowBase, kBase, validK, q, bulk, real, imag);
            return;
        }
    }
    PackTileHalf<kRight, kLower, false, false, false, kTileM>(
        a, b, tiling, half, rowBase, kBase, q, bulk, inputTiles, real, imag, transReal, transImag, triangleMask,
        diagonalMask);
}

template <uint32_t kTileM>
__aicore__ __attribute__((always_inline)) inline void PublishAivPanelInput(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t half, uint32_t feed,
    uint32_t ring, uint32_t ubBank, bool packP, bool syncSharedCache, __gm__ float* pCache, __ubuf__ float* p,
    __ubuf__ float* q)
{
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    StoreQHalf<kTileM>(InputSlot(workspace, tiling, core, ring), q, half);
    if (packP) {
        StorePHalf<true>(pCache + feed * CHEMM_INPUT_MATRIX_ELEMENTS, p, half, tiling.panelM);
    }
    if (syncSharedCache) {
        ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0U, kSharedCacheReady));
        wait_flag_dev(kSharedCacheReady);
    }
    if (ubBank == 0U) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    CHEMM_NOTIFY(PIPE_MTE3, kInputReady + ring);
}

template <bool kRight, bool kLower, uint32_t kTileM>
__aicore__ __attribute__((always_inline)) inline void PackAivPanelFeed(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t ordinal, uint32_t feed, uint32_t job, uint32_t rowBase, uint32_t colBase,
    uint32_t inputBankElements, uint32_t pElements, bool packP, bool sharedCache, __gm__ float* pCache,
    __ubuf__ float* inputUb, __ubuf__ float* inputTiles, __ubuf__ float* bulk, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t ring = job % CHEMM_INPUT_RING;
    const uint32_t ubBank = job & 1U;
    WaitInputBank(job, ring, ubBank);
    __ubuf__ float* p = inputUb + ubBank * inputBankElements;
    __ubuf__ float* q = p + pElements;
    const uint32_t kBase = feed * CHEMM_K_FEED;
    if (packP) {
        PackCachedHalf<kRight, kLower, false, false, true>(
            a, b, tiling, half, colBase, kBase, tiling.panelM, p, q, q, inputTiles, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
        asc_sync_pipe(PIPE_V);
    }
    PackAivPanelQ<kRight, kLower, kTileM>(
        a, b, tiling, half, rowBase, kBase, q, bulk, inputTiles, real, imag, transReal, transImag, triangleMask,
        diagonalMask);
    PublishAivPanelInput<kTileM>(
        workspace, tiling, core, half, feed, ring, ubBank, packP, sharedCache && ordinal == 0U, pCache, p, q);
}

template <bool kRight, bool kLower, uint32_t kTileM>
__aicore__ __attribute__((always_inline)) inline void ProduceAivPanelTiles(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t tileCount, uint32_t inputBankElements, uint32_t pElements,
    __ubuf__ float* inputUb, __ubuf__ float* inputTiles, __ubuf__ float* bulk, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const bool sharedCache = UseSharedPanelCache(tiling);
    __gm__ float* pCache = PrepareAivPanelCache(workspace, tiling, core, sharedCache);
    const uint32_t colBase = tiling.tileColBase * CHEMM_WORK_N;
    uint32_t job = 0U;
    for (uint32_t ordinal = 0U; ordinal < tileCount; ++ordinal) {
        const uint32_t rowBase = (tiling.tileRowBase + tileBegin + ordinal) * kTileM;
        const bool packP = ordinal == 0U && (!sharedCache || core == 0U);
        for (uint32_t feed = 0U; feed < tiling.kFeeds; ++feed, ++job) {
            PackAivPanelFeed<kRight, kLower, kTileM>(
                a, b, workspace, tiling, core, half, ordinal, feed, job, rowBase, colBase, inputBankElements, pElements,
                packP, sharedCache, pCache, inputUb, inputTiles, bulk, real, imag, transReal, transImag, triangleMask,
                diagonalMask);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if (job > 1U) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    WaitComplexTile<0U>();
    WaitComplexTile<1U>();
}

template <bool kRight, bool kLower, uint32_t kTileM>
__aicore__ __attribute__((always_inline)) inline void RunAivPanel(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t tileEnd, __ubuf__ float* scratch)
{
    constexpr bool kBulkQ = kRight && kTileM % (2U * kMicro) == 0U;
    constexpr uint32_t halfRows = kTileM / 2U;
    constexpr uint32_t paddedRows = (halfRows + kMicro - 1U) / kMicro * kMicro;
    constexpr uint32_t qMicroBlocks = 2U * paddedRows / kMicro;
    constexpr uint32_t bulkElements = kBulkQ ? 2U * paddedRows * CHEMM_K_FEED : 1U;
    constexpr uint32_t planeElements = kBulkQ ? qMicroBlocks * kMicroElements : kMicroElements;
    const uint32_t inputBankElements = AivInputBankElements<true, kTileM>(tiling.panelM);
    const uint32_t pElements = tiling.panelM * CHEMM_EXP_K;
    AivPackBuffers buffers;
    BindAivPackBuffers(scratch, 2U * inputBankElements, 2U * bulkElements, planeElements, buffers);
    uint32_t tileCount;
    if (!BeginAivTileRange(tileBegin, tileEnd, tileCount)) {
        return;
    }

    ProduceAivPanelTiles<kRight, kLower, kTileM>(
        a, b, workspace, tiling, core, half, tileBegin, tileCount, inputBankElements, pElements, buffers.input,
        buffers.tiles, buffers.bulk, buffers.real, buffers.imag, buffers.transReal, buffers.transImag,
        buffers.triangleMask, buffers.diagonalMask);
}

template <bool kLower>
__aicore__ __attribute__((always_inline)) inline void PackAivPanelPairQ(
    __gm__ float* a, __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t rowBase, uint32_t validK,
    __ubuf__ float* q, __ubuf__ float* bulk, __ubuf__ float* inputTiles, __ubuf__ float* real, __ubuf__ float* imag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    constexpr uint32_t kTileM = 64U;
    constexpr uint32_t kPairM = 2U * kTileM;
    if (rowBase + kTileM <= static_cast<uint32_t>(tiling.m)) {
        PackBulkQRightHalf<kPairM>(b, tiling, 0U, rowBase, 0U, validK, q, bulk, real, imag);
    } else {
        PackTileHalf<true, kLower, false, false, false, kPairM>(
            a, b, tiling, 0U, rowBase, 0U, q, bulk, inputTiles, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
    }
}

__aicore__ __attribute__((always_inline)) inline void PublishAivPanelPair(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t half, uint32_t ordinal,
    uint32_t ring, uint32_t ubBank, bool packP, bool sharedCache, bool deferPeerReady, __gm__ float* pCache,
    __ubuf__ float* p, __ubuf__ float* q)
{
    constexpr uint32_t kPairM = 128U;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    StoreQHalf<kPairM>(InputSlot(workspace, tiling, core, ring), q, half);
    if (packP) {
        StorePHalf<true>(pCache, p, half, tiling.panelM);
    }
    if (sharedCache && ordinal == 0U) {
        ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0U, kPairSharedCacheReady));
        wait_flag_dev(kPairSharedCacheReady);
    }
    if (deferPeerReady) {
        CHEMM_NOTIFY(PIPE_MTE3, kPairReady + 2U * ring + (half ^ 1U));
    }
    if (ubBank == 0U) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    CHEMM_NOTIFY(PIPE_MTE3, kPairReady + 2U * ring + half);
}

template <bool kLower>
__aicore__ __attribute__((always_inline)) inline void PackAivPanelPairTile(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t ordinal, uint32_t pElements, uint32_t qElements, uint32_t validK,
    bool sharedCache, __gm__ float* pCache, __ubuf__ float* p, __ubuf__ float* inputTiles, __ubuf__ float* bulk,
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    constexpr uint32_t kTileM = 64U;
    constexpr uint32_t kPairM = 2U * kTileM;
    const uint32_t ring = ordinal % CHEMM_INPUT_RING;
    if (ordinal >= CHEMM_INPUT_RING) {
        CHEMM_WAIT(kPairFree + ring);
    }
    const uint32_t ubBank = ordinal & 1U;
    if (ordinal >= 2U) {
        wait_flag(PIPE_MTE3, PIPE_V, ubBank == 0U ? EVENT_ID0 : EVENT_ID1);
    }
    __ubuf__ float* q = p + pElements + ubBank * qElements;
    const bool packP = ordinal == 0U && (!sharedCache || core == 0U);
    const bool deferPeerReady = ordinal == 0U && !sharedCache;
    if (!deferPeerReady) {
        CHEMM_NOTIFY(PIPE_MTE3, kPairReady + 2U * ring + (half ^ 1U));
    }
    if (packP) {
        const uint32_t colBase = tiling.tileColBase * CHEMM_WORK_N;
        PackCachedHalf<true, kLower, false, false, true>(
            a, b, tiling, half, colBase, 0U, tiling.panelM, p, q, q, inputTiles, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
        asc_sync_pipe(PIPE_V);
    }
    const uint32_t rowBase = (tiling.tileRowBase + tileBegin + ordinal) * kPairM + half * kTileM;
    PackAivPanelPairQ<kLower>(
        a, b, tiling, rowBase, validK, q, bulk, inputTiles, real, imag, transReal, transImag, triangleMask,
        diagonalMask);
    PublishAivPanelPair(
        workspace, tiling, core, half, ordinal, ring, ubBank, packP, sharedCache, deferPeerReady, pCache, p, q);
}

template <bool kLower>
__aicore__ __attribute__((always_inline)) inline void ProduceAivPanelPairs(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t tileCount, uint32_t pElements, uint32_t qElements,
    __ubuf__ float* inputUb, __ubuf__ float* inputTiles, __ubuf__ float* bulk, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const bool sharedCache = UseSharedPanelCache(tiling);
    __gm__ float* pCache = PrepareAivPanelCache(workspace, tiling, core, sharedCache);
    const uint32_t validK = static_cast<uint32_t>(tiling.dimA);
    for (uint32_t ordinal = 0U; ordinal < tileCount; ++ordinal) {
        PackAivPanelPairTile<kLower>(
            a, b, workspace, tiling, core, half, tileBegin, ordinal, pElements, qElements, validK, sharedCache, pCache,
            inputUb, inputTiles, bulk, real, imag, transReal, transImag, triangleMask, diagonalMask);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if (tileCount > 1U) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    WaitComplexTile<0U>();
    WaitComplexTile<1U>();
}

template <bool kLower>
__aicore__ __attribute__((always_inline)) inline void RunAivPanelPair(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tileBegin, uint32_t tileEnd, __ubuf__ float* scratch)
{
    constexpr uint32_t tileM = 64U;
    constexpr uint32_t pairM = 2U * tileM;
    constexpr uint32_t halfRows = pairM / 2U;
    constexpr uint32_t qElements = 2U * tileM * CHEMM_EXP_K;
    const uint32_t pElements = tiling.panelM * CHEMM_EXP_K;
    constexpr uint32_t bulkElements = 2U * halfRows * CHEMM_K_FEED;
    constexpr uint32_t qMicroBlocks = 2U * halfRows / kMicro;
    AivPackBuffers buffers;
    BindAivPackBuffers(scratch, pElements + 2U * qElements, bulkElements, qMicroBlocks * kMicroElements, buffers);
    uint32_t tileCount;
    if (!BeginAivTileRange(tileBegin, tileEnd, tileCount)) {
        return;
    }
    ProduceAivPanelPairs<kLower>(
        a, b, workspace, tiling, core, half, tileBegin, tileCount, pElements, qElements, buffers.input, buffers.tiles,
        buffers.bulk, buffers.real, buffers.imag, buffers.transReal, buffers.transImag, buffers.triangleMask,
        buffers.diagonalMask);
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
    asc_copy_gm2ub_align(triangleMask, maskGm + kTriangleOffset, kMaskBytes);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(transReal), reinterpret_cast<__ubuf__ uint32_t*>(complexTile), nullptr, 8U,
        1U, 1U, 8U, 8U);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(transImag), reinterpret_cast<__ubuf__ uint32_t*>(complexTile), nullptr, 8U,
        1U, 2U, 8U, 8U);
    asc_sync_pipe(PIPE_V);
    TransposePlane16(transReal, real);
    TransposePlane16(transImag, imag);
    asc_sync_pipe(PIPE_V);
    asc_mul_scalar(transImag, transImag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
    ApplyHermitianMasks(real, imag, transReal, transImag, triangleMask, diagonalMask);
}

template <bool kRight, bool kFull>
__aicore__ inline __attribute__((always_inline)) void LoadAivBPlanes(
    __gm__ float* b, const ChemmAivTilingData& tiling, __ubuf__ float* raw, __ubuf__ float* rowReal,
    __ubuf__ float* rowImag, __ubuf__ float* colReal, __ubuf__ float* colImag)
{
    LoadAivComplexTile<kFull, EVENT_ID3>(
        b, static_cast<uint32_t>(tiling.ldb), static_cast<uint32_t>(tiling.m), static_cast<uint32_t>(tiling.n), raw);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    if constexpr (kRight) {
        // Column-major B is the natural broadcast operand of B * A.
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(colReal), reinterpret_cast<__ubuf__ uint32_t*>(raw), nullptr, 8U, 1U,
            1U, 8U, 8U);
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(colImag), reinterpret_cast<__ubuf__ uint32_t*>(raw), nullptr, 8U, 1U,
            2U, 8U, 8U);
    } else {
        // Row-major B is reused by every output row of A * B.
        CInterleavedToPlanes(raw, rowReal, rowImag);
    }
}

} // namespace
