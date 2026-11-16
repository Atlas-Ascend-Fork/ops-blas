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

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "chemm_tiling_data.h"

namespace {

// chemm (arch22) shared device helpers: Hermitian decode (conjugate mirror,
// zeroed diagonal imaginary part, uplo triangle selection), on-chip transpose
// and the complex multiply-accumulate primitives reused by every strategy.

// Shared geometry, workspace slots and instruction helpers.
// Mode 2 rendezvouses both AIV producers with their paired AIC consumer.
#define CHEMM_NOTIFY(pipe, flag) ffts_cross_core_sync((pipe), AscendC::GetffstMsg(2U, (flag)))
#define CHEMM_WAIT(flag) wait_flag_dev((flag))

constexpr uint32_t kMicro = CHEMM_MICRO_TILE;
constexpr uint32_t kMicroElements = kMicro * kMicro;
constexpr uint32_t kComplexTileFloats = 2U * kMicroElements;
constexpr uint32_t kComplexColumnFloats = 2U * kMicro;
constexpr uint32_t kFp32DataBlockElements = 32U / sizeof(float);
constexpr uint32_t kHalfInputElements = CHEMM_INPUT_MATRIX_ELEMENTS / 2U;
constexpr uint32_t kAivInputBankElements = 3U * kHalfInputElements;
constexpr uint32_t kHalfProductElements = CHEMM_PRODUCT_ELEMENTS / 2U;
constexpr uint32_t kPlanarPHalfElements = CHEMM_PLANE_P_ELEMENTS / 2U;
constexpr uint32_t kPlanarQHalfElements = CHEMM_PLANE_Q_ELEMENTS / 2U;
constexpr uint32_t kPlanarInputBankElements = 2U * kPlanarPHalfElements + 2U * kPlanarQHalfElements;
constexpr uint32_t kPlanarProductHalfElements = CHEMM_PLANAR_PRODUCT_ELEMENTS / 2U;
constexpr uint32_t kPlanarProductHalfPlaneElements = CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS / 2U;
constexpr uint32_t kStageElements = (CHEMM_TILE_N / 2U) * CHEMM_TILE_M * 2U;
constexpr uint32_t kStagePanelCols = CHEMM_TILE_N / 4U;
constexpr uint32_t kStagePanelElements = kStagePanelCols * CHEMM_TILE_M * 2U;
static_assert(
    kStagePanelElements == kPlanarProductHalfPlaneElements, "planar C panel must share one product-plane scratch bank");
constexpr uint32_t kL1FractalElements = kMicro * (32U / sizeof(float));
constexpr uint32_t kL0aMFractals = CHEMM_EXP_M / kMicro;
constexpr uint32_t kL0aKFractals = CHEMM_EXP_K / (32U / sizeof(float));
constexpr uint32_t kPlanarL0aKFractals = CHEMM_K_FEED / (32U / sizeof(float));
constexpr uint32_t kPlanarQFractals = CHEMM_PLANE_Q_ELEMENTS / kL1FractalElements;
constexpr uint8_t kVectorRepeats = 4U;
constexpr uint32_t kAivRowsPerVectorCore = 8U;
constexpr uint8_t kAivRowsPerBroadcastRepeat = 8U;
constexpr uint8_t kAivRowsPerTransposeRepeat = 4U;
// Two 32 KiB planes leave more than 32 KiB of the A2 vector-local memory
// free in the largest 16-row Grid specialization.
constexpr uint32_t kAivYCachePlaneElements = CHEMM_AIV_CACHE_FEEDS * kMicroElements;
constexpr uint64_t kFullFp32Mask = 0xffffffffffffffffULL;
constexpr uint64_t kMaskAddressVectorMask = 0xffffffffULL;
// All regular MIX-AIV algorithms fit the same 178 KiB scratch layout. The
// active algorithm partitions this one buffer; mutually exclusive modes do
// not emit duplicate UB objects.
constexpr uint32_t kMixedAivScratchElements = 2U * kAivInputBankElements + kHalfProductElements + kStageElements +
                                              3U * kComplexTileFloats + 12U * kMicroElements + 16U;
constexpr uint32_t kDirectAivScratchElements = 2U * kAivInputBankElements + 2U * kComplexTileFloats +
                                               2U * CHEMM_TILE_M * CHEMM_K_FEED + 10U * kMicroElements + 16U;
static_assert(
    kDirectAivScratchElements <= kMixedAivScratchElements, "direct MIX scratch must fit the shared MIX allocation");
// Keep both operands on the same 16-complex cyclic K order. This is math
// invariant and avoids the A2 FP32 MMAD cancellation failure seen in natural order.

constexpr uint32_t kInputReady = 0U;
constexpr uint32_t kInputFree = 3U;
constexpr uint32_t kProductReady = 6U;
constexpr uint32_t kProductFree = 8U;
constexpr uint32_t kSharedCacheReady = 8U;
constexpr uint32_t kPairReady = 0U;
constexpr uint32_t kPairFree = 6U;
constexpr uint32_t kPairSharedCacheReady = 10U;
constexpr uint32_t kNarrowReady = 0U;
constexpr uint32_t kNarrowFree = 3U;
constexpr uint32_t kNarrowPReady = 6U;
constexpr uint64_t kOddFp32Mask = 0xaaaaaaaaaaaaaaaaULL;
static __gm__ uint64_t g_packMasks[12] __attribute__((aligned(32))) = {
    0xfff8fffcfffeffffULL, 0xff80ffc0ffe0fff0ULL, 0xf800fc00fe00ff00ULL, 0x8000c000e000f000ULL,
    0x000f000700030001ULL, 0x00ff007f003f001fULL, 0x0fff07ff03ff01ffULL, 0xffff7fff3fff1fffULL,
    0x0008000400020001ULL, 0x0080004000200010ULL, 0x0800040002000100ULL, 0x8000400020001000ULL};
constexpr uint32_t kNarrowStage0Rows = CHEMM_NARROW_STAGE0_ROWS;
constexpr uint32_t kNarrowStage0N = CHEMM_NARROW_STAGE0_N;
constexpr uint32_t kNarrowStage0QElements = CHEMM_NARROW_EXP_K * kNarrowStage0N;
constexpr uint32_t kNarrowStage1MinN = CHEMM_NARROW_STAGE1_MIN_N;
constexpr uint32_t kNarrowStage1MaxN = CHEMM_NARROW_STAGE1_MAX_N;
constexpr uint32_t kNarrowStageN = CHEMM_NARROW_STAGE_N;
constexpr uint32_t kNarrowStageL1Elements = kMicro * kNarrowStageN + 2U * kMicro * (kNarrowStage0N + kNarrowStage1MaxN);
constexpr uint32_t kNarrowAivScratchElements = kMicro * CHEMM_NARROW_EXP_K + 2U * kComplexTileFloats +
                                               4U * CHEMM_NARROW_PANEL_N * CHEMM_NARROW_PRODUCT_N +
                                               32U * CHEMM_NARROW_PRODUCT_N + 4U * CHEMM_NARROW_TILE_M * kMicro + 16U;
static_assert(
    kNarrowStageL1Elements == CHEMM_NARROW_STAGE_Q_ELEMENTS, "staged narrow B panels must exactly fit the L1 bank");

template <bool kOnePanel, uint32_t kTileM>
__aicore__ inline uint32_t AivInputBankElements(uint32_t panelM)
{
    constexpr uint32_t halfRows = kTileM / 2U;
    constexpr uint32_t packedRows = 2U * ((halfRows + kMicro - 1U) / kMicro * kMicro);
    return kOnePanel ? (panelM + packedRows) * CHEMM_EXP_K : kAivInputBankElements;
}

__aicore__ inline uint64_t ElementIndex(uint32_t row, uint32_t col, uint32_t ld)
{
    return static_cast<uint64_t>(col) * ld + row;
}

template <typename Tiling>
__aicore__ inline __gm__ uint8_t* MaskGm(const Tiling&)
{
    return reinterpret_cast<__gm__ uint8_t*>(g_packMasks);
}

__aicore__ inline __gm__ float* InputSlot(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t slot)
{
    return workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride +
           static_cast<uint64_t>(slot) * tiling.inputSlotStride;
}

__aicore__ inline uint32_t TileCount(const ChemmMixedTilingData& tiling) { return tiling.tileRows * tiling.tileCols; }

__aicore__ inline __gm__ float* ProductSlot(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t slot)
{
    return workspace + tiling.productOffset + static_cast<uint64_t>(core) * tiling.productCoreStride +
           static_cast<uint64_t>(slot) * tiling.productSlotStride;
}

__aicore__ inline __gm__ float* NarrowStageP(__gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core)
{
    return workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride;
}

__aicore__ inline __gm__ float* NarrowStageTail(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t chunk)
{
    return NarrowStageP(workspace, tiling, core) + CHEMM_NARROW_STAGE_P_ELEMENTS +
           static_cast<uint64_t>(chunk) * CHEMM_NARROW_STAGE_TAIL_ELEMENTS;
}

__aicore__ inline uint32_t NarrowStageRowBegin(const ChemmMixedTilingData& tiling, uint32_t core)
{
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(tiling.m) * core / tiling.blockDim) / CHEMM_NARROW_STAGE_ROW_ALIGN *
        CHEMM_NARROW_STAGE_ROW_ALIGN);
}

__aicore__ inline uint32_t NarrowStageRowEnd(const ChemmMixedTilingData& tiling, uint32_t core)
{
    return core + 1U == tiling.blockDim ? static_cast<uint32_t>(tiling.m) : NarrowStageRowBegin(tiling, core + 1U);
}

__aicore__ inline uint32_t TileRow(const ChemmMixedTilingData& tiling, uint32_t tile)
{
    return tiling.tileRowBase + tile % tiling.tileRows;
}

__aicore__ inline uint32_t TileCol(const ChemmMixedTilingData& tiling, uint32_t tile)
{
    return tiling.tileColBase + tile / tiling.tileRows;
}

__aicore__ inline uint32_t CacheGroup(const ChemmMixedTilingData& tiling, uint32_t tile)
{
    return TileCol(tiling, tile);
}

template <uint32_t kSrcStride>
__aicore__ __attribute__((always_inline)) inline void StoreComplexTail(
    __gm__ float* dst, __ubuf__ float* src, uint32_t cols, uint32_t rows, uint32_t ld)
{
    const uint32_t floats = 2U * rows;
    const uint32_t alignedFloats = (floats + 7U) & ~7U;
    // One aligned multi-burst replaces a scalar column loop. UB gap is in
    // 32-byte blocks; GM gap and burst length are in bytes on arch22.
    asc_copy_ub2gm_align(
        dst, src, static_cast<uint16_t>(cols), floats * sizeof(float), 0U, 0U, (kSrcStride - alignedFloats) / 8U,
        (2U * ld - floats) * sizeof(float));
}

__aicore__ inline bool ReuseCache(const ChemmMixedTilingData& tiling) { return tiling.cacheSlots != 0U; }

__aicore__ inline __gm__ float* CacheSlot(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t bank, uint32_t feed,
    uint32_t panel)
{
    // Two banks let AIV produce macro t+1 while AIC consumes macro t; the
    // non-macro panel cache uses the same layout.
    const uint64_t offset = static_cast<uint64_t>(tiling.workspaceCores) * tiling.inputCoreStride;
    const uint64_t coreStride = static_cast<uint64_t>(tiling.cacheSlots) * CHEMM_INPUT_MATRIX_ELEMENTS;
    return workspace + offset + static_cast<uint64_t>(core) * coreStride +
           static_cast<uint64_t>((bank * tiling.cachePanels + panel) * tiling.cacheFeeds + feed) *
               CHEMM_INPUT_MATRIX_ELEMENTS;
}

template <uint32_t kCols = kMicro>
__aicore__ inline __attribute__((always_inline)) void CInterleavedToPlanesStride(
    __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag, uint32_t dstStride,
    uint32_t srcStride = kComplexColumnFloats, uint8_t repeats = kMicro / kAivRowsPerTransposeRepeat)
{
    static_assert(kCols > 0U && kCols <= kMicro, "transpose source width must fit one vector bank");
    __ubuf__ float* src[16];
    __ubuf__ float* dst[16];
    for (uint32_t col = 0U; col < kCols; ++col) {
        src[col] = complexTile + col * srcStride;
    }
    for (uint32_t col = kCols; col < kMicro; ++col) {
        src[col] = complexTile;
    }
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        const uint32_t row = lane / 2U;
        __ubuf__ float* plane = (lane & 1U) == 0U ? real : imag;
        dst[2U * lane] = plane + row * dstStride;
        dst[2U * lane + 1U] = plane + row * dstStride + 8U;
    }
    asc_set_va_reg(VA0, dst);
    asc_set_va_reg(VA1, dst + 8U);
    asc_set_va_reg(VA2, src);
    asc_set_va_reg(VA3, src + 8U);
    // Each repeat advances source by one datablock and destination by four
    // planar rows. Four repeats cover one 16-row tile; bulk callers extend
    // the same instruction across adjacent tiles.
    asc_transto5hd_b32(VA0, VA2, repeats, static_cast<uint16_t>(dstStride / 2U), 1U);
}

__aicore__ inline __attribute__((always_inline)) void CInterleavedToPlanes(
    __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag)
{
    CInterleavedToPlanesStride(complexTile, real, imag, kMicro);
}

__aicore__ inline __attribute__((always_inline)) void PlanesToCInterleavedStride(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* complexTile, uint32_t dstStride, uint32_t rowShift = 0U)
{
    __ubuf__ float* src[16];
    __ubuf__ float* dst[16];
    for (uint32_t group = 0U; group < 2U; ++group) {
        for (uint32_t row = 0U; row < 8U; ++row) {
            const uint32_t planeRow = (group * 8U + row + rowShift) & (kMicro - 1U);
            src[2U * row] = real + planeRow * kMicro;
            src[2U * row + 1U] = imag + planeRow * kMicro;
        }
        for (uint32_t col = 0U; col < 8U; ++col) {
            const uint32_t rowOffset = group * 16U;
            dst[2U * col] = complexTile + col * dstStride + rowOffset;
            dst[2U * col + 1U] = complexTile + col * dstStride + rowOffset + 8U;
        }
        asc_set_va_reg(VA0, dst);
        asc_set_va_reg(VA1, dst + 8U);
        asc_set_va_reg(VA2, src);
        asc_set_va_reg(VA3, src + 8U);
        asc_transto5hd_b32(VA0, VA2, 2U, static_cast<uint16_t>(dstStride), 1U);
    }
}

template <bool kSingleRepeat = false>
__aicore__ inline __attribute__((always_inline)) void PlanesToCInterleavedRows8(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* complexTile, uint32_t dstStride)
{
    __ubuf__ float* src[16];
    __ubuf__ float* dst[16];
    for (uint32_t row = 0U; row < 8U; ++row) {
        src[2U * row] = real + row * kMicro;
        src[2U * row + 1U] = imag + row * kMicro;
    }
    for (uint32_t col = 0U; col < 8U; ++col) {
        dst[2U * col] = complexTile + col * dstStride;
        dst[2U * col + 1U] = complexTile + col * dstStride + 8U;
    }
    asc_set_va_reg(VA0, dst);
    asc_set_va_reg(VA1, dst + 8U);
    asc_set_va_reg(VA2, src);
    asc_set_va_reg(VA3, src + 8U);
    if constexpr (kSingleRepeat) {
        // Narrow AIVs own eight output columns.  A second repeat would write
        // p8..p15 beyond the 8-column stage bank and corrupt adjacent UB.
        asc_transto5hd_b32(VA0, VA2, 1U, 0U, 0U);
    } else {
        constexpr uint8_t kRepeats = static_cast<uint8_t>(kAivRowsPerVectorCore / kAivRowsPerTransposeRepeat);
        // One instruction transforms the complete row batch owned by this AIV.
        asc_transto5hd_b32(VA0, VA2, kRepeats, static_cast<uint16_t>(dstStride), 1U);
    }
}

__aicore__ inline __attribute__((always_inline)) void TransposePlane16Strides(
    __ubuf__ float* srcTile, __ubuf__ float* dstTile, uint32_t srcStride, uint32_t dstStride)
{
    __ubuf__ float* src[16];
    __ubuf__ float* dst[16];
    for (uint32_t row = 0U; row < kMicro; ++row) {
        src[row] = srcTile + row * srcStride;
    }
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        dst[2U * lane] = dstTile + lane * dstStride;
        dst[2U * lane + 1U] = dstTile + lane * dstStride + 8U;
    }
    asc_set_va_reg(VA0, dst);
    asc_set_va_reg(VA1, dst + 8U);
    asc_set_va_reg(VA2, src);
    asc_set_va_reg(VA3, src + 8U);
    asc_transto5hd_b32(VA0, VA2, 2U, static_cast<uint16_t>(dstStride), 1U);
}

__aicore__ inline __attribute__((always_inline)) void TransposePlane16Stride(
    __ubuf__ float* srcTile, __ubuf__ float* dstTile, uint32_t dstStride)
{
    TransposePlane16Strides(srcTile, dstTile, kMicro, dstStride);
}

__aicore__ inline __attribute__((always_inline)) void TransposePlane16(__ubuf__ float* src, __ubuf__ float* dst)
{
    TransposePlane16Stride(src, dst, kMicro);
}

__aicore__ inline __attribute__((always_inline)) void ScaleComplexPlanes(
    const ChemmMixedTilingData& tiling, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* scaledReal,
    __ubuf__ float* scaledImag)
{
    asc_mul_scalar(scaledReal, real, tiling.alphaReal, kVectorRepeats, 1U, 1U, 8U, 8U);
    asc_mul_scalar(scaledImag, imag, tiling.alphaReal, kVectorRepeats, 1U, 1U, 8U, 8U);
    asc_sync_pipe(PIPE_V);
    asc_axpy(scaledReal, imag, -tiling.alphaImag, kVectorRepeats, 1U, 1U, 8U, 8U);
    asc_axpy(scaledImag, real, tiling.alphaImag, kVectorRepeats, 1U, 1U, 8U, 8U);
    // The packed panel consumes both vector results through UB.
    asc_sync_pipe(PIPE_V);
}

template <uint32_t kBank>
__aicore__ inline __attribute__((always_inline)) void WaitComplexTile()
{
    if constexpr (kBank == 0U) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    } else if constexpr (kBank == 1U) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    } else if constexpr (kBank == 2U) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    } else {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    }
}

template <uint32_t kBank>
__aicore__ inline __attribute__((always_inline)) void ReleaseComplexTile()
{
    if constexpr (kBank == 0U) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    } else if constexpr (kBank == 1U) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    } else if constexpr (kBank == 2U) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    } else {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    }
}

template <uint32_t kBank>
__aicore__ inline __attribute__((always_inline)) void CommitComplexTile()
{
    if constexpr (kBank == 0U) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    } else if constexpr (kBank == 1U) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    } else if constexpr (kBank == 2U) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    }
}

// Complex tile loads and Hermitian decoding.
template <bool kTokenized>
__aicore__ inline __attribute__((always_inline)) void CopyFullComplexTile(
    __gm__ float* source, uint32_t rowBase, uint32_t colBase, uint32_t ld, __ubuf__ float* complexTile)
{
    if constexpr (!kTokenized) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    }
    const uint64_t offset = ElementIndex(rowBase, colBase, ld) * 2U;
    const uint32_t gmGap = (ld - kMicro) * 2U * sizeof(float);
    copy_gm_to_ubuf_align_b32(
        complexTile, source + offset, 0U, kMicro, kComplexColumnFloats * sizeof(float), 0U, 0U, gmGap, 0U);
}

__aicore__ inline __attribute__((always_inline)) void LoadStridedComplexTile(
    __gm__ float* source, uint32_t rowBase, uint32_t colBase, uint32_t ld, uint32_t validRows, uint32_t validCols,
    __ubuf__ float* complexTile)
{
    const uint32_t floats = validRows * 2U;
    const uint32_t alignedFloats = (floats + 7U) & ~7U;
    const uint64_t offset = ElementIndex(rowBase, colBase, ld) * 2U;
    asc_copy_gm2ub_align(
        complexTile, source + offset, static_cast<uint16_t>(validCols), floats * sizeof(float), 0U,
        static_cast<uint8_t>(alignedFloats - floats), (2U * ld - floats) * sizeof(float),
        (kComplexColumnFloats - alignedFloats) / 8U);
}

template <bool kFull, uint32_t kBank, bool kTokenized = true>
__aicore__ inline __attribute__((always_inline)) void LoadComplexTile16(
    __gm__ float* source, uint32_t rowBase, uint32_t colBase, uint32_t ld, uint32_t rows, uint32_t cols,
    __ubuf__ float* complexTile)
{
    // Do not collapse this into an immediate Set/Wait pair. Each token owns one
    // complex UB bank from its last V consumer until the following MTE2 write.
    if constexpr (kTokenized) {
        WaitComplexTile<kBank>();
    }
    if constexpr (kFull) {
        CopyFullComplexTile<kTokenized>(source, rowBase, colBase, ld, complexTile);
        return;
    }
    const uint32_t validRows = rowBase < rows ? (rows - rowBase < kMicro ? rows - rowBase : kMicro) : 0U;
    const uint32_t validCols = colBase < cols ? (cols - colBase < kMicro ? cols - colBase : kMicro) : 0U;
    // A complete runtime tile bypasses zero-fill so V never writes a UB bank
    // concurrently with the following MTE2 transfer.
    if (validRows == kMicro && validCols == kMicro && (ld & 3U) == 0U) {
        CopyFullComplexTile<kTokenized>(source, rowBase, colBase, ld, complexTile);
        return;
    }
    asc_duplicate(complexTile, 0.0f, 2U * kVectorRepeats, 1U, 8U);
    if constexpr (kTokenized) {
        ReleaseComplexTile<kBank>();
        WaitComplexTile<kBank>();
    }
    // Padded Cube feeds legitimately request an entirely dummy micro-tile.
    // Keep the vector-produced zeros; subtracting past the matrix boundary
    // would underflow and turn this into an out-of-range GM DMA.
    if (validRows == 0U || validCols == 0U) {
        return;
    }
    if constexpr (!kTokenized) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    }
    LoadStridedComplexTile(source, rowBase, colBase, ld, validRows, validCols, complexTile);
}

template <bool kFull>
__aicore__ inline __attribute__((always_inline)) void LoadOutputTile16(
    __gm__ float* source, uint32_t rowBase, uint32_t colBase, uint32_t ld, uint32_t rows, uint32_t cols,
    __ubuf__ float* complexTile)
{
    if constexpr (!kFull) {
        asc_duplicate(complexTile, 0.0f, 2U * kVectorRepeats, 1U, 8U);
    }
    // The output tile is single-buffered; retire its previous V consumer before
    // MTE2 overwrites it with the old C tile.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    if constexpr (kFull) {
        const uint64_t offset = ElementIndex(rowBase, colBase, ld) * 2U;
        const uint32_t gmGap = (ld - kMicro) * 2U * sizeof(float);
        copy_gm_to_ubuf_align_b32(
            complexTile, source + offset, 0U, kMicro, kComplexColumnFloats * sizeof(float), 0U, 0U, gmGap, 0U);
        return;
    }
    const uint32_t validRows = rowBase < rows ? (rows - rowBase < kMicro ? rows - rowBase : kMicro) : 0U;
    const uint32_t validCols = colBase < cols ? (cols - colBase < kMicro ? cols - colBase : kMicro) : 0U;
    if (validRows == 0U || validCols == 0U) {
        return;
    }
    if (validRows == kMicro && validCols == kMicro && (ld & 3U) == 0U) {
        const uint64_t offset = ElementIndex(rowBase, colBase, ld) * 2U;
        const uint32_t gmGap = (ld - kMicro) * 2U * sizeof(float);
        copy_gm_to_ubuf_align_b32(
            complexTile, source + offset, 0U, kMicro, kComplexColumnFloats * sizeof(float), 0U, 0U, gmGap, 0U);
        return;
    }
    LoadStridedComplexTile(source, rowBase, colBase, ld, validRows, validCols, complexTile);
}

template <bool kFull>
__aicore__ inline __attribute__((always_inline)) void LoadOutputPanel(
    __gm__ float* source, uint32_t rowBase, uint32_t colBase, uint32_t ld, uint32_t validRows, uint32_t validCols,
    __ubuf__ float* stage)
{
    const uint64_t offset = ElementIndex(rowBase, colBase, ld) * 2U;
    if constexpr (kFull) {
        copy_gm_to_ubuf_align_b32(
            stage, source + offset, 0U, kStagePanelCols, CHEMM_TILE_M * 2U * sizeof(float), 0U, 0U,
            (ld - CHEMM_TILE_M) * 2U * sizeof(float), 0U);
        return;
    }
    const uint32_t floats = validRows * 2U;
    const uint32_t alignedFloats = (floats + 7U) & ~7U;
    asc_copy_gm2ub_align(
        stage, source + offset, static_cast<uint16_t>(validCols), floats * sizeof(float), 0U,
        static_cast<uint8_t>(alignedFloats - floats), (2U * ld - floats) * sizeof(float),
        (CHEMM_TILE_M * 2U - alignedFloats) / 8U);
}

template <bool kFull, event_t kVToMte2Event>
__aicore__ inline __attribute__((always_inline)) void LoadAivComplexTile(
    __gm__ float* source, uint32_t ld, uint32_t rows, uint32_t cols, __ubuf__ float* complexTile)
{
    if constexpr (!kFull) {
        asc_duplicate(complexTile, 0.0f, 2U * kVectorRepeats, 1U, 8U);
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
    asc_copy_gm2ub_align(
        complexTile, source, static_cast<uint16_t>(cols), floats * sizeof(float), 0U,
        static_cast<uint8_t>(alignedFloats - floats), (2U * ld - floats) * sizeof(float),
        (kComplexColumnFloats - alignedFloats) / 8U);
}

template <uint32_t kBank, bool kRelease>
__aicore__ inline __attribute__((always_inline)) void ReleaseDecodedTile()
{
    if constexpr (kRelease) {
        ReleaseComplexTile<kBank>();
    }
}

template <uint32_t kBank, bool kRelease>
__aicore__ inline __attribute__((always_inline)) void DecodeReflectedHermitian(
    __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal,
    __ubuf__ float* transImag)
{
    CInterleavedToPlanes(complexTile, transReal, transImag);
    ReleaseDecodedTile<kBank, kRelease>();
    asc_sync_pipe(PIPE_V);
    TransposePlane16(transReal, real);
    TransposePlane16(transImag, imag);
    asc_sync_pipe(PIPE_V);
    asc_mul_scalar(imag, imag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
}

__aicore__ inline __attribute__((always_inline)) void ApplyHermitianMasks(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    __ubuf__ uint32_t maskAddress[32] __attribute__((aligned(32)));
    const uint32_t triangleAddress =
        static_cast<uint32_t>(reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(triangleMask)));
    asc_set_vector_mask(0ULL, kMaskAddressVectorMask);
    asc_duplicate(maskAddress, triangleAddress, 1U, 1U, 8U);
    asc_sync_pipe(PIPE_V);
    asc_set_cmp_mask(maskAddress);
    asc_sync_pipe(PIPE_V);
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    vsel(real, real, transReal, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U, 2U);
    vsel(imag, imag, transImag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_duplicate(transReal, 0.0f, kVectorRepeats, 1U, 8U);
    const uint32_t diagonalAddress =
        static_cast<uint32_t>(reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(diagonalMask)));
    asc_set_vector_mask(0ULL, kMaskAddressVectorMask);
    asc_duplicate(maskAddress, diagonalAddress, 1U, 1U, 8U);
    asc_sync_pipe(PIPE_V);
    asc_set_cmp_mask(maskAddress);
    asc_sync_pipe(PIPE_V);
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    vsel(imag, transReal, imag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U, 2U);
    asc_set_mask_norm();
    asc_set_vector_mask(0ULL, kFullFp32Mask);
}

__aicore__ inline __attribute__((always_inline)) void SelectHermitianTriangle(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    TransposePlane16(real, transReal);
    TransposePlane16(imag, transImag);
    asc_sync_pipe(PIPE_V);
    asc_mul_scalar(transImag, transImag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
    ApplyHermitianMasks(real, imag, transReal, transImag, triangleMask, diagonalMask);
}

template <bool kLower, uint32_t kBank, bool kRelease, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void DecodeHermitianPlanes16(
    const Tiling&, uint32_t rowBase, uint32_t colBase, __ubuf__ float* complexTile, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const bool direct = kLower ? rowBase > colBase : rowBase < colBase;
    const bool reflected = kLower ? rowBase < colBase : rowBase > colBase;
    if (direct) {
        CInterleavedToPlanes(complexTile, real, imag);
        ReleaseDecodedTile<kBank, kRelease>();
        return;
    }
    asc_set_mask_norm();
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    if (reflected) {
        DecodeReflectedHermitian<kBank, kRelease>(complexTile, real, imag, transReal, transImag);
        return;
    }
    CInterleavedToPlanes(complexTile, real, imag);
    ReleaseDecodedTile<kBank, kRelease>();
    asc_sync_pipe(PIPE_V);
    SelectHermitianTriangle(real, imag, transReal, transImag, triangleMask, diagonalMask);
}

template <bool kLower, bool kFull, uint32_t kBank, bool kTokenized = true, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void LoadHermitianPlanes16(
    __gm__ float* a, __gm__ uint8_t* maskGm, const Tiling& tiling, uint32_t rowBase, uint32_t colBase,
    __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal,
    __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    const bool direct = kLower ? rowBase > colBase : rowBase < colBase;
    const bool reflected = kLower ? rowBase < colBase : rowBase > colBase;
    const uint32_t sourceRow = reflected ? colBase : rowBase;
    const uint32_t sourceCol = reflected ? rowBase : colBase;
    LoadComplexTile16<kFull, kBank, kTokenized>(
        a, sourceRow, sourceCol, static_cast<uint32_t>(tiling.lda), static_cast<uint32_t>(tiling.dimA),
        static_cast<uint32_t>(tiling.dimA), complexTile);
    if (!direct && !reflected) {
        constexpr uint64_t triangleOffset = kLower ? 32U : 0U;
        asc_copy_gm2ub_align(triangleMask, maskGm + triangleOffset, 32U);
        asc_copy_gm2ub_align(diagonalMask, maskGm + 64U, 32U);
    }
    CommitComplexTile<kBank>();
    DecodeHermitianPlanes16<kLower, kBank, kTokenized>(
        tiling, rowBase, colBase, complexTile, real, imag, transReal, transImag, triangleMask, diagonalMask);
}

template <bool kRight, bool kLower, bool kFull, uint32_t kBank, bool kTokenized = true, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void LoadYPlanes(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const Tiling& tiling, uint32_t row, uint32_t col,
    __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal,
    __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    if constexpr (!kRight) {
        LoadComplexTile16<kFull, kBank, kTokenized>(
            b, row, col, static_cast<uint32_t>(tiling.ldb), static_cast<uint32_t>(tiling.m),
            static_cast<uint32_t>(tiling.n), complexTile);
        CommitComplexTile<kBank>();
        CInterleavedToPlanes(complexTile, real, imag);
        if constexpr (kTokenized) {
            ReleaseComplexTile<kBank>();
        }
    } else {
        LoadHermitianPlanes16<kLower, kFull, kBank, kTokenized>(
            a, maskGm, tiling, row, col, complexTile, real, imag, transReal, transImag, triangleMask, diagonalMask);
    }
}

template <bool kRight, bool kLower, bool kFull, uint32_t kBank, bool kTokenized = true, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void LoadXPlanes(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const Tiling& tiling, uint32_t row, uint32_t col,
    __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal,
    __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    if constexpr (!kRight) {
        LoadHermitianPlanes16<kLower, kFull, kBank, kTokenized>(
            a, maskGm, tiling, row, col, complexTile, real, imag, transReal, transImag, triangleMask, diagonalMask);
    } else {
        LoadComplexTile16<kFull, kBank, kTokenized>(
            b, row, col, static_cast<uint32_t>(tiling.ldb), static_cast<uint32_t>(tiling.m),
            static_cast<uint32_t>(tiling.n), complexTile);
        CommitComplexTile<kBank>();
        CInterleavedToPlanes(complexTile, real, imag);
        if constexpr (kTokenized) {
            ReleaseComplexTile<kBank>();
        }
    }
}

// Grid tile preparation and planar packing.
template <bool kFull>
__aicore__ inline __attribute__((always_inline)) void PrepareGridTile(
    uint32_t row, uint32_t col, uint32_t ld, uint32_t rows, uint32_t cols, __ubuf__ float* raw)
{
    if constexpr (!kFull) {
        const uint32_t validRows = row < rows ? (rows - row < kMicro ? rows - row : kMicro) : 0U;
        const uint32_t validCols = col < cols ? (cols - col < kMicro ? cols - col : kMicro) : 0U;
        if (validRows != kMicro || validCols != kMicro || (ld & 3U) != 0U) {
            asc_duplicate(raw, 0.0f, 2U * kVectorRepeats, 1U, 8U);
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
    asc_copy_gm2ub_align(
        raw, source + 2U * ElementIndex(row, col, ld), static_cast<uint16_t>(validCols), floats * sizeof(float), 0U,
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

template <uint32_t kBank>
__aicore__ inline __attribute__((always_inline)) void SubmitGridPair()
{
    if constexpr (kBank == 0U) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    } else {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    }
}

template <uint32_t kBank>
__aicore__ inline __attribute__((always_inline)) void CommitGridPair()
{
    if constexpr (kBank == 0U) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    } else {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    }
}

template <uint32_t kBank>
__aicore__ inline __attribute__((always_inline)) void WaitGridPair()
{
    if constexpr (kBank == 0U) {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    } else {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    }
}

template <bool kRight, bool kLower, uint32_t kBank, bool kSkipY, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void DecodeGridPair(
    const Tiling& tiling, uint32_t xRow, uint32_t xCol, uint32_t yRow, uint32_t yCol, __ubuf__ float* xRaw,
    __ubuf__ float* yRaw, __ubuf__ float* leftReal, __ubuf__ float* leftImag, __ubuf__ float* rightReal,
    __ubuf__ float* rightImag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    if constexpr (kRight) {
        CInterleavedToPlanes(xRaw, leftReal, leftImag);
        if constexpr (!kSkipY) {
            DecodeHermitianPlanes16<kLower, kBank, false>(
                tiling, yRow, yCol, yRaw, rightReal, rightImag, transReal, transImag, triangleMask, diagonalMask);
        }
    } else {
        DecodeHermitianPlanes16<kLower, kBank, false>(
            tiling, xRow, xCol, xRaw, leftReal, leftImag, transReal, transImag, triangleMask, diagonalMask);
        if constexpr (!kSkipY) {
            CInterleavedToPlanes(yRaw, rightReal, rightImag);
        }
    }
}

__aicore__ inline __attribute__((always_inline)) void PackQPlanes(
    __ubuf__ float* sourceReal, __ubuf__ float* sourceImag, uint32_t localRow, uint32_t localK, __ubuf__ float* q)
{
    __ubuf__ float* top = q + 2U * localRow * CHEMM_EXP_K;
    __ubuf__ float* bottom = top + CHEMM_EXP_K;
    constexpr uint16_t kDstGap = (2U * CHEMM_EXP_K - kMicro) / (32U / sizeof(float));
    asc_copy_ub2ub(top + localK, sourceReal, kMicro, 2U, 0U, kDstGap);
    asc_copy_ub2ub(bottom + CHEMM_K_FEED + localK, sourceReal, kMicro, 2U, 0U, kDstGap);
    asc_copy_ub2ub(bottom + localK, sourceImag, kMicro, 2U, 0U, kDstGap);
    asc_mul_scalar(
        top + CHEMM_K_FEED + localK, sourceImag, -1.0f, 2U, 2U * CHEMM_EXP_K / 8U, kMicro / 8U, 2U * CHEMM_EXP_K,
        kMicro);
    asc_mul_scalar(
        top + CHEMM_K_FEED + localK + 8U, sourceImag + 8U, -1.0f, 2U, 2U * CHEMM_EXP_K / 8U, kMicro / 8U,
        2U * CHEMM_EXP_K, kMicro);
}

__aicore__ inline __attribute__((always_inline)) void PlaceReflectedQ(
    __ubuf__ float* real, __ubuf__ float* imag, uint32_t localRow, uint32_t localK, __ubuf__ float* q)
{
    __ubuf__ float* top = q + 2U * localRow * CHEMM_EXP_K;
    // The reflected source is already conjugate-transposed by where these
    // two transposes land: +Im becomes the expanded -Im plane, while the
    // bottom plane receives its negation.
    TransposePlane16Strides(real, top + localK, kMicro, 2U * CHEMM_EXP_K);
    TransposePlane16Strides(imag, top + CHEMM_K_FEED + localK, kMicro, 2U * CHEMM_EXP_K);
}

__aicore__ inline __attribute__((always_inline)) void FinishReflectedQ(
    uint32_t localRow, uint32_t localK, __ubuf__ float* q)
{
    __ubuf__ float* top = q + 2U * localRow * CHEMM_EXP_K;
    __ubuf__ float* bottom = top + CHEMM_EXP_K;
    constexpr uint16_t kQGap = (2U * CHEMM_EXP_K - kMicro) / (32U / sizeof(float));
    constexpr uint16_t kQBlockStride = 2U * CHEMM_EXP_K / 8U;
    constexpr uint16_t kQRepeatStride = 2U * CHEMM_EXP_K;
    asc_copy_ub2ub(bottom + CHEMM_K_FEED + localK, top + localK, kMicro, 2U, kQGap, kQGap);
    asc_mul_scalar(
        bottom + localK, top + CHEMM_K_FEED + localK, -1.0f, 2U, kQBlockStride, kQBlockStride, kQRepeatStride,
        kQRepeatStride);
    asc_mul_scalar(
        bottom + localK + 8U, top + CHEMM_K_FEED + localK + 8U, -1.0f, 2U, kQBlockStride, kQBlockStride, kQRepeatStride,
        kQRepeatStride);
}

// Narrow-panel packing.
template <bool kOnePanel>
__aicore__ inline uint32_t PanelHalfRowBase(uint32_t half, uint32_t panelM);

template <bool kOnePanel>
__aicore__ inline uint32_t PanelHalfRows(uint32_t half, uint32_t panelM);

template <bool kLower, uint32_t kBank, uint32_t kPanelM>
__aicore__ inline __attribute__((always_inline)) void PackNarrowPBlock(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const ChemmMixedTilingData& tiling, uint32_t col,
    uint32_t localN, __ubuf__ float* p, __ubuf__ float* complexTiles, __ubuf__ float* real, __ubuf__ float* imag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    constexpr uint32_t localK = kBank * kMicro;
    __ubuf__ float* complexTile = complexTiles + kBank * kComplexTileFloats;
    LoadYPlanes<true, kLower, false, kBank>(
        a, b, maskGm, tiling, localK, col + localN, complexTile, real, imag, transReal, transImag, triangleMask,
        diagonalMask);
    asc_sync_pipe(PIPE_V);
    if constexpr (kBank == 0U) {
        TransposePlane16Stride(real, p + localN * CHEMM_NARROW_EXP_K, CHEMM_NARROW_EXP_K);
        TransposePlane16Stride(imag, p + localN * CHEMM_NARROW_EXP_K + CHEMM_NARROW_K, CHEMM_NARROW_EXP_K);
    } else {
        TransposePlane16(real, transReal);
        TransposePlane16(imag, transImag);
        asc_sync_pipe(PIPE_V);
        constexpr uint16_t kDstGap = CHEMM_NARROW_EXP_K / 8U - 1U;
        asc_copy_ub2ub(p + localN * CHEMM_NARROW_EXP_K + localK, transReal, kMicro, 1U, 1U, kDstGap);
        asc_copy_ub2ub(p + localN * CHEMM_NARROW_EXP_K + CHEMM_NARROW_K + localK, transImag, kMicro, 1U, 1U, kDstGap);
    }
}

template <uint32_t kBegin, uint32_t kPaddedN>
__aicore__ inline __attribute__((always_inline)) void LoadNarrowStageB(
    __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t rowBegin, uint32_t rowEnd, __ubuf__ float* raw)
{
    static_assert(kBegin == 0U || kBegin == 8U, "staged narrow B load owns K0:8 or K8:24");
    constexpr uint32_t kRows = kBegin == 0U ? 8U : 16U;
    const uint32_t rows = rowEnd - rowBegin;
    const uint32_t burstBytes = 2U * rows * sizeof(float);
    const uint32_t alignedBytes = (burstBytes + 31U) & ~31U;
    const uint32_t gmGap = 2U * (static_cast<uint32_t>(tiling.ldb) - rows) * sizeof(float);
    const uint32_t ubGap = (kPaddedN * sizeof(float) - alignedBytes) / 32U;
    __gm__ float* src = b + 2U * ElementIndex(rowBegin, kBegin, static_cast<uint32_t>(tiling.ldb));
    copy_gm_to_ubuf_align_b32(raw, src, 0U, static_cast<uint16_t>(kRows), burstBytes, 0U, 0U, gmGap, ubGap);
}

template <uint32_t kPaddedN>
__aicore__ inline __attribute__((always_inline)) void StoreNarrowStageB(
    __gm__ float* dst, __ubuf__ float* lowRaw, __ubuf__ float* highRaw, __ubuf__ float* pack)
{
    static_assert(kPaddedN % kMicro == 0U, "staged narrow B pack uses complete N16 fractals");
    constexpr uint8_t kRepeats = static_cast<uint8_t>(kPaddedN / 8U);
    __ubuf__ float* srcList[16];
    __ubuf__ float* dstList[16];

    // The first physical K16 group combines untouched raw K16:24 with
    // J(B) K0:8.  b32 transdata emits one K8 DataBlock from each half of
    // the address list, so independent destination permutations preserve
    // the raw lanes and exchange each signed real/imag pair for J(B).
    asc_set_vector_mask(0ULL, kOddFp32Mask);
    asc_mul_scalar(lowRaw, lowRaw, -1.0f, 8U * kPaddedN / 64U, 1U, 1U, 8U, 8U);
    asc_sync_pipe(PIPE_V);
#pragma unroll
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        srcList[lane] = highRaw + (8U + lane) * kPaddedN;
        srcList[8U + lane] = lowRaw + lane * kPaddedN;
        dstList[2U * lane] = pack + lane * 8U;
        dstList[2U * lane + 1U] = pack + 8U * kPaddedN + (lane ^ 1U) * 8U;
    }
    asc_set_va_reg(VA0, dstList);
    asc_set_va_reg(VA1, dstList + 8U);
    asc_set_va_reg(VA2, srcList);
    asc_set_va_reg(VA3, srcList + 8U);
    asc_transto5hd_b32(VA0, VA2, kRepeats, 8U, 1U);

    // Retire the group1 read before highRaw is changed in place.  Its first
    // MTE3 reads a disjoint half of pack while V produces group2.
    asc_sync_pipe(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm(dst, pack, 16U * kPaddedN * sizeof(float));

    asc_mul_scalar(highRaw, highRaw, -1.0f, 16U * kPaddedN / 64U, 1U, 1U, 8U, 8U);
    asc_sync_pipe(PIPE_V);
#pragma unroll
    for (uint32_t row = 0U; row < 16U; ++row) {
        srcList[row] = highRaw + row * kPaddedN;
    }
#pragma unroll
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        dstList[2U * lane] = pack + 16U * kPaddedN + (lane ^ 1U) * 8U;
        dstList[2U * lane + 1U] = pack + 24U * kPaddedN + (lane ^ 1U) * 8U;
    }
    asc_set_va_reg(VA0, dstList);
    asc_set_va_reg(VA1, dstList + 8U);
    asc_set_va_reg(VA2, srcList);
    asc_set_va_reg(VA3, srcList + 8U);
    asc_transto5hd_b32(VA0, VA2, kRepeats, 8U, 1U);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_copy_ub2gm(dst + 16U * kPaddedN, pack + 16U * kPaddedN, 16U * kPaddedN * sizeof(float));
}

template <bool kRight, bool kLower, bool kFull, bool kGeneral, uint32_t kBank, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void PackPBlock(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const Tiling& tiling, uint32_t col, uint32_t kBase,
    uint32_t localN, __ubuf__ float* p, __ubuf__ float* complexTiles, __ubuf__ float* real, __ubuf__ float* imag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    constexpr uint32_t localK = kBank * kMicro;
    __ubuf__ float* complexTile = complexTiles + kBank * kComplexTileFloats;
    LoadYPlanes<kRight, kLower, kFull, kBank>(
        a, b, maskGm, tiling, kBase + localK, col + localN, complexTile, real, imag, transReal, transImag, triangleMask,
        diagonalMask);
    asc_sync_pipe(PIPE_V);
    // General coefficients scale P. Both MMAD operands must retain the same
    // 16-complex cyclic K order to preserve FP32 cancellation accuracy.
    __ubuf__ float* sourceReal = real;
    __ubuf__ float* sourceImag = imag;
    if constexpr (kGeneral) {
        ScaleComplexPlanes(tiling, real, imag, transReal, transImag);
        sourceReal = transReal;
        sourceImag = transImag;
    }
    TransposePlane16Stride(sourceReal, p + localN * CHEMM_EXP_K + localK, CHEMM_EXP_K);
    TransposePlane16Stride(sourceImag, p + localN * CHEMM_EXP_K + CHEMM_K_FEED + localK, CHEMM_EXP_K);
}

template <bool kOnePanel>
__aicore__ inline uint32_t PanelHalfRowBase(uint32_t half, uint32_t panelM)
{
    // Split at a 16-row Cube fractal. A single 16-row panel is assigned to
    // AIV1, which is otherwise the shorter producer in the one-panel path.
    const uint32_t firstRows = kOnePanel ? (panelM / (2U * kMicro)) * kMicro : CHEMM_EXP_M / 2U;
    return half * firstRows;
}

template <bool kOnePanel>
__aicore__ inline uint32_t PanelHalfRows(uint32_t half, uint32_t panelM)
{
    const uint32_t firstRows = kOnePanel ? (panelM / (2U * kMicro)) * kMicro : CHEMM_EXP_M / 2U;
    return half == 0U ? firstRows : panelM - firstRows;
}

template <bool kRight, bool kLower, bool kFull, bool kGeneral, bool kOnePanel = false>
__aicore__ inline void PackPHalf(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const ChemmMixedTilingData& tiling, uint32_t half,
    uint32_t colBase, uint32_t kBase, uint32_t panelM, __ubuf__ float* p, __ubuf__ float* complexTiles,
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t rowBaseInPanel = PanelHalfRowBase<kOnePanel>(half, panelM);
    const uint32_t rows = PanelHalfRows<kOnePanel>(half, panelM);
    const uint32_t col = colBase + rowBaseInPanel;
    for (uint32_t localN = 0U; localN < rows; localN += kMicro) {
        PackPBlock<kRight, kLower, kFull, kGeneral, 0U>(
            a, b, maskGm, tiling, col, kBase, localN, p, complexTiles, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
        PackPBlock<kRight, kLower, kFull, kGeneral, 1U>(
            a, b, maskGm, tiling, col, kBase, localN, p, complexTiles, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
    }
}

template <bool kRight, bool kLower, bool kFull, bool kGeneral, uint32_t kBank, typename Tiling>
__aicore__ inline __attribute__((always_inline)) void PackQBlock(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const Tiling& tiling, uint32_t srcRow, uint32_t kBase,
    uint32_t dstRow, __ubuf__ float* q, __ubuf__ float* complexTiles, __ubuf__ float* real, __ubuf__ float* imag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    constexpr uint32_t localK = kBank * kMicro;
    __ubuf__ float* complexTile = complexTiles + kBank * kComplexTileFloats;
    LoadXPlanes<kRight, kLower, kFull, kBank>(
        a, b, maskGm, tiling, srcRow, kBase + localK, complexTile, real, imag, transReal, transImag, triangleMask,
        diagonalMask);
    asc_sync_pipe(PIPE_V);
    // Alpha is fused into P for both SIDE variants.  Q consequently remains
    // byte-for-byte on the established cyclic K order.
    PackQPlanes(real, imag, dstRow, localK, q);
}

template <bool kRight, bool kLower, bool kFull, bool kGeneral, uint32_t kTileM = CHEMM_TILE_M>
__aicore__ inline void PackQHalf(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const ChemmMixedTilingData& tiling, uint32_t half,
    uint32_t rowBase, uint32_t kBase, __ubuf__ float* q, __ubuf__ float* complexTiles, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    static_assert(
        kTileM >= 32U && kTileM <= CHEMM_MAX_MIX_TILE_M && kTileM % kMicro == 0U,
        "mixed row tile must be 16-row aligned");
    constexpr uint32_t halfRows = kTileM / 2U;
    constexpr uint32_t tasks = (halfRows + kMicro - 1U) / kMicro;
    for (uint32_t task = 0U; task < tasks; ++task) {
        const uint32_t dstRow = task * kMicro;
        const uint32_t srcRow = rowBase + half * halfRows + dstRow;
        PackQBlock<kRight, kLower, kFull, kGeneral, 0U>(
            a, b, maskGm, tiling, srcRow, kBase, dstRow, q, complexTiles, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
        PackQBlock<kRight, kLower, kFull, kGeneral, 1U>(
            a, b, maskGm, tiling, srcRow, kBase, dstRow, q, complexTiles, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
    }
}

// Bulk operand packing and workspace publication.
__aicore__ inline __attribute__((always_inline)) void PackReflectedBulkQ(
    __ubuf__ float* raw, __ubuf__ float* q, __ubuf__ float* real, __ubuf__ float* imag)
{
    constexpr uint32_t block = CHEMM_K_FEED;
    constexpr uint32_t rawStride = 2U * block;
    for (uint32_t localRow = 0U; localRow < block; localRow += kMicro) {
        const uint32_t strip = localRow / kMicro;
        CInterleavedToPlanesStride(
            raw + localRow * rawStride, real + 2U * strip * kMicroElements, imag + 2U * strip * kMicroElements, kMicro,
            rawStride, 8U);
    }
    asc_sync_pipe(PIPE_V);
    for (uint32_t localRow = 0U; localRow < block; localRow += kMicro) {
        const uint32_t strip = localRow / kMicro;
        for (uint32_t bank = 0U; bank < 2U; ++bank) {
            const uint32_t localK = bank * kMicro;
            const uint32_t microBlock = 2U * strip + bank;
            PlaceReflectedQ(
                real + microBlock * kMicroElements, imag + microBlock * kMicroElements, localRow, localK, q);
        }
    }
    asc_sync_pipe(PIPE_V);
    for (uint32_t localRow = 0U; localRow < block; localRow += kMicro) {
        for (uint32_t bank = 0U; bank < 2U; ++bank) {
            FinishReflectedQ(localRow, bank * kMicro, q);
        }
    }
}

__aicore__ inline __attribute__((always_inline)) void PackDirectBulkQ(
    __ubuf__ float* raw, __ubuf__ float* q, __ubuf__ float* real, __ubuf__ float* imag)
{
    constexpr uint32_t block = CHEMM_K_FEED;
    constexpr uint32_t rawStride = 2U * block;
    for (uint32_t bank = 0U; bank < 2U; ++bank) {
        CInterleavedToPlanesStride(
            raw + bank * kMicro * rawStride, real + 2U * bank * kMicroElements, imag + 2U * bank * kMicroElements,
            kMicro, rawStride, 8U);
    }
    asc_sync_pipe(PIPE_V);
    for (uint32_t localRow = 0U; localRow < block; localRow += kMicro) {
        const uint32_t strip = localRow / kMicro;
        for (uint32_t bank = 0U; bank < 2U; ++bank) {
            const uint32_t microBlock = 2U * bank + strip;
            PackQPlanes(
                real + microBlock * kMicroElements, imag + microBlock * kMicroElements, localRow, bank * kMicro, q);
        }
    }
}

template <bool kLower, uint32_t kTileM = CHEMM_TILE_M>
__aicore__ inline void PackBulkHermitianQHalf(
    __gm__ float* a, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t rowBase, uint32_t kBase,
    __ubuf__ float* q, __ubuf__ float* raw, __ubuf__ float* complexTiles, __ubuf__ float* real, __ubuf__ float* imag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    if constexpr (kTileM != CHEMM_TILE_M) {
        __gm__ uint8_t* maskGm = MaskGm(tiling);
        PackQHalf<false, kLower, true, false, kTileM>(
            a, a, maskGm, tiling, half, rowBase, kBase, q, complexTiles, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
        return;
    }
    constexpr uint32_t block = CHEMM_K_FEED;
    constexpr uint32_t rawStride = 2U * block;
    const uint32_t row = rowBase + half * (CHEMM_TILE_M / 2U);
    const bool direct = kLower ? row >= kBase + block : row + block <= kBase;
    const bool reflected = kLower ? row + block <= kBase : row >= kBase + block;
    if (!direct && !reflected) {
        __gm__ uint8_t* maskGm = MaskGm(tiling);
        PackQHalf<false, kLower, true, false>(
            a, a, maskGm, tiling, half, rowBase, kBase, q, complexTiles, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
        return;
    }

    const uint32_t sourceRow = reflected ? kBase : row;
    const uint32_t sourceCol = reflected ? row : kBase;
    const uint64_t offset = 2U * ElementIndex(sourceRow, sourceCol, static_cast<uint32_t>(tiling.lda));
    const uint32_t gmGap = 2U * (static_cast<uint32_t>(tiling.lda) - block) * sizeof(float);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    copy_gm_to_ubuf_align_b32(raw, a + offset, 0U, block, rawStride * sizeof(float), 0U, 0U, gmGap, 0U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);

    if (reflected) {
        PackReflectedBulkQ(raw, q, real, imag);
        return;
    }
    PackDirectBulkQ(raw, q, real, imag);
}

template <uint32_t kTileM = CHEMM_TILE_M>
__aicore__ inline void PackBulkQRightHalf(
    __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t rowBase, uint32_t kBase,
    uint32_t validK, __ubuf__ float* q, __ubuf__ float* raw, __ubuf__ float* real, __ubuf__ float* imag)
{
    constexpr uint32_t block = CHEMM_K_FEED;
    constexpr uint32_t halfRows = kTileM / 2U;
    static_assert(halfRows % kMicro == 0U, "bulk Q requires complete vector rows");
    constexpr uint32_t rawStride = 2U * halfRows;
    const uint32_t row = rowBase + half * halfRows;
    const uint64_t offset = 2U * ElementIndex(row, kBase, static_cast<uint32_t>(tiling.ldb));
    const uint32_t gmGap = 2U * (static_cast<uint32_t>(tiling.ldb) - halfRows) * sizeof(float);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    if (validK < CHEMM_K_FEED) {
        asc_duplicate(raw + validK * rawStride, 0.0f, static_cast<uint8_t>((block - validK) * rawStride / 64U), 1U, 8U);
    }
    copy_gm_to_ubuf_align_b32(
        raw, b + offset, 0U, static_cast<uint16_t>(validK), rawStride * sizeof(float), 0U, 0U, gmGap, 0U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);

    for (uint32_t bank = 0U; bank < 2U; ++bank) {
        CInterleavedToPlanesStride(
            raw + bank * kMicro * rawStride, real + 2U * bank * kMicroElements, imag + 2U * bank * kMicroElements,
            kMicro, rawStride, 8U);
    }
    asc_sync_pipe(PIPE_V);
    for (uint32_t localRow = 0U; localRow < halfRows; localRow += kMicro) {
        const uint32_t strip = localRow / kMicro;
        for (uint32_t bank = 0U; bank < 2U; ++bank) {
            const uint32_t localK = bank * kMicro;
            const uint32_t microBlock = 2U * bank + strip;
            PackQPlanes(real + microBlock * kMicroElements, imag + microBlock * kMicroElements, localRow, localK, q);
        }
    }
}

template <bool kRight, bool kLower, bool kFull, uint32_t kBank>
__aicore__ inline __attribute__((always_inline)) void PackPlanarPBlock(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const ChemmMixedTilingData& tiling, uint32_t col,
    uint32_t kBase, uint32_t localN, __ubuf__ float* p, __ubuf__ float* complexTiles, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ float* alphaReal,
    __ubuf__ float* alphaImag, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    constexpr uint32_t localK = kBank * kMicro;
    __ubuf__ float* complexTile = complexTiles + kBank * kComplexTileFloats;
    LoadYPlanes<kRight, kLower, kFull, kBank>(
        a, b, maskGm, tiling, kBase + localK, col + localN, complexTile, real, imag, transReal, transImag, triangleMask,
        diagonalMask);
    asc_sync_pipe(PIPE_V);
    // Netlib CHEMM multiplies alpha into the streamed operand before the
    // inner product. Fuse the same complex scaling into the reusable P panel
    // and leave the output micro-tiles free of repeated alpha multiplies.
    asc_mul(transReal, real, alphaReal, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    asc_mul(transImag, imag, alphaImag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    asc_mul(real, real, alphaImag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    asc_mul(imag, imag, alphaReal, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    asc_sync_pipe(PIPE_V);
    asc_sub(transReal, transReal, transImag, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    asc_add(imag, imag, real, kVectorRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
    asc_sync_pipe(PIPE_V);
    TransposePlane16Stride(transReal, p + localN * CHEMM_K_FEED + localK, CHEMM_K_FEED);
    TransposePlane16Stride(imag, p + kPlanarPHalfElements + localN * CHEMM_K_FEED + localK, CHEMM_K_FEED);
}

template <bool kRight, bool kLower, bool kFull>
__aicore__ inline void PackPlanarPHalf(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const ChemmMixedTilingData& tiling, uint32_t half,
    uint32_t colBase, uint32_t kBase, __ubuf__ float* p, __ubuf__ float* complexTiles, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ float* alphaReal,
    __ubuf__ float* alphaImag, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t col = colBase + half * (CHEMM_EXP_M / 2U);
    for (uint32_t localN = 0U; localN < CHEMM_EXP_M / 2U; localN += kMicro) {
        PackPlanarPBlock<kRight, kLower, kFull, 0U>(
            a, b, maskGm, tiling, col, kBase, localN, p, complexTiles, real, imag, transReal, transImag, alphaReal,
            alphaImag, triangleMask, diagonalMask);
        PackPlanarPBlock<kRight, kLower, kFull, 1U>(
            a, b, maskGm, tiling, col, kBase, localN, p, complexTiles, real, imag, transReal, transImag, alphaReal,
            alphaImag, triangleMask, diagonalMask);
    }
}

template <bool kRight, bool kLower, bool kFull, uint32_t kBank>
__aicore__ inline __attribute__((always_inline)) void PackPlanarQBlock(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const ChemmMixedTilingData& tiling, uint32_t row,
    uint32_t kBase, uint32_t localRow, __ubuf__ float* q, __ubuf__ float* complexTiles, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    constexpr uint32_t localK = kBank * kMicro;
    __ubuf__ float* qReal = q + localRow * CHEMM_K_FEED + localK;
    __ubuf__ float* qImag = q + kPlanarQHalfElements + localRow * CHEMM_K_FEED + localK;
    __ubuf__ float* complexTile = complexTiles + kBank * kComplexTileFloats;
    if constexpr (kRight) {
        LoadXPlanes<true, kLower, kFull, kBank>(
            a, b, maskGm, tiling, row + localRow, kBase + localK, complexTile, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
    } else {
        LoadHermitianPlanes16<kLower, kFull, kBank>(
            a, maskGm, tiling, row + localRow, kBase + localK, complexTile, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
    }
    // Both producers expose a dense 16x16 plane.  The former implementation
    // transposed it twice; their composition is identity, so one strided
    // UB copy writes the same rows directly into the 32-float Q stride.
    asc_sync_pipe(PIPE_V);
    asc_copy_ub2ub(qReal, real, kMicro, 2U, 0U, (CHEMM_K_FEED - kMicro) / (32U / sizeof(float)));
    asc_copy_ub2ub(qImag, imag, kMicro, 2U, 0U, (CHEMM_K_FEED - kMicro) / (32U / sizeof(float)));
}

template <bool kRight, bool kLower, bool kFull>
__aicore__ inline void PackPlanarQHalf(
    __gm__ float* a, __gm__ float* b, __gm__ uint8_t* maskGm, const ChemmMixedTilingData& tiling, uint32_t half,
    uint32_t rowBase, uint32_t kBase, __ubuf__ float* q, __ubuf__ float* complexTiles, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t row = rowBase + half * (CHEMM_TILE_M / 2U);
    for (uint32_t localRow = 0U; localRow < CHEMM_TILE_M / 2U; localRow += kMicro) {
        PackPlanarQBlock<kRight, kLower, kFull, 0U>(
            a, b, maskGm, tiling, row, kBase, localRow, q, complexTiles, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
        PackPlanarQBlock<kRight, kLower, kFull, 1U>(
            a, b, maskGm, tiling, row, kBase, localRow, q, complexTiles, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
    }
}

template <uint32_t kHalfElements>
__aicore__ inline void StorePlanarHalf(__gm__ float* dst, __ubuf__ float* src, uint32_t half)
{
    asc_copy_ub2gm(dst + half * kHalfElements, src, 2U, kHalfElements / 8U, 0U, kHalfElements / 8U);
}

template <bool kOnePanel = false>
__aicore__ inline void StorePHalf(__gm__ float* dst, __ubuf__ float* src, uint32_t half, uint32_t panelM)
{
    const uint32_t rowBase = PanelHalfRowBase<kOnePanel>(half, panelM);
    const uint32_t rows = PanelHalfRows<kOnePanel>(half, panelM);
    if (rows != 0U) {
        asc_copy_ub2gm(dst + static_cast<uint64_t>(rowBase) * CHEMM_EXP_K, src, rows * CHEMM_EXP_K * sizeof(float));
    }
}

template <uint32_t kTileM = CHEMM_TILE_M>
__aicore__ inline void StoreQHalf(__gm__ float* dst, __ubuf__ float* src, uint32_t half)
{
    constexpr uint32_t halfElements = kTileM * CHEMM_EXP_K;
    asc_copy_ub2gm(dst + half * halfElements, src, halfElements * sizeof(float));
}

template <bool kOnePanel = false, uint32_t kTileM = CHEMM_TILE_M>
__aicore__ inline void ClearInputHalf(
    __ubuf__ float* op0, __ubuf__ float* op1, __ubuf__ float* shared, uint32_t half, uint32_t panelM)
{
    constexpr uint8_t kFullRepeats = static_cast<uint8_t>(kHalfInputElements / 64U);
    const uint32_t rows = PanelHalfRows<kOnePanel>(half, panelM);
    if (rows != 0U) {
        asc_duplicate(op0, 0.0f, static_cast<uint8_t>(rows), 1U, 8U);
    }
    if constexpr (!kOnePanel) {
        asc_duplicate(op1, 0.0f, kFullRepeats, 1U, 8U);
    }
    asc_duplicate(shared, 0.0f, static_cast<uint8_t>(kTileM), 1U, 8U);
}

__aicore__ inline void ClearPlanarInputHalf(__ubuf__ float* p, __ubuf__ float* q)
{
    asc_duplicate(p, 0.0f, static_cast<uint8_t>((2U * kPlanarPHalfElements) / 64U), 1U, 8U);
    asc_duplicate(q, 0.0f, static_cast<uint8_t>((2U * kPlanarQHalfElements) / 64U), 1U, 8U);
}

// Load one complete B half-panel with a single 2-D DMA. Four independent
// plane scratch pairs batch the transpose dependency behind one V barrier.
template <bool kOnePanel = false>
__aicore__ inline void PackBulkPLeftHalf(
    __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t colBase, uint32_t kBase,
    uint32_t panelM, __ubuf__ float* p, __ubuf__ float* raw, __ubuf__ float* real, __ubuf__ float* imag)
{
    constexpr uint32_t rawStride = 2U * CHEMM_K_FEED;
    const uint32_t rowBase = PanelHalfRowBase<kOnePanel>(half, panelM);
    const uint32_t rows = PanelHalfRows<kOnePanel>(half, panelM);
    if (rows == 0U) {
        return;
    }
    const uint32_t col = colBase + rowBase;
    const uint64_t offset = 2U * ElementIndex(kBase, col, static_cast<uint32_t>(tiling.ldb));
    const uint32_t gmGap = 2U * (static_cast<uint32_t>(tiling.ldb) - CHEMM_K_FEED) * sizeof(float);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    copy_gm_to_ubuf_align_b32(
        raw, b + offset, 0U, static_cast<uint16_t>(rows), rawStride * sizeof(float), 0U, 0U, gmGap, 0U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);

    for (uint32_t batch = 0U; batch < rows; batch += 2U * kMicro) {
        uint32_t microBlock = 0U;
        const uint32_t batchEnd = batch + 2U * kMicro < rows ? batch + 2U * kMicro : rows;
        for (uint32_t localN = batch; localN < batchEnd; localN += kMicro) {
            for (uint32_t bank = 0U; bank < 2U; ++bank, ++microBlock) {
                const uint32_t localK = bank * kMicro;
                CInterleavedToPlanesStride(
                    raw + localN * rawStride + 2U * localK, real + microBlock * kMicroElements,
                    imag + microBlock * kMicroElements, kMicro, rawStride);
            }
        }
        asc_sync_pipe(PIPE_V);
        microBlock = 0U;
        for (uint32_t localN = batch; localN < batchEnd; localN += kMicro) {
            for (uint32_t bank = 0U; bank < 2U; ++bank, ++microBlock) {
                const uint32_t localK = bank * kMicro;
                TransposePlane16Stride(
                    real + microBlock * kMicroElements, p + localN * CHEMM_EXP_K + localK, CHEMM_EXP_K);
                TransposePlane16Stride(
                    imag + microBlock * kMicroElements, p + localN * CHEMM_EXP_K + CHEMM_K_FEED + localK, CHEMM_EXP_K);
            }
        }
    }
}

} // namespace
