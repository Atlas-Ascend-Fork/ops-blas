/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chemm (arch22) device kernel translation unit: pulls in the per-strategy
// kernel headers and exposes the launch symbols consumed by the host side.

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "chemm_tiling_data.h"

namespace {

// chemm (arch22) shared device helpers: Hermitian decode (conjugate mirror,
// zeroed diagonal imaginary part, uplo triangle selection), on-chip transpose
// and the complex multiply-accumulate primitives reused by every strategy.

#pragma once

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

// chemm (arch22) mixed strategy: routes between direct / narrow / panel
// sub-kernels according to leading-dim alignment and available core count.

#pragma once

// Operand packing and shared-panel layout.
template <bool kRight, bool kLower, bool kFull, bool kGeneral = false, bool kOnePanel = false>
__aicore__ inline void PackCachedHalf(
    __gm__ float* a, __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t colBase,
    uint32_t kBase, uint32_t panelM, __ubuf__ float* op0, __ubuf__ float* op1, __ubuf__ float* bulk,
    __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal,
    __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    if constexpr (!kRight && kFull && !kGeneral) {
        PackBulkPLeftHalf<kOnePanel>(b, tiling, half, colBase, kBase, panelM, op0, bulk, real, imag);
        if constexpr (!kOnePanel) {
            PackBulkPLeftHalf(b, tiling, half, colBase + CHEMM_TILE_N, kBase, CHEMM_EXP_M, op1, bulk, real, imag);
        }
    } else {
        __gm__ uint8_t* maskGm = MaskGm(tiling);
        PackPHalf<kRight, kLower, kFull, kGeneral, kOnePanel>(
            a, b, maskGm, tiling, half, colBase, kBase, panelM, op0, complexTile, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
        if constexpr (!kOnePanel) {
            PackPHalf<kRight, kLower, kFull, kGeneral>(
                a, b, maskGm, tiling, half, colBase + CHEMM_TILE_N, kBase, CHEMM_EXP_M, op1, complexTile, real, imag,
                transReal, transImag, triangleMask, diagonalMask);
        }
    }
}

template <
    bool kRight, bool kLower, bool kFull, bool kGeneral = false, bool kKeepDummy = false,
    uint32_t kTileM = CHEMM_TILE_M>
__aicore__ inline void PackTileHalf(
    __gm__ float* a, __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t rowBase,
    uint32_t kBase, __ubuf__ float* shared, __ubuf__ float* bulk, __ubuf__ float* complexTile, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    if constexpr (!kRight && kFull && !kGeneral) {
        PackBulkHermitianQHalf<kLower, kTileM>(
            a, tiling, half, rowBase, kBase, shared, bulk, complexTile, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
    } else if constexpr (kRight && (kFull || kKeepDummy) && !kGeneral) {
        if constexpr (kFull) {
            PackBulkQRightHalf<kTileM>(b, tiling, half, rowBase, kBase, CHEMM_K_FEED, shared, bulk, real, imag);
        } else {
            __gm__ uint8_t* maskGm = MaskGm(tiling);
            PackQHalf<kRight, kLower, kFull, kGeneral, kTileM>(
                a, b, maskGm, tiling, half, rowBase, kBase, shared, complexTile, real, imag, transReal, transImag,
                triangleMask, diagonalMask);
        }
    } else {
        __gm__ uint8_t* maskGm = MaskGm(tiling);
        PackQHalf<kRight, kLower, kFull, kGeneral, kTileM>(
            a, b, maskGm, tiling, half, rowBase, kBase, shared, complexTile, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
    }
}

template <bool kRight, bool kLower, bool kFull, bool kSimpleCoeff, bool kOnePanel, bool kKeepDummy, uint32_t kTileM>
__aicore__ inline __attribute__((always_inline)) void PackInputOperands(
    __gm__ float* a, __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t half, uint32_t rowBase,
    uint32_t colBase, uint32_t kBase, uint32_t panelM, bool packCache, __ubuf__ float* op0, __ubuf__ float* op1,
    __ubuf__ float* shared, __ubuf__ float* bulk, __ubuf__ float* complexTile, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    if constexpr (!kFull && !kKeepDummy) {
        ClearInputHalf<kOnePanel, kTileM>(op0, op1, shared, half, panelM);
    }
    const bool active = kSimpleCoeff || tiling.alphaReal != 0.0f || tiling.alphaImag != 0.0f;
    if (active && packCache) {
        PackCachedHalf<kRight, kLower, kFull, !kSimpleCoeff, kOnePanel>(
            a, b, tiling, half, colBase, kBase, panelM, op0, op1, shared, complexTile, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
        if constexpr (kSimpleCoeff) {
            asc_sync_pipe(PIPE_V);
        }
    }
    if (active) {
        PackTileHalf<kRight, kLower, kFull, !kSimpleCoeff, kKeepDummy, kTileM>(
            a, b, tiling, half, rowBase, kBase, shared, bulk, complexTile, real, imag, transReal, transImag,
            triangleMask, diagonalMask);
    } else if constexpr (kFull) {
        ClearInputHalf<false, kTileM>(op0, op1, shared, half, CHEMM_EXP_M);
    }
}

template <bool kOnePanel, uint32_t kTileM>
__aicore__ inline __attribute__((always_inline)) void StorePackedInput(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t half, uint32_t feed,
    uint32_t ringSlot, uint32_t cacheBank, __gm__ float* slot, uint32_t ubBank, uint32_t panelM, bool packCache,
    bool sharedCache, bool syncSharedCache, __ubuf__ float* op0, __ubuf__ float* op1, __ubuf__ float* shared)
{
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    if (ReuseCache(tiling)) {
        StoreQHalf<kTileM>(slot, shared, half);
        if (packCache) {
            const uint32_t cacheCore = sharedCache ? 0U : core;
            StorePHalf<kOnePanel>(CacheSlot(workspace, tiling, cacheCore, cacheBank, feed, 0U), op0, half, panelM);
            if constexpr (!kOnePanel) {
                StorePHalf(CacheSlot(workspace, tiling, cacheCore, cacheBank, feed, 1U), op1, half, CHEMM_EXP_M);
            }
        }
    } else {
        StorePHalf<kOnePanel>(slot, op0, half, panelM);
        if constexpr (!kOnePanel) {
            StorePHalf(slot + CHEMM_INPUT_MATRIX_ELEMENTS, op1, half, CHEMM_EXP_M);
        }
        StoreQHalf<kTileM>(slot + 2U * CHEMM_INPUT_MATRIX_ELEMENTS, shared, half);
    }
    if constexpr (kOnePanel) {
        if (syncSharedCache) {
            ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0U, kSharedCacheReady));
            wait_flag_dev(kSharedCacheReady);
        }
    }
    if (ubBank == 0U) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    CHEMM_NOTIFY(PIPE_MTE3, kInputReady + ringSlot);
}

template <
    bool kRight, bool kLower, bool kFull, bool kSimpleCoeff, bool kOnePanel = false, bool kKeepDummy = false,
    uint32_t kTileM = CHEMM_TILE_M>
__aicore__ inline void PackInputHalf(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t feed, uint32_t ringSlot, uint32_t rowBase, uint32_t colBase, uint32_t cacheBank,
    __gm__ float* slot, uint32_t ubBank, bool packCache, bool sharedCache, bool syncSharedCache,
    __ubuf__ float* inputUb, __ubuf__ float* complexTile, __ubuf__ float* bulk, __ubuf__ float* real,
    __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t panelM = kOnePanel ? tiling.panelM : CHEMM_EXP_M;
    const uint32_t bankElements = AivInputBankElements<kOnePanel, kTileM>(panelM);
    const uint32_t opElements = kOnePanel ? panelM * CHEMM_EXP_K : kHalfInputElements;
    __ubuf__ float* op0 = inputUb + ubBank * bankElements;
    __ubuf__ float* op1 = op0 + opElements;
    __ubuf__ float* shared = kOnePanel ? op1 : op1 + kHalfInputElements;
    const uint32_t kBase = feed * CHEMM_K_FEED;
    PackInputOperands<kRight, kLower, kFull, kSimpleCoeff, kOnePanel, kKeepDummy, kTileM>(
        a, b, tiling, half, rowBase, colBase, kBase, panelM, packCache, op0, op1, shared, bulk, complexTile, real, imag,
        transReal, transImag, triangleMask, diagonalMask);
    StorePackedInput<kOnePanel, kTileM>(
        workspace, tiling, core, half, feed, ringSlot, cacheBank, slot, ubBank, panelM, packCache, sharedCache,
        syncSharedCache, op0, op1, shared);
}

template <bool kRight, bool kLower, bool kFull>
__aicore__ inline void PackPlanarInputHalf(
    __gm__ float* a, __gm__ float* b, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    uint32_t half, uint32_t tile, uint32_t feed, uint32_t ringSlot, uint32_t ubBank, bool packCache,
    __ubuf__ float* inputUb, __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ float* alphaReal, __ubuf__ float* alphaImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    __ubuf__ float* p = inputUb + ubBank * kPlanarInputBankElements;
    __ubuf__ float* q = p + 2U * kPlanarPHalfElements;
    __gm__ float* slot = InputSlot(workspace, tiling, core, ringSlot);
    const uint32_t rowBase = TileRow(tiling, tile) * CHEMM_TILE_M;
    const uint32_t colBase = TileCol(tiling, tile) * CHEMM_TILE_N;
    const uint32_t kBase = feed * CHEMM_K_FEED;
    __gm__ uint8_t* maskGm = MaskGm(tiling);

    if constexpr (!kFull) {
        ClearPlanarInputHalf(p, q);
    }
    if (tiling.alphaReal != 0.0f || tiling.alphaImag != 0.0f) {
        if (packCache) {
            PackPlanarPHalf<kRight, kLower, kFull>(
                a, b, maskGm, tiling, half, colBase, kBase, p, complexTile, real, imag, transReal, transImag, alphaReal,
                alphaImag, triangleMask, diagonalMask);
        }
        PackPlanarQHalf<kRight, kLower, kFull>(
            a, b, maskGm, tiling, half, rowBase, kBase, q, complexTile, real, imag, transReal, transImag, triangleMask,
            diagonalMask);
    } else if constexpr (kFull) {
        ClearPlanarInputHalf(p, q);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    if (ReuseCache(tiling)) {
        StorePlanarHalf<kPlanarQHalfElements>(slot, q, half);
        if (packCache) {
            const uint32_t bank = CacheGroup(tiling, tile) & 1U;
            StorePlanarHalf<kPlanarPHalfElements>(CacheSlot(workspace, tiling, core, bank, feed, 0U), p, half);
        }
    } else {
        StorePlanarHalf<kPlanarPHalfElements>(slot, p, half);
        StorePlanarHalf<kPlanarQHalfElements>(slot + 2U * CHEMM_PLANE_P_ELEMENTS, q, half);
    }
    if (ubBank == 0U) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    CHEMM_NOTIFY(PIPE_MTE3, kInputReady + ringSlot);
}

template <uint32_t kMFractals, uint32_t kKFractals>
__aicore__ inline __attribute__((always_inline)) void LoadL1NzToL0a(__ca__ float* dst, __cbuf__ float* src)
{
    for (uint32_t m1 = 0U; m1 < kMFractals; ++m1) {
        asc_copy_l12l0a(
            dst + m1 * kKFractals * kL1FractalElements, src + m1 * kL1FractalElements, 0U, kKFractals, kMFractals, 0U);
    }
}

template <uint32_t kPanelM = CHEMM_EXP_M>
__aicore__ inline void LoadNzToL0a(__ca__ float* dst, __cbuf__ float* src)
{
    LoadL1NzToL0a<kPanelM / kMicro, kL0aKFractals>(dst, src);
}

// One dynamic one-panel loader serves every panelM selected by the host.
// The NZ source stride is the runtime M-fractal count; the K-fractal repeat
// remains fixed by the FP32 Cube layout.  This removes panelM dispatch from
// the kernel without widening the L0A footprint.
__aicore__ inline void LoadNzToL0aDynamic(__ca__ float* dst, __cbuf__ float* src, uint32_t panelM)
{
    const uint32_t mFractals = panelM / kMicro;
    for (uint32_t m1 = 0U; m1 < mFractals; ++m1) {
        asc_copy_l12l0a(
            dst + m1 * kL0aKFractals * kL1FractalElements, src + m1 * kL1FractalElements, 0U, kL0aKFractals,
            static_cast<uint16_t>(mFractals), 0U);
    }
}

__aicore__ inline void LoadPlanarNzToL0a(__ca__ float* dst, __cbuf__ float* src)
{
    LoadL1NzToL0a<kL0aMFractals, kPlanarL0aKFractals>(dst, src);
}

// AIC feed, MMAD and result pipelines.
__aicore__ __attribute__((always_inline)) inline void InitAicPipeline()
{
    asc_set_fp32_mode();
    asc_set_mmad_direction_n();
    asc_set_l0c_copy_params(1U, 0U, 0U);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
}

template <bool kOnePanel>
__aicore__ __attribute__((always_inline)) inline void WaitAicOutputBank(uint32_t ordinal)
{
    if constexpr (kOnePanel) {
        if (ordinal >= 2U) {
            if ((ordinal & 1U) == 0U) {
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            } else {
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
            }
        }
    } else if (ordinal != 0U) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
}

template <bool kOnePanel>
__aicore__ __attribute__((always_inline)) inline void LoadAicFeedToL1(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t cacheBank, uint32_t feed, uint32_t job,
    uint32_t totalJobs, uint32_t ringSlot, uint32_t sharedBank, uint32_t panelM, bool reuseCache, uint32_t cacheCore,
    __gm__ float* input, __cbuf__ float* pL1, __cbuf__ float* qL1)
{
    CHEMM_WAIT(kInputReady + ringSlot);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    __gm__ float* p0 = reuseCache ? CacheSlot(workspace, tiling, cacheCore, cacheBank, feed, 0U) : input;
    __gm__ float* p1 =
        reuseCache ? CacheSlot(workspace, tiling, cacheCore, cacheBank, feed, 1U) : input + CHEMM_INPUT_MATRIX_ELEMENTS;
    __gm__ float* q = reuseCache ? input : input + 2U * CHEMM_INPUT_MATRIX_ELEMENTS;
    asc_copy_gm2l1_nd2nz(pL1, p0, 1U, panelM, CHEMM_EXP_K, 0U, CHEMM_EXP_K, panelM, 1U, 0U);
    if constexpr (!kOnePanel) {
        asc_copy_gm2l1_nd2nz(
            pL1 + CHEMM_INPUT_MATRIX_ELEMENTS, p1, 1U, CHEMM_EXP_M, CHEMM_EXP_K, 0U, CHEMM_EXP_K, CHEMM_EXP_M, 1U, 0U);
    }
    asc_copy_gm2l1_nd2nz(
        qL1 + sharedBank * CHEMM_INPUT_MATRIX_ELEMENTS, q, 1U, CHEMM_EXP_M, CHEMM_EXP_K, 0U, CHEMM_EXP_K, CHEMM_EXP_M,
        1U, 0U);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    if (job + CHEMM_INPUT_RING < totalJobs) {
        CHEMM_NOTIFY(PIPE_MTE2, kInputFree + ringSlot);
    }
}

template <bool kOnePanel, uint32_t kPanelM>
__aicore__ __attribute__((always_inline)) inline void ComputeAicFeed(
    uint32_t feed, uint32_t sharedBank, uint32_t panelM, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0,
    __cb__ float* qL0, __cc__ float* cTile)
{
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (sharedBank == 0U) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    } else {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    }
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    __cb__ float* qSharedL0 = qL0 + sharedBank * CHEMM_INPUT_MATRIX_ELEMENTS;
    if constexpr (kOnePanel) {
        LoadNzToL0aDynamic(pL0, pL1, panelM);
    } else {
        LoadNzToL0a<kPanelM>(pL0, pL1);
        LoadNzToL0a(pL0 + CHEMM_INPUT_MATRIX_ELEMENTS, pL1 + CHEMM_INPUT_MATRIX_ELEMENTS);
    }
    asc_copy_l12l0b(qSharedL0, qL1 + sharedBank * CHEMM_INPUT_MATRIX_ELEMENTS, 0U, CHEMM_EXP_K, 1U, 0U);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    asc_mmad(cTile, pL0, qSharedL0, panelM, CHEMM_EXP_K, CHEMM_EXP_N, 0U, false, false, feed == 0U);
    if constexpr (!kOnePanel) {
        asc_mmad(
            cTile + CHEMM_PRODUCT_ELEMENTS, pL0 + CHEMM_INPUT_MATRIX_ELEMENTS, qSharedL0, CHEMM_EXP_M, CHEMM_EXP_K,
            CHEMM_EXP_N, 0U, false, false, feed == 0U);
    }
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (sharedBank == 0U) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    } else {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    }
}

__aicore__ __attribute__((always_inline)) inline void AdvanceAicInput(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t& ringSlot,
    __gm__ float*& input)
{
    ++ringSlot;
    if (ringSlot == CHEMM_INPUT_RING) {
        ringSlot = 0U;
        input = workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride;
    } else {
        input += tiling.inputSlotStride;
    }
}

template <bool kOnePanel>
__aicore__ __attribute__((always_inline)) inline void StoreAicDirectTile(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t tile, uint32_t panelM, __cc__ float* cTile)
{
    const uint32_t row = TileRow(tiling, tile) * CHEMM_TILE_M;
    const uint32_t col = TileCol(tiling, tile) * CHEMM_WORK_N;
    const uint32_t validRows =
        static_cast<uint32_t>(tiling.m) - row < CHEMM_TILE_M ? static_cast<uint32_t>(tiling.m) - row : CHEMM_TILE_M;
    const uint32_t validCols0 =
        static_cast<uint32_t>(tiling.n) - col < CHEMM_TILE_N ? static_cast<uint32_t>(tiling.n) - col : CHEMM_TILE_N;
    const uint32_t dstStride = 2U * static_cast<uint32_t>(tiling.ldc);
    asc_copy_l0c2gm(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(tiling.ldc)), cTile, 2U * validRows, validCols0,
        dstStride, panelM, 0U, static_cast<uint64_t>(QuantMode_t::NoQuant), 0U, false, true);
    if constexpr (!kOnePanel) {
        const uint32_t col1 = col + CHEMM_TILE_N;
        if (col1 < static_cast<uint32_t>(tiling.n)) {
            const uint32_t validCols1 = static_cast<uint32_t>(tiling.n) - col1 < CHEMM_TILE_N ?
                                            static_cast<uint32_t>(tiling.n) - col1 :
                                            CHEMM_TILE_N;
            asc_copy_l0c2gm(
                c + 2U * ElementIndex(row, col1, static_cast<uint32_t>(tiling.ldc)), cTile + CHEMM_PRODUCT_ELEMENTS,
                2U * validRows, validCols1, dstStride, CHEMM_EXP_N, 0U, static_cast<uint64_t>(QuantMode_t::NoQuant), 0U,
                false, true);
        }
    }
}

template <bool kDirect, bool kOnePanel>
__aicore__ __attribute__((always_inline)) inline void StoreAicTile(
    __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t tile,
    uint32_t ordinal, uint32_t panelM, __cc__ float* cTile)
{
    const uint32_t productGroup = ordinal & 1U;
    if constexpr (!kDirect) {
        if (ordinal >= CHEMM_PRODUCT_GROUPS) {
            CHEMM_WAIT(kProductFree + productGroup);
        }
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    if constexpr (kDirect) {
        StoreAicDirectTile<kOnePanel>(c, tiling, tile, panelM, cTile);
    } else {
        asc_copy_l0c2gm(
            ProductSlot(workspace, tiling, core, productGroup * CHEMM_PRODUCTS_PER_GROUP), cTile, CHEMM_EXP_N,
            CHEMM_EXP_M, CHEMM_EXP_N, CHEMM_EXP_N, 0U, static_cast<uint64_t>(QuantMode_t::NoQuant), 0U, false, true);
        asc_copy_l0c2gm(
            ProductSlot(workspace, tiling, core, productGroup * CHEMM_PRODUCTS_PER_GROUP + 1U),
            cTile + CHEMM_PRODUCT_ELEMENTS, CHEMM_EXP_N, CHEMM_EXP_M, CHEMM_EXP_N, CHEMM_EXP_N, 0U,
            static_cast<uint64_t>(QuantMode_t::NoQuant), 0U, false, true);
    }
    if constexpr (kOnePanel) {
        if ((ordinal & 1U) == 0U) {
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        } else {
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        }
    } else {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    if constexpr (!kDirect) {
        CHEMM_NOTIFY(PIPE_FIX, kProductReady + productGroup);
    }
}

template <bool kOnePanel>
__aicore__ __attribute__((always_inline)) inline void DrainAicPipeline(uint32_t tileCount)
{
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    if constexpr (kOnePanel) {
        if (tileCount != 0U) {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            if (tileCount > 1U) {
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
            }
        }
    } else {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
}

// chemm (arch22) AIC (cube) compute path.

#pragma once

template <bool kDirect, bool kOnePanel = false, uint32_t kPanelM = CHEMM_EXP_M>
__aicore__ __attribute__((always_inline)) inline void RunAic(
    __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t tileBegin,
    uint32_t tileEnd, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    InitAicPipeline();
    const uint32_t panelM = kOnePanel ? tiling.panelM : kPanelM;
    const uint32_t tileCount = tileEnd - tileBegin;
    const uint32_t totalJobs = tileCount * tiling.kFeeds;
    const bool reuseCache = ReuseCache(tiling);
    const bool sharedCache = kOnePanel && reuseCache && tiling.tileColBase == 0U && tiling.blockDim >= 8U;
    const uint32_t cacheCore = sharedCache ? 0U : core;
    uint32_t job = 0U;
    uint32_t ringSlot = 0U;
    uint32_t sharedBank = 0U;
    __gm__ float* inputCursor = workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride;
    for (uint32_t ordinal = 0U; ordinal < tileCount; ++ordinal) {
        WaitAicOutputBank<kOnePanel>(ordinal);
        __cc__ float* cTile = kOnePanel ? cL0 + (ordinal & 1U) * CHEMM_PRODUCT_ELEMENTS : cL0;
        const uint32_t tile = tileBegin + ordinal;
        const uint32_t cacheBank = CacheGroup(tiling, tile) & 1U;
        for (uint32_t feed = 0U; feed < tiling.kFeeds; ++feed) {
            LoadAicFeedToL1<kOnePanel>(
                workspace, tiling, cacheBank, feed, job, totalJobs, ringSlot, sharedBank, panelM, reuseCache, cacheCore,
                inputCursor, pL1, qL1);
            ComputeAicFeed<kOnePanel, kPanelM>(feed, sharedBank, panelM, pL1, qL1, pL0, qL0, cTile);
            ++job;
            sharedBank ^= 1U;
            AdvanceAicInput(workspace, tiling, core, ringSlot, inputCursor);
        }
        StoreAicTile<kDirect, kOnePanel>(c, workspace, tiling, core, tile, ordinal, panelM, cTile);
    }
    DrainAicPipeline<kOnePanel>(tileCount);
}

// Common one-panel MIX pipeline selected by host geometry. P is invariant
// across output-row tiles: every feed is fetched to L1 once, and a one-feed
// panel remains resident in L0A. Q uses two L1 stages but one L0B stage;
// L0C ping-pong overlaps Cube with the preceding FixPipe write.
__aicore__ __attribute__((always_inline)) inline void InitAicPanelPipeline()
{
    asc_set_fp32_mode();
    asc_set_mmad_direction_n();
    asc_set_l0c_copy_params(1U, 0U, 0U);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
}

__aicore__ __attribute__((always_inline)) inline void WaitAicPanelOutput(uint32_t ordinal, uint32_t bank)
{
    if (ordinal >= 2U) {
        if (bank == 0U) {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        } else {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        }
    }
}

template <uint32_t kTileM>
__aicore__ __attribute__((always_inline)) inline void LoadAicPanelFeedToL1(
    __gm__ float* input, __gm__ float* pCache, const ChemmMixedTilingData& tiling, uint32_t ordinal, uint32_t feed,
    uint32_t job, uint32_t totalJobs, uint32_t ring, uint32_t qBank, uint32_t pElements, uint32_t panelM,
    __cbuf__ float* pL1, __cbuf__ float* qL1)
{
    constexpr uint32_t qElements = 2U * kTileM * CHEMM_EXP_K;
    CHEMM_WAIT(kInputReady + ring);
    if (qBank == 0U) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    } else {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    }
    __cbuf__ float* qBankL1 = qL1 + qBank * qElements;
    asc_copy_gm2l1_nd2nz(qBankL1, input, 1U, 2U * kTileM, CHEMM_EXP_K, 0U, CHEMM_EXP_K, 2U * kTileM, 1U, 0U);
    if (ordinal == 0U) {
        asc_copy_gm2l1_nd2nz(
            pL1 + feed * pElements, pCache + feed * CHEMM_INPUT_MATRIX_ELEMENTS, 1U, panelM, CHEMM_EXP_K, 0U,
            CHEMM_EXP_K, panelM, 1U, 0U);
    }
    if (qBank == 0U) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    }
    if (job + CHEMM_INPUT_RING < totalJobs) {
        CHEMM_NOTIFY(PIPE_MTE2, kInputFree + ring);
    }
}

template <uint32_t kTileM>
__aicore__ __attribute__((always_inline)) inline void ComputeAicPanelFeed(
    const ChemmMixedTilingData& tiling, uint32_t ordinal, uint32_t feed, uint32_t qBank, uint32_t pElements,
    uint32_t panelM, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0,
    __cc__ float* cTile)
{
    constexpr uint32_t qElements = 2U * kTileM * CHEMM_EXP_K;
    const bool loadP = tiling.kFeeds != 1U || ordinal == 0U;
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (loadP) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    if (qBank == 0U) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    }
    if (loadP) {
        LoadNzToL0aDynamic(pL0, pL1 + feed * pElements, panelM);
    }
    asc_copy_l12l0b(qL0, qL1 + qBank * qElements, 0U, kTileM, 1U, 0U);
    if (qBank == 0U) {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    }
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (loadP) {
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    }
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (loadP) {
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    }
    asc_mmad(cTile, pL0, qL0, panelM, CHEMM_EXP_K, 2U * kTileM, 0U, false, false, feed == 0U);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (tiling.kFeeds != 1U) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
}

__aicore__ __attribute__((always_inline)) inline void WriteAicPanelOutput(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t row, uint32_t validRows, uint32_t col,
    uint32_t validCols, uint32_t dstStride, uint32_t panelM, uint32_t bank, __cc__ float* cTile)
{
    asc_copy_l0c2gm(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(tiling.ldc)), cTile, 2U * validRows, validCols, dstStride,
        panelM, 0U, static_cast<uint64_t>(QuantMode_t::NoQuant), 0U, false, true);
    if (bank == 0U) {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    } else {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    }
}

__aicore__ __attribute__((always_inline)) inline void StoreAicPanelOutput(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t row, uint32_t validRows, uint32_t col,
    uint32_t validCols, uint32_t dstStride, uint32_t panelM, uint32_t bank, __cc__ float* cTile)
{
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    WriteAicPanelOutput(c, tiling, row, validRows, col, validCols, dstStride, panelM, bank, cTile);
}

__aicore__ __attribute__((always_inline)) inline void DrainAicPanelPipeline(
    const ChemmMixedTilingData& tiling, uint32_t tileCount)
{
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (tiling.kFeeds != 1U) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }
    if (tileCount != 0U) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        if (tileCount > 1U) {
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        }
    }
}

template <uint32_t kTileM>
__aicore__ __attribute__((always_inline)) inline void RunAicPanel(
    __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t tileBegin,
    uint32_t tileEnd, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    static_assert(
        kTileM >= CHEMM_TILE_M && kTileM <= CHEMM_MAX_MIX_TILE_M && kTileM % kMicro == 0U, "unsupported MIX row tile");
    const uint32_t panelM = tiling.panelM;
    const uint32_t pElements = panelM * CHEMM_EXP_K;
    constexpr uint32_t qElements = 2U * kTileM * CHEMM_EXP_K;
    static_assert(2U * qElements <= 4U * CHEMM_INPUT_MATRIX_ELEMENTS, "L1 Q overflow");
    static_assert(qElements <= 2U * CHEMM_INPUT_MATRIX_ELEMENTS, "L0B overflow");

    InitAicPanelPipeline();
    const uint32_t tileCount = tileEnd - tileBegin;
    const uint32_t totalJobs = tileCount * tiling.kFeeds;
    const bool sharedCache = ReuseCache(tiling) && tiling.tileColBase == 0U && tiling.blockDim >= 8U;
    const uint32_t cacheCore = sharedCache ? 0U : core;
    const uint32_t cacheBank = tiling.tileColBase & 1U;
    __gm__ float* pCache = CacheSlot(workspace, tiling, cacheCore, cacheBank, 0U, 0U);
    const uint32_t dstStride = 2U * static_cast<uint32_t>(tiling.ldc);
    const uint32_t col = tiling.tileColBase * CHEMM_WORK_N;
    const uint32_t validCols =
        static_cast<uint32_t>(tiling.n) - col < panelM ? static_cast<uint32_t>(tiling.n) - col : panelM;

    uint32_t job = 0U;
    uint32_t ring = 0U;
    uint32_t qBank = 0U;
    __gm__ float* input = workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride;
    for (uint32_t ordinal = 0U; ordinal < tileCount; ++ordinal) {
        const uint32_t row = (tiling.tileRowBase + tileBegin + ordinal) * kTileM;
        const uint32_t validRows =
            static_cast<uint32_t>(tiling.m) - row < kTileM ? static_cast<uint32_t>(tiling.m) - row : kTileM;
        const uint32_t cBank = ordinal & 1U;
        WaitAicPanelOutput(ordinal, cBank);
        __cc__ float* cTile = cL0 + cBank * (2U * kTileM * panelM);
        for (uint32_t feed = 0U; feed < tiling.kFeeds; ++feed) {
            LoadAicPanelFeedToL1<kTileM>(
                input, pCache, tiling, ordinal, feed, job, totalJobs, ring, qBank, pElements, panelM, pL1, qL1);
            ComputeAicPanelFeed<kTileM>(tiling, ordinal, feed, qBank, pElements, panelM, pL1, qL1, pL0, qL0, cTile);
            ++job;
            qBank ^= 1U;
            AdvanceAicInput(workspace, tiling, core, ring, input);
        }
        StoreAicPanelOutput(c, tiling, row, validRows, col, validCols, dstStride, panelM, cBank, cTile);
    }
    DrainAicPanelPipeline(tiling, tileCount);
}

// Narrow-path data preparation.
__aicore__ __attribute__((always_inline)) inline void LoadNarrowBToL1(
    __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t row, uint32_t validRows, __cbuf__ float* dst)
{
    const uint32_t ld = static_cast<uint32_t>(tiling.ldb);
    __gm__ float* src = b + 2U * ElementIndex(row, 0U, ld);
    const uint16_t width = static_cast<uint16_t>(2U * validRows);
    asc_copy_gm2l1_nd2nz(dst, src, 1U, kMicro, width, 0U, static_cast<uint16_t>(2U * ld), kMicro, 1U, 0U);
    const uint16_t tail = static_cast<uint16_t>(tiling.dimA - kMicro);
    asc_copy_gm2l1_nd2nz(
        dst + CHEMM_NARROW_PRODUCT_N * kMicro, src + 2ULL * kMicro * ld, 1U, tail, width, 0U,
        static_cast<uint16_t>(2U * ld), kMicro, 1U, 0U);
}

template <bool kResidentB>
__aicore__ __attribute__((always_inline)) inline void LoadNarrowBResidentToL1(
    __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t tileBegin, uint32_t tileEnd, __cbuf__ float* dst)
{
    if constexpr (kResidentB) {
        const uint32_t rowTile = tiling.rowTile;
        const uint16_t tailK = static_cast<uint16_t>(tiling.dimA - kMicro);
        constexpr uint32_t tileElements = CHEMM_NARROW_L1_BANK_ELEMENTS;
        const uint32_t localTiles = tileEnd - tileBegin;
        const uint32_t firstRow = TileRow(tiling, tileBegin) * rowTile;
        const uint32_t fullTiles = localTiles > 0U ? localTiles - 1U : 0U;
        __gm__ float* src = b + 2ULL * ElementIndex(firstRow, 0U, static_cast<uint32_t>(tiling.ldb));
        if (fullTiles != 0U) {
            // Keep the proven K16/Ktail NZ layout, but batch all complete row
            // tiles.  The two calls preserve the L0B transpose offsets while
            // reducing the GM->L1 command count from two per tile to two total.
            asc_copy_gm2l1_nd2nz(
                dst, src, static_cast<uint16_t>(fullTiles), kMicro, static_cast<uint16_t>(2U * rowTile),
                static_cast<uint16_t>(2U * rowTile), static_cast<uint16_t>(2U * tiling.ldb), kMicro, 1U,
                static_cast<uint16_t>(tileElements));
            asc_copy_gm2l1_nd2nz(
                dst + CHEMM_NARROW_PRODUCT_N * kMicro, src + 2ULL * kMicro * static_cast<uint32_t>(tiling.ldb),
                static_cast<uint16_t>(fullTiles), tailK, static_cast<uint16_t>(2U * rowTile),
                static_cast<uint16_t>(2U * rowTile), static_cast<uint16_t>(2U * tiling.ldb), kMicro, 1U,
                static_cast<uint16_t>(tileElements));
        }
        const uint32_t tailRow = firstRow + fullTiles * rowTile;
        const uint32_t validRows = static_cast<uint32_t>(tiling.m) - tailRow;
        if (validRows != 0U) {
            __cbuf__ float* tailDst = dst + static_cast<uint64_t>(fullTiles) * tileElements;
            __gm__ float* tailSrc = b + 2ULL * ElementIndex(tailRow, 0U, static_cast<uint32_t>(tiling.ldb));
            asc_copy_gm2l1_nd2nz(
                tailDst, tailSrc, 1U, kMicro, static_cast<uint16_t>(2U * validRows), 0U,
                static_cast<uint16_t>(2U * tiling.ldb), kMicro, 1U, 0U);
            asc_copy_gm2l1_nd2nz(
                tailDst + CHEMM_NARROW_PRODUCT_N * kMicro, tailSrc + 2ULL * kMicro * static_cast<uint32_t>(tiling.ldb),
                1U, tailK, static_cast<uint16_t>(2U * validRows), 0U, static_cast<uint16_t>(2U * tiling.ldb), kMicro,
                1U, 0U);
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void LoadNarrowBToL0(__cb__ float* dst, __cbuf__ float* src)
{
    constexpr uint8_t kRepeats = CHEMM_NARROW_PRODUCT_N / kMicro;
    constexpr uint16_t kFracGap = kRepeats - 1U;
    // FP32 transpose emits two K8 fractals per repeat group.  The K8 tail's
    // second fractal is dummy and lives in the fourth 16 KiB L0B quarter;
    // MMAD(K=17..24) never consumes it.
    asc_copy_l12l0b_trans(dst, src, 0U, kRepeats, 1U, 0U, false, kFracGap);
    asc_copy_l12l0b_trans(
        dst + CHEMM_NARROW_PRODUCT_N * kMicro, src + CHEMM_NARROW_PRODUCT_N * kMicro, 0U, kRepeats, 1U, 0U, false,
        kFracGap);
}

__aicore__ __attribute__((always_inline)) inline void LoadNarrowStageHead(
    __gm__ float* b, const ChemmMixedTilingData& tiling, uint32_t row, uint32_t validN, __cbuf__ float* qL1)
{
    __gm__ float* src = b + 2U * ElementIndex(row, 0U, static_cast<uint32_t>(tiling.ldb));
    const uint16_t srcStride = static_cast<uint16_t>(2U * tiling.ldb);
    asc_copy_gm2l1_nd2nz(qL1, src, 1U, kMicro, static_cast<uint16_t>(validN), 0U, srcStride, kMicro, 1U, 0U);
}

__aicore__ __attribute__((always_inline)) inline void LoadNarrowStageP(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, __cbuf__ float* pL1)
{
    asc_copy_gm2l1_nd2nz(
        pL1, NarrowStageP(workspace, tiling, core), 1U, CHEMM_NARROW_K, CHEMM_NARROW_EXP_K, 0U, CHEMM_NARROW_EXP_K,
        CHEMM_NARROW_PANEL_M, 1U, 0U);
}

__aicore__ __attribute__((always_inline)) inline void LoadNarrowStageTail(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t chunk, uint32_t paddedN,
    __cbuf__ float* packL1)
{
    asc_copy_gm2l1(packL1, NarrowStageTail(workspace, tiling, core, chunk), 2U * kMicro * paddedN * sizeof(float));
}

__aicore__ __attribute__((always_inline)) inline void LoadNarrowStagePToL0(__cbuf__ float* pL1, __ca__ float* pL0)
{
    LoadL1NzToL0a<2U, 6U>(pL0, pL1);
}

__aicore__ __attribute__((always_inline)) inline void LoadNarrowStageHeadToL0(
    uint32_t paddedN, __cbuf__ float* qL1, __cb__ float* qL0)
{
    const uint8_t repeats = static_cast<uint8_t>(paddedN / kMicro);
    const uint16_t fracGap = static_cast<uint16_t>(repeats - 1U);
    asc_copy_l12l0b_trans(qL0, qL1, 0U, repeats, 1U, 0U, false, fracGap);
}

__aicore__ __attribute__((always_inline)) inline void LoadNarrowStageTailToL0(
    uint32_t paddedN, __cbuf__ float* packL1, __cb__ float* packL0)
{
    const uint8_t repeats = static_cast<uint8_t>(paddedN / kMicro);
    asc_copy_l12l0b(packL0, packL1, 0U, static_cast<uint8_t>(4U * repeats), 1U, 0U);
}

__aicore__ __attribute__((always_inline)) inline void RunNarrowCompute(
    __gm__ float* b, __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core,
    __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    const uint32_t row0 = NarrowStageRowBegin(tiling, core);
    const uint32_t row1 = row0 + kNarrowStage0Rows;
    const uint32_t rows1 = NarrowStageRowEnd(tiling, core) - row1;
    const uint32_t validN1 = 2U * rows1;
    const uint32_t paddedN1 = (validN1 + kMicro - 1U) & ~(kMicro - 1U);
    __cbuf__ float* rawL1 = qL1;
    __cbuf__ float* packL10 = rawL1 + kMicro * kNarrowStageN;
    __cbuf__ float* packL11 = packL10 + 2U * kMicro * kNarrowStage0N;
    __cb__ float* qL00 = qL0;
    __cb__ float* qL01 = qL0 + kNarrowStage0QElements;
    __cc__ float* cL00 = cL0;
    __cc__ float* cL01 = cL0 + CHEMM_NARROW_PANEL_M * kNarrowStage0N;

    asc_set_fp32_mode();
    asc_set_mmad_direction_n();
    asc_set_l0c_copy_params(1U, 0U, 0U);

    // K0:16 for both stages is one contiguous ND panel.  Keep it resident in
    // L1 and slice the two complete N16 regions directly into disjoint L0B
    // banks.  The final core's six padded FP32 values remain unconsumed dummy.
    // The three initial GM->L1 transfers are independent; publish one event
    // after issuing all of them so MTE1 can consume the batch without two
    // avoidable MTE2/MTE1 round trips.
    LoadNarrowStageHead(b, tiling, row0, kNarrowStage0N + validN1, rawL1);
    LoadNarrowStageTail(workspace, tiling, core, 0U, kNarrowStage0N, packL10);
    LoadNarrowStageP(workspace, tiling, 0U, pL1);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    LoadNarrowStageHeadToL0(kNarrowStage0N, rawL1, qL00);
    LoadNarrowStageHeadToL0(paddedN1, rawL1 + kMicro * kNarrowStage0N, qL01);
    LoadNarrowStageTailToL0(kNarrowStage0N, packL10, qL00 + kMicro * kNarrowStage0N);
    LoadNarrowStagePToL0(pL1, pL0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    asc_mmad(cL00, pL0, qL00, CHEMM_NARROW_K, CHEMM_NARROW_EXP_K, kNarrowStage0N, 3U, false, false, true);
    asc_copy_l0c2gm(
        c + 2U * ElementIndex(row0, 0U, static_cast<uint32_t>(tiling.ldc)), cL00, kNarrowStage0N, CHEMM_NARROW_K,
        2U * static_cast<uint32_t>(tiling.ldc), CHEMM_NARROW_PANEL_M, 3U, static_cast<uint64_t>(QuantMode_t::NoQuant),
        0U, false, true);

    // Stage0 MMAD/FixPipe is already issued when the independent stage1 tail
    // enters L1/L0B, preserving AIC pipeline overlap inside the compute kernel.
    LoadNarrowStageTail(workspace, tiling, core, 1U, paddedN1, packL11);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    LoadNarrowStageTailToL0(paddedN1, packL11, qL01 + kMicro * paddedN1);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

    asc_mmad(
        cL01, pL0, qL01, CHEMM_NARROW_K, CHEMM_NARROW_EXP_K, static_cast<uint16_t>(validN1), 3U, false, false, true);
    asc_copy_l0c2gm(
        c + 2U * ElementIndex(row1, 0U, static_cast<uint32_t>(tiling.ldc)), cL01, static_cast<uint16_t>(validN1),
        CHEMM_NARROW_K, 2U * static_cast<uint32_t>(tiling.ldc), CHEMM_NARROW_PANEL_M, 3U,
        static_cast<uint64_t>(QuantMode_t::NoQuant), 0U, false, true);
    asc_sync_pipe(PIPE_FIX);
}

// chemm (arch22) pipelined AIC path overlapping copy and cube stages.

#pragma once

// RIGHT narrow super-tile: each AIV publishes one complete 64-row Q panel.
// The peer sends the second mode-2 token before packing its own panel, so the
// AIC can consume Q0 while Q1 is still being produced. L0C remains R64
// ping-pong; only the GM transfer size grows.

// AIC direct, shared-panel and planar pipelines.
__aicore__ __attribute__((always_inline)) inline void LoadAicPanelPairInput(
    __gm__ float* q, __gm__ float* pCache, uint32_t ordinal, uint32_t tileCount, uint32_t ring, uint32_t sub,
    uint32_t logical, uint32_t panelM, __cbuf__ float* pL1, __cbuf__ float* qL1)
{
    constexpr uint32_t tileM = 64U;
    constexpr uint32_t qElements = 2U * tileM * CHEMM_EXP_K;
    CHEMM_WAIT(kPairReady + 2U * ring + sub);
    if (sub == 0U) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    } else {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    }
    asc_copy_gm2l1_nd2nz(
        qL1 + sub * qElements, q + sub * qElements, 1U, 2U * tileM, CHEMM_EXP_K, 0U, CHEMM_EXP_K, 2U * tileM, 1U, 0U);
    if (sub == 0U) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        if (ordinal + CHEMM_INPUT_RING < tileCount) {
            ffts_cross_core_sync(PIPE_MTE2, AscendC::GetffstMsg(2U, kPairFree + ring));
        }
    }
    if (logical == 0U) {
        asc_copy_gm2l1_nd2nz(pL1, pCache, 1U, panelM, CHEMM_EXP_K, 0U, CHEMM_EXP_K, panelM, 1U, 0U);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    }
}

__aicore__ __attribute__((always_inline)) inline void ComputeAicPanelPair(
    uint32_t sub, uint32_t logical, uint32_t panelM, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0,
    __cb__ float* qL0, __cc__ float* cTile)
{
    constexpr uint32_t tileM = 64U;
    constexpr uint32_t qElements = 2U * tileM * CHEMM_EXP_K;
    if (logical >= 2U) {
        wait_flag(PIPE_FIX, PIPE_M, sub == 0U ? EVENT_ID0 : EVENT_ID1);
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (sub == 0U) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    }
    if (logical == 0U) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        LoadNzToL0aDynamic(pL0, pL1, panelM);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    }
    asc_copy_l12l0b(qL0, qL1 + sub * qElements, 0U, tileM, 1U, 0U);
    if (sub == 0U) {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    }
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    asc_mmad(cTile, pL0, qL0, panelM, CHEMM_EXP_K, 2U * tileM, 0U, false, false, true);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
}

__aicore__ __attribute__((always_inline)) inline void StoreAicPanelPairOutput(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t tileBegin, uint32_t ordinal, uint32_t sub,
    uint32_t col, uint32_t validCols, uint32_t dstStride, uint32_t panelM, __cc__ float* cTile)
{
    constexpr uint32_t tileM = 64U;
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    const uint32_t row = (tiling.tileRowBase + tileBegin + ordinal) * (2U * tileM) + sub * tileM;
    const uint32_t validRows =
        static_cast<uint32_t>(tiling.m) - row < tileM ? static_cast<uint32_t>(tiling.m) - row : tileM;
    WriteAicPanelOutput(c, tiling, row, validRows, col, validCols, dstStride, panelM, sub, cTile);
}

__aicore__ __attribute__((always_inline)) inline void DrainAicPanelPair(uint32_t tileCount)
{
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (tileCount != 0U) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    }
}

__aicore__ __attribute__((always_inline)) inline void RunAicPanelPair(
    __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t tileBegin,
    uint32_t tileEnd, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    constexpr uint32_t tileM = 64U;
    const uint32_t panelM = tiling.panelM;
    const uint32_t cElements = 2U * tileM * panelM;
    InitAicPanelPipeline();
    const uint32_t tileCount = tileEnd - tileBegin;
    const bool sharedCache = ReuseCache(tiling) && tiling.tileColBase == 0U && tiling.blockDim >= 8U;
    const uint32_t cacheCore = sharedCache ? 0U : core;
    const uint32_t cacheBank = tiling.tileColBase & 1U;
    __gm__ float* pCache = CacheSlot(workspace, tiling, cacheCore, cacheBank, 0U, 0U);
    const uint32_t col = tiling.tileColBase * CHEMM_WORK_N;
    const uint32_t validCols =
        static_cast<uint32_t>(tiling.n) - col < panelM ? static_cast<uint32_t>(tiling.n) - col : panelM;
    const uint32_t dstStride = 2U * static_cast<uint32_t>(tiling.ldc);
    uint32_t ring = 0U;
    uint32_t logical = 0U;
    __gm__ float* q = workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride;
    for (uint32_t ordinal = 0U; ordinal < tileCount; ++ordinal) {
        for (uint32_t sub = 0U; sub < 2U; ++sub) {
            __cc__ float* cTile = cL0 + sub * cElements;
            LoadAicPanelPairInput(q, pCache, ordinal, tileCount, ring, sub, logical, panelM, pL1, qL1);
            ComputeAicPanelPair(sub, logical, panelM, pL1, qL1, pL0, qL0, cTile);
            StoreAicPanelPairOutput(c, tiling, tileBegin, ordinal, sub, col, validCols, dstStride, panelM, cTile);
            ++logical;
        }
        ++ring;
        if (ring == CHEMM_INPUT_RING) {
            ring = 0U;
            q = workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride;
        } else {
            q += tiling.inputSlotStride;
        }
    }

    DrainAicPanelPair(tileCount);
}

__aicore__ __attribute__((always_inline)) inline void InitAicDirectPipeline()
{
    InitAicPanelPipeline();
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
}

__aicore__ __attribute__((always_inline)) inline void LoadAicDirectFeedToL1(
    __gm__ float* p0, __gm__ float* p1, __gm__ float* q, uint32_t job, uint32_t totalJobs, uint32_t ring, uint32_t bank,
    __cbuf__ float* pBank, __cbuf__ float* qBank)
{
    CHEMM_WAIT(kInputReady + ring);
    if (bank == 0U) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    } else {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    }
    asc_copy_gm2l1_nd2nz(pBank, p0, 1U, CHEMM_EXP_M, CHEMM_EXP_K, 0U, CHEMM_EXP_K, CHEMM_EXP_M, 1U, 0U);
    asc_copy_gm2l1_nd2nz(
        pBank + CHEMM_INPUT_MATRIX_ELEMENTS, p1, 1U, CHEMM_EXP_M, CHEMM_EXP_K, 0U, CHEMM_EXP_K, CHEMM_EXP_M, 1U, 0U);
    asc_copy_gm2l1_nd2nz(qBank, q, 1U, CHEMM_EXP_M, CHEMM_EXP_K, 0U, CHEMM_EXP_K, CHEMM_EXP_M, 1U, 0U);
    if (bank == 0U) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    }
    if (job + CHEMM_INPUT_RING < totalJobs) {
        CHEMM_NOTIFY(PIPE_MTE2, kInputFree + ring);
    }
}

__aicore__ __attribute__((always_inline)) inline void LoadAicDirectFeedToL0(
    uint32_t bank, __cbuf__ float* pBank, __cbuf__ float* qBank, __ca__ float* pL0, __cb__ float* qTile)
{
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (bank == 0U) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    }
    LoadNzToL0a(pL0, pBank);
    asc_copy_l12l0b(qTile, qBank, 0U, CHEMM_EXP_K, 1U, 0U);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    LoadNzToL0a(pL0 + CHEMM_INPUT_MATRIX_ELEMENTS, pBank + CHEMM_INPUT_MATRIX_ELEMENTS);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    if (bank == 0U) {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    } else {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    }
}

__aicore__ __attribute__((always_inline)) inline void StoreAicDirectProducts(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t row, uint32_t col, uint32_t col1, uint32_t validRows,
    uint32_t validCols0, uint32_t validCols1, uint32_t dstStride, __cc__ float* cL0)
{
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    asc_copy_l0c2gm(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(tiling.ldc)), cL0, 2U * validRows, validCols0, dstStride,
        CHEMM_EXP_M, 0U, static_cast<uint64_t>(QuantMode_t::NoQuant), 0U, false, true);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    asc_copy_l0c2gm(
        c + 2U * ElementIndex(row, col1, static_cast<uint32_t>(tiling.ldc)), cL0 + CHEMM_PRODUCT_ELEMENTS,
        2U * validRows, validCols1, dstStride, CHEMM_EXP_N, 0U, static_cast<uint64_t>(QuantMode_t::NoQuant), 0U, false,
        true);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
}

__aicore__ __attribute__((always_inline)) inline void ComputeAicDirectFeed(
    __gm__ float* c, const ChemmMixedTilingData& tiling, uint32_t ordinal, uint32_t feed, uint32_t bank, uint32_t row,
    uint32_t col, uint32_t col1, uint32_t validRows, uint32_t validCols0, uint32_t validCols1, uint32_t dstStride,
    __ca__ float* pL0, __cb__ float* qTile, __cc__ float* cL0)
{
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (feed == 0U && ordinal != 0U) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    asc_mmad(cL0, pL0, qTile, CHEMM_EXP_M, CHEMM_EXP_K, CHEMM_EXP_N, 0U, false, false, feed == 0U);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (feed + 1U == tiling.kFeeds) {
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    }
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    if (feed == 0U && ordinal != 0U) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    }
    asc_mmad(
        cL0 + CHEMM_PRODUCT_ELEMENTS, pL0 + CHEMM_INPUT_MATRIX_ELEMENTS, qTile, CHEMM_EXP_M, CHEMM_EXP_K, CHEMM_EXP_N,
        0U, false, false, feed == 0U);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    if (bank == 0U) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    } else {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    }
    if (feed + 1U == tiling.kFeeds) {
        StoreAicDirectProducts(c, tiling, row, col, col1, validRows, validCols0, validCols1, dstStride, cL0);
    }
}

__aicore__ __attribute__((always_inline)) inline void DrainAicDirectPipeline()
{
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
}

__aicore__ __attribute__((always_inline)) inline void RunAicDirectPipeline(
    __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t tileBegin,
    uint32_t tileEnd, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    InitAicDirectPipeline();
    const uint32_t tileCount = tileEnd - tileBegin;
    const uint32_t totalJobs = tileCount * tiling.kFeeds;
    const uint32_t dstStride = 2U * static_cast<uint32_t>(tiling.ldc);
    const bool reuse = ReuseCache(tiling);
    uint32_t job = 0U;
    uint32_t ring = 0U;
    uint32_t bank = 0U;
    __gm__ float* inputCursor = workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride;
    for (uint32_t ordinal = 0U; ordinal < tileCount; ++ordinal) {
        const uint32_t tile = tileBegin + ordinal;
        const uint32_t row = TileRow(tiling, tile) * CHEMM_TILE_M;
        const uint32_t col = TileCol(tiling, tile) * CHEMM_WORK_N;
        const uint32_t validRows =
            static_cast<uint32_t>(tiling.m) - row < CHEMM_TILE_M ? static_cast<uint32_t>(tiling.m) - row : CHEMM_TILE_M;
        const uint32_t validCols0 =
            static_cast<uint32_t>(tiling.n) - col < CHEMM_TILE_N ? static_cast<uint32_t>(tiling.n) - col : CHEMM_TILE_N;
        const uint32_t col1 = col + CHEMM_TILE_N;
        const uint32_t validCols1 = static_cast<uint32_t>(tiling.n) - col1 < CHEMM_TILE_N ?
                                        static_cast<uint32_t>(tiling.n) - col1 :
                                        CHEMM_TILE_N;

        const uint32_t cacheBank = CacheGroup(tiling, tile) & 1U;
        for (uint32_t feed = 0U; feed < tiling.kFeeds; ++feed) {
            __cbuf__ float* pBank = pL1 + bank * 2U * CHEMM_INPUT_MATRIX_ELEMENTS;
            __cbuf__ float* qBank = qL1 + bank * CHEMM_INPUT_MATRIX_ELEMENTS;
            __cb__ float* qTile = qL0 + bank * CHEMM_INPUT_MATRIX_ELEMENTS;
            __gm__ float* input = inputCursor;
            __gm__ float* p0 = reuse ? CacheSlot(workspace, tiling, core, cacheBank, feed, 0U) : input;
            __gm__ float* p1 =
                reuse ? CacheSlot(workspace, tiling, core, cacheBank, feed, 1U) : input + CHEMM_INPUT_MATRIX_ELEMENTS;
            __gm__ float* q = reuse ? input : input + 2U * CHEMM_INPUT_MATRIX_ELEMENTS;
            LoadAicDirectFeedToL1(p0, p1, q, job, totalJobs, ring, bank, pBank, qBank);
            LoadAicDirectFeedToL0(bank, pBank, qBank, pL0, qTile);
            ComputeAicDirectFeed(
                c, tiling, ordinal, feed, bank, row, col, col1, validRows, validCols0, validCols1, dstStride, pL0,
                qTile, cL0);
            ++job;
            bank ^= 1U;
            AdvanceAicInput(workspace, tiling, core, ring, inputCursor);
        }
    }
    DrainAicDirectPipeline();
}

__aicore__ __attribute__((always_inline)) inline void LoadAicPlanarFeedToL1(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t cacheBank, uint32_t feed,
    uint32_t job, uint32_t totalJobs, uint32_t ringSlot, uint32_t bank, bool reuseCache, __gm__ float* input,
    __cbuf__ float* pL1, __cbuf__ float* qL1)
{
    CHEMM_WAIT(kInputReady + ringSlot);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (bank == 0U) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    } else {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    }
    __gm__ float* p = reuseCache ? CacheSlot(workspace, tiling, core, cacheBank, feed, 0U) : input;
    __gm__ float* q = reuseCache ? input : input + 2U * CHEMM_PLANE_P_ELEMENTS;
    __cbuf__ float* qBank = qL1 + bank * (2U * CHEMM_PLANE_Q_ELEMENTS);
    asc_copy_gm2l1_nd2nz(pL1, p, 1U, CHEMM_EXP_M, CHEMM_K_FEED, 0U, CHEMM_K_FEED, CHEMM_EXP_M, 1U, 0U);
    asc_copy_gm2l1_nd2nz(
        pL1 + CHEMM_PLANE_P_ELEMENTS, p + CHEMM_PLANE_P_ELEMENTS, 1U, CHEMM_EXP_M, CHEMM_K_FEED, 0U, CHEMM_K_FEED,
        CHEMM_EXP_M, 1U, 0U);
    asc_copy_gm2l1_nd2nz(qBank, q, 1U, CHEMM_PLANE_N, CHEMM_K_FEED, 0U, CHEMM_K_FEED, CHEMM_PLANE_N, 1U, 0U);
    asc_copy_gm2l1_nd2nz(
        qBank + CHEMM_PLANE_Q_ELEMENTS, q + CHEMM_PLANE_Q_ELEMENTS, 1U, CHEMM_PLANE_N, CHEMM_K_FEED, 0U, CHEMM_K_FEED,
        CHEMM_PLANE_N, 1U, 0U);
    if (job + CHEMM_INPUT_RING < totalJobs) {
        CHEMM_NOTIFY(PIPE_MTE2, kInputFree + ringSlot);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
}

__aicore__ __attribute__((always_inline)) inline void ComputeAicPlanarFeed(
    uint32_t feed, uint32_t feedBegin, uint32_t bank, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0,
    __cb__ float* qL0, __cc__ float* cL0)
{
    __ca__ float* pReal = pL0;
    __ca__ float* pImag = pL0 + CHEMM_PLANE_P_ELEMENTS;
    __cb__ float* qBank = qL0 + bank * (2U * CHEMM_PLANE_Q_ELEMENTS);
    __cb__ float* qReal = qBank;
    __cb__ float* qImag = qBank + CHEMM_PLANE_Q_ELEMENTS;
    __cbuf__ float* qSource = qL1 + bank * (2U * CHEMM_PLANE_Q_ELEMENTS);
    LoadPlanarNzToL0a(pReal, pL1);
    LoadPlanarNzToL0a(pImag, pL1 + CHEMM_PLANE_P_ELEMENTS);
    asc_copy_l12l0b(qReal, qSource, 0U, kPlanarQFractals, 1U, 0U);
    asc_copy_l12l0b(qImag, qSource + CHEMM_PLANE_Q_ELEMENTS, 0U, kPlanarQFractals, 1U, 0U);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    asc_mmad(cL0, pReal, qReal, CHEMM_EXP_M, CHEMM_K_FEED, CHEMM_PLANE_N, 0U, false, false, feed == feedBegin);
    asc_mmad(
        cL0 + CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS, pImag, qImag, CHEMM_EXP_M, CHEMM_K_FEED, CHEMM_PLANE_N, 0U, false,
        false, feed == feedBegin);
    asc_mmad(
        cL0 + 2U * CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS, pReal, qImag, CHEMM_EXP_M, CHEMM_K_FEED, CHEMM_PLANE_N, 0U,
        false, false, feed == feedBegin);
    asc_mmad(
        cL0 + 3U * CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS, pImag, qReal, CHEMM_EXP_M, CHEMM_K_FEED, CHEMM_PLANE_N, 0U,
        false, false, feed == feedBegin);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    if (bank == 0U) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    } else {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    }
}

__aicore__ __attribute__((always_inline)) inline void StoreAicPlanarSlab(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t ordinal, uint32_t group,
    uint32_t slab, uint32_t slabs, __cc__ float* cL0)
{
    if (slab == 0U && ordinal >= CHEMM_PRODUCT_GROUPS) {
        CHEMM_WAIT(kProductFree + group);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    __gm__ float* product = ProductSlot(workspace, tiling, core, group * 2U + slab);
    for (uint32_t plane = 0U; plane < 4U; ++plane) {
        asc_copy_l0c2gm(
            product + plane * CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS, cL0 + plane * CHEMM_PLANAR_PRODUCT_PLANE_ELEMENTS,
            CHEMM_PLANE_N, CHEMM_EXP_M, CHEMM_PLANE_N, CHEMM_EXP_M, 0U, static_cast<uint64_t>(QuantMode_t::NoQuant), 0U,
            false, true);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    if (slab + 1U == slabs) {
        CHEMM_NOTIFY(PIPE_FIX, kProductReady + group);
    }
}

__aicore__ __attribute__((always_inline)) inline void RunAicPlanar(
    __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t tileBegin, uint32_t tileEnd,
    __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    InitAicPipeline();
    const uint32_t totalJobs = (tileEnd - tileBegin) * tiling.kFeeds;
    uint32_t job = 0U;
    uint32_t ringSlot = 0U;
    __gm__ float* inputCursor = workspace + static_cast<uint64_t>(core) * tiling.inputCoreStride;
    const bool reuseCache = ReuseCache(tiling);
    for (uint32_t ordinal = 0U; ordinal < tileEnd - tileBegin; ++ordinal) {
        const uint32_t group = ordinal & 1U;
        const uint32_t tile = tileBegin + ordinal;
        const uint32_t cacheBank = CacheGroup(tiling, tile) & 1U;
        const uint32_t split = (tiling.kFeeds + 1U) / 2U;
        const uint32_t slabs = tiling.kFeeds > 1U ? 2U : 1U;
        for (uint32_t slab = 0U; slab < slabs; ++slab) {
            if (ordinal != 0U || slab != 0U) {
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            }
            const uint32_t feedBegin = slab == 0U ? 0U : split;
            const uint32_t feedEnd = slab == 0U ? split : tiling.kFeeds;
            for (uint32_t feed = feedBegin; feed < feedEnd; ++feed) {
                const uint32_t bank = feed & 1U;
                LoadAicPlanarFeedToL1(
                    workspace, tiling, core, cacheBank, feed, job, totalJobs, ringSlot, bank, reuseCache, inputCursor,
                    pL1, qL1);
                ComputeAicPlanarFeed(feed, feedBegin, bank, pL1, qL1, pL0, qL0, cL0);
                ++job;
                AdvanceAicInput(workspace, tiling, core, ringSlot, inputCursor);
            }
            StoreAicPlanarSlab(workspace, tiling, core, ordinal, group, slab, slabs, cL0);
        }
    }
    DrainAicPipeline<false>(tileEnd - tileBegin);
}

// chemm (arch22) output epilogue: alpha/beta scaling and store-back.

#pragma once

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

#pragma once

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

constexpr uint32_t kAivAccumulatorBanks = 16U;
constexpr uint32_t kAivAccumulatorParts = 2U;

// chemm (arch22) AIV (vector) compute path.

#pragma once

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

#pragma once

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

// chemm (arch22) ordered AIV variant used for the general alpha/beta case.

#pragma once

// Ordered vector path.
__aicore__ inline __attribute__((always_inline)) void SplitOrderedFmaOperands(
    __ubuf__ float* src00, __ubuf__ float* src01, __ubuf__ float* src10, __ubuf__ float* src11,
    __ubuf__ float* product0, __ubuf__ float* product1, __ubuf__ float* highA0, __ubuf__ float* highA1,
    __ubuf__ float* lowA0, __ubuf__ float* lowA1, __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0,
    __ubuf__ float* lowB1, __ubuf__ float* temporary0, __ubuf__ float* temporary1, uint8_t rows,
    uint8_t src0BlockStride, uint8_t src1BlockStride, uint16_t src0RepeatStride, uint16_t src1RepeatStride)
{
    constexpr float kSplitter = 4097.0f;
    // kSplitClampLimit keeps `clamp(src, +-limit) * kSplitter` safely below FLT_MAX (~3.4028235e38) for any
    // finite FP32 input, including values with |src| close to FLT_MAX. The Dekker-style reconstruction below
    // (`hi = c - (c - src); lo = src - hi`) always satisfies hi + lo == src exactly, because every subtraction
    // step after this block still reads the TRUE, unclamped src00/src01/src10/src11; only the "c = ... *
    // kSplitter" step below reads the clamped copy. For normal-magnitude operands (|src| << kSplitClampLimit,
    // true for every case the public interface exercises in practice) the clamp is a no-op and the split stays
    // bit-identical to before. For the rare extreme-magnitude finite operand (e.g. |src| ~ 1e38, legal per the
    // "finite complex numbers" input contract) that would otherwise overflow `src * 4097` to Inf and poison the
    // result via a subsequent Inf-Inf == NaN, this keeps every intermediate finite while still reconstructing
    // src exactly, at the cost of a slightly coarser (but still correct and finite) high/low split.
    constexpr float kSplitClampLimit = 3.0e34f;
    asc_mul(product0, src00, src10, rows, 1U, src0BlockStride, src1BlockStride, 2U, src0RepeatStride, src1RepeatStride);
    asc_mul(product1, src01, src11, rows, 1U, src0BlockStride, src1BlockStride, 2U, src0RepeatStride, src1RepeatStride);
    asc_min_scalar(highA0, src00, kSplitClampLimit, rows, 1U, src0BlockStride, 2U, src0RepeatStride);
    asc_min_scalar(highA1, src01, kSplitClampLimit, rows, 1U, src0BlockStride, 2U, src0RepeatStride);
    asc_min_scalar(highB0, src10, kSplitClampLimit, rows, 1U, src1BlockStride, 2U, src1RepeatStride);
    asc_min_scalar(highB1, src11, kSplitClampLimit, rows, 1U, src1BlockStride, 2U, src1RepeatStride);
    asc_sync_pipe(PIPE_V);
    asc_max_scalar(highA0, highA0, -kSplitClampLimit, rows, 1U, 1U, 2U, 2U);
    asc_max_scalar(highA1, highA1, -kSplitClampLimit, rows, 1U, 1U, 2U, 2U);
    asc_max_scalar(highB0, highB0, -kSplitClampLimit, rows, 1U, 1U, 2U, 2U);
    asc_max_scalar(highB1, highB1, -kSplitClampLimit, rows, 1U, 1U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_mul_scalar(highA0, highA0, kSplitter, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(highA1, highA1, kSplitter, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(highB0, highB0, kSplitter, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(highB1, highB1, kSplitter, rows, 1U, 1U, 2U, 2U);
    asc_sync_pipe(PIPE_V);

    asc_sub(temporary0, highA0, src00, rows, 1U, 1U, src0BlockStride, 2U, 2U, src0RepeatStride);
    asc_sub(temporary1, highA1, src01, rows, 1U, 1U, src0BlockStride, 2U, 2U, src0RepeatStride);
    asc_sub(lowA0, highB0, src10, rows, 1U, 1U, src1BlockStride, 2U, 2U, src1RepeatStride);
    asc_sub(lowA1, highB1, src11, rows, 1U, 1U, src1BlockStride, 2U, 2U, src1RepeatStride);
    asc_sync_pipe(PIPE_V);

    asc_sub(highA0, highA0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(highA1, highA1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(highB0, highB0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(highB1, highB1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);

    asc_sub(lowA0, src00, highA0, rows, 1U, src0BlockStride, 1U, 2U, src0RepeatStride, 2U);
    asc_sub(lowA1, src01, highA1, rows, 1U, src0BlockStride, 1U, 2U, src0RepeatStride, 2U);
    asc_sub(lowB0, src10, highB0, rows, 1U, src1BlockStride, 1U, 2U, src1RepeatStride, 2U);
    asc_sub(lowB1, src11, highB1, rows, 1U, src1BlockStride, 1U, 2U, src1RepeatStride, 2U);
    asc_sync_pipe(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void ComputeOrderedFmaResidual(
    __ubuf__ float* product0, __ubuf__ float* product1, __ubuf__ float* highA0, __ubuf__ float* highA1,
    __ubuf__ float* lowA0, __ubuf__ float* lowA1, __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0,
    __ubuf__ float* lowB1, __ubuf__ float* error0, __ubuf__ float* error1, __ubuf__ float* temporary0,
    __ubuf__ float* temporary1, uint8_t rows)
{
    asc_mul(error0, highA0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(error1, highA1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_sub(error0, error0, product0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(error1, error1, product1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(temporary0, highA0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(temporary1, highA1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_mul(temporary0, lowA0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(temporary1, lowA1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_mul(temporary0, lowA0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(temporary1, lowA1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void AddOrderedFmaProduct(
    __ubuf__ float* dst0, __ubuf__ float* dst1, __ubuf__ float* product0, __ubuf__ float* product1,
    __ubuf__ float* highA0, __ubuf__ float* highA1, __ubuf__ float* lowA0, __ubuf__ float* lowA1,
    __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0, __ubuf__ float* lowB1,
    __ubuf__ float* error0, __ubuf__ float* error1, __ubuf__ float* temporary0, __ubuf__ float* temporary1,
    uint8_t rows)
{
    asc_add(highA0, dst0, product0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(highA1, dst1, product1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_sub(lowA0, highA0, dst0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(lowA1, highA1, dst1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_sub(highB0, highA0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(highB1, highA1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_sub(lowB0, dst0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(lowB1, dst1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(temporary0, product0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(temporary1, product1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(lowB0, lowB0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(lowB1, lowB1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(temporary0, error0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(temporary1, error1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(dst0, highA0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(dst1, highA1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
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

template <uint32_t kK>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivK(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* fmaScratch, uint8_t rows)
{
    constexpr uint32_t kVectorOffset = kK * kMicro;
    constexpr uint32_t kBroadcastOffset = kK * kMicro * kFp32DataBlockElements;
    __ubuf__ float* xr = broadcastReal + kBroadcastOffset;
    __ubuf__ float* xi = broadcastImag + kBroadcastOffset;
    __ubuf__ float* yr = vectorReal + kVectorOffset;
    __ubuf__ float* yi = vectorImag + kVectorOffset;
    __ubuf__ float* negativeYi = negativeVectorImag + kVectorOffset;
    OrderedSoftwareFmaPair(sumReal, sumImag, yr, yi, xr, xr, fmaScratch, rows, 1U, 0U, 0U, 1U);
    OrderedSoftwareFmaPair(sumReal, sumImag, negativeYi, yr, xi, xi, fmaScratch, rows, 1U, 0U, 0U, 1U);
}

template <uint32_t kK, uint32_t kEnd>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivRange(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* fmaScratch, uint8_t rows)
{
    AccumulateOrderedAivK<kK>(
        vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, fmaScratch, rows);
    if constexpr (kK + 1U < kEnd) {
        AccumulateOrderedAivRange<kK + 1U, kEnd>(
            vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, fmaScratch,
            rows);
    }
}

__aicore__ inline __attribute__((always_inline)) uint32_t OrderedAivKBlock(uint32_t remaining)
{
    constexpr uint32_t kBlock = 512U;
    if (remaining >= 2U * kBlock) {
        return kBlock;
    }
    if (remaining > kBlock) {
        constexpr uint32_t kUnroll = 8U;
        return ((remaining / 2U + kUnroll - 1U) / kUnroll) * kUnroll;
    }
    return remaining;
}

template <uint32_t kEnd>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivEnd(
    uint32_t begin, __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* fmaScratch, uint8_t rows)
{
    if (begin == 8U) {
        if constexpr (kEnd > 8U) {
            AccumulateOrderedAivRange<8U, kEnd>(
                vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, fmaScratch,
                rows);
        }
    } else {
        AccumulateOrderedAivRange<0U, kEnd>(
            vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, fmaScratch,
            rows);
    }
}

__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivSegment(
    uint32_t begin, uint32_t end, __ubuf__ float* vectorReal, __ubuf__ float* vectorImag,
    __ubuf__ float* negativeVectorImag, __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag,
    __ubuf__ float* sumReal, __ubuf__ float* sumImag, __ubuf__ float* fmaScratch, uint8_t rows)
{
#define CHEMM_ORDERED_END(value)                                                                               \
    case value:                                                                                                \
        AccumulateOrderedAivEnd<value>(                                                                        \
            begin, vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, \
            fmaScratch, rows);                                                                                 \
        break
    switch (end) {
        CHEMM_ORDERED_END(1U);
        CHEMM_ORDERED_END(2U);
        CHEMM_ORDERED_END(3U);
        CHEMM_ORDERED_END(4U);
        CHEMM_ORDERED_END(5U);
        CHEMM_ORDERED_END(6U);
        CHEMM_ORDERED_END(7U);
        CHEMM_ORDERED_END(8U);
        CHEMM_ORDERED_END(9U);
        CHEMM_ORDERED_END(10U);
        CHEMM_ORDERED_END(11U);
        CHEMM_ORDERED_END(12U);
        CHEMM_ORDERED_END(13U);
        CHEMM_ORDERED_END(14U);
        CHEMM_ORDERED_END(15U);
        default:
            AccumulateOrderedAivEnd<16U>(
                begin, vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag,
                fmaScratch, rows);
            break;
    }
#undef CHEMM_ORDERED_END
}

__aicore__ inline __attribute__((always_inline)) void InitializeOrderedAivOutput(
    const ChemmAivTilingData& tiling, __ubuf__ float* cReal, __ubuf__ float* cImag, __ubuf__ float* outReal,
    __ubuf__ float* outImag, __ubuf__ float* coefficients, __ubuf__ float* fmaScratch, uint8_t rows)
{
    if (tiling.betaReal == 0.0f && tiling.betaImag == 0.0f) {
        asc_duplicate(outReal, 0.0f, rows, 1U, 2U);
        asc_duplicate(outImag, 0.0f, rows, 1U, 2U);
        asc_sync_pipe(PIPE_V);
        return;
    }
    asc_mul_scalar(outReal, cReal, tiling.betaReal, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(outImag, cImag, tiling.betaReal, rows, 1U, 1U, 2U, 2U);
    asc_duplicate(coefficients, -tiling.betaImag, rows, 1U, 2U);
    asc_duplicate(coefficients + kMicroElements, tiling.betaImag, rows, 1U, 2U);
    asc_sync_pipe(PIPE_V);
    OrderedSoftwareFmaPair(
        outReal, outImag, cImag, cReal, coefficients, coefficients + kMicroElements, fmaScratch, rows, 1U, 1U, 2U, 2U);
}

__aicore__ inline __attribute__((always_inline)) void ApplyOrderedAivBlock(
    const ChemmAivTilingData& tiling, __ubuf__ float* sumReal, __ubuf__ float* sumImag, __ubuf__ float* outReal,
    __ubuf__ float* outImag, __ubuf__ float* coefficients, __ubuf__ float* fmaScratch, uint8_t rows)
{
    asc_duplicate(coefficients, tiling.alphaReal, rows, 1U, 2U);
    asc_duplicate(coefficients + kMicroElements, tiling.alphaImag, rows, 1U, 2U);
    asc_sync_pipe(PIPE_V);
    OrderedSoftwareFmaPair(
        outReal, outImag, sumReal, sumReal, coefficients, coefficients + kMicroElements, fmaScratch, rows, 1U, 1U, 2U,
        2U);
    asc_duplicate(coefficients, -tiling.alphaImag, rows, 1U, 2U);
    asc_duplicate(coefficients + kMicroElements, tiling.alphaReal, rows, 1U, 2U);
    asc_sync_pipe(PIPE_V);
    OrderedSoftwareFmaPair(
        outReal, outImag, sumImag, sumImag, coefficients, coefficients + kMicroElements, fmaScratch, rows, 1U, 1U, 2U,
        2U);
}

__aicore__ inline __attribute__((always_inline)) void PrepareOrderedAivOutput(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validCols,
    uint8_t rowRepeats, __ubuf__ uint8_t* masks, __ubuf__ float* outputTile, __ubuf__ float* xReal,
    __ubuf__ float* xImag, __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* broadcastReal,
    __ubuf__ float* fmaScratch)
{
    asc_set_mask_norm();
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    asc_copy_gm2ub_align(masks, MaskGm(tiling), 96U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    if (tiling.betaReal != 0.0f || tiling.betaImag != 0.0f) {
        LoadOutputTile16<false>(
            c, row, col, static_cast<uint32_t>(tiling.ldc), static_cast<uint32_t>(tiling.m),
            static_cast<uint32_t>(tiling.n), outputTile);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        CInterleavedToPlanes(outputTile, xReal, xImag);
        asc_sync_pipe(PIPE_V);
    }
    asc_set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
    InitializeOrderedAivOutput(tiling, xReal, xImag, outReal, outImag, broadcastReal, fmaScratch, rowRepeats);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivFeed(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validCols,
    uint8_t rowRepeats, uint32_t blockBegin, uint32_t blockEnd, uint32_t feed, __ubuf__ float* rawPair,
    __ubuf__ float* xReal, __ubuf__ float* xImag, __ubuf__ float* yReal, __ubuf__ float* yImag,
    __ubuf__ float* xTransReal, __ubuf__ float* xTransImag, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negativeYImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* fmaScratch, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    PrepareGridPair<kRight, kLower, false, false>(tiling, row, feed, feed, col, rawPair, rawPair + kComplexTileFloats);
    SubmitGridPair<0U>();
    CopyGridPair<kRight, kLower, false, false>(
        a, b, tiling, row, feed, feed, col, rawPair, rawPair + kComplexTileFloats);
    CommitGridPair<0U>();
    WaitGridPair<0U>();
    DecodeGridPair<kRight, kLower, 0U, false>(
        tiling, row, feed, feed, col, rawPair, rawPair + kComplexTileFloats, xReal, xImag, yReal, yImag, xTransReal,
        xTransImag, triangleMask, diagonalMask);
    asc_sync_pipe(PIPE_V);
    TransposePlane16(xReal, xTransReal);
    TransposePlane16(xImag, xTransImag);
    asc_sync_pipe(PIPE_V);
    BroadcastAivRange<0U, kMicro, kMicro>(xTransReal, xTransImag, broadcastReal, broadcastImag, 0U);
    asc_mul_scalar(negativeYImag, yImag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
    asc_sync_pipe(PIPE_V);
    asc_set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
    const uint32_t segmentBegin = blockBegin > feed ? blockBegin - feed : 0U;
    const uint32_t segmentEnd = blockEnd - feed < kMicro ? blockEnd - feed : kMicro;
    AccumulateOrderedAivSegment(
        segmentBegin, segmentEnd, yReal, yImag, negativeYImag, broadcastReal, broadcastImag, sumReal, sumImag,
        fmaScratch, rowRepeats);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivBlocks(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validCols,
    uint8_t rowRepeats, __ubuf__ float* rawPair, __ubuf__ float* xReal, __ubuf__ float* xImag, __ubuf__ float* yReal,
    __ubuf__ float* yImag, __ubuf__ float* xTransReal, __ubuf__ float* xTransImag, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negativeYImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* fmaScratch, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t dimA = static_cast<uint32_t>(tiling.dimA);
    uint32_t blockBegin = 0U;
    while (blockBegin < dimA) {
        const uint32_t blockEnd = blockBegin + OrderedAivKBlock(dimA - blockBegin);
        uint32_t feed = blockBegin & ~(kMicro - 1U);
        asc_duplicate(sumReal, 0.0f, rowRepeats, 1U, 2U);
        asc_duplicate(sumImag, 0.0f, rowRepeats, 1U, 2U);
        asc_sync_pipe(PIPE_V);
        while (feed < blockEnd) {
            AccumulateOrderedAivFeed<kRight, kLower>(
                a, b, tiling, row, col, validCols, rowRepeats, blockBegin, blockEnd, feed, rawPair, xReal, xImag, yReal,
                yImag, xTransReal, xTransImag, broadcastReal, broadcastImag, negativeYImag, sumReal, sumImag,
                fmaScratch, triangleMask, diagonalMask);
            feed += kMicro;
        }
        ApplyOrderedAivBlock(tiling, sumReal, sumImag, outReal, outImag, broadcastReal, fmaScratch, rowRepeats);
        blockBegin = blockEnd;
    }
}

__aicore__ inline __attribute__((always_inline)) void OrderedAivTile(
    const ChemmAivTilingData& tiling, uint32_t& row, uint32_t& col, uint32_t& validRows, uint32_t& validCols)
{
    const uint32_t tile = tiling.tileBase + static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t tileCol = tile / tiling.tileRows;
    const uint32_t tileRow = tile - tileCol * tiling.tileRows;
    row = tileRow * kMicro;
    col = tileCol * kMicro;
    validRows = static_cast<uint32_t>(tiling.m) - row < kMicro ? static_cast<uint32_t>(tiling.m) - row : kMicro;
    validCols = static_cast<uint32_t>(tiling.n) - col < kMicro ? static_cast<uint32_t>(tiling.n) - col : kMicro;
}

__aicore__ inline __attribute__((always_inline)) void StoreOrderedAivOutput(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validRows,
    uint32_t validCols, __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* outputTile)
{
    PlanesToCInterleavedStride(outReal, outImag, outputTile, kComplexColumnFloats);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    StoreComplexTail<kComplexColumnFloats>(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(tiling.ldc)), outputTile, validCols, validRows,
        static_cast<uint32_t>(tiling.ldc));
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void RunOrderedAiv(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* scratch)
{
    constexpr uint32_t kBroadcastElements = kMicro * kFp32DataBlockElements;
    __ubuf__ float* rawPair = scratch;
    __ubuf__ float* xPlanes = rawPair + 2U * kComplexTileFloats;
    __ubuf__ float* yPlanes = xPlanes + 2U * kMicroElements;
    __ubuf__ float* xTranspose = yPlanes + 2U * kMicroElements;
    __ubuf__ float* broadcasts = xTranspose + 2U * kMicroElements;
    __ubuf__ float* negativeYImag = broadcasts + 2U * kMicro * kBroadcastElements;
    __ubuf__ float* sums = negativeYImag + kMicroElements;
    __ubuf__ float* output = sums + 2U * kMicroElements;
    __ubuf__ float* outputTile = output + 2U * kMicroElements;
    __ubuf__ float* fmaScratch = outputTile + kComplexTileFloats;
    __ubuf__ uint8_t* masks = reinterpret_cast<__ubuf__ uint8_t*>(fmaScratch + 14U * kMicroElements);

    __ubuf__ float* xReal = xPlanes;
    __ubuf__ float* xImag = xPlanes + kMicroElements;
    __ubuf__ float* yReal = yPlanes;
    __ubuf__ float* yImag = yPlanes + kMicroElements;
    __ubuf__ float* xTransReal = xTranspose;
    __ubuf__ float* xTransImag = xTranspose + kMicroElements;
    __ubuf__ float* broadcastReal = broadcasts;
    __ubuf__ float* broadcastImag = broadcasts + kMicro * kBroadcastElements;
    __ubuf__ float* sumReal = sums;
    __ubuf__ float* sumImag = sums + kMicroElements;
    __ubuf__ float* outReal = output;
    __ubuf__ float* outImag = output + kMicroElements;
    __ubuf__ uint8_t* triangleMask = masks + (kLower ? 32U : 0U);
    __ubuf__ uint8_t* diagonalMask = masks + 64U;
    uint32_t row;
    uint32_t col;
    uint32_t validRows;
    uint32_t validCols;
    OrderedAivTile(tiling, row, col, validRows, validCols);
    const uint8_t rowRepeats = static_cast<uint8_t>(validRows);
    PrepareOrderedAivOutput(
        c, tiling, row, col, validCols, rowRepeats, masks, outputTile, xReal, xImag, outReal, outImag, broadcastReal,
        fmaScratch);
    AccumulateOrderedAivBlocks<kRight, kLower>(
        a, b, tiling, row, col, validCols, rowRepeats, rawPair, xReal, xImag, yReal, yImag, xTransReal, xTransImag,
        broadcastReal, broadcastImag, negativeYImag, sumReal, sumImag, outReal, outImag, fmaScratch, triangleMask,
        diagonalMask);
    StoreOrderedAivOutput(c, tiling, row, col, validRows, validCols, outReal, outImag, outputTile);
}

} // namespace

__aicore__ __attribute__((always_inline)) inline void RunAicMixedMode(
    __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling, uint32_t core, uint32_t begin,
    uint32_t end, __cbuf__ float* pL1, __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    const uint32_t flags = tiling.kernelFlags;
    if ((flags & CHEMM_FLAG_RESIDENT) != 0U) {
        if ((flags & CHEMM_FLAG_RIGHT) != 0U && tiling.tileM == CHEMM_MAX_MIX_TILE_M) {
            RunAicPanelPair(c, workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
        } else {
            RunAicPanel<CHEMM_TILE_M>(c, workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
        }
        return;
    }
    if ((flags & CHEMM_FLAG_SIMPLE) == 0U) {
        RunAicPlanar(workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
        return;
    }
    const bool direct = (flags & CHEMM_FLAG_DIRECT) != 0U;
    const bool onePanel = (flags & CHEMM_FLAG_ONE_PANEL) != 0U;
    if (direct && !onePanel) {
        RunAicDirectPipeline(c, workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
        return;
    }
    if (onePanel) {
        RunAic<true, true, CHEMM_EXP_M>(c, workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
    }
}

template <bool kRight, bool kLower>
__aicore__ __attribute__((always_inline)) inline void RunAivMixedMode(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, __gm__ float* workspace, const ChemmMixedTilingData& tiling,
    uint32_t core, uint32_t aiv, uint32_t begin, uint32_t end, __ubuf__ float* scratch)
{
    const uint32_t flags = tiling.kernelFlags;
    if ((flags & CHEMM_FLAG_RESIDENT) != 0U) {
        if constexpr (kRight) {
            if (tiling.tileM == CHEMM_MAX_MIX_TILE_M) {
                RunAivPanelPair<kLower>(a, b, workspace, tiling, core, aiv, begin, end, scratch);
                return;
            }
        }
        RunAivPanel<kRight, kLower, CHEMM_TILE_M>(a, b, workspace, tiling, core, aiv, begin, end, scratch);
        return;
    }
    if ((flags & CHEMM_FLAG_SIMPLE) == 0U) {
        RunAivPlanar<kRight, kLower>(a, b, c, workspace, tiling, core, aiv, begin, end, scratch);
        return;
    }
    const bool direct = (flags & CHEMM_FLAG_DIRECT) != 0U;
    if (direct) {
        RunAivDirect<kRight, kLower>(a, b, workspace, tiling, core, aiv, begin, end, scratch);
        return;
    }
}

enum class MixedAicPath : uint8_t { DIRECT, DYNAMIC };
enum class MixedAivPath : uint8_t { DYNAMIC };

template <MixedAicPath kPath, bool kOnePanel = false>
__aicore__ __attribute__((always_inline)) inline void RunMixedAicEntry(
    __gm__ uint8_t* gmC, __gm__ uint8_t* gmWorkspace, const ChemmMixedTilingData& tiling)
{
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    __gm__ float* workspace = reinterpret_cast<__gm__ float*>(gmWorkspace);
    __cbuf__ float pL1[4U * CHEMM_INPUT_MATRIX_ELEMENTS] __attribute__((aligned(512)));
    __cbuf__ float qL1[4U * CHEMM_INPUT_MATRIX_ELEMENTS] __attribute__((aligned(512)));
    __ca__ float pL0[2U * CHEMM_INPUT_MATRIX_ELEMENTS] __attribute__((aligned(512)));
    __cb__ float qL0[2U * CHEMM_INPUT_MATRIX_ELEMENTS] __attribute__((aligned(512)));
    __cc__ float cL0[2U * CHEMM_PRODUCT_ELEMENTS] __attribute__((aligned(512)));
    const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t count = TileCount(tiling);
    const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(count) * core / tiling.blockDim);
    const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(count) * (core + 1U) / tiling.blockDim);
    if constexpr (kPath == MixedAicPath::DIRECT) {
        if constexpr (kOnePanel) {
            RunAicFullPanel(c, workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
        } else {
            RunAicDirectPipeline(c, workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
        }
    } else {
        RunAicMixedMode(c, workspace, tiling, core, begin, end, pL1, qL1, pL0, qL0, cL0);
    }
}

template <MixedAivPath kPath, bool kRight = false, bool kLower = false>
__aicore__ __attribute__((always_inline)) inline void RunMixedAivEntry(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, __gm__ uint8_t* gmWorkspace,
    const ChemmMixedTilingData& tiling)
{
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    __gm__ float* workspace = reinterpret_cast<__gm__ float*>(gmWorkspace);
    __ubuf__ float scratch[kMixedAivScratchElements] __attribute__((aligned(32)));
    const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx() / 2U);
    const uint32_t aiv = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
    const uint32_t count = TileCount(tiling);
    const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(count) * core / tiling.blockDim);
    const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(count) * (core + 1U) / tiling.blockDim);
    const bool right = (tiling.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        RunAivMixedMode<true, true>(a, b, c, workspace, tiling, core, aiv, begin, end, scratch);
    } else if (right) {
        RunAivMixedMode<true, false>(a, b, c, workspace, tiling, core, aiv, begin, end, scratch);
    } else if (lower) {
        RunAivMixedMode<false, true>(a, b, c, workspace, tiling, core, aiv, begin, end, scratch);
    } else {
        RunAivMixedMode<false, false>(a, b, c, workspace, tiling, core, aiv, begin, end, scratch);
    }
}

template <bool kRight, bool kOnePanel, bool kLower>
__schedmode__(1) __global__ __mix__(1, 2) void chemm_direct_mixed_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, __gm__ uint8_t* gmWorkspace,
    ChemmMixedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    if ASCEND_IS_AIC {
        RunMixedAicEntry<MixedAicPath::DIRECT, kOnePanel>(gmC, gmWorkspace, tiling);
    }
    if ASCEND_IS_AIV {
        __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
        __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
        __gm__ float* workspace = reinterpret_cast<__gm__ float*>(gmWorkspace);
        __ubuf__ float scratch[kDirectAivScratchElements] __attribute__((aligned(32)));
        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx() / 2U);
        const uint32_t aiv = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        const uint32_t count = TileCount(tiling);
        const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(count) * core / tiling.blockDim);
        const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(count) * (core + 1U) / tiling.blockDim);
        RunAivDirectBody<kRight, kLower, true, kOnePanel, kOnePanel>(
            a, b, workspace, tiling, core, aiv, begin, end, scratch);
    }
}

__schedmode__(1) __global__ __mix__(1, 2) void chemm_mixed_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, __gm__ uint8_t* gmWorkspace,
    ChemmMixedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    if ASCEND_IS_AIC {
        RunMixedAicEntry<MixedAicPath::DYNAMIC>(gmC, gmWorkspace, tiling);
    }
    if ASCEND_IS_AIV {
        RunMixedAivEntry<MixedAivPath::DYNAMIC>(gmA, gmB, gmC, gmWorkspace, tiling);
    }
}

template <bool kRight, bool kLower>
__aicore__ inline void RunAivTinyMode(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* raw,
    __ubuf__ float* rowPlanes, __ubuf__ float* auxPlanes, __ubuf__ float* accumulators, __ubuf__ float* broadcasts,
    __ubuf__ float* negativeVectorImag, __ubuf__ uint8_t* masks)
{
    const bool full = tiling.m == static_cast<int32_t>(kMicro) && tiling.n == static_cast<int32_t>(kMicro) &&
                      tiling.dimA == static_cast<int32_t>(kMicro) && (static_cast<uint32_t>(tiling.lda) & 3U) == 0U &&
                      (static_cast<uint32_t>(tiling.ldb) & 3U) == 0U && (static_cast<uint32_t>(tiling.ldc) & 3U) == 0U;
    if (full) {
        RunAivTiny<kRight, kLower, true, kMicro, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (tiling.dimA == 1) {
        RunAivTiny<kRight, kLower, false, 1U, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (tiling.dimA == 2) {
        RunAivTiny<kRight, kLower, false, 2U, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (tiling.dimA <= 4) {
        RunAivTiny<kRight, kLower, false, 4U, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (tiling.dimA <= 8) {
        RunAivTiny<kRight, kLower, false, 8U, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else {
        RunAivTiny<kRight, kLower, false, kMicro, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    }
}

__global__ __aicore__ __vector__ void chemm_aiv_tiny_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, ChemmAivTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    constexpr uint32_t kTinyBroadcastElements = kAivRowsPerVectorCore * kFp32DataBlockElements;
    __ubuf__ float raw[kComplexTileFloats] __attribute__((aligned(32)));
    __ubuf__ float rowPlanes[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float auxPlanes[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float accumulators[kAivAccumulatorParts * kAivAccumulatorBanks * kMicroElements]
        __attribute__((aligned(32)));
    __ubuf__ float broadcasts[2U * kMicro * kTinyBroadcastElements] __attribute__((aligned(32)));
    __ubuf__ float negativeVectorImag[kMicroElements] __attribute__((aligned(32)));
    __ubuf__ uint8_t masks[96U] __attribute__((aligned(32)));
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    const bool right = (tiling.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        RunAivTinyMode<true, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (right) {
        RunAivTinyMode<true, false>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (lower) {
        RunAivTinyMode<false, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else {
        RunAivTinyMode<false, false>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    }
}

__global__ __aicore__ __vector__ void chemm_aiv_unit_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    RunAivUnit(
        reinterpret_cast<__gm__ float*>(gmA), reinterpret_cast<__gm__ float*>(gmB),
        reinterpret_cast<__gm__ float*>(gmC));
}

template <bool kRight, bool kLower>
__aicore__ inline void RunAivGridMode(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* rawSlots,
    __ubuf__ float* leftPlanes, __ubuf__ float* rightPlanes, __ubuf__ float* leftTranspose,
    __ubuf__ float* accumulators, __ubuf__ float* broadcasts, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* outputTile, __ubuf__ float* yCache, __ubuf__ uint8_t* masks)
{
    if (tiling.rowTile == kAivRowsPerVectorCore) {
        RunAivGrid<kRight, kLower, true, kAivRowsPerVectorCore>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    } else {
        RunAivGrid<kRight, kLower, true, kMicro>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    }
}

__global__ __aicore__ __vector__ void chemm_aiv_grid_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, ChemmAivTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    constexpr uint32_t kGridBroadcastElements = kMicro * kFp32DataBlockElements;
    __ubuf__ float rawSlots[4U * kComplexTileFloats] __attribute__((aligned(32)));
    __ubuf__ float leftPlanes[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float rightPlanes[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float leftTranspose[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float accumulators[kAivAccumulatorParts * kAivAccumulatorBanks * kMicroElements]
        __attribute__((aligned(32)));
    __ubuf__ float broadcasts[2U * kMicro * kGridBroadcastElements] __attribute__((aligned(32)));
    __ubuf__ float negativeVectorImag[kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float outputTile[kComplexTileFloats] __attribute__((aligned(32)));
    __ubuf__ float yCache[2U * kAivYCachePlaneElements] __attribute__((aligned(32)));
    __ubuf__ uint8_t masks[96U] __attribute__((aligned(32)));
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    const bool right = (tiling.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        RunAivGridMode<true, true>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    } else if (right) {
        RunAivGridMode<true, false>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    } else if (lower) {
        RunAivGridMode<false, true>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    } else {
        RunAivGridMode<false, false>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    }
}

__global__ __aicore__ __vector__ void chemm_ordered_aiv_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, ChemmAivTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    constexpr uint32_t kOrderedScratchElements =
        3U * kComplexTileFloats + 25U * kMicroElements + 2U * kMicro * kMicro * kFp32DataBlockElements + 24U;
    __ubuf__ float scratch[kOrderedScratchElements] __attribute__((aligned(32)));
    const bool right = (tiling.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        RunOrderedAiv<true, true>(a, b, c, tiling, scratch);
    } else if (right) {
        RunOrderedAiv<true, false>(a, b, c, tiling, scratch);
    } else if (lower) {
        RunOrderedAiv<false, true>(a, b, c, tiling, scratch);
    } else {
        RunOrderedAiv<false, false>(a, b, c, tiling, scratch);
    }
}

// Host-side launch dispatch.
template <bool kRight, bool kOnePanel, bool kLower>
void LaunchCompactDirect(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream)
{
    chemm_direct_mixed_kernel<kRight, kOnePanel, kLower>
        <<<tiling.blockDim, nullptr, stream>>>(a, b, c, workspace, tiling);
}

template <bool kRight, bool kOnePanel>
void LaunchCompactDirectByUplo(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream)
{
    if ((tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U) {
        LaunchCompactDirect<kRight, kOnePanel, true>(a, b, c, workspace, tiling, stream);
    } else {
        LaunchCompactDirect<kRight, kOnePanel, false>(a, b, c, workspace, tiling, stream);
    }
}

void LaunchMixed(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& base, void* stream,
    uint32_t flags, uint32_t panelM = CHEMM_EXP_M, uint32_t tileM = CHEMM_TILE_M)
{
    ChemmMixedTilingData tiling = base;
    tiling.kernelFlags = flags;
    tiling.panelM = panelM;
    tiling.tileM = tileM;
    constexpr uint32_t modeMask = CHEMM_FLAG_FULL | CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT | CHEMM_FLAG_ONE_PANEL |
                                  CHEMM_FLAG_KEEP_DUMMY | CHEMM_FLAG_RESIDENT;
    constexpr uint32_t fullPanelMode =
        CHEMM_FLAG_FULL | CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT | CHEMM_FLAG_ONE_PANEL | CHEMM_FLAG_KEEP_DUMMY;
    constexpr uint32_t directMode = CHEMM_FLAG_FULL | CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT;
    const uint32_t mode = flags & modeMask;
    if (mode == directMode) {
        const bool right = (flags & CHEMM_FLAG_RIGHT) != 0U;
        if (right) {
            LaunchCompactDirectByUplo<true, false>(a, b, c, workspace, tiling, stream);
        } else {
            LaunchCompactDirectByUplo<false, false>(a, b, c, workspace, tiling, stream);
        }
        return;
    }
    if (mode == fullPanelMode) {
        const bool right = (flags & CHEMM_FLAG_RIGHT) != 0U;
        if (right) {
            LaunchCompactDirectByUplo<true, true>(a, b, c, workspace, tiling, stream);
        } else {
            LaunchCompactDirectByUplo<false, true>(a, b, c, workspace, tiling, stream);
        }
        return;
    }
    chemm_mixed_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, workspace, tiling);
}

void chemm_aiv_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, const ChemmAivTilingData& base, bool right, bool lower, void* stream)
{
    ChemmAivTilingData tiling = base;
    tiling.kernelFlags = (right ? CHEMM_FLAG_RIGHT : 0U) | (lower ? CHEMM_FLAG_LOWER : 0U);
    if (tiling.m == 1 && tiling.n == 1 && tiling.dimA == 1) {
        chemm_aiv_unit_kernel<<<1U, nullptr, stream>>>(a, b, c);
        return;
    }
    const bool small = tiling.m <= static_cast<int32_t>(kMicro) && tiling.n <= static_cast<int32_t>(kMicro) &&
                       tiling.dimA <= static_cast<int32_t>(kMicro);
    if (small) {
        chemm_aiv_tiny_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, tiling);
    } else {
        chemm_aiv_grid_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, tiling);
    }
}

void chemm_ordered_aiv_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, const ChemmAivTilingData& base, bool right, bool lower, void* stream)
{
    ChemmAivTilingData tiling = base;
    tiling.kernelFlags = (right ? CHEMM_FLAG_RIGHT : 0U) | (lower ? CHEMM_FLAG_LOWER : 0U);
    chemm_ordered_aiv_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, tiling);
}

void LaunchRegion(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& base, void* stream,
    uint32_t rowBase, uint32_t colBase, uint32_t rows, uint32_t cols, uint32_t flags, uint32_t panelM = CHEMM_EXP_M,
    uint32_t tileM = CHEMM_TILE_M)
{
    ChemmMixedTilingData tiling = base;
    tiling.tileRowBase = rowBase;
    tiling.tileColBase = colBase;
    tiling.tileRows = rows;
    tiling.tileCols = cols;
    const bool onePanel = (flags & CHEMM_FLAG_ONE_PANEL) != 0U;
    const bool resident = (flags & CHEMM_FLAG_RESIDENT) != 0U;
    tiling.cachePanels = (flags & CHEMM_FLAG_SIMPLE) != 0U ? (onePanel ? 1U : 2U) : 1U;
    tiling.cacheFeeds = tiling.kFeeds;
    const uint32_t work = rows * cols;
    uint32_t groups = work;
    if (onePanel && !resident) {
        const uint32_t waves = (work + tiling.workspaceCores - 1U) / tiling.workspaceCores;
        groups = (work + waves - 1U) / waves;
    }
    tiling.blockDim = groups < tiling.workspaceCores ? groups : tiling.workspaceCores;
    LaunchMixed(a, b, c, workspace, tiling, stream, flags, panelM, tileM);
}

void LaunchPanelRows(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream,
    uint32_t rowBase, uint32_t colBase, uint32_t rows, uint32_t cols, uint32_t flags, uint32_t panelM, uint32_t tileM)
{
    flags |= CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT | CHEMM_FLAG_ONE_PANEL | CHEMM_FLAG_RESIDENT;
    LaunchRegion(a, b, c, workspace, tiling, stream, rowBase, colBase, rows, cols, flags, panelM, tileM);
}

void LaunchOnePanelSized(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream,
    uint32_t rowBase, uint32_t colBase, uint32_t rows, uint32_t cols, uint32_t validCols, uint32_t flags,
    uint32_t panelM, bool keepDummy)
{
    const bool right = (flags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (flags & CHEMM_FLAG_LOWER) != 0U;
    if (tiling.rowTile == CHEMM_TILE_M) {
        if (panelM <= 32U) {
            LaunchPanelRows(
                a, b, c, workspace, tiling, stream, rowBase, colBase, rows, cols, flags, panelM, CHEMM_TILE_M);
        }
        return;
    }
    if (right && tiling.rowTile == CHEMM_MAX_MIX_TILE_M) {
        if (panelM <= 32U) {
            LaunchPanelRows(
                a, b, c, workspace, tiling, stream, rowBase, colBase, rows, cols, flags, panelM, CHEMM_MAX_MIX_TILE_M);
        }
        return;
    }
    const bool full =
        keepDummy && validCols == panelM && (rowBase + rows) * CHEMM_TILE_M <= static_cast<uint32_t>(tiling.m);
    flags |= CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT | CHEMM_FLAG_ONE_PANEL;
    if (keepDummy) {
        flags |= CHEMM_FLAG_KEEP_DUMMY;
    }
    if (full) {
        flags |= CHEMM_FLAG_FULL;
    }
    LaunchRegion(a, b, c, workspace, tiling, stream, rowBase, colBase, rows, cols, flags, panelM);
}

void LaunchOnePanelRegion(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream,
    uint32_t rowBase, uint32_t colBase, uint32_t rows, uint32_t cols, uint32_t validCols, uint32_t flags, bool aligned)
{
    const uint32_t panelM = !aligned ? (validCols <= 16U ? 16U : 32U) :
                                       (validCols <= 16U ? 16U :
                                        validCols <= 32U ? 32U :
                                        validCols <= 64U ? 64U :
                                        validCols <= 96U ? 96U :
                                                           CHEMM_EXP_M);
    LaunchOnePanelSized(
        a, b, c, workspace, tiling, stream, rowBase, colBase, rows, cols, validCols, flags, panelM, aligned);
}

void DispatchFullRegion(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream,
    uint32_t baseFlags, bool simple, uint32_t fullRows, uint32_t fullCols)
{
    if (fullRows == 0U || fullCols == 0U) {
        return;
    }
    uint32_t flags = baseFlags | CHEMM_FLAG_FULL;
    if (simple) {
        flags |= CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT;
    }
    LaunchRegion(a, b, c, workspace, tiling, stream, 0U, 0U, fullRows, fullCols, flags);
}

void DispatchColumnTail(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream,
    uint32_t baseFlags, bool simple, uint32_t fullRows, uint32_t fullCols)
{
    if (fullRows == 0U || fullCols >= tiling.tileCols) {
        return;
    }
    if (!simple) {
        LaunchRegion(a, b, c, workspace, tiling, stream, 0U, fullCols, fullRows, 1U, baseFlags);
        return;
    }
    const uint32_t tailCols = static_cast<uint32_t>(tiling.n) - fullCols * CHEMM_WORK_N;
    if (tailCols <= CHEMM_TILE_N) {
        LaunchOnePanelRegion(a, b, c, workspace, tiling, stream, 0U, fullCols, fullRows, 1U, tailCols, baseFlags, true);
    } else {
        LaunchRegion(
            a, b, c, workspace, tiling, stream, 0U, fullCols, fullRows, 1U,
            baseFlags | CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT | CHEMM_FLAG_KEEP_DUMMY);
    }
}

void DispatchRowTail(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream,
    uint32_t baseFlags, bool simple, uint32_t fullRows, uint32_t fullCols)
{
    if (fullRows >= tiling.tileRows) {
        return;
    }
    if (!simple) {
        LaunchRegion(a, b, c, workspace, tiling, stream, fullRows, 0U, 1U, tiling.tileCols, baseFlags);
        return;
    }
    if (fullCols != 0U) {
        LaunchRegion(
            a, b, c, workspace, tiling, stream, fullRows, 0U, 1U, fullCols,
            baseFlags | CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT | CHEMM_FLAG_KEEP_DUMMY);
    }
    if (fullCols >= tiling.tileCols) {
        return;
    }
    const uint32_t tailCols = static_cast<uint32_t>(tiling.n) - fullCols * CHEMM_WORK_N;
    if (tailCols <= CHEMM_TILE_N) {
        LaunchOnePanelRegion(a, b, c, workspace, tiling, stream, fullRows, fullCols, 1U, 1U, tailCols, baseFlags, true);
    } else {
        LaunchRegion(
            a, b, c, workspace, tiling, stream, fullRows, fullCols, 1U, 1U,
            baseFlags | CHEMM_FLAG_SIMPLE | CHEMM_FLAG_DIRECT | CHEMM_FLAG_KEEP_DUMMY);
    }
}

void DispatchRegions(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream,
    uint32_t baseFlags, bool simple, bool aligned)
{
    const bool right = (baseFlags & CHEMM_FLAG_RIGHT) != 0U;
    if (simple && tiling.rowTile != 0U) {
        LaunchOnePanelRegion(
            a, b, c, workspace, tiling, stream, 0U, 0U, tiling.tileRows, 1U, static_cast<uint32_t>(tiling.n), baseFlags,
            false);
        return;
    }
    if (right && simple && !aligned && static_cast<uint32_t>(tiling.n) <= CHEMM_K_FEED) {
        LaunchOnePanelRegion(
            a, b, c, workspace, tiling, stream, 0U, 0U, tiling.tileRows, tiling.tileCols,
            static_cast<uint32_t>(tiling.n), baseFlags, false);
        return;
    }
    if (!aligned) {
        if (simple) {
            baseFlags |= CHEMM_FLAG_SIMPLE;
        }
        LaunchMixed(a, b, c, workspace, tiling, stream, baseFlags);
        return;
    }
    if (simple && static_cast<uint32_t>(tiling.n) <= CHEMM_TILE_N) {
        LaunchOnePanelRegion(
            a, b, c, workspace, tiling, stream, 0U, 0U, tiling.tileRows, tiling.tileCols,
            static_cast<uint32_t>(tiling.n), baseFlags, true);
        return;
    }
    const uint32_t fullRows = static_cast<uint32_t>(tiling.m) / CHEMM_TILE_M;
    const uint32_t workN = simple ? CHEMM_WORK_N : CHEMM_TILE_N;
    const uint32_t fullCols = static_cast<uint32_t>(tiling.n) / workN;
    DispatchFullRegion(a, b, c, workspace, tiling, stream, baseFlags, simple, fullRows, fullCols);
    DispatchColumnTail(a, b, c, workspace, tiling, stream, baseFlags, simple, fullRows, fullCols);
    DispatchRowTail(a, b, c, workspace, tiling, stream, baseFlags, simple, fullRows, fullCols);
}

void DispatchMixed(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream,
    uint32_t flags)
{
    const uint32_t ldcBlocks = static_cast<uint32_t>(tiling.ldc) / 4U;
    const bool aligned =
        static_cast<uint32_t>(tiling.dimA) % CHEMM_K_FEED == 0U && (static_cast<uint32_t>(tiling.lda) & 3U) == 0U &&
        (static_cast<uint32_t>(tiling.ldb) & 3U) == 0U && (static_cast<uint32_t>(tiling.ldc) & 3U) == 0U &&
        ldcBlocks >= CHEMM_TILE_M / 4U && ldcBlocks - CHEMM_TILE_M / 4U <= 65535U;
    const bool simple =
        tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f && tiling.betaReal == 0.0f && tiling.betaImag == 0.0f;
    DispatchRegions(a, b, c, workspace, tiling, stream, flags, simple, aligned);
}

void chemm_mixed_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, bool right, bool lower,
    void* stream)
{
    const uint32_t flags = (right ? CHEMM_FLAG_RIGHT : 0U) | (lower ? CHEMM_FLAG_LOWER : 0U);
    DispatchMixed(a, b, c, workspace, tiling, stream, flags);
}
