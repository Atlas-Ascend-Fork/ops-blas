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

// chemm (arch22) mixed strategy: routes between direct / narrow / panel
// sub-kernels according to leading-dim alignment and available core count.

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
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    __cb__ float* qSharedL0 = qL0;
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
            __cb__ float* qTile = qL0;
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
    __cb__ float* qBank = qL0;
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

} // namespace
