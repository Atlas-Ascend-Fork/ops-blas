/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chemm (arch22) mixed kernel translation unit: the DYNAMIC MIX kernel,
// the flag-based dispatch into the per-instantiation direct-mixed launchers
// and the host launch chain (chemm_mixed_kernel_do).

#include "chemm_kernel_common.h"
#include "chemm_kernel_aic.h"
#include "chemm_kernel_mix_aiv.h"
#include "chemm_kernel_aiv.h"
#include "chemm_kernel_ordered.h"

#include "chemm_kernel_entries.h"

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

void LaunchCompactDirectByUplo(
    bool right, bool onePanel, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace,
    const ChemmMixedTilingData& tiling, void* stream)
{
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (!right && !onePanel) {
        if (lower) {
            LaunchDirectMixedLeftLower(a, b, c, workspace, tiling, stream);
        } else {
            LaunchDirectMixedLeftUpper(a, b, c, workspace, tiling, stream);
        }
    } else if (right && !onePanel) {
        if (lower) {
            LaunchDirectMixedRightLower(a, b, c, workspace, tiling, stream);
        } else {
            LaunchDirectMixedRightUpper(a, b, c, workspace, tiling, stream);
        }
    } else if (!right && onePanel) {
        if (lower) {
            LaunchDirectMixedLeftPanelLower(a, b, c, workspace, tiling, stream);
        } else {
            LaunchDirectMixedLeftPanelUpper(a, b, c, workspace, tiling, stream);
        }
    } else {
        if (lower) {
            LaunchDirectMixedRightPanelLower(a, b, c, workspace, tiling, stream);
        } else {
            LaunchDirectMixedRightPanelUpper(a, b, c, workspace, tiling, stream);
        }
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
            LaunchCompactDirectByUplo(true, false, a, b, c, workspace, tiling, stream);
        } else {
            LaunchCompactDirectByUplo(false, false, a, b, c, workspace, tiling, stream);
        }
        return;
    }
    if (mode == fullPanelMode) {
        const bool right = (flags & CHEMM_FLAG_RIGHT) != 0U;
        if (right) {
            LaunchCompactDirectByUplo(true, true, a, b, c, workspace, tiling, stream);
        } else {
            LaunchCompactDirectByUplo(false, true, a, b, c, workspace, tiling, stream);
        }
        return;
    }
    chemm_mixed_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, workspace, tiling);
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
