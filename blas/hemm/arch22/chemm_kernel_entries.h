/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chemm (arch22) MIX kernel entry runners and the direct-mixed kernel
// template. Each direct-mixed template instantiation lives in its own
// translation unit so the per-object .cb (L0B) static budget (64 KB) is
// consumed by exactly one kernel; the launchers below are the cross-TU
// host symbols used by chemm_mixed_kernel.cpp.

#pragma once

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
    // The L1 panels use runtime address-space views instead of static __cbuf__
    // arrays: the arch22 lld budget check folds every section named ".cb*"
    // (including ".cbuf.*") into the single 64 KB "cb static variables"
    // budget, so static L1 panels overflow it while runtime views emit no
    // section at all.  Physical layout is unchanged: pL1 @ L1 byte 0 and
    // qL1 @ L1 byte 128KB, four 128x64 panel banks each, matching the
    // previous static declarations bit for bit.
    AscendC::LocalTensor<float> pL1View{AscendC::TPosition::A1, 0U, 4U * CHEMM_INPUT_MATRIX_ELEMENTS};
    AscendC::LocalTensor<float> qL1View{
        AscendC::TPosition::A1, 4U * CHEMM_INPUT_MATRIX_ELEMENTS * sizeof(float), 4U * CHEMM_INPUT_MATRIX_ELEMENTS};
    __cbuf__ float* pL1 = reinterpret_cast<__cbuf__ float*>(pL1View.GetPhyAddr());
    __cbuf__ float* qL1 = reinterpret_cast<__cbuf__ float*>(qL1View.GetPhyAddr());
    __ca__ float pL0[2U * CHEMM_INPUT_MATRIX_ELEMENTS] __attribute__((aligned(512)));
    __cb__ float qL0[CHEMM_INPUT_MATRIX_ELEMENTS] __attribute__((aligned(512)));
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

// Cross-TU launchers (defined in the per-instantiation translation units).
void LaunchDirectMixedLeftLower(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
void LaunchDirectMixedLeftUpper(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
void LaunchDirectMixedRightLower(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
void LaunchDirectMixedRightUpper(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
void LaunchDirectMixedLeftPanelLower(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
void LaunchDirectMixedLeftPanelUpper(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
void LaunchDirectMixedRightPanelLower(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
void LaunchDirectMixedRightPanelUpper(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream);
