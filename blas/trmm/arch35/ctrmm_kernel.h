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

/*!
 * \file ctrmm_kernel.h
 * \brief Kernel entry declarations for aclblasCtrmm (ascend950, arch35).
 *
 * The host side must not forward-declare kernel_do functions;
 * all declarations live in this header.
 */

#include <cstdint>
#include "ctrmm_tiling_data.h"

// M2 correctness baseline: single AIV SIMT kernel, naive 4M complex MAC,
// C = alpha * op(A) * B (or alpha * B * op(A)) written out-of-place to C.
// Reads A/B directly from GM, writes C directly to GM; workspace = 0.
void ctrmm_naive_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmC, const CtrmmTilingData& tiling, uint32_t numBlocks,
    void* stream);

// M3 4M all-device pipeline (cherk form):
//   Phase-A: split/mirror — complex A,B -> op(A) real/imag planes + B planes.
void ctrmm_split_mirror_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmAr, uint8_t* gmAi, uint8_t* gmBr, uint8_t* gmBi, uint8_t* gmAsum,
    uint8_t* gmBsum, const CtrmmSplitMirrorTilingData& tiling, uint32_t numBlocks, void* stream);

// Phase-C alternative: tensor_api (UB block DMA + Interleave) combine.
void ctrmm_combine_tla_kernel_do(
    const uint8_t* gmP1, const uint8_t* gmP2, const uint8_t* gmP3, const uint8_t* gmP4, uint8_t* gmC,
    const CtrmmCombineTilingData& tiling, uint32_t numBlocks, void* stream);

// Phase-A fused tensor_api variant (trans = N): A-side mirror + B-side split
// in ONE launch (per-column interval masking for the triangle, UNIT diagonal).
// trans = T/C keeps the SIMT A-mirror.
void ctrmm_split_fused_tla_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmAr, uint8_t* gmAi, uint8_t* gmAsum, uint8_t* gmBr, uint8_t* gmBi,
    uint8_t* gmBsum, const CtrmmSplitMirrorTilingData& tiling, uint32_t numBlocks, void* stream);

// Phase-A (B-side) tensor_api variant: complex B -> Br/Bi/Bsum planes via UB
// block DMA + DeInterleave. The A-side mirror stays SIMT.
void ctrmm_split_b_tla_kernel_do(
    const uint8_t* gmB, uint8_t* gmBr, uint8_t* gmBi, uint8_t* gmBsum, const CtrmmSplitMirrorTilingData& tiling,
    uint32_t numBlocks, void* stream);

// Phase-B alternative: strmm-lineage cube kernel (L0 lever). Computes one
// real plane product per launch, temp readable as P(i,j) @ [j*stride + i].
void ctrmm_phaseb_kernel_do(
    uint8_t* gmA, const uint8_t* gmB, uint8_t* gmTemp, const CtrmmPhaseBTilingData& tiling, uint32_t numBlocks,
    void* stream);

// Merged Phase-B: three products in one launch (same per-product work and
// scheduling; only the per-launch fixed cost is removed).
void ctrmm_phaseb3_kernel_do(
    uint8_t* gmA0, uint8_t* gmA1, uint8_t* gmA2, const uint8_t* gmB0, const uint8_t* gmB1, const uint8_t* gmB2,
    uint8_t* gmT0, uint8_t* gmT1, uint8_t* gmT2, const CtrmmPhaseBTilingData& tiling, uint32_t numBlocks, void* stream);

// TINY fast path: single-block flattened kernel for m,n <= 32 (acceptance
// gap-closure for the tiny performance band; numerics identical to naive).
void ctrmm_tiny_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmC, const CtrmmTilingData& tiling, void* stream);
// TINY K=32 specialization: compile-time-unrolled k loop (batched loads).
void ctrmm_tiny_k32_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmC, const CtrmmTilingData& tiling, void* stream);
