/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cher2k_kernel.h
 * \brief Declaration of kernel launchers for aclblasCher2k (arch35).
 *        Shared by host.cpp and kernel.cpp.
 *
 *        Kernel inventory (design §1.4):
 *          K1 cher2k_deinterleave_kernel_do : AIV SIMD, splits A and B into Ar, Ai, Br, Bi
 *          K2 gemm_kernel_do (launched four times) : AIC Cube, reuses blas/gemm/arch35/gemm_kernel.h
 *          K3 cher2k_combine_kernel_do      : AIV SIMD, forms M plus its conjugate transpose plus beta times old C
 *          K4 cher2k_simt_small_do          : AIV SIMT, fused direct path for n up to 8
 *                                           (CHER2K_ARCH35_SIMT_N_MAX, P1-F1)
 */

#pragma once

#include <cstdint>
#include "cher2k_tiling_data.h"
#include "gemm/arch35/gemm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// K1 / Phase 0: Deinterleave complex A and B into the real matrices Ar, Ai, Br, Bi (AIV-only)
// Reads column-major complex A and B from GM, splits each into real/imag float
// matrices in GM (tightly packed, row stride equals the physical row count).
// Core split: cores below tiling.splitCore process A, the rest process B.
void cher2k_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, const Cher2kDeinterleaveTilingData& tiling,
    uint32_t numBlocks, void* stream);

// K2f / Phase 1-fused: 4-product cube kernel (AIC-only, cher2k-owned).
// One launch computes all four real products of the 4M split into t1..t4,
// replacing the 4 separate gemm_kernel_do launches. Operands are the
// deinterleaved real matrices; the pairing is t1 as X1 times Y1, t2 as X2 times
// Y2, t3 as X1 times Y2, t4 as X2 times Y1 in the same post-swap row-major view
// the shared kernel receives (host maps X and Y so the products land in the same
// GM areas as before).
// tiling uses the shared GemmTilingData (gemm/arch35/gemm_tiling_data.h).
void cher2k_fused4_do(
    uint32_t numBlocks, void* stream, GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, GM_ADDR c1, GM_ADDR c2,
    GM_ADDR c3, GM_ADDR c4, const GemmTilingData& tilingData);

// [MIX fold] K2f MIX variant (AIC+AIV, KERNEL_TYPE_MIX_AIC_1_2): the AIC hands
// the 4 cube accumulators to the AIV through UB instead of GM. foldMode selects
// the AIV sink (CHER2K_FOLD_STORE4 / CHER2K_FOLD_PRPI, cher2k_tiling_data.h).
// The 4 output pointers keep the c1..c4 meaning; under CHER2K_FOLD_PRPI only
// c1 (PR) and c3 (PI) receive data.
void cher2k_fused4_mix_do(
    uint32_t numBlocks, void* stream, GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, GM_ADDR c1, GM_ADDR c2,
    GM_ADDR c3, GM_ADDR c4, const GemmTilingData& tilingData, uint32_t foldMode,
    const Cher2kCombineParams& combineParams);

// K3 / Phase 2: Combine kernel (AIV-only)
// Merges the 4 real GEMM results (t1..t4) into the complex output C with
// alpha/beta scaling and the Hermitian construction that adds M and its
// conjugate transpose:
//   M is alpha times the complex number whose real part PR is t1 plus t2 and
//   whose imaginary part PI is t3 minus t4; the result is M plus M-conjugate-
//   transpose plus beta times the old C, and only the uplo triangle is written.
// Element-wise, writing u for the real part of the (i,j) entry of PR, v for the
// imaginary part of that entry, w for the real part of its (j,i) mirror, x for
// the imaginary part of that mirror, and alpha for ar plus i times ai:
//   the real part of the (i,j) entry of C is ar times the sum of u and w, minus
//   ai times the sum of v and x, plus beta times the real part of the old C;
//   the imaginary part is ar times the difference of v and x, plus ai times the
//   difference of u and w, plus beta times the imaginary part of the old C;
//   on the diagonal (where u equals w and v equals x) the imaginary part
//   vanishes.
// [p1l R1] The (j,i) partner tile is read DIRECTLY from the t1..t4 frames with
// a swapped-origin 2D strided copy (no GM transposed twins): the pre-P1 read
// geometry, restored because the P1-D twin store (K2t) was unfixable through
// the CopyUbufToGmAlignV2 lowering (all-zeros-at-n≥9 regression). K3 consumes
// it with the pre-P1 element-wise transposed gather (scalar combine).
// [iter34] alphaGm/betaGm are the DEVICE pointers of alpha (2 contiguous floats
// {real,imag}) and beta (1 float). The kernel reads them straight from GM and
// derives alphaReal/alphaImag/betaVal/isAlphaZero/isBetaZero (the tiling fields
// are placeholders on this path); the host no longer stages them through a
// device-to-host scalar read inside the timed window (~24us/call).
void cher2k_combine_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, GM_ADDR alphaGm, GM_ADDR betaGm,
    const Cher2kCombineTilingData& tiling, uint32_t numBlocks, void* stream);

// [p1l R1] cher2k_transpose_kernel_do (K2t, [P1-D] twin builder) was REMOVED:
// its GM-side store could not express a per-row burst advance of cols*4 bytes
// for cols<64 tails (Normal mode mis-rows the tail, Compact mode is dropped by
// the CopyUbufToGmAlignV2 lowering for GM destinations), which made K3 read the
// twins as zero for every n of 9 or more. See tmp/p1k twinprobe / tmp/p1l delivery.md.

// K4: small-n fused SIMT direct path (AIV SIMT, n up to CHER2K_ARCH35_SIMT_N_MAX of 8, no workspace)
// Computes the result directly per element as alpha times A times the conjugate
// transpose of B, plus the conjugate of alpha times B times the conjugate
// transpose of A, plus beta times the old C.
// [iter34] alphaGm/betaGm are the device pointers of alpha ({real,imag}) and
// beta; the kernel reads them from GM instead of the tiling placeholders (the
// host no longer stages them via a D2H read + stream sync).
void cher2k_simt_small_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR alphaGm, GM_ADDR betaGm, const Cher2kSimtTilingData& tiling,
    uint32_t numBlocks, void* stream);
