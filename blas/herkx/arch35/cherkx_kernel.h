/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cherkx_kernel.h
 * \brief Declaration of kernel launchers for aclblasCherkx (arch35).
 *        Shared by host.cpp and kernel.cpp.
 *
 *        Phase 1 (AIC GEMM) reuses gemm_kernel_do from blas/gemm/arch35/gemm_kernel.h.
 *        Phase 0 and Phase 2 kernels are declared here.
 *
 *        cherkx differs from cherk: Phase 0 deinterleave is called twice (A and B).
 *        Phase 2 combine kernel is structurally identical to cherk (Cr=t1+t2, Ci=t3-t4),
 *        but the diagonal imaginary part is theoretically non-zero (A != B) and must
 *        be explicitly forced to 0 via ZeroDiagonal.
 */

#pragma once

#include <cstdint>
#include "cherkx_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0: Deinterleave complex matrix → real, imag (AIV-only)
// Reads complex matrix from GM, splits into real/imag float matrices in GM.
// Called twice by host: once for A → Ar, Ai; once for B → Br, Bi.
void cherkx_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai,
    const CherkxDeinterleaveTilingData& tiling,
    uint32_t numBlocks, void* stream);

// Phase 2: Combine/Scale/Hermitian kernel (AIV-only)
// Merges 4 real GEMM results (t1..t4) into complex C with alpha/beta scaling
// and Hermitian constraint.
//   P_r = t1 + t2, P_i = t3 - t4
//   alpha complex: Cr = αr·P_r − αi·P_i, Ci = αr·P_i + αi·P_r
//   (αi==0 keeps the real-alpha path: Cr = αr·P_r, Ci = αr·P_i)
//   C = (Cr + i·Ci) + beta * C_old
//   Only writes uplo triangle; diagonal imaginary part is forced to zero.
//   (cherkx: diagonal imag is theoretically non-zero since A != B, must force to 0.)
void cherkx_combine_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4,
    GM_ADDR c,
    const CherkxCombineTilingData& tiling,
    uint32_t numBlocks, void* stream);
