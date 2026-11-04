/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cgemm_batched_kernel.h
 * \brief Kernel launcher signatures for aclblasCgemmBatched on arch22, shared by
 *        the host and the kernel translation unit so that the compiler checks
 *        both against one declaration.
 */

#ifndef CGEMM_BATCHED_ARCH22_KERNEL_H
#define CGEMM_BATCHED_ARCH22_KERNEL_H

#include <cstdint>
#include "cgemm_batched_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0 (AIV): de-interleave BOTH operands' batch members into packed real /
// imaginary planes, one fused launch. The A and B device pointer arrays are
// dereferenced in-kernel.
void cgemm_batched_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR ptrA, GM_ADDR ptrB, GM_ADDR dstAr, GM_ADDR dstAi, GM_ADDR dstBr,
    GM_ADDR dstBi, CgbtSplitTilingData ta, CgbtSplitTilingData tb, GM_ADDR inputFlags, float riskScale);

// Phase 1 (AIC): the four real GEMM products (T1..T4) of every batch member,
// fused into one launch per K-chunk.
void cgemm_batched_gemm_kernel_do(uint32_t blockDim, void* stream, GM_ADDR workspace, CgbtGemmTilingData tiling);

// Phase 2 (AIV): assemble the complex result, scale, write back per batch.
void cgemm_batched_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR workspace, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray,
    CgbtCombineTilingData tiling, CgbtDirectTilingData direct);

// Early exit (AIV): C[i] = beta * C[i] per batch.
void cgemm_batched_beta_scale_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR carray, CgbtBetaScaleTilingData tiling);

void cgemm_batched_direct_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, CgbtDirectTilingData tiling);

// The compile-time singleK of the Matmul static tiling, so the host can split K
// without duplicating the constant.
uint32_t cgemm_batched_gemm_single_k();

void cgemm_batched_fused_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace,
    CgbtFusedTilingData tiling);

int cgemm_batched_vector16_kernel_do(
    uint32_t vectors, void* stream, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray,
    CgbtDirectTilingData tiling, void*& singleFunction, void*& groupedFunction);

int cgemm_batched_small_kernel_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
    CgbtDirectTilingData tiling, void*& function);

#endif // CGEMM_BATCHED_ARCH22_KERNEL_H
