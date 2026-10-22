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
 * \file cherkx_tiling_data.h
 * \brief Tiling data structures for aclblasCherkx (arch35).
 *        Shared by host side (const ref) and kernel side (by value).
 *
 *        Phase 1 (AIC GEMM) reuses GemmTilingData from blas/gemm/arch35/gemm_tiling_data.h.
 *        Phase 0 and Phase 2 tiling data are defined here.
 *
 *        cherkx differs from cherk in that A != B (dual-matrix input) and
 *        alpha is complex. Phase 0 calls deinterleave twice (A and B).
 *        Phase 2 combine merges P_r = t1+t2 / P_i = t3−t4 with complex-alpha
 *        scaling (Cr = αr·P_r − αi·P_i, Ci = αr·P_i + αi·P_r), and the diagonal
 *        imaginary part is theoretically non-zero (A != B) and must be
 *        explicitly forced to 0.
 */

#pragma once

#include <cstdint>

// [general] SIMD mode parameters for AIV kernels
constexpr uint32_t CHERKX_ARCH35_SCALE_BLOCK = 64;       // scale block size (rows/cols per tile)
constexpr uint32_t CHERKX_ARCH35_ELEMENTS_PER_BLOCK = 8; // 32 bytes / sizeof(float)

// Phase 0: Deinterleave complex A or B → real, imag (AIV-only)
// Called twice: once for A → Ar, Ai; once for B → Br, Bi.
struct CherkxDeinterleaveTilingData {
    uint32_t rows;         // matrix physical rows (= n for trans=N, = k for trans=T)
    uint32_t cols;         // matrix physical cols (= k for trans=N, = n for trans=T)
    uint32_t lda;          // matrix leading dimension (in complex elements)
    uint32_t rowsPerCore;  // rows per AIV core
};

// Phase 2: Combine/Scale/Hermitian Tiling (AIV-only)
// Merges 4 real GEMM results (t1..t4) into complex C with alpha/beta scaling
// and Hermitian constraint. alpha is complex (cuBLAS cherkx contract; cherk
// has real alpha): with P_r = t1+t2 and P_i = t3−t4, the combine computes
// Cr = αr·P_r − αi·P_i, Ci = αr·P_i + αi·P_r (αi==0 keeps the real-alpha path).
struct CherkxCombineTilingData {
    uint32_t n;              // C matrix order
    uint32_t ldc;            // C matrix leading dimension (in complex elements)
    uint32_t tempLdc;        // temp matrix row stride (= CeilAlign(n, 16))
    uint32_t rowsPerCore;    // rows per AIV core
    float alphaReal;         // alpha real part
    float alphaImag;         // alpha imaginary part (0.0f for real alpha)
    float betaVal;           // beta (real)
    uint8_t uploMode;        // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint8_t isAlphaZero;     // alpha == 0 short-circuit (complex zero: real==0 && imag==0)
    uint8_t isKZero;         // k == 0 short-circuit
    uint8_t isBetaZero;      // beta == 0 short-circuit
};
