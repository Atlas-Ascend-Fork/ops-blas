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
 * \file cherk_tiling_data.h
 * \brief Tiling data structures for aclblasCherk (arch35).
 *        Shared by host side (const ref) and kernel side (by value).
 *
 *        Phase 1 (AIC GEMM) reuses GemmTilingData from blas/gemm/arch35/gemm_tiling_data.h.
 *        Phase 0 and Phase 2 tiling data are defined here.
 */

#pragma once

#include <cstdint>

// [general] SIMD mode parameters for AIV kernels
constexpr uint32_t CHERK_ARCH35_SCALE_BLOCK = 64;       // scale block size (rows/cols per tile)
constexpr uint32_t CHERK_ARCH35_ELEMENTS_PER_BLOCK = 8; // 32 bytes / sizeof(float)

// Phase 0: Deinterleave A (complex) → Ar, Ai (real) Tiling (AIV-only)
struct CherkDeinterleaveTilingData {
    uint32_t rows;         // A physical rows (= n for trans=N, = k for trans=T)
    uint32_t cols;         // A physical cols (= k for trans=N, = n for trans=T)
    uint32_t lda;          // A leading dimension (in complex elements)
    uint32_t rowsPerCore;  // rows per AIV core
};

// Phase 2: Combine/Scale/Hermitian Tiling (AIV-only)
struct CherkCombineTilingData {
    uint32_t n;              // C matrix order
    uint32_t ldc;            // C matrix leading dimension (in complex elements)
    uint32_t tempLdc;        // temp matrix row stride (= CeilAlign(n, 16))
    uint32_t rowsPerCore;    // rows per AIV core
    float alphaVal;          // alpha (real)
    float betaVal;           // beta (real)
    uint8_t uploMode;        // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint8_t isAlphaZero;     // alpha == 0 short-circuit
    uint8_t isKZero;         // k == 0 short-circuit
    uint8_t isBetaZero;      // beta == 0 short-circuit
};
