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
 * \file ctrmm_tiling_data.h
 * \brief CTRMM tiling data for ascend950 (DAV_3510) — M2 correctness baseline.
 *
 * The M2 kernel is a single AIV SIMT pass over the output matrix (naive 4M
 * complex multiply-add, no mirror/GEMM pipeline, no workspace). The M3 expert
 * kernel (three-phase AIV-AIC-AIV + Gauss 3M + tile-level k truncation) will
 * extend this structure; the expert handover surface is documented in the
 * ctrmm_kernel.cpp header comment.
 */

#include <cstdint>

struct CtrmmTilingData {
    // Enum modes (values of aclblasSideMode_t / aclblasFillMode_t /
    // aclblasOperation_t / aclblasDiagType_t). The kernel dispatches to
    // compile-time template instantiations; no runtime enum branching inside
    // the compute loop.
    uint32_t sideMode;
    uint32_t uploMode;
    uint32_t transMode;
    uint32_t diagMode;
    // Problem dimensions (column-major):
    //   side = LEFT  : A is m x m (array lda x m),  B/C are m x n
    //   side = RIGHT : A is n x n (array lda x n),  B/C are m x n
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    // AIV partition: block b handles rows [b*rowsPerCore, min((b+1)*rowsPerCore, m)),
    // SIMT threads inside a block stride over those rows.
    uint32_t usedAivCoreNum;
    uint32_t rowsPerCore;
    uint32_t nthreads;
    // Complex alpha resolved on host (device-pointer alpha is read back via D2H
    // before launch, mirroring aclblasStrmm's ResolveStrmmScalars).
    float alphaReal;
    float alphaImag;
    // TINY fast path: 1 = B and C are the same buffer (two-pass + barrier),
};

// ── M3 4M all-device pipeline (three phases, cherk form) ─────────────────────
// Phase-A (AIV SIMT): complex A -> mirrored op(A) real/imag planes (UNIT diag
// written as 1+0i, non-referenced triangle zeroed, trans=C negates the imag
// plane) + complex B -> real/imag planes.
struct CtrmmSplitMirrorTilingData {
    uint32_t uploMode;
    uint32_t transMode;
    uint32_t diagMode;
    uint32_t dimA; // order of A (m for LEFT, n for RIGHT)
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t rowsPerCoreA;
    uint32_t rowsPerCoreB;
    uint32_t nthreads;
    // >0 = B-side column-band partition (thin shapes: row bands fragment to
    // < TLA_BLOCK rows, multiplying tile count and paying full MTE latency per
    // fragment; a full-height column band restores proper 64x64 tiles). Only
    // the fused kernel (trans = N) honors it; 0 = legacy row bands.
    uint32_t colsPerCoreB;
    // 1 = this launch handles the A-side mirror only (the B-side split runs in
    // the tensor_api kernel ctrmm_split_b_tla_kernel_do).
    uint32_t skipB;
};

// Phase-B (AIC): tensor_api cube kernel on the real plane products (merged launch).
// Phase-C (AIV SIMT): combine P1..P3 with complex alpha into C (P4 slot reserved).
struct CtrmmCombineTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;
    uint32_t tempRowStride; // leading stride of the P planes
    uint32_t rowsPerCore;
    uint32_t nthreads;
    float alphaReal;
    float alphaImag;
};

// Phase-B alternative kernel (strmm lineage, L0 lever): same field set as
// StrmmGemmTilingData; built by the host with the strmm recipe (m/n swapped
// to the C^T view + side flipped, lda/ldb unswapped).
struct CtrmmPhaseBTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t sideMode;
    uint32_t usedAicCoreNum;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileKChunk;
    uint32_t lda;
    uint32_t ldb;
    uint32_t tempRowStride;
    // Triangular K-range clipping (L2 lever, catlass gemm/kernel/trmm.hpp
    // semantics: "clipping each output tile to the valid triangular K range,
    // inactive elements expected zeroed by the caller" — our Phase-A mirror
    // zeroes them). Expressed in THIS kernel's swapped/flipped view:
    //   0 = no clip
    //   1 = clip K-end   by N-tile end   (triangle in the kernel's right operand, LOWER plane)
    //   2 = clip K-start by N-tile start (kernel's right operand, UPPER plane)
    //   3 = clip K-start by M-tile start (kernel's left  operand, LOWER plane)
    //   4 = clip K-end   by M-tile end   (kernel's left  operand, UPPER plane)
    uint32_t clipMode;
};
