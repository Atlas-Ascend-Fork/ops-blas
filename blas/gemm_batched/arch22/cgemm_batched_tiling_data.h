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
 * \file cgemm_batched_tiling_data.h
 * \brief Tiling data for arch22 (Atlas A2/A3) aclblasCgemmBatched.
 *
 *        C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i] over complex64 batch
 *        members, computed with the four-product decomposition
 *          A = Ar + j*Ai,  B = Br + j*Bi
 *          Pr = Ar*Br - Ai*Bi,  Pi = Ar*Bi + Ai*Br
 *          C  = alpha * (Pr + j*Pi) + beta * C_old
 *        carried out as four real FP32 GEMMs (T1 = Ar*Br, T2 = Ai*Bi,
 *        T3 = Ar*Bi, T4 = Ai*Br) through the pipeline
 *          Phase 0 (AIV) split  : A[i] -> (Ar, Ai), B[i] -> (Br, Bi)
 *          Phase 1 (AIC) gemm   : four real GEMMs, one launch per K chunk
 *          Phase 2 (AIV) combine: reassemble, scale, write back per batch
 *
 *        LAYOUT INVARIANTS (mirroring complex_blas3_arch22.h):
 *
 *        (1) Phase 0 output is tightly packed column-major. Read as row-major,
 *            a packed plane of a (rows x cols) physical matrix is the transpose
 *            (cols x rows). op() semantics fold into WHICH ORIENTATION the
 *            Matmul reads the plane with, never into a data pass:
 *              trans = N: the row-major view already is op(X)^T (plain read)
 *              trans = T/C: the view is op(X), so the Matmul reads it
 *                transposed (C's conjugation is applied during the split)
 *
 *        (2) Phase 1 computes D = C^T = op(B)^T * op(A)^T row-major
 *            (D is n x m), so the X operand is the packed B plane and the Y
 *            operand the packed A plane. A column of the column-major C is a
 *            row of D, which is what Phase 2 reads.
 *
 *        (3) Plane-major workspace layout (floats), every plane 512-byte
 *            aligned; base offsets are fixed, see the host planner:
 *              [Br * batches][Bi * batches][Ar * batches][Ai * batches]
 *              [T1 * batches][T2 * batches][T3 * batches][T4 * batches]
 *            so every kernel addresses its planes arithmetically
 *            (base + batch * planeStride + offset) and no pointer array is
 *            materialised for the workspace side. The caller-side Aarray /
 *            Barray / Carray device pointer arrays are dereferenced in-kernel.
 *
 *        Field layout must stay byte-identical between host and device.
 */

#ifndef CGEMM_BATCHED_TILING_DATA_H
#define CGEMM_BATCHED_TILING_DATA_H

#include <cstdint>

// Shared host/device layout units. A realified complex panel occupies four
// FP32 planes; each producer slot holds both operands plus a 512-byte flag area.
constexpr uint32_t CGBT_COMPLEX_COMPONENTS = 2U;
constexpr uint32_t CGBT_REALIFIED_OPERAND_PLANES = 4U;
constexpr uint32_t CGBT_SLOT_PLANES = 2U * CGBT_REALIFIED_OPERAND_PLANES;
constexpr uint32_t CGBT_SLOT_FLAG_FLOATS = 512U / sizeof(float);
// Range guards deliberately retain the validated thresholds and rounding.
constexpr float CGBT_SAFE_COMPONENT_MAGNITUDE = 1e18F;
// Approximately FLT_MAX / safe magnitude: overflow of scaled inputs marks risk.
constexpr float CGBT_INPUT_RISK_MULTIPLIER = 3.402823466e20F;

// CgbtGemmTilingData::transMode bits. bit0: the X operand (= packed op(B)^T
// plane) is stored transposed and the Matmul must read it transposed
// (transb != N); bit1: same for Y (= packed op(A)^T plane) when transa != N.
//
// Packed operands keep their original FP32 range. Scaling would change the
// overflow semantics of the complex multiplication. Extreme small outputs use
// the direct path's fused product helper to preserve Inf/NaN classification.
constexpr uint32_t CGBT_TRANS_X = 1U;
constexpr uint32_t CGBT_TRANS_Y = 2U;

// Phase 0: de-interleave one operand of every batch member into packed real /
// imaginary planes. One fused launch for A and B.
struct CgbtSplitTilingData {
    uint32_t batchCount;  // batches covered by this launch (the host chunks)
    uint32_t rows;        // physical rows of the source matrix
    uint32_t cols;        // physical cols of the source matrix
    uint32_t ld;          // source leading dimension, complex elements
    uint32_t packedLd;    // row stride of the packed planes, floats (== rows)
    uint32_t negateImag;  // 1: store -imag (op == C conjugates)
    uint64_t planeFloats; // per-batch plane stride, floats (512-byte aligned)
};

// Phase 1: the four real GEMM products of every batch member, fused into ONE
// launch. The task space is product-major:
//   task = p * (batchCount * tilesPerBatch) + batch * tilesPerBatch + tile,
// and each product's operand planes follow from p alone:
//   p=0: T1 = Br*Ar   p=1: T2 = Bi*Ai   p=2: T3 = Bi*Ar   p=3: T4 = Br*Ai
// (X is the packed B plane, Y the packed A plane; see the header comment.)
// Fusing removes three launch overheads and, more importantly, three
// wave-quantisation tails: four sequential launches each pay a partial last
// wave, one launch pays it once.
constexpr uint32_t CGBT_PRODUCTS = 4U;

struct CgbtGemmTilingData {
    uint32_t m;        // rows of D = C^T (== n of C)
    uint32_t n;        // cols of D = C^T (== m of C)
    uint32_t k;        // full reduction length
    uint32_t kBase;    // K-chunk offset
    uint32_t kCount;   // K-chunk length, <= the kernel's compile-time singleK
    uint32_t tileSize; // output extent assigned to one Matmul call
    uint32_t mBlocks;  // ceil(m / tileSize)
    uint32_t nBlocks;  // ceil(n / CBLAS3_TILE_N)
    uint32_t batchCount;
    uint32_t enAtomic;  // 0 = overwrite the temp plane, 1 = accumulate (K-chunk > 0)
    uint32_t transMode; // bit0: X (= op(B)^T plane) read transposed (transb != N);
                        // bit1: Y (= op(A)^T plane) read transposed (transa != N)
    uint32_t ldX;       // packed B plane row stride, floats
    uint32_t ldY;       // packed A plane row stride, floats
    uint32_t ldD;       // D row stride, floats (padded m of C)
    uint64_t offXr;     // float offset of the Br plane
    uint64_t offXi;     // float offset of the Bi plane
    uint64_t offYr;     // float offset of the Ar plane
    uint64_t offYi;     // float offset of the Ai plane
    uint64_t offT1;     // float offset of T1; T_p sits at offT1 + p*batchCount*tPlaneFloats
    uint64_t xPlaneFloats;
    uint64_t yPlaneFloats;
    uint64_t tPlaneFloats;
};

// Phase 2: Pr = T1 - T2, Pi = T3 + T4, complex alpha/beta, interleave,
// write back. Reads Carray in-kernel for the beta * C_old term.
struct CgbtCombineTilingData {
    uint32_t m; // rows of C
    uint32_t n; // cols of C
    uint32_t ldc;
    uint32_t batchCount;
    uint32_t ldD; // D row stride, floats
    uint32_t isBetaZero;
    uint32_t inputFlagCount; // splitter cores for a standalone combine
    float inputRiskScale;
    uint32_t preserveBetaOne; // Netlib skips beta=1 scaling only when transA=N.
    float alphaRe;
    float alphaIm;
    float betaRe;
    float betaIm;
    uint64_t offT1;
    uint64_t offT2;
    uint64_t offT3;
    uint64_t offT4;
    uint64_t tPlaneFloats;
    uint64_t wsFloats; // total workspace length, floats (GlobalTensor bound)
};

// Early exit (k = 0 or alpha = (0,0)): C[i] = beta * C[i] per batch, bit-exact.
struct CgbtBetaScaleTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;
    uint32_t batchCount;
    uint32_t isBetaZero;
    float betaRe;
    float betaIm;
};

struct CgbtDirectTilingData {
    uint32_t m, n, k, lda, ldb, ldc, batchCount, transA, transB;
    float alphaRe, alphaIm, betaRe, betaIm;
};

struct CgbtFusedTilingData {
    uint64_t offRangeFlags; // two range maxima per batch, after the product planes
    CgbtSplitTilingData a, b;
    CgbtGemmTilingData gemm;
    CgbtCombineTilingData combine;
};

#endif // CGEMM_BATCHED_TILING_DATA_H
