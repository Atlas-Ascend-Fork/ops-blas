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
 * \file cher2k_tiling_data.h
 * \brief Tiling data structures for aclblasCher2k (arch35).
 *        Shared by host side (const ref) and kernel side (by value).
 *
 *        Kernel inventory (design §1.4):
 *          K1 cher2k_deinterleave_kernel : AIV SIMD, splits A and B into Ar, Ai, Br, Bi (tiling below)
 *          K2 gemm_kernel_fp32           : AIC Cube launched four times, reuses GemmTilingData from
 *                                          blas/gemm/arch35/gemm_tiling_data.h (no new struct)
 *          K3 cher2k_combine_kernel      : AIV SIMD, forms M plus its conjugate transpose plus beta times old C (tiling
 * below) K4 cher2k_simt_small_kernel   : AIV SIMT, fused direct path for n up to 8 (tiling below)
 */

#pragma once

#include <cstdint>

// [general] SIMD mode parameters for AIV kernels
constexpr uint32_t CHER2K_ARCH35_SCALE_BLOCK = 64;       // scale block size (rows or cols per tile)
constexpr uint32_t CHER2K_ARCH35_ELEMENTS_PER_BLOCK = 8; // 32 bytes divided by the float size
// Small-n threshold: [P1-F1] for n up to 8 the fused SIMT direct path is taken
// (K4, no workspace, no Phase0/Phase1) to avoid the 6-launch overhead of the
// general pipeline. It was 16 through r2final; P1 lowers it to 8 so n in the
// (8,16] range now runs the K1+K2f+K3 pipeline (fused cube path) instead of the
// quadratic-in-n, linear-in-k SIMT dot loop — n from 9 to 16 are all K4-bound in
// the 200-case sweep (expect the mid/small n band to rise). n up to 8 stays on
// K4: below that the pipeline's fixed launch cost dominates and K4 is still the
// fastest option.
constexpr uint32_t CHER2K_ARCH35_SIMT_N_MAX = 8;
// K4 SIMT warp width: with n up to 8 there are at most 36 uplo elements per
// launch (for n of 8 that is 8 times 9 divided by 2, giving 36), so a single
// small thread group is enough (kept at the repo-common 128).
constexpr uint32_t CHER2K_ARCH35_SIMT_THREADS = 128;

// K1 / Phase 0: Deinterleave complex A and B into the real matrices Ar, Ai, Br, Bi tiling (AIV-only)
struct Cher2kDeinterleaveTilingData {
    uint32_t rows;        // physical rows of A and B (equals n in transpose mode N, k in mode C)
    uint32_t cols;        // physical cols of A and B (equals k in transpose mode N, n in mode C)
    uint32_t lda;         // A leading dimension (in complex elements)
    uint32_t ldb;         // B leading dimension (in complex elements)
    uint32_t rowsPerCore; // rows per core within one matrix partition (tileMode 0)
    uint32_t splitCore;   // cores below this index process A, the rest process B
    // [deint-2d] value 0 selects the legacy 1D row band (rowsPerCore), value 1
    // the 2D tile list. The row-band split starves cores when the row count is
    // below the split core count times 64 (for example transpose mode C with k
    // of 64 gives a single 64-row band: only 8 of 28 cores per matrix get any
    // row, 40 of 56 cores idle). Mode 1 enumerates the 64x64 tiles of the
    // matrix and hands each core (indexed by `localIdx`) the tiles whose index
    // leaves remainder `localIdx` when divided by the split core count, so every
    // core gets roughly the tile count divided by the split core count (0 idle).
    // Pure copy path: same elements, same bit-exact Ar/Ai/Br/Bi.
    uint8_t tileMode;
};

// K3 / Phase 2: combine tiling for forming M plus its conjugate transpose plus beta times old C (AIV-only)
// M is alpha times the complex number whose real part PR is t1 plus t2 and whose
// imaginary part PI is t3 minus t4 (t1..t4 from the 4 real GEMMs)
// [L2] Two allocation modes for the uplo triangle, selected by the host
// (TuneCombineCoreNum):
//   mode 0 (legacy row-band, default): core b owns rows from b times rowsPerCore
//     up to the smaller of (b+1) times rowsPerCore and n — the pre-L2 rule, used
//     for every order where n is not a multiple of 64 or is small. Zero drift vs
//     the pre-L2 binary: the per-core tile walk (Cher2kProcessCombineLoop) is
//     unchanged.
//   mode 1 (triangular 2D tile list): the uplo triangle is enumerated as 64x64
//     tiles, of which there are tileTotal in all; each core owns the tiles whose
//     index leaves its own core number as remainder when divided by the core
//     count. Every tile is visited by exactly one core, partner tiles are read
//     from GM directly (no cross-core dependency), so any tile-to-core bijection
//     is numerically equivalent to the row-band walk (each
//     Cher2kProcessCombineBlock call depends only on its own tile origin and the
//     GM state). The 2D spread removes the row-band load imbalance: the UPPER
//     first row band is n divided by 64 tiles long while the last is 1, so a
//     16-core n of 1024 launch leaves the last cores nearly idle (16-tile
//     critical path); the tile list spreads 136 tiles over up to 56 cores
//     (2.4 tiles per core, critical path 3 tiles).
struct Cher2kCombineTilingData {
    uint32_t n;           // C matrix order
    uint32_t ldc;         // C matrix leading dimension (in complex elements)
    uint32_t tempLdc;     // temp matrix row stride (n rounded up to a multiple of 16)
    uint32_t rowsPerCore; // rows per AIV core (mode 0: row band height)
    float alphaReal;      // alpha real part
    float alphaImag;      // alpha imaginary part
    float betaVal;        // beta (real)
    uint8_t uploMode;     // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint8_t isAlphaZero;  // when alpha is the zero complex value the computation short-circuits (skips t1..t4)
    uint8_t isKZero;      // when k is zero the computation short-circuits (skips t1..t4)
    uint8_t isBetaZero;   // when beta is zero the computation short-circuits (does not read the old C)
    uint8_t isSimt;       // reserved, no consumer yet: SIMT combine variant
                          // switch (kept 0; M3/M6 placeholder, do not repurpose
                          // without extending the struct)
    uint8_t tileMode;     // [L2] value 0 selects the legacy row band, value 1 the triangular tile list
    uint8_t numCores;     // [L2] mode 1: launched core count (tile stride)
    uint16_t tileTotal;   // [L2] mode 1: number of uplo 64x64 tiles (at most 65535,
                          // the uint16 field limit: the triangular count grows as
                          // q times q plus 1 over 2 and stays within 65535 while
                          // q, the n rounded up to a multiple of 64, covers n up
                          // to 23104; the host guards larger n to mode 0)
    // [MIX fold] value 1 means the K2f MIX kernel already folded t1..t4 into
    // PR/PI on chip, so the combine receives PR in t1 and PI in t3 and must NOT
    // run Cher2kSumPrPi again (t2 and t4 are not read). Value 0 means the legacy
    // 4-frame input.
    uint8_t isPrPiFolded;
    // [iter34] alpha/beta source selectors. Value 1 means the alphaGm/betaGm
    // KERNEL ARGUMENTS are DEVICE pointers and the kernel reads the scalar from
    // GM (no host device-to-host staging plus stream sync — that cost about
    // 24us per call inside the timed window). Value 0 means the pointers were
    // HOST pointers; the host read them once and the kernel uses the
    // alphaReal/alphaImag/betaVal fields instead (identical to the pre-iter34
    // host-dereference behaviour; the mirror/quick-return unit tests pass host
    // pointers).
    uint8_t isAlphaDev;
    uint8_t isBetaDev;
};

// [stage2 Step 3a] Combine parameter channel for the K2f MIX kernel.
// Threads the K3 combine inputs (C pointer, ldc, alpha/beta, uplo and the
// zero/short-circuit flags) down to the AIV side of the MIX kernel so a later
// step can fuse the combine in place. This step is a PURE PARAMETER PIPE: the
// MIX kernel only stores it into FusedState and does NOT consume it, so the
// default behaviour is byte-for-byte identical to the current binary. The
// fields mirror the corresponding Cher2kCombineTilingData members 1:1.
struct Cher2kCombineParams {
    uint8_t* cGm;        // C matrix base address (GM_ADDR)
    uint32_t ldc;        // C matrix leading dimension (in complex elements)
    uint32_t tempLdc;    // temp matrix row stride (n rounded up to a multiple of 16)
    float alphaReal;     // alpha real part
    float alphaImag;     // alpha imaginary part
    float betaVal;       // beta (real)
    uint8_t uploMode;    // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint8_t isAlphaZero; // when alpha is the zero complex value the computation short-circuits
    uint8_t isKZero;     // when k is zero the computation short-circuits
    uint8_t isBetaZero;  // when beta is zero the computation short-circuits (does not read the old C)
};

// [MIX fold] K2f MIX kernel: AIC computes the 4 cube accumulators and hands
// them to the AIV through UB (CopyL0C2UB, no GM round trip). foldMode selects
// what the AIV does with them:
//   CHER2K_FOLD_STORE4 : write t1..t4 straight to GM (M1 — bit-identical to the
//                        legacy fixpipe output, used as the MIX handoff gate).
//   CHER2K_FOLD_PRPI   : write PR as t1 plus t2 and PI as t3 minus t4 to GM (M2 — the
//                        combine then only reads 2 frames).
constexpr uint32_t CHER2K_FOLD_STORE4 = 0;
constexpr uint32_t CHER2K_FOLD_PRPI = 1;
// Sentinel for the legacy AIC-only fixpipe path (no UB handoff). Any value
// other than this selects the MIX kernel.
constexpr uint32_t CHER2K_FOLD_LEGACY = 0xFFFFFFFFU;
// [stage2 Step 2] Pair-walk enable bit, OR-ed into the foldMode argument of the
// MIX kernel so the AIC/AIV tile walk can switch from the mi×ni full grid to the
// symmetric tile-PAIR walk (design §2.3) WITHOUT changing the launch signature
// (which lives in cher2k_kernel.h, out of this step's edit scope). The kernel
// masks this bit off before comparing the foldMode with the CHER2K_FOLD_*
// sink selectors, so the bit is orthogonal to the sink choice. Default: unset
// (grid walk, byte-for-byte unchanged).
constexpr uint32_t CHER2K_FOLD_PAIRWALK_BIT = 0x100U;
// [stage2 Step 2] Pair-walk uplo bit (only meaningful together with
// CHER2K_FOLD_PAIRWALK_BIT). Set selects the LOWER triangle enumeration order,
// clear the UPPER. The pair SET is uplo-independent (design §2.5 — it covers the whole
// qc² grid either way), so this bit only fixes the emission ORDER; it is
// threaded through foldMode because the fused tiling has no uplo field.
constexpr uint32_t CHER2K_FOLD_LOWER_BIT = 0x200U;

// K4: small-n fused SIMT direct path (n up to CHER2K_ARCH35_SIMT_N_MAX, no workspace)
struct Cher2kSimtTilingData {
    uint32_t n;           // C matrix order
    uint32_t k;           // reduction length
    uint32_t ldc;         // C matrix leading dimension (in complex elements)
    uint32_t lda;         // A leading dimension (in complex elements)
    uint32_t ldb;         // B leading dimension (in complex elements)
    uint32_t rowsPerCore; // rows per AIV core
    uint32_t transN;  // value 1 means transpose mode OP_N, where A and B are n by k; value 0 means OP_C, where they are
                      // k by n
    float alphaReal;  // alpha real part
    float alphaImag;  // alpha imaginary part
    float betaVal;    // beta (real)
    uint8_t uploMode; // ACLBLAS_UPPER / ACLBLAS_LOWER
    // [iter34] alpha/beta source selectors (see Cher2kCombineTilingData).
    // A value of 1 for isAlphaDev/isBetaDev means the kernel reads the scalar
    // from the alphaGm/betaGm KERNEL ARGUMENTS (device pointers); a value of 0
    // means the kernel uses alphaReal/alphaImag/betaVal (host-dereferenced once
    // by the host).
    uint8_t isAlphaDev;
    uint8_t isBetaDev;
};

// --------------------------------------------------------------------------
//  [stage2 Step 1] Symmetric tile-PAIR enumeration on the 64×64 combine grid.
//  Design ref: stage2-mix-combine-design.md §2.2 (geometry) / §2.3 (pairing) /
//  §2.6 (load balancing). This section is PURE COMPUTATION — zero signal
//  primitives, no UB/GM access, no cross-core state — so both the host and the
//  (future) AIC/AIV tile walk can call it and independently re-derive the exact
//  same pair sequence and per-core quota without any communication.
//
//  Two grids (design §2.2): the triangle/pairing is defined on the 64×64
//  COMBINE grid (qc, the n rounded up to a multiple of 64, in both directions
//  for the square Cher2k), NOT on the fused bm×bn grid — the two are
//  non-isomorphic when bn is 32 (nLoopCount is twice mLoopCount). A combine tile
//  (I,J) maps to fused tiles via Cher2kPairFusedCols() below.
//
//  Pairing rule (design §2.3): the diagonal (I,I) is a self-pair carrying ONE
//  product tile; every off-diagonal triangle tile (I,J) carries TWO product
//  tiles (itself and its symmetric partner (J,I)). Over the whole grid this
//  covers qc diagonals plus qc(qc-1)/2 off-diagonals, each counted twice, giving
//  qc² product tiles — the FULL qc×qc grid, not the triangle (design §2.5).
// --------------------------------------------------------------------------

// Host+device callable marker for the pure pair helpers below. The bisheng
// compiler compiles the host and kernel TUs with the same preprocessor macros
// and treats a plain `inline` as host-only and a bare `__aicore__` as
// device-only, so a helper used by BOTH needs the dual target attribute. Under a
// non-ASC compiler (e.g. the plain-CPU Step-1 verification build) it degrades to
// a plain `inline`.
#ifndef CHER2K_PAIR_HOSTDEV
#if defined(__CCE__) || defined(__CCE_AICORE__)
#define CHER2K_PAIR_HOSTDEV __attribute__((cce_aicore, cce_host)) inline
#else
#define CHER2K_PAIR_HOSTDEV inline
#endif
#endif

// The 64×64 combine-grid block edge (matches CHER2K_ARCH35_SCALE_BLOCK).
constexpr uint32_t CHER2K_PAIR_GRID_BLOCK = 64;
// Pair weights in PRODUCT-TILE units (design §2.6): the diagonal pair carries
// 1 product tile, an off-diagonal pair carries 2.
constexpr uint32_t CHER2K_PAIR_DIAG_WEIGHT = 1;
constexpr uint32_t CHER2K_PAIR_OFFDIAG_WEIGHT = 2;
// Extra weight of a LOWER FULL diagonal pair: it takes the scalar-restore store
// path (Cher2kStoreLowerDiagColumn, ~57us), so it is the load-balancing unit.
// Mirrors CHER2K_COMBINE_DIAG_WEIGHT in cher2k_kernel.cpp (design §2.6).
constexpr uint32_t CHER2K_PAIR_W_DIAG = 9;

// One entry of the pair sequence. `i`/`j` are 64×64 combine-grid indices; for
// an off-diagonal pair (i,j) is the in-triangle representative and (j,i) is its
// symmetric partner. `weight` is the product-tile count (1 or 2); `heavy` marks
// a LOWER full diagonal (carries the extra W_DIAG in the balancing total).
struct Cher2kTilePair {
    uint16_t i;
    uint16_t j;
    uint8_t weight; // product tiles covered by this pair (1 diag / 2 off-diag)
    uint8_t heavy;  // nonzero marks a LOWER full diagonal (adds CHER2K_PAIR_W_DIAG)
};

// Number of pairs in the uplo triangle: qc diagonals + qc(qc-1)/2 off-diagonals.
// (equals the classic triangular count qc(qc+1)/2).
CHER2K_PAIR_HOSTDEV uint32_t Cher2kPairCount(uint32_t qc) { return qc * (qc + 1U) / 2U; }

// A LOWER diagonal is "heavy" when it is a FULL 64×64 block; with a tail band
// the last diagonal (the one whose index is qc minus one) is clipped to a block
// whose row and col counts are equal and below 64 and is cheap, so it is NOT
// counted (design §2.6, mirrors Cher2kProcessCombineTileList).
CHER2K_PAIR_HOSTDEV bool Cher2kPairDiagIsHeavy(uint32_t i, uint32_t qc, bool hasTail)
{
    return (!hasTail) || (i + 1U < qc);
}

// Pair `p` (0-based, in enumeration order) yields its (i,j) representative and weight.
// Enumeration (design §2.3): per row I ascending, the diagonal (I,I) first, then
// the off-diagonal tiles ascending (for upper, j runs from I+1 up to qc-1; for
// lower, from 0 up to I-1).
CHER2K_PAIR_HOSTDEV Cher2kTilePair Cher2kBuildPair(uint32_t qc, uint32_t p, bool upper, bool hasTail)
{
    Cher2kTilePair out{0, 0, 0, 0};
    uint32_t base = 0;
    for (uint32_t i = 0; i < qc; ++i) {
        const uint32_t offCount = upper ? (qc - 1U - i) : i; // off-diagonal pairs in row i
        const uint32_t rowCount = 1U + offCount;             // + the diagonal pair
        if (p < base + rowCount) {
            const uint32_t k = p - base; // zero selects the diagonal, otherwise the k-th off-diagonal
            out.i = static_cast<uint16_t>(i);
            if (k == 0U) {
                out.j = static_cast<uint16_t>(i);
                out.weight = static_cast<uint8_t>(CHER2K_PAIR_DIAG_WEIGHT);
                out.heavy = (!upper && Cher2kPairDiagIsHeavy(i, qc, hasTail)) ? 1U : 0U;
            } else {
                // for upper, the k-th off-diagonal has j equal to i plus k; for lower, j is k minus one (ranging 0 to
                // i-1).
                out.j = upper ? static_cast<uint16_t>(i + k) : static_cast<uint16_t>(k - 1U);
                out.weight = static_cast<uint8_t>(CHER2K_PAIR_OFFDIAG_WEIGHT);
                out.heavy = 0;
            }
            return out;
        }
        base += rowCount;
    }
    return out; // a p out of range yields the sentinel (0,0,0,0)
}

// Balancing weight of pair `p` in product-tile units (diag 1 / off-diag 2,
// +W_DIAG for a LOWER full diagonal).
CHER2K_PAIR_HOSTDEV uint32_t Cher2kPairWeight(uint32_t qc, uint32_t p, bool upper, bool hasTail)
{
    const Cher2kTilePair pr = Cher2kBuildPair(qc, p, upper, hasTail);
    return static_cast<uint32_t>(pr.weight) + (pr.heavy ? CHER2K_PAIR_W_DIAG : 0U);
}

// Total balancing weight over the whole pair list is qc² product tiles plus the
// W_DIAG surcharge per heavy (LOWER full diagonal) pair (design §2.6).
CHER2K_PAIR_HOSTDEV uint32_t Cher2kPairTotalWeight(uint32_t qc, bool upper, bool hasTail)
{
    const uint32_t numHeavy = upper ? 0U : (hasTail ? (qc - 1U) : qc);
    return qc * qc + numHeavy * CHER2K_PAIR_W_DIAG;
}

// Upper bound on the cores the running cursor tracks. The hardware AIV count is
// at most 65; 128 is a safe ceiling for the pair-walk scalar state.
constexpr uint32_t CHER2K_PAIR_MAX_CORES = 128;

// Greedy water-filling rule (design §2.6): hand each pair, in enumeration order,
// to the currently LEAST-LOADED core (ties go to the lowest core index). This is
// the classic water-filling assignment and the best achievable balance for
// indivisible pairs — the per-core load spread is provably bounded by the
// largest single pair weight (a heavier core is always the one that just took a
// big pair). No cross-core communication: every core replays the same
// deterministic assignment locally.
CHER2K_PAIR_HOSTDEV uint32_t Cher2kPairLeastLoaded(const uint32_t* loads, uint32_t numCores)
{
    uint32_t best = 0U;
    for (uint32_t c = 1U; c < numCores; ++c) {
        if (loads[c] < loads[best]) {
            best = c;
        }
    }
    return best;
}

// Running cursor over the pair list: the per-core running load plus the next
// pair index. A core walks the cursor in enumeration order and processes the
// pairs it owns (design §2.6, "the running cursor walks the pair list").
struct Cher2kPairCursor {
    uint32_t loads[CHER2K_PAIR_MAX_CORES];
    uint32_t next;
};

CHER2K_PAIR_HOSTDEV void Cher2kPairCursorInit(Cher2kPairCursor& cur, uint32_t numCores)
{
    const uint32_t n = (numCores < CHER2K_PAIR_MAX_CORES) ? numCores : CHER2K_PAIR_MAX_CORES;
    for (uint32_t c = 0U; c < n; ++c) {
        cur.loads[c] = 0U;
    }
    cur.next = 0U;
}

// Advance the cursor by one pair, write it to `out` and return its owner core.
// Caller must keep `cur.next < Cher2kPairCount(qc)`.
CHER2K_PAIR_HOSTDEV uint32_t Cher2kPairCursorNext(
    Cher2kPairCursor& cur, uint32_t qc, bool upper, bool hasTail, uint32_t numCores, Cher2kTilePair& out)
{
    out = Cher2kBuildPair(qc, cur.next, upper, hasTail);
    const uint32_t owner = Cher2kPairLeastLoaded(cur.loads, numCores);
    cur.loads[owner] += Cher2kPairWeight(qc, cur.next, upper, hasTail);
    cur.next += 1U;
    return owner;
}

// Owner core of pair `p` under the greedy water-filling rule. O(p) replay; use
// the cursor for a full O(numPairs) walk.
CHER2K_PAIR_HOSTDEV uint32_t Cher2kPairOwner(uint32_t qc, uint32_t p, bool upper, bool hasTail, uint32_t numCores)
{
    if (numCores == 0U) {
        return 0U;
    }
    Cher2kPairCursor cur;
    Cher2kPairCursorInit(cur, numCores);
    uint32_t owner = 0U;
    Cher2kTilePair pr{0, 0, 0, 0};
    for (uint32_t k = 0U; k <= p; ++k) {
        owner = Cher2kPairCursorNext(cur, qc, upper, hasTail, numCores, pr);
    }
    return owner;
}

// Total balancing weight owned by `core` (design §2.6).
CHER2K_PAIR_HOSTDEV uint32_t Cher2kPairCoreLoad(uint32_t qc, bool upper, bool hasTail, uint32_t numCores, uint32_t core)
{
    if (numCores == 0U) {
        return 0U;
    }
    Cher2kPairCursor cur;
    Cher2kPairCursorInit(cur, numCores);
    const uint32_t numPairs = Cher2kPairCount(qc);
    uint32_t load = 0U;
    Cher2kTilePair pr{0, 0, 0, 0};
    for (uint32_t p = 0U; p < numPairs; ++p) {
        const uint32_t owner = Cher2kPairCursorNext(cur, qc, upper, hasTail, numCores, pr);
        if (owner == core) {
            load += Cher2kPairWeight(qc, p, upper, hasTail);
        }
    }
    return load;
}

// Assembly mapping for a fused block width bn of 32 (design §2.2/§2.4): a 64×64
// combine column J is assembled from the n-adjacent fused columns 2J and 2J+1 of
// width 32. When bn is 64 the mapping is one-to-one (fused column J). Returns
// the first fused column and its count.
struct Cher2kFusedCols {
    uint32_t first; // first fused n-column index
    uint32_t count; // number of adjacent fused n-columns (1 when bn is 64, 2 when bn is 32)
};

CHER2K_PAIR_HOSTDEV Cher2kFusedCols Cher2kPairFusedCols(uint32_t combineCol, uint32_t bn)
{
    Cher2kFusedCols out{combineCol, 1U};
    if (bn == 32U) {
        out.first = 2U * combineCol;
        out.count = 2U;
    }
    return out;
}
