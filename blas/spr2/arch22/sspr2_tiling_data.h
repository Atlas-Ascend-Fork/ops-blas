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
 * \file sspr2_tiling_data.h
 * \brief Tiling data structures for aclblasSspr2 (arch22 / Ascend910B3, DAV_2201).
 *        Shared by host side and kernel side (passed by value, POD).
 *
 *        sspr2: A = alpha * x * y^T + alpha * y * x^T + A, FLOAT32, column-major
 *        packed AP (length n*(n+1)/2, only the uplo triangle stored, in-place
 *        update, no beta / no lda).
 *
 *        Packed index:
 *          UPPER: k = i + j*(j+1)/2          (j >= i) -> column j is a contiguous
 *                 segment [j*(j+1)/2, j*(j+1)/2 + j]
 *          LOWER: k = i + j*(2n-j-1)/2       (i >= j) -> column j is a contiguous
 *                 segment [j*(2n-j+1)/2, j*(2n-j+1)/2 + n-j-1]
 *                 (Netlib sspr2.f / cblas_sspr2 column-major packed layout,
 *                 not the row-major formula k = i*(i+1)/2 + j. Column start
 *                 j*(2n-j+1)/2 = j + j*(2n-j-1)/2 = j*n - j*(j-1)/2 is the sum
 *                 of the preceding column lengths n-m; the product j*(2n-j+1)
 *                 is always even and fits uint64 with huge margin for n <= 4096.)
 *        Both uplo modes traverse packed column j (UPPER length j+1, LOWER
 *        length n-j) and differ in the segment base/length formulas and in
 *        which vector (x or y) pairs with which segment scalar.
 */

#ifndef SSPR2_ARCH22_TILING_DATA_H
#define SSPR2_ARCH22_TILING_DATA_H

#include <cstdint>

// [general] arch22 (DAV_2201) tuning / shape constants
inline constexpr uint32_t SSPR2_ARCH22_MAX_N = 4096;    // shape bound: n in [0, 4096]
// Chunk granularity in elements. 2048 keeps the chunk buffers + LOWER aligned
// slice buffers inside the 184KB UB budget for the largest n (worst-case
// footprint ~= 129KB for n=4096).
inline constexpr uint32_t SSPR2_ARCH22_SEG_TILE = 2048;
inline constexpr uint32_t SSPR2_ARCH22_ELE_PER_BLOCK = 8; // 32 bytes / sizeof(float)
// Max |inc| exercised by the ST suite (+-1/+-2/+-3) and the worst-case corner
// of the length bound at n == MAX_N (1+4095*3 = 12286 <= 12288). The
// ENFORCED bound is the length formula 1+(n-1)*|inc| <=
// SSPR2_ARCH22_MAX_VEC_LEN (host-validated), NOT this constant: for n < MAX_N
// the shape contract allows |inc| > 3 up to the length bound (e.g. n == 2
// admits |inc| <= 12287), so an |inc|-only gate would reject legal inputs.
inline constexpr uint32_t SSPR2_ARCH22_MAX_ABS_INC = 3; // ST-suite coverage: |incx|/|incy| in {1,2,3}
// Shape bound: logical vector length 1+(n-1)*|inc| <= 12288
// (LX/LY range [0, 12288]). Host validates this exact formula in int64;
// n == 0 evaluates the formula to 1-|inc| <= 12288 for every |inc|, so the
// legal no-op quick return stays intact. For n >= 2 the bound also pins
// (n-1)*|inc| <= 12287, keeping every kernel-side uint32 physical-window
// product (axU*(n-1), pos) in the no-wraparound domain.
inline constexpr uint32_t SSPR2_ARCH22_MAX_VEC_LEN = 12288;
// With the contiguous-slab protocol the per-column fixed cost is small (~2
// vector issues + scalar head), while the per-slab fixed cost
// (shifted-source builds + copy-in/out + 3 flag pairs) dominates. The weight
// therefore balances COLUMN counts (each column pays the in-UB per-column
// work) with a large C so that per-core SLAB counts (element volumes of the
// long-column end) stay bounded; C=8192 caps the worst-case per-core slab
// count at ~4 for n <= 4096 on 48 AIVs.
inline constexpr uint32_t SSPR2_ARCH22_SPLIT_CHUNK_COST = 8192;

// Small-n core thrift: block dispatch and per-core fixed cost scale with
// usedCoreNum while the per-core vector work of a mid-size n stays small, so
// mid-size n keeps every core's fixed cost under one wave with FEWER cores.
// The host caps usedCoreNum = min(cores, n, ceil(n / CORE_COLS_TARGET));
// with the target of 16 columns per core: n=512 -> 32 cores, n>=768 ->
// GetAivCoreCount() (= 48), tiny n keeps usedCoreNum = n. On mid-size n, 32
// cores is never slower than the full 48.
inline constexpr uint32_t SSPR2_ARCH22_CORE_COLS_TARGET = 16;
// Contiguous-slab kernel constants. The slab capacity is derived AT KERNEL
// RUNTIME from this usable UB budget minus the resident buffers
// (xPacked/yPacked and the two shifted-source buffers): one contiguous AP
// slab region serves both uplo modes; see sspr2_kernel.cpp.
// 184320B = 180KB, inside the NPU TPipe pool (UB nominal 256KB/core, usable
// 248KB) with margin; every n <= 4096 / packMode stays within.
inline constexpr uint32_t SSPR2_ARCH22_UB_BUDGET_BYTES = 184320;
// Upper bound of one slab region (elements). 32768 elems = 128KB keeps the
// single slab region inside the budget for every n <= 4096.
inline constexpr uint32_t SSPR2_ARCH22_SLAB_CAP_MAX = 32768;
// Per-DataCopyPad unit cap (elements): keeps one copy block at <= 32KB, safely
// inside the DataCopyPad per-block length envelope for BOTH transfer
// directions (UB->GM blockLen is the tighter of the two in the API doc).
inline constexpr uint32_t SSPR2_ARCH22_SLAB_CHUNK_MAX = 8192;

// Tiling struct. Passed to the kernel BY VALUE; broadcast to all
// cores, so it carries no per-core data: per-core segment ranges are derived
// in-kernel from GetBlockIdx() with Sspr2CoreSegStart (same formula as host).
//   alphaVal: alpha is a HOST-memory pointer; the host dereferences it and the
//             value is shipped in the tiling struct. n=0 / alpha=0 never
//             reach the kernel (host
//             short-circuit, no launch).
//   packMode: 0 (default) = scalar/direct packing of x/y (safe path);
//             1 = GatherMask stride-2 fast path probe (disabled in the
//                 production configuration; falls back to 0).
struct Sspr2TilingData {
    int64_t n;             // matrix order (AP length n*(n+1)/2, x/y logical length n)
    int32_t uploMode;      // ACLBLAS_UPPER(121) / ACLBLAS_LOWER(122), redundant with UPLO template arg
    int64_t ax;            // |incx|, x packing stride
    int64_t ay;            // |incy|, y packing stride
    uint8_t xIsNeg;        // 1: incx < 0 (Netlib reverse-order traversal absorbed at pack time)
    uint8_t yIsNeg;        // 1: incy < 0
    uint32_t usedCoreNum;  // cores actually enabled (= min(GetAivCoreCount(), n), dynamic)
    uint32_t segTile;      // filled with SSPR2_ARCH22_SEG_TILE; the kernel derives
                           // its slab capacity at runtime, so the field is carried
                           // for tiling-struct ABI / UT-mirror stability only
    float alphaVal;        // alpha read from the host pointer on host
    uint8_t packMode;      // x/y packing path selector (see above); TAIL-APPENDED
                           // (new fields are appended at the struct tail only,
                           // never inserted mid-struct, keeping the
                           // prefix layout stable)
    uint32_t classMode;    // class-major path selector, TAIL-APPENDED:
                           // 0 (default) = residual-class slab path;
                           // 1 = class-major core assignment (host sets it only
                           // for LOWER + direct-inc + n < 1536; kernel re-guards
                           // the direct-inc prerequisite before routing).
};


// Integer sqrt without libm (device-safe; the kernel calls it, the host may
// reuse it for cross-checking): Newton iterations. The convergence loop and
// both fix-up loops are shared with the kernel-side seeded variant
// (Sspr2IsqrtU64Seeded, sspr2_kernel.cpp) through this single
// Sspr2IsqrtConverge core. The monotone stopping condition t < s yields the
// same iteration sequence and terminal state as the classic break form over
// the whole domain. Precondition: v != 0 and the seed s is a positive Newton
// seed.
__aicore__ inline uint64_t Sspr2IsqrtConverge(uint64_t v, uint64_t s)
{
    // Divisor guard: both callers pass a positive seed and reject v == 0
    // upstream, so this branch is unreachable in practice; s = v is the
    // cold-start Newton seed, keeping the function total (no division by
    // zero for any input).
    if (s == 0) {
        s = v;
    }
    uint64_t t = (s + v / s) / 2ULL;
    while (t < s) {
        s = t;
        t = (s + v / s) / 2ULL;
    }
    while (s * s > v) {
        s--;
    }
    while ((s + 1ULL) * (s + 1ULL) <= v) {
        s++;
    }
    return s;
}

__aicore__ inline uint32_t Sspr2IsqrtU64(uint64_t v)
{
    if (v == 0) {
        return 0;
    }
    return static_cast<uint32_t>(Sspr2IsqrtConverge(v, v));
}

// Packed segment base of segment j (segment = packed column j in both uplo
// modes; the segment lengths differ: UPPER j+1, LOWER n-j):
//   UPPER: k = i + j*(j+1)/2        (j >= i) -> column j = [j*(j+1)/2, +j]
//   LOWER: k = i + j*(2n-j-1)/2     (i >= j) -> column j = [j*(2n-j+1)/2, +n-j-1]
// (LOWER formula is the Netlib sspr2.f / cblas_sspr2 column-major packed
// layout. Element (i, j) is at k = i + j*(2n-j-1)/2
// exactly; since i starts at j, the first element of column j sits at
// k = j + j*(2n-j-1)/2 = j*(2n-j+1)/2 = j*n - j*(j-1)/2, i.e. the sum of the
// n-m element counts of the preceding columns. The product j*(2n-j+1) is
// always even and both factors stay <= 8193 for n <= 4096: uint64-safe.)
__aicore__ inline uint64_t Sspr2SegBaseUpper(uint32_t j)
{
    return (static_cast<uint64_t>(j) * (static_cast<uint64_t>(j) + 1ULL)) / 2ULL;
}

__aicore__ inline uint64_t Sspr2SegBaseLower(uint32_t j, uint32_t n)
{
    // j*(2n-j+1)/2 == j*n - j*(j-1)/2 == j + j*(2n-j-1)/2 (column j of the
    // LOWER triangle starts right after the j-1 preceding columns of lengths
    // n, n-1, ..., n-j+1; element (i, j) = base + (i - j), stride 1).
    return (static_cast<uint64_t>(j) *
        (2ULL * static_cast<uint64_t>(n) - static_cast<uint64_t>(j) + 1ULL)) / 2ULL;
}

// Cumulative packed element count of segments [0, j) (segment m length:
// UPPER m+1, LOWER n-m). Equal to n*(n+1)/2 at j == n for both uplo modes;
// uint64-safe for n <= 4096.
__aicore__ inline uint64_t Sspr2CumSeg(bool upper, uint32_t j, uint32_t n)
{
    const uint64_t ju = static_cast<uint64_t>(j);
    const uint64_t nu = static_cast<uint64_t>(n);
    if (upper) {
        return ju * (ju + 1ULL) / 2ULL; // sum_{m<j}(m+1)
    }
    // identical to the LOWER column segment base Sspr2SegBaseLower(j, n).
    return ju * (2ULL * nu - ju + 1ULL) / 2ULL;
}

// Weighted cumulative used by the multi-core split.
// Weight of segment m = len_m + SPLIT_CHUNK_COST; the split
// balances weighted sums so per-core chunk counts and element volumes are both
// near-uniform (pure element balance starves late cores of chunks / floods the
// first core with short columns, each paying a fixed pipe round trip).
__aicore__ inline uint64_t Sspr2CumWeight(bool upper, uint32_t j, uint32_t n)
{
    return Sspr2CumSeg(upper, j, n) +
        static_cast<uint64_t>(j) * static_cast<uint64_t>(SSPR2_ARCH22_SPLIT_CHUNK_COST);
}

// Balanced multi-core split boundary: the first segment owned by core c,
// chosen so that the cumulative WEIGHTED count per core is balanced (weight =
// packed elements + SPLIT_CHUNK_COST per segment — a pure element count would
// leave the first core with O(sqrt(total)) short-column chunks each paying a
// fixed pipe round trip). Exact closed form
// of "largest j with cumWeight(j) <= floor(totalWeight*c/C)" where cumWeight is
// the cumulative weighted count before segment j (quadratic in j for both uplo
// modes, solved with the shared integer sqrt + monotone exactness fix-ups).
// Host and kernel share this one formula (same header; device call sites only),
// guaranteeing a unique, gap-free and overlap-free split:
// core c owns segments [f(c), f(c+1)). LOWER uses the Netlib column-major
// layout, hence its own cumulative formula and quadratic seed
// j >= ((2n+1+2C) - sqrt((2n+1+2C)^2 - 8*target)) / 2.
// 32B-block-aligned column boundary test: the
// packed byte offset of column j must be a multiple of 32B so that adjacent
// cores never touch the same L2 sector of the in-place AP tensor (concurrent
// byte-granular DataCopyPad access to one sector from two AIVs correlated with
// nondeterministic task wedges / VECTOR_CORE_TIMEOUT under multi-core). UPPER:
// base = j(j+1)/2 is 8-elem aligned iff j*(j+1) ≡ 0 (mod 16). LOWER: base =
// j(2n-j+1)/2 iff j*(2n-j+1) ≡ 0 (mod 16).
__aicore__ inline bool Sspr2IsAlignedBoundary(bool upper, uint32_t j, uint32_t n)
{
    const uint64_t ju = static_cast<uint64_t>(j);
    const uint64_t prod = upper ? ju * (ju + 1ULL)
                                : ju * (2ULL * static_cast<uint64_t>(n) - ju + 1ULL);
    return (prod % 16ULL) == 0ULL;
}

__aicore__ inline uint32_t Sspr2CoreSegStart(uint32_t c, uint32_t coreNum, uint32_t n, bool upper)
{
    if (c == 0 || coreNum == 0) {
        return 0;
    }
    if (c >= coreNum) {
        return n;
    }
    // Weighted balance (elements + SPLIT_CHUNK_COST per segment).
    const uint64_t total = Sspr2CumWeight(upper, n, n);
    const uint64_t target = total * c / coreNum;
    const uint64_t bigC = static_cast<uint64_t>(SSPR2_ARCH22_SPLIT_CHUNK_COST);
    uint32_t j;
    if (upper) {
        // j*(j+1)/2 + C*j <= target  ->  j^2 + (2C+1)j - 2*target <= 0
        const uint64_t b = 2ULL * bigC + 1ULL;
        j = static_cast<uint32_t>((Sspr2IsqrtU64(b * b + 8ULL * target) - b) / 2ULL);
    } else {
        // j*(2n-j+1)/2 + C*j <= target
        //   ->  j^2 - (2n+1+2C)j + 2*target >= 0
        //   ->  j >= ((2n+1+2C) - sqrt((2n+1+2C)^2 - 8*target)) / 2
        const uint64_t twoNp1 = 2ULL * static_cast<uint64_t>(n) + 1ULL + 2ULL * bigC;
        const uint64_t disc = twoNp1 * twoNp1 - 8ULL * target;
        j = static_cast<uint32_t>((twoNp1 - static_cast<uint64_t>(Sspr2IsqrtU64(disc))) / 2ULL);
    }
    if (j > n) {
        j = n;
    }
    // Exactness guard: the closed-form seed is at most ~1 off, these two
    // monotone fix-ups pin the boundary exactly
    // (cumWeight(j) <= target < cumWeight(j+1)).
    while (j > 0 && Sspr2CumWeight(upper, j, n) > target) {
        j--;
    }
    while (j < n && Sspr2CumWeight(upper, j + 1U, n) <= target) {
        j++;
    }
    // Round the boundary UP to the next 32B-block-aligned column
    // so concurrent cores never share an L2 sector of the in-place AP tensor.
    // Monotone and gap-free (adjusted boundaries stay non-decreasing); a core
    // whose range collapses to empty is a no-op (segStart == segEnd). When no
    // aligned column exists below n the boundary becomes n and the balance
    // degrades gracefully towards fewer effective cores (n <= 15 -> single
    // effective core), trading a little parallelism for correctness.
    while (j < n && !Sspr2IsAlignedBoundary(upper, j, n)) {
        j++;
    }
    return j;
}

// ---- Class-major core assignment ---------------------------------------------
// LOWER + direct-inc + n < 1536 only (host sets tiling.classMode = 1). Core b
// of usedCoreNum = C claims residual CLASS c and one contiguous sub-segment of
// that class's column list {c, c+8, c+16, ...} (columns are ELE_PER_BLOCK
// apart, so every per-column shifted-source offset stays 32B aligned by
// construction). Shared by the kernel (device) and the host UT mirror so the
// partition property (gap-free / overlap-free / full coverage) is testable.
//
// Core assignment (full coverage for EVERY C in [1, aivCoreNum]):
//   C <= 8 : core b runs classes {b, b+C, b+2C, ...} (strided), each full
//            range with a single sub-segment. C == 8 is the base case
//            (one class per core); C < 8 keeps every class served (the
//            naive b % 8 mapping would drop classes >= C).
//   C >  8 : core b runs class c = b % 8 as sub-segment s = b / 8 of an
//            Nc-way column-count split, Nc = (C-1-c)/8 + 1 = number of cores
//            serving class c. (An "S = ceil(C/8)" split per class would drop
//            the second half of classes with only one serving core when C is
//            not a multiple of 8, e.g. C = 10; the Nc-way split is the
//            coverage-complete generalization, identical whenever every class
//            has Nc == S, i.e. C in {8, 16}.)
// Class c has K columns (K = 0 for c >= n); sub-segment s of Nc covers
// column indices [floor(K*s/Nc), floor(K*(s+1)/Nc)) of the class list -- a
// proper partition (bnd(0)=0, bnd(Nc)=K, non-decreasing; empty sub-segments
// are legal and retired). Column j of the class list is j = c + 8*idx, so
// sub-segment boundaries in absolute columns are 8-aligned and the per-core
// anchor j0 = the sub-segment's first column (Xs = x[j0..n) built once).

__aicore__ inline uint32_t Sspr2ClassColCount(uint32_t c, uint32_t n)
{
    // Number of class-c columns below n: {c, c+8, ...} ∩ [0, n). c < n -> 1 +
    // floor((n-1-c)/8) (<= 192 for the class-mode envelope n < 1536); c >= n
    // -> 0 (empty class, retired by the caller). uint32-safe for n <= 4096.
    return (c < n) ? ((n - 1u - c) / SSPR2_ARCH22_ELE_PER_BLOCK + 1u) : 0u;
}

__aicore__ inline uint32_t Sspr2ClassSubStart(uint32_t K, uint32_t s, uint32_t nc)
{
    if (nc == 0U) {
        return 0U;  // nc == 0 guard, never hit: every call site derives nc >= 1
    }
    return K * s / nc; // first column index (within the class list) of sub-seg s
}

__aicore__ inline uint32_t Sspr2ClassSubEnd(uint32_t K, uint32_t s, uint32_t nc)
{
    if (nc == 0U) {
        return 0U;  // nc == 0 guard, never hit: every call site derives nc >= 1
    }
    return K * (s + 1u) / nc; // one-past-last column index (proper partition)
}

#endif // SSPR2_ARCH22_TILING_DATA_H
