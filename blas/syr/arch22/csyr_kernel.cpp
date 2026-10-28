/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/compute/simd.h"
// CsyrKernelTiling lives in the shared header (one copy for both TUs). The launcher
// definition keeps its GM_ADDR parameters; the host-facing uint8_t* prototype stays in
// the host TU, so no prototype is seen here.
#include "csyr_tiling.h"

#if __DAV_C220_VEC__

// Output group count S: the main path is S=3 x B group-level TRIPLE buffering (batch b's
// MTE3 atomic store drains under batches b+1 AND b+2's V compute, and the completion gate
// (MTE3-pipe barrier) fires once per triple instead of once per pair -- the per-PAIR form
// leaves one MTE3 drain tail exposed per 2 batches). The T1 fallback configuration (S=1,
// no software pipelining) is carried by this conditional section and is NOT compiled by
// default; it is only enabled by explicitly defining CSYR_ENABLE_FALLBACK_T1 from the
// build switch.
#ifndef CSYR_ENABLE_FALLBACK_T1
constexpr uint32_t CSYR_GROUPS = 3;
#else
constexpr uint32_t CSYR_GROUPS = 1;
#endif
// The core count is no longer a kernel-side constant -- it travels in the by-value tiling
// struct (coreNum): the HOST queries the platform AIV count at runtime (GetAivCoreCount in
// csyr_host.cpp csyr_pick_core_num; arch35 precedent blas/syr/arch35/syr_host.cpp) and
// clamps it to [1, CSYR_VEC_CORES]. CSYR_MAX_CORES stays the 910B3 AIV ceiling (40) and
// carries only DEFENSIVE bounds on the kernel side: the coreNum clamp in csyr_read_ctx
// (a host/kernel desync can only degrade the launch shape, never exceed the ceiling) and
// the twin-value anchor (host twin CSYR_VEC_CORES in csyr_host.cpp; values MUST stay
// identical across the twins -- cross-arch port concern only).
constexpr uint32_t CSYR_MAX_CORES = 40;

// Named constants, kernel side (host twins in csyr_host.cpp; values MUST stay identical
// across the twins):
constexpr uint32_t CSYR_UB_FLOATS = 49152u;  // per-core UB budget in floats (192KB)
constexpr uint32_t CSYR_VEC_GRAN = 64u;      // V-pipe repeat granularity in floats (one repeat
                                             // / 256B; nPad and repeat-count rounding)
constexpr uint32_t CSYR_HEAD_FLTS = 64u;     // interleaved-slot head region in floats
                                             // (32 rows x 2 floats; slotStride 2nPad + 64)
// UB-budget footprint terms of the layout-planning predicates (csyr_plan_batch /
// csyr_take_group_views). All values are in the unit of their name: *NPAD counts
// nPad units, *FLTS counts floats. Host twins: the budgetFixed/CSYR_SOA_BUDGET
// arithmetic in csyr_host.cpp -- values MUST stay identical across the twins.
constexpr uint32_t CSYR_RESIDENT_NPAD = 4u;    // resident prefix of every interleaved-domain
                                               // layout, in nPad units: the xraw (2*nPad) and
                                               // s' (2*nPad) interleaved pairs
constexpr uint32_t CSYR_SLOT_STREAM_NPAD = 2u; // per-column slot stream width, in nPad units
                                               // (the 2*nPad interleaved stream: n rows x 2
                                               // float components)
constexpr uint32_t CSYR_SOA_BASE_NPAD = 8u;     // pair-layout fixed prefix, in nPad units:
                                               // resident pair (4) + calc pair calc_re/calc_im
                                               // (2) + second calc pair calc_re2/calc_im2 (2)
constexpr uint32_t CSYR_TABLE_FIX_FLTS = 512u; // interleave-table fixed term of the
                                               // SoA/pair budgets, in floats
constexpr uint32_t CSYR_PAT_RESERVE_FLTS = 32u; // head-shift pattern region reservation, in
                                               // floats (4 select word pairs at stride-8
                                               // uint32 placement, footprint 26 uint32 =
                                               // 104B, reserved as 32 floats)
constexpr uint32_t CSYR_TEMPS_NPAD = 2u;        // s'-build temps region, in nPad units (one
                                               // 2*nPad interleaved span, parked past the
                                               // slot region when no slot can host it)
// Gated barrier-only schedule band (host twin: CSYR_BARRIER_GATE_NMIN/NMAX in
// csyr_host.cpp; values MUST stay identical across the twins): the barrier-only column
// schedule (csyr_sched_barrier below) is dispatched only on LOWER n in [900,2048] --
// the measured gains concentrate in LOWER n in [912,2000]; regressions are n<=534 /
// UPPER.
constexpr uint32_t CSYR_BARRIER_GATE_NMIN = 900u;
constexpr uint32_t CSYR_BARRIER_GATE_NMAX = 2048u;

// Zigzag round-robin column map (host twin: the closed-form comment in csyr_host.cpp): block b
// covers columns [b*coreNum, (b+1)*coreNum); core c takes column b*coreNum + c for even blocks
// and b*coreNum + (coreNum-1-c) for odd blocks. One column per core per block with alternating
// low/high positions balances BOTH per-core work and per-core column count.
__aicore__ __inline__ __attribute__((always_inline)) uint32_t csyr_column_of(
    uint32_t blockIdx, uint32_t ordinal, uint32_t coreNum)
{
    uint32_t base = ordinal * coreNum;
    return base + ((ordinal & 1u) ? (coreNum - 1u - blockIdx) : blockIdx);
}

// Closed form for the per-core zigzag column count. Exact derivation: with full =
// n / coreNum and r = n % coreNum, every core takes one column from each of the `full`
// complete blocks; the partial block bp = full (exists iff r > 0) gives core c the column
//   bp even: bp*coreNum + c              < n  iff c < r
//   bp odd : bp*coreNum + coreNum-1-c    < n  iff c >= coreNum - r
// so calNum(c) = full + (r > 0 ? ((bp odd ? c >= coreNum-r : c < r) ? 1 : 0) : 0).
__aicore__ __inline__ __attribute__((always_inline)) uint32_t csyr_cal_num_of(
    uint32_t blockIdx, uint32_t n, uint32_t coreNum)
{
    uint32_t full = n / coreNum;
    uint32_t r = n - full * coreNum;
    if (r == 0u) {
        return full;
    }
    if (full & 1u) {
        return full + ((blockIdx >= coreNum - r) ? 1u : 0u);
    }
    return full + ((blockIdx < r) ? 1u : 0u);
}

__aicore__ __inline__ __attribute__((always_inline)) void csyr_copy_gm2ub(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0,
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void csyr_copy_ub2gm(
    AscendC::GlobalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0,
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

// Host-staged prologue (kernel twin of the staging note in csyr_host.cpp): the resident
// x / s' views arrive PRE-BUILT in one combined 512B-aligned device buffer
// [x_staged | s'_staged] (each half ceil(2n/128)*128 floats, valid 8n bytes + zero tail),
// uploaded by the host. x_staged is a byte-exact copy of the interleaved logical x;
// s'_staged[2k] = sign-xor(xim(k)) is bit-identical to a multiply-by--1 build (IEEE
// multiply by -1 is exact and only touches the sign bit, including -0.0 -> +0.0), and
// s'_staged[2k+1] = xre(k). The strided x access is resolved host-side in
// csyr_read_x_host with the identical logical->physical mapping. The zero tails never
// feed any stored element (store lengths are exact, n-bounded forms; the vector-op
// repeat round-up excess is discarded before every store).

// LOWER interleaved-domain head-shift store contract: the store needs a ONE-TIME drop of
// the first 2h floats (h = j & 3); a single-pattern GatherMask over the FULL repeat count
// would drop 2h floats from EVERY 64-float repeat, landing chunk r (r>=1) h*r complex
// rows off the store contract. Fix: head-split store (repeatNum=1 head GatherMask +
// natural-aligned tail store), see csyr_compute_batch / csyr_store_batch. The SoA calc
// ping-pong interleave is not used: it regressed every measured case, refuting the
// cross-column WAR hypothesis for the per-column V-pipe stall.

// The interleaved-domain per-column chain is the 2-op form
//   out = muls(x, ajReal); out = Axpy(out, s', ajImag)
// with the fold scalars UNCHANGED (csyr_fold_scalar_ilv). Per element this is the
// identical rounded-op sequence as the 3-op chain (fl(x*ajRe), fl(s'*ajIm), fl(w+q')) --
// the Axpy product enters the component sum with its own rounding (the same two-rounding
// shape the SoA muls+Axpy chain relies on), so results are bit-identical for EVERY
// alpha, including the alpha=1e6 near-cancellation set. The fold is always applied
// (never the raw x(j) components), so the golden evaluation order is preserved
// unconditionally and no alpha-mode dispatch is needed.
// Slot economics: without a q' working region the interleaved-domain slot is
// (2*nPad + 64) floats -- the 64-float tail is the head-shift GatherMask dst of the
// head-split store. The interleaved domain fits the 192KB budget for EVERY n <= 4864
// (nPad=4096 -> batchCols=1, 2048 -> 4, 1024 -> 10, 512 -> 21). An aligned-window
// Duplicate(0)-head form is NOT adopted: it atomically adds +0 into the unreferenced
// triangle, which the G3 bit-level snapshot contract forbids.

// Fold t_j = alpha * x(j) from the UB-resident SoA x (scalars only from UB).
// Fold rounding shape (pinned empirically via per-element hex comparison on the
// alpha=1e6 near-cancellation cases): bisheng contracts the IMAG add into a
// single-rounding fma (ajIm = fma(alphaReal, xjImag, fl(alphaImag*xjReal))) while
// the REAL sub keeps both products separately rounded; the golden fold mirrors this
// shape (csyr_golden.h ComputeGoldenCsyrColumnScalar). The ASC frontend ignores
// -ffp-contract=off and #pragma float_control, and forcing the fold two-rounding by
// hand (volatile) corrupts the kernel via compiler spill collision — do not "fix"
// the fold on the kernel side. A fold-shape mismatch is amplified by alpha=1e6
// near-cancellation elements past max(1e-2, 32*ULP) via delta = x(i)*t_j.
__aicore__ __inline__ __attribute__((always_inline)) void csyr_fold_scalar(
    AscendC::LocalTensor<float> ub_xre,
    AscendC::LocalTensor<float> ub_xim,
    float alphaReal, float alphaImag, uint32_t j,
    float& ajReal, float& ajImag)
{
    float xjReal = ub_xre.GetValue(j);
    float xjImag = ub_xim.GetValue(j);
    ajReal = alphaReal * xjReal - alphaImag * xjImag;
    ajImag = alphaReal * xjImag + alphaImag * xjReal;
}

// Interleaved-source variant: identical fold expression, x(j) read from the resident
// interleaved xraw buffer (values bit-identical to the SoA GatherMask view: pattern-1/2
// select exactly xraw[2k] / xraw[2k+1]).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_fold_scalar_ilv(
    AscendC::LocalTensor<float> ub_xraw,
    float alphaReal, float alphaImag, uint32_t j,
    float& ajReal, float& ajImag)
{
    float xjReal = ub_xraw.GetValue(2 * j);
    float xjImag = ub_xraw.GetValue(2 * j + 1);
    ajReal = alphaReal * xjReal - alphaImag * xjImag;
    ajImag = alphaReal * xjImag + alphaImag * xjReal;
}

// Per-column slot stride (2*nPad + 64 floats in the interleaved domains, 2*nPad
// otherwise) in one always-inline helper.
__aicore__ __inline__ __attribute__((always_inline)) uint32_t csyr_slot_stride(
    uint32_t nPad, uint32_t lowerIlv)
{
    return (lowerIlv != 0) ? (2 * nPad + CSYR_HEAD_FLTS) : (2 * nPad);
}

// The LOWER SoA per-column chain (rowBegin/len/headShift scalar derivation + the
// muls/Axpy/muls/Axpy/Gather issue sequence), shared by its two call sites in
// csyr_compute_batch (the pair-path odd-column tail and the SoA fallback loop body).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_compute_soa_column(
    AscendC::LocalTensor<float> ub_xre,
    AscendC::LocalTensor<float> ub_xim,
    AscendC::LocalTensor<float> ub_calc_re,
    AscendC::LocalTensor<float> ub_calc_im,
    AscendC::LocalTensor<float> ub_out,
    AscendC::LocalTensor<uint32_t> ub_table,
    uint32_t n, uint32_t j, float ajReal, float ajImag)
{
    // LOWER SoA fallback (CSYR_DISABLE_LOWER_ILV): compute from the 8-aligned rowBegin (UB
    // vector reads need 32B alignment), then shift the Gather base by (j - rowBegin) so the
    // interleaved output starts at row j, and store precisely rows [j, n) from the unaligned
    // GM address (both are supported; neighbors stay bitwise untouched). The
    // unreferenced rows [rowBegin, j) are never written, so G3 bitwise snapshot holds even
    // for NaN payload / -0.0.
    uint32_t rowBegin = j & ~7u;
    uint32_t len = n - j;
    uint32_t headShift = j - rowBegin;
    uint32_t repeatNum = (len + headShift + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN;
    uint32_t gatherCount = ((2 * len + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN) * CSYR_VEC_GRAN;

    // Complex multiply chain (no conjugate; the diagonal i==j is the complex square
    // alpha*x(j)^2 with a generally non-zero imaginary part): muls (rounded product) +
    // per-component Axpy (dst += src*scalar; the Axpy product enters the component sum
    // with its own rounding -- with the golden fold shape matched, the whole triangle
    // matches the two-rounding golden bitwise). A literal 6-op chain (muls x2 -> sub -> muls x2 -> add) is
    // NOT used: that form hangs the V220 core in this batched Gather schedule
    // (reproduced on clean devices, multiple kernel variants); this form is
    // bit-equivalent two-rounding arithmetic.
    muls_v<ArchType::ASCEND_V220, float>(ub_calc_re, ub_xre[rowBegin], ajReal, repeatNum, 1, 1, 8, 8);
    AscendC::Axpy<float, float, false>(ub_calc_re, ub_xim[rowBegin], -ajImag, (uint64_t)0,
                                       (uint8_t)repeatNum, AscendC::UnaryRepeatParams(1, 1, 8, 8));

    muls_v<ArchType::ASCEND_V220, float>(ub_calc_im, ub_xre[rowBegin], ajImag, repeatNum, 1, 1, 8, 8);
    AscendC::Axpy<float, float, false>(ub_calc_im, ub_xim[rowBegin], ajReal, (uint64_t)0,
                                       (uint8_t)repeatNum, AscendC::UnaryRepeatParams(1, 1, 8, 8));

    AscendC::Gather(ub_out, ub_calc_re, ub_table, static_cast<uint32_t>(headShift * sizeof(float)),
                    gatherCount);
}

// One batch of columns: compute phase only. No events/barriers inside -- every op here is
// V-pipe (in-order); the MTE3 drain happens after the batch-level SET(V,MTE3).
// Compute paths, selected by the uplo dispatch bit and the LOWER pair-enable flag:
// - UPPER (interleaved domain 2-op): w = muls(slot, xraw, ajReal) + Axpy(slot, s', ajImag).
//   One op fewer and no q' buffer per column; bit-identical to the 3-op chain (same fold
//   scalars, same rounded-op sequence).
// - LOWER + slim-slot 2-op interleave (lowerIlv != 0): as UPPER plus the head-split store
//   contract (see above); slot = 2nPad stream + 64-float head dst.
// - LOWER + two-column interleave (lowerPair != 0): pairs of consecutive ordinals
//   interleave their two INDEPENDENT dependency chains on the V pipe. Per-column op order
//   is IDENTICAL to the solo chain (muls_re -> Axpy_re -> muls_im -> Axpy_im -> Gather
//   over the same operands), so results are bit-identical; only the cross-column issue
//   order changes. Column b computes into its own calc pair ub_calc_re2/ub_calc_im2
//   (2*nPad extra UB).
// - LOWER solo (SoA 4-op chain + headShift Gather; exact-row store): used only when
//   CSYR_DISABLE_LOWER_ILV forces the SoA fallback (the slim-slot 2-op layout fits every
//   n <= 4864) and for odd column tails of the pair path. (The interleaved-domain store
//   would need a UB src at 8B misalignment, which faults 507035 deterministically --
//   MTE3 UB src must stay 32B aligned.)
// Scalar fold: the fold is issued IN PLACE, immediately before its column's chain. A
// software-pipelined fold ring (fold issued D columns ahead) buys nothing here: the
// array-indexed ring forces the fold scalars to stack memory, adding scalar LSU traffic
// (register-spill class). In-place fold keeps the expression, sources and order
// identical -- bit-identical results; the fold's scalar latency hides under the previous
// column's Gather execution.

// The four compute paths of csyr_compute_batch are split into per-path always-inline
// functions so each stays under the size/complexity budget; the dispatcher below selects
// them with the original branch order.

// -------- UPPER interleaved domain, 2-op form --------
// out = muls(x, ajReal) then Axpy(out, s', ajImag) with the UNCHANGED fold scalars --
// identical rounded-op sequence, bit-identical for every alpha. The UPPER slot is the
// plain 2*nPad interleaved stream (rows [0, j]).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_compute_upper_batch(
    AscendC::LocalTensor<float> ub_xraw,
    AscendC::LocalTensor<float> ub_sprime,
    AscendC::LocalTensor<float> ub_out_group,
    float alphaReal, float alphaImag,
    uint32_t nPad, uint32_t blockIdx, uint32_t colBegin, uint32_t colEnd, uint32_t coreNum)
{
    for (uint32_t t = colBegin; t < colEnd; t++) {
        uint32_t slot = t - colBegin;
        uint32_t j = csyr_column_of(blockIdx, t, coreNum);
        float ajReal;
        float ajImag;
        csyr_fold_scalar_ilv(ub_xraw, alphaReal, alphaImag, j, ajReal, ajImag);
        AscendC::LocalTensor<float> ub_out = ub_out_group[slot * 2 * nPad];
        uint32_t len = j + 1;
        uint32_t repeatNum = (2 * len + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN;
        muls_v<ArchType::ASCEND_V220, float>(ub_out, ub_xraw, ajReal, repeatNum, 1, 1, 8, 8);
        AscendC::Axpy<float, float, false>(ub_out, ub_sprime, ajImag, (uint64_t)0,
                                           (uint8_t)repeatNum, AscendC::UnaryRepeatParams(1, 1, 8, 8));
    }
}

// -------- LOWER interleaved domain, 2-op + slim slot ----
// slot = 2*nPad stream + 64-float head-shift dst; the interleaved domain fits every
// n <= 4864. Store discipline: head-split ONLY (rows [rowBegin, j) are never written --
// no +0 atomic RMW into the unreferenced triangle; G3 snapshot contract).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_compute_lower_ilv_batch(
    AscendC::LocalTensor<uint32_t> ub_pat0,
    AscendC::LocalTensor<float> ub_out_group,
    AscendC::LocalTensor<float> ub_xraw,
    AscendC::LocalTensor<float> ub_sprime,
    float alphaReal, float alphaImag,
    uint32_t n, uint32_t nPad, uint32_t slotStride,
    uint32_t blockIdx, uint32_t colBegin, uint32_t colEnd, uint32_t coreNum)
{
    for (uint32_t t = colBegin; t < colEnd; t++) {
        uint32_t slot = t - colBegin;
        uint32_t j = csyr_column_of(blockIdx, t, coreNum);
        float ajReal;
        float ajImag;
        csyr_fold_scalar_ilv(ub_xraw, alphaReal, alphaImag, j, ajReal, ajImag);
        AscendC::LocalTensor<float> ub_out = ub_out_group[slot * slotStride];
        uint32_t rowBegin = j & ~3u;
        uint32_t h = j - rowBegin;
        uint32_t lenCalc = n - rowBegin;
        uint32_t repeatNum = (2 * lenCalc + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN;
        // w = x * ajReal over the interleaved rows [rowBegin, n)
        muls_v<ArchType::ASCEND_V220, float>(ub_out, ub_xraw[2 * rowBegin], ajReal, repeatNum, 1, 1, 8, 8);
        // out += s' * ajImag (Axpy accumulate; two-rounding shape, with the SAME fold
        // scalars)
        AscendC::Axpy<float, float, false>(ub_out, ub_sprime[2 * rowBegin], ajImag, (uint64_t)0,
                                           (uint8_t)repeatNum, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        if (h != 0u) {
            // head-only shift: ONE repeat, drop its first 2h floats. repeatNum=1 makes
            // the per-repeat select identical to the one-time head drop.
            // Only the first pattern word pair [8h, 8h+2) is consumed (prologue-written).
            // dst = the slot's 64-float head region past the stream (dead until store).
            uint64_t rsvdCnt = 0;
            AscendC::LocalTensor<uint32_t> ub_pat = ub_pat0[h * 8];
            AscendC::GatherMask<float>(ub_out[2 * nPad], ub_out, ub_pat, false, 0,
                                       {1, 1, 8, 0}, rsvdCnt);
        }
    }
}

// -------- LOWER two-column software interleave --------
__aicore__ __inline__ __attribute__((always_inline)) void csyr_compute_lower_pair_batch(
    AscendC::LocalTensor<float> ub_calc_re,
    AscendC::LocalTensor<float> ub_calc_im,
    AscendC::LocalTensor<float> ub_calc_re2,
    AscendC::LocalTensor<float> ub_calc_im2,
    AscendC::LocalTensor<float> ub_xre,
    AscendC::LocalTensor<float> ub_xim,
    AscendC::LocalTensor<float> ub_out_group,
    AscendC::LocalTensor<uint32_t> ub_table,
    float alphaReal, float alphaImag,
    uint32_t n, uint32_t nPad,
    uint32_t blockIdx, uint32_t colBegin, uint32_t colEnd, uint32_t coreNum)
{
    uint32_t t = colBegin;
    for (; t + 1 < colEnd; t += 2) {
        uint32_t jA = csyr_column_of(blockIdx, t, coreNum);
        uint32_t jB = csyr_column_of(blockIdx, t + 1, coreNum);
        // In-place folds: same expression/sources/order in every path -- the scalars
        // are bit-identical whenever they are issued.
        float ajRealA;
        float ajImagA;
        float ajRealB;
        float ajImagB;
        csyr_fold_scalar(ub_xre, ub_xim, alphaReal, alphaImag, jA, ajRealA, ajImagA);
        csyr_fold_scalar(ub_xre, ub_xim, alphaReal, alphaImag, jB, ajRealB, ajImagB);
        uint32_t rowA = jA & ~7u;
        uint32_t rowB = jB & ~7u;
        uint32_t lenA = n - jA;
        uint32_t lenB = n - jB;
        uint32_t headA = jA - rowA;
        uint32_t headB = jB - rowB;
        uint32_t repA = (lenA + headA + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN;
        uint32_t repB = (lenB + headB + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN;
        uint32_t gcA = ((2 * lenA + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN) * CSYR_VEC_GRAN;
        uint32_t gcB = ((2 * lenB + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN) * CSYR_VEC_GRAN;
        AscendC::LocalTensor<float> ub_out_a = ub_out_group[(t - colBegin) * 2 * nPad];
        AscendC::LocalTensor<float> ub_out_b = ub_out_group[(t - colBegin + 1) * 2 * nPad];
        // Interleaved issue: A-chain and B-chain ops alternate so each column's
        // muls->Axpy->Gather latency is hidden under the other column's independent
        // work. Per-column sequence identical to the solo chain (bit-identical).
        muls_v<ArchType::ASCEND_V220, float>(ub_calc_re, ub_xre[rowA], ajRealA, repA, 1, 1, 8, 8);
        AscendC::Axpy<float, float, false>(ub_calc_re, ub_xim[rowA], -ajImagA, (uint64_t)0,
                                           (uint8_t)repA, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        muls_v<ArchType::ASCEND_V220, float>(ub_calc_re2, ub_xre[rowB], ajRealB, repB, 1, 1, 8, 8);
        AscendC::Axpy<float, float, false>(ub_calc_re2, ub_xim[rowB], -ajImagB, (uint64_t)0,
                                           (uint8_t)repB, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        muls_v<ArchType::ASCEND_V220, float>(ub_calc_im, ub_xre[rowA], ajImagA, repA, 1, 1, 8, 8);
        AscendC::Axpy<float, float, false>(ub_calc_im, ub_xim[rowA], ajRealA, (uint64_t)0,
                                           (uint8_t)repA, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        muls_v<ArchType::ASCEND_V220, float>(ub_calc_im2, ub_xre[rowB], ajImagB, repB, 1, 1, 8, 8);
        AscendC::Axpy<float, float, false>(ub_calc_im2, ub_xim[rowB], ajRealB, (uint64_t)0,
                                           (uint8_t)repB, AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::Gather(ub_out_a, ub_calc_re, ub_table, static_cast<uint32_t>(headA * sizeof(float)), gcA);
        AscendC::Gather(ub_out_b, ub_calc_re2, ub_table, static_cast<uint32_t>(headB * sizeof(float)), gcB);
    }
    for (; t < colEnd; t++) {
        // Solo tail (odd column count): the solo per-column chain.
        uint32_t j = csyr_column_of(blockIdx, t, coreNum);
        float ajReal;
        float ajImag;
        csyr_fold_scalar(ub_xre, ub_xim, alphaReal, alphaImag, j, ajReal, ajImag);
        AscendC::LocalTensor<float> ub_out = ub_out_group[(t - colBegin) * 2 * nPad];
        csyr_compute_soa_column(ub_xre, ub_xim, ub_calc_re, ub_calc_im, ub_out, ub_table, n, j,
                                ajReal, ajImag);
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void csyr_compute_batch(
    AscendC::LocalTensor<float> ub_xraw,
    AscendC::LocalTensor<float> ub_sprime,
    AscendC::LocalTensor<float> ub_xre,
    AscendC::LocalTensor<float> ub_xim,
    AscendC::LocalTensor<float> ub_calc_re,
    AscendC::LocalTensor<float> ub_calc_im,
    AscendC::LocalTensor<float> ub_calc_re2,
    AscendC::LocalTensor<float> ub_calc_im2,
    AscendC::LocalTensor<float> ub_out_group,
    AscendC::LocalTensor<uint32_t> ub_table,
    AscendC::LocalTensor<uint32_t> ub_pat0,
    uint32_t uplo, uint32_t n, uint32_t nPad, uint32_t lowerPair, uint32_t lowerIlv,
    float alphaReal, float alphaImag,
    uint32_t blockIdx, uint32_t colBegin, uint32_t colEnd, uint32_t calNum,
    uint32_t coreNum)
{
    uint32_t slotStride = csyr_slot_stride(nPad, lowerIlv);
    if (uplo != 0u) {
        // -------- UPPER interleaved domain, 2-op form --------
        // See csyr_compute_upper_batch above.
        csyr_compute_upper_batch(ub_xraw, ub_sprime, ub_out_group, alphaReal, alphaImag,
                                 nPad, blockIdx, colBegin, colEnd, coreNum);
        return;
    }
    if (lowerIlv != 0) {
        // -------- LOWER interleaved domain, 2-op + slim slot ----
        // See csyr_compute_lower_ilv_batch above (head-split store contract).
        csyr_compute_lower_ilv_batch(ub_pat0, ub_out_group, ub_xraw, ub_sprime, alphaReal,
                                     alphaImag, n, nPad, slotStride, blockIdx, colBegin,
                                     colEnd, coreNum);
        return;
    }
    if (uplo == 0 && lowerPair != 0) {
        // -------- LOWER two-column software interleave --------
        // See csyr_compute_lower_pair_batch above (solo tail chain).
        csyr_compute_lower_pair_batch(ub_calc_re, ub_calc_im, ub_calc_re2, ub_calc_im2,
                                      ub_xre, ub_xim, ub_out_group, ub_table, alphaReal,
                                      alphaImag, n, nPad, blockIdx, colBegin, colEnd, coreNum);
        return;
    }
    for (uint32_t t = colBegin; t < colEnd; t++) {
        uint32_t slot = t - colBegin;
        uint32_t j = csyr_column_of(blockIdx, t, coreNum);
        // Column scalar t_j = alpha * x(j), folded in place (see the function comment
        // above; identical expression/sources/order, bit-identical results).
        float ajReal;
        float ajImag;
        csyr_fold_scalar(ub_xre, ub_xim, alphaReal, alphaImag, j, ajReal, ajImag);

        AscendC::LocalTensor<float> ub_out = ub_out_group[slot * 2 * nPad];

        // LOWER SoA fallback (CSYR_DISABLE_LOWER_ILV): the per-column chain lives in
        // csyr_compute_soa_column above (see its comment for the rowBegin/Gather-shift
        // contract).
        csyr_compute_soa_column(ub_xre, ub_xim, ub_calc_re, ub_calc_im, ub_out, ub_table, n, j,
                                ajReal, ajImag);
    }
}

// One batch of columns: MTE3 drain phase (per-column precise atomic store; each column owns a
// private output slot, so no intra-batch dependency). NOTE: a dummy plain-MTE3 write after the
// atomics (to restore the SET_FLAG(MTE3,V) gate) was tried here and made EVERY case fault
// (507034) including n=1 — the SetAtomicNone/SetAtomicAdd toggle mid-stream is not safe on
// this stack; the batch-boundary race is instead handled by the per-batch compute->store
// ordering in csyr_process_columns (one full batch of compute+ store between a group's stores
// and its reuse).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_store_batch(
    AscendC::GlobalTensor<float> gm_A,
    AscendC::LocalTensor<float> ub_out_group,
    uint32_t uplo, uint32_t n, uint32_t lda, uint32_t nPad, uint32_t lowerIlv,
    uint32_t blockIdx, uint32_t colBegin, uint32_t colEnd, uint32_t coreNum)
{
    if (uplo == 0 && lowerIlv != 0) {
        // LOWER interleaved domain, head-split store (see the head-shift store contract
        // note at the top of this file). h==0: single burst, rows [j, n) -- identical
        // form to SoA/UPPER. h!=0: head rows [j, rowBegin+32) from the repeatNum=1
        // GatherMask dst (the slot's 64-float head region past the stream), then tail
        // rows [rowBegin+32, n) directly from the unshifted stream at its natural
        // 256B-aligned float offset 64. Union = rows [j, n), each element written
        // exactly once (atomic add); rows [rowBegin, j) are never touched (G3 snapshot
        // safe). Slot stride is (2nPad + 64).
        uint32_t slotStride = 2 * nPad + CSYR_HEAD_FLTS;
        for (uint32_t t = colBegin; t < colEnd; t++) {
            uint32_t slot = t - colBegin;
            uint32_t j = csyr_column_of(blockIdx, t, coreNum);
            uint32_t rowBegin = j & ~3u;
            uint32_t h = j - rowBegin;
            uint32_t lenCalc = n - rowBegin;
            AscendC::LocalTensor<float> ub_slot = ub_out_group[slot * slotStride];
            if (h == 0u) {
                // j*lda is computed in the uint64 domain -- lda is only bounded by the
                // int parameter range (the host element-offset guard allows up to 2^31
                // element offsets), so the uint32 product could wrap.
                csyr_copy_ub2gm(gm_A[(static_cast<uint64_t>(j) * lda + j) * 2], ub_slot, (n - j) * 2);
            } else {
                // head: min(CSYR_HEAD_FLTS-2h, 2*(n-j)) floats; when 2*lenCalc < CSYR_HEAD_FLTS
                // the store stops before the GatherMask-selected garbage tail
                // (src floats >= 2*lenCalc).
                uint32_t headFlts =
                    (2 * lenCalc < CSYR_HEAD_FLTS) ? (2u * (n - j)) : (CSYR_HEAD_FLTS - 2u * h);
                csyr_copy_ub2gm(gm_A[(static_cast<uint64_t>(j) * lda + j) * 2], ub_slot[2 * nPad], headFlts);
                if (2 * lenCalc > CSYR_HEAD_FLTS) {
                    // Tail src = stream offset of row rowBegin+32 == CSYR_HEAD_FLTS floats.
                    csyr_copy_ub2gm(gm_A[(static_cast<uint64_t>(j) * lda + rowBegin + 32) * 2],
                                    ub_slot[CSYR_HEAD_FLTS], 2 * lenCalc - CSYR_HEAD_FLTS);
                }
            }
        }
        return;
    }
    for (uint32_t t = colBegin; t < colEnd; t++) {
        uint32_t slot = t - colBegin;
        uint32_t j = csyr_column_of(blockIdx, t, coreNum);
        uint32_t len = (uplo != 0) ? (j + 1) : (n - j);
        // uint64 offset domain (same wrap risk as above).
        csyr_copy_ub2gm(gm_A[(static_cast<uint64_t>(j) * lda + (uplo != 0 ? 0u : j)) * 2],
                        ub_out_group[slot * 2 * nPad], len * 2);
    }
}

// Batched S-group schedule (S = CSYR_GROUPS = 3, triple buffering). Columns are processed
// in batches of batchCols; group g owns batchCols output slots and event slot E2/E3/E4 + g.
// Batches are processed three at a time (kb, kb+1, kb+2): each batch's compute runs in its
// group, its MTE3 stores drain under the next two batches' V work, and the full-pipe
// completion gate fires once per triple, before the first group's slots are
// reused at batch kb+3 (MTE3 atomic writes do not self-gate). Event slots stay 1:1 balanced;
// no V->MTE2 events.
//
// The two column schedules (the production event form and the gated barrier form) share ONE
// context (CsyrCtx below, built once in csyr) with the bodies in per-phase always-inline
// helpers: csyr_sched_vonly (the dbgMode==1 V-only attribution pass), csyr_sched_events
// (the production per-TRIPLE event schedule) and csyr_sched_barrier (the barrier-only
// schedule). Structural guarantees: MTE3 atomic writes do not self-gate (the per-TRIPLE
// completion barrier stays the only real store-completion gate of the event form; the
// per-batch barrier of the gated form), event slots stay 1:1 balanced, and the barrier
// schedule uses no flag pair at all (the flag-nonconvergence deadlock class is structurally
// impossible).
struct CsyrCtx {
    // resident / working UB views (taken by csyr_take_resident_views / csyr_take_group_views)
    AscendC::LocalTensor<float> ub_xraw;
    AscendC::LocalTensor<float> ub_sprime;
    AscendC::LocalTensor<float> ub_xre;
    AscendC::LocalTensor<float> ub_xim;
    AscendC::LocalTensor<float> ub_calc_re;
    AscendC::LocalTensor<float> ub_calc_im;
    AscendC::LocalTensor<float> ub_calc_re2;
    AscendC::LocalTensor<float> ub_calc_im2;
    AscendC::LocalTensor<float> ub_out_base;
    AscendC::LocalTensor<uint32_t> ub_table;
    AscendC::LocalTensor<uint32_t> ub_pat0;
    AscendC::GlobalTensor<float> gm_A;
    // launch scalars (read by csyr_read_ctx) and layout scalars (set by csyr_plan_batch /
    // csyr_take_group_views); batchCols carries the clamped effBatch once planning is done.
    uint32_t uplo;
    uint32_t n;
    uint32_t lda;
    uint32_t nPad;
    uint32_t stagedFloats;
    uint32_t slotStride;
    uint32_t lowerIlv;
    uint32_t lowerPair;
    uint32_t batchCols;
    uint32_t coreNum;
    uint32_t dbgMode;
    uint32_t barrierGate;
    uint32_t blockIdx;
    uint32_t calNum;
    float alphaReal;
    float alphaImag;
};

// dbgMode==1 V-only attribution mode (shared by both schedules): compute only, no MTE3,
// no V<->MTE3 events. Reads the context only (the compute helpers take the UB views and
// scalars by value).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_sched_vonly(const CsyrCtx& c)
{
    uint32_t nb = (c.calNum + c.batchCols - 1) / c.batchCols;
    for (uint32_t k = 0; k < nb; k++) {
        uint32_t colBegin = k * c.batchCols;
        uint32_t colEnd = colBegin + c.batchCols;
        if (colEnd > c.calNum) {
            colEnd = c.calNum;
        }
        csyr_compute_batch(c.ub_xraw, c.ub_sprime, c.ub_xre, c.ub_xim, c.ub_calc_re,
                           c.ub_calc_im, c.ub_calc_re2, c.ub_calc_im2, c.ub_out_base, c.ub_table,
                           c.ub_pat0, c.uplo, c.n, c.nPad, c.lowerPair, c.lowerIlv, c.alphaReal,
                           c.alphaImag, c.blockIdx, colBegin, colEnd, c.calNum, c.coreNum);
    }
    // Barrier scope (dbg-only path): the only ops in flight here are the
    // V-pipe computes of this attribution pass -- no MTE2/MTE3 op is ever issued in
    // vonly (the staged loads are drained by the prologue barrier, stores are skipped),
    // so a V-pipe barrier is the exact drain set. The shared kernel-exit barrier still
    // drains every pipe before SetAtomicNone / exit.
    PIPE_BARRIER(V);
}

// Production column schedule (event form).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_sched_events(CsyrCtx& c)
{
    // Output slot stride: 2*nPad + 64 floats per column in the interleaved domains
    // (stream half + 64-float head-shift dst), 2*nPad otherwise (UPPER slim / LOWER SoA).
    uint32_t slotStride = csyr_slot_stride(c.nPad, c.lowerIlv);
    SET_FLAG(MTE3, V, EVENT_ID2);
    SET_FLAG(MTE3, V, EVENT_ID3);
    SET_FLAG(MTE3, V, EVENT_ID4);

    uint32_t groupFloats = c.batchCols * slotStride;
    uint32_t nb = (c.calNum + c.batchCols - 1) / c.batchCols;

    // Per-TRIPLE batch schedule (S=3): batches are processed three at a time and
    // the full-pipe completion gate fires once per TRIPLE, before the first
    // group's slots are reused at batch kb+3. The barrier is the only real store-completion
    // gate (with SetAtomicAdd the trailing SET(MTE3,V) does NOT self-gate on in-flight
    // atomic stores), but it only needs to precede a group-slot REUSE. Stores(kb+j) drain
    // under compute(kb+j+1) AND compute(kb+j+2), so the MTE3 drain tail is hidden twice as
    // deep. Group-reuse safety (for every nb, incl. the UT slot-reuse partitions):
    // compute(kb+j) rewrites group (kb+j)%3 whose previous stores (kb+j-3) completed at the
    // previous triple barrier (kb >= 3); for kb < 3 every group is stored exactly once
    // (prologue-primed, no barrier).
    // The schedule granularity is CSYR_GROUPS. With the T1 fallback build (CSYR_GROUPS=1)
    // the triple degenerates to a per-batch schedule with the completion barrier firing
    // before EVERY batch and gj==0/evtj==EVENT_ID2 for all batches -- the (kb+j)%1 slot
    // mapping and the 1:1 event pairing stay sound.
    for (uint32_t kb = 0; kb < nb; kb += CSYR_GROUPS) {
        if (kb >= CSYR_GROUPS) {
            // Store-completion gate -- kept full-pipe by hardware evidence:
            // this gate carries the completion of the in-flight MTE3 ATOMIC stores of
            // batches kb-S..kb-1 before their group slots are REUSED by
            // compute(kb..kb+2) (a WAR gate on the slots' UB source). A narrowed
            // PIPE_BARRIER(MTE3) form was evaluated on hardware at the
            // barrier-form twin of this gate: single-pipe MTE3 (and the V+MTE3
            // pair) deterministically corrupted roughly 30% of elements on
            // every heavy-reuse in-band shape (maxAbsErr ~1e2) while the full-barrier
            // form passes the same probes -- on this stack a single-pipe barrier is
            // NOT a real completion gate for in-flight SetAtomicAdd stores (the
            // atomic engine's UB-source reads outlive MTE3-pipe idle). The
            // compute->store ordering inside each batch is carried by the
            // SET/WAIT(V,MTE3) pair below; this barrier is the only real
            // store-completion gate of the schedule, so the full-pipe form is load-bearing here.
            PIPE_BARRIER(ALL); // stores(kb-S..kb-1) complete before their groups are reused
        }
        for (uint32_t j = 0; j < CSYR_GROUPS && kb + j < nb; j++) {
            uint32_t gj = (kb + j) % CSYR_GROUPS;
            int32_t evtj = EVENT_ID2 + gj;
            uint32_t colBeginj = (kb + j) * c.batchCols;
            uint32_t colEndj = colBeginj + c.batchCols;
            if (colEndj > c.calNum) {
                colEndj = c.calNum;
            }
            AscendC::LocalTensor<float> ub_out_gj = c.ub_out_base[gj * groupFloats];
            WAIT_FLAG(MTE3, V, evtj); // group gj free (prologue prime or triple barrier above);
                                      // stores(kb+j-1) drain under compute(kb+j)
            if (c.dbgMode != 2) {
                csyr_compute_batch(c.ub_xraw, c.ub_sprime, c.ub_xre, c.ub_xim, c.ub_calc_re,
                                   c.ub_calc_im, c.ub_calc_re2, c.ub_calc_im2, ub_out_gj, c.ub_table,
                                   c.ub_pat0, c.uplo, c.n, c.nPad, c.lowerPair, c.lowerIlv, c.alphaReal,
                                   c.alphaImag, c.blockIdx, colBeginj, colEndj, c.calNum, c.coreNum);
            }
            SET_FLAG(V, MTE3, evtj);
            WAIT_FLAG(V, MTE3, evtj); // V self-gates: compute(kb+j) complete before its stores
            csyr_store_batch(c.gm_A, ub_out_gj, c.uplo, c.n, c.lda, c.nPad, c.lowerIlv, c.blockIdx,
                             colBeginj, colEndj, c.coreNum);
            SET_FLAG(MTE3, V, evtj); // re-arm for group gj reuse at batch kb+j+3 (triple barrier)
        }
    }

    // Trailing completion: the last triple's stores may still be in flight; the SET(MTE3,V)
    // gates above fired early (atomic stores do not self-gate), so drain with a real
    // barrier before the epilogue (SetAtomicNone / kernel exit). Narrowed
    // from PIPE_BARRIER(ALL) to the MTE3 pipe: the only ops in
    // flight are the last batches' MTE3 atomic stores (each batch's V compute already
    // self-gated through its own SET/WAIT(V,MTE3) pair), so this MTE3 drain narrows the
    // pre-epilogue stall without touching any reuse gate. NOTE this is sound ONLY
    // because the kernel-exit full barrier below is the real completion gate before
    // SetAtomicNone (the single-pipe barrier is not a real in-flight-atomic gate on
    // this stack -- see the top-of-triple note above); nothing between the two
    // barriers consumes the store data.
    PIPE_BARRIER(MTE3);
    WAIT_FLAG(MTE3, V, EVENT_ID2);
    WAIT_FLAG(MTE3, V, EVENT_ID3);
    WAIT_FLAG(MTE3, V, EVENT_ID4);
}

// The barrier-only column schedule. The per-batch event quad
//   WAIT(MTE3,V) -> SET(V,MTE3) -> WAIT(V,MTE3) -> SET(MTE3,V)
// and the prologue primes are REPLACED by ONE full PIPE_BARRIER(ALL) per batch, issued between
// the batch's compute and its store (kept full-pipe per the hardware evidence at the
// gate -- a single-pipe barrier is not a real completion gate for in-flight atomic stores
// on this stack). Ordering guarantees vs the event form:
//   (1) compute(kb) -> store(kb): the per-batch barrier gates the store on the completion of
//       the batch's V ops;
//   (2) store(k) -> compute(k+3) group-slot reuse: store(k-S) completed at the barrier of
//       iteration kb-S+1, strictly before compute(kb) (the event form gated this with
//       prologue primes / re-arm SETs + WAIT(MTE3,V) + the top-of-triple barrier -- the
//       barrier being the only real store-completion gate);
//   (3) kernel-exit drain: the trailing PIPE_BARRIER(MTE3) (the event form's trailing 3x
//       WAIT(MTE3,V) is redundant behind a real barrier and is dropped).
// The full barrier IS a real store-completion gate on this stack (the event form already
// relies on it for reuse), so the variant is semantically equivalent. No event slots are used
// in the column phase at all, and no flag pair exists in the schedule, so the
// flag-nonconvergence deadlock class is structurally impossible.
__aicore__ __inline__ __attribute__((always_inline)) void csyr_sched_barrier(CsyrCtx& c)
{
    // Output slot stride: 2*nPad + 64 floats per column in the interleaved domains
    // (stream half + 64-float head-shift dst), 2*nPad otherwise (UPPER slim / LOWER SoA).
    uint32_t slotStride = csyr_slot_stride(c.nPad, c.lowerIlv);
    uint32_t groupFloats = c.batchCols * slotStride;
    uint32_t nb = (c.calNum + c.batchCols - 1) / c.batchCols;

    // Barrier-only sparse sync: the per-batch event quad
    //   WAIT(MTE3,V) -> SET(V,MTE3) -> WAIT(V,MTE3) -> SET(MTE3,V)
    // becomes ONE full PIPE_BARRIER(ALL) per batch, issued between the batch's compute and its
    // store. The overlap structure is PRESERVED: store(kb) is issued immediately after
    // compute(kb) completes and drains under compute(kb+1) (the MTE3 queue is in-order, so
    // store(kb+1) cannot execute before store(kb) regardless of issue timing). The barrier
    // gates store(kb) on the completion of compute(kb) AND on all previously issued work; the
    // group-slot reuse is gated directly by the per-batch barriers: compute(kb) rewrites
    // group kb%S whose previous store(kb-S) completed at the barrier of iteration kb-S+1
    // (strictly earlier). Same S=3 group-slot mapping, zero event slots in the column phase,
    // and no flag pair left in the schedule (flag-nonconvergence deadlock class structurally
    // impossible).
    for (uint32_t kb = 0; kb < nb; kb++) {
        uint32_t gj = kb % CSYR_GROUPS;
        uint32_t colBegin = kb * c.batchCols;
        uint32_t colEnd = colBegin + c.batchCols;
        if (colEnd > c.calNum) {
            colEnd = c.calNum;
        }
        AscendC::LocalTensor<float> ub_out_gj = c.ub_out_base[gj * groupFloats];
        if (c.dbgMode != 2) {
            csyr_compute_batch(c.ub_xraw, c.ub_sprime, c.ub_xre, c.ub_xim, c.ub_calc_re,
                               c.ub_calc_im, c.ub_calc_re2, c.ub_calc_im2, ub_out_gj, c.ub_table,
                               c.ub_pat0, c.uplo, c.n, c.nPad, c.lowerPair, c.lowerIlv, c.alphaReal,
                               c.alphaImag, c.blockIdx, colBegin, colEnd, c.calNum, c.coreNum);
        }
        if (c.dbgMode != 1) {
            // compute/store gate -- kept full-pipe by hardware
            // evidence: a single full-pipe barrier. The narrowings
            // PIPE_BARRIER(V)+PIPE_BARRIER(MTE3) AND single PIPE_BARRIER(MTE3) were
            // both implemented and evaluated on hardware here; both deterministically
            // corrupted roughly 30% of elements on the in-band LOWER shapes with
            // multiple group-slot reuses (maxAbsErr ~1e2)
            // while the full-barrier form passes the same probes -- on this
            // stack a single-pipe barrier is NOT a real completion gate for in-flight
            // SetAtomicAdd stores (the atomic engine's UB-source reads outlive
            // MTE3-pipe idle), and the reuse chain is exactly what breaks. The two
            // orderings this one barrier carries:
            //   (a) compute(kb)'s V-pipe slot writes -> store(kb)'s MTE3 source reads
            //       (the event form's SET/WAIT(V,MTE3)-pair equivalent);
            //   (b) the group-slot reuse chain -- compute(kb) rewrites group kb%S
            //       whose previous store(kb-S) must have READ its UB source, gated by
            //       the full-pipe drain of the barrier at body kb-S+1 (two iterations
            //       back). This is the ONE full-pipe barrier inside a hot loop; the
            //       per-band dispatch keeps it off every other shape.
            PIPE_BARRIER(ALL);
            csyr_store_batch(c.gm_A, ub_out_gj, c.uplo, c.n, c.lda, c.nPad, c.lowerIlv, c.blockIdx,
                             colBegin, colEnd, c.coreNum);
        }
    }

    // Trailing completion: the last batch's stores may still be in flight; drain with a real
    // barrier before the epilogue (SetAtomicNone / kernel exit). Narrowed
    // from PIPE_BARRIER(ALL) to the MTE3 pipe: the only ops in flight
    // are the last batch's MTE3 atomic stores. Sound ONLY because the kernel-exit full
    // barrier below is the real completion gate before SetAtomicNone (a single-pipe
    // barrier is not a real in-flight-atomic gate on this stack -- see the per-batch
    // note above); nothing between the two barriers consumes the store data.
    PIPE_BARRIER(MTE3);
}

// Gated dispatch (host twin: the barrierGate assignment in csyr_host.cpp): gate=1 ->
// csyr_sched_barrier (the barrier-only schedule); gate=0 -> csyr_sched_events, the
// production schedule. Both arms share the identical context; dbgMode==1 -> the shared
// V-only attribution pass. The gate itself is re-validated in csyr_read_ctx, so a
// host/kernel desync can only fall back to the production schedule.
__aicore__ __inline__ __attribute__((always_inline)) void csyr_process_columns_gated(CsyrCtx& c)
{
    if (c.dbgMode == 1) {
        csyr_sched_vonly(c);
        return;
    }
    if (c.barrierGate != 0u) {
        csyr_sched_barrier(c);
    } else {
        csyr_sched_events(c);
    }
}

// Stage helper: reads the tiling struct into the shared context.
__aicore__ __inline__ __attribute__((always_inline)) CsyrCtx csyr_read_ctx(CsyrKernelTiling tiling,
    uint32_t blockIdx)
{
    // The tiling travels as a BY-VALUE kernel parameter -- no GM scalar reads are
    // needed, and the per-core calNum is derived in closed form from (n, coreNum)
    // (a struct param costs zero MTE2/SCALAR pipe time, unlike a UB round-trip that
    // pays MTE2 pipe busy per extra burst).
    CsyrCtx c;
    c.uplo = tiling.uplo;
    c.n = tiling.n;
    c.lda = tiling.lda;
    // incxAbs/negx travel in the tiling for launch-descriptor stability but are consumed
    // HOST-side only in the staged form (csyr_read_x_host resolves the stride before the
    // upload); the kernel never reads them.
    c.alphaReal = tiling.alphaReal;
    c.alphaImag = tiling.alphaImag;
    c.batchCols = tiling.batchCols;
    c.coreNum = (tiling.coreNum >= 1u && tiling.coreNum <= CSYR_MAX_CORES)
                                 ? tiling.coreNum : CSYR_MAX_CORES;
    // Debug/perf-attribution mode (tiling field; env CSYR_DBG_MODE on host):
    // 0=normal; 1=V-only (skip MTE3 stores AND the V<->MTE3 event pairs); 2=MTE3-only (skip
    // the V compute, keep the event pairs so the store path is identical); 3=prologue-only
    // (tiling reads + the two staged loads, no pattern setup or column work). Results are
    // INVALID for any dbgMode != 0 -- attribution only, never shipped paths.
    c.dbgMode = tiling.dbgMode;
    // Gated barrier-only dispatch (host twin: the barrierGate assignment in
    // csyr_host.cpp): the band is re-validated here defensively, so a host/kernel desync
    // can only fall back to the production schedule.
    c.barrierGate = (tiling.barrierGate != 0u && c.uplo == 0u &&
                     c.n >= CSYR_BARRIER_GATE_NMIN && c.n <= CSYR_BARRIER_GATE_NMAX)
                        ? 1u : 0u;
    c.blockIdx = blockIdx;
    c.calNum = csyr_cal_num_of(blockIdx, c.n, c.coreNum);
    return c;
}

// Stage helper: the layout-planning block (interleaved-domain dispatch, slim-slot budget
// clamp); batchCols leaves this helper as the clamped effBatch.
__aicore__ __inline__ __attribute__((always_inline)) void csyr_plan_batch(CsyrCtx& c)
{
    // Interleaved-domain dispatch, host-staged form. Both UPPER and LOWER use the slim
    // 2-op layout: per-column slot = 2*nPad interleaved stream (+64-float head dst on
    // LOWER). LOWER is unconditionally interleaved -- the slim slot fits every
    // n <= CSYR_MAX_N at B=1 (the host rejects n > CSYR_MAX_N before the launch). The
    // kernel still re-validates the 192KB budget defensively and clamps batchCols.
    c.lowerIlv = 0;
    c.slotStride = 2 * c.nPad;
    if (c.uplo == 0) {
        c.lowerIlv = 1;
        c.slotStride = 2 * c.nPad + CSYR_HEAD_FLTS;
    }
    uint32_t effBatch = (c.batchCols >= 1u) ? c.batchCols : 1u;
    uint32_t ilvLayout = (c.uplo != 0u || c.lowerIlv != 0u) ? 1u : 0u;
    if (ilvLayout != 0u) {
        // Slim-slot budget (host twin below, MUST stay in sync):
        //   CSYR_RESIDENT_NPAD*nPad (xraw + s') + S*B*slotStride (slots: S=CSYR_GROUPS
        //   groups x B columns) + CSYR_PAT_RESERVE_FLTS (head patterns: 4 select word pairs
        //   at stride-8 uint32 placement, footprint 26 uint32 = 104B, reserved as 32 floats)
        //   + (S*B < 2 ? CSYR_TEMPS_NPAD*nPad : 0) (temps parking term) <= 49152 floats.
        while (effBatch > 2u &&
               (CSYR_RESIDENT_NPAD * c.nPad + CSYR_GROUPS * effBatch * c.slotStride +
                   CSYR_PAT_RESERVE_FLTS) > CSYR_UB_FLOATS) {
            effBatch--;
        }
        if ((CSYR_RESIDENT_NPAD * c.nPad + CSYR_GROUPS * effBatch * c.slotStride +
                CSYR_PAT_RESERVE_FLTS) > CSYR_UB_FLOATS) {
            effBatch = 1u;
        }
        uint32_t parkTemps = (CSYR_GROUPS * effBatch < 2u) ? (CSYR_TEMPS_NPAD * c.nPad) : 0u;
        (void)parkTemps;  // reservation term only (see the budget comment); no writer remains
        // No defensive SoA fallback is needed here: B=1 is the terminal clamp (B=1
        // total <= 10*nPad + 224 <= 48864, parkTemps == 0 whenever S*B >= 2), and the
        // host bounds n <= CSYR_MAX_N before the launch.
    }
    c.batchCols = effBatch;
}

// Stage helper: the resident UB views + the SetAtomicAdd position. UB layout (floats;
// byte offset = floats * 4, all nPad-multiple => 256B aligned):
//   UPPER interleaved (slim 2-op layout; total (4 + 2*S*B)*nPad + 32 [+2nPad at B=1] <=
//     49152): xraw RESIDENT (scalar source + muls source), loaded directly from x_staged;
//     [2nPad,4nPad) s' = [-xim | xre], loaded directly from s'_staged; [4nPad,
//     +2*S*B*nPad) slots of 2nPad each.
//   LOWER interleaved (slim 2-op + head-split store; total (4+2*S*B)*nPad + 2*S*B*64 +
//     32 [+2nPad at B=1] <= 49152): same as UPPER with slot = (2nPad + 64): stream half +
//     64-float head-shift dst; the 4 head-shift pattern word pairs (stride-8 uint32
//     placement, footprint 26 uint32 = 104B, reserved as 32 floats) live past the slot
//     region.
//   The calc_re/calc_im and SoA x views below are retained only so the frozen
//   compute/store function signatures stay identical (their SoA branches are unreachable:
//   lowerIlv is forced to 1 for every LOWER case).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_take_resident_views(
    AsdopsBuffer<ArchType::ASCEND_V220>& buf, CsyrCtx& c)
{
    // Byte offsets are floats*sizeof(float), not bare *4.
    c.ub_xraw = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    c.ub_sprime =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(static_cast<uint32_t>(2 * c.nPad * sizeof(float)));
    c.ub_calc_re = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    c.ub_calc_im =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(static_cast<uint32_t>(c.nPad * sizeof(float)));
    c.ub_xre =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(static_cast<uint32_t>(2 * c.nPad * sizeof(float)));
    c.ub_xim =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(static_cast<uint32_t>(3 * c.nPad * sizeof(float)));
    // SetAtomicAdd is issued AFTER all buffer views are taken.
    AscendC::SetAtomicAdd<float>();
}

// Stage helper: the LOWER pair-enable check, the group/overlay base arithmetic and the
// group UB views.
__aicore__ __inline__ __attribute__((always_inline)) void csyr_take_group_views(
    AsdopsBuffer<ArchType::ASCEND_V220>& buf, CsyrCtx& c)
{
    uint32_t slotBaseFloats = CSYR_RESIDENT_NPAD * c.nPad;
    // LOWER two-column software interleave enable (host twin: the uplo-dependent batchCols
    // budget in csyr_host.cpp -- MUST stay in sync): enabled iff the augmented layout with
    // one extra 2*nPad calc pair still fits the 192KB UB,
    //   (CSYR_SOA_BASE_NPAD + CSYR_SLOT_STREAM_NPAD*S*B)*nPad + CSYR_TABLE_FIX_FLTS <=
    //   CSYR_UB_FLOATS  (S = CSYR_GROUPS).
    // Superseded by the interleaved domain whenever lowerIlv is active (every LOWER n in
    // the production dispatch).
    uint32_t lowerPair = 0;
    if (c.uplo == 0 && c.lowerIlv == 0 &&
        ((CSYR_SOA_BASE_NPAD + CSYR_SLOT_STREAM_NPAD * CSYR_GROUPS * c.batchCols) * c.nPad +
            CSYR_TABLE_FIX_FLTS) <= CSYR_UB_FLOATS) {
        lowerPair = 1;
    }
    c.lowerPair = lowerPair;
    uint32_t calcBBaseFloats =
        (lowerPair != 0) ? ((CSYR_RESIDENT_NPAD + CSYR_SLOT_STREAM_NPAD * CSYR_GROUPS * c.batchCols) * c.nPad)
                         : 0u;
    uint32_t tableBaseFloats;
    uint32_t tempsBaseFloats;
    uint32_t patBaseFloats;
    uint32_t ilvLayout = (c.uplo != 0u || c.lowerIlv != 0u) ? 1u : 0u;
    if (ilvLayout != 0u) {
        // Slim ilv layout: the setup interleave table overlays the first slot (first
        // slotStride floats >= 2nPad); the s'-build temps (2nPad) overlay the second slot in
        // index order (S*B >= 2 with S=3, dead during setup), else park past the slot region;
        // the head-shift patterns live past the slots/temps (stride-8 uint32 placement,
        // footprint 26 uint32 = 104B; budget reserves 32 floats).
        tableBaseFloats = CSYR_RESIDENT_NPAD * c.nPad;
        tempsBaseFloats = (CSYR_GROUPS * c.batchCols >= 2u)
                              ? (CSYR_RESIDENT_NPAD * c.nPad + c.slotStride)
                              : (CSYR_RESIDENT_NPAD * c.nPad +
                                    CSYR_GROUPS * c.batchCols * c.slotStride);
        patBaseFloats = CSYR_RESIDENT_NPAD * c.nPad + CSYR_GROUPS * c.batchCols * c.slotStride +
                        ((CSYR_GROUPS * c.batchCols < 2u) ? (CSYR_TEMPS_NPAD * c.nPad) : 0u);
    } else {
        tableBaseFloats = (CSYR_RESIDENT_NPAD + CSYR_SLOT_STREAM_NPAD * lowerPair +
                           CSYR_SLOT_STREAM_NPAD * CSYR_GROUPS * c.batchCols) * c.nPad;
        tempsBaseFloats = 0; // s'-build does not run on the SoA paths
        patBaseFloats = 0;
    }
    c.ub_out_base =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(static_cast<uint32_t>(slotBaseFloats * sizeof(float)));
    c.ub_calc_re2 =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(static_cast<uint32_t>(calcBBaseFloats * sizeof(float)));
    c.ub_calc_im2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(
        static_cast<uint32_t>((calcBBaseFloats + c.nPad) * sizeof(float)));
    AscendC::LocalTensor<float> ub_table_f =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(static_cast<uint32_t>(tableBaseFloats * sizeof(float)));
    (void)ub_table_f;  // float-typed alias of ub_table (same offset); no reader in the
                       // staged form
    c.ub_table = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(
        static_cast<uint32_t>(tableBaseFloats * sizeof(float)));
    // 4 head-shift select patterns past the slot region, placed at stride 8 uint32
    // (the pattern region footprint is 26 uint32 = 104B: pair h writes words [8h, 8h+2),
    // h=3 reaches word 25): pair [8h, 8h+2) selects elements [2h, 64) of the single head
    // repeat. The budget above reserves 32 floats (128B) for the region.
    c.ub_pat0 = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(
        static_cast<uint32_t>(patBaseFloats * sizeof(float)));
    // s'-build temps: overlay slot[1] (effBatch >= 2, dead during setup) or park past the
    // slot region (effBatch == 1).
    AscendC::LocalTensor<float> ub_temps =
        buf.GetBuffer<BufferType::ASCEND_UB, float>(static_cast<uint32_t>(tempsBaseFloats * sizeof(float)));
    (void)ub_temps;  // overlay reservation only (see the budget comment); no writer remains
}

// Stage helper: the host-staged prologue (two resident MTE2 loads + event pairing + the
// full-pipe barrier).
__aicore__ __inline__ __attribute__((always_inline)) void csyr_staged_loads(
    CsyrCtx& c, __gm__ float* d_A, __gm__ float* d_staged)
{
    AscendC::GlobalTensor<float> gm_staged_tensor;
    AscendC::GlobalTensor<float> gm_sprime_tensor;
    c.gm_A.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_A));
    gm_staged_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_staged));
    gm_sprime_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_staged) + c.stagedFloats);

    // Host-staged prologue: the two resident views load directly from the combined
    // [x_staged | s'_staged] device buffer, each half a clean 512B-multiple MTE2 burst
    // (ceil(2n/128)*128 floats: the valid 8n bytes plus the host-written zero tail -- the
    // tail never feeds a stored element; see the staging note at the top of this file).
    // MTE2->V direction only; 1:1 SET/WAIT pairing on E0/E1. The MTE2 queue is in-order
    // and x_staged is issued first -- it is the operand the fold scalars and the first
    // muls consume.
    csyr_copy_gm2ub(c.ub_xraw, gm_staged_tensor, c.stagedFloats);
    SET_FLAG(MTE2, V, EVENT_ID0);
    csyr_copy_gm2ub(c.ub_sprime, gm_sprime_tensor, c.stagedFloats);
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID0); // x_staged resident (fold scalar source + muls source)
    WAIT_FLAG(MTE2, V, EVENT_ID1); // s'_staged resident (Axpy source)
    // The two staged loads are followed by one full-pipe barrier: csyr_fold_scalar_ilv
    // reads x_staged with GetValue (a scalar-pipe UB read), and without this barrier the
    // fold scalar read races the two staged MTE2 loads (intermittent UPPER precision
    // failures; LOWER was masked by the head-pattern block's own PIPE_BARRIER). The
    // barrier orders the MTE2 writes against the SCALAR pipe as well; it runs in every
    // dbgMode (dbg3 keeps its own barrier, inert duplicate) and on both uplos.
    // Kept full-pipe deliberately: this is a genuine
    // cross-pipe memory-visibility gate (MTE2 writes -> SCALAR-pipe reads), and the
    // adjacent WAIT_FLAG(MTE2,V) pairs empirically gate ONLY the V pipe on this stack
    // (see the note above) -- a single-pipe MTE2 barrier's SCALAR-side ordering is not a
    // proven substitute here, and the failure mode is intermittent precision loss. It
    // is O(1) per kernel (prologue) and sits on no hot loop, so narrowing it buys no
    // throughput while carrying real regression risk.
    PIPE_BARRIER(ALL);
}

// Stage helper: the LOWER head-shift pattern prologue.
__aicore__ __inline__ __attribute__((always_inline)) void csyr_write_head_patterns(
    AscendC::LocalTensor<uint32_t> ub_pat0)
{
    // Head-shift select patterns (LSB-first): element i <-> bit i; the
    // pair [8h, 8h+2) selects floats [2h, 64) of the single head repeat for h = j & 3
    // in [0,3]. The per-column head GatherMask runs with repeatNum=1, so only this
    // word pair is consumed (no uninitialized pattern words are ever read).
    for (uint32_t h = 0; h < 4; h++) {
        uint32_t dropF = 2 * h;
        uint32_t low = (dropF == 0) ? 0xFFFFFFFFu : (0xFFFFFFFFu << dropF);
        ub_pat0.SetValue(8 * h, low);
        ub_pat0.SetValue(8 * h + 1, 0xFFFFFFFFu);
    }
    // Kept full-pipe deliberately: a genuine cross-pipe
    // memory-visibility gate (SCALAR-pipe SetValue writes -> V-pipe GatherMask reads in
    // every LOWER compute batch); the same intermittent-precision failure class as the
    // staged-load barrier above (this block's barrier is what MASKED that bug on LOWER).
    // It is O(1) per kernel (prologue, before the column loop) and sits on no hot loop.
    PIPE_BARRIER(ALL);
}

#endif

// d_x is retained in the signature for launch-descriptor stability only: the resident x
// view is host-staged inside d_staged ([x_staged | s'_staged]), so the kernel never
// dereferences d_x (the host read it back before the launch).
extern "C" __global__ __aicore__ __vector__ void csyr(__gm__ float *__restrict__ d_x,
                                                      __gm__ float *__restrict__ d_A,
                                                      __gm__ float *__restrict__ d_staged,
                                                      __gm__ float *__restrict__ workspace,
                                                      CsyrKernelTiling tiling)
{
#if __DAV_C220_VEC__
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    CsyrCtx c = csyr_read_ctx(tiling, blockIdx);

    if (c.n == 0 || c.calNum == 0) {
        return;
    }

    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    c.nPad = (c.n + (CSYR_VEC_GRAN - 1u)) / CSYR_VEC_GRAN * CSYR_VEC_GRAN;
    // Staged half length in floats (host twin: CSYR_STAGE_GRAN arithmetic in
    // csyr_host.cpp -- MUST stay in sync): ceil(2n/128)*128, i.e. the 8n valid bytes
    // rounded up to a 512B-multiple span, zero tail included. Fits the 2*nPad xraw / s'
    // regions for every n (128 | 2*nPad, and 2*nPad >= 2n).
    c.stagedFloats = ((2u * c.n + 127u) / 128u) * 128u;

    // The body is sequenced through the always-inline stage helpers above (see each
    // helper's header note).
    csyr_plan_batch(c);
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    csyr_take_resident_views(buf, c);
    csyr_take_group_views(buf, c);
    csyr_staged_loads(c, d_A, d_staged);

    if (c.dbgMode == 3) {
        // Prologue-only attribution mode (results INVALID -- never shipped): return
        // immediately after the two staged loads complete, before the head-pattern setup
        // and any column work or MTE3. Complements dbgMode=1 (V-only): prologue = dbg3
        // wall, column stream = dbg1 - dbg3. In the staged form the prologue IS the two
        // loads, so dbg3 is the pure staged-load floor. The previous
        // PIPE_BARRIER(ALL) here is REMOVED: nothing is in flight (the
        // staged loads and every other issued op are fully drained by the staged-load
        // barrier above) and dbgMode==3 issues no MTE3 store, so the SetAtomicNone
        // toggle below faces no in-flight atomics (the toggle hazard is specific to
        // stores in flight); the kernel-exit drain covers the rest.
        AscendC::SetAtomicNone();
        return;
    }

    if (c.lowerIlv != 0) {
        csyr_write_head_patterns(c.ub_pat0);
    }

    // Gated dispatch: barrier-only schedule inside the band, production schedule
    // everywhere else. Both arms share the identical context; the gate itself was
    // re-validated in csyr_read_ctx.
    csyr_process_columns_gated(c);

    // Kernel-exit drain (kept full-pipe deliberately):
    // the last batches' MTE3 ATOMIC stores must be FULLY complete before the
    // SetAtomicNone toggle -- the SetAtomicAdd/SetAtomicNone toggle mid-flight is not
    // safe on this stack (see the csyr_store_batch note) and the atomic stores do not
    // self-gate through the event pairs. This cross-pipe drain is the load-bearing
    // safety point of the whole schedule (the host-side per-call free discipline and
    // the sync-API contract both rely on it), fires O(1) times per kernel, and sits on
    // no per-batch/per-column hot loop.
    PIPE_BARRIER(ALL);
    AscendC::SetAtomicNone();
#endif
}

void csyr_kernel_do(GM_ADDR gm_x, GM_ADDR gm_A, GM_ADDR gm_staged, GM_ADDR workSpace,
                    const CsyrKernelTiling &tiling, uint32_t numBlocks, void *stream)
{
    // The tiling is copied into the launch descriptor by value; no GM tiling buffer
    // exists. gm_staged carries the host-uploaded [x_staged | s'_staged] image; gm_x is
    // forwarded for signature stability (not dereferenced by the kernel).
    csyr<<<numBlocks, nullptr, stream>>>((float *)gm_x, (float *)gm_A, (float *)gm_staged,
                                         (float *)workSpace, tiling);
}
