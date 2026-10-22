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
 * \file srotmg_compute.h
 * \brief Pure-scalar srotmg algorithm used by the host CPU path. The arch22
 *        device kernel keeps an equivalent member-function copy (the kernel
 *        compiler does not inline free functions across units); both stay in
 *        sync via the dual-path CSV tests.
 *
 *        Algorithm aligned with the netlib reference srotmg
 *        (https://www.netlib.org/blas/srotmg.f) as specified by the task
 *        book: the d1 < 0 branch precedes the flag = -2 quick return; no
 *        degenerate-input special case (those fall through to the regular
 *        branches); the su <= 0 safety branch is kept; two combined rescale
 *        loops reset the implied h values to +-1 as the flag flips;
 *        GAMSQ / RGAMSQ use the netlib constants (1.67772e7 / 5.96046e-8).
 *
 *        Plain inline C++ only — no ACL / AscendC dependencies.
 */

#pragma once

#include <cstdint>

// Internal linkage (static) keeps each including unit's copy private (no ODR clash).

namespace srotmg {

constexpr float ZERO = 0.0f;
constexpr float ONE = 1.0f;
constexpr float FLAG_M2 = -2.0f;
// netlib srotmg constants (the task book lists the same values)
constexpr float GAM = 4096.0f;
constexpr float GAMSQ = 1.67772e7f;
constexpr float RGAMSQ = 5.96046e-8f;
// Non-finite guard: inf keeps the rescale loops spinning forever (the golden
// does too); finite inputs converge within a handful of passes, far below
// the cap.
constexpr uint32_t MAX_RESCALE_LOOPS = 64;

static inline float Abs(float value) { return value < ZERO ? -value : value; }

// netlib: flip the stored flag to -1 and reset the implied h slots to +-1
// before each rescale round, exactly as the FORTRAN does inside its loops.
static inline void FlipFlagAndResetH(float& dh11, float& dh12, float& dh21, float& dh22, float& dflag)
{
    if (dflag == ZERO) {
        dh11 = ONE;
        dh22 = ONE;
        dflag = -ONE;
    } else {
        dh21 = -ONE;
        dh12 = ONE;
        dflag = -ONE;
    }
}

// netlib SCALE-CHECK for SD1: single combined loop; the implied h values
// are reset to +-1 each round as the flag flips to -1.
static inline void RescaleD1(float& sd1, float& sx1, float& dh11, float& dh12, float& dh21, float& dh22,
                             float& dflag)
{
    if (sd1 != ZERO) {
        uint32_t guard = 0;
        while ((sd1 <= RGAMSQ) || (sd1 >= GAMSQ)) {
            if (++guard > MAX_RESCALE_LOOPS) {
                break;
            }
            FlipFlagAndResetH(dh11, dh12, dh21, dh22, dflag);
            if (sd1 <= RGAMSQ) {
                sd1 = sd1 * GAM * GAM;
                sx1 = sx1 / GAM;
                dh11 = dh11 / GAM;
                dh12 = dh12 / GAM;
            } else {
                sd1 = sd1 / (GAM * GAM);
                sx1 = sx1 * GAM;
                dh11 = dh11 * GAM;
                dh12 = dh12 * GAM;
            }
        }
    }
}

// netlib SCALE-CHECK for SD2: same structure, updating dh21/dh22.
static inline void RescaleD2(float& sd2, float& dh11, float& dh12, float& dh21, float& dh22, float& dflag)
{
    if (sd2 != ZERO) {
        uint32_t guard = 0;
        while ((Abs(sd2) <= RGAMSQ) || (Abs(sd2) >= GAMSQ)) {
            if (++guard > MAX_RESCALE_LOOPS) {
                break;
            }
            FlipFlagAndResetH(dh11, dh12, dh21, dh22, dflag);
            if (Abs(sd2) <= RGAMSQ) {
                sd2 = sd2 * GAM * GAM;
                dh21 = dh21 / GAM;
                dh22 = dh22 / GAM;
            } else {
                sd2 = sd2 / (GAM * GAM);
                dh21 = dh21 * GAM;
                dh22 = dh22 * GAM;
            }
        }
    }
}

/*!
 * \brief netlib SCALE-CHECK entry: two combined loops; the implied h values
 *        are reset to +-1 each round as the flag flips to -1. Each loop is
 *        guarded against non-finite leftovers only; finite inputs converge
 *        within a handful of passes, far below the cap.
 */
static inline void Rescale(float& sd1, float& sd2, float& sx1,
                           float& dh11, float& dh12, float& dh21, float& dh22, float& dflag)
{
    RescaleD1(sd1, sx1, dh11, dh12, dh21, dh22, dflag);
    RescaleD2(sd2, dh11, dh12, dh21, dh22, dflag);
}

/*!
 * \brief Main branch decision of the netlib srotmg algorithm, reached only
 *        after the caller has screened the flag == -2 quick-return inputs
 *        (sp2 = d2*y1 == 0) and the d1 < 0 all-zero branch. Divisors below
 *        cannot be zero on the paths that reach them: a zero sx1 makes
 *        dq1 = 0 which never dominates, and sy1 == 0 is screened by the
 *        caller's -2 return; the ?: guards keep that invariant visible to
 *        static tools.
 */
static inline void ComputeBranch(
    float& sd1, float& sd2, float& sx1, const float sy1, float& dflag, float& dh11, float& dh21, float& dh12,
    float& dh22)
{
    const float sp2 = sd2 * sy1;
    const float sp1 = sd1 * sx1;
    const float sq2 = sp2 * sy1;
    const float sq1 = sp1 * sx1;

    if (Abs(sq1) > Abs(sq2)) {
        const float safeSx1 = (sx1 != ZERO) ? sx1 : ONE;
        dh21 = -sy1 / safeSx1;
        dh12 = sp2 / sp1;
        const float su = ONE - dh12 * dh21;
        if (su > ZERO) {
            dflag = ZERO;
            sd1 = sd1 / su;
            sd2 = sd2 / su;
            sx1 = sx1 * su;
        } else {
            // netlib safety branch (rounding edge cases, DOI: 10.1145/355841.355847):
            // the whole H matrix is invalidated, matching the arch35 reference.
            dflag = -ONE;
            dh11 = ZERO;
            dh12 = ZERO;
            dh21 = ZERO;
            dh22 = ZERO;
            sd1 = ZERO;
            sd2 = ZERO;
            sx1 = ZERO;
        }
    } else if (sq2 < ZERO) {
        dflag = -ONE;
        sd1 = ZERO;
        sd2 = ZERO;
        sx1 = ZERO;
    } else {
        dflag = ONE;
        dh11 = sp1 / sp2;
        const float safeSy1 = (sy1 != ZERO) ? sy1 : ONE;
        dh22 = sx1 / safeSy1;
        const float su = ONE + dh11 * dh22;
        const float dtemp = sd2 / su;
        sd2 = sd1 / su;
        sd1 = dtemp;
        sx1 = sy1 * su;
    }
}

/*!
 * \brief Core scalar computation. On return dflag is one of -2/-1/0/1; for
 *        dflag == -2 the caller returns immediately (identity transform,
 *        inputs untouched, only param[0] is meaningful).
 */
static inline void ComputeScalars(
    float& sd1, float& sd2, float& sx1, const float sy1, float& dflag, float& dh11, float& dh21, float& dh12,
    float& dh22)
{
    dflag = -ONE;
    dh11 = ZERO;
    dh21 = ZERO;
    dh12 = ZERO;
    dh22 = ZERO;

    // netlib order: the d1 < 0 all-zero branch precedes the flag = -2 return
    if (sd1 < ZERO) {
        sd1 = ZERO;
        sd2 = ZERO;
        sx1 = ZERO;
        return;
    }

    // netlib: sp2 = sd2*sy1 == 0 (covers d2 == 0 and sy1 == 0) -> identity
    if (sd2 * sy1 == ZERO) {
        dflag = FLAG_M2;
        return;
    }

    ComputeBranch(sd1, sd2, sx1, sy1, dflag, dh11, dh21, dh12, dh22);
    Rescale(sd1, sd2, sx1, dh11, dh12, dh21, dh22, dflag);
}
} // namespace srotmg
