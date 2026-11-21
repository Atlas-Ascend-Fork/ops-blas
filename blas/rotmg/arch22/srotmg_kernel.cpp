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
 * \file srotmg_kernel.cpp
 * \brief Device kernel for srotmg scalar computation (all-device path), arch22
 *        (Atlas A2/A3). Algorithm aligned with the netlib reference srotmg
 *        (https://www.netlib.org/blas/srotmg.f) as specified by the task
 *        book. The same algorithm is single-sourced for the host CPU path
 *        in srotmg_compute.h; this unit keeps its own member-function copy
 *        because the kernel compiler does not inline cross-unit free
 *        functions, and an out-of-line call on the AICore costs ~1.3us on
 *        this launch-bound scalar operator.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "acl/acl.h"
#include "srotmg_tiling_data.h"

using namespace AscendC;

namespace {

constexpr uint32_t SROTMG_DATA_COUNT = 1;
constexpr uint32_t SROTMG_PARAM_COUNT = 5;
constexpr float SROTMG_ZERO = 0.0f;
constexpr float SROTMG_ONE = 1.0f;
constexpr float SROTMG_FLAG_M2 = -2.0f;
// netlib srotmg constants (the task book lists the same values)
constexpr float SROTMG_GAM = 4096.0f;
constexpr float SROTMG_GAMSQ = 1.67772e7f;
constexpr float SROTMG_RGAMSQ = 5.96046e-8f;
// Non-finite guard: inf keeps the rescale loops spinning forever (the golden
// does too); finite inputs converge within a handful of passes, far below
// the cap.
constexpr uint32_t SROTMG_MAX_RESCALE_LOOPS = 64;

class SrotmgKernel {
public:
    __aicore__ inline SrotmgKernel() = default;
    __aicore__ inline void Init(const SrotmgTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline float Abs(float value) const;
    __aicore__ inline void StoreIdentityParam();
    __aicore__ inline void StoreZeroResult();
    __aicore__ inline void ComputeBranch(
        float& sd1, float& sd2, float& sx1, const float sy1, float& dflag, float& dh11, float& dh21, float& dh12,
        float& dh22) const;
    __aicore__ inline void RescaleD1(
        float& sd1, float& sx1, float& dh11, float& dh12, float& dh21, float& dh22, float& dflag) const;
    __aicore__ inline void RescaleD2(
        float& sd2, float& dh11, float& dh12, float& dh21, float& dh22, float& dflag) const;
    __aicore__ inline void FlipFlagAndResetH(float& dh11, float& dh12, float& dh21, float& dh22, float& dflag) const;
    __aicore__ inline void StoreParam(float dflag, float dh11, float dh21, float dh12, float dh22);

    GlobalTensor<float> d1GM;
    GlobalTensor<float> d2GM;
    GlobalTensor<float> x1GM;
    GlobalTensor<float> y1GM;
    GlobalTensor<float> paramGM;
};

__aicore__ inline void SrotmgKernel::Init(const SrotmgTilingData& tiling)
{
    d1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.d1), SROTMG_DATA_COUNT);
    d2GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.d2), SROTMG_DATA_COUNT);
    x1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x1), SROTMG_DATA_COUNT);
    y1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.y1), SROTMG_DATA_COUNT);
    paramGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.param), SROTMG_PARAM_COUNT);
}

__aicore__ inline float SrotmgKernel::Abs(float value) const { return value < SROTMG_ZERO ? -value : value; }

// flag = -2 identity transform: inputs untouched, param zeroed.
__aicore__ inline void SrotmgKernel::StoreIdentityParam()
{
    paramGM.SetValue(0, SROTMG_FLAG_M2);
    paramGM.SetValue(1, SROTMG_ZERO);
    paramGM.SetValue(2, SROTMG_ZERO);
    paramGM.SetValue(3, SROTMG_ZERO);
    paramGM.SetValue(4, SROTMG_ZERO);
}

// d1 < 0: flag = -1, H all zeros, d1/d2/x1 overwritten with zeros.
__aicore__ inline void SrotmgKernel::StoreZeroResult()
{
    StoreParam(-SROTMG_ONE, SROTMG_ZERO, SROTMG_ZERO, SROTMG_ZERO, SROTMG_ZERO);
    d1GM.SetValue(0, SROTMG_ZERO);
    d2GM.SetValue(0, SROTMG_ZERO);
    x1GM.SetValue(0, SROTMG_ZERO);
}

// Main branch decision of the netlib srotmg algorithm (regular case).
// Divisors below cannot be zero on the paths that reach them: a zero sx1
// makes sq1 = 0 which never dominates, and sy1 == 0 is screened by the
// caller's flag = -2 return; the ?: guards keep that invariant visible to
// static tools.
__aicore__ inline void SrotmgKernel::ComputeBranch(
    float& sd1, float& sd2, float& sx1, const float sy1, float& dflag, float& dh11, float& dh21, float& dh12,
    float& dh22) const
{
    const float sp2 = sd2 * sy1;
    const float sp1 = sd1 * sx1;
    const float sq2 = sp2 * sy1;
    const float sq1 = sp1 * sx1;

    if (Abs(sq1) > Abs(sq2)) {
        const float safeSx1 = (sx1 != SROTMG_ZERO) ? sx1 : SROTMG_ONE;
        dh21 = -sy1 / safeSx1;
        dh12 = sp2 / sp1;
        const float su = SROTMG_ONE - dh12 * dh21;
        if (su > SROTMG_ZERO) {
            dflag = SROTMG_ZERO;
            sd1 = sd1 / su;
            sd2 = sd2 / su;
            sx1 = sx1 * su;
        } else {
            // netlib safety branch (rounding edge cases, DOI: 10.1145/355841.355847):
            // the whole H matrix is invalidated, matching the arch35 reference.
            dflag = -SROTMG_ONE;
            dh11 = SROTMG_ZERO;
            dh12 = SROTMG_ZERO;
            dh21 = SROTMG_ZERO;
            dh22 = SROTMG_ZERO;
            sd1 = SROTMG_ZERO;
            sd2 = SROTMG_ZERO;
            sx1 = SROTMG_ZERO;
        }
    } else if (sq2 < SROTMG_ZERO) {
        dflag = -SROTMG_ONE;
        sd1 = SROTMG_ZERO;
        sd2 = SROTMG_ZERO;
        sx1 = SROTMG_ZERO;
    } else {
        dflag = SROTMG_ONE;
        dh11 = sp1 / sp2;
        const float safeSy1 = (sy1 != SROTMG_ZERO) ? sy1 : SROTMG_ONE;
        dh22 = sx1 / safeSy1;
        const float su = SROTMG_ONE + dh11 * dh22;
        const float dtemp = sd2 / su;
        sd2 = sd1 / su;
        sd1 = dtemp;
        sx1 = sy1 * su;
    }
}

// netlib: flip the stored flag to -1 and reset the implied h slots to +-1
// before each rescale round, exactly as the FORTRAN does inside its loops.
__aicore__ inline void SrotmgKernel::FlipFlagAndResetH(
    float& dh11, float& dh12, float& dh21, float& dh22, float& dflag) const
{
    if (dflag == SROTMG_ZERO) {
        dh11 = SROTMG_ONE;
        dh22 = SROTMG_ONE;
        dflag = -SROTMG_ONE;
    } else {
        dh21 = -SROTMG_ONE;
        dh12 = SROTMG_ONE;
        dflag = -SROTMG_ONE;
    }
}

// netlib SCALE-CHECK for SD1: single combined loop; the implied h values
// are reset to +-1 each round as the flag flips to -1.
__aicore__ inline void SrotmgKernel::RescaleD1(
    float& sd1, float& sx1, float& dh11, float& dh12, float& dh21, float& dh22, float& dflag) const
{
    if (sd1 == SROTMG_ZERO) {
        return;
    }
    for (uint32_t guard = 0; (sd1 <= SROTMG_RGAMSQ) || (sd1 >= SROTMG_GAMSQ); ++guard) {
        if (guard >= SROTMG_MAX_RESCALE_LOOPS) {
            break;
        }
        FlipFlagAndResetH(dh11, dh12, dh21, dh22, dflag);
        if (sd1 <= SROTMG_RGAMSQ) {
            sd1 = sd1 * SROTMG_GAM * SROTMG_GAM;
            sx1 = sx1 / SROTMG_GAM;
            dh11 = dh11 / SROTMG_GAM;
            dh12 = dh12 / SROTMG_GAM;
        } else {
            sd1 = sd1 / (SROTMG_GAM * SROTMG_GAM);
            sx1 = sx1 * SROTMG_GAM;
            dh11 = dh11 * SROTMG_GAM;
            dh12 = dh12 * SROTMG_GAM;
        }
    }
}

// netlib SCALE-CHECK for SD2: same structure, updating dh21/dh22.
__aicore__ inline void SrotmgKernel::RescaleD2(
    float& sd2, float& dh11, float& dh12, float& dh21, float& dh22, float& dflag) const
{
    if (sd2 == SROTMG_ZERO) {
        return;
    }
    for (uint32_t guard = 0; (Abs(sd2) <= SROTMG_RGAMSQ) || (Abs(sd2) >= SROTMG_GAMSQ); ++guard) {
        if (guard >= SROTMG_MAX_RESCALE_LOOPS) {
            break;
        }
        FlipFlagAndResetH(dh11, dh12, dh21, dh22, dflag);
        if (Abs(sd2) <= SROTMG_RGAMSQ) {
            sd2 = sd2 * SROTMG_GAM * SROTMG_GAM;
            dh21 = dh21 / SROTMG_GAM;
            dh22 = dh22 / SROTMG_GAM;
        } else {
            sd2 = sd2 / (SROTMG_GAM * SROTMG_GAM);
            dh21 = dh21 * SROTMG_GAM;
            dh22 = dh22 * SROTMG_GAM;
        }
    }
}

// Flag-implied slots are written as 0: the golden leaves the caller's
// zero-initialised array untouched there, so both sides compare equal.
__aicore__ inline void SrotmgKernel::StoreParam(float dflag, float dh11, float dh21, float dh12, float dh22)
{
    if (dflag < SROTMG_ZERO) {
        // flag = -1: full matrix stored
        paramGM.SetValue(0, dflag);
        paramGM.SetValue(1, dh11);
        paramGM.SetValue(2, dh21);
        paramGM.SetValue(3, dh12);
        paramGM.SetValue(4, dh22);
    } else if (dflag == SROTMG_ZERO) {
        // flag = 0: [[1, h12], [h21, 1]], implicit diagonal
        paramGM.SetValue(0, dflag);
        paramGM.SetValue(1, SROTMG_ZERO);
        paramGM.SetValue(2, dh21);
        paramGM.SetValue(3, dh12);
        paramGM.SetValue(4, SROTMG_ZERO);
    } else {
        // flag = 1: [[h11, 1], [-1, h22]], implicit off-diagonal
        paramGM.SetValue(0, dflag);
        paramGM.SetValue(1, dh11);
        paramGM.SetValue(2, SROTMG_ZERO);
        paramGM.SetValue(3, SROTMG_ZERO);
        paramGM.SetValue(4, dh22);
    }
}

__aicore__ inline void SrotmgKernel::Process()
{
    float sd1 = d1GM.GetValue(0);
    float sd2 = d2GM.GetValue(0);
    float sx1 = x1GM.GetValue(0);
    const float sy1 = y1GM.GetValue(0);
    // netlib order: the d1 < 0 all-zero branch precedes the flag = -2 return
    if (sd1 < SROTMG_ZERO) {
        StoreZeroResult();
        return;
    }
    // netlib: sp2 = sd2*sy1 == 0 (covers d2 == 0 and sy1 == 0) -> identity
    if (sd2 * sy1 == SROTMG_ZERO) {
        StoreIdentityParam();
        return;
    }

    float dflag = -SROTMG_ONE;
    float dh11 = SROTMG_ZERO;
    float dh21 = SROTMG_ZERO;
    float dh12 = SROTMG_ZERO;
    float dh22 = SROTMG_ZERO;

    ComputeBranch(sd1, sd2, sx1, sy1, dflag, dh11, dh21, dh12, dh22);
    RescaleD1(sd1, sx1, dh11, dh12, dh21, dh22, dflag);
    RescaleD2(sd2, dh11, dh12, dh21, dh22, dflag);

    StoreParam(dflag, dh11, dh21, dh12, dh22);

    d1GM.SetValue(0, sd1);
    d2GM.SetValue(0, sd2);
    x1GM.SetValue(0, sx1);
}

} // namespace

extern "C" __global__ __aicore__ void srotmg_kernel(const SrotmgTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SrotmgKernel op;
    op.Init(tiling);
    op.Process();
}

aclError srotmg_kernel_do(const SrotmgTilingData& tiling, uint32_t numBlocks, void* stream)
{
    // issue #1152: report the launch status instead of discarding it, mirroring the
    // chbmv/ssbmv pattern (aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs).
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(srotmg_kernel), &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        SrotmgTilingData tiling;
    } args{tiling};
    static_assert(sizeof(Arguments) == sizeof(SrotmgTilingData));
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}
