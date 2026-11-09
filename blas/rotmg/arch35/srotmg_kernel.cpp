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
 * \brief Device kernel for srotmg scalar computation (all-device path).
 *        Uses GlobalTensor::GetValue / SetValue for direct GM scalar access,
 *        avoiding both SIMT thread overhead and DataCopy pipe complexity.
 *        Only block 0 executes the computation.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "common/helper/kernel_constant.h"
#include "srotmg_tiling_data.h"

using namespace AscendC;

// ==========================================================================
// Helper: netlib flag flip / implied-h reset shared by the SD1/SD2 rescale
// loops below. Kernel-side copy of the arch35 host helper, following the
// arch22 convention (srotmg_compute.h vs the SrotmgKernel member copy): the
// device unit keeps its own copy of the algorithm, with the arch22 kernel's
// dflag/dh11.. parameter naming, so the host and device copies stay
// textually distinct and in sync via the dual-path CSV tests.
// ==========================================================================
static __aicore__ inline void SrotmgFlipFlagAndResetH(float& dh11, float& dh12, float& dh21, float& dh22, float& dflag)
{
    constexpr float ZERO = 0.0f;
    constexpr float ONE = 1.0f;
    if (dflag == ZERO) {
        dh11 = ONE;
        dh22 = ONE;
    } else {
        dh21 = -ONE;
        dh12 = ONE;
    }
    dflag = -ONE;
}

// ==========================================================================
// Kernel entry — scalar GM read/write via GetValue / SetValue
// ==========================================================================
__global__ __aicore__ void srotmg_kernel(
    GM_ADDR d1, GM_ADDR d2, GM_ADDR x1, GM_ADDR y1, GM_ADDR param, SrotmgTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (GetBlockIdx() != 0)
        return;

    constexpr float ZERO = 0.0f;
    constexpr float ONE = 1.0f;
    constexpr float GAM = 4096.0f;
    constexpr float GAMSQ = 1.67772e7f;
    constexpr float RGAMSQ = 5.96046e-8f;
    // issue #871: non-finite d1/d2 keeps the rescale loops spinning forever; cap the
    // rounds like the arch22 kernel (SROTMG_MAX_RESCALE_LOOPS).
    constexpr uint32_t SROTMG_MAX_RESCALE_LOOPS = 64;

    GlobalTensor<float> d1GM, d2GM, x1GM, y1GM, paramGM;
    d1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d1), 1);
    d2GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d2), 1);
    x1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1), 1);
    y1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y1), 1);
    paramGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(param), 5);

    float sd1 = d1GM.GetValue(0);
    float sd2 = d2GM.GetValue(0);
    float sx1 = x1GM.GetValue(0);
    float sy1 = y1GM.GetValue(0);

    float sflag = ZERO;
    float sh11 = ZERO, sh12 = ZERO, sh21 = ZERO, sh22 = ZERO;

    if (sd1 < ZERO) {
        sflag = -ONE;
        sh11 = ZERO;
        sh12 = ZERO;
        sh21 = ZERO;
        sh22 = ZERO;
        sd1 = ZERO;
        sd2 = ZERO;
        sx1 = ZERO;
    } else {
        float sp2 = sd2 * sy1;
        if (sp2 == ZERO) {
            sflag = -2.0f;
            paramGM.SetValue(0, sflag);
            paramGM.SetValue(1, ZERO);
            paramGM.SetValue(2, ZERO);
            paramGM.SetValue(3, ZERO);
            paramGM.SetValue(4, ZERO);
            return;
        }

        float sp1 = sd1 * sx1;
        float sq2 = sp2 * sy1;
        float sq1 = sp1 * sx1;

        float absSq1 = (sq1 < ZERO) ? -sq1 : sq1;
        float absSq2 = (sq2 < ZERO) ? -sq2 : sq2;
        if (absSq1 > absSq2) {
            sh21 = -sy1 / sx1;
            sh12 = sp2 / sp1;
            float su = ONE - sh12 * sh21;
            if (su > ZERO) {
                sflag = ZERO;
                sd1 = sd1 / su;
                sd2 = sd2 / su;
                sx1 = sx1 * su;
            } else {
                sflag = -ONE;
                sh11 = ZERO;
                sh12 = ZERO;
                sh21 = ZERO;
                sh22 = ZERO;
                sd1 = ZERO;
                sd2 = ZERO;
                sx1 = ZERO;
            }
        } else {
            if (sq2 < ZERO) {
                sflag = -ONE;
                sh11 = ZERO;
                sh12 = ZERO;
                sh21 = ZERO;
                sh22 = ZERO;
                sd1 = ZERO;
                sd2 = ZERO;
                sx1 = ZERO;
            } else {
                sflag = ONE;
                sh11 = sp1 / sp2;
                sh22 = sx1 / sy1;
                float su = ONE + sh11 * sh22;
                float stemp = sd2 / su;
                sd2 = sd1 / su;
                sd1 = stemp;
                sx1 = sy1 * su;
            }
        }

        // SCALE-CHECK for SD1
        if (sd1 != ZERO) {
            uint32_t guard = 0;
            while ((sd1 <= RGAMSQ) || (sd1 >= GAMSQ)) {
                if (++guard > SROTMG_MAX_RESCALE_LOOPS) {
                    // issue #871: 非有限值不收敛，超上限退出（对齐 arch22 kernel）
                    break;
                }
                SrotmgFlipFlagAndResetH(sh11, sh12, sh21, sh22, sflag);
                if (sd1 <= RGAMSQ) {
                    sd1 = sd1 * GAM * GAM;
                    sx1 = sx1 / GAM;
                    sh11 = sh11 / GAM;
                    sh12 = sh12 / GAM;
                } else {
                    sd1 = sd1 / (GAM * GAM);
                    sx1 = sx1 * GAM;
                    sh11 = sh11 * GAM;
                    sh12 = sh12 * GAM;
                }
            }
        }

        // SCALE-CHECK for SD2
        if (sd2 != ZERO) {
            float absSd2 = (sd2 < ZERO) ? -sd2 : sd2;
            uint32_t guard = 0;
            while ((absSd2 <= RGAMSQ) || (absSd2 >= GAMSQ)) {
                if (++guard > SROTMG_MAX_RESCALE_LOOPS) {
                    // issue #871: 非有限值不收敛，超上限退出（对齐 arch22 kernel）
                    break;
                }
                SrotmgFlipFlagAndResetH(sh11, sh12, sh21, sh22, sflag);
                if (absSd2 <= RGAMSQ) {
                    sd2 = sd2 * GAM * GAM;
                    sh21 = sh21 / GAM;
                    sh22 = sh22 / GAM;
                } else {
                    sd2 = sd2 / (GAM * GAM);
                    sh21 = sh21 * GAM;
                    sh22 = sh22 * GAM;
                }
                absSd2 = (sd2 < ZERO) ? -sd2 : sd2;
            }
        }
    }

    // STORE results
    if (sflag < ZERO) {
        paramGM.SetValue(0, sflag);
        paramGM.SetValue(1, sh11);
        paramGM.SetValue(2, sh21);
        paramGM.SetValue(3, sh12);
        paramGM.SetValue(4, sh22);
    } else if (sflag == ZERO) {
        paramGM.SetValue(0, sflag);
        paramGM.SetValue(1, ZERO);
        paramGM.SetValue(2, sh21);
        paramGM.SetValue(3, sh12);
        paramGM.SetValue(4, ZERO);
    } else {
        paramGM.SetValue(0, sflag);
        paramGM.SetValue(1, sh11);
        paramGM.SetValue(2, ZERO);
        paramGM.SetValue(3, ZERO);
        paramGM.SetValue(4, sh22);
    }

    d1GM.SetValue(0, sd1);
    d2GM.SetValue(0, sd2);
    x1GM.SetValue(0, sx1);
}

// ==========================================================================
// Kernel launcher
// ==========================================================================
void srotmg_kernel_do(
    uint8_t* d1, uint8_t* d2, uint8_t* x1, uint8_t* y1, uint8_t* param, const SrotmgTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    srotmg_kernel<<<numBlocks, nullptr, stream>>>(d1, d2, x1, y1, param, tiling);
}
