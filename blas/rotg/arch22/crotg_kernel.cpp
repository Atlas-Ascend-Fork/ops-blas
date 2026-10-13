/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "crotg_tiling_data.h"

using namespace AscendC;

namespace {

// Scaled hypot (LAPY2): sqrt(x*x + y*y) computed with max/abs/sqrt and basic
// arithmetic only, scaling by the larger magnitude first to avoid intermediate
// overflow/underflow of the squared terms for extreme |x|/|y| ratios.
__aicore__ inline float hypotScaled(float x, float y)
{
    float ax = abs(x);
    float ay = abs(y);
    float big = max(ax, ay);
    if (big == 0.0f) {
        return 0.0f;
    }
    return big * sqrt((ax / big) * (ax / big) + (ay / big) * (ay / big));
}

} // namespace

extern "C" __global__ __aicore__ void crotg_kernel(const CrotgTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> aGM, bGM, cGM, sGM;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.a), 2);
    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.b), 2);
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.c), 1);
    sGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.s), 2);

    float a_r = aGM.GetValue(0), a_i = aGM.GetValue(1);
    float b_r = bGM.GetValue(0), b_i = bGM.GetValue(1);
    float c, s_r, s_i, r_r, r_i;

    float ma = max(max(abs(a_r), abs(a_i)), max(abs(b_r), abs(b_i)));
    if (ma == 0.0f) {
        c = 1.0f; s_r = 0.0f; s_i = 0.0f; r_r = 0.0f; r_i = 0.0f;
    } else {
        float a_sr = a_r / ma, a_si = a_i / ma;
        float b_sr = b_r / ma, b_si = b_i / ma;
        float abs_a = hypotScaled(a_sr, a_si);
        float abs_b = hypotScaled(b_sr, b_si);
        if (abs_b == 0.0f) {
            c = 1.0f; s_r = 0.0f; s_i = 0.0f; r_r = a_r; r_i = a_i;
        } else if (abs_a == 0.0f) {
            c = 0.0f; s_r = b_sr / abs_b; s_i = -b_si / abs_b;
            r_r = abs_b * ma; r_i = 0.0f;
        } else {
            float norm_s = hypotScaled(abs_a, abs_b);
            c = abs_a / norm_s;
            float sgn_ar = a_sr / abs_a, sgn_ai = a_si / abs_a;
            s_r = (sgn_ar * b_sr + sgn_ai * b_si) / norm_s;
            s_i = (sgn_ai * b_sr - sgn_ar * b_si) / norm_s;
            r_r = sgn_ar * norm_s * ma;
            r_i = sgn_ai * norm_s * ma;
        }
    }
    aGM.SetValue(0, r_r); aGM.SetValue(1, r_i);
    cGM.SetValue(0, c);
    sGM.SetValue(0, s_r); sGM.SetValue(1, s_i);
}

void crotg_kernel_do(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* s,
                     const CrotgTilingData& tiling, uint32_t numBlocks, void* stream)
{
    crotg_kernel<<<numBlocks, nullptr, stream>>>(tiling);
}
