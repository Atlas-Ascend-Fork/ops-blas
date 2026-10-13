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
#include <cmath>
#include <algorithm>
#include "cann_ops_blas.h"

// 缩放式 hypot（LAPY2），避免极端比值下 x*x 下溢
inline float goldenHypot(float x, float y)
{
    float ax = std::abs(x);
    float ay = std::abs(y);
    float big = std::max(ax, ay);
    if (big == 0.0f) return 0.0f;
    return big * std::sqrt((ax / big) * (ax / big) + (ay / big) * (ay / big));
}

inline aclblasStatus_t aclblasCrotg_cpu(
    aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b, float* c, aclblasComplex* s)
{
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    
    float a_r = a->real;
    float a_i = a->imag;
    float b_r = b->real;
    float b_i = b->imag;
    float abs_a = goldenHypot(a_r, a_i);
    float abs_b = goldenHypot(b_r, b_i);
    if (abs_b == 0.0f) {
        *c = 1.0f;
        s->real = 0.0f;
        s->imag = 0.0f;
        a->real = a_r;
        a->imag = a_i;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (abs_a == 0.0f) {
        *c = 0.0f;
        s->real = b_r / abs_b;
        s->imag = -b_i / abs_b;
        a->real = abs_b;
        a->imag = 0.0f;
        return ACLBLAS_STATUS_SUCCESS;
    }

    float scale = std::max(abs_a, abs_b);
    // 除零防护。上面的 abs_b == 0 / abs_a == 0 两个分支已提前返回，走到此处必有 abs_a > 0 且
    // abs_b > 0，故 scale = max(abs_a, abs_b) > 0，此分支在精确算术下不可达。它只为让「除数
    // 非 0」在每个除法点本地可证（静态检查器无法跨分支推断），写入的是与「a、b 全零」时相同的
    // crotg 约定（c=1、s=0、r=0），与 abs_b == 0 分支在 a、b 同为零时完全自洽。
    //
    // 用 `== 0.0f` 而非 `<= 0.0f`：scale 的取值范围恒为非负有限值、+0.0、+Inf 或 NaN
    // （goldenHypot 经 sqrt 只会产生非负值或 NaN），两种写法在该范围上完全等价；特别是
    // NaN 时二者皆为 false，故本分支不会吞掉 NaN 传播路径。检查器只认 `== 0.0f` 形式。
    if (scale == 0.0f) {
        *c = 1.0f;
        s->real = 0.0f;
        s->imag = 0.0f;
        a->real = 0.0f;
        a->imag = 0.0f;
        return ACLBLAS_STATUS_SUCCESS;
    }

    float sum_sq = (a_r / scale) * (a_r / scale) + (a_i / scale) * (a_i / scale) +
                   (b_r / scale) * (b_r / scale) + (b_i / scale) * (b_i / scale);
    float norm = scale * std::sqrt(sum_sq);

    *c = abs_a / norm;
    float sgn_ar = a_r / abs_a;
    float sgn_ai = a_i / abs_a;

    s->real = (sgn_ar * b_r + sgn_ai * b_i) / norm;
    s->imag = (sgn_ai * b_r - sgn_ar * b_i) / norm;
    
    a->real = sgn_ar * norm;
    a->imag = sgn_ai * norm;

    return ACLBLAS_STATUS_SUCCESS;
}
