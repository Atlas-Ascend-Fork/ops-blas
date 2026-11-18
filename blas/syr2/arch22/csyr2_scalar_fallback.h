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
#include "csyr2_vector_common.h"
#include "csyr2_precise_scalar.h"

namespace Csyr2Vector {
__aicore__ inline bool RequiresPrecise(Vectors& v, uint32_t n, uint32_t rows)
{
    // Design note: the scalar precise fallback is deliberately gated to n == 32.
    // It is calibrated for the unit-stride boundary kernels at that single size
    // (csyr2_banded_vector upper / csyr2_lower_vector lower); every other size
    // intentionally keeps the vectorized path. The direct_vector route carries
    // its own full HasNonfinite scan, which the wide-reference numerics
    // regression (n = 3073/4097/6144) exercises; widening this gate to further
    // paths would change behavior without dedicated regression coverage.
    // `rows` is the per-core loaded span (upper: prefix [0, range.last);
    // lower: suffix [range.first, n)), so the scan bound tracks the loaded
    // data instead of n and must not be folded into the gate.
    if (n != 32U) {
        return false;
    }
    for (uint32_t i = 0; i < 2U * rows; ++i) {
        const float xx = v.xv.GetValue(i);
        const float yy = v.yv.GetValue(i);
        if (xx != xx || yy != yy || xx >= 1.0e10F || xx <= -1.0e10F || yy >= 1.0e10F || yy <= -1.0e10F) {
            return true;
        }
    }
    return false;
}
__aicore__ inline void PreciseColumn(
    LocalTensor<float> dst, Vectors& v, uint32_t coefficient, uint32_t begin, uint32_t end, uint32_t origin,
    const Csyr2TilingData& t)
{
    const float xre = v.xv.GetValue(coefficient);
    const float xim = v.xv.GetValue(coefficient + 1U);
    const float yre = v.yv.GetValue(coefficient);
    const float yim = v.yv.GetValue(coefficient + 1U);
    for (uint32_t r = begin; r < end; ++r) {
        const uint32_t z = 2U * (r - origin);
        const float xrr = v.xv.GetValue(z);
        const float xri = v.xv.GetValue(z + 1U);
        const float yrr = v.yv.GetValue(z);
        const float yri = v.yv.GetValue(z + 1U);
        float ar = dst.GetValue(z);
        float ai = dst.GetValue(z + 1U);
        Csyr2Precise::Update(ar, ai, t.ar, t.ai, xre, xim, yre, yim, xrr, xri, yrr, yri);
        dst.SetValue(z, ar);
        dst.SetValue(z + 1U, ai);
    }
}
} // namespace Csyr2Vector
