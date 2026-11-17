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
#include "kernel_operator.h"

// Compensated scalar arithmetic for the existing slow paths. Two float32
// components retain product residuals through cancellation; this is not a
// claim of arbitrary-range float64 equivalence.
namespace Csyr2Precise {
struct Pair {
    float hi;
    float lo;
};
__aicore__ inline bool Finite(float x) { return x == x && x <= 3.402823466e38F && x >= -3.402823466e38F; }
__aicore__ inline Pair Normalize(float hi, float lo)
{
    const float s = hi + lo;
    return Finite(s) ? Pair{s, lo - (s - hi)} : Pair{s, 0.0F};
}
__aicore__ inline Pair Add(Pair a, Pair b)
{
    const float s = a.hi + b.hi;
    if (!Finite(s)) {
        return {s, 0.0F};
    }
    const float v = s - a.hi;
    const float e = (a.hi - (s - v)) + (b.hi - v);
    return Normalize(s, e + (a.lo + b.lo));
}
__aicore__ inline Pair Neg(Pair a) { return {-a.hi, -a.lo}; }
__aicore__ inline Pair Product(float a, float b)
{
    // Split the significand into two 12-bit parts. Explicit rounded products
    // avoid relying on an unavailable scalar fmaf device runtime symbol.
    const volatile float rounded = a * b;
    const float p = rounded;
    if (!Finite(p)) {
        return {p, 0.0F};
    }
    union Bits {
        float value;
        uint32_t bits;
    };
    Bits aa;
    Bits bb;
    aa.value = a;
    bb.value = b;
    aa.bits &= 0xfffff000U;
    bb.bits &= 0xfffff000U;
    const float ah = aa.value;
    const float bh = bb.value;
    const float al = a - ah;
    const float bl = b - bh;
    const volatile float hh = ah * bh;
    const volatile float lh = al * bh;
    const volatile float hl = ah * bl;
    const volatile float ll = al * bl;
    const float e1 = p - hh;
    const float e2 = e1 - lh;
    const float e3 = e2 - hl;
    return {p, ll - e3};
}
__aicore__ inline Pair Scale(Pair a, float b)
{
    const Pair p = Product(a.hi, b);
    return Normalize(p.hi, p.lo + a.lo * b);
}
__aicore__ inline void Update(
    float& ar, float& ai, float alphaR, float alphaI, float xcR, float xcI, float ycR, float ycI, float xR, float xI,
    float yR, float yI)
{
    if (!(Finite(ar) && Finite(ai) && Finite(alphaR) && Finite(alphaI) && Finite(xcR) && Finite(xcI) && Finite(ycR) &&
          Finite(ycI) && Finite(xR) && Finite(xI) && Finite(yR) && Finite(yI))) {
        const float pR = alphaR * xcR - alphaI * xcI;
        const float pI = alphaR * xcI + alphaI * xcR;
        const float qR = alphaR * ycR - alphaI * ycI;
        const float qI = alphaR * ycI + alphaI * ycR;
        ar += (pR * yR - pI * yI) + (qR * xR - qI * xI);
        ai += (pR * yI + pI * yR) + (qR * xI + qI * xR);
        return;
    }
    const Pair pR = Add(Product(alphaR, xcR), Neg(Product(alphaI, xcI)));
    const Pair pI = Add(Product(alphaR, xcI), Product(alphaI, xcR));
    const Pair qR = Add(Product(alphaR, ycR), Neg(Product(alphaI, ycI)));
    const Pair qI = Add(Product(alphaR, ycI), Product(alphaI, ycR));
    const Pair vr = Add(Add(Scale(pR, yR), Neg(Scale(pI, yI))), Add(Scale(qR, xR), Neg(Scale(qI, xI))));
    const Pair vi = Add(Add(Scale(pR, yI), Scale(pI, yR)), Add(Scale(qR, xI), Scale(qI, xR)));
    const Pair real = Add({ar, 0.0F}, vr);
    const Pair imag = Add({ai, 0.0F}, vi);
    ar = real.hi + real.lo;
    ai = imag.hi + imag.lo;
}
} // namespace Csyr2Precise
