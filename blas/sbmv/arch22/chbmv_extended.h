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
#include "chbmv_tiling_data.h"

namespace chbmv_extended {
union FloatBits {
    float value;
    uint32_t bits;
};

__aicore__ inline bool Finite(float value)
{
    FloatBits bits{value};
    return (bits.bits & 0x7fffffffU) < 0x7f800000U;
}

// An explicit exponent prevents intermediate overflow before cancellation;
// two FP32 mantissa parts retain multiplication and summation roundoff. This
// slow path is used only when the ordinary result has a nonfinite component.
struct Number {
    float fraction;
    int exponent;
    float tail = 0;
};

__aicore__ inline Number From(float value)
{
    FloatBits bits{value};
    int exponent = (bits.bits >> 23) & 255;
    int correction = 0;
    if (exponent == 255 || (bits.bits & 0x7fffffffU) == 0)
        return {value, 0};
    if (exponent == 0) {
        bits.value = value * 16777216.0f;
        exponent = (bits.bits >> 23) & 255;
        correction = -24;
    }
    bits.bits = (bits.bits & 0x807fffffU) | 0x3f800000U;
    return {bits.value, exponent - 127 + correction};
}

__aicore__ inline float Shift(float value, int exponent)
{
    if (!Finite(value) || value == 0)
        return value;
    if (exponent > 127) {
        FloatBits bits{value};
        bits.bits = (bits.bits & 0x80000000U) | 0x7f800000U;
        return bits.value;
    }
    if (exponent < -150)
        return value * 0.0f;
    if (exponent < -126) {
        value *= 0x1p-126f;
        exponent += 126;
    }
    FloatBits scale{};
    scale.bits = static_cast<uint32_t>(exponent + 127) << 23;
    return value * scale.value;
}

__aicore__ inline Number Normalize(float high, float low, int exponent)
{
    const float sum = high + low;
    const float error = low - (sum - high);
    auto result = From(sum);
    if (Finite(sum) && sum != 0)
        result.tail = Shift(error, -result.exponent);
    result.exponent += exponent;
    return result;
}

__aicore__ inline Number Add(Number a, Number b)
{
    if (!Finite(a.fraction) || !Finite(b.fraction))
        return From(a.fraction + b.fraction);
    if (a.fraction == 0)
        return b;
    if (b.fraction == 0)
        return a;
    const int exponent = a.exponent > b.exponent ? a.exponent : b.exponent;
    const float left = Shift(a.fraction, a.exponent - exponent);
    const float right = Shift(b.fraction, b.exponent - exponent);
    const float sum = left + right;
    const float rightPart = sum - left;
    const float error = (left - (sum - rightPart)) + (right - rightPart) + Shift(a.tail, a.exponent - exponent) +
                        Shift(b.tail, b.exponent - exponent);
    return Normalize(sum, error, exponent);
}

__aicore__ inline Number Multiply(Number a, Number b)
{
    const float product = a.fraction * b.fraction;
    if (!Finite(product) || a.fraction == 0 || b.fraction == 0)
        return From(product);
    // Dekker split on normalized mantissas: retain product roundoff without
    // depending on scalar FP64 or fused scalar arithmetic on AIV.
    const float ca = 4097.0f * a.fraction;
    const float cb = 4097.0f * b.fraction;
    const float ah = ca - (ca - a.fraction);
    const float bh = cb - (cb - b.fraction);
    const float al = a.fraction - ah;
    const float bl = b.fraction - bh;
    const float error = ((ah * bh - product) + ah * bl + al * bh) + al * bl + a.fraction * b.tail +
                        a.tail * b.fraction + a.tail * b.tail;
    return Normalize(product, error, a.exponent + b.exponent);
}

struct Complex {
    Number real, imag;
};

__aicore__ inline Complex Load(const AscendC::GlobalTensor<float>& source, int64_t index)
{
    return {From(source.GetValue(2 * index)), From(source.GetValue(2 * index + 1))};
}

__aicore__ inline Complex Add(Complex a, Complex b) { return {Add(a.real, b.real), Add(a.imag, b.imag)}; }

__aicore__ inline Complex Multiply(Complex a, Complex b)
{
    auto negative = Multiply(a.imag, b.imag);
    negative.fraction = -negative.fraction;
    negative.tail = -negative.tail;
    return {Add(Multiply(a.real, b.real), negative), Add(Multiply(a.real, b.imag), Multiply(a.imag, b.real))};
}

struct Result {
    float real, imag;
};

__aicore__ inline Complex ProductAt(
    const AscendC::GlobalTensor<float>& a, const AscendC::GlobalTensor<float>& x, const ChbmvTilingData& t,
    uint32_t row, uint32_t col)
{
    const auto xv = Load(x, t.xStart + static_cast<int64_t>(col) * t.incx);
    const int64_t r = row;
    const int64_t c = col;
    if (row == col) {
        const auto diag = From(a.GetValue(2 * (r * t.lda + (t.upper ? t.k : 0))));
        return {Multiply(diag, xv.real), Multiply(diag, xv.imag)};
    } else {
        const bool mirror = t.upper ? col < row : col > row;
        const int64_t index =
            mirror ? r * t.lda + (t.upper ? t.k + c - r : c - r) : c * t.lda + (t.upper ? t.k + r - c : r - c);
        auto av = Load(a, index);
        if (mirror) {
            av.imag.fraction = -av.imag.fraction;
            av.imag.tail = -av.imag.tail;
        }
        return Multiply(av, xv);
    }
}

__aicore__ __attribute__((noinline)) void Row(
    const AscendC::GlobalTensor<float>& a, const AscendC::GlobalTensor<float>& x, const AscendC::GlobalTensor<float>& y,
    const ChbmvTilingData& t, uint32_t row, Result& output)
{
    Complex result{{0, 0}, {0, 0}};
    if (t.betaReal != 0 || t.betaImag != 0) {
        result = Load(y, t.yStart + static_cast<int64_t>(row) * t.incy);
        if (t.betaReal != 1 || t.betaImag != 0)
            result = Multiply(Complex{From(t.betaReal), From(t.betaImag)}, result);
    }
    if (t.alphaReal != 0 || t.alphaImag != 0) {
        Complex sum{{0, 0}, {0, 0}};
        const uint32_t first = row > t.band ? row - t.band : 0;
        const uint32_t last = row + (t.band < t.n - 1 - row ? t.band : t.n - 1 - row) + 1;
        for (uint32_t col = first; col < last; ++col) {
            sum = Add(sum, ProductAt(a, x, t, row, col));
        }
        result = Add(result, Multiply(Complex{From(t.alphaReal), From(t.alphaImag)}, sum));
    }
    output.real = Shift(result.real.fraction, result.real.exponent);
    output.imag = Shift(result.imag.fraction, result.imag.exponent);
}
} // namespace chbmv_extended
