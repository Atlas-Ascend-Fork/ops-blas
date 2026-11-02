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
#include "ctrmm_kernel_common.h"

namespace {
__aicore__ inline void LoadTinyA(LocalTensor<float> al, GlobalTensor<float> ag, const CtrmmTiling& t)
{
    // At most three referenced coefficients; UNIT never reads its diagonal.
    for (uint32_t i = 0; i < t.q; ++i) {
        const uint32_t begin = t.upper ? i + t.unit : 0U;
        const uint32_t end = t.upper ? t.q : i + 1U - t.unit;
        for (uint32_t k = begin; k < end; ++k) {
            const uint64_t source =
                t.transpose ? static_cast<uint64_t>(i) * t.lda + k : static_cast<uint64_t>(k) * t.lda + i;
            Read(al[(i * 2U + k) * 8U], ag[2U * source], 2);
        }
    }
}

__aicore__ inline bool ReadTinyInputs(
    LocalTensor<float> al, LocalTensor<float> bl, ComplexValue* av, ComplexValue* bv, const CtrmmTiling& t)
{
    bool sensitive = (t.fastInputLimit == 0.0f) ||
                     ((ctrmm_ieee::BitsOf(t.alphaReal) & ctrmm_ieee::kMagnitudeMask) > kFastAlphaLimitBits ||
                      (ctrmm_ieee::BitsOf(t.alphaImag) & ctrmm_ieee::kMagnitudeMask) > kFastAlphaLimitBits);
    for (uint32_t i = 0; i < t.q; ++i) {
        for (uint32_t k = 0; k < t.q; ++k) {
            ComplexValue value{};
            if (t.unit && i == k) {
                value.real = 1.0f;
            } else if (t.upper ? k >= i : k <= i) {
                const uint32_t index = (i * 2U + k) * 8U;
                value = {al.GetValue(index), al.GetValue(index + 1U)};
                if (t.conjugate) {
                    value.imag = -value.imag;
                }
            }
            av[i * 2U + k] = value;
            sensitive |=
                (ctrmm_ieee::BitsOf(value.real) & ctrmm_ieee::kMagnitudeMask) > ctrmm_ieee::BitsOf(t.fastInputLimit) ||
                (ctrmm_ieee::BitsOf(value.imag) & ctrmm_ieee::kMagnitudeMask) > ctrmm_ieee::BitsOf(t.fastInputLimit);
        }
    }
    for (uint32_t j = 0; j < t.r; ++j) {
        for (uint32_t k = 0; k < t.q; ++k) {
            ComplexValue value{bl.GetValue(j * 8U + k * 2U), bl.GetValue(j * 8U + k * 2U + 1U)};
            bv[j * 2U + k] = value;
            sensitive |=
                (ctrmm_ieee::BitsOf(value.real) & ctrmm_ieee::kMagnitudeMask) > ctrmm_ieee::BitsOf(t.fastInputLimit) ||
                (ctrmm_ieee::BitsOf(value.imag) & ctrmm_ieee::kMagnitudeMask) > ctrmm_ieee::BitsOf(t.fastInputLimit);
        }
    }
    return sensitive;
}

__aicore__ inline bool ComputeTinyFast(
    const ComplexValue* av, const ComplexValue* bv, const CtrmmTiling& t, uint32_t i, uint32_t j, ComplexValue& y)
{
    const ComplexValue alpha{t.alphaReal, t.alphaImag};
    float magnitudeSum = 0.0f;
    for (uint32_t k = 0; k < t.q; ++k) {
        const auto x = bv[j * 2U + k];
        const auto f = av[i * 2U + k];
        const float fNorm = ctrmm_ieee::FromBits(ctrmm_ieee::BitsOf(f.real) & ctrmm_ieee::kMagnitudeMask) +
                            ctrmm_ieee::FromBits(ctrmm_ieee::BitsOf(f.imag) & ctrmm_ieee::kMagnitudeMask);
        const float xNorm = ctrmm_ieee::FromBits(ctrmm_ieee::BitsOf(x.real) & ctrmm_ieee::kMagnitudeMask) +
                            ctrmm_ieee::FromBits(ctrmm_ieee::BitsOf(x.imag) & ctrmm_ieee::kMagnitudeMask);
        magnitudeSum += fNorm * xNorm;
        y.real += f.real * x.real - f.imag * x.imag;
        y.imag += f.real * x.imag + f.imag * x.real;
    }
    const float real = alpha.real * y.real - alpha.imag * y.imag;
    y.imag = alpha.real * y.imag + alpha.imag * y.real;
    y.real = real;
    // Unit roundoff u=2^-24. Each complex product has error
    // bounded by gamma_2 times its absolute real-product sum.
    // The two q<=2 evaluation orders together need <=2*gamma_5
    // for general alpha, or <=2*gamma_(q+1) for alpha=1.
    // 32u / (8u,16u) leave slack for the computed magnitude sum.
    const float alphaNorm = ctrmm_ieee::FromBits(ctrmm_ieee::BitsOf(alpha.real) & ctrmm_ieee::kMagnitudeMask) +
                            ctrmm_ieee::FromBits(ctrmm_ieee::BitsOf(alpha.imag) & ctrmm_ieee::kMagnitudeMask);
    const float factor = alpha.real == 1.0f && alpha.imag == 0.0f ?
                             (t.q == 1U ? kTinyOneTermError : kTinyTwoTermError) :
                             kTinyScaledError;
    const float bound = magnitudeSum * alphaNorm * factor;
    const float absReal = ctrmm_ieee::FromBits(ctrmm_ieee::BitsOf(y.real) & ctrmm_ieee::kMagnitudeMask);
    const float absImag = ctrmm_ieee::FromBits(ctrmm_ieee::BitsOf(y.imag) & ctrmm_ieee::kMagnitudeMask);
    const float tolReal =
        (1.0f + (absReal > bound ? absReal - bound : 0.0f)) * (kTinyToleranceMargin / kMixedToleranceDivisor);
    const float tolImag =
        (1.0f + (absImag > bound ? absImag - bound : 0.0f)) * (kTinyToleranceMargin / kMixedToleranceDivisor);
    return bound > kTinyAbsoluteBudget || bound > tolReal || bound > tolImag;
}

__aicore__ inline ComplexValue ComputeTinyReference(
    const ComplexValue* av, const ComplexValue* bv, const CtrmmTiling& t, uint32_t i, uint32_t j)
{
    const ComplexValue alpha{t.alphaReal, t.alphaImag};
    ComplexValue y;
    const bool normal = t.originalTrans == ACLBLAS_OP_N;
    const auto x = bv[j * 2U + i];
    const auto diagonal = av[i * 2U + i];
    if (normal) {
        y = IsZero(x) ? x : ComplexProduct(alpha, x, true);
        if (!IsZero(x) && !t.unit) {
            y = ComplexProduct(y, diagonal);
        }
    } else {
        y = t.unit ? x : ComplexProduct(x, diagonal);
    }
    const uint32_t begin = t.upper ? i + 1U : 0U;
    const uint32_t end = t.upper ? t.q : i;
    for (uint32_t k = begin; k < end; ++k) {
        const auto xk = bv[j * 2U + k];
        if (normal && IsZero(xk)) {
            continue;
        }
        const auto f = av[i * 2U + k];
        const auto term = normal ? ComplexProduct(ComplexProduct(alpha, xk, true), f) : ComplexProduct(f, xk, true);
        y.real += term.real;
        y.imag += term.imag;
    }
    if (!normal) {
        y = ComplexProduct(alpha, y, true);
    }
    return y;
}

} // namespace
