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
template <uint32_t F>
__aicore__ inline void AccumulateFmaResidual(LocalTensor<float> s, LocalTensor<float> c, uint32_t n)
{
    // TwoSum(product_high, c).
    Sub(s[7U * F], s[6U * F], s[2U * F], n);
    PipeBarrier<PIPE_V>();
    Sub(s[8U * F], s[6U * F], s[7U * F], n);
    Sub(s[9U * F], c, s[7U * F], n);
    PipeBarrier<PIPE_V>();
    Sub(s[8U * F], s[2U * F], s[8U * F], n);
    PipeBarrier<PIPE_V>();
    Add(s[7U * F], s[8U * F], s[9U * F], n);
    PipeBarrier<PIPE_V>();
    // TwoSum(product_low, sum_low).
    Add(s[9U * F], s[3U * F], s[7U * F], n);
    PipeBarrier<PIPE_V>();
    Sub(s[10U * F], s[9U * F], s[3U * F], n);
    PipeBarrier<PIPE_V>();
    Sub(s[11U * F], s[9U * F], s[10U * F], n);
    Sub(s, s[7U * F], s[10U * F], n);
    PipeBarrier<PIPE_V>();
    Sub(s[11U * F], s[3U * F], s[11U * F], n);
    PipeBarrier<PIPE_V>();
    Add(s[10U * F], s[11U * F], s, n);
}

template <uint32_t F>
__aicore__ inline void RoundVectorFma(
    LocalTensor<float> dst, LocalTensor<float> s, LocalTensor<uint8_t> masks, uint32_t n)
{
    // At a rounding midpoint the low residual selects the correct side.
    auto other = masks[F / kBitsPerByte];
    Duplicate(s.ReinterpretCast<int32_t>(), kFmaMidpointMask, n);
    PipeBarrier<PIPE_V>();
    And(s[F].ReinterpretCast<uint16_t>(), s[9U * F].ReinterpretCast<uint16_t>(), s.ReinterpretCast<uint16_t>(), n * 2U);
    PipeBarrier<PIPE_V>();
    CompareScalar(masks, s[F].ReinterpretCast<int32_t>(), int32_t{0}, CMPMODE::EQ, n);
    CompareScalar(other, s[10U * F], 0.0f, CMPMODE::NE, n);
    PipeBarrier<PIPE_V>();
    And(masks.ReinterpretCast<uint16_t>(), masks.ReinterpretCast<uint16_t>(), other.ReinterpretCast<uint16_t>(),
        n / kBitsPerHalfWord);
    ShiftRight(
        s[F].ReinterpretCast<uint32_t>(), s[9U * F].ReinterpretCast<uint32_t>(), uint32_t{ctrmm_ieee::kSignShift}, n);
    ShiftRight(
        s[2U * F].ReinterpretCast<uint32_t>(), s[10U * F].ReinterpretCast<uint32_t>(), uint32_t{ctrmm_ieee::kSignShift},
        n);
    PipeBarrier<PIPE_V>();
    Compare(other, s[F].ReinterpretCast<int32_t>(), s[2U * F].ReinterpretCast<int32_t>(), CMPMODE::EQ, n);
    Muls(s[3U * F], s[9U * F], kResidualRoundUp, n);
    Muls(s[4U * F], s[9U * F], kResidualRoundDown, n);
    PipeBarrier<PIPE_V>();
    Select(s[5U * F], other, s[3U * F], s[4U * F], SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    PipeBarrier<PIPE_V>();
    Select(s[5U * F], masks, s[5U * F], s[9U * F], SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    PipeBarrier<PIPE_V>();
    Add(dst, s[6U * F], s[5U * F], n);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ScalarFmaProduct(
    LocalTensor<float> s, float a, LocalTensor<float> b, LocalTensor<float> c, uint32_t n)
{
    // Bisheng can fold the scalar Dekker split to ah=a even with
    // -ffp-contract=off. Rounding the significand by integer bits keeps
    // this essential rounding step explicit and leaves a <=11-bit low.
    const uint32_t bits = ctrmm_ieee::BitsOf(a);
    const uint32_t magnitude = bits & ctrmm_ieee::kMagnitudeMask;
    const uint32_t high = (magnitude + kSplitRoundBias + ((magnitude >> kSplitLowBits) & 1U)) & kSplitHighMask;
    const float ah = ctrmm_ieee::FromBits((bits & ctrmm_ieee::kSignMask) | high);
    const float al = a - ah;
    // Dekker product: s[2*RHS] + s[3*RHS] is the EXACT a*b.
    Muls(s[4U * RHS], b, kDekkerSplitter, n);
    Muls(s[2U * RHS], b, a, n);
    PipeBarrier<PIPE_V>();
    Sub(s[5U * RHS], b, s[4U * RHS], n);
    PipeBarrier<PIPE_V>();
    Add(s, s[4U * RHS], s[5U * RHS], n);
    PipeBarrier<PIPE_V>();
    Sub(s[RHS], b, s, n);
    Muls(s[3U * RHS], s, ah, n);
    Muls(s[5U * RHS], s, al, n);
    PipeBarrier<PIPE_V>();
    Sub(s[3U * RHS], s[3U * RHS], s[2U * RHS], n);
    Muls(s[4U * RHS], s[RHS], ah, n);
    Muls(s[RHS], s[RHS], al, n);
    PipeBarrier<PIPE_V>();
    Add(s[3U * RHS], s[3U * RHS], s[4U * RHS], n);
    PipeBarrier<PIPE_V>();
    Add(s[3U * RHS], s[3U * RHS], s[5U * RHS], n);
    PipeBarrier<PIPE_V>();
    Add(s[3U * RHS], s[3U * RHS], s[RHS], n);
    Add(s[6U * RHS], s[2U * RHS], c, n);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ScalarVectorFma(
    LocalTensor<float> dst, float a, LocalTensor<float> b, LocalTensor<float> c, uint32_t count,
    TBuf<TPosition::VECCALC>& fmaBuf, TBuf<TPosition::VECCALC>& fmaMaskBuf)
{
    const uint32_t n = (count + kVectorRoundUp) / kVectorFloats * kVectorFloats;
    auto s = fmaBuf.Get<float>();
    if ((ctrmm_ieee::BitsOf(a) & ctrmm_ieee::kFractionMask) == 0U) {
        Muls(s, b, a, n);
        PipeBarrier<PIPE_V>();
        Add(dst, s, c, n);
        PipeBarrier<PIPE_V>();
        return;
    }
    ScalarFmaProduct(s, a, b, c, n);
    AccumulateFmaResidual<RHS>(s, c, n);
    RoundVectorFma<RHS>(dst, s, fmaMaskBuf.Get<uint8_t>(), n);
}

__aicore__ inline void SplitFmaOperand(LocalTensor<float> ah, LocalTensor<float> al, LocalTensor<float> a, uint32_t n)
{
    Muls(ah, a, kDekkerSplitter, n);
    PipeBarrier<PIPE_V>();
    Sub(al, a, ah, n);
    PipeBarrier<PIPE_V>();
    Add(ah, ah, al, n);
    PipeBarrier<PIPE_V>();
    Sub(al, a, ah, n);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BatchedVectorFma(
    LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, LocalTensor<float> c, uint32_t n,
    LocalTensor<float> packed, TBuf<TPosition::VECCALC>& fmaBuf, TBuf<TPosition::VECCALC>& fmaMaskBuf)
{
    constexpr uint32_t F = 2U * RHS;
    auto s = fmaBuf.Get<float>();
    auto ah = packed;
    auto al = packed[2U * RHS];
    SplitFmaOperand(ah, al, a, n);
    // Dekker product: s[2*F] + s[3*F] is the EXACT a*b.
    Muls(s[4U * F], b, kDekkerSplitter, n);
    Mul(s[2U * F], b, a, n);
    PipeBarrier<PIPE_V>();
    Sub(s[5U * F], b, s[4U * F], n);
    PipeBarrier<PIPE_V>();
    Add(s, s[4U * F], s[5U * F], n);
    PipeBarrier<PIPE_V>();
    Sub(s[F], b, s, n);
    Mul(s[3U * F], s, ah, n);
    Mul(s[5U * F], s, al, n);
    PipeBarrier<PIPE_V>();
    Sub(s[3U * F], s[3U * F], s[2U * F], n);
    Mul(s[4U * F], s[F], ah, n);
    Mul(s[F], s[F], al, n);
    PipeBarrier<PIPE_V>();
    Add(s[3U * F], s[3U * F], s[4U * F], n);
    PipeBarrier<PIPE_V>();
    Add(s[3U * F], s[3U * F], s[5U * F], n);
    PipeBarrier<PIPE_V>();
    Add(s[3U * F], s[3U * F], s[F], n);
    Add(s[6U * F], s[2U * F], c, n);
    PipeBarrier<PIPE_V>();
    AccumulateFmaResidual<F>(s, c, n);
    RoundVectorFma<F>(dst, s, fmaMaskBuf.Get<uint8_t>(), n);
}

} // namespace
