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
#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "ctrmm_tiling_data.h"
#include "ctrmm_constants.h"
#include "ctrmm_ieee.h"
#pragma clang fp contract(off)

namespace {
using namespace AscendC;
using namespace ctrmm;
constexpr uint32_t CHUNK = 1024;
constexpr uint32_t RHS = 1024;
constexpr uint32_t RECT_ROWS = 64;
constexpr uint32_t RECT_COLS = 128;
constexpr uint32_t RECT_SIZE = RECT_ROWS * RECT_COLS;
template <HardEvent E>
__aicore__ inline void Fence()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline void Read(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t count)
{
    DataCopyPad(
        dst, src, DataCopyExtParams{1, count * kFloatBytes, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
}
__aicore__ inline void Write(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t count)
{
    DataCopyPad(dst, src, DataCopyExtParams{1, count * kFloatBytes, 0, 0, 0});
}
__aicore__ inline bool Exceptional(GM_ADDR workspace, const CtrmmTiling& t)
{
    const auto* flags = reinterpret_cast<const __gm__ uint32_t*>(workspace) + t.flagsOffset;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < t.vectorCores; ++i) {
        bad |= flags[i];
    }
    return (bad & kReferenceRequired) != 0;
}

// Near-collinear real/imaginary coefficient vectors can turn two large
// products into a tiny output component. Detect this independently of input
// magnitude. The squared correlation is a conditioning indicator, not a
// universal floating-point error certificate.
template <bool Batch = false>
__aicore__ inline bool CorrelatedComponents(
    LocalTensor<float> re, LocalTensor<float> im, LocalTensor<float> scratch, uint32_t count)
{
    if (count < kCorrelationMinElements) {
        return false;
    }
    const uint32_t n = (count + kVectorRoundUp) / kVectorFloats * kVectorFloats;
    auto work = scratch[3U * n];
    auto sums = scratch[4U * n];
    Mul(scratch, re, re, count);
    Mul(scratch[n], im, im, count);
    Mul(scratch[2U * n], re, im, count);
    PipeBarrier<PIPE_V>();
    if (Batch && count <= kVectorFloats) {
        WholeReduceSum(sums, scratch, count, kCorrelationPlanes, kBlockFloats, 1U, n / kBlockFloats);
    } else if (Batch && count % kCorrelationBatchElements == 0U) {
        const uint32_t partials = count / kVectorFloats;
        WholeReduceSum(work, scratch, kVectorFloats, kCorrelationPlanes * partials, 1U, 1U, kBlockFloats);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sums, work, partials, kCorrelationPlanes, kBlockFloats, 1U, partials / kBlockFloats);
    } else {
        ReduceSum(sums, scratch, work, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(sums[kBlockFloats], scratch[n], work, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(sums[2U * kBlockFloats], scratch[2U * n], work, count);
    }
    Fence<HardEvent::V_S>();
    const float rr = sums.GetValue(0U);
    const float ii = sums.GetValue(kBlockFloats);
    const float ri = sums.GetValue(2U * kBlockFloats);
    const float scale = rr * ii;
    return scale > 0.0f && ri * ri >= kCorrelationSquaredLimit * scale;
}

struct ComplexValue {
    float real;
    float imag;
};
__aicore__ inline bool IsZero(ComplexValue a) { return a.real == 0 && a.imag == 0; }
__aicore__ inline bool IsOne(ComplexValue a) { return a.real == 1 && a.imag == 0; }

__aicore__ inline bool IsNan(float x)
{
    return (ctrmm_ieee::BitsOf(x) & ctrmm_ieee::kMagnitudeMask) > ctrmm_ieee::kInfinityBits;
}
__aicore__ inline ComplexValue ComplexProduct(ComplexValue a, ComplexValue b, bool fuseAd = false)
{
    using ctrmm_ieee::Fma;
    // Finite identity factors do not need integer FMA emulation. Non-finite
    // operands still require the complex product (0*Inf must not be removed).
    const bool finiteA = (ctrmm_ieee::BitsOf(a.real) & ctrmm_ieee::kMagnitudeMask) < ctrmm_ieee::kInfinityBits &&
                         (ctrmm_ieee::BitsOf(a.imag) & ctrmm_ieee::kMagnitudeMask) < ctrmm_ieee::kInfinityBits;
    const bool finiteB = (ctrmm_ieee::BitsOf(b.real) & ctrmm_ieee::kMagnitudeMask) < ctrmm_ieee::kInfinityBits &&
                         (ctrmm_ieee::BitsOf(b.imag) & ctrmm_ieee::kMagnitudeMask) < ctrmm_ieee::kInfinityBits;
    if (finiteB && IsOne(a)) {
        return b;
    }
    if (finiteA && IsOne(b)) {
        return a;
    }
    const float ac = a.real * b.real;
    const float bd = a.imag * b.imag;
    const float ad = a.real * b.imag;
    const float bc = a.imag * b.real;
    // Same-type Ascend compound vector instructions are not IEEE FMA.
    // Use this single-rounding path for numerically sensitive inputs.
    ComplexValue y{Fma(a.real, b.real, -bd), fuseAd ? Fma(a.real, b.imag, bc) : Fma(a.imag, b.real, ad)};
    if (!IsNan(y.real) && !IsNan(y.imag)) {
        return y;
    }
    // The acceptance Netlib build propagates these IEEE invalid operations;
    // it does not use __mulsc3-style recovery of (NaN, NaN) to infinities.
    // In particular, (1, 0) * (Inf, Inf) must retain both 0*Inf contributions.
    return {ac - bd, ad + bc};
}

} // namespace
