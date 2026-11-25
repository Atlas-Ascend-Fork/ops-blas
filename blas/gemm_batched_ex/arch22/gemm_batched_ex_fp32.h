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
#include "gemm_batched_ex_tiling.h"

namespace GemmBatchedExFp32 {
// IEEE fp32 fused multiply-add, rounded once to nearest-even. The A2 vector
// multiply-add rounds the product separately, unlike the reference BLAS.
__aicore__ inline uint64_t ShiftJam(uint64_t value, uint32_t shift)
{
    return shift < 64 ? (value >> shift) | ((value << (64 - shift)) != 0) : value != 0;
}

union FloatWord {
    float value;
    uint32_t bits;
    __aicore__ inline uint32_t Exponent() const { return (bits >> 23) & 255; }
    __aicore__ inline uint32_t Mantissa() const { return bits & 0x7fffff; }
};

__aicore__ inline float FusedSpecial(FloatWord x, FloatWord y, FloatWord z)
{
    const uint32_t ae = x.Exponent();
    const uint32_t be = y.Exponent();
    const uint32_t ce = z.Exponent();
    const uint32_t ps = (x.bits ^ y.bits) >> 31;
    const bool invalid = (ae == 255 && x.Mantissa()) || (be == 255 && y.Mantissa()) || (ce == 255 && z.Mantissa()) ||
                         (ae == 255 && be == 0 && y.Mantissa() == 0) || (be == 255 && ae == 0 && x.Mantissa() == 0) ||
                         ((ae == 255 || be == 255) && ce == 255 && ps != (z.bits >> 31));
    FloatWord out{};
    out.bits = invalid ? 0x7fc00000U : ((ae == 255 || be == 255) ? (ps << 31) | 0x7f800000U : z.bits);
    return out.value;
}

__aicore__ inline int32_t AlignFusedProduct(
    uint64_t product, uint32_t cm, uint32_t ae, uint32_t be, uint32_t ce, uint64_t& p, uint64_t& q)
{
    int32_t pe = static_cast<int32_t>(ae ? ae : 1) + static_cast<int32_t>(be ? be : 1) - 300;
    int32_t ze = static_cast<int32_t>(ce ? ce : 1) - 150;
    int32_t pt = product ? pe + 63 - __builtin_clzll(product) : -512;
    int32_t zt = cm ? ze + 63 - __builtin_clzll(static_cast<uint64_t>(cm)) : -512;
    int32_t common = (pt > zt ? pt : zt) - 60;
    p = pe >= common ? product << (pe - common) : ShiftJam(product, common - pe);
    q = cm == 0 ? 0 : (ze >= common ? static_cast<uint64_t>(cm) << (ze - common) : ShiftJam(cm, common - ze));
    return common;
}

__aicore__ inline float RoundFusedFloat(uint64_t magnitude, int32_t common, uint32_t sign)
{
    FloatWord out{};
    if (magnitude == 0) {
        out.bits = 0;
        return out.value;
    }
    int32_t leading = 63 - __builtin_clzll(magnitude);
    int32_t exponent = common + leading;
    int32_t shift = exponent < -126 ? -149 - common : leading - 23;
    uint64_t rounded;
    if (shift <= 0)
        rounded = magnitude << -shift;
    else if (shift < 64) {
        rounded = magnitude >> shift;
        uint64_t rest = magnitude & ((1ULL << shift) - 1);
        uint64_t half = 1ULL << (shift - 1);
        rounded += rest > half || (rest == half && (rounded & 1));
    } else
        rounded = shift == 64 && magnitude > (1ULL << 63);
    if (exponent < -126)
        out.bits = (sign << 31) | static_cast<uint32_t>(rounded);
    else {
        if (rounded == 0x1000000) {
            rounded >>= 1;
            ++exponent;
        }
        out.bits = exponent > 127 ? (sign << 31) | 0x7f800000U :
                                    (sign << 31) | (static_cast<uint32_t>(exponent + 127) << 23) |
                                        (static_cast<uint32_t>(rounded) & 0x7fffff);
    }
    return out.value;
}

__aicore__ inline float FusedFloat(float a, float b, float c)
{
    FloatWord x{a};
    FloatWord y{b};
    FloatWord z{c};
    FloatWord out{};
    uint32_t ae = (x.bits >> 23) & 255;
    uint32_t be = (y.bits >> 23) & 255;
    uint32_t ce = (z.bits >> 23) & 255;
    uint32_t am = x.bits & 0x7fffff;
    uint32_t bm = y.bits & 0x7fffff;
    uint32_t cm = z.bits & 0x7fffff;
    uint32_t ps = (x.bits ^ y.bits) >> 31;
    uint32_t cs = z.bits >> 31;
    if (ae == 255 || be == 255 || ce == 255)
        return FusedSpecial(x, y, z);
    am |= ae ? 0x800000 : 0;
    bm |= be ? 0x800000 : 0;
    cm |= ce ? 0x800000 : 0;
    uint64_t product = static_cast<uint64_t>(am) * bm;
    if (product == 0 && cm == 0) {
        out.bits = ps == cs ? ps << 31 : 0;
        return out.value;
    }
    if (product == 0)
        return c;
    uint64_t p;
    uint64_t q;
    const int32_t common = AlignFusedProduct(product, cm, ae, be, ce, p, q);
    uint64_t magnitude;
    uint32_t sign;
    if (ps == cs) {
        magnitude = p + q;
        sign = ps;
    } else if (p >= q) {
        magnitude = p - q;
        sign = ps;
    } else {
        magnitude = q - p;
        sign = cs;
    }
    return RoundFusedFloat(magnitude, common, sign);
}

// A temporary ID is consumed by its matching wait before the next helper call.
// FetchEventID respects the current TPipe's reservations, including Matmul IDs.
template <AscendC::HardEvent Event>
__aicore__ inline void SyncEvent()
{
    const auto event = GetTPipePtr()->FetchEventID(Event);
    AscendC::SetFlag<Event>(event);
    AscendC::WaitFlag<Event>(event);
}

constexpr uint32_t REFERENCE_CAPACITY = 2048;

// Compensated product/sum transforms retain low bits that an A2 vector
// multiply-add rounds away, reducing error relative to the FP32 FMA reference.
__aicore__ inline void ProductResidual(
    const AscendC::LocalTensor<float>& input, float coefficient, const AscendC::LocalTensor<float>& work,
    uint32_t count)
{
    const auto high = work[REFERENCE_CAPACITY];
    const auto low = work[2 * REFERENCE_CAPACITY];
    const auto error = work[3 * REFERENCE_CAPACITY];
    const auto temporary = work[4 * REFERENCE_CAPACITY];
    volatile float split = coefficient * 4097.0f;
    volatile float upper = split - (split - coefficient);
    const float lower = coefficient - upper;
    AscendC::Muls(work, input, coefficient, count);
    AscendC::Muls(temporary, input, 4097.0f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(high, temporary, input, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(high, temporary, high, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(low, input, high, count);
    AscendC::Muls(error, high, static_cast<float>(upper), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(error, error, work, count);
    AscendC::Muls(temporary, high, lower, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(error, error, temporary, count);
    AscendC::Muls(temporary, low, static_cast<float>(upper), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(error, error, temporary, count);
    AscendC::Muls(temporary, low, lower, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(error, error, temporary, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AddProductResidual(
    const AscendC::LocalTensor<float>& sum, const AscendC::LocalTensor<float>& work, uint32_t count)
{
    const auto result = work[REFERENCE_CAPACITY];
    const auto part = work[2 * REFERENCE_CAPACITY];
    const auto error = work[3 * REFERENCE_CAPACITY];
    const auto temporary = work[4 * REFERENCE_CAPACITY];
    AscendC::Add(result, sum, work, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(part, result, work, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(temporary, result, part, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(temporary, work, temporary, count);
    AscendC::Sub(part, sum, part, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(temporary, temporary, part, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(error, error, temporary, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(sum, result, error, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReferenceFma(
    const AscendC::LocalTensor<float>& sum, const AscendC::LocalTensor<float>& input, float coefficient,
    const AscendC::LocalTensor<float>& work, uint32_t count)
{
    const auto absolute = work;
    const auto previous = work[REFERENCE_CAPACITY];
    const auto reduction = work[5 * REFERENCE_CAPACITY];
    AscendC::Abs(absolute, input, count);
    AscendC::Abs(previous, sum, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(absolute, absolute, previous, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum(reduction, absolute, reduction[32], count);
    SyncEvent<AscendC::HardEvent::V_S>();
    const float magnitude = reduction.GetValue(0);
    if (magnitude < 0x1p60f && coefficient < 0x1p60f && coefficient > -0x1p60f) {
        SyncEvent<AscendC::HardEvent::S_V>();
        ProductResidual(input, coefficient, work, count);
        AddProductResidual(sum, work, count);
    } else {
        for (uint32_t lane = 0; lane < count; ++lane)
            sum.SetValue(lane, FusedFloat(input.GetValue(lane), coefficient, sum.GetValue(lane)));
        SyncEvent<AscendC::HardEvent::S_V>();
    }
}

struct ReferenceScalars {
    float alpha;
    float beta;
};

// Preserve sequential K accumulation for sign-concentrated inputs. Cube
// reductions reorder additions, so their result can exceed the FP32 ULP bound.
__aicore__ inline void AccumulateReference(
    const AscendC::GlobalTensor<float>& ag, const AscendC::GlobalTensor<float>& bg, uint64_t row, uint64_t col,
    uint32_t rows, uint32_t count, const GemmBatchedExArch22Tiling& t, const ReferenceScalars& scalars,
    const AscendC::LocalTensor<float>& sum, const AscendC::LocalTensor<float>& input,
    const AscendC::LocalTensor<float>& work)
{
    const AscendC::DataCopyExtParams packed{1, rows * 4U, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<float> padding{true, 0, static_cast<uint8_t>(t.transA ? 7 : 0), 0};
    for (int32_t p = 0; p < t.k; ++p) {
        const float bv = bg.GetValue(t.transB ? static_cast<uint64_t>(p) * t.ldb + col : col * t.ldb + p);

        SyncEvent<AscendC::HardEvent::V_MTE2>();
        if (t.transA) {
            const AscendC::DataCopyExtParams strided{
                static_cast<uint16_t>(rows), 4U, static_cast<uint32_t>((static_cast<uint64_t>(t.lda) - 1) * 4U), 0, 0};
            AscendC::DataCopyPad(input, ag[row * t.lda + p], strided, padding);
        } else {
            AscendC::DataCopyPad(input, ag[static_cast<uint64_t>(p) * t.lda + row], packed, padding);
        }
        SyncEvent<AscendC::HardEvent::MTE2_V>();
        const float multiplier = t.transA ? bv : scalars.alpha * bv;
        ReferenceFma(sum, input, multiplier, work, count);
    }
}

__aicore__ inline void StoreReference(
    const AscendC::GlobalTensor<float>& cg, uint64_t offset, uint32_t rows, uint32_t count,
    const GemmBatchedExArch22Tiling& t, const ReferenceScalars& scalars, const AscendC::LocalTensor<float>& sum,
    const AscendC::LocalTensor<float>& old, const AscendC::LocalTensor<float>& work)
{
    const AscendC::DataCopyExtParams packed{1, rows * 4U, 0, 0, 0};
    const AscendC::DataCopyExtParams lanes{static_cast<uint16_t>(rows), 4U, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<float> padding{true, 0, static_cast<uint8_t>(t.transA ? 7 : 0), 0};
    if (t.transA) {
        if (scalars.alpha != 0.0f) {
            AscendC::Muls(sum, sum, scalars.alpha, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (scalars.beta != 0.0f) {
            AscendC::DataCopyPad(old, cg[offset], lanes, padding);
            SyncEvent<AscendC::HardEvent::MTE2_V>();
            if (scalars.alpha == 0.0f) {
                AscendC::Muls(sum, old, scalars.beta, count);
            } else {
                ReferenceFma(sum, old, scalars.beta, work, count);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        SyncEvent<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(cg[offset], sum, lanes);
    } else {
        SyncEvent<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(cg[offset], sum, packed);
    }
    SyncEvent<AscendC::HardEvent::MTE3_S>();
}
} // namespace GemmBatchedExFp32
