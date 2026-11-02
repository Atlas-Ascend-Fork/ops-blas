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
__aicore__ inline bool AllMaskBits(LocalTensor<uint8_t> mask, uint32_t count, TBuf<TPosition::VECCALC>& absBuf)
{
    if (count >= CHUNK && count % kBitsPerByte == 0U) {
        auto values = absBuf.Get<half>();
        Cast(values, mask, RoundMode::CAST_NONE, count / kBitsPerByte);
        PipeBarrier<PIPE_V>();
        ReduceMin(values[kMaskMinResultOffset], values, values[kMaskMinScratchOffset], count / kBitsPerByte, false);
        Fence<HardEvent::V_S>();
        return static_cast<float>(values.GetValue(kMaskMinResultOffset)) == kFullByteMaskValue;
    }
    Fence<HardEvent::V_S>();
    auto bits = mask.ReinterpretCast<uint32_t>();
    for (uint32_t i = 0; i < (count + kBitsPerWord - 1U) / kBitsPerWord; ++i) {
        const uint32_t active = (i + 1U) * kBitsPerWord <= count ? kFullWordMask : (1U << (count % kBitsPerWord)) - 1U;
        if ((bits.GetValue(i) & active) != active) {
            return false;
        }
    }
    return true;
}

__aicore__ inline uint32_t CheckInputFinite(
    LocalTensor<float> x, uint32_t count, LocalTensor<float> raw, TBuf<TPosition::VECCALC>& maskBuf, float limit)
{
    auto abs = raw;
    auto mask = maskBuf.Get<uint8_t>();
    const uint32_t aligned = (count + kVectorRoundUp) / kVectorFloats * kVectorFloats;
    if (aligned != count) {
        Duplicate(abs, 0.0f, aligned);
        PipeBarrier<PIPE_V>();
    }
    Abs(abs, x, count);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, abs, limit, CMPMODE::LE, aligned);
    CompareScalar(mask[kFmaMaxMaskOffset], abs, kVectorFmaMaxInput, CMPMODE::LE, aligned);
    CompareScalar(mask[kFmaMinMaskOffset], abs, kVectorFmaMinInput, CMPMODE::GE, aligned);
    CompareScalar(mask[kZeroMaskOffset], abs, 0.0f, CMPMODE::EQ, aligned);
    PipeBarrier<PIPE_V>();
    Or(mask[kFmaMinMaskOffset].ReinterpretCast<uint16_t>(), mask[kFmaMinMaskOffset].ReinterpretCast<uint16_t>(),
       mask[kZeroMaskOffset].ReinterpretCast<uint16_t>(), aligned / kBitsPerHalfWord);
    And(mask[kFastMaskOffset].ReinterpretCast<uint16_t>(), mask[kFastMaskOffset].ReinterpretCast<uint16_t>(),
        mask.ReinterpretCast<uint16_t>(), aligned / kBitsPerHalfWord);
    PipeBarrier<PIPE_V>();
    And(mask[kFmaMaxMaskOffset].ReinterpretCast<uint16_t>(), mask[kFmaMaxMaskOffset].ReinterpretCast<uint16_t>(),
        mask[kFmaMinMaskOffset].ReinterpretCast<uint16_t>(), aligned / kBitsPerHalfWord);
    PipeBarrier<PIPE_V>();
    And(mask[kFmaMaskOffset].ReinterpretCast<uint16_t>(), mask[kFmaMaskOffset].ReinterpretCast<uint16_t>(),
        mask[kFmaMaxMaskOffset].ReinterpretCast<uint16_t>(), aligned / kBitsPerHalfWord);
    PipeBarrier<PIPE_V>();
    return 0U;
}

} // namespace
