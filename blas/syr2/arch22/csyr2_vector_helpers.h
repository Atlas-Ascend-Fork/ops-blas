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

#ifndef CSYR2_VECTOR_HELPERS_H
#define CSYR2_VECTOR_HELPERS_H

#include <cstdint>
#include "kernel_operator.h"

namespace {
constexpr int32_t CSYR2_VEC_FLOATS = 64; /* 一次 repeat 处理的 float 数 */

template <int32_t Alignment>
__aicore__ inline int32_t Csyr2CeilAlign(int32_t v)
{
    static_assert(Alignment > 0, "CSYR2 alignment must be positive");
    return ((v + Alignment - 1) / Alignment) * Alignment;
}

__aicore__ inline void Csyr2CountMaskBegin(uint32_t count)
{
    AscendC::SetMaskCount();
    AscendC::AscendCUtils::SetMask<float>(0, static_cast<int32_t>(count));
}

__aicore__ inline void Csyr2CountMaskEnd()
{
    AscendC::ResetMask();
    AscendC::SetMaskNorm();
}

__aicore__ inline void Csyr2MulsCount(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float scalar)
{
    AscendC::Muls<float, false>(
        dst, src, scalar, static_cast<uint64_t>(0), static_cast<uint8_t>(1), AscendC::UnaryRepeatParams(1, 1, 8, 8));
}

__aicore__ inline void Csyr2AxpyCount(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float scalar)
{
    AscendC::PipeBarrier<PIPE_V>();
    vaxpy(
        reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
        scalar, 1, 1, 1, 8, 8);
}
} // namespace

// Padded local buffers permit a full final vector repeat. GM copies retain their exact lengths.
__aicore__ inline __attribute__((always_inline)) void Csyr2PaddedMuls(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float scalar, uint32_t count)
{
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    AscendC::PipeBarrier<PIPE_V>();
    vmuls(
        reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
        scalar, static_cast<uint16_t>((count + 63) / 64), 1, 1, 8, 8);
}
__aicore__ inline __attribute__((always_inline)) void Csyr2PaddedAxpy(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float scalar, uint32_t count)
{
    AscendC::PipeBarrier<PIPE_V>();
    vaxpy(
        reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
        scalar, static_cast<uint16_t>((count + 63) / 64), 1, 1, 8, 8);
}

// Padded local buffers permit a full final vector repeat. GM copies retain their exact lengths.
__aicore__ inline __attribute__((always_inline)) void Csyr2LayerMuls(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float scalar, uint32_t count)
{
    vmuls(
        reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
        scalar, static_cast<uint16_t>((count + 63) / 64), 1, 1, 8, 8);
}
__aicore__ inline __attribute__((always_inline)) void Csyr2LayerAxpy(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float scalar, uint32_t count)
{
    vaxpy(
        reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
        scalar, static_cast<uint16_t>((count + 63) / 64), 1, 1, 8, 8);
}

__aicore__ inline __attribute__((always_inline)) void Csyr2NativeMulsCount(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float value)
{
    vmuls(
        reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()), value,
        1, 1, 1, 8, 8);
}
__aicore__ inline __attribute__((always_inline)) void Csyr2NativeAxpyCount(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float value)
{
    vaxpy(
        reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()), value,
        1, 1, 1, 8, 8);
}

#endif
