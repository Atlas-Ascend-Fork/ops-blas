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
__aicore__ inline void InitRepackIndices(
    LocalTensor<float> out, LocalTensor<uint32_t> idx, LocalTensor<uint32_t> temporary, LocalTensor<uint8_t> mask)
{
    constexpr uint32_t L = kRepackElements;
    CreateVecIndex(idx.ReinterpretCast<int32_t>(), int32_t{0}, kExpandedComponents * L);
    Duplicate(temporary.ReinterpretCast<int32_t>(), kRepackSlotMask, kExpandedComponents * L);
    PipeBarrier<PIPE_V>();
    And(temporary.ReinterpretCast<uint16_t>(), idx.ReinterpretCast<uint16_t>(), temporary.ReinterpretCast<uint16_t>(),
        16U * L);
    PipeBarrier<PIPE_V>();
    Cast(out, temporary.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, kExpandedComponents * L);
    ShiftRight(idx, idx, kRepackSlotShift, kExpandedComponents * L);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, out, 2.0f, CMPMODE::LT, kExpandedComponents * L);
    ShiftLeft(temporary, temporary, kRepackPlaneShift, kExpandedComponents * L);
    PipeBarrier<PIPE_V>();
    Add(idx.ReinterpretCast<int32_t>(), idx.ReinterpretCast<int32_t>(), temporary.ReinterpretCast<int32_t>(),
        kExpandedComponents * L);
    PipeBarrier<PIPE_V>();
    Cast(out, idx.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, kExpandedComponents * L);
    PipeBarrier<PIPE_V>();
    Select(out, mask, out, kRepackZeroIndex, SELMODE::VSEL_TENSOR_SCALAR_MODE, kExpandedComponents * L);
    PipeBarrier<PIPE_V>();
    Muls(out, out, kGatherFloatByteScale, kExpandedComponents * L);
    PipeBarrier<PIPE_V>();
    Cast(idx.ReinterpretCast<int32_t>(), out, RoundMode::CAST_RINT, kExpandedComponents * L);
    PipeBarrier<PIPE_V>();
    Fence<HardEvent::S_V>();
}

__aicore__ inline void RepackTiles(
    GM_ADDR ws, const CtrmmTiling& t, LocalTensor<float> in, LocalTensor<float> out, LocalTensor<uint32_t> idx)
{
    constexpr uint32_t L = kRepackElements;
    GlobalTensor<float> w;
    w.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
    const bool descending =
        !t.upper && ((!t.right && t.originalTrans == ACLBLAS_OP_N) || (t.right && t.originalTrans != ACLBLAS_OP_N));
    const bool separateDiagonal = !t.upper && !descending;
    const uint64_t base = kOriginalAPlanes * t.aPlane + 2U * t.xPlane;
    for (uint64_t off = GetBlockIdx() * L; off < t.aPlane; off += GetBlockNum() * L) {
        const uint32_t n = Min(L, static_cast<uint32_t>(t.aPlane - off));
        Read(in, w[off], n);
        Read(in[L], w[t.aPlane + off], n);
        Fence<HardEvent::MTE2_V>();
        Gather(out, in, idx, 0U, kExpandedComponents * n);
        Fence<HardEvent::V_S>();
        for (uint64_t d = (off + t.qp) / (t.qp + 1U) * (t.qp + 1U); separateDiagonal && d < off + n; d += t.qp + 1U) {
            out.SetValue((d - off) * kExpandedComponents, 0.0f);
            out.SetValue((d - off) * kExpandedComponents + 1U, 0.0f);
        }
        Fence<HardEvent::S_MTE3>();
        Write(w[base + kExpandedComponents * off], out, kExpandedComponents * n);
        Fence<HardEvent::MTE3_MTE2>();
    }
    Fence<HardEvent::MTE3_V>();
    const uint64_t diag = kCubeAPlanes * t.aPlane + kCubeXPlanes * t.xPlane;
    for (uint32_t row = GetBlockIdx(); separateDiagonal && row < t.qp; row += GetBlockNum()) {
        Duplicate(out, 0.0f, kDiagonalRowFloats);
        Fence<HardEvent::V_MTE2>();
        const uint64_t off = static_cast<uint64_t>(row) * t.qp + row;
        Read(in, w[off], 1U);
        Read(in[L], w[t.aPlane + off], 1U);
        Fence<HardEvent::MTE2_S>();
        out.SetValue((row % kCubeTile) * 2U, in.GetValue(0));
        out.SetValue((row % kCubeTile) * 2U + 1U, in.GetValue(L));
        Fence<HardEvent::S_MTE3>();
        Write(w[diag + static_cast<uint64_t>(row) * kDiagonalRowFloats], out, kDiagonalRowFloats);
        Fence<HardEvent::MTE3_V>();
    }
}

} // namespace
