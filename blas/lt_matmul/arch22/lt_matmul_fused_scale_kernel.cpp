/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lt_matmul_kernel.h"
using namespace AscendC;
namespace {
template <HardEvent E>
__aicore__ inline void SyncScale()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}
__aicore__ inline bool IsNanBits(uint32_t bits) { return (bits & 0x7fffffffU) > 0x7f800000U; }
__aicore__ inline bool IsSignalingNan(uint32_t bits) { return IsNanBits(bits) && (bits & 0x00400000U) == 0U; }
} // namespace

// H10-only alpha0, nontrivial-beta Scale. Finite arithmetic remains FP32.
// The task's EXACT check also compares NaN payloads against its ARM host.
extern "C" __global__ __aicore__ __vector__ void lt_matmul22_fused_scale_exact(GM_ADDR c, GM_ADDR d, LtMatmul22Tiling t)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, outputBuffer;
    pipe.InitBuffer(inputBuffer, 2048U);
    pipe.InitBuffer(outputBuffer, 2048U);
    auto input = inputBuffer.Get<uint32_t>();
    auto output = outputBuffer.Get<uint32_t>();
    GlobalTensor<uint32_t> cg, dg;
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(c));
    dg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(d));
    const uint32_t beta = *reinterpret_cast<const uint32_t*>(&t.beta);
    for (uint32_t row = GetBlockIdx(); row < t.m; row += GetBlockNum()) {
        for (uint32_t col = 0U; col < t.n; col += 512U) {
            const uint32_t count = t.n - col < 512U ? t.n - col : 512U;
            const DataCopyExtParams copy{1U, count * 4U, 0U, 0U, 0U};
            const DataCopyPadExtParams<uint32_t> pad{false, 0U, 0U, 0U};
            DataCopyPad(input, cg[uint64_t(row) * t.ldc + col], copy, pad);
            SyncScale<HardEvent::MTE2_V>();
            Muls(output.ReinterpretCast<float>(), input.ReinterpretCast<float>(), t.beta, count);
            SyncScale<HardEvent::V_S>();
            SyncScale<HardEvent::MTE2_S>();
            for (uint32_t i = 0U; i < count; ++i) {
                const uint32_t value = input.GetValue(i);
                if (IsSignalingNan(beta))
                    output.SetValue(i, beta | 0x00400000U);
                else if (IsSignalingNan(value))
                    output.SetValue(i, value | 0x00400000U);
                else if (IsNanBits(beta))
                    output.SetValue(i, beta | 0x00400000U);
                else if (IsNanBits(value))
                    output.SetValue(i, value | 0x00400000U);
                else if ((beta & 0x7fffffffU) == 0x7f800000U && (value & 0x7fffffffU) == 0U)
                    output.SetValue(i, 0x7fc00000U);
            }
            SyncScale<HardEvent::S_MTE3>();
            DataCopyPad(dg[uint64_t(row) * t.ldd + col], output, copy);
            SyncScale<HardEvent::MTE3_V>();
            SyncScale<HardEvent::MTE3_MTE2>();
        }
    }
}

void LtMatmul22FusedScale(uint32_t blocks, void* stream, uint8_t* c, uint8_t* d, LtMatmul22Tiling t)
{
    lt_matmul22_fused_scale_exact<<<blocks, nullptr, stream>>>(c, d, t);
}
