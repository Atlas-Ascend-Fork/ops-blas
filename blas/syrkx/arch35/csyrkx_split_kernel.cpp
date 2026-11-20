/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/helper/devkit_version_compat.h"
#if ASC_DEVKIT_GE_9_1
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "c_api/asc_simd.h"
#include "csyrkx_kernel.h"

namespace {
using namespace AscendC;
constexpr uint32_t CHUNK = 4096;
constexpr uint32_t LANES = 64;
constexpr uint32_t STAGES = 2;
constexpr uint32_t STAGE_BYTES = 4 * CHUNK * sizeof(uint32_t);
constexpr uint32_t CONTROL_BYTES = LANES * sizeof(uint32_t);
constexpr uint32_t SAFE_BITS = 0x49800000u; // abs(x) <= 2^20, finite
__aicore__ inline uint64_t ScalarMin(uint64_t left, uint64_t right) { return left < right ? left : right; }
static_assert(STAGES * (STAGE_BYTES + CONTROL_BYTES) <= 192 * 1024, "Split UB budget exceeded");

// Deinterleave raw bits, preserving zeros/NaNs exactly. A masked maximum also
// detects nonfinite input and large finite input, including a finite four-real
// dot hiding overflow in the reference's per-term complex arithmetic.
__simd_vf__ inline void SplitAndCheck(
    __ubuf__ uint32_t* input, __ubuf__ uint32_t* output, __ubuf__ uint32_t* control, uint32_t count)
{
    namespace V = AscendC::MicroAPI;
    auto full = V::CreateMask<uint32_t, V::MaskPattern::ALL>();
    V::RegTensor<uint32_t> magnitudeMask, zero, largest;
    V::Duplicate(magnitudeMask, uint32_t(0x7fffffffu));
    V::Duplicate(zero, uint32_t(0));
    V::Duplicate(largest, uint32_t(0));
    uint32_t remaining = count;
    for (uint32_t offset = 0; offset < count; offset += LANES) {
        auto mask = V::UpdateMask<uint32_t>(remaining);
        V::RegTensor<uint32_t> real, imag, ar, ai;
        V::LoadAlign<uint32_t, V::LoadDist::DIST_DINTLV_B32>(real, imag, input + 2 * offset);
        V::StoreAlign(output + offset, real, mask);
        V::StoreAlign(output + CHUNK + offset, imag, mask);
        V::And(ar, real, magnitudeMask, mask);
        V::And(ai, imag, magnitudeMask, mask);
        V::Max(ar, ar, ai, mask);
        V::Select(ar, ar, zero, mask);
        V::Max(largest, largest, ar, full);
    }
    V::Reduce<V::ReduceType::MAX>(largest, largest, full);
    V::StoreAlign(control, largest, full);
}

__aicore__ inline uint32_t CopyChunk(
    GM_ADDR source, GM_ADDR real, GM_ADDR imag, uint64_t srcOffset, uint64_t dstOffset, uint32_t count, uint32_t stage)
{
    const uint32_t base = stage * STAGE_BYTES;
    LocalTensor<uint32_t> input(TPosition::VECCALC, base, 2 * CHUNK);
    LocalTensor<uint32_t> output(TPosition::VECCALC, base + 2 * CHUNK * sizeof(uint32_t), 2 * CHUNK);
    LocalTensor<uint32_t> control(TPosition::VECCALC, STAGES * STAGE_BYTES + stage * CONTROL_BYTES, LANES);
    GlobalTensor<uint32_t> src;
    GlobalTensor<uint32_t> outR;
    GlobalTensor<uint32_t> outI;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(source));
    outR.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(real));
    outI.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(imag));
    WaitFlag<HardEvent::MTE3_MTE2>(stage);
    DataCopyExtParams load{1, static_cast<uint32_t>(count * 2 * sizeof(uint32_t)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
    DataCopyPad(input, src[2 * srcOffset], load, pad);
    SetFlag<HardEvent::MTE2_V>(stage);
    WaitFlag<HardEvent::MTE2_V>(stage);
    asc_vf_call<SplitAndCheck>(
        reinterpret_cast<__ubuf__ uint32_t*>(input.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(output.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(control.GetPhyAddr()), count);
    SetFlag<HardEvent::V_MTE3>(stage);
    WaitFlag<HardEvent::V_MTE3>(stage);
    DataCopyExtParams store{1, static_cast<uint32_t>(count * sizeof(uint32_t)), 0, 0, 0};
    DataCopyPad(outR[dstOffset], output, store);
    DataCopyPad(outI[dstOffset], output[CHUNK], store);
    SetFlag<HardEvent::MTE3_MTE2>(stage);
    SetFlag<HardEvent::V_S>(stage);
    WaitFlag<HardEvent::V_S>(stage);
    return control.GetValue(0) > SAFE_BITS;
}

extern "C" __global__ __aicore__ void csyrkx_split_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR riskFlags, const CsyrkxTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t rows = t.trans ? t.k : t.n;
    const uint32_t cols = t.trans ? t.n : t.k;
    const uint64_t elements = static_cast<uint64_t>(rows) * cols;
    const uint64_t perBlock = (2 * elements + GetBlockNum() - 1) / GetBlockNum();
    const uint32_t chunk = static_cast<uint32_t>(ScalarMin(CHUNK, (perBlock + LANES - 1) / LANES * LANES));
    const uint64_t packedChunks = (elements + chunk - 1) / chunk;
    for (uint32_t stage = 0; stage < STAGES; ++stage)
        SetFlag<HardEvent::MTE3_MTE2>(stage);
    uint32_t risk = 0;
    uint32_t stage = 0;
    // Assign aligned contiguous output chunks to blocks. For padded input,
    // one block may need several row fragments to fill its chunk, but no
    // other block can write the same 32-byte destination block.
    for (uint64_t task = GetBlockIdx(); task < 2 * packedChunks; task += GetBlockNum()) {
        const bool isB = task >= packedChunks;
        const uint64_t local = isB ? task - packedChunks : task;
        const uint32_t ld = isB ? t.ldb : t.lda;
        uint64_t outputOffset = local * chunk;
        uint32_t remaining = static_cast<uint32_t>(ScalarMin(chunk, elements - outputOffset));
        if (ld == rows) {
            risk |= CopyChunk(isB ? b : a, isB ? br : ar, isB ? bi : ai, outputOffset, outputOffset, remaining, stage);
            stage ^= 1;
        } else {
            while (remaining != 0) {
                const uint64_t col = outputOffset / rows;
                const uint64_t row = outputOffset % rows;
                const uint32_t count = static_cast<uint32_t>(ScalarMin(remaining, rows - row));
                risk |=
                    CopyChunk(isB ? b : a, isB ? br : ar, isB ? bi : ai, col * ld + row, outputOffset, count, stage);
                outputOffset += count;
                remaining -= count;
                stage ^= 1;
            }
        }
    }
    for (uint32_t s = 0; s < STAGES; ++s)
        WaitFlag<HardEvent::MTE3_MTE2>(s);
    // Every launched block publishes its flag, including blocks with no input
    // chunks. Flags have disjoint ownership; no atomics or memset launch.
    LocalTensor<uint32_t> control(TPosition::VECCALC, STAGES * STAGE_BYTES, LANES);
    control.SetValue(0, risk);
    for (uint32_t i = 1; i < CSYRKX_RISK_SLOT_WORDS; ++i)
        control.SetValue(i, 0);
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    GlobalTensor<uint32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(riskFlags));
    DataCopyExtParams storeFlag{1, CSYRKX_RISK_SLOT_WORDS * sizeof(uint32_t), 0, 0, 0};
    DataCopyPad(flags[GetBlockIdx() * CSYRKX_RISK_SLOT_WORDS], control, storeFlag);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
}
} // namespace

void csyrkx_split_do(
    uint32_t blocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* ar, uint8_t* ai, uint8_t* br,
    uint8_t* bi, uint8_t* riskFlags, const CsyrkxTilingData& t)
{
    csyrkx_split_kernel<<<blocks, nullptr, stream>>>(
        const_cast<uint8_t*>(a), const_cast<uint8_t*>(b), ar, ai, br, bi, riskFlags, t);
}
#endif
