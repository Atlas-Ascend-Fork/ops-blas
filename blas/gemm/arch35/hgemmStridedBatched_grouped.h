/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

// Compact small batches share GM->L1 transfer commands. Each batch still has
// an independent Mmad and C region; no block-diagonal zero multiplications.
namespace hgemm_grouped {
constexpr uint32_t GROUP = 4;

template <uint32_t D>
__aicore__ inline void Copy(
    const AscendC::GlobalTensor<half>& a, const AscendC::GlobalTensor<half>& b, uint32_t first, uint32_t count,
    uint64_t slot)
{
    constexpr uint32_t P = (D + HGEMM_CUBE_C0_ELEMENTS - 1) / HGEMM_CUBE_C0_ELEMENTS * HGEMM_CUBE_C0_ELEMENTS;
    constexpr uint64_t INPUT_BYTES = GROUP * P * P * sizeof(half);
    const uint64_t offset = slot * HGEMM_DOUBLE_BUFFER_SLOTS * INPUT_BYTES;
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
    AscendC::LocalTensor<half> a1(AscendC::TPosition::A1, offset, INPUT_BYTES / sizeof(half));
    AscendC::LocalTensor<half> b1(AscendC::TPosition::B1, offset + INPUT_BYTES, INPUT_BYTES / sizeof(half));
    AscendC::Nd2NzParams params{};
    params.ndNum = count;
    params.nValue = D;
    params.dValue = D;
    // Both matrix strides are elements, per the 9.1 GM->L1 ND2NZ API.
    params.srcNdMatrixStride = D * D;
    params.srcDValue = D;
    params.dstNzC0Stride = P;
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = P * P;
    const uint64_t base = static_cast<uint64_t>(first) * D * D;
    AscendC::DataCopy(a1, a[base], params);
    AscendC::DataCopy(b1, b[base], params);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
}

template <uint32_t P>
__aicore__ inline void Write(
    GM_ADDR c, const GemmStridedCubeState& st, uint32_t batch, uint64_t cSlot, uint64_t cOffset)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(cSlot);
    AscendC::GlobalTensor<half> gc;
    const uint32_t d = st.tiling.m;
    gc.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c) + static_cast<uint64_t>(batch) * d * d);
    WriteStridedFixpipe<half, float, P, P, QuantMode_t::F322F16, false>(st, gc, 0, 1, 0, 1, cOffset);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cSlot);
}

template <uint32_t D, bool TA, bool TB>
struct GroupedKernel {
    static constexpr uint32_t P = (D + HGEMM_CUBE_C0_ELEMENTS - 1) / HGEMM_CUBE_C0_ELEMENTS * HGEMM_CUBE_C0_ELEMENTS,
                              C_SLOTS = 8, FIX_DELAY = 2;
    static constexpr uint64_t INPUT_BYTES = GROUP * P * P * sizeof(half), C_BYTES = 16 * 1024;
    static_assert(P * P * sizeof(float) <= C_BYTES, "C tile must fit its separated slot");
    static_assert(FIX_DELAY < C_SLOTS, "Delayed Fixpipe must precede C-slot reuse");
    static_assert((C_SLOTS & (C_SLOTS - 1)) == 0, "C-slot ring must be a power of two");
    static_assert(
        HGEMM_DOUBLE_BUFFER_SLOTS * HGEMM_DOUBLE_BUFFER_SLOTS * INPUT_BYTES <= HGEMM_L1_CAPACITY_BYTES,
        "Grouped L1 capacity");
    static_assert(P * P * sizeof(half) <= HGEMM_L0_AB_HALF_BYTES, "Grouped L0A/B slot capacity");
    static_assert(C_SLOTS * C_BYTES <= HGEMM_L0_C_CAPACITY_BYTES, "Grouped L0C capacity");
    GM_ADDR a;
    GM_ADDR b;
    GM_ADDR c;
    GemmStridedBatchedExTilingData t;
    AscendC::GlobalTensor<half> ga, gb;
    GemmStridedCubeState st{};
    uint64_t matrix = 0;

    __aicore__ inline void Initialize()
    {
        ga.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a));
        gb.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b));
        st.tiling = t;
        st.baseMCount = D / P;
        st.tailM = D % P;
        st.tailMAlign = P;
        st.baseNCount = D / P;
        st.tailN = D % P;
        st.tailNAlign = P;
        st.nBlockSize = 1;
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        for (uint32_t i = 0; i < C_SLOTS; ++i)
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(i);
    }

    __aicore__ inline void Multiply(uint64_t offset, uint32_t i)
    {
        const uint64_t l0 = matrix & (HGEMM_DOUBLE_BUFFER_SLOTS - 1), l0Offset = l0 * HGEMM_L0_AB_HALF_BYTES;
        const uint64_t cSlot = matrix & (C_SLOTS - 1), cOffset = cSlot * C_BYTES;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cSlot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0);
        AscendC::LocalTensor<half> a1(AscendC::TPosition::A1, offset + i * P * P * sizeof(half), P * P);
        AscendC::LocalTensor<half> b1(AscendC::TPosition::B1, offset + INPUT_BYTES + i * P * P * sizeof(half), P * P);
        AscendC::LocalTensor<half> a2(AscendC::TPosition::A2, l0Offset, P * P);
        AscendC::LocalTensor<half> b2(AscendC::TPosition::B2, l0Offset, P * P);
        AscendC::LoadData2DParamsV2 pa{};
        pa.mStep = P / HGEMM_CUBE_C0_ELEMENTS;
        pa.kStep = P / HGEMM_CUBE_C0_ELEMENTS;
        pa.srcStride = P / HGEMM_CUBE_C0_ELEMENTS;
        pa.dstStride = P / HGEMM_CUBE_C0_ELEMENTS;
        pa.ifTranspose = TA;
        auto pb = pa;
        pb.ifTranspose = !TB;
        AscendC::LoadData(a2, a1, pa);
        AscendC::LoadData(b2, b1, pb);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0);
        AscendC::LocalTensor<float> c0(AscendC::TPosition::CO1, cOffset, P * P);
        AscendC::MmadParams mp{};
        mp.m = D;
        mp.n = D;
        mp.k = D;
        mp.cmatrixInitVal = true;
        mp.disableGemv = true;
        AscendC::Mmad(c0, a2, b2, mp);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cSlot);
    }

    __aicore__ inline void ProcessGroup(uint32_t first, uint32_t end, uint64_t group)
    {
        const uint64_t slot = group & (HGEMM_DOUBLE_BUFFER_SLOTS - 1),
                       offset = slot * HGEMM_DOUBLE_BUFFER_SLOTS * INPUT_BYTES;
        const uint32_t count = end - first < GROUP ? end - first : GROUP;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        if (first + GROUP < end) {
            const uint32_t rest = end - first - GROUP;
            Copy<D>(ga, gb, first + GROUP, rest < GROUP ? rest : GROUP, slot ^ 1);
        }
        for (uint32_t i = 0; i < count; ++i, ++matrix) {
            Multiply(offset, i);
            // Keep two independent Mmad results pending, so MTE1 can prepare
            // the next operands before waiting for a prior Cube->Fixpipe event.
            if (matrix >= FIX_DELAY) {
                const uint64_t done = (matrix - FIX_DELAY) & (C_SLOTS - 1);
                Write<P>(c, st, first + i - FIX_DELAY, done, done * C_BYTES);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
    }

    __aicore__ inline void Finish(uint32_t begin, uint32_t end)
    {
        const uint32_t pending = end - begin < FIX_DELAY ? end - begin : FIX_DELAY;
        for (uint32_t batch = end - pending; batch < end; ++batch) {
            const uint64_t done = (batch - begin) & (C_SLOTS - 1);
            Write<P>(c, st, batch, done, done * C_BYTES);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        for (uint32_t i = 0; i < C_SLOTS; ++i)
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(i);
    }

    __aicore__ inline void Run()
    {
        const uint32_t block = AscendC::GetBlockIdx(), blocks = AscendC::GetBlockNum();
        if (blocks == 0)
            return;
        const uint32_t begin = static_cast<uint64_t>(t.batchCount) * block / blocks;
        const uint32_t end = static_cast<uint64_t>(t.batchCount) * (block + 1) / blocks;
        Initialize();
        if (begin < end)
            Copy<D>(ga, gb, begin, end - begin < GROUP ? end - begin : GROUP, 0);
        uint64_t group = 0;
        for (uint32_t first = begin; first < end; first += GROUP, ++group)
            ProcessGroup(first, end, group);
        Finish(begin, end);
    }
};

template <uint32_t D, bool TA, bool TB>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmStridedBatchedExTilingData t)
{
    GroupedKernel<D, TA, TB> kernel{a, b, c, t};
    kernel.Run();
}

template <uint32_t D>
__aicore__ inline void Dispatch(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmStridedBatchedExTilingData t)
{
    if (t.isTransA) {
        if (t.isTransB)
            Run<D, true, true>(a, b, c, t);
        else
            Run<D, true, false>(a, b, c, t);
    } else {
        if (t.isTransB)
            Run<D, false, true>(a, b, c, t);
        else
            Run<D, false, false>(a, b, c, t);
    }
}
} // namespace hgemm_grouped
