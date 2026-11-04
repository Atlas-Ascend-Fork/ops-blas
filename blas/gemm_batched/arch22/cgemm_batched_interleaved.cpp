/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "acl/acl.h"
#include "common/helper/complex_blas3_arch22.h"
#include "cgemm_batched_tiling_data.h"
#include "cgemm_batched_kernel.h"
#include "cgemm_batched_float.h"
#include "cgemm_batched_device_utils.h"

using namespace AscendC;
using namespace cgbt;
namespace {
struct CgbtInterleavedTilingData {
    struct Operand {
        uint32_t rows, cols, ld, negateImag;
    } a, b;
    struct Geometry {
        uint32_t m, n, k, transMode, batchCount, batchBase, batchExtra;
    } gemm;
    struct Output {
        uint32_t m, n, ldc;
    } combine;
};
// ============================================================================
// Realified Cube pipeline: producers, consumer and numerical repair
// ============================================================================
// Keep the uncommon scalar correction out of the producers' instruction
// working set and register allocation. Matrix ownership matches each producer.
__aicore__ __attribute__((noinline)) void CgbtInterleavedCorrect(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, const CgbtInterleavedTilingData& t, uint32_t begin, uint32_t end,
    uint32_t groupSize, LocalTensor<float> out)
{
    const uint32_t sub = GetSubBlockIdx();
    const uint32_t ta = (t.gemm.transMode & CGBT_TRANS_Y) ? (t.a.negateImag ? 2U : 1U) : 0U;
    const uint32_t tb = (t.gemm.transMode & CGBT_TRANS_X) ? (t.b.negateImag ? 2U : 1U) : 0U;
    const CgbtDirectTilingData direct{
        t.combine.m, t.combine.n, t.gemm.k, t.a.ld, t.b.ld, t.combine.ldc, t.gemm.batchCount,
        ta,          tb,          1.0F,     0.0F,   0.0F,   0.0F};
    for (uint32_t group = begin; group < end; group += groupSize) {
        const uint32_t count = end - group < groupSize ? end - group : groupSize;
        const uint32_t first = (count + 1U) / 2U;
        const uint32_t localBegin = group + (sub == 0U ? 0U : first);
        const uint32_t localEnd = localBegin + (sub == 0U ? first : count - first);
        for (uint32_t q = localBegin; q < localEnd; ++q) {
            GlobalTensor<float> a;
            GlobalTensor<float> b;
            GlobalTensor<float> c;
            a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(aarray, q, t.gemm.batchCount)));
            b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(barray, q, t.gemm.batchCount)));
            c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(carray, q, t.gemm.batchCount)));
            for (uint32_t col = 0; col < t.combine.n; ++col) {
                for (uint32_t row = 0; row < t.combine.m; ++row) {
                    const auto value = CgbtDirectValue(direct, a, b, c, row, col);
                    out.SetValue(2U * row, value.real);
                    out.SetValue(2U * row + 1U, value.imag);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopyPad(c[col * t.combine.ldc * 2U], out, DataCopyExtParams{1U, t.combine.m * 8U, 0U, 0U, 0U});
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
        }
    }
}

// Realify each complex coefficient into a 2x2 block. The Cube result then
// already has the public interleaved layout, so alpha=1,beta=0 needs no
// intermediate output or vector interleave pass. Inputs may be transposed,
// padded, independently allocated, or conjugated.
template <uint32_t P>
__aicore__ inline void CgbtInterleavedCheck(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, const CgbtInterleavedTilingData& t,
    uint32_t begin, uint32_t end, LocalTensor<float> flags);

template <uint32_t P, uint32_t COUNT>
__aicore__ inline void CgbtRealifyPanel(
    LocalTensor<float> out, LocalTensor<float> stage, LocalTensor<int32_t> idx, uint32_t count, bool negateImag,
    bool trans)
{
    constexpr uint32_t E = P * P;
    if constexpr (P == 32U) {
        Gather(
            out[2U * P], stage, idx.ReinterpretCast<uint32_t>(), 0U, uint64_t(64), static_cast<uint8_t>(P * count),
            uint16_t(16));
        Copy(out, stage, uint64_t(64), static_cast<uint8_t>(P * count), CopyRepeatParams{1U, 1U, 16U, 8U});
        PipeBarrier<PIPE_V>();
    } else {
        auto swapped = stage[2U * COUNT];
        Gather(
            swapped, stage, idx.ReinterpretCast<uint32_t>(), 0U, uint64_t(64),
            static_cast<uint8_t>(2U * E * count / 64U), uint16_t(8));
        PipeBarrier<PIPE_V>();
        for (uint32_t part = 0; part < (2U * P + 63U) / 64U; ++part) {
            const uint32_t width = 2U * P - part * 64U < 64U ? 2U * P - part * 64U : 64U;
            const CopyRepeatParams copy{1U, 1U, 4U * P / 8U, 2U * P / 8U};
            Copy(out[part * 64U], stage[part * 64U], uint64_t(width), static_cast<uint8_t>(P * count), copy);
            Copy(out[2U * P + part * 64U], swapped[part * 64U], uint64_t(width), static_cast<uint8_t>(P * count), copy);
        }
        PipeBarrier<PIPE_V>();
    }
    const bool reverseSigns = negateImag != trans;
    const uint64_t pattern = reverseSigns ? 0xAAAAAAAAAAAAAAAAULL : 0x5555555555555555ULL;
    const uint32_t offset = reverseSigns ? 0U : 2U * P;
    for (uint32_t part = 0; part < (2U * P + 63U) / 64U; ++part) {
        const uint32_t width = 2U * P - part * 64U < 64U ? 2U * P - part * 64U : 64U;
        uint64_t mask[2] = {pattern >> (64U - width), 0U};
        Muls(
            out[offset + part * 64U], out[offset + part * 64U], -1.0F, mask, static_cast<uint8_t>(P * count),
            UnaryRepeatParams{1U, 1U, 4U * P / 8U, 4U * P / 8U});
    }
}

__aicore__ inline void CgbtAccumulatePanelRange(
    LocalTensor<float> absolute, LocalTensor<float> reduced, LocalTensor<float> maximum, LocalTensor<float> range,
    uint32_t repeats)
{
    WholeReduceMax(reduced, absolute, uint64_t(64), repeats, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    if (repeats <= 64U) {
        WholeReduceMax(maximum, reduced, uint64_t(repeats), 1U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        WholeReduceMax(maximum, reduced, uint64_t(64), 1U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(
            maximum[8U], reduced[64U], uint64_t(repeats - 64U), 1U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        Max(maximum.ReinterpretCast<int32_t>(), maximum.ReinterpretCast<int32_t>(),
            maximum[8U].ReinterpretCast<int32_t>(), uint64_t(1), 1U, BinaryRepeatParams{1U, 1U, 1U, 8U, 8U, 8U});
    }
    PipeBarrier<PIPE_V>();
    // Absolute floating-point maxima have monotonic nonnegative bit
    // patterns. Integer Max preserves an exceptional NaN/Inf marker
    // across iterations without a vector-to-scalar round trip.
    Max(range.ReinterpretCast<int32_t>(), range.ReinterpretCast<int32_t>(), maximum.ReinterpretCast<int32_t>(),
        uint64_t(1), 1U, BinaryRepeatParams{1U, 1U, 1U, 8U, 8U, 8U});
}

struct CgbtSplitState {
    LocalTensor<float> range;
    LocalTensor<int32_t> idx;
    LocalTensor<float> out, stage;
    bool aSide;
    CgbtInterleavedTilingData::Operand split;
    uint32_t pitch, inputElements;
    bool trans;
    uint32_t len;
    bool materialize;
    GlobalTensor<float> ws;
    GlobalTensor<uint64_t> ptrs;
};

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtSplitLoad(CgbtSplitState& state, uint32_t batch, uint32_t count)
{
    constexpr uint32_t E = P * P;
    DataCachePreload(state.ptrs[batch], int16_t(0));
    DataCachePreload(state.ptrs[batch + count - 1U], int16_t(0));
    bool contiguous = state.aSide && state.split.rows == P && state.split.cols == P && state.split.ld == P;
    const uint64_t base = state.ptrs.GetValue(batch);
    if (contiguous)
        for (uint32_t q = 1; q < count; ++q)
            contiguous &= state.ptrs.GetValue(batch + q) == base + q * 2U * E * 4U;
    if (contiguous) {
        GlobalTensor<float> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base));
        DataCopy(state.stage, input, 2U * E * count);
    } else {
        Duplicate(state.stage, 0.0F, uint64_t(64), static_cast<uint8_t>(state.inputElements * count / 64U), 1U, 8U);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (uint32_t q = 0; q < count; ++q) {
            GlobalTensor<float> input;
            input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state.ptrs.GetValue(batch + q)));
            const uint32_t bytes = state.split.rows * 8U, rounded = cblas3::AlignUpU(bytes, 32U);
            DataCopyPad(
                state.stage[q * state.inputElements], input,
                DataCopyExtParams{
                    static_cast<uint16_t>(state.split.cols), bytes, (state.split.ld - state.split.rows) * 8U,
                    (2U * state.pitch * 4U - rounded) / 32U, 0U},
                DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((rounded - bytes) / 4U), 0.0F});
        }
    }
}

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtSplitRange(CgbtSplitState& state, uint32_t count)
{
    constexpr uint32_t E = P * P, COUNT = E * GROUP;
    if (!state.aSide) {
        Gather(
            state.out, state.stage, state.idx.ReinterpretCast<uint32_t>(), 0U, uint64_t(64),
            static_cast<uint8_t>(2U * E * count / 64U), uint16_t(8));
        PipeBarrier<PIPE_V>();
    }
    auto absolute = state.aSide ? state.out : state.stage;
    Abs(absolute, state.aSide ? state.stage : state.out, uint64_t(64), static_cast<uint8_t>(2U * E * count / 64U),
        UnaryRepeatParams{1U, 1U, 8U, 8U});
    PipeBarrier<PIPE_V>();
    // Two/three hardware reductions avoid the general reduction helper's
    // scalar bookkeeping; all source lengths are multiples of 64.
    const uint32_t repeats = 2U * E * count / 64U;
    auto reduced = state.out[2U * COUNT];
    CgbtAccumulatePanelRange(absolute, reduced, state.out[3U * COUNT], state.range, repeats);
}

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtSplitPublish(CgbtSplitState& state, uint32_t count, uint32_t slot)
{
    constexpr uint32_t E = P * P, S = CGBT_SLOT_PLANES * E + CGBT_SLOT_FLAG_FLOATS, COUNT = E * GROUP;
    // A non-transposed, complete B panel is already in the required
    // interleaved layout. The Cube can read the caller's panel directly.
    if (state.materialize) {
        if (state.aSide) {
            CgbtRealifyPanel<P, COUNT>(state.out, state.stage, state.idx, count, state.split.negateImag, state.trans);
        } else if (state.trans) {
            // One Gather applies the composed transpose+interleave
            // permutation directly to the loaded panel; conj for op=C
            // is folded in as the odd-lane negation on the way state.out.

            if (state.split.negateImag) {
                uint64_t mask[2] = {0xAAAAAAAAAAAAAAAAULL, 0U};
                Muls(
                    state.out, state.out, -1.0F, mask, static_cast<uint8_t>(2U * E * count / 64U),
                    UnaryRepeatParams{1U, 1U, 8U, 8U});
            }
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        auto packed = state.aSide || state.trans ? state.out : state.stage;
        DataCopyPad(
            state.ws[slot * S + (state.aSide ? 2U * E : 0U)], packed,
            DataCopyExtParams{static_cast<uint16_t>(count), state.len * 4U, 0U, (S - state.len) * 4U, 0U});
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }
}

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtSplitGroup(CgbtSplitState& state, uint32_t batch, uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P, S = CGBT_SLOT_PLANES * E + CGBT_SLOT_FLAG_FLOATS;
    const uint32_t count = end - batch < GROUP ? end - batch : GROUP;
    const uint32_t step = (batch - begin) / GROUP, phase = step % 2U;
    const uint32_t slot = begin + phase * GROUP;
    // Wait for the consumer before reusing either workspace slot.
    if (step >= 2U)
        CrossCoreWaitFlag<2, PIPE_MTE2>(1U + 2U * phase);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    CgbtSplitLoad<P, GROUP>(state, batch, count);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (batch != begin && state.materialize)
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    CgbtSplitRange<P, GROUP>(state, count);
    CgbtSplitPublish<P, GROUP>(state, count, slot);
    if (batch + count == end) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(
            state.ws[begin * S + 8U * E + (state.aSide ? 0U : 1U)], state.range, DataCopyExtParams{1U, 4U, 0U, 0U, 0U});
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(2U * phase);
}

template <uint32_t P>
__aicore__ inline void CgbtInterleavedSplit(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, const CgbtInterleavedTilingData& t,
    uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P, GROUP = P == 16U ? 15U : (P == 32U ? 4U : 1U), COUNT = E * GROUP;
    constexpr uint32_t RANGE = P == 32U ? 8U * COUNT + 8U * P * GROUP : 10U * COUNT;
    InitSocState();
    LocalTensor<float> storage(TPosition::VECCALC, 0U, RANGE + 8U);
    auto range = storage[RANGE];
    Duplicate(range, 0.0F, 8U);
    auto idx = storage.ReinterpretCast<int32_t>();
    auto out = storage[2U * COUNT], stage = storage[6U * COUNT];
    const bool aSide = GetSubBlockIdx() == 0U;
    const auto& split = aSide ? t.a : t.b;
    // An extra 32-byte block per B row rotates successive rows through UB
    // banks instead of making the transpose Gather repeatedly hit one bank.
    const uint32_t pitch = aSide ? P : P + 4U, inputElements = 2U * P * pitch;
    const bool trans = aSide ? (t.gemm.transMode & CGBT_TRANS_Y) : (t.gemm.transMode & CGBT_TRANS_X);
    const uint32_t len = aSide ? 4U * E : 2U * E;
    const bool materialize = aSide || trans || t.gemm.m != P || t.gemm.k != P;
    const uint32_t demand = end - begin < GROUP ? end - begin : GROUP;
    if (aSide)
        CgbtCreatePairSwapIndex(idx, 2U * E * demand);
    else if (materialize)
        CgbtCreateTransposeIndex<P, P + 4U>(idx, demand);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    GlobalTensor<uint64_t> ptrs;
    ptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aSide ? aarray : barray));
    CgbtSplitState state{range, idx, out, stage, aSide, split, pitch, inputElements, trans, len, materialize, ws, ptrs};
    for (uint32_t batch = begin; batch < end; batch += GROUP) {
        CgbtSplitGroup<P, GROUP>(state, batch, begin, end);
    }
    if (materialize)
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    const uint32_t steps = cblas3::CeilDivU(end - begin, GROUP);
    for (uint32_t step = steps > 2U ? steps - 2U : 0U; step < steps; ++step)
        CrossCoreWaitFlag<2, PIPE_MTE2>(1U + 2U * (step % 2U));
    // Workspace acknowledgements only release input storage; output writes
    // must also finish before an exceptional-value correction can begin.
    CgbtInterleavedCheck<P>(aarray, barray, carray, workspace, t, begin, end, storage);
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P>
__aicore__ inline void CgbtLoadSquarePanels(
    GlobalTensor<uint64_t> ptrs, LocalTensor<float> stage, const CgbtInterleavedTilingData::Operand& operand,
    uint32_t batch, uint32_t count)
{
    constexpr uint32_t E = P * P;
    DataCachePreload(ptrs[batch], int16_t(0));
    DataCachePreload(ptrs[batch + count - 1U], int16_t(0));
    bool contiguous = operand.rows == P && operand.cols == P && operand.ld == P;
    const uint64_t base = ptrs.GetValue(batch);
    if (contiguous)
        for (uint32_t q = 1; q < count; ++q)
            contiguous &= ptrs.GetValue(batch + q) == base + q * 2U * E * 4U;
    if (contiguous) {
        GlobalTensor<float> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(base));
        DataCopy(stage, input, 2U * E * count);
    } else {
        Duplicate(stage, 0.0F, uint64_t(64), static_cast<uint8_t>(2U * E * count / 64U), 1U, 8U);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (uint32_t q = 0; q < count; ++q) {
            GlobalTensor<float> input;
            input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ptrs.GetValue(batch + q)));
            const uint32_t bytes = operand.rows * 8U, rounded = cblas3::AlignUpU(bytes, 32U);
            DataCopyPad(
                stage[q * 2U * E], input,
                DataCopyExtParams{
                    static_cast<uint16_t>(operand.cols), bytes, (operand.ld - operand.rows) * 8U,
                    (2U * P * 4U - rounded) / 32U, 0U},
                DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((rounded - bytes) / 4U), 0.0F});
        }
    }
}

struct CgbtDualSplitState {
    LocalTensor<float> range;
    LocalTensor<int32_t> idx;
    LocalTensor<float> out, stage;
    bool lateB;
    uint32_t sub;
    CgbtInterleavedTilingData::Operand split;
    bool trans;
    GlobalTensor<float> ws;
    GlobalTensor<uint64_t> ptrs, bptrs;
};

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtDualLoad(
    CgbtDualSplitState& state, const CgbtInterleavedTilingData& t, uint32_t batch, uint32_t count, uint32_t groupBatch,
    uint32_t begin)
{
    constexpr uint32_t E = P * P, COUNT = E * GROUP;
    if (!state.lateB) {
        CgbtLoadSquarePanels<P>(state.bptrs, state.stage[2U * COUNT], t.b, batch, count);
    }
    CgbtLoadSquarePanels<P>(state.ptrs, state.stage, state.split, batch, count);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (state.lateB) {
        CgbtLoadSquarePanels<P>(state.bptrs, state.stage[2U * COUNT], t.b, batch, count);
    }
    if (state.lateB)
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (groupBatch != begin)
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtDualRange(CgbtDualSplitState& state, uint32_t count)
{
    constexpr uint32_t E = P * P, COUNT = E * GROUP;
    if constexpr (P != 32U) {
        Abs(state.out, state.stage, uint64_t(64), static_cast<uint8_t>(2U * E * count / 64U),
            UnaryRepeatParams{1U, 1U, 8U, 8U});
        Abs(state.out[2U * COUNT], state.stage[2U * COUNT], uint64_t(64), static_cast<uint8_t>(2U * E * count / 64U),
            UnaryRepeatParams{1U, 1U, 8U, 8U});
        PipeBarrier<PIPE_V>();
        Max(state.out.ReinterpretCast<int32_t>(), state.out.ReinterpretCast<int32_t>(),
            state.out[2U * COUNT].ReinterpretCast<int32_t>(), uint64_t(64), static_cast<uint8_t>(2U * E * count / 64U),
            BinaryRepeatParams{1U, 1U, 1U, 8U, 8U, 8U});
        PipeBarrier<PIPE_V>();
        // Two/three hardware reductions avoid the general reduction helper's
        // scalar bookkeeping; all source lengths are multiples of 64.
        const uint32_t repeats = 2U * E * count / 64U;
        auto reduced = state.out[2U * COUNT];
        CgbtAccumulatePanelRange(state.out, reduced, state.out[3U * COUNT], state.range, repeats);
    }
}

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtDualPublish(
    CgbtDualSplitState& state, const CgbtInterleavedTilingData& t, uint32_t slot, uint32_t count)
{
    constexpr uint32_t E = P * P, S = CGBT_SLOT_PLANES * E + CGBT_SLOT_FLAG_FLOATS, COUNT = E * GROUP, len = 4U * E;
    if (t.gemm.m != P || t.gemm.k != P) {
        // Both producers can pad incomplete B panels while sharing A's
        // realification work. Complete B panels are consumed directly.
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(
            state.ws[slot * S], state.stage[2U * COUNT],
            DataCopyExtParams{static_cast<uint16_t>(count), 2U * E * 4U, 0U, (S - 2U * E) * 4U, 0U});
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    // Both producers realify A; Cube reads or pads B separately.
    CgbtRealifyPanel<P, COUNT>(state.out, state.stage, state.idx, count, state.split.negateImag, state.trans);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(
        state.ws[slot * S + 2U * E], state.out,
        DataCopyExtParams{static_cast<uint16_t>(count), len * 4U, 0U, (S - len) * 4U, 0U});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtDualLateRange(CgbtDualSplitState& state, uint32_t count)
{
    constexpr uint32_t E = P * P, COUNT = E * GROUP;
    if constexpr (P == 32U) {
        if (count != 0U) {
            if (state.lateB)
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Abs(state.stage, state.stage, uint64_t(64), static_cast<uint8_t>(2U * E * count / 64U),
                UnaryRepeatParams{1U, 1U, 8U, 8U});
            Abs(state.stage[2U * COUNT], state.stage[2U * COUNT], uint64_t(64),
                static_cast<uint8_t>(2U * E * count / 64U), UnaryRepeatParams{1U, 1U, 8U, 8U});
            PipeBarrier<PIPE_V>();
            Max(state.stage.ReinterpretCast<int32_t>(), state.stage.ReinterpretCast<int32_t>(),
                state.stage[2U * COUNT].ReinterpretCast<int32_t>(), uint64_t(64),
                static_cast<uint8_t>(2U * E * count / 64U), BinaryRepeatParams{1U, 1U, 1U, 8U, 8U, 8U});
            PipeBarrier<PIPE_V>();
            // Two/three hardware reductions avoid the general reduction helper's
            // scalar bookkeeping; all source lengths are multiples of 64.
            const uint32_t repeats = 2U * E * count / 64U;
            auto reduced = state.stage[4U * COUNT + 16U];
            CgbtAccumulatePanelRange(state.stage, reduced, state.stage[4U * COUNT + 224U], state.range, repeats);
        }
    }
}

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtDualGroup(
    CgbtDualSplitState& state, const CgbtInterleavedTilingData& t, uint32_t groupBatch, uint32_t begin, uint32_t end)
{
    const uint32_t groupCount = end - groupBatch < 2U * GROUP ? end - groupBatch : 2U * GROUP;
    const uint32_t first = (groupCount + 1U) / 2U;
    const uint32_t offset = state.sub == 0U ? 0U : first;
    const uint32_t batch = groupBatch + offset, count = state.sub == 0U ? first : groupCount - first;
    const uint32_t step = (groupBatch - begin) / (2U * GROUP), phase = step % 2U;
    const uint32_t slot = begin + phase * 2U * GROUP + offset;
    // Wait for the consumer before reusing either workspace slot.
    if (step >= 2U)
        CrossCoreWaitFlag<2, PIPE_MTE2>(1U + 2U * phase);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if (count != 0U) {
        CgbtDualLoad<P, GROUP>(state, t, batch, count, groupBatch, begin);
        CgbtDualRange<P, GROUP>(state, count);
        CgbtDualPublish<P, GROUP>(state, t, slot, count);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(2U * phase);
    CgbtDualLateRange<P, GROUP>(state, count);
}

template <uint32_t P>
__aicore__ inline void CgbtInterleavedDualSplit(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, const CgbtInterleavedTilingData& t,
    uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P, GROUP = P == 16U ? 15U : (P == 32U ? 4U : 1U), COUNT = E * GROUP;
    InitSocState();
    // Hide pointer-array cache misses behind index generation, just as the
    // compact producers do. Cache lines contain eight independent pointers.
    GlobalTensor<uint64_t> prefetchA, prefetchB;
    prefetchA.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aarray));
    prefetchB.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(barray));
    for (uint32_t batch = begin; batch < end; batch += 8U) {
        DataCachePreload(prefetchA[batch], int16_t(0));
        DataCachePreload(prefetchB[batch], int16_t(0));
    }
    LocalTensor<float> storage(TPosition::VECCALC, 0U, 10U * COUNT + 256U);
    auto range = storage[10U * COUNT];
    Duplicate(range, 0.0F, 8U);
    auto idx = storage.ReinterpretCast<int32_t>();
    auto out = storage[2U * COUNT], stage = storage[6U * COUNT];
    const bool lateB = P == 32U && t.gemm.m == P && t.gemm.k == P;
    const uint32_t sub = GetSubBlockIdx();
    const auto& split = t.a;
    const bool trans = (t.gemm.transMode & CGBT_TRANS_Y);
    const uint32_t len = 4U * E;
    const uint32_t demand = (end - begin + 1U) / 2U < GROUP ? (end - begin + 1U) / 2U : GROUP;
    CgbtCreatePairSwapIndex(idx, 2U * E * demand);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    GlobalTensor<uint64_t> ptrs;
    ptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aarray));
    GlobalTensor<uint64_t> bptrs;
    bptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(barray));
    CgbtDualSplitState state{range, idx, out, stage, lateB, sub, split, trans, ws, ptrs, bptrs};
    for (uint32_t groupBatch = begin; groupBatch < end; groupBatch += 2U * GROUP) {
        CgbtDualGroup<P, GROUP>(state, t, groupBatch, begin, end);
    }
    if (sub == 0U || end - begin > 1U)
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    const uint32_t steps = cblas3::CeilDivU(end - begin, 2U * GROUP);
    for (uint32_t step = steps > 2U ? steps - 2U : 0U; step < steps; ++step)
        CrossCoreWaitFlag<2, PIPE_MTE2>(1U + 2U * (step % 2U));
    // Each producer checked both inputs for its own batch members. Keep the
    // range in UB; no GM flags or cross-producer numerical check is needed.
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const bool ordinary = range.GetValue(0) <= CGBT_SAFE_COMPONENT_MAGNITUDE;

    if (!ordinary)
        CgbtInterleavedCorrect(aarray, barray, carray, t, begin, end, 2U * GROUP, out);
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P, bool DUAL>
struct CgbtCubeState {
    static constexpr uint32_t E = P * P, S = CGBT_SLOT_PLANES * E + CGBT_SLOT_FLAG_FLOATS,
                              ALIGNED = (P + 15U) / 16U * 16U, X = 2U * P * ALIGNED,
                              ORIGINAL_HALF = P == 16U ? 15U : (P <= 32U ? 4U : (P == 40U ? 2U : 1U)),
                              BANKS = (P == 24U || P == 32U) ? 2U : 1U, HALF = ORIGINAL_HALF / BANKS,
                              GROUP = ORIGINAL_HALF * (DUAL ? 2U : 1U);
    LocalTensor<float> a1{TPosition::A1, 0U, X* GROUP};
    LocalTensor<float> b1{TPosition::B1, X* GROUP * 4U, 4U * E* GROUP};
    LocalTensor<float> a2{TPosition::A2, 0U, X* HALF* BANKS};
    LocalTensor<float> b2{TPosition::B2, 0U, 4U * E* HALF* BANKS};
    LocalTensor<float> c0{TPosition::CO1, 0U, X* GROUP};
    Nd2NzParams ca{}, cb{};
    MmadParams mm{};
    FixpipeParamsV220 fix{};
    GlobalTensor<float> ws;
    GlobalTensor<uint64_t> bptr, cptr;
    uint32_t issuedChunks = 0U;
};

template <uint32_t P, bool DUAL>
__aicore__ inline void CgbtInitCubeState(
    CgbtCubeState<P, DUAL>& state, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace,
    const CgbtInterleavedTilingData& t)
{
    state.ca.nValue = P;
    state.ca.dValue = state.ca.srcDValue = 2U * P;
    state.ca.dstNzC0Stride = CgbtCubeState<P, DUAL>::ALIGNED;
    state.ca.dstNzNStride = 1U;
    state.cb.nValue = state.cb.dValue = state.cb.srcDValue = state.cb.dstNzC0Stride = 2U * P;
    state.cb.dstNzNStride = 1U;
    state.ca.srcNdMatrixStride = state.cb.srcNdMatrixStride = CgbtCubeState<P, DUAL>::S;
    state.ca.dstNzMatrixStride = CgbtCubeState<P, DUAL>::X;
    state.cb.dstNzMatrixStride = 4U * CgbtCubeState<P, DUAL>::E;
    state.mm.m = P;
    state.mm.n = state.mm.k = 2U * P;
    state.mm.cmatrixInitVal = true;
    state.fix.mSize = t.combine.n;
    state.fix.nSize = 2U * t.combine.m;
    state.fix.srcStride = CgbtCubeState<P, DUAL>::ALIGNED;
    state.fix.dstStride = 2U * t.combine.ldc;
    state.fix.srcNdStride = CgbtCubeState<P, DUAL>::X / 256U;
    state.fix.dstNdStride = 2U * t.combine.ldc * t.combine.n;
    state.ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    state.bptr.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(barray));
    state.cptr.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(carray));
}

template <uint32_t P, bool DUAL>
__aicore__ inline void CgbtCubeLoad(
    CgbtCubeState<P, DUAL>& state, const CgbtInterleavedTilingData& t, uint32_t batch, uint32_t count, uint32_t slot,
    bool directB, bool contiguousB, uint64_t bbase)
{
    state.ca.ndNum = state.cb.ndNum = count;
    if (directB) {
        Nd2NzParams inputCopy = state.ca;
        inputCopy.srcDValue = 2U * t.b.ld;
        inputCopy.srcNdMatrixStride = 2U * t.b.ld * P;
        if (contiguousB) {
            GlobalTensor<float> input;
            input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bbase));
            DataCopy(state.a1, input, inputCopy);
        } else {
            inputCopy.ndNum = 1U;
            for (uint32_t q = 0; q < count; ++q) {
                GlobalTensor<float> input;
                input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state.bptr.GetValue(batch + q)));
                DataCopy(state.a1[q * CgbtCubeState<P, DUAL>::X], input, inputCopy);
            }
        }
    } else
        DataCopy(state.a1, state.ws[slot * CgbtCubeState<P, DUAL>::S], state.ca);
    DataCopy(state.b1, state.ws[slot * CgbtCubeState<P, DUAL>::S + 2U * CgbtCubeState<P, DUAL>::E], state.cb);
}

template <uint32_t P, bool DUAL>
__aicore__ inline void CgbtCubeMultiply(
    CgbtCubeState<P, DUAL>& state, const CgbtInterleavedTilingData& t, uint32_t batch, uint32_t begin, uint32_t count)
{
    auto la = CgbtLocalLoad(CgbtCubeState<P, DUAL>::ALIGNED, 2U * P, false), lb = CgbtLocalLoad(2U * P, 2U * P, true);
    for (uint32_t chunk = 0; chunk < count; chunk += CgbtCubeState<P, DUAL>::HALF) {
        const uint32_t chunkEnd =
            chunk + CgbtCubeState<P, DUAL>::HALF < count ? chunk + CgbtCubeState<P, DUAL>::HALF : count;
        const uint32_t bank = state.issuedChunks % CgbtCubeState<P, DUAL>::BANKS;
        const event_t bankEvent = static_cast<event_t>(bank);
        if (state.issuedChunks >= CgbtCubeState<P, DUAL>::BANKS)
            WaitFlag<HardEvent::M_MTE1>(bankEvent);
        for (uint32_t q = chunk; q < chunkEnd; ++q) {
            LoadData(
                state.a2[(bank * CgbtCubeState<P, DUAL>::HALF + q - chunk) * CgbtCubeState<P, DUAL>::X],
                state.a1[q * CgbtCubeState<P, DUAL>::X], la);
            if (t.gemm.transMode & CGBT_TRANS_Y) {
                const LoadData2dParams transpose{
                    0U, static_cast<uint8_t>(4U * CgbtCubeState<P, DUAL>::E / 128U), 1U, 0U, 0U, false, 0U};
                LoadData(
                    state.b2[(bank * CgbtCubeState<P, DUAL>::HALF + q - chunk) * 4U * CgbtCubeState<P, DUAL>::E],
                    state.b1[q * 4U * CgbtCubeState<P, DUAL>::E], transpose);
            } else
                LoadData(
                    state.b2[(bank * CgbtCubeState<P, DUAL>::HALF + q - chunk) * 4U * CgbtCubeState<P, DUAL>::E],
                    state.b1[q * 4U * CgbtCubeState<P, DUAL>::E], lb);
        }
        SetFlag<HardEvent::MTE1_M>(bankEvent);
        WaitFlag<HardEvent::MTE1_M>(bankEvent);
        if (chunk == 0U && batch != begin)
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (uint32_t q = chunk; q < chunkEnd; ++q)
            Mmad(
                state.c0[q * CgbtCubeState<P, DUAL>::X],
                state.a2[(bank * CgbtCubeState<P, DUAL>::HALF + q - chunk) * CgbtCubeState<P, DUAL>::X],
                state.b2[(bank * CgbtCubeState<P, DUAL>::HALF + q - chunk) * 4U * CgbtCubeState<P, DUAL>::E], state.mm);
        SetFlag<HardEvent::M_MTE1>(bankEvent);
        ++state.issuedChunks;
    }
}

template <uint32_t P, bool DUAL>
__aicore__ inline void CgbtCubeStore(
    CgbtCubeState<P, DUAL>& state, uint32_t batch, uint32_t count, bool contiguousC, uint64_t cbase)
{
    if (contiguousC) {
        GlobalTensor<float> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cbase));
        state.fix.ndNum = count;
        Fixpipe(c, state.c0, state.fix);
    } else {
        state.fix.ndNum = 1U;
        for (uint32_t q = 0; q < count; ++q) {
            GlobalTensor<float> c;
            c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state.cptr.GetValue(batch + q)));
            Fixpipe(c, state.c0[q * CgbtCubeState<P, DUAL>::X], state.fix);
        }
    }
}

template <uint32_t P, bool DUAL>
__aicore__ inline void CgbtCubeGroup(
    CgbtCubeState<P, DUAL>& state, const CgbtInterleavedTilingData& t, uint32_t batch, uint32_t begin, uint32_t end,
    bool directB)
{
    const uint32_t count = end - batch < CgbtCubeState<P, DUAL>::GROUP ? end - batch : CgbtCubeState<P, DUAL>::GROUP;
    const uint32_t phase = ((batch - begin) / CgbtCubeState<P, DUAL>::GROUP) % 2U,
                   slot = begin + phase * CgbtCubeState<P, DUAL>::GROUP;
    DataCachePreload(state.cptr[batch], int16_t(0));
    DataCachePreload(state.cptr[batch + count - 1U], int16_t(0));
    if (directB) {
        DataCachePreload(state.bptr[batch], int16_t(0));
        DataCachePreload(state.bptr[batch + count - 1U], int16_t(0));
    }
    const uint64_t cbase = state.cptr.GetValue(batch);
    const uint64_t bbase = directB ? state.bptr.GetValue(batch) : 0U;
    bool contiguousC = true, contiguousB = true;
    for (uint32_t q = 1; q < count; ++q) {
        contiguousC &= state.cptr.GetValue(batch + q) == cbase + q * state.fix.dstNdStride * 4U;
        if (directB)
            contiguousB &= state.bptr.GetValue(batch + q) == bbase + q * 2U * t.b.ld * P * 4U;
    }
    CrossCoreWaitFlag<2, PIPE_MTE2>(2U * phase);
    // A previous LoadData must release L1 before MTE2 refills it.
    if (batch != begin)
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    CgbtCubeLoad(state, t, batch, count, slot, directB, contiguousB, bbase);
    // Release the producer's GM slot as soon as both inputs reach L1.
    if (batch + count != end)
        CrossCoreSetFlag<2, PIPE_MTE2>(1U + 2U * phase);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    CgbtCubeMultiply(state, t, batch, begin, count);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    CgbtCubeStore(state, batch, count, contiguousC, cbase);
    if (batch + count == end)
        CrossCoreSetFlag<2, PIPE_FIX>(1U + 2U * phase);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
}

template <uint32_t P, bool DUAL = false>
__aicore__ inline void CgbtInterleavedCube(
    GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, const CgbtInterleavedTilingData& t, uint32_t begin, uint32_t end)
{
    static_assert(
        CgbtCubeState<P, DUAL>::X * CgbtCubeState<P, DUAL>::HALF * CgbtCubeState<P, DUAL>::BANKS * 4U <= 64U * 1024U,
        "L0A capacity");
    static_assert(
        4U * CgbtCubeState<P, DUAL>::E * CgbtCubeState<P, DUAL>::HALF * CgbtCubeState<P, DUAL>::BANKS * 4U <=
            64U * 1024U,
        "L0B capacity");
    InitSocState();
    SetHF32Mode(HF32Mode::DISABLE);
    CgbtCubeState<P, DUAL> state;
    CgbtInitCubeState(state, barray, carray, workspace, t);
    const bool directB = !(t.gemm.transMode & CGBT_TRANS_X) && t.gemm.m == P && t.gemm.k == P;
    // Alternate operand banks across batch boundaries as well as within
    // a group; wait only when the same bank is reused.
    for (uint32_t batch = begin; batch < end; batch += CgbtCubeState<P, DUAL>::GROUP) {
        CgbtCubeGroup(state, t, batch, begin, end, directB);
    }
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    for (uint32_t bank = 0U; bank < CgbtCubeState<P, DUAL>::BANKS && bank < state.issuedChunks; ++bank)
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(bank));
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}

template <uint32_t P>
__aicore__ inline void CgbtInterleavedCheck(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, const CgbtInterleavedTilingData& t,
    uint32_t begin, uint32_t end, LocalTensor<float> flags)
{
    constexpr uint32_t E = P * P, S = CGBT_SLOT_PLANES * E + CGBT_SLOT_FLAG_FLOATS;
    auto output = flags[8U];
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    DataCopyPad(
        flags, ws[begin * S + 8U * E], DataCopyExtParams{1U, 8U, 0U, 0U, 0U},
        DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    const bool ordinary =
        flags.GetValue(0) <= CGBT_SAFE_COMPONENT_MAGNITUDE && flags.GetValue(1) <= CGBT_SAFE_COMPONENT_MAGNITUDE;

    if (ordinary)
        return;
    CgbtInterleavedCorrect(aarray, barray, carray, t, begin, end, 2U, output);
}

// ============================================================================
// Compact input preparation and per-producer range checks
// ============================================================================
template <uint32_t P>
__aicore__ inline void CgbtCompactAIndex(LocalTensor<int32_t> idx, const CgbtInterleavedTilingData::Operand& split)
{
    auto seed = idx.ReinterpretCast<uint64_t>();
#pragma unroll
    for (uint32_t i = 0U; i < 4U; ++i) {
        const uint32_t lo = (2U * i + 8U) * 4U, hi = lo + 4U;
        seed.SetValue(i, (uint64_t(hi) << 32U) | lo);
        seed.SetValue(P + i, (uint64_t(lo) << 32U) | hi);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    for (uint32_t filled = 8U; filled < 2U * P; filled *= 2U) {
        const uint32_t size = filled < 2U * P - filled ? filled : 2U * P - filled;
        Adds(idx[filled], idx, int32_t(filled * 4U), size);
        Adds(idx[2U * P + filled], idx[2U * P], int32_t(filled * 4U), size);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t rows = 1U; rows < P; rows *= 2U) {
        Adds(
            idx[rows * 4U * P], idx, int32_t(rows * 2U * split.rows * 4U),
            (rows < P - rows ? rows : P - rows) * 4U * P);
        PipeBarrier<PIPE_V>();
    }
    if (split.rows < P) {
        for (uint32_t part = 0U; part < 2U * P; part += 64U) {
            const uint32_t width = 2U * P - part < 64U ? 2U * P - part : 64U;
            const uint32_t valid = 2U * split.rows > part ? 2U * split.rows - part : 0U;
            if (valid < width) {
                uint64_t invalid[2] = {(~0ULL << valid) & (~0ULL >> (64U - width)), 0U};
                Duplicate(idx[part], int32_t(0), invalid, uint8_t(2U * P), uint16_t(1), uint16_t(2U * P / 8U));
            }
        }
    }
    if (split.cols < P)
        Duplicate(idx[split.cols * 4U * P], int32_t(0), (P - split.cols) * 4U * P);
}

template <uint32_t P>
__aicore__ inline void CgbtCompactBIndex(LocalTensor<int32_t> idx, const CgbtInterleavedTilingData::Operand& split)
{
    auto seed = idx.ReinterpretCast<uint64_t>();
#pragma unroll
    for (uint32_t i = 0U; i < 4U; ++i) {
        const uint32_t lo = (2U * i + 8U) * 4U;
        seed.SetValue(i, (uint64_t(lo + 4U) << 32U) | lo);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    for (uint32_t filled = 8U; filled < 2U * P; filled *= 2U) {
        Adds(idx[filled], idx, int32_t(filled * 4U), filled < 2U * P - filled ? filled : 2U * P - filled);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t rows = 1U; rows < P; rows *= 2U) {
        Adds(idx[rows * 2U * P], idx, int32_t(rows * split.rows * 8U), (rows < P - rows ? rows : P - rows) * 2U * P);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t part = 0U; part < 2U * P; part += 64U) {
        const uint32_t width = 2U * P - part < 64U ? 2U * P - part : 64U;
        const uint32_t valid = 2U * split.rows > part ? 2U * split.rows - part : 0U;
        if (valid < width) {
            uint64_t mask[2] = {(~0ULL << valid) & (~0ULL >> (64U - width)), 0U};
            Duplicate(idx[part], int32_t(0), mask, uint8_t(P), uint16_t(1), uint16_t(2U * P / 8U));
        }
    }
    if (split.cols < P)
        Duplicate(idx[split.cols * 2U * P], int32_t(0), (P - split.cols) * 2U * P);
}

template <uint32_t P>
__aicore__ inline void CgbtCompactTransposedBIndex(
    LocalTensor<int32_t> idx, const CgbtInterleavedTilingData::Operand& split)
{
    auto seed = idx.ReinterpretCast<uint64_t>();
#pragma unroll
    for (uint32_t i = 0U; i < 4U; ++i) {
        const uint32_t lo = (i * 2U * split.rows + 8U) * 4U;
        seed.SetValue(i, (uint64_t(lo + 4U) << 32U) | lo);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    for (uint32_t filled = 8U; filled < 2U * P; filled *= 2U) {
        Adds(idx[filled], idx, int32_t(filled * split.rows * 4U), filled < 2U * P - filled ? filled : 2U * P - filled);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t rows = 1U; rows < P; rows *= 2U) {
        Adds(idx[rows * 2U * P], idx, int32_t(rows * 8U), (rows < P - rows ? rows : P - rows) * 2U * P);
        PipeBarrier<PIPE_V>();
    }
    if (split.cols < P) {
        for (uint32_t part = 0U; part < 2U * P; part += 64U) {
            const uint32_t width = 2U * P - part < 64U ? 2U * P - part : 64U;
            const uint32_t valid = 2U * split.cols > part ? 2U * split.cols - part : 0U;
            if (valid < width) {
                uint64_t invalid[2] = {(~0ULL << valid) & (~0ULL >> (64U - width)), 0U};
                Duplicate(idx[part], int32_t(0), invalid, uint8_t(P), uint16_t(1), uint16_t(2U * P / 8U));
            }
        }
    }
    if (split.rows < P)
        Duplicate(idx[split.rows * 2U * P], int32_t(0), (P - split.rows) * 2U * P);
}

template <uint32_t P>
__aicore__ inline void CgbtCompactIndex(
    LocalTensor<int32_t> idx, const CgbtInterleavedTilingData::Operand& split, bool aSide, bool transposeB = true)
{
    // The first block of each raw slot is a zero sentinel. Invalid padded
    // positions gather from that block, so no out-of-range input is touched.
    if (aSide) {
        CgbtCompactAIndex<P>(idx, split);
    } else if (!transposeB) {
        CgbtCompactBIndex<P>(idx, split);
    } else {
        CgbtCompactTransposedBIndex<P>(idx, split);
    }
}

// Compact physical panels can be loaded as one contiguous DMA. Construct the
// padded realification/transpose in UB, avoiding one short DMA block per row.
template <uint32_t P>
__aicore__ inline void CgbtCompactSigns(
    LocalTensor<float> out, const CgbtInterleavedTilingData& t, const CgbtInterleavedTilingData::Operand& split,
    bool aSide, uint32_t count)
{
    constexpr uint32_t E = P * P;
    if (aSide) {
        const bool reverseSigns = split.negateImag != bool(t.gemm.transMode & CGBT_TRANS_Y);
        // Preserve the same complex product while absorbing B's sign.
        const bool absorbB = t.b.negateImag && !t.a.negateImag;
        const uint64_t pattern = (reverseSigns || absorbB) ? 0xAAAAAAAAAAAAAAAAULL : 0x5555555555555555ULL;
        const uint32_t offset = reverseSigns && !absorbB ? 0U : 2U * P;
        for (uint32_t part = 0U; part < 2U * P; part += 64U) {
            const uint32_t width = 2U * P - part < 64U ? 2U * P - part : 64U;
            uint64_t mask[2] = {pattern >> (64U - width), 0U};
            Muls(
                out[offset + part], out[offset + part], -1.0F, mask, static_cast<uint8_t>(P * count),
                UnaryRepeatParams{1, 1, uint8_t(4U * P / 8U), uint8_t(4U * P / 8U)});
        }
    } else if (split.negateImag && t.a.negateImag) {
        uint64_t odd[2] = {0xAAAAAAAAAAAAAAAAULL, 0U};
        Muls(out, out, -1.0F, odd, static_cast<uint8_t>(2U * E * count / 64U), UnaryRepeatParams{1, 1, 8, 8});
    }
}

template <uint32_t P>
__aicore__ inline void CgbtCompactGather(
    LocalTensor<float> out, LocalTensor<float> stage, LocalTensor<int32_t> idx, uint32_t count, uint32_t len,
    uint32_t active)
{
    constexpr uint32_t RAW = 2U * P * P + 8U;
    for (uint32_t q = 0U; q < count; ++q) {
        for (uint32_t base = 0U; base < active; base += 8192U) {
            const uint32_t chunk = active - base < 8192U ? active - base : 8192U;
            Gather(
                out[q * len + base], stage, idx[base].ReinterpretCast<uint32_t>(), q * RAW * 4U, uint64_t(64),
                static_cast<uint8_t>(chunk / 64U), uint16_t(8));
        }
    }
}

struct CgbtCompactState {
    LocalTensor<int32_t> idx;
    LocalTensor<float> out, stage, range, maxima;
    bool aSide;
    CgbtInterleavedTilingData::Operand split;
    uint32_t len;
    GlobalTensor<float> ws;
    GlobalTensor<uint64_t> ptrs;
};

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtCompactRange(
    CgbtCompactState& state, uint32_t batch, uint32_t count, uint32_t begin, uint32_t end, uint32_t phase)
{
    constexpr uint32_t E = P * P, COUNT = E * GROUP, S = CGBT_SLOT_PLANES * E + CGBT_SLOT_FLAG_FLOATS;
    // Only the first of A's two realified rows is needed for state.range checks.
    if (state.aSide) {
        for (uint32_t part = 0U; part < 2U * P; part += 64U) {
            const uint32_t width = 2U * P - part < 64U ? 2U * P - part : 64U;
            Abs(state.stage[part], state.out[part], uint64_t(width), static_cast<uint8_t>(P * count),
                UnaryRepeatParams{1U, 1U, uint8_t(2U * P / 8U), uint8_t(4U * P / 8U)});
        }
    } else {
        Abs(state.stage, state.out, uint64_t(64), static_cast<uint8_t>(2U * E * count / 64U),
            UnaryRepeatParams{1U, 1U, 8U, 8U});
    }
    PipeBarrier<PIPE_V>();
    // Fold the two halves and accumulate across matrices in UB.
    // Reduce horizontally only when the final group publishes its state.range.
    Max(state.stage.ReinterpretCast<int32_t>(), state.stage.ReinterpretCast<int32_t>(),
        state.stage[COUNT].ReinterpretCast<int32_t>(), COUNT);
    PipeBarrier<PIPE_V>();
    Max(state.maxima.ReinterpretCast<int32_t>(), state.maxima.ReinterpretCast<int32_t>(),
        state.stage.ReinterpretCast<int32_t>(), COUNT);
    if (batch + count == end) {
        PipeBarrier<PIPE_V>();
        auto reduced = state.stage[2U * COUNT];
        constexpr uint32_t repeats = COUNT / 64U;
        WholeReduceMax(reduced, state.maxima, uint64_t(64), repeats, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(state.range, reduced, uint64_t(repeats), 1U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(
            state.ws[begin * S + 8U * E + (state.aSide ? 0U : 1U)], state.range, DataCopyExtParams{1U, 4U, 0U, 0U, 0U});
        CrossCoreSetFlag<2, PIPE_MTE3>(2U * phase);
    }
}

template <uint32_t P, uint32_t GROUP>
__aicore__ inline void CgbtCompactGroup(
    CgbtCompactState& state, const CgbtInterleavedTilingData& t, uint32_t batch, uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P, S = CGBT_SLOT_PLANES * E + CGBT_SLOT_FLAG_FLOATS, RAW = 2U * E + 8U;
    const uint32_t count = end - batch < GROUP ? end - batch : GROUP;
    const uint32_t step = (batch - begin) / GROUP, phase = step % 2U, slot = begin + phase * GROUP;
    if (step >= 2U)
        CrossCoreWaitFlag<2, PIPE_MTE2>(1U + 2U * phase);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    for (uint32_t q = 0U; q < count; ++q) {
        Duplicate(state.stage[q * RAW], 0.0F, 8U);
        GlobalTensor<float> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state.ptrs.GetValue(batch + q)));
        DataCopyPad(
            state.stage[q * RAW + 8U], input,
            DataCopyExtParams{1U, state.split.rows * state.split.cols * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (batch != begin)
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    PipeBarrier<PIPE_V>();
    CgbtCompactGather<P>(state.out, state.stage, state.idx, count, state.len, state.len);
    PipeBarrier<PIPE_V>();
    CgbtCompactSigns<P>(state.out, t, state.split, state.aSide, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(
        state.ws[slot * S + (state.aSide ? 2U * E : 0U)], state.out,
        DataCopyExtParams{static_cast<uint16_t>(count), state.len * 4U, 0U, (S - state.len) * 4U, 0U});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    if (batch + count != end)
        CrossCoreSetFlag<2, PIPE_MTE3>(2U * phase);
    CgbtCompactRange<P, GROUP>(state, batch, count, begin, end, phase);
}

template <uint32_t P>
__aicore__ inline void CgbtInterleavedCompact(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, const CgbtInterleavedTilingData& t,
    uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P, GROUP = P <= 32U ? 4U : (P == 40U ? 2U : 1U), COUNT = E * GROUP;

    static_assert(GROUP == 1U && COUNT / 64U <= 64U, "terminal range reduction extent");
    static_assert((4U * E + 7U * COUNT + 256U) * 4U <= 192U * 1024U, "UB capacity");
    InitSocState();
    GlobalTensor<uint64_t> ap, bp;
    ap.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aarray));
    bp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(barray));
    for (uint32_t batch = begin; batch < end; batch += 8U) {
        DataCachePreload(ap[batch], int16_t(0));
        DataCachePreload(bp[batch], int16_t(0));
    }
    LocalTensor<float> storage(TPosition::VECCALC, 0U, 4U * E + 7U * COUNT + 256U);
    auto idx = storage.ReinterpretCast<int32_t>();
    auto out = storage[4U * E], stage = out[4U * COUNT], range = stage[2U * COUNT + 248U];
    auto maxima = range[8U];
    Duplicate(maxima.ReinterpretCast<int32_t>(), int32_t(0), COUNT);
    const bool aSide = GetSubBlockIdx() == 0U;
    const auto& split = aSide ? t.a : t.b;
    const uint32_t len = aSide ? 4U * E : 2U * E;
    Duplicate(range, 0.0F, 8U);
    CgbtCompactIndex<P>(idx, split, aSide, t.gemm.transMode & CGBT_TRANS_X);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    GlobalTensor<uint64_t> ptrs;
    ptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aSide ? aarray : barray));
    CgbtCompactState state{idx, out, stage, range, maxima, aSide, split, len, ws, ptrs};
    for (uint32_t batch = begin; batch < end; batch += GROUP) {
        CgbtCompactGroup<P, GROUP>(state, t, batch, begin, end);
    }
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    const uint32_t steps = cblas3::CeilDivU(end - begin, GROUP);
    for (uint32_t step = steps > 2U ? steps - 2U : 0U; step < steps; ++step)
        CrossCoreWaitFlag<2, PIPE_MTE2>(1U + 2U * (step % 2U));
    CgbtInterleavedCheck<P>(aarray, barray, carray, workspace, t, begin, end, storage);
    PipeBarrier<PIPE_ALL>();
}

// Both AIVs own half of each batch group, balancing A's larger realification
// with B's transpose work. The Cube consumes their completed panels together.
template <uint32_t P>
__aicore__ inline void CgbtCompactPrefetch(
    GM_ADDR aarray, GM_ADDR barray, const CgbtInterleavedTilingData& t, uint32_t batch, uint32_t count,
    LocalTensor<float> stage, uint32_t phase)
{
    static_assert(P <= 32U, "two input banks are reserved for compact extents up to 32");
    constexpr uint32_t E = P * P, COUNT = 2U * E, RAW = 2U * E + 8U;
    for (uint32_t side = 0U; side < 2U; ++side) {
        const auto& split = side == 0U ? t.a : t.b;
        GlobalTensor<uint64_t> ptrs;
        ptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(side == 0U ? aarray : barray));
        auto raw = stage[side * (2U * COUNT + 64U)];
        for (uint32_t q = 0U; q < count; ++q) {
            GlobalTensor<float> input;
            input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ptrs.GetValue(batch + q)));
            DataCopyPad(
                raw[q * RAW + 8U], input, DataCopyExtParams{1U, split.rows * split.cols * 8U, 0U, 0U, 0U},
                DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
        }
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(phase + 2U * side));
    }
}

struct CgbtBalancedState {
    LocalTensor<int32_t> indices;
    LocalTensor<float> outputs, inputBase;
    bool directB;
    uint32_t sub;
    GlobalTensor<float> ws;
};

template <uint32_t P>
__aicore__ inline void CgbtBalancedPrefetchNext(
    CgbtBalancedState& state, GM_ADDR aarray, GM_ADDR barray, const CgbtInterleavedTilingData& t, uint32_t groupBatch,
    uint32_t total, uint32_t end, uint32_t step, uint32_t phase)
{
    constexpr uint32_t E = P * P;
    constexpr uint32_t HALF = P <= 32U ? 2U : 1U;
    constexpr uint32_t GROUP = HALF * 2U;
    constexpr uint32_t COUNT = E * HALF;
    constexpr uint32_t INPUT = 4U * COUNT + 128U;
    if constexpr (P <= 32U) {
        if (groupBatch + total < end) {
            if (step != 0U) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            }
            const uint32_t next = groupBatch + GROUP;
            const uint32_t nextTotal = end - next < GROUP ? end - next : GROUP;
            const uint32_t nextFirst = (nextTotal + 1U) / 2U;
            CgbtCompactPrefetch<P>(
                aarray, barray, t, next + (state.sub == 0U ? 0U : nextFirst),
                state.sub == 0U ? nextFirst : nextTotal - nextFirst, state.inputBase[(1U - phase) * INPUT], 1U - phase);
        }
    }
}

template <uint32_t P>
__aicore__ inline void CgbtBalancedLoad(
    CgbtBalancedState& state, GM_ADDR aarray, GM_ADDR barray, const CgbtInterleavedTilingData& t,
    LocalTensor<float> stage, uint32_t batch, uint32_t count, bool firstGroup)
{
    constexpr uint32_t E = P * P;
    constexpr uint32_t HALF = P <= 32U ? 2U : 1U;
    constexpr uint32_t COUNT = E * HALF;
    constexpr uint32_t RAW = 2U * E + 8U;
    if constexpr (P > 32U) {
        // The first input DMA writes fresh stage memory, disjoint from
        // the index and range initialization. Only later groups reuse
        // memory read by the previous range check.
        if (!firstGroup) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        for (uint32_t side = 0U; side < 2U; ++side) {
            const auto& split = side == 0U ? t.a : t.b;
            auto raw = stage[side * (2U * COUNT + 64U)];
            GlobalTensor<uint64_t> ptrs;
            ptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(side == 0U ? aarray : barray));
            for (uint32_t q = 0U; q < count; ++q) {
                Duplicate(raw[q * RAW], 0.0F, 8U);
                GlobalTensor<float> input;
                input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ptrs.GetValue(batch + q)));
                DataCopyPad(
                    raw[q * RAW + 8U], input, DataCopyExtParams{1U, split.rows * split.cols * 8U, 0U, 0U, 0U},
                    DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
            }
            if (side == 0U)
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            else
                SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        }
    }
    if constexpr (P <= 32U) {
        for (uint32_t q = 0U; q < count; ++q) {
            Duplicate(stage[q * RAW], 0.0F, 8U);
            Duplicate(stage[2U * COUNT + 64U + q * RAW], 0.0F, 8U);
        }
    }
}

template <uint32_t P>
__aicore__ inline uint32_t CgbtBalancedActiveRows(
    LocalTensor<float> out, const CgbtInterleavedTilingData& t, const CgbtInterleavedTilingData::Operand& split,
    bool aSide, uint32_t len, bool firstGroup)
{
    constexpr uint32_t HALF = P <= 32U ? 2U : 1U;
    uint32_t active = len;
    if constexpr (P == 32U) {
        const uint32_t rows = aSide ? split.cols : ((t.gemm.transMode & CGBT_TRANS_X) ? split.rows : split.cols);
        active = rows * (aSide ? 4U * P : 2U * P);
        if (firstGroup && active < len) {
            for (uint32_t q = 0U; q < HALF; ++q)
                Duplicate(out[q * len + active], 0.0F, len - active);
        }
    }
    return active;
}

template <uint32_t P>
__aicore__ inline void CgbtBalancedPanel(
    CgbtBalancedState& state, const CgbtInterleavedTilingData& t, LocalTensor<float> stage, bool aSide, uint32_t side,
    uint32_t count, uint32_t phase, uint32_t slot, bool firstGroup)
{
    constexpr uint32_t E = P * P;
    constexpr uint32_t HALF = P <= 32U ? 2U : 1U;
    constexpr uint32_t COUNT = E * HALF;
    constexpr uint32_t S = CGBT_SLOT_PLANES * E + CGBT_SLOT_FLAG_FLOATS;
    const auto& split = aSide ? t.a : t.b;
    const uint32_t len = aSide ? 4U * E : 2U * E;
    auto idx = state.indices[aSide ? 0U : 4U * E];
    auto out = state.outputs[aSide ? 0U : 4U * COUNT];
    auto raw = stage[side * (2U * COUNT + 64U)];
    if (firstGroup)
        CgbtCompactIndex<P>(idx, split, aSide, t.gemm.transMode & CGBT_TRANS_X);
    if (aSide)
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(P <= 32U ? phase : 0U));
    else
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(P <= 32U ? phase + 2U : 2U));
    if (aSide || firstGroup)
        PipeBarrier<PIPE_V>();
    // For extent32 each physical row is a whole vector repeat.
    // Padding rows stay zero in UB; do not gather those rows again
    // for every matrix. Logical tails within a row still use the
    // sentinel state.indices, preserving arbitrary rectangular inputs.
    const uint32_t active = CgbtBalancedActiveRows<P>(out, t, split, aSide, len, firstGroup);
    CgbtCompactGather<P>(out, raw, idx, count, len, active);
    if (aSide || (split.negateImag && t.a.negateImag))
        PipeBarrier<PIPE_V>();
    CgbtCompactSigns<P>(out, t, split, aSide, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(
        state.ws[slot * S + (aSide ? 2U * E : 0U)], out,
        DataCopyExtParams{static_cast<uint16_t>(count), len * 4U, 0U, (S - len) * 4U, 0U});
}

template <uint32_t P>
__aicore__ inline void CgbtBalancedRange(
    CgbtBalancedState& state, LocalTensor<float> stage, uint32_t count, uint32_t phase)
{
    constexpr uint32_t E = P * P;
    constexpr uint32_t HALF = P <= 32U ? 2U : 1U;
    constexpr uint32_t COUNT = E * HALF;
    constexpr uint32_t RAW = 2U * E + 8U;
    if (count != 0U) {
        // Check both complex inputs without a horizontal reduction. The
        // scaled copies are scratch; Cube inputs remain unchanged. This
        // conservative threshold keeps a <=64-term FP32 product in range.
        constexpr float riskScale = 3.5e20F;
        for (uint32_t part = 0U; part < 2U * P; part += 64U) {
            const uint32_t width = 2U * P - part < 64U ? 2U * P - part : 64U;
            Muls(
                stage[part], state.outputs[part], riskScale, uint64_t(width), static_cast<uint8_t>(P * count),
                UnaryRepeatParams{1U, 1U, uint8_t(2U * P / 8U), uint8_t(4U * P / 8U)});
        }
        if (state.directB) {
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(P <= 32U ? phase + 2U : 2U));
            auto rawB = stage[2U * COUNT + 64U];
            for (uint32_t q = 0U; q < count; ++q)
                Muls(rawB[q * RAW + 8U], rawB[q * RAW + 8U], riskScale, 2U * E);
        } else {
            Muls(
                stage[2U * COUNT], state.outputs[4U * COUNT], riskScale, uint64_t(64),
                static_cast<uint8_t>(2U * E * count / 64U), UnaryRepeatParams{1, 1, 8, 8});
        }
    }
}

template <uint32_t P>
__aicore__ inline void CgbtBalancedGroup(
    CgbtBalancedState& state, GM_ADDR aarray, GM_ADDR barray, const CgbtInterleavedTilingData& t, uint32_t groupBatch,
    uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P;
    constexpr uint32_t HALF = P <= 32U ? 2U : 1U;
    constexpr uint32_t GROUP = HALF * 2U;
    constexpr uint32_t COUNT = E * HALF;
    constexpr uint32_t INPUT = 4U * COUNT + 128U;
    const uint32_t total = end - groupBatch < GROUP ? end - groupBatch : GROUP;
    const uint32_t first = (total + 1U) / 2U;
    const uint32_t offset = state.sub == 0U ? 0U : first;
    const uint32_t count = state.sub == 0U ? first : total - first;
    const uint32_t batch = groupBatch + offset;
    const uint32_t step = (groupBatch - begin) / GROUP, phase = step % 2U;
    const uint32_t slot = begin + phase * GROUP + offset;
    auto stage = state.inputBase[P <= 32U ? phase * INPUT : 0U];
    CgbtBalancedPrefetchNext<P>(state, aarray, barray, t, groupBatch, total, end, step, phase);

    if (step >= 2U)
        CrossCoreWaitFlag<2, PIPE_MTE2>(1U + 2U * phase);
    if (groupBatch != begin)
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    if (count != 0U) {
        CgbtBalancedLoad<P>(state, aarray, barray, t, stage, batch, count, groupBatch == begin);
        for (uint32_t side = 0U; side < 2U; ++side) {
            const bool aSide = side == 0U;
            if (!aSide && state.directB)
                continue;
            CgbtBalancedPanel<P>(state, t, stage, aSide, side, count, phase, slot, groupBatch == begin);
        }
    }
    if constexpr (P <= 32U) {
        if (count == 0U) {
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(phase));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(phase + 2U));
        }
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(2U * phase);
    CgbtBalancedRange<P>(state, stage, count, phase);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
}

template <uint32_t P>
__aicore__ inline void CgbtCompactBalanced(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, const CgbtInterleavedTilingData& t,
    uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P, HALF = P <= 32U ? 2U : 1U, GROUP = HALF * 2U;
    constexpr uint32_t COUNT = E * HALF;
    constexpr uint32_t INPUT = 4U * COUNT + 128U;
    constexpr uint32_t STORAGE = 6U * E + 6U * COUNT + (P <= 32U ? 2U : 1U) * INPUT;
    static_assert(STORAGE * 4U <= 192U * 1024U, "UB capacity");
    InitSocState();
    (void)asc_get_status();
    GlobalTensor<uint64_t> ap, bp;
    ap.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aarray));
    bp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(barray));
    for (uint32_t batch = begin; batch < end; batch += 8U) {
        DataCachePreload(ap[batch], int16_t(0));
        DataCachePreload(bp[batch], int16_t(0));
    }
    LocalTensor<float> storage(TPosition::VECCALC, 0U, STORAGE);
    auto indices = storage.ReinterpretCast<int32_t>();
    auto outputs = storage[6U * E], inputBase = outputs[6U * COUNT];
    // Each bank holds both raw panels. Finished input banks also hold the
    // scaled range probes; sticky STATUS carries the result across groups.
    const bool directB = !(t.gemm.transMode & CGBT_TRANS_X) && t.gemm.m == P && t.gemm.k == P;
    // The Cube consumes a complete non-transposed B directly. Only its range
    // must be checked here; no B permutation or workspace panel is required.
    const uint32_t sub = GetSubBlockIdx();
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    if constexpr (P <= 32U) {
        const uint32_t total = end - begin < GROUP ? end - begin : GROUP;
        const uint32_t first = (total + 1U) / 2U;
        CgbtCompactPrefetch<P>(
            aarray, barray, t, begin + (sub == 0U ? 0U : first), sub == 0U ? first : total - first, inputBase, 0U);
    }
    CgbtBalancedState state{indices, outputs, inputBase, directB, sub, ws};
    for (uint32_t groupBatch = begin; groupBatch < end; groupBatch += GROUP) {
        CgbtBalancedGroup<P>(state, aarray, barray, t, groupBatch, begin, end);
    }
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    const uint32_t steps = cblas3::CeilDivU(end - begin, GROUP);
    for (uint32_t step = steps > 2U ? steps - 2U : 0U; step < steps; ++step)
        CrossCoreWaitFlag<2, PIPE_MTE2>(1U + 2U * (step % 2U));
    // Each producer checked both inputs for its own batch members. Read
    // sticky STATUS after all probes complete and Cube releases the output.
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const bool ordinary = asc_get_status() == 0;

    if (!ordinary)
        CgbtInterleavedCorrect(aarray, barray, carray, t, begin, end, GROUP, storage);
}

template <uint32_t P>
__aicore__ inline void CgbtDispatchPanelProducer(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgbtInterleavedTilingData& t, uint32_t begin, uint32_t end)
{
    if constexpr (P % 16U != 0U) {
        if constexpr (P <= 48U)
            CgbtCompactBalanced<P>(a, b, c, ws, t, begin, end);
        else
            CgbtInterleavedCompact<P>(a, b, c, ws, t, begin, end);
    } else if (!(t.gemm.transMode & CGBT_TRANS_X))
        CgbtInterleavedDualSplit<P>(a, b, c, ws, t, begin, end);
    else if (P >= 32U && t.a.ld == t.a.rows && t.b.ld == t.b.rows) {
        if constexpr (P < 64U)
            CgbtCompactBalanced<P>(a, b, c, ws, t, begin, end);
        else
            CgbtInterleavedCompact<P>(a, b, c, ws, t, begin, end);
    } else
        CgbtInterleavedSplit<P>(a, b, c, ws, t, begin, end);
}

// The realified N/K axes need multiples of 16 real values, hence an
// 8-complex padding extent. L0A/L0C keep their separate 16-row alignment.
// Each panel size shares batch ownership and pipeline dispatch. Keep the
// calculation out of the entry's frame so one layout does not raise the
// register/stack requirements of the others.
template <uint32_t P>
__aicore__ __attribute__((noinline)) void CgbtPanelBody(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgbtInterleavedTilingData& t)
{
    uint32_t owner;
    if ASCEND_IS_AIC {
        owner = GetBlockIdx();
    } else {
        owner = GetBlockIdx() / 2U;
    }
    // Balanced contiguous ownership keeps every available core busy when
    // batchCount >= workers, with at most one matrix of load imbalance.
    const uint32_t begin = owner * t.gemm.batchBase + (owner < t.gemm.batchExtra ? owner : t.gemm.batchExtra);
    const uint32_t end = begin + t.gemm.batchBase + (owner < t.gemm.batchExtra ? 1U : 0U);
    if (begin >= end)
        return;
    if ASCEND_IS_AIV {
        CgbtDispatchPanelProducer<P>(a, b, c, ws, t, begin, end);
    } else {
        if constexpr (P % 16U != 0U)
            CgbtInterleavedCube<P>(b, c, ws, t, begin, end);
        else if (!(t.gemm.transMode & CGBT_TRANS_X))
            CgbtInterleavedCube<P, true>(b, c, ws, t, begin, end);
        else if (P == 48U && t.a.ld == t.a.rows && t.b.ld == t.b.rows)
            CgbtInterleavedCube<P, true>(b, c, ws, t, begin, end);
        else
            CgbtInterleavedCube<P>(b, c, ws, t, begin, end);
    }
}

// ============================================================================
// Small vector outer products
// ============================================================================
template <uint32_t P>
__aicore__ inline void CgbtOuterLoad(
    LocalTensor<float> dst, GlobalTensor<float> src, uint32_t rows, uint32_t cols, uint32_t ld)
{
    Duplicate(dst, 0.0F, 2U * P * P);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    const uint32_t bytes = rows * 8U, rounded = cblas3::AlignUpU(bytes, 32U);
    DataCopyPad(
        dst, src, DataCopyExtParams{static_cast<uint16_t>(cols), bytes, (ld - rows) * 8U, (P * 8U - rounded) / 32U, 0U},
        DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((rounded - bytes) / 4U), 0.0F});
}

struct CgbtOuterBuffers {
    LocalTensor<float> a, rot, br, bi, out, stage, b;
    LocalTensor<int32_t> ia, ib, ix, tmp;
};

template <uint32_t P>
__aicore__ inline void CgbtOuterIndexA(CgbtOuterBuffers& buffers, const CgbtDirectTilingData& t)
{
    constexpr uint32_t E = P * P;
    constexpr int32_t L = P == 16U ? 4 : 5;
    // A is [K][row][complex component], B is [K][column].
    CgbtCreateIndex(buffers.ia, 2U * E);
    PipeBarrier<PIPE_V>();
    if (t.transA != 0U) {
        ShiftRight(buffers.rot.ReinterpretCast<int32_t>(), buffers.ia, L + 1, 2U * E);
        ShiftRight(buffers.a.ReinterpretCast<int32_t>(), buffers.ia, 1, 2U * E);
        Duplicate(buffers.out.ReinterpretCast<int32_t>(), int32_t(P - 1U), 2U * E);
        PipeBarrier<PIPE_V>();
        And(buffers.a.ReinterpretCast<uint16_t>(), buffers.a.ReinterpretCast<uint16_t>(),
            buffers.out.ReinterpretCast<uint16_t>(), 4U * E);
        Duplicate(buffers.out.ReinterpretCast<int32_t>(), int32_t(1), 2U * E);
        PipeBarrier<PIPE_V>();
        ShiftLeft(buffers.a.ReinterpretCast<int32_t>(), buffers.a.ReinterpretCast<int32_t>(), L, 2U * E);
        And(buffers.out.ReinterpretCast<uint16_t>(), buffers.ia.ReinterpretCast<uint16_t>(),
            buffers.out.ReinterpretCast<uint16_t>(), 4U * E);
        PipeBarrier<PIPE_V>();
        Add(buffers.ia, buffers.a.ReinterpretCast<int32_t>(), buffers.rot.ReinterpretCast<int32_t>(), 2U * E);
        PipeBarrier<PIPE_V>();
        ShiftLeft(buffers.ia, buffers.ia, 1, 2U * E);
        PipeBarrier<PIPE_V>();
        Add(buffers.ia, buffers.ia, buffers.out.ReinterpretCast<int32_t>(), 2U * E);
    }
    PipeBarrier<PIPE_V>();
    ShiftLeft(buffers.ia, buffers.ia, 2, 2U * E);
}

template <uint32_t P>
__aicore__ inline void CgbtOuterIndexB(CgbtOuterBuffers& buffers, const CgbtDirectTilingData& t)
{
    constexpr uint32_t E = P * P;
    constexpr int32_t L = P == 16U ? 4 : 5;
    CgbtCreateIndex(buffers.ib, E);
    PipeBarrier<PIPE_V>();
    if (t.transB == 0U) {
        ShiftRight(buffers.tmp, buffers.ib, L, E);
        Duplicate(buffers.ix, int32_t(P - 1U), E);
        PipeBarrier<PIPE_V>();
        And(buffers.ib.ReinterpretCast<uint16_t>(), buffers.ib.ReinterpretCast<uint16_t>(),
            buffers.ix.ReinterpretCast<uint16_t>(), 2U * E);
        PipeBarrier<PIPE_V>();
        ShiftLeft(buffers.ib, buffers.ib, L, E);
        PipeBarrier<PIPE_V>();
        Add(buffers.ib, buffers.ib, buffers.tmp, E);
    }
    PipeBarrier<PIPE_V>();
    ShiftLeft(buffers.ib, buffers.ib, 3, E);
    PipeBarrier<PIPE_ALL>();
    CgbtCreateIndex(buffers.ix, 2U * E);
    PipeBarrier<PIPE_V>();
    uint64_t odd[2] = {0xAAAAAAAAAAAAAAAAULL, 0U}, even[2] = {0x5555555555555555ULL, 0U};
    Adds(buffers.ix, buffers.ix, int32_t(-1), odd, 2U * E / 64U, UnaryRepeatParams{1, 1, 8, 8});
    Adds(buffers.ix, buffers.ix, int32_t(1), even, 2U * E / 64U, UnaryRepeatParams{1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    ShiftLeft(buffers.ix, buffers.ix, 2, 2U * E);
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P>
__aicore__ inline void CgbtOuterMultiply(CgbtOuterBuffers& buffers, const CgbtDirectTilingData& t)
{
    constexpr uint32_t E = P * P;

    uint64_t odd[2] = {0xAAAAAAAAAAAAAAAAULL, 0U}, even[2] = {0x5555555555555555ULL, 0U};
    Gather(buffers.a, buffers.stage, buffers.ia.ReinterpretCast<uint32_t>(), 0U, 2U * E);
    Gather(buffers.rot, buffers.b, buffers.ib.ReinterpretCast<uint32_t>(), 0U, E);
    Gather(buffers.rot[E], buffers.b, buffers.ib.ReinterpretCast<uint32_t>(), 4U, E);
    PipeBarrier<PIPE_V>();
    if (t.transA == 2U)
        Muls(buffers.a, buffers.a, -1.0F, odd, 2U * E / 64U, UnaryRepeatParams{1, 1, 8, 8});
    if (t.transB == 2U)
        Muls(buffers.rot[E], buffers.rot[E], -1.0F, E);
    PipeBarrier<PIPE_V>();
    Brcb(buffers.br, buffers.rot, E / 8U, {1, 8});
    Brcb(buffers.bi, buffers.rot[E], E / 8U, {1, 8});
    PipeBarrier<PIPE_V>();
    Gather(buffers.rot, buffers.a, buffers.ix.ReinterpretCast<uint32_t>(), 0U, 2U * E);
    Duplicate(buffers.out, 0.0F, 2U * E);
    PipeBarrier<PIPE_V>();
    Muls(buffers.rot, buffers.rot, -1.0F, even, 2U * E / 64U, UnaryRepeatParams{1, 1, 8, 8});
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams outer{1, 1, 0, 2U * P / 8U, 0, 1};
    for (uint32_t k = 0; k < t.k; ++k) {
        MulAddDst(
            buffers.out, buffers.a[k * 2U * P], buffers.br[k * P * 8U], uint64_t(2U * P), static_cast<uint8_t>(t.n),
            outer);
        PipeBarrier<PIPE_V>();
        MulAddDst(
            buffers.out, buffers.rot[k * 2U * P], buffers.bi[k * P * 8U], uint64_t(2U * P), static_cast<uint8_t>(t.n),
            outer);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t P>
__aicore__ inline void CgbtOuterBatch(
    CgbtOuterBuffers& buffers, GM_ADDR aa, GM_ADDR ba, GM_ADDR ca, const CgbtDirectTilingData& t, uint32_t q)
{
    constexpr uint32_t E = P * P;

    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(aa, q, t.batchCount)));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(ba, q, t.batchCount)));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(ca, q, t.batchCount)));
    CgbtOuterLoad<P>(buffers.stage, ag, t.transA == 0U ? t.m : t.k, t.transA == 0U ? t.k : t.m, t.lda);
    CgbtOuterLoad<P>(buffers.b, bg, t.transB == 0U ? t.k : t.n, t.transB == 0U ? t.n : t.k, t.ldb);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Abs(buffers.br, buffers.stage, 4U * E);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(buffers.bi, buffers.br, 64, 4U * E / 64U, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(buffers.bi[64], buffers.bi, 4U * E / 64U, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_ALL>();
    if (!(buffers.bi.GetValue(64) <= CGBT_SAFE_COMPONENT_MAGNITUDE)) {
        for (uint32_t col = 0; col < t.n; ++col)
            for (uint32_t row = 0; row < t.m; ++row) {
                auto value = CgbtDirectValue(t, ag, bg, cg, row, col);
                buffers.out.SetValue(2U * (col * P + row), value.real);
                buffers.out.SetValue(2U * (col * P + row) + 1U, value.imag);
            }
    } else {
        CgbtOuterMultiply<P>(buffers, t);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(
        cg, buffers.out,
        DataCopyExtParams{
            static_cast<uint16_t>(t.n), t.m * 8U, (P * 8U - cblas3::AlignUpU(t.m * 8U, 32U)) / 32U, (t.ldc - t.m) * 8U,
            0U});
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P>
__aicore__ inline void CgbtOuterBody(GM_ADDR aa, GM_ADDR ba, GM_ADDR ca, const CgbtDirectTilingData& t)
{
    constexpr uint32_t E = P * P;

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 32U * E * 4U);
    auto all = buf.Get<float>();
    auto a = all, rot = all[2U * E], br = all[4U * E], bi = all[12U * E];
    auto out = all[20U * E], stage = all[22U * E], b = all[24U * E];
    auto ia = all[26U * E].ReinterpretCast<int32_t>(), ib = all[28U * E].ReinterpretCast<int32_t>();
    auto ix = all[29U * E].ReinterpretCast<int32_t>(), tmp = all[31U * E].ReinterpretCast<int32_t>();
    CgbtOuterBuffers buffers{a, rot, br, bi, out, stage, b, ia, ib, ix, tmp};
    CgbtOuterIndexA<P>(buffers, t);
    CgbtOuterIndexB<P>(buffers, t);
    const uint32_t workers = GetBlockNum(), owner = GetBlockIdx();
    const uint32_t base = t.batchCount / workers, extra = t.batchCount % workers;
    const uint32_t begin = owner * base + (owner < extra ? owner : extra),
                   end = begin + base + (owner < extra ? 1U : 0U);
    for (uint32_t q = begin; q < end; ++q) {
        CgbtOuterBatch<P>(buffers, aa, ba, ca, t, q);
    }
}

} // namespace

// One entry for the panel pipeline. The physical extent selects the UB/L1
// layout; each specialization retains its compile-time storage bounds.
extern "C" __global__ __aicore__ __schedmode__(1) void cgemm_batched_panel_size_kernel(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, CgbtInterleavedTilingData tiling, uint32_t panel)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    switch (panel) {
        case 16U:
            CgbtPanelBody<16>(aarray, barray, carray, workspace, tiling);
            break;
        case 24U:
            CgbtPanelBody<24>(aarray, barray, carray, workspace, tiling);
            break;
        case 32U:
            CgbtPanelBody<32>(aarray, barray, carray, workspace, tiling);
            break;
        case 40U:
            CgbtPanelBody<40>(aarray, barray, carray, workspace, tiling);
            break;
        case 48U:
            CgbtPanelBody<48>(aarray, barray, carray, workspace, tiling);
            break;
        case 56U:
            CgbtPanelBody<56>(aarray, barray, carray, workspace, tiling);
            break;
        default:
            CgbtPanelBody<64>(aarray, barray, carray, workspace, tiling);
            break;
    }
}

// One definition, two compiled storage plans. A runtime branch joins their
// register/stack requirements and slows the grouped calculation.
template <bool BATCHED>
__global__ __aicore__ void cgemm_batched_vector16_kernel(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, CgbtDirectTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (BATCHED)
        CgbtTinyBody<16, 8>(aarray, barray, carray, t);
    else
        CgbtOuterBody<16>(aarray, barray, carray, t);
    // Restore the public vector register state before returning to runtime.
    ResetMask();
}

// Both launchers resolve one runtime-owned reference per handle. Arguments
// are copied by the runtime; they never outlive the call or alias another stream.
static int CgbtResolveFunction(const void* symbol, void*& function)
{
    if (function != nullptr)
        return ACL_SUCCESS;
    return aclrtGetFuncBySymbol(symbol, &function);
}

int cgemm_batched_vector16_kernel_do(
    uint32_t vectors, void* stream, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, CgbtDirectTilingData direct,
    void*& singleFunction, void*& groupedFunction)
{
    const bool batched = direct.batchCount > 2U * vectors;
    void*& function = batched ? groupedFunction : singleFunction;
    const void* symbol = batched ? reinterpret_cast<const void*>(&cgemm_batched_vector16_kernel<true>) :
                                   reinterpret_cast<const void*>(&cgemm_batched_vector16_kernel<false>);
    const int error = CgbtResolveFunction(symbol, function);
    if (error != ACL_SUCCESS)
        return error;
    struct Arguments {
        GM_ADDR a;
        GM_ADDR b;
        GM_ADDR c;
        CgbtDirectTilingData tiling;
    } args{aarray, barray, carray, direct};
    static_assert(offsetof(Arguments, tiling) == 3U * sizeof(uint8_t*));
    return aclrtLaunchKernelWithHostArgs(function, vectors, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}

static int CgbtLaunchInterleaved(
    uint32_t blockDim, void* stream, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace,
    const CgbtInterleavedTilingData& tiling, void*& function)
{
    const uint32_t extent = tiling.gemm.m > tiling.gemm.n ? tiling.gemm.m : tiling.gemm.n;
    const uint32_t largest = extent > tiling.gemm.k ? extent : tiling.gemm.k;
    const bool contiguous = tiling.a.ld == tiling.a.rows && tiling.b.ld == tiling.b.rows;
    // Contiguous inputs can use the additional 24/40/56 layouts. Strided
    // inputs use the 16/32/48/64 interleaved layouts.
    const uint32_t alignment = contiguous ? 8U : 16U;
    const uint32_t panel = largest <= 16U ? 16U : (largest + alignment - 1U) / alignment * alignment;
    const int error = CgbtResolveFunction(reinterpret_cast<const void*>(&cgemm_batched_panel_size_kernel), function);
    if (error != ACL_SUCCESS)
        return error;
    struct Arguments {
        GM_ADDR a;
        GM_ADDR b;
        GM_ADDR c;
        GM_ADDR workspace;
        CgbtInterleavedTilingData tiling;
        uint32_t panel;
    } args{aarray, barray, carray, workspace, tiling, panel};
    static_assert(offsetof(Arguments, tiling) == 4U * sizeof(uint8_t*));
    static_assert(offsetof(Arguments, panel) == 4U * sizeof(uint8_t*) + sizeof(CgbtInterleavedTilingData));
    return aclrtLaunchKernelWithHostArgs(function, blockDim, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}

int cgemm_batched_small_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, CgbtDirectTilingData t,
    void*& function)
{
    if (blockDim == 0U)
        return ACL_ERROR_INVALID_PARAM;
    const CgbtInterleavedTilingData tiling{
        {t.transA ? t.k : t.m, t.transA ? t.m : t.k, t.lda, t.transA == 2U},
        {t.transB ? t.n : t.k, t.transB ? t.k : t.n, t.ldb, t.transB == 2U},
        {t.n, t.m, t.k, (t.transA ? CGBT_TRANS_Y : 0U) | (t.transB ? CGBT_TRANS_X : 0U), t.batchCount,
         t.batchCount / blockDim, t.batchCount % blockDim},
        {t.m, t.n, t.ldc}};
    return CgbtLaunchInterleaved(blockDim, stream, a, b, c, workspace, tiling, function);
}
