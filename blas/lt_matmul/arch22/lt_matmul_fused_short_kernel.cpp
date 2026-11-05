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
#include "lib/matmul_intf.h"
#include "lt_matmul_tiling.h"
using namespace AscendC;
using namespace matmul;
namespace {
template <HardEvent E>
__aicore__ inline void SyncPack()
{
    auto id = GetTPipePtr()->FetchEventID<E>();
    SetFlag<E>(id);
    WaitFlag<E>(id);
}
__aicore__ inline uint32_t MinPack(uint32_t x, uint32_t y) { return x < y ? x : y; }
template <uint32_t R, uint32_t C>
__aicore__ inline void Tile(
    const GlobalTensor<float>& src, const GlobalTensor<float>& dst, const LocalTensor<float>& in,
    const LocalTensor<float>& out, uint32_t job, uint32_t rows, uint32_t cols, uint32_t ld)
{
    const uint32_t nc = (cols + C - 1U) / C, row = job / nc * R, col = job % nc * C;
    const uint32_t vr = MinPack(R, rows - row), vc = MinPack(C, cols - col), oc = (vc + 15U) / 16U * 16U, ip = oc + 8U;
    Duplicate(in, 0.0F, R * ip);
    SyncPack<HardEvent::V_MTE2>();
    const uint32_t ac = (vc + 7U) / 8U * 8U;
    const DataCopyExtParams read{static_cast<uint16_t>(vr), vc * 4U, (ld - vc) * 4U, (ip - ac) / 8U, 0U};
    const DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>(ac - vc), 0.0F};
    DataCopyPad(in, src[uint64_t(row) * ld + col], read, pad);
    SyncPack<HardEvent::MTE2_V>();
    const CopyRepeatParams rearrange{1U, static_cast<uint16_t>(ip / 8U), R, 1U};
    for (uint32_t rr = 0; rr < R; rr += 8U)
        Copy(out[rr * 8U], in[rr * ip], uint64_t(64U), static_cast<uint8_t>(oc / 8U), rearrange);
    SyncPack<HardEvent::V_MTE3>();
    const uint32_t pr = (rows + 15U) / 16U * 16U, wr = MinPack(R, pr - row);
    const DataCopyExtParams write{static_cast<uint16_t>(oc / 8U), wr * 8U * 4U, R - wr, (pr - wr) * 8U * 4U, 0U};
    DataCopyPad(dst[uint64_t(col / 8U) * pr * 8U + uint64_t(row) * 8U], out, write);
    SyncPack<HardEvent::MTE3_V>();
}
} // namespace

namespace {
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
using Input = MatmulType<TPosition::GM, CubeFormat::NZ, float, true>;
using Output = MatmulType<TPosition::GM, CubeFormat::NZ, float>;
constexpr MatmulFuncParams F{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                             true,  true};
constexpr MatmulBiasParams B{false};
template <uint32_t N>
constexpr MatmulShapeParams SHAPE{128U, 1440U, 32U, 32U, N, N >= 480U ? 16U : 32U};
template <uint32_t N>
__aicore__ constexpr auto MakeCfg()
{
    auto c = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(SHAPE<N>, F, B);
    c.enUnitFlag = false;
    return c;
}
template <uint32_t N>
constexpr auto CFG = MakeCfg<N>();
template <uint32_t N>
__aicore__ constexpr auto MakeT()
{
    auto t = GetMatmulApiTiling<Input, Input, Output, Output>(CFG<N>);
    t.dbL0C = 2;
    t.shareL0CSize = t.baseM * t.baseN * 4U * 2U;
    return t;
}
template <uint32_t N>
constexpr auto T = MakeT<N>();
template <uint32_t N>
__aicore__ inline void Produce(GM_ADDR a, GM_ADDR b, GM_ADDR scratch, const LtMatmul22Tiling& t, TPipe& pipe)
{
    const uint32_t chunks = (t.m + 15U) / 16U, row = chunks * GetBlockIdx() / GetBlockNum() * 16U;
    const uint32_t next = chunks * (GetBlockIdx() + 1U) / GetBlockNum() * 16U, rp = next - row,
                   rows = Min(t.m, next) - row, np = (t.n + 15U) / 16U * 16U, kp = (t.k + 15U) / 16U * 16U;
    MatmulImpl<Input, Input, Output, Output, T<N>> mm;
    mm.SetSubBlockIdx(0U);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> ag, bg, rg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    rg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scratch));
    // NZ inputs allow logical A rows to be rebased to this band's dense output
    // stride. A's physical NZ height is K; B's physical height stays global N.
    mm.SetOrgShape(rp, np, kp, kp, np);
    CrossCoreWaitFlag<2>(11U);
    for (uint32_t col = 0; col < np; col += N) {
        const uint32_t width = Min(N, np - col);
        mm.SetSingleShape(rp, width, kp);
        mm.SetTensorA(ag[uint64_t(row) * kp], true);
        mm.SetTensorB(bg[uint64_t(col) * 8U], true);
        mm.IterateAll(rg[uint64_t(row) * np + uint64_t(col) * rp], false);
        mm.End();
        CrossCoreSetFlag<2, PIPE_FIX>(9U);
    }
}
__aicore__ inline void BuildDenseBOffsets(const LocalTensor<float>& work, uint32_t k)
{
    const auto linear = work[4096U].ReinterpretCast<int32_t>();
    const auto rows = work[4608U].ReinterpretCast<int32_t>();
    const auto offsets = work[5120U].ReinterpretCast<int32_t>();
    const auto columns = work[5632U].ReinterpretCast<int32_t>();
    CreateVecIndex(linear, 0, 512U);
    PipeBarrier<PIPE_V>();
    ShiftRight(rows, linear, int32_t(3), 512U);
    PipeBarrier<PIPE_V>();
    // 4 * ((i / 8) * k + i % 8) = 4 * i + (4 * k - 32) * (i / 8).
    Muls(offsets, rows, static_cast<int32_t>(k) * 4 - 32, 512U);
    PipeBarrier<PIPE_V>();
    Muls(columns, linear, int32_t(4), 512U);
    PipeBarrier<PIPE_V>();
    Add(offsets, offsets, columns, 512U);
    PipeBarrier<PIPE_V>();
}

// Tight B rows form one contiguous input range. Gather constructs FP32 NZ C0=8
// directly in UB; explicit zero lanes retain the same physical padding as Tile.
__aicore__ inline void PackDenseB(
    const GlobalTensor<float>& src, const GlobalTensor<float>& dst, const LocalTensor<float>& work, uint32_t job,
    const LtMatmul22Tiling& t, bool& offsetsReady)
{
    const uint32_t row = job * 64U, rows = Min(64U, t.n - row);
    const uint32_t cols = (t.k + 15U) / 16U * 16U, pr = (t.n + 15U) / 16U * 16U;
    const auto raw = work, out = work[2048U];
    const auto offsets = work[5120U].ReinterpretCast<uint32_t>();
    // Earlier jobs finish MTE3_V before reusing this UB. Keep the raw-read
    // fence, then overlap the disjoint raw DMA and vector offset construction.
    SyncPack<HardEvent::V_MTE2>();
    const DataCopyExtParams read{1U, rows * t.k * 4U, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
    DataCopyPad(raw, src[uint64_t(row) * t.k], read, pad);
    if (!offsetsReady) {
        BuildDenseBOffsets(work, t.k);
        offsetsReady = true;
    }
    // Four gathers overwrite all KP=32 lanes for a full row tile. Tail rows
    // and skipped K slabs still need the original explicit initialization.
    if (rows != 64U || t.k <= 24U) {
        Duplicate(out, 0.0F, 64U * cols);
        PipeBarrier<PIPE_V>();
    }
    SyncPack<HardEvent::MTE2_V>();
    for (uint32_t slab = 0U; slab < cols / 8U; ++slab) {
        const uint32_t first = slab * 8U;
        if (first >= t.k)
            continue;
        // Slabs write disjoint output ranges and share read-only inputs.
        Gather(out[slab * 512U], raw, offsets, first * 4U, rows * 8U);
        const uint32_t valid = Min(8U, t.k - first);
        if (valid != 8U) {
            PipeBarrier<PIPE_V>(); // The zero mask overwrites this gather's padding lanes.
            uint64_t zeroMask[2] = {0x0101010101010101ULL * (255U ^ ((1U << valid) - 1U)), 0U};
            Duplicate(out[slab * 512U], 0.0F, zeroMask, uint8_t(8U), 1U, 8U);
            PipeBarrier<PIPE_V>();
        }
    }
    SyncPack<HardEvent::V_MTE3>();
    const uint32_t wr = Min(64U, pr - row);
    const DataCopyExtParams write{static_cast<uint16_t>(cols / 8U), wr * 8U * 4U, 64U - wr, (pr - wr) * 8U * 4U, 0U};
    DataCopyPad(dst[uint64_t(row) * 8U], out, write);
    SyncPack<HardEvent::MTE3_V>();
}

__aicore__ inline void GlobalPack(
    GM_ADDR a, GM_ADDR b, GM_ADDR ap, GM_ADDR bp, const LtMatmul22Tiling& t, const LocalTensor<float>& work)
{
    const auto in = work, out = work[16U * 264U];
    GlobalTensor<float> ag, bg, az, bz;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    az.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ap));
    bz.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bp));
    const uint32_t aj = ((t.k + 15U) / 16U) * ((t.m + 255U) / 256U), bj = ((t.n + 63U) / 64U) * ((t.k + 31U) / 32U);
    bool denseBOffsetsReady = false;
    for (uint32_t job = GetBlockIdx(); job < aj + bj; job += GetBlockNum() * 2U) {
        if (job < aj)
            Tile<16U, 256U>(ag, az, in, out, job, t.k, t.m, t.lda);
        else if (t.ldb == t.k)
            PackDenseB(bg, bz, work, job - aj, t, denseBOffsetsReady);
        else
            Tile<64U, 32U>(bg, bz, in, out, job - aj, t.n, t.k, t.ldb);
    }
    // Batch scheduling reserves all participants. Mode0 is AIV-only here;
    // mode2 then makes the complete packed operands visible to each AIC.
    CrossCoreSetFlag<0, PIPE_MTE3>(10U);
    CrossCoreWaitFlag<0>(10U);
    CrossCoreSetFlag<2, PIPE_MTE3>(11U);
}
struct Events {
    TEventID inReady[2], inFree[2], outReady[2], outFree[2];
};
__aicore__ inline void InitEvents(TPipe& pipe, Events& e)
{
    for (uint32_t i = 0; i < 2U; ++i) {
        e.inReady[i] = pipe.AllocEventID<HardEvent::MTE2_V>();
        e.inFree[i] = pipe.AllocEventID<HardEvent::V_MTE2>();
        e.outReady[i] = pipe.AllocEventID<HardEvent::V_MTE3>();
        e.outFree[i] = pipe.AllocEventID<HardEvent::MTE3_V>();
        SetFlag<HardEvent::V_MTE2>(e.inFree[i]);
        SetFlag<HardEvent::MTE3_V>(e.outFree[i]);
    }
}
__aicore__ inline void FreeEvents(TPipe& pipe, Events& e)
{
    for (uint32_t i = 0; i < 2U; ++i) {
        WaitFlag<HardEvent::V_MTE2>(e.inFree[i]);
        WaitFlag<HardEvent::MTE3_V>(e.outFree[i]);
    }
    SyncPack<HardEvent::MTE3_S>();
    for (uint32_t i = 0; i < 2U; ++i) {
        pipe.ReleaseEventID<HardEvent::MTE2_V>(e.inReady[i]);
        pipe.ReleaseEventID<HardEvent::V_MTE2>(e.inFree[i]);
        pipe.ReleaseEventID<HardEvent::V_MTE3>(e.outReady[i]);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(e.outFree[i]);
    }
}
struct ConsumerState {
    uint32_t row, first, rp, rows, np;
    GlobalTensor<float> scratch, output;
    Events events;
};
__aicore__ inline bool InitConsumer(
    GM_ADDR scratch, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe, ConsumerState& state)
{
    const uint32_t group = GetBlockIdx() / 2U, sub = GetBlockIdx() % 2U, chunks = (t.m + 15U) / 16U;
    const uint32_t row = chunks * group / GetBlockNum() * 16U, next = chunks * (group + 1U) / GetBlockNum() * 16U;
    const uint32_t actual = Min(next, t.m) - row, own = (actual + 7U) / 8U;
    const uint32_t first = own * sub / 2U * 8U, end = Min(own * (sub + 1U) / 2U * 8U, actual);
    state.row = row;
    state.first = first;
    state.rp = next - row;
    state.rows = first < end ? end - first : 0U;
    state.np = (t.n + 15U) / 16U * 16U;
    state.scratch.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scratch));
    state.output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    if (!state.rows) {
        for (uint32_t col = 0; col < t.n; col += t.cubeTileN)
            CrossCoreWaitFlag<2>(9U);
        return false;
    }
    InitEvents(pipe, state.events);
    return true;
}
template <uint32_t MAX_R, uint32_t PANEL>
__aicore__ inline void ReadPanel(
    const LocalTensor<float>& raw, const GlobalTensor<float>& rg, uint64_t base, uint32_t rows, uint32_t rp,
    uint32_t width, const Events& e, uint32_t slot)
{
    WaitFlag<HardEvent::V_MTE2>(e.inFree[slot]);
    const DataCopyExtParams in{
        static_cast<uint16_t>((width + 15U) / 16U), rows * 16U * 4U, (rp - rows) * 16U * 4U, 0U, 0U};
    const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
    DataCopyPad(raw, rg[base], in, pad);
    SetFlag<HardEvent::MTE2_V>(e.inReady[slot]);
}
template <uint32_t MAX_R, uint32_t PANEL>
__aicore__ inline void Consume(
    GM_ADDR scratch, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe, const LocalTensor<float>& work)
{
    constexpr uint32_t PITCH = PANEL + 8U, RAW = MAX_R * PANEL, OUT = MAX_R * PITCH;
    static_assert((RAW * 2U + OUT * 2U) * 4U <= 135168U, "UB phase-alias bound");
    ConsumerState state;
    if (!InitConsumer(scratch, d, t, pipe, state))
        return;
    const uint32_t row = state.row, first = state.first, rp = state.rp, rows = state.rows, np = state.np;
    const auto& rg = state.scratch;
    const auto& dg = state.output;
    auto& events = state.events;
    uint32_t iter = 0U;
    const uint64_t band = uint64_t(row) * np + uint64_t(first) * 16U;
    for (uint32_t stage = 0; stage < t.n; stage += t.cubeTileN) {
        CrossCoreWaitFlag<2>(9U);
        const uint32_t sw = Min(t.cubeTileN, t.n - stage), tiles = (sw + PANEL - 1U) / PANEL;
        ReadPanel<MAX_R, PANEL>(
            work[(iter % 2U) * RAW], rg, band + uint64_t(stage) * rp, rows, rp, Min(PANEL, sw), events, iter % 2U);
        for (uint32_t it = 0; it < tiles; ++it, ++iter) {
            if (it + 1U < tiles) {
                const uint32_t nc = stage + (it + 1U) * PANEL, ns = (iter + 1U) % 2U;
                ReadPanel<MAX_R, PANEL>(
                    work[ns * RAW], rg, band + uint64_t(nc) * rp, rows, rp, Min(PANEL, sw - (it + 1U) * PANEL), events,
                    ns);
            }
            const uint32_t slot = iter % 2U, col = stage + it * PANEL, width = Min(PANEL, sw - it * PANEL),
                           slabs = (width + 15U) / 16U;
            const auto raw = work[slot * RAW], nd = work[RAW * 2U + slot * OUT];
            WaitFlag<HardEvent::MTE2_V>(events.inReady[slot]);
            WaitFlag<HardEvent::MTE3_V>(events.outFree[slot]);
            const CopyRepeatParams cp{PITCH / 8U, 2U, 2U, static_cast<uint16_t>(rows * 2U)};
            for (uint32_t rr = 0; rr < rows; rr += 8U) {
                const uint64_t mask = Min(8U, rows - rr) * 8U;
                Copy(nd[rr * PITCH], raw[rr * 16U], mask, static_cast<uint8_t>(slabs), cp);
                Copy(nd[rr * PITCH + 8U], raw[rr * 16U + 8U], mask, static_cast<uint8_t>(slabs), cp);
            }
            SetFlag<HardEvent::V_MTE2>(events.inFree[slot]);
            SetFlag<HardEvent::V_MTE3>(events.outReady[slot]);
            WaitFlag<HardEvent::V_MTE3>(events.outReady[slot]);
            const DataCopyExtParams out{
                static_cast<uint16_t>(rows), width * 4U, (PITCH - (width + 7U) / 8U * 8U) / 8U,
                (t.productLd - width) * 4U, 0U};
            DataCopyPad(dg[uint64_t(row + first) * t.productLd + col], nd, out);
            SetFlag<HardEvent::MTE3_V>(events.outFree[slot]);
        }
    }
    FreeEvents(pipe, events);
}

template <uint32_t MAX_R, uint32_t PANEL>
__aicore__ inline void ConsumeWide(
    GM_ADDR scratch, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe, const LocalTensor<float>& work)
{
    constexpr uint32_t PITCH = PANEL + 8U, RAW = MAX_R * 2U * PANEL, OUT = MAX_R * PITCH;
    static_assert((RAW + OUT) * 4U <= 191440U, "UB full-band phase bound");
    ConsumerState state;
    if (!InitConsumer(scratch, d, t, pipe, state))
        return;
    const uint32_t row = state.row, first = state.first, rp = state.rp, rows = state.rows, np = state.np;
    const auto& rg = state.scratch;
    const auto& dg = state.output;
    auto& events = state.events;
    const auto raw = work, nd = work[RAW];
    for (uint32_t stage = 0; stage < t.n; stage += t.cubeTileN) {
        CrossCoreWaitFlag<2>(9U);
        const uint32_t width = Min(t.cubeTileN, t.n - stage), slabs = (width + 15U) / 16U;
        WaitFlag<HardEvent::V_MTE2>(events.inFree[0]);
        const DataCopyExtParams in{1U, slabs * rp * 16U * 4U, 0U, 0U, 0U};
        const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        DataCopyPad(raw, rg[uint64_t(row) * np + uint64_t(stage) * rp], in, pad);
        SetFlag<HardEvent::MTE2_V>(events.inReady[0]);
        WaitFlag<HardEvent::MTE2_V>(events.inReady[0]);
        WaitFlag<HardEvent::MTE3_V>(events.outFree[0]);
        const CopyRepeatParams cp{PITCH / 8U, 2U, 2U, static_cast<uint16_t>(rp * 2U)};
        for (uint32_t rr = 0; rr < rows; rr += 8U) {
            const uint64_t mask = Min(8U, rows - rr) * 8U;
            Copy(nd[rr * PITCH], raw[(first + rr) * 16U], mask, static_cast<uint8_t>(slabs), cp);
            Copy(nd[rr * PITCH + 8U], raw[(first + rr) * 16U + 8U], mask, static_cast<uint8_t>(slabs), cp);
        }
        SetFlag<HardEvent::V_MTE2>(events.inFree[0]);
        SetFlag<HardEvent::V_MTE3>(events.outReady[0]);
        WaitFlag<HardEvent::V_MTE3>(events.outReady[0]);
        const DataCopyExtParams out{
            static_cast<uint16_t>(rows), width * 4U, (PITCH - (width + 7U) / 8U * 8U) / 8U, (t.productLd - width) * 4U,
            0U};
        DataCopyPad(dg[uint64_t(row + first) * t.productLd + stage], nd, out);
        SetFlag<HardEvent::MTE3_V>(events.outFree[0]);
    }
    FreeEvents(pipe, events);
}

} // namespace
extern "C" __global__ __aicore__ __schedmode__(1) void lt_matmul22_followup_fused_stream(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, GM_ADDR d, LtMatmul22Tiling t, uint64_t syncAddress)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // Initialize explicitly when taking ownership of the FFTS register lifecycle.
    SetSyncBaseAddr(syncAddress);
    auto* plane = reinterpret_cast<GM_ADDR>((reinterpret_cast<uint64_t>(workspace) + 511U) & ~uint64_t(511U));
    const uint64_t ab = uint64_t((t.m + 15U) / 16U * 16U) * ((t.k + 15U) / 16U * 16U) * 4U,
                   bb = uint64_t((t.n + 15U) / 16U * 16U) * ((t.k + 15U) / 16U * 16U) * 4U;
    if ASCEND_IS_AIC {
        InitSocState();
        TPipe pipe;
        if (t.cubeTileN == 480U)
            Produce<480U>(plane, plane + ab, plane + ab + bb, t, pipe);
        else
            Produce<256U>(plane, plane + ab, plane + ab + bb, t, pipe);
    } else {
        TPipe pipe;
        TBuf<TPosition::VECCALC> storage;
        pipe.InitBuffer(storage, 192512U);
        auto work = storage.Get<float>();
        GlobalPack(a, b, plane, plane + ab, t, work);
        const uint32_t maxRows = (((t.m + 15U) / 16U + GetBlockNum() - 1U) / GetBlockNum()) * 16U;
        if (maxRows <= 64U)
            ConsumeWide<32U, 480U>(plane + ab + bb, d, t, pipe, work);
        else
            Consume<64U, 128U>(plane + ab + bb, d, t, pipe, work);
        // Packing and output Copy calls use partial FP32 masks on the AIVs.
        ResetMask();
    }
    PipeBarrier<PIPE_ALL>();
    SetSyncBaseAddr(0U);
}
void LtMatmul22FusedShort(
    uint32_t groups, uint32_t, void* stream, uint8_t* a, uint8_t* b, uint8_t* d, uint8_t* workspace, LtMatmul22Tiling t,
    uint64_t syncAddress)
{
    lt_matmul22_followup_fused_stream<<<groups, nullptr, stream>>>(a, b, workspace, d, t, syncAddress);
}
