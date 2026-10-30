/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "lib/matmul_intf.h"
#include "sgemm_gb_cube_packet.h"
using namespace AscendC;
namespace {
__aicore__ inline uint64_t MatrixA(const sgemm_gb::CubeSinglePacket& p, uint32_t) { return p.matrices.a[0]; }
__aicore__ inline uint64_t MatrixB(const sgemm_gb::CubeSinglePacket& p, uint32_t) { return p.matrices.b[0]; }
__aicore__ inline uint64_t MatrixC(const sgemm_gb::CubeSinglePacket& p, uint32_t) { return p.matrices.c[0]; }
__aicore__ inline uint64_t MatrixA(const sgemm_gb::CubePacket& p, uint32_t batch) { return p.matrices.a[batch]; }
__aicore__ inline uint64_t MatrixB(const sgemm_gb::CubePacket& p, uint32_t batch) { return p.matrices.b[batch]; }
__aicore__ inline uint64_t MatrixC(const sgemm_gb::CubePacket& p, uint32_t batch) { return p.matrices.c[batch]; }
__aicore__ inline uint64_t MatrixA(const sgemm_gb::CubeOffsetPacket& p, uint32_t batch)
{
    return p.matrices.baseA + p.matrices.a[batch];
}
__aicore__ inline uint64_t MatrixB(const sgemm_gb::CubeOffsetPacket& p, uint32_t batch)
{
    return p.matrices.baseB + p.matrices.b[batch];
}
__aicore__ inline uint64_t MatrixC(const sgemm_gb::CubeOffsetPacket& p, uint32_t batch)
{
    return p.matrices.baseC + p.matrices.c[batch];
}
template <class P>
__aicore__ inline void LoadCubeTiling(const P& p, TCubeTiling& tc)
{
    static_assert(sizeof(tc) <= sizeof(p.tiling), "TCubeTiling wire format too small");
    static_assert(sizeof(tc) % sizeof(uint32_t) == 0, "TCubeTiling is not word-sized");
    auto* bytes = reinterpret_cast<uint8_t*>(&tc);
    const auto* source = reinterpret_cast<const uint8_t*>(p.tiling);
    for (uint32_t i = 0; i < sizeof(tc); ++i)
        bytes[i] = source[i];
}
template <class P, bool TransLeft, bool TransRight, bool Single = false>
__aicore__ inline void CubeRun(const P& p)
{
    using Left = MatmulType<TPosition::GM, CubeFormat::ND, float, TransLeft>;
    using Right = MatmulType<TPosition::GM, CubeFormat::ND, float, TransRight>;
    using Output = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using Bias = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<Left, Right, Output, Bias> mm;
    TPipe pipe;
    TCubeTiling tc;
    LoadCubeTiling(p, tc);
    // SDK alignment point C2: pure-AIC single-core Matmul Init, no mixed-core GM workspace.
    mm.Init(&tc, &pipe);
    // SDK alignment point C3: assert the exact FP32/HF32 control API in the local SDK.
    // Intentionally explicit: never silently lower input precision to HF32/FP16 to pass timing.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    const auto& g = p.matrices.g;
    mm.SetOrgShape(g.n, g.m, g.k);
    const uint32_t tileN = p.tileN;
    const uint32_t tileM = p.tileM;
    const uint32_t partialK = g.k >= 2048 && g.k % 1024 == 0 ? 512 : g.k;
    mm.SetSingleShape(tileN, tileM, partialK);
    const uint64_t tasks = p.matrices.count;
    const uint32_t tilesN = g.n / tileN;
    const uint32_t tilesM = g.m / tileM;
    const uint64_t tilesPerBatch = uint64_t(tilesN) * tilesM;
    GlobalTensor<float> left, right, out;
    const uint64_t totalTasks = tasks * tilesPerBatch;
    for (uint64_t task = (Single ? 0 : GetBlockIdx()); task < totalTasks; task += (Single ? 1 : GetBlockNum())) {
        const uint32_t batch = static_cast<uint32_t>(task / tilesPerBatch);
        const uint64_t tile = task % tilesPerBatch;
        const uint32_t row = static_cast<uint32_t>(tile / tilesM) * tileN;
        const uint32_t col = static_cast<uint32_t>(tile % tilesM) * tileM;
        // The Cube view computes C^T = op(B)^T * op(A)^T. These offsets are
        // the corresponding column-major source views.
        const uint64_t leftOffset = TransLeft ? row : uint64_t(row) * g.k;
        const uint64_t rightOffset = TransRight ? uint64_t(col) * g.k : col;
        const uint64_t outOffset = uint64_t(row) * g.m + col;
        left.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(MatrixB(p, batch)) + leftOffset);
        right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(MatrixA(p, batch)) + rightOffset);
        out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(MatrixC(p, batch)) + outOffset);
        // Shorter FP32 reductions reduce rounding error for long dot products.
        // All partials of this output belong to this core; beta is already zero
        // on the Cube fast path. The first write must never read prior C.
        for (uint32_t firstK = 0; firstK < static_cast<uint32_t>(g.k); firstK += partialK) {
            const uint64_t leftK = TransLeft ? uint64_t(firstK) * g.n : firstK;
            const uint64_t rightK = TransRight ? firstK : uint64_t(firstK) * g.m;
            mm.SetTensorA(left[leftK], TransLeft);
            mm.SetTensorB(right[rightK], TransRight);
            mm.IterateAll(out[0], firstK == 0 ? 0 : 1);
            mm.End();
            if (partialK != static_cast<uint32_t>(g.k))
                PipeBarrier<PIPE_ALL>();
        }
    }
}
// Experimental FP32 TN path: four 32x128 MMADs form each 32x512 store.
__aicore__ inline void ConfigureTN96(
    const sgemm_gb::Group& g, Nd2NzParams& ac, Nd2NzParams& bc, LoadData3DParamsV2Pro& al, LoadData2DParams& bl,
    MmadParams& mm, FixpipeParamsV220& fp)
{
    ac.ndNum = 1;
    ac.nValue = 32;
    ac.dValue = g.k;
    ac.srcDValue = g.k;
    ac.dstNzC0Stride = 32;
    ac.dstNzNStride = 1;
    bc = ac;
    bc.nValue = 512;
    bc.dstNzC0Stride = 512;
    al.channelSize = g.k;
    al.extConfig = (uint64_t(32) << 16) | uint64_t(g.k);
    bl.repeatTimes = 8;
    bl.srcStride = 1;
    bl.dstGap = 0;
    bl.ifTranspose = false;
    mm.m = 32;
    mm.n = 128;
    mm.k = g.k;
    mm.cmatrixInitVal = true;
    fp.mSize = 32;
    fp.nSize = 512;
    fp.srcStride = 32;
    fp.dstStride = g.m;
    fp.ndNum = 1;
    fp.unitFlag = 0;
    fp.quantPre = QuantMode_t::NoQuant;
}
template <class P, class CopiedId, class EventId, class ComputedId, class FreeL1Id, class FreeCId>
__aicore__ inline void ProcessTN96Row(
    const P& p, const sgemm_gb::Group& g, uint32_t batch, uint64_t col, uint32_t row, uint32_t sequence,
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2All,
    LocalTensor<float>& c, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    CopiedId copied, EventId (&loaded)[2], ComputedId computed, FreeL1Id freeL1, EventId (&freeL0)[2], FreeCId freeC0,
    FreeCId freeC1, uint8_t (&padding)[4], Nd2NzParams& ac, Nd2NzParams& bc, LoadData3DParamsV2Pro& al,
    LoadData2DParams& bl, MmadParams& mm, FixpipeParamsV220& fp)
{
    const uint32_t slot = sequence % 2;
    const auto freeC = slot == 0 ? freeC0 : freeC1;
    auto result = c[slot * 32 * 512];
    WaitFlag<HardEvent::MTE1_MTE2>(freeL1);
    left.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(MatrixB(p, batch)) + row * g.k);
    DataCopy(a1, left, ac);
    if (row == 0) {
        right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(MatrixA(p, batch)) + col * g.k);
        DataCopy(b1, right, bc);
    }
    SetFlag<HardEvent::MTE2_MTE1>(copied);
    WaitFlag<HardEvent::MTE2_MTE1>(copied);
    WaitFlag<HardEvent::M_MTE1>(freeL0[0]);
    WaitFlag<HardEvent::M_MTE1>(freeL0[1]);
    SetFlag<HardEvent::M_MTE1>(freeL0[0]);
    SetFlag<HardEvent::M_MTE1>(freeL0[1]);
    Load3DSetFMatrixCal(1, 32, padding);
    LoadData<float>(a2, a1, al);
    WaitFlag<HardEvent::FIX_M>(freeC);
    for (uint32_t sub = 0; sub < 4; ++sub) {
        const uint32_t operandSlot = sub % 2;
        WaitFlag<HardEvent::M_MTE1>(freeL0[operandSlot]);
        auto b2 = b2All[operandSlot * 128 * 32];
        for (uint32_t kb = 0; kb < uint32_t(g.k) / 8; ++kb) {
            LoadData<float>(b2[kb * 128 * 8], b1[kb * 512 * 8 + sub * 128 * 8], bl);
        }
        SetFlag<HardEvent::MTE1_M>(loaded[operandSlot]);
        WaitFlag<HardEvent::MTE1_M>(loaded[operandSlot]);
        Mmad(result[sub * 32 * 128], a2, b2, mm);
        SetFlag<HardEvent::M_MTE1>(freeL0[operandSlot]);
    }
    SetFlag<HardEvent::MTE1_MTE2>(freeL1);
    SetFlag<HardEvent::M_FIX>(computed);
    WaitFlag<HardEvent::M_FIX>(computed);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(MatrixC(p, batch)) + row * g.m + col);
    Fixpipe<float, float, CFG_ROW_MAJOR>(output, result, fp);
    SetFlag<HardEvent::FIX_M>(freeC);
}
template <class FreeL1Id, class EventId, class FreeCId>
__aicore__ inline void SetInitialTN96Flags(FreeL1Id freeL1, EventId (&freeL0)[2], FreeCId freeC0, FreeCId freeC1)
{
    SetFlag<HardEvent::MTE1_MTE2>(freeL1);
    SetFlag<HardEvent::M_MTE1>(freeL0[0]);
    SetFlag<HardEvent::M_MTE1>(freeL0[1]);
    SetFlag<HardEvent::FIX_M>(freeC0);
    SetFlag<HardEvent::FIX_M>(freeC1);
}
template <class FreeL1Id, class EventId, class FreeCId>
__aicore__ inline void WaitFinalTN96Flags(FreeL1Id freeL1, EventId (&freeL0)[2], FreeCId freeC0, FreeCId freeC1)
{
    WaitFlag<HardEvent::MTE1_MTE2>(freeL1);
    WaitFlag<HardEvent::M_MTE1>(freeL0[0]);
    WaitFlag<HardEvent::M_MTE1>(freeL0[1]);
    WaitFlag<HardEvent::FIX_M>(freeC0);
    WaitFlag<HardEvent::FIX_M>(freeC1);
    PipeBarrier<PIPE_ALL>();
}
template <class P, class CopiedId, class EventId, class ComputedId, class FreeL1Id, class FreeCId>
__aicore__ inline void RunTN96Tasks(
    const P& p, const sgemm_gb::Group& g, LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2,
    LocalTensor<float>& b2All, LocalTensor<float>& c, GlobalTensor<float>& left, GlobalTensor<float>& right,
    GlobalTensor<float>& output, CopiedId copied, EventId (&loaded)[2], ComputedId computed, FreeL1Id freeL1,
    EventId (&freeL0)[2], FreeCId freeC0, FreeCId freeC1, uint8_t (&padding)[4], Nd2NzParams& ac, Nd2NzParams& bc,
    LoadData3DParamsV2Pro& al, LoadData2DParams& bl, MmadParams& mm, FixpipeParamsV220& fp)
{
    const uint64_t tiles = g.m / 512;
    const uint64_t tasks = tiles * p.matrices.count;
    const uint64_t begin = tasks * GetBlockIdx() / GetBlockNum();
    const uint64_t end = tasks * (GetBlockIdx() + 1) / GetBlockNum();
    uint32_t sequence = 0;
    for (uint64_t task = begin; task < end; ++task) {
        const uint32_t batch = task / tiles;
        const uint64_t col = (task % tiles) * 512;
        for (uint32_t row = 0; row < 96; row += 32, ++sequence)
            ProcessTN96Row(
                p, g, batch, col, row, sequence, a1, b1, a2, b2All, c, left, right, output, copied, loaded, computed,
                freeL1, freeL0, freeC0, freeC1, padding, ac, bc, al, bl, mm, fp);
    }
}
template <class P>
__aicore__ inline void ShortKDirectTN96(const P& p)
{
    if ASCEND_IS_AIV {
        return;
    }
    const auto& g = p.matrices.g;
    TPipe pipe;
    TBuf<TPosition::A1> a1Buf;
    TBuf<TPosition::B1> b1Buf;
    TBuf<TPosition::A2> a2Buf;
    TBuf<TPosition::B2> b2Buf;
    TBuf<TPosition::CO1> cBuf;
    pipe.InitBuffer(a1Buf, 32 * 32 * sizeof(float));
    pipe.InitBuffer(b1Buf, 512 * 32 * sizeof(float));
    pipe.InitBuffer(a2Buf, 32 * 32 * sizeof(float));
    pipe.InitBuffer(b2Buf, 2 * 128 * 32 * sizeof(float));
    pipe.InitBuffer(cBuf, 2 * 32 * 512 * sizeof(float));
    auto a1 = a1Buf.Get<float>();
    auto b1 = b1Buf.Get<float>();
    auto a2 = a2Buf.Get<float>();
    auto b2All = b2Buf.Get<float>();
    auto c = cBuf.Get<float>();
    const auto copied = pipe.FetchEventID(HardEvent::MTE2_MTE1);
    using EventId = decltype(pipe.FetchEventID(HardEvent::MTE1_M));
    EventId loaded[2] = {pipe.FetchEventID(HardEvent::MTE1_M), pipe.FetchEventID(HardEvent::MTE1_M)};
    const auto computed = pipe.FetchEventID(HardEvent::M_FIX);
    const auto freeL1 = pipe.FetchEventID(HardEvent::MTE1_MTE2);
    EventId freeL0[2] = {pipe.FetchEventID(HardEvent::M_MTE1), pipe.FetchEventID(HardEvent::M_MTE1)};
    const auto freeC0 = pipe.FetchEventID(HardEvent::FIX_M);
    const auto freeC1 = pipe.FetchEventID(HardEvent::FIX_M);
    SetInitialTN96Flags(freeL1, freeL0, freeC0, freeC1);
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    SetAtomicNone();
    uint8_t padding[4] = {0, 0, 0, 0};
    Nd2NzParams ac{};
    Nd2NzParams bc{};
    LoadData3DParamsV2Pro al;
    LoadData2DParams bl{};
    MmadParams mm{};
    FixpipeParamsV220 fp{};
    ConfigureTN96(g, ac, bc, al, bl, mm, fp);
    GlobalTensor<float> left, right, output;
    RunTN96Tasks(
        p, g, a1, b1, a2, b2All, c, left, right, output, copied, loaded, computed, freeL1, freeL0, freeC0, freeC1,
        padding, ac, bc, al, bl, mm, fp);
    WaitFinalTN96Flags(freeL1, freeL0, freeC0, freeC1);
}

template <class P, bool Single = false>
__aicore__ inline void CubeDispatch(const P& packet)
{
    if constexpr (!Single) {
        const auto& g = packet.matrices.g;
        if (g.ta && !g.tb && (g.k == 16 || g.k == 32) && g.m >= 512 && g.m % 512 == 0 && g.n == 96) {
            ShortKDirectTN96(packet);
            return;
        }
    }

    if (packet.matrices.g.tb) {
        if (packet.matrices.g.ta)
            CubeRun<P, true, true, Single>(packet);
        else
            CubeRun<P, true, false, Single>(packet);
    } else {
        if (packet.matrices.g.ta)
            CubeRun<P, false, true, Single>(packet);
        else
            CubeRun<P, false, false, Single>(packet);
    }
}
} // namespace
extern "C" __global__ __aicore__ void sgemm_gb_cube(sgemm_gb::CubePacket packet)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CubeDispatch(packet);
}
extern "C" __global__ __aicore__ void sgemm_gb_cube_offset(sgemm_gb::CubeOffsetPacket packet)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CubeDispatch(packet);
}
extern "C" __global__ __aicore__ void sgemm_gb_cube_groups(sgemm_gb::CubeGroupPacket packet)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    for (uint32_t batch = GetBlockIdx(); batch < packet.count; batch += GetBlockNum()) {
        uint32_t group = 0;
        while (group + 1 < packet.groups && batch >= packet.starts[group + 1])
            ++group;
        sgemm_gb::CubeSinglePacket one{};
        one.matrices.g = packet.g[group];
        one.matrices.count = 1;
        one.matrices.a[0] = packet.a[batch];
        one.matrices.b[0] = packet.b[batch];
        one.matrices.c[0] = packet.c[batch];
        one.tileM = packet.g[group].m;
        one.tileN = packet.g[group].n;
        for (uint32_t j = 0; j < sgemm_gb::kCubeTilingWords; ++j)
            one.tiling[j] = packet.tiling[group][j];
        CubeDispatch<sgemm_gb::CubeSinglePacket, true>(one);
    }
}
void sgemm_gb_cube_groups_do(sgemm_gb::CubeGroupPacket packet, uint32_t blocks, void* stream)
{
    sgemm_gb_cube_groups<<<blocks, nullptr, stream>>>(packet);
}
void sgemm_gb_cube_do(sgemm_gb::CubePacket packet, uint32_t blocks, void* stream)
{
    sgemm_gb_cube<<<blocks, nullptr, stream>>>(packet);
}
void sgemm_gb_cube_offset_do(sgemm_gb::CubeOffsetPacket packet, uint32_t blocks, void* stream)
{
    sgemm_gb_cube_offset<<<blocks, nullptr, stream>>>(packet);
}
