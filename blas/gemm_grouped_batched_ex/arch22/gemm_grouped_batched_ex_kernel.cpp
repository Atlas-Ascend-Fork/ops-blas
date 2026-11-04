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
#include "gemm_grouped_batched_ex_kernel.h"
#include "../../gemm_strided_batched/arch22/gemm_strided_batched_fp32.h"

using namespace AscendC;
using namespace matmul;
namespace {
constexpr uint32_t TILE = GEMM_GROUPED22_TILE;
constexpr MatmulShapeParams SHAPE{128, 128, GEMM_GROUPED22_K_CHUNK, 128, 128, 64};
constexpr MatmulFuncParams FUNC{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                true,  true};
constexpr MatmulBiasParams BIAS{false};
constexpr MatmulShapeParams WIDE_SHAPE{128, 256, 64, 128, 128, 64};
constexpr MatmulConfig CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(SHAPE, FUNC, BIAS);
constexpr MatmulConfig WIDE_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(WIDE_SHAPE, FUNC, BIAS);
template <bool TRANS>
using Input = MatmulType<TPosition::GM, CubeFormat::ND, float, TRANS>;
using Output = MatmulType<TPosition::GM, CubeFormat::ND, float>;
template <bool TA, bool TB, bool WIDE>
constexpr MatmulApiStaticTiling TILING =
    GetMatmulApiTiling<Input<TB>, Input<TA>, Output, Output>(WIDE ? WIDE_CONFIG : CONFIG);

__aicore__ inline float Scalar(uint64_t address, float host, bool device)
{
    if (!device)
        return host;
    GlobalTensor<float> scalar;
    scalar.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(address));
    return scalar.GetValue(0);
}
__aicore__ inline uint64_t Pointer(const GlobalTensor<uint64_t>& array, uint64_t index)
{
    return array.GetValue(index);
}

template <bool TA, bool TB, bool WIDE>
__aicore__ inline void CubeGroup(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const GemmGrouped22Group& g)
{
    const uint64_t first = (GetBlockIdx() + GetBlockNum() - g.cubeStart % GetBlockNum()) % GetBlockNum();
    if (first >= g.cubeTasks)
        return;
    TPipe pipe;
    MatmulImpl<Input<TB>, Input<TA>, Output, Output, TILING<TA, TB, WIDE>> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<uint64_t> ap, bp, cp;
    ap.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(a));
    bp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(b));
    cp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(c));
    const uint64_t rows = (static_cast<uint64_t>(g.n) + TILE - 1) / TILE;
    constexpr uint32_t COLUMN_TILE = WIDE ? 256 : TILE;
    const uint64_t cols = (static_cast<uint64_t>(g.m) + COLUMN_TILE - 1) / COLUMN_TILE;
    mm.SetOrgShape(TB ? g.ldb : g.n, TA ? g.m : g.lda, TB ? g.k : g.ldb, TA ? g.lda : g.k, g.tempLd);
    for (uint64_t task = first; task < g.cubeTasks; task += GetBlockNum()) {
        const uint64_t instance = task / (rows * cols);
        const uint64_t row = (task / cols % rows) * TILE;
        const uint64_t col = task % cols * COLUMN_TILE;
        const uint64_t index = g.instanceStart + instance;
        GlobalTensor<float> ag, bg, cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(ap, index)));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(bp, index)));
        if (g.direct)
            cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(cp, index)));
        else
            cg.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(workspace) + g.workspaceOffset + instance * g.tempLd * g.n);
        for (int64_t startK = 0; startK < g.k; startK += GEMM_GROUPED22_K_CHUNK) {
            mm.SetSingleShape(
                g.n - row < TILE ? g.n - row : TILE, g.m - col < COLUMN_TILE ? g.m - col : COLUMN_TILE,
                g.k - startK < GEMM_GROUPED22_K_CHUNK ? g.k - startK : GEMM_GROUPED22_K_CHUNK);
            mm.SetTensorA(bg[TB ? startK * g.ldb + row : row * g.ldb + startK], TB);
            mm.SetTensorB(ag[TA ? col * g.lda + startK : startK * g.lda + col], TA);
            mm.IterateAll(cg[row * g.tempLd + col], startK != 0);
            mm.End();
        }
    }
}

template <bool TA, bool TB>
__aicore__ inline void DispatchCube(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const GemmGrouped22Group& g)
{
    if (g.cubeTileN == 256)
        CubeGroup<TA, TB, true>(a, b, c, workspace, g);
    else
        CubeGroup<TA, TB, false>(a, b, c, workspace, g);
}

__aicore__ inline float ExactScale(float value, float beta)
{
    GemmSbFp32::Bits v{value}, b{beta}, result;
    result.bits = GemmSbFp32::FmaBits(v.bits, b.bits, (v.bits ^ b.bits) & 0x80000000U);
    return result.value;
}
__aicore__ inline bool Sensitive(float value)
{
    GemmSbFp32::Bits bits{value};
    return (bits.bits & 0x7fffffffU) >= 0x7149f2caU; // approximately 1e30, including Inf/NaN
}

__aicore__ inline void SmallProduct(
    const GemmGrouped22Group& g, const GlobalTensor<float>& ag, const GlobalTensor<float>& bg,
    LocalTensor<float>& input, LocalTensor<float>& acc, LocalTensor<float>& broadcast, const LocalTensor<float>& old,
    uint64_t col, uint64_t row, uint32_t count, float alpha, float beta)
{
    if (g.transA)
        Duplicate(acc, 0.0f, count);
    else {
        for (uint32_t r = 0; r < count; ++r)
            acc.SetValue(r, beta == 0.0f ? 0.0f : ExactScale(old.GetValue(r), beta));
    }
    PipeBarrier<PIPE_ALL>();
    const DataCopyExtParams copy{1, count * 4, 0, 0, 0};
    const DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    for (int64_t p = 0; p < g.k; ++p) {
        float bv = bg.GetValue(g.transB ? p * g.ldb + col : col * g.ldb + p);
        if (!g.transA)
            bv *= alpha;
        if (g.transA) {
            for (uint32_t r = 0; r < count; ++r)
                input.SetValue(r, ag.GetValue((row + r) * g.lda + p));
        } else
            DataCopyPad(input, ag[p * g.lda + row], copy, pad);
        PipeBarrier<PIPE_ALL>();
        bool sensitive = Sensitive(bv);
        for (uint32_t r = 0; r < count && !sensitive; ++r)
            sensitive = Sensitive(input.GetValue(r));
        if (sensitive) {
            for (uint32_t r = 0; r < count; ++r)
                acc.SetValue(r, GemmSbFp32::Fma(input.GetValue(r), bv, acc.GetValue(r)));
        } else {
            Duplicate(broadcast, bv, count);
            PipeBarrier<PIPE_V>();
            FusedMulAdd(input, broadcast, acc, count);
            PipeBarrier<PIPE_V>();
            DataCopy(acc, input, (count + 7) / 8 * 8);
        }
        PipeBarrier<PIPE_ALL>();
    }
    if (g.transA) {
        Muls(acc, acc, alpha, count);
        PipeBarrier<PIPE_V>();
        if (beta != 0.0f) {
            for (uint32_t r = 0; r < count; ++r)
                acc.SetValue(r, acc.GetValue(r) + ExactScale(old.GetValue(r), beta));
        }
    }
}
// Dense tiny matrices have at most 1024 values across both operands. Inspect exponent bits
// so NaNs and infinities are detected without floating-point reduction ambiguity.
__aicore__ inline bool DenseNeedsRepair(
    const GlobalTensor<float>& a, const GlobalTensor<float>& b, uint32_t aElements, uint32_t bElements,
    LocalTensor<float>& input, LocalTensor<float>& mask, LocalTensor<float>& result, LocalTensor<float>& scratch)
{
    const uint32_t count = (aElements + bElements + 7) / 8 * 8;
    const DataCopyExtParams copyA{1, aElements * 4, 0, 0, 0};
    const DataCopyExtParams copyB{1, bElements * 4, 0, 0, 0};
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
    const DataCopyPadExtParams<float> padB{true, 0, static_cast<uint8_t>(count - aElements - bElements), 0};
    DataCopyPad(input, a, copyA, noPad);
    DataCopyPad(input[aElements], b, copyB, padB);
    PipeBarrier<PIPE_ALL>();
    auto bits = input.ReinterpretCast<int32_t>();
    auto masks = mask.ReinterpretCast<uint32_t>();
    Duplicate(masks, 0x7fffffffU, count);
    PipeBarrier<PIPE_V>();
    And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(), masks.ReinterpretCast<uint16_t>(),
        count * 2);
    PipeBarrier<PIPE_V>();
    ShiftRight(bits, bits, int32_t(23), count);
    PipeBarrier<PIPE_V>();
    Cast(mask, bits, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    ReduceMax(result, mask, scratch, count, false);
    PipeBarrier<PIPE_ALL>();
    return result.GetValue(0) >= 226.0f;
}

__aicore__ inline void RepairVectorGroup(
    const GemmGrouped22Group& g, GlobalTensor<uint64_t>& ap, GlobalTensor<uint64_t>& bp, GlobalTensor<uint64_t>& cp,
    GM_ADDR workspace, LocalTensor<float>& input, LocalTensor<float>& acc, LocalTensor<float>& old,
    LocalTensor<float>& broadcast, float alpha, float beta)
{
    for (uint64_t task = (GetBlockIdx() + GetBlockNum() - g.vectorStart % GetBlockNum()) % GetBlockNum();
         task < g.instances; task += GetBlockNum()) {
        const uint64_t index = g.instanceStart + task;
        GlobalTensor<float> ag, bg, cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(ap, index)));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(bp, index)));
        const bool repair = DenseNeedsRepair(ag, bg, g.m * g.k, g.n * g.k, input, broadcast, acc, old);
        if (!repair)
            continue;
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(cp, index)));
        for (uint64_t col = 0; col < g.n; ++col) {
            SmallProduct(g, ag, bg, input, acc, broadcast, old, col, 0, g.m, alpha, beta);
            PipeBarrier<PIPE_ALL>();
            const DataCopyExtParams copy{1, static_cast<uint32_t>(g.m) * 4, 0, 0, 0};
            DataCopyPad(cg[col * g.ldc], acc, copy);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__aicore__ inline void ScaleVector(LocalTensor<float>& acc, LocalTensor<float>& old, float beta, uint32_t count)
{
    if (beta == 0.0f)
        Duplicate(acc, 0.0f, count);
    else
        for (uint32_t r = 0; r < count; ++r)
            acc.SetValue(r, ExactScale(old.GetValue(r), beta));
}

__aicore__ inline void ProcessVectorTasks(
    const GemmGrouped22Group& g, GlobalTensor<uint64_t>& ap, GlobalTensor<uint64_t>& bp, GlobalTensor<uint64_t>& cp,
    GM_ADDR workspace, LocalTensor<float>& input, LocalTensor<float>& acc, LocalTensor<float>& old,
    LocalTensor<float>& broadcast, float alpha, float beta)
{
    const bool scaling = g.k == 0 || alpha == 0.0f;
    const DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    const uint64_t chunks = (static_cast<uint64_t>(g.m) + TILE - 1) / TILE;
    for (uint64_t task = (GetBlockIdx() + GetBlockNum() - g.vectorStart % GetBlockNum()) % GetBlockNum();
         task < g.vectorTasks; task += GetBlockNum()) {
        const uint64_t instance = task / (chunks * g.n);
        const uint64_t col = task / chunks % g.n;
        const uint64_t row = task % chunks * TILE;
        const uint32_t count = g.m - row < TILE ? g.m - row : TILE;
        const DataCopyExtParams copy{1, count * 4, 0, 0, 0};
        const uint64_t index = g.instanceStart + instance;
        GlobalTensor<float> cg;
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(cp, index)));
        if (beta != 0.0f)
            DataCopyPad(old, cg[col * g.ldc + row], copy, pad);
        PipeBarrier<PIPE_ALL>();
        if (scaling) {
            ScaleVector(acc, old, beta, count);
        } else if (g.cube) {
            GlobalTensor<float> product;
            product.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(workspace) + g.workspaceOffset + instance * g.tempLd * g.n);
            DataCopyPad(acc, product[col * g.tempLd + row], copy, pad);
            PipeBarrier<PIPE_ALL>();
            Muls(acc, acc, alpha, count);
            if (beta != 0.0f) {
                Muls(old, old, beta, count);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, old, count);
            }
        } else {
            GlobalTensor<float> ag, bg;
            ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(ap, index)));
            bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Pointer(bp, index)));
            SmallProduct(g, ag, bg, input, acc, broadcast, old, col, row, count, alpha, beta);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(cg[col * g.ldc + row], acc, copy);
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace

extern "C" __global__ __aicore__ __cube__ void gemm_grouped22_cube(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GemmGrouped22Batch batch)
{
    for (uint32_t i = 0; i < batch.count; ++i) {
        const auto& g = batch.groups[i];
        if (!g.cube || Scalar(g.alphaAddress, g.alpha, g.scalarDevice & 1) == 0.0f)
            continue;
        if (g.transA) {
            if (g.transB)
                DispatchCube<true, true>(a, b, c, workspace, g);
            else
                DispatchCube<true, false>(a, b, c, workspace, g);
        } else {
            if (g.transB)
                DispatchCube<false, true>(a, b, c, workspace, g);
            else
                DispatchCube<false, false>(a, b, c, workspace, g);
        }
    }
}

extern "C" __global__ __aicore__ __vector__ void gemm_grouped22_vector(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GemmGrouped22Batch batch)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, accBuffer, oldBuffer, broadcastBuffer;
    pipe.InitBuffer(inputBuffer, 1024 * 4);
    pipe.InitBuffer(accBuffer, TILE * 4);
    pipe.InitBuffer(oldBuffer, 1024 * 4);
    pipe.InitBuffer(broadcastBuffer, 1024 * 4);
    auto input = inputBuffer.Get<float>(), acc = accBuffer.Get<float>();
    auto old = oldBuffer.Get<float>(), broadcast = broadcastBuffer.Get<float>();
    GlobalTensor<uint64_t> ap, bp, cp;
    ap.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(a));
    bp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(b));
    cp.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(c));
    for (uint32_t i = 0; i < batch.count; ++i) {
        const auto& g = batch.groups[i];
        if (!g.vectorTasks)
            continue;
        const float alpha = Scalar(g.alphaAddress, g.alpha, g.scalarDevice & 1);
        const float beta = Scalar(g.betaAddress, g.beta, g.scalarDevice & 2);
        if ((g.k == 0 || alpha == 0.0f) && beta == 1.0f)
            continue;
        if (g.repairSmall)
            RepairVectorGroup(g, ap, bp, cp, workspace, input, acc, old, broadcast, alpha, beta);
        else
            ProcessVectorTasks(g, ap, bp, cp, workspace, input, acc, old, broadcast, alpha, beta);
    }
}
void GemmGrouped22Cube(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, GemmGrouped22Batch batch)
{
    gemm_grouped22_cube<<<blocks, nullptr, stream>>>(a, b, c, workspace, batch);
}
void GemmGrouped22Vector(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, GemmGrouped22Batch batch)
{
    gemm_grouped22_vector<<<blocks, nullptr, stream>>>(a, b, c, workspace, batch);
}
