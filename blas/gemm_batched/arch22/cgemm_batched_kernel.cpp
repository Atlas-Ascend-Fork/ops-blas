/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cgemm_batched_kernel.cpp
 * \brief aclblasCgemmBatched kernels for Atlas A2 training series (arch22).
 *
 *        C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i], complex64, all
 *        matrices column-major, batch members addressed through device pointer
 *        arrays. See cgemm_batched_tiling_data.h for the pipeline and layout
 *        invariants.
 *
 *        Phase 0 (split) and the interleave tables of the epilogues reuse the
 *        shared building blocks in common/helper/complex_blas3_arch22.h; the
 *        batched GEMM and the two batched epilogues live here.
 *
 *        For one K chunk, split/GEMM/combine share a mixed kernel. Longer K
 *        uses one AIV split, one AIC launch per K chunk, and one AIV combine.
 *        The four real products share the Cube task space. Small shapes also
 *        have bounded local-storage implementations selected by the host.
 */

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "lib/matmul_intf.h"
#include "common/helper/complex_blas3_arch22.h"
#include "cgemm_batched_tiling_data.h"
#include "cgemm_batched_kernel.h"
#include "cgemm_batched_float.h"
#include "cgemm_batched_device_utils.h"

using namespace AscendC;
using namespace cgbt;

namespace {

// Complex elements per epilogue inner iteration; a whole multiple of a 256B
// fp32 repeat so the GatherMask repeat count is exact.
// Ten FP32 arrays' worth of scratch fit in 190 KiB. Larger chunks reduce
// copy/synchronization rounds. Reserve 2 KiB for metadata and input flags
// within the 192 KiB UB; both epilogues remain within this bound.
constexpr uint32_t COMBINE_CHUNK = 4864;
static_assert(10U * COMBINE_CHUNK * sizeof(float) + 2048U <= 192U * 1024U, "combine UB capacity");
static_assert(COMBINE_CHUNK % cblas3::ELENUM_REPEAT_FP32 == 0, "COMBINE_CHUNK must be a whole number of fp32 repeats");

// One transposed-Matmul instantiation cblas3 does not carry (transa = T/C and
// transb = T/C at once), built from the same config so the tile geometry and
// the host's SINGLE_K stay shared.
constexpr MatmulApiStaticTiling kTilingBothTrans =
    GetMatmulApiTiling<cblas3::MmTrans, cblas3::MmTrans, cblas3::MmPlain, cblas3::MmPlain>(cblas3::kConfig);
using MatmulBothTrans =
    MatmulImpl<cblas3::MmTrans, cblas3::MmTrans, cblas3::MmPlain, cblas3::MmPlain, kTilingBothTrans>;

// A larger per-core extent amortizes Matmul setup and reuses L1 operands
// across adjacent 128 x 128 Cube tiles when batch parallelism is sufficient.
constexpr MatmulConfig CGBT_REUSE_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(
    MatmulShapeParams{256, 256, cblas3::SINGLE_K, 128, 128, CBLAS3_BASE_K}, MatmulBiasParams{false});
template <bool XT, bool YT>
constexpr MatmulApiStaticTiling CGBT_REUSE_TILING = GetMatmulApiTiling<
    MatmulType<TPosition::GM, CubeFormat::ND, float, XT>, MatmulType<TPosition::GM, CubeFormat::ND, float, YT>,
    cblas3::MmPlain, cblas3::MmPlain>(CGBT_REUSE_CONFIG);
template <bool XT, bool YT>
using CgbtReuseMM = MatmulImpl<
    MatmulType<TPosition::GM, CubeFormat::ND, float, XT>, MatmulType<TPosition::GM, CubeFormat::ND, float, YT>,
    cblas3::MmPlain, cblas3::MmPlain, CGBT_REUSE_TILING<XT, YT>>;

// Same L1-reuse family one extent class higher, covering 257-320 order
// shapes. Without it those fall back to the 128 base tile and lose the
// operand-staging amortisation that lifted the 129-256 class past its bars
// (301-order CT measured within 0.1% of its limit on that fallback).
constexpr MatmulConfig CGBT_REUSE320_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(
    MatmulShapeParams{320, 320, cblas3::SINGLE_K, 128, 128, CBLAS3_BASE_K}, MatmulBiasParams{false});
template <bool XT, bool YT>
constexpr MatmulApiStaticTiling CGBT_REUSE320_TILING = GetMatmulApiTiling<
    MatmulType<TPosition::GM, CubeFormat::ND, float, XT>, MatmulType<TPosition::GM, CubeFormat::ND, float, YT>,
    cblas3::MmPlain, cblas3::MmPlain>(CGBT_REUSE320_CONFIG);
template <bool XT, bool YT>
using CgbtReuse320MM = MatmulImpl<
    MatmulType<TPosition::GM, CubeFormat::ND, float, XT>, MatmulType<TPosition::GM, CubeFormat::ND, float, YT>,
    cblas3::MmPlain, cblas3::MmPlain, CGBT_REUSE320_TILING<XT, YT>>;

template <uint32_t S>
constexpr MatmulConfig CGBT_MID_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(
    MatmulShapeParams{S, S, 256, S, S, S == 160U ? 32U : 64U}, MatmulBiasParams{false});
template <bool XT, bool YT, uint32_t S>
__aicore__ constexpr MatmulApiStaticTiling CgbtMakeMidTiling()
{
    auto tiling = GetMatmulApiTiling<
        MatmulType<TPosition::GM, CubeFormat::ND, float, XT>, MatmulType<TPosition::GM, CubeFormat::ND, float, YT>,
        cblas3::MmPlain, cblas3::MmPlain>(CGBT_MID_CONFIG<S>);
    tiling.dbL0C = 1;
    return tiling;
}
template <bool XT, bool YT, uint32_t S>
constexpr MatmulApiStaticTiling CGBT_MID_TILING = CgbtMakeMidTiling<XT, YT, S>();
template <bool XT, bool YT, uint32_t S = 176U>
using CgbtMidMM = MatmulImpl<
    MatmulType<TPosition::GM, CubeFormat::ND, float, XT>, MatmulType<TPosition::GM, CubeFormat::ND, float, YT>,
    cblas3::MmPlain, cblas3::MmPlain, CGBT_MID_TILING<XT, YT, S>>;

// Reduce L0 padding for medium panels while preserving the verified K tile.
template <uint32_t S>
constexpr MatmulConfig CGBT_MEDIUM_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(
    MatmulShapeParams{S, S, cblas3::SINGLE_K, S, S, CBLAS3_BASE_K}, MatmulBiasParams{false});
template <bool XT, bool YT, uint32_t S>
constexpr MatmulApiStaticTiling CGBT_MEDIUM_TILING = GetMatmulApiTiling<
    MatmulType<TPosition::GM, CubeFormat::ND, float, XT>, MatmulType<TPosition::GM, CubeFormat::ND, float, YT>,
    cblas3::MmPlain, cblas3::MmPlain>(CGBT_MEDIUM_CONFIG<S>);
template <bool XT, bool YT, uint32_t S>
using CgbtMediumMM = MatmulImpl<
    MatmulType<TPosition::GM, CubeFormat::ND, float, XT>, MatmulType<TPosition::GM, CubeFormat::ND, float, YT>,
    cblas3::MmPlain, cblas3::MmPlain, CGBT_MEDIUM_TILING<XT, YT, S>>;

// Dereferences one entry of a caller-side device pointer array.

__aicore__ inline void CgbtDirectBody(GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, const CgbtDirectTilingData& t)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> outputBuffer;
    pipe.InitBuffer(outputBuffer, 32U);
    auto output = outputBuffer.Get<float>();
    const uint64_t matrixSize = static_cast<uint64_t>(t.m) * t.n;
    for (uint64_t job = GetBlockIdx(); job < matrixSize * t.batchCount; job += GetBlockNum()) {
        const uint32_t batch = job / matrixSize;
        const uint32_t row = job % t.m;
        const uint32_t col = (job % matrixSize) / t.m;
        GlobalTensor<float> a;
        GlobalTensor<float> b;
        GlobalTensor<float> c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(aarray, batch, t.batchCount)));
        b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(barray, batch, t.batchCount)));
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(carray, batch, t.batchCount)));
        const uint64_t ci = 2U * (static_cast<uint64_t>(col) * t.ldc + row);
        const auto value = CgbtDirectValue(t, a, b, c, row, col);
        output.SetValue(0, value.real);
        output.SetValue(1, value.imag);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams params{1U, 2U * sizeof(float), 0U, 0U, 0U};
        DataCopyPad(c[ci], output, params);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
}

// ============================================================================
// Direct vector GEMM for small outputs and general alpha/beta
// ============================================================================
// Small panels reside entirely in UB. Strided DMA accepts arbitrary leading
// dimensions and independent batch pointers; all reductions retain K order.
template <uint32_t P>
__aicore__ inline void CgbtVectorLoad(
    GlobalTensor<float>& gm, uint32_t rows, uint32_t cols, uint32_t ld, LocalTensor<float>& stage,
    LocalTensor<float>& real, LocalTensor<float>& imag)
{
    const uint32_t bytes = rows * 8U;
    const uint32_t rounded = cblas3::AlignUpU(bytes, 32U);
    Duplicate(stage, 0.0F, P * P * 2U);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    const DataCopyExtParams copy{static_cast<uint16_t>(cols), bytes, (ld - rows) * 8U, (P * 8U - rounded) / 32U, 0U};
    DataCopyPad(
        stage, gm, copy, DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((rounded - bytes) / 4U), 0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint64_t count = 0U;
    GatherMask(real, stage, 1, false, 0, {1, P * P / 32U, 8, 8}, count);
    GatherMask(imag, stage, 2, false, 0, {1, P * P / 32U, 8, 8}, count);
    PipeBarrier<PIPE_ALL>();
}

struct CgbtVectorBuffers {
    LocalTensor<float> ax, ay, bz, bw, ar, ai, br, bi, sr, si, x, y, z, w, stage;
    LocalTensor<int32_t> ia, ib, ic;
};

template <uint32_t P>
__aicore__ inline void CgbtVectorInitializeSum(
    CgbtVectorBuffers& buffers, const CgbtDirectTilingData& t, GlobalTensor<float> c)
{
    constexpr uint32_t COUNT = P * P;
    if (t.transA == 2U)
        Muls(buffers.ai, buffers.ai, -1.0F, COUNT);
    if (t.transB == 2U)
        Muls(buffers.bi, buffers.bi, -1.0F, COUNT);
    Duplicate(buffers.sr, 0.0F, COUNT);
    Duplicate(buffers.si, 0.0F, COUNT);
    PipeBarrier<PIPE_V>();
    if (t.betaRe != 0.0F || t.betaIm != 0.0F) {
        CgbtVectorLoad<P>(c, t.m, t.n, t.ldc, buffers.stage, buffers.sr, buffers.si);
        if (!(t.transA == 0U && t.betaRe == 1.0F && t.betaIm == 0.0F)) {
            Muls(buffers.x, buffers.sr, t.betaRe, COUNT);
            Muls(buffers.y, buffers.si, t.betaIm, COUNT);
            Muls(buffers.z, buffers.si, t.betaRe, COUNT);
            Muls(buffers.w, buffers.sr, t.betaIm, COUNT);
            PipeBarrier<PIPE_V>();
            Sub(buffers.sr, buffers.x, buffers.y, COUNT);
            Add(buffers.si, buffers.z, buffers.w, COUNT);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <uint32_t P>
__aicore__ inline void CgbtVectorPrepareAlpha(CgbtVectorBuffers& buffers, const CgbtDirectTilingData& t)
{
    constexpr uint32_t COUNT = P * P;
    // T/C accumulates its dot product before alpha and beta. Save
    // beta*C in spare buffers.stage storage while preserving ordered sums.
    if (t.transA != 0U) {
        DataCopy(buffers.stage, buffers.sr, COUNT);
        DataCopy(buffers.stage[COUNT], buffers.si, COUNT);
        PipeBarrier<PIPE_V>();
        Duplicate(buffers.sr, 0.0F, COUNT);
        Duplicate(buffers.si, 0.0F, COUNT);
    } else {
        Muls(buffers.x, buffers.br, t.alphaRe, COUNT);
        Muls(buffers.y, buffers.bi, t.alphaIm, COUNT);
        Muls(buffers.z, buffers.bi, t.alphaRe, COUNT);
        Muls(buffers.w, buffers.br, t.alphaIm, COUNT);
        PipeBarrier<PIPE_V>();
        Sub(buffers.br, buffers.x, buffers.y, COUNT);
        Add(buffers.bi, buffers.z, buffers.w, COUNT);
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t P>
__aicore__ inline void CgbtVectorAccumulate(CgbtVectorBuffers& buffers, const CgbtDirectTilingData& t, uint32_t k)
{
    constexpr uint32_t COUNT = P * P;
    Gather(buffers.ax, buffers.ar, buffers.ia.ReinterpretCast<uint32_t>(), k * (t.transA == 0U ? P : 1U) * 4U, P);
    Gather(buffers.ay, buffers.ai, buffers.ia.ReinterpretCast<uint32_t>(), k * (t.transA == 0U ? P : 1U) * 4U, P);
    Gather(buffers.bz, buffers.br, buffers.ib.ReinterpretCast<uint32_t>(), k * (t.transB == 0U ? 1U : P) * 4U, P);
    Gather(buffers.bw, buffers.bi, buffers.ib.ReinterpretCast<uint32_t>(), k * (t.transB == 0U ? 1U : P) * 4U, P);
    PipeBarrier<PIPE_V>();
    Brcb(buffers.bz[P], buffers.bz, P / 8U, {1U, 8U});
    Brcb(buffers.bw[P], buffers.bw, P / 8U, {1U, 8U});
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams outer{1U, 1U, 0U, P / 8U, 0U, 1U};
    Mul(buffers.ic.ReinterpretCast<float>(), buffers.ax, buffers.bw[P], uint64_t(P), static_cast<uint8_t>(t.n), outer);
    Mul(buffers.ic.ReinterpretCast<float>()[COUNT], buffers.ay, buffers.bz[P], uint64_t(P), static_cast<uint8_t>(t.n),
        outer);
    // Product scratch buffers.z/buffers.w is disjoint from broadcast coefficients.
    Mul(buffers.x, buffers.ax, buffers.bz[P], uint64_t(P), static_cast<uint8_t>(t.n), outer);
    Mul(buffers.y, buffers.ay, buffers.bw[P], uint64_t(P), static_cast<uint8_t>(t.n), outer);
    PipeBarrier<PIPE_V>();
    Sub(buffers.x, buffers.x, buffers.y, COUNT);
    Add(buffers.z, buffers.ic.ReinterpretCast<float>(), buffers.ic.ReinterpretCast<float>()[COUNT], COUNT);
    PipeBarrier<PIPE_V>();
    Add(buffers.sr, buffers.sr, buffers.x, COUNT);
    Add(buffers.si, buffers.si, buffers.z, COUNT);
    PipeBarrier<PIPE_V>();
}

template <uint32_t P>
__aicore__ inline void CgbtVectorFinishAlpha(CgbtVectorBuffers& buffers, const CgbtDirectTilingData& t)
{
    constexpr uint32_t COUNT = P * P;
    if (t.transA != 0U) {
        Muls(buffers.x, buffers.sr, t.alphaRe, COUNT);
        Muls(buffers.y, buffers.si, t.alphaIm, COUNT);
        Muls(buffers.z, buffers.si, t.alphaRe, COUNT);
        Muls(buffers.w, buffers.sr, t.alphaIm, COUNT);
        PipeBarrier<PIPE_V>();
        Sub(buffers.sr, buffers.x, buffers.y, COUNT);
        Add(buffers.si, buffers.z, buffers.w, COUNT);
        PipeBarrier<PIPE_V>();
        Add(buffers.sr, buffers.sr, buffers.stage, COUNT);
        Add(buffers.si, buffers.si, buffers.stage[COUNT], COUNT);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t P>
__aicore__ inline void CgbtVectorRange(CgbtVectorBuffers& buffers, const CgbtDirectTilingData& t)
{
    constexpr uint32_t COUNT = P * P;
    Abs(buffers.x, buffers.ar, COUNT);
    Abs(buffers.y, buffers.ai, COUNT);
    Abs(buffers.z, buffers.br, COUNT);
    Abs(buffers.w, buffers.bi, COUNT);
    PipeBarrier<PIPE_V>();
    Max(buffers.x, buffers.x, buffers.y, COUNT);
    Max(buffers.z, buffers.z, buffers.w, COUNT);
    PipeBarrier<PIPE_V>();
    Max(buffers.x, buffers.x, buffers.z, COUNT);
    PipeBarrier<PIPE_V>();
    ReduceMax(buffers.sr, buffers.x, buffers.y, COUNT, false);
    PipeBarrier<PIPE_ALL>();
}

// C_old is still in GM. Only exceptional output lanes need scalar recovery.
template <uint32_t P>
__aicore__ inline void CgbtVectorRepair(
    CgbtVectorBuffers& buffers, GlobalTensor<float>& a, GlobalTensor<float>& b, GlobalTensor<float>& c,
    const CgbtDirectTilingData& t)
{
    ReduceSum(buffers.x, buffers.stage, buffers.y, 2U * P * P);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float sum = buffers.x.GetValue(0);
    if (sum <= 3.402823466e38F && sum >= -3.402823466e38F)
        return;
    auto bits = buffers.stage.ReinterpretCast<uint32_t>();
    for (uint32_t col = 0; col < t.n; ++col) {
        for (uint32_t row = 0; row < t.m; ++row) {
            const uint32_t index = 2U * (col * P + row);
            if ((bits.GetValue(index) & 0x7fffffffU) < 0x7f800000U &&
                (bits.GetValue(index + 1U) & 0x7fffffffU) < 0x7f800000U)
                continue;
            const auto value = CgbtDirectValue(t, a, b, c, row, col);
            buffers.stage.SetValue(index, value.real);
            buffers.stage.SetValue(index + 1U, value.imag);
        }
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
}

template <uint32_t P>
__aicore__ inline void CgbtVectorBatch(
    CgbtVectorBuffers& buffers, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, const CgbtDirectTilingData& t,
    uint32_t batch)
{
    constexpr uint32_t COUNT = P * P;
    GlobalTensor<float> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(aarray, batch, t.batchCount)));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(barray, batch, t.batchCount)));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(carray, batch, t.batchCount)));
    CgbtVectorLoad<P>(
        a, t.transA == 0U ? t.m : t.k, t.transA == 0U ? t.k : t.m, t.lda, buffers.stage, buffers.ar, buffers.ai);
    CgbtVectorLoad<P>(
        b, t.transB == 0U ? t.k : t.n, t.transB == 0U ? t.n : t.k, t.ldb, buffers.stage, buffers.br, buffers.bi);
    CgbtVectorRange<P>(buffers, t);
    if (!(buffers.sr.GetValue(0) <= CGBT_SAFE_COMPONENT_MAGNITUDE) ||
        !(t.alphaRe <= 1e10F && t.alphaRe >= -1e10F && t.alphaIm <= 1e10F && t.alphaIm >= -1e10F)) {
        for (uint32_t col = 0; col < t.n; ++col)
            for (uint32_t row = 0; row < t.m; ++row) {
                auto value = CgbtDirectValue(t, a, b, c, row, col);
                buffers.sr.SetValue(col * P + row, value.real);
                buffers.si.SetValue(col * P + row, value.imag);
            }
        PipeBarrier<PIPE_ALL>();
    } else {
        CgbtVectorInitializeSum<P>(buffers, t, c);
        CgbtVectorPrepareAlpha<P>(buffers, t);
        for (uint32_t k = 0; k < t.k; ++k) {
            CgbtVectorAccumulate<P>(buffers, t, k);
        }
        CgbtVectorFinishAlpha<P>(buffers, t);
    }
    cblas3::BuildInterleaveIndex(
        buffers.ic, buffers.stage.ReinterpretCast<int32_t>(), buffers.x.ReinterpretCast<int32_t>(), COUNT, COUNT);
    PipeBarrier<PIPE_V>();
    Gather(buffers.stage, buffers.sr, buffers.ic.ReinterpretCast<uint32_t>(), 0U, COUNT * 2U);
    PipeBarrier<PIPE_V>();
    // Alpha scaling can overflow even when beta is zero. Repair exceptional
    // lanes using the same sequential complex operations as the direct path.
    CgbtVectorRepair<P>(buffers, a, b, c, t);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const DataCopyExtParams output{
        static_cast<uint16_t>(t.n), t.m * 8U, (P * 8U - cblas3::AlignUpU(t.m * 8U, 32U)) / 32U, (t.ldc - t.m) * 8U, 0U};
    DataCopyPad(c, buffers.stage, output);
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P>
__aicore__ inline void CgbtVectorBody(GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, const CgbtDirectTilingData& t)
{
    constexpr uint32_t COUNT = P * P;
    constexpr int32_t SHIFT = P == 16U ? 4 : 5;
    TPipe pipe;
    TBuf<TPosition::VECCALC> values, indices, operands;
    pipe.InitBuffer(operands, P * 20U * sizeof(float));
    auto ax = operands.Get<float>(), ay = operands.Get<float>()[P];
    auto bz = operands.Get<float>()[2U * P], bw = operands.Get<float>()[11U * P];
    pipe.InitBuffer(values, 12U * COUNT * sizeof(float));
    pipe.InitBuffer(indices, 4U * COUNT * sizeof(uint32_t));
    auto all = values.Get<float>();
    auto ar = all, ai = all[COUNT], br = all[2U * COUNT], bi = all[3U * COUNT];
    auto sr = all[4U * COUNT], si = all[5U * COUNT];
    auto x = all[6U * COUNT], y = all[7U * COUNT], z = all[8U * COUNT], w = all[9U * COUNT];
    auto stage = all[10U * COUNT];
    auto ia = indices.Get<int32_t>(), ib = indices.Get<int32_t>()[COUNT];
    auto ic = indices.Get<int32_t>()[2U * COUNT];
    CreateVecIndex(ia, int32_t(0), P);
    CreateVecIndex(ib, int32_t(0), P);
    PipeBarrier<PIPE_V>();
    ShiftLeft(ia, ia, t.transA == 0U ? 2 : SHIFT + 2, P);
    ShiftLeft(ib, ib, t.transB == 0U ? SHIFT + 2 : 2, P);
    cblas3::BuildInterleaveIndex(ic, stage.ReinterpretCast<int32_t>(), x.ReinterpretCast<int32_t>(), COUNT, COUNT);
    PipeBarrier<PIPE_ALL>();
    CgbtVectorBuffers buffers{ax, ay, bz, bw, ar, ai, br, bi, sr, si, x, y, z, w, stage, ia, ib, ic};
    for (uint32_t batch = GetBlockIdx(); batch < t.batchCount; batch += GetBlockNum()) {
        CgbtVectorBatch<P>(buffers, aarray, barray, carray, t, batch);
    }
}

// ---------------------------------------------------------------------------
//  Phase 0: batched de-interleave (+ optional conjugation), A and B fused
// ---------------------------------------------------------------------------
// One source column of one batch member: stage it, split the interleaved run
// into re / im lanes, optionally conjugate, write the two packed planes.
__aicore__ inline void CgbtSplitOneColumn(
    const GlobalTensor<float>& srcGm, GlobalTensor<float>& reGm, GlobalTensor<float>& imGm,
    const CgbtSplitTilingData& tiling, uint64_t batch, uint32_t col, const LocalTensor<float>& stage,
    const LocalTensor<float>& re, const LocalTensor<float>& im, float riskScale = 0.0F)
{
    const uint64_t srcColOffset = static_cast<uint64_t>(col) * tiling.ld * cblas3::COMPLEX_ELENUM;
    const uint64_t dstColOffset = batch * tiling.planeFloats + static_cast<uint64_t>(col) * tiling.packedLd;

    for (uint32_t base = 0; base < tiling.rows; base += cblas3::SPLIT_CHUNK) {
        const uint32_t len = cblas3::SplitChunkLen(base, tiling.rows);
        cblas3::LoadChunkDeinterleaved(srcGm, srcColOffset, base, len, stage, re, im);
        if (tiling.negateImag != 0U) {
            Muls(im, im, -1.0F, len);
            PipeBarrier<PIPE_V>();
        }

        if (riskScale != 0.0F) {
            Muls(stage, re, riskScale, len);
            Muls(stage[cblas3::SPLIT_CHUNK], im, riskScale, len);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopyExtParams outParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(reGm[dstColOffset + base], re, outParams);
        DataCopyPad(imGm[dstColOffset + base], im, outParams);

        cblas3::SyncAfterChunkStore();
    }
}

__aicore__ inline void CgbtPrepareSplitStore(
    const CgbtSplitTilingData& t, const LocalTensor<float>& stage, const LocalTensor<float>& re,
    const LocalTensor<float>& im, uint32_t len, float riskScale)
{
    if (t.negateImag) {
        Muls(im, im, -1.0F, len);
        PipeBarrier<PIPE_V>();
    }
    if (riskScale != 0.0F) {
        Muls(stage, re, riskScale, len);
        Muls(stage[cblas3::SPLIT_CHUNK], im, riskScale, len);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

__aicore__ inline void CgbtSplitFlatSegment(
    const GlobalTensor<float>& src, GlobalTensor<float>& reGm, GlobalTensor<float>& imGm, const CgbtSplitTilingData& t,
    uint64_t batch, uint64_t segment, const LocalTensor<float>& stage, const LocalTensor<float>& re,
    const LocalTensor<float>& im, float riskScale = 0.0F)
{
    const uint64_t elements = static_cast<uint64_t>(t.rows) * t.cols;
    const uint64_t base = segment * cblas3::SPLIT_CHUNK;
    const uint32_t len = elements - base < cblas3::SPLIT_CHUNK ? elements - base : cblas3::SPLIT_CHUNK;
    cblas3::LoadChunkDeinterleaved(src, base * 2U, 0U, len, stage, re, im);
    CgbtPrepareSplitStore(t, stage, re, im, len, riskScale);
    const DataCopyExtParams copy{1U, len * 4U, 0U, 0U, 0U};
    DataCopyPad(reGm[batch * t.planeFloats + base], re, copy);
    DataCopyPad(imGm[batch * t.planeFloats + base], im, copy);
    cblas3::SyncAfterChunkStore();
}

__aicore__ inline uint32_t CgbtStoredColumns(const CgbtSplitTilingData& t)
{
    return t.rows > 64U && t.rows <= 176U && t.rows != t.packedLd && t.cols > 64U && t.cols <= 176U ?
               cblas3::AlignUpU(t.cols, t.cols <= 144U ? 32U : 16U) :
               t.cols;
}

__aicore__ inline void CgbtSplitColumns(
    const GlobalTensor<float>& src, GlobalTensor<float>& reGm, GlobalTensor<float>& imGm, const CgbtSplitTilingData& t,
    uint64_t batch, uint64_t segment, const LocalTensor<float>& stage, const LocalTensor<float>& re,
    const LocalTensor<float>& im, float riskScale = 0.0F)
{
    const uint32_t group = cblas3::SPLIT_CHUNK / t.packedLd;
    const uint32_t col = segment * group;
    const uint32_t columns = col >= t.cols ? 0U : (t.cols - col < group ? t.cols - col : group);
    const uint32_t stored = CgbtStoredColumns(t);
    const uint32_t len = (stored - col < group ? stored - col : group) * t.packedLd;
    const uint32_t alignedBytes = (t.rows * 8U + 31U) / 32U * 32U;
    const DataCopyExtParams copy{
        static_cast<uint16_t>(columns), t.rows * 8U, (t.ld - t.rows) * 8U, (t.packedLd * 8U - alignedBytes) / 32U, 0U};
    if (t.packedLd != t.rows) {
        Duplicate(stage, 0.0F, len * 2U);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    const DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>((alignedBytes - t.rows * 8U) / 4U), 0.0F};
    if (columns != 0U)
        DataCopyPad(stage, src[static_cast<uint64_t>(col) * t.ld * 2U], copy, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint64_t count = 0U;
    const uint16_t repeats = (len * 2U + 63U) / 64U;
    GatherMask(re, stage, 1, false, 0, {1, repeats, 8, 8}, count);
    GatherMask(im, stage, 2, false, 0, {1, repeats, 8, 8}, count);
    PipeBarrier<PIPE_V>();
    CgbtPrepareSplitStore(t, stage, re, im, len, riskScale);
    const DataCopyExtParams output{1U, len * 4U, 0U, 0U, 0U};
    const uint64_t offset = batch * t.planeFloats + static_cast<uint64_t>(col) * t.packedLd;
    DataCopyPad(reGm[offset], re, output);
    DataCopyPad(imGm[offset], im, output);
    cblas3::SyncAfterChunkStore();
}

__aicore__ inline uint64_t CgbtSplitSegmentCount(const CgbtSplitTilingData& t, bool flat)
{
    if (flat)
        return (static_cast<uint64_t>(t.rows) * t.cols + cblas3::SPLIT_CHUNK - 1U) / cblas3::SPLIT_CHUNK;
    if (t.packedLd <= cblas3::SPLIT_CHUNK) {
        const uint32_t columns = cblas3::SPLIT_CHUNK / t.packedLd;
        return (CgbtStoredColumns(t) + columns - 1U) / columns;
    }
    return t.cols;
}

__aicore__ inline void CgbtSplitOperandJob(
    GM_ADDR pointers, const CgbtSplitTilingData& t, bool flat, uint64_t segments, uint64_t job,
    GlobalTensor<float>& source, uint64_t& lastBatch, GlobalTensor<float> reGm, GlobalTensor<float> imGm,
    LocalTensor<float> stage, LocalTensor<float> re, LocalTensor<float> im, float riskScale)
{
    if (segments == 0U)
        return;
    const uint64_t batch = job / segments;
    if (batch != lastBatch) {
        source.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(static_cast<uintptr_t>(CgbtBatchPtr(pointers, batch, t.batchCount))));
        lastBatch = batch;
    }
    const uint64_t segment = job % segments;
    if (flat)
        CgbtSplitFlatSegment(source, reGm, imGm, t, batch, segment, stage, re, im, riskScale);
    else if (t.packedLd <= cblas3::SPLIT_CHUNK)
        CgbtSplitColumns(source, reGm, imGm, t, batch, segment, stage, re, im, riskScale);
    else
        CgbtSplitOneColumn(source, reGm, imGm, t, batch, segment, stage, re, im, riskScale);
}

__aicore__ inline void CgbtStoreInputRisk(GM_ADDR inputFlags, uint64_t coreIdx, LocalTensor<float> range)
{
    if (inputFlags != nullptr) {
        // A scaled probe detects wide finite inputs as well as Inf/NaN without
        // a reduction per input segment. Each core owns one four-byte flag.
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t status = asc_get_status();
        range.ReinterpretCast<uint32_t>().SetValue(0, status != 0U ? 0x7fc00000U : 0U);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        GlobalTensor<float> risk;
        risk.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputFlags));
        DataCopyPad(risk[coreIdx], range, DataCopyExtParams{1U, 4U, 0U, 0U, 0U});
        cblas3::SyncAfterChunkStore();
    }
}

struct CgbtSplitScratch {
    TBuf<TPosition::VECCALC> stageBuf, reBuf, imBuf, rangeBuffer;
    LocalTensor<float> stage, re, im, range;
};

__aicore__ inline void CgbtInitSplitScratch(TPipe& pipe, CgbtSplitScratch& scratch)
{
    pipe.InitBuffer(scratch.stageBuf, cblas3::SPLIT_CHUNK * cblas3::COMPLEX_ELENUM * sizeof(float));
    pipe.InitBuffer(scratch.reBuf, cblas3::SPLIT_CHUNK * sizeof(float));
    pipe.InitBuffer(scratch.imBuf, cblas3::SPLIT_CHUNK * sizeof(float));
    scratch.stage = scratch.stageBuf.Get<float>();
    scratch.re = scratch.reBuf.Get<float>();
    scratch.im = scratch.imBuf.Get<float>();
    pipe.InitBuffer(scratch.rangeBuffer, 1024U);
    scratch.range = scratch.rangeBuffer.Get<float>();
}

__aicore__ inline void CgbtSplitFusedBody(
    GM_ADDR ptrA, GM_ADDR ptrB, GM_ADDR dstAr, GM_ADDR dstAi, GM_ADDR dstBr, GM_ADDR dstBi,
    const CgbtSplitTilingData& ta, const CgbtSplitTilingData& tb, bool paired, GM_ADDR inputFlags, float riskScale)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum()) * (paired ? 2U : 1U);
    const uint32_t batches = ta.batchCount;
    const bool flatA = ta.ld == ta.rows && ta.packedLd == ta.rows;
    const bool flatB = tb.ld == tb.rows && tb.packedLd == tb.rows;
    const uint64_t segmentsA = CgbtSplitSegmentCount(ta, flatA);
    const uint64_t segmentsB = CgbtSplitSegmentCount(tb, flatB);
    const uint64_t jobsA = batches * segmentsA;
    const uint64_t jobsB = batches * segmentsB;
    if (coreNum == 0U || (jobsA + jobsB) == 0U) {
        return;
    }
    const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t jobs = jobsA + jobsB;

    GlobalTensor<float> reGmA;
    GlobalTensor<float> imGmA;
    GlobalTensor<float> reGmB;
    GlobalTensor<float> imGmB;
    reGmA.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstAr));
    imGmA.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstAi));
    reGmB.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstBr));
    imGmB.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstBi));

    TPipe pipe;
    CgbtSplitScratch scratch;
    CgbtInitSplitScratch(pipe, scratch);
    auto stage = scratch.stage;
    auto re = scratch.re;
    auto im = scratch.im;
    auto range = scratch.range;

    // One scalar GM read per (segment, batch): consecutive jobs of a core walk
    // columns inside a batch, so the source matrix pointer only changes when
    // the batch does. The sentinels are batch values that never occur.
    GlobalTensor<float> srcA;
    GlobalTensor<float> srcB;
    uint64_t lastBatchA = static_cast<uint64_t>(ta.batchCount);
    uint64_t lastBatchB = static_cast<uint64_t>(tb.batchCount);

    if (inputFlags != nullptr) {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        (void)asc_get_status();
    }
    for (uint64_t job = coreIdx; job < jobs; job += coreNum) {
        if (job < jobsA) {
            CgbtSplitOperandJob(
                ptrA, ta, flatA, segmentsA, job, srcA, lastBatchA, reGmA, imGmA, stage, re, im, riskScale);
        } else {
            CgbtSplitOperandJob(
                ptrB, tb, flatB, segmentsB, job - jobsA, srcB, lastBatchB, reGmB, imGmB, stage, re, im, riskScale);
        }
    }
    CgbtStoreInputRisk(inputFlags, coreIdx, range);
    pipe.Reset();
}

// ---------------------------------------------------------------------------
//  Phase 1: the four real GEMM products of every batch member, fused
// ---------------------------------------------------------------------------
// GM views of one output tile's three operands, sized so the declared length
// never runs past the real allocation (same rule as the shared layer's
// ResolveGemmOperands): the base pointer already carries the tile offset.
template <bool X_TRANS, bool Y_TRANS>
__aicore__ inline void CgbtResolveOperands(
    const CgbtGemmTilingData& t, uint64_t offX, uint64_t offY, uint64_t offT, uint64_t batch, uint32_t rowBase,
    uint32_t colBase, uint32_t rowCount, uint32_t colCount, __gm__ float* wsBase, GlobalTensor<float>& xGlobal,
    GlobalTensor<float>& yGlobal, GlobalTensor<float>& dGlobal)
{
    __gm__ float* xBase = wsBase + offX + batch * t.xPlaneFloats;
    __gm__ float* yBase = wsBase + offY + batch * t.yPlaneFloats;
    __gm__ float* dBase = wsBase + offT + batch * t.tPlaneFloats;

    if constexpr (X_TRANS) {
        // X stored as X^T: the k extent runs along the view's rows.
        const uint64_t xLen = static_cast<uint64_t>(t.kCount - 1U) * t.ldX + rowCount;
        xGlobal.SetGlobalBuffer(xBase + static_cast<uint64_t>(t.kBase) * t.ldX + rowBase, xLen);
    } else {
        const uint64_t xLen = static_cast<uint64_t>(rowCount - 1U) * t.ldX + t.kCount;
        xGlobal.SetGlobalBuffer(xBase + static_cast<uint64_t>(rowBase) * t.ldX + t.kBase, xLen);
    }
    if constexpr (Y_TRANS) {
        const uint64_t yLen = static_cast<uint64_t>(colCount - 1U) * t.ldY + t.kCount;
        yGlobal.SetGlobalBuffer(yBase + static_cast<uint64_t>(colBase) * t.ldY + t.kBase, yLen);
    } else {
        const uint64_t yLen = static_cast<uint64_t>(t.kCount - 1U) * t.ldY + colCount;
        yGlobal.SetGlobalBuffer(yBase + static_cast<uint64_t>(t.kBase) * t.ldY + colBase, yLen);
    }
    const uint64_t dLen = static_cast<uint64_t>(rowCount - 1U) * t.ldD + colCount;
    dGlobal.SetGlobalBuffer(dBase + static_cast<uint64_t>(rowBase) * t.ldD + colBase, dLen);
}

// SetOrgShape slots per mode, following SetGemmOrgShape in the shared layer:
// a PLAIN operand's row stride comes from orgKa (left) / orgN (right), a
// TRANSPOSED one's from orgM (left) / orgKb (right). orgKc is always the D row
// stride.
template <class MM, bool X_TRANS, bool Y_TRANS>
__aicore__ inline void CgbtSetOrgShape(MM& mm, const CgbtGemmTilingData& t)
{
    if constexpr (!X_TRANS && !Y_TRANS) {
        mm.SetOrgShape(
            static_cast<int32_t>(t.m), static_cast<int32_t>(t.ldY), static_cast<int32_t>(t.ldX),
            static_cast<int32_t>(t.ldX), static_cast<int32_t>(t.ldD));
    } else if constexpr (X_TRANS && !Y_TRANS) {
        mm.SetOrgShape(
            static_cast<int32_t>(t.ldX), static_cast<int32_t>(t.ldY), static_cast<int32_t>(t.k),
            static_cast<int32_t>(t.k), static_cast<int32_t>(t.ldD));
    } else if constexpr (!X_TRANS && Y_TRANS) {
        mm.SetOrgShape(
            static_cast<int32_t>(t.m), static_cast<int32_t>(t.n), static_cast<int32_t>(t.ldX),
            static_cast<int32_t>(t.ldY), static_cast<int32_t>(t.ldD));
    } else {
        mm.SetOrgShape(
            static_cast<int32_t>(t.ldX), static_cast<int32_t>(t.n), static_cast<int32_t>(t.k),
            static_cast<int32_t>(t.ldY), static_cast<int32_t>(t.ldD));
    }
}

template <bool Y_TRANS, uint32_t TILE, bool PAD_OUTPUT>
__aicore__ inline bool CgbtUsePaddedOutput(const CgbtGemmTilingData& t)
{
    return PAD_OUTPUT && t.ldD <= TILE &&
           ((!Y_TRANS && t.ldY == t.ldD) ||
            (Y_TRANS && t.k > 64U && t.k <= 176U && t.k != t.ldY && t.n > 64U && t.n <= 176U));
}

template <class MM, bool X_TRANS, bool Y_TRANS, uint32_t TILE = CBLAS3_TILE_M, bool PAD_OUTPUT = false>
__aicore__ inline void CgbtGemmBodyT(GM_ADDR workspace, const CgbtGemmTilingData& t)
{
    if (t.mBlocks == 0U || t.nBlocks == 0U || t.batchCount == 0U || t.kCount == 0U || GetBlockNum() == 0) {
        return;
    }
    const uint64_t tilesPerBatch = static_cast<uint64_t>(t.mBlocks) * t.nBlocks;
    const uint64_t batchTiles = tilesPerBatch * t.batchCount;
    const uint64_t totalTasks = batchTiles * CGBT_PRODUCTS;
    const uint64_t coreNum = static_cast<uint64_t>(GetBlockNum());
    const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());

    __gm__ float* wsBase = reinterpret_cast<__gm__ float*>(workspace);

    TPipe pipe;
    MM mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false); // full fp32 mantissa: HF32 would break the precision budget
    const bool paddedOutput = CgbtUsePaddedOutput<Y_TRANS, TILE, PAD_OUTPUT>(t);
    auto org = t;
    if (paddedOutput)
        org.n = t.ldD;
    CgbtSetOrgShape<MM, X_TRANS, Y_TRANS>(mm, org);

    for (uint64_t task = coreIdx; task < totalTasks; task += coreNum) {
        const uint32_t product = static_cast<uint32_t>(task / batchTiles);
        const uint64_t rem = task % batchTiles;
        const uint64_t batch = rem / tilesPerBatch;
        const uint32_t tile = static_cast<uint32_t>(rem % tilesPerBatch);
        // T1 = Br*Ar, T2 = Bi*Ai, T3 = Bi*Ar, T4 = Br*Ai; X is the B plane,
        // Y the A plane.
        const uint64_t offX = (product == 1U || product == 2U) ? t.offXi : t.offXr;
        const uint64_t offY = (product == 0U || product == 2U) ? t.offYr : t.offYi;
        const uint64_t offT = t.offT1 + static_cast<uint64_t>(product) * t.tPlaneFloats * t.batchCount;

        const uint32_t rowBase = (tile / t.nBlocks) * TILE;
        const uint32_t colBase = (tile % t.nBlocks) * TILE;
        const uint32_t rowCount = (rowBase + TILE > t.m) ? (t.m - rowBase) : TILE;
        const uint32_t colCount = paddedOutput ? t.ldD : ((colBase + TILE > t.n) ? (t.n - colBase) : TILE);

        GlobalTensor<float> xGlobal;
        GlobalTensor<float> yGlobal;
        GlobalTensor<float> dGlobal;
        CgbtResolveOperands<X_TRANS, Y_TRANS>(
            t, offX, offY, offT, batch, rowBase, colBase, rowCount, colCount, wsBase, xGlobal, yGlobal, dGlobal);

        mm.SetSingleShape(
            static_cast<int32_t>(rowCount), static_cast<int32_t>(colCount), static_cast<int32_t>(t.kCount));
        mm.SetTensorA(xGlobal, X_TRANS);
        mm.SetTensorB(yGlobal, Y_TRANS);
        mm.IterateAll(dGlobal, static_cast<uint8_t>(t.enAtomic));
    }

    mm.End();
}

// ============================================================================
// Packed Cube panels for general alpha/beta
// ============================================================================
// [Br Bi] [ Ar Ai] = [ Re(C) Im(C) ].  Full FP32 arithmetic; four
//         [-Ai Ar]
// real products share one Cube invocation and one output transfer.
struct CgbtPackedSplitBuffers {
    LocalTensor<float> stage, re;
    LocalTensor<int32_t> idx;
    LocalTensor<float> aux, range;
};

template <uint32_t P>
__aicore__ inline void CgbtPackedSplitIndex(CgbtPackedSplitBuffers& buffers, bool trans)
{
    constexpr uint32_t E = P * P, GROUP = 4096U / E;
    constexpr int32_t L = P == 16U ? 4 : (P == 32U ? 5 : 6);
    // Map [real-row | imag-row] into source planar storage, with optional
    // physical transpose. P is an on-chip padding extent, not an input shape.
    CreateVecIndex(buffers.idx, int32_t(0), 2U * E);
    PipeBarrier<PIPE_V>();
    ShiftRight(buffers.aux.ReinterpretCast<int32_t>(), buffers.idx, L + 1, 2U * E);
    PipeBarrier<PIPE_V>();
    ShiftLeft(buffers.stage.ReinterpretCast<int32_t>(), buffers.aux.ReinterpretCast<int32_t>(), L + 1, 2U * E);
    PipeBarrier<PIPE_V>();
    Sub(buffers.stage.ReinterpretCast<int32_t>(), buffers.idx, buffers.stage.ReinterpretCast<int32_t>(), 2U * E);
    PipeBarrier<PIPE_V>();
    ShiftRight(buffers.idx, buffers.stage.ReinterpretCast<int32_t>(), L, 2U * E);
    PipeBarrier<PIPE_V>();
    ShiftLeft(buffers.re.ReinterpretCast<int32_t>(), buffers.idx, L, 2U * E);
    PipeBarrier<PIPE_V>();
    Sub(buffers.stage.ReinterpretCast<int32_t>(), buffers.stage.ReinterpretCast<int32_t>(),
        buffers.re.ReinterpretCast<int32_t>(), 2U * E);
    if (trans)
        ShiftLeft(buffers.stage.ReinterpretCast<int32_t>(), buffers.stage.ReinterpretCast<int32_t>(), L, 2U * E);
    else
        ShiftLeft(buffers.aux.ReinterpretCast<int32_t>(), buffers.aux.ReinterpretCast<int32_t>(), L, 2U * E);
    ShiftLeft(buffers.idx, buffers.idx, 2 * L, 2U * E);
    PipeBarrier<PIPE_V>();
    Add(buffers.idx, buffers.idx, buffers.aux.ReinterpretCast<int32_t>(), 2U * E);
    PipeBarrier<PIPE_V>();
    Add(buffers.idx, buffers.idx, buffers.stage.ReinterpretCast<int32_t>(), 2U * E);
    PipeBarrier<PIPE_V>();
    ShiftLeft(buffers.idx, buffers.idx, 2, 2U * E);
    for (uint32_t q = 1; q < GROUP; ++q) {
        Adds(buffers.idx[q * 2U * E], buffers.idx, int32_t(q * 2U * E * 4U), 2U * E);
    }
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P>
__aicore__ inline void CgbtPackedSplitImaginary(
    CgbtPackedSplitBuffers& buffers, GlobalTensor<float> ws, const DataCopyExtParams& output, uint32_t batch,
    uint32_t count, uint32_t len)
{
    constexpr uint32_t E = P * P, S = 8U * E + 8U;
    constexpr int32_t L = P == 16U ? 4 : (P == 32U ? 5 : 6);
    for (uint32_t q = 0; q < count; ++q)
        Muls(buffers.re[q * 2U * E + E], buffers.re[q * 2U * E + E], -1.0F, E);
    ShiftRight(buffers.aux.ReinterpretCast<int32_t>(), buffers.idx, 2 * L + 2, len);
    Duplicate(buffers.stage.ReinterpretCast<int32_t>(), int32_t(1), len);
    PipeBarrier<PIPE_V>();
    And(buffers.aux.ReinterpretCast<uint16_t>(), buffers.aux.ReinterpretCast<uint16_t>(),
        buffers.stage.ReinterpretCast<uint16_t>(), len * 2U);
    PipeBarrier<PIPE_V>();
    ShiftLeft(buffers.aux.ReinterpretCast<int32_t>(), buffers.aux.ReinterpretCast<int32_t>(), 2 * L + 3, len);
    PipeBarrier<PIPE_V>();
    Sub(buffers.aux.ReinterpretCast<int32_t>(), buffers.idx, buffers.aux.ReinterpretCast<int32_t>(), len);
    PipeBarrier<PIPE_V>();
    Adds(buffers.aux.ReinterpretCast<int32_t>(), buffers.aux.ReinterpretCast<int32_t>(), int32_t(E * 4U), len);
    PipeBarrier<PIPE_V>();
    Gather(buffers.stage, buffers.re, buffers.aux.ReinterpretCast<uint32_t>(), 0U, len);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(ws[batch * S + 4U * E], buffers.stage, output);
    cblas3::SyncAfterChunkStore();
}

template <uint32_t P>
__aicore__ inline void CgbtPackedSplitGroup(
    CgbtPackedSplitBuffers& buffers, GM_ADDR aarray, GM_ADDR barray, GlobalTensor<float> ws,
    const CgbtFusedTilingData& t, const CgbtSplitTilingData& split, bool aSide, uint32_t batch, uint32_t end)
{
    constexpr uint32_t E = P * P, S = 8U * E + 8U, GROUP = 4096U / E;

    const uint32_t count = end - batch < GROUP ? end - batch : GROUP;
    const uint32_t len = 2U * E * count;
    for (uint32_t q = 0; q < count; ++q) {
        GlobalTensor<float> input;
        input.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(CgbtBatchPtr(aSide ? aarray : barray, batch + q, t.gemm.batchCount)));
        auto qr = buffers.re[q * 2U * E], qi = buffers.re[q * 2U * E + E];
        CgbtVectorLoad<P>(input, split.rows, split.cols, split.ld, buffers.stage, qr, qi);
        if (split.negateImag) {
            Muls(qi, qi, -1.0F, E);
            PipeBarrier<PIPE_V>();
        }
    }
    Abs(buffers.aux, buffers.re, len);
    PipeBarrier<PIPE_V>();
    ReduceMax(buffers.range, buffers.aux, buffers.stage, len, false);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    const float largest = buffers.range.GetValue(0);
    Duplicate(buffers.range, largest, count * 8U);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const DataCopyExtParams flagCopy{static_cast<uint16_t>(count), 4U, 0U, (S - 1U) * 4U, 0U};
    DataCopyPad(ws[batch * S + 8U * E + (aSide ? 0U : 1U)], buffers.range, flagCopy);
    cblas3::SyncAfterChunkStore();
    Gather(buffers.stage, buffers.re, buffers.idx.ReinterpretCast<uint32_t>(), 0U, len);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const DataCopyExtParams output{static_cast<uint16_t>(count), 2U * E * 4U, 0U, (S - 2U * E) * 4U, 0U};
    DataCopyPad(ws[batch * S + (aSide ? 2U * E : 0U)], buffers.stage, output);
    cblas3::SyncAfterChunkStore();
    if (aSide) {
        CgbtPackedSplitImaginary<P>(buffers, ws, output, batch, count, len);
    }
}

template <uint32_t P>
__aicore__ inline void CgbtPackedSplit(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR workspace, const CgbtFusedTilingData& t, uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P, GROUP = 4096U / E, COUNT = E * GROUP;

    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (8U * COUNT + 8U * GROUP) * 4U);
    auto all = storage.Get<float>();
    auto stage = all, re = all[2U * COUNT], im = all[2U * COUNT + E];
    auto idx = all[4U * COUNT].ReinterpretCast<int32_t>();
    auto aux = all[6U * COUNT];
    auto range = all[8U * COUNT];
    const bool aSide = GetSubBlockIdx() == 0U;
    const auto& split = aSide ? t.a : t.b;
    const bool trans = aSide ? (t.gemm.transMode & CGBT_TRANS_Y) : (t.gemm.transMode & CGBT_TRANS_X);
    CgbtPackedSplitBuffers buffers{stage, re, idx, aux, range};
    CgbtPackedSplitIndex<P>(buffers, trans);
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    for (uint32_t batch = begin; batch < end; batch += GROUP) {
        CgbtPackedSplitGroup<P>(buffers, aarray, barray, ws, t, split, aSide, batch, end);
    }
}

template <uint32_t P>
__aicore__ inline void CgbtPackedCubeParams(Nd2NzParams& ca, Nd2NzParams& cb, MmadParams& mm, FixpipeParamsV220& fix)
{
    constexpr uint32_t E = P * P, S = 8U * E + 8U;
    ca.ndNum = cb.ndNum = 1U;
    ca.nValue = P;
    ca.dValue = 2U * P;
    ca.srcDValue = 2U * P;
    ca.dstNzC0Stride = P;
    ca.dstNzNStride = 1U;
    cb.nValue = cb.dValue = cb.srcDValue = cb.dstNzC0Stride = 2U * P;
    cb.dstNzNStride = 1U;
    mm.m = P;
    mm.n = mm.k = 2U * P;
    mm.cmatrixInitVal = true;
    fix.mSize = P;
    fix.nSize = 2U * P;
    fix.srcStride = P;
    fix.dstStride = 2U * P;
    fix.ndNum = 1U;
    ca.srcNdMatrixStride = cb.srcNdMatrixStride = S;
    ca.dstNzMatrixStride = 2U * E;
    cb.dstNzMatrixStride = 4U * E;
    fix.srcNdStride = 2U * E / 256U;
    fix.dstNdStride = S;
}

template <uint32_t P>
__aicore__ inline void CgbtPackedCube(GM_ADDR workspace, uint32_t begin, uint32_t end)
{
    constexpr uint32_t E = P * P, S = 8U * E + 8U, GROUP = 4096U / E;
    InitSocState();
    SetHF32Mode(HF32Mode::DISABLE);
    LocalTensor<float> a1(TPosition::A1, 0U, 2U * E * GROUP), b1(TPosition::B1, 2U * E * GROUP * 4U, 4U * E * GROUP);
    LocalTensor<float> a2(TPosition::A2, 0U, 2U * E * GROUP), b2(TPosition::B2, 0U, 4U * E * GROUP),
        c(TPosition::CO1, 0U, 2U * E * GROUP);
    Nd2NzParams ca{}, cb{};
    MmadParams mm{};
    FixpipeParamsV220 fix{};
    CgbtPackedCubeParams<P>(ca, cb, mm, fix);
    auto la = CgbtLocalLoad(P, 2U * P, false), lb = CgbtLocalLoad(2U * P, 2U * P, true);
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    for (uint32_t batch = begin; batch < end; batch += GROUP) {
        const uint32_t count = end - batch < GROUP ? end - batch : GROUP;
        ca.ndNum = cb.ndNum = fix.ndNum = count;
        DataCopy(a1, ws[batch * S], ca);
        DataCopy(b1, ws[batch * S + 2U * E], cb);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        for (uint32_t q = 0; q < count; ++q) {
            LoadData(a2[q * 2U * E], a1[q * 2U * E], la);
            LoadData(b2[q * 4U * E], b1[q * 4U * E], lb);
        }
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        for (uint32_t q = 0; q < count; ++q)
            Mmad(c[q * 2U * E], a2[q * 2U * E], b2[q * 4U * E], mm);
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        Fixpipe(ws[batch * S + 6U * E], c, fix);
        PipeBarrier<PIPE_ALL>();
    }
}

struct CgbtPackedPostState {
    LocalTensor<float> stage, re, im, x, y, z, w;
    LocalTensor<int32_t> idx;
    LocalTensor<float> range;
    LocalTensor<uint64_t> pointers;
    GlobalTensor<float> ws;
    GlobalTensor<uint64_t> ptrs;
};

template <uint32_t P>
__aicore__ inline void CgbtPackedPostReference(
    CgbtPackedPostState& state, GM_ADDR aarray, GM_ADDR barray, GlobalTensor<float> c, LocalTensor<float> qr,
    LocalTensor<float> qi, const CgbtFusedTilingData& t, uint32_t batch)
{
    GlobalTensor<float> a, b;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(aarray, batch, t.gemm.batchCount)));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(barray, batch, t.gemm.batchCount)));
    uint32_t ta = (t.gemm.transMode & CGBT_TRANS_Y) ? (t.a.negateImag ? 2U : 1U) : 0U;
    uint32_t tb = (t.gemm.transMode & CGBT_TRANS_X) ? (t.b.negateImag ? 2U : 1U) : 0U;
    const CgbtDirectTilingData direct{
        t.combine.m,
        t.combine.n,
        t.gemm.k,
        t.a.ld,
        t.b.ld,
        t.combine.ldc,
        t.gemm.batchCount,
        ta,
        tb,
        t.combine.alphaRe,
        t.combine.alphaIm,
        t.combine.betaRe,
        t.combine.betaIm};
    for (uint32_t col = 0; col < t.combine.n; ++col)
        for (uint32_t row = 0; row < t.combine.m; ++row) {
            auto v = CgbtDirectValue(direct, a, b, c, row, col);
            qr.SetValue(col * P + row, v.real);
            qi.SetValue(col * P + row, v.imag);
        }
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P>
__aicore__ inline void CgbtPackedPostBeta(
    CgbtPackedPostState& state, GlobalTensor<float> c, LocalTensor<float> qr, LocalTensor<float> qi,
    const CgbtFusedTilingData& t)
{
    constexpr uint32_t E = P * P;
    CgbtVectorLoad<P>(c, t.combine.m, t.combine.n, t.combine.ldc, state.stage, state.x, state.y);
    if (t.combine.preserveBetaOne) {
        Add(qr, qr, state.x, E);
        Add(qi, qi, state.y, E);
    } else {
        Muls(state.z, state.x, t.combine.betaRe, E);
        Muls(state.w, state.y, t.combine.betaIm, E);
        PipeBarrier<PIPE_V>();
        Sub(state.z, state.z, state.w, E);
        PipeBarrier<PIPE_V>();
        Add(qr, qr, state.z, E);
        Muls(state.z, state.x, t.combine.betaIm, E);
        Muls(state.w, state.y, t.combine.betaRe, E);
        PipeBarrier<PIPE_V>();
        Add(state.z, state.z, state.w, E);
        PipeBarrier<PIPE_V>();
        Add(qi, qi, state.z, E);
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t P>
__aicore__ inline void CgbtPackedPostStore(
    CgbtPackedPostState& state, const CgbtFusedTilingData& t, uint32_t count, uint32_t len)
{
    constexpr uint32_t E = P * P, GROUP = 4096U / E, COUNT = E * GROUP;
    // Scratch can now hold the interleave mapping. Preserve extraction
    // indices for the next group.
    cblas3::BuildInterleaveIndex(
        state.x.ReinterpretCast<int32_t>(), state.stage.ReinterpretCast<int32_t>(), state.z.ReinterpretCast<int32_t>(),
        len, COUNT);
    PipeBarrier<PIPE_V>();
    Gather(state.stage, state.re, state.x.ReinterpretCast<uint32_t>(), 0U, 2U * len);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    bool contiguous = t.combine.m == P && t.combine.n == P && t.combine.ldc == P;
    for (uint32_t q = 1; q < count; ++q)
        contiguous &= state.pointers.GetValue(q) == state.pointers.GetValue(0) + q * E * 8U;
    if (contiguous) {
        GlobalTensor<float> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state.pointers.GetValue(0)));
        DataCopyPad(c, state.stage, DataCopyExtParams{1U, len * 8U, 0U, 0U, 0U});
    } else {
        for (uint32_t q = 0; q < count; ++q) {
            GlobalTensor<float> c;
            c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state.pointers.GetValue(q)));
            const uint32_t bytes = t.combine.m * 8U;
            DataCopyPad(
                c, state.stage[q * E * 2U],
                DataCopyExtParams{
                    static_cast<uint16_t>(t.combine.n), bytes, (P * 8U - cblas3::AlignUpU(bytes, 32U)) / 32U,
                    (t.combine.ldc - t.combine.m) * 8U, 0U});
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P>
__aicore__ inline void CgbtPackedPostLoad(
    CgbtPackedPostState& state, const CgbtFusedTilingData& t, uint32_t first, uint32_t count, uint32_t len)
{
    constexpr uint32_t E = P * P, S = 8U * E + 8U;
    DataCopyPad(
        state.range, state.ws[first * S + 8U * E],
        DataCopyExtParams{static_cast<uint16_t>(count), 8U, (S - 2U) * 4U, 0U, 0U},
        DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
    DataCopyPad(
        state.pointers, state.ptrs[first], DataCopyExtParams{1U, count * 8U, 0U, 0U, 0U},
        DataCopyPadExtParams<uint64_t>{false, 0U, 0U, 0U});
    DataCopyPad(
        state.stage, state.ws[first * S + 6U * E],
        DataCopyExtParams{static_cast<uint16_t>(count), 2U * E * 4U, (S - 2U * E) * 4U, 0U, 0U},
        DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Gather(state.re, state.stage, state.idx.ReinterpretCast<uint32_t>(), 0U, len);
    Gather(state.im, state.stage, state.idx.ReinterpretCast<uint32_t>(), P * 4U, len);
    PipeBarrier<PIPE_V>();
    if (t.combine.alphaRe != 1.0F || t.combine.alphaIm != 0.0F) {
        Muls(state.x, state.re, t.combine.alphaRe, len);
        Muls(state.y, state.im, t.combine.alphaIm, len);
        Muls(state.z, state.im, t.combine.alphaRe, len);
        Muls(state.w, state.re, t.combine.alphaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(state.re, state.x, state.y, len);
        Add(state.im, state.z, state.w, len);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_ALL>();
}

template <uint32_t P>
__aicore__ inline void CgbtPackedPostGroup(
    CgbtPackedPostState& state, GM_ADDR aarray, GM_ADDR barray, const CgbtFusedTilingData& t, uint32_t first,
    uint32_t end)
{
    constexpr uint32_t E = P * P, GROUP = 4096U / E;
    const uint32_t count = end - first < GROUP ? end - first : GROUP, len = count * E;
    CgbtPackedPostLoad<P>(state, t, first, count, len);
    for (uint32_t q = 0; q < count; ++q) {
        const uint32_t batch = first + q;
        GlobalTensor<float> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state.pointers.GetValue(q)));
        auto qr = state.re[q * E], qi = state.im[q * E];
        if (!(state.range.GetValue(q * 8U) * t.combine.inputRiskScale <= 3.402823466e38F &&
              state.range.GetValue(q * 8U + 1U) * t.combine.inputRiskScale <= 3.402823466e38F)) {
            CgbtPackedPostReference<P>(state, aarray, barray, c, qr, qi, t, batch);
        } else if (!t.combine.isBetaZero) {
            CgbtPackedPostBeta<P>(state, c, qr, qi, t);
            // beta*C can introduce nonfinite values even when A/B are finite.
            Add(state.x, qr, qi, E);
            PipeBarrier<PIPE_V>();
            ReduceSum(state.z, state.x, state.y, E);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float sum = state.z.GetValue(0);
            if (!(sum <= 3.402823466e38F && sum >= -3.402823466e38F))
                CgbtPackedPostReference<P>(state, aarray, barray, c, qr, qi, t, batch);
        }
    }
    CgbtPackedPostStore<P>(state, t, count, len);
}

template <uint32_t P>
__aicore__ inline void CgbtPackedPost(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, const CgbtFusedTilingData& t, uint32_t begin,
    uint32_t end)
{
    constexpr uint32_t E = P * P, GROUP = 4096U / E, COUNT = E * GROUP;
    constexpr int32_t L = P == 16U ? 4 : (P == 32U ? 5 : 6);
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (10U * COUNT + 16U * GROUP) * 4U);
    auto all = storage.Get<float>(), stage = all, re = all[2U * COUNT], im = all[3U * COUNT];
    auto x = all[4U * COUNT], y = all[5U * COUNT], z = all[6U * COUNT], w = all[7U * COUNT];
    auto idx = all[8U * COUNT].ReinterpretCast<int32_t>();
    auto range = all[10U * COUNT];
    auto pointers = all[10U * COUNT + 8U * GROUP].ReinterpretCast<uint64_t>();
    // Both mapping tables are built once and reused for all batches.
    CreateVecIndex(idx, int32_t(0), COUNT);
    PipeBarrier<PIPE_V>();
    ShiftRight(x.ReinterpretCast<int32_t>(), idx, L, COUNT);
    PipeBarrier<PIPE_V>();
    ShiftLeft(x.ReinterpretCast<int32_t>(), x.ReinterpretCast<int32_t>(), L, COUNT);
    PipeBarrier<PIPE_V>();
    Add(idx, idx, x.ReinterpretCast<int32_t>(), COUNT);
    PipeBarrier<PIPE_V>();
    ShiftLeft(idx, idx, 2, COUNT);
    PipeBarrier<PIPE_V>();
    // Extract the output directly into contiguous [real | imag] planes.
    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    GlobalTensor<uint64_t> ptrs;
    ptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(carray));
    CgbtPackedPostState state{stage, re, im, x, y, z, w, idx, range, pointers, ws, ptrs};
    for (uint32_t first = begin + GetSubBlockIdx() * GROUP; first < end; first += 2U * GROUP) {
        CgbtPackedPostGroup<P>(state, aarray, barray, t, first, end);
    }
}

template <uint32_t P>
__aicore__ inline void CgbtPackedBody(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CgbtFusedTilingData& t)
{
    uint32_t owner;
    if ASCEND_IS_AIC {
        owner = GetBlockIdx();
    } else {
        owner = GetBlockIdx() / 2U;
    }
    // Balanced contiguous ownership keeps every available core busy when
    // batchCount >= workers, with at most one matrix of load imbalance.
    const uint32_t workers = GetBlockNum();
    const uint32_t begin = static_cast<uint64_t>(owner) * t.gemm.batchCount / workers;
    const uint32_t end = static_cast<uint64_t>(owner + 1U) * t.gemm.batchCount / workers;
    if (begin >= end)
        return;
    if ASCEND_IS_AIV {
        CgbtPackedSplit<P>(a, b, ws, t, begin, end);
        CrossCoreSetFlag<2, PIPE_MTE3>(0);
        CrossCoreWaitFlag<2, PIPE_MTE2>(1);
        CgbtPackedPost<P>(a, b, c, ws, t, begin, end);
    } else {
        CrossCoreWaitFlag<2, PIPE_MTE2>(0);
        CgbtPackedCube<P>(ws, begin, end);
        CrossCoreSetFlag<2, PIPE_FIX>(1);
    }
}

__aicore__ __attribute__((noinline)) void CgbtGemmBody(GM_ADDR workspace, const CgbtGemmTilingData& t)
{
    // The four (X, Y) orientation pairs need four MatmulType specialisations;
    // transMode folds the runtime flags into a compile-time dispatch.
    const uint32_t mode = t.transMode & (CGBT_TRANS_X | CGBT_TRANS_Y);
    if (t.tileSize == 176U) {
        if (mode == 3U)
            CgbtGemmBodyT<CgbtMidMM<true, true>, true, true, 176>(workspace, t);
        else if (mode == 1U)
            CgbtGemmBodyT<CgbtMidMM<true, false>, true, false, 176>(workspace, t);
        else if (mode == 2U)
            CgbtGemmBodyT<CgbtMidMM<false, true>, false, true, 176>(workspace, t);
        else
            CgbtGemmBodyT<CgbtMidMM<false, false>, false, false, 176>(workspace, t);
        return;
    }
    if (t.tileSize == 256U) {
        if (mode == 3U)
            CgbtGemmBodyT<CgbtReuseMM<true, true>, true, true, 256>(workspace, t);
        else if (mode == 1U)
            CgbtGemmBodyT<CgbtReuseMM<true, false>, true, false, 256>(workspace, t);
        else if (mode == 2U)
            CgbtGemmBodyT<CgbtReuseMM<false, true>, false, true, 256>(workspace, t);
        else
            CgbtGemmBodyT<CgbtReuseMM<false, false>, false, false, 256>(workspace, t);
        return;
    }
    if (t.tileSize == 320U) {
        if (mode == 3U)
            CgbtGemmBodyT<CgbtReuse320MM<true, true>, true, true, 320>(workspace, t);
        else if (mode == 1U)
            CgbtGemmBodyT<CgbtReuse320MM<true, false>, true, false, 320>(workspace, t);
        else if (mode == 2U)
            CgbtGemmBodyT<CgbtReuse320MM<false, true>, false, true, 320>(workspace, t);
        else
            CgbtGemmBodyT<CgbtReuse320MM<false, false>, false, false, 320>(workspace, t);
        return;
    }

    if (mode == (CGBT_TRANS_X | CGBT_TRANS_Y)) {
        CgbtGemmBodyT<MatmulBothTrans, true, true>(workspace, t);
    } else if (mode == CGBT_TRANS_X) {
        CgbtGemmBodyT<cblas3::MatmulTransLeft, true, false>(workspace, t);
    } else if (mode == CGBT_TRANS_Y) {
        CgbtGemmBodyT<cblas3::MatmulTransRight, false, true>(workspace, t);
    } else {
        CgbtGemmBodyT<cblas3::MatmulPlainBoth, false, false>(workspace, t);
    }
}

__aicore__ inline bool CgbtUseTinyVector(const CgbtDirectTilingData& tiling)
{
    return tiling.m == 8U && tiling.n == tiling.m && tiling.k == tiling.m && tiling.lda == tiling.m &&
           tiling.ldb == tiling.m && tiling.ldc == tiling.m && tiling.transA == 0U && tiling.transB == 0U &&
           tiling.alphaRe == 1.0F && tiling.alphaIm == 0.0F && tiling.betaRe == 0.0F && tiling.betaIm == 0.0F;
}

} // namespace

// ---------------------------------------------------------------------------
//  Phase 2: batched combine
// ---------------------------------------------------------------------------
namespace {

// UB buffers and GM bindings for one combine kernel. TPipe stays a local of the
// kernel entry (rule R3); the TBufs live in the caller's frame so the
// LocalTensors handed to the ctx outlive every chunk.
struct CgbtCombineCtx {
    GlobalTensor<float> t1Gm;
    GlobalTensor<float> t2Gm;
    GlobalTensor<float> t3Gm;
    GlobalTensor<float> t4Gm;
    GlobalTensor<float> cGm;
    GM_ADDR aarray = nullptr;
    GM_ADDR barray = nullptr;
    CgbtDirectTilingData direct{};
    bool forceReference = false;
    bool compactOutput = false;
    LocalTensor<float> crci; // [re | im] halves, interleaved by Gather on the way out
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> pr;
    LocalTensor<float> pi;
    LocalTensor<float> tmpA;
    LocalTensor<float> tmpB;
    LocalTensor<float> stage;
    LocalTensor<uint32_t> idxU;
    LocalTensor<float> inputFlags;
};

struct CgbtCombineBuffers {
    TBuf<TPosition::VECCALC> crci;
    TBuf<TPosition::VECCALC> pr;
    TBuf<TPosition::VECCALC> pi;
    TBuf<TPosition::VECCALC> tmpA;
    TBuf<TPosition::VECCALC> tmpB;
    TBuf<TPosition::VECCALC> stage;
    TBuf<TPosition::VECCALC> idx;
    TBuf<TPosition::VECCALC> inputFlags;
};

__aicore__ inline void CgbtBuildCombineIndex(
    CgbtCombineBuffers& bufs, CgbtCombineCtx& ctx, const CgbtCombineTilingData& tiling)
{
    constexpr uint32_t chunk = COMBINE_CHUNK;
    constexpr uint32_t pairElems = chunk * cblas3::COMPLEX_ELENUM;
    LocalTensor<int32_t> idx = bufs.idx.Get<int32_t>();
    ctx.compactOutput = tiling.ldc == tiling.m && tiling.ldD != tiling.m && tiling.ldD <= chunk;
    if (ctx.compactOutput) {
        auto lane = ctx.stage.ReinterpretCast<int32_t>();
        auto scratch = ctx.crci.ReinterpretCast<int32_t>();
        CreateVecIndex(idx, 0, pairElems);
        PipeBarrier<PIPE_V>();
        ShiftRight(lane, idx, 1, pairElems);
        PipeBarrier<PIPE_V>();
        Muls(scratch, lane, 2, pairElems);
        PipeBarrier<PIPE_V>();
        Sub(idx, idx, scratch, pairElems);
        Cast(ctx.crci, lane, RoundMode::CAST_NONE, pairElems);
        PipeBarrier<PIPE_V>();
        Muls(idx, idx, static_cast<int32_t>(chunk * sizeof(float)), pairElems);
        const float rows = static_cast<float>(static_cast<int32_t>(tiling.m));
        Muls(ctx.crci, ctx.crci, 1.0F / rows, pairElems);
        PipeBarrier<PIPE_V>();
        Adds(ctx.crci, ctx.crci, 0.0001F, pairElems);
        PipeBarrier<PIPE_V>();
        Cast(scratch, ctx.crci, RoundMode::CAST_FLOOR, pairElems);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, static_cast<int32_t>(tiling.ldD - tiling.m), pairElems);
        PipeBarrier<PIPE_V>();
        Add(lane, lane, scratch, pairElems);
        PipeBarrier<PIPE_V>();
        Muls(lane, lane, static_cast<int32_t>(sizeof(float)), pairElems);
        PipeBarrier<PIPE_V>();
        Add(idx, idx, lane, pairElems);
        PipeBarrier<PIPE_V>();
    } else {
        cblas3::BuildInterleaveIndex(
            idx, ctx.stage.ReinterpretCast<int32_t>(), ctx.crci.ReinterpretCast<int32_t>(), chunk, chunk);
    }
    ctx.idxU = idx.ReinterpretCast<uint32_t>();
    // Index construction uses stage as scratch. Prevent the next MTE2 load
    // from overwriting it before the vector index operations have completed.
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

__aicore__ inline void CgbtInitCombineCtx(
    TPipe& pipe, CgbtCombineBuffers& bufs, CgbtCombineCtx& ctx, GM_ADDR workspace, const CgbtCombineTilingData& tiling,
    uint32_t inputFlagCount = 0U)
{
    ctx.t1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), tiling.wsFloats);
    ctx.t2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), tiling.wsFloats);
    ctx.t3Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), tiling.wsFloats);
    ctx.t4Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), tiling.wsFloats);

    constexpr uint32_t chunk = COMBINE_CHUNK;
    constexpr uint32_t pairElems = chunk * cblas3::COMPLEX_ELENUM;

    pipe.InitBuffer(bufs.crci, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.pr, chunk * sizeof(float));
    pipe.InitBuffer(bufs.pi, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpA, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpB, chunk * sizeof(float));
    pipe.InitBuffer(bufs.stage, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.idx, pairElems * sizeof(int32_t));

    ctx.crci = bufs.crci.Get<float>();
    ctx.cr = ctx.crci;
    ctx.ci = ctx.crci[chunk];
    ctx.pr = bufs.pr.Get<float>();
    ctx.pi = bufs.pi.Get<float>();
    ctx.tmpA = bufs.tmpA.Get<float>();
    ctx.tmpB = bufs.tmpB.Get<float>();
    ctx.stage = bufs.stage.Get<float>();

    if (inputFlagCount != 0U) {
        // Separate storage lets the small flag DMA overlap index construction;
        // the later product loads never overwrite this vector-status input.
        pipe.InitBuffer(bufs.inputFlags, 1024U);
        ctx.inputFlags = bufs.inputFlags.Get<float>();
        GlobalTensor<float> flags;
        flags.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace) + tiling.offT4 + tiling.tPlaneFloats * tiling.batchCount);
        DataCopyPad(
            ctx.inputFlags, flags, DataCopyExtParams{1U, inputFlagCount * 4U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    CgbtBuildCombineIndex(bufs, ctx, tiling);
}

// Matmul only writes logical rows. Never let workspace or C padding enter
// arithmetic: even a discarded NaN lane can set the sticky FP status flag.
__aicore__ inline void CgbtClearCombinePadding(
    LocalTensor<float> values, const CgbtCombineTilingData& tiling, uint32_t columns, uint32_t components)
{
    if (columns == 0U || tiling.ldD == tiling.m)
        return;
    // Duplicate requires a 32-byte aligned destination. Mask the first block
    // so logical rows sharing that block are preserved (at most 62 pad lanes).
    const uint32_t first = tiling.m * components;
    const uint32_t aligned = first / 8U * 8U;
    const uint32_t count = (tiling.ldD - tiling.m) * components;
    uint64_t mask[2] = {(~uint64_t{0} >> (64U - count)) << (first - aligned), 0U};
    const uint32_t stride = tiling.ldD * components / 8U;
    // A repeat clears the same masked lanes in the next column. The hardware
    // fields are uint8_t (stride in 32-byte blocks); use one repeat for long rows.
    const uint32_t maxRepeat = stride <= 255U ? 255U : 1U;
    for (uint32_t column = 0U; column < columns;) {
        const uint32_t repeats = columns - column < maxRepeat ? columns - column : maxRepeat;
        Duplicate(
            values[column * tiling.ldD * components + aligned], 0.0F, mask, static_cast<uint8_t>(repeats), 1U,
            static_cast<uint8_t>(stride <= 255U ? stride : 0U));
        column += repeats;
    }
}

// Pr = T1 - T2, Pi = T3 + T4, then complex alpha for ordinary finite
// arithmetic. Repair exceptional inputs/outputs in reference order.
__aicore__ inline void CgbtApplyAlpha(
    CgbtCombineCtx& ctx, const CgbtCombineTilingData& tiling, uint64_t tempBase, uint32_t len, uint32_t columns)
{
    DataCopyExtParams params{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};

    DataCopyPad(ctx.pr, ctx.t1Gm[tempBase + tiling.offT1], params, padParams);
    DataCopyPad(ctx.pi, ctx.t3Gm[tempBase + tiling.offT3], params, padParams);
    DataCopyPad(ctx.tmpA, ctx.t2Gm[tempBase + tiling.offT2], params, padParams);
    DataCopyPad(ctx.tmpB, ctx.t4Gm[tempBase + tiling.offT4], params, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    CgbtClearCombinePadding(ctx.pr, tiling, columns, 1U);
    CgbtClearCombinePadding(ctx.pi, tiling, columns, 1U);
    CgbtClearCombinePadding(ctx.tmpA, tiling, columns, 1U);
    CgbtClearCombinePadding(ctx.tmpB, tiling, columns, 1U);
    PipeBarrier<PIPE_V>();

    // For identity alpha, form the complex result in its final UB buffers.
    // This avoids two unit multiplies and an intermediate vector dependency.
    if (tiling.alphaRe == 1.0F && tiling.alphaIm == 0.0F) {
        Sub(ctx.cr, ctx.pr, ctx.tmpA, len);
        Add(ctx.ci, ctx.pi, ctx.tmpB, len);
        PipeBarrier<PIPE_V>();
        return;
    }

    Sub(ctx.pr, ctx.pr, ctx.tmpA, len);
    Add(ctx.pi, ctx.pi, ctx.tmpB, len);
    PipeBarrier<PIPE_V>();

    if (tiling.alphaRe != 0.0F) {
        Muls(ctx.cr, ctx.pr, tiling.alphaRe, len);
        Muls(ctx.ci, ctx.pi, tiling.alphaRe, len);
    } else {
        Duplicate(ctx.cr, 0.0F, len);
        Duplicate(ctx.ci, 0.0F, len);
    }
    if (tiling.alphaIm != 0.0F) {
        Muls(ctx.tmpA, ctx.pi, tiling.alphaIm, len);
        Muls(ctx.tmpB, ctx.pr, tiling.alphaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.cr, ctx.cr, ctx.tmpA, len);
        Add(ctx.ci, ctx.ci, ctx.tmpB, len);
    }
    PipeBarrier<PIPE_V>();
}

// Adds beta * C_old. beta is complex, so the old value contributes to both
// parts; the de-interleave reuses the shared GatherMask pass over one complex
// run.
__aicore__ inline void CgbtApplyBeta(
    CgbtCombineCtx& ctx, const CgbtCombineTilingData& tiling, uint64_t cOffset, uint32_t len, uint32_t columns = 0U)
{
    uint64_t rsvdCnt = 0;
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    DataCopyExtParams cParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    if (columns != 0U) {
        const uint32_t alignedBytes = (tiling.m * 8U + 31U) / 32U * 32U;
        cParams = {
            static_cast<uint16_t>(columns), tiling.m * 8U, (tiling.ldc - tiling.m) * 8U,
            (tiling.ldD * 8U - alignedBytes) / 32U, 0U};
    }
    DataCopyPad(ctx.stage, ctx.cGm[cOffset], cParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    CgbtClearCombinePadding(ctx.stage, tiling, columns, cblas3::COMPLEX_ELENUM);
    PipeBarrier<PIPE_V>();

    const uint16_t repeats = static_cast<uint16_t>(
        cblas3::CeilDivU(len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), cblas3::BYTENUM_REPEAT));
    GatherMask(ctx.tmpA, ctx.stage, 1, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    GatherMask(ctx.tmpB, ctx.stage, 2, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    if (tiling.preserveBetaOne != 0U) {
        Add(ctx.cr, ctx.cr, ctx.tmpA, len);
        Add(ctx.ci, ctx.ci, ctx.tmpB, len);
    } else {
        // Retain zero components: skipping them changes Inf/NaN classification.
        Muls(ctx.pr, ctx.tmpA, tiling.betaRe, len);
        Muls(ctx.pi, ctx.tmpB, tiling.betaRe, len);
        PipeBarrier<PIPE_V>();
        Muls(ctx.tmpA, ctx.tmpA, tiling.betaIm, len);
        Muls(ctx.tmpB, ctx.tmpB, tiling.betaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.pr, ctx.pr, ctx.tmpB, len);
        Add(ctx.pi, ctx.pi, ctx.tmpA, len);
        PipeBarrier<PIPE_V>();
        Add(ctx.cr, ctx.cr, ctx.pr, len);
        Add(ctx.ci, ctx.ci, ctx.pi, len);
    }
    PipeBarrier<PIPE_V>();
}

// Kept out of the normal epilogue so exceptional scalar arithmetic does not
// increase register lifetimes in the vector path. C_old has not been stored yet.
__aicore__ __attribute__((noinline)) void CgbtRepairNonfinite(
    CgbtCombineCtx& ctx, const CgbtCombineTilingData& t, uint32_t batch, uint32_t col, uint32_t row, uint32_t len,
    uint32_t columns)
{
    GlobalTensor<float> a;
    GlobalTensor<float> b;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(ctx.aarray, batch, t.batchCount)));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(CgbtBatchPtr(ctx.barray, batch, t.batchCount)));
    auto bits = ctx.stage.ReinterpretCast<uint32_t>();
    for (uint32_t i = 0U; i < len; ++i) {
        const uint32_t stride = ctx.compactOutput ? t.m : t.ldD;
        const uint32_t rr = columns != 0U ? i % stride : (row + i) % t.m;
        const uint32_t cc = col + (columns != 0U ? i / stride : (row + i) / t.m);
        if (rr >= t.m || (!ctx.forceReference && ((bits.GetValue(2U * i) & 0x7fffffffU) < 0x7f800000U &&
                                                  (bits.GetValue(2U * i + 1U) & 0x7fffffffU) < 0x7f800000U)))
            continue;
        const auto value = CgbtDirectValue(ctx.direct, a, b, ctx.cGm, rr, cc);
        ctx.stage.SetValue(2U * i, value.real);
        ctx.stage.SetValue(2U * i + 1U, value.imag);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
}

template <bool CHECK_NONFINITE = false>
__aicore__ inline void CgbtCombineOneChunk(
    CgbtCombineCtx& ctx, const CgbtCombineTilingData& tiling, uint64_t batch, uint32_t col, uint32_t row, uint32_t len,
    uint32_t columns = 0U)
{
    const uint64_t tempBase = batch * tiling.tPlaneFloats + static_cast<uint64_t>(col) * tiling.ldD + row;
    const uint64_t cOffset = (static_cast<uint64_t>(col) * tiling.ldc + row) * cblas3::COMPLEX_ELENUM;

    CgbtApplyAlpha(ctx, tiling, tempBase, len, columns);

    if (tiling.isBetaZero == 0U) {
        CgbtApplyBeta(ctx, tiling, cOffset, len, columns);
    }

    const uint32_t outputLen = ctx.compactOutput && columns != 0U ? columns * tiling.m : len;
    Gather(ctx.stage, ctx.crci, ctx.idxU, 0U, outputLen * cblas3::COMPLEX_ELENUM);
    PipeBarrier<PIPE_V>();
    if constexpr (CHECK_NONFINITE) {
        // Addition propagates a nonfinite lane (including mixed-sign Inf),
        // unlike a max reduction which can discard NaNs. Finite sum overflow
        // may conservatively enter the scan; only nonfinite lanes are repaired.
        ReduceSum(ctx.pr, ctx.stage, ctx.tmpA, outputLen * 2U);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float sum = ctx.pr.GetValue(0);
        if (ctx.forceReference || !(sum <= 3.402823466e38F && sum >= -3.402823466e38F))
            CgbtRepairNonfinite(ctx, tiling, batch, col, row, outputLen, columns);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopyExtParams outParams{
        1U, outputLen * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    if (columns != 0U && !ctx.compactOutput) {
        const uint32_t alignedBytes = (tiling.m * 8U + 31U) / 32U * 32U;
        outParams = {
            static_cast<uint16_t>(columns), tiling.m * 8U, (tiling.ldD * 8U - alignedBytes) / 32U,
            (tiling.ldc - tiling.m) * 8U, 0U};
    }
    DataCopyPad(ctx.cGm[cOffset], ctx.stage, outParams);

    // Both queues must wait on MTE3 before the next iteration reuses `stage`.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <bool CHECK_NONFINITE>
__aicore__ inline void CgbtCombineJobs(
    CgbtCombineCtx& ctx, GM_ADDR carray, const CgbtCombineTilingData& tiling, uint32_t coreNum, uint64_t coreIdx)
{
    const bool flat = tiling.ldD == tiling.m && tiling.ldc == tiling.m;
    const uint64_t elements = static_cast<uint64_t>(tiling.m) * tiling.n;
    const uint32_t columnsPerJob = !flat && tiling.ldD <= COMBINE_CHUNK ? COMBINE_CHUNK / tiling.ldD : 0U;
    const uint64_t segments = flat ? (elements + COMBINE_CHUNK - 1U) / COMBINE_CHUNK :
                                     (columnsPerJob != 0U ? (tiling.n + columnsPerJob - 1U) / columnsPerJob : tiling.n);
    const uint64_t jobs = static_cast<uint64_t>(tiling.batchCount) * segments;

    // One scalar GM read per batch (see CgbtSplitFusedBody); the sentinel is a
    // batch value that never occurs.
    uint64_t lastBatch = static_cast<uint64_t>(tiling.batchCount);
    for (uint64_t job = coreIdx; job < jobs; job += coreNum) {
        const uint64_t batch = job / segments;
        const uint32_t col = static_cast<uint32_t>(job % segments);
        if (batch != lastBatch) {
            ctx.cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                static_cast<uintptr_t>(CgbtBatchPtr(carray, batch, tiling.batchCount))));
            lastBatch = batch;
        }
        if (flat) {
            const uint64_t offset = static_cast<uint64_t>(col) * COMBINE_CHUNK;
            const uint32_t len = elements - offset < COMBINE_CHUNK ? elements - offset : COMBINE_CHUNK;
            CgbtCombineOneChunk<CHECK_NONFINITE>(ctx, tiling, batch, 0U, offset, len);
            continue;
        }
        if (columnsPerJob != 0U) {
            const uint32_t colBase = col * columnsPerJob;
            const uint32_t columns = tiling.n - colBase < columnsPerJob ? tiling.n - colBase : columnsPerJob;
            CgbtCombineOneChunk<CHECK_NONFINITE>(ctx, tiling, batch, colBase, 0U, tiling.ldD * columns, columns);
            continue;
        }
        for (uint32_t off = 0; off < tiling.m; off += COMBINE_CHUNK) {
            const uint32_t len = (off + COMBINE_CHUNK > tiling.m) ? (tiling.m - off) : COMBINE_CHUNK;
            CgbtCombineOneChunk<CHECK_NONFINITE>(ctx, tiling, batch, col, off, len);
        }
    }
}

// Keep the exception loop and reference arithmetic out of the common loop's
// register allocation and instruction working set.
__aicore__ __attribute__((noinline)) void CgbtCorrectCombineJobs(
    CgbtCombineCtx& ctx, GM_ADDR carray, const CgbtCombineTilingData& tiling, uint32_t coreNum, uint64_t coreIdx)
{
    CgbtCombineJobs<true>(ctx, carray, tiling, coreNum, coreIdx);
}

// Broadcast the splitter's range exception through the sticky vector status.
// No scalar reduction is needed: any NaN flag survives subsequent vector work.
__aicore__ inline void CgbtLoadInputStatus(CgbtCombineCtx& ctx, uint32_t count)
{
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    Adds(ctx.inputFlags, ctx.inputFlags, 0.0F, count);
}

template <bool WAIT_FOR_CUBE = false>
__aicore__ inline void CgbtCombineBody(
    GM_ADDR workspace, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, const CgbtCombineTilingData& tiling,
    const CgbtDirectTilingData& direct, bool paired = false)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum()) * (paired ? 2U : 1U);
    const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
    if (coreNum == 0U || tiling.batchCount == 0U || tiling.n == 0U) {
        return;
    }
    TPipe pipe;
    CgbtCombineBuffers bufs;
    CgbtCombineCtx ctx;
    CgbtInitCombineCtx(pipe, bufs, ctx, workspace, tiling, paired ? coreNum : tiling.inputFlagCount);
    ctx.aarray = aarray;
    ctx.barray = barray;
    ctx.direct = direct;
    // Index generation and flag prefetch use only local storage and completed
    // splitter output. Hide them under the independent Cube products.
    if constexpr (WAIT_FOR_CUBE)
        SyncAll<false>();
    CgbtLoadInputStatus(ctx, paired ? coreNum : tiling.inputFlagCount);
    if (tiling.isBetaZero != 0U) {
        CgbtCombineJobs<false>(ctx, carray, tiling, coreNum, coreIdx);
        // STATUS retains FP exceptions until read, including input Inf/NaN.
        // Finish this core's vector work before reading it. Ordinary beta=0
        // output requires no extra vector pass; exceptional cores alone retry.
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (asc_get_status() != 0) {
            ctx.forceReference = true;
            CgbtCorrectCombineJobs(ctx, carray, tiling, coreNum, coreIdx);
        }
    } else {
        // With beta!=0, repair must precede the first store over C_old.
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        ctx.forceReference = asc_get_status() != 0;
        CgbtCorrectCombineJobs(ctx, carray, tiling, coreNum, coreIdx);
    }
}

// ---------------------------------------------------------------------------
//  Early exit: C[i] = beta * C[i], bit-exact
// ---------------------------------------------------------------------------
struct CgbtBetaScaleBuffers {
    TBuf<TPosition::VECCALC> stage;
    TBuf<TPosition::VECCALC> pair; // [re | im] halves, interleaved by Gather on the way out
    TBuf<TPosition::VECCALC> tmpA;
    TBuf<TPosition::VECCALC> tmpB;
    TBuf<TPosition::VECCALC> idx;
};

__aicore__ inline void CgbtBetaScaleValues(
    LocalTensor<float> stage, LocalTensor<float> cr, LocalTensor<float> ci, LocalTensor<float> tmpA,
    LocalTensor<float> tmpB, const CgbtBetaScaleTilingData& tiling, uint32_t len)
{
    uint64_t rsvdCnt = 0U;
    const uint16_t repeats = static_cast<uint16_t>(
        cblas3::CeilDivU(len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), cblas3::BYTENUM_REPEAT));
    GatherMask(cr, stage, 1, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    GatherMask(ci, stage, 2, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // newRe = bre*re - bim*im, newIm = bim*re + bre*im; each part is
    // one multiply and one add/sub, matching the reference formula
    // term for term so the path stays bit-exact.
    Muls(tmpA, cr, tiling.betaRe, len);
    Muls(tmpB, cr, tiling.betaIm, len);
    Muls(cr, ci, tiling.betaIm, len);
    PipeBarrier<PIPE_V>();
    Sub(cr, tmpA, cr, len);
    Muls(ci, ci, tiling.betaRe, len);
    PipeBarrier<PIPE_V>();
    Add(ci, tmpB, ci, len);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void CgbtBetaScaleChunk(
    CgbtBetaScaleBuffers& bufs, GlobalTensor<float> cGm, const CgbtBetaScaleTilingData& tiling, uint32_t col,
    uint32_t off)
{
    constexpr uint32_t chunk = COMBINE_CHUNK;
    auto stage = bufs.stage.Get<float>();
    auto pair = bufs.pair.Get<float>();
    auto cr = pair;
    auto ci = pair[chunk];
    auto tmpA = bufs.tmpA.Get<float>();
    auto tmpB = bufs.tmpB.Get<float>();
    auto idxU = bufs.idx.Get<int32_t>().ReinterpretCast<uint32_t>();
    const DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    const uint32_t len = (off + chunk > tiling.m) ? (tiling.m - off) : chunk;
    const uint64_t cOffset = (static_cast<uint64_t>(col) * tiling.ldc + off) * cblas3::COMPLEX_ELENUM;

    DataCopyExtParams cParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    if (tiling.isBetaZero != 0U) {
        Duplicate(cr, 0.0F, len);
        Duplicate(ci, 0.0F, len);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopyPad(stage, cGm[cOffset], cParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        CgbtBetaScaleValues(stage, cr, ci, tmpA, tmpB, tiling, len);
    }

    Gather(stage, pair, idxU, 0U, len * cblas3::COMPLEX_ELENUM);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopyPad(cGm[cOffset], stage, cParams);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline void CgbtBetaScaleBody(GM_ADDR carray, const CgbtBetaScaleTilingData& tiling)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U || tiling.batchCount == 0U || tiling.n == 0U) {
        return;
    }
    const uint64_t jobs = static_cast<uint64_t>(tiling.batchCount) * tiling.n;

    GlobalTensor<float> cGm;

    TPipe pipe;
    CgbtBetaScaleBuffers bufs;
    constexpr uint32_t chunk = COMBINE_CHUNK;
    constexpr uint32_t pairElems = chunk * cblas3::COMPLEX_ELENUM;
    pipe.InitBuffer(bufs.stage, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.pair, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.tmpA, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpB, chunk * sizeof(float));
    pipe.InitBuffer(bufs.idx, pairElems * sizeof(int32_t));

    auto stage = bufs.stage.Get<float>();
    auto pair = bufs.pair.Get<float>();
    LocalTensor<int32_t> idx = bufs.idx.Get<int32_t>();
    cblas3::BuildInterleaveIndex(idx, stage.ReinterpretCast<int32_t>(), pair.ReinterpretCast<int32_t>(), chunk, chunk);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

    // One scalar GM read per batch (see CgbtSplitFusedBody).
    uint64_t lastBatch = static_cast<uint64_t>(tiling.batchCount);
    for (uint64_t job = static_cast<uint64_t>(GetBlockIdx()); job < jobs; job += coreNum) {
        const uint64_t batch = job / tiling.n;
        const uint32_t col = static_cast<uint32_t>(job % tiling.n);
        if (batch != lastBatch) {
            cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                static_cast<uintptr_t>(CgbtBatchPtr(carray, batch, tiling.batchCount))));
            lastBatch = batch;
        }

        for (uint32_t off = 0; off < tiling.m; off += chunk) {
            CgbtBetaScaleChunk(bufs, cGm, tiling, col, off);
        }
    }
}

} // namespace

// ===========================================================================
//  Kernel entries
// ===========================================================================
extern "C" __global__ __aicore__ __schedmode__(1) void cgemm_batched_panel_kernel(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, CgbtFusedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    const uint32_t extent = tiling.gemm.m > tiling.gemm.n ? tiling.gemm.m : tiling.gemm.n;
    const uint32_t largest = extent > tiling.gemm.k ? extent : tiling.gemm.k;
    if (largest <= 16U)
        CgbtPackedBody<16>(aarray, barray, carray, workspace, tiling);
    else if (largest <= 32U)
        CgbtPackedBody<32>(aarray, barray, carray, workspace, tiling);
    else
        CgbtPackedBody<64>(aarray, barray, carray, workspace, tiling);
}

__aicore__ inline CgbtDirectTilingData CgbtReferenceTiling(const CgbtFusedTilingData& t)
{
    return {
        t.combine.m,
        t.combine.n,
        t.gemm.k,
        t.a.ld,
        t.b.ld,
        t.combine.ldc,
        t.combine.batchCount,
        (t.gemm.transMode & CGBT_TRANS_Y) ? (t.a.negateImag ? 2U : 1U) : 0U,
        (t.gemm.transMode & CGBT_TRANS_X) ? (t.b.negateImag ? 2U : 1U) : 0U,
        t.combine.alphaRe,
        t.combine.alphaIm,
        t.combine.betaRe,
        t.combine.betaIm};
}

// Select each Matmul storage plan, including operand orientation. Preparation,
// synchronization, exceptional-value handling and writeback share one pipeline.
template <uint32_t S, bool XT, bool YT>
__aicore__ inline void CgbtMediumGemmCompute(GM_ADDR workspace, const CgbtGemmTilingData& t)
{
    if constexpr (S >= 144U)
        CgbtGemmBodyT<CgbtMidMM<XT, YT, S>, XT, YT, 176U, true>(workspace, t);
    else
        CgbtGemmBodyT<CgbtMediumMM<XT, YT, S>, XT, YT, 128U, S == 128U>(workspace, t);
}

template <uint32_t S>
__aicore__ inline void CgbtMediumGemmBody(GM_ADDR workspace, const CgbtGemmTilingData& t)
{
    const uint32_t mode = t.transMode & (CGBT_TRANS_X | CGBT_TRANS_Y);
    if (mode == 3U)
        CgbtMediumGemmCompute<S, true, true>(workspace, t);
    else if (mode == 1U)
        CgbtMediumGemmCompute<S, true, false>(workspace, t);
    else if (mode == 2U)
        CgbtMediumGemmCompute<S, false, true>(workspace, t);
    else
        CgbtMediumGemmCompute<S, false, false>(workspace, t);
}

// Keep runtime tile dispatch out of each Matmul calculation's frame.
template <uint32_t S>
__aicore__ __attribute__((noinline)) void CgbtMediumGemmIsolated(GM_ADDR workspace, const CgbtGemmTilingData& t)
{
    CgbtMediumGemmBody<S>(workspace, t);
}

// The 160 tile uses its own compiled storage plan: sharing the general entry's
// frame increases fixed overhead for small batches. Both instances use this
// same split/compute/combine pipeline.
template <uint32_t STATIC_TILE>
__global__ __aicore__ __schedmode__(1) void cgemm_batched_fused_kernel(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace, CgbtFusedTilingData tiling, uint32_t tile)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        CgbtSplitFusedBody(
            aarray, barray, workspace + tiling.gemm.offYr * sizeof(float),
            workspace + tiling.gemm.offYi * sizeof(float), workspace + tiling.gemm.offXr * sizeof(float),
            workspace + tiling.gemm.offXi * sizeof(float), tiling.a, tiling.b, true,
            workspace + tiling.offRangeFlags * sizeof(float), tiling.combine.inputRiskScale);
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        if constexpr (STATIC_TILE != 0U) {
            CgbtMediumGemmBody<STATIC_TILE>(workspace, tiling.gemm);
        } else {
            switch (tile) {
                case 80U:
                    CgbtMediumGemmIsolated<80>(workspace, tiling.gemm);
                    break;
                case 96U:
                    CgbtMediumGemmIsolated<96>(workspace, tiling.gemm);
                    break;
                case 112U:
                    CgbtMediumGemmIsolated<112>(workspace, tiling.gemm);
                    break;
                case 128U:
                    CgbtMediumGemmIsolated<128>(workspace, tiling.gemm);
                    break;
                case 176U:
                    CgbtMediumGemmIsolated<176>(workspace, tiling.gemm);
                    break;
                default:
                    CgbtGemmBody(workspace, tiling.gemm);
                    break;
            }
        }
        SyncAll<false>();
    }
    if ASCEND_IS_AIV {
        CgbtCombineBody<true>(workspace, aarray, barray, carray, tiling.combine, CgbtReferenceTiling(tiling), true);
    }
}

// Preserve the existing tile choices independently of kernel entry selection.
static uint32_t CgbtFusedTile(const CgbtFusedTilingData& t)
{
    const uint32_t extent = t.gemm.m > t.gemm.n ? t.gemm.m : t.gemm.n;
    if (t.gemm.k <= 128U) {
        if (extent <= 80U)
            return 80U;
        if (extent <= 96U)
            return 96U;
        if (extent <= 128U &&
            (!(t.gemm.transMode & CGBT_TRANS_Y) || (t.a.rows > 64U && t.a.rows != t.a.packedLd && t.a.cols > 64U)))
            return 128U;
        if (extent <= 112U)
            return 112U;
    }
    if (t.gemm.tileSize == 176U)
        return extent <= 144U ? 160U : 176U;
    return 0U;
}

void cgemm_batched_fused_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, GM_ADDR workspace,
    CgbtFusedTilingData tiling)
{
    // Unit-alpha/zero-beta small inputs are dispatched by the host before
    // general workspace planning. This panel path handles arbitrary scalars.
    if (tiling.gemm.m <= 64U && tiling.gemm.n <= 64U && tiling.gemm.k <= 64U) {
        cgemm_batched_panel_kernel<<<blockDim, nullptr, stream>>>(aarray, barray, carray, workspace, tiling);
        return;
    }
    const uint32_t tile = CgbtFusedTile(tiling);
    if (tile == 160U)
        cgemm_batched_fused_kernel<160><<<blockDim, nullptr, stream>>>(aarray, barray, carray, workspace, tiling, tile);
    else
        cgemm_batched_fused_kernel<0><<<blockDim, nullptr, stream>>>(aarray, barray, carray, workspace, tiling, tile);
}

extern "C" __global__ __aicore__ void cgemm_batched_split_kernel(
    GM_ADDR ptrA, GM_ADDR ptrB, GM_ADDR dstAr, GM_ADDR dstAi, GM_ADDR dstBr, GM_ADDR dstBi, CgbtSplitTilingData ta,
    CgbtSplitTilingData tb, GM_ADDR inputFlags, float riskScale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgbtSplitFusedBody(ptrA, ptrB, dstAr, dstAi, dstBr, dstBi, ta, tb, false, inputFlags, riskScale);
}

extern "C" __global__ __aicore__ void cgemm_batched_gemm_kernel(GM_ADDR workspace, CgbtGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CgbtGemmBody(workspace, tiling);
}

extern "C" __global__ __aicore__ void cgemm_batched_combine_kernel(
    GM_ADDR workspace, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, CgbtCombineTilingData tiling,
    CgbtDirectTilingData direct)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgbtCombineBody(workspace, aarray, barray, carray, tiling, direct);
}

extern "C" __global__ __aicore__ void cgemm_batched_beta_scale_kernel(GM_ADDR carray, CgbtBetaScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgbtBetaScaleBody(carray, tiling);
    // Restore the public vector register state before returning to runtime.
    ResetMask();
}

// ===========================================================================
//  Launchers
// ===========================================================================
void cgemm_batched_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR ptrA, GM_ADDR ptrB, GM_ADDR dstAr, GM_ADDR dstAi, GM_ADDR dstBr,
    GM_ADDR dstBi, CgbtSplitTilingData ta, CgbtSplitTilingData tb, GM_ADDR inputFlags, float riskScale)
{
    cgemm_batched_split_kernel<<<blockDim, nullptr, stream>>>(
        ptrA, ptrB, dstAr, dstAi, dstBr, dstBi, ta, tb, inputFlags, riskScale);
}

void cgemm_batched_gemm_kernel_do(uint32_t blockDim, void* stream, GM_ADDR workspace, CgbtGemmTilingData tiling)
{
    cgemm_batched_gemm_kernel<<<blockDim, nullptr, stream>>>(workspace, tiling);
}

void cgemm_batched_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR workspace, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray,
    CgbtCombineTilingData tiling, CgbtDirectTilingData direct)
{
    cgemm_batched_combine_kernel<<<blockDim, nullptr, stream>>>(workspace, aarray, barray, carray, tiling, direct);
}

void cgemm_batched_beta_scale_kernel_do(uint32_t blockDim, void* stream, GM_ADDR carray, CgbtBetaScaleTilingData tiling)
{
    cgemm_batched_beta_scale_kernel<<<blockDim, nullptr, stream>>>(carray, tiling);
}

uint32_t cgemm_batched_gemm_single_k() { return cblas3::SINGLE_K; }

extern "C" __global__ __aicore__ void cgemm_batched_direct_kernel(
    GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, CgbtDirectTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // DMA stride fields are uint32 byte counts. Larger legal BLAS strides
    // use the scalar path, whose matrix addresses are computed in uint64.
    constexpr uint32_t MAX_DMA_COMPLEX_STRIDE = 0xffffffffU / (CGBT_COMPLEX_COMPONENTS * sizeof(float));
    if (tiling.lda > MAX_DMA_COMPLEX_STRIDE || tiling.ldb > MAX_DMA_COMPLEX_STRIDE ||
        tiling.ldc > MAX_DMA_COMPLEX_STRIDE) {
        CgbtDirectBody(aarray, barray, carray, tiling);
        return;
    }
    const bool tiny = CgbtUseTinyVector(tiling);
    if (tiny) {
        if (tiling.batchCount <= 64U)
            CgbtTinyBody<8, 2>(aarray, barray, carray, tiling);
        else if (tiling.batchCount <= 256U)
            CgbtTinyBody<8, 8>(aarray, barray, carray, tiling);
        else
            CgbtTinyBody<8, 32>(aarray, barray, carray, tiling);
    } else if (tiling.m <= 16U && tiling.n <= 16U && tiling.k <= 16U) {
        CgbtVectorBody<16>(aarray, barray, carray, tiling);
    } else if (tiling.m <= 32U && tiling.n <= 32U && tiling.k <= 32U) {
        CgbtVectorBody<32>(aarray, barray, carray, tiling);
    } else {
        CgbtDirectBody(aarray, barray, carray, tiling);
    }
    ResetMask();
}

void cgemm_batched_direct_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, CgbtDirectTilingData tiling)
{
    cgemm_batched_direct_kernel<<<blockDim, nullptr, stream>>>(aarray, barray, carray, tiling);
}
