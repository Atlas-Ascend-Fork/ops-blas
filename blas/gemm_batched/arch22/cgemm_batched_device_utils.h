/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "cgemm_batched_tiling_data.h"
#include "cgemm_batched_float.h"

namespace cgbt {

__aicore__ inline uint64_t CgbtBatchPtr(GM_ADDR ptrArray, uint64_t batch, uint32_t batchCount)
{
    AscendC::GlobalTensor<uint64_t> ptrs;
    ptrs.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(ptrArray), batchCount);
    return ptrs.GetValue(batch);
}

// Preserve the reference's sequential complex reduction for small outputs.
// The four real reductions otherwise amplify cancellation when very few output
// elements are available for the required 99% match ratio.
__aicore__ inline CgbtComplexValue CgbtReadScaledC(
    const CgbtDirectTilingData& t, AscendC::GlobalTensor<float>& c, uint32_t row, uint32_t col)
{
    const uint64_t ci = 2U * (static_cast<uint64_t>(col) * t.ldc + row);
    float oldRe = 0.0F;
    float oldIm = 0.0F;
    if (t.betaRe != 0.0F || t.betaIm != 0.0F) {
        const float cr = c.GetValue(ci);
        const float cm = c.GetValue(ci + 1U);
        // Netlib's N path skips scaling for beta=1. Its T/C dot-product
        // paths still evaluate beta*C with complex infinity recovery.
        if (t.transA == 0U && t.betaRe == 1.0F && t.betaIm == 0.0F) {
            oldRe = cr;
            oldIm = cm;
        } else {
            const auto scaled = CgbtComplexMultiply({t.betaRe, t.betaIm}, {cr, cm});
            oldRe = scaled.real;
            oldIm = scaled.imag;
        }
    }
    return {oldRe, oldIm};
}

__aicore__ inline CgbtComplexValue CgbtSafeProduct(float ar, float am, float br, float bm)
{
    float pr = ar * br - am * bm;
    float pi = ar * bm + am * br;
    if (ar > CGBT_SAFE_COMPONENT_MAGNITUDE || ar < -CGBT_SAFE_COMPONENT_MAGNITUDE ||
        am > CGBT_SAFE_COMPONENT_MAGNITUDE || am < -CGBT_SAFE_COMPONENT_MAGNITUDE ||
        br > CGBT_SAFE_COMPONENT_MAGNITUDE || br < -CGBT_SAFE_COMPONENT_MAGNITUDE ||
        bm > CGBT_SAFE_COMPONENT_MAGNITUDE || bm < -CGBT_SAFE_COMPONENT_MAGNITUDE) {
        pr = CgbtFma(ar, br, -(am * bm));
        pi = CgbtFma(ar, bm, am * br);
    }
    if (pr != pr && pi != pi)
        return CgbtComplexMultiply({ar, am}, {br, bm});
    return {pr, pi};
}

__aicore__ inline CgbtComplexValue CgbtDirectValue(
    const CgbtDirectTilingData& t, AscendC::GlobalTensor<float>& a, AscendC::GlobalTensor<float>& b,
    AscendC::GlobalTensor<float>& c, uint32_t row, uint32_t col)
{
    const auto old = CgbtReadScaledC(t, c, row, col);
    const float oldRe = old.real;
    const float oldIm = old.imag;
    float sr = t.transA == 0U ? oldRe : 0.0F;
    float si = t.transA == 0U ? oldIm : 0.0F;
    for (uint32_t l = 0; l < t.k; ++l) {
        const uint64_t ai =
            2U * (t.transA == 0U ? static_cast<uint64_t>(l) * t.lda + row : static_cast<uint64_t>(row) * t.lda + l);
        const uint64_t bi =
            2U * (t.transB == 0U ? static_cast<uint64_t>(col) * t.ldb + l : static_cast<uint64_t>(l) * t.ldb + col);
        const float ar = a.GetValue(ai);
        const float am = t.transA == 2U ? -a.GetValue(ai + 1U) : a.GetValue(ai + 1U);
        float br = b.GetValue(bi);
        float bm = t.transB == 2U ? -b.GetValue(bi + 1U) : b.GetValue(bi + 1U);
        if (t.transA == 0U) {
            const auto scaled = CgbtComplexMultiply({t.alphaRe, t.alphaIm}, {br, bm});
            br = scaled.real;
            bm = scaled.imag;
        }
        const auto product = CgbtSafeProduct(ar, am, br, bm);
        sr += product.real;
        si += product.imag;
    }
    if (t.transA != 0U) {
        const auto scaled = CgbtComplexMultiply({t.alphaRe, t.alphaIm}, {sr, si});
        sr = scaled.real + oldRe;
        si = scaled.imag + oldIm;
    }
    return {sr, si};
}

__aicore__ inline AscendC::LoadData3DParamsV2<float> CgbtLocalLoad(uint32_t rows, uint32_t cols, bool transpose)
{
    AscendC::LoadData3DParamsV2<float> p{};
    p.l1H = 1U;
    p.l1W = rows;
    p.channelSize = cols;
    p.kExtension = cols;
    p.mExtension = rows;
    p.strideW = p.strideH = p.filterW = p.filterH = 1U;
    p.dilationFilterW = p.dilationFilterH = 1U;
    p.enTranspose = transpose;
    return p;
}

__aicore__ inline void CgbtCreateInterleaveIndex(AscendC::LocalTensor<int32_t> dst, uint32_t pairs, uint32_t halfStride)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < 8U; ++i)
        dst.SetValue(i, static_cast<int32_t>((i / 2U + (i % 2U) * halfStride) * 4U));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (uint32_t filled = 8U; filled < pairs * 2U; filled *= 2U) {
        const uint32_t size = pairs * 2U - filled < filled ? pairs * 2U - filled : filled;
        AscendC::Adds(dst[filled], dst, static_cast<int32_t>(filled * 2U), size);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// Construct byte offsets directly, instead of creating linear element
// indices and then permuting them with several full-panel vector passes.
__aicore__ inline void CgbtCreatePairSwapIndex(AscendC::LocalTensor<int32_t> dst, uint32_t count)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < 8U; ++i)
        dst.SetValue(i, static_cast<int32_t>((i ^ 1U) * 4U));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (uint32_t filled = 8U; filled < count; filled *= 2U) {
        const uint32_t size = count - filled < filled ? count - filled : filled;
        AscendC::Adds(dst[filled], dst, static_cast<int32_t>(filled * 4U), size);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <uint32_t P, uint32_t PITCH = P>
__aicore__ inline void CgbtCreateTransposeIndex(AscendC::LocalTensor<int32_t> dst, uint32_t matrices)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < 8U; ++i)
        dst.SetValue(i, static_cast<int32_t>(((i / 2U) * 2U * PITCH + i % 2U) * 4U));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    // Fill one output row, then duplicate whole rows with their source
    // column offset. Both lengths are multiples of a 32-byte UB block.
    for (uint32_t filled = 8U; filled < 2U * P; filled *= 2U) {
        const uint32_t size = 2U * P - filled < filled ? 2U * P - filled : filled;
        AscendC::Adds(dst[filled], dst, static_cast<int32_t>(filled * PITCH * 4U), size);
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t rows = 1U; rows < P; rows *= 2U) {
        const uint32_t size = P - rows < rows ? P - rows : rows;
        AscendC::Adds(dst[rows * 2U * P], dst, static_cast<int32_t>(rows * 8U), size * 2U * P);
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t q = 1U; q < matrices; ++q)
        AscendC::Adds(dst[q * 2U * P * P], dst, static_cast<int32_t>(q * 2U * P * PITCH * 4U), 2U * P * P);
    AscendC::PipeBarrier<PIPE_V>();
}

// Seed one block, then double the populated prefix. The generic index API
// extends by one vector repeat at a time on A2, issuing many scalar commands.
__aicore__ inline void CgbtCreateIndex(AscendC::LocalTensor<int32_t> dst, uint32_t count)
{
    // Release previous vector writes before seeding with scalar stores.
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint32_t seed = count < 8U ? count : 8U;
    for (uint32_t i = 0; i < seed; ++i)
        dst.SetValue(i, static_cast<int32_t>(i));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (uint32_t filled = 8U; filled < count; filled *= 2U) {
        const uint32_t size = count - filled < filled ? count - filled : filled;
        AscendC::Adds(dst[filled], dst, static_cast<int32_t>(filled), size);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// ============================================================================
// Batch vector accumulation shared by the two kernel entry files
// ============================================================================
template <uint32_t S>
__aicore__ inline void CgbtTinyLoad(
    AscendC::GlobalTensor<uint64_t>& ptrs, bool contiguous, AscendC::GlobalTensor<float>& gm,
    AscendC::LocalTensor<float>& re, AscendC::LocalTensor<float>& im, AscendC::LocalTensor<float>& stage,
    uint32_t matrices)
{
    if (contiguous) {
        cblas3::LoadChunkDeinterleaved(gm, 0U, 0U, matrices * S * S, stage, re, im);
    } else {
        for (uint32_t q = 0; q < matrices; ++q) {
            AscendC::GlobalTensor<float> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ptrs.GetValue(q)));
            cblas3::LoadChunkDeinterleaved(source, 0U, 0U, S * S, stage, re[q * S * S], im[q * S * S]);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Exceptional IEEE values use the sequential BLAS evaluation. Keep this
// scalar implementation outside the ordinary vector kernel's register scope.
template <uint32_t S>
__aicore__ __attribute__((noinline)) void CgbtTinyCorrect(
    const CgbtDirectTilingData& t, AscendC::GlobalTensor<uint64_t> pa, AscendC::GlobalTensor<uint64_t> pb,
    AscendC::GlobalTensor<uint64_t> pc, AscendC::LocalTensor<float> sr, AscendC::LocalTensor<float> si,
    uint32_t matrices)
{
    AscendC::GlobalTensor<float> ag, bg, cg;
    for (uint32_t q = 0; q < matrices; ++q) {
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pa.GetValue(q)));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pb.GetValue(q)));
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pc.GetValue(q)));
        for (uint32_t col = 0; col < S; ++col) {
            for (uint32_t row = 0; row < S; ++row) {
                const auto result = CgbtDirectValue(t, ag, bg, cg, row, col);
                sr.SetValue(q * S * S + col * S + row, result.real);
                si.SetValue(q * S * S + col * S + row, result.imag);
            }
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
}

struct CgbtTinyBuffers {
    AscendC::LocalTensor<float> ar, ai, br, bi, sr, si, x, y, z, w, stage, ar2, ai2, nai2;
    AscendC::LocalTensor<int32_t> ib;
    AscendC::LocalTensor<uint32_t> ic;
};

template <uint32_t S, uint32_t GROUP>
__aicore__ inline float CgbtTinyRange(CgbtTinyBuffers& buffers, uint32_t count)
{
    AscendC::Abs(buffers.x, buffers.ar, count);
    AscendC::Abs(buffers.y, buffers.ai, count);
    AscendC::Abs(buffers.z, buffers.br, count);
    AscendC::Abs(buffers.w, buffers.bi, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(buffers.x, buffers.x, buffers.y, count);
    AscendC::Max(buffers.z, buffers.z, buffers.w, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(buffers.x, buffers.x, buffers.z, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceMax(
        buffers.y, buffers.x, uint64_t(64), count / 64U, 1U, 1U, 8U, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceMax(
        buffers.sr, buffers.y, uint64_t(count / 64U), 1U, 1U, 1U, 8U, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    AscendC::PipeBarrier<PIPE_ALL>();
    return buffers.sr.GetValue(0);
}

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void CgbtTinyRead(
    CgbtTinyBuffers& buffers, AscendC::GlobalTensor<uint64_t> pa, AscendC::GlobalTensor<uint64_t> pb, bool contiguousA,
    bool contiguousB, AscendC::GlobalTensor<float> ag, AscendC::GlobalTensor<float> bg, uint32_t matrices,
    uint32_t count)
{
    if (S == 16U && contiguousA && contiguousB) {
        const AscendC::DataCopyExtParams input{1U, count * 8U, 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> padding{false, 0U, 0U, 0.0F};
        AscendC::DataCopyPad(buffers.stage, ag, input, padding);
        AscendC::DataCopyPad(buffers.x, bg, input, padding);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        uint64_t selected = 0U;
        const uint16_t repeats = static_cast<uint16_t>(count / 32U);
        AscendC::GatherMask(buffers.ar, buffers.stage, 1, false, 0, {1U, repeats, 8U, 8U}, selected);
        AscendC::GatherMask(buffers.ai, buffers.stage, 2, false, 0, {1U, repeats, 8U, 8U}, selected);
        AscendC::GatherMask(buffers.br, buffers.x, 1, false, 0, {1U, repeats, 8U, 8U}, selected);
        AscendC::GatherMask(buffers.bi, buffers.x, 2, false, 0, {1U, repeats, 8U, 8U}, selected);
        AscendC::PipeBarrier<PIPE_ALL>();
    } else {
        CgbtTinyLoad<S>(pa, contiguousA, ag, buffers.ar, buffers.ai, buffers.stage, matrices);
        CgbtTinyLoad<S>(pb, contiguousB, bg, buffers.br, buffers.bi, buffers.stage, matrices);
    }
}

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void CgbtTinyPrepare(CgbtTinyBuffers& buffers, uint32_t matrices, uint32_t count)
{
    // Exceptional inputs above retain Netlib's zero-times-infinity evaluation.
    if constexpr (S == 8U) {
        AscendC::Muls(buffers.x, buffers.bi, 0.0F, count);
        AscendC::Muls(buffers.y, buffers.br, 0.0F, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(buffers.br, buffers.br, buffers.x, count);
        AscendC::Add(buffers.bi, buffers.bi, buffers.y, count);
    }
    AscendC::Duplicate(buffers.sr, 0.0F, count);
    AscendC::Duplicate(buffers.si, 0.0F, count);
    if constexpr (S == 16U) {
        // Each source matrix is copied to both output column panels.
        // One strided UB burst covers every live batch member.
        const AscendC::DataCopyParams duplicatePanel{
            uint16_t(matrices), uint16_t(S * S / 8U), 0U, uint16_t(S * S / 8U)};
        AscendC::DataCopy(buffers.ar2, buffers.ar, duplicatePanel);
        AscendC::DataCopy(buffers.ar2[S * S], buffers.ar, duplicatePanel);
        AscendC::DataCopy(buffers.ai2, buffers.ai, duplicatePanel);
        AscendC::DataCopy(buffers.ai2[S * S], buffers.ai, duplicatePanel);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(
            buffers.nai2, buffers.ai2, -1.0F, uint64_t(64), static_cast<uint8_t>(2U * count / 64U), {1, 1, 8, 8});
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void CgbtTinyMultiply8(CgbtTinyBuffers& buffers, uint32_t p, uint32_t matrices)
{
    constexpr uint32_t COUNT = S * S * GROUP;
    AscendC::Gather(buffers.x, buffers.br, buffers.ib.ReinterpretCast<uint32_t>(), p * 4U, matrices * S);
    AscendC::Gather(buffers.y, buffers.bi, buffers.ib.ReinterpretCast<uint32_t>(), p * 4U, matrices * S);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(buffers.z, buffers.x, static_cast<uint8_t>(matrices), {1U, 8U});
    AscendC::Brcb(buffers.w, buffers.y, static_cast<uint8_t>(matrices), {1U, 8U});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(0ULL, ~0ULL);
    const AscendC::BinaryRepeatParams broadcastA{1U, 0U, 1U, 8U, 8U, 8U};
    AscendC::Mul<float, false>(
        buffers.stage, buffers.ar[p * S], buffers.w, uint64_t(64), static_cast<uint8_t>(matrices), broadcastA);
    AscendC::Mul<float, false>(
        buffers.stage[COUNT], buffers.ai[p * S], buffers.z, uint64_t(64), static_cast<uint8_t>(matrices), broadcastA);
    AscendC::Mul<float, false>(
        buffers.x, buffers.ar[p * S], buffers.z, uint64_t(64), static_cast<uint8_t>(matrices), broadcastA);
    AscendC::Mul<float, false>(
        buffers.y, buffers.ai[p * S], buffers.w, uint64_t(64), static_cast<uint8_t>(matrices), broadcastA);
}

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void CgbtTinyAddProducts(CgbtTinyBuffers& buffers, uint32_t count)
{
    constexpr uint32_t COUNT = S * S * GROUP;
    if constexpr (S == 8U) {
        AscendC::Sub<float, false>(
            buffers.x, buffers.x, buffers.y, uint64_t(64), static_cast<uint8_t>(count / 64U),
            AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        AscendC::Add<float, false>(
            buffers.stage, buffers.stage, buffers.stage[COUNT], uint64_t(64), static_cast<uint8_t>(count / 64U),
            AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(
            buffers.sr, buffers.sr, buffers.x, uint64_t(64), static_cast<uint8_t>(count / 64U),
            AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        AscendC::Add<float, false>(
            buffers.si, buffers.si, buffers.stage, uint64_t(64), static_cast<uint8_t>(count / 64U),
            AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void CgbtTinyAccumulate(CgbtTinyBuffers& buffers, uint32_t p, uint32_t matrices, uint32_t count)
{
    if constexpr (S == 8U) {
        CgbtTinyMultiply8<S, GROUP>(buffers, p, matrices);
    } else {
        AscendC::Gather(buffers.x, buffers.br, buffers.ib.ReinterpretCast<uint32_t>(), p * 4U, matrices * S);
        AscendC::Gather(buffers.y, buffers.bi, buffers.ib.ReinterpretCast<uint32_t>(), p * 4U, matrices * S);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(buffers.z, buffers.x, static_cast<uint8_t>(matrices * 2U), {1U, 8U});
        AscendC::Brcb(buffers.w, buffers.y, static_cast<uint8_t>(matrices * 2U), {1U, 8U});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(0ULL, ~0ULL);
        const AscendC::BinaryRepeatParams broadcastA{2U, 0U, 1U, 16U, 32U, 8U};
        for (uint32_t rowHalf = 0; rowHalf < 2U; ++rowHalf) {
            const uint32_t dst = rowHalf * 8U, aoff = p * S + rowHalf * 8U;
            AscendC::MulAddDst<float, float, false>(
                buffers.sr[dst], buffers.ar2[aoff], buffers.z, uint64_t(64), static_cast<uint8_t>(matrices * 2U),
                broadcastA);
            AscendC::MulAddDst<float, float, false>(
                buffers.si[dst], buffers.ar2[aoff], buffers.w, uint64_t(64), static_cast<uint8_t>(matrices * 2U),
                broadcastA);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t rowHalf = 0; rowHalf < 2U; ++rowHalf) {
            const uint32_t dst = rowHalf * 8U, aoff = p * S + rowHalf * 8U;
            AscendC::MulAddDst<float, float, false>(
                buffers.sr[dst], buffers.nai2[aoff], buffers.w, uint64_t(64), static_cast<uint8_t>(matrices * 2U),
                broadcastA);
            AscendC::MulAddDst<float, float, false>(
                buffers.si[dst], buffers.ai2[aoff], buffers.z, uint64_t(64), static_cast<uint8_t>(matrices * 2U),
                broadcastA);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
    CgbtTinyAddProducts<S, GROUP>(buffers, count);
}

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void CgbtTinyStore(
    CgbtTinyBuffers& buffers, AscendC::GlobalTensor<uint64_t> pc, AscendC::GlobalTensor<float> cg, uint64_t cbase,
    bool contiguousC, uint32_t matrices, uint32_t count)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Gather(buffers.stage, buffers.sr, buffers.ic, 0U, count * 2U);
    AscendC::PipeBarrier<PIPE_ALL>();
    if (contiguousC) {
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cbase));
        const AscendC::DataCopyExtParams copy{1U, count * 8U, 0U, 0U, 0U};
        AscendC::DataCopyPad(cg, buffers.stage, copy);
    } else {
        for (uint32_t q = 0; q < matrices; ++q) {
            cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pc.GetValue(q)));
            const AscendC::DataCopyExtParams copy{1U, S * S * 8U, 0U, 0U, 0U};
            AscendC::DataCopyPad(cg, buffers.stage[q * S * S * 2U], copy);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void CgbtTinyGroup(
    CgbtTinyBuffers& buffers, const CgbtDirectTilingData& t, AscendC::GlobalTensor<uint64_t> pag,
    AscendC::GlobalTensor<uint64_t> pbg, AscendC::GlobalTensor<uint64_t> pcg, uint32_t batch, uint32_t end)
{
    const uint32_t matrices = end - batch < GROUP ? end - batch : GROUP;
    const uint32_t count = matrices * S * S;
    auto pa = pag[batch], pb = pbg[batch], pc = pcg[batch];
    AscendC::DataCachePreload(pa[0], int16_t(0));
    AscendC::DataCachePreload(pb[0], int16_t(0));
    AscendC::DataCachePreload(pc[0], int16_t(0));
    const uint64_t abase = pa.GetValue(0), bbase = pb.GetValue(0), cbase = pc.GetValue(0);
    bool contiguousA = true, contiguousB = true, contiguousC = true;
    for (uint32_t q = 1; q < matrices; ++q) {
        contiguousA &= pa.GetValue(q) == abase + q * S * S * 8U;
        contiguousB &= pb.GetValue(q) == bbase + q * S * S * 8U;
        contiguousC &= pc.GetValue(q) == cbase + q * S * S * 8U;
    }
    AscendC::GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(abase));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bbase));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cbase));
    CgbtTinyRead<S, GROUP>(buffers, pa, pb, contiguousA, contiguousB, ag, bg, matrices, count);
    const float largest = CgbtTinyRange<S, GROUP>(buffers, count);
    if (!(largest <= CGBT_SAFE_COMPONENT_MAGNITUDE)) {
        CgbtTinyCorrect<S>(t, pa, pb, pc, buffers.sr, buffers.si, matrices);
    } else {
        CgbtTinyPrepare<S, GROUP>(buffers, matrices, count);
        for (uint32_t p = 0; p < S; ++p) {
            CgbtTinyAccumulate<S, GROUP>(buffers, p, matrices, count);
        }
    }
    CgbtTinyStore<S, GROUP>(buffers, pc, cg, cbase, contiguousC, matrices, count);
}

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void CgbtTinyBody(GM_ADDR aarray, GM_ADDR barray, GM_ADDR carray, const CgbtDirectTilingData& t)
{
    constexpr uint32_t COUNT = S * S * GROUP;
    AscendC::InitSocState();
    AscendC::LocalTensor<float> all(AscendC::TPosition::VECCALC, 0U, 12U * COUNT);
    AscendC::LocalTensor<int32_t> indices(AscendC::TPosition::VECCALC, 12U * COUNT * 4U, 4U * COUNT);
    auto ar = all, ai = all[COUNT], br = all[2U * COUNT], bi = all[3U * COUNT];
    auto sr = all[4U * COUNT], si = all[5U * COUNT];
    auto x = all[6U * COUNT], y = all[7U * COUNT], z = all[8U * COUNT], w = all[9U * COUNT];
    auto stage = all[10U * COUNT];
    auto negativeAi = indices.ReinterpretCast<float>();
    // Duplicate A along the column-panel axis so one hardware repeat stream
    // covers both column halves and all batch members.
    static_assert((S == 16U ? 22U : 16U) * COUNT * 4U <= 192U * 1024U, "UB capacity");
    AscendC::LocalTensor<float> broadcast(AscendC::TPosition::VECCALC, 16U * COUNT * 4U, S == 16U ? 6U * COUNT : 0U);
    auto ar2 = broadcast, ai2 = broadcast[2U * COUNT], nai2 = broadcast[4U * COUNT];
    auto ib = indices[COUNT];
    auto ic = indices[2U * COUNT].ReinterpretCast<uint32_t>();
    CgbtTinyBuffers buffers{ar, ai, br, bi, sr, si, x, y, z, w, stage, ar2, ai2, nai2, ib, ic};
    // One coefficient per B column, expanded to 8 lanes by Brcb.
    // Instruction strides broadcast A across each 8-column output panel.
    CgbtCreateIndex(ib, GROUP * S);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftLeft(ib, ib, int32_t(S == 8U ? 5 : 6), GROUP * S);
    CgbtCreateInterleaveIndex(ic.ReinterpretCast<int32_t>(), COUNT, COUNT);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint64_t> pag, pbg, pcg;
    pag.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(aarray));
    pbg.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(barray));
    pcg.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(carray));
    const uint32_t workers = AscendC::GetBlockNum(), owner = AscendC::GetBlockIdx();
    const uint32_t perCore = t.batchCount / workers, extra = t.batchCount % workers;
    const uint32_t begin = owner * perCore + (owner < extra ? owner : extra),
                   end = begin + perCore + (owner < extra ? 1U : 0U);
    for (uint32_t batch = begin; batch < end; batch += GROUP) {
        CgbtTinyGroup<S, GROUP>(buffers, t, pag, pbg, pcg, batch, end);
    }
}

} // namespace cgbt
