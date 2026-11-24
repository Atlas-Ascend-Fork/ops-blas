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
 * \file cgemm_kernel.cpp
 * \brief CGEMM kernels for Atlas A2 training series (arch22).
 *
 *        C = alpha * op(A) * op(B) + beta * C   (column-major complex64)
 *
 *        Three-stage pipeline:
 *          Phase 0 (AIV) split   : A -> (Ar, Ai), B -> (Br, Bi), each a packed
 *                                  real column-major plane
 *          Phase 1 (AIC) gemm    : four real GEMMs through the arch22 real
 *                                  aclblasSgemm entry, driven from the host
 *          Phase 2 (AIV) combine : C = alpha * ((re) + i*(im)) + beta * C_old
 *
 *        Only Phase 0 and Phase 2 live here. The real GEMMs deliberately reuse
 *        the tuned aclblasSgemm implementation instead of a second Cube kernel:
 *        the complex product expands to four real products whose signs are known
 *        on the host, so the only device-side work left is the de-interleave and
 *        the complex epilogue.
 *
 *        Phase 1 produces
 *          re = Ar*Br - Ai*Bi        im = Ar*Bi + Ai*Br
 *        which is exactly Re/Im of op(A)*op(B) once the conjugation of trans='C'
 *        has been folded into the imaginary plane by Phase 0.
 */

#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "cgemm_kernel.h"
#include "cgemm_tiling_data.h"

using namespace AscendC;

namespace {
// Complex elements handled by one Phase 2 inner iteration. A whole multiple of a
// 256B fp32 repeat so the GatherMask repeat count is exact. The buffer set below
// totals roughly 60KB, well inside the 192KB AIV UB.
constexpr uint32_t COMBINE_CHUNK = 1024;
static_assert(COMBINE_CHUNK % cblas3::ELENUM_REPEAT_FP32 == 0, "COMBINE_CHUNK must be a whole number of fp32 repeats");

// Everything the combine chunk loop needs. TPipe is deliberately NOT a member:
// rule R3 requires it to stay a local of the kernel entry.
struct CgemmCombineCtx {
    GlobalTensor<float> reGm;
    GlobalTensor<float> imGm;
    GlobalTensor<float> cGm;
    LocalTensor<float> crci; // [re | im] halves, interleaved by Gather
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> re;
    LocalTensor<float> im;
    LocalTensor<float> tmpA;
    LocalTensor<float> tmpB;
    LocalTensor<float> stage; // interleaved C_old in / C_new out
    LocalTensor<uint32_t> idxU;
    bool applyAlpha;
    bool applyBeta;
};

// The UB scratch buffers backing a CgemmCombineCtx. Declared in the caller's
// frame so their lifetime spans every CombineOneChunk call: the LocalTensors
// handed to ctx below stay in use for the whole column loop, so the TBufs they
// come from must outlive this function.
struct CgemmCombineBuffers {
    TBuf<TPosition::VECCALC> crci;
    TBuf<TPosition::VECCALC> re;
    TBuf<TPosition::VECCALC> im;
    TBuf<TPosition::VECCALC> tmpA;
    TBuf<TPosition::VECCALC> tmpB;
    TBuf<TPosition::VECCALC> stage;
    TBuf<TPosition::VECCALC> idx;
    TBuf<TPosition::VECCALC> aux0;
    TBuf<TPosition::VECCALC> aux1;
};

__aicore__ inline void InitCombineCtx(
    TPipe& pipe, CgemmCombineBuffers& bufs, CgemmCombineCtx& ctx, GM_ADDR re, GM_ADDR im, GM_ADDR c,
    const CgemmCombineTilingData& tiling)
{
    // alpha = (0,0) or k = 0 makes the host skip the whole product allocation, so
    // the two product planes come in null. The alpha term is not evaluated in that
    // case either (applyAlpha gates it below), so leave those tensors unbound
    // instead of handing a null address to SetGlobalBuffer. beta may still be
    // anything, and C is always bound.
    if (tiling.hasTemp != 0U) {
        ctx.reGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(re));
        ctx.imGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(im));
    }
    ctx.cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));

    constexpr uint32_t chunk = COMBINE_CHUNK;
    constexpr uint32_t pairElems = chunk * cblas3::COMPLEX_ELENUM;

    pipe.InitBuffer(bufs.crci, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.re, chunk * sizeof(float));
    pipe.InitBuffer(bufs.im, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpA, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpB, chunk * sizeof(float));
    pipe.InitBuffer(bufs.stage, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.idx, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux0, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux1, pairElems * sizeof(int32_t));

    ctx.crci = bufs.crci.Get<float>();
    ctx.cr = ctx.crci;
    ctx.ci = ctx.crci[chunk];
    ctx.re = bufs.re.Get<float>();
    ctx.im = bufs.im.Get<float>();
    ctx.tmpA = bufs.tmpA.Get<float>();
    ctx.tmpB = bufs.tmpB.Get<float>();
    ctx.stage = bufs.stage.Get<float>();

    LocalTensor<int32_t> idx = bufs.idx.Get<int32_t>();
    cblas3::BuildInterleaveIndex(idx, bufs.aux0.Get<int32_t>(), bufs.aux1.Get<int32_t>(), chunk, chunk);
    ctx.idxU = idx.ReinterpretCast<uint32_t>();

    ctx.applyAlpha = (tiling.hasTemp != 0U);
    ctx.applyBeta = (tiling.betaRe != 0.0F || tiling.betaIm != 0.0F);
}

// accRe = ar*re - ai*im, accIm = ar*im + ai*re.
// A zero component of alpha is skipped rather than multiplied through. Either
// product plane can reach the fp32 range limit on its own, and multiplying such
// an Inf by an alpha component of 0 yields NaN, which would carry into a part
// that is otherwise exact. Skipping matches the BLAS convention that a zero
// scalar leaves its operand unreferenced.
__aicore__ inline void ApplyAlphaScale(
    CgemmCombineCtx& ctx, const CgemmCombineTilingData& tiling, uint32_t col, uint32_t row, uint32_t len)
{
    DataCopyExtParams realParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    const uint64_t planeOffset = static_cast<uint64_t>(col) * tiling.ldRe + row;

    DataCopyPad(ctx.re, ctx.reGm[planeOffset], realParams, padParams);
    DataCopyPad(ctx.im, ctx.imGm[planeOffset], realParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (tiling.alphaRe != 0.0F) {
        Muls(ctx.cr, ctx.re, tiling.alphaRe, len);
        Muls(ctx.ci, ctx.im, tiling.alphaRe, len);
    } else {
        Duplicate(ctx.cr, 0.0F, len);
        Duplicate(ctx.ci, 0.0F, len);
    }
    if (tiling.alphaIm != 0.0F) {
        Muls(ctx.tmpA, ctx.im, tiling.alphaIm, len);
        Muls(ctx.tmpB, ctx.re, tiling.alphaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.cr, ctx.cr, ctx.tmpA, len);
        Add(ctx.ci, ctx.ci, ctx.tmpB, len);
    }
    PipeBarrier<PIPE_V>();
}

// accRe += br*Cr_old - bi*Ci_old, accIm += br*Ci_old + bi*Cr_old. The two product
// planes are free to reuse here, the alpha term having already been folded into
// cr/ci. A zero component of beta is skipped for the same reason as in
// ApplyAlphaScale: the incoming C may itself carry a non-finite value, and
// 0 * Inf is NaN.
__aicore__ inline void ApplyBetaScale(
    CgemmCombineCtx& ctx, const CgemmCombineTilingData& tiling, uint64_t cOffset, uint32_t len)
{
    uint64_t rsvdCnt = 0;
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    DataCopyExtParams cParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.stage, ctx.cGm[cOffset], cParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    const uint16_t repeats = static_cast<uint16_t>(
        cblas3::CeilDivU(len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), cblas3::BYTENUM_REPEAT));
    GatherMask(ctx.tmpA, ctx.stage, 1, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    GatherMask(ctx.tmpB, ctx.stage, 2, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    if (tiling.betaRe != 0.0F) {
        Muls(ctx.re, ctx.tmpA, tiling.betaRe, len);
        Muls(ctx.im, ctx.tmpB, tiling.betaRe, len);
        PipeBarrier<PIPE_V>();
        Add(ctx.cr, ctx.cr, ctx.re, len);
        Add(ctx.ci, ctx.ci, ctx.im, len);
        PipeBarrier<PIPE_V>();
    }
    if (tiling.betaIm != 0.0F) {
        Muls(ctx.re, ctx.tmpB, tiling.betaIm, len);
        Muls(ctx.im, ctx.tmpA, tiling.betaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.cr, ctx.cr, ctx.re, len);
        Add(ctx.ci, ctx.ci, ctx.im, len);
        PipeBarrier<PIPE_V>();
    }
}

// One contiguous run of `len` rows inside column `col`, starting at `row`.
__aicore__ inline void CombineOneChunk(
    CgemmCombineCtx& ctx, const CgemmCombineTilingData& tiling, uint32_t col, uint32_t row, uint32_t len)
{
    if (ctx.applyAlpha) {
        ApplyAlphaScale(ctx, tiling, col, row, len);
    } else {
        Duplicate(ctx.cr, 0.0F, len);
        Duplicate(ctx.ci, 0.0F, len);
        PipeBarrier<PIPE_V>();
    }

    const uint64_t cOffset = (static_cast<uint64_t>(col) * tiling.ldc + row) * cblas3::COMPLEX_ELENUM;
    if (ctx.applyBeta) {
        ApplyBetaScale(ctx, tiling, cOffset, len);
    }

    Gather(ctx.stage, ctx.crci, ctx.idxU, 0U, len * cblas3::COMPLEX_ELENUM);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopyExtParams outParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.cGm[cOffset], ctx.stage, outParams);

    // Both queues must wait on MTE3 before the next iteration reuses `stage`:
    // MTE3_V covers the Gather that writes it, MTE3_MTE2 the DataCopyPad that
    // reads C_old into it. Relying on MTE2_V transitivity would break on the path
    // where neither alpha nor beta is applied, which issues no MTE2 at all.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
} // namespace

// ===========================================================================
//  Kernel entries
// ===========================================================================
extern "C" __global__ __aicore__ void cgemm_split_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::SplitBody(a, ar, ai, tiling);
}

extern "C" __global__ __aicore__ void cgemm_accum_kernel(GM_ADDR dst, GM_ADDR src, CgemmAccumTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U || tiling.totalFloats == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    GlobalTensor<float> dstGm;
    GlobalTensor<float> srcGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst));
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));

    TPipe pipe;
    TBuf<TPosition::VECCALC> srcBuf;
    TBuf<TPosition::VECCALC> dstBuf;
    pipe.InitBuffer(srcBuf, COMBINE_CHUNK * sizeof(float));
    pipe.InitBuffer(dstBuf, COMBINE_CHUNK * sizeof(float));
    LocalTensor<float> srcLocal = srcBuf.Get<float>();
    LocalTensor<float> dstLocal = dstBuf.Get<float>();

    const uint64_t totalChunks = (tiling.totalFloats + COMBINE_CHUNK - 1ULL) / COMBINE_CHUNK;
    for (uint64_t chunk = coreIdx; chunk < totalChunks; chunk += coreNum) {
        const uint64_t off = chunk * COMBINE_CHUNK;
        const uint32_t len = static_cast<uint32_t>(
            (off + COMBINE_CHUNK > tiling.totalFloats) ? (tiling.totalFloats - off) : COMBINE_CHUNK);
        DataCopyExtParams inParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
        DataCopyPad(srcLocal, srcGm[off], inParams, padParams);
        if (tiling.overwrite == 0U) {
            DataCopyPad(dstLocal, dstGm[off], inParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (tiling.overwrite != 0U) {
            Muls(dstLocal, srcLocal, tiling.scale, len);
        } else {
            Axpy(dstLocal, srcLocal, tiling.scale, static_cast<int32_t>(len));
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams outParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(dstGm[off], dstLocal, outParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}

extern "C" __global__ __aicore__ void cgemm_combine_kernel(
    GM_ADDR re, GM_ADDR im, GM_ADDR c, CgemmCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());

    TPipe pipe;
    CgemmCombineBuffers bufs;
    CgemmCombineCtx ctx;
    InitCombineCtx(pipe, bufs, ctx, re, im, c, tiling);

    // Every column carries the same amount of work here (no triangle), so a plain
    // round robin balances the cores exactly.
    for (uint32_t col = coreIdx; col < tiling.n; col += coreNum) {
        for (uint32_t off = 0U; off < tiling.m; off += COMBINE_CHUNK) {
            const uint32_t len = (off + COMBINE_CHUNK > tiling.m) ? (tiling.m - off) : COMBINE_CHUNK;
            CombineOneChunk(ctx, tiling, col, off, len);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

// ===========================================================================
//  Launchers
// ===========================================================================
void cgemm_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling)
{
    cgemm_split_kernel<<<blockDim, nullptr, stream>>>(a, ar, ai, tiling);
}

void cgemm_accum_kernel_do(uint32_t blockDim, void* stream, GM_ADDR dst, GM_ADDR src, CgemmAccumTilingData tiling)
{
    cgemm_accum_kernel<<<blockDim, nullptr, stream>>>(dst, src, tiling);
}

void cgemm_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR re, GM_ADDR im, GM_ADDR c, CgemmCombineTilingData tiling)
{
    cgemm_combine_kernel<<<blockDim, nullptr, stream>>>(re, im, c, tiling);
}
