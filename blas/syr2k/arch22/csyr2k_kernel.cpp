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
 * \file csyr2k_kernel.cpp
 * \brief CSYR2K kernels for Atlas A2 training series (arch22).
 *
 *        C = alpha*A*B^T + alpha*B*A^T + beta*C   (trans = 'N', A and B are n x k)
 *        C = alpha*A^T*B + alpha*B^T*A + beta*C   (trans = 'T', A and B are k x n)
 *        alpha and beta are complex; C is symmetric and only the uplo triangle is
 *        referenced and updated. All matrices are column-major complex64.
 *
 *        Phases 0 and 1 are the shared implementations in
 *        common/helper/complex_blas3_arch22.h; only the combine stage below is
 *        operator-specific.
 *
 *        Unlike CSYRK, whose two GEMMs can absorb the sign pattern into
 *        K-concatenated operands, CSYR2K has two independent operands, so the
 *        four real products are kept apart and combined in Phase 2:
 *          T1 = Ar*Br^T + Br*Ar^T
 *          T2 = Ai*Bi^T + Bi*Ai^T
 *          T3 = Ar*Bi^T + Ai*Br^T + Br*Ai^T + Bi*Ar^T
 *        S = A*B^T + B*A^T is then (T1 - T2) + i*T3. All three products are
 *        symmetric, so the column-major (transposed) read Phase 2 performs on the
 *        row-major temp planes yields the same value -- no pointer swap anywhere.
 */

#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "csyr2k_kernel.h"
#include "csyr2k_tiling_data.h"

using namespace AscendC;

namespace {
// Complex elements handled by one Phase 2 inner iteration. A whole multiple of a
// 256B fp32 repeat so the GatherMask repeat count is exact. The buffer set below
// totals roughly 60KB, well inside the 192KB AIV UB.
constexpr uint32_t COMBINE_CHUNK = 1024;
static_assert(COMBINE_CHUNK % cblas3::ELENUM_REPEAT_FP32 == 0, "COMBINE_CHUNK must be a whole number of fp32 repeats");

// Everything the combine chunk loop needs. TPipe is deliberately NOT a member:
// rule R3 requires it to stay a local of the kernel entry.
struct Csyr2kCombineCtx {
    GlobalTensor<float> t1Gm;
    GlobalTensor<float> t2Gm;
    GlobalTensor<float> t3Gm;
    GlobalTensor<float> cGm;
    LocalTensor<float> crci; // [re | im] halves, interleaved by Gather
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> t1;
    LocalTensor<float> t2;
    LocalTensor<float> t3;
    LocalTensor<float> tmpA;
    LocalTensor<float> tmpB;
    LocalTensor<float> stage; // interleaved C_old in / C_new out
    LocalTensor<uint32_t> idxU;
    bool applyAlpha;
    bool applyBeta;
};

// The UB scratch buffers backing a Csyr2kCombineCtx. Declared in the caller's
// frame so their lifetime spans every CombineOneChunk call: the LocalTensors handed
// to ctx below stay in use for the whole column loop, so the TBufs they come from
// must outlive this function. Same arrangement as the shared layer's
// MirrorTriBuffers.
struct Csyr2kCombineBuffers {
    TBuf<TPosition::VECCALC> crci;
    TBuf<TPosition::VECCALC> t1;
    TBuf<TPosition::VECCALC> t2;
    TBuf<TPosition::VECCALC> t3;
    TBuf<TPosition::VECCALC> tmpA;
    TBuf<TPosition::VECCALC> tmpB;
    TBuf<TPosition::VECCALC> stage;
    TBuf<TPosition::VECCALC> idx;
    TBuf<TPosition::VECCALC> aux0;
    TBuf<TPosition::VECCALC> aux1;
};

__aicore__ inline void InitCombineCtx(
    TPipe& pipe, Csyr2kCombineBuffers& bufs, Csyr2kCombineCtx& ctx, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR c,
    const Csyr2kCombineTilingData& tiling)
{
    // alpha = (0,0) or k = 0 makes the host skip the whole temp allocation, so the
    // three temp planes come in null. The alpha term is not evaluated in that case
    // either (applyAlpha gates it below), so leave those tensors unbound instead of
    // handing a null address to SetGlobalBuffer. beta may still be anything, and C
    // is always bound.
    if (tiling.hasTemp != 0U) {
        ctx.t1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t1));
        ctx.t2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t2));
        ctx.t3Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t3));
    }
    ctx.cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));

    constexpr uint32_t chunk = COMBINE_CHUNK;
    constexpr uint32_t pairElems = chunk * cblas3::COMPLEX_ELENUM;

    pipe.InitBuffer(bufs.crci, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.t1, chunk * sizeof(float));
    pipe.InitBuffer(bufs.t2, chunk * sizeof(float));
    pipe.InitBuffer(bufs.t3, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpA, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpB, chunk * sizeof(float));
    pipe.InitBuffer(bufs.stage, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.idx, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux0, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux1, pairElems * sizeof(int32_t));

    ctx.crci = bufs.crci.Get<float>();
    ctx.cr = ctx.crci;
    ctx.ci = ctx.crci[chunk];
    ctx.t1 = bufs.t1.Get<float>();
    ctx.t2 = bufs.t2.Get<float>();
    ctx.t3 = bufs.t3.Get<float>();
    ctx.tmpA = bufs.tmpA.Get<float>();
    ctx.tmpB = bufs.tmpB.Get<float>();
    ctx.stage = bufs.stage.Get<float>();

    LocalTensor<int32_t> idx = bufs.idx.Get<int32_t>();
    cblas3::BuildInterleaveIndex(idx, bufs.aux0.Get<int32_t>(), bufs.aux1.Get<int32_t>(), chunk, chunk);
    ctx.idxU = idx.ReinterpretCast<uint32_t>();

    ctx.applyAlpha = (tiling.hasTemp != 0U);
    ctx.applyBeta = (tiling.betaRe != 0.0F || tiling.betaIm != 0.0F);
}

// accRe = ar*D - ai*T3, accIm = ar*T3 + ai*D, with D = T1 - T2 held in tmpA.
// A zero component of alpha is skipped rather than multiplied through. The three
// products are independent sums and any of them can reach the fp32 range limit on
// its own; multiplying such an Inf by an alpha component of 0 yields NaN, which
// would carry into a part that is otherwise exact. Skipping matches the BLAS
// convention that a zero scalar leaves its operand unreferenced.
__aicore__ inline void ApplyAlphaScale(
    Csyr2kCombineCtx& ctx, const Csyr2kCombineTilingData& tiling, uint32_t col, uint32_t row, uint32_t len)
{
    DataCopyExtParams realParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    const uint64_t tempOffset = static_cast<uint64_t>(col) * tiling.tempLd + row;

    DataCopyPad(ctx.t1, ctx.t1Gm[tempOffset], realParams, padParams);
    DataCopyPad(ctx.t2, ctx.t2Gm[tempOffset], realParams, padParams);
    DataCopyPad(ctx.t3, ctx.t3Gm[tempOffset], realParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    // D = T1 - T2, the real part of A*B^T + B*A^T.
    Sub(ctx.tmpA, ctx.t1, ctx.t2, len);
    PipeBarrier<PIPE_V>();

    if (tiling.alphaRe != 0.0F) {
        Muls(ctx.cr, ctx.tmpA, tiling.alphaRe, len);
        Muls(ctx.ci, ctx.t3, tiling.alphaRe, len);
    } else {
        Duplicate(ctx.cr, 0.0F, len);
        Duplicate(ctx.ci, 0.0F, len);
    }
    if (tiling.alphaIm != 0.0F) {
        Muls(ctx.tmpB, ctx.t3, tiling.alphaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.cr, ctx.cr, ctx.tmpB, len);
        Muls(ctx.tmpB, ctx.tmpA, tiling.alphaIm, len);
        PipeBarrier<PIPE_V>();
        Add(ctx.ci, ctx.ci, ctx.tmpB, len);
    }
    PipeBarrier<PIPE_V>();
}

// accRe += br*Cr_old - bi*Ci_old, accIm += br*Ci_old + bi*Cr_old. The three temp
// buffers are free to reuse here, the alpha term having already been folded into
// cr/ci. A zero component of beta is skipped for the same reason as in
// ApplyAlphaScale: the incoming C may itself carry a non-finite value, and
// 0 * Inf is NaN.
__aicore__ inline void ApplyBetaScale(
    Csyr2kCombineCtx& ctx, const Csyr2kCombineTilingData& tiling, uint64_t cOffset, uint32_t len)
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
        Muls(ctx.t1, ctx.tmpA, tiling.betaRe, len);
        Muls(ctx.t2, ctx.tmpB, tiling.betaRe, len);
        PipeBarrier<PIPE_V>();
        Add(ctx.cr, ctx.cr, ctx.t1, len);
        Add(ctx.ci, ctx.ci, ctx.t2, len);
        PipeBarrier<PIPE_V>();
    }
    if (tiling.betaIm != 0.0F) {
        Muls(ctx.t1, ctx.tmpB, tiling.betaIm, len);
        Muls(ctx.t2, ctx.tmpA, tiling.betaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.cr, ctx.cr, ctx.t1, len);
        Add(ctx.ci, ctx.ci, ctx.t2, len);
        PipeBarrier<PIPE_V>();
    }
}

// One contiguous run of `len` rows inside column `col`, starting at `row`.
__aicore__ inline void CombineOneChunk(
    Csyr2kCombineCtx& ctx, const Csyr2kCombineTilingData& tiling, uint32_t col, uint32_t row, uint32_t len)
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

    // C is symmetric, not Hermitian: the diagonal imaginary part is a real value
    // of the result and must be kept. This is the one place where CSYR2K must NOT
    // call cblas3::ZeroDiagonalImag.

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
extern "C" __global__ __aicore__ void csyr2k_split_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::SplitBody(a, ar, ai, tiling);
}

extern "C" __global__ __aicore__ void csyr2k_gemm_kernel_tl(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulTransLeft, CBLAS3_GEMM_TRANS_LEFT>(left, right, out, tiling);
}

extern "C" __global__ __aicore__ void csyr2k_gemm_kernel_tr(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulTransRight, CBLAS3_GEMM_TRANS_RIGHT>(left, right, out, tiling);
}

extern "C" __global__ __aicore__ void csyr2k_combine_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR c, Csyr2kCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    const bool isUpper = (tiling.uploMode == CBLAS3_UPLO_UPPER);

    TPipe pipe;
    Csyr2kCombineBuffers bufs;
    Csyr2kCombineCtx ctx;
    InitCombineCtx(pipe, bufs, ctx, t1, t2, t3, c, tiling);

    // Columns of a triangle carry very unequal work, so they are dealt out in a
    // round robin: every core then gets a uniform sample of the run lengths.
    for (uint32_t col = coreIdx; col < tiling.n; col += coreNum) {
        const uint32_t lo = isUpper ? 0U : col;
        const uint32_t hi = isUpper ? col : (tiling.n - 1U);
        const uint32_t runLen = hi - lo + 1U;

        for (uint32_t off = 0; off < runLen; off += COMBINE_CHUNK) {
            const uint32_t len = (off + COMBINE_CHUNK > runLen) ? (runLen - off) : COMBINE_CHUNK;
            CombineOneChunk(ctx, tiling, col, lo + off, len);
        }
    }
}

// ===========================================================================
//  Launchers
// ===========================================================================
void csyr2k_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling)
{
    csyr2k_split_kernel<<<blockDim, nullptr, stream>>>(a, ar, ai, tiling);
}

void csyr2k_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    if (tiling.transMode == CBLAS3_GEMM_TRANS_LEFT) {
        csyr2k_gemm_kernel_tl<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
    } else {
        csyr2k_gemm_kernel_tr<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
    }
}

void csyr2k_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR c, Csyr2kCombineTilingData tiling)
{
    csyr2k_combine_kernel<<<blockDim, nullptr, stream>>>(t1, t2, t3, c, tiling);
}

uint32_t csyr2k_gemm_single_k() { return cblas3::SINGLE_K; }
