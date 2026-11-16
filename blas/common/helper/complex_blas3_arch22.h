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
 * \file complex_blas3_arch22.h
 * \brief Device-side building blocks shared by the complex BLAS-3 operators on
 *        arch22 (chemm / csymm / cherk / csyrk / cher2k).
 *
 *        Contains the two stages that are operator-independent:
 *          Phase 0 - de-interleave a column-major complex matrix into packed
 *                    real and imaginary matrices;
 *          Phase 1 - real FP32 GEMM through the high-level Matmul API, with the
 *                    triangle skip and the K-chunk plumbing.
 *        Plus two helpers that Phase 2 needs in every operator: building the
 *        re/im interleave table, and forcing the Hermitian diagonal imaginary
 *        part to zero.
 *
 *        LAYOUT INVARIANTS - easy to get backwards, so spelled out once here and
 *        referenced from each operator.
 *
 *        (1) Phase 0 output is *packed column-major*. A column of the
 *            column-major complex input is contiguous, so one GatherMask pass
 *            over a column yields one whole column of Xr and of Xi. Read as a
 *            row-major matrix, that packed buffer is the transpose of the
 *            logical matrix, which is why Phase 1 transposes exactly one operand
 *            for the Gram-form products:
 *              trans='N': packed is A^T (k x n) -> transpose left
 *              trans='C': packed is A^T (n x k) -> transpose right
 *
 *        (2) Phase 1 writes its output *row-major*: out[i * ldc + j] = P(i, j).
 *
 *        (3) Phase 2 walks the output column by column, because the output is
 *            column-major and a column of the requested triangle is a contiguous
 *            run of rows. It therefore addresses the temp buffers as
 *            temp[j * ldc + i], which reads the *transpose* of what Phase 1
 *            wrote. Two consequences that each operator must honour:
 *              - a product and its transpose swap roles, so the host passes the
 *                corresponding pointer pair swapped;
 *              - the triangle required of Phase 1 is *mirrored* relative to
 *                uplo: uplo=UPPER needs the LOWER triangle of the row-major
 *                GEMM output.
 */

#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "complex_blas3_tiling_data.h"

namespace cblas3 {
// Targeted using-declarations (not a blanket `using namespace`) to avoid leaking
// the entire AscendC/matmul symbol set into every translation unit that
// includes this shared header (only these 19 symbols are actually used below).
using AscendC::CMPMODE;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GetMatmulApiTiling;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::MatmulImpl;
using AscendC::MatmulType;
using AscendC::PipeBarrier;
using AscendC::RoundMode;
using AscendC::SELMODE;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;

constexpr uint32_t COMPLEX_ELENUM = 2;
constexpr uint32_t BYTENUM_REPEAT = 256;
constexpr uint32_t ELENUM_REPEAT_FP32 = 64;
constexpr uint32_t BITS_PER_BYTE = 8;

// Complex elements handled by one Phase 0 inner iteration. A whole multiple of a
// 256B repeat so that the GatherMask repeat count is exact, and small enough that
// the repeat count stays under the 255-repeat instruction limit.
constexpr uint32_t SPLIT_CHUNK = 4096;
static_assert(SPLIT_CHUNK % ELENUM_REPEAT_FP32 == 0, "SPLIT_CHUNK must be a whole number of fp32 repeats");

constexpr __aicore__ uint32_t CeilDivU(uint32_t a, uint32_t b) { return (b == 0U) ? 0U : ((a + b - 1U) / b); }

constexpr __aicore__ uint32_t AlignUpU(uint32_t a, uint32_t align) { return CeilDivU(a, align) * align; }

// ---------------------------------------------------------------------------
//  Phase 1: Matmul instantiations and the tile loop
// ---------------------------------------------------------------------------
// One instantiation per transpose pattern, so isTrans is a compile-time constant
// of MatmulType. The runtime flag handed to SetTensorA/SetTensorB is what
// actually selects the transposed copy-in path, so both mechanisms are needed --
// setting only the template parameter silently computes the untransposed product.
using MmPlain = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MmTrans = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;

// Hard ceiling on the K of one Phase 1 launch; a larger runtime K is silently
// mis-computed by the static tiling, so the host splits K to this value.
// 8192, kept at the value the cherk performance runs were verified against.
//
// Raising it to 16384 was tried, so that csyrk's K-concatenated extent (2k =
// 16000 at k=8000) would fit in a single launch instead of two atomically
// accumulated chunks. Measured effect: TC_SQ_33 improved slightly (1.13e-2 ->
// 1.00e-2) but TC_SQ_58 regressed from passing to failing and TC_SQ_108 started
// failing too. All four n=8000 cases sit within a few percent of the repository's
// 1e-2 absolute cap, so the ordering change merely reshuffles which of them lands
// on which side. Reverted.
constexpr uint32_t SINGLE_K = 8192;

__aicore__ static constexpr MatmulShapeParams ShapeParams()
{
    return MatmulShapeParams{CBLAS3_TILE_M, CBLAS3_TILE_N, SINGLE_K, CBLAS3_TILE_M, CBLAS3_TILE_N, CBLAS3_BASE_K};
}

__aicore__ static constexpr MatmulBiasParams BiasParams() { return MatmulBiasParams{false}; }

// CONFIG_NORM rather than CONFIG_MDL: measured equal-or-better at every shape,
// and its default enUnitFlag = true removes the MMAD/FIXPIPE sync per base block.
constexpr MatmulConfig kConfig = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(ShapeParams(), BiasParams());

constexpr MatmulApiStaticTiling kTilingTransLeft = GetMatmulApiTiling<MmTrans, MmPlain, MmPlain, MmPlain>(kConfig);
constexpr MatmulApiStaticTiling kTilingTransRight = GetMatmulApiTiling<MmPlain, MmTrans, MmPlain, MmPlain>(kConfig);

constexpr MatmulApiStaticTiling kTilingPlain = GetMatmulApiTiling<MmPlain, MmPlain, MmPlain, MmPlain>(kConfig);

using MatmulTransLeft = MatmulImpl<MmTrans, MmPlain, MmPlain, MmPlain, kTilingTransLeft>;
using MatmulTransRight = MatmulImpl<MmPlain, MmTrans, MmPlain, MmPlain, kTilingTransRight>;
// chemm / csymm: both operands plain. See CBlas3GemmTransMode.
using MatmulPlainBoth = MatmulImpl<MmPlain, MmPlain, MmPlain, MmPlain, kTilingPlain>;

static_assert(ShapeParams().singleCoreM >= CBLAS3_TILE_M, "singleCoreM below host tile");
static_assert(ShapeParams().singleCoreN >= CBLAS3_TILE_N, "singleCoreN below host tile");
static_assert(ShapeParams().singleCoreK == SINGLE_K, "singleCoreK must match SINGLE_K");

struct GemmTile {
    uint32_t rowBase;
    uint32_t colBase;
    uint32_t rowCount;
    uint32_t colCount;
};

__aicore__ inline GemmTile ResolveGemmTile(const CBlas3GemmTilingData& t, uint32_t tile)
{
    GemmTile r;
    const uint32_t mb = tile / t.nBlocks;
    const uint32_t nb = tile % t.nBlocks;
    r.rowBase = mb * CBLAS3_TILE_M;
    r.colBase = nb * CBLAS3_TILE_N;
    r.rowCount = (r.rowBase + CBLAS3_TILE_M > t.m) ? (t.m - r.rowBase) : CBLAS3_TILE_M;
    r.colCount = (r.colBase + CBLAS3_TILE_N > t.n) ? (t.n - r.colBase) : CBLAS3_TILE_N;
    return r;
}

// Does this output tile carry any element that Phase 2 will read?
//
// Phase 2 reads temp index (row=j, col=i) for (i, j) inside the uplo triangle, so
// the triangle required of the row-major GEMM output is mirrored: uplo=UPPER
// needs col <= row, uplo=LOWER needs col >= row. Tiles fully outside are skipped,
// which removes close to half of the Cube work for large n. CBLAS3_UPLO_NONE
// keeps every tile, for the operators whose output is a full m x n matrix.
__aicore__ inline bool GemmTileIsNeeded(const GemmTile& r, uint32_t uploMode)
{
    if (uploMode == CBLAS3_UPLO_NONE) {
        return true;
    }
    if (uploMode == CBLAS3_UPLO_UPPER) {
        return r.colBase <= (r.rowBase + r.rowCount - 1U);
    }
    return (r.colBase + r.colCount - 1U) >= r.rowBase;
}

// SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc). The copy-in stage derives the
// GM row stride of A from orgM when A is transposed and from orgKa otherwise,
// and the row stride of B from orgKb when B is transposed and from orgN
// otherwise; orgKc is always the row stride of C. Filling these by the
// untransposed habit silently mis-computes the transposed case.
//
// Which of the five values actually carries a stride depends on the mode, so
// they are filled per mode rather than uniformly:
//   TRANS_LEFT  - A is transposed, so A's stride comes from orgM (= m = the
//                 packed row stride here) and B's from orgN. orgKa/orgKb are
//                 not consulted for strides, so they carry the K extent.
//   TRANS_RIGHT - A is not transposed and B is, so both strides come from
//                 orgKa / orgKb. With K-concatenated operands the row stride
//                 no longer equals K, so packedLd must be passed here instead
//                 of t.k.
// The K extent of this launch is carried by SetSingleShape in the caller.
template <class MM, uint32_t TRANS_MODE>
__aicore__ inline void SetGemmOrgShape(MM& mm, const CBlas3GemmTilingData& t)
{
    if constexpr (TRANS_MODE == CBLAS3_GEMM_TRANS_NONE) {
        // Neither operand transposed, so A's row stride comes from orgKa and B's
        // from orgN. For chemm / csymm the left operand is (m x k) row-major with
        // row stride k and the right one is (k x n) row-major with row stride n, so
        // both strides coincide with the logical extents and the plain filling is
        // correct for either side.
        // orgKa carries the left operand's row stride, orgN the right operand's.
        // The triangular operand of chemm / csymm is padded to a multiple of the
        // expansion tile, so its stride is larger than its logical extent and the
        // two must be passed separately.
        mm.SetOrgShape(
            static_cast<int>(t.m), static_cast<int>(t.ldB), static_cast<int>(t.ldA), static_cast<int>(t.ldA),
            static_cast<int>(t.ldc));
    } else if constexpr (TRANS_MODE == CBLAS3_GEMM_TRANS_LEFT) {
        mm.SetOrgShape(
            static_cast<int>(t.m), static_cast<int>(t.n), static_cast<int>(t.k), static_cast<int>(t.k),
            static_cast<int>(t.ldc));
    } else {
        mm.SetOrgShape(
            static_cast<int>(t.m), static_cast<int>(t.n), static_cast<int>(t.packedLd), static_cast<int>(t.packedLd),
            static_cast<int>(t.ldc));
    }
}

// One GEMM tile's three GM views, sized so the declared length never runs past
// the real allocation: the base pointer already carries the row/col offset, so
// the last row of the tile only extends to its own column count.
template <class MM, uint32_t TRANS_MODE>
__aicore__ inline void ResolveGemmOperands(
    const GemmTile& r, const CBlas3GemmTilingData& t, __gm__ float* leftBase, __gm__ float* rightBase,
    __gm__ float* outBase, GlobalTensor<float>& aGlobal, GlobalTensor<float>& bGlobal, GlobalTensor<float>& cGlobal)
{
    if constexpr (TRANS_MODE == CBLAS3_GEMM_TRANS_NONE) {
        // A is (m x k) row-major with row stride ldA, B is (k x n) row-major
        // with row stride ldB -- the strides differ between the two operands
        // here, unlike the Gram products where one packedLd covers both.
        const uint64_t aLen = static_cast<uint64_t>(r.rowCount - 1U) * t.ldA + t.kCount;
        const uint64_t bLen = static_cast<uint64_t>(t.kCount - 1U) * t.ldB + r.colCount;
        aGlobal.SetGlobalBuffer(leftBase + static_cast<uint64_t>(r.rowBase) * t.ldA + t.kBase, aLen);
        bGlobal.SetGlobalBuffer(rightBase + static_cast<uint64_t>(t.kBase) * t.ldB + r.colBase, bLen);
    } else if constexpr (TRANS_MODE == CBLAS3_GEMM_TRANS_LEFT) {
        const uint64_t aLen = static_cast<uint64_t>(t.kCount - 1U) * t.packedLd + r.rowCount;
        const uint64_t bLen = static_cast<uint64_t>(t.kCount - 1U) * t.packedLd + r.colCount;
        aGlobal.SetGlobalBuffer(leftBase + static_cast<uint64_t>(t.kBase) * t.packedLd + r.rowBase, aLen);
        bGlobal.SetGlobalBuffer(rightBase + static_cast<uint64_t>(t.kBase) * t.packedLd + r.colBase, bLen);
    } else {
        const uint64_t aLen = static_cast<uint64_t>(r.rowCount - 1U) * t.packedLd + t.kCount;
        const uint64_t bLen = static_cast<uint64_t>(r.colCount - 1U) * t.packedLd + t.kCount;
        aGlobal.SetGlobalBuffer(leftBase + static_cast<uint64_t>(r.rowBase) * t.packedLd + t.kBase, aLen);
        bGlobal.SetGlobalBuffer(rightBase + static_cast<uint64_t>(r.colBase) * t.packedLd + t.kBase, bLen);
    }
    cGlobal.SetGlobalBuffer(
        outBase + static_cast<uint64_t>(r.rowBase) * t.ldc + r.colBase,
        static_cast<uint64_t>(r.rowCount - 1U) * t.ldc + r.colCount);
}

// TRANS_MODE mirrors the isTrans of MM's A_TYPE / B_TYPE, so every branch on it
// folds away and the body is specialised for one layout.
template <class MM, uint32_t TRANS_MODE>
__aicore__ inline void GemmBody(GM_ADDR left, GM_ADDR right, GM_ADDR out, const CBlas3GemmTilingData& t)
{
    // Guard the divisor and the loop step before use: rule R8 requires an
    // explicit non-zero check on any divisor, and a zero step would spin.
    if (t.nBlocks == 0U || t.mBlocks == 0U || GetBlockNum() == 0) {
        return;
    }
    const uint32_t totalTiles = t.mBlocks * t.nBlocks;
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());

    __gm__ float* leftBase = reinterpret_cast<__gm__ float*>(left);
    __gm__ float* rightBase = reinterpret_cast<__gm__ float*>(right);
    __gm__ float* outBase = reinterpret_cast<__gm__ float*>(out);

    TPipe pipe;
    MM mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    SetGemmOrgShape<MM, TRANS_MODE>(mm, t);

    // Deal out only the tiles that are actually needed, rather than striding over
    // all tiles and skipping the ones outside the triangle.
    //
    // Striding over all tiles couples the core index to the tile geometry: with
    // tile = mb * nBlocks + nb, core c owns the tiles where mb * nBlocks + nb is
    // congruent to c modulo coreNum, which for nBlocks == coreNum collapses to
    // nb == c -- one fixed column band, so under uplo skipping the cores holding
    // the short bands idle. nBlocks == coreNum - 1 is worse still: the relation
    // becomes nb - mb == c, and combined with nb <= mb most cores get no tile at
    // all. Measured on 910B4: n=2432 (nBlocks=19) ran 2.13 ms against 1.75 ms for
    // the larger n=2688 (nBlocks=21), i.e. more time for 35% less work; counting
    // needed tiles first recovered 66% at that shape.
    //
    // The scan is scalar work proportional to nBlocks^2 (about 4k iterations at
    // n=8000), negligible next to the MADs it schedules.
    uint32_t needed = 0;
    for (uint32_t tile = 0; tile < totalTiles; ++tile) {
        const GemmTile r = ResolveGemmTile(t, tile);
        if (!GemmTileIsNeeded(r, t.uploMode)) {
            continue;
        }
        if ((needed++ % coreNum) != coreIdx) {
            continue;
        }

        GlobalTensor<float> aGlobal;
        GlobalTensor<float> bGlobal;
        GlobalTensor<float> cGlobal;
        ResolveGemmOperands<MM, TRANS_MODE>(r, t, leftBase, rightBase, outBase, aGlobal, bGlobal, cGlobal);

        mm.SetSingleShape(static_cast<int>(r.rowCount), static_cast<int>(r.colCount), static_cast<int>(t.kCount));
        mm.SetTensorA(aGlobal, TRANS_MODE == CBLAS3_GEMM_TRANS_LEFT);
        mm.SetTensorB(bGlobal, TRANS_MODE == CBLAS3_GEMM_TRANS_RIGHT);
        mm.IterateAll(cGlobal, static_cast<uint8_t>(t.enAtomic));
    }

    mm.End();
}

// ---------------------------------------------------------------------------
//  Phase 0: complex -> packed real Xr, Xi
// ---------------------------------------------------------------------------
// Length of the chunk starting at `base`, clipped to the tail of the column.
__aicore__ inline uint32_t SplitChunkLen(uint32_t base, uint32_t rows)
{
    return (base + SPLIT_CHUNK > rows) ? (rows - base) : SPLIT_CHUNK;
}

// Stages one chunk of a complex column into UB and de-interleaves it, leaving the
// real lanes in `re` and the imaginary ones in `im`. Shared by both Phase 0
// variants below, which differ only in what they write out afterwards.
__aicore__ inline void LoadChunkDeinterleaved(
    const GlobalTensor<float>& srcGm, uint64_t srcColOffset, uint32_t base, uint32_t len,
    const LocalTensor<float>& stage, const LocalTensor<float>& re, const LocalTensor<float>& im)
{
    uint64_t rsvdCnt = 0;

    DataCopyExtParams inParams{1U, len * COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    DataCopyPad(stage, srcGm[srcColOffset + static_cast<uint64_t>(base) * COMPLEX_ELENUM], inParams, padParams);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    const uint16_t repeats =
        static_cast<uint16_t>(CeilDivU(len * COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), BYTENUM_REPEAT));
    GatherMask(re, stage, 1, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    GatherMask(im, stage, 2, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();
}

// Both queues must wait on MTE3 before the next iteration reuses the staging
// buffers: MTE3_V covers the GatherMask that writes them, MTE3_MTE2 covers the
// DataCopyPad that refills stage.
__aicore__ inline void SyncAfterChunkStore()
{
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

// Operator-independent: the tiling only describes a column-major complex matrix
// and where to put its two real halves.
__aicore__ inline void SplitBody(GM_ADDR src, GM_ADDR dstRe, GM_ADDR dstIm, const CBlas3SplitTilingData& tiling)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());

    GlobalTensor<float> srcGm;
    GlobalTensor<float> reGm;
    GlobalTensor<float> imGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    reGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstRe));
    imGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstIm));

    TPipe pipe;
    TBuf<TPosition::VECCALC> stageBuf;
    TBuf<TPosition::VECCALC> reBuf;
    TBuf<TPosition::VECCALC> imBuf;
    pipe.InitBuffer(stageBuf, SPLIT_CHUNK * COMPLEX_ELENUM * sizeof(float));
    pipe.InitBuffer(reBuf, SPLIT_CHUNK * sizeof(float));
    pipe.InitBuffer(imBuf, SPLIT_CHUNK * sizeof(float));

    LocalTensor<float> stage = stageBuf.Get<float>();
    LocalTensor<float> re = reBuf.Get<float>();
    LocalTensor<float> im = imBuf.Get<float>();

    // Columns are uniform work here, so a plain round robin balances the cores
    // exactly and keeps every core busy when cols is just above the core count.
    for (uint32_t col = coreIdx; col < tiling.cols; col += coreNum) {
        const uint64_t srcColOffset = static_cast<uint64_t>(col) * tiling.lda * COMPLEX_ELENUM;
        const uint64_t dstColOffset = static_cast<uint64_t>(col) * tiling.packedLd;

        for (uint32_t base = 0; base < tiling.rows; base += SPLIT_CHUNK) {
            const uint32_t len = SplitChunkLen(base, tiling.rows);
            LoadChunkDeinterleaved(srcGm, srcColOffset, base, len, stage, re, im);
            if (tiling.scaleEnabled != 0U) {
                Muls(re, re, tiling.scaleFactor, len);
                Muls(im, im, tiling.negateImag != 0U ? -tiling.scaleFactor : tiling.scaleFactor, len);
                PipeBarrier<PIPE_V>();
            } else if (tiling.negateImag != 0U) {
                Muls(im, im, -1.0F, len);
                PipeBarrier<PIPE_V>();
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            DataCopyExtParams outParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(reGm[dstColOffset + base], re, outParams);
            DataCopyPad(imGm[dstColOffset + base], im, outParams);

            SyncAfterChunkStore();
        }
    }
}

// Phase 0 variant emitting the three K-concatenated operands P, Q, R described in
// CBlas3SplitConcatTilingData. One pass over the input produces all three, so the
// input is read once and every write is contiguous.
//
//   P = [Xr | Xi]      Q = [Xr | -Xi]      R = [Xi | Xr]
//
// Per input column the six destination segments are:
//   P: re at base,               im at base + blockStride
//   Q: re at base,              -im at base + blockStride
//   R: im at base,               re at base + blockStride
// where base = col * colStride. colStride and blockStride encode the trans mode
// (see the struct comment); the caller supplies both so this body stays layout
// agnostic.
// The three K-concatenated destinations, plus the UB scratch the chunk loop needs.
struct SplitConcatCtx {
    GlobalTensor<float> srcGm;
    GlobalTensor<float> pGm;
    GlobalTensor<float> qGm;
    GlobalTensor<float> rGm;
    LocalTensor<float> stage;
    LocalTensor<float> re;
    LocalTensor<float> im;
    LocalTensor<float> negIm;
};

// Scratch buffers backing a SplitConcatCtx, kept alive in the caller's frame.
struct SplitConcatBuffers {
    TBuf<TPosition::VECCALC> stage;
    TBuf<TPosition::VECCALC> re;
    TBuf<TPosition::VECCALC> im;
    TBuf<TPosition::VECCALC> neg;
};

__aicore__ inline void InitSplitConcatCtx(
    TPipe& pipe, SplitConcatBuffers& bufs, SplitConcatCtx& ctx, GM_ADDR src, GM_ADDR dstP, GM_ADDR dstQ, GM_ADDR dstR)
{
    ctx.srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    ctx.pGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstP));
    ctx.qGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstQ));
    ctx.rGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstR));

    pipe.InitBuffer(bufs.stage, SPLIT_CHUNK * COMPLEX_ELENUM * sizeof(float));
    pipe.InitBuffer(bufs.re, SPLIT_CHUNK * sizeof(float));
    pipe.InitBuffer(bufs.im, SPLIT_CHUNK * sizeof(float));
    pipe.InitBuffer(bufs.neg, SPLIT_CHUNK * sizeof(float));

    ctx.stage = bufs.stage.Get<float>();
    ctx.re = bufs.re.Get<float>();
    ctx.im = bufs.im.Get<float>();
    ctx.negIm = bufs.neg.Get<float>();
}

// Which pair of operands Phase 0 is asked to emit.
//
//   kConcatPQR   : P = [Xr | Xi], Q = [Xr | -Xi], R = [Xi | Xr] as concatenated
//                  halves, selected with blockStride spanning a whole half.
//   kInterleavePQ: P = [Xi_j, Xr_j], Q = [Xr_j, -Xi_j] per input column, selected
//                  with colStride = 2*packedLd and blockStride = packedLd. Only P
//                  and Q are written.
//
// The interleaved form exists for the sign-cancelling products. Its accumulator
// alternates the two signs per input column, so a term and its counterpart cancel
// while the running sum is still small; summing them as separate blocks lets each
// half reach the fp32 range limit on its own, and Inf - Inf is NaN. On the
// RANDOM_EXTREME shapes, whose input carries +/-FLT_MAX, that is the difference
// between matching the reference everywhere and returning NaN.
enum class SplitEmit { kConcatPQR, kInterleavePQ };

// Emits one chunk to the destination segments the mode asks for.
__aicore__ inline void StoreSplitConcatChunk(
    const SplitConcatCtx& ctx, SplitEmit mode, uint64_t lo, uint64_t hi, uint32_t len)
{
    DataCopyExtParams outParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    if (mode == SplitEmit::kInterleavePQ) {
        DataCopyPad(ctx.pGm[lo], ctx.im, outParams);
        DataCopyPad(ctx.pGm[hi], ctx.re, outParams);
        DataCopyPad(ctx.qGm[lo], ctx.re, outParams);
        DataCopyPad(ctx.qGm[hi], ctx.negIm, outParams);
        return;
    }
    DataCopyPad(ctx.pGm[lo], ctx.re, outParams);
    DataCopyPad(ctx.pGm[hi], ctx.im, outParams);
    DataCopyPad(ctx.qGm[lo], ctx.re, outParams);
    DataCopyPad(ctx.qGm[hi], ctx.negIm, outParams);
    DataCopyPad(ctx.rGm[lo], ctx.im, outParams);
    DataCopyPad(ctx.rGm[hi], ctx.re, outParams);
}

__aicore__ inline void SplitConcatCore(
    GM_ADDR src, GM_ADDR dstP, GM_ADDR dstQ, GM_ADDR dstR, const CBlas3SplitConcatTilingData& tiling, SplitEmit mode)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());

    TPipe pipe;
    SplitConcatBuffers bufs;
    SplitConcatCtx ctx;
    InitSplitConcatCtx(pipe, bufs, ctx, src, dstP, dstQ, dstR);

    for (uint32_t col = coreIdx; col < tiling.cols; col += coreNum) {
        const uint64_t srcColOffset = static_cast<uint64_t>(col) * tiling.lda * COMPLEX_ELENUM;
        const uint64_t dstBase = static_cast<uint64_t>(col) * tiling.colStride;

        for (uint32_t base = 0; base < tiling.rows; base += SPLIT_CHUNK) {
            const uint32_t len = SplitChunkLen(base, tiling.rows);
            LoadChunkDeinterleaved(ctx.srcGm, srcColOffset, base, len, ctx.stage, ctx.re, ctx.im);
            Muls(ctx.negIm, ctx.im, -1.0F, len);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            const uint64_t lo = dstBase + base;
            StoreSplitConcatChunk(ctx, mode, lo, lo + tiling.blockStride, len);

            SyncAfterChunkStore();
        }
    }
}

__aicore__ inline void SplitConcatBody(
    GM_ADDR src, GM_ADDR dstP, GM_ADDR dstQ, GM_ADDR dstR, const CBlas3SplitConcatTilingData& tiling)
{
    SplitConcatCore(src, dstP, dstQ, dstR, tiling, SplitEmit::kConcatPQR);
}

// dstP doubles as the unused R slot: kInterleavePQ writes only P and Q.
__aicore__ inline void SplitInterleaveBody(
    GM_ADDR src, GM_ADDR dstP, GM_ADDR dstQ, const CBlas3SplitConcatTilingData& tiling)
{
    SplitConcatCore(src, dstP, dstQ, dstP, tiling, SplitEmit::kInterleavePQ);
}

// ---------------------------------------------------------------------------
//  Phase 2 helpers (the combine body itself is per-operator)
// ---------------------------------------------------------------------------
// Negate a packed real buffer of `count` floats in place, spread over all cores.
//
// CHER2K needs Bi with both signs: its real part accumulates +Ai*Bi^T and
// +Bi*Ai^T, its imaginary part -Ar*Bi^T and -Bi*Ar^T. Atomic accumulation can
// only add, and a fifth packed buffer would push the n=k=8000 workspace to
// 2.15GiB against a 2GiB cap, so the sign is flipped in place between the two
// groups of GEMMs instead. At n=k=8000 this moves 512MB, roughly 1% of the GEMM
// time it enables.
__aicore__ inline void NegateBody(GM_ADDR buf, uint32_t count)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U || count == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());

    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(buf), count);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf0;
    pipe.InitBuffer(buf0, SPLIT_CHUNK * sizeof(float));
    LocalTensor<float> local = buf0.Get<float>();

    // Contiguous per-core slabs, aligned so every DataCopyPad starts on a 32B
    // boundary; only the last slab is short.
    const uint32_t perCore = AlignUpU(CeilDivU(count, coreNum), ELENUM_REPEAT_FP32);
    const uint64_t begin = static_cast<uint64_t>(coreIdx) * perCore;
    if (begin >= count) {
        return;
    }
    const uint64_t end = ((begin + perCore) > count) ? count : (begin + perCore);

    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    for (uint64_t off = begin; off < end; off += SPLIT_CHUNK) {
        const uint32_t len = static_cast<uint32_t>(((off + SPLIT_CHUNK) > end) ? (end - off) : SPLIT_CHUNK);
        DataCopyExtParams params{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(local, gm[off], params, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Muls(local, local, -1.0F, len);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(gm[off], local, params);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

// Byte-offset table that interleaves a [re | im] pair of half-buffers into one
// complex run: idx[2t] = 4*t, idx[2t+1] = 4*(halfStride + t). Built once per
// kernel with integer vector ops, so no host-side auxiliary buffer and no H2D
// copy is needed on the critical path. Integer rather than float arithmetic
// because Gather takes byte offsets and fp32 is exact only up to 2^24 bytes --
// a limit that would be invisible in the source if the chunk size grew.
__aicore__ inline void BuildInterleaveIndex(
    const LocalTensor<int32_t>& idx, const LocalTensor<int32_t>& s0, const LocalTensor<int32_t>& s1, uint32_t pairs,
    uint32_t halfStride)
{
    const uint32_t total = pairs * COMPLEX_ELENUM;
    CreateVecIndex(idx, 0, total);
    PipeBarrier<PIPE_V>();
    ShiftRight(s0, idx, 1, total);
    PipeBarrier<PIPE_V>();
    Muls(s1, s0, 2, total);
    PipeBarrier<PIPE_V>();
    Sub(s1, idx, s1, total);
    PipeBarrier<PIPE_V>();
    Muls(s0, s0, static_cast<int32_t>(sizeof(float)), total);
    Muls(s1, s1, static_cast<int32_t>(halfStride * sizeof(float)), total);
    PipeBarrier<PIPE_V>();
    Add(idx, s0, s1, total);
    PipeBarrier<PIPE_V>();
}

// Phase 0b (chemm / csymm triangle -> full square expansion) and its
// MirrorTriBody entry point moved to complex_blas3_expand_arch22.h: it is only
// pulled in by the two side/uplo operators, not by the three rank-K ones, and
// keeping it here would push this file over the per-header line guideline.

// Zero the imaginary lane that sits on the Hermitian diagonal.
//
// The two GEMM accumulations whose difference forms the imaginary part cancel
// exactly only in real arithmetic; in practice they differ by up to O(1), so the
// diagonal imaginary part must be forced to zero. Done with a compare/select mask
// rather than a scalar SetValue, which the repository coding rules forbid.
// Only the Hermitian operators (cherk / cher2k) call this; the symmetric ones
// (csyrk / csymm) must not.
__aicore__ inline void ZeroDiagonalImag(
    const LocalTensor<float>& ci, const LocalTensor<float>& laneIdx, const LocalTensor<uint8_t>& mask,
    uint32_t alignedLen, int32_t diagLocal)
{
    if (diagLocal < 0 || static_cast<uint32_t>(diagLocal) >= alignedLen) {
        return;
    }
    Compares(mask, laneIdx, static_cast<float>(diagLocal), CMPMODE::NE, alignedLen);
    PipeBarrier<PIPE_V>();
    Select(ci, mask, ci, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
    PipeBarrier<PIPE_V>();
}
} // namespace cblas3
