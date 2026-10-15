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
 * \file cherk_kernel.cpp
 * \brief CHERK Phase 0 and Phase 2 AIV kernels (arch35).
 *        Phase 0 (Deinterleave): splits complex A into real/imag float matrices.
 *        Phase 2 (Combine): merges 4 real GEMM results into complex C.
 *          Cr = t1 + t2 (symmetric), Ci = t3 - t4 (anti-symmetric, diagonal = 0)
 *          C = alpha*(Cr + i*Ci) + beta*C_old, only writes uplo triangle.
 *        Uses Te::Copy for GM↔UB, Te::Transform for vector compute,
 *        DeInterleave/Interleave for complex deinterleave/interleave.
 *        LOWER diagonal blocks: store full column + restore non-uplo from C_old.
 *        Hermitian diagonal imaginary part is forced to zero via SetValue,
 *        because t3[diag] and t4[diag] come from two independent GEMM
 *        computations whose floating-point difference is non-zero (O(0.1-1.0)).
 */

#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "cherk_tiling_data.h"
#include "cherk_kernel.h"
#include "tensor_api/tensor.h"

using namespace AscendC;
namespace te = AscendC::Te;

constexpr uint32_t SCALE_BLOCK = CHERK_ARCH35_SCALE_BLOCK;
constexpr uint32_t SCALE_UB_FLOATS = SCALE_BLOCK * SCALE_BLOCK;
constexpr uint32_t SCALE_UB_CPLX_FLOATS = SCALE_BLOCK * SCALE_BLOCK * 2;

template<typename T>
__aicore__ inline auto MakeUbTensor1D(const LocalTensor<T>& lt, uint32_t offset, uint32_t count) {
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, T>(
            lt.GetPhyAddr() + static_cast<uint64_t>(offset) * sizeof(T)),
        te::MakeFrameLayout<te::NDLayoutPtn>(
            static_cast<uint64_t>(1), static_cast<uint64_t>(count)));
}

// ==========================================================================
//  Phase 0: Deinterleave complex A → Ar, Ai (AIV-only)
//  Reads column-major complex A from GM, splits into real/imag float
//  matrices in GM (tightly packed, row stride = rows).
//  Multi-core split by rows; inner 2D tiles of SCALE_BLOCK × SCALE_BLOCK.
// ==========================================================================

struct DeintCtx {
    GlobalTensor<float> aGM, arGM, aiGM;
    TBuf<TPosition::VECIN> aInBuf, arOutBuf, aiOutBuf;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t lda = 0;
    uint32_t rowStart = 0;
    uint32_t rowEnd = 0;
};

// GM→UB: load complex A block via te::Copy (2D strided NDExtLayoutPtn)
__aicore__ inline void LoadComplexABlock(DeintCtx& ctx,
    uint32_t iBase, uint32_t jBase, uint32_t blockRows, uint32_t blockCols,
    uint32_t ubCplxStride)
{
    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    auto aGm = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.aGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(ctx.cols), static_cast<uint64_t>(ctx.lda * 2)));
    auto aBlock = aGm.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
        te::MakeShape(static_cast<uint64_t>(blockCols), static_cast<uint64_t>(blockRows * 2)));
    LocalTensor<float> aInUb = ctx.aInBuf.Get<float>();
    auto aUb = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(aInUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(blockCols), static_cast<uint64_t>(ubCplxStride)));
    te::Copy(copyGM2UB, aUb, aBlock);
}

// V: DeInterleave complex [re0,im0,re1,im1,...] → [re0,re1,...] + [im0,im1,...]
__aicore__ inline void DeinterleaveBlock(DeintCtx& ctx, uint32_t cplxCnt)
{
    LocalTensor<float> aInUb = ctx.aInBuf.Get<float>();
    LocalTensor<float> arUb = ctx.arOutBuf.Get<float>();
    LocalTensor<float> aiUb = ctx.aiOutBuf.Get<float>();
    DeInterleave(arUb, aiUb, aInUb, static_cast<int32_t>(cplxCnt));
    PipeBarrier<PIPE_ALL>();
}

// UB→GM: store Ar and Ai blocks via te::Copy (2D strided NDExtLayoutPtn)
__aicore__ inline void StoreRealBlocks(DeintCtx& ctx,
    uint32_t iBase, uint32_t jBase, uint32_t blockRows, uint32_t blockCols,
    uint32_t ubTempStride)
{
    auto copyUB2GM = te::MakeCopy(te::CopyUB2GM{});
    LocalTensor<float> arUb = ctx.arOutBuf.Get<float>();
    LocalTensor<float> aiUb = ctx.aiOutBuf.Get<float>();

    auto gmLayout = te::MakeFrameLayout<te::NDExtLayoutPtn>(
        static_cast<uint64_t>(ctx.cols), static_cast<uint64_t>(ctx.rows));
    auto ubLayout = te::MakeFrameLayout<te::NDExtLayoutPtn>(
        static_cast<uint64_t>(blockCols), static_cast<uint64_t>(ubTempStride));
    auto coord = te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase));
    auto shape = te::MakeShape(static_cast<uint64_t>(blockCols), static_cast<uint64_t>(blockRows));

    auto arGm = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.arGM.GetPhyAddr())), gmLayout);
    auto arUbT = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(arUb.GetPhyAddr()), ubLayout);
    te::Copy(copyUB2GM, arGm.Slice(coord, shape), arUbT);

    auto aiGm = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.aiGM.GetPhyAddr())), gmLayout);
    auto aiUbT = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(aiUb.GetPhyAddr()), ubLayout);
    te::Copy(copyUB2GM, aiGm.Slice(coord, shape), aiUbT);
}

// Process one 2D tile: load → deinterleave → store
__aicore__ inline void ProcessDeinterleaveTile(DeintCtx& ctx,
    uint32_t iBase, uint32_t jBase, uint32_t blockRows, uint32_t blockCols)
{
    uint32_t ubTempStride = RoundUp<uint32_t>(blockRows, CHERK_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t ubCplxStride = 2 * ubTempStride;
    uint32_t cplxCnt = ubCplxStride * blockCols;

    // aInBuf is shared by every tile of this core. The previous tile's V-stage
    // DeinterleaveBlock still reads aInBuf when this tile's MTE2 load is issued:
    // MTE2 and V are independent pipes with no implicit ordering, so without
    // this V-to-MTE2 handshake the load can overwrite aInBuf first and the
    // previous tile then deinterleaves THIS tile's data (mirrors the cher2k
    // arch35 K1 guard, whose comment records the same hazard).
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);

    LoadComplexABlock(ctx, iBase, jBase, blockRows, blockCols, ubCplxStride);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    DeinterleaveBlock(ctx, cplxCnt);

    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    StoreRealBlocks(ctx, iBase, jBase, blockRows, blockCols, ubTempStride);

    // WAR hazard on arOutBuf/aiOutBuf: StoreRealBlocks issues MTE3 reads of the
    // deinterleaved blocks, while the NEXT tile's DeinterleaveBlock (V pipe)
    // writes the same buffers; V and MTE3 have no implicit ordering. Without
    // this MTE3-to-V handshake the next tile's compute can retire first and
    // clobber lanes the store burst has not read yet, leaving whole row bands
    // of Ar/Ai stale -- the downstream GEMMs then spread them into error
    // stripes of t1..t4 (mirrors the cher2k arch35 K1 store-side guard).
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
}

// Loop over 2D tiles within core's row range
__aicore__ inline void ProcessDeinterleaveLoop(DeintCtx& ctx)
{
    for (uint32_t jBase = 0; jBase < ctx.cols; jBase += SCALE_BLOCK) {
        uint32_t blockCols = Min<uint32_t>(SCALE_BLOCK, ctx.cols - jBase);
        for (uint32_t iBase = ctx.rowStart; iBase < ctx.rowEnd; iBase += SCALE_BLOCK) {
            uint32_t blockRows = Min<uint32_t>(SCALE_BLOCK, ctx.rowEnd - iBase);
            ProcessDeinterleaveTile(ctx, iBase, jBase, blockRows, blockCols);
        }
    }
}

extern "C" __global__ __aicore__ void cherk_deinterleave_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai,
    const CherkDeinterleaveTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    DeintCtx ctx;
    ctx.aGM.SetGlobalBuffer((__gm__ float*)a);
    ctx.arGM.SetGlobalBuffer((__gm__ float*)ar);
    ctx.aiGM.SetGlobalBuffer((__gm__ float*)ai);
    ctx.rows = tiling.rows;
    ctx.cols = tiling.cols;
    ctx.lda = tiling.lda;
    ctx.rowStart = GetBlockIdx() * tiling.rowsPerCore;
    ctx.rowEnd = Min<uint32_t>(ctx.rowStart + tiling.rowsPerCore, tiling.rows);
    pipe.InitBuffer(ctx.aInBuf, SCALE_UB_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.arOutBuf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.aiOutBuf, SCALE_UB_FLOATS * sizeof(float));
    ProcessDeinterleaveLoop(ctx);
}

void cherk_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai,
    const CherkDeinterleaveTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    cherk_deinterleave_kernel<<<numBlocks, nullptr, stream>>>(
        a, ar, ai, tiling);
}

// ==========================================================================
//  Phase 2: Combine/Scale/Hermitian (AIV-only)
// ==========================================================================

struct CherkCtx {
    CherkCombineTilingData tiling;
    GlobalTensor<float> t1GM, t2GM, t3GM, t4GM, cGM;
    TBuf<TPosition::VECIN> t1Buf, t2Buf, t3Buf, t4Buf, cInBuf;
    TBuf<TPosition::VECOUT> cOutBuf;
    uint32_t rowStart = 0;
    uint32_t rowEnd = 0;
};

static __aicore__ inline uint32_t UploCount(bool uploUpper, uint32_t iBase, uint32_t absJ, uint32_t rows) {
    if (uploUpper) {
        return (absJ >= iBase) ? Min(absJ - iBase + 1, rows) : 0;
    }
    return (absJ < iBase) ? rows : (rows - (absJ - iBase));
}

// GM→UB: load t1..t4 (real GEMM results) via Te::Copy (2D strided NDExtLayoutPtn)
__aicore__ inline void LoadGemmResults(CherkCtx& ctx,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    uint32_t tempLdc = ctx.tiling.tempLdc;
    uint32_t ubStride = RoundUp<uint32_t>(rows, CHERK_ARCH35_ELEMENTS_PER_BLOCK);

    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();

    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    auto gmLayout = te::MakeFrameLayout<te::NDExtLayoutPtn>(
        static_cast<uint64_t>(ctx.tiling.n), static_cast<uint64_t>(tempLdc));
    auto ubLayout = te::MakeFrameLayout<te::NDExtLayoutPtn>(
        static_cast<uint64_t>(cols), static_cast<uint64_t>(ubStride));
    auto coord = te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase));
    auto shape = te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows));

    auto t1Gm = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.t1GM.GetPhyAddr())), gmLayout);
    auto t2Gm = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.t2GM.GetPhyAddr())), gmLayout);
    auto t3Gm = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.t3GM.GetPhyAddr())), gmLayout);
    auto t4Gm = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.t4GM.GetPhyAddr())), gmLayout);

    te::Copy(copyGM2UB, te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(t1Ub.GetPhyAddr()), ubLayout), t1Gm.Slice(coord, shape));
    te::Copy(copyGM2UB, te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(t2Ub.GetPhyAddr()), ubLayout), t2Gm.Slice(coord, shape));
    te::Copy(copyGM2UB, te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(t3Ub.GetPhyAddr()), ubLayout), t3Gm.Slice(coord, shape));
    te::Copy(copyGM2UB, te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(t4Ub.GetPhyAddr()), ubLayout), t4Gm.Slice(coord, shape));
}

// GM→UB: load C_old (interleaved complex) via Te::Copy (2D strided)
// ubCplxStride = 2 * ubTempStride ensures DeInterleave output matches t1Buf/t3Buf layout
__aicore__ inline void LoadCOld(CherkCtx& ctx,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    uint32_t ldc = ctx.tiling.ldc;
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CHERK_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t ubCplxStride = 2 * ubTempStride;
    LocalTensor<float> cInUb = ctx.cInBuf.Get<float>();

    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    auto gmT = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.cGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(ctx.tiling.n), static_cast<uint64_t>(ldc * 2)));
    auto gmBlock = gmT.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows * 2)));
    auto ubT = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cInUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(cols), static_cast<uint64_t>(ubCplxStride)));
    te::Copy(copyGM2UB, ubT, gmBlock);
}

// V: compute Cr = alpha*(t1+t2), Ci = alpha*(t3-t4) via Te::Transform
__aicore__ inline void ComputeCrCi(CherkCtx& ctx, uint32_t rows, uint32_t cols, float alpha)
{
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CHERK_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t cnt = ubTempStride * cols;

    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();

    auto t1T = MakeUbTensor1D(t1Ub, 0U, cnt);
    auto t2T = MakeUbTensor1D(t2Ub, 0U, cnt);
    auto t3T = MakeUbTensor1D(t3Ub, 0U, cnt);
    auto t4T = MakeUbTensor1D(t4Ub, 0U, cnt);

    Transform<te::Inst::Add>(t1T, t1T, t2T);
    Transform<te::Inst::Sub>(t3T, t3T, t4T);
    Transform<te::Inst::MulScalar>(t1T, t1T, alpha);
    Transform<te::Inst::MulScalar>(t3T, t3T, alpha);
    PipeBarrier<PIPE_ALL>();
}

// V: DeInterleave C_old → t2(real), t4(imag); then Axpy or MulScalar into Cr/Ci
__aicore__ inline void ApplyBeta(CherkCtx& ctx, uint32_t rows, uint32_t cols, float beta, bool skipTemp)
{
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CHERK_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t ubCplxStride = 2 * ubTempStride;
    uint32_t cnt = ubTempStride * cols;
    uint32_t srcCnt = ubCplxStride * cols;

    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();
    LocalTensor<float> cInUb = ctx.cInBuf.Get<float>();

    // DeInterleave entire block: [re0,im0,re1,im1,...] → [re0,re1,...] and [im0,im1,...]
    DeInterleave(t2Ub, t4Ub, cInUb, static_cast<int32_t>(srcCnt));
    PipeBarrier<PIPE_ALL>();

    auto t1T = MakeUbTensor1D(t1Ub, 0U, cnt);
    auto t2T = MakeUbTensor1D(t2Ub, 0U, cnt);
    auto t3T = MakeUbTensor1D(t3Ub, 0U, cnt);
    auto t4T = MakeUbTensor1D(t4Ub, 0U, cnt);

    if (skipTemp) {
        Transform<te::Inst::MulScalar>(t1T, t2T, beta);
        Transform<te::Inst::MulScalar>(t3T, t4T, beta);
    } else {
        Transform<te::Inst::Axpy>(t1T, t2T, beta);
        Transform<te::Inst::Axpy>(t3T, t4T, beta);
    }
    PipeBarrier<PIPE_ALL>();
}

// V: zero diagonal imaginary part (at most cols elements, typically 1 per block)
__aicore__ inline void ZeroDiagonal(CherkCtx& ctx,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CHERK_ARCH35_ELEMENTS_PER_BLOCK);
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();

    for (uint32_t c = 0; c < cols; c++) {
        uint32_t absJ = jBase + c;
        if (absJ >= iBase && absJ < iBase + rows) {
            uint32_t diagIdx = absJ - iBase;
            uint32_t colOffset = c * ubTempStride;
            t3Ub.SetValue(colOffset + diagIdx, 0.0f);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename CopyOp>
__aicore__ inline void StoreUploColumn(
    LocalTensor<float>& cOutDst0, CopyOp& copyUB2GM,
    __gm__ float* cGmBase, uint64_t gmOff, uint64_t ubOff,
    uint32_t uploCnt)
{
    if (uploCnt == 0) {
        return;
    }
    uint32_t copyCount = uploCnt * 2;
    auto gmCol = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(cGmBase + gmOff),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(1), static_cast<uint64_t>(copyCount)));
    auto ubCol = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cOutDst0.GetPhyAddr() + ubOff * sizeof(float)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(1), static_cast<uint64_t>(copyCount)));
    te::Copy(copyUB2GM, gmCol, ubCol);
}

// LOWER diagonal column, step 1: copy the non-uplo (strictly-upper) segment of
// C_old from cInUb into cOutDst0 (UB-side, no GM traffic) so the final column
// image is built entirely in UB and the store below is single-shot. The
// previous shape stored the whole column and then re-stored the non-uplo
// segment -- two MTE3 bursts targeting the same GM range whose completion
// order is not guaranteed: the full-column burst can land after the restore
// and overwrite non-uplo elements with computed values (the nonuplo exact
// canary failures; mirrors the cher2k arch35 K3 fix).
__aicore__ inline void RestoreLowerDiagSegment(
    LocalTensor<float>& cOutUb, LocalTensor<float>& cInUb, uint64_t ubOff, uint32_t nonUploCnt)
{
    if (nonUploCnt == 0) {
        return;
    }
    const uint32_t restoreCount = nonUploCnt * 2;
    Copy(cOutUb[static_cast<uint32_t>(ubOff)], cInUb[static_cast<uint32_t>(ubOff)], restoreCount);
}

// LOWER diagonal column, step 2: emit the ONE full-column MTE3 store. The
// caller must have run RestoreLowerDiagSegment for this column and crossed a
// barrier, so the UB image is complete and settled.
template <typename CopyOp>
__aicore__ inline void StoreLowerDiagColumn(
    LocalTensor<float>& cOutUb, CopyOp& copyUB2GM, __gm__ float* cGmBase,
    uint64_t gmOff, uint64_t ubOff, uint32_t rows)
{
    const uint32_t fullCount = rows * 2;
    auto gmFull = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(cGmBase + gmOff),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(1), static_cast<uint64_t>(fullCount)));
    auto ubFull = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cOutUb.GetPhyAddr() + ubOff * sizeof(float)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(1), static_cast<uint64_t>(fullCount)));
    te::Copy(copyUB2GM, gmFull, ubFull);
}

// UB→GM: store non-diagonal block (fully in uplo triangle) via single 2D te::Copy.
// Symmetric to LoadCOld's 2D GM→UB copy but reversed direction.
template <typename CopyOp>
__aicore__ inline void StoreNonDiagBlock(
    LocalTensor<float>& cOutDst0, CopyOp& copyUB2GM,
    __gm__ float* cGmBase, uint32_t n, uint32_t ldc,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols,
    uint32_t ubCplxStride)
{
    auto gmT = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(cGmBase),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(n), static_cast<uint64_t>(ldc * 2)));
    auto gmBlock = gmT.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows * 2)));
    auto ubT = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cOutDst0.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(cols), static_cast<uint64_t>(ubCplxStride)));
    te::Copy(copyUB2GM, gmBlock, ubT);
}

// V→MTE3: Interleave Cr,Ci → cOutBuf; store uplo to GM
__aicore__ inline void InterleaveAndStore(CherkCtx& ctx,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    bool uploUpper = (ctx.tiling.uploMode == ACLBLAS_UPPER);
    uint32_t ldc = ctx.tiling.ldc;
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CHERK_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t ubCplxStride = 2 * ubTempStride;
    uint32_t cnt = ubTempStride * cols;

    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> cInUb = ctx.cInBuf.Get<float>();
    LocalTensor<float> cOutDst0 = ctx.cOutBuf.Get<float>();
    LocalTensor<float> cOutDst1 = ctx.cOutBuf.GetWithOffset<float>(cnt, cnt * sizeof(float));

    Interleave(cOutDst0, cOutDst1, t1Ub, t3Ub, static_cast<int32_t>(cnt));
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    auto copyUB2GM = te::MakeCopy(te::CopyUB2GM{});
    auto cGmBase = const_cast<__gm__ float*>(ctx.cGM.GetPhyAddr());

    // Non-diagonal block: every column is fully in the uplo triangle (uploCnt = rows),
    // so a single 2D strided te::Copy replaces cols per-column DMAs.
    bool fullyUplo = uploUpper
        ? (jBase >= iBase + rows)
        : (jBase + cols <= iBase);

    if (fullyUplo) {
        StoreNonDiagBlock(cOutDst0, copyUB2GM, cGmBase,
            ctx.tiling.n, ldc, iBase, jBase, rows, cols, ubCplxStride);
    } else {
        // Diagonal block: per-column store (uploCnt varies per column).
        // Two passes instead of one. The LOWER diagonal columns need a UB-side
        // restore of their strictly-upper segment before the store, and that
        // restore is a V-pipe op whose result the MTE3 store must observe.
        // Every column's restore targets a disjoint UB span, so all of them can
        // be issued first and observed with a SINGLE barrier, then all the
        // stores go out -- each column's UB image and GM write are identical to
        // the inline shape (mirrors the cher2k arch35 K3 two-pass fix).
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t absJ = jBase + c;
            uint64_t ubOff = static_cast<uint64_t>(c) * ubCplxStride;
            bool isDiagonal = (absJ >= iBase && absJ < iBase + rows);
            if (isDiagonal && !uploUpper) {
                RestoreLowerDiagSegment(cOutDst0, cInUb, ubOff, absJ - iBase);
            }
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t absJ = jBase + c;
            uint64_t gmOff = static_cast<uint64_t>(absJ) * ldc * 2 + iBase * 2;
            uint64_t ubOff = static_cast<uint64_t>(c) * ubCplxStride;
            bool isDiagonal = (absJ >= iBase && absJ < iBase + rows);

            if (!isDiagonal || uploUpper) {
                uint32_t uploCnt = UploCount(uploUpper, iBase, absJ, rows);
                StoreUploColumn(cOutDst0, copyUB2GM, cGmBase, gmOff, ubOff, uploCnt);
            } else {
                StoreLowerDiagColumn(cOutDst0, copyUB2GM, cGmBase,
                    gmOff, ubOff, rows);
            }
        }
    }

    // Cross-block sync: ensure this block's MTE3 store finishes before the next
    // ProcessBlock reuses the same UB buffers:
    //   MTE3_V  — cOutBuf: MTE3 read → next V (Interleave) write
    //   MTE3_MTE2 — cInBuf: MTE3 read (LOWER diag restore) → next MTE2 (LoadCOld) write
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

__aicore__ inline void ProcessBlock(CherkCtx& ctx,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    float alpha = ctx.tiling.alphaVal;
    float beta = ctx.tiling.betaVal;
    bool skipTemp = (ctx.tiling.isAlphaZero || ctx.tiling.isKZero);
    bool isBetaZero = ctx.tiling.isBetaZero;

    // alpha==0 or k==0 with beta==0: C = 0 in uplo, non-uplo restored from C_old.
    // GEMM results are unused; zero t1/t3 so InterleaveAndStore writes (0,0) to
    // the uplo triangle. C_old is still needed for LOWER diagonal block restore.
    if (skipTemp && isBetaZero) {
        uint32_t ubTempStride = RoundUp<uint32_t>(rows, CHERK_ARCH35_ELEMENTS_PER_BLOCK);
        uint32_t cnt = ubTempStride * cols;
        LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
        LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
        Duplicate(t1Ub, 0.0f, cnt);
        Duplicate(t3Ub, 0.0f, cnt);
        LoadCOld(ctx, iBase, jBase, rows, cols);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        InterleaveAndStore(ctx, iBase, jBase, rows, cols);
        return;
    }

    if (!skipTemp) {
        LoadGemmResults(ctx, iBase, jBase, rows, cols);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ComputeCrCi(ctx, rows, cols, alpha);
    }

    // Always load C_old: for beta scaling when !isBetaZero, for non-uplo restore when isBetaZero
    LoadCOld(ctx, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    if (!isBetaZero) {
        ApplyBeta(ctx, rows, cols, beta, skipTemp);
    }

    // Only zero diagonal imag when GEMM ran (t3[diag]-t4[diag] float error)
    if (!skipTemp) {
        ZeroDiagonal(ctx, iBase, jBase, rows, cols);
    }
    InterleaveAndStore(ctx, iBase, jBase, rows, cols);
}

__aicore__ inline void ProcessLoop(CherkCtx& ctx)
{
    bool uploUpper = (ctx.tiling.uploMode == ACLBLAS_UPPER);
    for (uint32_t iBase = ctx.rowStart; iBase < ctx.rowEnd; iBase += SCALE_BLOCK) {
        uint32_t rows = Min<uint32_t>(SCALE_BLOCK, ctx.rowEnd - iBase);
        uint32_t jStart = uploUpper ? iBase : 0;
        uint32_t jLimit = uploUpper ? ctx.tiling.n : Min<uint32_t>(iBase + rows, ctx.tiling.n);
        for (uint32_t jBase = jStart; jBase < jLimit; jBase += SCALE_BLOCK) {
            uint32_t cols = Min<uint32_t>(SCALE_BLOCK, jLimit - jBase);
            ProcessBlock(ctx, iBase, jBase, rows, cols);
        }
    }
}

extern "C" __global__ __aicore__ void cherk_combine_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4,
    GM_ADDR c, const CherkCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CherkCtx ctx;
    ctx.tiling = tiling;
    ctx.t1GM.SetGlobalBuffer((__gm__ float*)t1);
    ctx.t2GM.SetGlobalBuffer((__gm__ float*)t2);
    ctx.t3GM.SetGlobalBuffer((__gm__ float*)t3);
    ctx.t4GM.SetGlobalBuffer((__gm__ float*)t4);
    ctx.cGM.SetGlobalBuffer((__gm__ float*)c);
    pipe.InitBuffer(ctx.t1Buf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t2Buf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t3Buf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t4Buf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.cInBuf, SCALE_UB_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.cOutBuf, SCALE_UB_CPLX_FLOATS * sizeof(float));
    uint32_t blockIdx = GetBlockIdx();
    ctx.rowStart = blockIdx * tiling.rowsPerCore;
    ctx.rowEnd = Min<uint32_t>(ctx.rowStart + tiling.rowsPerCore, tiling.n);
    ProcessLoop(ctx);
}

void cherk_combine_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4,
    GM_ADDR c, const CherkCombineTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    cherk_combine_kernel<<<numBlocks, nullptr, stream>>>(
        t1, t2, t3, t4, c, tiling);
}
