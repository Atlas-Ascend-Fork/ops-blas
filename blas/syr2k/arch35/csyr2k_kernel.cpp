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
 * \brief CSYR2K auxiliary kernels for ascend950 (DAV_3510).
 *
 * Phase 0 (AIV SIMD): validate/prepare six FP16 high/residual plane pairs,
 *                     or split A/B into four compact FP32 planes for strict fallback.
 * Phase 1 (AIC):       dispatch low-K 3M, high-K full-LL 4M, or strict FP32 4M.
 * Phase 2 (AIV SIMD): symmetrize the GEMM results and update only the selected C triangle.
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/syrk_gemm_arch35.h"
#include "cann_ops_blas_common.h"
#include "csyr2k_kernel.h"
#include "csyr2k_strict_narrow_impl.h"
#include "tensor_api/tensor.h"

using namespace AscendC;
namespace te = AscendC::Te;

constexpr uint32_t CSYR2K_AIV_BLOCK = CSYR2K_ARCH35_AIV_BLOCK;
constexpr uint32_t CSYR2K_AIV_TILE_FLOATS = CSYR2K_AIV_BLOCK * CSYR2K_AIV_BLOCK;
constexpr uint32_t CSYR2K_AIV_CPLX_FLOATS = CSYR2K_AIV_TILE_FLOATS * 2;
constexpr uint32_t CSYR2K_PREP_ROWS = 1024;
constexpr uint32_t CSYR2K_REDUCE_GROUP_FLOAT_COUNT = 64;
constexpr uint32_t CSYR2K_REDUCE_OUTPUT_BYTES = 32;
constexpr uint32_t CSYR2K_PREP_COLS = 4;
constexpr uint32_t CSYR2K_PREP_TILE_FLOATS = CSYR2K_PREP_ROWS * CSYR2K_PREP_COLS;
constexpr uint32_t CSYR2K_PREP_CPLX_FLOATS = CSYR2K_PREP_TILE_FLOATS * 2;

#ifndef __NPU_HOST__
static constexpr Reg::CastTrait CSYR2K_CAST_B16_TO_B32 = {
    Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait CSYR2K_CAST_B32_TO_B16 = {
    Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
#endif

__simd_callee__ inline void Csyr2kSplitFp32ToHalfHiLoInplace(
    Reg::RegTensor<half>& highHalf, Reg::RegTensor<half>& lowHalf, Reg::RegTensor<float>& value, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> highFloat;
    Reg::Cast<half, float, CSYR2K_CAST_B32_TO_B16>(highHalf, value, mask);
    Reg::Cast<float, half, CSYR2K_CAST_B16_TO_B32>(highFloat, highHalf, mask);
    Reg::Sub(value, value, highFloat, mask);
    Reg::Cast<half, float, CSYR2K_CAST_B32_TO_B16>(lowHalf, value, mask);
}

__simd_vf__ inline void Csyr2kSplitHalfPlanesVf(
    __ubuf__ float* arAddr, __ubuf__ float* aiAddr, __ubuf__ float* brAddr, __ubuf__ float* biAddr,
    __ubuf__ half* arHighAddr, __ubuf__ half* arLowAddr, __ubuf__ half* adHighAddr, __ubuf__ half* adLowAddr,
    __ubuf__ half* asHighAddr, __ubuf__ half* asLowAddr, __ubuf__ half* brHighAddr, __ubuf__ half* brLowAddr,
    __ubuf__ half* biHighAddr, __ubuf__ half* biLowAddr, __ubuf__ half* bsHighAddr, __ubuf__ half* bsLowAddr,
    uint32_t count, uint16_t loopNum)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    Reg::RegTensor<float> ar, ai, br, bi, ad, as, bs;
    Reg::RegTensor<half> highHalf, lowHalf;
    Reg::MaskReg mask;
    uint32_t remaining = count;

    for (uint16_t i = 0; i < loopNum; ++i) {
        uint32_t offset = static_cast<uint32_t>(i) * VL;
        mask = Reg::UpdateMask<float>(remaining);

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(ar, arAddr + offset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(ai, aiAddr + offset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(br, brAddr + offset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(bi, biAddr + offset);
        Reg::Sub(ad, ai, ar, mask);
        Reg::Add(as, ar, ai, mask);
        Reg::Add(bs, br, bi, mask);

        Csyr2kSplitFp32ToHalfHiLoInplace(highHalf, lowHalf, ar, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(arHighAddr + offset, highHalf, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(arLowAddr + offset, lowHalf, mask);
        Csyr2kSplitFp32ToHalfHiLoInplace(highHalf, lowHalf, ad, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(adHighAddr + offset, highHalf, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(adLowAddr + offset, lowHalf, mask);
        Csyr2kSplitFp32ToHalfHiLoInplace(highHalf, lowHalf, as, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(asHighAddr + offset, highHalf, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(asLowAddr + offset, lowHalf, mask);

        Csyr2kSplitFp32ToHalfHiLoInplace(highHalf, lowHalf, br, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(brHighAddr + offset, highHalf, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(brLowAddr + offset, lowHalf, mask);
        Csyr2kSplitFp32ToHalfHiLoInplace(highHalf, lowHalf, bi, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(biHighAddr + offset, highHalf, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(biLowAddr + offset, lowHalf, mask);
        Csyr2kSplitFp32ToHalfHiLoInplace(highHalf, lowHalf, bs, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(bsHighAddr + offset, highHalf, mask);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(bsLowAddr + offset, lowHalf, mask);
    }
}

__simd_vf__ inline void Csyr2kValidateMaxVf(
    __ubuf__ uint32_t* arAddr, __ubuf__ uint32_t* aiAddr, __ubuf__ uint32_t* brAddr, __ubuf__ uint32_t* biAddr,
    __ubuf__ uint32_t* maximumAddr, uint32_t count, uint16_t loopNum)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(uint32_t);
    Reg::RegTensor<uint32_t> x0, x1, x2, x3, signMask;
    Reg::MaskReg mask;
    Reg::Duplicate<uint32_t, uint32_t>(signMask, 0x7FFFFFFFU);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < loopNum; ++i) {
        uint32_t offset = static_cast<uint32_t>(i) * VL;
        mask = Reg::UpdateMask<uint32_t>(remaining);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(x0, arAddr + offset);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(x1, aiAddr + offset);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(x2, brAddr + offset);
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(x3, biAddr + offset);
        Reg::And<uint32_t, Reg::MaskMergeMode::ZEROING>(x0, x0, signMask, mask);
        Reg::And<uint32_t, Reg::MaskMergeMode::ZEROING>(x1, x1, signMask, mask);
        Reg::And<uint32_t, Reg::MaskMergeMode::ZEROING>(x2, x2, signMask, mask);
        Reg::And<uint32_t, Reg::MaskMergeMode::ZEROING>(x3, x3, signMask, mask);
        Reg::Max<uint32_t, Reg::MaskMergeMode::ZEROING>(x0, x0, x1, mask);
        Reg::Max<uint32_t, Reg::MaskMergeMode::ZEROING>(x2, x2, x3, mask);
        Reg::Max<uint32_t, Reg::MaskMergeMode::ZEROING>(x0, x0, x2, mask);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(maximumAddr + offset, x0, mask);
    }
}

// Fuse the exact Gauss-3M reconstruction with complex interleave for full
// macro-offdiagonal blocks.  The ordinary top-level vector sequence writes
// two real planes and then reads them again for Interleave; keeping both
// linear combinations in registers removes those four UB streams.
__simd_vf__ inline void Csyr2kExactCombineInterleaveVf(
    __ubuf__ float* t1Addr, __ubuf__ float* t2Addr, __ubuf__ float* t3Addr, __ubuf__ float* cOutAddr, uint16_t loopNum)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    Reg::RegTensor<float> p1, p2, p3;
    Reg::RegTensor<float> real, imag, out0, out1;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    for (uint16_t i = 0; i < loopNum; ++i) {
        uint32_t offset = static_cast<uint32_t>(i) * VL;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(p1, t1Addr + offset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(p2, t2Addr + offset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(p3, t3Addr + offset);
        Reg::Sub(real, p1, p2, mask);
        Reg::Add(imag, p1, p3, mask);
        Reg::Interleave<float>(out0, out1, real, imag);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(cOutAddr + offset * 2, out0, mask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(cOutAddr + offset * 2 + VL, out1, mask);
    }
}

__aicore__ inline bool Csyr2kExactFastEnabled(__gm__ uint32_t* flags, uint32_t flagCount)
{
    if (flagCount == 0) {
        return false;
    }
    for (uint32_t i = 0; i < flagCount; ++i) {
        if (AscendC::ReadGmByPassDCache(flags + i) != 0U) {
            return false;
        }
    }
    return true;
}

template <typename T>
__aicore__ inline auto MakeCsyr2kUbTensor1D(const LocalTensor<T>& tensor, uint32_t offset, uint32_t count)
{
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, T>(tensor.GetPhyAddr() + static_cast<uint64_t>(offset) * sizeof(T)),
        te::MakeFrameLayout<te::NDLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(count)));
}

// ============================================================================
// Phase 0: guarded FP16 prepare and vectorized FP32 A/B deinterleave.
// ============================================================================

struct Csyr2kDeinterleaveCtx {
    GlobalTensor<float> aGM;
    GlobalTensor<float> bGM;
    GlobalTensor<float> arGM;
    GlobalTensor<float> aiGM;
    GlobalTensor<float> brGM;
    GlobalTensor<float> biGM;
    TBuf<TPosition::VECIN> aInBuf;
    TBuf<TPosition::VECIN> bInBuf;
    TBuf<TPosition::VECIN> arOutBuf;
    TBuf<TPosition::VECIN> aiOutBuf;
    TBuf<TPosition::VECIN> brOutBuf;
    TBuf<TPosition::VECIN> biOutBuf;
    Csyr2kDeinterleaveTilingData tiling{};
};

template <uint32_t UB_ROWS, typename CopyOp>
__aicore__ inline void LoadCsyr2kComplexBlock(
    GlobalTensor<float>& srcGM, LocalTensor<float>& dstUb, CopyOp& copyGM2UB, uint32_t ld, uint32_t iBase,
    uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t matrixCols)
{
    auto gmTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(srcGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(matrixCols), static_cast<uint64_t>(ld * 2)));
    auto gmBlock = gmTensor.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows * 2)));
    auto ubTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(dstUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(UB_ROWS * 2)));
    te::Copy(copyGM2UB, ubTensor, gmBlock);
}

template <typename CopyOp>
__aicore__ inline void StoreCsyr2kRealBlock(
    GlobalTensor<float>& dstGM, LocalTensor<float>& srcUb, CopyOp& copyUB2GM, uint32_t matrixRows, uint32_t matrixCols,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    auto gmTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(dstGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(matrixCols), static_cast<uint64_t>(matrixRows)));
    auto ubTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(srcUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(CSYR2K_AIV_BLOCK)));
    te::Copy(
        copyUB2GM,
        gmTensor.Slice(
            te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase)),
            te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows))),
        ubTensor);
}

struct Csyr2kHalfPrepareCtx {
    GlobalTensor<float> aGM;
    GlobalTensor<float> bGM;
    GlobalTensor<half> arGM;
    GlobalTensor<half> arLowGM;
    GlobalTensor<half> adGM;
    GlobalTensor<half> adLowGM;
    GlobalTensor<half> asGM;
    GlobalTensor<half> asLowGM;
    GlobalTensor<half> brGM;
    GlobalTensor<half> brLowGM;
    GlobalTensor<half> biGM;
    GlobalTensor<half> biLowGM;
    GlobalTensor<half> bsGM;
    GlobalTensor<half> bsLowGM;
    TBuf<TPosition::VECIN> aInBuf;
    TBuf<TPosition::VECIN> bInBuf;
    TBuf<TPosition::VECIN> arFloatBuf;
    TBuf<TPosition::VECIN> aiFloatBuf;
    TBuf<TPosition::VECIN> brFloatBuf;
    TBuf<TPosition::VECIN> biFloatBuf;
    TBuf<TPosition::VECOUT> arHalfBuf;
    TBuf<TPosition::VECOUT> adHalfBuf;
    TBuf<TPosition::VECOUT> asHalfBuf;
    TBuf<TPosition::VECOUT> brHalfBuf;
    TBuf<TPosition::VECOUT> biHalfBuf;
    TBuf<TPosition::VECOUT> bsHalfBuf;
    TBuf<TPosition::VECOUT> arLowBuf;
    TBuf<TPosition::VECOUT> adLowBuf;
    TBuf<TPosition::VECOUT> asLowBuf;
    TBuf<TPosition::VECOUT> brLowBuf;
    TBuf<TPosition::VECOUT> biLowBuf;
    TBuf<TPosition::VECOUT> bsLowBuf;
    TBuf<TPosition::VECCALC> reduceWorkBuf;
    TBuf<TPosition::VECCALC> reduceOutBuf;
    Csyr2kFastPrepareTilingData tiling{};
};

template <typename CopyOp>
__aicore__ inline void StoreCsyr2kHalfBlock(
    GlobalTensor<half>& dstGM, LocalTensor<half>& srcUb, CopyOp& copyUB2GM, uint32_t matrixRows, uint32_t matrixCols,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    auto gmTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ half*>(dstGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP16_C0>>(
            static_cast<uint64_t>(matrixCols), static_cast<uint64_t>(matrixRows)));
    auto ubTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, half>(srcUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP16_C0>>(
            static_cast<uint64_t>(cols), static_cast<uint64_t>(CSYR2K_PREP_ROWS)));
    te::Copy(
        copyUB2GM,
        gmTensor.Slice(
            te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase)),
            te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows))),
        ubTensor);
}

__aicore__ inline bool Csyr2kValidateBoundedInputs(
    LocalTensor<float>& ar, LocalTensor<float>& ai, LocalTensor<float>& br, LocalTensor<float>& bi,
    LocalTensor<float>& violation, LocalTensor<float>& reduceWork, LocalTensor<float>& reduceOut, uint32_t count)
{
    // Abs clears the sign bit.  For non-negative FP32 encodings, unsigned
    // ordering matches numeric ordering; every NaN/Inf encoding is also larger
    // than +5.  Accumulate one maximum directly instead of materializing a
    // per-input "abs(x) - min(abs(x), 5)" violation vector four times.
    constexpr uint32_t validateVfWidth = VECTOR_REG_WIDTH / sizeof(uint32_t);
    uint16_t validateLoopNum = static_cast<uint16_t>((count + validateVfWidth - 1) / validateVfWidth);
    asc_vf_call<Csyr2kValidateMaxVf>(
        reinterpret_cast<__ubuf__ uint32_t*>(ar.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(ai.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(br.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(bi.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(violation.GetPhyAddr()), count, validateLoopNum);
    PipeBarrier<PIPE_V>();
    ReduceMax(
        reduceOut.ReinterpretCast<uint32_t>(), violation.ReinterpretCast<uint32_t>(),
        reduceWork.ReinterpretCast<uint32_t>(), static_cast<int32_t>(count), false);
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    return reduceOut.ReinterpretCast<uint32_t>().GetValue(0) <= 0x40A00000U;
}

struct Csyr2kHalfPrepareLocal {
    LocalTensor<float> aIn;
    LocalTensor<float> bIn;
    LocalTensor<float> ar;
    LocalTensor<float> ai;
    LocalTensor<float> br;
    LocalTensor<float> bi;
    LocalTensor<float> reduceWork;
    LocalTensor<float> reduceOut;
    LocalTensor<half> arHalf;
    LocalTensor<half> adHalf;
    LocalTensor<half> asHalf;
    LocalTensor<half> brHalf;
    LocalTensor<half> biHalf;
    LocalTensor<half> bsHalf;
    LocalTensor<half> arLow;
    LocalTensor<half> adLow;
    LocalTensor<half> asLow;
    LocalTensor<half> brLow;
    LocalTensor<half> biLow;
    LocalTensor<half> bsLow;
};

__aicore__ inline Csyr2kHalfPrepareLocal GetCsyr2kHalfPrepareLocal(Csyr2kHalfPrepareCtx& ctx)
{
    return {ctx.aInBuf.Get<float>(),        ctx.bInBuf.Get<float>(),       ctx.arFloatBuf.Get<float>(),
            ctx.aiFloatBuf.Get<float>(),    ctx.brFloatBuf.Get<float>(),   ctx.biFloatBuf.Get<float>(),
            ctx.reduceWorkBuf.Get<float>(), ctx.reduceOutBuf.Get<float>(), ctx.arHalfBuf.Get<half>(),
            ctx.adHalfBuf.Get<half>(),      ctx.asHalfBuf.Get<half>(),     ctx.brHalfBuf.Get<half>(),
            ctx.biHalfBuf.Get<half>(),      ctx.bsHalfBuf.Get<half>(),     ctx.arLowBuf.Get<half>(),
            ctx.adLowBuf.Get<half>(),       ctx.asLowBuf.Get<half>(),      ctx.brLowBuf.Get<half>(),
            ctx.biLowBuf.Get<half>(),       ctx.bsLowBuf.Get<half>()};
}

__aicore__ inline void LoadCsyr2kHalfPrepareCurrent(
    Csyr2kHalfPrepareCtx& ctx, Csyr2kHalfPrepareLocal& local, uint32_t iBase, uint32_t jBase, uint32_t rows,
    uint32_t cols, bool inputPrefetched, int32_t sourceCount)
{
    if (!inputPrefetched) {
        if (rows < CSYR2K_PREP_ROWS) {
            Duplicate(local.aIn, 0.0f, sourceCount);
            Duplicate(local.bIn, 0.0f, sourceCount);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        auto copy = te::MakeCopy(te::CopyGM2UB{});
        LoadCsyr2kComplexBlock<CSYR2K_PREP_ROWS>(
            ctx.aGM, local.aIn, copy, ctx.tiling.lda, iBase, jBase, rows, cols, ctx.tiling.cols);
        LoadCsyr2kComplexBlock<CSYR2K_PREP_ROWS>(
            ctx.bGM, local.bIn, copy, ctx.tiling.ldb, iBase, jBase, rows, cols, ctx.tiling.cols);
        SetFlag<HardEvent::MTE2_V>(0);
    }
    WaitFlag<HardEvent::MTE2_V>(0);
}

__aicore__ inline void PrefetchCsyr2kHalfPrepareNext(
    Csyr2kHalfPrepareCtx& ctx, Csyr2kHalfPrepareLocal& local, uint32_t nextTask, uint32_t rowTiles, uint32_t totalTiles)
{
    if (rowTiles == 0 || nextTask >= totalTiles) {
        return;
    }
    uint32_t nextRowTile = nextTask % rowTiles;
    uint32_t nextColTile = nextTask / rowTiles;
    uint32_t nextIBase = nextRowTile * CSYR2K_PREP_ROWS;
    uint32_t nextJBase = nextColTile * CSYR2K_PREP_COLS;
    uint32_t nextRows = Min<uint32_t>(CSYR2K_PREP_ROWS, ctx.tiling.rows - nextIBase);
    uint32_t nextCols = Min<uint32_t>(CSYR2K_PREP_COLS, ctx.tiling.cols - nextJBase);
    int32_t sourceCount = static_cast<int32_t>(CSYR2K_PREP_ROWS * nextCols * 2);
    if (nextRows < CSYR2K_PREP_ROWS) {
        Duplicate(local.aIn, 0.0f, sourceCount);
        Duplicate(local.bIn, 0.0f, sourceCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }
    auto copy = te::MakeCopy(te::CopyGM2UB{});
    LoadCsyr2kComplexBlock<CSYR2K_PREP_ROWS>(
        ctx.aGM, local.aIn, copy, ctx.tiling.lda, nextIBase, nextJBase, nextRows, nextCols, ctx.tiling.cols);
    LoadCsyr2kComplexBlock<CSYR2K_PREP_ROWS>(
        ctx.bGM, local.bIn, copy, ctx.tiling.ldb, nextIBase, nextJBase, nextRows, nextCols, ctx.tiling.cols);
    SetFlag<HardEvent::MTE2_V>(0);
}

__aicore__ inline void SplitAndStoreCsyr2kHalfPrepare(
    Csyr2kHalfPrepareCtx& ctx, Csyr2kHalfPrepareLocal& local, uint32_t iBase, uint32_t jBase, uint32_t rows,
    uint32_t cols, uint32_t count)
{
    constexpr uint32_t width = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t loops = static_cast<uint16_t>((count + width - 1) / width);
    asc_vf_call<Csyr2kSplitHalfPlanesVf>(
        reinterpret_cast<__ubuf__ float*>(local.ar.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(local.ai.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(local.br.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(local.bi.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.arHalf.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.arLow.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.adHalf.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.adLow.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.asHalf.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.asLow.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.brHalf.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.brLow.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.biHalf.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.biLow.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.bsHalf.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(local.bsLow.GetPhyAddr()), count, loops);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    auto copy = te::MakeCopy(te::CopyUB2GM{});
#define STORE_CSYR2K_HALF_BLOCK(GM, UB) \
    StoreCsyr2kHalfBlock(GM, UB, copy, ctx.tiling.rows, ctx.tiling.cols, iBase, jBase, rows, cols)
    STORE_CSYR2K_HALF_BLOCK(ctx.arGM, local.arHalf);
    STORE_CSYR2K_HALF_BLOCK(ctx.adGM, local.adHalf);
    STORE_CSYR2K_HALF_BLOCK(ctx.asGM, local.asHalf);
    STORE_CSYR2K_HALF_BLOCK(ctx.brGM, local.brHalf);
    STORE_CSYR2K_HALF_BLOCK(ctx.biGM, local.biHalf);
    STORE_CSYR2K_HALF_BLOCK(ctx.bsGM, local.bsHalf);
    STORE_CSYR2K_HALF_BLOCK(ctx.arLowGM, local.arLow);
    STORE_CSYR2K_HALF_BLOCK(ctx.adLowGM, local.adLow);
    STORE_CSYR2K_HALF_BLOCK(ctx.asLowGM, local.asLow);
    STORE_CSYR2K_HALF_BLOCK(ctx.brLowGM, local.brLow);
    STORE_CSYR2K_HALF_BLOCK(ctx.biLowGM, local.biLow);
    STORE_CSYR2K_HALF_BLOCK(ctx.bsLowGM, local.bsLow);
#undef STORE_CSYR2K_HALF_BLOCK
    SetFlag<HardEvent::MTE3_V>(0);
}

__aicore__ inline bool ProcessCsyr2kHalfPrepareBlock(
    Csyr2kHalfPrepareCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, bool inputPrefetched,
    uint32_t nextTask, uint32_t rowTiles, uint32_t totalTiles)
{
    if (rowTiles == 0) {
        return false;
    }
    Csyr2kHalfPrepareLocal local = GetCsyr2kHalfPrepareLocal(ctx);
    uint32_t count = CSYR2K_PREP_ROWS * cols;
    int32_t sourceCount = static_cast<int32_t>(count * 2);
    LoadCsyr2kHalfPrepareCurrent(ctx, local, iBase, jBase, rows, cols, inputPrefetched, sourceCount);
    DeInterleave(local.ar, local.ai, local.aIn, sourceCount);
    DeInterleave(local.br, local.bi, local.bIn, sourceCount);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> violation = local.aIn;
    bool valid = Csyr2kValidateBoundedInputs(
        local.ar, local.ai, local.br, local.bi, violation, local.reduceWork, local.reduceOut, count);
    PrefetchCsyr2kHalfPrepareNext(ctx, local, nextTask, rowTiles, totalTiles);
    WaitFlag<HardEvent::MTE3_V>(0);
    SplitAndStoreCsyr2kHalfPrepare(ctx, local, iBase, jBase, rows, cols, count);
    return valid;
}

__aicore__ inline bool ProcessCsyr2kHalfPrepare(Csyr2kHalfPrepareCtx& ctx)
{
    uint32_t rowTiles = (ctx.tiling.rows + CSYR2K_PREP_ROWS - 1) / CSYR2K_PREP_ROWS;
    uint32_t colTiles = (ctx.tiling.cols + CSYR2K_PREP_COLS - 1) / CSYR2K_PREP_COLS;
    uint32_t totalTiles = rowTiles * colTiles;
    uint32_t blockNum = GetBlockNum();
    bool valid = true;
    bool inputPrefetched = false;
    for (uint32_t task = GetBlockIdx(); task < totalTiles; task += blockNum) {
        uint32_t rowTile = task % rowTiles;
        uint32_t colTile = task / rowTiles;
        uint32_t iBase = rowTile * CSYR2K_PREP_ROWS;
        uint32_t jBase = colTile * CSYR2K_PREP_COLS;
        uint32_t rows = Min<uint32_t>(CSYR2K_PREP_ROWS, ctx.tiling.rows - iBase);
        uint32_t cols = Min<uint32_t>(CSYR2K_PREP_COLS, ctx.tiling.cols - jBase);
        uint32_t nextTask = task + blockNum;
        valid = ProcessCsyr2kHalfPrepareBlock(
                    ctx, iBase, jBase, rows, cols, inputPrefetched, nextTask, rowTiles, totalTiles) &&
                valid;
        inputPrefetched = nextTask < totalTiles;
    }
    return valid;
}

__aicore__ inline void InitCsyr2kHalfPrepareGlobal(
    Csyr2kHalfPrepareCtx& ctx, GM_ADDR a, GM_ADDR b, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow,
    GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf,
    GM_ADDR bsLow)
{
    ctx.aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    ctx.bGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    ctx.arGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(arHalf));
    ctx.arLowGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(arLow));
    ctx.adGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(adHalf));
    ctx.adLowGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(adLow));
    ctx.asGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(asHalf));
    ctx.asLowGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(asLow));
    ctx.brGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(brHalf));
    ctx.brLowGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(brLow));
    ctx.biGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(biHalf));
    ctx.biLowGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(biLow));
    ctx.bsGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bsHalf));
    ctx.bsLowGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(bsLow));
}

__aicore__ inline void InitCsyr2kHalfPrepareBuffers(TPipe& pipe, Csyr2kHalfPrepareCtx& ctx)
{
    pipe.InitBuffer(ctx.aInBuf, CSYR2K_PREP_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.bInBuf, CSYR2K_PREP_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.arFloatBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.aiFloatBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.brFloatBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.biFloatBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.arHalfBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.adHalfBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.asHalfBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.brHalfBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.biHalfBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.bsHalfBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.arLowBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.adLowBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.asLowBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.brLowBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.biLowBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    pipe.InitBuffer(ctx.bsLowBuf, CSYR2K_PREP_TILE_FLOATS * sizeof(half));
    constexpr uint32_t reduceWorkBytes =
        ((CSYR2K_PREP_TILE_FLOATS + CSYR2K_REDUCE_GROUP_FLOAT_COUNT - 1U) /
         CSYR2K_REDUCE_GROUP_FLOAT_COUNT) *
            sizeof(float) +
        CSYR2K_REDUCE_OUTPUT_BYTES;
    pipe.InitBuffer(ctx.reduceWorkBuf, reduceWorkBytes);
    pipe.InitBuffer(ctx.reduceOutBuf, CSYR2K_REDUCE_OUTPUT_BYTES);
}

__aicore__ inline void PublishCsyr2kHalfPrepareResult(GM_ADDR alpha, GM_ADDR fastFlags, bool valid)
{
    if (GetBlockIdx() == 0) {
        __gm__ const float* alphaValue = reinterpret_cast<__gm__ const float*>(alpha);
        if (alphaValue[0] != 1.0f || alphaValue[1] != 0.0f) {
            valid = false;
        }
    }
    if (!valid) {
        auto* flag = reinterpret_cast<__gm__ uint32_t*>(fastFlags);
        (void)AscendC::AtomicAdd<uint32_t>(flag, 1U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
}

extern "C" __global__ __aicore__ void csyr2k_half_prepare_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR alpha, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow, GM_ADDR asHalf,
    GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf, GM_ADDR bsLow,
    GM_ADDR fastFlags, const Csyr2kFastPrepareTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    Csyr2kHalfPrepareCtx ctx;
    ctx.tiling = tiling;
    InitCsyr2kHalfPrepareGlobal(
        ctx, a, b, arHalf, arLow, adHalf, adLow, asHalf, asLow, brHalf, brLow, biHalf, biLow, bsHalf, bsLow);
    InitCsyr2kHalfPrepareBuffers(pipe, ctx);
    SetFlag<HardEvent::MTE3_V>(0);
    bool valid = ProcessCsyr2kHalfPrepare(ctx);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
    PublishCsyr2kHalfPrepareResult(alpha, fastFlags, valid);
}

__aicore__ inline void ProcessCsyr2kDeinterleaveBlock(
    Csyr2kDeinterleaveCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    LocalTensor<float> aInUb = ctx.aInBuf.Get<float>();
    LocalTensor<float> bInUb = ctx.bInBuf.Get<float>();
    LocalTensor<float> arUb = ctx.arOutBuf.Get<float>();
    LocalTensor<float> aiUb = ctx.aiOutBuf.Get<float>();
    LocalTensor<float> brUb = ctx.brOutBuf.Get<float>();
    LocalTensor<float> biUb = ctx.biOutBuf.Get<float>();

    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    LoadCsyr2kComplexBlock<CSYR2K_AIV_BLOCK>(
        ctx.aGM, aInUb, copyGM2UB, ctx.tiling.lda, iBase, jBase, rows, cols, ctx.tiling.cols);
    LoadCsyr2kComplexBlock<CSYR2K_AIV_BLOCK>(
        ctx.bGM, bInUb, copyGM2UB, ctx.tiling.ldb, iBase, jBase, rows, cols, ctx.tiling.cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    int32_t sourceCount = static_cast<int32_t>(CSYR2K_AIV_BLOCK * cols * 2);
    DeInterleave(arUb, aiUb, aInUb, sourceCount);
    DeInterleave(brUb, biUb, bInUb, sourceCount);
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    auto copyUB2GM = te::MakeCopy(te::CopyUB2GM{});
    StoreCsyr2kRealBlock(ctx.arGM, arUb, copyUB2GM, ctx.tiling.rows, ctx.tiling.cols, iBase, jBase, rows, cols);
    StoreCsyr2kRealBlock(ctx.aiGM, aiUb, copyUB2GM, ctx.tiling.rows, ctx.tiling.cols, iBase, jBase, rows, cols);
    StoreCsyr2kRealBlock(ctx.brGM, brUb, copyUB2GM, ctx.tiling.rows, ctx.tiling.cols, iBase, jBase, rows, cols);
    StoreCsyr2kRealBlock(ctx.biGM, biUb, copyUB2GM, ctx.tiling.rows, ctx.tiling.cols, iBase, jBase, rows, cols);

    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
}

__aicore__ inline void ProcessCsyr2kDeinterleave(Csyr2kDeinterleaveCtx& ctx)
{
    uint32_t rowTiles = (ctx.tiling.rows + CSYR2K_AIV_BLOCK - 1) / CSYR2K_AIV_BLOCK;
    uint32_t colTiles = (ctx.tiling.cols + CSYR2K_AIV_BLOCK - 1) / CSYR2K_AIV_BLOCK;
    uint32_t totalTiles = rowTiles * colTiles;
    // In MIX 1:2 mode AIV GetBlockIdx spans 0..55 while GetBlockNum reports
    // the 28 launch groups.  Use the logical AIV count supplied by the host so
    // paired subblocks own disjoint strided tasks.
    uint32_t blockNum = ctx.tiling.blockCount == 0 ? GetBlockNum() : ctx.tiling.blockCount;
    for (uint32_t task = GetBlockIdx(); task < totalTiles; task += blockNum) {
        uint32_t rowTile = task % rowTiles;
        uint32_t colTile = task / rowTiles;
        uint32_t iBase = rowTile * CSYR2K_AIV_BLOCK;
        uint32_t jBase = colTile * CSYR2K_AIV_BLOCK;
        uint32_t rows = Min<uint32_t>(CSYR2K_AIV_BLOCK, ctx.tiling.rows - iBase);
        uint32_t cols = Min<uint32_t>(CSYR2K_AIV_BLOCK, ctx.tiling.cols - jBase);
        ProcessCsyr2kDeinterleaveBlock(ctx, iBase, jBase, rows, cols);
    }
}

__aicore__ inline void RunCsyr2kDeinterleave(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, const Csyr2kDeinterleaveTilingData& tiling,
    TPipe& pipe)
{
    Csyr2kDeinterleaveCtx ctx;
    ctx.tiling = tiling;
    ctx.aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    ctx.bGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    ctx.arGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ar));
    ctx.aiGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ai));
    ctx.brGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(br));
    ctx.biGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bi));
    pipe.InitBuffer(ctx.aInBuf, CSYR2K_AIV_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.bInBuf, CSYR2K_AIV_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.arOutBuf, CSYR2K_AIV_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.aiOutBuf, CSYR2K_AIV_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.brOutBuf, CSYR2K_AIV_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.biOutBuf, CSYR2K_AIV_TILE_FLOATS * sizeof(float));
    ProcessCsyr2kDeinterleave(ctx);
}

extern "C" __global__ __aicore__ void csyr2k_deinterleave_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR fastFlags,
    const Csyr2kDeinterleaveTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* flags = reinterpret_cast<__gm__ uint32_t*>(fastFlags);
    bool exactFast = Csyr2kExactFastEnabled(flags, tiling.fastFlagCount);
    if (exactFast) {
        return;
    }
    TPipe pipe;
    RunCsyr2kDeinterleave(a, b, ar, ai, br, bi, tiling, pipe);
}

constexpr int64_t CSYR2K_L1_SIZE = static_cast<int64_t>(CSYR2K_ARCH35_L1_SIZE_BYTES);

template <typename T, uint64_t C0>
__aicore__ inline auto MakeCsyr2kGmDnTensor(GM_ADDR address, uint64_t rows, uint64_t cols)
{
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ T*>(address)),
        te::MakeFrameLayout<te::DNExtLayoutPtn, AscendC::Std::Int<C0>>(rows, cols));
}

template <typename T, uint64_t C0>
__aicore__ inline auto MakeCsyr2kGmNdTensor(GM_ADDR address, uint64_t rows, uint64_t cols)
{
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ T*>(address)),
        te::MakeFrameLayout<te::NDExtLayoutPtn, AscendC::Std::Int<C0>>(rows, cols));
}

__aicore__ inline auto MakeCsyr2kTempTensor(GM_ADDR address, const Csyr2kGemmTilingData& tiling)
{
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(address)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.n, tiling.tempRowStride));
}

template <typename T, uint64_t C0>
__aicore__ inline void RunCsyr2kGemm(
    GM_ADDR left, GM_ADDR right, GM_ADDR temp, const Csyr2kGemmTilingData& tiling, uint32_t baseK)
{
    auto tempTensor = MakeCsyr2kTempTensor(temp, tiling);
    if (tiling.isTransN != 0) {
        auto leftTensor = MakeCsyr2kGmDnTensor<T, C0>(left, tiling.leftLd, tiling.k);
        auto rightTensor = MakeCsyr2kGmNdTensor<T, C0>(right, tiling.k, tiling.rightLd);
        SyrkGemmKernelImpl<decltype(leftTensor), decltype(rightTensor), decltype(tempTensor), Csyr2kGemmTilingData>(
            leftTensor, rightTensor, tempTensor, tiling, baseK, CSYR2K_ARCH35_L1_BUF_NUM, CSYR2K_L1_SIZE);
    } else {
        auto leftTensor = MakeCsyr2kGmNdTensor<T, C0>(left, tiling.n, tiling.leftLd);
        auto rightTensor = MakeCsyr2kGmDnTensor<T, C0>(right, tiling.rightLd, tiling.n);
        SyrkGemmKernelImpl<decltype(leftTensor), decltype(rightTensor), decltype(tempTensor), Csyr2kGemmTilingData>(
            leftTensor, rightTensor, tempTensor, tiling, baseK, CSYR2K_ARCH35_L1_BUF_NUM, CSYR2K_L1_SIZE);
    }
}

template <typename T, uint64_t C0, bool INCLUDE_LOW_LOW>
__aicore__ inline void RunCsyr2kResidualGemm(
    GM_ADDR leftHigh, GM_ADDR leftLow, GM_ADDR rightHigh, GM_ADDR rightLow, GM_ADDR temp,
    const Csyr2kGemmTilingData& tiling, uint32_t baseK)
{
    auto tempTensor = MakeCsyr2kTempTensor(temp, tiling);
    if (tiling.isTransN != 0) {
        auto leftHighTensor = MakeCsyr2kGmDnTensor<T, C0>(leftHigh, tiling.leftLd, tiling.k);
        auto leftLowTensor = MakeCsyr2kGmDnTensor<T, C0>(leftLow, tiling.leftLd, tiling.k);
        auto rightHighTensor = MakeCsyr2kGmNdTensor<T, C0>(rightHigh, tiling.k, tiling.rightLd);
        auto rightLowTensor = MakeCsyr2kGmNdTensor<T, C0>(rightLow, tiling.k, tiling.rightLd);
        SyrkGemmResidualKernelImpl<INCLUDE_LOW_LOW>(
            leftHighTensor, leftLowTensor, rightHighTensor, rightLowTensor, tempTensor, tiling, baseK,
            CSYR2K_ARCH35_L1_BUF_NUM, CSYR2K_L1_SIZE);
    } else {
        auto leftHighTensor = MakeCsyr2kGmNdTensor<T, C0>(leftHigh, tiling.n, tiling.leftLd);
        auto leftLowTensor = MakeCsyr2kGmNdTensor<T, C0>(leftLow, tiling.n, tiling.leftLd);
        auto rightHighTensor = MakeCsyr2kGmDnTensor<T, C0>(rightHigh, tiling.rightLd, tiling.n);
        auto rightLowTensor = MakeCsyr2kGmDnTensor<T, C0>(rightLow, tiling.rightLd, tiling.n);
        SyrkGemmResidualKernelImpl<INCLUDE_LOW_LOW>(
            leftHighTensor, leftLowTensor, rightHighTensor, rightLowTensor, tempTensor, tiling, baseK,
            CSYR2K_ARCH35_L1_BUF_NUM, CSYR2K_L1_SIZE);
    }
}

template <typename T, uint64_t C0, bool INCLUDE_LOW_LOW>
__aicore__ inline void RunCsyr2kResidualSymmetricTile(
    GM_ADDR leftHigh, GM_ADDR leftLow, GM_ADDR rightHigh, GM_ADDR rightLow, GM_ADDR temp,
    const Csyr2kGemmTilingData& tiling, uint32_t mOff, uint32_t nOff, uint32_t curTileM, uint32_t curTileN,
    bool addReverse, uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t& l0cPingPong, uint32_t baseK)
{
    auto tempTensor = MakeCsyr2kTempTensor(temp, tiling);
    if (tiling.isTransN != 0) {
        auto leftHighTensor = MakeCsyr2kGmDnTensor<T, C0>(leftHigh, tiling.leftLd, tiling.k);
        auto leftLowTensor = MakeCsyr2kGmDnTensor<T, C0>(leftLow, tiling.leftLd, tiling.k);
        auto rightHighTensor = MakeCsyr2kGmNdTensor<T, C0>(rightHigh, tiling.k, tiling.rightLd);
        auto rightLowTensor = MakeCsyr2kGmNdTensor<T, C0>(rightLow, tiling.k, tiling.rightLd);
        auto reverseLeftHighTensor = MakeCsyr2kGmDnTensor<T, C0>(rightHigh, tiling.rightLd, tiling.k);
        auto reverseLeftLowTensor = MakeCsyr2kGmDnTensor<T, C0>(rightLow, tiling.rightLd, tiling.k);
        auto reverseRightHighTensor = MakeCsyr2kGmNdTensor<T, C0>(leftHigh, tiling.k, tiling.leftLd);
        auto reverseRightLowTensor = MakeCsyr2kGmNdTensor<T, C0>(leftLow, tiling.k, tiling.leftLd);
        SyrkGemmProcessResidualSymmetricTile<T, C0, INCLUDE_LOW_LOW>(
            leftHighTensor, leftLowTensor, rightHighTensor, rightLowTensor, reverseLeftHighTensor, reverseLeftLowTensor,
            reverseRightHighTensor, reverseRightLowTensor, tempTensor, tiling.k, CSYR2K_ARCH35_DIRECT_TRIANGLE_K_CHUNK,
            baseK, mOff, nOff, curTileM, curTileN, addReverse, abL1LoopCnt, l0PingPong, l0cPingPong,
            CSYR2K_ARCH35_L1_BUF_NUM, CSYR2K_L1_SIZE, CSYR2K_ARCH35_DIRECT_TRIANGLE_L0C_BUF_NUM);
        return;
    }

    auto leftHighTensor = MakeCsyr2kGmNdTensor<T, C0>(leftHigh, tiling.n, tiling.leftLd);
    auto leftLowTensor = MakeCsyr2kGmNdTensor<T, C0>(leftLow, tiling.n, tiling.leftLd);
    auto rightHighTensor = MakeCsyr2kGmDnTensor<T, C0>(rightHigh, tiling.rightLd, tiling.n);
    auto rightLowTensor = MakeCsyr2kGmDnTensor<T, C0>(rightLow, tiling.rightLd, tiling.n);
    auto reverseLeftHighTensor = MakeCsyr2kGmNdTensor<T, C0>(rightHigh, tiling.n, tiling.rightLd);
    auto reverseLeftLowTensor = MakeCsyr2kGmNdTensor<T, C0>(rightLow, tiling.n, tiling.rightLd);
    auto reverseRightHighTensor = MakeCsyr2kGmDnTensor<T, C0>(leftHigh, tiling.leftLd, tiling.n);
    auto reverseRightLowTensor = MakeCsyr2kGmDnTensor<T, C0>(leftLow, tiling.leftLd, tiling.n);
    SyrkGemmProcessResidualSymmetricTile<T, C0, INCLUDE_LOW_LOW>(
        leftHighTensor, leftLowTensor, rightHighTensor, rightLowTensor, reverseLeftHighTensor, reverseLeftLowTensor,
        reverseRightHighTensor, reverseRightLowTensor, tempTensor, tiling.k, CSYR2K_ARCH35_DIRECT_TRIANGLE_K_CHUNK,
        baseK, mOff, nOff, curTileM, curTileN, addReverse, abL1LoopCnt, l0PingPong, l0cPingPong,
        CSYR2K_ARCH35_L1_BUF_NUM, CSYR2K_L1_SIZE, CSYR2K_ARCH35_DIRECT_TRIANGLE_L0C_BUF_NUM);
}

__aicore__ inline void DecodeCsyr2kOffDiagonalTile(
    uint32_t offDiagonalTask, uint32_t tileCount, bool upper, uint32_t& rowTile, uint32_t& colTile)
{
    uint32_t remaining = offDiagonalTask;
    for (rowTile = 0; rowTile < tileCount; ++rowTile) {
        uint32_t rowTasks = upper ? (tileCount - rowTile - 1) : rowTile;
        if (remaining < rowTasks) {
            colTile = upper ? (rowTile + remaining + 1) : remaining;
            return;
        }
        remaining -= rowTasks;
    }
    rowTile = 0;
    colTile = 0;
}

__aicore__ inline void DecodeCsyr2kTriangleTile(
    uint32_t triangleTask, uint32_t tileCount, bool upper, uint32_t& rowTile, uint32_t& colTile)
{
    uint32_t remaining = triangleTask;
    for (rowTile = 0; rowTile < tileCount; ++rowTile) {
        uint32_t rowTasks = upper ? (tileCount - rowTile) : (rowTile + 1);
        if (remaining < rowTasks) {
            colTile = upper ? (rowTile + remaining) : remaining;
            return;
        }
        remaining -= rowTasks;
    }
    rowTile = 0;
    colTile = 0;
}

struct Csyr2kDirectTriangleCtx {
    GM_ADDR arHalf;
    GM_ADDR arLow;
    GM_ADDR adHalf;
    GM_ADDR adLow;
    GM_ADDR asHalf;
    GM_ADDR asLow;
    GM_ADDR brHalf;
    GM_ADDR brLow;
    GM_ADDR biHalf;
    GM_ADDR biLow;
    GM_ADDR bsHalf;
    GM_ADDR bsLow;
    GM_ADDR t1;
    GM_ADDR t2;
    GM_ADDR t3;
    GM_ADDR t4;
    const Csyr2kGemmTilingData* tiling;
    uint32_t triangleTile;
    uint64_t abL1LoopCnt = 0;
    uint64_t l0PingPong = 0;
    uint64_t l0cPingPong = 0;
};

template <bool INCLUDE_LOW_LOW>
__aicore__ inline void RunCsyr2kResidualDirectTriangleTask(
    Csyr2kDirectTriangleCtx& ctx, uint32_t rowTile, uint32_t colTile, uint32_t channel)
{
    const Csyr2kGemmTilingData& tiling = *ctx.tiling;
    uint32_t mOff = colTile * ctx.triangleTile;
    uint32_t nOff = rowTile * ctx.triangleTile;
    uint32_t curTileM = Min<uint32_t>(ctx.triangleTile, tiling.n - mOff);
    uint32_t curTileN = Min<uint32_t>(ctx.triangleTile, tiling.n - nOff);
    bool addReverse = rowTile != colTile;
    if constexpr (INCLUDE_LOW_LOW) {
        if (channel == 0) {
            RunCsyr2kResidualSymmetricTile<half, SYRK_ARCH35_FP16_C0, true>(
                ctx.arHalf, ctx.arLow, ctx.brHalf, ctx.brLow, ctx.t1, tiling, mOff, nOff, curTileM, curTileN,
                addReverse, ctx.abL1LoopCnt, ctx.l0PingPong, ctx.l0cPingPong, CSYR2K_ARCH35_DIRECT_TRIANGLE_BASE_K);
        } else if (channel == 1) {
            RunCsyr2kResidualSymmetricTile<half, SYRK_ARCH35_FP16_C0, true>(
                ctx.adHalf, ctx.adLow, ctx.biHalf, ctx.biLow, ctx.t2, tiling, mOff, nOff, curTileM, curTileN,
                addReverse, ctx.abL1LoopCnt, ctx.l0PingPong, ctx.l0cPingPong, CSYR2K_ARCH35_DIRECT_TRIANGLE_BASE_K);
        } else if (channel == 2) {
            RunCsyr2kResidualSymmetricTile<half, SYRK_ARCH35_FP16_C0, true>(
                ctx.arHalf, ctx.arLow, ctx.biHalf, ctx.biLow, ctx.t3, tiling, mOff, nOff, curTileM, curTileN,
                addReverse, ctx.abL1LoopCnt, ctx.l0PingPong, ctx.l0cPingPong, CSYR2K_ARCH35_DIRECT_TRIANGLE_BASE_K);
        } else {
            RunCsyr2kResidualSymmetricTile<half, SYRK_ARCH35_FP16_C0, true>(
                ctx.adHalf, ctx.adLow, ctx.brHalf, ctx.brLow, ctx.t4, tiling, mOff, nOff, curTileM, curTileN,
                addReverse, ctx.abL1LoopCnt, ctx.l0PingPong, ctx.l0cPingPong, CSYR2K_ARCH35_DIRECT_TRIANGLE_BASE_K);
        }
    } else {
        if (channel == 0) {
            RunCsyr2kResidualSymmetricTile<half, SYRK_ARCH35_FP16_C0, false>(
                ctx.arHalf, ctx.arLow, ctx.bsHalf, ctx.bsLow, ctx.t1, tiling, mOff, nOff, curTileM, curTileN,
                addReverse, ctx.abL1LoopCnt, ctx.l0PingPong, ctx.l0cPingPong, CSYR2K_ARCH35_DIRECT_TRIANGLE_BASE_K);
        } else if (channel == 1) {
            RunCsyr2kResidualSymmetricTile<half, SYRK_ARCH35_FP16_C0, false>(
                ctx.asHalf, ctx.asLow, ctx.biHalf, ctx.biLow, ctx.t2, tiling, mOff, nOff, curTileM, curTileN,
                addReverse, ctx.abL1LoopCnt, ctx.l0PingPong, ctx.l0cPingPong, CSYR2K_ARCH35_DIRECT_TRIANGLE_BASE_K);
        } else {
            RunCsyr2kResidualSymmetricTile<half, SYRK_ARCH35_FP16_C0, false>(
                ctx.adHalf, ctx.adLow, ctx.brHalf, ctx.brLow, ctx.t3, tiling, mOff, nOff, curTileM, curTileN,
                addReverse, ctx.abL1LoopCnt, ctx.l0PingPong, ctx.l0cPingPong, CSYR2K_ARCH35_DIRECT_TRIANGLE_BASE_K);
        }
    }
}

__aicore__ inline void InitCsyr2kResidualEvents()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
}

__aicore__ inline void FinalizeCsyr2kResidualEvents()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
}

template <bool INCLUDE_LOW_LOW, bool INCLUDE_DIAGONAL>
__aicore__ inline void RunCsyr2kDirectTriangleRange(
    Csyr2kDirectTriangleCtx& ctx, uint32_t taskStart, uint32_t taskEnd, uint32_t taskStride, uint32_t tileCount,
    uint32_t channelCount, bool upper)
{
    for (uint32_t task = taskStart; task < taskEnd; task += taskStride) {
        uint32_t channel = task % channelCount;
        uint32_t triangleTask = task / channelCount;
        uint32_t rowTile = 0;
        uint32_t colTile = 0;
        if constexpr (INCLUDE_DIAGONAL) {
            DecodeCsyr2kTriangleTile(triangleTask, tileCount, upper, rowTile, colTile);
        } else {
            DecodeCsyr2kOffDiagonalTile(triangleTask, tileCount, upper, rowTile, colTile);
        }
        RunCsyr2kResidualDirectTriangleTask<INCLUDE_LOW_LOW>(ctx, rowTile, colTile, channel);
    }
}

template <bool INCLUDE_LOW_LOW>
__aicore__ inline void RunCsyr2kDirectTriangleDiagonal(
    Csyr2kDirectTriangleCtx& ctx, uint32_t tileCount, uint32_t channelCount, uint32_t offDiagonalTaskCount,
    uint32_t blockIdx, uint32_t blockNum)
{
    uint32_t lightCoreStart = offDiagonalTaskCount % blockNum;
    uint32_t lightCoreCount = blockNum - lightCoreStart;
    uint32_t lightCoreCapacity = lightCoreCount * 2;
    uint32_t diagonalTaskCount = tileCount * channelCount;
    for (uint32_t task = 0; task < diagonalTaskCount; ++task) {
        uint32_t assignedCore =
            task < lightCoreCapacity ? lightCoreStart + task % lightCoreCount : (task - lightCoreCapacity) % blockNum;
        if (assignedCore == blockIdx) {
            uint32_t diagonalTile = task / channelCount;
            RunCsyr2kResidualDirectTriangleTask<INCLUDE_LOW_LOW>(ctx, diagonalTile, diagonalTile, task % channelCount);
        }
    }
}

template <bool INCLUDE_LOW_LOW>
__aicore__ inline void RunCsyr2kResidualDirectTriangle(
    GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow, GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf,
    GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf, GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3,
    GM_ADDR t4, const Csyr2kGemmTilingData& tiling)
{
    uint32_t triangleTile = tiling.directTriangleTile;
    uint32_t tileCount = CeilDiv<uint32_t>(tiling.n, triangleTile);
    uint32_t offDiagonalTileCount = tileCount * (tileCount - 1) / 2;
    constexpr uint32_t channelCount = INCLUDE_LOW_LOW ? 4 : 3;
    uint32_t offDiagonalTaskCount = offDiagonalTileCount * channelCount;
    bool upper = tiling.uploMode == ACLBLAS_UPPER;
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum == 0) {
        return;
    }
    Csyr2kDirectTriangleCtx ctx{arHalf, arLow,  adHalf, adLow, asHalf, asLow, brHalf, brLow,   biHalf,
                                biLow,  bsHalf, bsLow,  t1,    t2,     t3,    t4,     &tiling, triangleTile};
    InitCsyr2kResidualEvents();
    if (tileCount < 8) {
        uint32_t triangleTaskCount = tileCount * (tileCount + 1) / 2 * channelCount;
        RunCsyr2kDirectTriangleRange<INCLUDE_LOW_LOW, true>(
            ctx, blockIdx, triangleTaskCount, blockNum, tileCount, channelCount, upper);
    } else {
        RunCsyr2kDirectTriangleRange<INCLUDE_LOW_LOW, false>(
            ctx, blockIdx, offDiagonalTaskCount, blockNum, tileCount, channelCount, upper);
        RunCsyr2kDirectTriangleDiagonal<INCLUDE_LOW_LOW>(
            ctx, tileCount, channelCount, offDiagonalTaskCount, blockIdx, blockNum);
    }
    FinalizeCsyr2kResidualEvents();
}

constexpr uint32_t CSYR2K_MIX_READY_FLAG = 12;
constexpr uint32_t CSYR2K_MIX_DONE_FLAG = 13;
constexpr uint32_t CSYR2K_MIX_SUBBLOCK_FLAG_OFFSET = 16;

// q8 has exactly 28 macro-offdiagonal tiles and 28 AICs.  Give every AIC all
// three Gauss channels of one macro so its paired AIVs can consume that macro
// while Cube works on the lighter diagonal tail.
__aicore__ inline void RunCsyr2kResidualDirectTriangleMixQ8(
    GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow, GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf,
    GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf, GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3,
    GM_ADDR t4, const Csyr2kGemmTilingData& tiling)
{
    constexpr uint32_t tileCount = 8;
    constexpr uint32_t diagonalTaskCount = tileCount * 3;
    bool upper = tiling.uploMode == ACLBLAS_UPPER;
    uint32_t blockIdx = AscendC::GetBlockIdx();
    Csyr2kDirectTriangleCtx ctx{arHalf, arLow, adHalf, adLow, asHalf,  asLow,
                                brHalf, brLow, biHalf, biLow, bsHalf,  bsLow,
                                t1,     t2,    t3,     t4,    &tiling, tiling.directTriangleTile};
    InitCsyr2kResidualEvents();
    uint32_t rowTile = 0;
    uint32_t colTile = 0;
    DecodeCsyr2kOffDiagonalTile(blockIdx, tileCount, upper, rowTile, colTile);
    for (uint32_t channel = 0; channel < 3; ++channel) {
        RunCsyr2kResidualDirectTriangleTask<false>(ctx, rowTile, colTile, channel);
    }

    // Cross-core mode 4 maps the two AIV subblocks into flag namespaces that
    // differ by sixteen, so publish one token for each paired consumer.
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(CSYR2K_MIX_READY_FLAG);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(CSYR2K_MIX_READY_FLAG + CSYR2K_MIX_SUBBLOCK_FLAG_OFFSET);

    // All 84 off-diagonal channel tasks are evenly assigned above.  The 24
    // diagonal channel tasks go one each to AIC 0..23, matching the balanced
    // generic schedule without changing any per-product accumulation order.
    if (blockIdx < diagonalTaskCount) {
        uint32_t diagonalTile = blockIdx / 3;
        RunCsyr2kResidualDirectTriangleTask<false>(ctx, diagonalTile, diagonalTile, blockIdx % 3);
    }
    FinalizeCsyr2kResidualEvents();
}

extern "C" __global__ __aicore__ void csyr2k_gemm_dispatch_kernel(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow,
    GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf,
    GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags, const Csyr2kGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    bool exactFast = Csyr2kExactFastEnabled(reinterpret_cast<__gm__ uint32_t*>(fastFlags), tiling.fastFlagCount);
    if (exactFast) {
        if (tiling.isResidual4M != 0) {
            RunCsyr2kResidualDirectTriangle<true>(
                arHalf, arLow, adHalf, adLow, asHalf, asLow, brHalf, brLow, biHalf, biLow, bsHalf, bsLow, t1, t2, t3,
                t4, tiling);
        } else {
            RunCsyr2kResidualDirectTriangle<false>(
                arHalf, arLow, adHalf, adLow, asHalf, asLow, brHalf, brLow, biHalf, biLow, bsHalf, bsLow, t1, t2, t3,
                t4, tiling);
        }
        return;
    }

    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    RunCsyr2kGemm<float, SYRK_ARCH35_FP32_C0>(ar, br, t1, tiling, CSYR2K_ARCH35_BASE_K);
    RunCsyr2kGemm<float, SYRK_ARCH35_FP32_C0>(ai, bi, t2, tiling, CSYR2K_ARCH35_BASE_K);
    RunCsyr2kGemm<float, SYRK_ARCH35_FP32_C0>(ar, bi, t3, tiling, CSYR2K_ARCH35_BASE_K);
    RunCsyr2kGemm<float, SYRK_ARCH35_FP32_C0>(ai, br, t4, tiling, CSYR2K_ARCH35_BASE_K);
    // Join the final Fixpipe GM stores before a stream-ordered partial
    // accumulator consumes (or the next segment reuses) the output planes.
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
}

__aicore__ inline void AccumulateCsyr2kPartialPlane(
    GlobalTensor<float>& finalGM, GlobalTensor<float>& partialGM, LocalTensor<float>& finalUb,
    LocalTensor<float>& partialUb, uint64_t offset, uint32_t count)
{
    DataCopy(finalUb, finalGM[offset], count);
    DataCopy(partialUb, partialGM[offset], count);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Add(finalUb, finalUb, partialUb, static_cast<int32_t>(count));
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(finalGM[offset], finalUb, count);
    // The same UB pair is reused by the next plane/tile.
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

extern "C" __global__ __aicore__ void csyr2k_partial_accumulate_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR t5, GM_ADDR t6, GM_ADDR t7, GM_ADDR t8,
    const Csyr2kPartialAccumulateTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> final1GM;
    GlobalTensor<float> final2GM;
    GlobalTensor<float> final3GM;
    GlobalTensor<float> final4GM;
    GlobalTensor<float> partial1GM;
    GlobalTensor<float> partial2GM;
    GlobalTensor<float> partial3GM;
    GlobalTensor<float> partial4GM;
    final1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t1), tiling.totalElements);
    final2GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t2), tiling.totalElements);
    final3GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t3), tiling.totalElements);
    final4GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t4), tiling.totalElements);
    partial1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t5), tiling.totalElements);
    partial2GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t6), tiling.totalElements);
    partial3GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t7), tiling.totalElements);
    partial4GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t8), tiling.totalElements);

    TPipe pipe;
    TBuf<TPosition::VECIN> finalBuf;
    TBuf<TPosition::VECIN> partialBuf;
    pipe.InitBuffer(finalBuf, CSYR2K_ARCH35_PARTIAL_ACCUM_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(partialBuf, CSYR2K_ARCH35_PARTIAL_ACCUM_TILE_FLOATS * sizeof(float));
    LocalTensor<float> finalUb = finalBuf.Get<float>();
    LocalTensor<float> partialUb = partialBuf.Get<float>();

    uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
    uint64_t tileSize = CSYR2K_ARCH35_PARTIAL_ACCUM_TILE_FLOATS;
    uint64_t tileStride = blockNum * tileSize;
    for (uint64_t offset = blockIdx * tileSize; offset < tiling.totalElements; offset += tileStride) {
        uint32_t count = static_cast<uint32_t>(Min<uint64_t>(tileSize, tiling.totalElements - offset));
        AccumulateCsyr2kPartialPlane(final1GM, partial1GM, finalUb, partialUb, offset, count);
        AccumulateCsyr2kPartialPlane(final2GM, partial2GM, finalUb, partialUb, offset, count);
        AccumulateCsyr2kPartialPlane(final3GM, partial3GM, finalUb, partialUb, offset, count);
        AccumulateCsyr2kPartialPlane(final4GM, partial4GM, finalUb, partialUb, offset, count);
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Phase 2: vectorized symmetry, complex alpha/beta, and triangular store.
// ============================================================================

template <bool UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Csyr2kCombineSmallSimt(
    uint32_t n, uint32_t ldc, uint32_t tempLdc, float alphaReal, float alphaImag, float betaReal, float betaImag,
    bool skipTemp, bool isBetaZero, bool exactFast, bool residual4M, __gm__ const float* t1, __gm__ const float* t2,
    __gm__ const float* t3, __gm__ const float* t4, __gm__ float* c)
{
    uint64_t total = 0;
    uint64_t linear = 0;
    uint64_t stride = 0;
    Csyr2kInitSimtRange(n, total, linear, stride);
    for (; linear < total; linear += stride) {
        uint32_t row = static_cast<uint32_t>(linear % n);
        uint32_t col = static_cast<uint32_t>(linear / n);
        if (Csyr2kOutsideTriangle<UPPER>(row, col)) {
            continue;
        }

        float outReal = 0.0f;
        float outImag = 0.0f;
        if (!skipTemp) {
            uint64_t direct = static_cast<uint64_t>(col) * tempLdc + row;
            uint64_t transpose = static_cast<uint64_t>(row) * tempLdc + col;
            float symT1 = t1[direct] + t1[transpose];
            float symT2 = t2[direct] + t2[transpose];
            float symT3 = t3[direct] + t3[transpose];
            float symReal = symT1 - symT2;
            float symImag = 0.0f;
            if (residual4M) {
                float symT4 = t4[direct] + t4[transpose];
                symReal -= symT3;
                symImag = (symT1 + symT4) + symT3;
            } else if (exactFast) {
                symImag = symT1 + symT3;
            } else {
                symImag = symT3 + (t4[direct] + t4[transpose]);
            }
            outReal = alphaReal * symReal - alphaImag * symImag;
            outImag = alphaReal * symImag + alphaImag * symReal;
        }

        uint64_t cOffset = (static_cast<uint64_t>(col) * ldc + row) * 2;
        if (!isBetaZero) {
            float cReal = c[cOffset];
            float cImag = c[cOffset + 1];
            outReal += betaReal * cReal - betaImag * cImag;
            outImag += betaReal * cImag + betaImag * cReal;
        }
        c[cOffset] = outReal;
        c[cOffset + 1] = outImag;
    }
}

struct Csyr2kCombineCtx {
    Csyr2kCombineTilingData tiling{};
    GlobalTensor<float> t1GM;
    GlobalTensor<float> t2GM;
    GlobalTensor<float> t3GM;
    GlobalTensor<float> t4GM;
    GlobalTensor<float> cGM;
    TBuf<TPosition::VECIN> t1Buf;
    TBuf<TPosition::VECIN> t2Buf;
    TBuf<TPosition::VECIN> t3Buf;
    TBuf<TPosition::VECIN> t4Buf;
    TBuf<TPosition::VECIN> cInBuf;
    TBuf<TPosition::VECOUT> cOutBuf;
    TBuf<TPosition::VECCALC> transposeIndexBuf;
    LocalTensor<uint32_t> transposeIndex;
};

__aicore__ inline void BuildCsyr2kTransposeIndex(Csyr2kCombineCtx& ctx)
{
    constexpr uint32_t baseGroup = 8;
    constexpr int32_t rowStrideBytes = static_cast<int32_t>(CSYR2K_AIV_BLOCK * sizeof(float));
    LocalTensor<int32_t> index = ctx.transposeIndexBuf.Get<int32_t>();
    for (uint32_t row = 0; row < baseGroup; ++row) {
        index.SetValue(row, static_cast<int32_t>(row) * rowStrideBytes);
    }
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
    for (uint32_t group = 1; group < CSYR2K_AIV_BLOCK / baseGroup; ++group) {
        Adds(
            index[group * baseGroup], index, static_cast<int32_t>(group * baseGroup) * rowStrideBytes,
            static_cast<int32_t>(baseGroup));
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t col = 1; col < CSYR2K_AIV_BLOCK; ++col) {
        Adds(
            index[col * CSYR2K_AIV_BLOCK], index, static_cast<int32_t>(col * sizeof(float)),
            static_cast<int32_t>(CSYR2K_AIV_BLOCK));
    }
    PipeBarrier<PIPE_V>();
    ctx.transposeIndex = ctx.transposeIndexBuf.Get<uint32_t>();
}

template <typename CopyOp>
__aicore__ inline void LoadCsyr2kTempDirect(
    GlobalTensor<float>& srcGM, LocalTensor<float>& dstUb, CopyOp& copyGM2UB, const Csyr2kCombineTilingData& tiling,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    auto gmTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(srcGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(tiling.n), static_cast<uint64_t>(tiling.tempLdc)));
    auto ubTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(dstUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(CSYR2K_AIV_BLOCK)));
    te::Copy(
        copyGM2UB, ubTensor,
        gmTensor.Slice(
            te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase)),
            te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows))));
}

template <typename CopyOp>
__aicore__ inline void LoadCsyr2kTempTransposeNatural(
    GlobalTensor<float>& srcGM, LocalTensor<float>& dstUb, CopyOp& copyGM2UB, const Csyr2kCombineTilingData& tiling,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    auto gmTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(srcGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(tiling.n), static_cast<uint64_t>(tiling.tempLdc)));
    auto ubTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(dstUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(rows), static_cast<uint64_t>(CSYR2K_AIV_BLOCK)));
    te::Copy(
        copyGM2UB, ubTensor,
        gmTensor.Slice(
            te::MakeCoord(static_cast<uint64_t>(iBase), static_cast<uint64_t>(jBase)),
            te::MakeShape(static_cast<uint64_t>(rows), static_cast<uint64_t>(cols))));
}

__aicore__ inline void LoadCsyr2kDirectResults(
    Csyr2kCombineCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();
    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    LoadCsyr2kTempDirect(ctx.t1GM, t1Ub, copyGM2UB, ctx.tiling, iBase, jBase, rows, cols);
    LoadCsyr2kTempDirect(ctx.t2GM, t2Ub, copyGM2UB, ctx.tiling, iBase, jBase, rows, cols);
    LoadCsyr2kTempDirect(ctx.t3GM, t3Ub, copyGM2UB, ctx.tiling, iBase, jBase, rows, cols);
    if (ctx.tiling.isExactFast == 0 || ctx.tiling.isResidual4M != 0) {
        LoadCsyr2kTempDirect(ctx.t4GM, t4Ub, copyGM2UB, ctx.tiling, iBase, jBase, rows, cols);
    }
}

__aicore__ inline bool Csyr2kIsDifferentDirectMacro(
    const Csyr2kCombineTilingData& tiling, uint32_t iBase, uint32_t jBase)
{
    if (tiling.isExactFast == 0 || tiling.directTriangleTile == 0) {
        return false;
    }
    uint32_t macroTile = tiling.directTriangleTile;
    uint32_t macroMask = macroTile - 1;
    return (macroTile & macroMask) == 0 ? (iBase & ~macroMask) != (jBase & ~macroMask) :
                                          iBase / macroTile != jBase / macroTile;
}

__aicore__ inline void CombineCsyr2kDirectPlanes(
    const Csyr2kCombineTilingData& tiling, LocalTensor<float>& t1, LocalTensor<float>& t2, LocalTensor<float>& t3,
    LocalTensor<float>& t4, uint32_t count)
{
    if (tiling.isResidual4M != 0) {
        Add(t4, t1, t4, static_cast<int32_t>(count));
        Sub(t1, t1, t2, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Sub(t1, t1, t3, static_cast<int32_t>(count));
        Add(t3, t3, t4, static_cast<int32_t>(count));
    } else if (tiling.isExactFast != 0) {
        Add(t3, t1, t3, static_cast<int32_t>(count));
        Sub(t1, t1, t2, static_cast<int32_t>(count));
    } else {
        Sub(t1, t1, t2, static_cast<int32_t>(count));
        Add(t3, t3, t4, static_cast<int32_t>(count));
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void AccumulateCsyr2kDiagonalTranspose(
    Csyr2kCombineCtx& ctx, LocalTensor<float>& t1, LocalTensor<float>& t2, LocalTensor<float>& t3,
    LocalTensor<float>& t4, uint32_t count)
{
    Gather(t2, t1, ctx.transposeIndex, 0U, count);
    Gather(t4, t3, ctx.transposeIndex, 0U, count);
    PipeBarrier<PIPE_V>();
    Add(t1, t1, t2, static_cast<int32_t>(count));
    Add(t3, t3, t4, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void AccumulateCsyr2kResidualTranspose(
    Csyr2kCombineCtx& ctx, LocalTensor<float>& t1, LocalTensor<float>& t2, LocalTensor<float>& t3,
    LocalTensor<float>& t4, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t count,
    uint32_t sourceCount)
{
    LocalTensor<float> scratch = ctx.cInBuf.Get<float>();
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    auto copy = te::MakeCopy(te::CopyGM2UB{});
    LoadCsyr2kTempTransposeNatural(ctx.t1GM, t2, copy, ctx.tiling, iBase, jBase, rows, cols);
    LoadCsyr2kTempTransposeNatural(ctx.t2GM, scratch, copy, ctx.tiling, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Muls(t4, t2, 1.0f, static_cast<int32_t>(sourceCount));
    Sub(t2, t2, scratch, static_cast<int32_t>(sourceCount));
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    LoadCsyr2kTempTransposeNatural(ctx.t3GM, scratch, copy, ctx.tiling, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Sub(t2, t2, scratch, static_cast<int32_t>(sourceCount));
    Add(t4, t4, scratch, static_cast<int32_t>(sourceCount));
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    LoadCsyr2kTempTransposeNatural(ctx.t4GM, scratch, copy, ctx.tiling, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Add(t4, t4, scratch, static_cast<int32_t>(sourceCount));
    PipeBarrier<PIPE_V>();
    Gather(scratch, t2, ctx.transposeIndex, 0U, count);
    PipeBarrier<PIPE_V>();
    Add(t1, t1, scratch, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    Gather(scratch, t4, ctx.transposeIndex, 0U, count);
    PipeBarrier<PIPE_V>();
    Add(t3, t3, scratch, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void AccumulateCsyr2kStandardTranspose(
    Csyr2kCombineCtx& ctx, LocalTensor<float>& t1, LocalTensor<float>& t2, LocalTensor<float>& t3,
    LocalTensor<float>& t4, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t count,
    uint32_t sourceCount)
{
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    auto copy = te::MakeCopy(te::CopyGM2UB{});
    LoadCsyr2kTempTransposeNatural(ctx.t1GM, t2, copy, ctx.tiling, iBase, jBase, rows, cols);
    LoadCsyr2kTempTransposeNatural(ctx.t2GM, t4, copy, ctx.tiling, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Sub(t2, t2, t4, static_cast<int32_t>(sourceCount));
    PipeBarrier<PIPE_V>();
    Gather(t4, t2, ctx.transposeIndex, 0U, count);
    PipeBarrier<PIPE_V>();
    Add(t1, t1, t4, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    LoadCsyr2kTempTransposeNatural(ctx.t3GM, t2, copy, ctx.tiling, iBase, jBase, rows, cols);
    GlobalTensor<float>& fourth = ctx.tiling.isExactFast != 0 ? ctx.t1GM : ctx.t4GM;
    LoadCsyr2kTempTransposeNatural(fourth, t4, copy, ctx.tiling, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Add(t2, t2, t4, static_cast<int32_t>(sourceCount));
    PipeBarrier<PIPE_V>();
    Gather(t4, t2, ctx.transposeIndex, 0U, count);
    PipeBarrier<PIPE_V>();
    Add(t3, t3, t4, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void AccumulateCsyr2kSymmetry(
    Csyr2kCombineCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    uint32_t count = CSYR2K_AIV_BLOCK * cols;
    uint32_t transposeSourceCount = CSYR2K_AIV_BLOCK * rows;
    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();

    CombineCsyr2kDirectPlanes(ctx.tiling, t1Ub, t2Ub, t3Ub, t4Ub, count);
    if (Csyr2kIsDifferentDirectMacro(ctx.tiling, iBase, jBase)) {
        return;
    }
    if (iBase == jBase) {
        AccumulateCsyr2kDiagonalTranspose(ctx, t1Ub, t2Ub, t3Ub, t4Ub, count);
        return;
    }
    if (ctx.tiling.isResidual4M != 0) {
        AccumulateCsyr2kResidualTranspose(
            ctx, t1Ub, t2Ub, t3Ub, t4Ub, iBase, jBase, rows, cols, count, transposeSourceCount);
        return;
    }
    AccumulateCsyr2kStandardTranspose(
        ctx, t1Ub, t2Ub, t3Ub, t4Ub, iBase, jBase, rows, cols, count, transposeSourceCount);
}

__aicore__ inline void LoadCsyr2kCOld(
    Csyr2kCombineCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    LocalTensor<float> cInUb = ctx.cInBuf.Get<float>();
    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    auto gmTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.cGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(ctx.tiling.n), static_cast<uint64_t>(ctx.tiling.ldc * 2)));
    auto ubTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cInUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(cols), static_cast<uint64_t>(CSYR2K_AIV_BLOCK * 2)));
    te::Copy(
        copyGM2UB, ubTensor,
        gmTensor.Slice(
            te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
            te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows * 2))));
}

__aicore__ inline void ApplyCsyr2kComplexAlpha(Csyr2kCombineCtx& ctx, uint32_t count)
{
    LocalTensor<float> real = ctx.t1Buf.Get<float>();
    LocalTensor<float> imag = ctx.t3Buf.Get<float>();
    LocalTensor<float> outReal = ctx.t2Buf.Get<float>();
    LocalTensor<float> outImag = ctx.t4Buf.Get<float>();

    Muls(outReal, real, ctx.tiling.alphaReal, static_cast<int32_t>(count));
    Muls(outImag, imag, ctx.tiling.alphaImag, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    Sub(outReal, outReal, outImag, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    Muls(outImag, imag, ctx.tiling.alphaReal, static_cast<int32_t>(count));
    Muls(imag, real, ctx.tiling.alphaImag, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    Add(outImag, outImag, imag, static_cast<int32_t>(count));
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void ApplyCsyr2kComplexBeta(
    LocalTensor<float> outReal, LocalTensor<float> outImag, LocalTensor<float> oldReal, LocalTensor<float> oldImag,
    float betaReal, float betaImag, uint32_t count)
{
    Axpy(outReal, oldReal, betaReal, static_cast<int32_t>(count));
    Axpy(outImag, oldImag, betaReal, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    Axpy(outReal, oldImag, -betaImag, static_cast<int32_t>(count));
    Axpy(outImag, oldReal, betaImag, static_cast<int32_t>(count));
    PipeBarrier<PIPE_ALL>();
}

template <typename CopyOp>
__aicore__ inline void StoreCsyr2kFullBlock(
    LocalTensor<float>& cOut, CopyOp& copyUB2GM, __gm__ float* cBase, const Csyr2kCombineTilingData& tiling,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    auto gmTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(cBase),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(tiling.n), static_cast<uint64_t>(tiling.ldc * 2)));
    auto ubTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cOut.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(cols), static_cast<uint64_t>(CSYR2K_AIV_BLOCK * 2)));
    te::Copy(
        copyUB2GM,
        gmTensor.Slice(
            te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
            te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows * 2))),
        ubTensor);
}

template <typename CopyOp>
__aicore__ inline void StoreCsyr2kColumnPrefix(
    LocalTensor<float>& src, CopyOp& copyUB2GM, __gm__ float* cBase, uint64_t gmOffset, uint64_t ubOffset,
    uint32_t complexCount)
{
    if (complexCount == 0) {
        return;
    }
    uint32_t floatCount = complexCount * 2;
    auto gmColumn = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(cBase + gmOffset),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(floatCount)));
    auto ubColumn = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(src.GetPhyAddr() + ubOffset * sizeof(float)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(floatCount)));
    te::Copy(copyUB2GM, gmColumn, ubColumn);
}

__aicore__ inline void ExactCombineInterleaveAndStoreCsyr2k(
    Csyr2kCombineCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    constexpr uint32_t vfWidth = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t count = CSYR2K_AIV_TILE_FLOATS;
    constexpr uint16_t loopNum = static_cast<uint16_t>(count / vfWidth);
    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> cOut = ctx.cOutBuf.Get<float>();

    // Consume the previous block's MTE3 before overwriting cOut, then expose
    // separate completion tokens for the three input planes and the output.
    WaitFlag<HardEvent::MTE3_V>(0);
    asc_vf_call<Csyr2kExactCombineInterleaveVf>(
        reinterpret_cast<__ubuf__ float*>(t1Ub.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(t2Ub.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(t3Ub.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(cOut.GetPhyAddr()),
        loopNum);
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    auto copyUB2GM = te::MakeCopy(te::CopyUB2GM{});
    auto cBase = const_cast<__gm__ float*>(ctx.cGM.GetPhyAddr());
    StoreCsyr2kFullBlock(cOut, copyUB2GM, cBase, ctx.tiling, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE3_V>(0);
}

__aicore__ inline void InterleaveAndStoreCsyr2k(
    Csyr2kCombineCtx& ctx, LocalTensor<float>& outReal, LocalTensor<float>& outImag, uint32_t iBase, uint32_t jBase,
    uint32_t rows, uint32_t cols)
{
    uint32_t count = CSYR2K_AIV_BLOCK * cols;
    LocalTensor<float> cOld = ctx.cInBuf.Get<float>();
    LocalTensor<float> cOut0 = ctx.cOutBuf.Get<float>();
    LocalTensor<float> cOut1 = ctx.cOutBuf.GetWithOffset<float>(count, count * sizeof(float));
    // Consume the previous block's store only when cOut is about to be reused.
    // The next block's GM2UB and vector arithmetic can overlap that pending MTE3.
    WaitFlag<HardEvent::MTE3_V>(0);
    Interleave(cOut0, cOut1, outReal, outImag, static_cast<int32_t>(count));
    // Interleave still reads t1/t3 (or the alpha/beta output aliases) while
    // the next block is free to start MTE2 loads into those same UB buffers.
    // Gate that reuse at V completion, independently of the cOut MTE3 store.
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    auto copyUB2GM = te::MakeCopy(te::CopyUB2GM{});
    auto cBase = const_cast<__gm__ float*>(ctx.cGM.GetPhyAddr());
    bool diagonal = (iBase == jBase);
    if (!diagonal) {
        StoreCsyr2kFullBlock(cOut0, copyUB2GM, cBase, ctx.tiling, iBase, jBase, rows, cols);
    } else {
        bool upper = (ctx.tiling.uploMode == ACLBLAS_UPPER);
        if (upper) {
            for (uint32_t col = 0; col < cols; ++col) {
                uint32_t absCol = jBase + col;
                uint64_t gmOffset = static_cast<uint64_t>(absCol) * ctx.tiling.ldc * 2 + iBase * 2;
                uint64_t ubOffset = static_cast<uint64_t>(col) * CSYR2K_AIV_BLOCK * 2;
                uint32_t uploCount = Min<uint32_t>(col + 1, rows);
                StoreCsyr2kColumnPrefix(cOut0, copyUB2GM, cBase, gmOffset, ubOffset, uploCount);
            }
        } else {
            // The lower suffix can start at an unaligned UB address.  Store all
            // full columns first, wait for those writes to complete, then
            // restore every non-selected prefix.  The phase boundary prevents
            // overlapping MTE3 writes from racing under optimized compilation.
            for (uint32_t col = 0; col < cols; ++col) {
                uint32_t absCol = jBase + col;
                uint64_t gmOffset = static_cast<uint64_t>(absCol) * ctx.tiling.ldc * 2 + iBase * 2;
                uint64_t ubOffset = static_cast<uint64_t>(col) * CSYR2K_AIV_BLOCK * 2;
                StoreCsyr2kColumnPrefix(cOut0, copyUB2GM, cBase, gmOffset, ubOffset, rows);
            }
            PipeBarrier<PIPE_MTE3>();
            for (uint32_t col = 0; col < cols; ++col) {
                uint32_t absCol = jBase + col;
                uint64_t gmOffset = static_cast<uint64_t>(absCol) * ctx.tiling.ldc * 2 + iBase * 2;
                uint64_t ubOffset = static_cast<uint64_t>(col) * CSYR2K_AIV_BLOCK * 2;
                uint32_t nonUploCount = Min<uint32_t>(col, rows);
                StoreCsyr2kColumnPrefix(cOld, copyUB2GM, cBase, gmOffset, ubOffset, nonUploCount);
            }
        }
    }

    SetFlag<HardEvent::MTE3_V>(0);
    if (ctx.tiling.uploMode == ACLBLAS_LOWER && diagonal) {
        // LOWER diagonal restores the non-selected prefix from cInBuf.  Only
        // that case needs to keep the next block's MTE2 from overwriting the
        // MTE3 source; all other stores source cOutBuf.
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}

__aicore__ inline bool ProcessCsyr2kSkipTemp(
    Csyr2kCombineCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t count)
{
    if (ctx.tiling.skipTemp == 0) {
        return false;
    }
    bool betaZero = ctx.tiling.isBetaZero != 0;
    bool needCOld = !betaZero || (ctx.tiling.uploMode == ACLBLAS_LOWER && iBase == jBase);
    LocalTensor<float> real = ctx.t1Buf.Get<float>();
    LocalTensor<float> imag = ctx.t3Buf.Get<float>();
    if (betaZero) {
        Duplicate(real, 0.0f, static_cast<int32_t>(count));
        Duplicate(imag, 0.0f, static_cast<int32_t>(count));
        PipeBarrier<PIPE_ALL>();
        if (needCOld) {
            LoadCsyr2kCOld(ctx, iBase, jBase, rows, cols);
            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);
        }
    } else {
        LoadCsyr2kCOld(ctx, iBase, jBase, rows, cols);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        LocalTensor<float> cIn = ctx.cInBuf.Get<float>();
        DeInterleave(real, imag, cIn, static_cast<int32_t>(count * 2));
        PipeBarrier<PIPE_ALL>();
        ApplyCsyr2kComplexBeta(
            ctx.t2Buf.Get<float>(), ctx.t4Buf.Get<float>(), real, imag, ctx.tiling.betaReal, ctx.tiling.betaImag,
            count);
        real = ctx.t2Buf.Get<float>();
        imag = ctx.t4Buf.Get<float>();
    }
    InterleaveAndStoreCsyr2k(ctx, real, imag, iBase, jBase, rows, cols);
    return true;
}

__aicore__ inline void AccumulateCsyr2kOutputBeta(
    Csyr2kCombineCtx& ctx, LocalTensor<float>& outReal, LocalTensor<float>& outImag, uint32_t iBase, uint32_t jBase,
    uint32_t rows, uint32_t cols, uint32_t count, bool alphaOne, bool needCOld)
{
    if (ctx.tiling.isBetaZero == 0) {
        LoadCsyr2kCOld(ctx, iBase, jBase, rows, cols);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        LocalTensor<float> betaReal = alphaOne ? ctx.t2Buf.Get<float>() : ctx.t1Buf.Get<float>();
        LocalTensor<float> betaImag = alphaOne ? ctx.t4Buf.Get<float>() : ctx.t3Buf.Get<float>();
        DeInterleave(betaReal, betaImag, ctx.cInBuf.Get<float>(), static_cast<int32_t>(count * 2));
        PipeBarrier<PIPE_ALL>();
        ApplyCsyr2kComplexBeta(outReal, outImag, betaReal, betaImag, ctx.tiling.betaReal, ctx.tiling.betaImag, count);
    } else if (needCOld) {
        LoadCsyr2kCOld(ctx, iBase, jBase, rows, cols);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
    }
}

__aicore__ inline void ProcessCsyr2kCombineBlock(
    Csyr2kCombineCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    uint32_t count = CSYR2K_AIV_BLOCK * cols;
    bool upper = (ctx.tiling.uploMode == ACLBLAS_UPPER);
    bool diagonal = (iBase == jBase);
    bool betaZero = (ctx.tiling.isBetaZero != 0);
    bool alphaOne = (ctx.tiling.alphaReal == 1.0f && ctx.tiling.alphaImag == 0.0f);
    bool needCOld = !betaZero || (!upper && diagonal);
    if (ProcessCsyr2kSkipTemp(ctx, iBase, jBase, rows, cols, count)) {
        return;
    }
    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();

    LoadCsyr2kDirectResults(ctx, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    // Official exact-fast performance blocks are full 64x64 tiles with
    // alpha=1 and beta=0.  A macro-offdiagonal Cube tile already contains both
    // SYR2K directions, so reconstruct and interleave directly in registers.
    if (alphaOne && betaZero && rows == CSYR2K_AIV_BLOCK && cols == CSYR2K_AIV_BLOCK && ctx.tiling.isResidual4M == 0 &&
        Csyr2kIsDifferentDirectMacro(ctx.tiling, iBase, jBase)) {
        ExactCombineInterleaveAndStoreCsyr2k(ctx, iBase, jBase, rows, cols);
        return;
    }

    AccumulateCsyr2kSymmetry(ctx, iBase, jBase, rows, cols);

    LocalTensor<float> outReal = t1Ub;
    LocalTensor<float> outImag = t3Ub;
    if (!alphaOne) {
        ApplyCsyr2kComplexAlpha(ctx, count);
        outReal = t2Ub;
        outImag = t4Ub;
    }

    AccumulateCsyr2kOutputBeta(ctx, outReal, outImag, iBase, jBase, rows, cols, count, alphaOne, needCOld);
    InterleaveAndStoreCsyr2k(ctx, outReal, outImag, iBase, jBase, rows, cols);
}

__aicore__ inline void ProcessCsyr2kCombineTile(Csyr2kCombineCtx& ctx, uint32_t rowTile, uint32_t colTile)
{
    uint32_t iBase = rowTile * CSYR2K_AIV_BLOCK;
    uint32_t jBase = colTile * CSYR2K_AIV_BLOCK;
    uint32_t rows = Min<uint32_t>(CSYR2K_AIV_BLOCK, ctx.tiling.n - iBase);
    uint32_t cols = Min<uint32_t>(CSYR2K_AIV_BLOCK, ctx.tiling.n - jBase);
    ProcessCsyr2kCombineBlock(ctx, iBase, jBase, rows, cols);
}

struct Csyr2kBalancedSchedule {
    uint32_t tileCount;
    uint32_t fullMacroCount;
    uint32_t tailSubTiles;
    uint32_t macroCount;
    uint32_t fullInternalCount;
    uint32_t internalOffDiagonalCount;
    uint32_t heavyCount;
    uint32_t cheapCount;
    uint32_t blockIdx;
    uint32_t blockNum;
    bool upper;
};

__aicore__ inline Csyr2kBalancedSchedule MakeCsyr2kBalancedSchedule(
    const Csyr2kCombineTilingData& tiling, uint32_t blockIdx, uint32_t blockNum)
{
    constexpr uint32_t subTilesPerMacro = CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE / CSYR2K_AIV_BLOCK;
    constexpr uint32_t internalOffDiagonalPerMacro = subTilesPerMacro * (subTilesPerMacro - 1) / 2;
    uint32_t tileCount = CeilDiv<uint32_t>(tiling.n, CSYR2K_AIV_BLOCK);
    uint32_t fullMacroCount = tileCount / subTilesPerMacro;
    uint32_t tailSubTiles = tileCount % subTilesPerMacro;
    uint32_t macroCount = fullMacroCount + (tailSubTiles != 0 ? 1U : 0U);
    uint32_t fullInternalCount = fullMacroCount * internalOffDiagonalPerMacro;
    uint32_t tailInternalCount = tailSubTiles > 1 ? tailSubTiles * (tailSubTiles - 1) / 2 : 0;
    uint32_t internalOffDiagonalCount = fullInternalCount + tailInternalCount;
    uint32_t diagonalCount = tileCount;
    uint32_t heavyCount = internalOffDiagonalCount + diagonalCount;
    uint32_t totalOffDiagonalCount = tileCount * (tileCount - 1) / 2;
    uint32_t cheapCount = totalOffDiagonalCount - internalOffDiagonalCount;
    return {
        tileCount,
        fullMacroCount,
        tailSubTiles,
        macroCount,
        fullInternalCount,
        internalOffDiagonalCount,
        heavyCount,
        cheapCount,
        blockIdx,
        blockNum,
        tiling.uploMode == ACLBLAS_UPPER};
}

__aicore__ inline void ProcessCsyr2kBalancedHeavy(Csyr2kCombineCtx& ctx, const Csyr2kBalancedSchedule& schedule)
{
    constexpr uint32_t subTilesPerMacro = CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE / CSYR2K_AIV_BLOCK;
    constexpr uint32_t internalOffDiagonalPerMacro = subTilesPerMacro * (subTilesPerMacro - 1) / 2;
    // Blocks inside a diagonal Cube macro tile require Gather/transpose work;
    // model them as weight two and spread them first.
    for (uint32_t task = schedule.blockIdx; task < schedule.heavyCount; task += schedule.blockNum) {
        uint32_t rowTile = 0;
        uint32_t colTile = 0;
        if (task < schedule.internalOffDiagonalCount) {
            uint32_t macro =
                task < schedule.fullInternalCount ? task / internalOffDiagonalPerMacro : schedule.fullMacroCount;
            uint32_t subTask = task < schedule.fullInternalCount ? task % internalOffDiagonalPerMacro :
                                                                   task - schedule.fullInternalCount;
            uint32_t macroSubTiles = macro < schedule.fullMacroCount ? subTilesPerMacro : schedule.tailSubTiles;
            DecodeCsyr2kOffDiagonalTile(subTask, macroSubTiles, schedule.upper, rowTile, colTile);
            rowTile += macro * subTilesPerMacro;
            colTile += macro * subTilesPerMacro;
        } else {
            uint32_t diagonalTask = task - schedule.internalOffDiagonalCount;
            rowTile = diagonalTask;
            colTile = rowTile;
        }
        ProcessCsyr2kCombineTile(ctx, rowTile, colTile);
    }
}

struct Csyr2kCheapRange {
    uint32_t start;
    uint32_t end;
};

__aicore__ inline Csyr2kCheapRange GetCsyr2kCheapRange(const Csyr2kBalancedSchedule& schedule)
{
    // Water-fill the weight-one macro-offdiagonal blocks around the heavy
    // assignments.  Each core gets a contiguous range, so no per-candidate
    // modulus or full-triangle scan remains.
    uint32_t totalWeight = schedule.heavyCount * 2 + schedule.cheapCount;
    uint32_t finalBaseLoad = totalWeight / schedule.blockNum;
    uint32_t finalLoadRemainder = totalWeight % schedule.blockNum;
    uint32_t heavyBaseCount = schedule.heavyCount / schedule.blockNum;
    uint32_t heavyRemainder = schedule.heavyCount % schedule.blockNum;
    uint32_t finalLoad = finalBaseLoad + (schedule.blockIdx < finalLoadRemainder ? 1U : 0U);
    uint32_t heavyLoad = 2 * (heavyBaseCount + (schedule.blockIdx < heavyRemainder ? 1U : 0U));
    uint32_t assignedCheapCount = finalLoad - heavyLoad;
    uint32_t cheapStart = schedule.blockIdx * finalBaseLoad + Min<uint32_t>(schedule.blockIdx, finalLoadRemainder) -
                          2 * (schedule.blockIdx * heavyBaseCount + Min<uint32_t>(schedule.blockIdx, heavyRemainder));
    return {cheapStart, Min<uint32_t>(cheapStart + assignedCheapCount, schedule.cheapCount)};
}

__aicore__ inline void ProcessCsyr2kCheapMacroPair(
    Csyr2kCombineCtx& ctx, const Csyr2kCheapRange& owned, uint32_t rowMacro, uint32_t colMacro, uint32_t pairStart,
    uint32_t pairEnd, uint32_t rowSubTiles)
{
    constexpr uint32_t subTilesPerMacro = CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE / CSYR2K_AIV_BLOCK;
    if (rowSubTiles == 0) {
        return;
    }
    uint32_t ownedStart = owned.start > pairStart ? owned.start : pairStart;
    uint32_t ownedEnd = Min<uint32_t>(owned.end, pairEnd);
    for (uint32_t task = ownedStart; task < ownedEnd; ++task) {
        uint32_t subTask = task - pairStart;
        uint32_t rowTile = rowMacro * subTilesPerMacro + subTask % rowSubTiles;
        uint32_t colTile = colMacro * subTilesPerMacro + subTask / rowSubTiles;
        ProcessCsyr2kCombineTile(ctx, rowTile, colTile);
    }
}

__aicore__ inline void ProcessCsyr2kBalancedCheap(Csyr2kCombineCtx& ctx, const Csyr2kBalancedSchedule& schedule)
{
    constexpr uint32_t subTilesPerMacro = CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE / CSYR2K_AIV_BLOCK;
    Csyr2kCheapRange owned = GetCsyr2kCheapRange(schedule);
    uint32_t pairStart = 0;
    // Enumerate only macro pairs (at most 136 for the supported range), then
    // process the overlap with this core's contiguous cheap-task interval.
    // This also handles a partial final 256x256 macro without sending any AIV
    // outside the logical triangle.
    for (uint32_t rowMacro = 0; rowMacro < schedule.macroCount; ++rowMacro) {
        uint32_t rowSubTiles = Min<uint32_t>(subTilesPerMacro, schedule.tileCount - rowMacro * subTilesPerMacro);
        for (uint32_t colMacro = 0; colMacro < schedule.macroCount; ++colMacro) {
            if ((schedule.upper && colMacro <= rowMacro) || (!schedule.upper && colMacro >= rowMacro)) {
                continue;
            }
            uint32_t colSubTiles = Min<uint32_t>(subTilesPerMacro, schedule.tileCount - colMacro * subTilesPerMacro);
            uint32_t pairTaskCount = rowSubTiles * colSubTiles;
            uint32_t pairEnd = pairStart + pairTaskCount;
            ProcessCsyr2kCheapMacroPair(ctx, owned, rowMacro, colMacro, pairStart, pairEnd, rowSubTiles);
            pairStart = pairEnd;
        }
    }
}

__aicore__ inline void ProcessCsyr2kCombineExactBalanced(Csyr2kCombineCtx& ctx)
{
    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0) {
        return;
    }
    Csyr2kBalancedSchedule schedule = MakeCsyr2kBalancedSchedule(ctx.tiling, GetBlockIdx(), blockNum);
    ProcessCsyr2kBalancedHeavy(ctx, schedule);
    if (ctx.tiling.mixOffdiagEnabled == 0) {
        ProcessCsyr2kBalancedCheap(ctx, schedule);
    }
}

__aicore__ inline void ProcessCsyr2kCombine(Csyr2kCombineCtx& ctx)
{
    if (ctx.tiling.isExactFast != 0 && ctx.tiling.directTriangleTile == CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE &&
        CeilDiv<uint32_t>(ctx.tiling.n, ctx.tiling.directTriangleTile) >= 3) {
        ProcessCsyr2kCombineExactBalanced(ctx);
        return;
    }

    uint32_t tileCount = (ctx.tiling.n + CSYR2K_AIV_BLOCK - 1) / CSYR2K_AIV_BLOCK;
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0) {
        return;
    }
    bool upper = (ctx.tiling.uploMode == ACLBLAS_UPPER);

    // Decode only the blocks owned by this core instead of making every AIV
    // scan the entire triangular grid and execute a modulus per candidate.
    uint32_t offDiagonalCount = tileCount * (tileCount - 1) / 2;
    for (uint32_t task = blockIdx; task < offDiagonalCount; task += blockNum) {
        uint32_t rowTile = 0;
        uint32_t colTile = 0;
        DecodeCsyr2kOffDiagonalTile(task, tileCount, upper, rowTile, colTile);
        ProcessCsyr2kCombineTile(ctx, rowTile, colTile);
    }

    // A diagonal block has extra Gather/triangular-store work.  Give each one
    // first to a core with one fewer off-diagonal block, then round-robin any
    // remainder.  No two cores ever write the same output block.
    uint32_t offDiagonalRemainder = offDiagonalCount % blockNum;
    uint32_t lightCoreStart = offDiagonalRemainder;
    uint32_t lightCoreCount = blockNum - lightCoreStart;
    for (uint32_t diagonalTile = 0; diagonalTile < tileCount; ++diagonalTile) {
        uint32_t assignedCore =
            diagonalTile < lightCoreCount ? lightCoreStart + diagonalTile : (diagonalTile - lightCoreCount) % blockNum;
        if (assignedCore != blockIdx) {
            continue;
        }
        uint32_t base = diagonalTile * CSYR2K_AIV_BLOCK;
        uint32_t size = Min<uint32_t>(CSYR2K_AIV_BLOCK, ctx.tiling.n - base);
        ProcessCsyr2kCombineBlock(ctx, base, base, size, size);
    }
}

template <bool UPPER>
__aicore__ inline void LaunchCsyr2kCombineSmall(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, const Csyr2kCombineTilingData& tiling,
    const Csyr2kScalars& scalars, bool skipTemp, bool exactFast, bool residual4M)
{
    asc_vf_call<Csyr2kCombineSmallSimt<UPPER>>(
        dim3{CSYR2K_SMALL_SIMT_THREADS, 1, 1}, tiling.n, tiling.ldc, tiling.tempLdc, scalars.alphaReal,
        scalars.alphaImag, scalars.betaReal, scalars.betaImag, skipTemp, scalars.betaZero, exactFast, residual4M,
        reinterpret_cast<__gm__ const float*>(t1), reinterpret_cast<__gm__ const float*>(t2),
        reinterpret_cast<__gm__ const float*>(t3), reinterpret_cast<__gm__ const float*>(t4),
        reinterpret_cast<__gm__ float*>(c));
}

__aicore__ inline void SetCsyr2kCombineTiling(
    Csyr2kCombineCtx& ctx, const Csyr2kCombineTilingData& tiling, const Csyr2kScalars& scalars, bool skipTemp,
    bool exactFast)
{
    ctx.tiling = tiling;
    ctx.tiling.alphaReal = scalars.alphaReal;
    ctx.tiling.alphaImag = scalars.alphaImag;
    ctx.tiling.betaReal = scalars.betaReal;
    ctx.tiling.betaImag = scalars.betaImag;
    ctx.tiling.skipTemp = static_cast<uint8_t>(skipTemp);
    ctx.tiling.isBetaZero = static_cast<uint8_t>(scalars.betaZero);
    ctx.tiling.isExactFast = static_cast<uint8_t>(exactFast);
    ctx.tiling.isResidual4M = static_cast<uint8_t>(exactFast && tiling.isResidual4M != 0);
    ctx.tiling.mixOffdiagEnabled = static_cast<uint8_t>(
        tiling.mixOffdiagEnabled != 0 && exactFast && scalars.betaZero && scalars.alphaReal == 1.0f &&
        scalars.alphaImag == 0.0f);
}

__aicore__ inline void SetCsyr2kCombineGlobal(
    Csyr2kCombineCtx& ctx, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c)
{
    ctx.t1GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t1));
    ctx.t2GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t2));
    ctx.t3GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t3));
    ctx.t4GM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t4));
    ctx.cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
}

__aicore__ inline void InitCsyr2kCombineBuffers(TPipe& pipe, Csyr2kCombineCtx& ctx)
{
    pipe.InitBuffer(ctx.t1Buf, CSYR2K_AIV_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t2Buf, CSYR2K_AIV_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t3Buf, CSYR2K_AIV_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t4Buf, CSYR2K_AIV_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.cInBuf, CSYR2K_AIV_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.cOutBuf, CSYR2K_AIV_CPLX_FLOATS * sizeof(float));
}

__aicore__ inline void RunCsyr2kCombineContext(TPipe& pipe, Csyr2kCombineCtx& ctx)
{
    pipe.InitBuffer(ctx.transposeIndexBuf, CSYR2K_AIV_TILE_FLOATS * sizeof(uint32_t));
    BuildCsyr2kTransposeIndex(ctx);
    // Seed one cross-iteration token.  Each block consumes it before reusing
    // cOut and produces a replacement after its GM store.
    SetFlag<HardEvent::MTE3_V>(0);
    ProcessCsyr2kCombine(ctx);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

extern "C" __global__ __aicore__ void csyr2k_combine_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags, GM_ADDR alpha, GM_ADDR beta, GM_ADDR c,
    const Csyr2kCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csyr2kScalars scalars = LoadCsyr2kScalars(alpha, beta);
    bool skipTemp = (tiling.skipTemp != 0 || scalars.alphaZero);
    bool exactFast =
        !skipTemp && Csyr2kExactFastEnabled(reinterpret_cast<__gm__ uint32_t*>(fastFlags), tiling.fastFlagCount);
    bool residual4M = exactFast && tiling.isResidual4M != 0;
    if (skipTemp && scalars.betaReal == 1.0f && scalars.betaImag == 0.0f) {
        return;
    }
    if (tiling.n <= CSYR2K_AIV_BLOCK) {
        if (tiling.uploMode == ACLBLAS_UPPER) {
            LaunchCsyr2kCombineSmall<true>(t1, t2, t3, t4, c, tiling, scalars, skipTemp, exactFast, residual4M);
        } else {
            LaunchCsyr2kCombineSmall<false>(t1, t2, t3, t4, c, tiling, scalars, skipTemp, exactFast, residual4M);
        }
        return;
    }
    TPipe pipe;
    Csyr2kCombineCtx ctx;
    SetCsyr2kCombineTiling(ctx, tiling, scalars, skipTemp, exactFast);
    SetCsyr2kCombineGlobal(ctx, t1, t2, t3, t4, c);
    InitCsyr2kCombineBuffers(pipe, ctx);
    RunCsyr2kCombineContext(pipe, ctx);
}

__aicore__ inline void RunCsyr2kMixDeinterleave(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR fastFlags,
    const Csyr2kGemmTilingData& gemmTiling, const Csyr2kDeinterleaveTilingData& deinterleaveTiling)
{
    // The q8 host omits the standalone deinterleave launch.  On a rejected
    // exact candidate the paired AIV materializes the strict FP32 planes and
    // contributes to the task-wide counter consumed by every AIC.
    TPipe pipe;
    RunCsyr2kDeinterleave(a, b, ar, ai, br, bi, deinterleaveTiling, pipe);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    auto* flags = reinterpret_cast<__gm__ uint32_t*>(fastFlags);
    (void)AscendC::AtomicAdd<uint32_t>(flags + gemmTiling.fastFlagCount, 1U);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

__aicore__ inline void RunCsyr2kMixExactCombine(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, const Csyr2kCombineTilingData& combineTiling,
    const Csyr2kScalars& scalars)
{
    TPipe pipe;
    Csyr2kCombineCtx ctx;
    SetCsyr2kCombineTiling(ctx, combineTiling, scalars, false, true);
    SetCsyr2kCombineGlobal(ctx, t1, t2, t3, t4, c);
    InitCsyr2kCombineBuffers(pipe, ctx);
    constexpr uint32_t subTilesPerMacro = CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE / CSYR2K_AIV_BLOCK;
    uint32_t aivIdx = AscendC::GetBlockIdx();
    uint32_t rowMacro = 0;
    uint32_t colMacro = 0;
    DecodeCsyr2kOffDiagonalTile(aivIdx >> 1, 8, combineTiling.uploMode == ACLBLAS_UPPER, rowMacro, colMacro);
    SetFlag<HardEvent::MTE3_V>(0);
    for (uint32_t subTask = aivIdx & 1; subTask < subTilesPerMacro * subTilesPerMacro; subTask += 2) {
        uint32_t rowTile = rowMacro * subTilesPerMacro + subTask % subTilesPerMacro;
        uint32_t colTile = colMacro * subTilesPerMacro + subTask / subTilesPerMacro;
        ProcessCsyr2kCombineTile(ctx, rowTile, colTile);
    }
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
}

__aicore__ inline void RunCsyr2kMixAiv(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3,
    GM_ADDR t4, GM_ADDR fastFlags, GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, const Csyr2kGemmTilingData& gemmTiling,
    const Csyr2kDeinterleaveTilingData& deinterleaveTiling, const Csyr2kCombineTilingData& combineTiling)
{
    Csyr2kScalars scalars = LoadCsyr2kScalars(alpha, beta);
    bool exactFast = Csyr2kExactFastEnabled(reinterpret_cast<__gm__ uint32_t*>(fastFlags), gemmTiling.fastFlagCount);
    if (!exactFast) {
        RunCsyr2kMixDeinterleave(a, b, ar, ai, br, bi, fastFlags, gemmTiling, deinterleaveTiling);
    }
    if (exactFast && scalars.betaZero) {
        // MTE2 consumes the result immediately, so no extra task barrier is needed.
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(CSYR2K_MIX_READY_FLAG);
        RunCsyr2kMixExactCombine(t1, t2, t3, t4, c, combineTiling, scalars);
    } else if (exactFast) {
        // Pair the exact beta!=0 READY counter before waiting for DONE.
        AscendC::CrossCoreWaitFlag<4, PIPE_S>(CSYR2K_MIX_READY_FLAG);
    }
    AscendC::CrossCoreWaitFlag<4, PIPE_S>(CSYR2K_MIX_DONE_FLAG);
}

__aicore__ inline void RunCsyr2kMixAic(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow,
    GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf,
    GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags,
    const Csyr2kGemmTilingData& gemmTiling, const Csyr2kDeinterleaveTilingData& deinterleaveTiling)
{
    AscendC::InitSocState();
    bool exactFast = Csyr2kExactFastEnabled(reinterpret_cast<__gm__ uint32_t*>(fastFlags), gemmTiling.fastFlagCount);
    if (exactFast) {
        RunCsyr2kResidualDirectTriangleMixQ8(
            arHalf, arLow, adHalf, adLow, asHalf, asLow, brHalf, brLow, biHalf, biLow, bsHalf, bsLow, t1, t2, t3, t4,
            gemmTiling);
    } else {
        auto* flags = reinterpret_cast<__gm__ uint32_t*>(fastFlags);
        uint32_t readyCount = AscendC::ReadGmByPassDCache(flags + gemmTiling.fastFlagCount);
        while (readyCount < deinterleaveTiling.blockCount) {
            readyCount = AscendC::ReadGmByPassDCache(flags + gemmTiling.fastFlagCount);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
        RunCsyr2kGemm<float, SYRK_ARCH35_FP32_C0>(ar, br, t1, gemmTiling, CSYR2K_ARCH35_BASE_K);
        RunCsyr2kGemm<float, SYRK_ARCH35_FP32_C0>(ai, bi, t2, gemmTiling, CSYR2K_ARCH35_BASE_K);
        RunCsyr2kGemm<float, SYRK_ARCH35_FP32_C0>(ar, bi, t3, gemmTiling, CSYR2K_ARCH35_BASE_K);
        RunCsyr2kGemm<float, SYRK_ARCH35_FP32_C0>(ai, br, t4, gemmTiling, CSYR2K_ARCH35_BASE_K);
    }
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(CSYR2K_MIX_DONE_FLAG);
    AscendC::CrossCoreSetFlag<4, PIPE_FIX>(CSYR2K_MIX_DONE_FLAG + CSYR2K_MIX_SUBBLOCK_FLAG_OFFSET);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ __aicore__ __schedmode__(1) void csyr2k_gemm_mix_epilogue_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf,
    GM_ADDR adLow, GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow,
    GM_ADDR bsHalf, GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags, GM_ADDR alpha,
    GM_ADDR beta, GM_ADDR c, const Csyr2kGemmTilingData gemmTiling,
    const Csyr2kDeinterleaveTilingData deinterleaveTiling, const Csyr2kCombineTilingData combineTiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        RunCsyr2kMixAiv(
            a, b, ar, ai, br, bi, t1, t2, t3, t4, fastFlags, alpha, beta, c, gemmTiling, deinterleaveTiling,
            combineTiling);
    }
    if ASCEND_IS_AIC {
        RunCsyr2kMixAic(
            ar, ai, br, bi, arHalf, arLow, adHalf, adLow, asHalf, asLow, brHalf, brLow, biHalf, biLow, bsHalf, bsLow,
            t1, t2, t3, t4, fastFlags, gemmTiling, deinterleaveTiling);
    }
}

void csyr2k_half_prepare_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR alpha, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow, GM_ADDR asHalf,
    GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf, GM_ADDR bsLow,
    GM_ADDR fastFlags, const Csyr2kFastPrepareTilingData& tiling, uint32_t numBlocks, void* stream)
{
    csyr2k_half_prepare_kernel<<<numBlocks, nullptr, stream>>>(
        a, b, alpha, arHalf, arLow, adHalf, adLow, asHalf, asLow, brHalf, brLow, biHalf, biLow, bsHalf, bsLow,
        fastFlags, tiling);
}

void csyr2k_strict_narrow_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, const Csyr2kStrictNarrowTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    csyr2k_strict_narrow_kernel<<<numBlocks, nullptr, stream>>>(a, b, alpha, beta, c, tiling);
}

void csyr2k_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR fastFlags,
    const Csyr2kDeinterleaveTilingData& tiling, uint32_t numBlocks, void* stream)
{
    csyr2k_deinterleave_kernel<<<numBlocks, nullptr, stream>>>(a, b, ar, ai, br, bi, fastFlags, tiling);
}

void csyr2k_gemm_dispatch_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow,
    GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf,
    GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags,
    const Csyr2kGemmTilingData& tiling, uint32_t numBlocks, void* stream)
{
    csyr2k_gemm_dispatch_kernel<<<numBlocks, nullptr, stream>>>(
        ar, ai, br, bi, arHalf, arLow, adHalf, adLow, asHalf, asLow, brHalf, brLow, biHalf, biLow, bsHalf, bsLow, t1,
        t2, t3, t4, fastFlags, tiling);
}

void csyr2k_partial_accumulate_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR t5, GM_ADDR t6, GM_ADDR t7, GM_ADDR t8,
    const Csyr2kPartialAccumulateTilingData& tiling, uint32_t numBlocks, void* stream)
{
    csyr2k_partial_accumulate_kernel<<<numBlocks, nullptr, stream>>>(t1, t2, t3, t4, t5, t6, t7, t8, tiling);
}

void csyr2k_gemm_mix_epilogue_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf,
    GM_ADDR adLow, GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow,
    GM_ADDR bsHalf, GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags, GM_ADDR alpha,
    GM_ADDR beta, GM_ADDR c, const Csyr2kGemmTilingData& gemmTiling,
    const Csyr2kDeinterleaveTilingData& deinterleaveTiling, const Csyr2kCombineTilingData& combineTiling,
    uint32_t numBlocks, void* stream)
{
    csyr2k_gemm_mix_epilogue_kernel<<<numBlocks, nullptr, stream>>>(
        a, b, ar, ai, br, bi, arHalf, arLow, adHalf, adLow, asHalf, asLow, brHalf, brLow, biHalf, biLow, bsHalf, bsLow,
        t1, t2, t3, t4, fastFlags, alpha, beta, c, gemmTiling, deinterleaveTiling, combineTiling);
}

void csyr2k_combine_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags, GM_ADDR alpha, GM_ADDR beta, GM_ADDR c,
    const Csyr2kCombineTilingData& tiling, uint32_t numBlocks, void* stream)
{
    csyr2k_combine_kernel<<<numBlocks, nullptr, stream>>>(t1, t2, t3, t4, fastFlags, alpha, beta, c, tiling);
}
