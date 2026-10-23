/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cgbmv_tiling_data.h"
#include "cgbmv_kernel.h"

using namespace AscendC;

namespace CgbmvMath {

struct PlanePair {
    LocalTensor<float> re;
    LocalTensor<float> im;
};

struct MaskPair {
    LocalTensor<uint8_t> re;
    LocalTensor<uint8_t> im;
};

struct ProductPlanes {
    LocalTensor<float> rr;
    LocalTensor<float> ii;
    LocalTensor<float> ri;
    LocalTensor<float> ir;
};

template <AscendC::HardEvent EVENT>
__aicore__ inline void SyncPipe()
{
    using namespace AscendC;
    SetFlag<EVENT>(EVENT_ID0);
    WaitFlag<EVENT>(EVENT_ID0);
}

template <AscendC::HardEvent EVT>
__aicore__ inline void PipeSync(TPipe& pipe)
{
    const auto evt = static_cast<int32_t>(pipe.FetchEventID(EVT));
    SetFlag<EVT>(evt);
    WaitFlag<EVT>(evt);
}

__aicore__ inline uint32_t RoundUpTo(uint32_t value, uint32_t granularity)
{
    if (granularity == 0U) {
        return value;
    }
    return (value + granularity - 1U) / granularity * granularity;
}

__aicore__ inline int64_t MinOf(int64_t lhs, int64_t rhs) { return lhs < rhs ? lhs : rhs; }

__aicore__ inline int64_t MaxOf(int64_t lhs, int64_t rhs) { return lhs > rhs ? lhs : rhs; }

__aicore__ inline int64_t StridedElementOffset(uint32_t idx, uint32_t len, int64_t inc)
{
    return inc > 0 ? static_cast<int64_t>(idx) * inc : static_cast<int64_t>(len - 1U - idx) * (-inc);
}

__aicore__ inline void CopyComplexIn(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t floatIdx, uint32_t nComplex)
{
    DataCopyExtParams cp{1U, nComplex * 2U * 4U, 0U, 0U, 0U};
    DataCopyPadExtParams<float> pp{false, 0U, 0U, 0.0F};
    DataCopyPad(dst, src[floatIdx], cp, pp);
}

__aicore__ inline void CopyComplexOut(
    GlobalTensor<float>& dst, int64_t floatIdx, const LocalTensor<float>& src, uint32_t nComplex)
{
    DataCopyExtParams cp{1U, nComplex * 2U * 4U, 0U, 0U, 0U};
    DataCopyPad(dst[floatIdx], src, cp);
}

__aicore__ inline void BindOperandTensors(
    const CgbmvTilingData& t, GlobalTensor<float>& aGM, GlobalTensor<float>& xGM, GlobalTensor<float>& yGM,
    uint32_t xLen, uint32_t yLen)
{
    const int64_t absIncx = t.incx > 0 ? t.incx : -t.incx;
    const int64_t absIncy = t.incy > 0 ? t.incy : -t.incy;
    aGM.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(t.a), static_cast<int64_t>(t.lda) * static_cast<int64_t>(t.n) * 2);
    xGM.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(t.x), (xLen > 0) ? (absIncx * static_cast<int64_t>(xLen - 1U) + 1) * 2 : 0);
    yGM.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(t.y), (yLen > 0) ? (absIncy * static_cast<int64_t>(yLen - 1U) + 1) * 2 : 0);
}

struct ComplexF {
    float re;
    float im;
};

__aicore__ inline ComplexF ComplexMul(float ar, float ai, float br, float bi)
{
    return ComplexF{ar * br - ai * bi, ar * bi + ai * br};
}

__aicore__ inline void MulComplexScalar(
    const PlanePair& dst, const PlanePair& src, float sRe, float sIm, const ProductPlanes& prod, uint32_t count)
{
    Muls(prod.rr, src.re, sRe, count);
    Muls(prod.ii, src.im, sIm, count);
    Muls(prod.ri, src.re, sIm, count);
    Muls(prod.ir, src.im, sRe, count);
    PipeBarrier<PIPE_V>();
    Sub(dst.re, prod.rr, prod.ii, count);
    Add(dst.im, prod.ri, prod.ir, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void RebuildNaNProducts(
    const LocalTensor<float>& commonFactor, const PlanePair& factor, const PlanePair& result, const PlanePair& addend,
    const LocalTensor<float>& zeroPlane, const MaskPair& selfEq, uint32_t count)
{
    constexpr uint32_t COMPARE_LANES = 64U;
    const uint32_t lanes = RoundUpTo(count, COMPARE_LANES);
    Muls(zeroPlane, commonFactor, 0.0F, lanes);
    Muls(factor.re, factor.re, 0.0F, lanes);
    Muls(factor.im, factor.im, 0.0F, lanes);
    Compare(selfEq.re, result.re, result.re, CMPMODE::EQ, lanes);
    Compare(selfEq.im, result.im, result.im, CMPMODE::EQ, lanes);
    PipeBarrier<PIPE_V>();
    Add(factor.re, zeroPlane, factor.re, lanes);
    Add(factor.im, zeroPlane, factor.im, lanes);
    PipeBarrier<PIPE_V>();
    Add(factor.re, factor.re, addend.re, lanes);
    Add(factor.im, factor.im, addend.im, lanes);
    PipeBarrier<PIPE_V>();
    Select(result.re, selfEq.re, result.re, factor.re, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    Select(result.im, selfEq.im, result.im, factor.im, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    PipeBarrier<PIPE_V>();
}

struct TileTailCtx {
    GlobalTensor<float>* yGm;
    LocalTensor<float> yIo;
    LocalTensor<int32_t> betaIdx;
    LocalTensor<int32_t> packIdx;
};

__aicore__ inline void FoldBetaTerm(
    const TileTailCtx& tail, const PlanePair& acc, const PlanePair& old, const ProductPlanes& prod,
    const CgbmvTilingData& t, int64_t row, uint32_t count, bool padTail)
{
    if (t.betaR == 0.0F && t.betaI == 0.0F) {
        return;
    }
    SyncPipe<HardEvent::V_MTE2>();
    if (padTail) {
        const uint8_t right = static_cast<uint8_t>((8U - 2U * count % 8U) % 8U);
        DataCopyPad(
            tail.yIo, (*tail.yGm)[2 * row], DataCopyExtParams{1U, count * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{true, 0U, right, 0.0F});
    } else {
        DataCopyPad(
            tail.yIo, (*tail.yGm)[2 * row], DataCopyExtParams{1U, count * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
    }
    SyncPipe<HardEvent::MTE2_V>();
    Gather(old.re, tail.yIo, tail.betaIdx.ReinterpretCast<uint32_t>(), 0U, count);
    Gather(old.im, tail.yIo, tail.betaIdx.ReinterpretCast<uint32_t>(), sizeof(float), count);
    PipeBarrier<PIPE_V>();
    if (t.betaR != 1.0F || t.betaI != 0.0F) {
        MulComplexScalar(old, old, t.betaR, t.betaI, prod, count);
    }
    Add(acc.re, acc.re, old.re, count);
    Add(acc.im, acc.im, old.im, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void WriteTileOut(
    const TileTailCtx& tail, const LocalTensor<float>& accRe, int64_t row, uint32_t count, bool drain)
{
    Gather(tail.yIo, accRe, tail.packIdx.ReinterpretCast<uint32_t>(), 0U, 2U * count);
    SyncPipe<HardEvent::V_MTE3>();
    DataCopyPad((*tail.yGm)[2 * row], tail.yIo, DataCopyExtParams{1U, count * 8U, 0U, 0U, 0U});
    SyncPipe<HardEvent::MTE3_V>();
    if (drain) {
        SyncPipe<HardEvent::MTE3_MTE2>();
    }
}

__aicore__ inline void SplitInterleaved(
    const LocalTensor<float>& dR, const LocalTensor<float>& dI, const LocalTensor<float>& src,
    const LocalTensor<int32_t>& byteIdx, uint32_t n)
{
    Gather(dR, src, byteIdx.ReinterpretCast<uint32_t>(), 0U, n);
    Gather(dI, src, byteIdx.ReinterpretCast<uint32_t>(), sizeof(float), n);
}

__aicore__ inline void BuildPackTable(
    const LocalTensor<int32_t>& pack, const LocalTensor<int32_t>& half, const LocalTensor<int32_t>& odd,
    uint32_t segLen)
{
    CreateVecIndex(pack, 0, 2U * segLen);
    PipeBarrier<PIPE_V>();
    ShiftRight(half, pack, 1, 2U * segLen);
    PipeBarrier<PIPE_V>();
    Muls(odd, half, 2, 2U * segLen);
    Sub(odd, pack, odd, 2U * segLen);
    PipeBarrier<PIPE_V>();
    Muls(half, half, 4, 2U * segLen);
    Muls(odd, odd, static_cast<int32_t>(segLen * sizeof(float)), 2U * segLen);
    PipeBarrier<PIPE_V>();
    Add(pack, half, odd, 2U * segLen);
    PipeBarrier<PIPE_V>();
}
} // namespace CgbmvMath
using namespace CgbmvMath;

namespace CgbmvGeneral {
using namespace AscendC;

__aicore__ inline void ScalarBandDot(
    const GlobalTensor<float>& aGM, const GlobalTensor<float>& xGM, int64_t outer, uint32_t lo, uint32_t hi,
    const CgbmvTilingData& t, bool transpose, bool conjugate, uint32_t xLen, float& accR, float& accI)
{
    for (uint32_t k = lo; k <= hi; ++k) {
        const int64_t column = transpose ? outer : static_cast<int64_t>(k);
        const int64_t row = transpose ? static_cast<int64_t>(k) : outer;
        const int64_t aOff = 2 * (column * t.lda + t.ku + row - column);
        const int64_t xOff = 2 * StridedElementOffset(k, xLen, t.incx);
        const float vr = aGM.GetValue(aOff);
        const float vi = conjugate ? -aGM.GetValue(aOff + 1) : aGM.GetValue(aOff + 1);
        const float xr = xGM.GetValue(xOff);
        const float xi = xGM.GetValue(xOff + 1);
        accR += vr * xr - vi * xi;
        accI += vr * xi + vi * xr;
    }
}

class GeneralKernel {
public:
    __aicore__ inline void Init(const CgbmvTilingData& tiling, TPipe* pipe)
    {
        plan_ = tiling;
        pipe_ = pipe;
        BindOperandTensors(
            plan_, aGlobal_, xGlobal_, yGlobal_, plan_.trans != 0U ? plan_.m : plan_.n,
            plan_.trans != 0U ? plan_.n : plan_.m);
    }
    __aicore__ inline void Run()
    {
        if (plan_.kl == 0U && plan_.ku == 0U && plan_.lda == 1U && plan_.incx == 1 && plan_.incy == 1 &&
            plan_.m == plan_.n && plan_.trans != 2U && plan_.lenY > 0 && plan_.outputTile > 0U &&
            plan_.blockCount > 0U) {
            RunVectorDiagonal();
            return;
        }
        RunScalarBanded();
    }

private:
    struct DiagPlanes {
        LocalTensor<float> aR;
        LocalTensor<float> aI;
        LocalTensor<float> xR;
        LocalTensor<float> xI;
        LocalTensor<float> p1;
        LocalTensor<float> p2;
        LocalTensor<float> p3;
        LocalTensor<float> p4;
        LocalTensor<float> aC;
        LocalTensor<float> xC;
        LocalTensor<int32_t> gatherIdx;
    };

    __aicore__ inline void DiagMulTile(DiagPlanes& q, uint32_t count)
    {
        Mul(q.p1, q.aR, q.xR, count);
        Mul(q.p2, q.aI, q.xI, count);
        Mul(q.p3, q.aR, q.xI, count);
        Mul(q.p4, q.aI, q.xR, count);
        PipeBarrier<PIPE_V>();
        Sub(q.aR, q.p1, q.p2, count);
        Add(q.aI, q.p3, q.p4, count);
        PipeBarrier<PIPE_V>();
        CgbmvMath::MulComplexScalar(
            PlanePair{q.xR, q.xI}, PlanePair{q.aR, q.aI}, plan_.alphaR, plan_.alphaI,
            ProductPlanes{q.p1, q.p2, q.p3, q.p4}, count);
    }

    __aicore__ inline void DiagFoldBeta(DiagPlanes& q, int64_t start, uint32_t count)
    {
        if (plan_.betaR == 0.0F && plan_.betaI == 0.0F) {
            return;
        }
        PipeSync<HardEvent::V_MTE2>(*pipe_);
        CopyComplexIn(q.aC, yGlobal_, 2 * start, count);
        PipeSync<HardEvent::MTE2_V>(*pipe_);
        Gather(q.aR, q.aC, q.gatherIdx.ReinterpretCast<uint32_t>(), 0U, count);
        Gather(q.aI, q.aC, q.gatherIdx.ReinterpretCast<uint32_t>(), sizeof(float), count);
        PipeBarrier<PIPE_V>();
        CgbmvMath::MulComplexScalar(
            PlanePair{q.aR, q.aI}, PlanePair{q.aR, q.aI}, plan_.betaR, plan_.betaI,
            ProductPlanes{q.p1, q.p2, q.p3, q.p4}, count);
        Add(q.xR, q.xR, q.aR, count);
        Add(q.xI, q.xI, q.aI, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DiagStoreTile(DiagPlanes& q, int64_t start, uint32_t count)
    {
        PipeSync<HardEvent::V_S>(*pipe_);
        for (uint32_t k = 0; k < count; ++k) {
            q.xC.SetValue(2 * k, q.xR.GetValue(k));
            q.xC.SetValue(2 * k + 1, q.xI.GetValue(k));
        }
        PipeSync<HardEvent::S_MTE3>(*pipe_);
        CopyComplexOut(yGlobal_, 2 * start, q.xC, count);
        PipeSync<HardEvent::MTE3_MTE2>(*pipe_);
    }

    __aicore__ inline void RunVectorDiagonal()
    {
        const int64_t tileCount = (plan_.lenY + plan_.outputTile - 1) / plan_.outputTile;
        const uint32_t cap = CgbmvMath::RoundUpTo(plan_.outputTile, 64U);
        TBuf<TPosition::VECCALC> planeBuf;
        TBuf<TPosition::VECCALC> ioBuf;
        TBuf<TPosition::VECCALC> idxBuf;
        pipe_->InitBuffer(planeBuf, 8U * cap * sizeof(float));
        pipe_->InitBuffer(ioBuf, 4U * cap * sizeof(float));
        pipe_->InitBuffer(idxBuf, cap * sizeof(int32_t));
        DiagPlanes q;
        LocalTensor<float> plane = planeBuf.Get<float>();
        LocalTensor<float> packed = ioBuf.Get<float>();
        q.aR = plane;
        q.aI = plane[cap];
        q.xR = plane[2U * cap];
        q.xI = plane[3U * cap];
        q.p1 = plane[4U * cap];
        q.p2 = plane[5U * cap];
        q.p3 = plane[6U * cap];
        q.p4 = plane[7U * cap];
        q.aC = packed;
        q.xC = packed[2U * cap];
        q.gatherIdx = idxBuf.Get<int32_t>();

        CreateVecIndex(q.gatherIdx, 0, cap);
        Muls(q.gatherIdx, q.gatherIdx, static_cast<int32_t>(2U * sizeof(float)), cap);
        PipeBarrier<PIPE_V>();
        for (int64_t tile = GetBlockIdx(); tile < tileCount; tile += plan_.blockCount) {
            const int64_t start = tile * plan_.outputTile;
            const uint32_t count = static_cast<uint32_t>(CgbmvMath::MinOf(plan_.outputTile, plan_.lenY - start));
            PipeSync<HardEvent::V_MTE2>(*pipe_);
            CopyComplexIn(q.aC, aGlobal_, 2 * start, count);
            CopyComplexIn(q.xC, xGlobal_, 2 * start, count);
            PipeSync<HardEvent::MTE2_V>(*pipe_);
            Gather(q.aR, q.aC, q.gatherIdx.ReinterpretCast<uint32_t>(), 0U, count);
            Gather(q.aI, q.aC, q.gatherIdx.ReinterpretCast<uint32_t>(), sizeof(float), count);
            Gather(q.xR, q.xC, q.gatherIdx.ReinterpretCast<uint32_t>(), 0U, count);
            Gather(q.xI, q.xC, q.gatherIdx.ReinterpretCast<uint32_t>(), sizeof(float), count);
            PipeBarrier<PIPE_V>();
            DiagMulTile(q, count);
            DiagFoldBeta(q, start, count);
            DiagStoreTile(q, start, count);
        }
    }

    __aicore__ inline void RunScalarBanded()
    {
        if (GetBlockIdx() != 0) {
            return;
        }
        const bool skipDot = plan_.alphaR == 0.0F && plan_.alphaI == 0.0F;
        const bool isTrans = plan_.trans != 0U;
        const uint32_t outLen = isTrans ? plan_.n : plan_.m;
        for (uint32_t o = 0; o < outLen; ++o) {
            float accR = 0.0F;
            float accI = 0.0F;
            const uint32_t lo = isTrans ? ((o > plan_.ku) ? o - plan_.ku : 0U) : ((o > plan_.kl) ? o - plan_.kl : 0U);
            const uint32_t span = isTrans ? plan_.m : plan_.n;
            const uint32_t width = isTrans ? plan_.kl : plan_.ku;
            const uint32_t hi = (span - 1U < o + width) ? span - 1U : o + width;
            if (!skipDot) {
                ScalarBandDot(
                    aGlobal_, xGlobal_, o, lo, hi, plan_, isTrans, plan_.trans == 2U, isTrans ? plan_.m : plan_.n, accR,
                    accI);
            }
            const ComplexF acc = ComplexMul(plan_.alphaR, plan_.alphaI, accR, accI);
            const int64_t yOff = 2 * StridedElementOffset(o, outLen, plan_.incy);
            float outR = acc.re;
            float outI = acc.im;
            if (plan_.betaR != 0.0F || plan_.betaI != 0.0F) {
                const ComplexF betaTerm =
                    ComplexMul(plan_.betaR, plan_.betaI, yGlobal_.GetValue(yOff), yGlobal_.GetValue(yOff + 1));
                outR += betaTerm.re;
                outI += betaTerm.im;
            }
            yGlobal_.SetValue(yOff, outR);
            yGlobal_.SetValue(yOff + 1, outI);
        }
    }
    CgbmvTilingData plan_{};
    TPipe* pipe_ = nullptr;
    GlobalTensor<float> aGlobal_, xGlobal_, yGlobal_;
};
} // namespace CgbmvGeneral
namespace CgbmvRowTiles {
using namespace AscendC;
using namespace CgbmvMath;

constexpr uint32_t SWEEP_COLS = 64U;
constexpr uint32_t CPLX_BYTES = 8U;
constexpr uint32_t BLK_BYTES = 32U;
constexpr uint32_t F32_PER_BLK = BLK_BYTES / sizeof(float);
constexpr uint32_t TRANSPOSE_SLACK_ROWS = 1U << 2;

template <uint32_t ROW_COUNT>
class RowTilesKernel {
public:
    static constexpr uint32_t ROWS = ROW_COUNT;
    static constexpr uint32_t TILE_LANES = ROWS * SWEEP_COLS;
    static constexpr uint32_t PLANE_PITCH = TILE_LANES + F32_PER_BLK;
    static constexpr uint32_t DMA_ROWS = ROWS + TRANSPOSE_SLACK_ROWS;
    static constexpr uint32_t DMA_A_CPLX = DMA_ROWS * SWEEP_COLS;
    static constexpr uint32_t MASK_BYTES = TILE_LANES / 8U;
    static constexpr uint32_t MASK_WORDS = MASK_BYTES / sizeof(uint16_t);
    static constexpr uint32_t DMA_FLOATS = 2U * (DMA_A_CPLX + SWEEP_COLS + ROWS);
    static constexpr uint32_t UB_FOOTPRINT =
        (DMA_FLOATS + 6U * PLANE_PITCH + F32_PER_BLK + TILE_LANES + 4U * ROWS) * sizeof(float) +
        (2U * TILE_LANES + 4U * ROWS + SWEEP_COLS) * sizeof(int32_t) + 2U * MASK_BYTES;
    static_assert(
        ROWS == 8U || ROWS == 32U || ROWS == 48U || ROWS == 64U, "unsupported row-panel width: expected 8/32/48/64");
    static_assert(UB_FOOTPRINT < 184U * 1024U, "row-panel layout cannot fit into user UB");
    __aicore__ static inline bool Matches(const CgbmvTilingData& tiling)
    {
        return tiling.mode == 0U && tiling.trans == 0U && tiling.outputTile == ROWS && tiling.incx == 1 &&
               tiling.incy == 1 && tiling.kl + tiling.ku > 0;
    }
    __aicore__ inline void Init(const CgbmvTilingData& tiling, TPipe* pipe)
    {
        plan_ = tiling;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(plan_.a));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(plan_.x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(plan_.y));
        CarveUb(pipe);
        BuildLaneTables();
        BuildStoreTables();
    }
    __aicore__ inline void Run()
    {
        if (plan_.blockCount == 0U || plan_.lenY <= 0) {
            return;
        }
        const int64_t tiles = (plan_.lenY + ROWS - 1) / ROWS;
        for (int64_t tile = GetBlockIdx(); tile < tiles; tile += plan_.blockCount) {
            const int64_t row = tile * ROWS;
            const uint32_t count = static_cast<uint32_t>(MinOf(ROWS, plan_.lenY - row));
            const bool hasWork = ComputeTileSums(row, count);
            if (!hasWork && plan_.betaR == 0.0F && plan_.betaI == 0.0F) {
                Duplicate(yIo_, 0.0F, 2U * count);
                PipeBarrier<PIPE_V>();
                SyncPipe<HardEvent::V_MTE3>();
                DataCopyPad(yGm_[2 * row], yIo_, DataCopyExtParams{1U, count * CPLX_BYTES, 0U, 0U, 0U});
                SyncPipe<HardEvent::MTE3_V>();
            } else {
                FoldBetaTerm(
                    tail_, PlanePair{accRe_, accIm_}, PlanePair{redRe_, redIm_},
                    ProductPlanes{mulRR_, mulII_, mulRI_, mulIR_}, plan_, row, count, false);
                WriteTileOut(tail_, accRe_, row, count, false);
            }
            if (hasWork && ROWS >= 32U && tile + plan_.blockCount < tiles) {
                const int64_t nRow = (tile + plan_.blockCount) * ROWS;
                const uint32_t nCount = static_cast<uint32_t>(MinOf(ROWS, plan_.lenY - nRow));
                const int64_t nActive = MinOf(nCount, plan_.n + plan_.kl - nRow);
                const int64_t nBegin = MaxOf(0, nRow - plan_.kl);
                const int64_t nEnd = MinOf(plan_.n, nRow + nActive + plan_.ku);
                if (nActive > 0 && nBegin < nEnd) {
                    SyncPipe<HardEvent::V_MTE2>();
                    DmaPanelInputs(
                        nRow, nBegin, static_cast<uint32_t>(nActive),
                        static_cast<uint32_t>(MinOf(SWEEP_COLS, nEnd - nBegin)));
                    pfRow_ = nRow;
                    pfCol_ = nBegin;
                    pfInFlight_ = true;
                }
            }
        }
    }

private:
    __aicore__ inline void CarveUb(TPipe* pipe)
    {
        pipe->InitBuffer(dmaBuf_, DMA_FLOATS * sizeof(float));
        pipe->InitBuffer(planeBuf_, 4U * PLANE_PITCH * sizeof(float));
        pipe->InitBuffer(scratchBuf_, (2U * PLANE_PITCH + F32_PER_BLK) * sizeof(float));
        pipe->InitBuffer(idxBuf_, (2U * TILE_LANES + 4U * ROWS + SWEEP_COLS) * sizeof(int32_t));
        pipe->InitBuffer(laneBuf_, TILE_LANES * sizeof(float));
        pipe->InitBuffer(outBuf_, 4U * ROWS * sizeof(float));
        pipe->InitBuffer(maskBuf_, 2U * MASK_BYTES);
        aDma_ = dmaBuf_.Get<float>();
        xDma_ = aDma_[2U * DMA_A_CPLX];
        yIo_ = xDma_[2U * SWEEP_COLS];
        aRe_ = planeBuf_.Get<float>();
        aIm_ = aRe_[PLANE_PITCH];
        xRe_ = aRe_[2U * PLANE_PITCH];
        xIm_ = aRe_[3U * PLANE_PITCH];
        mulRR_ = scratchBuf_.Get<float>()[F32_PER_BLK];
        mulRI_ = mulRR_[PLANE_PITCH];
        mulII_ = aDma_;
        mulIR_ = aDma_[TILE_LANES];
        aIdx_ = idxBuf_.Get<int32_t>();
        colLaneF_ = aIdx_.ReinterpretCast<float>()[TILE_LANES];
        betaIdx_ = aIdx_[2U * TILE_LANES];
        packIdx_ = betaIdx_[2U * ROWS];
        xIdx_ = packIdx_[2U * ROWS];
        idxWork_ = mulRR_.ReinterpretCast<int32_t>();
        idxAux_ = mulII_.ReinterpretCast<int32_t>();
        idxTemp_ = mulRI_.ReinterpretCast<int32_t>();
        colIdx_ = mulIR_.ReinterpretCast<int32_t>();
        diagOffset_ = laneBuf_.Get<float>();
        accRe_ = outBuf_.Get<float>();
        accIm_ = accRe_[ROWS];
        redRe_ = accRe_[2U * ROWS];
        redIm_ = accRe_[3U * ROWS];
        laneMask_ = maskBuf_.Get<uint8_t>();
        laneMaskTmp_ = laneMask_[MASK_BYTES];
        tail_ = TileTailCtx{&yGm_, yIo_, betaIdx_, packIdx_};
    }
    __aicore__ inline void BuildLaneTables()
    {
        CreateVecIndex(idxWork_, 0, TILE_LANES);
        PipeBarrier<PIPE_V>();
        ShiftRight(idxAux_, idxWork_, 6, TILE_LANES);
        PipeBarrier<PIPE_V>();
        Muls(idxTemp_, idxAux_, static_cast<int32_t>(SWEEP_COLS), TILE_LANES);
        PipeBarrier<PIPE_V>();
        Sub(colIdx_, idxWork_, idxTemp_, TILE_LANES);
        PipeBarrier<PIPE_V>();
        const int32_t aStride =
            plan_.lda < ROWS + 1U ? static_cast<int32_t>(plan_.lda - 1) : static_cast<int32_t>(DMA_ROWS);
        const bool gatherPath = aStride % 4 != 0;
        if (gatherPath) {
            Muls(aIdx_, colIdx_, aStride, TILE_LANES);
        }
        Muls(xIdx_, colIdx_, static_cast<int32_t>(CPLX_BYTES), SWEEP_COLS);
        Cast(mulRR_, idxAux_, RoundMode::CAST_NONE, TILE_LANES);
        Cast(colLaneF_, colIdx_, RoundMode::CAST_NONE, TILE_LANES);
        PipeBarrier<PIPE_V>();
        if (gatherPath) {
            Add(aIdx_, aIdx_, idxAux_, TILE_LANES);
        }
        Sub(diagOffset_, colLaneF_, mulRR_, TILE_LANES);
        PipeBarrier<PIPE_V>();
        if (gatherPath) {
            Muls(aIdx_, aIdx_, static_cast<int32_t>(CPLX_BYTES), TILE_LANES);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void BuildStoreTables()
    {
        CreateVecIndex(idxWork_, 0, 2U * ROWS);
        PipeBarrier<PIPE_V>();
        if (plan_.betaR != 0.0F || plan_.betaI != 0.0F) {
            Muls(betaIdx_, idxWork_, static_cast<int32_t>(CPLX_BYTES), ROWS);
        }
        ShiftRight(idxAux_, idxWork_, 1, 2U * ROWS);
        PipeBarrier<PIPE_V>();
        Muls(idxTemp_, idxAux_, 2, 2U * ROWS);
        Sub(idxTemp_, idxWork_, idxTemp_, 2U * ROWS);
        PipeBarrier<PIPE_V>();
        Muls(idxTemp_, idxTemp_, static_cast<int32_t>(ROWS * sizeof(float)), 2U * ROWS);
        Muls(idxAux_, idxAux_, static_cast<int32_t>(sizeof(float)), 2U * ROWS);
        PipeBarrier<PIPE_V>();
        Add(packIdx_, idxAux_, idxTemp_, 2U * ROWS);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void DmaBandA(int64_t row, int64_t col, uint32_t activeRows, uint32_t countCols)
    {
        const DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        const int64_t head = col * (plan_.lda - 1) + plan_.ku + row;
        if (plan_.lda < ROWS + 1U) {
            const uint32_t span = static_cast<uint32_t>((countCols - 1U) * (plan_.lda - 1) + activeRows);
            const DataCopyExtParams oneSpan{1U, span * CPLX_BYTES, 0U, 0U, 0U};
            DataCopyPad(aDma_, aGm_[2 * head], oneSpan, noPad);
            return;
        }
        const uint32_t bytes = activeRows * CPLX_BYTES;
        const uint32_t blocks = (bytes + BLK_BYTES - 1U) / BLK_BYTES;
        const uint32_t dstGap = DMA_ROWS * CPLX_BYTES / BLK_BYTES - blocks;
        const uint64_t srcGap = static_cast<uint64_t>(plan_.lda - 1 - activeRows) * CPLX_BYTES;
        if (srcGap <= 0xffffffffU) {
            const DataCopyExtParams byColumn{
                static_cast<uint16_t>(countCols), bytes, static_cast<uint32_t>(srcGap), dstGap, 0U};
            DataCopyPad(aDma_, aGm_[2 * head], byColumn, noPad);
            return;
        }
        const DataCopyExtParams oneColumn{1U, bytes, 0U, 0U, 0U};
        for (uint32_t index = 0U; index < countCols; ++index) {
            DataCopyPad(aDma_[2U * DMA_ROWS * index], aGm_[2 * (head + index * (plan_.lda - 1))], oneColumn, noPad);
        }
    }
    __aicore__ inline void DmaPanelInputs(int64_t row, int64_t col, uint32_t activeRows, uint32_t countCols)
    {
        DmaBandA(row, col, activeRows, countCols);
        const DataCopyExtParams vec{1U, countCols * CPLX_BYTES, 0U, 0U, 0U};
        DataCopyPad(xDma_, xGm_[2 * col], vec, DataCopyPadExtParams<float>{false, 0U, 0U, 0.0F});
    }
    __aicore__ inline void TransposeStagedA(uint32_t countCols)
    {
        const uint32_t stride = plan_.lda < ROWS + 1U ? static_cast<uint32_t>(plan_.lda - 1) : DMA_ROWS;
        if (stride % 4U != 0U) {
            Gather(aRe_, aDma_, aIdx_.ReinterpretCast<uint32_t>(), 0U, TILE_LANES);
            Gather(aIm_, aDma_, aIdx_.ReinterpretCast<uint32_t>(), sizeof(float), TILE_LANES);
            return;
        }
        constexpr bool WIDE_TILE = ROWS >= 32U;
        const uint32_t groups = WIDE_TILE ? (countCols + 15U) / 16U : ROWS / 4U;
        const TransDataTo5HDParams params{
            false, false, WIDE_TILE ? ROWS / 4U : 4U, WIDE_TILE ? 32U : 2U,
            static_cast<uint16_t>(WIDE_TILE ? 1U : 4U * stride)};
        for (uint32_t group = 0U; group < groups; ++group) {
            const uint32_t baseRow = WIDE_TILE ? 0U : group * 4U;
            const uint32_t baseCol = WIDE_TILE ? group * 16U : 0U;
            uint64_t src[16];
            uint64_t dst[16];
            for (uint32_t k = 0U; k < 16U; ++k) {
                src[k] = reinterpret_cast<uint64_t>(aDma_[2U * ((k + baseCol) * stride + baseRow)].GetPhyAddr());
            }
            for (uint32_t k = 0U; k < 4U; ++k) {
                const uint32_t at = (baseRow + k) * SWEEP_COLS + baseCol;
                dst[4U * k] = reinterpret_cast<uint64_t>(aRe_[at].GetPhyAddr());
                dst[4U * k + 1U] = reinterpret_cast<uint64_t>(aRe_[at + 8U].GetPhyAddr());
                dst[4U * k + 2U] = reinterpret_cast<uint64_t>(aIm_[at].GetPhyAddr());
                dst[4U * k + 3U] = reinterpret_cast<uint64_t>(aIm_[at + 8U].GetPhyAddr());
            }
            TransDataTo5HD<float>(dst, src, params);
        }
    }
    __aicore__ inline void MaterializeOperands(uint32_t countCols)
    {
        TransposeStagedA(countCols);
        Gather(xRe_, xDma_, xIdx_.ReinterpretCast<uint32_t>(), 0U, SWEEP_COLS);
        Gather(xIm_, xDma_, xIdx_.ReinterpretCast<uint32_t>(), sizeof(float), SWEEP_COLS);
        PipeBarrier<PIPE_V>();
        ScaleInPlace(xRe_, xIm_, plan_.alphaR, plan_.alphaI, SWEEP_COLS);
        const CopyRepeatParams broadcast{1U, 1U, SWEEP_COLS / F32_PER_BLK, 0U};
        Copy(xRe_[SWEEP_COLS], xRe_, static_cast<uint64_t>(SWEEP_COLS), static_cast<uint8_t>(ROWS - 1U), broadcast);
        Copy(xIm_[SWEEP_COLS], xIm_, static_cast<uint64_t>(SWEEP_COLS), static_cast<uint8_t>(ROWS - 1U), broadcast);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StagePanel(int64_t row, int64_t col, uint32_t activeRows, uint32_t countCols)
    {
        SyncPipe<HardEvent::V_MTE2>();
        DmaPanelInputs(row, col, activeRows, countCols);
        SyncPipe<HardEvent::MTE2_V>();
        MaterializeOperands(countCols);
    }
    __aicore__ inline void ScaleInPlace(
        const LocalTensor<float>& re, const LocalTensor<float>& im, float sRe, float sIm, uint32_t count)
    {
        if (sRe == 1.0F) {
            Muls(mulII_, im, sIm, count);
            Muls(mulRI_, re, sIm, count);
            PipeBarrier<PIPE_V>();
            Sub(re, re, mulII_, count);
            Add(im, mulRI_, im, count);
            PipeBarrier<PIPE_V>();
            return;
        }
        MulComplexScalar(
            PlanePair{re, im}, PlanePair{re, im}, sRe, sIm, ProductPlanes{mulRR_, mulII_, mulRI_, mulIR_}, count);
    }
    __aicore__ inline void IntersectMask()
    {
        And(laneMask_.ReinterpretCast<uint16_t>(), laneMask_.ReinterpretCast<uint16_t>(),
            laneMaskTmp_.ReinterpretCast<uint16_t>(), MASK_WORDS);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void MakeDiagonalMask(int32_t lower, int32_t upper)
    {
        const bool clipLow = lower > 1 - static_cast<int32_t>(ROWS);
        const bool clipHigh = upper < static_cast<int32_t>(SWEEP_COLS - 1U);
        if (!clipLow) {
            if (clipHigh) {
                Compares(laneMask_, diagOffset_, static_cast<float>(upper), CMPMODE::LE, TILE_LANES);
                PipeBarrier<PIPE_V>();
                return;
            }
            Duplicate(laneMask_.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0xffffU), MASK_WORDS);
            PipeBarrier<PIPE_V>();
            return;
        }
        Compares(laneMask_, diagOffset_, static_cast<float>(lower), CMPMODE::GE, TILE_LANES);
        if (!clipHigh) {
            PipeBarrier<PIPE_V>();
            return;
        }
        Compares(laneMaskTmp_, diagOffset_, static_cast<float>(upper), CMPMODE::LE, TILE_LANES);
        PipeBarrier<PIPE_V>();
        IntersectMask();
    }
    __aicore__ inline void MaskPanelLanes(int64_t row, int64_t col, uint32_t activeRows, uint32_t countCols)
    {
        const int64_t lowestInBand = row + static_cast<int64_t>(activeRows) - 1 - plan_.kl;
        const int64_t highestInBand = row + plan_.ku;
        if (countCols == SWEEP_COLS && col >= lowestInBand && col + SWEEP_COLS - 1 <= highestInBand) {
            return;
        }
        const int64_t rawLow = row - col - static_cast<int64_t>(plan_.kl);
        const int64_t rawHigh = row - col + static_cast<int64_t>(plan_.ku);
        const int32_t lower =
            static_cast<int32_t>(MaxOf(-static_cast<int64_t>(ROWS), MinOf(static_cast<int64_t>(SWEEP_COLS), rawLow)));
        const int32_t upper =
            static_cast<int32_t>(MaxOf(-static_cast<int64_t>(ROWS), MinOf(static_cast<int64_t>(SWEEP_COLS), rawHigh)));
        MakeDiagonalMask(lower, upper);
        if (countCols < SWEEP_COLS) {
            Compares(
                laneMaskTmp_, colLaneF_, static_cast<float>(static_cast<int32_t>(countCols)), CMPMODE::LT, TILE_LANES);
            PipeBarrier<PIPE_V>();
            IntersectMask();
        }
        Select(aRe_, laneMask_, aRe_, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_LANES);
        Select(aIm_, laneMask_, aIm_, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_LANES);
        Select(xRe_, laneMask_, xRe_, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_LANES);
        Select(xIm_, laneMask_, xIm_, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_LANES);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void MulReduceQuick(uint32_t activeRows)
    {
        Mul(mulRR_, aRe_, xIm_, TILE_LANES);
        Mul(mulRI_, aIm_, xRe_, TILE_LANES);
        PipeBarrier<PIPE_V>();
        Mul(aRe_, aRe_, xRe_, TILE_LANES);
        Mul(aIm_, aIm_, xIm_, TILE_LANES);
        PipeBarrier<PIPE_V>();
        Sub(aRe_, aRe_, aIm_, TILE_LANES);
        Add(mulRR_, mulRR_, mulRI_, TILE_LANES);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(redRe_, aRe_, SWEEP_COLS, activeRows, 1, 1, SWEEP_COLS / F32_PER_BLK);
        WholeReduceSum(redIm_, mulRR_, SWEEP_COLS, activeRows, 1, 1, SWEEP_COLS / F32_PER_BLK);
        PipeBarrier<PIPE_V>();
        Add(accRe_, accRe_, redRe_, activeRows);
        Add(accIm_, accIm_, redIm_, activeRows);
    }
    __aicore__ inline void MulReduceStrict(bool restore, uint32_t activeRows)
    {
        Mul(mulRR_, aRe_, xRe_, TILE_LANES);
        Mul(mulII_, aIm_, xIm_, TILE_LANES);
        Mul(mulRI_, aRe_, xIm_, TILE_LANES);
        Mul(mulIR_, aIm_, xRe_, TILE_LANES);
        PipeBarrier<PIPE_V>();
        Sub(mulRR_, mulRR_, mulII_, TILE_LANES);
        Add(mulRI_, mulRI_, mulIR_, TILE_LANES);
        PipeBarrier<PIPE_V>();
        if (restore) {
            Muls(mulII_, mulII_, -1.0F, TILE_LANES);
            PipeBarrier<PIPE_V>();
            RebuildNaNProducts(
                aRe_, PlanePair{xRe_, xIm_}, PlanePair{mulRR_, mulRI_}, PlanePair{mulII_, mulIR_}, aIm_,
                MaskPair{laneMask_, laneMaskTmp_}, TILE_LANES);
        }
        WholeReduceSum(redRe_, mulRR_, SWEEP_COLS, activeRows, 1, 1, SWEEP_COLS / F32_PER_BLK);
        WholeReduceSum(redIm_, mulRI_, SWEEP_COLS, activeRows, 1, 1, SWEEP_COLS / F32_PER_BLK);
        PipeBarrier<PIPE_V>();
        Add(accRe_, accRe_, redRe_, activeRows);
        Add(accIm_, accIm_, redIm_, activeRows);
    }
    __aicore__ inline void SweepBandPanels(int64_t row, uint32_t active, int64_t begin, int64_t end, bool strict)
    {
        const bool pipelined = (ROWS >= 32U) && !strict;
        if (pipelined) {
            if (pfInFlight_ && pfRow_ == row && pfCol_ == begin) {
                SyncPipe<HardEvent::MTE2_V>();
                MaterializeOperands(static_cast<uint32_t>(MinOf(SWEEP_COLS, end - begin)));
            } else {
                StagePanel(row, begin, active, static_cast<uint32_t>(MinOf(SWEEP_COLS, end - begin)));
            }
            pfInFlight_ = false;
        }
        for (int64_t col = begin; col < end; col += SWEEP_COLS) {
            const uint32_t countCols = static_cast<uint32_t>(MinOf(SWEEP_COLS, end - col));
            const int64_t next = col + SWEEP_COLS;
            if (pipelined && next < end) {
                SyncPipe<HardEvent::V_MTE2>();
                DmaPanelInputs(row, next, active, static_cast<uint32_t>(MinOf(SWEEP_COLS, end - next)));
            }
            if (!pipelined) {
                StagePanel(row, col, active, countCols);
            }
            MaskPanelLanes(row, col, active, countCols);
            if (pipelined) {
                MulReduceQuick(active);
            } else {
                MulReduceStrict(strict, active);
            }
            if (pipelined && next < end) {
                PipeBarrier<PIPE_V>();
                SyncPipe<HardEvent::MTE2_V>();
                MaterializeOperands(static_cast<uint32_t>(MinOf(SWEEP_COLS, end - next)));
            }
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline bool ComputeTileSums(int64_t row, uint32_t count)
    {
        Duplicate(accRe_, 0.0F, 2U * ROWS);
        PipeBarrier<PIPE_V>();
        const int64_t active = MinOf(count, plan_.n + plan_.kl - row);
        if (active <= 0) {
            return false;
        }
        const int64_t begin = MaxOf(0, row - plan_.kl);
        const int64_t end = MinOf(plan_.n, row + active + plan_.ku);
        if (begin >= end) {
            return false;
        }
        SweepBandPanels(row, static_cast<uint32_t>(active), begin, end, ROWS < 32U);
        if constexpr (ROWS >= 32U) {
            Add(aIm_, accRe_, accIm_, ROWS);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(aIm_[SWEEP_COLS], aIm_, ROWS, 1, 1, 1, SWEEP_COLS / F32_PER_BLK);
            SyncPipe<HardEvent::V_S>();
            const uint32_t probeBits = aIm_.ReinterpretCast<uint32_t>().GetValue(SWEEP_COLS);
            const bool hasNaN = (probeBits << 1U) > 0xff000000U;
            SyncPipe<HardEvent::S_V>();
            if (hasNaN) {
                Duplicate(accRe_, 0.0F, 2U * ROWS);
                PipeBarrier<PIPE_V>();
                SweepBandPanels(row, static_cast<uint32_t>(active), begin, end, true);
            }
        }
        return true;
    }
    CgbmvTilingData plan_{};
    GlobalTensor<float> aGm_, xGm_, yGm_;
    TBuf<TPosition::VECCALC> dmaBuf_, planeBuf_, scratchBuf_, idxBuf_, laneBuf_, outBuf_, maskBuf_;
    LocalTensor<float> aDma_, xDma_, yIo_, aRe_, aIm_, xRe_, xIm_;
    LocalTensor<float> mulRR_, mulII_, mulRI_, mulIR_, diagOffset_, colLaneF_;
    LocalTensor<float> accRe_, accIm_, redRe_, redIm_;
    LocalTensor<int32_t> aIdx_, idxWork_, idxAux_, idxTemp_, colIdx_, betaIdx_, packIdx_, xIdx_;
    LocalTensor<uint8_t> laneMask_, laneMaskTmp_;
    TileTailCtx tail_;
    bool pfInFlight_ = false;
    int64_t pfRow_ = 0;
    int64_t pfCol_ = 0;
};
} // namespace CgbmvRowTiles

namespace CgbmvSweep {
using namespace AscendC;
using namespace CgbmvMath;

constexpr uint32_t GROUP_COLS = 16U;
constexpr uint32_t ROW_CHUNK = 128U;
constexpr uint32_t SLOT_HEAD = 8U;
constexpr uint32_t SLOT_STRIDE = ROW_CHUNK + SLOT_HEAD;
constexpr uint32_t OUT_CHUNK = 1024U;
constexpr uint32_t CPLX_BYTES = 8U;

class SweepKernel {
public:
    __aicore__ static inline bool Matches(const CgbmvTilingData& t)
    {
        if (t.incx != 1 || t.incy != 1 || t.lenY <= 0 || t.outputTile == 0U || t.blockCount == 0U || t.y % 32U != 0U) {
            return false;
        }
        if (t.mode == 1U) {
            return t.kl + t.ku > 1 && t.kl + t.ku + 1 <= 4096U && t.lda <= 0xffffffffU / 8U;
        }
        return t.mode == 0U && t.trans == 0U && t.kl + t.ku > 1 && t.kl + t.ku + 1 <= 128U;
    }

    __aicore__ inline void Init(const CgbmvTilingData& tiling, TPipe* pipe)
    {
        plan_ = tiling;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(plan_.a));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(plan_.x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(plan_.y));
        CarveUb(pipe);
        BuildSweepTables();
    }

    __aicore__ inline void Run()
    {
        const int64_t lo = static_cast<int64_t>(GetBlockIdx()) * plan_.outputTile;
        if (lo >= plan_.lenY) {
            return;
        }
        const int64_t hi = MinOf(plan_.lenY, lo + plan_.outputTile);
        for (int64_t segLo = lo; segLo < hi; segLo += OUT_CHUNK) {
            const uint32_t seg = static_cast<uint32_t>(MinOf(OUT_CHUNK, hi - segLo));
            LocalTensor<float> accRe = accBuf_.Get<float>();
            LocalTensor<float> accIm = accRe[OUT_CHUNK];
            if (plan_.mode == 1U) {
                SweepTransposed(segLo, seg, accRe, accIm);
            } else {
                SweepNormal(segLo, seg, accRe, accIm);
            }
            FinishSegment(segLo, seg, accRe, accIm);
        }
    }

private:
    __aicore__ inline void CarveUb(TPipe* pipe)
    {
        pipe->InitBuffer(stagedBuf_, 2U * GROUP_COLS * SLOT_STRIDE * sizeof(float));
        pipe->InitBuffer(soaBuf_, 2U * GROUP_COLS * SLOT_STRIDE * sizeof(float));
        pipe->InitBuffer(workBuf_, (11U * ROW_CHUNK + 2U * GROUP_COLS) * sizeof(float));
        pipe->InitBuffer(xIoBuf_, 2U * (ROW_CHUNK + GROUP_COLS) * sizeof(float));
        pipe->InitBuffer(xSoaBuf_, 2U * (ROW_CHUNK + GROUP_COLS) * sizeof(float));
        pipe->InitBuffer(accBuf_, 2U * OUT_CHUNK * sizeof(float));
        pipe->InitBuffer(yIoBuf_, 2U * OUT_CHUNK * sizeof(float));
        pipe->InitBuffer(packBuf_, 6U * OUT_CHUNK * sizeof(int32_t));
        pipe->InitBuffer(interBuf_, (GROUP_COLS * SLOT_STRIDE + ROW_CHUNK + 32U) * sizeof(int32_t));
        staged_ = stagedBuf_.Get<float>();
        aSoaR_ = soaBuf_.Get<float>();
        aSoaI_ = aSoaR_[GROUP_COLS * SLOT_STRIDE];
        LocalTensor<float> work = workBuf_.Get<float>();
        p1_ = work;
        p2_ = work[ROW_CHUNK];
        p3_ = work[2U * ROW_CHUNK];
        p4_ = work[3U * ROW_CHUNK];
        pr_ = work[4U * ROW_CHUNK];
        pi_ = work[5U * ROW_CHUNK];
        redTmp_ = work[6U * ROW_CHUNK];
        xSlR_ = work[7U * ROW_CHUNK];
        xSlI_ = work[8U * ROW_CHUNK];
        partialSlots_ = work[9U * ROW_CHUNK];
        partial_ = work[9U * ROW_CHUNK + 2U * GROUP_COLS * 8U];
        xIo_ = xIoBuf_.Get<float>();
        xSoaR_ = xSoaBuf_.Get<float>();
        xSoaI_ = xSoaR_[ROW_CHUNK + GROUP_COLS];
        pack_ = packBuf_.Get<int32_t>();
        half_ = pack_[2U * OUT_CHUNK];
        odd_ = half_[2U * OUT_CHUNK];
        interIdx_ = interBuf_.Get<int32_t>();
    }

    __aicore__ inline void BuildSweepTables()
    {
        CreateVecIndex(interIdx_, 0, GROUP_COLS * SLOT_STRIDE);
        PipeBarrier<PIPE_V>();
        Muls(interIdx_, interIdx_, static_cast<int32_t>(CPLX_BYTES), GROUP_COLS * SLOT_STRIDE);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0U; j < GROUP_COLS; ++j) {
            Duplicate(StagedCol(j), 0.0F, 2U * SLOT_HEAD);
        }
        PipeBarrier<PIPE_V>();
        floatIdx_ = interIdx_[GROUP_COLS * SLOT_STRIDE];
        CreateVecIndex(floatIdx_, 0, ROW_CHUNK);
        PipeBarrier<PIPE_V>();
        Muls(floatIdx_, floatIdx_, static_cast<int32_t>(sizeof(float)), ROW_CHUNK);
        PipeBarrier<PIPE_V>();
        slotIdx_ = floatIdx_[ROW_CHUNK];
        CreateVecIndex(slotIdx_, 0, 2U * GROUP_COLS);
        PipeBarrier<PIPE_V>();
        Muls(slotIdx_, slotIdx_, static_cast<int32_t>(8U * sizeof(float)), 2U * GROUP_COLS);
        PipeBarrier<PIPE_V>();
        BuildPackTable(pack_, half_, odd_, OUT_CHUNK);
    }

    __aicore__ inline LocalTensor<float> StagedCol(uint32_t j) { return staged_[2U * j * SLOT_STRIDE]; }

    __aicore__ inline LocalTensor<float> DataColumn(uint32_t j) { return staged_[2U * (j * SLOT_STRIDE + SLOT_HEAD)]; }

    __aicore__ inline int64_t BandHead(int64_t col) { return 2 * (col * (plan_.lda - 1) + plan_.ku); }

    __aicore__ inline void LoadGroupColumns(
        int64_t c0, uint32_t g, int64_t r, uint32_t blk, int64_t* t0Arr, uint32_t* lenArr, uint32_t* offArr)
    {
        SyncPipe<HardEvent::V_MTE2>();
        for (uint32_t j = 0U; j < g; ++j) {
            const int64_t c = c0 + j;
            const int64_t t0 = MaxOf(c - plan_.ku, r);
            const int64_t t1 = MinOf(c + plan_.kl + 1, r + blk);
            t0Arr[j] = t0;
            lenArr[j] = t1 > t0 ? static_cast<uint32_t>(t1 - t0) : 0U;
            offArr[j] = static_cast<uint32_t>(t0 - r);
            if (lenArr[j] > 0U && offArr[j] < ROW_CHUNK) {
                CopyComplexIn(DataColumn(j), aGm_, BandHead(c) + 2 * t0, lenArr[j]);
            }
        }
    }

    __aicore__ inline void TransposedGroup(
        int64_t c0, uint32_t g, int64_t r, uint32_t blk, const LocalTensor<float>& accRe,
        const LocalTensor<float>& accIm, int64_t segLo, bool conj)
    {
        int64_t t0Arr[GROUP_COLS];
        uint32_t lenArr[GROUP_COLS];
        uint32_t offArr[GROUP_COLS];
        LoadGroupColumns(c0, g, r, blk, t0Arr, lenArr, offArr);
        SyncPipe<HardEvent::MTE2_V>();
        SplitInterleaved(aSoaR_, aSoaI_, staged_, interIdx_, GROUP_COLS * SLOT_STRIDE);
        PipeBarrier<PIPE_V>();
        Duplicate(partialSlots_, 0.0F, 2U * GROUP_COLS * 8U);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0U; j < g; ++j) {
            const uint32_t len = lenArr[j];
            if (len == 0U) {
                continue;
            }
            const uint32_t off = static_cast<uint32_t>(t0Arr[j] - r);
            Adds(half_, floatIdx_, static_cast<int32_t>(off * sizeof(float)), len);
            PipeBarrier<PIPE_V>();
            Gather(xSlR_, xSoaR_, half_.ReinterpretCast<uint32_t>(), 0U, len);
            Gather(xSlI_, xSoaI_, half_.ReinterpretCast<uint32_t>(), 0U, len);
            PipeBarrier<PIPE_V>();
            const LocalTensor<float> aR = aSoaR_[j * SLOT_STRIDE + SLOT_HEAD];
            const LocalTensor<float> aI = aSoaI_[j * SLOT_STRIDE + SLOT_HEAD];
            Mul(p1_, aR, xSlR_, len);
            Mul(p2_, aI, xSlI_, len);
            Mul(p3_, aR, xSlI_, len);
            Mul(p4_, aI, xSlR_, len);
            PipeBarrier<PIPE_V>();
            if (conj) {
                Add(pr_, p1_, p2_, len);
                Sub(pi_, p3_, p4_, len);
            } else {
                Sub(pr_, p1_, p2_, len);
                Add(pi_, p3_, p4_, len);
            }
            PipeBarrier<PIPE_V>();
            ReduceSum(partialSlots_[j * 8U], pr_, redTmp_, static_cast<int32_t>(len));
            ReduceSum(partialSlots_[(GROUP_COLS + j) * 8U], pi_, redTmp_, static_cast<int32_t>(len));
        }
        Gather(partial_, partialSlots_, slotIdx_.ReinterpretCast<uint32_t>(), 0U, g);
        Gather(
            partial_[GROUP_COLS], partialSlots_, slotIdx_.ReinterpretCast<uint32_t>(), GROUP_COLS * 8U * sizeof(float),
            g);
        PipeBarrier<PIPE_V>();
        const uint32_t accOff = static_cast<uint32_t>(c0 - segLo);
        Add(accRe[accOff], accRe[accOff], partial_, g);
        Add(accIm[accOff], accIm[accOff], partial_[GROUP_COLS], g);
        PipeBarrier<PIPE_V>();
        SyncPipe<HardEvent::V_MTE2>();
    }

    __aicore__ inline void SweepTransposed(
        int64_t segLo, uint32_t seg, const LocalTensor<float>& accRe, const LocalTensor<float>& accIm)
    {
        Duplicate(accRe, 0.0F, seg);
        Duplicate(accIm, 0.0F, seg);
        PipeBarrier<PIPE_V>();
        const int64_t winLo = MaxOf(0, segLo - plan_.ku);
        const int64_t winHi = MinOf(plan_.m, segLo + seg + plan_.kl);
        const bool conj = (plan_.trans == 2U);
        for (int64_t r = winLo; r < winHi; r += ROW_CHUNK) {
            const uint32_t blk = static_cast<uint32_t>(MinOf(ROW_CHUNK, winHi - r));
            CopyComplexIn(xIo_, xGm_, 2 * r, blk);
            SyncPipe<HardEvent::MTE2_V>();
            SplitInterleaved(xSoaR_, xSoaI_, xIo_, interIdx_, blk);
            PipeBarrier<PIPE_V>();
            for (int64_t c0 = segLo; c0 < segLo + seg; c0 += GROUP_COLS) {
                const uint32_t g = static_cast<uint32_t>(MinOf(GROUP_COLS, segLo + seg - c0));
                TransposedGroup(c0, g, r, blk, accRe, accIm, segLo, conj);
            }
        }
    }

    __aicore__ inline void NonFiniteColumn(
        int64_t c0, uint32_t j, uint32_t len, int64_t t0, uint32_t accOff, uint32_t off, float xvr, float xvi,
        const LocalTensor<float>& accRe, const LocalTensor<float>& accIm)
    {
        for (uint32_t q = 0U; q < len; ++q) {
            const int64_t aOff = BandHead(c0 + j) + 2 * (t0 + q);
            const float ar = aGm_.GetValue(aOff);
            const float ai = aGm_.GetValue(aOff + 1);
            const uint32_t at = accOff + off + q;
            accRe.SetValue(at, accRe.GetValue(at) + (ar * xvr - ai * xvi));
            accIm.SetValue(at, accIm.GetValue(at) + (ar * xvi + ai * xvr));
        }
    }

    __aicore__ inline void NormalColumn(
        int64_t c0, uint32_t j, uint32_t len, uint32_t off, uint32_t accOff, float xvr, float xvi,
        const LocalTensor<float>& accRe, const LocalTensor<float>& accIm)
    {
        const uint32_t k8 = off >> 3;
        const uint32_t m8 = off & 7U;
        const uint32_t span = m8 + len;
        Adds(half_, interIdx_, static_cast<int32_t>((SLOT_HEAD - m8) * CPLX_BYTES), span);
        PipeBarrier<PIPE_V>();
        Gather(xSlR_, StagedCol(j), half_.ReinterpretCast<uint32_t>(), 0U, span);
        Gather(xSlI_, StagedCol(j), half_.ReinterpretCast<uint32_t>(), sizeof(float), span);
        PipeBarrier<PIPE_V>();
        Muls(p1_, xSlR_, xvr, span);
        Muls(p2_, xSlI_, xvi, span);
        Muls(p3_, xSlR_, xvi, span);
        Muls(p4_, xSlI_, xvr, span);
        PipeBarrier<PIPE_V>();
        Sub(pr_, p1_, p2_, span);
        Add(pi_, p3_, p4_, span);
        PipeBarrier<PIPE_V>();
        Add(accRe[accOff + 8U * k8], accRe[accOff + 8U * k8], pr_, span);
        Add(accIm[accOff + 8U * k8], accIm[accOff + 8U * k8], pi_, span);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalBlock(
        int64_t c0, uint32_t g, int64_t r, uint32_t blk, int64_t segLo, uint32_t seg, const float* xr, const float* xi,
        const LocalTensor<float>& accRe, const LocalTensor<float>& accIm)
    {
        int64_t t0Arr[GROUP_COLS];
        uint32_t lenArr[GROUP_COLS];
        uint32_t offArr[GROUP_COLS];
        LoadGroupColumns(c0, g, r, blk, t0Arr, lenArr, offArr);
        SyncPipe<HardEvent::MTE2_V>();
        const uint32_t accOff = static_cast<uint32_t>(r - segLo);
        for (uint32_t j = 0U; j < g; ++j) {
            const uint32_t len = lenArr[j];
            if (len == 0U) {
                continue;
            }
            const float xvr = xr[j];
            const float xvi = xi[j];
            const bool nonFinite = (xvr * 0.0F != xvr * 0.0F) || (xvi * 0.0F != xvi * 0.0F) ||
                                   (xvr + xvr == xvr && xvr != 0.0F) || (xvi + xvi == xvi && xvi != 0.0F);
            if (nonFinite) {
                NonFiniteColumn(c0, j, len, t0Arr[j], accOff, offArr[j], xvr, xvi, accRe, accIm);
                continue;
            }
            NormalColumn(c0, j, len, offArr[j], accOff, xvr, xvi, accRe, accIm);
        }
    }

    __aicore__ inline void SweepNormal(
        int64_t segLo, uint32_t seg, const LocalTensor<float>& accRe, const LocalTensor<float>& accIm)
    {
        Duplicate(accRe, 0.0F, seg);
        Duplicate(accIm, 0.0F, seg);
        PipeBarrier<PIPE_V>();
        const int64_t colLo = MaxOf(0, segLo - plan_.kl);
        const int64_t colHi = MinOf(plan_.n, segLo + seg + plan_.ku);
        for (int64_t c0 = colLo; c0 < colHi; c0 += GROUP_COLS) {
            const uint32_t g = static_cast<uint32_t>(MinOf(GROUP_COLS, colHi - c0));
            float xr[GROUP_COLS];
            float xi[GROUP_COLS];
            for (uint32_t j = 0U; j < g; ++j) {
                xr[j] = xGm_.GetValue(2 * (c0 + j));
                xi[j] = xGm_.GetValue(2 * (c0 + j) + 1);
            }
            for (int64_t r = segLo; r < segLo + seg; r += ROW_CHUNK) {
                const uint32_t blk = static_cast<uint32_t>(MinOf(ROW_CHUNK, segLo + seg - r));
                NormalBlock(c0, g, r, blk, segLo, seg, xr, xi, accRe, accIm);
            }
        }
    }

    __aicore__ inline void FinishSegment(
        int64_t segLo, uint32_t seg, const LocalTensor<float>& accRe, const LocalTensor<float>& accIm)
    {
        LocalTensor<float> t1 = staged_;
        LocalTensor<float> t2 = staged_[OUT_CHUNK];
        LocalTensor<float> t3 = staged_[2U * OUT_CHUNK];
        LocalTensor<float> t4 = staged_[3U * OUT_CHUNK];
        Muls(t1, accRe, plan_.alphaR, seg);
        Muls(t2, accIm, plan_.alphaI, seg);
        Muls(t3, accRe, plan_.alphaI, seg);
        Muls(t4, accIm, plan_.alphaR, seg);
        PipeBarrier<PIPE_V>();
        Sub(accRe, t1, t2, seg);
        Add(accIm, t3, t4, seg);
        PipeBarrier<PIPE_V>();
        if (plan_.betaR != 0.0F || plan_.betaI != 0.0F) {
            LocalTensor<float> yIo = yIoBuf_.Get<float>();
            SyncPipe<HardEvent::V_MTE2>();
            CopyComplexIn(yIo, yGm_, 2 * segLo, seg);
            SyncPipe<HardEvent::MTE2_V>();
            SplitInterleaved(aSoaR_, aSoaI_, yIo, interIdx_, seg);
            PipeBarrier<PIPE_V>();
            if (plan_.betaR != 1.0F || plan_.betaI != 0.0F) {
                Muls(t1, aSoaR_, plan_.betaR, seg);
                Muls(t2, aSoaI_, plan_.betaI, seg);
                Muls(t3, aSoaR_, plan_.betaI, seg);
                Muls(t4, aSoaI_, plan_.betaR, seg);
                PipeBarrier<PIPE_V>();
                Sub(aSoaR_, t1, t2, seg);
                Add(aSoaI_, t3, t4, seg);
                PipeBarrier<PIPE_V>();
            }
            Add(accRe, accRe, aSoaR_, seg);
            Add(accIm, accIm, aSoaI_, seg);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> yIo = yIoBuf_.Get<float>();
        Gather(yIo, accRe, pack_.ReinterpretCast<uint32_t>(), 0U, 2U * seg);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopyPad(yGm_[2 * segLo], yIo, DataCopyExtParams{1U, seg * CPLX_BYTES, 0U, 0U, 0U});
        SyncPipe<HardEvent::MTE3_V>();
        SyncPipe<HardEvent::MTE3_MTE2>();
    }

    CgbmvTilingData plan_{};
    GlobalTensor<float> aGm_, xGm_, yGm_;
    TBuf<TPosition::VECCALC> stagedBuf_, soaBuf_, workBuf_, xIoBuf_, xSoaBuf_, accBuf_, yIoBuf_, packBuf_, interBuf_;
    LocalTensor<float> staged_, aSoaR_, aSoaI_;
    LocalTensor<float> p1_, p2_, p3_, p4_, pr_, pi_;
    LocalTensor<float> xIo_, xSoaR_, xSoaI_;
    LocalTensor<int32_t> pack_, half_, odd_, interIdx_, floatIdx_, slotIdx_;
    LocalTensor<float> xSlR_, xSlI_, redTmp_, partial_, partialSlots_;
};
} // namespace CgbmvSweep

namespace {

constexpr uint32_t CGBMV_DOT_CHUNK = 2048U;
constexpr uint32_t CGBMV_RED_SLOT = 16U;
constexpr uint32_t ROW_DOT_WINDOW_LIMIT = 1200U;
constexpr uint32_t ROW_DOT_BATCH2_WINDOW = 845U;
constexpr uint32_t ROW_DOT_BATCH_FLOATS = 3200U;
constexpr uint32_t ROW_DOT_BATCH_CAP = 16U;

__aicore__ inline void SplitInterleaved(
    const LocalTensor<float>& dR, const LocalTensor<float>& dI, const LocalTensor<float>& src, uint32_t n)
{
    const int64_t repeats = static_cast<int64_t>((n * 2U + 63U) / 64U);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(dR.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(src.GetPhyAddr()),
        nullptr, repeats, 1, 1, 8, 8);
    vreducev2(
        reinterpret_cast<__ubuf__ uint32_t*>(dI.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t*>(src.GetPhyAddr()),
        nullptr, repeats, 1, 2, 8, 8);
}

__aicore__ inline uint32_t RowDotIssueBatch(
    const CgbmvTilingData& t, const GlobalTensor<float>& aGM, const LocalTensor<float>& aCplx, uint32_t c0, uint32_t c1,
    uint32_t batch, uint32_t slotC, uint32_t bi, uint32_t* segOut, uint32_t* iBOut)
{
    const uint32_t j0b = c0 + bi * batch;
    const uint32_t cntB = (c1 - j0b < batch) ? (c1 - j0b) : batch;
    const uint32_t baseB = (bi & 1U) * batch * slotC;
    for (uint32_t c = 0; c < cntB; ++c) {
        const uint32_t j = j0b + c;
        const uint32_t lo = (t.ku > j) ? (t.ku - j) : 0U;
        const int64_t hiMax = static_cast<int64_t>(t.ku) + t.kl;
        const int64_t hiLim = static_cast<int64_t>(t.ku) + t.m - 1 - static_cast<int64_t>(j);
        const int64_t hiSigned = (hiLim < hiMax) ? hiLim : hiMax;
        iBOut[c] = j - (t.ku - lo);
        if (hiSigned >= static_cast<int64_t>(lo)) {
            const uint32_t segLen = static_cast<uint32_t>(hiSigned) - lo + 1U;
            const int64_t aOff = 2 * (static_cast<int64_t>(j) * t.lda + lo);
            CopyComplexIn(aCplx[2U * (baseB + c * slotC)], aGM, aOff, segLen);
            segOut[c] = segLen;
        } else {
            segOut[c] = 0;
        }
    }
    return cntB;
}

__aicore__ inline void RowDotColumn(
    const LocalTensor<float>& aR, const LocalTensor<float>& aI, const LocalTensor<float>& xwR,
    const LocalTensor<float>& xwI, const LocalTensor<float>& m1, const LocalTensor<float>& m2,
    const LocalTensor<float>& redDst, const LocalTensor<float>& redTmp, uint32_t base, uint32_t sc, uint32_t xs,
    uint32_t c, bool isConj)
{
    Mul(m1, aR[base], xwR[xs], sc);
    Mul(m2, aI[base], xwI[xs], sc);
    if (isConj) {
        Add(m2, m1, m2, sc);
    } else {
        Sub(m2, m1, m2, sc);
    }
    ReduceSum(redDst[c * 2U * CGBMV_RED_SLOT], m2, redTmp, static_cast<int32_t>(sc));
    Mul(m1, aR[base], xwI[xs], sc);
    Mul(m2, aI[base], xwR[xs], sc);
    if (isConj) {
        Sub(m2, m1, m2, sc);
    } else {
        Add(m2, m1, m2, sc);
    }
    ReduceSum(redDst[(c * 2U + 1U) * CGBMV_RED_SLOT], m2, redTmp, static_cast<int32_t>(sc));
}

__aicore__ inline void DrainRowDotBatch(
    const CgbmvTilingData& t, GlobalTensor<float>& yGM, const LocalTensor<float>& redDst, const uint32_t* segArr,
    uint32_t cnt, uint32_t c0, uint32_t batch, uint32_t b, bool betaZero)
{
    for (uint32_t c = 0; c < cnt; ++c) {
        float dR = 0.0f;
        float dI = 0.0f;
        if (segArr[c] != 0U) {
            const uint32_t rb = c * 2U * CGBMV_RED_SLOT;
            dR = redDst.GetValue(rb);
            dI = redDst.GetValue(rb + CGBMV_RED_SLOT);
        }
        const uint32_t j = c0 + b * batch + c;
        float outR = t.alphaR * dR - t.alphaI * dI;
        float outI = t.alphaR * dI + t.alphaI * dR;
        const int64_t yOff = 2 * StridedElementOffset(j, t.n, t.incy);
        if (!betaZero) {
            const float yr = yGM.GetValue(yOff);
            const float yi = yGM.GetValue(yOff + 1);
            outR += t.betaR * yr - t.betaI * yi;
            outI += t.betaR * yi + t.betaI * yr;
        }
        yGM.SetValue(yOff, outR);
        yGM.SetValue(yOff + 1, outI);
    }
}

struct RowDotCtx {
    uint32_t c0;
    uint32_t c1;
    uint32_t slotC;
    uint32_t iLo;
    uint32_t iHi;
    uint32_t batch;
    uint32_t xwSlot;
    bool isConj;
    bool betaZero;
    bool xwBatched;
    LocalTensor<float> aCplx;
    LocalTensor<float> aR;
    LocalTensor<float> aI;
    LocalTensor<float> m1;
    LocalTensor<float> m2;
    LocalTensor<float> redDst;
    LocalTensor<float> redTmp;
    LocalTensor<float> xwCplx;
    LocalTensor<float> xwR;
    LocalTensor<float> xwI;
    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;
    GlobalTensor<float> yGM;
};

__aicore__ inline void PrefetchXWindow(TPipe& pipe, const CgbmvTilingData& t, const RowDotCtx& q, uint32_t winP)
{
    if (q.iLo > q.iHi) {
        return;
    }
    if (q.xwBatched) {
        for (uint32_t r = 0; r < 8U; ++r) {
            const uint32_t span = static_cast<uint32_t>(MinOf(winP, t.m - q.iLo - r));
            if (span == 0U) {
                break;
            }
            CopyComplexIn(q.xwCplx[2U * r * q.xwSlot], q.xGM, 2 * static_cast<int64_t>(q.iLo + r), span);
        }
        PipeSync<HardEvent::MTE2_V>(pipe);
        SplitInterleaved(q.xwR, q.xwI, q.xwCplx, 8U * q.xwSlot);
        PipeBarrier<PIPE_V>();
        return;
    }
    for (uint32_t r = 0; r < 8U; ++r) {
        const uint32_t span = static_cast<uint32_t>(MinOf(winP, t.m - q.iLo - r));
        if (span == 0U) {
            break;
        }
        CopyComplexIn(q.xwCplx, q.xGM, 2 * static_cast<int64_t>(q.iLo + r), span);
        PipeSync<HardEvent::MTE2_V>(pipe);
        SplitInterleaved(q.xwR[r * q.xwSlot], q.xwI[r * q.xwSlot], q.xwCplx, span);
        PipeSync<HardEvent::V_MTE2>(pipe);
    }
}

__aicore__ inline void RunRowDotBatches(TPipe& pipe, const CgbmvTilingData& t, RowDotCtx& q)
{
    const uint32_t nBatch = (q.c1 - q.c0 + q.batch - 1U) / q.batch;
    uint32_t segArr[16];
    uint32_t iBArr[16];
    uint32_t cnt = RowDotIssueBatch(t, q.aGM, q.aCplx, q.c0, q.c1, q.batch, q.slotC, 0U, segArr, iBArr);
    for (uint32_t b = 0; b < nBatch; ++b) {
        const uint32_t setBase = (b & 1U) * q.batch * q.slotC;
        PipeSync<HardEvent::MTE2_V>(pipe);
        uint32_t segN[16];
        uint32_t iBN[16];
        uint32_t cntN = 0;
        if (b + 1U < nBatch) {
            cntN = RowDotIssueBatch(t, q.aGM, q.aCplx, q.c0, q.c1, q.batch, q.slotC, b + 1U, segN, iBN);
        }
        SplitInterleaved(q.aR, q.aI, q.aCplx[2U * setBase], cnt * q.slotC);
        for (uint32_t c = 0; c < cnt; ++c) {
            if (segArr[c] == 0U) {
                continue;
            }
            const uint32_t base = c * q.slotC;
            const uint32_t xBase = iBArr[c] - q.iLo;
            const uint32_t xs = (xBase & 7U) * q.xwSlot + (xBase >> 3U) * 8U;
            RowDotColumn(q.aR, q.aI, q.xwR, q.xwI, q.m1, q.m2, q.redDst, q.redTmp, base, segArr[c], xs, c, q.isConj);
        }
        PipeSync<HardEvent::V_S>(pipe);
        DrainRowDotBatch(t, q.yGM, q.redDst, segArr, cnt, q.c0, q.batch, b, q.betaZero);
        PipeSync<HardEvent::V_MTE2>(pipe);
        cnt = cntN;
        for (uint32_t c = 0; c < 16U; ++c) {
            segArr[c] = segN[c];
            iBArr[c] = iBN[c];
        }
    }
}

__aicore__ inline void PrepareRowDotCtx(
    TPipe& pipe, const CgbmvTilingData& t, RowDotCtx& q, uint32_t c0, uint32_t c1, uint32_t slotC, uint32_t iLo,
    uint32_t winP)
{
    const uint32_t win = q.iHi - iLo + 1U;
    const uint32_t slotGuarded = (slotC == 0U) ? 1U : slotC;
    const uint32_t batchBudget = ROW_DOT_BATCH_FLOATS / slotGuarded;
    uint32_t batch = (batchBudget < ROW_DOT_BATCH_CAP) ? ((batchBudget < 1U) ? 1U : batchBudget) : ROW_DOT_BATCH_CAP;
    if (win > ROW_DOT_BATCH2_WINDOW && batch > 2U) {
        batch = 2U;
    }
    q.c0 = c0;
    q.c1 = c1;
    q.slotC = slotGuarded;
    q.iLo = iLo;
    q.batch = batch;
    q.isConj = (t.trans == 2U);
    q.betaZero = (t.betaZero != 0U);
    q.xwSlot = ((winP + 63U) / 64U) * 64U;
    q.xwBatched = winP <= 384U;
    const uint32_t aSlotTotal = batch * slotGuarded;
    TBuf<TPosition::VECCALC> aCplxBuf;
    TBuf<TPosition::VECCALC> aRBuf;
    TBuf<TPosition::VECCALC> aIBuf;
    TBuf<TPosition::VECCALC> m1Buf;
    TBuf<TPosition::VECCALC> m2Buf;
    TBuf<TPosition::VECCALC> redBuf;
    TBuf<TPosition::VECCALC> redTmpBuf;
    TBuf<TPosition::VECCALC> xwCplxBuf;
    TBuf<TPosition::VECCALC> xwRBuf;
    TBuf<TPosition::VECCALC> xwIBuf;
    pipe.InitBuffer(aCplxBuf, (2U * CgbmvMath::RoundUpTo(2 * aSlotTotal, 64U) + 128U) * sizeof(float));
    pipe.InitBuffer(aRBuf, (CgbmvMath::RoundUpTo(aSlotTotal, 64U) + 64U) * sizeof(float));
    pipe.InitBuffer(aIBuf, (CgbmvMath::RoundUpTo(aSlotTotal, 64U) + 64U) * sizeof(float));
    pipe.InitBuffer(m1Buf, (CgbmvMath::RoundUpTo(slotGuarded, 64U) + 64U) * sizeof(float));
    pipe.InitBuffer(m2Buf, (CgbmvMath::RoundUpTo(slotGuarded, 64U) + 64U) * sizeof(float));
    pipe.InitBuffer(redBuf, (2U * batch * CGBMV_RED_SLOT + 64U) * sizeof(float));
    pipe.InitBuffer(redTmpBuf, (CgbmvMath::RoundUpTo(CGBMV_DOT_CHUNK, 64U) + 128U) * sizeof(float));
    pipe.InitBuffer(xwCplxBuf, (q.xwBatched ? (16U * q.xwSlot) : CgbmvMath::RoundUpTo(2U * winP, 64U)) * sizeof(float));
    pipe.InitBuffer(xwRBuf, (8U * q.xwSlot + 64U) * sizeof(float));
    pipe.InitBuffer(xwIBuf, (8U * q.xwSlot + 64U) * sizeof(float));
    q.aCplx = aCplxBuf.Get<float>();
    q.aR = aRBuf.Get<float>();
    q.aI = aIBuf.Get<float>();
    q.m1 = m1Buf.Get<float>();
    q.m2 = m2Buf.Get<float>();
    q.redDst = redBuf.Get<float>();
    q.redTmp = redTmpBuf.Get<float>();
    q.xwCplx = xwCplxBuf.Get<float>();
    q.xwR = xwRBuf.Get<float>();
    q.xwI = xwIBuf.Get<float>();
    BindOperandTensors(t, q.aGM, q.xGM, q.yGM, t.m, t.n);
}

__aicore__ inline void RowDotPath(const CgbmvTilingData& t)
{
    const uint32_t p = GetBlockIdx();
    const uint32_t c0 = p * t.outPerCore;
    if (c0 >= t.n) {
        return;
    }
    const uint32_t c1 = (t.n < c0 + t.outPerCore) ? t.n : (c0 + t.outPerCore);
    const uint32_t bandW = t.kl + t.ku + 1U;
    const uint32_t slotC = ((bandW + 7U) / 8U) * 8U;
    if (slotC == 0U) {
        return;
    }
    const uint32_t iLo = (c0 > t.ku) ? (c0 - t.ku) : 0U;
    const uint32_t iHi = ((t.m - 1U) < (c1 - 1U + t.kl)) ? (t.m - 1U) : (c1 - 1U + t.kl);
    const uint32_t win = (iLo <= iHi) ? (iHi - iLo + 1U) : 1U;
    const uint32_t winP = ((win + 8U + 31U) / 32U) * 32U;
    if ((t.incx != 1) || (bandW > CGBMV_DOT_CHUNK) || (win > ROW_DOT_WINDOW_LIMIT)) {
        return;
    }
    TPipe pipe;
    RowDotCtx q;
    q.iHi = ((t.m - 1U) < (c1 - 1U + t.kl)) ? (t.m - 1U) : (c1 - 1U + t.kl);
    PrepareRowDotCtx(pipe, t, q, c0, c1, slotC, iLo, winP);
    PrefetchXWindow(pipe, t, q, winP);
    RunRowDotBatches(pipe, t, q);
}

} // namespace

#define CGBMV_TRY(TYPE)          \
    if (TYPE::Matches(tiling)) { \
        TYPE op;                 \
        op.Init(tiling, &pipe);  \
        op.Run();                \
    } else

extern "C" __global__ __aicore__ void cgbmv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, const CgbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)a;
    (void)x;
    (void)y;
    TPipe pipe;
    CGBMV_TRY(CgbmvRowTiles::RowTilesKernel<64>)
    CGBMV_TRY(CgbmvRowTiles::RowTilesKernel<48>)
    CGBMV_TRY(CgbmvRowTiles::RowTilesKernel<32>)
    CGBMV_TRY(CgbmvRowTiles::RowTilesKernel<8>)
    CGBMV_TRY(CgbmvSweep::SweepKernel)
    {
        CgbmvGeneral::GeneralKernel op;
        op.Init(tiling, &pipe);
        op.Run();
    }
}
#undef CGBMV_TRY

extern "C" __global__ __aicore__
    __vector__ void cgbmv_kernel_vec(GM_ADDR a, GM_ADDR x, GM_ADDR y, const CgbmvTilingData tiling)
{
    (void)a;
    (void)x;
    (void)y;
    RowDotPath(tiling);
}

void cgbmv_kernel_do(GM_ADDR a, GM_ADDR x, GM_ADDR y, const CgbmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    if (tiling.panel != 0U || tiling.path == CGBMV_PATH_TINY || tiling.path == CGBMV_PATH_BETA_SCALE) {
        cgbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, tiling);
    } else {
        cgbmv_kernel_vec<<<numBlocks, nullptr, stream>>>(a, x, y, tiling);
    }
}
