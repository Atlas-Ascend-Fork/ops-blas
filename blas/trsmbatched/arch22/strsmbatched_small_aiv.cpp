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
#include "strsmbatched_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t UB_TILE_STRIDE = 64U;
constexpr uint32_t UB_TILE_ELEMS = UB_TILE_STRIDE * UB_TILE_STRIDE;
constexpr uint32_t FLOAT_BLOCK_ELEMS = 8U;

__aicore__ inline uint32_t CeilDivU(uint32_t a, uint32_t b)
{
    if (b == 0U) {
        return 0U;
    }
    return (a + b - 1U) / b;
}

__aicore__ inline bool IsForwardSweep(const StrsmbatchedTilingData& t)
{
    if (t.side == 0U) {
        return (t.uplo == 0U) ? (t.trans != 0U) : (t.trans == 0U);
    }
    return (t.uplo == 0U) ? (t.trans == 0U) : (t.trans != 0U);
}

__aicore__ inline void SyncMte2ToVOnly()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

__aicore__ inline void SyncVToMte3()
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

__aicore__ inline void SyncMte3ToNext()
{
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <uint32_t S>
__aicore__ inline void InitBaseLaneValues(
    LocalTensor<uint32_t>& packIdx, LocalTensor<uint32_t>& unpackIdx, LocalTensor<uint32_t>& aDiagIdx,
    LocalTensor<int32_t>& aCol0Int, LocalTensor<int32_t>& baseLaneInt)
{
    constexpr uint32_t matElems = S * S;
    constexpr uint32_t blocksPerBatch = S / FLOAT_BLOCK_ELEMS;
#pragma clang loop unroll(disable)
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        const uint32_t g0 = lane / blocksPerBatch;
        const uint32_t matBase = g0 * matElems * static_cast<uint32_t>(sizeof(float));
        packIdx.SetValue(lane, lane * S * static_cast<uint32_t>(sizeof(float)));
        unpackIdx.SetValue(lane, lane * 64U * static_cast<uint32_t>(sizeof(float)));
        aDiagIdx.SetValue(lane, matBase);
        aCol0Int.SetValue(lane, static_cast<int32_t>(matBase));
        baseLaneInt.SetValue(lane, static_cast<int32_t>(lane * static_cast<uint32_t>(sizeof(float))));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
}

template <uint32_t S>
__aicore__ inline void ExpandStep8LaneIndices(
    LocalTensor<int32_t>& packInt, LocalTensor<int32_t>& unpackInt, LocalTensor<int32_t>& aDiagInt,
    LocalTensor<int32_t>& aCol0Int, LocalTensor<int32_t>& baseLaneInt, uint32_t jStride, uint32_t stepLim)
{
#pragma clang loop unroll(disable)
    for (uint32_t step = 1U; step < stepLim; step <<= 1U) {
        const uint64_t msk = static_cast<uint64_t>(step * 8U);
        const int32_t unpackDelta = (step * 8U < S) ? static_cast<int32_t>(step * 8U * 64U * sizeof(float)) :
                                                      static_cast<int32_t>(((step * 8U) / S) * sizeof(float));
        Adds(
            packInt[step * 8U], packInt, static_cast<int32_t>(step * 8U * S * sizeof(float)), msk, 1U,
            UnaryRepeatParams(1, 1, 8, 8));
        Adds(unpackInt[step * 8U], unpackInt, unpackDelta, msk, 1U, UnaryRepeatParams(1, 1, 8, 8));
        Adds(
            aDiagInt[step * 8U], aDiagInt, static_cast<int32_t>(step * (S + 1U) * sizeof(float)), msk, 1U,
            UnaryRepeatParams(1, 1, 8, 8));
        Adds(
            aCol0Int[step * 8U], aCol0Int, static_cast<int32_t>(step * jStride * sizeof(float)), msk, 1U,
            UnaryRepeatParams(1, 1, 8, 8));
        Adds(baseLaneInt[step * 8U], baseLaneInt, 0, msk, 1U, UnaryRepeatParams(1, 1, 8, 8));
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t S>
__aicore__ inline void ExpandStepSIndices(
    LocalTensor<int32_t>& packInt, LocalTensor<int32_t>& unpackInt, LocalTensor<int32_t>& aDiagInt,
    LocalTensor<int32_t>& aCol0Int, LocalTensor<int32_t>& baseLaneInt, uint32_t jStride, uint32_t stepLim)
{
    constexpr uint32_t G = 64U / S;
#pragma clang loop unroll(disable)
    for (uint32_t step = 1U; step < stepLim; step <<= 1U) {
        Adds(
            packInt[step * 64U], packInt, static_cast<int32_t>(step * sizeof(float)), 64ULL, static_cast<uint8_t>(step),
            UnaryRepeatParams(1, 1, 8, 8));
        Adds(
            unpackInt[step * 64U], unpackInt, static_cast<int32_t>(step * G * sizeof(float)), 64ULL,
            static_cast<uint8_t>(step), UnaryRepeatParams(1, 1, 8, 8));
        if constexpr (S > 8U) {
            if (step >= 8U) {
                const uint8_t rep8 = static_cast<uint8_t>(step / 8U);
                Adds(
                    aDiagInt[step * 8U], aDiagInt, static_cast<int32_t>(step * (S + 1U) * sizeof(float)), 64ULL, rep8,
                    UnaryRepeatParams(1, 1, 8, 8));
                Adds(
                    aCol0Int[step * 8U], aCol0Int, static_cast<int32_t>(step * jStride * sizeof(float)), 64ULL, rep8,
                    UnaryRepeatParams(1, 1, 8, 8));
                Adds(baseLaneInt[step * 8U], baseLaneInt, 0, 64ULL, rep8, UnaryRepeatParams(1, 1, 8, 8));
            }
        }
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void ExpandStepPairOffset(
    LocalTensor<int32_t>& aPackInt, LocalTensor<int32_t>& aDiagPackInt, const LocalTensor<int32_t>& aCol0Int,
    const LocalTensor<int32_t>& baseLaneInt, uint32_t pairOff, uint32_t jStart, uint32_t elemCnt, int32_t colByteOff,
    int32_t diagByteOff)
{
    if (elemCnt >= 64U) {
        const uint8_t fullRep = static_cast<uint8_t>(elemCnt / 64U);
        const uint32_t fullCnt = static_cast<uint32_t>(fullRep) * 64U;
        Adds(aPackInt[pairOff * 8U], aCol0Int[jStart * 8U], colByteOff, 64ULL, fullRep, UnaryRepeatParams(1, 1, 8, 8));
        Adds(aDiagPackInt[pairOff * 8U], baseLaneInt, diagByteOff, 64ULL, fullRep, UnaryRepeatParams(1, 1, 8, 8));
        if (elemCnt > fullCnt) {
            Adds(
                aPackInt[pairOff * 8U + fullCnt], aCol0Int[jStart * 8U + fullCnt], colByteOff,
                static_cast<uint64_t>(elemCnt - fullCnt), 1U, UnaryRepeatParams(1, 1, 8, 8));
            Adds(
                aDiagPackInt[pairOff * 8U + fullCnt], baseLaneInt[fullCnt], diagByteOff,
                static_cast<uint64_t>(elemCnt - fullCnt), 1U, UnaryRepeatParams(1, 1, 8, 8));
        }
    } else {
        Adds(
            aPackInt[pairOff * 8U], aCol0Int[jStart * 8U], colByteOff, static_cast<uint64_t>(elemCnt), 1U,
            UnaryRepeatParams(1, 1, 8, 8));
        Adds(
            aDiagPackInt[pairOff * 8U], baseLaneInt, diagByteOff, static_cast<uint64_t>(elemCnt), 1U,
            UnaryRepeatParams(1, 1, 8, 8));
    }
}

template <uint32_t S>
__aicore__ inline void ExpandPairOffsetIndices(
    LocalTensor<int32_t>& aPackInt, LocalTensor<int32_t>& aDiagPackInt, LocalTensor<int32_t>& aCol0Int,
    LocalTensor<int32_t>& baseLaneInt, uint32_t kStride, bool forward)
{
    uint32_t pairOff = 0U;
#pragma clang loop unroll(disable)
    for (uint32_t s = 0U; s + 1U < S; ++s) {
        const uint32_t k = forward ? s : (S - 1U - s);
        const uint32_t jStart = forward ? (k + 1U) : 0U;
        const uint32_t rem = S - 1U - s;
        const int32_t colByteOff = static_cast<int32_t>(k * kStride * sizeof(float));
        const int32_t diagByteOff = static_cast<int32_t>(k * 8U * sizeof(float));
        ExpandStepPairOffset(
            aPackInt, aDiagPackInt, aCol0Int, baseLaneInt, pairOff, jStart, rem * 8U, colByteOff, diagByteOff);
        pairOff += rem;
    }
    if constexpr (S == 16U) {
        Duplicate(aPackInt[120U * 8U], 0, 64ULL, 1U, 1U, 8U);
        Duplicate(aDiagPackInt[120U * 8U], 0, 64ULL, 1U, 1U, 8U);
    } else if constexpr (S == 32U) {
        Duplicate(aPackInt[496U * 8U], 0, 64ULL, 2U, 1U, 8U);
        Duplicate(aDiagPackInt[496U * 8U], 0, 64ULL, 2U, 1U, 8U);
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t S>
__aicore__ inline void InitMultiBatchIdxUb(
    LocalTensor<uint32_t>& packIdx, LocalTensor<uint32_t>& unpackIdx, LocalTensor<uint32_t>& aDiagIdx,
    LocalTensor<uint32_t>& aPackIdx, LocalTensor<uint32_t>& aDiagPackIdx, LocalTensor<float>& onesUb,
    LocalTensor<float>& scratchUb, bool aColMajorStep, bool forward)
{
    const uint32_t jStride = aColMajorStep ? 1U : S;
    const uint32_t kStride = aColMajorStep ? S : 1U;
    LocalTensor<int32_t> aCol0Int = scratchUb.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> baseLaneInt = scratchUb.ReinterpretCast<int32_t>()[256U];
    Duplicate(onesUb, 1.0f, 64ULL, 1U, 1U, 8U);
    InitBaseLaneValues<S>(packIdx, unpackIdx, aDiagIdx, aCol0Int, baseLaneInt);
    LocalTensor<int32_t> packInt = packIdx.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> unpackInt = unpackIdx.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> aDiagInt = aDiagIdx.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> aPackInt = aPackIdx.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> aDiagPackInt = aDiagPackIdx.ReinterpretCast<int32_t>();
    ExpandStep8LaneIndices<S>(packInt, unpackInt, aDiagInt, aCol0Int, baseLaneInt, jStride, 8U);
    ExpandStepSIndices<S>(packInt, unpackInt, aDiagInt, aCol0Int, baseLaneInt, jStride, S);
    ExpandPairOffsetIndices<S>(aPackInt, aDiagPackInt, aCol0Int, baseLaneInt, kStride, forward);
}

template <uint32_t S>
__aicore__ __attribute__((noinline)) void LoadMultiBatchNonContigUb(
    LocalTensor<float>& dstUb, __gm__ const uint64_t* ptrsGm, uint32_t count, uint32_t padCount,
    __gm__ float* cachedPtrs[32])
{
    constexpr uint32_t matElems = S * S;
#pragma clang loop unroll(disable)
    for (uint32_t m = 0U; m < count; ++m) {
        __gm__ float* p = reinterpret_cast<__gm__ float*>(ptrsGm[m]);
        cachedPtrs[m] = p;
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer(p);
        DataCopy(dstUb[m * matElems], gm, matElems);
    }
    if (padCount > count && count > 0U) {
        GlobalTensor<float> gm0;
        gm0.SetGlobalBuffer(cachedPtrs[0]);
#pragma clang loop unroll(disable)
        for (uint32_t m = count; m < padCount; ++m) {
            DataCopy(dstUb[m * matElems], gm0, matElems);
        }
    }
}

template <uint32_t S>
__aicore__ inline void LoadMultiBatchSpanUb(
    LocalTensor<float>& dstUb, GM_ADDR ptrArray, uint32_t bBase, uint32_t count, uint32_t padCount,
    __gm__ float* cachedPtrs[32])
{
    constexpr uint32_t matElems = S * S;
    constexpr uint64_t matBytes = static_cast<uint64_t>(matElems) * sizeof(float);
    __gm__ const uint64_t* ptrsGm = reinterpret_cast<__gm__ const uint64_t*>(ptrArray) + bBase;
    const uint64_t p0 = ptrsGm[0];
    cachedPtrs[0] = reinterpret_cast<__gm__ float*>(p0);
    if (count > 1U && (ptrsGm[1U] - p0 == matBytes)) {
        cachedPtrs[1] = nullptr;
        GlobalTensor<float> gm0;
        gm0.SetGlobalBuffer(cachedPtrs[0]);
        DataCopy(dstUb, gm0, count * matElems);
        if (padCount > count) {
            DataCopy(dstUb[count * matElems], gm0, (padCount - count) * matElems);
        }
        return;
    }
    LoadMultiBatchNonContigUb<S>(dstUb, ptrsGm, count, padCount, cachedPtrs);
}

template <uint32_t S>
__aicore__ __attribute__((noinline)) void StoreMultiBatchNonContigUb(
    __gm__ float* const cachedPtrs[32], const LocalTensor<float>& srcUb, uint32_t count)
{
    constexpr uint32_t matElems = S * S;
#pragma clang loop unroll(disable)
    for (uint32_t m = 0U; m < count; ++m) {
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer(cachedPtrs[m]);
        DataCopy(gm, srcUb[m * matElems], matElems);
    }
}

template <uint32_t S>
__aicore__ inline void StoreMultiBatchSpanUb(
    __gm__ float* const cachedPtrs[32], const LocalTensor<float>& srcUb, uint32_t count)
{
    constexpr uint32_t matElems = S * S;
    if (count > 1U && cachedPtrs[1] == nullptr) {
        GlobalTensor<float> gm0;
        gm0.SetGlobalBuffer(cachedPtrs[0]);
        DataCopy(gm0, srcUb, count * matElems);
        return;
    }
    StoreMultiBatchNonContigUb<S>(cachedPtrs, srcUb, count);
}

__aicore__ inline void InitMultiBatchIdxUb8(
    LocalTensor<uint32_t>& packIdx, LocalTensor<uint32_t>& unpackIdx, LocalTensor<uint32_t>& aDiagIdx,
    LocalTensor<uint32_t>& aPackIdx, LocalTensor<uint32_t>& aDiagPackIdx, LocalTensor<float>& onesUb,
    LocalTensor<float>& scratchUb, bool aColMajorStep, bool forward, bool fullInit)
{
    const uint32_t jStride = aColMajorStep ? 1U : 8U;
    const uint32_t kStride = aColMajorStep ? 8U : 1U;
    LocalTensor<int32_t> aCol0Int = scratchUb.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> baseLaneInt = scratchUb.ReinterpretCast<int32_t>()[256U];
    LocalTensor<int32_t> packInt = packIdx.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> unpackInt = unpackIdx.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> aDiagInt = aDiagIdx.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> aPackInt = aPackIdx.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> aDiagPackInt = aDiagPackIdx.ReinterpretCast<int32_t>();
    Duplicate(onesUb, 1.0f, 64ULL, 1U, 1U, 8U);
    if (fullInit) {
        InitBaseLaneValues<8U>(packIdx, unpackIdx, aDiagIdx, aCol0Int, baseLaneInt);
    } else {
        Duplicate(packInt, 0, 64ULL, 28U, 1U, 8U);
        Duplicate(aCol0Int, 0, 64ULL, 8U, 1U, 8U);
        PipeBarrier<PIPE_V>();
    }
    const uint32_t stepLim = fullInit ? 8U : 2U;
    ExpandStep8LaneIndices<8U>(packInt, unpackInt, aDiagInt, aCol0Int, baseLaneInt, jStride, stepLim);
    ExpandStepSIndices<8U>(packInt, unpackInt, aDiagInt, aCol0Int, baseLaneInt, jStride, stepLim);
    const uint32_t sLim = fullInit ? 7U : 1U;
    uint32_t pairOff = 0U;
#pragma clang loop unroll(disable)
    for (uint32_t s = 0U; s < sLim; ++s) {
        const uint32_t k = forward ? s : (7U - s);
        const uint32_t jStart = forward ? (k + 1U) : 0U;
        const uint32_t rem = 7U - s;
        const uint64_t elemCnt = static_cast<uint64_t>(rem * 8U);
        const int32_t colByteOff = static_cast<int32_t>(k * kStride * sizeof(float));
        const int32_t diagByteOff = static_cast<int32_t>(k * 8U * sizeof(float));
        Adds(aPackInt[pairOff * 8U], aCol0Int[jStart * 8U], colByteOff, elemCnt, 1U, UnaryRepeatParams(1, 1, 8, 8));
        Adds(aDiagPackInt[pairOff * 8U], baseLaneInt, diagByteOff, elemCnt, 1U, UnaryRepeatParams(1, 1, 8, 8));
        pairOff += rem;
    }
    Duplicate(aPackInt[28U * 8U], 0, 32ULL, 1U, 1U, 8U);
    Duplicate(aDiagPackInt[28U * 8U], 0, 32ULL, 1U, 1U, 8U);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline bool ComputeBatchRange8(
    const StrsmbatchedTilingData& t, uint32_t& bStart, uint32_t& bEnd, bool& doStore)
{
    const uint32_t aivNum = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());
    const uint32_t totalGroups = CeilDivU(t.batchCount, 8U);
    if (aivNum == 0U || totalGroups == 0U) {
        return false;
    }
    const uint32_t aivIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t activeLimit = (totalGroups <= 8U) ? totalGroups : CeilDivU(totalGroups, 4U);
    const uint32_t effAiv = (aivNum < activeLimit) ? aivNum : activeLimit;
    const uint32_t baseGroups = (aivIdx < effAiv) ? (totalGroups / effAiv) : 0U;
    const uint32_t remGroups = totalGroups % effAiv;
    const uint32_t grpStart =
        (aivIdx < effAiv) ? (aivIdx * baseGroups + ((aivIdx < remGroups) ? aivIdx : remGroups)) : 0U;
    const uint32_t grpEnd = (aivIdx < effAiv) ? (grpStart + baseGroups + ((aivIdx < remGroups) ? 1U : 0U)) : 0U;
    doStore = (grpStart < grpEnd);
    bStart = doStore ? (grpStart * 8U) : 0U;
    const uint32_t bEndLim = doStore ? (grpEnd * 8U) : 8U;
    bEnd = (bEndLim < t.batchCount) ? bEndLim : t.batchCount;
    return (bStart < bEnd);
}

__aicore__ inline void InvertDiag8(
    LocalTensor<float>& diagUb, LocalTensor<float>& onesUb, LocalTensor<float>& diagBrcbFullUb,
    LocalTensor<float>& brcbUb, LocalTensor<float>& bUb, LocalTensor<uint32_t>& aDiagPackIdx, uint32_t V)
{
    Div(diagUb, onesUb, diagUb, 64U, static_cast<uint8_t>(V), BinaryRepeatParams(1, 1, 1, 8, 0, 8));
    PipeBarrier<PIPE_V>();
    Brcb(diagBrcbFullUb, diagUb, static_cast<uint8_t>(V * 8U), BrcbRepeatParams(1, 8));
#pragma clang loop unroll(disable)
    for (uint32_t v = 0U; v < V; ++v) {
        Gather(brcbUb[v * 256U], diagUb, aDiagPackIdx, v * 64U * static_cast<uint32_t>(sizeof(float)), 64U, 4U, 8U);
    }
    PipeBarrier<PIPE_V>();
    Mul(bUb, bUb, brcbUb, 64U, static_cast<uint8_t>(V * 4U), BinaryRepeatParams(1, 1, 1, 8, 8, 8));
}

__aicore__ inline void EliminateMultiBatch8(
    LocalTensor<float>& brcbUb, LocalTensor<float>& bWorkUb, uint32_t V, uint32_t sLim, bool forward)
{
    uint32_t pairOff = 0U;
#pragma clang loop unroll(disable)
    for (uint32_t s = 0U; s < sLim; ++s) {
        const uint32_t k = forward ? s : (7U - s);
        const uint32_t jStart = forward ? (k + 1U) : 0U;
        const uint8_t rem = static_cast<uint8_t>(7U - s);
#pragma clang loop unroll(disable)
        for (uint32_t v = 0U; v < V; ++v) {
            LocalTensor<float> curBrcb = brcbUb[(v * 32U + pairOff) * 64U];
            Mul(curBrcb, curBrcb, bWorkUb[v * 512U + k * 64U], 64U, rem, BinaryRepeatParams(1, 1, 1, 8, 8, 0));
        }
        PipeBarrier<PIPE_V>();
#pragma clang loop unroll(disable)
        for (uint32_t v = 0U; v < V; ++v) {
            LocalTensor<float> curBrcb = brcbUb[(v * 32U + pairOff) * 64U];
            Sub(bWorkUb[v * 512U + jStart * 64U], bWorkUb[v * 512U + jStart * 64U], curBrcb, 64U, rem,
                BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        }
        PipeBarrier<PIPE_V>();
        pairOff += rem;
    }
}

struct SmallMultiBatch8Tensors {
    LocalTensor<float> bUb;
    LocalTensor<float> bWorkUb;
    LocalTensor<float> aCompactUb;
    LocalTensor<float> diagBrcbFullUb;
    LocalTensor<float> brcbUb;
    LocalTensor<float> diagUb;
    LocalTensor<float> onesUb;
    LocalTensor<uint32_t> packIdx;
    LocalTensor<uint32_t> unpackIdx;
    LocalTensor<uint32_t> aDiagIdx;
    LocalTensor<uint32_t> aPackIdx;
    LocalTensor<uint32_t> aDiagPackIdx;
};

__aicore__ inline void InitSmallMultiBatch8Tensors(
    TPipe& pipe, TBuf<TPosition::VECCALC>& floatBuf, TBuf<TPosition::VECCALC>& idxBuf, SmallMultiBatch8Tensors& ub)
{
    constexpr uint32_t floatElems = 5U * UB_TILE_ELEMS + 512U;
    constexpr uint32_t idxElems = 512U + 512U + 256U + 256U + 256U;
    pipe.InitBuffer(floatBuf, floatElems * sizeof(float));
    pipe.InitBuffer(idxBuf, idxElems * sizeof(uint32_t));
    LocalTensor<float> fBase = floatBuf.Get<float>();
    ub.bUb = fBase;
    ub.bWorkUb = fBase[2048U];
    ub.aCompactUb = fBase[UB_TILE_ELEMS];
    ub.diagBrcbFullUb = fBase[UB_TILE_ELEMS + 2048U];
    ub.brcbUb = fBase[2U * UB_TILE_ELEMS];
    ub.diagUb = fBase[4U * UB_TILE_ELEMS];
    ub.onesUb = fBase[4U * UB_TILE_ELEMS + 256U];
    LocalTensor<uint32_t> iBase = idxBuf.Get<uint32_t>();
    ub.packIdx = iBase;
    ub.unpackIdx = iBase[512U];
    ub.aDiagIdx = iBase[1024U];
    ub.aPackIdx = iBase[1280U];
    ub.aDiagPackIdx = iBase[1536U];
}

__aicore__ inline void ProcessBatch32Ub8(
    SmallMultiBatch8Tensors& ub, __gm__ float* bPtrs[32], uint32_t V, uint32_t curBatches, uint32_t sLim, bool forward,
    const StrsmbatchedTilingData& t, bool doStore, uint32_t bBase, uint32_t bEnd)
{
#pragma clang loop unroll(disable)
    for (uint32_t v = 0U; v < V; ++v) {
        const uint32_t vOffBytes = v * 512U * static_cast<uint32_t>(sizeof(float));
        Gather(ub.bWorkUb[v * 512U], ub.bUb, ub.packIdx, vOffBytes, 64U, 8U, 8U);
        Gather(ub.bUb[v * 256U], ub.aCompactUb, ub.aPackIdx, vOffBytes, 64U, 4U, 8U);
        if (t.diag == 0U) {
            Gather(ub.diagUb[v * 64U], ub.aCompactUb, ub.aDiagIdx, vOffBytes, 64U, 1U, 8U);
        }
    }
    PipeBarrier<PIPE_V>();
    if (t.alpha != 1.0f) {
        Muls(ub.bWorkUb, ub.bWorkUb, t.alpha, 64ULL, static_cast<uint8_t>(V * 8U), UnaryRepeatParams(1, 1, 8, 8));
    }
    if (t.diag == 0U) {
        InvertDiag8(ub.diagUb, ub.onesUb, ub.diagBrcbFullUb, ub.brcbUb, ub.bUb, ub.aDiagPackIdx, V);
    }
    PipeBarrier<PIPE_V>();
    Brcb(ub.brcbUb, ub.bUb, static_cast<uint8_t>(V * 32U), BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    EliminateMultiBatch8(ub.brcbUb, ub.bWorkUb, V, sLim, forward);
    if (t.diag == 0U) {
        Mul(ub.bWorkUb, ub.bWorkUb, ub.diagBrcbFullUb, 64U, static_cast<uint8_t>(V * 8U),
            BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        PipeBarrier<PIPE_V>();
    }
#pragma clang loop unroll(disable)
    for (uint32_t v = 0U; v < V; ++v) {
        Gather(
            ub.bUb[v * 512U], ub.bWorkUb, ub.unpackIdx, v * 512U * static_cast<uint32_t>(sizeof(float)), 64U, 8U, 8U);
    }
    PipeBarrier<PIPE_V>();
    if (doStore) {
        SyncVToMte3();
        if (bBase + 32U < bEnd) {
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        StoreMultiBatchSpanUb<8U>(bPtrs, ub.bUb, curBatches);
    }
}

__aicore__ __attribute__((noinline)) void SolveSmallMultiBatchAiv8(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, const StrsmbatchedTilingData& t)
{
    uint32_t bStart = 0U;
    uint32_t bEnd = 0U;
    bool doStore = false;
    if (!ComputeBatchRange8(t, bStart, bEnd, doStore)) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> floatBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    SmallMultiBatch8Tensors ub;
    InitSmallMultiBatch8Tensors(pipe, floatBuf, idxBuf, ub);
    const bool aColMajorStep = (t.trans == 0U);
    const bool forward = IsForwardSweep(t);
    const uint32_t sLim = doStore ? 7U : 1U;
    __gm__ float* aPtrs[32];
    __gm__ float* bPtrs[32];
#pragma clang loop unroll(disable)
    for (uint32_t bBase = bStart; bBase < bEnd; bBase += 32U) {
        const uint32_t curBatches = (bBase + 32U <= bEnd) ? 32U : (bEnd - bBase);
        const uint32_t V = CeilDivU(curBatches, 8U);
        if (doStore) {
            LoadMultiBatchSpanUb<8U>(ub.aCompactUb, aPtrArray, bBase, curBatches, V * 8U, aPtrs);
            if (bBase > bStart) {
                SyncMte3ToNext();
            }
            LoadMultiBatchSpanUb<8U>(ub.bUb, bPtrArray, bBase, curBatches, curBatches, bPtrs);
        }
        if (bBase == bStart) {
            InitMultiBatchIdxUb8(
                ub.packIdx, ub.unpackIdx, ub.aDiagIdx, ub.aPackIdx, ub.aDiagPackIdx, ub.onesUb, ub.bWorkUb,
                aColMajorStep, forward, doStore);
        }
        if (doStore) {
            SyncMte2ToVOnly();
        }
        ProcessBatch32Ub8(ub, bPtrs, V, curBatches, sLim, forward, t, doStore, bBase, bEnd);
    }
    if (doStore) {
        SyncMte3ToNext();
    }
}

template <uint32_t S>
__aicore__ inline uint32_t GetChunkEnd(uint32_t sStart)
{
    if constexpr (S == 16U) {
        return (sStart == 0U) ? 6U : 15U;
    } else {
        if (sStart == 0U) {
            return 3U;
        } else if (sStart == 3U) {
            return 7U;
        } else if (sStart == 7U) {
            return 12U;
        } else if (sStart == 12U) {
            return 19U;
        } else {
            return 31U;
        }
    }
}

template <uint32_t S>
__aicore__ inline void PrepareChunkBrcbMultiBatch(
    LocalTensor<float>& stageUb, LocalTensor<float>& brcbUb, LocalTensor<float>& aCompactUb, LocalTensor<float>& diagUb,
    LocalTensor<uint32_t>& aPackIdx, LocalTensor<uint32_t>& aDiagPackIdx, uint32_t V, uint32_t chunkPairBase,
    uint32_t groupElems, uint32_t diag, uint8_t chunkRepeats, uint32_t chunkBlocks)
{
#pragma clang loop unroll(disable)
    for (uint32_t v = 0U; v < V; ++v) {
        Gather(
            stageUb[v * chunkBlocks * 8U], aCompactUb, aPackIdx[chunkPairBase * 8U],
            v * groupElems * static_cast<uint32_t>(sizeof(float)), 64U, chunkRepeats, 8U);
        if (diag == 0U) {
            Gather(
                brcbUb[v * chunkBlocks * 8U], diagUb, aDiagPackIdx[chunkPairBase * 8U],
                v * S * 8U * static_cast<uint32_t>(sizeof(float)), 64U, chunkRepeats, 8U);
        }
    }
    PipeBarrier<PIPE_V>();
    if (diag == 0U) {
        Mul(stageUb, stageUb, brcbUb, 64U, static_cast<uint8_t>(V * chunkRepeats),
            BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        PipeBarrier<PIPE_V>();
    }
    const uint32_t totalChunkBlocks = V * chunkBlocks;
#pragma clang loop unroll(disable)
    for (uint32_t off = 0U; off < totalChunkBlocks; off += 128U) {
        const uint32_t blkCnt = (off + 128U <= totalChunkBlocks) ? 128U : (totalChunkBlocks - off);
        Brcb(brcbUb[off * 64U], stageUb[off * FLOAT_BLOCK_ELEMS], static_cast<uint8_t>(blkCnt), BrcbRepeatParams(1, 8));
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t S>
__aicore__ inline void EliminateChunkMultiBatch(
    LocalTensor<float>& bWorkUb, LocalTensor<float>& aCompactUb, LocalTensor<float>& diagUb,
    LocalTensor<float>& stageUb, LocalTensor<float>& brcbUb, LocalTensor<uint32_t>& aPackIdx,
    LocalTensor<uint32_t>& aDiagPackIdx, uint32_t V, uint32_t sStart, uint32_t sEnd, uint32_t chunkPairBase,
    uint32_t chunkPairs, uint32_t groupElems, uint32_t diag, bool forward)
{
    const uint8_t chunkRepeats = static_cast<uint8_t>((chunkPairs + 7U) / 8U);
    const uint32_t chunkBlocks = static_cast<uint32_t>(chunkRepeats) * 8U;
    PrepareChunkBrcbMultiBatch<S>(
        stageUb, brcbUb, aCompactUb, diagUb, aPackIdx, aDiagPackIdx, V, chunkPairBase, groupElems, diag, chunkRepeats,
        chunkBlocks);
    uint32_t localPairOff = 0U;
#pragma clang loop unroll(disable)
    for (uint32_t s = sStart; s < sEnd; ++s) {
        const uint32_t k = forward ? s : (S - 1U - s);
        const uint32_t jStart = forward ? (k + 1U) : 0U;
        const uint8_t rem = static_cast<uint8_t>(S - 1U - s);
        for (uint32_t v = 0U; v < V; ++v) {
            LocalTensor<float> curBrcb = brcbUb[(v * chunkBlocks + localPairOff) * 64U];
            Mul(curBrcb, curBrcb, bWorkUb[v * groupElems + k * 64U], 64U, rem, BinaryRepeatParams(1, 1, 1, 8, 8, 0));
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t v = 0U; v < V; ++v) {
            LocalTensor<float> curBrcb = brcbUb[(v * chunkBlocks + localPairOff) * 64U];
            Sub(bWorkUb[v * groupElems + jStart * 64U], bWorkUb[v * groupElems + jStart * 64U], curBrcb, 64U, rem,
                BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        }
        PipeBarrier<PIPE_V>();
        localPairOff += rem;
    }
}

template <uint32_t S>
__aicore__ inline void FinalizeAndStoreMultiBatch(
    LocalTensor<float>& bWorkUb, LocalTensor<float>& diagUb, LocalTensor<float>& brcbUb, LocalTensor<float>& bUb,
    LocalTensor<uint32_t>& unpackIdx, __gm__ float* bPtrs[32], uint32_t V, uint32_t curG, uint32_t groupElems,
    uint32_t diag)
{
    if (diag == 0U) {
        Brcb(brcbUb, diagUb, static_cast<uint8_t>(V * S), BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
        Mul(bWorkUb, bWorkUb, brcbUb, 64U, static_cast<uint8_t>(V * S), BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        PipeBarrier<PIPE_V>();
    }
#pragma clang loop unroll(disable)
    for (uint32_t v = 0U; v < V; ++v) {
        Gather(
            bUb[v * groupElems], bWorkUb, unpackIdx, v * groupElems * static_cast<uint32_t>(sizeof(float)), 64U,
            static_cast<uint8_t>(S), 8U);
    }
    PipeBarrier<PIPE_V>();
    SyncVToMte3();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    StoreMultiBatchSpanUb<S>(bPtrs, bUb, curG);
}

template <uint32_t S>
struct SmallMultiBatchTensors {
    LocalTensor<float> bWorkUb;
    LocalTensor<float> aCompactUb;
    LocalTensor<float> brcbUb;
    LocalTensor<float> stageUb;
    LocalTensor<float> bUb;
    LocalTensor<float> diagUb;
    LocalTensor<float> onesUb;
    LocalTensor<uint32_t> packIdx;
    LocalTensor<uint32_t> unpackIdx;
    LocalTensor<uint32_t> aDiagIdx;
    LocalTensor<uint32_t> aPackIdx;
    LocalTensor<uint32_t> aDiagPackIdx;
};

template <uint32_t S>
__aicore__ inline void InitSmallMultiBatchTensors(
    TPipe& pipe, TBuf<TPosition::VECCALC>& floatBuf, TBuf<TPosition::VECCALC>& idxBuf, SmallMultiBatchTensors<S>& ub)
{
    constexpr uint32_t groupElems = 64U * S;
    constexpr uint32_t maxV = (S == 16U) ? 4U : 3U;
    constexpr uint32_t activePairs = (S * (S - 1U)) / 2U;
    constexpr uint8_t gatherRepeats = static_cast<uint8_t>((activePairs + 7U) / 8U);
    constexpr uint32_t totalBlocks = static_cast<uint32_t>(gatherRepeats) * 8U;
    constexpr uint32_t maxChunkBlocks = (S == 16U) ? 80U : 112U;
    constexpr uint32_t workElems = maxV * groupElems;
    constexpr uint32_t stageOff = maxV * maxChunkBlocks * 56U + 64U;
    constexpr uint32_t brcbElems = maxV * maxChunkBlocks * 64U + 64U;
    constexpr uint32_t diagElems = maxV * S * FLOAT_BLOCK_ELEMS;
    constexpr uint32_t floatElems = workElems + workElems + brcbElems + diagElems + 64U;
    constexpr uint32_t packAlloc = (totalBlocks + 16U) * 8U;
    constexpr uint32_t idxElems = 2048U + 2048U + 256U + 2U * packAlloc;
    pipe.InitBuffer(floatBuf, floatElems * sizeof(float));
    pipe.InitBuffer(idxBuf, idxElems * sizeof(uint32_t));
    LocalTensor<float> fBase = floatBuf.Get<float>();
    ub.bWorkUb = fBase;
    ub.aCompactUb = fBase[workElems];
    ub.brcbUb = fBase[2U * workElems];
    ub.stageUb = ub.brcbUb[stageOff];
    ub.bUb = ub.brcbUb;
    ub.diagUb = fBase[2U * workElems + brcbElems];
    ub.onesUb = fBase[2U * workElems + brcbElems + diagElems];
    LocalTensor<uint32_t> iBase = idxBuf.Get<uint32_t>();
    ub.packIdx = iBase;
    ub.unpackIdx = iBase[2048U];
    ub.aDiagIdx = iBase[4096U];
    ub.aPackIdx = iBase[4352U];
    ub.aDiagPackIdx = iBase[4352U + packAlloc];
}

__aicore__ inline bool ComputeAivGroupRange(uint32_t totalGroups, uint32_t aivNum, uint32_t& grpStart, uint32_t& grpEnd)
{
    if (aivNum == 0U || totalGroups == 0U) {
        return false;
    }
    const uint32_t aivIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t baseGroups = totalGroups / aivNum;
    const uint32_t remGroups = totalGroups % aivNum;
    grpStart = aivIdx * baseGroups + ((aivIdx < remGroups) ? aivIdx : remGroups);
    grpEnd = grpStart + baseGroups + ((aivIdx < remGroups) ? 1U : 0U);
    return grpStart < grpEnd;
}

template <uint32_t S>
__aicore__ inline void GatherGroupSpanUb(SmallMultiBatchTensors<S>& ub, uint32_t V, uint32_t groupElems, uint32_t diag)
{
#pragma clang loop unroll(disable)
    for (uint32_t v = 0U; v < V; ++v) {
        Gather(
            ub.bWorkUb[v * groupElems], ub.bUb, ub.packIdx, v * groupElems * static_cast<uint32_t>(sizeof(float)), 64U,
            static_cast<uint8_t>(S), 8U);
        if (diag == 0U) {
            Gather(
                ub.diagUb[v * S * 8U], ub.aCompactUb, ub.aDiagIdx,
                v * groupElems * static_cast<uint32_t>(sizeof(float)), 64U, static_cast<uint8_t>(S / FLOAT_BLOCK_ELEMS),
                8U);
        }
    }
}

template <uint32_t S>
__aicore__ inline void ProcessGroupSpanUb(
    SmallMultiBatchTensors<S>& ub, __gm__ float* bPtrs[32], uint32_t V, uint32_t curG, uint32_t groupElems,
    const StrsmbatchedTilingData& t, bool forward)
{
    GatherGroupSpanUb<S>(ub, V, groupElems, t.diag);
    PipeBarrier<PIPE_V>();
    if (t.alpha != 1.0f) {
        Muls(ub.bWorkUb, ub.bWorkUb, t.alpha, 64ULL, static_cast<uint8_t>(V * S), UnaryRepeatParams(1, 1, 8, 8));
    }
    if (t.diag == 0U) {
        Div(ub.diagUb, ub.onesUb, ub.diagUb, 64U, static_cast<uint8_t>(V * (S / FLOAT_BLOCK_ELEMS)),
            BinaryRepeatParams(1, 1, 1, 8, 0, 8));
    }
    PipeBarrier<PIPE_V>();

    uint32_t sStart = 0U;
    uint32_t chunkPairBase = 0U;
#pragma clang loop unroll(disable)
    while (sStart + 1U < S) {
        const uint32_t sEnd = GetChunkEnd<S>(sStart);
        const uint32_t chunkPairs = ((S - 1U - sStart) + (S - sEnd)) * (sEnd - sStart) / 2U;
        EliminateChunkMultiBatch<S>(
            ub.bWorkUb, ub.aCompactUb, ub.diagUb, ub.stageUb, ub.brcbUb, ub.aPackIdx, ub.aDiagPackIdx, V, sStart, sEnd,
            chunkPairBase, chunkPairs, groupElems, t.diag, forward);
        chunkPairBase += chunkPairs;
        sStart = sEnd;
    }
    FinalizeAndStoreMultiBatch<S>(
        ub.bWorkUb, ub.diagUb, ub.brcbUb, ub.bUb, ub.unpackIdx, bPtrs, V, curG, groupElems, t.diag);
}

template <uint32_t S>
__aicore__ __attribute__((noinline)) void SolveSmallMultiBatchAiv(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, const StrsmbatchedTilingData& t)
{
    const uint32_t aivNum = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());
    constexpr uint32_t G = 64U / S;
    constexpr uint32_t groupElems = 64U * S;
    constexpr uint32_t maxV = (S == 16U) ? 4U : 3U;
    uint32_t grpStart = 0U;
    uint32_t grpEnd = 0U;
    if (!ComputeAivGroupRange(CeilDivU(t.batchCount, G), aivNum, grpStart, grpEnd)) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> floatBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    SmallMultiBatchTensors<S> ub;
    InitSmallMultiBatchTensors<S>(pipe, floatBuf, idxBuf, ub);
    const bool aColMajorStep = (t.trans == 0U);
    const bool forward = IsForwardSweep(t);

    __gm__ float* aPtrs[32];
    __gm__ float* bPtrs[32];

#pragma clang loop unroll(disable)
    for (uint32_t grp = grpStart; grp < grpEnd; grp += maxV) {
        const uint32_t V = ((grpEnd - grp) >= maxV) ? maxV : (grpEnd - grp);
        const uint32_t bBase = grp * G;
        const uint32_t curG = (bBase + V * G <= t.batchCount) ? (V * G) : (t.batchCount - bBase);
        LoadMultiBatchSpanUb<S>(ub.aCompactUb, aPtrArray, bBase, curG, V * G, aPtrs);
        if (grp > grpStart) {
            SyncMte3ToNext();
        }
        LoadMultiBatchSpanUb<S>(ub.bUb, bPtrArray, bBase, curG, curG, bPtrs);
        if (grp == grpStart) {
            InitMultiBatchIdxUb<S>(
                ub.packIdx, ub.unpackIdx, ub.aDiagIdx, ub.aPackIdx, ub.aDiagPackIdx, ub.onesUb, ub.bWorkUb,
                aColMajorStep, forward);
        }
        SyncMte2ToVOnly();
        ProcessGroupSpanUb<S>(ub, bPtrs, V, curG, groupElems, t, forward);
    }
    SyncMte3ToNext();
}

} // namespace

extern "C" __global__ __aicore__ void strsmbatched_small_aiv_kernel(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, StrsmbatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (t.m == 8U) {
        SolveSmallMultiBatchAiv8(aPtrArray, bPtrArray, t);
        return;
    }
    if (t.m == 16U) {
        SolveSmallMultiBatchAiv<16U>(aPtrArray, bPtrArray, t);
        return;
    }
    if (t.m == 32U) {
        SolveSmallMultiBatchAiv<32U>(aPtrArray, bPtrArray, t);
        return;
    }
}

void strsmbatched_small_aiv_kernel_do(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, const StrsmbatchedTilingData& tiling, uint32_t numBlocks, void* stream)
{
    strsmbatched_small_aiv_kernel<<<numBlocks, nullptr, stream>>>(aPtrArray, bPtrArray, tiling);
}
