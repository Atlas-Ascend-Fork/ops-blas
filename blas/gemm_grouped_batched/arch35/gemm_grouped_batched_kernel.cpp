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
#include "gemm_grouped_batched_tiling_data.h"

using namespace AscendC;

constexpr uint32_t VL = 256 / sizeof(float);  // 64 FP32 elements per vector register

class GemmGroupedBatchedAIV {
public:
    __aicore__ inline GemmGroupedBatchedAIV() {}
    __aicore__ inline void Init(GM_ADDR tilingGm, GM_ADDR groupParamsGm,
                                GM_ADDR aPtrArrayGm, GM_ADDR bPtrArrayGm,
                                GM_ADDR cPtrArrayGm);
    __aicore__ inline void Process(uint32_t startBatch, uint32_t numBatch);

private:
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline const __gm__ GroupParam* FindGroupParam(uint32_t batchIdx);
    __aicore__ inline __gm__ float* ReadPtrFromArray(GM_ADDR ptrArrayGm, uint32_t batchIdx);
    __aicore__ inline void ProcessGemmBatch(uint32_t batchIdx, const __gm__ GroupParam* gp);
    __aicore__ inline void ProcessBetaOnly(uint32_t batchIdx, const __gm__ GroupParam* gp);
    __aicore__ inline void ComputeColumnWise(uint32_t mCurr, uint32_t kCurr, uint32_t nCurr,
                                              const __gm__ GroupParam* gp);
    __aicore__ inline void ComputeDotProduct(uint32_t mCurr, uint32_t kCurr, uint32_t nCurr,
                                              const __gm__ GroupParam* gp);
    __aicore__ inline __ubuf__ float* GetDotProductBColAddr(
        __ubuf__ float* bAddr, __ubuf__ float* bPackedAddr, uint32_t j,
        uint32_t kCurr, uint32_t tileKAligned, uint32_t tileNAligned,
        const __gm__ GroupParam* gp);
    __aicore__ inline void AccumulateDotProductElem(
        __ubuf__ float* aRowAddr, __ubuf__ float* bColAddr, __ubuf__ float* cElem,
        uint16_t kVecNum, uint16_t kTail);
    __aicore__ inline void ApplyAlphaBeta(float alpha, float beta,
                                           uint32_t mCurr, uint32_t nCurr,
                                           uint32_t tileMAligned);
    __aicore__ inline void CopyInA(int32_t lda,
                                    uint32_t mOff, uint32_t kOff,
                                    uint32_t mCurr, uint32_t kCurr,
                                    const __gm__ GroupParam* gp);
    __aicore__ inline void CopyInB(int32_t ldb,
                                    uint32_t kOff, uint32_t nOff,
                                    uint32_t kCurr, uint32_t nCurr,
                                    const __gm__ GroupParam* gp);
    __aicore__ inline void CopyInC(int32_t ldc,
                                    uint32_t mOff, uint32_t nOff,
                                    uint32_t mCurr, uint32_t nCurr,
                                    const __gm__ GroupParam* gp);
    __aicore__ inline void CopyOutC(int32_t ldc,
                                     uint32_t mOff, uint32_t nOff,
                                     uint32_t mCurr, uint32_t nCurr,
                                     const __gm__ GroupParam* gp);

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> bufA_;
    TBuf<QuePosition::VECCALC> bufB_;
    TBuf<QuePosition::VECCALC> bufC_;
    TBuf<QuePosition::VECCALC> bufCIn_;
    TBuf<QuePosition::VECCALC> bufMulTmp_;
    TBuf<QuePosition::VECCALC> bufVecTmp_;

    GlobalTensor<float> inAGM_;
    GlobalTensor<float> inBGM_;
    GlobalTensor<float> inCGM_;

    GemmGroupedBatchedTilingData tiling_;
    GM_ADDR groupParamsGm_;
    GM_ADDR aPtrArrayGm_;
    GM_ADDR bPtrArrayGm_;
    GM_ADDR cPtrArrayGm_;
};

__aicore__ inline void GemmGroupedBatchedAIV::Init(
    GM_ADDR tilingGm, GM_ADDR groupParamsGm,
    GM_ADDR aPtrArrayGm, GM_ADDR bPtrArrayGm,
    GM_ADDR cPtrArrayGm)
{
    ParseTilingData(tilingGm);
    groupParamsGm_ = groupParamsGm;
    aPtrArrayGm_   = aPtrArrayGm;
    bPtrArrayGm_   = bPtrArrayGm;
    cPtrArrayGm_   = cPtrArrayGm;

    pipe_.InitBuffer(bufA_,      tiling_.maxBufSizeA);
    pipe_.InitBuffer(bufB_,      tiling_.maxBufSizeB);
    pipe_.InitBuffer(bufC_,      tiling_.maxBufSizeC);
    pipe_.InitBuffer(bufCIn_,    tiling_.maxBufSizeCIn);
    pipe_.InitBuffer(bufMulTmp_, tiling_.maxBufSizeMulTmp);
    pipe_.InitBuffer(bufVecTmp_, tiling_.maxBufSizeVecTmp);
}

__aicore__ inline void GemmGroupedBatchedAIV::ParseTilingData(GM_ADDR tilingGm)
{
    auto* td = reinterpret_cast<__gm__ GemmGroupedBatchedTilingData*>(tilingGm);
    tiling_.groupCount        = td->groupCount;
    tiling_.totalBatchCount   = td->totalBatchCount;
    tiling_.dtype             = td->dtype;
    tiling_.coreNum           = td->coreNum;
    tiling_.usedCoreNum       = td->usedCoreNum;
    tiling_.batchPerCore      = td->batchPerCore;
    tiling_.batchTail         = td->batchTail;
    tiling_.groupParamsGmAddr = td->groupParamsGmAddr;
    tiling_.aPtrArrayGmAddr   = td->aPtrArrayGmAddr;
    tiling_.bPtrArrayGmAddr   = td->bPtrArrayGmAddr;
    tiling_.cPtrArrayGmAddr   = td->cPtrArrayGmAddr;
    tiling_.maxBufSizeA      = td->maxBufSizeA;
    tiling_.maxBufSizeB      = td->maxBufSizeB;
    tiling_.maxBufSizeC      = td->maxBufSizeC;
    tiling_.maxBufSizeCIn    = td->maxBufSizeCIn;
    tiling_.maxBufSizeMulTmp = td->maxBufSizeMulTmp;
    tiling_.maxBufSizeVecTmp = td->maxBufSizeVecTmp;
}

__aicore__ inline const __gm__ GroupParam* GemmGroupedBatchedAIV::FindGroupParam(uint32_t batchIdx)
{
    auto* params = reinterpret_cast<__gm__ GroupParam*>(groupParamsGm_);
    uint32_t lo = 0;
    uint32_t hi = tiling_.groupCount;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (params[mid].groupOffset <= batchIdx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return &params[lo - 1];
}

__aicore__ inline __gm__ float* GemmGroupedBatchedAIV::ReadPtrFromArray(
    GM_ADDR ptrArrayGm, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(ptrArrayGm) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ float*>(rawAddr);
}

__aicore__ inline void GemmGroupedBatchedAIV::Process(
    uint32_t startBatch, uint32_t numBatch)
{
    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        const __gm__ GroupParam* gp = FindGroupParam(batchIdx);

        if (gp->m == 0 || gp->n == 0) {
            continue;
        }

        float alphaReal = gp->alphaReal;
        float alphaImag = gp->alphaImag;
        if (gp->k == 0 || (alphaReal == 0.0f && alphaImag == 0.0f)) {
            ProcessBetaOnly(batchIdx, gp);
            continue;
        }
        ProcessGemmBatch(batchIdx, gp);
    }
}

__aicore__ inline void GemmGroupedBatchedAIV::ProcessBetaOnly(
    uint32_t batchIdx, const __gm__ GroupParam* gp)
{
    __gm__ float* cRaw = ReadPtrFromArray(cPtrArrayGm_, batchIdx);
    inCGM_.SetGlobalBuffer(cRaw);
    uint32_t m = gp->m;
    uint32_t n = gp->n;
    int32_t ldc = gp->ldc;
    float beta = gp->betaReal;

    if (beta == 0.0f) {
        for (uint32_t mOff = 0; mOff < m; mOff += gp->tileM) {
            uint32_t mCurr = (mOff + gp->tileM > m) ? (m - mOff) : gp->tileM;
            for (uint32_t nOff = 0; nOff < n; nOff += gp->tileN) {
                uint32_t nCurr = (nOff + gp->tileN > n) ? (n - nOff) : gp->tileN;
                LocalTensor<float> cLocal = bufC_.Get<float>();
                // Use padded count: UB layout has tileM_aligned stride between columns;
                // contiguous mCurr*nCurr misses column data when tileM_aligned != mCurr.
                Duplicate(cLocal, 0.0f, gp->tileM_aligned * nCurr);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyOutC(ldc, mOff, nOff, mCurr, nCurr, gp);
                // CopyOutC (MTE3) reads bufC_; wait until the copy-out has
                // drained before the next tile's vector writes overwrite it.
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        return;
    }

    for (uint32_t mOff = 0; mOff < m; mOff += gp->tileM) {
        uint32_t mCurr = (mOff + gp->tileM > m) ? (m - mOff) : gp->tileM;
        for (uint32_t nOff = 0; nOff < n; nOff += gp->tileN) {
            uint32_t nCurr = (nOff + gp->tileN > n) ? (n - nOff) : gp->tileN;
            CopyInC(ldc, mOff, nOff, mCurr, nCurr, gp);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            LocalTensor<float> cInLocal = bufCIn_.Get<float>();
            LocalTensor<float> cLocal   = bufC_.Get<float>();
            // Use padded count: UB layout has tileM_aligned stride between columns;
            // contiguous mCurr*nCurr misses column data when tileM_aligned != mCurr.
            Muls(cLocal, cInLocal, beta, gp->tileM_aligned * nCurr);
            PipeBarrier<PIPE_V>();
            // The next tile's CopyInC reuses bufCIn_; wait until this tile's
            // vector reads (Muls above) have drained before MTE2 overwrites it.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyOutC(ldc, mOff, nOff, mCurr, nCurr, gp);
            // CopyOutC (MTE3) reads bufC_; wait until the copy-out has
            // drained before the next tile's vector writes overwrite it.
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
}

__aicore__ inline void GemmGroupedBatchedAIV::ProcessGemmBatch(
    uint32_t batchIdx, const __gm__ GroupParam* gp)
{
    // Dynamically set GlobalTensor addresses from pointer arrays
    inAGM_.SetGlobalBuffer(ReadPtrFromArray(aPtrArrayGm_, batchIdx));
    inBGM_.SetGlobalBuffer(ReadPtrFromArray(bPtrArrayGm_, batchIdx));
    inCGM_.SetGlobalBuffer(ReadPtrFromArray(cPtrArrayGm_, batchIdx));

    uint32_t m = gp->m;
    uint32_t n = gp->n;
    uint32_t k = gp->k;
    uint32_t tileM = gp->tileM;
    uint32_t tileN = gp->tileN;
    uint32_t tileK = gp->tileK;
    int32_t ldc = gp->ldc;
    float alpha = gp->alphaReal;
    float beta  = gp->betaReal;

    for (uint32_t mOff = 0; mOff < m; mOff += tileM) {
        uint32_t mCurr = (mOff + tileM > m) ? (m - mOff) : tileM;
        for (uint32_t nOff = 0; nOff < n; nOff += tileN) {
            uint32_t nCurr = (nOff + tileN > n) ? (n - nOff) : tileN;

            if (beta != 0.0f) {
                CopyInC(ldc, mOff, nOff, mCurr, nCurr, gp);
            }

            LocalTensor<float> cLocal = bufC_.Get<float>();
            // Use padded count: UB layout has tileM_aligned stride between columns;
            // contiguous mCurr*nCurr misses column data when tileM_aligned != mCurr.
            Duplicate(cLocal, 0.0f, gp->tileM_aligned * nCurr);
            PipeBarrier<PIPE_V>();

        for (uint32_t kOff = 0; kOff < k; kOff += tileK) {
            uint32_t kCurr = (kOff + tileK > k) ? (k - kOff) : tileK;
            CopyInA(gp->lda, mOff, kOff, mCurr, kCurr, gp);
            CopyInB(gp->ldb, kOff, nOff, kCurr, nCurr, gp);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // ComputeColumnWise reads B scalars and ComputeDotProduct packs B
            // columns via scalar-pipe UB accesses; the V-pipe wait above does
            // not order the scalar pipe, so an explicit MTE2_S pair is required
            // after the copy-in batch (same pattern as trsmbatched/strmm).
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

            if (gp->transa == 0) {
                ComputeColumnWise(mCurr, kCurr, nCurr, gp);
            } else {
                ComputeDotProduct(mCurr, kCurr, nCurr, gp);
            }

            // The next kOff iteration re-issues CopyInA/CopyInB into the
            // same bufA_/bufB_; wait until this iteration's vector reads
            // AND scalar reads (bScalar / bPackedAddr pack loop) have
            // drained before MTE2 overwrites the buffers.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        }

        ApplyAlphaBeta(alpha, beta, mCurr, nCurr, gp->tileM_aligned);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyOutC(ldc, mOff, nOff, mCurr, nCurr, gp);
        // CopyOutC (MTE3) reads bufC_; wait until the copy-out has
        // drained before the next tile's vector writes overwrite it.
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
}

__aicore__ inline void GemmGroupedBatchedAIV::ComputeColumnWise(
    uint32_t mCurr, uint32_t kCurr, uint32_t nCurr,
    const __gm__ GroupParam* gp)
{
    LocalTensor<float> aLocal = bufA_.Get<float>();
    LocalTensor<float> bLocal = bufB_.Get<float>();
    LocalTensor<float> cLocal = bufC_.Get<float>();

    uint32_t tileM_aligned = gp->tileM_aligned;
    uint32_t tileK_aligned = gp->tileK_aligned;
    uint32_t tileN_aligned = gp->tileN_aligned;

    __ubuf__ float* aAddr = (__ubuf__ float*)aLocal.GetPhyAddr();
    __ubuf__ float* bAddr = (__ubuf__ float*)bLocal.GetPhyAddr();
    __ubuf__ float* cAddr = (__ubuf__ float*)cLocal.GetPhyAddr();

    uint16_t mVecNum = static_cast<uint16_t>((mCurr + VL - 1) / VL);
    uint16_t mTail   = static_cast<uint16_t>(mCurr % VL);

    for (uint32_t l = 0; l < kCurr; l++) {
        __ubuf__ float* aColAddr = aAddr + l * tileM_aligned;
        for (uint32_t j = 0; j < nCurr; j++) {
            float bScalar;
            if (gp->transb == 0) {
                bScalar = *(bAddr + l + j * tileK_aligned);
            } else {
                bScalar = *(bAddr + j + l * tileN_aligned);
            }
            __ubuf__ float* cColAddr = cAddr + j * tileM_aligned;

            __VEC_SCOPE__ {
                AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregA;
                AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregMul;
                AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregC;
                auto maskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

                for (uint16_t mv = 0; mv < mVecNum; mv++) {
                    uint32_t rOff = static_cast<uint32_t>(mv) * VL;
                    uint32_t chunk = (mv == mVecNum - 1 && mTail != 0) ? mTail : VL;
                    auto mask = (chunk < VL)
                        ? AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(chunk)
                        : maskAll;

                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        vregA, aColAddr + rOff);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        vregC, cColAddr + rOff);
                    AscendC::MicroAPI::Muls<float>(vregMul, vregA, bScalar, mask);
                    AscendC::MicroAPI::Add<float>(vregC, vregC, vregMul, mask);
                    AscendC::MicroAPI::DataCopy<float>(cColAddr + rOff, vregC, mask);
                }
            }
        }
    }
}

__aicore__ inline __ubuf__ float* GemmGroupedBatchedAIV::GetDotProductBColAddr(
    __ubuf__ float* bAddr, __ubuf__ float* bPackedAddr, uint32_t j,
    uint32_t kCurr, uint32_t tileKAligned, uint32_t tileNAligned,
    const __gm__ GroupParam* gp)
{
    if (gp->transb == 0) {
        return bAddr + j * tileKAligned;
    }
    // B^T is row-major in the UB tile. Pack one logical column so
    // multiplication and reduction can still use Vector Core SIMD.
    for (uint32_t l = 0; l < kCurr; l++) {
        bPackedAddr[l] = *(bAddr + j + l * tileNAligned);
    }
    // bPackedAddr is written by the scalar pipe and consumed by the vector
    // pipe in AccumulateDotProductElem; order the two pipes explicitly.
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    return bPackedAddr;
}

__aicore__ inline void GemmGroupedBatchedAIV::AccumulateDotProductElem(
    __ubuf__ float* aRowAddr, __ubuf__ float* bColAddr, __ubuf__ float* cElem,
    uint16_t kVecNum, uint16_t kTail)
{
    __VEC_SCOPE__ {
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregA;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregB;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregMul;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregSum;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregC;
        auto maskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        uint32_t one = 1;
        auto maskOne = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(one);

        AscendC::MicroAPI::Duplicate<float>(vregSum, 0.0f, maskAll);
        for (uint16_t kv = 0; kv < kVecNum; kv++) {
            uint32_t kOff = static_cast<uint32_t>(kv) * VL;
            uint32_t chunk = (kv == kVecNum - 1 && kTail != 0) ? kTail : VL;
            auto mask = (chunk < VL)
                ? AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(chunk)
                : maskAll;
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                vregA, aRowAddr + kOff);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                vregB, bColAddr + kOff);
            AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                vregMul, vregA, vregB, mask);
            AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                vregSum, vregSum, vregMul, maskAll);
        }
        AscendC::MicroAPI::ReduceSum(vregSum, vregSum, maskAll);
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vregC, cElem);
        AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
            vregSum, vregSum, vregC, maskOne);
        AscendC::MicroAPI::DataCopy<float,
            AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(cElem, vregSum, 1, maskAll);
    }
}

__aicore__ inline void GemmGroupedBatchedAIV::ComputeDotProduct(
    uint32_t mCurr, uint32_t kCurr, uint32_t nCurr,
    const __gm__ GroupParam* gp)
{
    LocalTensor<float> aLocal = bufA_.Get<float>();
    LocalTensor<float> bLocal = bufB_.Get<float>();
    LocalTensor<float> cLocal = bufC_.Get<float>();

    uint32_t tileKAligned = gp->tileK_aligned;
    uint32_t tileMAligned = gp->tileM_aligned;
    uint32_t tileNAligned = gp->tileN_aligned;

    __ubuf__ float* aAddr = (__ubuf__ float*)aLocal.GetPhyAddr();
    __ubuf__ float* bAddr = (__ubuf__ float*)bLocal.GetPhyAddr();
    __ubuf__ float* cAddr = (__ubuf__ float*)cLocal.GetPhyAddr();
    __ubuf__ float* bPackedAddr = (__ubuf__ float*)bufVecTmp_.Get<float>().GetPhyAddr();

    uint16_t kVecNum = static_cast<uint16_t>((kCurr + VL - 1) / VL);
    uint16_t kTail = static_cast<uint16_t>(kCurr % VL);

    for (uint32_t r = 0; r < mCurr; r++) {
        __ubuf__ float* aRowAddr = aAddr + r * tileKAligned;
        for (uint32_t j = 0; j < nCurr; j++) {
            __ubuf__ float* bColAddr = GetDotProductBColAddr(
                bAddr, bPackedAddr, j, kCurr, tileKAligned, tileNAligned, gp);
            __ubuf__ float* cElem = cAddr + r + j * tileMAligned;
            AccumulateDotProductElem(aRowAddr, bColAddr, cElem, kVecNum, kTail);
            if (gp->transb != 0) {
                // The next j iteration rewrites bPackedAddr on the scalar pipe;
                // wait until this iteration's vector reads have drained.
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
            }
        }
    }
}

__aicore__ inline void GemmGroupedBatchedAIV::ApplyAlphaBeta(
    float alpha, float beta, uint32_t mCurr, uint32_t nCurr, uint32_t tileMAligned)
{
    LocalTensor<float> cLocal   = bufC_.Get<float>();
    LocalTensor<float> mulLocal = bufMulTmp_.Get<float>();
    // Use padded count: UB layout has tileMAligned stride between columns;
    // contiguous mCurr*nCurr misses column data when tileMAligned != mCurr.
    uint32_t paddedCount = tileMAligned * nCurr;

    if (alpha != 1.0f) {
        Muls(cLocal, cLocal, alpha, paddedCount);
        PipeBarrier<PIPE_V>();
    }
    if (beta != 0.0f) {
        LocalTensor<float> cInLocal = bufCIn_.Get<float>();
        Muls(mulLocal, cInLocal, beta, paddedCount);
        PipeBarrier<PIPE_V>();
        Add(cLocal, cLocal, mulLocal, paddedCount);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void GemmGroupedBatchedAIV::CopyInA(
    int32_t lda, uint32_t mOff, uint32_t kOff,
    uint32_t mCurr, uint32_t kCurr, const __gm__ GroupParam* gp)
{
    LocalTensor<float> aLocal = bufA_.Get<float>();
    uint32_t mCurrAligned = ((mCurr + GEMM_GROUPED_FP32_ALIGN - 1) &
                             ~(GEMM_GROUPED_FP32_ALIGN - 1));
    uint32_t kCurrAligned = ((kCurr + GEMM_GROUPED_FP32_ALIGN - 1) &
                             ~(GEMM_GROUPED_FP32_ALIGN - 1));

    if (gp->transa == 0) {
        uint64_t aBase = (uint64_t)kOff * lda + mOff;
        uint32_t srcStride = (lda - (int32_t)mCurr) * sizeof(float);
        uint32_t dstStride = (gp->tileM_aligned - mCurrAligned) * sizeof(float) / 32;
        DataCopyExtParams cp{static_cast<uint16_t>(kCurr),
            static_cast<uint32_t>(mCurr * sizeof(float)), srcStride, dstStride, 0};
        DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
        DataCopyPad(aLocal, inAGM_[aBase], cp, pp);
    } else {
        uint64_t aBase = (uint64_t)mOff * lda + kOff;
        uint32_t srcStride = (lda - (int32_t)kCurr) * sizeof(float);
        uint32_t dstStride = (gp->tileK_aligned - kCurrAligned) * sizeof(float) / 32;
        DataCopyExtParams cp{static_cast<uint16_t>(mCurr),
            static_cast<uint32_t>(kCurr * sizeof(float)), srcStride, dstStride, 0};
        DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
        DataCopyPad(aLocal, inAGM_[aBase], cp, pp);
    }
}

__aicore__ inline void GemmGroupedBatchedAIV::CopyInB(
    int32_t ldb, uint32_t kOff, uint32_t nOff,
    uint32_t kCurr, uint32_t nCurr, const __gm__ GroupParam* gp)
{
    LocalTensor<float> bLocal = bufB_.Get<float>();
    uint32_t kCurrAligned = ((kCurr + GEMM_GROUPED_FP32_ALIGN - 1) &
                             ~(GEMM_GROUPED_FP32_ALIGN - 1));
    uint32_t nCurrAligned = ((nCurr + GEMM_GROUPED_FP32_ALIGN - 1) &
                             ~(GEMM_GROUPED_FP32_ALIGN - 1));

    if (gp->transb == 0) {
        uint64_t bBase = (uint64_t)nOff * ldb + kOff;
        uint32_t srcStride = (ldb - (int32_t)kCurr) * sizeof(float);
        uint32_t dstStride = (gp->tileK_aligned - kCurrAligned) * sizeof(float) / 32;
        DataCopyExtParams cp{static_cast<uint16_t>(nCurr),
            static_cast<uint32_t>(kCurr * sizeof(float)), srcStride, dstStride, 0};
        DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
        DataCopyPad(bLocal, inBGM_[bBase], cp, pp);
    } else {
        uint64_t bBase = (uint64_t)kOff * ldb + nOff;
        uint32_t srcStride = (ldb - (int32_t)nCurr) * sizeof(float);
        uint32_t dstStride = (gp->tileN_aligned - nCurrAligned) * sizeof(float) / 32;
        DataCopyExtParams cp{static_cast<uint16_t>(kCurr),
            static_cast<uint32_t>(nCurr * sizeof(float)), srcStride, dstStride, 0};
        DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
        DataCopyPad(bLocal, inBGM_[bBase], cp, pp);
    }
}

__aicore__ inline void GemmGroupedBatchedAIV::CopyInC(
    int32_t ldc, uint32_t mOff, uint32_t nOff,
    uint32_t mCurr, uint32_t nCurr, const __gm__ GroupParam* gp)
{
    LocalTensor<float> cInLocal = bufCIn_.Get<float>();
    uint32_t mCurrAligned = ((mCurr + GEMM_GROUPED_FP32_ALIGN - 1) &
                             ~(GEMM_GROUPED_FP32_ALIGN - 1));

    uint64_t cBase = (uint64_t)nOff * ldc + mOff;
    uint32_t srcStride = (ldc - (int32_t)mCurr) * sizeof(float);
    uint32_t dstStride = (gp->tileM_aligned - mCurrAligned) * sizeof(float) / 32;
    DataCopyExtParams cp{static_cast<uint16_t>(nCurr),
        static_cast<uint32_t>(mCurr * sizeof(float)), srcStride, dstStride, 0};
    // isPad=true: zero-fill padding from mCurr to mCurrAligned in UB,
    // ensuring column-by-column vector ops process clean data.
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    DataCopyPad(cInLocal, inCGM_[cBase], cp, pp);
}

__aicore__ inline void GemmGroupedBatchedAIV::CopyOutC(
    int32_t ldc, uint32_t mOff, uint32_t nOff,
    uint32_t mCurr, uint32_t nCurr, const __gm__ GroupParam* gp)
{
    LocalTensor<float> cLocal = bufC_.Get<float>();
    uint32_t mCurrAligned = ((mCurr + GEMM_GROUPED_FP32_ALIGN - 1) &
                             ~(GEMM_GROUPED_FP32_ALIGN - 1));

    uint64_t cBase = (uint64_t)nOff * ldc + mOff;
    // UB→GM: src is UB (stride in 32B units), dst is GM (stride in bytes)
    uint32_t srcStride = (gp->tileM_aligned - mCurrAligned) * sizeof(float) / 32;
    uint32_t dstStride = (ldc - (int32_t)mCurr) * sizeof(float);
    DataCopyExtParams cp{static_cast<uint16_t>(nCurr),
        static_cast<uint32_t>(mCurr * sizeof(float)), srcStride, dstStride, 0};
    // UB→GM DataCopyPad takes 3 args (GlobalTensor dst, LocalTensor src, DataCopyExtParams)
    DataCopyPad(inCGM_[cBase], cLocal, cp);
}

extern "C" __global__ __aicore__ void gemm_grouped_batched_kernel(
    GM_ADDR tilingGm, GM_ADDR groupParamsGm,
    GM_ADDR aPtrArrayGm, GM_ADDR bPtrArrayGm, GM_ADDR cPtrArrayGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* td = reinterpret_cast<__gm__ GemmGroupedBatchedTilingData*>(tilingGm);
    uint32_t coreIdx = GetBlockIdx();
    uint32_t startBatch = coreIdx * td->batchPerCore;
    uint32_t numBatch = (coreIdx == td->usedCoreNum - 1)
                        ? td->batchTail : td->batchPerCore;
    if (numBatch == 0) {
        return;
    }

    GemmGroupedBatchedAIV op;
    op.Init(tilingGm, groupParamsGm, aPtrArrayGm, bPtrArrayGm, cPtrArrayGm);
    op.Process(startBatch, numBatch);
}

void gemm_grouped_batched_kernel_do(
    GM_ADDR tilingGm, GM_ADDR groupParamsGm,
    GM_ADDR aPtrArrayGm, GM_ADDR bPtrArrayGm, GM_ADDR cPtrArrayGm,
    uint32_t numBlocks, void* stream)
{
    gemm_grouped_batched_kernel<<<numBlocks, nullptr, stream>>>(
        tilingGm, groupParamsGm, aPtrArrayGm, bPtrArrayGm, cPtrArrayGm);
}
