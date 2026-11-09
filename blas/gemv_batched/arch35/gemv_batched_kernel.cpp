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
#include <type_traits>
#include "acl/acl_rt.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "gemv_batched_tiling_data.h"

using namespace AscendC;

// Helper: read a device-side pointer from the pointer array stored in GM.
// For use inside SIMT kernel blocks (__simt_vf__ functions).
template <typename T>
__simt_callee__ __aicore__ inline __gm__ T* ReadPtrFromArray(GM_ADDR aarrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(aarrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ T*>(rawAddr);
}

// Same as ReadPtrFromArray but without __simt_callee__ decorator — for use in AIV context.
template <typename T>
__aicore__ inline __gm__ T* ReadPtrFromAivArray(GM_ADDR aarrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(aarrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ T*>(rawAddr);
}

constexpr uint32_t ELENUM_BLOCK_FP16 = 16;
constexpr uint32_t ELENUM_REPEAT_FP16 = 128;
// V pipe round-down granularity (= B32 / sizeof(float) = 8)
constexpr uint32_t VEC_FLOAT_PER_REPEAT = 32u / sizeof(float);

template <typename T, const bool TRANS_T = false, bool HSS_MODE = false>
class GemvBatchedAIV {
public:
    __aicore__ inline GemvBatchedAIV() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR x, GM_ADDR y, const GemvBatchedTilingData& tdata);
    __aicore__ inline void SetBatchPointers(GM_ADDR A, GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process(uint32_t batchCountToProcess = 0);

    __aicore__ inline void InitUbuf();
    __aicore__ inline void ParseTilingData(const GemvBatchedTilingData& tdata);
    __aicore__ inline void CopyInMatAndVec(
        uint32_t curBatchId, uint32_t nOffset, uint32_t nCurr, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void CopyInY(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void ApplyAlphaBeta(uint32_t curCalMatNum);
    __aicore__ inline void CastleHalfToFloat(uint32_t dotCnt);
    __aicore__ inline void CastleYHalfToFloat();
    __aicore__ inline void CastFloatToHalf();
    __aicore__ inline void CopyOutRsltFloat(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void CopyOutRsltHalf(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void ComputeDotProduct(uint32_t outCnt, uint32_t dotCnt, uint32_t dotTile, bool isFirstDot);
    __aicore__ inline void OutputMChunkAndSync(uint32_t b, uint32_t mOffset, uint32_t mCurr, uint32_t mCurrAlign);

private:
    static constexpr bool IS_FLOAT = IsSameType<T, float>::value;
    static constexpr bool IS_FLOAT_OUT = IS_FLOAT || HSS_MODE; // y output is float

    TPipe pipe;

    GlobalTensor<T> inAGM;
    GlobalTensor<T> inxGM;
    GlobalTensor<T> inyGM;
    GlobalTensor<T> outGM;
    GlobalTensor<float> inyFloatGM; // HSS: float y input
    GlobalTensor<float> outFloatGM; // HSS: float y output

    TBuf<QuePosition::VECCALC> inABuf;
    TBuf<QuePosition::VECCALC> inxBuf;
    TBuf<QuePosition::VECCALC> inyBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    TBuf<QuePosition::VECCALC> matTmpBuf;
    TBuf<QuePosition::VECCALC> vecTmpBuf;
    TBuf<QuePosition::VECCALC> matPreBuf;
    TBuf<QuePosition::VECCALC> vecPreBuf;

    LocalTensor<float> matTmpLocal;
    LocalTensor<float> vecTmpLocal;
    LocalTensor<T> matPreLocal;
    LocalTensor<T> vecPreLocal;
    LocalTensor<float> matFloatLocal;
    LocalTensor<float> vecFloatLocal;
    LocalTensor<float> yInFloatLocal;
    LocalTensor<float> mulResultLocal;
    LocalTensor<float> sumResultLocal;

    float alpha = 1.0f;
    float beta = 0.0f;

    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t outDim = 0;  // 输出向量维度
    uint32_t dotDim = 0;  // 点积向量维度
    uint32_t outTile = 0; // outTile — 输出维度切分
    uint32_t dotTile = 0; // dotTile — 点积维度切分
    uint32_t mCurr = 0;
    uint32_t calMatNum = 0;
    uint32_t batchGroupSize = 0; // 一次搬多个batch进UB时使用
    uint32_t maxMatEleNum = 0;
    uint32_t maxVecEleNum = 0;
    // Host-precomputed buffer sizes (256B-aligned), read in ParseTilingData
    uint32_t szInA = 0, szInx = 0, szInY = 0, szOut = 0, szMatTmp = 0, szVecTmp = 0;
    int32_t lda = 0;
    int32_t vecIdx;
};

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::Init(
    GM_ADDR A, GM_ADDR x, GM_ADDR y, const GemvBatchedTilingData& tdata)
{
    this->vecIdx = GetBlockIdx();
    ParseTilingData(tdata);
    this->inAGM.SetGlobalBuffer((__gm__ T*)A);
    this->inxGM.SetGlobalBuffer((__gm__ T*)x);
    if constexpr (HSS_MODE) {
        this->inyFloatGM.SetGlobalBuffer((__gm__ float*)y);
        this->outFloatGM.SetGlobalBuffer((__gm__ float*)y);
    } else {
        this->inyGM.SetGlobalBuffer((__gm__ T*)y);
        this->outGM.SetGlobalBuffer((__gm__ T*)y);
    }
    InitUbuf();
}

// Update per-batch GM pointers only — does not re-allocate UB buffers.
// Used in transpose AIV path where Init() is called once before the batch loop,
// then SetBatchPointers() is called inside the loop to point at each batch's data.
template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::SetBatchPointers(GM_ADDR A, GM_ADDR x, GM_ADDR y)
{
    this->inAGM.SetGlobalBuffer((__gm__ T*)A);
    this->inxGM.SetGlobalBuffer((__gm__ T*)x);
    if constexpr (HSS_MODE) {
        this->inyFloatGM.SetGlobalBuffer((__gm__ float*)y);
        this->outFloatGM.SetGlobalBuffer((__gm__ float*)y);
    } else {
        this->inyGM.SetGlobalBuffer((__gm__ T*)y);
        this->outGM.SetGlobalBuffer((__gm__ T*)y);
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::ParseTilingData(const GemvBatchedTilingData& tdata)
{
    this->alpha = tdata.alpha;
    this->beta = tdata.beta;
    this->m = tdata.m;
    this->n = tdata.n;
    this->outDim = tdata.outSize;
    this->dotDim = tdata.dotSize;
    this->batchGroupSize = tdata.batchGroupSize;
    this->dotTile = tdata.dotTile;
    this->outTile = tdata.outTile;
    this->szInA = tdata.bufInA;
    this->szInx = tdata.bufInx;
    this->szInY = tdata.bufInY;
    this->szOut = tdata.bufOut;
    this->szMatTmp = tdata.bufMatTmp;
    this->szVecTmp = tdata.bufVecTmp;
    this->lda = tdata.lda;

    this->calMatNum = (this->vecIdx == tdata.usedCoreNum - 1) ? tdata.batchTail : tdata.batchPerCore;
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::InitUbuf()
{
    this->maxMatEleNum = this->outTile * this->dotTile;
    this->maxVecEleNum = (this->dotTile < 64u) ? 64u : this->dotTile;
    if constexpr (!TRANS_T) {
        if (this->outTile > this->maxVecEleNum)
            this->maxVecEleNum = this->outTile;
    }

    // Host-precomputed buffer sizes (256B-aligned)
    this->pipe.InitBuffer(this->inABuf, this->szInA);
    this->pipe.InitBuffer(this->inxBuf, this->szInx);
    this->pipe.InitBuffer(this->inyBuf, this->szInY);
    this->pipe.InitBuffer(this->outBuf, this->szOut);
    this->pipe.InitBuffer(this->matTmpBuf, this->szMatTmp);
    this->pipe.InitBuffer(this->vecTmpBuf, this->szVecTmp);

    if constexpr (!IS_FLOAT) {
        this->pipe.InitBuffer(this->matPreBuf, this->maxMatEleNum * sizeof(T));
        this->pipe.InitBuffer(this->vecPreBuf, this->maxVecEleNum * sizeof(T));
        this->matPreLocal = matPreBuf.Get<T>();
        this->vecPreLocal = vecPreBuf.Get<T>();
    }
    this->matTmpLocal = matTmpBuf.Get<float>();
    this->vecTmpLocal = vecTmpBuf.Get<float>();

    this->matFloatLocal = this->matTmpLocal[0];
    uint32_t yOff = ((this->outTile + VEC_FLOAT_PER_REPEAT - 1) & ~(VEC_FLOAT_PER_REPEAT - 1));
    if constexpr (IS_FLOAT) {
        this->mulResultLocal = this->matTmpLocal[0];
        this->sumResultLocal = this->vecTmpLocal[0];
        this->yInFloatLocal = this->vecTmpLocal[yOff];
    } else {
        this->mulResultLocal = this->matTmpLocal[this->maxMatEleNum];
        this->vecFloatLocal = this->vecTmpLocal[0];
        this->sumResultLocal = this->vecTmpLocal[this->dotTile];
        this->yInFloatLocal = this->vecTmpLocal[this->dotTile + yOff];
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyInMatAndVec(
    uint32_t curBatchId, uint32_t nOffset, uint32_t nCurr, uint32_t mOffset, uint32_t mCurr)
{
    LocalTensor<T> inALocal = inABuf.Get<T>();
    LocalTensor<T> inxLocal = inxBuf.Get<T>();

    // issue #860: 乘法在 uint64 域完成后求和（nOffset*lda、batchId*lda*n 超 2^32 时
    // uint32 算术先行回绕）
    uint64_t inABase = static_cast<uint64_t>(curBatchId) * static_cast<uint64_t>(this->lda) * this->n +
                       static_cast<uint64_t>(nOffset) * static_cast<uint64_t>(this->lda) + mOffset;

    // Align mCurr for dstStride only; DataCopyPad handles non-32B blockLen internally.
    uint32_t elemAlign = 32u / sizeof(T);
    uint32_t mCurrAligned = ((mCurr + elemAlign - 1) & ~(elemAlign - 1));
    uint32_t matSrcStride = uint32_t(((uint32_t)this->lda - mCurr) * sizeof(T));
    uint32_t matDstStride = uint32_t((this->dotTile - mCurrAligned) * sizeof(T) / 32);

    DataCopyExtParams cp{
        static_cast<uint16_t>(nCurr), static_cast<uint32_t>(mCurr * sizeof(T)), matSrcStride, matDstStride, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(inALocal, this->inAGM[inABase], cp, pp);

    uint64_t inxOffset = curBatchId * this->dotDim + mOffset;
    DataCopyPad(
        inxLocal, this->inxGM[inxOffset], DataCopyExtParams{1, static_cast<uint32_t>(mCurr * sizeof(T)), 0, 0, 0},
        DataCopyPadExtParams<T>{false, 0, 0, 0});
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyInY(
    uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr)
{
    uint64_t offset = curBatchId * this->outDim + mOffset;
    if constexpr (HSS_MODE) {
        auto inyLocal = inyBuf.Get<float>();
        DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(inyLocal, this->inyFloatGM[offset], copyParams, padParams);
    } else {
        LocalTensor<T> inyLocal = inyBuf.Get<T>();
        DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inyLocal, this->inyGM[offset], copyParams, padParams);
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::ApplyAlphaBeta(uint32_t curCalMatNum)
{
    uint32_t count = curCalMatNum;
    if (alpha != 1.0f)
        Muls(this->sumResultLocal, this->sumResultLocal, alpha, count);
    if (beta != 0.0f) {
        Muls(this->mulResultLocal, this->yInFloatLocal, beta, count);
        PipeBarrier<PIPE_V>();
        Add(this->sumResultLocal, this->sumResultLocal, this->mulResultLocal, count);
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CastleYHalfToFloat()
{
    LocalTensor<T> inyLocal = inyBuf.Get<T>();
    // Align to 32B to prevent DataCopy truncation (non-32B-aligned
    // byte counts lose trailing elements on this hardware).
    uint32_t cnt = ((this->mCurr + (32u / sizeof(T) - 1)) & ~(32u / sizeof(T) - 1));
    DataCopy(this->vecPreLocal, inyLocal, cnt);
    PipeBarrier<PIPE_V>();
    uint16_t castRepeatNum = (cnt - 1) / ELENUM_REPEAT_FP16 + 1;
    Cast(this->yInFloatLocal, this->vecPreLocal, RoundMode::CAST_NONE, castRepeatNum * ELENUM_REPEAT_FP16);
    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CastleHalfToFloat(uint32_t dotCnt)
{
    // inALocal rows are at r*dotTile (padded by DataCopyPad rightPadding).
    // Copy full rows including padding; ComputeDotProduct masks to dotCnt.
    LocalTensor<T> inALocal = inABuf.Get<T>();
    LocalTensor<T> inxLocal = inxBuf.Get<T>();

    uint32_t matRows = this->mCurr;
    uint16_t matCastRepeatNum = (matRows * this->dotTile - 1) / ELENUM_REPEAT_FP16 + 1;
    DataCopy(this->matPreLocal, inALocal, matRows * this->dotTile);

    DataCopy(this->vecPreLocal, inxLocal, this->dotTile);
    PipeBarrier<PIPE_V>();
    Cast(this->vecFloatLocal, this->vecPreLocal, RoundMode::CAST_NONE, this->dotTile);
    PipeBarrier<PIPE_V>();
    Cast(this->matFloatLocal, this->matPreLocal, RoundMode::CAST_NONE, matCastRepeatNum * ELENUM_REPEAT_FP16);
    PipeBarrier<PIPE_V>();
}
template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CastFloatToHalf()
{
    LocalTensor<T> outLocal = outBuf.Get<T>();
    constexpr RoundMode rm = IsSameType<T, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_ROUND;
    uint16_t blockNum = (this->mCurr - 1) / ELENUM_BLOCK_FP16 + 1;
    Cast(outLocal, this->sumResultLocal, rm, blockNum * ELENUM_BLOCK_FP16);
    PipeBarrier<PIPE_ALL>();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyOutRsltFloat(
    uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr)
{
    uint64_t outOffset = curBatchId * this->outDim + mOffset;
    DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(float)), 0, 0, 0};
    if constexpr (HSS_MODE)
        DataCopyPad(this->outFloatGM[outOffset], this->sumResultLocal, copyParams);
    else
        DataCopyPad(this->outGM[outOffset], this->sumResultLocal, copyParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::CopyOutRsltHalf(
    uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr)
{
    LocalTensor<T> outLocal = outBuf.Get<T>();
    uint64_t outOffset = curBatchId * this->outDim + mOffset;
    DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->outGM[outOffset], outLocal, copyParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::Process(uint32_t batchCountToProcess)
{
    uint32_t effectiveBatchCount = (batchCountToProcess > 0) ? batchCountToProcess : this->calMatNum;
    if (0 == effectiveBatchCount)
        return;

    uint32_t outTile = this->outTile;
    uint32_t dotTile = this->dotTile;
    uint32_t outDim = this->outDim;
    uint32_t dotDim = this->dotDim;

    for (uint32_t b = 0; b < effectiveBatchCount; b++) {
        for (uint32_t outOff = 0; outOff < outDim; outOff += outTile) {
            uint32_t outCnt = (outOff + outTile > outDim) ? (outDim - outOff) : outTile;
            // outCntAlign may exceed outTile by up to VEC_FLOAT_PER_REPEAT-1;
            // the excess is absorbed by round256-aligned vecTmpBuf allocation
            // and by the aligned yInFloatLocal offset. CopyOut uses outCnt
            // (the exact count), so no garbage is written to GM.
            uint32_t outCntAlign = ((outCnt + VEC_FLOAT_PER_REPEAT - 1) & ~(VEC_FLOAT_PER_REPEAT - 1));
            this->mCurr = outCnt;

            CopyInY(b, outOff, outCnt);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            bool isFirstDot = true;
            for (uint32_t dotOff = 0; dotOff < dotDim; dotOff += dotTile) {
                uint32_t dotCnt = (dotOff + dotTile > dotDim) ? (dotDim - dotOff) : dotTile;
                CopyInMatAndVec(b, outOff, outCnt, dotOff, dotCnt);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                if constexpr (!IS_FLOAT)
                    CastleHalfToFloat(dotCnt);

                ComputeDotProduct(outCnt, dotCnt, dotTile, isFirstDot);

                if (!isFirstDot)
                    Add(this->sumResultLocal, this->sumResultLocal, this->yInFloatLocal, outCntAlign);
                isFirstDot = false;
            }
            OutputMChunkAndSync(b, outOff, outCnt, outCntAlign);
        }
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::ComputeDotProduct(
    uint32_t outCnt, uint32_t dotCnt, uint32_t dotTile, bool isFirstDot)
{
    constexpr uint32_t VL = 256 / sizeof(float);
    uint32_t lM = outCnt;
    uint32_t lNTile = dotTile;
    auto& dstLT = isFirstDot ? this->sumResultLocal : this->yInFloatLocal;
    __ubuf__ float* aAddr = IS_FLOAT ? (__ubuf__ float*)inABuf.Get<float>().GetPhyAddr() :
                                       (__ubuf__ float*)this->matFloatLocal.GetPhyAddr();
    __ubuf__ float* xAddr = IS_FLOAT ? (__ubuf__ float*)inxBuf.Get<float>().GetPhyAddr() :
                                       (__ubuf__ float*)this->vecFloatLocal.GetPhyAddr();
    __ubuf__ float* dAddr = (__ubuf__ float*)dstLT.GetPhyAddr();
    uint16_t vLoopNum = (dotCnt + VL - 1) / VL;
    uint32_t tailLen = dotCnt % VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregA;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregX;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregMul;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregSum;
        auto maskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t row = 0; row < static_cast<uint16_t>(lM); row++) {
            AscendC::MicroAPI::Duplicate<float>(vregSum, 0.0f, maskAll);
            uint16_t matOff = row * static_cast<uint16_t>(lNTile);
            for (uint16_t i = 0; i < vLoopNum; i++) {
                uint32_t col = i * static_cast<uint32_t>(VL);
                uint32_t chunk = (i == vLoopNum - 1 && tailLen) ? tailLen : VL;
                auto mask = (chunk < VL) ?
                                AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(chunk) :
                                maskAll;
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregA, (__ubuf__ float*)(aAddr + matOff + col));
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregX, (__ubuf__ float*)(xAddr + col));
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregMul, vregA, vregX, mask);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    vregSum, vregSum, vregMul, maskAll);
            }
            AscendC::MicroAPI::ReduceSum(vregSum, vregSum, maskAll);
            __ubuf__ float* ptr = dAddr + row;
            AscendC::MicroAPI::DataCopy<
                float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ptr, vregSum, 1, maskAll);
        }
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvBatchedAIV<T, TRANS_T, HSS_MODE>::OutputMChunkAndSync(
    uint32_t b, uint32_t mOffset, uint32_t mCurr, uint32_t mCurrAlign)
{
    PipeBarrier<PIPE_V>();
    if constexpr (IS_FLOAT_OUT) {
        DataCopy(this->yInFloatLocal, inyBuf.Get<float>(), mCurrAlign);
        PipeBarrier<PIPE_V>();
    } else {
        CastleYHalfToFloat();
    }
    ApplyAlphaBeta(mCurrAlign);

    if constexpr (!IS_FLOAT && !HSS_MODE)
        CastFloatToHalf();

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if constexpr (IS_FLOAT_OUT)
        CopyOutRsltFloat(b, mOffset, mCurr);
    else
        CopyOutRsltHalf(b, mOffset, mCurr);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

// ============================================================
// SIMT — one thread per output element
// Column-major: A(row, col) = A[col * m + row]
// ============================================================
constexpr uint32_t GEMV_SIMT_MAX_THREADS = 2048;

// SIMT 路径的 dtype 装载/回写转换（CodeCheck 超大函数治理：抽出后 GemvSimt 的
// if-constexpr 分支内联进 helper，语义与拆分前完全一致），写法对齐
// gemm_strided_batched_ex_kernel.cpp 的 StridedToFloat/StridedFromFloat 先例。
template <typename T>
__simt_callee__ __aicore__ inline float GemvBatchedSimtToFloat(T value)
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return __bfloat162float(value);
    } else if constexpr (std::is_same<T, half>::value) {
        return __half2float(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T GemvBatchedSimtFromFloat(float value)
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return __float2bfloat16_rn_sat(value);
    } else if constexpr (std::is_same<T, half>::value) {
        return __float2half_rn_sat(value);
    } else {
        return static_cast<T>(value);
    }
}

template <typename T_IN, typename T_OUT, bool IS_TRANS>
__simt_vf__ __aicore__ LAUNCH_BOUND(GEMV_SIMT_MAX_THREADS) inline void GemvSimt(
    uint32_t m, uint32_t n, uint32_t numB, uint32_t startB, float alpha, float beta, int32_t lda, int32_t incx,
    int32_t incy, GM_ADDR AarrayBase, GM_ADDR xarrayBase, GM_ADDR yarrayBase)
{
    uint32_t outDim = IS_TRANS ? n : m, dotDim = IS_TRANS ? m : n;
    // issue #860: total/索引乘法在 64 位域完成后收窄（host 跨度校验已拒绝超限形状，
    // 此处双保险，对照 cgemv_batched_kernel.cpp 修复后写法）
    uint64_t total = static_cast<uint64_t>(numB) * outDim;
    for (uint64_t tid = threadIdx.x; tid < total; tid += blockDim.x) {
        uint32_t b = static_cast<uint32_t>(tid / outDim);
        uint32_t outIdx = static_cast<uint32_t>(tid % outDim);
        float acc = 0.0f;
        // Read per-batch pointers from pointer arrays (cublas-style batched interface)
        __gm__ T_IN* batchA = ReadPtrFromArray<T_IN>(AarrayBase, startB + b);
        __gm__ T_IN* batchX = ReadPtrFromArray<T_IN>(xarrayBase, startB + b);
        __gm__ T_OUT* batchY = ReadPtrFromArray<T_OUT>(yarrayBase, startB + b);
        int32_t aOff = 0;
        int64_t xOff = 0;
        int64_t yOff = 0;
        if (incx < 0)
            xOff += static_cast<int64_t>(dotDim - 1u) * (-static_cast<int64_t>(incx));
        if (incy < 0)
            yOff += static_cast<int64_t>(outDim - 1u) * (-static_cast<int64_t>(incy));
        for (uint32_t dotIdx = 0; dotIdx < dotDim; dotIdx++) {
            int64_t aIdx = IS_TRANS ? (static_cast<int64_t>(outIdx) * lda + static_cast<int64_t>(dotIdx)) :
                                      (static_cast<int64_t>(dotIdx) * lda + static_cast<int64_t>(outIdx));
            int64_t xIdx = static_cast<int64_t>(dotIdx) * incx;
            float av = GemvBatchedSimtToFloat(batchA[aOff + aIdx]);
            float xv = GemvBatchedSimtToFloat(batchX[xOff + xIdx]);
            acc += av * xv;
        }
        int64_t yIdx = static_cast<int64_t>(outIdx) * incy;
        float oldY = GemvBatchedSimtToFloat(batchY[yOff + yIdx]);
        float result = alpha * acc + beta * oldY;
        batchY[yOff + yIdx] = GemvBatchedSimtFromFloat<T_OUT>(result);
    }
}

// ============================================================
// Kernel entry helpers
// ============================================================
template <typename T_IN, typename T_OUT>
__aicore__ inline void DispatchNormal(
    const GemvBatchedTilingData& td, uint32_t nB, uint32_t sB, int32_t lda, int32_t incx, int32_t incy, GM_ADDR Aarray,
    GM_ADDR xarray, GM_ADDR yarray)
{
    asc_vf_call<GemvSimt<T_IN, T_OUT, false>>(
        dim3{GEMV_SIMT_MAX_THREADS, 1, 1}, td.m, td.n, nB, sB, td.alpha, td.beta, lda, incx, incy, Aarray, xarray,
        yarray);
}

template <typename T, bool HSS>
__aicore__ inline void DispatchTranspose(
    const GemvBatchedTilingData& td, uint32_t nB, uint32_t sB, int32_t lda, int32_t incx, int32_t incy, GM_ADDR Aarray,
    GM_ADDR xarray, GM_ADDR yarray)
{
    using TO = typename std::conditional<HSS, float, T>::type;
    if (td.incx == 1 && td.incy == 1) {
        GemvBatchedAIV<T, true, HSS> op;
        // Init once: parse tiling data and allocate UB buffers
        __gm__ T* batchA0 = ReadPtrFromAivArray<T>(Aarray, sB);
        __gm__ T* batchX0 = ReadPtrFromAivArray<T>(xarray, sB);
        __gm__ TO* batchY0 = ReadPtrFromAivArray<TO>(yarray, sB);
        op.Init(
            reinterpret_cast<GM_ADDR>(batchA0), reinterpret_cast<GM_ADDR>(batchX0), reinterpret_cast<GM_ADDR>(batchY0),
            td);
        op.Process(1);
        for (uint32_t b = 1; b < nB; b++) {
            __gm__ T* batchA = ReadPtrFromAivArray<T>(Aarray, sB + b);
            __gm__ T* batchX = ReadPtrFromAivArray<T>(xarray, sB + b);
            __gm__ TO* batchY = ReadPtrFromAivArray<TO>(yarray, sB + b);
            op.SetBatchPointers(
                reinterpret_cast<GM_ADDR>(batchA), reinterpret_cast<GM_ADDR>(batchX),
                reinterpret_cast<GM_ADDR>(batchY));
            op.Process(1);
        }
        return;
    }
    asc_vf_call<GemvSimt<T, TO, true>>(
        dim3{GEMV_SIMT_MAX_THREADS, 1, 1}, td.m, td.n, nB, sB, td.alpha, td.beta, lda, incx, incy, Aarray, xarray,
        yarray);
}

extern "C" __global__ __aicore__ void gemv_batched(
    GM_ADDR Aarray, GM_ADDR xarray, GM_ADDR yarray, GemvBatchedTilingData td)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t sB = GetBlockIdx() * td.batchPerCore;
    uint32_t nB = (GetBlockIdx() == td.usedCoreNum - 1) ? td.batchTail : td.batchPerCore;
    if (nB == 0)
        return;
    int32_t lda = td.lda, incx = td.incx, incy = td.incy;

#define D_NORM(T_IN, T_OUT) DispatchNormal<T_IN, T_OUT>(td, nB, sB, lda, incx, incy, Aarray, xarray, yarray)
#define D_TRANS(T, HSS) DispatchTranspose<T, HSS>(td, nB, sB, lda, incx, incy, Aarray, xarray, yarray)

    if (td.trans == 0) {
        if (td.dtype == 1)
            D_NORM(float, float);
        else if (td.dtype == 0)
            D_NORM(half, half);
        else if (td.dtype == 2)
            D_NORM(half, float);
        else if (td.dtype == 3)
            D_NORM(bfloat16_t, bfloat16_t);
        else
            D_NORM(bfloat16_t, float);
    } else {
        if (td.dtype == 1)
            D_TRANS(float, false);
        else if (td.dtype == 0)
            D_TRANS(half, false);
        else if (td.dtype == 2)
            D_TRANS(half, true);
        else if (td.dtype == 3)
            D_TRANS(bfloat16_t, false);
        else
            D_TRANS(bfloat16_t, true);
    }

#undef D_NORM
#undef D_TRANS
}

aclError gemv_batched_kernel_do(
    GM_ADDR A, GM_ADDR x, GM_ADDR y, const GemvBatchedTilingData& tiling, uint32_t numBlocks, void* stream)
{
    gemv_batched<<<numBlocks, nullptr, stream>>>(A, x, y, tiling);
    // issue #862: 返回发射阶段错误码，host 据此返回 EXECUTION_FAILED（对齐 chpmv 口径）
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
