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

using namespace AscendC;

namespace Nrm2 {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class Nrm2AIV {
public:
    __aicore__ inline Nrm2AIV(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void SetZeroBuf();
    __aicore__ inline void ClearGM(GlobalTensor<T> dstGM);
    __aicore__ inline void CopyIn(GlobalTensor<T> srcGM, uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(GlobalTensor<T> srcGM, uint32_t offset, uint32_t dataCount);
    __aicore__ inline float ComputeMax(uint32_t dataCount);
    __aicore__ inline float ComputeScaledSsq(uint32_t dataCount, float scale);
    __aicore__ inline void WriteWorkspace(float scale, float ssq);
    __aicore__ inline void PostProcess();
    __aicore__ inline void WriteResult(float globalScale, float globalSsq, float nanValue);

private:
    TPipe pipe;

    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    GlobalTensor<T> workGM;

    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> zeroBuf;
    TBuf<TPosition::VECCALC> summaryBuf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;

    uint32_t n = 0; // total elements num(float32)
    uint32_t useCoreNum = 0;
    uint32_t calNum = 0;
    uint32_t startOffset = 0;
    uint32_t maxDataCount = 0;
    uint32_t totalVecCoreNum = 40;
    float scaleLocal = 0.0f;
    float ssqLocal = 0.0f;
    int32_t vecIdx;
    int32_t blockNum;

    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void Nrm2AIV<T>::Init(GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->blockNum = GetBlockNum();
    this->vecIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    inGM.SetGlobalBuffer((__gm__ T*)x, this->n);
    outGM.SetGlobalBuffer((__gm__ T*)result, 1);

    workGM.SetGlobalBuffer((__gm__ T*)(workSpace), this->blockNum * 2);

    this->maxDataCount = 90 * 1024 / BYTENUM_PER_FLOAT32; // 90kb / 4b

    // Compute the min space size for ReduceSum
    int firstMaxRepeat = maxDataCount / elementsPerRepeat; // 376
    int iter1OutputCount = firstMaxRepeat;
    int iter1AlignEnd = (iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    int finalWorkLocalNeedSize = iter1AlignEnd;

    pipe.InitBuffer(workBuf, finalWorkLocalNeedSize * sizeof(T)); // 1504bytes
    pipe.InitBuffer(inQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(zeroBuf, UB_BYTENUM_PER_BLOCK);               // 32bytes
    pipe.InitBuffer(summaryBuf, UB_BYTENUM_PER_BLOCK);

    // Set zero for ubuf
    SetZeroBuf();
    // Clear workspace
    ClearGM(workGM[this->vecIdx * 2]);
    ClearGM(workGM[this->vecIdx * 2 + 1]);

    return;
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::SetZeroBuf()
{
    LocalTensor<T> zeroLocal = zeroBuf.Get<T>();
    Duplicate<T>(zeroLocal, 0, this->elementsPerBlock);

    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::ClearGM(GlobalTensor<T> dstGM)
{
    LocalTensor<T> zeroLocal = zeroBuf.Get<T>();
    DataCopyExtParams copyParams{1, sizeof(T), 0, 0, 0};
    DataCopyPad(dstGM, zeroLocal, copyParams);

    AscendC::PipeBarrier<PIPE_MTE3>();
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t*>(tilingGm);

    this->n = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingBuf));
    this->useCoreNum = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingBuf + sizeof(uint32_t)));
    this->startOffset =
        (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingBuf + 2 * sizeof(uint32_t) + sizeof(uint32_t) * this->vecIdx));
    this->calNum = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingBuf + 2 * sizeof(uint32_t) +
                                        this->totalVecCoreNum * sizeof(uint32_t) + sizeof(uint32_t) * this->vecIdx));
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::PostProcess()
{
    CopyInPad(workGM, 0, this->blockNum * 2);
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

    constexpr float MAX_FLOAT = 3.402823466e+38F;
    float globalScale = 0.0f;
    for (int32_t i = 0; i < this->blockNum; i++) {
        float scale = inLocal.GetValue(i * 2);
        if (scale != scale) {
            globalScale = scale;
            break;
        }
        if (scale > globalScale) {
            globalScale = scale;
        }
    }

    float nanValue = 0.0f;
    for (int32_t i = 0; i < this->blockNum; i++) {
        float scale = inLocal.GetValue(i * 2);
        float ssq = inLocal.GetValue(i * 2 + 1);
        if (scale != scale) {
            nanValue = scale;
            break;
        }
        if (ssq != ssq) {
            nanValue = ssq;
        }
    }

    float globalSsq = 0.0f;
    if (nanValue == nanValue && globalScale > 0.0f && globalScale <= MAX_FLOAT) {
        for (int32_t i = 0; i < this->blockNum; i++) {
            float scale = inLocal.GetValue(i * 2);
            float ssq = inLocal.GetValue(i * 2 + 1);
            float ratio = scale / globalScale;
            globalSsq += ssq * ratio * ratio;
        }
    }
    inQueue.FreeTensor(inLocal);
    WriteResult(globalScale, globalSsq, nanValue);
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::WriteResult(float globalScale, float globalSsq, float nanValue)
{
    constexpr float MAX_FLOAT = 3.402823466e+38F;
    LocalTensor<T> outLocal = summaryBuf.Get<T>();
    if (nanValue != nanValue) {
        outLocal.SetValue(0, nanValue);
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    } else if (globalScale <= MAX_FLOAT) {
        outLocal.SetValue(0, globalSsq);
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);

        Sqrt(outLocal, outLocal, this->elementsPerBlock);
        Muls(outLocal, outLocal, globalScale, this->elementsPerBlock);
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    } else {
        outLocal.SetValue(0, globalScale);
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

    DataCopyExtParams copyParams{1, sizeof(T), 0, 0, 0};
    DataCopyPad(outGM, outLocal, copyParams);
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::Process()
{
    uint32_t currOffset = this->startOffset;
    uint32_t repeatTimes = this->calNum / this->maxDataCount;
    uint32_t remainNum = this->calNum % this->maxDataCount;

    // Pass 1: find a per-core scale before any input element is squared.
    this->scaleLocal = 0.0f;
    for (uint32_t i = 0; i < repeatTimes; i++) {
        CopyIn(inGM, currOffset, this->maxDataCount);
        float tileMax = ComputeMax(this->maxDataCount);
        if (tileMax != tileMax) {
            this->scaleLocal = tileMax;
        } else if (this->scaleLocal == this->scaleLocal && tileMax > this->scaleLocal) {
            this->scaleLocal = tileMax;
        }
        currOffset += this->maxDataCount;
    }
    if (remainNum > 0) {
        CopyInPad(inGM, currOffset, remainNum);
        uint32_t alignedCount =
            (remainNum + this->elementsPerBlock - 1) / this->elementsPerBlock * this->elementsPerBlock;
        float tileMax = ComputeMax(alignedCount);
        if (tileMax != tileMax) {
            this->scaleLocal = tileMax;
        } else if (this->scaleLocal == this->scaleLocal && tileMax > this->scaleLocal) {
            this->scaleLocal = tileMax;
        }
    }

    // Pass 2: sum squares after scaling; each partial stays in the range [0, calNum].
    this->ssqLocal = 0.0f;
    if (this->scaleLocal != 0.0f) {
        currOffset = this->startOffset;
        for (uint32_t i = 0; i < repeatTimes; i++) {
            CopyIn(inGM, currOffset, this->maxDataCount);
            this->ssqLocal += ComputeScaledSsq(this->maxDataCount, this->scaleLocal);
            currOffset += this->maxDataCount;
        }
        if (remainNum > 0) {
            CopyInPad(inGM, currOffset, remainNum);
            uint32_t alignedCount =
                (remainNum + this->elementsPerBlock - 1) / this->elementsPerBlock * this->elementsPerBlock;
            this->ssqLocal += ComputeScaledSsq(alignedCount, this->scaleLocal);
        }
    }

    WriteWorkspace(this->scaleLocal, this->ssqLocal);
    SyncAll();

    if (this->vecIdx == 0) {
        PostProcess();
    }

    return;
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::CopyIn(GlobalTensor<T> srcGM, uint32_t offset, uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, srcGM[offset], dataCount);
    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::CopyInPad(GlobalTensor<T> srcGM, uint32_t offset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;

    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, srcGM[offset], copyParams, padParams);

    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline float Nrm2AIV<T>::ComputeMax(uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> outLocal = summaryBuf.Get<T>();

    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    Abs(inLocal, inLocal, static_cast<int32_t>(dataCount));
    AscendC::PipeBarrier<PIPE_V>();
    ReduceMax<T>(outLocal, inLocal, workLocal, static_cast<int32_t>(dataCount), false);

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    float tileMax = static_cast<float>(outLocal.GetValue(0));

    inQueue.FreeTensor(inLocal);
    return tileMax;
}

template <typename T>
__aicore__ inline float Nrm2AIV<T>::ComputeScaledSsq(uint32_t dataCount, float scale)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> outLocal = summaryBuf.Get<T>();

    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    constexpr float MAX_FLOAT = 3.402823466e+38F;
    if (scale > MAX_FLOAT) {
        // Preserve the existing IEEE result for inputs containing infinity.
        Mul(inLocal, inLocal, inLocal, static_cast<int32_t>(dataCount));
    } else {
        LocalTensor<T> scaleBuf = inQueue.AllocTensor<T>();
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);
        Duplicate<T>(scaleBuf, static_cast<T>(scale), static_cast<int32_t>(dataCount));
        AscendC::PipeBarrier<PIPE_V>();
        Div<T>(inLocal, inLocal, scaleBuf, static_cast<int32_t>(dataCount));
        AscendC::PipeBarrier<PIPE_V>();
        Mul(inLocal, inLocal, inLocal, static_cast<int32_t>(dataCount));
        inQueue.FreeTensor(scaleBuf);
    }
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSum<T>(outLocal, inLocal, workLocal, static_cast<int32_t>(dataCount));

    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    float tileSsq = static_cast<float>(outLocal.GetValue(0));

    inQueue.FreeTensor(inLocal);
    return tileSsq;
}

template <typename T>
__aicore__ inline void Nrm2AIV<T>::WriteWorkspace(float scale, float ssq)
{
    LocalTensor<T> summaryLocal = summaryBuf.Get<T>();
    summaryLocal.SetValue(0, scale);
    summaryLocal.SetValue(1, ssq);

    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

    DataCopyExtParams copyParams{1, 2 * sizeof(T), 0, 0, 0};
    DataCopyPad(workGM[this->vecIdx * 2], summaryLocal, copyParams);
    AscendC::PipeBarrier<PIPE_MTE3>();
}
} // namespace Nrm2

__global__ __aicore__ void snrm2_kernel(GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Nrm2::Nrm2AIV<float> op;
    op.Init(x, result, workSpace, tilingGm);
    op.Process();
}

void snrm2_kernel_do(GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    snrm2_kernel<<<numBlocks, nullptr, stream>>>(x, result, workSpace, tilingGm);
}
