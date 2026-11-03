/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCOPY_KERNEL_H
#define SCOPY_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class CopyAIV {
public:
    __aicore__ inline CopyAIV(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void SingleIteration(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void SingleIterationAligned(uint32_t curOffset, uint32_t dataCount);

private:
    TPipe pipe;

    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;

    int32_t blockNum;
    int32_t vecIdx;

    uint32_t totalVecCoreNum = 40;
    uint32_t maxDataCount = 0;
    uint32_t startOffset = 0;
    uint32_t calNum = 0;
    uint32_t useCoreNum = 0;
    uint32_t n = 0; // total elements num(float32)

    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / BYTENUM_PER_FLOAT32;
    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void CopyAIV<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->blockNum = GetBlockNum();
    this->vecIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    inGM.SetGlobalBuffer((__gm__ T*)x, this->n);
    outGM.SetGlobalBuffer((__gm__ T*)y, this->n);

    this->maxDataCount = 90 * 1024 / BYTENUM_PER_FLOAT32; // 90kb / 4b

    pipe.InitBuffer(inQueue, BUFFER_NUM, maxDataCount * sizeof(T));

    return;
}

template <typename T>
__aicore__ inline void CopyAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tilingAddr = reinterpret_cast<__gm__ uint8_t*>(tilingGm);

    this->n = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingAddr));
    this->useCoreNum = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingAddr + sizeof(uint32_t)));
    this->startOffset =
        (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingAddr + sizeof(uint32_t) * this->vecIdx + 2 * sizeof(uint32_t)));
    this->calNum = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tilingAddr + this->totalVecCoreNum * sizeof(uint32_t) +
                                        sizeof(uint32_t) * this->vecIdx + 2 * sizeof(uint32_t)));
}

template <typename T>
__aicore__ inline void CopyAIV<T>::SingleIteration(uint32_t curOffset, uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, inGM[curOffset], dataCount);
    inQueue.EnQue<T>(inLocal);
    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    LocalTensor<T> outLocal = inQueue.DeQue<T>();
    DataCopy(outGM[curOffset], outLocal, dataCount);
    inQueue.FreeTensor(outLocal);
    int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void CopyAIV<T>::SingleIterationAligned(uint32_t curOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, inGM[curOffset], copyParams, padParams);
    inQueue.EnQue<T>(inLocal);

    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

    LocalTensor<T> outLocal = inQueue.DeQue<T>();
    DataCopyPad(outGM[curOffset], outLocal, copyParams);
    inQueue.FreeTensor(outLocal);
    int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void CopyAIV<T>::Process()
{
    if (this->calNum <= 0) {
        return;
    }

    // 连续路径：按 float 元素分核拷贝
    uint32_t repeatTimes = this->calNum / this->maxDataCount;
    uint32_t remainNum = this->calNum % this->maxDataCount;

    uint32_t curOffset = this->startOffset;

    for (uint32_t i = 0; i < repeatTimes; i++) {
        SingleIteration(curOffset, this->maxDataCount);
        curOffset += this->maxDataCount;
    }

    if (remainNum > 0) {
        SingleIterationAligned(curOffset, remainNum);
    }
    return;
}

__global__ __aicore__ void scopy_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CopyAIV<float> op;
    op.Init(x, y, workSpace, tilingGm);
    op.Process();
}

void scopy_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    scopy_kernel<<<numBlocks, nullptr, stream>>>(x, y, nullptr, tilingGm);
}
#endif // COPY_AIV_H