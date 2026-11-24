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
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "scasum_tiling_data.h"
#include "scasum_kernel.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t ACC_LANE_NUM = 8;
constexpr uint32_t STRIDED_CHUNK_COMPLEX = 2048;

template <typename T>
class ScasumKernel {
public:
    __aicore__ inline ScasumKernel() {}
    __aicore__ inline void Init(GM_ADDR inDevice, GM_ADDR outDevice, const ScasumTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInDense(uint32_t floatOffset, uint32_t dataCount, uint32_t padNum);
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void CopyInStrided(uint32_t complexBase, uint32_t cnt);
    __aicore__ inline void WritePartial();

    TPipe pipe;

    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    TBuf<TPosition::VECCALC> reBuf;
    TBuf<TPosition::VECCALC> imBuf;
    TBuf<TPosition::VECCALC> pairBuf;

    int32_t incx = 1;
    uint32_t perCoreNum = 0;
    uint32_t remainder = 0;
    uint32_t myComplexOffset = 0;
    uint32_t myComplexNum = 0;
    uint32_t maxDataCount = 0;
};

template <typename T>
__aicore__ inline void ScasumKernel<T>::Init(GM_ADDR inDevice, GM_ADDR outDevice, const ScasumTilingData& tiling)
{
    this->incx = tiling.incx;
    this->perCoreNum = tiling.perCoreN;
    this->remainder = tiling.remainder;

    const uint32_t coreIdx = GetBlockIdx();
    this->myComplexOffset = coreIdx * this->perCoreNum + (coreIdx < this->remainder ? coreIdx : this->remainder);
    this->myComplexNum = this->perCoreNum + ((coreIdx < this->remainder) ? 1U : 0U);

    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inDevice));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outDevice));

    this->maxDataCount = SCASUM_PAD_CHUNK_FLOATS;

    constexpr int32_t typeSize = BYTENUM_PER_FLOAT32;
    constexpr int32_t elementsPerBlock = 32 / typeSize;
    constexpr int32_t elementsPerRepeat = 256 / typeSize;
    constexpr int32_t firstMaxRepeat =
        (static_cast<int32_t>(SCASUM_PAD_CHUNK_FLOATS) + elementsPerRepeat - 1) / elementsPerRepeat;
    constexpr int32_t iter1AlignEnd =
        (firstMaxRepeat + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    pipe.InitBuffer(workBuf, iter1AlignEnd * typeSize + 32);

    pipe.InitBuffer(inQueue, BUFFER_NUM, (SCASUM_PAD_CHUNK_FLOATS + ACC_LANE_NUM) * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, ACC_LANE_NUM * sizeof(T));
    pipe.InitBuffer(sumBuf, ACC_LANE_NUM * sizeof(T));
    pipe.InitBuffer(accBuf, ACC_LANE_NUM * sizeof(T));
    pipe.InitBuffer(idxBuf, STRIDED_CHUNK_COMPLEX * sizeof(int32_t));
    pipe.InitBuffer(reBuf, STRIDED_CHUNK_COMPLEX * sizeof(T));
    pipe.InitBuffer(imBuf, STRIDED_CHUNK_COMPLEX * sizeof(T));
    pipe.InitBuffer(pairBuf, (STRIDED_CHUNK_COMPLEX + ACC_LANE_NUM) * sizeof(T));

    LocalTensor<T> accLocal = accBuf.Get<T>(ACC_LANE_NUM);
    Duplicate<T>(accLocal, 0.0f, ACC_LANE_NUM);
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
    DataCopyPad(outGM, accLocal, copyParams);

    SyncAll();
}

template <typename T>
__aicore__ inline void ScasumKernel<T>::Process()
{
    SetAtomicAdd<T>();

    if (this->incx == 1) {
        const uint32_t baseF = static_cast<uint32_t>(2ULL * this->myComplexOffset);
        const uint32_t totalF = 2U * this->myComplexNum;
        uint32_t done = 0;
        while (done < totalF) {
            const uint32_t remainF = totalF - done;
            const uint32_t chunk = (remainF > this->maxDataCount) ? this->maxDataCount : remainF;
            const uint32_t padNum = (ACC_LANE_NUM - (chunk & (ACC_LANE_NUM - 1U))) & (ACC_LANE_NUM - 1U);
            this->CopyInDense(baseF + done, chunk, padNum);
            this->Compute(chunk + padNum);
            done += chunk;
        }
    } else {
        const uint64_t spanPerElement = 2ULL * static_cast<uint64_t>(this->incx);
        uint64_t capBySpan = (static_cast<uint64_t>(this->maxDataCount) - 2ULL) / spanPerElement + 1ULL;
        uint32_t complexPerChunk = static_cast<uint32_t>(capBySpan > STRIDED_CHUNK_COMPLEX ? STRIDED_CHUNK_COMPLEX
                                                                                           : capBySpan);
        if (complexPerChunk == 0) {
            complexPerChunk = 1;
        }
        uint32_t done = 0;
        while (done < this->myComplexNum) {
            const uint32_t remain = this->myComplexNum - done;
            const uint32_t cnt = (remain > complexPerChunk) ? complexPerChunk : remain;
            this->CopyInStrided(this->myComplexOffset + done, cnt);
            done += cnt;
        }
    }

    this->WritePartial();

    SetAtomicNone();
}

template <typename T>
__aicore__ inline void ScasumKernel<T>::CopyInDense(uint32_t floatOffset, uint32_t dataCount, uint32_t padNum)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataCount * static_cast<uint32_t>(sizeof(T))), 0, 0};
    // padValue 为位模式（uint64_t），0 即 float 0.0f
    DataCopyPadParams padParams{padNum > 0, 0, static_cast<uint8_t>(padNum), 0ULL};
    DataCopyPad(inLocal, inGM[floatOffset], copyParams, padParams);
    inQueue.EnQue<T>(inLocal);
}

template <typename T>
__aicore__ inline void ScasumKernel<T>::Compute(uint32_t dataCount)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> sumLocal = sumBuf.Get<T>();
    LocalTensor<T> accLocal = accBuf.Get<T>(ACC_LANE_NUM);

    Abs(inLocal, inLocal, dataCount);
    pipe_barrier(PIPE_V);
    ReduceSum<T>(sumLocal, inLocal, workLocal, dataCount);
    inQueue.FreeTensor(inLocal);

    Add(accLocal, accLocal, sumLocal, ACC_LANE_NUM);
}

// incx > 1：先稠密载入本核 chunk 的 float 跨度（span），再用 Gather 按字节索引压缩出
// 各复数的实部/虚部并求和。GatherMask 的 mask 以 32B 块为单位，任意 incx 无法构造统一
// mask，故采用"稠密载入 + Gather"的两步向量化（免逐元素标量访问）。
template <typename T>
__aicore__ inline void ScasumKernel<T>::CopyInStrided(uint32_t complexBase, uint32_t cnt)
{
    const uint64_t spanStart = static_cast<uint64_t>(complexBase) * static_cast<uint64_t>(this->incx) * 2ULL;
    const uint32_t spanFloats = static_cast<uint32_t>(
        (static_cast<uint64_t>(cnt) - 1ULL) * static_cast<uint64_t>(this->incx) * 2ULL + 2ULL);

    LocalTensor<T> inSpan = inQueue.AllocTensor<T>();
    DataCopyParams spanParams{1, static_cast<uint16_t>(spanFloats * static_cast<uint32_t>(sizeof(T))), 0, 0};
    DataCopyPadParams noPad{false, 0, 0, 0ULL};
    DataCopyPad(inSpan, inGM[spanStart], spanParams, noPad);
    inQueue.EnQue<T>(inSpan);

    LocalTensor<T> span = inQueue.DeQue<T>();
    LocalTensor<int32_t> idx = idxBuf.Get<int32_t>();
    LocalTensor<T> reLocal = reBuf.Get<T>();
    LocalTensor<T> imLocal = imBuf.Get<T>();
    LocalTensor<T> pairLocal = pairBuf.Get<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> sumLocal = sumBuf.Get<T>();
    LocalTensor<T> accLocal = accBuf.Get<T>(ACC_LANE_NUM);

    CreateVecIndex(idx, 0, static_cast<int32_t>(cnt));
    Muls(idx, idx, static_cast<int32_t>(2U * static_cast<uint32_t>(this->incx) * sizeof(T)), cnt);
    pipe_barrier(PIPE_V);

    Gather(reLocal, span, idx.ReinterpretCast<uint32_t>(), 0U, cnt);
    Gather(imLocal, span, idx.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(sizeof(T)), cnt);
    pipe_barrier(PIPE_V);

    Abs(pairLocal, reLocal, cnt);
    Abs(imLocal, imLocal, cnt);
    pipe_barrier(PIPE_V);
    Add(pairLocal, pairLocal, imLocal, cnt);

    const uint32_t padNum = (ACC_LANE_NUM - (cnt & (ACC_LANE_NUM - 1U))) & (ACC_LANE_NUM - 1U);
    for (uint32_t k = cnt; k < cnt + padNum; k++) {
        pairLocal.SetValue(k, static_cast<T>(0));
    }
    if (padNum > 0) {
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);
    }
    pipe_barrier(PIPE_V); // Add 与 ReduceSum 同属 V 流水，保证读取顺序

    ReduceSum<T>(sumLocal, pairLocal, workLocal, cnt + padNum);
    inQueue.FreeTensor(span);

    Add(accLocal, accLocal, sumLocal, ACC_LANE_NUM);
}

template <typename T>
__aicore__ inline void ScasumKernel<T>::WritePartial()
{
    // accBuf(V) → outQueue(MTE3) 的同步由 EnQue/DeQue 事件携带
    LocalTensor<T> accLocal = accBuf.Get<T>(ACC_LANE_NUM);
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
    Adds<T>(outLocal, accLocal, 0.0f, ACC_LANE_NUM);
    outQueue.EnQue<T>(outLocal);

    LocalTensor<T> outDeq = outQueue.DeQue<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
    DataCopyPad(outGM, outDeq, copyParams);
    outQueue.FreeTensor(outDeq);
}

__global__ __aicore__ void scasum_kernel(GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workSpace, ScasumTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ScasumKernel<float> op;
    op.Init(inGM, outGM, tiling);
    op.Process();
}

aclblasStatus_t scasum_kernel_do(
    uint8_t* inGM, uint8_t* outGM, uint8_t* workSpace, const ScasumTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);

    scasum_kernel<<<numBlocks, nullptr, aclStream>>>(inGM, outGM, workSpace, tiling);
    return ACLBLAS_STATUS_SUCCESS;
}
