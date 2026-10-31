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
#include <cfloat>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/index_reduce_arch35.h"
#include "common/helper/kernel_constant.h"
#include "isamin_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t ELEMENTS_PER_BLOCK = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
constexpr uint32_t REDUCE_REPEAT_BYTES = 256;
constexpr uint32_t ELEMENTS_PER_REPEAT = REDUCE_REPEAT_BYTES / BYTENUM_PER_FLOAT32;

class IsaminAIVBase {
protected:
    TPipe* pipe_;
    TBuf<TPosition::VECIN> inBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    GlobalTensor<float> inGM;
    int32_t blockIdx;
    uint32_t computeNum;
    uint32_t startOffset;
    uint32_t maxDataCount;
    float bestVal_;
    uint32_t bestIdx_;
    bool hasValue_;

    __aicore__ inline void InitBase(TPipe* pipe, GM_ADDR inDevice, const IsaminTilingData& tdata)
    {
        pipe_ = pipe;
        blockIdx = GetBlockIdx();
        uint32_t blockId = static_cast<uint32_t>(blockIdx);
        startOffset = blockId * tdata.perCoreN;
        computeNum = (blockId == tdata.useCoreNum - 1) ? tdata.lastCoreN : tdata.perCoreN;

        inGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inDevice), tdata.totalN);
        maxDataCount = tdata.tileSize;

        int elementsPerRepeat = REDUCE_REPEAT_BYTES / sizeof(float);
        int level1RepeatCnt = (maxDataCount + elementsPerRepeat - 1) / elementsPerRepeat;
        int level1AlignEnd = (level1RepeatCnt + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
        uint32_t workBufByteLen = level1AlignEnd * sizeof(float);

        pipe_->InitBuffer(inBuf_, maxDataCount * sizeof(float));
        pipe_->InitBuffer(workBuf_, workBufByteLen + UB_BYTENUM_PER_BLOCK);
        pipe_->InitBuffer(outBuf_, UB_BYTENUM_PER_BLOCK);

        bestVal_ = FLT_MAX;
        bestIdx_ = 0;
        hasValue_ = false;
    }

    __aicore__ inline void ProcessFullTiles()
    {
        uint32_t repeatTimes = computeNum / maxDataCount;
        uint32_t localStartOffset = startOffset;
        uint32_t localMaxDataCount = maxDataCount;
        float bestVal = bestVal_;
        uint32_t bestIdx = bestIdx_;
        bool hasValue = hasValue_;

        for (uint32_t i = 0; i < repeatTimes; i++) {
            uint32_t currOffset = localStartOffset + i * localMaxDataCount;

            LocalTensor<float> inLocal = inBuf_.Get<float>();
            DataCopy(inLocal, inGM[currOffset], localMaxDataCount);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            LocalTensor<float> workLocal = workBuf_.Get<float>();
            LocalTensor<float> outLocal = outBuf_.Get<float>();

            Abs(inLocal, inLocal, localMaxDataCount);
            ReduceMin(outLocal, inLocal, workLocal, localMaxDataCount, true);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);

            float tileMinVal = outLocal.GetValue(0);
            float idxFloat = outLocal.GetValue(1);
            uint32_t tileLocalIdx = *reinterpret_cast<uint32_t*>(&idxFloat);
            uint32_t tileGlobalIdx = localStartOffset + i * localMaxDataCount + tileLocalIdx;

            if (!(tileMinVal != tileMinVal) &&
                (!hasValue || tileMinVal < bestVal || (tileMinVal == bestVal && tileGlobalIdx < bestIdx))) {
                bestVal = tileMinVal;
                bestIdx = tileGlobalIdx;
                hasValue = true;
            }
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }

        bestVal_ = bestVal;
        bestIdx_ = bestIdx;
        hasValue_ = hasValue;
    }

    __aicore__ inline void ProcessRemainder()
    {
        uint32_t repeatTimes = computeNum / maxDataCount;
        uint32_t remainNum = computeNum % maxDataCount;
        if (remainNum == 0) {
            return;
        }

        uint32_t currOffset = startOffset + repeatTimes * maxDataCount;

        DataCopyParams copyParams{1, static_cast<uint16_t>(remainNum * sizeof(float)), 0, 0};
        uint8_t paddingNum =
            static_cast<uint8_t>((ELEMENTS_PER_BLOCK - remainNum % ELEMENTS_PER_BLOCK) % ELEMENTS_PER_BLOCK);
        DataCopyPadParams padParams{true, 0, paddingNum, 0x7F800000};

        LocalTensor<float> inLocal = inBuf_.Get<float>();
        DataCopyPad(inLocal, inGM[currOffset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        uint32_t alignedRemain = ((remainNum + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;

        LocalTensor<float> workLocal = workBuf_.Get<float>();
        LocalTensor<float> outLocal = outBuf_.Get<float>();

        Abs(inLocal, inLocal, alignedRemain);
        ReduceMin(outLocal, inLocal, workLocal, alignedRemain, true);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);

        float tileMinVal = outLocal.GetValue(0);
        float idxFloat = outLocal.GetValue(1);
        uint32_t tileLocalIdx = *reinterpret_cast<uint32_t*>(&idxFloat);
        uint32_t tileGlobalIdx = startOffset + repeatTimes * maxDataCount + tileLocalIdx;

        if (!(tileMinVal != tileMinVal) &&
            (!hasValue_ || tileMinVal < bestVal_ || (tileMinVal == bestVal_ && tileGlobalIdx < bestIdx_))) {
            bestVal_ = tileMinVal;
            bestIdx_ = tileGlobalIdx;
            hasValue_ = true;
        }
    }
};

class IsaminAIV : public IsaminAIVBase {
public:
    __aicore__ inline IsaminAIV() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR inDevice, GM_ADDR wsDevice, const IsaminTilingData& tdata);
    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteWorkspace();
    GlobalTensor<float> wsGM;
};

__aicore__ inline void IsaminAIV::Init(TPipe* pipe, GM_ADDR inDevice, GM_ADDR wsDevice, const IsaminTilingData& tdata)
{
    InitBase(pipe, inDevice, tdata);
    wsGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsDevice), tdata.useCoreNum * 2);
}

__aicore__ inline void IsaminAIV::Process()
{
    if (computeNum == 0) {
        return;
    }
    ProcessFullTiles();
    ProcessRemainder();
    WriteWorkspace();
}

__aicore__ inline void IsaminAIV::WriteWorkspace()
{
    WriteIndexReduceWorkspace(
        outBuf_.Get<float>(), wsGM, static_cast<uint32_t>(blockIdx), bestVal_, bestIdx_);
}

class IsaminAIVSmall : public IsaminAIVBase {
public:
    __aicore__ inline IsaminAIVSmall() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR inDevice, GM_ADDR resultDevice, const IsaminTilingData& tdata);
    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteResult();
    GlobalTensor<int32_t> resultGM;
};

__aicore__ inline void IsaminAIVSmall::Init(
    TPipe* pipe, GM_ADDR inDevice, GM_ADDR resultDevice, const IsaminTilingData& tdata)
{
    InitBase(pipe, inDevice, tdata);
    resultGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(resultDevice), 1);
}

__aicore__ inline void IsaminAIVSmall::Process()
{
    if (computeNum == 0) {
        return;
    }
    ProcessFullTiles();
    ProcessRemainder();
    WriteResult();
}

__aicore__ inline void IsaminAIVSmall::WriteResult()
{
    int32_t result = static_cast<int32_t>(bestIdx_) + 1;

    LocalTensor<int32_t> outLocal = outBuf_.Get<int32_t>();
    Duplicate<int32_t>(outLocal, result, ELEMENTS_PER_BLOCK);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    DataCopyParams writeParams{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    DataCopyPad(resultGM, outLocal, writeParams);
}

__simt_callee__ __aicore__ inline void IsaminSimtTreeReduce(
    __ubuf__ float* ubPartialVals, __ubuf__ uint32_t* ubPartialIdxs)
{
    unsigned int blockPow2 = 1;
    while (blockPow2 < blockDim.x) {
        blockPow2 <<= 1;
    }

    for (unsigned int s = blockPow2 >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
            float otherVal = ubPartialVals[threadIdx.x + s];
            uint32_t otherIdx = ubPartialIdxs[threadIdx.x + s];
            if (!(otherVal != otherVal) &&
                (otherVal < ubPartialVals[threadIdx.x] ||
                 (otherVal == ubPartialVals[threadIdx.x] && otherIdx < ubPartialIdxs[threadIdx.x]))) {
                ubPartialVals[threadIdx.x] = otherVal;
                ubPartialIdxs[threadIdx.x] = otherIdx;
            }
        }
        asc_syncthreads();
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void IsaminSimtCompute(
    uint32_t calNum, uint32_t blockStartOffset, uint32_t stride, __gm__ const float* xGm, __gm__ float* wsSlotPtr)
{
    if (calNum == 0) {
        return;
    }

    __ubuf__ float ubPartialVals[SIMT_MAX_THREAD_NUM];
    __ubuf__ uint32_t ubPartialIdxs[SIMT_MAX_THREAD_NUM];

    float bestVal = FLT_MAX;
    uint32_t bestIdx = 0;
    bool hasValue = false;

    for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
        float xVal = xGm[(blockStartOffset + i) * stride];
        float absVal = (xVal >= 0.0f) ? xVal : -xVal;
        if (!(absVal != absVal) &&
            (!hasValue || absVal < bestVal || (absVal == bestVal && (blockStartOffset + i) < bestIdx))) {
            bestVal = absVal;
            bestIdx = blockStartOffset + i;
            hasValue = true;
        }
    }

    ubPartialVals[threadIdx.x] = bestVal;
    ubPartialIdxs[threadIdx.x] = bestIdx;
    asc_syncthreads();

    IsaminSimtTreeReduce(ubPartialVals, ubPartialIdxs);

    if (threadIdx.x == 0) {
        wsSlotPtr[0] = ubPartialVals[0];
        reinterpret_cast<__gm__ uint32_t*>(wsSlotPtr + 1)[0] = ubPartialIdxs[0];
    }
}

extern "C" __global__ __aicore__ void isamin_aiv_kernel(GM_ADDR x, GM_ADDR workSpace, IsaminTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IsaminAIV op;
    op.Init(&pipe, x, workSpace, tdata);
    op.Process();
}

extern "C" __global__ __aicore__ void isamin_small_kernel(GM_ADDR x, GM_ADDR result, IsaminTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IsaminAIVSmall op;
    op.Init(&pipe, x, result, tdata);
    op.Process();
}

extern "C" __global__ __aicore__ void isamin_simt_kernel(GM_ADDR x, GM_ADDR workSpace, IsaminTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockIdx = GetBlockIdx();
    uint32_t blockId = static_cast<uint32_t>(blockIdx);
    uint32_t calNum = (blockId == tdata.useCoreNum - 1) ? tdata.lastCoreN : tdata.perCoreN;
    uint32_t startOffset = blockId * tdata.perCoreN;

    if (calNum > 0) {
        uint32_t wsBase = blockId * 2;
        __gm__ float* wsSlotPtr = reinterpret_cast<__gm__ float*>(workSpace) + wsBase;
        asc_vf_call<IsaminSimtCompute>(
            dim3{tdata.nthreads, 1, 1}, calNum, startOffset, tdata.incx, reinterpret_cast<__gm__ const float*>(x),
            wsSlotPtr);
    }
}

extern "C" __global__ __aicore__ void isamin_reduce_kernel(GM_ADDR workSpace, GM_ADDR resultGM, IsaminTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    uint32_t useCoreNum = tdata.useCoreNum;
    uint32_t totalFloats = useCoreNum * 2;
    constexpr uint32_t ALIGN_FLOATS = ELEMENTS_PER_REPEAT;
    uint32_t alignedFloats = ((totalFloats + ALIGN_FLOATS - 1) / ALIGN_FLOATS) * ALIGN_FLOATS;

    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, alignedFloats * sizeof(float));
    LocalTensor<float> wsData = buf.Get<float>();

    GlobalTensor<float> wsGM;
    wsGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), alignedFloats);
    DataCopy(wsData, wsGM, alignedFloats);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);

    float bestVal = FLT_MAX;
    uint32_t bestIdx = 0;
    bool hasValue = false;

    for (uint32_t i = 0; i < useCoreNum; i++) {
        float val = wsData.GetValue(i * 2);
        float idxFloat = wsData.GetValue(i * 2 + 1);
        uint32_t idx = *reinterpret_cast<uint32_t*>(&idxFloat);

        if (!(val != val) && (!hasValue || val < bestVal || (val == bestVal && idx < bestIdx))) {
            bestVal = val;
            bestIdx = idx;
            hasValue = true;
        }
    }

    if (!hasValue) {
        bestIdx = 0;
    }

    int32_t result = static_cast<int32_t>(bestIdx) + 1;

    TBuf<TPosition::VECCALC> outBuf;
    pipe.InitBuffer(outBuf, UB_BYTENUM_PER_BLOCK);
    LocalTensor<int32_t> outLocal = outBuf.Get<int32_t>();
    Duplicate<int32_t>(outLocal, result, ELEMENTS_PER_BLOCK);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    GlobalTensor<int32_t> outGM;
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(resultGM), 1);
    DataCopyParams writeParams{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    DataCopyPad(outGM, outLocal, writeParams);
}

void isamin_kernel_do(
    GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, const IsaminTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);

    if (tiling.incx == 1) {
        if (numBlocks == 1 && tiling.lastCoreN <= FP32_MAX_DATA_COUNT) {
            isamin_small_kernel<<<1, nullptr, aclStream>>>(x, result, tiling);
        } else {
            isamin_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(x, workSpace, tiling);
            isamin_reduce_kernel<<<1, nullptr, aclStream>>>(workSpace, result, tiling);
        }
    } else {
        isamin_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, workSpace, tiling);
        isamin_reduce_kernel<<<1, nullptr, aclStream>>>(workSpace, result, tiling);
    }
}
