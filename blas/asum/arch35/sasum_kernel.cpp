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
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/kernel_utils.h"
#include "sasum_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;

// All UB buffer allocations and Vector/MTE data transfers must be 32B-aligned.
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t ELEMENTS_PER_BLOCK = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;

// ReduceSum Level-1 processes data in 256B repeats (64 fp32 elements per repeat),
// producing one intermediate result per repeat. These intermediates are stored in
// the caller-provided sharedTmpBuffer and must be 32B-aligned.
constexpr uint32_t REDUCE_REPEAT_BYTES = 256;
constexpr uint32_t ELEMENTS_PER_REPEAT = REDUCE_REPEAT_BYTES / BYTENUM_PER_FLOAT32;

// SAFETY_MARGIN 为 TPipe/TQue 运行时元数据预留 32KB（字节）。
// UB 分配构成（UB_SIZE = 248×1024 = 253952 byte）：
//   inQueue  : BUFFER_NUM=2 × maxDataCount × sizeof(float)
//   outQueue : BUFFER_NUM=2 × UB_BYTENUM_PER_BLOCK = 64 byte
//   workBuf  : ceil(maxDataCount / ELEMENTS_PER_REPEAT) 按 ELEMENTS_PER_BLOCK 对齐 × sizeof(float)
//            + UB_BYTENUM_PER_BLOCK
//
// 约束方程（设 M = maxDataCount，单位 float）：
//   SAFETY_MARGIN + 8M + 64 + ((ceil(M/64) + 7) / 8 * 8) * 4 + 32 ≤ UB_SIZE
//   → 8M + ((ceil(M/64) + 7) / 8 * 8) * 4 ≤ 221088
// 数值求解 M ≤ 27416。为与 REDUCE_REPEAT_BYTES（256B）对齐，取 64 的倍数得 27392。
constexpr uint32_t SAFETY_MARGIN = 32 * 1024;
constexpr uint32_t UB_MAX_CHUNK_FLOATS = 27392;

class SasumAIV {
public:
    __aicore__ inline SasumAIV() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR inGM, GM_ADDR outGM, const SasumTilingData& tdata);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const SasumTilingData& tdata);

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void CopyOut();

    __aicore__ inline void SingleIteration(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void SingleIterationAligned(uint32_t offset, uint32_t dataCount);

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> workBuf;

    GlobalTensor<float> inGM;
    GlobalTensor<float> outGM;

    int32_t blockIdx;
    int32_t blockNum;

    uint32_t n;
    uint32_t computeNum;
    uint32_t startOffset;
    uint32_t maxDataCount;
};

__aicore__ inline void SasumAIV::ParseTilingData(const SasumTilingData& tdata)
{
    this->n = static_cast<uint32_t>(tdata.n);
    this->startOffset = tdata.startOffset[this->blockIdx];
    this->computeNum = tdata.calNum[this->blockIdx];
}

__aicore__ inline void SasumAIV::Init(TPipe* pipe, GM_ADDR inDevice, GM_ADDR outDevice, const SasumTilingData& tdata)
{
    pipe_ = pipe;
    this->blockIdx = GetBlockIdx();
    this->blockNum = GetBlockNum();

    ParseTilingData(tdata);

    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inDevice), this->n);
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outDevice), 1);

    maxDataCount = UB_MAX_CHUNK_FLOATS;

    // Work buffer for ReduceSum: Level-1 processes up to maxDataCount elements
    // in REDUCE_REPEAT_BYTES-sized repeats, producing ceil(maxDataCount / ELEMENTS_PER_REPEAT)
    // intermediate results. Those must be 32B-aligned in the sharedTmpBuffer.
    int elementsPerRepeat = REDUCE_REPEAT_BYTES / sizeof(float);
    int level1RepeatCnt = (maxDataCount + elementsPerRepeat - 1) / elementsPerRepeat;
    int level1OutputCount = level1RepeatCnt;
    int level1AlignEnd = (level1OutputCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
    uint32_t workBufByteLen = level1AlignEnd * sizeof(float);

    // Init buffers: workBuf + inQueue(2×) + outQueue(2×)
    // Extra UB_BYTENUM_PER_BLOCK padding on workBuf to satisfy MTE alignment.
    pipe_->InitBuffer(workBuf, workBufByteLen + UB_BYTENUM_PER_BLOCK);
    pipe_->InitBuffer(inQueue, BUFFER_NUM, maxDataCount * sizeof(float));
    // outQueue: ReduceSum produces a scalar, but MTE requires at least 32B for transfer.
    pipe_->InitBuffer(outQueue, BUFFER_NUM, UB_BYTENUM_PER_BLOCK);

    // Zero-initialize output GM (required for subsequent atomic add accumulation)
    LocalTensor<float> workLocal = workBuf.Get<float>(ELEMENTS_PER_BLOCK);
    Duplicate<float>(workLocal, 0.0f, ELEMENTS_PER_BLOCK);
    DataCopyParams zeroParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
    DataCopyPad(outGM, workLocal, zeroParams);
}

__aicore__ inline void SasumAIV::Process()
{
    if (this->computeNum == 0) {
        return;
    }

    SetAtomicAdd<float>();

    uint32_t repeatTimes = computeNum / maxDataCount;
    uint32_t remainNum = computeNum % maxDataCount;
    uint32_t maxCopyPadNum = UINT16_MAX / sizeof(float) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;

    uint32_t currOffset = startOffset;
    for (uint32_t i = 0; i < repeatTimes; i++) {
        SingleIteration(currOffset, maxDataCount);
        currOffset += maxDataCount;
    }

    if (remainNum > 0) {
        if (remainNum >= maxCopyPadNum) {
            SingleIteration(currOffset, maxCopyPadNum);
            currOffset += maxCopyPadNum;
            remainNum -= maxCopyPadNum;
        }

        if (remainNum > 0) {
            SingleIterationAligned(currOffset, remainNum);
        }
    }

    DisableDmaAtomic();
}

__aicore__ inline void SasumAIV::SingleIteration(uint32_t offset, uint32_t dataCount)
{
    CopyIn(offset, dataCount);
    Compute(dataCount);
    CopyOut();
}

__aicore__ inline void SasumAIV::SingleIterationAligned(uint32_t offset, uint32_t dataCount)
{
    uint32_t dataCountAligned = (dataCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
    CopyInPad(offset, dataCount);
    Compute(dataCountAligned);
    CopyOut();
}

__aicore__ inline void SasumAIV::CopyIn(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
    DataCopy(inLocal, inGM[offset], dataCount);
    inQueue.EnQue<float>(inLocal);
}

__aicore__ inline void SasumAIV::CopyInPad(uint32_t offset, uint32_t dataCount)
{
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataCount * sizeof(float)), 0, 0};
    uint8_t paddingNum = ELEMENTS_PER_BLOCK - dataCount % ELEMENTS_PER_BLOCK;
    DataCopyPadParams padParams{true, 0, paddingNum, 0};

    LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
    DataCopyPad(inLocal, inGM[offset], copyParams, padParams);
    inQueue.EnQue<float>(inLocal);
}

__aicore__ inline void SasumAIV::Compute(uint32_t dataCount)
{
    LocalTensor<float> inLocal = inQueue.DeQue<float>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

    Abs(inLocal, inLocal, dataCount);
    ReduceSum(outLocal, inLocal, workLocal, dataCount);

    outQueue.EnQue<float>(outLocal);
    inQueue.FreeTensor(inLocal);
}

__aicore__ inline void SasumAIV::CopyOut()
{
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
    DataCopyPad(outGM, outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SasumSimtCompute(
    uint32_t calNum, uint32_t startOffset, uint32_t stride, __gm__ const float* xGm, __gm__ float* partialOut,
    uint32_t blockDimPow2)
{
    if (calNum == 0) {
        return;
    }

    __ubuf__ float ubPartialSums[SIMT_MAX_THREAD_NUM];
    float partial = 0.0f;

    for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
        float xVal = xGm[(startOffset + i) * stride];
        partial += (xVal >= 0.0f) ? xVal : -xVal;
    }

    ubPartialSums[threadIdx.x] = partial;
    asc_syncthreads();

    uint32_t n = blockDimPow2;

    for (uint32_t s = n >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
            ubPartialSums[threadIdx.x] += ubPartialSums[threadIdx.x + s];
        }
        asc_syncthreads();
    }

    if (threadIdx.x == 0) {
        partialOut[0] = ubPartialSums[0];
    }
}

class SasumReduce {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR inGM, GM_ADDR outGM, uint32_t count);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    GlobalTensor<float> inGM;
    GlobalTensor<float> outGM;

    uint32_t count;
    uint32_t paddedCount;
};

__aicore__ inline void SasumReduce::Init(TPipe* pipe, GM_ADDR inDevice, GM_ADDR outDevice, uint32_t count)
{
    pipe_ = pipe;
    this->count = count;

    // Pad to ELEMENTS_PER_BLOCK alignment for DataCopy safety
    paddedCount = (count + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
    if (paddedCount < ELEMENTS_PER_BLOCK) {
        paddedCount = ELEMENTS_PER_BLOCK;
    }

    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inDevice), paddedCount);
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outDevice), 1);

    pipe_->InitBuffer(inQueue, 1, paddedCount * sizeof(float));
    pipe_->InitBuffer(outQueue, 1, sizeof(float));
}

__aicore__ inline void SasumReduce::Process()
{
    if (count == 0) {
        return;
    }
    CopyIn();
    Compute();
    CopyOut();
}

__aicore__ inline void SasumReduce::CopyIn()
{
    LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
    if (count % ELEMENTS_PER_BLOCK != 0) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(count * sizeof(float)), 0, 0};
        uint8_t paddingNum = static_cast<uint8_t>(paddedCount - count);
        DataCopyPadParams padParams{true, 0, paddingNum, 0};
        DataCopyPad(inLocal, inGM, copyParams, padParams);
    } else {
        DataCopy(inLocal, inGM, count);
    }
    inQueue.EnQue<float>(inLocal);
}

__aicore__ inline void SasumReduce::Compute()
{
    LocalTensor<float> inLocal = inQueue.DeQue<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        sum += inLocal.GetValue(i);
    }
    outLocal.SetValue(0, sum);

    outQueue.EnQue<float>(outLocal);
    inQueue.FreeTensor(inLocal);
}

__aicore__ inline void SasumReduce::CopyOut()
{
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
    DataCopyPad(outGM, outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}

// AIV path (incx == 1): per-block Abs+ReduceSum with atomic add to result
__global__ __aicore__ void sasum_aiv_kernel(GM_ADDR inGM, GM_ADDR outGM, SasumTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    SasumAIV op;
    op.Init(&pipe, inGM, outGM, tdata);
    op.Process();
}

// SIMT path (incx != 1): per-block strided partial sums → workspace[blockIdx]
__global__ __aicore__ void sasum_simt_kernel(GM_ADDR inGM, GM_ADDR workSpace, SasumTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockIdx = GetBlockIdx();
    uint32_t calNum = tdata.calNum[blockIdx];

    if (calNum > 0) {
        asc_vf_call<SasumSimtCompute>(
            dim3{tdata.nthreads, 1, 1}, calNum, tdata.startOffset[blockIdx], static_cast<uint32_t>(tdata.incx),
            reinterpret_cast<__gm__ const float*>(inGM), reinterpret_cast<__gm__ float*>(workSpace) + blockIdx,
            RoundUpPow2(tdata.nthreads));
    }
}

// Reduce: 1-block AIV sums workspace partials into result
__global__ __aicore__ void sasum_reduce_kernel(GM_ADDR workSpace, GM_ADDR outGM, uint32_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    SasumReduce op;
    op.Init(&pipe, workSpace, outGM, count);
    op.Process();
}

// issue #588: 返回值改为 aclblasStatus_t——incx≠1 路径的 workspace 清零失败时结果会静默错误，
// 失败须传播给调用方且不下发后续内核
aclblasStatus_t sasum_kernel_do(
    GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workSpace, const SasumTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);

    if (tiling.incx == 1) {
        sasum_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(inGM, outGM, tiling);
        return ACLBLAS_STATUS_SUCCESS;
    }
    size_t workspaceBytes = static_cast<size_t>(tiling.useCoreNum) * sizeof(float);
    aclError memsetRet = aclrtMemsetAsync(workSpace, workspaceBytes, 0, workspaceBytes, aclStream);
    if (memsetRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    sasum_simt_kernel<<<numBlocks, nullptr, aclStream>>>(inGM, workSpace, tiling);
    sasum_reduce_kernel<<<1, nullptr, aclStream>>>(workSpace, outGM, tiling.useCoreNum);
    return ACLBLAS_STATUS_SUCCESS;
}
