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
#include "snrm2_kernel.h"

using namespace AscendC;

constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;

// All UB buffer allocations and Vector/MTE data transfers must be 32B-aligned.
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t ELEMENTS_PER_BLOCK = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;

// ReduceSum Level-1 processes data in 256B repeats (64 fp32 elements per repeat),
// producing one intermediate result per repeat.
constexpr uint32_t REDUCE_REPEAT_BYTES = 256;
constexpr uint32_t ELEMENTS_PER_REPEAT = REDUCE_REPEAT_BYTES / BYTENUM_PER_FLOAT32;

// SAFETY_MARGIN reserves 32KB for TPipe/TQue runtime metadata.
// UB budget (UB_SIZE = 248*1024 = 253952 bytes):
//   inQueue  : BUFFER_NUM=2 * maxDataCount * sizeof(float)
//   outQueue : BUFFER_NUM=2 * UB_BYTENUM_PER_BLOCK = 64 bytes
//   workBuf  : ceil(maxDataCount / ELEMENTS_PER_REPEAT) aligned * sizeof(float) + UB_BYTENUM_PER_BLOCK
//   accBuf   : ELEMENTS_PER_BLOCK * sizeof(float) = 32 bytes
//   SAFETY_MARGIN = 32768
// Constraint: 8M + ((ceil(M/64)+7)/8*8)*4 <= 221056
// Numerical solution M <= 27416, aligned to 64 multiple -> 27392.
constexpr uint32_t SAFETY_MARGIN = 32 * 1024;
constexpr uint32_t UB_MAX_CHUNK_FLOATS = 27392;

// For Ascend950 (arch35), Sqrt Level 2 template has a mandatory config parameter
// with default DEFAULT_SQRT_CONFIG = {SqrtAlgo::INTRINSIC}.
// We call Sqrt<float>(...) which uses the default config (MED-1).

// ============================================================================
// Snrm2AIV -- SIMD path (incx==1): Mul -> ReduceSum -> local accumulate
// ============================================================================

class Snrm2AIV {
public:
    __aicore__ inline Snrm2AIV() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR xDevice, GM_ADDR wsDevice,
                                const Snrm2TilingData& tdata);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const Snrm2TilingData& tdata);
    __aicore__ inline void SingleIteration(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void SingleIterationAligned(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t dataCount);
    __aicore__ inline void ComputeSquareAndReduceSum(uint32_t dataCount);
    __aicore__ inline void AccumulateChunk();
    __aicore__ inline void WriteOut();

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> accBuf_;  // local accumulator for chunk partial sums

    GlobalTensor<float> xGM_;
    GlobalTensor<float> wsGM_;

    uint32_t blockIdx_;
    uint32_t computeNum_;
    uint32_t startOffset_;
    uint32_t maxDataCount_;
    bool hasAccumulated_;
};

__aicore__ inline void Snrm2AIV::ParseTilingData(const Snrm2TilingData& tdata)
{
    uint32_t perCoreN = tdata.perCoreN;
    uint32_t remainder = tdata.remainder;

    startOffset_ = blockIdx_ * perCoreN;
    if (blockIdx_ < remainder) {
        startOffset_ += blockIdx_;
    } else {
        startOffset_ += remainder;
    }
    computeNum_ = perCoreN + (blockIdx_ < remainder ? 1 : 0);
}

__aicore__ inline void Snrm2AIV::Init(TPipe* pipe, GM_ADDR xDevice, GM_ADDR wsDevice,
                                       const Snrm2TilingData& tdata)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    ParseTilingData(tdata);

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xDevice),
                          static_cast<uint32_t>(tdata.n));
    // Each SIMD core writes its final accumulated sum to workspace[blockIdx_]
    wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsDevice) + blockIdx_, 1);

    maxDataCount_ = UB_MAX_CHUNK_FLOATS;
    hasAccumulated_ = false;

    // workBuf for ReduceSum: ceil(maxDataCount / ELEMENTS_PER_REPEAT) aligned
    int elementsPerRepeat = static_cast<int>(ELEMENTS_PER_REPEAT);
    int level1RepeatCnt = (static_cast<int>(maxDataCount_) + elementsPerRepeat - 1) / elementsPerRepeat;
    int level1AlignEnd = (level1RepeatCnt + static_cast<int>(ELEMENTS_PER_BLOCK) - 1) /
                         static_cast<int>(ELEMENTS_PER_BLOCK) * static_cast<int>(ELEMENTS_PER_BLOCK);
    uint32_t workBufBytes = static_cast<uint32_t>(level1AlignEnd) * sizeof(float);

    pipe_->InitBuffer(workBuf_, workBufBytes + UB_BYTENUM_PER_BLOCK);
    pipe_->InitBuffer(inQueue_, BUFFER_NUM, maxDataCount_ * sizeof(float));
    pipe_->InitBuffer(outQueue_, BUFFER_NUM, UB_BYTENUM_PER_BLOCK);
    pipe_->InitBuffer(accBuf_, ELEMENTS_PER_BLOCK * sizeof(float));  // accumulator buffer
}

__aicore__ inline void Snrm2AIV::Process()
{
    if (computeNum_ == 0) {
        return;
    }

    // Initialize accumulator to zero
    LocalTensor<float> accLocal = accBuf_.Get<float>();
    Duplicate<float>(accLocal, 0.0f, static_cast<int32_t>(ELEMENTS_PER_BLOCK));

    uint32_t repeatTimes = computeNum_ / maxDataCount_;
    uint32_t remainNum = computeNum_ % maxDataCount_;
    uint32_t maxCopyPadNum = UINT16_MAX / sizeof(float) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;

    uint32_t currOffset = startOffset_;
    for (uint32_t i = 0; i < repeatTimes; i++) {
        SingleIteration(currOffset, maxDataCount_);
        currOffset += maxDataCount_;
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

    // Write final accumulated sum to workspace[blockIdx_]
    WriteOut();
}

__aicore__ inline void Snrm2AIV::SingleIteration(uint32_t offset, uint32_t dataCount)
{
    CopyIn(offset, dataCount);
    ComputeSquareAndReduceSum(dataCount);
    AccumulateChunk();
}

__aicore__ inline void Snrm2AIV::SingleIterationAligned(uint32_t offset, uint32_t dataCount)
{
    uint32_t dataCountAligned = (dataCount + ELEMENTS_PER_BLOCK - 1) /
                                ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
    CopyInPad(offset, dataCount);
    ComputeSquareAndReduceSum(dataCountAligned);
    AccumulateChunk();
}

__aicore__ inline void Snrm2AIV::CopyIn(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
    DataCopy(inLocal, xGM_[offset], dataCount);
    inQueue_.EnQue<float>(inLocal);
}

__aicore__ inline void Snrm2AIV::CopyInPad(uint32_t offset, uint32_t dataCount)
{
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataCount * sizeof(float)), 0, 0};
    uint8_t paddingNum = static_cast<uint8_t>(ELEMENTS_PER_BLOCK - dataCount % ELEMENTS_PER_BLOCK);
    DataCopyPadParams padParams{true, 0, paddingNum, 0};

    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
    DataCopyPad(inLocal, xGM_[offset], copyParams, padParams);
    inQueue_.EnQue<float>(inLocal);
}

__aicore__ inline void Snrm2AIV::ComputeSquareAndReduceSum(uint32_t dataCount)
{
    LocalTensor<float> inLocal = inQueue_.DeQue<float>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

    // x^2 (in-place)
    Mul<float>(inLocal, inLocal, inLocal, static_cast<int32_t>(dataCount));

    // sum(x^2)
    ReduceSum<float>(outLocal, inLocal, workLocal, static_cast<int32_t>(dataCount));

    outQueue_.EnQue<float>(outLocal);
    inQueue_.FreeTensor(inLocal);
}

__aicore__ inline void Snrm2AIV::AccumulateChunk()
{
    LocalTensor<float> outLocal = outQueue_.DeQue<float>();
    LocalTensor<float> accLocal = accBuf_.Get<float>();

    // acc[0] += outLocal[0] (accumulate chunk partial sum)
    Add<float>(accLocal, accLocal, outLocal, 1);

    outQueue_.FreeTensor(outLocal);
    hasAccumulated_ = true;
}

__aicore__ inline void Snrm2AIV::WriteOut()
{
    if (!hasAccumulated_) {
        return;
    }
    LocalTensor<float> accLocal = accBuf_.Get<float>();
    // AccumulateChunk updates accBuf_ on the V pipe; wait before MTE3 reads it.
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
    DataCopyPad(wsGM_, accLocal, copyParams);
}

// ============================================================================
// Snrm2SimtCompute -- SIMT path (incx!=1): thread-level partial sum reduction
// ============================================================================

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Snrm2SimtCompute(
    uint32_t calNum, uint32_t startOffset, uint32_t stride,
    __gm__ const float* xGm, __gm__ float* partialOut, uint32_t blockDimPow2)
{
    if (calNum == 0) {
        return;
    }

    __ubuf__ float ubPartialSums[SIMT_MAX_THREAD_NUM];
    float partial = 0.0f;

    // Each thread accumulates its own partial sum of squares
    for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
        float xVal = xGm[(startOffset + i) * stride];
        partial += xVal * xVal;
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

// ============================================================================
// Snrm2Reduce -- Final reduction: ReduceSum + Sqrt (shared by both paths)
// ============================================================================

class Snrm2Reduce {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR wsDevice, GM_ADDR outDevice, uint32_t count);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> workBuf_;

    GlobalTensor<float> wsGM_;
    GlobalTensor<float> outGM_;

    uint32_t count_;
    uint32_t paddedCount_;
};

__aicore__ inline void Snrm2Reduce::Init(TPipe* pipe, GM_ADDR wsDevice,
                                          GM_ADDR outDevice, uint32_t count)
{
    pipe_ = pipe;
    count_ = count;

    paddedCount_ = (count + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
    if (paddedCount_ < ELEMENTS_PER_BLOCK) {
        paddedCount_ = ELEMENTS_PER_BLOCK;
    }

    wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsDevice), paddedCount_);
    outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outDevice), 1);

    pipe_->InitBuffer(inQueue_, 1, paddedCount_ * sizeof(float));
    pipe_->InitBuffer(outQueue_, 1, ELEMENTS_PER_BLOCK * sizeof(float));
    pipe_->InitBuffer(workBuf_, ELEMENTS_PER_BLOCK * sizeof(float));
}

__aicore__ inline void Snrm2Reduce::Process()
{
    if (count_ == 0) {
        return;
    }
    CopyIn();
    Compute();
    CopyOut();
}

__aicore__ inline void Snrm2Reduce::CopyIn()
{
    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
    if (count_ % ELEMENTS_PER_BLOCK != 0) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(count_ * sizeof(float)), 0, 0};
        uint8_t paddingNum = static_cast<uint8_t>(paddedCount_ - count_);
        DataCopyPadParams padParams{true, 0, paddingNum, 0};
        DataCopyPad(inLocal, wsGM_, copyParams, padParams);
    } else {
        DataCopy(inLocal, wsGM_, count_);
    }
    inQueue_.EnQue<float>(inLocal);
}

__aicore__ inline void Snrm2Reduce::Compute()
{
    LocalTensor<float> inLocal = inQueue_.DeQue<float>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

    // ReduceSum: sum(workspace[0..paddedCount-1])
    ReduceSum<float>(outLocal, inLocal, workLocal, static_cast<int32_t>(paddedCount_));

    // Sqrt: result = sqrt(sum) -- Ascend950 uses default config (INTRINSIC) (MED-1)
    Sqrt<float>(outLocal, outLocal, static_cast<int32_t>(ELEMENTS_PER_BLOCK));

    outQueue_.EnQue<float>(outLocal);
    inQueue_.FreeTensor(inLocal);
}

__aicore__ inline void Snrm2Reduce::CopyOut()
{
    LocalTensor<float> outLocal = outQueue_.DeQue<float>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
    DataCopyPad(outGM_, outLocal, copyParams);
    outQueue_.FreeTensor(outLocal);
}

// ============================================================================
// Kernel entry points
// ============================================================================

// SIMD AIV kernel (incx==1): per-block square + ReduceSum -> own workspace[blockIdx] slot
__global__ __aicore__ void snrm2_aiv_kernel(GM_ADDR xGm, GM_ADDR workspace, Snrm2TilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    Snrm2AIV op;
    op.Init(&pipe, xGm, workspace, tdata);
    op.Process();
}

// SIMT kernel (incx!=1): per-block strided partial sums -> workspace[blockIdx]
__global__ __aicore__ void snrm2_simt_kernel(GM_ADDR xGm, GM_ADDR workspace, Snrm2TilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockIdx = GetBlockIdx();
    int64_t myOffset = static_cast<int64_t>(blockIdx) * tdata.perCoreN;
    if (static_cast<uint32_t>(blockIdx) < tdata.remainder) {
        myOffset += static_cast<uint32_t>(blockIdx);
    } else {
        myOffset += tdata.remainder;
    }
    uint32_t myCount = tdata.perCoreN + (static_cast<uint32_t>(blockIdx) < tdata.remainder ? 1 : 0);

    if (myCount > 0) {
        asc_vf_call<Snrm2SimtCompute>(
            dim3{tdata.nthreads, 1, 1},
            myCount, static_cast<uint32_t>(myOffset), static_cast<uint32_t>(tdata.incx),
            reinterpret_cast<__gm__ const float*>(xGm),
            reinterpret_cast<__gm__ float*>(workspace) + blockIdx,
            RoundUpPow2(tdata.nthreads));
    }
}

// Reduce kernel (shared): single-core ReduceSum + Sqrt -> result
__global__ __aicore__ void snrm2_reduce_kernel(GM_ADDR workspace, GM_ADDR resultGm, uint32_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    Snrm2Reduce op;
    op.Init(&pipe, workspace, resultGm, count);
    op.Process();
}

// ============================================================================
// Host-callable dispatch function
// ============================================================================

void snrm2_kernel_do(uint8_t* xGm, uint8_t* resultGm, uint8_t* workspace,
                      const Snrm2TilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    if (tiling.incx == 1) {
        // SIMD path: each AIV core writes its partial sum to its own workspace[blockIdx] slot
        snrm2_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(xGm, workspace, tiling);
    } else {
        // SIMT path: multi-core SIMT -> workspace[blockIdx]
        snrm2_simt_kernel<<<numBlocks, nullptr, aclStream>>>(xGm, workspace, tiling);
    }

    // Common: single-core reduce + sqrt (implicit sync via two kernel launches, MED-2)
    snrm2_reduce_kernel<<<1, nullptr, aclStream>>>(workspace, resultGm, tiling.useCoreNum);
}
