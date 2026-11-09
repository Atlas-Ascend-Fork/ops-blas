/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <type_traits>
#include <cstdint>
#include "acl/acl_rt.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "complexmatdot_kernel.h"

using namespace AscendC;
using namespace matmul;

constexpr int32_t BUFFER_NUM = 2;

namespace ComplexMatDot {

struct ComplexMatDotKernelParam {
    GM_ADDR matx;
    GM_ADDR maty;
    GM_ADDR aug;
    GM_ADDR result;
    uint32_t m;
    uint32_t n;
    uint64_t offset;
    uint32_t calNumPerCore;
};

template <typename T>
class ComplexMatDotAIV {
public:
    __aicore__ inline ComplexMatDotAIV(){};
    __aicore__ inline void Init(ComplexMatDotKernelParam kernelParam);
    __aicore__ inline void Process();
    __aicore__ inline void SingleIteration(uint64_t offset, uint64_t dataCount, LocalTensor<uint32_t> offsetLocal);
    __aicore__ inline void SingleIterationAligned(
        uint64_t offset, uint32_t dataCount, LocalTensor<uint32_t> offsetLocal);
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t dataCount);
    __aicore__ inline void CopyInPad(uint64_t offset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t dataCount, LocalTensor<uint32_t> offsetLocal);
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t dataCount, uint32_t isAligned);

private:
    TPipe pipe;

    GlobalTensor<T> matxGM;
    GlobalTensor<T> matyGM;
    GlobalTensor<T> resultGM;
    GlobalTensor<uint32_t> augGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> xMatQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> yMatQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outMatQueue;

    TBuf<TPosition::VECCALC> offsetBuf;

    uint32_t calNum;

    uint64_t startOffset;
    uint64_t maxDataCount;
};

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::Init(ComplexMatDotKernelParam kernelParam)
{
    matxGM.SetGlobalBuffer((__gm__ T*)(kernelParam.matx), kernelParam.m * kernelParam.n);
    matyGM.SetGlobalBuffer((__gm__ T*)(kernelParam.maty), kernelParam.m * kernelParam.n);
    resultGM.SetGlobalBuffer((__gm__ T*)(kernelParam.result), kernelParam.m * kernelParam.n);
    augGM.SetGlobalBuffer((__gm__ uint32_t*)(kernelParam.aug));

    calNum = kernelParam.calNumPerCore;

    startOffset = kernelParam.offset;
    maxDataCount = 27 * 1024 / 4; // 27kb / 4b

    // ub 192kb
    pipe.InitBuffer(xMatQueue, 2, maxDataCount * sizeof(T));   // 54kb
    pipe.InitBuffer(yMatQueue, 2, maxDataCount * sizeof(T));   // 54kb
    pipe.InitBuffer(outMatQueue, 2, maxDataCount * sizeof(T)); // 54kb

    uint64_t offsetLen = maxDataCount * 4;
    pipe.InitBuffer(offsetBuf, offsetLen); // 27kb

    return;
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::Process()
{
    SetAtomicNone();

    // cal offset
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    DataCopy(offsetLocal, augGM, maxDataCount);

    uint32_t repeatTimes = calNum * 2 / static_cast<uint32_t>(maxDataCount);
    uint32_t remainNum = calNum * 2 % static_cast<uint32_t>(maxDataCount);
    uint64_t currOffset = startOffset;

    if (repeatTimes > 0) {
        pipe_barrier(PIPE_ALL);
        for (uint32_t i = 0; i < repeatTimes; i++) {
            SingleIteration(currOffset, maxDataCount, offsetLocal);
            currOffset += maxDataCount;
        }
    }

    if (remainNum > 0) {
        pipe_barrier(PIPE_ALL);
        SingleIterationAligned(currOffset, remainNum, offsetLocal);
    }

    pipe_barrier(PIPE_ALL);
    return;
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::SingleIteration(
    uint64_t offset, uint64_t dataCount, LocalTensor<uint32_t> offsetLocal)
{
    CopyIn(offset, dataCount);
    Compute(dataCount, offsetLocal);
    CopyOut(offset, dataCount, 1);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::SingleIterationAligned(
    uint64_t offset, uint32_t dataCount, LocalTensor<uint32_t> offsetLocal)
{
    uint32_t dataCountAligned = (dataCount + 7) / 8 * 8;
    CopyInPad(offset, dataCount);
    Compute(dataCountAligned, offsetLocal);
    CopyOut(offset, dataCount, 0);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::CopyIn(uint64_t offset, uint32_t dataCount)
{
    LocalTensor<T> xMatLocal = xMatQueue.AllocTensor<T>();
    LocalTensor<T> yMatLocal = yMatQueue.AllocTensor<T>();
    DataCopy(xMatLocal, matxGM[offset], dataCount);
    DataCopy(yMatLocal, matyGM[offset], dataCount);
    xMatQueue.EnQue<T>(xMatLocal);
    yMatQueue.EnQue<T>(yMatLocal);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::CopyInPad(uint64_t offset, uint32_t dataCount)
{
    LocalTensor<T> xMatLocal = xMatQueue.AllocTensor<T>();
    LocalTensor<T> yMatLocal = yMatQueue.AllocTensor<T>();
    DataCopyParams copyParams{1, static_cast<uint16_t>((dataCount) * sizeof(float)), 0, 0};
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(xMatLocal, matxGM[offset], copyParams, padParams);
    DataCopyPad(yMatLocal, matyGM[offset], copyParams, padParams);
    xMatQueue.EnQue<T>(xMatLocal);
    yMatQueue.EnQue<T>(yMatLocal);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::Compute(uint32_t dataCount, LocalTensor<uint32_t> offsetLocal)
{
    LocalTensor<T> xMatLocal = xMatQueue.DeQue<T>();
    LocalTensor<T> yMatLocal = yMatQueue.DeQue<T>();
    LocalTensor<T> outMatLocal = outMatQueue.AllocTensor<T>();

    uint32_t complexNum = dataCount / 2;
    uint32_t alignedComplexNum = (complexNum + 7) / 8 * 8;

    uint32_t maxComplexNum = (27 * 1024 / sizeof(float)) / 2; // 27kb
    uint32_t realOffset = 0;
    uint32_t imagOffset = (maxComplexNum + 7) / 8 * 8;

    uint64_t rsvdCnt = 64;
    uint16_t repeatTimes = (alignedComplexNum * 2 + 63) / 64;

    // Rx
    GatherMask(outMatLocal[realOffset], xMatLocal, 1, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
    // Ix
    GatherMask(outMatLocal[imagOffset], xMatLocal, 2, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // Ry
    GatherMask(xMatLocal[realOffset], yMatLocal, 1, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
    // Iy
    GatherMask(xMatLocal[imagOffset], yMatLocal, 2, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // Rx * Ry
    Mul(yMatLocal[realOffset], xMatLocal[realOffset], outMatLocal[realOffset], alignedComplexNum);
    // Ix * Iy
    Mul(yMatLocal[imagOffset], xMatLocal[imagOffset], outMatLocal[imagOffset], alignedComplexNum);
    PipeBarrier<PIPE_V>();
    // Rx * Ry - Ix * Iy
    Sub(yMatLocal[realOffset], yMatLocal[realOffset], yMatLocal[imagOffset], alignedComplexNum);

    // Rx * Iy
    Mul(outMatLocal[realOffset], outMatLocal[realOffset], xMatLocal[imagOffset], alignedComplexNum);
    // Ix * Ry
    Mul(outMatLocal[imagOffset], outMatLocal[imagOffset], xMatLocal[realOffset], alignedComplexNum);
    PipeBarrier<PIPE_V>();
    // Rx * Iy + Ix * Ry
    Add(yMatLocal[imagOffset], outMatLocal[realOffset], outMatLocal[imagOffset], alignedComplexNum);

    PipeBarrier<PIPE_V>();
    // restore position
    Gather(outMatLocal, yMatLocal, offsetLocal, 0, imagOffset * 2);

    PipeBarrier<PIPE_V>();
    outMatQueue.EnQue<T>(outMatLocal);
    xMatQueue.FreeTensor(xMatLocal);
    yMatQueue.FreeTensor(yMatLocal);
}

template <typename T>
__aicore__ inline void ComplexMatDotAIV<T>::CopyOut(uint64_t offset, uint32_t dataCount, uint32_t isAligned)
{
    LocalTensor<T> outMatLocal = outMatQueue.DeQue<T>();
    if (isAligned) {
        DataCopy(resultGM[offset], outMatLocal, dataCount);
    } else {
        DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(T) * dataCount), 0, 0};
        DataCopyPad(resultGM[offset], outMatLocal, copyParams);
    }
    outMatQueue.FreeTensor(outMatLocal);
}

} // namespace ComplexMatDot

__global__ __aicore__ void complex_mat_dot_kernel(
    GM_ADDR matx, GM_ADDR maty, GM_ADDR aug, GM_ADDR result, GM_ADDR tiling_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto core_idx = get_block_idx() * get_subblockdim() + get_subblockid(); // 0 ~ 39
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t*>(tiling_gm);

    uint32_t m = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf));     // num of float elements
    uint32_t n = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 4)); // num of float elements

    uint64_t offset = (*(__gm__ uint64_t*)((__gm__ uint8_t*)tiling_buf + 8 + 8 * core_idx));           // FP32
    uint32_t cal_num = (*(__gm__ uint32_t*)((__gm__ uint8_t*)tiling_buf + 8 + 40 * 8 + 4 * core_idx)); // complex num

    ComplexMatDot::ComplexMatDotAIV<float> op;
    op.Init({matx, maty, aug, result, m, n, offset, cal_num});
    op.Process();
}

aclError complex_mat_dot_kernel_do(
    GM_ADDR matx, GM_ADDR maty, GM_ADDR aug, GM_ADDR result, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    complex_mat_dot_kernel<<<numBlocks, nullptr, stream>>>(matx, maty, aug, result, tilingGm);
    // issue #862: 返回发射阶段错误码，host 据此返回 EXECUTION_FAILED（对齐 chpmv 口径）
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
