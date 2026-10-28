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
#include "cann_ops_blas_common.h"
#include "ssymv_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class SymvAIV {
public:
    __aicore__ inline SymvAIV() = default;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SymvTilingData& tiling);
    __aicore__ inline void Process();

private:
    TPipe pipe;

    __aicore__ inline void CopyInV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputeV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInPadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputePadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutPadV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInVec(uint32_t curOffset, uint32_t dataCount, bool needPad);
    __aicore__ inline void ComputeVec(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutVec(uint32_t curOffset, uint32_t dataCount, bool needPad);

    __aicore__ inline void LoadX2(uint32_t rowOffset)
    {
        uint8_t paddingNum2 = elementsPerBlock - 1;
        DataCopyExtParams copyParams2{1, BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
        LocalTensor<T> LocalX2 = x2Queue.AllocTensor<T>();
        DataCopyPad(LocalX2, xGM[XPhysicalPos(rowOffset)], copyParams2, padParams2);
        x2Queue.EnQue<T>(LocalX2);
    }

    __aicore__ inline uint32_t XPhysicalPos(uint32_t logical)
    {
        return (incx >= 0) ? (logical * absIncx) : ((n - 1U - logical) * absIncx);
    }
    __aicore__ inline uint32_t YPhysicalPos(uint32_t logical)
    {
        return (incy >= 0) ? (logical * absIncy) : ((n - 1U - logical) * absIncy);
    }

    // chunk 是否全程位于对角线一侧（不含对角元）：UPPER 走 [0, row]，纯三角 chunk
    // 完全在对角线下方；LOWER 走 [row, n)，纯三角 chunk 完全在对角线上方
    __aicore__ inline bool IsPureTriangleChunk(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
    {
        if (uplo == ACLBLAS_UPPER) {
            return colOffset + dataCount <= rowOffset;
        }
        return colOffset > rowOffset;
    }

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> z2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> workspaceQueue;

    uint32_t vecIdx = 0;
    uint32_t n = 0;
    uint32_t lda = 0;
    uint32_t useCoreNum = 0;
    uint32_t maxDataCount = 0;

    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);
    int64_t incx = 1;
    int64_t incy = 1;
    uint32_t absIncx = 1;
    uint32_t absIncy = 1;
    uint32_t uplo = ACLBLAS_LOWER;

    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / BYTENUM_PER_FLOAT32;
    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void SymvAIV<T>::Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SymvTilingData& tiling)
{
    vecIdx = GetBlockIdx();

    n = tiling.n;
    lda = tiling.lda;
    useCoreNum = tiling.useCoreNum;
    alpha = tiling.alpha;
    beta = tiling.beta;
    incx = tiling.incx;
    incy = tiling.incy;
    uplo = tiling.uplo;

    if (useCoreNum == 0 || useCoreNum > SYMV_MAX_CORE_NUM) {
        useCoreNum = 1;
    }

    absIncx = (incx >= 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);
    absIncy = (incy >= 0) ? static_cast<uint32_t>(incy) : static_cast<uint32_t>(-incy);

    xGM.SetGlobalBuffer((__gm__ T*)x, (n - 1U) * absIncx + 1U);
    yGM.SetGlobalBuffer((__gm__ T*)y, (n - 1U) * absIncy + 1U);
    aGM.SetGlobalBuffer((__gm__ T*)a, static_cast<uint64_t>(this->n) * this->lda);

    maxDataCount = 30 * 1024 / BYTENUM_PER_FLOAT32;

    pipe.InitBuffer(aQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(xQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(x2Queue, BUFFER_NUM, sizeof(T));
    pipe.InitBuffer(zQueue, BUFFER_NUM, sizeof(T));
    pipe.InitBuffer(z2Queue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(workspaceQueue, BUFFER_NUM, maxDataCount * sizeof(T));
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyInV2(
    uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint64_t r = static_cast<uint64_t>(rowOffset) * lda + colOffset;
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    DataCopy(LocalA, aGM[r], dataCount);
    aQueue.EnQue<T>(LocalA);

    LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
    if (incx == 1) {
        DataCopy(LocalX, xGM[colOffset], dataCount);
    } else {
        for (uint32_t i = 0; i < dataCount; i++) {
            LocalX.SetValue(i, xGM.GetValue(XPhysicalPos(colOffset + i)));
        }
    }
    xQueue.EnQue<T>(LocalX);

    LoadX2(rowOffset);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::ComputeV2(
    uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalX2 = x2Queue.DeQue<T>();
    LocalTensor<T> LocalZ = zQueue.AllocTensor<T>();
    LocalTensor<T> LocalZ2 = z2Queue.AllocTensor<T>();
    LocalTensor<T> workLocal = workspaceQueue.AllocTensor<T>();

    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Mul(LocalZ, LocalA, LocalX, dataCount);
    ReduceSum(LocalZ, LocalZ, LocalZ, dataCount);
    Muls(LocalZ, LocalZ, alpha, 1);
    auto scalar = LocalX2(0) * alpha;
    if (IsPureTriangleChunk(rowOffset, colOffset, dataCount)) {
        Muls(LocalZ2, LocalA, scalar, dataCount);
    } else if (dataCount > 1) {
        Muls(LocalZ2, LocalA, scalar, dataCount - 1);
    }
    zQueue.EnQue<T>(LocalZ);
    z2Queue.EnQue<T>(LocalZ2);
    aQueue.FreeTensor(LocalA);
    xQueue.FreeTensor(LocalX);
    x2Queue.FreeTensor(LocalX2);
    workspaceQueue.FreeTensor(workLocal);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyOutV2(
    uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> zLocal = zQueue.DeQue<T>();
    if (incy == 1) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(yGM[rowOffset], zLocal, copyParams);
    } else {
        uint32_t pos = YPhysicalPos(rowOffset);
        yGM.SetValue(pos, yGM.GetValue(pos) + zLocal.GetValue(0));
    }
    zQueue.FreeTensor(zLocal);

    LocalTensor<T> z2Local = z2Queue.DeQue<T>();
    if (incy == 1) {
        if (IsPureTriangleChunk(rowOffset, colOffset, dataCount)) {
            uint8_t paddingNum2 = elementsPerBlock - dataCount % elementsPerBlock;
            DataCopyExtParams copyParams2{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
            DataCopyPad(yGM[colOffset], z2Local, copyParams2);
        } else {
            uint8_t paddingNum2 = elementsPerBlock - (dataCount - 1) % elementsPerBlock;
            DataCopyExtParams copyParams2{1, (dataCount - 1) * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
            DataCopyPad(yGM[colOffset], z2Local, copyParams2);
        }
    } else {
        if (IsPureTriangleChunk(rowOffset, colOffset, dataCount)) {
            for (uint32_t i = 0; i < dataCount; i++) {
                uint32_t pos = YPhysicalPos(colOffset + i);
                yGM.SetValue(pos, yGM.GetValue(pos) + z2Local.GetValue(i));
            }
        } else if (dataCount > 1) {
            for (uint32_t i = 0; i < dataCount - 1; i++) {
                uint32_t pos = YPhysicalPos(colOffset + i);
                yGM.SetValue(pos, yGM.GetValue(pos) + z2Local.GetValue(i));
            }
        }
    }
    z2Queue.FreeTensor(z2Local);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyInPadV2(
    uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    uint64_t r = static_cast<uint64_t>(rowOffset) * lda + colOffset;
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    DataCopyPad(LocalA, aGM[r], copyParams, padParams);
    aQueue.EnQue<T>(LocalA);

    LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
    if (incx == 1) {
        DataCopyPad(LocalX, xGM[colOffset], copyParams, padParams);
    } else {
        for (uint32_t i = 0; i < dataCount; i++) {
            LocalX.SetValue(i, xGM.GetValue(XPhysicalPos(colOffset + i)));
        }
    }
    xQueue.EnQue<T>(LocalX);

    LoadX2(rowOffset);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::ComputePadV2(
    uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalX2 = x2Queue.DeQue<T>();
    LocalTensor<T> LocalZ = zQueue.AllocTensor<T>();
    LocalTensor<T> LocalZ2 = z2Queue.AllocTensor<T>();
    LocalTensor<T> workLocal = workspaceQueue.AllocTensor<T>();

    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Mul(LocalZ, LocalA, LocalX, dataCount);
    ReduceSum(LocalZ, LocalZ, LocalZ, dataCount);
    Muls(LocalZ, LocalZ, alpha, 1);
    auto scalar = LocalX2(0) * alpha;
    if (IsPureTriangleChunk(rowOffset, colOffset, dataCount)) {
        Muls(LocalZ2, LocalA, scalar, dataCount);
    } else if (dataCount > 1) {
        Muls(LocalZ2, LocalA, scalar, dataCount - 1);
    }
    zQueue.EnQue<T>(LocalZ);
    z2Queue.EnQue<T>(LocalZ2);
    aQueue.FreeTensor(LocalA);
    xQueue.FreeTensor(LocalX);
    x2Queue.FreeTensor(LocalX2);
    workspaceQueue.FreeTensor(workLocal);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyOutPadV2(
    uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> zLocal = zQueue.DeQue<T>();
    if (incy == 1) {
        DataCopyPad(yGM[rowOffset], zLocal, copyParams);
    } else {
        uint32_t pos = YPhysicalPos(rowOffset);
        yGM.SetValue(pos, yGM.GetValue(pos) + zLocal.GetValue(0));
    }
    zQueue.FreeTensor(zLocal);

    LocalTensor<T> z2Local = z2Queue.DeQue<T>();
    if (incy == 1) {
        if (IsPureTriangleChunk(rowOffset, colOffset, dataCount)) {
            uint8_t paddingNum2 = elementsPerBlock - dataCount % elementsPerBlock;
            DataCopyExtParams copyParams2{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
            DataCopyPad(yGM[colOffset], z2Local, copyParams2);
        } else {
            uint8_t paddingNum2 = elementsPerBlock - (dataCount - 1) % elementsPerBlock;
            DataCopyExtParams copyParams2{1, (dataCount - 1) * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
            DataCopyPad(yGM[colOffset], z2Local, copyParams2);
        }
    } else {
        if (IsPureTriangleChunk(rowOffset, colOffset, dataCount)) {
            for (uint32_t i = 0; i < dataCount; i++) {
                uint32_t pos = YPhysicalPos(colOffset + i);
                yGM.SetValue(pos, yGM.GetValue(pos) + z2Local.GetValue(i));
            }
        } else if (dataCount > 1) {
            for (uint32_t i = 0; i < dataCount - 1; i++) {
                uint32_t pos = YPhysicalPos(colOffset + i);
                yGM.SetValue(pos, yGM.GetValue(pos) + z2Local.GetValue(i));
            }
        }
    }
    z2Queue.FreeTensor(z2Local);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyInVec(uint32_t curOffset, uint32_t dataCount, bool needPad)
{
    if (incy == 1) {
        if (!needPad) {
            LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
            DataCopy(LocalX, yGM[curOffset], dataCount);
            xQueue.EnQue<T>(LocalX);
            return;
        }

        uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
        DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

        LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
        DataCopyPad(LocalX, yGM[curOffset], copyParams, padParams);
        xQueue.EnQue<T>(LocalX);
    } else {
        LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
        for (uint32_t i = 0; i < dataCount; i++) {
            LocalX.SetValue(i, yGM.GetValue(YPhysicalPos(curOffset + i)));
        }
        xQueue.EnQue<T>(LocalX);
    }
}

template <typename T>
__aicore__ inline void SymvAIV<T>::ComputeVec(uint32_t curOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalZ = z2Queue.AllocTensor<T>();
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    T betaMinusOne = beta - static_cast<T>(1.0f);
    Muls(LocalZ, LocalX, betaMinusOne, dataCount);
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    z2Queue.EnQue<T>(LocalZ);
    xQueue.FreeTensor(LocalX);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyOutVec(uint32_t curOffset, uint32_t dataCount, bool needPad)
{
    if (incy == 1) {
        if (!needPad) {
            LocalTensor<T> LocalZ = z2Queue.DeQue<T>();
            DataCopy(yGM[curOffset], LocalZ, dataCount);
            z2Queue.FreeTensor(LocalZ);
            return;
        }

        uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
        DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

        LocalTensor<T> LocalZ = z2Queue.DeQue<T>();
        DataCopyPad(yGM[curOffset], LocalZ, copyParams);
        z2Queue.FreeTensor(LocalZ);
    } else {
        LocalTensor<T> LocalZ = z2Queue.DeQue<T>();
        for (uint32_t i = 0; i < dataCount; i++) {
            uint32_t pos = YPhysicalPos(curOffset + i);
            yGM.SetValue(pos, yGM.GetValue(pos) + LocalZ.GetValue(i));
        }
        z2Queue.FreeTensor(LocalZ);
    }
}

template <typename T>
__aicore__ inline void SymvAIV<T>::Process()
{
    SetAtomicAdd<T>();
    if (vecIdx >= useCoreNum) {
        SetAtomicNone();
        return;
    }
    if (vecIdx == 0) {
        uint32_t repeatTimes = n / maxDataCount;
        uint32_t remainNum = n % maxDataCount;
        uint32_t curOffset = 0;
        for (uint32_t i = 0; i < repeatTimes; i++) {
            uint32_t dataCount = maxDataCount;
            CopyInVec(curOffset, dataCount, false);
            ComputeVec(curOffset, dataCount);
            CopyOutVec(curOffset, dataCount, false);
            curOffset += dataCount;
        }
        if (remainNum > 0) {
            uint32_t dataCount = remainNum;
            CopyInVec(curOffset, dataCount, true);
            ComputeVec(curOffset, dataCount);
            CopyOutVec(curOffset, dataCount, true);
        }
    }

    for (uint32_t taskIdx = vecIdx; taskIdx < n; taskIdx += useCoreNum) {
        uint32_t row = taskIdx;
        if (uplo == ACLBLAS_UPPER) {
            uint32_t colLen = row + 1;
            for (uint32_t col = 0; col < colLen; col += maxDataCount) {
                uint32_t dataCount = maxDataCount;
                if (col + dataCount > colLen) {
                    dataCount = colLen - col;
                    CopyInPadV2(taskIdx, row, col, dataCount);
                    ComputePadV2(taskIdx, row, col, dataCount);
                    CopyOutPadV2(taskIdx, row, col, dataCount);
                    continue;
                }
                CopyInV2(taskIdx, row, col, dataCount);
                ComputeV2(taskIdx, row, col, dataCount);
                CopyOutV2(taskIdx, row, col, dataCount);
            }
        } else {
            // LOWER：对角元单独作 1 元素 chunk（dataCount=1 时 z2 自然为空），
            // 其余 chunk 严格位于对角线上方，全部走纯三角路径
            CopyInPadV2(taskIdx, row, row, 1);
            ComputePadV2(taskIdx, row, row, 1);
            CopyOutPadV2(taskIdx, row, row, 1);
            for (uint32_t col = row + 1; col < n; col += maxDataCount) {
                uint32_t dataCount = maxDataCount;
                if (col + dataCount > n) {
                    dataCount = n - col;
                    CopyInPadV2(taskIdx, row, col, dataCount);
                    ComputePadV2(taskIdx, row, col, dataCount);
                    CopyOutPadV2(taskIdx, row, col, dataCount);
                    continue;
                }
                CopyInV2(taskIdx, row, col, dataCount);
                ComputeV2(taskIdx, row, col, dataCount);
                CopyOutV2(taskIdx, row, col, dataCount);
            }
        }
    }
    SetAtomicNone();
}

__global__ __aicore__ void ssymv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SymvAIV<float> op;
    op.Init(a, x, y, tiling);
    op.Process();
}

void ssymv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SymvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ssymv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
}
