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
#include "sspmv_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class SpmvAIV {
public:
    __aicore__ inline SpmvAIV() = default;
    __aicore__ inline void Init(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, const SpmvTilingData& tiling);
    __aicore__ inline void Process();

private:
    TPipe pipe;

    __aicore__ inline void CopyIn(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputePad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void ProcessLower();
    __aicore__ inline void ProcessLowerTile(uint32_t packedColumn, uint32_t rowStart, uint32_t dataCount);

    __aicore__ inline void CopyInVec(uint32_t curOffset, uint32_t dataCount, bool needPad);
    __aicore__ inline void ComputeVec(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutVec(uint32_t curOffset, uint32_t dataCount, bool needPad);

    __aicore__ inline uint32_t XPhysicalPos(uint32_t logical)
    {
        return (incx >= 0) ? (logical * absIncx) : ((n - 1U - logical) * absIncx);
    }
    __aicore__ inline uint32_t YPhysicalPos(uint32_t logical)
    {
        return (incy >= 0) ? (logical * absIncy) : ((n - 1U - logical) * absIncy);
    }
    __aicore__ inline uint64_t PackedOffset(uint32_t rowOffset, uint32_t colOffset)
    {
        if (uplo == ACLBLAS_LOWER) {
            // LOWER packed storage is column-major: rowOffset is the packed column.
            uint64_t column = rowOffset;
            uint64_t columnStart = column * (2ULL * static_cast<uint64_t>(n) - column + 1ULL) / 2ULL;
            return columnStart + colOffset - rowOffset;
        }
        return static_cast<uint64_t>(rowOffset) * (rowOffset + 1ULL) / 2ULL + colOffset;
    }

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> zQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> z2Queue;

    uint32_t vecIdx = 0;
    uint32_t n = 0;
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
__aicore__ inline void SpmvAIV<T>::Init(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, const SpmvTilingData& tiling)
{
    vecIdx = GetBlockIdx();

    n = tiling.n;
    useCoreNum = tiling.useCoreNum;
    alpha = tiling.alpha;
    beta = tiling.beta;
    incx = tiling.incx;
    incy = tiling.incy;
    uplo = tiling.uplo;

    if (useCoreNum == 0 || useCoreNum > SPMV_MAX_CORE_NUM) {
        useCoreNum = 1;
    }

    absIncx = (incx >= 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);
    absIncy = (incy >= 0) ? static_cast<uint32_t>(incy) : static_cast<uint32_t>(-incy);

    xGM.SetGlobalBuffer((__gm__ T*)x, (n - 1U) * absIncx + 1U);
    yGM.SetGlobalBuffer((__gm__ T*)y, (n - 1U) * absIncy + 1U);
    aGM.SetGlobalBuffer((__gm__ T*)aPacked, static_cast<uint64_t>(this->n) * this->n);

    maxDataCount = 30 * 1024 / BYTENUM_PER_FLOAT32;

    pipe.InitBuffer(aQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(xQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(x2Queue, BUFFER_NUM, sizeof(T));
    pipe.InitBuffer(zQueue, BUFFER_NUM, sizeof(T));
    pipe.InitBuffer(z2Queue, BUFFER_NUM, maxDataCount * sizeof(T));
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::CopyIn(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    uint64_t packedOffset = PackedOffset(rowOffset, colOffset);
    DataCopy(LocalA, aGM[packedOffset], dataCount);
    aQueue.EnQue<T>(LocalA);

    uint8_t paddingNum2 = elementsPerBlock - 1;
    DataCopyExtParams copyParams2{1, BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};

    LocalTensor<T> LocalX2 = x2Queue.AllocTensor<T>();
    DataCopyPad(LocalX2, xGM[XPhysicalPos(rowOffset)], copyParams2, padParams2);
    x2Queue.EnQue<T>(LocalX2);
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::Compute(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalX2 = x2Queue.DeQue<T>();
    LocalTensor<T> LocalZ = zQueue.AllocTensor<T>();
    LocalTensor<T> LocalZ2 = z2Queue.AllocTensor<T>();

    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Mul(LocalZ, LocalA, LocalX, dataCount);
    ReduceSum(LocalZ, LocalZ, LocalZ, dataCount);
    Muls(LocalZ, LocalZ, alpha, 1);
    auto scalar = LocalX2(0) * alpha;
    if (uplo == ACLBLAS_LOWER) {
        Muls(LocalZ2, LocalA, scalar, dataCount);
        if (colOffset == rowOffset) {
            LocalZ2.SetValue(0, static_cast<T>(0.0f));
        }
    } else if (colOffset + dataCount <= rowOffset) {
        Muls(LocalZ2, LocalA, scalar, dataCount);
    } else if (dataCount > 1) {
        Muls(LocalZ2, LocalA, scalar, dataCount - 1);
    }

    zQueue.EnQue<T>(LocalZ);
    z2Queue.EnQue<T>(LocalZ2);
    aQueue.FreeTensor(LocalA);
    x2Queue.FreeTensor(LocalX2);
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::CopyOut(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> zLocal = zQueue.DeQue<T>();
    if (incy == 1) {
        DataCopyPad(yGM[YPhysicalPos(rowOffset)], zLocal, copyParams);
    } else {
        uint32_t pos = YPhysicalPos(rowOffset);
        yGM.SetValue(pos, yGM.GetValue(pos) + zLocal.GetValue(0));
    }
    zQueue.FreeTensor(zLocal);

    LocalTensor<T> z2Local = z2Queue.DeQue<T>();
    if (incy == 1) {
        if (uplo == ACLBLAS_LOWER) {
            DataCopyExtParams copyParams2{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPad(yGM[colOffset], z2Local, copyParams2);
        } else if (colOffset + dataCount <= rowOffset) {
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
        if (uplo == ACLBLAS_LOWER) {
            for (uint32_t i = 0; i < dataCount; i++) {
                uint32_t pos = YPhysicalPos(colOffset + i);
                yGM.SetValue(pos, yGM.GetValue(pos) + z2Local.GetValue(i));
            }
        } else if (colOffset + dataCount <= rowOffset) {
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
__aicore__ inline void SpmvAIV<T>::CopyInPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = (elementsPerBlock - dataCount % elementsPerBlock) % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    uint64_t packedOffset = PackedOffset(rowOffset, colOffset);
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    DataCopyPad(LocalA, aGM[packedOffset], copyParams, padParams);
    aQueue.EnQue<T>(LocalA);

    uint8_t paddingNum2 = elementsPerBlock - 1;
    DataCopyExtParams copyParams2{1, BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};

    LocalTensor<T> LocalX2 = x2Queue.AllocTensor<T>();
    DataCopyPad(LocalX2, xGM[XPhysicalPos(rowOffset)], copyParams2, padParams2);
    x2Queue.EnQue<T>(LocalX2);
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::ComputePad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalX2 = x2Queue.DeQue<T>();
    LocalTensor<T> LocalZ = zQueue.AllocTensor<T>();
    LocalTensor<T> LocalZ2 = z2Queue.AllocTensor<T>();

    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    Mul(LocalZ, LocalA, LocalX, dataCount);
    ReduceSum(LocalZ, LocalZ, LocalZ, dataCount);
    Muls(LocalZ, LocalZ, alpha, 1);
    auto scalar = LocalX2(0) * alpha;
    if (uplo == ACLBLAS_LOWER) {
        Muls(LocalZ2, LocalA, scalar, dataCount);
        if (colOffset == rowOffset) {
            LocalZ2.SetValue(0, static_cast<T>(0.0f));
        }
    } else if (colOffset + dataCount <= rowOffset) {
        Muls(LocalZ2, LocalA, scalar, dataCount);
    } else if (dataCount > 1) {
        Muls(LocalZ2, LocalA, scalar, dataCount - 1);
    }

    zQueue.EnQue<T>(LocalZ);
    z2Queue.EnQue<T>(LocalZ2);
    aQueue.FreeTensor(LocalA);
    x2Queue.FreeTensor(LocalX2);
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::CopyOutPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> zLocal = zQueue.DeQue<T>();
    if (incy == 1) {
        DataCopyPad(yGM[YPhysicalPos(rowOffset)], zLocal, copyParams);
    } else {
        uint32_t pos = YPhysicalPos(rowOffset);
        yGM.SetValue(pos, yGM.GetValue(pos) + zLocal.GetValue(0));
    }
    zQueue.FreeTensor(zLocal);

    LocalTensor<T> z2Local = z2Queue.DeQue<T>();
    if (incy == 1) {
        if (uplo == ACLBLAS_LOWER) {
            DataCopyExtParams copyParams2{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPad(yGM[colOffset], z2Local, copyParams2);
        } else if (colOffset + dataCount <= rowOffset) {
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
        if (uplo == ACLBLAS_LOWER) {
            for (uint32_t i = 0; i < dataCount; i++) {
                uint32_t pos = YPhysicalPos(colOffset + i);
                yGM.SetValue(pos, yGM.GetValue(pos) + z2Local.GetValue(i));
            }
        } else if (colOffset + dataCount <= rowOffset) {
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
__aicore__ inline void SpmvAIV<T>::CopyInVec(uint32_t curOffset, uint32_t dataCount, bool needPad)
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
__aicore__ inline void SpmvAIV<T>::ComputeVec(uint32_t curOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalZ = zQueue.AllocTensor<T>();
    T betaMinusOne = beta - static_cast<T>(1.0f);
    Muls(LocalZ, LocalX, betaMinusOne, dataCount);
    zQueue.EnQue<T>(LocalZ);
    xQueue.FreeTensor(LocalX);
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::CopyOutVec(uint32_t curOffset, uint32_t dataCount, bool needPad)
{
    if (incy == 1) {
        if (!needPad) {
            LocalTensor<T> LocalZ = zQueue.DeQue<T>();
            DataCopy(yGM[curOffset], LocalZ, dataCount);
            zQueue.FreeTensor(LocalZ);
            return;
        }
        uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
        DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        LocalTensor<T> LocalZ = zQueue.DeQue<T>();
        DataCopyPad(yGM[curOffset], LocalZ, copyParams);
        zQueue.FreeTensor(LocalZ);
    } else {
        LocalTensor<T> LocalZ = zQueue.DeQue<T>();
        for (uint32_t i = 0; i < dataCount; i++) {
            uint32_t pos = YPhysicalPos(curOffset + i);
            yGM.SetValue(pos, yGM.GetValue(pos) + LocalZ.GetValue(i));
        }
        zQueue.FreeTensor(LocalZ);
    }
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::ProcessLowerTile(uint32_t packedColumn, uint32_t rowStart, uint32_t dataCount)
{
    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    if (incx == 1) {
        if (dataCount == maxDataCount) {
            DataCopy(xLocal, xGM[rowStart], dataCount);
        } else {
            uint8_t paddingNum = (elementsPerBlock - dataCount % elementsPerBlock) % elementsPerBlock;
            DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(xLocal, xGM[rowStart], copyParams, padParams);
        }
    } else {
        for (uint32_t i = 0; i < dataCount; ++i) {
            xLocal.SetValue(i, xGM.GetValue(XPhysicalPos(rowStart + i)));
        }
    }
    xQueue.EnQue<T>(xLocal);
    if (dataCount == maxDataCount) {
        CopyIn(packedColumn, rowStart, dataCount);
        Compute(packedColumn, rowStart, dataCount);
        CopyOut(packedColumn, rowStart, dataCount);
    } else {
        CopyInPad(packedColumn, rowStart, dataCount);
        ComputePad(packedColumn, rowStart, dataCount);
        CopyOutPad(packedColumn, rowStart, dataCount);
    }
    xQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::ProcessLower()
{
    for (uint32_t packedColumn = vecIdx; packedColumn < n; packedColumn += useCoreNum) {
        for (uint32_t rowStart = packedColumn; rowStart < n; rowStart += maxDataCount) {
            uint32_t dataCount = (rowStart + maxDataCount <= n) ? maxDataCount : n - rowStart;
            ProcessLowerTile(packedColumn, rowStart, dataCount);
        }
    }
}

template <typename T>
__aicore__ inline void SpmvAIV<T>::Process()
{
    if (vecIdx == 0) {
        SetAtomicAdd<T>();
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
        SetAtomicNone();
    }

    SyncAll();
    if (vecIdx >= useCoreNum) {
        return;
    }
    SetAtomicAdd<T>();

    if (uplo == ACLBLAS_LOWER) {
        ProcessLower();
        SetAtomicNone();
        return;
    }

    for (uint32_t col = 0; col < n; col += maxDataCount) {
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        uint32_t colBatchSize = (col + maxDataCount <= n) ? maxDataCount : (n - col);
        if (incx == 1) {
            if (col + maxDataCount <= n) {
                DataCopy(xLocal, xGM[col], maxDataCount);
            } else {
                uint8_t paddingNum = elementsPerBlock - colBatchSize % elementsPerBlock;
                DataCopyExtParams copyParams{1, colBatchSize * BYTENUM_PER_FLOAT32, 0, 0, 0};
                DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
                DataCopyPad(xLocal, xGM[col], copyParams, padParams);
            }
        } else {
            for (uint32_t i = 0; i < colBatchSize; i++) {
                xLocal.SetValue(i, xGM.GetValue(XPhysicalPos(col + i)));
            }
        }
        for (uint32_t row = vecIdx; row < n; row += useCoreNum) {
            if (col > row) {
                continue;
            }
            uint32_t dataCount = maxDataCount;
            if (col + dataCount > row + 1) {
                dataCount = row + 1 - col;
                xQueue.EnQue<T>(xLocal);
                CopyInPad(row, col, dataCount);
                ComputePad(row, col, dataCount);
                CopyOutPad(row, col, dataCount);
                continue;
            }
            xQueue.EnQue<T>(xLocal);
            CopyIn(row, col, dataCount);
            Compute(row, col, dataCount);
            CopyOut(row, col, dataCount);
        }
        xQueue.FreeTensor(xLocal);
    }
    SetAtomicNone();
}

__global__ __aicore__ void sspmv_kernel(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SpmvAIV<float> op;
    op.Init(aPacked, x, y, tiling);
    op.Process();
}

void sspmv_kernel_do(
    GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SpmvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    sspmv_kernel<<<numBlocks, nullptr, stream>>>(aPacked, x, y, workSpace, tiling);
}
