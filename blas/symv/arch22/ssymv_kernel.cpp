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
    __aicore__ inline void CopyInY(uint32_t curOffset, uint32_t dataCount, bool needPad);
    __aicore__ inline void ComputeYBeta(uint32_t dataCount);
    __aicore__ inline void CopyOutYBeta(uint32_t curOffset, uint32_t dataCount, bool needPad);

private:
    TPipe pipe;

    // issue #840: PadV2 变体与 V2 逐行比对仅 DataCopy/DataCopyPad 及 pad 参数之别
    // （Compute 两版完全相同），合并为单一实现以 needPad 参数区分；同批删除无
    // 调用点的 CopyInVec/ComputeVec/CopyOutVec 旧路径（两阶段改造前遗留，
    // ComputeVec 的 betaMinusOne 语义与现行 phase 0 直接乘 beta 不一致）。
    // CopyOut 的 V2/PadV2 两版逐行比对仅 sizeof(T) 的 cast 写法不同（uint32 宽化
    // vs uint16，字段实际为 uint16_t，语义等价），z/z2 半段完全一致，直接合一
    // 并统一为 uint16，无 needPad 参数。
    __aicore__ inline void CopyInV2(
        uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount, bool needPad);
    __aicore__ inline void ComputeV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutV2(uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void LoadX2(uint32_t rowOffset)
    {
        uint8_t paddingNum2 = elementsPerBlock - 1;
        DataCopyExtParams copyParams2{1, BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams2{true, 0, paddingNum2, 0};
        LocalTensor<T> LocalX2 = x2Queue.AllocTensor<T>();
        DataCopyPad(LocalX2, xGM[XPhysicalPos(rowOffset)], copyParams2, padParams2);
        x2Queue.EnQue<T>(LocalX2);
    }

    __aicore__ inline uint64_t XPhysicalPos(uint32_t logical)
    {
        // issue #807: 物理位置换算统一 uint64 域（对齐同函数 aGM 尺寸的 64 位写法），
        // (n-1)*|incx| 不小于 2^32 时 uint32 乘法回绕会索引错误地址。
        return (incx >= 0) ? (static_cast<uint64_t>(logical) * absIncx) :
                             (static_cast<uint64_t>(n - 1U - logical) * absIncx);
    }
    __aicore__ inline uint64_t YPhysicalPos(uint32_t logical)
    {
        return (incy >= 0) ? (static_cast<uint64_t>(logical) * absIncy) :
                             (static_cast<uint64_t>(n - 1U - logical) * absIncy);
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
    uint32_t tilingPhase = 1;

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
    tilingPhase = tiling.phase;

    if (useCoreNum == 0 || useCoreNum > SYMV_MAX_CORE_NUM) {
        useCoreNum = 1;
    }

    absIncx = (incx >= 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);
    absIncy = (incy >= 0) ? static_cast<uint32_t>(incy) : static_cast<uint32_t>(-incy);

    // issue #807: x/y 缓冲长度按 uint64 计算（原 uint32 表达式在跨度不小于 2^32 时溢出，
    // SetGlobalBuffer 拿到错误长度）；aGM 尺寸已是 uint64 口径。
    xGM.SetGlobalBuffer((__gm__ T*)x, static_cast<uint64_t>(n - 1U) * absIncx + 1U);
    yGM.SetGlobalBuffer((__gm__ T*)y, static_cast<uint64_t>(n - 1U) * absIncy + 1U);
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
    uint32_t taskIdx, uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount, bool needPad)
{
    uint64_t r = static_cast<uint64_t>(rowOffset) * lda + colOffset;
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    if (!needPad) {
        DataCopy(LocalA, aGM[r], dataCount);
    } else {
        uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
        DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        DataCopyPad(LocalA, aGM[r], copyParams, padParams);
    }
    aQueue.EnQue<T>(LocalA);

    LocalTensor<T> LocalX = xQueue.AllocTensor<T>();
    if (incx == 1) {
        if (!needPad) {
            DataCopy(LocalX, xGM[colOffset], dataCount);
        } else {
            uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
            DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(LocalX, xGM[colOffset], copyParams, padParams);
        }
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
        // issue #840: 统一 cast 类型（DataCopyExtParams 的 blockLen 字段为
        // uint16_t，原 V2 用 uint32 宽化、PadV2 用 uint16，同语义不同写法）
        DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(yGM[rowOffset], zLocal, copyParams);
    } else {
        uint64_t pos = YPhysicalPos(rowOffset);
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
                uint64_t pos = YPhysicalPos(colOffset + i);
                yGM.SetValue(pos, yGM.GetValue(pos) + z2Local.GetValue(i));
            }
        } else if (dataCount > 1) {
            for (uint32_t i = 0; i < dataCount - 1; i++) {
                uint64_t pos = YPhysicalPos(colOffset + i);
                yGM.SetValue(pos, yGM.GetValue(pos) + z2Local.GetValue(i));
            }
        }
    }
    z2Queue.FreeTensor(z2Local);
}

// ---- phase 0：y = beta*y，覆盖写（非原子），单核执行 ----
template <typename T>
__aicore__ inline void SymvAIV<T>::CopyInY(uint32_t curOffset, uint32_t dataCount, bool needPad)
{
    if (incy == 1) {
        LocalTensor<T> LocalY = xQueue.AllocTensor<T>();
        if (!needPad) {
            DataCopy(LocalY, yGM[curOffset], dataCount);
        } else {
            uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
            DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(LocalY, yGM[curOffset], copyParams, padParams);
        }
        xQueue.EnQue<T>(LocalY);
    } else {
        LocalTensor<T> LocalY = xQueue.AllocTensor<T>();
        for (uint32_t i = 0; i < dataCount; i++) {
            LocalY.SetValue(i, yGM.GetValue(YPhysicalPos(curOffset + i)));
        }
        xQueue.EnQue<T>(LocalY);
    }
}

template <typename T>
__aicore__ inline void SymvAIV<T>::ComputeYBeta(uint32_t dataCount)
{
    LocalTensor<T> LocalY = xQueue.DeQue<T>();
    LocalTensor<T> LocalZ = z2Queue.AllocTensor<T>();
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    Muls(LocalZ, LocalY, beta, dataCount);
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    z2Queue.EnQue<T>(LocalZ);
    xQueue.FreeTensor(LocalY);
}

template <typename T>
__aicore__ inline void SymvAIV<T>::CopyOutYBeta(uint32_t curOffset, uint32_t dataCount, bool needPad)
{
    if (incy == 1) {
        LocalTensor<T> LocalZ = z2Queue.DeQue<T>();
        if (!needPad) {
            DataCopy(yGM[curOffset], LocalZ, dataCount);
        } else {
            DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPad(yGM[curOffset], LocalZ, copyParams);
        }
        z2Queue.FreeTensor(LocalZ);
    } else {
        LocalTensor<T> LocalZ = z2Queue.DeQue<T>();
        for (uint32_t i = 0; i < dataCount; i++) {
            uint64_t pos = YPhysicalPos(curOffset + i);
            yGM.SetValue(pos, LocalZ.GetValue(i));
        }
        z2Queue.FreeTensor(LocalZ);
    }
}

template <typename T>
__aicore__ inline void SymvAIV<T>::Process()
{
    // y = alpha*A*x + beta*y 分两次 kernel 启动执行（host 侧下发两次，同一
    // stream 顺序保证互斥），消除原"单启动内 core0 读改写 y 与其余核原子
    // 累加 y"的跨核竞态（n=255 UPPER 冒烟实测 diff=(beta-1)*row3 标量贡献）：
    // phase 0：y = beta*y，core0 单核覆盖写（非原子）；
    // phase 1：SetAtomicAdd 下各核原子累加 alpha*A*x 各元素贡献。
    if (tilingPhase == 0) {
        if (vecIdx != 0) {
            return;
        }
        uint32_t repeatTimes = n / maxDataCount;
        uint32_t remainNum = n % maxDataCount;
        uint32_t curOffset = 0;
        for (uint32_t i = 0; i < repeatTimes; i++) {
            uint32_t dataCount = maxDataCount;
            CopyInY(curOffset, dataCount, false);
            ComputeYBeta(dataCount);
            CopyOutYBeta(curOffset, dataCount, false);
            curOffset += dataCount;
        }
        if (remainNum > 0) {
            uint32_t dataCount = remainNum;
            CopyInY(curOffset, dataCount, true);
            ComputeYBeta(dataCount);
            CopyOutYBeta(curOffset, dataCount, true);
        }
        return;
    }

    SetAtomicAdd<T>();
    if (vecIdx >= useCoreNum) {
        SetAtomicNone();
        return;
    }
    for (uint32_t taskIdx = vecIdx; taskIdx < n; taskIdx += useCoreNum) {
        uint32_t row = taskIdx;
        if (uplo == ACLBLAS_UPPER) {
            uint32_t colLen = row + 1;
            for (uint32_t col = 0; col < colLen; col += maxDataCount) {
                uint32_t dataCount = maxDataCount;
                if (col + dataCount > colLen) {
                    dataCount = colLen - col;
                    CopyInV2(taskIdx, row, col, dataCount, true);
                    ComputeV2(taskIdx, row, col, dataCount);
                    CopyOutV2(taskIdx, row, col, dataCount);
                    continue;
                }
                CopyInV2(taskIdx, row, col, dataCount, false);
                ComputeV2(taskIdx, row, col, dataCount);
                CopyOutV2(taskIdx, row, col, dataCount);
            }
        } else {
            // LOWER：对角元单独作 1 元素 chunk（dataCount=1 时 z2 自然为空），
            // 其余 chunk 严格位于对角线上方，全部走纯三角路径
            CopyInV2(taskIdx, row, row, 1, true);
            ComputeV2(taskIdx, row, row, 1);
            CopyOutV2(taskIdx, row, row, 1);
            for (uint32_t col = row + 1; col < n; col += maxDataCount) {
                uint32_t dataCount = maxDataCount;
                if (col + dataCount > n) {
                    dataCount = n - col;
                    CopyInV2(taskIdx, row, col, dataCount, true);
                    ComputeV2(taskIdx, row, col, dataCount);
                    CopyOutV2(taskIdx, row, col, dataCount);
                    continue;
                }
                CopyInV2(taskIdx, row, col, dataCount, false);
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
