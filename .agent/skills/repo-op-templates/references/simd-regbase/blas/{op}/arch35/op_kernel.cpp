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
#include "{{op}}_kernel.h"

using namespace AscendC;

// TEMPLATE: RegBase 算子通常使用 Single Buffer（BN=1）
constexpr uint32_t BN = 1;

template <typename T>
class {
    {
        Op
    }
} AIV
{
public:
    __aicore__ inline
    {
        {
            Op
        }
    }
    AIV() {}
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workSpace, const {
            {
                Op
            }
        } TilingData & tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyTilingData(const {
        {
            Op
        }
    } TilingData & src);
    __aicore__ inline void InitUbuf();
    __aicore__ inline void CopyIn(uint32_t batchId, uint32_t offset, uint32_t count);
    __aicore__ inline void ComputeWithRegBase(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t batchId, uint32_t offset, uint32_t count);

    TPipe pipe;

    // TEMPLATE: 根据算子需求声明 GlobalTensor
    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    // TEMPLATE: UB buffers（使用 TBuf 而非 TQue）
    TBuf<QuePosition::VECCALC> inputBuf;
    TBuf<QuePosition::VECCALC> outputBuf;
    TBuf<QuePosition::VECCALC> tempBuf;

    // TEMPLATE: LocalTensor 指向 UB buffer
    LocalTensor<T> inputLocal;
    LocalTensor<T> outputLocal;
    LocalTensor<float> tempLocal;

    {
        {
            Op
        }
    }
    TilingData tiling;
    uint32_t blockIdx;
    uint32_t startBatchId;
    uint32_t calBatchNum;
};

template <typename T>
__aicore__ inline void
{
    {
        Op
    }
}
AIV<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workSpace, const {
        {
            Op
        }
    } TilingData& tiling)
{
    blockIdx = GetBlockIdx();
    CopyTilingData(tiling);

    inputGM.SetGlobalBuffer((__gm__ T*)input);
    outputGM.SetGlobalBuffer((__gm__ T*)output);

    // TEMPLATE: 计算当前 core 负责的 batch 范围
    startBatchId = blockIdx * tiling.batchPerCore;
    calBatchNum = (blockIdx == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;

    InitUbuf();
}

template <typename T>
__aicore__ inline void
{
    {
        Op
    }
}
AIV<T>::CopyTilingData(const {
    {
        Op
    }
} TilingData& src)
{
    // tiling 已通过 launch 参数（const 引用）拷贝至本地，直接按字段赋值
    // TEMPLATE: tiling.m = src.m; ...
}

template <typename T>
__aicore__ inline void
{
    {
        Op
    }
}
AIV<T>::InitUbuf()
{
    // TEMPLATE: 使用 host 预计算的 buffer 大小初始化 UB
    // pipe.InitBuffer(inputBuf, tiling.bufInput);
    // pipe.InitBuffer(outputBuf, tiling.bufOutput);
    // pipe.InitBuffer(tempBuf, tiling.bufTemp);

    // TEMPLATE: 获取 LocalTensor
    // inputLocal = inputBuf.Get<T>();
    // outputLocal = outputBuf.Get<T>();
    // tempLocal = tempBuf.Get<float>();
}

template <typename T>
__aicore__ inline void
{
    {
        Op
    }
}
AIV<T>::CopyIn(uint32_t batchId, uint32_t offset, uint32_t count)
{
    // TEMPLATE: 从 GM 拷贝数据到 UB
    // DataCopyExtParams copyParams{1, count * sizeof(T), 0, 0, 0};
    // DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    // DataCopyPad(inputLocal, inputGM[batchOffset + offset], copyParams, padParams);

    // TEMPLATE: 同步 MTE2->V
    // SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    // WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void
{
    {
        Op
    }
}
AIV<T>::ComputeWithRegBase(uint32_t count)
{
    // TEMPLATE: 获取 UB 物理地址
    __ubuf__ T* inputAddr = (__ubuf__ T*)inputLocal.GetPhyAddr();
    __ubuf__ float* outputAddr = (__ubuf__ float*)tempLocal.GetPhyAddr();

    // TEMPLATE: RegBase 计算核心 - 在 __VEC_SCOPE__ 块内使用寄存器级 API
    __VEC_SCOPE__
    {
        // TEMPLATE: 声明 RegTensor（寄存器张量）
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInput;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregOutput;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregTemp;

        // TEMPLATE: 创建全宽 mask
        auto maskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        // TEMPLATE: 初始化输出寄存器
        AscendC::MicroAPI::Duplicate<float>(vregOutput, 0.0f, maskAll);

        // TEMPLATE: 分块处理（每块 VL=64 个 float）
        constexpr uint32_t VL = 256 / sizeof(float); // 64 floats per vector register
        uint32_t loopNum = (count + VL - 1) / VL;
        uint32_t tailLen = count % VL;

        for (uint32_t i = 0; i < loopNum; i++) {
            uint32_t offset = i * VL;
            uint32_t chunk = (i == loopNum - 1 && tailLen > 0) ? tailLen : VL;

            // TEMPLATE: 处理尾部元素时使用 UpdateMask
            auto mask =
                (chunk < VL) ? AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(chunk) : maskAll;

            // TEMPLATE: 从 UB 加载数据到寄存器
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                vregInput, (__ubuf__ float*)(inputAddr + offset));

            // TEMPLATE: 寄存器级计算（Mul/Add/ReduceSum 等）
            AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                vregTemp, vregInput, vregInput, mask);
            AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                vregOutput, vregOutput, vregTemp, maskAll);
        }

        // TEMPLATE: 归约求和（如需要）
        // AscendC::MicroAPI::ReduceSum(vregOutput, vregOutput, maskAll);

        // TEMPLATE: 将结果写回 UB
        AscendC::MicroAPI::DataCopy<
            float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(outputAddr, vregOutput, 1, maskAll);
    }
    // __VEC_SCOPE__ 结束，自动同步 V->MTE3
}

template <typename T>
__aicore__ inline void
{
    {
        Op
    }
}
AIV<T>::CopyOut(uint32_t batchId, uint32_t offset, uint32_t count)
{
    // TEMPLATE: 同步 V->MTE3（等待 Vector 计算完成）
    // SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    // TEMPLATE: 从 UB 拷贝结果到 GM
    // DataCopyExtParams copyParams{1, count * sizeof(T), 0, 0, 0};
    // DataCopyPad(outputGM[batchOffset + offset], outputLocal, copyParams);

    // TEMPLATE: 同步 V->MTE2（等待下一次 CopyIn 前确保 MTE3 完成）
    // SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    // WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void
{
    {
        Op
    }
}
AIV<T>::Process()
{
    if (calBatchNum == 0)
        return;

    // TEMPLATE: 遍历每个 batch
    for (uint32_t b = 0; b < calBatchNum; b++) {
        uint32_t batchId = startBatchId + b;

        // TEMPLATE: 拷贝输入数据
        CopyIn(batchId, 0, /* count */);

        // TEMPLATE: RegBase 计算
        ComputeWithRegBase(/* count */);

        // TEMPLATE: 拷贝输出结果
        CopyOut(batchId, 0, /* count */);
    }
}

extern "C" __global__ __aicore__ void
{
    {
        op
    }
}
_kernel(
    // TEMPLATE: 添加算子特定的 GM 指针参数
    GM_ADDR workSpace, const {
        {
            Op
        }
    } TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TEMPLATE: 根据 dtype 实例化具体类型的算子
    // 示例：
    // if (tiling.dtype == 0) {  // half
    //     {{Op}}AIV<half> op;
    //     op.Init(/* GM 指针 */, workSpace, tiling);
    //     op.Process();
    // } else if (tiling.dtype == 1) {  // float
    //     {{Op}}AIV<float> op;
    //     op.Init(/* GM 指针 */, workSpace, tiling);
    //     op.Process();
    // }
}

// Tiling 使用 const 值传递（通过 launch 参数从 host 侧自动拷贝）
void
{
    {
        op
    }
}
_kernel_do(
    // TEMPLATE: 添加算子特定的 GM 指针参数
    GM_ADDR workSpace, uint32_t numBlocks,
    const {
        {
            Op
        }
    } TilingData& tiling,
    void* stream)
{
    {
        {
            op
        }
    }
    _kernel<<<numBlocks, nullptr, stream>>>(
        // TEMPLATE: 传入算子特定的 GM 指针
        workSpace, tiling);
}
