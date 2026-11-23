/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: SIMD membase Kernel 实现
// 标准骨架：Kernel 类（Init + Process）+ kernel entry + launcher
// 关键模式：
//   1. TPipe 在 kernel entry 中创建，传指针给类
//   2. TQue/TQueBind 管理 UB buffer
//   3. DataCopyPad 处理 GM↔UB 搬运（含尾部对齐）
//   4. EnQue/DeQue 提供隐式同步
//   5. KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)

#include <cstdint>
#include "kernel_operator.h"
#include "{{op}}_kernel.h"

using namespace AscendC;

// TEMPLATE: BUFFER_NUM=1 纯搬运算子；BUFFER_NUM=2 有 Vector 计算需 CopyIn/Compute 重叠
constexpr uint32_t BUFFER_NUM = 1;

// TEMPLATE: 算子 Kernel 类
// - 简单算子用 XxxAIV（无模板参数）
// - 多 dtype 算子用 XxxKernel<T>
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
    // TEMPLATE: Init 参数与 kernel entry 签名对应，按算子的 GM 指针数量调整
    __aicore__ inline void Init(
        /* GM_ADDR 各输入/输出, */
        const {
            {
                Op
            }
        } TilingData &
            tiling,
        TPipe * pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyTilingData(const {
        {
            Op
        }
    } TilingData & src);
    __aicore__ inline void SingleIteration(/* 当前 tile 的偏移和长度等 */);

    TPipe* pipe_;
    // TEMPLATE: 按算子的输入/输出数量和 dtype 声明 GlobalTensor
    // GlobalTensor<float> inputGM_;
    // GlobalTensor<float> outputGM_;
    // TEMPLATE: 按算子需求选择队列类型
    // - TQue<VECIN, BUFFER_NUM>: 纯输入队列
    // - TQue<VECOUT, BUFFER_NUM>: 纯输出队列
    // - TQueBind<VECIN, VECOUT, 1>: 同一 buffer 既输入又输出
    // TEMPLATE: 如有 Vector 计算阶段，可能需要 TBuf<VECCALC> 做暂存
    {
        {
            Op
        }
    }
    TilingData tiling_;
    uint32_t blockIdx_;
    // TEMPLATE: 按多核切分策略声明（标量 or 数组索引等）
};

__aicore__ inline void
{
    {
        Op
    }
}
AIV::Init(
    /* GM_ADDR 各输入/输出, */
    const {
        {
            Op
        }
    } TilingData& tiling,
    TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    CopyTilingData(tiling);

    // TEMPLATE: 多核切分 — 计算当前核负责的偏移和数量
    // 方式 A（标量均分）：myOffset_ = blockIdx_ * perCoreN; lastCore += remainder;
    // 方式 B（数组查表）：myOffset_ = tiling_.startOffset[blockIdx_]; myCount_ = tiling_.calCount[blockIdx_];

    // TEMPLATE: SetGlobalBuffer — 按算子的 GM 指针逐个绑定
    // inputGM_.SetGlobalBuffer((__gm__ float*)inputPtr, totalElements);

    // TEMPLATE: InitBuffer — 按队列数量和 tile 大小初始化
    // pipe_->InitBuffer(queue_, BUFFER_NUM, tileSize * sizeof(dtype));
}

__aicore__ inline void
{
    {
        Op
    }
}
AIV::CopyTilingData(const {
    {
        Op
    }
} TilingData& src)
{
    // tiling 已通过 launch 参数拷贝至本地，直接按字段赋值即可
    // TEMPLATE: tiling_.field1 = src.field1; ...
}

__aicore__ inline void
{
    {
        Op
    }
}
AIV::SingleIteration(/* 当前 tile 参数 */)
{
    // TEMPLATE: 单次 tile 处理逻辑，标准三段式：
    //
    // 1. CopyIn: GM -> UB (MTE2)
    //    LocalTensor<T> local = queue_.AllocTensor<T>();
    //    DataCopy(local, gm_[offset], alignedCount);          // 对齐部分
    //    DataCopyPad(local[aligned], gm_[offset+aligned], ...); // 尾部
    //    queue_.EnQue<T>(local);
    //    // 注意：EnQue 提供隐式 MTE2->MTE3 同步，无需额外 SetFlag/WaitFlag
    //
    // 2. Vector 计算（如有）
    //    // 如果需要显式同步（如等待 MTE3 完成），使用：
    //    SetFlag<HardEvent::MTE3_V>();
    //    WaitFlag<HardEvent::MTE3_V>();
    //    计算：Mul / Add / ReduceSum / ...
    //    // 计算完成后同步：
    //    SetFlag<HardEvent::V_MTE3>();
    //    WaitFlag<HardEvent::V_MTE3>();
    //
    // 3. CopyOut: UB -> GM (MTE3)
    //    LocalTensor<T> out = queue_.DeQue<T>();
    //    DataCopy(gm_[offset], out, alignedCount);
    //    queue_.FreeTensor(out);
}

__aicore__ inline void
{
    {
        Op
    }
}
AIV::Process()
{
    // TEMPLATE: 空数据提前返回
    // if (myCount_ == 0) return;

    // TEMPLATE: tile 循环 + 尾部处理
    // uint32_t tileLoop = myCount_ / tiling_.tileSize;
    // uint32_t tileTail = myCount_ % tiling_.tileSize;
    // for (...) SingleIteration(...);
    // if (tileTail > 0) SingleIteration(...);
}

// TEMPLATE: Kernel 入口函数
// - 参数列表 = 所有 GM 指针 + workSpace + const TilingData tiling（by value，运行时 launch 参数自动拷贝）
// - workSpace 是标准参数，即使不用也保留（传 nullptr）
// - KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
// - TPipe 在入口函数中创建
extern "C" __global__ __aicore__ void
{
    {
        op
    }
}
_kernel(
    /* GM_ADDR ..., */
    GM_ADDR workSpace, const {
        {
            Op
        }
    } TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    {
        {
            Op
        }
    }
    AIV op;
    op.Init(/* ..., */ tiling, &pipe);
    op.Process();
}

// TEMPLATE: Kernel 启动器（host 侧调用）
// - 参数 = GM 指针 + workSpace + numBlocks + const TilingData 引用 + stream
// - Tiling 通过 const 引用从 host 传入，kernel launch 时自动拷贝至 kernel 函数参数（by value）
void
{
    {
        op
    }
}
_kernel_do(
    /* GM_ADDR ..., */
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
    _kernel<<<numBlocks, nullptr, stream>>>(/* ..., */ workSpace, tiling);
}
