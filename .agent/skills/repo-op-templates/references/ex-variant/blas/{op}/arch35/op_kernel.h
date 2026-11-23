/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: Ex 变体 kernel 公共头（host.cpp / kernel.cpp 共用）
// ============================================================================
// kernel_do 签名由算子 API 与计算模型共同决定，本模板不写死参数列表。
// 共性约定（所有 ex 变体）：
//   - tiling 以 const 引用传入（禁止 GM_ADDR tilingGm），分组类例外（tiling 平铺 GM）
//   - 末位参数固定为 DTypeCase，kernel.cpp 内 switch 派发到各模板实例
//   - host.cpp 通过 #include 本头引入签名，禁止写 extern 前向声明
//
// 常见 kernel_do 拆分（按算子需求选用，非强制全部）：
//   1. {{op}}_kernel_do         主计算 kernel（Cube 或 Vector）
//   2. {{op}}_alpha_beta_do     alpha/beta 后处理 kernel（矩阵类，可选）
//   3. {{op}}_epilogue_kernel_do epilogue kernel（分组类，可选）
// ============================================================================

#pragma once

#include <cstdint>

#include "{{op}}_tiling_data.h"

// TEMPLATE: 主计算 kernel 启动器（签名按算子调整数据指针与核数参数）
// - 矩阵类：a/b/c 三 GM 指针 + tiling + DTypeCase
// - 向量类：x + alpha + tiling + numBlocks
// - 分组类：aarray/barray/workspace/tilingGm + dtypeCase（tiling 走 GM）
// 末位 DTypeCase 固定，驱动 kernel.cpp 内 switch 派发
void
{
    {
        op
    }
}
_kernel_do(
    /* GM_ADDR 数据指针（按算子）, */
    uint32_t numBlocks, void* stream,
    const {
        {
            Op
        }
    } TilingData& tiling,
    {
        {
            Op
        }
    } DTypeCase dtypeCase);

// TEMPLATE: alpha/beta 后处理 kernel 启动器（矩阵类可选）
// - 当 alpha!=1.0 || beta!=0.0 时启用：C = alpha * tempAB + beta * C_orig
// - useFP32Temp: 中间结果是否为 FP32（FP16/BF16 输入 + 需后处理时为 true）
// - 无后处理需求的 ex 变体（如向量类 scalex）删除此声明
void
{
    {
        op
    }
}
_alpha_beta_do(
    uint32_t numBlocks, void* stream, GM_ADDR tempAB, GM_ADDR cOrig, GM_ADDR cOut,
    const {
        {
            Op
        }
    } TilingData& tiling,
    {
        {
            Op
        }
    } DTypeCase dtypeCase,
    bool useFP32Temp);

// VARIANT: 分组类追加 epilogue kernel 启动器
// void {{op}}_epilogue_kernel_do(uint32_t numBlocks, void* stream, GM_ADDR carray,
//                                GM_ADDR workspace, GM_ADDR tilingGm, int dtypeCase);
