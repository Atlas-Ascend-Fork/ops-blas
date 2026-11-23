/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: Tiling 数据结构头文件（host/kernel 共享）
// - host 侧计算后以 const 引用传入 kernel_do，kernel 以 by value 接收（运行时 launch 参数自动拷贝）
// - 字段完全由算子的 Tiling 策略决定

#pragma once

#include <cstdint>

// TEMPLATE: 根据算子的多核切分策略自定义字段
// 示例 A — 简单向量算子（标量均分）：
//   uint32_t totalN; uint32_t perCoreN; uint32_t remainder; uint32_t tileSize;
// 示例 B — 归约类算子（数组分配）：
//   int64_t n; uint32_t useCoreNum;
//   uint32_t startOffset[MAX_CORE]; uint32_t calCount[MAX_CORE];
// 示例 C — 矩阵算子（二维切分）：
//   uint32_t m; uint32_t n; uint32_t lda; uint32_t tileM; uint32_t tileN;
struct {
    {
        Op
    }
} TilingData{
    // TEMPLATE: 按算子需求填写字段
};

// TEMPLATE: 根据 dtype 调整（float=8, half=16, double=4）
// 建议：使用算子实际的数据类型，而非硬编码 float
constexpr uint32_t ELEMENTS_PER_BLOCK = 32 / sizeof(/* dtype */ float);
