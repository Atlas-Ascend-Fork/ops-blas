/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 4)

// TEMPLATE: 根据算子需求定义 Tiling 结构体
// RegBase 算子通常需要额外的 UB buffer 大小信息
struct {
    {
        Op
    }
} TilingData{
    // TEMPLATE: 基础维度参数
    // uint32_t m;
    // uint32_t n;

    // TEMPLATE: 标量参数
    // float alpha;
    // float beta;

    // TEMPLATE: 多核切分参数
    // uint32_t coreNum;
    // uint32_t usedCoreNum;
    // uint32_t batchPerCore;
    // uint32_t batchTail;

    // TEMPLATE: UB buffer 大小（host 侧预计算，256B 对齐）
    // uint32_t bufInput;
    // uint32_t bufOutput;
    // uint32_t bufTemp;
};

#pragma pack(pop)
