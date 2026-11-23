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
#include "{{op}}_tiling_data.h"

// kernel_do 函数声明：host 侧调用此函数启动 kernel
// - tiling 以 const 引用传入（禁止 GM_ADDR tilingGm）
// - 数据 GM 指针通过 GM_ADDR 统一传递（与 kernel.cpp 签名一致）
// - 异步 launch（host 返回后 kernel 仍在 stream 上执行）
void
{
    {
        op
    }
}
_kernel_do(
    /* GM_ADDR 各数据指针（按算子需求调整）, */
    GM_ADDR workSpace, uint32_t numBlocks,
    const {
        {
            Op
        }
    } TilingData& tiling,
    void* stream);
