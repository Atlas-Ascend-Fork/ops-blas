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
// SIMT 算子的 TilingData 通常包含 nthreads（每 block 线程数）和算子标量参数
// SIMT 常量：SIMT_MIN_THREAD_NUM=128, SIMT_MAX_THREAD_NUM=2048

#pragma once

#include <cstdint>

// TEMPLATE: 按算子需求填写字段
// SIMT 特有字段：nthreads（每 block 线程数，host 侧计算）
// 标量参数（alpha/beta/uplo/trans 等）直接放入 tiling 传给 kernel，避免 kernel 再读 GM
struct {
    {
        Op
    }
} TilingData
{
    uint32_t nthreads;
    // TEMPLATE: 算子维度参数（如 n, m, k, lda, incx, incy, ...）
    // TEMPLATE: 算子标量参数（如 alpha, beta, uplo, trans, ...）
};
