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

// gemv_batched 与 cgemv_batched arch35 共享的形状跨度域校验，从两处 host 的同源
// 校验块提取为公共函数（CodeCheck 重复代码治理）。kernel 侧 A/x/y 下标与总输出
// 的 32 位安全上界：
//  - A 单批元素跨度（两种 trans 布局的 A 下标最大值同式）：(n-1)*lda + m；
//  - x 跨度 1 + (dotDim-1)*|incx|、y 跨度 1 + (outDim-1)*|incy|
//    （dotDim = N ? n : m，outDim = N ? m : n，含负步长场景）；
//  - 总输出 batchCount * outDim。
// isOpN 传 trans 是否为 ACLBLAS_OP_N；elemFactor 为单元素按 float 寻址的放大因子
// （实数按元素寻址取 1，复数按 2×float 寻址取 2），A/x/y 跨度乘该因子后与
// INT32_MAX 比较，总输出数为元素个数不放大。返回 true 表示超限，调用方记录日志
// 并拒绝下发 kernel。前提：incx/incy 非 0 且非 INT_MIN、batchCount > 0（由调用方校验）。
inline bool GemvBatchedSpanExceedsInt32(
    bool isOpN, int64_t elemFactor, int m, int n, int lda, int incx, int incy, int batchCount)
{
    const int64_t dotDimSpan = isOpN ? static_cast<int64_t>(n) : static_cast<int64_t>(m);
    const int64_t outDimSpan = isOpN ? static_cast<int64_t>(m) : static_cast<int64_t>(n);
    const int64_t absIncx64 = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    const int64_t absIncy64 = (incy >= 0) ? static_cast<int64_t>(incy) : -static_cast<int64_t>(incy);
    const int64_t aSpan = (static_cast<int64_t>(n) - 1) * static_cast<int64_t>(lda) + static_cast<int64_t>(m);
    const int64_t xSpan = (dotDimSpan - 1) * absIncx64 + 1;
    const int64_t ySpan = (outDimSpan - 1) * absIncy64 + 1;
    const int64_t totalOutputs = static_cast<int64_t>(batchCount) * outDimSpan;
    return aSpan * elemFactor > INT32_MAX || xSpan * elemFactor > INT32_MAX || ySpan * elemFactor > INT32_MAX ||
           totalOutputs > INT32_MAX;
}
