/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ctpsv_kernel_utils.h
 * \brief Shared helpers for ctpsv arch22 kernel files. Included only from kernel compilation units.
 *
 * packed 索引（0 基，列主序堆叠各列含对角，单位 = 复数元素；LOWER 式为任务书 §2.1.2
 * 勘误版公式，CP1 裁决采信——spec.yaml shape_constraints.notes 三方一致锁定）：
 *   UPPER: A(i,j) = AP[i + j(j+1)/2], i ≤ j
 *   LOWER: A(i,j) = AP[i + (2n-j-1)·j/2], i ≥ j
 * 索引中间积 uint64/int64：n 近 65535 时 (2n-j-1)*j 与 j(2n-j+1)/2 均超 uint32 域
 * （DESIGN §3.9 索引安全域论证，n ≤ 65535 ⇒ n(n+1)/2 < 2^31）。
 */

#pragma once
#include <cstdint>

// 元素级 packed 索引（文档/对拍口径；kernel 主路径用下方列起点函数）
__aicore__ inline uint32_t CtpsvPackedUpperIdx(uint32_t i, uint32_t j)
{
    return static_cast<uint32_t>(
        static_cast<uint64_t>(i) + static_cast<uint64_t>(j) * (static_cast<uint64_t>(j) + 1) / 2);
}

__aicore__ inline uint32_t CtpsvPackedLowerIdx(uint32_t i, uint32_t j, uint32_t n)
{
    const uint64_t mid = (2ULL * n - j - 1) * j / 2;
    return static_cast<uint32_t>(static_cast<uint64_t>(i) + mid);
}

// LOWER 列 j 在 AP 内的起点（列存行 j..n-1，长 n-j）：j(2n-j+1)/2（复数元素单位；
// int64 中间积防上溢）
__aicore__ inline int64_t CtpsvLowerColStart(int32_t j, int32_t n)
{
    return static_cast<int64_t>(j) * (2LL * n - j + 1) / 2;
}

// UPPER 列 j 在 AP 内的起点（列存行 0..j，长 j+1）：j(j+1)/2（复数元素单位）
__aicore__ inline int64_t CtpsvUpperColStart(int32_t j)
{
    return static_cast<int64_t>(j) * (static_cast<int64_t>(j) + 1) / 2;
}
