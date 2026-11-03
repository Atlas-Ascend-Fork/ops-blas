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

#include <algorithm>
#include <cstdint>

// issue #743: gemv_batched / gemv_strided_batched arch35 共享的类型定义与
// tile 搜索逻辑。从两个 host 的复制粘贴版本提取为共享头文件（CodeCheck
// 重复代码治理）。

enum class GemvType { S, HSH, HSS, TST, TSS };

struct GemvBufCfg {
    uint32_t szIn; // sizeof(A/x element)
    uint32_t szY;  // sizeof(y element)
    uint32_t tn;   // matTmpBuf copies (1=reuse, 2=separate)
    bool hasExtra; // matPreBuf + vecPreBuf needed
};

inline constexpr GemvBufCfg kGmvCfg_F32 = {sizeof(float), sizeof(float), 1, false};
inline constexpr GemvBufCfg kGmvCfg_HSH = {sizeof(uint16_t), sizeof(uint16_t), 2, true};
inline constexpr GemvBufCfg kGmvCfg_HSS = {sizeof(uint16_t), sizeof(float), 2, true};
inline constexpr GemvBufCfg kGmvCfg_TST = {sizeof(uint16_t), sizeof(uint16_t), 2, true};
inline constexpr GemvBufCfg kGmvCfg_TSS = {sizeof(uint16_t), sizeof(float), 2, true};

inline GemvType SelectGemvType(uint32_t dtype)
{
    return (dtype == 2) ? GemvType::HSH :
           (dtype == 3) ? GemvType::TST :
           (dtype == 4) ? GemvType::TSS :
           (dtype != 0) ? GemvType::S :
                          GemvType::HSH;
}

// Binary-search max values of `arg` (row count or col count) s.t. f(arg) <= UB.
// `f` takes a single uint32_t and returns total bytes per batch.
template <typename F>
inline uint32_t BinaryMaxRowsCols(uint32_t UB, uint32_t hi, F&& f)
{
    uint32_t lo = 0;
    while (lo < hi) {
        uint32_t mid = (lo + hi + 1) / 2;
        if (f(mid) <= UB) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}
