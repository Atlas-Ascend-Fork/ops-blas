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

struct CtpttrTilingData {
    uint64_t packedElements;
    uint32_t n;
    uint32_t lda;
    uint32_t upper;
    uint32_t useCoreNum;
};

inline CtpttrTilingData MakeCtpttrTiling(uint32_t n, uint32_t lda, bool upper, uint32_t coreCount)
{
    CtpttrTilingData tiling{};
    tiling.n = n;
    tiling.lda = lda;
    tiling.upper = upper ? 1U : 0U;
    tiling.packedElements = static_cast<uint64_t>(n) * (static_cast<uint64_t>(n) + 1U) / 2U;
    // One column is the smallest copy unit, so the blocks never outnumber the columns.
    tiling.useCoreNum = std::min<uint32_t>(coreCount, std::max<uint32_t>(n, 1U));
    if (tiling.useCoreNum == 0U) {
        tiling.useCoreNum = 1U;
    }
    return tiling;
}
