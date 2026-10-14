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

// Packed layout constraints shared by Host and Kernel.
constexpr uint32_t CTRSM_CONVERT_TILE = 64;
constexpr uint32_t CTRSM_PANEL_SIZE = 32;
constexpr uint32_t CTRSM_GROUP_PANELS = 4;
constexpr uint32_t CTRSM_SPLIT_COLS = 64;
constexpr uint32_t CTRSM_RISK_WORDS = 8; // Each producer owns one 32-byte GM block.

struct CtrsmTilingData {
    uint32_t side;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    // Public dimensions are nonnegative int values; keep loop bounds in 32 bits.
    // Leading dimensions and derived GM pitches participate in 64-bit addressing.
    uint32_t m;
    uint32_t n;
    uint64_t lda;
    uint64_t ldb;
    float alphaReal;
    float alphaImag;
    uint32_t numThreads;
    uint32_t systemCount;
    uint32_t perCoreSystems;
    uint32_t coreRemainder;
    uint32_t effectiveUpper;
    uint32_t order;
    uint32_t rhsCount;
    uint32_t panelSize;
    uint32_t groupPanels;
    uint32_t splitCount;
    uint32_t splitCols;
    uint32_t originalOrder;
    uint64_t combinedCols;
    uint64_t riskOffset;
    uint32_t prepareCores;
};
