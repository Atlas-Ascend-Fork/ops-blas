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

// The solver leaves one third of the platform-reported UB outside its
// cache/scratch budget. This preserves the established arch22 headroom without
// assuming a fixed hardware capacity.
constexpr uint32_t STBSV_UB_BUDGET_NUMERATOR = 2;
constexpr uint32_t STBSV_UB_BUDGET_DENOMINATOR = 3;
constexpr uint32_t STBSV_UB_ALIGNMENT_BYTES = 32;

// Cache-policy limit in float elements, not the device's UB capacity.
constexpr uint32_t STBSV_MAX_NARROW_WINDOW_ELEMENTS = 8 * 1024;

struct StbsvTilingData {
    uint64_t a;
    uint64_t x;
    uint32_t n;
    uint32_t k;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    int32_t incx;
    uint32_t lda;
    // Usable, aligned cache/scratch budget derived from the current platform.
    // Zero selects the constant-memory scalar fallback.
    uint64_t ubSizeBytes;
};
