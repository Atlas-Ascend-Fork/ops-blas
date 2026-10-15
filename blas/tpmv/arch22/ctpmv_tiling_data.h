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

constexpr uint32_t CTPMV_ARCH22_MAX_AIV_CORES = 40U;
constexpr uint32_t CTPMV_MAX_N = 4096U;
constexpr uint64_t CTPMV_BATCHED_COL_FIXED_COST = 512ULL;
constexpr uint64_t CTPMV_BATCHED_CORE_ENGAGE_WEIGHT = 3800ULL;

constexpr uint64_t CtpmvBatchedTotalWeight(uint32_t n)
{
    const uint64_t nn = n;
    return nn * CTPMV_BATCHED_COL_FIXED_COST + ((nn * (nn + 1ULL)) >> 1ULL);
}

enum CtpmvPath : uint32_t {
    CTPMV_PATH_COLUMN_PARTIALS = 15U,
    CTPMV_PATH_UPPER_TRANSPOSE_PANEL = 19U,
    CTPMV_PATH_BATCHED_REFERENCE = 22U,
    CTPMV_PATH_STRICT_SCALAR = 24U,
};

struct CtpmvTilingData {
    uint32_t n;
    uint32_t coreNum;
    int64_t incx;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    uint32_t path;
    uint64_t syncAddress;
};

static_assert(sizeof(CtpmvTilingData) == 40U, "Ctpmv launch tiling must stay on the compact common-path ABI");
