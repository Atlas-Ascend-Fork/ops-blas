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

constexpr uint32_t STRSMBATCHED_PANEL_NB = 64U;
constexpr uint32_t STRSMBATCHED_GEMM_TILE = 128U;
constexpr uint32_t STRSMBATCHED_GEMM_BASE_K = 64U;
constexpr uint32_t STRSMBATCHED_GEMM_SINGLE_K = 512U;

struct StrsmbatchedTilingData {
    uint32_t side;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t batchCount;
    uint32_t panelNb;
    uint32_t groupPanels;
    uint32_t tempLd;
    uint64_t tempBatchStride;
    float alpha;
    uint32_t reserved;
};
