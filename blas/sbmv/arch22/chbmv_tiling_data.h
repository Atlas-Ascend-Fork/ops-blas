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

constexpr uint32_t CHBMV_ROWS = 64;
constexpr uint32_t CHBMV_DIAGONAL_ROWS = 128;
constexpr uint32_t CHBMV_DIAGONAL_SMALL_ROWS = 32;
constexpr uint32_t CHBMV_DIAGONAL_MEDIUM_ROWS = 64;
constexpr uint32_t CHBMV_DIAGONAL_SMALL_N = 256;
constexpr uint32_t CHBMV_DIAGONAL_MEDIUM_N = 512;
constexpr uint32_t CHBMV_DIAGONAL_SPAN_BYTES = 144 * 1024;
constexpr uint32_t CHBMV_REDUCE = 256;
constexpr uint32_t CHBMV_SMALL_BAND = 64;
constexpr uint32_t CHBMV_SPAN_BYTES = 128 * 1024;
constexpr uint32_t CHBMV_PATH_ROW_DOT = 0;
constexpr uint32_t CHBMV_PATH_DIAGONALS = 1;

struct ChbmvTilingData {
    uint32_t n;
    uint32_t k; // Physical band origin: never replace this with min(k, n-1).
    uint32_t effectiveK;
    uint32_t lda;
    uint32_t upper;
    uint32_t blockCount;
    uint32_t path;
    uint32_t reserved;
    int64_t incx;
    int64_t incy;
    float alphaR;
    float alphaI;
    float betaR;
    float betaI;
};
