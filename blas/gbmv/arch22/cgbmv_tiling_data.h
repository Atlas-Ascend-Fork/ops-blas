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

constexpr uint32_t CGBMV_PATH_BETA_SCALE = 0;
constexpr uint32_t CGBMV_PATH_TINY = 1;
constexpr uint32_t CGBMV_PATH_ROW_DOT = 2;
constexpr uint32_t CGBMV_PATH_COL_OWNER = 3;

struct CgbmvTilingData {
    uint64_t a;
    uint64_t x;
    uint64_t y;
    uint32_t m;
    uint32_t n;
    uint32_t kl;
    uint32_t ku;
    uint32_t lda;
    uint32_t trans;
    float alphaR;
    float alphaI;
    float betaR;
    float betaI;
    int64_t incx;
    int64_t incy;
    uint32_t path;
    uint32_t coreNum;
    uint32_t outPerCore;
    uint32_t betaZero;
    uint32_t alphaZero;

    int64_t xStart;
    int64_t yStart;
    int64_t lenX;
    int64_t lenY;
    uint32_t panel;
    uint32_t mode;
    uint32_t outputTile;
    uint32_t reduceTile;
    uint32_t blockCount;
};
