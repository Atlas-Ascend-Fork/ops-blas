/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cher_tiling_data.h
 * \brief Tiling data structure for aclblasCher (arch22 / dav-2201).
 */

#pragma once

#include <cstdint>

inline constexpr uint32_t CHER_ARCH22_TILE_ROWS = 2048U;
inline constexpr uint32_t CHER_ARCH22_WIN_MAX_CPLX = 4096U;

struct CherTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t uplo;
    uint32_t alphaIsDevice;
    float alpha;
    int64_t incx;
    uint32_t numBlocks;
    uint32_t tileRows;
};

