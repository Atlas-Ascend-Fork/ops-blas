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
 * \file sdgmm_tiling_data.h
 * \brief Tiling data structure for aclblasSdgmm (arch22 / dav-2201).
 *        Shared by host side (const ref) and kernel side (by value).
 */

#pragma once

#include <cstdint>

inline constexpr uint32_t SDGMM_ARCH22_MODE_LEFT = 0U;
inline constexpr uint32_t SDGMM_ARCH22_MODE_RIGHT = 1U;
inline constexpr uint32_t SDGMM_ARCH22_TILE_M = 8192U;
inline constexpr uint32_t SDGMM_ARCH22_WIN_MAX_ELEMS = 8192U;

struct SdgmmTilingData {
    uint32_t mode;
    uint32_t m;
    uint32_t n;
    int32_t incx;
    uint32_t lda;
    uint32_t ldc;
    uint32_t colBlocks;
    uint32_t perCoreN;
    uint32_t remainder;
    uint32_t mBlocks;
    uint32_t perCoreMTile;
    uint32_t mTileRemainder;
    uint32_t tileM;
};

