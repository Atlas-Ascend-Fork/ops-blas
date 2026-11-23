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
 * \file gemv_strided_batched_tiling_data.h
 * \brief Tiling data definition for aclblasGemvStridedBatched (arch22 / dav-2201).
 */

#pragma once

#include <cstdint>

constexpr uint32_t GEMV_SB_ARCH22_TILE = 2048U;
constexpr uint32_t GEMV_SB_ARCH22_WIN_MAX = 4096U;

struct GemvStridedBatchedTilingData {
    uint32_t dtype;
    uint32_t trans;
    float alpha;
    float beta;
    uint32_t m;
    uint32_t n;
    uint32_t usedCoreNum;
    uint32_t batchCount;
    uint32_t batchPerCore;
    uint32_t batchTail;
    int32_t lda;
    int32_t incx;
    int32_t incy;
    int64_t strideA;
    int64_t stridex;
    int64_t stridey;
};
