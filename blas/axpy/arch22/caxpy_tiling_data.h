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

// Single source of truth for the CAXPY host->device tiling block. BOTH the host
// (caxpy_host.cpp) and the device kernel (caxpy_kernel.cpp) include this header
// and access fields by name. Do not re-declare this layout anywhere else and do
// not read it via hard-coded byte offsets -- that is exactly the drift hazard
// this shared definition exists to remove.

struct CaxpyTilingData {
    uint32_t totalN;
    uint32_t activeBlocks;
    float alphaReal;
    float alphaImag;
    int64_t incx;
    int64_t incy;
    // Shared-pipeline span-read tile capacity, in complex elements. Computed by
    // the host from incx so that the span read
    // ((sharedTileCapacity-1)*|incx|+1) complex elements always fits in the
    // 38 KiB lowScratch UB buffer -- see CaxpySharedTileCapacity in
    // caxpy_dispatch_policy.h for the formula. Unused (left at its default of
    // CAXPY_STRIDED_TILE_COUNT) by the disjoint and scalar kernel variants,
    // which never span-read.
    uint32_t sharedTileCapacity;
};
