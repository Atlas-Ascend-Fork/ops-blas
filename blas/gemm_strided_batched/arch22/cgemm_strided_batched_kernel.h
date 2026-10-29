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
#include "cgemm_strided_batched_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Keep entry alignment stable across independently compiled kernel families.
constexpr uint32_t kCgemmSbKernelAlignment = 256U;

void cgemm_sb_launch(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling& tiling, uint32_t aivCores,
    uint32_t aicCores, bool useCube, bool parallelC, void* stream);

void cgemm_sb_direct_batched_launch(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmSbTiling& tiling, uint32_t blocks, void* stream);

// Internal host-side launch routing; device implementation stays in each kernel family.
namespace cgemm_sb {
struct CgemmSbDispatcher {
    GM_ADDR a;
    GM_ADDR b;
    GM_ADDR c;
    GM_ADDR workspace;
    const CgemmSbTiling& tiling;
    uint32_t aivCores, aicCores;
    void* stream;
    CgemmSbCompactTiling compact;
    bool LaunchSmall();
    bool LaunchDiagonal();
    bool SquareShape();
    bool SquareLayout();
    void LaunchLargeSquare();
    void LaunchSmallSquare();
    bool LaunchSquare();
    bool LaunchGroupedAugmented();
    void PackAugmented();
    void ComputeAugmented();
    bool LaunchAugmented();
    void LaunchFast();
    void LaunchGeneral();
    void LaunchNz(uint32_t width);
    void LaunchNzCopy();
    void LaunchDirectNt(uint32_t width);
};
} // namespace cgemm_sb
