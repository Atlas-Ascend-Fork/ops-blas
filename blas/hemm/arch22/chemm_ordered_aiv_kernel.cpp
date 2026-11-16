/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chemm (arch22) ordered AIV kernel translation unit and its launch symbol.

#include "chemm_kernel_common.h"
#include "chemm_kernel_aic.h"
#include "chemm_kernel_mix_aiv.h"
#include "chemm_kernel_aiv.h"
#include "chemm_kernel_ordered.h"

__global__ __aicore__ __vector__ void chemm_ordered_aiv_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, ChemmAivTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    constexpr uint32_t kOrderedScratchElements =
        3U * kComplexTileFloats + 25U * kMicroElements + 2U * kMicro * kMicro * kFp32DataBlockElements + 24U;
    __ubuf__ float scratch[kOrderedScratchElements] __attribute__((aligned(32)));
    const bool right = (tiling.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        RunOrderedAiv<true, true>(a, b, c, tiling, scratch);
    } else if (right) {
        RunOrderedAiv<true, false>(a, b, c, tiling, scratch);
    } else if (lower) {
        RunOrderedAiv<false, true>(a, b, c, tiling, scratch);
    } else {
        RunOrderedAiv<false, false>(a, b, c, tiling, scratch);
    }
}

void chemm_ordered_aiv_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, const ChemmAivTilingData& base, bool right, bool lower, void* stream)
{
    ChemmAivTilingData tiling = base;
    tiling.kernelFlags = (right ? CHEMM_FLAG_RIGHT : 0U) | (lower ? CHEMM_FLAG_LOWER : 0U);
    chemm_ordered_aiv_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, tiling);
}
