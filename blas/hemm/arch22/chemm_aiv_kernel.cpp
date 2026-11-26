/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chemm (arch22) AIV kernel translation unit: tiny/unit/grid entries and launch.

#include "chemm_kernel_common.h"
#include "chemm_kernel_aic.h"
#include "chemm_kernel_mix_aiv.h"
#include "chemm_kernel_aiv.h"
#include "chemm_kernel_ordered.h"
#include "acl/acl.h"

template <bool kRight, bool kLower>
__aicore__ inline void RunAivTinyMode(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* raw,
    __ubuf__ float* rowPlanes, __ubuf__ float* auxPlanes, __ubuf__ float* accumulators, __ubuf__ float* broadcasts,
    __ubuf__ float* negativeVectorImag, __ubuf__ uint8_t* masks)
{
    const bool full = tiling.m == static_cast<int32_t>(kMicro) && tiling.n == static_cast<int32_t>(kMicro) &&
                      tiling.dimA == static_cast<int32_t>(kMicro) && (static_cast<uint32_t>(tiling.lda) & 3U) == 0U &&
                      (static_cast<uint32_t>(tiling.ldb) & 3U) == 0U && (static_cast<uint32_t>(tiling.ldc) & 3U) == 0U;
    if (full) {
        RunAivTiny<kRight, kLower, true, kMicro, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (tiling.dimA == 1) {
        RunAivTiny<kRight, kLower, false, 1U, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (tiling.dimA == 2) {
        RunAivTiny<kRight, kLower, false, 2U, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (tiling.dimA <= 4) {
        RunAivTiny<kRight, kLower, false, 4U, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (tiling.dimA <= 8) {
        RunAivTiny<kRight, kLower, false, 8U, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else {
        RunAivTiny<kRight, kLower, false, kMicro, kAivRowsPerVectorCore, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    }
}

__global__ __aicore__ __vector__ void chemm_aiv_tiny_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, ChemmAivTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    constexpr uint32_t kTinyBroadcastElements = kAivRowsPerVectorCore * kFp32DataBlockElements;
    __ubuf__ float raw[kComplexTileFloats] __attribute__((aligned(32)));
    __ubuf__ float rowPlanes[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float auxPlanes[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float accumulators[kAivAccumulatorParts * kAivAccumulatorBanks * kMicroElements]
        __attribute__((aligned(32)));
    __ubuf__ float broadcasts[2U * kMicro * kTinyBroadcastElements] __attribute__((aligned(32)));
    __ubuf__ float negativeVectorImag[kMicroElements] __attribute__((aligned(32)));
    __ubuf__ uint8_t masks[96U] __attribute__((aligned(32)));
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    const bool right = (tiling.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        RunAivTinyMode<true, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (right) {
        RunAivTinyMode<true, false>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else if (lower) {
        RunAivTinyMode<false, true>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    } else {
        RunAivTinyMode<false, false>(
            a, b, c, tiling, raw, rowPlanes, auxPlanes, accumulators, broadcasts, negativeVectorImag, masks);
    }
}

__global__ __aicore__ __vector__ void chemm_aiv_unit_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    RunAivUnit(
        reinterpret_cast<__gm__ float*>(gmA), reinterpret_cast<__gm__ float*>(gmB),
        reinterpret_cast<__gm__ float*>(gmC));
}

template <bool kRight, bool kLower>
__aicore__ inline void RunAivGridMode(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* rawSlots,
    __ubuf__ float* leftPlanes, __ubuf__ float* rightPlanes, __ubuf__ float* leftTranspose,
    __ubuf__ float* accumulators, __ubuf__ float* broadcasts, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* outputTile, __ubuf__ float* yCache, __ubuf__ uint8_t* masks)
{
    if (tiling.rowTile == kAivRowsPerVectorCore) {
        RunAivGrid<kRight, kLower, true, kAivRowsPerVectorCore>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    } else {
        RunAivGrid<kRight, kLower, true, kMicro>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    }
}

__global__ __aicore__ __vector__ void chemm_aiv_grid_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, ChemmAivTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    constexpr uint32_t kGridBroadcastElements = kMicro * kFp32DataBlockElements;
    __ubuf__ float rawSlots[4U * kComplexTileFloats] __attribute__((aligned(32)));
    __ubuf__ float leftPlanes[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float rightPlanes[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float leftTranspose[2U * kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float accumulators[kAivAccumulatorParts * kAivAccumulatorBanks * kMicroElements]
        __attribute__((aligned(32)));
    __ubuf__ float broadcasts[2U * kMicro * kGridBroadcastElements] __attribute__((aligned(32)));
    __ubuf__ float negativeVectorImag[kMicroElements] __attribute__((aligned(32)));
    __ubuf__ float outputTile[kComplexTileFloats] __attribute__((aligned(32)));
    __ubuf__ float yCache[2U * kAivYCachePlaneElements] __attribute__((aligned(32)));
    __ubuf__ uint8_t masks[96U] __attribute__((aligned(32)));
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    const bool right = (tiling.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (tiling.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        RunAivGridMode<true, true>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    } else if (right) {
        RunAivGridMode<true, false>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    } else if (lower) {
        RunAivGridMode<false, true>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    } else {
        RunAivGridMode<false, false>(
            a, b, c, tiling, rawSlots, leftPlanes, rightPlanes, leftTranspose, accumulators, broadcasts,
            negativeVectorImag, outputTile, yCache, masks);
    }
}

aclError chemm_aiv_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, const ChemmAivTilingData& base, bool right, bool lower, void* stream)
{
    ChemmAivTilingData tiling = base;
    tiling.kernelFlags = (right ? CHEMM_FLAG_RIGHT : 0U) | (lower ? CHEMM_FLAG_LOWER : 0U);
    // issue #1255: 发射前清掉线程级陈旧错误，发射后返回运行时状态。
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    if (tiling.m == 1 && tiling.n == 1 && tiling.dimA == 1) {
        chemm_aiv_unit_kernel<<<1U, nullptr, stream>>>(a, b, c);
        return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    }
    const bool small = tiling.m <= static_cast<int32_t>(kMicro) && tiling.n <= static_cast<int32_t>(kMicro) &&
                       tiling.dimA <= static_cast<int32_t>(kMicro);
    if (small) {
        chemm_aiv_tiny_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, tiling);
    } else {
        chemm_aiv_grid_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, tiling);
    }
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
