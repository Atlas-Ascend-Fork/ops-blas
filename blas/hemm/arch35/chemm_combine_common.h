/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "kernel_operator.h"
#include "chemm_tiling_data.h"

constexpr int64_t kChemmRiskReferenceK = 2048;
constexpr float kChemmPrecisionErrorBudget = 0.01f;
constexpr float kChemmFloatUnitRoundoff = 0x1p-24f;
constexpr float kChemmCancellationSafetyFactor = 2.f;

struct ChemmCombineInput {
    __gm__ const float* w;
    int64_t m;
    int64_t n;
    int64_t ldc;
    int64_t plane;
    uint32_t mp;
    uint32_t splits;
};

template <bool FOUR, bool SPLIT1 = false>
__simt_callee__ __aicore__ inline void ChemmLoadPartialSums(
    __gm__ const float* w, int64_t plane, int64_t q, uint32_t splits, float* sums)
{
    constexpr uint32_t products = FOUR ? 4 : 3;
#pragma unroll
    for (uint32_t comp = 0; comp < products; ++comp) {
        if constexpr (SPLIT1) {
            sums[comp] = w[comp * plane + q];
        } else {
            sums[comp] = w[comp * plane + q] + w[(products + comp) * plane + q];
            if (splits >= 4)
                sums[comp] += (w[(2 * products + comp) * plane + q] + w[(3 * products + comp) * plane + q]);
            if (splits == 8)
                sums[comp] += (w[(4 * products + comp) * plane + q] + w[(5 * products + comp) * plane + q]) +
                              (w[(6 * products + comp) * plane + q] + w[(7 * products + comp) * plane + q]);
        }
    }
}

struct ChemmCombinedElement {
    int64_t row;
    int64_t col;
    int64_t q;
    float re;
    float im;
    float sums[4];
};

template <bool FOUR, bool SPLIT1 = false>
__simt_callee__ __aicore__ inline ChemmCombinedElement ChemmLoadCombinedElement(
    const ChemmCombineInput& input, int64_t pos)
{
    ChemmCombinedElement element{0, 0, 0, 0, 0, {0, 0, 0, 0}};
    const bool narrow = input.ldc * input.n <= uint64_t(0xffffffffU);
    element.row = narrow ? uint32_t(pos) % uint32_t(input.ldc) : pos % input.ldc;
    element.col = narrow ? uint32_t(pos) / uint32_t(input.ldc) : pos / input.ldc;
    if (element.row >= input.m)
        return element;

    element.q = element.col * input.mp + element.row;
    float sums[FOUR ? 4 : 3];
    ChemmLoadPartialSums<FOUR, SPLIT1>(input.w, input.plane, element.q, input.splits, sums);
    for (uint32_t comp = 0; comp < (FOUR ? 4U : 3U); ++comp)
        element.sums[comp] = sums[comp];
    element.re = sums[0] - sums[1];
    if constexpr (FOUR)
        element.im = sums[2] + sums[3];
    else
        element.im = (sums[2] - sums[0]) - sums[1];
    return element;
}

__simt_callee__ __aicore__ inline bool ChemmHasCancellationRisk(const float* sums, int64_t kdim, uint32_t riskProfile)
{
    const float componentMagnitude = __builtin_fabsf(sums[0]) + __builtin_fabsf(sums[1]) + __builtin_fabsf(sums[2]);
    const bool deepKPolicy = riskProfile == static_cast<uint32_t>(ChemmRiskProfile::DeepKConservative);
    return (deepKPolicy || kdim > kChemmRiskReferenceK) &&
           componentMagnitude * kChemmFloatUnitRoundoff * kChemmCancellationSafetyFactor > kChemmPrecisionErrorBudget;
}

__simt_callee__ __aicore__ inline bool ChemmQueueFallback(
    int64_t pos, uint32_t core, uint64_t capacity, __gm__ uint64_t* queue)
{
    if (!capacity)
        return true;
    __gm__ uint64_t* local = queue + uint64_t(core) * (capacity + 64);
    const uint32_t mask = __ballot(1);
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t leader = __builtin_ctz(mask);
    const uint32_t prefix = __popc(mask & ((uint32_t(1) << lane) - 1));
    uint64_t base = 0;
    if (lane == leader)
        base = atomicAdd(local, uint64_t(__popc(mask)));
    base = __shfl(base, leader);
    local[64 + base + prefix] = uint64_t(pos);
    return true;
}

__simt_callee__ __aicore__ inline void ChemmRecoverNonFiniteElement(
    __gm__ const float* a, __gm__ const float* b, bool left, bool upper, int64_t m, int64_t n, int64_t lda, int64_t ldb,
    int64_t row, int64_t col, float& re, float& im)
{
    re = 0;
    im = 0;
    const int64_t k = left ? m : n;
    for (int64_t z = 0; z < k; ++z) {
        int64_t x = left ? row : z;
        int64_t y = left ? z : col;
        bool direct = upper ? x <= y : x >= y;
        int64_t off = direct ? y * lda + x : x * lda + y;
        float xr = a[2 * off];
        float xi = x == y ? 0 : (direct ? a[2 * off + 1] : -a[2 * off + 1]);
        int64_t bo = left ? col * ldb + z : z * ldb + row;
        float yr = b[2 * bo];
        float yi = b[2 * bo + 1];
        re += xr * yr - xi * yi;
        im += xr * yi + xi * yr;
    }
}

__simt_callee__ __aicore__ inline void ChemmWriteCombinedElement(
    __gm__ float* c, int64_t pos, float re, float im, float ar, float ai, float br, float bi)
{
    float cr = 0;
    float ci = 0;
    if (br != 0 || bi != 0) {
        cr = c[2 * pos];
        ci = c[2 * pos + 1];
    }
    c[2 * pos] = (ai == 0 ? ar * re : ar * re - ai * im) + (bi == 0 ? br * cr : br * cr - bi * ci);
    c[2 * pos + 1] = (ai == 0 ? ar * im : ar * im + ai * re) + (bi == 0 ? br * ci : br * ci + bi * cr);
}
