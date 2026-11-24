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

#include <cstdint>

#if defined(__CCE_AICORE__)
#include "simt_api/device_functions.h"
#endif

struct CdgmmComplexValue {
    float real;
    float imag;
};

template <bool Streaming>
__simt_callee__ __aicore__ inline CdgmmComplexValue CdgmmLoadPair(const __gm__ float* pointer)
{
#if defined(__CCE_AICORE__)
    if ((reinterpret_cast<uintptr_t>(pointer) & 7U) == 0U) {
        auto* packed = reinterpret_cast<__gm__ float2*>(const_cast<__gm__ float*>(pointer));
        const float2 value = Streaming ? asc_ldcg(packed) : asc_ldca(packed);
        return {value.x, value.y};
    }
#endif
    return {pointer[0], pointer[1]};
}

__simt_callee__ __aicore__ inline void CdgmmStorePair(__gm__ float* pointer, float real, float imag)
{
#if defined(__CCE_AICORE__)
    if ((reinterpret_cast<uintptr_t>(pointer) & 7U) == 0U) {
        float2 value;
        value.x = real;
        value.y = imag;
        asc_stcg(reinterpret_cast<__gm__ float2*>(pointer), value);
        return;
    }
#endif
    pointer[0] = real;
    pointer[1] = imag;
}

// Rare finite-input overflow path. Scale only the affected output component;
// scaling a finite companion component could erase small but relevant terms.
__simt_callee__ __aicore__ inline float CdgmmScaledSum(float a, float b, float c, float d, bool subtract)
{
    constexpr float SAFE_PRODUCT = 0x1.fffffep+125F;
    uint32_t scale = 0U;
    volatile float first = a * b;
    volatile float second = c * d;
    while (first > SAFE_PRODUCT || first < -SAFE_PRODUCT || second > SAFE_PRODUCT || second < -SAFE_PRODUCT) {
        a *= 0.5F;
        c *= 0.5F;
        ++scale;
        first = a * b;
        second = c * d;
    }
    float result = subtract ? first - second : first + second;
    for (uint32_t index = 0U; index < scale; ++index) {
        result *= 2.0F;
    }
    return result;
}

// Shared with the local CPU model; device qualifiers are supplied by the caller.
// No restrict qualifiers: A and C may alias when lda == ldc.
template <typename Index>
struct CdgmmLoadedElement {
    CdgmmComplexValue a;
    CdgmmComplexValue x;
    Index cOffset;
};

template <typename Index, bool Left, bool Compact>
__simt_callee__ __aicore__ inline CdgmmLoadedElement<Index> CdgmmLoadElementAtCoordinates(
    Index linear, Index row, Index column, uint32_t lda, uint32_t ldc, uint32_t xLength, int32_t incx,
    const __gm__ float* a, const __gm__ float* x)
{
    const Index logicalX = Left ? row : column;
    const Index absIncx = incx < 0 ? static_cast<Index>(-static_cast<int64_t>(incx)) : static_cast<Index>(incx);
    const Index xElement = incx < 0 ? (static_cast<Index>(xLength) - 1U - logicalX) * absIncx : logicalX * absIncx;
    const Index aElement = Compact ? linear : row + column * static_cast<Index>(lda);
    const Index cElement = Compact ? linear : row + column * static_cast<Index>(ldc);
    const Index aOffset = aElement * 2U;
    const Index cOffset = cElement * 2U;
    const Index xOffset = xElement * 2U;

    // Read both A components before either store, including the in-place case.
    const CdgmmComplexValue aValue = CdgmmLoadPair<true>(a + aOffset);
    const CdgmmComplexValue xValue = CdgmmLoadPair<false>(x + xOffset);
    return {aValue, xValue, cOffset};
}

__simt_callee__ __aicore__ inline CdgmmComplexValue CdgmmMultiply(
    const CdgmmComplexValue& a, const CdgmmComplexValue& x)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float ar = a.real;
    const float ai = a.imag;
    const float xr = x.real;
    const float xi = x.imag;
    // Disable contraction explicitly so the four rounded float32 products
    // remain in registers instead of forcing volatile thread-local stores.
    const float rr = ar * xr;
    const float ii = ai * xi;
    const float ri = ar * xi;
    const float ir = ai * xr;
    float real = rr - ii;
    float imag = ri + ir;
    constexpr float MAX_FLOAT = 0x1.fffffep+127F;
    const bool finiteInputs = ar <= MAX_FLOAT && ar >= -MAX_FLOAT && ai <= MAX_FLOAT && ai >= -MAX_FLOAT &&
                              xr <= MAX_FLOAT && xr >= -MAX_FLOAT && xi <= MAX_FLOAT && xi >= -MAX_FLOAT;
    if (finiteInputs) {
        if (!(real <= MAX_FLOAT && real >= -MAX_FLOAT)) {
            real = CdgmmScaledSum(ar, xr, ai, xi, true);
        }
        if (!(imag <= MAX_FLOAT && imag >= -MAX_FLOAT)) {
            imag = CdgmmScaledSum(ar, xi, ai, xr, false);
        }
    }
    return {real, imag};
}

template <typename Index>
__simt_callee__ __aicore__ inline void CdgmmComputeStore(const CdgmmLoadedElement<Index>& loaded, __gm__ float* c)
{
    const CdgmmComplexValue result = CdgmmMultiply(loaded.a, loaded.x);
    CdgmmStorePair(c + loaded.cOffset, result.real, result.imag);
}

template <typename Index, bool Left, bool Compact>
__simt_callee__ __aicore__ inline void CdgmmComputeElementAtCoordinates(
    Index linear, Index row, Index column, uint32_t lda, uint32_t ldc, uint32_t xLength, int32_t incx,
    const __gm__ float* a, const __gm__ float* x, __gm__ float* c)
{
    const auto loaded =
        CdgmmLoadElementAtCoordinates<Index, Left, Compact>(linear, row, column, lda, ldc, xLength, incx, a, x);
    CdgmmComputeStore(loaded, c);
}

template <typename Index, bool Left, bool Compact>
__simt_callee__ __aicore__ inline void CdgmmComputeElement(
    Index linear, uint32_t m, uint32_t lda, uint32_t ldc, uint32_t xLength, int32_t incx, const __gm__ float* a,
    const __gm__ float* x, __gm__ float* c)
{
    const Index column = linear / static_cast<Index>(m);
    const Index row = linear - column * static_cast<Index>(m);
    CdgmmComputeElementAtCoordinates<Index, Left, Compact>(linear, row, column, lda, ldc, xLength, incx, a, x, c);
}
