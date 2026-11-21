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

#include "cgemm_tiling_data.h"

namespace cgemm_deinterleave {

template <bool Packed, bool ThreeProducts>
__simt_callee__ __aicore__ inline void DeinterleaveElement(
    __gm__ const float* input, __gm__ float* realOut, __gm__ float* imagOut, __gm__ float* auxiliary, bool isB,
    uint64_t index, uint32_t rows, uint32_t ld, uint32_t outputRows, bool conjugate, bool standardThreeProducts)
{
    uint64_t source = CGEMM_COMPLEX_COMPONENTS * index;
    if constexpr (!Packed) {
        const uint64_t col = index / rows;
        const uint64_t row = index - col * rows;
        source = CGEMM_COMPLEX_COMPONENTS * (col * ld + row);
    }
    uint64_t output = index;
    if (outputRows != rows) {
        const uint64_t col = index / rows;
        output = col * outputRows + index - col * rows;
    }
    const float real = input[source];
    const float imag = conjugate ? -input[source + 1] : input[source + 1];
    // Fused-path identity: P0=(Ar+Ai)*Br, P1=Ar*(Bi-Br), P2=Ai*(Br+Bi).
    // Generic products instead use the standard Ar, Ai, Ar+Ai input planes.
    if constexpr (ThreeProducts) {
        if (standardThreeProducts) {
            realOut[output] = real;
            imagOut[output] = imag;
            auxiliary[output] = real + imag;
            return;
        }
        realOut[output] = isB ? real : real + imag;
        imagOut[output] = isB ? imag - real : real;
        auxiliary[output] = isB ? real + imag : imag;
    } else {
        realOut[output] = real;
        imagOut[output] = imag;
    }
}

template <bool Packed, bool ThreeProducts>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_SIMT_THREADS) inline void CgemmDeinterleaveVf(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* ar, __gm__ float* ai, __gm__ float* br,
    __gm__ float* bi, __gm__ float* auxiliaryA, __gm__ float* auxiliaryB, uint32_t aRows, uint32_t aCols, uint32_t lda,
    uint32_t aOutRows, uint32_t bRows, uint32_t bCols, uint32_t ldb, uint32_t bOutRows, bool conjugateA,
    bool conjugateB, bool standardThreeProducts, uint32_t block, uint32_t blocks)
{
    const uint64_t aElements = static_cast<uint64_t>(aRows) * aCols;
    const uint64_t total = aElements + static_cast<uint64_t>(bRows) * bCols;
    const uint64_t step = static_cast<uint64_t>(blocks) * blockDim.x;
    for (uint64_t index = static_cast<uint64_t>(block) * blockDim.x + threadIdx.x; index < total; index += step) {
        if (index < aElements) {
            DeinterleaveElement<Packed, ThreeProducts>(
                a, ar, ai, auxiliaryA, false, index, aRows, lda, aOutRows, conjugateA, standardThreeProducts);
        } else {
            DeinterleaveElement<Packed, ThreeProducts>(
                b, br, bi, auxiliaryB, true, index - aElements, bRows, ldb, bOutRows, conjugateB,
                standardThreeProducts);
        }
    }
}

template <bool Packed, bool ThreeProducts>
__aicore__ inline void LaunchDeinterleaveVf(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB,
    const CgemmDeinterleaveTilingData& tiling)
{
    asc_vf_call<CgemmDeinterleaveVf<Packed, ThreeProducts>>(
        dim3{CGEMM_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ const float*>(a),
        reinterpret_cast<__gm__ const float*>(b), reinterpret_cast<__gm__ float*>(ar),
        reinterpret_cast<__gm__ float*>(ai), reinterpret_cast<__gm__ float*>(br), reinterpret_cast<__gm__ float*>(bi),
        reinterpret_cast<__gm__ float*>(auxiliaryA), reinterpret_cast<__gm__ float*>(auxiliaryB), tiling.aRows,
        tiling.aCols, tiling.lda, tiling.aOutRows, tiling.bRows, tiling.bCols, tiling.ldb, tiling.bOutRows,
        tiling.conjugateA != 0, tiling.conjugateB != 0, tiling.standardThreeProducts != 0, AscendC::GetBlockIdx(),
        AscendC::GetBlockNum());
}

} // namespace cgemm_deinterleave
