/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#include "cgemm_kernel_common.h"

// ============================================================================
// Cgemm phase 0 — jointly deinterleave A/B on device
// ============================================================================
#include "cgemm_deinterleave_common.h"

namespace cgemm_deinterleave {
using namespace AscendC;

// Two input components and at most three transformed output planes share each stage.
constexpr uint32_t PREPROCESS_STAGES = 2;
constexpr uint32_t PREPROCESS_PLANES = CGEMM_COMPLEX_COMPONENTS + CGEMM_THREE_PRODUCTS;
constexpr uint32_t PREPROCESS_VECTOR_FLOATS = CGEMM_VECTOR_BYTES / sizeof(float);
constexpr uint32_t PREPROCESS_MAX_ELEMENTS =
    CGEMM_UB_WORKSPACE_BYTES / (PREPROCESS_STAGES * PREPROCESS_PLANES * sizeof(float) * PREPROCESS_VECTOR_FLOATS) *
    PREPROCESS_VECTOR_FLOATS;
constexpr uint32_t PREPROCESS_STAGE_BYTES = PREPROCESS_PLANES * PREPROCESS_MAX_ELEMENTS * sizeof(float);

template <bool ThreeProducts, bool StandardProducts = false>
__simd_vf__ inline void TransformPacked(
    __ubuf__ float* input, __ubuf__ float* output, uint32_t elements, float sign, bool isB)
{
    namespace V = AscendC::MicroAPI;
    V::MaskReg mask = V::CreateMask<float, V::MaskPattern::ALL>();
    for (uint32_t offset = 0; offset < elements; offset += PREPROCESS_VECTOR_FLOATS) {
        V::RegTensor<float> real, imag, first, second, third;
        V::LoadAlign<float, V::LoadDist::DIST_DINTLV_B32>(real, imag, input + CGEMM_COMPLEX_COMPONENTS * offset);
        V::Muls(imag, imag, sign, mask);
        if constexpr (ThreeProducts && StandardProducts) {
            V::Add(third, real, imag, mask);
            V::StoreAlign(output + offset, real, mask);
            V::StoreAlign(output + PREPROCESS_MAX_ELEMENTS + offset, imag, mask);
            V::StoreAlign(output + 2 * PREPROCESS_MAX_ELEMENTS + offset, third, mask);
        } else if constexpr (ThreeProducts) {
            if (isB) {
                V::Sub(second, imag, real, mask);
                V::Add(third, real, imag, mask);
                V::StoreAlign(output + offset, real, mask);
            } else {
                V::Add(first, real, imag, mask);
                V::StoreAlign(output + offset, first, mask);
                V::StoreAlign(output + PREPROCESS_MAX_ELEMENTS + offset, real, mask);
                V::StoreAlign(output + 2 * PREPROCESS_MAX_ELEMENTS + offset, imag, mask);
            }
            if (isB) {
                V::StoreAlign(output + PREPROCESS_MAX_ELEMENTS + offset, second, mask);
                V::StoreAlign(output + 2 * PREPROCESS_MAX_ELEMENTS + offset, third, mask);
            }
        } else {
            V::StoreAlign(output + offset, real, mask);
            V::StoreAlign(output + PREPROCESS_MAX_ELEMENTS + offset, imag, mask);
        }
    }
}

template <bool ThreeProducts, bool StandardProducts = false>
__aicore__ inline void CopyPackedChunk(
    GM_ADDR source, GM_ADDR first, GM_ADDR second, GM_ADDR third, uint64_t offset, uint32_t count, uint32_t stage,
    bool conjugate, bool isB)
{
    const uint32_t base = stage * PREPROCESS_STAGE_BYTES;
    LocalTensor<float> input(TPosition::VECCALC, base, CGEMM_COMPLEX_COMPONENTS * PREPROCESS_MAX_ELEMENTS);
    LocalTensor<float> output(
        TPosition::VECCALC, base + CGEMM_COMPLEX_COMPONENTS * PREPROCESS_MAX_ELEMENTS * sizeof(float),
        CGEMM_THREE_PRODUCTS * PREPROCESS_MAX_ELEMENTS);
    GlobalTensor<float> src, out0, out1, out2;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source));
    out0.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(first));
    out1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(second));
    if constexpr (ThreeProducts) {
        out2.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(third));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(stage);
    DataCopyExtParams load{1, static_cast<uint32_t>(count * CGEMM_COMPLEX_COMPONENTS * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    DataCopyPad(input, src[CGEMM_COMPLEX_COMPONENTS * offset], load, padding);
    SetFlag<HardEvent::MTE2_V>(stage);
    WaitFlag<HardEvent::MTE2_V>(stage);
    asc_vf_call<TransformPacked<ThreeProducts, StandardProducts>>(
        reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()),
        count, conjugate ? -1.0f : 1.0f, isB);
    SetFlag<HardEvent::V_MTE3>(stage);
    WaitFlag<HardEvent::V_MTE3>(stage);
    DataCopyExtParams store{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPad(out0[offset], output, store);
    DataCopyPad(out1[offset], output[PREPROCESS_MAX_ELEMENTS], store);
    if constexpr (ThreeProducts) {
        DataCopyPad(out2[offset], output[2 * PREPROCESS_MAX_ELEMENTS], store);
    }
    SetFlag<HardEvent::MTE3_MTE2>(stage);
}

template <bool ThreeProducts, bool StandardProducts = false>
__aicore__ inline void PreprocessPacked(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB,
    const CgemmDeinterleaveTilingData& t)
{
    const uint64_t aElements = static_cast<uint64_t>(t.aRows) * t.aCols;
    const uint64_t bElements = static_cast<uint64_t>(t.bRows) * t.bCols;
    const uint32_t chunk = Min<uint64_t>(
        PREPROCESS_MAX_ELEMENTS,
        RoundUp<uint64_t>(CeilDiv<uint64_t>(aElements + bElements, GetBlockNum()), PREPROCESS_VECTOR_FLOATS));
    const uint64_t aChunks = CeilDiv<uint64_t>(aElements, chunk);
    const uint64_t total = aChunks + CeilDiv<uint64_t>(bElements, chunk);
    for (uint32_t stage = 0; stage < PREPROCESS_STAGES; ++stage) {
        SetFlag<HardEvent::MTE3_MTE2>(stage);
    }
    uint32_t stage = 0;
    for (uint64_t task = GetBlockIdx(); task < total; task += GetBlockNum(), stage ^= 1) {
        const bool isB = task >= aChunks;
        const uint64_t offset = (isB ? task - aChunks : task) * chunk;
        const uint32_t count = Min<uint64_t>(chunk, (isB ? bElements : aElements) - offset);
        CopyPackedChunk<ThreeProducts, StandardProducts>(
            isB ? b : a, isB ? br : ar, isB ? bi : ai, isB ? auxiliaryB : auxiliaryA, offset, count, stage,
            isB ? t.conjugateB : t.conjugateA, isB);
    }
    for (uint32_t stage = 0; stage < PREPROCESS_STAGES; ++stage) {
        WaitFlag<HardEvent::MTE3_MTE2>(stage);
    }
}

extern "C" __global__ __aicore__ void gemm_cgemm_deinterleave_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB,
    const CgemmDeinterleaveTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const bool packed = tiling.lda == tiling.aRows && tiling.ldb == tiling.bRows && tiling.aOutRows == tiling.aRows &&
                        tiling.bOutRows == tiling.bRows;
    if (tiling.threeProducts && packed) {
        if (tiling.standardThreeProducts) {
            PreprocessPacked<true, true>(a, b, ar, ai, br, bi, auxiliaryA, auxiliaryB, tiling);
        } else {
            PreprocessPacked<true>(a, b, ar, ai, br, bi, auxiliaryA, auxiliaryB, tiling);
        }
    } else if (tiling.threeProducts) {
        LaunchDeinterleaveVf<false, true>(a, b, ar, ai, br, bi, auxiliaryA, auxiliaryB, tiling);
    } else if (packed) {
        PreprocessPacked<false>(a, b, ar, ai, br, bi, auxiliaryA, auxiliaryB, tiling);
    } else {
        LaunchDeinterleaveVf<false, false>(a, b, ar, ai, br, bi, auxiliaryA, auxiliaryB, tiling);
    }
}

} // namespace cgemm_deinterleave

void gemm_cgemm_deinterleave_do(
    uint32_t numBlocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi,
    GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, const CgemmDeinterleaveTilingData& tilingData)
{
    cgemm_deinterleave::gemm_cgemm_deinterleave_kernel<<<numBlocks, nullptr, stream>>>(
        a, b, ar, ai, br, bi, auxiliaryA, auxiliaryB, tilingData);
}

#endif
