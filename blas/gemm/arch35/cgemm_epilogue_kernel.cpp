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
#include "gemm_strided_batched/arch35/cgemm_strided_batched_reference.h"

// Cgemm phase 2 — vector alpha/beta epilogue and interleaved store
// ============================================================================
namespace cgemm_epilogue {
using namespace AscendC;

// Plane sizes follow the UB budget; rows are rounded to a vector register.
constexpr uint32_t EPILOGUE_VECTOR_FLOATS = CGEMM_VECTOR_BYTES / sizeof(float);
constexpr uint32_t EPILOGUE_INPUT_PLANES = CGEMM_THREE_PRODUCTS;
constexpr uint32_t EPILOGUE_SUM_PLANES = CGEMM_COMPLEX_COMPONENTS;
constexpr uint32_t EPILOGUE_OUTPUT_PLANES = CGEMM_COMPLEX_COMPONENTS;
constexpr uint32_t EPILOGUE_PLANES = EPILOGUE_INPUT_PLANES + EPILOGUE_SUM_PLANES + EPILOGUE_OUTPUT_PLANES;
constexpr uint32_t EPILOGUE_PLANE =
    CGEMM_UB_WORKSPACE_BYTES / (EPILOGUE_PLANES * CGEMM_VECTOR_BYTES) * EPILOGUE_VECTOR_FLOATS;
constexpr uint32_t EPILOGUE_SUM_OFFSET = EPILOGUE_INPUT_PLANES * EPILOGUE_PLANE;
constexpr uint32_t EPILOGUE_OUTPUT_OFFSET = (EPILOGUE_INPUT_PLANES + EPILOGUE_SUM_PLANES) * EPILOGUE_PLANE;

__simd_vf__ inline void MergePlanarOutput(__ubuf__ float* p, uint32_t count, bool first, bool last)
{
    namespace V = AscendC::MicroAPI;
    V::MaskReg mask = V::CreateMask<float, V::MaskPattern::ALL>();
    for (uint32_t offset = 0; offset < count; offset += EPILOGUE_VECTOR_FLOATS) {
        V::RegTensor<float> a, b, c, real, imag, previous, lo, hi;
        V::LoadAlign(a, p + offset);
        V::LoadAlign(b, p + EPILOGUE_PLANE + offset);
        V::LoadAlign(c, p + 2 * EPILOGUE_PLANE + offset);
        V::Sub(real, a, c, mask);
        V::Add(imag, a, b, mask);
        if (!first) {
            V::LoadAlign(previous, p + EPILOGUE_SUM_OFFSET + offset);
            V::Add(real, previous, real, mask);
            V::LoadAlign(previous, p + EPILOGUE_SUM_OFFSET + EPILOGUE_PLANE + offset);
            V::Add(imag, previous, imag, mask);
        }
        if (last) {
            V::Interleave(lo, hi, real, imag);
            V::StoreAlign(p + EPILOGUE_OUTPUT_OFFSET + 2 * offset, lo, mask);
            V::StoreAlign(p + EPILOGUE_OUTPUT_OFFSET + 2 * offset + EPILOGUE_VECTOR_FLOATS, hi, mask);
        } else {
            V::StoreAlign(p + EPILOGUE_SUM_OFFSET + offset, real, mask);
            V::StoreAlign(p + EPILOGUE_SUM_OFFSET + EPILOGUE_PLANE + offset, imag, mask);
        }
    }
}

__aicore__ inline void RunPlanarOutputTile(
    const GlobalTensor<float>* products, GlobalTensor<float>& output, const LocalTensor<float>& buffer,
    const CgemmEpilogueTilingData& tiling, uint32_t row, uint32_t col, uint32_t rows, uint32_t cols, uint32_t width)
{
    constexpr uint32_t BLOCK_FLOATS = CGEMM_DATA_BLOCK_BYTES / sizeof(float);
    DataCopyExtParams load{};
    load.blockCount = cols;
    load.blockLen = rows * sizeof(float);
    load.srcStride = (tiling.tempLdc - rows) * sizeof(float);
    load.dstStride = (width - RoundUp<uint32_t>(rows, BLOCK_FLOATS)) / BLOCK_FLOATS;
    DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    const uint64_t offset = static_cast<uint64_t>(col) * tiling.tempLdc + row;
    const uint32_t parts = tiling.kPartitions > 0 ? tiling.kPartitions : 1;
    for (uint32_t part = 0; part < parts; ++part) {
        for (uint32_t product = 0; product < EPILOGUE_INPUT_PLANES; ++product) {
            DataCopyPad(
                buffer[product * EPILOGUE_PLANE], products[product][offset + part * tiling.partStride], load, padding);
        }
        SetFlag<HardEvent::MTE2_V>(GEMM_ZERO_FLAG);
        WaitFlag<HardEvent::MTE2_V>(GEMM_ZERO_FLAG);
        MergePlanarOutput(
            reinterpret_cast<__ubuf__ float*>(buffer.GetPhyAddr()), width * cols, part == 0, part + 1 == parts);
        SetFlag<HardEvent::V_MTE2>(GEMM_ZERO_FLAG);
        WaitFlag<HardEvent::V_MTE2>(GEMM_ZERO_FLAG);
    }
    SetFlag<HardEvent::V_MTE3>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::V_MTE3>(GEMM_ZERO_FLAG);
    DataCopyExtParams store{};
    store.blockCount = cols;
    store.blockLen = CGEMM_COMPLEX_COMPONENTS * rows * sizeof(float);
    store.srcStride =
        (CGEMM_COMPLEX_COMPONENTS * width - RoundUp<uint32_t>(CGEMM_COMPLEX_COMPONENTS * rows, BLOCK_FLOATS)) /
        BLOCK_FLOATS;
    store.dstStride = CGEMM_COMPLEX_COMPONENTS * (tiling.ldc - rows) * sizeof(float);
    DataCopyPad(
        output[CGEMM_COMPLEX_COMPONENTS * (static_cast<uint64_t>(col) * tiling.ldc + row)],
        buffer[EPILOGUE_OUTPUT_OFFSET], store);
    SetFlag<HardEvent::MTE3_V>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::MTE3_V>(GEMM_ZERO_FLAG);
}

__aicore__ inline void LaunchPlanarOutput(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR c, const CgemmEpilogueTilingData& tiling)
{
    GlobalTensor<float> products[EPILOGUE_INPUT_PLANES];
    GlobalTensor<float> output;
    products[0].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t1));
    products[1].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t2));
    products[2].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t3));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    LocalTensor<float> buffer(TPosition::VECCALC, 0, EPILOGUE_PLANES * EPILOGUE_PLANE);
    const uint32_t width = Min<uint32_t>(RoundUp<uint32_t>(tiling.m, EPILOGUE_VECTOR_FLOATS), EPILOGUE_PLANE);
    const uint32_t colBatch = Min<uint32_t>(EPILOGUE_PLANE / width, CeilDiv<uint32_t>(tiling.n, GetBlockNum()));
    const uint32_t rowTiles = CeilDiv<uint32_t>(tiling.m, width);
    const uint32_t colTiles = CeilDiv<uint32_t>(tiling.n, colBatch);
    for (uint64_t task = GetBlockIdx(); task < static_cast<uint64_t>(rowTiles) * colTiles; task += GetBlockNum()) {
        const uint32_t row = (task % rowTiles) * width;
        const uint32_t col = (task / rowTiles) * colBatch;
        RunPlanarOutputTile(
            products, output, buffer, tiling, row, col, Min<uint32_t>(width, tiling.m - row),
            Min<uint32_t>(colBatch, tiling.n - col), width);
    }
}

// A coherent long reduction can drift from the ordered BLAS result even when
// the three matrix products are individually accurate. Reuse the family's
// ordered arithmetic only for the affected output elements.
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void CorrectOrderedOutput(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, uint64_t index, uint32_t m, uint32_t ldc, uint32_t depth,
    uint32_t lda, uint32_t ldb, uint32_t transA, uint32_t transB, float alphaReal, float alphaImag, float betaReal,
    float betaImag, float& real, float& imag)
{
#pragma clang fp contract(off)
    const uint32_t row = index % m, col = index / m;
    const uint64_t offset = uint64_t(col) * ldc + row;
    const bool normalA = transA == ACLBLAS_OP_N;
    real = 0;
    imag = 0;
    if (normalA && (betaReal != 0 || betaImag != 0)) {
        CgemmSbScale(betaReal, betaImag, c[2 * offset], c[2 * offset + 1], real, imag);
    }
    for (uint32_t k = 0; k < depth; ++k) {
        const uint64_t ai = 2 * (normalA ? uint64_t(k) * lda + row : uint64_t(row) * lda + k);
        const uint64_t bi = 2 * (transB == ACLBLAS_OP_N ? uint64_t(col) * ldb + k : uint64_t(k) * ldb + col);
        const float ar = a[ai], av = transA == ACLBLAS_OP_C ? -a[ai + 1] : a[ai + 1];
        const float br = b[bi], bv = transB == ACLBLAS_OP_C ? -b[bi + 1] : b[bi + 1];
        if (normalA) {
            const float scaledReal = alphaReal * br - alphaImag * bv;
            const float scaledImag = alphaReal * bv + alphaImag * br;
            CgemmSbAccumulateComplex(scaledReal, scaledImag, ar, av, real, imag);
        } else {
            CgemmSbAccumulateComplex(ar, av, br, bv, real, imag);
        }
    }
    if (!normalA) {
        float scaledReal, scaledImag;
        CgemmSbScale(alphaReal, alphaImag, real, imag, scaledReal, scaledImag);
        real = scaledReal;
        imag = scaledImag;
        if (betaReal != 0 || betaImag != 0) {
            CgemmSbScale(betaReal, betaImag, c[2 * offset], c[2 * offset + 1], scaledReal, scaledImag);
            real += scaledReal;
            imag += scaledImag;
        }
    }
}

constexpr uint32_t CORRECTION_THREADS = 256;
constexpr uint32_t DISPATCH_FIELD_BITS = 8;
constexpr uint32_t DISPATCH_FIELD_MASK = (1U << DISPATCH_FIELD_BITS) - 1;
constexpr uint32_t CORRECTION_PART_SHIFT = 2 * DISPATCH_FIELD_BITS;

// A separate 3m epilogue keeps the SIMT argument block compact. Native 4m and
// pure scaling retain their existing kernel path.
__simt_vf__ __aicore__ LAUNCH_BOUND(CORRECTION_THREADS) inline void CorrectableEpilogue(
    __gm__ const float* p0, __gm__ const float* p1, __gm__ const float* p2, __gm__ float* a, __gm__ float* b,
    __gm__ float* c, uint32_t m, uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc, float ar, float ai,
    float br, float bi, uint32_t operations, uint32_t dispatch)
{
    const uint32_t block = dispatch & DISPATCH_FIELD_MASK;
    const uint32_t blocks = (dispatch >> DISPATCH_FIELD_BITS) & DISPATCH_FIELD_MASK;
    const uint32_t parts = dispatch >> CORRECTION_PART_SHIFT;
    const uint32_t transA = operations & DISPATCH_FIELD_MASK, transB = operations >> DISPATCH_FIELD_BITS;
    const uint32_t tempLdc = (m + CGEMM_CUBE_FRACTAL - 1) / CGEMM_CUBE_FRACTAL * CGEMM_CUBE_FRACTAL;
    const uint64_t partStride =
        uint64_t(tempLdc) * ((n + CGEMM_CUBE_FRACTAL - 1) / CGEMM_CUBE_FRACTAL * CGEMM_CUBE_FRACTAL);
    for (uint64_t i = uint64_t(block) * blockDim.x + threadIdx.x; i < uint64_t(m) * n;
         i += uint64_t(blocks) * blockDim.x) {
        const uint32_t row = i % m, col = i / m;
        const uint64_t source = uint64_t(col) * tempLdc + row, dest = 2 * (uint64_t(col) * ldc + row);
        float pr = 0, pi = 0;
        for (uint32_t part = 0; part < parts; ++part) {
            const uint64_t offset = source + part * partStride;
            pr += p0[offset] - p2[offset];
            pi += p0[offset] + p1[offset];
        }
        float real = ar * pr - ai * pi, imag = ar * pi + ai * pr;
        constexpr float CANCELLATION_RATIO = 0.125f;
        const bool correct = !isfinite(pr) || !isfinite(pi) || fabsf(pr) > ORDERED_MOMENT_THRESHOLD ||
                             fabsf(pi) > ORDERED_MOMENT_THRESHOLD ||
                             fabsf(real) < CANCELLATION_RATIO * (fabsf(ar * pr) + fabsf(ai * pi)) ||
                             fabsf(imag) < CANCELLATION_RATIO * (fabsf(ar * pi) + fabsf(ai * pr));
        if (correct) {
            CorrectOrderedOutput(a, b, c, i, m, ldc, k, lda, ldb, transA, transB, ar, ai, br, bi, real, imag);
        } else if (br != 0 || bi != 0) {
            const float oldReal = c[dest], oldImag = c[dest + 1];
            real += br * oldReal - bi * oldImag;
            imag += br * oldImag + bi * oldReal;
        }
        c[dest] = real;
        c[dest + 1] = imag;
    }
}

__aicore__ inline void LaunchCorrectableEpilogue(
    GM_ADDR p0, GM_ADDR p1, GM_ADDR p2, GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmEpilogueTilingData& t)
{
    const uint32_t operations = uint32_t(t.referenceTransA) | (uint32_t(t.referenceTransB) << DISPATCH_FIELD_BITS);
    const uint32_t dispatch =
        GetBlockIdx() | (GetBlockNum() << DISPATCH_FIELD_BITS) | (t.kPartitions << CORRECTION_PART_SHIFT);
    asc_vf_call<CorrectableEpilogue>(
        dim3{CORRECTION_THREADS, 1, 1}, reinterpret_cast<__gm__ const float*>(p0),
        reinterpret_cast<__gm__ const float*>(p1), reinterpret_cast<__gm__ const float*>(p2),
        reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b), reinterpret_cast<__gm__ float*>(c), t.m,
        t.n, t.referenceK, t.referenceLda, t.referenceLdb, t.ldc, t.alphaReal, t.alphaImag, t.betaReal, t.betaImag,
        operations, dispatch);
}

template <bool AlphaZero, bool BetaZero, bool Packed>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_SIMT_THREADS) inline void CgemmEpilogueVf(
    __gm__ const float* t1, __gm__ const float* t2, __gm__ const float* t3, __gm__ const float* t4, __gm__ float* c,
    uint32_t m, uint32_t n, uint32_t ldc, uint32_t tempLdc, float alphaReal, float alphaImag, float betaReal,
    float betaImag, bool threeProducts, uint32_t kPartitions, uint64_t partStride, uint32_t block, uint32_t blocks)
{
    const uint64_t total = static_cast<uint64_t>(m) * n;
    const uint64_t thread = static_cast<uint64_t>(block) * blockDim.x + threadIdx.x;
    const uint64_t step = static_cast<uint64_t>(blocks) * blockDim.x;
    for (uint64_t index = thread; index < total; index += step) {
        uint64_t cOffset = 2 * index;
        uint64_t tempOffset = index;
        if constexpr (!Packed) {
            const uint64_t col = index / m;
            const uint64_t row = index - col * m;
            cOffset = 2 * (col * ldc + row);
            tempOffset = col * tempLdc + row;
        }
        float resultReal = 0.0f;
        float resultImag = 0.0f;
        if constexpr (!AlphaZero) {
            float productReal = 0.0f;
            float productImag = 0.0f;
            const uint32_t parts = kPartitions > 0 ? kPartitions : 1;
            for (uint32_t part = 0; part < parts; ++part) {
                const uint64_t offset = tempOffset + part * partStride;
                productReal += threeProducts ? t1[offset] - t3[offset] : t1[offset] - t2[offset];
                productImag += threeProducts ? t1[offset] + t2[offset] : t3[offset] + t4[offset];
            }
            resultReal = alphaReal * productReal - alphaImag * productImag;
            resultImag = alphaReal * productImag + alphaImag * productReal;
        }
        if constexpr (!BetaZero) {
            const float cReal = c[cOffset];
            const float cImag = c[cOffset + 1];
            resultReal += betaReal * cReal - betaImag * cImag;
            resultImag += betaReal * cImag + betaImag * cReal;
        }
        c[cOffset] = resultReal;
        c[cOffset + 1] = resultImag;
    }
}

template <bool AlphaZero, bool BetaZero, bool Packed>
__aicore__ inline void LaunchCgemmEpilogueVf(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, const CgemmEpilogueTilingData& tiling)
{
    asc_vf_call<CgemmEpilogueVf<AlphaZero, BetaZero, Packed>>(
        dim3{CGEMM_SIMT_THREADS, 1, 1}, reinterpret_cast<__gm__ const float*>(t1),
        reinterpret_cast<__gm__ const float*>(t2), reinterpret_cast<__gm__ const float*>(t3),
        reinterpret_cast<__gm__ const float*>(t4), reinterpret_cast<__gm__ float*>(c), tiling.m, tiling.n, tiling.ldc,
        tiling.tempLdc, tiling.alphaReal, tiling.alphaImag, tiling.betaReal, tiling.betaImag, tiling.threeProducts != 0,
        tiling.kPartitions, tiling.partStride, GetBlockIdx(), GetBlockNum());
}

template <bool Packed>
__aicore__ inline void LaunchGenericCgemmEpilogue(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, const CgemmEpilogueTilingData& tiling)
{
    if (tiling.alphaZero != 0) {
        if (tiling.betaZero != 0) {
            LaunchCgemmEpilogueVf<true, true, Packed>(t1, t2, t3, t4, c, tiling);
        } else {
            LaunchCgemmEpilogueVf<true, false, Packed>(t1, t2, t3, t4, c, tiling);
        }
    } else if (tiling.betaZero != 0) {
        LaunchCgemmEpilogueVf<false, true, Packed>(t1, t2, t3, t4, c, tiling);
    } else {
        LaunchCgemmEpilogueVf<false, false, Packed>(t1, t2, t3, t4, c, tiling);
    }
}

extern "C" __global__ __aicore__ void gemm_cgemm_epilogue_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, const CgemmEpilogueTilingData tiling, GM_ADDR originalA,
    GM_ADDR originalB)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.referenceK != 0) {
        LaunchCorrectableEpilogue(t1, t2, t3, originalA, originalB, c, tiling);
        return;
    }
    if (tiling.threeProducts != 0 && tiling.alphaZero == 0 && tiling.betaZero != 0 && tiling.alphaReal == 1.0f &&
        tiling.alphaImag == 0.0f) {
        LaunchPlanarOutput(t1, t2, t3, c, tiling);
        return;
    }
    const bool packed = tiling.ldc == tiling.m && tiling.tempLdc == tiling.m;
    if (packed) {
        LaunchGenericCgemmEpilogue<true>(t1, t2, t3, t4, c, tiling);
    } else {
        LaunchGenericCgemmEpilogue<false>(t1, t2, t3, t4, c, tiling);
    }
}

} // namespace cgemm_epilogue
void gemm_cgemm_epilogue_do(
    uint32_t numBlocks, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR cInOut,
    const CgemmEpilogueTilingData& tilingData, GM_ADDR originalA, GM_ADDR originalB)
{
    cgemm_epilogue::gemm_cgemm_epilogue_kernel<<<numBlocks, nullptr, stream>>>(
        t1, t2, t3, t4, cInOut, tilingData, originalA, originalB);
}
#endif
