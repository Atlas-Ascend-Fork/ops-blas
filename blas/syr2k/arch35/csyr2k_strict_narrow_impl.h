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
 * \file csyr2k_strict_narrow_impl.h
 * \brief Strict narrow-matrix implementation for aclblasCsyr2k on Ascend 950.
 */

#pragma once

namespace AscendC {

constexpr uint32_t CSYR2K_SMALL_SIMT_THREADS = 2048;

template <bool UPPER>
__simt_callee__ __aicore__ inline bool Csyr2kOutsideTriangle(uint32_t row, uint32_t col)
{
    if constexpr (UPPER) {
        return row > col;
    }
    return row < col;
}

__simt_callee__ __aicore__ inline void Csyr2kInitSimtRange(
    uint32_t n, uint64_t& total, uint64_t& linear, uint64_t& stride)
{
    total = static_cast<uint64_t>(n) * n;
    linear = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
}

struct Csyr2kScalars {
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    bool alphaZero;
    bool betaZero;
};

__aicore__ inline Csyr2kScalars LoadCsyr2kScalars(GM_ADDR alpha, GM_ADDR beta)
{
    const auto* alphaGm = reinterpret_cast<const __gm__ float*>(alpha);
    const auto* betaGm = reinterpret_cast<const __gm__ float*>(beta);
    float alphaReal = alphaGm[0];
    float alphaImag = alphaGm[1];
    float betaReal = betaGm[0];
    float betaImag = betaGm[1];
    return {alphaReal,
            alphaImag,
            betaReal,
            betaImag,
            alphaReal == 0.0f && alphaImag == 0.0f,
            betaReal == 0.0f && betaImag == 0.0f};
}

struct Csyr2kStrictAccum {
    float valueReal;
    float valueImag;
    float plainReal;
    float plainImag;
    float strictReal;
    float strictImag;
    float strictPlainReal;
    float strictPlainImag;
    float correctionReal = 0.0f;
    float correctionImag = 0.0f;
    float strictCorrectionReal = 0.0f;
    float strictCorrectionImag = 0.0f;
};

struct Csyr2kStrictTerm {
    float real;
    float imag;
    float strictReal;
    float strictImag;
};

__simt_callee__ __aicore__ inline void InitCsyr2kStrictAccum(
    Csyr2kStrictAccum& accum, __gm__ const float* c, uint64_t offset, bool betaZero, float betaReal, float betaImag)
{
    float real = 0.0f;
    float imag = 0.0f;
    if (!betaZero) {
        float cReal = c[offset];
        float cImag = c[offset + 1];
        real = betaReal * cReal - betaImag * cImag;
        imag = betaReal * cImag + betaImag * cReal;
    }
    accum.valueReal = real;
    accum.valueImag = imag;
    accum.plainReal = real;
    accum.plainImag = imag;
    accum.strictReal = real;
    accum.strictImag = imag;
    accum.strictPlainReal = real;
    accum.strictPlainImag = imag;
}

template <bool TRANS_N>
__simt_callee__ __aicore__ inline void ComputeCsyr2kStrictTerm(
    Csyr2kStrictTerm& term, uint32_t inner, uint32_t row, uint32_t col, uint32_t lda, uint32_t ldb, float alphaReal,
    float alphaImag, __gm__ const float* a, __gm__ const float* b)
{
#pragma clang fp contract(off)
    uint64_t aRow = (TRANS_N ? static_cast<uint64_t>(inner) * lda + row : static_cast<uint64_t>(row) * lda + inner) * 2;
    uint64_t bCol = (TRANS_N ? static_cast<uint64_t>(inner) * ldb + col : static_cast<uint64_t>(col) * ldb + inner) * 2;
    uint64_t bRow = (TRANS_N ? static_cast<uint64_t>(inner) * ldb + row : static_cast<uint64_t>(row) * ldb + inner) * 2;
    uint64_t aCol = (TRANS_N ? static_cast<uint64_t>(inner) * lda + col : static_cast<uint64_t>(col) * lda + inner) * 2;
    float ar = a[aRow];
    float ai = a[aRow + 1];
    float bcReal = b[bCol];
    float bcImag = b[bCol + 1];
    float br = b[bRow];
    float bi = b[bRow + 1];
    float acReal = a[aCol];
    float acImag = a[aCol + 1];
    float p1Real = ar * bcReal - ai * bcImag;
    float p1Imag = ar * bcImag + ai * bcReal;
    float p2Real = br * acReal - bi * acImag;
    float p2Imag = br * acImag + bi * acReal;
    volatile float p1rr = ar * bcReal;
    volatile float p1ii = ai * bcImag;
    volatile float p1ri = ar * bcImag;
    volatile float p1ir = ai * bcReal;
    volatile float p2rr = br * acReal;
    volatile float p2ii = bi * acImag;
    volatile float p2ri = br * acImag;
    volatile float p2ir = bi * acReal;
    float strictP1Real = p1rr - p1ii;
    float strictP1Imag = p1ri + p1ir;
    float strictP2Real = p2rr - p2ii;
    float strictP2Imag = p2ri + p2ir;
    term.real = (alphaReal * p1Real - alphaImag * p1Imag) + (alphaReal * p2Real - alphaImag * p2Imag);
    term.imag = (alphaReal * p1Imag + alphaImag * p1Real) + (alphaReal * p2Imag + alphaImag * p2Real);
    term.strictReal =
        (alphaReal * strictP1Real - alphaImag * strictP1Imag) + (alphaReal * strictP2Real - alphaImag * strictP2Imag);
    term.strictImag =
        (alphaReal * strictP1Imag + alphaImag * strictP1Real) + (alphaReal * strictP2Imag + alphaImag * strictP2Real);
}

__simt_callee__ __aicore__ inline void AccumulateCsyr2kStrictTerm(
    Csyr2kStrictAccum& accum, const Csyr2kStrictTerm& term)
{
    constexpr float fp32Max = 3.402823466e+38F;
    accum.plainReal += term.real;
    accum.plainImag += term.imag;
    accum.strictPlainReal += term.strictReal;
    accum.strictPlainImag += term.strictImag;
    float adjustedReal = term.real - accum.correctionReal;
    float nextReal = accum.valueReal + adjustedReal;
    bool finiteReal =
        nextReal <= fp32Max && nextReal >= -fp32Max && adjustedReal <= fp32Max && adjustedReal >= -fp32Max;
    accum.correctionReal = finiteReal ? (nextReal - accum.valueReal) - adjustedReal : 0.0f;
    accum.valueReal = nextReal;
    float adjustedImag = term.imag - accum.correctionImag;
    float nextImag = accum.valueImag + adjustedImag;
    bool finiteImag =
        nextImag <= fp32Max && nextImag >= -fp32Max && adjustedImag <= fp32Max && adjustedImag >= -fp32Max;
    accum.correctionImag = finiteImag ? (nextImag - accum.valueImag) - adjustedImag : 0.0f;
    accum.valueImag = nextImag;
    float strictAdjustedReal = term.strictReal - accum.strictCorrectionReal;
    float strictNextReal = accum.strictReal + strictAdjustedReal;
    bool strictFiniteReal = strictNextReal <= fp32Max && strictNextReal >= -fp32Max && strictAdjustedReal <= fp32Max &&
                            strictAdjustedReal >= -fp32Max;
    accum.strictCorrectionReal = strictFiniteReal ? (strictNextReal - accum.strictReal) - strictAdjustedReal : 0.0f;
    accum.strictReal = strictNextReal;
    float strictAdjustedImag = term.strictImag - accum.strictCorrectionImag;
    float strictNextImag = accum.strictImag + strictAdjustedImag;
    bool strictFiniteImag = strictNextImag <= fp32Max && strictNextImag >= -fp32Max && strictAdjustedImag <= fp32Max &&
                            strictAdjustedImag >= -fp32Max;
    accum.strictCorrectionImag = strictFiniteImag ? (strictNextImag - accum.strictImag) - strictAdjustedImag : 0.0f;
    accum.strictImag = strictNextImag;
}

__simt_callee__ __aicore__ inline void StoreCsyr2kStrictResult(
    __gm__ float* c, uint64_t offset, bool alphaZero, const Csyr2kStrictAccum& accum)
{
    if (alphaZero) {
        c[offset] = accum.valueReal;
        c[offset + 1] = accum.valueImag;
        return;
    }
    float correctedReal = accum.valueReal - accum.correctionReal;
    float correctedImag = accum.valueImag - accum.correctionImag;
    float strictCorrectedReal = accum.strictReal - accum.strictCorrectionReal;
    float strictCorrectedImag = accum.strictImag - accum.strictCorrectionImag;
    c[offset] = correctedReal;
    c[offset + 1] = correctedImag;
}

template <bool UPPER, bool TRANS_N>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Csyr2kStrictNarrowSimt(
    uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaReal, float alphaImag, float betaReal,
    float betaImag, bool alphaZero, bool betaZero, __gm__ const float* a, __gm__ const float* b, __gm__ float* c)
{
    uint64_t total = 0;
    uint64_t linear = 0;
    uint64_t stride = 0;
    Csyr2kInitSimtRange(n, total, linear, stride);
    while (linear < total) {
        uint32_t row = static_cast<uint32_t>(linear % n);
        uint32_t col = static_cast<uint32_t>(linear / n);
        if (Csyr2kOutsideTriangle<UPPER>(row, col)) {
            linear += stride;
            continue;
        }

        uint64_t cOffset = (static_cast<uint64_t>(col) * ldc + row) * 2;
        Csyr2kStrictAccum accum{};
        InitCsyr2kStrictAccum(accum, c, cOffset, betaZero, betaReal, betaImag);

        if (!alphaZero) {
            for (uint32_t inner = 0; inner < k; ++inner) {
                Csyr2kStrictTerm term{};
                ComputeCsyr2kStrictTerm<TRANS_N>(term, inner, row, col, lda, ldb, alphaReal, alphaImag, a, b);
                AccumulateCsyr2kStrictTerm(accum, term);
            }
        }
        StoreCsyr2kStrictResult(c, cOffset, alphaZero, accum);
        linear += stride;
    }
}

extern "C" __global__ __aicore__ void csyr2k_strict_narrow_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, const Csyr2kStrictNarrowTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Csyr2kScalars scalars = LoadCsyr2kScalars(alpha, beta);
    bool betaOne = (scalars.betaReal == 1.0f && scalars.betaImag == 0.0f);
    if (scalars.alphaZero && betaOne) {
        return;
    }

    if (tiling.uploMode == ACLBLAS_UPPER) {
        asc_vf_call<Csyr2kStrictNarrowSimt<true, false>>(
            dim3{CSYR2K_SMALL_SIMT_THREADS, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.ldb, tiling.ldc,
            scalars.alphaReal, scalars.alphaImag, scalars.betaReal, scalars.betaImag, scalars.alphaZero,
            scalars.betaZero, reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(b),
            reinterpret_cast<__gm__ float*>(c));
    } else {
        asc_vf_call<Csyr2kStrictNarrowSimt<false, false>>(
            dim3{CSYR2K_SMALL_SIMT_THREADS, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.ldb, tiling.ldc,
            scalars.alphaReal, scalars.alphaImag, scalars.betaReal, scalars.betaImag, scalars.alphaZero,
            scalars.betaZero, reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(b),
            reinterpret_cast<__gm__ float*>(c));
    }
}

} // namespace AscendC
