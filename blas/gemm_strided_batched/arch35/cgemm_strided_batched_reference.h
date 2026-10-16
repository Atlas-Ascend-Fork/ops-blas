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
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "cgemm_strided_batched_kernel.h"
#include "simt_api/asc_simt.h"

// Input-driven precision routing and ordered-prefix boundaries.
// Cube and normalization share the exact boundary of the ordered suffix.
// Existing prefix caps reserve an ordered suffix for long reductions.
constexpr uint32_t PREFIX_SHORT_K_BOUNDARY = 1024;
constexpr uint32_t PREFIX_SHORT_CAP = 832;
constexpr uint32_t PREFIX_MAX_K = 1280;
constexpr uint32_t PREFIX_MEDIUM_K_BOUNDARY = 1536;
constexpr uint32_t PREFIX_MIN_SUFFIX = 32;
constexpr uint32_t PREFIX_SHORT_ALIGNMENT = 16;
constexpr uint32_t PREFIX_LONG_ALIGNMENT = 32;
constexpr float ORDERED_MOMENT_THRESHOLD = 2048;

// The 4/5 short and 3/4 long prefix caps leave respectively 1/5 and 1/4
// for ordered FP32 accumulation, before the existing alignment and max cap.
// Keep numerator multiplication before denominator division for truncation.
constexpr uint32_t PREFIX_SHORT_NUMERATOR = 4;
constexpr uint32_t PREFIX_SHORT_DENOMINATOR = 5;
constexpr uint32_t PREFIX_LONG_NUMERATOR = 3;
constexpr uint32_t PREFIX_LONG_DENOMINATOR = 4;

__aicore__ inline uint32_t CgemmSbPrefixK(const CgemmSbTiling& t)
{
    if (t.k < CGEMM_SB_ORDERED_MIN_K)
        return 0;
    if (t.k < PREFIX_SHORT_K_BOUNDARY) {
        uint32_t length = (t.k - PREFIX_MIN_SUFFIX) / PREFIX_SHORT_ALIGNMENT * PREFIX_SHORT_ALIGNMENT;
        return length < PREFIX_SHORT_CAP ? length : PREFIX_SHORT_CAP;
    }
    uint32_t length = t.k <= PREFIX_MAX_K ? uint32_t(
                                                uint64_t(t.k) * PREFIX_SHORT_NUMERATOR / PREFIX_SHORT_DENOMINATOR /
                                                PREFIX_SHORT_ALIGNMENT * PREFIX_SHORT_ALIGNMENT) :
                                            (t.k <= PREFIX_MEDIUM_K_BOUNDARY ?
                                                 PREFIX_SHORT_K_BOUNDARY :
                                                 uint32_t(
                                                     uint64_t(t.k) * PREFIX_LONG_NUMERATOR / PREFIX_LONG_DENOMINATOR /
                                                     PREFIX_LONG_ALIGNMENT * PREFIX_LONG_ALIGNMENT));
    return length < PREFIX_MAX_K ? length : PREFIX_MAX_K;
}

// Coherent long reductions can accumulate a systematic FP32 rounding drift.
// Use the reference order in that numerical regime, based only on input moments.
template <bool Global, class Tensor>
__aicore__ inline uint32_t CgemmSbReadReferenceMask(Tensor flagValues, const CgemmSbTiling& t)
{
#pragma clang fp contract(off)
    // Keep the input-moment threshold independent of automatic FMA contraction.
    if constexpr (Global) {
        // All producers have completed before this phase. Invalidate once,
        // rather than invalidating a cache line again for each flag record.
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(flagValues);
    }
    float ar = 0, ai = 0, br = 0, bi = 0, ca = 0, cb = 0;
    for (uint32_t i = 0; i < t.flagCores; ++i) {
        uint64_t offset = uint64_t(i) * CGEMM_SB_FLAG_STRIDE;
        if (flagValues.GetValue(offset + CGEMM_SB_META_EXCEPTION) != 0.0f) {
            return CGEMM_SB_REFERENCE_EXCEPTIONAL;
        }
        if (t.threeM && t.k >= CGEMM_SB_ORDERED_MIN_K) {
            ar += flagValues.GetValue(offset + CGEMM_SB_META_A_REAL);
            ai += flagValues.GetValue(offset + CGEMM_SB_META_A_IMAG);
            br += flagValues.GetValue(offset + CGEMM_SB_META_B_REAL);
            bi += flagValues.GetValue(offset + CGEMM_SB_META_B_IMAG);
            ca += flagValues.GetValue(offset + CGEMM_SB_META_A_COUNT);
            cb += flagValues.GetValue(offset + CGEMM_SB_META_B_COUNT);
        }
    }
    if (t.threeM && t.k >= CGEMM_SB_ORDERED_MIN_K && ca > 0 && cb > 0) {
        ar /= ca;
        ai /= ca;
        br /= cb;
        bi /= cb;
        float real = float(t.k) * (ar * br - ai * bi), imag = float(t.k) * (ar * bi + ai * br);
        return (real > ORDERED_MOMENT_THRESHOLD || real < -ORDERED_MOMENT_THRESHOLD ? CGEMM_SB_REFERENCE_REAL :
                                                                                      CGEMM_SB_REFERENCE_NONE) |
               (imag > ORDERED_MOMENT_THRESHOLD || imag < -ORDERED_MOMENT_THRESHOLD ? CGEMM_SB_REFERENCE_IMAG :
                                                                                      CGEMM_SB_REFERENCE_NONE);
    }
    return CGEMM_SB_REFERENCE_NONE;
}

__aicore__ inline uint32_t CgemmSbReferenceMask(__gm__ float* flags, const CgemmSbTiling& t)
{
    AscendC::GlobalTensor<float> flagValues;
    flagValues.SetGlobalBuffer(flags);
    if ASCEND_IS_AIV {
        // Called before phase-specific UB buffers are initialized. One DMA
        // observes the published flags without serial scalar GM/cache reads.
        uint32_t count = t.flagCores * CGEMM_SB_FLAG_STRIDE;
        AscendC::LocalTensor<float> cached(AscendC::TPosition::VECIN, 0, count);
        AscendC::DataCopy(cached, flagValues, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        uint32_t result = CgemmSbReadReferenceMask<false>(cached, t);
        AscendC::PipeBarrier<PIPE_ALL>();
        return result;
    } else {
        return CgemmSbReadReferenceMask<true>(flagValues, t);
    }
}

// Bits 0/1 request an ordered real/imaginary component; bit 2 requests the
// exceptional-value path. Each component uses the same input-moment criterion.
__aicore__ inline bool CgemmSbNeedsReference(__gm__ float* flags, const CgemmSbTiling& t)
{
    return CgemmSbReferenceMask(flags, t) != CGEMM_SB_REFERENCE_NONE;
}

__aicore__ inline bool CgemmSbPartialReference(uint32_t mask, const CgemmSbTiling& t)
{
    return (mask == CGEMM_SB_REFERENCE_REAL || mask == CGEMM_SB_REFERENCE_IMAG) && t.threeM &&
           t.k >= CGEMM_SB_ORDERED_MIN_K;
}

// Shared inline arithmetic and SIMT fallbacks must remain visible to the
// direct, local and dense Vector kernel translation units.

__simt_callee__ __aicore__ inline uint32_t CgemmSbNanBits(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    if ((a & CGEMM_SB_FP32_ABS_MASK) > CGEMM_SB_FP32_INFINITY_BITS)
        return a | CGEMM_SB_FP32_QUIET_NAN_BIT;
    if ((b & CGEMM_SB_FP32_ABS_MASK) > CGEMM_SB_FP32_INFINITY_BITS)
        return b | CGEMM_SB_FP32_QUIET_NAN_BIT;
    if ((c & CGEMM_SB_FP32_ABS_MASK) > CGEMM_SB_FP32_INFINITY_BITS)
        return c | CGEMM_SB_FP32_QUIET_NAN_BIT;
    if ((d & CGEMM_SB_FP32_ABS_MASK) > CGEMM_SB_FP32_INFINITY_BITS)
        return d | CGEMM_SB_FP32_QUIET_NAN_BIT;
    return CGEMM_SB_FP32_INVALID_NAN_BITS;
}

__simt_callee__ __aicore__ inline void CgemmSbRecoverInfinite(
    float ar, float ai, float br, float bi, float& real, float& imag)
{
#pragma clang fp contract(off)
    bool infA = isinf(ar) || isinf(ai);
    bool infB = isinf(br) || isinf(bi);
    if (infA) {
        ar = copysignf(isinf(ar) ? 1.0f : 0.0f, ar);
        ai = copysignf(isinf(ai) ? 1.0f : 0.0f, ai);
        if (isnan(br))
            br = copysignf(0.0f, br);
        if (isnan(bi))
            bi = copysignf(0.0f, bi);
    }
    if (infB) {
        br = copysignf(isinf(br) ? 1.0f : 0.0f, br);
        bi = copysignf(isinf(bi) ? 1.0f : 0.0f, bi);
        if (isnan(ar))
            ar = copysignf(0.0f, ar);
        if (isnan(ai))
            ai = copysignf(0.0f, ai);
    }
    if (infA || infB) {
        real = __builtin_inff() * (ar * br - ai * bi);
        imag = __builtin_inff() * (ar * bi + ai * br);
    }
}

__simt_callee__ __aicore__ inline void CgemmSbScale(float ar, float ai, float br, float bi, float& real, float& imag)
{
#pragma clang fp contract(off)
    // Preserve the separate roundings used by the complex scaling reference.
    volatile float rr = ar * br;
    volatile float ii = ai * bi;
    volatile float ri = ar * bi;
    volatile float ir = ai * br;
    real = rr - ii;
    imag = ri + ir;
    bool inputNan = isnan(ar) || isnan(ai) || isnan(br) || isnan(bi);
    if (isnan(real) && isnan(imag)) {
        CgemmSbRecoverInfinite(ar, ai, br, bi, real, imag);
    }
    // Netlib on IEEE x86 produces a negative quiet NaN for invalid arithmetic.
    if (!inputNan) {
        if (isnan(real))
            real = -__builtin_nanf("");
        if (isnan(imag))
            imag = -__builtin_nanf("");
    }
}

__simt_callee__ __aicore__ inline void CgemmSbAccumulateComplex(
    float ar, float ai, float br, float bi, float& real, float& imag)
{
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    float rr = ar * br;
    float ii = ai * bi;
    float ri = ar * bi;
    float ir = ai * br;
    float pr = rr - ii;
    float pi = ri + ir;
    if (isnan(pr) && isnan(pi))
        CgemmSbScale(ar, ai, br, bi, pr, pi);
    real += pr;
    imag += pi;
}

// Register accumulators for unit alpha / zero beta. The address increments are
// invariant along K; no workspace or intermediate output planes are needed.
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_SB_SIMT_LAUNCH_BOUND) inline void CgemmSbUnitVector(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, CgemmSbTiling t, uint32_t core, uint32_t cores)
{
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    uint32_t perBatch = t.m * t.n;
    for (uint32_t idx = core * blockDim.x + threadIdx.x; idx < perBatch * t.batch; idx += cores * blockDim.x) {
        uint32_t batch = idx / perBatch;
        uint32_t row = idx % t.m;
        uint32_t col = idx % perBatch / t.m;
        auto ap = a + 2 * (int64_t(batch) * t.strideA + (t.transA == ACLBLAS_OP_N ? row : int64_t(row) * t.lda));
        auto bp = b + 2 * (int64_t(batch) * t.strideB + (t.transB == ACLBLAS_OP_N ? int64_t(col) * t.ldb : col));
        uint32_t astep = t.transA == ACLBLAS_OP_N ? 2 * t.lda : 2;
        uint32_t bstep = t.transB == ACLBLAS_OP_N ? 2 : 2 * t.ldb;
        float sr = 0;
        float si = 0;
        for (uint32_t p = 0; p < t.k; ++p) {
            float ar = ap[0];
            float ai = ap[1];
            float br = bp[0];
            float bi = bp[1];
            if (t.transA == ACLBLAS_OP_C)
                ai = -ai;
            if (t.transB == ACLBLAS_OP_C)
                bi = -bi;
            if (t.transA == ACLBLAS_OP_N && (!isfinite(br) || !isfinite(bi))) {
                float tr = br - 0.0f * bi;
                float ti = bi + 0.0f * br;
                br = tr;
                bi = ti;
            }
            CgemmSbAccumulateComplex(ar, ai, br, bi, sr, si);
            ap += astep;
            bp += bstep;
        }
        if (t.transA != ACLBLAS_OP_N && (!isfinite(sr) || !isfinite(si))) {
            float real;
            float imag;
            CgemmSbScale(1.0f, 0.0f, sr, si, real, imag);
            sr = real;
            si = imag;
        }
        auto out = c + 2 * (int64_t(batch) * t.strideC + int64_t(col) * t.ldc + row);
        out[0] = sr;
        out[1] = si;
    }
}

__simt_callee__ __aicore__ inline void CgemmSbDirectReduction(
    __gm__ float* a, __gm__ float* b, const CgemmSbTiling& t, uint64_t batch, uint32_t row, uint32_t col, float& sumR,
    float& sumI)
{
#pragma clang fp contract(off)
    bool noTransA = t.transA == ACLBLAS_OP_N;
    for (uint32_t p = 0; p < t.k; ++p) {
        int64_t ax = int64_t(batch) * t.strideA +
                     (t.transA == ACLBLAS_OP_N ? int64_t(p) * t.lda + row : int64_t(row) * t.lda + p);
        int64_t bx = int64_t(batch) * t.strideB +
                     (t.transB == ACLBLAS_OP_N ? int64_t(col) * t.ldb + p : int64_t(p) * t.ldb + col);
        float ar = a[2 * ax];
        float ai = a[2 * ax + 1];
        float br = b[2 * bx];
        float bi = b[2 * bx + 1];
        if (t.transA == ACLBLAS_OP_C)
            ai = -ai;
        if (t.transB == ACLBLAS_OP_C)
            bi = -bi;
        if (noTransA) {
            float tempR = t.ar * br - t.ai * bi;
            float tempI = t.ar * bi + t.ai * br;
            volatile float rr = tempR * ar;
            volatile float ii = tempI * ai;
            volatile float ri = tempR * ai;
            volatile float ir = tempI * ar;
            float productR = rr - ii;
            float productI = ri + ir;
            if (isnan(productR) && isnan(productI))
                CgemmSbScale(tempR, tempI, ar, ai, productR, productI);
            sumR += productR;
            sumI += productI;
        } else {
            volatile float rr = ar * br;
            volatile float ii = ai * bi;
            volatile float ri = ar * bi;
            volatile float ir = ai * br;
            float productR = rr - ii;
            float productI = ri + ir;
            if (isnan(productR) && isnan(productI))
                CgemmSbScale(ar, ai, br, bi, productR, productI);
            sumR += productR;
            sumI += productI;
        }
    }
}

__simt_callee__ __aicore__ inline void CgemmSbDirectElement(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const CgemmSbTiling& t, uint64_t batch, uint32_t row,
    uint32_t col, int64_t ci, float& real, float& imag)
{
#pragma clang fp contract(off)
    float sumR = 0;
    float sumI = 0;
    bool noTransA = t.transA == ACLBLAS_OP_N;
    if (noTransA && (t.br != 0 || t.bi != 0)) {
        float cr = c[2 * ci];
        float cv = c[2 * ci + 1];
        if (t.br == 1 && t.bi == 0) {
            sumR = cr;
            sumI = cv;
        } else {
            sumR = t.br * cr - t.bi * cv;
            sumI = t.br * cv + t.bi * cr;
        }
    }
    CgemmSbDirectReduction(a, b, t, batch, row, col, sumR, sumI);
    real = sumR;
    imag = sumI;
}

__simt_callee__ __aicore__ inline void CgemmSbStoreExactNan(
    __gm__ float* c, const CgemmSbTiling& t, int64_t ci, float real, float imag)
{
    // Preserve IEEE NaN payload/sign for the EXACT scaling contract.
    // Floating stores may canonicalize NaNs on the vector processor.
    auto bits = reinterpret_cast<__gm__ uint32_t*>(c);
    uint32_t cr = bits[2 * ci];
    uint32_t cv = bits[2 * ci + 1];
    union FloatBits {
        float value;
        uint32_t bits;
    } brBits, biBits;
    brBits.value = t.br;
    biBits.value = t.bi;
    if (isnan(real))
        bits[2 * ci] = CgemmSbNanBits(brBits.bits, cr, biBits.bits, cv);
    else
        c[2 * ci] = real;
    if (isnan(imag))
        bits[2 * ci + 1] = CgemmSbNanBits(brBits.bits, cv, biBits.bits, cr);
    else
        c[2 * ci + 1] = imag;
}

template <typename Index>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_SB_SIMT_LAUNCH_BOUND) inline void CgemmSbVector(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, __gm__ float* ws, CgemmSbTiling t, uint32_t mode, uint32_t core,
    uint32_t cores)
{
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    Index start = Index(core) * blockDim.x + threadIdx.x;
    Index step = Index(cores) * blockDim.x;
    Index perBatch = Index(t.m) * t.n;
    for (Index idx = start; idx < perBatch * t.batch; idx += step) {
        Index batch = idx / perBatch;
        uint32_t row = idx % t.m;
        uint32_t col = (idx % perBatch) / t.m;
        int64_t ci = int64_t(batch) * t.strideC + int64_t(col) * t.ldc + row;
        float real = 0;
        float imag = 0;
        if (mode == CGEMM_SB_MODE_DIRECT) {
            CgemmSbDirectElement(a, b, c, t, batch, row, col, ci, real, imag);
            if (t.transA == ACLBLAS_OP_N) {
                c[2 * ci] = real;
                c[2 * ci + 1] = imag;
                continue;
            }
        }
        if (mode != CGEMM_SB_MODE_SCALE) {
            float scaledReal;
            float scaledImag;
            CgemmSbScale(t.ar, t.ai, real, imag, scaledReal, scaledImag);
            real = scaledReal;
            imag = scaledImag;
        }
        if (t.br != 0 || t.bi != 0) {
            float cr = c[2 * ci];
            float ciVal = c[2 * ci + 1];
            float betaReal;
            float betaImag;
            CgemmSbScale(t.br, t.bi, cr, ciVal, betaReal, betaImag);
            if (mode == CGEMM_SB_MODE_SCALE) {
                real = betaReal;
                imag = betaImag;
            } else {
                real += betaReal;
                imag += betaImag;
            }
        }
        if (mode == CGEMM_SB_MODE_SCALE && (isnan(real) || isnan(imag))) {
            CgemmSbStoreExactNan(c, t, ci, real, imag);
        } else {
            c[2 * ci] = real;
            c[2 * ci + 1] = imag;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_SB_SIMT_LAUNCH_BOUND) inline void CgemmSbMergeComponent(
    __gm__ float* c, __gm__ float* ws, CgemmSbTiling t, uint32_t mask, uint32_t core, uint32_t cores)
{
    uint64_t matrix = uint64_t(t.m) * t.n;
    for (uint64_t i = uint64_t(core) * blockDim.x + threadIdx.x; i < matrix * t.batch;
         i += uint64_t(cores) * blockDim.x) {
        uint64_t batch = i / matrix;
        uint64_t row = i % t.m;
        uint64_t col = i % matrix / t.m;
        int64_t dest = int64_t(batch) * t.strideC + int64_t(col) * t.ldc + row;
        uint64_t source = t.cOffset + batch * uint64_t(t.mp) * t.np + col * t.mp + row;
        if (mask == CGEMM_SB_REFERENCE_IMAG)
            c[2 * dest] = ws[source] - ws[source + t.cSize];
        else
            c[2 * dest + 1] = ws[source] + ws[source + t.cSize];
    }
}
