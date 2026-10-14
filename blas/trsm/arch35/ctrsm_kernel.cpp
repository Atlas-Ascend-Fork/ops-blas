/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#include "ctrsm_kernel.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "simt_api/asc_simt.h"

// Error compensation needs separately rounded products; explicit fmaf remains fused.
#pragma clang fp contract(off)

using namespace AscendC;

namespace {

constexpr uint32_t CTRSM_SIMT_MAX_THREADS = 64;
constexpr uint32_t CTRSM_SIMT_WARP_SIZE = 32;
constexpr uint32_t CTRSM_AUX_THREADS = 128;
// Keep solutions from near-FLT_MAX diagonals normalized until all dependent rows have consumed them.
constexpr float CTRSM_SCALE_THRESHOLD = 0x1p120f;
constexpr float CTRSM_SOLUTION_SCALE = 0x1p24f;
constexpr float CTRSM_INVERSE_SOLUTION_SCALE = 0x1p-24f;

struct ComplexValue {
    float real;
    float imag;
};

__simt_callee__ inline ComplexValue LoadComplex(__gm__ const float* data, uint64_t elementOffset)
{
    const uint64_t floatOffset = elementOffset * 2;
    return {data[floatOffset], data[floatOffset + 1]};
}

__simt_callee__ inline void StoreComplex(__gm__ float* data, uint64_t elementOffset, const ComplexValue& value)
{
    const uint64_t floatOffset = elementOffset * 2;
    data[floatOffset] = value.real;
    data[floatOffset + 1] = value.imag;
}

__simt_callee__ inline ComplexValue Multiply(const ComplexValue& lhs, const ComplexValue& rhs)
{
    ComplexValue result = {lhs.real * rhs.real - lhs.imag * rhs.imag, lhs.real * rhs.imag + lhs.imag * rhs.real};
    if (isnan(result.real) && isnan(result.imag)) {
        const bool leftInfinite = isinf(lhs.real) || isinf(lhs.imag);
        const bool rightInfinite = isinf(rhs.real) || isinf(rhs.imag);
        if (leftInfinite || rightInfinite) {
            // Preserve the direction of an infinite complex operand. A NaN in
            // its finite companion component must not erase a defined infinity.
            const ComplexValue left = {
                leftInfinite ? copysignf(isinf(lhs.real) ? 1.0f : 0.0f, lhs.real) :
                               (isnan(lhs.real) ? copysignf(0.0f, lhs.real) : lhs.real),
                leftInfinite ? copysignf(isinf(lhs.imag) ? 1.0f : 0.0f, lhs.imag) :
                               (isnan(lhs.imag) ? copysignf(0.0f, lhs.imag) : lhs.imag)};
            const ComplexValue right = {
                rightInfinite ? copysignf(isinf(rhs.real) ? 1.0f : 0.0f, rhs.real) :
                                (isnan(rhs.real) ? copysignf(0.0f, rhs.real) : rhs.real),
                rightInfinite ? copysignf(isinf(rhs.imag) ? 1.0f : 0.0f, rhs.imag) :
                                (isnan(rhs.imag) ? copysignf(0.0f, rhs.imag) : rhs.imag)};
            const float infinity = __builtin_huge_valf();
            result.real = infinity * (left.real * right.real - left.imag * right.imag);
            result.imag = infinity * (left.real * right.imag + left.imag * right.real);
        }
    }
    return result;
}

__simt_callee__ inline void SubtractProduct(ComplexValue& target, const ComplexValue& lhs, const ComplexValue& rhs)
{
    const ComplexValue product = Multiply(lhs, rhs);
    target.real -= product.real;
    target.imag -= product.imag;
}

struct FloatPair {
    float high;
    float low;
};

// Preserve product and division residuals until the final FLOAT32 rounding.
// The Netlib CPU complex division uses wider intermediates than a reciprocal
// rounded to FLOAT32; those lost bits matter after cancellation in later rows.
__simt_callee__ inline FloatPair SumProducts(float a, float b, float c, float d)
{
    const float first = a * b;
    const float second = c * d;
    const float sum = first + second;
    const float secondPart = sum - first;
    const float sumError = (first - (sum - secondPart)) + (second - secondPart);
    return {sum, sumError + fmaf(a, b, -first) + fmaf(c, d, -second)};
}

__simt_callee__ inline FloatPair DividePair(FloatPair numerator, FloatPair denominator)
{
    const float quotient = numerator.high / denominator.high;
    const float residual =
        fmaf(-quotient, denominator.high, numerator.high) + numerator.low - quotient * denominator.low;
    return {quotient, residual / denominator.high};
}

__simt_callee__ inline float ScalePowerOfTwo(float value, int power)
{
    const uint32_t bits = __float_as_uint(value);
    const int exponent = static_cast<int>((bits >> 23) & 0xffU);
    const int shiftedExponent = exponent + power;
    if (exponent > 0 && exponent < 255 && shiftedExponent > 0 && shiftedExponent < 255) {
        return __uint_as_float((bits & 0x807fffffU) | (static_cast<uint32_t>(shiftedExponent) << 23));
    }
    return scalbnf(value, power);
}

__simt_callee__ inline float RoundDivision(FloatPair quotient, int outputExponent, float resultScale)
{
    const float scaledHigh = ScalePowerOfTwo(quotient.high, outputExponent);
    if (resultScale == CTRSM_SOLUTION_SCALE && fabsf(scaledHigh) < 0x1p-102f) {
        // FLOAT32 subnormal spacing is 2^-149, hence 2^-125 at scale 2^24.
        // Round the high/low pair directly to this grid to avoid double rounding.
        const float highUnits = ScalePowerOfTwo(quotient.high, outputExponent + 125);
        const float lowUnits = ScalePowerOfTwo(quotient.low, outputExponent + 125);
        float roundedUnits = rintf(highUnits);
        const float highRemainder = highUnits - roundedUnits;
        const float remainder = highRemainder + lowUnits;
        const float lowPart = remainder - highRemainder;
        const float remainderError = (highRemainder - (remainder - lowPart)) + (lowUnits - lowPart);
        const bool odd = (static_cast<int32_t>(roundedUnits) & 1) != 0;
        if (remainder > 0.5f || (remainder == 0.5f && (remainderError > 0.0f || (remainderError == 0.0f && odd)))) {
            roundedUnits += 1.0f;
        } else if (
            remainder < -0.5f || (remainder == -0.5f && (remainderError < 0.0f || (remainderError == 0.0f && odd)))) {
            roundedUnits -= 1.0f;
        }
        return ScalePowerOfTwo(roundedUnits, -125);
    }
    return ScalePowerOfTwo(quotient.high + quotient.low, outputExponent);
}

__simt_callee__ inline ComplexValue DivideFinite(const ComplexValue& lhs, const ComplexValue& rhs, float resultScale)
{
    const float numeratorMax = fmaxf(fabsf(lhs.real), fabsf(lhs.imag));
    if (numeratorMax == 0.0f) {
        return {0.0f, 0.0f};
    }
    const int numeratorExponent = ilogbf(numeratorMax);
    const int denominatorExponent = ilogbf(fmaxf(fabsf(rhs.real), fabsf(rhs.imag)));
    const float a = ScalePowerOfTwo(lhs.real, -numeratorExponent);
    const float b = ScalePowerOfTwo(lhs.imag, -numeratorExponent);
    const float c = ScalePowerOfTwo(rhs.real, -denominatorExponent);
    const float d = ScalePowerOfTwo(rhs.imag, -denominatorExponent);
    const FloatPair denominator = SumProducts(c, c, d, d);
    const int outputExponent = numeratorExponent - denominatorExponent + (resultScale == CTRSM_SOLUTION_SCALE ? 24 : 0);
    return {
        RoundDivision(DividePair(SumProducts(a, c, b, d), denominator), outputExponent, resultScale),
        RoundDivision(DividePair(SumProducts(b, c, -a, d), denominator), outputExponent, resultScale)};
}

__simt_callee__ inline ComplexValue DivideExceptional(
    const ComplexValue& lhs, const ComplexValue& rhs, bool numeratorFinite, bool denominatorFinite)
{
    const float infinity = __builtin_huge_valf();
    const float notANumber = __builtin_nanf("");
    ComplexValue result = {notANumber, notANumber};
    if (rhs.real == 0.0f && rhs.imag == 0.0f) {
        return {copysignf(infinity, rhs.real) * lhs.real, copysignf(infinity, rhs.real) * lhs.imag};
    }
    if (numeratorFinite && (isinf(rhs.real) || isinf(rhs.imag))) {
        const float realDirection = copysignf(isinf(rhs.real) ? 1.0f : 0.0f, rhs.real);
        const float imagDirection = copysignf(isinf(rhs.imag) ? 1.0f : 0.0f, rhs.imag);
        return {
            0.0f * ((lhs.real * 0.25f) * realDirection + (lhs.imag * 0.25f) * imagDirection),
            0.0f * ((lhs.imag * 0.25f) * realDirection - (lhs.real * 0.25f) * imagDirection)};
    }
    if (!numeratorFinite && denominatorFinite) {
        // For finite nonzero divisors the squared magnitude is positive.
        // Use the original divisor to retain nonzero components whose
        // reciprocal would underflow before multiplication by infinity.
        result.real = lhs.real * rhs.real + lhs.imag * rhs.imag;
        result.imag = lhs.imag * rhs.real - lhs.real * rhs.imag;
    }
    // An infinite numerator can still have a defined infinite component even
    // when ordinary complex arithmetic produces NaN in both components.
    if (isnan(result.real) && isnan(result.imag) && (isinf(lhs.real) || isinf(lhs.imag)) && denominatorFinite) {
        const float realDirection = copysignf(isinf(lhs.real) ? 1.0f : 0.0f, lhs.real);
        const float imagDirection = copysignf(isinf(lhs.imag) ? 1.0f : 0.0f, lhs.imag);
        result.real = infinity * (realDirection * rhs.real + imagDirection * rhs.imag);
        result.imag = infinity * (imagDirection * rhs.real - realDirection * rhs.imag);
    }
    return result;
}

__simt_callee__ inline ComplexValue Divide(const ComplexValue& lhs, const ComplexValue& rhs, float resultScale)
{
    const bool numeratorFinite = isfinite(lhs.real) && isfinite(lhs.imag);
    const bool denominatorFinite = isfinite(rhs.real) && isfinite(rhs.imag);
    if (numeratorFinite && denominatorFinite && (rhs.real != 0.0f || rhs.imag != 0.0f)) {
        return DivideFinite(lhs, rhs, resultScale);
    }
    return DivideExceptional(lhs, rhs, numeratorFinite, denominatorFinite);
}

template <bool SIDE_RIGHT, uint32_t TRANS>
__simt_callee__ inline ComplexValue LoadCoefficient(
    __gm__ const float* aGm, uint32_t row, uint32_t dependency, uint64_t lda)
{
    uint64_t offset;
    if constexpr (!SIDE_RIGHT) {
        if constexpr (TRANS == ACLBLAS_OP_N) {
            offset = static_cast<uint64_t>(dependency) * lda + row;
        } else {
            offset = static_cast<uint64_t>(row) * lda + dependency;
        }
    } else {
        if constexpr (TRANS == ACLBLAS_OP_N) {
            offset = static_cast<uint64_t>(row) * lda + dependency;
        } else {
            offset = static_cast<uint64_t>(dependency) * lda + row;
        }
    }

    ComplexValue value = LoadComplex(aGm, offset);
    if constexpr (TRANS == ACLBLAS_OP_C) {
        value.imag = -value.imag;
    }
    return value;
}

template <bool SIDE_RIGHT, uint32_t TRANS, bool DIAG_UNIT>
__simt_callee__ inline float SolutionScaleForRow(__gm__ const float* aGm, uint32_t row, uint64_t lda)
{
    if constexpr (DIAG_UNIT || SIDE_RIGHT) {
        return 1.0f;
    }
    const ComplexValue diagonal = LoadCoefficient<SIDE_RIGHT, TRANS>(aGm, row, row, lda);
    const float realAbs = diagonal.real < 0.0f ? -diagonal.real : diagonal.real;
    const float imagAbs = diagonal.imag < 0.0f ? -diagonal.imag : diagonal.imag;
    return realAbs > CTRSM_SCALE_THRESHOLD || imagAbs > CTRSM_SCALE_THRESHOLD ? CTRSM_SOLUTION_SCALE : 1.0f;
}

template <bool SIDE_RIGHT>
__simt_callee__ inline uint64_t BOffset(uint32_t system, uint32_t row, uint64_t ldb)
{
    if constexpr (SIDE_RIGHT) {
        return static_cast<uint64_t>(row) * ldb + system;
    }
    return static_cast<uint64_t>(system) * ldb + row;
}

template <bool SIDE_RIGHT, uint32_t TRANS>
__simt_callee__ inline void ScaleSystem(
    __gm__ float* bGm, uint32_t system, uint32_t order, uint64_t ldb, const ComplexValue& alpha)
{
    if ((!SIDE_RIGHT && TRANS != ACLBLAS_OP_N) || alpha.real != 1.0f || alpha.imag != 0.0f) {
        for (uint32_t row = threadIdx.x; row < order; row += blockDim.x) {
            const uint64_t offset = BOffset<SIDE_RIGHT>(system, row, ldb);
            StoreComplex(bGm, offset, Multiply(alpha, LoadComplex(bGm, offset)));
        }
    }
    asc_syncthreads();
}

template <bool SIDE_RIGHT, uint32_t TRANS, bool DIAG_UNIT>
__simt_callee__ inline void RestoreSolutionScale(
    __gm__ const float* aGm, __gm__ float* bGm, uint32_t system, uint32_t order, uint64_t lda, uint64_t ldb)
{
    for (uint32_t row = threadIdx.x; row < order; row += blockDim.x) {
        if (SolutionScaleForRow<SIDE_RIGHT, TRANS, DIAG_UNIT>(aGm, row, lda) == CTRSM_SOLUTION_SCALE) {
            const uint64_t offset = BOffset<SIDE_RIGHT>(system, row, ldb);
            ComplexValue result = LoadComplex(bGm, offset);
            result.real *= CTRSM_INVERSE_SOLUTION_SCALE;
            result.imag *= CTRSM_INVERSE_SOLUTION_SCALE;
            StoreComplex(bGm, offset, result);
        }
    }
    asc_syncthreads();
}

template <bool SIDE_RIGHT, uint32_t TRANS, bool DIAG_UNIT>
__simt_callee__ inline void DivideSystemRow(
    __gm__ const float* aGm, __gm__ float* bGm, uint32_t system, uint32_t row, uint64_t lda, uint64_t ldb,
    float solutionScale)
{
    if (threadIdx.x == 0 && !DIAG_UNIT) {
        const uint64_t offset = BOffset<SIDE_RIGHT>(system, row, ldb);
        const ComplexValue diagonal = LoadCoefficient<SIDE_RIGHT, TRANS>(aGm, row, row, lda);
        const ComplexValue value = LoadComplex(bGm, offset);
        const ComplexValue result =
            SIDE_RIGHT ? Multiply(Divide({1.0f, 0.0f}, diagonal, 1.0f), value) : Divide(value, diagonal, solutionScale);
        StoreComplex(bGm, offset, result);
    }
    asc_syncthreads();
}

template <bool SIDE_RIGHT, uint32_t TRANS>
__simt_callee__ inline void UpdateSystemRows(
    __gm__ const float* aGm, __gm__ float* bGm, uint32_t system, uint32_t row, uint32_t start, uint32_t end,
    uint64_t lda, uint64_t ldb, float solutionScale)
{
    const ComplexValue solved = LoadComplex(bGm, BOffset<SIDE_RIGHT>(system, row, ldb));
    const float scale = solutionScale == CTRSM_SOLUTION_SCALE ? CTRSM_INVERSE_SOLUTION_SCALE : 1.0f;
    for (uint32_t target = start + threadIdx.x; target < end; target += blockDim.x) {
        const uint64_t offset = BOffset<SIDE_RIGHT>(system, target, ldb);
        ComplexValue coefficient = LoadCoefficient<SIDE_RIGHT, TRANS>(aGm, target, row, lda);
        if (SIDE_RIGHT && coefficient.real == 0.0f && coefficient.imag == 0.0f) {
            continue;
        }
        if (scale != 1.0f && isfinite(solved.real) && isfinite(solved.imag)) {
            coefficient.real *= scale;
            coefficient.imag *= scale;
        }
        ComplexValue value = LoadComplex(bGm, offset);
        SubtractProduct(value, solved, coefficient);
        StoreComplex(bGm, offset, value);
    }
    asc_syncthreads();
}

template <bool SIDE_RIGHT, bool UPLO_UPPER, uint32_t TRANS, bool DIAG_UNIT>
__simt_callee__ inline void SolveSystem(
    __gm__ const float* aGm, __gm__ float* bGm, uint32_t system, uint32_t order, uint64_t lda, uint64_t ldb,
    const ComplexValue& alpha)
{
    if constexpr (!SIDE_RIGHT || TRANS == ACLBLAS_OP_N) {
        ScaleSystem<SIDE_RIGHT, TRANS>(bGm, system, order, ldb, alpha);
    }

    constexpr bool effectiveUpper = SIDE_RIGHT ? (TRANS == ACLBLAS_OP_N ? !UPLO_UPPER : UPLO_UPPER) :
                                                 (TRANS == ACLBLAS_OP_N ? UPLO_UPPER : !UPLO_UPPER);
    constexpr bool forward = !effectiveUpper;
    for (uint32_t iteration = 0; iteration < order; ++iteration) {
        const uint32_t row = forward ? iteration : order - 1 - iteration;
        // The branch is uniform across this system's threads and precedes division.
        if constexpr (!SIDE_RIGHT && TRANS == ACLBLAS_OP_N) {
            const ComplexValue rhs = LoadComplex(bGm, BOffset<SIDE_RIGHT>(system, row, ldb));
            asc_syncthreads();
            if (rhs.real == 0.0f && rhs.imag == 0.0f) {
                continue;
            }
        }
        const float solutionScale = SolutionScaleForRow<SIDE_RIGHT, TRANS, DIAG_UNIT>(aGm, row, lda);
        DivideSystemRow<SIDE_RIGHT, TRANS, DIAG_UNIT>(aGm, bGm, system, row, lda, ldb, solutionScale);
        const uint32_t updateStart = forward ? row + 1 : 0;
        const uint32_t updateEnd = forward ? order : row;
        UpdateSystemRows<SIDE_RIGHT, TRANS>(aGm, bGm, system, row, updateStart, updateEnd, lda, ldb, solutionScale);
        if constexpr (SIDE_RIGHT && TRANS != ACLBLAS_OP_N) {
            if (threadIdx.x == 0 && (alpha.real != 1.0f || alpha.imag != 0.0f)) {
                const uint64_t offset = BOffset<SIDE_RIGHT>(system, row, ldb);
                StoreComplex(bGm, offset, Multiply(alpha, LoadComplex(bGm, offset)));
            }
            asc_syncthreads();
        }
    }

    RestoreSolutionScale<SIDE_RIGHT, TRANS, DIAG_UNIT>(aGm, bGm, system, order, lda, ldb);
}

template <bool SIDE_RIGHT, bool UPLO_UPPER, uint32_t TRANS, bool DIAG_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(CTRSM_SIMT_MAX_THREADS) void CtrsmSimt(
    __gm__ const float* aGm, __gm__ float* bGm, uint32_t systemStart, uint32_t systemEnd, uint32_t order, uint64_t lda,
    uint64_t ldb, ComplexValue alpha)
{
    for (uint32_t system = systemStart; system < systemEnd; ++system) {
        SolveSystem<SIDE_RIGHT, UPLO_UPPER, TRANS, DIAG_UNIT>(aGm, bGm, system, order, lda, ldb, alpha);
    }
}

template <bool SIDE_RIGHT, bool UPLO_UPPER, uint32_t TRANS>
__aicore__ inline void DispatchDiag(
    const CtrsmTilingData& tiling, __gm__ const float* aGm, __gm__ float* bGm, uint32_t systemStart, uint32_t systemEnd,
    uint32_t order)
{
    const dim3 threads = {tiling.numThreads, 1, 1};
    const ComplexValue alpha = {tiling.alphaReal, tiling.alphaImag};
    if (tiling.diag == ACLBLAS_UNIT) {
        asc_vf_call<CtrsmSimt<SIDE_RIGHT, UPLO_UPPER, TRANS, true>>(
            threads, aGm, bGm, systemStart, systemEnd, order, tiling.lda, tiling.ldb, alpha);
    } else {
        asc_vf_call<CtrsmSimt<SIDE_RIGHT, UPLO_UPPER, TRANS, false>>(
            threads, aGm, bGm, systemStart, systemEnd, order, tiling.lda, tiling.ldb, alpha);
    }
}

template <bool SIDE_RIGHT, bool UPLO_UPPER>
__aicore__ inline void DispatchTrans(
    const CtrsmTilingData& tiling, __gm__ const float* aGm, __gm__ float* bGm, uint32_t systemStart, uint32_t systemEnd,
    uint32_t order)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        DispatchDiag<SIDE_RIGHT, UPLO_UPPER, ACLBLAS_OP_N>(tiling, aGm, bGm, systemStart, systemEnd, order);
    } else if (tiling.trans == ACLBLAS_OP_T) {
        DispatchDiag<SIDE_RIGHT, UPLO_UPPER, ACLBLAS_OP_T>(tiling, aGm, bGm, systemStart, systemEnd, order);
    } else {
        DispatchDiag<SIDE_RIGHT, UPLO_UPPER, ACLBLAS_OP_C>(tiling, aGm, bGm, systemStart, systemEnd, order);
    }
}

template <bool SIDE_RIGHT>
__aicore__ inline void DispatchUplo(
    const CtrsmTilingData& tiling, __gm__ const float* aGm, __gm__ float* bGm, uint32_t systemStart, uint32_t systemEnd,
    uint32_t order)
{
    if (tiling.uplo == ACLBLAS_UPPER) {
        DispatchTrans<SIDE_RIGHT, true>(tiling, aGm, bGm, systemStart, systemEnd, order);
    } else {
        DispatchTrans<SIDE_RIGHT, false>(tiling, aGm, bGm, systemStart, systemEnd, order);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CTRSM_AUX_THREADS) void CtrsmZero(
    __gm__ float* bGm, uint64_t logicalStart, uint64_t logicalEnd, uint32_t m, uint64_t ldb)
{
    for (uint64_t logical = logicalStart + threadIdx.x; logical < logicalEnd; logical += blockDim.x) {
        const uint64_t row = logical % m;
        const uint64_t col = logical / m;
        const ComplexValue zero = {0.0f, 0.0f};
        StoreComplex(bGm, col * ldb + row, zero);
    }
}

} // namespace

__aicore__ inline void RunCtrsmSimt(GM_ADDR a, GM_ADDR b, const CtrsmTilingData& tiling)
{
    const uint32_t coreId = AscendC::GetBlockIdx();
    const uint32_t systemStart =
        coreId * tiling.perCoreSystems + (coreId < tiling.coreRemainder ? coreId : tiling.coreRemainder);
    uint32_t systemEnd = systemStart + tiling.perCoreSystems + (coreId < tiling.coreRemainder ? 1 : 0);
    if (systemStart >= tiling.systemCount) {
        return;
    }
    if (systemEnd > tiling.systemCount) {
        systemEnd = tiling.systemCount;
    }

    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* bGm = reinterpret_cast<__gm__ float*>(b);
    if (tiling.side == ACLBLAS_SIDE_RIGHT) {
        DispatchUplo<true>(tiling, aGm, bGm, systemStart, systemEnd, tiling.n);
    } else {
        DispatchUplo<false>(tiling, aGm, bGm, systemStart, systemEnd, tiling.m);
    }
}

__global__ __aicore__ void ctrsm_kernel(GM_ADDR a, GM_ADDR b, const CtrsmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunCtrsmSimt(a, b, tiling);
}

__global__ __aicore__ void ctrsm_zero_kernel(GM_ADDR b, uint32_t m, uint32_t n, int32_t ldb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint64_t total = static_cast<uint64_t>(m) * n;
    const uint32_t numBlocks = AscendC::GetBlockNum();
    if (numBlocks == 0) {
        return;
    }
    const uint64_t perCore = (total + numBlocks - 1) / numBlocks;
    const uint64_t logicalStart = AscendC::GetBlockIdx() * perCore;
    uint64_t logicalEnd = logicalStart + perCore;
    if (logicalStart >= total) {
        return;
    }
    if (logicalEnd > total) {
        logicalEnd = total;
    }
    const dim3 threads = {CTRSM_AUX_THREADS, 1, 1};
    asc_vf_call<CtrsmZero>(
        threads, reinterpret_cast<__gm__ float*>(b), logicalStart, logicalEnd, m, static_cast<uint32_t>(ldb));
}

void ctrsm_kernel_do(uint8_t* a, uint8_t* b, const CtrsmTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctrsm_kernel<<<numBlocks, nullptr, stream>>>(a, b, tiling);
}

void ctrsm_zero_kernel_do(uint8_t* b, uint32_t m, uint32_t n, int32_t ldb, uint32_t numBlocks, void* stream)
{
    ctrsm_zero_kernel<<<numBlocks, nullptr, stream>>>(b, m, n, ldb);
}

namespace {
// Complete each pair before reusing the temporary event; do not reserve IDs used by TPipe.
template <AscendC::HardEvent EVENT>
__aicore__ inline void CtrsmSyncPipeline()
{
    const auto eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    AscendC::SetFlag<EVENT>(eventId);
    AscendC::WaitFlag<EVENT>(eventId);
}
} // namespace

namespace {
constexpr uint32_t CONVERT_TILE = CTRSM_CONVERT_TILE;
constexpr uint32_t CONVERT_STRIDE = 2 * CONVERT_TILE;
// One extra 32-byte block breaks the repeated bank mapping during transpose gathers.
constexpr uint32_t CONVERT_INPUT_STRIDE = CONVERT_STRIDE + 8;
// Exceptional and widely scaled input needs the compensated SIMT arithmetic.
// These are numerical guards, independent of shapes and official test cases.
constexpr float CTRSM_BLOCK_MAX = 0x1p30f;
constexpr float CTRSM_BLOCK_MIN = 0x1p-30f;
constexpr uint32_t CTRSM_BLOCK_MAX_BITS = 0x4e800000U;
constexpr uint32_t CTRSM_RISK_STRIDE = CTRSM_RISK_WORDS;
static_assert(CONVERT_TILE * (CONVERT_INPUT_STRIDE + CONVERT_STRIDE + 1) * sizeof(float) < 248U * 1024U);

__simd_callee__ inline void AccumulateMagnitude(
    Reg::RegTensor<uint32_t>& maximum, Reg::RegTensor<float>& real, Reg::RegTensor<float>& imag, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> absolute;
    Reg::Abs(absolute, real, mask);
    Reg::Max(maximum, maximum, reinterpret_cast<Reg::RegTensor<uint32_t>&>(absolute), mask);
    Reg::Abs(absolute, imag, mask);
    Reg::Max(maximum, maximum, reinterpret_cast<Reg::RegTensor<uint32_t>&>(absolute), mask);
}

__aicore__ inline bool NeedsCompensatedSolve(GM_ADDR aPacked, const CtrsmTilingData& t)
{
    GlobalTensor<uint32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(aPacked + t.riskOffset));
    for (uint32_t core = 0; core < t.prepareCores; ++core) {
        if (flags.GetValue(core * CTRSM_RISK_STRIDE) >= CTRSM_BLOCK_MAX_BITS) {
            return true;
        }
    }
    for (uint32_t tile = 0; tile < t.order / CONVERT_TILE; ++tile) {
        const uint32_t offset = (t.prepareCores + tile) * CTRSM_RISK_STRIDE;
        if (flags.GetValue(offset) != 0 || flags.GetValue(offset + 1) != 0) {
            return true;
        }
    }
    return false;
}

__simd_vf__ inline void ReduceMagnitude(__ubuf__ uint32_t* flags)
{
    Reg::RegTensor<uint32_t> values, result;
    uint32_t count = CONVERT_TILE;
    auto mask = Reg::UpdateMask<uint32_t>(count);
    Reg::LoadAlign(values, flags);
    Reg::ReduceMax(result, values, mask);
    Reg::StoreAlign(flags, result, mask);
}

// Diagonal tiles use guarded loads: ignored triangular entries and UNIT diagonals
// must not be read, even when their backing memory contains exceptional values.
__simt_vf__ __aicore__ LAUNCH_BOUND(CTRSM_SIMT_MAX_THREADS) void PrepareDiagonalTile(
    __gm__ const float* a, __gm__ float* out, uint32_t start, uint32_t rows, CtrsmTilingData t)
{
    bool risky = false;
    for (uint32_t index = threadIdx.x; index < rows * rows; index += blockDim.x) {
        const uint32_t row = start + index / rows;
        const uint32_t col = start + index % rows;
        float real = 0.0f;
        float imag = 0.0f;
        const bool referenced = t.effectiveUpper != 0 ? row <= col : row >= col;
        if (row == col && (t.diag == ACLBLAS_UNIT || row >= t.originalOrder)) {
            real = 1.0f;
        } else if (referenced && row < t.originalOrder && col < t.originalOrder) {
            const bool transpose = (t.trans != ACLBLAS_OP_N) != (t.side == ACLBLAS_SIDE_RIGHT);
            const uint64_t source =
                transpose ? static_cast<uint64_t>(row) * t.lda + col : static_cast<uint64_t>(col) * t.lda + row;
            real = a[2 * source];
            imag = a[2 * source + 1];
            if (t.trans == ACLBLAS_OP_C) {
                imag = -imag;
            }
        }
        const uint64_t target = static_cast<uint64_t>(row) * 2 * t.order + 2 * col;
        out[target] = real;
        out[target + 1] = imag;
        const float magnitude = fmaxf(fabsf(real), fabsf(imag));
        if (!isfinite(real) || !isfinite(imag) || magnitude >= CTRSM_BLOCK_MAX ||
            (row == col && magnitude < CTRSM_BLOCK_MIN)) {
            risky = true;
        }
    }
    const int32_t warpRisk = asc_any(risky);
    if (threadIdx.x % CTRSM_SIMT_WARP_SIZE == 0) {
        auto* flags = reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(out) + t.riskOffset);
        flags[(t.prepareCores + start / CONVERT_TILE) * CTRSM_RISK_STRIDE + threadIdx.x / CTRSM_SIMT_WARP_SIZE] =
            warpRisk;
    }
}

__simd_callee__ inline void MaskTriangularInput(
    Reg::RegTensor<float>& real, Reg::RegTensor<float>& imag, Reg::RegTensor<uint32_t>& columns, uint32_t absoluteRow,
    uint32_t order, uint32_t upper, uint32_t unit, uint32_t conjugate, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> zero, one;
    Reg::Duplicate(zero, 0.0f);
    Reg::Duplicate(one, 1.0f);
    Reg::MaskReg referenced, valid, diagonal;
    if (upper != 0) {
        Reg::Compares<uint32_t, CMPMODE::GE>(referenced, columns, absoluteRow, mask);
    } else {
        Reg::Compares<uint32_t, CMPMODE::LE>(referenced, columns, absoluteRow, mask);
    }
    Reg::Compares<uint32_t, CMPMODE::LT>(valid, columns, order, mask);
    Reg::Select(real, real, zero, referenced);
    Reg::Select(imag, imag, zero, referenced);
    Reg::Select(real, real, zero, valid);
    Reg::Select(imag, imag, zero, valid);
    if (absoluteRow >= order) {
        Reg::Duplicate(real, 0.0f);
        Reg::Duplicate(imag, 0.0f);
    }
    if (unit != 0 || absoluteRow >= order) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(diagonal, columns, absoluteRow, mask);
        Reg::Select(real, one, real, diagonal);
        Reg::Select(imag, zero, imag, diagonal);
    }
    if (conjugate != 0) {
        Reg::Muls(imag, imag, -1.0f, mask);
    }
}

template <bool MATRIX_A>
__simd_vf__ inline void ConvertInputTile(
    __ubuf__ float* src, __ubuf__ float* dst, uint32_t rows, uint32_t rowStart, uint32_t colStart, uint32_t order,
    uint32_t transpose, uint32_t upper, uint32_t unit, uint32_t conjugate, float alphaReal, float alphaImag,
    __ubuf__ uint32_t* flags)
{
    Reg::RegTensor<uint32_t> lane, offsets, columns;
    Reg::RegTensor<float> real, imag, tmp, scaledReal, scaledImag, first, second;
    uint32_t count = CONVERT_TILE;
    auto mask = Reg::UpdateMask<float>(count);
    Reg::RegTensor<uint32_t> maximum;
    Reg::LoadAlign(maximum, flags);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, transpose != 0 ? CONVERT_INPUT_STRIDE : uint32_t(2), mask);
    Reg::Adds(columns, lane, colStart, mask);
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t base = transpose != 0 ? 2 * row : CONVERT_INPUT_STRIDE * row;
        Reg::Gather(real, src + base, offsets, mask);
        Reg::Gather(imag, src + base + 1, offsets, mask);
        if constexpr (MATRIX_A) {
            MaskTriangularInput(real, imag, columns, rowStart + row, order, upper, unit, conjugate, mask);
            AccumulateMagnitude(maximum, real, imag, mask);
            Reg::Interleave(first, second, real, imag);
            Reg::StoreAlign(dst + row * CONVERT_STRIDE, first, mask);
            Reg::StoreAlign(dst + row * CONVERT_STRIDE + CONVERT_TILE, second, mask);
        } else {
            AccumulateMagnitude(maximum, real, imag, mask);
            Reg::Muls(scaledReal, real, alphaReal, mask);
            Reg::Muls(tmp, imag, alphaImag, mask);
            Reg::Sub(scaledReal, scaledReal, tmp, mask);
            Reg::Muls(scaledImag, imag, alphaReal, mask);
            Reg::Muls(tmp, real, alphaImag, mask);
            Reg::Add(scaledImag, scaledImag, tmp, mask);
            Reg::StoreAlign(dst + row * CONVERT_STRIDE, scaledReal, mask);
            Reg::StoreAlign(dst + row * CONVERT_STRIDE + CONVERT_TILE, scaledImag, mask);
        }
    }
    Reg::StoreAlign(flags, maximum, mask);
}

__simd_vf__ inline void ConvertOutputTile(__ubuf__ float* src, __ubuf__ float* dst, uint32_t rows, uint32_t transpose)
{
    Reg::RegTensor<uint32_t> lane, offsets;
    Reg::RegTensor<float> real, imag, first, second;
    uint32_t count = CONVERT_TILE;
    auto mask = Reg::UpdateMask<float>(count);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    Reg::Muls(offsets, lane, transpose != 0 ? CONVERT_INPUT_STRIDE : uint32_t(1), mask);
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t base = transpose != 0 ? row : CONVERT_INPUT_STRIDE * row;
        Reg::Gather(real, src + base, offsets, mask);
        Reg::Gather(imag, src + base + CONVERT_TILE, offsets, mask);
        Reg::Interleave(first, second, real, imag);
        Reg::StoreAlign(dst + row * CONVERT_STRIDE, first, mask);
        Reg::StoreAlign(dst + row * CONVERT_STRIDE + CONVERT_TILE, second, mask);
    }
}

__aicore__ inline uint32_t TileCount(uint32_t remaining) { return remaining < CONVERT_TILE ? remaining : CONVERT_TILE; }

__aicore__ inline void ReadComplexTile(
    LocalTensor<float>& dst, GlobalTensor<float>& src, uint64_t start, uint32_t rows, uint32_t cols, uint64_t stride)
{
    // The preceding conversion must finish reading this input buffer before clearing it.
    PipeBarrier<PIPE_V>();
    Duplicate(dst, 0.0f, CONVERT_TILE * CONVERT_INPUT_STRIDE);
    if (rows != 0 && cols != 0) {
        // MTE2 overwrites the valid region after Vector has initialized its padding.
        CtrsmSyncPipeline<HardEvent::V_MTE2>();
        const uint32_t aligned = (2 * cols + 7) / 8 * 8;
        DataCopyExtParams params(
            static_cast<uint16_t>(rows), 2 * cols * sizeof(float), 2 * (stride - cols) * sizeof(float),
            (CONVERT_INPUT_STRIDE - aligned) / 8, 0);
        // Normal mode: blockLen + rightPadding + dstStride * 32 gives the UB row pitch.
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(aligned - 2 * cols), 0.0f};
        DataCopyPad(dst, src[start], params, pad);
        CtrsmSyncPipeline<HardEvent::MTE2_V>();
    } else {
        // Unreferenced tiles consume only the Vector-initialized zeros.
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void WriteTile(
    GlobalTensor<float>& dst, uint64_t start, LocalTensor<float>& src, uint32_t rows, uint32_t floats, uint64_t stride)
{
    const uint32_t aligned = (floats + 7) / 8 * 8;
    // Normal UBToGM consumes an aligned UB block but writes only blockLen bytes to GM.
    // The implicit padding is discarded, preserving the caller's lda/ldb padding.
    DataCopyExtParams params(
        static_cast<uint16_t>(rows), floats * sizeof(float), (CONVERT_STRIDE - aligned) / 8,
        (stride - floats) * sizeof(float), 0);
    DataCopyPad(dst[start], src, params);
    // Do not overwrite the output UB while MTE3 is still reading it.
    CtrsmSyncPipeline<HardEvent::MTE3_V>();
}

__aicore__ inline void StoreConvertedInput(
    LocalTensor<float>& input, LocalTensor<float>& output, GlobalTensor<float>& target, const CtrsmTilingData& t,
    bool matrixA, uint32_t rows, uint32_t cols, uint32_t rowStart, uint32_t colStart, bool transpose,
    __ubuf__ uint32_t* flags)
{
    if (matrixA) {
        asc_vf_call<ConvertInputTile<true>>(
            reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), rows, rowStart, colStart, t.originalOrder,
            transpose, t.effectiveUpper, t.diag == ACLBLAS_UNIT, t.trans == ACLBLAS_OP_C, t.alphaReal, t.alphaImag,
            flags);
        CtrsmSyncPipeline<HardEvent::V_MTE3>();
        WriteTile(
            target, static_cast<uint64_t>(rowStart) * 2 * t.order + 2 * colStart, output, rows, 2 * cols, 2 * t.order);
    } else {
        asc_vf_call<ConvertInputTile<false>>(
            reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), rows, rowStart, colStart, t.originalOrder,
            transpose, 0, 0, 0, t.alphaReal, t.alphaImag, flags);
        CtrsmSyncPipeline<HardEvent::V_MTE3>();
        const uint64_t packedStride = t.combinedCols;
        const uint64_t dest = static_cast<uint64_t>(rowStart) * packedStride + 2 * colStart;
        WriteTile(target, dest, output, rows, cols, packedStride);
        auto imag = output[CONVERT_TILE];
        WriteTile(target, dest + t.splitCols, imag, rows, cols, packedStride);
    }
}

struct CtrsmConvertBuffers {
    LocalTensor<float> input;
    LocalTensor<float> output;
    GlobalTensor<float> sourceA;
    GlobalTensor<float> sourceB;
    GlobalTensor<float> targetA;
    GlobalTensor<float> targetB;
};

__aicore__ inline void PrepareInputTask(
    CtrsmConvertBuffers& buffers, const CtrsmTilingData& t, uint64_t task, __ubuf__ uint32_t* flags)
{
    const uint32_t rowTiles = t.order / CONVERT_TILE;
    const uint32_t rhsTiles = (t.rhsCount + CONVERT_TILE - 1) / CONVERT_TILE;
    const uint64_t aTiles = static_cast<uint64_t>(rowTiles) * rowTiles;
    const bool matrixA = task < aTiles;
    const uint64_t index = matrixA ? task : task - aTiles;
    const uint32_t colsTiles = matrixA ? rowTiles : rhsTiles;
    const uint32_t rowStart = index / colsTiles * CONVERT_TILE;
    const uint32_t colStart = index % colsTiles * CONVERT_TILE;
    const uint32_t rows = TileCount(t.order - rowStart);
    const uint32_t cols = TileCount((matrixA ? t.order : t.rhsCount) - colStart);
    if (matrixA && rowStart == colStart) {
        asc_vf_call<PrepareDiagonalTile>(
            dim3{CTRSM_SIMT_MAX_THREADS, 1, 1}, buffers.sourceA.GetPhyAddr(),
            const_cast<__gm__ float*>(buffers.targetA.GetPhyAddr()), rowStart, rows, t);
        return;
    }
    const uint32_t validRows = rowStart < t.originalOrder ? TileCount(t.originalOrder - rowStart) : 0;
    const uint32_t validCols =
        matrixA ? (colStart < t.originalOrder ? TileCount(t.originalOrder - colStart) : 0) : cols;
    const bool transpose =
        matrixA ? !((t.trans != ACLBLAS_OP_N) != (t.side == ACLBLAS_SIDE_RIGHT)) : t.side == ACLBLAS_SIDE_LEFT;
    const uint64_t stride = matrixA ? t.lda : t.ldb;
    const uint64_t start = transpose ? static_cast<uint64_t>(colStart) * stride + rowStart :
                                       static_cast<uint64_t>(rowStart) * stride + colStart;
    auto& source = matrixA ? buffers.sourceA : buffers.sourceB;
    const bool referenced = !matrixA || (t.effectiveUpper != 0 ? rowStart < colStart : rowStart > colStart);
    ReadComplexTile(
        buffers.input, source, 2 * start, referenced ? (transpose ? validCols : validRows) : 0,
        transpose ? validRows : validCols, stride);
    StoreConvertedInput(
        buffers.input, buffers.output, matrixA ? buffers.targetA : buffers.targetB, t, matrixA, rows, cols, rowStart,
        colStart, transpose, flags);
}

__aicore__ inline void PrepareBlockedDma(
    GM_ADDR a, GM_ADDR b, GM_ADDR aPacked, GM_ADDR bPacked, CtrsmTilingData t, TPipe* pipe)
{
    TBuf<QuePosition::VECCALC> inputBuffer, outputBuffer;
    pipe->InitBuffer(inputBuffer, CONVERT_TILE * CONVERT_INPUT_STRIDE * sizeof(float));
    pipe->InitBuffer(outputBuffer, CONVERT_TILE * CONVERT_STRIDE * sizeof(float));
    CtrsmConvertBuffers buffers;
    buffers.input = inputBuffer.Get<float>();
    buffers.output = outputBuffer.Get<float>();
    TBuf<QuePosition::VECCALC> riskBuffer;
    pipe->InitBuffer(riskBuffer, CONVERT_TILE * sizeof(uint32_t));
    auto risk = riskBuffer.Get<uint32_t>();
    const bool ordinaryAlpha = t.alphaReal < CTRSM_BLOCK_MAX && t.alphaReal > -CTRSM_BLOCK_MAX &&
                               t.alphaImag < CTRSM_BLOCK_MAX && t.alphaImag > -CTRSM_BLOCK_MAX;
    Duplicate(risk, ordinaryAlpha ? 0U : CTRSM_BLOCK_MAX_BITS, CONVERT_TILE);
    PipeBarrier<PIPE_V>();
    auto* flags = reinterpret_cast<__ubuf__ uint32_t*>(risk.GetPhyAddr());
    buffers.sourceA.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    buffers.sourceB.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    buffers.targetA.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aPacked));
    buffers.targetB.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bPacked));
    const uint32_t rowTiles = (t.order + CONVERT_TILE - 1) / CONVERT_TILE;
    const uint32_t rhsTiles = (t.rhsCount + CONVERT_TILE - 1) / CONVERT_TILE;
    const uint64_t aTiles = static_cast<uint64_t>(rowTiles) * rowTiles;
    const uint64_t tasks = aTiles + static_cast<uint64_t>(rowTiles) * rhsTiles;
    for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        PrepareInputTask(buffers, t, task, flags);
    }
    asc_vf_call<ReduceMagnitude>(flags);
    CtrsmSyncPipeline<HardEvent::V_MTE3>();
    GlobalTensor<uint32_t> riskGlobal;
    riskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(aPacked + t.riskOffset));
    DataCopy(riskGlobal[GetBlockIdx() * CTRSM_RISK_STRIDE], risk, CTRSM_RISK_STRIDE);
}

__aicore__ inline void FinishBlockedDma(GM_ADDR bPacked, GM_ADDR b, CtrsmTilingData t, TPipe* pipe)
{
    TBuf<QuePosition::VECCALC> inputBuffer, outputBuffer;
    pipe->InitBuffer(inputBuffer, CONVERT_TILE * CONVERT_INPUT_STRIDE * sizeof(float));
    pipe->InitBuffer(outputBuffer, CONVERT_TILE * CONVERT_STRIDE * sizeof(float));
    auto input = inputBuffer.Get<float>();
    auto output = outputBuffer.Get<float>();
    GlobalTensor<float> source, target;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bPacked));
    target.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    const uint32_t rowTiles = (t.originalOrder + CONVERT_TILE - 1) / CONVERT_TILE;
    const uint32_t rhsTiles = (t.rhsCount + CONVERT_TILE - 1) / CONVERT_TILE;
    const uint64_t stride = t.combinedCols;
    const uint64_t tasks = static_cast<uint64_t>(rowTiles) * rhsTiles;
    for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        const uint32_t rowStart = task / rhsTiles * CONVERT_TILE;
        const uint32_t colStart = task % rhsTiles * CONVERT_TILE;
        const uint32_t rows = TileCount(t.originalOrder - rowStart);
        const uint32_t cols = TileCount(t.rhsCount - colStart);
        const uint64_t start = static_cast<uint64_t>(rowStart) * stride + 2 * colStart;
        const uint32_t alignedCols = (cols + 7) / 8 * 8;
        DataCopyExtParams params(
            static_cast<uint16_t>(rows), cols * sizeof(float), (stride - cols) * sizeof(float),
            (CONVERT_INPUT_STRIDE - alignedCols) / 8, 0);
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(alignedCols - cols), 0.0f};
        // The previous tile's Vector gather must release input UB before the next load.
        CtrsmSyncPipeline<HardEvent::V_MTE2>();
        DataCopyPad(input, source[start], params, pad);
        DataCopyPad(input[CONVERT_TILE], source[start + t.splitCols], params, pad);
        CtrsmSyncPipeline<HardEvent::MTE2_V>();
        const bool transpose = t.side == ACLBLAS_SIDE_LEFT;
        asc_vf_call<ConvertOutputTile>(
            reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(output.GetPhyAddr()), transpose ? cols : rows, transpose);
        CtrsmSyncPipeline<HardEvent::V_MTE3>();
        const uint64_t dest = transpose ? static_cast<uint64_t>(colStart) * t.ldb + rowStart :
                                          static_cast<uint64_t>(rowStart) * t.ldb + colStart;
        WriteTile(target, 2 * dest, output, transpose ? cols : rows, 2 * (transpose ? rows : cols), 2 * t.ldb);
    }
}
} // namespace

__global__ __aicore__ void ctrsm_block_prepare_dma_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR aPacked, GM_ADDR bPacked, const CtrsmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    PrepareBlockedDma(a, b, aPacked, bPacked, tiling, &pipe);
}

__global__ __aicore__ void ctrsm_block_finish_dma_kernel(
    GM_ADDR a, GM_ADDR aPacked, GM_ADDR bPacked, GM_ADDR b, const CtrsmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (NeedsCompensatedSolve(aPacked, tiling)) {
        RunCtrsmSimt(a, b, tiling);
        return;
    }
    TPipe pipe;
    FinishBlockedDma(bPacked, b, tiling, &pipe);
}

void ctrsm_block_prepare_dma_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* aPacked, uint8_t* bPacked, const CtrsmTilingData& tiling, uint32_t blocks,
    void* stream)
{
    ctrsm_block_prepare_dma_kernel<<<blocks, nullptr, stream>>>(a, b, aPacked, bPacked, tiling);
}

void ctrsm_block_finish_dma_kernel_do(
    uint8_t* a, uint8_t* aPacked, uint8_t* bPacked, uint8_t* b, const CtrsmTilingData& tiling, uint32_t blocks,
    void* stream)
{
    ctrsm_block_finish_dma_kernel<<<blocks, nullptr, stream>>>(a, aPacked, bPacked, b, tiling);
}

namespace {

using namespace matmul;

constexpr uint32_t CTRSM_MIX_PANEL = CTRSM_PANEL_SIZE;
constexpr uint32_t CTRSM_MIX_MAX_LOCAL_COLS = CTRSM_SPLIT_COLS;
constexpr int32_t CTRSM_MIX_FLAG_PANEL = 6;
constexpr int32_t CTRSM_MIX_FLAG_UPDATE = 7;
constexpr int32_t CTRSM_MIX_FLAG_DUAL_AIV = 8;

using CtrsmMatmulType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using CtrsmMatmul = Matmul<CtrsmMatmulType, CtrsmMatmulType, CtrsmMatmulType, CtrsmMatmulType>;

__aicore__ inline void LoadPanelB(
    GlobalTensor<float>& bCombined, LocalTensor<float>& panelB, uint32_t panelStart, uint32_t panelSize,
    uint64_t workspaceCol, uint32_t colCount, uint32_t colAligned, uint32_t splitCols, uint64_t combinedCols)
{
    // Each real/imag block is explicitly padded before adding the other half-row gap.
    const uint32_t destinationGap = colAligned;
    DataCopyExtParams params(
        static_cast<uint16_t>(panelSize), colCount * sizeof(float), (combinedCols - colCount) * sizeof(float),
        destinationGap * sizeof(float) / 32, 0);
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(colAligned - colCount), 0.0f};
    const uint64_t offset = static_cast<uint64_t>(panelStart) * combinedCols + workspaceCol;
    DataCopyPad(panelB, bCombined[offset], params, pad);
    DataCopyPad(panelB[colAligned], bCombined[offset + splitCols], params, pad);
}

__aicore__ inline void LoadPanelA(
    GlobalTensor<float>& aPacked, LocalTensor<float>& panelA, uint32_t panelStart, uint32_t panelSize, uint32_t order)
{
    const uint32_t rowFloats = 2 * panelSize;
    DataCopyExtParams params(
        static_cast<uint16_t>(panelSize), rowFloats * sizeof(float), (2 * order - rowFloats) * sizeof(float), 0, 0);
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
    const uint64_t offset = static_cast<uint64_t>(panelStart) * 2 * order + 2 * panelStart;
    DataCopyPad(panelA, aPacked[offset], params, noPad);
}

__simd_callee__ inline void DividePanelRow(
    __ubuf__ float* diagonal, Reg::RegTensor<float>& real, Reg::RegTensor<float>& imag, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> ar, ai, denominator, x, y, z;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ar, diagonal);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ai, diagonal + 1);
    Reg::Mul(x, ar, ar, mask);
    Reg::Mul(y, ai, ai, mask);
    Reg::Add(denominator, x, y, mask);
    Reg::Mul(x, real, ar, mask);
    Reg::Mul(y, imag, ai, mask);
    Reg::Add(z, x, y, mask);
    Reg::Mul(x, imag, ar, mask);
    Reg::Mul(y, real, ai, mask);
    Reg::Sub(x, x, y, mask);
    Reg::Div(real, z, denominator, mask);
    Reg::Div(imag, x, denominator, mask);
}

template <bool FORWARD>
__simd_vf__ inline void SolvePanelVector(
    __ubuf__ float* a, __ubuf__ float* b, uint32_t panelSize, uint32_t cols, uint32_t nonUnit)
{
    Reg::RegTensor<float> real, imag, ar, ai, x, y, z, targetReal, targetImag;
    uint32_t count = cols;
    auto mask = Reg::UpdateMask<float>(count);
    for (uint16_t step = 0; step < panelSize; ++step) {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        uint32_t row = FORWARD ? step : panelSize - 1 - step;
        uint32_t offset = row * 2 * cols;
        Reg::LoadAlign(real, b + offset);
        Reg::LoadAlign(imag, b + offset + cols);
        if (nonUnit != 0) {
            uint32_t diagonal = row * 2 * panelSize + 2 * row;
            DividePanelRow(a + diagonal, real, imag, mask);
            Reg::StoreAlign(b + offset, real, mask);
            Reg::StoreAlign(b + offset + cols, imag, mask);
        }
        uint32_t first = FORWARD ? row + 1 : 0;
        uint32_t last = FORWARD ? panelSize : row;
        for (uint16_t dependency = first; dependency < last; ++dependency) {
            uint32_t coefficient = dependency * 2 * panelSize + 2 * row;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ar, a + coefficient);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ai, a + coefficient + 1);
            uint32_t target = dependency * 2 * cols;
            Reg::LoadAlign(targetReal, b + target);
            Reg::LoadAlign(targetImag, b + target + cols);
            Reg::Mul(x, ar, real, mask);
            Reg::Mul(y, ai, imag, mask);
            Reg::Sub(z, x, y, mask);
            Reg::Sub(targetReal, targetReal, z, mask);
            Reg::Mul(x, ar, imag, mask);
            Reg::Mul(y, ai, real, mask);
            Reg::Add(z, x, y, mask);
            Reg::Sub(targetImag, targetImag, z, mask);
            Reg::StoreAlign(b + target, targetReal, mask);
            Reg::StoreAlign(b + target + cols, targetImag, mask);
        }
    }
}

__aicore__ inline void StorePanelAndPackedX(
    GlobalTensor<float>& bCombined, GlobalTensor<float>& xCombined, LocalTensor<float>& panelB,
    LocalTensor<float>& negative, uint32_t panelStart, uint32_t panelSize, uint64_t globalCol, uint32_t localCol,
    uint32_t colCount, uint32_t colAligned, uint64_t combinedCols, uint32_t splitCols, uint32_t xPanelOffset)
{
    Muls(negative, panelB, -1.0f, panelSize * 2 * colAligned);
    CtrsmSyncPipeline<HardEvent::V_MTE3>();
    const uint32_t sourceGap = colAligned / 8;
    DataCopyExtParams bParams(
        static_cast<uint16_t>(panelSize), colCount * sizeof(float), sourceGap,
        (combinedCols - colCount) * sizeof(float), 0);
    const uint64_t bOffset = static_cast<uint64_t>(panelStart) * combinedCols + globalCol;
    DataCopyPad(bCombined[bOffset], panelB, bParams);
    DataCopyPad(bCombined[bOffset + splitCols], panelB[colAligned], bParams);

    DataCopyExtParams xParams(
        static_cast<uint16_t>(panelSize), colCount * sizeof(float), sourceGap,
        (4 * splitCols - colCount) * sizeof(float), 0);
    DataCopyPad(xCombined[xPanelOffset + localCol], negative, xParams);
    DataCopyPad(xCombined[xPanelOffset + splitCols + localCol], negative[colAligned], xParams);
    DataCopyPad(xCombined[xPanelOffset + 2 * splitCols + localCol], panelB[colAligned], xParams);
    DataCopyPad(xCombined[xPanelOffset + 3 * splitCols + localCol], negative, xParams);
    // Release panelB for the next MTE2 load and negative for the next Vector write.
    CtrsmSyncPipeline<HardEvent::MTE3_MTE2>();
    CtrsmSyncPipeline<HardEvent::MTE3_V>();
}

template <bool FORWARD>
__aicore__ inline void RunAivPanels(
    const CtrsmTilingData& tiling, LocalTensor<float>& panelB, LocalTensor<float>& panelA,
    LocalTensor<float>& packedReal, GlobalTensor<float>& bCombined, GlobalTensor<float>& aPacked,
    GlobalTensor<float>& xCombined, uint32_t split, uint32_t localCol, uint32_t colCount, uint32_t colAligned,
    uint64_t combinedCols)
{
    const uint32_t panelCount = tiling.order / tiling.panelSize;
    for (uint32_t step = 0; step < panelCount; ++step) {
        const uint32_t panelIndex = FORWARD ? step : panelCount - 1 - step;
        const uint32_t panelStart = panelIndex * tiling.panelSize;
        const uint32_t groupStartStep = step / tiling.groupPanels * tiling.groupPanels;
        const uint32_t groupSize =
            tiling.groupPanels < panelCount - groupStartStep ? tiling.groupPanels : panelCount - groupStartStep;
        const uint32_t localStep = step - groupStartStep;
        const uint32_t xSlot = FORWARD ? localStep : groupSize - 1 - localStep;
        const uint32_t xPanelOffset = xSlot * 4 * tiling.panelSize * tiling.splitCols;
        if (colCount != 0) {
            const uint64_t workspaceCol = static_cast<uint64_t>(split) * 2 * tiling.splitCols + localCol;
            LoadPanelA(aPacked, panelA, panelStart, tiling.panelSize, tiling.order);
            LoadPanelB(
                bCombined, panelB, panelStart, tiling.panelSize, workspaceCol, colCount, colAligned, tiling.splitCols,
                combinedCols);
            CtrsmSyncPipeline<HardEvent::MTE2_V>();
            asc_vf_call<SolvePanelVector<FORWARD>>(
                reinterpret_cast<__ubuf__ float*>(panelA.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(panelB.GetPhyAddr()), tiling.panelSize, colAligned,
                tiling.diag == ACLBLAS_NON_UNIT);
            // Muls consumes the solution written by the preceding Vector function.
            PipeBarrier<PIPE_V>();
            StorePanelAndPackedX(
                bCombined, xCombined, panelB, packedReal, panelStart, tiling.panelSize, workspaceCol, localCol,
                colCount, colAligned, combinedCols, tiling.splitCols, xPanelOffset);
        }
        const uint32_t trailRows = FORWARD ? tiling.order - panelStart - tiling.panelSize : panelStart;
        if (trailRows != 0) {
            CrossCoreSetFlag<1, PIPE_MTE3>(CTRSM_MIX_FLAG_DUAL_AIV);
            CrossCoreWaitFlag<1, PIPE_MTE3>(CTRSM_MIX_FLAG_DUAL_AIV);
            CrossCoreSetFlag<2, PIPE_MTE3>(CTRSM_MIX_FLAG_PANEL);
            CrossCoreWaitFlag<2, PIPE_MTE2>(CTRSM_MIX_FLAG_UPDATE);
            SetFlag<HardEvent::MTE2_S>(1);
            WaitFlag<HardEvent::MTE2_S>(1);
        }
    }
}

template <bool FORWARD>
__aicore__ inline void RunAiv(
    __gm__ float* aPackedPtr, __gm__ float* bCombinedPtr, __gm__ float* xCombinedPtr, CtrsmTilingData tiling,
    TPipe* pipe)
{
    const uint32_t aivIndex = GetBlockIdx();
    const uint32_t split = aivIndex / 2;
    const uint32_t aivHalf = aivIndex % 2;
    const uint32_t splitStart = split * tiling.splitCols;
    if (split >= tiling.splitCount || splitStart >= tiling.rhsCount) {
        return;
    }
    const uint32_t splitCount =
        tiling.splitCols < tiling.rhsCount - splitStart ? tiling.splitCols : tiling.rhsCount - splitStart;
    const uint32_t localCol = aivHalf * CTRSM_MIX_MAX_LOCAL_COLS;
    const uint32_t remainingCols = localCol < splitCount ? splitCount - localCol : 0;
    const uint32_t colCount = remainingCols < CTRSM_MIX_MAX_LOCAL_COLS ? remainingCols : CTRSM_MIX_MAX_LOCAL_COLS;
    const uint32_t colAligned = ((colCount + 7) / 8) * 8;
    TBuf<QuePosition::VECCALC> panelBuffer;
    TBuf<QuePosition::VECCALC> panelABuffer;
    TBuf<QuePosition::VECCALC> packedRealBuffer;
    pipe->InitBuffer(panelBuffer, CTRSM_MIX_PANEL * 2 * CTRSM_MIX_MAX_LOCAL_COLS * sizeof(float));
    pipe->InitBuffer(panelABuffer, CTRSM_MIX_PANEL * 2 * CTRSM_MIX_PANEL * sizeof(float));
    pipe->InitBuffer(packedRealBuffer, 2 * CTRSM_MIX_PANEL * CTRSM_MIX_MAX_LOCAL_COLS * sizeof(float));
    LocalTensor<float> panelB = panelBuffer.Get<float>();
    LocalTensor<float> panelA = panelABuffer.Get<float>();
    LocalTensor<float> packedReal = packedRealBuffer.Get<float>();

    GlobalTensor<float> bCombined;
    GlobalTensor<float> aPacked;
    GlobalTensor<float> xCombined;
    const uint64_t combinedCols = tiling.combinedCols;
    bCombined.SetGlobalBuffer(bCombinedPtr, static_cast<uint64_t>(tiling.order) * combinedCols);
    aPacked.SetGlobalBuffer(aPackedPtr, static_cast<uint64_t>(tiling.order) * 2 * tiling.order);
    const uint64_t xSplitElements = static_cast<uint64_t>(tiling.groupPanels) * 4 * tiling.panelSize * tiling.splitCols;
    xCombined.SetGlobalBuffer(xCombinedPtr + split * xSplitElements, xSplitElements);

    RunAivPanels<FORWARD>(
        tiling, panelB, panelA, packedReal, bCombined, aPacked, xCombined, split, localCol, colCount, colAligned,
        combinedCols);
}

__aicore__ inline void RunHighLevelMatmul(
    CtrsmMatmul& mm, GlobalTensor<float> a, GlobalTensor<float> x, GlobalTensor<float> output, uint32_t m, uint32_t n,
    uint32_t k, uint64_t lda, uint64_t ldb, uint64_t ldc)
{
    mm.SetOrgShape(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(lda), static_cast<int>(ldb), static_cast<int>(ldc));
    mm.SetSingleShape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    mm.SetTensorA(a, false);
    mm.SetTensorB(x, false);
    mm.SetHF32(false, 0);
    mm.IterateAll(output, 1);
}

struct CtrsmAicUpdate {
    uint32_t updateStart;
    uint32_t updateRows;
    uint32_t solvedPanelStart;
    uint32_t solvedPanelCount;
    uint32_t xPanelSlot;
};

template <bool FORWARD>
__aicore__ inline CtrsmAicUpdate CalcAicUpdate(
    const CtrsmTilingData& tiling, uint32_t localStep, uint32_t groupSize, uint32_t panelStart, uint32_t groupStartStep)
{
    CtrsmAicUpdate update{};
    const bool groupLast = localStep + 1 == groupSize;
    if (!groupLast) {
        update.updateStart = FORWARD ? panelStart + tiling.panelSize : panelStart - tiling.panelSize;
        update.updateRows = tiling.panelSize;
        update.solvedPanelCount = localStep + 1;
        update.solvedPanelStart = FORWARD ? groupStartStep * tiling.panelSize : panelStart;
        update.xPanelSlot = FORWARD ? 0 : groupSize - update.solvedPanelCount;
    } else {
        update.solvedPanelCount = groupSize;
        update.solvedPanelStart = FORWARD ? groupStartStep * tiling.panelSize : panelStart;
        update.updateStart = FORWARD ? panelStart + tiling.panelSize : 0;
        update.updateRows = FORWARD ? tiling.order - update.updateStart : panelStart;
        update.xPanelSlot = 0;
    }

    return update;
}

template <bool FORWARD>
__aicore__ inline void RunAic(
    __gm__ float* aPackedPtr, __gm__ float* bCombinedPtr, __gm__ float* xCombinedPtr, CtrsmTilingData tiling,
    CtrsmMatmul& mm)
{
    const uint32_t split = GetBlockIdx();
    const uint32_t splitStart = split * tiling.splitCols;
    if (split >= tiling.splitCount || splitStart >= tiling.rhsCount) {
        return;
    }
    const uint64_t xSplitElements = static_cast<uint64_t>(tiling.groupPanels) * 4 * tiling.panelSize * tiling.splitCols;
    const uint64_t combinedCols = tiling.combinedCols;
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> bGlobal;
    GlobalTensor<float> xGlobal;
    aGlobal.SetGlobalBuffer(aPackedPtr, static_cast<uint64_t>(tiling.order) * 2 * tiling.order);
    bGlobal.SetGlobalBuffer(bCombinedPtr, static_cast<uint64_t>(tiling.order) * combinedCols);
    xGlobal.SetGlobalBuffer(xCombinedPtr + split * xSplitElements, xSplitElements);

    const uint32_t panelCount = tiling.order / tiling.panelSize;
    for (uint32_t groupStartStep = 0; groupStartStep < panelCount; groupStartStep += tiling.groupPanels) {
        const uint32_t groupSize =
            tiling.groupPanels < panelCount - groupStartStep ? tiling.groupPanels : panelCount - groupStartStep;
        for (uint32_t localStep = 0; localStep < groupSize; ++localStep) {
            const uint32_t step = groupStartStep + localStep;
            const uint32_t panelIndex = FORWARD ? step : panelCount - 1 - step;
            const uint32_t panelStart = panelIndex * tiling.panelSize;
            const uint32_t trailRows = FORWARD ? tiling.order - panelStart - tiling.panelSize : panelStart;
            if (trailRows == 0) {
                continue;
            }
            CrossCoreWaitFlag<2, PIPE_MTE2>(CTRSM_MIX_FLAG_PANEL);

            const auto update = CalcAicUpdate<FORWARD>(tiling, localStep, groupSize, panelStart, groupStartStep);
            if (update.updateRows != 0) {
                const uint32_t k = 2 * update.solvedPanelCount * tiling.panelSize;
                const uint64_t aOffset =
                    static_cast<uint64_t>(update.updateStart) * 2 * tiling.order + 2 * update.solvedPanelStart;
                const uint64_t bOffset =
                    static_cast<uint64_t>(update.updateStart) * combinedCols + split * 2 * tiling.splitCols;
                const uint64_t xOffset =
                    static_cast<uint64_t>(update.xPanelSlot) * 4 * tiling.panelSize * tiling.splitCols;
                RunHighLevelMatmul(
                    mm, aGlobal[aOffset], xGlobal[xOffset], bGlobal[bOffset], update.updateRows, 2 * tiling.splitCols,
                    k, 2 * tiling.order, 2 * tiling.splitCols, combinedCols);
            }
            CrossCoreSetFlag<2, PIPE_FIX>(CTRSM_MIX_FLAG_UPDATE);
        }
    }
}

} // namespace

__global__ __aicore__ void ctrsm_mix_block_kernel(
    GM_ADDR aPacked, GM_ADDR bCombined, GM_ADDR xCombined, GM_ADDR systemWorkspace, const CtrsmTilingData tiling,
    const TCubeTiling cubeTiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // Every AIC/AIV observes the same completed prepare-kernel flags before any cross-core wait.
    if (NeedsCompensatedSolve(aPacked, tiling)) {
        return;
    }
    TPipe pipe;
    if ASCEND_IS_AIC {
        CtrsmMatmul mm;
        REGIST_MATMUL_OBJ(&pipe, systemWorkspace, mm, &cubeTiling);
        if (tiling.effectiveUpper != 0) {
            RunAic<false>(
                reinterpret_cast<__gm__ float*>(aPacked), reinterpret_cast<__gm__ float*>(bCombined),
                reinterpret_cast<__gm__ float*>(xCombined), tiling, mm);
        } else {
            RunAic<true>(
                reinterpret_cast<__gm__ float*>(aPacked), reinterpret_cast<__gm__ float*>(bCombined),
                reinterpret_cast<__gm__ float*>(xCombined), tiling, mm);
        }
        mm.End();
    }
    if ASCEND_IS_AIV {
        if (tiling.effectiveUpper != 0) {
            RunAiv<false>(
                reinterpret_cast<__gm__ float*>(aPacked), reinterpret_cast<__gm__ float*>(bCombined),
                reinterpret_cast<__gm__ float*>(xCombined), tiling, &pipe);
        } else {
            RunAiv<true>(
                reinterpret_cast<__gm__ float*>(aPacked), reinterpret_cast<__gm__ float*>(bCombined),
                reinterpret_cast<__gm__ float*>(xCombined), tiling, &pipe);
        }
    }
}

void ctrsm_mix_block_kernel_do(
    uint8_t* aPacked, uint8_t* bCombined, uint8_t* xCombined, uint8_t* systemWorkspace, const CtrsmTilingData& tiling,
    const void* cubeTilingData, uint32_t numBlocks, void* stream)
{
    const auto& cubeTiling = *static_cast<const TCubeTiling*>(cubeTilingData);
    ctrsm_mix_block_kernel<<<numBlocks, nullptr, stream>>>(
        aPacked, bCombined, xCombined, systemWorkspace, tiling, cubeTiling);
}
