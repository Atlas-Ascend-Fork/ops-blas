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
 * \file ctbmv_fallback_kernel.cpp
 * \brief single-precision complex triangular band matrix-vector multiply SIMT fallback kernels for ascend950
 */

#include <cstdint>

#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "ctbmv_common.h"

// Keep separate multiply and add/subtract operations; do not contract to FMA.
#pragma clang fp contract(off)

__simt_callee__ __aicore__ __inline__ int64_t CtbmvVectorIndex(uint32_t idx, uint32_t n, int64_t incx)
{
    int64_t stride = incx;
    return (stride > 0) ? static_cast<int64_t>(idx) * stride :
                          static_cast<int64_t>(n - 1U - idx) * (-stride);
}

__simt_callee__ __aicore__ __inline__ void CtbmvComplexMac(
    float aRe, float aIm, float xRe, float xIm, float& accRe, float& accIm)
{
    // Products are thread-local; volatile would constrain register optimization.
    const float reProduct = aRe * xRe;
    const float imProduct = aIm * xIm;
    const float crossReProduct = aRe * xIm;
    const float crossImProduct = aIm * xRe;
    accRe = accRe + reProduct - imProduct;
    accIm = accIm + crossReProduct + crossImProduct;
}

__simt_callee__ __aicore__ __inline__ void CtbmvComplexMultiply(
    float lhsRe, float lhsIm, float rhsRe, float rhsIm, float& outRe, float& outIm)
{
    volatile float reProduct = lhsRe * rhsRe;
    volatile float imProduct = lhsIm * rhsIm;
    volatile float crossReProduct = lhsRe * rhsIm;
    volatile float crossImProduct = lhsIm * rhsRe;
    outRe = reProduct - imProduct;
    outIm = crossReProduct + crossImProduct;
}

__simt_callee__ __aicore__ __inline__ void CtbmvComplexAdd(
    float lhsRe, float lhsIm, float rhsRe, float rhsIm, float& outRe, float& outIm)
{
    outRe = lhsRe + rhsRe;
    outIm = lhsIm + rhsIm;
}

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ __inline__ int64_t CtbmvBandIndex(uint32_t row, uint32_t col, uint32_t k, uint32_t lda)
{
    int64_t rowValue = static_cast<int64_t>(row);
    int64_t colValue = static_cast<int64_t>(col);
    int64_t ldaValue = static_cast<int64_t>(lda);
    if constexpr (UPLO_IS_UPPER) {
        return static_cast<int64_t>(k) + rowValue - colValue + colValue * ldaValue;
    } else {
        return rowValue - colValue + colValue * ldaValue;
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N>
__simt_callee__ __aicore__ __inline__ void CtbmvColumnRange(
    uint32_t row, uint32_t n, uint32_t k, uint32_t& colStart, uint32_t& colEnd)
{
    if constexpr (TRANS_IS_N) {
        if constexpr (UPLO_IS_UPPER) {
            colStart = row;
            uint64_t end = static_cast<uint64_t>(row) + k + 1U;
            colEnd = (end < n) ? static_cast<uint32_t>(end) : n;
        } else {
            colStart = (row >= k) ? (row - k) : 0U;
            colEnd = row + 1U;
        }
    } else {
        if constexpr (UPLO_IS_UPPER) {
            colStart = (row >= k) ? (row - k) : 0U;
            colEnd = row + 1U;
        } else {
            colStart = row;
            uint64_t end = static_cast<uint64_t>(row) + k + 1U;
            colEnd = (end < n) ? static_cast<uint32_t>(end) : n;
        }
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ __inline__ void CtbmvLoadA(
    uint32_t row, uint32_t col, uint32_t k, uint32_t lda, __gm__ const aclblasComplex* aGm, float& aRe, float& aIm)
{
    if constexpr (DIAG_IS_UNIT) {
        if (row == col) {
            aRe = 1.0f;
            aIm = 0.0f;
            return;
        }
    }
    uint32_t aRow = TRANS_IS_N ? row : col;
    uint32_t aCol = TRANS_IS_N ? col : row;
    int64_t aIdx = CtbmvBandIndex<UPLO_IS_UPPER>(aRow, aCol, k, lda);
    aRe = aGm[aIdx].real;
    aIm = aGm[aIdx].imag;
}

// Loads the band coefficient for (row, col) when the caller has already worked
// out the band index, then applies the conjugate-transpose negation. The stored
// diagonal is never read: diag=UNIT supplies the implicit one instead.
template <bool DIAG_IS_UNIT, bool CONJ>
__simt_callee__ __aicore__ __inline__ void CtbmvLoadBandCoeff(
    uint32_t row, uint32_t col, int64_t aIdx, __gm__ const aclblasComplex* aGm, float& aRe, float& aIm)
{
    aRe = 0.0f;
    aIm = 0.0f;
    if constexpr (DIAG_IS_UNIT) {
        if (row == col) {
            aRe = 1.0f;
        } else {
            aRe = aGm[aIdx].real;
            aIm = aGm[aIdx].imag;
        }
    } else {
        aRe = aGm[aIdx].real;
        aIm = aGm[aIdx].imag;
    }
    if constexpr (CONJ) {
        aIm = -aIm;
    }
}

// Same coefficient load for the kernels that recompute the band index per
// column instead of carrying it in a register.
template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool DIAG_IS_UNIT, bool CONJ>
__simt_callee__ __aicore__ __inline__ void CtbmvLoadRowCoeff(
    uint32_t row, uint32_t col, uint32_t k, uint32_t lda, __gm__ const aclblasComplex* aGm, float& aRe, float& aIm)
{
    aRe = 0.0f;
    aIm = 0.0f;
    CtbmvLoadA<UPLO_IS_UPPER, TRANS_IS_N, DIAG_IS_UNIT>(row, col, k, lda, aGm, aRe, aIm);
    if constexpr (CONJ) {
        aIm = -aIm;
    }
}

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ __inline__ void CtbmvLoadNetlibA(
    uint32_t row, uint32_t col, uint32_t k, uint32_t lda, __gm__ const aclblasComplex* aGm, float& aRe, float& aIm)
{
    int64_t aIdx = CtbmvBandIndex<UPLO_IS_UPPER>(row, col, k, lda);
    aRe = aGm[aIdx].real;
    aIm = aGm[aIdx].imag;
}

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ __inline__ void CtbmvNetlibUpdate(
    uint32_t row, uint32_t col, uint32_t k, uint32_t lda, uint32_t n, int64_t incx, float tempRe, float tempIm,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    float aRe = 0.0F;
    float aIm = 0.0F;
    CtbmvLoadNetlibA<UPLO_IS_UPPER>(row, col, k, lda, aGm, aRe, aIm);
    float productRe = 0.0F;
    float productIm = 0.0F;
    CtbmvComplexMultiply(tempRe, tempIm, aRe, aIm, productRe, productIm);

    int64_t xIdx = CtbmvVectorIndex(row, n, incx);
    float updatedRe = 0.0F;
    float updatedIm = 0.0F;
    CtbmvComplexAdd(xGm[xIdx].real, xGm[xIdx].imag, productRe, productIm, updatedRe, updatedIm);
    xGm[xIdx].real = updatedRe;
    xGm[xIdx].imag = updatedIm;
}

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ __inline__ void CtbmvNetlibScale(
    uint32_t row, uint32_t k, uint32_t lda, bool conj,
    __gm__ const aclblasComplex* aGm, float& valueRe, float& valueIm)
{
    float aRe = 0.0F;
    float aIm = 0.0F;
    CtbmvLoadNetlibA<UPLO_IS_UPPER>(row, row, k, lda, aGm, aRe, aIm);
    if (conj) {
        aIm = -aIm;
    }
    float scaledRe = 0.0F;
    float scaledIm = 0.0F;
    CtbmvComplexMultiply(valueRe, valueIm, aRe, aIm, scaledRe, scaledIm);
    valueRe = scaledRe;
    valueIm = scaledIm;
}

// Folds one coefficient of the transposed / conjugate-transposed path into the
// running accumulator. This is the Netlib "TEMP = X(J)" scalar carried in a
// register: the whole matrix column is consumed before it is written back.
template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ __inline__ void CtbmvNetlibFold(
    uint32_t row, uint32_t col, uint32_t k, uint32_t lda, uint32_t n, int64_t incx, bool conj,
    float& tempRe, float& tempIm, __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm)
{
    float aRe = 0.0F;
    float aIm = 0.0F;
    CtbmvLoadNetlibA<UPLO_IS_UPPER>(row, col, k, lda, aGm, aRe, aIm);
    if (conj) {
        aIm = -aIm;
    }
    int64_t xIdx = CtbmvVectorIndex(row, n, incx);
    float termRe = 0.0F;
    float termIm = 0.0F;
    CtbmvComplexMultiply(aRe, aIm, xGm[xIdx].real, xGm[xIdx].imag, termRe, termIm);
    CtbmvComplexAdd(tempRe, tempIm, termRe, termIm, tempRe, tempIm);
}

// The small serial path exists in four flavours, one per (uplo, trans!=N)
// corner. Each is the reference BLAS loop nest verbatim: trans=N walks the
// columns and adds each scaled one into the band rows around it, trans!=N
// accumulates one output element before writing it back. Keeping them separate
// preserves the reference iteration order each one depends on.

__simt_callee__ __aicore__ __inline__ void CtbmvNetlibAxpyUpper(
    int nValue, int kValue, uint32_t k, uint32_t lda, uint32_t n, int64_t incx, bool nonUnit,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    for (int j = 0; j < nValue; ++j) {
        int64_t xjIdx = CtbmvVectorIndex(static_cast<uint32_t>(j), n, incx);
        float tempRe = xGm[xjIdx].real;
        float tempIm = xGm[xjIdx].imag;
        // No zero test on TEMP: the reference updates the column
        // unconditionally, so a zero x entry still multiplies A and 0*Inf has
        // to propagate as NaN.
        const int rowStart = (j > kValue) ? (j - kValue) : 0;
        for (int row = rowStart; row < j; ++row) {
            CtbmvNetlibUpdate<true>(static_cast<uint32_t>(row), static_cast<uint32_t>(j), k, lda, n, incx, tempRe,
                tempIm, aGm, xGm);
        }
        if (nonUnit) {
            CtbmvNetlibScale<true>(static_cast<uint32_t>(j), k, lda, false, aGm, tempRe, tempIm);
            xGm[xjIdx].real = tempRe;
            xGm[xjIdx].imag = tempIm;
        }
    }
}

__simt_callee__ __aicore__ __inline__ void CtbmvNetlibAxpyLower(
    int nValue, int kValue, uint32_t k, uint32_t lda, uint32_t n, int64_t incx, bool nonUnit,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    for (int j = nValue - 1; j >= 0; --j) {
        int64_t xjIdx = CtbmvVectorIndex(static_cast<uint32_t>(j), n, incx);
        float tempRe = xGm[xjIdx].real;
        float tempIm = xGm[xjIdx].imag;
        // No zero test on TEMP: the reference updates the column
        // unconditionally, so a zero x entry still multiplies A and 0*Inf has
        // to propagate as NaN.
        int rowEnd = j + kValue;
        rowEnd = (rowEnd < nValue) ? rowEnd : nValue - 1;
        for (int row = rowEnd; row > j; --row) {
            CtbmvNetlibUpdate<false>(static_cast<uint32_t>(row), static_cast<uint32_t>(j), k, lda, n, incx, tempRe,
                tempIm, aGm, xGm);
        }
        if (nonUnit) {
            CtbmvNetlibScale<false>(static_cast<uint32_t>(j), k, lda, false, aGm, tempRe, tempIm);
            xGm[xjIdx].real = tempRe;
            xGm[xjIdx].imag = tempIm;
        }
    }
}

__simt_callee__ __aicore__ __inline__ void CtbmvNetlibDotUpper(
    int nValue, int kValue, uint32_t k, uint32_t lda, uint32_t n, int64_t incx, bool conj, bool nonUnit,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    for (int j = nValue - 1; j >= 0; --j) {
        int64_t xjIdx = CtbmvVectorIndex(static_cast<uint32_t>(j), n, incx);
        float tempRe = xGm[xjIdx].real;
        float tempIm = xGm[xjIdx].imag;
        if (nonUnit) {
            CtbmvNetlibScale<true>(static_cast<uint32_t>(j), k, lda, conj, aGm, tempRe, tempIm);
        }
        const int rowStart = (j > kValue) ? (j - kValue) : 0;
        for (int row = j - 1; row >= rowStart; --row) {
            CtbmvNetlibFold<true>(static_cast<uint32_t>(row), static_cast<uint32_t>(j), k, lda, n, incx, conj, tempRe,
                tempIm, aGm, xGm);
        }
        xGm[xjIdx].real = tempRe;
        xGm[xjIdx].imag = tempIm;
    }
}

__simt_callee__ __aicore__ __inline__ void CtbmvNetlibDotLower(
    int nValue, int kValue, uint32_t k, uint32_t lda, uint32_t n, int64_t incx, bool conj, bool nonUnit,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    for (int j = 0; j < nValue; ++j) {
        int64_t xjIdx = CtbmvVectorIndex(static_cast<uint32_t>(j), n, incx);
        float tempRe = xGm[xjIdx].real;
        float tempIm = xGm[xjIdx].imag;
        if (nonUnit) {
            CtbmvNetlibScale<false>(static_cast<uint32_t>(j), k, lda, conj, aGm, tempRe, tempIm);
        }
        int rowEnd = j + kValue;
        rowEnd = (rowEnd < nValue) ? rowEnd : nValue - 1;
        for (int row = j + 1; row <= rowEnd; ++row) {
            CtbmvNetlibFold<false>(static_cast<uint32_t>(row), static_cast<uint32_t>(j), k, lda, n, incx, conj, tempRe,
                tempIm, aGm, xGm);
        }
        xGm[xjIdx].real = tempRe;
        xGm[xjIdx].imag = tempIm;
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtbmvComputeNetlibSmall(
    uint32_t n, uint32_t k, uint32_t lda, uint32_t trans, uint32_t diag, int64_t incx,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    const bool nonUnit = diag == ACLBLAS_NON_UNIT;
    const int nValue = static_cast<int>(n);
    const int kValue = static_cast<int>(k);

    if (trans == ACLBLAS_OP_N) {
        if constexpr (UPLO_IS_UPPER) {
            CtbmvNetlibAxpyUpper(nValue, kValue, k, lda, n, incx, nonUnit, aGm, xGm);
        } else {
            CtbmvNetlibAxpyLower(nValue, kValue, k, lda, n, incx, nonUnit, aGm, xGm);
        }
        return;
    }
    const bool conj = trans == ACLBLAS_OP_C;
    if constexpr (UPLO_IS_UPPER) {
        CtbmvNetlibDotUpper(nValue, kValue, k, lda, n, incx, conj, nonUnit, aGm, xGm);
    } else {
        CtbmvNetlibDotLower(nValue, kValue, k, lda, n, incx, conj, nonUnit, aGm, xGm);
    }
}


template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtbmvComputeGm(uint32_t n, uint32_t k,
    uint32_t lda, int64_t incx, __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm,
    __gm__ aclblasComplex* midOutGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        CtbmvColumnRange<UPLO_IS_UPPER, TRANS_IS_N>(row, n, k, colStart, colEnd);

        float accRe = 0.0f;
        float accIm = 0.0f;
        for (uint32_t col = colStart; col < colEnd; ++col) {
            float aRe = 0.0f;
            float aIm = 0.0f;
            CtbmvLoadRowCoeff<UPLO_IS_UPPER, TRANS_IS_N, DIAG_IS_UNIT, CONJ>(row, col, k, lda, aGm, aRe, aIm);
            int64_t xIdx = CtbmvVectorIndex(col, n, incx);
            float xRe = xGm[xIdx].real;
            float xIm = xGm[xIdx].imag;
            CtbmvComplexMac(aRe, aIm, xRe, xIm, accRe, accIm);
        }
        midOutGm[row].real = accRe;
        midOutGm[row].imag = accIm;
    }
}

__simt_callee__ __aicore__ __inline__ float CtbmvWarpReduceSum(float value)
{
    for (uint32_t offset = CTBMV_WARP_SIZE / 2U; offset > 0U; offset /= 2U) {
        value += asc_shfl_down(value, offset, CTBMV_WARP_SIZE);
    }
    return value;
}

// A wide row has enough independent complex products to amortize the warp
// shuffle reduction. Each lane reads every 32nd coefficient and lane zero
// writes the reduced result to workspace. x is already cached before this
// function is entered, so the in-place public API remains race-free.
template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ __inline__ void CtbmvComputeWideBandWarpUb(uint32_t n, uint32_t k, uint32_t lda,
    __gm__ const aclblasComplex* aGm, __ubuf__ const float* xUbRe, __ubuf__ const float* xUbIm,
    __gm__ aclblasComplex* midOutGm, uint32_t rowStart, uint32_t rowEnd, uint32_t xBase)
{
    const uint32_t laneId = threadIdx.x & (CTBMV_WARP_SIZE - 1U);
    const uint32_t warpId = threadIdx.x / CTBMV_WARP_SIZE;
    const uint32_t warpCount = blockDim.x / CTBMV_WARP_SIZE;

    for (uint32_t row = rowStart + warpId; row < rowEnd; row += warpCount) {
        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        CtbmvColumnRange<UPLO_IS_UPPER, TRANS_IS_N>(row, n, k, colStart, colEnd);

        float accRe = 0.0f;
        float accIm = 0.0f;
        // The band address is an affine function of the column, so hoist the
        // 64-bit index arithmetic out of the loop and advance a register by
        // the constant lane stride; recomputing the index per coefficient
        // makes the scalar unit the instruction-issue bottleneck.
        const uint32_t firstCol = colStart + laneId;
        int64_t aIdx = CtbmvBandIndex<UPLO_IS_UPPER>(
            TRANS_IS_N ? row : firstCol, TRANS_IS_N ? firstCol : row, k, lda);
        const int64_t aIdxStep = TRANS_IS_N ?
            static_cast<int64_t>(CTBMV_WARP_SIZE) * (static_cast<int64_t>(lda) - 1) :
            static_cast<int64_t>(CTBMV_WARP_SIZE);
        for (uint32_t col = firstCol; col < colEnd; col += CTBMV_WARP_SIZE, aIdx += aIdxStep) {
            float aRe = 0.0f;
            float aIm = 0.0f;
            CtbmvLoadBandCoeff<DIAG_IS_UNIT, CONJ>(row, col, aIdx, aGm, aRe, aIm);
            const uint32_t xi = col - xBase;
            CtbmvComplexMac(aRe, aIm, xUbRe[xi], xUbIm[xi], accRe, accIm);
        }

        accRe = CtbmvWarpReduceSum(accRe);
        accIm = CtbmvWarpReduceSum(accIm);
        if (laneId == 0U) {
            midOutGm[row].real = accRe;
            midOutGm[row].imag = accIm;
        }
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtbmvComputeUb(uint32_t n, uint32_t k,
    uint32_t lda, __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm, __gm__ aclblasComplex* midOutGm,
    uint32_t rowStart, uint32_t rowEnd, uint32_t xBase, uint32_t xLen)
{
    __ubuf__ float xUbRe[CTBMV_UB_X_ELEMS];
    __ubuf__ float xUbIm[CTBMV_UB_X_ELEMS];

    for (uint32_t i = threadIdx.x; i < xLen; i += blockDim.x) {
        int64_t xIdx = static_cast<int64_t>(xBase) + i;
        xUbRe[i] = xGm[xIdx].real;
        xUbIm[i] = xGm[xIdx].imag;
    }
    asc_syncthreads();

    if (k >= CTBMV_WIDE_BAND_MIN_K && blockDim.x >= CTBMV_WARP_SIZE) {
        CtbmvComputeWideBandWarpUb<UPLO_IS_UPPER, TRANS_IS_N, CONJ, DIAG_IS_UNIT>(
            n, k, lda, aGm, xUbRe, xUbIm, midOutGm, rowStart, rowEnd, xBase);
        return;
    }

    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        CtbmvColumnRange<UPLO_IS_UPPER, TRANS_IS_N>(row, n, k, colStart, colEnd);

        float accRe = 0.0f;
        float accIm = 0.0f;
        // Same affine-address hoisting as the warp-reduction path: advance a
        // register per column instead of recomputing the 64-bit band index.
        int64_t aIdx = CtbmvBandIndex<UPLO_IS_UPPER>(
            TRANS_IS_N ? row : colStart, TRANS_IS_N ? colStart : row, k, lda);
        const int64_t aIdxStep = TRANS_IS_N ? (static_cast<int64_t>(lda) - 1) : 1;
        for (uint32_t col = colStart; col < colEnd; ++col, aIdx += aIdxStep) {
            float aRe = 0.0f;
            float aIm = 0.0f;
            CtbmvLoadBandCoeff<DIAG_IS_UNIT, CONJ>(row, col, aIdx, aGm, aRe, aIm);
            uint32_t xi = col - xBase;
            float xRe = xUbRe[xi];
            float xIm = xUbIm[xi];
            CtbmvComplexMac(aRe, aIm, xRe, xIm, accRe, accIm);
        }
        midOutGm[row].real = accRe;
        midOutGm[row].imag = accIm;
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtbmvComputeSmallUb(
    uint32_t n, uint32_t k, uint32_t lda, __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    // One block owns the whole vector. Cache x before any row is written so
    // parallel row evaluation remains safe for the in-place BLAS API.
    __ubuf__ float xUbRe[CTBMV_UB_X_ELEMS];
    __ubuf__ float xUbIm[CTBMV_UB_X_ELEMS];
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        xUbRe[i] = xGm[i].real;
        xUbIm[i] = xGm[i].imag;
    }
    asc_syncthreads();

    for (uint32_t row = threadIdx.x; row < n; row += blockDim.x) {
        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        CtbmvColumnRange<UPLO_IS_UPPER, TRANS_IS_N>(row, n, k, colStart, colEnd);
        float accRe = 0.0f;
        float accIm = 0.0f;
        for (uint32_t col = colStart; col < colEnd; ++col) {
            float aRe = 0.0f;
            float aIm = 0.0f;
            CtbmvLoadRowCoeff<UPLO_IS_UPPER, TRANS_IS_N, DIAG_IS_UNIT, CONJ>(row, col, k, lda, aGm, aRe, aIm);
            CtbmvComplexMac(aRe, aIm, xUbRe[col], xUbIm[col], accRe, accIm);
        }
        xGm[row].real = accRe;
        xGm[row].imag = accIm;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtbmvCopyBack(
    uint32_t n, int64_t incx, __gm__ const aclblasComplex* midOutGm, __gm__ aclblasComplex* xGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        int64_t xIdx = CtbmvVectorIndex(row, n, incx);
        xGm[xIdx].real = midOutGm[row].real;
        xGm[xIdx].imag = midOutGm[row].imag;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtbmvDiagScale(
    uint32_t n, uint32_t lda, int64_t incx, uint32_t trans, __gm__ const aclblasComplex* aGm,
    __gm__ aclblasComplex* xGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        int64_t xIdx = CtbmvVectorIndex(row, n, incx);
        int64_t aIdx = static_cast<int64_t>(row) * static_cast<int64_t>(lda);
        float dRe = aGm[aIdx].real;
        float dIm = aGm[aIdx].imag;
        float xRe = xGm[xIdx].real;
        float xIm = xGm[xIdx].imag;
        if (trans == ACLBLAS_OP_C) {
            dIm = -dIm;
        }
        float outRe = 0.0f;
        float outIm = 0.0f;
        CtbmvComplexMac(dRe, dIm, xRe, xIm, outRe, outIm);
        xGm[xIdx].real = outRe;
        xGm[xIdx].imag = outIm;
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ>
__aicore__ __inline__ void CtbmvDispatchDiagGm(uint32_t diag, const CtbmvTilingData& tdata,
    __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm, __gm__ aclblasComplex* midOutGm)
{
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<CtbmvComputeGm<UPLO_IS_UPPER, TRANS_IS_N, CONJ, true>>(
            dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.k, tdata.lda, tdata.incx, aGm, xGm, midOutGm);
    } else {
        asc_vf_call<CtbmvComputeGm<UPLO_IS_UPPER, TRANS_IS_N, CONJ, false>>(
            dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.k, tdata.lda, tdata.incx, aGm, xGm, midOutGm);
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ>
__aicore__ __inline__ void CtbmvDispatchDiagUb(uint32_t diag, const CtbmvTilingData& tdata,
    __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm, __gm__ aclblasComplex* midOutGm,
    uint32_t rowStart, uint32_t rowEnd, uint32_t xBase, uint32_t xLen)
{
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<CtbmvComputeUb<UPLO_IS_UPPER, TRANS_IS_N, CONJ, true>>(dim3{tdata.numThreads, 1, 1},
            tdata.n, tdata.k, tdata.lda, aGm, xGm, midOutGm, rowStart, rowEnd, xBase, xLen);
    } else {
        asc_vf_call<CtbmvComputeUb<UPLO_IS_UPPER, TRANS_IS_N, CONJ, false>>(dim3{tdata.numThreads, 1, 1},
            tdata.n, tdata.k, tdata.lda, aGm, xGm, midOutGm, rowStart, rowEnd, xBase, xLen);
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ>
__aicore__ __inline__ void CtbmvDispatchComputeGm(
    const CtbmvTilingData& tdata, __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm,
    __gm__ aclblasComplex* midOutGm)
{
    CtbmvDispatchDiagGm<UPLO_IS_UPPER, TRANS_IS_N, CONJ>(tdata.diag, tdata, aGm, xGm, midOutGm);
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ>
__aicore__ __inline__ void CtbmvDispatchComputeUb(const CtbmvTilingData& tdata, __gm__ const aclblasComplex* aGm,
    __gm__ const aclblasComplex* xGm, __gm__ aclblasComplex* midOutGm)
{
    uint32_t blkIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    uint32_t rowStart = blkIdx * tdata.rowsPerBlock;
    uint32_t rowEnd = rowStart + tdata.rowsPerBlock;
    rowEnd = (rowEnd < tdata.n) ? rowEnd : tdata.n;
    if (rowStart >= rowEnd) {
        return;
    }

    uint32_t xBase = 0;
    uint32_t xEnd = 0;
    if constexpr ((TRANS_IS_N && UPLO_IS_UPPER) || (!TRANS_IS_N && !UPLO_IS_UPPER)) {
        xBase = rowStart;
        uint64_t end = static_cast<uint64_t>(rowEnd) + tdata.k;
        xEnd = (end < tdata.n) ? static_cast<uint32_t>(end) : tdata.n;
    } else {
        xBase = (rowStart >= tdata.k) ? (rowStart - tdata.k) : 0U;
        xEnd = rowEnd;
    }
    uint32_t xLen = xEnd - xBase;
    if (xLen == 0 || xLen > CTBMV_UB_X_ELEMS) {
        CtbmvDispatchComputeGm<UPLO_IS_UPPER, TRANS_IS_N, CONJ>(tdata, aGm, xGm, midOutGm);
        return;
    }
    CtbmvDispatchDiagUb<UPLO_IS_UPPER, TRANS_IS_N, CONJ>(
        tdata.diag, tdata, aGm, xGm, midOutGm, rowStart, rowEnd, xBase, xLen);
}

template <bool UPLO_IS_UPPER>
__aicore__ __inline__ void CtbmvDispatchTrans(
    const CtbmvTilingData& tdata, __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm,
    __gm__ aclblasComplex* midOutGm)
{
    bool transIsN = (tdata.trans == ACLBLAS_OP_N);
    bool conj = (tdata.trans == ACLBLAS_OP_C);
    if (tdata.useUb != 0U) {
        if (transIsN) {
            CtbmvDispatchComputeUb<UPLO_IS_UPPER, true, false>(tdata, aGm, xGm, midOutGm);
        } else if (conj) {
            CtbmvDispatchComputeUb<UPLO_IS_UPPER, false, true>(tdata, aGm, xGm, midOutGm);
        } else {
            CtbmvDispatchComputeUb<UPLO_IS_UPPER, false, false>(tdata, aGm, xGm, midOutGm);
        }
        return;
    }

    if (transIsN) {
        CtbmvDispatchComputeGm<UPLO_IS_UPPER, true, false>(tdata, aGm, xGm, midOutGm);
    } else if (conj) {
        CtbmvDispatchComputeGm<UPLO_IS_UPPER, false, true>(tdata, aGm, xGm, midOutGm);
    } else {
        CtbmvDispatchComputeGm<UPLO_IS_UPPER, false, false>(tdata, aGm, xGm, midOutGm);
    }
}

__global__ __aicore__ void ctbmv_compute_kernel(
    __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm, __gm__ uint8_t* workspaceGm,
    CtbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ aclblasComplex* midOutGm = reinterpret_cast<__gm__ aclblasComplex*>(workspaceGm);
    if (tdata.uplo == ACLBLAS_UPPER) {
        CtbmvDispatchTrans<true>(tdata, aGm, xGm, midOutGm);
    } else {
        CtbmvDispatchTrans<false>(tdata, aGm, xGm, midOutGm);
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool CONJ>
__aicore__ __inline__ void CtbmvDispatchSmallUb(uint32_t diag, const CtbmvTilingData& tdata,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<CtbmvComputeSmallUb<UPLO_IS_UPPER, TRANS_IS_N, CONJ, true>>(
            dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.k, tdata.lda, aGm, xGm);
    } else {
        asc_vf_call<CtbmvComputeSmallUb<UPLO_IS_UPPER, TRANS_IS_N, CONJ, false>>(
            dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.k, tdata.lda, aGm, xGm);
    }
}
__global__ __aicore__ void ctbmv_netlib_small_kernel(
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm, CtbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tdata.uplo == ACLBLAS_UPPER) {
        asc_vf_call<CtbmvComputeNetlibSmall<true>>(
            dim3{1, 1, 1}, tdata.n, tdata.k, tdata.lda, tdata.trans, tdata.diag, tdata.incx, aGm, xGm);
    } else {
        asc_vf_call<CtbmvComputeNetlibSmall<false>>(
            dim3{1, 1, 1}, tdata.n, tdata.k, tdata.lda, tdata.trans, tdata.diag, tdata.incx, aGm, xGm);
    }
}

__global__ __aicore__ void ctbmv_small_ub_kernel(
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm, CtbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    bool transIsN = (tdata.trans == ACLBLAS_OP_N);
    bool conj = (tdata.trans == ACLBLAS_OP_C);
    if (tdata.uplo == ACLBLAS_UPPER) {
        if (transIsN) {
            CtbmvDispatchSmallUb<true, true, false>(tdata.diag, tdata, aGm, xGm);
        } else if (conj) {
            CtbmvDispatchSmallUb<true, false, true>(tdata.diag, tdata, aGm, xGm);
        } else {
            CtbmvDispatchSmallUb<true, false, false>(tdata.diag, tdata, aGm, xGm);
        }
    } else {
        if (transIsN) {
            CtbmvDispatchSmallUb<false, true, false>(tdata.diag, tdata, aGm, xGm);
        } else if (conj) {
            CtbmvDispatchSmallUb<false, false, true>(tdata.diag, tdata, aGm, xGm);
        } else {
            CtbmvDispatchSmallUb<false, false, false>(tdata.diag, tdata, aGm, xGm);
        }
    }
}

__global__ __aicore__ void ctbmv_copy_kernel(
    __gm__ aclblasComplex* xGm, __gm__ const uint8_t* workspaceGm, CtbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ const aclblasComplex* midOutGm = reinterpret_cast<__gm__ const aclblasComplex*>(workspaceGm);
    asc_vf_call<CtbmvCopyBack>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.incx, midOutGm, xGm);
}

__global__ __aicore__ void ctbmv_diag_kernel(
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm, CtbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<CtbmvDiagScale>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.lda, tdata.incx, tdata.trans, aGm, xGm);
}

void ctbmv_arch35_kernel_do(
    const aclblasComplex* a, aclblasComplex* x, uint8_t* workspace, const CtbmvTilingData& tilingData,
    uint32_t numBlocks, void* stream)
{
    if (tilingData.k == 0U) {
        ctbmv_diag_kernel<<<numBlocks, nullptr, stream>>>(a, x, tilingData);
        return;
    }
    if (tilingData.n <= CTBMV_SINGLE_BLOCK_UB_N && tilingData.useUb != 0U &&
        tilingData.numBlocks == 1U) {
        ctbmv_small_ub_kernel<<<1, nullptr, stream>>>(a, x, tilingData);
        return;
    }
    if (tilingData.n <= CTBMV_NETLIB_SMALL_N) {
        ctbmv_netlib_small_kernel<<<1, nullptr, stream>>>(a, x, tilingData);
        return;
    }
    ctbmv_compute_kernel<<<numBlocks, nullptr, stream>>>(a, x, workspace, tilingData);
    ctbmv_copy_kernel<<<numBlocks, nullptr, stream>>>(x, workspace, tilingData);
}

