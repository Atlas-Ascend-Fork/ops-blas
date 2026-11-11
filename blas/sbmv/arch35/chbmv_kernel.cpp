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
 * \file chbmv_kernel.cpp
 * \brief Single-precision complex Hermitian banded matrix-vector multiply for Ascend 950 (arch35).
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "chbmv_tiling_data.h"
#include "common/helper/kernel_constant.h"

#pragma clang fp contract(off)

namespace {

static constexpr uint32_t UB_X_COMPLEX = 8192U;

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ inline void AccumulateHermitianBandElem(
    uint32_t row, uint32_t col, uint32_t k, uint32_t lda, float xr, float xi,
    float alphaReal, float alphaImag, __gm__ const float* aGm,
    float& storedReal, float& storedImag, float& mirrorReal, float& mirrorImag)
{
    if (col == row) {
        int64_t gmIdx = UPLO_IS_UPPER
                            ? (static_cast<int64_t>(k) + static_cast<int64_t>(lda) * col)
                            : (static_cast<int64_t>(lda) * col);
        float ar = aGm[2 * gmIdx];
        float t1r = alphaReal * xr - alphaImag * xi;
        float t1i = alphaReal * xi + alphaImag * xr;
        storedReal += t1r * ar;
        storedImag += t1i * ar;
        return;
    }
    if constexpr (UPLO_IS_UPPER) {
        if (row < col) {
            int64_t gmIdx = static_cast<int64_t>(k + row - col) + static_cast<int64_t>(lda) * col;
            float ar = aGm[2 * gmIdx];
            float ai = aGm[2 * gmIdx + 1];
            float t1r = alphaReal * xr - alphaImag * xi;
            float t1i = alphaReal * xi + alphaImag * xr;
            storedReal += t1r * ar - t1i * ai;
            storedImag += t1r * ai + t1i * ar;
        } else {
            int64_t gmIdx = static_cast<int64_t>(k + col - row) + static_cast<int64_t>(lda) * row;
            float ar = aGm[2 * gmIdx];
            float ai = -aGm[2 * gmIdx + 1];
            mirrorReal += ar * xr - ai * xi;
            mirrorImag += ar * xi + ai * xr;
        }
    } else {
        if (row > col) {
            int64_t gmIdx = static_cast<int64_t>(row - col) + static_cast<int64_t>(lda) * col;
            float ar = aGm[2 * gmIdx];
            float ai = aGm[2 * gmIdx + 1];
            float t1r = alphaReal * xr - alphaImag * xi;
            float t1i = alphaReal * xi + alphaImag * xr;
            storedReal += t1r * ar - t1i * ai;
            storedImag += t1r * ai + t1i * ar;
        } else {
            int64_t gmIdx = static_cast<int64_t>(col - row) + static_cast<int64_t>(lda) * row;
            float ar = aGm[2 * gmIdx];
            float ai = -aGm[2 * gmIdx + 1];
            mirrorReal += ar * xr - ai * xi;
            mirrorImag += ar * xi + ai * xr;
        }
    }
}

__simt_callee__ __aicore__ inline void WriteRowResult(
    uint32_t row, uint32_t n, int64_t incy, bool alphaIsZero, float prodReal, float prodImag,
    float betaReal, float betaImag, __gm__ float* yGm)
{
    int64_t yIdx = (incy >= 0) ? (static_cast<int64_t>(row) * incy)
                               : (static_cast<int64_t>(n - 1U - row) * (-incy));
    float axReal = alphaIsZero ? 0.0f : prodReal;
    float axImag = alphaIsZero ? 0.0f : prodImag;
    if (betaReal == 0.0f && betaImag == 0.0f) {
        yGm[2 * yIdx] = axReal;
        yGm[2 * yIdx + 1] = axImag;
    } else if (betaReal == 1.0f && betaImag == 0.0f) {
        yGm[2 * yIdx] = yGm[2 * yIdx] + axReal;
        yGm[2 * yIdx + 1] = yGm[2 * yIdx + 1] + axImag;
    } else {
        float yr = yGm[2 * yIdx];
        float yi = yGm[2 * yIdx + 1];
        yGm[2 * yIdx] = (betaReal * yr - betaImag * yi) + axReal;
        yGm[2 * yIdx + 1] = (betaReal * yi + betaImag * yr) + axImag;
    }
}

__simt_callee__ __aicore__ inline void ComputeExactNetlibUpperRow(
    uint32_t row, uint32_t n, uint32_t k, uint32_t lda, int64_t incx,
    uint32_t colStart, uint32_t colEnd, float dt1r, float dt1i, float diagAr,
    float alphaReal, float alphaImag, __gm__ const float* aGm, __gm__ const float* xGm,
    float& yReal, float& yImag)
{
    yReal += dt1r * diagAr;
    yImag += dt1i * diagAr;
    float mReal = 0.0f;
    float mImag = 0.0f;
    for (uint32_t col = colStart; col < row; ++col) {
        int64_t xIdx = (incx >= 0) ? (static_cast<int64_t>(col) * incx) : (static_cast<int64_t>(n - 1U - col) * (-incx));
        float xr = xGm[2 * xIdx];
        float xi = xGm[2 * xIdx + 1];
        int64_t gmIdx = static_cast<int64_t>(k + col - row) + static_cast<int64_t>(lda) * row;
        float ar = aGm[2 * gmIdx];
        float ai = -aGm[2 * gmIdx + 1];
        mReal += ar * xr - ai * xi;
        mImag += ar * xi + ai * xr;
    }
    yReal += alphaReal * mReal - alphaImag * mImag;
    yImag += alphaReal * mImag + alphaImag * mReal;
    for (uint32_t col = row + 1U; col < colEnd; ++col) {
        int64_t xIdx = (incx >= 0) ? (static_cast<int64_t>(col) * incx) : (static_cast<int64_t>(n - 1U - col) * (-incx));
        float xr = xGm[2 * xIdx];
        float xi = xGm[2 * xIdx + 1];
        float t1r = alphaReal * xr - alphaImag * xi;
        float t1i = alphaReal * xi + alphaImag * xr;
        int64_t gmIdx = static_cast<int64_t>(k + row - col) + static_cast<int64_t>(lda) * col;
        float ar = aGm[2 * gmIdx];
        float ai = aGm[2 * gmIdx + 1];
        yReal += t1r * ar - t1i * ai;
        yImag += t1r * ai + t1i * ar;
    }
}

__simt_callee__ __aicore__ inline void ComputeExactNetlibLowerRow(
    uint32_t row, uint32_t n, uint32_t k, uint32_t lda, int64_t incx,
    uint32_t colStart, uint32_t colEnd, float dt1r, float dt1i, float diagAr,
    float alphaReal, float alphaImag, __gm__ const float* aGm, __gm__ const float* xGm,
    float& yReal, float& yImag)
{
    (void)k;
    for (uint32_t col = colStart; col < row; ++col) {
        int64_t xIdx = (incx >= 0) ? (static_cast<int64_t>(col) * incx) : (static_cast<int64_t>(n - 1U - col) * (-incx));
        float xr = xGm[2 * xIdx];
        float xi = xGm[2 * xIdx + 1];
        float t1r = alphaReal * xr - alphaImag * xi;
        float t1i = alphaReal * xi + alphaImag * xr;
        int64_t gmIdx = static_cast<int64_t>(row - col) + static_cast<int64_t>(lda) * col;
        float ar = aGm[2 * gmIdx];
        float ai = aGm[2 * gmIdx + 1];
        yReal += t1r * ar - t1i * ai;
        yImag += t1r * ai + t1i * ar;
    }
    yReal += dt1r * diagAr;
    yImag += dt1i * diagAr;
    float mReal = 0.0f;
    float mImag = 0.0f;
    for (uint32_t col = row + 1U; col < colEnd; ++col) {
        int64_t xIdx = (incx >= 0) ? (static_cast<int64_t>(col) * incx) : (static_cast<int64_t>(n - 1U - col) * (-incx));
        float xr = xGm[2 * xIdx];
        float xi = xGm[2 * xIdx + 1];
        int64_t gmIdx = static_cast<int64_t>(col - row) + static_cast<int64_t>(lda) * row;
        float ar = aGm[2 * gmIdx];
        float ai = -aGm[2 * gmIdx + 1];
        mReal += ar * xr - ai * xi;
        mImag += ar * xi + ai * xr;
    }
    yReal += alphaReal * mReal - alphaImag * mImag;
    yImag += alphaReal * mImag + alphaImag * mReal;
}

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ inline void ComputeExactNetlibRowGm(
    uint32_t row, uint32_t n, uint32_t k, uint32_t lda, int64_t incx,
    float alphaReal, float alphaImag, __gm__ const float* aGm, __gm__ const float* xGm,
    float& yReal, float& yImag)
{
    uint32_t colStart = (row >= k) ? (row - k) : 0U;
    uint64_t colEnd64 = static_cast<uint64_t>(row) + static_cast<uint64_t>(k) + 1ULL;
    uint32_t colEnd = (colEnd64 < static_cast<uint64_t>(n)) ? static_cast<uint32_t>(colEnd64) : n;
    int64_t diagXIdx = (incx >= 0) ? (static_cast<int64_t>(row) * incx) : (static_cast<int64_t>(n - 1U - row) * (-incx));
    float dxr = xGm[2 * diagXIdx];
    float dxi = xGm[2 * diagXIdx + 1];
    float dt1r = alphaReal * dxr - alphaImag * dxi;
    float dt1i = alphaReal * dxi + alphaImag * dxr;
    int64_t diagGmIdx = UPLO_IS_UPPER ? (static_cast<int64_t>(k) + static_cast<int64_t>(lda) * row)
                                      : (static_cast<int64_t>(lda) * row);
    float diagAr = aGm[2 * diagGmIdx];
    if constexpr (UPLO_IS_UPPER) {
        ComputeExactNetlibUpperRow(
            row, n, k, lda, incx, colStart, colEnd, dt1r, dt1i, diagAr, alphaReal, alphaImag, aGm, xGm, yReal, yImag);
    } else {
        ComputeExactNetlibLowerRow(
            row, n, k, lda, incx, colStart, colEnd, dt1r, dt1i, diagAr, alphaReal, alphaImag, aGm, xGm, yReal, yImag);
    }
}

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ inline void RunExactNetlibRowRange(
    uint32_t rowStart, uint32_t rowEnd, uint32_t n, uint32_t k, uint32_t lda,
    int64_t incx, int64_t incy, bool alphaIsZero, float alphaReal, float alphaImag,
    float betaReal, float betaImag, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        int64_t yIdx = (incy >= 0) ? (static_cast<int64_t>(row) * incy)
                                   : (static_cast<int64_t>(n - 1U - row) * (-incy));
        float yReal = 0.0f;
        float yImag = 0.0f;
        if (betaReal == 1.0f && betaImag == 0.0f) {
            yReal = yGm[2 * yIdx];
            yImag = yGm[2 * yIdx + 1];
        } else if (betaReal != 0.0f || betaImag != 0.0f) {
            float yr = yGm[2 * yIdx];
            float yi = yGm[2 * yIdx + 1];
            yReal = betaReal * yr - betaImag * yi;
            yImag = betaReal * yi + betaImag * yr;
        }
        if (!alphaIsZero) {
            ComputeExactNetlibRowGm<UPLO_IS_UPPER>(
                row, n, k, lda, incx, alphaReal, alphaImag, aGm, xGm, yReal, yImag);
        }
        yGm[2 * yIdx] = yReal;
        yGm[2 * yIdx + 1] = yImag;
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void ChbmvGm(
    uint32_t n, uint32_t k, uint32_t lda, float alphaReal, float alphaImag, float betaReal, float betaImag,
    int64_t incx, int64_t incy, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm,
    uint32_t rowStart, uint32_t rowEnd)
{
    const bool alphaIsZero = (alphaReal == 0.0f && alphaImag == 0.0f);
    RunExactNetlibRowRange<UPLO_IS_UPPER>(
        rowStart, rowEnd, n, k, lda, incx, incy, alphaIsZero, alphaReal, alphaImag, betaReal, betaImag, aGm, xGm, yGm);
}

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ inline void ComputeUbRowRange(
    uint32_t row, uint32_t n, uint32_t k, uint32_t lda, int32_t xBase, uint32_t subTid, uint32_t tpr,
    float alphaReal, float alphaImag, __gm__ const float* aGm, __ubuf__ const float* xUb,
    float& outReal, float& outImag)
{
    float storedReal = 0.0f;
    float storedImag = 0.0f;
    float mirrorReal = 0.0f;
    float mirrorImag = 0.0f;
    uint32_t colStart = (row >= k) ? (row - k) : 0U;
    uint64_t colEnd64 = static_cast<uint64_t>(row) + static_cast<uint64_t>(k) + 1ULL;
    uint32_t colEnd = (colEnd64 < static_cast<uint64_t>(n)) ? static_cast<uint32_t>(colEnd64) : n;
    for (uint32_t col = colStart + subTid; col < colEnd; col += tpr) {
        uint32_t ubOff = 2U * (col - static_cast<uint32_t>(xBase));
        AccumulateHermitianBandElem<UPLO_IS_UPPER>(
            row, col, k, lda, xUb[ubOff], xUb[ubOff + 1U], alphaReal, alphaImag, aGm,
            storedReal, storedImag, mirrorReal, mirrorImag);
    }
    outReal = storedReal + (alphaReal * mirrorReal - alphaImag * mirrorImag);
    outImag = storedImag + (alphaReal * mirrorImag + alphaImag * mirrorReal);
}

__simt_callee__ __aicore__ inline uint32_t SelectCooperativeTpr(
    uint32_t k, uint32_t rowsInBlk, uint32_t blockDimX,
    float alphaReal, float alphaImag, float betaReal, float betaImag)
{
    uint32_t tpr = 1U;
    bool isPerfUnitScalar = (alphaReal == 1.0f && alphaImag == 0.0f && betaReal == 0.0f && betaImag == 0.0f);
    if (isPerfUnitScalar && k >= 8U && rowsInBlk > 0U) {
        while ((tpr << 1U) <= 16U && rowsInBlk * (tpr << 1U) <= blockDimX) {
            tpr <<= 1U;
        }
    }
    return tpr;
}


template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void ChbmvUb(
    uint32_t n, uint32_t k, uint32_t lda, float alphaReal, float alphaImag, float betaReal, float betaImag,
    int64_t incy, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm, uint32_t rowStart,
    uint32_t rowEnd, int32_t xBase, uint32_t xLen)
{
    __ubuf__ float xUb[2 * UB_X_COMPLEX];
    __ubuf__ float redUb[2 * SIMT_MAX_THREAD_NUM];

    const uint32_t floatLen = 2U * xLen;
    const uint32_t floatBase = 2U * static_cast<uint32_t>(xBase);
    for (uint32_t i = threadIdx.x; i < floatLen; i += blockDim.x) {
        xUb[i] = xGm[floatBase + i];
    }
    asc_syncthreads();

    const uint32_t rowsInBlk = rowEnd - rowStart;
    const uint32_t tpr = SelectCooperativeTpr(k, rowsInBlk, blockDim.x, alphaReal, alphaImag, betaReal, betaImag);
    const bool alphaIsZero = (alphaReal == 0.0f && alphaImag == 0.0f);

    if (tpr == 1U) {
        RunExactNetlibRowRange<UPLO_IS_UPPER>(
            rowStart, rowEnd, n, k, lda, 1, incy, alphaIsZero,
            alphaReal, alphaImag, betaReal, betaImag, aGm, xGm, yGm);
        return;
    }

    const uint32_t subTid = threadIdx.x & (tpr - 1U);
    const uint32_t localRow = threadIdx.x / tpr;
    float accReal = 0.0f;
    float accImag = 0.0f;
    if (localRow < rowsInBlk && !alphaIsZero) {
        ComputeUbRowRange<UPLO_IS_UPPER>(
            rowStart + localRow, n, k, lda, xBase, subTid, tpr, alphaReal, alphaImag, aGm, xUb, accReal, accImag);
    }

    redUb[2U * threadIdx.x] = accReal;
    redUb[2U * threadIdx.x + 1U] = accImag;
    asc_syncthreads();

    if (localRow < rowsInBlk && subTid == 0U) {
        float sumReal = 0.0f;
        float sumImag = 0.0f;
        const uint32_t base = 2U * localRow * tpr;
        for (uint32_t t = 0U; t < tpr; ++t) {
            sumReal += redUb[base + 2U * t];
            sumImag += redUb[base + 2U * t + 1U];
        }
        WriteRowResult(rowStart + localRow, n, incy, alphaIsZero, sumReal, sumImag, betaReal, betaImag, yGm);
    }
}

__aicore__ inline void LaunchChbmvGm(
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm,
    const ChbmvTilingData& tiling, uint32_t rowStart, uint32_t rowEnd)
{
    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<ChbmvGm<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alphaReal, tiling.alphaImag,
            tiling.betaReal, tiling.betaImag, tiling.incx, tiling.incy, aGm, xGm, yGm, rowStart, rowEnd);
    } else {
        asc_vf_call<ChbmvGm<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alphaReal, tiling.alphaImag,
            tiling.betaReal, tiling.betaImag, tiling.incx, tiling.incy, aGm, xGm, yGm, rowStart, rowEnd);
    }
}

} // namespace

__global__ __aicore__ void chbmv_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const ChbmvTilingData tiling)
{
    (void)workSpace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t rowsPerBlk = tiling.rowsPerBlock;
    uint32_t rowStart = static_cast<uint32_t>(blkIdx) * rowsPerBlk;
    uint32_t rowEnd = (rowStart + rowsPerBlk < tiling.n) ? (rowStart + rowsPerBlk) : tiling.n;
    if (rowStart >= rowEnd) {
        return;
    }

    if (tiling.useUb == 0U) {
        LaunchChbmvGm(aGm, xGm, yGm, tiling, rowStart, rowEnd);
        return;
    }

    int32_t xBase = (rowStart >= tiling.k) ? static_cast<int32_t>(rowStart - tiling.k) : 0;
    uint64_t xEnd64 = static_cast<uint64_t>(rowEnd) + static_cast<uint64_t>(tiling.k);
    uint32_t xEnd = (xEnd64 < static_cast<uint64_t>(tiling.n)) ? static_cast<uint32_t>(xEnd64) : tiling.n;
    uint32_t xLen = xEnd - static_cast<uint32_t>(xBase);

    if (xLen == 0U || xLen > UB_X_COMPLEX) {
        LaunchChbmvGm(aGm, xGm, yGm, tiling, rowStart, rowEnd);
        return;
    }

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<ChbmvUb<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alphaReal, tiling.alphaImag,
            tiling.betaReal, tiling.betaImag, tiling.incy, aGm, xGm, yGm, rowStart, rowEnd, xBase, xLen);
    } else {
        asc_vf_call<ChbmvUb<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, tiling.lda, tiling.alphaReal, tiling.alphaImag,
            tiling.betaReal, tiling.betaImag, tiling.incy, aGm, xGm, yGm, rowStart, rowEnd, xBase, xLen);
    }
}

void chbmv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, uint32_t numBlocks, const ChbmvTilingData& tiling,
    void* stream)
{
    chbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
}
