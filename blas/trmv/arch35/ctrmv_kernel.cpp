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
 * \file ctrmv_kernel.cpp
 * \brief single-precision complex (complex64) triangular matrix-vector multiply kernels for ascend950
 *
 * x = op(A) * x, op(A) = A | A^T | A^H
 * Two-stage SIMT scheme for in-place safety:
 *   stage 1: each thread accumulates a column slice of one row of op(A)*x; the
 *            threads of a row-group reduce via warp shuffle and thread 0 writes
 *            the complex64 slot into workspace (original x is only read, never
 *            overwritten before all rows are consumed);
 *   stage 2: copy workspace rows back to x[i*incx] with the (possibly negative) stride.
 *
 * Parallelism: rows are split across all AIV cores; within a core, each row is
 * cooperatively accumulated by `threadsPerRow` threads (column split), so
 * medium n (512/1024/2048) keeps every core busy and shortens the serial
 * GM-latency chain per thread by threadsPerRow.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#include "ctrmv_common.h"

__simt_callee__ __aicore__ __inline__ int64_t CtrmvVectorIndex(uint32_t idx, uint32_t n, int64_t incx)
{
    return (incx >= 0) ? static_cast<int64_t>(idx) * incx : static_cast<int64_t>(n - 1 - idx) * (-incx);
}

// Triangular column bounds [colStart, colEnd) for output row `row`.
// OP_N + UPPER: [row, n);  OP_N + LOWER: [0, row+1)
// OP_T/C + UPPER: [0, row+1);  OP_T/C + LOWER: [row, n)
template <bool UPLO_IS_UPPER, bool TRANS_IS_N>
__simt_callee__ __aicore__ __inline__ void CtrmvTriBounds(
    uint32_t row, uint32_t n, uint32_t& colStart, uint32_t& colEnd)
{
    colStart = 0;
    colEnd = n;
    if constexpr (TRANS_IS_N) {
        if constexpr (UPLO_IS_UPPER) {
            colStart = row;
        } else {
            colEnd = row + 1;
        }
    } else {
        if constexpr (UPLO_IS_UPPER) {
            colEnd = row + 1;
        } else {
            colStart = row;
        }
    }
}

// Load A(row, col); diag=UNIT treats the diagonal as 1 without accessing it.
template <bool TRANS_IS_N, bool TRANS_IS_C>
__simt_callee__ __aicore__ __inline__ void CtrmvLoadA(
    uint32_t row, uint32_t col, uint32_t lda, __gm__ const float* aF, float& aReal, float& aImag)
{
    const int64_t aOff =
        (TRANS_IS_N) ? (2 * (static_cast<int64_t>(col) * lda + row)) : (2 * (static_cast<int64_t>(row) * lda + col));
    const __gm__ float* aPtr = aF + aOff;
    aReal = aPtr[0];
    aImag = TRANS_IS_C ? -aPtr[1] : aPtr[1];
}

// One lane's partial accumulation over its column slice:
// cols [colStart + lane, colEnd) step threadsPerRow.
template <bool TRANS_IS_N, bool TRANS_IS_C, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ __inline__ void CtrmvLaneAccumulate(
    uint32_t row, uint32_t lane, uint32_t threadsPerRow, uint32_t colStart, uint32_t colEnd, uint32_t lda, int64_t incx,
    uint32_t n, __gm__ const float* aF, __gm__ const float* xF, float& accReal, float& accImag)
{
    for (uint32_t col = colStart + lane; col < colEnd; col += threadsPerRow) {
        float aReal = 1.0f;
        float aImag = 0.0f;
        if constexpr (DIAG_IS_UNIT) {
            if (col != row) {
                CtrmvLoadA<TRANS_IS_N, TRANS_IS_C>(row, col, lda, aF, aReal, aImag);
            }
        } else {
            CtrmvLoadA<TRANS_IS_N, TRANS_IS_C>(row, col, lda, aF, aReal, aImag);
        }
        const int64_t xOff = CtrmvVectorIndex(col, n, incx);
        const float xReal = xF[2 * xOff];
        const float xImag = xF[2 * xOff + 1];
        accReal += aReal * xReal - aImag * xImag;
        accImag += aReal * xImag + aImag * xReal;
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool TRANS_IS_C, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtrmvCompute(
    uint32_t rowsPerCore, uint32_t threadsPerRow, uint32_t n, uint32_t lda, int64_t incx,
    __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm, __gm__ aclblasComplex* midOutGm)
{
    __gm__ const float* aF = reinterpret_cast<__gm__ const float*>(aGm);
    __gm__ const float* xF = reinterpret_cast<__gm__ const float*>(xGm);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % threadsPerRow;
    const uint32_t rowGroup = tid / threadsPerRow;
    // numThreads is capped at SIMT_MAX_THREAD_NUM, so a block may hold fewer row
    // groups than rowsPerCore. Each group therefore iterates over multiple rows;
    // without this loop, rows >= blockDim.x / threadsPerRow would be dropped for
    // large n (precision error).
    const uint32_t rowGroupsPerBlock = blockDim.x / threadsPerRow;

    for (uint32_t r = rowGroup; r < rowsPerCore; r += rowGroupsPerBlock) {
        const uint32_t row = blockIdx.x * rowsPerCore + r;

        // Rows beyond n (last core) contribute 0; the whole group still joins the
        // warp shuffle so the group stays convergent.
        const bool valid = (row < n);
        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        if (valid) {
            CtrmvTriBounds<UPLO_IS_UPPER, TRANS_IS_N>(row, n, colStart, colEnd);
        }

        float accReal = 0.0f;
        float accImag = 0.0f;
        CtrmvLaneAccumulate<TRANS_IS_N, TRANS_IS_C, DIAG_IS_UNIT>(
            row, lane, threadsPerRow, colStart, colEnd, lda, incx, n, aF, xF, accReal, accImag);

        // Intra-group reduction of the threadsPerRow partial sums (warp shuffle).
        if (threadsPerRow > 1) {
            const int width = static_cast<int>(threadsPerRow);
            for (uint32_t d = 1; d < threadsPerRow; d <<= 1) {
                accReal += asc_shfl_down(accReal, d, width);
                accImag += asc_shfl_down(accImag, d, width);
            }
        }
        if (lane == 0 && valid) {
            midOutGm[row].real = accReal;
            midOutGm[row].imag = accImag;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtrmvCopyBack(
    uint32_t n, int64_t incx, __gm__ const aclblasComplex* midOutGm, __gm__ aclblasComplex* xGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        const float vReal = midOutGm[row].real;
        const float vImag = midOutGm[row].imag;
        xGm[CtrmvVectorIndex(row, n, incx)].real = vReal;
        xGm[CtrmvVectorIndex(row, n, incx)].imag = vImag;
    }
}

// Fused single-launch path for small n (<= 64, incx == 1): one 256-thread
// block caches x into UB once, then each row is accumulated by a 4-thread
// group (column split + warp-shuffle reduce) straight back to x. Avoids the
// two-launch (compute + copyback) latency of the large path on tiny shapes.
__simt_vf__ __aicore__ LAUNCH_BOUND(CTRMV_SMALL_BLOCK_THREADS) inline void CtrmvSmall(
    CtrmvTilingData tdata, __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    const uint32_t tid = threadIdx.x;
    __ubuf__ float2 xUb[CTRMV_SMALL_N_MAX];
    if (tid < tdata.n) {
        xUb[tid] = float2{xGm[tid].real, xGm[tid].imag};
    }
    asc_syncthreads();

    // Padding threads (row >= n) contribute 0 and still take part in the warp
    // shuffle below; returning early here would diverge the warp and leave the
    // shuffle reduction undefined for small n (e.g. n=1/2/3).
    const uint32_t row = tid / CTRMV_SMALL_THREADS_PER_ROW;
    const uint32_t lane = tid % CTRMV_SMALL_THREADS_PER_ROW;
    const bool valid = (row < tdata.n);
    const bool transN = tdata.trans == ACLBLAS_OP_N;
    const bool upper = tdata.uplo == ACLBLAS_UPPER;
    uint32_t begin = 0;
    uint32_t end = 0;
    if (valid) {
        end = tdata.n;
        if (transN ? upper : !upper) {
            begin = row;
        } else {
            end = row + 1;
        }
    }
    float real = 0.0f;
    float imag = 0.0f;
    for (uint32_t col = begin + lane; col < end; col += CTRMV_SMALL_THREADS_PER_ROW) {
        float ar = 1.0f;
        float ai = 0.0f;
        if (tdata.diag != ACLBLAS_UNIT || col != row) {
            const int64_t off =
                transN ? static_cast<int64_t>(col) * tdata.lda + row : static_cast<int64_t>(row) * tdata.lda + col;
            ar = aGm[off].real;
            ai = (tdata.trans == ACLBLAS_OP_C) ? -aGm[off].imag : aGm[off].imag;
        }
        real += ar * xUb[col].x - ai * xUb[col].y;
        imag += ar * xUb[col].y + ai * xUb[col].x;
    }
    for (uint32_t d = 1; d < CTRMV_SMALL_THREADS_PER_ROW; d <<= 1) {
        real += asc_shfl_down(real, d, CTRMV_SMALL_THREADS_PER_ROW);
        imag += asc_shfl_down(imag, d, CTRMV_SMALL_THREADS_PER_ROW);
    }
    if (lane == 0 && valid) {
        xGm[row].real = real;
        xGm[row].imag = imag;
    }
}

__global__ __aicore__ void ctrmv_small_kernel(
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm, CtrmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<CtrmvSmall>(dim3{256, 1, 1}, tdata, aGm, xGm);
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool TRANS_IS_C>
__aicore__ __inline__ void DispatchDiag(
    uint32_t diag, uint32_t numThreads, uint32_t rowsPerCore, uint32_t threadsPerRow, uint32_t n, uint32_t lda,
    int64_t incx, __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm, __gm__ aclblasComplex* midOutGm)
{
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<CtrmvCompute<UPLO_IS_UPPER, TRANS_IS_N, TRANS_IS_C, true>>(
            dim3{numThreads, 1, 1}, rowsPerCore, threadsPerRow, n, lda, incx, aGm, xGm, midOutGm);
    } else {
        asc_vf_call<CtrmvCompute<UPLO_IS_UPPER, TRANS_IS_N, TRANS_IS_C, false>>(
            dim3{numThreads, 1, 1}, rowsPerCore, threadsPerRow, n, lda, incx, aGm, xGm, midOutGm);
    }
}

__global__ __aicore__ void ctrmv_compute_kernel(
    __gm__ const aclblasComplex* aGm, __gm__ const aclblasComplex* xGm, __gm__ uint8_t* workspaceGm,
    CtrmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ aclblasComplex* midOutGm = reinterpret_cast<__gm__ aclblasComplex*>(workspaceGm);
    const bool transIsN = (tdata.trans == ACLBLAS_OP_N);
    const bool transIsC = (tdata.trans == ACLBLAS_OP_C);
    if (tdata.uplo == ACLBLAS_UPPER) {
        if (transIsN) {
            DispatchDiag<true, true, false>(
                tdata.diag, tdata.numThreads, tdata.rowsPerCore, tdata.threadsPerRow, tdata.n, tdata.lda, tdata.incx,
                aGm, xGm, midOutGm);
        } else if (transIsC) {
            DispatchDiag<true, false, true>(
                tdata.diag, tdata.numThreads, tdata.rowsPerCore, tdata.threadsPerRow, tdata.n, tdata.lda, tdata.incx,
                aGm, xGm, midOutGm);
        } else {
            DispatchDiag<true, false, false>(
                tdata.diag, tdata.numThreads, tdata.rowsPerCore, tdata.threadsPerRow, tdata.n, tdata.lda, tdata.incx,
                aGm, xGm, midOutGm);
        }
    } else {
        if (transIsN) {
            DispatchDiag<false, true, false>(
                tdata.diag, tdata.numThreads, tdata.rowsPerCore, tdata.threadsPerRow, tdata.n, tdata.lda, tdata.incx,
                aGm, xGm, midOutGm);
        } else if (transIsC) {
            DispatchDiag<false, false, true>(
                tdata.diag, tdata.numThreads, tdata.rowsPerCore, tdata.threadsPerRow, tdata.n, tdata.lda, tdata.incx,
                aGm, xGm, midOutGm);
        } else {
            DispatchDiag<false, false, false>(
                tdata.diag, tdata.numThreads, tdata.rowsPerCore, tdata.threadsPerRow, tdata.n, tdata.lda, tdata.incx,
                aGm, xGm, midOutGm);
        }
    }
}

__global__ __aicore__ void ctrmv_copy_kernel(
    __gm__ aclblasComplex* xGm, __gm__ const uint8_t* workspaceGm, CtrmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ const aclblasComplex* midOutGm = reinterpret_cast<__gm__ const aclblasComplex*>(workspaceGm);
    asc_vf_call<CtrmvCopyBack>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.incx, midOutGm, xGm);
}

void ctrmv_arch35_kernel_do(
    const aclblasComplex* a, aclblasComplex* x, uint8_t* workspace, const CtrmvTilingData& tilingData,
    uint32_t numBlocks, void* stream)
{
    ctrmv_compute_kernel<<<numBlocks, nullptr, stream>>>(a, x, workspace, tilingData);
    ctrmv_copy_kernel<<<numBlocks, nullptr, stream>>>(x, workspace, tilingData);
}

void ctrmv_arch35_small_kernel_do(
    const aclblasComplex* a, aclblasComplex* x, const CtrmvTilingData& tilingData, void* stream)
{
    ctrmv_small_kernel<<<1, nullptr, stream>>>(a, x, tilingData);
}
