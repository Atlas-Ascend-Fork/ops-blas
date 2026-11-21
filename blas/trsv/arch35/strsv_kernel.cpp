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
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "strsv_tiling_data.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

constexpr uint32_t UB_PARTIALS_FLOATS = 2048;
constexpr uint32_t UB_X_FLOATS = 4096;

__simt_callee__ inline float StrsvSimtReduce(float partial, __ubuf__ float* partialSums)
{
    partialSums[threadIdx.x] = partial;
    asc_syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
        }
        asc_syncthreads();
    }

    return partialSums[0];
}

template <bool ROW_MAJOR_ACCESS>
__simt_callee__ inline float StrsvSimtComputePartial(
    uint32_t row, uint32_t jStart, uint32_t jEnd, uint32_t ldaU32, __gm__ const float* aGm, __gm__ float* xGm,
    int32_t incx, uint32_t absIncx, uint32_t n)
{
    float partial = 0.0f;
    for (uint32_t j = jStart + threadIdx.x; j < jEnd; j += blockDim.x) {
        uint32_t xjOff = (incx >= 0) ? (j * absIncx) : ((n - 1 - j) * absIncx);
        float aVal;
        if constexpr (ROW_MAJOR_ACCESS) {
            aVal = aGm[row + j * ldaU32];
        } else {
            aVal = aGm[j + row * ldaU32];
        }
        partial += aVal * xGm[xjOff];
    }
    return partial;
}

__simt_callee__ inline void StrsvSimtLoadX(
    uint32_t n, int32_t incx, uint32_t absIncx, __gm__ float* xGm, __ubuf__ float* xUb)
{
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        uint32_t xjOff = (incx >= 0) ? (j * absIncx) : ((n - 1 - j) * absIncx);
        xUb[j] = xGm[xjOff];
    }
    asc_syncthreads();
}

template <bool ROW_MAJOR_ACCESS>
__simt_callee__ inline float StrsvSimtDotPartial(
    uint32_t row, uint32_t jStart, uint32_t jEnd, uint32_t ldaU32, __gm__ const float* aGm, __ubuf__ float* xUb)
{
    float partial = 0.0f;
    for (uint32_t j = jStart + threadIdx.x; j < jEnd; j += blockDim.x) {
        float aVal;
        if constexpr (ROW_MAJOR_ACCESS) {
            aVal = aGm[row + j * ldaU32];
        } else {
            aVal = aGm[j + row * ldaU32];
        }
        partial += aVal * xUb[j];
    }
    return partial;
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TILED, bool ROW_MAJOR>
__simt_callee__ inline void StrsvSimtProcessRow(
    uint32_t row, uint32_t n, int32_t incx, uint32_t absIncx, uint32_t ldaU32, __gm__ const float* aGm,
    __gm__ float* xGm, __ubuf__ float* partialSums, __ubuf__ float* xUb, uint32_t jStart, uint32_t jEnd)
{
    uint32_t xOff = (incx >= 0) ? (row * absIncx) : ((n - 1 - row) * absIncx);
    float sum = TILED ? xGm[xOff] : xUb[row];

    float partial;
    if constexpr (TILED) {
        partial = StrsvSimtComputePartial<ROW_MAJOR>(row, jStart, jEnd, ldaU32, aGm, xGm, incx, absIncx, n);
    } else {
        partial = StrsvSimtDotPartial<ROW_MAJOR>(row, jStart, jEnd, ldaU32, aGm, xUb);
    }

    float total = StrsvSimtReduce(partial, partialSums);
    if (threadIdx.x == 0) {
        sum -= total;
        if constexpr (!DIAG_IS_UNIT) {
            sum = sum / aGm[row * ldaU32 + row];
        }
        xGm[xOff] = sum;
        if constexpr (!TILED) {
            xUb[row] = sum;
        }
    }
    asc_syncthreads();
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool FORWARD, bool TILED>
__simt_callee__ inline void StrsvSimtProcess(
    uint32_t n, int32_t incx, int32_t lda, __gm__ const float* aGm, __gm__ float* xGm, __ubuf__ float* partialSums,
    __ubuf__ float* xUb)
{
    uint32_t ldaU32 = static_cast<uint32_t>(lda);
    uint32_t absIncx = static_cast<uint32_t>(incx >= 0 ? incx : -incx);

    if constexpr (!TILED) {
        StrsvSimtLoadX(n, incx, absIncx, xGm, xUb);
    }

    if constexpr (FORWARD) {
        for (uint32_t row = 0; row < n; ++row) {
            StrsvSimtProcessRow<UPLO_IS_UPPER, DIAG_IS_UNIT, TILED, !UPLO_IS_UPPER>(
                row, n, incx, absIncx, ldaU32, aGm, xGm, partialSums, xUb, 0, row);
        }
    } else {
        for (uint32_t row = n; row-- > 0;) {
            StrsvSimtProcessRow<UPLO_IS_UPPER, DIAG_IS_UNIT, TILED, UPLO_IS_UPPER>(
                row, n, incx, absIncx, ldaU32, aGm, xGm, partialSums, xUb, row + 1, n);
        }
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StrsvSimt(
    uint32_t n, int32_t incx, int32_t lda, __gm__ const float* aGm, __gm__ float* xGm)
{
    __ubuf__ float partialSums[UB_PARTIALS_FLOATS];
    __ubuf__ float xUb[UB_X_FLOATS];
    constexpr bool kForward = (!UPLO_IS_UPPER && !TRANS_IS_TRANS) || (UPLO_IS_UPPER && TRANS_IS_TRANS);

    if (n <= UB_X_FLOATS) {
        if constexpr (kForward) {
            StrsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, true, false>(n, incx, lda, aGm, xGm, partialSums, xUb);
        } else {
            StrsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, false, false>(n, incx, lda, aGm, xGm, partialSums, xUb);
        }
    } else {
        if constexpr (kForward) {
            StrsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, true, true>(n, incx, lda, aGm, xGm, partialSums, xUb);
        } else {
            StrsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, false, true>(n, incx, lda, aGm, xGm, partialSums, xUb);
        }
    }
}

__global__ __aicore__ void strsv_simt_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, const StrsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* aGm = reinterpret_cast<__gm__ float*>(a);
    auto* xGm = reinterpret_cast<__gm__ float*>(x);

    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StrsvSimt<false, false, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, tiling.lda, aGm, xGm);
            } else {
                asc_vf_call<StrsvSimt<false, false, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, tiling.lda, aGm, xGm);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StrsvSimt<false, true, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, tiling.lda, aGm, xGm);
            } else {
                asc_vf_call<StrsvSimt<false, true, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, tiling.lda, aGm, xGm);
            }
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StrsvSimt<true, false, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, tiling.lda, aGm, xGm);
            } else {
                asc_vf_call<StrsvSimt<true, false, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, tiling.lda, aGm, xGm);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StrsvSimt<true, true, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, tiling.lda, aGm, xGm);
            } else {
                asc_vf_call<StrsvSimt<true, true, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, tiling.lda, aGm, xGm);
            }
        }
    }
}

void strsv_kernel_do(GM_ADDR gmAddrA, GM_ADDR gmAddrX, const StrsvTilingData& tiling, void* stream)
{
    strsv_simt_kernel<<<1, nullptr, stream>>>(gmAddrA, gmAddrX, nullptr, tiling);
}
