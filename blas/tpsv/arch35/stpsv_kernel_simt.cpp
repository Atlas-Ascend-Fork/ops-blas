/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file stpsv_kernel_simt.cpp
 * \brief SIMT VF kernel for tpsv (n >= 128). Parallel inner product with tree reduction.
 */

#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "stpsv_tiling_data.h"
#include "common/helper/kernel_constant.h"
#include "stpsv_kernel_utils.h"

using namespace AscendC;

constexpr uint32_t UB_PARTIALS_FLOATS = 2048;

// ==========================================================================
//  SIMT VF — multi-threaded inner product with tree reduction
// ==========================================================================

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT>
__simt_callee__ inline void StpsvSimtReduceAndUpdate(
    float partial, float sum, uint32_t xOff, uint32_t row, uint32_t n, __gm__ const float* apGm, __gm__ float* xGm,
    __ubuf__ float* partialSums)
{
    partialSums[threadIdx.x] = partial;
    asc_syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
        }
        asc_syncthreads();
    }

    if (threadIdx.x == 0) {
        sum -= partialSums[0];
        if constexpr (!DIAG_IS_UNIT) {
            float diag;
            if constexpr (!UPLO_IS_UPPER) {
                diag = apGm[TpsvPackedLowerIdxSimt(row, row, n)];
            } else {
                diag = apGm[TpsvPackedUpperIdxSimt(row, row)];
            }
            sum = sum / diag;
        }
        xGm[xOff] = sum;
    }
    asc_syncthreads();
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT>
__simt_callee__ inline void StpsvSimtForward(
    uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ float* xGm, __ubuf__ float* partialSums)
{
    for (uint32_t row = 0; row < n; ++row) {
        uint32_t xOff =
            (incx >= 0) ? (row * static_cast<uint32_t>(incx)) : ((n - 1 - row) * static_cast<uint32_t>(-incx));
        float sum = xGm[xOff];

        float partial = 0.0f;
        for (uint32_t j = threadIdx.x; j < row; j += blockDim.x) {
            float aVal;
            if constexpr (!UPLO_IS_UPPER) {
                aVal = apGm[TpsvPackedLowerIdxSimt(row, j, n)];
            } else {
                aVal = apGm[TpsvPackedUpperIdxSimt(j, row)];
            }
            uint32_t xjOff =
                (incx >= 0) ? (j * static_cast<uint32_t>(incx)) : ((n - 1 - j) * static_cast<uint32_t>(-incx));
            partial += aVal * xGm[xjOff];
        }

        StpsvSimtReduceAndUpdate<UPLO_IS_UPPER, DIAG_IS_UNIT>(partial, sum, xOff, row, n, apGm, xGm, partialSums);
    }
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT>
__simt_callee__ inline void StpsvSimtBackward(
    uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ float* xGm, __ubuf__ float* partialSums)
{
    for (uint32_t row = n; row-- > 0;) {
        uint32_t xOff =
            (incx >= 0) ? (row * static_cast<uint32_t>(incx)) : ((n - 1 - row) * static_cast<uint32_t>(-incx));
        float sum = xGm[xOff];

        float partial = 0.0f;
        for (uint32_t j = row + 1 + threadIdx.x; j < n; j += blockDim.x) {
            float aVal;
            if constexpr (UPLO_IS_UPPER) {
                aVal = apGm[TpsvPackedUpperIdxSimt(row, j)];
            } else {
                aVal = apGm[TpsvPackedLowerIdxSimt(j, row, n)];
            }
            uint32_t xjOff =
                (incx >= 0) ? (j * static_cast<uint32_t>(incx)) : ((n - 1 - j) * static_cast<uint32_t>(-incx));
            partial += aVal * xGm[xjOff];
        }

        StpsvSimtReduceAndUpdate<UPLO_IS_UPPER, DIAG_IS_UNIT>(partial, sum, xOff, row, n, apGm, xGm, partialSums);
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StpsvSimt(
    uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ float* xGm)
{
    __ubuf__ float partialSums[UB_PARTIALS_FLOATS];
    constexpr bool kForward = (!UPLO_IS_UPPER && !TRANS_IS_TRANS) || (UPLO_IS_UPPER && TRANS_IS_TRANS);

    if constexpr (kForward) {
        StpsvSimtForward<UPLO_IS_UPPER, DIAG_IS_UNIT>(n, incx, apGm, xGm, partialSums);
    } else {
        StpsvSimtBackward<UPLO_IS_UPPER, DIAG_IS_UNIT>(n, incx, apGm, xGm, partialSums);
    }
}

// ==========================================================================
//  SIMT kernel dispatcher — single entry, dispatches via asc_vf_call
// ==========================================================================

__global__ __aicore__ void stpsv_simt_kernel(StpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* apGm = reinterpret_cast<__gm__ float*>(tiling.ap);
    auto* xGm = reinterpret_cast<__gm__ float*>(tiling.x);

    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StpsvSimt<false, false, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm);
            } else {
                asc_vf_call<StpsvSimt<false, false, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StpsvSimt<false, true, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm);
            } else {
                asc_vf_call<StpsvSimt<false, true, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm);
            }
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StpsvSimt<true, false, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm);
            } else {
                asc_vf_call<StpsvSimt<true, false, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StpsvSimt<true, true, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm);
            } else {
                asc_vf_call<StpsvSimt<true, true, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm);
            }
        }
    }
}

// ==========================================================================
//  SIMT kernel launch wrapper — called from stpsv_kernel_do
// ==========================================================================

// issue #1181: launch through aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs
// so the runtime status of the launch is returned to the caller.
aclError stpsv_simt_kernel_do(const StpsvTilingData& tiling, void* stream)
{
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(stpsv_simt_kernel), &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        StpsvTilingData tiling;
    } args{tiling};
    static_assert(sizeof(Arguments) == sizeof(StpsvTilingData), "kernel ABI mismatch");
    return aclrtLaunchKernelWithHostArgs(function, 1U, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}
