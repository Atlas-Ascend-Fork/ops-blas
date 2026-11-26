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
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cann_ops_blas_common.h"
#include "sspr2_tiling_data.h"
#include "common/helper/kernel_constant.h"

template <bool UPLO_IS_UPPER>
__simt_callee__ inline uint64_t Sspr2ColumnBase(uint32_t col, uint32_t n)
{
    if constexpr (UPLO_IS_UPPER) {
        return static_cast<uint64_t>(col) * (col + 1) / 2;
    } else {
        return static_cast<uint64_t>(col) * (2 * n - col + 1) / 2;
    }
}

template <bool INCX_POSITIVE, bool INCY_POSITIVE>
__simt_callee__ inline void Sspr2GmInner(
    uint32_t rowStart, uint32_t rowEnd, uint32_t n, int64_t incx, int64_t incy, float axCol, float ayCol,
    uint64_t colBase, uint32_t apOffset, __gm__ const float* __restrict xGm, __gm__ const float* __restrict yGm,
    __gm__ float* __restrict apGm)
{
    for (uint32_t row = rowStart; row < rowEnd; ++row) {
        float xRow, yRow;
        if constexpr (INCX_POSITIVE) {
            xRow = xGm[static_cast<int64_t>(row) * incx];
        } else {
            xRow = xGm[(static_cast<int64_t>(n) - 1 - static_cast<int64_t>(row)) * (-incx)];
        }
        if constexpr (INCY_POSITIVE) {
            yRow = yGm[static_cast<int64_t>(row) * incy];
        } else {
            yRow = yGm[(static_cast<int64_t>(n) - 1 - static_cast<int64_t>(row)) * (-incy)];
        }
        uint64_t apIdx = colBase + row - apOffset;
        apGm[apIdx] = apGm[apIdx] + axCol * yRow + ayCol * xRow;
    }
}

template <bool UPLO_IS_UPPER, bool INCX_POSITIVE, bool INCY_POSITIVE>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Sspr2GmImpl(
    uint32_t n, float alpha, int64_t incx, int64_t incy, __gm__ const float* __restrict xGm,
    __gm__ const float* __restrict yGm, __gm__ float* __restrict apGm)
{
    for (uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x) {
        float xCol, yCol;
        if constexpr (INCX_POSITIVE) {
            xCol = xGm[static_cast<int64_t>(col) * incx];
        } else {
            xCol = xGm[(static_cast<int64_t>(n) - 1 - static_cast<int64_t>(col)) * (-incx)];
        }
        if constexpr (INCY_POSITIVE) {
            yCol = yGm[static_cast<int64_t>(col) * incy];
        } else {
            yCol = yGm[(static_cast<int64_t>(n) - 1 - static_cast<int64_t>(col)) * (-incy)];
        }

        float axCol = alpha * xCol;
        float ayCol = alpha * yCol;

        uint64_t colBase = Sspr2ColumnBase<UPLO_IS_UPPER>(col, n);

        if constexpr (UPLO_IS_UPPER) {
            Sspr2GmInner<INCX_POSITIVE, INCY_POSITIVE>(
                0, col + 1, n, incx, incy, axCol, ayCol, colBase, 0, xGm, yGm, apGm);
        } else {
            Sspr2GmInner<INCX_POSITIVE, INCY_POSITIVE>(
                col, n, n, incx, incy, axCol, ayCol, colBase, col, xGm, yGm, apGm);
        }
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Sspr2Ub(
    uint32_t n, float alpha, __gm__ const float* __restrict xGm, __gm__ const float* __restrict yGm,
    __gm__ float* __restrict apGm, uint32_t colStart, uint32_t colEnd, uint32_t xyBase, uint32_t xyLen)
{
    __ubuf__ float xUb[UB_XY_FLOATS];
    __ubuf__ float yUb[UB_XY_FLOATS];

    for (uint32_t i = threadIdx.x; i < xyLen; i += blockDim.x) {
        xUb[i] = xGm[xyBase + i];
        yUb[i] = yGm[xyBase + i];
    }
    asc_syncthreads();

    for (uint32_t col = colStart + threadIdx.x; col < colEnd; col += blockDim.x) {
        float axCol = alpha * xUb[col - xyBase];
        float ayCol = alpha * yUb[col - xyBase];

        uint64_t colBase = Sspr2ColumnBase<UPLO_IS_UPPER>(col, n);

        if constexpr (UPLO_IS_UPPER) {
            for (uint32_t row = 0; row <= col; ++row) {
                float xRow = xUb[row - xyBase];
                float yRow = yUb[row - xyBase];
                uint64_t apIdx = colBase + row;
                apGm[apIdx] = apGm[apIdx] + axCol * yRow + ayCol * xRow;
            }
        } else {
            for (uint32_t row = col; row < n; ++row) {
                float xRow = xUb[row - xyBase];
                float yRow = yUb[row - xyBase];
                uint64_t apIdx = colBase + (row - col);
                apGm[apIdx] = apGm[apIdx] + axCol * yRow + ayCol * xRow;
            }
        }
    }
}

template <bool UPLO_IS_UPPER>
__aicore__ inline void Sspr2GmDispatch(
    const Sspr2TilingData& tiling, __gm__ const float* __restrict xGm, __gm__ const float* __restrict yGm,
    __gm__ float* __restrict apGm)
{
    if (tiling.incx >= 0) {
        if (tiling.incy >= 0) {
            asc_vf_call<Sspr2GmImpl<UPLO_IS_UPPER, true, true>>(
                dim3{SIMT_MAX_THREAD_NUM, 1, 1}, tiling.n, tiling.alpha, tiling.incx, tiling.incy, xGm, yGm, apGm);
        } else {
            asc_vf_call<Sspr2GmImpl<UPLO_IS_UPPER, true, false>>(
                dim3{SIMT_MAX_THREAD_NUM, 1, 1}, tiling.n, tiling.alpha, tiling.incx, tiling.incy, xGm, yGm, apGm);
        }
        return;
    }

    if (tiling.incy >= 0) {
        asc_vf_call<Sspr2GmImpl<UPLO_IS_UPPER, false, true>>(
            dim3{SIMT_MAX_THREAD_NUM, 1, 1}, tiling.n, tiling.alpha, tiling.incx, tiling.incy, xGm, yGm, apGm);
        return;
    }
    asc_vf_call<Sspr2GmImpl<UPLO_IS_UPPER, false, false>>(
        dim3{SIMT_MAX_THREAD_NUM, 1, 1}, tiling.n, tiling.alpha, tiling.incx, tiling.incy, xGm, yGm, apGm);
}

__aicore__ inline void Sspr2FallbackToGm(
    const Sspr2TilingData& tiling, __gm__ const float* __restrict xGm, __gm__ const float* __restrict yGm,
    __gm__ float* __restrict apGm)
{
    if (tiling.uplo == ACLBLAS_UPPER) {
        Sspr2GmDispatch<true>(tiling, xGm, yGm, apGm);
        return;
    }
    Sspr2GmDispatch<false>(tiling, xGm, yGm, apGm);
}

__global__ __aicore__ void sspr2_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR ap, const Sspr2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* __restrict xGm = reinterpret_cast<__gm__ const float* __restrict>(x);
    auto* __restrict yGm = reinterpret_cast<__gm__ const float* __restrict>(y);
    auto* __restrict apGm = reinterpret_cast<__gm__ float* __restrict>(ap);

    bool useUbPath = (tiling.incx == 1 && tiling.incy == 1 && tiling.n >= UB_THRESHOLD && tiling.n <= UB_XY_FLOATS);

    if (!useUbPath) {
        Sspr2FallbackToGm(tiling, xGm, yGm, apGm);
        return;
    }

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t colsPerBlk = (tiling.n + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint32_t colStart = static_cast<uint32_t>(blkIdx) * colsPerBlk;
    uint32_t colEnd = (colStart + colsPerBlk < tiling.n) ? (colStart + colsPerBlk) : tiling.n;
    if (colStart >= colEnd) {
        return;
    }

    uint32_t xyBase = (tiling.uplo == ACLBLAS_UPPER) ? 0 : colStart;
    uint32_t xyLen = (tiling.uplo == ACLBLAS_UPPER) ? colEnd : (tiling.n - colStart);

    uint32_t numThreads = colsPerBlk;
    numThreads = ((numThreads + SIMT_MIN_THREAD_NUM - 1) / SIMT_MIN_THREAD_NUM) * SIMT_MIN_THREAD_NUM;
    if (numThreads > SIMT_MAX_THREAD_NUM) {
        numThreads = SIMT_MAX_THREAD_NUM;
    }

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<Sspr2Ub<true>>(
            dim3{numThreads, 1, 1}, tiling.n, tiling.alpha, xGm, yGm, apGm, colStart, colEnd, xyBase, xyLen);
    } else {
        asc_vf_call<Sspr2Ub<false>>(
            dim3{numThreads, 1, 1}, tiling.n, tiling.alpha, xGm, yGm, apGm, colStart, colEnd, xyBase, xyLen);
    }
}

aclError sspr2_kernel_do(
    GM_ADDR x, GM_ADDR y, GM_ADDR ap, const Sspr2TilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    sspr2_kernel<<<numBlocks, nullptr, stream>>>(x, y, ap, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
