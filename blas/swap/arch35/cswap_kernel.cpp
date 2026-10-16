/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cswap_kernel.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"

namespace {

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CswapContiguousSimt(
    uint32_t totalN, __gm__ uint64_t* x, __gm__ uint64_t* y)
{
    uint32_t logicalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gridStride = gridDim.x * blockDim.x;
    for (; logicalIndex < totalN; logicalIndex += gridStride) {
        uint64_t xValue = x[logicalIndex];
        uint64_t yValue = y[logicalIndex];
        x[logicalIndex] = yValue;
        y[logicalIndex] = xValue;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CswapStridedSimt(
    uint32_t totalN, __gm__ uint64_t* x, int32_t incx, __gm__ uint64_t* y, int32_t incy)
{
    uint32_t logicalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gridStride = gridDim.x * blockDim.x;
    int64_t xStart = incx < 0 ? static_cast<int64_t>(totalN - 1) * -static_cast<int64_t>(incx) : 0;
    int64_t yStart = incy < 0 ? static_cast<int64_t>(totalN - 1) * -static_cast<int64_t>(incy) : 0;
    int64_t xIndex = xStart + static_cast<int64_t>(logicalIndex) * incx;
    int64_t yIndex = yStart + static_cast<int64_t>(logicalIndex) * incy;
    int64_t xGridStride = static_cast<int64_t>(gridStride) * incx;
    int64_t yGridStride = static_cast<int64_t>(gridStride) * incy;

    for (; logicalIndex < totalN; logicalIndex += gridStride) {
        uint64_t xValue = x[static_cast<uint64_t>(xIndex)];
        uint64_t yValue = y[static_cast<uint64_t>(yIndex)];
        x[static_cast<uint64_t>(xIndex)] = yValue;
        y[static_cast<uint64_t>(yIndex)] = xValue;
        xIndex += xGridStride;
        yIndex += yGridStride;
    }
}

} // namespace

extern "C" __global__ __aicore__ void cswap_contiguous_kernel(
    GM_ADDR x, GM_ADDR y, uint32_t totalN, uint32_t threadCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<CswapContiguousSimt>(
        dim3{threadCount, 1, 1}, totalN, reinterpret_cast<__gm__ uint64_t*>(x), reinterpret_cast<__gm__ uint64_t*>(y));
}

extern "C" __global__ __aicore__ void cswap_strided_kernel(
    GM_ADDR x, GM_ADDR y, uint32_t totalN, int32_t incx, int32_t incy, uint32_t threadCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<CswapStridedSimt>(
        dim3{threadCount, 1, 1}, totalN, reinterpret_cast<__gm__ uint64_t*>(x), incx,
        reinterpret_cast<__gm__ uint64_t*>(y), incy);
}

void cswap_kernel_do(
    uint8_t* x, uint8_t* y, uint32_t totalN, int32_t incx, int32_t incy, uint32_t threadCount, uint32_t numBlocks,
    void* stream)
{
    if (incx == 1 && incy == 1) {
        cswap_contiguous_kernel<<<numBlocks, nullptr, stream>>>(x, y, totalN, threadCount);
        return;
    }
    cswap_strided_kernel<<<numBlocks, nullptr, stream>>>(x, y, totalN, incx, incy, threadCount);
}
