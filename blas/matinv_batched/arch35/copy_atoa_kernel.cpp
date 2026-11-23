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
 * \file copy_atoa_kernel.cpp
 * \brief Kernel-side implementation for matrix copy between pointer arrays (SIMT).
 *
 * Copies each A[i] (column-major, srcLda x n) into dst[i] (column-major, dstLda x n).
 * Used by matinv_batched host as Step 1 (A -> Ainv) and Step 3 (Ainv LU -> matTmpBuf).
 */

#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "matinv_batched_tiling_data.h"

using namespace AscendC;

constexpr uint32_t COPY_ATOA_SIMT_MAX_THREADS = 256;

// Reads a device-side float pointer from the pointer array stored in GM.
__simt_callee__ __aicore__ inline __gm__ float* ReadPtrFromArray(GM_ADDR arrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(arrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ float*>(rawAddr);
}

// Reads a device-side const float pointer from a const pointer array stored in GM.
__simt_callee__ __aicore__ inline const __gm__ float* ReadPtrFromConstArray(
    const __gm__ uint8_t* arrayBase, uint32_t batchIdx)
{
    const __gm__ uint64_t* addrSlot = reinterpret_cast<const __gm__ uint64_t*>(arrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<const __gm__ float*>(rawAddr);
}

// Vector function: copies matrices from srcArray to dstArray, column-major with distinct ldas.
__simt_vf__ __aicore__ LAUNCH_BOUND(COPY_ATOA_SIMT_MAX_THREADS) inline void CopyMatrixVf(
    uint32_t n, uint32_t srcLda, uint32_t dstLda, uint32_t numBatch, uint32_t startBatch,
    const __gm__ uint8_t* srcArrayBase, GM_ADDR dstArrayBase)
{
    uint32_t totalElements = n * n;

    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        const __gm__ float* src = ReadPtrFromConstArray(srcArrayBase, batchIdx);
        __gm__ float* dst = ReadPtrFromArray(dstArrayBase, batchIdx);

        // Each thread handles a subset of elements (column-major layout).
        // Avoid integer div/mod in the hot loop by maintaining row/col incrementally.
        uint32_t row = threadIdx.x % n;
        uint32_t col = threadIdx.x / n;
        uint32_t stride = blockDim.x;
        uint32_t rowInc = stride % n;
        uint32_t colInc = stride / n;
        for (uint32_t idx = threadIdx.x; idx < totalElements; idx += stride) {
            dst[row + col * dstLda] = src[row + col * srcLda];
            row += rowInc;
            col += colInc;
            if (row >= n) {
                row -= n;
                col++;
            }
        }
        asc_syncthreads();
    }
}

extern "C" __global__ __aicore__ void copy_atoa_kernel(
    const __gm__ uint8_t* srcArray, GM_ADDR dstArray, const SmatinvBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.usedCoreNum == 0) {
        return;
    }

    uint32_t blockId = GetBlockIdx();
    if (blockId >= tiling.usedCoreNum) {
        return;
    }

    uint32_t startBatch = blockId * tiling.batchPerCore;
    uint32_t numBatch = (blockId == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;
    if (numBatch == 0) {
        return;
    }

    asc_vf_call<CopyMatrixVf>(
        dim3{COPY_ATOA_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, tiling.ldaInv, numBatch, startBatch, srcArray,
        dstArray);
}

// issue #1180: launch through aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs
// so the runtime status of the launch is returned to the caller: the <<<>>>
// statement form discards it. Same pattern as the gemm_batched arch35 launchers.
aclError copy_kernel_do(
    const uint8_t* srcArray, uint8_t* dstArray, const SmatinvBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(copy_atoa_kernel), &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        const uint8_t* srcArray;
        uint8_t* dstArray;
        SmatinvBatchedTilingData tiling;
    } args{srcArray, dstArray, tiling};
    static_assert(offsetof(Arguments, tiling) == 2U * sizeof(uint8_t*), "kernel ABI mismatch");
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}
