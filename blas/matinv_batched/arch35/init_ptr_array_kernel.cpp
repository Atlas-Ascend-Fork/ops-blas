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
 * \file init_ptr_array_kernel.cpp
 * \brief Kernel-side implementation for initializing workspace pointer array (SIMT).
 *
 * Populates matTmpPtrArray so that each entry points to the corresponding n*n block
 * inside the flat matTmpBuf buffer.
 */

#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr uint32_t INIT_PTR_ARRAY_SIMT_MAX_THREADS = 32;

// VF function: each thread handles a strided subset of the batch pointer entries in parallel.
__simt_vf__ __aicore__ LAUNCH_BOUND(INIT_PTR_ARRAY_SIMT_MAX_THREADS) inline void InitPtrArrayVf(
    uint32_t n, uint32_t numBatch, uint32_t startBatch, GM_ADDR ptrArrayBase, GM_ADDR flatBufBase)
{
    auto* ptrArr = reinterpret_cast<__gm__ uint64_t*>(ptrArrayBase);
    auto* matBuf = reinterpret_cast<__gm__ float*>(flatBufBase);
    uint64_t stride = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);
    for (uint32_t b = threadIdx.x; b < numBatch; b += blockDim.x) {
        uint32_t batchIdx = startBatch + b;
        ptrArr[batchIdx] = reinterpret_cast<uint64_t>(matBuf + static_cast<uint64_t>(batchIdx) * stride);
    }
}

extern "C" __global__ __aicore__ void init_ptr_array_kernel(
    GM_ADDR ptrArray, GM_ADDR flatBuf, uint32_t n, uint32_t batchSize, uint32_t numBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t blockId = GetBlockIdx();
    if (blockId >= numBlocks) {
        return;
    }

    uint32_t batchPerCore = (batchSize - 1) / numBlocks + 1;
    uint32_t startBatch = blockId * batchPerCore;
    uint32_t endBatch = startBatch + batchPerCore;
    if (endBatch > batchSize) {
        endBatch = batchSize;
    }
    uint32_t numBatch = endBatch - startBatch;
    if (numBatch == 0) {
        return;
    }

    asc_vf_call<InitPtrArrayVf>(
        dim3{INIT_PTR_ARRAY_SIMT_MAX_THREADS, 1, 1}, n, numBatch, startBatch, ptrArray, flatBuf);
}

// issue #1180: launch through aclrtGetFuncBySymbol + aclrtLaunchKernelWithHostArgs
// so the runtime status of the launch is returned to the caller: the <<<>>>
// statement form discards it. Same pattern as the gemm_batched arch35 launchers.
aclError init_ptr_array_kernel_do(
    uint8_t* ptrArray, uint8_t* flatBuf, uint32_t n, uint32_t batchSize, uint32_t numBlocks, void* stream)
{
    aclrtFuncHandle function = nullptr;
    const aclError lookup = aclrtGetFuncBySymbol(reinterpret_cast<const void*>(init_ptr_array_kernel), &function);
    if (lookup != ACL_SUCCESS) {
        return lookup;
    }
    struct Arguments {
        GM_ADDR ptrArray;
        GM_ADDR flatBuf;
        uint32_t n;
        uint32_t batchSize;
        uint32_t numBlocks;
    } args{ptrArray, flatBuf, n, batchSize, numBlocks};
    static_assert(offsetof(Arguments, n) == 2U * sizeof(uint8_t*), "kernel ABI mismatch");
    return aclrtLaunchKernelWithHostArgs(function, numBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0U);
}
