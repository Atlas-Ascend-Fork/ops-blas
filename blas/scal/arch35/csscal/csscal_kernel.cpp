/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * csscal_kernel.cpp - Kernel function registration for aclblasCsscal
 */

#include <cstdint>
#include <cstdlib>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "csscal_tiling.h"
#include "csscal_kernel.h"

using namespace AscendC;

// AIV kernel for incx == 1
__global__ __aicore__ void csscal_aiv_kernel(GM_ADDR x, uint32_t perCoreN,
    uint32_t remainder, uint32_t tileSize, float alpha)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CsscalContiguous(x, perCoreN, remainder, tileSize, alpha);
}

// SIMT kernel for incx != 1
__global__ __aicore__ void csscal_simt_kernel(GM_ADDR x, CsscalTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());

    // 由 blockIdx 推导本核分片（等价于原先 host 预填的 calCount/startOffset 数组）：
    // 前 remain 个核各多分 1 个元素
    uint32_t calNum = tdata.baseCount + (blockIdx < tdata.remain ? 1u : 0u);
    uint32_t startOffset =
        blockIdx * tdata.baseCount + (blockIdx < tdata.remain ? blockIdx : tdata.remain);

    if (calNum > 0) {
        uint32_t stride = static_cast<uint32_t>(tdata.incx);
        asc_vf_call<CsscalSimtCompute>(
            dim3{tdata.nthreads, 1, 1},
            calNum, startOffset,
            stride,
            tdata.alpha,
            reinterpret_cast<__gm__ float*>(x));
    }
}

void csscal_kernel_do(GM_ADDR x, GM_ADDR workSpace, const CsscalTilingData& tiling,
                      uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);

    if (tiling.incx == 1) {
        csscal_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(
            x, tiling.perCoreN, tiling.remainder, tiling.tileSize, tiling.alpha);
    } else {
        csscal_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, tiling);
    }
}
