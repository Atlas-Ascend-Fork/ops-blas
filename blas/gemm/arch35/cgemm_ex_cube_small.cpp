/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "tensor_api/tensor.h"
#include "cgemm_ex_kernel.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"

namespace cgemm_ex_blaze_small {

#include "cgemm_ex_cube_common.h"
using namespace AscendC;
using namespace AscendC::Te;
#include "cgemm_ex_cube_process.h"

__aicore__ inline void CubeOne(__gm__ float* a, __gm__ float* b, __gm__ float* c, const GemmTilingData& tiling)
{
    InitCubeFlags();
    if (tiling.baseM <= 32) {
        DispatchTrans<32, 32, 32, 256>(a, b, c, tiling);
    } else {
        DispatchTrans<64, 64, 32, 256>(a, b, c, tiling);
    }
    DrainCubeFlags();
}

} // namespace cgemm_ex_blaze_small

using namespace AscendC;

extern "C" __global__ __aicore__ __cube__ void cgemm_ex_cube_small_kernel(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    InitSocState();
    SetMMLayoutTransform(true);
    using namespace cgemm_ex_blaze_small;
    CubeOne(
        reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b), reinterpret_cast<__gm__ float*>(c),
        tiling);
    SetMMLayoutTransform(false);
}

void cgemm_ex_cube_small_do(
    uint32_t numBlocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmTilingData& tilingData)
{
    cgemm_ex_cube_small_kernel<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
}

#endif
