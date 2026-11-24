/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cdgmm_simt.h"
#include "cdgmm_tiled.h"

extern "C" __global__ __aicore__ void cdgmm_aiv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR c, CdgmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.compact != 0U && tiling.wideIndex == 0U && tiling.totalElements >= 65536U) {
        if (tiling.left != 0U) {
            CdgmmProcessTiled<true>(a, x, c, tiling);
        } else {
            CdgmmProcessTiled<false>(a, x, c, tiling);
        }
        return;
    }
    // Small, padded and wide-index cases use direct GM access.
    if (tiling.wideIndex != 0U) {
        DispatchCdgmm<uint64_t>(a, x, c, tiling);
    } else {
        DispatchCdgmm<uint32_t>(a, x, c, tiling);
    }
}

void cdgmm_kernel_do(const uint8_t* a, const uint8_t* x, uint8_t* c, const CdgmmTilingData& tiling, void* stream)
{
    const uint32_t dynamicUbBytes = tiling.compact != 0U && tiling.wideIndex == 0U && tiling.totalElements >= 65536U ?
                                        2U * CDGMM_TILE_COMPLEX * sizeof(float) :
                                        0U;
    cdgmm_aiv_kernel<<<tiling.blockCount, dynamicUbBytes, stream>>>(
        const_cast<uint8_t*>(a), const_cast<uint8_t*>(x), c, tiling);
}
