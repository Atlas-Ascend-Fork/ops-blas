/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef GEMMEX_CATLASS_ENABLED
#include "kernel_operator.h"
#include "gemm_ex_tiling.h"
#include "gemm_ex_dma_copy.h"
#include "gemm_ex_tile_primitives.h"

using namespace AscendC;

using GemmExNnArgs263 = gemm_ex::ShortArgs;

__global__ __aicore__ __cube__ void gemmex_nn263_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR cGm, GemmExNnArgs263 t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetHF32Mode(HF32Mode::DISABLE);
    gemm_ex::ShortNnTile<GemmExNnArgs263> tile(t);
    tile.Run(a, b, cGm);
}

void gemmex_nn263_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t, uint32_t ldc)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 367) / 368;
    const auto q = gemm_ex::MakeShortArgs(t, ldc, columns);
    gemmex_nn263_kernel<<<blocks, nullptr, stream>>>(a, b, c, q);
}

using namespace AscendC;
// Controlled no-count variant of the pinned SDK vreducev2 implementation.
// This compaction has a known result length, so no scalar count is consumed.

template <uint32_t COLUMNS>
__global__ __aicore__ void gemmex_nn263_output_kernel(GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunShortDrain<COLUMNS, 2>(inputGm, outputGm, m, n);
}
void gemmex_nn263_output_do(uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t)
{
    const uint32_t use =
        (blocks < (static_cast<uint32_t>(t.n) + 15) / 16 ? blocks : (static_cast<uint32_t>(t.n) + 15) / 16);
    const uint32_t cols = (static_cast<uint32_t>(t.m) + 367) / 368;
    if (cols == 1) {
        gemmex_nn263_output_kernel<1><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 2) {
        gemmex_nn263_output_kernel<2><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 3) {
        gemmex_nn263_output_kernel<3><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 4) {
        gemmex_nn263_output_kernel<4><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else {
        gemmex_nn263_output_kernel<5><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    }
}

#endif
