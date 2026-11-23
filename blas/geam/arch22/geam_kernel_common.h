/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "kernel_operator.h"

namespace geam_arch22 {

using namespace AscendC;

struct GeamKernelContext {
    TPipe* pipe = nullptr;
    bool active = false;
    uint32_t colStart = 0U;
    uint32_t colEnd = 0U;
    GlobalTensor<float> aGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> cGm;
    TBuf<TPosition::VECCALC> bufA;
    TBuf<TPosition::VECCALC> bufB;
    TBuf<TPosition::VECCALC> bufC;
    TBuf<TPosition::VECCALC> bufScalar;
    TBuf<TPosition::VECCALC> bufWin;
};

template <class TilingT>
__aicore__ inline bool InitGeamKernelContext(
    GeamKernelContext& ctx, GM_ADDR a, GM_ADDR b, GM_ADDR c, const TilingT& tiling, TPipe* pipe, uint32_t elemFloats,
    uint32_t tileElems, uint32_t winMaxElems)
{
    ctx.pipe = pipe;
    const uint32_t blockIdx = GetBlockIdx();
    ctx.active =
        (tiling.usedCoreNum > 0U) && (blockIdx < tiling.usedCoreNum) && (tiling.m > 0U) && (tiling.n > 0U);
    if (!ctx.active) {
        return false;
    }
    if (blockIdx < tiling.remainder) {
        ctx.colStart = blockIdx * (tiling.perCoreN + 1U);
        ctx.colEnd = ctx.colStart + tiling.perCoreN + 1U;
    } else {
        ctx.colStart = blockIdx * tiling.perCoreN + tiling.remainder;
        ctx.colEnd = ctx.colStart + tiling.perCoreN;
    }
    ctx.active = (ctx.colStart < ctx.colEnd) && (ctx.colStart < tiling.n);
    if (!ctx.active) {
        return false;
    }
    ctx.aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    ctx.bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    ctx.cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));

    ctx.pipe->InitBuffer(ctx.bufA, tileElems * elemFloats * sizeof(float));
    ctx.pipe->InitBuffer(ctx.bufB, tileElems * elemFloats * sizeof(float));
    ctx.pipe->InitBuffer(ctx.bufC, tileElems * elemFloats * sizeof(float));
    ctx.pipe->InitBuffer(ctx.bufScalar, 64U * sizeof(float));
    if (tiling.opA != ACLBLAS_OP_N || tiling.opB != ACLBLAS_OP_N) {
        ctx.pipe->InitBuffer(ctx.bufWin, winMaxElems * elemFloats * sizeof(float));
    }
    return true;
}

} // namespace geam_arch22
