/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "cdgmm_compute.h"
#include "cdgmm_tiling_data.h"

constexpr uint32_t CDGMM_TILE_COMPLEX = 16384U;
constexpr uint32_t CDGMM_TILE_THREADS = 512U;

template <bool Left>
__simt_vf__ __aicore__ LAUNCH_BOUND(CDGMM_TILE_THREADS) inline void CdgmmTileCompute(
    const __ubuf__ float* input, __ubuf__ float* output, const __gm__ float* x, uint32_t first, uint32_t count,
    uint32_t m, uint32_t xLength, int32_t incx)
{
    const uint32_t firstIndex = first + threadIdx.x;
    uint32_t column = firstIndex / m;
    uint32_t row = firstIndex - column * m;
    const uint32_t columnStep = blockDim.x / m;
    const uint32_t rowStep = blockDim.x - columnStep * m;
    const uint32_t absIncx =
        incx < 0 ? static_cast<uint32_t>(-static_cast<int64_t>(incx)) : static_cast<uint32_t>(incx);
    for (uint32_t local = threadIdx.x; local < count; local += blockDim.x) {
        const uint32_t logicalX = Left ? row : column;
        const uint32_t xElement = (incx < 0 ? xLength - 1U - logicalX : logicalX) * absIncx;
        const CdgmmComplexValue aValue = {input[2U * local], input[2U * local + 1U]};
        const CdgmmComplexValue xValue = CdgmmLoadPair<false>(x + 2U * xElement);
        const CdgmmComplexValue result = CdgmmMultiply(aValue, xValue);
        output[2U * local] = result.real;
        output[2U * local + 1U] = result.imag;
        row += rowStep;
        column += columnStep;
        if (row >= m) {
            row -= m;
            ++column;
        }
    }
}

template <bool Left>
__aicore__ inline void CdgmmProcessTiled(GM_ADDR a, GM_ADDR x, GM_ADDR c, const CdgmmTilingData& tiling)
{
    using namespace AscendC;
    const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * tiling.elementsPerBlock;
    const uint64_t limit = begin + tiling.elementsPerBlock;
    const uint64_t end = limit < tiling.totalElements ? limit : tiling.totalElements;
    TPipe pipe;
    TBuf<TPosition::VECIN> inputBuffer;
    pipe.InitBuffer(inputBuffer, 2U * CDGMM_TILE_COMPLEX * sizeof(float));
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outputGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    auto input = inputBuffer.Get<float>();
    const DataCopyPadExtParams<float> noPad = {false, 0, 0, 0};
    for (uint64_t offset = begin; offset < end; offset += CDGMM_TILE_COMPLEX) {
        const uint32_t count =
            static_cast<uint32_t>(end - offset < CDGMM_TILE_COMPLEX ? end - offset : CDGMM_TILE_COMPLEX);
        const DataCopyExtParams copy = {1, count * 8U, 0, 0, 0};
        DataCopyPad(input, inputGm[2U * offset], copy, noPad);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        asc_vf_call<CdgmmTileCompute<Left>>(
            dim3{CDGMM_TILE_THREADS, 1U, 1U}, reinterpret_cast<const __ubuf__ float*>(input.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()), reinterpret_cast<const __gm__ float*>(x),
            static_cast<uint32_t>(offset), count, tiling.m, Left ? tiling.m : tiling.n, tiling.incx);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(outputGm[2U * offset], input, copy);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}
