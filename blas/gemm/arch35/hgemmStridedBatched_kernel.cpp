/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#define ASCENDC_CUBE_ONLY
#include "common/helper/gemm_strided_cube_arch35.h"
#include "hgemmStridedBatched_kernel.h"

#include "hgemmStridedBatched_pipeline.h"
#include "hgemmStridedBatched_grouped.h"

__cube__ __global__ void hgemm_batch_direct_pipeline(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmStridedBatchedExTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    if (t.baseM == HGEMM_TILE_LARGE)
        hgemm_pipeline::Dispatch<HGEMM_TILE_LARGE, HGEMM_REDUCTION_SHALLOW, half, QuantMode_t::F322F16>(a, b, c, t);
    else if (t.baseM == HGEMM_TILE_SMALL)
        hgemm_pipeline::Dispatch<HGEMM_TILE_SMALL, HGEMM_REDUCTION_SHALLOW, half, QuantMode_t::F322F16>(a, b, c, t);
    else
        hgemm_pipeline::Dispatch<HGEMM_TILE_STANDARD, HGEMM_REDUCTION_STANDARD, half, QuantMode_t::F322F16>(a, b, c, t);
}

__cube__ __global__ void hgemm_batch_temporary_pipeline(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmStridedBatchedExTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    if (t.baseM == HGEMM_TILE_LARGE)
        hgemm_pipeline::Dispatch<HGEMM_TILE_LARGE, HGEMM_REDUCTION_SHALLOW, float, QuantMode_t::NoQuant>(a, b, c, t);
    else if (t.baseM == HGEMM_TILE_SMALL)
        hgemm_pipeline::Dispatch<HGEMM_TILE_SMALL, HGEMM_REDUCTION_SHALLOW, float, QuantMode_t::NoQuant>(a, b, c, t);
    else
        hgemm_pipeline::Dispatch<HGEMM_TILE_STANDARD, HGEMM_REDUCTION_STANDARD, float, QuantMode_t::NoQuant>(
            a, b, c, t);
}

void HgemmBatchCube(
    uint32_t blocks, void* stream, uint8_t* left, uint8_t* right, uint8_t* out, const GemmStridedBatchedExTilingData& t,
    bool direct)
{
    if (direct)
        hgemm_batch_direct_pipeline<<<blocks, nullptr, stream>>>(left, right, out, t);
    else
        hgemm_batch_temporary_pipeline<<<blocks, nullptr, stream>>>(left, right, out, t);
}

__cube__ __global__ void hgemm_batch_grouped_pipeline(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmStridedBatchedExTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    if (t.m == HGEMM_GROUPED_SIZES[0])
        hgemm_grouped::Dispatch<HGEMM_GROUPED_SIZES[0]>(a, b, c, t);
    else if (t.m == HGEMM_GROUPED_SIZES[1])
        hgemm_grouped::Dispatch<HGEMM_GROUPED_SIZES[1]>(a, b, c, t);
    else if (t.m == HGEMM_GROUPED_SIZES[2])
        hgemm_grouped::Dispatch<HGEMM_GROUPED_SIZES[2]>(a, b, c, t);
    else if (t.m == HGEMM_GROUPED_SIZES[3])
        hgemm_grouped::Dispatch<HGEMM_GROUPED_SIZES[3]>(a, b, c, t);
}

void HgemmBatchGrouped(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, const GemmStridedBatchedExTilingData& t)
{
    hgemm_batch_grouped_pipeline<<<blocks, nullptr, stream>>>(a, b, c, t);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(HGEMM_EPILOGUE_THREADS) inline void HgemmCombineVf(
    __gm__ float* tmp, __gm__ uint16_t* c, HgemmBatchEpilogue t, uint32_t block, uint32_t blocks)
{
    const int64_t matrix = static_cast<int64_t>(t.m) * t.n;
    const int64_t total = matrix * t.batchCount;
    const int64_t thread = static_cast<int64_t>(block) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(blocks) * blockDim.x;
    for (int64_t i = thread; i < total; i += step) {
        const int64_t b = i / matrix, q = i % matrix, col = q / t.m, row = q % t.m;
        const int64_t offset = b * t.strideC + col * t.ldc + row;
        float out = 0.0f;
        if (!t.scaleOnly)
            out = t.alpha * tmp[b * t.workspaceStride + col * t.workspaceLdc + row];
        if (t.beta != 0.0f)
            out += t.beta * __half2float(reinterpret_cast<__gm__ half*>(c)[offset]);
        reinterpret_cast<__gm__ half*>(c)[offset] = __float2half_rn(out);
    }
}

__global__ __aicore__ void hgemm_batch_combine(GM_ADDR tmp, GM_ADDR c, HgemmBatchEpilogue t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<HgemmCombineVf>(
        dim3(HGEMM_EPILOGUE_THREADS), reinterpret_cast<__gm__ float*>(tmp), reinterpret_cast<__gm__ uint16_t*>(c), t,
        AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}

void HgemmBatchCombine(uint32_t blocks, void* stream, uint8_t* tmp, uint8_t* c, const HgemmBatchEpilogue& t)
{
    hgemm_batch_combine<<<blocks, nullptr, stream>>>(tmp, c, t);
}
