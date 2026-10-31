/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/kernel_utils.h"
#include "cann_ops_blas_common.h"
#include "scasum_tiling_data.h"

using namespace AscendC;

// ---------------------------------------------------------------------------
// SIMT strided reduction (handles both incx == 1 contiguous and incx != 1
// strided paths): per-thread partial sums over a complex float2 view, block-level
// tree reduction into workspace[blockIdx], then a single-kernel cross-core reduce
// (block 0 sums the per-core partials) writes the scalar result.
//
// A complex element (aclblasComplex == {float real; float imag;}) is exactly one
// float2 (8 bytes), so the interleaved vector is read as a contiguous float2 view:
// result = Σ (|Re| + |Im|) == Σ |float| over the float2 lanes, accessed with a
// single vectorized 8-byte load per element.
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void ScasumSimtCompute(
    uint32_t calNum, uint32_t startOffset, uint32_t stride, __gm__ const aclblasComplex* xGm, __gm__ float* partialOut,
    uint32_t blockDimPow2)
{
    if (calNum == 0) {
        return;
    }

    __ubuf__ float ubPartialSums[SIMT_MAX_THREAD_NUM];
    float partial = 0.0f;

    const __gm__ float2* xf2 = reinterpret_cast<const __gm__ float2*>(xGm);
    uint32_t i = threadIdx.x;
    uint32_t step = blockDim.x;
    for (; i + step < calNum; i += step * 2) {
        float2 c0 = xf2[(startOffset + i) * stride];
        float2 c1 = xf2[(startOffset + i + step) * stride];
        partial += (c0.x >= 0.0f ? c0.x : -c0.x) + (c0.y >= 0.0f ? c0.y : -c0.y) + (c1.x >= 0.0f ? c1.x : -c1.x) +
                   (c1.y >= 0.0f ? c1.y : -c1.y);
    }
    for (; i < calNum; i += step) {
        uint32_t idx = (startOffset + i) * stride;
        float2 c = xf2[idx];
        float re = c.x;
        float im = c.y;
        partial += (re >= 0.0f ? re : -re) + (im >= 0.0f ? im : -im);
    }

    // Warp-level reduction: asc_reduce_add collapses each 32-thread warp in one
    // hardware step (no syncthreads), then only the numWarps partials are reduced
    // once more in warp 0. This drops the block-wide syncthreads count from
    // log2(nthreads) to 1, cutting the SCALAR pipe time.
    float warpSum = asc_reduce_add(partial);
    if ((threadIdx.x & 31u) == 0) {
        ubPartialSums[threadIdx.x >> 5] = warpSum;
    }
    asc_syncthreads();

    if (threadIdx.x < 32) {
        uint32_t numWarps = blockDim.x >> 5;
        float v = (threadIdx.x < numWarps) ? ubPartialSums[threadIdx.x] : 0.0f;
        float s = asc_reduce_add(v);
        if (threadIdx.x == 0) {
            float total = s;
            for (uint32_t w = 32; w < numWarps; w++) {
                total += ubPartialSums[w];
            }
            partialOut[0] = total;
            // Ensure this GM store is visible device-wide BEFORE CrossCoreSetFlag.
            // The flag is ordered against the MTE3 pipe; a plain VF GM store is not
            // on that pipe, so without this fence block 0 may read stale workspace
            // values (reproduced as mass accuracy failures on multi-core 950PR).
            asc_threadfence();
        }
    }
}

__global__ __aicore__ void scasum_simt_kernel(GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workSpace, ScasumTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockIdx = GetBlockIdx();
    // R4: derive this block's range from total/perCore/remainder instead of
    // per-core arrays serialised by the host (same distribution, see host
    // CalcScasumTilingData derivation note).
    const uint32_t coreIdx = static_cast<uint32_t>(blockIdx);
    uint32_t myOffset = coreIdx * tdata.perCoreN + (coreIdx < tdata.remainder ? coreIdx : tdata.remainder);
    uint32_t calNum = tdata.perCoreN + (coreIdx < tdata.remainder ? 1U : 0U);

    if (calNum > 0) {
        dim3 vfDim{tdata.nthreads, 1, 1};
        asc_vf_call<ScasumSimtCompute>(
            vfDim, calNum, myOffset, static_cast<uint32_t>(tdata.incx),
            reinterpret_cast<__gm__ const aclblasComplex*>(inGM), reinterpret_cast<__gm__ float*>(workSpace) + blockIdx,
            RoundUpPow2(tdata.nthreads));
    }

    // Cross-core sync: SyncAll (all blocks reach it, including calNum==0 blocks).
    // CrossCoreSetFlag/WaitFlag was measured unreliable for making the SIMT
    // thread-path workspace writes visible to the scalar/UB read path (same root
    // cause as cdotu PR #370).
    SyncAll();

    if (blockIdx == 0) {
        // Read the per-core partials through the DMA (DataCopyPad) path into UB
        // and reduce there. Scalar GM access misses the writes made by the SIMT
        // threads (thread path bypasses the scalar DCache), which yields zeros
        // once the device code is optimized (-O2).
        TPipe pipe;
        TBuf<TPosition::VECCALC> wsBuf;
        TBuf<TPosition::VECCALC> reduceBuf;
        uint32_t aligned = ((tdata.useCoreNum + 7u) / 8u) * 8u;
        pipe.InitBuffer(wsBuf, aligned * sizeof(float));
        pipe.InitBuffer(reduceBuf, 64u * sizeof(float));
        LocalTensor<float> wsUb = wsBuf.Get<float>();
        LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        GlobalTensor<float> wsGm;
        wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), tdata.useCoreNum);

        DataCopyExtParams ext{1, static_cast<uint32_t>(tdata.useCoreNum * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(wsUb, wsGm[0], ext, pad);
        event_t evtIn = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evtIn);
        WaitFlag<HardEvent::MTE2_V>(evtIn);

        ReduceSum<float>(wsUb, wsUb, reduceTmp, static_cast<int32_t>(tdata.useCoreNum));
        event_t evtOut = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evtOut);
        WaitFlag<HardEvent::V_MTE3>(evtOut);

        GlobalTensor<float> resGm;
        resGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outGM), 1);
        DataCopyExtParams one{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(resGm[0], wsUb, one);
    }
}

void scasum_kernel_do(
    GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workSpace, const ScasumTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);

    scasum_simt_kernel<<<numBlocks, nullptr, aclStream>>>(inGM, outGM, workSpace, tiling);
}
