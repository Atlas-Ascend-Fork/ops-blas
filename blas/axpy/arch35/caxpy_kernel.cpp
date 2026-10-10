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

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "caxpy_kernel.h"
#include "caxpy_tiling_data.h"

namespace {

using namespace AscendC;

constexpr uint32_t CONTIGUOUS_TILE_COMPLEX = 4096U;
constexpr uint32_t CONTIGUOUS_TILE_FLOAT = CONTIGUOUS_TILE_COMPLEX * 2U;
// Three AoS buffers and six real/imaginary or temporary buffers share UB.
static_assert(
    (3U * CONTIGUOUS_TILE_FLOAT + 6U * CONTIGUOUS_TILE_COMPLEX) * sizeof(float) <= UB_SIZE,
    "Caxpy contiguous buffers exceed UB capacity");

class CaxpyContiguousAiv {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CaxpyTilingData& tiling, TPipe* pipe)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        totalN_ = tiling.totalN;
        numBlocks_ = tiling.numBlocks;
        alphaReal_ = tiling.alphaReal;
        alphaImag_ = tiling.alphaImag;

        pipe->InitBuffer(xAosBuf_, CONTIGUOUS_TILE_FLOAT * sizeof(float));
        pipe->InitBuffer(yAosBuf_, CONTIGUOUS_TILE_FLOAT * sizeof(float));
        pipe->InitBuffer(outAosBuf_, CONTIGUOUS_TILE_FLOAT * sizeof(float));
        pipe->InitBuffer(xRealBuf_, CONTIGUOUS_TILE_COMPLEX * sizeof(float));
        pipe->InitBuffer(xImagBuf_, CONTIGUOUS_TILE_COMPLEX * sizeof(float));
        pipe->InitBuffer(yRealBuf_, CONTIGUOUS_TILE_COMPLEX * sizeof(float));
        pipe->InitBuffer(yImagBuf_, CONTIGUOUS_TILE_COMPLEX * sizeof(float));
        pipe->InitBuffer(tmp0Buf_, CONTIGUOUS_TILE_COMPLEX * sizeof(float));
        pipe->InitBuffer(tmp1Buf_, CONTIGUOUS_TILE_COMPLEX * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t tileCount = totalN_ / CONTIGUOUS_TILE_COMPLEX;
        for (uint32_t tile = GetBlockIdx(); tile < tileCount; tile += numBlocks_) {
            ProcessTile(tile * CONTIGUOUS_TILE_COMPLEX, CONTIGUOUS_TILE_COMPLEX);
        }
    }

    __aicore__ inline void ProcessTail(uint32_t elementOffset, uint32_t elementCount)
    {
        if (elementCount != 0U && GetBlockIdx() == 0U) {
            ProcessTile(elementOffset, elementCount);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t elementOffset, uint32_t elementCount)
    {
        LocalTensor<float> xAos = xAosBuf_.Get<float>();
        LocalTensor<float> yAos = yAosBuf_.Get<float>();
        LocalTensor<float> xReal = xRealBuf_.Get<float>();
        LocalTensor<float> xImag = xImagBuf_.Get<float>();
        LocalTensor<float> yReal = yRealBuf_.Get<float>();
        LocalTensor<float> yImag = yImagBuf_.Get<float>();
        LocalTensor<float> tmp0 = tmp0Buf_.Get<float>();
        LocalTensor<float> tmp1 = tmp1Buf_.Get<float>();

        const uint64_t floatOffset = static_cast<uint64_t>(elementOffset) * 2U;
        const uint32_t floatCount = elementCount * 2U;
        DataCopy(xAos, xGm_[floatOffset], floatCount);
        DataCopy(yAos, yGm_[floatOffset], floatCount);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        DeInterleave(xReal, xImag, xAos, static_cast<int32_t>(floatCount));
        DeInterleave(yReal, yImag, yAos, static_cast<int32_t>(floatCount));
        PipeBarrier<PIPE_V>();

        Muls<float>(tmp0, xReal, alphaImag_, elementCount);
        Muls<float>(tmp1, xImag, alphaImag_, elementCount);
        Muls<float>(xReal, xReal, alphaReal_, elementCount);
        Muls<float>(xImag, xImag, alphaReal_, elementCount);
        Sub<float>(xReal, xReal, tmp1, elementCount);
        Add<float>(xImag, xImag, tmp0, elementCount);
        Add<float>(xReal, xReal, yReal, elementCount);
        Add<float>(xImag, xImag, yImag, elementCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> out0 = outAosBuf_.Get<float>();
        LocalTensor<float> out1 = outAosBuf_.GetWithOffset<float>(
            elementCount, elementCount * sizeof(float));
        Interleave(out0, out1, xReal, xImag, static_cast<int32_t>(elementCount));
        PipeBarrier<PIPE_ALL>();
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(yGm_[floatOffset], out0, floatCount);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    TBuf<TPosition::VECIN> xAosBuf_;
    TBuf<TPosition::VECIN> yAosBuf_;
    TBuf<TPosition::VECOUT> outAosBuf_;
    TBuf<TPosition::VECCALC> xRealBuf_;
    TBuf<TPosition::VECCALC> xImagBuf_;
    TBuf<TPosition::VECCALC> yRealBuf_;
    TBuf<TPosition::VECCALC> yImagBuf_;
    TBuf<TPosition::VECCALC> tmp0Buf_;
    TBuf<TPosition::VECCALC> tmp1Buf_;
    uint32_t totalN_ = 0;
    uint32_t numBlocks_ = 1;
    float alphaReal_ = 0.0F;
    float alphaImag_ = 0.0F;
};

__simt_callee__ __aicore__ inline uint64_t AbsStride(int32_t stride)
{
    return stride >= 0 ? static_cast<uint64_t>(stride) : static_cast<uint64_t>(-static_cast<int64_t>(stride));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CaxpySimt(
    uint32_t totalN, int32_t incx, int32_t incy, float alphaReal, float alphaImag, __gm__ const float* x,
    __gm__ float* y)
{
    uint64_t absIncx = AbsStride(incx);
    uint64_t absIncy = AbsStride(incy);
    uint64_t globalThread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t gridStride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = globalThread; i < totalN; i += gridStride) {
        uint64_t xElement = incx < 0 ? (static_cast<uint64_t>(totalN) - 1U - i) * absIncx : i * absIncx;
        uint64_t yElement = incy < 0 ? (static_cast<uint64_t>(totalN) - 1U - i) * absIncy : i * absIncy;
        uint64_t xOffset = xElement * 2U;
        uint64_t yOffset = yElement * 2U;

        float xReal = x[xOffset];
        float xImag = x[xOffset + 1U];
        float yReal = y[yOffset];
        float yImag = y[yOffset + 1U];
        volatile float realProduct = alphaReal * xReal;
        volatile float imagProduct = alphaImag * xImag;
        volatile float realDifference = realProduct - imagProduct;
        volatile float imagProduct0 = alphaReal * xImag;
        volatile float imagProduct1 = alphaImag * xReal;
        volatile float imagSum = imagProduct0 + imagProduct1;
        y[yOffset] = realDifference + yReal;
        y[yOffset + 1U] = imagSum + yImag;
    }
}

} // namespace

extern "C" __global__ __aicore__ void caxpy_kernel(GM_ADDR x, GM_ADDR y, CaxpyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.incx == 1 && tiling.incy == 1 && tiling.totalN >= CONTIGUOUS_TILE_COMPLEX) {
        TPipe pipe;
        CaxpyContiguousAiv op;
        op.Init(x, y, tiling, &pipe);
        op.Process();

        // The AIV path handles all complete tiles.  Keep a small scalar suffix
        // for an arbitrary trailing length instead of sending the whole
        // vector through the scalar path when n is not a multiple of the tile size.
        const uint32_t fullN = (tiling.totalN / CONTIGUOUS_TILE_COMPLEX) * CONTIGUOUS_TILE_COMPLEX;
        const uint32_t tailN = tiling.totalN - fullN;
        const uint32_t vectorTailN = tailN & ~7U;
        op.ProcessTail(fullN, vectorTailN);
        if (tailN != vectorTailN && GetBlockIdx() == 0U) {
            const uint64_t tailFloatOffset = static_cast<uint64_t>(fullN + vectorTailN) * 2U;
            __gm__ const float* tailX = reinterpret_cast<__gm__ const float*>(x) + tailFloatOffset;
            __gm__ float* tailY = reinterpret_cast<__gm__ float*>(y) + tailFloatOffset;
            for (uint32_t i = 0U; i < tailN - vectorTailN; ++i) {
                const uint32_t offset = i * 2U;
                const float xReal = tailX[offset];
                const float xImag = tailX[offset + 1U];
                const float yReal = tailY[offset];
                const float yImag = tailY[offset + 1U];
                tailY[offset] = tiling.alphaReal * xReal - tiling.alphaImag * xImag + yReal;
                tailY[offset + 1U] = tiling.alphaReal * xImag + tiling.alphaImag * xReal + yImag;
            }
        }
        return;
    }
    asc_vf_call<CaxpySimt>(
        dim3{tiling.threadCount, 1, 1}, tiling.totalN, tiling.incx, tiling.incy, tiling.alphaReal, tiling.alphaImag,
        reinterpret_cast<const __gm__ float*>(x), reinterpret_cast<__gm__ float*>(y));
}

void caxpy_kernel_do(uint8_t* x, uint8_t* y, const CaxpyTilingData& tiling, uint32_t numBlocks, void* stream)
{
    caxpy_kernel<<<numBlocks, nullptr, stream>>>(x, y, tiling);
}
