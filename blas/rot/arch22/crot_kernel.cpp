/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "acl/acl.h"
#include "crot_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t TILE_FLOATS = 2 * CROT_TILE_COMPLEX;
constexpr uint32_t TILE_BYTES = TILE_FLOATS * sizeof(float);
constexpr uint32_t DATA_BLOCK_BYTES = 32U; // TransDataTo5HD address stride for one hardware DataBlock.

class CrotKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CrotTilingData& tiling, TPipe& pipe)
    {
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        t_ = tiling;
        // 160 KiB UB: two input/output slots and two planar input vectors.
        pipe.InitBuffer(input_, 2, 2 * TILE_BYTES);
        pipe.InitBuffer(output_, 2, 2 * TILE_BYTES);
        pipe.InitBuffer(scratch_, 2 * TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        const uint32_t tiles = (t_.n + CROT_TILE_COMPLEX - 1) / CROT_TILE_COMPLEX;
        const uint32_t blocks = GetBlockNum();
        if (blocks == 0) {
            return;
        }
        const uint32_t block = GetBlockIdx();
        const uint32_t base = tiles / blocks;
        const uint32_t extra = tiles % blocks;
        const uint32_t first = block * base + (block < extra ? block : extra);
        const uint32_t end = first + base + (block < extra ? 1U : 0U);
        for (uint32_t tile = first; tile < end; ++tile) {
            const uint32_t offset = tile * CROT_TILE_COMPLEX;
            const uint32_t count = t_.n - offset < CROT_TILE_COMPLEX ? t_.n - offset : CROT_TILE_COMPLEX;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        auto in = input_.AllocTensor<float>();
        const DataCopyExtParams params{1, count * 8U, 0, 0, 0};
        const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(in, x_[static_cast<uint64_t>(offset) * 2], params, pad);
        DataCopyPad(in[TILE_FLOATS], y_[static_cast<uint64_t>(offset) * 2], params, pad);
        input_.EnQue(in);
    }

    template <bool inverse>
    __aicore__ inline void TransposePairs(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count)
    {
        uint64_t sources[16];
        uint64_t targets[16];
        const uint64_t srcBase = reinterpret_cast<uint64_t>(src.GetPhyAddr());
        const uint64_t dstBase = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
        for (uint32_t i = 0; i < 16; ++i) {
            if constexpr (inverse) {
                sources[i] = srcBase + ((i % 8) * 16 + (i / 8) * 8) * sizeof(float);
                targets[i] = dstBase + ((i / 2) * 8 + (i % 2) * 64) * sizeof(float);
            } else {
                sources[i] = srcBase + i * DATA_BLOCK_BYTES;
                targets[i] = dstBase + i * DATA_BLOCK_BYTES;
            }
        }
        // A 16x8 float transpose produces alternating groups of 16 real and
        // 16 imaginary components. All addresses remain 32B aligned.
        const uint16_t repeats = count / 128;
        const uint16_t stride = repeats == 1 ? 0 : 16;
        TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams(false, false, repeats, stride, stride));
    }

    __aicore__ inline void Compute(uint32_t elements)
    {
        auto in = input_.DeQue<float>();
        auto out = output_.AllocTensor<float>();
        auto x = scratch_.Get<float>();
        auto y = x[TILE_FLOATS];
        auto nx = in;
        auto ny = in[TILE_FLOATS];
        auto product = out;
        auto cross = out[TILE_FLOATS];
        // Only valid elements are copied out; vector padding stays inside UB.
        const uint32_t count = (2 * elements + 127U) & ~127U;
        // Input AoS storage is reused for the planar results after both inputs
        // have been transposed. Output storage holds intermediate products
        // until the final inverse transpose. No extra full-tile copies are needed.
        TransposePairs<false>(x, in, count);
        TransposePairs<false>(y, in[TILE_FLOATS], count);
        PipeBarrier<PIPE_V>();
        Muls(nx, y, t_.sr, count);
        Muls(ny, x, t_.sr, count);
        PipeBarrier<PIPE_V>();
        // Select real groups; adding 16 floats selects the matching imaginary
        // groups. Separate products and sums keep the reference FP32 ordering.
        uint64_t realMask[2] = {0x0000FFFF0000FFFFULL, 0};
        const uint8_t repeats = count / 64;
        const UnaryRepeatParams unary(1, 1, 8, 8);
        const BinaryRepeatParams binary(1, 1, 1, 8, 8, 8);
        Muls(product, y[16], t_.si, realMask, repeats, unary);
        Muls(cross, x[16], t_.si, realMask, repeats, unary);
        PipeBarrier<PIPE_V>();
        Sub(nx, nx, product, realMask, repeats, binary);
        Add(ny, ny, cross, realMask, repeats, binary);
        PipeBarrier<PIPE_V>();
        Muls(product[16], y, t_.si, realMask, repeats, unary);
        Muls(cross[16], x, t_.si, realMask, repeats, unary);
        PipeBarrier<PIPE_V>();
        Add(nx[16], nx[16], product[16], realMask, repeats, binary);
        Sub(ny[16], ny[16], cross[16], realMask, repeats, binary);
        PipeBarrier<PIPE_V>();
        Muls(product, x, t_.c, count);
        Muls(cross, y, t_.c, count);
        PipeBarrier<PIPE_V>();
        Add(nx, product, nx, count);
        Sub(ny, cross, ny, count);
        PipeBarrier<PIPE_V>();
        TransposePairs<true>(out, nx, count);
        TransposePairs<true>(out[TILE_FLOATS], ny, count);
        PipeBarrier<PIPE_V>();
        input_.FreeTensor(in);
        output_.EnQue(out);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        auto out = output_.DeQue<float>();
        const DataCopyExtParams params{1, count * 8U, 0, 0, 0};
        DataCopyPad(x_[static_cast<uint64_t>(offset) * 2], out, params);
        DataCopyPad(y_[static_cast<uint64_t>(offset) * 2], out[TILE_FLOATS], params);
        output_.FreeTensor(out);
    }

    GlobalTensor<float> x_;
    GlobalTensor<float> y_;
    CrotTilingData t_;
    TQue<QuePosition::VECIN, 2> input_;
    TQue<QuePosition::VECOUT, 2> output_;
    TBuf<TPosition::VECCALC> scratch_;
};

__aicore__ inline uint64_t CrotIndex(uint32_t i, int32_t inc, uint32_t n)
{
    const int64_t magnitude = inc < 0 ? -static_cast<int64_t>(inc) : inc;
    return static_cast<uint64_t>(inc < 0 ? n - 1 - i : i) * magnitude * 2U;
}

__aicore__ inline void CrotStrided(GM_ADDR x, GM_ADDR y, const CrotTilingData& t, TPipe& pipe)
{
    constexpr uint32_t PASS = 256;
    constexpr uint32_t SLOT = 8; // each 8-byte complex output starts at a 32B UB boundary
    GlobalTensor<float> gx;
    GlobalTensor<float> gy;
    gx.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    gy.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    TBuf<TPosition::VECCALC> staging;
    pipe.InitBuffer(staging, 2 * PASS * SLOT * sizeof(float));
    auto ox = staging.Get<float>();
    auto oy = ox[PASS * SLOT];
    const DataCopyExtParams params{1, 2 * sizeof(float), 0, 0, 0};
    for (uint32_t base = 0; base < t.n; base += PASS) {
        const uint32_t count = t.n - base < PASS ? t.n - base : PASS;
        for (uint32_t i = 0; i < count; ++i) {
            const uint64_t ix = CrotIndex(base + i, t.incx, t.n);
            const uint64_t iy = CrotIndex(base + i, t.incy, t.n);
            const float xr = gx.GetValue(ix);
            const float xi = gx.GetValue(ix + 1);
            const float yr = gy.GetValue(iy);
            const float yi = gy.GetValue(iy + 1);
            // Separate products preserve FP32 operation order (no fused multiply-add).
            volatile float cxr = t.c * xr;
            volatile float cxi = t.c * xi;
            volatile float cyr = t.c * yr;
            volatile float cyi = t.c * yi;
            volatile float sryr = t.sr * yr;
            volatile float siyi = t.si * yi;
            volatile float sryi = t.sr * yi;
            volatile float siyr = t.si * yr;
            volatile float srxr = t.sr * xr;
            volatile float sixi = t.si * xi;
            volatile float srxi = t.sr * xi;
            volatile float sixr = t.si * xr;
            ox.SetValue(i * SLOT, cxr + (sryr - siyi));
            ox.SetValue(i * SLOT + 1, cxi + (sryi + siyr));
            oy.SetValue(i * SLOT, cyr - (srxr + sixi));
            oy.SetValue(i * SLOT + 1, cyi - (srxi - sixr));
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < count; ++i) {
            DataCopyPad(gx[CrotIndex(base + i, t.incx, t.n)], ox[i * SLOT], params);
            DataCopyPad(gy[CrotIndex(base + i, t.incy, t.n)], oy[i * SLOT], params);
        }
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace

__global__ __aicore__ void crot_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR c, GM_ADDR s, CrotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (c != nullptr) {
        tiling.c = *reinterpret_cast<__gm__ float*>(c);
    }
    if (s != nullptr) {
        tiling.sr = reinterpret_cast<__gm__ float*>(s)[0];
        tiling.si = reinterpret_cast<__gm__ float*>(s)[1];
    }
    if (tiling.incx == 1 && tiling.incy == 1) {
        CrotKernel op;
        op.Init(x, y, tiling, pipe);
        op.Process();
    } else {
        CrotStrided(x, y, tiling, pipe);
    }
}

aclError crot_kernel_do(
    uint8_t* x, uint8_t* y, const uint8_t* c, const uint8_t* s, CrotTilingData tiling, uint32_t blocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    crot_kernel<<<blocks, nullptr, stream>>>(x, y, const_cast<uint8_t*>(c), const_cast<uint8_t*>(s), tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
