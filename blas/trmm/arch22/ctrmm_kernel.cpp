/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ctrmm_kernel_common.h"
#include "ctrmm_pack.h"
#include "ctrmm_repack.h"
#include "ctrmm_cube.h"
#include "ctrmm_finish.h"
#include "ctrmm_small.h"
#include "ctrmm_small_scalar.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void ctrmm_pack_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR ws, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (t.useCube && t.right && t.q <= kCubeTile) {
        if (t.qp == kCubeNarrowTile) {
            CtrmmPack<kCubeNarrowTile> pack;
            pack.Init(a, b, ws, t);
            pack.Run();
        } else {
            CtrmmPack<kCubeTile> pack;
            pack.Init(a, b, ws, t);
            pack.Run();
        }
        return;
    }
    CtrmmPack<> pack;
    pack.Init(a, b, ws, t);
    pack.Run();
}

extern "C" __global__ __aicore__ void ctrmm_repack_kernel(GM_ADDR ws, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (Exceptional(ws, t)) {
        return;
    }
    constexpr uint32_t L = kRepackElements;
    TPipe pipe;
    TBuf<TPosition::VECCALC> inBuf;
    TBuf<TPosition::VECCALC> outBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> selectBuf;
    pipe.InitBuffer(inBuf, (2U * L + kBlockFloats) * kFloatBytes);
    pipe.InitBuffer(outBuf, kExpandedComponents * L * kFloatBytes);
    pipe.InitBuffer(idxBuf, kExpandedComponents * L * kFloatBytes);
    pipe.InitBuffer(tmpBuf, kExpandedComponents * L * kFloatBytes);
    pipe.InitBuffer(selectBuf, L);
    auto in = inBuf.Get<float>();
    auto out = outBuf.Get<float>();
    auto idx = idxBuf.Get<uint32_t>();
    in.SetValue(2U * L, 0.0f);
    auto temporary = tmpBuf.Get<uint32_t>();
    auto mask = selectBuf.Get<uint8_t>();
    InitRepackIndices(out, idx, temporary, mask);
    RepackTiles(ws, t, in, out, idx);
}

extern "C" __global__ __aicore__ void ctrmm_cube_kernel(GM_ADDR ws, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (Exceptional(ws, t)) {
        return;
    }
    if (t.q > kCubeTile && t.q <= kCompactMaxOrder &&
        ((t.qp + kCubeTileRoundUp) / kCubeTile) * (t.rp / kCubeTile) <= t.cubeCores / 2U) {
        RunCube<kCubeNarrowTile>(ws, t);
    } else {
        RunCube<kCubeTile>(ws, t);
    }
}

extern "C" __global__ __aicore__ void ctrmm_finish_kernel(GM_ADDR c, GM_ADDR ws, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (t.useCube && t.right && t.q <= kCubeTile && !Exceptional(ws, t)) {
        if (t.r >= kWideOutputMinRhs && t.rp % kWideOutputWidth == 0U) {
            FinishShortRight<kWideOutputWidth>(c, ws, t);
        } else {
            FinishShortRight<kCubeTile>(c, ws, t);
        }
        return;
    }
    CtrmmFinish finish;
    finish.Init(c, ws, t);
    finish.Run();
}

extern "C" __global__ __aicore__ void ctrmm_small_scalar_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> input;
    TBuf<TPosition::VECCALC> output;
    pipe.InitBuffer(input, 192U);
    pipe.InitBuffer(output, 64U);
    auto al = input.Get<float>();
    auto bl = al[32];
    auto cl = output.Get<float>();
    GlobalTensor<float> ag;
    GlobalTensor<float> bg;
    GlobalTensor<float> cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    LoadTinyA(al, ag, t);
    DataCopyPad(
        bl, bg, DataCopyExtParams{static_cast<uint16_t>(t.r), t.q * 8U, (t.ldb - t.q) * 8U, 0, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    Fence<HardEvent::MTE2_S>();
    ComplexValue av[4];
    ComplexValue bv[4];
    const bool sensitive = ReadTinyInputs(al, bl, av, bv, t);
    for (uint32_t i = 0; i < t.q; ++i) {
        for (uint32_t j = 0; j < t.r; ++j) {
            ComplexValue y{};
            bool repair = sensitive;
            if (!repair) {
                repair = ComputeTinyFast(av, bv, t, i, j, y);
            }
            if (repair) {
                y = ComputeTinyReference(av, bv, t, i, j);
            }
            cl.SetValue(j * 8U + i * 2U, y.real);
            cl.SetValue(j * 8U + i * 2U + 1U, y.imag);
        }
    }
    Fence<HardEvent::S_MTE3>();
    DataCopyPad(cg, cl, DataCopyExtParams{static_cast<uint16_t>(t.r), t.q * 8U, 0, (t.ldc - t.q) * 8U, 0});
    Fence<HardEvent::MTE3_S>();
}

extern "C" __global__ __aicore__ void ctrmm_small8_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrmmSmall<8> small;
    small.Run(a, b, c, t);
}

extern "C" __global__ __aicore__ void ctrmm_small16_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrmmSmall<16> small;
    small.Run(a, b, c, t);
}

extern "C" __global__ __aicore__ void ctrmm_small32_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrmmSmall<32> small;
    small.Run(a, b, c, t);
}

extern "C" __global__ __aicore__ void ctrmm_small64_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrmmSmall<64> small;
    small.Run(a, b, c, t);
}

extern "C" __global__ __aicore__ void ctrmm_small128_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrmmSmall<128> small;
    small.Run(a, b, c, t);
}

extern "C" __global__ __aicore__ void ctrmm_zero_kernel(GM_ADDR c, CtrmmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, CHUNK * 2U * kFloatBytes);
    auto zero = buf.Get<float>();
    GlobalTensor<float> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    Duplicate(zero, 0.0f, 2U * CHUNK);
    Fence<HardEvent::V_MTE3>();
    const uint32_t rows = (t.m + CHUNK - 1U) / CHUNK;
    for (uint64_t task = GetBlockIdx(); task < static_cast<uint64_t>(rows) * t.n; task += GetBlockNum()) {
        const uint32_t row = static_cast<uint32_t>(task % rows) * CHUNK;
        const uint64_t col = task / rows;
        Write(output[2U * (col * t.ldc + row)], zero, 2U * Min(CHUNK, t.m - row));
    }
    Fence<HardEvent::MTE3_V>();
}

namespace {
void LaunchSmall(GM_ADDR a, GM_ADDR b, GM_ADDR c, const CtrmmTiling& t, void* stream)
{
    const uint32_t cols = t.q <= 16U && t.r <= 16U ? 8U : t.q <= 32U && t.r <= 32U ? 16U : 32U;
    const uint32_t rows = t.q <= 8U && t.r <= 8U ? 2U : 4U;
    const uint32_t blocks = (t.q + rows - 1U) / rows * ((t.r + cols - 1U) / cols);
    if (t.q <= 8U && t.r <= 8U) {
        ctrmm_small8_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
    } else if (t.q <= 16U && t.r <= 16U) {
        ctrmm_small16_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
    } else if (t.q <= 32U && t.r <= 32U) {
        ctrmm_small32_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
    } else if (t.q <= 64U && t.r <= 64U) {
        ctrmm_small64_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
    } else {
        ctrmm_small128_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
    }
}
} // namespace

void ctrmm_kernel_do(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const CtrmmTiling& t, void* stream)
{
    if (t.alphaReal == 0.0f && t.alphaImag == 0.0f) {
        ctrmm_zero_kernel<<<t.vectorCores, nullptr, stream>>>(c, t);
        return;
    }
    if (!t.right && t.q <= 2U && t.r <= 2U && b != c) {
        ctrmm_small_scalar_kernel<<<1, nullptr, stream>>>(a, b, c, t);
        return;
    }
    if (!t.right && t.q <= kSmallMaxOrder && t.r <= kSmallMaxOrder && b != c) {
        LaunchSmall(a, b, c, t, stream);
        return;
    }
    ctrmm_pack_kernel<<<t.vectorCores, nullptr, stream>>>(a, b, ws, t);
    if (t.useCube) {
        if (t.q > kCompactMaxOrder) {
            ctrmm_repack_kernel<<<t.vectorCores, nullptr, stream>>>(ws, t);
        }
        const uint32_t tileN =
            t.q > kCubeTile && t.q <= kCompactMaxOrder &&
                    ((t.qp + kCubeTileRoundUp) / kCubeTile) * (t.rp / kCubeTile) <= t.cubeCores / 2U ?
                kCubeNarrowTile :
                kCubeTile;
        const uint32_t tileM = kCubeTile;
        const uint64_t tiles = static_cast<uint64_t>((t.qp + tileM - 1U) / tileM) * (t.rp / tileN);
        const uint32_t blocks = tiles < t.cubeCores ? static_cast<uint32_t>(tiles) : t.cubeCores;
        ctrmm_cube_kernel<<<blocks, nullptr, stream>>>(ws, t);
    }
    ctrmm_finish_kernel<<<t.vectorCores, nullptr, stream>>>(c, ws, t);
}
