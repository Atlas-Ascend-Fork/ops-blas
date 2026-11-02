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
#include "ctrmm_kernel_common.h"

namespace {
// Constant rectangle extents let the compiler specialize vector repeats and
// strides. Only the ordinary short RIGHT result uses this path; sensitive
// inputs still use the CtrmmFinish reference path.
template <uint32_t Size>
__aicore__ inline void InitShortOutputIndices(LocalTensor<float> data, LocalTensor<uint32_t> index)
{
    auto parity = data[2U * Size].ReinterpretCast<uint32_t>();
    CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t{0}, 2U * Size);
    Duplicate(parity.ReinterpretCast<int32_t>(), int32_t{1}, 2U * Size);
    PipeBarrier<PIPE_V>();
    And(parity.ReinterpretCast<uint16_t>(), index.ReinterpretCast<uint16_t>(), parity.ReinterpretCast<uint16_t>(),
        4U * Size);
    ShiftRight(index, index, uint32_t{1}, 2U * Size);
    PipeBarrier<PIPE_V>();
    ShiftLeft(index, index, uint32_t{2}, 2U * Size);
    ShiftLeft(parity, parity, kShortOutputPlaneByteShift, 2U * Size);
    PipeBarrier<PIPE_V>();
    Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(),
        2U * Size);
    Fence<HardEvent::V_MTE2>();
}

template <uint32_t Size>
__aicore__ inline LocalTensor<float> ScaleShortOutput(LocalTensor<float> data, const CtrmmTiling& t)
{
    auto result = data;
    if (t.alphaReal != 1.0f || t.alphaImag != 0.0f) {
        Muls(data[2U * Size], data, t.alphaReal, Size);
        Muls(data[3U * Size], data[Size], t.alphaImag, Size);
        PipeBarrier<PIPE_V>();
        Muls(data, data, t.alphaImag, Size);
        Muls(data[Size], data[Size], t.alphaReal, Size);
        PipeBarrier<PIPE_V>();
        Sub(data[2U * Size], data[2U * Size], data[3U * Size], Size);
        Add(data[3U * Size], data, data[Size], Size);
        PipeBarrier<PIPE_V>();
        result = data[2U * Size];
    }
    return result;
}

template <uint32_t Width>
__aicore__ inline void FinishShortRight(GM_ADDR c, GM_ADDR workspace, const CtrmmTiling& t)
{
    constexpr uint32_t Size = kShortOutputElements;
    constexpr uint32_t Rows = Size / Width;
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuf;
    TBuf<TPosition::VECCALC> packedBuf;
    TBuf<TPosition::VECCALC> indexBuf;
    pipe.InitBuffer(dataBuf, 4U * Size * sizeof(float));
    pipe.InitBuffer(packedBuf, 2U * Size * sizeof(float));
    pipe.InitBuffer(indexBuf, 2U * Size * sizeof(uint32_t));
    auto data = dataBuf.Get<float>();
    auto packed = packedBuf.Get<float>();
    auto index = indexBuf.Get<uint32_t>();
    InitShortOutputIndices<Size>(data, index);
    GlobalTensor<float> w;
    GlobalTensor<float> output;
    w.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    const uint32_t columns = (t.r + Width - 1U) / Width;
    const uint64_t tasks = static_cast<uint64_t>((t.q + Rows - 1U) / Rows) * columns;
    const uint64_t base = kCubeAPlanes * t.aPlane + kCubeOutputXPlane * t.xPlane;
    for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        const uint32_t row = task / columns * Rows;
        const uint32_t col = task % columns * Width;
        const uint32_t rows = Min(Rows, t.q - row);
        const uint32_t cols = Min(Width, t.r - col);
        const uint64_t offset = base + static_cast<uint64_t>(row) * t.rp + col;
        const DataCopyExtParams read{Rows, Width * 4U, (t.rp - Width) * 4U, 0, 0};
        DataCopyPad(data, w[offset], read, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        DataCopyPad(data[Size], w[offset + t.xPlane], read, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        Fence<HardEvent::MTE2_V>();
        auto result = ScaleShortOutput<Size>(data, t);
        Gather(packed, result, index, 0U, 2U * Size);
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(
            output[2U * (static_cast<uint64_t>(row) * t.ldc + col)], packed,
            DataCopyExtParams{
                static_cast<uint16_t>(rows), cols * 8U, (Width - (cols + 3U) / 4U * 4U) / 4U, (t.ldc - cols) * 8U, 0});
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::V_MTE2>();
    }
}

template <uint32_t RectSize>
__aicore__ inline LocalTensor<float> ScaleRectangle(
    LocalTensor<float> input, LocalTensor<float> output, uint32_t size, const CtrmmTiling& t)
{
    auto result = input;
    if (t.alphaReal != 1.0f || t.alphaImag != 0.0f) {
        Muls(output, input, t.alphaReal, size);
        Muls(output[RectSize], input[RectSize], t.alphaImag, size);
        Muls(input[2U * RectSize], input, t.alphaImag, size);
        Muls(input[3U * RectSize], input[RectSize], t.alphaReal, size);
        PipeBarrier<PIPE_V>();
        Sub(output, output, output[RectSize], size);
        Add(output[RectSize], input[2U * RectSize], input[3U * RectSize], size);
        PipeBarrier<PIPE_V>();
        result = output;
    }
    return result;
}

template <uint32_t RectSize>
__aicore__ inline void PackRectangle(
    LocalTensor<float> packed, LocalTensor<float> result, LocalTensor<uint32_t> index, uint32_t rectRows,
    uint32_t width, uint32_t size, bool right)
{
    if (right) {
        Gather(packed, result, index, 0U, 2U * size);
    } else {
        for (uint32_t rb = 0; rb < rectRows; rb += 8U) {
            uint64_t srcList[16];
            uint64_t dstList[16];
#pragma unroll
            for (uint32_t r = 0; r < 8U; ++r) {
                srcList[2U * r] = reinterpret_cast<uint64_t>(result[(rb + r) * width].GetPhyAddr());
                srcList[2U * r + 1U] = reinterpret_cast<uint64_t>(result[RectSize + (rb + r) * width].GetPhyAddr());
            }
#pragma unroll
            for (uint32_t c = 0; c < 8U; ++c) {
                dstList[2U * c] = reinterpret_cast<uint64_t>(packed[2U * c * rectRows + 2U * rb].GetPhyAddr());
                dstList[2U * c + 1U] =
                    reinterpret_cast<uint64_t>(packed[2U * c * rectRows + 2U * rb + 8U].GetPhyAddr());
            }
            TransDataTo5HD<float>(dstList, srcList, TransDataTo5HDParams(false, false, width / 8U, 2U * rectRows, 1U));
        }
    }
}

template <uint32_t RectSize>
__aicore__ inline void InitRectangleIndices(
    LocalTensor<uint32_t> index, LocalTensor<uint32_t> scratch, uint32_t rectRows, uint32_t width, bool right)
{
    auto parity = scratch[2U * RectSize];
    const uint32_t count = 2U * rectRows * width;
    CreateVecIndex(index.ReinterpretCast<int32_t>(), int32_t{0}, count);
    Duplicate(parity.ReinterpretCast<int32_t>(), int32_t{1}, count);
    PipeBarrier<PIPE_V>();
    And(parity.ReinterpretCast<uint16_t>(), index.ReinterpretCast<uint16_t>(), parity.ReinterpretCast<uint16_t>(),
        count * 2U);
    ShiftRight(index, index, uint32_t{1}, count);
    PipeBarrier<PIPE_V>();
    if (!right) {
        Duplicate(scratch.ReinterpretCast<int32_t>(), static_cast<int32_t>(rectRows - 1U), count);
        PipeBarrier<PIPE_V>();
        And(scratch.ReinterpretCast<uint16_t>(), index.ReinterpretCast<uint16_t>(), scratch.ReinterpretCast<uint16_t>(),
            count * 2U);
        ShiftRight(index, index, rectRows == 8U ? 3U : 5U, count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(scratch, scratch, width == 256U ? 8U : width == 128U ? 7U : 6U, count);
        PipeBarrier<PIPE_V>();
        Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), scratch.ReinterpretCast<int32_t>(),
            count);
        PipeBarrier<PIPE_V>();
    }
    ShiftLeft(index, index, uint32_t{2}, count);
    ShiftLeft(parity, parity, kRectanglePlaneByteShift, count);
    PipeBarrier<PIPE_V>();
    Add(index.ReinterpretCast<int32_t>(), index.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), count);
    Fence<HardEvent::V_MTE2>();
}

} // namespace
