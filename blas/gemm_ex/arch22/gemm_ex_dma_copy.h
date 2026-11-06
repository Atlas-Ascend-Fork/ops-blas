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
#include "kernel_operator.h"

namespace gemm_ex {
template <typename T>
__aicore__ inline AscendC::GlobalTensor<T> BindGlobal(__gm__ T* address)
{
    AscendC::GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(address);
    return tensor;
}

using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::PipeBarrier;
using AscendC::SetFlag;
using AscendC::WaitFlag;

struct DmaCopyParams {
    uint32_t width;
    uint32_t destLd;
    uint32_t sourceLd;
    uint32_t first;
    uint32_t end;
    uint32_t batch;
    uint32_t waves;
};
struct DmaCopyBuffers {
    AscendC::LocalTensor<float> local0;
    AscendC::LocalTensor<float> local1;
    AscendC::GlobalTensor<float> source;
    AscendC::GlobalTensor<float> dest;
};
__aicore__ inline uint32_t Smaller(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline void WaitCopyReuse(bool slot)
{
    if (slot) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}
__aicore__ inline void FinishCopyReuse(uint32_t waves)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    if (waves >= 2) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <bool AllowDenseRead>
class StridedDmaCopy : private DmaCopyParams, private DmaCopyBuffers {
public:
    __aicore__ inline StridedDmaCopy(DmaCopyParams parameters, DmaCopyBuffers buffers)
        : DmaCopyParams(parameters), DmaCopyBuffers(buffers)
    {}
    __aicore__ inline void Run()
    {
        for (uint32_t wave = 0; wave < waves; ++wave) {
            const bool slot = wave & 1U;
            if (wave >= 2) {
                WaitCopyReuse(slot);
            }
            const uint32_t col = first + wave * batch;
            const uint32_t count = Smaller(batch, end - col);
            Load(slot, col, count);
            if (wave) {
                const uint32_t prev = col - batch;
                const uint32_t prevCount = Smaller(batch, end - prev);
                Store(!slot, prev, prevCount);
            }
        }
        const uint32_t last = first + (waves - 1) * batch;
        const uint32_t count = Smaller(batch, end - last);
        Store((waves - 1) & 1U, last, count);
        FinishCopyReuse(waves);
    }

private:
    __aicore__ inline void Read(AscendC::LocalTensor<float> local, uint32_t col, uint32_t count)
    {
        const uint32_t dma = (width + 7) / 8 * 8;
        const AscendC::DataCopyExtParams load{static_cast<uint16_t>(count), width * 4U, (sourceLd - width) * 4U, 0, 0};
        const AscendC::DataCopyPadExtParams<float> padding{true, 0, static_cast<uint8_t>(dma - width), 0.0f};
        if (AllowDenseRead && sourceLd == dma) {
            AscendC::DataCopy(local, source[static_cast<uint64_t>(col) * sourceLd], count * dma);
        } else {
            AscendC::DataCopyPad(local, source[static_cast<uint64_t>(col) * sourceLd], load, padding);
        }
    }
    __aicore__ inline void Load(bool slot, uint32_t col, uint32_t count)
    {
        if (slot) {
            Read(local1, col, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        } else {
            Read(local0, col, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        }
    }
    __aicore__ inline void Store(bool slot, uint32_t col, uint32_t count)
    {
        const AscendC::DataCopyExtParams store{static_cast<uint16_t>(count), width * 4U, 0, (destLd - width) * 4U, 0};
        if (slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
            AscendC::DataCopyPad(dest[static_cast<uint64_t>(col) * destLd], local1, store);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(dest[static_cast<uint64_t>(col) * destLd], local0, store);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
};
class NzCopyWaves {
public:
    __aicore__ inline NzCopyWaves(
        uint32_t rows, uint32_t columns, uint32_t rowWidth, uint32_t waveStep, AscendC::LocalTensor<float> r0,
        AscendC::LocalTensor<float> r1, AscendC::LocalTensor<float> o0, AscendC::LocalTensor<float> o1,
        AscendC::GlobalTensor<float> input, AscendC::GlobalTensor<float> output, AscendC::DataCopyParams loadParams,
        AscendC::CopyRepeatParams scatterParams)
        : m(rows),
          n(columns),
          width(rowWidth),
          step(waveStep),
          raw0(r0),
          raw1(r1),
          out0(o0),
          out1(o1),
          source(input),
          dest(output),
          load(loadParams),
          scatter(scatterParams)
    {}
    __aicore__ inline void Run(uint32_t col, uint32_t wave)
    {
        const bool odd = wave & 1U;
        auto raw = odd ? raw1 : raw0;
        auto out = odd ? out1 : out0;
        if (odd) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        if (wave >= 2) {
            if (odd) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        Transform(raw, out, col);
        Store(out, col, odd);
        Refill(raw, col, odd);
    }

private:
    uint32_t m;
    uint32_t n;
    uint32_t width;
    uint32_t step;
    AscendC::LocalTensor<float> raw0;
    AscendC::LocalTensor<float> raw1;
    AscendC::LocalTensor<float> out0;
    AscendC::LocalTensor<float> out1;
    AscendC::GlobalTensor<float> source;
    AscendC::GlobalTensor<float> dest;
    AscendC::DataCopyParams load;
    AscendC::CopyRepeatParams scatter;
    __aicore__ inline void Transform(AscendC::LocalTensor<float> raw, AscendC::LocalTensor<float> out, uint32_t col)
    {
        AscendC::Copy(out, raw, uint64_t(64), static_cast<uint8_t>(width / 16), scatter);
        AscendC::Copy(out[8], raw[8], uint64_t(64), static_cast<uint8_t>(width / 16), scatter);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t count = (8U < n - col ? 8U : n - col);
        const uint32_t elements = count * m;
        uint64_t reserved = 0;
        AscendC::GatherMask(
            out, out, 7, true, m, {1, static_cast<uint16_t>(count), static_cast<uint16_t>(width / 8), 0}, reserved);
    }
    __aicore__ inline void Store(AscendC::LocalTensor<float> out, uint32_t col, bool odd)
    {
        const uint32_t count = (8U < n - col ? 8U : n - col);
        const uint32_t elements = count * m;
        if (odd) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }
        if (elements % 8 == 0) {
            AscendC::DataCopy(dest[static_cast<uint64_t>(col) * m], out, elements);
        } else {
            AscendC::DataCopyPad(dest[static_cast<uint64_t>(col) * m], out, {1, elements * 4U, 0, 0, 0});
        }
        if (odd) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
    __aicore__ inline void Refill(AscendC::LocalTensor<float> raw, uint32_t col, bool odd)
    {
        if (col + 2 * step < n) {
            // RAW has been consumed by V. Its refill may overlap the
            // separate OUT buffer's MTE3 write; only OUT reuse waits MTE3.
            if (odd) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            AscendC::DataCopy(raw, source[static_cast<uint64_t>(col + 2 * step) * 16], load);
            if (odd) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
        }
    }
};

__aicore__ inline void CompactDrainRows(AscendC::LocalTensor<float> rows, uint32_t m, uint32_t width, uint32_t count)
{
    if ASCEND_IS_AIV {
        AscendC::SetMaskCount();
        set_vector_mask(0, m);
        auto* data = reinterpret_cast<__ubuf__ uint32_t*>(rows.GetPhyAddr());
        vreducev2(data, data, data + 1024, static_cast<uint16_t>(count), 1, 7, static_cast<uint16_t>(width / 8), 0);
        AscendC::SetMaskNorm();
    }
}

__aicore__ inline void StoreDrainTail(
    AscendC::LocalTensor<float> out1, AscendC::GlobalTensor<float> dest, uint32_t second, uint32_t m, uint32_t n,
    uint32_t pitch)
{
    const uint32_t elements0 = 8 * m;
    if (second + 8 <= n) {
        CompactDrainRows(out1, m, pitch, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopy(dest[static_cast<uint64_t>(second) * m], out1, elements0);
    } else {
        const uint32_t count1 = n - second;
        const uint32_t elements1 = count1 * m;
        CompactDrainRows(out1, m, pitch, count1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::DataCopyPad(dest[static_cast<uint64_t>(second) * m], out1, {1, elements1 * 4U, 0, 0, 0});
    }
}

template <uint32_t COLUMNS, bool FIXED_WIDTH>
__aicore__ inline void LoadDrainRows(
    AscendC::LocalTensor<float> raw, const AscendC::GlobalTensor<float>& source, uint32_t m, uint32_t first)
{
    if ASCEND_IS_AIV {
        constexpr uint32_t TM = 64;
        constexpr uint32_t TN = 368;
        constexpr uint32_t columns = COLUMNS;
        const uint32_t row = first / 64U;
        const uint32_t within = first % 64U;
        const uint32_t width = FIXED_WIDTH ? 736 : (m + 15) / 16 * 16;
        const uint64_t offset = static_cast<uint64_t>(row * columns) * TM * TN + within * 16;
        AscendC::DataCopy(
            raw, source[offset],
            AscendC::DataCopyParams{static_cast<uint16_t>(width / 16), 16, static_cast<uint16_t>((TM - 8) * 2), 0});
    }
}

template <uint32_t COLUMNS>
__aicore__ inline void LoadDrainTiles(
    AscendC::LocalTensor<float> raw, const AscendC::GlobalTensor<float>& source, uint32_t m, uint32_t first)
{
    if ASCEND_IS_AIV {
        constexpr uint32_t TM = 64;
        constexpr uint32_t TN = 368;
        constexpr uint32_t columns = COLUMNS;
        const uint32_t row = first / TM;
        const uint32_t within = first % TM;
#pragma unroll
        for (uint32_t col = 0; col < columns; ++col) {
            const uint32_t begin = col * TN;
            const uint32_t width = (min(TN, m - begin) + 15) / 16 * 16;
            const uint64_t offset = static_cast<uint64_t>(row * columns + col) * TM * TN + within * 16;
            AscendC::DataCopy(
                raw[8 * begin], source[offset],
                AscendC::DataCopyParams{static_cast<uint16_t>(width / 16), 16, static_cast<uint16_t>((TM - 8) * 2), 0});
        }
    }
}

template <uint32_t COLUMNS, uint32_t MODE>
__aicore__ inline void LoadDrain(
    AscendC::LocalTensor<float> raw, const AscendC::GlobalTensor<float>& source, uint32_t m, uint32_t first)
{
    if constexpr (MODE == 0) {
        LoadDrainTiles<COLUMNS>(raw, source, m, first);
    } else {
        LoadDrainRows<COLUMNS, MODE == 2>(raw, source, m, first);
    }
}

template <uint32_t COLUMNS, uint32_t MODE>
__aicore__ inline void RunShortDrain(GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    constexpr uint32_t EXTRA = 24;
    const uint32_t first = AscendC::GetBlockIdx() * 8U;
    const uint32_t second = (AscendC::GetBlockIdx() + AscendC::GetBlockNum()) * 8U;
    // With n>=16 and blocks=ceil(n/16), the first eight columns are always valid.
    AscendC::SetMaskNorm(); // AscendC::Copy sets mask bits, but requires Normal mode explicitly.
    const uint32_t width = MODE == 2 ? 736 : (m + 15) / 16 * 16;
    const uint32_t pitch = width + EXTRA;
    const uint32_t rawCap = 8 * width;
    const uint32_t outCap = 8 * pitch;
    AscendC::LocalTensor<float> raw0(AscendC::TPosition::VECCALC, 0, rawCap);
    AscendC::LocalTensor<float> out0(AscendC::TPosition::VECCALC, rawCap * 4, outCap);
    AscendC::LocalTensor<float> raw1(AscendC::TPosition::VECCALC, (rawCap + outCap) * 4, rawCap);
    AscendC::LocalTensor<float> out1(AscendC::TPosition::VECCALC, (2 * rawCap + outCap) * 4, outCap);
    AscendC::GlobalTensor<float> source;
    AscendC::GlobalTensor<float> dest;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputGm));
    dest.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputGm));
    LoadDrain<COLUMNS, MODE>(raw0, source, m, first);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    if (second < n) {
        LoadDrain<COLUMNS, MODE>(raw1, source, m, second);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    }
    AscendC::CopyRepeatParams scatter{static_cast<uint16_t>(pitch / 8), 2, 2, 16};
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Copy(out0, raw0, uint64_t(64), static_cast<uint8_t>(width / 16), scatter);
    AscendC::Copy(out0[8], raw0[8], uint64_t(64), static_cast<uint8_t>(width / 16), scatter);
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t elements0 = 8 * m;
    CompactDrainRows(out0, m, pitch, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopy(dest[static_cast<uint64_t>(first) * m], out0, elements0);
    if (second < n) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Copy(out1, raw1, uint64_t(64), static_cast<uint8_t>(width / 16), scatter);
        AscendC::Copy(out1[8], raw1[8], uint64_t(64), static_cast<uint8_t>(width / 16), scatter);
        AscendC::PipeBarrier<PIPE_V>();
        StoreDrainTail(out1, dest, second, m, n, pitch);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

template <typename Args>
__aicore__ inline void Pack512Compact(
    const AscendC::GlobalTensor<float>& source, const AscendC::GlobalTensor<float>& dest,
    AscendC::LocalTensor<float> local, Args t)
{
    const uint32_t per = t.perAndBatch & 65535U;
    const uint32_t batch = t.perAndBatch >> 16;
    const uint32_t first = AscendC::GetBlockIdx() * per;
    const uint32_t end = min(first + per, t.rows);
    const uint32_t dma = (t.width + 7) / 8 * 8;
    for (uint32_t row = first; row < end; row += batch) {
        const uint32_t count = min(batch, end - row);
        AscendC::Duplicate(local, 0.0f, count * t.outputLd);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp{static_cast<uint16_t>(count), t.width * 4U, 0, (t.outputLd - dma) / 8U, 0};
        AscendC::DataCopyPad(
            local, source[static_cast<uint64_t>(row) * t.width], cp,
            {true, 0, static_cast<uint8_t>(dma - t.width), 0.0f});
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(dest[static_cast<uint64_t>(row) * t.outputLd], local, count * t.outputLd);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

} // namespace gemm_ex
