/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lt_matmul_device_utils.h"
#include "lt_matmul_tiling.h"

using namespace AscendC;

namespace {

constexpr uint32_t OUTPUT_COLUMNS = 64U;
constexpr uint32_t K_CHUNK = 128U;
constexpr uint32_t FLOAT_BYTES = 4U;
constexpr uint32_t FLOATS_PER_BLOCK = 8U;

using LtMatmul22Device::MinOf;
using LtMatmul22Device::SyncPipe;

__aicore__ inline uint32_t AlignRow(uint32_t count)
{
    return (count + FLOATS_PER_BLOCK - 1U) / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK;
}

__aicore__ inline void ReadContiguous(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint64_t offset, uint32_t count)
{
    const DataCopyExtParams copy{1U, count * FLOAT_BYTES, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> padding{false, 0U, 0U, 0.0F};
    DataCopyPad(dst, src[offset], copy, padding);
}

__aicore__ inline void ReadRows(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint64_t offset, uint32_t rows, uint32_t columns,
    uint32_t pitch)
{
    const uint64_t gap = static_cast<uint64_t>(pitch - columns) * FLOAT_BYTES;
    if (gap <= 0xFFFFFFFFULL) {
        const DataCopyExtParams copy{
            static_cast<uint16_t>(rows), columns * FLOAT_BYTES, static_cast<uint32_t>(gap), 0U, 0U};
        const DataCopyPadExtParams<float> padding{false, 0U, 0U, 0.0F};
        DataCopyPad(dst, src[offset], copy, padding);
    } else {
        const uint32_t ubPitch = AlignRow(columns);
        for (uint32_t row = 0U; row < rows; ++row) {
            ReadContiguous(dst[row * ubPitch], src, offset + static_cast<uint64_t>(row) * pitch, columns);
        }
    }
}

struct OrderedBuffers {
    TBuf<TPosition::VECCALC> aStageBuffer, aBuffer, bStageBuffer;
    TBuf<TPosition::VECCALC> aOffsetBuffer, columnIndexBuffer, bOffsetBuffer;
    TBuf<TPosition::VECCALC> bBuffer, productBuffer, accumulatorBuffer, oldBuffer;

    __aicore__ inline void Init(TPipe& pipe)
    {
        // 39,424 bytes including every 32-byte padded DMA row:
        // Astage=4096, A=512, Bstage=32768, Aoffset=512,
        // columnIndex/Boffset/B/product/accumulator/oldC=6*256.
        pipe.InitBuffer(aStageBuffer, K_CHUNK * FLOATS_PER_BLOCK * FLOAT_BYTES);
        pipe.InitBuffer(aBuffer, K_CHUNK * FLOAT_BYTES);
        pipe.InitBuffer(bStageBuffer, K_CHUNK * OUTPUT_COLUMNS * FLOAT_BYTES);
        pipe.InitBuffer(aOffsetBuffer, K_CHUNK * sizeof(int32_t));
        pipe.InitBuffer(columnIndexBuffer, OUTPUT_COLUMNS * sizeof(int32_t));
        pipe.InitBuffer(bOffsetBuffer, OUTPUT_COLUMNS * sizeof(int32_t));
        pipe.InitBuffer(bBuffer, OUTPUT_COLUMNS * FLOAT_BYTES);
        pipe.InitBuffer(productBuffer, OUTPUT_COLUMNS * FLOAT_BYTES);
        pipe.InitBuffer(accumulatorBuffer, OUTPUT_COLUMNS * FLOAT_BYTES);
        pipe.InitBuffer(oldBuffer, OUTPUT_COLUMNS * FLOAT_BYTES);
        auto aOffsets = aOffsetBuffer.Get<int32_t>();
        CreateVecIndex(aOffsets, 0, K_CHUNK);
        CreateVecIndex(columnIndexBuffer.Get<int32_t>(), 0, OUTPUT_COLUMNS);
        PipeBarrier<PIPE_V>();
        Muls(aOffsets, aOffsets, static_cast<int32_t>(FLOATS_PER_BLOCK * FLOAT_BYTES), K_CHUNK);
        PipeBarrier<PIPE_V>();
    }
};

__aicore__ inline void StoreTile(
    const GlobalTensor<float>& d, uint64_t offset, const LocalTensor<float>& value, uint32_t columns)
{
    const DataCopyExtParams copy{1U, columns * FLOAT_BYTES, 0U, 0U, 0U};
    DataCopyPad(d[offset], value, copy);
    // A following tile can overwrite the same UB accumulator or DMA staging.
    SyncPipe<HardEvent::MTE3_V>();
    SyncPipe<HardEvent::MTE3_MTE2>();
}

__aicore__ inline void ScaleOnly(
    OrderedBuffers& buffers, const GlobalTensor<float>& c, const GlobalTensor<float>& d, uint64_t row, uint32_t column,
    uint32_t columns, const LtMatmul22Tiling& t)
{
    auto accumulator = buffers.accumulatorBuffer.Get<float>();
    if (t.beta == 0.0F) {
        Duplicate(accumulator, 0.0F, columns);
        SyncPipe<HardEvent::V_MTE3>();
    } else {
        ReadContiguous(accumulator, c, row * t.ldc + column, columns);
        if (t.beta == 1.0F) {
            // Preserve signed zeros and NaN payloads on the exact copy path.
            SyncPipe<HardEvent::MTE2_MTE3>();
        } else {
            SyncPipe<HardEvent::MTE2_V>();
            Muls(accumulator, accumulator, t.beta, columns);
            SyncPipe<HardEvent::V_MTE3>();
        }
    }
    StoreTile(d, row * t.ldd + column, accumulator, columns);
}

template <bool TB>
__aicore__ inline void InitAccumulator(
    OrderedBuffers& buffers, const GlobalTensor<float>& c, uint64_t row, uint32_t column, uint32_t columns,
    const LtMatmul22Tiling& t)
{
    auto accumulator = buffers.accumulatorBuffer.Get<float>();
    if constexpr (TB) {
        Duplicate(accumulator, 0.0F, columns);
    } else if (t.beta == 0.0F) {
        Duplicate(accumulator, 0.0F, columns);
    } else {
        ReadContiguous(accumulator, c, row * t.ldc + column, columns);
        SyncPipe<HardEvent::MTE2_V>();
        if (t.beta != 1.0F) {
            Muls(accumulator, accumulator, t.beta, columns);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <bool TA, bool TB>
__aicore__ inline void StageChunk(
    OrderedBuffers& buffers, const GlobalTensor<float>& a, const GlobalTensor<float>& b, uint64_t row, uint32_t column,
    uint32_t columns, uint32_t startK, uint32_t countK, const LtMatmul22Tiling& t)
{
    auto aValues = buffers.aBuffer.Get<float>();
    if constexpr (TA) {
        ReadRows(buffers.aStageBuffer.Get<float>(), a, static_cast<uint64_t>(startK) * t.lda + row, countK, 1U, t.lda);
    } else {
        ReadContiguous(aValues, a, row * t.lda + startK, countK);
    }
    if constexpr (TB) {
        ReadRows(
            buffers.bStageBuffer.Get<float>(), b, static_cast<uint64_t>(column) * t.ldb + startK, columns, countK,
            t.ldb);
    } else {
        ReadRows(
            buffers.bStageBuffer.Get<float>(), b, static_cast<uint64_t>(startK) * t.ldb + column, countK, columns,
            t.ldb);
    }
    SyncPipe<HardEvent::MTE2_V>();
    if constexpr (TA) {
        Gather(aValues, buffers.aStageBuffer.Get<float>(), buffers.aOffsetBuffer.Get<uint32_t>(), 0U, countK);
        PipeBarrier<PIPE_V>();
    }
    if constexpr (!TB) {
        // CBLAS row-major swaps A/B. Public transB=N maps to Fortran NOTA,
        // whose TEMP=ALPHA*publicA rounds before multiplying publicB.
        Muls(aValues, aValues, t.alpha, countK);
        PipeBarrier<PIPE_V>();
    } else {
        Muls(
            buffers.bOffsetBuffer.Get<int32_t>(), buffers.columnIndexBuffer.Get<int32_t>(),
            static_cast<int32_t>(AlignRow(countK) * FLOAT_BYTES), columns);
        PipeBarrier<PIPE_V>();
    }
    // Avalues are subsequently read with UB scalar GetValue. Both DMA writes
    // and the optional Gather/Muls must complete before those scalar reads.
    SyncPipe<HardEvent::MTE2_S>();
    SyncPipe<HardEvent::V_S>();
}

template <bool TB>
__aicore__ inline void AccumulateChunk(OrderedBuffers& buffers, uint32_t columns, uint32_t countK)
{
    auto aValues = buffers.aBuffer.Get<float>();
    auto bStage = buffers.bStageBuffer.Get<float>();
    auto bValues = buffers.bBuffer.Get<float>();
    auto product = buffers.productBuffer.Get<float>();
    auto accumulator = buffers.accumulatorBuffer.Get<float>();
    const uint32_t bPitch = AlignRow(columns);
    for (uint32_t p = 0U; p < countK; ++p) {
        const float aValue = aValues.GetValue(p);
        if constexpr (TB) {
            Gather(bValues, bStage, buffers.bOffsetBuffer.Get<uint32_t>(), p * FLOAT_BYTES, columns);
            PipeBarrier<PIPE_V>();
            Muls(product, bValues, aValue, columns);
        } else {
            Muls(product, bStage[p * bPitch], aValue, columns);
        }
        // Netlib 3.10.0 is built with -ffp-contract=off. Separate vector
        // instructions with an intervening barrier retain both FP32 rounding
        // steps and the original K order. No tree reduction or FMA is used.
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, product, columns);
        PipeBarrier<PIPE_V>();
        // The actual Netlib 3.10.0 SGEMM has no zero-TEMP skip in these loops:
        // do not drop 0*Inf or a signed-zero contribution here.
    }
}

__aicore__ inline void FinishDot(
    OrderedBuffers& buffers, const GlobalTensor<float>& c, uint64_t row, uint32_t column, uint32_t columns,
    const LtMatmul22Tiling& t)
{
    auto accumulator = buffers.accumulatorBuffer.Get<float>();
    auto old = buffers.oldBuffer.Get<float>();
    Muls(accumulator, accumulator, t.alpha, columns);
    PipeBarrier<PIPE_V>();
    if (t.beta != 0.0F) {
        ReadContiguous(old, c, row * t.ldc + column, columns);
        SyncPipe<HardEvent::MTE2_V>();
        Muls(old, old, t.beta, columns);
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, old, columns);
        PipeBarrier<PIPE_V>();
    }
}

template <bool TA, bool TB>
__aicore__ inline void OrderedMain(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    OrderedBuffers buffers;
    buffers.Init(pipe);
    GlobalTensor<float> aGm, bGm, cGm, dGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    const uint64_t columnTiles = (static_cast<uint64_t>(t.n) + OUTPUT_COLUMNS - 1U) / OUTPUT_COLUMNS;
    const uint64_t tasks = static_cast<uint64_t>(t.m) * columnTiles;
    for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        const uint64_t row = task / columnTiles;
        const uint32_t column = static_cast<uint32_t>(task % columnTiles) * OUTPUT_COLUMNS;
        const uint32_t columns = MinOf(OUTPUT_COLUMNS, t.n - column);
        if (t.alpha == 0.0F || t.k == 0U) {
            ScaleOnly(buffers, cGm, dGm, row, column, columns, t);
            continue;
        }
        InitAccumulator<TB>(buffers, cGm, row, column, columns, t);
        for (uint64_t startK = 0U; startK < t.k; startK += K_CHUNK) {
            const uint32_t countK = MinOf(K_CHUNK, t.k - static_cast<uint32_t>(startK));
            StageChunk<TA, TB>(buffers, aGm, bGm, row, column, columns, static_cast<uint32_t>(startK), countK, t);
            AccumulateChunk<TB>(buffers, columns, countK);
            SyncPipe<HardEvent::V_MTE2>();
            SyncPipe<HardEvent::S_MTE2>();
        }
        if constexpr (TB) {
            FinishDot(buffers, cGm, row, column, columns, t);
        }
        SyncPipe<HardEvent::V_MTE3>();
        StoreTile(dGm, row * t.ldd + column, buffers.accumulatorBuffer.Get<float>(), columns);
    }
}

constexpr uint32_t TINY_ROWS = 4U;
constexpr uint32_t TINY_K = 32U;
constexpr uint32_t TINY_N = 16U;
constexpr uint32_t TINY_LANES = TINY_ROWS * TINY_N;

struct TinyBuffers {
    TBuf<TPosition::VECCALC> aStageBuffer, bStageBuffer, aBuffer, bBuffer;
    TBuf<TPosition::VECCALC> productBuffer, accumulatorBuffer, oldBuffer;
    TBuf<TPosition::VECCALC> aOffsetBuffer, bOffsetBuffer;

    __aicore__ inline void Init(TPipe& pipe)
    {
        // 4,864 bytes: A staging 1024, B staging 2048, five 64-lane FP32
        // planes 1280, and two 64-lane byte-offset tables 512. All DMA rows
        // include ceil8 padding; no generic ordered buffers are allocated.
        pipe.InitBuffer(aStageBuffer, TINY_K * FLOATS_PER_BLOCK * FLOAT_BYTES);
        pipe.InitBuffer(bStageBuffer, TINY_N * TINY_K * FLOAT_BYTES);
        pipe.InitBuffer(aBuffer, TINY_LANES * FLOAT_BYTES);
        pipe.InitBuffer(bBuffer, TINY_LANES * FLOAT_BYTES);
        pipe.InitBuffer(productBuffer, TINY_LANES * FLOAT_BYTES);
        pipe.InitBuffer(accumulatorBuffer, TINY_LANES * FLOAT_BYTES);
        pipe.InitBuffer(oldBuffer, TINY_LANES * FLOAT_BYTES);
        pipe.InitBuffer(aOffsetBuffer, TINY_LANES * sizeof(int32_t));
        pipe.InitBuffer(bOffsetBuffer, TINY_LANES * sizeof(int32_t));
    }
};

template <bool TA, bool TB, uint32_t N>
__aicore__ inline void TinyOffsets(TinyBuffers& buffers, uint32_t k)
{
    constexpr uint32_t lanes = TINY_ROWS * N;
    auto aOffsets = buffers.aOffsetBuffer.Get<int32_t>();
    auto bOffsets = buffers.bOffsetBuffer.Get<int32_t>();
    auto scratch = buffers.productBuffer.Get<int32_t>();
    CreateVecIndex(bOffsets, 0, lanes);
    PipeBarrier<PIPE_V>();
    ShiftRight(aOffsets, bOffsets, static_cast<int32_t>(N == 16U ? 4 : 3), lanes);
    PipeBarrier<PIPE_V>();
    Muls(scratch, aOffsets, static_cast<int32_t>(N), lanes);
    PipeBarrier<PIPE_V>();
    Sub(bOffsets, bOffsets, scratch, lanes);
    PipeBarrier<PIPE_V>();
    // aOffsets starts with flattened output row, bOffsets with column.
    Muls(aOffsets, aOffsets, static_cast<int32_t>((TA ? 1U : AlignRow(k)) * FLOAT_BYTES), lanes);
    Muls(bOffsets, bOffsets, static_cast<int32_t>((TB ? AlignRow(k) : 1U) * FLOAT_BYTES), lanes);
    PipeBarrier<PIPE_V>();
}

template <bool TA, bool TB, uint32_t N>
__aicore__ inline void TinyStage(
    TinyBuffers& buffers, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    uint64_t row, uint32_t rows, const LtMatmul22Tiling& t)
{
    if constexpr (TA) {
        ReadRows(buffers.aStageBuffer.Get<float>(), a, row, t.k, rows, t.lda);
    } else {
        ReadRows(buffers.aStageBuffer.Get<float>(), a, row * t.lda, rows, t.k, t.lda);
    }
    if constexpr (TB) {
        ReadRows(buffers.bStageBuffer.Get<float>(), b, 0U, N, t.k, t.ldb);
    } else {
        ReadRows(buffers.bStageBuffer.Get<float>(), b, 0U, t.k, N, t.ldb);
    }
    if (t.beta != 0.0F) {
        if constexpr (TB) {
            ReadRows(buffers.oldBuffer.Get<float>(), c, row * t.ldc, rows, N, t.ldc);
        } else {
            ReadRows(buffers.accumulatorBuffer.Get<float>(), c, row * t.ldc, rows, N, t.ldc);
        }
    }
    SyncPipe<HardEvent::MTE2_V>();
    if constexpr (!TB) {
        const uint32_t stagedA = TA ? t.k * FLOATS_PER_BLOCK : rows * AlignRow(t.k);
        Muls(buffers.aStageBuffer.Get<float>(), buffers.aStageBuffer.Get<float>(), t.alpha, stagedA);
        if (t.beta != 0.0F && t.beta != 1.0F) {
            Muls(buffers.accumulatorBuffer.Get<float>(), buffers.accumulatorBuffer.Get<float>(), t.beta, rows * N);
        }
    }
    if (TB || t.beta == 0.0F) {
        Duplicate(buffers.accumulatorBuffer.Get<float>(), 0.0F, rows * N);
    }
    PipeBarrier<PIPE_V>();
}

template <bool TA, bool TB, uint32_t N>
__aicore__ inline void TinyOuterProducts(TinyBuffers& buffers, uint32_t rows, const LtMatmul22Tiling& t)
{
    auto aValues = buffers.aBuffer.Get<float>();
    auto bValues = buffers.bBuffer.Get<float>();
    auto product = buffers.productBuffer.Get<float>();
    auto accumulator = buffers.accumulatorBuffer.Get<float>();
    const uint32_t lanes = rows * N;
    for (uint32_t p = 0U; p < t.k; ++p) {
        Gather(
            aValues, buffers.aStageBuffer.Get<float>(), buffers.aOffsetBuffer.Get<uint32_t>(),
            p * (TA ? FLOATS_PER_BLOCK : 1U) * FLOAT_BYTES, lanes);
        Gather(
            bValues, buffers.bStageBuffer.Get<float>(), buffers.bOffsetBuffer.Get<uint32_t>(),
            p * (TB ? 1U : N) * FLOAT_BYTES, lanes);
        PipeBarrier<PIPE_V>();
        Mul(product, aValues, bValues, lanes);
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, product, lanes);
        PipeBarrier<PIPE_V>();
    }
    if constexpr (TB) {
        Muls(accumulator, accumulator, t.alpha, lanes);
        if (t.beta != 0.0F) {
            Muls(buffers.oldBuffer.Get<float>(), buffers.oldBuffer.Get<float>(), t.beta, lanes);
            PipeBarrier<PIPE_V>();
            Add(accumulator, accumulator, buffers.oldBuffer.Get<float>(), lanes);
        }
    }
    SyncPipe<HardEvent::V_MTE3>();
}

template <uint32_t N>
__aicore__ inline void TinyStore(
    const GlobalTensor<float>& d, const LocalTensor<float>& value, uint64_t row, uint32_t rows, uint32_t pitch)
{
    const uint64_t gap = static_cast<uint64_t>(pitch - N) * FLOAT_BYTES;
    if (gap <= 0xFFFFFFFFULL) {
        const DataCopyExtParams copy{static_cast<uint16_t>(rows), N * FLOAT_BYTES, 0U, static_cast<uint32_t>(gap), 0U};
        DataCopyPad(d[row * pitch], value, copy);
    } else {
        const DataCopyExtParams copy{1U, N * FLOAT_BYTES, 0U, 0U, 0U};
        for (uint32_t r = 0U; r < rows; ++r) {
            DataCopyPad(d[(row + r) * pitch], value[r * N], copy);
        }
    }
    SyncPipe<HardEvent::MTE3_MTE2>();
    SyncPipe<HardEvent::MTE3_V>();
}

template <bool TA, bool TB, uint32_t N>
__aicore__ inline void TinyMain(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    const uint64_t firstRow = static_cast<uint64_t>(GetBlockIdx()) * TINY_ROWS;
    if (firstRow >= t.m) {
        return;
    }
    TinyBuffers buffers;
    buffers.Init(pipe);
    TinyOffsets<TA, TB, N>(buffers, t.k);
    GlobalTensor<float> aGm, bGm, cGm, dGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    for (uint64_t row = firstRow; row < t.m; row += static_cast<uint64_t>(GetBlockNum()) * TINY_ROWS) {
        const uint32_t rows = MinOf(TINY_ROWS, t.m - static_cast<uint32_t>(row));
        TinyStage<TA, TB, N>(buffers, aGm, bGm, cGm, row, rows, t);
        TinyOuterProducts<TA, TB, N>(buffers, rows, t);
        TinyStore<N>(dGm, buffers.accumulatorBuffer.Get<float>(), row, rows, t.ldd);
    }
}

template <uint32_t N>
__aicore__ inline void DispatchTiny(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    if (t.transA != 0U) {
        if (t.transB != 0U) {
            TinyMain<true, true, N>(a, b, c, d, t, pipe);
        } else {
            TinyMain<true, false, N>(a, b, c, d, t, pipe);
        }
    } else if (t.transB != 0U) {
        TinyMain<false, true, N>(a, b, c, d, t, pipe);
    } else {
        TinyMain<false, false, N>(a, b, c, d, t, pipe);
    }
}

} // namespace

extern "C" __global__ __aicore__ __vector__ void lt_matmul22_ordered(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, LtMatmul22Tiling t)
{
    TPipe pipe;
    if (t.m <= 32U && t.k > 0U && t.k <= TINY_K && t.alpha != 0.0F) {
        if (t.n == 8U) {
            DispatchTiny<8U>(a, b, c, d, t, pipe);
            return;
        }
        if (t.n == 16U) {
            DispatchTiny<16U>(a, b, c, d, t, pipe);
            return;
        }
    }
    if (t.transA != 0U) {
        if (t.transB != 0U) {
            OrderedMain<true, true>(a, b, c, d, t, pipe);
        } else {
            OrderedMain<true, false>(a, b, c, d, t, pipe);
        }
    } else if (t.transB != 0U) {
        OrderedMain<false, true>(a, b, c, d, t, pipe);
    } else {
        OrderedMain<false, false>(a, b, c, d, t, pipe);
    }
}

void LtMatmul22Ordered(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, LtMatmul22Tiling t)
{
    lt_matmul22_ordered<<<blocks, nullptr, stream>>>(a, b, c, d, t);
}
