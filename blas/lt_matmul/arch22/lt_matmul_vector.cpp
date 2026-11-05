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
#include "lt_matmul_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t OUTPUT_ROWS = 8U;
constexpr uint32_t OUTPUT_CHUNK = 512U;
constexpr uint32_t K_CHUNK = 256U;
constexpr uint32_t TAIL_COLUMNS = 16U;
constexpr uint32_t FLOATS_PER_BLOCK = 8U;
constexpr uint32_t FLOAT_BYTES = sizeof(float);

using LtMatmul22Device::MinOf;
using LtMatmul22Device::SyncPipe;

__aicore__ inline uint32_t AlignFloatRow(uint32_t count)
{
    return (count + FLOATS_PER_BLOCK - 1U) / FLOATS_PER_BLOCK * FLOATS_PER_BLOCK;
}

__aicore__ inline void CopyIn(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint64_t offset, uint32_t count)
{
    const DataCopyExtParams copy{1U, count * FLOAT_BYTES, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
    DataCopyPad(dst, src[offset], copy, pad);
}

__aicore__ inline void CopyOut(
    const GlobalTensor<float>& dst, uint64_t offset, const LocalTensor<float>& src, uint32_t count)
{
    const DataCopyExtParams copy{1U, count * FLOAT_BYTES, 0U, 0U, 0U};
    DataCopyPad(dst[offset], src, copy);
}

// DataCopyPad places each UB row on a 32-byte boundary. Large legal leading
// dimensions can exceed the byte-stride field; the row-DMA fallback retains
// 64-bit GM addressing and does not use scalar GM loads.
__aicore__ inline void CopyRowsIn(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint64_t offset, uint32_t rows, uint32_t width,
    uint32_t gmPitch, bool zeroPadding = false)
{
    const uint64_t gap = static_cast<uint64_t>(gmPitch - width) * FLOAT_BYTES;
    const uint32_t ubPitch = AlignFloatRow(width);
    const DataCopyPadExtParams<float> pad{
        zeroPadding, 0U, static_cast<uint8_t>(zeroPadding ? ubPitch - width : 0U), 0.0F};
    if (gap <= 0xFFFFFFFFULL) {
        const DataCopyExtParams copy{
            static_cast<uint16_t>(rows), width * FLOAT_BYTES, static_cast<uint32_t>(gap), 0U, 0U};
        DataCopyPad(dst, src[offset], copy, pad);
    } else {
        const DataCopyExtParams copy{1U, width * FLOAT_BYTES, 0U, 0U, 0U};
        for (uint32_t row = 0U; row < rows; ++row) {
            DataCopyPad(dst[row * ubPitch], src[offset + static_cast<uint64_t>(row) * gmPitch], copy, pad);
        }
    }
}

__aicore__ inline void CopyRowsOut(
    const GlobalTensor<float>& dst, uint64_t offset, const LocalTensor<float>& src, uint32_t rows, uint32_t width,
    uint32_t gmPitch)
{
    const uint64_t gap = static_cast<uint64_t>(gmPitch - width) * FLOAT_BYTES;
    if (gap <= 0xFFFFFFFFULL) {
        // GM stride is bytes; zero UB stride advances by ceil32(blockLen),
        // so the padding lanes computed in UB are discarded for every row.
        const DataCopyExtParams copy{
            static_cast<uint16_t>(rows), width * FLOAT_BYTES, 0U, static_cast<uint32_t>(gap), 0U};
        DataCopyPad(dst[offset], src, copy);
    } else {
        const uint32_t ubPitch = AlignFloatRow(width);
        for (uint32_t row = 0U; row < rows; ++row) {
            CopyOut(dst, offset + static_cast<uint64_t>(row) * gmPitch, src[row * ubPitch], width);
        }
    }
}

// For H1's default copy, read the padded product rows as one continuous DMA.
// The caller checks n<=512 and productLd=ceil16(n), so an eight-row transfer
// uses at most 4096 floats, with a 64-byte-aligned length and 16 KiB UB.
__aicore__ inline void CopyPaddedProductOutput(
    GM_ADDR product, GM_ADDR d, const LtMatmul22Tiling& t, uint64_t firstRow, TPipe& pipe)
{
    TBuf<TPosition::VECCALC> valueBuffer;
    pipe.InitBuffer(valueBuffer, OUTPUT_ROWS * OUTPUT_CHUNK * FLOAT_BYTES);
    auto value = valueBuffer.Get<float>();
    GlobalTensor<float> productGm, dGm;
    productGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(product));
    dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    const uint32_t rowBytes = t.n * FLOAT_BYTES;
    // DataCopyPad consumes ceil32(rowBytes) UB bytes per row before srcStride.
    // Its UB stride is in 32-byte blocks, while its GM stride is in bytes.
    const uint32_t srcStride = (t.productLd - AlignFloatRow(t.n)) / FLOATS_PER_BLOCK;
    const uint64_t dstGap = (static_cast<uint64_t>(t.ldd) - t.n) * FLOAT_BYTES;
    for (uint64_t row = firstRow; row < t.m; row += static_cast<uint64_t>(GetBlockNum()) * OUTPUT_ROWS) {
        const uint32_t rows = MinOf(OUTPUT_ROWS, t.m - static_cast<uint32_t>(row));
        DataCopy(value, productGm[row * t.productLd], rows * t.productLd);
        SyncPipe<HardEvent::MTE2_MTE3>();
        if (dstGap <= 0xFFFFFFFFULL) {
            const DataCopyExtParams copy{
                static_cast<uint16_t>(rows), rowBytes, srcStride, static_cast<uint32_t>(dstGap), 0U};
            DataCopyPad(dGm[row * t.ldd], value, copy);
        } else {
            // Keep 64-bit GM offsets and the full productLd UB pitch when
            // the destination byte gap does not fit the 32-bit DMA field.
            for (uint32_t r = 0U; r < rows; ++r) {
                CopyOut(dGm, (row + r) * t.ldd, value[r * t.productLd], t.n);
            }
        }
        SyncPipe<HardEvent::MTE3_MTE2>();
    }
}

__aicore__ inline void ScaleOutputTile(
    const LocalTensor<float>& value, const GlobalTensor<float>& cGm, uint64_t cOffset, uint32_t rows, uint32_t columns,
    uint32_t count, const LtMatmul22Tiling& t)
{
    if (t.beta == 0.0F) {
        Duplicate(value, 0.0F, count);
        SyncPipe<HardEvent::V_MTE3>();
    } else {
        CopyRowsIn(value, cGm, cOffset, rows, columns, t.ldc, true);
        if (t.beta == 1.0F) {
            // A byte-preserving copy is required for alpha=0,
            // beta=1, including signed zeros and NaN payloads.
            SyncPipe<HardEvent::MTE2_MTE3>();
        } else {
            SyncPipe<HardEvent::MTE2_V>();
            Muls(value, value, t.beta, count);
            SyncPipe<HardEvent::V_MTE3>();
        }
    }
}

__aicore__ inline void EpilogueOutputTile(
    const LocalTensor<float>& value, const LocalTensor<float>& old, const GlobalTensor<float>& productGm,
    const GlobalTensor<float>& cGm, uint64_t row, uint64_t col, uint64_t cOffset, uint32_t rows, uint32_t columns,
    uint32_t count, const LtMatmul22Tiling& t)
{
    CopyRowsIn(value, productGm, row * t.productLd + col, rows, columns, t.productLd, true);
    if (t.beta != 0.0F) {
        CopyRowsIn(old, cGm, cOffset, rows, columns, t.ldc, true);
    }
    if (t.alpha == 1.0F && t.beta == 0.0F) {
        // Padded H1's default epilogue is a pure two-dimensional
        // copy. There is no vector stage on this path.
        SyncPipe<HardEvent::MTE2_MTE3>();
    } else {
        SyncPipe<HardEvent::MTE2_V>();
        if (t.alpha != 1.0F) {
            Muls(value, value, t.alpha, count);
        }
        if (t.beta != 0.0F) {
            if (t.beta != 1.0F) {
                Muls(old, old, t.beta, count);
            }
            PipeBarrier<PIPE_V>();
            Add(value, value, old, count);
        }
        SyncPipe<HardEvent::V_MTE3>();
    }
}

template <bool SCALE_ONLY>
__aicore__ inline void FinishOutputTile(const LtMatmul22Tiling& t)
{
    if constexpr (SCALE_ONLY) {
        if (t.beta == 0.0F) {
            SyncPipe<HardEvent::MTE3_V>();
        } else {
            SyncPipe<HardEvent::MTE3_MTE2>();
        }
    } else {
        SyncPipe<HardEvent::MTE3_MTE2>();
    }
}

template <bool SCALE_ONLY>
__aicore__ inline void ComputeOutput(GM_ADDR product, GM_ADDR c, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    const uint32_t width = SCALE_ONLY ? t.n : t.nCube;
    const uint64_t firstRow = static_cast<uint64_t>(GetBlockIdx()) * OUTPUT_ROWS;
    if (firstRow >= t.m || width == 0U) {
        return;
    }
    if constexpr (!SCALE_ONLY) {
        // Bound n before rounding/multiplying it. H1 owns every padded R row;
        // this path reads neither C nor the previous D values.
        if (t.n <= OUTPUT_CHUNK && t.n != 0U && t.nCube == t.n && t.ldd >= t.n &&
            t.productLd == (t.n + 15U) / 16U * 16U && t.alpha == 1.0F && t.beta == 0.0F) {
            CopyPaddedProductOutput(product, d, t, firstRow, pipe);
            return;
        }
    }
    // Two 8*512 FP32 planes occupy 32,768 bytes. The short final column
    // tile uses ceil8(width) UB pitch, including its zero-filled padding.
    TBuf<TPosition::VECCALC> valueBuffer, oldBuffer;
    pipe.InitBuffer(valueBuffer, OUTPUT_ROWS * OUTPUT_CHUNK * FLOAT_BYTES);
    pipe.InitBuffer(oldBuffer, OUTPUT_ROWS * OUTPUT_CHUNK * FLOAT_BYTES);
    auto value = valueBuffer.Get<float>();
    auto old = oldBuffer.Get<float>();
    GlobalTensor<float> productGm, cGm, dGm;
    productGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(product));
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    for (uint64_t row = firstRow; row < t.m; row += static_cast<uint64_t>(GetBlockNum()) * OUTPUT_ROWS) {
        const uint32_t rows = MinOf(OUTPUT_ROWS, t.m - static_cast<uint32_t>(row));
        for (uint64_t col = 0U; col < width; col += OUTPUT_CHUNK) {
            const uint32_t columns = MinOf(OUTPUT_CHUNK, width - static_cast<uint32_t>(col));
            const uint32_t count = rows * AlignFloatRow(columns);
            const uint64_t cOffset = row * t.ldc + col;
            const uint64_t dOffset = row * t.ldd + col;
            if constexpr (SCALE_ONLY) {
                ScaleOutputTile(value, cGm, cOffset, rows, columns, count, t);
            } else {
                EpilogueOutputTile(value, old, productGm, cGm, row, col, cOffset, rows, columns, count, t);
            }
            CopyRowsOut(dGm, dOffset, value, rows, columns, t.ldd);
            FinishOutputTile<SCALE_ONLY>(t);
        }
    }
}

struct TailBuffers {
    TBuf<TPosition::VECCALC> aStageBuffer, bStageBuffer, aBuffer, bBuffer, mulBuffer, reduceBuffer;
    TBuf<TPosition::VECCALC> aIndexBuffer, bIndexBuffer, slotIndexBuffer;
    TBuf<TPosition::VECCALC> partialBuffer, valueBuffer, oldBuffer, smallBuffer;

    __aicore__ inline void Init(TPipe& pipe)
    {
        // A transpose stages K rows of one float, padded to eight floats.
        // B stages either K rows of <=15 floats or <=15 rows of K floats.
        // Including all padded rows, these buffers occupy 31,488 bytes.
        pipe.InitBuffer(aStageBuffer, K_CHUNK * FLOATS_PER_BLOCK * FLOAT_BYTES);
        pipe.InitBuffer(bStageBuffer, K_CHUNK * TAIL_COLUMNS * FLOAT_BYTES);
        pipe.InitBuffer(aBuffer, K_CHUNK * FLOAT_BYTES);
        pipe.InitBuffer(bBuffer, K_CHUNK * FLOAT_BYTES);
        pipe.InitBuffer(mulBuffer, K_CHUNK * FLOAT_BYTES);
        pipe.InitBuffer(reduceBuffer, K_CHUNK * FLOAT_BYTES);
        pipe.InitBuffer(aIndexBuffer, K_CHUNK * sizeof(int32_t));
        pipe.InitBuffer(bIndexBuffer, K_CHUNK * sizeof(int32_t));
        pipe.InitBuffer(slotIndexBuffer, TAIL_COLUMNS * sizeof(int32_t));
        pipe.InitBuffer(partialBuffer, TAIL_COLUMNS * FLOATS_PER_BLOCK * FLOAT_BYTES);
        pipe.InitBuffer(valueBuffer, TAIL_COLUMNS * FLOAT_BYTES);
        pipe.InitBuffer(oldBuffer, TAIL_COLUMNS * FLOAT_BYTES);
        pipe.InitBuffer(smallBuffer, TAIL_COLUMNS * FLOAT_BYTES);
    }
};

__aicore__ inline void MakeIndices(TailBuffers& buffers, uint32_t tailWidth)
{
    auto aIndex = buffers.aIndexBuffer.Get<int32_t>();
    auto bIndex = buffers.bIndexBuffer.Get<int32_t>();
    auto slotIndex = buffers.slotIndexBuffer.Get<int32_t>();
    CreateVecIndex(aIndex, 0, K_CHUNK);
    CreateVecIndex(bIndex, 0, K_CHUNK);
    CreateVecIndex(slotIndex, 0, TAIL_COLUMNS);
    PipeBarrier<PIPE_V>();
    Muls(aIndex, aIndex, static_cast<int32_t>(FLOATS_PER_BLOCK * FLOAT_BYTES), K_CHUNK);
    Muls(bIndex, bIndex, static_cast<int32_t>(AlignFloatRow(tailWidth) * FLOAT_BYTES), K_CHUNK);
    Muls(slotIndex, slotIndex, static_cast<int32_t>(FLOATS_PER_BLOCK * FLOAT_BYTES), TAIL_COLUMNS);
    PipeBarrier<PIPE_V>();
}

template <bool TA, bool TB>
__aicore__ inline void StageTailInputs(
    TailBuffers& buffers, const GlobalTensor<float>& aGm, const GlobalTensor<float>& bGm, uint64_t row, uint32_t startK,
    uint32_t count, const LtMatmul22Tiling& t)
{
    const uint32_t tailWidth = t.n - t.nCube;
    if constexpr (TA) {
        CopyRowsIn(
            buffers.aStageBuffer.Get<float>(), aGm, static_cast<uint64_t>(startK) * t.lda + row, count, 1U, t.lda);
    } else {
        CopyIn(buffers.aBuffer.Get<float>(), aGm, row * t.lda + startK, count);
    }
    if constexpr (TB) {
        CopyRowsIn(
            buffers.bStageBuffer.Get<float>(), bGm, static_cast<uint64_t>(t.nCube) * t.ldb + startK, tailWidth, count,
            t.ldb);
    } else {
        CopyRowsIn(
            buffers.bStageBuffer.Get<float>(), bGm, static_cast<uint64_t>(startK) * t.ldb + t.nCube, count, tailWidth,
            t.ldb);
    }
    SyncPipe<HardEvent::MTE2_V>();
    if constexpr (TA) {
        Gather(
            buffers.aBuffer.Get<float>(), buffers.aStageBuffer.Get<float>(), buffers.aIndexBuffer.Get<uint32_t>(), 0U,
            count);
        PipeBarrier<PIPE_V>();
    }
}

template <bool TB>
__aicore__ inline void AccumulateTailChunk(TailBuffers& buffers, uint32_t count, uint32_t tailWidth)
{
    auto aValues = buffers.aBuffer.Get<float>();
    auto bValues = buffers.bBuffer.Get<float>();
    auto mul = buffers.mulBuffer.Get<float>();
    auto bStage = buffers.bStageBuffer.Get<float>();
    auto partial = buffers.partialBuffer.Get<float>();
    auto reduce = buffers.reduceBuffer.Get<float>();
    auto small = buffers.smallBuffer.Get<float>();
    auto value = buffers.valueBuffer.Get<float>();
    for (uint32_t col = 0U; col < tailWidth; ++col) {
        if constexpr (TB) {
            Mul(mul, aValues, bStage[col * AlignFloatRow(count)], count);
        } else {
            Gather(bValues, bStage, buffers.bIndexBuffer.Get<uint32_t>(), col * FLOAT_BYTES, count);
            PipeBarrier<PIPE_V>();
            Mul(mul, aValues, bValues, count);
        }
        PipeBarrier<PIPE_V>();
        ReduceSum(partial[col * FLOATS_PER_BLOCK], mul, reduce, count);
        // The next output column reuses mul, bValues, and reduction scratch.
        PipeBarrier<PIPE_V>();
    }
    Gather(small, partial, buffers.slotIndexBuffer.Get<uint32_t>(), 0U, tailWidth);
    PipeBarrier<PIPE_V>();
    Add(value, value, small, tailWidth);
    PipeBarrier<PIPE_V>();
}

template <bool TA, bool TB>
__aicore__ inline void ComputeTail(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, const LtMatmul22Tiling& t, TPipe& pipe)
{
    const uint32_t tailWidth = t.n - t.nCube;
    if (tailWidth == 0U) {
        return;
    }
    TailBuffers buffers;
    buffers.Init(pipe);
    MakeIndices(buffers, tailWidth);
    auto value = buffers.valueBuffer.Get<float>();
    auto old = buffers.oldBuffer.Get<float>();
    GlobalTensor<float> aGm, bGm, cGm, dGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    dGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(d));
    for (uint64_t row = GetBlockIdx(); row < t.m; row += GetBlockNum()) {
        Duplicate(value, 0.0F, tailWidth);
        PipeBarrier<PIPE_V>();
        if (t.alpha != 0.0F && t.k != 0U) {
            for (uint64_t startK = 0U; startK < t.k; startK += K_CHUNK) {
                const uint32_t count = MinOf(K_CHUNK, t.k - static_cast<uint32_t>(startK));
                StageTailInputs<TA, TB>(buffers, aGm, bGm, row, static_cast<uint32_t>(startK), count, t);
                AccumulateTailChunk<TB>(buffers, count, tailWidth);
                SyncPipe<HardEvent::V_MTE2>();
            }
            if (t.alpha != 1.0F) {
                Muls(value, value, t.alpha, tailWidth);
                PipeBarrier<PIPE_V>();
            }
        }
        if (t.beta != 0.0F) {
            // Read old C before writing any element of this row's tail, so
            // the same path is valid for C==D and for independent C/D pitches.
            CopyIn(old, cGm, row * t.ldc + t.nCube, tailWidth);
            SyncPipe<HardEvent::MTE2_V>();
            if (t.beta != 1.0F) {
                Muls(old, old, t.beta, tailWidth);
                PipeBarrier<PIPE_V>();
            }
            Add(value, value, old, tailWidth);
        }
        SyncPipe<HardEvent::V_MTE3>();
        CopyOut(dGm, row * t.ldd + t.nCube, value, tailWidth);
        SyncPipe<HardEvent::MTE3_MTE2>();
        SyncPipe<HardEvent::MTE3_V>();
    }
}

} // namespace

extern "C" __global__ __aicore__ __vector__ void lt_matmul22_scale(GM_ADDR c, GM_ADDR d, LtMatmul22Tiling t)
{
    TPipe pipe;
    ComputeOutput<true>(nullptr, c, d, t, pipe);
}

extern "C" __global__ __aicore__ __vector__ void lt_matmul22_epilogue(
    GM_ADDR product, GM_ADDR c, GM_ADDR d, LtMatmul22Tiling t)
{
    TPipe pipe;
    ComputeOutput<false>(product, c, d, t, pipe);
}

extern "C" __global__ __aicore__ __vector__ void lt_matmul22_tail(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d, LtMatmul22Tiling t)
{
    TPipe pipe;
    if (t.transA != 0U) {
        if (t.transB != 0U) {
            ComputeTail<true, true>(a, b, c, d, t, pipe);
        } else {
            ComputeTail<true, false>(a, b, c, d, t, pipe);
        }
    } else if (t.transB != 0U) {
        ComputeTail<false, true>(a, b, c, d, t, pipe);
    } else {
        ComputeTail<false, false>(a, b, c, d, t, pipe);
    }
}

void LtMatmul22Scale(uint32_t blocks, void* stream, uint8_t* c, uint8_t* d, LtMatmul22Tiling t)
{
    lt_matmul22_scale<<<blocks, nullptr, stream>>>(c, d, t);
}

void LtMatmul22Epilogue(uint32_t blocks, void* stream, uint8_t* product, uint8_t* c, uint8_t* d, LtMatmul22Tiling t)
{
    lt_matmul22_epilogue<<<blocks, nullptr, stream>>>(product, c, d, t);
}

void LtMatmul22Tail(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* d, LtMatmul22Tiling t)
{
    lt_matmul22_tail<<<blocks, nullptr, stream>>>(a, b, c, d, t);
}
