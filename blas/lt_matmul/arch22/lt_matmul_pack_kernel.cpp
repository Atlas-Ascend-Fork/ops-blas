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
#include "lt_matmul_tiling.h"

using namespace AscendC;

namespace LtMatmul22Device {
using namespace AscendC;

template <uint32_t ROWS>
struct NZPacker {
    static constexpr uint32_t TILE_ROWS = ROWS;
    static constexpr uint32_t TILE_COLUMNS = 256U;
    static constexpr uint32_t C0 = 8U;
    static constexpr uint32_t FLOAT_BYTES = 4U;
    static constexpr uint32_t INPUT_PITCH_MAX = TILE_COLUMNS + C0;

    template <HardEvent E>
    __aicore__ inline static void SyncNZ()
    {
        SetFlag<E>(EVENT_ID0);
        WaitFlag<E>(EVENT_ID0);
    }

    __aicore__ inline static uint32_t MinNZ(uint32_t a, uint32_t b) { return a < b ? a : b; }

    __aicore__ inline static uint64_t AlignedRowsNZ(uint32_t rows)
    {
        return (static_cast<uint64_t>(rows) + 15U) / 16U * 16U;
    }

    __aicore__ inline static uint64_t TileCountNZ(uint32_t rows, uint32_t cols)
    {
        return ((static_cast<uint64_t>(rows) + TILE_ROWS - 1U) / TILE_ROWS) *
               ((static_cast<uint64_t>(cols) + TILE_COLUMNS - 1U) / TILE_COLUMNS);
    }

    __aicore__ inline static void ReadPhysicalND(
        const LocalTensor<float>& input, const GlobalTensor<float>& source, uint64_t offset, uint32_t rows,
        uint32_t cols, uint32_t sourceLd, uint32_t inputPitch)
    {
        const uint32_t alignedCols = (cols + C0 - 1U) / C0 * C0;
        const uint64_t sourceGap = (static_cast<uint64_t>(sourceLd) - cols) * FLOAT_BYTES;
        const DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>(alignedCols - cols), 0.0F};
        if (sourceGap <= 0xFFFFFFFFULL) {
            const DataCopyExtParams copy{
                static_cast<uint16_t>(rows), cols * FLOAT_BYTES, static_cast<uint32_t>(sourceGap),
                (inputPitch - alignedCols) / C0, 0U};
            DataCopyPad(input, source[offset], copy, pad);
        } else {
            const DataCopyExtParams copy{1U, cols * FLOAT_BYTES, 0U, 0U, 0U};
            for (uint32_t row = 0; row < rows; ++row) {
                DataCopyPad(input[row * inputPitch], source[offset + static_cast<uint64_t>(row) * sourceLd], copy, pad);
            }
        }
    }

    __aicore__ inline static void WritePhysicalNZ(
        const GlobalTensor<float>& destination, const LocalTensor<float>& output, uint64_t offset, uint64_t paddedRows,
        uint32_t paddedColumns, uint32_t writtenRows)
    {
        // Preserve global16-row NZ rounding. The final packing job can have fewer
        // allocated rows; skip its unused UB rows between column slabs.
        const uint64_t gap = (paddedRows - writtenRows) * C0 * FLOAT_BYTES;
        const uint32_t slabs = paddedColumns / C0;
        if (gap <= 0xFFFFFFFFULL) {
            const DataCopyExtParams copy{
                static_cast<uint16_t>(slabs), writtenRows * C0 * FLOAT_BYTES, TILE_ROWS - writtenRows,
                static_cast<uint32_t>(gap), 0U};
            DataCopyPad(destination[offset], output, copy);
        } else {
            const DataCopyExtParams copy{1U, writtenRows * C0 * FLOAT_BYTES, 0U, 0U, 0U};
            for (uint32_t slab = 0; slab < slabs; ++slab) {
                DataCopyPad(
                    destination[offset + static_cast<uint64_t>(slab) * paddedRows * C0], output[slab * TILE_ROWS * C0],
                    copy);
            }
        }
    }

    __aicore__ inline static void PackOneTile(
        const GlobalTensor<float>& source, const GlobalTensor<float>& destination, const LocalTensor<float>& input,
        const LocalTensor<float>& output, uint64_t tile, uint32_t rows, uint32_t cols, uint32_t ld)
    {
        const uint64_t columnTiles = (static_cast<uint64_t>(cols) + TILE_COLUMNS - 1U) / TILE_COLUMNS;
        const uint32_t row = static_cast<uint32_t>(tile / columnTiles * TILE_ROWS);
        const uint32_t column = static_cast<uint32_t>(tile % columnTiles * TILE_COLUMNS);
        const uint32_t validRows = MinNZ(TILE_ROWS, rows - row);
        const uint32_t validCols = MinNZ(TILE_COLUMNS, cols - column);
        const uint32_t outputCols = (validCols + C0 - 1U) / C0 * C0;
        // One extra DataBlock between ND rows changes the inter-row bank stride;
        // no extra padding is emitted into the public NZ layout.
        const uint32_t inputPitch = outputCols + C0;
        if constexpr (ROWS == 16U) {
            Duplicate(input, 0.0F, TILE_ROWS * inputPitch);
        } else if (validRows < TILE_ROWS) {
            Duplicate(input[validRows * inputPitch], 0.0F, (TILE_ROWS - validRows) * inputPitch);
        }
        // MTE2 must wait for the previous V reader, including full-row jobs.
        SyncNZ<HardEvent::V_MTE2>();
        ReadPhysicalND(input, source, static_cast<uint64_t>(row) * ld + column, validRows, validCols, ld, inputPitch);
        SyncNZ<HardEvent::MTE2_V>();

        // One repeat copies one8-float C0 from each of8 rows. Subsequent repeats
        // advance one input C0 and one tile-height NZ slab. This is a pure bit copy.
        const CopyRepeatParams rearrange{1U, static_cast<uint16_t>(inputPitch / C0), TILE_ROWS, 1U};
        const uint8_t repeats = static_cast<uint8_t>(outputCols / C0);
#pragma unroll
        for (uint32_t r = 0; r < TILE_ROWS; r += 8U) {
            Copy(output[r * C0], input[r * inputPitch], uint64_t(64U), repeats, rearrange);
        }
        SyncNZ<HardEvent::V_MTE3>();
        const uint64_t paddedRows = AlignedRowsNZ(rows);
        const uint64_t offset = static_cast<uint64_t>(column / C0) * paddedRows * C0 + static_cast<uint64_t>(row) * C0;
        WritePhysicalNZ(
            destination, output, offset, paddedRows, outputCols,
            static_cast<uint32_t>(MinNZ(TILE_ROWS, paddedRows - row)));
        // Both buffers may be reused by the next job; drain the output reader.
        SyncNZ<HardEvent::MTE3_V>();
    }

    __aicore__ inline static void PackBothNZ(
        GM_ADDR a, GM_ADDR b, GM_ADDR aNz, GM_ADDR bNz, const LtMatmul22Tiling& t, TPipe& pipe)
    {
        const uint32_t aRows = t.transA ? t.k : t.m;
        const uint32_t aCols = t.transA ? t.m : t.k;
        const uint32_t bRows = t.transB ? t.n : t.k;
        const uint32_t bCols = t.transB ? t.k : t.n;
        const uint64_t aTiles = TileCountNZ(aRows, aCols);
        const uint64_t bTiles = TileCountNZ(bRows, bCols);
        TBuf<TPosition::VECCALC> inputBuffer, outputBuffer;
        pipe.InitBuffer(inputBuffer, TILE_ROWS * INPUT_PITCH_MAX * FLOAT_BYTES);
        pipe.InitBuffer(outputBuffer, TILE_ROWS * TILE_COLUMNS * FLOAT_BYTES);
        const auto input = inputBuffer.Get<float>();
        const auto output = outputBuffer.Get<float>();
        GlobalTensor<float> aGm, bGm, aNzGm, bNzGm;
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        aNzGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aNz));
        bNzGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bNz));
        for (uint64_t job = GetBlockIdx(); job < aTiles + bTiles; job += GetBlockNum()) {
            if (job < aTiles) {
                PackOneTile(aGm, aNzGm, input, output, job, aRows, aCols, t.lda);
            } else {
                PackOneTile(bGm, bNzGm, input, output, job - aTiles, bRows, bCols, t.ldb);
            }
        }
        // Copy uses a 64-element FP32 mask. Restore both mask registers before
        // returning, including idle blocks and the final partial packing job.
        ResetMask();
    }

    __aicore__ inline static void PackOnlyNZ(
        GM_ADDR source, GM_ADDR packed, const LtMatmul22Tiling& t, uint32_t matrix, TPipe& pipe)
    {
        TBuf<TPosition::VECCALC> ib, ob;
        pipe.InitBuffer(ib, TILE_ROWS * INPUT_PITCH_MAX * FLOAT_BYTES);
        pipe.InitBuffer(ob, TILE_ROWS * TILE_COLUMNS * FLOAT_BYTES);
        GlobalTensor<float> src, dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packed));
        const uint32_t rows = matrix == 0 ? (t.transA ? t.k : t.m) : (t.transB ? t.n : t.k);
        const uint32_t cols = matrix == 0 ? (t.transA ? t.m : t.k) : (t.transB ? t.k : t.n);
        const uint32_t ld = matrix == 0 ? t.lda : t.ldb;
        for (uint64_t job = GetBlockIdx(); job < TileCountNZ(rows, cols); job += GetBlockNum()) {
            PackOneTile(src, dst, ib.Get<float>(), ob.Get<float>(), job, rows, cols, ld);
        }
        ResetMask();
    }
};
} // namespace LtMatmul22Device

extern "C" __global__ __aicore__ __vector__ void lt_matmul22_pack_ab_nz(
    GM_ADDR a, GM_ADDR b, GM_ADDR aNz, GM_ADDR bNz, LtMatmul22Tiling t)
{
    TPipe pipe;
    LtMatmul22Device::NZPacker<16U>::PackBothNZ(a, b, aNz, bNz, t, pipe);
}

void LtMatmul22PackNZ(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* aNz, uint8_t* bNz, LtMatmul22Tiling t)
{
    if (blocks == 0U || t.m == 0U || t.n == 0U || t.k == 0U)
        return;
    // H2 rounds both NZ plane starts to32B within its advertised capacity.
    // Serialize an unaligned internal call to avoid shared32B writers.
    const bool aligned = ((reinterpret_cast<uintptr_t>(aNz) | reinterpret_cast<uintptr_t>(bNz)) & 31U) == 0U;
    lt_matmul22_pack_ab_nz<<<aligned ? blocks : 1U, nullptr, stream>>>(a, b, aNz, bNz, t);
}

extern "C" __global__ __aicore__ __vector__ void lt_matmul22_pack_ab_nz64(
    GM_ADDR a, GM_ADDR b, GM_ADDR aNz, GM_ADDR bNz, LtMatmul22Tiling t)
{
    TPipe pipe;
    LtMatmul22Device::NZPacker<64U>::PackBothNZ(a, b, aNz, bNz, t, pipe);
}

void LtMatmul22PackNZ64(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* aNz, uint8_t* bNz, LtMatmul22Tiling t)
{
    if (blocks == 0U || t.m == 0U || t.n == 0U || t.k == 0U)
        return;
    // Public workspace16B alignment remains legal. Root may internally round
    // both plane starts to32B; otherwise serialize to avoid shared32B writers.
    const bool aligned = ((reinterpret_cast<uintptr_t>(aNz) | reinterpret_cast<uintptr_t>(bNz)) & 31U) == 0U;
    lt_matmul22_pack_ab_nz64<<<aligned ? blocks : 1U, nullptr, stream>>>(a, b, aNz, bNz, t);
}

extern "C" __global__ __aicore__ __vector__ void lt_matmul22_pack_ab_nz80(
    GM_ADDR a, GM_ADDR b, GM_ADDR aNz, GM_ADDR bNz, LtMatmul22Tiling t)
{
    TPipe pipe;
    LtMatmul22Device::NZPacker<80U>::PackBothNZ(a, b, aNz, bNz, t, pipe);
}

void LtMatmul22PackNZ80(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* aNz, uint8_t* bNz, LtMatmul22Tiling t)
{
    if (blocks == 0U || t.m == 0U || t.n == 0U || t.k == 0U)
        return;
    // Public workspace16B alignment remains legal. Root may internally round
    // both plane starts to32B; otherwise serialize to avoid shared32B writers.
    const bool aligned = ((reinterpret_cast<uintptr_t>(aNz) | reinterpret_cast<uintptr_t>(bNz)) & 31U) == 0U;
    lt_matmul22_pack_ab_nz80<<<aligned ? blocks : 1U, nullptr, stream>>>(a, b, aNz, bNz, t);
}

extern "C" __global__ __aicore__ __vector__ void lt_matmul22_pack_one_nz(
    GM_ADDR source, GM_ADDR packed, LtMatmul22Tiling t, uint32_t matrix)
{
    TPipe pipe;
    LtMatmul22Device::NZPacker<16U>::PackOnlyNZ(source, packed, t, matrix, pipe);
}
void LtMatmul22PackOneNZ(
    uint32_t blocks, void* stream, uint8_t* source, uint8_t* packed, LtMatmul22Tiling t, uint32_t matrix)
{
    lt_matmul22_pack_one_nz<<<blocks, nullptr, stream>>>(source, packed, t, matrix);
}
