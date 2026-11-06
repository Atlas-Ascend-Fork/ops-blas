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
#include "lib/matmul_intf.h"
#include "gemm_ex_tiling.h"

using namespace AscendC;
using namespace matmul;
namespace {
// The public BLAS operands are column-major. Their byte order is row-major
// for the transposed operands consumed by C^T = op(B)^T * op(A)^T.
using DenseFp32 = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using DenseNoTransFp32 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulFuncParams kFunc{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                 true,  true};
constexpr MatmulBiasParams kBias{false};

constexpr int32_t ROWS_MATMUL_MAX_ROWS = 64;
// Partition internal output rows into whole waves of the runtime Cube-core
// count; rowsPerBlock stays within the proven single-core envelope.
constexpr MatmulShapeParams kRowsShape{ROWS_MATMUL_MAX_ROWS, 1024, 1024, 64, 256, 64};
constexpr MatmulConfig kRowsConfig = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kRowsShape, kFunc, kBias);
constexpr MatmulApiStaticTiling kRowsStaticTiling =
    GetMatmulApiTiling<DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32>(kRowsConfig);
using RowsMatmul =
    MatmulImpl<DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, kRowsStaticTiling>;
constexpr MatmulApiStaticTiling kRowsRightTransStaticTiling =
    GetMatmulApiTiling<DenseNoTransFp32, DenseFp32, DenseNoTransFp32, DenseNoTransFp32>(kRowsConfig);
using RowsRightTransMatmul =
    MatmulImpl<DenseNoTransFp32, DenseFp32, DenseNoTransFp32, DenseNoTransFp32, kRowsRightTransStaticTiling>;
constexpr MatmulApiStaticTiling kRowsLeftTransStaticTiling =
    GetMatmulApiTiling<DenseFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32>(kRowsConfig);
using RowsLeftTransMatmul =
    MatmulImpl<DenseFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, kRowsLeftTransStaticTiling>;
constexpr MatmulApiStaticTiling kRowsBothTransStaticTiling =
    GetMatmulApiTiling<DenseFp32, DenseFp32, DenseNoTransFp32, DenseNoTransFp32>(kRowsConfig);
using RowsBothTransMatmul =
    MatmulImpl<DenseFp32, DenseFp32, DenseNoTransFp32, DenseNoTransFp32, kRowsBothTransStaticTiling>;

constexpr MatmulShapeParams kRowsLowKShape{64, 1024, 128, 32, 256, 32};
constexpr MatmulConfig kRowsLowKConfig = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kRowsLowKShape, kFunc, kBias);
constexpr MatmulApiStaticTiling kRowsLowKTiling =
    GetMatmulApiTiling<DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32>(kRowsLowKConfig);
using RowsLowKMatmul =
    MatmulImpl<DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, kRowsLowKTiling>;
constexpr MatmulApiStaticTiling kRowsRightTransLowKTiling =
    GetMatmulApiTiling<DenseNoTransFp32, DenseFp32, DenseNoTransFp32, DenseNoTransFp32>(kRowsLowKConfig);
using RowsRightTransLowKMatmul =
    MatmulImpl<DenseNoTransFp32, DenseFp32, DenseNoTransFp32, DenseNoTransFp32, kRowsRightTransLowKTiling>;
constexpr MatmulApiStaticTiling kRowsLeftTransLowKTiling =
    GetMatmulApiTiling<DenseFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32>(kRowsLowKConfig);
using RowsLeftTransLowKMatmul =
    MatmulImpl<DenseFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, kRowsLeftTransLowKTiling>;
constexpr MatmulApiStaticTiling kRowsBothTransLowKTiling =
    GetMatmulApiTiling<DenseFp32, DenseFp32, DenseNoTransFp32, DenseNoTransFp32>(kRowsLowKConfig);
using RowsBothTransLowKMatmul =
    MatmulImpl<DenseFp32, DenseFp32, DenseNoTransFp32, DenseNoTransFp32, kRowsBothTransLowKTiling>;

uint32_t GetRowsMatmulBlocks(int32_t pn, uint32_t coreBlocks)
{
    // Fewest blocks at the proven 64-row envelope, padded up to whole waves
    // of the runtime Cube-core count. The kernel derives its slice width from
    // GetBlockNum(), so host and device always agree on ownership.
    const uint32_t minBlocks = (static_cast<uint32_t>(pn) + ROWS_MATMUL_MAX_ROWS - 1) / ROWS_MATMUL_MAX_ROWS;
    if (coreBlocks == 0) {
        return 0;
    }
    const uint32_t waves = (minBlocks + coreBlocks - 1) / coreBlocks;
    const uint32_t blocks = waves * coreBlocks;
    const uint32_t rowsPerBlock = (static_cast<uint32_t>(pn) + blocks - 1) / blocks;
    return (static_cast<uint32_t>(pn) + rowsPerBlock - 1) / rowsPerBlock;
}

// Computes the contiguous row slice owned by the current Cube block. Returns
// false when the block index exceeds pn (no output rows to cover).
__aicore__ __inline__ bool GetRowsWindow(int32_t pn, int32_t& rowBase, int32_t& rows)
{
    const int32_t blockNum = static_cast<int32_t>(GetBlockNum());
    const int32_t rowsPerBlock = (pn + blockNum - 1) / blockNum;
    const int32_t base = static_cast<int32_t>(GetBlockIdx()) * rowsPerBlock;
    if (base >= pn) {
        return false;
    }
    rowBase = base;
    rows = pn - base < rowsPerBlock ? pn - base : rowsPerBlock;
    return true;
}

// Bind the GM tensors for the row kernels whose left operand stays in its raw
// compact [N][K] layout (public transB = N).
__aicore__ __inline__ void BindRowsDenseLeft(
    const GemmExTiling& tiling, int32_t rowBase, int32_t rows, GM_ADDR bGm, GM_ADDR aGm, GM_ADDR outputGm,
    GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output)
{
    left.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(bGm) + static_cast<uint64_t>(rowBase) * tiling.pk,
        static_cast<uint64_t>(rows) * tiling.pk);
    right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm), static_cast<uint64_t>(tiling.pk) * tiling.pm);
    output.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(outputGm) + tiling.productOffset + static_cast<uint64_t>(rowBase) * tiling.pm,
        static_cast<uint64_t>(rows) * tiling.pm);
}

// Bind the GM tensors for the row kernels whose left operand keeps its raw
// transposed [K][N] layout (public transB = T).
__aicore__ __inline__ void BindRowsTransposedLeft(
    const GemmExTiling& tiling, int32_t rowBase, int32_t rows, GM_ADDR bGm, GM_ADDR aGm, GM_ADDR outputGm,
    GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output)
{
    left.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(bGm) + rowBase, static_cast<uint64_t>(tiling.pk) * tiling.pn - rowBase);
    right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm), static_cast<uint64_t>(tiling.pk) * tiling.pm);
    output.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(outputGm) + tiling.productOffset + static_cast<uint64_t>(rowBase) * tiling.pm,
        static_cast<uint64_t>(rows) * tiling.pm);
}

// Left/right refer to the internal C^T = op(B)^T * op(A)^T operands.
// Keep the raw transposed left operand's full row stride in SetOrgShape,
// while SetSingleShape describes only this block's output rows.
template <bool LeftTrans, bool RightTrans, class MatmulT>
__aicore__ inline void RunRows(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, const GemmExTiling& tiling)
{
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    int32_t rowBase = 0;
    int32_t rows = 0;
    if (!GetRowsWindow(tiling.pn, rowBase, rows)) {
        return;
    }

    GlobalTensor<float> left;
    GlobalTensor<float> right;
    GlobalTensor<float> output;
    if constexpr (LeftTrans) {
        BindRowsTransposedLeft(tiling, rowBase, rows, bGm, aGm, outputGm, left, right, output);
    } else {
        BindRowsDenseLeft(tiling, rowBase, rows, bGm, aGm, outputGm, left, right, output);
    }

    TPipe pipe;
    MatmulT matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    matmul.DisableBias();
    matmul.SetOrgShape(LeftTrans ? tiling.pn : rows, tiling.pm, tiling.pk);
    matmul.SetSingleShape(rows, tiling.pm, tiling.pk);
    matmul.SetTensorA(left, LeftTrans);
    matmul.SetTensorB(right, RightTrans);
    matmul.IterateAll(output, 0);
    matmul.End();
}
} // namespace

// For public transB=N, raw compact B is already the row-major [N][K]
// internal left operand. Each Cube block owns a contiguous row slice of
// internal C^T=[N][M], while public A is shared read-only.
extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_matmul_rows_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (tiling.pk <= 128) {
        RunRows<false, false, RowsLowKMatmul>(aGm, bGm, outputGm, tiling);
    } else {
        RunRows<false, false, RowsMatmul>(aGm, bGm, outputGm, tiling);
    }
}

extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_matmul_rows_right_trans_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (tiling.pk <= 128) {
        RunRows<false, true, RowsRightTransLowKMatmul>(aGm, bGm, outputGm, tiling);
    } else {
        RunRows<false, true, RowsRightTransMatmul>(aGm, bGm, outputGm, tiling);
    }
}

extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_matmul_rows_left_trans_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (tiling.pk <= 128) {
        RunRows<true, false, RowsLeftTransLowKMatmul>(aGm, bGm, outputGm, tiling);
    } else {
        RunRows<true, false, RowsLeftTransMatmul>(aGm, bGm, outputGm, tiling);
    }
}

extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_matmul_rows_both_trans_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (tiling.pk <= 128) {
        RunRows<true, true, RowsBothTransLowKMatmul>(aGm, bGm, outputGm, tiling);
    } else {
        RunRows<true, true, RowsBothTransMatmul>(aGm, bGm, outputGm, tiling);
    }
}

void gemmex_fallback_matmul_rows_do(void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling)
{
    const uint32_t blocks = GetRowsMatmulBlocks(tiling.pn, tiling.cubeBlocks);
    if (blocks == 0) {
        return;
    }
    gemmex_fallback_matmul_rows_kernel<<<blocks, nullptr, stream>>>(a, b, output, tiling);
}

void gemmex_fallback_matmul_rows_right_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling)
{
    const uint32_t blocks = GetRowsMatmulBlocks(tiling.pn, tiling.cubeBlocks);
    gemmex_fallback_matmul_rows_right_trans_kernel<<<blocks, nullptr, stream>>>(a, b, output, tiling);
}

void gemmex_fallback_matmul_rows_left_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling)
{
    const uint32_t blocks = GetRowsMatmulBlocks(tiling.pn, tiling.cubeBlocks);
    gemmex_fallback_matmul_rows_left_trans_kernel<<<blocks, nullptr, stream>>>(a, b, output, tiling);
}

void gemmex_fallback_matmul_rows_both_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling)
{
    const uint32_t blocks = GetRowsMatmulBlocks(tiling.pn, tiling.cubeBlocks);
    gemmex_fallback_matmul_rows_both_trans_kernel<<<blocks, nullptr, stream>>>(a, b, output, tiling);
}

// ---- Merged from gemm_tiles_kernel.cpp ----

// 2D-tiled multi-Cube path. Each of the 20 logical AIC blocks owns one
// [rows][cols] tile of the internal row-major
// product [pn][pm], reading only its left row slab and right column slab.
// This cuts per-block L2 reads from (rows*pk + pk*pm) to
// (rows*pk + pk*cols): the right operand is replicated across row slabs
// (4x) and the left across column slabs (5x) instead of the right being
// replicated across all 20 blocks. Tiles land packed in workspace; a
// scatter kernel merges them into the destination.
//
// This path handles compact no-transpose operands with one K contribution.
// Other layouts and multi-contribution reductions use the rows kernels.

using namespace AscendC;
using namespace matmul;
namespace {
using DenseNoTransFp32 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

constexpr int32_t TILE_MAX_ROWS = 256; // ceil(1024 / 4 row slabs)
constexpr int32_t TILE_MAX_COLS = 208; // align8(ceil(1024 / 5 col slabs))
constexpr int32_t TILE_ROW_SLABS = 4;
constexpr int32_t TILE_COL_SLABS = 5;
constexpr MatmulShapeParams kTilesShape{TILE_MAX_ROWS, TILE_MAX_COLS, 1024, 64, 208, 64};
constexpr MatmulFuncParams kTilesFunc{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                      true,  true};
constexpr MatmulBiasParams kTilesBias{false};
constexpr MatmulConfig kTilesConfig = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kTilesShape, kTilesFunc, kTilesBias);
constexpr MatmulApiStaticTiling kTilesStaticTiling =
    GetMatmulApiTiling<DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32>(kTilesConfig);
using TilesMatmul =
    MatmulImpl<DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, kTilesStaticTiling>;

constexpr int32_t SCATTER_TILE_FLOATS = 1024;
} // namespace

// Scatter one (tile,row) segment of the internal product into public C with
// the alpha/beta epilogue. Internal tile row r of block b covers public
// column (nBase + rowBase_b + r), elements [mBase + colBase_b, +cols_b).
__aicore__ __inline__ void ScatterTileSegment(
    const GemmExTiling* t, const GlobalTensor<float>& tiles, const GlobalTensor<float>& c, int32_t tile, int32_t row,
    int32_t rowsPerSlab, int32_t colsPerSlab, TQue<QuePosition::VECIN, 2>& inQueue, TQue<QuePosition::VECIN, 2>& cQueue,
    TQue<QuePosition::VECOUT, 2>& outQueue)
{
    const int32_t rowSlab = tile / TILE_COL_SLABS;
    const int32_t colSlab = tile % TILE_COL_SLABS;
    const int32_t rowBase = rowSlab * rowsPerSlab;
    const int32_t colBase = colSlab * colsPerSlab;
    const int32_t globalRow = rowBase + row;
    if (globalRow >= t->pn || colBase >= t->pm) {
        return;
    }
    const int32_t cols = t->pm - colBase < colsPerSlab ? t->pm - colBase : colsPerSlab;
    const uint64_t srcOffset =
        static_cast<uint64_t>(tile) * TILE_MAX_ROWS * TILE_MAX_COLS + static_cast<uint64_t>(row) * TILE_MAX_COLS;
    // Public C column index: internal product [pn][pm] is C^T, so
    // internal row globalRow = public column (nBase + globalRow).
    const uint64_t ci =
        static_cast<uint64_t>(t->mBase + colBase) + (static_cast<uint64_t>(t->nBase) + globalRow) * t->ldc;

    const uint32_t aligned = (static_cast<uint32_t>(cols) + 7) / 8 * 8;
    const uint8_t padding = static_cast<uint8_t>(aligned - static_cast<uint32_t>(cols));
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(cols * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{true, 0, padding, 0.0f};

    LocalTensor<float> in = inQueue.AllocTensor<float>();
    DataCopyPad(in, tiles[srcOffset], copyParams, padParams);
    inQueue.EnQue(in);
    LocalTensor<float> vin = inQueue.DeQue<float>();
    LocalTensor<float> out = outQueue.AllocTensor<float>();
    Muls(out, vin, t->alpha, aligned);
    inQueue.FreeTensor(vin);
    if (t->beta != 0.0f) {
        LocalTensor<float> cLocal = cQueue.AllocTensor<float>();
        DataCopyPad(cLocal, c[ci], copyParams, padParams);
        cQueue.EnQue(cLocal);
        LocalTensor<float> vc = cQueue.DeQue<float>();
        Muls(vc, vc, t->beta, aligned);
        Add(out, out, vc, aligned);
        cQueue.FreeTensor(vc);
    }
    outQueue.EnQue(out);
    LocalTensor<float> vout = outQueue.DeQue<float>();
    DataCopyPad(c[ci], vout, copyParams);
    outQueue.FreeTensor(vout);
}

// tiling.pm/pn/pk describe the full internal product. Tiles are written
// packed at outputGm + productOffset, block b at
// b * TILE_MAX_ROWS * TILE_MAX_COLS, row-major [rows_b][cols_b].
extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_matmul_tiles_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    const int32_t block = static_cast<int32_t>(GetBlockIdx());
    const int32_t rowSlab = block / TILE_COL_SLABS;
    const int32_t colSlab = block % TILE_COL_SLABS;
    const int32_t rowsPerSlab = (tiling.pn + TILE_ROW_SLABS - 1) / TILE_ROW_SLABS;
    const int32_t colsPerSlab = ((tiling.pm + TILE_COL_SLABS - 1) / TILE_COL_SLABS + 7) / 8 * 8;
    const int32_t rowBase = rowSlab * rowsPerSlab;
    const int32_t colBase = colSlab * colsPerSlab;
    if (rowBase >= tiling.pn || colBase >= tiling.pm) {
        return;
    }
    const int32_t rows = tiling.pn - rowBase < rowsPerSlab ? tiling.pn - rowBase : rowsPerSlab;
    const int32_t cols = tiling.pm - colBase < colsPerSlab ? tiling.pm - colBase : colsPerSlab;

    GlobalTensor<float> left;
    GlobalTensor<float> right;
    GlobalTensor<float> output;
    left.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(bGm) + static_cast<uint64_t>(rowBase) * tiling.pk,
        static_cast<uint64_t>(rows) * tiling.pk);
    // Column slab of the row-major [pk][pm] right operand: element base
    // colBase, full row stride pm preserved through SetOrgShape.
    right.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(aGm) + colBase, static_cast<uint64_t>(tiling.pk) * tiling.pm - colBase);
    output.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(outputGm) + tiling.productOffset +
            static_cast<uint64_t>(block) * TILE_MAX_ROWS * TILE_MAX_COLS,
        static_cast<uint64_t>(rows) * TILE_MAX_COLS);

    TPipe pipe;
    TilesMatmul matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    matmul.DisableBias();
    // Preserve the full input strides (orgN controls the non-transposed
    // right operand) while making each workspace tile's C row stride fixed.
    matmul.SetOrgShape(tiling.pn, tiling.pm, tiling.pk, tiling.pk, TILE_MAX_COLS);
    matmul.SetSingleShape(rows, cols, tiling.pk);
    matmul.SetTensorA(left, false);
    matmul.SetTensorB(right, false);
    matmul.IterateAll(output, 0);
    matmul.End();
}

// Fused scatter + epilogue: tiles -> public C (column-major, ldc stride)
// applying alpha/beta. Internal tile row r of block b covers public column
// (nBase + rowBase_b + r), elements [mBase + colBase_b, +cols_b) of that
// column. Replaces the generic epilogue for the 2D-tiled path.
extern "C" __global__ __aicore__ __vector__ void gemmex_fallback_scatter_tiles_kernel(
    GM_ADDR workspaceGm, GM_ADDR cGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    GlobalTensor<float> tiles;
    GlobalTensor<float> c;
    tiles.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm) + t->productOffset);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECIN, 2> cQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(inQueue, 2, SCATTER_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(cQueue, 2, SCATTER_TILE_FLOATS * sizeof(float));
    pipe.InitBuffer(outQueue, 2, SCATTER_TILE_FLOATS * sizeof(float));

    const int32_t rowsPerSlab = (t->pn + TILE_ROW_SLABS - 1) / TILE_ROW_SLABS;
    const int32_t colsPerSlab = ((t->pm + TILE_COL_SLABS - 1) / TILE_COL_SLABS + 7) / 8 * 8;

    // Each AIV block strides over tile-row segments of all 20 tiles.
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    // Enumerate all (tile, row) segments with nonzero extent.
    // segment index s -> tile = s / rowsPerSlab, row = s % rowsPerSlab
    const int32_t totalSegments = TILE_ROW_SLABS * TILE_COL_SLABS * rowsPerSlab;
    for (int32_t s = static_cast<int32_t>(block); s < totalSegments; s += static_cast<int32_t>(stride)) {
        ScatterTileSegment(
            t, tiles, c, s / rowsPerSlab, s % rowsPerSlab, rowsPerSlab, colsPerSlab, inQueue, cQueue, outQueue);
    }
}

void gemmex_fallback_matmul_tiles_do(void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling)
{
    gemmex_fallback_matmul_tiles_kernel<<<TILE_ROW_SLABS * TILE_COL_SLABS, nullptr, stream>>>(a, b, output, tiling);
}

void gemmex_fallback_scatter_tiles_do(
    uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmExTiling& tiling)
{
    gemmex_fallback_scatter_tiles_kernel<<<blocks, nullptr, stream>>>(workspace, c, tiling);
}

// ---- Merged from gemm_vector_kernel.cpp ----

// The rows Matmul section above is Cube-only. The following pack and
// post-processing kernels require the full AscendC Vector interface.
#undef ASCENDC_CUBE_ONLY

using namespace AscendC;
namespace {
constexpr uint32_t PACK_TILE_ELEMENTS = 1024;
constexpr uint32_t ACCUMULATE_TILE_ELEMENTS = 4096;
constexpr uint32_t EPILOGUE_TILE_ELEMENTS = 1024;
constexpr uint32_t FP32_PER_DATA_BLOCK = 8;
} // namespace

// Keep transposed public operands in their raw physical orientations. Each
// physical row is copied with one DMA and an aligned UB tail is zero-filled.
// Batch physical columns into one padded DMA transaction. All added rows and
// stride gaps are initialized, with explicit MTE2/MTE3 ordering before reuse.
__aicore__ inline void PackUnalignedRows(
    LocalTensor<float> local, const GlobalTensor<float>& input, const GlobalTensor<float>& output, uint64_t sourceBase,
    uint32_t sourceRows, uint32_t sourceWidth, uint32_t sourceLd, uint32_t paddedRows, uint32_t paddedWidth)
{
    constexpr uint32_t capacity = 16384;

    if (sourceLd == sourceWidth && paddedWidth == sourceWidth) {
        const uint64_t size = static_cast<uint64_t>(paddedRows) * paddedWidth;
        const uint64_t validSize = static_cast<uint64_t>(sourceRows) * sourceWidth;
        for (uint64_t pos = static_cast<uint64_t>(GetBlockIdx()) * capacity; pos < size;
             pos += static_cast<uint64_t>(GetBlockNum()) * capacity) {
            uint32_t len = min(static_cast<uint64_t>(capacity), size - pos);
            Duplicate(local, 0.0f, capacity);
            PipeBarrier<PIPE_ALL>();
            if (pos < validSize) {
                uint32_t valid = min(static_cast<uint64_t>(len), validSize - pos);
                DataCopyPad(
                    local, input[sourceBase + pos], {1, valid * 4U, 0, 0, 0},
                    {true, 0, static_cast<uint8_t>((8 - valid % 8) % 8), 0.0f});
                PipeBarrier<PIPE_ALL>();
            }
            DataCopyPad(output[pos], local, {1, len * 4U, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
    } else {
        for (uint32_t row = GetBlockIdx(); row < paddedRows; row += GetBlockNum()) {
            Duplicate(local, 0.0f, paddedWidth);
            PipeBarrier<PIPE_ALL>();
            if (row < sourceRows) {
                DataCopyPad(
                    local, input[sourceBase + static_cast<uint64_t>(row) * sourceLd], {1, sourceWidth * 4U, 0, 0, 0},
                    {true, 0, static_cast<uint8_t>((8 - sourceWidth % 8) % 8), 0.0f});
                PipeBarrier<PIPE_ALL>();
            }
            DataCopyPad(output[static_cast<uint64_t>(row) * paddedWidth], local, {1, paddedWidth * 4U, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__aicore__ inline void PackPhysicalMatrix(
    const GlobalTensor<float>& input, const GlobalTensor<float>& output, uint64_t sourceBase, uint32_t sourceRows,
    uint32_t sourceWidth, uint32_t sourceLd, uint32_t paddedRows, uint32_t paddedWidth)
{
    if (paddedWidth == 0 || GetBlockNum() == 0) {
        return;
    }
    constexpr uint32_t capacity = 16384;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, capacity * sizeof(float));
    auto local = buffer.Get<float>();
    if (paddedWidth % 8 != 0) {
        PackUnalignedRows(local, input, output, sourceBase, sourceRows, sourceWidth, sourceLd, paddedRows, paddedWidth);
        return;
    }
    uint32_t batch =
        min(static_cast<uint32_t>((paddedRows + GetBlockNum() - 1) / GetBlockNum()), capacity / paddedWidth);
    if (batch == 0) {
        return;
    }
    const uint32_t dmaWidth = (sourceWidth + 7) / 8 * 8;
    for (uint32_t first = GetBlockIdx() * batch; first < paddedRows; first += GetBlockNum() * batch) {
        uint32_t rows = min(batch, paddedRows - first);
        Duplicate(local, 0.0f, rows * paddedWidth);
        PipeBarrier<PIPE_ALL>();
        if (first < sourceRows) {
            uint32_t valid = min(rows, sourceRows - first);
            DataCopyExtParams cp{
                static_cast<uint16_t>(valid), sourceWidth * 4U, (sourceLd - sourceWidth) * 4U,
                (paddedWidth - dmaWidth) / 8U, 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(dmaWidth - sourceWidth), 0.0f};
            DataCopyPad(local, input[sourceBase + static_cast<uint64_t>(first) * sourceLd], cp, pad);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopy(output[static_cast<uint64_t>(first) * paddedWidth], local, rows * paddedWidth);
        PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void PackRawOperands(
    const GemmExTiling* t, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& packA,
    const GlobalTensor<float>& packB)
{
    const bool at = t->transA != 0;
    const bool bt = t->transB != 0;
    PackPhysicalMatrix(
        a, packA,
        at ? static_cast<uint64_t>(t->mBase) * t->lda + t->kBase : static_cast<uint64_t>(t->kBase) * t->lda + t->mBase,
        at ? t->m : t->k, at ? t->k : t->m, t->lda, at ? t->pm : t->pk, at ? t->pk : t->pm);
    PackPhysicalMatrix(
        b, packB,
        bt ? static_cast<uint64_t>(t->kBase) * t->ldb + t->nBase : static_cast<uint64_t>(t->nBase) * t->ldb + t->kBase,
        bt ? t->k : t->n, bt ? t->n : t->k, t->ldb, bt ? t->pk : t->pn, bt ? t->pn : t->pk);
}
__aicore__ inline void PackNnOperands(
    const GemmExTiling* t, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& packA,
    const GlobalTensor<float>& packB)
{
    PackRawOperands(t, a, b, packA, packB);
}

__aicore__ __inline__ void PackGenericOperands(
    const GemmExTiling* t, const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& packA,
    const GlobalTensor<float>& packB)
{
    TPipe pipe;
    TBuf<QuePosition::VECCALC> rowBuf;
    pipe.InitBuffer(rowBuf, PACK_TILE_ELEMENTS * sizeof(float));
    LocalTensor<float> row = rowBuf.Get<float>();
    for (uint32_t p = GetBlockIdx(); p < static_cast<uint32_t>(t->pk); p += GetBlockNum()) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(t->pm); ++i) {
            const uint64_t gi = static_cast<uint64_t>(t->mBase) + i;
            const uint64_t gp = static_cast<uint64_t>(t->kBase) + p;
            const float value = (p < static_cast<uint32_t>(t->k) && i < static_cast<uint32_t>(t->m)) ?
                                    a.GetValue(t->transA == 0 ? gi + gp * t->lda : gp + gi * t->lda) :
                                    0.0f;
            row.SetValue(i, value);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(packA[uint64_t(p) * t->pm], row, static_cast<uint32_t>(t->pm));
        PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t j = GetBlockIdx(); j < static_cast<uint32_t>(t->pn); j += GetBlockNum()) {
        for (uint32_t p = 0; p < static_cast<uint32_t>(t->pk); ++p) {
            const uint64_t gj = static_cast<uint64_t>(t->nBase) + j;
            const uint64_t gp = static_cast<uint64_t>(t->kBase) + p;
            const float value = (j < static_cast<uint32_t>(t->n) && p < static_cast<uint32_t>(t->k)) ?
                                    b.GetValue(t->transB == 0 ? gp + gj * t->ldb : gj + gp * t->ldb) :
                                    0.0f;
            row.SetValue(p, value);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(packB[uint64_t(j) * t->pk], row, static_cast<uint32_t>(t->pk));
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ __vector__ void gemmex_fallback_pack_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR workspaceGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    GlobalTensor<float> a;
    GlobalTensor<float> b;
    GlobalTensor<float> packA;
    GlobalTensor<float> packB;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bGm));
    auto* w = reinterpret_cast<__gm__ float*>(workspaceGm);
    packA.SetGlobalBuffer(w + t->packAOffset);
    packB.SetGlobalBuffer(w + t->packBOffset);
    if (t->rawPack != 0) {
        PackRawOperands(t, a, b, packA, packB);
    } else if (t->transA == 0 && t->transB == 0) {
        PackNnOperands(t, a, b, packA, packB);
    } else {
        PackGenericOperands(t, a, b, packA, packB);
    }
}

__aicore__ __inline__ void AccumulateStage0(
    const GemmExTiling* t, const GlobalTensor<float>& product, const GlobalTensor<float>& accum, uint64_t offset,
    TQue<QuePosition::VECIN, 2>& productQueue, TQue<QuePosition::VECIN, 2>& accumInQueue)
{
    const uint64_t count = static_cast<uint64_t>(t->pn) * t->pm;
    const uint32_t current =
        static_cast<uint32_t>(count - offset < ACCUMULATE_TILE_ELEMENTS ? count - offset : ACCUMULATE_TILE_ELEMENTS);
    LocalTensor<float> productLocal = productQueue.AllocTensor<float>();
    DataCopy(productLocal, product[offset], current);
    productQueue.EnQue(productLocal);
    if (t->firstK == 0) {
        LocalTensor<float> accumLocal = accumInQueue.AllocTensor<float>();
        DataCopy(accumLocal, accum[offset], current);
        accumInQueue.EnQue(accumLocal);
    }
}

__aicore__ __inline__ void AccumulateStage1(
    const GemmExTiling* t, const GlobalTensor<float>& product, const GlobalTensor<float>& accum, uint64_t offset,
    TQue<QuePosition::VECIN, 2>& productQueue, TQue<QuePosition::VECIN, 2>& accumInQueue,
    TQue<QuePosition::VECOUT, 2>& accumOutQueue)
{
    const uint64_t count = static_cast<uint64_t>(t->pn) * t->pm;
    const uint32_t current =
        static_cast<uint32_t>(count - offset < ACCUMULATE_TILE_ELEMENTS ? count - offset : ACCUMULATE_TILE_ELEMENTS);
    LocalTensor<float> productLocal = productQueue.DeQue<float>();
    LocalTensor<float> outLocal = accumOutQueue.AllocTensor<float>();
    if (t->firstK != 0) {
        Adds(outLocal, productLocal, 0.0f, current);
    } else {
        LocalTensor<float> accumLocal = accumInQueue.DeQue<float>();
        Add(outLocal, accumLocal, productLocal, current);
        accumInQueue.FreeTensor(accumLocal);
    }
    productQueue.FreeTensor(productLocal);
    accumOutQueue.EnQue(outLocal);
}

__aicore__ __inline__ void AccumulateStage2(
    const GemmExTiling* t, const GlobalTensor<float>& accum, uint64_t offset,
    TQue<QuePosition::VECOUT, 2>& accumOutQueue)
{
    const uint64_t count = static_cast<uint64_t>(t->pn) * t->pm;
    const uint32_t current =
        static_cast<uint32_t>(count - offset < ACCUMULATE_TILE_ELEMENTS ? count - offset : ACCUMULATE_TILE_ELEMENTS);
    LocalTensor<float> outLocal = accumOutQueue.DeQue<float>();
    DataCopy(accum[offset], outLocal, current);
    accumOutQueue.FreeTensor(outLocal);
}

extern "C" __global__ __aicore__ void gemmex_fallback_accumulate_kernel(GM_ADDR workspaceGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    auto* w = reinterpret_cast<__gm__ float*>(workspaceGm);
    GlobalTensor<float> product;
    GlobalTensor<float> accum;
    product.SetGlobalBuffer(w + t->productOffset);
    accum.SetGlobalBuffer(w + t->accumOffset);
    const uint64_t count = static_cast<uint64_t>(t->pn) * t->pm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> productQueue;
    TQue<QuePosition::VECIN, 2> accumInQueue;
    TQue<QuePosition::VECOUT, 2> accumOutQueue;
    pipe.InitBuffer(productQueue, 2, ACCUMULATE_TILE_ELEMENTS * sizeof(float));
    pipe.InitBuffer(accumInQueue, 2, ACCUMULATE_TILE_ELEMENTS * sizeof(float));
    pipe.InitBuffer(accumOutQueue, 2, ACCUMULATE_TILE_ELEMENTS * sizeof(float));
    const uint64_t tileCount = (count + ACCUMULATE_TILE_ELEMENTS - 1) / ACCUMULATE_TILE_ELEMENTS;
    const uint64_t block = GetBlockIdx();
    const uint64_t stride = GetBlockNum();
    const uint64_t iterations = block < tileCount ? (tileCount - block + stride - 1) / stride : 0;
    // Three-stage software pipeline:
    //   stage 0: GM -> UB reads for tile i (product, and accum when adding)
    //   stage 1: vector add/copy for tile i-1
    //   stage 2: UB -> GM write for tile i-2
    for (uint64_t step = 0; step < iterations + 2; ++step) {
        if (step < iterations) {
            AccumulateStage0(
                t, product, accum, (block + step * stride) * ACCUMULATE_TILE_ELEMENTS, productQueue, accumInQueue);
        }
        if (step >= 1 && step - 1 < iterations) {
            AccumulateStage1(
                t, product, accum, (block + (step - 1) * stride) * ACCUMULATE_TILE_ELEMENTS, productQueue, accumInQueue,
                accumOutQueue);
        }
        if (step >= 2) {
            AccumulateStage2(t, accum, (block + (step - 2) * stride) * ACCUMULATE_TILE_ELEMENTS, accumOutQueue);
        }
    }
}
__aicore__ __inline__ void EpilogueStage0(
    const GemmExTiling* t, const GlobalTensor<float>& accum, const GlobalTensor<float>& c, uint32_t j,
    const DataCopyExtParams& copyParams, const DataCopyPadExtParams<float>& padParams, float beta,
    TQue<QuePosition::VECIN, 2>& accumQueue, TQue<QuePosition::VECIN, 2>& cQueue)
{
    const uint64_t ci = static_cast<uint64_t>(t->mBase) + (static_cast<uint64_t>(t->nBase) + j) * t->ldc;
    LocalTensor<float> accumLocal = accumQueue.AllocTensor<float>();
    DataCopyPad(accumLocal, accum[uint64_t(j) * t->pm], copyParams, padParams);
    accumQueue.EnQue(accumLocal);
    if (beta != 0.0f) {
        LocalTensor<float> cLocal = cQueue.AllocTensor<float>();
        DataCopyPad(cLocal, c[ci], copyParams, padParams);
        cQueue.EnQue(cLocal);
    }
}

__aicore__ __inline__ void EpilogueStage1(
    const GemmExTiling* t, uint32_t aligned, float alpha, float beta, TQue<QuePosition::VECIN, 2>& accumQueue,
    TQue<QuePosition::VECIN, 2>& cQueue, TQue<QuePosition::VECOUT, 2>& outQueue)
{
    LocalTensor<float> accumLocal = accumQueue.DeQue<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
    Muls(outLocal, accumLocal, alpha, aligned);
    if (beta != 0.0f) {
        LocalTensor<float> cLocal = cQueue.DeQue<float>();
        Muls(cLocal, cLocal, beta, aligned);
        Add(outLocal, outLocal, cLocal, aligned);
        cQueue.FreeTensor(cLocal);
    }
    accumQueue.FreeTensor(accumLocal);
    outQueue.EnQue(outLocal);
}

__aicore__ __inline__ void EpilogueStage2(
    const GemmExTiling* t, const GlobalTensor<float>& c, uint32_t j, const DataCopyExtParams& copyParams,
    TQue<QuePosition::VECOUT, 2>& outQueue)
{
    const uint64_t ci = static_cast<uint64_t>(t->mBase) + (static_cast<uint64_t>(t->nBase) + j) * t->ldc;
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopyPad(c[ci], outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}

extern "C" __global__ __aicore__ void gemmex_fallback_epilogue_kernel(
    GM_ADDR workspaceGm, GM_ADDR cGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    GlobalTensor<float> accum;
    GlobalTensor<float> c;
    accum.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm) + t->accumOffset);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> accumQueue;
    TQue<QuePosition::VECIN, 2> cQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(accumQueue, 2, EPILOGUE_TILE_ELEMENTS * sizeof(float));
    pipe.InitBuffer(cQueue, 2, EPILOGUE_TILE_ELEMENTS * sizeof(float));
    pipe.InitBuffer(outQueue, 2, EPILOGUE_TILE_ELEMENTS * sizeof(float));

    const float alpha = t->alpha;
    const float beta = t->beta;
    const uint32_t logical = static_cast<uint32_t>(t->m);
    const uint32_t aligned = (logical + FP32_PER_DATA_BLOCK - 1) / FP32_PER_DATA_BLOCK * FP32_PER_DATA_BLOCK;
    const uint8_t padding = static_cast<uint8_t>(aligned - logical);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(logical * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{true, 0, padding, 0.0f};
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    const uint32_t iterations =
        block < static_cast<uint32_t>(t->n) ? (static_cast<uint32_t>(t->n) - block + stride - 1) / stride : 0;

    // Three-stage software pipeline:
    //   stage 0: GM -> UB for row i
    //   stage 1: alpha/beta vector epilogue for row i-1
    //   stage 2: UB -> GM for row i-2
    for (uint32_t step = 0; step < iterations + 2; ++step) {
        if (step < iterations) {
            EpilogueStage0(t, accum, c, block + step * stride, copyParams, padParams, beta, accumQueue, cQueue);
        }
        if (step >= 1 && step - 1 < iterations) {
            EpilogueStage1(t, aligned, alpha, beta, accumQueue, cQueue, outQueue);
        }
        if (step >= 2) {
            EpilogueStage2(t, c, block + (step - 2) * stride, copyParams, outQueue);
        }
    }
}
extern "C" __global__ __aicore__ void gemmex_fallback_zero_kernel(GM_ADDR cGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    GlobalTensor<float> c;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));
    TPipe pipe;
    TBuf<QuePosition::VECCALC> zeroBuf;
    pipe.InitBuffer(zeroBuf, EPILOGUE_TILE_ELEMENTS * sizeof(float));
    LocalTensor<float> zero = zeroBuf.Get<float>();
    Duplicate(zero, 0.0f, EPILOGUE_TILE_ELEMENTS);
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    const uint32_t rowChunks = (static_cast<uint32_t>(t->m) + EPILOGUE_TILE_ELEMENTS - 1) / EPILOGUE_TILE_ELEMENTS;
    for (uint32_t col = block; col < static_cast<uint32_t>(t->n); col += stride) {
        for (uint32_t chunk = 0; chunk < rowChunks; ++chunk) {
            const uint32_t row = chunk * EPILOGUE_TILE_ELEMENTS;
            const uint32_t count = min(EPILOGUE_TILE_ELEMENTS, static_cast<uint32_t>(t->m) - row);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(c[static_cast<uint64_t>(row) + static_cast<uint64_t>(col) * t->ldc], zero, copyParams);
        }
    }
}
void gemmex_fallback_pack_do(uint32_t b, void* s, GM_ADDR a, GM_ADDR d, GM_ADDR w, const GemmExTiling& t)
{
    gemmex_fallback_pack_kernel<<<b, nullptr, s>>>(a, d, w, t);
}

void gemmex_fallback_accumulate_do(uint32_t b, void* s, GM_ADDR w, const GemmExTiling& t)
{
    gemmex_fallback_accumulate_kernel<<<b, nullptr, s>>>(w, t);
}
void gemmex_fallback_epilogue_do(uint32_t b, void* s, GM_ADDR w, GM_ADDR c, const GemmExTiling& t)
{
    gemmex_fallback_epilogue_kernel<<<b, nullptr, s>>>(w, c, t);
}
void gemmex_fallback_zero_do(uint32_t b, void* s, GM_ADDR c, const GemmExTiling& t)
{
    gemmex_fallback_zero_kernel<<<b, nullptr, s>>>(c, t);
}

// ---- Merged from gemm_wholegrid_adapter_kernel.cpp ----

using namespace AscendC;
namespace {
constexpr uint32_t ADAPTER_ELEMENTS = 1024;
}

// Pad physical operands without scalar transposition.
extern "C" __global__ __aicore__ __vector__ void gemmex_fallback_wholegrid_pack_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR workspaceGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> a;
    GlobalTensor<float> b;
    GlobalTensor<float> packA;
    GlobalTensor<float> packB;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bGm));
    auto* workspace = reinterpret_cast<__gm__ float*>(workspaceGm);
    packA.SetGlobalBuffer(workspace + tiling.packAOffset);
    packB.SetGlobalBuffer(workspace + tiling.packBOffset);

    const bool at = tiling.transA != 0;
    const bool bt = tiling.transB != 0;
    PackPhysicalMatrix(
        a, packA, 0, at ? tiling.m : tiling.k, at ? tiling.k : tiling.m, tiling.lda, at ? tiling.pm : tiling.computeK,
        at ? tiling.computeK : tiling.pm);
    PackPhysicalMatrix(
        b, packB, 0, bt ? tiling.k : tiling.n, bt ? tiling.n : tiling.k, tiling.ldb, bt ? tiling.computeK : tiling.pn,
        bt ? tiling.pn : tiling.computeK);
}

__aicore__ __inline__ void WholeGridEpilogueRead(
    const GemmExTiling& tiling, const GlobalTensor<float>& product, const GlobalTensor<float>& c, uint64_t task,
    uint32_t segments, TQue<QuePosition::VECIN, 2>& productQueue, TQue<QuePosition::VECIN, 2>& cQueue)
{
    if (segments == 0) {
        return;
    }
    const uint32_t j = static_cast<uint32_t>(task / segments);
    const uint32_t segment = static_cast<uint32_t>(task % segments);
    const uint32_t mBase = segment * ADAPTER_ELEMENTS;
    const uint32_t count = min(ADAPTER_ELEMENTS, static_cast<uint32_t>(tiling.m) - mBase);
    LocalTensor<float> productLocal = productQueue.AllocTensor<float>();
    DataCopyPad(
        productLocal, product[static_cast<uint64_t>(j) * tiling.pm + mBase], {1, count * 4U, 0, 0, 0},
        {false, 0, 0, 0.0f});
    productQueue.EnQue(productLocal);
    if (tiling.beta != 0.0f) {
        LocalTensor<float> cLocal = cQueue.AllocTensor<float>();
        DataCopyPad(
            cLocal, c[static_cast<uint64_t>(j) * tiling.ldc + mBase], {1, count * 4U, 0, 0, 0}, {false, 0, 0, 0.0f});
        cQueue.EnQue(cLocal);
    }
}

__aicore__ __inline__ void WholeGridEpilogueCompute(
    const GemmExTiling& tiling, uint64_t task, uint32_t segments, TQue<QuePosition::VECIN, 2>& productQueue,
    TQue<QuePosition::VECIN, 2>& cQueue, TQue<QuePosition::VECOUT, 2>& outQueue)
{
    if (segments == 0) {
        return;
    }
    const uint32_t segment = static_cast<uint32_t>(task % segments);
    const uint32_t mBase = segment * ADAPTER_ELEMENTS;
    const uint32_t count = min(ADAPTER_ELEMENTS, static_cast<uint32_t>(tiling.m) - mBase);
    LocalTensor<float> productLocal = productQueue.DeQue<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
    Muls(outLocal, productLocal, tiling.alpha, count);
    productQueue.FreeTensor(productLocal);
    if (tiling.beta != 0.0f) {
        LocalTensor<float> cLocal = cQueue.DeQue<float>();
        Muls(cLocal, cLocal, tiling.beta, count);
        Add(outLocal, outLocal, cLocal, count);
        cQueue.FreeTensor(cLocal);
    }
    outQueue.EnQue(outLocal);
}

__aicore__ __inline__ void WholeGridEpilogueWrite(
    const GemmExTiling& tiling, const GlobalTensor<float>& c, uint64_t task, uint32_t segments,
    TQue<QuePosition::VECOUT, 2>& outQueue)
{
    if (segments == 0) {
        return;
    }
    const uint32_t j = static_cast<uint32_t>(task / segments);
    const uint32_t segment = static_cast<uint32_t>(task % segments);
    const uint32_t mBase = segment * ADAPTER_ELEMENTS;
    const uint32_t count = min(ADAPTER_ELEMENTS, static_cast<uint32_t>(tiling.m) - mBase);
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopyPad(c[static_cast<uint64_t>(j) * tiling.ldc + mBase], outLocal, {1, count * 4U, 0, 0, 0});
    outQueue.FreeTensor(outLocal);
}

// Shorter independent reduction segments reduce FP32 accumulation error for
// deep K. Merge every segment; no result-dependent filtering or CPU fallback.
__aicore__ inline void WholeGridSplitEpilogue(
    const GemmExTiling& t, const GlobalTensor<float>& product, const GlobalTensor<float>& c)
{
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 3 * ADAPTER_ELEMENTS * sizeof(float));
    auto sum = buffer.Get<float>();
    auto tmp = sum[ADAPTER_ELEMENTS];
    auto old = sum[2 * ADAPTER_ELEMENTS];
    const uint64_t plane = static_cast<uint64_t>(t.pm) * t.pn;
    const uint32_t segments = (t.m + ADAPTER_ELEMENTS - 1) / ADAPTER_ELEMENTS;
    for (uint64_t task = GetBlockIdx(); task < static_cast<uint64_t>(segments) * t.n; task += GetBlockNum()) {
        uint64_t col = task / segments;
        uint64_t row = (task % segments) * ADAPTER_ELEMENTS;
        uint32_t count = min(ADAPTER_ELEMENTS, static_cast<uint32_t>(t.m - row));
        DataCopyPad(sum, product[col * t.pm + row], {1, count * 4U, 0, 0, 0}, {false, 0, 0, 0.0f});
        PipeBarrier<PIPE_ALL>();
        for (int part = 1; part < t.splitParts; ++part) {
            DataCopyPad(tmp, product[part * plane + col * t.pm + row], {1, count * 4U, 0, 0, 0}, {false, 0, 0, 0.0f});
            PipeBarrier<PIPE_ALL>();
            Add(sum, sum, tmp, count);
            PipeBarrier<PIPE_ALL>();
        }
        Muls(sum, sum, t.alpha, count);
        PipeBarrier<PIPE_ALL>();
        if (t.beta != 0.0f) {
            DataCopyPad(old, c[col * t.ldc + row], {1, count * 4U, 0, 0, 0}, {false, 0, 0, 0.0f});
            PipeBarrier<PIPE_ALL>();
            Muls(old, old, t.beta, count);
            PipeBarrier<PIPE_ALL>();
            Add(sum, sum, old, count);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(c[col * t.ldc + row], sum, {1, count * 4U, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void WholeGridIdentityCopy(
    const GemmExTiling& t, const GlobalTensor<float>& product, const GlobalTensor<float>& c)
{
    constexpr uint32_t CAP = 16384;
    const uint32_t width = (t.m + 7) / 8 * 8;
    const uint32_t batch =
        min(CAP / static_cast<uint32_t>(t.pm), static_cast<uint32_t>((t.n + GetBlockNum() - 1) / GetBlockNum()));
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 2 * CAP * sizeof(float));
    auto local = buffer.Get<float>();
    auto partial = local[CAP];
    for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * batch; first < t.n;
         first += static_cast<uint64_t>(GetBlockNum()) * batch) {
        const uint32_t count = min(batch, static_cast<uint32_t>(t.n - first));
        DataCopy(local, product[first * t.pm], count * t.pm);
        PipeBarrier<PIPE_ALL>();
        for (int32_t plane = 1; plane < t.splitParts; ++plane) {
            DataCopy(partial, product[static_cast<uint64_t>(plane) * t.pm * t.pn + first * t.pm], count * t.pm);
            PipeBarrier<PIPE_ALL>();
            Add(local, local, partial, count * t.pm);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyExtParams cp{
            static_cast<uint16_t>(count), static_cast<uint32_t>(t.m) * 4U, (static_cast<uint32_t>(t.pm) - width) / 8U,
            static_cast<uint32_t>(t.ldc - t.m) * 4U, 0};
        DataCopyPad(c[first * t.ldc], local, cp);
        PipeBarrier<PIPE_ALL>();
    }
}

// Assemble full output columns using DMA destination strides in UB. This
// exchanges vector Gather work for a bounded set of independent MTE2 reads.
__aicore__ inline void WholeGridDmaTileCopy(
    const GemmExTiling& t, const GlobalTensor<float>& product, const GlobalTensor<float>& c)
{
    // Only positive <=8192 whole-grid plans reach this private writeback.
    if (t.m <= 0 || t.pm <= 0 || t.productTileM <= 0 || t.productTileN <= 0) {
        return;
    }

    constexpr uint32_t CAP = 12288;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, CAP * sizeof(float));
    auto local = buffer.Get<float>();
    const uint32_t tileM = t.productTileM;
    const uint32_t tileN = t.productTileN;
    const uint32_t nTiles = (t.pn + tileN - 1) / tileN;
    const uint32_t mTiles = (t.pm + tileM - 1) / tileM;
    const uint32_t width = (t.m + 7) / 8 * 8;
    const uint32_t batch = min(CAP / width, tileN);
    if (batch == 0) {
        return;
    }

    const uint32_t batches = (tileN + batch - 1) / batch;
    for (uint32_t job = GetBlockIdx(); job < nTiles * batches; job += GetBlockNum()) {
        const uint32_t nTile = job / batches;
        const uint32_t within = (job % batches) * batch;
        const uint32_t first = nTile * tileN + within;
        if (first >= static_cast<uint32_t>(t.n)) {
            continue;
        }
        const uint32_t count = min(batch, min(tileN - within, static_cast<uint32_t>(t.n) - first));
        for (uint32_t mt = 0; mt < mTiles; ++mt) {
            const uint32_t row = mt * tileM;
            const uint32_t valid = min(tileM, static_cast<uint32_t>(t.m) - row);
            const uint32_t dmaWidth = (valid + 7) / 8 * 8;
            DataCopyExtParams read{
                static_cast<uint16_t>(count), valid * 4U, (tileM - valid) * 4U, (width - dmaWidth) / 8U, 0};
            const uint64_t source = (static_cast<uint64_t>(mt) * nTiles + nTile) * tileM * tileN + within * tileM;
            DataCopyPad(local[row], product[source], read, {true, 0, static_cast<uint8_t>(dmaWidth - valid), 0.0f});
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams write{
            static_cast<uint16_t>(count), static_cast<uint32_t>(t.m) * 4U, 0, static_cast<uint32_t>(t.ldc - t.m) * 4U,
            0};
        DataCopyPad(c[static_cast<uint64_t>(first) * t.ldc], local, write);
        PipeBarrier<PIPE_ALL>();
    }
}

// Cube writes contiguous local tiles; AIV restores the public column stride.
__aicore__ inline void MakeWholeGridCopyIndices(
    LocalTensor<float> raw, LocalTensor<float> out, LocalTensor<uint32_t> indices, uint32_t width, uint32_t tileM,
    uint32_t batch)
{
    auto base = out.ReinterpretCast<int32_t>();
    auto scratch = raw.ReinterpretCast<int32_t>();
    CreateVecIndex(base, 0, width);
    PipeBarrier<PIPE_V>();
    ShiftRight(scratch.ReinterpretCast<uint32_t>(), base.ReinterpretCast<uint32_t>(), tileM == 128 ? 7U : 6U, width);
    PipeBarrier<PIPE_V>();
    Muls(scratch, scratch, static_cast<int32_t>((batch - 1) * tileM), width);
    PipeBarrier<PIPE_V>();
    Add(base, base, scratch, width);
    PipeBarrier<PIPE_V>();
    ShiftLeft(base.ReinterpretCast<uint32_t>(), base.ReinterpretCast<uint32_t>(), 2U, width);
    PipeBarrier<PIPE_V>();
    for (uint32_t col = 0; col < batch; ++col) {
        Adds(indices[col * width].ReinterpretCast<int32_t>(), base, static_cast<int32_t>(col * tileM * 4), width);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void StoreWholeGridColumns(
    const GemmExTiling& t, GlobalTensor<float> c, LocalTensor<float> out, uint32_t first, uint32_t count)
{
    DataCopyExtParams write{
        static_cast<uint16_t>(count), static_cast<uint32_t>(t.m) * 4U, 0, static_cast<uint32_t>(t.ldc - t.m) * 4U, 0};
    DataCopyPad(c[static_cast<uint64_t>(first) * t.ldc], out, write);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void WholeGridTileCopy(
    const GemmExTiling& t, const GlobalTensor<float>& product, const GlobalTensor<float>& c)
{
    // Only positive <=8192 whole-grid plans reach this private writeback.
    if (t.m <= 0 || t.pm <= 0 || t.productTileM <= 0 || t.productTileN <= 0) {
        return;
    }

    if ((t.pm + t.productTileM - 1) / t.productTileM <= 16) {
        WholeGridDmaTileCopy(t, product, c);
        return;
    }
    constexpr uint32_t CAP = 12288;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> values;
    TBuf<QuePosition::VECCALC> indexBuffer;
    pipe.InitBuffer(values, 2 * CAP * sizeof(float));
    pipe.InitBuffer(indexBuffer, CAP * sizeof(uint32_t));
    auto raw = values.Get<float>();
    auto out = raw[CAP];
    auto indices = indexBuffer.Get<uint32_t>();
    const uint32_t tileM = t.productTileM;
    const uint32_t tileN = t.productTileN;
    const uint32_t nTiles = (t.pn + tileN - 1) / tileN;
    const uint32_t mTiles = (t.pm + tileM - 1) / tileM;
    const uint32_t batch = min(CAP / (mTiles * tileM), tileN);
    if (batch == 0) {
        return;
    }

    const uint32_t width = (t.m + 7) / 8 * 8;
    MakeWholeGridCopyIndices(raw, out, indices, width, tileM, batch);
    const uint32_t batches = (tileN + batch - 1) / batch;
    for (uint32_t job = GetBlockIdx(); job < nTiles * batches; job += GetBlockNum()) {
        const uint32_t nTile = job / batches;
        const uint32_t within = (job % batches) * batch;
        const uint32_t first = nTile * tileN + within;
        if (first >= static_cast<uint32_t>(t.n)) {
            continue;
        }
        const uint32_t count = min(batch, min(tileN - within, static_cast<uint32_t>(t.n) - first));
        DataCopyExtParams read{
            static_cast<uint16_t>(mTiles), count * tileM * 4U, (nTiles * tileM * tileN - count * tileM) * 4U,
            (batch - count) * tileM / 8U, 0};
        DataCopyPad(
            raw, product[static_cast<uint64_t>(nTile) * tileM * tileN + within * tileM], read, {false, 0, 0, 0.0f});
        PipeBarrier<PIPE_ALL>();
        Gather(out, raw, indices, 0U, count * width);
        PipeBarrier<PIPE_ALL>();
        StoreWholeGridColumns(t, c, out, first, count);
    }
}

// Three-stage product/C read -> alpha/beta vector work -> C write pipeline.
extern "C" __global__ __aicore__ __vector__ void gemmex_fallback_wholegrid_epilogue_kernel(
    GM_ADDR workspaceGm, GM_ADDR cGm, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> product;
    GlobalTensor<float> c;
    product.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm) + tiling.productOffset);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));
    if (tiling.productTileM > 0) {
        WholeGridTileCopy(tiling, product, c);
        return;
    }
    if (tiling.alpha == 1.0f && tiling.beta == 0.0f && tiling.pm % 8 == 0) {
        WholeGridIdentityCopy(tiling, product, c);
        return;
    }
    if (tiling.splitParts > 1) {
        WholeGridSplitEpilogue(tiling, product, c);
        return;
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> productQueue;
    TQue<QuePosition::VECIN, 2> cQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(productQueue, 2, ADAPTER_ELEMENTS * sizeof(float));
    pipe.InitBuffer(cQueue, 2, ADAPTER_ELEMENTS * sizeof(float));
    pipe.InitBuffer(outQueue, 2, ADAPTER_ELEMENTS * sizeof(float));

    const uint32_t segments = (static_cast<uint32_t>(tiling.m) + ADAPTER_ELEMENTS - 1) / ADAPTER_ELEMENTS;
    const uint64_t total = static_cast<uint64_t>(tiling.n) * segments;
    const uint64_t block = GetBlockIdx();
    const uint64_t stride = GetBlockNum();
    const uint64_t iterations = block < total ? (total - block + stride - 1) / stride : 0;
    for (uint64_t step = 0; step < iterations + 2; ++step) {
        if (step < iterations) {
            WholeGridEpilogueRead(tiling, product, c, block + step * stride, segments, productQueue, cQueue);
        }
        if (step >= 1 && step - 1 < iterations) {
            WholeGridEpilogueCompute(tiling, block + (step - 1) * stride, segments, productQueue, cQueue, outQueue);
        }
        if (step >= 2) {
            WholeGridEpilogueWrite(tiling, c, block + (step - 2) * stride, segments, outQueue);
        }
    }
}

void gemmex_fallback_wholegrid_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const GemmExTiling& tiling)
{
    gemmex_fallback_wholegrid_pack_kernel<<<blocks, nullptr, stream>>>(a, b, workspace, tiling);
}

void gemmex_fallback_wholegrid_epilogue_do(
    uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmExTiling& tiling)
{
    gemmex_fallback_wholegrid_epilogue_kernel<<<blocks, nullptr, stream>>>(workspace, c, tiling);
}

// ---- Merged from gemm_wholegrid_nosplit_kernel.cpp ----

// Full-K square output ownership. Public transpose/layout handling is kept in
// the input adapter; each fixed specialization consumes one compact raw
// orientation and produces the same compact [N][M] product.

using namespace AscendC;
using namespace matmul;
namespace {
constexpr int32_t ROWS = 256;
constexpr int32_t COLS = 256;
constexpr int32_t MAX_K = 8192;
using Fp32NoTrans = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using Fp32Trans = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
// NN/NT/TT use baseK=32. TN keeps baseK=64 because aligned TN cases regress with baseK=32.
// baseM x baseN is capped at 64x256 so the fp32 accumulator occupies 64KB of L0C.
// A 128x256 base tile needs the full 128KB and leaves no headroom, which corrupts a
// subset of output tiles on the 2048-and-larger square shapes.
constexpr MatmulShapeParams kShape{ROWS, COLS, MAX_K, 64, 256, 32};
constexpr MatmulShapeParams kTnShape{ROWS, COLS, MAX_K, 64, 256, 64};
constexpr MatmulFuncParams kWholeGridFunc{
    false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true};
constexpr MatmulBiasParams kWholeGridBias{false};
constexpr MatmulConfig kConfig = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kShape, kWholeGridFunc, kWholeGridBias);
constexpr MatmulConfig kTnConfig = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kTnShape, kWholeGridFunc, kWholeGridBias);

constexpr MatmulApiStaticTiling kNnTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kConfig);
constexpr MatmulApiStaticTiling kNtTiling =
    GetMatmulApiTiling<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kConfig);
constexpr MatmulApiStaticTiling kTnTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kTnConfig);
constexpr MatmulApiStaticTiling kTtTiling = GetMatmulApiTiling<Fp32Trans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kConfig);
using NnMatmul = MatmulImpl<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kNnTiling>;
using NtMatmul = MatmulImpl<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kNtTiling>;
using TnMatmul = MatmulImpl<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kTnTiling>;
using TtMatmul = MatmulImpl<Fp32Trans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kTtTiling>;

constexpr MatmulShapeParams kSmallShape{64, 64, MAX_K, 32, 64, 128};
constexpr MatmulConfig kSmallConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kSmallShape, kWholeGridFunc, kWholeGridBias);
constexpr MatmulApiStaticTiling kSmallNnTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kSmallConfig);
using SmallNnMatmul = MatmulImpl<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kSmallNnTiling>;
constexpr MatmulApiStaticTiling kSmallNtTiling =
    GetMatmulApiTiling<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kSmallConfig);
using SmallNtMatmul = MatmulImpl<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kSmallNtTiling>;
constexpr MatmulApiStaticTiling kSmallTnTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kSmallConfig);
using SmallTnMatmul = MatmulImpl<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kSmallTnTiling>;
constexpr MatmulApiStaticTiling kSmallTtTiling =
    GetMatmulApiTiling<Fp32Trans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kSmallConfig);
using SmallTtMatmul = MatmulImpl<Fp32Trans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kSmallTtTiling>;
constexpr MatmulShapeParams kMidShape{128, 128, MAX_K, 64, 128, 64};
constexpr MatmulConfig kMidConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kMidShape, kWholeGridFunc, kWholeGridBias);
constexpr MatmulApiStaticTiling kMidNnTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kMidConfig);
using MidNnMatmul = MatmulImpl<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kMidNnTiling>;
constexpr MatmulApiStaticTiling kMidNtTiling =
    GetMatmulApiTiling<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kMidConfig);
using MidNtMatmul = MatmulImpl<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kMidNtTiling>;
constexpr MatmulApiStaticTiling kMidTnTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kMidConfig);
using MidTnMatmul = MatmulImpl<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kMidTnTiling>;
constexpr MatmulApiStaticTiling kMidTtTiling =
    GetMatmulApiTiling<Fp32Trans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kMidConfig);
using MidTtMatmul = MatmulImpl<Fp32Trans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kMidTtTiling>;
constexpr MatmulShapeParams kMidLowKShape{128, 128, 128, 64, 128, 32};
constexpr MatmulConfig kMidLowKConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kMidLowKShape, kWholeGridFunc, kWholeGridBias);
constexpr MatmulApiStaticTiling kMidNnLowKTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kMidLowKConfig);
using MidNnLowKMatmul = MatmulImpl<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kMidNnLowKTiling>;
constexpr MatmulApiStaticTiling kMidNtLowKTiling =
    GetMatmulApiTiling<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kMidLowKConfig);
using MidNtLowKMatmul = MatmulImpl<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kMidNtLowKTiling>;
constexpr MatmulApiStaticTiling kMidTnLowKTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kMidLowKConfig);
using MidTnLowKMatmul = MatmulImpl<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kMidTnLowKTiling>;
constexpr MatmulApiStaticTiling kMidTtLowKTiling =
    GetMatmulApiTiling<Fp32Trans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kMidLowKConfig);
using MidTtLowKMatmul = MatmulImpl<Fp32Trans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kMidTtLowKTiling>;

constexpr MatmulShapeParams kWideShortShape{256, 256, 128, 64, 256, 32};
constexpr MatmulShapeParams kWideShortTnShape{256, 256, 128, 64, 256, 64};
constexpr MatmulShapeParams kWideMediumNtShape{64, 256, 1024, 64, 256, 32};
constexpr MatmulConfig kWideShortConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kWideShortShape, kWholeGridFunc, kWholeGridBias);
constexpr MatmulConfig kWideShortTnConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kWideShortTnShape, kWholeGridFunc, kWholeGridBias);
constexpr MatmulConfig kWideMediumNtConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kWideMediumNtShape, kWholeGridFunc, kWholeGridBias);
constexpr MatmulApiStaticTiling kWideShortNnTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kWideShortConfig);
constexpr MatmulApiStaticTiling kWideShortTnTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans>(kWideShortTnConfig);
constexpr MatmulApiStaticTiling kWideMediumNtTiling =
    GetMatmulApiTiling<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kWideMediumNtConfig);
using WideShortNnMatmul = MatmulImpl<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kWideShortNnTiling>;
using WideShortTnMatmul = MatmulImpl<Fp32NoTrans, Fp32Trans, Fp32NoTrans, Fp32NoTrans, kWideShortTnTiling>;
using WideMediumNtMatmul = MatmulImpl<Fp32Trans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kWideMediumNtTiling>;

constexpr MatmulShapeParams kRectShortShape{112, 384, 128, 112, 192, 32};
constexpr MatmulConfig kRectShortConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kRectShortShape, kWholeGridFunc, kWholeGridBias);
constexpr MatmulApiStaticTiling kRectShortTiling =
    GetMatmulApiTiling<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans>(kRectShortConfig);
using RectShortNnMatmul = MatmulImpl<Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, Fp32NoTrans, kRectShortTiling>;
template <bool LeftTrans, bool RightTrans, class MatmulT>
__aicore__ inline void ComputeSquare(
    TPipe& pipe, GlobalTensor<float> left, GlobalTensor<float> right, GlobalTensor<float> output,
    const GemmExTiling& tiling, int32_t rows, int32_t cols, int32_t computeK, int32_t outputStride, int32_t partK,
    int32_t kStart)
{
    MatmulT matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    matmul.DisableBias();
    matmul.SetOrgShape(LeftTrans ? tiling.n : rows, tiling.m, computeK, computeK, outputStride);
    matmul.SetSingleShape(rows, cols, partK);
    matmul.SetTensorA(left[LeftTrans ? static_cast<uint64_t>(kStart) * tiling.n : kStart], LeftTrans);
    matmul.SetTensorB(right[RightTrans ? kStart : static_cast<uint64_t>(kStart) * tiling.m], RightTrans);
    matmul.IterateAll(output, 0);
    matmul.End();
}

template <bool LeftTrans, bool RightTrans, class MatmulT, int Tile = 256, int RowTile = Tile>
__aicore__ inline void RunSquare(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GemmExTiling tiling)
{
    const bool direct = tiling.splitParts <= 1 && tiling.alpha == 1.0f && tiling.beta == 0.0f &&
                        tiling.ldc == tiling.m &&
                        !(tiling.k <= 256 && static_cast<uint64_t>(tiling.m) * tiling.n >= 65536 && tiling.m % 8 != 0);
    const int32_t outputRows = direct ? tiling.n : tiling.pn;
    const int32_t outputCols = direct ? tiling.m : tiling.pm;
    const int32_t outputStride = tiling.productTileM > 0 ? Tile : (direct ? tiling.ldc : tiling.pm);
    tiling.m = tiling.pm;
    tiling.n = tiling.pn;
    const int32_t nTiles = (tiling.n + RowTile - 1) / RowTile;
    const int32_t grid = nTiles * ((tiling.m + Tile - 1) / Tile);
    const int32_t split = tiling.splitParts > 1 ? tiling.splitParts : 1;
    const int32_t part = static_cast<int32_t>(GetBlockIdx()) / grid;
    const int32_t tile = static_cast<int32_t>(GetBlockIdx()) % grid;
    const int32_t nTile = tile % nTiles;
    const int32_t mTile = tile / nTiles;
    const int32_t nBase = nTile * RowTile;
    const int32_t mBase = mTile * Tile;
    const int32_t rows = min(RowTile, outputRows - nBase);
    const int32_t cols = min(Tile, outputCols - mBase);
    const int32_t computeK = tiling.computeK;
    const int32_t partK = computeK / split;
    const int32_t kStart = part * partK;

    GlobalTensor<float> left;
    GlobalTensor<float> right;
    GlobalTensor<float> output;
    auto* b = reinterpret_cast<__gm__ float*>(bGm);
    auto* a = reinterpret_cast<__gm__ float*>(aGm);
    if constexpr (LeftTrans) {
        left.SetGlobalBuffer(b + nBase, static_cast<uint64_t>(computeK) * tiling.n - nBase);
    } else {
        left.SetGlobalBuffer(b + static_cast<uint64_t>(nBase) * computeK, static_cast<uint64_t>(rows) * computeK);
    }
    if constexpr (RightTrans) {
        right.SetGlobalBuffer(a + static_cast<uint64_t>(mBase) * computeK, static_cast<uint64_t>(cols) * computeK);
    } else {
        right.SetGlobalBuffer(a + mBase, static_cast<uint64_t>(computeK) * tiling.m - mBase);
    }
    output.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(cGm) + (tiling.productTileM > 0 ?
                                                    static_cast<uint64_t>(tile) * Tile * RowTile :
                                                    static_cast<uint64_t>(part) * tiling.m * tiling.n +
                                                        static_cast<uint64_t>(nBase) * outputStride + mBase),
        static_cast<uint64_t>(rows) * outputStride);

    TPipe pipe;
    ComputeSquare<LeftTrans, RightTrans, MatmulT>(
        pipe, left, right, output, tiling, rows, cols, computeK, outputStride, partK, kStart);
}
} // namespace

extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_wholegrid_square_nn_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    const uint64_t area = static_cast<uint64_t>(tiling.pm) * tiling.pn;
    if ((area >= 262144 && area < 4194304 && tiling.computeK <= 128 && tiling.transB == 0 && tiling.m % 16 == 0 &&
         tiling.n % 16 == 0) &&
        ((tiling.pm + 383) / 384) * ((tiling.pn + 111) / 112) <= tiling.cubeBlocks) {
        RunSquare<false, false, RectShortNnMatmul, 384, 112>(a, b, c, tiling);
    } else if (
        area >= 262144 && area < 4194304 && tiling.computeK <= 128 && tiling.transB == 0 && tiling.m % 16 == 0 &&
        tiling.n % 16 == 0) {
        RunSquare<false, false, WideShortNnMatmul, 256>(a, b, c, tiling);
    } else if (area < 65536) {
        RunSquare<false, false, SmallNnMatmul, 64>(a, b, c, tiling);
    } else if (area < 262144) {
        RunSquare<false, false, MidNnMatmul, 128, 64>(a, b, c, tiling);
    } else if (area < 1048576 && tiling.computeK <= 96) {
        RunSquare<false, false, MidNnLowKMatmul, 128>(a, b, c, tiling);
    } else if (area < 4194304) {
        RunSquare<false, false, MidNnMatmul, 128>(a, b, c, tiling);
    } else {
        RunSquare<false, false, NnMatmul>(a, b, c, tiling);
    }
}
extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_wholegrid_square_nt_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    const uint64_t area = static_cast<uint64_t>(tiling.pm) * tiling.pn;
    if (area >= 65536 && area < 262144 && tiling.computeK > 128 && tiling.computeK <= 1024 && tiling.transA == 0 &&
        tiling.transB != 0 && tiling.m % 16 == 0 && tiling.n % 16 == 0) {
        RunSquare<true, false, WideMediumNtMatmul, 256, 64>(a, b, c, tiling);
    } else if (area < 65536) {
        RunSquare<true, false, SmallNtMatmul, 64>(a, b, c, tiling);
    } else if (area < 262144) {
        RunSquare<true, false, MidNtMatmul, 128, 64>(a, b, c, tiling);
    } else if (area < 1048576 && tiling.computeK <= 96) {
        RunSquare<true, false, MidNtLowKMatmul, 128>(a, b, c, tiling);
    } else if (area < 4194304) {
        RunSquare<true, false, MidNtMatmul, 128>(a, b, c, tiling);
    } else {
        RunSquare<true, false, NtMatmul>(a, b, c, tiling);
    }
}
extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_wholegrid_square_tn_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    const uint64_t area = static_cast<uint64_t>(tiling.pm) * tiling.pn;
    if (area >= 262144 && area < 4194304 && tiling.computeK <= 128 && tiling.transB == 0 && tiling.m % 16 == 0 &&
        tiling.n % 16 == 0) {
        RunSquare<false, true, WideShortTnMatmul, 256>(a, b, c, tiling);
    } else if (area < 65536) {
        RunSquare<false, true, SmallTnMatmul, 64>(a, b, c, tiling);
    } else if (area < 262144) {
        RunSquare<false, true, MidTnMatmul, 128, 64>(a, b, c, tiling);
    } else if (area < 1048576 && tiling.computeK <= 96) {
        RunSquare<false, true, MidTnLowKMatmul, 128>(a, b, c, tiling);
    } else if (area < 4194304) {
        RunSquare<false, true, MidTnMatmul, 128>(a, b, c, tiling);
    } else {
        RunSquare<false, true, TnMatmul>(a, b, c, tiling);
    }
}
extern "C" __global__ __aicore__ __cube__ void gemmex_fallback_wholegrid_square_tt_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    const uint64_t area = static_cast<uint64_t>(tiling.pm) * tiling.pn;
    if (area < 65536) {
        RunSquare<true, true, SmallTtMatmul, 64>(a, b, c, tiling);
    } else if (area < 262144) {
        RunSquare<true, true, MidTtMatmul, 128, 64>(a, b, c, tiling);
    } else if (area < 1048576 && tiling.computeK <= 96) {
        RunSquare<true, true, MidTtLowKMatmul, 128>(a, b, c, tiling);
    } else if (area < 4194304) {
        RunSquare<true, true, MidTtMatmul, 128>(a, b, c, tiling);
    } else {
        RunSquare<true, true, TtMatmul>(a, b, c, tiling);
    }
}

void gemmex_fallback_wholegrid_nosplit_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmExTiling& tiling)
{
    if (tiling.transB == 0) {
        if (tiling.transA == 0) {
            gemmex_fallback_wholegrid_square_nn_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
        } else {
            gemmex_fallback_wholegrid_square_tn_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
        }
    } else if (tiling.transA == 0) {
        gemmex_fallback_wholegrid_square_nt_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
    } else {
        gemmex_fallback_wholegrid_square_tt_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
    }
}
