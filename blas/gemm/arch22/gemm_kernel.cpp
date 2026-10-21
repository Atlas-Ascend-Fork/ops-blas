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
#include "gemm_kernel.h"

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

constexpr int32_t kRowsMatmulMaxRows = 64;
// Partition internal output rows into whole waves of the runtime Cube-core
// count; rowsPerBlock stays within the proven single-core envelope.
constexpr MatmulShapeParams kRowsShape{kRowsMatmulMaxRows, 1024, 1024, 64, 256, 64};
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

uint32_t GetRowsMatmulBlocks(int32_t pn, uint32_t coreBlocks)
{
    // Fewest blocks at the proven 64-row envelope, padded up to whole waves
    // of the runtime Cube-core count. The kernel derives its slice width from
    // GetBlockNum(), so host and device always agree on ownership.
    const uint32_t minBlocks = (static_cast<uint32_t>(pn) + kRowsMatmulMaxRows - 1) / kRowsMatmulMaxRows;
    if (coreBlocks == 0)
        return 0;
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
    const GemmArch22Tiling& tiling, int32_t rowBase, int32_t rows, GM_ADDR bGm, GM_ADDR aGm, GM_ADDR outputGm,
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
    const GemmArch22Tiling& tiling, int32_t rowBase, int32_t rows, GM_ADDR bGm, GM_ADDR aGm, GM_ADDR outputGm,
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
__aicore__ inline void RunRows(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, const GemmArch22Tiling& tiling)
{
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    int32_t rowBase = 0;
    int32_t rows = 0;
    if (!GetRowsWindow(tiling.pn, rowBase, rows))
        return;

    GlobalTensor<float> left, right, output;
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
extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_matmul_rows_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunRows<false, false, RowsMatmul>(aGm, bGm, outputGm, tiling);
}

extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_matmul_rows_right_trans_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunRows<false, true, RowsRightTransMatmul>(aGm, bGm, outputGm, tiling);
}

extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_matmul_rows_left_trans_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunRows<true, false, RowsLeftTransMatmul>(aGm, bGm, outputGm, tiling);
}

extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_matmul_rows_both_trans_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunRows<true, true, RowsBothTransMatmul>(aGm, bGm, outputGm, tiling);
}

void sgemm_arch22_matmul_rows_do(void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling)
{
    const uint32_t blocks = GetRowsMatmulBlocks(tiling.pn, tiling.cubeBlocks);
    if (blocks == 0)
        return;
    sgemm_arch22_matmul_rows_kernel<<<blocks, nullptr, stream>>>(a, b, output, tiling);
}

void sgemm_arch22_matmul_rows_right_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling)
{
    const uint32_t blocks = GetRowsMatmulBlocks(tiling.pn, tiling.cubeBlocks);
    sgemm_arch22_matmul_rows_right_trans_kernel<<<blocks, nullptr, stream>>>(a, b, output, tiling);
}

void sgemm_arch22_matmul_rows_left_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling)
{
    const uint32_t blocks = GetRowsMatmulBlocks(tiling.pn, tiling.cubeBlocks);
    sgemm_arch22_matmul_rows_left_trans_kernel<<<blocks, nullptr, stream>>>(a, b, output, tiling);
}

void sgemm_arch22_matmul_rows_both_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling)
{
    const uint32_t blocks = GetRowsMatmulBlocks(tiling.pn, tiling.cubeBlocks);
    sgemm_arch22_matmul_rows_both_trans_kernel<<<blocks, nullptr, stream>>>(a, b, output, tiling);
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

constexpr int32_t kTileMaxRows = 256; // ceil(1024 / 4 row slabs)
constexpr int32_t kTileMaxCols = 208; // align8(ceil(1024 / 5 col slabs))
constexpr int32_t kTileRowSlabs = 4;
constexpr int32_t kTileColSlabs = 5;
constexpr MatmulShapeParams kTilesShape{kTileMaxRows, kTileMaxCols, 1024, 64, 208, 64};
constexpr MatmulFuncParams kTilesFunc{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                      true,  true};
constexpr MatmulBiasParams kTilesBias{false};
constexpr MatmulConfig kTilesConfig = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(kTilesShape, kTilesFunc, kTilesBias);
constexpr MatmulApiStaticTiling kTilesStaticTiling =
    GetMatmulApiTiling<DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32>(kTilesConfig);
using TilesMatmul =
    MatmulImpl<DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, DenseNoTransFp32, kTilesStaticTiling>;

constexpr int32_t kScatterTileFloats = 1024;
} // namespace

// Scatter one (tile,row) segment of the internal product into public C with
// the alpha/beta epilogue. Internal tile row r of block b covers public
// column (nBase + rowBase_b + r), elements [mBase + colBase_b, +cols_b).
__aicore__ __inline__ void ScatterTileSegment(
    const GemmArch22Tiling* t, const GlobalTensor<float>& tiles, const GlobalTensor<float>& c, int32_t tile,
    int32_t row, int32_t rowsPerSlab, int32_t colsPerSlab, TQue<QuePosition::VECIN, 2>& inQueue,
    TQue<QuePosition::VECIN, 2>& cQueue, TQue<QuePosition::VECOUT, 2>& outQueue)
{
    const int32_t rowSlab = tile / kTileColSlabs;
    const int32_t colSlab = tile % kTileColSlabs;
    const int32_t rowBase = rowSlab * rowsPerSlab;
    const int32_t colBase = colSlab * colsPerSlab;
    const int32_t globalRow = rowBase + row;
    if (globalRow >= t->pn || colBase >= t->pm)
        return;
    const int32_t cols = t->pm - colBase < colsPerSlab ? t->pm - colBase : colsPerSlab;
    const uint64_t srcOffset =
        static_cast<uint64_t>(tile) * kTileMaxRows * kTileMaxCols + static_cast<uint64_t>(row) * kTileMaxCols;
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
// b * kTileMaxRows * kTileMaxCols, row-major [rows_b][cols_b].
extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_matmul_tiles_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outputGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    const int32_t block = static_cast<int32_t>(GetBlockIdx());
    const int32_t rowSlab = block / kTileColSlabs;
    const int32_t colSlab = block % kTileColSlabs;
    const int32_t rowsPerSlab = (tiling.pn + kTileRowSlabs - 1) / kTileRowSlabs;
    const int32_t colsPerSlab = ((tiling.pm + kTileColSlabs - 1) / kTileColSlabs + 7) / 8 * 8;
    const int32_t rowBase = rowSlab * rowsPerSlab;
    const int32_t colBase = colSlab * colsPerSlab;
    if (rowBase >= tiling.pn || colBase >= tiling.pm)
        return;
    const int32_t rows = tiling.pn - rowBase < rowsPerSlab ? tiling.pn - rowBase : rowsPerSlab;
    const int32_t cols = tiling.pm - colBase < colsPerSlab ? tiling.pm - colBase : colsPerSlab;

    GlobalTensor<float> left, right, output;
    left.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(bGm) + static_cast<uint64_t>(rowBase) * tiling.pk,
        static_cast<uint64_t>(rows) * tiling.pk);
    // Column slab of the row-major [pk][pm] right operand: element base
    // colBase, full row stride pm preserved through SetOrgShape.
    right.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(aGm) + colBase, static_cast<uint64_t>(tiling.pk) * tiling.pm - colBase);
    output.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(outputGm) + tiling.productOffset +
            static_cast<uint64_t>(block) * kTileMaxRows * kTileMaxCols,
        static_cast<uint64_t>(rows) * kTileMaxCols);

    TPipe pipe;
    TilesMatmul matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    matmul.DisableBias();
    // Preserve the full input strides (orgN controls the non-transposed
    // right operand) while making each workspace tile's C row stride fixed.
    matmul.SetOrgShape(tiling.pn, tiling.pm, tiling.pk, tiling.pk, kTileMaxCols);
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
extern "C" __global__ __aicore__ __vector__ void sgemm_arch22_scatter_tiles_kernel(
    GM_ADDR workspaceGm, GM_ADDR cGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    GlobalTensor<float> tiles, c;
    tiles.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm) + t->productOffset);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECIN, 2> cQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(inQueue, 2, kScatterTileFloats * sizeof(float));
    pipe.InitBuffer(cQueue, 2, kScatterTileFloats * sizeof(float));
    pipe.InitBuffer(outQueue, 2, kScatterTileFloats * sizeof(float));

    const int32_t rowsPerSlab = (t->pn + kTileRowSlabs - 1) / kTileRowSlabs;
    const int32_t colsPerSlab = ((t->pm + kTileColSlabs - 1) / kTileColSlabs + 7) / 8 * 8;

    // Each AIV block strides over tile-row segments of all 20 tiles.
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    // Enumerate all (tile, row) segments with nonzero extent.
    // segment index s -> tile = s / rowsPerSlab, row = s % rowsPerSlab
    const int32_t totalSegments = kTileRowSlabs * kTileColSlabs * rowsPerSlab;
    for (int32_t s = static_cast<int32_t>(block); s < totalSegments; s += static_cast<int32_t>(stride)) {
        ScatterTileSegment(
            t, tiles, c, s / rowsPerSlab, s % rowsPerSlab, rowsPerSlab, colsPerSlab, inQueue, cQueue, outQueue);
    }
}

void sgemm_arch22_matmul_tiles_do(void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling)
{
    sgemm_arch22_matmul_tiles_kernel<<<kTileRowSlabs * kTileColSlabs, nullptr, stream>>>(a, b, output, tiling);
}

void sgemm_arch22_scatter_tiles_do(
    uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmArch22Tiling& tiling)
{
    sgemm_arch22_scatter_tiles_kernel<<<blocks, nullptr, stream>>>(workspace, c, tiling);
}

// ---- Merged from gemm_vector_kernel.cpp ----

// The rows Matmul section above is Cube-only. The following pack and
// post-processing kernels require the full AscendC Vector interface.
#undef ASCENDC_CUBE_ONLY

using namespace AscendC;
namespace {
constexpr uint32_t kPackTileElements = 1024;
constexpr uint32_t kAccumulateTileElements = 4096;
constexpr uint32_t kEpilogueTileElements = 1024;
constexpr uint32_t kFp32PerDataBlock = 8;
} // namespace

// Keep transposed public operands in their raw physical orientations. Each
// physical row is copied with one DMA and an aligned UB tail is zero-filled.
__aicore__ __inline__ void PackRawOperandA(
    const GemmArch22Tiling* t, const GlobalTensor<float>& a, const GlobalTensor<float>& packA)
{
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> rawQueue;
    pipe.InitBuffer(rawQueue, 2, kPackTileElements * sizeof(float));
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    const uint32_t aRows = t->transA != 0 ? static_cast<uint32_t>(t->m) : static_cast<uint32_t>(t->k);
    const uint32_t aWidth = t->transA != 0 ? static_cast<uint32_t>(t->pk) : static_cast<uint32_t>(t->pm);
    if (block < aRows) {
        uint32_t i = block;
        uint32_t nextAOut = block;
        LocalTensor<float> first = rawQueue.AllocTensor<float>();
        uint64_t sourceOffset = t->transA != 0 ?
                                    static_cast<uint64_t>(t->kBase) + (static_cast<uint64_t>(t->mBase) + i) * t->lda :
                                    static_cast<uint64_t>(t->mBase) + (static_cast<uint64_t>(t->kBase) + i) * t->lda;
        DataCopyPad(
            first, a[sourceOffset], {1, static_cast<uint32_t>((t->transA != 0 ? t->k : t->m) * sizeof(float)), 0, 0, 0},
            DataCopyPadExtParams<float>{
                true, 0, static_cast<uint8_t>(aWidth - static_cast<uint32_t>(t->transA != 0 ? t->k : t->m)), 0.0f});
        rawQueue.EnQue(first);
        i += stride;
        for (; i < aRows; i += stride) {
            LocalTensor<float> next = rawQueue.AllocTensor<float>();
            sourceOffset = t->transA != 0 ?
                               static_cast<uint64_t>(t->kBase) + (static_cast<uint64_t>(t->mBase) + i) * t->lda :
                               static_cast<uint64_t>(t->mBase) + (static_cast<uint64_t>(t->kBase) + i) * t->lda;
            DataCopyPad(
                next, a[sourceOffset],
                {1, static_cast<uint32_t>((t->transA != 0 ? t->k : t->m) * sizeof(float)), 0, 0, 0},
                DataCopyPadExtParams<float>{
                    true, 0, static_cast<uint8_t>(aWidth - static_cast<uint32_t>(t->transA != 0 ? t->k : t->m)), 0.0f});
            LocalTensor<float> previous = rawQueue.DeQue<float>();
            DataCopy(packA[uint64_t(nextAOut) * aWidth], previous, aWidth);
            rawQueue.FreeTensor(previous);
            rawQueue.EnQue(next);
            nextAOut += stride;
        }
        LocalTensor<float> last = rawQueue.DeQue<float>();
        DataCopy(packA[uint64_t(nextAOut) * aWidth], last, aWidth);
        rawQueue.FreeTensor(last);
    }
}

__aicore__ __inline__ void PackRawOperandB(
    const GemmArch22Tiling* t, const GlobalTensor<float>& b, const GlobalTensor<float>& packB)
{
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> rawQueue;
    pipe.InitBuffer(rawQueue, 2, kPackTileElements * sizeof(float));
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    const uint32_t bRows = t->transB != 0 ? static_cast<uint32_t>(t->k) : static_cast<uint32_t>(t->n);
    const uint32_t bWidth = t->transB != 0 ? static_cast<uint32_t>(t->pn) : static_cast<uint32_t>(t->pk);
    if (block < bRows) {
        uint32_t p = block;
        uint32_t nextBOut = block;
        LocalTensor<float> first = rawQueue.AllocTensor<float>();
        uint64_t sourceOffset = t->transB != 0 ?
                                    static_cast<uint64_t>(t->nBase) + (static_cast<uint64_t>(t->kBase) + p) * t->ldb :
                                    static_cast<uint64_t>(t->kBase) + (static_cast<uint64_t>(t->nBase) + p) * t->ldb;
        DataCopyPad(
            first, b[sourceOffset], {1, static_cast<uint32_t>((t->transB != 0 ? t->n : t->k) * sizeof(float)), 0, 0, 0},
            DataCopyPadExtParams<float>{
                true, 0, static_cast<uint8_t>(bWidth - static_cast<uint32_t>(t->transB != 0 ? t->n : t->k)), 0.0f});
        rawQueue.EnQue(first);
        p += stride;
        for (; p < bRows; p += stride) {
            LocalTensor<float> next = rawQueue.AllocTensor<float>();
            sourceOffset = t->transB != 0 ?
                               static_cast<uint64_t>(t->nBase) + (static_cast<uint64_t>(t->kBase) + p) * t->ldb :
                               static_cast<uint64_t>(t->kBase) + (static_cast<uint64_t>(t->nBase) + p) * t->ldb;
            DataCopyPad(
                next, b[sourceOffset],
                {1, static_cast<uint32_t>((t->transB != 0 ? t->n : t->k) * sizeof(float)), 0, 0, 0},
                DataCopyPadExtParams<float>{
                    true, 0, static_cast<uint8_t>(bWidth - static_cast<uint32_t>(t->transB != 0 ? t->n : t->k)), 0.0f});
            LocalTensor<float> previous = rawQueue.DeQue<float>();
            DataCopy(packB[uint64_t(nextBOut) * bWidth], previous, bWidth);
            rawQueue.FreeTensor(previous);
            rawQueue.EnQue(next);
            nextBOut += stride;
        }
        LocalTensor<float> last = rawQueue.DeQue<float>();
        DataCopy(packB[uint64_t(nextBOut) * bWidth], last, bWidth);
        rawQueue.FreeTensor(last);
    }
}

__aicore__ __inline__ void PackRawOperands(
    const GemmArch22Tiling* t, const GlobalTensor<float>& a, const GlobalTensor<float>& b,
    const GlobalTensor<float>& packA, const GlobalTensor<float>& packB)
{
    PackRawOperandA(t, a, packA);
    PackRawOperandB(t, b, packB);
}

// NN-only safe DMA path. Complete 32-B transfers use DMA; short tails are
// filled element-wise because some 910B runtime combinations reject
// non-block-aligned GM->UB DataCopyPad lengths.
__aicore__ __inline__ void PackNnOperandA(
    const GemmArch22Tiling* t, const GlobalTensor<float>& a, const GlobalTensor<float>& packA)
{
    // MTE2 transfer before the tensor is enqueued for MTE3.
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> packQueue;
    pipe.InitBuffer(packQueue, 2, kPackTileElements * sizeof(float));
    const uint32_t aPrefix = static_cast<uint32_t>(t->m) / 8 * 8;
    uint32_t pending = 0;
    uint32_t nextAOut = GetBlockIdx();
    for (uint32_t p = GetBlockIdx(); p < static_cast<uint32_t>(t->pk); p += GetBlockNum()) {
        LocalTensor<float> current = packQueue.AllocTensor<float>();
        const bool valid = p < static_cast<uint32_t>(t->k);
        const uint64_t sourceOffset = static_cast<uint64_t>(t->mBase) + (static_cast<uint64_t>(t->kBase) + p) * t->lda;
        if (pending != 0) {
            LocalTensor<float> previous = packQueue.DeQue<float>();
            DataCopy(packA[uint64_t(nextAOut) * t->pm], previous, static_cast<uint32_t>(t->pm));
            packQueue.FreeTensor(previous);
            nextAOut += GetBlockNum();
            --pending;
        }
        PipeBarrier<PIPE_ALL>();
        if (valid && aPrefix != 0)
            DataCopy(current, a[sourceOffset], aPrefix);
        for (uint32_t i = valid ? aPrefix : 0; i < static_cast<uint32_t>(t->pm); ++i)
            current.SetValue(i, 0.0f);
        if (valid)
            for (uint32_t i = aPrefix; i < static_cast<uint32_t>(t->m); ++i)
                current.SetValue(i, a.GetValue(sourceOffset + i));
        packQueue.EnQue(current);
        ++pending;
    }
    if (pending != 0) {
        LocalTensor<float> previous = packQueue.DeQue<float>();
        DataCopy(packA[uint64_t(nextAOut) * t->pm], previous, static_cast<uint32_t>(t->pm));
        packQueue.FreeTensor(previous);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ __inline__ void PackNnOperandB(
    const GemmArch22Tiling* t, const GlobalTensor<float>& b, const GlobalTensor<float>& packB)
{
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> packQueue;
    pipe.InitBuffer(packQueue, 2, kPackTileElements * sizeof(float));
    const uint32_t bPrefix = static_cast<uint32_t>(t->k) / 8 * 8;
    uint32_t pending = 0;
    uint32_t nextBOut = GetBlockIdx();
    for (uint32_t j = GetBlockIdx(); j < static_cast<uint32_t>(t->pn); j += GetBlockNum()) {
        LocalTensor<float> current = packQueue.AllocTensor<float>();
        const bool valid = j < static_cast<uint32_t>(t->n);
        const uint64_t sourceOffset = static_cast<uint64_t>(t->kBase) + (static_cast<uint64_t>(t->nBase) + j) * t->ldb;
        if (pending != 0) {
            LocalTensor<float> previous = packQueue.DeQue<float>();
            DataCopy(packB[uint64_t(nextBOut) * t->pk], previous, static_cast<uint32_t>(t->pk));
            packQueue.FreeTensor(previous);
            nextBOut += GetBlockNum();
            --pending;
        }
        PipeBarrier<PIPE_ALL>();
        if (valid && bPrefix != 0)
            DataCopy(current, b[sourceOffset], bPrefix);
        for (uint32_t p = valid ? bPrefix : 0; p < static_cast<uint32_t>(t->pk); ++p)
            current.SetValue(p, 0.0f);
        if (valid)
            for (uint32_t p = bPrefix; p < static_cast<uint32_t>(t->k); ++p)
                current.SetValue(p, b.GetValue(sourceOffset + p));
        packQueue.EnQue(current);
        ++pending;
    }
    if (pending != 0) {
        LocalTensor<float> previous = packQueue.DeQue<float>();
        DataCopy(packB[uint64_t(nextBOut) * t->pk], previous, static_cast<uint32_t>(t->pk));
        packQueue.FreeTensor(previous);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ __inline__ void PackNnOperands(
    const GemmArch22Tiling* t, const GlobalTensor<float>& a, const GlobalTensor<float>& b,
    const GlobalTensor<float>& packA, const GlobalTensor<float>& packB)
{
    PackNnOperandA(t, a, packA);
    PackNnOperandB(t, b, packB);
}

// Generic strided path for transpose/layout combinations that cannot use the
// contiguous NN DMA contract.
__aicore__ __inline__ void PackGenericOperands(
    const GemmArch22Tiling* t, const GlobalTensor<float>& a, const GlobalTensor<float>& b,
    const GlobalTensor<float>& packA, const GlobalTensor<float>& packB)
{
    TPipe pipe;
    TBuf<QuePosition::VECCALC> rowBuf;
    pipe.InitBuffer(rowBuf, kPackTileElements * sizeof(float));
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

extern "C" __global__ __aicore__ __vector__ void sgemm_arch22_pack_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR workspaceGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    GlobalTensor<float> a, b, packA, packB;
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
    const GemmArch22Tiling* t, const GlobalTensor<float>& product, const GlobalTensor<float>& accum, uint64_t offset,
    TQue<QuePosition::VECIN, 2>& productQueue, TQue<QuePosition::VECIN, 2>& accumInQueue)
{
    const uint64_t count = static_cast<uint64_t>(t->pn) * t->pm;
    const uint32_t current =
        static_cast<uint32_t>(count - offset < kAccumulateTileElements ? count - offset : kAccumulateTileElements);
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
    const GemmArch22Tiling* t, const GlobalTensor<float>& product, const GlobalTensor<float>& accum, uint64_t offset,
    TQue<QuePosition::VECIN, 2>& productQueue, TQue<QuePosition::VECIN, 2>& accumInQueue,
    TQue<QuePosition::VECOUT, 2>& accumOutQueue)
{
    const uint64_t count = static_cast<uint64_t>(t->pn) * t->pm;
    const uint32_t current =
        static_cast<uint32_t>(count - offset < kAccumulateTileElements ? count - offset : kAccumulateTileElements);
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
    const GemmArch22Tiling* t, const GlobalTensor<float>& accum, uint64_t offset,
    TQue<QuePosition::VECOUT, 2>& accumOutQueue)
{
    const uint64_t count = static_cast<uint64_t>(t->pn) * t->pm;
    const uint32_t current =
        static_cast<uint32_t>(count - offset < kAccumulateTileElements ? count - offset : kAccumulateTileElements);
    LocalTensor<float> outLocal = accumOutQueue.DeQue<float>();
    DataCopy(accum[offset], outLocal, current);
    accumOutQueue.FreeTensor(outLocal);
}

extern "C" __global__ __aicore__ void sgemm_arch22_accumulate_kernel(GM_ADDR workspaceGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    auto* w = reinterpret_cast<__gm__ float*>(workspaceGm);
    GlobalTensor<float> product, accum;
    product.SetGlobalBuffer(w + t->productOffset);
    accum.SetGlobalBuffer(w + t->accumOffset);
    const uint64_t count = static_cast<uint64_t>(t->pn) * t->pm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> productQueue;
    TQue<QuePosition::VECIN, 2> accumInQueue;
    TQue<QuePosition::VECOUT, 2> accumOutQueue;
    pipe.InitBuffer(productQueue, 2, kAccumulateTileElements * sizeof(float));
    pipe.InitBuffer(accumInQueue, 2, kAccumulateTileElements * sizeof(float));
    pipe.InitBuffer(accumOutQueue, 2, kAccumulateTileElements * sizeof(float));
    const uint64_t tileCount = (count + kAccumulateTileElements - 1) / kAccumulateTileElements;
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
                t, product, accum, (block + step * stride) * kAccumulateTileElements, productQueue, accumInQueue);
        }
        if (step >= 1 && step - 1 < iterations) {
            AccumulateStage1(
                t, product, accum, (block + (step - 1) * stride) * kAccumulateTileElements, productQueue, accumInQueue,
                accumOutQueue);
        }
        if (step >= 2) {
            AccumulateStage2(t, accum, (block + (step - 2) * stride) * kAccumulateTileElements, accumOutQueue);
        }
    }
}
__aicore__ __inline__ void EpilogueStage0(
    const GemmArch22Tiling* t, const GlobalTensor<float>& accum, const GlobalTensor<float>& c, uint32_t j,
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
    const GemmArch22Tiling* t, uint32_t aligned, float alpha, float beta, TQue<QuePosition::VECIN, 2>& accumQueue,
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
    const GemmArch22Tiling* t, const GlobalTensor<float>& c, uint32_t j, const DataCopyExtParams& copyParams,
    TQue<QuePosition::VECOUT, 2>& outQueue)
{
    const uint64_t ci = static_cast<uint64_t>(t->mBase) + (static_cast<uint64_t>(t->nBase) + j) * t->ldc;
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopyPad(c[ci], outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}

extern "C" __global__ __aicore__ void sgemm_arch22_epilogue_kernel(
    GM_ADDR workspaceGm, GM_ADDR cGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    GlobalTensor<float> accum, c;
    accum.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm) + t->accumOffset);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> accumQueue;
    TQue<QuePosition::VECIN, 2> cQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(accumQueue, 2, kEpilogueTileElements * sizeof(float));
    pipe.InitBuffer(cQueue, 2, kEpilogueTileElements * sizeof(float));
    pipe.InitBuffer(outQueue, 2, kEpilogueTileElements * sizeof(float));

    const float alpha = t->alpha;
    const float beta = t->beta;
    const uint32_t logical = static_cast<uint32_t>(t->m);
    const uint32_t aligned = (logical + kFp32PerDataBlock - 1) / kFp32PerDataBlock * kFp32PerDataBlock;
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
extern "C" __global__ __aicore__ void sgemm_arch22_zero_kernel(GM_ADDR cGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const auto* t = &tiling;
    GlobalTensor<float> c;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));
    TPipe pipe;
    TBuf<QuePosition::VECCALC> zeroBuf;
    pipe.InitBuffer(zeroBuf, kEpilogueTileElements * sizeof(float));
    LocalTensor<float> zero = zeroBuf.Get<float>();
    Duplicate(zero, 0.0f, kEpilogueTileElements);
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    const uint32_t rowChunks = (static_cast<uint32_t>(t->m) + kEpilogueTileElements - 1) / kEpilogueTileElements;
    for (uint32_t col = block; col < static_cast<uint32_t>(t->n); col += stride) {
        for (uint32_t chunk = 0; chunk < rowChunks; ++chunk) {
            const uint32_t row = chunk * kEpilogueTileElements;
            const uint32_t count = min(kEpilogueTileElements, static_cast<uint32_t>(t->m) - row);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(c[static_cast<uint64_t>(row) + static_cast<uint64_t>(col) * t->ldc], zero, copyParams);
        }
    }
}
void sgemm_arch22_pack_do(uint32_t b, void* s, GM_ADDR a, GM_ADDR d, GM_ADDR w, const GemmArch22Tiling& t)
{
    sgemm_arch22_pack_kernel<<<b, nullptr, s>>>(a, d, w, t);
}

void sgemm_arch22_accumulate_do(uint32_t b, void* s, GM_ADDR w, const GemmArch22Tiling& t)
{
    sgemm_arch22_accumulate_kernel<<<b, nullptr, s>>>(w, t);
}
void sgemm_arch22_epilogue_do(uint32_t b, void* s, GM_ADDR w, GM_ADDR c, const GemmArch22Tiling& t)
{
    sgemm_arch22_epilogue_kernel<<<b, nullptr, s>>>(w, c, t);
}
void sgemm_arch22_zero_do(uint32_t b, void* s, GM_ADDR c, const GemmArch22Tiling& t)
{
    sgemm_arch22_zero_kernel<<<b, nullptr, s>>>(c, t);
}

// ---- Merged from gemm_wholegrid_adapter_kernel.cpp ----

using namespace AscendC;
namespace {
constexpr uint32_t kAdapterElements = 1024;
}

// Zero the added K-padding tail of packed A and B so the Cube accumulate over
// the padded K contributes nothing beyond the logical problem.
__aicore__ __inline__ void WholeGridZeroPadding(
    const GemmArch22Tiling& tiling, const GlobalTensor<float>& packA, const GlobalTensor<float>& packB,
    const LocalTensor<float>& zero, uint32_t block, uint32_t stride)
{
    const uint32_t kPadding = tiling.computeK - tiling.k;
    // A's physical orientation depends on transA: NN stores [K][M], while
    // transpose stores [M][K]. Zero the padded reduction tail in either form.
    if (tiling.transA == 0) {
        for (uint32_t row = tiling.k + block; row < static_cast<uint32_t>(tiling.computeK); row += stride) {
            for (uint32_t base = 0; base < static_cast<uint32_t>(tiling.m); base += 128) {
                DataCopy(
                    packA[static_cast<uint64_t>(row) * tiling.m + base], zero,
                    min(128U, static_cast<uint32_t>(tiling.m) - base));
            }
        }
    } else {
        for (uint32_t row = block; row < static_cast<uint32_t>(tiling.m); row += stride) {
            DataCopy(packA[static_cast<uint64_t>(row) * tiling.computeK + tiling.k], zero, kPadding);
        }
    }
    if (tiling.transB == 0) {
        // B/N is raw [N][Kcompute].
        for (uint32_t row = block; row < static_cast<uint32_t>(tiling.n); row += stride) {
            DataCopy(packB[static_cast<uint64_t>(row) * tiling.computeK + tiling.k], zero, kPadding);
        }
    } else {
        // B/T is raw [Kcompute][N]; zero the added complete rows.
        const uint32_t chunks = (static_cast<uint32_t>(tiling.n) + 63) / 64;
        const uint64_t tasks = static_cast<uint64_t>(kPadding) * chunks;
        for (uint64_t task = block; task < tasks; task += stride) {
            const uint32_t padRow = static_cast<uint32_t>(task / chunks);
            const uint32_t chunk = static_cast<uint32_t>(task % chunks);
            const uint32_t base = chunk * 64;
            const uint32_t count = min(64U, static_cast<uint32_t>(tiling.n) - base);
            DataCopy(packB[static_cast<uint64_t>(tiling.k + padRow) * tiling.n + base], zero, count);
        }
    }
}

// Strip the A operand: remove leading-dimension gaps while preserving the raw
// physical orientation. The depth-2 bound queue overlaps each GM->UB read with
// the previous UB->packed-GM write.
__aicore__ __inline__ void WholeGridStripOperandA(
    const GemmArch22Tiling& tiling, const GlobalTensor<float>& a, const GlobalTensor<float>& packA)
{
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue;
    pipe.InitBuffer(copyQueue, 2, kAdapterElements * sizeof(float));
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    const uint32_t aRows = tiling.transA == 0 ? tiling.k : tiling.m;
    const uint32_t aWidth = tiling.transA == 0 ? tiling.m : tiling.k;
    const uint32_t aPackedWidth = tiling.transA == 0 ? tiling.m : tiling.computeK;
    const uint32_t aChunks = (aWidth + kAdapterElements - 1) / kAdapterElements;
    const uint64_t aTasks = static_cast<uint64_t>(aRows) * aChunks;
    bool pending = false;
    uint64_t pendingDestination = 0;
    uint32_t pendingCount = 0;
    for (uint64_t task = block; task < aTasks; task += stride) {
        const uint32_t rowIndex = static_cast<uint32_t>(task / aChunks);
        const uint32_t chunk = static_cast<uint32_t>(task % aChunks);
        const uint32_t base = chunk * kAdapterElements;
        const uint32_t count = min(kAdapterElements, aWidth - base);
        const uint64_t source = static_cast<uint64_t>(rowIndex) * tiling.lda + base;
        const uint64_t destination = static_cast<uint64_t>(rowIndex) * aPackedWidth + base;
        LocalTensor<float> current = copyQueue.AllocTensor<float>();
        DataCopy(current, a[source], count);
        if (pending) {
            LocalTensor<float> previous = copyQueue.DeQue<float>();
            DataCopy(packA[pendingDestination], previous, pendingCount);
            copyQueue.FreeTensor(previous);
        }
        copyQueue.EnQue(current);
        pending = true;
        pendingDestination = destination;
        pendingCount = count;
    }
    if (pending) {
        LocalTensor<float> previous = copyQueue.DeQue<float>();
        DataCopy(packA[pendingDestination], previous, pendingCount);
        copyQueue.FreeTensor(previous);
    }
}

// Strip the B operand: remove leading-dimension gaps while preserving the raw
// physical orientation. Independent queue so the depth-2 pipeline is local.
__aicore__ __inline__ void WholeGridStripOperandB(
    const GemmArch22Tiling& tiling, const GlobalTensor<float>& b, const GlobalTensor<float>& packB)
{
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue;
    pipe.InitBuffer(copyQueue, 2, kAdapterElements * sizeof(float));
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();
    const uint32_t bRows = tiling.transB == 0 ? tiling.n : tiling.k;
    const uint32_t bWidth = tiling.transB == 0 ? tiling.k : tiling.n;
    const uint32_t bPackedWidth = tiling.transB == 0 ? tiling.computeK : tiling.n;
    const uint32_t bChunks = (bWidth + kAdapterElements - 1) / kAdapterElements;
    const uint64_t bTasks = static_cast<uint64_t>(bRows) * bChunks;
    bool pending = false;
    uint64_t pendingDestination = 0;
    uint32_t pendingCount = 0;
    for (uint64_t task = block; task < bTasks; task += stride) {
        const uint32_t rowIndex = static_cast<uint32_t>(task / bChunks);
        const uint32_t chunk = static_cast<uint32_t>(task % bChunks);
        const uint32_t base = chunk * kAdapterElements;
        const uint32_t count = min(kAdapterElements, bWidth - base);
        const uint64_t source = static_cast<uint64_t>(rowIndex) * tiling.ldb + base;
        const uint64_t destination = static_cast<uint64_t>(rowIndex) * bPackedWidth + base;
        LocalTensor<float> current = copyQueue.AllocTensor<float>();
        DataCopy(current, b[source], count);
        if (pending) {
            LocalTensor<float> previous = copyQueue.DeQue<float>();
            DataCopy(packB[pendingDestination], previous, pendingCount);
            copyQueue.FreeTensor(previous);
        }
        copyQueue.EnQue(current);
        pending = true;
        pendingDestination = destination;
        pendingCount = count;
    }
    if (pending) {
        LocalTensor<float> previous = copyQueue.DeQue<float>();
        DataCopy(packB[pendingDestination], previous, pendingCount);
        copyQueue.FreeTensor(previous);
    }
}

// Strip public leading-dimension gaps while preserving raw physical
// orientations. The depth-2 bound queue overlaps each GM->UB read with the
// previous UB->packed-GM write.
extern "C" __global__ __aicore__ __vector__ void sgemm_arch22_wholegrid_pack_kernel(
    GM_ADDR aGm, GM_ADDR bGm, GM_ADDR workspaceGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> a, b, packA, packB;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bGm));
    auto* workspace = reinterpret_cast<__gm__ float*>(workspaceGm);
    packA.SetGlobalBuffer(workspace + tiling.packAOffset);
    packB.SetGlobalBuffer(workspace + tiling.packBOffset);

    TPipe pipe;
    TBuf<TPosition::VECCALC> zeroBuffer;
    pipe.InitBuffer(zeroBuffer, 128 * sizeof(float));
    const uint32_t block = GetBlockIdx();
    const uint32_t stride = GetBlockNum();

    const uint32_t kPadding = tiling.computeK - tiling.k;
    if (kPadding != 0) {
        LocalTensor<float> zero = zeroBuffer.Get<float>();
        Duplicate(zero, 0.0f, 128);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WholeGridZeroPadding(tiling, packA, packB, zero, block, stride);
    }

    WholeGridStripOperandA(tiling, a, packA);
    WholeGridStripOperandB(tiling, b, packB);
}

__aicore__ __inline__ void WholeGridEpilogueRead(
    const GemmArch22Tiling& tiling, const GlobalTensor<float>& product, const GlobalTensor<float>& c, uint64_t task,
    uint32_t segments, TQue<QuePosition::VECIN, 2>& productQueue, TQue<QuePosition::VECIN, 2>& cQueue)
{
    if (segments == 0)
        return;
    const uint32_t j = static_cast<uint32_t>(task / segments);
    const uint32_t segment = static_cast<uint32_t>(task % segments);
    const uint32_t mBase = segment * kAdapterElements;
    const uint32_t count = min(kAdapterElements, static_cast<uint32_t>(tiling.m) - mBase);
    LocalTensor<float> productLocal = productQueue.AllocTensor<float>();
    DataCopy(productLocal, product[static_cast<uint64_t>(j) * tiling.m + mBase], count);
    productQueue.EnQue(productLocal);
    if (tiling.beta != 0.0f) {
        LocalTensor<float> cLocal = cQueue.AllocTensor<float>();
        DataCopy(cLocal, c[static_cast<uint64_t>(j) * tiling.ldc + mBase], count);
        cQueue.EnQue(cLocal);
    }
}

__aicore__ __inline__ void WholeGridEpilogueCompute(
    const GemmArch22Tiling& tiling, uint64_t task, uint32_t segments, TQue<QuePosition::VECIN, 2>& productQueue,
    TQue<QuePosition::VECIN, 2>& cQueue, TQue<QuePosition::VECOUT, 2>& outQueue)
{
    if (segments == 0)
        return;
    const uint32_t segment = static_cast<uint32_t>(task % segments);
    const uint32_t mBase = segment * kAdapterElements;
    const uint32_t count = min(kAdapterElements, static_cast<uint32_t>(tiling.m) - mBase);
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
    const GemmArch22Tiling& tiling, const GlobalTensor<float>& c, uint64_t task, uint32_t segments,
    TQue<QuePosition::VECOUT, 2>& outQueue)
{
    if (segments == 0)
        return;
    const uint32_t j = static_cast<uint32_t>(task / segments);
    const uint32_t segment = static_cast<uint32_t>(task % segments);
    const uint32_t mBase = segment * kAdapterElements;
    const uint32_t count = min(kAdapterElements, static_cast<uint32_t>(tiling.m) - mBase);
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopy(c[static_cast<uint64_t>(j) * tiling.ldc + mBase], outLocal, count);
    outQueue.FreeTensor(outLocal);
}

// Three-stage product/C read -> alpha/beta vector work -> C write pipeline.
extern "C" __global__ __aicore__ __vector__ void sgemm_arch22_wholegrid_epilogue_kernel(
    GM_ADDR workspaceGm, GM_ADDR cGm, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> product, c;
    product.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm) + tiling.productOffset);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> productQueue;
    TQue<QuePosition::VECIN, 2> cQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(productQueue, 2, kAdapterElements * sizeof(float));
    pipe.InitBuffer(cQueue, 2, kAdapterElements * sizeof(float));
    pipe.InitBuffer(outQueue, 2, kAdapterElements * sizeof(float));

    const uint32_t segments = (static_cast<uint32_t>(tiling.m) + kAdapterElements - 1) / kAdapterElements;
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

void sgemm_arch22_wholegrid_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const GemmArch22Tiling& tiling)
{
    sgemm_arch22_wholegrid_pack_kernel<<<blocks, nullptr, stream>>>(a, b, workspace, tiling);
}

void sgemm_arch22_wholegrid_epilogue_do(
    uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmArch22Tiling& tiling)
{
    sgemm_arch22_wholegrid_epilogue_kernel<<<blocks, nullptr, stream>>>(workspace, c, tiling);
}

// ---- Merged from gemm_wholegrid_nosplit_kernel.cpp ----

// Full-K square output ownership. Public transpose/layout handling is kept in
// the input adapter; each fixed specialization consumes one compact raw
// orientation and produces the same compact [N][M] product.

using namespace AscendC;
using namespace matmul;
namespace {
constexpr int32_t kRows = 256;
constexpr int32_t kCols = 256;
constexpr int32_t kMaxK = 8192;
using Fp32NoTrans = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using Fp32Trans = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
// NN/NT/TT use baseK=32. TN keeps baseK=64 because aligned TN cases regress with baseK=32.
// baseM x baseN is capped at 64x256 so the fp32 accumulator occupies 64KB of L0C.
// A 128x256 base tile needs the full 128KB and leaves no headroom, which corrupts a
// subset of output tiles on the 2048-and-larger square shapes.
constexpr MatmulShapeParams kShape{kRows, kCols, kMaxK, 64, 256, 32};
constexpr MatmulShapeParams kTnShape{kRows, kCols, kMaxK, 64, 256, 64};
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

template <bool LeftTrans, bool RightTrans, class MatmulT>
__aicore__ inline void RunSquare(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR cGm, GemmArch22Tiling tiling)
{
    const int32_t nTiles = (tiling.n + kRows - 1) / kRows;
    const int32_t nTile = static_cast<int32_t>(GetBlockIdx()) % nTiles;
    const int32_t mTile = static_cast<int32_t>(GetBlockIdx()) / nTiles;
    const int32_t nBase = nTile * kRows;
    const int32_t mBase = mTile * kCols;
    const int32_t rows = min(kRows, tiling.n - nBase);
    const int32_t cols = min(kCols, tiling.m - mBase);
    const int32_t computeK = tiling.computeK;

    GlobalTensor<float> left, right, output;
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
        reinterpret_cast<__gm__ float*>(cGm) + static_cast<uint64_t>(nBase) * tiling.m + mBase,
        static_cast<uint64_t>(rows) * tiling.m);

    TPipe pipe;
    MatmulT matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    matmul.DisableBias();
    matmul.SetOrgShape(LeftTrans ? tiling.n : rows, tiling.m, computeK, computeK, tiling.m);
    matmul.SetSingleShape(rows, cols, computeK);
    matmul.SetTensorA(left, LeftTrans);
    matmul.SetTensorB(right, RightTrans);
    matmul.IterateAll(output, 0);
    matmul.End();
}
} // namespace

extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_wholegrid_square_nn_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    RunSquare<false, false, NnMatmul>(a, b, c, tiling);
}
extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_wholegrid_square_nt_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    RunSquare<true, false, NtMatmul>(a, b, c, tiling);
}
extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_wholegrid_square_tn_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    RunSquare<false, true, TnMatmul>(a, b, c, tiling);
}
extern "C" __global__ __aicore__ __cube__ void sgemm_arch22_wholegrid_square_tt_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // FP32 Cube defaults to HF32 on some CANN/SOC combinations, which truncates the
    // mantissa and drifts against the FP32 CPU golden over long K reductions.
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    RunSquare<true, true, TtMatmul>(a, b, c, tiling);
}

void sgemm_arch22_wholegrid_nosplit_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmArch22Tiling& tiling)
{
    if (tiling.transB == 0) {
        if (tiling.transA == 0) {
            sgemm_arch22_wholegrid_square_nn_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
        } else {
            sgemm_arch22_wholegrid_square_tn_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
        }
    } else if (tiling.transA == 0) {
        sgemm_arch22_wholegrid_square_nt_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
    } else {
        sgemm_arch22_wholegrid_square_tt_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
    }
}
