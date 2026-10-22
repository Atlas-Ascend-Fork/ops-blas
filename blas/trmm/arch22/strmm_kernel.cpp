/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file strmm_kernel.cpp
 * \brief Atlas A2/A3 STRMM: triangular pack, Cube GEMM, and output scaling.
 */

#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "strmm_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t PACK_UPPER = 121;
constexpr uint32_t PACK_TRANS_N = 111;
constexpr uint32_t PACK_DIAG_UNIT = 132;
constexpr uint32_t PACK_CHUNK = 4096;
constexpr uint32_t SCALE_CHUNK = 2048;
constexpr uint32_t SCALE_BATCH_UB_ELEMS = 4096;
constexpr uint32_t DIRECT_LIMIT = 32;

// Small-K, narrow transposed products benefit from a smaller M tile when it
// reduces the busiest core's output rows. The compile-time K bound is 128
// instead of the general path's 8192-element limit.
constexpr uint32_t NARROW_TILE_M = 96;
constexpr uint32_t NARROW_SINGLE_K = 128;
__aicore__ static constexpr MatmulShapeParams NarrowTransposeShape()
{
    return MatmulShapeParams{NARROW_TILE_M, CBLAS3_TILE_N, NARROW_SINGLE_K,
                             NARROW_TILE_M, CBLAS3_TILE_N, CBLAS3_BASE_K};
}
constexpr MatmulConfig NARROW_CONFIG =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(NarrowTransposeShape(), cblas3::BiasParams());
constexpr MatmulApiStaticTiling NARROW_TILING =
    GetMatmulApiTiling<cblas3::MmPlain, cblas3::MmTrans, cblas3::MmPlain, cblas3::MmPlain>(NARROW_CONFIG);
using NarrowTransposeMatmul =
    MatmulImpl<cblas3::MmPlain, cblas3::MmTrans, cblas3::MmPlain, cblas3::MmPlain, NARROW_TILING>;

__aicore__ inline bool UseNarrowTransposeTiles(const CBlas3GemmTilingData& t)
{
    const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
    if (cores == 0U || t.n <= CBLAS3_BASE_K || t.n > CBLAS3_TILE_N || t.k > NARROW_SINGLE_K) {
        return false;
    }
    // Account for the last partial tile: just past a complete wave, its one
    // extra row must not be mistaken for an entire extra tile of work.
    const uint32_t wideFullTiles = t.m / CBLAS3_TILE_M;
    const uint32_t narrowFullTiles = t.m / NARROW_TILE_M;
    const uint32_t wideMaxRows = (wideFullTiles / cores) * CBLAS3_TILE_M +
        (wideFullTiles % cores != 0U ? CBLAS3_TILE_M : t.m % CBLAS3_TILE_M);
    const uint32_t narrowMaxRows = (narrowFullTiles / cores) * NARROW_TILE_M +
        (narrowFullTiles % cores != 0U ? NARROW_TILE_M : t.m % NARROW_TILE_M);
    return narrowMaxRows < wideMaxRows;
}

__aicore__ inline void DirectLeftNoTransVector(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
                                                LocalTensor<float> cLocal, const StrmmDirectTilingData& t,
                                                uint32_t outputCol)
{
    const uint32_t offset = outputCol * t.m;
    Duplicate(cLocal[offset], 0.0F, t.m);
    for (uint32_t k = 0; k < t.m; ++k) {
        const float bValue = bLocal.GetValue(offset + k);
        if (bValue != 0.0F) {
            Axpy(cLocal[offset], aLocal[k * t.m], t.alpha * bValue, static_cast<int32_t>(t.m));
        }
    }
    PipeBarrier<PIPE_V>();
}

// An 8-row output tile avoids every column core fetching all of
// A/B. All GM output blocks are 32-byte aligned and owned by exactly one core.
__aicore__ inline void StageSmallLowerInputs(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    LocalTensor<float> aLocal, LocalTensor<float> bLocal, const StrmmDirectTilingData& t, uint32_t rowBase,
    uint32_t colBase, uint32_t cols)
{
    DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
    if (rowBase != 0U) {
        DataCopyExtParams full{static_cast<uint16_t>(rowBase), 32U, (t.lda - 8U) * 4U, 0U, 0U};
        DataCopyPad(aLocal, aGm[rowBase], full, noPad);
    }
    for (uint32_t d = 0; d < 8U; ++d) {
        const uint32_t begin = d + static_cast<uint32_t>(t.diag == PACK_DIAG_UNIT);
        if (begin < 8U) {
            const uint32_t k = rowBase + d;
            DataCopyExtParams edge{1U, (8U - begin) * 4U, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{true, static_cast<uint8_t>(begin), 0U, 0.0F};
            DataCopyPad(aLocal[k * 8U], aGm[static_cast<uint64_t>(k) * t.lda + rowBase + begin], edge, pad);
        }
    }
    if (t.ldb == t.m) {
        DataCopyExtParams copy{1U, t.m * cols * 4U, 0U, 0U, 0U};
        DataCopyPad(bLocal, bGm[static_cast<uint64_t>(colBase) * t.ldb], copy, noPad);
    } else {
        DataCopyExtParams copy{static_cast<uint16_t>(cols), t.m * 4U, (t.ldb - t.m) * 4U, 0U, 0U};
        DataCopyPad(bLocal, bGm[static_cast<uint64_t>(colBase) * t.ldb], copy, noPad);
    }
    if (t.diag == PACK_DIAG_UNIT) {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t d = 0; d < 8U; ++d) {
            aLocal.SetValue((rowBase + d) * 8U + d, 1.0F);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
}

__aicore__ inline bool SmallLowerOrdinaryInputs(LocalTensor<float> input, LocalTensor<float> absLocal,
    LocalTensor<uint8_t> mask, uint32_t inputCount)
{
    // Check every input (including NaN, which a max reduction may ignore).
    // This reassociated sum also needs a conservative overflow guard:
    // m <= 32, |alpha| < 2 and |A|,|B| < 1e18 bound every partial sum
    // below 6.4e37. Larger values keep Netlib's scalar operation order.
    Abs(absLocal, input, inputCount);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, absLocal, 1.0e18F, CMPMODE::LT, inputCount);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    bool ordinary = true;
    auto words = mask.ReinterpretCast<uint64_t>();
    for (uint32_t i = 0; i < inputCount / 64U; ++i) {
        ordinary = ordinary && words.GetValue(i) == ~0ULL;
    }
    return ordinary;
}

__aicore__ inline void ComputeSmallLowerVector(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
    LocalTensor<float> broadcast, LocalTensor<float> product, LocalTensor<float> output,
    const StrmmDirectTilingData& t, uint32_t cols)
{
    Brcb(broadcast, bLocal, static_cast<uint8_t>(t.m * cols / 8U), {1, 8});
    PipeBarrier<PIPE_V>();
    Muls(broadcast, broadcast, t.alpha, t.m * cols * 8U);
    PipeBarrier<PIPE_V>();
    for (uint32_t col = 0; col < cols; ++col) {
        const uint32_t offset = col * t.m * 8U;
        Mul(product[offset], aLocal, broadcast[offset], t.m * 8U);
    }
    PipeBarrier<PIPE_V>();
    const uint8_t stride = static_cast<uint8_t>(t.m);
    const uint8_t repeats = static_cast<uint8_t>(cols);
    // Reduce all output columns together: a repeat selects a column,
    // and each group of eight lanes contains the eight output rows.
    for (uint32_t count = t.m * 4U; count > 8U; count /= 2U) {
        for (uint32_t offset = 0; offset < count; offset += 64U) {
            const uint64_t lanes = count - offset < 64U ? count - offset : 64U;
            Add(product[offset], product[offset], product[count + offset], lanes, repeats,
                {1, 1, 1, stride, stride, stride});
        }
        PipeBarrier<PIPE_V>();
    }
    Add(output, product, product[8U], 8ULL, repeats, {1, 1, 1, 1, stride, stride});
}

__aicore__ inline void ComputeSmallLowerScalar(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
    LocalTensor<float> output, const StrmmDirectTilingData& t, uint32_t rowBase, uint32_t cols)
{
    for (uint32_t col = 0; col < cols; ++col) {
        for (uint32_t r = 0; r < 8U; ++r) {
            const uint32_t row = rowBase + r;
            const float self = bLocal.GetValue(col * t.m + row);
            float value = self;
            if (self != 0.0F) {
                const float scaled = t.alpha * self;
                value = t.diag == PACK_DIAG_UNIT ? scaled : scaled * aLocal.GetValue(row * 8U + r);
            }
            for (uint32_t k = row; k > 0U;) {
                --k;
                const float bValue = bLocal.GetValue(col * t.m + k);
                if (bValue != 0.0F) {
                    value = value + (t.alpha * bValue) * aLocal.GetValue(k * 8U + r);
                }
            }
            output.SetValue(col * 8U + r, value);
        }
    }
}

__aicore__ inline void ProcessSmallLowerTile(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    GlobalTensor<float> cGm, LocalTensor<float> input, LocalTensor<float> absLocal, LocalTensor<uint8_t> mask,
    LocalTensor<float> broadcast, LocalTensor<float> product, LocalTensor<float> output,
    const StrmmDirectTilingData& t, uint32_t tile, uint32_t tiles, uint32_t cores, uint32_t rowTiles)
{
    if (rowTiles == 0U) {
        return;
    }
    auto aLocal = input;
    auto bLocal = input[t.m * 8U];
    constexpr uint32_t colWidth = 16U;
    const uint32_t inputCount = t.m * (8U + colWidth);
    const uint32_t rowBase = (tile % rowTiles) * 8U;
    const uint32_t colBase = (tile / rowTiles) * colWidth;
    const uint32_t cols = t.n - colBase < colWidth ? t.n - colBase : colWidth;
    Duplicate(input, 0.0F, inputCount);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    StageSmallLowerInputs(aGm, bGm, aLocal, bLocal, t, rowBase, colBase, cols);
    const bool ordinary = SmallLowerOrdinaryInputs(input, absLocal, mask, inputCount);
    if (ordinary) {
        ComputeSmallLowerVector(aLocal, bLocal, broadcast, product, output, t, cols);
    } else {
        ComputeSmallLowerScalar(aLocal, bLocal, output, t, rowBase, cols);
    }
    if (ordinary) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    }
    DataCopyExtParams copyOut{static_cast<uint16_t>(cols), 32U, 0U, (t.ldc - 8U) * 4U, 0U};
    DataCopyPad(cGm[static_cast<uint64_t>(colBase) * t.ldc + rowBase], output, copyOut);
    if (tile + cores < tiles) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

__aicore__ inline void DirectSmallLower(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
                                       GlobalTensor<float> cGm, const StrmmDirectTilingData& t)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> maskBuffer, inputBuffer, absBuffer, broadcastBuffer, productBuffer, outputBuffer;
    pipe.InitBuffer(maskBuffer, 256U);
    pipe.InitBuffer(inputBuffer, DIRECT_LIMIT * 24U * sizeof(float));
    pipe.InitBuffer(absBuffer, DIRECT_LIMIT * 24U * sizeof(float));
    pipe.InitBuffer(broadcastBuffer, DIRECT_LIMIT * 128U * sizeof(float));
    pipe.InitBuffer(productBuffer, DIRECT_LIMIT * 128U * sizeof(float));
    pipe.InitBuffer(outputBuffer, 128U * sizeof(float));
    auto input = inputBuffer.Get<float>();
    auto absLocal = absBuffer.Get<float>();
    auto mask = maskBuffer.Get<uint8_t>();
    auto broadcast = broadcastBuffer.Get<float>();
    auto product = productBuffer.Get<float>();
    auto output = outputBuffer.Get<float>();
    const uint32_t rowTiles = t.m / 8U;
    constexpr uint32_t colWidth = 16U;
    const uint32_t tiles = rowTiles * ((t.n + colWidth - 1U) / colWidth);
    const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t tile = static_cast<uint32_t>(GetBlockIdx()); tile < tiles; tile += cores) {
        ProcessSmallLowerTile(aGm, bGm, cGm, input, absLocal, mask, broadcast, product, output, t,
            tile, tiles, cores, rowTiles);
    }
}

// Tiny transposed LEFT products are scalar-latency bound. Keep their dot
// products compile-time sized and stage only the selected triangle. Eight-row
// columns are independently owned by cores; shorter columns use one core and
// one contiguous output DMA. The accumulation order is unchanged.
template <uint32_t DIM, bool UPPER, bool UNIT>
__aicore__ inline void StageTinyTransposeA(GlobalTensor<float> aGm, LocalTensor<float> aLocal,
    const StrmmDirectTilingData& t)
{
#pragma unroll
    for (uint32_t col = 0; col < DIM; ++col) {
        const uint32_t begin = UPPER ? 0U : col + static_cast<uint32_t>(UNIT);
        const uint32_t end = UPPER ? col + static_cast<uint32_t>(!UNIT) : DIM;
        if (begin < end) {
            DataCopyExtParams copy{1U, (end - begin) * 4U, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{true, static_cast<uint8_t>(begin),
                                           static_cast<uint8_t>(8U - end), 0.0F};
            DataCopyPad(aLocal[col * 8U], aGm[static_cast<uint64_t>(col) * t.lda + begin], copy, pad);
        }
    }
}

template <uint32_t DIM, bool UPPER, bool UNIT>
__aicore__ inline void ComputeTinyTransposeColumn(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
    LocalTensor<float> cLocal, const StrmmDirectTilingData& t, uint32_t localOffset)
{
#pragma unroll
    for (uint32_t row = 0; row < DIM; ++row) {
        float sum = 0.0F;
#pragma unroll
        for (uint32_t k = 0; k < DIM; ++k) {
            if (UPPER ? k <= row : k >= row) {
                const float av = UNIT && k == row ? 1.0F : aLocal.GetValue(row * 8U + k);
                sum += av * bLocal.GetValue(localOffset + k);
            }
        }
        cLocal.SetValue(localOffset + row, t.alpha * sum);
    }
}

template <uint32_t DIM, bool UPPER, bool UNIT>
__aicore__ inline void DirectTinyTranspose(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
                                            GlobalTensor<float> cGm, const StrmmDirectTilingData& t)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> aBuffer, bBuffer, cBuffer;
    constexpr uint32_t bcCapacity = DIM == 8U ? 8U : DIM * DIRECT_LIMIT;
    pipe.InitBuffer(aBuffer, DIM * 8U * sizeof(float));
    pipe.InitBuffer(bBuffer, bcCapacity * sizeof(float));
    pipe.InitBuffer(cBuffer, bcCapacity * sizeof(float));
    auto aLocal = aBuffer.Get<float>();
    auto bLocal = bBuffer.Get<float>();
    auto cLocal = cBuffer.Get<float>();
    DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
    StageTinyTransposeA<DIM, UPPER, UNIT>(aGm, aLocal, t);
    if constexpr (DIM < 8U) {
        DataCopyExtParams copy{1U, DIM * t.n * 4U, 0U, 0U, 0U};
        DataCopyPad(bLocal, bGm, copy, noPad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t col = static_cast<uint32_t>(GetBlockIdx()); col < t.n; col += cores) {
        const uint32_t localOffset = DIM == 8U ? 0U : col * DIM;
        if constexpr (DIM == 8U) {
            DataCopyExtParams copy{1U, 32U, 0U, 0U, 0U};
            DataCopyPad(bLocal, bGm[static_cast<uint64_t>(col) * t.ldb], copy, noPad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        ComputeTinyTransposeColumn<DIM, UPPER, UNIT>(aLocal, bLocal, cLocal, t, localOffset);
        if constexpr (DIM == 8U) {
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyExtParams copy{1U, 32U, 0U, 0U, 0U};
            DataCopyPad(cGm[static_cast<uint64_t>(col) * t.ldc], cLocal, copy);
            if (col + cores < t.n) {
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
        }
    }
    if constexpr (DIM < 8U) {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams copy{1U, DIM * t.n * 4U, 0U, 0U, 0U};
        DataCopyPad(cGm, cLocal, copy);
    }
}

template <uint32_t DIM>
__aicore__ inline void DispatchTinyTranspose(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
                                             GlobalTensor<float> cGm, const StrmmDirectTilingData& t)
{
    if (t.uplo == PACK_UPPER) {
        if (t.diag == PACK_DIAG_UNIT) {
            DirectTinyTranspose<DIM, true, true>(aGm, bGm, cGm, t);
        } else {
            DirectTinyTranspose<DIM, true, false>(aGm, bGm, cGm, t);
        }
    } else if (t.diag == PACK_DIAG_UNIT) {
        DirectTinyTranspose<DIM, false, true>(aGm, bGm, cGm, t);
    } else {
        DirectTinyTranspose<DIM, false, false>(aGm, bGm, cGm, t);
    }
}

template <class MM, bool TRANS_LEFT, bool TRANS_RIGHT>
__aicore__ inline void SetStrmmGemmOrgShape(MM& mm, const CBlas3GemmTilingData& t)
{
    if constexpr (TRANS_LEFT) {
        // For transposed A, orgM carries A's physical row stride; orgN carries
        // the plain B stride. SetSingleShape below supplies logical M/N/K.
        mm.SetOrgShape(static_cast<int>(t.ldA), static_cast<int>(t.ldB),
                       static_cast<int>(t.k), static_cast<int>(t.k), static_cast<int>(t.ldc));
    } else if constexpr (TRANS_RIGHT) {
        // For transposed B, orgKa/orgKb carry the two independent strides.
        mm.SetOrgShape(static_cast<int>(t.m), static_cast<int>(t.n),
                       static_cast<int>(t.ldA), static_cast<int>(t.ldB), static_cast<int>(t.ldc));
    } else {
        mm.SetOrgShape(static_cast<int>(t.m), static_cast<int>(t.ldB),
                       static_cast<int>(t.ldA), static_cast<int>(t.ldA), static_cast<int>(t.ldc));
    }
}

template <class MM, bool TRANS_LEFT, bool TRANS_RIGHT>
__aicore__ inline void RunStrmmGemmTile(MM& mm, __gm__ float* leftBase, __gm__ float* rightBase,
    __gm__ float* outBase, const CBlas3GemmTilingData& t, const cblas3::GemmTile& r, uint32_t kBase,
    uint32_t kCount)
{
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> bGlobal;
    GlobalTensor<float> cGlobal;
    if constexpr (TRANS_LEFT) {
        aGlobal.SetGlobalBuffer(
            leftBase + static_cast<uint64_t>(kBase) * t.ldA + r.rowBase,
            static_cast<uint64_t>(kCount - 1U) * t.ldA + r.rowCount);
        bGlobal.SetGlobalBuffer(
            rightBase + static_cast<uint64_t>(kBase) * t.ldB + r.colBase,
            static_cast<uint64_t>(kCount - 1U) * t.ldB + r.colCount);
    } else {
        aGlobal.SetGlobalBuffer(
            leftBase + static_cast<uint64_t>(r.rowBase) * t.ldA + kBase,
            static_cast<uint64_t>(r.rowCount - 1U) * t.ldA + kCount);
        if constexpr (TRANS_RIGHT) {
            bGlobal.SetGlobalBuffer(
                rightBase + static_cast<uint64_t>(r.colBase) * t.ldB + kBase,
                static_cast<uint64_t>(r.colCount - 1U) * t.ldB + kCount);
        } else {
            bGlobal.SetGlobalBuffer(
                rightBase + static_cast<uint64_t>(kBase) * t.ldB + r.colBase,
                static_cast<uint64_t>(kCount - 1U) * t.ldB + r.colCount);
        }
    }
    cGlobal.SetGlobalBuffer(
        outBase + static_cast<uint64_t>(r.rowBase) * t.ldc + r.colBase,
        static_cast<uint64_t>(r.rowCount - 1U) * t.ldc + r.colCount);
    mm.SetSingleShape(static_cast<int>(r.rowCount), static_cast<int>(r.colCount), static_cast<int>(kCount));
    mm.SetTensorA(aGlobal, TRANS_LEFT);
    mm.SetTensorB(bGlobal, TRANS_RIGHT);
    mm.IterateAll(cGlobal, static_cast<uint8_t>(t.enAtomic));
}

template <class MM, bool TRANS_LEFT, uint32_t TILE_M, bool TRANS_RIGHT>
__aicore__ inline void ProcessStrmmGemmTile(MM& mm, __gm__ float* leftBase, __gm__ float* rightBase,
    __gm__ float* outBase, const StrmmGemmTilingData& info, uint32_t tile, uint32_t mBlocks, bool clipK)
{
    const CBlas3GemmTilingData& t = info.gemm;
    // Make the triangular dimension the slow tile dimension, distributing
    // both long and short reductions across cores rather than pinning the
    // longest reduction to one core for every output row.
    const bool columnMajorTiles = clipK && info.triangleOnRight != 0U;
    const uint32_t rowBlock = columnMajorTiles ? tile % mBlocks : tile / t.nBlocks;
    const uint32_t colBlock = columnMajorTiles ? tile / mBlocks : tile % t.nBlocks;
    const uint32_t rowBase = rowBlock * TILE_M;
    const uint32_t colBase = colBlock * CBLAS3_TILE_N;
    const cblas3::GemmTile r{rowBase, colBase,
                            t.m - rowBase < TILE_M ? t.m - rowBase : TILE_M,
                            t.n - colBase < CBLAS3_TILE_N ? t.n - colBase : CBLAS3_TILE_N};
    uint32_t kBase = t.kBase;
    uint32_t kCount = t.kCount;
    if (clipK) {
        const uint32_t begin = info.triangleOnRight != 0U ? r.colBase : r.rowBase;
        const uint32_t count = info.triangleOnRight != 0U ? r.colCount : r.rowCount;
        const bool prefix = info.triangleOnRight != 0U ? info.lowerTriangle == 0U : info.lowerTriangle != 0U;
        kBase = prefix ? 0U : begin;
        kCount = prefix ? begin + count : t.k - begin;
    }
    RunStrmmGemmTile<MM, TRANS_LEFT, TRANS_RIGHT>(mm, leftBase, rightBase, outBase, t, r, kBase, kCount);
}

template <class MM, bool TRANS_LEFT, uint32_t TILE_M = CBLAS3_TILE_M, bool TRANS_RIGHT = !TRANS_LEFT>
__aicore__ inline void StrmmGemmBody(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, const StrmmGemmTilingData& info)
{
    const CBlas3GemmTilingData& t = info.gemm;
    if (t.nBlocks == 0U || t.mBlocks == 0U || GetBlockNum() == 0) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
    __gm__ float* leftBase = reinterpret_cast<__gm__ float*>(left);
    __gm__ float* rightBase = reinterpret_cast<__gm__ float*>(right);
    __gm__ float* outBase = reinterpret_cast<__gm__ float*>(out);

    TPipe pipe;
    MM mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    SetStrmmGemmOrgShape<MM, TRANS_LEFT, TRANS_RIGHT>(mm, t);
    const uint32_t mBlocks = TILE_M == CBLAS3_TILE_M ? t.mBlocks : cblas3::CeilDivU(t.m, TILE_M);
    const uint32_t totalTiles = mBlocks * t.nBlocks;
    // Chunked reductions retain the dense path: an empty first chunk would
    // otherwise leave an uninitialized destination for later atomic chunks.
    const bool clipK = t.kBase == 0U && t.kCount == t.k &&
                      (info.triangleOnRight != 0U ? t.nBlocks : mBlocks) > 1U;
    for (uint32_t tile = coreIdx; tile < totalTiles; tile += cores) {
        ProcessStrmmGemmTile<MM, TRANS_LEFT, TILE_M, TRANS_RIGHT>(mm, leftBase, rightBase, outBase, info, tile, mBlocks, clipK);
    }
    mm.End();
}

__aicore__ inline float DirectLeftNoTransScalar(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
    const StrmmDirectTilingData& tiling, uint32_t dim, uint32_t row, uint32_t col)
{
    float value;
    const float self = bLocal.GetValue(col * tiling.m + row);
    value = self;
    if (self != 0.0F) {
        const float scaledB = tiling.alpha * self;
        value = (tiling.diag == PACK_DIAG_UNIT)
                    ? scaledB : scaledB * aLocal.GetValue(row * dim + row);
    }
    if (tiling.uplo == PACK_UPPER) {
        for (uint32_t k = row + 1U; k < tiling.m; ++k) {
            const float bValue = bLocal.GetValue(col * tiling.m + k);
            if (bValue != 0.0F) {
                value = value + (tiling.alpha * bValue) * aLocal.GetValue(k * dim + row);
            }
        }
    } else {
        for (uint32_t k = row; k > 0U;) {
            --k;
            const float bValue = bLocal.GetValue(col * tiling.m + k);
            if (bValue != 0.0F) {
                value = value + (tiling.alpha * bValue) * aLocal.GetValue(k * dim + row);
            }
        }
    }
    return value;
}

__aicore__ inline float DirectRightNoTransScalar(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
    const StrmmDirectTilingData& tiling, uint32_t dim, uint32_t row, uint32_t col)
{
    float value;
    float diagonalScale = tiling.alpha;
    if (tiling.diag != PACK_DIAG_UNIT) {
        diagonalScale = diagonalScale * aLocal.GetValue(col * dim + col);
    }
    value = diagonalScale * bLocal.GetValue(col * tiling.m + row);
    if (tiling.uplo == PACK_UPPER) {
        for (uint32_t k = 0; k < col; ++k) {
            const float aValue = aLocal.GetValue(col * dim + k);
            if (aValue != 0.0F) {
                value = value + (tiling.alpha * aValue) * bLocal.GetValue(k * tiling.m + row);
            }
        }
    } else {
        for (uint32_t k = col + 1U; k < tiling.n; ++k) {
            const float aValue = aLocal.GetValue(col * dim + k);
            if (aValue != 0.0F) {
                value = value + (tiling.alpha * aValue) * bLocal.GetValue(k * tiling.m + row);
            }
        }
    }
    return value;
}

__aicore__ inline float DirectTransposeScalar(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
    const StrmmDirectTilingData& tiling, uint32_t dim, uint32_t row, uint32_t col)
{
    float value;
    float sum = 0.0F;
    for (uint32_t k = 0; k < dim; ++k) {
        const uint32_t aRow = (tiling.side == 141U) ? row : k;
        const uint32_t aCol = (tiling.side == 141U) ? k : col;
        const uint32_t srcRow = aCol;
        const uint32_t srcCol = aRow;
        const bool valid = (tiling.uplo == PACK_UPPER) ? (srcRow <= srcCol) : (srcRow >= srcCol);
        if (!valid) {
            continue;
        }
        const float av = (aRow == aCol && tiling.diag == PACK_DIAG_UNIT)
                             ? 1.0F : aLocal.GetValue(srcCol * dim + srcRow);
        const float bv = (tiling.side == 141U)
                             ? bLocal.GetValue(col * tiling.m + k)
                             : bLocal.GetValue(k * tiling.m + row);
        sum += (tiling.side == 141U) ? av * bv : bv * av;
    }
    value = tiling.alpha * sum;
    return value;
}

__aicore__ inline void ComputeDirectScalarColumn(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
    LocalTensor<float> cLocal, const StrmmDirectTilingData& tiling, uint32_t dim, uint32_t col)
{
    for (uint32_t row = 0; row < tiling.m; ++row) {
        float value;
        if (tiling.trans == PACK_TRANS_N && tiling.side == 141U) {
            value = DirectLeftNoTransScalar(aLocal, bLocal, tiling, dim, row, col);
        } else if (tiling.trans == PACK_TRANS_N) {
            value = DirectRightNoTransScalar(aLocal, bLocal, tiling, dim, row, col);
        } else {
            value = DirectTransposeScalar(aLocal, bLocal, tiling, dim, row, col);
        }
        cLocal.SetValue(col * tiling.m + row, value);
    }
}

__aicore__ inline void StageDirectAlignedInputs(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    LocalTensor<float> aLocal, LocalTensor<float> bLocal, const StrmmDirectTilingData& tiling, uint32_t dim,
    bool bCompact)
{
    DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
    // Each column starts at a 32-byte boundary in UB. Only the selected
    // triangle is transferred; DataCopyPad supplies zero edge padding.
    Duplicate(aLocal, 0.0F, dim * dim);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    for (uint32_t col = 0; col < dim; ++col) {
        const uint32_t begin = (tiling.uplo == PACK_UPPER) ? 0U :
            col + static_cast<uint32_t>(tiling.diag == PACK_DIAG_UNIT);
        const uint32_t end = (tiling.uplo == PACK_UPPER) ?
            col + static_cast<uint32_t>(tiling.diag != PACK_DIAG_UNIT) : dim;
        if (begin >= end) {
            continue;
        }
        const uint32_t alignedBegin = begin & ~7U;
        const uint32_t leftPad = begin - alignedBegin;
        const uint32_t rightPad = (8U - ((leftPad + end - begin) & 7U)) & 7U;
        DataCopyExtParams copy{1U, (end - begin) * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> trianglePad{true, static_cast<uint8_t>(leftPad),
                                                static_cast<uint8_t>(rightPad), 0.0F};
        DataCopyPad(aLocal[col * dim + alignedBegin],
                    aGm[static_cast<uint64_t>(col) * tiling.lda + begin], copy, trianglePad);
    }
    if (bCompact) {
        DataCopyExtParams copy{1U, tiling.m * tiling.n * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(bLocal, bGm[0], copy, pad);
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    if (tiling.diag == PACK_DIAG_UNIT) {
        for (uint32_t col = 0; col < dim; ++col) {
            aLocal.SetValue(col * dim + col, 1.0F);
        }
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
}

__aicore__ inline void StageDirectScalarInputs(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    LocalTensor<float> aLocal, LocalTensor<float> bLocal, const StrmmDirectTilingData& tiling, uint32_t dim,
    bool bCompact)
{
    DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
    if (bCompact) {
        DataCopyExtParams copy{1U, tiling.m * tiling.n * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(bLocal, bGm[0], copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    for (uint32_t col = 0; col < dim; ++col) {
        for (uint32_t row = 0; row < dim; ++row) {
            float value = 0.0F;
            if (row == col && tiling.diag == PACK_DIAG_UNIT) {
                value = 1.0F;
            } else if ((tiling.uplo == PACK_UPPER && row <= col) ||
                       (tiling.uplo != PACK_UPPER && row >= col)) {
                value = aGm.GetValue(static_cast<uint64_t>(col) * tiling.lda + row);
            }
            aLocal.SetValue(col * dim + row, value);
        }
    }
}

__aicore__ inline void StageDirectInputs(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    LocalTensor<float> aLocal, LocalTensor<float> bLocal, const StrmmDirectTilingData& tiling, uint32_t dim,
    bool bCompact)
{
    if ((dim & 7U) == 0U && dim >= 16U) {
        StageDirectAlignedInputs(aGm, bGm, aLocal, bLocal, tiling, dim, bCompact);
    } else {
        StageDirectScalarInputs(aGm, bGm, aLocal, bLocal, tiling, dim, bCompact);
    }
    if (!bCompact) {
        for (uint32_t col = 0; col < tiling.n; ++col) {
            for (uint32_t row = 0; row < tiling.m; ++row) {
                bLocal.SetValue(col * tiling.m + row,
                                bGm.GetValue(static_cast<uint64_t>(col) * tiling.ldb + row));
            }
        }
    }
}

__aicore__ inline bool DirectSpecialA(LocalTensor<float> aLocal, LocalTensor<float> absLocal,
    LocalTensor<float> reduceWork, LocalTensor<float> reduceOut, uint32_t dim, bool bCompact)
{
    bool aSpecial = false;
    if (!bCompact) {
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    const uint32_t aCount = dim * dim;
    Abs(absLocal, aLocal, aCount);
    PipeBarrier<PIPE_V>();
    ReduceMax(reduceOut, absLocal, reduceWork, static_cast<int32_t>(aCount), false);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    aSpecial = !(reduceOut.GetValue(0) < 1.0e20F);
    return aSpecial;
}

__aicore__ inline bool DirectSpecialB(LocalTensor<float> bLocal, LocalTensor<float> absLocal,
    LocalTensor<float> reduceWork, LocalTensor<float> reduceOut, const StrmmDirectTilingData& tiling, uint32_t col)
{
    bool bSpecial = false;
    if (tiling.m == 32U) {
        Abs(absLocal, bLocal[col * tiling.m], tiling.m);
        PipeBarrier<PIPE_V>();
        ReduceMax(reduceOut, absLocal, reduceWork, static_cast<int32_t>(tiling.m), false);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        bSpecial = !(reduceOut.GetValue(0) < 1.0e20F);
    } else {
        for (uint32_t k = 0; k < tiling.m; ++k) {
            const float value = bLocal.GetValue(col * tiling.m + k);
            if (!(value > -1.0e20F && value < 1.0e20F)) {
                bSpecial = true;
                break;
            }
        }
    }
    return bSpecial;
}

__aicore__ inline void ComputeDirectColumns(GlobalTensor<float> cGm, LocalTensor<float> aLocal,
    LocalTensor<float> bLocal, LocalTensor<float> cLocal, LocalTensor<float> absLocal,
    LocalTensor<float> reduceWork, LocalTensor<float> reduceOut, const StrmmDirectTilingData& tiling, uint32_t dim,
    uint32_t cores, bool fastCandidate, bool aSpecial)
{
    for (uint32_t col = static_cast<uint32_t>(GetBlockIdx()); col < tiling.n; col += cores) {
        bool bSpecial = false;
        if (fastCandidate && !aSpecial) {
            bSpecial = DirectSpecialB(bLocal, absLocal, reduceWork, reduceOut, tiling, col);
        }
        if (fastCandidate && !aSpecial && !bSpecial) {
            DirectLeftNoTransVector(aLocal, bLocal, cLocal, tiling, col);
        } else {
            ComputeDirectScalarColumn(aLocal, bLocal, cLocal, tiling, dim, col);
        }
        if (cores > 1U) {
            if (fastCandidate && !aSpecial && !bSpecial) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            }
            DataCopyExtParams copy{1U, tiling.m * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(cGm[static_cast<uint64_t>(col) * tiling.ldc], cLocal[col * tiling.m], copy);
        }
    }
}

__aicore__ inline void StoreDirectSingleCore(GlobalTensor<float> cGm, LocalTensor<float> cLocal,
    const StrmmDirectTilingData& tiling, uint32_t cores)
{
    if (cores == 1U && tiling.ldc == tiling.m) {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams copy{1U, tiling.m * tiling.n * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(cGm[0], cLocal, copy);
    } else if (cores == 1U) {
        for (uint32_t col = 0; col < tiling.n; ++col) {
            for (uint32_t row = 0; row < tiling.m; ++row) {
                cGm.SetValue(static_cast<uint64_t>(col) * tiling.ldc + row,
                             cLocal.GetValue(col * tiling.m + row));
            }
        }
    }
}

__aicore__ inline void ComputeDirectStaged(GlobalTensor<float> cGm, LocalTensor<float> aLocal,
    LocalTensor<float> bLocal, LocalTensor<float> cLocal, LocalTensor<float> absLocal,
    LocalTensor<float> reduceWork, LocalTensor<float> reduceOut, const StrmmDirectTilingData& tiling, uint32_t dim,
    uint32_t cores, bool bCompact)
{
    const bool fastCandidate = cores > 1U && tiling.side == 141U &&
                               tiling.trans == PACK_TRANS_N && (tiling.m == 16U || tiling.m == 32U);
    // Preserve the scalar fallback for extreme finite values too: vector
    // accumulation can otherwise change FP32 overflow/NaN behavior vs Netlib.
    bool aSpecial = false;
    if (fastCandidate) {
        aSpecial = DirectSpecialA(aLocal, absLocal, reduceWork, reduceOut, dim, bCompact);
    }

    ComputeDirectColumns(cGm, aLocal, bLocal, cLocal, absLocal, reduceWork, reduceOut, tiling,
        dim, cores, fastCandidate, aSpecial);
    StoreDirectSingleCore(cGm, cLocal, tiling, cores);
}

__aicore__ inline void RunDirectStaged(GlobalTensor<float> aGm, GlobalTensor<float> bGm, GlobalTensor<float> cGm,
    const StrmmDirectTilingData& tiling, uint32_t dim, uint32_t cores)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> aBuffer;
    TBuf<TPosition::VECCALC> bBuffer;
    TBuf<TPosition::VECCALC> cBuffer;
    TBuf<TPosition::VECCALC> absBuffer;
    TBuf<TPosition::VECCALC> reduceWorkBuffer;
    TBuf<TPosition::VECCALC> reduceOutBuffer;
    pipe.InitBuffer(aBuffer, DIRECT_LIMIT * DIRECT_LIMIT * sizeof(float));
    pipe.InitBuffer(bBuffer, DIRECT_LIMIT * DIRECT_LIMIT * sizeof(float));
    pipe.InitBuffer(cBuffer, DIRECT_LIMIT * DIRECT_LIMIT * sizeof(float));
    pipe.InitBuffer(absBuffer, DIRECT_LIMIT * DIRECT_LIMIT * sizeof(float));
    pipe.InitBuffer(reduceWorkBuffer, 128U);
    pipe.InitBuffer(reduceOutBuffer, 32U);
    LocalTensor<float> aLocal = aBuffer.Get<float>();
    LocalTensor<float> bLocal = bBuffer.Get<float>();
    LocalTensor<float> cLocal = cBuffer.Get<float>();
    LocalTensor<float> absLocal = absBuffer.Get<float>();
    LocalTensor<float> reduceWork = reduceWorkBuffer.Get<float>();
    LocalTensor<float> reduceOut = reduceOutBuffer.Get<float>();

    const bool bCompact = tiling.ldb == tiling.m;
    StageDirectInputs(aGm, bGm, aLocal, bLocal, tiling, dim, bCompact);
    ComputeDirectStaged(cGm, aLocal, bLocal, cLocal, absLocal, reduceWork, reduceOut, tiling, dim, cores, bCompact);
}

__aicore__ inline bool TryDirectTinyTranspose(GM_ADDR b, GM_ADDR c, GlobalTensor<float> aGm,
    GlobalTensor<float> bGm, GlobalTensor<float> cGm, const StrmmDirectTilingData& tiling, uint32_t cores)
{
    if (b != c && tiling.ldc == tiling.m && tiling.side == 141U && tiling.trans != PACK_TRANS_N) {
        if (tiling.m == 8U) {
            DispatchTinyTranspose<8U>(aGm, bGm, cGm, tiling);
            return true;
        }
        if (tiling.m == 4U && tiling.ldb == 4U && cores == 1U) {
            DispatchTinyTranspose<4U>(aGm, bGm, cGm, tiling);
            return true;
        }
    }
    return false;
}

__aicore__ inline float ReferenceLeftNoTrans(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    const StrmmReferenceTilingData& t, uint32_t row, uint32_t col)
{
    float value;
    const float self = bGm.GetValue(static_cast<uint64_t>(col) * t.ldb + row);
    value = self;
    if (self != 0.0F) {
        const float scaledB = t.alpha * self;
        value = (t.diag == PACK_DIAG_UNIT)
                    ? scaledB : scaledB * aGm.GetValue(static_cast<uint64_t>(row) * t.lda + row);
    }
    if (t.uplo == PACK_UPPER) {
        for (uint32_t k = row + 1U; k < t.m; ++k) {
            const float bValue = bGm.GetValue(static_cast<uint64_t>(col) * t.ldb + k);
            if (bValue != 0.0F) {
                const float scaledB = t.alpha * bValue;
                value = value + scaledB * aGm.GetValue(static_cast<uint64_t>(k) * t.lda + row);
            }
        }
    } else {
        for (uint32_t k = row; k > 0U;) {
            --k;
            const float bValue = bGm.GetValue(static_cast<uint64_t>(col) * t.ldb + k);
            if (bValue != 0.0F) {
                const float scaledB = t.alpha * bValue;
                value = value + scaledB * aGm.GetValue(static_cast<uint64_t>(k) * t.lda + row);
            }
        }
    }
    return value;
}

__aicore__ inline float ReferenceRightNoTrans(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    const StrmmReferenceTilingData& t, uint32_t row, uint32_t col)
{
    float value;
    float diagonalScale = t.alpha;
    if (t.diag != PACK_DIAG_UNIT) {
        diagonalScale = diagonalScale * aGm.GetValue(static_cast<uint64_t>(col) * t.lda + col);
    }
    value = diagonalScale * bGm.GetValue(static_cast<uint64_t>(col) * t.ldb + row);
    if (t.uplo == PACK_UPPER) {
        for (uint32_t k = 0; k < col; ++k) {
            const float aValue = aGm.GetValue(static_cast<uint64_t>(col) * t.lda + k);
            if (aValue != 0.0F) {
                const float scaledA = t.alpha * aValue;
                value = value + scaledA *
                                bGm.GetValue(static_cast<uint64_t>(k) * t.ldb + row);
            }
        }
    } else {
        for (uint32_t k = col + 1U; k < t.n; ++k) {
            const float aValue = aGm.GetValue(static_cast<uint64_t>(col) * t.lda + k);
            if (aValue != 0.0F) {
                const float scaledA = t.alpha * aValue;
                value = value + scaledA *
                                bGm.GetValue(static_cast<uint64_t>(k) * t.ldb + row);
            }
        }
    }
    return value;
}

__aicore__ inline float ReferenceLeftTranspose(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    const StrmmReferenceTilingData& t, uint32_t row, uint32_t col)
{
    float value;
    value = bGm.GetValue(static_cast<uint64_t>(col) * t.ldb + row);
    if (t.diag != PACK_DIAG_UNIT) {
        value = value * aGm.GetValue(static_cast<uint64_t>(row) * t.lda + row);
    }
    if (t.uplo == PACK_UPPER) {
        for (uint32_t k = 0; k < row; ++k) {
            value = value + aGm.GetValue(static_cast<uint64_t>(row) * t.lda + k) *
                            bGm.GetValue(static_cast<uint64_t>(col) * t.ldb + k);
        }
    } else {
        for (uint32_t k = row + 1U; k < t.m; ++k) {
            value = value + aGm.GetValue(static_cast<uint64_t>(row) * t.lda + k) *
                            bGm.GetValue(static_cast<uint64_t>(col) * t.ldb + k);
        }
    }
    value = t.alpha * value;
    return value;
}

__aicore__ inline float ReferenceRightTranspose(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    const StrmmReferenceTilingData& t, uint32_t row, uint32_t col)
{
    float value;
    float diagonalScale = t.alpha;
    if (t.diag != PACK_DIAG_UNIT) {
        diagonalScale = diagonalScale * aGm.GetValue(static_cast<uint64_t>(col) * t.lda + col);
    }
    const float source = bGm.GetValue(static_cast<uint64_t>(col) * t.ldb + row);
    value = (diagonalScale == 1.0F) ? source : diagonalScale * source;
    if (t.uplo == PACK_UPPER) {
        for (uint32_t k = col + 1U; k < t.n; ++k) {
            const float aValue = aGm.GetValue(static_cast<uint64_t>(k) * t.lda + col);
            if (aValue != 0.0F) {
                const float scaledA = t.alpha * aValue;
                value = value + scaledA *
                                bGm.GetValue(static_cast<uint64_t>(k) * t.ldb + row);
            }
        }
    } else {
        for (uint32_t k = col; k > 0U;) {
            --k;
            const float aValue = aGm.GetValue(static_cast<uint64_t>(k) * t.lda + col);
            if (aValue != 0.0F) {
                const float scaledA = t.alpha * aValue;
                value = value + scaledA *
                                bGm.GetValue(static_cast<uint64_t>(k) * t.ldb + row);
            }
        }
    }
    return value;
}

__aicore__ inline void PackSingleColumn(GlobalTensor<float> source, GlobalTensor<float> target,
    LocalTensor<float> local, const StrmmPackTilingData& tiling, uint32_t col, uint32_t coreNum, uint64_t outBase)
{
    const bool unit = tiling.diag == PACK_DIAG_UNIT;
    const uint32_t begin = (tiling.uplo == PACK_UPPER) ? 0U : col + static_cast<uint32_t>(unit);
    const uint32_t end = (tiling.uplo == PACK_UPPER) ? col + static_cast<uint32_t>(!unit) : tiling.dim;
    Duplicate(local, 0.0F, tiling.packedLd);
    if (begin < end) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        const uint32_t alignedBegin = begin & ~7U;
        const uint32_t leftPad = begin - alignedBegin;
        const uint32_t rightPad = (8U - ((leftPad + end - begin) & 7U)) & 7U;
        DataCopyExtParams copy{1U, (end - begin) * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> trianglePad{true, static_cast<uint8_t>(leftPad),
                                                static_cast<uint8_t>(rightPad), 0.0F};
        DataCopyPad(local[alignedBegin], source[static_cast<uint64_t>(col) * tiling.lda + begin],
                    copy, trianglePad);
        if (unit) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        } else {
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        }
    } else {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    if (unit) {
        local.SetValue(col, 1.0F);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    }
    DataCopyExtParams output{1U, tiling.packedLd * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(target[outBase], local, output);
    if (col + coreNum < tiling.dim) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

__aicore__ inline void PackValidRun(GlobalTensor<float> source, GlobalTensor<float> target,
    LocalTensor<float> local, uint64_t sourceBase, uint64_t outBase, uint32_t validEnd, uint32_t& block)
{
    DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
    // Both GM and UB transfers cover only valid A elements.
    const uint32_t available = (validEnd & ~7U) - block;
    const uint32_t count = available < PACK_CHUNK ? available : PACK_CHUNK;
    DataCopyExtParams copy{1U, count * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(local, source[sourceBase + block], copy, pad);
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopyPad(target[outBase + block], local, copy);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    block += count;
}

__aicore__ inline void PackZeroRun(GlobalTensor<float> target, LocalTensor<float> zeroLocal,
    const StrmmPackTilingData& tiling, uint64_t outBase, uint32_t validBegin, bool unit, uint32_t diagBlock,
    uint32_t& block)
{
    uint32_t runEnd = (block < validBegin) ? (validBegin & ~7U) : tiling.packedLd;
    if (unit && diagBlock > block && diagBlock < runEnd) {
        runEnd = diagBlock;
    }
    const uint32_t available = runEnd - block;
    const uint32_t count = available < PACK_CHUNK ? available : PACK_CHUNK;
    DataCopyExtParams zeroCopy{1U, count * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(target[outBase + block], zeroLocal, zeroCopy);
    block += count;
}

__aicore__ inline void PackEdgeBlock(GlobalTensor<float> source, GlobalTensor<float> target,
    LocalTensor<float> edgeLocal, uint64_t sourceBase, uint64_t outBase, uint32_t col, uint32_t validBegin,
    uint32_t validEnd, bool unit, uint32_t& block)
{
    // A partial 32-byte block is built in a separate UB buffer.
    // Invalid triangle elements and UNIT diagonal never enter a GM read.
    for (uint32_t i = 0; i < 8U; ++i) {
        const uint32_t row = block + i;
        float value = 0.0F;
        if (unit && row == col) {
            value = 1.0F;
        } else if (row >= validBegin && row < validEnd) {
            value = source.GetValue(sourceBase + row);
        }
        edgeLocal.SetValue(i, value);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopyExtParams edgeCopy{1U, 8U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(target[outBase + block], edgeLocal, edgeCopy);
    // Finish reading edgeLocal before scalar stores reuse it, including the next column.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    block += 8U;
}

__aicore__ inline void PackBlockedColumn(GlobalTensor<float> source, GlobalTensor<float> target,
    LocalTensor<float> local, LocalTensor<float> zeroLocal, LocalTensor<float> edgeLocal,
    const StrmmPackTilingData& tiling, uint32_t col, uint64_t outBase)
{
    const bool unit = tiling.diag == PACK_DIAG_UNIT;
    const uint32_t validBegin = (tiling.uplo == PACK_UPPER) ? 0U : col + static_cast<uint32_t>(unit);
    const uint32_t validEnd = (tiling.uplo == PACK_UPPER) ? col + static_cast<uint32_t>(!unit) : tiling.dim;
    const uint64_t sourceBase = static_cast<uint64_t>(col) * tiling.lda;
    const uint32_t diagBlock = col & ~7U;
    for (uint32_t block = 0U; block < tiling.packedLd;) {
        if (block >= validBegin && block + 8U <= validEnd) {
            PackValidRun(source, target, local, sourceBase, outBase, validEnd, block);
            continue;
        }
        if ((block + 8U <= validBegin || block >= validEnd) && !(unit && block == diagBlock)) {
            PackZeroRun(target, zeroLocal, tiling, outBase, validBegin, unit, diagBlock, block);
            continue;
        }
        PackEdgeBlock(source, target, edgeLocal, sourceBase, outBase, col, validBegin, validEnd, unit, block);
    }
}

__aicore__ inline void PackTransposedColumn(GlobalTensor<float> source, GlobalTensor<float> target,
    const StrmmPackTilingData& tiling, uint32_t col, uint64_t outBase)
{
    for (uint32_t row = 0; row < tiling.packedLd; ++row) {
        float value = 0.0F;
        if (row < tiling.dim) {
            const uint32_t srcRow = (tiling.trans == PACK_TRANS_N) ? row : col;
            const uint32_t srcCol = (tiling.trans == PACK_TRANS_N) ? col : row;
            if (row == col && tiling.diag == PACK_DIAG_UNIT) {
                value = 1.0F;
            } else if ((tiling.uplo == PACK_UPPER && srcRow <= srcCol) ||
                       (tiling.uplo != PACK_UPPER && srcRow >= srcCol)) {
                value = source.GetValue(static_cast<uint64_t>(srcCol) * tiling.lda + srcRow);
            }
        }
        target.SetValue(outBase + row, value);
    }
}

__aicore__ inline void ScaleBatchedColumns(GlobalTensor<float> tempGm, GlobalTensor<float> cGm,
    LocalTensor<float> local, const StrmmScaleTilingData& tiling, uint32_t coreNum)
{
    const uint32_t alignedRows = ((tiling.m + 7U) / 8U) * 8U;
    const uint32_t batchCols = SCALE_BATCH_UB_ELEMS / alignedRows;
    DataCopyPadExtParams<float> batchPad{true, 0U, 0U, 0.0F};
    for (uint32_t col = static_cast<uint32_t>(GetBlockIdx()) * batchCols;
         col < tiling.n; col += coreNum * batchCols) {
        const uint32_t count = (col + batchCols > tiling.n) ? (tiling.n - col) : batchCols;
        DataCopyExtParams input{static_cast<uint16_t>(count),
                                 tiling.m * static_cast<uint32_t>(sizeof(float)),
                                 (tiling.tempLd - tiling.m) * static_cast<uint32_t>(sizeof(float)), 0U, 0U};
        DataCopyPad(local, tempGm[static_cast<uint64_t>(col) * tiling.tempLd], input, batchPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (tiling.alpha != 1.0F) {
            Muls(local, local, tiling.alpha, count * alignedRows);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams output{static_cast<uint16_t>(count),
                                  tiling.m * static_cast<uint32_t>(sizeof(float)), 0U,
                                  (tiling.ldc - tiling.m) * static_cast<uint32_t>(sizeof(float)), 0U};
        DataCopyPad(cGm[static_cast<uint64_t>(col) * tiling.ldc], local, output);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}


__aicore__ inline float ReferenceElement(GlobalTensor<float> aGm, GlobalTensor<float> bGm,
    const StrmmReferenceTilingData& t, uint32_t row, uint32_t col)
{
    float value;
    if (t.trans == PACK_TRANS_N && t.side == 141U) {
        value = ReferenceLeftNoTrans(aGm, bGm, t, row, col);
    } else if (t.trans == PACK_TRANS_N) {
        value = ReferenceRightNoTrans(aGm, bGm, t, row, col);
    } else if (t.side == 141U) {
        value = ReferenceLeftTranspose(aGm, bGm, t, row, col);
    } else {
        value = ReferenceRightTranspose(aGm, bGm, t, row, col);
    }
    return value;
}


}

// One launch for small matrices. Stage B completely before writing C, so B=C is safe.
extern "C" __global__ __aicore__ void strmm_direct_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, StrmmDirectTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> aGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> cGm;
    const uint32_t dim = (tiling.side == 141U) ? tiling.m : tiling.n;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * dim);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint64_t>(tiling.ldb) * tiling.n);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * tiling.n);

    const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
    if (cores == 0U) {
        return;
    }

    if (TryDirectTinyTranspose(b, c, aGm, bGm, cGm, tiling, cores)) {
        return;
    }
    if (cores > 1U && b != c && tiling.ldc == tiling.m && tiling.side == 141U &&
        tiling.uplo != PACK_UPPER && tiling.trans == PACK_TRANS_N &&
        (tiling.m == 16U || tiling.m == 32U) && tiling.alpha == tiling.alpha) {
        DirectSmallLower(aGm, bGm, cGm, tiling);
        return;
    }

    RunDirectStaged(aGm, bGm, cGm, tiling, dim, cores);
}

// Reference-order path for larger alpha. The task's fixed absolute
// tolerance is sensitive to FP32 reduction order after alpha amplifies error.
// Each output element follows Netlib STRMM's scalar operation sequence on NPU.
extern "C" __global__ __aicore__ void strmm_reference_order_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR temp, StrmmReferenceTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> aGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> outGm;
    const uint32_t dim = t.side == 141U ? t.m : t.n;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(t.lda) * dim);
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint64_t>(t.ldb) * t.n);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temp), static_cast<uint64_t>(t.tempLd) * t.n);
    const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
    if (cores == 0U) {
        return;
    }

    for (uint32_t col = static_cast<uint32_t>(GetBlockIdx()); col < t.n; col += cores) {
        for (uint32_t row = 0; row < t.m; ++row) {
            const float value = ReferenceElement(aGm, bGm, t, row, col);
            outGm.SetValue(static_cast<uint64_t>(col) * t.tempLd + row, value);
        }
    }
}

__aicore__ inline void StrmmPackB(GM_ADDR b, GM_ADDR packedB,
    const StrmmPackTilingData& t, LocalTensor<float> local);

// Pack A without fetching the unused triangle or UNIT diagonal.
extern "C" __global__ __aicore__ void strmm_pack_kernel(
    GM_ADDR a, GM_ADDR packed, GM_ADDR b, GM_ADDR packedB, StrmmPackTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> source;
    GlobalTensor<float> target;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * tiling.dim);
    target.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packed),
                           static_cast<uint64_t>(tiling.packedLd) * tiling.dim);

    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> packBuffer;
    TBuf<TPosition::VECCALC> zeroBuffer;
    TBuf<TPosition::VECCALC> edgeBuffer;
    pipe.InitBuffer(packBuffer, PACK_CHUNK * sizeof(float));
    pipe.InitBuffer(zeroBuffer, PACK_CHUNK * sizeof(float));
    pipe.InitBuffer(edgeBuffer, 8U * sizeof(float));
    LocalTensor<float> local = packBuffer.Get<float>();
    LocalTensor<float> zeroLocal = zeroBuffer.Get<float>();
    LocalTensor<float> edgeLocal = edgeBuffer.Get<float>();
    const bool singleCopyColumn = tiling.dim <= PACK_CHUNK && tiling.packedLd <= PACK_CHUNK &&
                                  tiling.trans == PACK_TRANS_N;
    if (!singleCopyColumn) {
        Duplicate(zeroLocal, 0.0F, PACK_CHUNK);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }
    for (uint32_t col = static_cast<uint32_t>(GetBlockIdx()); col < tiling.dim; col += coreNum) {
        const uint64_t outBase = static_cast<uint64_t>(col) * tiling.packedLd;
        if (singleCopyColumn) {
            PackSingleColumn(source, target, local, tiling, col, coreNum, outBase);
            continue;
        }
        if (tiling.trans == PACK_TRANS_N) {
            PackBlockedColumn(source, target, local, zeroLocal, edgeLocal, tiling, col, outBase);
            continue;
        }
        PackTransposedColumn(source, target, tiling, col, outBase);
    }
    if (tiling.bRows != 0U) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        StrmmPackB(b, packedB, tiling, local);
    }
}

// The optional B copy shares the pack launch but uses a separate GM region.
// It handles arbitrary positive leading dimensions using 64-bit addresses.
__aicore__ inline void StrmmPackB(GM_ADDR b, GM_ADDR packedB,
    const StrmmPackTilingData& t, LocalTensor<float> local)
{
    if (t.bRows == 0U || GetBlockNum() == 0) return;
    GlobalTensor<float> src, dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint64_t>(t.ldb) * t.bCols);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packedB), static_cast<uint64_t>(t.packedBld) * t.bCols);
    for (uint32_t col = static_cast<uint32_t>(GetBlockIdx()); col < t.bCols; col += GetBlockNum()) {
        for (uint32_t row = 0U; row < t.packedBld; row += PACK_CHUNK) {
            const uint32_t count = t.packedBld - row < PACK_CHUNK ? t.packedBld - row : PACK_CHUNK;
            const uint32_t valid = row >= t.bRows ? 0U : (t.bRows - row < count ? t.bRows - row : count);
            Duplicate(local, 0.0F, count);
            if (valid != 0U) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                DataCopyExtParams input{1U, valid * 4U, 0U, 0U, 0U};
                DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F};
                DataCopyPad(local, src[static_cast<uint64_t>(col) * t.ldb + row], input, pad);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            DataCopyExtParams output{1U, count * 4U, 0U, 0U, 0U};
            DataCopyPad(dst[static_cast<uint64_t>(col) * t.packedBld + row], local, output);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
}

extern "C" __global__ __aicore__ void strmm_gemm_kernel(
    GM_ADDR left, GM_ADDR right, GM_ADDR temp, StrmmGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetHF32Mode(HF32Mode::DISABLE);
    if (tiling.gemm.transMode == CBLAS3_GEMM_TRANS_RIGHT) {
        if (UseNarrowTransposeTiles(tiling.gemm)) {
            StrmmGemmBody<NarrowTransposeMatmul, false, NARROW_TILE_M>(left, right, temp, tiling);
        } else {
            StrmmGemmBody<cblas3::MatmulTransRight, false>(left, right, temp, tiling);
        }
    } else if (tiling.gemm.transMode == CBLAS3_GEMM_TRANS_LEFT) {
        StrmmGemmBody<cblas3::MatmulTransLeft, true>(left, right, temp, tiling);
    } else {
        StrmmGemmBody<cblas3::MatmulPlainBoth, false, CBLAS3_TILE_M, false>(left, right, temp, tiling);
    }
}

extern "C" __global__ __aicore__ void strmm_scale_kernel(
    GM_ADDR temp, GM_ADDR c, StrmmScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> tempGm;
    GlobalTensor<float> cGm;
    tempGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temp),
                           static_cast<uint64_t>(tiling.tempLd) * tiling.n);
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c),
                        static_cast<uint64_t>(tiling.ldc) * tiling.n);

    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, SCALE_BATCH_UB_ELEMS * sizeof(float));
    LocalTensor<float> local = buffer.Get<float>();
    DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};

    if (tiling.m <= 256U) {
        ScaleBatchedColumns(tempGm, cGm, local, tiling, coreNum);
        return;
    }

    for (uint32_t col = static_cast<uint32_t>(GetBlockIdx()); col < tiling.n; col += coreNum) {
        for (uint32_t row = 0; row < tiling.m; row += SCALE_CHUNK) {
            const uint32_t count = (row + SCALE_CHUNK > tiling.m) ? (tiling.m - row) : SCALE_CHUNK;
            DataCopyExtParams copy{1U, count * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(local, tempGm[static_cast<uint64_t>(col) * tiling.tempLd + row], copy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Muls(local, local, tiling.alpha, count);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(cGm[static_cast<uint64_t>(col) * tiling.ldc + row], local, copy);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
}

void strmm_pack_kernel_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* packed,
                          uint8_t* b, uint8_t* packedB, StrmmPackTilingData tiling)
{
    strmm_pack_kernel<<<blocks, nullptr, stream>>>(a, packed, b, packedB, tiling);
}

void strmm_direct_kernel_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c,
                            StrmmDirectTilingData tiling)
{
    strmm_direct_kernel<<<blocks, nullptr, stream>>>(a, b, c, tiling);
}

void strmm_reference_order_kernel_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* temp,
                                     StrmmReferenceTilingData tiling)
{
    strmm_reference_order_kernel<<<blocks, nullptr, stream>>>(a, b, temp, tiling);
}

void strmm_gemm_kernel_do(uint32_t blocks, void* stream, uint8_t* left, uint8_t* right, uint8_t* temp,
                          StrmmGemmTilingData tiling)
{
    strmm_gemm_kernel<<<blocks, nullptr, stream>>>(left, right, temp, tiling);
}

void strmm_scale_kernel_do(uint32_t blocks, void* stream, uint8_t* temp, uint8_t* c, StrmmScaleTilingData tiling)
{
    strmm_scale_kernel<<<blocks, nullptr, stream>>>(temp, c, tiling);
}

uint32_t strmm_gemm_single_k() { return cblas3::SINGLE_K; }
