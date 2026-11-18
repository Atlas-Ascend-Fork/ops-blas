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
#include "common/helper/kernel_utils.h"
#include "cann_ops_blas_common.h"
#include "csyr2_kernel.h"
#include "csyr2_triangle_copy.h"
#include "csyr2_vector_common.h"
using namespace AscendC;

// Bounded UB storage for unit strides, including arbitrary complex alpha.
// Every A transfer stays inside the selected triangle, including lda padding.
template <bool GeneralAlpha>
struct UnitTiles {
    static constexpr uint32_t Tile = GeneralAlpha ? 1984U : 512U;
    Csyr2Vector::Vectors v;
    TBuf<TPosition::VECCALC> ab;
    TBuf<TPosition::VECCALC> ib;
    TBuf<TPosition::VECCALC> cb;
    TBuf<TPosition::VECCALC> hb;
    TBuf<TPosition::VECCALC> work;
    LocalTensor<float> av;
    LocalTensor<float> coeff;
    LocalTensor<int32_t> index;
    LocalTensor<int32_t> head;
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm)
    {
        v.InitGlobals(xGm, yGm, aGm);
        if constexpr (GeneralAlpha) {
            pipe.InitBuffer(work, 24U * Tile);
        }
        pipe.InitBuffer(hb, 128U);
        pipe.InitBuffer(cb, 64U);
        v.Allocate(pipe, 8U * Tile);
        pipe.InitBuffer(ab, 32U * Tile);
        pipe.InitBuffer(ib, 8U * Tile);
        v.Bind();
        av = ab.Get<float>();
        coeff = cb.Get<float>();
        index = ib.Get<int32_t>();
        head = hb.Get<int32_t>();
        Csyr2Vector::InitRotationIndex(index, 2U * Tile);
        CreateVecIndex(head, int32_t(0), 8U);
        PipeBarrier<PIPE_V>();
        ShiftLeft(head, head, int32_t(2), 8U);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadTile(
        bool upper, uint32_t col, uint32_t row, uint32_t rows, uint32_t columns, uint32_t pitch,
        const Csyr2TilingData& t)
    {
        gm_to_ub_align<ArchType::ASCEND_V220, float>(v.xv, v.x[2ULL * row], 0, 1, 8U * rows, 0, 0, 0, 0);
        gm_to_ub_align<ArchType::ASCEND_V220, float>(v.yv, v.y[2ULL * row], 0, 1, 8U * rows, 0, 0, 0, 0);
        if (upper) {
            for (uint32_t j = 0; j < columns; ++j) {
                const uint32_t count = col + j + 1U - row < rows ? col + j + 1U - row : rows;
                gm_to_ub_align<ArchType::ASCEND_V220, float>(
                    av[pitch * j], v.a[2ULL * (uint64_t(col + j) * t.lda + row)], 0, 1, 8U * count, 0, 0, 0, 0);
            }
        } else {
            Csyr2Triangle::LowerIn(av, v.a, col, row, rows, columns, t.lda);
        }
    }
    __aicore__ inline void GeneralUpdate(
        LocalTensor<float> dst, float xre, float xim, float yre, float yim, const Csyr2TilingData& t)
    {
        const float pr = t.ar * xre - t.ai * xim;
        const float pi = t.ar * xim + t.ai * xre;
        const float qr = t.ar * yre - t.ai * yim;
        const float qi = t.ar * yim + t.ai * yre;
        auto p = work.Get<float>();
        auto q = p[2U * Tile];
        auto tmp = p[4U * Tile];
        Muls<float, false>(p, v.yv, pr, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        Muls<float, false>(tmp, v.yr, pi, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Add<float, false>(p, p, tmp, MASK_PLACEHOLDER, 1U, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Muls<float, false>(q, v.xv, qr, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        Muls<float, false>(tmp, v.xr, qi, MASK_PLACEHOLDER, 1U, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        Add<float, false>(q, q, tmp, MASK_PLACEHOLDER, 1U, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Add<float, false>(p, p, q, MASK_PLACEHOLDER, 1U, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Add<float, false>(dst, dst, p, MASK_PLACEHOLDER, 1U, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void UpdateTile(
        bool upper, uint32_t col, uint32_t row, uint32_t rows, uint32_t columns, uint32_t pitch,
        const Csyr2TilingData& t)
    {
        for (uint32_t j = 0; j < columns; ++j) {
            const uint32_t count = upper && col + j + 1U - row < rows ? col + j + 1U - row : rows;
            auto dst = av[pitch * j];
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(0U, 2U * count);
            const float xre = coeff.GetValue(2U * j);
            const float xim = coeff.GetValue(2U * j + 1U);
            const float yre = coeff.GetValue(8U + 2U * j);
            const float yim = coeff.GetValue(9U + 2U * j);
            if constexpr (GeneralAlpha) {
                GeneralUpdate(dst, xre, xim, yre, yim, t);
            } else {
                Csyr2Vector::Update(dst, v.xv, v.xr, v.yv, v.yr, xre, xim, yre, yim);
            }
            SetMaskNorm();
            ResetMask();
        }
    }
    __aicore__ inline void StoreTile(
        bool upper, uint32_t col, uint32_t row, uint32_t rows, uint32_t columns, uint32_t pitch,
        const Csyr2TilingData& t)
    {
        if (upper) {
            for (uint32_t j = 0; j < columns; ++j) {
                const uint32_t count = col + j + 1U - row < rows ? col + j + 1U - row : rows;
                ub_to_gm_align<ArchType::ASCEND_V220, float>(
                    v.a[2ULL * (uint64_t(col + j) * t.lda + row)], av[pitch * j], 0, 1, 8U * count, 0, 0, 0, 0);
            }
        } else {
            Csyr2Triangle::LowerOut(v.a, av, head.ReinterpretCast<float>(), col, row, rows, columns, t.lda);
        }
    }
    __aicore__ inline void Run(uint32_t first, uint32_t last, const Csyr2TilingData& t)
    {
        const bool upper = t.uplo == ACLBLAS_UPPER;
        for (uint32_t col = first; col < last; col += 4U) {
            const uint32_t columns = t.n - col < 4U ? t.n - col : 4U;
            gm_to_ub_align<ArchType::ASCEND_V220, float>(coeff, v.x[2ULL * col], 0, 1, 8U * columns, 0, 0, 0, 0);
            gm_to_ub_align<ArchType::ASCEND_V220, float>(coeff[8U], v.y[2ULL * col], 0, 1, 8U * columns, 0, 0, 0, 0);
            PipeBarrier<PIPE_ALL>();
            const uint32_t end = upper ? col + columns : t.n;
            for (uint32_t row = upper ? 0U : col; row < end; row += Tile) {
                const uint32_t rows = end - row < Tile ? end - row : Tile;
                const uint32_t pitch = (2U * rows + 7U) & ~7U;
                LoadTile(upper, col, row, rows, columns, pitch, t);
                PipeBarrier<PIPE_ALL>();
                Csyr2Vector::Rotate(v, index.ReinterpretCast<uint32_t>(), 2U * rows);
                PipeBarrier<PIPE_V>();
                UpdateTile(upper, col, row, rows, columns, pitch, t);
                PipeBarrier<PIPE_ALL>();
                StoreTile(upper, col, row, rows, columns, pitch, t);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
};
template <bool GeneralAlpha>
__global__ __aicore__ __vector__ void csyr2_unit_tiled(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR aGm, Csyr2TilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // Twelve row buffers plus the coefficient/head buffers must fit in 192 KiB.
    constexpr uint32_t Tile = GeneralAlpha ? 1984U : 512U;
    static_assert(!GeneralAlpha || 12U * 8U * Tile + 192U <= 192U * 1024U);
    const uint32_t bands = (t.n + 3U) / 4U;
    const uint32_t first = uint64_t(bands) * GetBlockIdx() / GetBlockNum() * 4U;
    const uint32_t limit = uint64_t(bands) * (GetBlockIdx() + 1U) / GetBlockNum() * 4U;
    const uint32_t last = limit < t.n ? limit : t.n;
    if (first >= last) {
        return;
    }
    TPipe pipe;
    UnitTiles<GeneralAlpha> tiles;
    tiles.Init(pipe, xGm, yGm, aGm);
    tiles.Run(first, last, t);
}

void csyr2_unit_tiled_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    csyr2_unit_tiled<false><<<blocks, nullptr, stream>>>(x, y, a, t);
}

void csyr2_alpha_tiled_do(uint32_t blocks, void* stream, uint8_t* x, uint8_t* y, uint8_t* a, const Csyr2TilingData& t)
{
    csyr2_unit_tiled<true><<<blocks, nullptr, stream>>>(x, y, a, t);
}
