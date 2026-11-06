/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CATLASS-backed arch22 implementation; dependency revision is pinned by CMake.
#pragma once
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#include "gemm_ex_dma_copy.h"
#include "gemm_ex_tile_primitives.h"
#include "gemm_ex_tiling.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace gemm_ex {
using Catlass::MakeCoord;
using Catlass::MatrixCoord;

template <class LA, class LB, uint32_t TM, uint32_t TN, uint32_t TK, bool SMALL, typename Args>
__aicore__ inline void RunColumnOwner(GM_ADDR a, GM_ADDR b, GM_ADDR c, Args t)
{
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    using Policy = std::conditional_t<
        SMALL, Catlass::Gemm::MmadAtlasA2Small<1, true, false>, Catlass::Gemm::MmadAtlasA2Pingpong<true>>;
    using Block = Catlass::Gemm::Block::BlockMmad<
        Policy, Catlass::GemmShape<TM, TN, TK>, Catlass::GemmShape<TM, TN, 32>, Catlass::Gemm::GemmType<float, LA>,
        Catlass::Gemm::GemmType<float, LB>, Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>>;
    const uint32_t groups = (static_cast<uint32_t>(t.n) + TM - 1) / TM;
    if (AscendC::GetBlockIdx() >= groups) {
        return;
    }
    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
    Block block(resource);
    auto left = BindGlobal(reinterpret_cast<__gm__ float*>(b));
    auto right = BindGlobal(reinterpret_cast<__gm__ float*>(a));
    auto out = BindGlobal(reinterpret_cast<__gm__ float*>(c));
    const LA la(t.n, t.k, t.ldb);
    const LB lb(t.k, t.m, t.lda);
    const Catlass::layout::RowMajor lc(t.n, t.m, t.ldc);
    for (uint32_t group = AscendC::GetBlockIdx(); group < groups; group += AscendC::GetBlockNum()) {
        const uint32_t mb = group * TM;
        const uint32_t am = min(TM, static_cast<uint32_t>(t.n) - mb);
        for (uint32_t nb = 0; nb < static_cast<uint32_t>(t.m); nb += TN) {
            const uint32_t an = min(TN, static_cast<uint32_t>(t.m) - nb);
            block(
                left[la.GetOffset(MatrixCoord(mb, 0))], la, right[lb.GetOffset(MatrixCoord(0, nb))], lb,
                out[lc.GetOffset(MatrixCoord(mb, nb))], lc, Catlass::GemmCoord(am, an, t.k));
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <class LayoutA, class LayoutB, uint32_t TM, uint32_t TN, uint32_t TK, typename Args>
__aicore__ inline void RunCatlass(GM_ADDR a, GM_ADDR b, GM_ADDR c, Args t)
{
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    using Policy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
    using Block = Catlass::Gemm::Block::BlockMmad<
        Policy, Catlass::GemmShape<TM, TN, TK>, Catlass::GemmShape<TM, TN, 32>, Catlass::Gemm::GemmType<float, LayoutA>,
        Catlass::Gemm::GemmType<float, LayoutB>, Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>>;

    const uint32_t columns = t.gridAndColumns >> 16;
    const uint32_t grid = t.gridAndColumns & 65535U;
    const uint32_t first = AscendC::GetBlockIdx();
    if (first >= grid) {
        return;
    }
    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
    Block block(resource);
    auto left = BindGlobal(reinterpret_cast<__gm__ float*>(b));
    auto right = BindGlobal(reinterpret_cast<__gm__ float*>(a));
    auto out = BindGlobal(reinterpret_cast<__gm__ float*>(c));
    LayoutA layoutA(t.n, t.k, t.ldb);
    LayoutB layoutB(t.k, t.m, t.lda);
    Catlass::layout::RowMajor layoutC(t.n, t.m, t.ldc);
    for (uint32_t linear = first; linear < grid; linear += AscendC::GetBlockNum()) {
        // numerator<1024, denominator<=64; the ceil reciprocal is exact in this
        // bounded domain. Host asserts this mapping for every launched tile.
        const uint32_t row =
            columns == 1 ? linear : static_cast<uint32_t>((static_cast<uint64_t>(linear) * t.reciprocalColumns) >> 32);
        const uint32_t col = linear - row * columns;
        const uint32_t mb = row * TM;
        const uint32_t nb = col * TN;
        const uint32_t remainingM = static_cast<uint32_t>(t.n) - mb;
        const uint32_t remainingN = static_cast<uint32_t>(t.m) - nb;
        Catlass::GemmCoord actual(remainingM < TM ? remainingM : TM, remainingN < TN ? remainingN : TN, t.k);
        const auto offsetA = layoutA.GetOffset(MatrixCoord(mb, 0));
        const auto offsetB = layoutB.GetOffset(MatrixCoord(0, nb));
        const auto offsetC = layoutC.GetOffset(MatrixCoord(mb, nb));
        block(left[offsetA], layoutA, right[offsetB], layoutB, out[offsetC], layoutC, actual);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename Args>
__aicore__ inline void RunShortNative(GM_ADDR a, GM_ADDR b, GM_ADDR cGm, Args t)
{
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    using LA = Catlass::layout::RowMajor;
    using LB = Catlass::layout::RowMajor;
    constexpr uint32_t TM = 64;
    constexpr uint32_t TN = 384;
    const uint32_t core = AscendC::GetBlockIdx();
    using Types = gemm_ex::TileTypes<LA, LB>;
    const uint32_t row = gemm_ex::TileRow(core, t.columns, t.reciprocal);
    const uint32_t col = core - row * t.columns;
    const uint32_t mb = row * TM;
    const uint32_t nb = col * TN;
    const uint32_t am = min(TM, t.n - mb);
    const uint32_t an = min(TN, t.m - nb);
    const uint32_t mr = ::RoundUp<Catlass::Gemm::helper::L1AlignHelper<float, LA>::M_ALIGNED>(am);
    const uint32_t nr = ::RoundUp<Catlass::Gemm::helper::L1AlignHelper<float, LB>::N_ALIGNED>(an);
    AscendC::LocalTensor<float> l1a(AscendC::TPosition::A1, 0, TM * 32);
    AscendC::LocalTensor<float> l1b(AscendC::TPosition::A1, TM * 32 * 4, 32 * TN);
    AscendC::LocalTensor<float> l0a(AscendC::TPosition::A2, 0, TM * 32);
    AscendC::LocalTensor<float> l0b(AscendC::TPosition::B2, 0, 32 * TN);
    AscendC::LocalTensor<float> acc(AscendC::TPosition::CO1, 0, TM * TN);
    auto left = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(b));
    auto right = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(a));
    auto scratch = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(cGm) + static_cast<uint64_t>(mb) * t.ldc + nb);
    LA la(t.n, t.k, t.ldb);
    LB lb(t.k, t.m, t.lda);
    auto layoutL1A = Types::L1A::template MakeLayout<float>(TM, 32);
    auto layoutL1B = Types::L1B::template MakeLayout<float>(32, TN);
    typename Types::Copy::CopyGmToL1A gmToA;
    typename Types::Copy::CopyGmToL1B gmToB;
    gmToA(l1a, left[la.GetOffset(MatrixCoord(mb, 0))], layoutL1A, la.GetTileLayout(MakeCoord(am, t.k)));
    gmToB(l1b, right[lb.GetOffset(MatrixCoord(0, nb))], layoutL1B, lb.GetTileLayout(MakeCoord(t.k, an)));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    typename Types::Copy::CopyL1ToL0A l1ToA;
    typename Types::Copy::CopyL1ToL0B l1ToB;
    l1ToA(l0a, l1a, Types::L0A::template MakeLayout<float>(mr, t.k), layoutL1A);
    l1ToB(l0b, l1b, Types::L0B::template MakeLayout<float>(t.k, nr), layoutL1B);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    typename Types::Multiply multiply;
    multiply(acc, l0a, l0b, mr, nr, t.k, true, 0b11);
    typename Types::Copy::CopyL0CToGm store;
    store(
        scratch, acc, Catlass::layout::RowMajor(am, an, t.ldc), Catlass::layout::zN::MakeLayoutInL0C(MakeCoord(mr, nr)),
        0b11);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename Args>
__aicore__ inline void RunModerateNz(GM_ADDR a, GM_ADDR b, GM_ADDR cGm, Args t)
{
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    AscendC::SetMMColumnMajor();
    using R = Catlass::layout::RowMajor;
    using Types = gemm_ex::TileTypes<R, R>;
    using TileCopy = typename Types::Copy;
    using L1A = typename Types::L1A;
    using L1B = typename Types::L1B;
    constexpr uint32_t TM = 80;
    constexpr uint32_t TN = 256;
    constexpr uint32_t TK = 128;
    constexpr uint32_t BK = 32;

    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t columns = t.columns;
    const uint32_t row =
        columns == 1 ? core : static_cast<uint32_t>((static_cast<uint64_t>(core) * t.reciprocal) >> 32);
    const uint32_t mb = row * TM;
    const uint32_t nb = (core - row * columns) * TN;
    const uint32_t am = min(TM, t.n - mb);
    const uint32_t an = min(TN, t.m - nb);
    const uint32_t mr = (am + 15) / 16 * 16;
    const uint32_t nr = (an + 15) / 16 * 16;
    const uint32_t np = (t.n + 15) / 16 * 16;
    AscendC::LocalTensor<float> l1a(AscendC::TPosition::A1, 0, TM * TK);
    AscendC::LocalTensor<float> l1b(AscendC::TPosition::A1, TM * TK * sizeof(float), TN * TK);
    AscendC::LocalTensor<float> l0a(AscendC::TPosition::A2, 0, 2 * TM * BK);
    AscendC::LocalTensor<float> l0b(AscendC::TPosition::B2, 0, 2 * TN * BK);
    AscendC::LocalTensor<float> acc(AscendC::TPosition::CO1, 0, TM * TN);
    auto left = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(b));
    auto right = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(a));
    auto scratch = gemm_ex::BindGlobal(
        reinterpret_cast<__gm__ float*>(cGm) + static_cast<uint64_t>(mb) * 16 + static_cast<uint64_t>(nb) * np);
    const R la(t.n, t.k, t.ldb);
    const R lb(t.k, t.m, t.lda);
    const auto aLayout = L1A::template MakeLayout<float>(TM, TK);
    const auto bLayout = L1B::template MakeLayout<float>(TK, TN);
    gemm_ex::LoadResidentTile<Types>(
        {l1a, l1b, l0a, l0b, acc}, aLayout, bLayout, left, right, la, lb, mb, nb, am, an, nr, t.k);
    gemm_ex::ComputeResidentTile<Args, Types>({l1a, l1b, l0a, l0b, acc}, aLayout, bLayout, scratch, t, mr, nr, np);
}

template <uint32_t TM, uint32_t TK, uint32_t SPLITS, typename Args>
__aicore__ inline void RunSplitNt(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR outGm, Args t)
{
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    constexpr uint32_t TN = 128;
    using LA = Catlass::layout::ColumnMajor;
    using LB = Catlass::layout::RowMajor;
    using Policy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
    using Block = Catlass::Gemm::Block::BlockMmad<
        Policy, Catlass::GemmShape<TM, TN, TK>, Catlass::GemmShape<TM, TN, 32>, Catlass::Gemm::GemmType<float, LA>,
        Catlass::Gemm::GemmType<float, LB>, Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>>;
    const uint32_t groups = (t.n + TM - 1) / TM;
    const uint32_t chunks = (t.k + 31) / 32;
    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
    Block block(resource);
    auto left = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(bGm));
    auto right = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(aGm));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outGm));
    const LA la(t.n, t.k, t.ldb);
    const LB lb(t.k, t.m, t.lda);
    const Catlass::layout::RowMajor lc(t.n, t.m, t.ldc);
    for (uint32_t job = AscendC::GetBlockIdx(); job < groups * SPLITS; job += AscendC::GetBlockNum()) {
        const uint32_t part =
            groups == 1 ? job : static_cast<uint32_t>((static_cast<uint64_t>(job) * t.reciprocalColumns) >> 32);
        const uint32_t row = job - part * groups;
        const uint32_t mb = row * TM;
        const uint32_t first = (chunks * part / SPLITS) * 32;
        const uint32_t last = min(static_cast<uint32_t>(t.k), (chunks * (part + 1) / SPLITS) * 32);
        block(
            left[la.GetOffset(MatrixCoord(mb, first))], la, right[lb.GetOffset(MatrixCoord(first, 0))], lb,
            dest[static_cast<uint64_t>(part) * t.n * t.ldc + static_cast<uint64_t>(mb) * t.ldc], lc,
            Catlass::GemmCoord(min(TM, static_cast<uint32_t>(t.n) - mb), static_cast<uint32_t>(t.m), last - first));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunCatlassCopy(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n, uint32_t ldc, uint32_t sourceLd, uint32_t per,
    uint32_t batch)
{
    constexpr uint32_t CAP = 49152;
    const uint32_t first = AscendC::GetBlockIdx() * per;
    const uint32_t end = min(first + per, static_cast<uint32_t>(n));
    if (first >= end) {
        return;
    }
    AscendC::LocalTensor<float> local(AscendC::TPosition::VECCALC, 0, CAP);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(inputGm));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outputGm));

    for (uint32_t col = first; col < end; col += batch) {
        const uint32_t count = min(batch, end - col);
        AscendC::DataCopy(local, source[static_cast<uint64_t>(col) * sourceLd], count * sourceLd);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::GatherMaskParams compact{1, static_cast<uint16_t>(count), static_cast<uint16_t>(sourceLd / 8), 0};
        uint64_t reserved = 0;
        AscendC::GatherMask(local, local, 7, true, m, compact, reserved);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        const uint32_t elements = count * m;
        if (elements % 8 == 0) {
            AscendC::DataCopy(dest[static_cast<uint64_t>(col) * ldc], local, elements);
        } else {
            AscendC::DataCopyPad(dest[static_cast<uint64_t>(col) * ldc], local, {1, elements * 4U, 0, 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}

template <typename Args>
__aicore__ inline void RunCatlassPack(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR paGm, GM_ADDR pbGm, Args t)
{
    AscendC::LocalTensor<float> local(AscendC::TPosition::VECCALC, 0, 32768);
    auto a = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(aGm));
    auto b = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(bGm));
    auto pa = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(paGm));
    auto pb = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(pbGm));
    Pack512Compact(a, pa, local, t.a);
    Pack512Compact(b, pb, local, t.b);
}

template <typename Args>
__aicore__ inline void RunDmaPack(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR paGm, GM_ADDR pbGm, Args t)
{
    AscendC::LocalTensor<float> localA(AscendC::TPosition::VECCALC, 0, 8192);
    AscendC::LocalTensor<float> localB(AscendC::TPosition::VECCALC, 8192 * sizeof(float), 2048);
    auto a = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(aGm));
    auto b = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(bGm));
    auto pa = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(paGm));
    auto pb = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(pbGm));
    const uint32_t perA = t.a.perAndBatch;
    const uint32_t perB = t.b.perAndBatch;
    const uint32_t firstA = AscendC::GetBlockIdx() * perA;
    const uint32_t firstB = AscendC::GetBlockIdx() * perB;
    const uint32_t countA = firstA < t.a.rows ? min(perA, t.a.rows - firstA) : 0;
    const uint32_t countB = firstB < t.b.rows ? min(perB, t.b.rows - firstB) : 0;
    if (countA) {
        const uint32_t dma = (t.a.width + 7) / 8 * 8;
        const AscendC::DataCopyExtParams cp{static_cast<uint16_t>(countA), t.a.width * 4U, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(dma - t.a.width), 0.0f};
        AscendC::DataCopyPad(localA, a[static_cast<uint64_t>(firstA) * t.a.width], cp, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    }
    if (countB) {
        const uint32_t dma = (t.b.width + 7) / 8 * 8;
        const AscendC::DataCopyExtParams cp{static_cast<uint16_t>(countB), t.b.width * 4U, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(dma - t.b.width), 0.0f};
        AscendC::DataCopyPad(localB, b[static_cast<uint64_t>(firstB) * t.b.width], cp, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
    }
    if (countA) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        const AscendC::DataCopyExtParams cp{
            static_cast<uint16_t>(countA), t.a.width * 4U, 0, (t.a.outputLd - t.a.width) * 4U, 0};
        AscendC::DataCopyPad(pa[static_cast<uint64_t>(firstA) * t.a.outputLd], localA, cp);
    }
    if (countB) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        const AscendC::DataCopyExtParams cp{
            static_cast<uint16_t>(countB), t.b.width * 4U, 0, (t.b.outputLd - t.b.width) * 4U, 0};
        AscendC::DataCopyPad(pb[static_cast<uint64_t>(firstB) * t.b.outputLd], localB, cp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunNzFourrowCopy(GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    const uint32_t first = AscendC::GetBlockIdx() * 4;
    if (first >= n) {
        return;
    }
    const uint32_t width = (m + 15) / 16 * 16;
    const uint32_t np = (n + 15) / 16 * 16;
    const uint32_t cap = 4 * width;
    AscendC::LocalTensor<float> raw(AscendC::TPosition::VECCALC, 0, cap);
    AscendC::LocalTensor<float> rows(AscendC::TPosition::VECCALC, cap * sizeof(float), cap);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(inputGm));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outputGm));
    const AscendC::DataCopyParams load{static_cast<uint16_t>(width / 16), 8, static_cast<uint16_t>((np - 4) * 2), 0};
    AscendC::DataCopy(raw, source[static_cast<uint64_t>(first) * 16], load);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    const AscendC::CopyRepeatParams transpose{static_cast<uint16_t>(width / 8), 2, 2, 8};
    AscendC::Copy(rows, raw, uint64_t(32), static_cast<uint8_t>(width / 16), transpose);
    AscendC::Copy(rows[8], raw[8], uint64_t(32), static_cast<uint8_t>(width / 16), transpose);
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t count = min(4U, n - first);
    const uint32_t elements = count * m;
    uint64_t reserved = 0;
    AscendC::GatherMask(
        rows, rows, 7, true, m, {1, static_cast<uint16_t>(count), static_cast<uint16_t>(width / 8), 0}, reserved);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    const uint64_t offset = static_cast<uint64_t>(first) * m;
    if (elements % 8 == 0 && offset % 8 == 0) {
        AscendC::DataCopy(dest[offset], rows, elements);
    } else {
        AscendC::DataCopyPad(dest[offset], rows, {1, elements * 4U, 0, 0, 0});
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunNzPipeline40(GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    const uint32_t first = AscendC::GetBlockIdx() * 8U;
    const uint32_t step = AscendC::GetBlockNum() * 8U;
    if (first >= n) {
        return;
    }
    AscendC::SetMaskNorm();
    const uint32_t width = (m + 15) / 16 * 16;
    const uint32_t np = (n + 15) / 16 * 16;
    const uint32_t cap = 8 * width;
    // Host bounds aligned m<=1536: four tensors occupy <=192 KiB UB.
    AscendC::LocalTensor<float> raw0(AscendC::TPosition::VECCALC, 0, cap);
    AscendC::LocalTensor<float> out0(AscendC::TPosition::VECCALC, cap * 4, cap);
    AscendC::LocalTensor<float> raw1(AscendC::TPosition::VECCALC, cap * 8, cap);
    AscendC::LocalTensor<float> out1(AscendC::TPosition::VECCALC, cap * 12, cap);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(inputGm));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(outputGm));
    AscendC::DataCopyParams load{static_cast<uint16_t>(width / 16), 16, static_cast<uint16_t>((np - 8) * 2), 0};
    AscendC::CopyRepeatParams scatter{static_cast<uint16_t>(width / 8), 2, 2, 16};
    AscendC::DataCopy(raw0, source[static_cast<uint64_t>(first) * 16], load);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    if (first + step < n) {
        AscendC::DataCopy(raw1, source[static_cast<uint64_t>(first + step) * 16], load);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    }
    gemm_ex::NzCopyWaves waves(m, n, width, step, raw0, raw1, out0, out1, source, dest, load, scatter);
    uint32_t wave = 0;
    for (uint32_t col = first; col < n; col += step, ++wave) {
        waves.Run(col, wave);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    if (first + step < n) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunSplitSum(
    GM_ADDR sourceGm, GM_ADDR destGm, uint32_t m, uint32_t n, uint32_t sourceLd, uint32_t splits)
{
    const uint32_t first = AscendC::GetBlockIdx() * 8;
    if (first >= n) {
        return;
    }
    const uint32_t count = min(8U, n - first);
    const uint32_t cap = count * sourceLd;
    AscendC::LocalTensor<float> local(AscendC::TPosition::VECCALC, 0, 8192);
    auto source = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(sourceGm));
    auto dest = gemm_ex::BindGlobal(reinterpret_cast<__gm__ float*>(destGm));
    for (uint32_t part = 0; part < splits; ++part) {
        AscendC::DataCopy(
            local[part * cap],
            source[static_cast<uint64_t>(part) * n * sourceLd + static_cast<uint64_t>(first) * sourceLd], cap);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t part = 1; part < splits; ++part) {
        AscendC::Add(local, local, local[part * cap], cap);
        AscendC::PipeBarrier<PIPE_V>();
    }
    uint64_t reserved = 0;
    AscendC::GatherMask(
        local, local, 7, true, m, {1, static_cast<uint16_t>(count), static_cast<uint16_t>(sourceLd / 8), 0}, reserved);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    const uint32_t elements = count * m;
    if (elements % 8 == 0) {
        AscendC::DataCopy(dest[static_cast<uint64_t>(first) * m], local, elements);
    } else {
        AscendC::DataCopyPad(dest[static_cast<uint64_t>(first) * m], local, {1, elements * 4U, 0, 0, 0});
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace gemm_ex
