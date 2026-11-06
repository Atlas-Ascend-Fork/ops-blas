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
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

namespace gemm_ex {
__aicore__ inline uint32_t TileRow(uint32_t core, uint32_t columns, uint32_t reciprocal)
{
    return columns == 1 ? core : static_cast<uint32_t>((static_cast<uint64_t>(core) * reciprocal) >> 32);
}

template <typename LA, typename LB>
struct TileTypes {
    using AT = Catlass::Gemm::GemmType<float, LA>;
    using BT = Catlass::Gemm::GemmType<float, LB>;
    using CT = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using Copy = Catlass::Gemm::Tile::TileCopy<Catlass::Arch::AtlasA2, AT, BT, CT>;
    using L1A = typename Copy::CopyGmToL1A::LayoutDst;
    using L1B = typename Copy::CopyGmToL1B::LayoutDst;
    using L0A = typename Copy::CopyL1ToL0A::LayoutDst;
    using L0B = typename Copy::CopyL1ToL0B::LayoutDst;
    using Multiply = Catlass::Gemm::Tile::TileMmad<Catlass::Arch::AtlasA2, AT, BT, void>;
};

__aicore__ inline void StoreNz(
    AscendC::GlobalTensor<float> destination, AscendC::LocalTensor<float> accumulator, uint32_t columns, uint32_t rows,
    uint32_t stride)
{
    AscendC::FixpipeParamsV220 output;
    output.nSize = columns;
    output.mSize = rows;
    output.srcStride = rows;
    output.dstStride = stride;
    output.quantPre = ::QuantMode_t::NoQuant;
    output.isChannelSplit = false;
    output.unitFlag = 3;
    AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(destination, accumulator, output);
}
// Shared tile primitives only. Each public interface retains a separately
// compiled, independently named kernel and its own argument type.
template <typename Args>
class ShortNnTile {
    using LA = Catlass::layout::RowMajor;
    using LB = Catlass::layout::RowMajor;
    static constexpr uint32_t TM = 64;
    static constexpr uint32_t TN = 368;
    using AT = Catlass::Gemm::GemmType<float, LA>;
    using BT = Catlass::Gemm::GemmType<float, LB>;
    using CT = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using Copy = Catlass::Gemm::Tile::TileCopy<Catlass::Arch::AtlasA2, AT, BT, CT>;
    using L1A = typename Copy::CopyGmToL1A::LayoutDst;
    using L1B = typename Copy::CopyGmToL1B::LayoutDst;
    using L0A = typename Copy::CopyL1ToL0A::LayoutDst;
    using L0B = typename Copy::CopyL1ToL0B::LayoutDst;

public:
    __aicore__ inline explicit ShortNnTile(Args parameters) : t(parameters), la(t.n, t.k, t.ldb), lb(t.k, t.m, t.lda)
    {
        core = AscendC::GetBlockIdx();
        row = core >> 1;
        col = core & 1U;
        mb = row * 64U;
        nb = col * TN;
        am = (64U < t.n - mb ? 64U : t.n - mb);
        an = (TN < t.m - nb ? TN : t.m - nb);
        mr = ::RoundUp<Catlass::Gemm::helper::L1AlignHelper<float, LA>::M_ALIGNED>(am);
        nr = 368;
        const bool reverse = ((row & 1U) != 0);
        firstStart = reverse ? 16U : 0U;
        secondStart = reverse ? 0U : 16U;
        firstK = reverse ? t.k - 16U : 16U;
        secondK = reverse ? 16U : t.k - 16U;
    }
    __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR cGm)
    {
        InitGlobal(a, b, cGm);
        Load();
        Compute();
        Store();
    }

private:
    Args t;
    uint32_t core;
    uint32_t row;
    uint32_t col;
    uint32_t mb;
    uint32_t nb;
    uint32_t am;
    uint32_t an;
    uint32_t mr;
    uint32_t nr;
    uint32_t firstStart;
    uint32_t secondStart;
    uint32_t firstK;
    uint32_t secondK;
    AscendC::LocalTensor<float> a1{AscendC::TPosition::A1, 0, TM * 32};
    AscendC::LocalTensor<float> b1{AscendC::TPosition::A1, TM * 32 * 4, 32 * TN};
    // Head and tail occupy disjoint physical L0 regions; first MMAD never reads tail.
    AscendC::LocalTensor<float> aHead{AscendC::TPosition::A2, 0, TM * 16};
    AscendC::LocalTensor<float> aTail{AscendC::TPosition::A2, TM * 16 * 4, TM * 16};
    AscendC::LocalTensor<float> bHead{AscendC::TPosition::B2, 0, 16 * TN};
    AscendC::LocalTensor<float> bTail{AscendC::TPosition::B2, 32768, 16 * TN};
    AscendC::LocalTensor<float> acc{AscendC::TPosition::CO1, 0, TM* TN};
    AscendC::GlobalTensor<float> right;
    AscendC::GlobalTensor<float> scratch;
    AscendC::GlobalTensor<half> leftBits;
    AscendC::GlobalTensor<uint8_t> rightView;
    LA la;
    LB lb;
    decltype(L1A::template MakeLayout<float>(TM, 32)) aLayout = L1A::template MakeLayout<float>(TM, 32);
    decltype(L1B::template MakeLayout<float>(16, TN)) bLayout = L1B::template MakeLayout<float>(16, TN);
    typename Copy::CopyL1ToL0A toA;
    typename Copy::CopyL1ToL0B toB;
    __aicore__ inline void InitGlobal(GM_ADDR a, GM_ADDR b, GM_ADDR cGm)
    {
        leftBits.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b));
        right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        scratch.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm) + static_cast<uint64_t>(core) * TM * TN);
        rightView.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a));
    }
    __aicore__ inline void Load()
    {
        AscendC::DataCopy(
            a1.ReinterpretCast<half>(), leftBits[la.GetOffset(Catlass::MatrixCoord(mb, 0)) * 2],
            AscendC::Nd2NzParams{
                1, static_cast<uint16_t>(am), static_cast<uint16_t>(t.k * 2), 0, static_cast<uint16_t>(t.ldb * 2), TM,
                1, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::DataCopy(
            b1.ReinterpretCast<uint8_t>(), rightView[(static_cast<uint64_t>(firstStart) * t.lda + nb) * 4],
            AscendC::Nd2NzParams{
                1, static_cast<uint16_t>(firstK), static_cast<uint16_t>(an * 4), 0, static_cast<uint16_t>(t.lda * 4),
                16, 1, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        AscendC::DataCopy(
            b1[16 * TN].ReinterpretCast<uint8_t>(), rightView[(static_cast<uint64_t>(secondStart) * t.lda + nb) * 4],
            AscendC::Nd2NzParams{
                1, static_cast<uint16_t>(secondK), static_cast<uint16_t>(an * 4), 0, static_cast<uint16_t>(t.lda * 4),
                16, 1, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
    }
    __aicore__ inline void Compute()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        toA(aHead, a1[aLayout.GetOffset(Catlass::MatrixCoord(0U, firstStart))],
            L0A::template MakeLayout<float>(mr, firstK), aLayout);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        toB(bHead, b1, L0B::template MakeLayout<float>(firstK, nr), bLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        Catlass::Gemm::Tile::TileMmad<Catlass::Arch::AtlasA2, AT, BT, void> multiply;
        multiply(acc, aHead, bHead, mr, nr, firstK, true, 2);
        // Delay only this already-ready A copy; overlap it with the B-tail DMA.
        toA(aTail, a1[aLayout.GetOffset(Catlass::MatrixCoord(0U, secondStart))],
            L0A::template MakeLayout<float>(mr, secondK), aLayout);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        toB(bTail, b1[16 * TN], L0B::template MakeLayout<float>(secondK, nr), bLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID1);
        multiply(acc, aTail, bTail, mr, nr, secondK, false, 3);
    }
    __aicore__ inline void Store()
    {
        AscendC::FixpipeParamsV220 out;
        out.nSize = nr;
        out.mSize = mr;
        out.srcStride = mr;
        out.dstStride = TM * 2;
        out.quantPre = ::QuantMode_t::NoQuant;
        out.isChannelSplit = false;
        out.unitFlag = 3;
        AscendC::Fixpipe<float, float, AscendC::CFG_NZ>(scratch, acc, out);
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
    }
};
template <typename Args, bool Stagger>
class ShortTtTile {
public:
    __aicore__ inline explicit ShortTtTile(Args parameters) : t(parameters) {}
    __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR cGm)
    {
        if (core >= grid) {
            return;
        }
        left.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        scratch.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm));
        for (uint32_t linear = core; linear < grid; linear += AscendC::GetBlockNum()) {
            RunTile(linear);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Args t;
    using LA = Catlass::layout::ColumnMajor;
    using LB = Catlass::layout::ColumnMajor;
    static constexpr uint32_t TM = 192;
    static constexpr uint32_t TN = 128;
    static constexpr uint32_t TK = 32;
    const uint32_t core = AscendC::GetBlockIdx();
    const uint32_t columns = t.columns;
    const uint32_t grid = ((t.n + TM - 1) / TM) * columns;
    using Types = gemm_ex::TileTypes<LA, LB>;
    AscendC::LocalTensor<float> l1a{AscendC::TPosition::A1, 0, TM* TK};
    AscendC::LocalTensor<float> l1b{AscendC::TPosition::A1, TM* TK * 4, TK* TN};
    AscendC::LocalTensor<float> l0a{AscendC::TPosition::A2, 0, TM* TK};
    AscendC::LocalTensor<float> l0b{AscendC::TPosition::B2, 0, TK* TN};
    AscendC::LocalTensor<float> acc{AscendC::TPosition::CO1, 0, TM* TN};
    AscendC::GlobalTensor<float> left;
    AscendC::GlobalTensor<float> right;
    AscendC::GlobalTensor<float> scratch;
    LA la{t.n, t.k, t.ldb};
    LB lb{t.k, t.m, t.lda};
    decltype(Types::L1A::template MakeLayout<float>(TM, TK)) layoutL1A = Types::L1A::template MakeLayout<float>(TM, TK);
    decltype(Types::L1B::template MakeLayout<float>(TK, TN)) layoutL1B = Types::L1B::template MakeLayout<float>(TK, TN);
    typename Types::Copy::CopyGmToL1A gmToA;
    typename Types::Copy::CopyGmToL1B gmToB;
    typename Types::Copy::CopyL1ToL0A l1ToA;
    typename Types::Copy::CopyL1ToL0B l1ToB;
    typename Types::Copy::CopyL0CToGm store;
    typename Types::Multiply multiply;
    __aicore__ inline void RunTile(uint32_t linear)
    {
        const uint32_t row =
            columns == 1 ? linear : static_cast<uint32_t>((static_cast<uint64_t>(linear) * t.reciprocal) >> 32);
        const uint32_t col = linear - row * columns;
        const uint32_t mb = row * TM;
        const uint32_t nb = col * TN;
        const uint32_t am = (TM < t.n - mb ? TM : t.n - mb);
        const uint32_t an = (TN < t.m - nb ? TN : t.m - nb);
        const uint32_t mr = (am + 15) / 16 * 16;
        const uint32_t nr = ::RoundUp<Catlass::Gemm::helper::L1AlignHelper<float, LB>::N_ALIGNED>(an);
        if (linear != core) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        }
        gmToA(
            l1a, left[la.GetOffset(Catlass::MatrixCoord(mb, 0))], layoutL1A,
            la.GetTileLayout(Catlass::MakeCoord(am, t.k)));
        if constexpr (Stagger) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        }
        gmToB(
            l1b, right[lb.GetOffset(Catlass::MatrixCoord(0, nb))], layoutL1B,
            lb.GetTileLayout(Catlass::MakeCoord(t.k, an)));
        if constexpr (Stagger) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        }
        if (linear != core) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        l1ToA(l0a, l1a, Types::L0A::template MakeLayout<float>(mr, t.k), layoutL1A);
        if constexpr (Stagger) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        }
        l1ToB(l0b, l1b, Types::L0B::template MakeLayout<float>(t.k, nr), layoutL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        if (linear != core) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        multiply(acc, l0a, l0b, mr, nr, t.k, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        const uint32_t np = (t.n + 15) / 16 * 16;
        gemm_ex::StoreNz(scratch[static_cast<uint64_t>(mb) * 16 + static_cast<uint64_t>(nb) * np], acc, nr, mr, np * 2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }
};
struct ResidentBuffers {
    AscendC::LocalTensor<float> l1a;
    AscendC::LocalTensor<float> l1b;
    AscendC::LocalTensor<float> l0a;
    AscendC::LocalTensor<float> l0b;
    AscendC::LocalTensor<float> acc;
};
template <typename Types, typename ALayout, typename BLayout, typename Layout>
__aicore__ inline void LoadResidentTile(
    ResidentBuffers buffers, ALayout aLayout, BLayout bLayout, AscendC::GlobalTensor<float> left,
    AscendC::GlobalTensor<float> right, Layout la, Layout lb, uint32_t mb, uint32_t nb, uint32_t am, uint32_t an,
    uint32_t nr, uint32_t k)
{
    // ND2NZ pads FP32 columns to eight; NZ output consumes sixteen.
    if ((an + 7) / 8 * 8 != nr) {
        const auto tailOffset = bLayout.GetOffset(Catlass::MatrixCoord(0, (an + 7) / 8 * 8));
        AscendC::InitConstValue(buffers.l1b[tailOffset], AscendC::InitConstValueParams<float>(1, 128, 0, 0.0f));
    }
    typename Types::Copy::CopyGmToL1A gmToA;
    typename Types::Copy::CopyGmToL1B gmToB;
    gmToA(
        buffers.l1a, left[la.GetOffset(Catlass::MatrixCoord(mb, 0))], aLayout,
        la.GetTileLayout(Catlass::MakeCoord(am, k)));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    gmToB(
        buffers.l1b, right[lb.GetOffset(Catlass::MatrixCoord(0, nb))], bLayout,
        lb.GetTileLayout(Catlass::MakeCoord(k, an)));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
}
template <typename Types, typename ALayout, typename BLayout>
__aicore__ inline void ComputeLoadedTile(
    ResidentBuffers buffers, ALayout layoutL1A, BLayout layoutL1B, AscendC::GlobalTensor<float> scratch, uint32_t mr,
    uint32_t nr, uint32_t k, uint32_t stride)
{
    auto l1a = buffers.l1a;
    auto l1b = buffers.l1b;
    auto l0a = buffers.l0a;
    auto l0b = buffers.l0b;
    auto acc = buffers.acc;
    typename Types::Copy::CopyL1ToL0A l1ToA;
    typename Types::Copy::CopyL1ToL0B l1ToB;
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    l1ToA(l0a, l1a, Types::L0A::template MakeLayout<float>(mr, k), layoutL1A);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
    l1ToB(l0b, l1b, Types::L0B::template MakeLayout<float>(k, nr), layoutL1B);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    typename Types::Multiply multiply;
    multiply(acc, l0a, l0b, mr, nr, k, true, 3);
    gemm_ex::StoreNz(scratch, acc, nr, mr, stride);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
}
template <typename Args, typename Types, typename ALayout, typename BLayout>
__aicore__ inline void ComputeResidentTile(
    ResidentBuffers buffers, ALayout aLayout, BLayout bLayout, AscendC::GlobalTensor<float> scratch, Args t,
    uint32_t mr, uint32_t nr, uint32_t np)
{
    constexpr uint32_t TM = 80;
    constexpr uint32_t TN = 256;
    constexpr uint32_t BK = 32;
    constexpr uint32_t TK = 128;
    static_assert((TM + TN) * TK * sizeof(float) <= Catlass::Arch::AtlasA2::L1_SIZE);
    static_assert(2 * TM * BK * sizeof(float) <= Catlass::Arch::AtlasA2::L0A_SIZE);
    static_assert(2 * TN * BK * sizeof(float) <= Catlass::Arch::AtlasA2::L0B_SIZE);
    static_assert(TM * TN * sizeof(float) <= Catlass::Arch::AtlasA2::L0C_SIZE);
    auto l1a = buffers.l1a;
    auto l1b = buffers.l1b;
    auto l0a = buffers.l0a;
    auto l0b = buffers.l0b;
    auto acc = buffers.acc;
    typename Types::Copy::CopyL1ToL0A l1ToA;
    typename Types::Copy::CopyL1ToL0B l1ToB;
    typename Types::Multiply multiply;
    uint32_t part = 0;
    for (uint32_t k0 = 0; k0 < t.k; k0 += BK, ++part) {
        const uint32_t kk = (BK < t.k - k0 ? BK : t.k - k0);
        const uint32_t slot = part & 1U;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(slot);
        if (k0 == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        }
        l1ToA(
            l0a[slot * TM * BK], l1a[aLayout.GetOffset(Catlass::MatrixCoord(0, k0))],
            Types::L0A::template MakeLayout<float>(mr, kk), aLayout);
        if (k0 == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        }
        l1ToB(
            l0b[slot * TN * BK], l1b[bLayout.GetOffset(Catlass::MatrixCoord(k0, 0))],
            Types::L0B::template MakeLayout<float>(kk, nr), bLayout);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
        // Keep L0C occupied across K; only the last MMAD releases it to FIX.
        multiply(acc, l0a[slot * TM * BK], l0b[slot * TN * BK], mr, nr, kk, k0 == 0, k0 + kk == t.k ? 3 : 2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slot);
    }
    gemm_ex::StoreNz(scratch, acc, nr, mr, np * 2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::PipeBarrier<PIPE_ALL>();
}
} // namespace gemm_ex
