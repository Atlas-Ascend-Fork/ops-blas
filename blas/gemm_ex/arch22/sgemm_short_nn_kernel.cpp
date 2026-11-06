/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef GEMMEX_CATLASS_ENABLED
#include "kernel_operator.h"
#include "gemm_ex_tile_primitives.h"
#include "gemm_ex_tiling.h"
#include "gemm_ex_dma_copy.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"

using namespace AscendC;

struct Split101Args232 {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t columns;
    uint32_t reciprocal;
};

namespace {
template <typename Args>
class SplitShortNnTile {
public:
    __aicore__ inline explicit SplitShortNnTile(Args parameters)
        : t(parameters),
          core(GetBlockIdx()),
          row(t.columns == 1 ? core : static_cast<uint32_t>((static_cast<uint64_t>(core) * t.reciprocal) >> 32)),
          col(core - row * t.columns),
          mb(row * TM),
          nb(col * TN),
          am(min(TM, t.n - mb)),
          an(min(TN, t.m - nb)),
          mr(::RoundUp<Catlass::Gemm::helper::L1AlignHelper<float, LA>::M_ALIGNED>(am)),
          nr((an + 15) / 16 * 16),
          a1(TPosition::A1, 0, TM * 32),
          b1(TPosition::A1, TM * 32 * 4, 32 * TN),
          aHead(TPosition::A2, 0, TM * 16),
          aTail(TPosition::A2, TM * 16 * 4, TM * 16),
          bHead(TPosition::B2, 0, 16 * TN),
          bTail(TPosition::B2, 16 * TN * 4, 16 * TN),
          acc(TPosition::CO1, 0, TM * TN),
          la(t.n, t.k, t.ldb),
          lb(t.k, t.m, t.lda),
          aLayout(Types::L1A::template MakeLayout<float>(TM, 32)),
          bLayout(Types::L1B::template MakeLayout<float>(32, TN))
    {}

    __aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR cGm)
    {
        leftBits.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b));
        right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        scratch.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm) + static_cast<uint64_t>(core) * TM * TN);
        rightView.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(a));
        Load();
        Compute();
    }

private:
    Args t;
    using LA = Catlass::layout::RowMajor;
    using LB = Catlass::layout::RowMajor;
    static constexpr uint32_t TM = 64;
    static constexpr uint32_t TN = 368;
    using Types = gemm_ex::TileTypes<LA, LB>;
    const uint32_t core;
    const uint32_t row;
    const uint32_t col;
    const uint32_t mb;
    const uint32_t nb;
    const uint32_t am;
    const uint32_t an;
    const uint32_t mr;
    const uint32_t nr;
    LocalTensor<float> a1;
    LocalTensor<float> b1;
    // Head and tail occupy disjoint physical L0 regions; first MMAD never reads tail.
    LocalTensor<float> aHead;
    LocalTensor<float> aTail;
    LocalTensor<float> bHead;
    LocalTensor<float> bTail;
    LocalTensor<float> acc;
    GlobalTensor<float> right;
    GlobalTensor<float> scratch;
    GlobalTensor<half> leftBits;
    GlobalTensor<uint8_t> rightView;
    LA la;
    LB lb;
    decltype(Types::L1A::template MakeLayout<float>(TM, 32)) aLayout;
    decltype(Types::L1B::template MakeLayout<float>(32, TN)) bLayout;
    typename Types::Copy::CopyGmToL1B gmToB;
    typename Types::Copy::CopyL1ToL0A toA;
    typename Types::Copy::CopyL1ToL0B toB;
    __aicore__ inline void Load()
    {
        DataCopy(
            a1.ReinterpretCast<half>(), leftBits[la.GetOffset(Catlass::MatrixCoord(mb, 0)) * 2],
            Nd2NzParams{
                1, static_cast<uint16_t>(am), static_cast<uint16_t>(t.k * 2), 0, static_cast<uint16_t>(t.ldb * 2), TM,
                1, 0});
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        DataCopy(
            b1.ReinterpretCast<uint8_t>(), rightView[static_cast<uint64_t>(nb) * 4],
            Nd2NzParams{1, 16, static_cast<uint16_t>(an * 4), 0, static_cast<uint16_t>(t.lda * 4), 32, 1, 0});
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    }
    __aicore__ inline void Compute()
    {
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        toA(aHead, a1, Types::L0A::template MakeLayout<float>(mr, 16), aLayout);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        toB(bHead, b1, Types::L0B::template MakeLayout<float>(16, nr), bLayout);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        typename Types::Multiply multiply;
        multiply(acc, aHead, bHead, mr, nr, 16, true, t.k > 16 ? 2 : 3);
        if (t.k > 16) {
            ComputeTail(multiply);
        }
        gemm_ex::StoreNz(scratch, acc, nr, mr, TM * 2);
        SetFlag<HardEvent::FIX_S>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_S>(EVENT_ID0);
    }
    __aicore__ inline void ComputeTail(typename Types::Multiply& multiply)
    {
        DataCopy(
            b1[bLayout.GetOffset(Catlass::MatrixCoord(16U, 0U))].ReinterpretCast<uint8_t>(),
            rightView[(static_cast<uint64_t>(16) * t.lda + nb) * 4],
            Nd2NzParams{
                1, static_cast<uint16_t>(t.k - 16), static_cast<uint16_t>(an * 4), 0, static_cast<uint16_t>(t.lda * 4),
                32, 1, 0});
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
        // Delay only this already-ready A copy; overlap it with the B-tail DMA.
        toA(aTail, a1[aLayout.GetOffset(Catlass::MatrixCoord(0U, 16U))],
            Types::L0A::template MakeLayout<float>(mr, t.k - 16), aLayout);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
        toB(bTail, b1[bLayout.GetOffset(Catlass::MatrixCoord(16U, 0U))],
            Types::L0B::template MakeLayout<float>(t.k - 16, nr), bLayout);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
        multiply(acc, aTail, bTail, mr, nr, t.k - 16, false, 3);
    }
};
} // namespace
__global__ __aicore__ __cube__ void gemmex_fallback_split232_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR cGm, Split101Args232 t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetHF32Mode(HF32Mode::DISABLE);
    SplitShortNnTile<Split101Args232> tile(t);
    tile.Run(a, b, cGm);
}

void gemmex_fallback_split232_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t, uint32_t ldc)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + 367) / 368;
    Split101Args232 q{
        static_cast<uint32_t>(t.m),
        static_cast<uint32_t>(t.n),
        static_cast<uint32_t>(t.k),
        static_cast<uint32_t>(t.lda),
        static_cast<uint32_t>(t.ldb),
        ldc,
        columns,
        columns == 1 ? 0U : static_cast<uint32_t>(((1ULL << 32) + columns - 1) / columns)};
    gemmex_fallback_split232_kernel<<<blocks, nullptr, stream>>>(a, b, c, q);
}

using namespace AscendC;
// Controlled no-count variant of the pinned SDK vreducev2 implementation.
// This compaction has a known result length, so no scalar count is consumed.
template <uint32_t COLUMNS>
__global__ __aicore__ void gemmex_fallback_singleload174_kernel(
    GM_ADDR inputGm, GM_ADDR outputGm, uint32_t m, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    gemm_ex::RunShortDrain<COLUMNS, 1>(inputGm, outputGm, m, n);
}
void gemmex_fallback_singleload174_do(uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t)
{
    const uint32_t use =
        (blocks < (static_cast<uint32_t>(t.n) + 15) / 16 ? blocks : (static_cast<uint32_t>(t.n) + 15) / 16);
    const uint32_t cols = (static_cast<uint32_t>(t.m) + 367) / 368;
    if (cols == 1) {
        gemmex_fallback_singleload174_kernel<1><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 2) {
        gemmex_fallback_singleload174_kernel<2><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 3) {
        gemmex_fallback_singleload174_kernel<3><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else if (cols == 4) {
        gemmex_fallback_singleload174_kernel<4><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    } else {
        gemmex_fallback_singleload174_kernel<5><<<use, nullptr, stream>>>(input, output, t.m, t.n);
    }
}

#endif
