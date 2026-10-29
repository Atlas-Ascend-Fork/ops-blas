/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "tensor_api/tensor.h"
#include "cgemm_ex_kernel.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"

namespace cgemm_ex_blaze3 {

#include "cgemm_ex_cube_common.h"
using namespace AscendC;
using namespace AscendC::Te;

__aicore__ inline void DecodeMnTile(
    uint32_t idx, uint32_t divN, uint64_t BM, uint64_t BN, uint64_t m, uint64_t n, uint64_t& mOff, uint64_t& nOff,
    uint64_t& curM, uint64_t& curN)
{
    uint32_t im = idx / divN;
    uint32_t in = idx % divN;
    if ((im & 1U) != 0U) {
        in = divN - 1U - in;
    }
    mOff = static_cast<uint64_t>(im) * BM;
    nOff = static_cast<uint64_t>(in) * BN;
    curM = Min(m - mOff, BM);
    curN = Min(n - nOff, BN);
}

__aicore__ inline void SplitKRange(
    const GemmTilingData& tiling, uint32_t blk, uint32_t blkNum, uint32_t mnTotal, uint32_t& mnIdx, uint32_t& kPart,
    uint64_t& kBase, uint64_t& kLen, bool& oneTile)
{
    uint32_t kParts = static_cast<uint32_t>(tiling.mBlocks);
    if (kParts < 1) {
        kParts = 1;
    }
    uint64_t kFull = static_cast<uint64_t>(tiling.k);
    if (kParts <= 1) {
        mnIdx = blk;
        kPart = 0;
        kBase = 0;
        kLen = kFull;
        oneTile = false;
        return;
    }
    kPart = blk % kParts;
    mnIdx = blk / kParts;
    oneTile = true;
    uint64_t kChunk = (kFull + static_cast<uint64_t>(kParts) - 1) / static_cast<uint64_t>(kParts);
    kBase = static_cast<uint64_t>(kPart) * kChunk;
    if (kBase >= kFull || mnIdx >= mnTotal) {
        kLen = 0;
        return;
    }
    kLen = kFull - kBase;
    if (kLen > kChunk) {
        kLen = kChunk;
    }
    (void)blkNum;
}

template <bool IS_TRANS_A, bool IS_TRANS_B, uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename ShT>
__aicore__ inline void ProcessAllTiles(
    ShT& sh, uint32_t blk, uint32_t blkNum, uint32_t total, uint32_t divN, uint64_t m, uint64_t n, uint64_t kLen,
    uint64_t kBase, bool oneTile, uint32_t mnIdx)
{
    uint64_t l0Ping = 0;
    if (oneTile) {
        uint64_t mOff = 0;
        uint64_t nOff = 0;
        uint64_t curM = 0;
        uint64_t curN = 0;
        DecodeMnTile(mnIdx, divN, BM, BN, m, n, mOff, nOff, curM, curN);
        ProcessTile<BM, BN, BK, KL1>(sh, static_cast<int32_t>(kLen), kBase, mOff, nOff, curM, curN, l0Ping, true, true);
        return;
    }
    for (uint32_t idx = blk; idx < total; idx += blkNum) {
        uint64_t mOff = 0;
        uint64_t nOff = 0;
        uint64_t curM = 0;
        uint64_t curN = 0;
        DecodeMnTile(idx, divN, BM, BN, m, n, mOff, nOff, curM, curN);
        ProcessTile<BM, BN, BK, KL1>(sh, static_cast<int32_t>(kLen), kBase, mOff, nOff, curM, curN, l0Ping, true, true);
    }
}

template <bool IS_TRANS_A, bool IS_TRANS_B, uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void ProcessAllRun(
    __gm__ float* a, __gm__ float* b, __gm__ float* cPtr, uint64_t m, uint64_t n, uint64_t k, uint64_t lda,
    uint64_t ldb, uint64_t ldc, uint32_t blk, uint32_t blkNum, uint32_t total, uint32_t divN, uint64_t kLen,
    uint64_t kBase, bool oneTile, uint32_t mnIdx)
{
    auto copyGM2L1 = MakeCopy(CopyGM2L1{}, CopyGM2L1TraitDefault{});
    auto copyL12L0A = MakeCopy(CopyL12L0A{}, CopyL12L0ATraitDefault{});
    auto copyL12L0B = MakeCopy(CopyL12L0B{}, CopyL12L0BTraitDefault{});
    auto copyL0C2GM = MakeCopy(CopyL0C2GM{}, CopyL0C2GMTraitDefault{});
    auto mmad = MakeMmad(MmadOperation{}, MmadTraitDefault{});
    auto gmB = MakeGmB<IS_TRANS_A>(a, m, lda, k);
    auto gmA = MakeGmA<IS_TRANS_B>(b, k, ldb, n);
    auto gmC = MakeTensor(MakeMemPtr<Location::GM>(cPtr), MakeFrameLayout<NDExtLayoutPtn>(m, ldc));
    SharedRes<
        decltype(copyGM2L1), decltype(copyL12L0A), decltype(copyL12L0B), decltype(copyL0C2GM), decltype(mmad),
        decltype(gmB), decltype(gmA), decltype(gmC)>
        sh{copyGM2L1, copyL12L0A, copyL12L0B, copyL0C2GM, mmad, gmB, gmA, gmC};
    ProcessAllTiles<IS_TRANS_A, IS_TRANS_B, BM, BN, BK, KL1>(
        sh, blk, blkNum, total, divN, m, n, kLen, kBase, oneTile, mnIdx);
}

template <bool IS_TRANS_A, bool IS_TRANS_B, uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void ProcessAll(__gm__ float* a, __gm__ float* b, __gm__ float* c, const GemmTilingData& tiling)
{
    MnkGrid g;
    if (!FillMnkGrid<BM, BN>(tiling, g)) {
        return;
    }
    uint32_t total = g.divM * g.divN;
    uint32_t mnIdx = 0;
    uint32_t kPart = 0;
    uint64_t kBase = 0;
    uint64_t kLen = g.k;
    bool oneTile = false;
    SplitKRange(tiling, g.blk, g.blkNum, total, mnIdx, kPart, kBase, kLen, oneTile);
    if (kLen == 0) {
        return;
    }
    uint64_t ldc = static_cast<uint64_t>(tiling.ldc);
    __gm__ float* cPtr = (kPart > 0) ? (c + static_cast<uint64_t>(kPart) * ldc * g.m) : c;
    ProcessAllRun<IS_TRANS_A, IS_TRANS_B, BM, BN, BK, KL1>(
        a, b, cPtr, g.m, g.n, g.k, static_cast<uint64_t>(tiling.lda), static_cast<uint64_t>(tiling.ldb), ldc, g.blk,
        g.blkNum, total, g.divN, kLen, kBase, oneTile, mnIdx);
}

struct ShortTileOff {
    uint64_t mOff;
    uint64_t nOff;
};

__aicore__ inline void WaitFixBuf(uint64_t id)
{
    if (id == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    }
}

__aicore__ inline void SetFixBuf(uint64_t id)
{
    if (id == 0) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7,
    typename T8, typename T9>
struct PingPongBuf {
    T0 l0C0;
    T1 l0C1;
    T2 l0A0;
    T3 l0A1;
    T4 l0B0;
    T5 l0B1;
    T6 l1A0;
    T7 l1A1;
    T8 l1B0;
    T9 l1B1;
};

template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline auto MakePingPongBuf()
{
    constexpr uint64_t kL0CPing = 32 * 1024;
    auto l0C0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<L0C_C0>>(BM, BN));
    auto l0C1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(kL0CPing),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<L0C_C0>>(BM, BN));
    auto l0A0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, float>(0),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(BM, BK));
    auto l0A1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, float>(L0_HALF_SIZE),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(BM, BK));
    auto l0B0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, float>(0),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(BK, BN));
    auto l0B1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, float>(L0_HALF_SIZE),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(BK, BN));
    auto l1A0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1AOff(0)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(BM, KL1));
    auto l1A1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1AOff(1)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(BM, KL1));
    auto l1B0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1BOff<BM, KL1>(0)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(KL1, BN));
    auto l1B1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1BOff<BM, KL1>(1)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(KL1, BN));
    return PingPongBuf<decltype(l0C0), decltype(l0C1), decltype(l0A0), decltype(l0A1), decltype(l0B0), decltype(l0B1),
        decltype(l1A0), decltype(l1A1), decltype(l1B0), decltype(l1B1)>{
        l0C0, l0C1, l0A0, l0A1, l0B0, l0B1, l1A0, l1A1, l1B0, l1B1};
}

template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename ShT, typename Buf>
__aicore__ inline void PingPongGemmStep(
    Buf& buf, ShT** shs, const ShortTileOff* tiles, int g, int nGemm, bool flagPerTile)
{
    uint64_t slot = static_cast<uint64_t>(g & 1);
    int tile = g / 3;
    if (g + 1 < nGemm) {
        int ntile = (g + 1) / 3;
        uint64_t nslot = static_cast<uint64_t>((g + 1) & 1);
        IssueGm2L1<BM, BN, BK, KL1>(
            *shs[(g + 1) % 3], buf.l1A0, buf.l1A1, buf.l1B0, buf.l1B1, tiles[ntile].mOff, tiles[ntile].nOff, BM, BN,
            nslot, 0, KL1);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
    WaitFixBuf(slot);
    auto& l0A = (slot == 0) ? buf.l0A0 : buf.l0A1;
    auto& l0B = (slot == 0) ? buf.l0B0 : buf.l0B1;
    auto& l0C = (slot == 0) ? buf.l0C0 : buf.l0C1;
    AscendC::Te::MmadParams mp{
        static_cast<uint16_t>(BM), static_cast<uint16_t>(BN), static_cast<uint16_t>(BK), UNIT_FINAL, true};
    AscendC::Te::Mmad(shs[g % 3]->mmad.with(mp), l0C, l0A, l0B);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slot);
    if (g + 1 < nGemm) {
        uint64_t nslot = static_cast<uint64_t>((g + 1) & 1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(nslot);
        auto& l1A = (nslot == 0) ? buf.l1A0 : buf.l1A1;
        auto& l1B = (nslot == 0) ? buf.l1B0 : buf.l1B1;
        IssueL12L0<BM, BN, BK, KL1>(
            *shs[(g + 1) % 3], buf.l0A0, buf.l0A1, buf.l0B0, buf.l0B1, l1A, l1B, BM, BN, KL1, BK, 0, nslot, nslot);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(nslot);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::Te::FixpipeParams fp{UNIT_FINAL};
    AscendC::Te::Copy(
        shs[g % 3]->copyL0C2GM.with(fp),
        shs[g % 3]->gmC.Slice(
            AscendC::Te::MakeCoord(tiles[tile].mOff, tiles[tile].nOff), AscendC::Te::MakeShape(BM, BN)),
        l0C);
    SetFixBuf(slot);
    if (flagPerTile && (g % 3) == 2) {
        CrossCoreSetFlag<2, PIPE_FIX>(8);
    }
}

// k=64 tiles use two L0C buffers so fixpipe of gemm g overlaps Mmad of gemm g+1.
// The next GM→L1/L1→L0 is issued while gemm g's Mmad is in flight. Per-tile
// PIPE_FIX is still signaled after that tile's third fixpipe.
template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename ShT>
__aicore__ inline void ProcessAll3TilesPingPong(
    ShT& sh1, ShT& sh2, ShT& sh3, const ShortTileOff* tiles, int nt, bool flagPerTile)
{
    auto buf = MakePingPongBuf<BM, BN, BK, KL1>();
    SetFixBuf(1);
    ShT* shs[3] = {&sh1, &sh2, &sh3};
    int nGemm = nt * 3;
    IssueGm2L1<BM, BN, BK, KL1>(
        *shs[0], buf.l1A0, buf.l1A1, buf.l1B0, buf.l1B1, tiles[0].mOff, tiles[0].nOff, BM, BN, 0, 0, KL1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    IssueL12L0<BM, BN, BK, KL1>(
        *shs[0], buf.l0A0, buf.l0A1, buf.l0B0, buf.l0B1, buf.l1A0, buf.l1B0, BM, BN, KL1, BK, 0, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    for (int g = 0; g < nGemm; ++g) {
        PingPongGemmStep<BM, BN, BK, KL1>(buf, shs, tiles, g, nGemm, flagPerTile);
    }
}

template <bool IS_TRANS_A, bool IS_TRANS_B, uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename ShT>
__aicore__ inline void ProcessAll3Tiles(
    ShT& sh1, ShT& sh2, ShT& sh3, uint32_t blk, uint32_t blkNum, uint32_t total, uint32_t divN, uint64_t m, uint64_t n,
    int32_t kLen, bool flagPerTile)
{
    if constexpr (BM == 64 && BN == 64 && BK == 64 && KL1 == 64) {
        if (kLen == static_cast<int32_t>(KL1)) {
            constexpr int kMaxTiles = 16;
            ShortTileOff tiles[kMaxTiles];
            int nt = 0;
            bool full = true;
            for (uint32_t idx = blk; idx < total; idx += blkNum) {
                uint64_t mOff = 0;
                uint64_t nOff = 0;
                uint64_t curM = 0;
                uint64_t curN = 0;
                DecodeMnTile(idx, divN, BM, BN, m, n, mOff, nOff, curM, curN);
                if (curM != BM || curN != BN || nt >= kMaxTiles) {
                    full = false;
                    break;
                }
                tiles[nt].mOff = mOff;
                tiles[nt].nOff = nOff;
                ++nt;
            }
            if (full) {
                if (nt > 0) {
                    ProcessAll3TilesPingPong<BM, BN, BK, KL1>(sh1, sh2, sh3, tiles, nt, flagPerTile);
                }
                return;
            }
        }
    }
    uint64_t l0Ping = 0;
    for (uint32_t idx = blk; idx < total; idx += blkNum) {
        uint64_t mOff = 0;
        uint64_t nOff = 0;
        uint64_t curM = 0;
        uint64_t curN = 0;
        DecodeMnTile(idx, divN, BM, BN, m, n, mOff, nOff, curM, curN);
        ProcessTile<BM, BN, BK, KL1>(sh1, kLen, 0, mOff, nOff, curM, curN, l0Ping, true, true);
        ProcessTile<BM, BN, BK, KL1>(sh2, kLen, 0, mOff, nOff, curM, curN, l0Ping, true, true);
        ProcessTile<BM, BN, BK, KL1>(sh3, kLen, 0, mOff, nOff, curM, curN, l0Ping, true, true);
        if (flagPerTile) {
            CrossCoreSetFlag<2, PIPE_FIX>(8);
        }
    }
}

template <bool IS_TRANS_A, bool IS_TRANS_B, uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void ProcessAll3Run(
    __gm__ float* a1, __gm__ float* b1, __gm__ float* c1, __gm__ float* a2, __gm__ float* b2, __gm__ float* c2,
    __gm__ float* a3, __gm__ float* b3, __gm__ float* c3, uint64_t m, uint64_t n, uint64_t k, uint64_t lda,
    uint64_t ldb, uint64_t ldc, uint32_t blk, uint32_t blkNum, uint32_t divM, uint32_t divN, bool flagPerTile)
{
    auto copyGM2L1 = MakeCopy(CopyGM2L1{}, CopyGM2L1TraitDefault{});
    auto copyL12L0A = MakeCopy(CopyL12L0A{}, CopyL12L0ATraitDefault{});
    auto copyL12L0B = MakeCopy(CopyL12L0B{}, CopyL12L0BTraitDefault{});
    auto copyL0C2GM = MakeCopy(CopyL0C2GM{}, CopyL0C2GMTraitDefault{});
    auto mmad = MakeMmad(MmadOperation{}, MmadTraitDefault{});
    auto gmB1 = MakeGmB<IS_TRANS_A>(a1, m, lda, k);
    auto gmA1 = MakeGmA<IS_TRANS_B>(b1, k, ldb, n);
    auto gmC1 = MakeTensor(MakeMemPtr<Location::GM>(c1), MakeFrameLayout<NDExtLayoutPtn>(m, ldc));
    auto gmB2 = MakeGmB<IS_TRANS_A>(a2, m, lda, k);
    auto gmA2 = MakeGmA<IS_TRANS_B>(b2, k, ldb, n);
    auto gmC2 = MakeTensor(MakeMemPtr<Location::GM>(c2), MakeFrameLayout<NDExtLayoutPtn>(m, ldc));
    auto gmB3 = MakeGmB<IS_TRANS_A>(a3, m, lda, k);
    auto gmA3 = MakeGmA<IS_TRANS_B>(b3, k, ldb, n);
    auto gmC3 = MakeTensor(MakeMemPtr<Location::GM>(c3), MakeFrameLayout<NDExtLayoutPtn>(m, ldc));
    using ShT = SharedRes<
        decltype(copyGM2L1), decltype(copyL12L0A), decltype(copyL12L0B), decltype(copyL0C2GM), decltype(mmad),
        decltype(gmB1), decltype(gmA1), decltype(gmC1)>;
    ShT sh1{copyGM2L1, copyL12L0A, copyL12L0B, copyL0C2GM, mmad, gmB1, gmA1, gmC1};
    ShT sh2{copyGM2L1, copyL12L0A, copyL12L0B, copyL0C2GM, mmad, gmB2, gmA2, gmC2};
    ShT sh3{copyGM2L1, copyL12L0A, copyL12L0B, copyL0C2GM, mmad, gmB3, gmA3, gmC3};
    ProcessAll3Tiles<IS_TRANS_A, IS_TRANS_B, BM, BN, BK, KL1>(
        sh1, sh2, sh3, blk, blkNum, divM * divN, divN, m, n, static_cast<int32_t>(k), flagPerTile);
}

template <bool IS_TRANS_A, bool IS_TRANS_B, uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void ProcessAll3(
    __gm__ float* a1, __gm__ float* b1, __gm__ float* c1, __gm__ float* a2, __gm__ float* b2, __gm__ float* c2,
    __gm__ float* a3, __gm__ float* b3, __gm__ float* c3, const GemmTilingData& tiling, bool flagPerTile)
{
    MnkGrid g;
    if (!FillMnkGrid<BM, BN>(tiling, g)) {
        return;
    }
    ProcessAll3Run<IS_TRANS_A, IS_TRANS_B, BM, BN, BK, KL1>(
        a1, b1, c1, a2, b2, c2, a3, b3, c3, g.m, g.n, g.k, static_cast<uint64_t>(tiling.lda),
        static_cast<uint64_t>(tiling.ldb), static_cast<uint64_t>(tiling.ldc), g.blk, g.blkNum, g.divM, g.divN,
        flagPerTile);
}

template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void DispatchTrans3(
    __gm__ float* a1, __gm__ float* b1, __gm__ float* c1, __gm__ float* a2, __gm__ float* b2, __gm__ float* c2,
    __gm__ float* a3, __gm__ float* b3, __gm__ float* c3, const GemmTilingData& tiling, bool flagPerTile)
{
    if (tiling.isTransA) {
        if (tiling.isTransB) {
            ProcessAll3<true, true, BM, BN, BK, KL1>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
        } else {
            ProcessAll3<true, false, BM, BN, BK, KL1>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
        }
    } else if (tiling.isTransB) {
        ProcessAll3<false, true, BM, BN, BK, KL1>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
    } else {
        ProcessAll3<false, false, BM, BN, BK, KL1>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
    }
}

__aicore__ inline void CubeThreePerTile(
    __gm__ float* a1, __gm__ float* b1, __gm__ float* c1, __gm__ float* a2, __gm__ float* b2, __gm__ float* c2,
    __gm__ float* a3, __gm__ float* b3, __gm__ float* c3, const GemmTilingData& tiling, bool flagPerTile)
{
    if (tiling.baseM <= 32) {
        DispatchTrans3<32, 32, 32, 256>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
    } else if (tiling.baseM <= 64) {
        if (tiling.k <= 96) {
            DispatchTrans3<64, 64, 64, 64>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
        } else {
            DispatchTrans3<64, 64, 32, 256>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
        }
    } else if (tiling.baseM <= 128) {
        if (tiling.k >= 1024) {
            DispatchTrans3<128, 128, 64, 256>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
        } else {
            DispatchTrans3<128, 128, 64, 64>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
        }
    } else {
        DispatchTrans3<256, 256, 32, 128>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
    }
}

template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void CubeThreeSerial3(
    __gm__ float* a1, __gm__ float* b1, __gm__ float* c1, __gm__ float* a2, __gm__ float* b2, __gm__ float* c2,
    __gm__ float* a3, __gm__ float* b3, __gm__ float* c3, const GemmTilingData& tiling)
{
    DispatchTrans<BM, BN, BK, KL1>(a1, b1, c1, tiling);
    DispatchTrans<BM, BN, BK, KL1>(a2, b2, c2, tiling);
    DispatchTrans<BM, BN, BK, KL1>(a3, b3, c3, tiling);
}

__aicore__ inline void CubeThreeSerial(
    __gm__ float* a1, __gm__ float* b1, __gm__ float* c1, __gm__ float* a2, __gm__ float* b2, __gm__ float* c2,
    __gm__ float* a3, __gm__ float* b3, __gm__ float* c3, const GemmTilingData& tiling)
{
    if (tiling.baseM <= 32) {
        CubeThreeSerial3<32, 32, 32, 256>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling);
    } else if (tiling.baseM <= 64) {
        if (tiling.k <= 96) {
            CubeThreeSerial3<64, 64, 64, 64>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling);
        } else {
            CubeThreeSerial3<64, 64, 32, 256>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling);
        }
    } else if (tiling.baseM <= 128) {
        if (tiling.k >= 1024) {
            CubeThreeSerial3<128, 128, 64, 256>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling);
        } else {
            CubeThreeSerial3<128, 128, 64, 64>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling);
        }
    } else {
        CubeThreeSerial3<256, 256, 32, 128>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling);
    }
}

__aicore__ inline void CubeThree(
    __gm__ float* a1, __gm__ float* b1, __gm__ float* c1, __gm__ float* a2, __gm__ float* b2, __gm__ float* c2,
    __gm__ float* a3, __gm__ float* b3, __gm__ float* c3, const GemmTilingData& tiling, bool flagPerTile = false)
{
    InitCubeFlags();
    if (tiling.mBlocks <= 1) {
        CubeThreePerTile(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling, flagPerTile);
    } else {
        CubeThreeSerial(a1, b1, c1, a2, b2, c2, a3, b3, c3, tiling);
    }
    DrainCubeFlags();
}

struct Cube3Ptr {
    uint64_t a1;
    uint64_t b1;
    uint64_t c1;
    uint64_t a2;
    uint64_t b2;
    uint64_t c2;
    uint64_t a3;
    uint64_t b3;
    uint64_t c3;
};

} // namespace cgemm_ex_blaze3

#ifndef CGEMM_EX_CUBE3_SKIP_ENTRY
extern "C" __global__ __aicore__ __cube__ void cgemm_ex_cube3_kernel(
    __gm__ uint8_t* a1, __gm__ uint8_t* b1, __gm__ uint8_t* c1, __gm__ uint8_t* a2, __gm__ uint8_t* b2,
    __gm__ uint8_t* c2, __gm__ uint8_t* a3, __gm__ uint8_t* b3, __gm__ uint8_t* c3, GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    AscendC::SetMMLayoutTransform(true);
    using namespace cgemm_ex_blaze3;
    CubeThree(
        reinterpret_cast<__gm__ float*>(a1), reinterpret_cast<__gm__ float*>(b1), reinterpret_cast<__gm__ float*>(c1),
        reinterpret_cast<__gm__ float*>(a2), reinterpret_cast<__gm__ float*>(b2), reinterpret_cast<__gm__ float*>(c2),
        reinterpret_cast<__gm__ float*>(a3), reinterpret_cast<__gm__ float*>(b3), reinterpret_cast<__gm__ float*>(c3),
        tiling);
    AscendC::SetMMLayoutTransform(false);
}

void cgemm_ex_cube3_do(
    uint32_t numBlocks, void* stream, GM_ADDR a1, GM_ADDR b1, GM_ADDR c1, GM_ADDR a2, GM_ADDR b2, GM_ADDR c2,
    GM_ADDR a3, GM_ADDR b3, GM_ADDR c3, const GemmTilingData& tilingData)
{
    cgemm_ex_cube3_kernel<<<numBlocks, nullptr, stream>>>(a1, b1, c1, a2, b2, c2, a3, b3, c3, tilingData);
}

#endif // CGEMM_EX_CUBE3_SKIP_ENTRY

#endif
