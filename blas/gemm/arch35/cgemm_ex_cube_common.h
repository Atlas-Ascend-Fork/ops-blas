/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/** Shared cube helpers; include inside blaze namespace. */

constexpr uint64_t L1_BUF_NUM = 2;
constexpr uint64_t L1_SIZE = 512 * 1024;
constexpr uint64_t L0_HALF_SIZE = 32 * 1024;
constexpr uint64_t FP32_C0 = 8;
constexpr uint64_t L0C_C0 = 16;
constexpr uint8_t UNIT_NON_FINAL = 2;
constexpr uint8_t UNIT_FINAL = 3;

template <
    typename CopyGM2L1T, typename CopyL12L0AT, typename CopyL12L0BT, typename CopyL0C2GMT, typename MmadT,
    typename GmBT, typename GmAT, typename GmCT>
struct SharedRes {
    CopyGM2L1T copyGM2L1;
    CopyL12L0AT copyL12L0A;
    CopyL12L0BT copyL12L0B;
    CopyL0C2GMT copyL0C2GM;
    MmadT mmad;
    GmBT gmB;
    GmAT gmA;
    GmCT gmC;
};

template <bool IS_TRANS>
__aicore__ inline auto MakeGmB(__gm__ float* ptr, uint64_t m, uint64_t lda, uint64_t k)
{
    if constexpr (IS_TRANS) {
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(ptr),
            AscendC::Te::MakeFrameLayout<AscendC::Te::DNExtLayoutPtn>(lda, k));
    } else {
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(ptr),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(m, lda));
    }
}

template <bool IS_TRANS>
__aicore__ inline auto MakeGmA(__gm__ float* ptr, uint64_t k, uint64_t ldb, uint64_t n)
{
    if constexpr (IS_TRANS) {
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(ptr),
            AscendC::Te::MakeFrameLayout<AscendC::Te::DNExtLayoutPtn>(ldb, n));
    } else {
        return AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(ptr),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(k, ldb));
    }
}

__aicore__ inline uint64_t L1AOff(uint64_t slot) { return slot * (L1_SIZE / L1_BUF_NUM); }

template <uint64_t BM, uint64_t KL1>
__aicore__ inline uint64_t L1BOff(uint64_t slot)
{
    return L1AOff(slot) + BM * KL1 * sizeof(float);
}

template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename SharedT, typename L1AT, typename L1BT>
__aicore__ inline void IssueGm2L1(
    const SharedT& sh, L1AT& l1A0, L1AT& l1A1, L1BT& l1B0, L1BT& l1B1, uint64_t mOff, uint64_t nOff, uint64_t curM,
    uint64_t curN, uint64_t slot, uint64_t kOff, uint64_t curKL1)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
    auto& l1A = (slot == 0) ? l1A0 : l1A1;
    auto& l1B = (slot == 0) ? l1B0 : l1B1;
    if (curKL1 == KL1) {
        AscendC::Te::Copy(
            sh.copyGM2L1, l1A, sh.gmB.Slice(AscendC::Te::MakeCoord(mOff, kOff), AscendC::Te::MakeShape(curM, curKL1)));
        AscendC::Te::Copy(
            sh.copyGM2L1, l1B, sh.gmA.Slice(AscendC::Te::MakeCoord(kOff, nOff), AscendC::Te::MakeShape(curKL1, curN)));
    } else {
        auto l1At = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1AOff(slot)),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(curM, curKL1));
        auto l1Bt = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1BOff<BM, KL1>(slot)),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(curKL1, curN));
        AscendC::Te::Copy(
            sh.copyGM2L1, l1At, sh.gmB.Slice(AscendC::Te::MakeCoord(mOff, kOff), AscendC::Te::MakeShape(curM, curKL1)));
        AscendC::Te::Copy(
            sh.copyGM2L1, l1Bt, sh.gmA.Slice(AscendC::Te::MakeCoord(kOff, nOff), AscendC::Te::MakeShape(curKL1, curN)));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
}

template <
    uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename SharedT, typename L0AT, typename L0BT, typename L1AT,
    typename L1BT>
__aicore__ inline void IssueL12L0(
    const SharedT& sh, L0AT& l0A0, L0AT& l0A1, L0BT& l0B0, L0BT& l0B1, L1AT& l1A, L1BT& l1B, uint64_t curM,
    uint64_t curN, uint64_t curKL1, uint64_t curKL0, uint64_t kL0Off, uint64_t slot, uint64_t l0Id)
{
    uint64_t l0Off = l0Id * L0_HALF_SIZE;
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Id);
    auto& l0A = (l0Id == 0) ? l0A0 : l0A1;
    auto& l0B = (l0Id == 0) ? l0B0 : l0B1;
    if (curKL1 == KL1 && curKL0 == BK) {
        AscendC::Te::Copy(
            sh.copyL12L0A, l0A, l1A.Slice(AscendC::Te::MakeCoord(0, kL0Off), AscendC::Te::MakeShape(curM, BK)));
        AscendC::Te::Copy(
            sh.copyL12L0B, l0B, l1B.Slice(AscendC::Te::MakeCoord(kL0Off, 0), AscendC::Te::MakeShape(BK, curN)));
    } else {
        auto l1As = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1AOff(slot)),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(curM, curKL1));
        auto l1Bs = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1BOff<BM, KL1>(slot)),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(curKL1, curN));
        auto l0At = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, float>(l0Off),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(curM, curKL0));
        auto l0Bt = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, float>(l0Off),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(curKL0, curN));
        AscendC::Te::Copy(
            sh.copyL12L0A, l0At, l1As.Slice(AscendC::Te::MakeCoord(0, kL0Off), AscendC::Te::MakeShape(curM, curKL0)));
        AscendC::Te::Copy(
            sh.copyL12L0B, l0Bt, l1Bs.Slice(AscendC::Te::MakeCoord(kL0Off, 0), AscendC::Te::MakeShape(curKL0, curN)));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Id);
}

template <typename SharedT, typename L0CT>
__aicore__ inline void WriteL0C(
    const SharedT& sh, L0CT& l0C, uint64_t mOff, uint64_t nOff, uint64_t curM, uint64_t curN)
{
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::Te::FixpipeParams fp{UNIT_FINAL};
    AscendC::Te::Copy(
        sh.copyL0C2GM.with(fp), sh.gmC.Slice(AscendC::Te::MakeCoord(mOff, nOff), AscendC::Te::MakeShape(curM, curN)),
        l0C);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
}

template <
    uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename SharedT, typename L0AT, typename L0BT, typename L1AT,
    typename L1BT, typename L0CT>
__aicore__ inline void MmadOneKL0(
    const SharedT& sh, uint64_t curM, uint64_t curN, uint64_t curKL1, uint64_t ik, uint64_t jk, uint64_t kIter,
    uint64_t kL0Iter, uint64_t slot, uint64_t& l0Ping, bool clearC, bool writeOut, L0AT& l0A0, L0AT& l0A1, L0BT& l0B0,
    L0BT& l0B1, L1AT& l1A, L1BT& l1B, L0CT& l0C)
{
    uint64_t kL0Off = jk * BK;
    uint64_t curKL0 = Min(curKL1 - kL0Off, BK);
    uint64_t l0Id = l0Ping & 1;
    uint64_t l0Off = l0Id * L0_HALF_SIZE;
    if (jk + 1 < kL0Iter) {
        uint64_t nId = (l0Ping + 1) & 1;
        uint64_t nOffK = (jk + 1) * BK;
        uint64_t nKL0 = Min(curKL1 - nOffK, BK);
        IssueL12L0<BM, BN, BK, KL1>(sh, l0A0, l0A1, l0B0, l0B1, l1A, l1B, curM, curN, curKL1, nKL0, nOffK, slot, nId);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Id);
    auto& l0A = (l0Id == 0) ? l0A0 : l0A1;
    auto& l0B = (l0Id == 0) ? l0B0 : l0B1;
    bool first = clearC && (ik == 0 && jk == 0);
    bool last = writeOut && (ik + 1 == kIter && jk + 1 == kL0Iter);
    uint8_t uf = last ? UNIT_FINAL : UNIT_NON_FINAL;
    AscendC::Te::MmadParams mp{
        static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(curKL0), uf, first};
    if (curKL1 == KL1 && curKL0 == BK) {
        AscendC::Te::Mmad(sh.mmad.with(mp), l0C, l0A, l0B);
    } else {
        auto l0At = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, float>(l0Off), AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(curM, curKL0));
        auto l0Bt = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, float>(l0Off), AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(curKL0, curN));
        AscendC::Te::Mmad(sh.mmad.with(mp), l0C, l0At, l0Bt);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Id);
    l0Ping++;
}

template <
    uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename SharedT, typename L0AT, typename L0BT, typename L1AT,
    typename L1BT, typename L0CT>
__aicore__ inline void RunOneKL1(
    const SharedT& sh, int32_t kLen, uint64_t kBase, uint64_t mOff, uint64_t nOff, uint64_t curM, uint64_t curN,
    uint64_t ik, uint64_t kIter, uint64_t& l0Ping, bool clearC, bool writeOut, L0AT& l0A0, L0AT& l0A1, L0BT& l0B0,
    L0BT& l0B1, L1AT& l1A0, L1AT& l1A1, L1BT& l1B0, L1BT& l1B1, L0CT& l0C)
{
    uint64_t slot = ik % L1_BUF_NUM;
    uint64_t kOff = kBase + ik * KL1;
    uint64_t curKL1 = Min(static_cast<uint64_t>(kLen) - ik * KL1, KL1);
    IssueGm2L1<BM, BN, BK, KL1>(sh, l1A0, l1A1, l1B0, l1B1, mOff, nOff, curM, curN, slot, kOff, curKL1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
    auto& l1A = (slot == 0) ? l1A0 : l1A1;
    auto& l1B = (slot == 0) ? l1B0 : l1B1;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, BK);
    if (kL0Iter > 0) {
        uint64_t l0Id = l0Ping & 1;
        uint64_t curKL0 = Min(curKL1, BK);
        IssueL12L0<BM, BN, BK, KL1>(sh, l0A0, l0A1, l0B0, l0B1, l1A, l1B, curM, curN, curKL1, curKL0, 0, slot, l0Id);
    }
    // L0C is idle during GM→L1/L0. Wait for the previous fixpipe only once the next Mmad needs L0C,
    // so that fixpipe overlaps the next tile's MTE2.
    if (clearC && ik == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }
    for (uint64_t jk = 0; jk < kL0Iter; ++jk) {
        MmadOneKL0<BM, BN, BK, KL1>(
            sh, curM, curN, curKL1, ik, jk, kIter, kL0Iter, slot, l0Ping, clearC, writeOut, l0A0, l0A1, l0B0, l0B1, l1A,
            l1B, l0C);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
}

template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename SharedT>
__aicore__ inline void ProcessTile(
    const SharedT& sh, int32_t kLen, uint64_t kBase, uint64_t mOff, uint64_t nOff, uint64_t curM, uint64_t curN,
    uint64_t& l0Ping, bool clearC, bool writeOut)
{
    auto l0C = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<L0C_C0>>(curM, curN));
    auto l0A0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, float>(0),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(curM, BK));
    auto l0A1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, float>(L0_HALF_SIZE),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(curM, BK));
    auto l0B0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, float>(0),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(BK, curN));
    auto l0B1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, float>(L0_HALF_SIZE),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(BK, curN));
    auto l1A0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1AOff(0)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(curM, KL1));
    auto l1A1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1AOff(1)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::Int<FP32_C0>>(curM, KL1));
    auto l1B0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1BOff<BM, KL1>(0)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(KL1, curN));
    auto l1B1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, float>(L1BOff<BM, KL1>(1)),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::Int<FP32_C0>>(KL1, curN));

    uint64_t kIter = CeilDiv<uint64_t>(static_cast<uint64_t>(kLen), KL1);
    for (uint64_t ik = 0; ik < kIter; ++ik) {
        RunOneKL1<BM, BN, BK, KL1>(
            sh, kLen, kBase, mOff, nOff, curM, curN, ik, kIter, l0Ping, clearC, writeOut, l0A0, l0A1, l0B0, l0B1, l1A0,
            l1A1, l1B0, l1B1, l0C);
    }
    if (writeOut) {
        WriteL0C(sh, l0C, mOff, nOff, curM, curN);
    }
}

// ProcessAll is defined after this include in each TU; declare for two-phase lookup.
template <bool IS_TRANS_A, bool IS_TRANS_B, uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void ProcessAll(__gm__ float* a, __gm__ float* b, __gm__ float* c, const GemmTilingData& tiling);

template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void DispatchTrans(__gm__ float* a, __gm__ float* b, __gm__ float* c, const GemmTilingData& tiling)
{
    if (tiling.isTransA) {
        if (tiling.isTransB) {
            ProcessAll<true, true, BM, BN, BK, KL1>(a, b, c, tiling);
        } else {
            ProcessAll<true, false, BM, BN, BK, KL1>(a, b, c, tiling);
        }
    } else if (tiling.isTransB) {
        ProcessAll<false, true, BM, BN, BK, KL1>(a, b, c, tiling);
    } else {
        ProcessAll<false, false, BM, BN, BK, KL1>(a, b, c, tiling);
    }
}

struct MnkGrid {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    uint32_t divM = 0;
    uint32_t divN = 0;
    uint32_t blk = 0;
    uint32_t blkNum = 0;
};

template <uint64_t BM, uint64_t BN>
__aicore__ inline bool FillMnkGrid(const GemmTilingData& tiling, MnkGrid& g)
{
    g.m = static_cast<uint64_t>(tiling.m);
    g.n = static_cast<uint64_t>(tiling.n);
    g.k = static_cast<uint64_t>(tiling.k);
    g.divM = static_cast<uint32_t>(CeilDiv<uint64_t>(g.m, BM));
    g.divN = static_cast<uint32_t>(CeilDiv<uint64_t>(g.n, BN));
    uint32_t used = static_cast<uint32_t>(tiling.usedCoreNum);
    g.blkNum = used < 1 ? AscendC::GetBlockNum() : used;
    g.blk = AscendC::GetBlockIdx();
    return g.divN != 0 && g.blkNum != 0 && g.blk < g.blkNum;
}

__aicore__ inline void InitCubeFlags()
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    for (uint32_t i = 0; i < L1_BUF_NUM; ++i) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
}

__aicore__ inline void DrainCubeFlags()
{
    for (uint32_t i = 0; i < L1_BUF_NUM; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
}
