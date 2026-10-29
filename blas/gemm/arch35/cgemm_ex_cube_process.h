/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/** Default ProcessAll; include after cube_common.h inside the blaze namespace. */

template <uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1, typename ShT>
__aicore__ inline void ProcessAllLoop(
    ShT& sh, uint32_t blk, uint32_t blkNum, uint32_t total, uint32_t divN, uint64_t m, uint64_t n, int32_t k)
{
    uint64_t l0Ping = 0;
    for (uint32_t idx = blk; idx < total; idx += blkNum) {
        uint32_t im = idx / divN;
        uint32_t in = idx % divN;
        if ((im & 1U) != 0U) {
            in = divN - 1U - in;
        }
        uint64_t mOff = static_cast<uint64_t>(im) * BM;
        uint64_t nOff = static_cast<uint64_t>(in) * BN;
        uint64_t curM = Min(m - mOff, BM);
        uint64_t curN = Min(n - nOff, BN);
        ProcessTile<BM, BN, BK, KL1>(sh, k, 0, mOff, nOff, curM, curN, l0Ping, true, true);
    }
}

template <bool IS_TRANS_A, bool IS_TRANS_B, uint64_t BM, uint64_t BN, uint64_t BK, uint64_t KL1>
__aicore__ inline void ProcessAll(__gm__ float* a, __gm__ float* b, __gm__ float* c, const GemmTilingData& tiling)
{
    uint64_t m = static_cast<uint64_t>(tiling.m);
    uint64_t n = static_cast<uint64_t>(tiling.n);
    uint64_t k = static_cast<uint64_t>(tiling.k);
    uint64_t lda = static_cast<uint64_t>(tiling.lda);
    uint64_t ldb = static_cast<uint64_t>(tiling.ldb);
    uint64_t ldc = static_cast<uint64_t>(tiling.ldc);
    auto copyGM2L1 = MakeCopy(CopyGM2L1{}, CopyGM2L1TraitDefault{});
    auto copyL12L0A = MakeCopy(CopyL12L0A{}, CopyL12L0ATraitDefault{});
    auto copyL12L0B = MakeCopy(CopyL12L0B{}, CopyL12L0BTraitDefault{});
    auto copyL0C2GM = MakeCopy(CopyL0C2GM{}, CopyL0C2GMTraitDefault{});
    auto mmad = MakeMmad(MmadOperation{}, MmadTraitDefault{});
    auto gmB = MakeGmB<IS_TRANS_A>(a, m, lda, k);
    auto gmA = MakeGmA<IS_TRANS_B>(b, k, ldb, n);
    auto gmC = MakeTensor(MakeMemPtr<Location::GM>(c), MakeFrameLayout<NDExtLayoutPtn>(m, ldc));
    SharedRes<
        decltype(copyGM2L1), decltype(copyL12L0A), decltype(copyL12L0B), decltype(copyL0C2GM), decltype(mmad),
        decltype(gmB), decltype(gmA), decltype(gmC)>
        sh{copyGM2L1, copyL12L0A, copyL12L0B, copyL0C2GM, mmad, gmB, gmA, gmC};
    uint32_t divM = static_cast<uint32_t>(CeilDiv<uint64_t>(m, BM));
    uint32_t divN = static_cast<uint32_t>(CeilDiv<uint64_t>(n, BN));
    uint32_t blkNum = GetBlockNum();
    if (divN == 0 || blkNum == 0) {
        return;
    }
    ProcessAllLoop<BM, BN, BK, KL1>(sh, GetBlockIdx(), blkNum, divM * divN, divN, m, n, static_cast<int32_t>(k));
}
