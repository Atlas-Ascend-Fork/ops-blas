/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ssyrk_kernel.h"

#include "common/helper/complex_blas3_arch22.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t SSYRK_COMBINE_CHUNK = 4096U;
constexpr uint32_t SSYRK_BYTES_PER_BLOCK = 32U;

struct SsyrkGemmTile {
    uint32_t rowBase;
    uint32_t colBase;
    uint32_t rowCount;
    uint32_t colCount;
};

__aicore__ inline SsyrkGemmTile ResolveSsyrkTile(const SsyrkGemmTilingData& t, uint32_t tile)
{
    SsyrkGemmTile r{};
    const uint32_t mb = tile / t.nBlocks;
    const uint32_t nb = tile % t.nBlocks;
    r.rowBase = mb * SSYRK_TILE_M;
    r.colBase = nb * SSYRK_TILE_N;
    r.rowCount = (r.rowBase + SSYRK_TILE_M > t.n) ? (t.n - r.rowBase) : SSYRK_TILE_M;
    r.colCount = (r.colBase + SSYRK_TILE_N > t.n) ? (t.n - r.colBase) : SSYRK_TILE_N;
    return r;
}

__aicore__ inline bool IsSsyrkTileNeeded(const SsyrkGemmTile& r, uint32_t uploMode)
{
    if (uploMode == SSYRK_UPLO_UPPER) {
        return r.colBase <= (r.rowBase + r.rowCount - 1U);
    }
    return (r.colBase + r.colCount - 1U) >= r.rowBase;
}

template <class MM, bool IS_TRANS>
__aicore__ inline void SetupSsyrkGemmTile(
    MM& mm, const SsyrkGemmTile& r, const SsyrkGemmTilingData& t, __gm__ float* aBase, __gm__ float* tempBase)
{
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> bGlobal;
    GlobalTensor<float> cGlobal;
    if constexpr (!IS_TRANS) {
        const uint64_t aLen = static_cast<uint64_t>(t.kCount - 1U) * t.lda + r.rowCount;
        const uint64_t bLen = static_cast<uint64_t>(t.kCount - 1U) * t.lda + r.colCount;
        aGlobal.SetGlobalBuffer(aBase + static_cast<uint64_t>(t.kBase) * t.lda + r.rowBase, aLen);
        bGlobal.SetGlobalBuffer(aBase + static_cast<uint64_t>(t.kBase) * t.lda + r.colBase, bLen);
    } else {
        const uint64_t aLen = static_cast<uint64_t>(r.rowCount - 1U) * t.lda + t.kCount;
        const uint64_t bLen = static_cast<uint64_t>(r.colCount - 1U) * t.lda + t.kCount;
        aGlobal.SetGlobalBuffer(aBase + static_cast<uint64_t>(r.rowBase) * t.lda + t.kBase, aLen);
        bGlobal.SetGlobalBuffer(aBase + static_cast<uint64_t>(r.colBase) * t.lda + t.kBase, bLen);
    }
    cGlobal.SetGlobalBuffer(
        tempBase + static_cast<uint64_t>(r.rowBase) * t.tempLd + r.colBase,
        static_cast<uint64_t>(r.rowCount - 1U) * t.tempLd + r.colCount);

    mm.SetSingleShape(static_cast<int>(r.rowCount), static_cast<int>(r.colCount), static_cast<int>(t.kCount));
    mm.SetTensorA(aGlobal, !IS_TRANS);
    mm.SetTensorB(bGlobal, IS_TRANS);
    mm.IterateAll(cGlobal, static_cast<uint8_t>(t.enAtomic));
}

template <class MM, bool IS_TRANS>
__aicore__ inline void RunSsyrkGemmBody(GM_ADDR aAddr, GM_ADDR tempAddr, const SsyrkGemmTilingData& t)
{
    if (t.nBlocks == 0U || t.mBlocks == 0U || GetBlockNum() == 0) {
        return;
    }
    const uint32_t totalTiles = t.mBlocks * t.nBlocks;
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());

    __gm__ float* aBase = reinterpret_cast<__gm__ float*>(aAddr);
    __gm__ float* tempBase = reinterpret_cast<__gm__ float*>(tempAddr);

    TPipe pipe;
    MM mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    if constexpr (!IS_TRANS) {
        mm.SetOrgShape(
            static_cast<int>(t.lda), static_cast<int>(t.lda), static_cast<int>(t.k), static_cast<int>(t.k),
            static_cast<int>(t.tempLd));
    } else {
        mm.SetOrgShape(
            static_cast<int>(t.n), static_cast<int>(t.n), static_cast<int>(t.lda), static_cast<int>(t.lda),
            static_cast<int>(t.tempLd));
    }

    uint32_t needed = 0U;
    for (uint32_t tile = 0U; tile < totalTiles; ++tile) {
        const SsyrkGemmTile r = ResolveSsyrkTile(t, tile);
        if (!IsSsyrkTileNeeded(r, t.uploMode)) {
            continue;
        }
        if ((needed++ % coreNum) != coreIdx) {
            continue;
        }
        SetupSsyrkGemmTile<MM, IS_TRANS>(mm, r, t, aBase, tempBase);
    }
    mm.End();
}

__aicore__ inline void ProcessCombineChunk(
    const GlobalTensor<float>& tempGm, const GlobalTensor<float>& cGm, uint64_t tempOff, uint64_t cOff,
    uint32_t len, const LocalTensor<float>& tempLoc, const LocalTensor<float>& cLoc,
    const SsyrkCombineTilingData& t)
{
    const uint32_t bytes = len * static_cast<uint32_t>(sizeof(float));
    const DataCopyExtParams copyParams{1U, bytes, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};

    if (t.hasTemp != 0U) {
        DataCopyPad(tempLoc, tempGm[tempOff], copyParams, padParams);
    }
    if (t.beta != 0.0F) {
        DataCopyPad(cLoc, cGm[cOff], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    const uint32_t vecLen = ((bytes + SSYRK_BYTES_PER_BLOCK - 1U) / SSYRK_BYTES_PER_BLOCK) *
                            (SSYRK_BYTES_PER_BLOCK / static_cast<uint32_t>(sizeof(float)));
    if (t.beta == 0.0F && t.hasTemp == 0U) {
        Duplicate(cLoc, 0.0F, vecLen);
    } else if (t.beta == 0.0F) {
        Muls(cLoc, tempLoc, t.alpha, vecLen);
    } else if (t.hasTemp == 0U) {
        Muls(cLoc, cLoc, t.beta, vecLen);
    } else {
        Muls(cLoc, cLoc, t.beta, vecLen);
        PipeBarrier<PIPE_V>();
        Axpy(cLoc, tempLoc, t.alpha, vecLen);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopyPad(cGm[cOff], cLoc, copyParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline void RunSsyrkCombineBody(GM_ADDR tempAddr, GM_ADDR cAddr, const SsyrkCombineTilingData& t)
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());

    GlobalTensor<float> tempGm;
    GlobalTensor<float> cGm;
    tempGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tempAddr));
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cAddr));

    TPipe pipe;
    TBuf<TPosition::VECCALC> tempBuf;
    TBuf<TPosition::VECCALC> cBuf;
    pipe.InitBuffer(tempBuf, SSYRK_COMBINE_CHUNK * sizeof(float));
    pipe.InitBuffer(cBuf, SSYRK_COMBINE_CHUNK * sizeof(float));
    LocalTensor<float> tempLoc = tempBuf.Get<float>();
    LocalTensor<float> cLoc = cBuf.Get<float>();

    for (uint32_t col = coreIdx; col < t.n; col += coreNum) {
        const uint32_t rowStart = (t.uploMode == SSYRK_UPLO_UPPER) ? 0U : col;
        const uint32_t rowCount = (t.uploMode == SSYRK_UPLO_UPPER) ? (col + 1U) : (t.n - col);
        for (uint32_t off = 0U; off < rowCount; off += SSYRK_COMBINE_CHUNK) {
            const uint32_t len = (off + SSYRK_COMBINE_CHUNK > rowCount) ? (rowCount - off) : SSYRK_COMBINE_CHUNK;
            const uint64_t row = static_cast<uint64_t>(rowStart + off);
            const uint64_t tempOff = static_cast<uint64_t>(col) * t.tempLd + row;
            const uint64_t cOff = static_cast<uint64_t>(col) * t.ldc + row;
            ProcessCombineChunk(tempGm, cGm, tempOff, cOff, len, tempLoc, cLoc, t);
        }
    }
}

} // namespace

extern "C" __global__ __aicore__ void ssyrk_gemm_trans_left(
    GM_ADDR aAddr, GM_ADDR tempAddr, SsyrkGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunSsyrkGemmBody<cblas3::MatmulTransLeft, false>(aAddr, tempAddr, tiling);
}

extern "C" __global__ __aicore__ void ssyrk_gemm_trans_right(
    GM_ADDR aAddr, GM_ADDR tempAddr, SsyrkGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    RunSsyrkGemmBody<cblas3::MatmulTransRight, true>(aAddr, tempAddr, tiling);
}

extern "C" __global__ __aicore__ void ssyrk_combine(
    GM_ADDR tempAddr, GM_ADDR cAddr, SsyrkCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunSsyrkCombineBody(tempAddr, cAddr, tiling);
}

void ssyrk_gemm_launch(
    uint32_t blockDim, void* stream, uint8_t* a, uint8_t* temp, const SsyrkGemmTilingData& tiling, bool isTrans)
{
    if (isTrans) {
        ssyrk_gemm_trans_right<<<blockDim, nullptr, stream>>>(a, temp, tiling);
    } else {
        ssyrk_gemm_trans_left<<<blockDim, nullptr, stream>>>(a, temp, tiling);
    }
}

void ssyrk_combine_launch(
    uint32_t blockDim, void* stream, uint8_t* temp, uint8_t* c, const SsyrkCombineTilingData& tiling)
{
    ssyrk_combine<<<blockDim, nullptr, stream>>>(temp, c, tiling);
}
