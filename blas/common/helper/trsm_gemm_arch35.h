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
 * \file trsm_gemm_arch35.h
 * \brief Shared arch35 AIC cube GEMM + scale helpers for the triangular-solve
 *        family (strsm_blocked_kernel.cpp and strsmbatched_kernel.cpp).
 *
 * Both operators run the same "temp = op(A) * B" cube GEMM and the same
 * "b = alpha * b" SIMT scale; the copies tripped the duplicate-code scan. The
 * shared bodies live here, and each operator keeps its own __global__ entry
 * points (they export distinct kernel symbols) plus its own tiling struct:
 * the GEMM impl is templated on the tiling type, and the caller resolves the
 * leading dimensions the way its own host code expects.
 */

#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"

using namespace AscendC::Te;

constexpr uint16_t PIPE_FLAG = 0;
constexpr int64_t L0A_SIZE = 64 * 1024;
constexpr int64_t L0C_SIZE = 256 * 1024;
constexpr int64_t L1_SIZE = 512 * 1024;
constexpr uint32_t FINAL_ACC = 3;
constexpr uint32_t NON_FINAL_ACC = 2;
constexpr uint64_t FP32_C0 = 8;
constexpr uint64_t FRACTAL = 16;
constexpr uint64_t L0C_C0 = 16;
constexpr uint64_t L1_BUF_NUM = 2;
constexpr uint64_t L1_BUF_MASK = L1_BUF_NUM - 1;
constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / 2;
constexpr uint64_t BASE_K = 8;

// ===================== AIV kernel: scale =====================

constexpr uint32_t TRSM_SCALE_THREAD_NUM = 128;

__simt_vf__ __aicore__ LAUNCH_BOUND(TRSM_SCALE_THREAD_NUM) inline void TrsmScaleSimt(
    uint32_t m, uint32_t colStart, uint32_t colEnd, uint32_t ldb, float alpha, __gm__ float* bGm)
{
    uint64_t ldbU64 = static_cast<uint64_t>(ldb);
    for (uint32_t col = colStart + threadIdx.x; col < colEnd; col += blockDim.x) {
        for (uint32_t row = 0; row < m; ++row) {
            uint64_t offset = static_cast<uint64_t>(col) * ldbU64 + row;
            bGm[offset] = alpha * bGm[offset];
        }
    }
}

// ===================== AIC kernel: Cube GEMM =====================

template <typename TensorGM, typename TensorL1>
__aicore__ inline void TrsmGemmCopyGM2L1(
    TensorGM gmTensor, TensorL1 tensorL1, uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(MakeCoord(off0, off1), MakeShape(dim0, dim1));
    Copy(MakeCopy(CopyGM2L1{}), tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void TrsmGemmLoadL0Chunk(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t curML1, uint64_t curKL1, uint64_t nL0, uint64_t iter1,
    uint64_t& l0PingPong)
{
    using T = float;
    uint64_t kL0Offset = iter1 * BASE_K;
    uint64_t curKL0 = (kL0Offset + BASE_K > curKL1) ? (curKL1 - kL0Offset) : BASE_K;
    uint64_t l0BufId = l0PingPong & 0x1;
    uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
    auto layoutAL0 = MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curML1, curKL0);
    auto tensorAL0 = MakeTensor(MakeMemPtr<Location::L0A, T>(l0Offset), layoutAL0);
    auto tensorBlockAL1 = tensorAL1.Slice(MakeCoord(0, kL0Offset), MakeShape(curML1, curKL0));
    Copy(MakeCopy(CopyL12L0A{}), tensorAL0, tensorBlockAL1);
    auto layoutBL0 = MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKL0, nL0);
    auto tensorBL0 = MakeTensor(MakeMemPtr<Location::L0B, T>(l0Offset), layoutBL0);
    auto tensorBlockBL1 = tensorBL1.Slice(MakeCoord(kL0Offset, 0), MakeShape(curKL0, nL0));
    Copy(MakeCopy(CopyL12L0B{}), tensorBL0, tensorBlockBL1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
    l0PingPong++;
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void TrsmGemmL0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t curML1, uint64_t curKL1, uint64_t nL0, uint64_t kL1Iter,
    uint64_t iter0, uint64_t& l0PingPong)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, BASE_K);
    if (kL0Iter == 0)
        return;

    TrsmGemmLoadL0Chunk(tensorAL1, tensorBL1, curML1, curKL1, nL0, 0, l0PingPong);

    auto layoutL0C = MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(curML1, nL0);
    auto tensorL0C = MakeTensor(MakeMemPtr<Location::L0C, float>(0), layoutL0C);

    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t l0BufIdCur = (l0PingPong - 1) & 0x1;
        uint64_t l0OffsetCur = HALF_L0_SIZE * l0BufIdCur;
        uint64_t kL0Offset = iter1 * BASE_K;
        uint64_t curKL0 = (kL0Offset + BASE_K > curKL1) ? (curKL1 - kL0Offset) : BASE_K;

        if (iter1 + 1 < kL0Iter) {
            TrsmGemmLoadL0Chunk(tensorAL1, tensorBL1, curML1, curKL1, nL0, iter1 + 1, l0PingPong);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufIdCur);
        bool isLastK = (iter0 + 1 == kL1Iter) && (iter1 + 1 == kL0Iter);
        uint8_t unitFlag = isLastK ? FINAL_ACC : NON_FINAL_ACC;
        MmadParams mmadParams{
            static_cast<uint16_t>(curML1), static_cast<uint16_t>(nL0), static_cast<uint16_t>(curKL0), unitFlag,
            (iter0 == 0 && iter1 == 0)};
        auto layoutAL0Cur = MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curML1, curKL0);
        auto tensorAL0Cur = MakeTensor(MakeMemPtr<Location::L0A, T>(l0OffsetCur), layoutAL0Cur);
        auto layoutBL0Cur = MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKL0, nL0);
        auto tensorBL0Cur = MakeTensor(MakeMemPtr<Location::L0B, T>(l0OffsetCur), layoutBL0Cur);
        Mmad(MmadAtom<MmadTraits<MmadOperation>>{}.with(mmadParams), tensorL0C, tensorAL0Cur, tensorBL0Cur);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufIdCur);
    }
}

template <typename TensorGM>
__aicore__ inline void TrsmGemmLoadL1Chunk(
    TensorGM gmLeftTensor, TensorGM gmRightTensor, uint64_t mOff, uint64_t nOff, uint64_t k, uint64_t tileKChunk,
    uint64_t mL0, uint64_t nL0, uint64_t kOff, uint64_t& abL1LoopCnt)
{
    uint64_t curK = Min<uint64_t>(tileKChunk, k - kOff);
    uint64_t l1BufId = abL1LoopCnt & L1_BUF_MASK;
    uint64_t l1OffsetA = l1BufId * (L1_SIZE / L1_BUF_NUM);
    uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, FRACTAL) * RoundUp<uint64_t>(curK, FP32_C0);
    uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(float);
    auto tensorAL1 = MakeTensor(
        MakeMemPtr<Location::L1, float>(l1OffsetA),
        MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(mL0, curK));
    auto tensorBL1 = MakeTensor(
        MakeMemPtr<Location::L1, float>(l1OffsetB),
        MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curK, nL0));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    TrsmGemmCopyGM2L1(gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    TrsmGemmCopyGM2L1(gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    abL1LoopCnt++;
}

template <typename TensorGM>
__aicore__ inline void TrsmGemmProcessTile(
    TensorGM gmLeftTensor, TensorGM gmRightTensor, TensorGM gmTempTensor, uint64_t mOff, uint64_t nOff, uint64_t mL0,
    uint64_t nL0, uint64_t k, uint64_t tileKChunk, uint64_t kL1Iter, uint64_t& l0PingPong, uint64_t& abL1LoopCnt)
{
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);

    TrsmGemmLoadL1Chunk(gmLeftTensor, gmRightTensor, mOff, nOff, k, tileKChunk, mL0, nL0, 0, abL1LoopCnt);
    uint64_t curBufId = (abL1LoopCnt - 1) & L1_BUF_MASK;
    uint64_t curKForL1 = Min<uint64_t>(tileKChunk, k);

    for (uint64_t kOff = 0; kOff < k; kOff += tileKChunk) {
        if (kOff + tileKChunk < k) {
            TrsmGemmLoadL1Chunk(
                gmLeftTensor, gmRightTensor, mOff, nOff, k, tileKChunk, mL0, nL0, kOff + tileKChunk, abL1LoopCnt);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(curBufId);

        uint64_t l1OffsetA = curBufId * (L1_SIZE / L1_BUF_NUM);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, FRACTAL) * RoundUp<uint64_t>(curKForL1, FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(float);
        auto tensorAL1Cur = MakeTensor(
            MakeMemPtr<Location::L1, float>(l1OffsetA),
            MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(mL0, curKForL1));
        auto tensorBL1Cur = MakeTensor(
            MakeMemPtr<Location::L1, float>(l1OffsetB),
            MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKForL1, nL0));

        TrsmGemmL0MmadLoop(tensorAL1Cur, tensorBL1Cur, mL0, curKForL1, nL0, kL1Iter, kOff / tileKChunk, l0PingPong);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(curBufId);

        if (kOff + tileKChunk < k) {
            curBufId = (abL1LoopCnt - 1) & L1_BUF_MASK;
            curKForL1 = Min<uint64_t>(tileKChunk, k - (kOff + tileKChunk));
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(PIPE_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(PIPE_FLAG);

    auto gmBlockC = gmTempTensor.Slice(MakeCoord(mOff, nOff), MakeShape(mL0, nL0));
    auto tensorL0C = MakeTensor(
        MakeMemPtr<Location::L0C, float>(0), MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(mL0, nL0));
    MakeCopy(CopyL0C2GM{}).Call(gmBlockC, tensorL0C, te::l0c_to_gm_params{te::unit_flag_mode::enable_update});

    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
}

__aicore__ inline void TrsmGemmSetFlags()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
}

__aicore__ inline void TrsmGemmWaitFlags()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
}

// temp = op(A) * x. lda/ldb are resolved by the caller (strsm falls back to k/n
// when the tiling carries 0; strsmbatched always sets them).
template <typename TilingData>
__aicore__ inline void TrsmGemmKernelImpl(
    GM_ADDR a, GM_ADDR x, GM_ADDR temp, const TilingData& tiling, uint32_t lda, uint32_t ldb)
{
    const uint32_t m = tiling.m;
    const uint32_t n = tiling.n;
    const uint32_t k = tiling.k;
    // CeilDiv yields T_MAX (not 0) for a zero divisor, so a zero tile parameter
    // would explode totalTiles/kL1Iter; bail out before any loop is set up.
    if (m == 0 || n == 0 || k == 0 || tiling.tileM == 0 || tiling.tileN == 0 || tiling.tileKChunk == 0) {
        return;
    }

    auto gmLeftTensor = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(a) + tiling.aOffset),
        MakeFrameLayout<NDExtLayoutPtn>(m, lda));
    auto gmRightTensor = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(x) + tiling.bOffset),
        MakeFrameLayout<NDExtLayoutPtn>(k, ldb));
    auto gmTempTensor = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(temp)),
        MakeFrameLayout<NDExtLayoutPtn>(m, tiling.tempRowStride));

    const uint64_t tileM = tiling.tileM;
    const uint64_t tileN = tiling.tileN;
    const uint64_t divM = CeilDiv<uint64_t>(m, tileM);
    const uint64_t divN = CeilDiv<uint64_t>(n, tileN);
    const uint64_t totalTiles = divM * divN;
    const uint64_t kL1Iter = CeilDiv<uint64_t>(k, tiling.tileKChunk);

    TrsmGemmSetFlags();
    uint64_t l0PingPong = 0;
    uint64_t abL1LoopCnt = 0;

    for (uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < totalTiles; tileIdx += AscendC::GetBlockNum()) {
        uint64_t coreIdxM = tileIdx / divN;
        uint64_t coreIdxN = tileIdx % divN;
        if (coreIdxM % 2 == 1) {
            coreIdxN = divN - 1 - coreIdxN;
        }
        uint64_t mEnd = Min<uint64_t>((coreIdxM + 1) * tileM, m);
        uint64_t nEnd = Min<uint64_t>((coreIdxN + 1) * tileN, n);
        for (uint64_t mOff = coreIdxM * tileM; mOff < mEnd; mOff += tileM) {
            uint64_t mL0 = Min<uint64_t>(tileM, mEnd - mOff);
            for (uint64_t nOff = coreIdxN * tileN; nOff < nEnd; nOff += tileN) {
                uint64_t nL0 = Min<uint64_t>(tileN, nEnd - nOff);
                TrsmGemmProcessTile(
                    gmLeftTensor, gmRightTensor, gmTempTensor, mOff, nOff, mL0, nL0, k, tiling.tileKChunk, kL1Iter,
                    l0PingPong, abL1LoopCnt);
            }
        }
    }

    TrsmGemmWaitFlags();
}
