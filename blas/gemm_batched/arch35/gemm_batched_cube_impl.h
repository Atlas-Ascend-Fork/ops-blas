/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE file.
 */

#pragma once

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#ifndef KERNEL_UTILS_LITE
#define KERNEL_UTILS_LITE
#endif
#include "common/helper/kernel_utils.h"
#include "gemm_batched_tiling_data.h"

constexpr uint16_t PIPE_FLAG = 0;

constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

// ============================================================================
// Dtype traits
// ============================================================================
template <uint32_t DtypeId>
struct GbDtypeTraits;

template <>
struct GbDtypeTraits<0> {
    using type = float;
    static constexpr uint32_t C0 = GEMM_BATCHED_FP32_C0;
    static constexpr uint32_t elemSize = sizeof(float);
};

template <>
struct GbDtypeTraits<1> {
    using type = half;
    static constexpr uint32_t C0 = GEMM_BATCHED_FP16_C0;
    static constexpr uint32_t elemSize = sizeof(half);
};

template <>
struct GbDtypeTraits<2> {
    using type = bfloat16_t;
    static constexpr uint32_t C0 = GEMM_BATCHED_FP16_C0;
    static constexpr uint32_t elemSize = sizeof(bfloat16_t);
};

// Layout helper for column-major data:
// AscendC::Te::NDExtLayoutPtn(rows, ld): stride=(ld, 1), element (i,j) at i*ld+j → row-major read.
//   Used for transposed access: op(A)[i][j] = A[j][i] at i*ld+j.
// AscendC::Te::DNExtLayoutPtn(ld, cols): stride=(1, ld), element (i,j) at j*ld+i → column-major read.
//   Used for non-transposed access: A[i][j] at j*ld+i.
template <typename DType>
__aicore__ inline auto MakeGmLayout(AscendC::Te::NDExtLayoutPtn, uint32_t rows, uint32_t cols, uint32_t ld)
{
    (void)cols;
    constexpr uint64_t dtypeC0 = GEMM_BATCHED_UB_ALIGN_BYTES / sizeof(DType);
    return AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<dtypeC0>>(rows, ld);
}
template <typename DType>
__aicore__ inline auto MakeGmLayout(AscendC::Te::DNExtLayoutPtn, uint32_t rows, uint32_t cols, uint32_t ld)
{
    (void)rows;
    constexpr uint64_t dtypeC0 = GEMM_BATCHED_UB_ALIGN_BYTES / sizeof(DType);
    return AscendC::Te::MakeFrameLayout<AscendC::Te::DNExtLayoutPtn, AscendC::Std::Int<dtypeC0>>(ld, cols);
}

// ============================================================================
// Kernel 1: GEMM (AIC-only, tensor_api)
// ================================================================================

template <uint32_t DtypeId, typename TensorAL1, typename TensorBL1>
__aicore__ inline void GbLoadL0Chunk(
    const TensorAL1& tensorAL1, const TensorBL1& tensorBL1, uint32_t curTileM, uint32_t curTileN, uint32_t curK,
    uint32_t baseK, uint32_t kStep, uint64_t& l0PingPong)
{
    using DType = typename GbDtypeTraits<DtypeId>::type;
    constexpr uint32_t C0 = GbDtypeTraits<DtypeId>::C0;
    const uint32_t curBaseK = Min<uint32_t>(baseK, curK - kStep * baseK);
    const uint32_t l0Half = l0PingPong & 1;
    const uint32_t l0aOff = l0Half * (AscendC::TOTAL_L0A_SIZE >> 1);
    const uint32_t l0bOff = l0Half * (AscendC::TOTAL_L0B_SIZE >> 1);
    auto tensorAL0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, DType>(l0aOff),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0>>(curTileM, curBaseK));
    auto tensorBL0 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, DType>(l0bOff),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0>>(curBaseK, curTileN));
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Half);
    const uint32_t kL0Offset = kStep * baseK;
    auto tensorBlockAL1 =
        tensorAL1.Slice(AscendC::Te::MakeCoord(0, (long)kL0Offset), AscendC::Te::MakeShape(curTileM, curBaseK));
    auto tensorBlockBL1 =
        tensorBL1.Slice(AscendC::Te::MakeCoord((long)kL0Offset, 0), AscendC::Te::MakeShape(curBaseK, curTileN));
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{}), tensorAL0, tensorBlockAL1);
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{}), tensorBL0, tensorBlockBL1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Half);
    l0PingPong++;
}

template <uint32_t DtypeId, typename TensorAL1, typename TensorBL1>
__aicore__ inline void GbProcessL0Loop(
    const TensorAL1& tensorAL1, const TensorBL1& tensorBL1, uint32_t curTileM, uint32_t curTileN, uint32_t curK,
    uint32_t baseK, uint64_t l1LoopCnt, uint32_t kL1Iter, uint32_t kOff, uint32_t tileKChunk, uint32_t l0cOffsetBytes,
    uint64_t& l0PingPong)
{
    using DType = typename GbDtypeTraits<DtypeId>::type;
    constexpr uint32_t C0 = GbDtypeTraits<DtypeId>::C0;
    const uint32_t kSteps = (curK + baseK - 1) / baseK;
    GbLoadL0Chunk<DtypeId>(tensorAL1, tensorBL1, curTileM, curTileN, curK, baseK, 0, l0PingPong);
    auto tensorL0C = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffsetBytes),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<GEMM_BATCHED_L0C_C0>>(
            curTileM, curTileN));
    for (uint32_t kk = 0; kk < kSteps; ++kk) {
        const uint32_t currentHalf = (l0PingPong - 1) & 1;
        const uint32_t currentBaseK = Min<uint32_t>(baseK, curK - kk * baseK);
        if (kk + 1 < kSteps) {
            GbLoadL0Chunk<DtypeId>(tensorAL1, tensorBL1, curTileM, curTileN, curK, baseK, kk + 1, l0PingPong);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(currentHalf);
        const uint32_t l0aOff = currentHalf * (AscendC::TOTAL_L0A_SIZE >> 1);
        const uint32_t l0bOff = currentHalf * (AscendC::TOTAL_L0B_SIZE >> 1);
        auto tensorAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, DType>(l0aOff),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0>>(curTileM, currentBaseK));
        auto tensorBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, DType>(l0bOff),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0>>(currentBaseK, curTileN));
        const bool isFirst = (l1LoopCnt == 0 && kk == 0);
        const bool isLastK = (kOff / tileKChunk + 1 == kL1Iter) && (kk + 1 == kSteps);
        const uint8_t unitFlag = isLastK ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        AscendC::Te::MmadParams mmapParams{
            static_cast<uint16_t>(curTileM), static_cast<uint16_t>(curTileN), static_cast<uint16_t>(currentBaseK),
            unitFlag, isFirst};
        AscendC::Te::Mmad(
            AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(mmapParams), tensorL0C,
            tensorAL0, tensorBL0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(currentHalf);
    }
}

template <uint32_t DtypeId, typename GmAT, typename GmBT>
__aicore__ inline void GbLoadL1Chunk(
    const GmAT& gmATensor, const GmBT& gmBTensor, uint32_t mOff, uint32_t nOff, uint32_t curTileM, uint32_t curTileN,
    uint32_t K, uint32_t tileKChunk, uint32_t kOff, uint64_t& l1LoopCnt)
{
    using DType = typename GbDtypeTraits<DtypeId>::type;
    constexpr uint32_t C0 = GbDtypeTraits<DtypeId>::C0;
    const uint32_t curK = Min<uint32_t>(tileKChunk, K - kOff);
    const uint32_t l1BufId = l1LoopCnt & GEMM_BATCHED_L1_BUF_MASK;
    const uint32_t l1BaseBytes = l1BufId * (AscendC::TOTAL_L1_SIZE >> 1);
    const uint32_t aL1Elems = RoundUp<uint32_t>(curTileM, GEMM_BATCHED_FRACTAL) * RoundUp<uint32_t>(curK, C0);
    const uint32_t bL1ByteOff = l1BaseBytes + aL1Elems * sizeof(DType);
    auto tensorAL1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, DType>(l1BaseBytes),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0>>(curTileM, curK));
    auto tensorBL1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, DType>(bL1ByteOff),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0>>(curK, curTileN));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    auto gmBlockA =
        gmATensor.Slice(AscendC::Te::MakeCoord((long)mOff, (long)kOff), AscendC::Te::MakeShape(curTileM, curK));
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{}), tensorAL1, gmBlockA);
    auto gmBlockB =
        gmBTensor.Slice(AscendC::Te::MakeCoord((long)kOff, (long)nOff), AscendC::Te::MakeShape(curK, curTileN));
    AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{}), tensorBL1, gmBlockB);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    l1LoopCnt++;
}

template <uint32_t DtypeId, typename GmAT, typename GmBT>
__aicore__ inline void GbProcessKL1Loop(
    const GmAT& gmATensor, const GmBT& gmBTensor, uint32_t mOff, uint32_t nOff, uint32_t curTileM, uint32_t curTileN,
    uint32_t K, uint32_t tileKChunk, uint32_t baseK, uint32_t kL1Iter, uint32_t l0cOffsetBytes, uint64_t& l0PingPong,
    uint64_t& l1LoopCnt)
{
    using DType = typename GbDtypeTraits<DtypeId>::type;
    constexpr uint32_t C0 = GbDtypeTraits<DtypeId>::C0;
    GbLoadL1Chunk<DtypeId>(gmATensor, gmBTensor, mOff, nOff, curTileM, curTileN, K, tileKChunk, 0, l1LoopCnt);
    for (uint32_t kL1Index = 0; kL1Index < kL1Iter; ++kL1Index) {
        const uint32_t kOff = kL1Index * tileKChunk;
        const uint32_t currentBuf = (l1LoopCnt - 1) & GEMM_BATCHED_L1_BUF_MASK;
        const uint32_t currentK = Min<uint32_t>(tileKChunk, K - kOff);
        if (kL1Index + 1 < kL1Iter) {
            GbLoadL1Chunk<DtypeId>(
                gmATensor, gmBTensor, mOff, nOff, curTileM, curTileN, K, tileKChunk, kOff + tileKChunk, l1LoopCnt);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(currentBuf);
        const uint32_t l1BaseBytes = currentBuf * (AscendC::TOTAL_L1_SIZE >> 1);
        const uint32_t aL1Elems = RoundUp<uint32_t>(curTileM, GEMM_BATCHED_FRACTAL) * RoundUp<uint32_t>(currentK, C0);
        const uint32_t bL1ByteOff = l1BaseBytes + aL1Elems * sizeof(DType);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, DType>(l1BaseBytes),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0>>(curTileM, currentK));
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, DType>(bL1ByteOff),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0>>(currentK, curTileN));
        GbProcessL0Loop<DtypeId>(
            tensorAL1, tensorBL1, curTileM, curTileN, currentK, baseK, kL1Index, kL1Iter, kOff, tileKChunk,
            l0cOffsetBytes, l0PingPong);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(currentBuf);
    }
}

template <uint32_t DtypeId, typename GmAT, typename GmBT, typename GmCT>
__aicore__ inline uint32_t GbProcessMNTiles(
    const GmAT& gmATensor, const GmBT& gmBTensor, const GmCT& gmCTensor, uint32_t mStart, uint32_t mEnd,
    uint32_t nStart, uint32_t nEnd, uint32_t tileM, uint32_t tileN, uint32_t K, uint32_t tileKChunk, uint32_t baseK,
    uint32_t kL1Iter, uint32_t outputPingPong = 0)
{
    // Decide the L0C banking mode from the configured tile size (not the
    // per-tile tail), so every tile in the kernel uses the same bank so that
    // the FIX_M handshake flags always match.  A tile larger than half of L0C
    // cannot be double-buffered and instead uses the whole L0C as one bank.
    const bool singleL0cBank =
        static_cast<uint64_t>(tileM) * tileN * sizeof(float) > (AscendC::TOTAL_L0C_SIZE >> 1);
    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);

            const uint16_t outputBank = singleL0cBank ? static_cast<uint16_t>(0)
                                                      : static_cast<uint16_t>(outputPingPong & 1U);
            const uint32_t l0cOffsetBytes = singleL0cBank ? 0 : outputBank * (AscendC::TOTAL_L0C_SIZE >> 1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(outputBank);
            uint64_t l1LoopCnt = 0;
            uint64_t l0PingPong = 0;

            GbProcessKL1Loop<DtypeId>(
                gmATensor, gmBTensor, mOff, nOff, curTileM, curTileN, K, tileKChunk, baseK, kL1Iter, l0cOffsetBytes,
                l0PingPong, l1LoopCnt);

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(outputBank);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(outputBank);

            auto tensorL0C = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffsetBytes),
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<GEMM_BATCHED_L0C_C0>>(
                    curTileM, curTileN));
            auto gmBlockC = gmCTensor.Slice(
                AscendC::Te::MakeCoord((long)mOff, (long)nOff), AscendC::Te::MakeShape(curTileM, curTileN));
            AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{})
                .Call(gmBlockC, tensorL0C, AscendC::Te::FixpipeParams{FINAL_ACCUMULATION});
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(outputBank);
            ++outputPingPong;
        }
    }
    return outputPingPong;
}

struct GbTaskCoordinates {
    uint32_t batch;
    uint32_t mBlock;
    uint32_t nBlock;
};

__aicore__ inline GbTaskCoordinates GbResolveTaskCoordinates(uint32_t taskId, const GemmBatchedGemmTilingData& tiling)
{
    uint32_t batch;
    uint32_t spatialTask;
    if (tiling.balancedSplit == 4) {
        batch = taskId % tiling.batchCount;
        const uint32_t ordinal = taskId / tiling.batchCount;
        spatialTask = ordinal < 3 ? 2 * ordinal + 1 : (ordinal < 6 ? 2 * (ordinal - 3) : 6);
    } else if (tiling.balancedSplit == 3) {
        batch = taskId % tiling.batchCount;
        const uint32_t ordinal = taskId / tiling.batchCount;
        const uint32_t orderedM = ordinal < 3 ? ordinal * 2 : ((ordinal - 3) < 3 ? (ordinal - 3) * 2 + 1 : 6);
        spatialTask = orderedM * tiling.nBlocks;
    } else if (tiling.balancedSplit == 2) {
        batch = taskId % tiling.batchCount;
        spatialTask = taskId / tiling.batchCount;
    } else {
        const uint32_t spatialTasks = tiling.mBlocks * tiling.nBlocks;
        batch = taskId / spatialTasks;
        spatialTask = taskId % spatialTasks;
    }
    const uint32_t mBlock = spatialTask / tiling.nBlocks;
    uint32_t nBlock = spatialTask % tiling.nBlocks;
    if ((mBlock & 1U) != 0) {
        nBlock = tiling.nBlocks - 1 - nBlock;
    }
    return {batch, mBlock, nBlock};
}

struct GbTaskBounds {
    uint32_t mStart;
    uint32_t mEnd;
    uint32_t nStart;
    uint32_t nEnd;
};

__aicore__ inline GbTaskBounds GbResolveTaskBounds(
    const GbTaskCoordinates& task, const GemmBatchedGemmTilingData& tiling)
{
    if (tiling.balancedSplit == 0) {
        const uint32_t mStart = task.mBlock * tiling.singleCoreM;
        const uint32_t nStart = task.nBlock * tiling.singleCoreN;
        return {
            mStart, Min<uint32_t>(mStart + tiling.singleCoreM, tiling.m), nStart,
            Min<uint32_t>(nStart + tiling.singleCoreN, tiling.n)};
    }
    const uint32_t mTiles = (tiling.m + GEMM_BATCHED_BASE_M - 1) / GEMM_BATCHED_BASE_M;
    const uint32_t nTiles = (tiling.n + GEMM_BATCHED_BASE_N - 1) / GEMM_BATCHED_BASE_N;
    const uint32_t mStart = (task.mBlock * mTiles / tiling.mBlocks) * GEMM_BATCHED_BASE_M;
    const uint32_t mEnd = Min<uint32_t>(((task.mBlock + 1) * mTiles / tiling.mBlocks) * GEMM_BATCHED_BASE_M, tiling.m);
    const uint32_t nStart = (task.nBlock * nTiles / tiling.nBlocks) * GEMM_BATCHED_BASE_N;
    const uint32_t nEnd = Min<uint32_t>(((task.nBlock + 1) * nTiles / tiling.nBlocks) * GEMM_BATCHED_BASE_N, tiling.n);
    return {mStart, mEnd, nStart, nEnd};
}

template <class LayoutA, class LayoutB, uint32_t DtypeId, typename COutT = float>
__aicore__ inline uint32_t GbProcessOneTask(
    __gm__ uint64_t* aPtrArr, __gm__ uint64_t* bPtrArr, __gm__ uint64_t* cPtrArr, uint32_t taskId,
    const GemmBatchedGemmTilingData& tiling, uint32_t baseK, uint32_t kL1Iter, __gm__ uint8_t* directA = nullptr,
    __gm__ uint8_t* directB = nullptr, __gm__ uint8_t* directC = nullptr, uint32_t outputPingPong = 0)
{
    using DType = typename GbDtypeTraits<DtypeId>::type;
    const GbTaskCoordinates task = GbResolveTaskCoordinates(taskId, tiling);
    const GbTaskBounds bounds = GbResolveTaskBounds(task, tiling);
    __gm__ DType* aGm = directA == nullptr ? reinterpret_cast<__gm__ DType*>(aPtrArr[task.batch]) :
                                             reinterpret_cast<__gm__ DType*>(directA);
    __gm__ DType* bGm = directB == nullptr ? reinterpret_cast<__gm__ DType*>(bPtrArr[task.batch]) :
                                             reinterpret_cast<__gm__ DType*>(directB);
    // COutT selects the GM output element type.  When it differs from the L0C
    // accumulator (float), the fixpipe performs the cast on copy-out (e.g. the
    // FLOAT16 path writes half directly and skips a separate FP32 temporary).
    __gm__ COutT* cGm = directC == nullptr ? reinterpret_cast<__gm__ COutT*>(cPtrArr[task.batch]) :
                                             reinterpret_cast<__gm__ COutT*>(directC);
    auto gmATensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGm),
        MakeGmLayout<DType>(LayoutA{}, tiling.m, tiling.k, tiling.lda));
    auto gmBTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGm),
        MakeGmLayout<DType>(LayoutB{}, tiling.k, tiling.n, tiling.ldb));
    auto gmCTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGm),
        MakeGmLayout<COutT>(AscendC::Te::NDExtLayoutPtn{}, tiling.m, tiling.n, tiling.ldc));
    return GbProcessMNTiles<DtypeId>(
        gmATensor, gmBTensor, gmCTensor, bounds.mStart, bounds.mEnd, bounds.nStart, bounds.nEnd, tiling.tileM,
        tiling.tileN, tiling.k, tiling.tileKChunk, baseK, kL1Iter, outputPingPong);
}

template <uint32_t DtypeId>
__aicore__ inline void GbInitializeCubePipeline()
{
    AscendC::InitSocState();
    if constexpr (DtypeId == 0) {
        AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    }
    AscendC::SetMMRowMajor();
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
}

__aicore__ inline void GbFinalizeCubePipeline()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::SetMMColumnMajor();
}

template <class LayoutA, class LayoutB, uint32_t DtypeId>
__aicore__ inline void gemm_batched_gemm_kernel_impl(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* carray, GemmBatchedGemmTilingData tiling)
{
    GbInitializeCubePipeline<DtypeId>();
    using DType = typename GbDtypeTraits<DtypeId>::type;

    const uint32_t totalTasks = tiling.totalTasks;
    uint32_t baseK = (sizeof(DType) == sizeof(float)) ? GEMM_BATCHED_FP32_L0_BASE_K : GEMM_BATCHED_FP16_L0_BASE_K;
    if constexpr (DtypeId == 0) {
        if (tiling.balancedSplit == 6) {
            baseK = 32;
        } else if (tiling.balancedSplit != 0) {
            baseK = 64;
        }
    }
    const uint32_t K = tiling.k;
    const uint32_t tileKChunk = tiling.tileKChunk;
    const uint32_t kL1Iter = (K + tileKChunk - 1) / tileKChunk;

    __gm__ uint64_t* aPtrArr = reinterpret_cast<__gm__ uint64_t*>(aarray);
    __gm__ uint64_t* bPtrArr = reinterpret_cast<__gm__ uint64_t*>(barray);
    __gm__ uint64_t* cPtrArr = reinterpret_cast<__gm__ uint64_t*>(carray);

    for (uint32_t taskId = AscendC::GetBlockIdx(); taskId < totalTasks; taskId += AscendC::GetBlockNum()) {
        GbProcessOneTask<LayoutA, LayoutB, DtypeId>(aPtrArr, bPtrArr, cPtrArr, taskId, tiling, baseK, kL1Iter);
    }

    GbFinalizeCubePipeline();
}

// FLOAT16-input GEMM that writes FLOAT16 output directly from the FP32 L0C
// accumulator via the fixpipe cast, avoiding an FP32 temporary and a separate
// epilogue pass.  Used by aclblasHgemmBatched's alpha==1 / beta==0 fast path.
template <class LayoutA, class LayoutB, uint32_t DtypeId>
__aicore__ inline void gemm_batched_gemm_kernel_impl_halfout(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* carray, GemmBatchedGemmTilingData tiling)
{
    GbInitializeCubePipeline<DtypeId>();
    using DType = typename GbDtypeTraits<DtypeId>::type;

    const uint32_t totalTasks = tiling.totalTasks;
    const uint32_t baseK = (sizeof(DType) == sizeof(float)) ? GEMM_BATCHED_FP32_L0_BASE_K : GEMM_BATCHED_FP16_L0_BASE_K;
    const uint32_t K = tiling.k;
    const uint32_t tileKChunk = tiling.tileKChunk;
    const uint32_t kL1Iter = (K + tileKChunk - 1) / tileKChunk;

    __gm__ uint64_t* aPtrArr = reinterpret_cast<__gm__ uint64_t*>(aarray);
    __gm__ uint64_t* bPtrArr = reinterpret_cast<__gm__ uint64_t*>(barray);
    __gm__ uint64_t* cPtrArr = reinterpret_cast<__gm__ uint64_t*>(carray);

    for (uint32_t taskId = AscendC::GetBlockIdx(); taskId < totalTasks; taskId += AscendC::GetBlockNum()) {
        GbProcessOneTask<LayoutA, LayoutB, DtypeId, half>(
            aPtrArr, bPtrArr, cPtrArr, taskId, tiling, baseK, kL1Iter);
    }

    GbFinalizeCubePipeline();
}
