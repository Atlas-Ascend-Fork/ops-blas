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

#include <cstdint>
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"

namespace te = AscendC::Te;

constexpr int64_t SYRK_ARCH35_L0A_SIZE = 64 * 1024;
constexpr int64_t SYRK_ARCH35_L0C_SIZE = 256 * 1024;

constexpr uint64_t SYRK_ARCH35_FP32_C0 = 8;
constexpr uint64_t SYRK_ARCH35_FP16_C0 = 16;
constexpr uint64_t SYRK_ARCH35_FRACTAL = 16;
constexpr uint64_t SYRK_ARCH35_L0C_C0 = 16;
constexpr uint64_t SYRK_ARCH35_L0_BUF_MASK = 0x1;
constexpr uint64_t SYRK_ARCH35_HALF_L0_SIZE = SYRK_ARCH35_L0A_SIZE / 2;
constexpr uint64_t SYRK_ARCH35_L0C_BUF_MASK = 0x1;
constexpr uint64_t SYRK_ARCH35_HALF_L0C_SIZE = SYRK_ARCH35_L0C_SIZE / 2;

template <typename CopyAtom, typename TensorGM, typename TensorL1>
__aicore__ inline void SyrkGemmCopyGM2L1(
    CopyAtom copyGM2L1, TensorGM gmTensor, TensorL1 tensorL1, uint64_t off0, uint64_t off1, uint64_t dim0,
    uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(te::MakeCoord(off0, off1), te::MakeShape(dim0, dim1));
    te::Copy(copyGM2L1, tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void SyrkGemmL0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t curML1, uint64_t curKL1, uint64_t nL0, uint64_t iter0,
    uint64_t baseK, uint64_t& l0PingPong, uint64_t l0cBufId)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, baseK);

    uint64_t l0cOffset = l0cBufId * SYRK_ARCH35_HALF_L0C_SIZE;
    auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curML1, nL0);
    auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layoutL0C);
    auto copyL12L0A = te::MakeCopy(te::CopyL12L0A{});
    auto copyL12L0B = te::MakeCopy(te::CopyL12L0B{});
    auto mmadAtom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};

    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Offset = iter1 * baseK;
        uint64_t curKL0 = (kL0Offset + baseK > curKL1) ? (curKL1 - kL0Offset) : baseK;
        uint64_t l0BufId = l0PingPong & SYRK_ARCH35_L0_BUF_MASK;
        uint64_t l0Offset = SYRK_ARCH35_HALF_L0_SIZE * l0BufId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        auto layoutAL0 = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curML1, curKL0);
        auto tensorAL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(l0Offset), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.Slice(te::MakeCoord(0, kL0Offset), te::MakeShape(curML1, curKL0));
        te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        auto layoutBL0 = te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curKL0, nL0);
        auto tensorBL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0B, T>(l0Offset), layoutBL0);
        auto tensorBlockBL1 = tensorBL1.Slice(te::MakeCoord(kL0Offset, 0), te::MakeShape(curKL0, nL0));
        te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        bool isFirstK = (iter0 == 0 && iter1 == 0);
        te::MmadParams mmadParams{
            static_cast<uint16_t>(curML1), static_cast<uint16_t>(nL0), static_cast<uint16_t>(curKL0), 0, isFirstK};
        te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }
}

template <typename CopyAtom, typename TensorA, typename TensorB, typename TensorAL1, typename TensorBL1>
__aicore__ inline void SyrkGemmProcessKChunk(
    CopyAtom copyGM2L1, TensorA gmLeftTensor, TensorB gmRightTensor, TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t l1BufId, uint64_t mOff, uint64_t nOff, uint64_t kOff, uint64_t curK, uint64_t mL0, uint64_t nL0,
    uint64_t iter0, uint64_t baseK, uint64_t& l0PingPong, uint64_t l0cBufId)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    SyrkGemmCopyGM2L1(copyGM2L1, gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    SyrkGemmCopyGM2L1(copyGM2L1, gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    SyrkGemmL0MmadLoop(tensorAL1, tensorBL1, mL0, curK, nL0, iter0, baseK, l0PingPong, l0cBufId);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <
    typename T, uint64_t C0, bool INCLUDE_LOW_LOW, typename MmadAtom, typename TensorL0C, typename TensorLeftHigh,
    typename TensorLeftLow, typename TensorRightHigh, typename TensorRightLow>
__aicore__ inline void SyrkGemmResidualCrossMmad(
    MmadAtom mmolAtom, te::MmadParams& params, TensorL0C tensorL0C, TensorLeftHigh leftHighL0, TensorLeftLow leftLowL0,
    TensorRightHigh rightHighL0, TensorRightLow rightLowL0, uint64_t curK, uint64_t baseK)
{
    te::Mmad(mmolAtom.with(params), tensorL0C, leftHighL0, rightHighL0);
    params.cmatrixInitVal = false;
    if constexpr (INCLUDE_LOW_LOW) {
        te::Mmad(mmolAtom.with(params), tensorL0C, leftHighL0, rightLowL0);
        te::Mmad(mmolAtom.with(params), tensorL0C, leftLowL0, rightHighL0);
        te::Mmad(mmolAtom.with(params), tensorL0C, leftLowL0, rightLowL0);
    } else if (curK == baseK) {
        auto crossALayout = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<C0>>(params.m, curK * 2);
        auto crossBLayout = te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<C0>>(curK * 2, params.n);
        auto crossAL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(0), crossALayout);
        auto crossBL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0B, T>(0), crossBLayout);
        params.k = static_cast<uint16_t>(curK * 2);
        te::Mmad(mmolAtom.with(params), tensorL0C, crossAL0, crossBL0);
    } else {
        te::Mmad(mmolAtom.with(params), tensorL0C, leftHighL0, rightLowL0);
        te::Mmad(mmolAtom.with(params), tensorL0C, leftLowL0, rightHighL0);
    }
}

template <
    typename T, uint64_t C0, bool INCLUDE_LOW_LOW, typename TensorL0C, typename TensorLeftHighL1,
    typename TensorLeftLowL1, typename TensorRightHighL1, typename TensorRightLowL1>
__aicore__ inline void SyrkGemmResidualL0MmadStep(
    TensorL0C tensorL0C, TensorLeftHighL1 leftHighL1, TensorLeftLowL1 leftLowL1, TensorRightHighL1 rightHighL1,
    TensorRightLowL1 rightLowL1, uint64_t curM, uint64_t curK, uint64_t curN, uint64_t iter0, uint64_t iter1,
    uint64_t baseK, bool initializeAccumulator)
{
    uint64_t kOffset = iter1 * baseK;
    uint64_t stepK = Min<uint64_t>(baseK, curK - kOffset);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    auto layoutAL0 = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<C0>>(curM, stepK);
    auto layoutBL0 = te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<C0>>(stepK, curN);
    uint64_t leftBytes = RoundUp<uint64_t>(curM, SYRK_ARCH35_FRACTAL) * RoundUp<uint64_t>(stepK, C0) * sizeof(T);
    uint64_t rightBytes = RoundUp<uint64_t>(curN, SYRK_ARCH35_FRACTAL) * RoundUp<uint64_t>(stepK, C0) * sizeof(T);
    auto leftHighL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(0), layoutAL0);
    auto leftLowL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(leftBytes), layoutAL0);
    auto rightLowL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0B, T>(0), layoutBL0);
    auto rightHighL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0B, T>(rightBytes), layoutBL0);
    auto copyA = te::MakeCopy(te::CopyL12L0A{});
    auto copyB = te::MakeCopy(te::CopyL12L0B{});
    te::Copy(copyA, leftHighL0, leftHighL1.Slice(te::MakeCoord(0, kOffset), te::MakeShape(curM, stepK)));
    te::Copy(copyB, rightHighL0, rightHighL1.Slice(te::MakeCoord(kOffset, 0), te::MakeShape(stepK, curN)));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
    te::Copy(copyA, leftLowL0, leftLowL1.Slice(te::MakeCoord(0, kOffset), te::MakeShape(curM, stepK)));
    te::Copy(copyB, rightLowL0, rightLowL1.Slice(te::MakeCoord(kOffset, 0), te::MakeShape(stepK, curN)));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
    auto atom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};
    te::MmadParams params{
        static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(stepK), 0,
        initializeAccumulator && iter0 == 0 && iter1 == 0};
    SyrkGemmResidualCrossMmad<T, C0, INCLUDE_LOW_LOW>(
        atom, params, tensorL0C, leftHighL0, leftLowL0, rightHighL0, rightLowL0, stepK, baseK);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
}

template <
    typename T, uint64_t C0, bool INCLUDE_LOW_LOW, typename TensorLeftHighL1, typename TensorLeftLowL1,
    typename TensorRightHighL1, typename TensorRightLowL1>
__aicore__ inline void SyrkGemmResidualL0MmadLoop(
    TensorLeftHighL1 leftHighL1, TensorLeftLowL1 leftLowL1, TensorRightHighL1 rightHighL1, TensorRightLowL1 rightLowL1,
    uint64_t curM, uint64_t curK, uint64_t curN, uint64_t iter0, uint64_t baseK, uint64_t l0cBufId,
    bool initializeAccumulator)
{
    uint64_t kL0Iter = CeilDiv<uint64_t>(curK, baseK);
    uint64_t l0cOffset = l0cBufId * SYRK_ARCH35_HALF_L0C_SIZE;
    auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curM, curN);
    auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layoutL0C);
    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        SyrkGemmResidualL0MmadStep<T, C0, INCLUDE_LOW_LOW>(
            tensorL0C, leftHighL1, leftLowL1, rightHighL1, rightLowL1, curM, curK, curN, iter0, iter1, baseK,
            initializeAccumulator);
    }
}

template <typename TensorA, typename TensorB>
__aicore__ inline void SyrkGemmProcessKChunks(
    TensorA gmLeftTensor, TensorB gmRightTensor, uint32_t K, uint32_t tileKChunk, uint64_t baseK, uint64_t mOff,
    uint64_t nOff, uint64_t mL0, uint64_t nL0, uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t l0cBufId,
    uint32_t l1BufNum, int64_t l1Size)
{
    using T = float;
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    if (tileKChunk == 0) {
        return;
    }
    for (uint32_t kOff = 0; kOff < K; kOff += tileKChunk) {
        uint32_t curK = Min<uint32_t>(tileKChunk, K - kOff);
        uint64_t l1BufId = abL1LoopCnt & (l1BufNum - 1);
        uint64_t l1OffsetA = l1BufId * (l1Size / l1BufNum);
        uint64_t aSideL1Size =
            RoundUp<uint64_t>(mL0, SYRK_ARCH35_FRACTAL) * RoundUp<uint64_t>(curK, SYRK_ARCH35_FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(T);
        auto tensorAL1 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L1, T>(l1OffsetA),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(mL0, curK));
        auto tensorBL1 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L1, T>(l1OffsetB),
            te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curK, nL0));
        SyrkGemmProcessKChunk(
            copyGM2L1, gmLeftTensor, gmRightTensor, tensorAL1, tensorBL1, l1BufId, mOff, nOff, kOff, curK, mL0, nL0,
            kOff / tileKChunk, baseK, l0PingPong, l0cBufId);
        abL1LoopCnt++;
    }
}

template <typename T, uint64_t C0>
__aicore__ inline auto SyrkGemmMakeNzL1(uint64_t offset, uint64_t rows, uint64_t cols)
{
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(offset),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<C0>>(rows, cols));
}

template <typename T, uint64_t C0>
__aicore__ inline auto SyrkGemmMakeZnL1(uint64_t offset, uint64_t rows, uint64_t cols)
{
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(offset),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<C0>>(rows, cols));
}

template <
    typename T, uint64_t C0, typename TensorLeftHigh, typename TensorLeftLow, typename TensorRightHigh,
    typename TensorRightLow>
__aicore__ inline void SyrkGemmLoadResidualL1(
    TensorLeftHigh gmLeftHigh, TensorLeftLow gmLeftLow, TensorRightHigh gmRightHigh, TensorRightLow gmRightLow,
    uint64_t mOff, uint64_t nOff, uint32_t kOff, uint32_t curK, uint64_t mL0, uint64_t nL0, uint64_t l1BufId,
    uint32_t l1BufNum, int64_t l1Size)
{
    uint64_t l1Base = l1BufId * (l1Size / l1BufNum);
    uint64_t aElements = RoundUp<uint64_t>(mL0, SYRK_ARCH35_FRACTAL) * RoundUp<uint64_t>(curK, C0);
    uint64_t bElements = RoundUp<uint64_t>(nL0, SYRK_ARCH35_FRACTAL) * RoundUp<uint64_t>(curK, C0);
    uint64_t leftHighOffset = l1Base;
    uint64_t leftLowOffset = leftHighOffset + aElements * sizeof(T);
    uint64_t rightHighOffset = leftLowOffset + aElements * sizeof(T);
    uint64_t rightLowOffset = rightHighOffset + bElements * sizeof(T);
    auto leftHighL1 = SyrkGemmMakeNzL1<T, C0>(leftHighOffset, mL0, curK);
    auto leftLowL1 = SyrkGemmMakeNzL1<T, C0>(leftLowOffset, mL0, curK);
    auto rightHighL1 = SyrkGemmMakeZnL1<T, C0>(rightHighOffset, curK, nL0);
    auto rightLowL1 = SyrkGemmMakeZnL1<T, C0>(rightLowOffset, curK, nL0);
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    SyrkGemmCopyGM2L1(copyGM2L1, gmLeftHigh, leftHighL1, mOff, kOff, mL0, curK);
    SyrkGemmCopyGM2L1(copyGM2L1, gmLeftLow, leftLowL1, mOff, kOff, mL0, curK);
    SyrkGemmCopyGM2L1(copyGM2L1, gmRightHigh, rightHighL1, kOff, nOff, curK, nL0);
    SyrkGemmCopyGM2L1(copyGM2L1, gmRightLow, rightLowL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
}

template <
    typename T, uint64_t C0, bool INCLUDE_LOW_LOW, typename TensorLeftHigh, typename TensorLeftLow,
    typename TensorRightHigh, typename TensorRightLow>
__aicore__ inline void SyrkGemmProcessResidualKChunks(
    TensorLeftHigh gmLeftHigh, TensorLeftLow gmLeftLow, TensorRightHigh gmRightHigh, TensorRightLow gmRightLow,
    uint32_t K, uint32_t tileKChunk, uint64_t baseK, uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t l0cBufId, uint32_t l1BufNum, int64_t l1Size,
    bool initializeAccumulator)
{
    if (tileKChunk == 0 || K == 0)
        return;
    uint32_t chunkCount = CeilDiv<uint32_t>(K, tileKChunk);
    uint64_t firstBufId = abL1LoopCnt & (l1BufNum - 1);
    uint32_t firstK = Min<uint32_t>(tileKChunk, K);
    SyrkGemmLoadResidualL1<T, C0>(
        gmLeftHigh, gmLeftLow, gmRightHigh, gmRightLow, mOff, nOff, 0, firstK, mL0, nL0, firstBufId, l1BufNum, l1Size);
    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        uint32_t kOff = chunk * tileKChunk;
        uint32_t curK = Min<uint32_t>(tileKChunk, K - kOff);
        uint64_t l1BufId = (abL1LoopCnt + chunk) & (l1BufNum - 1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        if (chunk + 1 < chunkCount) {
            uint32_t nextKOff = kOff + tileKChunk;
            uint32_t nextK = Min<uint32_t>(tileKChunk, K - nextKOff);
            uint64_t nextBufId = (abL1LoopCnt + chunk + 1) & (l1BufNum - 1);
            SyrkGemmLoadResidualL1<T, C0>(
                gmLeftHigh, gmLeftLow, gmRightHigh, gmRightLow, mOff, nOff, nextKOff, nextK, mL0, nL0, nextBufId,
                l1BufNum, l1Size);
        }
        uint64_t l1Base = l1BufId * (l1Size / l1BufNum);
        uint64_t aElements = RoundUp<uint64_t>(mL0, SYRK_ARCH35_FRACTAL) * RoundUp<uint64_t>(curK, C0);
        uint64_t bElements = RoundUp<uint64_t>(nL0, SYRK_ARCH35_FRACTAL) * RoundUp<uint64_t>(curK, C0);
        uint64_t leftLowOffset = l1Base + aElements * sizeof(T);
        uint64_t rightHighOffset = leftLowOffset + aElements * sizeof(T);
        uint64_t rightLowOffset = rightHighOffset + bElements * sizeof(T);
        auto leftHighL1 = SyrkGemmMakeNzL1<T, C0>(l1Base, mL0, curK);
        auto leftLowL1 = SyrkGemmMakeNzL1<T, C0>(leftLowOffset, mL0, curK);
        auto rightHighL1 = SyrkGemmMakeZnL1<T, C0>(rightHighOffset, curK, nL0);
        auto rightLowL1 = SyrkGemmMakeZnL1<T, C0>(rightLowOffset, curK, nL0);
        SyrkGemmResidualL0MmadLoop<T, C0, INCLUDE_LOW_LOW>(
            leftHighL1, leftLowL1, rightHighL1, rightLowL1, mL0, curK, nL0, chunk, baseK, l0cBufId,
            initializeAccumulator);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    }
    abL1LoopCnt += chunkCount;
}

template <typename TensorTemp>
__aicore__ inline void SyrkGemmStoreL0C(
    TensorTemp gmTempTensor, uint64_t mOff, uint64_t nOff, uint64_t curM, uint64_t curN, uint64_t l0cBufId)
{
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cBufId);
    uint64_t l0cOffset = l0cBufId * SYRK_ARCH35_HALF_L0C_SIZE;
    auto layout = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curM, curN);
    auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(l0cOffset), layout);
    auto gmBlock = gmTempTensor.Slice(te::MakeCoord(mOff, nOff), te::MakeShape(curM, curN));
    te::MakeCopy(te::CopyL0C2GM{}).Call(gmBlock, tensorL0C, te::FixpipeParams{0});
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void SyrkGemmProcessTile(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor, uint32_t K, uint32_t tileKChunk,
    uint64_t baseK, uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd, uint32_t tileM, uint32_t tileN,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t l0cPingPong, uint32_t l1BufNum, int64_t l1Size)
{
    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);
            uint64_t l0cBufId = l0cPingPong & SYRK_ARCH35_L0C_BUF_MASK;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
            SyrkGemmProcessKChunks(
                gmLeftTensor, gmRightTensor, K, tileKChunk, baseK, mOff, nOff, curTileM, curTileN, abL1LoopCnt,
                l0PingPong, l0cBufId, l1BufNum, l1Size);
            SyrkGemmStoreL0C(gmTempTensor, mOff, nOff, curTileM, curTileN, l0cBufId);
            l0cPingPong++;
        }
    }
}

template <
    typename T, uint64_t C0, bool INCLUDE_LOW_LOW, typename TensorLeftHigh, typename TensorLeftLow,
    typename TensorRightHigh, typename TensorRightLow, typename TensorTemp>
__aicore__ inline void SyrkGemmProcessResidualTile(
    TensorLeftHigh gmLeftHigh, TensorLeftLow gmLeftLow, TensorRightHigh gmRightHigh, TensorRightLow gmRightLow,
    TensorTemp gmTempTensor, uint32_t K, uint32_t tileKChunk, uint64_t baseK, uint32_t mStart, uint32_t mEnd,
    uint32_t nStart, uint32_t nEnd, uint32_t tileM, uint32_t tileN, uint64_t& abL1LoopCnt, uint64_t& l0PingPong,
    uint64_t& l0cPingPong, uint32_t l1BufNum, int64_t l1Size)
{
    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);
            uint64_t l0cBufId = l0cPingPong & SYRK_ARCH35_L0C_BUF_MASK;
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
            SyrkGemmProcessResidualKChunks<T, C0, INCLUDE_LOW_LOW>(
                gmLeftHigh, gmLeftLow, gmRightHigh, gmRightLow, K, tileKChunk, baseK, mOff, nOff, curTileM, curTileN,
                abL1LoopCnt, l0PingPong, l0cBufId, l1BufNum, l1Size, true);
            SyrkGemmStoreL0C(gmTempTensor, mOff, nOff, curTileM, curTileN, l0cBufId);
            l0cPingPong++;
        }
    }
}

template <
    typename T, uint64_t C0, bool INCLUDE_LOW_LOW, typename TensorLeftHigh, typename TensorLeftLow,
    typename TensorRightHigh, typename TensorRightLow, typename TensorReverseLeftHigh, typename TensorReverseLeftLow,
    typename TensorReverseRightHigh, typename TensorReverseRightLow, typename TensorTemp>
__aicore__ inline void SyrkGemmProcessResidualSymmetricTile(
    TensorLeftHigh gmLeftHigh, TensorLeftLow gmLeftLow, TensorRightHigh gmRightHigh, TensorRightLow gmRightLow,
    TensorReverseLeftHigh gmReverseLeftHigh, TensorReverseLeftLow gmReverseLeftLow,
    TensorReverseRightHigh gmReverseRightHigh, TensorReverseRightLow gmReverseRightLow, TensorTemp gmTempTensor,
    uint32_t K, uint32_t tileKChunk, uint64_t baseK, uint32_t mOff, uint32_t nOff, uint32_t curTileM, uint32_t curTileN,
    bool addReverse, uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t& l0cPingPong, uint32_t l1BufNum,
    int64_t l1Size, uint32_t l0cBufNum)
{
    uint64_t l0cBufId = (l0cBufNum > 1) ? (l0cPingPong & SYRK_ARCH35_L0C_BUF_MASK) : 0;
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cBufId);
    // First direction: X_i * Y_j^T.
    SyrkGemmProcessResidualKChunks<T, C0, INCLUDE_LOW_LOW>(
        gmLeftHigh, gmLeftLow, gmRightHigh, gmRightLow, K, tileKChunk, baseK, mOff, nOff, curTileM, curTileN,
        abL1LoopCnt, l0PingPong, l0cBufId, l1BufNum, l1Size, true);
    if (addReverse) {
        // Off-diagonal target tiles accumulate Y_i * X_j^T in the same L0C.
        // The resulting tile is already the symmetric sum required by SYR2K.
        SyrkGemmProcessResidualKChunks<T, C0, INCLUDE_LOW_LOW>(
            gmReverseLeftHigh, gmReverseLeftLow, gmReverseRightHigh, gmReverseRightLow, K, tileKChunk, baseK, mOff,
            nOff, curTileM, curTileN, abL1LoopCnt, l0PingPong, l0cBufId, l1BufNum, l1Size, false);
    }
    SyrkGemmStoreL0C(gmTempTensor, mOff, nOff, curTileM, curTileN, l0cBufId);
    ++l0cPingPong;
}

struct SyrkGemmTileRange {
    uint32_t mStart;
    uint32_t mEnd;
    uint32_t nStart;
    uint32_t nEnd;
};

// Zero tiling parameters would explode the tile count (CeilDiv yields T_MAX,
// not 0, for a zero divisor), so every tile-loop entry must validate them.
// Shared tile-loop setup for the syrk kernel impls: resolves the M/N split and
// the total tile count, returning 0 (and leaving divM/divN untouched) when the
// tiling parameters are invalid.
template <typename TilingData>
__aicore__ inline uint64_t SyrkGemmTotalTiles(uint32_t n, const TilingData& tiling, uint32_t& divM, uint32_t& divN)
{
    if (n == 0 || tiling.singleCoreM == 0 || tiling.singleCoreN == 0 || tiling.tileM == 0 || tiling.tileN == 0) {
        return 0;
    }
    divM = CeilDiv<uint32_t>(n, tiling.singleCoreM);
    divN = CeilDiv<uint32_t>(n, tiling.singleCoreN);
    return static_cast<uint64_t>(divM) * divN;
}

template <typename TilingData>
__aicore__ inline SyrkGemmTileRange DecodeSyrkGemmTile(
    uint64_t tileIdx, uint32_t divN, uint32_t n, const TilingData& tiling)
{
    if (divN == 0) {
        return {0, 0, 0, 0};
    }
    uint64_t coreIdxM = tileIdx / divN;
    uint64_t coreIdxN = tileIdx % divN;
    coreIdxN = ((coreIdxM & 1U) != 0) ? (divN - 1 - coreIdxN) : coreIdxN;
    uint32_t mStart = coreIdxM * tiling.singleCoreM;
    uint32_t nStart = coreIdxN * tiling.singleCoreN;
    return {
        mStart, Min<uint32_t>(mStart + tiling.singleCoreM, n), nStart, Min<uint32_t>(nStart + tiling.singleCoreN, n)};
}

__aicore__ inline void SyrkGemmInitEvents()
{
    for (int32_t flagId = 0; flagId < 2; flagId++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(flagId);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(flagId);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(flagId);
    }
}

__aicore__ inline void SyrkGemmFinalizeEvents()
{
    for (int32_t flagId = 0; flagId < 2; flagId++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(flagId);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(flagId);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(flagId);
    }
}

// Shared tile-loop context for the syrk kernel impls: validates the tiling,
// resolves the M/N split and total tile count, and decodes per-tile ranges.
template <typename TilingData>
struct SyrkGemmTileLoopCtx {
    const TilingData* tiling = nullptr;
    uint32_t n = 0;
    uint32_t divM = 0;
    uint32_t divN = 0;
    uint64_t totalTiles = 0;
    uint64_t blockNum = 0;
    uint64_t l0PingPong = 0;
    uint64_t l0cPingPong = 0;
    uint64_t abL1LoopCnt = 0;

    __aicore__ inline bool Init(const TilingData& t)
    {
        tiling = &t;
        n = t.n;
        totalTiles = SyrkGemmTotalTiles(n, t, divM, divN);
        blockNum = AscendC::GetBlockNum();
        return totalTiles != 0;
    }
    __aicore__ inline SyrkGemmTileRange Decode(uint64_t tileIdx) const
    {
        return DecodeSyrkGemmTile(tileIdx, divN, n, *tiling);
    }
};

// Shared tile-loop driver: init/context/events and the grid-stride tile loop are
// defined once; the per-tile body is injected as a functor that receives the
// decoded range and the three loop-carried counters by reference.
template <typename TilingData, typename ProcessTileFn>
__aicore__ inline void SyrkGemmRunTileLoop(const TilingData& tiling, ProcessTileFn processTile)
{
    SyrkGemmTileLoopCtx<TilingData> ctx;
    if (!ctx.Init(tiling)) {
        return;
    }
    SyrkGemmInitEvents();
    for (uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < ctx.totalTiles; tileIdx += ctx.blockNum) {
        SyrkGemmTileRange range = ctx.Decode(tileIdx);
        processTile(range, ctx.abL1LoopCnt, ctx.l0PingPong, ctx.l0cPingPong);
    }
    SyrkGemmFinalizeEvents();
}

template <
    bool INCLUDE_LOW_LOW, typename TensorLeftHigh, typename TensorLeftLow, typename TensorRightHigh,
    typename TensorRightLow, typename TensorTemp, typename TilingData, typename T = half,
    uint64_t C0 = SYRK_ARCH35_FP16_C0>
__aicore__ inline void SyrkGemmResidualKernelImpl(
    TensorLeftHigh gmLeftHigh, TensorLeftLow gmLeftLow, TensorRightHigh gmRightHigh, TensorRightLow gmRightLow,
    TensorTemp gmTempTensor, const TilingData& tiling, uint32_t baseK, uint32_t l1BufNum, int64_t l1Size)
{
    struct ProcessTileFn {
        TensorLeftHigh gmLeftHigh;
        TensorLeftLow gmLeftLow;
        TensorRightHigh gmRightHigh;
        TensorRightLow gmRightLow;
        TensorTemp gmTempTensor;
        const TilingData* tiling;
        uint32_t baseK;
        uint32_t l1BufNum;
        int64_t l1Size;
        __aicore__ inline void operator()(
            const SyrkGemmTileRange& range, uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t& l0cPingPong) const
        {
            SyrkGemmProcessResidualTile<T, C0, INCLUDE_LOW_LOW>(
                gmLeftHigh, gmLeftLow, gmRightHigh, gmRightLow, gmTempTensor, tiling->k, tiling->tileKChunk, baseK,
                range.mStart, range.mEnd, range.nStart, range.nEnd, tiling->tileM, tiling->tileN, abL1LoopCnt,
                l0PingPong, l0cPingPong, l1BufNum, l1Size);
        }
    };
    SyrkGemmRunTileLoop(
        tiling,
        ProcessTileFn{gmLeftHigh, gmLeftLow, gmRightHigh, gmRightLow, gmTempTensor, &tiling, baseK, l1BufNum, l1Size});
}

template <typename TensorA, typename TensorB, typename TensorTemp, typename TilingData>
__aicore__ inline void SyrkGemmKernelImpl(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor, const TilingData& tiling, uint32_t baseK,
    uint32_t l1BufNum, int64_t l1Size)
{
    struct ProcessTileFn {
        TensorA gmLeftTensor;
        TensorB gmRightTensor;
        TensorTemp gmTempTensor;
        const TilingData* tiling;
        uint32_t K;
        uint32_t baseK;
        uint32_t l1BufNum;
        int64_t l1Size;
        __aicore__ inline void operator()(
            const SyrkGemmTileRange& range, uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint64_t& l0cPingPong) const
        {
            SyrkGemmProcessTile(
                gmLeftTensor, gmRightTensor, gmTempTensor, K, tiling->tileKChunk, baseK, range.mStart, range.mEnd,
                range.nStart, range.nEnd, tiling->tileM, tiling->tileN, abL1LoopCnt, l0PingPong, l0cPingPong, l1BufNum,
                l1Size);
        }
    };
    SyrkGemmRunTileLoop(
        tiling, ProcessTileFn{gmLeftTensor, gmRightTensor, gmTempTensor, &tiling, tiling.k, baseK, l1BufNum, l1Size});
}
