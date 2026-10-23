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

/*!
 * \file ctrmm_phaseb_tile.h
 * \brief Phase-B tile walk (extracted verbatim from ctrmm_kernel.cpp so that
 *        experimental variants can instantiate it from a separate translation
 *        unit WITHOUT changing the existing instantiation's code generation).
 */

#include <cstdint>

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "ctrmm_tiling_data.h"

// Same alias the production translation unit uses.
namespace te = AscendC::Te;

// Same event-flag id the production translation unit uses (file-local there;
// repeated here as a constant so the header is self-contained). Must NOT be a
// macro: the production TU declares it as constexpr and a macro would rewrite
// that declaration.
constexpr uint16_t CTRMM_PIPE_FLAG = 0;

namespace ctrmm_phaseb {

constexpr int64_t PB_L0A_SIZE = 64 * 1024;
constexpr int64_t PB_L1_SIZE = 512 * 1024;
constexpr uint32_t PB_FINAL_ACC = 3;
constexpr uint32_t PB_NONFINAL_ACC = 2;
constexpr uint64_t PB_FP32_C0 = 8;
constexpr uint64_t PB_FRACTAL = 16;
constexpr uint64_t PB_L0C_C0 = 16;
constexpr uint64_t PB_L1_BUF_NUM = 2;
constexpr uint64_t PB_L1_BUF_MASK = PB_L1_BUF_NUM - 1;
constexpr uint64_t PB_L0_BUF_MASK = 0x1;
constexpr uint64_t PB_HALF_L0_SIZE = PB_L0A_SIZE / 2;
constexpr uint32_t PB_BASE_M = 16;
constexpr uint32_t PB_BASE_N = 16;
constexpr uint32_t PB_BASE_K = 8;
constexpr uint32_t PB_FIXPIPE_N_ALIGN = 8;

template <typename TensorGM, typename TensorL1>
__aicore__ inline void CopyGM2L1(
    TensorGM gmTensor, TensorL1 tensorL1, uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(te::MakeCoord(off0, off1), te::MakeShape(dim0, dim1));
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    te::Copy(copyGM2L1, tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void L0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t curML1, uint64_t curKL1, uint64_t nL0, uint64_t kL1Iter,
    uint64_t iter0, uint64_t baseK, uint64_t& l0PingPong)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, baseK);
    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Offset = iter1 * baseK;
        uint64_t curKL0 = (kL0Offset + baseK > curKL1) ? (curKL1 - kL0Offset) : baseK;
        uint64_t l0BufId = l0PingPong & PB_L0_BUF_MASK;
        uint64_t l0Offset = PB_HALF_L0_SIZE * l0BufId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        auto layoutAL0 = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<PB_FP32_C0>>(curML1, curKL0);
        auto tensorAL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(l0Offset), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.Slice(te::MakeCoord(0, kL0Offset), te::MakeShape(curML1, curKL0));
        te::Copy(te::MakeCopy(te::CopyL12L0A{}), tensorAL0, tensorBlockAL1);
        auto layoutBL0 = te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<PB_FP32_C0>>(curKL0, nL0);
        auto tensorBL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0B, T>(l0Offset), layoutBL0);
        auto tensorBlockBL1 = tensorBL1.Slice(te::MakeCoord(kL0Offset, 0), te::MakeShape(curKL0, nL0));
        te::Copy(te::MakeCopy(te::CopyL12L0B{}), tensorBL0, tensorBlockBL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        bool isLastK = (iter0 + 1 == kL1Iter) && (iter1 + 1 == kL0Iter);
        bool isFirstK = (iter0 == 0 && iter1 == 0);
        uint8_t unitFlag = isLastK ? PB_FINAL_ACC : PB_NONFINAL_ACC;
        te::MmadParams mmadParams{
            static_cast<uint16_t>(curML1), static_cast<uint16_t>(nL0), static_cast<uint16_t>(curKL0), unitFlag,
            isFirstK};
        auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<PB_L0C_C0>>(curML1, nL0);
        auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(0), layoutL0C);
        te::Mmad(te::MmadAtom<te::MmadTraits<te::MmadOperation>>{}.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }
}

template <typename TensorA, typename TensorB, typename TensorAL1, typename TensorBL1>
__aicore__ inline void ProcessKChunk(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t l1BufId,
    uint64_t mOff, uint64_t nOff, uint64_t kOff, uint64_t curK, uint64_t mL0, uint64_t nL0, uint64_t kL1Iter,
    uint64_t iter0, uint64_t baseK, uint64_t& l0PingPong)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    CopyGM2L1(gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    CopyGM2L1(gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    L0MmadLoop(tensorAL1, tensorBL1, mL0, curK, nL0, kL1Iter, iter0, baseK, l0PingPong);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <typename TensorA, typename TensorB>
__aicore__ inline void ProcessKChunks(
    TensorA gmLeftTensor, TensorB gmRightTensor, uint32_t kStart, uint32_t kEnd, uint32_t tileKChunk, uint64_t kL1Iter,
    uint64_t baseK, uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0, uint64_t& abL1LoopCnt,
    uint64_t& l0PingPong)
{
    using T = float;
    for (uint32_t kOff = kStart; kOff < kEnd; kOff += tileKChunk) {
        uint32_t curK = Min<uint32_t>(tileKChunk, kEnd - kOff);
        uint64_t l1BufId = abL1LoopCnt & PB_L1_BUF_MASK;
        uint64_t l1OffsetA = l1BufId * (PB_L1_SIZE / PB_L1_BUF_NUM);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, PB_FRACTAL) * RoundUp<uint64_t>(curK, PB_FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(T);
        auto tensorAL1 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L1, T>(l1OffsetA),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<PB_FP32_C0>>(mL0, curK));
        auto tensorBL1 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L1, T>(l1OffsetB),
            te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<PB_FP32_C0>>(curK, nL0));
        ProcessKChunk(
            gmLeftTensor, gmRightTensor, tensorAL1, tensorBL1, l1BufId, mOff, nOff, kOff, curK, mL0, nL0, kL1Iter,
            (kOff - kStart) / tileKChunk, baseK, l0PingPong);
        abL1LoopCnt++;
    }
}

template <typename TensorTemp>
__aicore__ inline void CopyOutL0C2GM(TensorTemp gmTempTensor, uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0)
{
    auto gmBlockC = gmTempTensor.Slice(te::MakeCoord(mOff, nOff), te::MakeShape(mL0, nL0));
    auto tensorL0C = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, float>(0),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<PB_L0C_C0>>(mL0, nL0));
    te::MakeCopy(te::CopyL0C2GM{}).Call(gmBlockC, tensorL0C, te::FixpipeParams{PB_FINAL_ACC});
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void ProcessTile(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor, uint32_t K, uint32_t tileKChunk,
    uint64_t baseK, uint32_t clipMode, uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd, uint32_t tileM,
    uint32_t tileN, uint64_t& abL1LoopCnt, uint64_t& l0PingPong)
{
    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);
            // Triangular K-range clipping (L2 lever): every operand plane has
            // triangular support (Phase-A zeroed the non-referenced triangle),
            // so contributions outside [kStart, kEnd) are exactly zero.
            uint32_t kStart = 0;
            uint32_t kEnd = K;
            if (clipMode == 1) {
                kEnd = Min<uint32_t>(kEnd, nOff + curTileN);
            } else if (clipMode == 2) {
                kStart = (nOff > kStart) ? nOff : kStart;
            } else if (clipMode == 3) {
                kStart = (mOff > kStart) ? static_cast<uint32_t>(mOff) : kStart;
            } else if (clipMode == 4) {
                kEnd = Min<uint32_t>(kEnd, mOff + curTileM);
            }
            if (kStart >= kEnd) {
                continue; // tile entirely outside the triangle: exactly zero
            }
            const uint64_t kL1Iter = CeilDiv<uint64_t>(kEnd - kStart, tileKChunk);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(CTRMM_PIPE_FLAG);
            ProcessKChunks(
                gmLeftTensor, gmRightTensor, kStart, kEnd, tileKChunk, kL1Iter, baseK, mOff, nOff, curTileM, curTileN,
                abL1LoopCnt, l0PingPong);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(CTRMM_PIPE_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(CTRMM_PIPE_FLAG);
            CopyOutL0C2GM(gmTempTensor, mOff, nOff, curTileM, curTileN);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(CTRMM_PIPE_FLAG);
        }
    }
}

} // namespace ctrmm_phaseb
