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
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "cgemm_strided_batched_kernel.h"
#include "cgemm_strided_batched_reference.h"

// FP32 Cube operand transfers and accumulation.
// Shared by global-workspace and local Cube kernels. Device templates require
// their definitions in each calling translation unit.
namespace {

// FP32 Cube pipeline private to complex strided GEMM.
namespace {
constexpr int64_t GEMM_SB_L0A_SIZE = 64 * 1024;
constexpr uint16_t GEMM_SB_PIPE_FLAG = 0;
constexpr uint32_t GEMM_SB_FINAL_ACCUMULATION = 3;
constexpr uint32_t GEMM_SB_NON_FINAL_ACCUMULATION = 2;

constexpr uint64_t GEMM_SB_L1_SIZE = 512 * 1024;
constexpr uint64_t GEMM_SB_FP32_C0 = 8;
constexpr uint64_t GEMM_SB_FRACTAL = 16;
constexpr uint64_t GEMM_SB_L0C_C0 = 16;
constexpr uint64_t GEMM_SB_L1_BUF_NUM = 2;
constexpr uint64_t GEMM_SB_L1_BUF_MASK = GEMM_SB_L1_BUF_NUM - 1;
// Two alternating L0 banks retain operands until M releases them.
constexpr uint64_t GEMM_SB_L0_BUF_MASK = 1;
constexpr uint64_t GEMM_SB_HALF_L0_SIZE = GEMM_SB_L0A_SIZE / 2;

constexpr uint32_t GEMM_SB_VEC_BLOCK = 8; // 32B / sizeof(float): FP32 elements per aligned block
} // namespace

// ================================================================================
// GEMM kernel (AIC, tensor_api). Adapted from strmm_gemm_kernel: GM -> L1 (NZ/ZN) -> L0A/L0B ->
// AscendC::Te::Mmad -> L0C -> GM (Fixpipe). transLeft/transRight select the GM LayoutPtn (NDExt / DNExt); the
// L1 destination stays NZ (left) / ZN (right) and the routing auto-dispatches ND2Nz/DN2Nz/ND2Zn/DN2Zn.
// ================================================================================

template <typename TensorGM, typename TensorL1>
__aicore__ inline void GemmSbCopyGM2L1(
    TensorGM gmTensor, TensorL1 tensorL1, uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(AscendC::Te::MakeCoord(off0, off1), AscendC::Te::MakeShape(dim0, dim1));
    auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    AscendC::Te::Copy(copyGM2L1, tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void GemmSbL0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t curML1, uint64_t curKL1, uint64_t nL0, uint64_t kL1Iter,
    uint64_t iter0, uint64_t baseK, uint64_t& l0PingPong)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, baseK);
    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Offset = iter1 * baseK;
        uint64_t curKL0 = (kL0Offset + baseK > curKL1) ? (curKL1 - kL0Offset) : baseK;
        uint64_t l0BufId = l0PingPong & GEMM_SB_L0_BUF_MASK;
        uint64_t l0Offset = GEMM_SB_HALF_L0_SIZE * l0BufId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        auto layoutAL0 =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<GEMM_SB_FP32_C0>>(curML1, curKL0);
        auto tensorAL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, T>(l0Offset), layoutAL0);
        auto tensorBlockAL1 =
            tensorAL1.Slice(AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curML1, curKL0));
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{}), tensorAL0, tensorBlockAL1);
        auto layoutBL0 =
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<GEMM_SB_FP32_C0>>(curKL0, nL0);
        auto tensorBL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, T>(l0Offset), layoutBL0);
        auto tensorBlockBL1 =
            tensorBL1.Slice(AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, nL0));
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{}), tensorBL0, tensorBlockBL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        bool isLastK = (iter0 + 1 == kL1Iter) && (iter1 + 1 == kL0Iter);
        bool isFirstK = (iter0 == 0 && iter1 == 0);
        uint8_t unitFlag = isLastK ? GEMM_SB_FINAL_ACCUMULATION : GEMM_SB_NON_FINAL_ACCUMULATION;
        AscendC::Te::MmadParams mmadParams{
            static_cast<uint16_t>(curML1), static_cast<uint16_t>(nL0), static_cast<uint16_t>(curKL0), unitFlag,
            isFirstK};
        auto layoutL0C =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<GEMM_SB_L0C_C0>>(curML1, nL0);
        auto tensorL0C =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0), layoutL0C);
        AscendC::Te::Mmad(
            AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(mmadParams), tensorL0C,
            tensorAL0, tensorBL0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }
}

template <typename TensorLeft, typename TensorRight, typename TensorAL1, typename TensorBL1>
__aicore__ inline void GemmSbProcessKChunk(
    TensorLeft gmLeftTensor, TensorRight gmRightTensor, TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t l1BufId,
    uint64_t mOff, uint64_t nOff, uint64_t kOff, uint64_t curK, uint64_t mL0, uint64_t nL0, uint64_t kL1Iter,
    uint64_t iter0, uint64_t baseK, uint64_t& l0PingPong)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    GemmSbCopyGM2L1(gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    GemmSbCopyGM2L1(gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    GemmSbL0MmadLoop(tensorAL1, tensorBL1, mL0, curK, nL0, kL1Iter, iter0, baseK, l0PingPong);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <typename TensorLeft, typename TensorRight>
__aicore__ inline void GemmSbProcessKChunks(
    TensorLeft gmLeftTensor, TensorRight gmRightTensor, uint32_t kEff, uint32_t tileKChunk, uint64_t kL1Iter,
    uint64_t baseK, uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0, uint64_t& abL1LoopCnt,
    uint64_t& l0PingPong)
{
    using T = float;
    for (uint32_t kOff = 0; kOff < kEff; kOff += tileKChunk) {
        uint32_t curK = Min<uint32_t>(tileKChunk, kEff - kOff);
        uint64_t l1BufId = abL1LoopCnt & GEMM_SB_L1_BUF_MASK;
        uint64_t l1OffsetA = l1BufId * (GEMM_SB_L1_SIZE / GEMM_SB_L1_BUF_NUM);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, GEMM_SB_FRACTAL) * RoundUp<uint64_t>(curK, GEMM_SB_FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(T);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(l1OffsetA),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<GEMM_SB_FP32_C0>>(mL0, curK));
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(l1OffsetB),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<GEMM_SB_FP32_C0>>(curK, nL0));
        GemmSbProcessKChunk(
            gmLeftTensor, gmRightTensor, tensorAL1, tensorBL1, l1BufId, mOff, nOff, kOff, curK, mL0, nL0, kL1Iter,
            kOff / tileKChunk, baseK, l0PingPong);
        abL1LoopCnt++;
    }
}

template <typename TensorOut>
__aicore__ inline void GemmSbCopyOutL0C2GM(
    TensorOut gmOutTensor, uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0)
{
    auto gmBlockC = gmOutTensor.Slice(AscendC::Te::MakeCoord(mOff, nOff), AscendC::Te::MakeShape(mL0, nL0));
    auto tensorL0C = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<GEMM_SB_L0C_C0>>(mL0, nL0));
    AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{})
        .Call(gmBlockC, tensorL0C, AscendC::Te::FixpipeParams{GEMM_SB_FINAL_ACCUMULATION});
}

template <typename TensorLeft, typename TensorRight, typename TensorOut>
__aicore__ inline void GemmSbProcessTile(
    TensorLeft gmLeftTensor, TensorRight gmRightTensor, TensorOut gmOutTensor, uint32_t kEff, uint32_t tileKChunk,
    uint64_t kL1Iter, uint64_t baseK, uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd, uint32_t tileM,
    uint32_t tileN, uint64_t& abL1LoopCnt, uint64_t& l0PingPong)
{
    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GEMM_SB_PIPE_FLAG);
            GemmSbProcessKChunks(
                gmLeftTensor, gmRightTensor, kEff, tileKChunk, kL1Iter, baseK, mOff, nOff, curTileM, curTileN,
                abL1LoopCnt, l0PingPong);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(GEMM_SB_PIPE_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(GEMM_SB_PIPE_FLAG);
            GemmSbCopyOutL0C2GM(gmOutTensor, mOff, nOff, curTileM, curTileN);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(GEMM_SB_PIPE_FLAG);
        }
    }
}

} // namespace

// Compensated FP16 Cube operand transfers and FP32 accumulation.
namespace {

// FP16 input / FP32 accumulation pipeline for compensated complex products.
namespace {
constexpr int64_t CGEMM_HALF_L0A_SIZE = 64 * 1024;
constexpr uint16_t CGEMM_HALF_PIPE_FLAG = 0;
constexpr uint32_t CGEMM_HALF_FINAL_ACCUMULATION = 3;
constexpr uint32_t CGEMM_HALF_NON_FINAL_ACCUMULATION = 2;

constexpr uint64_t CGEMM_HALF_L1_SIZE = 512 * 1024;
constexpr uint64_t CGEMM_HALF_FP32_C0 = 16;
constexpr uint64_t CGEMM_HALF_FRACTAL = 16;
constexpr uint64_t CGEMM_HALF_L0C_C0 = 16;
constexpr uint64_t CGEMM_HALF_L1_BUF_NUM = 2;
constexpr uint64_t CGEMM_HALF_L1_BUF_MASK = CGEMM_HALF_L1_BUF_NUM - 1;
// Two alternating L0 banks retain operands until M releases them.
constexpr uint64_t CGEMM_HALF_L0_BUF_MASK = 1;
constexpr uint64_t CGEMM_HALF_HALF_L0_SIZE = CGEMM_HALF_L0A_SIZE / 2;

constexpr uint32_t CGEMM_HALF_VEC_BLOCK = 8; // 32B / sizeof(float): FP32 elements per aligned block
} // namespace

// ================================================================================
// GEMM kernel (AIC, tensor_api). Adapted from strmm_gemm_kernel: GM -> L1 (NZ/ZN) -> L0A/L0B ->
// AscendC::Te::Mmad -> L0C -> GM (Fixpipe). transLeft/transRight select the GM LayoutPtn (NDExt / DNExt); the
// L1 destination stays NZ (left) / ZN (right) and the routing auto-dispatches ND2Nz/DN2Nz/ND2Zn/DN2Zn.
// ================================================================================

template <typename TensorGM, typename TensorL1>
__aicore__ inline void CgemmHalfCopyGM2L1(
    TensorGM gmTensor, TensorL1 tensorL1, uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(AscendC::Te::MakeCoord(off0, off1), AscendC::Te::MakeShape(dim0, dim1));
    auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    AscendC::Te::Copy(copyGM2L1, tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void CgemmHalfL0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t curML1, uint64_t curKL1, uint64_t nL0, uint64_t kL1Iter,
    uint64_t iter0, uint64_t baseK, uint64_t& l0PingPong, bool initialize, bool finalize, uint64_t cOffset = 0)
{
    using T = half;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, baseK);
    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Offset = iter1 * baseK;
        uint64_t curKL0 = (kL0Offset + baseK > curKL1) ? (curKL1 - kL0Offset) : baseK;
        uint64_t l0BufId = l0PingPong & CGEMM_HALF_L0_BUF_MASK;
        uint64_t l0Offset = CGEMM_HALF_HALF_L0_SIZE * l0BufId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_HALF_FP32_C0>>(
            curML1, curKL0);
        auto tensorAL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, T>(l0Offset), layoutAL0);
        auto tensorBlockAL1 =
            tensorAL1.Slice(AscendC::Te::MakeCoord(0, kL0Offset), AscendC::Te::MakeShape(curML1, curKL0));
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{}), tensorAL0, tensorBlockAL1);
        auto layoutBL0 =
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<CGEMM_HALF_FP32_C0>>(curKL0, nL0);
        auto tensorBL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, T>(l0Offset), layoutBL0);
        auto tensorBlockBL1 =
            tensorBL1.Slice(AscendC::Te::MakeCoord(kL0Offset, 0), AscendC::Te::MakeShape(curKL0, nL0));
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{}), tensorBL0, tensorBlockBL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        bool isLastK = finalize && (iter0 + 1 == kL1Iter) && (iter1 + 1 == kL0Iter);
        bool isFirstK = initialize && (iter0 == 0 && iter1 == 0);
        uint8_t unitFlag = isLastK ? CGEMM_HALF_FINAL_ACCUMULATION : CGEMM_HALF_NON_FINAL_ACCUMULATION;
        AscendC::Te::MmadParams mmadParams{
            static_cast<uint16_t>(curML1), static_cast<uint16_t>(nL0), static_cast<uint16_t>(curKL0), unitFlag,
            isFirstK};
        auto layoutL0C =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_HALF_L0C_C0>>(curML1, nL0);
        auto tensorL0C =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(cOffset), layoutL0C);
        AscendC::Te::Mmad(
            AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(mmadParams), tensorL0C,
            tensorAL0, tensorBL0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }
}

// Two one-step products leave high A in the latest L0A slot and high B in
// the preceding L0B slot. Keep both slots busy through the final product.
__aicore__ inline void CgemmHalfReuseOperands(uint32_t cm, uint32_t cn, uint32_t ck, uint64_t l0, uint64_t cOffset)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    auto a = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, half>(
            ((l0 - 1) & CGEMM_HALF_L0_BUF_MASK) * CGEMM_HALF_HALF_L0_SIZE),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_HALF_FP32_C0>>(cm, ck));
    auto b = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, half>(
            ((l0 - 2) & CGEMM_HALF_L0_BUF_MASK) * CGEMM_HALF_HALF_L0_SIZE),
        AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<CGEMM_HALF_FP32_C0>>(ck, cn));
    auto c = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(cOffset),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_HALF_L0C_C0>>(cm, cn));
    AscendC::Te::MmadParams params{
        static_cast<uint16_t>(cm), static_cast<uint16_t>(cn), static_cast<uint16_t>(ck), CGEMM_HALF_FINAL_ACCUMULATION,
        false};
    AscendC::Te::Mmad(
        AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(params), c, a, b);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
}

template <typename TensorLeft, typename TensorRight, typename TensorAL1, typename TensorBL1>
__aicore__ inline void CgemmHalfProcessKChunk(
    TensorLeft gmLeftTensor, TensorRight gmRightTensor, TensorAL1 tensorAL1, TensorBL1 tensorBL1, uint64_t l1BufId,
    uint64_t mOff, uint64_t nOff, uint64_t kOff, uint64_t curK, uint64_t mL0, uint64_t nL0, uint64_t kL1Iter,
    uint64_t iter0, uint64_t baseK, uint64_t& l0PingPong, bool initialize, bool finalize, uint64_t cOffset = 0)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    CgemmHalfCopyGM2L1(gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    CgemmHalfCopyGM2L1(gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    CgemmHalfL0MmadLoop(
        tensorAL1, tensorBL1, mL0, curK, nL0, kL1Iter, iter0, baseK, l0PingPong, initialize, finalize, cOffset);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <typename TensorLeft, typename TensorRight>
__aicore__ inline void CgemmHalfProcessKChunks(
    TensorLeft gmLeftTensor, TensorRight gmRightTensor, uint32_t kEff, uint32_t tileKChunk, uint64_t kL1Iter,
    uint64_t baseK, uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0, uint64_t& abL1LoopCnt,
    uint64_t& l0PingPong, bool initialize = true, bool finalize = true, uint64_t cOffset = 0)
{
    using T = half;
    for (uint32_t kOff = 0; kOff < kEff; kOff += tileKChunk) {
        uint32_t curK = Min<uint32_t>(tileKChunk, kEff - kOff);
        uint64_t l1BufId = abL1LoopCnt & CGEMM_HALF_L1_BUF_MASK;
        uint64_t l1OffsetA = l1BufId * (CGEMM_HALF_L1_SIZE / CGEMM_HALF_L1_BUF_NUM);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, CGEMM_HALF_FRACTAL) * RoundUp<uint64_t>(curK, CGEMM_HALF_FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(T);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(l1OffsetA),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_HALF_FP32_C0>>(mL0, curK));
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, T>(l1OffsetB),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<CGEMM_HALF_FP32_C0>>(curK, nL0));
        CgemmHalfProcessKChunk(
            gmLeftTensor, gmRightTensor, tensorAL1, tensorBL1, l1BufId, mOff, nOff, kOff, curK, mL0, nL0, kL1Iter,
            kOff / tileKChunk, baseK, l0PingPong, initialize, finalize, cOffset);
        abL1LoopCnt++;
    }
}

template <typename TensorOut>
__aicore__ inline void CgemmHalfCopyOutL0C2GM(
    TensorOut gmOutTensor, uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0, uint64_t cOffset = 0)
{
    auto gmBlockC = gmOutTensor.Slice(AscendC::Te::MakeCoord(mOff, nOff), AscendC::Te::MakeShape(mL0, nL0));
    auto tensorL0C = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(cOffset),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<CGEMM_HALF_L0C_C0>>(mL0, nL0));
    AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{})
        .Call(gmBlockC, tensorL0C, AscendC::Te::FixpipeParams{CGEMM_HALF_FINAL_ACCUMULATION});
}

template <typename TensorLeft, typename TensorRight, typename TensorOut>
__aicore__ inline void CgemmHalfProcessTile(
    TensorLeft gmLeftTensor, TensorRight gmRightTensor, TensorOut gmOutTensor, uint32_t kEff, uint32_t tileKChunk,
    uint64_t kL1Iter, uint64_t baseK, uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd, uint32_t tileM,
    uint32_t tileN, uint64_t& abL1LoopCnt, uint64_t& l0PingPong)
{
    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(CGEMM_HALF_PIPE_FLAG);
            CgemmHalfProcessKChunks(
                gmLeftTensor, gmRightTensor, kEff, tileKChunk, kL1Iter, baseK, mOff, nOff, curTileM, curTileN,
                abL1LoopCnt, l0PingPong);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(CGEMM_HALF_PIPE_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(CGEMM_HALF_PIPE_FLAG);
            CgemmHalfCopyOutL0C2GM(gmOutTensor, mOff, nOff, curTileM, curTileN);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(CGEMM_HALF_PIPE_FLAG);
        }
    }
}

} // namespace
