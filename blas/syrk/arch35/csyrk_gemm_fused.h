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
 * \file csyrk_gemm_fused.h
 * \brief Csyrk-specific fused-4M cube GEMM drivers (BM=BN=64, single-owner tiles).
 *
 *        Q0 = Ar*Ar^T  Q1 = Ai*Ai^T  Q2 = Ar*Ai^T  Q3 = Ai*Ar^T, all four quads
 *        of each owned 64x64 C tile computed back to back with dual L0C acc-set
 *        ping-pong and K-chunk L1/L0 pipelining. Two variants:
 *          - fp32: four fp32 mmads per K sub-block;
 *          - HF32x3: three compensated mmads per quad (xh*yh + xl*yh + xh*yl).
 *        These are csyrk-private; they include the shared syrk GEMM helper only
 *        for the tensor_api (`te::`) machinery and the NZ/ZN layout constants.
 */

#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "common/helper/syrk_gemm_arch35.h"

namespace te = AscendC::Te;

// Pre-set / drain the fused-4M driver event flags. The fp32 driver also uses four
// M_MTE1 slots (L0 single-set ping-pong); the HF32x3 driver uses one.
__aicore__ inline void SyrkGemmFusedFlagsInit(bool fourL0Slots)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    if (fourL0Slots) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
    }
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
}

__aicore__ inline void SyrkGemmFusedFlagsDrain(bool fourL0Slots)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    if (fourL0Slots) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
    }
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
}

// Kept-index residue C-tile selection shared by the fused-4M drivers: returns
// true when the calling core owns tile (mStart, nStart); outputs the tile base
// (mStart/nStart), the clamped extents (curM/curN) and the L0C acc-set base.
__aicore__ inline bool SyrkGemmFusedTileSelect(
    uint64_t tileIdx, uint64_t divN, uint32_t n, uint32_t triangleMode, uint64_t kept, uint64_t blockNum,
    uint64_t coreStart, uint64_t tileCount, uint32_t& mStart, uint32_t& nStart, uint32_t& curM, uint32_t& curN,
    uint64_t& setBase)
{
    constexpr uint32_t BM64 = 64;
    if (divN == 0 || blockNum == 0) {
        return false;
    }
    uint64_t cm = tileIdx / divN;
    uint64_t cn = tileIdx % divN;
    if (cm % 2 == 1) {
        cn = divN - 1 - cn; // serpentine n-order keeps L1 panel reuse locality
    }
    mStart = static_cast<uint32_t>(cm * BM64);
    nStart = static_cast<uint32_t>(cn * BM64);
    if (triangleMode == 1 && mStart > nStart + BM64 - 1) {
        return false;
    } // below UPPER band
    if (triangleMode == 2 && nStart > mStart + BM64 - 1) {
        return false;
    } // above LOWER band
    if ((kept % blockNum) != coreStart) {
        return false;
    } // not this core's tile
    curM = Min<uint32_t>(BM64, n - mStart);
    curN = Min<uint32_t>(BM64, n - nStart);
    setBase = (tileCount & 0x1) ? 64 * 1024 : 0; // two L0C acc sets ping-pong
    return true;
}

// Drain four L0C accumulators (one per quad) to the four temp quadrants.
template <typename TensorTemp, typename CopyL0C2GM>
__aicore__ inline void SyrkGemmDrainQuads(
    uint64_t l0cSetBase, uint32_t mStart, uint32_t nStart, uint32_t curM, uint32_t curN, TensorTemp gmTempQ0,
    TensorTemp gmTempQ1, TensorTemp gmTempQ2, TensorTemp gmTempQ3, CopyL0C2GM copyL0C2GMAtom)
{
    using T = float;
    constexpr uint32_t ACC_STRIDE = 16 * 1024; // 64x64 fp32 NZ
    TensorTemp quads[4] = {gmTempQ0, gmTempQ1, gmTempQ2, gmTempQ3};
    for (uint32_t q = 0; q < 4; q++) {
        auto acc = te::MakeTensor(
            te::MakeMemPtr<te::Location::L0C, T>(l0cSetBase + q * ACC_STRIDE),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curM, curN));
        auto dst = quads[q].Slice(te::MakeCoord(mStart, nStart), te::MakeShape(curM, curN));
        copyL0C2GMAtom.Call(dst, acc, te::FixpipeParams{3});
    }
}

// ==========================================================================
// BM=BN=64 fused-4M single-owner tile (K-chunk panels read once, 4 L0C accs).
// 64x64 output tile: 4 quads accumulate in 4 L0C accs of 16 KB (NZ C0=16).
// Two acc SETS (0/64KB) ping-pong across C-tiles so tile t+1's mmad overlaps
// tile t's four fixpipes (drain stall of the 4x64KB/256KB-full BM=128 variant
// is gone). Per C-tile: one K pass; each K-chunk reads the four input panels
// (Ar[m], Ai[m], Ar[n], Ai[n]) once from GM into L1 (4x16 KB in a 128 KB L1
// buf), then baseK=8 sub-blocks double-buffer into L0A/L0B slot pairs and run
// the four quad mmads. Single owner per C tile (the AIC/AIV mix combine
// prerequisite); temp bytes per tile identical to separate per-quad GEMMs.
// ==========================================================================
// The four fp32 fused-4M quad mmads (Ar*Ar, Ai*Ai, Ar*Ai, Ai*Ar) into 4 L0C accs.
template <typename MmadAtom, typename MmadParams, typename TensorL0A, typename TensorL0B>
__aicore__ inline void SyrkGemmQuadFusedFp32Mmad(
    MmadAtom mmadAtom, MmadParams p, uint32_t curM, uint32_t curN, uint64_t l0cSetBase, TensorL0A aAr, TensorL0A aAi,
    TensorL0B bAr, TensorL0B bAi)
{
    using T = float;
    constexpr uint32_t ACC_STRIDE = 16 * 1024; // 64x64 fp32 NZ
    auto acc0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, T>(l0cSetBase + 0 * ACC_STRIDE),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curM, curN));
    auto acc1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, T>(l0cSetBase + 1 * ACC_STRIDE),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curM, curN));
    auto acc2 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, T>(l0cSetBase + 2 * ACC_STRIDE),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curM, curN));
    auto acc3 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, T>(l0cSetBase + 3 * ACC_STRIDE),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curM, curN));
    te::Mmad(mmadAtom.with(p), acc0, aAr, bAr);
    te::Mmad(mmadAtom.with(p), acc1, aAi, bAi);
    te::Mmad(mmadAtom.with(p), acc2, aAr, bAi);
    te::Mmad(mmadAtom.with(p), acc3, aAi, bAr);
}

// L1 -> L0A (Ar/Ai M panels) and L0B (Ar/Ai N panels) for one BASE_K sub-block.
template <typename TensorL1A, typename TensorL1B, typename CopyA, typename CopyB>
__aicore__ inline void SyrkGemmFp32L1ToL0(
    uint64_t l0Off, uint32_t kL0Off, uint32_t curM, uint32_t curN, uint32_t curKL0, CopyA copyL12L0A, CopyB copyL12L0B,
    TensorL1A l1ArM, TensorL1A l1AiM, TensorL1B l1ArN, TensorL1B l1AiN)
{
    using T = float;
    constexpr uint32_t L0_SLOT = 16 * 1024;
    {
        auto aL0 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L0A, T>(l0Off),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curKL0));
        te::Copy(copyL12L0A, aL0, l1ArM.Slice(te::MakeCoord(0, kL0Off), te::MakeShape(curM, curKL0)));
    }
    {
        auto aL0 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L0A, T>(l0Off + L0_SLOT),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curKL0));
        te::Copy(copyL12L0A, aL0, l1AiM.Slice(te::MakeCoord(0, kL0Off), te::MakeShape(curM, curKL0)));
    }
    {
        auto bL0 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L0B, T>(l0Off),
            te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curKL0, curN));
        te::Copy(copyL12L0B, bL0, l1ArN.Slice(te::MakeCoord(kL0Off, 0), te::MakeShape(curKL0, curN)));
    }
    {
        auto bL0 = te::MakeTensor(
            te::MakeMemPtr<te::Location::L0B, T>(l0Off + L0_SLOT),
            te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curKL0, curN));
        te::Copy(copyL12L0B, bL0, l1AiN.Slice(te::MakeCoord(kL0Off, 0), te::MakeShape(curKL0, curN)));
    }
}

// One BASE_K sub-block of the fp32 fused-4M tile: double-buffered L1->L0 load of
// the four panels and the four quad mmads (dual L0 ping-pong slot).
template <typename TensorL1A, typename TensorL1B>
__aicore__ inline void SyrkGemmQuadFusedFp32L0(
    uint32_t K, uint32_t kOff, uint32_t kb, uint32_t kL0Iter, uint32_t curK, uint32_t mStart, uint32_t nStart,
    uint32_t curM, uint32_t curN, uint64_t l0cSetBase, uint64_t& l0PingPong, TensorL1A l1ArM, TensorL1A l1AiM,
    TensorL1B l1ArN, TensorL1B l1AiN)
{
    using T = float;
#ifndef SYRK_BASE_K
#define SYRK_BASE_K 64
#endif
    constexpr uint32_t BASE_K = SYRK_BASE_K;
    constexpr uint32_t ACC_STRIDE = 16 * 1024; // 64x64 fp32 NZ
    constexpr uint32_t L0_SLOT = 16 * 1024;
    auto copyL12L0A = te::MakeCopy(te::CopyL12L0A{});
    auto copyL12L0B = te::MakeCopy(te::CopyL12L0B{});
    auto mmadAtom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};
    (void)mStart;
    (void)nStart;
    uint32_t kL0Off = kb * BASE_K;
    uint32_t curKL0 = (kL0Off + BASE_K > curK) ? (curK - kL0Off) : BASE_K;
    uint64_t slotId = l0PingPong & 0x1;
    uint64_t l0Off = slotId * 2 * L0_SLOT; // pair at 0 or 32 KB
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(slotId);
    SyrkGemmFp32L1ToL0(l0Off, kL0Off, curM, curN, curKL0, copyL12L0A, copyL12L0B, l1ArM, l1AiM, l1ArN, l1AiN);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slotId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slotId);
    bool isFirstK = (kOff == 0 && kb == 0);
    bool isLastK = (kOff + curK == K) && (kb + 1 == kL0Iter);
    uint8_t unitFlag = isLastK ? 3 : 2;
    auto aAr = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0A, T>(l0Off),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curKL0));
    auto aAi = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0A, T>(l0Off + L0_SLOT),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curKL0));
    auto bAr = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0B, T>(l0Off),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curKL0, curN));
    auto bAi = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0B, T>(l0Off + L0_SLOT),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curKL0, curN));
    te::MmadParams p{
        static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(curKL0), unitFlag, isFirstK};
    SyrkGemmQuadFusedFp32Mmad(mmadAtom, p, curM, curN, l0cSetBase, aAr, aAi, bAr, bAi);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slotId);
    l0PingPong++;
}

// GM -> L1: the four fp32 fused-4M panels (Ar/Ai M-side NZ, Ar/Ai N-side ZN).
template <typename CopyGM2L1, typename TensorA, typename TensorB, typename TensorL1A, typename TensorL1B>
__aicore__ inline void SyrkGemmFp32GM2L1(
    CopyGM2L1 copyGM2L1, uint32_t kOff, uint32_t mStart, uint32_t nStart, uint32_t curM, uint32_t curN, uint32_t curK,
    TensorL1A l1ArM, TensorL1A l1AiM, TensorL1B l1ArN, TensorL1B l1AiN, TensorA gmLeftAr, TensorA gmLeftAi,
    TensorB gmRightAr, TensorB gmRightAi)
{
    te::Copy(copyGM2L1, l1ArM, gmLeftAr.Slice(te::MakeCoord(mStart, kOff), te::MakeShape(curM, curK)));
    te::Copy(copyGM2L1, l1AiM, gmLeftAi.Slice(te::MakeCoord(mStart, kOff), te::MakeShape(curM, curK)));
    te::Copy(copyGM2L1, l1ArN, gmRightAr.Slice(te::MakeCoord(kOff, nStart), te::MakeShape(curK, curN)));
    te::Copy(copyGM2L1, l1AiN, gmRightAi.Slice(te::MakeCoord(kOff, nStart), te::MakeShape(curK, curN)));
}

// One K-chunk of the fp32 fused-4M tile: load the four panels GM->L1->L0 and run
// the four quad mmads, then release L1. isFirstK zeroes the accumulators.
template <typename TensorA, typename TensorB>
__aicore__ inline void SyrkGemmQuadFusedFp32Chunk(
    TensorA gmLeftAr, TensorA gmLeftAi, TensorB gmRightAr, TensorB gmRightAi, uint32_t K, uint32_t kOff,
    uint32_t mStart, uint32_t nStart, uint32_t curM, uint32_t curN, uint64_t l0cSetBase, uint64_t& abL1LoopCnt,
    uint64_t& l0PingPong)
{
    using T = float;
    constexpr uint32_t BM = 64;
#ifndef SYRK_CHUNK_K
#define SYRK_CHUNK_K 128
#endif
    constexpr uint32_t CHUNK_K = SYRK_CHUNK_K;
#ifndef SYRK_BASE_K
#define SYRK_BASE_K 64
#endif
    constexpr uint32_t BASE_K = SYRK_BASE_K;
    constexpr uint32_t ACC_STRIDE = 16 * 1024; // 64x64 fp32 NZ
    constexpr uint32_t L1_BUF = 4;
    constexpr uint32_t PANEL_B = BM * CHUNK_K * sizeof(T);
    constexpr uint32_t L1_OFF = 128 * 1024;
    constexpr uint32_t L0_SLOT = 16 * 1024;
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    auto copyL12L0A = te::MakeCopy(te::CopyL12L0A{});
    auto copyL12L0B = te::MakeCopy(te::CopyL12L0B{});
    auto mmadAtom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};
    uint32_t curK = Min<uint32_t>(CHUNK_K, K - kOff);
    uint64_t l1BufId = abL1LoopCnt & (L1_BUF - 1);
    uint64_t l1Off = l1BufId * L1_OFF;
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    auto l1ArM = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1Off),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curK));
    auto l1AiM = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1Off + PANEL_B),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curK));
    auto l1ArN = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1Off + 2 * PANEL_B),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curK, curN));
    auto l1AiN = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, T>(l1Off + 3 * PANEL_B),
        te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curK, curN));
    SyrkGemmFp32GM2L1(
        copyGM2L1, kOff, mStart, nStart, curM, curN, curK, l1ArM, l1AiM, l1ArN, l1AiN, gmLeftAr, gmLeftAi, gmRightAr,
        gmRightAi);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    uint32_t kL0Iter = CeilDiv<uint32_t>(curK, BASE_K);
    for (uint32_t kb = 0; kb < kL0Iter; kb++) {
        SyrkGemmQuadFusedFp32L0(
            K, kOff, kb, kL0Iter, curK, mStart, nStart, curM, curN, l0cSetBase, l0PingPong, l1ArM, l1AiM, l1ArN, l1AiN);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    abL1LoopCnt++;
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void SyrkGemmProcessTileQuadFused64(
    TensorA gmLeftAr, TensorA gmLeftAi, TensorB gmRightAr, TensorB gmRightAi, TensorTemp gmTempQ0, TensorTemp gmTempQ1,
    TensorTemp gmTempQ2, TensorTemp gmTempQ3, uint32_t K, uint32_t mStart, uint32_t nStart, uint32_t curM,
    uint32_t curN, uint64_t l0cSetBase, uint64_t& abL1LoopCnt, uint64_t& l0PingPong, uint32_t l1BufNum, int64_t l1Size)
{
    using T = float;
    constexpr uint32_t ACC_STRIDE = 16 * 1024; // 64x64 fp32 NZ
    auto copyL0C2GMAtom = te::MakeCopy(te::CopyL0C2GM{});
    // Wait for the same-set tile two tiles ago to release the L0C accumulators.
    const uint64_t setParity = (l0cSetBase / (4 * ACC_STRIDE)) & 0x1;
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(setParity);
    for (uint32_t kOff = 0; kOff < K; kOff += SYRK_CHUNK_K) {
        SyrkGemmQuadFusedFp32Chunk(
            gmLeftAr, gmLeftAi, gmRightAr, gmRightAi, K, kOff, mStart, nStart, curM, curN, l0cSetBase, abL1LoopCnt,
            l0PingPong);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(setParity);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(setParity);
    SyrkGemmDrainQuads(l0cSetBase, mStart, nStart, curM, curN, gmTempQ0, gmTempQ1, gmTempQ2, gmTempQ3, copyL0C2GMAtom);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(setParity);
}

// HF32x3 compensated fused-4M tile (BM=64, CHUNK_K=64): the cube runs in HF32
// mode (hardware rounds every fp32 L0A/L0B operand to 10-bit mantissa), and the
// lost mantissa is repaired by three mmads per quad:
//   Q = xh*yh + xl*yh + xh*yl
// where xh = HF32(x) (the hardware rounding of the full-precision input) and
// xl = x - xh is the residual the deinterleave pre-computed.
// L1 buffer: 8 panels of 64x64 fp32 = 128 KB per buffer (4 buffers = 512 KB),
// CHUNK_K = 64 so each L1 buffer holds one full chunk of the 8 sources.
// L0A/L0B each hold the 4 M/N panels (no intra-chunk ping-pong); the four L0C
// accumulators keep the cross-tile dual-set ping-pong (l0cSetBase).
// HF32x3 L1 load: 8 panels (M: arH/aiH/arL/aiL at 0..3, N: at 4..7) for one chunk.
template <typename TensorA, typename TensorB>
__aicore__ inline void SyrkGemmQuadFusedHf32x3L1(
    uint32_t kOff, uint32_t curM, uint32_t curN, uint32_t curK, uint64_t l1Off, uint32_t mStart, uint32_t nStart,
    TensorA gmLeftArH, TensorA gmLeftAiH, TensorA gmLeftArL, TensorA gmLeftAiL, TensorB gmRightArH, TensorB gmRightAiH,
    TensorB gmRightArL, TensorB gmRightAiL)
{
    using T = float;
    constexpr uint32_t BM = 64;
    constexpr uint32_t CHUNK_K = 64;
    constexpr uint32_t PANEL = BM * CHUNK_K * sizeof(T); // 64x64 = 16 KB
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    TensorA left[4] = {gmLeftArH, gmLeftAiH, gmLeftArL, gmLeftAiL};
    for (uint32_t q = 0; q < 4; q++) {
        te::Copy(
            copyGM2L1,
            te::MakeTensor(
                te::MakeMemPtr<te::Location::L1, T>(l1Off + q * PANEL),
                te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curK)),
            left[q].Slice(te::MakeCoord(mStart, kOff), te::MakeShape(curM, curK)));
    }
    TensorB right[4] = {gmRightArH, gmRightAiH, gmRightArL, gmRightAiL};
    for (uint32_t q = 0; q < 4; q++) {
        te::Copy(
            copyGM2L1,
            te::MakeTensor(
                te::MakeMemPtr<te::Location::L1, T>(l1Off + (4 + q) * PANEL),
                te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curK, curN)),
            right[q].Slice(te::MakeCoord(kOff, nStart), te::MakeShape(curK, curN)));
    }
}

// The 12 HF32x3 mmads: L0A panels 0=arH,1=aiH,2=arL,3=aiL (M), same on L0B (N).
// Each quad c has 3 ordered mmads; role 0 zeroes the acc on the first K chunk
// (initAcc) and role 2 gets unitFlag 3 on the last chunk to hold the L0C result.
__aicore__ inline void SyrkGemmQuadFusedHf32x3MmadTerms(
    uint32_t curM, uint32_t curN, uint32_t curK, uint64_t l0cSetBase, bool first, bool last)
{
    using T = float;
    constexpr uint32_t ACC_STRIDE = 16 * 1024;
    constexpr uint32_t L0_PANEL = 16 * 1024;
    auto mmadAtom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};
    struct {
        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint8_t role;
    } terms[12] = {
        {2, 0, 0, 0}, {0, 2, 0, 1}, {0, 0, 0, 2}, {3, 1, 1, 0}, {1, 3, 1, 1}, {1, 1, 1, 2},
        {2, 1, 2, 0}, {0, 3, 2, 1}, {0, 1, 2, 2}, {3, 0, 3, 0}, {1, 2, 3, 1}, {1, 0, 3, 2},
    };
    for (int i = 0; i < 12; i++) {
        bool initAcc = first && (terms[i].role == 0);
        uint8_t unit = (last && terms[i].role == 2) ? 3 : 2;
        te::MmadParams p{
            static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(curK), unit, initAcc};
        auto acc = te::MakeTensor(
            te::MakeMemPtr<te::Location::L0C, T>(l0cSetBase + terms[i].c * ACC_STRIDE),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_L0C_C0>>(curM, curN));
        auto aA = te::MakeTensor(
            te::MakeMemPtr<te::Location::L0A, T>(terms[i].a * L0_PANEL),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curK));
        auto bB = te::MakeTensor(
            te::MakeMemPtr<te::Location::L0B, T>(terms[i].b * L0_PANEL),
            te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curK, curN));
        te::Mmad(mmadAtom.with(p), acc, aA, bB);
    }
}

// HF32x3 mmad phase: L1->L0A/L0B (4+4 panels) then the 12 ordered hi/lo mmads.
__aicore__ inline void SyrkGemmQuadFusedHf32x3Mmad(
    uint32_t kOff, uint32_t curM, uint32_t curN, uint32_t curK, uint32_t K, uint64_t l1Off, uint64_t l0cSetBase)
{
    using T = float;
    constexpr uint32_t BM = 64;
    constexpr uint32_t CHUNK_K = 64;
    constexpr uint32_t PANEL = BM * CHUNK_K * sizeof(T);
    constexpr uint32_t ACC_STRIDE = 16 * 1024;
    constexpr uint32_t L0_PANEL = 16 * 1024;
    auto copyL12L0A = te::MakeCopy(te::CopyL12L0A{});
    auto copyL12L0B = te::MakeCopy(te::CopyL12L0B{});
    auto mmadAtom = te::MmadAtom<te::MmadTraits<te::MmadOperation>>{};
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    for (uint32_t q = 0; q < 4; q++) {
        te::Copy(
            copyL12L0A,
            te::MakeTensor(
                te::MakeMemPtr<te::Location::L0A, T>(q * L0_PANEL),
                te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curK)),
            te::MakeTensor(
                te::MakeMemPtr<te::Location::L1, T>(l1Off + q * PANEL),
                te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curM, curK)));
    }
    for (uint32_t q = 0; q < 4; q++) {
        te::Copy(
            copyL12L0B,
            te::MakeTensor(
                te::MakeMemPtr<te::Location::L0B, T>(q * L0_PANEL),
                te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curK, curN)),
            te::MakeTensor(
                te::MakeMemPtr<te::Location::L1, T>(l1Off + (4 + q) * PANEL),
                te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<SYRK_ARCH35_FP32_C0>>(curK, curN)));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    bool first = (kOff == 0);
    bool last = (kOff + curK == K);
    SyrkGemmQuadFusedHf32x3MmadTerms(curM, curN, curK, l0cSetBase, first, last);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void SyrkGemmProcessTileQuadFused64Hf32x3(
    TensorA gmLeftArH, TensorA gmLeftAiH, TensorA gmLeftArL, TensorA gmLeftAiL, TensorB gmRightArH, TensorB gmRightAiH,
    TensorB gmRightArL, TensorB gmRightAiL, TensorTemp gmTempQ0, TensorTemp gmTempQ1, TensorTemp gmTempQ2,
    TensorTemp gmTempQ3, uint32_t K, uint32_t mStart, uint32_t nStart, uint32_t curM, uint32_t curN,
    uint64_t l0cSetBase, uint64_t& abL1LoopCnt)
{
    using T = float;
    constexpr uint32_t BM = 64;
    constexpr uint32_t CHUNK_K = 64;
    constexpr uint32_t ACC_STRIDE = 16 * 1024;           // 64x64 fp32 NZ
    constexpr uint32_t L1_BUF = 4;                       // 4 x 128 KB
    constexpr uint32_t PANEL = BM * CHUNK_K * sizeof(T); // 64x64 = 16 KB
    constexpr uint32_t L1_OFF = 8 * PANEL;               // 8 panels per buffer
    auto copyL0C2GMAtom = te::MakeCopy(te::CopyL0C2GM{});
    const uint64_t setParity = (l0cSetBase / (4 * ACC_STRIDE)) & 0x1;
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(setParity);
    for (uint32_t kOff = 0; kOff < K; kOff += CHUNK_K) {
        uint32_t curK = Min<uint32_t>(CHUNK_K, K - kOff);
        uint64_t l1BufId = abL1LoopCnt & (L1_BUF - 1);
        uint64_t l1Off = l1BufId * L1_OFF;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        SyrkGemmQuadFusedHf32x3L1(
            kOff, curM, curN, curK, l1Off, mStart, nStart, gmLeftArH, gmLeftAiH, gmLeftArL, gmLeftAiL, gmRightArH,
            gmRightAiH, gmRightArL, gmRightAiL);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        SyrkGemmQuadFusedHf32x3Mmad(kOff, curM, curN, curK, K, l1Off, l0cSetBase);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        abL1LoopCnt++;
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(setParity);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(setParity);
    SyrkGemmDrainQuads(l0cSetBase, mStart, nStart, curM, curN, gmTempQ0, gmTempQ1, gmTempQ2, gmTempQ3, copyL0C2GMAtom);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(setParity);
}

// Shared banded fused-4M driver: kept-index residue C-tile distribution with
// triangular skip over launched cores; each owned tile is dispatched to the
// fp32 or HF32x3 per-tile processor. UseHf32 / FourL0Slots are compile-time so
// only the needed path is instantiated (fp32 uses four M_MTE1 event slots,
// HF32x3 one; the L (residual) tensors are unused on the fp32 path).
template <bool UseHf32, bool FourL0Slots, typename TensorA, typename TensorB, typename TensorTemp, typename TilingData>
__aicore__ inline void SyrkGemmQuadFusedDriver(
    TensorA lAr, TensorA lAi, TensorA lArL, TensorA lAiL, TensorB rAr, TensorB rAi, TensorB rArL, TensorB rAiL,
    TensorTemp gmTempQ0, TensorTemp gmTempQ1, TensorTemp gmTempQ2, TensorTemp gmTempQ3, const TilingData& tiling,
    uint32_t triangleMode, int64_t coreIdx, int64_t coreNum)
{
    uint32_t n = tiling.n;
    uint32_t K = tiling.k;
    constexpr uint32_t BM64 = 64;
    uint32_t divM = CeilDiv<uint32_t>(n, BM64);
    uint32_t divN = CeilDiv<uint32_t>(n, BM64);
    uint64_t totalTiles = static_cast<uint64_t>(divM) * static_cast<uint64_t>(divN);
    SyrkGemmFusedFlagsInit(FourL0Slots);
    uint64_t abL1LoopCnt = 0;
    uint64_t l0PingPong = 0;
    uint64_t blockNum = (coreNum > 0) ? static_cast<uint64_t>(coreNum) : static_cast<uint64_t>(AscendC::GetBlockNum());
    uint64_t coreStart =
        (coreIdx >= 0) ? static_cast<uint64_t>(coreIdx) : static_cast<uint64_t>(AscendC::GetBlockIdx());
    uint64_t kept = 0;
    uint64_t tileCount = 0;
    for (uint64_t tileIdx = 0; tileIdx < totalTiles; tileIdx++) {
        uint32_t mStart, nStart, curM, curN;
        uint64_t setBase;
        bool owned = SyrkGemmFusedTileSelect(
            tileIdx, divN, n, triangleMode, kept, blockNum, coreStart, tileCount, mStart, nStart, curM, curN, setBase);
        kept++;
        if (!owned) {
            continue;
        }
        if constexpr (UseHf32) {
            SyrkGemmProcessTileQuadFused64Hf32x3(
                lAr, lAi, lArL, lAiL, rAr, rAi, rArL, rAiL, gmTempQ0, gmTempQ1, gmTempQ2, gmTempQ3, K, mStart, nStart,
                curM, curN, setBase, abL1LoopCnt);
        } else {
            SyrkGemmProcessTileQuadFused64(
                lAr, lAi, rAr, rAi, gmTempQ0, gmTempQ1, gmTempQ2, gmTempQ3, K, mStart, nStart, curM, curN, setBase,
                abL1LoopCnt, l0PingPong, 4, 512 * 1024);
        }
        tileCount++;
    }
    SyrkGemmFusedFlagsDrain(FourL0Slots);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// HF32x3 banded fused-4M driver: kept-index residue C-tile distribution, each
// owned tile processes four quads with three HF32 mmads apiece. Same tile walk
// as the fp32 driver (triangle skip + balance), so temp layout is identical.
template <typename TensorA, typename TensorB, typename TensorTemp, typename TilingData>
__aicore__ inline void SyrkGemmQuadFusedTileKernelImpl64Hf32x3(
    TensorA gmLeftArH, TensorA gmLeftAiH, TensorA gmLeftArL, TensorA gmLeftAiL, TensorB gmRightArH, TensorB gmRightAiH,
    TensorB gmRightArL, TensorB gmRightAiL, TensorTemp gmTempQ0, TensorTemp gmTempQ1, TensorTemp gmTempQ2,
    TensorTemp gmTempQ3, const TilingData& tiling, uint32_t triangleMode, int64_t coreIdx, int64_t coreNum)
{
    SyrkGemmQuadFusedDriver<true, false>(
        gmLeftArH, gmLeftAiH, gmLeftArL, gmLeftAiL, gmRightArH, gmRightAiH, gmRightArL, gmRightAiL, gmTempQ0, gmTempQ1,
        gmTempQ2, gmTempQ3, tiling, triangleMode, coreIdx, coreNum);
}

// distribution over launched cores; each owned C tile's 64x64 base tiles
// alternate the two L0C acc sets (base 0 and 64 KB) so a 64x64 tile's fixpipe
// of set X overlaps the next tile's mmad of set Y (the BM=128 4x64KB-full
// variant could not ping-pong and stalled on tile drain). Flag pre-sets/drain
// mirror the other drivers (MTE1/MTE2 self-balanced per K chunk; FIX_M(0/1)
// pre-set once, each 64x64 tile consumes+produces its parity once).
template <typename TensorA, typename TensorB, typename TensorTemp, typename TilingData>
__aicore__ inline void SyrkGemmQuadFusedTileKernelImpl64(
    TensorA gmLeftAr, TensorA gmLeftAi, TensorB gmRightAr, TensorB gmRightAi, TensorTemp gmTempQ0, TensorTemp gmTempQ1,
    TensorTemp gmTempQ2, TensorTemp gmTempQ3, const TilingData& tiling, uint32_t triangleMode, int64_t coreIdx,
    int64_t coreNum)
{
    // The fp32 path has no residual inputs; the driver's unused L slots reuse
    // the high tensors.
    SyrkGemmQuadFusedDriver<false, true>(
        gmLeftAr, gmLeftAi, gmLeftAr, gmLeftAi, gmRightAr, gmRightAi, gmRightAr, gmRightAi, gmTempQ0, gmTempQ1,
        gmTempQ2, gmTempQ3, tiling, triangleMode, coreIdx, coreNum);
}
