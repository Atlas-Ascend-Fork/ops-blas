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
#include "cgemm_kernel_common.h"

constexpr uint32_t CGEMM_CUBE_BLOCK = CGEMM_CUBE_FRACTAL;

template <typename T>
__aicore__ inline void CgemmCopyInA1(
    const AscendC::GlobalTensor<T>& gm, const AscendC::LocalTensor<T>& l1, uint32_t first, uint32_t second,
    uint32_t srcStride)
{
    AscendC::Nd2NzParams p{};
    p.ndNum = 1;
    p.nValue = first;
    p.dValue = second;
    p.srcNdMatrixStride = 1;
    p.srcDValue = srcStride;
    p.dstNzC0Stride = RoundUp<uint32_t>(first, CGEMM_CUBE_BLOCK);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    AscendC::DataCopy(l1, gm, p);
}

template <typename T>
__aicore__ inline void CgemmCopyInB1(
    const AscendC::GlobalTensor<T>& gm, const AscendC::LocalTensor<T>& l1, uint32_t first, uint32_t second,
    uint32_t srcStride)
{
    AscendC::Nd2NzParams p{};
    p.ndNum = 1;
    p.nValue = second;
    p.dValue = first;
    p.srcNdMatrixStride = 1;
    p.srcDValue = srcStride;
    p.dstNzC0Stride = RoundUp<uint32_t>(second, CGEMM_CUBE_BLOCK);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 1;
    AscendC::DataCopy(l1, gm, p);
}

template <typename T>
__aicore__ inline void CgemmCopyInA2(
    const AscendC::LocalTensor<T>& l0, const AscendC::LocalTensor<T>& l1, uint32_t curM, uint32_t curKChunk,
    uint32_t curK, bool trans)
{
    AscendC::LoadData2DParamsV2 p{};
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    if (trans) {
        p.mStep = CeilDiv<uint32_t>(curK, CGEMM_CUBE_BLOCK);
        p.kStep = RoundUp<uint32_t>(CeilDiv(curM, GEMM_FP32_C0), 2);
        p.srcStride = CeilDiv<uint32_t>(curKChunk, CGEMM_CUBE_BLOCK);
        p.dstStride = CeilDiv<uint32_t>(curM, CGEMM_CUBE_BLOCK);
        p.ifTranspose = true;
        AscendC::LoadData<T>(l0, l1, p);
        return;
    }
    p.mStep = CeilDiv<uint32_t>(curM, CGEMM_CUBE_BLOCK);
    p.kStep = CeilDiv(curK, GEMM_FP32_C0);
    p.srcStride = CeilDiv<uint32_t>(curM, CGEMM_CUBE_BLOCK);
    p.dstStride = p.mStep;
    p.ifTranspose = trans;
    AscendC::LoadData<T>(l0, l1, p);
}

template <typename T>
__aicore__ inline void CgemmCopyInB2(
    const AscendC::LocalTensor<T>& l0, const AscendC::LocalTensor<T>& l1, uint32_t curN, uint32_t curKChunk,
    uint32_t curK, bool trans)
{
    AscendC::LoadData2DParamsV2 p{};
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    if (trans) {
        p.mStep = CeilDiv<uint32_t>(curN, CGEMM_CUBE_BLOCK);
        p.kStep = CeilDiv<uint32_t>(curK, GEMM_FP32_C0);
        p.srcStride = CeilDiv<uint32_t>(curN, CGEMM_CUBE_BLOCK);
        p.dstStride = p.srcStride;
        p.ifTranspose = false;
        AscendC::LoadData<T>(l0, l1, p);
        return;
    }
    p.mStep = CeilDiv<uint32_t>(curK, CGEMM_CUBE_BLOCK);
    p.kStep = CeilDiv<uint32_t>(curN, CGEMM_CUBE_BLOCK) * 2;
    p.dstStride = p.kStep >> 1;
    p.srcStride = CeilDiv<uint32_t>(curKChunk, CGEMM_CUBE_BLOCK);
    p.ifTranspose = true;
    AscendC::LoadData<T>(l0, l1, p);
}

__aicore__ inline void CgemmMmad(
    const AscendC::LocalTensor<float>& c, const AscendC::LocalTensor<float>& a, const AscendC::LocalTensor<float>& b,
    uint32_t m, uint32_t n, uint32_t k, bool first)
{
    AscendC::MmadParams p{};
    p.m = RoundUp<uint32_t>(m, CGEMM_CUBE_BLOCK);
    p.n = n;
    p.k = k;
    p.cmatrixSource = false;
    p.cmatrixInitVal = first;
    p.unitFlag = 0;
    p.disableGemv = true;
    AscendC::Mmad(c, a, b, p);
}

template <uint32_t BS>
__aicore__ inline void CgemmFixpipe(
    AscendC::GlobalTensor<float>& gm, const AscendC::LocalTensor<float>& c, uint64_t offset, uint32_t m, uint32_t n,
    uint32_t ldc)
{
    AscendC::DataCopyCO12DstParams p{};
    p.nSize = n;
    p.mSize = m;
    p.dstStride = ldc;
    p.srcStride = RoundUp<uint32_t>(m, CGEMM_CUBE_BLOCK);
    p.quantPre = QuantMode_t::NoQuant;
    p.reluPre = 0;
    p.nz2ndEn = true;
    p.unitFlag = 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(gm[offset], c, p);
}

template <bool IsTransA>
__aicore__ __inline__ __attribute__((always_inline)) void CopyCgemmLeftPlane(
    const AscendC::GlobalTensor<float>& gm, const AscendC::LocalTensor<float>& l1, uint64_t offset, uint32_t rows,
    uint32_t depth, uint32_t lda)
{
    if constexpr (IsTransA) {
        CgemmCopyInA1(gm[offset], l1, depth, rows, lda);
    } else {
        CgemmCopyInA1(gm[offset], l1, rows, depth, lda);
    }
}

template <bool IsTransB>
__aicore__ __inline__ __attribute__((always_inline)) void CopyCgemmRightPlane(
    const AscendC::GlobalTensor<float>& gm, const AscendC::LocalTensor<float>& l1, uint64_t offset, uint32_t cols,
    uint32_t depth, uint32_t ldb)
{
    if constexpr (IsTransB) {
        CgemmCopyInB1(gm[offset], l1, depth, cols, ldb);
    } else {
        CgemmCopyInB1(gm[offset], l1, cols, depth, ldb);
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void InitFourProductPipeline()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(GEMM_FIRST_FLAG);
}

__aicore__ __inline__ __attribute__((always_inline)) void FinishFourProductPipeline()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(GEMM_FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GEMM_FIRST_FLAG);
}

template <bool IsTransA, bool IsTransB>
__aicore__ __inline__ __attribute__((always_inline)) void AdvanceCgemmOffsets(
    uint64_t& leftOffset, uint64_t& rightOffset, uint32_t curKChunk, const GemmTilingData& tiling)
{
    if constexpr (IsTransA) {
        leftOffset += static_cast<uint64_t>(curKChunk) * tiling.lda;
    } else {
        leftOffset += curKChunk;
    }
    if constexpr (IsTransB) {
        rightOffset += curKChunk;
    } else {
        rightOffset += static_cast<uint64_t>(curKChunk) * tiling.ldb;
    }
}

struct CgemmProductTensors {
    AscendC::GlobalTensor<float> left;
    AscendC::GlobalTensor<float> right;
    AscendC::GlobalTensor<float> output;
};

__aicore__ __inline__ __attribute__((always_inline)) void InitCgemmProductTensors(
    CgemmProductTensors& gm, uint32_t product, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA,
    GM_ADDR auxiliaryB, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, bool threeProducts)
{
    GM_ADDR left = (product == 0 || product == 3) ? br : bi;
    GM_ADDR right = (product == 0 || product == 2) ? ar : ai;
    if (threeProducts) {
        left = product == 0 ? br : product == 1 ? bi : auxiliaryB;
        right = product == 0 ? ar : product == 1 ? ai : auxiliaryA;
    }
    GM_ADDR output = product == 0 ? t1 : product == 1 ? t2 : product == 2 ? t3 : t4;
    gm.left.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(left));
    gm.right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(right));
    gm.output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
}

template <uint32_t BS, bool IsTransA, bool IsTransB>
__aicore__ inline void RunCgemmProductChunk(
    const CgemmProductTensors& gm, const GemmTilingData& tiling, uint32_t curM, uint32_t curN, uint32_t kOffset,
    uint64_t& leftOffset, uint64_t& rightOffset, uint64_t& l1PingPong, uint64_t& l0PingPong,
    const AscendC::LocalTensor<float>& accum)
{
    constexpr uint32_t K_L1 = CGEMM_L1_OPERAND_STAGE_ELEMENTS / BS;
    constexpr uint32_t K_L0 = CGEMM_HALF_L0_FLOAT_ELEMENTS / BS;
    constexpr uint32_t COMPONENT_BYTES = BS * K_L1 * sizeof(float);
    constexpr uint32_t SLOT_BYTES = 2 * COMPONENT_BYTES;
    const uint32_t curKChunk = Min<uint32_t>(K_L1, static_cast<uint32_t>(tiling.k) - kOffset);
    const uint32_t l1Slot = l1PingPong & 1;
    const uint32_t l1Base = l1Slot * SLOT_BYTES;
    AscendC::LocalTensor<float> leftL1(AscendC::TPosition::A1, l1Base, BS * K_L1);
    AscendC::LocalTensor<float> rightL1(AscendC::TPosition::A1, l1Base + COMPONENT_BYTES, BS * K_L1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Slot);
    CopyCgemmLeftPlane<IsTransA>(gm.left, leftL1, leftOffset, curM, curKChunk, tiling.lda);
    CopyCgemmRightPlane<IsTransB>(gm.right, rightL1, rightOffset, curN, curKChunk, tiling.ldb);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1Slot);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1Slot);
    for (uint32_t kInner = 0; kInner < curKChunk; kInner += K_L0) {
        const uint32_t curK = Min<uint32_t>(K_L0, curKChunk - kInner);
        const uint64_t leftInner =
            IsTransA ? kInner * GEMM_FP32_C0 : kInner * RoundUp<uint32_t>(curM, CGEMM_CUBE_BLOCK);
        const uint64_t rightInner =
            IsTransB ? kInner * RoundUp<uint32_t>(curN, CGEMM_CUBE_BLOCK) : kInner * GEMM_FP32_C0;
        const uint32_t l0Slot = l0PingPong & 1;
        const uint32_t l0Base = l0Slot * BS * K_L0 * sizeof(float);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Slot);
        AscendC::LocalTensor<float> leftL0(AscendC::TPosition::A2, l0Base, BS * K_L0);
        AscendC::LocalTensor<float> rightL0(AscendC::TPosition::B2, l0Base, BS * K_L0);
        CgemmCopyInA2(leftL0, leftL1[leftInner], curM, curKChunk, curK, IsTransA);
        CgemmCopyInB2(rightL0, rightL1[rightInner], curN, curKChunk, curK, IsTransB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
        CgemmMmad(accum, leftL0, rightL0, curM, curN, curK, kOffset == 0 && kInner == 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Slot);
        ++l0PingPong;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Slot);
    ++l1PingPong;
    AdvanceCgemmOffsets<IsTransA, IsTransB>(leftOffset, rightOffset, curKChunk, tiling);
}

template <uint32_t BS, bool IsTransA, bool IsTransB>
__aicore__ inline void RunCgemmProductTask(
    CgemmProductTensors& gm, const GemmTilingData& tiling, uint32_t tile, uint32_t nTiles, uint32_t tileN,
    uint64_t& l1PingPong, uint64_t& l0PingPong, uint32_t outputSlot)
{
    constexpr uint32_t K_L1 = CGEMM_L1_OPERAND_STAGE_ELEMENTS / BS;
    const uint32_t mOffset = (tile / nTiles) * BS;
    const uint32_t nOffset = (tile % nTiles) * tileN;
    const uint32_t curM = Min<uint32_t>(BS, static_cast<uint32_t>(tiling.m) - mOffset);
    const uint32_t curN = Min<uint32_t>(tileN, static_cast<uint32_t>(tiling.n) - nOffset);
    uint64_t leftOffset = IsTransA ? mOffset : static_cast<uint64_t>(mOffset) * tiling.lda;
    uint64_t rightOffset = IsTransB ? static_cast<uint64_t>(nOffset) * tiling.ldb : nOffset;
    AscendC::LocalTensor<float> accum(AscendC::TPosition::CO1, outputSlot * BS * BS * sizeof(float), BS * BS);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(outputSlot);
    for (uint32_t kOffset = 0; kOffset < static_cast<uint32_t>(tiling.k); kOffset += K_L1) {
        RunCgemmProductChunk<BS, IsTransA, IsTransB>(
            gm, tiling, curM, curN, kOffset, leftOffset, rightOffset, l1PingPong, l0PingPong, accum);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(outputSlot);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(outputSlot);
    CgemmFixpipe<BS>(gm.output, accum, static_cast<uint64_t>(mOffset) * tiling.ldc + nOffset, curM, curN, tiling.ldc);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(outputSlot);
}

template <uint32_t BS, bool IsTransA, bool IsTransB>
__aicore__ inline void CgemmProductsImpl(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, GM_ADDR t1, GM_ADDR t2,
    GM_ADDR t3, GM_ADDR t4, GemmTilingData& tiling)
{
    constexpr uint32_t K_L1 = CGEMM_L1_OPERAND_STAGE_ELEMENTS / BS;
    constexpr uint32_t K_L0 = CGEMM_HALF_L0_FLOAT_ELEMENTS / BS;
    static_assert(4 * BS * K_L1 * sizeof(float) <= CGEMM_L1_BYTES, "Cgemm product L1 overflow");
    static_assert(2 * BS * K_L0 * sizeof(float) <= CGEMM_L0_OPERAND_BYTES, "Cgemm product L0 overflow");
    static_assert(CGEMM_OUTPUT_STAGES * BS * BS * sizeof(float) <= CGEMM_L0C_BYTES, "Cgemm product L0C overflow");
    const uint32_t block = AscendC::GetBlockIdx();
    const uint32_t blocks = AscendC::GetBlockNum();
    const uint32_t tileN = Min<uint32_t>(BS, static_cast<uint32_t>(tiling.baseN));
    const uint32_t nTiles = CeilDiv(static_cast<uint32_t>(tiling.n), tileN);
    const uint64_t tileCountWide = static_cast<uint64_t>(CeilDiv(static_cast<uint32_t>(tiling.m), BS)) * nTiles;
    const uint32_t partitions = tiling.cgemmKPartitions > 0 ? tiling.cgemmKPartitions : 1;
    const uint32_t products = tiling.cgemmThreeProducts ? CGEMM_THREE_PRODUCTS : CGEMM_FOUR_PRODUCTS;
    const uint64_t taskCountWide = products * tileCountWide * partitions;
    if (taskCountWide > CGEMM_UINT32_MAX_VALUE || block >= taskCountWide) {
        return;
    }
    const uint32_t tileCount = static_cast<uint32_t>(tileCountWide);
    const uint32_t taskCount = static_cast<uint32_t>(taskCountWide);
    const uint64_t partStride = static_cast<uint64_t>(tiling.ldc) * RoundUp<uint32_t>(tiling.m, CGEMM_CUBE_BLOCK);
    InitFourProductPipeline();
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;
    for (uint32_t task = block; task < taskCount; task += blocks) {
        const uint32_t partition = task / (products * tileCount);
        const uint32_t product = (task / tileCount) % products;
        const uint32_t tile = task % tileCount;
        const uint32_t kStart = partition * tiling.cgemmKSpan;
        GemmTilingData partTiling = tiling;
        if (partitions > 1) {
            partTiling.k = Min<uint32_t>(tiling.cgemmKSpan, tiling.k - kStart);
        }
        const uint64_t leftStart = IsTransA ? static_cast<uint64_t>(kStart) * tiling.lda : kStart;
        const uint64_t rightStart = IsTransB ? kStart : static_cast<uint64_t>(kStart) * tiling.ldb;
        const uint64_t outputStart = partition * partStride * sizeof(float);
        CgemmProductTensors gm;
        InitCgemmProductTensors(
            gm, product, ar + rightStart * sizeof(float), ai + rightStart * sizeof(float),
            br + leftStart * sizeof(float), bi + leftStart * sizeof(float),
            auxiliaryA == nullptr ? nullptr : auxiliaryA + rightStart * sizeof(float),
            auxiliaryB == nullptr ? nullptr : auxiliaryB + leftStart * sizeof(float), t1 + outputStart,
            t2 + outputStart, t3 + outputStart, t4 + outputStart, tiling.cgemmThreeProducts != 0);
        RunCgemmProductTask<BS, IsTransA, IsTransB>(
            gm, partTiling, tile, nTiles, tileN, l1PingPong, l0PingPong, (task / blocks) % CGEMM_OUTPUT_STAGES);
    }
    FinishFourProductPipeline();
}

template <bool TransA, bool TransB, uint32_t Tile = CGEMM_MAX_SQUARE_TILE>
__aicore__ inline void DispatchProducts(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, GM_ADDR t1, GM_ADDR t2,
    GM_ADDR t3, GM_ADDR t4, GemmTilingData& tiling)
{
    if constexpr (Tile > CGEMM_CUBE_FRACTAL) {
        if (tiling.baseM < Tile) {
            DispatchProducts<TransA, TransB, Tile / 2>(ar, ai, br, bi, auxiliaryA, auxiliaryB, t1, t2, t3, t4, tiling);
            return;
        }
    }
    CgemmProductsImpl<Tile, TransA, TransB>(ar, ai, br, bi, auxiliaryA, auxiliaryB, t1, t2, t3, t4, tiling);
}

extern "C" __global__ __cube__ void gemm_cgemm_products_kernel(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, GM_ADDR t1, GM_ADDR t2,
    GM_ADDR t3, GM_ADDR t4, GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    if (tiling.isTransA != 0 && tiling.isTransB != 0) {
        DispatchProducts<true, true>(ar, ai, br, bi, auxiliaryA, auxiliaryB, t1, t2, t3, t4, tiling);
    } else if (tiling.isTransA != 0) {
        DispatchProducts<true, false>(ar, ai, br, bi, auxiliaryA, auxiliaryB, t1, t2, t3, t4, tiling);
    } else if (tiling.isTransB != 0) {
        DispatchProducts<false, true>(ar, ai, br, bi, auxiliaryA, auxiliaryB, t1, t2, t3, t4, tiling);
    } else {
        DispatchProducts<false, false>(ar, ai, br, bi, auxiliaryA, auxiliaryB, t1, t2, t3, t4, tiling);
    }
}

void gemm_cgemm_products_do(
    uint32_t numBlocks, void* stream, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA,
    GM_ADDR auxiliaryB, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, const GemmTilingData& tiling)
{
    gemm_cgemm_products_kernel<<<numBlocks, nullptr, stream>>>(
        ar, ai, br, bi, auxiliaryA, auxiliaryB, t1, t2, t3, t4, tiling);
}
namespace cgemm_fused {
using namespace AscendC;
constexpr uint16_t FREE_FLAG = 6;
constexpr uint16_t READY_FLAG = FREE_FLAG + CGEMM_OUTPUT_STAGES;
constexpr uint16_t PEER_OFFSET = 16; // Mode-4 event bank for the second paired AIV.
constexpr uint16_t SYNC_MODE = 4;
constexpr uint32_t INPUT_PLANES = CGEMM_COMPLEX_COMPONENTS * CGEMM_THREE_PRODUCTS;
constexpr uint32_t VECTOR_FLOATS = CGEMM_VECTOR_BYTES / sizeof(float);
constexpr uint32_t REPAIR_THREADS = CGEMM_VECTOR_BYTES / sizeof(float);
constexpr uint32_t FP32_EXPONENT_MASK = 0x7f800000; // All exponent bits set identify infinity or NaN.
constexpr FixpipeConfig ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

template <uint32_t BS>
struct Storage {
    static constexpr uint32_t ROWS = BS / CGEMM_COMPLEX_COMPONENTS;
    static constexpr uint32_t PLANE = ROWS * BS;
    static constexpr uint32_t BYTES = PLANE * sizeof(float);
    static constexpr uint32_t UB_STAGE_BYTES = (CGEMM_FOUR_PRODUCTS + CGEMM_COMPLEX_COMPONENTS) * BYTES;
    static constexpr uint32_t STAGES =
        CGEMM_UB_WORKSPACE_BYTES / UB_STAGE_BYTES >= CGEMM_OUTPUT_STAGES ? CGEMM_OUTPUT_STAGES : 1;
    static constexpr uint32_t INNER_K = CGEMM_HALF_L0_FLOAT_ELEMENTS / BS;
    static constexpr uint32_t PANEL_CAPACITY =
        CGEMM_L1_BYTES / (CGEMM_L1_PING_PONG_STAGES * INPUT_PLANES * BS * sizeof(float));
    static constexpr uint32_t PANEL_K = PANEL_CAPACITY / INNER_K * INNER_K;
    static_assert((CGEMM_FOUR_PRODUCTS + CGEMM_COMPLEX_COMPONENTS) * BYTES <= CGEMM_UB_WORKSPACE_BYTES);
    static_assert(CGEMM_FOUR_PRODUCTS * BS * BS * sizeof(float) <= CGEMM_L0C_BYTES);
};

template <uint32_t BS>
__simd_vf__ inline void MergeProducts(__ubuf__ float* p, bool first, uint32_t rows)
{
    namespace V = AscendC::MicroAPI;
    constexpr uint32_t PLANE = Storage<BS>::PLANE;
    V::MaskReg mask = V::CreateMask<float, V::MaskPattern::ALL>();
    for (uint32_t offset = 0; offset < rows * BS; offset += VECTOR_FLOATS) {
        V::RegTensor<float> a, b, c, real, imag, previous;
        V::LoadAlign(a, p + offset);
        V::LoadAlign(b, p + PLANE + offset);
        V::LoadAlign(c, p + 2 * PLANE + offset);
        V::Sub(real, a, c, mask);
        V::Add(imag, a, b, mask);
        if (!first) {
            V::LoadAlign(previous, p + 4 * PLANE + offset);
            V::Add(real, real, previous, mask);
            V::LoadAlign(previous, p + 5 * PLANE + offset);
            V::Add(imag, imag, previous, mask);
        }
        V::StoreAlign(p + 4 * PLANE + offset, real, mask);
        V::StoreAlign(p + 5 * PLANE + offset, imag, mask);
    }
}

// With one reduction part, the sum buffers are free and can hold the final complex output directly.
template <uint32_t BS, bool CheckNonFinite = false>
__simd_vf__ inline void BuildSinglePartOutput(__ubuf__ float* p, uint32_t rows)
{
    namespace V = AscendC::MicroAPI;
    constexpr uint32_t PLANE = Storage<BS>::PLANE;
    V::MaskReg mask = V::CreateMask<float, V::MaskPattern::ALL>();
    // Exponent bits identify both infinity and NaN without floating-point comparisons.
    V::RegTensor<uint32_t> exponentMask, largest, exponent;
    if constexpr (CheckNonFinite) {
        V::Duplicate(exponentMask, FP32_EXPONENT_MASK);
        V::Duplicate(largest, uint32_t(0));
    }
    for (uint32_t offset = 0; offset < rows * BS; offset += VECTOR_FLOATS) {
        V::RegTensor<float> a, b, c, real, imag, lo, hi;
        V::LoadAlign(a, p + offset);
        V::LoadAlign(b, p + PLANE + offset);
        V::LoadAlign(c, p + 2 * PLANE + offset);
        V::Sub(real, a, c, mask);
        V::Add(imag, a, b, mask);
        if constexpr (CheckNonFinite) {
            V::And(exponent, reinterpret_cast<V::RegTensor<uint32_t>&>(real), exponentMask, mask);
            V::Max(largest, largest, exponent, mask);
            V::And(exponent, reinterpret_cast<V::RegTensor<uint32_t>&>(imag), exponentMask, mask);
            V::Max(largest, largest, exponent, mask);
        }
        V::Interleave(lo, hi, real, imag);
        V::StoreAlign(p + 4 * PLANE + 2 * offset, lo, mask);
        V::StoreAlign(p + 4 * PLANE + 2 * offset + VECTOR_FLOATS, hi, mask);
    }
    if constexpr (CheckNonFinite) {
        V::Reduce<V::ReduceType::MAX>(largest, largest, mask);
        // The unused fourth product plane holds one per-tile control value.
        V::StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(p + 3 * PLANE), largest, mask);
    }
}

template <uint32_t BS>
__simd_vf__ inline void InterleaveOutput(__ubuf__ float* p, uint32_t rows)
{
    namespace V = AscendC::MicroAPI;
    constexpr uint32_t PLANE = Storage<BS>::PLANE;
    V::MaskReg mask = V::CreateMask<float, V::MaskPattern::ALL>();
    for (uint32_t offset = 0; offset < rows * BS; offset += VECTOR_FLOATS) {
        V::RegTensor<float> real, imag, lo, hi;
        V::LoadAlign(real, p + 4 * PLANE + offset);
        V::LoadAlign(imag, p + 5 * PLANE + offset);
        V::Interleave(lo, hi, real, imag);
        V::StoreAlign(p + 2 * offset, lo, mask);
        V::StoreAlign(p + 2 * offset + VECTOR_FLOATS, hi, mask);
    }
}

struct TileShape {
    uint32_t row, col, rows, cols, paddedRows;
};

template <uint32_t BS>
__aicore__ inline TileShape Shape(const GemmTilingData& t, uint64_t tile)
{
    const uint32_t columns = CeilDiv<uint32_t>(t.n, BS);
    TileShape s{static_cast<uint32_t>((tile / columns) * BS), static_cast<uint32_t>((tile % columns) * BS), 0, 0, 0};
    s.rows = Min<uint32_t>(BS, t.m - s.row);
    s.cols = Min<uint32_t>(BS, t.n - s.col);
    s.paddedRows = RoundUp<uint32_t>(s.rows, CGEMM_CUBE_FRACTAL);
    return s;
}

template <uint32_t BS, bool TransA, bool TransB>
__aicore__ inline void LoadPanel(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, const GemmTilingData& t,
    const TileShape& s, uint32_t start, uint32_t depth, uint32_t stage)
{
    constexpr uint32_t ELEMENTS = BS * Storage<BS>::PANEL_K;
    constexpr uint32_t BYTES = ELEMENTS * sizeof(float);
    const uint32_t l1Base = stage * INPUT_PLANES * BYTES;
    WaitFlag<HardEvent::MTE1_MTE2>(stage);
    const uint64_t leftOffset =
        TransA ? s.row + static_cast<uint64_t>(start) * t.lda : static_cast<uint64_t>(s.row) * t.lda + start;
    const uint64_t rightOffset =
        TransB ? static_cast<uint64_t>(s.col) * t.ldb + start : s.col + static_cast<uint64_t>(start) * t.ldb;
    GlobalTensor<float> gar, gai, gbr, gbi, gauxa, gauxb;
    gar.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ar));
    gai.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ai));
    gbr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(br));
    gbi.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bi));
    gauxa.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(auxiliaryA));
    gauxb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(auxiliaryB));
    LocalTensor<float> lbr(TPosition::A1, l1Base, ELEMENTS), lbi(TPosition::A1, l1Base + BYTES, ELEMENTS);
    LocalTensor<float> lbaux(TPosition::A1, l1Base + 2 * BYTES, ELEMENTS);
    LocalTensor<float> lar(TPosition::A1, l1Base + 3 * BYTES, ELEMENTS),
        lai(TPosition::A1, l1Base + 4 * BYTES, ELEMENTS);
    LocalTensor<float> laaux(TPosition::A1, l1Base + 5 * BYTES, ELEMENTS);
    CopyCgemmLeftPlane<TransA>(gbr, lbr, leftOffset, s.rows, depth, t.lda);
    CopyCgemmLeftPlane<TransA>(gbi, lbi, leftOffset, s.rows, depth, t.lda);
    CopyCgemmRightPlane<TransB>(gar, lar, rightOffset, s.cols, depth, t.ldb);
    CopyCgemmRightPlane<TransB>(gai, lai, rightOffset, s.cols, depth, t.ldb);
    CopyCgemmLeftPlane<TransA>(gauxb, lbaux, leftOffset, s.rows, depth, t.lda);
    CopyCgemmRightPlane<TransB>(gauxa, laaux, rightOffset, s.cols, depth, t.ldb);
    SetFlag<HardEvent::MTE2_MTE1>(stage);
}

template <uint32_t BS, bool TransA, bool TransB>
__aicore__ inline void MultiplyPanel(const TileShape& s, uint32_t depth, bool first, uint32_t& sequence, uint32_t stage)
{
    constexpr uint32_t ELEMENTS = BS * Storage<BS>::PANEL_K;
    constexpr uint32_t INNER_K = Storage<BS>::INNER_K;
    const uint32_t l1Base = stage * INPUT_PLANES * ELEMENTS * sizeof(float);
    for (uint32_t inner = 0; inner < depth; inner += INNER_K) {
        const uint32_t count = Min<uint32_t>(INNER_K, depth - inner);
        const uint32_t leftInner = TransA ? inner * GEMM_FP32_C0 : inner * s.paddedRows;
        const uint32_t rightInner =
            TransB ? inner * RoundUp<uint32_t>(s.cols, CGEMM_CUBE_FRACTAL) : inner * GEMM_FP32_C0;
        for (uint32_t product = 0; product < CGEMM_THREE_PRODUCTS; ++product, ++sequence) {
            const uint32_t leftPlane = product;
            const uint32_t rightPlane = product + CGEMM_THREE_PRODUCTS;
            LocalTensor<float> a1(TPosition::A1, l1Base + leftPlane * ELEMENTS * sizeof(float), ELEMENTS);
            LocalTensor<float> b1(TPosition::A1, l1Base + rightPlane * ELEMENTS * sizeof(float), ELEMENTS);
            const uint32_t slot = sequence % CGEMM_OUTPUT_STAGES;
            const uint32_t base = slot * CGEMM_HALF_L0_FLOAT_ELEMENTS * sizeof(float);
            WaitFlag<HardEvent::M_MTE1>(slot);
            LocalTensor<float> a2(TPosition::A2, base, BS * INNER_K), b2(TPosition::B2, base, BS * INNER_K);
            CgemmCopyInA2(a2, a1[leftInner], s.rows, depth, count, TransA);
            CgemmCopyInB2(b2, b1[rightInner], s.cols, depth, count, TransB);
            SetFlag<HardEvent::MTE1_M>(slot);
            WaitFlag<HardEvent::MTE1_M>(slot);
            LocalTensor<float> accum(TPosition::CO1, product * BS * BS * sizeof(float), BS * BS);
            // Input loads may overlap the previous tile's writeback; only accumulation reuses L0C.
            if (first && inner == 0 && product == 0) {
                WaitFlag<HardEvent::FIX_M>(GEMM_ZERO_FLAG);
            }
            CgemmMmad(accum, a2, b2, s.rows, s.cols, count, first && inner == 0);
            SetFlag<HardEvent::M_MTE1>(slot);
        }
    }
    SetFlag<HardEvent::MTE1_MTE2>(stage);
}

template <uint32_t BS>
__aicore__ inline void PublishProducts(const TileShape& s, uint32_t outputStage)
{
    SetFlag<HardEvent::M_FIX>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::M_FIX>(GEMM_ZERO_FLAG);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FREE_FLAG + outputStage);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FREE_FLAG + outputStage + PEER_OFFSET);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fp{};
    fp.nSize = RoundUp<uint32_t>(s.cols, CGEMM_CUBE_FRACTAL);
    fp.mSize = s.paddedRows;
    fp.dstStride = BS;
    fp.srcStride = s.paddedRows;
    fp.quantPre = QuantMode_t::NoQuant;
    fp.dualDstCtl = 1;
    for (uint32_t product = 0; product < CGEMM_THREE_PRODUCTS; ++product) {
        LocalTensor<float> accum(TPosition::CO1, product * BS * BS * sizeof(float), BS * BS);
        LocalTensor<float> out(
            TPosition::VECCALC, outputStage * Storage<BS>::UB_STAGE_BYTES + product * Storage<BS>::BYTES,
            Storage<BS>::PLANE);
        Fixpipe<float, float, ROW_MAJOR_UB>(out, accum, fp);
    }
    SetFlag<HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(READY_FLAG + outputStage);
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(READY_FLAG + outputStage + PEER_OFFSET);
}

template <uint32_t BS, bool TransA, bool TransB>
__aicore__ inline void ComputePart(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, const GemmTilingData& t,
    const TileShape& s, uint32_t start, uint32_t end, uint32_t outputStage)
{
    SetFlag<HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
    SetFlag<HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
    SetFlag<HardEvent::MTE1_MTE2>(GEMM_ZERO_FLAG);
    SetFlag<HardEvent::MTE1_MTE2>(GEMM_FIRST_FLAG);
    constexpr uint32_t PANEL_K = Storage<BS>::PANEL_K;
    uint32_t sequence = 0;
    LoadPanel<BS, TransA, TransB>(
        ar, ai, br, bi, auxiliaryA, auxiliaryB, t, s, start, Min<uint32_t>(PANEL_K, end - start), 0);
    for (uint32_t k = start, stage = 0; k < end; k += PANEL_K, stage ^= 1) {
        WaitFlag<HardEvent::MTE2_MTE1>(stage);
        const uint32_t next = k + PANEL_K;
        if (next < end) {
            LoadPanel<BS, TransA, TransB>(
                ar, ai, br, bi, auxiliaryA, auxiliaryB, t, s, next, Min<uint32_t>(PANEL_K, end - next), stage ^ 1);
        }
        MultiplyPanel<BS, TransA, TransB>(s, Min<uint32_t>(PANEL_K, end - k), k == start, sequence, stage);
    }
    WaitFlag<HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
    WaitFlag<HardEvent::MTE1_MTE2>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::MTE1_MTE2>(GEMM_FIRST_FLAG);
    PublishProducts<BS>(s, outputStage);
}

// Recompute only non-finite outputs with explicit FP32 operations. Cube and
// scalar BLAS can differ when individual products overflow before cancellation.
template <uint32_t BS>
__simt_vf__ __aicore__ LAUNCH_BOUND(REPAIR_THREADS) inline void RepairNonFiniteOutput(
    __gm__ float* output, __gm__ const float* a, __gm__ const float* b, uint32_t depth, uint32_t lda, uint32_t ldb,
    uint32_t ldc, bool transA, bool transB, bool conjugateA, bool conjugateB, uint32_t rowOrigin, uint32_t colOrigin,
    uint32_t rows, uint32_t cols)
{
#pragma clang fp contract(off)
    for (uint32_t index = threadIdx.x; index < rows * cols; index += blockDim.x) {
        const uint32_t localRow = index / cols, localCol = index % cols;
        const uint64_t offset =
            CGEMM_COMPLEX_COMPONENTS * (static_cast<uint64_t>(rowOrigin + localRow) * ldc + colOrigin + localCol);
        if (isfinite(output[offset]) && isfinite(output[offset + 1])) {
            continue;
        }
        const uint32_t row = colOrigin + localCol, col = rowOrigin + localRow;
        float real = 0.0f, imag = 0.0f;
        for (uint32_t k = 0; k < depth; ++k) {
            const uint64_t ai = CGEMM_COMPLEX_COMPONENTS *
                                (transA ? static_cast<uint64_t>(row) * lda + k : static_cast<uint64_t>(k) * lda + row);
            const uint64_t bi = CGEMM_COMPLEX_COMPONENTS *
                                (transB ? static_cast<uint64_t>(k) * ldb + col : static_cast<uint64_t>(col) * ldb + k);
            const float ar = a[ai], av = conjugateA ? -a[ai + 1] : a[ai + 1];
            const float br = b[bi], bv = conjugateB ? -b[bi + 1] : b[bi + 1];
            // Materialize each FP32 product before combining it. This preserves
            // overflow and invalid-operation behavior instead of contracting it.
            volatile float rr = ar * br;
            volatile float ii = av * bv;
            volatile float ri = ar * bv;
            volatile float ir = av * br;
            real += rr - ii;
            imag += ri + ir;
        }
        output[offset] = real;
        output[offset + 1] = imag;
    }
}

template <uint32_t BS>
__aicore__ inline bool OutputNeedsRepair(const LocalTensor<float>& buffer, uint32_t rows)
{
    SetFlag<HardEvent::V_S>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::V_S>(GEMM_ZERO_FLAG);
    // Read one control value, never matrix elements, on the scalar side.
    const uint32_t exponent = buffer.template ReinterpretCast<uint32_t>().GetValue(3 * Storage<BS>::PLANE);
    return exponent == FP32_EXPONENT_MASK && rows > 0;
}

template <uint32_t BS>
__aicore__ inline void RepairOutputTile(
    GM_ADDR c, GM_ADDR a, GM_ADDR b, const CgemmOnchipTilingData& t, const TileShape& s, uint32_t start, uint32_t rows)
{
    SetFlag<HardEvent::MTE3_V>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::MTE3_V>(GEMM_ZERO_FLAG);
    asc_vf_call<RepairNonFiniteOutput<BS>>(
        dim3{REPAIR_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(c), reinterpret_cast<__gm__ const float*>(a),
        reinterpret_cast<__gm__ const float*>(b), t.k, t.lda, t.ldb, t.ldc, t.transA, t.transB, t.conjugateA,
        t.conjugateB, s.row + start, s.col, rows, s.cols);
}

template <uint32_t BS>
__simd_vf__ inline void CheckMergedOutput(__ubuf__ float* p, uint32_t rows)
{
    namespace V = AscendC::MicroAPI;
    auto mask = V::CreateMask<float, V::MaskPattern::ALL>();
    V::RegTensor<uint32_t> exponentMask, largest, exponent, values;
    V::Duplicate(exponentMask, FP32_EXPONENT_MASK);
    V::Duplicate(largest, uint32_t(0));
    for (uint32_t offset = 0; offset < CGEMM_COMPLEX_COMPONENTS * rows * BS; offset += VECTOR_FLOATS) {
        V::LoadAlign(values, reinterpret_cast<__ubuf__ uint32_t*>(p + offset));
        V::And(exponent, values, exponentMask, mask);
        V::Max(largest, largest, exponent, mask);
    }
    V::Reduce<V::ReduceType::MAX>(largest, largest, mask);
    V::StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(p + 3 * Storage<BS>::PLANE), largest, mask);
}

template <uint32_t BS>
__aicore__ inline void StorePartOutput(
    GM_ADDR cAddr, const GemmTilingData& t, const TileShape& s, LocalTensor<float> buffer, uint32_t start,
    uint32_t rows, bool direct)
{
    GlobalTensor<float> c;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cAddr));
    const uint32_t bytes = CGEMM_COMPLEX_COMPONENTS * s.cols * sizeof(float);
    const uint32_t rowBytes = CGEMM_COMPLEX_COMPONENTS * BS * sizeof(float);
    const uint32_t sourceGap = (rowBytes - RoundUp<uint32_t>(bytes, CGEMM_DATA_BLOCK_BYTES)) / CGEMM_DATA_BLOCK_BYTES;
    const uint32_t destGap = (CGEMM_COMPLEX_COMPONENTS * t.ldc - CGEMM_COMPLEX_COMPONENTS * s.cols) * sizeof(float);
    DataCopyExtParams copy{static_cast<uint16_t>(rows), bytes, sourceGap, destGap, 0};
    const uint64_t offset = CGEMM_COMPLEX_COMPONENTS * (static_cast<uint64_t>(s.row + start) * t.ldc + s.col);
    DataCopyPad(c[offset], buffer[direct ? CGEMM_FOUR_PRODUCTS * Storage<BS>::PLANE : 0], copy);
}

template <uint32_t BS, bool CheckNonFinite = false>
__aicore__ inline void ConsumePart(
    GM_ADDR cAddr, const GemmTilingData& t, const TileShape& s, bool first, bool last, uint32_t outputStage,
    GM_ADDR a = nullptr, GM_ADDR b = nullptr, const CgemmOnchipTilingData* original = nullptr)
{
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(READY_FLAG + outputStage);
    SetFlag<HardEvent::MTE3_V>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::MTE3_V>(GEMM_ZERO_FLAG);
    constexpr uint32_t ELEMENTS = (CGEMM_FOUR_PRODUCTS + CGEMM_COMPLEX_COMPONENTS) * Storage<BS>::PLANE;
    LocalTensor<float> buffer(TPosition::VECCALC, outputStage * Storage<BS>::UB_STAGE_BYTES, ELEMENTS);
    auto ptr = reinterpret_cast<__ubuf__ float*>(buffer.GetPhyAddr());
    const uint32_t half = s.paddedRows / CGEMM_COMPLEX_COMPONENTS;
    const uint32_t start = GetSubBlockIdx() * half;
    const uint32_t rows = start < s.rows ? Min<uint32_t>(half, s.rows - start) : 0;
    const bool direct = first && last;
    bool repairTile = false;
    if (direct) {
        asc_vf_call<BuildSinglePartOutput<BS, CheckNonFinite>>(ptr, half);
        if constexpr (CheckNonFinite) {
            repairTile = OutputNeedsRepair<BS>(buffer, rows);
        }
    } else {
        asc_vf_call<MergeProducts<BS>>(ptr, first, half);
        if (last) {
            asc_vf_call<InterleaveOutput<BS>>(ptr, half);
            if constexpr (CheckNonFinite) {
                asc_vf_call<CheckMergedOutput<BS>>(ptr, half);
                repairTile = OutputNeedsRepair<BS>(buffer, rows);
            }
        }
    }
    SetFlag<HardEvent::V_MTE3>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::V_MTE3>(GEMM_ZERO_FLAG);
    if (last && rows > 0) {
        StorePartOutput<BS>(cAddr, t, s, buffer, start, rows, direct);
        if constexpr (CheckNonFinite) {
            if (repairTile) {
                RepairOutputTile<BS>(cAddr, a, b, *original, s, start, rows);
            }
        }
    }
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FREE_FLAG + outputStage);
}

template <uint32_t BS, bool TransA, bool TransB>
__aicore__ inline void Run(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, GM_ADDR c,
    const GemmTilingData& t)
{
    uint32_t block = GetBlockIdx();
    if ASCEND_IS_AIC {
        SetFlag<HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    }
    if ASCEND_IS_AIV {
        block /= GetTaskRation();
        for (uint32_t stage = 0; stage < Storage<BS>::STAGES; ++stage) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FREE_FLAG + stage);
        }
    }
    const uint64_t tiles = static_cast<uint64_t>(CeilDiv<uint32_t>(t.m, BS)) * CeilDiv<uint32_t>(t.n, BS);
    const uint32_t parts = t.cgemmKPartitions;
    uint32_t iteration = 0;
    for (uint64_t tile = block; tile < tiles; tile += t.usedCoreNum, ++iteration) {
        const uint32_t outputStage = iteration % Storage<BS>::STAGES;
        const TileShape s = Shape<BS>(t, tile);
        for (uint32_t part = 0; part < parts; ++part) {
            const uint32_t start = part * t.cgemmKSpan;
            const uint32_t end = Min<uint32_t>(t.k, start + t.cgemmKSpan);
            if ASCEND_IS_AIC {
                ComputePart<BS, TransA, TransB>(ar, ai, br, bi, auxiliaryA, auxiliaryB, t, s, start, end, outputStage);
            }
            if ASCEND_IS_AIV {
                ConsumePart<BS>(c, t, s, part == 0, part + 1 == parts, outputStage);
            }
        }
    }
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::FIX_M>(GEMM_ZERO_FLAG);
        for (uint32_t stage = 0; stage < Storage<BS>::STAGES; ++stage) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FREE_FLAG + stage);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FREE_FLAG + stage + PEER_OFFSET);
        }
    }
}

template <bool TransA, bool TransB>
__aicore__ inline void Dispatch(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, GM_ADDR c,
    const GemmTilingData& t)
{
    if (t.baseM == CGEMM_FUSED_MAX_TILE) {
        Run<CGEMM_FUSED_MAX_TILE, TransA, TransB>(ar, ai, br, bi, auxiliaryA, auxiliaryB, c, t);
    } else {
        Run<CGEMM_FUSED_TILE, TransA, TransB>(ar, ai, br, bi, auxiliaryA, auxiliaryB, c, t);
    }
}

extern "C" __global__ __aicore__ void gemm_cgemm_fused_kernel(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, GM_ADDR c, GemmTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    if ASCEND_IS_AIC {
        SetHF32Mode(HF32Mode::DISABLE);
    }
    if (t.isTransA && t.isTransB) {
        Dispatch<true, true>(ar, ai, br, bi, auxiliaryA, auxiliaryB, c, t);
    } else if (t.isTransA) {
        Dispatch<true, false>(ar, ai, br, bi, auxiliaryA, auxiliaryB, c, t);
    } else if (t.isTransB) {
        Dispatch<false, true>(ar, ai, br, bi, auxiliaryA, auxiliaryB, c, t);
    } else {
        Dispatch<false, false>(ar, ai, br, bi, auxiliaryA, auxiliaryB, c, t);
    }
}
} // namespace cgemm_fused

void gemm_cgemm_fused_do(
    uint32_t blocks, void* stream, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA,
    GM_ADDR auxiliaryB, GM_ADDR c, const GemmTilingData& tiling)
{
    cgemm_fused::gemm_cgemm_fused_kernel<<<blocks, nullptr, stream>>>(
        ar, ai, br, bi, auxiliaryA, auxiliaryB, c, tiling);
}

#include "cgemm_onchip_kernel.h"

#endif
