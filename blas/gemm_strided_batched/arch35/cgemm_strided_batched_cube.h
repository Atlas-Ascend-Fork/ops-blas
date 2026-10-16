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
#include "cgemm_strided_batched_cube_mma.h"

// Cube tiling, product selection and execution.
// Fused and ordered kernels instantiate these templates in separate source
// files; their definitions must be visible without relocatable Device linking.
namespace {
constexpr uint32_t HALF_REDUCTION_CHUNK_K = 512;
constexpr uint32_t HALF_DOUBLE_OUTPUT_MAX_K = 512;
constexpr uint32_t CUBE_L1_CHUNK_K = 256;
constexpr uint32_t SMALL_CUBE_TILE_DIM = 128;
constexpr uint32_t GENERAL_SPLIT_K = 32;

struct CgemmSbCubePlan {
    uint32_t group, effectiveN, effectiveK, effectiveBatch, products, mask;
    uint64_t divN, tiles;
    bool doubleOutput;
};

template <bool Partial, bool Prefix>
__aicore__ inline CgemmSbCubePlan CgemmSbMakeCubePlan(const CgemmSbTiling& t, uint32_t mask)
{
    CgemmSbCubePlan p{};
    p.group = t.threeM ? t.batchGroup : 1;
    p.effectiveN = t.np * p.group;
    p.effectiveK = t.kp * p.group;
    p.effectiveBatch = t.batch / p.group;
    p.divN = CeilDiv<uint64_t>(t.mp, t.tileN);
    p.tiles = CeilDiv<uint64_t>(p.effectiveN, t.tileM) * p.divN;
    p.mask = mask;
    p.doubleOutput = p.effectiveK <= HALF_DOUBLE_OUTPUT_MAX_K &&
                     uint64_t(t.tileM) * t.tileN * CGEMM_SB_FP32_BYTES <= CGEMM_SB_L0C_SLOT_BYTES;
    bool partial = Partial && CgemmSbPartialReference(mask, t);
    p.products =
        Prefix && mask == CGEMM_SB_REFERENCE_BOTH ?
            CGEMM_SB_GENERAL_PRODUCTS :
            (partial ? CGEMM_SB_PARTIAL_PRODUCTS : (t.threeM ? CGEMM_SB_THREE_M_PLANES : CGEMM_SB_GENERAL_PRODUCTS));
    return p;
}

template <typename Input>
struct CgemmSbCubeOperands {
    __gm__ Input* left;
    __gm__ Input* right;
    __gm__ float* out;
    uint32_t m, n;
};

template <typename Input, bool Partial, bool Prefix>
__aicore__ inline CgemmSbCubeOperands<Input> CgemmSbDecodeCubeTask(
    __gm__ float* ws, const CgemmSbTiling& t, const CgemmSbCubePlan& p, uint64_t task)
{
    uint64_t product = task / (p.tiles * p.effectiveBatch), batch = (task / p.tiles) % p.effectiveBatch;
    uint64_t tile = task % p.tiles;
    uint64_t leftImag = product == CGEMM_SB_PRODUCT_II || product == CGEMM_SB_PRODUCT_IR,
             rightImag = product == CGEMM_SB_PRODUCT_II || product == CGEMM_SB_PRODUCT_RI;
    if (t.threeM && product == CGEMM_SB_PRODUCT_SUM && !(Prefix && p.mask == CGEMM_SB_REFERENCE_BOTH)) {
        leftImag = CGEMM_SB_PLANE_SUM;
        rightImag = CGEMM_SB_PLANE_SUM;
    }
    if (Partial && CgemmSbPartialReference(p.mask, t) &&
        p.mask == (Prefix ? CGEMM_SB_REFERENCE_IMAG : CGEMM_SB_REFERENCE_REAL)) {
        leftImag = product;
        rightImag = 1 - product;
    }
    CgemmSbCubeOperands<Input> o{};
    o.left = reinterpret_cast<__gm__ Input*>(ws + t.bOffset + leftImag * t.bSize) +
             (t.batchesB == 1 ? 0 : batch * p.group) * uint64_t(t.np) * t.kp * t.batchGroup +
             (p.group == 1 && t.batchGroup > 1 ? (batch % t.batchGroup) * t.kp : 0);
    o.right = reinterpret_cast<__gm__ Input*>(ws + rightImag * t.aSize) +
              (t.batchesA == 1 ? 0 : batch * p.group) * uint64_t(t.kp) * t.mp;
    o.out = ws + t.cOffset + product * t.cSize + batch * uint64_t(p.effectiveN) * t.mp;
    o.m = (tile / p.divN) * t.tileM;
    o.n = (tile % p.divN) * t.tileN;
    return o;
}

template <bool Initialize>
__aicore__ inline void CgemmSbCubePipelineFlags()
{
    if constexpr (Initialize) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    }
}

template <typename LayoutPtn>
__aicore__ inline auto CgemmSbMakeHalfTensor(__gm__ half* data, uint32_t rows, uint32_t cols)
{
    return AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(data), AscendC::Te::MakeFrameLayout<LayoutPtn>(rows, cols));
}

template <typename LeftPtn, typename RightPtn, bool Prefix>
__aicore__ inline void CgemmSbComputeHalfTile(
    const CgemmSbCubeOperands<half>& o, const CgemmSbTiling& t, const CgemmSbCubePlan& p, uint64_t& l1, uint64_t& l0,
    uint32_t& outputIndex)
{
    auto ct = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(o.out),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(p.effectiveN, t.mp));
    auto lt = CgemmSbMakeHalfTensor<LeftPtn>(o.left, p.effectiveN, p.effectiveK);
    auto rt = CgemmSbMakeHalfTensor<RightPtn>(o.right, p.effectiveK, t.mp);
    auto ll =
        CgemmSbMakeHalfTensor<LeftPtn>(o.left + CGEMM_SB_HALF_RESIDUAL_OFFSET * t.bSize, p.effectiveN, p.effectiveK);
    auto rl = CgemmSbMakeHalfTensor<RightPtn>(o.right + CGEMM_SB_HALF_RESIDUAL_OFFSET * t.aSize, p.effectiveK, t.mp);
    uint32_t reductionK = Prefix ? CgemmSbPrefixK(t) : p.effectiveK;
    for (uint32_t off = 0; off < reductionK; off += HALF_REDUCTION_CHUNK_K) {
        uint32_t ck = Min<uint32_t>(HALF_REDUCTION_CHUNK_K, reductionK - off);
        uint32_t cm = Min<uint32_t>(t.tileM, p.effectiveN - o.m), cn = Min<uint32_t>(t.tileN, t.mp - o.n);
        auto lh = lt.Slice(AscendC::Te::MakeCoord(0, off), AscendC::Te::MakeShape(p.effectiveN, ck));
        auto rh = rt.Slice(AscendC::Te::MakeCoord(off, 0), AscendC::Te::MakeShape(ck, t.mp));
        auto lowl = ll.Slice(AscendC::Te::MakeCoord(0, off), AscendC::Te::MakeShape(p.effectiveN, ck));
        auto lowr = rl.Slice(AscendC::Te::MakeCoord(off, 0), AscendC::Te::MakeShape(ck, t.mp));
        uint32_t outputSlot = p.doubleOutput ? outputIndex++ % 2 : 0;
        uint64_t cOffset = uint64_t(outputSlot) * CGEMM_SB_L0C_SLOT_BYTES;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(outputSlot);
        CgemmHalfProcessKChunks(
            lowl, rh, ck, CUBE_L1_CHUNK_K, CeilDiv<uint64_t>(ck, CUBE_L1_CHUNK_K), t.baseK, o.m, o.n, cm, cn, l1, l0,
            true, false, cOffset);
        CgemmHalfProcessKChunks(
            lh, lowr, ck, CUBE_L1_CHUNK_K, CeilDiv<uint64_t>(ck, CUBE_L1_CHUNK_K), t.baseK, o.m, o.n, cm, cn, l1, l0,
            false, false, cOffset);
        if (t.mp <= SMALL_CUBE_TILE_DIM && t.np <= SMALL_CUBE_TILE_DIM && t.kp <= SMALL_CUBE_TILE_DIM &&
            ck <= t.baseK) {
            CgemmHalfReuseOperands(cm, cn, ck, l0, cOffset);
        } else {
            CgemmHalfProcessKChunks(
                lh, rh, ck, CUBE_L1_CHUNK_K, CeilDiv<uint64_t>(ck, CUBE_L1_CHUNK_K), t.baseK, o.m, o.n, cm, cn, l1, l0,
                false, true, cOffset);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        if (off == 0)
            AscendC::SetAtomicNone();
        else
            AscendC::SetAtomicAdd<float>();
        CgemmHalfCopyOutL0C2GM(ct, o.m, o.n, cm, cn, cOffset);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(outputSlot);
    }
    AscendC::SetAtomicNone();
}

template <typename LeftPtn, typename RightPtn>
__aicore__ inline void CgemmSbComputeFloatTile(
    const CgemmSbCubeOperands<float>& o, const CgemmSbTiling& t, const CgemmSbCubePlan& p, uint64_t& l1, uint64_t& l0)
{
    auto ct = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(o.out),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(p.effectiveN, t.mp));
    auto lt = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(o.left),
        AscendC::Te::MakeFrameLayout<LeftPtn>(p.effectiveN, t.kp * t.batchGroup));
    auto rt = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(o.right),
        AscendC::Te::MakeFrameLayout<RightPtn>(p.effectiveK, t.mp));
    uint32_t splitK = t.threeM ? (t.kp <= CUBE_L1_CHUNK_K ? CUBE_L1_CHUNK_K : SMALL_CUBE_TILE_DIM) : GENERAL_SPLIT_K;
    for (uint32_t off = 0; off < p.effectiveK; off += splitK) {
        uint32_t ck = Min<uint32_t>(splitK, p.effectiveK - off);
        auto left = lt.Slice(AscendC::Te::MakeCoord(0, off), AscendC::Te::MakeShape(p.effectiveN, ck));
        auto right = rt.Slice(AscendC::Te::MakeCoord(off, 0), AscendC::Te::MakeShape(ck, t.mp));
        if (off == 0)
            AscendC::SetAtomicNone();
        else
            AscendC::SetAtomicAdd<float>();
        GemmSbProcessTile(
            left, right, ct, ck, CUBE_L1_CHUNK_K, CeilDiv<uint64_t>(ck, CUBE_L1_CHUNK_K), t.baseK, o.m,
            Min<uint32_t>(o.m + t.tileM, p.effectiveN), o.n, Min<uint32_t>(o.n + t.tileN, t.mp), t.tileM, t.tileN, l1,
            l0);
    }
    AscendC::SetAtomicNone();
}

template <bool TRANS_LEFT, bool TRANS_RIGHT, bool HALF, bool Partial = false, bool Prefix = false>
__aicore__ inline void CgemmSbCube(GM_ADDR workspace, CgemmSbTiling t)
{
    auto ws = reinterpret_cast<__gm__ float*>(workspace);
    uint32_t mask = CgemmSbReferenceMask(ws + t.flagOffset, t);
    if constexpr (Prefix) {
        if (mask == CGEMM_SB_REFERENCE_NONE || mask == CGEMM_SB_REFERENCE_EXCEPTIONAL)
            return;
        if (mask != CGEMM_SB_REFERENCE_BOTH)
            t.cOffset += CGEMM_SB_PARTIAL_PRODUCTS * t.cSize;
    }
    if (!Prefix && mask != CGEMM_SB_REFERENCE_NONE && !(Partial && CgemmSbPartialReference(mask, t)))
        return;
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    AscendC::SetHF32TransMode(AscendC::HF32TransMode::NEAREST_EVEN);
    auto plan = CgemmSbMakeCubePlan<Partial, Prefix>(t, mask);
    CgemmSbCubePipelineFlags<true>();
    uint64_t l1 = 0, l0 = 0;
    uint32_t outputIndex = 0;
    using Input = AscendC::Std::conditional_t<HALF, half, float>;
    using LeftPtn = AscendC::Std::conditional_t<TRANS_LEFT, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using RightPtn = AscendC::Std::conditional_t<TRANS_RIGHT, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    for (uint64_t task = AscendC::GetBlockIdx(); task < plan.tiles * plan.effectiveBatch * plan.products;
         task += AscendC::GetBlockNum()) {
        auto operands = CgemmSbDecodeCubeTask<Input, Partial, Prefix>(ws, t, plan, task);
        if constexpr (HALF)
            CgemmSbComputeHalfTile<LeftPtn, RightPtn, Prefix>(operands, t, plan, l1, l0, outputIndex);
        else
            CgemmSbComputeFloatTile<LeftPtn, RightPtn>(operands, t, plan, l1, l0);
    }
    CgemmSbCubePipelineFlags<false>();
}

template <bool Partial = false, bool Prefix = false>
__aicore__ inline void CgemmSbRunCube(GM_ADDR workspace, CgemmSbTiling t)
{
    if (t.transB == ACLBLAS_OP_N) {
        if (t.transA == ACLBLAS_OP_N) {
            if (t.compensatedHalf)
                CgemmSbCube<false, false, true, Partial, Prefix>(workspace, t);
            else
                CgemmSbCube<false, false, false, Partial, Prefix>(workspace, t);
        } else {
            if (t.compensatedHalf)
                CgemmSbCube<false, true, true, Partial, Prefix>(workspace, t);
            else
                CgemmSbCube<false, true, false, Partial, Prefix>(workspace, t);
        }
    } else {
        if (t.transA == ACLBLAS_OP_N) {
            if (t.compensatedHalf)
                CgemmSbCube<true, false, true, Partial, Prefix>(workspace, t);
            else
                CgemmSbCube<true, false, false, Partial, Prefix>(workspace, t);
        } else {
            if (t.compensatedHalf)
                CgemmSbCube<true, true, true, Partial, Prefix>(workspace, t);
            else
                CgemmSbCube<true, true, false, Partial, Prefix>(workspace, t);
        }
    }
}

} // namespace
