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

template <bool Load = true, bool Compute = true, typename Left, typename Right>
__aicore__ inline void CgemmSbCachedHalfProducts(
    Left lh, Right rh, Left ll, Right rl, uint32_t cm, uint32_t cn, uint32_t ck, uint32_t m, uint32_t n, uint32_t baseK,
    uint64_t bank, uint64_t& l0, uint64_t cOffset)
{
    namespace T = AscendC::Te;
    const uint64_t start = bank * CGEMM_HALF_L1_SIZE / CGEMM_HALF_L1_BUF_NUM;
    const uint64_t leftBytes =
        RoundUp<uint64_t>(cm, CGEMM_HALF_FRACTAL) * RoundUp<uint64_t>(ck, CGEMM_HALF_FP32_C0) * sizeof(half);
    const uint64_t rightBytes =
        RoundUp<uint64_t>(cn, CGEMM_HALF_FRACTAL) * RoundUp<uint64_t>(ck, CGEMM_HALF_FP32_C0) * sizeof(half);
    const auto leftLayout = T::MakeFrameLayout<T::NZLayoutPtn, AscendC::Std::Int<CGEMM_HALF_FP32_C0>>(cm, ck);
    const auto rightLayout = T::MakeFrameLayout<T::ZNLayoutPtn, AscendC::Std::Int<CGEMM_HALF_FP32_C0>>(ck, cn);
    auto highA = T::MakeTensor(T::MakeMemPtr<T::Location::L1, half>(start), leftLayout);
    auto highB = T::MakeTensor(T::MakeMemPtr<T::Location::L1, half>(start + leftBytes), rightLayout);
    auto lowA = T::MakeTensor(T::MakeMemPtr<T::Location::L1, half>(start + leftBytes + rightBytes), leftLayout);
    auto lowB = T::MakeTensor(T::MakeMemPtr<T::Location::L1, half>(start + 2 * leftBytes + rightBytes), rightLayout);
    if constexpr (Load) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bank);
        CgemmHalfCopyGM2L1(lh, highA, m, 0, cm, ck);
        CgemmHalfCopyGM2L1(rh, highB, 0, n, ck, cn);
        CgemmHalfCopyGM2L1(ll, lowA, m, 0, cm, ck);
        CgemmHalfCopyGM2L1(rl, lowB, 0, n, ck, cn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bank);
    }
    if constexpr (Compute) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bank);
        CgemmHalfL0MmadLoop(lowA, highB, cm, ck, cn, 1, 0, baseK, l0, true, false, cOffset);
        CgemmHalfL0MmadLoop(highA, lowB, cm, ck, cn, 1, 0, baseK, l0, false, false, cOffset);
        CgemmHalfL0MmadLoop(highA, highB, cm, ck, cn, 1, 0, baseK, l0, false, true, cOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bank);
    }
}

template <typename TensorOut>
__aicore__ inline void CgemmSbStoreHalfReduction(
    TensorOut ct, const CgemmSbCubeOperands<half>& o, uint32_t cm, uint32_t cn, uint32_t off, uint32_t outputSlot,
    uint64_t cOffset)
{
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
    if (off == 0)
        AscendC::SetAtomicNone();
    else
        AscendC::SetAtomicAdd<float>();
    CgemmHalfCopyOutL0C2GM(ct, o.m, o.n, cm, cn, cOffset);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(outputSlot);
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
        const uint64_t cacheBytes =
            2 * sizeof(half) * (RoundUp<uint64_t>(cm, CGEMM_HALF_FRACTAL) + RoundUp<uint64_t>(cn, CGEMM_HALF_FRACTAL)) *
            RoundUp<uint64_t>(ck, CGEMM_HALF_FP32_C0);
        if (t.cacheHalfInputs && cacheBytes <= CGEMM_HALF_L1_SIZE / CGEMM_HALF_L1_BUF_NUM) {
            CgemmSbCachedHalfProducts(
                lh, rh, lowl, lowr, cm, cn, ck, o.m, o.n, t.baseK, l1++ % CGEMM_HALF_L1_BUF_NUM, l0, cOffset);
        } else {
            CgemmHalfProcessKChunks(
                lowl, rh, ck, CUBE_L1_CHUNK_K, CeilDiv<uint64_t>(ck, CUBE_L1_CHUNK_K), t.baseK, o.m, o.n, cm, cn, l1,
                l0, true, false, cOffset);
            CgemmHalfProcessKChunks(
                lh, lowr, ck, CUBE_L1_CHUNK_K, CeilDiv<uint64_t>(ck, CUBE_L1_CHUNK_K), t.baseK, o.m, o.n, cm, cn, l1,
                l0, false, false, cOffset);
            if (t.mp <= SMALL_CUBE_TILE_DIM && t.np <= SMALL_CUBE_TILE_DIM && t.kp <= SMALL_CUBE_TILE_DIM &&
                ck <= t.baseK) {
                CgemmHalfReuseOperands(cm, cn, ck, l0, cOffset);
            } else {
                CgemmHalfProcessKChunks(
                    lh, rh, ck, CUBE_L1_CHUNK_K, CeilDiv<uint64_t>(ck, CUBE_L1_CHUNK_K), t.baseK, o.m, o.n, cm, cn, l1,
                    l0, false, true, cOffset);
            }
        }
        CgemmSbStoreHalfReduction(ct, o, cm, cn, off, outputSlot, cOffset);
    }
    AscendC::SetAtomicNone();
}

// Whole-K tiles fit in one L1 bank. Separate producer and consumer calls allow
// the next independent tile to load before the current tile enters MTE1/MMAD.
template <typename LeftPtn, typename RightPtn, bool Load>
__aicore__ inline void CgemmSbPrefetchedHalfTile(
    const CgemmSbCubeOperands<half>& o, const CgemmSbTiling& t, const CgemmSbCubePlan& p, uint64_t bank, uint64_t& l0)
{
    auto lh = CgemmSbMakeHalfTensor<LeftPtn>(o.left, p.effectiveN, p.effectiveK);
    auto rh = CgemmSbMakeHalfTensor<RightPtn>(o.right, p.effectiveK, t.mp);
    auto ll =
        CgemmSbMakeHalfTensor<LeftPtn>(o.left + CGEMM_SB_HALF_RESIDUAL_OFFSET * t.bSize, p.effectiveN, p.effectiveK);
    auto rl = CgemmSbMakeHalfTensor<RightPtn>(o.right + CGEMM_SB_HALF_RESIDUAL_OFFSET * t.aSize, p.effectiveK, t.mp);
    const uint32_t cm = Min<uint32_t>(t.tileM, p.effectiveN - o.m);
    const uint32_t cn = Min<uint32_t>(t.tileN, t.mp - o.n);
    const uint64_t cOffset = bank * CGEMM_SB_L0C_SLOT_BYTES;
    if constexpr (!Load)
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(bank);
    CgemmSbCachedHalfProducts<Load, !Load>(lh, rh, ll, rl, cm, cn, p.effectiveK, o.m, o.n, t.baseK, bank, l0, cOffset);
    if constexpr (!Load) {
        auto ct = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(o.out),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(p.effectiveN, t.mp));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        CgemmHalfCopyOutL0C2GM(ct, o.m, o.n, cm, cn, cOffset);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
    }
}

__aicore__ inline bool CgemmSbCanPrefetch(const CgemmSbTiling& t, const CgemmSbCubePlan& p)
{
    const uint64_t bytes = 2 * sizeof(half) *
                           (RoundUp<uint64_t>(Min(t.tileM, p.effectiveN), CGEMM_HALF_FRACTAL) +
                            RoundUp<uint64_t>(Min(t.tileN, t.mp), CGEMM_HALF_FRACTAL)) *
                           RoundUp<uint64_t>(p.effectiveK, CGEMM_HALF_FP32_C0);
    return t.cacheHalfInputs && p.effectiveK <= HALF_REDUCTION_CHUNK_K && p.doubleOutput &&
           bytes <= CGEMM_HALF_L1_SIZE / CGEMM_HALF_L1_BUF_NUM &&
           p.tiles * p.effectiveBatch * p.products > AscendC::GetBlockNum();
}

template <typename LeftPtn, typename RightPtn>
__aicore__ inline void CgemmSbPrefetchedHalfCube(
    __gm__ float* ws, const CgemmSbTiling& t, const CgemmSbCubePlan& p, uint64_t& l0)
{
    const uint64_t step = AscendC::GetBlockNum();
    const uint64_t total = p.tiles * p.effectiveBatch * p.products;
    uint64_t task = AscendC::GetBlockIdx();
    if (task >= total)
        return;
    auto current = CgemmSbDecodeCubeTask<half, false, false>(ws, t, p, task);
    CgemmSbPrefetchedHalfTile<LeftPtn, RightPtn, true>(current, t, p, 0, l0);
    for (uint64_t index = 0; task < total; task += step, ++index) {
        const uint64_t bank = index % CGEMM_HALF_L1_BUF_NUM;
        if (task + step < total) {
            auto next = CgemmSbDecodeCubeTask<half, false, false>(ws, t, p, task + step);
            CgemmSbPrefetchedHalfTile<LeftPtn, RightPtn, true>(next, t, p, (index + 1) % CGEMM_HALF_L1_BUF_NUM, l0);
        }
        current = CgemmSbDecodeCubeTask<half, false, false>(ws, t, p, task);
        CgemmSbPrefetchedHalfTile<LeftPtn, RightPtn, false>(current, t, p, bank, l0);
    }
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
    // Short-K exceptions are repaired from the original inputs by the final
    // Vector phase. Cube writes only scratch products; avoid making every AIC
    // scan the same producer flags before computing those disposable products.
    uint32_t mask = CGEMM_SB_REFERENCE_NONE;
    const bool deferredException = HALF && !Partial && !Prefix && t.cacheHalfInputs && t.k < CGEMM_SB_ORDERED_MIN_K;
    if (!deferredException)
        mask = CgemmSbReferenceMask(ws + t.flagOffset, t);
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
    if constexpr (HALF && !Partial && !Prefix) {
        if (CgemmSbCanPrefetch(t, plan)) {
            AscendC::SetAtomicNone();
            CgemmSbPrefetchedHalfCube<LeftPtn, RightPtn>(ws, t, plan, l0);
            CgemmSbCubePipelineFlags<false>();
            return;
        }
    }
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
