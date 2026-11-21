/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * csscal_kernel.h - Kernel implementation for aclblasCsscal
 * 功能：x[j] = alpha * x[j]（实数标量 alpha 乘以复数向量 x）
 * alpha 为 float 实数，x 为 complex64（实部虚部各 float32）
 */

#pragma once

#include "kernel_operator.h"
#include "csscal_tiling.h"

using namespace AscendC;

// Full vectors share one full mask; only the final partial vector updates it.
// FixedFloats lets the compiler specialize the complete-tile loop.
// Dual vlds/vsts move two vectors per instruction and break the WAR chain
// that serializes single-register load->muls->store iterations. On dav_3510
// the dual forms only support (de)interleave modes; DINTLV/INTLV_B32 split
// and re-join real/imag lanes, which is exact for a uniform real alpha.
template <uint32_t FixedFloats = 0>
__simd_vf__ inline void CsscalScaleVF(__ubuf__ float* data, float alpha, uint32_t runtimeFloats)
{
    constexpr uint32_t lanes = GetVecLen() / sizeof(float);
    const uint32_t floats = FixedFloats != 0 ? FixedFloats : runtimeFloats;
    const uint16_t fullVectors = static_cast<uint16_t>(floats / lanes);
    Reg::RegTensor<float> v0, v1, v2, v3;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    const uint16_t dualVectors = fullVectors >> 1;
    const uint16_t quadVectors = dualVectors >> 1;
    // Loads use hardware post-update addressing so their pointer advance costs
    // no scalar-pipe instruction; stores stay index-addressed off i.
    __ubuf__ float* lp = data;
    uint16_t i = 0;
    for (; i < quadVectors; ++i) {
        Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
            v0, v1, lp, 2 * lanes);
        Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
            v2, v3, lp, 2 * lanes);
        Reg::Muls(v0, v0, alpha, fullMask);
        Reg::Muls(v1, v1, alpha, fullMask);
        Reg::Muls(v2, v2, alpha, fullMask);
        Reg::Muls(v3, v3, alpha, fullMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(data + i * 4 * lanes, v0, v1, fullMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(data + (i * 4 + 2) * lanes, v2, v3, fullMask);
    }
    uint16_t doneDual = quadVectors * 2;
    if ((dualVectors & 1) != 0) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(v0, v1, data + doneDual * 2 * lanes);
        Reg::Muls(v0, v0, alpha, fullMask);
        Reg::Muls(v1, v1, alpha, fullMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(data + doneDual * 2 * lanes, v0, v1, fullMask);
        doneDual += 1;
    }
    uint32_t processed = static_cast<uint32_t>(doneDual) * 2 * lanes;
    if ((fullVectors & 1) != 0) {
        Reg::LoadAlign(v0, data + processed);
        Reg::Muls(v0, v0, alpha, fullMask);
        Reg::StoreAlign(data + processed, v0, fullMask);
        processed += lanes;
    }
    uint32_t tail = floats % lanes;
    if (tail != 0) {
        Reg::MaskReg tailMask = Reg::UpdateMask<float>(tail);
        Reg::LoadAlign(v0, data + processed);
        Reg::Muls(v0, v0, alpha, tailMask);
        Reg::StoreAlign(data + processed, v0, tailMask);
    }
}

template <uint32_t FixedFloats = 0>
__aicore__ inline void CsscalTile(GlobalTensor<float>& gm, LocalTensor<float>& ub,
                                uint64_t offset, uint32_t runtimeFloats, float alpha)
{
    const uint32_t floats = FixedFloats != 0 ? FixedFloats : runtimeFloats;
    if constexpr (FixedFloats != 0) {
        static_assert(FixedFloats % 8 == 0, "Full tile must be transfer aligned");
        DataCopy(ub, gm[offset], FixedFloats);
    } else if (floats % 8 == 0) {
        DataCopy(ub, gm[offset], floats);
    } else {
        DataCopyExtParams copy{1, static_cast<uint32_t>(floats * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(ub, gm[offset], copy, pad);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    CsscalScaleVF<FixedFloats>(reinterpret_cast<__ubuf__ float*>(ub.GetPhyAddr()), alpha, floats);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if constexpr (FixedFloats != 0) {
        DataCopy(gm[offset], ub, FixedFloats);
    } else if (floats % 8 == 0) {
        DataCopy(gm[offset], ub, floats);
    } else {
        DataCopyExtParams copy{1, static_cast<uint32_t>(floats * sizeof(float)), 0, 0, 0};
        DataCopyPad(gm[offset], ub, copy);
    }
}

// Owns UB [0, tileSize * 8) and event 0 on each dependency channel.
// Do not add other UB/event users without updating that ownership.
__aicore__ inline void CsscalContiguous(GM_ADDR x, uint32_t perCoreN,
                                        uint32_t remainder, uint32_t tileSize, float alpha)
{
    const uint32_t block = GetBlockIdx();
    uint32_t count = perCoreN + (block + 1 == GetBlockNum() ? remainder : 0);
    if (count == 0) return;
    uint64_t offset = static_cast<uint64_t>(block) * perCoreN * 2;
    GlobalTensor<float> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    LocalTensor<float> ub(TPosition::VECCALC, 0, tileSize * 2);
    if (count <= tileSize) {
        CsscalTile(gm, ub, offset, count * 2, alpha);
    } else {
        // The host only clamps tileSize when every core fits in one tile.
        // Thus all multi-tile cores use this compile-time capacity.
        do {
            CsscalTile<CSSCAL_TILE_CAPACITY * 2>(gm, ub, offset, 0, alpha);
            count -= CSSCAL_TILE_CAPACITY;
            if (count != 0) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                offset += CSSCAL_TILE_CAPACITY * 2;
            }
        } while (count >= CSSCAL_TILE_CAPACITY);
        if (count != 0) {
            CsscalTile(gm, ub, offset, count * 2, alpha);
        }
    }
    // Drain the final store before completing this kernel.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

// SIMT 模式：用于 incx != 1 的跳步访问
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void CsscalSimtCompute(
    uint32_t calNum, uint32_t startOffset, uint32_t stride, float alpha,
    __gm__ float* xGm)
{
    if (calNum == 0) {
        return;
    }

    // calNum 是复数元素个数
    // 每个复数 2 个 float：实部、虚部
    for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
        uint64_t idx = (static_cast<uint64_t>(startOffset) + i) * stride;
        uint64_t floatIdx = idx * 2;
        xGm[floatIdx] = alpha * xGm[floatIdx];      // 实部
        xGm[floatIdx + 1] = alpha * xGm[floatIdx + 1];  // 虚部
    }
}

