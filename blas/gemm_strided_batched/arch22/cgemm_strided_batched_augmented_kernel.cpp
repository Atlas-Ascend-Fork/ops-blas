/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cgemm_strided_batched_common.h"

namespace cgemm_sb {
using AscendC::Mul;

// A 32 x 32 complex tile has one extra 32-byte UB block per column.
constexpr uint32_t kAugmentedPackTile = 32U;
constexpr uint32_t kAugmentedPackInputPitchBytes = kAugmentedPackTile * 2U * sizeof(float) + 32U;
constexpr uint32_t kAugmentedPackInputElements = kAugmentedPackTile * kAugmentedPackInputPitchBytes / sizeof(float);

template <uint32_t tile>
__aicore__ inline void SbAugmentedIndices(
    LocalTensor<float> tmp, LocalTensor<float> row, LocalTensor<float> col, LocalTensor<float> rp,
    LocalTensor<float> cp, LocalTensor<float> src, LocalTensor<int32_t> ix, LocalTensor<float> sign)
{
    constexpr uint32_t size = 4U * tile * tile;
    CreateVecIndex(tmp, 0.0f, size);
    Muls(row, tmp, 1.0f / static_cast<float>(static_cast<int32_t>(2U * tile)), size);
    Cast(ix, row, RoundMode::CAST_FLOOR, size);
    Cast(row, ix, RoundMode::CAST_NONE, size);
    Muls(col, row, -static_cast<float>(static_cast<int32_t>(2U * tile)), size);
    Add(col, col, tmp, size);
    Muls(rp, row, 0.5f, size);
    Cast(ix, rp, RoundMode::CAST_FLOOR, size);
    Cast(rp, ix, RoundMode::CAST_NONE, size);
    Muls(cp, col, 0.5f, size);
    Cast(ix, cp, RoundMode::CAST_FLOOR, size);
    Cast(cp, ix, RoundMode::CAST_NONE, size);
    Muls(src, rp, static_cast<float>(static_cast<int32_t>((2U * tile + 8U) * 4U)), size);
    Muls(tmp, cp, 8.0f, size);
    Add(src, src, tmp, size);
    Muls(rp, rp, -2.0f, size);
    Add(rp, rp, row, size);
    Muls(cp, cp, -2.0f, size);
    Add(cp, cp, col, size);
    Mul(tmp, rp, cp, size);
    Muls(sign, tmp, 2.0f, size);
    Muls(row, rp, -2.0f, size);
    Add(sign, sign, row, size);
    Adds(sign, sign, 1.0f, size);
    Muls(tmp, tmp, -2.0f, size);
    Add(tmp, tmp, rp, size);
    Add(tmp, tmp, cp, size);
    Muls(tmp, tmp, 4.0f, size);
    Add(src, src, tmp, size);
    Cast(ix, src, RoundMode::CAST_RINT, size);
}

template <uint32_t tile>
__aicore__ inline LocalTensor<float> SbPrepareAugmentedIndices(
    TBuf<TPosition::VECCALC>& work, TBuf<TPosition::VECCALC>& indices, TBuf<TPosition::VECCALC>& signs)
{
    constexpr uint32_t size = 4U * tile * tile;
    auto tmp = work.Get<float>();
    auto row = work.GetWithOffset<float>(size, size * 4U);
    auto col = work.GetWithOffset<float>(size, size * 8U);
    auto rp = work.GetWithOffset<float>(size, size * 12U);
    auto cp = work.GetWithOffset<float>(size, size * 16U);
    auto src = work.GetWithOffset<float>(size, size * 20U);
    auto ix = indices.Get<int32_t>();
    auto sign = signs.Get<float>();
    SbAugmentedIndices<tile>(tmp, row, col, rp, cp, src, ix, sign);
    return sign;
}

// B^T [real, imag] times [[Ar,Ai],[-Ai,Ar]] writes interleaved C^T directly.
template <
    typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6,
    typename Arg7, typename Arg8, typename Arg9>
__aicore__ inline void SbPackAugmentedTile(
    Arg0& t, Arg1& inputQ, Arg2& outputQ, Arg3& sign, Arg4& offsets, Arg5& mt, Arg6& tiles, Arg7& ag, Arg8& ws,
    Arg9& global)
{
    constexpr uint32_t size = 4096U;
    const uint32_t batch = global / tiles;
    const uint32_t mb = (global % mt) * 32U;
    const uint32_t kb = ((global % tiles) / mt) * 32U;
    const uint32_t rows = t.m - mb < 32U ? t.m - mb : 32U;
    const uint32_t cols = t.k - kb < 32U ? t.k - kb : 32U;
    auto input = inputQ.template AllocTensor<float>();
    Duplicate(input, 0.0f, kAugmentedPackInputElements);
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams ic{
        static_cast<uint16_t>(cols), rows * 8U, (t.lda - rows) * 8U, (kAugmentedPackInputPitchBytes - rows * 8U) / 32U,
        0U};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(
        input, ag[(static_cast<int64_t>(batch) * t.strideA + static_cast<int64_t>(kb) * t.lda + mb) * 2], ic, pad);
    inputQ.EnQue(input);
    input = inputQ.template DeQue<float>();
    auto out = outputQ.template AllocTensor<float>();
    Gather(out, input, offsets, 0U, size);
    Mul(out, out, sign, size);
    PipeBarrier<PIPE_ALL>();
    inputQ.FreeTensor(input);
    outputQ.EnQue(out);
    out = outputQ.template DeQue<float>();
    DataCopyExtParams oc{static_cast<uint16_t>(cols * 2U), rows * 8U, (64U - rows * 2U) / 8U, (t.m - rows) * 8U, 0U};
    DataCopyPad(ws[batch * t.workspaceStride + static_cast<uint64_t>(kb * 2U) * t.m * 2U + mb * 2U], out, oc);
    outputQ.FreeTensor(out);
}

extern "C" __attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_augmented_pack_kernel(
    GM_ADDR a, GM_ADDR workspace, const CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t size = 4096U;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inputQ;
    TQue<TPosition::VECOUT, 1> outputQ;
    TBuf<TPosition::VECCALC> work, indices, signs;
    pipe.InitBuffer(inputQ, 1, kAugmentedPackInputElements * sizeof(float));
    pipe.InitBuffer(outputQ, 1, size * sizeof(float));
    pipe.InitBuffer(work, 6U * size * sizeof(float));
    pipe.InitBuffer(indices, size * sizeof(uint32_t));
    pipe.InitBuffer(signs, size * sizeof(float));
    auto sign = SbPrepareAugmentedIndices<32U>(work, indices, signs);
    PipeBarrier<PIPE_ALL>();
    auto offsets = indices.Get<uint32_t>();
    const uint32_t mt = (t.m + 31U) / 32U;
    const uint32_t kt = (t.k + 31U) / 32U;
    const uint32_t tiles = mt * kt;
    GlobalTensor<float> ag, ws;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    for (uint32_t global = GetBlockIdx(); global < tiles * t.batchCount; global += GetBlockNum()) {
        SbPackAugmentedTile(t, inputQ, outputQ, sign, offsets, mt, tiles, ag, ws, global);
    }
}

template <
    uint32_t tile, uint32_t groupCap, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5>
__aicore__ inline void SbRunCompactPack(Arg0& a, Arg1& workspace, Arg2& t, Arg3& work, Arg4& indices, Arg5& sign)
{
    constexpr uint32_t size = 4U * tile * tile;
    constexpr uint32_t inputSize = tile * (tile * 2U + 8U);
    auto offsets = indices.template Get<uint32_t>();
    auto input = work.template Get<float>();
    auto output = work.template GetWithOffset<float>(groupCap * size, groupCap * inputSize * sizeof(float));
    const uint32_t balanced = (t.batchCount + GetBlockNum() - 1U) / GetBlockNum();
    const uint32_t group = balanced < groupCap ? balanced : groupCap;
    GlobalTensor<float> ag, ws;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    for (uint32_t first = GetBlockIdx() * group; first < t.batchCount; first += GetBlockNum() * group) {
        const uint32_t nb = t.batchCount - first < group ? t.batchCount - first : group;
        DataCopyExtParams ic{static_cast<uint16_t>(nb * tile), tile * 8U, 0U, 1U, 0U};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(input, ag[static_cast<uint64_t>(first) * tile * tile * 2U], ic, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t batch = 0; batch < nb; ++batch) {
            Gather(output[batch * size], input, offsets, batch * inputSize * sizeof(float), size);
            Mul(output[batch * size], output[batch * size], sign, size);
        }
        SbStoreAugmentedGroup<size>(ws, output, t.workspaceStride, first, nb);
    }
}

template <uint32_t tile, uint32_t groupCap>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_augmented_compact_pack_kernel(
    GM_ADDR a, GM_ADDR workspace, const CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t size = 4U * tile * tile;
    constexpr uint32_t inputSize = tile * (tile * 2U + 8U);
    constexpr uint32_t transferSize = groupCap * (inputSize + size);
    constexpr uint32_t workSize = transferSize > 6U * size ? transferSize : 6U * size;
    TPipe pipe;
    TBuf<TPosition::VECCALC> work, indices, signs;
    pipe.InitBuffer(work, workSize * sizeof(float));
    pipe.InitBuffer(indices, size * sizeof(uint32_t));
    pipe.InitBuffer(signs, size * sizeof(float));
    auto sign = SbPrepareAugmentedIndices<tile>(work, indices, signs);
    PipeBarrier<PIPE_ALL>();
    SbRunCompactPack<tile, groupCap>(a, workspace, t, work, indices, sign);
}

} // namespace cgemm_sb

namespace cgemm_sb {
template <uint32_t baseM, uint32_t baseN, uint32_t tileK>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_augmented_cube_kernel(
    GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    SetHF32Mode(HF32Mode::DISABLE);
    const uint32_t tiles = ((t.n + baseM - 1U) / baseM) * ((t.m * 2U + baseN - 1U) / baseN);
    for (uint32_t global = GetBlockIdx(); global < tiles * t.batchCount;) {
        const uint32_t batch = global / tiles;
        CgemmSbAugmentedMmad<tileK, baseM, baseN> mm;
        mm.Init(
            reinterpret_cast<__gm__ float*>(b) + static_cast<int64_t>(batch) * t.strideB * 2,
            reinterpret_cast<__gm__ float*>(workspace) + batch * t.workspaceStride,
            reinterpret_cast<__gm__ float*>(c) + static_cast<int64_t>(batch) * t.strideC * 2, t.n, t.m * 2U, t.k * 2U,
            t.ldb * 2U, t.ldc * 2U);
        mm.PrepareCo1();
        mm.Process(global % tiles, GetBlockNum());
        global = (batch + 1U) * tiles;
        global += (GetBlockIdx() + GetBlockNum() - global % GetBlockNum()) % GetBlockNum();
    }
}

template <uint32_t baseM, uint32_t baseN>
void cgemm_sb_augmented_launch(
    GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling& t, uint32_t cores, void* stream)
{
    if (t.k % 64U == 0U) {
        cgemm_sb_augmented_cube_kernel<baseM, baseN, 128U><<<cores, nullptr, stream>>>(b, c, workspace, t);
    } else if (t.k % 32U == 0U) {
        cgemm_sb_augmented_cube_kernel<baseM, baseN, 64U><<<cores, nullptr, stream>>>(b, c, workspace, t);
    } else {
        cgemm_sb_augmented_cube_kernel<baseM, baseN, 16U><<<cores, nullptr, stream>>>(b, c, workspace, t);
    }
}

template <
    uint32_t tile, uint32_t groupCap, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5>
__aicore__ inline void SbRunAugmentedCube(Arg0& b, Arg1& c, Arg2& workspace, Arg3& t, Arg4& cores, Arg5& group)
{
    SetHF32Mode(HF32Mode::DISABLE);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    uint32_t rounds = 0U;
    for (uint32_t first = GetBlockIdx() * group * 2U; first < t.batchCount; first += cores * group * 2U) {
        CrossCoreWaitFlag<2>(8U);
        const uint32_t count = t.batchCount - first < group * 2U ? t.batchCount - first : group * 2U;
        CgemmSbAugmentedMmad<tile * 2U, (tile < 16U ? 16U : tile), tile * 2U> mm;
        mm.Init(
            reinterpret_cast<__gm__ float*>(b) + static_cast<int64_t>(first) * t.strideB * 2,
            reinterpret_cast<__gm__ float*>(workspace) + first * t.workspaceStride,
            reinterpret_cast<__gm__ float*>(c) + static_cast<int64_t>(first) * t.strideC * 2, t.n, t.m * 2U, t.k * 2U,
            t.ldb * 2U, t.ldc * 2U);
        const bool canPreload = t.strideB >= 0 && t.strideB <= 32767 && t.workspaceStride <= 65535U;
        if (canPreload) {
            mm.Preload(count, static_cast<uint32_t>(t.strideB * 2), static_cast<uint32_t>(t.workspaceStride));
        }
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t batch = first + i;
            mm.Init(
                reinterpret_cast<__gm__ float*>(b) + static_cast<int64_t>(batch) * t.strideB * 2,
                reinterpret_cast<__gm__ float*>(workspace) + batch * t.workspaceStride,
                reinterpret_cast<__gm__ float*>(c) + static_cast<int64_t>(batch) * t.strideC * 2, t.n, t.m * 2U,
                t.k * 2U, t.ldb * 2U, t.ldc * 2U);
            mm.Process(0U, 1U, true, canPreload, i);
        }
        ++rounds;
        if (rounds % 16U == 0U || first + cores * group * 2U >= t.batchCount) {
            CrossCoreSetFlag<2, PIPE_FIX>(9U);
        }
    }
}

template <
    uint32_t tile, uint32_t groupCap, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7>
__aicore__ inline void SbPackAugmentedBatch(
    Arg0& t, Arg1& offsets, Arg2& input, Arg3& output, Arg4& ag, Arg5& ws, Arg6& first, Arg7& nb)
{
    constexpr uint32_t size = 4U * tile * tile;
    constexpr uint32_t inputSize = tile * (tile * 2U + 8U);
    DataCopyExtParams ic{static_cast<uint16_t>(nb * tile), tile * 8U, 0U, 1U, 0U};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(input, ag[static_cast<uint64_t>(first) * tile * tile * 2U], ic, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t batch = 0; batch < nb; ++batch) {
        for (uint32_t offset = 0; offset < size; offset += 8192U) {
            const uint32_t part = size - offset < 8192U ? size - offset : 8192U;
            Gather(output[batch * size + offset], input, offsets[offset], batch * inputSize * sizeof(float), part);
        }
        uint64_t mask[2] = {tile == 8U ? 0x5555ULL : (tile == 16U ? 0x55555555ULL : 0x5555555555555555ULL), 0U};
        for (uint32_t col = 0; col < 2U * tile; col += 64U) {
            Muls(
                output[batch * size + 2U * tile + col], output[batch * size + 2U * tile + col], -1.0f, mask, tile,
                {1, 1, tile / 2U, tile / 2U});
        }
    }
    SbStoreAugmentedGroup<size>(ws, output, t.workspaceStride, first, nb);
}

template <
    uint32_t tile, uint32_t groupCap, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7>
__aicore__ inline void SbRunAugmentedPack(
    Arg0& a, Arg1& workspace, Arg2& t, Arg3& cores, Arg4& group, Arg5& work, Arg6& indices, Arg7& ix)
{
    constexpr uint32_t size = 4U * tile * tile;
    constexpr uint32_t inputSize = tile * (tile * 2U + 8U);
    for (uint32_t rows = 1U; rows < tile; rows *= 2U) {
        SbIndexAdd(ix[rows * 4U * tile], ix, static_cast<int32_t>(rows * (2U * tile + 8U) * 4U), rows * 4U * tile);
    }
    PipeBarrier<PIPE_ALL>();
    auto offsets = indices.template Get<uint32_t>();
    auto input = work.template Get<float>();
    auto output = work.template GetWithOffset<float>(groupCap * size, groupCap * inputSize * sizeof(float));
    GlobalTensor<float> ag, ws;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    const uint32_t block = GetBlockIdx() / 2U;
    const uint32_t sub = GetSubBlockIdx();
    uint32_t rounds = 0;
    for (uint32_t base = block * group * 2U; base < t.batchCount; base += cores * group * 2U) {
        if (rounds != 0U && rounds % 16U == 0U) {
            CrossCoreWaitFlag<2>(9U);
        }
        const uint32_t first = base + sub * group;
        const uint32_t nb = first >= t.batchCount ? 0U : (t.batchCount - first < group ? t.batchCount - first : group);
        if (nb != 0U) {
            SbPackAugmentedBatch<tile, groupCap>(t, offsets, input, output, ag, ws, first, nb);
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(8U);
        ++rounds;
    }
    if (rounds != 0U) {
        CrossCoreWaitFlag<2>(9U);
    }
}

template <uint32_t tile, uint32_t groupCap>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_augmented_mixed_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const uint32_t balanced = (t.batchCount + cores * 2U - 1U) / (cores * 2U);
    const uint32_t group = balanced < groupCap ? balanced : groupCap;
    if ASCEND_IS_AIV {
        constexpr uint32_t size = 4U * tile * tile;
        constexpr uint32_t inputSize = tile * (tile * 2U + 8U);
        constexpr uint32_t transferSize = groupCap * (inputSize + size);
        constexpr uint32_t workSize = transferSize;
        TPipe pipe;
        TBuf<TPosition::VECCALC> work, indices;
        pipe.InitBuffer(work, workSize * sizeof(float));
        pipe.InitBuffer(indices, size * sizeof(uint32_t));
        auto ix = indices.Get<int32_t>();
        SetMaskNorm();
        SbIndex(ix, 0, 16U);
        SbIndexMul(ix, ix, 4, 16U);
        SbIndex(ix[2U * tile], 1, 16U);
        uint64_t oddSeed[2] = {0xaaaaULL, 0U};
        Adds(ix[2U * tile], ix[2U * tile], -2, oddSeed, 1U, {1, 1, 8, 8});
        SbIndexMul(ix[2U * tile], ix[2U * tile], 4, 16U);
        for (uint32_t columns = 16U; columns < 2U * tile; columns *= 2U) {
            SbIndexAdd(ix[columns], ix, static_cast<int32_t>(columns * 4U), columns);
            SbIndexAdd(ix[2U * tile + columns], ix[2U * tile], static_cast<int32_t>(columns * 4U), columns);
        }
        SbRunAugmentedPack<tile, groupCap>(a, workspace, t, cores, group, work, indices, ix);
    }
    if ASCEND_IS_AIC {
        AscendC::InitSocState();
        SbRunAugmentedCube<tile, groupCap>(b, c, workspace, t, cores, group);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
}

// Evaluate a small grid of products in one Cube operation and store its diagonal.
template <
    uint32_t tile, uint32_t group, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__aicore__ inline void SbPackDiagonalBatch(
    Arg0& t, Arg1& input, Arg2& output, Arg3& offsets, Arg4& ag, Arg5& ws, Arg6& sub, Arg7& rounds, Arg8& first,
    Arg9& start, Arg10& nb)
{
    constexpr uint32_t part = group / 2U;
    constexpr uint32_t inputPitch = 2U * tile + 8U;
    constexpr uint32_t inputSize = tile * inputPitch;
    constexpr uint32_t row = 2U * tile * part;
    constexpr uint32_t size = 4U * tile * tile * part;
    if (nb != part) {
        Duplicate(input, 0.0f, part * inputSize);
        PipeBarrier<PIPE_ALL>();
    }
    if (nb != 0U) {
        DataCopyExtParams ic{static_cast<uint16_t>(nb * tile), tile * 8U, 0U, 1U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0f};
        DataCopyPad(input, ag[static_cast<uint64_t>(start) * tile * tile * 2U], ic, noPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    Gather(output, input, offsets, 0U, size);
    uint64_t mask[2] = {0x5555555555555555ULL, 0U};
    for (uint32_t col = 0U; col < row; col += 64U) {
        Muls(output[row + col], output[row + col], -1.0f, mask, tile, {1, 1, row / 4U, row / 4U});
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams oc{static_cast<uint16_t>(2U * tile), row * 4U, 0U, row * 4U, 0U};
    DataCopyPad(ws[first * SbCompactWorkspaceStride(t.m) + sub * row], output, oc);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    CrossCoreSetFlag<2, PIPE_MTE3>(8U);
    ++rounds;
}

template <uint32_t tile, uint32_t group, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__aicore__ inline void SbRunDiagonalCube(Arg0& b, Arg1& c, Arg2& workspace, Arg3& t, Arg4& cores)
{
    AscendC::InitSocState();
    SetHF32Mode(HF32Mode::DISABLE);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    uint32_t rounds = 0U;
    for (uint32_t first = GetBlockIdx() * group; first < t.batchCount; first += cores * group) {
        CrossCoreWaitFlag<2>(8U);
        const uint32_t nb = t.batchCount - first < group ? t.batchCount - first : group;
        CgemmSbAugmentedMmad<2U * tile, tile * group, 2U * tile * group> mm;
        mm.Init(
            reinterpret_cast<__gm__ float*>(b) + static_cast<uint64_t>(first) * tile * tile * 2U,
            reinterpret_cast<__gm__ float*>(workspace) + first * SbCompactWorkspaceStride(t.m),
            reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(first) * tile * tile * 2U, nb * tile,
            group * tile * 2U, tile * 2U, tile * 2U, tile * 2U);
        mm.Process(0U, 1U, true, false, 0U, tile, nb);
        ++rounds;
        if (rounds % 8U == 0U || first + cores * group >= t.batchCount) {
            CrossCoreSetFlag<2, PIPE_FIX>(9U);
        }
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}

template <
    uint32_t tile, uint32_t group, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7>
__aicore__ inline void SbExpandDiagonalIndices(
    Arg0& a, Arg1& workspace, Arg2& t, Arg3& cores, Arg4& indexBuf, Arg5& input, Arg6& output, Arg7& ix)
{
    constexpr uint32_t part = group / 2U;
    constexpr uint32_t inputPitch = 2U * tile + 8U;
    constexpr uint32_t inputSize = tile * inputPitch;
    constexpr uint32_t row = 2U * tile * part;
    for (uint32_t cols = 16U; cols < 2U * tile; cols *= 2U) {
        SbIndexAdd(ix[cols], ix, static_cast<int32_t>(cols * 4U), cols);
        SbIndexAdd(ix[row + cols], ix[row], static_cast<int32_t>(cols * 4U), cols);
    }
    for (uint32_t batch = 1U; batch < part; batch *= 2U) {
        SbIndexAdd(ix[batch * 2U * tile], ix, static_cast<int32_t>(batch * inputSize * 4U), batch * 2U * tile);
        SbIndexAdd(
            ix[row + batch * 2U * tile], ix[row], static_cast<int32_t>(batch * inputSize * 4U), batch * 2U * tile);
    }
    for (uint32_t rows = 1U; rows < tile; rows *= 2U) {
        SbIndexAdd(ix[rows * 2U * row], ix, static_cast<int32_t>(rows * inputPitch * 4U), rows * 2U * row);
    }
    PipeBarrier<PIPE_ALL>();
    auto offsets = indexBuf.template Get<uint32_t>();
    GlobalTensor<float> ag, ws;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    const uint32_t sub = GetSubBlockIdx();
    uint32_t rounds = 0U;
    for (uint32_t first = (GetBlockIdx() / 2U) * group; first < t.batchCount; first += cores * group) {
        if (rounds != 0U && rounds % 8U == 0U) {
            CrossCoreWaitFlag<2>(9U);
        }
        const uint32_t start = first + sub * part;
        const uint32_t nb = start >= t.batchCount ? 0U : (t.batchCount - start < part ? t.batchCount - start : part);
        SbPackDiagonalBatch<tile, group>(t, input, output, offsets, ag, ws, sub, rounds, first, start, nb);
    }
    if (rounds != 0U) {
        CrossCoreWaitFlag<2>(9U);
    }
}

template <uint32_t tile, uint32_t group>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_diagonal_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    constexpr uint32_t part = group / 2U;
    constexpr uint32_t inputPitch = 2U * tile + 8U;
    constexpr uint32_t inputSize = tile * inputPitch;
    constexpr uint32_t row = 2U * tile * part;
    constexpr uint32_t size = 4U * tile * tile * part;
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> inputBuf, outputBuf, indexBuf;
        pipe.InitBuffer(inputBuf, part * inputSize * sizeof(float));
        pipe.InitBuffer(outputBuf, size * sizeof(float));
        pipe.InitBuffer(indexBuf, size * sizeof(uint32_t));
        auto input = inputBuf.Get<float>();
        auto output = outputBuf.Get<float>();
        auto ix = indexBuf.Get<int32_t>();
        SetMaskNorm();
        SbIndex(ix, 0, 16U);
        SbIndexMul(ix, ix, 4, 16U);
        SbIndex(ix[row], 1, 16U);
        uint64_t oddSeed[2] = {0xaaaaULL, 0U};
        Adds(ix[row], ix[row], -2, oddSeed, 1U, {1, 1, 8, 8});
        SbIndexMul(ix[row], ix[row], 4, 16U);
        SbExpandDiagonalIndices<tile, group>(a, workspace, t, cores, indexBuf, input, output, ix);
    }
    if ASCEND_IS_AIC {
        SbRunDiagonalCube<tile, group>(b, c, workspace, t, cores);
    }
}

} // namespace cgemm_sb

// ---- bulk implementation ----
namespace cgemm_sb {
template <uint32_t tile, uint32_t group, uint32_t chunks>
__aicore__ inline void SbBulkCopyInput(
    LocalTensor<float> input, GlobalTensor<float> ag, uint32_t sub, uint32_t batchCount, uint32_t first)
{
    constexpr uint32_t part = group / 2U;
    constexpr uint32_t inputMatrix = tile * tile * 2U;
    constexpr uint32_t inputSize = tile * (2U * tile);

    const uint32_t nb = batchCount - first < group * chunks ? batchCount - first : group * chunks;
    const uint32_t count = (nb + group - 1U) / group;
    if (nb == group * chunks) {
        DataCopyExtParams load{static_cast<uint16_t>(chunks), part * inputMatrix * 4U, part * inputMatrix * 4U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        DataCopyPad(input, ag[static_cast<uint64_t>(first + sub * part) * inputMatrix], load, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        return;
    }
    if (nb != group * chunks) {
        Duplicate(input, 0.0f, chunks * part * inputSize);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    for (uint32_t i = 0U; i < count; ++i) {
        const uint32_t offset = i * group + sub * part;
        const uint32_t active = offset >= nb ? 0U : (nb - offset < part ? nb - offset : part);
        if (active != 0U) {
            DataCopyExtParams load{static_cast<uint16_t>(active * tile), tile * 8U, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
            DataCopyPad(
                input[i * part * inputSize], ag[static_cast<uint64_t>(first + offset) * inputMatrix], load, pad);
        }
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

// Pack multiple diagonal products together and preload their operands into L1.
template <
    uint32_t tile, uint32_t group, uint32_t chunks, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
__aicore__ inline void SbRunBulkCube(Arg0& b, Arg1& c, Arg2& workspace, Arg3& t, Arg4& cores)
{
    constexpr uint32_t inputMatrix = tile * tile * 2U;
    constexpr uint32_t rightMatrix = 4U * tile * tile * group;
    uint32_t rounds = 0U;
    for (uint32_t first = GetBlockIdx() * group * chunks; first < t.batchCount; first += cores * group * chunks) {
        CrossCoreWaitFlag<2>(8U);
        const uint32_t nb = t.batchCount - first < group * chunks ? t.batchCount - first : group * chunks;
        const uint32_t full = nb / group;
        auto* bg = reinterpret_cast<__gm__ float*>(b) + static_cast<uint64_t>(first) * inputMatrix;
        auto* ws = reinterpret_cast<__gm__ float*>(workspace) + first * SbCompactWorkspaceStride(t.m);
        auto* cg = reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(first) * inputMatrix;
        CgemmSbAugmentedMmad<2U * tile, tile * group, 2U * tile * group, false, true> mm;
        mm.Init(bg, ws, cg, group * tile, group * 2U * tile, 2U * tile, 2U * tile, 2U * tile);
        if (full != 0U) {
            mm.Preload(full, group * inputMatrix, rightMatrix);
        }
        for (uint32_t i = 0U; i < full; ++i) {
            mm.Init(
                bg + i * group * inputMatrix, ws + i * rightMatrix, cg + i * group * inputMatrix, group * tile,
                group * 2U * tile, 2U * tile, 2U * tile, 2U * tile);
            mm.Process(0U, 1U, true, true, i, tile, group, 0U, 0U, i % 2U);
        }
        if (nb % group != 0U) {
            mm.Init(
                bg + full * group * inputMatrix, ws + full * rightMatrix, cg + full * group * inputMatrix,
                (nb % group) * tile, group * 2U * tile, 2U * tile, 2U * tile, 2U * tile);
            mm.Process(0U, 1U, true, false, 0U, tile, nb % group, 0U, 0U, full % 2U);
        }
        ++rounds;
        if (rounds % 8U == 0U || first + cores * group * chunks >= t.batchCount) {
            CrossCoreSetFlag<2, PIPE_FIX>(9U);
        }
    }
}

template <
    uint32_t tile, uint32_t group, uint32_t chunks, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
__aicore__ inline void SbPackBulkBatch(
    Arg0& t, Arg1& cores, Arg2& input, Arg3& output, Arg4& ag, Arg5& ws, Arg6& offsets, Arg7& sub, Arg8& rounds,
    Arg9& first)
{
    constexpr uint32_t part = group / 2U;
    constexpr uint32_t inputPitch = 2U * tile;
    constexpr uint32_t inputSize = tile * inputPitch;
    constexpr uint32_t row = 2U * tile * part;
    constexpr uint32_t size = 4U * tile * tile * part;
    if (rounds != 0U && rounds % 8U == 0U) {
        CrossCoreWaitFlag<2>(9U);
    }
    const uint32_t nb = t.batchCount - first < group * chunks ? t.batchCount - first : group * chunks;
    const uint32_t count = (nb + group - 1U) / group;
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (rounds % 2U == 0U) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    } else {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }
    auto packed = output[rounds % 2U * chunks * size];
    for (uint32_t chunk = 0U; chunk < count; ++chunk) {
        SbGather(packed[chunk * size], input, offsets, chunk * part * inputSize * 4U, size);
    }
    const uint32_t next = first + cores * group * chunks;
    if (next < t.batchCount) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SbBulkCopyInput<tile, group, chunks>(input, ag, sub, t.batchCount, next);
    }
    uint64_t sign[2] = {0x5555555555555555ULL, 0U};
    for (uint32_t col = 0U; col < row; col += 64U) {
        Muls(
            packed[row + col], packed[row + col], -1.0f, sign, static_cast<uint8_t>(tile * count),
            {1, 1, row / 4U, row / 4U});
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams cp{static_cast<uint16_t>(2U * tile * count), row * 4U, 0U, row * 4U, 0U};
    DataCopyPad(ws[first * SbCompactWorkspaceStride(t.m) + sub * row], packed, cp);
}

template <
    uint32_t tile, uint32_t group, uint32_t chunks, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7>
__aicore__ inline void SbRunBulkBatches(
    Arg0& a, Arg1& workspace, Arg2& t, Arg3& cores, Arg4& xb, Arg5& input, Arg6& output, Arg7& ix)
{
    constexpr uint32_t part = group / 2U;
    constexpr uint32_t inputPitch = 2U * tile;
    constexpr uint32_t row = 2U * tile * part;
    for (uint32_t rows = 1U; rows < tile; rows *= 2U) {
        SbIndexAdd(ix[rows * 2U * row], ix, rows * inputPitch * 4, rows * 2U * row);
    }
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> ag, ws;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    auto offsets = xb.template Get<uint32_t>();
    const uint32_t sub = GetSubBlockIdx();

    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    if (GetBlockIdx() / 2U * group * chunks < t.batchCount) {
        SbBulkCopyInput<tile, group, chunks>(input, ag, sub, t.batchCount, GetBlockIdx() / 2U * group * chunks);
    }
    uint32_t rounds = 0U;
    for (uint32_t first = GetBlockIdx() / 2U * group * chunks; first < t.batchCount; first += cores * group * chunks) {
        SbPackBulkBatch<tile, group, chunks>(t, cores, input, output, ag, ws, offsets, sub, rounds, first);
        SbSignalPackedRound(rounds);
        ++rounds;
    }
    SbDrainPackedOutput<false>(rounds);
}

template <
    uint32_t tile, uint32_t group, uint32_t chunks, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
__aicore__ inline void SbRunBulkVector(Arg0& a, Arg1& workspace, Arg2& t, Arg3& cores, Arg4& pipe)
{
    constexpr uint32_t part = group / 2U;
    constexpr uint32_t inputPitch = 2U * tile;
    constexpr uint32_t inputSize = tile * inputPitch;
    constexpr uint32_t row = 2U * tile * part;
    constexpr uint32_t size = 4U * tile * tile * part;
    TBuf<TPosition::VECCALC> ib, ob, xb;
    pipe.InitBuffer(ib, chunks * part * inputSize * 4U);
    pipe.InitBuffer(ob, 2U * chunks * size * 4U);
    pipe.InitBuffer(xb, size * 4U);
    auto input = ib.template Get<float>();
    auto output = ob.template Get<float>();
    auto ix = xb.template Get<int32_t>();
    SetMaskNorm();
    SbIndex(ix, 0, 16U);
    SbIndexMul(ix, ix, 4, 16U);
    SbIndex(ix[row], 1, 16U);
    uint64_t odd[2] = {0xaaaaULL, 0U};
    Adds(ix[row], ix[row], -2, odd, 1U, {1, 1, 8, 8});
    SbIndexMul(ix[row], ix[row], 4, 16U);
    for (uint32_t col = 16U; col < 2U * tile; col *= 2U) {
        SbIndexAdd(ix[col], ix, col * 4, col);
        SbIndexAdd(ix[row + col], ix[row], col * 4, col);
    }
    for (uint32_t batch = 1U; batch < part; batch *= 2U) {
        SbIndexAdd(ix[batch * 2U * tile], ix, batch * inputSize * 4, batch * 2U * tile);
        SbIndexAdd(ix[row + batch * 2U * tile], ix[row], batch * inputSize * 4, batch * 2U * tile);
    }
    SbRunBulkBatches<tile, group, chunks>(a, workspace, t, cores, xb, input, output, ix);
}

template <uint32_t tile, uint32_t group, uint32_t chunks>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_bulk_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    constexpr uint32_t part = group / 2U;

    constexpr uint32_t inputPitch = 2U * tile;

    if ASCEND_IS_AIV {
        TPipe pipe;
        SbRunBulkVector<tile, group, chunks>(a, workspace, t, cores, pipe);
    }
    if ASCEND_IS_AIC {
        InitSocState();
        SetHF32Mode(HF32Mode::DISABLE);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        SbRunBulkCube<tile, group, chunks>(b, c, workspace, t, cores);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
}
} // namespace cgemm_sb

bool cgemm_sb::CgemmSbDispatcher::LaunchDiagonal()
{
    if (!(CgemmSbAugmentedShape(tiling) && tiling.m == tiling.n && tiling.m == tiling.k && tiling.lda == tiling.m &&
          tiling.ldb == tiling.m && tiling.strideA == static_cast<int64_t>(tiling.m) * tiling.m &&
          tiling.strideB == tiling.strideA && tiling.strideC == tiling.strideA &&
          reinterpret_cast<uintptr_t>(c) % 64U == 0U && (tiling.m == 16U || tiling.m == 32U || tiling.m == 64U))) {
        return false;
    }

    if (tiling.m == 16U) {
        cgemm_sb::cgemm_sb_diagonal_kernel<16U, 4U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else if (tiling.m == 32U && tiling.batchCount >= 512U) {
        cgemm_sb::cgemm_sb_bulk_kernel<32U, 2U, 4U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else if (tiling.m == 32U) {
        cgemm_sb::cgemm_sb_diagonal_kernel<32U, 2U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else if (tiling.m == 64U) {
        cgemm_sb::cgemm_sb_augmented_mixed_kernel<64U, 1U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, tiling, aicCores);
    }
    return true;
}

bool cgemm_sb::CgemmSbDispatcher::LaunchGroupedAugmented()
{
    if (!(tiling.m == tiling.n && tiling.m == tiling.k && tiling.lda == tiling.m &&
          tiling.strideA == static_cast<int64_t>(tiling.m) * tiling.k &&
          (tiling.m == 8U || tiling.m == 16U || tiling.m == 32U))) {
        return false;
    }

    if (tiling.m == 8U) {
        cgemm_sb::cgemm_sb_augmented_mixed_kernel<8U, 32U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, tiling, aicCores);
    } else if (tiling.m == 16U) {
        cgemm_sb::cgemm_sb_augmented_mixed_kernel<16U, 8U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, tiling, aicCores);
    } else {
        cgemm_sb::cgemm_sb_augmented_mixed_kernel<32U, 4U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, tiling, aicCores);
    }
    return true;
}

void cgemm_sb::CgemmSbDispatcher::PackAugmented()
{
    if (tiling.m == tiling.k && tiling.lda == tiling.m && tiling.strideA == static_cast<int64_t>(tiling.m) * tiling.k &&
        tiling.m <= 32U) {
        if (tiling.m == 8U) {
            cgemm_sb::cgemm_sb_augmented_compact_pack_kernel<8U, 32U>
                <<<aivCores, nullptr, stream>>>(a, workspace, tiling);
        } else if (tiling.m == 16U) {
            cgemm_sb::cgemm_sb_augmented_compact_pack_kernel<16U, 8U>
                <<<aivCores, nullptr, stream>>>(a, workspace, tiling);
        } else if (tiling.m == 32U) {
            cgemm_sb::cgemm_sb_augmented_compact_pack_kernel<32U, 4U>
                <<<aivCores, nullptr, stream>>>(a, workspace, tiling);
        } else {
            cgemm_sb::cgemm_sb_augmented_pack_kernel<<<aivCores, nullptr, stream>>>(a, workspace, tiling);
        }
    } else {
        cgemm_sb::cgemm_sb_augmented_pack_kernel<<<aivCores, nullptr, stream>>>(a, workspace, tiling);
    }
}

void cgemm_sb::CgemmSbDispatcher::ComputeAugmented()
{
    if (tiling.m >= 64U && tiling.n >= 64U && tiling.k >= 64U) {
        cgemm_sb::cgemm_sb_augmented_launch<64U, 128U>(b, c, workspace, tiling, aicCores, stream);
    } else if (tiling.n <= 16U) {
        if (tiling.m <= 8U) {
            cgemm_sb::cgemm_sb_augmented_launch<16U, 16U>(b, c, workspace, tiling, aicCores, stream);
        } else if (tiling.m <= 16U) {
            cgemm_sb::cgemm_sb_augmented_launch<16U, 32U>(b, c, workspace, tiling, aicCores, stream);
        } else {
            cgemm_sb::cgemm_sb_augmented_launch<16U, 64U>(b, c, workspace, tiling, aicCores, stream);
        }
    } else {
        if (tiling.m <= 8U) {
            cgemm_sb::cgemm_sb_augmented_launch<32U, 16U>(b, c, workspace, tiling, aicCores, stream);
        } else if (tiling.m <= 16U) {
            cgemm_sb::cgemm_sb_augmented_launch<32U, 32U>(b, c, workspace, tiling, aicCores, stream);
        } else {
            cgemm_sb::cgemm_sb_augmented_launch<32U, 64U>(b, c, workspace, tiling, aicCores, stream);
        }
    }
}

bool cgemm_sb::CgemmSbDispatcher::LaunchAugmented()
{
    if (!(CgemmSbAugmentedShape(tiling) && reinterpret_cast<uintptr_t>(c) % 64U == 0U)) {
        return false;
    }
    if (LaunchGroupedAugmented()) {
        return true;
    }
    PackAugmented();
    ComputeAugmented();
    return true;
}
