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

// ---- nz implementation ----
namespace cgemm_sb {
using AscendC::ReducePairElem;
using AscendC::ReduceType;

template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbGatherNzBatch(
    const CgemmSbCompactTiling& t, uint32_t cores, uint32_t ownership, uint32_t chunks, TBuf<TPosition::VECCALC>& xb,
    LocalTensor<float> input, GlobalTensor<float> source, uint32_t chunk, uint32_t wholeLength,
    LocalTensor<float> packed)
{
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);
    auto offsets = xb.Get<uint32_t>();
    for (uint32_t pos = 0U; pos < wholeLength; pos += 8192U) {
        const uint32_t n = wholeLength - pos < 8192U ? wholeLength - pos : 8192U;
        SbGather(packed[pos], input, offsets[pos], 0U, n);
    }
    if constexpr (width <= 32U) {
        const uint32_t next =
            ((chunk + 1U) / chunks * cores + GetBlockIdx() / 2U) * ownership + (chunk + 1U) % chunks * groupCap;
        if (next < t.batchCount) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            const uint32_t count = t.batchCount - next < groupCap ? t.batchCount - next : groupCap;
            DataCopyExtParams cp{1U, count * t.m * t.m * 8U, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
            DataCopyPad(input, source[static_cast<uint64_t>(next) * t.m * t.m * 2U], cp, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }
    if (left) {
        const uint64_t bits = t.transB == ACLBLAS_OP_C ?
                                  (trans == ACLBLAS_OP_C ? 0xffaaffaaffaaffaaULL : 0xaa00aa00aa00aa00ULL) :
                                  (trans == ACLBLAS_OP_C ? 0x00aa00aa00aa00aaULL : 0x5500550055005500ULL);
        uint64_t mask[2] = {bits, 0U};
        for (uint32_t pos = 0U; pos < wholeLength; pos += 8192U) {
            const uint32_t n = wholeLength - pos < 8192U ? wholeLength - pos : 8192U;
            Muls(packed[pos], packed[pos], -1.0f, mask, static_cast<uint8_t>(n / 64U), {1, 1, 8, 8});
        }
    }
}

template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbPackNzBatch(
    const CgemmSbCompactTiling& t, uint32_t cores, uint32_t ownership, uint32_t chunks, TBuf<TPosition::VECCALC>& xb,
    LocalTensor<float> input, LocalTensor<float> output, uint32_t length, GlobalTensor<float> source,
    GlobalTensor<float> ws, uint32_t rounds, uint32_t chunk, uint32_t first)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t maxLength = 4U * area;
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);
    const uint32_t nb = t.batchCount - first < groupCap ? t.batchCount - first : groupCap;
    const uint32_t wholeLength = nb * length;
    if (rounds != 0U && rounds % 4U == 0U) {
        CrossCoreWaitFlag<2>(9U);
    }
    if constexpr (width > 32U) {
        DataCopyExtParams cp{1U, nb * t.m * t.m * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        DataCopyPad(input, source[static_cast<uint64_t>(first) * t.m * t.m * 2U], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    } else {
        if (rounds % 2U == 0U) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        } else {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
    }
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto packed = output[width <= 32U ? rounds % 2U * groupCap * maxLength : 0U];
    SbGatherNzBatch<width, left, trans>(t, cores, ownership, chunks, xb, input, source, chunk, wholeLength, packed);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams outCopy{
        static_cast<uint16_t>(nb), length * 4U, 0U,
        static_cast<uint32_t>((SbCompactWorkspaceStride(t.m) - length) * 4U), 0U};
    DataCopyPad(ws[first * SbCompactWorkspaceStride(t.m) + (left ? 0U : 4U * area)], packed, outCopy);
}

template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbRunNzBatches(
    const CgemmSbCompactTiling& t, uint32_t cores, uint32_t ownership, uint32_t chunks, TBuf<TPosition::VECCALC>& xb,
    LocalTensor<float> input, LocalTensor<float> output, uint32_t length, GlobalTensor<float> source,
    GlobalTensor<float> ws)
{
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);
    uint32_t rounds = 0U;
    for (uint32_t chunk = 0U;; ++chunk) {
        // Keep adjacent cache lines on one Cube core, while packing small chunks.
        const uint32_t first = (chunk / chunks * cores + GetBlockIdx() / 2U) * ownership + chunk % chunks * groupCap;
        if (first >= t.batchCount) {
            break;
        }
        SbPackNzBatch<width, left, trans>(
            t, cores, ownership, chunks, xb, input, output, length, source, ws, rounds, chunk, first);
        if constexpr (width <= 32U) {
            if (rounds % 2U == 0U) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            } else {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
        } else {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(8U);
        ++rounds;
    }
    if constexpr (width <= 32U) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }
    if (rounds != 0U) {
        CrossCoreWaitFlag<2>(9U);
    }
}

template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbPrepareNzOutput(
    const CgemmSbCompactTiling& t, GM_ADDR c, uint32_t cores, uint32_t ownership, bool atomicOutput,
    LocalTensor<float> input, LocalTensor<float> output, GlobalTensor<float> source, GlobalTensor<float>& cg)
{
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    if (atomicOutput) {
        SbFill(output, 0.0f, 4096U);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const uint64_t total = static_cast<uint64_t>(t.batchCount) * t.m * t.m * 2U;
        for (uint64_t pos = GetBlockIdx() * 4096U; pos < total; pos += cores * 8192U) {
            const uint32_t n = total - pos < 4096U ? total - pos : 4096U;
            DataCopyExtParams z{1U, n * 4U, 0U, 0U, 0U};
            DataCopyPad(cg[pos], output, z);
        }
        CrossCoreSetFlag<0, PIPE_MTE3>(10U);
        CrossCoreWaitFlag<0>(10U);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    if constexpr (width <= 32U) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        const uint32_t first = GetBlockIdx() / 2U * ownership;
        if (first < t.batchCount) {
            const uint32_t nb = t.batchCount - first < groupCap ? t.batchCount - first : groupCap;
            DataCopyExtParams cp{1U, nb * t.m * t.m * 8U, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
            DataCopyPad(input, source[static_cast<uint64_t>(first) * t.m * t.m * 2U], cp, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }
}

template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbPadNzIndices(
    const CgemmSbCompactTiling& t, TBuf<TPosition::VECCALC>& ob, LocalTensor<int32_t> ix, uint32_t rows,
    uint32_t length, uint32_t zeroIndex)
{
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);
    for (uint32_t batches = 1U; batches < groupCap; batches *= 2U) {
        SbIndexAdd(ix[batches * length], ix, batches * t.m * t.m * 8U, batches * length);
    }
    // Give padded lanes separate zero addresses to avoid Gather bank conflicts.
    auto zeros = ob.Get<int32_t>();
    SbIndex(zeros, 0, 8U);
    for (uint32_t span = 8U; span < 64U; span *= 2U) {
        SbIndexAdd(zeros[span], zeros, span, span);
    }
    SbIndexMul(zeros, zeros, 4, 64U);
    SbIndexAdd(zeros, zeros, static_cast<int32_t>(zeroIndex * 4U), 64U);
    const uint32_t activeRows = left ? 2U * t.m : t.m;
    for (uint32_t pos = activeRows * 8U; pos < rows * 8U; pos += 64U) {
        const uint32_t n = rows * 8U - pos < 64U ? rows * 8U - pos : 64U;
        Adds(
            ix[pos], zeros, 0, static_cast<uint64_t>(n), static_cast<uint8_t>(groupCap * width / 4U),
            {1, 1, static_cast<uint8_t>(rows), 0});
    }
    for (uint32_t slot = 0U; slot < groupCap; ++slot) {
        const uint32_t validBlocks = t.m / 4U;
        if (t.m % 4U != 0U) {
            uint64_t mask[2] = {0x0101010101010101ULL * (0xffULL & (~0ULL << (t.m % 4U * 2U))), 0U};
            Adds(
                ix[slot * length + validBlocks * rows * 8U], zeros, 0, mask, static_cast<uint8_t>(rows / 8U),
                {1, 1, 8, 0});
        }
        for (uint32_t block = (t.m + 3U) / 4U; block < width / 4U; ++block) {
            auto dst = ix[slot * length + block * rows * 8U];
            Adds(dst, zeros, 0, 64ULL, static_cast<uint8_t>(rows / 8U), {1, 1, 8, 0});
        }
    }
}

template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbRunNzPack(
    const CgemmSbCompactTiling& t, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, uint32_t cores,
    uint32_t ownership, uint32_t chunks, bool atomicOutput, TPipe& pipe, TBuf<TPosition::VECCALC>& ib,
    TBuf<TPosition::VECCALC>& ob, TBuf<TPosition::VECCALC>& xb)
{
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;
    constexpr uint32_t maxLength = 4U * area;
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);
    pipe.InitBuffer(xb, groupCap * maxLength * 4U);
    auto input = ib.Get<float>();
    auto output = ob.Get<float>();
    auto ix = xb.Get<int32_t>();
    const uint32_t rows = left ? 2U * width : cubeM;
    const uint32_t length = rows * 2U * width;
    const uint32_t sourcePitch = 2U * t.m;
    const uint32_t zeroIndex = (groupCap * 2U * t.m * t.m + 7U) / 8U * 8U;
    SbNzSeed<left, true>(ix, trans, sourcePitch);
    SbExpandNzMatrix<width, left, trans>(ix, rows, sourcePitch);
    SbPadNzIndices<width, left, trans>(t, ob, ix, rows, length, zeroIndex);
    SbFill(input[zeroIndex], 0.0f, 64U);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> source, ws, cg;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(left ? a : b));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    SbPrepareNzOutput<width, left, trans>(t, c, cores, ownership, atomicOutput, input, output, source, cg);
    SbRunNzBatches<width, left, trans>(t, cores, ownership, chunks, xb, input, output, length, source, ws);
}

template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbNzPack(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling& t, uint32_t cores)
{
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t pitch = 2U * width + 8U;

    constexpr uint32_t area = width * width;
    constexpr uint32_t maxLength = 4U * area;
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);

    const uint32_t balanced = (t.batchCount + cores * 2U - 1U) / (cores * 2U);
    const uint32_t ownership = t.m % 2U != 0U && t.batchCount >= cores * 4U ? 8U : groupCap;
    const uint32_t chunks = ownership / groupCap;
    const bool atomicOutput = (ownership * t.m * t.m) % 8U != 0U;

    TPipe pipe;
    TBuf<TPosition::VECCALC> ib, ob, xb;
    pipe.InitBuffer(ib, (groupCap * 2U * area + 64U) * 4U);
    pipe.InitBuffer(ob, (groupCap * maxLength > 4096U ? groupCap * maxLength : 4096U) * (width <= 32U ? 8U : 4U));
    SbRunNzPack<width, left, trans>(t, a, b, c, workspace, cores, ownership, chunks, atomicOutput, pipe, ib, ob, xb);
}

template <uint32_t width>
__aicore__ inline void SbComputeNzGroup(
    GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling& t, uint32_t first, uint32_t nb)
{
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;
    constexpr uint32_t cubeGroup = width == 24U ? 4U : (width <= 32U ? 2U : 1U);
    for (uint32_t slot = 0U; slot < nb; slot += cubeGroup) {
        const uint32_t part = nb - slot < cubeGroup ? nb - slot : cubeGroup;
        auto* ws = reinterpret_cast<__gm__ float*>(workspace) + (first + slot) * SbCompactWorkspaceStride(t.m);
        CgemmSbAugmentedMmad<2U * width, cubeM * cubeGroup, 2U * width * cubeGroup> mm;
        mm.Init(
            ws + 4U * area, ws,
            reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(first + slot) * t.m * t.m * 2U,
            cubeGroup == 1U ? t.m : part * cubeM, cubeGroup == 1U ? 2U * t.m : part * 2U * width,
            (2U * t.m + 15U) / 16U * 16U, 2U * width, 2U * t.m, 2U * width);
        if constexpr (cubeGroup == 1U) {
            mm.PreloadNz(1U, static_cast<uint32_t>(SbCompactWorkspaceStride(t.m)));
            mm.Process(0U, 1U, true, true);
        } else {
            mm.PreloadDiagonalNz(part, static_cast<uint32_t>(SbCompactWorkspaceStride(t.m)), cubeM, 2U * width);
            mm.Process(0U, 1U, true, true, 0U, t.m, part, cubeM, 2U * width);
        }
    }
}

template <uint32_t width>
__aicore__ inline void SbComputeNzBatch(
    GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling& t, uint32_t cores, uint32_t ownership, uint32_t chunks,
    uint32_t& rounds, uint32_t chunk, uint32_t first)
{
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);
    const uint32_t nb = t.batchCount - first < groupCap ? t.batchCount - first : groupCap;
    CrossCoreWaitFlag<2>(8U);
    if constexpr (width <= 32U) {
        // Preload the group once, then compute only its required products.
        // Alternating accumulators overlap the previous result's writeback.
        auto* ws = reinterpret_cast<__gm__ float*>(workspace) + first * SbCompactWorkspaceStride(t.m);
        CgemmSbAugmentedMmad<2U * width, cubeM, 2U * width, false, true> mm;
        mm.Init(
            ws + 4U * area, ws, reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(first) * t.m * t.m * 2U, t.m,
            2U * t.m, (2U * t.m + 15U) / 16U * 16U, 2U * width, 2U * t.m, 2U * width);
        mm.PreloadNz(nb, static_cast<uint32_t>(SbCompactWorkspaceStride(t.m)));
        for (uint32_t slot = 0U; slot < nb; ++slot) {
            mm.Init(
                ws + 4U * area, ws,
                reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(first + slot) * t.m * t.m * 2U, t.m,
                2U * t.m, (2U * t.m + 15U) / 16U * 16U, 2U * width, 2U * t.m, 2U * width);
            mm.Process(0U, 1U, true, true, slot, 0U, 0U, 0U, 0U, slot % 2U);
        }
    } else {
        SbComputeNzGroup<width>(c, workspace, t, first, nb);
    }
    ++rounds;
    if (rounds % 4U == 0U ||
        ((chunk + 1U) / chunks * cores + GetBlockIdx()) * ownership + (chunk + 1U) % chunks * groupCap >=
            t.batchCount) {
        CrossCoreSetFlag<2, PIPE_FIX>(9U);
    }
}

template <uint32_t width>
__aicore__ inline void SbRunNzCube(
    GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling& t, uint32_t cores, uint32_t ownership, uint32_t chunks,
    bool atomicOutput)
{
    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);
    InitSocState();
    SetHF32Mode(HF32Mode::DISABLE);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    if constexpr (width <= 32U) {
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
    if (atomicOutput) {
        SetAtomicAdd<float>();
    }
    uint32_t rounds = 0U;
    for (uint32_t chunk = 0U;; ++chunk) {
        // Keep adjacent cache lines on one Cube core, while packing small chunks.
        const uint32_t first = (chunk / chunks * cores + GetBlockIdx()) * ownership + chunk % chunks * groupCap;
        if (first >= t.batchCount) {
            break;
        }
        SbComputeNzBatch<width>(c, workspace, t, cores, ownership, chunks, rounds, chunk, first);
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    if constexpr (width <= 32U) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
    if (atomicOutput) {
        SetAtomicNone();
    }
}

template <uint32_t width>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_nz_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t pitch = 2U * width + 8U;

    constexpr uint32_t area = width * width;

    constexpr uint32_t groupCap = width == 24U ? 4U : (width <= 40U ? 2U : 1U);

    const uint32_t balanced = (t.batchCount + cores * 2U - 1U) / (cores * 2U);
    const uint32_t ownership = t.m % 2U != 0U && t.batchCount >= cores * 4U ? 8U : groupCap;
    const uint32_t chunks = ownership / groupCap;
    const bool atomicOutput = (ownership * t.m * t.m) % 8U != 0U;
    if ASCEND_IS_AIV {
        if (GetSubBlockIdx() == 0U) {
            if (t.transA == ACLBLAS_OP_N) {
                SbNzPack<width, true, ACLBLAS_OP_N>(a, b, c, workspace, t, cores);
            } else if (t.transA == ACLBLAS_OP_T) {
                SbNzPack<width, true, ACLBLAS_OP_T>(a, b, c, workspace, t, cores);
            } else {
                SbNzPack<width, true, ACLBLAS_OP_C>(a, b, c, workspace, t, cores);
            }
        } else {
            if (t.transB == ACLBLAS_OP_N) {
                SbNzPack<width, false, ACLBLAS_OP_N>(a, b, c, workspace, t, cores);
            } else if (t.transB == ACLBLAS_OP_T) {
                SbNzPack<width, false, ACLBLAS_OP_T>(a, b, c, workspace, t, cores);
            } else {
                SbNzPack<width, false, ACLBLAS_OP_C>(a, b, c, workspace, t, cores);
            }
        }
    }
    if ASCEND_IS_AIC {
        SbRunNzCube<width>(c, workspace, t, cores, ownership, chunks, atomicOutput);
    }
}
} // namespace cgemm_sb

// ---- nz copy implementation ----
namespace cgemm_sb {
template <
    uint32_t width, bool left, uint32_t trans, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7>
__aicore__ inline void SbPackNzCopyBatch(
    const CgemmSbCompactTiling& t, Arg0& xb, Arg1& input, Arg2& output, Arg3& length, Arg4& source, Arg5& ws,
    Arg6& first, Arg7& nb)
{
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;
    constexpr uint32_t workspaceStride =
        4U * area + cubeM * 2U * width + width * ((2U * width + 15U) / 16U * 16U) + 16U;
    const uint32_t wholeLength = nb * length;

    DataCopyExtParams cp{1U, nb * t.m * t.m * 8U, 0U, 0U, 0U};
    DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    DataCopyPad(input, source[static_cast<uint64_t>(first) * t.m * t.m * 2U], cp, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto offsets = xb.template Get<uint32_t>();
    for (uint32_t pos = 0U; pos < wholeLength; pos += 8192U) {
        const uint32_t n = wholeLength - pos < 8192U ? wholeLength - pos : 8192U;
        SbGather(output[pos], input, offsets[pos], 0U, n);
    }
    if (left) {
        const uint64_t bits = t.transB == ACLBLAS_OP_C ?
                                  (trans == ACLBLAS_OP_C ? 0xffaaffaaffaaffaaULL : 0xaa00aa00aa00aa00ULL) :
                                  (trans == ACLBLAS_OP_C ? 0x00aa00aa00aa00aaULL : 0x5500550055005500ULL);
        uint64_t mask[2] = {bits, 0U};
        for (uint32_t pos = 0U; pos < wholeLength; pos += 8192U) {
            const uint32_t n = wholeLength - pos < 8192U ? wholeLength - pos : 8192U;
            Muls(output[pos], output[pos], -1.0f, mask, static_cast<uint8_t>(n / 64U), {1, 1, 8, 8});
        }
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams outCopy{
        static_cast<uint16_t>(nb), length * 4U, 0U, static_cast<uint32_t>((workspaceStride - length) * 4U), 0U};
    DataCopyPad(ws[first * workspaceStride + (left ? 0U : 4U * area)], output, outCopy);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

template <
    uint32_t width, bool left, uint32_t trans, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10,
    typename Arg11, typename Arg12>
__aicore__ inline void SbWriteNzCopyOutput(
    const CgemmSbCompactTiling& t, Arg0& cores, Arg1& ownership, Arg2& chunks, Arg3& xb, Arg4& input, Arg5& output,
    Arg6& ix, Arg7& ws, Arg8& cg, Arg9& rowBegin, Arg10& lengthOut, Arg11& packedLength, Arg12& nativeSize)
{
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;
    constexpr uint32_t workspaceStride =
        4U * area + cubeM * 2U * width + width * ((2U * width + 15U) / 16U * 16U) + 16U;
    constexpr uint32_t groupCap = 2U;
    constexpr uint32_t outputOffset = 4U * area + cubeM * 2U * width;
    Muls(ix[2U * area], ix[2U * area], 16, packedLength);
    Add(ix, ix, ix[2U * area], packedLength);
    Adds(ix, ix, static_cast<int32_t>(rowBegin * 16U), packedLength);
    Muls(ix, ix, 4, packedLength);
    // Each output block is independently padded to the DMA alignment.
    SbIndexAdd(ix[packedLength], ix, static_cast<int32_t>(nativeSize * 4U), packedLength);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t chunk = 0U;; ++chunk) {
        const uint32_t first = (chunk / chunks * cores + GetBlockIdx() / 2U) * ownership + chunk % chunks * groupCap;
        if (first >= t.batchCount) {
            break;
        }
        const uint32_t nb = t.batchCount - first < groupCap ? t.batchCount - first : groupCap;
        CrossCoreWaitFlag<2>(10U);
        DataCopyParams cpIn{
            static_cast<uint16_t>(nb), static_cast<uint16_t>(nativeSize / 8U),
            static_cast<uint16_t>((workspaceStride - nativeSize) / 8U), 0U};
        DataCopy(input, ws[first * workspaceStride + outputOffset], cpIn);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SbGather(output, input, xb.template Get<uint32_t>(), 0U, nb * packedLength);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cpOut{static_cast<uint16_t>(nb), lengthOut * 4U, 0U, (t.m * t.m * 2U - lengthOut) * 4U, 0U};
        DataCopyPad(cg[(static_cast<uint64_t>(first) * t.m + rowBegin) * t.m * 2U], output, cpOut);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
}

template <
    uint32_t width, bool left, uint32_t trans, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
__aicore__ inline void SbRunNzCopyBatches(
    const CgemmSbCompactTiling& t, Arg0& cores, Arg1& ownership, Arg2& chunks, Arg3& ob, Arg4& xb, Arg5& input,
    Arg6& output, Arg7& ix, Arg8& ws, Arg9& cg)
{
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;
    const uint32_t rowBegin = left ? 0U : (t.m + 1U) / 2U;
    const uint32_t rowsOut = left ? (t.m + 1U) / 2U : t.m / 2U;

    const uint32_t lengthOut = rowsOut * t.m * 2U;
    const uint32_t packedLength = (lengthOut + 7U) / 8U * 8U;
    const uint32_t nativeSize = t.m * ((2U * t.m + 15U) / 16U * 16U);
    auto scratch = ob.template Get<int32_t>();
    SbIndex(ix, 0, 8U);
    for (uint32_t span = 8U; span < packedLength; span *= 2U) {
        const uint32_t part = packedLength - span < span ? packedLength - span : span;
        SbIndexAdd(ix[span], ix, static_cast<int32_t>(span), part);
    }
    const int32_t reciprocal = static_cast<int32_t>((1048576U + 2U * t.m - 1U) / (2U * t.m));
    Muls(ix[2U * area], ix, reciprocal, packedLength);
    ShiftRight(ix[2U * area], ix[2U * area], 20, packedLength);
    Muls(scratch, ix[2U * area], static_cast<int32_t>(2U * t.m), packedLength);
    Sub(ix, ix, scratch, packedLength);
    ShiftRight(scratch, ix, 4, packedLength);
    Muls(scratch, scratch, static_cast<int32_t>(t.m * 16U - 16U), packedLength);
    Add(ix, ix, scratch, packedLength);
    SbWriteNzCopyOutput<width, left, trans>(
        t, cores, ownership, chunks, xb, input, output, ix, ws, cg, rowBegin, lengthOut, packedLength, nativeSize);
}

template <
    uint32_t width, bool left, uint32_t trans, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10,
    typename Arg11, typename Arg12, typename Arg13, typename Arg14, typename Arg15>
__aicore__ inline void SbExpandNzCopyIndices(
    const CgemmSbCompactTiling& t, Arg0& a, Arg1& b, Arg2& c, Arg3& workspace, Arg4& cores, Arg5& ownership,
    Arg6& chunks, Arg7& ob, Arg8& xb, Arg9& input, Arg10& output, Arg11& ix, Arg12& rows, Arg13& length,
    Arg14& sourcePitch, Arg15& zeroIndex)
{
    constexpr uint32_t groupCap = 2U;
    SbExpandNzMatrix<width, left, trans>(ix, rows, sourcePitch);
    SbPadNzIndices<width, left, trans>(t, ob, ix, rows, length, zeroIndex);
    SbFill(input[zeroIndex], 0.0f, 64U);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> source, ws, cg;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(left ? a : b));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));

    uint32_t rounds = 0U;
    for (uint32_t chunk = 0U;; ++chunk) {
        // Keep adjacent cache lines on one Cube core, while packing small chunks.
        const uint32_t first = (chunk / chunks * cores + GetBlockIdx() / 2U) * ownership + chunk % chunks * groupCap;
        if (first >= t.batchCount) {
            break;
        }
        const uint32_t nb = t.batchCount - first < groupCap ? t.batchCount - first : groupCap;
        SbPackNzCopyBatch<width, left, trans>(t, xb, input, output, length, source, ws, first, nb);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CrossCoreSetFlag<2, PIPE_MTE3>(8U);
        ++rounds;
    }
    // All packed inputs have distinct workspace slots. The host bounds the
    // number of groups so the completion counters cannot overflow.
    SbRunNzCopyBatches<width, left, trans>(t, cores, ownership, chunks, ob, xb, input, output, ix, ws, cg);
}

template <uint32_t width, bool left, uint32_t trans>
__aicore__ inline void SbNzCopyPack(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling& t, uint32_t cores)
{
    static_assert(width == 40U, "Native epilogue packs two medium-size matrices per group");
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;
    constexpr uint32_t maxLength = 4U * area;
    // Keep each input/product slot compact and aligned to a complete cache line.

    constexpr uint32_t groupCap = 2U;

    const uint32_t ownership = t.m % 2U != 0U && t.batchCount >= cores * 4U ? 8U : groupCap;
    const uint32_t chunks = ownership / groupCap;

    TPipe pipe;
    TBuf<TPosition::VECCALC> ib, ob, xb;
    pipe.InitBuffer(ib, (groupCap * 2U * area + 64U) * 4U);
    pipe.InitBuffer(ob, (groupCap * maxLength > 4096U ? groupCap * maxLength : 4096U) * 4U);
    pipe.InitBuffer(xb, groupCap * maxLength * 4U);
    auto input = ib.Get<float>();
    auto output = ob.Get<float>();
    auto ix = xb.Get<int32_t>();
    const uint32_t rows = left ? 2U * width : cubeM;
    const uint32_t length = rows * 2U * width;
    const uint32_t sourcePitch = 2U * t.m;
    const uint32_t zeroIndex = (groupCap * 2U * t.m * t.m + 7U) / 8U * 8U;
    SbNzSeed<left, true>(ix, trans, sourcePitch);
    SbExpandNzCopyIndices<width, left, trans>(
        t, a, b, c, workspace, cores, ownership, chunks, ob, xb, input, output, ix, rows, length, sourcePitch,
        zeroIndex);
}

template <uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__aicore__ inline void SbRunNzCopyCube(Arg0& workspace, Arg1& t, Arg2& cores, Arg3& ownership, Arg4& chunks)
{
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;
    constexpr uint32_t workspaceStride =
        4U * area + cubeM * 2U * width + width * ((2U * width + 15U) / 16U * 16U) + 16U;
    constexpr uint32_t groupCap = 2U;
    constexpr uint32_t cubeGroup = 1U;
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    uint32_t rounds = 0U;
    for (uint32_t chunk = 0U;; ++chunk) {
        // Keep adjacent cache lines on one Cube core, while packing small chunks.
        const uint32_t first = (chunk / chunks * cores + GetBlockIdx()) * ownership + chunk % chunks * groupCap;
        if (first >= t.batchCount) {
            break;
        }
        const uint32_t nb = t.batchCount - first < groupCap ? t.batchCount - first : groupCap;
        CrossCoreWaitFlag<2>(8U);
        for (uint32_t slot = 0U; slot < nb; slot += cubeGroup) {
            const uint32_t part = nb - slot < cubeGroup ? nb - slot : cubeGroup;
            auto* ws = reinterpret_cast<__gm__ float*>(workspace) + (first + slot) * workspaceStride;
            CgemmSbAugmentedMmad<2U * width, cubeM * cubeGroup, 2U * width * cubeGroup, true, true> mm;
            mm.Init(
                ws + 4U * area, ws, ws + 4U * area + cubeM * 2U * width, cubeGroup == 1U ? t.m : part * cubeM,
                cubeGroup == 1U ? 2U * t.m : part * 2U * width, (2U * t.m + 15U) / 16U * 16U, 2U * width, 2U * t.m,
                2U * width);
            if constexpr (cubeGroup == 1U) {
                mm.PreloadNz(1U, static_cast<uint32_t>(workspaceStride));
                mm.Process(0U, 1U, true, true, 0U, 0U, 0U, 0U, 0U, (rounds * groupCap + slot) % 2U);
            } else {
                mm.PreloadDiagonalNz(part, static_cast<uint32_t>(workspaceStride), cubeM, 2U * width);
                mm.Process(0U, 1U, true, true, 0U, t.m, part, cubeM, 2U * width);
            }
        }
        ++rounds;
        CrossCoreSetFlag<2, PIPE_FIX>(10U);
    }
}

template <uint32_t width>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_nz_copy_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    static_assert(width == 40U, "Native epilogue packs two medium-size matrices per group");
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t area = width * width;

    // Keep each input/product slot compact and aligned to a complete cache line.

    constexpr uint32_t groupCap = 2U;

    const uint32_t ownership = t.m % 2U != 0U && t.batchCount >= cores * 4U ? 8U : groupCap;
    const uint32_t chunks = ownership / groupCap;
    if ASCEND_IS_AIV {
        if (GetSubBlockIdx() == 0U) {
            if (t.transA == ACLBLAS_OP_N) {
                SbNzCopyPack<width, true, ACLBLAS_OP_N>(a, b, c, workspace, t, cores);
            } else if (t.transA == ACLBLAS_OP_T) {
                SbNzCopyPack<width, true, ACLBLAS_OP_T>(a, b, c, workspace, t, cores);
            } else {
                SbNzCopyPack<width, true, ACLBLAS_OP_C>(a, b, c, workspace, t, cores);
            }
        } else {
            if (t.transB == ACLBLAS_OP_N) {
                SbNzCopyPack<width, false, ACLBLAS_OP_N>(a, b, c, workspace, t, cores);
            } else if (t.transB == ACLBLAS_OP_T) {
                SbNzCopyPack<width, false, ACLBLAS_OP_T>(a, b, c, workspace, t, cores);
            } else {
                SbNzCopyPack<width, false, ACLBLAS_OP_C>(a, b, c, workspace, t, cores);
            }
        }
    }
    if ASCEND_IS_AIC {
        InitSocState();
        SetHF32Mode(HF32Mode::DISABLE);
        SbRunNzCopyCube<width>(workspace, t, cores, ownership, chunks);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
}
} // namespace cgemm_sb

// ---- direct nt implementation ----
namespace cgemm_sb {
// Compute all four real component products from unmodified complex inputs.
// P = transpose(B_as_real) * A_as_real; the vector epilogue combines each 2x2 block.
template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11>
__aicore__ inline void SbAccumulateDirectNt(
    Arg0& workspaceStride, Arg1& wg, Arg2& al0, Arg3& bl0, Arg4& cl0, Arg5& mp, Arg6& fp, Arg7& rounds, Arg8& slot,
    Arg9& batch, Arg10& part, Arg11& dim)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t ownership = 2U;
    constexpr uint32_t group = 1U;
    constexpr uint32_t productSize = group * group * 4U * area;
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    const bool firstBuffer = (rounds * (ownership / group) + slot / group) % 2U == 0U;
    auto product = cl0[firstBuffer ? 0U : productSize];
    if (firstBuffer) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    } else {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
    Mmad(product, al0, bl0, mp);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    for (uint32_t item = 0U; item < part; ++item) {
        if constexpr (width <= 32U) {
            Fixpipe<float, float, AscendC::CFG_NZ>(wg[(batch + item) * workspaceStride], product, fp);
        } else {
            Fixpipe(
                wg[(batch + item) * workspaceStride], product[item * 2U * width * dim + item * 2U * width * 16U], fp);
        }
        CrossCoreSetFlag<2, PIPE_FIX>(8U);
    }
    if (firstBuffer) {
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8>
__aicore__ inline void SbPackDirectNtOperands(
    Arg0& t, Arg1& input, Arg2& output, Arg3& temp, Arg4& offsets, Arg5& length, Arg6& pairLength, Arg7& cg,
    Arg8& batch)
{
    constexpr uint32_t area = width * width;
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    if constexpr (width <= 32U) {
        SbGather(output, input, offsets, 0U, pairLength);
        uint64_t sign[2] = {t.transB == ACLBLAS_OP_C ? 0x8888888888888888ULL : 0x2222222222222222ULL, 0U};
        Muls(output, output, -1.0f, sign, static_cast<uint8_t>((pairLength + 63U) / 64U), {1, 1, 8, 8});
        const uint32_t full = pairLength / 64U;
        ReducePairElem<ReduceType::SUM>(temp, output, 64, static_cast<int32_t>(full), 1, 1, 8);
        if (pairLength % 64U != 0U) {
            ReducePairElem<ReduceType::SUM>(
                temp[full * 32U], output[full * 64U], static_cast<int32_t>(pairLength % 64U), 1, 1, 1, 8);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1U, length * 4U, 0U, 0U, 0U};
        DataCopyPad(cg[static_cast<uint64_t>(batch) * length], temp, cp);
    } else {
        SbGather(output, input, offsets, 0U, length);
        SbGather(temp, input, offsets[2U * area], 0U, length);
        uint64_t even[2] = {t.transB == ACLBLAS_OP_C ? 0xaaaaaaaaaaaaaaaaULL : 0x5555555555555555ULL, 0U};
        Muls(temp, temp, -1.0f, even, static_cast<uint8_t>((length + 63U) / 64U), {1, 1, 8, 8});
        Add(output, output, temp, length);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1U, length * 4U, 0U, 0U, 0U};
        DataCopyPad(cg[static_cast<uint64_t>(batch) * length], output, cp);
    }
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11>
__aicore__ inline void SbRunDirectNtBatches(
    Arg0& t, Arg1& cores, Arg2& mn, Arg3& workspaceStride, Arg4& input, Arg5& output, Arg6& temp, Arg7& offsets,
    Arg8& length, Arg9& pairLength, Arg10& wg, Arg11& cg)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t ownership = 2U;
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    uint32_t rounds = 0U;
    const uint32_t core = GetBlockIdx() / 2U;
    const uint32_t begin =
        width <= 32U ? core * (t.batchCount / cores) + (core < t.batchCount % cores ? core : t.batchCount % cores) :
                       core * ownership;
    const uint32_t end =
        width <= 32U ? begin + t.batchCount / cores + (core < t.batchCount % cores ? 1U : 0U) : t.batchCount;
    const uint32_t step = width <= 32U ? ownership : cores * ownership;
    for (uint32_t first = begin; first < end; first += step) {
        const uint32_t count = end - first < ownership ? end - first : ownership;
        for (uint32_t slot = 0U; slot < count; ++slot) {
            const uint32_t relative = slot;
            const uint32_t batch = first + relative;
            CrossCoreWaitFlag<2>(8U);
            if (relative == GetSubBlockIdx()) {
                DataCopy(input, wg[batch * workspaceStride], width <= 32U ? mn * mn : 4U * area);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SbPackDirectNtOperands<width>(t, input, output, temp, offsets, length, pairLength, cg, batch);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
        }
        if (++rounds % 4U == 0U || first + step >= end) {
            CrossCoreSetFlag<2, PIPE_MTE3>(9U);
        }
    }
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12,
    typename Arg13, typename Arg14, typename Arg15, typename Arg16, typename Arg17, typename Arg18>
__aicore__ inline void SbLoadDirectNtTile(
    Arg0& t, Arg1& mn, Arg2& workspaceStride, Arg3& ag, Arg4& bg, Arg5& wg, Arg6& al1, Arg7& bl1, Arg8& al0, Arg9& bl0,
    Arg10& cl0, Arg11& cp, Arg12& lp, Arg13& mp, Arg14& fp, Arg15& rounds, Arg16& slot, Arg17& batch, Arg18& part)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t group = 1U;
    const uint32_t dim = group == 1U ? mn : part * 2U * width;
    cp.ndNum = static_cast<uint16_t>(part);
    cp.srcNdMatrixStride = t.m * t.m * 2U;
    cp.dstNzMatrixStride = 2U * area;
    lp.channelSize = group == 1U ? 2U * width : dim;
    lp.kExtension = dim;
    mp.m = dim;
    mp.n = dim;
    fp.srcStride = dim;
    const uint64_t offset = static_cast<uint64_t>(batch) * t.m * t.m * 2U;
    if constexpr (width > 32U) {
        DataCopy(al1, bg[offset], cp);
        DataCopy(bl1, ag[offset], cp);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }
    const uint32_t inputOffset = width <= 32U ? slot * 2U * area : 0U;
    LoadData(al0, al1[inputOffset], lp);
    LoadData(bl0, bl1[inputOffset], lp);
    SbAccumulateDirectNt<width>(workspaceStride, wg, al0, bl0, cl0, mp, fp, rounds, slot, batch, part, dim);
}

template <uint32_t width, typename Arg0, typename Arg1, typename Arg2>
__aicore__ inline void SbBuildDirectNtCopyIndices(Arg0& t, Arg1& ix, Arg2& length)
{
    constexpr uint32_t area = width * width;
    const int32_t reciprocal = static_cast<int32_t>((1048576U + 2U * t.m - 1U) / (2U * t.m));
    SbIndex(ix, 0, 8U);
    for (uint32_t span = 8U; span < length; span *= 2U) {
        const uint32_t part = length - span < span ? length - span : span;
        SbIndexAdd(ix[span], ix, static_cast<int32_t>(span), part);
    }
    Muls(ix[2U * area], ix, reciprocal, length);
    ShiftRight(ix[2U * area], ix[2U * area], 20, length);
    Muls(ix[2U * area], ix[2U * area], static_cast<int32_t>(4U * width - 2U * t.m), length);
    Add(ix, ix, ix[2U * area], length);
    Muls(ix, ix, 4, length);
    SbIndexAdd(ix[2U * area], ix, 2U * width * 4U, length);
    uint64_t even[2] = {0x5555555555555555ULL, 0U}, odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
    const uint32_t rounded = (length + 63U) / 64U;
    Adds(ix[2U * area], ix[2U * area], 4, even, static_cast<uint8_t>(rounded), {1, 1, 8, 8});
    Adds(ix[2U * area], ix[2U * area], -4, odd, static_cast<uint8_t>(rounded), {1, 1, 8, 8});
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6>
__aicore__ inline void SbBuildDirectNtGatherIndices(
    Arg0& t, Arg1& mn, Arg2& tb, Arg3& ix, Arg4& pairColumn, Arg5& prefix, Arg6& repeats)
{
    constexpr uint32_t area = width * width;
    for (uint32_t span = 8U; span < prefix; span *= 2U) {
        const uint32_t part = prefix - span < span ? prefix - span : span;
        SbIndexAdd(ix[span], ix, static_cast<int32_t>(span), part);
    }
    auto rows = ix[2U * area];
    auto scratch = tb.template Get<int32_t>();
    Muls(rows, ix, static_cast<int32_t>((1048576U + pairColumn - 1U) / pairColumn), 64ULL, repeats, {1, 1, 8, 8});
    ShiftRight(rows, rows, 20, 64ULL, repeats, {1, 1, 8, 8});
    ShiftRight(ix, ix, 1, 64ULL, repeats, {1, 1, 8, 8});
    Muls(scratch, rows, static_cast<int32_t>(2U * t.m), 64ULL, repeats, {1, 1, 8, 8});
    Sub(ix, ix, scratch, 64ULL, repeats, {1, 1, 1, 8, 8, 8});
    uint64_t plus[2] = {0x2222222222222222ULL, 0U}, minus[2] = {0x8888888888888888ULL, 0U};
    Adds(ix, ix, 1, plus, repeats, {1, 1, 8, 8});
    Adds(ix, ix, -1, minus, repeats, {1, 1, 8, 8});
    ShiftRight(scratch, ix, 4, 64ULL, repeats, {1, 1, 8, 8});
    Muls(scratch, scratch, static_cast<int32_t>(mn * 16U - 16U), 64ULL, repeats, {1, 1, 8, 8});
    Add(ix, ix, scratch, 64ULL, repeats, {1, 1, 1, 8, 8, 8});
    Muls(rows, rows, 32, 64ULL, repeats, {1, 1, 8, 8});
    Add(ix, ix, rows, 64ULL, repeats, {1, 1, 1, 8, 8, 8});
    uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
    Adds(ix, ix, 16, odd, repeats, {1, 1, 8, 8});
    Muls(ix, ix, 4, 64ULL, repeats, {1, 1, 8, 8});
    for (uint32_t columns = 2U; columns < t.m; columns *= 2U) {
        const uint32_t part = t.m - columns < columns ? t.m - columns : columns;
        SbIndexAdd(ix[columns * pairColumn], ix, static_cast<int32_t>(columns * 128U), part * pairColumn);
    }
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12,
    typename Arg13, typename Arg14, typename Arg15, typename Arg16>
__aicore__ inline void SbRunDirectNtCubeBatches(
    Arg0& t, Arg1& cores, Arg2& mn, Arg3& workspaceStride, Arg4& ag, Arg5& bg, Arg6& wg, Arg7& al1, Arg8& bl1,
    Arg9& al0, Arg10& bl0, Arg11& cl0, Arg12& cp, Arg13& lp, Arg14& mp, Arg15& fp, Arg16& rounds)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t ownership = 2U;
    constexpr uint32_t group = 1U;
    const uint32_t core = GetBlockIdx();
    const uint32_t begin =
        width <= 32U ? core * (t.batchCount / cores) + (core < t.batchCount % cores ? core : t.batchCount % cores) :
                       core * ownership;
    const uint32_t end =
        width <= 32U ? begin + t.batchCount / cores + (core < t.batchCount % cores ? 1U : 0U) : t.batchCount;
    const uint32_t step = width <= 32U ? ownership : cores * ownership;
    for (uint32_t first = begin; first < end; first += step) {
        const uint32_t count = end - first < ownership ? end - first : ownership;
        if constexpr (width <= 32U) {
            cp.ndNum = static_cast<uint16_t>(count);
            cp.srcNdMatrixStride = t.m * t.m * 2U;
            cp.dstNzMatrixStride = 2U * area;
            const uint64_t initialOffset = static_cast<uint64_t>(first) * t.m * t.m * 2U;
            DataCopy(al1, bg[initialOffset], cp);
            DataCopy(bl1, ag[initialOffset], cp);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        }
        for (uint32_t slot = 0U; slot < count; slot += group) {
            const uint32_t relative = slot;
            const uint32_t batch = first + relative;
            const uint32_t part = count - slot < group ? count - slot : group;
            SbLoadDirectNtTile<width>(
                t, mn, workspaceStride, ag, bg, wg, al1, bl1, al0, bl0, cl0, cp, lp, mp, fp, rounds, slot, batch, part);
        }
        if (++rounds % 4U == 0U || first + step >= end) {
            CrossCoreWaitFlag<2>(9U);
        }
    }
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6>
__aicore__ inline void SbRunDirectNtVector(
    Arg0& c, Arg1& workspace, Arg2& t, Arg3& cores, Arg4& mn, Arg5& workspaceStride, Arg6& pipe)
{
    constexpr uint32_t area = width * width;
    TBuf<TPosition::VECCALC> ib, ob, tb, xb;
    pipe.InitBuffer(ib, 4U * area * 4U);
    pipe.InitBuffer(ob, (width <= 32U ? 4U : 2U) * area * 4U);
    pipe.InitBuffer(tb, 2U * area * 4U);
    pipe.InitBuffer(xb, 4U * area * 4U);
    auto input = ib.template Get<float>();
    auto output = ob.template Get<float>();
    auto temp = tb.template Get<float>();
    auto ix = xb.template Get<int32_t>();
    auto offsets = xb.template Get<uint32_t>();
    const uint32_t length = 2U * t.m * t.m;
    const uint32_t pairColumn = 4U * t.m;
    const uint32_t pairLength = pairColumn * t.m;
    if constexpr (width <= 32U) {
        // Build two logical columns, then replicate their native NZ addresses.
        // Two columns keep every replication start aligned even for odd m.
        const uint32_t prefix = 2U * pairColumn;
        const uint8_t repeats = static_cast<uint8_t>((prefix + 63U) / 64U);
        SbIndex(ix, 0, 8U);
        SbBuildDirectNtGatherIndices<width>(t, mn, tb, ix, pairColumn, prefix, repeats);
    } else {
        SbBuildDirectNtCopyIndices<width>(t, ix, length);
    }
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> wg, cg;
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    SbRunDirectNtBatches<width>(
        t, cores, mn, workspaceStride, input, output, temp, offsets, length, pairLength, wg, cg);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12,
    typename Arg13>
__aicore__ inline void SbPrepareDirectNtMmad(
    Arg0& t, Arg1& cores, Arg2& mn, Arg3& workspaceStride, Arg4& ag, Arg5& bg, Arg6& wg, Arg7& al1, Arg8& bl1,
    Arg9& al0, Arg10& bl0, Arg11& cl0, Arg12& cp, Arg13& lp)
{
    lp.filterH = 1U;
    lp.dilationFilterW = 1U;
    lp.dilationFilterH = 1U;
    lp.enTranspose = true;
    lp.filterSizeW = false;
    lp.filterSizeH = false;
    lp.fMatrixCtrl = false;
    MmadParams mp = {};
    mp.m = mn;
    mp.n = mn;
    mp.k = t.m;
    mp.cmatrixInitVal = true;
    mp.kDirectionAlign = true;
    FixpipeParamsV220 fp = {};
    fp.mSize = 2U * t.m;
    fp.nSize = 2U * t.m;
    fp.srcStride = mn;
    fp.dstStride = 2U * width;
    fp.ndNum = 1U;
    if constexpr (width <= 32U) {
        fp.mSize = mn;
        fp.nSize = mn;
        fp.dstStride = mn * 2U;
    }
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    uint32_t rounds = 0U;
    SbRunDirectNtCubeBatches<width>(
        t, cores, mn, workspaceStride, ag, bg, wg, al1, bl1, al0, bl0, cl0, cp, lp, mp, fp, rounds);
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__aicore__ inline void SbRunDirectNtCube(
    Arg0& a, Arg1& b, Arg2& workspace, Arg3& t, Arg4& cores, Arg5& mk, Arg6& mn, Arg7& workspaceStride, Arg8& ag,
    Arg9& bg, Arg10& wg)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t ownership = 2U;
    constexpr uint32_t group = 1U;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    wg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    LocalTensor<float> al1(TPosition::A1, 0U, ownership * 2U * area);
    LocalTensor<float> bl1(TPosition::B1, ownership * 2U * area * 4U, ownership * 2U * area);
    LocalTensor<float> al0(TPosition::A2, 0U, group * 2U * area);
    LocalTensor<float> bl0(TPosition::B2, 0U, group * 2U * area);
    constexpr uint32_t productSize = group * group * 4U * area;
    LocalTensor<float> cl0(TPosition::CO1, 0U, 2U * productSize);
    Nd2NzParams cp{
        1U,
        static_cast<uint16_t>(t.m),
        static_cast<uint16_t>(2U * t.m),
        0U,
        static_cast<uint16_t>(2U * t.m),
        static_cast<uint16_t>(width),
        1U,
        0U};
    LoadData3DParamsV2<float> lp;
    lp.l1H = 1U;
    lp.l1W = width;
    lp.channelSize = 2U * width;
    lp.kExtension = mn;
    lp.mExtension = mk;
    lp.strideW = 1U;
    lp.strideH = 1U;
    lp.filterW = 1U;
    SbPrepareDirectNtMmad<width>(t, cores, mn, workspaceStride, ag, bg, wg, al1, bl1, al0, bl0, cl0, cp, lp);
}

template <uint32_t width>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_direct_nt_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    const uint32_t mk = (t.m + 15U) / 16U * 16U;
    const uint32_t mn = (2U * t.m + 15U) / 16U * 16U;
    const uint64_t workspaceStride = width <= 32U ? mn * mn + 16U : SbCompactWorkspaceStride(t.m);
    if ASCEND_IS_AIC {
        InitSocState();
        SetHF32Mode(HF32Mode::DISABLE);
        GlobalTensor<float> ag, bg, wg;
        SbRunDirectNtCube<width>(a, b, workspace, t, cores, mk, mn, workspaceStride, ag, bg, wg);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        SbRunDirectNtVector<width>(c, workspace, t, cores, mn, workspaceStride, pipe);
    }
}
} // namespace cgemm_sb

void cgemm_sb::CgemmSbDispatcher::LaunchNz(uint32_t width)
{
    switch (width) {
        case 24U:
            cgemm_sb_nz_kernel<24U><<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
            return;
        case 32U:
            cgemm_sb_nz_kernel<32U><<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
            return;
        case 40U:
            cgemm_sb_nz_kernel<40U><<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
            return;
        case 48U:
            cgemm_sb_nz_kernel<48U><<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
            return;
        default:
            return;
    }
}

void cgemm_sb::CgemmSbDispatcher::LaunchNzCopy()
{
    cgemm_sb_nz_copy_kernel<40U><<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
}

void cgemm_sb::CgemmSbDispatcher::LaunchDirectNt(uint32_t width)
{
    if (width == 32U) {
        cgemm_sb_direct_nt_kernel<32U><<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else {
        cgemm_sb_direct_nt_kernel<64U><<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    }
}
