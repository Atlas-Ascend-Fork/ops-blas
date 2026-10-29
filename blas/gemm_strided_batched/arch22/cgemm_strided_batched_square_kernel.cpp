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

// ---- square implementation ----
namespace cgemm_sb {
// Paired vector cores pack complex tiles while Cube evaluates an augmented real GEMM.
// Atomic output protects partial cache lines at unaligned batch boundaries.
template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7>
__aicore__ inline void SbRunSquareCube(
    Arg0& c, Arg1& workspace, Arg2& t, Arg3& cores, Arg4& padded, Arg5& tiles, Arg6& area, Arg7& ownership)
{
    uint32_t rounds = 0U;
    for (uint32_t group = GetBlockIdx(); group * ownership < t.batchCount; group += cores) {
        const uint32_t first = group * ownership;
        const uint32_t nb = t.batchCount - first < ownership ? t.batchCount - first : ownership;
        CrossCoreWaitFlag<2>(8U);
        CgemmSbAugmentedMmad<2U * cubeWidth, (cubeWidth + 15U) / 16U * 16U, 2U * cubeWidth> mm;
        auto* initial = reinterpret_cast<__gm__ float*>(workspace) + first * SbCompactWorkspaceStride(t.m);
        mm.Init(
            initial + 4U * area, initial, initial + 6U * area, t.m, 2U * t.m, (2U * t.m + 15U) / 16U * 16U, 2U * padded,
            2U * padded, 2U * padded);
        if (tiles == 1U) {
            mm.Preload(
                nb, static_cast<uint32_t>(SbCompactWorkspaceStride(t.m)),
                static_cast<uint32_t>(SbCompactWorkspaceStride(t.m)));
        }
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t batch = first + i;
            auto* ws = reinterpret_cast<__gm__ float*>(workspace) + batch * SbCompactWorkspaceStride(t.m);
            mm.Init(
                ws + 4U * area, ws, reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(batch) * t.m * t.m * 2U,
                t.m, 2U * t.m, (2U * t.m + 15U) / 16U * 16U, 2U * padded, 2U * t.m, 2U * padded);
            mm.Process(0U, 1U, true, tiles == 1U, i);
        }
        ++rounds;
        if (rounds % 4U == 0U || (group + cores) * ownership >= t.batchCount) {
            CrossCoreSetFlag<2, PIPE_FIX>(9U);
        }
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}

template <uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__aicore__ inline void SbPadSquareTile(Arg0& left, Arg1& trans, Arg2& nr, Arg3& nc, Arg4& packed)
{
    if (nr != width || nc != width) {
        const uint32_t logicalRows = trans == ACLBLAS_OP_N ? nc : nr;
        const uint32_t logicalCols = trans == ACLBLAS_OP_N ? nr : nc;
        const uint32_t activeRows = left ? 2U * logicalRows : logicalRows;
        const uint32_t allRows = left ? 2U * width : width;
        if (activeRows < allRows) {
            SbFill(packed[activeRows * 2U * width], 0.0f, (allRows - activeRows) * 2U * width);
        }
        for (uint32_t col = (2U * logicalCols / 64U) * 64U; col < 2U * width; col += 64U) {
            const uint32_t firstInvalid = 2U * logicalCols > col ? 2U * logicalCols - col : 0U;
            const uint32_t validBits = 2U * width - col < 64U ? 2U * width - col : 64U;
            const uint64_t all = validBits == 64U ? ~0ULL : (1ULL << validBits) - 1ULL;
            uint64_t mask[2] = {all & (~0ULL << firstInvalid), 0U};
            Duplicate(packed[col], 0.0f, mask, static_cast<uint8_t>(activeRows), 1U, width / 4U);
        }
    }
}

template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
__aicore__ inline void SbGatherSquareTile(
    Arg0& tiles, Arg1& input, Arg2& output, Arg3& left, Arg4& trans, Arg5& length, Arg6& offsets, Arg7& nb,
    Arg8& physicalRow, Arg9& packed)
{
    constexpr uint32_t matrix = width * width;
    constexpr uint32_t count = 4U * matrix;
    auto whole = tiles == 1U ? output : packed;
    const uint32_t wholeLength = (tiles == 1U ? nb : 1U) * length;
    const uint32_t wholeRows = (tiles == 1U ? nb : 1U) * width;
    // Gather uses an 8-bit repeat count; keep each call below 256 repeats.
    for (uint32_t offset = 0; offset < wholeLength; offset += 8192U) {
        const uint32_t part = wholeLength - offset < 8192U ? wholeLength - offset : 8192U;
        SbGather(whole[offset], input, offsets[offset], tiles == 1U ? 0U : physicalRow * 8U, part);
    }
    if (left) {
        const uint32_t start = trans == ACLBLAS_OP_C ? 0U : 2U * width;
        for (uint32_t column = 0; column < 2U * width; column += 64U) {
            const uint32_t bits = 2U * width - column < 64U ? 2U * width - column : 64U;
            const uint64_t active = bits == 64U ? ~0ULL : ((1ULL << bits) - 1ULL);
            uint64_t mask[2] = {(trans == ACLBLAS_OP_C ? 0xaaaaaaaaaaaaaaaaULL : 0x5555555555555555ULL) & active, 0U};
            Muls(
                whole[start + column], whole[start + column], -1.0f, mask, static_cast<uint8_t>(wholeRows),
                {1, 1, width / 2U, width / 2U});
        }
    } else if (trans == ACLBLAS_OP_C) {
        uint64_t mask[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
        Muls(whole, whole, -1.0f, mask, static_cast<uint8_t>(wholeLength / 64U), {1, 1, 8, 8});
    }
}

template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6>
__aicore__ inline void SbExpandSquareIndices(
    Arg0& t, Arg1& tiles, Arg2& gatherPitch, Arg3& ix, Arg4& left, Arg5& trans, Arg6& length)
{
    constexpr uint32_t groupCap = width <= 24U ? 8U : (width <= 32U ? 4U : (width <= 48U ? 2U : 1U));
    for (uint32_t columns = 16U; columns < 2U * width; columns *= 2U) {
        const uint32_t step = trans == ACLBLAS_OP_N ? columns * 4U : columns / 2U * gatherPitch * 4U;
        const uint32_t part = 2U * width - columns < columns ? 2U * width - columns : columns;
        SbIndexAdd(ix[columns], ix, static_cast<int32_t>(step), part);
        if (left) {
            SbIndexAdd(ix[2U * width + columns], ix[2U * width], static_cast<int32_t>(step), part);
        }
    }
    const uint32_t rowFloats = left ? 4U * width : 2U * width;
    for (uint32_t rows = 1U; rows < width; rows *= 2U) {
        const uint32_t step = trans == ACLBLAS_OP_N ? rows * gatherPitch * 4U : rows * 8U;
        const uint32_t part = width - rows < rows ? width - rows : rows;
        SbIndexAdd(ix[rows * rowFloats], ix, static_cast<int32_t>(step), part * rowFloats);
    }
    if (tiles == 1U) {
        for (uint32_t batches = 1U; batches < groupCap; batches *= 2U) {
            SbIndexAdd(ix[batches * length], ix, static_cast<int32_t>(batches * t.m * t.m * 8U), batches * length);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__aicore__ inline void SbLoadSquareTile(
    Arg0& t, Arg1& tiles, Arg2& input, Arg3& source, Arg4& first, Arg5& nb, Arg6& i, Arg7& batch, Arg8& inner,
    Arg9& physicalCol, Arg10& nc)
{
    if (tiles == 1U) {
        if (i == 0U) {
            DataCopyExtParams ic{1U, nb * t.m * t.m * 8U, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
            DataCopyPad(input, source[static_cast<uint64_t>(first) * t.m * t.m * 2U], ic, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    } else {
        if (inner == 0U) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopyExtParams ic{1U, nc * t.m * 8U, 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
            DataCopyPad(input, source[(static_cast<uint64_t>(batch) * t.m * t.m + physicalCol * t.m) * 2U], ic, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }
    if (tiles != 1U) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6>
__aicore__ inline void SbSeedSquareIndices(
    Arg0& t, Arg1& tiles, Arg2& gatherPitch, Arg3& ix, Arg4& left, Arg5& trans, Arg6& length)
{
    uint64_t oddSeed[2] = {0xaaaaULL, 0U};
    uint64_t evenSeed[2] = {0x5555ULL, 0U};
    SetMaskNorm();
    SbIndex(ix, 0, 16U);
    if (trans == ACLBLAS_OP_N) {
        SbIndexMul(ix, ix, 4, 16U);
        if (left) {
            SbIndex(ix[2U * width], 1, 16U);
            Adds(ix[2U * width], ix[2U * width], -2, oddSeed, 1U, {1, 1, 8, 8});
            SbIndexMul(ix[2U * width], ix[2U * width], 4, 16U);
        }
    } else {
        SbIndexShift(ix, ix, 1, 16U);
        SbIndexMul(ix, ix, static_cast<int32_t>(gatherPitch * 4U), 16U);
        if (left) {
            SbIndexAdd(ix[2U * width], ix, 0, 16U);
        }
        Adds(ix, ix, 4, oddSeed, 1U, {1, 1, 8, 8});
        if (left) {
            Adds(ix[2U * width], ix[2U * width], 4, evenSeed, 1U, {1, 1, 8, 8});
        }
    }
    SbExpandSquareIndices<width, cubeWidth>(t, tiles, gatherPitch, ix, left, trans, length);
}

template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9>
__aicore__ inline void SbStoreSquareBatch(
    Arg0& t, Arg1& tiles, Arg2& area, Arg3& output, Arg4& left, Arg5& length, Arg6& ws, Arg7& rounds, Arg8& first,
    Arg9& nb)
{
    if (tiles == 1U) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams oc{
            static_cast<uint16_t>(nb), length * 4U, 0U,
            static_cast<uint32_t>((SbCompactWorkspaceStride(t.m) - length) * sizeof(float)), 0U};
        DataCopyPad(ws[first * SbCompactWorkspaceStride(t.m) + (left ? 0U : 4U * area)], output, oc);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(8U);
    ++rounds;
}

template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11,
    typename Arg12, typename Arg13, typename Arg14, typename Arg15, typename Arg16>
__aicore__ inline void SbPackSquareRow(
    Arg0& t, Arg1& padded, Arg2& tiles, Arg3& area, Arg4& input, Arg5& output, Arg6& left, Arg7& trans, Arg8& length,
    Arg9& offsets, Arg10& source, Arg11& ws, Arg12& first, Arg13& nb, Arg14& i, Arg15& batch, Arg16& outer)
{
    for (uint32_t inner = 0; inner < tiles; ++inner) {
        const uint32_t physicalRow = inner * width;
        const uint32_t physicalCol = outer * width;
        const uint32_t logicalRow = trans == ACLBLAS_OP_N ? physicalCol : physicalRow;
        const uint32_t logicalCol = trans == ACLBLAS_OP_N ? physicalRow : physicalCol;
        const uint32_t nr = t.m - physicalRow < width ? t.m - physicalRow : width;
        const uint32_t nc = t.m - physicalCol < width ? t.m - physicalCol : width;
        SbLoadSquareTile<width, cubeWidth>(t, tiles, input, source, first, nb, i, batch, inner, physicalCol, nc);
        auto packed = output[(tiles == 1U ? i : 0U) * length];
        if (tiles != 1U || i == 0U) {
            SbGatherSquareTile<width, cubeWidth>(
                tiles, input, output, left, trans, length, offsets, nb, physicalRow, packed);
        }
        SbPadSquareTile<width, cubeWidth>(left, trans, nr, nc, packed);
        if (tiles != 1U) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const uint32_t outRows = left ? 2U * width : width;
            const uint64_t dst = batch * SbCompactWorkspaceStride(t.m) + (left ? 0U : 4U * area) +
                                 static_cast<uint64_t>(left ? logicalRow * 2U : logicalRow) * padded * 2U +
                                 logicalCol * 2U;
            DataCopyExtParams oc{static_cast<uint16_t>(outRows), width * 8U, 0U, (padded - width) * 8U, 0U};
            DataCopyPad(ws[dst], packed, oc);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
}

template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11,
    typename Arg12, typename Arg13>
__aicore__ inline void SbRunSquareBatches(
    Arg0& t, Arg1& cores, Arg2& padded, Arg3& tiles, Arg4& area, Arg5& ownership, Arg6& input, Arg7& output, Arg8& left,
    Arg9& trans, Arg10& length, Arg11& offsets, Arg12& source, Arg13& ws)
{
    if (tiles != 1U) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    const uint32_t block = GetBlockIdx() / 2U;
    uint32_t rounds = 0U;
    for (uint32_t group = block; group * ownership < t.batchCount; group += cores) {
        const uint32_t first = group * ownership;
        const uint32_t nb = t.batchCount - first < ownership ? t.batchCount - first : ownership;
        if (rounds != 0U && rounds % 4U == 0U) {
            CrossCoreWaitFlag<2>(9U);
        }
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t batch = first + i;
            for (uint32_t outer = 0; outer < tiles; ++outer) {
                SbPackSquareRow<width, cubeWidth>(
                    t, padded, tiles, area, input, output, left, trans, length, offsets, source, ws, first, nb, i,
                    batch, outer);
            }
        }
        SbStoreSquareBatch<width, cubeWidth>(t, tiles, area, output, left, length, ws, rounds, first, nb);
    }
    if (tiles != 1U) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    if (rounds != 0U) {
        CrossCoreWaitFlag<2>(9U);
    }
}

template <
    uint32_t width, uint32_t cubeWidth, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11,
    typename Arg12, typename Arg13, typename Arg14>
__aicore__ inline void SbRunSquareVector(
    Arg0& a, Arg1& b, Arg2& c, Arg3& workspace, Arg4& t, Arg5& cores, Arg6& padded, Arg7& atomicOutput, Arg8& tiles,
    Arg9& gatherPitch, Arg10& area, Arg11& ownership, Arg12& outputBuf, Arg13& indices, Arg14& input)
{
    constexpr uint32_t matrix = width * width;
    constexpr uint32_t count = 4U * matrix;
    constexpr uint32_t copyCap = 4096U;
    auto output = outputBuf.template Get<float>();
    auto ix = indices.template Get<int32_t>();
    const bool left = GetSubBlockIdx() == 0U;
    const uint32_t trans = left ? t.transA : t.transB;
    const uint32_t length = left ? count : 2U * matrix;
    // Build a small periodic pattern, then duplicate complete rows geometrically.
    SbSeedSquareIndices<width, cubeWidth>(t, tiles, gatherPitch, ix, left, trans, length);
    auto offsets = indices.template Get<uint32_t>();
    GlobalTensor<float> source, ws;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(left ? a : b));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    // Unaligned batch boundaries use element-wise atomic stores. Initialize
    // C in disjoint cache-line-aligned ranges before any Cube result is written.
    if (atomicOutput) {
        GlobalTensor<float> cg;
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
        SbFill(output, 0.0f, copyCap);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const uint64_t total = static_cast<uint64_t>(t.batchCount) * t.m * t.m * 2U;
        for (uint64_t pos = GetBlockIdx() * copyCap; pos < total; pos += cores * 2U * copyCap) {
            const uint32_t length = total - pos < copyCap ? total - pos : copyCap;
            DataCopyExtParams zeroCopy{1U, length * 4U, 0U, 0U, 0U};
            DataCopyPad(cg[pos], output, zeroCopy);
        }
        CrossCoreSetFlag<0, PIPE_MTE3>(10U);
        CrossCoreWaitFlag<0>(10U);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    SbRunSquareBatches<width, cubeWidth>(
        t, cores, padded, tiles, area, ownership, input, output, left, trans, length, offsets, source, ws);
}

template <uint32_t width, uint32_t cubeWidth = width>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_square_mixed_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    constexpr uint32_t matrix = width * width;
    constexpr uint32_t count = 4U * matrix;
    constexpr uint32_t inputPitch = 2U * width + 8U;
    constexpr uint32_t inputSize = width * inputPitch;
    const uint32_t padded = (t.m + width - 1U) / width * width;
    const bool atomicOutput = t.m % 4U != 0U;
    const uint32_t tiles = padded / width;
    const uint32_t gatherPitch = t.m * 2U;
    const uint32_t area = padded * padded;
    constexpr uint32_t groupCap = width <= 24U ? 8U : (width <= 32U ? 4U : (width <= 48U ? 2U : 1U));
    const uint32_t balanced = (t.batchCount + cores * 2U - 1U) / (cores * 2U);
    const uint32_t ownership = tiles == 1U ? (balanced < groupCap ? balanced : groupCap) : 1U;
    constexpr uint32_t copyCap = 4096U;
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> inputBuf, outputBuf, indices;
        const uint32_t inputCapacity =
            tiles == 1U ? groupCap * inputSize : ((width - 1U) * 2U * t.m + 2U * padded + 7U) / 8U * 8U;
        const uint32_t bufferBatches = tiles == 1U ? groupCap : 1U;
        pipe.InitBuffer(inputBuf, (inputCapacity > copyCap ? inputCapacity : copyCap) * sizeof(float));
        pipe.InitBuffer(outputBuf, (bufferBatches * count > copyCap ? bufferBatches * count : copyCap) * sizeof(float));
        pipe.InitBuffer(
            indices, (bufferBatches * count > copyCap ? bufferBatches * count : copyCap) * sizeof(uint32_t));
        auto input = inputBuf.Get<float>();
        SbRunSquareVector<width, cubeWidth>(
            a, b, c, workspace, t, cores, padded, atomicOutput, tiles, gatherPitch, area, ownership, outputBuf, indices,
            input);
    }
    if ASCEND_IS_AIC {
        AscendC::InitSocState();
        SetHF32Mode(HF32Mode::DISABLE);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        if (atomicOutput) {
            SetAtomicAdd<float>();
        }
        SbRunSquareCube<width, cubeWidth>(c, workspace, t, cores, padded, tiles, area, ownership);
        if (atomicOutput) {
            SetAtomicNone();
        }
    }
}
} // namespace cgemm_sb

// ---- balanced implementation ----
namespace cgemm_sb {
template <
    uint32_t width, bool left, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6>
__aicore__ inline void SbExpandHalfNzIndices(
    Arg0& ix, Arg1& m, Arg2& trans, Arg3& sub, Arg4& zeros, Arg5& pitch, Arg6& zeroIndex)
{
    constexpr uint32_t rows = left ? 2U * width : (width + 15U) / 16U * 16U;
    constexpr uint32_t length = rows * width;
    SbNzSeed<left>(ix, trans, pitch);
    for (uint32_t r = left ? 2U : 1U; r < rows; r *= 2U) {
        const uint32_t part = rows - r < r ? rows - r : r;
        SbIndexAdd(ix[r * 8U], ix, (left ? r / 2U : r) * (trans == ACLBLAS_OP_N ? pitch * 4U : 8U), part * 8U);
    }
    for (uint32_t blocks = 1U; blocks < width / 8U; blocks *= 2U) {
        const uint32_t part = width / 8U - blocks < blocks ? width / 8U - blocks : blocks;
        SbIndexAdd(ix[blocks * rows * 8U], ix, blocks * (trans == ACLBLAS_OP_N ? 32U : pitch * 16U), part * rows * 8U);
    }
    if (trans == ACLBLAS_OP_N) {
        SbIndexAdd(ix, ix, sub * width / 2U * 8U, length);
    }
    const uint32_t activeRows = left ? 2U * m : m;
    for (uint32_t pos = activeRows * 8U; pos < rows * 8U; pos += 64U) {
        const uint32_t n = rows * 8U - pos < 64U ? rows * 8U - pos : 64U;
        if (m % 4U == 0U) {
            Adds(
                ix[pos], zeros, 0, static_cast<uint64_t>(n), static_cast<uint8_t>(width / 8U),
                {1, 1, static_cast<uint8_t>(rows), 0});
        } else {
            Duplicate(
                ix[pos], static_cast<int32_t>(zeroIndex * 4U), static_cast<uint64_t>(n),
                static_cast<uint8_t>(width / 8U), 1U, static_cast<uint8_t>(rows));
        }
    }
}

template <uint32_t width, bool left>
__aicore__ inline void SbHalfNzIndex(
    LocalTensor<int32_t> ix, uint32_t m, uint32_t trans, uint32_t sub, LocalTensor<int32_t> zeros)
{
    constexpr uint32_t rows = left ? 2U * width : (width + 15U) / 16U * 16U;

    const uint32_t pitch = 2U * m;
    const uint32_t zeroIndex = (2U * m * m + 7U) / 8U * 8U;
    SbExpandHalfNzIndices<width, left>(ix, m, trans, sub, zeros, pitch, zeroIndex);
    const uint32_t remaining = m - sub * width / 2U;
    const uint32_t cols = remaining < width / 2U ? remaining : width / 2U;
    if (cols % 4U != 0U) {
        uint64_t mask[2] = {0xffULL & (~0ULL << (cols % 4U * 2U)), 0U};
        Duplicate(
            ix[cols / 4U * rows * 8U], static_cast<int32_t>(zeroIndex * 4U), mask, static_cast<uint8_t>(rows), 1U, 1U);
    }
    for (uint32_t block = (cols + 3U) / 4U; block < width / 8U; ++block) {
        if (m % 4U == 0U) {
            Adds(ix[block * rows * 8U], zeros, 0, 64ULL, static_cast<uint8_t>(rows / 8U), {1, 1, 8, 0});
        } else {
            Duplicate(
                ix[block * rows * 8U], static_cast<int32_t>(zeroIndex * 4U), 64ULL, static_cast<uint8_t>(rows / 8U), 1U,
                8U);
        }
    }
}

template <
    uint32_t width, uint32_t staticAB, uint32_t fixedM, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10,
    typename Arg11, typename Arg12, typename Arg13, typename Arg14, typename Arg15, typename Arg16, typename Arg17,
    typename Arg18, typename Arg19, typename Arg20>
__aicore__ inline void SbPackBalancedOperands(
    Arg0& t, Arg1& cores, Arg2& dimension, Arg3& transA, Arg4& transB, Arg5& input, Arg6& sub, Arg7& ag, Arg8& bg,
    Arg9& ws, Arg10& aOffset, Arg11& bOffset, Arg12& cpA, Arg13& cpB, Arg14& pad, Arg15& rounds, Arg16& batch,
    Arg17& next, Arg18& offsets, Arg19& ao, Arg20& bo)
{
    constexpr uint32_t area = width * width;
    SbGather(ao, input, offsets, 0U, 2U * area);
    if (batch + cores < t.batchCount) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(input, ag[next + aOffset], cpA, pad);
    }
    const uint64_t signBits = transB == ACLBLAS_OP_C ?
                                  (transA == ACLBLAS_OP_C ? 0xffaaffaaffaaffaaULL : 0xaa00aa00aa00aa00ULL) :
                                  (transA == ACLBLAS_OP_C ? 0x00aa00aa00aa00aaULL : 0x5500550055005500ULL);
    uint64_t amask[2] = {signBits, 0U};
    Muls(ao, ao, -1.0f, amask, static_cast<uint8_t>(2U * area / 64U), {1, 1, 8, 8});
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(ws[batch * SbCompactWorkspaceStride(dimension) + sub * 2U * area], ao, 2U * area);
    if (rounds % 2U == 0U) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    } else {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SbGather(bo, input[2U * area], offsets[2U * area], 0U, area);
    if (batch + cores < t.batchCount) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(input[2U * area], bg[next + bOffset], cpB, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

template <
    uint32_t width, uint32_t staticAB, uint32_t fixedM, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
__aicore__ inline void SbRunBalancedCube(
    Arg0& c, Arg1& workspace, Arg2& t, Arg3& cores, Arg4& dimension, Arg5& atomicOutput)
{
    constexpr uint32_t area = width * width;
    InitSocState();
    SetHF32Mode(HF32Mode::DISABLE);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    if (atomicOutput) {
        SetAtomicAdd<float>();
    }
    uint32_t rounds = 0U;
    for (uint32_t batch = GetBlockIdx(); batch < t.batchCount; batch += cores) {
        CrossCoreWaitFlag<2>(8U);
        auto* ws = reinterpret_cast<__gm__ float*>(workspace) + batch * SbCompactWorkspaceStride(dimension);
        CgemmSbAugmentedMmad<2U * width, width, 2U * width, false, true> mm;
        mm.Init(
            ws + 4U * area, ws,
            reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(batch) * dimension * dimension * 2U, dimension,
            2U * dimension, (2U * dimension + 15U) / 16U * 16U, 2U * width, 2U * dimension, 2U * width);
        mm.PreloadNz(1U, static_cast<uint32_t>(SbCompactWorkspaceStride(dimension)));
        mm.Process(0U, 1U, true, true, 0U, 0U, 0U, 0U, 0U, rounds % 2U);
        ++rounds;
        if (rounds % 4U == 0U || batch + cores >= t.batchCount) {
            CrossCoreSetFlag<2, PIPE_FIX>(9U);
        }
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    if (atomicOutput) {
        SetAtomicNone();
    }
}

template <
    uint32_t width, uint32_t staticAB, uint32_t fixedM, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10,
    typename Arg11, typename Arg12, typename Arg13, typename Arg14, typename Arg15, typename Arg16>
__aicore__ inline void SbRunBalancedBatches(
    Arg0& t, Arg1& cores, Arg2& dimension, Arg3& transA, Arg4& transB, Arg5& xb, Arg6& input, Arg7& output, Arg8& sub,
    Arg9& ag, Arg10& bg, Arg11& ws, Arg12& aOffset, Arg13& bOffset, Arg14& cpA, Arg15& cpB, Arg16& pad)
{
    constexpr uint32_t area = width * width;
    uint32_t rounds = 0U;
    for (uint32_t batch = GetBlockIdx() / 2U; batch < t.batchCount; batch += cores) {
        if (rounds != 0U && rounds % 4U == 0U) {
            CrossCoreWaitFlag<2>(9U);
        }
        const uint64_t next = static_cast<uint64_t>(batch + cores) * dimension * dimension * 2U;
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto offsets = xb.template Get<uint32_t>();
        if (rounds % 2U == 0U) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        } else {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
        auto ao = output[rounds % 2U * 2U * area];
        auto bo = output[4U * area];
        SbPackBalancedOperands<width, staticAB, fixedM>(
            t, cores, dimension, transA, transB, input, sub, ag, bg, ws, aOffset, bOffset, cpA, cpB, pad, rounds, batch,
            next, offsets, ao, bo);
        DataCopy(ws[batch * SbCompactWorkspaceStride(dimension) + 4U * area + sub * area], bo, area);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        CrossCoreSetFlag<2, PIPE_MTE3>(8U);
        ++rounds;
    }
    SbDrainPackedOutput<true>(rounds);
}

template <
    uint32_t width, uint32_t staticAB, uint32_t fixedM, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6>
__aicore__ inline void SbPrepareBalancedIndices(
    Arg0& dimension, Arg1& transA, Arg2& transB, Arg3& ib, Arg4& input, Arg5& ix, Arg6& sub)
{
    constexpr uint32_t area = width * width;
    SetMaskNorm();
    auto zeroIndices = ib.template Get<int32_t>();
    if (dimension % 4U == 0U && dimension != width) {
        SbIndex(zeroIndices, 0, 8U);
        for (uint32_t span = 8U; span < 64U; span *= 2U) {
            SbIndexAdd(zeroIndices[span], zeroIndices, span, span);
        }
        SbIndexMul(zeroIndices, zeroIndices, 4, 64U);
        SbIndexAdd(zeroIndices, zeroIndices, static_cast<int32_t>((dimension * dimension * 2U + 7U) / 8U * 32U), 64U);
    }
    SbHalfNzIndex<width, true>(ix, dimension, transA, sub, zeroIndices);
    SbHalfNzIndex<width, false>(ix[2U * area], dimension, transB, sub, zeroIndices);
    if (dimension != width) {
        const uint32_t zero = (dimension * dimension * 2U + 7U) / 8U * 8U;
        SbFill(input[zero], 0.0f, dimension % 4U == 0U ? 64U : 8U);
        SbFill(input[2U * area + zero], 0.0f, dimension % 4U == 0U ? 64U : 8U);
    }
    PipeBarrier<PIPE_ALL>();
}

template <
    uint32_t width, uint32_t staticAB, uint32_t fixedM, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10,
    typename Arg11>
__aicore__ inline void SbPrefetchBalancedBatches(
    Arg0& t, Arg1& cores, Arg2& dimension, Arg3& transA, Arg4& transB, Arg5& xb, Arg6& input, Arg7& output, Arg8& sub,
    Arg9& ag, Arg10& bg, Arg11& ws)
{
    constexpr uint32_t area = width * width;
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    const uint32_t halfColumns = dimension - sub * width / 2U < width / 2U ? dimension - sub * width / 2U : width / 2U;
    const uint32_t aOffset = transA == ACLBLAS_OP_N ? 0U : sub * width * dimension;
    const uint32_t bOffset = transB == ACLBLAS_OP_N ? 0U : sub * width * dimension;
    DataCopyExtParams cpA{1U, dimension * (transA == ACLBLAS_OP_N ? dimension : halfColumns) * 8U, 0U, 0U, 0U};
    DataCopyExtParams cpB{1U, dimension * (transB == ACLBLAS_OP_N ? dimension : halfColumns) * 8U, 0U, 0U, 0U};
    DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    if (GetBlockIdx() / 2U < t.batchCount) {
        const uint64_t initial = static_cast<uint64_t>(GetBlockIdx() / 2U) * dimension * dimension * 2U;
        DataCopyPad(input, ag[initial + aOffset], cpA, pad);
        DataCopyPad(input[2U * area], bg[initial + bOffset], cpB, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    SbRunBalancedBatches<width, staticAB, fixedM>(
        t, cores, dimension, transA, transB, xb, input, output, sub, ag, bg, ws, aOffset, bOffset, cpA, cpB, pad);
}

template <
    uint32_t width, uint32_t staticAB, uint32_t fixedM, typename Arg0, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10,
    typename Arg11, typename Arg12, typename Arg13>
__aicore__ inline void SbRunBalancedVector(
    Arg0& a, Arg1& b, Arg2& c, Arg3& workspace, Arg4& t, Arg5& cores, Arg6& dimension, Arg7& transA, Arg8& transB,
    Arg9& atomicOutput, Arg10& pipe, Arg11& ib, Arg12& ob, Arg13& xb)
{
    constexpr uint32_t area = width * width;
    pipe.InitBuffer(ob, 5U * area * 4U);
    pipe.InitBuffer(xb, 3U * area * 4U);
    auto input = ib.template Get<float>();
    auto output = ob.template Get<float>();
    auto ix = xb.template Get<int32_t>();
    const uint32_t sub = GetSubBlockIdx();
    SbPrepareBalancedIndices<width, staticAB, fixedM>(dimension, transA, transB, ib, input, ix, sub);
    GlobalTensor<float> ag, bg, cg, ws;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    if (atomicOutput) {
        SbFill(output, 0.0f, area);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const uint64_t total = static_cast<uint64_t>(t.batchCount) * dimension * dimension * 2U;
        SbZeroOutput<area>(cg, output, total, cores);
    }
    SbPrefetchBalancedBatches<width, staticAB, fixedM>(
        t, cores, dimension, transA, transB, xb, input, output, sub, ag, bg, ws);
}

template <uint32_t width, uint32_t staticAB = 9U, uint32_t fixedM = 0U>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_balanced_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const uint32_t dimension = fixedM == 0U ? t.m : fixedM;
    const uint32_t transA = staticAB == 9U ? t.transA : ACLBLAS_OP_N + staticAB / 3U;
    const uint32_t transB = staticAB == 9U ? t.transB : ACLBLAS_OP_N + staticAB % 3U;
    constexpr uint32_t area = width * width;
    const bool atomicOutput = dimension % 4U != 0U;
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> ib, ob, xb;
        pipe.InitBuffer(ib, 4U * area * 4U);
        SbRunBalancedVector<width, staticAB, fixedM>(
            a, b, c, workspace, t, cores, dimension, transA, transB, atomicOutput, pipe, ib, ob, xb);
    }
    if ASCEND_IS_AIC {
        SbRunBalancedCube<width, staticAB, fixedM>(c, workspace, t, cores, dimension, atomicOutput);
    }
}
template <uint32_t width, uint32_t transposePair>
void SbLaunchBalancedShape(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling& t, uint32_t cores, void* stream)
{
    if (t.m == width - 4U) {
        cgemm_sb_balanced_kernel<width, transposePair, width - 4U>
            <<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
    } else {
        cgemm_sb_balanced_kernel<width, transposePair><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
    }
}

template <uint32_t width>
void cgemm_sb_balanced_launch(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling& t, uint32_t cores, void* stream)
{
    switch ((t.transA - ACLBLAS_OP_N) * 3U + t.transB - ACLBLAS_OP_N) {
        case 0U:
            SbLaunchBalancedShape<width, 0U>(a, b, c, workspace, t, cores, stream);
            break;
        case 1U:
            SbLaunchBalancedShape<width, 1U>(a, b, c, workspace, t, cores, stream);
            break;
        case 2U:
            SbLaunchBalancedShape<width, 2U>(a, b, c, workspace, t, cores, stream);
            break;
        case 3U:
            SbLaunchBalancedShape<width, 3U>(a, b, c, workspace, t, cores, stream);
            break;
        case 4U:
            SbLaunchBalancedShape<width, 4U>(a, b, c, workspace, t, cores, stream);
            break;
        case 5U:
            SbLaunchBalancedShape<width, 5U>(a, b, c, workspace, t, cores, stream);
            break;
        case 6U:
            SbLaunchBalancedShape<width, 6U>(a, b, c, workspace, t, cores, stream);
            break;
        case 7U:
            SbLaunchBalancedShape<width, 7U>(a, b, c, workspace, t, cores, stream);
            break;
        case 8U:
            SbLaunchBalancedShape<width, 8U>(a, b, c, workspace, t, cores, stream);
            break;
        default:
            // Host validation rejects unsupported transpose values before dispatch.
            return;
    }
}
} // namespace cgemm_sb

// ---- balanced pair implementation ----
namespace cgemm_sb {
template <
    uint32_t width, uint32_t staticAB, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11,
    typename Arg12, typename Arg13, typename Arg14, typename Arg15, typename Arg16>
__aicore__ inline void SbPackBalancedPair(
    Arg0& t, Arg1& xb, Arg2& input, Arg3& sub, Arg4& ag, Arg5& bg, Arg6& ws, Arg7& aOffset, Arg8& bOffset, Arg9& cpA,
    Arg10& cpB, Arg11& pad, Arg12& first, Arg13& nb, Arg14& next, Arg15& ao, Arg16& bo)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t bHalf = cubeM * width;
    constexpr uint32_t groupSize = 4U * area + 2U * bHalf;
    constexpr uint32_t workspaceStride = width == 32U ? 8U * area : groupSize + 16U;
    constexpr uint32_t transA = ACLBLAS_OP_N + staticAB / 3U;
    constexpr uint32_t transB = ACLBLAS_OP_N + staticAB % 3U;
    auto offsets = xb.template Get<uint32_t>();
    SbGather(ao, input, offsets, 0U, nb * 2U * area);
    if (next < t.batchCount) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        cpA.blockCount = t.batchCount - next < 2U ? 1U : 2U;
        DataCopyPad(input, ag[static_cast<uint64_t>(next) * t.m * t.m * 2U + aOffset], cpA, pad);
    }
    constexpr uint64_t signBits = transB == ACLBLAS_OP_C ?
                                      (transA == ACLBLAS_OP_C ? 0xffaaffaaffaaffaaULL : 0xaa00aa00aa00aa00ULL) :
                                      (transA == ACLBLAS_OP_C ? 0x00aa00aa00aa00aaULL : 0x5500550055005500ULL);
    uint64_t amask[2] = {signBits, 0U};
    Muls(ao, ao, -1.0f, amask, static_cast<uint8_t>(nb * 2U * area / 64U), {1, 1, 8, 8});
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams outA{
        static_cast<uint16_t>(nb), 2U * area * 4U, 0U, static_cast<uint32_t>((workspaceStride - 2U * area) * 4U), 0U};
    DataCopyPad(ws[first * workspaceStride + sub * 2U * area], ao, outA);
    SbGather(bo, input[4U * area], offsets[4U * area], 0U, nb * bHalf);
    if (next < t.batchCount) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        cpB.blockCount = t.batchCount - next < 2U ? 1U : 2U;
        DataCopyPad(input[4U * area], bg[static_cast<uint64_t>(next) * t.m * t.m * 2U + bOffset], cpB, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

template <uint32_t width, uint32_t staticAB, typename Arg0, typename Arg1, typename Arg2, typename Arg3>
__aicore__ inline void SbRunPairCube(Arg0& c, Arg1& workspace, Arg2& t, Arg3& cores)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t bHalf = cubeM * width;
    constexpr uint32_t groupSize = 4U * area + 2U * bHalf;
    constexpr uint32_t workspaceStride = width == 32U ? 8U * area : groupSize + 16U;
    InitSocState();
    SetHF32Mode(HF32Mode::DISABLE);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    uint32_t rounds = 0U;
    for (uint32_t first = GetBlockIdx() * 2U; first < t.batchCount; first += cores * 2U) {
        const uint32_t nb = t.batchCount - first < 2U ? 1U : 2U;
        CrossCoreWaitFlag<2>(8U);
        auto* ws = reinterpret_cast<__gm__ float*>(workspace) + first * workspaceStride;
        auto* cg = reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(first) * t.m * t.m * 2U;
        CgemmSbAugmentedMmad<2U * width, cubeM, 2U * width, false, true> mm;
        mm.Init(ws + 4U * area, ws, cg, t.m, 2U * t.m, (2U * t.m + 15U) / 16U * 16U, 2U * width, 2U * t.m, 2U * width);
        mm.PreloadNz(nb, static_cast<uint32_t>(workspaceStride));
        for (uint32_t slot = 0U; slot < nb; ++slot) {
            auto* current = ws + slot * workspaceStride;
            mm.Init(
                current + 4U * area, current, cg + slot * t.m * t.m * 2U, t.m, 2U * t.m, (2U * t.m + 15U) / 16U * 16U,
                2U * width, 2U * t.m, 2U * width);
            mm.Process(0U, 1U, true, true, slot, 0U, 0U, 0U, 0U, slot);
        }
        if (++rounds % 4U == 0U || first + cores * 2U >= t.batchCount) {
            CrossCoreSetFlag<2, PIPE_FIX>(9U);
        }
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
}

template <
    uint32_t width, uint32_t staticAB, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11,
    typename Arg12, typename Arg13, typename Arg14, typename Arg15>
__aicore__ inline void SbRunPairBatches(
    Arg0& t, Arg1& cores, Arg2& xb, Arg3& input, Arg4& output, Arg5& sub, Arg6& ag, Arg7& bg, Arg8& ws, Arg9& aOffset,
    Arg10& bOffset, Arg11& cpA, Arg12& cpB, Arg13& pad, Arg14& initial, Arg15& rounds)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t bHalf = cubeM * width;
    constexpr uint32_t groupSize = 4U * area + 2U * bHalf;
    constexpr uint32_t workspaceStride = width == 32U ? 8U * area : groupSize + 16U;
    for (uint32_t first = initial; first < t.batchCount; first += cores * 2U) {
        const uint32_t nb = t.batchCount - first < 2U ? 1U : 2U;
        const uint32_t next = first + cores * 2U;
        if (rounds != 0U && rounds % 4U == 0U) {
            CrossCoreWaitFlag<2>(9U);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (rounds % 2U == 0U) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        } else {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
        auto ao = output[rounds % 2U * groupSize];
        auto bo = output[rounds % 2U * groupSize + 4U * area];
        SbPackBalancedPair<width, staticAB>(
            t, xb, input, sub, ag, bg, ws, aOffset, bOffset, cpA, cpB, pad, first, nb, next, ao, bo);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams outB{
            static_cast<uint16_t>(nb), bHalf * 4U, 0U, static_cast<uint32_t>((workspaceStride - bHalf) * 4U), 0U};
        DataCopyPad(ws[first * workspaceStride + 4U * area + sub * bHalf], bo, outB);
        SbSignalPackedRound(rounds);
        ++rounds;
    }
    SbDrainPackedOutput<false>(rounds);
}

template <
    uint32_t width, uint32_t staticAB, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5, typename Arg6, typename Arg7, typename Arg8>
__aicore__ inline void SbPrefetchPairBatches(
    Arg0& a, Arg1& b, Arg2& workspace, Arg3& t, Arg4& cores, Arg5& xb, Arg6& input, Arg7& output, Arg8& sub)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t transA = ACLBLAS_OP_N + staticAB / 3U;
    constexpr uint32_t transB = ACLBLAS_OP_N + staticAB % 3U;
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> ag, bg, ws;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    const uint32_t cols = t.m - sub * width / 2U < width / 2U ? t.m - sub * width / 2U : width / 2U;
    const uint32_t aOffset = transA == ACLBLAS_OP_N ? 0U : sub * width * t.m;
    const uint32_t bOffset = transB == ACLBLAS_OP_N ? 0U : sub * width * t.m;
    const uint32_t aCount = t.m * (transA == ACLBLAS_OP_N ? t.m : cols) * 2U;
    const uint32_t bCount = t.m * (transB == ACLBLAS_OP_N ? t.m : cols) * 2U;
    DataCopyExtParams cpA{2U, aCount * 4U, (t.m * t.m * 2U - aCount) * 4U, (2U * area - aCount) / 8U, 0U};
    DataCopyExtParams cpB{2U, bCount * 4U, (t.m * t.m * 2U - bCount) * 4U, (2U * area - bCount) / 8U, 0U};
    DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    const uint32_t initial = GetBlockIdx() / 2U * 2U;
    if (initial < t.batchCount) {
        cpA.blockCount = cpB.blockCount = t.batchCount - initial < 2U ? 1U : 2U;
        const uint64_t base = static_cast<uint64_t>(initial) * t.m * t.m * 2U;
        DataCopyPad(input, ag[base + aOffset], cpA, pad);
        DataCopyPad(input[4U * area], bg[base + bOffset], cpB, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    uint32_t rounds = 0U;
    SbRunPairBatches<width, staticAB>(
        t, cores, xb, input, output, sub, ag, bg, ws, aOffset, bOffset, cpA, cpB, pad, initial, rounds);
}

template <
    uint32_t width, uint32_t staticAB, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4,
    typename Arg5>
__aicore__ inline void SbRunPairVector(Arg0& a, Arg1& b, Arg2& workspace, Arg3& t, Arg4& cores, Arg5& pipe)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t bHalf = cubeM * width;
    constexpr uint32_t groupSize = 4U * area + 2U * bHalf;
    constexpr uint32_t transA = ACLBLAS_OP_N + staticAB / 3U;
    constexpr uint32_t transB = ACLBLAS_OP_N + staticAB % 3U;
    TBuf<TPosition::VECCALC> ib, ob, xb;
    pipe.InitBuffer(ib, 8U * area * 4U);
    pipe.InitBuffer(ob, 2U * groupSize * 4U);
    pipe.InitBuffer(xb, groupSize * 4U);
    auto input = ib.template Get<float>();
    auto output = ob.template Get<float>();
    auto ix = xb.template Get<int32_t>();
    const uint32_t sub = GetSubBlockIdx();
    SetMaskNorm();
    auto zeros = ib.template Get<int32_t>();
    if (t.m != width) {
        SbIndex(zeros, 0, 8U);
        for (uint32_t span = 8U; span < 64U; span *= 2U) {
            SbIndexAdd(zeros[span], zeros, span, span);
        }
        SbIndexMul(zeros, zeros, 4, 64U);
        SbIndexAdd(zeros, zeros, static_cast<int32_t>((2U * t.m * t.m + 7U) / 8U * 32U), 64U);
    }
    SbHalfNzIndex<width, true>(ix, t.m, transA, sub, zeros);
    SbHalfNzIndex<width, false>(ix[4U * area], t.m, transB, sub, zeros);
    SbIndexAdd(ix[2U * area], ix, 2U * area * 4U, 2U * area);
    SbIndexAdd(ix[4U * area + bHalf], ix[4U * area], 2U * area * 4U, bHalf);
    if (t.m != width) {
        const uint32_t zero = (t.m * t.m * 2U + 7U) / 8U * 8U;
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            SbFill(input[slot * 2U * area + zero], 0.0f, 64U);
        }
    }
    SbPrefetchPairBatches<width, staticAB>(a, b, workspace, t, cores, xb, input, output, sub);
}

template <uint32_t width, uint32_t staticAB>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_balanced_pair_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    static_assert(width == 32U || width == 40U, "Paired packing must fit the small-matrix UB layout");
    constexpr uint32_t area = width * width;
    constexpr uint32_t cubeM = (width + 15U) / 16U * 16U;
    constexpr uint32_t bHalf = cubeM * width;
    constexpr uint32_t groupSize = 4U * area + 2U * bHalf;

    if ASCEND_IS_AIV {
        TPipe pipe;
        SbRunPairVector<width, staticAB>(a, b, workspace, t, cores, pipe);
    }
    if ASCEND_IS_AIC {
        SbRunPairCube<width, staticAB>(c, workspace, t, cores);
    }
}
template <uint32_t width>
void cgemm_sb_balanced_pair_launch(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling& t, uint32_t cores, void* stream)
{
    switch ((t.transA - ACLBLAS_OP_N) * 3U + t.transB - ACLBLAS_OP_N) {
        case 0U:
            cgemm_sb_balanced_pair_kernel<width, 0U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        case 1U:
            cgemm_sb_balanced_pair_kernel<width, 1U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        case 2U:
            cgemm_sb_balanced_pair_kernel<width, 2U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        case 3U:
            cgemm_sb_balanced_pair_kernel<width, 3U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        case 4U:
            cgemm_sb_balanced_pair_kernel<width, 4U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        case 5U:
            cgemm_sb_balanced_pair_kernel<width, 5U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        case 6U:
            cgemm_sb_balanced_pair_kernel<width, 6U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        case 7U:
            cgemm_sb_balanced_pair_kernel<width, 7U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        case 8U:
            cgemm_sb_balanced_pair_kernel<width, 8U><<<cores, nullptr, stream>>>(a, b, c, workspace, t, cores);
            break;
        default:
            // Host validation rejects unsupported transpose values before dispatch.
            return;
    }
}
} // namespace cgemm_sb

// ---- padded implementation ----
namespace cgemm_sb {
template <uint32_t width, bool left>
__aicore__ inline void SbPaddedHalfNzIndex(LocalTensor<int32_t> ix, uint32_t m, uint32_t trans, uint32_t sub)
{
    constexpr uint32_t rows = left ? 2U * width : width;
    constexpr uint32_t length = width * width;
    constexpr uint32_t blocksCount = left ? width / 16U : width / 8U;
    constexpr uint32_t pitch = 2U * width + 8U;
    SbNzSeed<left>(ix, trans, pitch);
    for (uint32_t r = left ? 2U : 1U; r < rows; r *= 2U) {
        SbIndexAdd(ix[r * 8U], ix, (left ? r / 2U : r) * (trans == ACLBLAS_OP_N ? pitch * 4U : 8U), r * 8U);
    }
    for (uint32_t blocks = 1U; blocks < blocksCount; blocks *= 2U) {
        SbIndexAdd(
            ix[blocks * rows * 8U], ix, blocks * (trans == ACLBLAS_OP_N ? 32U : pitch * 16U), blocks * rows * 8U);
    }
    SbIndexAdd(ix, ix, sub * width / 2U * (trans == ACLBLAS_OP_N ? 8U : pitch * 4U), length);
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11, typename Arg12,
    typename Arg13, typename Arg14>
__aicore__ inline void SbPackPaddedOperands(
    Arg0& t, Arg1& input, Arg2& sub, Arg3& ag, Arg4& bg, Arg5& ws, Arg6& cp, Arg7& pad, Arg8& rounds, Arg9& batch,
    Arg10& nextBatch, Arg11& next, Arg12& offsets, Arg13& ao, Arg14& bo)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t inputPitch = 2U * width + 8U;
    constexpr uint32_t inputSize = width * inputPitch;
    SbGather(ao, input, offsets, 0U, area);
    SbGather(ao[area], input, offsets, width / 4U * (t.transA == ACLBLAS_OP_N ? 8U : inputPitch * 4U), area);
    if (nextBatch < t.batchCount) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(input, ag[next], cp, pad);
    }
    const uint64_t signBits = t.transB == ACLBLAS_OP_C ?
                                  (t.transA == ACLBLAS_OP_C ? 0xffaaffaaffaaffaaULL : 0xaa00aa00aa00aa00ULL) :
                                  (t.transA == ACLBLAS_OP_C ? 0x00aa00aa00aa00aaULL : 0x5500550055005500ULL);
    uint64_t amask[2] = {signBits, 0U};
    Muls(ao, ao, -1.0f, amask, static_cast<uint8_t>(2U * area / 64U), {1, 1, 8, 8});
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(ws[batch * SbCompactWorkspaceStride(t.m) + sub * 2U * area], ao, 2U * area);
    if (rounds % 2U == 0U) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    } else {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SbGather(bo, input[inputSize], offsets[area], 0U, area);
    if (nextBatch < t.batchCount) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(input[inputSize], bg[next], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

template <uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
__aicore__ inline void SbRunPaddedCube(Arg0& c, Arg1& workspace, Arg2& t, Arg3& cores, Arg4& atomicOutput)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t ownership = 1U;
    InitSocState();
    SetHF32Mode(HF32Mode::DISABLE);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    if (atomicOutput) {
        SetAtomicAdd<float>();
    }
    uint32_t rounds = 0U;
    for (uint32_t batch = GetBlockIdx() * ownership; batch < t.batchCount;
         batch += batch % ownership + 1U < ownership ? 1U : (cores - 1U) * ownership + 1U) {
        CrossCoreWaitFlag<2>(8U);
        auto* ws = reinterpret_cast<__gm__ float*>(workspace) + batch * SbCompactWorkspaceStride(t.m);
        CgemmSbAugmentedMmad<2U * width, width, 2U * width> mm;
        mm.Init(
            ws + 4U * area, ws, reinterpret_cast<__gm__ float*>(c) + static_cast<uint64_t>(batch) * t.m * t.m * 2U, t.m,
            2U * t.m, (2U * t.m + 15U) / 16U * 16U, 2U * width, 2U * t.m, 2U * width);
        mm.PreloadNz(1U, static_cast<uint32_t>(SbCompactWorkspaceStride(t.m)));
        mm.Process(0U, 1U, true, true);
        ++rounds;
        if (rounds % 4U == 0U ||
            batch + (batch % ownership + 1U < ownership ? 1U : (cores - 1U) * ownership + 1U) >= t.batchCount) {
            CrossCoreSetFlag<2, PIPE_FIX>(9U);
        }
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    if (atomicOutput) {
        SetAtomicNone();
    }
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10>
__aicore__ inline void SbRunPaddedBatches(
    Arg0& t, Arg1& cores, Arg2& xb, Arg3& input, Arg4& output, Arg5& sub, Arg6& ag, Arg7& bg, Arg8& ws, Arg9& cp,
    Arg10& pad)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t ownership = 1U;
    uint32_t rounds = 0U;
    for (uint32_t batch = GetBlockIdx() / 2U * ownership; batch < t.batchCount;
         batch += batch % ownership + 1U < ownership ? 1U : (cores - 1U) * ownership + 1U) {
        if (rounds != 0U && rounds % 4U == 0U) {
            CrossCoreWaitFlag<2>(9U);
        }
        const uint32_t nextBatch = batch + (batch % ownership + 1U < ownership ? 1U : (cores - 1U) * ownership + 1U);
        const uint64_t next = static_cast<uint64_t>(nextBatch) * t.m * t.m * 2U;
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto offsets = xb.template Get<uint32_t>();
        if (rounds % 2U == 0U) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        } else {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
        auto ao = output[rounds % 2U * 2U * area];
        auto bo = output[4U * area];
        SbPackPaddedOperands<width>(
            t, input, sub, ag, bg, ws, cp, pad, rounds, batch, nextBatch, next, offsets, ao, bo);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(ws[batch * SbCompactWorkspaceStride(t.m) + 4U * area + sub * area], bo, area);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        CrossCoreSetFlag<2, PIPE_MTE3>(8U);
        ++rounds;
    }
    SbDrainPackedOutput<true>(rounds);
}

template <
    uint32_t width, typename Arg0, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5,
    typename Arg6, typename Arg7, typename Arg8, typename Arg9, typename Arg10, typename Arg11>
__aicore__ inline void SbPreparePaddedOutput(
    Arg0& workspace, Arg1& t, Arg2& cores, Arg3& atomicOutput, Arg4& xb, Arg5& input, Arg6& output, Arg7& sub, Arg8& ag,
    Arg9& bg, Arg10& cg, Arg11& ws)
{
    constexpr uint32_t area = width * width;
    constexpr uint32_t inputPitch = 2U * width + 8U;
    constexpr uint32_t inputSize = width * inputPitch;
    constexpr uint32_t ownership = 1U;
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    if (atomicOutput) {
        SbFill(output, 0.0f, area);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        const uint64_t total = static_cast<uint64_t>(t.batchCount) * t.m * t.m * 2U;
        SbZeroOutput<area>(cg, output, total, cores);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
    DataCopyExtParams cp{t.m, t.m * 8U, 0U, (inputPitch - (2U * t.m + 7U) / 8U * 8U) / 8U, 0U};
    DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>((8U - 2U * t.m % 8U) % 8U), 0.0f};
    if (GetBlockIdx() / 2U * ownership < t.batchCount) {
        const uint64_t initial = static_cast<uint64_t>(GetBlockIdx() / 2U * ownership) * t.m * t.m * 2U;
        DataCopyPad(input, ag[initial], cp, pad);
        DataCopyPad(input[inputSize], bg[initial], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    SbRunPaddedBatches<width>(t, cores, xb, input, output, sub, ag, bg, ws, cp, pad);
}

template <uint32_t width>
__attribute__((aligned(kCgemmSbKernelAlignment))) __global__ __aicore__ void cgemm_sb_padded_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CgemmSbCompactTiling t, uint32_t cores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    constexpr uint32_t area = width * width;
    constexpr uint32_t inputPitch = 2U * width + 8U;
    constexpr uint32_t inputSize = width * inputPitch;

    const bool atomicOutput = t.m % 4U != 0U;
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> ib, ob, xb;
        pipe.InitBuffer(ib, 2U * inputSize * 4U);
        pipe.InitBuffer(ob, 5U * area * 4U);
        pipe.InitBuffer(xb, 2U * area * 4U);
        auto input = ib.Get<float>();
        auto output = ob.Get<float>();
        auto ix = xb.Get<int32_t>();
        const uint32_t sub = GetSubBlockIdx();
        SetMaskNorm();
        SbPaddedHalfNzIndex<width, true>(ix, t.m, t.transA, sub);
        SbPaddedHalfNzIndex<width, false>(ix[area], t.m, t.transB, sub);
        SbFill(input, 0.0f, inputSize);
        SbFill(input[inputSize], 0.0f, inputSize);
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<float> ag, bg, cg, ws;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
        SbPreparePaddedOutput<width>(workspace, t, cores, atomicOutput, xb, input, output, sub, ag, bg, cg, ws);
    }
    if ASCEND_IS_AIC {
        SbRunPaddedCube<width>(c, workspace, t, cores, atomicOutput);
    }
}
} // namespace cgemm_sb

bool cgemm_sb::CgemmSbDispatcher::SquareShape()
{
    return tiling.m >= 16U && tiling.m <= 192U && tiling.m == tiling.n && tiling.m == tiling.k &&
           tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f && tiling.betaReal == 0.0f && tiling.betaImag == 0.0f;
}

bool cgemm_sb::CgemmSbDispatcher::SquareLayout()
{
    return tiling.lda == tiling.m && tiling.ldb == tiling.m && tiling.ldc == tiling.m &&
           tiling.strideA == static_cast<int64_t>(tiling.m) * tiling.m && tiling.strideB == tiling.strideA &&
           tiling.strideC == tiling.strideA && reinterpret_cast<uintptr_t>(c) % 64U == 0U;
}

void cgemm_sb::CgemmSbDispatcher::LaunchLargeSquare()
{
    if (tiling.m <= 40U && tiling.batchCount <= aicCores * 16U) {
        if (tiling.m % 4U == 2U) {
            cgemm_sb::cgemm_sb_balanced_pair_launch<40U>(a, b, c, workspace, compact, aicCores, stream);
        } else {
            LaunchNzCopy();
        }
    } else if (tiling.m <= 40U) {
        LaunchNz(40U);
    } else if (tiling.m <= 48U) {
        LaunchNz(48U);
    } else if ((tiling.m > 64U && tiling.m <= 96U) || (tiling.m > 128U && tiling.m <= 144U)) {
        cgemm_sb::cgemm_sb_square_mixed_kernel<48U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else if (tiling.m > 96U && tiling.m <= 128U) {
        cgemm_sb::cgemm_sb_square_mixed_kernel<64U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else if (tiling.m > 96U) {
        cgemm_sb::cgemm_sb_square_mixed_kernel<32U, 64U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else if (tiling.m == 64U) {
        cgemm_sb::cgemm_sb_padded_kernel<64U><<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else {
        cgemm_sb::cgemm_sb_balanced_launch<64U>(a, b, c, workspace, compact, aicCores, stream);
    }
}

void cgemm_sb::CgemmSbDispatcher::LaunchSmallSquare()
{
    if (tiling.m > 16U && tiling.m <= 64U && tiling.transA == ACLBLAS_OP_N && tiling.transB != ACLBLAS_OP_N) {
        if (tiling.m <= 24U) {
            LaunchNz(24U);
        } else if (tiling.m <= 32U) {
            LaunchDirectNt(32U);
        } else {
            LaunchDirectNt(64U);
        }
    } else if (tiling.m == 16U) {
        cgemm_sb::cgemm_sb_square_mixed_kernel<16U>
            <<<aicCores, nullptr, stream>>>(a, b, c, workspace, compact, aicCores);
    } else if (tiling.m <= 24U) {
        LaunchNz(24U);
    } else if (tiling.m <= 32U && tiling.m % 4U == 0U) {
        cgemm_sb::cgemm_sb_balanced_pair_launch<32U>(a, b, c, workspace, compact, aicCores, stream);
    } else if (tiling.m <= 32U) {
        LaunchNz(32U);
    } else {
        LaunchLargeSquare();
    }
}

bool cgemm_sb::CgemmSbDispatcher::LaunchSquare()
{
    if (!SquareShape() || !SquareLayout()) {
        return false;
    }
    LaunchSmallSquare();
    return true;
}
