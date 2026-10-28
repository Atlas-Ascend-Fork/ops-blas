/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "chpr_kernel.h"

using namespace AscendC;

constexpr uint32_t ACLBLAS_UPPER_VALUE = 121;
constexpr uint32_t UB_FLOATS = (192U * 1024U) / static_cast<uint32_t>(sizeof(float));
// xLocal + ixLocal + apLocal
constexpr uint32_t BUF_COMPLEX = (UB_FLOATS / 6U) & ~31U;
constexpr uint32_t ELEM_PER_32B_BLOCK = 8U;
constexpr uint16_t PIPE_EVT = 0;

__aicore__ inline uint32_t AlignUpElems(uint32_t value)
{
    return (value + ELEM_PER_32B_BLOCK - 1U) & ~(ELEM_PER_32B_BLOCK - 1U);
}

template <bool UPLO_IS_UPPER>
__aicore__ inline uint64_t ChprColumnBase(uint32_t col, uint32_t n)
{
    if constexpr (UPLO_IS_UPPER) {
        return static_cast<uint64_t>(col) * (col + 1U) / 2U;
    } else {
        return static_cast<uint64_t>(col) * (2U * n - col + 1U) / 2U;
    }
}

__aicore__ inline uint64_t XPhysicalIndex(uint32_t n, uint32_t logical, int64_t incx)
{
    if (incx >= 0) {
        return static_cast<uint64_t>(static_cast<int64_t>(logical) * incx);
    }
    return static_cast<uint64_t>((static_cast<int64_t>(n) - 1 - static_cast<int64_t>(logical)) * (-incx));
}

class ChprKernel {
public:
    __aicore__ inline ChprKernel() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR ap, const ChprTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAos(
        LocalTensor<float> dst, GlobalTensor<float> src, uint64_t gmFloatIndex, uint32_t count, bool skipLastFloat);
    __aicore__ inline void CopyOutAos(
        GlobalTensor<float> dst, LocalTensor<float> src, uint64_t gmFloatIndex, uint32_t count);
    __aicore__ inline void CopyStrided(
        LocalTensor<float> dst, uint32_t rowStart, uint32_t count, uint32_t blockSize, uint32_t srcStride,
        uint32_t gmFloatOffset, bool leftPadOne);
    __aicore__ inline void CopyStridedGather(
        LocalTensor<float> copyDst, LocalTensor<float> gatherDst, uint32_t rowStart, uint32_t count, uint32_t blockSize,
        uint32_t srcStride, uint32_t gmFloatOffset, bool leftPadOne);
    __aicore__ inline void FillIx(LocalTensor<float> ix, LocalTensor<float> x, uint32_t count);
    __aicore__ inline void Gather8To2(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count);
    __aicore__ inline void ApplyEvenLaneNeg(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count);
    __aicore__ inline void SyncIxReady(bool upper);
    __aicore__ inline void LoadXIxUnitStride(uint32_t rowStart, uint32_t count, bool upper);
    __aicore__ inline void LoadXIxScaler(uint32_t rowStart, uint32_t count, bool upper);
    __aicore__ inline void LoadXIxStridedLe2048(uint32_t rowStart, uint32_t count, bool upper);
    __aicore__ inline void LoadXIxStridedLe4096(uint32_t rowStart, uint32_t count, bool upper);
    __aicore__ inline void LoadXIxStridedLe8192(uint32_t rowStart, uint32_t count, bool upper);
    __aicore__ inline void LoadXIx(uint32_t rowStart, uint32_t count, bool lowerPath);
    __aicore__ inline void ComputeChunk(LocalTensor<float> ap, uint32_t xOff, uint32_t chunk, float xrCol, float xiCol);
    __aicore__ inline void ProcessUpperCols(uint32_t begin, uint32_t end);
    __aicore__ inline void ProcessLowerCols(uint32_t begin, uint32_t end);

    LocalMemAllocator<Hardware::UB> ubAllocator;
    LocalTensor<float> ubLocal;
    LocalTensor<float> xLocal;
    LocalTensor<float> ixLocal;
    LocalTensor<float> apLocal;

    GlobalTensor<float> xGM;
    GlobalTensor<float> apGM;

    uint32_t n = 0;
    uint32_t useCoreNum = 1;
    uint32_t uplo = ACLBLAS_UPPER_VALUE;
    int64_t incx = 1;
    float alpha = 0.0f;
    uint32_t blockIdx = 0;
    uint32_t frontColStart = 0;
    uint32_t frontend = 0;
};

__aicore__ inline void ChprKernel::Init(GM_ADDR x, GM_ADDR ap, const ChprTilingData& tiling)
{
    n = tiling.n;
    useCoreNum = tiling.useCoreNum;
    uplo = tiling.uplo;
    incx = tiling.incx;
    alpha = tiling.alpha;
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    blockIdx = static_cast<uint32_t>(GetBlockIdx());
    frontColStart = 0;
    frontend = 0;
    if (blockIdx < useCoreNum) {
        const uint32_t baseCols = n / useCoreNum;
        const uint32_t extraCols = n % useCoreNum;
        const uint32_t cols = baseCols + ((blockIdx < extraCols) ? 1U : 0U);
        const uint32_t extraOffset = (blockIdx < extraCols) ? blockIdx : extraCols;
        frontColStart = blockIdx * baseCols + extraOffset;
        frontend = frontColStart + cols;
    }

    uint64_t xFloats = 0;
    if (n > 0) {
        const uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : static_cast<uint64_t>(-incx);
        xFloats = (static_cast<uint64_t>(n - 1U) * absIncx + 1U) * 2U;
    }
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), xFloats);
    apGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ap), static_cast<uint64_t>(n) * (n + 1U));

    ubLocal = ubAllocator.Alloc<TPosition::VECIN, float>(UB_FLOATS);
    ixLocal = ubLocal[0U];
    xLocal = ubLocal[16384U];
    apLocal = ubLocal[32768U];
}

__aicore__ inline void ChprKernel::CopyInAos(
    LocalTensor<float> dst, GlobalTensor<float> src, uint64_t gmFloatIndex, uint32_t count, bool skipLastFloat)
{
    const uint32_t floatCount = skipLastFloat ? (count * 2U - 1U) : (count * 2U);
    DataCopyExtParams params{1, static_cast<uint32_t>(floatCount * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{skipLastFloat, 0, static_cast<uint8_t>(skipLastFloat ? 1U : 0U), 0.0f};
    DataCopyPad(dst, src[gmFloatIndex], params, pad);
}

__aicore__ inline void ChprKernel::CopyOutAos(
    GlobalTensor<float> dst, LocalTensor<float> src, uint64_t gmFloatIndex, uint32_t count)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(count * 2U * sizeof(float)), 0, 0, 0};
    DataCopyPad(dst[gmFloatIndex], src, params);
}

__aicore__ inline void ChprKernel::CopyStrided(
    LocalTensor<float> dst, uint32_t rowStart, uint32_t count, uint32_t blockSize, uint32_t srcStride,
    uint32_t gmFloatOffset, bool leftPadOne)
{
    constexpr uint32_t MAX_BURST = 255U;
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    const DataCopyPadExtParams<float> leftPadOneParam{true, 1, 0, 0.0f};
    const uint32_t blockBytes = blockSize * static_cast<uint32_t>(sizeof(float));
    const uint32_t srcStrideBytes = srcStride * static_cast<uint32_t>(sizeof(float));
    uint32_t copied = 0U;
    while (copied < count) {
        const uint32_t step = (copied + MAX_BURST <= count) ? MAX_BURST : (count - copied);
        const uint64_t aosBase = XPhysicalIndex(n, rowStart + copied, incx) * 2U;
        const DataCopyExtParams copyParams{static_cast<uint16_t>(step), blockBytes, srcStrideBytes, 0, 0};
        DataCopyPad(dst[copied * 8U], xGM[aosBase + gmFloatOffset], copyParams, leftPadOne ? leftPadOneParam : noPad);
        copied += step;
    }
}

__aicore__ inline void ChprKernel::CopyStridedGather(
    LocalTensor<float> copyDst, LocalTensor<float> gatherDst, uint32_t rowStart, uint32_t count, uint32_t blockSize,
    uint32_t srcStride, uint32_t gmFloatOffset, bool leftPadOne)
{
    CopyStrided(copyDst, rowStart, count, blockSize, srcStride, gmFloatOffset, leftPadOne);
    SetFlag<HardEvent::MTE2_V>(PIPE_EVT);
    WaitFlag<HardEvent::MTE2_V>(PIPE_EVT);
    Gather8To2(gatherDst, copyDst, count);
}

__aicore__ inline void ChprKernel::FillIx(LocalTensor<float> ix, LocalTensor<float> x, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        const float xr = x.GetValue(i * 2U);
        const float xi = x.GetValue(i * 2U + 1U);
        ix.SetValue(i * 2U, -xi);
        ix.SetValue(i * 2U + 1U, xr);
    }
}

__aicore__ inline void ChprKernel::Gather8To2(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count)
{
    uint64_t reservedCount = 0U;
    GatherMask<float>(dst, src, 7U, true, 2U, {1U, static_cast<uint16_t>(count), 1U, 0U}, reservedCount);
}

__aicore__ inline void ChprKernel::ApplyEvenLaneNeg(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count)
{
    const uint32_t total = count * 2U;
    const uint32_t fullRep = total >= 64U * 256U ? 255U : total / 64U;
    const uint32_t tail = total - fullRep * 64U;
    const UnaryRepeatParams unaryParams;
    constexpr uint64_t EVEN_LANE_MASK = 0x5555555555555555ULL;
    uint64_t evenMask[2] = {EVEN_LANE_MASK, 0ULL};

    if (fullRep > 0U) {
        Muls(dst, src, -1.0f, evenMask, static_cast<uint8_t>(fullRep), unaryParams);
    }
    if (tail > 0U) {
        uint64_t tailEven = 0ULL;
        for (uint32_t bit = 0U; bit < tail; bit += 2U) {
            tailEven |= (1ULL << bit);
        }
        uint64_t tailEvenMask[2] = {tailEven, 0ULL};
        Muls(dst[fullRep * 64U], src[fullRep * 64U], -1.0f, tailEvenMask, 1, unaryParams);
    }
}

__aicore__ inline void ChprKernel::SyncIxReady(bool upper)
{
    if (upper) {
        SetFlag<HardEvent::V_S>(PIPE_EVT);
        WaitFlag<HardEvent::V_S>(PIPE_EVT);
    } else {
        SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
        WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);
    }
}

__aicore__ inline void ChprKernel::LoadXIxUnitStride(uint32_t rowStart, uint32_t count, bool upper)
{
    CopyInAos(xLocal, xGM, static_cast<uint64_t>(rowStart) * 2U, count, false);
    const uint32_t total = count * 2U;

    // Build ixLocal as [0, xr0, xi0, xr1, ...].
    DataCopyExtParams ixParams{1, static_cast<uint32_t>((total - 1U) * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> ixPad{true, 1, 0, 0.0f};
    DataCopyPad(ixLocal, xGM[static_cast<uint64_t>(rowStart) * 2U], ixParams, ixPad);

    // apLocal temporarily stores x shifted by +1 float from GM.
    CopyInAos(apLocal, xGM, static_cast<uint64_t>(rowStart) * 2U + 1U, count, true);

    SetFlag<HardEvent::MTE2_V>(PIPE_EVT);
    WaitFlag<HardEvent::MTE2_V>(PIPE_EVT);

    // Construct ixLocal as [-xi0, xr0, -xi1, xr1, ...] in place.
    ApplyEvenLaneNeg(ixLocal, apLocal, count);
    SyncIxReady(upper);
}

__aicore__ inline void ChprKernel::LoadXIxScaler(uint32_t rowStart, uint32_t count, bool upper)
{
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t xIdx = XPhysicalIndex(n, rowStart + i, incx) * 2U;
        xLocal.SetValue(i * 2U, xGM.GetValue(xIdx));
        xLocal.SetValue(i * 2U + 1U, xGM.GetValue(xIdx + 1U));
    }
    FillIx(ixLocal, xLocal, count);
    if (upper) {
        SetFlag<HardEvent::S_V>(PIPE_EVT);
        WaitFlag<HardEvent::S_V>(PIPE_EVT);
    } else {
        SetFlag<HardEvent::S_MTE2>(PIPE_EVT);
        WaitFlag<HardEvent::S_MTE2>(PIPE_EVT);
    }
}

__aicore__ inline void ChprKernel::LoadXIxStridedLe2048(uint32_t rowStart, uint32_t count, bool upper)
{
    constexpr uint32_t MAX_BURST = 255U;
    const uint32_t srcStrideBytes = static_cast<uint32_t>((static_cast<uint64_t>(incx) - 1U) * 2U * sizeof(float));
    const uint32_t scalarSrcStrideBytes = srcStrideBytes + static_cast<uint32_t>(sizeof(float));
    const DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    const DataCopyPadExtParams<float> leftPadOne{true, 1, 0, 0.0f};

    uint32_t copied = 0U;
    while (copied < count) {
        const uint32_t step = (copied + MAX_BURST <= count) ? MAX_BURST : (count - copied);
        const uint64_t aosBase = XPhysicalIndex(n, rowStart + copied, incx) * 2U;

        const DataCopyExtParams copyX{
            static_cast<uint16_t>(step), static_cast<uint32_t>(2U * sizeof(float)), srcStrideBytes, 0, 0};
        DataCopyPad(xLocal[copied * 8U], xGM[aosBase], copyX, noPad);
        const DataCopyExtParams copyReal{
            static_cast<uint16_t>(step), static_cast<uint32_t>(sizeof(float)), scalarSrcStrideBytes, 0, 0};
        DataCopyPad(ixLocal[copied * 8U], xGM[aosBase], copyReal, leftPadOne);
        const DataCopyExtParams copyImag{
            static_cast<uint16_t>(step), static_cast<uint32_t>(sizeof(float)), scalarSrcStrideBytes, 0, 0};
        DataCopyPad(apLocal[copied * 8U], xGM[aosBase + 1U], copyImag, noPad);

        copied += step;
    }

    SetFlag<HardEvent::MTE2_V>(PIPE_EVT);
    WaitFlag<HardEvent::MTE2_V>(PIPE_EVT);

    // Copy in and gather x in its own buffer
    uint64_t reservedCount = 0U;
    GatherMask<float>(xLocal, xLocal, 7U, true, 2U, {1U, static_cast<uint16_t>(count), 1U, 0U}, reservedCount);

    // Construct ix in its own buffer, using apLocal as temporary buffer to hold x shifted by +1 float from GM.
    reservedCount = 0U;
    GatherMask<float>(ixLocal, ixLocal, 7U, true, 2U, {1U, static_cast<uint16_t>(count), 1U, 0U}, reservedCount);
    reservedCount = 0U;
    GatherMask<float>(apLocal, apLocal, 7U, true, 2U, {1U, static_cast<uint16_t>(count), 1U, 0U}, reservedCount);
    ApplyEvenLaneNeg(ixLocal, apLocal, count);

    SyncIxReady(upper);
}

__aicore__ inline void ChprKernel::LoadXIxStridedLe4096(uint32_t rowStart, uint32_t count, bool upper)
{
    const uint32_t srcStride = static_cast<uint32_t>((static_cast<uint64_t>(incx) - 1U) * 2U);
    const uint32_t scalarSrcStride = srcStride + 1U;

    // Construct ix at ub address [0, 8192), using [8192, 49152) as temporary buffer.
    CopyStridedGather(xLocal, ixLocal, rowStart, count, 1U, scalarSrcStride, 0U, true);
    SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
    WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);
    CopyStridedGather(xLocal, ixLocal[8192U], rowStart, count, 1U, scalarSrcStride, 1U, false);
    SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
    WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);
    ApplyEvenLaneNeg(ixLocal, ixLocal[8192U], count);

    // Copy x from GM to ub at [16384, 24576), using [24576, 49152) as temporary buffer.
    CopyStridedGather(xLocal, xLocal, rowStart, count, 2U, srcStride, 0U, false);

    SyncIxReady(upper);
}

__aicore__ inline void ChprKernel::LoadXIxStridedLe8192(uint32_t rowStart, uint32_t count, bool upper)
{
    const uint32_t srcStride = static_cast<uint32_t>((static_cast<uint64_t>(incx) - 1U) * 2U);
    const uint32_t scalarSrcStride = srcStride + 1U;
    const uint32_t firstCount = 4096U;
    const uint32_t remainCount = count - firstCount;
    const uint32_t xFirstCount = 4096U;
    uint32_t xRemain = count - xFirstCount;
    const uint32_t xSecondCount = (xRemain <= 3072U) ? xRemain : 3072U;
    xRemain -= xSecondCount;
    const uint32_t xThirdCount = xRemain;

    // Construct ix at ub address [0, 16384), using [16384, 49152) as temporary buffer.
    CopyStridedGather(xLocal, ixLocal, rowStart, firstCount, 1U, scalarSrcStride, 0U, true);
    SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
    WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);
    CopyStridedGather(xLocal, ixLocal[8192U], rowStart + firstCount, remainCount, 1U, scalarSrcStride, 0U, true);
    SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
    WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);
    CopyStridedGather(xLocal, xLocal, rowStart, firstCount, 1U, scalarSrcStride, 1U, false);
    ApplyEvenLaneNeg(ixLocal, xLocal, firstCount); // Step 1: ixLocal[0:8192]
    SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
    WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);
    CopyStridedGather(xLocal, xLocal, rowStart + firstCount, remainCount, 1U, scalarSrcStride, 1U, false);
    ApplyEvenLaneNeg(ixLocal[8192U], xLocal, remainCount); // Step 2: ixLocal[8192:16384]

    SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
    WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);

    // Copy x from GM to ub at [16384, 32768), using [32768, 49152) as temporary buffer.
    CopyStridedGather(xLocal, xLocal, rowStart, xFirstCount, 2U, srcStride, 0U, false); // Step 1: xLocal[0:8192]
    SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
    WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);
    CopyStridedGather(
        xLocal[8192U], xLocal[8192U], rowStart + xFirstCount, xSecondCount, 2U, srcStride, 0U,
        false); // Step 2: xLocal[8192:14336]
    if (xThirdCount > 0U) {
        SetFlag<HardEvent::V_MTE2>(PIPE_EVT);
        WaitFlag<HardEvent::V_MTE2>(PIPE_EVT);
        CopyStridedGather(
            xLocal[14336U], xLocal[14336U], rowStart + xFirstCount + xSecondCount, xThirdCount, 2U, srcStride, 0U,
            false); // Step 3: xLocal[14336:16384]
    }

    SyncIxReady(upper);
}

__aicore__ inline void ChprKernel::LoadXIx(uint32_t rowStart, uint32_t count, bool upper)
{
    if (incx == 1) {
        LoadXIxUnitStride(rowStart, count, upper);
    } else if (incx < 1) {
        LoadXIxScaler(rowStart, count, upper);
    } else {
        const uint64_t srcStrideBytes64 = (static_cast<uint64_t>(incx) - 1U) * 2U * sizeof(float);
        if (srcStrideBytes64 > static_cast<uint64_t>(UINT32_MAX)) {
            LoadXIxScaler(rowStart, count, upper);
        } else if (count <= 2048U) {
            LoadXIxStridedLe2048(rowStart, count, upper);
        } else if (count <= 4096U) {
            LoadXIxStridedLe4096(rowStart, count, upper);
        } else {
            LoadXIxStridedLe8192(rowStart, count, upper);
        }
    }
}

__aicore__ inline void ChprKernel::ComputeChunk(
    LocalTensor<float> ap, uint32_t xOff, uint32_t chunk, float xrCol, float xiCol)
{
    Axpy(ap, xLocal[xOff], alpha * xrCol, chunk * 2U);
    PipeBarrier<PIPE_V>();
    Axpy(ap, ixLocal[xOff], -alpha * xiCol, chunk * 2U);
}

__aicore__ inline void ChprKernel::ProcessUpperCols(uint32_t begin, uint32_t end)
{
    const uint32_t rowTileBegin = 0U;
    const uint32_t rowTileEndLimit = end;
    const uint32_t rowTileStep = 8192U;
    for (uint32_t rowTileStart = rowTileBegin; rowTileStart < rowTileEndLimit; rowTileStart += rowTileStep) {
        const uint32_t rowTileCount =
            (rowTileStart + rowTileStep <= rowTileEndLimit) ? rowTileStep : (rowTileEndLimit - rowTileStart);
        const uint32_t rowTileEnd = rowTileStart + rowTileCount;
        LoadXIx(rowTileStart, rowTileCount, true);

        for (uint32_t col = begin; col < end; ++col) {
            const uint32_t rowStart = rowTileStart;
            const uint32_t rowEnd = ((col + 1U) < rowTileEnd) ? (col + 1U) : rowTileEnd;
            if (rowStart >= rowEnd) {
                continue;
            }

            const uint32_t chunk = rowEnd - rowStart;
            const uint32_t xOff = 0U;
            const uint64_t colBase = ChprColumnBase<true>(col, n);
            const uint64_t apOff = colBase + rowStart;
            float xrCol = 0.0f;
            float xiCol = 0.0f;
            if ((col >= rowTileStart) && (col < rowTileEnd)) {
                const uint32_t colOff = (col - rowTileStart) * 2U;
                xrCol = xLocal.GetValue(colOff);
                xiCol = xLocal.GetValue(colOff + 1U);
            } else {
                const uint64_t xColIdx = XPhysicalIndex(n, col, incx) * 2U;
                xrCol = xGM.GetValue(xColIdx);
                xiCol = xGM.GetValue(xColIdx + 1U);
            }
            const bool skipDiagFloat = col < rowTileEnd;

            CopyInAos(apLocal, apGM, apOff * 2U, chunk, skipDiagFloat);
            SetFlag<HardEvent::MTE2_V>(PIPE_EVT);
            WaitFlag<HardEvent::MTE2_V>(PIPE_EVT);
            ComputeChunk(apLocal, xOff, chunk, xrCol, xiCol);

            SetFlag<HardEvent::V_MTE3>(PIPE_EVT);
            WaitFlag<HardEvent::V_MTE3>(PIPE_EVT);
            CopyOutAos(apGM, apLocal, apOff * 2U, chunk);
            // apLocal 会在下一列（或下一个 rowTile）被 CopyInAos 重新写入，
            // 必须先等本列 CopyOutAos（MTE3 读）完成，否则 MTE2 写与 MTE3 读竞争。
            SetFlag<HardEvent::MTE3_MTE2>(PIPE_EVT);
            WaitFlag<HardEvent::MTE3_MTE2>(PIPE_EVT);
        }
    }
}

__aicore__ inline void ChprKernel::ProcessLowerCols(uint32_t begin, uint32_t end)
{
    const uint32_t rowTileBegin = begin;
    const uint32_t rowTileEndLimit = n;
    const uint32_t rowTileStep = 8192U;
    for (uint32_t rowTileStart = rowTileBegin; rowTileStart < rowTileEndLimit; rowTileStart += rowTileStep) {
        const uint32_t rowTileCount =
            (rowTileStart + rowTileStep <= rowTileEndLimit) ? rowTileStep : (rowTileEndLimit - rowTileStart);
        const uint32_t rowTileEnd = rowTileStart + rowTileCount;

        for (uint32_t col = begin; col < end; ++col) {
            const uint32_t rowStart = (col > rowTileStart) ? col : rowTileStart;
            const uint32_t rowEnd = rowTileEnd;
            if (rowStart >= rowEnd) {
                continue;
            }

            const uint32_t chunk = rowEnd - rowStart;
            const uint32_t xOff = 0U;
            const uint64_t colBase = ChprColumnBase<false>(col, n);
            const uint64_t apOff = colBase + (rowStart - col);
            const uint64_t xColIdx = XPhysicalIndex(n, col, incx) * 2U;
            const float xrCol = xGM.GetValue(xColIdx);
            const float xiCol = xGM.GetValue(xColIdx + 1U);

            LoadXIx(rowStart, chunk, false);

            CopyInAos(apLocal, apGM, apOff * 2U, chunk, false);
            SetFlag<HardEvent::MTE2_V>(PIPE_EVT);
            WaitFlag<HardEvent::MTE2_V>(PIPE_EVT);
            ComputeChunk(apLocal, xOff, chunk, xrCol, xiCol);

            const bool needLowerDiagFix = rowStart == col;
            if (needLowerDiagFix) {
                SetFlag<HardEvent::V_S>(PIPE_EVT);
                WaitFlag<HardEvent::V_S>(PIPE_EVT);
                apLocal.SetValue(1U, 0.0f);
                SetFlag<HardEvent::S_MTE3>(PIPE_EVT);
                WaitFlag<HardEvent::S_MTE3>(PIPE_EVT);
            } else {
                SetFlag<HardEvent::V_MTE3>(PIPE_EVT);
                WaitFlag<HardEvent::V_MTE3>(PIPE_EVT);
            }
            CopyOutAos(apGM, apLocal, apOff * 2U, chunk);
            // 同 upper 路径：防止下一列 CopyInAos(MTE2 写)与本列 CopyOutAos(MTE3 读)竞争 apLocal。
            SetFlag<HardEvent::MTE3_MTE2>(PIPE_EVT);
            WaitFlag<HardEvent::MTE3_MTE2>(PIPE_EVT);
        }
    }
}

__aicore__ inline void ChprKernel::Process()
{
    if (n == 0 || blockIdx >= useCoreNum || frontColStart >= frontend) {
        return;
    }
    if (uplo == ACLBLAS_UPPER_VALUE) {
        ProcessUpperCols(frontColStart, frontend);
    } else {
        ProcessLowerCols(frontColStart, frontend);
    }
}

extern "C" __global__ __aicore__ void chpr_kernel(GM_ADDR x, GM_ADDR ap, const ChprTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ChprKernel op;
    op.Init(x, ap, tiling);
    op.Process();
}

void chpr_kernel_do(GM_ADDR x, GM_ADDR ap, const ChprTilingData& tiling, uint32_t numBlocks, void* stream)
{
    chpr_kernel<<<numBlocks, nullptr, stream>>>(x, ap, tiling);
}
