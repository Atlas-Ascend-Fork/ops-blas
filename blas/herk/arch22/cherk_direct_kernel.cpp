/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_utils.h"
#include "common/compute/simd.h"
#define CHERK_ARCH22_DEVICE_IMPL
#include "cherk_kernel.h"

namespace {
constexpr uint32_t DIRECT_MIX_FLAG_COUNT = 5U;
constexpr uint32_t DIRECT_MIX_ACK_BASE = DIRECT_MIX_FLAG_COUNT;
constexpr uint32_t PACK_COMPLEX = 8192U;
__aicore__ inline CherkCopyEvents GetCherkCopyEvents()
{
    return {
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V))};
}

__aicore__ inline void LoadCherkMetadataOffsets(
    AscendC::LocalTensor<uint32_t>& offsets, GM_ADDR metadataGm, uint32_t offset, uint32_t count)
{
    AscendC::GlobalTensor<uint32_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
    AscendC::DataCopy(offsets, metadata[offset], count);
    const event_t metadataToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
}

__aicore__ inline void InitCherkFloatBuffers(
    AscendC::GlobalTensor<float>& input, AscendC::GlobalTensor<float>& workspace, GM_ADDR inputGm, GM_ADDR workspaceGm,
    uint64_t inputCount, uint64_t workspaceCount)
{
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputGm), inputCount);
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm), workspaceCount);
}

__aicore__ inline uint32_t CherkNormalizeBlockCount(uint32_t blocks) { return blocks == 0U ? 1U : blocks; }

__aicore__ inline uint64_t WorkspaceCount(const CherkArch22TilingData& tiling)
{
    return CherkArch22WorkspaceFloats(tiling.cubeN, tiling.cubeK);
}

__aicore__ inline void FillCherkPackNInput(
    AscendC::LocalTensor<float> aos, AscendC::GlobalTensor<float>& a, const CherkArch22TilingData& tiling,
    uint32_t packedRow, uint32_t offset, uint32_t chunk, uint32_t valid)
{
    if (valid != chunk) {
        AscendC::Duplicate(aos, 0.0f, chunk * 2U);
        const event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
    }
    if (valid != 0U) {
        const uint32_t blockBytes = valid * 2U * sizeof(float);
        const uint32_t alignedBytes = (blockBytes + 31U) & ~31U;
        const AscendC::DataCopyExtParams copy{1U, blockBytes, 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{
            alignedBytes != blockBytes, 0U, static_cast<uint8_t>((alignedBytes - blockBytes) / sizeof(float)), 0.0f};
        const uint64_t sourceOffset = (static_cast<uint64_t>(packedRow) * tiling.lda + offset) * 2ULL;
        AscendC::DataCopyPad(aos, a[sourceOffset], copy, pad);
    }
}

__aicore__ inline AscendC::LocalTensor<float> LoadCherkPackNInput(
    AscendC::TQue<AscendC::TPosition::VECIN, 1>& inputQueue, AscendC::GlobalTensor<float>& a,
    const CherkArch22TilingData& tiling, uint32_t packedRow, uint32_t offset, uint32_t chunk, uint32_t valid)
{
    AscendC::LocalTensor<float> aos = inputQueue.AllocTensor<float>();
    FillCherkPackNInput(aos, a, tiling, packedRow, offset, chunk, valid);
    return aos;
}

} // namespace

namespace {
struct CherkDirectPackCContext {
    AscendC::LocalTensor<float>& aos;
    AscendC::LocalTensor<float>& real;
    AscendC::LocalTensor<float>& imag;
    AscendC::LocalTensor<float>& transReal;
    AscendC::LocalTensor<float>& transNegImag;
    AscendC::LocalTensor<float>& transImag;
    AscendC::LocalTensor<uint32_t>& offsets;
    AscendC::GlobalTensor<float>& a;
    AscendC::GlobalTensor<float>& workspace;
    const CherkArch22TilingData& tiling;
    uint64_t realBase;
    uint64_t imagBase;
    uint64_t negImagBase;
    event_t vToMte2;
    event_t mte2ToV;
    event_t vToMte3;
    event_t mte3ToV;
};

constexpr uint32_t DIRECT_PACK_COMPLEX = 128U;

struct DirectPackNContext {
    AscendC::TQue<AscendC::TPosition::VECIN, 1>& inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1>& outputQueue;
    AscendC::GlobalTensor<float>& a;
    AscendC::GlobalTensor<float>& workspace;
    const CherkArch22TilingData& tiling;
    uint64_t realBase;
    uint64_t imagBase;
    uint64_t negImagBase;
};

__aicore__ inline void ProcessDirectPackNChunk(DirectPackNContext& ctx, uint32_t packedRow, uint32_t offset)
{
    const uint32_t chunk = Min<uint32_t>(DIRECT_PACK_COMPLEX, ctx.tiling.cubeN - offset);
    const uint32_t valid =
        packedRow < ctx.tiling.k && offset < ctx.tiling.n ? Min<uint32_t>(chunk, ctx.tiling.n - offset) : 0U;
    AscendC::LocalTensor<float> aos = ctx.inputQueue.AllocTensor<float>();
    FillCherkPackNInput(aos, ctx.a, ctx.tiling, packedRow, offset, chunk, valid);
    ctx.inputQueue.EnQue(aos);
    AscendC::LocalTensor<float> source = ctx.inputQueue.DeQue<float>();
    AscendC::LocalTensor<float> packed = ctx.outputQueue.AllocTensor<float>();
    AscendC::LocalTensor<float> real = packed;
    AscendC::LocalTensor<float> imag = packed[DIRECT_PACK_COMPLEX];
    AscendC::LocalTensor<float> negImag = packed[DIRECT_PACK_COMPLEX * 2U];
    const uint16_t repeats = static_cast<uint16_t>((chunk * 2U) / 64U);
    uint64_t reserved = 0ULL;
    AscendC::GatherMask<float>(real, source, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(imag, source, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(negImag, imag, -1.0f, chunk);
    ctx.outputQueue.EnQue(packed);
    ctx.inputQueue.FreeTensor(source);
    AscendC::LocalTensor<float> output = ctx.outputQueue.DeQue<float>();
    const uint64_t outputOffset = static_cast<uint64_t>(packedRow) * ctx.tiling.cubeN + offset;
    AscendC::DataCopy(ctx.workspace[ctx.realBase + outputOffset], output, chunk);
    AscendC::DataCopy(ctx.workspace[ctx.imagBase + outputOffset], output[DIRECT_PACK_COMPLEX], chunk);
    AscendC::DataCopy(ctx.workspace[ctx.negImagBase + outputOffset], output[DIRECT_PACK_COMPLEX * 2U], chunk);
    ctx.outputQueue.FreeTensor(output);
}
} // namespace

__aicore__ inline void CherkArch22DirectPackNImpl(
    GM_ADDR aGm, GM_ADDR workspaceGm, const CherkArch22TilingData& tiling, uint32_t blockIdx, uint32_t blockNum,
    AscendC::TPipe& pipe)
{
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    pipe.InitBuffer(inputQueue, 1, DIRECT_PACK_COMPLEX * 2U * sizeof(float));
    pipe.InitBuffer(outputQueue, 1, DIRECT_PACK_COMPLEX * 3U * sizeof(float));

    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> workspace;
    InitCherkFloatBuffers(
        a, workspace, aGm, workspaceGm, static_cast<uint64_t>(tiling.lda) * tiling.k * 2ULL,
        CherkArch22DirectWorkspaceFloats(tiling.cubeN, tiling.cubeK));

    const uint64_t realBase = CherkArch22DirectOffsetRealInput(tiling.cubeN, tiling.cubeK);
    const uint64_t imagBase = CherkArch22DirectOffsetImagInput(tiling.cubeN, tiling.cubeK);
    const uint64_t negImagBase = CherkArch22DirectOffsetNegImagInput(tiling.cubeN, tiling.cubeK);
    blockNum = CherkNormalizeBlockCount(blockNum);

    DirectPackNContext context{inputQueue, outputQueue, a, workspace, tiling, realBase, imagBase, negImagBase};
    for (uint32_t packedRow = blockIdx; packedRow < tiling.cubeK; packedRow += blockNum) {
        for (uint32_t offset = 0U; offset < tiling.cubeN; offset += DIRECT_PACK_COMPLEX) {
            ProcessDirectPackNChunk(context, packedRow, offset);
        }
    }
}

namespace {
constexpr uint32_t DIRECT_PACK_C_ROWS = 32U;
constexpr uint32_t DIRECT_PACK_C_COLS = 64U;
constexpr uint32_t DIRECT_PACK_C_ELEMENTS = DIRECT_PACK_C_ROWS * DIRECT_PACK_C_COLS;

__aicore__ inline void LoadDirectPackCTile(CherkDirectPackCContext& ctx, uint32_t rowBase, uint32_t colBase)
{
    const uint32_t validRows = rowBase < ctx.tiling.k ? Min<uint32_t>(DIRECT_PACK_C_ROWS, ctx.tiling.k - rowBase) : 0U;
    const uint32_t validCols = colBase < ctx.tiling.n ? Min<uint32_t>(DIRECT_PACK_C_COLS, ctx.tiling.n - colBase) : 0U;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::Duplicate(ctx.aos, 0.0f, DIRECT_PACK_C_ELEMENTS * 2U);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    if (validRows != 0U && validCols != 0U) {
        const uint32_t blockBytes = validRows * 2U * sizeof(float);
        const uint32_t alignedBytes = (blockBytes + 31U) & ~31U;
        const AscendC::DataCopyExtParams inputCopy{
            static_cast<uint16_t>(validCols), blockBytes,
            static_cast<uint32_t>((ctx.tiling.lda - validRows) * 2U * sizeof(float)),
            static_cast<uint32_t>((DIRECT_PACK_C_ROWS * 2U * sizeof(float) - alignedBytes) / 32U), 0U};
        const AscendC::DataCopyPadExtParams<float> inputPad{
            alignedBytes != blockBytes, 0U, static_cast<uint8_t>((alignedBytes - blockBytes) / sizeof(float)), 0.0f};
        const uint64_t inputOffset = (static_cast<uint64_t>(colBase) * ctx.tiling.lda + rowBase) * 2ULL;
        AscendC::DataCopyPad(ctx.aos, ctx.a[inputOffset], inputCopy, inputPad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV);
}

__aicore__ inline void ProcessDirectPackCTile(CherkDirectPackCContext& ctx, uint32_t tile, uint32_t colTiles)
{
    colTiles = colTiles == 0U ? 1U : colTiles;
    const uint32_t rowBase = (tile / colTiles) * DIRECT_PACK_C_ROWS;
    const uint32_t colBase = (tile % colTiles) * DIRECT_PACK_C_COLS;
    LoadDirectPackCTile(ctx, rowBase, colBase);
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = DIRECT_PACK_C_ELEMENTS * 2U / 64U;
    AscendC::GatherMask<float>(ctx.real, ctx.aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(ctx.imag, ctx.aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
    AscendC::Gather(ctx.transReal, ctx.real, ctx.offsets, 0U, DIRECT_PACK_C_ELEMENTS);
    AscendC::Gather(ctx.transImag, ctx.imag, ctx.offsets, 0U, DIRECT_PACK_C_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ctx.transNegImag, ctx.transImag, -1.0f, DIRECT_PACK_C_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3);
    const AscendC::DataCopyExtParams outputCopy{
        DIRECT_PACK_C_ROWS, DIRECT_PACK_C_COLS * sizeof(float), 0U,
        static_cast<uint32_t>((ctx.tiling.cubeN - DIRECT_PACK_C_COLS) * sizeof(float)), 0U};
    const uint64_t outputOffset = static_cast<uint64_t>(rowBase) * ctx.tiling.cubeN + colBase;
    AscendC::DataCopyPad(ctx.workspace[ctx.realBase + outputOffset], ctx.transReal, outputCopy);
    AscendC::DataCopyPad(ctx.workspace[ctx.imagBase + outputOffset], ctx.transNegImag, outputCopy);
    AscendC::DataCopyPad(ctx.workspace[ctx.negImagBase + outputOffset], ctx.transImag, outputCopy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
}

__aicore__ inline void RunDirectPackCLoop(
    AscendC::LocalTensor<float>& aos, AscendC::LocalTensor<float>& real, AscendC::LocalTensor<float>& imag,
    AscendC::LocalTensor<float>& transReal, AscendC::LocalTensor<float>& transNegImag,
    AscendC::LocalTensor<float>& transImag, AscendC::LocalTensor<uint32_t>& offsets, GM_ADDR aGm, GM_ADDR workspaceGm,
    const CherkArch22TilingData& tiling, GM_ADDR metadataGm, uint32_t blockIdx, uint32_t blockNum, AscendC::TPipe& pipe)
{
    AscendC::GlobalTensor<uint32_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
    AscendC::DataCopy(offsets, metadata[CHERK_ARCH22_META_PACK_C_OFFSET], CHERK_ARCH22_META_PACK_C_COUNT);
    const event_t metadataToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> workspace;
    InitCherkFloatBuffers(
        a, workspace, aGm, workspaceGm, static_cast<uint64_t>(tiling.lda) * tiling.n * 2ULL,
        CherkArch22DirectWorkspaceFloats(tiling.cubeN, tiling.cubeK));
    const CherkCopyEvents events = GetCherkCopyEvents();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(events.vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(events.mte3ToV);
    const uint64_t realBase = CherkArch22DirectOffsetRealInput(tiling.cubeN, tiling.cubeK);
    const uint64_t imagBase = CherkArch22DirectOffsetImagInput(tiling.cubeN, tiling.cubeK);
    const uint64_t negImagBase = CherkArch22DirectOffsetNegImagInput(tiling.cubeN, tiling.cubeK);
    CherkDirectPackCContext context{
        aos,           real,   imag,     transReal, transNegImag, transImag,      offsets,        a,
        workspace,     tiling, realBase, imagBase,  negImagBase,  events.vToMte2, events.mte2ToV, events.vToMte3,
        events.mte3ToV};
    const uint32_t rowTiles = tiling.cubeK / DIRECT_PACK_C_ROWS;
    const uint32_t colTiles = tiling.cubeN / DIRECT_PACK_C_COLS;
    const uint32_t tileCount = rowTiles * colTiles;
    blockNum = CherkNormalizeBlockCount(blockNum);
    for (uint32_t tile = blockIdx; tile < tileCount; tile += blockNum) {
        ProcessDirectPackCTile(context, tile, colTiles);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(events.vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(events.mte3ToV);
}
} // namespace

__aicore__ inline void CherkArch22DirectPackCImpl(
    GM_ADDR aGm, GM_ADDR workspaceGm, GM_ADDR metadataGm, const CherkArch22TilingData& tiling, uint32_t blockIdx,
    uint32_t blockNum, AscendC::TPipe& pipe)
{
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf, planarBuf, transposedBuf, offsetBuf;
    pipe.InitBuffer(aosBuf, DIRECT_PACK_C_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(planarBuf, DIRECT_PACK_C_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(transposedBuf, DIRECT_PACK_C_ELEMENTS * 3U * sizeof(float));
    pipe.InitBuffer(offsetBuf, DIRECT_PACK_C_ELEMENTS * sizeof(uint32_t));

    AscendC::LocalTensor<float> aos = aosBuf.Get<float>();
    AscendC::LocalTensor<float> planar = planarBuf.Get<float>();
    AscendC::LocalTensor<float> real = planar;
    AscendC::LocalTensor<float> imag = planar[DIRECT_PACK_C_ELEMENTS];
    AscendC::LocalTensor<float> transposed = transposedBuf.Get<float>();
    AscendC::LocalTensor<float> transReal = transposed;
    AscendC::LocalTensor<float> transNegImag = transposed[DIRECT_PACK_C_ELEMENTS];
    AscendC::LocalTensor<float> transImag = transposed[DIRECT_PACK_C_ELEMENTS * 2U];
    AscendC::LocalTensor<uint32_t> offsets = offsetBuf.Get<uint32_t>();

    RunDirectPackCLoop(
        aos, real, imag, transReal, transNegImag, transImag, offsets, aGm, workspaceGm, tiling, metadataGm, blockIdx,
        blockNum, pipe);
}

// Low-rank Cube scheduling and matrix-product device class.
namespace {
constexpr uint32_t DIRECT_MMAD_M = 128U;
constexpr uint32_t DIRECT_MMAD_N = 128U;
constexpr uint32_t DIRECT_MMAD_K = 64U;
constexpr uint32_t DIRECT_MMAD_A_SIZE = DIRECT_MMAD_M * DIRECT_MMAD_K;
constexpr uint32_t DIRECT_MMAD_B_SIZE = DIRECT_MMAD_K * DIRECT_MMAD_N;
constexpr uint32_t DIRECT_MMAD_C_SIZE = DIRECT_MMAD_M * DIRECT_MMAD_N;
constexpr uint32_t DIRECT_MMAD_A_CAPACITY = 2U * DIRECT_MMAD_A_SIZE;

struct DirectMmadResources {
    AscendC::LocalTensor<float> aReal1;
    AscendC::LocalTensor<float> aImag1;
    AscendC::LocalTensor<float> aNegImag1;
    AscendC::LocalTensor<float> bReal1;
    AscendC::LocalTensor<float> bImag1;
    AscendC::LocalTensor<float> a2;
    AscendC::LocalTensor<float> a2Alt;
    AscendC::LocalTensor<float> b2;
    AscendC::LocalTensor<float> b2Alt;
    AscendC::LocalTensor<float> realC1;
    AscendC::LocalTensor<float> imagC1;
};

struct DirectMmadParams {
    AscendC::Nd2NzParams aCopy;
    AscendC::Nd2NzParams rhsNd2Nz;
    AscendC::LoadData3DParamsV2<float> aLoad;
    AscendC::LoadData3DParamsV2<float> bLoad;
    AscendC::MmadParams first;
    AscendC::MmadParams accum;
};

struct DirectMmadSchedule {
    uint64_t taskBegin;
    uint64_t taskEnd;
    uint64_t rowTaskBase;
    uint32_t tileAxis;
    uint32_t mTile;
    uint32_t iteration;
    uint32_t taskStride;
    bool logicalUpper;
    bool mixedConsumer;
    bool stridedMixed;
};

struct DirectMmadTask {
    uint64_t task;
    uint32_t half;
    uint32_t nTile;
    uint16_t ringSlot;
    bool merge;
};

__aicore__ inline void InitDirectMmadCopyParams(DirectMmadParams& p, uint32_t sourceStride)
{
    p.aCopy.ndNum = 1U;
    p.aCopy.nValue = DIRECT_MMAD_K;
    p.aCopy.dValue = DIRECT_MMAD_M;
    p.aCopy.srcDValue = sourceStride;
    p.aCopy.dstNzC0Stride = DIRECT_MMAD_K;
    p.aCopy.dstNzNStride = 1U;
    p.rhsNd2Nz.ndNum = 1U;
    p.rhsNd2Nz.nValue = DIRECT_MMAD_K;
    p.rhsNd2Nz.dValue = DIRECT_MMAD_N;
    p.rhsNd2Nz.srcDValue = sourceStride;
    p.rhsNd2Nz.dstNzC0Stride = DIRECT_MMAD_K;
    p.rhsNd2Nz.dstNzNStride = 1U;
}

__aicore__ inline void InitDirectMmadLoadParams(DirectMmadParams& p)
{
    p.aLoad.l1H = 1U;
    p.aLoad.l1W = DIRECT_MMAD_K;
    p.aLoad.channelSize = DIRECT_MMAD_M;
    p.aLoad.kExtension = DIRECT_MMAD_M;
    p.aLoad.mExtension = DIRECT_MMAD_K;
    p.aLoad.strideW = 1U;
    p.aLoad.strideH = 1U;
    p.aLoad.filterW = 1U;
    p.aLoad.filterH = 1U;
    p.aLoad.dilationFilterW = 1U;
    p.aLoad.dilationFilterH = 1U;
    p.aLoad.enTranspose = true;
    p.bLoad = p.aLoad;
    p.bLoad.channelSize = DIRECT_MMAD_N;
    p.bLoad.kExtension = DIRECT_MMAD_N;
}

__aicore__ inline void InitDirectMmadComputeParams(DirectMmadParams& p)
{
    p.first.m = DIRECT_MMAD_M;
    p.first.n = DIRECT_MMAD_N;
    p.first.k = DIRECT_MMAD_K;
    p.first.cmatrixInitVal = true;
    p.first.cmatrixSource = false;
    p.first.kDirectionAlign = false;
    p.accum = p.first;
    p.accum.cmatrixInitVal = false;
}

__aicore__ inline void InitDirectMmadPipeline()
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
}

__aicore__ inline DirectMmadSchedule InitDirectMmadSchedule(
    uint32_t taskIndex, uint32_t taskStride, uint32_t tileAxis, bool logicalUpper, bool mixedConsumer,
    bool stridedMixed)
{
    taskStride = taskStride == 0U ? 1U : taskStride;
    const uint64_t tasks = static_cast<uint64_t>(tileAxis) * (tileAxis + 1U) / 2ULL;
    const uint64_t work = mixedConsumer && !stridedMixed ? 2ULL * tasks : tasks;
    const uint64_t begin = stridedMixed ? taskIndex : static_cast<uint64_t>(taskIndex) * work / taskStride;
    const uint64_t end = stridedMixed ? tasks : static_cast<uint64_t>(taskIndex + 1U) * work / taskStride;
    DirectMmadSchedule s{begin, end, 0ULL, tileAxis, 0U, 0U, taskStride, logicalUpper, mixedConsumer, stridedMixed};
    const uint64_t firstTask = mixedConsumer && !stridedMixed ? begin / 2ULL : begin;
    while (s.mTile < tileAxis) {
        const uint32_t first = logicalUpper ? 0U : s.mTile;
        const uint32_t last = logicalUpper ? s.mTile + 1U : tileAxis;
        const uint32_t rowTasks = last - first;
        if (firstTask < s.rowTaskBase + rowTasks) {
            break;
        }
        s.rowTaskBase += rowTasks;
        ++s.mTile;
    }
    return s;
}

__aicore__ inline DirectMmadTask ResolveDirectMmadTask(DirectMmadSchedule& s, uint64_t workUnit)
{
    const bool split = s.mixedConsumer && !s.stridedMixed;
    const uint64_t task = split ? workUnit / 2ULL : workUnit;
    const uint32_t half = split ? static_cast<uint32_t>(workUnit & 1ULL) : 0U;
    const bool merge = !s.mixedConsumer || s.stridedMixed || (half == 0U && workUnit + 1ULL < s.taskEnd);
    uint32_t first = s.logicalUpper ? 0U : s.mTile;
    uint32_t last = s.logicalUpper ? s.mTile + 1U : s.tileAxis;
    uint32_t rowTasks = last - first;
    while (task >= s.rowTaskBase + rowTasks) {
        s.rowTaskBase += rowTasks;
        ++s.mTile;
        first = s.logicalUpper ? 0U : s.mTile;
        last = s.logicalUpper ? s.mTile + 1U : s.tileAxis;
        rowTasks = last - first;
    }
    const uint32_t nTile = first + static_cast<uint32_t>(task - s.rowTaskBase);
    const uint16_t slot = static_cast<uint16_t>(s.iteration % DIRECT_MIX_FLAG_COUNT);
    if (s.mixedConsumer && s.iteration >= DIRECT_MIX_FLAG_COUNT) {
        AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(static_cast<uint16_t>(DIRECT_MIX_ACK_BASE + slot));
    }
    return {task, half, nTile, slot, merge};
}

__aicore__ inline void ConfigureDirectMmadTile(DirectMmadParams& p, uint32_t tileHeight, uint32_t tileWidth)
{
    p.aCopy.dValue = tileHeight;
    p.aLoad.channelSize = tileHeight;
    p.aLoad.kExtension = tileHeight;
    p.first.m = tileHeight;
    p.accum.m = tileHeight;
    p.rhsNd2Nz.dValue = tileWidth;
    p.bLoad.channelSize = tileWidth;
    p.bLoad.kExtension = tileWidth;
    p.first.n = tileWidth;
    p.accum.n = tileWidth;
}

__aicore__ inline void IssueDirectMmadTerm(
    DirectMmadResources& r, DirectMmadParams& p, uint32_t& term, AscendC::LocalTensor<float>& output,
    AscendC::LocalTensor<float>& left, AscendC::LocalTensor<float>& right, bool initialize)
{
    const uint16_t slot = static_cast<uint16_t>(term & 1U);
    AscendC::LocalTensor<float>& a2 = slot == 0U ? r.a2 : r.a2Alt;
    AscendC::LocalTensor<float>& b2 = slot == 0U ? r.b2 : r.b2Alt;
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(slot);
    AscendC::LoadData(b2, right, p.bLoad);
    AscendC::LoadData(a2, left, p.aLoad);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
    AscendC::MmadParams& params = initialize ? p.first : p.accum;
    AscendC::Mmad(output, a2, b2, params);
    if ((params.m / 16U) * (params.n / 16U) < 10U) {
        AscendC::PipeBarrier<PIPE_M>();
    }
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(slot);
    ++term;
}
} // namespace

class CherkArch22DirectMmad {
public:
    __aicore__ inline void Init(__gm__ float* base, const CherkArch22TilingData& tiling)
    {
        const uint64_t nk = CherkArch22DirectInputElements(tiling.cubeN, tiling.cubeK);
        const uint64_t nn = static_cast<uint64_t>(tiling.cubeN) * tiling.cubeN;
        realInput_.SetGlobalBuffer(base + CherkArch22DirectOffsetRealInput(tiling.cubeN, tiling.cubeK), nk);
        imagInput_.SetGlobalBuffer(base + CherkArch22DirectOffsetImagInput(tiling.cubeN, tiling.cubeK), nk);
        negImagInput_.SetGlobalBuffer(base + CherkArch22DirectOffsetNegImagInput(tiling.cubeN, tiling.cubeK), nk);
        realOutput_.SetGlobalBuffer(base + CherkArch22DirectOffsetRealOutput(tiling.cubeN, tiling.cubeK), nn);
        imagOutput_.SetGlobalBuffer(base + CherkArch22DirectOffsetImagOutput(tiling.cubeN, tiling.cubeK), nn);
        n_ = tiling.cubeN;
        validK_ = tiling.k;
        uplo_ = tiling.uplo;
    }

    __aicore__ inline void Process(
        uint32_t taskIndex, uint32_t taskStride, bool mixedConsumer = false, bool stridedMixed = false)
    {
        taskStride = taskStride == 0U ? 1U : taskStride;
        constexpr uint64_t bBase = static_cast<uint64_t>(DIRECT_MMAD_A_CAPACITY) * 3ULL * sizeof(float);
        DirectMmadResources resources{
            {AscendC::TPosition::A1, 0ULL, DIRECT_MMAD_A_CAPACITY},
            {AscendC::TPosition::A1, static_cast<uint64_t>(DIRECT_MMAD_A_CAPACITY) * sizeof(float),
             DIRECT_MMAD_A_CAPACITY},
            {AscendC::TPosition::A1, static_cast<uint64_t>(DIRECT_MMAD_A_CAPACITY) * 2ULL * sizeof(float),
             DIRECT_MMAD_A_CAPACITY},
            {AscendC::TPosition::B1, bBase, DIRECT_MMAD_B_SIZE},
            {AscendC::TPosition::B1, bBase + static_cast<uint64_t>(DIRECT_MMAD_B_SIZE) * sizeof(float),
             DIRECT_MMAD_B_SIZE},
            {AscendC::TPosition::A2, 0U, DIRECT_MMAD_A_SIZE},
            {AscendC::TPosition::A2, static_cast<uint64_t>(DIRECT_MMAD_A_SIZE) * sizeof(float), DIRECT_MMAD_A_SIZE},
            {AscendC::TPosition::B2, 0U, DIRECT_MMAD_B_SIZE},
            {AscendC::TPosition::B2, static_cast<uint64_t>(DIRECT_MMAD_B_SIZE) * sizeof(float), DIRECT_MMAD_B_SIZE},
            {AscendC::TPosition::CO1, 0U, DIRECT_MMAD_C_SIZE},
            {AscendC::TPosition::CO1, static_cast<uint64_t>(DIRECT_MMAD_C_SIZE) * sizeof(float), DIRECT_MMAD_C_SIZE}};
        DirectMmadParams params{};
        InitDirectMmadCopyParams(params, n_);
        InitDirectMmadLoadParams(params);
        InitDirectMmadComputeParams(params);
        InitDirectMmadPipeline();
        DirectMmadSchedule schedule = InitDirectMmadSchedule(
            taskIndex, taskStride, n_ / DIRECT_MMAD_M, uplo_ == ACLBLAS_UPPER, mixedConsumer, stridedMixed);
        RunTasks(resources, params, schedule);
        DrainPipeline(schedule);
    }

private:
    __aicore__ inline void LoadDirectMmadPanel(
        DirectMmadResources& r, DirectMmadParams& p, uint32_t mBase, uint32_t nBase, uint32_t kBase, uint32_t currentK)
    {
        p.aCopy.nValue = currentK;
        p.aCopy.dstNzC0Stride = currentK;
        p.rhsNd2Nz.nValue = currentK;
        p.rhsNd2Nz.dstNzC0Stride = currentK;
        p.aLoad.l1W = currentK;
        p.aLoad.mExtension = currentK;
        p.bLoad.l1W = currentK;
        p.bLoad.mExtension = currentK;
        p.first.k = currentK;
        p.accum.k = currentK;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        const uint64_t aOffset = static_cast<uint64_t>(kBase) * n_ + mBase;
        const uint64_t bOffset = static_cast<uint64_t>(kBase) * n_ + nBase;
        AscendC::DataCopy(r.aReal1, realInput_[aOffset], p.aCopy);
        AscendC::DataCopy(r.aImag1, imagInput_[aOffset], p.aCopy);
        AscendC::DataCopy(r.aNegImag1, negImagInput_[aOffset], p.aCopy);
        AscendC::DataCopy(r.bReal1, realInput_[bOffset], p.rhsNd2Nz);
        AscendC::DataCopy(r.bImag1, imagInput_[bOffset], p.rhsNd2Nz);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void ComputeDirectMmadTile(
        DirectMmadResources& r, DirectMmadParams& p, uint32_t mBase, uint32_t nBase)
    {
        const uint32_t computeK = ((validK_ + 15U) / 16U) * 16U;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        for (uint32_t kBase = 0U; kBase < computeK; kBase += DIRECT_MMAD_K) {
            const uint32_t currentK = Min<uint32_t>(DIRECT_MMAD_K, computeK - kBase);
            LoadDirectMmadPanel(r, p, mBase, nBase, kBase, currentK);
            const bool initialize = kBase == 0U;
            uint32_t term = 0U;
            IssueDirectMmadTerm(r, p, term, r.realC1, r.aReal1, r.bReal1, initialize);
            IssueDirectMmadTerm(r, p, term, r.imagC1, r.aReal1, r.bImag1, initialize);
            IssueDirectMmadTerm(r, p, term, r.realC1, r.aImag1, r.bImag1, false);
            IssueDirectMmadTerm(r, p, term, r.imagC1, r.aNegImag1, r.bReal1, false);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void StoreDirectMmadTile(
        DirectMmadResources& r, uint32_t mBase, uint32_t nBase, uint32_t height, uint32_t width)
    {
        AscendC::FixpipeParamsV220 store{};
        store.mSize = height;
        store.nSize = width;
        store.srcStride = height;
        store.dstStride = n_;
        store.ndNum = 1U;
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::Fixpipe(realOutput_[static_cast<uint64_t>(mBase) * n_ + nBase], r.realC1, store);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID1);
        AscendC::Fixpipe(imagOutput_[static_cast<uint64_t>(mBase) * n_ + nBase], r.imagC1, store);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    }

    __aicore__ inline uint64_t ProcessTask(
        DirectMmadResources& r, DirectMmadParams& p, DirectMmadSchedule& s, uint64_t workUnit)
    {
        const DirectMmadTask task = ResolveDirectMmadTask(s, workUnit);
        const uint32_t mBase = s.mTile * DIRECT_MMAD_M;
        const uint32_t nBase = task.nTile * DIRECT_MMAD_N + (task.merge ? 0U : task.half * (DIRECT_MMAD_N / 2U));
        const uint32_t height = Min<uint32_t>(DIRECT_MMAD_M, n_ - mBase);
        const uint32_t width =
            nBase < n_ ? Min<uint32_t>(task.merge ? DIRECT_MMAD_N : DIRECT_MMAD_N / 2U, n_ - nBase) : 0U;
        if (width != 0U) {
            ConfigureDirectMmadTile(p, height, width);
            ComputeDirectMmadTile(r, p, mBase, nBase);
            StoreDirectMmadTile(r, mBase, nBase, height, width);
        }
        if (s.mixedConsumer) {
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(task.ringSlot);
        }
        ++s.iteration;
        if (!s.mixedConsumer) {
            return 1ULL;
        }
        return s.stridedMixed ? s.taskStride : (task.merge ? 2ULL : 1ULL);
    }

    __aicore__ inline void RunTasks(DirectMmadResources& r, DirectMmadParams& p, DirectMmadSchedule& schedule)
    {
        for (uint64_t work = schedule.taskBegin; work < schedule.taskEnd;) {
            work += ProcessTask(r, p, schedule, work);
        }
    }

    __aicore__ inline void DrainPipeline(const DirectMmadSchedule& schedule)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        if (!schedule.mixedConsumer) {
            return;
        }
        const uint32_t count = Min<uint32_t>(schedule.iteration, DIRECT_MIX_FLAG_COUNT);
        for (uint32_t drain = 0U; drain < count; ++drain) {
            const uint16_t slot = static_cast<uint16_t>((schedule.iteration - 1U - drain) % DIRECT_MIX_FLAG_COUNT);
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(static_cast<uint16_t>(DIRECT_MIX_ACK_BASE + slot));
        }
    }

    AscendC::GlobalTensor<float> realInput_, imagInput_, negImagInput_;
    AscendC::GlobalTensor<float> realOutput_, imagOutput_;
    uint32_t n_ = 0U;
    uint32_t validK_ = 0U;
    uint32_t uplo_ = ACLBLAS_LOWER;
};

// Low-rank product scaling and triangular output device helpers.
namespace {
constexpr uint32_t DIRECT_EPILOGUE_TILE = 64U;
constexpr uint32_t DIRECT_EPILOGUE_ELEMENTS = DIRECT_EPILOGUE_TILE * DIRECT_EPILOGUE_TILE;
constexpr uint32_t DIRECT_EPILOGUE_AOS_ELEMENTS = DIRECT_EPILOGUE_ELEMENTS * 2U;

struct CherkDirectEpilogueBuffers {
    AscendC::LocalTensor<float> real;
    AscendC::LocalTensor<float> imag;
    AscendC::LocalTensor<float> outputAos;
    AscendC::LocalTensor<float> oldAos;
    AscendC::LocalTensor<float> zero;
    AscendC::LocalTensor<uint32_t> interleaveOffsets;
    AscendC::LocalTensor<uint8_t> diagonalMask;
    AscendC::LocalTensor<uint8_t> triangleMask;
};

__aicore__ inline void LoadDirectProduct(
    AscendC::GlobalTensor<float>& workspace, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, CherkDirectEpilogueBuffers& b);

__aicore__ inline void LoadDirectOldC(
    AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, CherkDirectEpilogueBuffers& b);

__aicore__ inline void ApplyDirectScalars(
    float alpha, float beta, bool oldLoaded, bool diagonalTile, CherkDirectEpilogueBuffers& b,
    bool mergeTriangle = true);

__aicore__ inline void StoreDirectTile(
    AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, bool diagonalTile, bool oldLoaded, event_t vToMte3, event_t mte3ToV,
    CherkDirectEpilogueBuffers& b);

__aicore__ inline CherkDirectEpilogueBuffers InitDirectEpilogueBuffers(
    AscendC::TPipe& pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& planarBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& outputBuf, AscendC::TBuf<AscendC::TPosition::VECCALC>& oldBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& interleaveOffsetBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& diagonalMaskBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& triangleMaskBuf)
{
    pipe.InitBuffer(planarBuf, DIRECT_EPILOGUE_ELEMENTS * 3U * sizeof(float));
    pipe.InitBuffer(outputBuf, DIRECT_EPILOGUE_AOS_ELEMENTS * sizeof(float));
    pipe.InitBuffer(oldBuf, DIRECT_EPILOGUE_AOS_ELEMENTS * sizeof(float));
    pipe.InitBuffer(interleaveOffsetBuf, DIRECT_EPILOGUE_AOS_ELEMENTS * sizeof(uint32_t));
    pipe.InitBuffer(diagonalMaskBuf, DIRECT_EPILOGUE_ELEMENTS / 8U);
    pipe.InitBuffer(triangleMaskBuf, DIRECT_EPILOGUE_ELEMENTS / 8U);

    AscendC::LocalTensor<float> planar = planarBuf.Get<float>();
    return {
        planar,
        planar[DIRECT_EPILOGUE_ELEMENTS],
        outputBuf.Get<float>(),
        oldBuf.Get<float>(),
        planar[DIRECT_EPILOGUE_ELEMENTS * 2U],
        interleaveOffsetBuf.Get<uint32_t>(),
        diagonalMaskBuf.Get<uint8_t>(),
        triangleMaskBuf.Get<uint8_t>()};
}

__aicore__ inline void ProcessDirectEpilogueTile(
    float alpha, float beta, bool hasProduct, AscendC::GlobalTensor<float>& c, AscendC::GlobalTensor<float>& workspace,
    const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols,
    bool diagonalTile, CherkDirectEpilogueBuffers& buffers, event_t mte2ToV, event_t vToMte2, event_t vToMte3,
    event_t mte3ToV)
{
    const bool loadOld = beta != 0.0f || (diagonalTile && tiling.uplo == ACLBLAS_LOWER);
    if (hasProduct) {
        LoadDirectProduct(workspace, tiling, rowBase, colBase, validRows, validCols, buffers);
    }
    if (loadOld) {
        LoadDirectOldC(c, tiling, rowBase, colBase, validRows, validCols, buffers);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    ApplyDirectScalars(alpha, beta, loadOld, diagonalTile, buffers);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    StoreDirectTile(
        c, tiling, rowBase, colBase, validRows, validCols, diagonalTile, loadOld, vToMte3, mte3ToV, buffers);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
}

__aicore__ inline void LoadDirectEpilogueMetadata(
    GM_ADDR metadataGm, uint32_t uplo, CherkDirectEpilogueBuffers& b, bool needTriangleMask = true)
{
    AscendC::GlobalTensor<uint32_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
    AscendC::GlobalTensor<uint8_t> metadataBytes;
    metadataBytes.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(metadataGm), CHERK_ARCH22_META_WORDS * sizeof(uint32_t));
    AscendC::DataCopy(
        b.interleaveOffsets, metadata[CHERK_ARCH22_META_INTERLEAVE_OFFSET], CHERK_ARCH22_META_INTERLEAVE_COUNT);
    AscendC::DataCopy(
        b.diagonalMask, metadataBytes[CHERK_ARCH22_META_DIAGONAL_OFFSET * sizeof(uint32_t)],
        DIRECT_EPILOGUE_ELEMENTS / 8U);
    // LoadProductTile stores each GM row (logical column) as one UB block,
    // so the flattened UB index is [logical_col][logical_row].  AscendC::Select the
    // transposed mask to keep the requested logical triangle.
    if (needTriangleMask) {
        const uint32_t triangleOffset =
            uplo == ACLBLAS_UPPER ? CHERK_ARCH22_META_LOWER_OFFSET : CHERK_ARCH22_META_UPPER_OFFSET;
        AscendC::DataCopy(
            b.triangleMask, metadataBytes[triangleOffset * sizeof(uint32_t)], DIRECT_EPILOGUE_ELEMENTS / 8U);
    }
    // DEBUG_MASK_ZERO
    const event_t metadataToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(metadataToV);
}

__aicore__ inline void LoadDirectProduct(
    AscendC::GlobalTensor<float>& workspace, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, CherkDirectEpilogueBuffers& b)
{
    const uint32_t blockBytes = validRows * sizeof(float);
    const uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(validCols), blockBytes, static_cast<uint32_t>((tiling.cubeN - validRows) * sizeof(float)),
        static_cast<uint32_t>((DIRECT_EPILOGUE_TILE * sizeof(float) - alignedBlockBytes) / 32U), 0U};
    const AscendC::DataCopyPadExtParams<float> pad{
        alignedBlockBytes != blockBytes, 0U, static_cast<uint8_t>((alignedBlockBytes - blockBytes) / sizeof(float)),
        0.0f};
    // Cube publishes row-major H^T planes; column-major C consumes [col,row].
    const uint64_t sourceOffset = static_cast<uint64_t>(colBase) * tiling.cubeN + rowBase;
    AscendC::DataCopyPad(
        b.real, workspace[CherkArch22DirectOffsetRealOutput(tiling.cubeN, tiling.cubeK) + sourceOffset], copy, pad);
    AscendC::DataCopyPad(
        b.imag, workspace[CherkArch22DirectOffsetImagOutput(tiling.cubeN, tiling.cubeK) + sourceOffset], copy, pad);
}

__aicore__ inline void LoadDirectOldC(
    AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, CherkDirectEpilogueBuffers& b)
{
    const uint32_t blockBytes = validRows * 2U * sizeof(float);
    const uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(validCols), blockBytes,
        static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)),
        static_cast<uint32_t>((DIRECT_EPILOGUE_TILE * 2U * sizeof(float) - alignedBlockBytes) / 32U), 0U};
    const AscendC::DataCopyPadExtParams<float> pad{
        alignedBlockBytes != blockBytes, 0U, static_cast<uint8_t>((alignedBlockBytes - blockBytes) / sizeof(float)),
        0.0f};
    const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
    AscendC::DataCopyPad(b.oldAos, c[cOffset], copy, pad);
}

__aicore__ inline void ScaleDirectProduct(float alpha, CherkDirectEpilogueBuffers& b)
{
    if (alpha == 0.0f) {
        AscendC::Duplicate(b.real, 0.0f, DIRECT_EPILOGUE_ELEMENTS);
        AscendC::Duplicate(b.imag, 0.0f, DIRECT_EPILOGUE_ELEMENTS);
    } else if (alpha != 1.0f) {
        AscendC::Muls(b.real, b.real, alpha, DIRECT_EPILOGUE_ELEMENTS);
        AscendC::Muls(b.imag, b.imag, alpha, DIRECT_EPILOGUE_ELEMENTS);
    }
}

__aicore__ inline void MergeDirectOldC(float beta, CherkDirectEpilogueBuffers& b)
{
    AscendC::LocalTensor<float> oldReal = b.outputAos;
    AscendC::LocalTensor<float> oldImag = b.outputAos[DIRECT_EPILOGUE_ELEMENTS];
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = DIRECT_EPILOGUE_AOS_ELEMENTS / 64U;
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::GatherMask<float>(oldReal, b.oldAos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(oldImag, b.oldAos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    if (beta == 0.0f) {
        return;
    }

    // Preserve outputAos for the triangle merge and scale through a separate
    // temporary plane.
    AscendC::Muls(b.zero, oldReal, beta, DIRECT_EPILOGUE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(b.real, b.real, b.zero, DIRECT_EPILOGUE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(b.zero, oldImag, beta, DIRECT_EPILOGUE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(b.imag, b.imag, b.zero, DIRECT_EPILOGUE_ELEMENTS);
}

__aicore__ inline void ClearDirectDiagonalImag(CherkDirectEpilogueBuffers& b)
{
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(b.zero, 0.0f, DIRECT_EPILOGUE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select(
        b.imag, b.diagonalMask, b.zero, b.imag, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, DIRECT_EPILOGUE_ELEMENTS);
}

__aicore__ inline void MergeDirectTriangle(CherkDirectEpilogueBuffers& b)
{
    AscendC::LocalTensor<float> oldReal = b.outputAos;
    AscendC::LocalTensor<float> oldImag = b.outputAos[DIRECT_EPILOGUE_ELEMENTS];
    AscendC::Select(
        b.real, b.triangleMask, b.real, oldReal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, DIRECT_EPILOGUE_ELEMENTS);
    AscendC::Select(
        b.imag, b.triangleMask, b.imag, oldImag, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, DIRECT_EPILOGUE_ELEMENTS);
}

__aicore__ inline void InterleaveDirectOutput(CherkDirectEpilogueBuffers& b)
{
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(b.outputAos, b.real, b.interleaveOffsets, 0U, DIRECT_EPILOGUE_AOS_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ApplyDirectScalars(
    float alpha, float beta, bool oldLoaded, bool diagonalTile, CherkDirectEpilogueBuffers& b, bool mergeTriangle)
{
    ScaleDirectProduct(alpha, b);
    if (oldLoaded) {
        MergeDirectOldC(beta, b);
    }
    if (diagonalTile) {
        ClearDirectDiagonalImag(b);
    }
    if (diagonalTile && oldLoaded && mergeTriangle) {
        MergeDirectTriangle(b);
    }
    InterleaveDirectOutput(b);
}

__aicore__ inline void StoreDirectDiagonalTile(
    AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, event_t vToMte3, event_t mte3ToV, CherkDirectEpilogueBuffers& b)
{
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    for (uint32_t column = 0U; column < validCols; ++column) {
        const uint32_t absoluteColumn = colBase + column;
        const uint32_t columnOffset = column * DIRECT_EPILOGUE_TILE * 2U;
        const uint32_t selectedRows = upper ? Min<uint32_t>(validRows, absoluteColumn - rowBase + 1U) :
                                              validRows - Min<uint32_t>(validRows, absoluteColumn - rowBase);
        const uint32_t preservedRows = validRows - selectedRows;
        if (upper) {
            if (selectedRows == 0U) {
                continue;
            }
            const uint64_t dstOffset = (static_cast<uint64_t>(absoluteColumn) * tiling.ldc + rowBase) * 2ULL;
            const AscendC::DataCopyExtParams columnCopy{
                1U, static_cast<uint32_t>(selectedRows * 2U * sizeof(float)), 0U, 0U, 0U};
            // Upper ownership always starts at rowBase, so the source lane is
            // aligned and no old-C merge is required for the unowned suffix.
            // The source UB is immutable across this loop, so all bursts can
            // be queued after the caller's single V_MTE3 ordering point.
            AscendC::DataCopyPad(c[dstOffset], b.outputAos[columnOffset], columnCopy);
        } else {
            const uint32_t oldRows = preservedRows;
            if (oldRows != 0U) {
                AscendC::Adds(b.outputAos[columnOffset], b.oldAos[columnOffset], 0.0f, oldRows * 2U);
            }
            const uint64_t dstOffset = (static_cast<uint64_t>(absoluteColumn) * tiling.ldc + rowBase) * 2ULL;
            const AscendC::DataCopyExtParams columnCopy{
                1U, static_cast<uint32_t>(validRows * 2U * sizeof(float)), 0U, 0U, 0U};
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            AscendC::DataCopyPad(c[dstOffset], b.outputAos[columnOffset], columnCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
        }
    }
}

__aicore__ inline void StoreDirectTile(
    AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, bool diagonalTile, bool oldLoaded, event_t vToMte3, event_t mte3ToV,
    CherkDirectEpilogueBuffers& b)
{
    if (diagonalTile) {
        const bool fullTile = validRows == DIRECT_EPILOGUE_TILE && validCols == DIRECT_EPILOGUE_TILE;
        if (!fullTile || !oldLoaded) {
            StoreDirectDiagonalTile(c, tiling, rowBase, colBase, validRows, validCols, vToMte3, mte3ToV, b);
            return;
        }
    }
    const uint32_t blockBytes = validRows * 2U * sizeof(float);
    const uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(validCols), blockBytes,
        static_cast<uint32_t>((DIRECT_EPILOGUE_TILE * 2U * sizeof(float) - alignedBlockBytes) / 32U),
        static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)), 0U};
    const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
    AscendC::DataCopyPad(c[cOffset], b.outputAos, copy);
}

// Unit-alpha/beta-zero UPPER output.  The unselected half of a diagonal
// tile must retain old C, so publish only the selected prefix of each column
// instead of reading and merging a full old-C tile.  Off-diagonal tiles retain
// the single aligned 2D burst used by the general epilogue.
__aicore__ inline void StoreDirectUnitUpperTile(
    AscendC::GlobalTensor<float>& c, const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols, bool diagonalTile, CherkDirectEpilogueBuffers& b)
{
    const uint32_t blockBytes = validRows * 2U * sizeof(float);
    const uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams fullCopy{
        static_cast<uint16_t>(validCols), blockBytes,
        static_cast<uint32_t>((DIRECT_EPILOGUE_TILE * 2U * sizeof(float) - alignedBlockBytes) / 32U),
        static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)), 0U};
    const uint64_t baseOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
    if (!diagonalTile) {
        AscendC::DataCopyPad(c[baseOffset], b.outputAos, fullCopy);
        return;
    }

    for (uint32_t column = 0U; column < validCols; ++column) {
        const uint32_t absoluteColumn = colBase + column;
        const uint32_t selectedRows = Min<uint32_t>(validRows, absoluteColumn - rowBase + 1U);
        if (selectedRows == 0U) {
            continue;
        }
        const uint64_t dstOffset = (static_cast<uint64_t>(absoluteColumn) * tiling.ldc + rowBase) * 2ULL;
        const AscendC::DataCopyExtParams columnCopy{
            1U, static_cast<uint32_t>(selectedRows * 2U * sizeof(float)), 0U, 0U, 0U};
        AscendC::DataCopyPad(c[dstOffset], b.outputAos[column * DIRECT_EPILOGUE_TILE * 2U], columnCopy);
    }
}

__aicore__ inline void ApplyDirectUnitUpper(bool diagonalTile, CherkDirectEpilogueBuffers& b)
{
    if (diagonalTile) {
        AscendC::Duplicate(b.zero, 0.0f, DIRECT_EPILOGUE_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            b.imag, b.diagonalMask, b.zero, b.imag, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
            DIRECT_EPILOGUE_ELEMENTS);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(b.outputAos, b.real, b.interleaveOffsets, 0U, DIRECT_EPILOGUE_AOS_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
}
} // namespace

/** OP_N direct-H preprocess: publish Ar, Ai and -Ai as Kp x Np planes. */
extern "C" __global__ __aicore__ void cherk_arch22_direct_pack_n_kernel(
    GM_ADDR aGm, GM_ADDR workspaceGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    CherkArch22DirectPackNImpl(
        aGm, workspaceGm, tiling, static_cast<uint32_t>(AscendC::GetBlockIdx()),
        static_cast<uint32_t>(AscendC::GetBlockNum()), pipe);
}
/** Batched OP_N pack for K=64 direct-H launches. */
template <uint32_t BATCH_ROWS>
__aicore__ inline void ProcessDirectPackNBatchTile(
    AscendC::LocalTensor<float>& aos, AscendC::LocalTensor<float>& real, AscendC::LocalTensor<float>& imag,
    AscendC::LocalTensor<float>& negImag, AscendC::GlobalTensor<float>& a, AscendC::GlobalTensor<float>& workspace,
    const CherkArch22TilingData& tiling, uint32_t rowBase, uint32_t validRows, uint32_t validCols, uint64_t realBase,
    uint64_t imagBase, uint64_t negImagBase, event_t vToMte2, event_t mte2ToV, event_t vToMte3, event_t mte3ToV)
{
    const uint32_t batchElements = BATCH_ROWS * tiling.cubeN;
    AscendC::Duplicate(aos, 0.0f, batchElements * 2U);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    if (validRows != 0U && validCols != 0U) {
        const uint32_t blockBytes = validCols * 2U * sizeof(float);
        const uint32_t alignedBytes = (blockBytes + 31U) & ~31U;
        const AscendC::DataCopyExtParams copy{
            static_cast<uint16_t>(validRows), blockBytes,
            static_cast<uint32_t>((tiling.lda - validCols) * 2U * sizeof(float)),
            static_cast<uint32_t>((tiling.cubeN * 2U * sizeof(float) - alignedBytes) / 32U), 0U};
        const AscendC::DataCopyPadExtParams<float> pad{
            alignedBytes != blockBytes, 0U, static_cast<uint8_t>((alignedBytes - blockBytes) / sizeof(float)), 0.0f};
        AscendC::DataCopyPad(aos, a[static_cast<uint64_t>(rowBase) * tiling.lda * 2ULL], copy, pad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    uint64_t reserved = 0ULL;
    const uint16_t repeats = static_cast<uint16_t>(batchElements * 2U / 64U);
    AscendC::GatherMask<float>(real, aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(imag, aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(negImag, imag, -1.0f, batchElements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    const uint64_t outputOffset = static_cast<uint64_t>(rowBase) * tiling.cubeN;
    AscendC::DataCopy(workspace[realBase + outputOffset], real, batchElements);
    AscendC::DataCopy(workspace[imagBase + outputOffset], imag, batchElements);
    AscendC::DataCopy(workspace[negImagBase + outputOffset], negImag, batchElements);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
}

template <uint32_t BATCH_ROWS>
__aicore__ inline void CherkArch22DirectPackNBatchImpl(
    GM_ADDR aGm, GM_ADDR workspaceGm, const CherkArch22TilingData& tiling, uint32_t blockIdx, uint32_t blockNum,
    AscendC::TPipe& pipe)
{
    const uint32_t batchElements = BATCH_ROWS * tiling.cubeN;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf, planarBuf;
    pipe.InitBuffer(aosBuf, batchElements * 2U * sizeof(float));
    pipe.InitBuffer(planarBuf, batchElements * 3U * sizeof(float));
    AscendC::LocalTensor<float> aos = aosBuf.Get<float>();
    AscendC::LocalTensor<float> real = planarBuf.Get<float>();
    AscendC::LocalTensor<float> imag = real[batchElements];
    AscendC::LocalTensor<float> negImag = real[batchElements * 2U];

    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> workspace;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm), static_cast<uint64_t>(tiling.lda) * tiling.k * 2ULL);
    workspace.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspaceGm), CherkArch22DirectWorkspaceFloats(tiling.cubeN, tiling.cubeK));
    const uint64_t realBase = CherkArch22DirectOffsetRealInput(tiling.cubeN, tiling.cubeK);
    const uint64_t imagBase = CherkArch22DirectOffsetImagInput(tiling.cubeN, tiling.cubeK);
    const uint64_t negImagBase = CherkArch22DirectOffsetNegImagInput(tiling.cubeN, tiling.cubeK);
    blockNum = CherkNormalizeBlockCount(blockNum);

    const uint32_t batchCount = (tiling.cubeK + BATCH_ROWS - 1U) / BATCH_ROWS;
    const event_t vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
    const event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    const event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    const event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    for (uint32_t batch = blockIdx; batch < batchCount; batch += blockNum) {
        const uint32_t rowBase = batch * BATCH_ROWS;
        const uint32_t validRows = rowBase < tiling.k ? Min<uint32_t>(BATCH_ROWS, tiling.k - rowBase) : 0U;
        const uint32_t validCols = tiling.n;

        ProcessDirectPackNBatchTile<BATCH_ROWS>(
            aos, real, imag, negImag, a, workspace, tiling, rowBase, validRows, validCols, realBase, imagBase,
            negImagBase, vToMte2, mte2ToV, vToMte3, mte3ToV);
    }
}

extern "C" __global__ __aicore__ void cherk_arch22_direct_pack_n_batch_kernel(
    GM_ADDR aGm, GM_ADDR workspaceGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    if (tiling.directPackBatchRows == 8U) {
        CherkArch22DirectPackNBatchImpl<8U>(aGm, workspaceGm, tiling, blockIdx, blockNum, pipe);
    } else if (tiling.directPackBatchRows == 4U) {
        CherkArch22DirectPackNBatchImpl<4U>(aGm, workspaceGm, tiling, blockIdx, blockNum, pipe);
    } else if (tiling.directPackBatchRows == 2U) {
        CherkArch22DirectPackNBatchImpl<2U>(aGm, workspaceGm, tiling, blockIdx, blockNum, pipe);
    } else {
        CherkArch22DirectPackNBatchImpl<1U>(aGm, workspaceGm, tiling, blockIdx, blockNum, pipe);
    }
}
/** OP_C direct-H preprocess: transpose AoS N x K into signed K x N planes. */
extern "C" __global__ __aicore__ void cherk_arch22_direct_pack_c_kernel(
    GM_ADDR aGm, GM_ADDR workspaceGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    CherkArch22DirectPackCImpl(
        aGm, workspaceGm, metadataGm, tiling, static_cast<uint32_t>(AscendC::GetBlockIdx()),
        static_cast<uint32_t>(AscendC::GetBlockNum()), pipe);
}
/** Batched OP_C direct-H preprocess for wide K=64 inputs. */
namespace {
constexpr uint32_t DIRECT_PACK_C_BATCH_ROWS = 32U;
constexpr uint32_t DIRECT_PACK_C_BATCH_COLS = 128U;
constexpr uint32_t DIRECT_PACK_C_BATCH_ELEMENTS = DIRECT_PACK_C_BATCH_ROWS * DIRECT_PACK_C_BATCH_COLS;

__aicore__ inline void InitDirectPackCBatchStorage(
    AscendC::TPipe& pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& aosBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& planarBuf, AscendC::TBuf<AscendC::TPosition::VECCALC>& transposedBuf,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& offsetBuf)
{
    pipe.InitBuffer(aosBuf, DIRECT_PACK_C_BATCH_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(planarBuf, DIRECT_PACK_C_BATCH_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(transposedBuf, DIRECT_PACK_C_BATCH_ELEMENTS * 3U * sizeof(float));
    pipe.InitBuffer(offsetBuf, DIRECT_PACK_C_BATCH_ELEMENTS * sizeof(uint32_t));
}

__aicore__ inline void LoadDirectPackCBatchInput(
    CherkDirectPackCContext& ctx, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols)
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::Duplicate(ctx.aos, 0.0f, DIRECT_PACK_C_BATCH_ELEMENTS * 2U);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    if (validRows != 0U && validCols != 0U) {
        const uint32_t blockBytes = validRows * 2U * sizeof(float);
        const uint32_t alignedBytes = (blockBytes + 31U) & ~31U;
        const AscendC::DataCopyExtParams inputCopy{
            static_cast<uint16_t>(validCols), blockBytes,
            static_cast<uint32_t>((ctx.tiling.lda - validRows) * 2U * sizeof(float)),
            static_cast<uint32_t>((DIRECT_PACK_C_BATCH_ROWS * 2U * sizeof(float) - alignedBytes) / 32U), 0U};
        const AscendC::DataCopyPadExtParams<float> inputPad{
            alignedBytes != blockBytes, 0U, static_cast<uint8_t>((alignedBytes - blockBytes) / sizeof(float)), 0.0f};
        const uint64_t inputOffset = (static_cast<uint64_t>(colBase) * ctx.tiling.lda + rowBase) * 2ULL;
        AscendC::DataCopyPad(ctx.aos, ctx.a[inputOffset], inputCopy, inputPad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV);
}

__aicore__ inline void StoreDirectPackCBatchOutput(CherkDirectPackCContext& ctx, uint32_t rowBase, uint32_t colBase)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3);
    const AscendC::DataCopyExtParams outputCopy{
        DIRECT_PACK_C_BATCH_ROWS, DIRECT_PACK_C_BATCH_COLS * sizeof(float), 0U,
        static_cast<uint32_t>((ctx.tiling.cubeN - DIRECT_PACK_C_BATCH_COLS) * sizeof(float)), 0U};
    const uint64_t outputOffset = static_cast<uint64_t>(rowBase) * ctx.tiling.cubeN + colBase;
    AscendC::DataCopyPad(ctx.workspace[ctx.realBase + outputOffset], ctx.transReal, outputCopy);
    AscendC::DataCopyPad(ctx.workspace[ctx.imagBase + outputOffset], ctx.transNegImag, outputCopy);
    AscendC::DataCopyPad(ctx.workspace[ctx.negImagBase + outputOffset], ctx.transImag, outputCopy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
}

__aicore__ inline void ProcessDirectPackCBatchTile(CherkDirectPackCContext& ctx, uint32_t tile, uint32_t colTiles)
{
    colTiles = colTiles == 0U ? 1U : colTiles;
    const uint32_t rowBase = (tile / colTiles) * DIRECT_PACK_C_BATCH_ROWS;
    const uint32_t colBase = (tile % colTiles) * DIRECT_PACK_C_BATCH_COLS;
    const uint32_t validRows =
        rowBase < ctx.tiling.k ? Min<uint32_t>(DIRECT_PACK_C_BATCH_ROWS, ctx.tiling.k - rowBase) : 0U;
    const uint32_t validCols =
        colBase < ctx.tiling.n ? Min<uint32_t>(DIRECT_PACK_C_BATCH_COLS, ctx.tiling.n - colBase) : 0U;
    LoadDirectPackCBatchInput(ctx, rowBase, colBase, validRows, validCols);
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = DIRECT_PACK_C_BATCH_ELEMENTS * 2U / 64U;
    AscendC::GatherMask<float>(ctx.real, ctx.aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(ctx.imag, ctx.aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
    AscendC::Gather(ctx.transReal, ctx.real, ctx.offsets, 0U, DIRECT_PACK_C_BATCH_ELEMENTS);
    AscendC::Gather(ctx.transImag, ctx.imag, ctx.offsets, 0U, DIRECT_PACK_C_BATCH_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ctx.transNegImag, ctx.transImag, -1.0f, DIRECT_PACK_C_BATCH_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    StoreDirectPackCBatchOutput(ctx, rowBase, colBase);
}

__aicore__ inline void RunDirectPackCBatchLoop(
    CherkDirectPackCContext& context, uint32_t blockIdx, uint32_t blockNum, const CherkArch22TilingData& tiling)
{
    const uint32_t rowTiles = (tiling.cubeK + DIRECT_PACK_C_BATCH_ROWS - 1U) / DIRECT_PACK_C_BATCH_ROWS;
    const uint32_t colTiles = (tiling.cubeN + DIRECT_PACK_C_BATCH_COLS - 1U) / DIRECT_PACK_C_BATCH_COLS;
    const uint32_t tileCount = rowTiles * colTiles;
    blockNum = CherkNormalizeBlockCount(blockNum);
    for (uint32_t tile = blockIdx; tile < tileCount; tile += blockNum) {
        ProcessDirectPackCBatchTile(context, tile, colTiles);
    }
}
} // namespace

__aicore__ inline void CherkArch22DirectPackCBatchImpl(
    GM_ADDR aGm, GM_ADDR workspaceGm, GM_ADDR metadataGm, const CherkArch22TilingData& tiling, uint32_t blockIdx,
    uint32_t blockNum, AscendC::TPipe& pipe)
{
    // One AoS input tile, two planar extraction tiles, three transposed output
    // planes and the fixed transpose table fit below the A3 UB budget.
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf, planarBuf, transposedBuf, offsetBuf;
    InitDirectPackCBatchStorage(pipe, aosBuf, planarBuf, transposedBuf, offsetBuf);

    AscendC::LocalTensor<float> aos = aosBuf.Get<float>();
    AscendC::LocalTensor<float> planar = planarBuf.Get<float>();
    AscendC::LocalTensor<float> real = planar;
    AscendC::LocalTensor<float> imag = planar[DIRECT_PACK_C_BATCH_ELEMENTS];
    AscendC::LocalTensor<float> transposed = transposedBuf.Get<float>();
    AscendC::LocalTensor<float> transReal = transposed;
    AscendC::LocalTensor<float> transNegImag = transposed[DIRECT_PACK_C_BATCH_ELEMENTS];
    AscendC::LocalTensor<float> transImag = transposed[DIRECT_PACK_C_BATCH_ELEMENTS * 2U];
    AscendC::LocalTensor<uint32_t> offsets = offsetBuf.Get<uint32_t>();

    LoadCherkMetadataOffsets(
        offsets, metadataGm, CHERK_ARCH22_META_PACK_C128_OFFSET, CHERK_ARCH22_META_PACK_C128_COUNT);

    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> workspace;
    InitCherkFloatBuffers(
        a, workspace, aGm, workspaceGm, static_cast<uint64_t>(tiling.lda) * tiling.n * 2ULL,
        CherkArch22DirectWorkspaceFloats(tiling.cubeN, tiling.cubeK));
    const uint64_t realBase = CherkArch22DirectOffsetRealInput(tiling.cubeN, tiling.cubeK);
    const uint64_t imagBase = CherkArch22DirectOffsetImagInput(tiling.cubeN, tiling.cubeK);
    const uint64_t negImagBase = CherkArch22DirectOffsetNegImagInput(tiling.cubeN, tiling.cubeK);

    const CherkCopyEvents events = GetCherkCopyEvents();
    const event_t vToMte2 = events.vToMte2;
    const event_t mte2ToV = events.mte2ToV;
    const event_t vToMte3 = events.vToMte3;
    const event_t mte3ToV = events.mte3ToV;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);

    CherkDirectPackCContext context{aos,         real,    imag,      transReal, transNegImag, transImag,
                                    offsets,     a,       workspace, tiling,    realBase,     imagBase,
                                    negImagBase, vToMte2, mte2ToV,   vToMte3,   mte3ToV};
    RunDirectPackCBatchLoop(context, blockIdx, blockNum, tiling);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
}

extern "C" __global__ __aicore__ void cherk_arch22_direct_pack_c_batch_kernel(
    GM_ADDR aGm, GM_ADDR workspaceGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    CherkArch22DirectPackCBatchImpl(
        aGm, workspaceGm, metadataGm, tiling, static_cast<uint32_t>(AscendC::GetBlockIdx()),
        static_cast<uint32_t>(AscendC::GetBlockNum()), pipe);
}

/** Low-rank direct-H Cube producer for KxN planar input. */
extern "C" __global__ __aicore__ void cherk_arch22_direct_mmad_kernel(
    GM_ADDR workspaceGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    const uint32_t blocks = CherkNormalizeBlockCount(static_cast<uint32_t>(AscendC::GetBlockNum()));
    CherkArch22DirectMmad mm;
    mm.Init(reinterpret_cast<__gm__ float*>(workspaceGm), tiling);
    mm.Process(static_cast<uint32_t>(AscendC::GetBlockIdx()), blocks);
}
/** Low-rank direct-H consumer: scale, merge beta*C and publish one triangle. */
template <uint32_t PRODUCER_TILE>
__aicore__ inline void ProcessDirectEpilogueTask(
    uint32_t producerRow, uint32_t producerCol, float alpha, float beta, bool hasProduct, bool upper,
    AscendC::GlobalTensor<float>& c, AscendC::GlobalTensor<float>& workspace, const CherkArch22TilingData& tiling,
    CherkDirectEpilogueBuffers& buffers, event_t mte2ToV, event_t vToMte2, event_t vToMte3, event_t mte3ToV)
{
    constexpr uint32_t parts = PRODUCER_TILE / DIRECT_EPILOGUE_TILE;
    for (uint32_t rowPart = 0U; rowPart < parts; ++rowPart) {
        for (uint32_t colPart = 0U; colPart < parts; ++colPart) {
            const uint32_t rowBase = producerRow + rowPart * DIRECT_EPILOGUE_TILE;
            const uint32_t colBase = producerCol + colPart * DIRECT_EPILOGUE_TILE;
            if (rowBase >= tiling.n || colBase >= tiling.n) {
                continue;
            }
            if (PRODUCER_TILE != DIRECT_EPILOGUE_TILE) {
                const bool selected = upper ? rowBase <= colBase : rowBase >= colBase;
                if (!selected) {
                    continue;
                }
            }
            const uint32_t validRows = Min<uint32_t>(DIRECT_EPILOGUE_TILE, tiling.n - rowBase);
            const uint32_t validCols = Min<uint32_t>(DIRECT_EPILOGUE_TILE, tiling.n - colBase);
            ProcessDirectEpilogueTile(
                alpha, beta, hasProduct, c, workspace, tiling, rowBase, colBase, validRows, validCols,
                rowBase == colBase, buffers, mte2ToV, vToMte2, vToMte3, mte3ToV);
        }
    }
}

template <uint32_t PRODUCER_TILE>
__aicore__ inline void RunDirectEpilogueTasks(
    uint64_t taskBegin, uint64_t taskEnd, uint32_t tileAxis, bool upper, float alpha, float beta, bool hasProduct,
    AscendC::GlobalTensor<float>& c, AscendC::GlobalTensor<float>& workspace, const CherkArch22TilingData& tiling,
    CherkDirectEpilogueBuffers& buffers, const CherkCopyEvents& events)
{
    CherkTriangleTaskCursor cursor = InitCherkTriangleCursor(taskBegin, tileAxis, upper);
    for (uint64_t task = taskBegin; task < taskEnd; ++task) {
        const uint32_t colTile = ResolveCherkTriangleCol(cursor, task);
        ProcessDirectEpilogueTask<PRODUCER_TILE>(
            cursor.rowTile * PRODUCER_TILE, colTile * PRODUCER_TILE, alpha, beta, hasProduct, upper, c, workspace,
            tiling, buffers, events.mte2ToV, events.vToMte2, events.vToMte3, events.mte3ToV);
    }
}

template <uint32_t PRODUCER_TILE>
__aicore__ inline void RunDirectEpilogue(
    GM_ADDR alphaGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    if (alpha == 0.0f && beta == 1.0f) {
        return;
    }
    const bool hasProduct = alpha != 0.0f;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> planarBuf, outputBuf, oldBuf, interleaveOffsetBuf, diagonalMaskBuf,
        triangleMaskBuf;
    CherkDirectEpilogueBuffers buffers = InitDirectEpilogueBuffers(
        pipe, planarBuf, outputBuf, oldBuf, interleaveOffsetBuf, diagonalMaskBuf, triangleMaskBuf);
    LoadDirectEpilogueMetadata(metadataGm, tiling.uplo, buffers);

    AscendC::GlobalTensor<float> c;
    AscendC::GlobalTensor<float> workspace;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2ULL);
    workspace.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspaceGm), CherkArch22DirectWorkspaceFloats(tiling.cubeN, tiling.cubeK));

    uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    blockNum = CherkNormalizeBlockCount(blockNum);
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t tileAxis = (tiling.n + PRODUCER_TILE - 1U) / PRODUCER_TILE;
    const uint64_t taskCount = static_cast<uint64_t>(tileAxis) * (tileAxis + 1U) / 2ULL;
    const uint64_t taskBegin = static_cast<uint64_t>(blockIdx) * taskCount / blockNum;
    const uint64_t taskEnd = static_cast<uint64_t>(blockIdx + 1U) * taskCount / blockNum;
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    const CherkCopyEvents events = GetCherkCopyEvents();
    RunDirectEpilogueTasks<PRODUCER_TILE>(
        taskBegin, taskEnd, tileAxis, upper, alpha, beta, hasProduct, c, workspace, tiling, buffers, events);
}

extern "C" __global__ __aicore__ void cherk_arch22_direct_epilogue_kernel(
    GM_ADDR alphaGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunDirectEpilogue<DIRECT_EPILOGUE_TILE>(alphaGm, betaGm, cGm, workspaceGm, metadataGm, tiling);
}

// Producer-aligned consumer for direct-H standalone launches.  The Cube
// producer publishes 128x128 tiles, while the common epilogue schedules every
// 64x64 quadrant as an independent block.
extern "C" __global__ __aicore__ void cherk_arch22_direct_epilogue128_kernel(
    GM_ADDR alphaGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunDirectEpilogue<128U>(alphaGm, betaGm, cGm, workspaceGm, metadataGm, tiling);
}
void cherk_arch22_direct_compute_do(
    GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData& tiling,
    void* stream)
{
    cherk_arch22_direct_mmad_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(workspace, tiling);
    const bool use128Tile = tiling.n >= 768U && tiling.n < 1200U;
    const uint32_t tileSize = use128Tile ? 128U : 64U;
    const uint32_t axis = (tiling.n + tileSize - 1U) / tileSize;
    const uint64_t tasks = static_cast<uint64_t>(axis) * (axis + 1U) / 2ULL;
    const uint32_t blocks = static_cast<uint32_t>(tasks < tiling.aivCoreNum ? tasks : tiling.aivCoreNum);
    if (use128Tile) {
        cherk_arch22_direct_epilogue128_kernel<<<blocks, nullptr, stream>>>(
            alpha, beta, c, workspace, metadata, tiling);
    } else {
        cherk_arch22_direct_epilogue_kernel<<<blocks, nullptr, stream>>>(alpha, beta, c, workspace, metadata, tiling);
    }
}

/** Direct-H AIC/AIV producer-consumer paths for low-rank K=64 and square-64. */
__aicore__ inline void CherkArch22DirectMixPack(
    GM_ADDR aGm, GM_ADDR workspaceGm, GM_ADDR metadataGm, const CherkArch22TilingData& tiling, uint32_t blockIndex,
    uint32_t groupCount)
{
    AscendC::TPipe packPipe;
    if (tiling.trans == ACLBLAS_OP_N) {
        CherkArch22DirectPackNImpl(aGm, workspaceGm, tiling, blockIndex, groupCount * 2U, packPipe);
    } else {
        CherkArch22DirectPackCImpl(aGm, workspaceGm, metadataGm, tiling, blockIndex, groupCount * 2U, packPipe);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    packPipe.Reset();
}

struct DirectMixConsumerState {
    float alpha;
    float beta;
    bool unitUpper;
    bool upper;
    AscendC::GlobalTensor<float>& c;
    AscendC::GlobalTensor<float>& workspace;
    const CherkArch22TilingData& tiling;
    CherkDirectEpilogueBuffers& buffers;
    event_t mte2ToV;
    event_t vToMte2;
    event_t vToMte3;
    event_t mte3ToV;
};

__aicore__ inline void ProcessDirectMixTile(
    DirectMixConsumerState& state, uint32_t logicalRowBase, uint32_t logicalColBase);

struct DirectMixSchedule {
    uint64_t taskBegin;
    uint64_t taskEnd;
    uint64_t rowTaskBase;
    uint32_t tileAxis;
    uint32_t mTile;
    uint32_t subBlock;
    uint32_t groupCount;
    bool strided;
    bool upper;
};

__aicore__ inline DirectMixSchedule InitDirectMixSchedule(
    uint32_t groupIndex, uint32_t groupCount, uint32_t tileAxis, uint32_t subBlock, bool strided, bool upper)
{
    groupCount = groupCount == 0U ? 1U : groupCount;
    const uint64_t totalTasks = static_cast<uint64_t>(tileAxis) * (tileAxis + 1U) / 2ULL;
    const uint64_t totalWorkUnits = strided ? totalTasks : 2ULL * totalTasks;
    const uint64_t taskBegin = strided ? groupIndex : static_cast<uint64_t>(groupIndex) * totalWorkUnits / groupCount;
    const uint64_t taskEnd =
        strided ? totalTasks : static_cast<uint64_t>(groupIndex + 1U) * totalWorkUnits / groupCount;
    DirectMixSchedule schedule{taskBegin, taskEnd, 0ULL, tileAxis, 0U, subBlock, groupCount, strided, upper};
    const uint64_t firstTask = strided ? taskBegin : taskBegin / 2ULL;
    while (schedule.mTile < tileAxis) {
        const uint32_t rowTasks = upper ? schedule.mTile + 1U : tileAxis - schedule.mTile;
        if (firstTask < schedule.rowTaskBase + rowTasks) {
            break;
        }
        schedule.rowTaskBase += rowTasks;
        ++schedule.mTile;
    }
    return schedule;
}

__aicore__ inline uint64_t ProcessDirectMixWork(
    DirectMixConsumerState& state, DirectMixSchedule& schedule, uint64_t workUnit, uint32_t iteration)
{
    const uint64_t task = schedule.strided ? workUnit : workUnit / 2ULL;
    const uint32_t half = schedule.strided ? 0U : static_cast<uint32_t>(workUnit & 1ULL);
    const bool merge = schedule.strided || (half == 0U && workUnit + 1ULL < schedule.taskEnd);
    uint32_t rowTasks = schedule.upper ? schedule.mTile + 1U : schedule.tileAxis - schedule.mTile;
    while (task >= schedule.rowTaskBase + rowTasks) {
        schedule.rowTaskBase += rowTasks;
        ++schedule.mTile;
        rowTasks = schedule.upper ? schedule.mTile + 1U : schedule.tileAxis - schedule.mTile;
    }
    const uint32_t nBegin = schedule.upper ? 0U : schedule.mTile;
    const uint32_t nTile = nBegin + static_cast<uint32_t>(task - schedule.rowTaskBase);
    const uint16_t slot = static_cast<uint16_t>(iteration % DIRECT_MIX_FLAG_COUNT);
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(slot);
    if (state.alpha != 0.0f || state.beta != 1.0f) {
        constexpr uint32_t producerTile = 128U;
        const uint32_t rowBase = nTile * producerTile + (merge ? schedule.subBlock : half) * DIRECT_EPILOGUE_TILE;
        if (merge || schedule.subBlock == half) {
            const uint32_t parts = merge ? 2U : 1U;
            for (uint32_t part = 0U; part < parts; ++part) {
                const uint32_t colBase = schedule.mTile * producerTile + (merge ? part * DIRECT_EPILOGUE_TILE : 0U);
                ProcessDirectMixTile(state, rowBase, colBase);
            }
        }
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(static_cast<uint16_t>(DIRECT_MIX_ACK_BASE + slot));
    return schedule.strided ? schedule.groupCount : (merge ? 2ULL : 1ULL);
}

template <bool UNIT_UPPER>
__aicore__ inline void RunDirectMixConsumer(
    GM_ADDR alphaGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData& tiling, uint32_t blockIndex, uint32_t groupCount, bool stridedMixed)
{
    const float alpha = UNIT_UPPER ? 1.0f : *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = UNIT_UPPER ? 0.0f : *reinterpret_cast<__gm__ const float*>(betaGm);
    const bool unitUpper = UNIT_UPPER || (tiling.uplo == ACLBLAS_UPPER && alpha == 1.0f && beta == 0.0f);
    const uint32_t groupIndex = blockIndex / 2U;
    const uint32_t subBlock = static_cast<uint32_t>(AscendC::GetSubBlockIdx()) & 1U;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> planarBuf, outputBuf, oldBuf, interleaveOffsetBuf, diagonalMaskBuf,
        triangleMaskBuf;
    CherkDirectEpilogueBuffers buffers = InitDirectEpilogueBuffers(
        pipe, planarBuf, outputBuf, oldBuf, interleaveOffsetBuf, diagonalMaskBuf, triangleMaskBuf);
    LoadDirectEpilogueMetadata(metadataGm, tiling.uplo, buffers, !unitUpper);
    AscendC::GlobalTensor<float> c;
    AscendC::GlobalTensor<float> workspace;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2ULL);
    workspace.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspaceGm), CherkArch22DirectWorkspaceFloats(tiling.cubeN, tiling.cubeK));
    const CherkCopyEvents events = GetCherkCopyEvents();
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    DirectMixConsumerState state{alpha,          beta,          unitUpper, upper,          c,
                                 workspace,      tiling,        buffers,   events.mte2ToV, events.vToMte2,
                                 events.vToMte3, events.mte3ToV};
    DirectMixSchedule schedule =
        InitDirectMixSchedule(groupIndex, groupCount, tiling.cubeN / 128U, subBlock, stridedMixed, upper);
    uint32_t iteration = 0U;
    for (uint64_t work = schedule.taskBegin; work < schedule.taskEnd; ++iteration) {
        work += ProcessDirectMixWork(state, schedule, work, iteration);
    }
}

__aicore__ inline void ApplyDirectMixScalars(DirectMixConsumerState& state, bool loadOld, bool diagonalTile)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(state.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(state.mte2ToV);
    if (state.unitUpper) {
        ApplyDirectUnitUpper(diagonalTile, state.buffers);
    } else {
        ApplyDirectScalars(state.alpha, state.beta, loadOld, diagonalTile, state.buffers);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(state.vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(state.vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(state.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(state.vToMte3);
}

__aicore__ inline void ProcessDirectMixTile(
    DirectMixConsumerState& state, uint32_t logicalRowBase, uint32_t logicalColBase)
{
    if (logicalRowBase >= state.tiling.n || logicalColBase >= state.tiling.n) {
        return;
    }
    const bool selectedTile = state.upper ? logicalRowBase <= logicalColBase : logicalRowBase >= logicalColBase;
    if (!selectedTile) {
        return;
    }
    const uint32_t validRows = Min<uint32_t>(DIRECT_EPILOGUE_TILE, state.tiling.n - logicalRowBase);
    const uint32_t validCols = Min<uint32_t>(DIRECT_EPILOGUE_TILE, state.tiling.n - logicalColBase);
    const bool diagonalTile = logicalRowBase == logicalColBase;
    const bool fullDiagonal = diagonalTile && validRows == DIRECT_EPILOGUE_TILE && validCols == DIRECT_EPILOGUE_TILE;
    const bool loadOld =
        !state.unitUpper && (state.beta != 0.0f || (diagonalTile && state.tiling.uplo == ACLBLAS_LOWER) ||
                             (fullDiagonal && state.tiling.uplo == ACLBLAS_UPPER));
    if (state.alpha != 0.0f) {
        LoadDirectProduct(
            state.workspace, state.tiling, logicalRowBase, logicalColBase, validRows, validCols, state.buffers);
    }
    if (loadOld) {
        LoadDirectOldC(state.c, state.tiling, logicalRowBase, logicalColBase, validRows, validCols, state.buffers);
    }
    ApplyDirectMixScalars(state, loadOld, diagonalTile);
    if (state.unitUpper) {
        StoreDirectUnitUpperTile(
            state.c, state.tiling, logicalRowBase, logicalColBase, validRows, validCols, diagonalTile, state.buffers);
    } else {
        StoreDirectTile(
            state.c, state.tiling, logicalRowBase, logicalColBase, validRows, validCols, diagonalTile, loadOld,
            state.vToMte3, state.mte3ToV, state.buffers);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(state.mte3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(state.mte3ToV);
}

template <bool UNIT_UPPER>
__aicore__ inline void CherkArch22DirectMixImpl(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    const uint32_t blockIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t groupCount = tiling.aicCoreNum == 0U ? 1U : tiling.aicCoreNum;
    const bool fusedPack = tiling.directMixMode != 3U;
    const bool stridedMixed = tiling.directMixMode >= 2U;
    if ASCEND_IS_AIV {
        // Every fused low-K MIX launch consumes the planar workspace produced
        // by the AIV pack stage.  The n<512 bucket is deliberately fused too;
        // skipping pack there leaves the Cube producer reading stale GM.
        if (fusedPack) {
            CherkArch22DirectMixPack(aGm, workspaceGm, metadataGm, tiling, blockIndex, groupCount);
        }
    }
    // Keep repeated stream launches from entering the ring with different
    // producer/consumer phases. The barrier is outside the tile loop and does
    // not serialize the steady-state ready/ACK pipeline.
    if (fusedPack) {
        AscendC::SyncAll<false>();
    }
    // Match CHER2K's mode-2 topology: every resident AIC group owns a
    // disjoint slice of the triangular stream and its two AIV consumers drain
    // the same producer tile.  Flag slots are scoped by the MIX group.
    if ASCEND_IS_AIC {
        AscendC::InitSocState();
        CherkArch22DirectMmad producer;
        producer.Init(reinterpret_cast<__gm__ float*>(workspaceGm), tiling);
        producer.Process(blockIndex, groupCount, true, stridedMixed);
        return;
    }

    if ASCEND_IS_AIV {
        RunDirectMixConsumer<UNIT_UPPER>(
            alphaGm, betaGm, cGm, workspaceGm, metadataGm, tiling, blockIndex, groupCount, stridedMixed);
    }
}

extern "C" __global__ __aicore__ __schedmode__(1) void cherk_arch22_direct_mix_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    CherkArch22DirectMixImpl<false>(alphaGm, aGm, betaGm, cGm, workspaceGm, metadataGm, tiling);
}

void cherk_arch22_direct_pack_do(
    GM_ADDR a, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData& tiling, void* stream)
{
    if (tiling.trans == ACLBLAS_OP_N && tiling.cubeK == 64U && tiling.n >= 512U) {
        const uint32_t batchRows = tiling.directPackBatchRows;
        const uint32_t batchCount = (tiling.cubeK + batchRows - 1U) / batchRows;
        const uint32_t blocks = tiling.aivCoreNum < batchCount ? tiling.aivCoreNum : batchCount;
        cherk_arch22_direct_pack_n_batch_kernel<<<blocks, nullptr, stream>>>(a, workspace, tiling);
        return;
    }
    if (tiling.trans == ACLBLAS_OP_N) {
        const uint32_t blocks = tiling.cubeK < 40U ? tiling.cubeK : 40U;
        cherk_arch22_direct_pack_n_kernel<<<blocks, nullptr, stream>>>(a, workspace, tiling);
        return;
    }
    if (tiling.cubeK == 64U && tiling.n >= 512U) {
        constexpr uint32_t batchCols = 128U;
        constexpr uint32_t batchRows = 32U;
        const uint32_t taskCount = (tiling.cubeK / batchRows) * ((tiling.cubeN + batchCols - 1U) / batchCols);
        const uint32_t blocks = tiling.aivCoreNum < taskCount ? tiling.aivCoreNum : taskCount;
        cherk_arch22_direct_pack_c_batch_kernel<<<blocks, nullptr, stream>>>(a, workspace, metadata, tiling);
        return;
    }
    const uint64_t packTasks =
        static_cast<uint64_t>(tiling.cubeK / 32U) * static_cast<uint64_t>((tiling.cubeN + 63U) / 64U);
    const uint64_t usableTasks = packTasks == 0ULL ? 1ULL : packTasks;
    const uint32_t blocks = static_cast<uint32_t>(tiling.aivCoreNum < usableTasks ? tiling.aivCoreNum : usableTasks);
    cherk_arch22_direct_pack_c_kernel<<<blocks, nullptr, stream>>>(a, workspace, metadata, tiling);
}

void cherk_arch22_direct_mix_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream)
{
    cherk_arch22_direct_mix_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(
        alpha, a, beta, c, workspace, metadata, tiling);
}

/** True 64x64 single-launch path for UPPER/OP_N/n=k=64. */
namespace {
constexpr uint32_t SQUARE64_N = 64U;
constexpr uint32_t SQUARE64_ELEMENTS = SQUARE64_N * SQUARE64_N;
constexpr uint32_t SQUARE64_PACK_ROWS = SQUARE64_N / 2U;
constexpr uint32_t SQUARE64_PACK_ELEMENTS = SQUARE64_PACK_ROWS * SQUARE64_N;
constexpr uint16_t SQUARE64_READY_FLAG = 0U;

__aicore__ inline void Square64LoadBoth(
    AscendC::LocalTensor<float>& a2, AscendC::LocalTensor<float>& b2, AscendC::LocalTensor<float>& left,
    AscendC::LocalTensor<float>& right, const AscendC::LoadData3DParamsV2<float>& load)
{
    AscendC::LoadData(b2, right, load);
    AscendC::LoadData(a2, left, load);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
}

__aicore__ inline void Square64Compute(
    AscendC::LocalTensor<float>& output, AscendC::LocalTensor<float>& a2, AscendC::LocalTensor<float>& b2,
    const AscendC::MmadParams& params)
{
    AscendC::Mmad(output, a2, b2, params);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void InitSquare64MmadParams(
    AscendC::Nd2NzParams& copy, AscendC::LoadData3DParamsV2<float>& load, AscendC::MmadParams& first,
    AscendC::MmadParams& accum)
{
    copy.ndNum = 1U;
    copy.nValue = SQUARE64_N;
    copy.dValue = SQUARE64_N;
    copy.srcDValue = SQUARE64_N;
    copy.dstNzC0Stride = SQUARE64_N;
    copy.dstNzNStride = 1U;
    load.l1H = 1U;
    load.l1W = SQUARE64_N;
    load.channelSize = SQUARE64_N;
    load.kExtension = SQUARE64_N;
    load.mExtension = SQUARE64_N;
    load.strideW = 1U;
    load.strideH = 1U;
    load.filterW = 1U;
    load.filterH = 1U;
    load.dilationFilterW = 1U;
    load.dilationFilterH = 1U;
    load.enTranspose = true;
    first.m = SQUARE64_N;
    first.n = SQUARE64_N;
    first.k = SQUARE64_N;
    first.cmatrixInitVal = true;
    accum = first;
    accum.cmatrixInitVal = false;
}

__aicore__ inline void StoreSquare64Mmad(
    AscendC::GlobalTensor<float>& workspace, AscendC::LocalTensor<float>& realC1, AscendC::LocalTensor<float>& imagC1)
{
    AscendC::FixpipeParamsV220 store{};
    store.mSize = SQUARE64_N;
    store.nSize = SQUARE64_N;
    store.srcStride = SQUARE64_N;
    store.dstStride = SQUARE64_N;
    store.ndNum = 1U;
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::Fixpipe(workspace[CherkArch22DirectOffsetRealOutput(SQUARE64_N, SQUARE64_N)], realC1, store);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::Fixpipe(workspace[CherkArch22DirectOffsetImagOutput(SQUARE64_N, SQUARE64_N)], imagC1, store);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SQUARE64_READY_FLAG);
}

__aicore__ inline void CherkArch22Square64Mmad(GM_ADDR workspaceGm)
{
    constexpr uint32_t matrixBytes = SQUARE64_ELEMENTS * sizeof(float);
    AscendC::LocalTensor<float> aReal1(AscendC::TPosition::A1, 0U, SQUARE64_ELEMENTS);
    AscendC::LocalTensor<float> aImag1(AscendC::TPosition::A1, matrixBytes, SQUARE64_ELEMENTS);
    constexpr uint32_t bBase = matrixBytes * 2U;
    AscendC::LocalTensor<float> bReal1(AscendC::TPosition::B1, bBase, SQUARE64_ELEMENTS);
    AscendC::LocalTensor<float> bImag1(AscendC::TPosition::B1, bBase + matrixBytes, SQUARE64_ELEMENTS);
    AscendC::LocalTensor<float> a2(AscendC::TPosition::A2, 0U, SQUARE64_ELEMENTS);
    AscendC::LocalTensor<float> b2(AscendC::TPosition::B2, 0U, SQUARE64_ELEMENTS);
    AscendC::LocalTensor<float> realC1(AscendC::TPosition::CO1, 0U, SQUARE64_ELEMENTS);
    AscendC::LocalTensor<float> imagC1(AscendC::TPosition::CO1, matrixBytes, SQUARE64_ELEMENTS);

    AscendC::GlobalTensor<float> workspace;
    workspace.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspaceGm), CherkArch22DirectWorkspaceFloats(SQUARE64_N, SQUARE64_N));
    AscendC::Nd2NzParams copy{};
    AscendC::LoadData3DParamsV2<float> load{};
    AscendC::MmadParams first{};
    AscendC::MmadParams accum{};
    InitSquare64MmadParams(copy, load, first, accum);

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::DataCopy(aReal1, workspace[CherkArch22DirectOffsetRealInput(SQUARE64_N, SQUARE64_N)], copy);
    AscendC::DataCopy(aImag1, workspace[CherkArch22DirectOffsetImagInput(SQUARE64_N, SQUARE64_N)], copy);
    AscendC::DataCopy(bReal1, workspace[CherkArch22DirectOffsetRealInput(SQUARE64_N, SQUARE64_N)], copy);
    AscendC::DataCopy(bImag1, workspace[CherkArch22DirectOffsetImagInput(SQUARE64_N, SQUARE64_N)], copy);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

    // H = A*A^H needs three real products: two terms accumulate directly in
    // the real plane and Q=R^T*I is published once.  The AIV reconstructs the
    // imaginary plane as Q-Q^T, eliminating the fourth MMAD and the -I plane.
    Square64LoadBoth(a2, b2, aReal1, bReal1, load);
    Square64Compute(realC1, a2, b2, first);
    Square64LoadBoth(a2, b2, aImag1, bImag1, load);
    Square64Compute(realC1, a2, b2, accum);
    Square64LoadBoth(a2, b2, aReal1, bImag1, load);
    Square64Compute(imagC1, a2, b2, first);

    StoreSquare64Mmad(workspace, realC1, imagC1);
}

__aicore__ inline void CherkArch22Square64Pack(GM_ADDR aGm, GM_ADDR workspaceGm, const CherkArch22TilingData& tiling)
{
    const uint32_t packedRowBase = static_cast<uint32_t>(AscendC::GetSubBlockIdx()) * SQUARE64_PACK_ROWS;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf, planarBuf;
    pipe.InitBuffer(aosBuf, SQUARE64_PACK_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(planarBuf, SQUARE64_PACK_ELEMENTS * 2U * sizeof(float));
    AscendC::LocalTensor<float> aos = aosBuf.Get<float>();
    AscendC::LocalTensor<float> real = planarBuf.Get<float>();
    AscendC::LocalTensor<float> imag = real[SQUARE64_PACK_ELEMENTS];
    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> workspace;
    InitCherkFloatBuffers(
        a, workspace, aGm, workspaceGm, static_cast<uint64_t>(tiling.lda) * SQUARE64_N * 2ULL,
        CherkArch22DirectWorkspaceFloats(SQUARE64_N, SQUARE64_N));
    const AscendC::DataCopyExtParams copy{
        SQUARE64_PACK_ROWS, SQUARE64_N * 2U * sizeof(float),
        static_cast<uint32_t>((tiling.lda - SQUARE64_N) * 2U * sizeof(float)), 0U, 0U};
    AscendC::DataCopyPad(aos, a[static_cast<uint64_t>(packedRowBase) * tiling.lda * 2ULL], copy, {false, 0U, 0U, 0.0f});
    const CherkCopyEvents events = GetCherkCopyEvents();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(events.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(events.mte2ToV);
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = SQUARE64_PACK_ELEMENTS * 2U / 64U;
    AscendC::GatherMask(real, aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask(imag, aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(events.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(events.vToMte3);
    const uint64_t outputOffset = static_cast<uint64_t>(packedRowBase) * SQUARE64_N;
    AscendC::DataCopy(
        workspace[CherkArch22DirectOffsetRealInput(SQUARE64_N, SQUARE64_N) + outputOffset], real,
        SQUARE64_PACK_ELEMENTS);
    AscendC::DataCopy(
        workspace[CherkArch22DirectOffsetImagInput(SQUARE64_N, SQUARE64_N) + outputOffset], imag,
        SQUARE64_PACK_ELEMENTS);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(events.mte3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(events.mte3ToV);
}

__aicore__ inline void LoadSquare64Metadata(
    GM_ADDR metadataGm, CherkDirectEpilogueBuffers& buffers, AscendC::LocalTensor<uint32_t>& transposeOffsets)
{
    AscendC::GlobalTensor<uint32_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
    AscendC::GlobalTensor<uint8_t> metadataBytes;
    metadataBytes.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t*>(metadataGm), CHERK_ARCH22_META_WORDS * sizeof(uint32_t));
    AscendC::DataCopy(
        transposeOffsets, metadata[CHERK_ARCH22_META_TRANSPOSE_OFFSET], CHERK_ARCH22_META_TRANSPOSE_COUNT);
    AscendC::DataCopy(
        buffers.interleaveOffsets, metadata[CHERK_ARCH22_META_INTERLEAVE_OFFSET], CHERK_ARCH22_META_INTERLEAVE_COUNT);
    AscendC::DataCopy(
        buffers.diagonalMask, metadataBytes[CHERK_ARCH22_META_DIAGONAL_OFFSET * sizeof(uint32_t)],
        DIRECT_EPILOGUE_ELEMENTS / 8U);
    AscendC::DataCopy(
        buffers.triangleMask, metadataBytes[CHERK_ARCH22_META_LOWER_OFFSET * sizeof(uint32_t)],
        DIRECT_EPILOGUE_ELEMENTS / 8U);
}

__aicore__ inline void ProcessSquare64Result(
    float alpha, float beta, GM_ADDR cGm, GM_ADDR workspaceGm, const CherkArch22TilingData& tiling,
    CherkDirectEpilogueBuffers& buffers, AscendC::LocalTensor<uint32_t>& transposeOffsets)
{
    CherkArch22TilingData squareTiling = tiling;
    squareTiling.cubeN = SQUARE64_N;
    squareTiling.cubeK = SQUARE64_N;
    AscendC::GlobalTensor<float> c;
    AscendC::GlobalTensor<float> workspace;
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm), static_cast<uint64_t>(tiling.ldc) * SQUARE64_N * 2ULL);
    workspace.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspaceGm), CherkArch22DirectWorkspaceFloats(SQUARE64_N, SQUARE64_N));
    LoadDirectOldC(c, squareTiling, 0U, 0U, SQUARE64_N, SQUARE64_N, buffers);
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(SQUARE64_READY_FLAG);
    LoadDirectProduct(workspace, squareTiling, 0U, 0U, SQUARE64_N, SQUARE64_N, buffers);
    const CherkCopyEvents events = GetCherkCopyEvents();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(events.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(events.mte2ToV);
    AscendC::LocalTensor<float> crossTranspose = buffers.outputAos;
    AscendC::Gather(crossTranspose, buffers.imag, transposeOffsets, 0U, DIRECT_EPILOGUE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(buffers.imag, buffers.imag, crossTranspose, DIRECT_EPILOGUE_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    ApplyDirectScalars(alpha, beta, true, true, buffers);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(events.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(events.vToMte3);
    StoreDirectTile(
        c, squareTiling, 0U, 0U, SQUARE64_N, SQUARE64_N, true, true, events.vToMte3, events.mte3ToV, buffers);
}

__aicore__ inline void CherkArch22Square64Consume(
    GM_ADDR alphaGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData& tiling)
{
    if (static_cast<uint32_t>(AscendC::GetSubBlockIdx()) != 0U) {
        return;
    }
    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> planarBuf, outputBuf, oldBuf, interleaveOffsetBuf, transposeOffsetBuf,
        diagonalMaskBuf, triangleMaskBuf;
    CherkDirectEpilogueBuffers buffers = InitDirectEpilogueBuffers(
        pipe, planarBuf, outputBuf, oldBuf, interleaveOffsetBuf, diagonalMaskBuf, triangleMaskBuf);
    pipe.InitBuffer(transposeOffsetBuf, CHERK_ARCH22_META_TRANSPOSE_COUNT * sizeof(uint32_t));
    AscendC::LocalTensor<uint32_t> transposeOffsets = transposeOffsetBuf.Get<uint32_t>();
    LoadSquare64Metadata(metadataGm, buffers, transposeOffsets);
    ProcessSquare64Result(alpha, beta, cGm, workspaceGm, tiling, buffers, transposeOffsets);
}
} // namespace

extern "C" __global__ __aicore__ __schedmode__(1) void cherk_arch22_square64_mix_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR workspaceGm, GM_ADDR metadataGm,
    const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        CherkArch22Square64Pack(aGm, workspaceGm, tiling);
    }
    AscendC::SyncAll<false>();

    if ASCEND_IS_AIC {
        AscendC::InitSocState();
        CherkArch22Square64Mmad(workspaceGm);
        return;
    }
    if ASCEND_IS_AIV {
        CherkArch22Square64Consume(alphaGm, betaGm, cGm, workspaceGm, metadataGm, tiling);
    }
}

void cherk_arch22_square64_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream)
{
    cherk_arch22_square64_mix_kernel<<<1U, nullptr, stream>>>(alpha, a, beta, c, workspace, metadata, tiling);
}

namespace {
struct PackNContext {
    AscendC::TQue<AscendC::TPosition::VECIN, 1>& inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1>& outputQueue;
    AscendC::GlobalTensor<float>& a;
    AscendC::GlobalTensor<float>& workspace;
    const CherkArch22TilingData& tiling;
    uint64_t realBase;
    uint64_t imagBase;
};

__aicore__ inline void ProcessPackNChunk(PackNContext& ctx, uint32_t packedRow, uint32_t offset)
{
    const uint32_t chunk = Min<uint32_t>(PACK_COMPLEX, ctx.tiling.cubeN - offset);
    const uint32_t valid =
        packedRow < ctx.tiling.k && offset < ctx.tiling.n ? Min<uint32_t>(chunk, ctx.tiling.n - offset) : 0U;
    AscendC::LocalTensor<float> aos =
        LoadCherkPackNInput(ctx.inputQueue, ctx.a, ctx.tiling, packedRow, offset, chunk, valid);
    ctx.inputQueue.EnQue(aos);
    AscendC::LocalTensor<float> source = ctx.inputQueue.DeQue<float>();
    AscendC::LocalTensor<float> packed = ctx.outputQueue.AllocTensor<float>();
    AscendC::LocalTensor<float> real = packed;
    AscendC::LocalTensor<float> imag = packed[PACK_COMPLEX];
    const uint16_t repeats = static_cast<uint16_t>((chunk * 2U) / 64U);
    uint64_t reserved = 0ULL;
    AscendC::GatherMask<float>(real, source, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(imag, source, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    ctx.outputQueue.EnQue(packed);
    ctx.inputQueue.FreeTensor(source);
    AscendC::LocalTensor<float> output = ctx.outputQueue.DeQue<float>();
    const uint64_t outputOffset = static_cast<uint64_t>(packedRow) * ctx.tiling.cubeN + offset;
    AscendC::DataCopy(ctx.workspace[ctx.realBase + outputOffset], output, chunk);
    AscendC::DataCopy(ctx.workspace[ctx.imagBase + outputOffset], output[PACK_COMPLEX], chunk);
    ctx.outputQueue.FreeTensor(output);
}
} // namespace

extern "C" __global__ __aicore__ void cherk_arch22_pack_n_kernel(
    GM_ADDR aGm, GM_ADDR workspaceGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    pipe.InitBuffer(inputQueue, 1, PACK_COMPLEX * 2U * sizeof(float));
    pipe.InitBuffer(outputQueue, 1, PACK_COMPLEX * 2U * sizeof(float));

    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> workspace;
    InitCherkFloatBuffers(
        a, workspace, aGm, workspaceGm, static_cast<uint64_t>(tiling.lda) * tiling.k * 2U, WorkspaceCount(tiling));

    const uint64_t realBase = CherkArch22OffsetRealInput(tiling.cubeN, tiling.cubeK);
    const uint64_t imagBase = CherkArch22OffsetImagInput(tiling.cubeN, tiling.cubeK);
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockNum = CherkNormalizeBlockCount(static_cast<uint32_t>(AscendC::GetBlockNum()));

    PackNContext context{inputQueue, outputQueue, a, workspace, tiling, realBase, imagBase};
    for (uint32_t packedRow = blockIdx; packedRow < tiling.cubeK; packedRow += blockNum) {
        for (uint32_t offset = 0U; offset < tiling.cubeN; offset += PACK_COMPLEX) {
            ProcessPackNChunk(context, packedRow, offset);
        }
    }
}
/** OP_C preprocess: transpose source N x K rows into the shared Kp x Np contract. */
namespace {
constexpr uint32_t PACK_C_ROWS = 32U;
constexpr uint32_t PACK_C_COLS = 64U;
constexpr uint32_t PACK_C_ELEMENTS = PACK_C_ROWS * PACK_C_COLS;

struct PackCContext {
    AscendC::LocalTensor<float>& aos;
    AscendC::LocalTensor<float>& real;
    AscendC::LocalTensor<float>& imag;
    AscendC::LocalTensor<float>& transReal;
    AscendC::LocalTensor<float>& transImag;
    AscendC::LocalTensor<uint32_t>& offsets;
    AscendC::GlobalTensor<float>& a;
    AscendC::GlobalTensor<float>& workspace;
    const CherkArch22TilingData& tiling;
    uint64_t realBase;
    uint64_t imagBase;
    event_t vToMte2;
    event_t mte2ToV;
    event_t vToMte3;
    event_t mte3ToV;
};

__aicore__ inline void ProcessPackCTile(PackCContext& ctx, uint32_t tile, uint32_t colTiles)
{
    colTiles = colTiles == 0U ? 1U : colTiles;
    const uint32_t rowBase = (tile / colTiles) * PACK_C_ROWS;
    const uint32_t colBase = (tile % colTiles) * PACK_C_COLS;
    const uint32_t validRows = rowBase < ctx.tiling.k ? Min<uint32_t>(PACK_C_ROWS, ctx.tiling.k - rowBase) : 0U;
    const uint32_t validCols = colBase < ctx.tiling.n ? Min<uint32_t>(PACK_C_COLS, ctx.tiling.n - colBase) : 0U;

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::Duplicate(ctx.aos, 0.0f, PACK_C_ELEMENTS * 2U);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    if (validRows != 0U && validCols != 0U) {
        const uint32_t blockBytes = validRows * 2U * sizeof(float);
        const uint32_t alignedBytes = (blockBytes + 31U) & ~31U;
        const AscendC::DataCopyExtParams inputCopy{
            static_cast<uint16_t>(validCols), blockBytes,
            static_cast<uint32_t>((ctx.tiling.lda - validRows) * 2U * sizeof(float)),
            static_cast<uint32_t>((PACK_C_ROWS * 2U * sizeof(float) - alignedBytes) / 32U), 0U};
        const AscendC::DataCopyPadExtParams<float> inputPad{
            alignedBytes != blockBytes, 0U, static_cast<uint8_t>((alignedBytes - blockBytes) / sizeof(float)), 0.0f};
        const uint64_t inputOffset = (static_cast<uint64_t>(colBase) * ctx.tiling.lda + rowBase) * 2ULL;
        AscendC::DataCopyPad(ctx.aos, ctx.a[inputOffset], inputCopy, inputPad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV);

    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = PACK_C_ELEMENTS * 2U / 64U;
    AscendC::GatherMask<float>(ctx.real, ctx.aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(ctx.imag, ctx.aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
    AscendC::Gather(ctx.transReal, ctx.real, ctx.offsets, 0U, PACK_C_ELEMENTS);
    AscendC::Gather(ctx.transImag, ctx.imag, ctx.offsets, 0U, PACK_C_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3);

    const AscendC::DataCopyExtParams outputCopy{
        PACK_C_ROWS, PACK_C_COLS * sizeof(float), 0U,
        static_cast<uint32_t>((ctx.tiling.cubeN - PACK_C_COLS) * sizeof(float)), 0U};
    const uint64_t outputOffset = static_cast<uint64_t>(rowBase) * ctx.tiling.cubeN + colBase;
    AscendC::DataCopyPad(ctx.workspace[ctx.realBase + outputOffset], ctx.transReal, outputCopy);
    AscendC::DataCopyPad(ctx.workspace[ctx.imagBase + outputOffset], ctx.transImag, outputCopy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
}

__aicore__ inline void BuildPackCOffsets(
    AscendC::LocalTensor<int32_t>& scratch, AscendC::LocalTensor<int32_t>& offsets, uint32_t count)
{
    AscendC::CreateVecIndex(scratch, 0, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(offsets, scratch, static_cast<int32_t>(6), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(scratch, scratch, static_cast<int32_t>(PACK_C_ROWS * sizeof(float)), count);
    AscendC::Muls(
        offsets, offsets, static_cast<int32_t>(PACK_C_ROWS * PACK_C_COLS * sizeof(float) - sizeof(float)), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(offsets, scratch, offsets, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunPackCLoop(
    PackCContext& context, uint32_t blockIdx, uint32_t blockNum, uint32_t tileCount, uint32_t colTiles)
{
    for (uint32_t tile = blockIdx; tile < tileCount; tile += blockNum) {
        ProcessPackCTile(context, tile, colTiles);
    }
}
} // namespace

extern "C" __global__ __aicore__ void cherk_arch22_pack_c_kernel(
    GM_ADDR aGm, GM_ADDR workspaceGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf, planarBuf, transposedBuf, offsetBuf;
    pipe.InitBuffer(aosBuf, PACK_C_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(planarBuf, PACK_C_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(transposedBuf, PACK_C_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(offsetBuf, PACK_C_ELEMENTS * sizeof(uint32_t));

    AscendC::LocalTensor<float> aos = aosBuf.Get<float>();
    AscendC::LocalTensor<float> planar = planarBuf.Get<float>();
    AscendC::LocalTensor<float> real = planar;
    AscendC::LocalTensor<float> imag = planar[PACK_C_ELEMENTS];
    AscendC::LocalTensor<float> transposed = transposedBuf.Get<float>();
    AscendC::LocalTensor<float> transReal = transposed;
    AscendC::LocalTensor<float> transImag = transposed[PACK_C_ELEMENTS];
    AscendC::LocalTensor<uint32_t> offsets = offsetBuf.Get<uint32_t>();
    AscendC::LocalTensor<int32_t> offsetsI32 = offsetBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> offsetScratch = transposedBuf.Get<int32_t>();

    BuildPackCOffsets(offsetScratch, offsetsI32, PACK_C_ELEMENTS);

    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> workspace;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm), static_cast<uint64_t>(tiling.lda) * tiling.n * 2U);
    workspace.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGm), WorkspaceCount(tiling));
    const uint64_t realBase = CherkArch22OffsetRealInput(tiling.cubeN, tiling.cubeK);
    const uint64_t imagBase = CherkArch22OffsetImagInput(tiling.cubeN, tiling.cubeK);

    const CherkCopyEvents events = GetCherkCopyEvents();
    const event_t vToMte2 = events.vToMte2;
    const event_t mte2ToV = events.mte2ToV;
    const event_t vToMte3 = events.vToMte3;
    const event_t mte3ToV = events.mte3ToV;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);

    const uint32_t rowTiles = tiling.cubeK / PACK_C_ROWS;
    const uint32_t colTiles = tiling.cubeN / PACK_C_COLS;
    const uint32_t tileCount = rowTiles * colTiles;
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blockNum = CherkNormalizeBlockCount(static_cast<uint32_t>(AscendC::GetBlockNum()));
    PackCContext context{aos,    real,     imag,     transReal, transImag, offsets, a,      workspace,
                         tiling, realBase, imagBase, vToMte2,   mte2ToV,   vToMte3, mte3ToV};
    RunPackCLoop(context, blockIdx, blockNum, tileCount, colTiles);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
}
void cherk_arch22_pack_do(GM_ADDR a, GM_ADDR workspace, const CherkArch22TilingData& tiling, void* stream)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        cherk_arch22_pack_n_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(a, workspace, tiling);
    } else {
        cherk_arch22_pack_c_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(a, workspace, tiling);
    }
}
