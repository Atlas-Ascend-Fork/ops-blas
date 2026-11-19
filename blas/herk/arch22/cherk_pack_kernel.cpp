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
#define CHERK_ARCH22_DEVICE_IMPL
#include "cherk_kernel.h"

namespace {
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
constexpr uint32_t PACK_C_ROWS = 64U;
constexpr uint32_t PACK_C_COLS = 32U;
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

__aicore__ inline void TransposePackC(PackCContext& ctx)
{
    const bool directImag = CherkArch22DirectImaginary(ctx.tiling);
    if (directImag) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
    }
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = PACK_C_ELEMENTS * 2U / 64U;
    AscendC::GatherMask<float>(ctx.real, ctx.aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(ctx.imag, ctx.aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    if (!directImag) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
    }
    AscendC::Gather(ctx.transReal, ctx.real, ctx.offsets, 0U, PACK_C_ELEMENTS);
    AscendC::Gather(ctx.transImag, ctx.imag, ctx.offsets, 0U, PACK_C_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    if (directImag) {
        // The extracted real plane is dead after transpose. It now carries
        // -I, and the earlier MTE3_V wait protects it on the next tile.
        AscendC::Muls(ctx.real, ctx.transImag, -1.0f, PACK_C_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.vToMte2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3);
}

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

    const bool directImag = CherkArch22DirectImaginary(ctx.tiling);
    TransposePackC(ctx);

    const AscendC::DataCopyExtParams outputCopy{
        PACK_C_ROWS, PACK_C_COLS * sizeof(float), 0U,
        static_cast<uint32_t>((ctx.tiling.cubeN - PACK_C_COLS) * sizeof(float)), 0U};
    const uint64_t outputOffset = static_cast<uint64_t>(rowBase) * ctx.tiling.cubeN + colBase;
    AscendC::DataCopyPad(ctx.workspace[ctx.realBase + outputOffset], ctx.transReal, outputCopy);
    AscendC::DataCopyPad(ctx.workspace[ctx.imagBase + outputOffset], ctx.transImag, outputCopy);
    if (directImag) {
        const uint64_t base = CherkArch22OffsetNegativeImagInput(ctx.tiling.cubeN, ctx.tiling.cubeK);
        AscendC::DataCopyPad(ctx.workspace[base + outputOffset], ctx.real, outputCopy);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV);
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
    GM_ADDR aGm, GM_ADDR workspaceGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> planarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transposedBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf;
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

    AscendC::GlobalTensor<uint32_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
    AscendC::DataCopy(offsets, metadata[CHERK_ARCH22_META_GENERAL_PACK_C_OFFSET], PACK_C_ELEMENTS);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

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
void cherk_arch22_pack_do(
    GM_ADDR a, GM_ADDR workspace, GM_ADDR metadata, const CherkArch22TilingData& tiling, void* stream)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        cherk_arch22_pack_n_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(a, workspace, tiling);
    } else {
        cherk_arch22_pack_c_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(a, workspace, metadata, tiling);
    }
}
