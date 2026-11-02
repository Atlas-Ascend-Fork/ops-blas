/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccopy_kernel.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

namespace {

constexpr uint32_t CCOPY_BUFFER_NUM = 2;

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CcopyContiguousSimt(
    uint32_t totalN, const __gm__ uint64_t* x, __gm__ uint64_t* y)
{
    uint32_t globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gridStride = gridDim.x * blockDim.x;
    for (uint32_t i = globalThread; i < totalN; i += gridStride) {
        y[i] = x[i];
    }
}

__aicore__ inline uint64_t AbsStride(int32_t stride)
{
    return stride > 0 ? static_cast<uint64_t>(stride) : static_cast<uint64_t>(-static_cast<int64_t>(stride));
}

class CcopyKernel {
public:
    __aicore__ inline CcopyKernel() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CcopyTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ContinuousIteration(uint32_t complexOffset, uint32_t complexCount);
    __aicore__ inline void ContinuousPumpDrain();
    __aicore__ inline void CopyContiguousIn(LocalTensor<uint32_t>& local, uint64_t complexBase, uint32_t complexCount);
    __aicore__ inline void CopyContiguousOut(
        const LocalTensor<uint32_t>& local, uint64_t complexBase, uint32_t complexCount);
    __aicore__ inline void CompactRead(
        LocalTensor<uint32_t>& local, uint32_t complexCount, uint64_t absIncx, uint64_t xBase);
    __aicore__ inline void CompactWrite(
        const LocalTensor<uint32_t>& local, uint32_t complexCount, uint64_t absIncy, uint64_t yBase);
    __aicore__ inline LocalTensor<uint32_t> ReorderComplexPairs(
        const LocalTensor<uint32_t>& input, uint32_t complexCount);
    __aicore__ inline void StridedIteration(uint32_t elementOffset, uint32_t elementCount);

    TPipe* pipe_ = nullptr;
    GlobalTensor<uint32_t> xGm_;
    GlobalTensor<uint32_t> yGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, CCOPY_BUFFER_NUM> inputQueue_;
    TBuf<QuePosition::VECCALC> reorderBuffer_;
    CcopyTilingData tiling_{};
    uint32_t blockIdx_ = 0;
    uint32_t myOffset_ = 0;
    uint32_t myCount_ = 0;
    bool needReorder_ = false;
};

__aicore__ inline void CcopyKernel::Init(GM_ADDR x, GM_ADDR y, const CcopyTilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    tiling_ = tiling;
    blockIdx_ = GetBlockIdx();

    myCount_ = tiling_.perCoreN;
    if (blockIdx_ < tiling_.extraBlockCores) {
        myCount_ += CCOPY_COMPLEX_PER_BLOCK;
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        myCount_ += tiling_.tailElements;
    }
    myOffset_ = blockIdx_ * tiling_.perCoreN +
                (blockIdx_ < tiling_.extraBlockCores ? blockIdx_ : tiling_.extraBlockCores) * CCOPY_COMPLEX_PER_BLOCK;

    uint64_t absIncx = AbsStride(tiling_.incx);
    uint64_t absIncy = AbsStride(tiling_.incy);
    uint64_t xSpan = absIncx * (static_cast<uint64_t>(tiling_.totalN) - 1) + 1;
    uint64_t ySpan = absIncy * (static_cast<uint64_t>(tiling_.totalN) - 1) + 1;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x), xSpan * CCOPY_LANES_PER_COMPLEX);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), ySpan * CCOPY_LANES_PER_COMPLEX);

    uint32_t tileBytes = tiling_.tileSize * CCOPY_BYTES_PER_COMPLEX;
    pipe_->InitBuffer(inputQueue_, tiling_.queueBufferCount, tileBytes);

    bool continuous = tiling_.incx == 1 && tiling_.incy == 1;
    if (!continuous) {
        pipe_->InitBuffer(reorderBuffer_, 3 * tileBytes);
    }
    needReorder_ = (tiling_.incx < 0) != (tiling_.incy < 0);
}

__aicore__ inline void CcopyKernel::CopyContiguousIn(
    LocalTensor<uint32_t>& local, uint64_t complexBase, uint32_t complexCount)
{
    uint32_t laneCount = complexCount * CCOPY_LANES_PER_COMPLEX;
    uint32_t alignedLanes = (laneCount / CCOPY_LANES_PER_BLOCK) * CCOPY_LANES_PER_BLOCK;
    uint32_t tailLanes = laneCount - alignedLanes;
    uint64_t laneBase = complexBase * CCOPY_LANES_PER_COMPLEX;

    if (alignedLanes > 0) {
        DataCopy(local, xGm_[laneBase], alignedLanes);
    }
    if (tailLanes > 0) {
        uint8_t padding = static_cast<uint8_t>(CCOPY_LANES_PER_BLOCK - tailLanes);
        DataCopyExtParams params{1, static_cast<uint32_t>(tailLanes * sizeof(uint32_t)), 0, 0, 0};
        DataCopyPadExtParams<uint32_t> padParams{true, 0, padding, 0U};
        DataCopyPad(local[alignedLanes], xGm_[laneBase + alignedLanes], params, padParams);
    }
}

__aicore__ inline void CcopyKernel::CopyContiguousOut(
    const LocalTensor<uint32_t>& local, uint64_t complexBase, uint32_t complexCount)
{
    uint32_t laneCount = complexCount * CCOPY_LANES_PER_COMPLEX;
    uint32_t alignedLanes = (laneCount / CCOPY_LANES_PER_BLOCK) * CCOPY_LANES_PER_BLOCK;
    uint32_t tailLanes = laneCount - alignedLanes;
    uint64_t laneBase = complexBase * CCOPY_LANES_PER_COMPLEX;

    if (alignedLanes > 0) {
        DataCopy(yGm_[laneBase], local, alignedLanes);
    }
    if (tailLanes > 0) {
        DataCopyExtParams params{1, static_cast<uint32_t>(tailLanes * sizeof(uint32_t)), 0, 0, 0};
        DataCopyPad(yGm_[laneBase + alignedLanes], local[alignedLanes], params);
    }
}

__aicore__ inline void CcopyKernel::ContinuousIteration(uint32_t complexOffset, uint32_t complexCount)
{
    LocalTensor<uint32_t> local = inputQueue_.AllocTensor<uint32_t>();
    CopyContiguousIn(local, complexOffset, complexCount);
    inputQueue_.EnQue<uint32_t>(local);

    LocalTensor<uint32_t> output = inputQueue_.DeQue<uint32_t>();
    CopyContiguousOut(output, complexOffset, complexCount);
    inputQueue_.FreeTensor(output);
}

__aicore__ inline void CcopyKernel::ContinuousPumpDrain()
{
    uint32_t fullTiles = myCount_ / tiling_.tileSize;
    uint32_t tail = myCount_ % tiling_.tileSize;
    uint32_t curOffset = myOffset_;
    uint32_t tileLanes = tiling_.tileSize * CCOPY_LANES_PER_COMPLEX;

    if (fullTiles == 0) {
        if (tail > 0) {
            ContinuousIteration(curOffset, tail);
        }
        return;
    }

    LocalTensor<uint32_t> first = inputQueue_.AllocTensor<uint32_t>();
    DataCopy(first, xGm_[static_cast<uint64_t>(curOffset) * CCOPY_LANES_PER_COMPLEX], tileLanes);
    inputQueue_.EnQue<uint32_t>(first);
    curOffset += tiling_.tileSize;

    for (uint32_t i = 1; i < fullTiles; ++i) {
        LocalTensor<uint32_t> next = inputQueue_.AllocTensor<uint32_t>();
        DataCopy(next, xGm_[static_cast<uint64_t>(curOffset) * CCOPY_LANES_PER_COMPLEX], tileLanes);

        LocalTensor<uint32_t> current = inputQueue_.DeQue<uint32_t>();
        uint64_t outputOffset = static_cast<uint64_t>(curOffset - tiling_.tileSize) * CCOPY_LANES_PER_COMPLEX;
        DataCopy(yGm_[outputOffset], current, tileLanes);
        inputQueue_.FreeTensor(current);

        inputQueue_.EnQue<uint32_t>(next);
        curOffset += tiling_.tileSize;
    }

    LocalTensor<uint32_t> last = inputQueue_.DeQue<uint32_t>();
    uint64_t lastOffset = static_cast<uint64_t>(curOffset - tiling_.tileSize) * CCOPY_LANES_PER_COMPLEX;
    DataCopy(yGm_[lastOffset], last, tileLanes);
    inputQueue_.FreeTensor(last);

    if (tail > 0) {
        ContinuousIteration(curOffset, tail);
    }
}

__aicore__ inline void CcopyKernel::CompactRead(
    LocalTensor<uint32_t>& local, uint32_t complexCount, uint64_t absIncx, uint64_t xBase)
{
    uint32_t remaining = complexCount;
    uint32_t offset = 0;
    int64_t srcStride = static_cast<int64_t>((absIncx - 1) * CCOPY_BYTES_PER_COMPLEX);
    while (remaining > 0) {
        uint32_t batch = remaining > CCOPY_MAX_COMPACT_BLOCKS ? CCOPY_MAX_COMPACT_BLOCKS : remaining;
        DataCopyExtParams params{static_cast<uint16_t>(batch), CCOPY_BYTES_PER_COMPLEX, srcStride, 0, 0};
        DataCopyPadExtParams<uint32_t> padParams{true, 0, 0, 0U};
        uint64_t sourceLane = (xBase + static_cast<uint64_t>(offset) * absIncx) * CCOPY_LANES_PER_COMPLEX;
        DataCopyPad<uint32_t, PaddingMode::Compact>(
            local[offset * CCOPY_LANES_PER_COMPLEX], xGm_[sourceLane], params, padParams);
        remaining -= batch;
        offset += batch;
    }
}

__aicore__ inline void CcopyKernel::CompactWrite(
    const LocalTensor<uint32_t>& local, uint32_t complexCount, uint64_t absIncy, uint64_t yBase)
{
    uint32_t remaining = complexCount;
    uint32_t offset = 0;
    int64_t dstStride = static_cast<int64_t>((absIncy - 1) * CCOPY_BYTES_PER_COMPLEX);
    while (remaining > 0) {
        uint32_t batch = remaining > CCOPY_MAX_COMPACT_BLOCKS ? CCOPY_MAX_COMPACT_BLOCKS : remaining;
        DataCopyExtParams params{static_cast<uint16_t>(batch), CCOPY_BYTES_PER_COMPLEX, 0, dstStride, 0};
        uint64_t destinationLane = (yBase + static_cast<uint64_t>(offset) * absIncy) * CCOPY_LANES_PER_COMPLEX;
        DataCopyPad<uint32_t, PaddingMode::Compact>(
            yGm_[destinationLane], local[offset * CCOPY_LANES_PER_COMPLEX], params);
        remaining -= batch;
        offset += batch;
    }
}

__aicore__ inline LocalTensor<uint32_t> CcopyKernel::ReorderComplexPairs(
    const LocalTensor<uint32_t>& input, uint32_t complexCount)
{
    uint32_t laneCount = complexCount * CCOPY_LANES_PER_COMPLEX;
    uint32_t interleaveCount =
        ((complexCount + CCOPY_LANES_PER_BLOCK - 1) / CCOPY_LANES_PER_BLOCK) * CCOPY_LANES_PER_BLOCK;
    uint32_t offsetBufferBytes = tiling_.tileSize * CCOPY_BYTES_PER_COMPLEX;
    // reorderBuffer_ 为裸 TBuf：上一轮 tile 的 MTE3 写出(CompactWrite/CopyContiguousOut)可能
    // 尚未读完本缓冲，本轮 V 管线(Gather/Interleave)复用前必须等待 MTE3 完成，
    // 避免多 tile 场景下的 WAR 竞态导致数据错乱
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    LocalTensor<uint32_t> reordered = reorderBuffer_.GetWithOffset<uint32_t>(laneCount, 0);
    LocalTensor<int32_t> offsets = reorderBuffer_.GetWithOffset<int32_t>(2 * interleaveCount, offsetBufferBytes);
    LocalTensor<int32_t> offsets1 =
        reorderBuffer_.GetWithOffset<int32_t>(interleaveCount, offsetBufferBytes + interleaveCount * sizeof(int32_t));
    LocalTensor<int32_t> evenOffsets = reorderBuffer_.GetWithOffset<int32_t>(interleaveCount, 2 * offsetBufferBytes);
    LocalTensor<int32_t> oddOffsets = reorderBuffer_.GetWithOffset<int32_t>(
        interleaveCount, 2 * offsetBufferBytes + tiling_.tileSize * sizeof(int32_t));

    // Interleave the byte offsets of each complex element's real and imaginary
    // lanes. Padding to a vector block keeps every tensor 32-byte aligned; Gather
    // consumes only the first laneCount entries.
    int32_t firstOffset = static_cast<int32_t>((complexCount - 1) * CCOPY_BYTES_PER_COMPLEX);
    ArithProgression<int32_t>(
        evenOffsets, firstOffset, -static_cast<int32_t>(CCOPY_BYTES_PER_COMPLEX),
        static_cast<int32_t>(interleaveCount));
    ArithProgression<int32_t>(
        oddOffsets, firstOffset + static_cast<int32_t>(sizeof(uint32_t)),
        -static_cast<int32_t>(CCOPY_BYTES_PER_COMPLEX), static_cast<int32_t>(interleaveCount));
    PipeBarrier<PIPE_V>();
    Interleave(offsets, offsets1, evenOffsets, oddOffsets, static_cast<int32_t>(interleaveCount));
    PipeBarrier<PIPE_V>();
    Gather(reordered, input, offsets.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), laneCount);
    // reorderBuffer_ 是裸 TBuf 缓冲，不在队列事件管理内：Gather(PIPE_V) 写入完成后必须
    // 显式插入 V->MTE3 事件同步，否则后续 CompactWrite/CopyContiguousOut 在 MTE3 上
    // 读到的是未就绪的 UB 残留内容（表现为 needReorder_ 场景 y 全为垃圾值）
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    return reordered;
}

__aicore__ inline void CcopyKernel::StridedIteration(uint32_t elementOffset, uint32_t elementCount)
{
    LocalTensor<uint32_t> local = inputQueue_.AllocTensor<uint32_t>();
    uint64_t absIncx = AbsStride(tiling_.incx);
    uint64_t xBase = tiling_.incx < 0 ?
                         (static_cast<uint64_t>(tiling_.totalN) - elementOffset - elementCount) * absIncx :
                         static_cast<uint64_t>(elementOffset) * absIncx;

    if (absIncx == 1) {
        CopyContiguousIn(local, xBase, elementCount);
    } else {
        CompactRead(local, elementCount, absIncx, xBase);
    }

    inputQueue_.EnQue<uint32_t>(local);
    LocalTensor<uint32_t> input = inputQueue_.DeQue<uint32_t>();
    LocalTensor<uint32_t> reordered;
    if (needReorder_) {
        reordered = ReorderComplexPairs(input, elementCount);
    }
    LocalTensor<uint32_t> output = needReorder_ ? reordered : input;

    uint64_t absIncy = AbsStride(tiling_.incy);
    uint64_t yBase = tiling_.incy < 0 ?
                         (static_cast<uint64_t>(tiling_.totalN) - elementOffset - elementCount) * absIncy :
                         static_cast<uint64_t>(elementOffset) * absIncy;

    if (absIncy == 1) {
        CopyContiguousOut(output, yBase, elementCount);
    } else {
        CompactWrite(output, elementCount, absIncy, yBase);
    }
    inputQueue_.FreeTensor(input);
}

__aicore__ inline void CcopyKernel::Process()
{
    if (myCount_ == 0) {
        return;
    }
    if (tiling_.incx == 1 && tiling_.incy == 1) {
        ContinuousPumpDrain();
        return;
    }

    uint32_t fullTiles = myCount_ / tiling_.tileSize;
    uint32_t tail = myCount_ % tiling_.tileSize;
    uint32_t curOffset = myOffset_;
    for (uint32_t i = 0; i < fullTiles; ++i) {
        StridedIteration(curOffset, tiling_.tileSize);
        curOffset += tiling_.tileSize;
    }
    if (tail > 0) {
        StridedIteration(curOffset, tail);
    }
}

} // namespace

extern "C" __global__ __aicore__ void ccopy_kernel(GM_ADDR x, GM_ADDR y, CcopyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CcopyKernel op;
    op.Init(x, y, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void ccopy_contiguous_simt_kernel(
    GM_ADDR x, GM_ADDR y, uint32_t totalN, uint32_t threadCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<CcopyContiguousSimt>(
        dim3{threadCount, 1, 1}, totalN, reinterpret_cast<const __gm__ uint64_t*>(x),
        reinterpret_cast<__gm__ uint64_t*>(y));
}

void ccopy_kernel_do(uint8_t* x, uint8_t* y, const CcopyTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ccopy_kernel<<<numBlocks, nullptr, stream>>>(x, y, tiling);
}

void ccopy_contiguous_simt_kernel_do(
    uint8_t* x, uint8_t* y, uint32_t totalN, uint32_t threadCount, uint32_t numBlocks, void* stream)
{
    ccopy_contiguous_simt_kernel<<<numBlocks, nullptr, stream>>>(x, y, totalN, threadCount);
}
