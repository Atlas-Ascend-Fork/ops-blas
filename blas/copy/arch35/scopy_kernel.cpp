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
#include "scopy_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

class ScopyAIV {
public:
    __aicore__ inline ScopyAIV() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const ScopyTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ContinuousIteration(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void ContinuousPumpDrain();
    __aicore__ inline void DiscreteIteration(uint32_t elementOffset, uint32_t dataCount, int32_t incx, int32_t incy);
    __aicore__ inline void DiscreteRead(
        LocalTensor<float>& xLocal, uint32_t elementOffset, uint32_t dataCount, int32_t incx);
    __aicore__ inline void DiscreteWrite(
        const LocalTensor<float>& xOut, uint32_t elementOffset, uint32_t dataCount, int32_t incy);
    __aicore__ inline void DiscreteCompactRead(
        LocalTensor<float>& xLocal, uint32_t dataCount, uint32_t absIncx, uint32_t xBase);
    __aicore__ inline void DiscreteCompactWrite(
        const LocalTensor<float>& xOut, uint32_t dataCount, uint32_t absIncy, uint32_t yBase);
    __aicore__ inline LocalTensor<float> DiscreteReorder(LocalTensor<float>& xIn, uint32_t dataCount);

    TPipe* pipe_;
    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueueX_;
    TBuf<QuePosition::VECCALC> tmpBuf_; // Gather temp buffers (gatherDst + srcOffset_)
    LocalTensor<uint32_t> srcOffset_;   // Offset table filled in Init via vector instruction
    bool needReorder_;                  // Cached flag: incx<0 XOR incy<0
    ScopyTilingData tiling_;
    uint32_t blockIdx_;
    uint32_t myOffset_;
    uint32_t myCount_;
};

__aicore__ inline void ScopyAIV::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const ScopyTilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    tiling_ = tiling;

    // distribute work evenly: base count + extra block for first extraBlockCores cores,
    // unaligned tail elements go to the last core. each core self-calculates offset.
    myCount_ = tiling_.perCoreN;
    if (blockIdx_ < tiling_.extraBlockCores) {
        myCount_ += ELEMENTS_PER_BLOCK;
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        myCount_ += tiling_.tailElements;
    }
    myOffset_ = blockIdx_ * tiling_.perCoreN +
                (blockIdx_ < tiling_.extraBlockCores ? blockIdx_ : tiling_.extraBlockCores) * ELEMENTS_PER_BLOCK;

    // Compute buffer spans for strided access.
    uint32_t xBufSize = tiling_.totalN;
    if (tiling_.incx != 1) {
        uint32_t absIncx =
            (tiling_.incx > 0) ? static_cast<uint32_t>(tiling_.incx) : static_cast<uint32_t>(-tiling_.incx);
        xBufSize = absIncx * (tiling_.totalN - 1) + 1;
    }
    uint32_t yBufSize = tiling_.totalN;
    if (tiling_.incy != 1) {
        uint32_t absIncy =
            (tiling_.incy > 0) ? static_cast<uint32_t>(tiling_.incy) : static_cast<uint32_t>(-tiling_.incy);
        yBufSize = absIncy * (tiling_.totalN - 1) + 1;
    }
    xGM_.SetGlobalBuffer((__gm__ float*)x, xBufSize);
    yGM_.SetGlobalBuffer((__gm__ float*)y, yBufSize);

    uint32_t bufSize = tiling_.tileSize * sizeof(float);
    pipe_->InitBuffer(inQueueX_, BUFFER_NUM, bufSize);

    bool isContinuous = (tiling_.incx == 1 && tiling_.incy == 1);
    if (!isContinuous) {
        uint32_t gatherBufSize = tiling_.tileSize * (sizeof(float) + sizeof(uint32_t));
        pipe_->InitBuffer(tmpBuf_, gatherBufSize);
    }

    needReorder_ = (tiling_.incx < 0) != (tiling_.incy < 0);

    // Build reverse-index offset table directly in UB for Gather.
    // Eliminates Host-side aclrtMalloc/aclrtMemcpy/aclrtFree/async-sync lifecycle.
    // srcOffset_[i] = (tileSize - 1 - i) * sizeof(float), byte offset for Gather.
    // This runs once in Init (not per-tile), ~4088 iterations at ~AIV core clock rate.
    if (needReorder_) {
        srcOffset_ = tmpBuf_.GetWithOffset<uint32_t>(tiling_.tileSize, tiling_.tileSize * sizeof(float));
        if (workSpace != nullptr) {
            // Load pre-generated offset table from workspace via DataCopy (single MTE2 transfer)
            GlobalTensor<uint32_t> wsTensor;
            wsTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workSpace), tiling_.tileSize);
            DataCopy(srcOffset_, wsTensor, tiling_.tileSize);
        } else {
            // Fallback: scalar loop to compute reverse offsets inline.
            // Compatible with no-workspace callers (e.g., legacy or small-workspace scenarios).
            for (uint32_t i = 0; i < tiling_.tileSize; i++) {
                srcOffset_.SetValue(i, (tiling_.tileSize - 1 - i) * sizeof(float));
            }
        }
        // Gather reads srcOffset_ on the V pipe. When the table came from GM
        // via DataCopy above, wait for MTE2 before the first Gather.
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }
}

__aicore__ inline void ScopyAIV::ContinuousIteration(uint32_t curOffset, uint32_t dataCount)
{
    // aligned portion for DataCopy (32B = 8 floats aligned)
    uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    // CopyIn: GM -> UB (MTE2) - DataCopy for aligned, DataCopyPad for tail
    LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();

    if (alignedCount > 0) {
        DataCopy(xLocal, xGM_[curOffset], alignedCount);
    }

    if (tailCount > 0) {
        uint8_t paddingNum = ELEMENTS_PER_BLOCK - tailCount;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0.0f};
        DataCopyPad(xLocal[alignedCount], xGM_[curOffset + alignedCount], copyParams, padParams);
    }

    // EnQue provides implicit MTE2->MTE3 sync (no explicit FetchEventID needed)
    inQueueX_.EnQue<float>(xLocal);

    // CopyOut: UB -> GM (MTE3)
    LocalTensor<float> xIn = inQueueX_.DeQue<float>();

    if (alignedCount > 0) {
        DataCopy(yGM_[curOffset], xIn, alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(yGM_[curOffset + alignedCount], xIn[alignedCount], copyParams);
    }

    inQueueX_.FreeTensor(xIn);
}

__aicore__ inline void ScopyAIV::ContinuousPumpDrain()
{
    // Prime-Pump-Drain pattern for double buffer (BUFFER_NUM=2) continuous path.
    //
    // The pattern overlaps MTE2 (GM->UB read) with MTE3 (UB->GM write) by using
    // two buffer slots:
    //
    //   PRIME:  DataCopy(bufA, xGM[t0])  -- MTE2 read tile 0
    //           EnQue(bufA)              -- signal MTE2 done, bufA -> VECOUT
    //
    //   PUMP:   DataCopy(bufB, xGM[t1])  -- MTE2 read tile 1 (starts in-flight)
    //           DeQue(bufA)              -- get bufA from VECOUT (tile 0 ready)
    //           DataCopy(yGM[t0], bufA)  -- MTE3 write tile 0 (overlaps MTE2 t1)
    //           FreeTensor(bufA)         -- release bufA
    //           EnQue(bufB)              -- signal MTE2 done on bufB
    //           swap roles and repeat
    //
    //   DRAIN:  DeQue(lastBuf)           -- get last tile from VECOUT
    //           DataCopy(yGM[last], buf) -- MTE3 write last tile
    //           FreeTensor(lastBuf)
    //
    // Key insight: MTE2 read of tile i and MTE3 write of tile i-1 execute on
    // separate hardware units (MTE2 = read engine, MTE3 = write engine), so
    // they can operate in parallel when using different buffer slots.
    //
    // EnQue(pipe_) acts as MTE2 synchronization barrier: it waits for outstanding
    // MTE2 operations on the tensor before moving it from VECIN to VECOUT.
    //
    // The tail tile (if any) may be non-aligned and uses the existing
    // single-buffer ContinuousIteration with DataCopyPad handling.

    uint32_t tileLoop = myCount_ / tiling_.tileSize;
    uint32_t tileTail = myCount_ % tiling_.tileSize;
    uint32_t curOffset = myOffset_;

    // If no full tiles, fall back to single-buffer tail handling
    if (tileLoop == 0) {
        if (tileTail > 0) {
            ContinuousIteration(curOffset, tileTail);
        }
        return;
    }

    // ---- PRIME: read tile 0 into first buffer (slot A) ----
    // AllocTensor grants a VECIN buffer. DataCopy starts MTE2 GM->UB read.
    // EnQue waits for MTE2 completion, then moves buffer from VECIN to VECOUT.
    LocalTensor<float> bufA = inQueueX_.AllocTensor<float>();
    DataCopy(bufA, xGM_[curOffset], tiling_.tileSize);
    inQueueX_.EnQue<float>(bufA);
    curOffset += tiling_.tileSize;

    // ---- PUMP: tiles 1 .. tileLoop-1 ----
    // In each iteration: start MTE2 read for tile i, then issue MTE3 write
    // for tile i-1. MTE2 and MTE3 run on separate engines and overlap.
    for (uint32_t i = 1; i < tileLoop; i++) {
        // Start MTE2 read for tile i into nextBuf (VECIN, starts in-flight)
        LocalTensor<float> nextBuf = inQueueX_.AllocTensor<float>();
        DataCopy(nextBuf, xGM_[curOffset], tiling_.tileSize);

        // DeQue gets previous tile's buffer from VECOUT (MTE2 already done).
        // DataCopy issues MTE3 write, overlapping with ongoing MTE2 read of tile i.
        LocalTensor<float> curBuf = inQueueX_.DeQue<float>();
        DataCopy(yGM_[curOffset - tiling_.tileSize], curBuf, tiling_.tileSize);
        inQueueX_.FreeTensor(curBuf);

        // Signal MTE2 completion on nextBuf so it can be consumed next iteration
        inQueueX_.EnQue<float>(nextBuf);
        curOffset += tiling_.tileSize;
    }

    // ---- DRAIN: write last full tile ----
    // DeQue gets the final buffer from VECOUT (last tile's MTE2 complete).
    // DataCopy writes it to GM via MTE3.
    LocalTensor<float> lastBuf = inQueueX_.DeQue<float>();
    DataCopy(yGM_[curOffset - tiling_.tileSize], lastBuf, tiling_.tileSize);
    inQueueX_.FreeTensor(lastBuf);

    // ---- Handle tail tile (if any) ----
    // Tail tile may be non-32B-aligned, requiring DataCopyPad.
    // Reuse existing ContinuousIteration which handles aligned+tail split
    // with DataCopy and DataCopyPad.
    if (tileTail > 0) {
        ContinuousIteration(curOffset, tileTail);
    }
}

__aicore__ inline void ScopyAIV::DiscreteCompactRead(
    LocalTensor<float>& xLocal, uint32_t dataCount, uint32_t absIncx, uint32_t xBase)
{
    constexpr uint32_t compactMax = 4088;
    uint32_t readRemaining = dataCount;
    uint32_t readOff = 0;
    int64_t srcStride = static_cast<int64_t>(absIncx - 1) * sizeof(float);
    while (readRemaining > 0) {
        uint32_t batch = (readRemaining > compactMax) ? compactMax : readRemaining;
        DataCopyExtParams copyParams{static_cast<uint16_t>(batch), sizeof(float), srcStride, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0.0f};
        DataCopyPad<float, PaddingMode::Compact>(
            xLocal[readOff], xGM_[xBase + readOff * absIncx], copyParams, padParams);
        readRemaining -= batch;
        readOff += batch;
    }
}

__aicore__ inline void ScopyAIV::DiscreteCompactWrite(
    const LocalTensor<float>& xOut, uint32_t dataCount, uint32_t absIncy, uint32_t yBase)
{
    constexpr uint32_t compactMax = 4088;
    uint32_t writeRemaining = dataCount;
    uint32_t writeOff = 0;
    int64_t dstStride = static_cast<int64_t>(absIncy - 1) * sizeof(float);
    while (writeRemaining > 0) {
        uint32_t batch = (writeRemaining > compactMax) ? compactMax : writeRemaining;
        DataCopyExtParams copyParams{static_cast<uint16_t>(batch), sizeof(float), 0, dstStride, 0};
        DataCopyPad<float, PaddingMode::Compact>(yGM_[yBase + writeOff * absIncy], xOut[writeOff], copyParams);
        writeRemaining -= batch;
        writeOff += batch;
    }
}

__aicore__ inline LocalTensor<float> ScopyAIV::DiscreteReorder(LocalTensor<float>& xIn, uint32_t dataCount)
{
    LocalTensor<float> gatherDst;
    if (!needReorder_) {
        return gatherDst;
    }

    gatherDst = tmpBuf_.Get<float>(0);
    uint32_t offsetStart = tiling_.tileSize - dataCount;
    if ((offsetStart & 0x7) == 0) {
        Gather(gatherDst, xIn, srcOffset_[offsetStart], static_cast<uint32_t>(0), dataCount);
    } else {
        for (uint32_t i = 0; i < dataCount; i++) {
            gatherDst.SetValue(i, xIn.GetValue(dataCount - 1 - i));
        }
    }
    return gatherDst;
}

__aicore__ inline void ScopyAIV::DiscreteIteration(
    uint32_t elementOffset, uint32_t dataCount, int32_t incx, int32_t incy)
{
    LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
    DiscreteRead(xLocal, elementOffset, dataCount, incx);

    inQueueX_.EnQue<float>(xLocal);
    LocalTensor<float> xIn = inQueueX_.DeQue<float>();

    if (needReorder_) {
        // The scalar fallback in DiscreteReorder reads xIn with GetValue;
        // make the MTE2 data visible to the scalar pipe first.
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
    }
    LocalTensor<float> gatherDst = DiscreteReorder(xIn, dataCount);
    if (needReorder_) {
        // Gather writes tmpBuf_ on the V pipe; the MTE3 stores below must
        // wait for it, otherwise they can read the pre-Gather contents.
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
    }

    LocalTensor<float> xOut = needReorder_ ? gatherDst : xIn;
    DiscreteWrite(xOut, elementOffset, dataCount, incy);

    if (needReorder_) {
        // tmpBuf_ is reused by the next tile's Gather; wait for this tile's
        // MTE3 stores before overwriting it.
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }

    inQueueX_.FreeTensor(xIn);
}

__aicore__ inline void ScopyAIV::DiscreteRead(
    LocalTensor<float>& xLocal, uint32_t elementOffset, uint32_t dataCount, int32_t incx)
{
    uint32_t absIncx = (incx > 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);
    uint32_t xBase = (incx < 0) ? (tiling_.totalN - elementOffset - dataCount) * absIncx : elementOffset * absIncx;

    if (absIncx == 1) {
        uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
        uint32_t tailCount = dataCount - alignedCount;
        if (alignedCount > 0) {
            DataCopy(xLocal, xGM_[xBase], alignedCount);
        }
        if (tailCount > 0) {
            uint8_t paddingNum = static_cast<uint8_t>(ELEMENTS_PER_BLOCK - tailCount);
            DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0.0f};
            DataCopyPad(
                xLocal[alignedCount], xGM_[xBase + alignedCount],
                DataCopyExtParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0}, padParams);
        }
    } else {
        DiscreteCompactRead(xLocal, dataCount, absIncx, xBase);
    }
}

__aicore__ inline void ScopyAIV::DiscreteWrite(
    const LocalTensor<float>& xOut, uint32_t elementOffset, uint32_t dataCount, int32_t incy)
{
    uint32_t absIncy = (incy > 0) ? static_cast<uint32_t>(incy) : static_cast<uint32_t>(-incy);
    uint32_t yBase = (incy < 0) ? (tiling_.totalN - elementOffset - dataCount) * absIncy : elementOffset * absIncy;

    if (absIncy == 1) {
        uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
        uint32_t tailCount = dataCount - alignedCount;
        if (alignedCount > 0) {
            DataCopy(yGM_[yBase], xOut, alignedCount);
        }
        if (tailCount > 0) {
            DataCopyPad(
                yGM_[yBase + alignedCount], xOut[alignedCount],
                DataCopyExtParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0});
        }
    } else {
        DiscreteCompactWrite(xOut, dataCount, absIncy, yBase);
    }
}

__aicore__ inline void ScopyAIV::Process()
{
    if (myCount_ == 0) {
        return;
    }

    bool isContinuous = (tiling_.incx == 1 && tiling_.incy == 1);

    if (isContinuous) {
        // Double-buffer (BUFFER_NUM=2) prime-pump-drain for MTE2/MTE3 overlap
        ContinuousPumpDrain();
    } else {
        uint32_t tileLoop = myCount_ / tiling_.tileSize;
        uint32_t tileTail = myCount_ % tiling_.tileSize;
        uint32_t curOffset = myOffset_;

        for (uint32_t i = 0; i < tileLoop; i++) {
            DiscreteIteration(curOffset, tiling_.tileSize, tiling_.incx, tiling_.incy);
            curOffset += tiling_.tileSize;
        }

        if (tileTail > 0) {
            DiscreteIteration(curOffset, tileTail, tiling_.incx, tiling_.incy);
        }
    }
}

extern "C" __global__ __aicore__ void scopy_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, ScopyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe; // R3: TPipe created in kernel entry, not as class member
    ScopyAIV op;
    op.Init(x, y, workSpace, tiling, &pipe);
    op.Process();
}

void scopy_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* workSpace, const ScopyTilingData& tiling, uint32_t numBlocks, void* stream)
{
    scopy_kernel<<<numBlocks, nullptr, stream>>>(x, y, workSpace, tiling);
}
