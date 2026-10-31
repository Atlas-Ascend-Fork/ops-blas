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
#include "sswap_tiling_data.h"

using namespace AscendC;

class SswapAIV {
public:
    __aicore__ inline SswapAIV() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SswapTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SingleIteration(uint32_t curOffset, uint32_t dataCount);

    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    TBuf<TPosition::VECIN> xBuf_;
    TBuf<TPosition::VECIN> yBuf_;
    SswapTilingData tiling_;
    uint32_t blockIdx_;
    uint32_t myOffset_;
    uint32_t myCount_;
};

__aicore__ inline void SswapAIV::Init(GM_ADDR x, GM_ADDR y, const SswapTilingData& tiling, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tiling;

    myOffset_ = blockIdx_ * tiling_.perCoreN;
    myCount_ = tiling_.perCoreN;
    if (blockIdx_ == GetBlockNum() - 1) {
        myCount_ += tiling_.remainder;
    }

    xGM_.SetGlobalBuffer((__gm__ float*)x, tiling_.xLength);
    yGM_.SetGlobalBuffer((__gm__ float*)y, tiling_.yLength);

    uint32_t bufSize = tiling_.tileSize * sizeof(float);
    pipe->InitBuffer(xBuf_, bufSize);
    pipe->InitBuffer(yBuf_, bufSize);
}

__aicore__ inline void SswapAIV::SingleIteration(uint32_t curOffset, uint32_t dataCount)
{
    uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    LocalTensor<float> xLocal = xBuf_.Get<float>();
    LocalTensor<float> yLocal = yBuf_.Get<float>();

    if (alignedCount > 0) {
        DataCopy(xLocal, xGM_[curOffset], alignedCount);
        DataCopy(yLocal, yGM_[curOffset], alignedCount);
    }

    if (tailCount > 0) {
        uint8_t paddingNum = ELEMENTS_PER_BLOCK - tailCount;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0.0f};
        DataCopyPad(xLocal[alignedCount], xGM_[curOffset + alignedCount], copyParams, padParams);
        DataCopyPad(yLocal[alignedCount], yGM_[curOffset + alignedCount], copyParams, padParams);
    }

    // DataCopy/DataCopyPad are async. The stores below read UB written by
    // the MTE2 loads above, so wait for MTE2 before issuing MTE3.
    SetFlag<HardEvent::MTE2_MTE3>(0);
    WaitFlag<HardEvent::MTE2_MTE3>(0);

    if (alignedCount > 0) {
        DataCopy(yGM_[curOffset], xLocal, alignedCount);
        DataCopy(xGM_[curOffset], yLocal, alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(yGM_[curOffset + alignedCount], xLocal[alignedCount], copyParams);
        DataCopyPad(xGM_[curOffset + alignedCount], yLocal[alignedCount], copyParams);
    }

    // The next tile reuses xBuf_/yBuf_; wait for this tile's MTE3 stores
    // before the next MTE2 load overwrites them.
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

__aicore__ inline void SswapAIV::Process()
{
    if (myCount_ == 0) {
        return;
    }

    if (tiling_.incx != 1 || tiling_.incy != 1) {
        for (uint32_t i = 0; i < myCount_; ++i) {
            int64_t logicalIndex = static_cast<int64_t>(myOffset_) + i;
            int64_t xIndex = tiling_.kx + logicalIndex * tiling_.incx;
            int64_t yIndex = tiling_.ky + logicalIndex * tiling_.incy;
            float xValue = xGM_.GetValue(static_cast<uint64_t>(xIndex));
            float yValue = yGM_.GetValue(static_cast<uint64_t>(yIndex));
            xGM_.SetValue(static_cast<uint64_t>(xIndex), yValue);
            yGM_.SetValue(static_cast<uint64_t>(yIndex), xValue);
        }
        return;
    }

    uint32_t tileLoop = myCount_ / tiling_.tileSize;
    uint32_t tileTail = myCount_ % tiling_.tileSize;
    uint32_t curOffset = myOffset_;

    for (uint32_t i = 0; i < tileLoop; i++) {
        SingleIteration(curOffset, tiling_.tileSize);
        curOffset += tiling_.tileSize;
    }

    if (tileTail > 0) {
        SingleIteration(curOffset, tileTail);
    }
}

extern "C" __global__ __aicore__ void sswap_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SswapTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SswapAIV op;
    op.Init(x, y, tiling, &pipe);
    op.Process();
}

void sswap_kernel_do(
    GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SswapTilingData& tiling, uint32_t numBlocks, void* stream)
{
    sswap_kernel<<<numBlocks, nullptr, stream>>>(x, y, nullptr, tiling);
}
