/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cfloat>
#include <cstdint>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "isamin_tiling_data.h"

using namespace AscendC;

namespace {

constexpr uint32_t ISAMIN_FLOAT_BYTES = sizeof(float);
constexpr uint32_t ISAMIN_BLOCK_ELEMENTS = ISAMIN_ARCH22_ALIGN_ELEMENTS;
constexpr uint32_t ISAMIN_REDUCE_ELEMENTS = 256U / ISAMIN_FLOAT_BYTES;
constexpr uint32_t ISAMIN_INF_BITS = 0x7F800000U;

class IsaminKernel {
public:
    __aicore__ inline IsaminKernel() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR result, GM_ADDR workspace, const IsaminTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessContiguous();
    __aicore__ inline void ProcessStrided();
    __aicore__ inline void ReduceTile(
        LocalTensor<float>& input, uint32_t validCount, uint32_t alignedCount, uint32_t globalOffset);
    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t validCount, uint32_t alignedCount, bool padded);
    __aicore__ inline void ScanSmall(LocalTensor<float>& input, uint32_t validCount, uint32_t globalOffset);
    __aicore__ inline void UpdateBest(float value, uint32_t index);
    __aicore__ inline void WriteWorkspace();
    __aicore__ inline void Finalize();

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TBuf<TPosition::VECCALC> reduceWork_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    GlobalTensor<float> inputGM_;
    GlobalTensor<float> workspaceGM_;
    GlobalTensor<int32_t> resultGM_;

    uint32_t blockIdx_ = 0;
    uint32_t useCoreNum_ = 0;
    uint32_t startOffset_ = 0;
    uint32_t computeNum_ = 0;
    uint32_t inputSpan_ = 0;
    uint32_t tileSize_ = 0;
    int32_t incx_ = 1;
    float bestValue_ = FLT_MAX;
    uint32_t bestIndex_ = 0;
    bool hasValue_ = false;
};

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    if (align == 0U) {
        return value;
    }
    return (value + align - 1U) / align * align;
}

__aicore__ inline void IsaminKernel::Init(
    GM_ADDR x, GM_ADDR result, GM_ADDR workspace, const IsaminTilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    useCoreNum_ = tiling.useCoreNum;
    startOffset_ = blockIdx_ * tiling.perCoreN;
    computeNum_ = startOffset_ >= tiling.totalN ? 0U : min(tiling.perCoreN, tiling.totalN - startOffset_);
    tileSize_ = tiling.tileSize;
    incx_ = tiling.incx;
    inputSpan_ = tiling.totalN == 0U ? 0U : (tiling.totalN - 1U) * static_cast<uint32_t>(incx_) + 1U;

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), inputSpan_);
    workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), useCoreNum_ * 2U);
    resultGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(result), 1U);

    pipe_->InitBuffer(inputQueue_, 2, tileSize_ * ISAMIN_FLOAT_BYTES);
    const uint32_t partials =
        AlignUp((tileSize_ + ISAMIN_REDUCE_ELEMENTS - 1U) / ISAMIN_REDUCE_ELEMENTS, ISAMIN_BLOCK_ELEMENTS);
    pipe_->InitBuffer(reduceWork_, partials * ISAMIN_FLOAT_BYTES + 32U);
    pipe_->InitBuffer(scalarBuf_, 512U);
}

__aicore__ inline void IsaminKernel::UpdateBest(float value, uint32_t index)
{
    if (value != value) {
        return;
    }
    if (!hasValue_ && value >= FLT_MAX) {
        return;
    }
    if (!hasValue_ || value < bestValue_ || (value == bestValue_ && index < bestIndex_)) {
        bestValue_ = value;
        bestIndex_ = index;
        hasValue_ = true;
    }
}

__aicore__ inline void IsaminKernel::ScanSmall(LocalTensor<float>& input, uint32_t validCount, uint32_t globalOffset)
{
    for (uint32_t i = 0; i < validCount; ++i) {
        const float value = input.GetValue(i);
        const float absValue = value >= 0.0F ? value : -value;
        UpdateBest(absValue, globalOffset + i);
    }
}

__aicore__ inline void IsaminKernel::ReduceTile(
    LocalTensor<float>& input, uint32_t validCount, uint32_t alignedCount, uint32_t globalOffset)
{
    LocalTensor<float> reduceOut = scalarBuf_.Get<float>();
    LocalTensor<float> reduceWork = reduceWork_.Get<float>();
    Abs(input, input, alignedCount);
    PipeBarrier<PIPE_V>();
    ReduceMin(reduceOut, input, reduceWork, static_cast<int32_t>(alignedCount), true);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);

    const float tileValue = reduceOut.GetValue(0);
    const float indexBits = reduceOut.GetValue(1);
    const uint32_t localIndex = *reinterpret_cast<const uint32_t*>(&indexBits);
    if (tileValue != tileValue) {
        ScanSmall(input, validCount, globalOffset);
    } else if (localIndex < validCount) {
        UpdateBest(tileValue, globalOffset + localIndex);
    }
    SetFlag<HardEvent::S_V>(0);
    WaitFlag<HardEvent::S_V>(0);
}

__aicore__ inline void IsaminKernel::ProcessTile(
    uint32_t offset, uint32_t validCount, uint32_t alignedCount, bool padded)
{
    LocalTensor<float> input = inputQueue_.AllocTensor<float>();
    if (padded) {
        const uint8_t padding =
            static_cast<uint8_t>((ISAMIN_BLOCK_ELEMENTS - validCount % ISAMIN_BLOCK_ELEMENTS) % ISAMIN_BLOCK_ELEMENTS);
        DataCopyParams copyParams{1, static_cast<uint16_t>(validCount * ISAMIN_FLOAT_BYTES), 0, 0};
        DataCopyPadParams padParams{true, 0, padding, ISAMIN_INF_BITS};
        DataCopyPad(input, inputGM_[offset], copyParams, padParams);
    } else {
        DataCopy(input, inputGM_[offset], validCount);
    }
    inputQueue_.EnQue(input);
    input = inputQueue_.DeQue<float>();

    if (validCount <= ISAMIN_BLOCK_ELEMENTS) {
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        ScanSmall(input, validCount, offset);
        inputQueue_.FreeTensor(input);
        return;
    }

    ReduceTile(input, validCount, alignedCount, offset);
    inputQueue_.FreeTensor(input);
}

__aicore__ inline void IsaminKernel::ProcessContiguous()
{
    const uint32_t fullTiles = computeNum_ / tileSize_;
    uint32_t processed = 0;
    for (uint32_t tile = 0; tile < fullTiles; ++tile) {
        ProcessTile(startOffset_ + processed, tileSize_, tileSize_, false);
        processed += tileSize_;
    }

    const uint32_t remainder = computeNum_ - processed;
    if (remainder > 0) {
        const uint32_t alignedRemainder = AlignUp(remainder, ISAMIN_BLOCK_ELEMENTS);
        ProcessTile(startOffset_ + processed, remainder, alignedRemainder, true);
    }
}

__aicore__ inline void IsaminKernel::ProcessStrided()
{
    uint32_t processed = 0;
    while (processed < computeNum_) {
        const uint32_t validCount = min(tileSize_, computeNum_ - processed);
        const uint32_t alignedCount = AlignUp(validCount, ISAMIN_BLOCK_ELEMENTS);
        LocalTensor<float> input = inputQueue_.AllocTensor<float>();
        Duplicate<float>(input, FLT_MAX, alignedCount);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        for (uint32_t i = 0; i < validCount; ++i) {
            const uint32_t logicalIndex = startOffset_ + processed + i;
            input.SetValue(i, inputGM_.GetValue(static_cast<uint64_t>(logicalIndex) * static_cast<uint64_t>(incx_)));
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        inputQueue_.EnQue(input);
        input = inputQueue_.DeQue<float>();

        if (validCount <= ISAMIN_BLOCK_ELEMENTS) {
            ScanSmall(input, validCount, startOffset_ + processed);
        } else {
            ReduceTile(input, validCount, alignedCount, startOffset_ + processed);
        }
        inputQueue_.FreeTensor(input);
        processed += validCount;
    }
}

__aicore__ inline void IsaminKernel::WriteWorkspace()
{
    LocalTensor<float> output = scalarBuf_.Get<float>();
    output.SetValue(0, bestValue_);
    output.SetValue(1, *reinterpret_cast<float*>(&bestIndex_));
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyParams copyParams{1, static_cast<uint16_t>(2U * ISAMIN_FLOAT_BYTES), 0, 0};
    DataCopyPad(workspaceGM_[blockIdx_ * 2U], output, copyParams);
}

__aicore__ inline void IsaminKernel::Finalize()
{
    const uint32_t pairCount = useCoreNum_ * 2U;
    const uint32_t alignedPairs = AlignUp(pairCount, ISAMIN_BLOCK_ELEMENTS);
    const uint8_t padding = static_cast<uint8_t>(alignedPairs - pairCount);
    LocalTensor<float> allResults = scalarBuf_.Get<float>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(pairCount * ISAMIN_FLOAT_BYTES), 0, 0};
    DataCopyPadParams padParams{true, 0, padding, ISAMIN_INF_BITS};
    DataCopyPad(allResults, workspaceGM_, copyParams, padParams);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);

    float bestValue = FLT_MAX;
    uint32_t bestIndex = 0;
    bool hasValue = false;
    for (uint32_t core = 0; core < useCoreNum_; ++core) {
        const float value = allResults.GetValue(core * 2U);
        const float indexBits = allResults.GetValue(core * 2U + 1U);
        const uint32_t index = *reinterpret_cast<const uint32_t*>(&indexBits);
        if (value == value && value < FLT_MAX &&
            (!hasValue || value < bestValue || (value == bestValue && index < bestIndex))) {
            bestValue = value;
            bestIndex = index;
            hasValue = true;
        }
    }

    LocalTensor<int32_t> output = scalarBuf_.Get<int32_t>();
    output.SetValue(0, hasValue ? static_cast<int32_t>(bestIndex + 1U) : 1);
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyParams writeParams{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    DataCopyPad(resultGM_, output, writeParams);
}

__aicore__ inline void IsaminKernel::Process()
{
    if (incx_ == 1) {
        ProcessContiguous();
    } else {
        ProcessStrided();
    }
    WriteWorkspace();
    SyncAll();
    if (blockIdx_ == 0U) {
        Finalize();
    }
}

} // namespace

extern "C" __global__ __aicore__ void isamin_kernel(
    GM_ADDR x, GM_ADDR result, GM_ADDR workspace, IsaminTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IsaminKernel op;
    op.Init(x, result, workspace, tiling, &pipe);
    op.Process();
}

void isamin_kernel_do(
    uint8_t* x, uint8_t* result, uint8_t* workspace, const IsaminTilingData& tiling, uint32_t numBlocks, void* stream)
{
    isamin_kernel<<<numBlocks, nullptr, stream>>>(x, result, workspace, tiling);
}
