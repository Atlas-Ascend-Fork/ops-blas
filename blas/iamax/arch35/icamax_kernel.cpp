/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cfloat>
#include <cstdint>

#include "acl/acl.h"
#include "common/helper/kernel_constant.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "icamax_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t WORKSPACE_WORDS_PER_CORE = 2U;
constexpr uint32_t WORKSPACE_VALUE_OFFSET = 0U;
constexpr uint32_t WORKSPACE_INDEX_OFFSET = 1U;
constexpr uint32_t BYTES_PER_FLOAT = sizeof(float);
constexpr uint32_t UB_BLOCK_BYTES = 32U;
constexpr uint32_t FLOATS_PER_UB_BLOCK = UB_BLOCK_BYTES / BYTES_PER_FLOAT;
constexpr uint32_t SIMT_WARP_SIZE = 32U;
constexpr uint32_t SIMT_MAX_WARP_NUM = SIMT_MAX_THREAD_NUM / SIMT_WARP_SIZE;
constexpr uint32_t NAN_PRESENT_FLAG = 0x80000000U;
constexpr uint32_t LOGICAL_INDEX_MASK = 0x7FFFFFFFU;
constexpr uint32_t INVALID_INDEX = UINT32_MAX;

static_assert(
    ICAMAX_SIMD_VECTOR_COMPLEX_COUNT == VECTOR_REG_WIDTH / sizeof(float),
    "Icamax vector group must match the hardware vector width");

union PackedComplex {
    uint64_t packed;
    float components[2];
};

__simd_callee__ inline void IcamaxTrackNan(
    Reg::RegTensor<uint32_t>& flags, Reg::RegTensor<float>& scores, Reg::MaskReg valid)
{
    Reg::MaskReg mask;
    Reg::RegTensor<uint32_t> one;
    Reg::Duplicate(one, 1U);
    Reg::Compare<float, CMPMODE::NE>(mask, scores, scores, valid);
    Reg::Select(flags, one, flags, mask);
}

__simd_callee__ inline void IcamaxLoadSimdScore(Reg::RegTensor<float>& score, __ubuf__ float* input, Reg::MaskReg valid)
{
    constexpr uint32_t VECTOR_FLOAT_COUNT = VECTOR_REG_WIDTH / sizeof(float);
    Reg::RegTensor<float> packed0;
    Reg::RegTensor<float> packed1;
    Reg::RegTensor<float> realValue;
    Reg::RegTensor<float> imagValue;
    Reg::LoadAlign(packed0, input);
    Reg::LoadAlign(packed1, input + VECTOR_FLOAT_COUNT);
    Reg::DeInterleave(realValue, imagValue, packed0, packed1);
    Reg::Abs(realValue, realValue, valid);
    Reg::Abs(imagValue, imagValue, valid);
    Reg::Add(score, realValue, imagValue, valid);
}

__simd_callee__ inline void IcamaxUpdateLane(
    Reg::RegTensor<float>& bestValue, Reg::RegTensor<uint32_t>& bestIndex, Reg::RegTensor<uint32_t>& nextIndex,
    Reg::RegTensor<uint32_t>& nanFlags, __ubuf__ float* input)
{
    Reg::RegTensor<float> score;
    Reg::RegTensor<float> nextBestValue;
    Reg::MaskReg changed;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    IcamaxLoadSimdScore(score, input, fullMask);
    IcamaxTrackNan(nanFlags, score, fullMask);
    Reg::Max(nextBestValue, bestValue, score, fullMask);
    Reg::Compare<float, CMPMODE::NE>(changed, bestValue, nextBestValue, fullMask);
    Reg::Select(bestIndex, nextIndex, bestIndex, changed);
    bestValue = nextBestValue;
}

__simd_callee__ inline void IcamaxMergeLanes(
    Reg::RegTensor<float>& value0, Reg::RegTensor<uint32_t>& index0, Reg::RegTensor<float>& value1,
    Reg::RegTensor<uint32_t>& index1)
{
    Reg::RegTensor<float> nextValue;
    Reg::RegTensor<uint32_t> mergedIndex;
    Reg::MaskReg changed;
    Reg::MaskReg equal;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Max(nextValue, value0, value1, fullMask);
    Reg::Compare<float, CMPMODE::NE>(changed, value0, nextValue, fullMask);
    Reg::Select(index0, index1, index0, changed);
    Reg::Compare<float, CMPMODE::EQ>(equal, value0, value1, fullMask);
    Reg::Min(mergedIndex, index0, index1, indexMask);
    Reg::Select(index0, mergedIndex, index0, equal);
    value0 = nextValue;
}

__simd_callee__ inline void IcamaxReduceTail(
    Reg::RegTensor<float>& bestValue, Reg::RegTensor<uint32_t>& bestIndex, Reg::RegTensor<uint32_t>& nanFlags,
    __ubuf__ float* input, uint32_t count)
{
    constexpr uint32_t VECTOR_FLOAT_COUNT = VECTOR_REG_WIDTH / sizeof(float);
    const uint32_t fullCount = count / VECTOR_FLOAT_COUNT * VECTOR_FLOAT_COUNT;
    uint32_t tailCount = count - fullCount;
    if (tailCount == 0U) {
        return;
    }
    Reg::RegTensor<float> score;
    Reg::RegTensor<float> nextValue;
    Reg::RegTensor<uint32_t> tailIndex;
    Reg::MaskReg changed;
    Reg::MaskReg valid = Reg::UpdateMask<float>(tailCount);
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    IcamaxLoadSimdScore(score, input + fullCount * 2U, valid);
    IcamaxTrackNan(nanFlags, score, valid);
    Reg::Max(nextValue, bestValue, score, valid);
    Reg::Select(nextValue, nextValue, bestValue, valid);
    Reg::Arange((Reg::RegTensor<int32_t>&)tailIndex, 0);
    Reg::Adds(tailIndex, tailIndex, fullCount, indexMask);
    Reg::Compare<float, CMPMODE::NE>(changed, bestValue, nextValue, fullMask);
    Reg::Select(bestIndex, tailIndex, bestIndex, changed);
    bestValue = nextValue;
}

__simd_callee__ inline void IcamaxStoreReduction(
    __ubuf__ float* output, Reg::RegTensor<float>& bestValue, Reg::RegTensor<uint32_t>& bestIndex,
    Reg::RegTensor<uint32_t>& nanFlags)
{
    Reg::RegTensor<float> maximum;
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<uint32_t> anyNan;
    Reg::MaskReg equal;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::UnalignReg unalignReg;
    Reg::ReduceMax(maximum, bestValue, fullMask);
    Reg::StoreUnAlign(output, maximum, unalignReg, 1U);
    Reg::Duplicate(maximum, maximum, fullMask);
    Reg::Compare<float, CMPMODE::EQ>(equal, bestValue, maximum, fullMask);
    Reg::ReduceMin(index, bestIndex, equal);
    Reg::StoreUnAlign((__ubuf__ uint32_t*&)output, index, unalignReg, 1U);
    Reg::ReduceMax(anyNan, nanFlags, indexMask);
    Reg::StoreUnAlign((__ubuf__ uint32_t*&)output, anyNan, unalignReg, 1U);
    Reg::StoreUnAlignPost(output, unalignReg, 0U);
}

__simd_callee__ inline void IcamaxReducePairs(
    Reg::RegTensor<float>& value0, Reg::RegTensor<uint32_t>& index0, Reg::RegTensor<float>& value1,
    Reg::RegTensor<uint32_t>& index1, Reg::RegTensor<uint32_t>& nanFlags, __ubuf__ float* input, uint32_t fullLoopCount)
{
    constexpr uint32_t VECTOR_FLOAT_COUNT = VECTOR_REG_WIDTH / sizeof(float);
    Reg::RegTensor<uint32_t> nextIndex0;
    Reg::RegTensor<uint32_t> nextIndex1;
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(value0, -FLT_MAX);
    Reg::Duplicate(value1, -FLT_MAX);
    Reg::Arange((Reg::RegTensor<int32_t>&)nextIndex0, 0);
    Reg::Adds(nextIndex1, nextIndex0, VECTOR_FLOAT_COUNT, indexMask);
    index0 = nextIndex0;
    index1 = nextIndex1;
    const uint32_t pairLoopCount = fullLoopCount / 2U;
    for (uint32_t i = 0U; i < pairLoopCount; ++i) {
        const uint32_t offset = i * VECTOR_FLOAT_COUNT * 4U;
        IcamaxUpdateLane(value0, index0, nextIndex0, nanFlags, input + offset);
        IcamaxUpdateLane(value1, index1, nextIndex1, nanFlags, input + offset + VECTOR_FLOAT_COUNT * 2U);
        Reg::Adds(nextIndex0, nextIndex0, VECTOR_FLOAT_COUNT * 2U, indexMask);
        Reg::Adds(nextIndex1, nextIndex1, VECTOR_FLOAT_COUNT * 2U, indexMask);
    }
    if ((fullLoopCount & 1U) != 0U) {
        IcamaxUpdateLane(value0, index0, nextIndex0, nanFlags, input + pairLoopCount * VECTOR_FLOAT_COUNT * 4U);
    }
}

__simd_vf__ inline void IcamaxReduceVf(__ubuf__ float* output, __ubuf__ float* interleaved, uint32_t count)
{
    Reg::RegTensor<float> value0;
    Reg::RegTensor<float> value1;
    Reg::RegTensor<uint32_t> index0;
    Reg::RegTensor<uint32_t> index1;
    Reg::RegTensor<uint32_t> nanFlags;
    Reg::Duplicate(nanFlags, 0U);
    const uint32_t fullLoopCount = count / ICAMAX_SIMD_VECTOR_COMPLEX_COUNT;
    IcamaxReducePairs(value0, index0, value1, index1, nanFlags, interleaved, fullLoopCount);
    IcamaxReduceTail(value0, index0, nanFlags, interleaved, count);
    if (fullLoopCount / 2U > 0U) {
        IcamaxMergeLanes(value0, index0, value1, index1);
    }
    IcamaxStoreReduction(output, value0, index0, nanFlags);
}

__simd_callee__ inline void IcamaxInitQuadLane(Reg::RegTensor<float>& value, Reg::RegTensor<uint32_t>& iteration)
{
    Reg::Duplicate(value, -FLT_MAX);
    Reg::Duplicate(iteration, 0U);
}

__simd_callee__ inline void IcamaxReduceQuadLoops(
    Reg::RegTensor<float>& value0, Reg::RegTensor<uint32_t>& iteration0, Reg::RegTensor<float>& value1,
    Reg::RegTensor<uint32_t>& iteration1, Reg::RegTensor<float>& value2, Reg::RegTensor<uint32_t>& iteration2,
    Reg::RegTensor<float>& value3, Reg::RegTensor<uint32_t>& iteration3, Reg::RegTensor<uint32_t>& nanFlags,
    __ubuf__ float* input, uint32_t fullLoopCount)
{
    // Keep iteration IDs per lane; convert them to logical indices after the loop.
    constexpr uint32_t VECTOR_FLOAT_COUNT = VECTOR_REG_WIDTH / sizeof(float);
    Reg::RegTensor<uint32_t> iteration;
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(iteration, 0U);
    const uint32_t quadLoopCount = fullLoopCount / 4U;
    for (uint32_t i = 0U; i < quadLoopCount; ++i) {
        const uint32_t offset = i * VECTOR_FLOAT_COUNT * 8U;
        IcamaxUpdateLane(value0, iteration0, iteration, nanFlags, input + offset);
        IcamaxUpdateLane(value1, iteration1, iteration, nanFlags, input + offset + VECTOR_FLOAT_COUNT * 2U);
        IcamaxUpdateLane(value2, iteration2, iteration, nanFlags, input + offset + VECTOR_FLOAT_COUNT * 4U);
        IcamaxUpdateLane(value3, iteration3, iteration, nanFlags, input + offset + VECTOR_FLOAT_COUNT * 6U);
        Reg::Adds(iteration, iteration, 1U, indexMask);
    }
    const uint32_t remaining = fullLoopCount - quadLoopCount * 4U;
    const uint32_t offset = quadLoopCount * VECTOR_FLOAT_COUNT * 8U;
    if (remaining > 0U) {
        IcamaxUpdateLane(value0, iteration0, iteration, nanFlags, input + offset);
    }
    if (remaining > 1U) {
        IcamaxUpdateLane(value1, iteration1, iteration, nanFlags, input + offset + VECTOR_FLOAT_COUNT * 2U);
    }
    if (remaining > 2U) {
        IcamaxUpdateLane(value2, iteration2, iteration, nanFlags, input + offset + VECTOR_FLOAT_COUNT * 4U);
    }
}

__simd_callee__ inline void IcamaxQuadIndex(
    Reg::RegTensor<uint32_t>& index, Reg::RegTensor<uint32_t>& iteration, uint32_t lane)
{
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::ShiftLefts(index, iteration, static_cast<int16_t>(2), indexMask);
    if (lane != 0U) {
        Reg::Adds(index, index, lane, indexMask);
    }
}

__simd_callee__ inline void IcamaxExpandQuadIndex(Reg::RegTensor<uint32_t>& index)
{
    Reg::RegTensor<uint32_t> offset;
    Reg::MaskReg indexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::ShiftLefts(index, index, static_cast<int16_t>(6), indexMask);
    Reg::Arange((Reg::RegTensor<int32_t>&)offset, 0);
    Reg::Add(index, index, offset, indexMask);
}

__simd_callee__ inline void IcamaxReduceQuadImpl(__ubuf__ float* output, __ubuf__ float* interleaved, uint32_t count)
{
    Reg::RegTensor<float> value0;
    Reg::RegTensor<float> value1;
    Reg::RegTensor<float> value2;
    Reg::RegTensor<float> value3;
    Reg::RegTensor<uint32_t> iteration0;
    Reg::RegTensor<uint32_t> iteration1;
    Reg::RegTensor<uint32_t> iteration2;
    Reg::RegTensor<uint32_t> iteration3;
    Reg::RegTensor<uint32_t> index0;
    Reg::RegTensor<uint32_t> index1;
    Reg::RegTensor<uint32_t> index2;
    Reg::RegTensor<uint32_t> index3;
    Reg::RegTensor<uint32_t> nanFlags;
    Reg::Duplicate(nanFlags, 0U);
    IcamaxInitQuadLane(value0, iteration0);
    IcamaxInitQuadLane(value1, iteration1);
    IcamaxInitQuadLane(value2, iteration2);
    IcamaxInitQuadLane(value3, iteration3);
    IcamaxReduceQuadLoops(
        value0, iteration0, value1, iteration1, value2, iteration2, value3, iteration3, nanFlags, interleaved,
        count / ICAMAX_SIMD_VECTOR_COMPLEX_COUNT);
    IcamaxQuadIndex(index0, iteration0, 0U);
    IcamaxQuadIndex(index1, iteration1, 1U);
    IcamaxQuadIndex(index2, iteration2, 2U);
    IcamaxQuadIndex(index3, iteration3, 3U);
    IcamaxMergeLanes(value0, index0, value1, index1);
    IcamaxMergeLanes(value2, index2, value3, index3);
    IcamaxMergeLanes(value0, index0, value2, index2);
    IcamaxExpandQuadIndex(index0);
    IcamaxReduceTail(value0, index0, nanFlags, interleaved, count);
    IcamaxStoreReduction(output, value0, index0, nanFlags);
}

__simd_vf__ inline void IcamaxReduceQuadVf(__ubuf__ float* output, __ubuf__ float* interleaved, uint32_t count)
{
    IcamaxReduceQuadImpl(output, interleaved, count);
}

__simd_vf__ inline void IcamaxReduceQuadFullVf(__ubuf__ float* output, __ubuf__ float* interleaved)
{
    IcamaxReduceQuadImpl(output, interleaved, ICAMAX_SIMD_TILE_COMPLEX_COUNT);
}

class IcamaxKernel {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR x, GM_ADDR workspace, const IcamaxTilingData& tiling)
    {
        pipe_ = pipe;
        const uint32_t blockId = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t prefixRemainder = blockId < tiling.remainder ? blockId : tiling.remainder;
        const uint32_t remainderUnit = tiling.vectorAligned != 0U ? ICAMAX_SIMD_VECTOR_COMPLEX_COUNT : 1U;
        startOffset_ = blockId * tiling.perCoreN + prefixRemainder * remainderUnit;
        computeNum_ = tiling.perCoreN + (blockId < tiling.remainder ? remainderUnit : 0U);
        if (tiling.vectorAligned != 0U && blockId + 1U == tiling.useCoreNum) {
            computeNum_ += tiling.scalarTail;
        }
        tileSize_ = tiling.tileSize;
        blockId_ = blockId;
        quadReduce_ = computeNum_ >= ICAMAX_SIMD_QUAD_MIN_COMPLEX_COUNT;

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(tiling.totalN) * 2U);
        workspaceGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace), tiling.useCoreNum * WORKSPACE_WORDS_PER_CORE);

        pipe_->InitBuffer(aosBuf_, tileSize_ * 2U * BYTES_PER_FLOAT);
    }

    __aicore__ inline void Process()
    {
        bestValue_ = -FLT_MAX;
        bestIndex_ = INVALID_INDEX;

        if (blockId_ == 0U) {
            const float firstReal = xGM_.GetValue(0U);
            const float firstImag = xGM_.GetValue(1U);
            if (firstReal != firstReal || firstImag != firstImag) {
                bestValue_ = 0.0F;
                bestIndex_ = NAN_PRESENT_FLAG;
                WriteWorkspace();
                return;
            }
        }

        for (uint32_t offset = 0U; offset < computeNum_; offset += tileSize_) {
            const uint32_t currentCount = computeNum_ - offset < tileSize_ ? computeNum_ - offset : tileSize_;
            ProcessTile(offset, currentCount);
            if (bestIndex_ == NAN_PRESENT_FLAG) {
                break;
            }
        }
        WriteWorkspace();
    }

private:
    __aicore__ inline void CopyTile(uint32_t localOffset, uint32_t currentCount)
    {
        const uint32_t inputFloatCount = currentCount * 2U;
        const uint32_t inputOffset = (startOffset_ + localOffset) * 2U;
        LocalTensor<float> aosLocal = aosBuf_.Get<float>();
        const uint32_t alignedFloatCount = inputFloatCount / FLOATS_PER_UB_BLOCK * FLOATS_PER_UB_BLOCK;
        const uint32_t tailFloatCount = inputFloatCount - alignedFloatCount;
        if (tailFloatCount == 0U) {
            DataCopy(aosLocal, xGM_[inputOffset], inputFloatCount);
        } else {
            const uint8_t paddingCount = static_cast<uint8_t>(FLOATS_PER_UB_BLOCK - tailFloatCount);
            DataCopyPadExtParams<float> padParams{true, 0U, paddingCount, 0.0F};
            DataCopyPad(
                aosLocal, xGM_[inputOffset], DataCopyExtParams{1U, inputFloatCount * BYTES_PER_FLOAT, 0U, 0U, 0U},
                padParams);
        }

        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void ProcessTile(uint32_t localOffset, uint32_t currentCount)
    {
        CopyTile(localOffset, currentCount);
        LocalTensor<float> aosLocal = aosBuf_.Get<float>();
        LocalTensor<float> outLocal = aosBuf_.Get<float>();
        if (quadReduce_ && currentCount == ICAMAX_SIMD_TILE_COMPLEX_COUNT) {
            asc_vf_call<IcamaxReduceQuadFullVf>(
                reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(aosLocal.GetPhyAddr()));
        } else if (quadReduce_) {
            asc_vf_call<IcamaxReduceQuadVf>(
                reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(aosLocal.GetPhyAddr()), currentCount);
        } else {
            asc_vf_call<IcamaxReduceVf>(
                reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(aosLocal.GetPhyAddr()), currentCount);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        UpdateTileResult(outLocal, localOffset);
    }

    __aicore__ inline void UpdateTileResult(LocalTensor<float>& outLocal, uint32_t localOffset)
    {
        const float nanBits = outLocal.GetValue(2U);
        if (*reinterpret_cast<const uint32_t*>(&nanBits) != 0U) {
            bestValue_ = 0.0F;
            bestIndex_ = NAN_PRESENT_FLAG;
            return;
        }
        const float tileValue = outLocal.GetValue(0U);
        const float tileIndexBits = outLocal.GetValue(1U);
        const uint32_t tileIndex = *reinterpret_cast<const uint32_t*>(&tileIndexBits);
        if (tileValue >= 0.0F) {
            const uint32_t globalIndex = startOffset_ + localOffset + tileIndex;
            if (bestIndex_ == INVALID_INDEX || tileValue > bestValue_ ||
                (tileValue == bestValue_ && globalIndex < bestIndex_)) {
                bestValue_ = tileValue;
                bestIndex_ = globalIndex;
            }
        }
    }

    __aicore__ inline void WriteWorkspace()
    {
        LocalTensor<float> outLocal = aosBuf_.Get<float>();
        outLocal.SetValue(WORKSPACE_VALUE_OFFSET, bestValue_);
        const float indexBits = *reinterpret_cast<float*>(&bestIndex_);
        outLocal.SetValue(WORKSPACE_INDEX_OFFSET, indexBits);
        DataCopyParams copyParams{1U, static_cast<uint16_t>(WORKSPACE_WORDS_PER_CORE * BYTES_PER_FLOAT), 0U, 0U};
        DataCopyPad(workspaceGM_[blockId_ * WORKSPACE_WORDS_PER_CORE], outLocal, copyParams);
    }

    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECIN> aosBuf_;
    GlobalTensor<float> xGM_;
    GlobalTensor<float> workspaceGM_;
    uint32_t blockId_ = 0U;
    uint32_t startOffset_ = 0U;
    uint32_t computeNum_ = 0U;
    uint32_t tileSize_ = 0U;
    float bestValue_ = -FLT_MAX;
    uint32_t bestIndex_ = INVALID_INDEX;
    bool quadReduce_ = false;
};

__simt_callee__ __aicore__ inline bool IcamaxHasNan(uint32_t index)
{
    return index != INVALID_INDEX && (index & NAN_PRESENT_FLAG) != 0U;
}

__simt_callee__ __aicore__ inline uint32_t IcamaxLogicalIndex(uint32_t index) { return index & LOGICAL_INDEX_MASK; }

__simt_callee__ __aicore__ inline void IcamaxSelectBetter(
    float otherValue, uint32_t otherIndex, float& currentValue, uint32_t& currentIndex)
{
    if (otherIndex == INVALID_INDEX) {
        return;
    }
    if (currentIndex == INVALID_INDEX) {
        currentValue = otherValue;
        currentIndex = otherIndex;
        return;
    }

    const bool otherIsFirstNan = IcamaxHasNan(otherIndex);
    const bool currentIsFirstNan = IcamaxHasNan(currentIndex);
    const uint32_t otherLogicalIndex = IcamaxLogicalIndex(otherIndex);
    const uint32_t currentLogicalIndex = IcamaxLogicalIndex(currentIndex);
    if ((otherIsFirstNan && !currentIsFirstNan) ||
        (otherIsFirstNan == currentIsFirstNan &&
         (otherValue > currentValue || (otherValue == currentValue && otherLogicalIndex < currentLogicalIndex)))) {
        currentValue = otherValue;
        currentIndex = otherIndex;
    }
}

__simt_callee__ __aicore__ inline void IcamaxWarpReduce(float& value, uint32_t& index)
{
    for (uint32_t offset = SIMT_WARP_SIZE >> 1U; offset > 0U; offset >>= 1U) {
        const float otherValue = asc_shfl_down(value, offset, SIMT_WARP_SIZE);
        const uint32_t otherIndex = asc_shfl_down(index, offset, SIMT_WARP_SIZE);
        IcamaxSelectBetter(otherValue, otherIndex, value, index);
    }
}

__simt_callee__ __aicore__ inline void IcamaxSimtTreeReduce(
    float value, uint32_t index, __ubuf__ float* partialValues, __ubuf__ uint32_t* partialIndices)
{
    const uint32_t laneId = threadIdx.x % SIMT_WARP_SIZE;
    const uint32_t warpId = threadIdx.x / SIMT_WARP_SIZE;
    const uint32_t numWarps = (blockDim.x + SIMT_WARP_SIZE - 1U) / SIMT_WARP_SIZE;

    IcamaxWarpReduce(value, index);

    if (laneId == 0U) {
        partialValues[warpId] = value;
        partialIndices[warpId] = index;
    }
    asc_syncthreads();

    if (warpId == 0U) {
        value = laneId < numWarps ? partialValues[laneId] : -FLT_MAX;
        index = laneId < numWarps ? partialIndices[laneId] : INVALID_INDEX;
        const uint32_t upperWarp = laneId + SIMT_WARP_SIZE;
        if (upperWarp < numWarps) {
            IcamaxSelectBetter(partialValues[upperWarp], partialIndices[upperWarp], value, index);
        }
        IcamaxWarpReduce(value, index);
        if (laneId == 0U) {
            partialValues[0] = value;
            partialIndices[0] = index;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void IcamaxSimtCompute(
    uint32_t count, uint32_t blockStart, uint32_t stride, __gm__ const uint64_t* x, __gm__ uint32_t* workspaceSlot)
{
    __ubuf__ float partialValues[SIMT_MAX_WARP_NUM];
    __ubuf__ uint32_t partialIndices[SIMT_MAX_WARP_NUM];

    float bestValue = -FLT_MAX;
    uint32_t bestIndex = INVALID_INDEX;

    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const uint32_t logicalIndex = blockStart + i;
        const uint64_t physicalIndex = static_cast<uint64_t>(logicalIndex) * static_cast<uint64_t>(stride);
        PackedComplex inputValue{};
        inputValue.packed = x[physicalIndex];
        const float realValue = inputValue.components[0];
        const float imagValue = inputValue.components[1];
        const float absReal = realValue >= 0.0F ? realValue : -realValue;
        const float absImag = imagValue >= 0.0F ? imagValue : -imagValue;
        const float score = absReal + absImag;

        if (score != score) {
            IcamaxSelectBetter(0.0F, NAN_PRESENT_FLAG, bestValue, bestIndex);
            continue;
        }
        IcamaxSelectBetter(score, logicalIndex, bestValue, bestIndex);
    }

    IcamaxSimtTreeReduce(bestValue, bestIndex, partialValues, partialIndices);

    if (threadIdx.x == 0U) {
        reinterpret_cast<__gm__ float*>(workspaceSlot)[WORKSPACE_VALUE_OFFSET] = partialValues[0];
        workspaceSlot[WORKSPACE_INDEX_OFFSET] = partialIndices[0];
    }
}


__simt_callee__ __aicore__ inline float IcamaxLoadScore(
    __gm__ const uint64_t* x, uint32_t i, uint32_t stride)
{
    PackedComplex v{};
    v.packed = x[static_cast<uint64_t>(i) * stride];
    const float re = v.components[0];
    const float im = v.components[1];
    return (re >= 0.0F ? re : -re) + (im >= 0.0F ? im : -im);
}

__simt_callee__ __aicore__ inline void IcamaxNanResult(
    __gm__ const uint64_t* x, uint32_t n, uint32_t stride, __gm__ int32_t* result)
{
    __ubuf__ float partialValues[SIMT_MAX_WARP_NUM];
    __ubuf__ uint32_t partialIndices[SIMT_MAX_WARP_NUM];
    // Netlib retains the first element if its score is NaN, and ignores later NaNs.
    const float first = IcamaxLoadScore(x, 0U, stride);
    if (first != first) {
        if (threadIdx.x == 0U) {
            result[0] = 1;
        }
        return;
    }
    float value = first;
    uint32_t index = 0U;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float score = IcamaxLoadScore(x, i, stride);
        if (score == score) {
            IcamaxSelectBetter(score, i, value, index);
        }
    }
    IcamaxSimtTreeReduce(value, index, partialValues, partialIndices);
    if (threadIdx.x == 0U) {
        result[0] = static_cast<int32_t>(partialIndices[0] + 1U);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void IcamaxNanResultVf(
    __gm__ const uint64_t* x, uint32_t n, uint32_t stride, __gm__ int32_t* result)
{
    IcamaxNanResult(x, n, stride, result);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void IcamaxSimtFinalReduce(
    uint32_t count, __gm__ const uint32_t* workspace, __gm__ int32_t* result,
    __gm__ const uint64_t* x, uint32_t n, uint32_t stride)
{
    __ubuf__ float partialValues[SIMT_MAX_WARP_NUM];
    __ubuf__ uint32_t partialIndices[SIMT_MAX_WARP_NUM];

    const uint32_t core = threadIdx.x;
    float value = -FLT_MAX;
    uint32_t index = INVALID_INDEX;
    if (core < count) {
        const uint32_t base = core * WORKSPACE_WORDS_PER_CORE;
        index = workspace[base + WORKSPACE_INDEX_OFFSET];
        if (index != INVALID_INDEX) {
            value = reinterpret_cast<__gm__ const float*>(workspace)[base + WORKSPACE_VALUE_OFFSET];
        }
    }

    IcamaxSimtTreeReduce(value, index, partialValues, partialIndices);

    asc_syncthreads();
    if (IcamaxHasNan(partialIndices[0])) {
        IcamaxNanResult(x, n, stride, result);
        return;
    }
    if (core == 0U) {
        const uint32_t bestIndex = partialIndices[0];
        result[0] = bestIndex != INVALID_INDEX ? static_cast<int32_t>(IcamaxLogicalIndex(bestIndex) + 1U) : 1;
    }
}

} // namespace

extern "C" __global__ __aicore__ void icamax_simt_kernel(GM_ADDR x, GM_ADDR workspace, IcamaxTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const uint32_t blockId = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t count = tiling.perCoreN + (blockId < tiling.remainder ? 1U : 0U);
    const uint32_t prefixRemainder = blockId < tiling.remainder ? blockId : tiling.remainder;
    const uint32_t blockStart = blockId * tiling.perCoreN + prefixRemainder;
    __gm__ uint32_t* workspaceSlot = reinterpret_cast<__gm__ uint32_t*>(workspace) + blockId * WORKSPACE_WORDS_PER_CORE;

    asc_vf_call<IcamaxSimtCompute>(
        dim3{tiling.nthreads, 1, 1}, count, blockStart, tiling.incx, reinterpret_cast<__gm__ const uint64_t*>(x),
        workspaceSlot);
}

extern "C" __global__ __aicore__ void icamax_aiv_kernel(
    GM_ADDR x, GM_ADDR workspace, GM_ADDR result, IcamaxTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IcamaxKernel op;
    op.Init(&pipe, x, workspace, tiling);
    op.Process();

    SyncAll();
    if (GetBlockIdx() == 0) {
        __gm__ const uint32_t* workspaceWords = reinterpret_cast<__gm__ const uint32_t*>(workspace);
        float bestValue = -FLT_MAX;
        uint32_t bestIndex = INVALID_INDEX;
        for (uint32_t core = 0U; core < tiling.useCoreNum; ++core) {
            const uint64_t candidatePacked = reinterpret_cast<__gm__ const uint64_t*>(workspaceWords)[core];
            const uint32_t candidateValueBits = static_cast<uint32_t>(candidatePacked);
            const uint32_t candidateIndex = static_cast<uint32_t>(candidatePacked >> 32U);
            if (candidateIndex == INVALID_INDEX) {
                continue;
            }
            const float candidateValue = *reinterpret_cast<const float*>(&candidateValueBits);
            const bool candidateIsFirstNan = (candidateIndex & NAN_PRESENT_FLAG) != 0U;
            const bool bestIsFirstNan = bestIndex != INVALID_INDEX && (bestIndex & NAN_PRESENT_FLAG) != 0U;
            const uint32_t candidateLogicalIndex = candidateIndex & LOGICAL_INDEX_MASK;
            const uint32_t bestLogicalIndex = bestIndex & LOGICAL_INDEX_MASK;
            if (bestIndex == INVALID_INDEX || (candidateIsFirstNan && !bestIsFirstNan) ||
                (candidateIsFirstNan == bestIsFirstNan &&
                 (candidateValue > bestValue ||
                  (candidateValue == bestValue && candidateLogicalIndex < bestLogicalIndex)))) {
                bestValue = candidateValue;
                bestIndex = candidateIndex;
            }
        }
        if (bestIndex != INVALID_INDEX && (bestIndex & NAN_PRESENT_FLAG) != 0U) {
            asc_vf_call<IcamaxNanResultVf>(
                dim3{SIMT_MIN_THREAD_NUM, 1, 1}, reinterpret_cast<__gm__ const uint64_t*>(x),
                tiling.totalN, tiling.incx, reinterpret_cast<__gm__ int32_t*>(result));
            return;
        }
        reinterpret_cast<__gm__ int32_t*>(result)[0] =
            bestIndex != INVALID_INDEX ? static_cast<int32_t>((bestIndex & LOGICAL_INDEX_MASK) + 1U) : 1;
    }
}

extern "C" __global__ __aicore__ void icamax_reduce_kernel(
    GM_ADDR x, GM_ADDR workspace, GM_ADDR result, IcamaxTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<IcamaxSimtFinalReduce>(
        dim3{SIMT_MIN_THREAD_NUM, 1, 1}, tiling.useCoreNum, reinterpret_cast<__gm__ const uint32_t*>(workspace),
        reinterpret_cast<__gm__ int32_t*>(result), reinterpret_cast<__gm__ const uint64_t*>(x),
        tiling.totalN, tiling.incx);
}

void icamax_kernel_do(
    GM_ADDR x, GM_ADDR result, GM_ADDR workspace, const IcamaxTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    if (tiling.incx == 1U) {
        icamax_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(x, workspace, result, tiling);
    } else {
        icamax_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, workspace, tiling);
        icamax_reduce_kernel<<<1, nullptr, aclStream>>>(x, workspace, result, tiling);
    }
}
