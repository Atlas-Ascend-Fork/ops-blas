/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>

namespace ctrmm {
// A DMA block is 32 bytes; a vector repeat contains 64 FP32 elements.
constexpr uint32_t kFloatBytes = 4U;
constexpr uint32_t kBlockFloats = 8U;
constexpr uint32_t kVectorFloats = 64U;
constexpr uint32_t kVectorRoundUp = 63U;
constexpr uint32_t kBitsPerByte = 8U;
constexpr uint32_t kBitsPerHalfWord = 16U;
constexpr uint32_t kBitsPerWord = 32U;
constexpr uint32_t kFullWordMask = 0xffffffffU;
constexpr uint16_t kFullHalfWordMask = 0xffffU;
constexpr float kFullByteMaskValue = 255.0f;

// Dispatch limits and Cube tiles, in logical complex elements unless noted.
constexpr uint32_t kCubeMinOrder = 64U;
constexpr uint32_t kCubeMinRhs = 16U;
constexpr uint32_t kCubeTile = 128U;
constexpr uint32_t kCubeTileRoundUp = 127U;
constexpr uint32_t kCubeNarrowTile = 64U;
constexpr uint32_t kCompactMaxOrder = 256U;
constexpr uint32_t kWideOutputWidth = 256U;
constexpr uint32_t kWideOutputMinRhs = 512U;
constexpr uint32_t kSmallMaxOrder = 128U;
constexpr uint32_t kCubeFractalRows = 16U;
constexpr uint32_t kCubeGroupK = 32U;
constexpr uint32_t kCubeL0K = 64U;

// Workspace contains original A/X planes, expanded operands and two outputs.
// Offsets below count complete planes; the diagonal has a separate row pitch.
constexpr uint32_t kOriginalAPlanes = 2U;
constexpr uint32_t kCubeAPlanes = 10U;
constexpr uint32_t kCubeXPlanes = 8U;
constexpr uint32_t kVectorXPlanes = 4U;
constexpr uint32_t kCubeOutputXPlane = 6U;
constexpr uint32_t kDiagonalRowFloats = 1024U;
constexpr uint32_t kFlagSlotFloats = 8U;
constexpr uint32_t kExpandedComponents = 8U;
constexpr uint32_t kCompactComponents = 2U;
constexpr uint32_t kRepackElements = 512U;
constexpr uint32_t kRepackSlotShift = 3U;
constexpr uint32_t kRepackPlaneShift = 9U;
constexpr int32_t kRepackSlotMask = 7;
constexpr float kRepackZeroIndex = 1024.0f;
constexpr float kGatherFloatByteScale = 4.0f;
constexpr uint32_t kShortOutputElements = 2048U;
constexpr uint32_t kShortOutputPlaneByteShift = 13U;
constexpr uint32_t kRectanglePlaneByteShift = 14U;
constexpr uint32_t kReferenceColumns = 64U;
constexpr uint32_t kReferencePlaneByteShift = 12U;
constexpr uint32_t kInvalidCachedRow = 0xffffffffU;

// Heuristics select a fast path; they do not restrict the public input domain.
constexpr double kFastAlphaNormLimit = 8.0;
constexpr uint32_t kFastAlphaLimitBits = 0x41000000U; // binary32 8.0
constexpr float kFastInputLimit = 32.0f;
constexpr float kCompactInputLimit = 16.0f;
constexpr uint32_t kCorrelationMinElements = 16U;
constexpr uint32_t kCorrelationMinActive = 32U;
constexpr uint32_t kCorrelationBatchElements = 512U;
constexpr uint32_t kCorrelationPlanes = 3U;
constexpr float kCorrelationSquaredLimit = 0.999f;
constexpr uint32_t kReferenceRequired = 1U;
constexpr uint32_t kScalarFmaRequired = 2U;

// Packed input masks: one bit per component, six planes of 2048 bytes.
constexpr uint32_t kInputMaskBytes = 12288U;
constexpr uint32_t kInputMaskElements = 16384U;
constexpr uint32_t kInputMaskHalfWords = 1024U;
constexpr uint32_t kFastMaskOffset = 2048U;
constexpr uint32_t kFastMaskHalfOffset = 1024U;
constexpr uint32_t kFmaMaxMaskOffset = 4096U;
constexpr uint32_t kFmaMinMaskOffset = 6144U;
constexpr uint32_t kZeroMaskOffset = 8192U;
constexpr uint32_t kFmaMaskOffset = 10240U;
constexpr uint32_t kFmaMaskHalfOffset = 5120U;
constexpr uint32_t kInputAbsBytes = 32768U;
constexpr uint32_t kMaskMinResultOffset = 4096U;  // half elements
constexpr uint32_t kMaskMinScratchOffset = 8192U; // half elements
constexpr float kVectorFmaMaxInput = 65536.0f;
constexpr float kVectorFmaMinInput = 0x1p-24f;
constexpr float kVectorFmaMinAlpha = 0x1p-8f;
constexpr float kVectorFmaMaxAlpha = 256.0f;

// Dekker splitting and midpoint correction for vector IEEE FMA emulation.
constexpr float kDekkerSplitter = 4097.0f;
constexpr uint32_t kSplitLowBits = 12U;
constexpr uint32_t kSplitRoundBias = 0x7ffU;
constexpr uint32_t kSplitHighMask = 0x7ffff000U;
constexpr int32_t kFmaMidpointMask = 0x003fffff;
constexpr float kResidualRoundUp = 1.125f;
constexpr float kResidualRoundDown = 0.875f;

// Tiny-kernel error budgets retain the existing safety margin.
constexpr float kTinyOneTermError = 0x1p-21f;
constexpr float kTinyTwoTermError = 0x1p-20f;
constexpr float kTinyScaledError = 0x1p-19f;
constexpr float kTinyToleranceMargin = 0.9f;
constexpr float kMixedToleranceDivisor = 8192.0f;
constexpr float kTinyAbsoluteBudget = 0.009f;
} // namespace ctrmm
