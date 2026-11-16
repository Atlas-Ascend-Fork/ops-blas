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

#include "chemm_kernel_common.h"

namespace {

// chemm (arch22) ordered AIV variant used for the general alpha/beta case.

// Ordered vector path.
__aicore__ inline __attribute__((always_inline)) void SplitOrderedFmaOperands(
    __ubuf__ float* src00, __ubuf__ float* src01, __ubuf__ float* src10, __ubuf__ float* src11,
    __ubuf__ float* product0, __ubuf__ float* product1, __ubuf__ float* highA0, __ubuf__ float* highA1,
    __ubuf__ float* lowA0, __ubuf__ float* lowA1, __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0,
    __ubuf__ float* lowB1, __ubuf__ float* temporary0, __ubuf__ float* temporary1, uint8_t rows,
    uint8_t src0BlockStride, uint8_t src1BlockStride, uint16_t src0RepeatStride, uint16_t src1RepeatStride)
{
    constexpr float kSplitter = 4097.0f;
    // kSplitClampLimit keeps `clamp(src, +-limit) * kSplitter` safely below FLT_MAX (~3.4028235e38) for any
    // finite FP32 input, including values with |src| close to FLT_MAX. The Dekker-style reconstruction below
    // (`hi = c - (c - src); lo = src - hi`) always satisfies hi + lo == src exactly, because every subtraction
    // step after this block still reads the TRUE, unclamped src00/src01/src10/src11; only the "c = ... *
    // kSplitter" step below reads the clamped copy. For normal-magnitude operands (|src| << kSplitClampLimit,
    // true for every case the public interface exercises in practice) the clamp is a no-op and the split stays
    // bit-identical to before. For the rare extreme-magnitude finite operand (e.g. |src| ~ 1e38, legal per the
    // "finite complex numbers" input contract) that would otherwise overflow `src * 4097` to Inf and poison the
    // result via a subsequent Inf-Inf == NaN, this keeps every intermediate finite while still reconstructing
    // src exactly, at the cost of a slightly coarser (but still correct and finite) high/low split.
    constexpr float kSplitClampLimit = 3.0e34f;
    asc_mul(product0, src00, src10, rows, 1U, src0BlockStride, src1BlockStride, 2U, src0RepeatStride, src1RepeatStride);
    asc_mul(product1, src01, src11, rows, 1U, src0BlockStride, src1BlockStride, 2U, src0RepeatStride, src1RepeatStride);
    asc_min_scalar(highA0, src00, kSplitClampLimit, rows, 1U, src0BlockStride, 2U, src0RepeatStride);
    asc_min_scalar(highA1, src01, kSplitClampLimit, rows, 1U, src0BlockStride, 2U, src0RepeatStride);
    asc_min_scalar(highB0, src10, kSplitClampLimit, rows, 1U, src1BlockStride, 2U, src1RepeatStride);
    asc_min_scalar(highB1, src11, kSplitClampLimit, rows, 1U, src1BlockStride, 2U, src1RepeatStride);
    asc_sync_pipe(PIPE_V);
    asc_max_scalar(highA0, highA0, -kSplitClampLimit, rows, 1U, 1U, 2U, 2U);
    asc_max_scalar(highA1, highA1, -kSplitClampLimit, rows, 1U, 1U, 2U, 2U);
    asc_max_scalar(highB0, highB0, -kSplitClampLimit, rows, 1U, 1U, 2U, 2U);
    asc_max_scalar(highB1, highB1, -kSplitClampLimit, rows, 1U, 1U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_mul_scalar(highA0, highA0, kSplitter, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(highA1, highA1, kSplitter, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(highB0, highB0, kSplitter, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(highB1, highB1, kSplitter, rows, 1U, 1U, 2U, 2U);
    asc_sync_pipe(PIPE_V);

    asc_sub(temporary0, highA0, src00, rows, 1U, 1U, src0BlockStride, 2U, 2U, src0RepeatStride);
    asc_sub(temporary1, highA1, src01, rows, 1U, 1U, src0BlockStride, 2U, 2U, src0RepeatStride);
    asc_sub(lowA0, highB0, src10, rows, 1U, 1U, src1BlockStride, 2U, 2U, src1RepeatStride);
    asc_sub(lowA1, highB1, src11, rows, 1U, 1U, src1BlockStride, 2U, 2U, src1RepeatStride);
    asc_sync_pipe(PIPE_V);

    asc_sub(highA0, highA0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(highA1, highA1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(highB0, highB0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(highB1, highB1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);

    asc_sub(lowA0, src00, highA0, rows, 1U, src0BlockStride, 1U, 2U, src0RepeatStride, 2U);
    asc_sub(lowA1, src01, highA1, rows, 1U, src0BlockStride, 1U, 2U, src0RepeatStride, 2U);
    asc_sub(lowB0, src10, highB0, rows, 1U, src1BlockStride, 1U, 2U, src1RepeatStride, 2U);
    asc_sub(lowB1, src11, highB1, rows, 1U, src1BlockStride, 1U, 2U, src1RepeatStride, 2U);
    asc_sync_pipe(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void ComputeOrderedFmaResidual(
    __ubuf__ float* product0, __ubuf__ float* product1, __ubuf__ float* highA0, __ubuf__ float* highA1,
    __ubuf__ float* lowA0, __ubuf__ float* lowA1, __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0,
    __ubuf__ float* lowB1, __ubuf__ float* error0, __ubuf__ float* error1, __ubuf__ float* temporary0,
    __ubuf__ float* temporary1, uint8_t rows)
{
    asc_mul(error0, highA0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(error1, highA1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_sub(error0, error0, product0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(error1, error1, product1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(temporary0, highA0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(temporary1, highA1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_mul(temporary0, lowA0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(temporary1, lowA1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_mul(temporary0, lowA0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_mul(temporary1, lowA1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(error0, error0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(error1, error1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void AddOrderedFmaProduct(
    __ubuf__ float* dst0, __ubuf__ float* dst1, __ubuf__ float* product0, __ubuf__ float* product1,
    __ubuf__ float* highA0, __ubuf__ float* highA1, __ubuf__ float* lowA0, __ubuf__ float* lowA1,
    __ubuf__ float* highB0, __ubuf__ float* highB1, __ubuf__ float* lowB0, __ubuf__ float* lowB1,
    __ubuf__ float* error0, __ubuf__ float* error1, __ubuf__ float* temporary0, __ubuf__ float* temporary1,
    uint8_t rows)
{
    asc_add(highA0, dst0, product0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(highA1, dst1, product1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_sub(lowA0, highA0, dst0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(lowA1, highA1, dst1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_sub(highB0, highA0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(highB1, highA1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_sub(lowB0, dst0, highB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(lowB1, dst1, highB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(temporary0, product0, lowA0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sub(temporary1, product1, lowA1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(lowB0, lowB0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(lowB1, lowB1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(temporary0, error0, lowB0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(temporary1, error1, lowB1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
    asc_add(dst0, highA0, temporary0, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_add(dst1, highA1, temporary1, rows, 1U, 1U, 1U, 2U, 2U, 2U);
    asc_sync_pipe(PIPE_V);
}

__aicore__ inline __attribute__((always_inline)) void OrderedSoftwareFmaPair(
    __ubuf__ float* dst0, __ubuf__ float* dst1, __ubuf__ float* src00, __ubuf__ float* src01, __ubuf__ float* src10,
    __ubuf__ float* src11, __ubuf__ float* scratch, uint8_t rows, uint8_t src0BlockStride, uint8_t src1BlockStride,
    uint16_t src0RepeatStride, uint16_t src1RepeatStride)
{
    __ubuf__ float* product0 = scratch;
    __ubuf__ float* product1 = product0 + kMicroElements;
    __ubuf__ float* highA0 = product1 + kMicroElements;
    __ubuf__ float* highA1 = highA0 + kMicroElements;
    __ubuf__ float* lowA0 = highA1 + kMicroElements;
    __ubuf__ float* lowA1 = lowA0 + kMicroElements;
    __ubuf__ float* highB0 = lowA1 + kMicroElements;
    __ubuf__ float* highB1 = highB0 + kMicroElements;
    __ubuf__ float* lowB0 = highB1 + kMicroElements;
    __ubuf__ float* lowB1 = lowB0 + kMicroElements;
    __ubuf__ float* error0 = lowB1 + kMicroElements;
    __ubuf__ float* error1 = error0 + kMicroElements;
    __ubuf__ float* temporary0 = error1 + kMicroElements;
    __ubuf__ float* temporary1 = temporary0 + kMicroElements;
    SplitOrderedFmaOperands(
        src00, src01, src10, src11, product0, product1, highA0, highA1, lowA0, lowA1, highB0, highB1, lowB0, lowB1,
        temporary0, temporary1, rows, src0BlockStride, src1BlockStride, src0RepeatStride, src1RepeatStride);
    ComputeOrderedFmaResidual(
        product0, product1, highA0, highA1, lowA0, lowA1, highB0, highB1, lowB0, lowB1, error0, error1, temporary0,
        temporary1, rows);
    AddOrderedFmaProduct(
        dst0, dst1, product0, product1, highA0, highA1, lowA0, lowA1, highB0, highB1, lowB0, lowB1, error0, error1,
        temporary0, temporary1, rows);
}

template <uint32_t kK>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivK(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* fmaScratch, uint8_t rows)
{
    constexpr uint32_t kVectorOffset = kK * kMicro;
    constexpr uint32_t kBroadcastOffset = kK * kMicro * kFp32DataBlockElements;
    __ubuf__ float* xr = broadcastReal + kBroadcastOffset;
    __ubuf__ float* xi = broadcastImag + kBroadcastOffset;
    __ubuf__ float* yr = vectorReal + kVectorOffset;
    __ubuf__ float* yi = vectorImag + kVectorOffset;
    __ubuf__ float* negativeYi = negativeVectorImag + kVectorOffset;
    OrderedSoftwareFmaPair(sumReal, sumImag, yr, yi, xr, xr, fmaScratch, rows, 1U, 0U, 0U, 1U);
    OrderedSoftwareFmaPair(sumReal, sumImag, negativeYi, yr, xi, xi, fmaScratch, rows, 1U, 0U, 0U, 1U);
}

template <uint32_t kK, uint32_t kEnd>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivRange(
    __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* fmaScratch, uint8_t rows)
{
    AccumulateOrderedAivK<kK>(
        vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, fmaScratch, rows);
    if constexpr (kK + 1U < kEnd) {
        AccumulateOrderedAivRange<kK + 1U, kEnd>(
            vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, fmaScratch,
            rows);
    }
}

__aicore__ inline __attribute__((always_inline)) uint32_t OrderedAivKBlock(uint32_t remaining)
{
    constexpr uint32_t kBlock = 512U;
    if (remaining >= 2U * kBlock) {
        return kBlock;
    }
    if (remaining > kBlock) {
        constexpr uint32_t kUnroll = 8U;
        return ((remaining / 2U + kUnroll - 1U) / kUnroll) * kUnroll;
    }
    return remaining;
}

template <uint32_t kEnd>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivEnd(
    uint32_t begin, __ubuf__ float* vectorReal, __ubuf__ float* vectorImag, __ubuf__ float* negativeVectorImag,
    __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* fmaScratch, uint8_t rows)
{
    if (begin == 8U) {
        if constexpr (kEnd > 8U) {
            AccumulateOrderedAivRange<8U, kEnd>(
                vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, fmaScratch,
                rows);
        }
    } else {
        AccumulateOrderedAivRange<0U, kEnd>(
            vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, fmaScratch,
            rows);
    }
}

__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivSegment(
    uint32_t begin, uint32_t end, __ubuf__ float* vectorReal, __ubuf__ float* vectorImag,
    __ubuf__ float* negativeVectorImag, __ubuf__ float* broadcastReal, __ubuf__ float* broadcastImag,
    __ubuf__ float* sumReal, __ubuf__ float* sumImag, __ubuf__ float* fmaScratch, uint8_t rows)
{
#define CHEMM_ORDERED_END(value)                                                                               \
    case value:                                                                                                \
        AccumulateOrderedAivEnd<value>(                                                                        \
            begin, vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag, \
            fmaScratch, rows);                                                                                 \
        break
    switch (end) {
        CHEMM_ORDERED_END(1U);
        CHEMM_ORDERED_END(2U);
        CHEMM_ORDERED_END(3U);
        CHEMM_ORDERED_END(4U);
        CHEMM_ORDERED_END(5U);
        CHEMM_ORDERED_END(6U);
        CHEMM_ORDERED_END(7U);
        CHEMM_ORDERED_END(8U);
        CHEMM_ORDERED_END(9U);
        CHEMM_ORDERED_END(10U);
        CHEMM_ORDERED_END(11U);
        CHEMM_ORDERED_END(12U);
        CHEMM_ORDERED_END(13U);
        CHEMM_ORDERED_END(14U);
        CHEMM_ORDERED_END(15U);
        default:
            AccumulateOrderedAivEnd<16U>(
                begin, vectorReal, vectorImag, negativeVectorImag, broadcastReal, broadcastImag, sumReal, sumImag,
                fmaScratch, rows);
            break;
    }
#undef CHEMM_ORDERED_END
}

__aicore__ inline __attribute__((always_inline)) void InitializeOrderedAivOutput(
    const ChemmAivTilingData& tiling, __ubuf__ float* cReal, __ubuf__ float* cImag, __ubuf__ float* outReal,
    __ubuf__ float* outImag, __ubuf__ float* coefficients, __ubuf__ float* fmaScratch, uint8_t rows)
{
    if (tiling.betaReal == 0.0f && tiling.betaImag == 0.0f) {
        asc_duplicate(outReal, 0.0f, rows, 1U, 2U);
        asc_duplicate(outImag, 0.0f, rows, 1U, 2U);
        asc_sync_pipe(PIPE_V);
        return;
    }
    asc_mul_scalar(outReal, cReal, tiling.betaReal, rows, 1U, 1U, 2U, 2U);
    asc_mul_scalar(outImag, cImag, tiling.betaReal, rows, 1U, 1U, 2U, 2U);
    asc_duplicate(coefficients, -tiling.betaImag, rows, 1U, 2U);
    asc_duplicate(coefficients + kMicroElements, tiling.betaImag, rows, 1U, 2U);
    asc_sync_pipe(PIPE_V);
    OrderedSoftwareFmaPair(
        outReal, outImag, cImag, cReal, coefficients, coefficients + kMicroElements, fmaScratch, rows, 1U, 1U, 2U, 2U);
}

__aicore__ inline __attribute__((always_inline)) void ApplyOrderedAivBlock(
    const ChemmAivTilingData& tiling, __ubuf__ float* sumReal, __ubuf__ float* sumImag, __ubuf__ float* outReal,
    __ubuf__ float* outImag, __ubuf__ float* coefficients, __ubuf__ float* fmaScratch, uint8_t rows)
{
    asc_duplicate(coefficients, tiling.alphaReal, rows, 1U, 2U);
    asc_duplicate(coefficients + kMicroElements, tiling.alphaImag, rows, 1U, 2U);
    asc_sync_pipe(PIPE_V);
    OrderedSoftwareFmaPair(
        outReal, outImag, sumReal, sumReal, coefficients, coefficients + kMicroElements, fmaScratch, rows, 1U, 1U, 2U,
        2U);
    asc_duplicate(coefficients, -tiling.alphaImag, rows, 1U, 2U);
    asc_duplicate(coefficients + kMicroElements, tiling.alphaReal, rows, 1U, 2U);
    asc_sync_pipe(PIPE_V);
    OrderedSoftwareFmaPair(
        outReal, outImag, sumImag, sumImag, coefficients, coefficients + kMicroElements, fmaScratch, rows, 1U, 1U, 2U,
        2U);
}

__aicore__ inline __attribute__((always_inline)) void PrepareOrderedAivOutput(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validCols,
    uint8_t rowRepeats, __ubuf__ uint8_t* masks, __ubuf__ float* outputTile, __ubuf__ float* xReal,
    __ubuf__ float* xImag, __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* broadcastReal,
    __ubuf__ float* fmaScratch)
{
    asc_set_mask_norm();
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    asc_copy_gm2ub_align(masks, MaskGm(tiling), 96U);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    if (tiling.betaReal != 0.0f || tiling.betaImag != 0.0f) {
        LoadOutputTile16<false>(
            c, row, col, static_cast<uint32_t>(tiling.ldc), static_cast<uint32_t>(tiling.m),
            static_cast<uint32_t>(tiling.n), outputTile);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        CInterleavedToPlanes(outputTile, xReal, xImag);
        asc_sync_pipe(PIPE_V);
    }
    asc_set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
    InitializeOrderedAivOutput(tiling, xReal, xImag, outReal, outImag, broadcastReal, fmaScratch, rowRepeats);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivFeed(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validCols,
    uint8_t rowRepeats, uint32_t blockBegin, uint32_t blockEnd, uint32_t feed, __ubuf__ float* rawPair,
    __ubuf__ float* xReal, __ubuf__ float* xImag, __ubuf__ float* yReal, __ubuf__ float* yImag,
    __ubuf__ float* xTransReal, __ubuf__ float* xTransImag, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negativeYImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* fmaScratch, __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    asc_set_vector_mask(0ULL, kFullFp32Mask);
    PrepareGridPair<kRight, kLower, false, false>(tiling, row, feed, feed, col, rawPair, rawPair + kComplexTileFloats);
    SubmitGridPair<0U>();
    CopyGridPair<kRight, kLower, false, false>(
        a, b, tiling, row, feed, feed, col, rawPair, rawPair + kComplexTileFloats);
    CommitGridPair<0U>();
    WaitGridPair<0U>();
    DecodeGridPair<kRight, kLower, 0U, false>(
        tiling, row, feed, feed, col, rawPair, rawPair + kComplexTileFloats, xReal, xImag, yReal, yImag, xTransReal,
        xTransImag, triangleMask, diagonalMask);
    asc_sync_pipe(PIPE_V);
    TransposePlane16(xReal, xTransReal);
    TransposePlane16(xImag, xTransImag);
    asc_sync_pipe(PIPE_V);
    BroadcastAivRange<0U, kMicro, kMicro>(xTransReal, xTransImag, broadcastReal, broadcastImag, 0U);
    asc_mul_scalar(negativeYImag, yImag, -1.0f, kVectorRepeats, 1U, 1U, 8U, 8U);
    asc_sync_pipe(PIPE_V);
    asc_set_vector_mask(0ULL, (1ULL << validCols) - 1ULL);
    const uint32_t segmentBegin = blockBegin > feed ? blockBegin - feed : 0U;
    const uint32_t segmentEnd = blockEnd - feed < kMicro ? blockEnd - feed : kMicro;
    AccumulateOrderedAivSegment(
        segmentBegin, segmentEnd, yReal, yImag, negativeYImag, broadcastReal, broadcastImag, sumReal, sumImag,
        fmaScratch, rowRepeats);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void AccumulateOrderedAivBlocks(
    __gm__ float* a, __gm__ float* b, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validCols,
    uint8_t rowRepeats, __ubuf__ float* rawPair, __ubuf__ float* xReal, __ubuf__ float* xImag, __ubuf__ float* yReal,
    __ubuf__ float* yImag, __ubuf__ float* xTransReal, __ubuf__ float* xTransImag, __ubuf__ float* broadcastReal,
    __ubuf__ float* broadcastImag, __ubuf__ float* negativeYImag, __ubuf__ float* sumReal, __ubuf__ float* sumImag,
    __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* fmaScratch, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const uint32_t dimA = static_cast<uint32_t>(tiling.dimA);
    uint32_t blockBegin = 0U;
    while (blockBegin < dimA) {
        const uint32_t blockEnd = blockBegin + OrderedAivKBlock(dimA - blockBegin);
        uint32_t feed = blockBegin & ~(kMicro - 1U);
        asc_duplicate(sumReal, 0.0f, rowRepeats, 1U, 2U);
        asc_duplicate(sumImag, 0.0f, rowRepeats, 1U, 2U);
        asc_sync_pipe(PIPE_V);
        while (feed < blockEnd) {
            AccumulateOrderedAivFeed<kRight, kLower>(
                a, b, tiling, row, col, validCols, rowRepeats, blockBegin, blockEnd, feed, rawPair, xReal, xImag, yReal,
                yImag, xTransReal, xTransImag, broadcastReal, broadcastImag, negativeYImag, sumReal, sumImag,
                fmaScratch, triangleMask, diagonalMask);
            feed += kMicro;
        }
        ApplyOrderedAivBlock(tiling, sumReal, sumImag, outReal, outImag, broadcastReal, fmaScratch, rowRepeats);
        blockBegin = blockEnd;
    }
}

__aicore__ inline __attribute__((always_inline)) void OrderedAivTile(
    const ChemmAivTilingData& tiling, uint32_t& row, uint32_t& col, uint32_t& validRows, uint32_t& validCols)
{
    const uint32_t tile = tiling.tileBase + static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t tileCol = tile / tiling.tileRows;
    const uint32_t tileRow = tile - tileCol * tiling.tileRows;
    row = tileRow * kMicro;
    col = tileCol * kMicro;
    validRows = static_cast<uint32_t>(tiling.m) - row < kMicro ? static_cast<uint32_t>(tiling.m) - row : kMicro;
    validCols = static_cast<uint32_t>(tiling.n) - col < kMicro ? static_cast<uint32_t>(tiling.n) - col : kMicro;
}

__aicore__ inline __attribute__((always_inline)) void StoreOrderedAivOutput(
    __gm__ float* c, const ChemmAivTilingData& tiling, uint32_t row, uint32_t col, uint32_t validRows,
    uint32_t validCols, __ubuf__ float* outReal, __ubuf__ float* outImag, __ubuf__ float* outputTile)
{
    PlanesToCInterleavedStride(outReal, outImag, outputTile, kComplexColumnFloats);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    StoreComplexTail<kComplexColumnFloats>(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(tiling.ldc)), outputTile, validCols, validRows,
        static_cast<uint32_t>(tiling.ldc));
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

template <bool kRight, bool kLower>
__aicore__ inline __attribute__((always_inline)) void RunOrderedAiv(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, const ChemmAivTilingData& tiling, __ubuf__ float* scratch)
{
    constexpr uint32_t kBroadcastElements = kMicro * kFp32DataBlockElements;
    __ubuf__ float* rawPair = scratch;
    __ubuf__ float* xPlanes = rawPair + 2U * kComplexTileFloats;
    __ubuf__ float* yPlanes = xPlanes + 2U * kMicroElements;
    __ubuf__ float* xTranspose = yPlanes + 2U * kMicroElements;
    __ubuf__ float* broadcasts = xTranspose + 2U * kMicroElements;
    __ubuf__ float* negativeYImag = broadcasts + 2U * kMicro * kBroadcastElements;
    __ubuf__ float* sums = negativeYImag + kMicroElements;
    __ubuf__ float* output = sums + 2U * kMicroElements;
    __ubuf__ float* outputTile = output + 2U * kMicroElements;
    __ubuf__ float* fmaScratch = outputTile + kComplexTileFloats;
    __ubuf__ uint8_t* masks = reinterpret_cast<__ubuf__ uint8_t*>(fmaScratch + 14U * kMicroElements);

    __ubuf__ float* xReal = xPlanes;
    __ubuf__ float* xImag = xPlanes + kMicroElements;
    __ubuf__ float* yReal = yPlanes;
    __ubuf__ float* yImag = yPlanes + kMicroElements;
    __ubuf__ float* xTransReal = xTranspose;
    __ubuf__ float* xTransImag = xTranspose + kMicroElements;
    __ubuf__ float* broadcastReal = broadcasts;
    __ubuf__ float* broadcastImag = broadcasts + kMicro * kBroadcastElements;
    __ubuf__ float* sumReal = sums;
    __ubuf__ float* sumImag = sums + kMicroElements;
    __ubuf__ float* outReal = output;
    __ubuf__ float* outImag = output + kMicroElements;
    __ubuf__ uint8_t* triangleMask = masks + (kLower ? 32U : 0U);
    __ubuf__ uint8_t* diagonalMask = masks + 64U;
    uint32_t row;
    uint32_t col;
    uint32_t validRows;
    uint32_t validCols;
    OrderedAivTile(tiling, row, col, validRows, validCols);
    const uint8_t rowRepeats = static_cast<uint8_t>(validRows);
    PrepareOrderedAivOutput(
        c, tiling, row, col, validCols, rowRepeats, masks, outputTile, xReal, xImag, outReal, outImag, broadcastReal,
        fmaScratch);
    AccumulateOrderedAivBlocks<kRight, kLower>(
        a, b, tiling, row, col, validCols, rowRepeats, rawPair, xReal, xImag, yReal, yImag, xTransReal, xTransImag,
        broadcastReal, broadcastImag, negativeYImag, sumReal, sumImag, outReal, outImag, fmaScratch, triangleMask,
        diagonalMask);
    StoreOrderedAivOutput(c, tiling, row, col, validRows, validCols, outReal, outImag, outputTile);
}

} // namespace
