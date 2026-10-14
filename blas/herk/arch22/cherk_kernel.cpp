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
__aicore__ inline void CopyGmToUb(AscendC::LocalTensor<float> dst, AscendC::GlobalTensor<float> src, uint32_t count)
{
    gm_to_ub_align<ArchType::ASCEND_V220, float>(dst, src, 0, 1, count * sizeof(float), 0, 0, 0, 0);
}

__aicore__ inline void CopyUbToGm(AscendC::GlobalTensor<float> dst, AscendC::LocalTensor<float> src, uint32_t count)
{
    ub_to_gm_align<ArchType::ASCEND_V220, float>(dst, src, 0, 1, count * sizeof(float), 0, 0, 0, 0);
}

__aicore__ inline void InitCherkTinyGlobals(
    AscendC::GlobalTensor<float>& a, AscendC::GlobalTensor<float>& c, AscendC::GlobalTensor<uint32_t>& metadata,
    GM_ADDR aGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData& tiling, uint32_t n)
{
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm), static_cast<uint64_t>(tiling.lda) * n * 2ULL);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm), static_cast<uint64_t>(tiling.ldc) * n * 2ULL);
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
}

__aicore__ inline void LoadCherkOldColumn(
    AscendC::LocalTensor<float>& aos, AscendC::LocalTensor<float>& oldReal, AscendC::LocalTensor<float>& oldImag,
    AscendC::GlobalTensor<float>& c, uint64_t cOffset, uint32_t count, uint32_t planeSize, float beta, event_t vToMte2,
    event_t mte2ToV)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2);
    AscendC::Duplicate(aos, 0.0f, planeSize * 2U);
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t oldBytes = count * 2U * sizeof(float);
    const uint32_t alignedBytes = (oldBytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams copy{1U, oldBytes, 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> pad{
        alignedBytes != oldBytes, 0U, static_cast<uint8_t>((alignedBytes - oldBytes) / sizeof(float)), 0.0f};
    AscendC::DataCopyPad(aos, c[cOffset], copy, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    uint64_t reserved = 0ULL;
    AscendC::GatherMask(oldReal, aos, 1U, false, 0U, {1U, 1U, 8U, 8U}, reserved);
    AscendC::GatherMask(oldImag, aos, 2U, false, 0U, {1U, 1U, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(oldReal, oldReal, beta, planeSize);
    AscendC::Muls(oldImag, oldImag, beta, planeSize);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CherkScaleUpperTriangle(
    GM_ADDR betaGm, GM_ADDR cGm, const CherkArch22TilingData& tiling, uint32_t firstCol, uint32_t columnCount)
{
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(cGm);
    const uint32_t endCol = Min<uint32_t>(tiling.n, firstCol + columnCount);
    for (uint32_t col = firstCol; col < endCol; ++col) {
        for (uint32_t row = 0U; row <= col; ++row) {
            const uint64_t index = (static_cast<uint64_t>(col) * tiling.ldc + row) * 2ULL;
            c[index] = beta == 0.0f ? 0.0f : beta * c[index];
            c[index + 1ULL] = row == col || beta == 0.0f ? 0.0f : beta * c[index + 1ULL];
        }
    }
}

template <uint32_t N>
__aicore__ inline void CherkTinyReferenceFallback(
    float alpha, float beta, GM_ADDR aGm, GM_ADDR cGm, const CherkArch22TilingData& tiling)
{
    __gm__ float* cPtr = reinterpret_cast<__gm__ float*>(cGm);
    __gm__ const float* aPtr = reinterpret_cast<__gm__ const float*>(aGm);
    uint32_t blocks = static_cast<uint32_t>(AscendC::GetBlockNum());
    if (blocks == 0U) {
        blocks = 1U;
    }
    for (uint32_t col = static_cast<uint32_t>(AscendC::GetBlockIdx()); col < N; col += blocks) {
        for (uint32_t row = 0U; row <= col; ++row) {
            volatile float sumReal = 0.0f;
            volatile float sumImag = 0.0f;
            for (uint32_t l = 0U; l < N; ++l) {
                const uint64_t rowIndex = (static_cast<uint64_t>(l) * tiling.lda + row) * 2ULL;
                const uint64_t colIndex = (static_cast<uint64_t>(l) * tiling.lda + col) * 2ULL;
                const float ar = aPtr[rowIndex];
                const float ai = aPtr[rowIndex + 1ULL];
                const float br = aPtr[colIndex];
                const float bi = aPtr[colIndex + 1ULL];
                const volatile float rr = ar * br;
                const volatile float ii = ai * bi;
                const volatile float ri = ai * br;
                const volatile float ir = ar * bi;
                sumReal = sumReal + (rr + ii);
                sumImag = sumImag + (ri - ir);
            }
            // HERK alpha is real. Do not mix the components: zero times
            // a nonfinite imaginary sum would contaminate the real result.
            const float scaledReal = alpha * sumReal;
            const float scaledImag = alpha * sumImag;
            const uint64_t index = (static_cast<uint64_t>(col) * tiling.ldc + row) * 2ULL;
            const float oldReal = cPtr[index];
            const float oldImag = cPtr[index + 1ULL];
            cPtr[index] = beta == 0.0f ? scaledReal : scaledReal + beta * oldReal;
            cPtr[index + 1ULL] = row == col ? 0.0f : (beta == 0.0f ? scaledImag : scaledImag + beta * oldImag);
        }
    }
}
__aicore__ inline CherkCopyEvents GetCherkCopyEvents()
{
    return {
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V))};
}

__aicore__ inline uint32_t CherkNormalizeBlockCount(uint32_t blocks) { return blocks == 0U ? 1U : blocks; }

} // namespace

// Vector kernels, general packing and host dispatch.

__aicore__ inline void ComputeSmallElement(
    uint32_t row, uint32_t col, bool transN, const CherkArch22TilingData& tiling, __gm__ const float* a, float& real,
    float& imag)
{
    real = 0.0f;
    imag = 0.0f;
    for (uint32_t l = 0U; l < tiling.k; ++l) {
        const uint64_t rowOffset = transN ? (static_cast<uint64_t>(l) * tiling.lda + row) * 2ULL :
                                            (static_cast<uint64_t>(row) * tiling.lda + l) * 2ULL;
        const uint64_t colOffset = transN ? (static_cast<uint64_t>(l) * tiling.lda + col) * 2ULL :
                                            (static_cast<uint64_t>(col) * tiling.lda + l) * 2ULL;
        const float ar = a[rowOffset];
        const float ai = a[rowOffset + 1ULL];
        const float br = a[colOffset];
        const float bi = a[colOffset + 1ULL];
        real += ar * br + ai * bi;
        imag += transN ? ai * br - ar * bi : ar * bi - ai * br;
    }
}

__aicore__ inline void StoreSmallElement(
    uint32_t row, uint32_t rowBegin, float real, float imag, AscendC::LocalTensor<float>& output)
{
    const uint32_t local = (row - rowBegin) * 2U;
    uint64_t realMask[2] = {1ULL << local, 0ULL};
    uint64_t imagMask[2] = {1ULL << (local + 1U), 0ULL};
    AscendC::Duplicate(output, real, realMask, 1U, 1U, 8U);
    AscendC::Duplicate(output, imag, imagMask, 1U, 1U, 8U);
}

__aicore__ inline void ProcessSmallColumn(
    uint32_t col, float alpha, float beta, bool hasProduct, bool upper, bool transN,
    const CherkArch22TilingData& tiling, __gm__ const float* a, __gm__ float* c, AscendC::LocalTensor<float>& output,
    event_t vectorToMte3, event_t mte3ToVector)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVector);
    const uint32_t rowBegin = upper ? 0U : col;
    const uint32_t rowEnd = upper ? col + 1U : tiling.n;
    for (uint32_t row = rowBegin; row < rowEnd; ++row) {
        float real = 0.0f;
        float imag = 0.0f;
        if (hasProduct) {
            ComputeSmallElement(row, col, transN, tiling, a, real, imag);
            real *= alpha;
            imag *= alpha;
        }
        const uint64_t cOffset = (static_cast<uint64_t>(col) * tiling.ldc + row) * 2ULL;
        if (beta != 0.0f) {
            real += beta * c[cOffset];
            imag += beta * c[cOffset + 1ULL];
        }
        if (row == col) {
            imag = 0.0f;
        }
        StoreSmallElement(row, rowBegin, real, imag, output);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vectorToMte3);
    const uint64_t outputOffset = (static_cast<uint64_t>(col) * tiling.ldc + rowBegin) * 2ULL;
    const AscendC::DataCopyExtParams copy{
        1U, static_cast<uint32_t>((rowEnd - rowBegin) * 2U * sizeof(float)), 0U, 0U, 0U};
    AscendC::GlobalTensor<float> outputGm;
    outputGm.SetGlobalBuffer(c + outputOffset, static_cast<uint64_t>(rowEnd - rowBegin) * 2ULL);
    AscendC::DataCopyPad(outputGm, output, copy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVector);
}

/** Workspace-free single-launch path for the n,k <= 32 bucket. */
extern "C" __global__ __aicore__ void cherk_arch22_small_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t MAX_N = 32U;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf;
    pipe.InitBuffer(outputBuf, MAX_N * 2U * sizeof(float));
    AscendC::LocalTensor<float> output = outputBuf.Get<float>();

    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    const bool hasProduct = tiling.k != 0U && alpha != 0.0f;
    if (!hasProduct && beta == 1.0f) {
        return;
    }

    __gm__ const float* a = reinterpret_cast<__gm__ const float*>(aGm);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(cGm);
    const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t blocks = CherkNormalizeBlockCount(static_cast<uint32_t>(AscendC::GetBlockNum()));
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    const bool transN = tiling.trans == ACLBLAS_OP_N;

    const event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    const event_t mte3ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVector);

    for (uint32_t col = block; col < tiling.n; col += blocks) {
        ProcessSmallColumn(
            col, alpha, beta, hasProduct, upper, transN, tiling, a, c, output, vectorToMte3, mte3ToVector);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVector);
}
/** Register/GM-only path for the 1x1 Hermitian update. */
extern "C" __global__ __aicore__ void cherk_arch22_tiny1_direct_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    __gm__ const float* a = reinterpret_cast<__gm__ const float*>(aGm);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(cGm);
    if (alpha == 0.0f && beta == 1.0f) {
        return;
    }
    if (alpha == 0.0f) {
        c[0] = beta == 0.0f ? 0.0f : beta * c[0];
        c[1] = 0.0f;
        return;
    }
    float real = alpha * (a[0] * a[0] + a[1] * a[1]);
    if (beta != 0.0f) {
        real += beta * c[0];
    }
    c[0] = real;
    c[1] = 0.0f;
}
/** Compile-time-sized one-column-per-AIV paths for the 8/16/32 square family. */
template <uint32_t ELEMENTS>
__aicore__ inline bool CherkArch22TinyVectorPlaneIsExtreme(
    const AscendC::LocalTensor<float>& src, const AscendC::LocalTensor<float>& scratch,
    const AscendC::LocalTensor<float>& threshold, const AscendC::LocalTensor<uint8_t>& mask)
{
    const AscendC::LocalTensor<uint64_t> words = mask.ReinterpretCast<uint64_t>();
    AscendC::Duplicate(threshold, 0x1p126f, ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs(scratch, src, ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compare(mask, scratch, threshold, AscendC::CMPMODE::GE, ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t word = 0U; word < ELEMENTS / 64U; ++word) {
        if (words.GetValue(word) != 0ULL) {
            return true;
        }
    }
    AscendC::Compare(mask, src, src, AscendC::CMPMODE::NE, ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t word = 0U; word < ELEMENTS / 64U; ++word) {
        if (words.GetValue(word) != 0ULL) {
            return true;
        }
    }
    return false;
}

namespace {
constexpr uint32_t TINY_VECTOR_MAX_N = 32U;
constexpr uint32_t TINY_VECTOR_ROWS = 64U;
constexpr uint32_t TINY_VECTOR_BATCH_COLS = 8U;
constexpr uint32_t TINY_VECTOR_BATCH_ELEMENTS = TINY_VECTOR_BATCH_COLS * TINY_VECTOR_ROWS;
constexpr uint32_t TINY_VECTOR_COEFFICIENT_PLANE = TINY_VECTOR_BATCH_COLS * 8U * 4U;

template <uint32_t N>
struct TinyVectorContext {
    AscendC::LocalTensor<float> aos;
    AscendC::LocalTensor<float> realA;
    AscendC::LocalTensor<float> imagA;
    AscendC::LocalTensor<float> realOut;
    AscendC::LocalTensor<float> imagOut;
    AscendC::LocalTensor<float> productReal;
    AscendC::LocalTensor<float> productImag;
    AscendC::LocalTensor<float> oldReal;
    AscendC::LocalTensor<float> oldImag;
    AscendC::LocalTensor<float> coefficientReal;
    AscendC::LocalTensor<float> coefficientImag;
    AscendC::LocalTensor<uint32_t> interleaveOffsets;
    AscendC::LocalTensor<float> outputStorage;
    AscendC::LocalTensor<float> extremeScratch;
    AscendC::LocalTensor<float> extremeThreshold;
    AscendC::LocalTensor<uint8_t> extremeMask;
    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> c;
    AscendC::GlobalTensor<uint32_t> metadata;
    CherkCopyEvents events;
};

template <uint32_t N>
__aicore__ inline void InitTinyVectorStorage(
    AscendC::TPipe& pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& aos,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& planar, AscendC::TBuf<AscendC::TPosition::VECCALC>& accum,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& old, AscendC::TBuf<AscendC::TPosition::VECCALC>& coefficient,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& offsets, AscendC::TBuf<AscendC::TPosition::VECCALC>& output,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& scratch, AscendC::TBuf<AscendC::TPosition::VECCALC>& threshold,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& mask)
{
    constexpr uint32_t matrix = N * N;
    pipe.InitBuffer(aos, N * TINY_VECTOR_ROWS * 2U * sizeof(float));
    pipe.InitBuffer(planar, matrix * 2U * sizeof(float));
    pipe.InitBuffer(accum, TINY_VECTOR_BATCH_ELEMENTS * 4U * sizeof(float));
    pipe.InitBuffer(old, TINY_VECTOR_MAX_N * 2U * sizeof(float));
    pipe.InitBuffer(coefficient, TINY_VECTOR_COEFFICIENT_PLANE * 2U * sizeof(float));
    pipe.InitBuffer(offsets, CHERK_ARCH22_META_TINY_INTERLEAVE_COUNT * sizeof(uint32_t));
    pipe.InitBuffer(output, TINY_VECTOR_BATCH_COLS * TINY_VECTOR_MAX_N * 2U * sizeof(float));
    pipe.InitBuffer(scratch, matrix * sizeof(float));
    pipe.InitBuffer(threshold, matrix * sizeof(float));
    pipe.InitBuffer(mask, matrix / 8U < 32U ? 32U : matrix / 8U);
}

template <uint32_t N>
__aicore__ inline bool LoadTinyVectorInput(TinyVectorContext<N>& ctx, const CherkArch22TilingData& tiling)
{
    constexpr uint32_t matrix = N * N;
    const uint32_t bytes = N * 2U * sizeof(float);
    const uint32_t aligned = (bytes + 31U) & ~31U;
    const AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(N), bytes, static_cast<uint32_t>((tiling.lda - N) * 2U * sizeof(float)), 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> pad{
        aligned != bytes, 0U, static_cast<uint8_t>((aligned - bytes) / sizeof(float)), 0.0f};
    AscendC::DataCopyPad(ctx.aos, ctx.a, copy, pad);
    AscendC::DataCopy(
        ctx.interleaveOffsets, ctx.metadata[CHERK_ARCH22_META_TINY_INTERLEAVE_OFFSET],
        CHERK_ARCH22_META_TINY_INTERLEAVE_COUNT);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.events.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.events.mte2ToV);
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = matrix * 2U / 64U;
    AscendC::GatherMask<float>(ctx.realA, ctx.aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(ctx.imagA, ctx.aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (N != 32U) {
        return false;
    }
    const bool realExtreme = CherkArch22TinyVectorPlaneIsExtreme<matrix>(
        ctx.realA, ctx.extremeScratch, ctx.extremeThreshold, ctx.extremeMask);
    return realExtreme || CherkArch22TinyVectorPlaneIsExtreme<matrix>(
                              ctx.imagA, ctx.extremeScratch, ctx.extremeThreshold, ctx.extremeMask);
}

template <uint32_t N>
__aicore__ inline void AccumulateTinyVectorLayer(
    TinyVectorContext<N>& ctx, uint32_t base, uint32_t colBase, uint32_t rowCount, uint8_t rowBlocks,
    const AscendC::BinaryRepeatParams& mul, const AscendC::BinaryRepeatParams& add)
{
    for (uint32_t rep = 0U; rep < rowBlocks; ++rep) {
        AscendC::Brcb(ctx.coefficientReal[rep * 8U], ctx.realA[base + colBase], 1U, {rowBlocks, 8U});
        AscendC::Brcb(ctx.coefficientImag[rep * 8U], ctx.imagA[base + colBase], 1U, {rowBlocks, 8U});
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(ctx.productReal, ctx.realA[base], ctx.coefficientReal, rowCount, TINY_VECTOR_BATCH_COLS, mul);
    AscendC::Mul(ctx.productImag, ctx.imagA[base], ctx.coefficientImag, rowCount, TINY_VECTOR_BATCH_COLS, mul);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(ctx.productReal, ctx.productReal, ctx.productImag, rowCount, TINY_VECTOR_BATCH_COLS, add);
    AscendC::Add(ctx.realOut, ctx.realOut, ctx.productReal, rowCount, TINY_VECTOR_BATCH_COLS, add);
    AscendC::Mul(ctx.productReal, ctx.imagA[base], ctx.coefficientReal, rowCount, TINY_VECTOR_BATCH_COLS, mul);
    AscendC::Mul(ctx.productImag, ctx.realA[base], ctx.coefficientImag, rowCount, TINY_VECTOR_BATCH_COLS, mul);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(ctx.productReal, ctx.productReal, ctx.productImag, rowCount, TINY_VECTOR_BATCH_COLS, add);
    AscendC::Add(ctx.imagOut, ctx.imagOut, ctx.productReal, rowCount, TINY_VECTOR_BATCH_COLS, add);
    AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t N>
__aicore__ inline void ComputeTinyVectorBatch(TinyVectorContext<N>& ctx, uint32_t colBase, float alpha)
{
    AscendC::Duplicate(ctx.realOut, 0.0f, TINY_VECTOR_BATCH_ELEMENTS);
    AscendC::Duplicate(ctx.imagOut, 0.0f, TINY_VECTOR_BATCH_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    if (alpha == 0.0f) {
        return;
    }
    constexpr uint8_t stride = TINY_VECTOR_ROWS / 8U;
    const uint32_t rowCount = colBase + TINY_VECTOR_BATCH_COLS;
    const uint8_t rowBlocks = static_cast<uint8_t>(rowCount / 8U);
    const AscendC::BinaryRepeatParams mul{1U, 1U, 0U, stride, 0U, rowBlocks};
    const AscendC::BinaryRepeatParams add{1U, 1U, 1U, stride, stride, stride};
    for (uint32_t layer = 0U; layer < N; ++layer) {
        AccumulateTinyVectorLayer(ctx, layer * N, colBase, rowCount, rowBlocks, mul, add);
    }
    if (alpha != 1.0f) {
        AscendC::Muls(ctx.realOut, ctx.realOut, alpha, TINY_VECTOR_BATCH_ELEMENTS);
        AscendC::Muls(ctx.imagOut, ctx.imagOut, alpha, TINY_VECTOR_BATCH_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <uint32_t N>
__aicore__ inline void StoreTinyVectorColumn(
    TinyVectorContext<N>& ctx, const CherkArch22TilingData& tiling, uint32_t colBase, uint32_t column, float beta)
{
    const uint32_t col = colBase + column;
    if (col >= N) {
        return;
    }
    const uint32_t count = col + 1U;
    AscendC::LocalTensor<float> real = ctx.realOut[column * TINY_VECTOR_ROWS];
    AscendC::LocalTensor<float> imag = ctx.imagOut[column * TINY_VECTOR_ROWS];
    if (beta != 0.0f) {
        const uint64_t cOffset = static_cast<uint64_t>(col) * tiling.ldc * 2ULL;
        LoadCherkOldColumn(
            ctx.aos, ctx.oldReal, ctx.oldImag, ctx.c, cOffset, count, N, beta, ctx.events.vToMte2, ctx.events.mte2ToV);
        AscendC::Add(real, real, ctx.oldReal, count);
        AscendC::Add(imag, imag, ctx.oldImag, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    uint64_t diagonalMask[2] = {1ULL << col, 0ULL};
    AscendC::Duplicate(imag, 0.0f, diagonalMask, 1U, 1U, 8U);
    AscendC::Adds(ctx.oldReal, real, 0.0f, N);
    AscendC::Adds(ctx.oldImag, imag, 0.0f, N);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::LocalTensor<float> output = ctx.outputStorage[column * TINY_VECTOR_MAX_N * 2U];
    AscendC::Gather(output, ctx.oldReal, ctx.interleaveOffsets, 0U, count * 2U);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.events.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.events.vToMte3);
    CopyUbToGm(ctx.c[static_cast<uint64_t>(col) * tiling.ldc * 2ULL], output, count * 2U);
}
} // namespace

template <uint32_t TINY_N>
__aicore__ inline void RunTinyVector(
    TinyVectorContext<TINY_N>& ctx, GM_ADDR aGm, GM_ADDR cGm, const CherkArch22TilingData& tiling, float alpha,
    float beta)
{
    if (LoadTinyVectorInput(ctx, tiling)) {
        CherkTinyReferenceFallback<TINY_N>(alpha, beta, aGm, cGm, tiling);
        return;
    }
    const uint32_t colBase = static_cast<uint32_t>(AscendC::GetBlockIdx()) * TINY_VECTOR_BATCH_COLS;
    if (colBase >= TINY_N) {
        return;
    }
    ComputeTinyVectorBatch(ctx, colBase, alpha);
    for (uint32_t column = 0U; column < TINY_VECTOR_BATCH_COLS; ++column) {
        StoreTinyVectorColumn(ctx, tiling, colBase, column, beta);
    }
}

template <uint32_t TINY_N>
__aicore__ inline void CherkArch22TinyVectorImpl(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData& tiling)
{
    constexpr uint32_t matrix = TINY_N * TINY_N;
    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    if (alpha == 0.0f && beta == 1.0f) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf, planarBuf, accumBuf, oldBuf, coefficientBuf, offsetBuf,
        outputBuf, scratchBuf, thresholdBuf, maskBuf;
    InitTinyVectorStorage<TINY_N>(
        pipe, aosBuf, planarBuf, accumBuf, oldBuf, coefficientBuf, offsetBuf, outputBuf, scratchBuf, thresholdBuf,
        maskBuf);
    AscendC::LocalTensor<float> planar = planarBuf.Get<float>();
    AscendC::LocalTensor<float> accum = accumBuf.Get<float>();
    AscendC::LocalTensor<float> old = oldBuf.Get<float>();
    AscendC::LocalTensor<float> coefficient = coefficientBuf.Get<float>();
    TinyVectorContext<TINY_N> ctx{
        aosBuf.Get<float>(),
        planar,
        planar[matrix],
        accum,
        accum[TINY_VECTOR_BATCH_ELEMENTS],
        accum[2U * TINY_VECTOR_BATCH_ELEMENTS],
        accum[3U * TINY_VECTOR_BATCH_ELEMENTS],
        old,
        old[TINY_VECTOR_MAX_N],
        coefficient,
        coefficient[TINY_VECTOR_COEFFICIENT_PLANE],
        offsetBuf.Get<uint32_t>(),
        outputBuf.Get<float>(),
        scratchBuf.Get<float>(),
        thresholdBuf.Get<float>(),
        maskBuf.Get<uint8_t>(),
        {},
        {},
        {},
        GetCherkCopyEvents()};
    InitCherkTinyGlobals(ctx.a, ctx.c, ctx.metadata, aGm, cGm, metadataGm, tiling, TINY_N);
    RunTinyVector(ctx, aGm, cGm, tiling, alpha, beta);
}

#define CHERK_TINY_VECTOR_KERNEL(N)                                                     \
    extern "C" __global__ __aicore__ void cherk_arch22_tiny_vector_##N##_kernel(        \
        GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR metadataGm,  \
        const CherkArch22TilingData tiling)                                             \
    {                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                 \
        CherkArch22TinyVectorImpl<N##U>(alphaGm, aGm, betaGm, cGm, metadataGm, tiling); \
    }

CHERK_TINY_VECTOR_KERNEL(8)
CHERK_TINY_VECTOR_KERNEL(16)
CHERK_TINY_VECTOR_KERNEL(32)
#undef CHERK_TINY_VECTOR_KERNEL
/** Vector-reduction path for the UPPER/OP_N square tiny family. */
namespace {
template <uint32_t N>
struct TinyTransposeContext {
    AscendC::LocalTensor<float> aos;
    AscendC::LocalTensor<float> realA;
    AscendC::LocalTensor<float> imagA;
    AscendC::LocalTensor<float> realT;
    AscendC::LocalTensor<float> imagT;
    AscendC::LocalTensor<float> rr;
    AscendC::LocalTensor<float> ii;
    AscendC::LocalTensor<float> ri;
    AscendC::LocalTensor<float> ir;
    AscendC::LocalTensor<float> realOut;
    AscendC::LocalTensor<float> imagOut;
    AscendC::LocalTensor<float> oldReal;
    AscendC::LocalTensor<float> oldImag;
    AscendC::LocalTensor<float> output;
    AscendC::LocalTensor<uint32_t> transposeOffsets;
    AscendC::LocalTensor<uint32_t> interleaveOffsets;
    AscendC::LocalTensor<float> extremeScratch;
    AscendC::LocalTensor<float> extremeThreshold;
    AscendC::LocalTensor<uint8_t> extremeMask;
    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> c;
    AscendC::GlobalTensor<uint32_t> metadata;
    CherkCopyEvents events;
};

template <uint32_t N, uint32_t TRANSPOSE_COUNT, uint32_t INTERLEAVE_COUNT>
__aicore__ inline void InitTinyTransposeStorage(
    AscendC::TPipe& pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& aos,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& planar, AscendC::TBuf<AscendC::TPosition::VECCALC>& transpose,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& product, AscendC::TBuf<AscendC::TPosition::VECCALC>& accum,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& old, AscendC::TBuf<AscendC::TPosition::VECCALC>& output,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& metadata, AscendC::TBuf<AscendC::TPosition::VECCALC>& scratch,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& threshold, AscendC::TBuf<AscendC::TPosition::VECCALC>& mask)
{
    constexpr uint32_t matrix = N * N;
    constexpr uint32_t outputPlane = N <= 32U ? 32U : N;
    pipe.InitBuffer(aos, matrix * 2U * sizeof(float));
    pipe.InitBuffer(planar, matrix * 2U * sizeof(float));
    pipe.InitBuffer(transpose, matrix * 2U * sizeof(float));
    pipe.InitBuffer(product, matrix * 4U * sizeof(float));
    pipe.InitBuffer(accum, outputPlane * 2U * sizeof(float));
    pipe.InitBuffer(old, N * 2U * sizeof(float));
    pipe.InitBuffer(output, matrix * 2U * sizeof(float));
    pipe.InitBuffer(metadata, (TRANSPOSE_COUNT + INTERLEAVE_COUNT) * sizeof(uint32_t));
    pipe.InitBuffer(scratch, matrix * sizeof(float));
    pipe.InitBuffer(threshold, matrix * sizeof(float));
    pipe.InitBuffer(mask, matrix / 8U < 32U ? 32U : matrix / 8U);
}

template <
    uint32_t N, uint32_t TRANSPOSE_OFFSET, uint32_t TRANSPOSE_COUNT, uint32_t INTERLEAVE_OFFSET,
    uint32_t INTERLEAVE_COUNT>
__aicore__ inline bool LoadTinyTransposeInput(TinyTransposeContext<N>& ctx, const CherkArch22TilingData& tiling)
{
    constexpr uint32_t matrix = N * N;
    const AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(N), N * 2U * sizeof(float), static_cast<uint32_t>((tiling.lda - N) * 2U * sizeof(float)),
        0U, 0U};
    AscendC::DataCopyPad(ctx.aos, ctx.a, copy, {false, 0U, 0U, 0.0f});
    AscendC::DataCopy(ctx.transposeOffsets, ctx.metadata[TRANSPOSE_OFFSET], TRANSPOSE_COUNT);
    AscendC::DataCopy(ctx.interleaveOffsets, ctx.metadata[INTERLEAVE_OFFSET], INTERLEAVE_COUNT);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.events.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.events.mte2ToV);
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = matrix * 2U / 64U;
    AscendC::GatherMask<float>(ctx.realA, ctx.aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(ctx.imagA, ctx.aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (N > 32U) {
        return false;
    }
    const bool realExtreme = CherkArch22TinyVectorPlaneIsExtreme<matrix>(
        ctx.realA, ctx.extremeScratch, ctx.extremeThreshold, ctx.extremeMask);
    if (realExtreme) {
        return true;
    }
    return CherkArch22TinyVectorPlaneIsExtreme<matrix>(
        ctx.imagA, ctx.extremeScratch, ctx.extremeThreshold, ctx.extremeMask);
}

template <uint32_t N>
__aicore__ inline void ComputeTinyTransposeColumn(TinyTransposeContext<N>& ctx, uint32_t col, float alpha)
{
    constexpr uint32_t matrix = N * N;
    constexpr uint8_t repeats = static_cast<uint8_t>(N);
    constexpr uint8_t stride = static_cast<uint8_t>(N / 8U);
    const AscendC::BinaryRepeatParams params{1U, 1U, 1U, stride, stride, 0U};
    const AscendC::LocalTensor<float> realCoeff = ctx.realT[col * N];
    const AscendC::LocalTensor<float> imagCoeff = ctx.imagT[col * N];
    AscendC::Mul(ctx.rr, ctx.realT, realCoeff, N, repeats, params);
    AscendC::Mul(ctx.ii, ctx.imagT, imagCoeff, N, repeats, params);
    AscendC::Mul(ctx.ri, ctx.imagT, realCoeff, N, repeats, params);
    AscendC::Mul(ctx.ir, ctx.realT, imagCoeff, N, repeats, params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(ctx.rr, ctx.rr, ctx.ii, matrix);
    AscendC::Sub(ctx.ri, ctx.ri, ctx.ir, matrix);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(ctx.realOut, ctx.rr, N, repeats, 1U, 1U, stride);
    AscendC::WholeReduceSum(ctx.imagOut, ctx.ri, N, repeats, 1U, 1U, stride);
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t count = col + 1U;
    if (alpha != 1.0f) {
        AscendC::Muls(ctx.realOut, ctx.realOut, alpha, count);
        AscendC::Muls(ctx.imagOut, ctx.imagOut, alpha, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <uint32_t N>
__aicore__ inline void StoreTinyTransposeColumn(
    TinyTransposeContext<N>& ctx, const CherkArch22TilingData& tiling, uint32_t col, float beta)
{
    const uint32_t count = col + 1U;
    if (beta != 0.0f) {
        const uint64_t cOffset = static_cast<uint64_t>(col) * tiling.ldc * 2ULL;
        LoadCherkOldColumn(
            ctx.aos, ctx.oldReal, ctx.oldImag, ctx.c, cOffset, count, N, beta, ctx.events.vToMte2, ctx.events.mte2ToV);
        AscendC::Add(ctx.realOut, ctx.realOut, ctx.oldReal, count);
        AscendC::Add(ctx.imagOut, ctx.imagOut, ctx.oldImag, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    uint64_t diagonalMask[2] = {1ULL << col, 0ULL};
    AscendC::Duplicate(ctx.imagOut, 0.0f, diagonalMask, 1U, 1U, 8U);
    AscendC::LocalTensor<float> output = ctx.output[col * N * 2U];
    AscendC::Gather(output, ctx.realOut, ctx.interleaveOffsets, 0U, count * 2U);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.events.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.events.vToMte3);
    CopyUbToGm(ctx.c[static_cast<uint64_t>(col) * tiling.ldc * 2ULL], output, count * 2U);
}

template <uint32_t N>
__aicore__ inline void RunTinyTransposeColumns(
    TinyTransposeContext<N>& ctx, const CherkArch22TilingData& tiling, float alpha, float beta)
{
    AscendC::Gather(ctx.realT, ctx.realA, ctx.transposeOffsets, 0U, N * N);
    AscendC::Gather(ctx.imagT, ctx.imagA, ctx.transposeOffsets, 0U, N * N);
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t blocks = CherkNormalizeBlockCount(static_cast<uint32_t>(AscendC::GetBlockNum()));
    for (uint32_t col = static_cast<uint32_t>(AscendC::GetBlockIdx()); col < N; col += blocks) {
        ComputeTinyTransposeColumn(ctx, col, alpha);
        StoreTinyTransposeColumn(ctx, tiling, col, beta);
    }
}
} // namespace

template <uint32_t TINY_N>
__aicore__ inline bool TryHandleTinyTransposeScale(
    float alpha, float beta, GM_ADDR betaGm, GM_ADDR cGm, const CherkArch22TilingData& tiling)
{
    if (alpha != 0.0f) {
        return false;
    }
    if (beta != 1.0f) {
        const uint32_t blocks = CherkNormalizeBlockCount(static_cast<uint32_t>(AscendC::GetBlockNum()));
        for (uint32_t col = static_cast<uint32_t>(AscendC::GetBlockIdx()); col < TINY_N; col += blocks) {
            CherkScaleUpperTriangle(betaGm, cGm, tiling, col, 1U);
        }
    }
    return true;
}

template <
    uint32_t TINY_N, uint32_t TRANSPOSE_OFFSET, uint32_t TRANSPOSE_COUNT, uint32_t INTERLEAVE_OFFSET,
    uint32_t INTERLEAVE_COUNT>
__aicore__ inline void RunTinyTranspose(
    TinyTransposeContext<TINY_N>& ctx, GM_ADDR aGm, GM_ADDR cGm, const CherkArch22TilingData& tiling, float alpha,
    float beta)
{
    if (LoadTinyTransposeInput<TINY_N, TRANSPOSE_OFFSET, TRANSPOSE_COUNT, INTERLEAVE_OFFSET, INTERLEAVE_COUNT>(
            ctx, tiling)) {
        CherkTinyReferenceFallback<TINY_N>(alpha, beta, aGm, cGm, tiling);
        return;
    }
    RunTinyTransposeColumns(ctx, tiling, alpha, beta);
}

template <
    uint32_t TINY_N, uint32_t TRANSPOSE_OFFSET, uint32_t TRANSPOSE_COUNT, uint32_t INTERLEAVE_OFFSET,
    uint32_t INTERLEAVE_COUNT>
__aicore__ inline void CherkArch22TinyTransposeImpl(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    constexpr uint32_t matrix = TINY_N * TINY_N;
    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    if (TryHandleTinyTransposeScale<TINY_N>(alpha, beta, betaGm, cGm, tiling)) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aosBuf, planarBuf, transposeBuf, productBuf, accumBuf, oldBuf, outputBuf,
        metadataBuf, scratchBuf, thresholdBuf, maskBuf;
    InitTinyTransposeStorage<TINY_N, TRANSPOSE_COUNT, INTERLEAVE_COUNT>(
        pipe, aosBuf, planarBuf, transposeBuf, productBuf, accumBuf, oldBuf, outputBuf, metadataBuf, scratchBuf,
        thresholdBuf, maskBuf);
    AscendC::LocalTensor<float> planar = planarBuf.Get<float>();
    AscendC::LocalTensor<float> transposed = transposeBuf.Get<float>();
    AscendC::LocalTensor<float> product = productBuf.Get<float>();
    AscendC::LocalTensor<float> accum = accumBuf.Get<float>();
    AscendC::LocalTensor<float> old = oldBuf.Get<float>();
    AscendC::LocalTensor<uint32_t> offsets = metadataBuf.Get<uint32_t>();
    TinyTransposeContext<TINY_N> ctx{
        aosBuf.Get<float>(),
        planar,
        planar[matrix],
        transposed,
        transposed[matrix],
        product,
        product[matrix],
        product[2U * matrix],
        product[3U * matrix],
        accum,
        accum[TINY_N <= 32U ? 32U : TINY_N],
        old,
        old[TINY_N],
        outputBuf.Get<float>(),
        offsets,
        offsets[TRANSPOSE_COUNT],
        scratchBuf.Get<float>(),
        thresholdBuf.Get<float>(),
        maskBuf.Get<uint8_t>(),
        {},
        {},
        {},
        GetCherkCopyEvents()};
    InitCherkTinyGlobals(ctx.a, ctx.c, ctx.metadata, aGm, cGm, metadataGm, tiling, TINY_N);
    RunTinyTranspose<TINY_N, TRANSPOSE_OFFSET, TRANSPOSE_COUNT, INTERLEAVE_OFFSET, INTERLEAVE_COUNT>(
        ctx, aGm, cGm, tiling, alpha, beta);
}

extern "C" __global__ __aicore__ void cherk_arch22_tiny8_transpose_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CherkArch22TinyTransposeImpl<
        8U, CHERK_ARCH22_META_TINY8_TRANSPOSE_OFFSET, CHERK_ARCH22_META_TINY8_TRANSPOSE_COUNT,
        CHERK_ARCH22_META_TINY_INTERLEAVE_OFFSET, CHERK_ARCH22_META_TINY_INTERLEAVE_COUNT>(
        alphaGm, aGm, betaGm, cGm, metadataGm, tiling);
}

extern "C" __global__ __aicore__ void cherk_arch22_tiny16_transpose_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CherkArch22TinyTransposeImpl<
        16U, CHERK_ARCH22_META_TINY16_TRANSPOSE_OFFSET, CHERK_ARCH22_META_TINY16_TRANSPOSE_COUNT,
        CHERK_ARCH22_META_TINY_INTERLEAVE_OFFSET, CHERK_ARCH22_META_TINY_INTERLEAVE_COUNT>(
        alphaGm, aGm, betaGm, cGm, metadataGm, tiling);
}

extern "C" __global__ __aicore__ void cherk_arch22_tiny32_transpose_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CherkArch22TinyTransposeImpl<
        32U, CHERK_ARCH22_META_TINY_TRANSPOSE_OFFSET, CHERK_ARCH22_META_TINY_TRANSPOSE_COUNT,
        CHERK_ARCH22_META_TINY_INTERLEAVE_OFFSET, CHERK_ARCH22_META_TINY_INTERLEAVE_COUNT>(
        alphaGm, aGm, betaGm, cGm, metadataGm, tiling);
}
/** Workspace-free vector/reduction path for UPPER/OP_N/n=k=64. */
extern "C" __global__ __aicore__ void cherk_arch22_tiny64_transpose_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CherkArch22TinyTransposeImpl<
        64U, CHERK_ARCH22_META_TRANSPOSE_OFFSET, CHERK_ARCH22_META_TRANSPOSE_COUNT,
        CHERK_ARCH22_META_TINY64_INTERLEAVE_OFFSET, CHERK_ARCH22_META_TINY64_INTERLEAVE_COUNT>(
        alphaGm, aGm, betaGm, cGm, metadataGm, tiling);
}
/** Fixed-rank OP_N path for full panels within the UB row capacity. */
namespace {
constexpr uint32_t OUTER_K2_ROWS = CHERK_ARCH22_OUTER_ROWS;
constexpr uint32_t OUTER_K2_RANK = CHERK_ARCH22_OUTER_RANK;
constexpr uint32_t OUTER_K2_PANEL_COLS = CHERK_ARCH22_OUTER_PANEL_COLS;
constexpr uint32_t OUTER_K2_PLANE = OUTER_K2_ROWS * OUTER_K2_PANEL_COLS;
constexpr uint32_t OUTER_K2_OFFSET_COUNT = OUTER_K2_PLANE * 2U;

struct OuterK2Context {
    AscendC::LocalTensor<float> input;
    AscendC::LocalTensor<float> realA;
    AscendC::LocalTensor<float> imagA;
    AscendC::LocalTensor<float> outReal;
    AscendC::LocalTensor<float> outImag;
    AscendC::LocalTensor<float> prodReal;
    AscendC::LocalTensor<float> prodImag;
    AscendC::LocalTensor<float> oldReal;
    AscendC::LocalTensor<float> oldImag;
    AscendC::LocalTensor<uint32_t> offsets;
    AscendC::LocalTensor<float> output;
    AscendC::GlobalTensor<float> a;
    AscendC::GlobalTensor<float> c;
    AscendC::GlobalTensor<uint32_t> metadata;
    CherkCopyEvents events;
};

__aicore__ inline void LoadOuterK2Inputs(OuterK2Context& ctx, const CherkArch22TilingData& tiling, uint32_t firstCol)
{
    const uint32_t inputBytes = tiling.n * 2U * sizeof(float);
    AscendC::Duplicate(ctx.input, 0.0f, OUTER_K2_RANK * OUTER_K2_ROWS * 2U);
    const AscendC::DataCopyExtParams inputCopy{
        OUTER_K2_RANK, inputBytes, static_cast<uint32_t>((tiling.lda - tiling.n) * 2U * sizeof(float)),
        static_cast<uint32_t>((OUTER_K2_ROWS * 2U * sizeof(float) - inputBytes) / 32U), 0U};
    AscendC::DataCopyPad(ctx.input, ctx.a, inputCopy, {false, 0U, 0U, 0.0f});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.events.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.events.mte2ToV);
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = OUTER_K2_RANK * OUTER_K2_ROWS * 2U / 64U;
    AscendC::GatherMask<float>(ctx.realA, ctx.input, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask<float>(ctx.imagA, ctx.input, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(ctx.offsets, ctx.metadata[CHERK_ARCH22_META_OUTER_K2_INTERLEAVE_OFFSET], OUTER_K2_OFFSET_COUNT);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.events.vToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.events.vToMte2);
    const uint64_t cOffset = static_cast<uint64_t>(firstCol) * tiling.ldc * 2ULL;
    const AscendC::DataCopyExtParams cCopy{
        OUTER_K2_PANEL_COLS, inputBytes, static_cast<uint32_t>((tiling.ldc - tiling.n) * 2U * sizeof(float)),
        static_cast<uint32_t>((OUTER_K2_ROWS * 2U * sizeof(float) - inputBytes) / 32U), 0U};
    AscendC::DataCopyPad(ctx.input, ctx.c[cOffset], cCopy, {false, 0U, 0U, 0.0f});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.events.mte2ToV);
}

__aicore__ inline void ComputeOuterK2(OuterK2Context& ctx, uint32_t firstCol)
{
    AscendC::Duplicate(ctx.outReal, 0.0f, OUTER_K2_PLANE);
    AscendC::Duplicate(ctx.outImag, 0.0f, OUTER_K2_PLANE);
    AscendC::PipeBarrier<PIPE_V>();
    constexpr uint8_t repeats = OUTER_K2_PANEL_COLS;
    constexpr uint8_t repeatStride = OUTER_K2_ROWS / 8U;
    const AscendC::BinaryRepeatParams mulParams{1U, 1U, 0U, repeatStride, 0U, 1U};
    const AscendC::BinaryRepeatParams addParams{1U, 1U, 1U, repeatStride, repeatStride, repeatStride};
    for (uint32_t layer = 0U; layer < OUTER_K2_RANK; ++layer) {
        const uint32_t src = layer * OUTER_K2_ROWS;
        AscendC::Brcb(ctx.oldReal, ctx.realA[src + firstCol], 1U, {1U, 8U});
        AscendC::Brcb(ctx.oldImag, ctx.imagA[src + firstCol], 1U, {1U, 8U});
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t rowBase = 0U; rowBase < OUTER_K2_ROWS; rowBase += 64U) {
            const uint32_t rowSrc = src + rowBase;
            AscendC::Mul(ctx.prodReal[rowBase], ctx.realA[rowSrc], ctx.oldReal, 64U, repeats, mulParams);
            AscendC::Mul(ctx.prodImag[rowBase], ctx.imagA[rowSrc], ctx.oldImag, 64U, repeats, mulParams);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ctx.prodReal[rowBase], ctx.prodReal[rowBase], ctx.prodImag[rowBase], 64U, repeats, addParams);
            AscendC::Add(ctx.outReal[rowBase], ctx.outReal[rowBase], ctx.prodReal[rowBase], 64U, repeats, addParams);
            AscendC::Mul(ctx.prodReal[rowBase], ctx.imagA[rowSrc], ctx.oldReal, 64U, repeats, mulParams);
            AscendC::Mul(ctx.prodImag[rowBase], ctx.realA[rowSrc], ctx.oldImag, 64U, repeats, mulParams);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(ctx.prodReal[rowBase], ctx.prodReal[rowBase], ctx.prodImag[rowBase], 64U, repeats, addParams);
            AscendC::Add(ctx.outImag[rowBase], ctx.outImag[rowBase], ctx.prodReal[rowBase], 64U, repeats, addParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

__aicore__ inline void ApplyOuterK2Scalars(OuterK2Context& ctx, float alpha, float beta)
{
    if (alpha != 1.0f) {
        AscendC::Muls(ctx.outReal, ctx.outReal, alpha, OUTER_K2_PLANE);
        AscendC::Muls(ctx.outImag, ctx.outImag, alpha, OUTER_K2_PLANE);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.events.mte2ToV);
    if (beta == 0.0f) {
        return;
    }
    uint64_t reserved = 0ULL;
    constexpr uint16_t repeats = OUTER_K2_PANEL_COLS * OUTER_K2_ROWS * 2U / 64U;
    AscendC::GatherMask(ctx.oldReal, ctx.input, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::GatherMask(ctx.oldImag, ctx.input, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Axpy(ctx.outReal, ctx.oldReal, beta, OUTER_K2_PLANE);
    AscendC::Axpy(ctx.outImag, ctx.oldImag, beta, OUTER_K2_PLANE);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void StoreOuterK2(OuterK2Context& ctx, const CherkArch22TilingData& tiling, uint32_t firstCol)
{
    for (uint32_t col = 0U; col < OUTER_K2_PANEL_COLS; ++col) {
        const uint32_t diagonal = firstCol + col;
        const uint32_t aligned = diagonal & ~63U;
        uint64_t diagonalMask[2] = {1ULL << (diagonal & 63U), 0ULL};
        AscendC::Duplicate(ctx.outImag[col * OUTER_K2_ROWS + aligned], 0.0f, diagonalMask, 1U, 1U, 8U);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(ctx.output, ctx.outReal, ctx.offsets, 0U, OUTER_K2_OFFSET_COUNT);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t col = 0U; col < OUTER_K2_PANEL_COLS; ++col) {
        const uint32_t selectedFloats = (firstCol + col + 1U) * 2U;
        AscendC::Adds(ctx.input[col * OUTER_K2_ROWS * 2U], ctx.output[col * OUTER_K2_ROWS * 2U], 0.0f, selectedFloats);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.events.vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.events.vToMte3);
    const uint32_t rowBytes = tiling.n * 2U * sizeof(float);
    const uint64_t outputOffset = static_cast<uint64_t>(firstCol) * tiling.ldc * 2ULL;
    const AscendC::DataCopyExtParams outputCopy{
        OUTER_K2_PANEL_COLS, rowBytes, static_cast<uint32_t>((OUTER_K2_ROWS * 2U * sizeof(float) - rowBytes) / 32U),
        static_cast<uint32_t>((tiling.ldc - tiling.n) * 2U * sizeof(float)), 0U};
    AscendC::DataCopyPad(ctx.c[outputOffset], ctx.input, outputCopy);
}
} // namespace

__aicore__ inline void RunOuterK2(
    OuterK2Context& ctx, GM_ADDR aGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData& tiling,
    uint32_t firstCol, float alpha, float beta)
{
    ctx.a.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(aGm), static_cast<uint64_t>(tiling.lda) * OUTER_K2_RANK * 2ULL);
    ctx.c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cGm), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2ULL);
    ctx.metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadataGm), CHERK_ARCH22_META_WORDS);
    LoadOuterK2Inputs(ctx, tiling, firstCol);
    ComputeOuterK2(ctx, firstCol);
    ApplyOuterK2Scalars(ctx, alpha, beta);
    StoreOuterK2(ctx, tiling, firstCol);
}

extern "C" __global__ __aicore__ void cherk_arch22_outer_k2_kernel(
    GM_ADDR alphaGm, GM_ADDR aGm, GM_ADDR betaGm, GM_ADDR cGm, GM_ADDR metadataGm, const CherkArch22TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t firstCol = static_cast<uint32_t>(AscendC::GetBlockIdx()) * OUTER_K2_PANEL_COLS;
    if (!CherkArch22OuterGeometry(tiling.n, tiling.k) || firstCol >= tiling.n) {
        return;
    }
    const float alpha = *reinterpret_cast<__gm__ const float*>(alphaGm);
    const float beta = *reinterpret_cast<__gm__ const float*>(betaGm);
    if (alpha == 0.0f && beta == 1.0f) {
        return;
    }
    if (alpha == 0.0f) {
        CherkScaleUpperTriangle(betaGm, cGm, tiling, firstCol, OUTER_K2_PANEL_COLS);
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuf, planarBuf, accumBuf, ioBuf;
    pipe.InitBuffer(inputBuf, 2U * OUTER_K2_PLANE * sizeof(float));
    pipe.InitBuffer(planarBuf, OUTER_K2_RANK * OUTER_K2_ROWS * 2U * sizeof(float));
    pipe.InitBuffer(accumBuf, 6U * OUTER_K2_PLANE * sizeof(float));
    pipe.InitBuffer(ioBuf, 2U * OUTER_K2_OFFSET_COUNT * sizeof(float));
    AscendC::LocalTensor<float> planar = planarBuf.Get<float>();
    AscendC::LocalTensor<float> accum = accumBuf.Get<float>();
    AscendC::LocalTensor<float> io = ioBuf.Get<float>();
    OuterK2Context ctx{
        inputBuf.Get<float>(),
        planar,
        planar[OUTER_K2_RANK * OUTER_K2_ROWS],
        accum,
        accum[OUTER_K2_PLANE],
        accum[2U * OUTER_K2_PLANE],
        accum[3U * OUTER_K2_PLANE],
        accum[4U * OUTER_K2_PLANE],
        accum[5U * OUTER_K2_PLANE],
        ioBuf.Get<uint32_t>(),
        io[OUTER_K2_OFFSET_COUNT],
        {},
        {},
        {},
        GetCherkCopyEvents()};
    RunOuterK2(ctx, aGm, cGm, metadataGm, tiling, firstCol, alpha, beta);
}
/** OP_N preprocess: source columns are already the required Kp x Np rows. */
void cherk_arch22_small_vector_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream)
{
    if (tiling.smallPath == 1U) {
        cherk_arch22_small_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(alpha, a, beta, c, tiling);
    } else if (tiling.smallPath == 2U) {
        if (tiling.n == 8U) {
            cherk_arch22_tiny_vector_8_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
                alpha, a, beta, c, metadata, tiling);
        } else if (tiling.n == 16U) {
            cherk_arch22_tiny_vector_16_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
                alpha, a, beta, c, metadata, tiling);
        } else {
            cherk_arch22_tiny_vector_32_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
                alpha, a, beta, c, metadata, tiling);
        }
    } else if (tiling.smallPath == 4U) {
        cherk_arch22_tiny64_transpose_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
            alpha, a, beta, c, metadata, tiling);
    } else if (tiling.smallPath == 6U) {
        cherk_arch22_tiny1_direct_kernel<<<1U, nullptr, stream>>>(alpha, a, beta, c, tiling);
    } else if (tiling.smallPath == 7U) {
        if (tiling.n == 16U) {
            cherk_arch22_tiny16_transpose_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
                alpha, a, beta, c, metadata, tiling);
        } else {
            cherk_arch22_tiny32_transpose_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(
                alpha, a, beta, c, metadata, tiling);
        }
    }
}

void cherk_arch22_outer_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR metadata, const CherkArch22TilingData& tiling,
    void* stream)
{
    cherk_arch22_outer_k2_kernel<<<tiling.aivCoreNum, nullptr, stream>>>(alpha, a, beta, c, metadata, tiling);
}

namespace {
void LaunchSmallPath(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream)
{
    if (tiling.smallPath == 3U) {
        cherk_arch22_square64_do(alpha, a, beta, c, workspace, metadata, tiling, stream);
    } else if (tiling.smallPath == 5U) {
        cherk_arch22_square128_do(alpha, a, beta, c, workspace, metadata, tiling, stream);
    } else {
        cherk_arch22_small_vector_do(alpha, a, beta, c, workspace, metadata, tiling, stream);
    }
}

void LaunchGeneralPath(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream)
{
    if (tiling.skipProduct == 0U) {
        cherk_arch22_pack_do(a, workspace, tiling, stream);
    }
    cherk_arch22_compute_do(alpha, beta, c, workspace, metadata, tiling, stream);
}
inline void LaunchDirectPath(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream)
{
    if (tiling.directMixMode == 1U || tiling.directMixMode == 2U) {
        cherk_arch22_direct_mix_do(alpha, a, beta, c, workspace, metadata, tiling, stream);
        return;
    }
    cherk_arch22_direct_pack_do(a, workspace, metadata, tiling, stream);
    if (tiling.directMixMode == 3U) {
        cherk_arch22_direct_mix_do(alpha, a, beta, c, workspace, metadata, tiling, stream);
        return;
    }
    cherk_arch22_direct_compute_do(alpha, beta, c, workspace, metadata, tiling, stream);
}
} // namespace

/** Stream-ordered host-visible launch graph. */
void cherk_arch22_kernel_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, GM_ADDR workspace, GM_ADDR metadata,
    const CherkArch22TilingData& tiling, void* stream)
{
    if (tiling.outerPath != 0U) {
        cherk_arch22_outer_do(alpha, a, beta, c, metadata, tiling, stream);
        return;
    }
    if (tiling.smallPath != 0U) {
        LaunchSmallPath(alpha, a, beta, c, workspace, metadata, tiling, stream);
        return;
    }
    if (tiling.directPath != 0U) {
        LaunchDirectPath(alpha, a, beta, c, workspace, metadata, tiling, stream);
        return;
    }
    LaunchGeneralPath(alpha, a, beta, c, workspace, metadata, tiling, stream);
}
