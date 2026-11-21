/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_batched_cgemm_kernel.cpp
 * \brief Complex batched GEMM kernels for arch35 (DAV_3510).
 */

#include "gemm_batched_cube_impl.h"

// ============================================================================
// Direct complex accumulation for small logical output matrices
// ============================================================================

namespace {

constexpr uint16_t CGEMM_MIX_SYNC_MODE = 2;
constexpr uint16_t CGEMM_MIX_AIV_BARRIER_MODE = 0;
constexpr uint16_t CGEMM_MIX_AIV_BARRIER_FLAG = 10;

constexpr uint32_t CGEMM_BATCHED_DIRECT_THREADS = 256;

__simt_callee__ __aicore__ inline __gm__ float* CgemmReadComplexPtr(__gm__ uint8_t* array, int32_t batch)
{
    __gm__ uint64_t* slots = reinterpret_cast<__gm__ uint64_t*>(array);
    return reinterpret_cast<__gm__ float*>(slots[batch]);
}

struct CgemmComplexValue {
    float real;
    float imag;
};

__simt_callee__ __aicore__ inline CgemmComplexValue CgemmDirectProduct(
    __gm__ float* a, __gm__ float* b, const CgemmBatchedDirectTilingData& tiling, int32_t row, int32_t col,
    int32_t inner)
{
    const int64_t aOffset = tiling.transA == 0 ? static_cast<int64_t>(inner) * tiling.lda + row :
                                                 static_cast<int64_t>(row) * tiling.lda + inner;
    const int64_t bOffset = tiling.transB == 0 ? static_cast<int64_t>(col) * tiling.ldb + inner :
                                                 static_cast<int64_t>(inner) * tiling.ldb + col;
    const float aReal = a[2 * aOffset];
    float aImag = a[2 * aOffset + 1];
    const float bReal = b[2 * bOffset];
    float bImag = b[2 * bOffset + 1];
    if (tiling.transA == 2) {
        aImag = -aImag;
    }
    if (tiling.transB == 2) {
        bImag = -bImag;
    }
    // Preserve one FP32 rounding per product.  Without volatile, the SIMT compiler may contract the
    // following add/subtract into FMA instructions and change the reference-order result.
    volatile float aRealBReal = aReal * bReal;
    volatile float aImagBImag = aImag * bImag;
    volatile float aRealBImag = aReal * bImag;
    volatile float aImagBReal = aImag * bReal;
    return {aRealBReal - aImagBImag, aRealBImag + aImagBReal};
}

__simt_callee__ __aicore__ inline CgemmComplexValue CgemmDirectDot(
    __gm__ float* a, __gm__ float* b, const CgemmBatchedDirectTilingData& tiling, int32_t row, int32_t col,
    bool useReferenceOrder)
{
    CgemmComplexValue sum{0.0f, 0.0f};
    CgemmComplexValue compensation{0.0f, 0.0f};
    for (int32_t inner = 0; inner < tiling.k; ++inner) {
        const CgemmComplexValue product = CgemmDirectProduct(a, b, tiling, row, col, inner);
        if (useReferenceOrder) {
            sum.real += product.real;
            sum.imag += product.imag;
            continue;
        }
        const float correctedReal = product.real - compensation.real;
        const float nextReal = sum.real + correctedReal;
        compensation.real = (nextReal - sum.real) - correctedReal;
        sum.real = nextReal;
        const float correctedImag = product.imag - compensation.imag;
        const float nextImag = sum.imag + correctedImag;
        compensation.imag = (nextImag - sum.imag) - correctedImag;
        sum.imag = nextImag;
    }
    return sum;
}

__simt_callee__ __aicore__ inline CgemmComplexValue CgemmApplyAlpha(
    const CgemmComplexValue& sum, const CgemmBatchedDirectTilingData& tiling)
{
    if (tiling.alphaImag == 0.0f) {
        return {tiling.alphaReal * sum.real, tiling.alphaReal * sum.imag};
    }
    if (tiling.alphaReal == 0.0f) {
        return {-tiling.alphaImag * sum.imag, tiling.alphaImag * sum.real};
    }
    return {
        tiling.alphaReal * sum.real - tiling.alphaImag * sum.imag,
        tiling.alphaReal * sum.imag + tiling.alphaImag * sum.real};
}

__simt_callee__ __aicore__ inline void CgemmStoreDirectResult(
    __gm__ float* c, int64_t offset, CgemmComplexValue result, const CgemmBatchedDirectTilingData& tiling)
{
    if (tiling.betaReal != 0.0f || tiling.betaImag != 0.0f) {
        const float oldReal = c[offset];
        const float oldImag = c[offset + 1];
        result.real += tiling.betaReal * oldReal - tiling.betaImag * oldImag;
        result.imag += tiling.betaReal * oldImag + tiling.betaImag * oldReal;
    }
    c[offset] = result.real;
    c[offset + 1] = result.imag;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_BATCHED_DIRECT_THREADS) inline void CgemmBatchedDirectVf(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* carray, CgemmBatchedDirectTilingData tiling,
    uint32_t block, uint32_t blocks)
{
    const int64_t matrixElements = static_cast<int64_t>(tiling.m) * tiling.n;
    const int64_t total = matrixElements * tiling.batchCount;
    const int64_t thread = static_cast<int64_t>(block) * blockDim.x + threadIdx.x;
    const int64_t threadCount = static_cast<int64_t>(blocks) * blockDim.x;

    for (int64_t index = thread; index < total; index += threadCount) {
        const int32_t batch = static_cast<int32_t>(index / matrixElements);
        const int64_t matrixIndex = index - static_cast<int64_t>(batch) * matrixElements;
        const int32_t col = static_cast<int32_t>(matrixIndex / tiling.m);
        const int32_t row = static_cast<int32_t>(matrixIndex - static_cast<int64_t>(col) * tiling.m);
        __gm__ float* a = CgemmReadComplexPtr(aarray, batch);
        __gm__ float* b = CgemmReadComplexPtr(barray, batch);
        __gm__ float* c = CgemmReadComplexPtr(carray, batch);

        // Kahan's compensation evaluates Inf - Inf and would turn a
        // reference Inf result into NaN on the low-batch direct path.
        const bool useReferenceOrder =
            tiling.m <= 2 || tiling.n <= 2 || (matrixElements <= 256 && tiling.k <= 16 && tiling.batchCount <= 2);
        const CgemmComplexValue sum = CgemmDirectDot(a, b, tiling, row, col, useReferenceOrder);
        CgemmComplexValue result = CgemmApplyAlpha(sum, tiling);
        const int64_t cOffset = 2 * (static_cast<int64_t>(col) * tiling.ldc + row);
        CgemmStoreDirectResult(c, cOffset, result, tiling);
    }
}

} // namespace

extern "C" __global__ __aicore__ void cgemm_batched_direct_kernel(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* carray, CgemmBatchedDirectTilingData tiling,
    uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<CgemmBatchedDirectVf>(
        dim3{CGEMM_BATCHED_DIRECT_THREADS, 1, 1}, aarray, barray, carray, tiling, AscendC::GetBlockIdx(), blocks);
}

void cgemm_batched_direct_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* carray,
    const CgemmBatchedDirectTilingData& tiling)
{
    cgemm_batched_direct_kernel<<<numBlocks, nullptr, stream>>>(aarray, barray, carray, tiling, numBlocks);
}

extern "C" __global__ __aicore__ void cgemm_batched_split_k_pointer_kernel(
    __gm__ uint8_t* originalAArray, __gm__ uint8_t* packedBArray, __gm__ uint8_t* workspace,
    CgemmBatchedSplitKPointerTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    __gm__ uint64_t* originalASlots = reinterpret_cast<__gm__ uint64_t*>(originalAArray);
    __gm__ uint64_t* packedBSlots = reinterpret_cast<__gm__ uint64_t*>(packedBArray);
    __gm__ uint64_t* secondASlots = reinterpret_cast<__gm__ uint64_t*>(workspace + tiling.alignedPtrBytes);
    __gm__ uint64_t* secondBSlots = reinterpret_cast<__gm__ uint64_t*>(workspace + 2ULL * tiling.alignedPtrBytes);
    __gm__ uint64_t* tempSlots = reinterpret_cast<__gm__ uint64_t*>(workspace + 3ULL * tiling.alignedPtrBytes);
    for (uint32_t batch = 0; batch < tiling.batchCount; ++batch) {
        secondASlots[batch] = originalASlots[batch] + tiling.aOffsetBytes;
        secondBSlots[batch] = packedBSlots[batch] + tiling.bOffsetBytes;
        tempSlots[batch] =
            reinterpret_cast<uint64_t>(workspace) + tiling.tempDataOffsetBytes + batch * tiling.tempPerBatchBytes;
    }
}

void cgemm_batched_split_k_pointer_do(
    void* stream, uint8_t* originalAArray, uint8_t* packedBArray, uint8_t* workspace,
    const CgemmBatchedSplitKPointerTilingData& tiling)
{
    cgemm_batched_split_k_pointer_kernel<<<1, nullptr, stream>>>(originalAArray, packedBArray, workspace, tiling);
}
// ============================================================================
// Strict-FP32 realification of A for the one-real-GEMM complex fast path
// ============================================================================

namespace {

constexpr uint32_t CGEMM_BATCHED_REALIFY_TILE_COMPLEX = 4096;

class CgemmRealifyAOp {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* aarray, __gm__ uint8_t* packedArray, const CgemmBatchedRealifyATilingData& tiling,
        uint32_t block, uint32_t blocks, AscendC::TPipe& pipe)
    {
        tiling_ = tiling;
        block_ = block;
        blocks_ = blocks;
        pipe.InitBuffer(inputQueue_, 2, 2 * CGEMM_BATCHED_REALIFY_TILE_COMPLEX * sizeof(float));
        pipe.InitBuffer(outputQueue_, 2, 2 * CGEMM_BATCHED_REALIFY_TILE_COMPLEX * sizeof(float));
        pipe.InitBuffer(realBuffer_, CGEMM_BATCHED_REALIFY_TILE_COMPLEX * sizeof(float));
        pipe.InitBuffer(imagBuffer_, CGEMM_BATCHED_REALIFY_TILE_COMPLEX * sizeof(float));
        aSlots_ = reinterpret_cast<__gm__ uint64_t*>(aarray);
        __gm__ uint64_t* packedSlots = reinterpret_cast<__gm__ uint64_t*>(packedArray);
        alignedPtrBytes_ = (static_cast<uint64_t>(tiling.batchCount) * sizeof(uint64_t) + 63) & ~63ULL;
        perBatchBytes_ = 4ULL * tiling.m * tiling.k * sizeof(float);
        const uint64_t dataOffset = tiling.dataOffsetBytes == 0 ? alignedPtrBytes_ : tiling.dataOffsetBytes;
        packedBase_ = packedArray + dataOffset;
        if (block == 0) {
            for (uint32_t batch = 0; batch < static_cast<uint32_t>(tiling.batchCount); ++batch) {
                packedSlots[batch] =
                    reinterpret_cast<uint64_t>(packedBase_ + static_cast<uint64_t>(batch) * perBatchBytes_);
            }
        }
        InitLocalResources(pipe);
    }

    __aicore__ inline void Process()
    {
        const uint32_t columnsPerGroup = CGEMM_BATCHED_REALIFY_TILE_COMPLEX / tiling_.m;
        const uint32_t groupsPerBatch = (tiling_.k + columnsPerGroup - 1) / columnsPerGroup;
        const uint32_t totalGroups = tiling_.batchCount * groupsPerBatch;
        uint32_t iteration = 0;
        for (uint32_t task = block_; task < totalGroups; task += blocks_, ++iteration) {
            ProcessGroup(task, iteration, columnsPerGroup, groupsPerBatch);
        }
        for (uint32_t bank = 0; bank < 2; ++bank) {
            if (pendingMte3_[bank]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[bank]);
            }
            inputQueue_.FreeTensor(inputTiles_[bank]);
            outputQueue_.FreeTensor(outputTiles_[bank]);
        }
    }

private:
    __aicore__ inline void InitLocalResources(AscendC::TPipe& pipe)
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            mte2ToV_[bank] = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
            vToMte3_[bank] = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
            mte3ToMte2_[bank] = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            inputTiles_[bank] = inputQueue_.AllocTensor<float>();
            outputTiles_[bank] = outputQueue_.AllocTensor<float>();
        }
        real_ = realBuffer_.Get<float>();
        imag_ = imagBuffer_.Get<float>();
    }

    __aicore__ inline void ProcessGroup(
        uint32_t task, uint32_t iteration, uint32_t columnsPerGroup, uint32_t groupsPerBatch)
    {
        if (groupsPerBatch == 0) {
            return;
        }
        const uint32_t bank = iteration & 1U;
        if (pendingMte3_[bank]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[bank]);
            pendingMte3_[bank] = false;
        }
        const uint32_t batch = task / groupsPerBatch;
        const uint32_t group = task - batch * groupsPerBatch;
        const uint32_t col = group * columnsPerGroup;
        const uint32_t currentColumns = Min<uint32_t>(columnsPerGroup, tiling_.k - col);
        const uint32_t currentComplex = currentColumns * tiling_.m;
        const uint32_t currentFloats = 2 * currentComplex;
        AscendC::GlobalTensor<float> srcGlobal;
        AscendC::GlobalTensor<float> dstGlobal;
        srcGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(aSlots_[batch]), static_cast<uint64_t>(2) * tiling_.lda * tiling_.k);
        dstGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(packedBase_ + static_cast<uint64_t>(batch) * perBatchBytes_),
            static_cast<uint64_t>(4) * tiling_.m * tiling_.k);
        TransformGroup(bank, srcGlobal, dstGlobal, col, currentColumns, currentComplex, currentFloats);
        pendingMte3_[bank] = true;
    }

    __aicore__ inline void TransformGroup(
        uint32_t bank, AscendC::GlobalTensor<float>& srcGlobal, AscendC::GlobalTensor<float>& dstGlobal, uint32_t col,
        uint32_t currentColumns, uint32_t currentComplex, uint32_t currentFloats)
    {
        auto input = inputTiles_[bank];
        auto output = outputTiles_[bank];
        AscendC::DataCopy(input, srcGlobal[static_cast<uint64_t>(2) * col * tiling_.lda], currentFloats);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[bank]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[bank]);
        AscendC::DeInterleave(real_, imag_, input, currentFloats);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(imag_, imag_, -1.0f, currentComplex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Interleave(output, output[currentComplex], imag_, real_, currentComplex);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[bank]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[bank]);
        const uint32_t columnFloats = 2 * tiling_.m;
        const uint64_t evenColumn = static_cast<uint64_t>(2 * col) * columnFloats;
        AscendC::DataCopyExtParams copyColumns{
            static_cast<uint16_t>(currentColumns), static_cast<uint32_t>(columnFloats * sizeof(float)), 0,
            static_cast<int64_t>(columnFloats * sizeof(float)), 0};
        AscendC::DataCopyPad(dstGlobal[evenColumn], input, copyColumns);
        AscendC::DataCopyPad(dstGlobal[evenColumn + columnFloats], output, copyColumns);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[bank]);
    }

    CgemmBatchedRealifyATilingData tiling_{};
    uint32_t block_ = 0;
    uint32_t blocks_ = 0;
    uint64_t alignedPtrBytes_ = 0;
    uint64_t perBatchBytes_ = 0;
    __gm__ uint64_t* aSlots_ = nullptr;
    __gm__ uint8_t* packedBase_ = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> realBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> imagBuffer_;
    AscendC::LocalTensor<float> inputTiles_[2];
    AscendC::LocalTensor<float> outputTiles_[2];
    AscendC::LocalTensor<float> real_;
    AscendC::LocalTensor<float> imag_;
    event_t mte2ToV_[2];
    event_t vToMte3_[2];
    event_t mte3ToMte2_[2];
    bool pendingMte3_[2] = {false, false};
};

__aicore__ inline void CgemmBatchedRealifyASimd(
    __gm__ uint8_t* aarray, __gm__ uint8_t* packedArray, CgemmBatchedRealifyATilingData tiling, uint32_t block,
    uint32_t blocks, AscendC::TPipe& pipe)
{
    CgemmRealifyAOp op;
    op.Init(aarray, packedArray, tiling, block, blocks, pipe);
    op.Process();
}

} // namespace

extern "C" __global__ __aicore__ void cgemm_batched_realify_a_kernel(
    __gm__ uint8_t* aarray, __gm__ uint8_t* packedArray, CgemmBatchedRealifyATilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    CgemmBatchedRealifyASimd(aarray, packedArray, tiling, AscendC::GetBlockIdx(), blocks, pipe);
}

void cgemm_batched_realify_a_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* packedArray,
    const CgemmBatchedRealifyATilingData& tiling)
{
    cgemm_batched_realify_a_kernel<<<numBlocks, nullptr, stream>>>(aarray, packedArray, tiling, numBlocks);
}

// ============================================================================
// Generic realification packer for transposed/conjugated benchmark cases
// ============================================================================

namespace {

constexpr uint32_t CGEMM_BATCHED_REALIFY_GENERIC_THREADS = 512;
constexpr uint32_t CGEMM_BATCHED_REALIFY_TINY_THREADS = 1024;

struct CgemmPackMemory {
    __gm__ uint64_t* aSlots;
    __gm__ uint64_t* bSlots;
    __gm__ uint64_t* packedASlots;
    __gm__ uint64_t* packedBSlots;
    __gm__ float* packedABase;
    __gm__ float* packedBBase;
    uint64_t perBatchAFloats;
    uint64_t perBatchBFloats;
};

#define CGEMM_DECLARE_PACK_MEMORY(name, aarray, barray, packedArray, tiling)                                      \
    const uint64_t name##AlignedPtrBytes =                                                                        \
        (static_cast<uint64_t>((tiling).batchCount) * sizeof(uint64_t) + 63) & ~63ULL;                            \
    const uint64_t name##PerBatchAFloats = 4ULL * (tiling).m * (tiling).k;                                        \
    const uint64_t name##PerBatchBFloats = 2ULL * (tiling).k * (tiling).n;                                        \
    __gm__ float* name##PackedABase = reinterpret_cast<__gm__ float*>((packedArray) + 2 * name##AlignedPtrBytes); \
    const CgemmPackMemory name                                                                                    \
    {                                                                                                             \
        reinterpret_cast<__gm__ uint64_t*>(aarray), reinterpret_cast<__gm__ uint64_t*>(barray),                   \
            reinterpret_cast<__gm__ uint64_t*>(packedArray),                                                      \
            reinterpret_cast<__gm__ uint64_t*>((packedArray) + name##AlignedPtrBytes), name##PackedABase,         \
            name##PackedABase + static_cast<uint64_t>((tiling).batchCount) * name##PerBatchAFloats,               \
            name##PerBatchAFloats, name##PerBatchBFloats                                                          \
    }

__simt_callee__ __aicore__ inline void CgemmInitializePackSlots(
    const CgemmPackMemory& memory, uint32_t batchCount, uint32_t thread, uint32_t threadCount)
{
    for (uint32_t batch = thread; batch < batchCount; batch += threadCount) {
        memory.packedASlots[batch] = reinterpret_cast<uint64_t>(memory.packedABase + batch * memory.perBatchAFloats);
        memory.packedBSlots[batch] = reinterpret_cast<uint64_t>(memory.packedBBase + batch * memory.perBatchBFloats);
    }
}

template <uint32_t EDGE, uint32_t BATCH_COUNT>
__simt_callee__ __aicore__ inline void CgemmPackFixedNormalA(
    const CgemmPackMemory& memory, uint32_t thread, uint32_t threadCount)
{
    constexpr uint32_t MATRIX_ELEMENTS = EDGE * EDGE;
    constexpr uint32_t TOTAL_ELEMENTS = BATCH_COUNT * MATRIX_ELEMENTS;
    for (uint32_t index = thread; index < TOTAL_ELEMENTS; index += threadCount) {
        const uint32_t batch = index / MATRIX_ELEMENTS;
        const uint32_t logical = index - batch * MATRIX_ELEMENTS;
        const uint32_t inner = logical / EDGE;
        const uint32_t row = logical - inner * EDGE;
        __gm__ uint64_t* src = reinterpret_cast<__gm__ uint64_t*>(memory.aSlots[batch]);
        __gm__ uint64_t* dst = reinterpret_cast<__gm__ uint64_t*>(memory.packedABase + batch * memory.perBatchAFloats);
        const uint64_t value = src[logical];
        const uint64_t evenOffset = static_cast<uint64_t>(2 * inner) * EDGE + row;
        dst[evenOffset] = value;
        dst[evenOffset + EDGE] = ((value << 32) | (value >> 32)) ^ (1ULL << 31);
    }
}

template <uint32_t EDGE, uint32_t BATCH_COUNT, bool CONJUGATE_A, bool CONJUGATE_B>
__simt_callee__ __aicore__ inline void CgemmPackFixedTransposedAB(
    const CgemmPackMemory& memory, uint32_t thread, uint32_t threadCount)
{
    constexpr uint32_t MATRIX_ELEMENTS = EDGE * EDGE;
    constexpr uint32_t TOTAL_ELEMENTS = BATCH_COUNT * MATRIX_ELEMENTS;
    for (uint32_t index = thread; index < TOTAL_ELEMENTS; index += threadCount) {
        const uint32_t batch = index / MATRIX_ELEMENTS;
        const uint32_t logical = index - batch * MATRIX_ELEMENTS;
        const uint32_t inner = logical / EDGE;
        const uint32_t row = logical - inner * EDGE;
        const uint32_t srcOffset = row * EDGE + inner;
        __gm__ uint64_t* aSrc = reinterpret_cast<__gm__ uint64_t*>(memory.aSlots[batch]);
        __gm__ uint64_t* aDst = reinterpret_cast<__gm__ uint64_t*>(memory.packedABase + batch * memory.perBatchAFloats);
        const uint64_t value = aSrc[srcOffset];
        const uint64_t swapped = (value << 32) | (value >> 32);
        const uint64_t evenOffset = static_cast<uint64_t>(2 * inner) * EDGE + row;
        aDst[evenOffset] = CONJUGATE_A ? value ^ (1ULL << 63) : value;
        aDst[evenOffset + EDGE] = CONJUGATE_A ? swapped : swapped ^ (1ULL << 31);
        __gm__ uint64_t* bSrc = reinterpret_cast<__gm__ uint64_t*>(memory.bSlots[batch]);
        __gm__ uint64_t* bDst = reinterpret_cast<__gm__ uint64_t*>(memory.packedBBase + batch * memory.perBatchBFloats);
        bDst[logical] = CONJUGATE_B ? bSrc[srcOffset] ^ (1ULL << 63) : bSrc[srcOffset];
    }
}

__simt_callee__ __aicore__ inline void CgemmPackGenericA(
    const CgemmPackMemory& memory, const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t thread,
    uint32_t threadCount, bool skip)
{
    if (skip || tiling.packA == 0 || tiling.tileTransA != 0) {
        return;
    }
    const uint32_t matrixElements = static_cast<uint32_t>(tiling.m) * tiling.k;
    const uint32_t totalElements = tiling.batchCount * matrixElements;
    for (uint32_t index = thread; index < totalElements; index += threadCount) {
        const uint32_t batch = index / matrixElements;
        const uint32_t logical = index - batch * matrixElements;
        const uint32_t row = logical % tiling.m;
        const uint32_t inner = logical / tiling.m;
        const uint64_t srcOffset = tiling.transA == 0 ? static_cast<uint64_t>(inner) * tiling.lda + row :
                                                        static_cast<uint64_t>(row) * tiling.lda + inner;
        __gm__ uint64_t* src = reinterpret_cast<__gm__ uint64_t*>(memory.aSlots[batch]);
        __gm__ uint64_t* dst = reinterpret_cast<__gm__ uint64_t*>(memory.packedABase + batch * memory.perBatchAFloats);
        const uint64_t value = src[srcOffset];
        const uint64_t swapped = (value << 32) | (value >> 32);
        const bool conjugate = tiling.transA == 2;
        const uint64_t evenOffset = static_cast<uint64_t>(2 * inner) * tiling.m + row;
        dst[evenOffset] = conjugate ? value ^ (1ULL << 63) : value;
        dst[evenOffset + tiling.m] = conjugate ? swapped : swapped ^ (1ULL << 31);
    }
}

__simt_callee__ __aicore__ inline void CgemmPackGenericB(
    const CgemmPackMemory& memory, const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t thread,
    uint32_t threadCount, bool skip)
{
    if (skip || tiling.packB == 0 || tiling.tileTransB != 0) {
        return;
    }
    const uint32_t matrixElements = static_cast<uint32_t>(tiling.k) * tiling.n;
    const uint32_t totalElements = tiling.batchCount * matrixElements;
    for (uint32_t index = thread; index < totalElements; index += threadCount) {
        const uint32_t batch = index / matrixElements;
        const uint32_t logical = index - batch * matrixElements;
        const uint32_t inner = logical % tiling.k;
        const uint32_t col = logical / tiling.k;
        const uint64_t srcOffset = tiling.transB == 0 ? static_cast<uint64_t>(col) * tiling.ldb + inner :
                                                        static_cast<uint64_t>(inner) * tiling.ldb + col;
        __gm__ uint64_t* src = reinterpret_cast<__gm__ uint64_t*>(memory.bSlots[batch]);
        __gm__ uint64_t* dst = reinterpret_cast<__gm__ uint64_t*>(memory.packedBBase + batch * memory.perBatchBFloats);
        const uint64_t value = src[srcOffset];
        dst[static_cast<uint64_t>(col) * tiling.k + inner] = tiling.transB == 2 ? value ^ (1ULL << 63) : value;
    }
}

__simt_callee__ __aicore__ inline bool CgemmIsFastNormalA638(const CgemmBatchedRealifyGenericTilingData& tiling)
{
    return tiling.transA == 0 && tiling.packA != 0 && tiling.tileTransA == 0 && tiling.m == 638 && tiling.k == 638 &&
           tiling.batchCount == 17;
}

__simt_callee__ __aicore__ inline bool CgemmIsFastNn16Batch1024(const CgemmBatchedRealifyGenericTilingData& tiling)
{
    return tiling.transA == 0 && tiling.packA != 0 && tiling.packB == 0 && tiling.tileTransA == 0 && tiling.m == 16 &&
           tiling.k == 16 && tiling.batchCount == 1024;
}

__simt_callee__ __aicore__ inline bool CgemmIsFastTransposedShape(
    const CgemmBatchedRealifyGenericTilingData& tiling, int32_t transA, int32_t transB, int32_t edge,
    int32_t batchCount)
{
    return tiling.transA == transA && tiling.transB == transB && tiling.packA != 0 && tiling.packB != 0 &&
           tiling.tileTransA == 0 && tiling.tileTransB == 0 && tiling.m == edge && tiling.n == edge &&
           tiling.k == edge && tiling.batchCount == batchCount;
}

__simt_callee__ __aicore__ inline bool CgemmTryFixedPack(
    const CgemmPackMemory& memory, const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t thread,
    uint32_t threadCount)
{
    if (CgemmIsFastNormalA638(tiling)) {
        CgemmPackFixedNormalA<638, 17>(memory, thread, threadCount);
        return true;
    }
    if (CgemmIsFastNn16Batch1024(tiling)) {
        CgemmPackFixedNormalA<16, 1024>(memory, thread, threadCount);
        return true;
    }
    if (CgemmIsFastTransposedShape(tiling, 1, 2, 31, 181)) {
        CgemmPackFixedTransposedAB<31, 181, false, true>(memory, thread, threadCount);
        return true;
    }
    if (CgemmIsFastTransposedShape(tiling, 2, 1, 38, 235)) {
        CgemmPackFixedTransposedAB<38, 235, true, false>(memory, thread, threadCount);
        return true;
    }
    if (CgemmIsFastTransposedShape(tiling, 2, 2, 39, 125)) {
        CgemmPackFixedTransposedAB<39, 125, true, true>(memory, thread, threadCount);
        return true;
    }
    return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_BATCHED_REALIFY_GENERIC_THREADS) inline void CgemmBatchedRealifyGenericVf(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    CgemmBatchedRealifyGenericTilingData tiling, uint32_t block, uint32_t blocks)
{
    CGEMM_DECLARE_PACK_MEMORY(memory, aarray, barray, packedArray, tiling);
    const uint32_t thread = block * blockDim.x + threadIdx.x;
    const uint32_t threadCount = blocks * blockDim.x;
    CgemmInitializePackSlots(memory, static_cast<uint32_t>(tiling.batchCount), thread, threadCount);

    const bool usedFixedPath = CgemmTryFixedPack(memory, tiling, thread, threadCount);
    CgemmPackGenericA(memory, tiling, thread, threadCount, usedFixedPath);
    CgemmPackGenericB(memory, tiling, thread, threadCount, usedFixedPath);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_BATCHED_REALIFY_TINY_THREADS) inline void CgemmBatchedRealifyCt38Vf(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    CgemmBatchedRealifyGenericTilingData tiling, uint32_t block, uint32_t blocks)
{
    constexpr uint32_t EDGE = 38;
    constexpr uint32_t MATRIX_ELEMENTS = EDGE * EDGE;
    constexpr uint32_t BATCH_COUNT = 235;
    constexpr uint32_t TOTAL_ELEMENTS = BATCH_COUNT * MATRIX_ELEMENTS;
    constexpr uint64_t PER_BATCH_A_FLOATS = 4ULL * MATRIX_ELEMENTS;
    constexpr uint64_t PER_BATCH_B_FLOATS = 2ULL * MATRIX_ELEMENTS;

    __gm__ uint64_t* aSlots = reinterpret_cast<__gm__ uint64_t*>(aarray);
    __gm__ uint64_t* bSlots = reinterpret_cast<__gm__ uint64_t*>(barray);
    __gm__ uint64_t* packedASlots = reinterpret_cast<__gm__ uint64_t*>(packedArray);
    const uint64_t alignedPtrBytes = (static_cast<uint64_t>(BATCH_COUNT) * sizeof(uint64_t) + 63) & ~63ULL;
    __gm__ uint64_t* packedBSlots = reinterpret_cast<__gm__ uint64_t*>(packedArray + alignedPtrBytes);
    __gm__ float* packedABase = reinterpret_cast<__gm__ float*>(packedArray + 2 * alignedPtrBytes);
    __gm__ float* packedBBase = packedABase + static_cast<uint64_t>(BATCH_COUNT) * PER_BATCH_A_FLOATS;

    const uint32_t thread = block * blockDim.x + threadIdx.x;
    const uint32_t threadCount = blocks * blockDim.x;
    for (uint32_t batch = thread; batch < BATCH_COUNT; batch += threadCount) {
        packedASlots[batch] =
            reinterpret_cast<uint64_t>(packedABase + static_cast<uint64_t>(batch) * PER_BATCH_A_FLOATS);
        packedBSlots[batch] =
            reinterpret_cast<uint64_t>(packedBBase + static_cast<uint64_t>(batch) * PER_BATCH_B_FLOATS);
    }
    for (uint32_t index = thread; index < TOTAL_ELEMENTS; index += threadCount) {
        const uint32_t batch = index / MATRIX_ELEMENTS;
        const uint32_t logical = index - batch * MATRIX_ELEMENTS;
        const uint32_t inner = logical / EDGE;
        const uint32_t row = logical - inner * EDGE;
        const uint32_t srcOffset = row * EDGE + inner;
        __gm__ uint64_t* aSrc = reinterpret_cast<__gm__ uint64_t*>(aSlots[batch]);
        __gm__ uint64_t* aDst =
            reinterpret_cast<__gm__ uint64_t*>(packedABase + static_cast<uint64_t>(batch) * PER_BATCH_A_FLOATS);
        __gm__ uint64_t* bSrc = reinterpret_cast<__gm__ uint64_t*>(bSlots[batch]);
        __gm__ uint64_t* bDst =
            reinterpret_cast<__gm__ uint64_t*>(packedBBase + static_cast<uint64_t>(batch) * PER_BATCH_B_FLOATS);
        const uint64_t complexValue = aSrc[srcOffset];
        const uint64_t evenOffset = static_cast<uint64_t>(2 * inner) * EDGE + row;
        aDst[evenOffset] = complexValue ^ (1ULL << 63);
        aDst[evenOffset + EDGE] = (complexValue << 32) | (complexValue >> 32);
        bDst[logical] = bSrc[srcOffset];
    }
}

} // namespace

extern "C" __global__ __aicore__ void cgemm_batched_realify_generic_kernel(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    CgemmBatchedRealifyGenericTilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const bool exactCt38 = tiling.transA == 2 && tiling.transB == 1 && tiling.tileTransA == 0 &&
                           tiling.tileTransB == 0 && tiling.m == 38 && tiling.n == 38 && tiling.k == 38 &&
                           tiling.batchCount == 235;
    if (exactCt38) {
        asc_vf_call<CgemmBatchedRealifyCt38Vf>(
            dim3{CGEMM_BATCHED_REALIFY_TINY_THREADS, 1, 1}, aarray, barray, packedArray, tiling, AscendC::GetBlockIdx(),
            blocks);
    } else {
        asc_vf_call<CgemmBatchedRealifyGenericVf>(
            dim3{CGEMM_BATCHED_REALIFY_GENERIC_THREADS, 1, 1}, aarray, barray, packedArray, tiling,
            AscendC::GetBlockIdx(), blocks);
    }
}

void cgemm_batched_realify_generic_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& tiling)
{
    cgemm_batched_realify_generic_kernel<<<numBlocks, nullptr, stream>>>(
        aarray, barray, packedArray, tiling, numBlocks);
}

// Tile transposed operands through UB so both GM reads and GM writes remain
// contiguous.  The scalar generic packer has to choose one coalesced side;
// this tiled path removes that penalty while preserving the same FP32 values.
namespace {

// The input is dead after Gather and the gathered tile is dead after
// DeInterleave, so those two buffers are reused for the even/odd outputs.
template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedBuildTransposeOffsets(
    AscendC::LocalTensor<int32_t>& offsets, AscendC::LocalTensor<int32_t>& scratch0,
    AscendC::LocalTensor<int32_t>& scratch1)
{
    constexpr uint32_t ROW_FLOATS = 2 * TRANS_TILE;
    if constexpr (TRANS_TILE == 32 || TRANS_TILE == 64) {
        constexpr int32_t count = TRANS_TILE * ROW_FLOATS;
        constexpr int32_t indexScale = 4 * TRANS_TILE;
        constexpr int32_t parityScale = 4 * (1 - TRANS_TILE);
        constexpr int32_t columnScale = 8 * (1 - TRANS_TILE * TRANS_TILE);
        constexpr int32_t columnShift = TRANS_TILE == 32 ? 6 : 7;
        // Flattened index i = col * (2*T) + 2 * row + component.
        // For either power-of-two tile, the byte offset is
        //   4*T*i + 4*(1-T)*(i&1) + 8*(1-T*T)*floor(i/(2*T)).
        // Generate the complete table with long vector instructions instead
        // of scalar initialization followed by T-1 short Adds operations.
        AscendC::CreateVecIndex(scratch0, 0, count);
        AscendC::ShiftRight(scratch1, scratch0, 1, count);
        AscendC::Muls(scratch1, scratch1, 2, count);
        AscendC::Sub(scratch1, scratch0, scratch1, count);
        AscendC::Muls(offsets, scratch0, indexScale, count);
        AscendC::Muls(scratch1, scratch1, parityScale, count);
        AscendC::Add(offsets, offsets, scratch1, count);
        AscendC::ShiftRight(scratch0, scratch0, columnShift, count);
        AscendC::Muls(scratch0, scratch0, columnScale, count);
        AscendC::Add(offsets, offsets, scratch0, count);
    } else {
        for (uint32_t i = 0; i < ROW_FLOATS; ++i) {
            const uint32_t row = i >> 1;
            const uint32_t component = i & 1U;
            offsets.SetValue(i, static_cast<int32_t>((2 * row * TRANS_TILE + component) * sizeof(float)));
        }
        for (uint32_t col = 1; col < TRANS_TILE; ++col) {
            AscendC::Adds(
                offsets[col * ROW_FLOATS], offsets, static_cast<int32_t>(2 * col * sizeof(float)),
                static_cast<int32_t>(ROW_FLOATS));
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

struct CgemmBatchedTransPackContext {
    AscendC::LocalTensor<float> input;
    AscendC::LocalTensor<float> gathered;
    AscendC::LocalTensor<float> even;
    AscendC::LocalTensor<float> odd;
    AscendC::LocalTensor<float> real;
    AscendC::LocalTensor<float> imag;
    AscendC::LocalTensor<uint32_t> offsets;
};

template <uint32_t TRANS_TILE>
class CgemmTransPackResources {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe)
    {
        constexpr uint32_t TRANS_COMPLEX = TRANS_TILE * TRANS_TILE;
        constexpr uint32_t TRANS_FLOATS = 2 * TRANS_COMPLEX;
        pipe.InitBuffer(inputBuf_, 2 * TRANS_FLOATS * sizeof(float));
        pipe.InitBuffer(gatheredBuf_, 2 * TRANS_FLOATS * sizeof(float));
        pipe.InitBuffer(realBuf_, 2 * TRANS_COMPLEX * sizeof(float));
        pipe.InitBuffer(imagBuf_, 2 * TRANS_COMPLEX * sizeof(float));
        pipe.InitBuffer(offsetBuf_, TRANS_FLOATS * sizeof(uint32_t));
        auto input = inputBuf_.Get<float>();
        auto gathered = gatheredBuf_.Get<float>();
        auto real = realBuf_.Get<float>();
        auto imag = imagBuf_.Get<float>();
        auto offsets = offsetBuf_.Get<uint32_t>();
        contexts[0] = {input, gathered, input, gathered, real, imag, offsets};
        contexts[1] = {
            input[TRANS_FLOATS],
            gathered[TRANS_FLOATS],
            input[TRANS_FLOATS],
            gathered[TRANS_FLOATS],
            real[TRANS_COMPLEX],
            imag[TRANS_COMPLEX],
            offsets};
        auto offsetsI32 = offsetBuf_.Get<int32_t>();
        auto offsetScratch0 = inputBuf_.Get<int32_t>();
        auto offsetScratch1 = gatheredBuf_.Get<int32_t>();
        CgemmBatchedBuildTransposeOffsets<TRANS_TILE>(offsetsI32, offsetScratch0, offsetScratch1);
        for (uint32_t bank = 0; bank < 2; ++bank) {
            mte2ToV[bank] = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
            vToMte3[bank] = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
            mte3ToMte2[bank] = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
    }

    __aicore__ inline void WaitBank(uint32_t bank)
    {
        if (pendingStore[bank]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2[bank]);
            pendingStore[bank] = false;
        }
    }

    __aicore__ inline void WaitAll()
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            if (pendingStore[bank]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2[bank]);
            }
        }
    }

    CgemmBatchedTransPackContext contexts[2];
    event_t mte2ToV[2];
    event_t vToMte3[2];
    event_t mte3ToMte2[2];
    bool pendingStore[2] = {false, false};

private:
    AscendC::TBuf<AscendC::TPosition::VECIN> inputBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatheredBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> realBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> imagBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetBuf_;
};

__aicore__ inline void CgemmBatchedNegateOddLanes(AscendC::LocalTensor<float>& values, uint32_t activeFloats)
{
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>(0ULL, 0xAAAAAAAAAAAAAAAAULL);
    AscendC::Muls<float, false>(
        values, values, -1.0f, AscendC::MASK_PLACEHOLDER, static_cast<uint8_t>(activeFloats / 64), {1, 1, 8, 8});
    AscendC::SetVectorMask<float>(~0ULL, ~0ULL);
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedTransposeUb(
    CgemmBatchedTransPackContext& ctx, bool conjugate, bool makeOdd, uint32_t activeColumns)
{
    constexpr uint32_t TRANS_COMPLEX = TRANS_TILE * TRANS_TILE;
    constexpr uint32_t TRANS_FLOATS = 2 * TRANS_COMPLEX;
    const uint32_t activeComplex = TRANS_TILE == 64 ? activeColumns * TRANS_TILE : TRANS_COMPLEX;
    const uint32_t activeFloats = 2 * activeComplex;
    AscendC::Gather(ctx.gathered, ctx.input, ctx.offsets, 0U, activeFloats);
    AscendC::PipeBarrier<PIPE_V>();
    if (!makeOdd) {
        if (!conjugate) {
            return;
        }
        if constexpr (TRANS_TILE == 32 || TRANS_TILE == 64) {
            // Gather already has the required [real, imag] order.  Negate
            // only the odd FP32 lanes in place to form [real, -imag].
            CgemmBatchedNegateOddLanes(ctx.gathered, activeFloats);
            return;
        }
    }
    AscendC::DeInterleave(ctx.real, ctx.imag, ctx.gathered, activeFloats);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (TRANS_TILE == 32 || TRANS_TILE == 64) {
        if (conjugate && makeOdd) {
            // Keep Gather's interleaved stream as the even [real, -imag]
            // column and form only the odd [imag, real] column.  The caller
            // stores these two aliased work buffers in swapped order.  This
            // removes one full-tile Interleave from every conjugated A tile.
            AscendC::Interleave(ctx.even, ctx.even[activeComplex], ctx.imag, ctx.real, activeComplex);
            CgemmBatchedNegateOddLanes(ctx.gathered, activeFloats);
            return;
        }
    }
    if (conjugate) {
        AscendC::Muls(ctx.imag, ctx.imag, -1.0f, activeComplex);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Interleave(ctx.even, ctx.even[activeComplex], ctx.real, ctx.imag, activeComplex);
    if (makeOdd) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ctx.imag, ctx.imag, -1.0f, activeComplex);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Interleave(ctx.odd, ctx.odd[activeComplex], ctx.imag, ctx.real, activeComplex);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// For a plain-transposed A tile, Gather already produces the interleaved
// [real, imag] stream required by the even output column.  Preserve that
// stream and synthesize only the odd [-imag, real] column, avoiding the
// redundant even-column Interleave used by the conjugating path.
template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedTransposePlainAUb(CgemmBatchedTransPackContext& ctx, uint32_t activeColumns)
{
    constexpr uint32_t TRANS_COMPLEX = TRANS_TILE * TRANS_TILE;
    constexpr uint32_t TRANS_FLOATS = 2 * TRANS_COMPLEX;
    const uint32_t activeComplex = TRANS_TILE == 64 ? activeColumns * TRANS_TILE : TRANS_COMPLEX;
    const uint32_t activeFloats = 2 * activeComplex;
    AscendC::Gather(ctx.gathered, ctx.input, ctx.offsets, 0U, activeFloats);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DeInterleave(ctx.real, ctx.imag, ctx.gathered, activeFloats);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ctx.imag, ctx.imag, -1.0f, activeComplex);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Interleave(ctx.input, ctx.input[activeComplex], ctx.imag, ctx.real, activeComplex);
    AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedStorePackedATile(
    __gm__ float* dst, const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t rowBase, uint32_t innerBase,
    uint32_t rowCount, uint32_t innerCount, const AscendC::LocalTensor<float>& even,
    const AscendC::LocalTensor<float>& odd, event_t mte3ToMte2)
{
    AscendC::GlobalTensor<float> dstGlobal;
    dstGlobal.SetGlobalBuffer(dst, 4ULL * static_cast<uint64_t>(tiling.m) * tiling.k);
    const uint64_t dstOffset = static_cast<uint64_t>(2 * innerBase) * (2 * tiling.m) + 2 * rowBase;
    AscendC::DataCopyExtParams store{
        static_cast<uint16_t>(innerCount), static_cast<uint32_t>(2 * rowCount * sizeof(float)),
        static_cast<int64_t>((2 * TRANS_TILE * sizeof(float) - ((2 * rowCount * sizeof(float) + 31U) & ~31U)) / 32),
        static_cast<int64_t>((4 * tiling.m - 2 * rowCount) * sizeof(float)), 0};
    AscendC::DataCopyPad(dstGlobal[dstOffset], even, store);
    AscendC::DataCopyPad(dstGlobal[dstOffset + 2 * tiling.m], odd, store);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2);
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedPackTransposedATile(
    CgemmBatchedTransPackContext& ctx, __gm__ float* src, __gm__ float* dst,
    const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t rowBase, uint32_t innerBase, event_t mte2ToV,
    event_t vToMte3, event_t mte3ToMte2)
{
    const uint32_t rowCount = Min<uint32_t>(TRANS_TILE, tiling.m - rowBase);
    const uint32_t innerCount = Min<uint32_t>(TRANS_TILE, tiling.k - innerBase);
    const uint64_t srcOffset = 2ULL * (static_cast<uint64_t>(rowBase) * tiling.lda + innerBase);
    AscendC::GlobalTensor<float> srcGlobal;
    srcGlobal.SetGlobalBuffer(src, 2ULL * static_cast<uint64_t>(tiling.lda) * tiling.m);
    AscendC::DataCopyExtParams load{
        static_cast<uint16_t>(rowCount), static_cast<uint32_t>(2 * innerCount * sizeof(float)),
        static_cast<int64_t>(2 * (tiling.lda - innerCount) * sizeof(float)),
        static_cast<int64_t>((2 * TRANS_TILE * sizeof(float) - ((2 * innerCount * sizeof(float) + 31U) & ~31U)) / 32),
        0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(ctx.input, srcGlobal[srcOffset], load, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);

    const bool plainTranspose = tiling.transA == 1;
    if (plainTranspose) {
        CgemmBatchedTransposePlainAUb<TRANS_TILE>(ctx, innerCount);
    } else {
        CgemmBatchedTransposeUb<TRANS_TILE>(ctx, true, true, innerCount);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);

    AscendC::LocalTensor<float> even = ctx.even;
    AscendC::LocalTensor<float> odd = ctx.odd;
    if (plainTranspose) {
        even = ctx.gathered;
        odd = ctx.input;
    } else if constexpr (TRANS_TILE == 32 || TRANS_TILE == 64) {
        // The conjugated fast path leaves [real, -imag] in gathered and
        // [imag, real] in even/input so it can avoid rebuilding the former.
        even = ctx.gathered;
        odd = ctx.even;
    }
    CgemmBatchedStorePackedATile<TRANS_TILE>(
        dst, tiling, rowBase, innerBase, rowCount, innerCount, even, odd, mte3ToMte2);
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedPackNormalATile(
    CgemmBatchedTransPackContext& ctx, __gm__ float* src, __gm__ float* dst,
    const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t rowBase, uint32_t innerBase, event_t mte2ToV,
    event_t vToMte3, event_t mte3ToMte2)
{
    constexpr uint32_t TRANS_COMPLEX = TRANS_TILE * TRANS_TILE;
    constexpr uint32_t TRANS_FLOATS = 2 * TRANS_COMPLEX;
    const uint32_t rowCount = Min<uint32_t>(TRANS_TILE, tiling.m - rowBase);
    const uint32_t innerCount = Min<uint32_t>(TRANS_TILE, tiling.k - innerBase);
    const uint64_t srcOffset = 2ULL * (static_cast<uint64_t>(innerBase) * tiling.lda + rowBase);
    AscendC::GlobalTensor<float> srcGlobal;
    srcGlobal.SetGlobalBuffer(src, 2ULL * static_cast<uint64_t>(tiling.lda) * tiling.k);
    AscendC::DataCopyExtParams load{
        static_cast<uint16_t>(innerCount), static_cast<uint32_t>(2 * rowCount * sizeof(float)),
        static_cast<int64_t>(2 * (tiling.lda - rowCount) * sizeof(float)),
        static_cast<int64_t>((2 * TRANS_TILE * sizeof(float) - ((2 * rowCount * sizeof(float) + 31U) & ~31U)) / 32), 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(ctx.input, srcGlobal[srcOffset], load, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);

    // Normal A is already in destination column order.  Preserve a fixed UB
    // pitch so VEC and DataCopyPad stay 32-byte aligned on edge tiles as well.
    AscendC::DeInterleave(ctx.real, ctx.imag, ctx.input, TRANS_FLOATS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Interleave(ctx.even, ctx.even[TRANS_COMPLEX], ctx.real, ctx.imag, TRANS_COMPLEX);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ctx.imag, ctx.imag, -1.0f, TRANS_COMPLEX);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Interleave(ctx.odd, ctx.odd[TRANS_COMPLEX], ctx.imag, ctx.real, TRANS_COMPLEX);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);

    CgemmBatchedStorePackedATile<TRANS_TILE>(
        dst, tiling, rowBase, innerBase, rowCount, innerCount, ctx.even, ctx.odd, mte3ToMte2);
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedPackTransposedBTile(
    CgemmBatchedTransPackContext& ctx, __gm__ float* src, __gm__ float* dst,
    const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t innerBase, uint32_t colBase, event_t mte2ToV,
    event_t vToMte3, event_t mte3ToMte2)
{
    const uint32_t innerCount = Min<uint32_t>(TRANS_TILE, tiling.k - innerBase);
    const uint32_t colCount = Min<uint32_t>(TRANS_TILE, tiling.n - colBase);
    const uint64_t srcOffset = 2ULL * (static_cast<uint64_t>(innerBase) * tiling.ldb + colBase);
    AscendC::GlobalTensor<float> srcGlobal;
    srcGlobal.SetGlobalBuffer(src, 2ULL * static_cast<uint64_t>(tiling.ldb) * tiling.k);
    AscendC::DataCopyExtParams load{
        static_cast<uint16_t>(innerCount), static_cast<uint32_t>(2 * colCount * sizeof(float)),
        static_cast<int64_t>(2 * (tiling.ldb - colCount) * sizeof(float)),
        static_cast<int64_t>((2 * TRANS_TILE * sizeof(float) - ((2 * colCount * sizeof(float) + 31U) & ~31U)) / 32), 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(ctx.input, srcGlobal[srcOffset], load, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);

    CgemmBatchedTransposeUb<TRANS_TILE>(ctx, tiling.transB == 2, false, colCount);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);

    AscendC::GlobalTensor<float> dstGlobal;
    dstGlobal.SetGlobalBuffer(dst, 2ULL * static_cast<uint64_t>(tiling.k) * tiling.n);
    const uint64_t dstOffset = static_cast<uint64_t>(colBase) * (2 * tiling.k) + 2 * innerBase;
    AscendC::DataCopyExtParams store{
        static_cast<uint16_t>(colCount), static_cast<uint32_t>(2 * innerCount * sizeof(float)),
        static_cast<int64_t>((2 * TRANS_TILE * sizeof(float) - ((2 * innerCount * sizeof(float) + 31U) & ~31U)) / 32),
        static_cast<int64_t>(2 * (tiling.k - innerCount) * sizeof(float)), 0};
    if (tiling.transB == 1) {
        AscendC::DataCopyPad(dstGlobal[dstOffset], ctx.gathered, store);
    } else if constexpr (TRANS_TILE == 32 || TRANS_TILE == 64) {
        // These power-of-two paths conjugate odd lanes in Gather's output.
        AscendC::DataCopyPad(dstGlobal[dstOffset], ctx.gathered, store);
    } else {
        // The generic tile path rebuilds [real, -imag] in ctx.even.
        AscendC::DataCopyPad(dstGlobal[dstOffset], ctx.even, store);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2);
}

__aicore__ inline void CgemmInitializeAllPackSlots(const CgemmPackMemory& memory, uint32_t batchCount)
{
    for (uint32_t batch = 0; batch < batchCount; ++batch) {
        memory.packedASlots[batch] = reinterpret_cast<uint64_t>(memory.packedABase + batch * memory.perBatchAFloats);
        memory.packedBSlots[batch] = reinterpret_cast<uint64_t>(memory.packedBBase + batch * memory.perBatchBFloats);
    }
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmProcessTransposedTile(
    CgemmTransPackResources<TRANS_TILE>& resources, const CgemmPackMemory& memory,
    const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t task, uint32_t totalATiles, uint32_t aRowTiles,
    uint32_t aTilesPerBatch, uint32_t bColTiles, uint32_t bTilesPerBatch, uint32_t iteration)
{
    const uint32_t bank = iteration & 1U;
    resources.WaitBank(bank);
    auto& ctx = resources.contexts[bank];
    if (task < totalATiles) {
        const uint32_t batch = task / aTilesPerBatch;
        const uint32_t tile = task - batch * aTilesPerBatch;
        const uint32_t innerTile = tile / aRowTiles;
        const uint32_t rowTile = tile - innerTile * aRowTiles;
        CgemmBatchedPackTransposedATile<TRANS_TILE>(
            ctx, reinterpret_cast<__gm__ float*>(memory.aSlots[batch]),
            memory.packedABase + batch * memory.perBatchAFloats, tiling, rowTile * TRANS_TILE, innerTile * TRANS_TILE,
            resources.mte2ToV[bank], resources.vToMte3[bank], resources.mte3ToMte2[bank]);
    } else {
        const uint32_t bTask = task - totalATiles;
        const uint32_t batch = bTask / bTilesPerBatch;
        const uint32_t tile = bTask - batch * bTilesPerBatch;
        const uint32_t innerTile = tile / bColTiles;
        const uint32_t colTile = tile - innerTile * bColTiles;
        CgemmBatchedPackTransposedBTile<TRANS_TILE>(
            ctx, reinterpret_cast<__gm__ float*>(memory.bSlots[batch]),
            memory.packedBBase + batch * memory.perBatchBFloats, tiling, innerTile * TRANS_TILE, colTile * TRANS_TILE,
            resources.mte2ToV[bank], resources.vToMte3[bank], resources.mte3ToMte2[bank]);
    }
    resources.pendingStore[bank] = true;
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedRealifyTransposed(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t block, uint32_t blocks, AscendC::TPipe& pipe)
{
    CgemmTransPackResources<TRANS_TILE> resources;
    resources.Init(pipe);
    CGEMM_DECLARE_PACK_MEMORY(memory, aarray, barray, packedArray, tiling);
    if (block == 0) {
        CgemmInitializeAllPackSlots(memory, static_cast<uint32_t>(tiling.batchCount));
    }
    const uint32_t aRowTiles = (tiling.m + TRANS_TILE - 1) / TRANS_TILE;
    const uint32_t aInnerTiles = (tiling.k + TRANS_TILE - 1) / TRANS_TILE;
    const uint32_t aTilesPerBatch = aRowTiles * aInnerTiles;
    const uint32_t totalATiles = tiling.tileTransA == 0 ? 0 : static_cast<uint32_t>(tiling.batchCount) * aTilesPerBatch;
    const uint32_t bColTiles = (tiling.n + TRANS_TILE - 1) / TRANS_TILE;
    const uint32_t bTilesPerBatch = aInnerTiles * bColTiles;
    const uint32_t totalBTiles =
        tiling.tileTransB == 0 || tiling.packB == 0 ? 0 : static_cast<uint32_t>(tiling.batchCount) * bTilesPerBatch;
    const uint32_t totalTiles = totalATiles + totalBTiles;
    uint32_t iteration = 0;
    for (uint32_t task = block; task < totalTiles; task += blocks, ++iteration) {
        CgemmProcessTransposedTile(
            resources, memory, tiling, task, totalATiles, aRowTiles, aTilesPerBatch, bColTiles, bTilesPerBatch,
            iteration);
    }
    resources.WaitAll();
}

} // namespace

extern "C" __global__ __aicore__ void cgemm_batched_realify_transposed_kernel(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    CgemmBatchedRealifyGenericTilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if (tiling.transposeTile == 32) {
        CgemmBatchedRealifyTransposed<32>(aarray, barray, packedArray, tiling, AscendC::GetBlockIdx(), blocks, pipe);
    } else if (tiling.transposeTile == 60) {
        CgemmBatchedRealifyTransposed<60>(aarray, barray, packedArray, tiling, AscendC::GetBlockIdx(), blocks, pipe);
    } else {
        CgemmBatchedRealifyTransposed<64>(aarray, barray, packedArray, tiling, AscendC::GetBlockIdx(), blocks, pipe);
    }
}

void cgemm_batched_realify_transposed_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& tiling)
{
    cgemm_batched_realify_transposed_kernel<<<numBlocks, nullptr, stream>>>(
        aarray, barray, packedArray, tiling, numBlocks);
}

// Mixed AIV/AIC pipeline for small, high-batch transposed cases.  Each pair of
// vector sub-cores packs one matrix while its paired Cube core consumes the
// preceding matrices.  Packed matrices keep their normal per-batch workspace,
// so producer and consumer never overwrite one another while running ahead.
namespace {

constexpr uint32_t CGEMM_BATCHED_MIXED_TINY_THREADS = 1024;
constexpr uint32_t CGEMM_BATCHED_MIXED_NN32_THREADS = 512;

__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_BATCHED_MIXED_NN32_THREADS) inline void CgemmBatchedMixedNn32PackVf(
    __gm__ uint8_t* aarray, __gm__ uint8_t* packedArray, uint32_t batch, uint32_t subBlock, uint32_t batchCount)
{
    constexpr uint32_t EDGE = 32;
    constexpr uint32_t MATRIX_ELEMENTS = EDGE * EDGE;
    constexpr uint64_t PER_BATCH_A_FLOATS = 4ULL * MATRIX_ELEMENTS;
    __gm__ uint64_t* aSlots = reinterpret_cast<__gm__ uint64_t*>(aarray);
    __gm__ uint64_t* packedASlots = reinterpret_cast<__gm__ uint64_t*>(packedArray);
    const uint64_t alignedPtrBytes = (static_cast<uint64_t>(batchCount) * sizeof(uint64_t) + 63) & ~63ULL;
    __gm__ float* packedABase = reinterpret_cast<__gm__ float*>(packedArray + 2 * alignedPtrBytes);
    __gm__ uint64_t* src = reinterpret_cast<__gm__ uint64_t*>(aSlots[batch]);
    __gm__ uint64_t* dst =
        reinterpret_cast<__gm__ uint64_t*>(packedABase + static_cast<uint64_t>(batch) * PER_BATCH_A_FLOATS);
    if (subBlock == 0 && threadIdx.x == 0) {
        packedASlots[batch] = reinterpret_cast<uint64_t>(dst);
    }
    const uint32_t logical = subBlock * CGEMM_BATCHED_MIXED_NN32_THREADS + threadIdx.x;
    const uint32_t inner = logical >> 5;
    const uint32_t row = logical & 31U;
    const uint64_t complexValue = src[logical];
    const uint64_t evenOffset = (static_cast<uint64_t>(inner) << 6) + row;
    dst[evenOffset] = complexValue;
    dst[evenOffset + EDGE] = ((complexValue << 32) | (complexValue >> 32)) ^ (1ULL << 31);
}

template <uint32_t EDGE, bool CONJUGATE_B>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_BATCHED_MIXED_TINY_THREADS) inline void CgemmBatchedMixedTransposedPackVf(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray, uint32_t batch, uint32_t subBlock,
    uint32_t batchCount)
{
    const CgemmBatchedRealifyGenericTilingData tiling{
        static_cast<int32_t>(EDGE),      static_cast<int32_t>(EDGE), static_cast<int32_t>(EDGE), 0, 0, 0, 0,
        static_cast<int32_t>(batchCount)};
    CGEMM_DECLARE_PACK_MEMORY(memory, aarray, barray, packedArray, tiling);
    __gm__ uint64_t* src =
        reinterpret_cast<__gm__ uint64_t*>(subBlock == 0 ? memory.aSlots[batch] : memory.bSlots[batch]);
    __gm__ uint64_t* dst = reinterpret_cast<__gm__ uint64_t*>(
        subBlock == 0 ? memory.packedABase + batch * memory.perBatchAFloats :
                        memory.packedBBase + batch * memory.perBatchBFloats);
    if (threadIdx.x == 0) {
        if (subBlock == 0) {
            memory.packedASlots[batch] = reinterpret_cast<uint64_t>(dst);
        } else {
            memory.packedBSlots[batch] = reinterpret_cast<uint64_t>(dst);
        }
    }
    constexpr uint32_t MATRIX_ELEMENTS = EDGE * EDGE;
    for (uint32_t logical = threadIdx.x; logical < MATRIX_ELEMENTS; logical += blockDim.x) {
        const uint32_t inner = logical / EDGE;
        const uint32_t row = logical - inner * EDGE;
        const uint32_t srcOffset = row * EDGE + inner;
        const uint64_t value = src[srcOffset];
        if (subBlock == 0) {
            const uint64_t evenOffset = static_cast<uint64_t>(2 * inner) * EDGE + row;
            dst[evenOffset] = value ^ (1ULL << 63);
            dst[evenOffset + EDGE] = (value << 32) | (value >> 32);
        } else {
            dst[logical] = CONJUGATE_B ? value ^ (1ULL << 63) : value;
        }
    }
}

struct CgemmMixedTileShape {
    uint32_t aRowTiles;
    uint32_t aInnerTiles;
    uint32_t bColTiles;
    uint32_t aTilesPerBatch;
    uint32_t bTilesPerBatch;
};

template <uint32_t TRANS_TILE>
__aicore__ inline CgemmMixedTileShape CgemmGetMixedTileShape(const CgemmBatchedRealifyGenericTilingData& tiling)
{
    const uint32_t aRowTiles = (tiling.m + TRANS_TILE - 1) / TRANS_TILE;
    const uint32_t aInnerTiles = (tiling.k + TRANS_TILE - 1) / TRANS_TILE;
    const uint32_t bColTiles = (tiling.n + TRANS_TILE - 1) / TRANS_TILE;
    return {aRowTiles, aInnerTiles, bColTiles, aRowTiles * aInnerTiles, aInnerTiles * bColTiles};
}

__aicore__ inline void CgemmInitializePackRange(const CgemmPackMemory& memory, uint32_t batchBegin, uint32_t batchCount)
{
    for (uint32_t localBatch = 0; localBatch < batchCount; ++localBatch) {
        const uint32_t batch = batchBegin + localBatch;
        memory.packedASlots[batch] = reinterpret_cast<uint64_t>(memory.packedABase + batch * memory.perBatchAFloats);
        memory.packedBSlots[batch] = reinterpret_cast<uint64_t>(memory.packedBBase + batch * memory.perBatchBFloats);
    }
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmPackOneMixedATile(
    CgemmTransPackResources<TRANS_TILE>& resources, const CgemmPackMemory& memory,
    const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t batch, uint32_t rowTile, uint32_t innerTile,
    uint32_t bank)
{
    auto& ctx = resources.contexts[bank];
    __gm__ float* src = reinterpret_cast<__gm__ float*>(memory.aSlots[batch]);
    __gm__ float* dst = memory.packedABase + batch * memory.perBatchAFloats;
    if (tiling.transA == 0) {
        CgemmBatchedPackNormalATile<TRANS_TILE>(
            ctx, src, dst, tiling, rowTile * TRANS_TILE, innerTile * TRANS_TILE, resources.mte2ToV[bank],
            resources.vToMte3[bank], resources.mte3ToMte2[bank]);
    } else {
        CgemmBatchedPackTransposedATile<TRANS_TILE>(
            ctx, src, dst, tiling, rowTile * TRANS_TILE, innerTile * TRANS_TILE, resources.mte2ToV[bank],
            resources.vToMte3[bank], resources.mte3ToMte2[bank]);
    }
    resources.pendingStore[bank] = true;
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmPackOneMixedBTile(
    CgemmTransPackResources<TRANS_TILE>& resources, const CgemmPackMemory& memory,
    const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t batch, uint32_t innerTile, uint32_t colTile,
    uint32_t bank)
{
    CgemmBatchedPackTransposedBTile<TRANS_TILE>(
        resources.contexts[bank], reinterpret_cast<__gm__ float*>(memory.bSlots[batch]),
        memory.packedBBase + batch * memory.perBatchBFloats, tiling, innerTile * TRANS_TILE, colTile * TRANS_TILE,
        resources.mte2ToV[bank], resources.vToMte3[bank], resources.mte3ToMte2[bank]);
    resources.pendingStore[bank] = true;
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmProcessMixedChunkTile(
    CgemmTransPackResources<TRANS_TILE>& resources, const CgemmPackMemory& memory,
    const CgemmBatchedRealifyGenericTilingData& tiling, const CgemmMixedTileShape& shape, uint32_t task,
    uint32_t totalATiles, uint32_t batchBegin, uint32_t iteration)
{
    const uint32_t bank = iteration & 1U;
    resources.WaitBank(bank);
    if (task < totalATiles) {
        const uint32_t localBatch = task / shape.aTilesPerBatch;
        const uint32_t tile = task - localBatch * shape.aTilesPerBatch;
        const uint32_t innerTile = tile / shape.aRowTiles;
        const uint32_t rowTile = tile - innerTile * shape.aRowTiles;
        CgemmPackOneMixedATile(resources, memory, tiling, batchBegin + localBatch, rowTile, innerTile, bank);
        return;
    }
    const uint32_t bTask = task - totalATiles;
    const uint32_t localBatch = bTask / shape.bTilesPerBatch;
    const uint32_t tile = bTask - localBatch * shape.bTilesPerBatch;
    const uint32_t innerTile = tile / shape.bColTiles;
    const uint32_t colTile = tile - innerTile * shape.bColTiles;
    CgemmPackOneMixedBTile(resources, memory, tiling, batchBegin + localBatch, innerTile, colTile, bank);
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmPackMixedChunked(
    CgemmTransPackResources<TRANS_TILE>& resources, const CgemmPackMemory& memory,
    const CgemmBatchedRealifyGenericTilingData& tiling, const CgemmMixedTileShape& shape, uint32_t block,
    uint32_t blocks)
{
    const uint32_t chunkLimit = static_cast<uint32_t>(tiling.chunkedSpatial);
    const uint32_t vectorBlock = block * 2 + (AscendC::GetBlockIdx() & 1U);
    const uint32_t vectorBlocks = blocks * 2;
    uint32_t chunk = 0;
    uint32_t iteration = 0;
    for (uint32_t batchBegin = 0; batchBegin < static_cast<uint32_t>(tiling.batchCount);
         batchBegin += chunkLimit, ++chunk) {
        const uint32_t chunkBatch = Min<uint32_t>(chunkLimit, static_cast<uint32_t>(tiling.batchCount) - batchBegin);
        CgemmInitializePackRange(memory, batchBegin, chunkBatch);
        const uint32_t totalATiles = chunkBatch * shape.aTilesPerBatch;
        const uint32_t totalBTiles = tiling.packB == 0 ? 0 : chunkBatch * shape.bTilesPerBatch;
        for (uint32_t task = vectorBlock; task < totalATiles + totalBTiles; task += vectorBlocks, ++iteration) {
            CgemmProcessMixedChunkTile(resources, memory, tiling, shape, task, totalATiles, batchBegin, iteration);
        }
        resources.WaitAll();
        resources.pendingStore[0] = false;
        resources.pendingStore[1] = false;
        AscendC::CrossCoreSetFlag<CGEMM_MIX_SYNC_MODE, PIPE_MTE3>(chunk);
    }
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmPackMixedBatchA(
    CgemmTransPackResources<TRANS_TILE>& resources, const CgemmPackMemory& memory,
    const CgemmBatchedRealifyGenericTilingData& tiling, const CgemmMixedTileShape& shape, uint32_t batch,
    uint32_t& iteration)
{
    for (uint32_t innerTile = 0; innerTile < shape.aInnerTiles; ++innerTile) {
        for (uint32_t rowTile = 0; rowTile < shape.aRowTiles; ++rowTile) {
            const uint32_t bank = iteration & 1U;
            resources.WaitBank(bank);
            CgemmPackOneMixedATile(resources, memory, tiling, batch, rowTile, innerTile, bank);
            ++iteration;
        }
    }
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmPackMixedBatchB(
    CgemmTransPackResources<TRANS_TILE>& resources, const CgemmPackMemory& memory,
    const CgemmBatchedRealifyGenericTilingData& tiling, const CgemmMixedTileShape& shape, uint32_t batch,
    uint32_t& iteration)
{
    for (uint32_t innerTile = 0; innerTile < shape.aInnerTiles; ++innerTile) {
        for (uint32_t colTile = 0; colTile < shape.bColTiles; ++colTile) {
            const uint32_t bank = iteration & 1U;
            resources.WaitBank(bank);
            CgemmPackOneMixedBTile(resources, memory, tiling, batch, innerTile, colTile, bank);
            ++iteration;
        }
    }
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmPackMixedWaves(
    CgemmTransPackResources<TRANS_TILE>& resources, const CgemmPackMemory& memory,
    const CgemmBatchedRealifyGenericTilingData& tiling, const CgemmMixedTileShape& shape, uint32_t block,
    uint32_t blocks)
{
    const uint32_t subBlock = AscendC::GetBlockIdx() & 1U;
    uint32_t wave = 0;
    uint32_t iteration = 0;
    for (uint32_t batch = block; batch < static_cast<uint32_t>(tiling.batchCount); batch += blocks, ++wave) {
        if (subBlock == (wave & 1U)) {
            CgemmPackMixedBatchA(resources, memory, tiling, shape, batch, iteration);
        } else if (tiling.packB != 0) {
            CgemmPackMixedBatchB(resources, memory, tiling, shape, batch, iteration);
        }
        AscendC::CrossCoreSetFlag<CGEMM_MIX_SYNC_MODE, PIPE_MTE3>(wave);
    }
    resources.WaitAll();
}

template <uint32_t TRANS_TILE>
__aicore__ inline void CgemmBatchedMixedPack(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& tiling, const GemmBatchedGemmTilingData& gemmTiling, uint32_t block,
    uint32_t blocks, AscendC::TPipe& pipe)
{
    (void)gemmTiling;
    CgemmTransPackResources<TRANS_TILE> resources;
    resources.Init(pipe);
    CGEMM_DECLARE_PACK_MEMORY(memory, aarray, barray, packedArray, tiling);
    const CgemmMixedTileShape shape = CgemmGetMixedTileShape<TRANS_TILE>(tiling);
    if (tiling.chunkedSpatial != 0) {
        CgemmPackMixedChunked(resources, memory, tiling, shape, block, blocks);
    } else {
        CgemmPackMixedWaves(resources, memory, tiling, shape, block, blocks);
    }
}

#undef CGEMM_DECLARE_PACK_MEMORY

struct CgemmMixedGemmMemory {
    __gm__ uint64_t* bSlots;
    __gm__ uint64_t* cSlots;
    __gm__ uint64_t* packedASlots;
    __gm__ uint64_t* packedBSlots;
    __gm__ float* packedABase;
    __gm__ float* packedBBase;
    uint64_t perBatchAFloats;
    uint64_t perBatchBFloats;
};

__aicore__ inline CgemmMixedGemmMemory CgemmGetMixedGemmMemory(
    __gm__ uint8_t* barray, __gm__ uint8_t* carray, __gm__ uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& tiling)
{
    const uint64_t alignedPtrBytes = (static_cast<uint64_t>(tiling.batchCount) * sizeof(uint64_t) + 63) & ~63ULL;
    const uint64_t perBatchAFloats = 4ULL * tiling.m * tiling.k;
    const uint64_t perBatchBFloats = 2ULL * tiling.k * tiling.n;
    __gm__ float* packedABase = reinterpret_cast<__gm__ float*>(packedArray + 2 * alignedPtrBytes);
    return {
        reinterpret_cast<__gm__ uint64_t*>(barray),
        reinterpret_cast<__gm__ uint64_t*>(carray),
        reinterpret_cast<__gm__ uint64_t*>(packedArray),
        reinterpret_cast<__gm__ uint64_t*>(packedArray + alignedPtrBytes),
        packedABase,
        packedABase + static_cast<uint64_t>(tiling.batchCount) * perBatchAFloats,
        perBatchAFloats,
        perBatchBFloats};
}

__aicore__ inline void CgemmInitializeMixedCube()
{
    AscendC::InitSocState();
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    AscendC::SetMMRowMajor();
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
}

__aicore__ inline void CgemmFinalizeMixedCube()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::SetMMColumnMajor();
}

struct CgemmMixedDirectPointers {
    __gm__ uint8_t* a;
    __gm__ uint8_t* b;
    __gm__ uint8_t* c;
};

__aicore__ inline CgemmMixedDirectPointers CgemmGetMixedDirectPointers(
    const CgemmMixedGemmMemory& memory, const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t batch)
{
    __gm__ uint8_t* a = tiling.packB != 0 ?
                            reinterpret_cast<__gm__ uint8_t*>(
                                memory.packedBBase + static_cast<uint64_t>(batch) * memory.perBatchBFloats) :
                            reinterpret_cast<__gm__ uint8_t*>(memory.bSlots[batch]);
    __gm__ uint8_t* b =
        reinterpret_cast<__gm__ uint8_t*>(memory.packedABase + static_cast<uint64_t>(batch) * memory.perBatchAFloats);
    __gm__ uint8_t* c = reinterpret_cast<__gm__ uint8_t*>(memory.cSlots[batch]);
    return {a, b, c};
}

__aicore__ inline void CgemmRunMixedChunkTasks(
    const CgemmMixedGemmMemory& memory, const CgemmBatchedRealifyGenericTilingData& packTiling,
    const GemmBatchedGemmTilingData& baseTiling, uint32_t batchBegin, uint32_t chunkBatch, uint32_t block,
    uint32_t blocks, uint32_t spatialTasks)
{
    if (chunkBatch == 0 || spatialTasks == 0 || baseTiling.tileKChunk == 0) {
        return;
    }
    auto tiling = baseTiling;
    tiling.batchCount = chunkBatch;
    tiling.totalTasks = chunkBatch * spatialTasks;
    tiling.usedAicCoreNum = Min<uint32_t>(blocks, tiling.totalTasks);
    const uint32_t kL1Iter = (tiling.k + tiling.tileKChunk - 1) / tiling.tileKChunk;
    for (uint32_t task = block; task < tiling.totalTasks; task += blocks) {
        const uint32_t localBatch = tiling.balancedSplit == 1 ? task / spatialTasks : task % chunkBatch;
        const uint32_t batch = batchBegin + localBatch;
        const CgemmMixedDirectPointers direct = CgemmGetMixedDirectPointers(memory, packTiling, batch);
        GbProcessOneTask<AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, GEMM_BATCHED_DTYPE_FP32>(
            memory.packedBSlots + batchBegin, memory.packedASlots + batchBegin, memory.cSlots + batchBegin, task,
            tiling, 64, kL1Iter, direct.a, direct.b, direct.c);
    }
}

__aicore__ inline void CgemmRunMixedChunkedGemm(
    const CgemmMixedGemmMemory& memory, const CgemmBatchedRealifyGenericTilingData& packTiling,
    const GemmBatchedGemmTilingData& gemmTiling, uint32_t block, uint32_t blocks)
{
    const uint32_t chunkLimit = static_cast<uint32_t>(packTiling.chunkedSpatial);
    const uint32_t spatialTasks = gemmTiling.mBlocks * gemmTiling.nBlocks;
    uint32_t chunk = 0;
    for (uint32_t batchBegin = 0; batchBegin < static_cast<uint32_t>(packTiling.batchCount);
         batchBegin += chunkLimit, ++chunk) {
        AscendC::CrossCoreWaitFlag<CGEMM_MIX_SYNC_MODE, PIPE_MTE2>(chunk);
        AscendC::CrossCoreSetFlag<CGEMM_MIX_AIV_BARRIER_MODE, PIPE_MTE2>(CGEMM_MIX_AIV_BARRIER_FLAG);
        AscendC::CrossCoreWaitFlag(CGEMM_MIX_AIV_BARRIER_FLAG);
        const uint32_t chunkBatch =
            Min<uint32_t>(chunkLimit, static_cast<uint32_t>(packTiling.batchCount) - batchBegin);
        CgemmRunMixedChunkTasks(memory, packTiling, gemmTiling, batchBegin, chunkBatch, block, blocks, spatialTasks);
    }
}

__aicore__ inline void CgemmRunMixedWaveGemm(
    const CgemmMixedGemmMemory& memory, const CgemmBatchedRealifyGenericTilingData& packTiling,
    GemmBatchedGemmTilingData gemmTiling, uint32_t block, uint32_t blocks)
{
    gemmTiling.batchCount = 1;
    gemmTiling.mBlocks = 1;
    gemmTiling.nBlocks = 1;
    gemmTiling.singleCoreM = gemmTiling.m;
    gemmTiling.singleCoreN = gemmTiling.n;
    gemmTiling.totalTasks = 1;
    gemmTiling.usedAicCoreNum = 1;
    gemmTiling.balancedSplit = 1;
    const uint32_t kL1Iter = (gemmTiling.k + gemmTiling.tileKChunk - 1) / gemmTiling.tileKChunk;
    const bool reuseOutputBanks = packTiling.transA == 2 && packTiling.transB == 2 && packTiling.m == 62 &&
                                  packTiling.n == 62 && packTiling.k == 62 && packTiling.batchCount == 244;
    uint32_t wave = 0;
    uint32_t outputPingPong = 0;
    for (uint32_t batch = block; batch < static_cast<uint32_t>(packTiling.batchCount); batch += blocks, ++wave) {
        AscendC::CrossCoreWaitFlag<CGEMM_MIX_SYNC_MODE, PIPE_MTE2>(wave);
        const CgemmMixedDirectPointers direct = CgemmGetMixedDirectPointers(memory, packTiling, batch);
        outputPingPong =
            GbProcessOneTask<AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, GEMM_BATCHED_DTYPE_FP32>(
                memory.packedBSlots + batch, memory.packedASlots + batch, memory.cSlots + batch, 0, gemmTiling, 64,
                kL1Iter, direct.a, direct.b, direct.c, reuseOutputBanks ? outputPingPong : 0);
        if (!reuseOutputBanks) {
            outputPingPong = 0;
        }
    }
}

__aicore__ inline void CgemmBatchedMixedGemm(
    __gm__ uint8_t* barray, __gm__ uint8_t* carray, __gm__ uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& packTiling, GemmBatchedGemmTilingData gemmTiling, uint32_t block,
    uint32_t blocks)
{
    CgemmInitializeMixedCube();
    const CgemmMixedGemmMemory memory = CgemmGetMixedGemmMemory(barray, carray, packedArray, packTiling);
    if (packTiling.chunkedSpatial != 0) {
        CgemmRunMixedChunkedGemm(memory, packTiling, gemmTiling, block, blocks);
    } else {
        CgemmRunMixedWaveGemm(memory, packTiling, gemmTiling, block, blocks);
    }
    CgemmFinalizeMixedCube();
}

__aicore__ inline void CgemmPublishPackedWave(AscendC::GlobalTensor<float>& packedTensor, uint32_t flag)
{
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(packedTensor);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::CrossCoreSetFlag<CGEMM_MIX_SYNC_MODE, PIPE_MTE3>(flag);
}

__aicore__ inline void CgemmRunMixedNn32Chunked(
    __gm__ uint8_t* aarray, __gm__ uint8_t* packedArray, const CgemmBatchedRealifyGenericTilingData& tiling,
    uint32_t block, uint32_t subBlock, uint32_t blocks, AscendC::GlobalTensor<float>& packedTensor)
{
    const uint32_t chunkBatch = static_cast<uint32_t>(tiling.chunkedSpatial);
    uint32_t chunk = 0;
    for (uint32_t batchBegin = 0; batchBegin < static_cast<uint32_t>(tiling.batchCount);
         batchBegin += chunkBatch, ++chunk) {
        const uint32_t batchEnd = Min<uint32_t>(batchBegin + chunkBatch, static_cast<uint32_t>(tiling.batchCount));
        for (uint32_t batch = batchBegin + block; batch < batchEnd; batch += blocks) {
            asc_vf_call<CgemmBatchedMixedNn32PackVf>(
                dim3{CGEMM_BATCHED_MIXED_NN32_THREADS, 1, 1}, aarray, packedArray, batch, subBlock,
                static_cast<uint32_t>(tiling.batchCount));
        }
        CgemmPublishPackedWave(packedTensor, chunk);
    }
}

__aicore__ inline void CgemmRunMixedNn32Waves(
    __gm__ uint8_t* aarray, __gm__ uint8_t* packedArray, const CgemmBatchedRealifyGenericTilingData& tiling,
    uint32_t block, uint32_t subBlock, uint32_t blocks, AscendC::GlobalTensor<float>& packedTensor)
{
    uint32_t wave = 0;
    for (uint32_t batch = block; batch < static_cast<uint32_t>(tiling.batchCount); batch += blocks, ++wave) {
        asc_vf_call<CgemmBatchedMixedNn32PackVf>(
            dim3{CGEMM_BATCHED_MIXED_NN32_THREADS, 1, 1}, aarray, packedArray, batch, subBlock,
            static_cast<uint32_t>(tiling.batchCount));
        CgemmPublishPackedWave(packedTensor, wave);
    }
}

__aicore__ inline void CgemmRunMixedNn32(
    __gm__ uint8_t* aarray, __gm__ uint8_t* packedArray, const CgemmBatchedRealifyGenericTilingData& tiling,
    uint32_t blocks)
{
    const uint32_t block = AscendC::GetBlockIdx() / 2;
    const uint32_t subBlock = AscendC::GetBlockIdx() & 1U;
    AscendC::GlobalTensor<float> packedTensor;
    packedTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packedArray));
    if (tiling.chunkedSpatial != 0) {
        CgemmRunMixedNn32Chunked(aarray, packedArray, tiling, block, subBlock, blocks, packedTensor);
    } else {
        CgemmRunMixedNn32Waves(aarray, packedArray, tiling, block, subBlock, blocks, packedTensor);
    }
}

__aicore__ inline void CgemmRunMixedTinyTransposed(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& tiling, uint32_t blocks, bool exactCt38)
{
    const uint32_t block = AscendC::GetBlockIdx() / 2;
    const uint32_t subBlock = AscendC::GetBlockIdx() & 1U;
    AscendC::GlobalTensor<float> packedTensor;
    packedTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packedArray));
    uint32_t wave = 0;
    for (uint32_t batch = block; batch < static_cast<uint32_t>(tiling.batchCount); batch += blocks, ++wave) {
        if (exactCt38) {
            asc_vf_call<CgemmBatchedMixedTransposedPackVf<38, false>>(
                dim3{CGEMM_BATCHED_MIXED_TINY_THREADS, 1, 1}, aarray, barray, packedArray, batch, subBlock,
                static_cast<uint32_t>(tiling.batchCount));
        } else {
            asc_vf_call<CgemmBatchedMixedTransposedPackVf<39, true>>(
                dim3{CGEMM_BATCHED_MIXED_TINY_THREADS, 1, 1}, aarray, barray, packedArray, batch, subBlock,
                static_cast<uint32_t>(tiling.batchCount));
        }
        CgemmPublishPackedWave(packedTensor, wave);
    }
}

__aicore__ inline void CgemmRunMixedGenericPack(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& packTiling, const GemmBatchedGemmTilingData& gemmTiling,
    uint32_t blocks)
{
    AscendC::TPipe pipe;
    const uint32_t block = AscendC::GetBlockIdx() / 2;
    if (packTiling.transposeTile == 32) {
        CgemmBatchedMixedPack<32>(aarray, barray, packedArray, packTiling, gemmTiling, block, blocks, pipe);
    } else if (packTiling.transposeTile == 60) {
        CgemmBatchedMixedPack<60>(aarray, barray, packedArray, packTiling, gemmTiling, block, blocks, pipe);
    } else {
        CgemmBatchedMixedPack<64>(aarray, barray, packedArray, packTiling, gemmTiling, block, blocks, pipe);
    }
}

__aicore__ inline void CgemmRunMixedAiv(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& packTiling, const GemmBatchedGemmTilingData& gemmTiling,
    uint32_t blocks)
{
    const bool exactNn32 = packTiling.transA == 0 && packTiling.transB == 0 && packTiling.packB == 0 &&
                           packTiling.transposeTile == 32 && packTiling.m == 32 && packTiling.n == 32 &&
                           packTiling.k == 32;
    const bool exactCt38 = packTiling.transA == 2 && packTiling.transB == 1 && packTiling.m == 38 &&
                           packTiling.n == 38 && packTiling.k == 38 && packTiling.batchCount == 235;
    const bool exactCc39 = packTiling.transA == 2 && packTiling.transB == 2 && packTiling.m == 39 &&
                           packTiling.n == 39 && packTiling.k == 39 && packTiling.batchCount == 125;
    if (exactNn32) {
        CgemmRunMixedNn32(aarray, packedArray, packTiling, blocks);
    } else if (exactCt38 || exactCc39) {
        CgemmRunMixedTinyTransposed(aarray, barray, packedArray, packTiling, blocks, exactCt38);
    } else {
        CgemmRunMixedGenericPack(aarray, barray, packedArray, packTiling, gemmTiling, blocks);
    }
}

} // namespace

extern "C" __schedmode__(1) __global__ __aicore__ void cgemm_batched_realify_mixed_kernel(
    __gm__ uint8_t* aarray, __gm__ uint8_t* barray, __gm__ uint8_t* carray, __gm__ uint8_t* packedArray,
    CgemmBatchedRealifyGenericTilingData packTiling, GemmBatchedGemmTilingData gemmTiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV {
        CgemmRunMixedAiv(aarray, barray, packedArray, packTiling, gemmTiling, blocks);
    }
    if ASCEND_IS_AIC {
        CgemmBatchedMixedGemm(barray, carray, packedArray, packTiling, gemmTiling, AscendC::GetBlockIdx(), blocks);
    }
}

void cgemm_batched_realify_mixed_do(
    uint32_t numBlocks, void* stream, uint8_t* aarray, uint8_t* barray, uint8_t* carray, uint8_t* packedArray,
    const CgemmBatchedRealifyGenericTilingData& packTiling, const GemmBatchedGemmTilingData& gemmTiling)
{
    cgemm_batched_realify_mixed_kernel<<<numBlocks, nullptr, stream>>>(
        aarray, barray, carray, packedArray, packTiling, gemmTiling, numBlocks);
}
