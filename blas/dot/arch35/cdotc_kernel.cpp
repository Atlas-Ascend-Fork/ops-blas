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

#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/kernel_utils.h"
#include "cdotc_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t ELEMENTS_PER_BLOCK = 8;
constexpr uint32_t SHARED_TMP_ELEMENTS = 256;
constexpr uint32_t SAFETY_MARGIN = 4096;
constexpr uint32_t REGBASE_PARTIAL_FLOATS = 16;
constexpr uint32_t REGBASE_REAL_PARTIAL_OFFSET = 0;
constexpr uint32_t REGBASE_IMAG_PARTIAL_OFFSET = 8;
constexpr uint32_t CBLAS_COMPAT_UNIT_CHUNK_SIZE = 4096;
constexpr uint32_t CBLAS_COMPAT_STRIDED_CHUNK_SIZE = 1024;
constexpr int64_t CBLAS_COMPAT_MAX_DMA_ABS_STRIDE = static_cast<int64_t>(UINT32_MAX) / (2 * sizeof(float)) + 1;
// Below this crossover, sequential accumulation avoids parallel-reduction
// rounding differences without sacrificing large-vector throughput.
constexpr int64_t PARALLEL_MIN_ELEMENT_COUNT = 512 * 1024;
constexpr uint32_t MAX_WORKSPACE_STRIDE =
    (CDOTC_MAX_PARALLEL_CORE_NUM + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
constexpr uint32_t MAX_PREFIX_FLOATS = 2 * MAX_WORKSPACE_STRIDE + ELEMENTS_PER_BLOCK;
constexpr uint32_t AVAILABLE_UB_FLOATS = (UB_SIZE - SAFETY_MARGIN) / sizeof(float);
constexpr uint32_t MIN_REGBASE_BUFFER_FLOATS =
    MAX_PREFIX_FLOATS + REGBASE_PARTIAL_FLOATS + 4 * BUFFER_NUM * (VECTOR_REG_WIDTH / sizeof(float));
static_assert(AVAILABLE_UB_FLOATS >= MIN_REGBASE_BUFFER_FLOATS, "Cdotc RegBase buffers exceed available UB");

// Ordered floating-point policy: scalar compatibility and correction paths use
// volatile to materialize FP32 products, sums, and residuals at their intended
// rounding boundaries. A file-wide contraction/reassociation policy has not
// been verified for every __aicore__ and __simt_vf__ path, so these qualifiers
// must not be removed without full precision and performance regression.

__aicore__ inline uint32_t Align8(uint32_t value)
{
    return (value + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
}

} // namespace

template <bool FIXED_GROUP>
__simd_callee__ inline void CdotcInterleavedGroupVf(
    __ubuf__ float* xAddr, __ubuf__ float* yAddr, Reg::RegTensor<float>& accumulator,
    Reg::RegTensor<int32_t>& signedShiftIndex, Reg::MaskReg maskAll, Reg::MaskReg maskTwo, Reg::MaskReg mask,
    uint32_t count)
{
    Reg::RegTensor<float> xReal;
    Reg::RegTensor<float> xImag;
    Reg::RegTensor<float> yReal;
    Reg::RegTensor<float> yImag;
    Reg::RegTensor<float> realTerm;
    Reg::RegTensor<float> imagTerm;
    Reg::RegTensor<float> scratch;
    Reg::RegTensor<float> packed0;
    Reg::RegTensor<float> packed1;
    const uint32_t activeCount = FIXED_GROUP ? 8U : count;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(xReal, xImag, xAddr);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(yReal, yImag, yAddr);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(realTerm, xReal, yReal, mask);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(scratch, xImag, yImag, mask);
    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(realTerm, realTerm, scratch, mask);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(imagTerm, xReal, yImag, mask);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(scratch, xImag, yReal, mask);
    Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(imagTerm, imagTerm, scratch, mask);
    Reg::Interleave<float>(packed0, packed1, realTerm, imagTerm);
#pragma unroll
    for (uint32_t i = 0; i < activeCount; ++i) {
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(accumulator, accumulator, packed0, maskTwo);
        if (i + 1 < activeCount) {
            Reg::Gather<float, uint32_t>(
                packed0, packed0, reinterpret_cast<Reg::RegTensor<uint32_t>&>(signedShiftIndex));
        }
    }
}

__simd_vf__ inline void CdotcInterleavedOrderedVf(
    __ubuf__ float* xAddr, __ubuf__ float* yAddr, __ubuf__ float* partialAddr, uint32_t count)
{
    constexpr uint32_t GROUP = 8;
    Reg::RegTensor<float> accumulator;
    Reg::RegTensor<float> previousReal;
    Reg::RegTensor<float> previousImag;
    Reg::RegTensor<float> previousPacked0;
    Reg::RegTensor<float> previousPacked1;
    Reg::RegTensor<float> selected;
    Reg::RegTensor<int32_t> signedShiftIndex;
    Reg::RegTensor<uint32_t> imagIndex;
    auto maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t two = 2;
    uint32_t group = GROUP;
    auto maskTwo = Reg::UpdateMask<float>(two);
    auto maskGroup = Reg::UpdateMask<float>(group);
    Reg::Arange<int32_t>(signedShiftIndex, 2);
    Reg::Duplicate<uint32_t>(imagIndex, 1U, maskAll);
    Reg::Duplicate<float>(accumulator, 0.0f, maskAll);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(previousReal, partialAddr + REGBASE_REAL_PARTIAL_OFFSET);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(previousImag, partialAddr + REGBASE_IMAG_PARTIAL_OFFSET);
    Reg::Interleave<float>(previousPacked0, previousPacked1, previousReal, previousImag);
    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(accumulator, accumulator, previousPacked0, maskTwo);

    const uint16_t fullLoopNum = static_cast<uint16_t>(count / GROUP);
    for (uint16_t i = 0; i < fullLoopNum; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * GROUP;
        CdotcInterleavedGroupVf<true>(
            xAddr + 2 * offset, yAddr + 2 * offset, accumulator, signedShiftIndex, maskAll, maskTwo, maskGroup, GROUP);
    }
    const uint32_t tail = count - static_cast<uint32_t>(fullLoopNum) * GROUP;
    if (tail > 0) {
        const uint32_t offset = static_cast<uint32_t>(fullLoopNum) * GROUP;
        uint32_t tailMaskCount = tail;
        auto tailMask = Reg::UpdateMask<float>(tailMaskCount);
        CdotcInterleavedGroupVf<false>(
            xAddr + 2 * offset, yAddr + 2 * offset, accumulator, signedShiftIndex, maskAll, maskTwo, tailMask, tail);
    }
    Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        partialAddr + REGBASE_REAL_PARTIAL_OFFSET, accumulator, maskAll);
    Reg::Gather<float, uint32_t>(selected, accumulator, imagIndex);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        partialAddr + REGBASE_IMAG_PARTIAL_OFFSET, selected, maskAll);
}

template <bool USE_KAHAN>
__simd_callee__ inline void CdotcKahanAccumulateVf(
    Reg::RegTensor<float>& xReal, Reg::RegTensor<float>& xImag, Reg::RegTensor<float>& yReal,
    Reg::RegTensor<float>& yImag, Reg::RegTensor<float>& realAcc, Reg::RegTensor<float>& imagAcc,
    Reg::RegTensor<float>& realComp, Reg::RegTensor<float>& imagComp, Reg::RegTensor<float>& zero, Reg::MaskReg maskAll,
    Reg::MaskReg mask)
{
    Reg::RegTensor<float> term;
    Reg::RegTensor<float> scratch;
    Reg::RegTensor<float> corrected;
    Reg::RegTensor<float> updated;
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(term, xReal, yReal, mask);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(scratch, xImag, yImag, mask);
    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(term, term, scratch, maskAll);
    if constexpr (USE_KAHAN) {
        Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(corrected, term, realComp, maskAll);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(updated, realAcc, corrected, maskAll);
        Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(scratch, updated, realAcc, maskAll);
        Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(realComp, scratch, corrected, maskAll);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(realAcc, updated, zero, maskAll);
    } else {
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(realAcc, realAcc, term, maskAll);
    }
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(term, xReal, yImag, mask);
    Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(scratch, xImag, yReal, mask);
    Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(term, term, scratch, maskAll);
    if constexpr (USE_KAHAN) {
        Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(corrected, term, imagComp, maskAll);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(updated, imagAcc, corrected, maskAll);
        Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(scratch, updated, imagAcc, maskAll);
        Reg::Sub<float, Reg::MaskMergeMode::ZEROING>(imagComp, scratch, corrected, maskAll);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(imagAcc, updated, zero, maskAll);
    } else {
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(imagAcc, imagAcc, term, maskAll);
    }
}

template <bool USE_KAHAN>
__simd_callee__ inline void CdotcKahanChunkVf(
    __ubuf__ float* xAddr, __ubuf__ float* yAddr, Reg::RegTensor<float>& realAcc, Reg::RegTensor<float>& imagAcc,
    Reg::RegTensor<float>& realComp, Reg::RegTensor<float>& imagComp, Reg::RegTensor<float>& zero, Reg::MaskReg maskAll,
    uint32_t count)
{
    Reg::RegTensor<float> xReal;
    Reg::RegTensor<float> xImag;
    Reg::RegTensor<float> yReal;
    Reg::RegTensor<float> yImag;
    Reg::MaskReg mask = maskAll;
    if constexpr (!USE_KAHAN) {
        mask = Reg::UpdateMask<float>(count);
    }
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(xReal, xImag, xAddr);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(yReal, yImag, yAddr);
    CdotcKahanAccumulateVf<USE_KAHAN>(
        xReal, xImag, yReal, yImag, realAcc, imagAcc, realComp, imagComp, zero, maskAll, mask);
}

__simd_callee__ inline void CdotcInitKahanStateVf(
    Reg::RegTensor<float>& zero, Reg::RegTensor<float>& realAcc, Reg::RegTensor<float>& imagAcc,
    Reg::RegTensor<float>& realComp, Reg::RegTensor<float>& imagComp, Reg::MaskReg maskAll)
{
    Reg::Duplicate<float>(zero, 0.0f, maskAll);
    Reg::Duplicate<float>(realAcc, 0.0f, maskAll);
    Reg::Duplicate<float>(imagAcc, 0.0f, maskAll);
    Reg::Duplicate<float>(realComp, 0.0f, maskAll);
    Reg::Duplicate<float>(imagComp, 0.0f, maskAll);
}

__simd_callee__ inline void CdotcFinalizeKahanStateVf(
    __ubuf__ float* partialAddr, Reg::RegTensor<float>& realAcc, Reg::RegTensor<float>& imagAcc,
    Reg::RegTensor<float>& previousReal, Reg::RegTensor<float>& previousImag, Reg::MaskReg maskAll)
{
    Reg::ReduceSum<float>(realAcc, realAcc, maskAll);
    Reg::ReduceSum<float>(imagAcc, imagAcc, maskAll);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(previousReal, partialAddr + REGBASE_REAL_PARTIAL_OFFSET);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(previousImag, partialAddr + REGBASE_IMAG_PARTIAL_OFFSET);
    uint32_t one = 1;
    auto maskOne = Reg::UpdateMask<float>(one);
    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(realAcc, realAcc, previousReal, maskOne);
    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(imagAcc, imagAcc, previousImag, maskOne);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        partialAddr + REGBASE_REAL_PARTIAL_OFFSET, realAcc, maskAll);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        partialAddr + REGBASE_IMAG_PARTIAL_OFFSET, imagAcc, maskAll);
}

template <bool FULL_VECTOR>
__simd_callee__ inline void CdotcGatherKahanChunkVf(
    __ubuf__ float* xAddr, __ubuf__ float* yAddr, Reg::RegTensor<float>& realAcc, Reg::RegTensor<float>& imagAcc,
    Reg::RegTensor<float>& realComp, Reg::RegTensor<float>& imagComp, Reg::RegTensor<float>& zero, Reg::MaskReg maskAll,
    uint32_t offset, uint32_t count, uint32_t totalCount, uint32_t localStride, uint32_t xPositive, uint32_t yPositive)
{
    Reg::RegTensor<float> xReal;
    Reg::RegTensor<float> xImag;
    Reg::RegTensor<float> yReal;
    Reg::RegTensor<float> yImag;
    Reg::RegTensor<int32_t> forwardIndex;
    Reg::RegTensor<int32_t> reverseIndex;
    Reg::RegTensor<int32_t> reverseBase;
    Reg::MaskReg mask = maskAll;
    if constexpr (!FULL_VECTOR) {
        mask = Reg::UpdateMask<float>(count);
    }
    // The compatibility path is also used when either unit stride is negative,
    // so Gather is required to preserve each input's logical traversal direction.
    Reg::Arange<int32_t>(forwardIndex, 0);
    Reg::Muls<int32_t>(forwardIndex, forwardIndex, static_cast<int32_t>(localStride), maskAll);
    if (xPositive) {
        Reg::Gather(
            xReal, xAddr + offset * localStride, reinterpret_cast<Reg::RegTensor<uint32_t>&>(forwardIndex), mask);
        Reg::Gather(
            xImag, xAddr + offset * localStride + 1, reinterpret_cast<Reg::RegTensor<uint32_t>&>(forwardIndex), mask);
    } else {
        Reg::Duplicate<int32_t>(reverseBase, static_cast<int32_t>((totalCount - 1 - offset) * localStride), maskAll);
        Reg::Sub<int32_t, Reg::MaskMergeMode::ZEROING>(reverseIndex, reverseBase, forwardIndex, mask);
        Reg::Gather(xReal, xAddr, reinterpret_cast<Reg::RegTensor<uint32_t>&>(reverseIndex), mask);
        Reg::Gather(xImag, xAddr + 1, reinterpret_cast<Reg::RegTensor<uint32_t>&>(reverseIndex), mask);
    }
    if (yPositive) {
        Reg::Gather(
            yReal, yAddr + offset * localStride, reinterpret_cast<Reg::RegTensor<uint32_t>&>(forwardIndex), mask);
        Reg::Gather(
            yImag, yAddr + offset * localStride + 1, reinterpret_cast<Reg::RegTensor<uint32_t>&>(forwardIndex), mask);
    } else {
        Reg::Duplicate<int32_t>(reverseBase, static_cast<int32_t>((totalCount - 1 - offset) * localStride), maskAll);
        Reg::Sub<int32_t, Reg::MaskMergeMode::ZEROING>(reverseIndex, reverseBase, forwardIndex, mask);
        Reg::Gather(yReal, yAddr, reinterpret_cast<Reg::RegTensor<uint32_t>&>(reverseIndex), mask);
        Reg::Gather(yImag, yAddr + 1, reinterpret_cast<Reg::RegTensor<uint32_t>&>(reverseIndex), mask);
    }
    CdotcKahanAccumulateVf<FULL_VECTOR>(
        xReal, xImag, yReal, yImag, realAcc, imagAcc, realComp, imagComp, zero, maskAll, mask);
}

template <bool GATHER>
__simd_vf__ inline void CdotcKahanVectorVf(
    __ubuf__ float* xAddr, __ubuf__ float* yAddr, __ubuf__ float* partialAddr, uint32_t count, uint32_t localStride,
    uint32_t xPositive, uint32_t yPositive)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    Reg::RegTensor<float> zero;
    Reg::RegTensor<float> realAcc;
    Reg::RegTensor<float> imagAcc;
    Reg::RegTensor<float> realComp;
    Reg::RegTensor<float> imagComp;
    Reg::RegTensor<float> previousReal;
    Reg::RegTensor<float> previousImag;
    auto maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    CdotcInitKahanStateVf(zero, realAcc, imagAcc, realComp, imagComp, maskAll);
    const uint16_t fullLoopNum = static_cast<uint16_t>(count / VL);
    for (uint16_t i = 0; i < fullLoopNum; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * VL;
        if constexpr (GATHER) {
            CdotcGatherKahanChunkVf<true>(
                xAddr, yAddr, realAcc, imagAcc, realComp, imagComp, zero, maskAll, offset, VL, count, localStride,
                xPositive, yPositive);
        } else {
            CdotcKahanChunkVf<true>(
                xAddr + 2 * offset, yAddr + 2 * offset, realAcc, imagAcc, realComp, imagComp, zero, maskAll, VL);
        }
    }
    const uint32_t tail = count - static_cast<uint32_t>(fullLoopNum) * VL;
    if (tail > 0) {
        const uint32_t offset = static_cast<uint32_t>(fullLoopNum) * VL;
        if constexpr (GATHER) {
            CdotcGatherKahanChunkVf<false>(
                xAddr, yAddr, realAcc, imagAcc, realComp, imagComp, zero, maskAll, offset, tail, count, localStride,
                xPositive, yPositive);
        } else {
            CdotcKahanChunkVf<false>(
                xAddr + 2 * offset, yAddr + 2 * offset, realAcc, imagAcc, realComp, imagComp, zero, maskAll, tail);
        }
    }
    CdotcFinalizeKahanStateVf(partialAddr, realAcc, imagAcc, previousReal, previousImag, maskAll);
}

__simd_vf__ inline void CdotcPackPartialVf(__ubuf__ float* partialAddr)
{
    Reg::RegTensor<float> realPartial;
    Reg::RegTensor<float> imagPartial;
    Reg::LoadAlign<float>(realPartial, partialAddr + REGBASE_REAL_PARTIAL_OFFSET);
    Reg::LoadAlign<float>(imagPartial, partialAddr + REGBASE_IMAG_PARTIAL_OFFSET);
    Reg::Store<float>(partialAddr, realPartial, 1);
    Reg::Store<float>(partialAddr + 1, imagPartial, 1);
}

template <bool UNIT_STRIDE>
class CdotcCblasCompatKernel {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR result, CdotcTilingData tiling);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    // The explicit prefetch below can enqueue the next chunk before the
    // current one is dequeued, so the queue depth must cover two entries.
    TQue<QuePosition::VECIN, 2> xQueue_;
    TQue<QuePosition::VECIN, 2> yQueue_;
    TBuf<TPosition::VECCALC> partialBuf_;
    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    GlobalTensor<float> resultGM_;
    int64_t n_;
    int64_t incx_;
    int64_t incy_;
    __aicore__ inline void CopyIn(uint32_t logicalStart, uint32_t count);
    __aicore__ inline void ComputeChunk(LocalTensor<float> xLocal, LocalTensor<float> yLocal, uint32_t count);
};

template <bool UNIT_STRIDE>
__aicore__ inline void CdotcCblasCompatKernel<UNIT_STRIDE>::Init(
    TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR result, CdotcTilingData tiling)
{
    pipe_ = pipe;
    n_ = tiling.n;
    incx_ = tiling.incx;
    incy_ = tiling.incy;
    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    resultGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result), 2);
    constexpr uint32_t localFloats =
        UNIT_STRIDE ? 2 * CBLAS_COMPAT_UNIT_CHUNK_SIZE : ELEMENTS_PER_BLOCK * CBLAS_COMPAT_STRIDED_CHUNK_SIZE;
    // Two input queues with two buffers each keep the next GM->UB transfer in
    // flight while the vector pipeline consumes the current chunk.
    pipe_->InitBuffer(xQueue_, 2, localFloats * sizeof(float));
    pipe_->InitBuffer(yQueue_, 2, localFloats * sizeof(float));
    pipe_->InitBuffer(partialBuf_, REGBASE_PARTIAL_FLOATS * sizeof(float));
}

template <bool UNIT_STRIDE>
__aicore__ inline void CdotcCblasCompatKernel<UNIT_STRIDE>::CopyIn(uint32_t logicalStart, uint32_t count)
{
    LocalTensor<float> xLocal = xQueue_.AllocTensor<float>();
    LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();
    const int64_t incxAbs = incx_ >= 0 ? incx_ : -incx_;
    const int64_t incyAbs = incy_ >= 0 ? incy_ : -incy_;
    const int64_t xStart =
        incx_ > 0 ? static_cast<int64_t>(logicalStart) * incxAbs : (n_ - logicalStart - count) * incxAbs;
    const int64_t yStart =
        incy_ > 0 ? static_cast<int64_t>(logicalStart) * incyAbs : (n_ - logicalStart - count) * incyAbs;
    DataCopyExtParams xCopy;
    DataCopyExtParams yCopy;
    if constexpr (UNIT_STRIDE) {
        xCopy = {1, 2 * count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        yCopy = xCopy;
    } else {
        // Each complex value is one 8-byte block. DataCopyPad expands every
        // destination row to one 32-byte DataBlock, hence ELEMENTS_PER_BLOCK
        // floats per logical complex value in UB.
        xCopy = {
            static_cast<uint16_t>(count), 2 * static_cast<uint32_t>(sizeof(float)),
            static_cast<uint32_t>((incxAbs - 1) * 2 * sizeof(float)), 0, 0};
        yCopy = {
            static_cast<uint16_t>(count), 2 * static_cast<uint32_t>(sizeof(float)),
            static_cast<uint32_t>((incyAbs - 1) * 2 * sizeof(float)), 0, 0};
    }
    const DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    DataCopyPad(xLocal, xGM_[2 * xStart], xCopy, padParams);
    DataCopyPad(yLocal, yGM_[2 * yStart], yCopy, padParams);
    xQueue_.EnQue(xLocal);
    yQueue_.EnQue(yLocal);
}

template <bool UNIT_STRIDE>
__aicore__ inline void CdotcCblasCompatKernel<UNIT_STRIDE>::ComputeChunk(
    LocalTensor<float> xLocal, LocalTensor<float> yLocal, uint32_t count)
{
    constexpr uint32_t localStride = UNIT_STRIDE ? 2 : ELEMENTS_PER_BLOCK;
    LocalTensor<float> partialLocal = partialBuf_.Get<float>();
    __ubuf__ float* xAddr = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
    __ubuf__ float* yAddr = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
    __ubuf__ float* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());
    asc_vf_call<CdotcKahanVectorVf<true>>(
        xAddr, yAddr, partialAddr, count, localStride, static_cast<uint32_t>(incx_ > 0),
        static_cast<uint32_t>(incy_ > 0));
    PipeBarrier<PIPE_V>();
}

template <bool UNIT_STRIDE>
__aicore__ inline void CdotcCblasCompatKernel<UNIT_STRIDE>::Process()
{
    constexpr uint32_t chunkSize = UNIT_STRIDE ? CBLAS_COMPAT_UNIT_CHUNK_SIZE : CBLAS_COMPAT_STRIDED_CHUNK_SIZE;
    LocalTensor<float> partialLocal = partialBuf_.Get<float>();
    Duplicate(partialLocal, 0.0f, REGBASE_PARTIAL_FLOATS);
    PipeBarrier<PIPE_V>();
    uint32_t logicalStart = 0;
    uint32_t remaining = static_cast<uint32_t>(n_);
    uint32_t current = remaining < chunkSize ? remaining : chunkSize;
    CopyIn(logicalStart, current);
    logicalStart += current;
    remaining -= current;

    while (remaining > 0) {
        const uint32_t next = remaining < chunkSize ? remaining : chunkSize;
        CopyIn(logicalStart, next);
        LocalTensor<float> xLocal = xQueue_.DeQue<float>();
        LocalTensor<float> yLocal = yQueue_.DeQue<float>();
        ComputeChunk(xLocal, yLocal, current);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
        logicalStart += next;
        remaining -= next;
        current = next;
    }

    LocalTensor<float> xLocal = xQueue_.DeQue<float>();
    LocalTensor<float> yLocal = yQueue_.DeQue<float>();
    ComputeChunk(xLocal, yLocal, current);
    xQueue_.FreeTensor(xLocal);
    yQueue_.FreeTensor(yLocal);

    __ubuf__ float* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());
    asc_vf_call<CdotcPackPartialVf>(partialAddr);
    PipeBarrier<PIPE_V>();
    event_t vToMte3Event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3Event);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Event);
    const DataCopyExtParams oneComplex{1, 2 * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(resultGM_, partialLocal, oneComplex);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void CdotcExtremeStrideVf(
    int64_t n, int64_t incx, int64_t incy, __gm__ const float* xGM, __gm__ const float* yGM, __gm__ float* resultGM)
{
    const int64_t incxAbs = incx >= 0 ? incx : -incx;
    const int64_t incyAbs = incy >= 0 ? incy : -incy;
    // Extreme-stride fallback preserves scalar multiply/add and accumulation order.
    volatile float real = 0.0f;
    volatile float imag = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t xIndex = incx > 0 ? i * incxAbs : (n - 1 - i) * incxAbs;
        const int64_t yIndex = incy > 0 ? i * incyAbs : (n - 1 - i) * incyAbs;
        const float xr = xGM[2 * xIndex];
        const float xi = xGM[2 * xIndex + 1];
        const float yr = yGM[2 * yIndex];
        const float yi = yGM[2 * yIndex + 1];
        volatile float realProduct0 = xr * yr;
        volatile float realProduct1 = xi * yi;
        volatile float imagProduct0 = xr * yi;
        volatile float imagProduct1 = xi * yr;
        volatile float realTerm = realProduct0 + realProduct1;
        volatile float imagTerm = imagProduct0 - imagProduct1;
        real = real + realTerm;
        imag = imag + imagTerm;
    }
    resultGM[0] = real;
    resultGM[1] = imag;
}

class CdotcRegBaseKernel {
public:
    __aicore__ inline void Init(
        TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, CdotcTilingData tiling);
    __aicore__ inline void ProcessInitial();
    __aicore__ inline void ProcessReplay();

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> yQueue_;
    TBuf<TPosition::VECCALC> partialBuf_;
    TBuf<TPosition::VECCALC> prefixBuf_;
    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    GlobalTensor<float> workspaceGM_;
    GlobalTensor<float> resultGM_;
    int64_t n_;
    bool positive_;
    uint32_t useCoreNum_;
    uint32_t workspaceStride_;
    uint32_t coreIdx_;
    uint32_t startIdx_;
    uint32_t calCount_;
    uint32_t chunkSize_;
    // Materialize block-boundary values between the initial and replay passes.
    volatile float initialReal_;
    volatile float initialImag_;
    volatile float approximateReal_;
    volatile float approximateImag_;

    __aicore__ inline void ComputeTile(
        LocalTensor<float> xLocal, LocalTensor<float> yLocal, uint32_t count, bool ordered);
    __aicore__ inline void ProcessPass(bool ordered);
    __aicore__ inline void StorePartialSum();
    __aicore__ inline void LoadBlockBoundary();
    __aicore__ inline void StoreResidual();
};

__aicore__ inline void CdotcRegBaseKernel::Init(
    TPipe* pipe, GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, CdotcTilingData tiling)
{
    pipe_ = pipe;
    n_ = tiling.n;
    positive_ = tiling.incx > 0;
    useCoreNum_ = tiling.useCoreNum;
    coreIdx_ = GetBlockIdx();
    const uint32_t n = static_cast<uint32_t>(n_);
    const uint32_t baseCount = n / useCoreNum_;
    const uint32_t remain = n % useCoreNum_;
    startIdx_ = coreIdx_ * baseCount + (coreIdx_ < remain ? coreIdx_ : remain);
    calCount_ = baseCount + (coreIdx_ < remain ? 1 : 0);
    workspaceStride_ = Align8(useCoreNum_);

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    workspaceGM_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workSpace), static_cast<int64_t>(2) * workspaceStride_);
    resultGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result), 2);

    const uint32_t prefixFloats = 2 * workspaceStride_ + ELEMENTS_PER_BLOCK;
    const uint32_t fixedFloats = prefixFloats + REGBASE_PARTIAL_FLOATS;
    const uint32_t availableFloats = AVAILABLE_UB_FLOATS > fixedFloats ? AVAILABLE_UB_FLOATS - fixedFloats : 0;
    const uint32_t rawChunk = availableFloats / (4 * BUFFER_NUM);
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    chunkSize_ = rawChunk / VL * VL;
    if (chunkSize_ == 0) {
        // The shared core cap and the static UB bound make this unreachable for
        // valid tiling data; keep a nonzero step as a defensive loop guard.
        chunkSize_ = VL;
    }
    pipe_->InitBuffer(xQueue_, BUFFER_NUM, 2 * chunkSize_ * sizeof(float));
    pipe_->InitBuffer(yQueue_, BUFFER_NUM, 2 * chunkSize_ * sizeof(float));
    pipe_->InitBuffer(partialBuf_, REGBASE_PARTIAL_FLOATS * sizeof(float));
    pipe_->InitBuffer(prefixBuf_, prefixFloats * sizeof(float));
}

__aicore__ inline void CdotcRegBaseKernel::ComputeTile(
    LocalTensor<float> xLocal, LocalTensor<float> yLocal, uint32_t count, bool ordered)
{
    __ubuf__ float* xAddr = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
    __ubuf__ float* yAddr = reinterpret_cast<__ubuf__ float*>(yLocal.GetPhyAddr());
    LocalTensor<float> partialLocal = partialBuf_.Get<float>();
    __ubuf__ float* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());
    if (ordered) {
        asc_vf_call<CdotcInterleavedOrderedVf>(xAddr, yAddr, partialAddr, count);
    } else {
        asc_vf_call<CdotcKahanVectorVf<false>>(xAddr, yAddr, partialAddr, count, 2, 1, 1);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void CdotcRegBaseKernel::StorePartialSum()
{
    LocalTensor<float> partialLocal = partialBuf_.Get<float>();
    __ubuf__ float* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());
    asc_vf_call<CdotcPackPartialVf>(partialAddr);
    PipeBarrier<PIPE_V>();
    event_t vToMte3Event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3Event);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Event);
    const DataCopyExtParams oneComplex{1, 2 * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(workspaceGM_[static_cast<int64_t>(2) * coreIdx_], partialLocal, oneComplex);
}

__aicore__ inline void CdotcRegBaseKernel::ProcessPass(bool ordered)
{
    if (calCount_ == 0) {
        return;
    }
    uint32_t remaining = calCount_;
    uint32_t offset = positive_ ? startIdx_ : static_cast<uint32_t>(n_ - startIdx_ - calCount_);
    uint32_t current = remaining < chunkSize_ ? remaining : chunkSize_;
    DataCopyExtParams copyParams{1, 2 * current * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    const DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    LocalTensor<float> xLocal = xQueue_.AllocTensor<float>();
    LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();
    int64_t floatOffset = static_cast<int64_t>(2) * offset;
    DataCopyPad(xLocal, xGM_[floatOffset], copyParams, padParams);
    DataCopyPad(yLocal, yGM_[floatOffset], copyParams, padParams);
    xQueue_.EnQue(xLocal);
    yQueue_.EnQue(yLocal);
    offset += current;
    remaining -= current;

    while (remaining > 0) {
        const uint32_t next = remaining < chunkSize_ ? remaining : chunkSize_;
        copyParams = {1, 2 * next * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        LocalTensor<float> xNext = xQueue_.AllocTensor<float>();
        LocalTensor<float> yNext = yQueue_.AllocTensor<float>();
        floatOffset = static_cast<int64_t>(2) * offset;
        DataCopyPad(xNext, xGM_[floatOffset], copyParams, padParams);
        DataCopyPad(yNext, yGM_[floatOffset], copyParams, padParams);
        xQueue_.EnQue(xNext);
        yQueue_.EnQue(yNext);

        xLocal = xQueue_.DeQue<float>();
        yLocal = yQueue_.DeQue<float>();
        ComputeTile(xLocal, yLocal, current, ordered);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
        current = next;
        offset += next;
        remaining -= next;
    }

    xLocal = xQueue_.DeQue<float>();
    yLocal = yQueue_.DeQue<float>();
    ComputeTile(xLocal, yLocal, current, ordered);
    xQueue_.FreeTensor(xLocal);
    yQueue_.FreeTensor(yLocal);
}

__aicore__ inline void CdotcRegBaseKernel::LoadBlockBoundary()
{
    LocalTensor<float> prefixLocal = prefixBuf_.Get<float>();
    const uint32_t boundaryCount = coreIdx_ == 0 ? 1 : 2;
    const uint32_t boundaryStart = coreIdx_ == 0 ? 0 : coreIdx_ - 1;
    const DataCopyExtParams boundaryCopy{1, 2 * boundaryCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    const DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(prefixLocal, workspaceGM_[2 * boundaryStart], boundaryCopy, padParams);
    event_t mte2ToSEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(mte2ToSEvent);
    WaitFlag<HardEvent::MTE2_S>(mte2ToSEvent);

    if (coreIdx_ == 0) {
        initialReal_ = 0.0f;
        initialImag_ = 0.0f;
        approximateReal_ = prefixLocal.GetValue(0);
        approximateImag_ = prefixLocal.GetValue(1);
    } else {
        initialReal_ = prefixLocal.GetValue(0);
        initialImag_ = prefixLocal.GetValue(1);
        approximateReal_ = prefixLocal.GetValue(2);
        approximateImag_ = prefixLocal.GetValue(3);
    }
}

__aicore__ inline void CdotcRegBaseKernel::StoreResidual()
{
    LocalTensor<float> partialLocal = partialBuf_.Get<float>();
    __ubuf__ float* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());
    asc_vf_call<CdotcPackPartialVf>(partialAddr);
    PipeBarrier<PIPE_V>();
    event_t vToSEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(vToSEvent);
    WaitFlag<HardEvent::V_S>(vToSEvent);
    // Materialize each residual before it is stored for the correction pass.
    volatile float residualReal = partialLocal.GetValue(0) - approximateReal_;
    volatile float residualImag = partialLocal.GetValue(1) - approximateImag_;
    partialLocal.SetValue(0, residualReal);
    partialLocal.SetValue(1, residualImag);
    event_t sToMte3Event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(sToMte3Event);
    WaitFlag<HardEvent::S_MTE3>(sToMte3Event);
    const DataCopyExtParams oneComplex{1, 2 * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(workspaceGM_[static_cast<int64_t>(2) * coreIdx_], partialLocal, oneComplex);
}

__aicore__ inline void CdotcRegBaseKernel::ProcessInitial()
{
    LocalTensor<float> partialLocal = partialBuf_.Get<float>();
    Duplicate(partialLocal, 0.0f, REGBASE_PARTIAL_FLOATS);
    PipeBarrier<PIPE_V>();
    ProcessPass(false);
    StorePartialSum();
}

__aicore__ inline void CdotcRegBaseKernel::ProcessReplay()
{
    LoadBlockBoundary();
    LocalTensor<float> partialLocal = partialBuf_.Get<float>();
    partialLocal.SetValue(REGBASE_REAL_PARTIAL_OFFSET, initialReal_);
    partialLocal.SetValue(REGBASE_IMAG_PARTIAL_OFFSET, initialImag_);
    event_t sToVEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(sToVEvent);
    WaitFlag<HardEvent::S_V>(sToVEvent);
    ProcessPass(true);
    StoreResidual();
}

class CdotcPackedReduceKernel {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR workSpace, GM_ADDR result, uint32_t useCoreNum);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<float> workspaceGM_;
    GlobalTensor<float> resultGM_;
    uint32_t useCoreNum_;
    uint32_t workspaceStride_;
};

class CdotcOrderedReduceKernel {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR workSpace, GM_ADDR result, uint32_t useCoreNum)
    {
        pipe_ = pipe;
        useCoreNum_ = useCoreNum;
        outputOffset_ = Align8(2 * useCoreNum_);
        workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), static_cast<int64_t>(2) * useCoreNum_);
        resultGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result), 2);
        pipe_->InitBuffer(localBuf_, (outputOffset_ + ELEMENTS_PER_BLOCK) * static_cast<uint32_t>(sizeof(float)));
    }

    __aicore__ inline void BuildPrefix()
    {
        LocalTensor<float> local = localBuf_.Get<float>();
        const DataCopyExtParams allPartials{1, 2 * useCoreNum_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(local, workspaceGM_, allPartials, padParams);

        event_t mte2ToSEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ToSEvent);
        WaitFlag<HardEvent::MTE2_S>(mte2ToSEvent);

        float real = 0.0f;
        float imag = 0.0f;
        uint32_t core = 0;
        for (; core + 3 < useCoreNum_; core += 4) {
            real = real + local.GetValue(2 * core);
            imag = imag + local.GetValue(2 * core + 1);
            local.SetValue(2 * core, real);
            local.SetValue(2 * core + 1, imag);

            real = real + local.GetValue(2 * (core + 1));
            imag = imag + local.GetValue(2 * (core + 1) + 1);
            local.SetValue(2 * (core + 1), real);
            local.SetValue(2 * (core + 1) + 1, imag);

            real = real + local.GetValue(2 * (core + 2));
            imag = imag + local.GetValue(2 * (core + 2) + 1);
            local.SetValue(2 * (core + 2), real);
            local.SetValue(2 * (core + 2) + 1, imag);

            real = real + local.GetValue(2 * (core + 3));
            imag = imag + local.GetValue(2 * (core + 3) + 1);
            local.SetValue(2 * (core + 3), real);
            local.SetValue(2 * (core + 3) + 1, imag);
        }
        for (; core < useCoreNum_; ++core) {
            real = real + local.GetValue(2 * core);
            imag = imag + local.GetValue(2 * core + 1);
            local.SetValue(2 * core, real);
            local.SetValue(2 * core + 1, imag);
        }
        local.SetValue(outputOffset_, real);
        local.SetValue(outputOffset_ + 1, imag);

        event_t sToMte3Event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(sToMte3Event);
        WaitFlag<HardEvent::S_MTE3>(sToMte3Event);
        DataCopyPad(workspaceGM_, local, allPartials);
        const DataCopyExtParams oneComplex{1, 2 * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(resultGM_, local[outputOffset_], oneComplex);
    }

    __aicore__ inline void FinishResidual()
    {
        LocalTensor<float> local = localBuf_.Get<float>();
        const DataCopyExtParams allResiduals{1, 2 * useCoreNum_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const DataCopyExtParams oneComplex{1, 2 * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(local, workspaceGM_, allResiduals, padParams);
        DataCopyPad(local[outputOffset_], resultGM_, oneComplex, padParams);

        event_t mte2ToSEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ToSEvent);
        WaitFlag<HardEvent::MTE2_S>(mte2ToSEvent);

        float correctionReal = 0.0f;
        float correctionImag = 0.0f;
        uint32_t core = 0;
        for (; core + 3 < useCoreNum_; core += 4) {
            correctionReal = correctionReal + local.GetValue(2 * core);
            correctionImag = correctionImag + local.GetValue(2 * core + 1);

            correctionReal = correctionReal + local.GetValue(2 * (core + 1));
            correctionImag = correctionImag + local.GetValue(2 * (core + 1) + 1);

            correctionReal = correctionReal + local.GetValue(2 * (core + 2));
            correctionImag = correctionImag + local.GetValue(2 * (core + 2) + 1);

            correctionReal = correctionReal + local.GetValue(2 * (core + 3));
            correctionImag = correctionImag + local.GetValue(2 * (core + 3) + 1);
        }
        for (; core < useCoreNum_; ++core) {
            correctionReal = correctionReal + local.GetValue(2 * core);
            correctionImag = correctionImag + local.GetValue(2 * core + 1);
        }
        local.SetValue(outputOffset_, local.GetValue(outputOffset_) + correctionReal);
        local.SetValue(outputOffset_ + 1, local.GetValue(outputOffset_ + 1) + correctionImag);

        event_t sToMte3Event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(sToMte3Event);
        WaitFlag<HardEvent::S_MTE3>(sToMte3Event);
        DataCopyPad(resultGM_, local[outputOffset_], oneComplex);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> localBuf_;
    GlobalTensor<float> workspaceGM_;
    GlobalTensor<float> resultGM_;
    uint32_t useCoreNum_;
    uint32_t outputOffset_;
};

__aicore__ inline void CdotcPackedReduceKernel::Init(
    TPipe* pipe, GM_ADDR workSpace, GM_ADDR result, uint32_t useCoreNum)
{
    pipe_ = pipe;
    useCoreNum_ = useCoreNum;
    workspaceStride_ = Align8(useCoreNum_);
    workspaceGM_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workSpace), static_cast<int64_t>(2) * workspaceStride_);
    resultGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result), 2);
    pipe_->InitBuffer(tmpBuf_, (4 * workspaceStride_ + SHARED_TMP_ELEMENTS) * sizeof(float));
}

__aicore__ inline void CdotcPackedReduceKernel::Process()
{
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> packed = tmp;
    LocalTensor<float> realLocal = tmp[2 * workspaceStride_];
    LocalTensor<float> imagLocal = tmp[3 * workspaceStride_];
    const uint32_t copyBytes = 2 * useCoreNum_ * static_cast<uint32_t>(sizeof(float));
    const DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
    const DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(packed, workspaceGM_[0], copyParams, padParams);

    event_t mte2ToVEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(mte2ToVEvent);
    WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent);

    DeInterleave(realLocal, imagLocal, packed, static_cast<int32_t>(2 * useCoreNum_));
    PipeBarrier<PIPE_V>();
    LocalTensor<float> sharedTmp = tmp[4 * workspaceStride_];
    ReduceSum(realLocal, realLocal, sharedTmp, static_cast<int32_t>(useCoreNum_));
    PipeBarrier<PIPE_V>();
    ReduceSum(imagLocal, imagLocal, sharedTmp, static_cast<int32_t>(useCoreNum_));
    PipeBarrier<PIPE_V>();

    event_t vToMte3Event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3Event);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Event);
    const DataCopyExtParams oneFloat{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(resultGM_[0], realLocal, oneFloat);
    DataCopyPad(resultGM_[1], imagLocal, oneFloat);
}

__global__ __aicore__ void CdotcPartialKernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, CdotcTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CdotcRegBaseKernel op;
    op.Init(&pipe, x, y, result, workSpace, tiling);
    op.ProcessInitial();
}

__global__ __aicore__ void CdotcReplayKernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, CdotcTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CdotcRegBaseKernel op;
    op.Init(&pipe, x, y, result, workSpace, tiling);
    op.ProcessReplay();
}

__global__ __aicore__ void CdotcBuildPrefixKernelEntry(GM_ADDR workSpace, GM_ADDR result, uint32_t useCoreNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CdotcOrderedReduceKernel op;
    op.Init(&pipe, workSpace, result, useCoreNum);
    op.BuildPrefix();
}

__global__ __aicore__ void CdotcFinishResidualKernelEntry(GM_ADDR workSpace, GM_ADDR result, uint32_t useCoreNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CdotcOrderedReduceKernel op;
    op.Init(&pipe, workSpace, result, useCoreNum);
    op.FinishResidual();
}

__global__ __aicore__ void CdotcPackedReduceKernelEntry(GM_ADDR workSpace, GM_ADDR result, uint32_t useCoreNum)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CdotcPackedReduceKernel op;
    op.Init(&pipe, workSpace, result, useCoreNum);
    op.Process();
}

__global__ __aicore__ void CdotcCblasCompatUnitKernelEntry(GM_ADDR x, GM_ADDR y, GM_ADDR result, CdotcTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CdotcCblasCompatKernel<true> op;
    op.Init(&pipe, x, y, result, tiling);
    op.Process();
}

__global__ __aicore__ void CdotcCblasCompatStridedKernelEntry(
    GM_ADDR x, GM_ADDR y, GM_ADDR result, CdotcTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CdotcCblasCompatKernel<false> op;
    op.Init(&pipe, x, y, result, tiling);
    op.Process();
}

__global__ __aicore__ void CdotcExtremeStrideKernelEntry(GM_ADDR x, GM_ADDR y, GM_ADDR result, CdotcTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<CdotcExtremeStrideVf>(
        dim3{1, 1, 1}, tiling.n, tiling.incx, tiling.incy, reinterpret_cast<__gm__ const float*>(x),
        reinterpret_cast<__gm__ const float*>(y), reinterpret_cast<__gm__ float*>(result));
}

void CdotcKernelDo(
    GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, uint32_t numBlocks, const CdotcTilingData& tiling,
    void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    const int64_t incxAbs = tiling.incx >= 0 ? tiling.incx : -tiling.incx;
    const int64_t incyAbs = tiling.incy >= 0 ? tiling.incy : -tiling.incy;
    if (incxAbs > CBLAS_COMPAT_MAX_DMA_ABS_STRIDE || incyAbs > CBLAS_COMPAT_MAX_DMA_ABS_STRIDE) {
        CdotcExtremeStrideKernelEntry<<<1, nullptr, aclStream>>>(x, y, result, tiling);
        return;
    }
    const bool positiveUnitStride = tiling.incx == 1 && tiling.incy == 1;
    const bool useSequentialPath = !positiveUnitStride;
    if (useSequentialPath) {
        if (incxAbs == 1 && incyAbs == 1) {
            CdotcCblasCompatUnitKernelEntry<<<1, nullptr, aclStream>>>(x, y, result, tiling);
        } else {
            CdotcCblasCompatStridedKernelEntry<<<1, nullptr, aclStream>>>(x, y, result, tiling);
        }
        return;
    }
    const uint64_t elementCount = static_cast<uint64_t>(tiling.n);
    const bool isPowerOfTwo = elementCount != 0 && (elementCount & (elementCount - 1)) == 0;
    CdotcTilingData launchTiling = tiling;
    const uint32_t launchBlocks = numBlocks < CDOTC_MAX_PARALLEL_CORE_NUM ? numBlocks : CDOTC_MAX_PARALLEL_CORE_NUM;
    launchTiling.useCoreNum = launchBlocks;
    if (tiling.n >= PARALLEL_MIN_ELEMENT_COUNT || !isPowerOfTwo) {
        CdotcPartialKernel<<<launchBlocks, nullptr, aclStream>>>(x, y, result, workSpace, launchTiling);
        CdotcPackedReduceKernelEntry<<<1, nullptr, aclStream>>>(workSpace, result, launchTiling.useCoreNum);
        return;
    }
    CdotcPartialKernel<<<launchBlocks, nullptr, aclStream>>>(x, y, result, workSpace, launchTiling);
    CdotcBuildPrefixKernelEntry<<<1, nullptr, aclStream>>>(workSpace, result, launchTiling.useCoreNum);
    CdotcReplayKernel<<<launchBlocks, nullptr, aclStream>>>(x, y, result, workSpace, launchTiling);
    CdotcFinishResidualKernelEntry<<<1, nullptr, aclStream>>>(workSpace, result, launchTiling.useCoreNum);
}
