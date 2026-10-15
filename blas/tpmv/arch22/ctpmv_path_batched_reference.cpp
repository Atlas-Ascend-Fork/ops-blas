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
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "ctpmv_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t FLOATS_PER_COMPLEX = 2U;
constexpr uint32_t BYTES_PER_FLOAT = 4U;
constexpr uint32_t BYTES_PER_COMPLEX = FLOATS_PER_COMPLEX * BYTES_PER_FLOAT;

// GatherMask patterns 1 and 2 select the real and imaginary lanes of FP32 AoS data.
constexpr uint8_t PATTERN_EVEN = 1U;
constexpr uint8_t PATTERN_ODD = 2U;

// Deinterleave whole 256-byte repeats; round lengths up to ELEMS_PER_REPEAT.
constexpr uint32_t ELEMS_PER_REPEAT = 64U;
constexpr uint16_t REPEAT_BLOCKS = 8U;

// Complex elements processed per step of the in-place path. The supported maximum fits one slab.
constexpr uint32_t APPLY_CHUNK = CTPMV_MAX_N;

// Stage consecutive packed columns in one slab, then synchronize and deinterleave once.
constexpr uint32_t SLAB_ELEMS = CTPMV_MAX_N;
constexpr uint32_t SLAB_MAX_COLS = 128U;

// Planar vector operands start at multiples of this many floats for 32-byte alignment.
constexpr uint32_t ALIGN_ELEMS = 8U;
constexpr uint32_t ALIGN_MASK = ALIGN_ELEMS - 1U;

// Slice one TBuf so Gather byte offsets share a base.
// Pad each full-size plane by one repeat to contain rounded deinterleave writes.
constexpr uint32_t PLANE = CTPMV_MAX_N + ELEMS_PER_REPEAT;

// Each complex reduction uses two 32-byte result blocks per column.
constexpr uint32_t RED_SECOND = 8U;
constexpr uint32_t RED_SLOT_FLOATS = 2U * RED_SECOND;
constexpr uint32_t RED_SLOTS = SLAB_MAX_COLS * RED_SLOT_FLOATS;
constexpr uint32_t OFF_X_RE = 0U;
constexpr uint32_t OFF_X_IM = OFF_X_RE + PLANE;
constexpr uint32_t OFF_ACC_RE = OFF_X_IM + PLANE;
constexpr uint32_t OFF_ACC_IM = OFF_ACC_RE + PLANE;
constexpr uint32_t OFF_A_RE = OFF_ACC_IM + PLANE;
constexpr uint32_t OFF_A_IM = OFF_A_RE + PLANE;
constexpr uint32_t OFF_AOS = OFF_A_IM + PLANE;
constexpr uint32_t OFF_PROD_RE = OFF_AOS + FLOATS_PER_COMPLEX * PLANE;
constexpr uint32_t OFF_PROD_IM = OFF_PROD_RE + APPLY_CHUNK;
constexpr uint32_t OFF_TMP = OFF_PROD_IM + APPLY_CHUNK;
constexpr uint32_t OFF_RED = OFF_TMP + APPLY_CHUNK;
// Reduction scratch occupies the padding after the full AoS slab, before the product planes.
constexpr uint32_t DOT_REDUCE_SCRATCH_OFFSET = 2U * SLAB_ELEMS;
constexpr uint32_t UB_FLOATS = OFF_RED + RED_SLOTS;
constexpr uint32_t UB_BYTES = UB_FLOATS * BYTES_PER_FLOAT + SLAB_MAX_COLS * sizeof(uint32_t);
static_assert(UB_BYTES <= 192U * 1024U, "Ctpmv UB must fit arch22 AIV");

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

__aicore__ inline uint32_t CeilAlignU32(uint32_t v) { return (v + ALIGN_MASK) & ~ALIGN_MASK; }

// Match the reference diagonal product: round the addend, then fuse the second product.
__aicore__ inline void CtpmvTensorFmaCompat(
    LocalTensor<float> dst, LocalTensor<float> lhs, LocalTensor<float> rhs, LocalTensor<float> addend, uint32_t count)
{
    if (count == 0U) {
        return;
    }
    const int32_t vectorCount = static_cast<int32_t>(count);
    // Copy the independently rounded addend before MulAddDst computes lhs * rhs + dst.
    Muls(dst, addend, 1.0F, vectorCount);
    PipeBarrier<PIPE_V>();
    MulAddDst<float, float>(dst, lhs, rhs, vectorCount);
    PipeBarrier<PIPE_V>();
}

// AoS DMA destinations align to 32 bytes, or four complex elements.
constexpr uint32_t DMA_ELEMS = 4U;
constexpr uint32_t DMA_MASK = DMA_ELEMS - 1U;

} // namespace

class CtpmvAIV {
public:
    __aicore__ inline CtpmvAIV() = default;
    __aicore__ inline void Init(GM_ADDR aPacked, GM_ADDR x, const CtpmvTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOptions(const CtpmvTilingData& tiling);
    __aicore__ inline void InitGlobalBuffers(GM_ADDR aPacked, GM_ADDR x);
    __aicore__ inline void InitLocalBuffers(TPipe* pipe);
    __aicore__ inline void InitEvents(TPipe* pipe);
    __aicore__ inline uint64_t ColumnBase(uint32_t j) const;
    __aicore__ inline uint64_t ColumnApOffset(uint32_t j) const;
    __aicore__ inline uint32_t ColumnFirstRow(uint32_t j) const;
    __aicore__ inline uint32_t ColumnLen(uint32_t j) const;
    __aicore__ inline void SplitPlanes(
        const LocalTensor<float>& re, const LocalTensor<float>& im, const LocalTensor<float>& src, uint32_t count);
    __aicore__ inline uint64_t CostPrefix(uint32_t j) const;
    __aicore__ inline uint32_t SplitAt(uint64_t target) const;
    __aicore__ inline void ClaimColumns();
    __aicore__ inline void LoadX();
    __aicore__ inline void InterleaveRange(uint32_t lo, uint32_t hi);
    __aicore__ inline void StoreX();
    __aicore__ inline void RunColumns();
    __aicore__ inline void ZeroAccumulators();
    __aicore__ inline void InitNonUnitDiagonal();
    __aicore__ inline void AddUnitDiagonal();
    __aicore__ inline void ProductDotChunk(uint32_t slab, uint32_t row, uint32_t cnt);
    __aicore__ inline void ReduceDotColumn(uint32_t k, uint32_t cnt);
    __aicore__ inline void PublishDotColumns(uint32_t first, uint32_t cols);
    __aicore__ inline uint32_t PlanBatch(uint32_t first, uint32_t& used);
    __aicore__ inline void StageBatchColumn(uint32_t j, uint32_t slab);
    __aicore__ inline void LoadBatch(uint32_t first, uint32_t cols, uint32_t used);
    __aicore__ inline void RunBatchColumn(uint32_t j, uint32_t slab, uint32_t k);
    __aicore__ inline void RunBatch(uint32_t first, uint32_t cols, uint32_t used, bool lastBatch);

    GlobalTensor<float> apGm_;
    GlobalTensor<float> xGm_;
    TBuf<TPosition::VECCALC> ub_;
    LocalTensor<float> base_;
    LocalTensor<float> xRe_;
    LocalTensor<float> xIm_;
    LocalTensor<float> accRe_;
    LocalTensor<float> accIm_;
    LocalTensor<float> aRe_;
    LocalTensor<float> aIm_;
    LocalTensor<float> prodRe_;
    LocalTensor<float> prodIm_;
    LocalTensor<float> tmp_;
    LocalTensor<float> red_;
    LocalTensor<float> aos_;
    LocalTensor<uint32_t> idx_;
    // Store each column's aligned slab offset from the batch plan.
    uint32_t slabOff_[SLAB_MAX_COLS] = {0U};
    uint32_t n_ = 0U;
    uint32_t numCores_ = 1U;
    uint32_t coreIdx_ = 0U;
    uint32_t colLo_ = 0U;
    uint32_t colHi_ = 0U;
    bool isUpper_ = true;
    bool isUnit_ = false;
    bool isConj_ = false;
    event_t evtMte2V_;
    event_t evtVMte2_;
    event_t evtVS_;
    event_t evtSV_;
    event_t evtVMte3_;
};

__aicore__ inline void CtpmvAIV::InitOptions(const CtpmvTilingData& tiling)
{
    n_ = tiling.n;
    numCores_ = static_cast<uint32_t>(GetBlockNum());
    coreIdx_ = static_cast<uint32_t>(GetBlockIdx());
    isUpper_ = tiling.uplo == ACLBLAS_UPPER;
    isUnit_ = tiling.diag == ACLBLAS_UNIT;
    isConj_ = tiling.trans == ACLBLAS_OP_C;
}

__aicore__ inline void CtpmvAIV::InitGlobalBuffers(GM_ADDR aPacked, GM_ADDR x)
{
    const uint64_t nn = n_;
    const uint64_t apFloats = nn * (nn + 1U);
    apGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aPacked), apFloats);
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), nn * FLOATS_PER_COMPLEX);
}

__aicore__ inline void CtpmvAIV::InitLocalBuffers(TPipe* pipe)
{
    pipe->InitBuffer(ub_, UB_BYTES);
    base_ = ub_.Get<float>();
    aRe_ = base_[OFF_A_RE];
    aIm_ = base_[OFF_A_IM];
    prodRe_ = base_[OFF_PROD_RE];
    prodIm_ = base_[OFF_PROD_IM];
    tmp_ = base_[OFF_TMP];
    red_ = base_[OFF_RED];
    accRe_ = base_[OFF_ACC_RE];
    accIm_ = base_[OFF_ACC_IM];
    xRe_ = base_[OFF_X_RE];
    xIm_ = base_[OFF_X_IM];
    aos_ = base_[OFF_AOS];
    idx_ = ub_.Get<uint32_t>()[UB_FLOATS];
}

__aicore__ inline void CtpmvAIV::InitEvents(TPipe* pipe)
{
    evtMte2V_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
    evtVMte2_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
    evtVS_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_S>());
    evtSV_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_V>());
    evtVMte3_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
}

__aicore__ inline void CtpmvAIV::Init(GM_ADDR aPacked, GM_ADDR x, const CtpmvTilingData& tiling, TPipe* pipe)
{
    InitOptions(tiling);
    InitGlobalBuffers(aPacked, x);
    InitLocalBuffers(pipe);
    InitEvents(pipe);
}

// Column-major packed offset: UPPER stores rows 0..j; LOWER stores rows j..n-1.
__aicore__ inline uint64_t CtpmvAIV::ColumnBase(uint32_t j) const
{
    const uint64_t jj = j;
    if (isUpper_) {
        return (jj * (jj + 1UL)) >> 1UL;
    }
    return jj + (((2UL * n_ - jj - 1UL) * jj) >> 1UL);
}

// First row of the strictly triangular part; the diagonal is handled separately.
__aicore__ inline uint32_t CtpmvAIV::ColumnFirstRow(uint32_t j) const
{
    if (isUpper_) {
        return 0U;
    }
    return j + 1U;
}

__aicore__ inline uint32_t CtpmvAIV::ColumnLen(uint32_t j) const
{
    if (isUpper_) {
        return j;
    }
    return n_ - 1U - j;
}

__aicore__ inline uint64_t CtpmvAIV::ColumnApOffset(uint32_t j) const
{
    if (isUpper_) {
        return ColumnBase(j);
    }
    return ColumnBase(j) + (ColumnFirstRow(j) - j);
}

// Normal-mode GatherMask writes repeats * ELEMS_PER_REPEAT / 2 elements per plane.
__aicore__ inline void CtpmvAIV::SplitPlanes(
    const LocalTensor<float>& re, const LocalTensor<float>& im, const LocalTensor<float>& src, uint32_t count)
{
    uint64_t rsvd = 0UL;
    const uint16_t repeats = static_cast<uint16_t>((count + ELEMS_PER_REPEAT - 1U) / ELEMS_PER_REPEAT);
    GatherMaskParams params{1U, repeats, REPEAT_BLOCKS, REPEAT_BLOCKS};
    GatherMask(re, src, PATTERN_EVEN, false, ELEMS_PER_REPEAT, params, rsvd);
    GatherMask(im, src, PATTERN_ODD, false, ELEMS_PER_REPEAT, params, rsvd);
}

// Prefix weight includes column lengths and transfer cost; it increases monotonically.
__aicore__ inline uint64_t CtpmvAIV::CostPrefix(uint32_t j) const
{
    if (j == 0U) {
        return 0UL;
    }
    const uint64_t jj = j;
    const uint64_t fixed = jj * CTPMV_BATCHED_COL_FIXED_COST;
    if (isUpper_) {
        return fixed + ((jj * (jj + 1UL)) >> 1UL);
    }
    return fixed + jj * n_ - ((jj * (jj - 1UL)) >> 1UL);
}

// Bisect prefix weights and align boundaries to ALIGN_ELEMS for vector operands.
__aicore__ inline uint32_t CtpmvAIV::SplitAt(uint64_t target) const
{
    uint32_t lo = 0U;
    uint32_t hi = n_;
    while (lo < hi) {
        const uint32_t mid = lo + ((hi - lo) >> 1U);
        if (CostPrefix(mid) < target) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo & ~ALIGN_MASK;
}

__aicore__ inline void CtpmvAIV::ClaimColumns()
{
    if (numCores_ <= 1U) {
        colLo_ = 0U;
        colHi_ = n_;
        return;
    }
    const uint64_t total = CostPrefix(n_);
    colLo_ = SplitAt(total * coreIdx_ / numCores_);
    colHi_ = (coreIdx_ + 1U >= numCores_) ? n_ : SplitAt(total * (coreIdx_ + 1UL) / numCores_);
}

// Capture the dense input vector before any in-place output store.
__aicore__ inline void CtpmvAIV::LoadX()
{
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};
    DataCopyExtParams params{1U, n_ * BYTES_PER_COMPLEX, 0U, 0U, 0U};
    DataCopyPad(aos_, xGm_, params, padParams);
    SetFlag<HardEvent::MTE2_V>(evtMte2V_);
    WaitFlag<HardEvent::MTE2_V>(evtMte2V_);
    // Deinterleave the full snapshot from offset zero to keep plane writes in bounds.
    SplitPlanes(xRe_, xIm_, aos_, n_ * FLOATS_PER_COMPLEX);
    PipeBarrier<PIPE_V>();
}

// Reuse the retired matrix planes for one index stream; source offsets remain zero.
__aicore__ inline void CtpmvAIV::InterleaveRange(uint32_t lo, uint32_t hi)
{
    const uint32_t count = (hi - lo) * FLOATS_PER_COMPLEX;
    auto offsets = aRe_.template ReinterpretCast<int32_t>();
    SetFlag<HardEvent::V_S>(evtVS_);
    WaitFlag<HardEvent::V_S>(evtVS_);
    for (uint32_t i = 0U; i < ALIGN_ELEMS; ++i) {
        offsets.SetValue(i, (i / 2U) * BYTES_PER_FLOAT + (i % 2U) * PLANE * BYTES_PER_FLOAT);
    }
    SetFlag<HardEvent::S_V>(evtSV_);
    WaitFlag<HardEvent::S_V>(evtSV_);
    for (uint32_t span = ALIGN_ELEMS; span < count; span *= 2U) {
        const uint32_t length = MinU32(span, count - span);
        Adds(offsets[span], offsets, static_cast<int32_t>(span / 2U * BYTES_PER_FLOAT), length);
        PipeBarrier<PIPE_V>();
    }
    Gather(aos_[lo * FLOATS_PER_COMPLEX], accRe_[lo], offsets.template ReinterpretCast<uint32_t>(), 0U, count);
    SetFlag<HardEvent::V_MTE3>(evtVMte3_);
    WaitFlag<HardEvent::V_MTE3>(evtVMte3_);
}

__aicore__ inline void CtpmvAIV::StoreX()
{
    // All cores must finish taking their input snapshot before any in-place store.
    if (numCores_ > 1U) {
        SyncAll();
    }
    const uint32_t count = colHi_ - colLo_;
    if (count == 0U) {
        return;
    }
    InterleaveRange(colLo_, colHi_);
    DataCopyExtParams params{1U, count * BYTES_PER_COMPLEX, 0U, 0U, 0U};
    const uint64_t offset = static_cast<uint64_t>(colLo_) * FLOATS_PER_COMPLEX;
    DataCopyPad(xGm_[offset], aos_[offset], params);
}

__aicore__ inline void CtpmvAIV::ZeroAccumulators()
{
    for (uint32_t off = colLo_; off < colHi_; off += APPLY_CHUNK) {
        const int32_t count = static_cast<int32_t>(MinU32(APPLY_CHUNK, colHi_ - off));
        Duplicate(accRe_[off], 0.0F, count);
        Duplicate(accIm_[off], 0.0F, count);
    }
}

// Apply the non-unit diagonal before triangular accumulation.
// Round each first product with Mul, then fuse its cross term to match the reference.
__aicore__ inline void CtpmvAIV::InitNonUnitDiagonal()
{
    // Initialize the diagonal for this core before adding its column dot products.
    for (uint32_t j = colLo_; j < colHi_; ++j) {
        const uint64_t diagonal = (ColumnBase(j) + (isUpper_ ? j : 0U)) * FLOATS_PER_COMPLEX;
        aRe_.SetValue(j, apGm_.GetValue(diagonal));
        aIm_.SetValue(j, apGm_.GetValue(diagonal + 1U));
    }
    SetFlag<HardEvent::S_V>(evtSV_);
    WaitFlag<HardEvent::S_V>(evtSV_);

    const uint32_t count = colHi_ - colLo_;
    if (count == 0U) {
        return;
    }
    const int32_t vectorCount = static_cast<int32_t>(count);
    Mul(prodRe_, aIm_[colLo_], xIm_[colLo_], vectorCount);
    Mul(prodIm_, aIm_[colLo_], xRe_[colLo_], vectorCount);
    PipeBarrier<PIPE_V>();
    if (isConj_) {
        Muls(prodIm_, prodIm_, -1.0f, vectorCount);
    } else {
        Muls(prodRe_, prodRe_, -1.0f, vectorCount);
    }
    PipeBarrier<PIPE_V>();
    // Preserve aRe_ for both FMA calls and each addend until it is consumed.
    CtpmvTensorFmaCompat(accRe_[colLo_], aRe_[colLo_], xRe_[colLo_], prodRe_, count);
    CtpmvTensorFmaCompat(accIm_[colLo_], aRe_[colLo_], xIm_[colLo_], prodIm_, count);
    PipeBarrier<PIPE_V>();
}

// A unit diagonal contributes x[j] only to the owning core's output columns.
__aicore__ inline void CtpmvAIV::AddUnitDiagonal()
{
    for (uint32_t off = colLo_; off < colHi_; off += APPLY_CHUNK) {
        const int32_t cnt = static_cast<int32_t>(MinU32(APPLY_CHUNK, colHi_ - off));
        Add(accRe_[off], accRe_[off], xRe_[off], cnt);
        Add(accIm_[off], accIm_[off], xIm_[off], cnt);
    }
}

// Pack columns at aligned slab offsets; zero leading padding before multiplication.
__aicore__ inline uint32_t CtpmvAIV::PlanBatch(uint32_t first, uint32_t& used)
{
    uint32_t end = 0U;
    uint32_t cols = 0U;
    const uint32_t room = colHi_ - first;
    const uint32_t limit = MinU32(SLAB_MAX_COLS, room);
    while (cols < limit) {
        const uint32_t j = first + cols;
        const uint32_t len = ColumnLen(j);
        // Include leading alignment padding in each column span.
        const uint32_t columnPad = ColumnFirstRow(j) & ALIGN_MASK;
        const uint32_t off = CeilAlignU32(end);
        const uint32_t span = columnPad + len;
        if (off + span > SLAB_ELEMS) {
            break;
        }
        slabOff_[cols] = off;
        end = off + span;
        cols++;
    }
    // Round non-final batches to eight columns so the next output start stays 32-byte aligned.
    if (cols < room && cols >= ALIGN_ELEMS) {
        cols &= ~ALIGN_MASK;
        end = slabOff_[cols - 1U] + ((ColumnFirstRow(first + cols - 1U) & ALIGN_MASK) + ColumnLen(first + cols - 1U));
    }
    used = end;
    return cols;
}

// Widen DMA backwards by up to three complex elements into padding.
// This keeps the transfer within its reserved slot; clear padding before use.
__aicore__ inline void CtpmvAIV::StageBatchColumn(uint32_t j, uint32_t slab)
{
    const uint32_t len = ColumnLen(j);
    if (len == 0U) {
        return;
    }
    const uint32_t pad = ColumnFirstRow(j) & ALIGN_MASK;
    const uint32_t back = pad & DMA_MASK;
    const uint64_t src = (ColumnApOffset(j) - back) * FLOATS_PER_COMPLEX;
    const uint32_t dst = (slab + pad - back) * FLOATS_PER_COMPLEX;
    DataCopyExtParams params{1U, (len + back) * BYTES_PER_COMPLEX, 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};
    DataCopyPad(aos_[dst], apGm_[src], params, padParams);
}

__aicore__ inline void CtpmvAIV::LoadBatch(uint32_t first, uint32_t cols, uint32_t used)
{
    for (uint32_t k = 0U; k < cols; ++k) {
        StageBatchColumn(first + k, slabOff_[k]);
    }
    SetFlag<HardEvent::MTE2_V>(evtMte2V_);
    WaitFlag<HardEvent::MTE2_V>(evtMte2V_);
    SplitPlanes(aRe_, aIm_, aos_, used * FLOATS_PER_COMPLEX);
    PipeBarrier<PIPE_V>();
}

// Clear the column's leading padding before accumulation.
__aicore__ inline void CtpmvAIV::RunBatchColumn(uint32_t j, uint32_t slab, uint32_t k)
{
    const uint32_t len = ColumnLen(j);
    if (len == 0U) {
        const uint32_t slot = k * RED_SLOT_FLOATS;
        Duplicate(red_[slot], 0.0f, RED_SLOT_FLOATS);
        return;
    }
    const uint32_t firstRow = ColumnFirstRow(j);
    const uint32_t pad = firstRow & ALIGN_MASK;
    const uint32_t cnt = pad + len;
    if (pad > 0U) {
        const int32_t padCnt = static_cast<int32_t>(pad);
        Duplicate(aRe_[slab], 0.0f, padCnt);
        Duplicate(aIm_[slab], 0.0f, padCnt);
        PipeBarrier<PIPE_V>();
    }
    const uint32_t row = firstRow - pad;
    ProductDotChunk(slab, row, cnt);
    ReduceDotColumn(k, cnt);
}

__aicore__ inline void CtpmvAIV::RunBatch(uint32_t first, uint32_t cols, uint32_t used, bool lastBatch)
{
    // An empty strictly triangular range requires no transfer or deinterleave.

    // Zero only the result slots consumed by this batch's Gather publication.
    if ((first & ALIGN_MASK) != 0U) {
        Duplicate(red_[0], 0.0f, cols * RED_SLOT_FLOATS);
        PipeBarrier<PIPE_V>();
    }
    if (used != 0U) {
        LoadBatch(first, cols, used);
    }
    for (uint32_t k = 0U; k < cols; ++k) {
        RunBatchColumn(first + k, slabOff_[k], k);
    }
    PublishDotColumns(first, cols);
    // Finish accumulator writes before the next vector consumer.
    PipeBarrier<PIPE_V>();
    // Drain slab reads before the next batch refills it.
    if (!lastBatch) {
        SetFlag<HardEvent::V_MTE2>(evtVMte2_);
        WaitFlag<HardEvent::V_MTE2>(evtVMte2_);
    }
}

// Form each complex product with a rounded first term and a fused cross term,
// preserving the reference exceptional-value boundary.
__aicore__ inline void CtpmvAIV::ProductDotChunk(uint32_t slab, uint32_t row, uint32_t cnt)
{
    const int32_t c = static_cast<int32_t>(cnt);
    // Exact rhs negation lets MulAddDst fuse the cross product with the rounded first term.
    Mul(prodRe_, aRe_[slab], xRe_[row], c);
    Muls(tmp_, xIm_[row], isConj_ ? 1.0F : -1.0F, c);
    PipeBarrier<PIPE_V>();
    MulAddDst<float, float>(prodRe_, aIm_[slab], tmp_, c);
    PipeBarrier<PIPE_V>();

    Mul(prodIm_, aRe_[slab], xIm_[row], c);
    Muls(tmp_, xRe_[row], isConj_ ? -1.0F : 1.0F, c);
    PipeBarrier<PIPE_V>();
    MulAddDst<float, float>(prodIm_, aIm_[slab], tmp_, c);
}

// trans=T/C: each staged column is multiplied once and reduced to its output element.

// Keep per-column reductions in UB; PublishDotColumns drains the vector pipe once per batch.
__aicore__ inline void CtpmvAIV::ReduceDotColumn(uint32_t k, uint32_t cnt)
{
    const int32_t width = static_cast<int32_t>(cnt);
    const uint32_t slot = k * RED_SLOT_FLOATS;
    LocalTensor<float> reduceScratch = aos_[DOT_REDUCE_SCRATCH_OFFSET];
    ReduceSum(red_[slot], prodRe_, reduceScratch, width);
    PipeBarrier<PIPE_V>();
    ReduceSum(red_[slot + RED_SECOND], prodIm_, reduceScratch, width);
    PipeBarrier<PIPE_V>();
}

// Publish reduced slot k to column first + k in ascending order.
__aicore__ inline void CtpmvAIV::PublishDotColumns(uint32_t first, uint32_t cols)
{
    if (cols == 0U) {
        return;
    }
    // A batch with fewer than eight columns can leave the next start unaligned.
    // Publish through aligned scratch blocks while preserving neighboring outputs.
    if ((first & ALIGN_MASK) != 0U) {
        // Publish each slot through an aligned eight-lane block.
        LocalTensor<int32_t> idxI32 = idx_.template ReinterpretCast<int32_t>();
        for (uint32_t k = 0U; k < cols; ++k) {
            const uint32_t column = first + k;
            const uint32_t alignedFirst = column & ~ALIGN_MASK;
            const uint32_t lane = column - alignedFirst;
            const uint32_t slot = k * RED_SLOT_FLOATS;
            // The four-byte offset selects a zero slot for every unselected lane.
            Duplicate<int32_t>(idxI32, static_cast<int32_t>(sizeof(float)), ALIGN_ELEMS);
            SetFlag<HardEvent::V_S>(evtVS_);
            WaitFlag<HardEvent::V_S>(evtVS_);
            idxI32.SetValue(lane, 0);
            SetFlag<HardEvent::S_V>(evtSV_);
            WaitFlag<HardEvent::S_V>(evtSV_);

            Gather(tmp_, red_[slot], idx_, 0U, ALIGN_ELEMS);
            PipeBarrier<PIPE_V>();
            Add(accRe_[alignedFirst], accRe_[alignedFirst], tmp_, ALIGN_ELEMS);
            PipeBarrier<PIPE_V>();

            Gather(tmp_, red_[slot + RED_SECOND], idx_, 0U, ALIGN_ELEMS);
            PipeBarrier<PIPE_V>();
            Add(accIm_[alignedFirst], accIm_[alignedFirst], tmp_, ALIGN_ELEMS);
            PipeBarrier<PIPE_V>();
        }
        return;
    }
    // Aligned batches share one byte-offset stream; add 32 bytes for the imaginary result.
    const int32_t byteStride = static_cast<int32_t>(RED_SLOT_FLOATS * sizeof(float));
    LocalTensor<int32_t> idxI32 = idx_.template ReinterpretCast<int32_t>();
    CreateVecIndex<int32_t>(idxI32, 0, cols);
    Muls<int32_t>(idxI32, idxI32, byteStride, cols);
    PipeBarrier<PIPE_V>();
    Gather(prodRe_, red_, idx_, 0U, cols);
    PipeBarrier<PIPE_V>();
    Add(accRe_[first], accRe_[first], prodRe_, static_cast<int32_t>(cols));
    PipeBarrier<PIPE_V>();
    Gather(prodIm_, red_, idx_, RED_SECOND * sizeof(float), cols);
    PipeBarrier<PIPE_V>();
    Add(accIm_[first], accIm_[first], prodIm_, static_cast<int32_t>(cols));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void CtpmvAIV::RunColumns()
{
    uint32_t first = colLo_;
    while (first < colHi_) {
        uint32_t used = 0U;
        const uint32_t columns = PlanBatch(first, used);
        RunBatch(first, columns, used, first + columns == colHi_);
        first += columns;
    }
}

__aicore__ inline void CtpmvAIV::Process()
{
    ClaimColumns();
    LoadX();
    ZeroAccumulators();
    if (!isUnit_) {
        InitNonUnitDiagonal();
    }
    RunColumns();
    if (isUnit_) {
        AddUnitDiagonal();
    }
    StoreX();
}

__global__ __aicore__ void ctpmv_path_batched_reference_kernel(GM_ADDR aPacked, GM_ADDR x, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetMaskNorm();
    TPipe pipe;
    SetSyncBaseAddr(tiling.syncAddress);
    CtpmvAIV op;
    op.Init(aPacked, x, tiling, &pipe);
    op.Process();
    // TPipe drains work but leaves mask and cross-core base registers unchanged.
    pipe.Destroy();
    SetSyncBaseAddr(0);
    ResetMask();
}

void ctpmv_batched_reference_kernel_do(
    GM_ADDR aPacked, GM_ADDR x, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctpmv_path_batched_reference_kernel<<<numBlocks, nullptr, stream>>>(aPacked, x, tiling);
}
