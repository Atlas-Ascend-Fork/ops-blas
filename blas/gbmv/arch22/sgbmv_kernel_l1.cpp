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
 * \file sgbmv_kernel_l1.cpp
 * \brief L1 column-split path (compact band storage, N domain) of the arch22
 * sgbmv kernel; split out of sgbmv_kernel.cpp for source-size limits.
 */


// ============================================================================
// L1 path: N-domain column split for compact
// band storage (lda == kl+ku+1, i.e. A is stored densely with no inter-column
// gap). Each core owns a contiguous column range and streams its A columns as
// one contiguous GM run (per-column 1D DMA from an 8-aligned base; >= 8 zero
// floats follow every staged segment). Partial y contributions accumulate in
// a per-core UB window; partials land in workSpace rows (stride ceil8(m)) and
// all cores fold them back into y after a SyncAll barrier. Numerics are
// deterministic: within a core the columns accumulate in index order and the
// cross-core fold is a fixed halving tree.
// ============================================================================

constexpr uint32_t L1_WIN_MAX = 8192U;     // per-core y window cap (floats)
constexpr uint32_t L1_SLOT_ELEMS = 6144U;  // per ping-pong staging slot (floats)
constexpr uint32_t L1_LT = 1024U;          // mid-piece length cap (gather table stride - 16)
constexpr uint32_t L1_MAX_CORES = 64U;     // reduction fold-tree width
constexpr uint32_t L1_RC = 256U;           // reduce row chunk (floats)
constexpr uint32_t L1_ZERO_SLOT = L1_LT + 8U; // zero slot inside the A2 temp (floats)
constexpr uint32_t L1_BAND_MAX = 6120U;    // band slot cap (colStride <= L1_SLOT_ELEMS)
constexpr uint32_t L1_X_ELEMS = 2048U;     // per-core x column window (floats)

// UB layout for the L1 path (aliases the vector-fallback VB region; the paths
// are mutually exclusive at runtime).
constexpr uint32_t L1_ALIGN32(uint32_t v)
{
    return (v + 31U) & ~31U;
}
constexpr uint32_t L1_YWIN = 0U;
constexpr uint32_t L1_SLOT0 = L1_YWIN + (L1_WIN_MAX + 48U) * BYTES_PER_FLOAT;
constexpr uint32_t L1_SLOT1 = L1_SLOT0 + L1_SLOT_ELEMS * BYTES_PER_FLOAT;
constexpr uint32_t L1_T = L1_SLOT1 + L1_SLOT_ELEMS * BYTES_PER_FLOAT;
constexpr uint32_t L1_A2 = L1_ALIGN32(L1_T + (L1_LT + 16U) * BYTES_PER_FLOAT);
constexpr uint32_t L1_X = L1_ALIGN32(L1_A2 + (L1_LT + 16U) * BYTES_PER_FLOAT);
constexpr uint32_t L1_TBLV = L1_ALIGN32(L1_X + L1_X_ELEMS * BYTES_PER_FLOAT);
constexpr uint32_t L1_WS = L1_ALIGN32(L1_TBLV + SIMD_WIDTH * SIMD_WIDTH * 4U);
// Big variant tables for the single-shot column AXPY: 8 tables
// of L1_LT+16 entries, aliasing the reduction workspace region — they are
// streaming-phase-only and the reduce phase overwrites them after SyncAll.
constexpr uint32_t L1_BIG_H = L1_LT + 16U;                 // entries per variant table
constexpr uint32_t L1_TBLIN = L1_WS + L1_MAX_CORES * L1_RC * BYTES_PER_FLOAT;
constexpr uint32_t L1_TBLBIG = L1_WS;                        // aliases the reduce workspace
constexpr uint32_t L1_BIG0 = L1_TBLBIG + 8U * L1_BIG_H * 4U; // doubled base table (2*L1_BIG_H entries)
static_assert(L1_BIG0 + 2U * L1_BIG_H * 4U <= L1_TBLIN, "L1 variant tables must fit in the WS region");
static_assert(L1_WS + L1_MAX_CORES * L1_RC * BYTES_PER_FLOAT <= 192U * 1024U, "L1 buffers must fit in UB");
static_assert(L1_TBLIN + (L1_LT + 16U) * 4U <= 192U * 1024U, "L1 gather table must fit in UB");

// Big variant tables: BIG[h][m] = (m >= h) ? (m-h)*4 : zeroSlotBytes, built
// with 8 vector gathers from a doubled linear base table whose last 8 entries
// per copy are the zero-slot constant (see L1ColumnAXPY).
__aicore__ inline void L1BuildBigTables(
    AsdopsBuffer<ArchType::ASCEND_V220>& buf, LocalTensor<uint32_t>& tblLin, uint32_t neededMask)
{
    if (neededMask == 0U) {
        return; // all of this core's columns are 8-aligned: no variant table needed
    }
    auto big0 = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(L1_BIG0);
    MakeOffsetSequence(big0.ReinterpretCast<int32_t>(), L1_BIG_H, static_cast<int32_t>(BYTES_PER_FLOAT));
    Adds(big0.ReinterpretCast<int32_t>()[L1_BIG_H], big0.ReinterpretCast<int32_t>(), 0, L1_BIG_H);
    const uint32_t zeroBytes = L1_ZERO_SLOT * BYTES_PER_FLOAT;
    Duplicate(big0[L1_BIG_H - SIMD_WIDTH], zeroBytes, SIMD_WIDTH);
    Duplicate(big0[2U * L1_BIG_H - SIMD_WIDTH], zeroBytes, SIMD_WIDTH);
    PIPE_BARRIER(V);
    for (uint32_t h = 0U; h < SIMD_WIDTH; ++h) {
        if (((neededMask >> h) & 1U) == 0U) {
            continue; // build only the shift variants this core's columns use
        }
        auto bigH = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(L1_TBLBIG + h * L1_BIG_H * 4U);
        Gather(bigH, big0, tblLin, (L1_BIG_H - h) * BYTES_PER_FLOAT, L1_BIG_H);
    }
    PIPE_BARRIER(V);
}

// Stage columns [cs, ce) into one ping-pong slot. Consecutive compact columns
// are contiguous in GM, so this issues a contiguous run of 1D DMAs (each from
// an 8-float-aligned GM base, zero-padded to 8-element alignment).
__aicore__ inline void L1IssueChunk(
    LocalTensor<float> slot, GlobalTensor<float>& aGlobal, uint32_t cs, uint32_t ce,
    const SgbmvTilingData& tiling, uint32_t m, uint32_t band, uint32_t colStride)
{
    for (uint32_t jc = cs; jc < ce; ++jc) {
        uint32_t segLo = (jc > tiling.ku) ? jc - tiling.ku : 0U;
        uint32_t segHi = MinVal(m, jc + tiling.kl + 1U);
        if (segLo >= segHi) {
            continue; // column does not intersect the row range
        }
        uint64_t aOff = static_cast<uint64_t>(band) * jc + (tiling.ku + segLo - jc);
        uint32_t r = static_cast<uint32_t>(aOff & 7ULL);
        VecCopyIn(slot[(jc - cs) * colStride], aGlobal[aOff - r], segHi - segLo + r);
    }
}

// Aligned column (h == 0): every piece is a single linear gather + Muls + Add.
__aicore__ inline void L1AxpyAlignedPieces(
    LocalTensor<float>& ubY, LocalTensor<float>& ubT, LocalTensor<float>& slot, LocalTensor<uint32_t>& tblLin,
    float xj, uint32_t segOff, uint32_t a0, uint32_t total)
{
    for (uint32_t p = 0U; p < total; p += L1_LT) {
        uint32_t l = MinVal(L1_LT, total - p);
        Gather(ubT, slot, tblLin, (segOff + p) * BYTES_PER_FLOAT, l);
        PIPE_BARRIER(V);
        Muls(ubT, ubT, xj, l);
        PIPE_BARRIER(V);
        Add(ubY[a0 + p], ubY[a0 + p], ubT, l);
        PIPE_BARRIER(V);
    }
}

// Misaligned column: a tiny 8-lane head (linearize + variant-table shift,
// lanes below the segment read the zero slot) lands on the 8-aligned floor
// offset a0-h; all continuation pieces are single linear gathers starting
// h floats earlier (segOff + p - h), which aligns them to the same 8-grid
// and keeps coverage contiguous (head covers rows [segLo, segLo+8-h),
// the piece at p starts at segLo+p-h). Without the -h every piece
// boundary dropped the h rows in [segLo+p-h, segLo+p).
__aicore__ inline void L1AxpyMisalignedColumn(
    LocalTensor<float>& ubY, LocalTensor<float>& ubT, LocalTensor<float>& ubA2, LocalTensor<uint32_t>& tblLin,
    LocalTensor<uint32_t>& bigBase, LocalTensor<float>& slot, float xj, uint32_t segOff, uint32_t a0, uint32_t h,
    uint32_t total)
{
    auto bigH = bigBase[h * L1_BIG_H];
    Gather(ubA2, slot, tblLin, segOff * BYTES_PER_FLOAT, SIMD_WIDTH);
    PIPE_BARRIER(V);
    Gather(ubT, ubA2, bigH, 0U, SIMD_WIDTH);
    PIPE_BARRIER(V);
    Muls(ubT, ubT, xj, SIMD_WIDTH);
    PIPE_BARRIER(V);
    Add(ubY[a0 - h], ubY[a0 - h], ubT, SIMD_WIDTH);
    PIPE_BARRIER(V);
    for (uint32_t p = SIMD_WIDTH; p < total; p += L1_LT) {
        uint32_t l = MinVal(L1_LT, total - p);
        Gather(ubT, slot, tblLin, (segOff + p - h) * BYTES_PER_FLOAT, l);
        PIPE_BARRIER(V);
        Muls(ubT, ubT, xj, l);
        PIPE_BARRIER(V);
        Add(ubY[a0 - h + p], ubY[a0 - h + p], ubT, l);
        PIPE_BARRIER(V);
    }
}

__aicore__ inline void L1ColumnAXPY(
    LocalTensor<float> ubY, LocalTensor<float> ubT, LocalTensor<float> ubA2, LocalTensor<uint32_t> tblLin,
    LocalTensor<uint32_t> bigBase, LocalTensor<float> slot, __ubuf__ const float* xPtr, uint32_t jc, uint32_t cs,
    uint32_t c0, uint32_t wLo, const SgbmvTilingData& tiling, uint32_t m, uint32_t colStride)
{
    uint32_t segLo = (jc > tiling.ku) ? jc - tiling.ku : 0U;
    uint32_t segHi = MinVal(m, jc + tiling.kl + 1U);
    if (segLo >= segHi) {
        return;
    }
    float xj = xPtr[jc - c0];
    if (xj == 0.0f) {
        return;
    }
    uint32_t len = segHi - segLo;
    uint64_t aOff = static_cast<uint64_t>(tiling.lda) * jc + (tiling.ku + segLo - jc);
    uint32_t r = static_cast<uint32_t>(aOff & 7ULL);
    uint32_t colBase = (jc - cs) * colStride;
    uint32_t segOff = colBase + r; // slot offset of row segLo
    uint32_t staged = (len + r + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    // 16-float zero slack: the full-column gather reads up to r+staged+7
    // floats past the column base (colStride >= staged+17, so this stays
    // inside this column's stride slot).
    Duplicate(slot[colBase + staged], 0.0f, 2U * SIMD_WIDTH);
    PIPE_BARRIER(V);

    uint32_t a0 = segLo - wLo;
    uint32_t h = a0 & (SIMD_WIDTH - 1U);
    uint32_t total = staged + SIMD_WIDTH;
    if (h == 0U) {
        L1AxpyAlignedPieces(ubY, ubT, slot, tblLin, xj, segOff, a0, total);
        return;
    }
    L1AxpyMisalignedColumn(ubY, ubT, ubA2, tblLin, bigBase, slot, xj, segOff, a0, h, total);
}

// One-time setup for this core's L1 column split: linear gather table,
// head-shift tables, the gather zero slot, and the y-window zero span.
__aicore__ inline uint32_t L1ComputeNeededHeadMask(
    uint32_t c0, uint32_t c1, uint32_t wLo, uint32_t m, const SgbmvTilingData& tiling)
{
    uint32_t neededH = 0U;
    for (uint32_t jc = c0; jc < c1; ++jc) {
        uint32_t segLo = (jc > tiling.ku) ? jc - tiling.ku : 0U;
        uint32_t segHi = MinVal(m, jc + tiling.kl + 1U);
        if (segLo >= segHi) {
            continue; // column does not intersect the row range
        }
        uint32_t h = (segLo - wLo) & (SIMD_WIDTH - 1U);
        if (h != 0U) {
            neededH |= 1U << h; // only misaligned heads use a variant table
        }
    }
    return neededH;
}

// workspace slice zero-init for rows outside this core's window (the
// reduce phase sums all K partial slices over every row); ubT is the zero source
__aicore__ inline void L1InitColumnKernel(
    uint32_t c0, uint32_t c1, uint32_t wLo, uint32_t m, uint32_t winLen, const SgbmvTilingData& tiling,
    LocalTensor<float>& ubY, LocalTensor<float>& ubA2, LocalTensor<uint32_t>& tblLin,
    LocalTensor<uint32_t>& bigBase, AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    // one-time setup: linear gather table, head-shift tables, zero slot
    MakeOffsetSequence(tblLin.ReinterpretCast<int32_t>(), L1_LT + 16U, static_cast<int32_t>(BYTES_PER_FLOAT));
    uint32_t neededH = L1ComputeNeededHeadMask(c0, c1, wLo, m, tiling);
    L1BuildBigTables(buf, tblLin, neededH);
    bigBase = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(L1_TBLBIG);
    Duplicate(ubA2[L1_ZERO_SLOT], 0.0f, SIMD_WIDTH);
    // zero only the span the column AXPYs read-modify-write: the Add range
    // ends at most winLen + 24 floats past the window start (staged rounding
    // plus head shift), so the whole 8K window cap need not be cleared
    const uint32_t yZeroLen = (winLen == 0U)
        ? 0U
        : MinVal((L1_WIN_MAX + 48U) & ~(SIMD_WIDTH - 1U), (winLen + 32U + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U));
    if (yZeroLen != 0U) {
        Duplicate(ubY, 0.0f, yZeroLen);
    }
    PIPE_BARRIER(V);
}

// Scan this core's column range for the head-shift variants actually used
// (only misaligned heads need a variant table).
__aicore__ inline void L1ZeroInitWorkspaceSlice(
    GlobalTensor<float>& wsGlobal, LocalTensor<float>& ubT, uint32_t blkIdx, uint32_t S, uint32_t c0, uint32_t c1,
    uint32_t wLo, uint32_t wHi, uint32_t winLen, uint32_t zeroEnd)
{
    Duplicate(ubT, 0.0f, L1_LT + 16U);
    PIPE_BARRIER(V);
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
    const uint64_t wsBase = static_cast<uint64_t>(blkIdx) * S;
    const bool noWindow = (c0 >= c1) || (winLen == 0U);
    const uint32_t headZeroEnd = noWindow ? zeroEnd : MinVal(wLo, zeroEnd);
    for (uint32_t zLo = 0U; zLo < headZeroEnd;) {
        uint32_t zLen = MinVal(L1_LT + 16U, headZeroEnd - zLo);
        CopyUbToGm(wsGlobal[wsBase + zLo], ubT, zLen);
        zLo += zLen;
    }
    const uint32_t tailZeroStart = noWindow ? zeroEnd : MinVal(wHi, zeroEnd);
    for (uint32_t zLo = tailZeroStart; zLo < zeroEnd;) {
        uint32_t zLen = MinVal(L1_LT + 16U, zeroEnd - zLo);
        CopyUbToGm(wsGlobal[wsBase + zLo], ubT, zLen);
        zLo += zLen;
    }
    SET_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
}

// Consume one staged chunk: column AXPYs in index order over the y window,
// then drain the V pipe before this slot may be overwritten by the DMA
// staged for the next chunk.
__aicore__ inline void L1ConsumeChunkAndDrain(
    LocalTensor<float>& ubY, LocalTensor<float>& ubT, LocalTensor<float>& ubA2, LocalTensor<uint32_t>& tblLin,
    LocalTensor<uint32_t>& bigBase, LocalTensor<float>& slot, __ubuf__ const float* xPtr, uint32_t cs, uint32_t ce,
    uint32_t c0, uint32_t wLo, const SgbmvTilingData& tiling, uint32_t m, uint32_t colStride)
{
    for (uint32_t jc = cs; jc < ce; ++jc) {
        L1ColumnAXPY(
            ubY, ubT, ubA2, tblLin, bigBase, slot, xPtr, jc, cs, c0, wLo, tiling, m, colStride);
    }
    // drain the V pipe before this slot may be overwritten by the DMA
    // staged for the next chunk
    SET_FLAG(V, S, EVENT_ID2);
    WAIT_FLAG(V, S, EVENT_ID2);
}

// main streaming loop: 2-slot ping-pong over column chunks (chunk i+1 DMA
// is in flight while chunk i computes; slot reuse is guarded by the (V,S)
// drain, rotating EVENT_ID0/1 per slot — the batch ping-pong pattern)
__aicore__ inline void L1StreamColumnChunks(
    uint32_t blkIdx, uint32_t c0, uint32_t c1, uint32_t wLo, uint32_t winLen, uint32_t S, uint32_t m, uint32_t band,
    const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal, GlobalTensor<float>& xGlobal,
    GlobalTensor<float>& wsGlobal, LocalTensor<float>& ubX, LocalTensor<float>& ubY, LocalTensor<float>& ubT,
    LocalTensor<float>& ubA2, LocalTensor<uint32_t>& tblLin, LocalTensor<uint32_t>& bigBase,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    const uint32_t colStride = (band + 24U + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    const uint32_t W = MaxVal(1U, MinVal(64U, L1_SLOT_ELEMS / colStride));
    VecCopyIn(ubX, xGlobal[c0], c1 - c0);
    SET_FLAG(MTE2, S, EVENT_ID0);
    WAIT_FLAG(MTE2, S, EVENT_ID0);
    PIPE_BARRIER(V);
    auto xPtr = reinterpret_cast<__ubuf__ const float*>(ubX.GetPhyAddr());
    auto slot0 = buf.GetBuffer<BufferType::ASCEND_UB, float>(L1_SLOT0);
    auto slot1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(L1_SLOT1);
    uint32_t chunk = 0U;
    uint32_t cs = c0;
    uint32_t ce = MinVal(c0 + W, c1);
    L1IssueChunk(slot0, aGlobal, cs, ce, tiling, m, band, colStride);
    SetMte2SlotReady(chunk & 1U);
    bool hasNext = ce < c1;
    while (hasNext) {
        uint32_t next = MinVal(ce + W, c1);
        auto slotN = (chunk & 1U) != 0U ? slot0 : slot1;
        L1IssueChunk(slotN, aGlobal, ce, next, tiling, m, band, colStride);
        SetMte2SlotReady((chunk & 1U) ^ 1U);
        auto slot = (chunk & 1U) != 0U ? slot1 : slot0;
        WaitMte2SlotReady(chunk & 1U);
        L1ConsumeChunkAndDrain(ubY, ubT, ubA2, tblLin, bigBase, slot, xPtr, cs, ce, c0, wLo, tiling, m, colStride);
        cs = ce;
        ce = next;
        chunk += 1U;
        hasNext = ce < c1;
    }
    auto slot = (chunk & 1U) != 0U ? slot1 : slot0;
    WaitMte2SlotReady(chunk & 1U);
    L1ConsumeChunkAndDrain(ubY, ubT, ubA2, tblLin, bigBase, slot, xPtr, cs, ce, c0, wLo, tiling, m, colStride);
    // partial write into this core's workspace row (V must finish first)
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
    CopyUbToGm(wsGlobal[static_cast<uint64_t>(blkIdx) * S + wLo], ubY, winLen);
    SET_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
}

// rows past coverEnd: y = beta * y (all-zero when beta == 0);
// no K-slice workspace read and no fold
__aicore__ inline void L1ReduceBetaOnlyChunk(
    uint32_t r0, uint32_t len, uint32_t len8, const SgbmvTilingData& tiling, GlobalTensor<float>& yGlobal,
    LocalTensor<float>& wsT)
{
    if (tiling.beta == 0.0f) {
        Duplicate(wsT, 0.0f, len8);
        PIPE_BARRIER(V);
    } else {
        VecCopyIn(wsT, yGlobal[r0], len);
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        Muls(wsT, wsT, tiling.beta, len8);
        PIPE_BARRIER(V);
    }
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
    CopyUbToGm(yGlobal[r0], wsT, len);
    SET_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    PIPE_BARRIER(ALL);
}

// Reduce one L1_RC-row chunk: sum all K workspace slices over the chunk rows
// (zero padding up to the pow2 fold width), apply alpha/beta and write y.
__aicore__ inline void L1ReduceFoldedChunk(
    uint32_t ch, uint32_t m, uint32_t nFoldChunks, uint32_t foldW, uint32_t K, uint32_t S,
    const SgbmvTilingData& tiling, GlobalTensor<float>& yGlobal, GlobalTensor<float>& wsGlobal,
    LocalTensor<float>& wsT, LocalTensor<float>& ubT)
{
    uint32_t r0 = ch * L1_RC;
    uint32_t len = MinVal(L1_RC, m - r0);
    uint32_t len8 = (len + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    if (ch >= nFoldChunks) {
        L1ReduceBetaOnlyChunk(r0, len, len8, tiling, yGlobal, wsT);
        return;
    }
    AscendC::DataCopyPad(
        wsT, wsGlobal[r0],
        AscendC::DataCopyExtParams(static_cast<uint16_t>(K), len8 * BYTES_PER_FLOAT,
                                   (S - len8) * BYTES_PER_FLOAT, 0, 0),
        AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0.0f));
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
    if (foldW > K) {
        Duplicate(wsT[K * len8], 0.0f, (foldW - K) * len8);
        PIPE_BARRIER(V);
    }
    for (uint32_t half = foldW / 2U; half >= 1U; half >>= 1U) {
        Add(wsT, wsT, wsT[half * len8], half * len8);
        PIPE_BARRIER(V);
    }
    if (tiling.alpha != 1.0f) {
        Muls(wsT, wsT, tiling.alpha, len8);
        PIPE_BARRIER(V);
    }
    if (tiling.beta != 0.0f) {
        VecCopyIn(ubT, yGlobal[r0], len);
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        Muls(ubT, ubT, tiling.beta, len8);
        PIPE_BARRIER(V);
        Add(wsT, wsT, ubT, len8);
        PIPE_BARRIER(V);
    }
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
    CopyUbToGm(yGlobal[r0], wsT, len);
    SET_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    PIPE_BARRIER(ALL);
}

// cooperative reduction: chunks of L1_RC rows, contiguous split over K
// cores; every chunk sums all K partial slices (zero-initialized outside
// the owning core's window), applies alpha/beta and writes y.
__aicore__ inline void L1ReduceChunks(
    uint32_t blkIdx, uint32_t K, uint32_t m, uint32_t nFoldChunks, uint32_t S, const SgbmvTilingData& tiling,
    GlobalTensor<float>& yGlobal, GlobalTensor<float>& wsGlobal, LocalTensor<float>& ubT,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    // K is the caller's grid size and >= 1 on every real launch; bail out
    // early if that precondition is ever broken.
    if (K == 0U) {
        return;
    }
    auto wsT = buf.GetBuffer<BufferType::ASCEND_UB, float>(L1_WS);
    const uint32_t nChunks = (m + L1_RC - 1U) / L1_RC;
    // fold width: round K up to a power of two instead of the full
    // L1_MAX_CORES tree, so the zero padding and the fold adds scale with
    // the actual core count (K=20 folds 5 levels over 32 slots, not 6
    // levels over 64)
    uint32_t foldW = 1U;
    while (foldW < K) {
        foldW <<= 1U;
    }
    uint32_t chLo = static_cast<uint32_t>((static_cast<uint64_t>(blkIdx) * nChunks) / K);
    uint32_t chHi = static_cast<uint32_t>((static_cast<uint64_t>(blkIdx + 1U) * nChunks) / K);
    for (uint32_t ch = chLo; ch < chHi; ++ch) {
        L1ReduceFoldedChunk(ch, m, nFoldChunks, foldW, K, S, tiling, yGlobal, wsGlobal, wsT, ubT);
    }
}

__aicore__ inline void ProcessSgbmvNColSplit(
    uint32_t blkIdx, uint32_t numBlocks, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y,
    GM_ADDR workSpace)
{
    const uint32_t m = tiling.m;
    const uint32_t n = tiling.n;
    const uint32_t band = tiling.lda; // compact: lda == kl + ku + 1
    const uint32_t K = numBlocks;
    const uint32_t c0 = static_cast<uint32_t>((static_cast<uint64_t>(blkIdx) * n) / K);
    const uint32_t c1 = static_cast<uint32_t>((static_cast<uint64_t>(blkIdx + 1U) * n) / K);
    const uint32_t wLo = (c0 > tiling.ku) ? c0 - tiling.ku : 0U;
    const uint32_t wHi = MinVal(m, c1 + tiling.kl);
    const uint32_t winLen = (wHi > wLo) ? (wHi - wLo) : 0U;
    const uint32_t S = (m + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    // Row r receives an alpha contribution iff some column j < n satisfies
    // j >= r - kl, i.e. r <= n - 1 + kl. Rows [coverEnd, m) are pure beta*y
    // (zeros when beta == 0): the workspace fold only has to cover rows
    // [0, coverEnd), and the remaining reduction chunks take a cheap
    // zero/beta-rescale path instead of reading and folding all K slices.
    const uint32_t coverEnd = MinVal(m, tiling.n + tiling.kl);
    const uint32_t nFoldChunks = (coverEnd + L1_RC - 1U) / L1_RC;
    const uint32_t zeroEnd = MinVal(S, nFoldChunks * L1_RC); // rows the fold reads

    GlobalTensor<float> aGlobal;
    GlobalTensor<float> xGlobal;
    GlobalTensor<float> yGlobal;
    GlobalTensor<float> wsGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * n);
    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), n);
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), m);
    wsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), static_cast<uint64_t>(K) * S);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto ubY = buf.GetBuffer<BufferType::ASCEND_UB, float>(L1_YWIN);
    auto ubT = buf.GetBuffer<BufferType::ASCEND_UB, float>(L1_T);
    auto ubA2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(L1_A2);
    auto ubX = buf.GetBuffer<BufferType::ASCEND_UB, float>(L1_X);
    auto tblLin = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(L1_TBLIN);

    LocalTensor<uint32_t> bigBase;
    L1InitColumnKernel(c0, c1, wLo, m, winLen, tiling, ubY, ubA2, tblLin, bigBase, buf);

    L1ZeroInitWorkspaceSlice(wsGlobal, ubT, blkIdx, S, c0, c1, wLo, wHi, winLen, zeroEnd);

    if (c0 < c1 && winLen > 0U) {
        L1StreamColumnChunks(
            blkIdx, c0, c1, wLo, winLen, S, m, band, tiling, aGlobal, xGlobal, wsGlobal, ubX, ubY, ubT, ubA2, tblLin,
            bigBase, buf);
    }
    PIPE_BARRIER(ALL);
    SyncAll();

    L1ReduceChunks(blkIdx, K, m, nFoldChunks, S, tiling, yGlobal, wsGlobal, ubT, buf);
}

// Dedicated entry for the L1 column-split path: kept as a separate kernel so
// the legacy kernel's launch footprint is unchanged (inlining the L1 body —
// and its SyncAll — into sgbmv_kernel added a constant ~2us to every launch,
// measured on the fast/small-case floor).
__global__ __aicore__ void sgbmv_kernel_l1(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const SgbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t numBlocks = tiling.useCoreNum;
    if (numBlocks == 0U || blkIdx >= numBlocks) {
        return;
    }
    if (tiling.colSplit == 0U || workSpace == nullptr || numBlocks > L1_MAX_CORES || tiling.alpha == 0.0f ||
        IsFastpathN(tiling) || tiling.trans != 0U || tiling.incx != 1 || tiling.incy != 1 ||
        tiling.lda != tiling.kl + tiling.ku + 1U || tiling.lda > L1_BAND_MAX) {
        // Contract violation (cannot happen when the host set colSplit=1):
        // fall back to the legacy row-split path, valid for any grid size.
        uint32_t outDim = tiling.m;
        uint32_t startRow = 0U;
        uint32_t endRow = outDim;
        CalcCoreRowRange(outDim, numBlocks, blkIdx, startRow, endRow, CORE_ALIGN_SIZE);
        if (startRow >= endRow) {
            return;
        }
        auto* aGm = reinterpret_cast<__gm__ const float*>(a);
        auto* xGm = reinterpret_cast<__gm__ const float*>(x);
        auto* yGm = reinterpret_cast<__gm__ float*>(y);
        DispatchSgbmvCompute(startRow, endRow, outDim, tiling, a, x, y, aGm, xGm, yGm, false);
        return;
    }
    ProcessSgbmvNColSplit(blkIdx, numBlocks, tiling, a, x, y, workSpace);
}

void sgbmv_kernel_l1_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgbmvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    sgbmv_kernel_l1<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
}
